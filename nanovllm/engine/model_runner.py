import pickle
import torch
import torch.distributed as dist
from dataclasses import dataclass
from time import perf_counter
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

try:
    import flashinfer
except ImportError:  # pragma: no cover - optional decode acceleration
    flashinfer = None


@dataclass(slots=True)
class ModelRunnerStats:
    prepare_prefill_ns: int = 0
    prepare_decode_ns: int = 0
    prepare_sample_ns: int = 0
    run_model_prefill_ns: int = 0
    run_model_decode_ns: int = 0
    sampler_ns: int = 0
    tolist_ns: int = 0
    prefill_calls: int = 0
    decode_calls: int = 0
    decode_graph_calls: int = 0
    decode_eager_calls: int = 0


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.enable_runner_stats = config.enable_runner_stats
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.graph_includes_logits = self.world_size == 1
        self.graph_includes_sampler = self.world_size == 1
        self.use_flashinfer_decode = flashinfer is not None and self.world_size == 1 and not self.enforce_eager
        self.stats = ModelRunnerStats()

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.default_temperatures = torch.ones(config.max_num_seqs, dtype=torch.float32)
        self.allocate_decode_workspaces()
        self.allocate_flashinfer_workspaces()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.graph_bs = self.build_graph_batch_sizes(config.max_num_seqs)
        self.warmup_sampler()
        if not self.enforce_eager:
            self.capture_cudagraph()
        self.reset_stats()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def reset_stats(self):
        self.stats = ModelRunnerStats()

    def allocate_decode_workspaces(self):
        max_bs = self.config.max_num_seqs
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        self.decode_input_ids_cpu = torch.empty(max_bs, dtype=torch.int64, device="cpu", pin_memory=True)
        self.decode_positions_cpu = torch.empty(max_bs, dtype=torch.int64, device="cpu", pin_memory=True)
        self.decode_slot_mapping_cpu = torch.empty(max_bs, dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_context_lens_cpu = torch.empty(max_bs, dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_block_tables_cpu = torch.empty((max_bs, max_num_blocks), dtype=torch.int32, device="cpu", pin_memory=True)
        self.decode_input_ids_cpu_np = self.decode_input_ids_cpu.numpy()
        self.decode_positions_cpu_np = self.decode_positions_cpu.numpy()
        self.decode_slot_mapping_cpu_np = self.decode_slot_mapping_cpu.numpy()
        self.decode_context_lens_cpu_np = self.decode_context_lens_cpu.numpy()
        self.decode_block_tables_cpu_np = self.decode_block_tables_cpu.numpy()
        self.decode_input_ids_gpu = torch.empty(max_bs, dtype=torch.int64)
        self.decode_positions_gpu = torch.empty(max_bs, dtype=torch.int64)
        self.decode_slot_mapping_gpu = torch.empty(max_bs, dtype=torch.int32)
        self.decode_context_lens_gpu = torch.empty(max_bs, dtype=torch.int32)
        self.decode_block_tables_gpu = torch.empty((max_bs, max_num_blocks), dtype=torch.int32)

    def allocate_flashinfer_workspaces(self):
        if not self.use_flashinfer_decode:
            return
        max_bs = self.config.max_num_seqs
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        self.flashinfer_indptr_cpu = torch.empty(max_bs + 1, dtype=torch.int32, device="cpu", pin_memory=True)
        self.flashinfer_indices_cpu = torch.empty(max_bs * max_num_blocks, dtype=torch.int32, device="cpu", pin_memory=True)
        self.flashinfer_last_page_len_cpu = torch.empty(max_bs, dtype=torch.int32, device="cpu", pin_memory=True)
        self.flashinfer_indptr_cpu_np = self.flashinfer_indptr_cpu.numpy()
        self.flashinfer_indices_cpu_np = self.flashinfer_indices_cpu.numpy()
        self.flashinfer_last_page_len_cpu_np = self.flashinfer_last_page_len_cpu.numpy()
        self.flashinfer_indptr_gpu = torch.empty(max_bs + 1, dtype=torch.int32)
        self.flashinfer_indices_gpu = torch.empty(max_bs * max_num_blocks, dtype=torch.int32)
        self.flashinfer_last_page_len_gpu = torch.empty(max_bs, dtype=torch.int32)
        self.flashinfer_workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8)
        self.flashinfer_decode_wrapper = flashinfer.CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.flashinfer_workspace,
            self.flashinfer_indptr_gpu,
            self.flashinfer_indices_gpu,
            self.flashinfer_last_page_len_gpu,
            kv_layout="NHD",
        )

    @staticmethod
    def build_graph_batch_sizes(max_bs: int) -> list[int]:
        graph_bs = [bs for bs in [1, 2, 4, 8] if bs <= max_bs]
        graph_bs.extend(range(16, max_bs + 1, 16))
        if graph_bs[-1] < max_bs:
            graph_bs.append(max_bs)
        return graph_bs

    @staticmethod
    def build_graph_batch_lookup(graph_bs: list[int], max_bs: int) -> list[int]:
        graph_bs_for_actual_bs = [0] * (max_bs + 1)
        graph_index = 0
        for bs in range(1, max_bs + 1):
            while graph_bs[graph_index] < bs:
                graph_index += 1
            graph_bs_for_actual_bs[bs] = graph_bs[graph_index]
        return graph_bs_for_actual_bs

    def warmup_sampler(self):
        if self.rank != 0:
            return
        hf_config = self.config.hf_config
        batch_sizes = getattr(self, "graph_bs", [1, self.config.max_num_seqs])
        for bs in batch_sizes:
            logits = torch.zeros(bs, hf_config.vocab_size, dtype=hf_config.dtype)
            self.sampler(logits, self.default_temperatures[:bs])
        torch.cuda.synchronize()

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        warmup_shapes = []
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        warmup_shapes.append((num_seqs, seq_len))
        num_seqs = min(max_num_batched_tokens, self.config.max_num_seqs)
        seq_len = min(max_model_len, max_num_batched_tokens // num_seqs)
        warmup_shapes.append((num_seqs, seq_len))
        seen_shapes = set()
        for num_seqs, seq_len in warmup_shapes:
            if (num_seqs, seq_len) in seen_shapes:
                continue
            seen_shapes.add((num_seqs, seq_len))
            seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
            for seq in seqs:
                seq.num_scheduled_tokens = seq_len
            self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def _uses_decode_graph(self, bs: int):
        return not self.enforce_eager and hasattr(self, "graph_bs") and bs <= self.graph_bs[-1]

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def prepare_decode(self, seqs: list[Sequence], input_ids: torch.Tensor | None = None):
        bs = len(seqs)
        max_len = max(len(seq.block_table) for seq in seqs)
        if input_ids is None:
            self.decode_input_ids_cpu_np[:bs] = [seq.last_token for seq in seqs]
        self.decode_positions_cpu_np[:bs] = [len(seq) - 1 for seq in seqs]
        self.decode_context_lens_cpu_np[:bs] = [len(seq) for seq in seqs]
        self.decode_slot_mapping_cpu_np[:bs] = [
            seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1 for seq in seqs
        ]
        self.decode_block_tables_cpu_np[:bs, :max_len] = -1
        for i, seq in enumerate(seqs):
            block_table = seq.block_table
            self.decode_block_tables_cpu_np[i, :len(block_table)] = block_table

        if self._uses_decode_graph(bs):
            graph_bs = self.graph_bs_for_actual_bs[bs]
            graph_vars = self.graph_vars
            if input_ids is None:
                graph_vars["input_ids"][:bs].copy_(self.decode_input_ids_cpu[:bs], non_blocking=True)
            else:
                graph_vars["input_ids"][:bs].copy_(input_ids.to(dtype=torch.int64), non_blocking=True)
            graph_vars["positions"][:bs].copy_(self.decode_positions_cpu[:bs], non_blocking=True)
            graph_vars["slot_mapping"][:graph_bs].fill_(-1)
            graph_vars["slot_mapping"][:bs].copy_(self.decode_slot_mapping_cpu[:bs], non_blocking=True)
            graph_vars["context_lens"][:graph_bs].zero_()
            graph_vars["context_lens"][:bs].copy_(self.decode_context_lens_cpu[:bs], non_blocking=True)
            graph_vars["block_tables"][:bs, :max_len].copy_(self.decode_block_tables_cpu[:bs, :max_len], non_blocking=True)
            flashinfer_decode_wrapper = None
            if self._uses_flashinfer_decode_graph(graph_bs):
                self._plan_flashinfer_decode(seqs, graph_bs)
                flashinfer_decode_wrapper = self.flashinfer_decode_wrapper
            set_context(
                False,
                slot_mapping=graph_vars["slot_mapping"][:bs],
                context_lens=graph_vars["context_lens"][:bs],
                block_tables=graph_vars["block_tables"][:bs, :max_len],
                graph_resident=True,
                flashinfer_decode_wrapper=flashinfer_decode_wrapper,
            )
            return graph_vars["input_ids"][:bs], graph_vars["positions"][:bs]

        if input_ids is None:
            self.decode_input_ids_gpu[:bs].copy_(self.decode_input_ids_cpu[:bs], non_blocking=True)
            input_ids = self.decode_input_ids_gpu[:bs]
        else:
            input_ids = input_ids.to(device="cuda", dtype=torch.int64, non_blocking=True)
        self.decode_positions_gpu[:bs].copy_(self.decode_positions_cpu[:bs], non_blocking=True)
        self.decode_slot_mapping_gpu[:bs].copy_(self.decode_slot_mapping_cpu[:bs], non_blocking=True)
        self.decode_context_lens_gpu[:bs].copy_(self.decode_context_lens_cpu[:bs], non_blocking=True)
        self.decode_block_tables_gpu[:bs, :max_len].copy_(self.decode_block_tables_cpu[:bs, :max_len], non_blocking=True)
        positions = self.decode_positions_gpu[:bs]
        slot_mapping = self.decode_slot_mapping_gpu[:bs]
        context_lens = self.decode_context_lens_gpu[:bs]
        block_tables = self.decode_block_tables_gpu[:bs, :max_len]
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def _plan_flashinfer_decode(self, seqs: list[Sequence] | None, graph_bs: int):
        cursor = 0
        self.flashinfer_indptr_cpu_np[0] = 0
        for i in range(graph_bs):
            if seqs is not None and i < len(seqs):
                block_table = seqs[i].block_table
                num_blocks = len(block_table)
                self.flashinfer_indices_cpu_np[cursor:cursor + num_blocks] = block_table
                cursor += num_blocks
                self.flashinfer_last_page_len_cpu_np[i] = seqs[i].last_block_num_tokens
            else:
                self.flashinfer_last_page_len_cpu_np[i] = 0
            self.flashinfer_indptr_cpu_np[i + 1] = cursor

        self.flashinfer_indptr_gpu[:graph_bs + 1].copy_(self.flashinfer_indptr_cpu[:graph_bs + 1], non_blocking=True)
        if cursor:
            self.flashinfer_indices_gpu[:cursor].copy_(self.flashinfer_indices_cpu[:cursor], non_blocking=True)
        self.flashinfer_last_page_len_gpu[:graph_bs].copy_(self.flashinfer_last_page_len_cpu[:graph_bs], non_blocking=True)

        hf_config = self.config.hf_config
        num_q_heads = hf_config.num_attention_heads // self.world_size
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        self.flashinfer_decode_wrapper.plan(
            self.flashinfer_indptr_gpu[:graph_bs + 1],
            self.flashinfer_indices_gpu[:max(cursor, 1)],
            self.flashinfer_last_page_len_gpu[:graph_bs],
            num_q_heads,
            num_kv_heads,
            head_dim,
            self.block_size,
            q_data_type=hf_config.dtype,
            kv_data_type=hf_config.dtype,
            sm_scale=head_dim ** -0.5,
            disable_split_kv=True,
        )

    def prepare_sample(self, seqs: list[Sequence]):
        if all(seq.temperature == 1.0 for seq in seqs):
            return self.default_temperatures[:len(seqs)]
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    def _decode_graph_outputs_tokens(self, bs: int):
        return self.graph_includes_sampler and self._uses_decode_graph(bs)

    def _uses_flashinfer_decode_graph(self, graph_bs: int):
        return self.use_flashinfer_decode and graph_bs == self.config.max_num_seqs

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool, temperatures: torch.Tensor | None = None):
        if is_prefill or self.enforce_eager or input_ids.size(0) > self.graph_bs[-1]:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[self.graph_bs_for_actual_bs[bs]]
            graph_vars = self.graph_vars
            if not context.graph_resident:
                graph_vars["input_ids"][:bs] = input_ids
                graph_vars["positions"][:bs] = positions
                graph_vars["slot_mapping"].fill_(-1)
                graph_vars["slot_mapping"][:bs] = context.slot_mapping
                graph_vars["context_lens"].zero_()
                graph_vars["context_lens"][:bs] = context.context_lens
                graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            if self.graph_includes_sampler:
                graph_vars["temperatures"][:bs].copy_(temperatures, non_blocking=True)
            graph.replay()
            if self.graph_includes_sampler:
                return graph_vars["outputs"][:bs]
            if self.graph_includes_logits:
                return graph_vars["outputs"][:bs]
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if not self.enable_runner_stats:
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            outputs = self.run_model(input_ids, positions, is_prefill, temperatures)
            if self.rank == 0 and not is_prefill and self._decode_graph_outputs_tokens(input_ids.size(0)):
                token_ids = outputs.tolist()
            else:
                token_ids = self.sampler(outputs, temperatures).tolist() if self.rank == 0 else None
            reset_context()
            return token_ids

        prepare_start = perf_counter()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        prepare_end = perf_counter()
        sample_prepare_start = prepare_end
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        sample_prepare_end = perf_counter()
        model_start = sample_prepare_end
        outputs = self.run_model(input_ids, positions, is_prefill, temperatures)
        model_end = perf_counter()
        if self.rank == 0:
            sampler_start = model_end
            if not is_prefill and self._decode_graph_outputs_tokens(input_ids.size(0)):
                token_tensor = outputs
            else:
                token_tensor = self.sampler(outputs, temperatures)
            sampler_end = perf_counter()
            token_ids = token_tensor.tolist()
            tolist_end = perf_counter()
        else:
            sampler_start = sampler_end = tolist_end = model_end
            token_ids = None
        if is_prefill:
            self.stats.prepare_prefill_ns += int((prepare_end - prepare_start) * 1e9)
            self.stats.run_model_prefill_ns += int((model_end - model_start) * 1e9)
            self.stats.prefill_calls += 1
        else:
            self.stats.prepare_decode_ns += int((prepare_end - prepare_start) * 1e9)
            self.stats.run_model_decode_ns += int((model_end - model_start) * 1e9)
            self.stats.decode_calls += 1
            if self.enforce_eager or input_ids.size(0) > self.graph_bs[-1]:
                self.stats.decode_eager_calls += 1
            else:
                self.stats.decode_graph_calls += 1
        self.stats.prepare_sample_ns += int((sample_prepare_end - sample_prepare_start) * 1e9)
        self.stats.sampler_ns += int((sampler_end - sampler_start) * 1e9) if self.rank == 0 else 0
        self.stats.tolist_ns += int((tolist_end - sampler_end) * 1e9) if self.rank == 0 else 0
        reset_context()
        return token_ids

    def run_decode_tokens(self, seqs: list[Sequence], input_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        if not self.enable_runner_stats:
            input_ids, positions = self.prepare_decode(seqs, input_ids)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            outputs = self.run_model(input_ids, positions, False, temperatures)
            if self.rank == 0 and self._decode_graph_outputs_tokens(input_ids.size(0)):
                token_ids = outputs
            else:
                token_ids = self.sampler(outputs, temperatures) if self.rank == 0 else None
            reset_context()
            return token_ids

        prepare_start = perf_counter()
        input_ids, positions = self.prepare_decode(seqs, input_ids)
        prepare_end = perf_counter()
        sample_prepare_start = prepare_end
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        sample_prepare_end = perf_counter()
        model_start = sample_prepare_end
        outputs = self.run_model(input_ids, positions, False, temperatures)
        model_end = perf_counter()
        if self.rank == 0:
            sampler_start = model_end
            if self._decode_graph_outputs_tokens(input_ids.size(0)):
                token_ids = outputs
            else:
                token_ids = self.sampler(outputs, temperatures)
            sampler_end = perf_counter()
        else:
            sampler_start = sampler_end = model_end
            token_ids = None
        self.stats.prepare_decode_ns += int((prepare_end - prepare_start) * 1e9)
        self.stats.run_model_decode_ns += int((model_end - model_start) * 1e9)
        self.stats.prepare_sample_ns += int((sample_prepare_end - sample_prepare_start) * 1e9)
        self.stats.sampler_ns += int((sampler_end - sampler_start) * 1e9) if self.rank == 0 else 0
        self.stats.decode_calls += 1
        if self.enforce_eager or input_ids.size(0) > self.graph_bs[-1]:
            self.stats.decode_eager_calls += 1
        else:
            self.stats.decode_graph_calls += 1
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = self.config.max_num_seqs
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        if not hasattr(self, "graph_bs"):
            self.graph_bs = self.build_graph_batch_sizes(max_bs)
        if self.graph_includes_sampler:
            outputs = torch.zeros(max_bs, dtype=torch.int64)
            temperatures = torch.ones(max_bs, dtype=torch.float32)
        else:
            output_size = hf_config.vocab_size if self.graph_includes_logits else hf_config.hidden_size
            outputs = torch.zeros(max_bs, output_size)
            temperatures = None
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            flashinfer_decode_wrapper = None
            if self._uses_flashinfer_decode_graph(bs):
                self._plan_flashinfer_decode(None, bs)
                flashinfer_decode_wrapper = self.flashinfer_decode_wrapper
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
                flashinfer_decode_wrapper=flashinfer_decode_wrapper,
            )
            hidden_states = self.model(input_ids[:bs], positions[:bs])    # warmup
            if self.graph_includes_sampler:
                outputs[:bs] = self.sampler(self.model.compute_logits(hidden_states), temperatures[:bs])
            else:
                outputs[:bs] = self.model.compute_logits(hidden_states) if self.graph_includes_logits else hidden_states
            with torch.cuda.graph(graph, self.graph_pool):
                hidden_states = self.model(input_ids[:bs], positions[:bs])
                if self.graph_includes_sampler:
                    outputs[:bs] = self.sampler(self.model.compute_logits(hidden_states), temperatures[:bs])
                else:
                    outputs[:bs] = self.model.compute_logits(hidden_states) if self.graph_includes_logits else hidden_states
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        if temperatures is not None:
            self.graph_vars["temperatures"] = temperatures
        self.graph_bs_for_actual_bs = self.build_graph_batch_lookup(self.graph_bs, max_bs)
