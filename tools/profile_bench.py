import argparse
import os
import time
from collections import Counter, defaultdict
from random import randint, seed

import torch

from nanovllm import LLM, SamplingParams
from nanovllm.engine.model_runner import ModelRunner


def install_phase_profiler():
    stats = defaultdict(float)
    counts = Counter()
    bs_hist = Counter()
    token_counts = Counter()

    orig_prepare_prefill = ModelRunner.prepare_prefill
    orig_prepare_decode = ModelRunner.prepare_decode
    orig_prepare_sample = ModelRunner.prepare_sample
    orig_run_model = ModelRunner.run_model
    orig_run = ModelRunner.run

    def sync():
        torch.cuda.synchronize()

    def timed_prepare_prefill(self, seqs):
        t = time.perf_counter()
        out = orig_prepare_prefill(self, seqs)
        sync()
        stats["prepare_prefill_s"] += time.perf_counter() - t
        counts["prepare_prefill_calls"] += 1
        return out

    def timed_prepare_decode(self, seqs):
        t = time.perf_counter()
        out = orig_prepare_decode(self, seqs)
        sync()
        stats["prepare_decode_s"] += time.perf_counter() - t
        counts["prepare_decode_calls"] += 1
        return out

    def timed_prepare_sample(self, seqs):
        t = time.perf_counter()
        out = orig_prepare_sample(self, seqs)
        sync()
        stats["prepare_sample_s"] += time.perf_counter() - t
        counts["prepare_sample_calls"] += 1
        return out

    def timed_run_model(self, input_ids, positions, is_prefill):
        t = time.perf_counter()
        out = orig_run_model(self, input_ids, positions, is_prefill)
        sync()
        key = "run_model_prefill_s" if is_prefill else "run_model_decode_s"
        stats[key] += time.perf_counter() - t
        counts[f"{key}_calls"] += 1
        return out

    def timed_run(self, seqs, is_prefill):
        kind = "prefill" if is_prefill else "decode"
        bs_hist[(kind, len(seqs))] += 1
        token_counts[kind] += sum(s.num_scheduled_tokens for s in seqs) if is_prefill else len(seqs)

        t = time.perf_counter()
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)

        t_sample = time.perf_counter()
        token_tensor = self.sampler(logits, temperatures) if self.rank == 0 else None
        sync()
        stats["sampler_s"] += time.perf_counter() - t_sample

        t_tolist = time.perf_counter()
        token_ids = token_tensor.tolist() if self.rank == 0 else None
        stats["tolist_s"] += time.perf_counter() - t_tolist

        from nanovllm.utils.context import reset_context

        reset_context()
        stats["run_total_s"] += time.perf_counter() - t
        counts["run_calls"] += 1
        return token_ids

    ModelRunner.prepare_prefill = timed_prepare_prefill
    ModelRunner.prepare_decode = timed_prepare_decode
    ModelRunner.prepare_sample = timed_prepare_sample
    ModelRunner.run_model = timed_run_model
    ModelRunner.run = timed_run
    return stats, counts, bs_hist, token_counts


def run_bench(args):
    if args.profile:
        stats, counts, bs_hist, token_counts = install_phase_profiler()
    else:
        stats = counts = bs_hist = token_counts = None

    seed(args.seed)
    path = os.path.expanduser(args.model)
    llm = LLM(
        path,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    prompt_token_ids = [
        [randint(0, 10000) for _ in range(randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=randint(args.min_output_len, args.max_output_len))
        for _ in range(args.num_seqs)
    ]

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    if args.profile:
        stats.clear()
        counts.clear()
        bs_hist.clear()
        token_counts.clear()

    t = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - t
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    print(f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, Throughput: {total_tokens / elapsed:.2f}tok/s")

    if args.profile:
        print(f"profile_wall_s {elapsed:.6f}")
        print("profile_counts", dict(counts))
        print("profile_tokens", dict(token_counts))
        for key in sorted(stats):
            print(f"profile_stat {key} {stats[key]:.6f}s {stats[key] / elapsed * 100:.1f}%")
        for (kind, bs), count in bs_hist.most_common(args.hist_top):
            print(f"profile_batch {kind} bs={bs} count={count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/huggingface/Qwen3-0.6B/")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--hist-top", type=int, default=20)
    args = parser.parse_args()
    run_bench(args)


if __name__ == "__main__":
    main()
