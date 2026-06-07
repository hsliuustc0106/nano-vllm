import argparse
import signal
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable

import msgpack
import torch
import zmq

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.serve.openai import CompletionRequestError, normalize_completion_request
from nanovllm.serve.text import StreamingTextDecoder, find_stop, next_delta, trim_stop


EventSink = Callable[[dict[str, Any]], None]
DEFAULT_STREAM_TOKEN_FLUSH_INTERVAL = 16
FIXED_DECODE_TOKEN_READBACK_INTERVAL = 64


@dataclass(slots=True)
class ActiveRequest:
    request_id: str
    model: str
    created: int
    prompt_tokens: int
    remaining: int


@dataclass(slots=True)
class ChoiceState:
    seq: Sequence
    request_id: str
    choice_index: int
    prompt_text: str
    prompt_tokens: int
    stop: list[str]
    echo: bool
    stream: bool
    decoder: StreamingTextDecoder
    completion_text: str = ""
    emitted_text: str = ""
    pending_text: str = ""
    pending_tokens: int = 0
    emitted_any: bool = False
    finished: bool = False


@dataclass(slots=True)
class ServingStats:
    steps: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    emitted_events: int = 0
    requests: int = 0
    request_messages: int = 0
    request_normalize_ns: int = 0
    request_add_ns: int = 0
    step_schedule_ns: int = 0
    step_model_ns: int = 0
    step_postprocess_ns: int = 0
    token_handle_ns: int = 0
    event_emit_ns: int = 0


class ServingLLMEngine(LLMEngine):

    def __init__(
        self,
        model,
        event_sink: EventSink,
        stream_token_flush_interval: int = DEFAULT_STREAM_TOKEN_FLUSH_INTERVAL,
        **kwargs,
    ):
        if stream_token_flush_interval < 1:
            raise ValueError("stream_token_flush_interval must be >= 1")
        super().__init__(model, **kwargs)
        self.event_sink = event_sink
        self.stream_token_flush_interval = stream_token_flush_interval
        self.active_requests: dict[str, ActiveRequest] = {}
        self.choices: dict[int, ChoiceState] = {}
        self._fixed_decode_token_gpu_buffer = None
        self._fixed_decode_token_cpu_buffer = None
        self.stats = ServingStats()

    def add_completion_request(self, payload: dict[str, Any]):
        self.add_completion_requests([payload])

    def add_completion_requests(self, payloads: list[dict[str, Any]]):
        normalize_start = perf_counter()
        requests = []
        simple_payloads = []
        simple_indexes = []
        for index, payload in enumerate(payloads):
            prompt = payload.get("prompt")
            echo = payload.get("echo", False)
            if isinstance(prompt, str) and echo is False:
                simple_indexes.append(index)
                simple_payloads.append(payload)
                requests.append(None)
                continue
            requests.append(self._normalize_completion_payload(payload))

        if simple_payloads:
            encoded_prompts = self.tokenizer([payload["prompt"] for payload in simple_payloads]).input_ids
            for index, payload, token_ids in zip(simple_indexes, simple_payloads, encoded_prompts):
                tokenized_payload = dict(payload)
                tokenized_payload["prompt"] = token_ids
                requests[index] = self._normalize_completion_payload(tokenized_payload)

        self.stats.request_normalize_ns += int((perf_counter() - normalize_start) * 1e9)

        for request in requests:
            if request is not None:
                self._add_normalized_completion_request(request)

    def _normalize_completion_payload(self, payload: dict[str, Any]):
        try:
            return normalize_completion_request(payload, self.tokenizer)
        except CompletionRequestError as exc:
            self._emit_error(payload.get("request_id"), str(exc))
            return None

    def _add_normalized_completion_request(self, request):
        add_start = perf_counter()
        self.stats.requests += 1
        num_choices = len(request.prompts) * request.n
        prompt_tokens = sum(len(prompt.token_ids) for prompt in request.prompts) * request.n
        created = int(time.time())
        self.active_requests[request.request_id] = ActiveRequest(
            request_id=request.request_id,
            model=request.model,
            created=created,
            prompt_tokens=prompt_tokens,
            remaining=num_choices,
        )
        self._emit({
            "type": "started",
            "request_id": request.request_id,
            "created": created,
            "model": request.model,
            "num_choices": num_choices,
            "prompt_tokens": prompt_tokens,
        })

        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            ignore_eos=request.ignore_eos,
        )
        choice_index = 0
        for prompt in request.prompts:
            for _ in range(request.n):
                seq = self.add_request(prompt.token_ids, sampling_params)
                seq.request_id = request.request_id
                seq.choice_index = choice_index
                self.choices[seq.seq_id] = ChoiceState(
                    seq=seq,
                    request_id=request.request_id,
                    choice_index=choice_index,
                    prompt_text=prompt.text,
                    prompt_tokens=len(prompt.token_ids),
                    stop=request.stop,
                    echo=request.echo,
                    stream=request.stream,
                    decoder=StreamingTextDecoder(self.tokenizer),
                )
                if request.stream and request.echo:
                    self._emit_token(request.request_id, choice_index, prompt.text, None)
                choice_index += 1
        self.stats.request_add_ns += int((perf_counter() - add_start) * 1e9)

    def cancel_request(self, request_id: str):
        active = self.active_requests.get(request_id)
        if active is None:
            return
        for seq_id, choice in list(self.choices.items()):
            if choice.request_id != request_id:
                continue
            self.scheduler.cancel(seq_id)
            self._finish_choice(choice, "cancelled")

    def step_serving(self):
        schedule_start = perf_counter()
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        schedule_end = perf_counter()
        if self._can_batch_decode_token_readback(seqs, is_prefill):
            self._step_fixed_decode(seqs, schedule_start, schedule_end)
            return
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        model_end = perf_counter()
        appended = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        postprocess_end = perf_counter()
        self.stats.step_schedule_ns += int((schedule_end - schedule_start) * 1e9)
        self.stats.step_model_ns += int((model_end - schedule_end) * 1e9)
        self.stats.step_postprocess_ns += int((postprocess_end - model_end) * 1e9)
        self._record_step(num_tokens)
        for seq, token_id in appended:
            choice = self.choices.get(seq.seq_id)
            if choice is not None:
                handle_start = perf_counter()
                self._handle_token(choice, token_id)
                self.stats.token_handle_ns += int((perf_counter() - handle_start) * 1e9)

    def _can_batch_decode_token_readback(self, seqs: list[Sequence], is_prefill: bool) -> bool:
        if is_prefill or not seqs:
            return False
        if self.scheduler.waiting and len(self.scheduler.running) < self.scheduler.max_num_seqs:
            return False
        if getattr(self.model_runner, "world_size", 1) != 1:
            return False
        for seq in seqs:
            choice = self.choices.get(seq.seq_id)
            if choice is None:
                return False
            if choice.stop or choice.echo:
                return False
            if choice.stream and (self.stream_token_flush_interval <= 1 or not choice.emitted_any):
                return False
            if seq.max_tokens <= seq.num_completion_tokens:
                return False
        return True

    def _fixed_decode_steps(self, seqs: list[Sequence]) -> int:
        steps = min(seq.max_tokens - seq.num_completion_tokens for seq in seqs)
        if any((choice := self.choices.get(seq.seq_id)) is not None and choice.stream for seq in seqs):
            steps = min(steps, self.stream_token_flush_interval)
        else:
            steps = min(steps, FIXED_DECODE_TOKEN_READBACK_INTERVAL)
        return max(0, steps)

    def _step_fixed_decode(self, seqs: list[Sequence], schedule_start: float, schedule_end: float):
        model_start = schedule_end
        next_input_ids = None
        num_token_rows = 0
        steps = self._fixed_decode_steps(seqs)
        for step in range(steps):
            if step > 0 and not self._append_decode_slots(seqs):
                break
            token_tensor = self.model_runner.call("run_decode_tokens", seqs, next_input_ids)
            if token_tensor is None:
                break
            token_buffer = self._fixed_decode_token_buffer(token_tensor, steps, len(seqs))
            token_buffer[step, :len(seqs)].copy_(token_tensor)
            num_token_rows += 1
            next_input_ids = token_tensor
            for seq in seqs:
                seq.num_cached_tokens += seq.num_scheduled_tokens
                seq.num_scheduled_tokens = 0
                seq.append_token(-1)
            self._record_step(-len(seqs))
        model_end = perf_counter()
        if num_token_rows:
            token_rows = self._copy_fixed_decode_tokens_to_cpu(num_token_rows, len(seqs))
            postprocess_start = perf_counter()
            self._postprocess_fixed_decode_tokens(seqs, token_rows)
            postprocess_end = perf_counter()
        else:
            postprocess_start = postprocess_end = model_end
        self.stats.step_schedule_ns += int((schedule_end - schedule_start) * 1e9)
        self.stats.step_model_ns += int((model_end - model_start) * 1e9)
        self.stats.step_postprocess_ns += int((postprocess_end - postprocess_start) * 1e9)

    def _postprocess_fixed_decode_tokens(self, seqs: list[Sequence], token_rows: list[list[int]]):
        completion_starts: dict[int, int] = {}
        for seq_index, seq in enumerate(seqs):
            completion_start = seq.num_tokens - len(token_rows)
            completion_starts[seq.seq_id] = completion_start
            for offset, token_row in enumerate(token_rows):
                seq.token_ids[completion_start + offset] = token_row[seq_index]
            seq.last_token = seq.token_ids[-1]

        for offset, token_row in enumerate(token_rows):
            for seq_index, seq in enumerate(seqs):
                if seq.is_finished:
                    continue
                token_id = token_row[seq_index]
                completion_start = completion_starts[seq.seq_id]
                completion_tokens = completion_start + offset + 1 - seq.num_prompt_tokens
                finish_reason = None
                if not seq.ignore_eos and token_id == self.tokenizer.eos_token_id:
                    finish_reason = "stop"
                elif completion_tokens >= seq.max_tokens:
                    finish_reason = "length"

                if finish_reason is not None:
                    keep_tokens = completion_start + offset + 1
                    if keep_tokens < seq.num_tokens:
                        seq.token_ids = seq.token_ids[:keep_tokens]
                        seq.num_tokens = keep_tokens
                        seq.num_cached_tokens = min(seq.num_cached_tokens, seq.num_tokens)
                        seq.last_token = seq.token_ids[-1]
                    self.scheduler.finish(seq, finish_reason)

                choice = self.choices.get(seq.seq_id)
                if choice is not None:
                    handle_start = perf_counter()
                    self._handle_token(choice, token_id)
                    self.stats.token_handle_ns += int((perf_counter() - handle_start) * 1e9)

    def _fixed_decode_token_buffer(self, token_tensor: torch.Tensor, steps: int, bs: int):
        max_num_seqs = getattr(self.scheduler, "max_num_seqs", bs)
        shape = (max(steps, FIXED_DECODE_TOKEN_READBACK_INTERVAL), max_num_seqs)
        if (
            getattr(self, "_fixed_decode_token_gpu_buffer", None) is None
            or self._fixed_decode_token_gpu_buffer.shape[0] < shape[0]
            or self._fixed_decode_token_gpu_buffer.shape[1] < bs
            or self._fixed_decode_token_gpu_buffer.dtype != token_tensor.dtype
            or self._fixed_decode_token_gpu_buffer.device != token_tensor.device
        ):
            self._fixed_decode_token_gpu_buffer = torch.empty(shape, dtype=token_tensor.dtype, device=token_tensor.device)
            self._fixed_decode_token_cpu_buffer = torch.empty(shape, dtype=token_tensor.dtype, device="cpu", pin_memory=True)
        return self._fixed_decode_token_gpu_buffer

    def _copy_fixed_decode_tokens_to_cpu(self, steps: int, bs: int):
        gpu_tokens = self._fixed_decode_token_gpu_buffer[:steps, :bs]
        cpu_tokens = self._fixed_decode_token_cpu_buffer[:steps, :bs]
        cpu_tokens.copy_(gpu_tokens, non_blocking=True)
        if gpu_tokens.device.type == "cuda":
            torch.cuda.current_stream(gpu_tokens.device).synchronize()
        return cpu_tokens.tolist()

    def _append_decode_slots(self, seqs: list[Sequence]) -> bool:
        for seq in seqs:
            if not self.scheduler.block_manager.can_append(seq):
                return False
        for seq in seqs:
            seq.num_scheduled_tokens = 1
            seq.is_prefill = False
            self.scheduler.block_manager.may_append(seq)
        return True

    def _record_step(self, num_tokens: int):
        self.stats.steps += 1
        if num_tokens > 0:
            self.stats.prefill_steps += 1
            self.stats.prefill_tokens += num_tokens
        else:
            self.stats.decode_steps += 1
            self.stats.decode_tokens += -num_tokens

    def reset_stats(self):
        self.stats = ServingStats()
        if hasattr(self.model_runner, "reset_stats"):
            self.model_runner.reset_stats()

    def _handle_token(self, choice: ChoiceState, token_id: int):
        seq = choice.seq
        finish_reason = seq.finish_reason if seq.is_finished else None
        if not choice.stream and not choice.stop:
            if finish_reason is not None:
                self._finish_choice(choice, finish_reason)
            return
        if not choice.stop:
            self._handle_token_without_stop(choice, token_id, finish_reason)
            return
        completion_text = self._decode_completion(choice, token_id)
        completion_text, stopped = trim_stop(completion_text, choice.stop)
        if stopped:
            if not seq.is_finished:
                self.scheduler.finish(seq, "stop")
            else:
                seq.finish_reason = "stop"
        choice.completion_text = completion_text
        finish_reason = seq.finish_reason if seq.is_finished else None
        finished = finish_reason is not None
        delta = self._next_delta(choice, completion_text, finished)
        if choice.stream and delta:
            self._queue_or_emit_token(choice, delta, token_id, finished)
        elif choice.stream and finished:
            self._flush_pending_token(choice)
        if finished:
            self._finish_choice(choice, finish_reason)

    def _handle_token_without_stop(self, choice: ChoiceState, token_id: int, finish_reason: str | None):
        seq = choice.seq
        finished = finish_reason is not None
        if not (seq.finish_reason == "stop" and token_id == self.tokenizer.eos_token_id):
            choice.decoder.buffer(token_id)
            choice.pending_tokens += 1
        if choice.stream and (not choice.emitted_any or finished or choice.pending_tokens >= self.stream_token_flush_interval):
            self._flush_decoded_delta(choice, finished)
        if finished:
            self._finish_choice(choice, finish_reason)

    def _decode_completion(self, choice: ChoiceState, token_id: int) -> str:
        seq = choice.seq
        if seq.finish_reason == "stop" and token_id == self.tokenizer.eos_token_id:
            return choice.decoder.flush(finished=True)
        return choice.decoder.append(token_id, finished=seq.is_finished)

    def _next_delta(self, choice: ChoiceState, completion_text: str, finished: bool) -> str:
        choice.emitted_text, delta = next_delta(choice.emitted_text, completion_text, choice.stop, finished)
        return delta

    def _queue_or_emit_token(self, choice: ChoiceState, delta: str, token_id: int, finished: bool):
        choice.pending_text += delta
        choice.pending_tokens += 1
        if not choice.emitted_any or finished or choice.pending_tokens >= self.stream_token_flush_interval:
            self._flush_pending_token(choice)

    def _flush_pending_token(self, choice: ChoiceState):
        if not choice.pending_text:
            return
        self._emit_token(choice.request_id, choice.choice_index, choice.pending_text, None)
        choice.pending_text = ""
        choice.pending_tokens = 0
        choice.emitted_any = True

    def _flush_decoded_delta(self, choice: ChoiceState, finished: bool):
        completion_text = choice.decoder.flush(finished=finished)
        choice.completion_text = completion_text
        if len(completion_text) < len(choice.emitted_text):
            choice.emitted_text = completion_text
            return
        delta = completion_text[len(choice.emitted_text):]
        choice.emitted_text = completion_text
        choice.pending_tokens = 0
        if delta:
            self._emit_token(choice.request_id, choice.choice_index, delta, None)
            choice.emitted_any = True

    def _finish_choice(self, choice: ChoiceState, finish_reason: str):
        if choice.finished:
            return
        choice.finished = True
        if choice.stream:
            self._flush_pending_token(choice)
            self._emit_finished(choice, finish_reason)
            self._cleanup_finished_choice(choice)
            return
        seq = choice.seq
        if not choice.stop and not choice.decoder.text and not choice.decoder.pending_token_ids:
            completion_token_ids = seq.completion_token_ids
            if seq.finish_reason == "stop" and completion_token_ids and completion_token_ids[-1] == self.tokenizer.eos_token_id:
                completion_token_ids = completion_token_ids[:-1]
            completion_text = self.tokenizer.decode(completion_token_ids)
        else:
            completion_text = choice.decoder.flush(finished=True)
            if len(completion_text) < len(choice.completion_text):
                completion_text = choice.completion_text
        stop_index = find_stop(completion_text, choice.stop)
        if stop_index is not None:
            completion_text = completion_text[:stop_index]
        text = choice.prompt_text + completion_text if choice.echo else completion_text
        self._emit_finished(choice, finish_reason, text=text, token_ids=seq.completion_token_ids)
        self._cleanup_finished_choice(choice)

    def _emit_finished(
        self,
        choice: ChoiceState,
        finish_reason: str,
        text: str | None = None,
        token_ids: list[int] | None = None,
    ):
        seq = choice.seq
        event = {
            "type": "finished",
            "request_id": choice.request_id,
            "choice_index": choice.choice_index,
            "finish_reason": finish_reason,
            "prompt_tokens": choice.prompt_tokens,
            "completion_tokens": seq.num_completion_tokens,
        }
        if text is not None:
            event["text"] = text
        if token_ids is not None:
            event["token_ids"] = token_ids
        self._emit(event)

    def _cleanup_finished_choice(self, choice: ChoiceState):
        seq = choice.seq
        self.choices.pop(seq.seq_id, None)
        active = self.active_requests.get(choice.request_id)
        if active is not None:
            active.remaining -= 1
            if active.remaining <= 0:
                self.active_requests.pop(choice.request_id, None)

    def _emit_token(self, request_id: str, choice_index: int, text: str, token_id: int | None):
        self._emit({
            "type": "token",
            "request_id": request_id,
            "choice_index": choice_index,
            "text": text,
            "token_id": token_id,
        })

    def _emit_error(self, request_id: str | None, message: str):
        self._emit({
            "type": "error",
            "request_id": request_id,
            "message": message,
        })

    def _emit(self, event: dict[str, Any]):
        emit_start = perf_counter()
        stats = getattr(self, "stats", None)
        if isinstance(stats, ServingStats):
            stats.emitted_events += 1
        self.event_sink(event)
        if isinstance(stats, ServingStats):
            stats.event_emit_ns += int((perf_counter() - emit_start) * 1e9)


class ZmqEngineServer:

    def __init__(self, engine: ServingLLMEngine, request_endpoint: str, event_endpoint: str, stats_interval: float = 0.0):
        self.engine = engine
        self.context = zmq.Context.instance()
        self.request_socket = self.context.socket(zmq.PULL)
        self.request_socket.bind(request_endpoint)
        self.poller = zmq.Poller()
        self.poller.register(self.request_socket, zmq.POLLIN)
        self.shutdown = False
        self.stats_interval = stats_interval
        self.last_stats_log = perf_counter()

    def run_forever(self):
        while not self.shutdown:
            timeout_ms = 0 if not self.engine.is_finished() else 100
            events = dict(self.poller.poll(timeout_ms))
            if self.request_socket in events:
                self._drain_requests()
            if not self.engine.is_finished():
                self.engine.step_serving()
            self._maybe_log_stats()

    def _maybe_log_stats(self):
        if self.stats_interval <= 0:
            return
        now = perf_counter()
        if now - self.last_stats_log < self.stats_interval:
            return
        self.last_stats_log = now
        stats = self.engine.stats
        runner_stats = getattr(self.engine.model_runner, "stats", None)
        prefill_batch = stats.prefill_tokens / stats.prefill_steps if stats.prefill_steps else 0.0
        decode_batch = stats.decode_tokens / stats.decode_steps if stats.decode_steps else 0.0
        runner_text = ""
        if runner_stats is not None:
            runner_text = (
                f" runner_prepare_prefill_ms={runner_stats.prepare_prefill_ns / 1e6:.2f}"
                f" runner_prepare_decode_ms={runner_stats.prepare_decode_ns / 1e6:.2f}"
                f" runner_prepare_sample_ms={runner_stats.prepare_sample_ns / 1e6:.2f}"
                f" runner_model_prefill_ms={runner_stats.run_model_prefill_ns / 1e6:.2f}"
                f" runner_model_decode_ms={runner_stats.run_model_decode_ns / 1e6:.2f}"
                f" runner_sampler_ms={runner_stats.sampler_ns / 1e6:.2f}"
                f" runner_tolist_ms={runner_stats.tolist_ns / 1e6:.2f}"
                f" runner_prefill_calls={runner_stats.prefill_calls}"
                f" runner_decode_calls={runner_stats.decode_calls}"
                f" runner_decode_graph_calls={runner_stats.decode_graph_calls}"
                f" runner_decode_eager_calls={runner_stats.decode_eager_calls}"
            )
        print(
            "serving_stats "
            f"requests={stats.requests} request_messages={stats.request_messages} "
            f"steps={stats.steps} prefill_steps={stats.prefill_steps} decode_steps={stats.decode_steps} "
            f"prefill_tokens={stats.prefill_tokens} decode_tokens={stats.decode_tokens} "
            f"avg_prefill_batch={prefill_batch:.2f} avg_decode_batch={decode_batch:.2f} "
            f"events={stats.emitted_events} "
            f"request_normalize_ms={stats.request_normalize_ns / 1e6:.2f} "
            f"request_add_ms={stats.request_add_ns / 1e6:.2f} "
            f"schedule_ms={stats.step_schedule_ns / 1e6:.2f} "
            f"model_ms={stats.step_model_ns / 1e6:.2f} "
            f"postprocess_ms={stats.step_postprocess_ns / 1e6:.2f} "
            f"token_handle_ms={stats.token_handle_ns / 1e6:.2f} "
            f"event_emit_ms={stats.event_emit_ns / 1e6:.2f}"
            f"{runner_text}",
            flush=True,
        )

    def stop(self):
        self.shutdown = True

    def close(self):
        self.poller.unregister(self.request_socket)
        self.request_socket.close(linger=0)

    def _drain_requests(self):
        completion_messages = []
        while True:
            try:
                data = self.request_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                if completion_messages:
                    self.engine.add_completion_requests(completion_messages)
                return
            message = msgpack.unpackb(data, raw=False)
            if message.get("type") == "completion":
                if isinstance(getattr(self.engine, "stats", None), ServingStats):
                    self.engine.stats.request_messages += 1
                completion_messages.append(message)
                continue
            if completion_messages:
                self.engine.add_completion_requests(completion_messages)
                completion_messages = []
            self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]):
        message_type = message.get("type")
        if isinstance(getattr(self.engine, "stats", None), ServingStats):
            self.engine.stats.request_messages += 1
        if message_type == "completion":
            self.engine.add_completion_request(message)
        elif message_type == "cancel":
            request_id = message.get("request_id")
            if isinstance(request_id, str):
                self.engine.cancel_request(request_id)
        elif message_type == "profile_start":
            self.engine.reset_stats()
            torch.cuda.cudart().cudaProfilerStart()
        elif message_type == "profile_stop":
            torch.cuda.cudart().cudaProfilerStop()
        elif message_type == "reset_stats":
            self.engine.reset_stats()
        elif message_type == "shutdown":
            self.shutdown = True
        else:
            self.engine._emit_error(message.get("request_id"), f"unknown message type: {message_type}")


def _pack_event_sink(socket):
    def emit(event: dict[str, Any]):
        socket.send(msgpack.packb(event, use_bin_type=True))
    return emit


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Nano-vLLM Python serving engine.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--request-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--event-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--stream-token-flush-interval", type=int, default=DEFAULT_STREAM_TOKEN_FLUSH_INTERVAL)
    parser.add_argument("--log-serving-stats-interval", type=float, default=0.0)
    return parser


def main(argv: list[str] | None = None):
    args = build_arg_parser().parse_args(argv)
    context = zmq.Context.instance()
    event_socket = context.socket(zmq.PUSH)
    event_socket.connect(args.event_endpoint)
    engine = ServingLLMEngine(
        args.model,
        _pack_event_sink(event_socket),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_runner_stats=args.log_serving_stats_interval > 0,
        stream_token_flush_interval=args.stream_token_flush_interval,
    )
    server = ZmqEngineServer(
        engine,
        args.request_endpoint,
        args.event_endpoint,
        stats_interval=args.log_serving_stats_interval,
    )

    def handle_signal(signum, frame):
        server.stop()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    try:
        server.run_forever()
    finally:
        server.close()
        event_socket.close(linger=0)


if __name__ == "__main__":
    main()
