import argparse
import signal
import time
from dataclasses import dataclass
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
STREAM_TOKEN_FLUSH_INTERVAL = 4


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


class ServingLLMEngine(LLMEngine):

    def __init__(self, model, event_sink: EventSink, **kwargs):
        super().__init__(model, **kwargs)
        self.event_sink = event_sink
        self.active_requests: dict[str, ActiveRequest] = {}
        self.choices: dict[int, ChoiceState] = {}

    def add_completion_request(self, payload: dict[str, Any]):
        try:
            request = normalize_completion_request(payload, self.tokenizer)
        except CompletionRequestError as exc:
            self._emit_error(payload.get("request_id"), str(exc))
            return

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
        appended, _ = self.step_with_token_events()
        for seq, token_id in appended:
            choice = self.choices.get(seq.seq_id)
            if choice is not None:
                self._handle_token(choice, token_id)

    def _handle_token(self, choice: ChoiceState, token_id: int):
        seq = choice.seq
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

    def _decode_completion(self, choice: ChoiceState, token_id: int) -> str:
        seq = choice.seq
        if seq.finish_reason == "stop" and token_id == self.tokenizer.eos_token_id:
            return choice.decoder.flush(finished=True)
        return choice.decoder.append(token_id, finished=seq.is_finished)

    def _next_delta(self, choice: ChoiceState, completion_text: str, finished: bool) -> str:
        choice.emitted_text, delta = next_delta(choice.emitted_text, completion_text, choice.stop, finished)
        return delta

    def _queue_or_emit_token(self, choice: ChoiceState, delta: str, token_id: int, finished: bool):
        if choice.stop:
            self._emit_token(choice.request_id, choice.choice_index, delta, token_id)
            choice.emitted_any = True
            return
        choice.pending_text += delta
        choice.pending_tokens += 1
        if not choice.emitted_any or finished or choice.pending_tokens >= STREAM_TOKEN_FLUSH_INTERVAL:
            self._flush_pending_token(choice)

    def _flush_pending_token(self, choice: ChoiceState):
        if not choice.pending_text:
            return
        self._emit_token(choice.request_id, choice.choice_index, choice.pending_text, None)
        choice.pending_text = ""
        choice.pending_tokens = 0
        choice.emitted_any = True

    def _finish_choice(self, choice: ChoiceState, finish_reason: str):
        if choice.finished:
            return
        choice.finished = True
        if choice.stream:
            self._flush_pending_token(choice)
        seq = choice.seq
        completion_text = choice.decoder.flush(finished=True)
        if len(completion_text) < len(choice.completion_text):
            completion_text = choice.completion_text
        stop_index = find_stop(completion_text, choice.stop)
        if stop_index is not None:
            completion_text = completion_text[:stop_index]
        text = choice.prompt_text + completion_text if choice.echo else completion_text
        self._emit({
            "type": "finished",
            "request_id": choice.request_id,
            "choice_index": choice.choice_index,
            "text": text,
            "token_ids": seq.completion_token_ids,
            "finish_reason": finish_reason,
            "prompt_tokens": choice.prompt_tokens,
            "completion_tokens": seq.num_completion_tokens,
        })
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
        self.event_sink(event)


class ZmqEngineServer:

    def __init__(self, engine: ServingLLMEngine, request_endpoint: str, event_endpoint: str):
        self.engine = engine
        self.context = zmq.Context.instance()
        self.request_socket = self.context.socket(zmq.PULL)
        self.request_socket.bind(request_endpoint)
        self.poller = zmq.Poller()
        self.poller.register(self.request_socket, zmq.POLLIN)
        self.shutdown = False

    def run_forever(self):
        while not self.shutdown:
            timeout_ms = 0 if not self.engine.is_finished() else 100
            events = dict(self.poller.poll(timeout_ms))
            if self.request_socket in events:
                self._drain_requests()
            if not self.engine.is_finished():
                self.engine.step_serving()

    def stop(self):
        self.shutdown = True

    def close(self):
        self.poller.unregister(self.request_socket)
        self.request_socket.close(linger=0)

    def _drain_requests(self):
        while True:
            try:
                data = self.request_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            message = msgpack.unpackb(data, raw=False)
            self._handle_message(message)

    def _handle_message(self, message: dict[str, Any]):
        message_type = message.get("type")
        if message_type == "completion":
            self.engine.add_completion_request(message)
        elif message_type == "cancel":
            request_id = message.get("request_id")
            if isinstance(request_id, str):
                self.engine.cancel_request(request_id)
        elif message_type == "profile_start":
            torch.cuda.cudart().cudaProfilerStart()
        elif message_type == "profile_stop":
            torch.cuda.cudart().cudaProfilerStop()
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
    )
    server = ZmqEngineServer(engine, args.request_endpoint, args.event_endpoint)

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
