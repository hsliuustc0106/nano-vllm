import unittest
from unittest.mock import Mock, patch

import torch

from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.serve.engine import ChoiceState, ServingLLMEngine, ServingStats, ZmqEngineServer
from nanovllm.serve.text import StreamingTextDecoder


class FakeTokenizer:

    eos_token_id = -1

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids):
        self.decode_calls.append(list(token_ids))
        return "".join(chr(token_id) for token_id in token_ids)

    def encode(self, prompt):
        return [ord(char) for char in prompt]

    def __call__(self, prompts):
        self.batch_prompts = list(prompts)
        return Mock(input_ids=[[ord(char) for char in prompt] for prompt in prompts])


class ServingEngineTests(unittest.TestCase):

    def test_profile_control_messages_call_cuda_profiler(self):
        server = ZmqEngineServer.__new__(ZmqEngineServer)
        server.engine = Mock()
        cudart = Mock()
        with patch("nanovllm.serve.engine.torch.cuda.cudart", return_value=cudart):
            server._handle_message({"type": "profile_start"})
            server._handle_message({"type": "profile_stop"})

        server.engine.reset_stats.assert_called_once_with()
        cudart.cudaProfilerStart.assert_called_once_with()
        cudart.cudaProfilerStop.assert_called_once_with()

    def test_reset_stats_control_message_resets_engine_stats(self):
        server = ZmqEngineServer.__new__(ZmqEngineServer)
        server.engine = Mock()

        server._handle_message({"type": "reset_stats"})

        server.engine.reset_stats.assert_called_once_with()

    def test_streaming_tokens_flush_first_and_then_coalesce(self):
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.stream_token_flush_interval = 16
        choice = ChoiceState(
            seq=Mock(),
            request_id="req",
            choice_index=0,
            prompt_text="",
            prompt_tokens=1,
            stop=[],
            echo=False,
            stream=True,
            decoder=Mock(),
        )

        engine._queue_or_emit_token(choice, "a", 1, finished=False)
        for token_id in range(2, 18):
            engine._queue_or_emit_token(choice, chr(96 + token_id), token_id, finished=False)

        self.assertEqual([event["text"] for event in events], ["a", "bcdefghijklmnopq"])
        self.assertEqual(choice.pending_text, "")

    def test_stop_strings_still_coalesce_after_safe_delta_extraction(self):
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.stream_token_flush_interval = 16
        choice = ChoiceState(
            seq=Mock(),
            request_id="req",
            choice_index=0,
            prompt_text="",
            prompt_tokens=1,
            stop=["END"],
            echo=False,
            stream=True,
            decoder=Mock(),
        )

        engine._queue_or_emit_token(choice, "a", 1, finished=False)
        for token_id in range(2, 18):
            engine._queue_or_emit_token(choice, chr(96 + token_id), token_id, finished=False)

        self.assertEqual([event["text"] for event in events], ["a", "bcdefghijklmnopq"])

    def test_no_stop_streaming_path_buffers_decode_work(self):
        tokenizer = FakeTokenizer()
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.tokenizer = tokenizer
        engine.stream_token_flush_interval = 16
        seq = Mock()
        seq.finish_reason = None
        seq.is_finished = False
        choice = ChoiceState(
            seq=seq,
            request_id="req",
            choice_index=0,
            prompt_text="",
            prompt_tokens=1,
            stop=[],
            echo=False,
            stream=True,
            decoder=StreamingTextDecoder(tokenizer),
        )

        for token_id in range(65, 82):
            engine._handle_token(choice, token_id)

        self.assertEqual([event["text"] for event in events], ["A", "BCDEFGHIJKLMNOPQ"])
        self.assertEqual(tokenizer.decode_calls, [[65], list(range(66, 82))])

    def test_custom_stream_flush_interval(self):
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.stream_token_flush_interval = 2
        choice = ChoiceState(
            seq=Mock(),
            request_id="req",
            choice_index=0,
            prompt_text="",
            prompt_tokens=1,
            stop=[],
            echo=False,
            stream=True,
            decoder=Mock(),
        )

        for token_id, text in enumerate(["a", "b", "c", "d"], start=1):
            engine._queue_or_emit_token(choice, text, token_id, finished=False)

        self.assertEqual([event["text"] for event in events], ["a", "bc"])
        self.assertEqual(choice.pending_text, "d")

    def test_streaming_finished_event_omits_unused_payload(self):
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.choices = {}
        engine.active_requests = {}
        seq = Mock()
        seq.seq_id = 1
        seq.num_completion_tokens = 2
        choice = ChoiceState(
            seq=seq,
            request_id="req",
            choice_index=0,
            prompt_text="prompt",
            prompt_tokens=1,
            stop=[],
            echo=False,
            stream=True,
            decoder=Mock(),
        )

        engine._finish_choice(choice, "length")

        self.assertEqual(events[0]["type"], "finished")
        self.assertNotIn("text", events[0])
        self.assertNotIn("token_ids", events[0])
        self.assertEqual(events[0]["completion_tokens"], 2)

    def test_blocking_no_stop_decodes_once_at_finish(self):
        tokenizer = FakeTokenizer()
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.tokenizer = tokenizer
        engine.choices = {}
        engine.active_requests = {}
        seq = Mock()
        seq.seq_id = 1
        seq.finish_reason = "length"
        seq.completion_token_ids = [65, 66, 67]
        seq.num_completion_tokens = 3
        choice = ChoiceState(
            seq=seq,
            request_id="req",
            choice_index=0,
            prompt_text="",
            prompt_tokens=1,
            stop=[],
            echo=False,
            stream=False,
            decoder=StreamingTextDecoder(tokenizer),
        )

        for token_id in seq.completion_token_ids:
            engine._handle_token(choice, token_id)

        self.assertEqual(tokenizer.decode_calls, [[65, 66, 67]])
        self.assertEqual(events[0]["text"], "ABC")

    def test_blocking_no_stop_finish_decode_skips_eos_text(self):
        tokenizer = FakeTokenizer()
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.tokenizer = tokenizer
        engine.choices = {}
        engine.active_requests = {}
        seq = Mock()
        seq.seq_id = 1
        seq.finish_reason = "stop"
        seq.completion_token_ids = [65, -1]
        seq.num_completion_tokens = 2
        choice = ChoiceState(
            seq=seq,
            request_id="req",
            choice_index=0,
            prompt_text="",
            prompt_tokens=1,
            stop=[],
            echo=False,
            stream=False,
            decoder=StreamingTextDecoder(tokenizer),
        )

        engine._finish_choice(choice, "stop")

        self.assertEqual(tokenizer.decode_calls, [[65]])
        self.assertEqual(events[0]["text"], "A")

    def test_serving_stats_records_step_shape(self):
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.stats = ServingStats()

        engine._record_step(128)
        engine._record_step(-64)

        self.assertEqual(engine.stats.steps, 2)
        self.assertEqual(engine.stats.prefill_steps, 1)
        self.assertEqual(engine.stats.decode_steps, 1)
        self.assertEqual(engine.stats.prefill_tokens, 128)
        self.assertEqual(engine.stats.decode_tokens, 64)

    def test_fixed_decode_copies_sampled_tokens_once_at_finish(self):
        class FakeBlockManager:

            def __init__(self):
                self.append_calls = 0

            def can_append(self, seq):
                return True

            def may_append(self, seq):
                self.append_calls += 1

        class FakeScheduler:

            def __init__(self):
                self.waiting = []
                self.block_manager = FakeBlockManager()
                self.finished = []

            def finish(self, seq, finish_reason):
                seq.finish_reason = finish_reason
                self.finished.append((seq.seq_id, finish_reason))

        class FakeRunner:

            world_size = 1

            def __init__(self):
                self.calls = []

            def call(self, method, seqs, input_ids=None):
                self.calls.append((method, input_ids.tolist() if input_ids is not None else None))
                return torch.tensor([65 + len(self.calls)])

        tokenizer = FakeTokenizer()
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.tokenizer = tokenizer
        engine.stats = ServingStats()
        engine.scheduler = FakeScheduler()
        engine.model_runner = FakeRunner()
        engine.choices = {}
        engine.active_requests = {"req": Mock(remaining=1)}
        seq = Sequence([80], SamplingParams(max_tokens=3, ignore_eos=True))
        seq.num_cached_tokens = 1
        seq.num_scheduled_tokens = 1
        choice = ChoiceState(
            seq=seq,
            request_id="req",
            choice_index=0,
            prompt_text="",
            prompt_tokens=1,
            stop=[],
            echo=False,
            stream=False,
            decoder=StreamingTextDecoder(tokenizer),
        )
        engine.choices[seq.seq_id] = choice

        self.assertTrue(engine._can_batch_decode_token_readback([seq], is_prefill=False))
        engine._step_fixed_decode([seq], schedule_start=0.0, schedule_end=0.0)

        self.assertEqual([call[0] for call in engine.model_runner.calls], ["run_decode_tokens"] * 3)
        self.assertEqual([call[1] for call in engine.model_runner.calls], [None, [66], [67]])
        self.assertEqual(seq.completion_token_ids, [66, 67, 68])
        self.assertEqual(events[0]["type"], "finished")
        self.assertEqual(events[0]["text"], "BCD")
        self.assertEqual(events[0]["token_ids"], [66, 67, 68])
        self.assertEqual(events[0]["finish_reason"], "length")
        self.assertEqual(engine.scheduler.finished, [(seq.seq_id, "length")])
        self.assertEqual(engine.stats.decode_steps, 3)
        self.assertEqual(engine.stats.decode_tokens, 3)

    def test_batch_completion_requests_tokenizes_simple_string_prompts_together(self):
        tokenizer = FakeTokenizer()
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.tokenizer = tokenizer
        engine.stats = ServingStats()
        added_prompts = []

        def add_request(prompt, sampling_params):
            added_prompts.append(prompt)
            seq = Mock()
            seq.seq_id = len(added_prompts)
            return seq

        engine.add_request = add_request
        engine.active_requests = {}
        engine.choices = {}

        engine.add_completion_requests([
            {
                "type": "completion",
                "request_id": "a",
                "model": "m",
                "prompt": "AB",
                "max_tokens": 1,
                "stream": True,
            },
            {
                "type": "completion",
                "request_id": "b",
                "model": "m",
                "prompt": "CD",
                "max_tokens": 1,
                "stream": True,
            },
        ])

        self.assertEqual(tokenizer.batch_prompts, ["AB", "CD"])
        self.assertEqual(added_prompts, [[65, 66], [67, 68]])
        self.assertEqual(engine.stats.requests, 2)

    def test_batch_completion_requests_keeps_echo_prompt_text(self):
        tokenizer = FakeTokenizer()
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.tokenizer = tokenizer
        engine.stats = ServingStats()
        seq = Mock()
        seq.seq_id = 1
        engine.add_request = Mock(return_value=seq)
        engine.active_requests = {}
        engine.choices = {}

        engine.add_completion_requests([
            {
                "type": "completion",
                "request_id": "a",
                "model": "m",
                "prompt": "AB",
                "max_tokens": 1,
                "stream": True,
                "echo": True,
            },
        ])

        self.assertFalse(hasattr(tokenizer, "batch_prompts"))
        self.assertEqual(engine.choices[1].prompt_text, "AB")


if __name__ == "__main__":
    unittest.main()
