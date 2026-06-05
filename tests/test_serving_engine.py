import unittest
from unittest.mock import Mock, patch

from nanovllm.serve.engine import ChoiceState, ServingLLMEngine, ZmqEngineServer
from nanovllm.serve.text import StreamingTextDecoder


class FakeTokenizer:

    eos_token_id = -1

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids):
        self.decode_calls.append(list(token_ids))
        return "".join(chr(token_id) for token_id in token_ids)


class ServingEngineTests(unittest.TestCase):

    def test_profile_control_messages_call_cuda_profiler(self):
        server = ZmqEngineServer.__new__(ZmqEngineServer)
        server.engine = Mock()
        cudart = Mock()
        with patch("nanovllm.serve.engine.torch.cuda.cudart", return_value=cudart):
            server._handle_message({"type": "profile_start"})
            server._handle_message({"type": "profile_stop"})

        cudart.cudaProfilerStart.assert_called_once_with()
        cudart.cudaProfilerStop.assert_called_once_with()

    def test_streaming_tokens_flush_first_and_then_coalesce(self):
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
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
        engine._queue_or_emit_token(choice, "b", 2, finished=False)
        engine._queue_or_emit_token(choice, "c", 3, finished=False)
        engine._queue_or_emit_token(choice, "d", 4, finished=False)
        engine._queue_or_emit_token(choice, "e", 5, finished=False)

        self.assertEqual([event["text"] for event in events], ["a", "bcde"])
        self.assertEqual(choice.pending_text, "")

    def test_stop_strings_disable_streaming_token_coalescing(self):
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
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
        engine._queue_or_emit_token(choice, "b", 2, finished=False)

        self.assertEqual([event["text"] for event in events], ["a", "b"])

    def test_no_stop_streaming_path_buffers_decode_work(self):
        tokenizer = FakeTokenizer()
        events = []
        engine = ServingLLMEngine.__new__(ServingLLMEngine)
        engine.event_sink = events.append
        engine.tokenizer = tokenizer
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

        for token_id in [65, 66, 67, 68, 69]:
            engine._handle_token(choice, token_id)

        self.assertEqual([event["text"] for event in events], ["A", "BCDE"])
        self.assertEqual(tokenizer.decode_calls, [[65], [66, 67, 68, 69]])

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


if __name__ == "__main__":
    unittest.main()
