import unittest
from unittest.mock import Mock, patch

from nanovllm.serve.engine import ChoiceState, ServingLLMEngine, ZmqEngineServer


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


if __name__ == "__main__":
    unittest.main()
