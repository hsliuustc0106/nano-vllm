import unittest
from unittest.mock import Mock, patch

from nanovllm.serve.engine import ZmqEngineServer


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


if __name__ == "__main__":
    unittest.main()
