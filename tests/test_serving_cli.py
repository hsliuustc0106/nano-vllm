import os
import signal
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from nanovllm.serve.cli import _frontend_binary, _serve, build_arg_parser


class ServingCliTests(unittest.TestCase):

    def test_serve_subcommand_defaults(self):
        args = build_arg_parser().parse_args(["serve", "--model", "/models/Qwen3-0.6B"])
        self.assertEqual(args.command, "serve")
        self.assertEqual(args.model, "/models/Qwen3-0.6B")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8000)
        self.assertEqual(args.request_endpoint, "tcp://127.0.0.1:5557")
        self.assertEqual(args.event_endpoint, "tcp://127.0.0.1:5558")
        self.assertEqual(args.kvcache_block_size, 256)
        self.assertEqual(args.stream_token_flush_interval, 16)
        self.assertEqual(args.log_serving_stats_interval, 0.0)

    def test_serve_subcommand_accepts_served_name_and_frontend_binary(self):
        args = build_arg_parser().parse_args([
            "serve",
            "--model",
            "/models/qwen",
            "--served-model-name",
            "Qwen3-0.6B",
            "--frontend-binary",
            "/tmp/nanovllm-serve",
        ])
        self.assertEqual(args.served_model_name, "Qwen3-0.6B")
        self.assertEqual(args.frontend_binary, "/tmp/nanovllm-serve")

    def test_serve_subcommand_accepts_stream_flush_interval(self):
        args = build_arg_parser().parse_args([
            "serve",
            "--model",
            "/models/qwen",
            "--stream-token-flush-interval",
            "1",
            "--log-serving-stats-interval",
            "2.5",
            "--kvcache-block-size",
            "16",
        ])
        self.assertEqual(args.stream_token_flush_interval, 1)
        self.assertEqual(args.log_serving_stats_interval, 2.5)
        self.assertEqual(args.kvcache_block_size, 16)

    def test_frontend_binary_finds_scripts_dir_sibling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binary = os.path.join(tmpdir, "nanovllm-serve")
            with open(binary, "w", encoding="utf-8") as handle:
                handle.write("")
            with patch("nanovllm.serve.cli.sysconfig.get_path", return_value=tmpdir):
                self.assertEqual(_frontend_binary(None), binary)

    def test_serve_returns_zero_on_intentional_shutdown(self):
        engine = Mock()
        frontend = Mock()
        engine.poll.side_effect = [None, None, -signal.SIGTERM]
        frontend.poll.return_value = None
        engine.wait.return_value = -signal.SIGTERM
        frontend.wait.return_value = -signal.SIGTERM
        args = SimpleNamespace(
            model="/models/qwen",
            served_model_name="qwen",
            host="127.0.0.1",
            port=8000,
            request_endpoint="tcp://127.0.0.1:5557",
            event_endpoint="tcp://127.0.0.1:5558",
            tensor_parallel_size=1,
            max_model_len=4096,
            max_num_seqs=512,
            max_num_batched_tokens=16384,
            kvcache_block_size=256,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
            frontend_binary="/tmp/nanovllm-serve",
        )

        def interrupt_once(_seconds):
            import nanovllm.serve.cli as cli
            cli.signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

        with patch("nanovllm.serve.cli.subprocess.Popen", side_effect=[engine, frontend]):
            with patch("nanovllm.serve.cli.time.sleep", side_effect=interrupt_once):
                self.assertEqual(_serve(args), 0)

    def test_serve_preserves_child_failure_exit_code(self):
        engine = Mock()
        frontend = Mock()
        engine.poll.return_value = 7
        frontend.poll.return_value = None
        engine.wait.return_value = 7
        frontend.wait.return_value = -signal.SIGTERM
        args = SimpleNamespace(
            model="/models/qwen",
            served_model_name="qwen",
            host="127.0.0.1",
            port=8000,
            request_endpoint="tcp://127.0.0.1:5557",
            event_endpoint="tcp://127.0.0.1:5558",
            tensor_parallel_size=1,
            max_model_len=4096,
            max_num_seqs=512,
            max_num_batched_tokens=16384,
            kvcache_block_size=256,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
            frontend_binary="/tmp/nanovllm-serve",
        )

        with patch("nanovllm.serve.cli.subprocess.Popen", side_effect=[engine, frontend]):
            self.assertEqual(_serve(args), 7)

    def test_serve_cleans_up_engine_when_frontend_launch_fails(self):
        engine = Mock()
        engine.poll.return_value = None
        engine.wait.return_value = 0
        args = SimpleNamespace(
            model="/models/qwen",
            served_model_name="qwen",
            host="127.0.0.1",
            port=8000,
            request_endpoint="tcp://127.0.0.1:5557",
            event_endpoint="tcp://127.0.0.1:5558",
            tensor_parallel_size=1,
            max_model_len=4096,
            max_num_seqs=512,
            max_num_batched_tokens=16384,
            kvcache_block_size=256,
            gpu_memory_utilization=0.9,
            enforce_eager=False,
            frontend_binary="/tmp/nanovllm-serve",
        )

        with patch(
            "nanovllm.serve.cli.subprocess.Popen",
            side_effect=[engine, OSError("frontend launch failed")],
        ):
            with self.assertRaisesRegex(OSError, "frontend launch failed"):
                _serve(args)

        engine.terminate.assert_called_once_with()
        engine.wait.assert_called_once_with(timeout=10)


if __name__ == "__main__":
    unittest.main()
