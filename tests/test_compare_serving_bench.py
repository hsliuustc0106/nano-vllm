import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tools.compare_serving_bench import apply_preset, option_was_provided, resolve_vllm_cmd


class CompareServingBenchTests(unittest.TestCase):

    def test_applies_short_throughput_preset(self):
        args = SimpleNamespace(preset="short-throughput", overrides=set())

        apply_preset(args)

        self.assertEqual(args.num_prompts, 128)
        self.assertEqual(args.min_input_len, 128)
        self.assertEqual(args.max_input_len, 128)
        self.assertEqual(args.max_tokens, 64)
        self.assertEqual(args.concurrency, 64)
        self.assertEqual(args.max_model_len, 1024)
        self.assertEqual(args.max_num_seqs, 64)
        self.assertEqual(args.max_num_batched_tokens, 8192)

    def test_applies_long_context_presets(self):
        args = SimpleNamespace(preset="long-throughput-8k-1k", overrides=set())

        apply_preset(args)

        self.assertEqual(args.num_prompts, 16)
        self.assertEqual(args.min_input_len, 8192)
        self.assertEqual(args.max_input_len, 8192)
        self.assertEqual(args.max_tokens, 1024)
        self.assertEqual(args.concurrency, 16)
        self.assertEqual(args.max_model_len, 32768)
        self.assertEqual(args.max_num_seqs, 16)
        self.assertEqual(args.max_num_batched_tokens, 32768)

    def test_explicit_values_override_preset(self):
        args = SimpleNamespace(
            preset="low-latency-32k-2k",
            overrides={"concurrency", "max_tokens"},
            concurrency=2,
            max_tokens=128,
        )

        apply_preset(args)

        self.assertEqual(args.num_prompts, 4)
        self.assertEqual(args.concurrency, 2)
        self.assertEqual(args.max_tokens, 128)
        self.assertEqual(args.max_model_len, 40960)

    def test_detects_equals_style_options(self):
        self.assertTrue(option_was_provided("--max-tokens", ["--max-tokens=128"]))
        self.assertTrue(option_was_provided("--max-tokens", ["--max-tokens", "128"]))
        self.assertFalse(option_was_provided("--max-tokens", ["--max-model-len=128"]))

    def test_explicit_vllm_bin_overrides_environment_resolution(self):
        self.assertEqual(resolve_vllm_cmd("/tmp/vllm"), ["/tmp/vllm"])

    def test_vllm_module_resolution_prefers_active_python(self):
        with patch("tools.compare_serving_bench.importlib.util.find_spec", return_value=object()):
            self.assertEqual(resolve_vllm_cmd(None), [sys.executable, "-m", "vllm.entrypoints.cli.main"])

    def test_path_resolution_when_module_missing(self):
        with patch("tools.compare_serving_bench.importlib.util.find_spec", return_value=None):
            with patch("tools.compare_serving_bench.shutil.which", return_value="/usr/bin/vllm"):
                self.assertEqual(resolve_vllm_cmd(None), ["/usr/bin/vllm"])


if __name__ == "__main__":
    unittest.main()
