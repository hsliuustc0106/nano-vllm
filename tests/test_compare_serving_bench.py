import unittest
from types import SimpleNamespace

from tools.compare_serving_bench import apply_preset, option_was_provided


class CompareServingBenchTests(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
