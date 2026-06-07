import unittest

from nanovllm.engine.model_runner import ModelRunner


class ModelRunnerGraphBatchTests(unittest.TestCase):

    def test_graph_batch_sizes_include_non_multiple_max_batch(self):
        self.assertEqual(ModelRunner.build_graph_batch_sizes(31), [1, 2, 4, 8, 16, 31])

    def test_graph_batch_lookup_maps_to_next_captured_batch(self):
        graph_bs = ModelRunner.build_graph_batch_sizes(31)
        lookup = ModelRunner.build_graph_batch_lookup(graph_bs, 31)

        self.assertEqual(lookup[1], 1)
        self.assertEqual(lookup[3], 4)
        self.assertEqual(lookup[9], 16)
        self.assertEqual(lookup[17], 31)
        self.assertEqual(lookup[31], 31)


if __name__ == "__main__":
    unittest.main()
