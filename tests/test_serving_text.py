import unittest

from nanovllm.serve.text import StreamingTextDecoder, next_delta, trim_stop


class FakeTokenizer:

    def __init__(self):
        self.decode_calls = []

    def decode(self, token_ids):
        self.decode_calls.append(list(token_ids))
        pieces = {
            (1,): "hello",
            (2,): " \ufffd",
            (3,): "\ufffd",
            (2, 3): " 😊",
            (4,): " world",
        }
        return pieces[tuple(token_ids)]


class ServingTextTests(unittest.TestCase):

    def test_streaming_decoder_buffers_incomplete_unicode(self):
        tokenizer = FakeTokenizer()
        decoder = StreamingTextDecoder(tokenizer)

        self.assertEqual(decoder.append(1), "hello")
        self.assertEqual(decoder.append(2), "hello")
        self.assertEqual(decoder.append(3), "hello 😊")
        self.assertEqual(decoder.append(4), "hello 😊 world")
        self.assertEqual(tokenizer.decode_calls, [[1], [2], [2, 3], [4]])

    def test_streaming_decoder_flushes_pending_text_when_finished(self):
        tokenizer = FakeTokenizer()
        decoder = StreamingTextDecoder(tokenizer)

        self.assertEqual(decoder.append(2), "")
        self.assertEqual(decoder.flush(finished=True), " \ufffd")

    def test_stop_string_trims_completion_text(self):
        text, stopped = trim_stop("hello END trailing", ["END"])
        self.assertEqual(text, "hello ")
        self.assertTrue(stopped)

    def test_next_delta_withholds_potential_stop_suffix(self):
        emitted, delta = next_delta("", "hello EN", ["END"], finished=False)
        self.assertEqual(emitted, "hello ")
        self.assertEqual(delta, "hello ")

        trimmed, stopped = trim_stop("hello END", ["END"])
        emitted, delta = next_delta(emitted, trimmed, ["END"], finished=True)
        self.assertEqual(emitted, "hello ")
        self.assertEqual(delta, "")
        self.assertTrue(stopped)


if __name__ == "__main__":
    unittest.main()
