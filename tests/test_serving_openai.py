import unittest

from nanovllm.serve.openai import CompletionRequestError, normalize_completion_request


class FakeTokenizer:

    def __init__(self):
        self.decode_calls = []

    def encode(self, text):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        self.decode_calls.append(list(token_ids))
        return "".join(chr(token_id) for token_id in token_ids)


class CompletionRequestTests(unittest.TestCase):

    def setUp(self):
        self.tokenizer = FakeTokenizer()

    def normalize(self, **payload):
        base = {"request_id": "req", "model": "model", "prompt": "hi"}
        base.update(payload)
        return normalize_completion_request(base, self.tokenizer)

    def test_normalizes_string_prompt(self):
        request = self.normalize(prompt="hi", max_tokens=4, temperature=0.7)
        self.assertEqual(request.prompts[0].token_ids, [104, 105])
        self.assertEqual(request.prompts[0].text, "hi")
        self.assertEqual(request.max_tokens, 4)
        self.assertEqual(request.temperature, 0.7)

    def test_normalizes_prompt_arrays(self):
        strings = self.normalize(prompt=["a", "b"])
        self.assertEqual([prompt.text for prompt in strings.prompts], ["a", "b"])

        tokens = self.normalize(prompt=[65, 66])
        self.assertEqual(tokens.prompts[0].text, "")

        token_prompts = self.normalize(prompt=[[65], [66]])
        self.assertEqual([prompt.text for prompt in token_prompts.prompts], ["", ""])
        self.assertEqual(self.tokenizer.decode_calls, [])

    def test_decodes_token_prompts_when_echo_needs_prompt_text(self):
        tokens = self.normalize(prompt=[65, 66], echo=True)
        self.assertEqual(tokens.prompts[0].text, "AB")

        token_prompts = self.normalize(prompt=[[65], [66]], echo=True)
        self.assertEqual([prompt.text for prompt in token_prompts.prompts], ["A", "B"])

    def test_normalizes_n_stop_stream_echo_and_ignore_eos(self):
        request = self.normalize(n=2, stop=["\n", "END"], stream=True, echo=True, ignore_eos=True)
        self.assertEqual(request.n, 2)
        self.assertEqual(request.stop, ["\n", "END"])
        self.assertTrue(request.stream)
        self.assertTrue(request.echo)
        self.assertTrue(request.ignore_eos)

    def test_rejects_unsupported_fields(self):
        with self.assertRaisesRegex(CompletionRequestError, "top_p"):
            self.normalize(top_p=0.5)
        with self.assertRaisesRegex(CompletionRequestError, "logprobs"):
            self.normalize(logprobs=1)
        with self.assertRaisesRegex(CompletionRequestError, "best_of"):
            self.normalize(best_of=2)

    def test_allows_default_like_unsupported_fields(self):
        request = self.normalize(
            top_p=1,
            presence_penalty=0,
            frequency_penalty=0,
            repetition_penalty=1,
            best_of=1,
            stream_options={"include_usage": True},
        )
        self.assertEqual(request.max_tokens, 16)

    def test_rejects_invalid_prompt_and_booleans(self):
        with self.assertRaisesRegex(CompletionRequestError, "prompt"):
            self.normalize(prompt=[])
        with self.assertRaisesRegex(CompletionRequestError, "prompt"):
            self.normalize(prompt=[-1])
        with self.assertRaisesRegex(CompletionRequestError, "prompt"):
            self.normalize(prompt=[[1], [-1]])
        with self.assertRaisesRegex(CompletionRequestError, "stream"):
            self.normalize(stream="true")
        with self.assertRaisesRegex(CompletionRequestError, "echo"):
            self.normalize(echo="true")
        with self.assertRaisesRegex(CompletionRequestError, "ignore_eos"):
            self.normalize(ignore_eos="true")


if __name__ == "__main__":
    unittest.main()
