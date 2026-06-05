class StreamingTextDecoder:

    def __init__(self, tokenizer, max_pending_tokens: int = 8):
        self.tokenizer = tokenizer
        self.max_pending_tokens = max_pending_tokens
        self.pending_token_ids: list[int] = []
        self.text = ""

    def append(self, token_id: int, finished: bool = False) -> str:
        self.pending_token_ids.append(token_id)
        return self.flush(finished=finished)

    def flush(self, finished: bool = False) -> str:
        if not self.pending_token_ids:
            return self.text
        decoded = self.tokenizer.decode(self.pending_token_ids)
        has_replacement = "\ufffd" in decoded
        if has_replacement and not finished and len(self.pending_token_ids) < self.max_pending_tokens:
            return self.text
        self.text += decoded
        self.pending_token_ids.clear()
        return self.text


def find_stop(text: str, stops: list[str]) -> int | None:
    indexes = [text.find(stop) for stop in stops if stop in text]
    return min(indexes) if indexes else None


def trim_stop(text: str, stops: list[str]) -> tuple[str, bool]:
    stop_index = find_stop(text, stops)
    if stop_index is None:
        return text, False
    return text[:stop_index], True


def next_delta(emitted_text: str, completion_text: str, stops: list[str], finished: bool) -> tuple[str, str]:
    target = completion_text
    max_stop_len = max((len(stop) for stop in stops), default=0)
    if not finished and max_stop_len > 1:
        target = completion_text[:max(0, len(completion_text) - max_stop_len + 1)]
    if len(target) < len(emitted_text):
        return target, ""
    return target, target[len(emitted_text):]
