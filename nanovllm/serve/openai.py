from dataclasses import dataclass
from typing import Any


class CompletionRequestError(ValueError):
    pass


@dataclass(slots=True)
class NormalizedPrompt:
    token_ids: list[int]
    text: str


@dataclass(slots=True)
class NormalizedCompletionRequest:
    request_id: str
    model: str
    prompts: list[NormalizedPrompt]
    max_tokens: int
    temperature: float
    stream: bool
    n: int
    stop: list[str]
    echo: bool
    ignore_eos: bool


_ALLOWED_FIELDS = {
    "type",
    "request_id",
    "model",
    "prompt",
    "max_tokens",
    "temperature",
    "stream",
    "n",
    "stop",
    "echo",
    "ignore_eos",
}


def _is_int_list(value: list[Any]) -> bool:
    return all(isinstance(item, int) and not isinstance(item, bool) for item in value)


def _is_str_list(value: list[Any]) -> bool:
    return all(isinstance(item, str) for item in value)


def _is_token_prompt_list(value: list[Any]) -> bool:
    return all(isinstance(item, list) and _is_int_list(item) for item in value)


def _is_default_like(name: str, value: Any) -> bool:
    if value is None:
        return True
    if name in {"top_p", "best_of"}:
        return value == 1
    if name in {"presence_penalty", "frequency_penalty"}:
        return value == 0
    if name == "repetition_penalty":
        return value == 1
    if name == "logit_bias":
        return value == {}
    if name == "stream_options":
        return isinstance(value, dict) and set(value) <= {"include_usage"} and isinstance(
            value.get("include_usage", False), bool
        )
    return False


def _validate_unsupported(payload: dict[str, Any]):
    for name in (
        "logprobs",
        "suffix",
        "top_p",
        "presence_penalty",
        "frequency_penalty",
        "repetition_penalty",
        "logit_bias",
        "best_of",
        "stream_options",
    ):
        if name in payload and not _is_default_like(name, payload[name]):
            raise CompletionRequestError(f"unsupported field: {name}")
    for name, value in payload.items():
        if name not in _ALLOWED_FIELDS and not _is_default_like(name, value):
            raise CompletionRequestError(f"unsupported field: {name}")


def _normalize_stop(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stops = [value]
    elif isinstance(value, list) and _is_str_list(value):
        stops = value
    else:
        raise CompletionRequestError("stop must be a string or a list of strings")
    if len(stops) > 4:
        raise CompletionRequestError("stop supports at most 4 sequences")
    if any(stop == "" for stop in stops):
        raise CompletionRequestError("stop sequences must be non-empty")
    return stops


def _normalize_prompt(prompt: Any, tokenizer) -> list[NormalizedPrompt]:
    if isinstance(prompt, str):
        prompts = [(tokenizer.encode(prompt), prompt)]
    elif isinstance(prompt, list) and _is_str_list(prompt):
        prompts = [(tokenizer.encode(item), item) for item in prompt]
    elif isinstance(prompt, list) and _is_int_list(prompt):
        prompts = [(prompt, tokenizer.decode(prompt))]
    elif isinstance(prompt, list) and _is_token_prompt_list(prompt):
        prompts = [(item, tokenizer.decode(item)) for item in prompt]
    else:
        raise CompletionRequestError("prompt must be a string, list of strings, token ids, or list of token id lists")
    normalized = [NormalizedPrompt(list(token_ids), text) for token_ids, text in prompts]
    if not normalized:
        raise CompletionRequestError("prompt must contain at least one item")
    if any(not item.token_ids for item in normalized):
        raise CompletionRequestError("prompt must produce at least one token")
    return normalized


def normalize_completion_request(payload: dict[str, Any], tokenizer) -> NormalizedCompletionRequest:
    _validate_unsupported(payload)
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise CompletionRequestError("request_id must be a non-empty string")
    model = payload.get("model")
    if not isinstance(model, str) or not model:
        raise CompletionRequestError("model must be a non-empty string")
    if "prompt" not in payload:
        raise CompletionRequestError("prompt is required")
    max_tokens = payload.get("max_tokens", 16)
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
        raise CompletionRequestError("max_tokens must be a positive integer")
    temperature = payload.get("temperature", 1.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool) or temperature <= 1e-10:
        raise CompletionRequestError("temperature must be greater than 1e-10")
    n = payload.get("n", 1)
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise CompletionRequestError("n must be a positive integer")
    stream = payload.get("stream", False)
    if not isinstance(stream, bool):
        raise CompletionRequestError("stream must be a boolean")
    echo = payload.get("echo", False)
    if not isinstance(echo, bool):
        raise CompletionRequestError("echo must be a boolean")
    ignore_eos = payload.get("ignore_eos", False)
    if not isinstance(ignore_eos, bool):
        raise CompletionRequestError("ignore_eos must be a boolean")
    return NormalizedCompletionRequest(
        request_id=request_id,
        model=model,
        prompts=_normalize_prompt(payload["prompt"], tokenizer),
        max_tokens=max_tokens,
        temperature=float(temperature),
        stream=stream,
        n=n,
        stop=_normalize_stop(payload.get("stop")),
        echo=echo,
        ignore_eos=ignore_eos,
    )
