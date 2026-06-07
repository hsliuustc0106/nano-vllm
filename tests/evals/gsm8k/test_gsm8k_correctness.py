"""GSM8K accuracy regression test for Nano-vLLM serving."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover
    pytest = None

from .gsm8k_eval import check_server_ready, evaluate_gsm8k


def _parse_scalar(raw: str) -> str | int | float | None:
    """Parse a scalar value from a minimal YAML-like config file."""

    text = raw.strip()
    if not text:
        return None

    if text.startswith(("'", '"')) and text.endswith(text[0:1]):
        return text[1:-1]

    lower = text.lower()
    if lower in {"true", "false"}:
        return lower == "true"

    if re_match := re.fullmatch(r"-?\d+", text):
        return int(re_match.group(0))

    if re_match := re.fullmatch(r"-?(?:\d+\.\d+|\d+)", text):
        return float(re_match.group(0))

    return text


def _parse_config_file(config_path: Path) -> dict[str, Any]:
    """Parse a tiny subset of YAML used by upstream style config files."""

    data: dict[str, Any] = {}
    with open(config_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if ":" not in line:
                raise ValueError(f"Invalid config entry: {line}")

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue

            if value == "" and key != "base_url":
                data[key] = ""
            elif value.startswith("[") and value.endswith("]"):
                data[key] = [item.strip() for item in value[1:-1].split(",") if item.strip()]
            elif key in {"stop", "stop_tokens"}:
                parsed = _parse_scalar(value)
                if isinstance(parsed, str):
                    data[key] = [token.strip() for token in parsed.split(",") if token.strip()]
                else:
                    data[key] = parsed
            else:
                data[key] = _parse_scalar(value)

    if not data.get("base_url"):
        data["base_url"] = "http://127.0.0.1:8000"

    return data


def _assert_with_tolerance(result: dict[str, float | int], config: dict[str, Any]) -> None:
    measured = float(result["accuracy"])
    expected = float(config.get("accuracy_threshold", 0.0))
    tolerance = float(config.get("tolerance", 0.08))

    assert measured >= expected - tolerance, (
        f"GSM8K metric too low: {measured:.4f} < "
        f"{expected:.4f} - {tolerance:.4f} = {expected - tolerance:.4f}"
    )


def test_gsm8k_correctness(config_filename: Path) -> None:
    """Run GSM8K accuracy check for one server config."""

    config = _parse_config_file(config_filename)

    base_url = str(config.get("base_url", "http://127.0.0.1:8000"))
    if not check_server_ready(base_url):
        if pytest is None:
            print(f"Skipping: server not ready at {base_url}")
            return
        pytest.skip(f"Skipping: server not ready at {base_url}")

    stop = config.get("stop") or ["Question", "Assistant:", "<|separator|>"]

    result = evaluate_gsm8k(
        num_questions=int(config.get("num_questions", 20)),
        num_shots=int(config.get("num_fewshot", 5)),
        max_tokens=int(config.get("max_tokens", 256)),
        base_url=base_url,
        stop=stop if isinstance(stop, list) else [str(stop)],
        temperature=float(config.get("temperature", 0.0)),
        seed=int(config.get("seed", 42)),
        request_timeout_seconds=float(config.get("request_timeout_seconds", 600)),
        workers=int(config.get("workers", 16)),
    )

    print(f"GSM8K Results for {config.get('model_name', 'unknown')}:")
    print(f"  accuracy: {result['accuracy']:.4f}")
    print(f"  invalid_rate: {result['invalid_rate']:.4f}")
    print(f"  latency: {result['latency']:.1f}s")
    print(f"  qps: {result['questions_per_second']:.1f}")

    _assert_with_tolerance(result, config)
