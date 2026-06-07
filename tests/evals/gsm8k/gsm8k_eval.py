"""GSM8K evaluation helper for Nano-vLLM."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pathlib import Path

INVALID = -9999999


def _cache_path_for_url(url: str) -> str:
    filename = url.split("/")[-1]
    return os.path.join("/tmp", filename)


def _download_and_cache_file(url: str, filename: str | None = None) -> str:
    """Download and cache a remote text file if needed."""

    if filename is None:
        filename = _cache_path_for_url(url)

    path = Path(filename)
    if path.exists():
        return str(path)

    print(f"Downloading {url} to {filename}")
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return str(path)


def read_jsonl(filename: str) -> Iterable[dict[str, Any]]:
    """Read a JSONL file."""

    with open(filename, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield json.loads(line)


def load_gsm8k_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load GSM8K train and test splits from OpenAI's grade-school-math dataset."""

    train_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
    test_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

    train_file = _download_and_cache_file(train_url)
    test_file = _download_and_cache_file(test_url)

    train_data = list(read_jsonl(train_file))
    test_data = list(read_jsonl(test_file))
    return train_data, test_data


def _extract_number_like(text: str) -> str | None:
    """Extract the most likely final numeric answer candidate from text."""

    marker_match = re.findall(r"####\s*([\-+]?\d+[\d,]*)", text)
    if marker_match:
        return marker_match[-1]

    numbers = re.findall(r"[\-+]?\d+[\d,]*", text)
    if not numbers:
        return None

    return numbers[-1]


def get_answer_value(answer_str: str) -> int:
    """Convert response text into a numeric answer when possible."""

    answer_str = answer_str.replace(",", "")
    value = _extract_number_like(answer_str)
    if value is None:
        return INVALID

    try:
        return int(value)
    except ValueError:
        return INVALID


def build_gsm8k_prompts(
    num_questions: int = 1319,
    num_shots: int = 5,
) -> tuple[list[str], list[int]]:
    """Build few-shot prompts and labels for GSM8K."""

    if num_questions == 0:
        return [], []

    train_data, test_data = load_gsm8k_data()
    num_questions = min(num_questions, len(test_data))

    few_shot_examples = []
    for i in range(min(num_shots, len(train_data))):
        few_shot_examples.append(
            f"Question: {train_data[i]['question']}\n"
            f"Answer: {train_data[i]['answer']}\n\n"
        )
    few_shot_prefix = "".join(few_shot_examples)

    prompts = []
    labels = []
    for i in range(num_questions):
        prompts.append(few_shot_prefix + f"Question: {test_data[i]['question']}\nAnswer:")
        labels.append(get_answer_value(test_data[i]["answer"]))

    if labels and any(label == INVALID for label in labels):
        raise ValueError("GSM8K labels contain values we cannot parse as integers")

    return prompts, labels


def build_completion_url(base_url: str) -> str:
    if base_url.endswith("/v1/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/completions"

    return f"{base_url.rstrip('/')}/v1/completions"


def call_completion_endpoint(
    url: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    stop: list[str] | None = None,
    timeout_seconds: int = 600,
    seed: int | None = None,
) -> tuple[str, int]:
    """Call Nano-vLLM completions endpoint using stdlib networking."""

    payload: dict[str, Any] = {
        "prompt": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": stop,
    }
    if seed is not None:
        payload["seed"] = seed

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8"))
        text = result["choices"][0]["text"]
        completion_tokens = result.get("usage", {}).get("completion_tokens", 0)
        return text, completion_tokens
    except (urllib.error.URLError, TimeoutError, OSError, KeyError, ValueError) as e:
        print(f"Error calling /v1/completions: {type(e).__name__}: {e}")
        return "", 0


def score_gsm8k(
    states: list[str],
    output_tokens: list[int],
    labels: list[int],
    num_shots: int,
    max_tokens: int,
    latency: float,
) -> dict[str, float | int]:
    """Build final score dict for GSM8K results."""

    num_questions = len(labels)
    if num_questions == 0:
        return {
            "accuracy": 0.0,
            "invalid_rate": 0.0,
            "latency": 0.0,
            "questions_per_second": 0.0,
            "total_output_tokens": 0,
            "tokens_per_second": 0.0,
            "num_questions": 0,
            "num_shots": num_shots,
            "max_tokens": max_tokens,
            "timestamp": time.time(),
        }

    preds = [get_answer_value(state) for state in states]
    accuracy = sum(1 for pred, label in zip(preds, labels) if pred == label) / num_questions
    invalid_rate = sum(1 for pred in preds if pred == INVALID) / num_questions
    total_output_tokens = sum(output_tokens)
    questions_per_second = num_questions / latency if latency > 0 else 0.0
    tokens_per_second = total_output_tokens / latency if latency > 0 else 0.0

    return {
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "latency": latency,
        "questions_per_second": questions_per_second,
        "total_output_tokens": total_output_tokens,
        "tokens_per_second": tokens_per_second,
        "num_questions": num_questions,
        "num_shots": num_shots,
        "max_tokens": max_tokens,
        "timestamp": time.time(),
    }


def evaluate_gsm8k(
    num_questions: int = 1319,
    num_shots: int = 5,
    max_tokens: int = 256,
    base_url: str = "http://127.0.0.1:8000",
    stop: list[str] | None = None,
    temperature: float = 0.0,
    seed: int | None = 42,
    request_timeout_seconds: float = 600,
    workers: int = 16,
) -> dict[str, float | int]:
    """Evaluate GSM8K against a live Nano-vLLM server."""

    prompts, labels = build_gsm8k_prompts(num_questions, num_shots)
    num_questions = len(prompts)

    stop_tokens = stop or ["Question", "Assistant:", "<|separator|>"]
    endpoint = build_completion_url(base_url)

    def run_one(index_prompt: tuple[int, str]):
        index, prompt = index_prompt
        text, tokens = call_completion_endpoint(
            url=endpoint,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop_tokens,
            timeout_seconds=request_timeout_seconds,
            seed=seed,
        )
        return index, text, tokens

    questions = list(enumerate(prompts))
    states: list[str] = [""] * num_questions
    output_tokens: list[int] = [0] * num_questions

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=min(workers, max(1, num_questions))) as pool:
        futures = [pool.submit(run_one, qp) for qp in questions]
        for future in as_completed(futures):
            index, text, tokens = future.result()
            states[index] = text
            output_tokens[index] = tokens

    latency = time.perf_counter() - start
    return score_gsm8k(states, output_tokens, labels, num_shots, max_tokens, latency)


def check_server_ready(url: str, timeout_seconds: float = 5.0) -> bool:
    """Check if /v1/models responds, which implies serving process is running."""

    models_url = url.rstrip("/")
    if models_url.endswith("/v1/completions"):
        models_url = models_url[: -len("/completions")]
    if not models_url.endswith("/v1"):
        models_url = f"{models_url.rstrip('/')}/v1"
    models_url = f"{models_url}/models"

    request = urllib.request.Request(models_url)
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return response.getcode() == 200
    except Exception:
        return False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run GSM8K evaluation")
    parser.add_argument("--num-shots", type=int, default=5)
    parser.add_argument("--num-questions", type=int, default=1319)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--stop", type=str, default="Question,Assistant:,<|separator|>")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout-seconds", type=float, default=600)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--save-results", type=str)
    args = parser.parse_args()

    stop_tokens = [token for token in args.stop.split(",") if token]
    result = evaluate_gsm8k(
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
        stop=stop_tokens,
        temperature=args.temperature,
        seed=args.seed,
        request_timeout_seconds=args.request_timeout_seconds,
        workers=args.workers,
    )

    print("\nResults:")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print(f"Invalid responses: {result['invalid_rate']:.3f}")
    print(f"Total latency: {result['latency']:.3f} s")
    print(f"Questions per second: {result['questions_per_second']:.3f}")
    print(f"Total output tokens: {result['total_output_tokens']}")
    print(f"Output tokens per second: {result['tokens_per_second']:.3f}")

    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to: {args.save_results}")


if __name__ == "__main__":
    main()
