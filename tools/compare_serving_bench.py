import argparse
import importlib.util
import os
import shutil
import statistics
import subprocess
import sys
import time
from random import Random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


BENCHMARK_PRESETS = {
    "short-throughput": {
        "num_prompts": 128,
        "min_input_len": 128,
        "max_input_len": 128,
        "max_tokens": 64,
        "concurrency": 64,
        "max_model_len": 1024,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 8192,
    },
    "long-throughput-8k-1k": {
        "num_prompts": 16,
        "min_input_len": 8192,
        "max_input_len": 8192,
        "max_tokens": 1024,
        "concurrency": 16,
        "max_model_len": 32768,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 32768,
    },
    "low-latency-32k-2k": {
        "num_prompts": 4,
        "min_input_len": 32768,
        "max_input_len": 32768,
        "max_tokens": 2048,
        "concurrency": 1,
        "max_model_len": 40960,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 32768,
    },
}


def apply_preset(args):
    if args.preset is None:
        return
    preset = BENCHMARK_PRESETS[args.preset]
    for name, value in preset.items():
        if name not in args.overrides:
            setattr(args, name, value)


def option_was_provided(option: str, argv: list[str]) -> bool:
    return option in argv or any(arg.startswith(f"{option}=") for arg in argv)


def make_prompts(args):
    rng = Random(args.seed)
    return [
        [rng.randint(0, args.vocab_size - 1) for _ in range(rng.randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_prompts)
    ]


def percentile(values, pct):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def print_result(label, elapsed, completion_units, request_latencies=None, unit_name="completion_tokens"):
    throughput = completion_units / elapsed if elapsed > 0 else 0.0
    print(f"{label}_wall_s {elapsed:.6f}")
    print(f"{label}_{unit_name} {completion_units}")
    suffix = "tok" if unit_name == "completion_tokens" else "chunk"
    print(f"{label}_throughput_{suffix}_s {throughput:.2f}")
    if request_latencies:
        print(f"{label}_request_latency_mean_s {statistics.mean(request_latencies):.6f}")
        print(f"{label}_request_latency_p50_s {percentile(request_latencies, 50):.6f}")
        print(f"{label}_request_latency_p95_s {percentile(request_latencies, 95):.6f}")


def run_offline(args, prompts):
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        os.path.expanduser(args.model),
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=False,
    )
    llm.generate([[1]], SamplingParams(max_tokens=1), use_tqdm=False)
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    completion_tokens = sum(len(output["token_ids"]) for output in outputs)
    print_result("offline", elapsed, completion_tokens)


def random_input_len_and_ratio(args):
    if args.min_input_len <= 0 or args.max_input_len <= 0:
        raise ValueError("input lengths must be positive")
    if args.min_input_len > args.max_input_len:
        raise ValueError("min-input-len must be <= max-input-len")
    input_len = max(1, round((args.min_input_len + args.max_input_len) / 2))
    ratio = (args.max_input_len - args.min_input_len) / (args.max_input_len + args.min_input_len)
    return input_len, ratio


def run_online(args):
    input_len, range_ratio = random_input_len_and_ratio(args)
    if args.stream:
        print("note: vllm bench serve uses streaming OpenAI completions for this backend.", flush=True)
    cmd = resolve_vllm_cmd(args.vllm_bin) + [
        "bench",
        "serve",
        "--backend",
        "openai",
        "--endpoint",
        "/v1/completions",
        "--base-url",
        args.base_url,
        "--model",
        args.served_model,
        "--tokenizer",
        os.path.expanduser(args.model),
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(args.max_tokens),
        "--random-range-ratio",
        f"{range_ratio:.6g}",
        "--num-prompts",
        str(args.num_prompts),
        "--max-concurrency",
        str(args.concurrency),
        "--request-rate",
        args.request_rate,
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--disable-tqdm",
    ]
    if args.num_warmups:
        cmd.extend(["--num-warmups", str(args.num_warmups)])
    if args.top_p is not None:
        cmd.extend(["--top-p", str(args.top_p)])
    if args.top_k is not None:
        cmd.extend(["--top-k", str(args.top_k)])
    if args.min_p is not None:
        cmd.extend(["--min-p", str(args.min_p)])
    if args.extra_body:
        cmd.extend(["--extra-body", args.extra_body])
    print("vllm_bench_cmd", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def resolve_vllm_cmd(vllm_bin: str | None) -> list[str]:
    if vllm_bin:
        return [vllm_bin]
    if importlib.util.find_spec("vllm") is not None:
        return [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    binary = shutil.which("vllm")
    if binary is not None:
        return [binary]
    raise RuntimeError("vllm is not installed; run `uv pip install vllm` or pass --vllm-bin")


def main():
    parser = argparse.ArgumentParser(description="Compare Nano-vLLM offline batch and online HTTP serving speed.")
    parser.add_argument(
        "--preset",
        choices=sorted(BENCHMARK_PRESETS),
        help="Apply a named benchmark shape. Explicit command-line values override preset defaults.",
    )
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", default="Qwen3-0.6B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--min-input-len", type=int, default=16)
    parser.add_argument("--max-input-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--extra-body", help="JSON object forwarded to `vllm bench serve --extra-body`.")
    parser.add_argument("--num-warmups", type=int, default=0, help="Forwarded to `vllm bench serve --num-warmups`; 0 preserves cold first-burst behavior.")
    parser.add_argument("--stream", action="store_true", help="Accepted for compatibility; vLLM's OpenAI completions benchmark streams by default.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--vllm-bin", default=None)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()
    args.overrides = {
        action.dest
        for action in parser._actions
        if action.option_strings
        for option in action.option_strings
        if option_was_provided(option, sys.argv[1:])
    }
    apply_preset(args)

    if args.mode == "offline":
        prompts = make_prompts(args)
        run_offline(args, prompts)
    else:
        run_online(args)


if __name__ == "__main__":
    main()
