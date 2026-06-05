import argparse
import os
import statistics
import subprocess
import sys
import time
from random import Random

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


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
    cmd = [
        args.vllm_bin,
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
    print("vllm_bench_cmd", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Compare Nano-vLLM offline batch and online HTTP serving speed.")
    parser.add_argument("--mode", choices=["offline", "online"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", default="Qwen3-0.6B")
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--num-prompts", type=int, default=32)
    parser.add_argument("--min-input-len", type=int, default=16)
    parser.add_argument("--max-input-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--stream", action="store_true", help="Accepted for compatibility; vLLM's OpenAI completions benchmark streams by default.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--vllm-bin", default="vllm")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    args = parser.parse_args()

    if args.mode == "offline":
        prompts = make_prompts(args)
        run_offline(args, prompts)
    else:
        run_online(args)


if __name__ == "__main__":
    main()
