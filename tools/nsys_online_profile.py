import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request


def wait_for_port(host: str, port: int, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sock = socket.socket()
        sock.settimeout(0.25)
        try:
            sock.connect((host, port))
            return
        except OSError:
            time.sleep(0.25)
        finally:
            sock.close()
    raise TimeoutError(f"timed out waiting for {host}:{port}")


def post_profile_control(base_url: str, action: str, timeout_s: float):
    request = urllib.request.Request(
        f"{base_url}/_debug/profile/{action}",
        data=b"",
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        response.read()


def run(args):
    output = os.path.abspath(args.output)
    endpoint_base = args.endpoint_base
    request_endpoint = f"tcp://127.0.0.1:{endpoint_base}"
    event_endpoint = f"tcp://127.0.0.1:{endpoint_base + 1}"
    base_url = f"http://{args.host}:{args.port}"

    serve_cmd = [
        sys.executable,
        "-m",
        "nanovllm.serve.cli",
        "serve",
        "--model",
        os.path.expanduser(args.model),
        "--served-model-name",
        args.served_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--request-endpoint",
        request_endpoint,
        "--event-endpoint",
        event_endpoint,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.enforce_eager:
        serve_cmd.append("--enforce-eager")
    if args.frontend_binary:
        serve_cmd.extend(["--frontend-binary", args.frontend_binary])

    nsys_cmd = [
        "nsys",
        "profile",
        "--trace=cuda,nvtx,osrt",
        "--cuda-memory-usage=true",
        "--cuda-graph-trace=graph",
        "--cuda-trace-scope=process-tree",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--force-overwrite=true",
        "--output",
        output,
    ]
    if not args.include_startup:
        nsys_cmd.extend([
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
        ])
    nsys_cmd += serve_cmd

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    print("starting:", " ".join(nsys_cmd), flush=True)
    proc = subprocess.Popen(nsys_cmd, env=env)
    try:
        wait_for_port(args.host, args.port, args.startup_timeout)
        if args.prewarm:
            prewarm_cmd = [
                sys.executable,
                "tools/compare_serving_bench.py",
                "--mode",
                "online",
                "--model",
                os.path.expanduser(args.model),
                "--served-model",
                args.served_model,
                "--base-url",
                base_url,
                "--num-prompts",
                str(args.prewarm_prompts),
                "--max-tokens",
                str(args.prewarm_tokens),
                "--concurrency",
                str(args.prewarm_prompts),
                "--request-rate",
                args.request_rate,
                "--timeout",
                str(args.timeout),
            ]
            if args.vllm_bin:
                prewarm_cmd.extend(["--vllm-bin", args.vllm_bin])
            print("prewarm:", " ".join(prewarm_cmd), flush=True)
            subprocess.run(prewarm_cmd, check=True)

        bench_cmd = [
            sys.executable,
            "tools/compare_serving_bench.py",
            "--mode",
            "online",
            "--model",
            os.path.expanduser(args.model),
            "--served-model",
            args.served_model,
            "--base-url",
            base_url,
            "--num-prompts",
            str(args.num_prompts),
            "--min-input-len",
            str(args.min_input_len),
            "--max-input-len",
            str(args.max_input_len),
            "--max-tokens",
            str(args.max_tokens),
            "--concurrency",
            str(args.concurrency),
            "--request-rate",
            args.request_rate,
            "--timeout",
            str(args.timeout),
        ]
        if args.vllm_bin:
            bench_cmd.extend(["--vllm-bin", args.vllm_bin])
        print("benchmark:", " ".join(bench_cmd), flush=True)
        if not args.include_startup:
            print("profile_start", flush=True)
            post_profile_control(base_url, "start", args.timeout)
            time.sleep(args.profile_start_delay)
        try:
            subprocess.run(bench_cmd, check=True)
        finally:
            if not args.include_startup:
                print("profile_stop", flush=True)
                post_profile_control(base_url, "stop", args.timeout)
                time.sleep(args.profile_stop_delay)
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=args.shutdown_timeout)

    report = output if output.endswith(".nsys-rep") else output + ".nsys-rep"
    print(f"nsys_report {report}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Capture an Nsight Systems profile for Nano-vLLM online serving.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", default="Qwen3-0.6B")
    parser.add_argument("--output", default="/tmp/nanovllm-online")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--endpoint-base", type=int, default=6357)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--startup-timeout", type=float, default=120)
    parser.add_argument("--shutdown-timeout", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--prewarm", action="store_true")
    parser.add_argument("--prewarm-prompts", type=int, default=8)
    parser.add_argument("--prewarm-tokens", type=int, default=8)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--min-input-len", type=int, default=16)
    parser.add_argument("--max-input-len", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--vllm-bin", default=None)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--frontend-binary")
    parser.add_argument("--include-startup", action="store_true", help="Profile model load and prewarm instead of only the measured benchmark window.")
    parser.add_argument("--profile-start-delay", type=float, default=0.5)
    parser.add_argument("--profile-stop-delay", type=float, default=0.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
