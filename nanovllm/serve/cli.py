import argparse
import os
import shutil
import signal
import subprocess
import sys
import sysconfig
import time


def _add_serve_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--request-endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--event-endpoint", default="tcp://127.0.0.1:5558")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--frontend-binary", default=None)
    parser.add_argument("--trace-http", action="store_true", help="Enable per-request tracing in the Rust HTTP frontend.")
    parser.add_argument("--stream-token-flush-interval", type=int, default=16)
    parser.add_argument("--log-serving-stats-interval", type=float, default=0.0)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nanovllm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve_parser = subparsers.add_parser("serve", help="Run the online serving stack.")
    _add_serve_args(serve_parser)
    return parser


def _frontend_binary(explicit: str | None) -> str:
    if explicit:
        return explicit
    candidates = [
        os.path.join(sysconfig.get_path("scripts"), "nanovllm-serve"),
        os.path.join(os.path.dirname(sys.executable), "nanovllm-serve"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    binary = shutil.which("nanovllm-serve")
    if binary is None:
        raise RuntimeError("nanovllm-serve was not found on PATH; install the wheel or pass --frontend-binary")
    return binary


def _serve(args: argparse.Namespace) -> int:
    served_model = args.served_model_name or os.path.basename(os.path.normpath(args.model)) or args.model
    engine_cmd = [
        sys.executable,
        "-m",
        "nanovllm.serve.engine",
        "--model",
        args.model,
        "--request-endpoint",
        args.request_endpoint,
        "--event-endpoint",
        args.event_endpoint,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--stream-token-flush-interval",
        str(getattr(args, "stream_token_flush_interval", 16)),
        "--log-serving-stats-interval",
        str(getattr(args, "log_serving_stats_interval", 0.0)),
    ]
    if args.enforce_eager:
        engine_cmd.append("--enforce-eager")

    frontend_cmd = [
        _frontend_binary(args.frontend_binary),
        "--model",
        served_model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--request-endpoint",
        args.request_endpoint,
        "--event-endpoint",
        args.event_endpoint,
    ]
    if getattr(args, "trace_http", False):
        frontend_cmd.append("--trace-http")

    engine = None
    frontend = None
    shutting_down = False
    intentional_shutdown = False

    def shutdown(signum=None, frame=None):
        nonlocal shutting_down, intentional_shutdown
        if signum is not None:
            intentional_shutdown = True
        if shutting_down:
            return
        shutting_down = True
        for process in (frontend, engine):
            if process is not None and process.poll() is None:
                process.terminate()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    try:
        engine = subprocess.Popen(engine_cmd)
        frontend = subprocess.Popen(frontend_cmd)
        while True:
            engine_code = engine.poll()
            frontend_code = frontend.poll()
            if engine_code is not None:
                shutdown()
                frontend.wait()
                if intentional_shutdown:
                    return 0
                return engine_code
            if frontend_code is not None:
                shutdown()
                engine.wait()
                if intentional_shutdown:
                    return 0
                return frontend_code
            time.sleep(0.5)
    finally:
        shutdown()
        for process in (frontend, engine):
            if process is not None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    raise AssertionError(f"unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
