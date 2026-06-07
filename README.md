<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Online Serving

Nano-vLLM also includes an experimental serving stack. The user-facing command
starts both the Python engine process, which owns model execution and
scheduling, and the Rust HTTP frontend, which serves an OpenAI-compatible API.

Start the service:

```bash
nanovllm serve \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model-name Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000
```

Internally, this launches a Python engine process and the Rust HTTP frontend
with local IPC endpoints. Most users should only need the single `nanovllm
serve` command.

Request a completion:

```bash
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B","prompt":"Hello, Nano-vLLM.","max_tokens":32}'
```

The first serving endpoint supports `/v1/completions` and `/v1/models`.
Completions support `model`, `prompt`, `max_tokens`, `temperature`, `stream`,
`n`, `stop`, and `echo`. Streaming uses server-sent events and ends with
`data: [DONE]`.

To compare online serving with offline batch generation, run:

```bash
python tools/compare_serving_bench.py \
  --mode offline \
  --model ~/huggingface/Qwen3-0.6B/ \
  --num-prompts 64 \
  --max-tokens 32

python tools/compare_serving_bench.py \
  --mode online \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model Qwen3-0.6B \
  --base-url http://127.0.0.1:8000 \
  --num-prompts 64 \
  --concurrency 64 \
  --max-tokens 32
```

The online mode uses the formal `vllm bench serve` harness with
`--backend openai`, `--endpoint /v1/completions`, and the random dataset. It
uses the `vllm` module from the active Python environment when available, then
falls back to a `vllm` binary on `PATH`. Pass `--vllm-bin /path/to/vllm` to
force a specific executable.

Short online runs are sensitive to first-burst warmup effects. Use the default
`--num-warmups 0` to measure cold first-burst behavior, or pass
`--num-warmups N` to ask `vllm bench serve` to issue warmup requests before the
measured run. When comparing against upstream vLLM, launch vLLM with
`--generation-config vllm` or pass explicit sampling flags such as `--top-p`
and `--top-k` through this script so model `generation_config.json` defaults do
not silently change the sampling path.

For serving bottleneck work, pass `--log-serving-stats-interval N` to
`nanovllm serve` to print lightweight engine counters every `N` seconds,
including request count, prefill/decode step counts, average prefill batch
size, average decode batch size, and emitted event count.

Serving presets are included for repeatable comparisons:

```bash
# Short throughput run:
# input 128, output 64, 128 prompts, concurrency 64.
nanovllm serve \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model-name Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 1024 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 8192 \
  --stream-token-flush-interval 16

python tools/compare_serving_bench.py \
  --preset short-throughput \
  --mode online \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model Qwen3-0.6B \
  --base-url http://127.0.0.1:8000

# Warmed steady-state variant:
python tools/compare_serving_bench.py \
  --preset short-throughput \
  --mode online \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model Qwen3-0.6B \
  --base-url http://127.0.0.1:8000 \
  --num-warmups 16

# Throughput-oriented long-context run:
# input 8K, output 1K, 16 prompts, concurrency 16.
nanovllm serve \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model-name Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 32768 \
  --kvcache-block-size 32 \
  --stream-token-flush-interval 16

python tools/compare_serving_bench.py \
  --preset long-throughput-8k-1k \
  --mode online \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model Qwen3-0.6B \
  --base-url http://127.0.0.1:8000

# Low-latency long-context run:
# input 32K, output 2K, 4 prompts, concurrency 1.
nanovllm serve \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model-name Qwen3-0.6B \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 40960 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 32768 \
  --stream-token-flush-interval 1

python tools/compare_serving_bench.py \
  --preset low-latency-32k-2k \
  --mode online \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model Qwen3-0.6B \
  --base-url http://127.0.0.1:8000
```

To capture a clean Nsight Systems profile of only the measured online serving
window, excluding model load and prewarm, run:

```bash
python tools/nsys_online_profile.py \
  --model ~/huggingface/Qwen3-0.6B/ \
  --served-model Qwen3-0.6B \
  --output /tmp/nanovllm-online \
  --port 8780 \
  --endpoint-base 6357 \
  --prewarm \
  --num-prompts 64 \
  --concurrency 64 \
  --max-tokens 32

python tools/analyze_nsys.py /tmp/nanovllm-online.nsys-rep
```

Recent local online comparison results are recorded in
`tools/serving_benchmark_results.md`.

On one NVIDIA L20X with Qwen3-0.6B, 64 concurrent prompts, and 32 generated
tokens per prompt, local smoke runs showed the online path preserving most of
the offline engine throughput once enough concurrent requests were in flight.
Expect some variance on short runs; use the comparison script on the target
machine for a stable baseline.

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
