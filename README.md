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
requires the `vllm` CLI to be available on `PATH`.

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
