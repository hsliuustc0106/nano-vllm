# Serving Benchmark Results

Date: 2026-06-07

Environment:
- Model: Qwen/Qwen3-0.6B local snapshot
- GPU: NVIDIA L20X, single GPU
- Python: 3.12.13 in `.venv312`
- vLLM: 0.22.1
- Benchmark harness: `vllm bench serve --backend openai --endpoint /v1/completions --dataset-name random`
- Nano-vLLM frontend: rebuilt `rust/nanovllm-serve/target/debug/nanovllm-serve`

## Latest Online Serving Comparison (formal `vllm bench serve`, cold first burst, all runs: 0 failures)

| Preset | Server | Successful | Duration (s) | Output tok/s | Mean TTFT ms | Mean TPOT ms | Mean ITL ms |
|:--|:--|--:|--:|--:|--:|--:|--:|
| short-throughput, input 128, output 64, prompts 128, concurrency 64 | Nano-vLLM | 128 | 1.54 | 5324.41 | 590.15 | 2.75 | 34.69 |
| short-throughput, input 128, output 64, prompts 128, concurrency 64 | vLLM 0.22.1 | 128 | 0.70 | 11688.98 | 151.18 | 3.05 | 3.20 |
| long-throughput-8k-1k, input 8192, output 1024, prompts 16, concurrency 16 | Nano-vLLM | 16 | 10.66 | 1537.41 | 1595.60 | 8.85 | 139.28 |
| long-throughput-8k-1k, input 8192, output 1024, prompts 16, concurrency 16 | vLLM 0.22.1 | 16 | 6.64 | 2466.68 | 954.72 | 5.50 | 6.13 |
| low-latency-32k-2k, input 32768, output 2048, prompts 4, concurrency 1 | Nano-vLLM | 4 | 29.66 | 276.21 | 803.21 | 3.23 | 3.23 |
| low-latency-32k-2k, input 32768, output 2048, prompts 4, concurrency 1 | vLLM 0.22.1 | 4 | 25.22 | 324.81 | 442.66 | 2.86 | 2.87 |

These rows use the first measured benchmark burst after the server is ready,
with no explicit `vllm bench serve --num-warmups` requests.

## High-Concurrency 4K/1K Follow-up

The run below uses the same formal `vllm bench serve` OpenAI completions
benchmark, seed `0`, random token prompts, no explicit warmup requests, and
64 prompts at concurrency 64. Both servers used the same local
`Qwen/Qwen3-0.6B` snapshot after fetching the missing `model.safetensors`;
an earlier Nano-only smoke run against the config/tokenizer-only cache was
discarded as invalid for performance comparison.

Server launch settings:
- Nano-vLLM: `--max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 32768 --stream-token-flush-interval 16`
- vLLM 0.22.1: `--max-model-len 32768 --max-num-seqs 64 --max-num-batched-tokens 32768 --generation-config vllm`

| Preset | Server | Successful | Duration (s) | Output tok/s | Mean TTFT ms | Mean TPOT ms | Mean ITL ms |
|:--|:--|--:|--:|--:|--:|--:|--:|
| input 4096, output 1024, prompts 64, concurrency 64 | Nano-vLLM | 64 | 12.90 | 5078.45 | 1284.65 | 11.34 | 178.49 |
| input 4096, output 1024, prompts 64, concurrency 64 | vLLM 0.22.1 | 64 | 10.94 | 5990.11 | 1190.37 | 9.39 | 9.45 |

For this high-concurrency 4K/1K shape, upstream vLLM is `1.18x` faster on
output token throughput. Raw logs:
`/tmp/nano_qwen3_0_6b_4k_1k_c64_weighted.log` and
`/tmp/vllm_qwen3_0_6b_4k_1k_c64_weighted.log`.

## Short-Throughput Warmup/Profile Follow-up

Short online serving is sensitive to first-burst warmup effects. A small
serving warmup before the measured short-throughput run reproduced the
previously surprising high Nano-vLLM result:

| Run | Server | Successful | Duration (s) | Output tok/s | Mean TTFT ms | Mean TPOT ms | Mean ITL ms |
|:--|:--|--:|--:|--:|--:|--:|--:|
| normal warmed rerun | Nano-vLLM | 128 | 0.59 | 13791.14 | 108.83 | 2.84 | 35.74 |
| Nsight Systems profiled warmed run | Nano-vLLM | 128 | 0.63 | 13018.16 | 111.96 | 3.13 | 39.46 |
| Nsight Systems profiled warmed run | vLLM 0.22.1 | 128 | 0.76 | 10832.31 | 137.51 | 3.72 | 3.79 |

The earlier refreshed table that reported Nano-vLLM short-throughput around
14k tok/s was therefore a warmed steady-state style result, while the cold
first-burst table above measures startup-adjacent serving behavior. Future
comparisons should report both baselines instead of mixing them.

VeloQ analysis of the warmed Nsight traces (`/tmp/nano_short_profile.nsys-rep`
and `/tmp/vllm_short_profile.nsys-rep`) showed:

| Metric | Nano-vLLM | vLLM 0.22.1 | Interpretation |
|:--|--:|--:|:--|
| Trace span | 600.0 ms | 719.4 ms | measured window |
| GPU kernel time | 59.7 ms | 121.5 ms | Nano is not slower on device kernels in the warmed short run |
| CUDA runtime/API time | 268.4 ms | 125.0 ms | Nano has more host-side CUDA overhead |
| Top 20 GPU idle gap total | 112.0 ms | 61.5 ms | Nano has larger host-side bubbles |
| CUDA graph replays | 126 | 190 | both paths use CUDA graphs |
| Device memcpy time | 1.33 ms | 1.53 ms | copies are not bandwidth-bound |

Nano's largest runtime bucket was `cudaMemcpyAsync` host/API time
(208.6 ms), while actual device memcpy work was only 1.33 ms. That points to
host-side queueing/synchronization around small transfers rather than device
copy bandwidth as the next short-path optimization target.

## Historical Online Serving Results

The historical runs below preserve earlier instrumentation/probe settings and tuning checkpoints while the section above remains the current comparison baseline.

### Notes
- Earlier warm-cache tables in this file are historical optimization probes kept for tuning context.
- A historical front-end caveat still applies: throughput presets used `--stream-token-flush-interval 16` in Nano while low-latency used `1`.


## Short-Throughput Optimization Probe

After changing the scheduler to fill available running sequence slots before forcing a decode step, the short-throughput Nano-vLLM result was:

| Preset | Server | Successful | Output tok/s | Mean TTFT ms | Mean TPOT ms | Mean ITL ms | Engine stats |
|:--|:--|--:|--:|--:|--:|--:|:--|
| short-throughput, input 128, output 64, prompts 128, concurrency 64 | Nano-vLLM | 128 | 4689.56 | 675.40 | 3.05 | 38.37 | 6 prefill steps, 126 decode steps, avg decode batch 64.00 |

This is a modest improvement over the instrumented pre-change probe, which measured 4554.77 output tok/s, 682.41 ms mean TTFT, 3.29 ms mean TPOT, and avg decode batch 62.51 with the same command and stats logging. It remains below the earlier no-stats Nano warm-cache result of 4847.77 tok/s, so the larger short-request gap still needs deeper profiling around TTFT/prefill and frontend/event overhead.

After adding batched tokenization for drained simple string prompts, the same short-throughput run measured 4985.09 output tok/s, 629.96 ms mean TTFT, 2.97 ms mean TPOT, and 37.46 ms mean ITL. Engine timing counters for that run were:

| Metric | Value |
|:--|--:|
| Requests | 128 |
| Prefill steps | 6 |
| Decode steps | 126 |
| Avg prefill batch | 2730.67 tokens |
| Avg decode batch | 64.00 seqs |
| Request normalize | 14.33 ms |
| Request add | 1.81 ms |
| Schedule | 3.92 ms |
| Model runner | 1536.03 ms |
| Postprocess | 5.82 ms |
| Token handling | 20.10 ms |
| Event emit | 5.60 ms |

The batched tokenizer change reduced request normalization from 62.17 ms in the previous timed run to 14.33 ms. The remaining short-request gap is dominated by model runner time rather than Rust frontend event emission or Python scheduler overhead.

A clean counter run using `POST /_debug/profile/reset` before the benchmark measured 4694.92 output tok/s, 677.42 ms mean TTFT, 2.99 ms mean TPOT, and 37.69 ms mean ITL. The post-reset engine split was:

| Metric | Value |
|:--|--:|
| Requests | 128 |
| Prefill steps | 6 |
| Decode steps | 126 |
| Avg prefill batch | 2730.67 tokens |
| Avg decode batch | 64.00 seqs |
| Request normalize | 14.14 ms |
| Request add | 2.83 ms |
| Schedule | 4.04 ms |
| Model runner | 1639.20 ms |
| Postprocess | 5.62 ms |
| Token handling | 21.40 ms |
| Event emit | 6.71 ms |
| Runner prepare prefill | 7.21 ms |
| Runner prepare decode | 28.75 ms |
| Runner prepare sample | 3.23 ms |
| Runner model prefill | 1039.12 ms |
| Runner model decode | 19.83 ms |
| Runner sampler | 321.54 ms |
| Runner tolist/sync | 216.54 ms |
| Decode graph calls | 126 |
| Decode eager calls | 0 |

This confirms CUDA graph coverage for short decode is complete. A Gumbel-max sampler experiment was rejected because it regressed the same short run to 2749.67 output tok/s and increased runner sampler time to 1400.33 ms.

After prewarming sampler compilation for CUDA graph batch sizes during model runner startup, the same short run measured 4855.84 output tok/s, 640.87 ms mean TTFT, 3.13 ms mean TPOT, and 39.42 ms mean ITL. The post-reset runner sampler time dropped from 321.54 ms to 274.67 ms, while decode graph coverage remained complete at 126 graph calls and 0 eager decode calls. End-to-end throughput stayed within the normal short-run variance band, but the prewarm removes sampler first-use compile overhead from the measured burst.

A short-prefill warmup experiment for a 64 x 128 prefill batch was rejected. It measured 4687.51 output tok/s and increased runner model prefill time to 1102.42 ms, so it did not improve the short-serving path.

An explicit sampler synchronization probe was also rejected for production because it slowed the short run to 4349.15 output tok/s. It clarified the timing split: `.tolist()` conversion itself was only 6.43 ms, while the explicit post-sampler CUDA synchronization was 217.35 ms. The previous `runner_tolist_ms` bucket was therefore mostly pending GPU sampling work, not Python list conversion.

Detailed model-runner timing is now enabled only when `--log-serving-stats-interval` is set, so normal serving does not pay the profiling timer overhead. A no-stats short run after making runner timing opt-in measured 4550.50 output tok/s, which was within the observed run-to-run variance band and did not change the bottleneck diagnosis.

Increasing the short-throughput serving configuration from `--max-num-batched-tokens 4096` to `8192` produced a clear improvement by admitting a full 64 x 128-token prefill batch. The no-stats short run measured 5258.63 output tok/s, 597.88 ms mean TTFT, 2.74 ms mean TPOT, and 34.48 ms mean ITL. The `short-throughput` preset now uses `max_num_batched_tokens=8192`.

Increasing short-throughput `--max-num-seqs` to 128 with `--max-num-batched-tokens 16384` was rejected. It measured 5206.25 output tok/s, 608.97 ms mean TTFT, 2.73 ms mean TPOT, and 34.45 ms mean ITL, roughly matching but not improving on the 64-sequence, 8192-token configuration while increasing graph capture and runtime resource footprint.
