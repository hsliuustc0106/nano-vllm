# Serving Benchmark Results

Date: 2026-06-06

Environment:
- Model: Qwen/Qwen3-0.6B local snapshot
- GPU: NVIDIA L20X, single GPU
- Python: 3.12.13 in `.venv312`
- vLLM: 0.22.1
- Benchmark harness: `vllm bench serve --backend openai --endpoint /v1/completions --dataset-name random`
- Nano-vLLM frontend: rebuilt `rust/nanovllm-serve/target/debug/nanovllm-serve`

## Online Serving, Warm Cache Rerun

| Preset | Server | Successful | Output tok/s | Mean TTFT ms | Mean TPOT ms | Mean ITL ms |
|:--|:--|--:|--:|--:|--:|--:|
| short-throughput, input 128, output 64, prompts 128, concurrency 64 | Nano-vLLM | 128 | 4847.77 | 638.64 | 3.20 | 40.37 |
| short-throughput, input 128, output 64, prompts 128, concurrency 64 | vLLM 0.22.1 | 128 | 9917.22 | 213.21 | 3.08 | 3.22 |
| long-throughput-8k-1k, input 8192, output 1024, prompts 16, concurrency 16 | Nano-vLLM | 16 | 1507.62 | 1730.38 | 8.92 | 140.31 |
| long-throughput-8k-1k, input 8192, output 1024, prompts 16, concurrency 16 | vLLM 0.22.1 | 16 | 2522.90 | 852.63 | 5.47 | 5.48 |
| low-latency-32k-2k, input 32768, output 2048, prompts 4, concurrency 1 | Nano-vLLM | 4 | 258.65 | 836.49 | 3.46 | 3.61 |
| low-latency-32k-2k, input 32768, output 2048, prompts 4, concurrency 1 | vLLM 0.22.1 | 4 | 309.66 | 461.36 | 3.01 | 3.03 |

Notes:
- Nano-vLLM used `--stream-token-flush-interval 16` for throughput presets and `1` for the low-latency preset.
- Throughput presets intentionally coalesce Nano streaming chunks, so Nano ITL is chunk interval, not per-token latency, for those rows.
- Low-latency uses flush interval `1`, so Nano ITL is directly comparable to vLLM.
- The first formal benchmark attempt against Nano failed because the old Rust binary rejected `ignore_eos`; rebuilding the current frontend fixed this.
- The initial vLLM short-throughput and 8K-throughput measurements were discarded because they included cold compilation/JIT work during the measured request window. The table above uses the second pass after vLLM loaded cached compiled graphs.

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
