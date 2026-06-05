# Sampler Benchmark

This directory contains the benchmark entry used to reproduce the sampler latency results.

## Figure Mapping
- Operator: HPC-Ops Sampler
- Providers:
  - `HPC-Ops`
  - `vLLM/PyTorch`
  - `FlashInfer`
- Timing unit: microseconds per sampler call
- Timing modes:
  - `--timing nsys`: release-style timing using `nsys`, NVTX `step`, eager sampler calls, and median latency. This keeps the profiler/post-processing style aligned with FusedMoE while avoiding CUDA Graph capture paths that are not safe for sampler APIs in this environment.
  - `--timing event`: quick CUDA event timing with eager sampler calls by default. `--graph` is available only for experiments because some sampler paths are not CUDA-graph-capture safe.
- Default config: vocab size `120832`, BF16 logits, batch sizes `1,8,16,32,64,128,256,512`.

## Scenario Names

- `temperature`: `Temperature Sampling`, temperature-only fast path.
- `full`: `Full Sampling`, repetition penalty + temperature + topk/topp + sampling.

## Reproduction Commands

Full sweep with the FusedMoE-aligned `nsys` timing path:

```bash
python3 benchmark_sampler.py \
  --timing nsys \
  --output-dir sampler_nsys \
  --csv sampler_latency.csv \
  --jsonl sampler_latency.jsonl
```

Fast smoke test with CUDA event timing:

```bash
python3 benchmark_sampler.py \
  --timing event \
  --scenes temperature full \
  --batches 1 8 \
  --providers hpc torch \
  --warmup 1 \
  --iters 3
```

If FlashInfer is unavailable or its API is incompatible with the local environment, the benchmark records the provider error and continues with the remaining providers.
