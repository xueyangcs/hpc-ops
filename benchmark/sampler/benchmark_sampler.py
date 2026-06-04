#!/usr/bin/env python3
# Copyright (C) 2026 Tencent.

"""Sampler benchmark.

Scenarios:
    - temperature: temperature-only sampling fast path.
    - full: repetition penalty + temperature + topk/topp + sampling.

operator: nsys profile, NVTX ``step`` ranges, and median latency. Sampler kernels
are measured in eager mode because the temperature fast path and baseline
sampling APIs are not CUDA-graph-capture safe in this environment.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean, median

import torch


sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../../build/lib.*/"))[0]))

import hpc  # noqa: E402
from hpc.sampler import SoftmaxPolicy  # noqa: E402


VOCAB_SIZE = 120832
BATCHES = [1, 8, 16, 32, 64, 128, 256, 512]
SCENES = ["temperature", "full"]
PROVIDERS = ["hpc", "torch", "flashinfer"]
DISPLAY = {
    "hpc": "HPC-Ops",
    "torch": "vLLM/PyTorch",
    "flashinfer": "FlashInfer",
}


def make_gumbel(shape, device="cuda") -> torch.Tensor:
    u = torch.rand(shape, dtype=torch.float32, device=device).clamp_min_(1e-20)
    return -(-u.log()).log()


def build_inputs(batch: int, dtype: torch.dtype, seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    logits = torch.randn(batch, VOCAB_SIZE, dtype=dtype, device="cuda")
    gumbel = make_gumbel((batch, VOCAB_SIZE))
    temperature = torch.full((batch,), 0.7, dtype=torch.float32, device="cuda")
    topk = torch.full((batch,), 20, dtype=torch.int32, device="cuda")
    topp = torch.full((batch,), 0.9, dtype=torch.float32, device="cuda")
    row_bytes = (VOCAB_SIZE + 7) // 8
    penalty_mask = torch.zeros((batch, row_bytes), dtype=torch.uint8, device="cuda")
    slot_id = torch.arange(batch, dtype=torch.int32, device="cuda")
    return {
        "logits": logits,
        "gumbel": gumbel,
        "temperature": temperature,
        "topk": topk,
        "topp": topp,
        "penalty_mask": penalty_mask,
        "slot_id": slot_id,
    }


def setup_call(provider: str, scene: str, batch: int, dtype: torch.dtype, seed: int):
    data = build_inputs(batch, dtype, seed)
    logits = data["logits"]
    gumbel = data["gumbel"]
    temperature = data["temperature"]

    if provider == "hpc":
        if scene == "temperature":
            return lambda: hpc.fused_sampler(logits, temperature=temperature, gumbel_noise=gumbel)
        if scene == "full":
            return lambda: hpc.fused_sampler(
                logits,
                penalty_mask=data["penalty_mask"],
                slot_id=data["slot_id"],
                repetition_penalty=1.05,
                temperature=temperature,
                softmax_policy=SoftmaxPolicy.AFTER_TOPK,
                topk=data["topk"],
                topp=data["topp"],
                max_topk=32,
                gumbel_noise=gumbel,
            )
        raise ValueError(scene)

    if provider == "torch":
        if scene == "temperature":
            return lambda: (logits.float() / temperature[:, None] + gumbel).argmax(dim=-1)
        if scene == "full":
            def run_torch_full():
                work = logits.float() / temperature[:, None]
                vals, idx = torch.topk(work, k=20, dim=-1)
                probs = torch.softmax(vals, dim=-1)
                keep = probs.cumsum(dim=-1) <= 0.9
                keep[:, 0] = True
                scores = torch.where(keep, probs.log(), torch.full_like(probs, -float("inf")))
                scores = scores + torch.gather(gumbel, 1, idx)
                return idx.gather(1, scores.argmax(dim=-1, keepdim=True))
            return run_torch_full
        raise ValueError(scene)

    if provider == "flashinfer":
        import flashinfer.sampling as sampling

        if scene == "temperature":
            return lambda: sampling.sampling_from_logits(
                logits.float() / temperature[:, None],
                deterministic=True,
                seed=seed,
            )
        if scene == "full":
            return lambda: sampling.top_k_top_p_sampling_from_logits(
                logits.float() / temperature[:, None],
                top_k=20,
                top_p=0.9,
                filter_apply_order="top_k_first",
                deterministic=True,
                seed=seed,
            )
        raise ValueError(scene)

    raise ValueError(provider)


def run_nsys_steps(call_fn, *, warmup: int, n_timed: int) -> None:
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(n_timed):
        torch.cuda.nvtx.range_push("step")
        call_fn()
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    torch.cuda.cudart().cudaProfilerStop()


def bench_event(call_fn, warmup: int, iters: int, use_graph: bool) -> tuple[float, float, int]:
    for _ in range(warmup):
        call_fn()
    torch.cuda.synchronize()

    if use_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call_fn()
        for _ in range(warmup):
            graph.replay()
        torch.cuda.synchronize()
        call_fn = graph.replay

    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in events:
        start.record()
        call_fn()
        end.record()
    torch.cuda.synchronize()
    vals = sorted(start.elapsed_time(end) * 1000.0 for start, end in events)
    return float(median(vals)), float(mean(vals)), len(vals)


def extract_nvtx_us(report_prefix: Path) -> list[float]:
    cmd = [
        "nsys",
        "stats",
        "--report",
        "nvtx_gpu_proj_trace",
        "--force-export=true",
        "-q",
        "-f",
        "json",
        str(report_prefix) + ".nsys-rep",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        raw = json.loads(out.decode())
        data = raw[0]["data"] if isinstance(raw, list) and raw and "data" in raw[0] else raw
        samples = []
        for entry in data:
            name = entry.get("Name", "").strip().strip('"')
            if name in ("step", ":step"):
                samples.append(float(entry["Projected Duration (ns)"]) / 1000.0)
        return samples[2:]
    except Exception:
        return []


def run_nsys_profile(args, scene: str, batch: int, provider: str, out_dir: Path) -> tuple[float | None, float | None, int, str | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_prefix = out_dir / f"{scene}_bs{batch}_{provider}"
    report_file = str(report_prefix) + ".nsys-rep"
    if os.path.exists(report_file):
        os.remove(report_file)

    cmd = [
        "nsys",
        "profile",
        "-f",
        "true",
        "-o",
        str(report_prefix),
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop-shutdown",
        "--cuda-graph-trace=node",
        "-t",
        "cuda,nvtx",
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--scene",
        scene,
        "--batch",
        str(batch),
        "--provider",
        provider,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--seed",
        str(args.seed),
    ]
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=args.nsys_timeout,
            env=os.environ.copy(),
        )
    except subprocess.TimeoutExpired:
        return None, None, 0, "nsys profile timeout"

    if not os.path.exists(report_file) and proc.returncode != 0:
        err = proc.stderr.decode(errors="replace").strip().splitlines()
        return None, None, 0, err[-1] if err else "nsys profile failed"

    vals = extract_nvtx_us(report_prefix)
    if not vals:
        return None, None, 0, "no NVTX step samples"
    return float(median(vals)), float(mean(vals)), len(vals), None


def dtype_from_name(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16}[name]


def run_worker(args) -> None:
    call_fn = setup_call(args.provider, args.scene, args.batch, dtype_from_name(args.dtype), args.seed)
    run_nsys_steps(call_fn, warmup=args.warmup, n_timed=args.iters + 2)


def print_table(rows: list[dict]) -> None:
    scenes = {"temperature": "Temperature Sampling", "full": "Full Sampling"}
    print("")
    print("=" * 110)
    print("Sampler latency | us per call (lower is better)")
    print("-" * 110)
    print(f"{'scene':>22} | {'batch':>5} | {'provider':>12} | {'median_us':>10} | {'mean_us':>10} | {'samples':>7} | {'error':>18}")
    print("-" * 110)
    for r in rows:
        med = f"{r['median_us']:.2f}" if isinstance(r.get("median_us"), (int, float)) else "ERR"
        avg = f"{r['mean_us']:.2f}" if isinstance(r.get("mean_us"), (int, float)) else "ERR"
        err = r.get("error") or ""
        print(f"{scenes[r['scene']]:>22} | {r['batch']:5d} | {DISPLAY[r['provider']]:>12} | {med:>10} | {avg:>10} | {r['samples']:7d} | {err[:18]:>18}")
    print("=" * 110)


def write_csv(path: str, rows: list[dict]) -> None:
    if not path or not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: str, rows: list[dict]) -> None:
    if not path:
        return
    with Path(path).open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark HPC-Ops Sampler for figure 11.")
    p.add_argument("--scenes", nargs="+", default=SCENES, choices=SCENES)
    p.add_argument("--batches", type=int, nargs="+", default=BATCHES)
    p.add_argument("--providers", nargs="+", default=PROVIDERS, choices=PROVIDERS)
    p.add_argument("--timing", choices=["nsys", "event"], default="nsys")
    p.add_argument(
        "--graph",
        dest="graph",
        action="store_true",
        help="try CUDA Graph replay for --timing event; sampler paths are measured eager by default.",
    )
    p.add_argument("--no-graph", dest="graph", action="store_false", help=argparse.SUPPRESS)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=52)
    p.add_argument("--seed", type=int, default=10086)
    p.add_argument("--output-dir", default="")
    p.add_argument("--tag", default="")
    p.add_argument("--nsys-timeout", type=int, default=300)
    p.add_argument("--csv", default="")
    p.add_argument("--jsonl", default="")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--scene", choices=SCENES, default="temperature", help=argparse.SUPPRESS)
    p.add_argument("--batch", type=int, default=1, help=argparse.SUPPRESS)
    p.add_argument("--provider", choices=PROVIDERS, default="hpc", help=argparse.SUPPRESS)
    p.set_defaults(graph=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return

    rows = []
    tag = args.tag or f"sampler_{int(time.time())}"
    out_dir = Path(args.output_dir) / tag if args.output_dir else Path(__file__).resolve().parent / "log" / tag
    for scene in args.scenes:
        for batch in args.batches:
            for provider in args.providers:
                if args.timing == "nsys":
                    median_us, mean_us, n, err = run_nsys_profile(args, scene, batch, provider, out_dir)
                else:
                    try:
                        call_fn = setup_call(provider, scene, batch, dtype_from_name(args.dtype), args.seed)
                        median_us, mean_us, n = bench_event(call_fn, args.warmup, args.iters, args.graph)
                        err = None
                    except Exception as exc:
                        median_us, mean_us, n, err = None, None, 0, repr(exc)
                rows.append(
                    {
                        "scene": scene,
                        "batch": batch,
                        "provider": provider,
                        "median_us": median_us,
                        "mean_us": mean_us,
                        "samples": n,
                        "timing": "nsys_eager_nvtx_median" if args.timing == "nsys" else ("event_graph_median" if args.graph else "event_eager_median"),
                        "dtype": args.dtype,
                        "vocab_size": VOCAB_SIZE,
                        "error": err,
                    }
                )
    print_table(rows)
    write_csv(args.csv, rows)
    write_jsonl(args.jsonl, rows)
    if args.timing == "nsys":
        print(f"nsys reports: {out_dir}")


if __name__ == "__main__":
    main()
