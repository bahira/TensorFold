#!/usr/bin/env python3
"""Measure the GPU engine's code on CPU torch: the extrapolation rung for bench-cuda.

``tensorfold bench-cuda`` needs an NVIDIA GPU. This runs the same SuperSpear champion
formulas (``torch_ops``: the Triton fallback) and the same tiny GPU models, sampler and
n-gram drafts (``engine.cuda_spear``) on a CPU box, in the same JSON schema, so a CPU-only
instance can measure the code a GPU would run and extrapolate the GPU numbers. On CPU the
SwiGLU "fused" path is two eager ops like the baseline (the fuse is the Triton kernel):
``speedup_exact`` here is therefore the no-fuse floor, not the GPU fuse.

  python tools/bench_spear_torch_cpu.py
  python tools/bench_spear_torch_cpu.py --tokens 32 --output docs/recipes/spear-torch-cpu-results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any, Callable

DEFAULT_OUT = "docs/recipes/spear-torch-cpu-results.json"


def _median_s(fn: Callable[[], None], *, warmup: int = 3, reps: int = 9) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _host() -> dict[str, Any]:
    info: dict[str, Any] = {"python": platform.python_version(), "machine": platform.machine()}
    try:
        import torch

        info.update(torch=torch.__version__, cuda=torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        info["torch"] = None
    try:
        import numpy as np

        a = np.empty(16 << 20, dtype=np.uint8)
        b = np.empty_like(a)
        dt = _median_s(lambda: a.__setitem__(slice(None), b), reps=5)
        info["dram_bw_gibs"] = round(a.nbytes / dt / 1e9, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


def bench_activations(n: int = 1 << 22) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    from tensorfold.kernels.spear.v1 import cuda as spear_cuda

    rows = []
    for dtype in (torch.float16, torch.float32):
        x = torch.randn(n, dtype=dtype)
        specs = [
            ("gelu", lambda: F.gelu(x), "gelu_alg", "gelu_fast"),
            ("silu", lambda: F.silu(x), "silu_alg", "silu_fast"),
            ("sigmoid", lambda: torch.sigmoid(x), None, "sigmoid_fast"),
        ]
        for name, ref, alg, fast in specs:
            ns_ref = _median_s(ref) * 1e9 / n
            row: dict[str, Any] = {"op": name, "n": n, "ns_elem_torch": ns_ref,
                                   "dtype": "fp16" if dtype == torch.float16 else "fp32"}
            if alg:
                ns_alg = _median_s(lambda op=alg: spear_cuda.act(op, x)) * 1e9 / n
                mse = float(((ref() - spear_cuda.act(alg, x)).float().pow(2).mean()).item())
                row.update(ns_elem_alg=ns_alg, speedup_alg=ns_ref / ns_alg, mse_alg=mse)
            ns_fast = _median_s(lambda op=fast: spear_cuda.act(op, x)) * 1e9 / n
            mse_f = float(((ref() - spear_cuda.act(fast, x)).float().pow(2).mean()).item())
            row.update(ns_elem_fast=ns_fast, speedup_fast=ns_ref / ns_fast, mse_fast=mse_f)
            rows.append(row)
            print(json.dumps({k: row[k] for k in row if k in ("dtype", "op", "speedup_alg", "speedup_fast",
                                                              "ns_elem_torch", "ns_elem_fast")}), flush=True)
    return rows


def bench_swiglu() -> list[dict[str, Any]]:
    import torch

    from tensorfold.kernels.spear.v1 import cuda as spear_cuda

    rows = []
    for m, n in ((1, 2048), (1, 4096), (1, 11008), (128, 4096)):
        g = torch.randn(m, n, dtype=torch.float16)
        u = torch.randn(m, n, dtype=torch.float16)
        ns_torch = _median_s(lambda: spear_cuda.torch_swiglu(g, u)) * 1e6
        ns_exact = _median_s(lambda: spear_cuda.swiglu(g, u, silu="silu_exact")) * 1e6
        ns_alg = _median_s(lambda: spear_cuda.swiglu(g, u, silu="silu_alg")) * 1e6
        ns_fast = _median_s(lambda: spear_cuda.swiglu(g, u, silu="silu_fast")) * 1e6
        row = {
            "M": m, "N": n, "us_torch": ns_torch, "us_fused_exact": ns_exact,
            "us_fused_alg": ns_alg, "us_fused_fast": ns_fast,
            "speedup_exact": ns_torch / ns_exact, "speedup_alg": ns_torch / ns_alg,
            "speedup_fast": ns_torch / ns_fast,
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("M", "N", "speedup_exact", "speedup_alg", "speedup_fast")}), flush=True)
    return rows


def _params(model: Any) -> tuple[int, int]:
    """(matmul weights, all elements) the model holds: the bytes one decode token streams."""

    matmul = allp = 0
    names = ("wte", "wpe", "c_attn", "c_proj", "c_fc", "c_down", "Wq", "Wk", "Wv", "Wo", "Wgate", "Wup", "Wdown")
    for b in model.blocks:
        for name, t in b.items():
            allp += t.numel()
            if name in names:
                matmul += t.numel()
    allp += model.wte.numel()
    return matmul, allp


def bench_tiny(n_new: int) -> list[dict[str, Any]]:
    import torch

    from tensorfold.drafters.draft_ngram import SessionNGram
    from tensorfold.engine.cuda_spear import generate, tiny_gpu
    from tensorfold.engine.exact_sampling import Sampling, seed_for

    prompt = list(range(16, 48))
    rows = []
    configs = [
        ("swiglu-128", dict(kind="swiglu", n_layer=2, n_embd=128, n_head=4, n_inner=256)),
        ("swiglu-768x4", dict(kind="swiglu", n_layer=4, n_embd=768, n_head=12, n_inner=3072, vocab=512, seq=128)),
        ("gpt2-768x4", dict(kind="gpt2", n_layer=4, n_embd=768, n_head=12, n_inner=3072, vocab=512, seq=128)),
    ]
    for name, kw in configs:
        for act, drafts, label in (
            ("exact", 0, "fp16-torch"),
            ("alg", 0, "fp16-spear-alg"),
            ("fast", 0, "fp16-spear-fast"),
            ("alg", 4, "fp16-spear-alg-ngram"),
        ):
            model = tiny_gpu(act=act, **kw)
            ngram = SessionNGram(vocab=model.cfg.vocab_size) if drafts else None
            sampling = Sampling(seed=seed_for(prompt), temperature=0.0)
            generate(model, prompt, 2, sampling=sampling, drafts=0)
            runs = []
            last = None
            for _ in range(3):
                t0 = time.perf_counter()
                last = generate(model, prompt, n_new, sampling=sampling, drafts=drafts, ngram=ngram)
                dt = time.perf_counter() - t0
                runs.append(last["tokens"] / dt if dt else 0.0)
            matmul, allp = _params(model)
            row = {
                "model": name, "label": label, "act": act, "drafts": drafts,
                "tok_s": statistics.median(runs), "tok_s_all": [round(x, 2) for x in runs],
                "accepted": last["accepted"], "proposed": last["proposed"],
                "accept_rate": (last["accepted"] / last["proposed"]) if last["proposed"] else None,
                "matmul_params": matmul, "all_params": allp,
                "bytes_token_fp32": matmul * 4, "bytes_token_fp16": matmul * 2,
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("model", "label", "tok_s", "accept_rate")}), flush=True)
    model = tiny_gpu(kind="swiglu", act="alg")
    a = generate(model, prompt, 12, sampling=Sampling(seed=7, temperature=0.0), drafts=0)
    ngram = SessionNGram(vocab=model.cfg.vocab_size)
    model = tiny_gpu(kind="swiglu", act="alg")
    b = generate(model, prompt, 12, sampling=Sampling(seed=7, temperature=0.0), drafts=4, ngram=ngram)
    rows.append({"check": "drafts_byte_identical", "ok": a["new"] == b["new"]})
    print(json.dumps({"check": "drafts_byte_identical", "ok": a["new"] == b["new"]}), flush=True)
    return rows


def bench_step_split() -> list[dict[str, Any]]:
    """Where a decode token's wall time goes on CPU torch: forward vs host sampling vs Python loop."""

    import torch

    from tensorfold.engine.cuda_spear import tiny_gpu
    from tensorfold.engine.exact_sampling import Sampling

    sampling = Sampling(seed=3, temperature=0.0)
    rows = []
    for name, kw in (("swiglu-128", dict(kind="swiglu", n_layer=2, n_embd=128, n_head=4, n_inner=256)),
                     ("swiglu-768x4", dict(kind="swiglu", n_layer=4, n_embd=768, n_head=12, n_inner=3072,
                                           vocab=512, seq=128))):
        model = tiny_gpu(act="alg", **kw)
        ids = list(range(16, 48))
        with torch.no_grad():
            model.forward(ids)
            t_fwd = _median_s(lambda: model.forward([4]), warmup=3, reps=9) * 1e3
            logits = model.forward([4])
            t_sample = _median_s(lambda: model.sample(logits, 50, sampling), warmup=3, reps=25) * 1e3
        rows.append({"model": name, "ms_forward": round(t_fwd, 4), "ms_sample": round(t_sample, 4)})
        print(json.dumps(rows[-1]), flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--output", default=DEFAULT_OUT)
    args = p.parse_args(argv)
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        print("bench_spear_torch_cpu: needs PyTorch (pip install torch).", flush=True)
        return 2
    info = _host()
    print(f"torch CPU engine: {info}", flush=True)
    result: dict[str, Any] = {"info": info}
    print("=== activations (torch vs torch_ops champions) ===", flush=True)
    result["activations"] = bench_activations()
    print("=== SwiGLU (torch baseline vs champions; CPU fallback is not fused) ===", flush=True)
    result["swiglu"] = bench_swiglu()
    print("=== tiny GPU-engine models on CPU torch ===", flush=True)
    result["tiny"] = bench_tiny(args.tokens)
    print("=== per-token split (forward vs sampling) ===", flush=True)
    result["steps"] = bench_step_split()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
