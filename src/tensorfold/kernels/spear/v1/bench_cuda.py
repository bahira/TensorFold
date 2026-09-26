"""Notebook CUDA bench: SuperSpear Triton vs fused PyTorch, plus tiny GPU decode.

  tensorfold bench-cuda
  python -m tensorfold.kernels.spear.v1.bench_cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Callable


def _need_cuda() -> int:
    from tensorfold.kernels.spear.v1 import cuda as spear_cuda

    if not spear_cuda.available():
        print("tensorfold bench-cuda: needs PyTorch with an NVIDIA GPU "
              "(pip install torch, then a machine with CUDA).", flush=True)
        return 2
    return 0


def _median_s(fn: Callable[[], None], *, warmup: int = 10, reps: int = 50) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def bench_activations(n: int = 1 << 22) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from tensorfold.kernels.spear.v1 import cuda as spear_cuda

    x = torch.randn(n, device="cuda", dtype=torch.float16)
    rows = []
    specs = [
        ("gelu", lambda: F.gelu(x), "gelu_alg", "gelu_fast"),
        ("silu", lambda: F.silu(x), "silu_alg", "silu_fast"),
        ("sigmoid", lambda: torch.sigmoid(x), None, "sigmoid_fast"),
    ]
    for name, ref, alg, fast in specs:
        ns_ref = _median_s(ref) * 1e9 / n
        row: dict[str, Any] = {"op": name, "n": n, "ns_elem_torch": ns_ref, "dtype": "fp16"}
        if alg:
            ns_alg = _median_s(lambda op=alg: spear_cuda.act(op, x)) * 1e9 / n
            mse = float(((ref() - spear_cuda.act(alg, x)).float().pow(2).mean()).item())
            row.update(ns_elem_alg=ns_alg, speedup_alg=ns_ref / ns_alg, mse_alg=mse)
        ns_fast = _median_s(lambda op=fast: spear_cuda.act(op, x)) * 1e9 / n
        mse_f = float(((ref() - spear_cuda.act(fast, x)).float().pow(2).mean()).item())
        row.update(ns_elem_fast=ns_fast, speedup_fast=ns_ref / ns_fast, mse_fast=mse_f)
        rows.append(row)
        print(json.dumps({k: row[k] for k in row if k in ("op", "speedup_alg", "speedup_fast",
                                                          "mse_alg", "mse_fast")}), flush=True)
    return rows


def bench_swiglu() -> list[dict[str, Any]]:
    """The kernel that matters: fused silu(gate)*up at decode (M=1) and prefill (M=128)."""

    import torch
    from tensorfold.kernels.spear.v1 import cuda as spear_cuda

    rows = []
    for m, n in ((1, 2048), (1, 4096), (1, 11008), (128, 4096)):
        g = torch.randn(m, n, device="cuda", dtype=torch.float16)
        u = torch.randn(m, n, device="cuda", dtype=torch.float16)
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
        print(json.dumps({k: row[k] for k in ("M", "N", "speedup_exact", "speedup_alg", "speedup_fast")}),
              flush=True)
    return rows


def bench_tiny(n_new: int) -> list[dict[str, Any]]:
    from tensorfold.drafters.draft_ngram import SessionNGram
    from tensorfold.engine.cuda_spear import generate, tiny_gpu
    from tensorfold.engine.exact_sampling import Sampling, seed_for

    import torch

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
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                last = generate(model, prompt, n_new, sampling=sampling, drafts=drafts, ngram=ngram)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
                runs.append(last["tokens"] / dt if dt else 0.0)
            row = {
                "model": name, "label": label, "act": act, "drafts": drafts,
                "tok_s": statistics.median(runs), "tok_s_all": [round(x, 2) for x in runs],
                "accepted": last["accepted"], "proposed": last["proposed"],
                "accept_rate": (last["accepted"] / last["proposed"]) if last["proposed"] else None,
            }
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("model", "label", "tok_s", "accept_rate")}), flush=True)
    # drafts == serial
    model = tiny_gpu(kind="swiglu", act="alg")
    a = generate(model, prompt, 12, sampling=Sampling(seed=7, temperature=0.0), drafts=0)
    ngram = SessionNGram(vocab=model.cfg.vocab_size)
    model = tiny_gpu(kind="swiglu", act="alg")
    b = generate(model, prompt, 12, sampling=Sampling(seed=7, temperature=0.0), drafts=4, ngram=ngram)
    rows.append({"check": "drafts_byte_identical", "ok": a["new"] == b["new"]})
    print(json.dumps({"check": "drafts_byte_identical", "ok": a["new"] == b["new"]}), flush=True)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--output", default="docs/recipes/spear-cuda-results.json")
    args = p.parse_args(argv)
    if _need_cuda():
        return 2
    from tensorfold.kernels.spear.v1 import cuda as spear_cuda

    print(spear_cuda.info(), flush=True)
    result: dict[str, Any] = {"info": spear_cuda.info()}
    print("=== activations fp16 (vs F.gelu / F.silu / sigmoid) ===", flush=True)
    result["activations"] = bench_activations()
    print("=== fused SwiGLU (vs F.silu(g)*u) ===", flush=True)
    result["swiglu"] = bench_swiglu()
    print("=== tiny GPU models ===", flush=True)
    result["tiny"] = bench_tiny(args.tokens)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
