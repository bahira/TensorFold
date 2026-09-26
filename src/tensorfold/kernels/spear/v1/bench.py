#!/usr/bin/env python3
"""Benchmark SuperSpear champion kernels and tiny CPU models on this machine.

  python tools/bench_spear_cpu.py
  python tools/bench_spear_cpu.py --model distilgpt2 --tokens 32
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any, Callable

import numpy as np

from tensorfold.drafters.draft_ngram import SessionNGram
from tensorfold.engine.cpu_spear import generate, load_gpt2_safetensors, tiny_gpt2, tiny_swiglu
from tensorfold.engine.exact_sampling import Sampling, seed_for
from tensorfold.kernels.spear.v1 import champions as C
from tensorfold.kernels.spear.v1 import native as spear


def _median_ns(fn: Callable[[], None], *, warmup: int = 3, reps: int = 9, inner: int = 20) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(inner):
            fn()
        samples.append((time.perf_counter() - t0) / inner)
    return statistics.median(samples) * 1e9


def bench_activations(n: int = 1 << 20) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.5, n).astype(np.float32)
    y = np.empty_like(x)
    rows = []
    pairs = [
        ("gelu", C.GELU_EXACT, C.GELU_ALG, C.GELU_FAST),
        ("silu", C.SILU_EXACT, C.SILU_ALG, C.SILU_FAST),
        ("sigmoid", C.SIGMOID_EXACT, None, C.SIGMOID_FAST),
    ]
    for name, exact, alg, fast in pairs:
        def run(op: int) -> None:
            y[:] = spear.act(op, x)

        ns_exact = _median_ns(lambda: run(exact))
        mse_alg = mse_fast = None
        ns_alg = None
        if alg is not None:
            ns_alg = _median_ns(lambda op=alg: run(op))
            mse_alg = float(np.mean((spear.act(exact, x) - spear.act(alg, x)) ** 2))
        ns_fast = _median_ns(lambda op=fast: run(op))
        mse_fast = float(np.mean((spear.act(exact, x) - spear.act(fast, x)) ** 2))
        row = {
            "op": name, "n": n,
            "ns_elem_exact": ns_exact / n,
            "ns_elem_alg": None if ns_alg is None else ns_alg / n,
            "ns_elem_fast": ns_fast / n,
            "speedup_alg": None if ns_alg is None else ns_exact / ns_alg,
            "speedup_fast": ns_exact / ns_fast,
            "mse_alg": mse_alg, "mse_fast": mse_fast,
        }
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("op", "speedup_alg", "speedup_fast", "mse_alg", "mse_fast",
                                               "ns_elem_exact", "ns_elem_fast")}), flush=True)
    return rows


def bench_gemv(n: int = 3072, k: int = 768) -> dict[str, Any]:
    rng = np.random.default_rng(1)
    W = rng.normal(0, 0.02, (n, k)).astype(np.float32)
    x = rng.normal(0, 1, k).astype(np.float32)
    packed = spear.pack_i8(W) if spear.available() else None

    def np_gemv() -> None:
        _ = W @ x

    ns_np = _median_ns(np_gemv, inner=10)
    ns_avx = _median_ns(lambda: spear.gemv_f32(W, x), inner=10)
    out = {
        "N": n, "K": k,
        "ns_numpy": ns_np, "ns_avx": ns_avx,
        "speedup_avx_vs_numpy": ns_np / ns_avx,
    }
    if packed is not None:
        ns_i8 = _median_ns(lambda: spear.gemv_i8(packed, x), inner=10)
        y_ref = W @ x
        y_i8 = spear.gemv_i8(packed, x)
        out.update({
            "ns_vnni": ns_i8,
            "speedup_vnni_vs_numpy": ns_np / ns_i8,
            "speedup_vnni_vs_avx": ns_avx / ns_i8,
            "cos_i8": float(np.dot(y_ref, y_i8) / (np.linalg.norm(y_ref) * np.linalg.norm(y_i8) + 1e-12)),
            "max_abs_i8": float(np.max(np.abs(y_ref - y_i8))),
        })
        print(json.dumps({k: out[k] for k in out if k.startswith("speedup") or k in ("cos_i8", "N", "K")}),
              flush=True)
    return out


def _decode_speed(model, prompt, n_new, drafts, ngram, *, reps: int = 3) -> dict[str, Any]:
    sampling = Sampling(seed=seed_for(prompt), temperature=0.0)
    generate(model, prompt, 2, sampling=sampling, drafts=0)  # warmup kernels
    runs = []
    last = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = generate(model, prompt, n_new, sampling=sampling, drafts=drafts, ngram=ngram)
        dt = time.perf_counter() - t0
        last = out
        runs.append(out["tokens"] / dt if dt else 0.0)
    out = last
    return {
        "tokens": out["tokens"], "seconds": out["tokens"] / statistics.median(runs) if runs else None,
        "tok_s": statistics.median(runs),
        "tok_s_all": [round(x, 2) for x in runs],
        "rounds": out["rounds"],
        "accepted": out["accepted"], "proposed": out["proposed"],
        "accept_rate": (out["accepted"] / out["proposed"]) if out["proposed"] else None,
    }


def bench_mlp_block() -> dict[str, Any]:
    """One DistilGPT2-shaped MLP: 768→3072 GELU 3072→768."""

    rng = np.random.default_rng(2)
    d, inner = 768, 3072
    W1 = rng.normal(0, 0.02, (inner, d)).astype(np.float32)
    b1 = rng.normal(0, 0.01, inner).astype(np.float32)
    W2 = rng.normal(0, 0.02, (d, inner)).astype(np.float32)
    b2 = rng.normal(0, 0.01, d).astype(np.float32)
    x = rng.normal(0, 1, d).astype(np.float32)
    p1, p2 = spear.pack_i8(W1), spear.pack_i8(W2)

    def fp32_exact() -> None:
        h = np.asarray(C.gelu_exact(W1 @ x + b1), dtype=np.float32)
        _ = W2 @ h + b2

    def spear_int8() -> None:
        h = spear.gemv_i8(p1, x, b1, act="gelu_alg")
        _ = spear.gemv_i8(p2, h, b2)

    ns_fp32 = _median_ns(fp32_exact, inner=8)
    ns_i8 = _median_ns(spear_int8, inner=8)
    row = {
        "shape": "768-3072-768",
        "ns_fp32_exact": ns_fp32,
        "ns_int8_spear": ns_i8,
        "speedup": ns_fp32 / ns_i8,
    }
    print(json.dumps(row), flush=True)
    return row


def bench_tiny(n_new: int = 32) -> list[dict[str, Any]]:
    prompt = list(range(16, 48))
    rows = []
    for kind, factory in (("gpt2-gelu", tiny_gpt2), ("llama-swiglu", tiny_swiglu)):
        for act, quant, drafts, label in (
            ("exact", False, 0, "fp32-exact"),
            ("alg", False, 0, "fp32-spear-alg"),
            ("fast", False, 0, "fp32-spear-fast"),
            ("alg", True, 0, "int8-spear-alg"),
            ("alg", True, 4, "int8-spear-alg-ngram"),
        ):
            model = factory(act=act, quant=quant)
            ngram = SessionNGram(vocab=model.config.vocab_size) if drafts else None
            stats = _decode_speed(model, prompt, n_new, drafts, ngram)
            row = {"model": kind, "label": label, "act": act, "quant": quant, "drafts": drafts, **stats}
            rows.append(row)
            print(json.dumps({k: row[k] for k in ("model", "label", "tok_s", "accept_rate")}), flush=True)
    # exactness: alg/fast/int8 drafts vs serial exact must match at temperature 0 only for exact slot;
    # for alg vs exact the tokens may differ. Check drafts vs serial on the same slot.
    model = tiny_gpt2(act="alg", quant=False)
    a = generate(model, prompt, 16, sampling=Sampling(seed=7, temperature=0.0), drafts=0)
    model = tiny_gpt2(act="alg", quant=False)
    ngram = SessionNGram(vocab=model.config.vocab_size)
    b = generate(model, prompt, 16, sampling=Sampling(seed=7, temperature=0.0), drafts=4, ngram=ngram)
    rows.append({"check": "drafts_byte_identical", "ok": a["new"] == b["new"],
                 "serial": a["new"], "drafted": b["new"]})
    print(json.dumps({"check": "drafts_byte_identical", "ok": a["new"] == b["new"]}), flush=True)

    # DistilGPT2-shaped synthetic (HF is unreachable here): GEMV-bound decode
    prompt_w = list(range(8, 40))
    for act, quant, drafts, label in (
        ("exact", False, 0, "fp32-exact"),
        ("alg", True, 0, "int8-spear-alg"),
        ("alg", True, 4, "int8-spear-alg-ngram"),
    ):
        model = tiny_gpt2(n_layer=4, n_embd=768, n_head=12, n_inner=3072, vocab=512,
                          seq=128, act=act, quant=quant)
        ngram = SessionNGram(vocab=model.config.vocab_size) if drafts else None
        stats = _decode_speed(model, prompt_w, n_new, drafts, ngram)
        row = {"model": "gpt2-768x4", "label": label, "act": act, "quant": quant,
               "drafts": drafts, **stats}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("model", "label", "tok_s", "accept_rate")}), flush=True)
    return rows


def try_distilgpt2(n_new: int, cache: Path) -> dict[str, Any] | None:
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        return None
    repo = "distilbert/distilgpt2"
    try:
        # prefer safetensors; some mirrors ship only pytorch_model.bin
        try:
            hf_hub_download(repo, "model.safetensors")
        except Exception:
            print(json.dumps({"distilgpt2": "no safetensors, skip"}), flush=True)
            return None
        path = Path(snapshot_download(repo, allow_patterns=[
            "config.json", "model.safetensors", "tokenizer.json", "vocab.json", "merges.txt",
        ]))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"distilgpt2": f"download failed: {exc}"}), flush=True)
        return None
    model = load_gpt2_safetensors(path, act="alg", quant=True, max_seq=256)
    prompt = [50256, 464, 3666, 318, 257, 922, 1517, 13]  # roughly BOS + "The world is a"
    if prompt[-1] >= model.config.vocab_size:
        prompt = [0, 1, 2, 3, 4, 5, 6, 7]
    ngram = SessionNGram(vocab=model.config.vocab_size)
    stats = _decode_speed(model, prompt, n_new, 4, ngram)
    stats.update({"model": "distilgpt2", "label": "int8-spear-alg-ngram",
                  "n_layer": model.config.n_layer, "n_embd": model.config.n_embd})
    print(json.dumps({k: stats[k] for k in ("model", "label", "tok_s", "tokens", "accept_rate")}), flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=32)
    p.add_argument("--model", default="tiny", help="tiny | distilgpt2 | all")
    p.add_argument("--output", default="docs/recipes/spear-cpu-results.json")
    args = p.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

    print(spear.info(), flush=True)
    result: dict[str, Any] = {
        "features": spear.features(),
        "info": spear.info(),
        "host": {
            "omp": os.environ.get("OMP_NUM_THREADS"),
        },
    }
    print("=== activations ===", flush=True)
    result["activations"] = bench_activations()
    print("=== gemv 3072x768 (FFN up) ===", flush=True)
    result["gemv"] = bench_gemv()
    print("=== mlp block 768-3072-768 ===", flush=True)
    result["mlp"] = bench_mlp_block()
    if args.model in ("tiny", "all"):
        print("=== tiny models ===", flush=True)
        result["tiny"] = bench_tiny(args.tokens)
    if args.model in ("distilgpt2", "all"):
        print("=== distilgpt2 ===", flush=True)
        result["distilgpt2"] = try_distilgpt2(args.tokens, Path.home() / ".cache" / "huggingface")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
