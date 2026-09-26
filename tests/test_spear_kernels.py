"""SuperSpear champion kernels: numpy formulas, AVX parity, int8 GEMV, drafted decode."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.drafters.draft_ngram import SessionNGram
from tensorfold.engine.cpu_spear import generate, tiny_gpt2, tiny_swiglu
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.kernels.spear.v1 import champions as C
from tensorfold.kernels.spear.v1 import native as spear


def test_numpy_gelu_alg_close_to_exact():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.2, 4096).astype(np.float32)
    mse = float(np.mean((C.gelu_exact(x) - C.gelu_alg(x)) ** 2))
    assert mse < 2e-3


def test_numpy_silu_alg_close_to_exact():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1.2, 4096).astype(np.float32)
    mse = float(np.mean((C.silu_exact(x) - C.silu_alg(x)) ** 2))
    assert mse < 5e-3


def test_numpy_sigmoid_fast_close_to_exact():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1.2, 4096).astype(np.float32)
    mse = float(np.mean((C.sigmoid_exact(x) - C.sigmoid_fast(x)) ** 2))
    assert mse < 2e-3


@pytest.mark.skipif(not spear.available(), reason="spear C kernels did not compile")
def test_avx_matches_numpy():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1.5, 1025).astype(np.float32)  # not a multiple of 16
    for name, op in C.OP_BY_NAME.items():
        got = spear.act(op, x)
        ref = C.apply_numpy(op, x)
        err = float(np.max(np.abs(got - ref)))
        # exact slots: C uses libm, numpy uses A&S 7.1.26 / exp — a few ulps, not bit-identical
        limit = 3e-3 if name.endswith("_exact") else 2e-5
        assert err < limit, (name, err)


@pytest.mark.skipif(not spear.available(), reason="spear C kernels did not compile")
def test_gemv_f32_matches_numpy():
    rng = np.random.default_rng(4)
    W = rng.normal(0, 0.05, (65, 33)).astype(np.float32)
    x = rng.normal(0, 1, 33).astype(np.float32)
    b = rng.normal(0, 0.01, 65).astype(np.float32)
    got = spear.gemv_f32(W, x, b)
    ref = W @ x + b
    assert float(np.max(np.abs(got - ref))) < 2e-4


@pytest.mark.skipif(not spear.available(), reason="spear C kernels did not compile")
def test_gemv_i8_tracks_fp32():
    rng = np.random.default_rng(5)
    W = rng.normal(0, 0.05, (128, 64)).astype(np.float32)
    x = rng.normal(0, 1, 64).astype(np.float32)
    packed = spear.pack_i8(W)
    got = spear.gemv_i8(packed, x)
    ref = W @ x
    cos = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
    assert cos > 0.99


def test_tiny_gpt2_greedy_deterministic():
    prompt = list(range(8, 24))
    a = generate(tiny_gpt2(act="alg"), prompt, 8, sampling=Sampling(seed=1, temperature=0.0))
    b = generate(tiny_gpt2(act="alg"), prompt, 8, sampling=Sampling(seed=1, temperature=0.0))
    assert a["new"] == b["new"]
    assert len(a["new"]) == 8


def test_ngram_drafts_match_serial():
    prompt = list(range(8, 40))
    sampling = Sampling(seed=11, temperature=0.0)
    serial = generate(tiny_gpt2(act="alg"), prompt, 12, sampling=sampling, drafts=0)
    ngram = SessionNGram(vocab=256)
    drafted = generate(tiny_gpt2(act="alg"), prompt, 12, sampling=sampling, drafts=4, ngram=ngram)
    assert serial["new"] == drafted["new"]


def test_tiny_swiglu_runs():
    out = generate(tiny_swiglu(act="fast"), list(range(4, 20)), 4,
                   sampling=Sampling(seed=0, temperature=0.0))
    assert len(out["new"]) == 4
