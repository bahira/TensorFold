"""SuperSpear CUDA kernels: skip the whole tests/cuda package when there is no GPU."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorfold.drafters.draft_ngram import SessionNGram
from tensorfold.engine.cuda_spear import generate, tiny_gpu
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.kernels.spear.v1 import cuda as spear_cuda


def test_cuda_available():
    assert spear_cuda.available()
    assert torch.cuda.is_available()


def test_gelu_alg_tracks_torch():
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    got = spear_cuda.act("gelu_alg", x)
    ref = F.gelu(x)
    mse = float((got - ref).pow(2).mean())
    assert mse < 2e-3


def test_silu_alg_tracks_torch():
    x = torch.randn(4096, device="cuda", dtype=torch.float32)
    mse = float((spear_cuda.act("silu_alg", x) - F.silu(x)).pow(2).mean())
    assert mse < 5e-3


def test_swiglu_exact_matches_torch():
    g = torch.randn(32, 256, device="cuda", dtype=torch.float16)
    u = torch.randn(32, 256, device="cuda", dtype=torch.float16)
    got = spear_cuda.swiglu(g, u, silu="silu_exact")
    ref = spear_cuda.torch_swiglu(g, u)
    assert torch.allclose(got.float(), ref.float(), rtol=1e-3, atol=1e-3)


def test_tiny_swiglu_greedy_and_drafts():
    prompt = list(range(8, 24))
    sampling = Sampling(seed=3, temperature=0.0)
    a = generate(tiny_gpu(act="alg"), prompt, 8, sampling=sampling, drafts=0)
    ngram = SessionNGram(vocab=256)
    b = generate(tiny_gpu(act="alg"), prompt, 8, sampling=sampling, drafts=4, ngram=ngram)
    assert a["new"] == b["new"]
    assert len(a["new"]) == 8


def test_warmup_and_fp16_swiglu():
    spear_cuda.warmup()
    g = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
    u = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
    y = spear_cuda.swiglu(g, u, silu="silu_alg")
    assert y.shape == g.shape and y.dtype == torch.float16
    y2 = spear_cuda.act("gelu_fast", g)
    assert y2.dtype == torch.float16
