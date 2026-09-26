"""SuperSpear torch formulas and the GPU engine on CPU torch (skip if torch is missing)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.kernels.spear.v1 import champions as C

torch = pytest.importorskip("torch")


def test_torch_ops_match_numpy():
    from tensorfold.kernels.spear.v1 import torch_ops

    rng = np.random.default_rng(0)
    x_np = rng.normal(0, 1.2, 2048).astype(np.float32)
    x = torch.from_numpy(x_np)
    pairs = [
        ("gelu_alg", C.gelu_alg, torch_ops.gelu_alg),
        ("gelu_fast", C.gelu_fast, torch_ops.gelu_fast),
        ("silu_alg", C.silu_alg, torch_ops.silu_alg),
        ("silu_fast", C.silu_fast, torch_ops.silu_fast),
        ("sigmoid_fast", C.sigmoid_fast, torch_ops.sigmoid_fast),
    ]
    for name, np_fn, th_fn in pairs:
        err = float(np.max(np.abs(np_fn(x_np) - th_fn(x).numpy())))
        assert err < 2e-5, (name, err)


def test_cuda_act_falls_back_on_cpu():
    from tensorfold.kernels.spear.v1 import cuda as spear_cuda

    x = torch.randn(512)
    y = spear_cuda.act("gelu_alg", x)
    assert y.shape == x.shape
    assert not spear_cuda.available() or y.device.type in ("cpu", "cuda")


def test_gpu_engine_drafts_on_cpu_torch():
    from tensorfold.drafters.draft_ngram import SessionNGram
    from tensorfold.engine.cuda_spear import generate, tiny_gpu
    from tensorfold.engine.exact_sampling import Sampling

    prompt = list(range(8, 24))
    sampling = Sampling(seed=3, temperature=0.0)
    a = generate(tiny_gpu(act="alg", device=torch.device("cpu")), prompt, 8, sampling=sampling, drafts=0)
    ngram = SessionNGram(vocab=256)
    b = generate(tiny_gpu(act="alg", device=torch.device("cpu")), prompt, 8, sampling=sampling, drafts=4,
                 ngram=ngram)
    assert a["new"] == b["new"]
    assert len(a["new"]) == 8


def test_gpu_engine_swiglu_and_gpt2_run():
    from tensorfold.engine.cuda_spear import generate, tiny_gpu
    from tensorfold.engine.exact_sampling import Sampling

    sampling = Sampling(seed=0, temperature=0.0)
    for kind in ("swiglu", "gpt2"):
        out = generate(tiny_gpu(kind=kind, act="fast", device=torch.device("cpu")),
                       list(range(4, 12)), 4, sampling=sampling)
        assert len(out["new"]) == 4
