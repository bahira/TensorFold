"""Vectorized PyTorch formulas (CPU or CUDA). Used as the Triton fallback and in tests."""

from __future__ import annotations

from typing import Any

from . import champions as C


def gelu_alg(x: Any) -> Any:
    t = (0.306923 * x + 0.501).clamp(min=0.0, max=1.002)
    return 0.997729 * x * t - 0.004004


def gelu_fast(x: Any) -> Any:
    return 1.010719 * x.clamp(min=0.0) - 0.057684


def silu_alg(x: Any) -> Any:
    return x * (0.501 + 0.587 * x / (0.815 + (1.0 + x * x).sqrt()))


def silu_fast(x: Any) -> Any:
    return 1.016356 * x.clamp(min=0.0) - 0.15849


def sigmoid_fast(x: Any) -> Any:
    return 0.605014 * x / (1.24384 + x.abs()) + 0.5


def swiglu_silu(gate: Any, up: Any, *, silu: str) -> Any:
    import torch.nn.functional as F

    if silu == "silu_exact":
        return F.silu(gate) * up
    return apply(silu, gate) * up


_FN = {
    "gelu_alg": gelu_alg,
    "gelu_fast": gelu_fast,
    "silu_alg": silu_alg,
    "silu_fast": silu_fast,
    "sigmoid_fast": sigmoid_fast,
}


def apply(op: int | str, x: Any) -> Any:
    import torch.nn.functional as F
    import torch

    name = op if isinstance(op, str) else next(k for k, v in C.OP_BY_NAME.items() if v == op)
    if name == "gelu_exact":
        return F.gelu(x)
    if name == "silu_exact":
        return F.silu(x)
    if name == "sigmoid_exact":
        return torch.sigmoid(x)
    return _FN[name](x)
