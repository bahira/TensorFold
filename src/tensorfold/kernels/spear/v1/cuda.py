"""SuperSpear champion kernels on CUDA (Triton). No nvcc: any PyTorch+NVIDIA notebook.

Kernels load the tensor's own dtype, compute in fp32, store back — no host-side fp32 clone
(that copy is what made a first draft lose to ``F.silu``). If Triton is missing, the same
algebra runs as vectorized PyTorch (``torch_ops``), CPU or GPU.

The measurement that can win on a laptop is fused SwiGLU, not elementwise GELU.
"""

from __future__ import annotations

import os
from typing import Any

from . import champions as C
from . import torch_ops

_TRITON_READY: bool | None = None
_KERN = None


def available() -> bool:
    """True when we can run on an NVIDIA GPU (Triton or the PyTorch fallback)."""

    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def triton_ok() -> bool:
    global _TRITON_READY
    if _TRITON_READY is not None:
        return _TRITON_READY
    if not available():
        _TRITON_READY = False
        return False
    try:
        import triton  # noqa: F401

        _TRITON_READY = True
    except Exception:  # noqa: BLE001
        _TRITON_READY = False
    return _TRITON_READY


def info() -> str:
    if not available():
        return "spear CUDA: unavailable (needs PyTorch with CUDA)"
    import torch

    d = torch.cuda.get_device_properties(0)
    backend = "triton" if triton_ok() else "torch-fallback"
    return (f"spear CUDA: {torch.cuda.get_device_name(0)}, {d.total_memory / 1e9:.1f} GB, "
            f"sm_{d.major}{d.minor}, torch {torch.__version__}, {backend}")


def _kernels():
    global _KERN
    if _KERN is not None:
        return _KERN
    os.environ.setdefault("TRITON_CACHE_DIR", str(
        __import__("pathlib").Path.home() / ".cache" / "tensorfold" / "triton"))
    import triton
    import triton.language as tl

    @triton.jit
    def _act(X, Y, N, OP: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if OP == 1:
            t = 0.306923 * x + 0.501
            t = tl.where(t < 0.0, 0.0, t)
            t = tl.where(t > 1.002, 1.002, t)
            y = 0.997729 * x * t - 0.004004
        elif OP == 2:
            y = 1.010719 * tl.where(x > 0.0, x, 0.0) - 0.057684
        elif OP == 4:
            y = x * (0.501 + 0.587 * x / (0.815 + tl.sqrt(1.0 + x * x)))
        elif OP == 5:
            y = 1.016356 * tl.where(x > 0.0, x, 0.0) - 0.15849
        else:
            ax = tl.where(x >= 0.0, x, -x)
            y = 0.605014 * x / (1.24384 + ax) + 0.5
        tl.store(Y + off, y, mask=mask)

    @triton.jit
    def _swiglu(G, U, Y, N, OP: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        g = tl.load(G + off, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U + off, mask=mask, other=0.0).to(tl.float32)
        if OP == 3:
            s = g / (1.0 + tl.exp(-g))
        elif OP == 4:
            s = g * (0.501 + 0.587 * g / (0.815 + tl.sqrt(1.0 + g * g)))
        else:
            s = 1.016356 * tl.where(g > 0.0, g, 0.0) - 0.15849
        tl.store(Y + off, s * u, mask=mask)

    _KERN = (triton, _act, _swiglu)
    return _KERN


def _launch(kind: str, n: int, op: int, *tensors) -> None:
    triton, act_k, swiglu_k = _kernels()
    block = 1024
    grid = (triton.cdiv(max(n, 1), block),)
    kern = act_k if kind == "act" else swiglu_k
    kern[grid](*tensors, n, OP=op, BLOCK=block, num_warps=4)


def warmup() -> None:
    """Compile every champion once so the first timed run is not the JIT."""

    if not (available() and triton_ok()):
        return
    import torch

    x = torch.zeros(1024, device="cuda", dtype=torch.float16)
    for name in ("gelu_alg", "gelu_fast", "silu_alg", "silu_fast", "sigmoid_fast"):
        act(name, x)
    swiglu(x, x, silu="silu_exact")
    swiglu(x, x, silu="silu_alg")
    swiglu(x, x, silu="silu_fast")
    torch.cuda.synchronize()


def act(op: int | str, x: Any) -> Any:
    """Elementwise SuperSpear champion. CUDA+Triton when possible, else ``torch_ops``."""

    import torch.nn.functional as F
    import torch

    name = op if isinstance(op, str) else next(k for k, v in C.OP_BY_NAME.items() if v == op)
    if name == "gelu_exact":
        return F.gelu(x)
    if name == "silu_exact":
        return F.silu(x)
    if name == "sigmoid_exact":
        return torch.sigmoid(x)
    if x.is_cuda and triton_ok():
        src = x.reshape(-1).contiguous()
        y = torch.empty_like(src)
        _launch("act", src.numel(), C.OP_BY_NAME[name], src, y)
        return y.reshape(x.shape)
    return torch_ops.apply(name, x)


def swiglu(gate: Any, up: Any, *, silu: str = "silu_alg") -> Any:
    """Fused ``silu(gate) * up`` (one write)."""

    import torch

    if gate.shape != up.shape:
        raise ValueError("gate and up must match")
    if gate.is_cuda and triton_ok():
        g = gate.reshape(-1).contiguous()
        u = up.reshape(-1).contiguous()
        y = torch.empty_like(g)
        _launch("swiglu", g.numel(), C.OP_BY_NAME[silu], g, u, y)
        return y.reshape(gate.shape)
    return torch_ops.swiglu_silu(gate, up, silu=silu)


def torch_swiglu(gate: Any, up: Any) -> Any:
    """Framework baseline: ``F.silu(gate) * up`` (two kernels, extra write)."""

    import torch.nn.functional as F

    return F.silu(gate) * up
