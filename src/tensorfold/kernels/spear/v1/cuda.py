"""SuperSpear champion kernels on CUDA (Triton). No nvcc: any PyTorch+NVIDIA notebook.

Elementwise GELU/SiLU against ``F.gelu`` / ``F.silu`` is usually a wash on GPU (the SFU is
fast; the paper said so). The interesting kernel is **fused SwiGLU**: ``silu(gate) * up`` in
one write, which is what Qwen/LLaMA/Mistral and TensorFold's CUDA engines run every layer.
"""

from __future__ import annotations

from typing import Any

from . import champions as C

_TRITON_READY: bool | None = None
_KERN = None


def available() -> bool:
    global _TRITON_READY
    if _TRITON_READY is not None:
        return _TRITON_READY
    try:
        import torch
        import triton  # noqa: F401

        _TRITON_READY = bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        _TRITON_READY = False
    return _TRITON_READY


def info() -> str:
    if not available():
        return "spear CUDA: unavailable (needs PyTorch with CUDA)"
    import torch

    d = torch.cuda.get_device_properties(0)
    return (f"spear CUDA: {torch.cuda.get_device_name(0)}, {d.total_memory / 1e9:.1f} GB, "
            f"sm_{d.major}{d.minor}, torch {torch.__version__}")


def _kernels():
    global _KERN
    if _KERN is not None:
        return _KERN
    import triton
    import triton.language as tl

    @triton.jit
    def _act(X, Y, N, OP: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        x = tl.load(X + off, mask=mask, other=0.0).to(tl.float32)
        if OP == 1:                                      # gelu alg
            t = 0.306923 * x + 0.501
            t = tl.minimum(tl.maximum(t, 0.0), 1.002)
            y = 0.997729 * x * t - 0.004004
        elif OP == 2:                                    # gelu fast
            y = 1.010719 * tl.maximum(x, 0.0) - 0.057684
        elif OP == 4:                                    # silu alg
            y = x * (0.501 + 0.587 * x / (0.815 + tl.sqrt(1.0 + x * x)))
        elif OP == 5:                                    # silu fast
            y = 1.016356 * tl.maximum(x, 0.0) - 0.15849
        else:                                            # sigmoid fast
            y = 0.605014 * x / (1.24384 + tl.abs(x)) + 0.5
        tl.store(Y + off, y, mask=mask)

    @triton.jit
    def _swiglu(G, U, Y, N, OP: tl.constexpr, BLOCK: tl.constexpr):
        off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = off < N
        g = tl.load(G + off, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U + off, mask=mask, other=0.0).to(tl.float32)
        if OP == 3:                                      # silu exact, fused multiply
            s = g / (1.0 + tl.exp(-g))
        elif OP == 4:
            s = g * (0.501 + 0.587 * g / (0.815 + tl.sqrt(1.0 + g * g)))
        else:
            s = 1.016356 * tl.maximum(g, 0.0) - 0.15849
        tl.store(Y + off, s * u, mask=mask)

    _KERN = (triton, _act, _swiglu)
    return _KERN


def _fp32_view(x):
    import torch

    v = x.reshape(-1).contiguous()
    return v if v.dtype == torch.float32 else v.to(torch.float32)


def _launch_act(n: int, src, dst, op: int) -> None:
    triton, kern, _ = _kernels()
    block = 1024
    kern[(triton.cdiv(n, block),)](src, dst, n, OP=op, BLOCK=block)


def _launch_swiglu(n: int, g, u, dst, op: int) -> None:
    triton, _, kern = _kernels()
    block = 1024
    kern[(triton.cdiv(n, block),)](g, u, dst, n, OP=op, BLOCK=block)


def act(op: int | str, x: Any) -> Any:
    """Elementwise SuperSpear champion on a CUDA tensor (fp16/bf16/fp32).

    ``exact`` slots call the fused PyTorch op (the GPU reference). ``alg`` / ``fast`` run Triton.
    """

    import torch
    import torch.nn.functional as F

    if not x.is_cuda:
        raise ValueError("spear CUDA act expects a CUDA tensor")
    name = op if isinstance(op, str) else next(k for k, v in C.OP_BY_NAME.items() if v == op)
    if name == "gelu_exact":
        return F.gelu(x)
    if name == "silu_exact":
        return F.silu(x)
    if name == "sigmoid_exact":
        return torch.sigmoid(x)
    code = C.OP_BY_NAME[name]
    src = _fp32_view(x)
    y = torch.empty_like(src)
    _launch_act(src.numel(), src, y, code)
    return y.reshape(x.shape).to(dtype=x.dtype)


def swiglu(gate: Any, up: Any, *, silu: str = "silu_alg") -> Any:
    """Fused ``silu(gate) * up`` (one write). ``silu`` is exact / alg / fast."""

    import torch

    if gate.shape != up.shape:
        raise ValueError("gate and up must match")
    code = C.OP_BY_NAME[silu]
    g, u = _fp32_view(gate), _fp32_view(up)
    y = torch.empty_like(g)
    _launch_swiglu(g.numel(), g, u, y, code)
    return y.reshape(gate.shape).to(dtype=gate.dtype)


def torch_swiglu(gate: Any, up: Any) -> Any:
    """The framework baseline: ``F.silu(gate) * up`` (two kernels, extra write)."""

    import torch.nn.functional as F

    return F.silu(gate) * up
