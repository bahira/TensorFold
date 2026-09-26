"""Compile-on-first-use SuperSpear C kernels (AVX-512 / VNNI), same pattern as the CUDA engines."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from . import champions as C

_LIB = None
_ERROR: str | None = None


def _cache_dir() -> Path:
    return Path(os.environ.get("TENSORFOLD_SPEAR_CACHE", Path.home() / ".cache" / "tensorfold" / "spear"))


def compile_lib() -> ctypes.CDLL:
    src = Path(__file__).parent / "kernels.c"
    hdr = Path(__file__).parent / "kernels.h"
    digest = hashlib.sha256(src.read_bytes() + hdr.read_bytes()).hexdigest()[:16]
    cache = _cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"spear_{digest}.so"
    if not so.is_file():
        cmd = [
            os.environ.get("CC", "gcc"),
            "-O3", "-shared", "-fPIC", "-std=c11",
            "-fopenmp", "-march=native", "-ffast-math", "-fno-math-errno",
            "-I", str(src.parent),
            str(src), "-o", str(so), "-lm",
        ]
        built = subprocess.run(cmd, capture_output=True, text=True)
        if built.returncode != 0:
            raise RuntimeError(built.stderr or built.stdout or f"gcc exited {built.returncode}")
    lib = ctypes.CDLL(str(so))
    f32p = ctypes.POINTER(ctypes.c_float)
    i8p = ctypes.POINTER(ctypes.c_int8)
    i32p = ctypes.POINTER(ctypes.c_int32)
    lib.spear_act_f32.argtypes = [ctypes.c_int, f32p, f32p, ctypes.c_int]
    lib.spear_swiglu_f32.argtypes = [ctypes.c_int, f32p, f32p, f32p, ctypes.c_int]
    lib.spear_gemv_f32.argtypes = [f32p, f32p, f32p, f32p, ctypes.c_int, ctypes.c_int]
    lib.spear_gemm_f32.argtypes = [f32p, f32p, f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.spear_quantize_weight_i8.argtypes = [f32p, i8p, f32p, i32p, ctypes.c_int, ctypes.c_int]
    lib.spear_pack_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    lib.spear_pack_bytes.restype = ctypes.c_int
    lib.spear_pack_i8.argtypes = [i8p, i8p, ctypes.c_int, ctypes.c_int]
    lib.spear_gemv_i8.argtypes = [i8p, f32p, i32p, f32p, f32p, f32p, ctypes.c_int, ctypes.c_int]
    lib.spear_gemv_i8_act.argtypes = [i8p, f32p, i32p, f32p, f32p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.spear_has_avx512.restype = ctypes.c_int
    lib.spear_has_vnni.restype = ctypes.c_int
    return lib


def lib() -> ctypes.CDLL:
    global _LIB, _ERROR
    if _LIB is not None:
        return _LIB
    if _ERROR is not None:
        raise RuntimeError(_ERROR)
    try:
        _LIB = compile_lib()
    except Exception as exc:  # noqa: BLE001 - first-use compile, surface the gcc line
        _ERROR = f"spear C kernels failed to build: {exc}"
        raise RuntimeError(_ERROR) from exc
    return _LIB


def available() -> bool:
    try:
        lib()
        return True
    except Exception:  # noqa: BLE001
        return False


def _f32(a: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(a, dtype=np.float32)
    if out.ndim != 1 and out.ndim != 2:
        raise ValueError(f"expected 1- or 2-d float32, got {out.shape}")
    return out


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i8ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def _i32ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def features() -> dict[str, bool]:
    if not available():
        return {"avx512": False, "vnni": False, "compiled": False}
    L = lib()
    return {"avx512": bool(L.spear_has_avx512()), "vnni": bool(L.spear_has_vnni()), "compiled": True}


def act(op: int | str, x: np.ndarray) -> np.ndarray:
    """Apply a SuperSpear champion (or its exact reference) elementwise."""

    code = C.OP_BY_NAME[op] if isinstance(op, str) else int(op)
    x = _f32(x)
    if not available():
        return C.apply_numpy(code, x)
    y = np.empty_like(x)
    lib().spear_act_f32(code, _ptr(x.ravel()), _ptr(y.ravel()), int(x.size))
    return y


def swiglu(gate: np.ndarray, up: np.ndarray, *, silu: str = "silu_alg") -> np.ndarray:
    code = C.OP_BY_NAME[silu]
    gate, up = _f32(gate), _f32(up)
    if gate.shape != up.shape:
        raise ValueError("gate and up must match")
    if not available():
        return (C.apply_numpy(code, gate) * up).astype(np.float32)
    y = np.empty_like(gate)
    lib().spear_swiglu_f32(code, _ptr(gate.ravel()), _ptr(up.ravel()), _ptr(y.ravel()), int(gate.size))
    return y


def gemv_f32(W: np.ndarray, x: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """y[N] = W[N, K] @ x[K] + bias[N], AVX-512 FMA."""

    W, x = _f32(W), _f32(x).ravel()
    N, K = int(W.shape[0]), int(W.shape[1])
    if x.size != K:
        raise ValueError(f"x has {x.size} elems, W is {W.shape}")
    y = np.empty(N, dtype=np.float32)
    b = _f32(bias).ravel() if bias is not None else None
    if not available():
        y[:] = W @ x
        if b is not None:
            y += b
        return y
    lib().spear_gemv_f32(_ptr(W), _ptr(x), _ptr(b) if b is not None else None, _ptr(y), N, K)
    return y


def gemm_f32(A: np.ndarray, W: np.ndarray, bias: np.ndarray | None = None) -> np.ndarray:
    """C[M, N] = A[M, K] @ W[N, K].T + bias[N]. Prefer numpy for large M; this is the decode/verify path."""

    A, W = _f32(A), _f32(W)
    if A.ndim == 1:
        return gemv_f32(W, A, bias)
    M, K = int(A.shape[0]), int(A.shape[1])
    N = int(W.shape[0])
    if int(W.shape[1]) != K:
        raise ValueError(f"W {W.shape} vs A {A.shape}")
    if M >= 8:
        C = A @ W.T
        if bias is not None:
            C += _f32(bias).ravel()
        return C.astype(np.float32, copy=False)
    out = np.empty((M, N), dtype=np.float32)
    b = _f32(bias).ravel() if bias is not None else None
    if not available():
        out[:] = A @ W.T
        if b is not None:
            out += b
        return out
    lib().spear_gemm_f32(_ptr(A), _ptr(W), _ptr(b) if b is not None else None, _ptr(out), M, N, K)
    return out


@dataclass
class PackedI8:
    packed: np.ndarray
    scale: np.ndarray
    col_sum: np.ndarray
    N: int
    K: int


def pack_i8(W: np.ndarray) -> PackedI8:
    W = _f32(W)
    N, K = int(W.shape[0]), int(W.shape[1])
    L = lib()
    Wq = np.empty((N, K), dtype=np.int8)
    scale = np.empty(N, dtype=np.float32)
    col_sum = np.empty(N, dtype=np.int32)
    L.spear_quantize_weight_i8(_ptr(W), _i8ptr(Wq), _ptr(scale), _i32ptr(col_sum), N, K)
    nbytes = int(L.spear_pack_bytes(N, K))
    packed = np.empty(nbytes, dtype=np.int8)
    L.spear_pack_i8(_i8ptr(Wq), _i8ptr(packed), N, K)
    return PackedI8(packed=packed, scale=scale, col_sum=col_sum, N=N, K=K)


def gemv_i8(p: PackedI8, x: np.ndarray, bias: np.ndarray | None = None, *, act: str | None = None) -> np.ndarray:
    x = _f32(x).ravel()
    if x.size != p.K:
        raise ValueError(f"x has {x.size}, packed K={p.K}")
    y = np.empty(p.N, dtype=np.float32)
    b = _f32(bias).ravel() if bias is not None else None
    bp = _ptr(b) if b is not None else None
    if act is None:
        lib().spear_gemv_i8(_i8ptr(p.packed), _ptr(p.scale), _i32ptr(p.col_sum), _ptr(x), bp, _ptr(y), p.N, p.K)
    else:
        lib().spear_gemv_i8_act(_i8ptr(p.packed), _ptr(p.scale), _i32ptr(p.col_sum), _ptr(x), bp, _ptr(y),
                                p.N, p.K, C.OP_BY_NAME[act])
    return y


def info() -> str:
    feat = features()
    bits = ", ".join(k for k, v in feat.items() if v) or "unavailable"
    return f"spear kernels: {bits} (python {sys.version.split()[0]})"
