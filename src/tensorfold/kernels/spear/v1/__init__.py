"""SuperSpear champion kernels, v1: AVX-512 activations and VNNI int8 GEMV for the CPU engine."""

from .champions import CHAMPIONS, OP_BY_NAME, apply_numpy
from .native import act, available, features, gemm_f32, gemv_f32, gemv_i8, info, pack_i8, swiglu

__all__ = [
    "CHAMPIONS", "OP_BY_NAME", "act", "apply_numpy", "available", "features",
    "gemm_f32", "gemv_f32", "gemv_i8", "info", "pack_i8", "swiglu",
]
