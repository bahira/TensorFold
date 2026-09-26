"""SuperSpear Hall-of-Fame formulas used by the CPU kernels.

Source: https://github.com/bahira/superspear ``spear-hall-of-fame.json`` / ``ledger/*.json``.
Each record is a closed-form replacement for a production LLM kernel. ``fast`` is the
measured cheap slot (relu / rational); ``alg`` is the evolved algebraic with no
exp/erf/tanh; ``exact`` is the mathematical definition (the quality reference).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INV_SQRT2 = 0.7071067811865476


def _erf(x: np.ndarray) -> np.ndarray:
    """Abramowitz-Stegun 7.1.26, same constants as SuperSpear's JS/C backends. numpy 2 has no ``np.erf``."""

    s = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-ax * ax)
    return s * y


@dataclass(frozen=True)
class Champion:
    name: str
    formula: str
    source: str
    measured_speedup: float | None
    mse_vs_exact: float | None
    op: int


# ctypes op ids, must match kernels.h
GELU_EXACT, GELU_ALG, GELU_FAST = 0, 1, 2
SILU_EXACT, SILU_ALG, SILU_FAST = 3, 4, 5
SIGMOID_EXACT, SIGMOID_FAST = 6, 7

CHAMPIONS: dict[str, Champion] = {
    "gelu_alg": Champion(
        "gelu_alg",
        "0.997729*x*min(1.002, relu(0.306923*x + 0.501)) - 0.004004",
        "superspear validation/gelu_policy_generated.py (evolved algebraic GELU)",
        6.57, 5.3e-4, GELU_ALG,
    ),
    "gelu_fast": Champion(
        "gelu_fast",
        "1.010719*relu(x) - 0.057684",
        "superspear ledger/gelu.json fast slot, ×6.49 gcc -O2",
        6.4889, 2.97e-3, GELU_FAST,
    ),
    "silu_alg": Champion(
        "silu_alg",
        "x*(0.501 + 0.587*x/(0.815 + sqrt(1+x^2)))",
        "superspear README SiLU/Swish algebraic, ×2.43 modelled",
        2.43, 7.8e-4, SILU_ALG,
    ),
    "silu_fast": Champion(
        "silu_fast",
        "1.016356*relu(x) - 0.15849",
        "superspear ledger/silu.json fast slot, ×5.43 gcc -O2",
        5.4343, 7.34e-3, SILU_FAST,
    ),
    "sigmoid_fast": Champion(
        "sigmoid_fast",
        "0.605014*x/(1.24384 + abs(x)) + 0.5",
        "superspear ledger/sigmoid.json fast slot, ×1.90 gcc -O2",
        1.9032, 4.87e-4, SIGMOID_FAST,
    ),
}


def gelu_exact(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (0.5 * x * (1.0 + _erf(x * np.float32(INV_SQRT2)))).astype(np.float32)


def gelu_alg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    t = np.clip(np.maximum(0.306923 * x + 0.501, 0.0), 0.0, 1.002)
    return (0.997729 * x * t - 0.004004).astype(np.float32)


def gelu_fast(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.010719 * np.maximum(x, 0.0) - 0.057684).astype(np.float32)


def silu_exact(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def silu_alg(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (x * (0.501 + 0.587 * x / (0.815 + np.sqrt(1.0 + x * x)))).astype(np.float32)


def silu_fast(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.016356 * np.maximum(x, 0.0) - 0.15849).astype(np.float32)


def sigmoid_exact(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def sigmoid_fast(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return (0.605014 * x / (1.24384 + np.abs(x)) + 0.5).astype(np.float32)


NUMPY_OPS = {
    GELU_EXACT: gelu_exact,
    GELU_ALG: gelu_alg,
    GELU_FAST: gelu_fast,
    SILU_EXACT: silu_exact,
    SILU_ALG: silu_alg,
    SILU_FAST: silu_fast,
    SIGMOID_EXACT: sigmoid_exact,
    SIGMOID_FAST: sigmoid_fast,
}

OP_BY_NAME = {
    "gelu_exact": GELU_EXACT,
    "gelu_alg": GELU_ALG,
    "gelu_fast": GELU_FAST,
    "silu_exact": SILU_EXACT,
    "silu_alg": SILU_ALG,
    "silu_fast": SILU_FAST,
    "sigmoid_exact": SIGMOID_EXACT,
    "sigmoid_fast": SIGMOID_FAST,
}


def apply_numpy(op: int, x: np.ndarray) -> np.ndarray:
    return NUMPY_OPS[op](x)
