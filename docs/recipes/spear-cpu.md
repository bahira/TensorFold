# SuperSpear CPU kernels

TensorFold's Mac and CUDA engines need Apple Silicon or an NVIDIA GPU. This package is the
path that still runs on a CPU box: SuperSpear's champion algebraic activations, compiled as
AVX-512 C, a VNNI int8 GEMV, and TensorFold's keyed sampler plus session n-gram drafts.

The formulas come from [bahira/superspear](https://github.com/bahira/superspear) (MIT). TensorFold
does not re-evolve them; it deploys the ledger champions.

```bash
tensorfold bench-cpu --model tiny          # synthetic GPT-2 + SwiGLU, including a 768×4 stand-in
tensorfold bench-cpu --model distilgpt2    # Hugging Face DistilGPT2, when safetensors is reachable
```

Raw numbers from this instance: [`spear-cpu-results.json`](spear-cpu-results.json).

## Operating points

| Slot | GELU | SiLU | What it is |
| --- | --- | --- | --- |
| `exact` | `0.5 x (1+erf(x/√2))` | `x / (1+e^{-x})` | quality reference, libm |
| `alg` | evolved clamp-relu (no erf) | `x (0.501 + 0.587 x / (0.815+√(1+x²)))` | deploy point |
| `fast` | `1.010719 relu(x) − 0.057684` | `1.016356 relu(x) − 0.15849` | SuperSpear fast slot |

`fast` is the measured cheap slot in the SPEAR ledger (GELU ×6.49, SiLU ×5.43, scalar gcc -O2).
It is a real speed/accuracy trade, not a bit-identical replacement. Prefill GEMM stays on
numpy/OpenBLAS; decode GEMVs go through AVX-512 FMA or VNNI int8.

Drafts use `SessionNGram` the same way the GPU engines do: a draft is kept only when it equals
the keyed serial sample, so `drafts=4` writes the same tokens as `drafts=0`.

## Measured on this instance

Host: 2-core Intel Xeon @ 2.60 GHz, AVX-512 F + VNNI, 3.8 GB RAM, **no GPU**. Hugging Face was
unreachable, so the 82M-class run is a DistilGPT2-shaped synthetic (4 layers, d=768, inner=3072,
vocab=512) rather than the HF checkpoint. Activation ns/elem is the median of 9×20 passes on a
2²⁰-element buffer. Decode tok/s is the median of 3 greedy runs.

### Activations (AVX-512 champion vs libm exact)

| Op | exact ns/elem | alg ns/elem | fast ns/elem | × alg | × fast | MSE alg | MSE fast |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GELU | 15.64 | 0.70 | 0.61 | **22.3** | **25.7** | 1.0e-3 | 5.4e-3 |
| SiLU | 4.03 | 0.65 | 0.64 | **6.2** | **6.3** | 4.9e-4 | 8.9e-3 |
| sigmoid | 3.94 | — | 0.68 | — | **5.8** | — | 1.1e-3 |

Vectorized ALU vs scalar `erff`/`expf` is a bigger jump than SPEAR's scalar-vs-scalar ledger
(GELU fast ×6.49). Same formulas, wider SIMD.

### Matmul and FFN

| Kernel | ns | vs numpy |
| --- | ---: | ---: |
| numpy OpenBLAS GEMV 3072×768 | 171 | ×1.00 |
| AVX-512 FMA GEMV | 240 | ×0.71 (OpenBLAS wins fp32) |
| **VNNI int8 GEMV** | **75** | **×2.29** (cos 0.99994 vs fp32) |
| MLP 768→3072 GELU→768, fp32+exact GELU | 425 | ×1.00 |
| **same MLP, VNNI + GELU-alg** | **181** | **×2.35** |

Hand-rolled fp32 GEMV does not beat OpenBLAS. Int8 VNNI does, because it moves 4× less weight.

### Decode, greedy, 16 new tokens

| Model | fp32 exact | int8 + GELU/SiLU-alg | + n-gram drafts (k=4) |
| --- | ---: | ---: | ---: |
| tiny GPT-2 (2×128) | 137 tok/s | 138 tok/s | 142 tok/s (accept 100%) |
| tiny SwiGLU (2×128) | 140 tok/s | 131 tok/s | 122 tok/s (accept 100%) |
| **GPT-2 768×4 (DistilGPT2-shaped)** | **67.1 tok/s** | **83.0 tok/s (×1.24)** | **96.8 tok/s (×1.44)** |

The 128-wide nets are Python-bound: swapping GELU does not move tok/s. Once the FFN is 768×3072,
VNNI shows up in the end-to-end number, and n-gram drafts add another 17% on this prompt (32% of
proposed tokens accepted; output **byte-identical** to serial).

## Honest limits

The same ones SPEAR's own paper records:

- Swapping a Python GELU for the algebraic form inside a fused framework kernel is often a
  *slowdown*. The wins here are native AVX-512 (no exp/erf) and VNNI int8 on the GEMV, not a
  torch monkeypatch.
- GEMM dominates a transformer. Activation-only end-to-end gains stay small unless the matmul
  itself is quantized. The 128-wide rows above are that case.
- N-gram drafts on CPU still run one verify GEMV per proposed token. They pay when the n-gram
  hits, and they never change the bytes.
- `alg` / `fast` are not bit-identical to `erf`/`exp`. MSE on N(0,1.5) is in the table. Use
  `exact` when you are checking quality; use `alg` when you are decoding.

On a notebook NVIDIA GPU the same champions run as Triton: [SPEAR CUDA](spear-cuda.md).
Elementwise vs `F.silu` is usually a wash there; fused SwiGLU is the measurement that can win.
