# SuperSpear CUDA kernels (notebook NVIDIA GPU)

The 27B / Flash Next CUDA engines need a Spark. This package is what you can run on a
**laptop GPU**: the same SuperSpear champions, compiled by Triton (no nvcc), fused SwiGLU,
and tiny GPT-2 / SwiGLU decode with TensorFold's keyed sampler and n-gram drafts.

```bash
# on the notebook, with PyTorch+CUDA
pip install -e ".[test]"
pip install torch --index-url https://download.pytorch.org/whl/cu128   # pick your CUDA

nvidia-smi
tensorfold bench-cuda           # full
tensorfold bench-cuda --quick   # skip 768-wide nets if VRAM is tight
```

It refuses to start without `torch.cuda.is_available()`. Triton compiles on first use
into `~/.cache/tensorfold/triton`. If Triton is missing, the same algebra still runs as
vectorized PyTorch (slower, but the decode contract stays testable). Raw JSON:
[`spear-cuda-results.json`](spear-cuda-results.json) (written on the machine that ran it).

## Why this, not the 27B engine

| VRAM | What fits | What does not |
| --- | --- | --- |
| 4 GB | DistilGPT2, SmolLM-135M, the synthetic 768×4 net in this bench | Qwen3.8-27B, Flash Next, GLM |
| 8 GB | Qwen2.5-0.5B, Llama-3.2-1B, Phi-3-mini 4-bit | same |
| 12–16 GB | Qwen2.5-3B, Llama-3.2-3B | 27B 4-bit is tight |
| Spark 128 GB | the CUDA families TensorFold already ships | — |

The interesting notebook measurement is **not** "can we serve 27B". It is:

1. **Elementwise vs `F.silu` / `F.gelu`.** On GPU the SFU is fast. SPEAR's own paper found that
   swapping GELU inside a framework often *loses*. Expect ~×0.8–×1.2 here, not the CPU ×25.
2. **Fused SwiGLU.** `silu(gate) * up` in one write vs `F.silu(gate) * up` (two kernels, extra
   activation traffic). This is the op Qwen/LLaMA/Mistral and TensorFold's CUDA engines run
   every layer (`_bsilu` in `families/qwen4_exp/cuda/glue.py`). A fused champion that wins
   here is a drop-in candidate for those engines.
3. **Decode tok/s** of a DistilGPT2-shaped net (4 × 768, SwiGLU or GELU) with `exact` /
   `alg` / `fast` and with n-gram drafts. Drafts stay byte-identical to serial.

## Operating points (same formulas as CPU)

| Slot | What the GPU runs |
| --- | --- |
| `exact` | `F.gelu` / `F.silu` / fused `silu(g)*u` with `exp` |
| `alg` | evolved algebraic, no exp/erf (deploy point) |
| `fast` | SuperSpear relu / rational slot |

Do **not** put `fast` or `alg` on a MoE router sigmoid: a wrong expert is not a cheap
approximation, it is a different model. Attention-gate sigmoid is the safer fuse.

## What to look at in the JSON

- `activations[].speedup_*` — if this is ≤1, that is the paper's GPU result, not a bug.
- `swiglu[].speedup_exact` — fused exact vs `F.silu*u`. A win here is bandwidth, not algebra.
- `swiglu[].speedup_alg` / `speedup_fast` — algebra on top of the fuse.
- `tiny[]` where `model` is `swiglu-768x4` — the number that should move if SwiGLU is the
  limit. `swiglu-128` is launch-overhead bound, like the CPU 128-wide rows.

## Drop-in for TensorFold's CUDA engines (later, on a Spark)

`kernels/spear/v1/cuda.py` `swiglu(..., silu="silu_alg")` is the same arithmetic as replacing
`_bsilu` in `qwen4_exp/cuda/glue.py` / `qwen3_5` Metal `bsilu`. Do that only after this bench
shows a fused win, and re-run the row-invariance tests: algebraic SiLU is not bit-identical
to `x/(1+exp(-x))`, so drafted==serial still holds (same kernel both paths) but hashes vs
the current CUDA engine will change.
