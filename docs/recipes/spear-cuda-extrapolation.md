# SuperSpear CUDA, extrapolated from a CPU box

`tensorfold bench-cuda` needs an NVIDIA GPU. This doc is what a **CPU-only instance** can say
instead: everything that runs without a GPU was run and verified here, and the numbers the
notebook bench would print are extrapolated from those CPU measurements. Raw JSON from this
instance: [`spear-torch-cpu-results.json`](spear-torch-cpu-results.json), measured with
[`tools/bench_spear_torch_cpu.py`](../../tools/bench_spear_torch_cpu.py).

Host: 2-vCPU Intel Xeon @ 2.60 GHz (AVX-512 F + VNNI), 3 GB RAM, **no GPU**, torch 2.14 (CPU).
DRAM stream copy measured at 13.6 GB/s. Same instance class as the
[CPU run](spear-cpu.md) whose committed numbers this also re-measures.

## Verified here, not extrapolated

- `pytest`: **146 passed, 26 skipped** with CPU torch installed. Among the passing ones:
  `torch_ops` formulas match the numpy champions to <2e-5; the **GPU engine** (`engine.cuda_spear`)
  runs on CPU torch and drafted output is **byte-identical** to serial (also re-checked by the
  bench below); `bench-cuda` without an NVIDIA GPU prints its note and **exits 2**, cleanly.
- `tensorfold bench-cpu` re-run against [`spear-cpu-results.json`](spear-cpu-results.json):
  the ratios hold on a fresh instance (noise on a shared vCPU is in the absolutes, not the claims):

  | Claim | committed | re-run |
  | --- | ---: | ---: |
  | GELU `fast` vs libm | ×25.7 | ×25.0 |
  | SiLU `fast` vs libm | ×6.3 | ×6.4 |
  | VNNI int8 GEMV vs OpenBLAS | ×2.29 | ×3.54 |
  | 768×4 decode, int8+alg vs fp32 exact | ×1.24 | ×1.34 |
  | drafts byte-identical | yes | yes |

- `tools/bench_spear_torch_cpu.py`: the GPU bench's **same code** on CPU torch — champion
  formulas through `torch_ops` (the Triton fallback), and `tiny_gpu`/`generate` with the keyed
  sampler and session n-gram drafts. Same JSON schema as `bench-cuda`, so the two files can be
  diffed row by row.

## What the CPU measurements say

Three facts calibrate the extrapolation:

1. **The tiny 768×4 forwards are pure weight-stream.** `swiglu-768x4` holds 37.7M matmul
   weights = 151 MB/token fp32. Measured forward: 11.1 ms ≈ 13.6 GB/s — exactly the box's DRAM
   rate (`gpt2-768x4`: 113 MB/token, 6.9 ms ≈ 16.5 GB/s; read-only streams beat copy). The
   128-wide net is the opposite: 1.3 MB/token cannot fill 0.88 ms — it is host/dispatch bound
   (~75 eager ops at ~11 µs each, sampling itself measures 4-6 µs).
2. **Algebra alone loses in a framework; only fusion wins.** On CPU torch the eager champions
   run *slower* than `F.gelu`/`F.silu` (speedups 0.13-1.70): each formula is several eager ops
   against ATen's vectorized kernels. And the SwiGLU fallback is **not fused** on CPU (two eager
   ops on both sides), so `speedup_exact` measures 0.94-1.01 — the no-fuse floor. The GPU win
   has to come from the Triton kernels being one pass with one write, not from the formulas.
3. **The champion formulas' accuracy is device-independent** (same polynomial): the MSE columns
   of `bench-cpu` carry over to the GPU rows unchanged.

## Extrapolated `bench-cuda` numbers (notebook GPU)

Model, per decode token: `t = max(host chain, bytes_fp16 / (η·B)) + sync`, where the host chain
is the eager Python loop (~8-20 µs per op × the counted ops: ~75 for the 2-layer net, ~150 for
the 4-layer ones), `sync` is the sampler's per-token device→host copy (20-100 µs), `η` is the
fraction of peak bandwidth a batch-1 GEMV stream achieves (0.5-0.7), and `B` is 256 GB/s
(RTX 4060 Laptop) or 512 GB/s (RTX 4090 Laptop). The traffic per tiny token is 0.7 MB fp16
(128-wide), 75.5 MB (`swiglu-768x4`), 56.6 MB (`gpt2-768x4`) — below the host chain at every
size here, so **all three stay launch-bound on GPU**, unlike on this CPU where 768×4 is
bandwidth-bound.

### `activations[]` (n = 2²², fp16)

| Field | Expect |
| --- | --- |
| `ns_elem_torch` | 0.02-0.05 ns/elem (traffic floor 4 B/elem at η·B) |
| `speedup_alg`, `speedup_fast` | **0.8-1.2** — both at the bandwidth floor; the CPU ×25 was libm `erf` vs ALU, and the GPU SFU computes `exp`/`erf` cheaply. ≤1 here is the paper's GPU result, not a bug. |
| `mse_alg`, `mse_fast` | the CPU table's MSEs, unchanged |

### `swiglu[]` — the row that can win

| Row | `speedup_exact` | `speedup_alg` / `speedup_fast` |
| --- | ---: | ---: |
| (1, 2048) · (1, 4096) · (1, 11008) | 1.3-2.0 (two launches + two passes → one; launch-bound) | 1.5-2.2 |
| (128, 4096) | 1.4-1.8 (traffic 10 B → 6 B per element pair) | 1.5-2.3 |

`speedup_exact` is the fuse (bandwidth + launches) and should be >1 before any algebra;
`speedup_alg`/`speedup_fast` add the no-`exp` formula inside that one kernel. A measured
`speedup_exact` > 1 is the green light for dropping `silu="silu_alg"` into the CUDA engines'
`_bsilu` (with the re-run of the row-invariance tests the CUDA recipe describes).

### `tiny[]` — decode tok/s, greedy, 32 new tokens

| Model | CPU torch (measured) | RTX 4060 Laptop | RTX 4090 Laptop |
| --- | ---: | ---: | ---: |
| `swiglu-128` | 1140 | 600-1400 | 650-1500 |
| `swiglu-768x4` | 88-99 | 350-750 | 400-800 |
| `gpt2-768x4` | 127-147 | 350-750 | 400-800 |

Rows within one model (`fp16-torch`, `fp16-spear-alg`, `fp16-spear-fast`) land within ±10% of
each other at these sizes — the champion choice is not what moves tiny decode; the host chain
is. These are single-stream eager-loop numbers; `torch.compile` or CUDA graphs would break the
launch roofline (a later lever, not this bench).

**On the draft rows.** In this tiny engine every drafted token is verified with the same forward
a serial step would take (the loop in `engine/cuda_spear.py`), so `fp16-spear-alg-ngram` rows sit
within run noise of the serial rows — including on the CPU rung, where they moved 768×4 by
±13% between runs. The n-gram rows demonstrate the **contract** (`drafts_byte_identical: true`,
accept rates 44-100% on this prompt), not a speedup. The real engines' draft wins come from
wide verify passes (the lane engine verifies up to 128 rows in one pass), which the tiny models
do not exercise.

## Checking this later on a real notebook GPU

Run `tensorfold bench-cuda` (and `--quick` on tight VRAM) and diff against the tables above:

- `activations[].speedup_*` ≤ 1 is expected; a *large* win there would be surprising.
- `swiglu[].speedup_exact` > 1 is the measurement that matters; below 1 means the fuse did not
  pay (check that Triton compiled — `info` says `triton` or `torch-fallback`; the fallback is
  unfused and returns the CPU rung's ≈1.0).
- `tiny[]` should land in the ranges above; far above them means the host chain got cheaper than
  assumed (good), far below means per-launch overhead is higher than 20 µs (netbooks, throttling).
- `drafts_byte_identical` must be `true`.
