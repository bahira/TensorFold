#!/usr/bin/env python3
"""Benchmark SuperSpear CUDA kernels. See tensorfold bench-cuda."""

from tensorfold.kernels.spear.v1.bench_cuda import main

if __name__ == "__main__":
    raise SystemExit(main())
