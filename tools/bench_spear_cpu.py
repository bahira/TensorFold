#!/usr/bin/env python3
"""Benchmark SuperSpear champion kernels and tiny CPU models. See tensorfold bench-cpu."""

from tensorfold.kernels.spear.v1.bench import main

if __name__ == "__main__":
    raise SystemExit(main())
