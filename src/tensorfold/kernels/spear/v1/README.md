# spear/v1

SuperSpear Hall-of-Fame GELU, SiLU and sigmoid, compiled as AVX-512 C, plus a VNNI int8 GEMV
for CPU decode. Formulas: [bahira/superspear](https://github.com/bahira/superspear). Method and
numbers: [the SPEAR CPU recipe](../../../../../docs/recipes/spear-cpu.md).

```python
from tensorfold.kernels.spear import v1 as spear
y = spear.act("gelu_alg", x)          # evolved algebraic, no erf
y = spear.gemv_i8(spear.pack_i8(W), x)  # VNNI int8

from tensorfold.kernels.spear.v1 import cuda as spear_cuda
y = spear_cuda.swiglu(gate, up, silu="silu_alg")  # Triton, notebook GPU
```
