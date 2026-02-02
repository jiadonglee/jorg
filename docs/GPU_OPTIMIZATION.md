# Jorg GPU Usage

This document summarizes GPU usage for Jorg. Core synthesis remains CPU‑first
for chemical equilibrium (SciPy solver), while JAX accelerates array‑heavy steps
and radiative transfer when GPU is available.

## Key File
- `src/jorg/gpu.py` — device detection, dtype selection, and timing helpers.

## Installation
```bash
pip install -e .
pip install -e ".[gpu]"  # CUDA-enabled JAX
```

## Checking GPU Status
```python
from jorg import init_jax, is_gpu_available

init_jax(verbose=True)
print(f"GPU available: {is_gpu_available()}")
```

## Notes
- JAX uses float32 by default on GPU for speed; float64 on CPU for accuracy.
- If you need full float64 on GPU, adjust JAX config in `gpu.py`.

## Troubleshooting
```python
import jax
print(jax.devices())  # Should show 'gpu' or 'cuda'
```
