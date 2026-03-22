# Current Work Status

- Updated at (UTC): 2026-02-26T00:00:00Z
- Workspace: `/Users/jdli/Project/jorg/jorg`
- Scope: End-to-end autodiff hardening (strict-by-default, JAX-by-default)

## Implemented

1. Switched public synthesis defaults to JAX strict path.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/synthesis.py`
- Changes:
  - `synthesize(..., engine="jax", ce_solver="jax")` is now default.
  - `synthesize_spectrum(..., engine="jax", ce_solver="jax")` is now default.
  - `synth(..., engine="jax", ce_solver="jax")` is now default.
  - Legacy remains available via explicit `engine="legacy"`.

2. Enabled strict autodiff by default in `synthesize_jax`.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/synthesis.py`
- Changes:
  - `autodiff_strict=True` default.
  - Strict mode now hard-requires:
    - `line_backend="jax"` when line list is non-empty.
    - `ce_jit=True` (rejects `ce_jit=False`).
  - `line_backend="numpy"` remains available only with `autodiff_strict=False`.
  - JAX CE layer solve now defaults `ce_warm_start=False` to prevent warm-start
    drift on deep layers and improve agreement with Korg.jl.

3. Removed continuum partition-function fallback that broke gradient fidelity.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/exact_physics_continuum.py`
- Changes:
  - Removed `stop_gradient/device_get`-driven fallback-to-ones behavior.
  - Added cached logU(H I / He I) table extraction and pure JAX runtime interpolation.
  - He- free-free term now uses JAX-native table evaluation in batch path.

4. Added JAX-native John (1994) He- free-free interpolator.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/continuum/helium.py`
- Changes:
  - New `_helium_free_free_john1994_jax(...)` for on-device interpolation.

5. Hardened non-tracer-safe atmosphere branch.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/atmosphere.py`
- Changes:
  - `grid_mode="cool_dwarf"` + `return_dense=True` now hard-fails with remediation guidance.

6. Centralized strict config validation for value-and-grad fitting.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/fit/stellar_parameters.py`
- Changes:
  - Added `_validate_jax_value_and_grad_synthesize_kwargs(...)`.
  - `optimizer="jax_value_and_grad"` now hard-requires:
    - `line_backend="jax"`
    - `autodiff_strict=True`
    - `ce_jit=True`

7. Added/updated tests for strict behavior and end-to-end gradients.
- Files:
  - `/Users/jdli/Project/jorg/jorg/tests/test_synthesis_jax_engine.py`
  - `/Users/jdli/Project/jorg/jorg/tests/test_synthesis_jax_gradients.py`
  - `/Users/jdli/Project/jorg/jorg/tests/test_stellar_parameter_fitting.py`
- Coverage additions:
  - strict reject `ce_jit=False`
  - hard-fail cool-dwarf dense interpolation
  - default `synthesize()` path equals explicit JAX path
  - finite gradients for `(Teff, logg, m_h, alpha_fe)` with/without lines
  - finite objective output for `optimizer="jax_value_and_grad"` pipeline

8. Reduced overhead in the strict autodiff stellar-parameter fitter.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/fit/stellar_parameters.py`
- Changes:
  - `fit_stellar_parameters_autodiff(..., execution_profile=...)` now supports:
    - `"debug"`: no JIT
    - `"interactive"`: atmosphere/objective JIT on, solver JIT off (new default)
    - `"batch"`: atmosphere/objective/solver JIT on
  - Real physical `jax_value_and_grad` runs currently force objective/solver JIT off at
    runtime because nested tracing through atmosphere and CE caches is still unstable.
    The profile API remains in place, and atmosphere interpolation JIT still applies.
  - The `jax_value_and_grad` path now prefilters the line list once to the fit span
    plus `line_buffer`, including aligned filtering of `line_loggf_deltas`.
  - Objective compilation is cached across repeated same-shape fits and the warmup
    compile is done explicitly before the solver starts.
  - Default statmech data now reuses synthesis-level singleton helpers so CE and
    partition caches can hit across repeated fits.
  - The final best-fit flux/chi2 reuse the JAX path result instead of re-running
    the legacy forward model.

9. Added static line-tensor packing cache for the JAX line-opacity backend.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/opacity/line_opacity_jax.py`
- Changes:
  - Added LRU caching for `LineTensorPack` keyed by `(id(linelist), species_layout signature)`.
  - Repeated fits with the same filtered linelist now reuse the packed tensors instead
    of rescanning and repacking line objects each call.

## Notes

- Legacy compatibility path is preserved but no longer default.
- Strict mode is fail-fast by design to prevent silent non-differentiable fallbacks.
- The strict autodiff fitter is now speed-oriented by default (`execution_profile="interactive"`),
  while `execution_profile="debug"` preserves the old no-JIT debugging behavior.
