# Current Work Status

- Updated at (UTC): 2026-02-23T17:02:19Z
- Workspace: `/Users/jdli/Project/jorg/jorg`
- Scope: Gradual autodiff upgrade for synthesis pipeline (continuum-first, legacy default kept)

## Implemented

1. Added strict autodiff controls in JAX synthesis path.
- File: `/Users/jdli/Project/jorg/jorg/src/jorg/synthesis.py`
- New args in `synthesize_jax`: `autodiff_strict`, `line_backend`
- Behavior:
  - `autodiff_strict=True` + non-empty `linelist` requires `line_backend="jax"`
  - `line_backend="numpy"` with lines emits one-time `RuntimeWarning` in non-strict mode
  - Legacy engine strips JAX-only kwargs (`autodiff_strict`, `line_backend`) to preserve compatibility

2. Added experimental JAX line-opacity backend.
- New file: `/Users/jdli/Project/jorg/jorg/src/jorg/opacity/line_opacity_jax.py`
- Exported in: `/Users/jdli/Project/jorg/jorg/src/jorg/opacity/__init__.py`
- Design:
  - Static linelist packing in Python/NumPy
  - JAX core accumulation via `vmap` + `lax.scan`
  - Soft window gate (sigmoid) for differentiable windowing
  - Partition function usage moved to JAX-side interpolation (cached static logU tables + runtime JAX interp)
  - `microturbulence_kms` kept as JAX scalar in runtime core (no `float(...)` cast), enabling vmic gradients

3. Added baseline collection script and generated baseline artifacts.
- Script: `/Users/jdli/Project/jorg/jorg/examples/collect_autodiff_baseline.py`
- Outputs:
  - `/Users/jdli/Project/jorg/jorg/output/autodiff_baseline/baseline.json`
  - `/Users/jdli/Project/jorg/jorg/output/autodiff_baseline/baseline.md`

4. Updated docs/config.
- README autodiff mode notes: `/Users/jdli/Project/jorg/jorg/README.md`
- Added pytest markers in config: `/Users/jdli/Project/jorg/jorg/pyproject.toml`

## Validation Status

Executed test suites:

```bash
pytest tests/test_synthesis_jax_engine.py tests/test_synthesis_jax_gradients.py tests/test_continuum_jax_fast.py -q
```

Result:
- `22 passed`

Baseline summary (from `baseline.json`):
- Gradient finite: `A_X[25] = 2.107842921326351e+12`
- Legacy vs JAX continuum relative error:
  - median: `1.0663609784091617e-05`
  - p95: `1.7075289599806363e-05`
  - max: `1.7595493595135004e-05`

## Current Git Working Tree

Modified tracked files:
- `/Users/jdli/Project/jorg/jorg/README.md`
- `/Users/jdli/Project/jorg/jorg/pyproject.toml`
- `/Users/jdli/Project/jorg/jorg/src/jorg/opacity/__init__.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/synthesis.py`

Untracked files:
- `/Users/jdli/Project/jorg/jorg/examples/collect_autodiff_baseline.py`
- `/Users/jdli/Project/jorg/jorg/src/jorg/opacity/line_opacity_jax.py`

Note:
- `tests/` is git-ignored in this repository, so local test edits are not shown by `git status`.

## Suggested Next Steps

1. If you want to commit now, ensure intended test-file changes are handled despite `.gitignore`.
2. Add CI target(s) for fast P0/P1 checks and optional marker-based line-backend tests.
3. Expand line-backend parity cases beyond weak-line scenario.
