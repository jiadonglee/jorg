# Jorg Fitting Flow (Practical Guide)

This note explains how Jorg fits stellar parameters, with emphasis on the
`fit_stellar_parameters_autodiff` path used in Gaia RVS notebooks.

## 1. Entry Points

- `fit_stellar_parameters(...)` is the core fitting API.
- `fit_stellar_parameters_autodiff(...)` is a wrapper that:
  - forces `optimizer="jax_value_and_grad"`
  - injects strict JAX synthesis defaults
  - supports `execution_profile` (`interactive`, `debug`, `batch`)

For most strict physical gradient fitting runs, use the wrapper.

## 2. Data Handling Before Optimization

For the selected fitting windows:

1. Observed spectrum is prepared (normalization, finite-mask checks).
2. Only pixels in fitting windows are used for objective evaluation.
3. Optional per-window linear continuum adjustment is applied inside each
   objective evaluation.

The objective is computed on the masked fitting pixels, not on the full
original spectrum.

## 3. What Is Being Optimized

Jorg fits a subset of:

- `Teff`
- `logg`
- `m_h`
- `alpha_fe`

Parameters can be fixed via `fixed_params`.

Bounds are enforced during optimization. In autodiff mode, candidate values are
clipped to bounds before forward synthesis.

## 4. Objective Function

Jorg minimizes:

- `total = chi2 + prior`

Where:

- `chi2 = sum(((model_flux - obs_flux) / obs_error)^2)` over fitting pixels
- `prior` is a soft boundary regularizer that discourages edge-hugging solutions

In autodiff mode, objective evaluation also returns auxiliary payload
(`chi2`, adjusted flux), so final outputs can often be reused without an extra
full forward pass.

## 5. Optimizer Paths

## 5.1 `optimizer="jax_value_and_grad"` (autodiff)

Pipeline:

1. Build differentiable forward model:
   - atmosphere interpolation
   - chemical equilibrium
   - continuum + line opacity
   - radiative transfer
   - optional LSF / rotation broadening
2. Build `value_and_grad(objective)` with JAX.
3. Solve using one of two modes:
   - default stable mode: projected gradient with backtracking (non-jitted solver)
   - optional mode: jaxopt `LBFGSB` (when solver-jit path is enabled)

Notes:

- Experimental `objective_jit` is guarded by
  `value_and_grad_enable_experimental_objective_jit`.
- If experimental JIT tracing fails, code attempts fallback to non-jitted
  objective and clears affected caches.

## 5.2 `optimizer="jax_surrogate"`

Pipeline:

1. Build local linear surrogate around current center via finite-difference
   Jacobian of flux wrt parameters.
2. Solve surrogate subproblem with L-BFGS-B.
3. Validate step on true forward model with conservative line search.
4. Repeat outer iterations.

This path is often more robust when strict autodiff steps are too conservative
for a specific star/window setup.

## 5.3 `optimizer="coordinate"`

Coordinate search with shrinking step sizes. Slower but simple fallback.

## 6. Why `converged=False` Can Still Look Good

`converged` is a solver-stop flag, not a visual-quality flag.

Common reasons for `converged=False` with acceptable-looking spectra:

- Iteration budget reached before formal tolerance was met.
- Gradient-scale tolerance not reached, even though residuals improved.
- Surrogate/autodiff step logic accepted useful improvements but did not hit
  the code path that marks formal convergence.

Therefore, evaluate fit quality with:

- `mse_window_improvement_pct`
- `reduced_chi2_fit`
- residual plots in fitting windows

Do not treat `converged` as the only acceptance criterion.

## 7. CPU/Memory Cost Drivers

Largest runtime contributors in strict physical fitting are usually:

- first-call JAX compilation
- chemical equilibrium
- repeated forward evaluations per solver step

Ways to keep load controlled:

1. Run in-process to reuse JAX compile cache between stars.
2. Restrict to target stars (`TARGET_SOURCE_IDS`) while debugging.
3. Keep low thread counts in subprocess mode (`OMP_NUM_THREADS`, etc.).
4. Start with fast attempt profiles, then fallback to robust profiles only when
   quality gates fail.

## 8. Recommended Notebook Strategy

For practical RVS fitting:

1. Fast attempt:
   - autodiff (`jax_value_and_grad`)
   - tight runtime budget
2. Quality gate:
   - require minimum `mse_window_improvement_pct`
3. Rescue attempt:
   - `jax_surrogate` fallback for stars where fast autodiff does not move

This gives a better speed/stability tradeoff than relying on `converged` alone.

