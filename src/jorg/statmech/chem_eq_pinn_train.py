"""Training utilities for chemical-equilibrium PINN experiments.

This module provides the full training pipeline for
:class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN`, including:

* **Label generation** – :func:`generate_training_data` calls a trusted
  chemical-equilibrium solver (JAX or SciPy backend) to produce
  ``(neutral_fractions, n_e)`` ground-truth labels for randomly sampled
  stellar-atmosphere conditions.
* **Optimizer construction** – :func:`create_optimizer` builds an AdamW
  optimizer with a linear warm-up followed by a cosine-decay schedule.
* **Two-phase training** – :func:`train_pinn` first performs a supervised
  phase that fits the network to solver labels, then a physics fine-tuning
  phase that minimises element-conservation and charge-neutrality residuals.
* **JIT-compiled training steps** –
  :func:`make_supervised_train_step` and :func:`make_physics_train_step`
  return pre-compiled step functions suitable for tight training loops.

Typical usage
-------------
>>> from jorg.statmech.chem_eq_jax import prepare_chem_eq_data
>>> from jorg.statmech.chem_eq_pinn_models import ChemicalEquilibriumPINN
>>> from jorg.statmech.chem_eq_pinn_train import generate_training_data, train_pinn
>>> chem_data = prepare_chem_eq_data(...)
>>> train_data = generate_training_data(1024)
>>> model = ChemicalEquilibriumPINN()
>>> params, history = train_pinn(model, chem_data, train_data, rng_key=jax.random.PRNGKey(0))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

import jax
import jax.numpy as jnp

try:  # pragma: no cover - optional runtime dependency
    import optax
except Exception as exc:  # pragma: no cover - handled at runtime
    optax = None
    _OPTAX_IMPORT_ERROR = exc
else:
    _OPTAX_IMPORT_ERROR = None

from jorg.abundances import format_abundances

from .chem_eq_jax import ChemEqData, chemical_equilibrium_jax, prepare_chem_eq_data
from .chem_eq_pinn_loss import total_physics_loss
from .chem_eq_pinn_models import ChemicalEquilibriumPINN
from .korg_chemical_equilibrium import chemical_equilibrium as chemical_equilibrium_scipy
from .korg_equilibrium_constants import (
    create_default_partition_functions_korg as create_default_partition_functions,
)
from .proper_ionization_energies import (
    create_proper_ionization_energy_dict as create_default_ionization_energies,
)
from .species import MAX_ATOMIC_NUMBER, Species


def _require_optax() -> None:
    if optax is None:  # pragma: no cover - exercised when optax missing
        raise RuntimeError(f"optax is required for chem_eq_pinn_train: {_OPTAX_IMPORT_ERROR}")


def _safe_log10(x: jnp.ndarray, floor: float = 1e-300) -> jnp.ndarray:
    """Compute :math:`\\log_{10}(x)` with a lower-bound clamp to avoid ``-inf``.

    Parameters
    ----------
    x:
        Input array.  Values below *floor* are clamped before taking the
        logarithm.
    floor:
        Minimum value before the logarithm is applied.  Default ``1e-300``.

    Returns
    -------
    jnp.ndarray
        Element-wise :math:`\\log_{10}` of the clamped input, same shape as
        *x*.
    """
    return jnp.log10(jnp.clip(x, floor, None))


def absolute_abundances_from_ax(a_x: np.ndarray) -> np.ndarray:
    """Convert A(X) log-scale abundances to normalised number fractions.

    The standard stellar spectroscopy notation defines
    :math:`A(X) = \\log_{10}(N_X / N_H) + 12`, so that :math:`A(H) = 12`
    by convention.  This function converts to absolute number fractions
    :math:`f_Z = N_Z / N_{\\rm tot}` that sum to unity.

    Parameters
    ----------
    a_x:
        1-D array of length ``MAX_ATOMIC_NUMBER`` containing :math:`A(X)`
        values for elements :math:`Z = 1, \\ldots, 92`.

    Returns
    -------
    np.ndarray
        Float64 array of normalised number fractions with the same length as
        *a_x* whose elements sum to 1.

    Raises
    ------
    ValueError
        If the converted values produce a non-positive (or non-finite) total,
        which would make normalisation ill-defined.
    """

    abs_abund = np.power(10.0, np.asarray(a_x, dtype=np.float64) - 12.0)
    abs_abund = np.clip(abs_abund, 0.0, None)
    total = float(np.sum(abs_abund))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("A(X) abundances produced non-positive total absolute abundance.")
    return abs_abund / total


def sample_thermodynamic_conditions(
    n_samples: int,
    rng: np.random.Generator,
    *,
    temp_range: Tuple[float, float] = (3000.0, 8000.0),
    log_n_total_range: Tuple[float, float] = (13.0, 17.0),
    metallicity_range: Tuple[float, float] = (-2.0, 0.5),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly sample stellar-atmosphere conditions for PINN training.

    Draws temperatures uniformly from *temp_range*, :math:`\\log_{10}
    n_{\\rm tot}` uniformly from *log_n_total_range*, and metallicity
    :math:`[M/H]` uniformly from *metallicity_range*.  Alpha-element
    offsets are drawn depending on metallicity (larger offsets at low
    metallicity to mimic metal-poor giant stars).  Abundance vectors are
    produced via :func:`~jorg.abundances.format_abundances` and converted to
    normalised number fractions with :func:`absolute_abundances_from_ax`.

    Parameters
    ----------
    n_samples:
        Number of thermodynamic conditions to sample.  Must be positive.
    rng:
        NumPy random-number generator (e.g. ``np.random.default_rng(0)``).
    temp_range:
        ``(T_min, T_max)`` in Kelvin for the uniform temperature draw.
        Default ``(3000.0, 8000.0)``.
    log_n_total_range:
        ``(log_n_min, log_n_max)`` in :math:`\\mathrm{cm}^{-3}` for the
        uniform density draw in log space.  Default ``(13.0, 17.0)``.
    metallicity_range:
        ``([M/H]_min, [M/H]_max)`` for the uniform metallicity draw.
        Default ``(-2.0, 0.5)``.

    Returns
    -------
    T : np.ndarray
        Float64 temperatures in Kelvin, shape ``(n_samples,)``.
    n_total : np.ndarray
        Float64 total number densities in cm\ :sup:`-3`, shape
        ``(n_samples,)``.
    abundances : np.ndarray
        Float64 normalised number fractions, shape
        ``(n_samples, MAX_ATOMIC_NUMBER)``.

    Raises
    ------
    ValueError
        If *n_samples* is not positive.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    T = rng.uniform(temp_range[0], temp_range[1], size=n_samples)
    log_n_total = rng.uniform(log_n_total_range[0], log_n_total_range[1], size=n_samples)
    n_total = np.power(10.0, log_n_total)

    metallicities = rng.uniform(metallicity_range[0], metallicity_range[1], size=n_samples)
    alpha_offsets = np.where(
        metallicities < -0.5,
        rng.uniform(0.05, 0.45, size=n_samples),
        rng.uniform(-0.1, 0.2, size=n_samples),
    )
    alpha_values = metallicities + alpha_offsets

    abundances = np.empty((n_samples, MAX_ATOMIC_NUMBER), dtype=np.float64)
    for idx in range(n_samples):
        a_x = format_abundances(
            default_metals_H=float(metallicities[idx]),
            default_alpha_H=float(alpha_values[idx]),
        )
        abundances[idx] = absolute_abundances_from_ax(np.asarray(a_x, dtype=np.float64))

    return T.astype(np.float64), n_total.astype(np.float64), abundances


def _species_to_neutral_fractions(
    species: Dict[Species, float],
    abundances: np.ndarray,
    n_total: float,
    ne: float,
) -> np.ndarray:
    """Compute per-element neutral fractions from a species number-density map.

    The neutral fraction of element :math:`Z` is defined as

    .. math::

        f_{\\rm neutral,\\,Z} = \\frac{n(Z,\\,0)}{A_Z \\cdot (n_{\\rm tot} - n_e)}

    where :math:`n(Z, 0)` is the number density of the neutral atom,
    :math:`A_Z` is the elemental number fraction, and
    :math:`n_{\\rm tot} - n_e` is the total nucleon number density.
    Elements with zero nucleon density are assigned a neutral fraction of 0.

    Parameters
    ----------
    species:
        Mapping from :class:`~.species.Species` to number density
        (cm\ :sup:`-3`) as returned by the SciPy chemical-equilibrium solver.
    abundances:
        Elemental number fractions :math:`A_Z`, shape ``(MAX_ATOMIC_NUMBER,)``.
    n_total:
        Total number density in cm\ :sup:`-3` (atoms + ions + electrons).
    ne:
        Electron number density in cm\ :sup:`-3`.

    Returns
    -------
    np.ndarray
        Float64 neutral fractions clipped to ``[0, 1]``, shape
        ``(MAX_ATOMIC_NUMBER,)``.
    """
    n0 = np.array(
        [species[Species.from_atomic_number(Z, 0)] for Z in range(1, MAX_ATOMIC_NUMBER + 1)],
        dtype=np.float64,
    )
    n_nuc = abundances * (n_total - ne)
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.where(n_nuc > 0, n0 / n_nuc, 0.0)
    return np.clip(f, 0.0, 1.0)


def generate_training_data(
    n_samples: int,
    *,
    ionization_energies: Optional[Dict[int, Tuple[float, float, float]]] = None,
    partition_funcs: Optional[Dict[Species, object]] = None,
    log_equilibrium_constants: Optional[Dict] = None,
    solver_backend: str = "jax",
    rng_seed: int = 0,
    max_attempt_factor: int = 5,
    return_jax_arrays: bool = False,
) -> Dict[str, np.ndarray]:
    """Generate supervised training labels from a trusted chemical-equilibrium solver.

    Randomly samples *n_samples* stellar-atmosphere conditions via
    :func:`sample_thermodynamic_conditions`, solves chemical equilibrium for
    each one with either the JAX or SciPy backend, and collects the results
    into a dict of stacked arrays.  Samples for which the solver fails or
    returns non-finite/unphysical values are silently discarded; up to
    ``n_samples * max_attempt_factor`` candidates are tried before giving up.

    Parameters
    ----------
    n_samples:
        Number of successfully solved atmospheres to return.  Must be
        positive.
    ionization_energies:
        Mapping from atomic number :math:`Z` to a tuple of the first three
        ionization energies in eV.  Defaults to
        ``create_default_ionization_energies()`` when ``None``.
    partition_funcs:
        Mapping from :class:`~.species.Species` to partition-function
        callables.  Defaults to ``create_default_partition_functions()``
        when ``None``.
    log_equilibrium_constants:
        Pre-computed :math:`\\log K` tables for molecular equilibria.  Passed
        directly to the solver.  The JAX solver uses an internal table when
        ``None``.
    solver_backend:
        ``"jax"`` (default) or ``"scipy"``.  The JAX backend is faster on
        accelerators; the SciPy backend provides an independent reference
        implementation.
    rng_seed:
        Integer seed for the NumPy random-number generator.  Default ``0``.
    max_attempt_factor:
        Multiplier applied to *n_samples* to determine the total number of
        candidate conditions tried.  Increase when the success rate is low
        (e.g. very extreme parameters).  Default ``5``.
    return_jax_arrays:
        When ``True``, return JAX arrays instead of NumPy arrays.  Useful
        to avoid host-device copies at the start of training.  Default
        ``False``.

    Returns
    -------
    dict
        Dataset with the following keys, each mapping to a float64 array:

        * ``"T"`` – temperatures (K), shape ``(n_samples,)``.
        * ``"log_T"`` – :math:`\\log_{10} T`, shape ``(n_samples,)``.
        * ``"n_total"`` – total number densities (cm\ :sup:`-3`), shape
          ``(n_samples,)``.
        * ``"log_n_total"`` – :math:`\\log_{10} n_{\\rm tot}`, shape
          ``(n_samples,)``.
        * ``"abundances"`` – normalised number fractions, shape
          ``(n_samples, MAX_ATOMIC_NUMBER)``.
        * ``"neutral_fractions"`` – per-element neutral fractions, shape
          ``(n_samples, MAX_ATOMIC_NUMBER)``.
        * ``"ne"`` – electron number densities (cm\ :sup:`-3`), shape
          ``(n_samples,)``.

    Raises
    ------
    ValueError
        If *n_samples* is not positive or *solver_backend* is unrecognised.
    RuntimeError
        If fewer than *n_samples* valid samples can be produced within
        ``n_samples * max_attempt_factor`` attempts.
    """

    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    if ionization_energies is None:
        ionization_energies = create_default_ionization_energies()
    if partition_funcs is None:
        partition_funcs = create_default_partition_functions()

    solver_backend = solver_backend.lower()
    if solver_backend not in {"jax", "scipy"}:
        raise ValueError("solver_backend must be 'jax' or 'scipy'.")

    chem_data = None
    if solver_backend == "jax":
        chem_data = prepare_chem_eq_data(
            ionization_energies,
            partition_funcs,
            log_equilibrium_constants,
        )

    rng = np.random.default_rng(rng_seed)
    n_candidates = n_samples * max(1, max_attempt_factor)

    T_samples, n_total_samples, abundance_samples = sample_thermodynamic_conditions(
        n_candidates,
        rng,
    )

    solved_T = []
    solved_n_total = []
    solved_abund = []
    solved_f = []
    solved_ne = []

    for T, n_total, abundances in zip(T_samples, n_total_samples, abundance_samples):
        model_ne = 1e-4 * n_total
        try:
            if solver_backend == "jax":
                ne, f = chemical_equilibrium_jax(
                    float(T),
                    float(n_total),
                    float(model_ne),
                    abundances,
                    ionization_energies,
                    partition_funcs,
                    log_equilibrium_constants=log_equilibrium_constants,
                    chem_data=chem_data,
                    maxiter=350,
                    tol=1e-10,
                    jit=True,
                    fallback=True,
                )
            else:
                ne, species = chemical_equilibrium_scipy(
                    float(T),
                    float(n_total),
                    float(model_ne),
                    abundances,
                    ionization_energies,
                    partition_funcs,
                    log_equilibrium_constants=log_equilibrium_constants,
                    warn_on_ne_discrepancy=False,
                )
                f = _species_to_neutral_fractions(species, abundances, float(n_total), float(ne))
        except Exception:
            continue

        if not (np.isfinite(ne) and np.all(np.isfinite(f))):
            continue
        if ne <= 0 or ne >= n_total:
            continue

        solved_T.append(float(T))
        solved_n_total.append(float(n_total))
        solved_abund.append(np.asarray(abundances, dtype=np.float64))
        solved_f.append(np.asarray(f, dtype=np.float64))
        solved_ne.append(float(ne))

        if len(solved_T) >= n_samples:
            break

    if len(solved_T) < n_samples:
        raise RuntimeError(
            f"Only generated {len(solved_T)} valid samples out of requested {n_samples}. "
            f"Try increasing max_attempt_factor (current: {max_attempt_factor})."
        )

    data = {
        "T": np.asarray(solved_T, dtype=np.float64),
        "log_T": np.log10(np.asarray(solved_T, dtype=np.float64)),
        "n_total": np.asarray(solved_n_total, dtype=np.float64),
        "log_n_total": np.log10(np.asarray(solved_n_total, dtype=np.float64)),
        "abundances": np.stack(solved_abund, axis=0),
        "neutral_fractions": np.stack(solved_f, axis=0),
        "ne": np.asarray(solved_ne, dtype=np.float64),
    }

    if return_jax_arrays:
        return {key: jnp.asarray(value) for key, value in data.items()}
    return data


def create_optimizer(
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    total_steps: int = 100_000,
    warmup_steps: int = 1_000,
    gradient_clip_norm: Optional[float] = None,
):
    """Create an AdamW optimizer with optional gradient clipping and LR schedule.

    Uses ``optax.adamw`` with a piecewise learning-rate schedule:

    1. **Warm-up** – learning rate rises linearly from 0 to *learning_rate*
       over the first *warmup_steps* gradient steps.
    2. **Cosine decay** – learning rate decays following a cosine schedule
       from *learning_rate* down to ``0.1 * learning_rate`` over the
       remaining steps.

    When *warmup_steps* is 0, only the cosine-decay schedule is used.

    Parameters
    ----------
    learning_rate:
        Peak learning rate after the warm-up phase.  Default ``1e-3``.
    weight_decay:
        L2 weight-decay coefficient for AdamW.  Default ``1e-4``.
    total_steps:
        Total number of gradient steps (warm-up + cosine phases combined).
        Clamped to at least 1.  Default ``100_000``.
    warmup_steps:
        Number of warm-up gradient steps.  Clamped to
        ``[0, total_steps - 1]``.  Default ``1_000``.
    gradient_clip_norm:
        If provided and positive, apply global-norm gradient clipping with
        this threshold before AdamW updates.  Default ``None`` (disabled).

    Returns
    -------
    optax.GradientTransformation
        A configured Optax optimizer ready for use in a training loop.

    Raises
    ------
    RuntimeError
        If ``optax`` is not installed.
    """

    _require_optax()

    total_steps = max(1, int(total_steps))
    warmup_steps = int(np.clip(warmup_steps, 0, total_steps - 1))

    if warmup_steps > 0:
        warmup = optax.linear_schedule(
            init_value=0.0,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        )
        cosine = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=max(1, total_steps - warmup_steps),
            alpha=0.1,
        )
        schedule = optax.join_schedules([warmup, cosine], boundaries=[warmup_steps])
    else:
        schedule = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=total_steps,
            alpha=0.1,
        )

    adamw = optax.adamw(learning_rate=schedule, weight_decay=weight_decay)
    if gradient_clip_norm is not None and float(gradient_clip_norm) > 0.0:
        return optax.chain(optax.clip_by_global_norm(float(gradient_clip_norm)), adamw)
    return adamw


def initialize_model_params(
    model: ChemicalEquilibriumPINN,
    rng_key: jax.Array,
):
    """Randomly initialise model parameters using representative scalar inputs.

    Calls ``model.init`` with a solar-like thermodynamic state so that the
    Flax module can infer all layer shapes.  The dummy inputs are:

    * ``log_T = log10(5770)`` K (solar effective temperature),
    * ``log_n_total = log10(1e16)`` cm\ :sup:`-3`,
    * ``abundances`` – uniform over all ``MAX_ATOMIC_NUMBER`` elements.

    Parameters
    ----------
    model:
        A :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN` instance
        whose parameters are to be initialised.
    rng_key:
        JAX PRNG key used for weight initialisation.

    Returns
    -------
    params
        JAX pytree of initialised model parameters.
    """

    log_T = jnp.log10(jnp.asarray(5770.0, dtype=jnp.float64))
    log_n_total = jnp.log10(jnp.asarray(1e16, dtype=jnp.float64))
    abundances = jnp.full((MAX_ATOMIC_NUMBER,), 1.0 / MAX_ATOMIC_NUMBER, dtype=jnp.float64)
    return model.init(rng_key, log_T, log_n_total, abundances)


def _to_jax_batch(batch: Dict[str, np.ndarray | jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Convert all values in a data dict to ``float64`` JAX arrays.

    Parameters
    ----------
    batch:
        Dictionary mapping string keys to NumPy or JAX arrays.

    Returns
    -------
    dict
        Same keys as *batch*, with every value replaced by a
        ``jnp.float64`` device array.
    """
    return {key: jnp.asarray(value, dtype=jnp.float64) for key, value in batch.items()}


def _take_batch(data: Dict[str, jnp.ndarray], indices: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Gather a mini-batch from a dataset dict using integer indices.

    Parameters
    ----------
    data:
        Dictionary of JAX arrays, each with at least one axis representing
        the sample dimension (axis 0).
    indices:
        1-D integer array of sample indices to collect.

    Returns
    -------
    dict
        Same keys as *data*; values are sub-arrays indexed by *indices*
        along axis 0.
    """
    return {key: jnp.take(value, indices, axis=0) for key, value in data.items()}


def supervised_objective(
    params,
    batch: Dict[str, jnp.ndarray],
    model: ChemicalEquilibriumPINN,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Supervised loss objective comparing PINN predictions to solver labels.

    Both the electron density and the neutral fractions are compared in
    :math:`\\log_{10}` space using mean-squared error, which treats all
    orders of magnitude equally:

    .. math::

        \\mathcal{L}_{\\rm sup} =
        \\underbrace{\\mathbb{E}[(\\log_{10}\\hat{n}_e - \\log_{10} n_e)^2]}_{\\mathcal{L}_{n_e}}
        +
        \\underbrace{\\mathbb{E}[(\\log_{10}\\hat{f}_Z - \\log_{10} f_Z)^2]}_{\\mathcal{L}_f}

    Parameters
    ----------
    params:
        Current PINN parameter pytree.
    batch:
        Mini-batch dict with keys ``"log_T"``, ``"log_n_total"``,
        ``"abundances"``, ``"n_total"``, ``"ne"``, ``"neutral_fractions"``.
    model:
        :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN` instance.

    Returns
    -------
    total : jnp.ndarray
        Scalar total loss :math:`\\mathcal{L}_{n_e} + \\mathcal{L}_f`.
    metrics : dict
        Sub-losses ``{"ne": ..., "f": ...}`` for monitoring.
    """

    neutral_pred, ne_logit = model.apply(
        params,
        batch["log_T"],
        batch["log_n_total"],
        batch["abundances"],
    )
    ne_pred = batch["n_total"] * jax.nn.sigmoid(ne_logit)

    log10_ne_pred = _safe_log10(ne_pred)
    log10_ne_true = _safe_log10(batch["ne"])
    loss_ne = jnp.mean((log10_ne_pred - log10_ne_true) ** 2)

    log10_f_pred = _safe_log10(jnp.clip(neutral_pred, 1e-30, 1.0))
    log10_f_true = _safe_log10(jnp.clip(batch["neutral_fractions"], 1e-30, 1.0))
    loss_f = jnp.mean((log10_f_pred - log10_f_true) ** 2)

    total = loss_ne + loss_f
    return total, {"ne": loss_ne, "f": loss_f}


def physics_objective(
    params,
    batch: Dict[str, jnp.ndarray],
    chem_data: ChemEqData,
    model: ChemicalEquilibriumPINN,
    *,
    w_element: float = 1.0,
    w_charge: float = 10.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Physics-informed loss using element-conservation and charge-neutrality residuals.

    Evaluates the PINN predictions and measures how well they satisfy the
    underlying physical constraints *without* requiring ground-truth labels.
    The total loss is computed by
    :func:`~.chem_eq_pinn_loss.total_physics_loss`:

    .. math::

        \\mathcal{L}_{\\rm phys} =
        w_{\\rm elem} \\cdot \\mathcal{L}_{\\rm elem}
        + w_{\\rm charge} \\cdot \\mathcal{L}_{\\rm charge}

    Parameters
    ----------
    params:
        Current PINN parameter pytree.
    batch:
        Mini-batch dict with keys ``"log_T"``, ``"log_n_total"``,
        ``"abundances"``, ``"n_total"``, ``"T"``.
    chem_data:
        Pre-computed chemical-equilibrium tables (partition functions,
        ionisation energies, equilibrium constants).
    model:
        :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN` instance.
    w_element:
        Weight for the element-conservation loss term.  Default ``1.0``.
    w_charge:
        Weight for the charge-neutrality loss term.  Default ``10.0``.

    Returns
    -------
    total : jnp.ndarray
        Scalar weighted physics loss.
    metrics : dict
        Sub-losses ``{"element": ..., "charge": ...}`` for monitoring.
    """

    neutral_pred, ne_logit = model.apply(
        params,
        batch["log_T"],
        batch["log_n_total"],
        batch["abundances"],
    )
    ne_pred = batch["n_total"] * jax.nn.sigmoid(ne_logit)

    total, components = total_physics_loss(
        neutral_pred,
        ne_pred,
        batch["T"],
        batch["n_total"],
        batch["abundances"],
        chem_data,
        w_element=w_element,
        w_charge=w_charge,
    )
    return total, {
        "element": components["element"],
        "charge": components["charge"],
    }


def phase2_joint_objective(
    params,
    batch: Dict[str, jnp.ndarray],
    chem_data: ChemEqData,
    model: ChemicalEquilibriumPINN,
    *,
    phase2_supervised_weight: float = 1.0,
    w_element: float = 1.0,
    w_charge: float = 10.0,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Joint objective for phase-2 fine-tuning: supervised + physics terms.

    The phase-2 objective is

    .. math::

        \\mathcal{L}_{\\rm phase2}
        = w_{\\rm sup,2}\\,\\mathcal{L}_{\\rm sup}
        + \\mathcal{L}_{\\rm phys}.

    This helps retain label fidelity while still enforcing physics residuals
    during fine-tuning.

    Parameters
    ----------
    params:
        Current PINN parameter pytree.
    batch:
        Mini-batch dict with the union of keys required by
        :func:`supervised_objective` and :func:`physics_objective`.
    chem_data:
        Pre-computed chemical-equilibrium tables used in physics loss.
    model:
        :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN` instance.
    phase2_supervised_weight:
        Multiplicative weight for the supervised loss term in phase-2.
    w_element:
        Weight for element-conservation residuals inside physics loss.
    w_charge:
        Weight for charge-neutrality residual inside physics loss.

    Returns
    -------
    total : jnp.ndarray
        Scalar joint loss.
    metrics : dict
        Dictionary with joint and component losses for logging.
    """

    sup_total, sup_metrics = supervised_objective(params, batch, model)
    phys_total, phys_metrics = physics_objective(
        params,
        batch,
        chem_data,
        model,
        w_element=w_element,
        w_charge=w_charge,
    )
    total = phase2_supervised_weight * sup_total + phys_total
    return total, {
        "supervised_total": sup_total,
        "supervised_ne": sup_metrics["ne"],
        "supervised_f": sup_metrics["f"],
        "physics_total": phys_total,
        "element": phys_metrics["element"],
        "charge": phys_metrics["charge"],
    }


def make_supervised_train_step(
    model: ChemicalEquilibriumPINN,
    optimizer,
):
    """Create a JIT-compiled supervised training step function.

    The returned callable performs one gradient step of the supervised
    objective (:func:`supervised_objective`) and updates the optimizer state.

    Parameters
    ----------
    model:
        :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN` instance.
    optimizer:
        An Optax-compatible optimizer (e.g. from :func:`create_optimizer`).

    Returns
    -------
    train_step : callable
        ``train_step(params, opt_state, batch)`` returns
        ``(new_params, new_opt_state, loss, metrics)``.

    Raises
    ------
    RuntimeError
        If ``optax`` is not installed.
    """

    _require_optax()

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            return supervised_objective(p, batch, model)

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    return train_step


def make_physics_train_step(
    model: ChemicalEquilibriumPINN,
    optimizer,
    chem_data: ChemEqData,
    *,
    w_element: float = 1.0,
    w_charge: float = 10.0,
):
    """Create a JIT-compiled physics fine-tuning step function.

    The returned callable performs one gradient step of the physics objective
    (:func:`physics_objective`) without requiring ground-truth labels.

    Parameters
    ----------
    model:
        :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN` instance.
    optimizer:
        An Optax-compatible optimizer.
    chem_data:
        Pre-computed chemical-equilibrium tables forwarded to
        :func:`physics_objective`.
    w_element:
        Weight for the element-conservation residual.  Default ``1.0``.
    w_charge:
        Weight for the charge-neutrality residual.  Default ``10.0``.

    Returns
    -------
    train_step : callable
        ``train_step(params, opt_state, batch)`` returns
        ``(new_params, new_opt_state, loss, metrics)``.

    Raises
    ------
    RuntimeError
        If ``optax`` is not installed.
    """

    _require_optax()

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            return physics_objective(
                p,
                batch,
                chem_data,
                model,
                w_element=w_element,
                w_charge=w_charge,
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    return train_step


def make_phase2_joint_train_step(
    model: ChemicalEquilibriumPINN,
    optimizer,
    chem_data: ChemEqData,
    *,
    phase2_supervised_weight: float = 1.0,
    w_element: float = 1.0,
    w_charge: float = 10.0,
):
    """Create a JIT-compiled phase-2 train step using joint loss.

    The returned callable performs one gradient step on
    :func:`phase2_joint_objective`, combining supervised and physics losses.
    """

    _require_optax()

    @jax.jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            return phase2_joint_objective(
                p,
                batch,
                chem_data,
                model,
                phase2_supervised_weight=phase2_supervised_weight,
                w_element=w_element,
                w_charge=w_charge,
            )

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params=params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, metrics

    return train_step


@dataclass
class TrainingHistory:
    """Container for per-step loss values recorded during :func:`train_pinn`.

    Attributes
    ----------
    supervised_losses : list of float
        Scalar total loss recorded after each supervised gradient step.
    physics_losses : list of float
        Scalar total loss recorded after each physics fine-tuning step.
    """

    supervised_losses: list
    physics_losses: list


def train_pinn(
    model: ChemicalEquilibriumPINN,
    chem_data: ChemEqData,
    train_data: Dict[str, np.ndarray | jnp.ndarray],
    *,
    rng_key: jax.Array,
    params=None,
    optimizer=None,
    batch_size: int = 128,
    supervised_steps: int = 1_000,
    physics_steps: int = 1_000,
    phase2_supervised_weight: float = 0.0,
    w_element: float = 1.0,
    w_charge: float = 10.0,
):
    """Run two-phase PINN training and return updated parameters plus loss history.

    **Phase 1 – Supervised** (``supervised_steps`` gradient steps)
        Mini-batches of solver labels are drawn from *train_data* and the
        network fitted by minimising the log-space MSE between predictions
        and labels via :func:`supervised_objective`.

    **Phase 2 – Physics fine-tuning** (``physics_steps`` gradient steps)
        The same optimizer state is continued.  Mini-batches are drawn and
        the network is refined by minimising element-conservation and
        charge-neutrality residuals via :func:`physics_objective`.  When
        ``phase2_supervised_weight > 0``, phase-2 instead minimises a joint
        objective:

        ``phase2_supervised_weight * supervised_loss + physics_loss``.

    Parameters
    ----------
    model:
        :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN` instance.
    chem_data:
        Pre-computed chemical-equilibrium tables used in the physics phase.
    train_data:
        Dataset dict as returned by :func:`generate_training_data`.  Required
        keys: ``"T"``, ``"log_T"``, ``"n_total"``, ``"log_n_total"``,
        ``"abundances"``, ``"neutral_fractions"``, ``"ne"``.
    rng_key:
        JAX PRNG key for mini-batch sampling during training.
    params:
        Initial model parameters.  If ``None``, fresh parameters are drawn
        via :func:`initialize_model_params`.
    optimizer:
        An Optax optimizer.  If ``None``, one is created by
        :func:`create_optimizer` with ``total_steps = supervised_steps +
        physics_steps``.
    batch_size:
        Number of samples per mini-batch.  Default ``128``.
    supervised_steps:
        Number of supervised gradient steps.  Default ``1000``.
    physics_steps:
        Number of physics fine-tuning steps.  Default ``1000``.
    phase2_supervised_weight:
        Supervised-loss weight used in phase-2 joint objective.  Set to
        ``0`` to recover physics-only fine-tuning.  Default ``0.0``.
    w_element:
        Weight for element-conservation residuals in the physics loss.
        Default ``1.0``.
    w_charge:
        Weight for the charge-neutrality residual in the physics loss.
        Default ``10.0``.

    Returns
    -------
    params
        Updated JAX parameter pytree after both training phases.
    history : dict
        Dict with keys ``"supervised_losses"`` (list of float) and
        ``"physics_losses"`` (list of float), each containing one scalar per
        gradient step in the respective phase.

    Raises
    ------
    ValueError
        If *train_data* is empty.
    RuntimeError
        If ``optax`` is not installed.
    """

    _require_optax()

    batch_size = max(1, int(batch_size))
    train_data = _to_jax_batch(train_data)
    n_samples = int(train_data["T"].shape[0])
    if n_samples == 0:
        raise ValueError("train_data is empty.")

    if params is None:
        rng_key, init_key = jax.random.split(rng_key)
        params = initialize_model_params(model, init_key)

    total_steps = max(1, int(supervised_steps) + int(physics_steps))
    if optimizer is None:
        optimizer = create_optimizer(total_steps=total_steps)

    opt_state = optimizer.init(params)

    supervised_step = make_supervised_train_step(model, optimizer)
    use_joint_phase2 = float(phase2_supervised_weight) > 0.0
    if use_joint_phase2:
        phase2_step = make_phase2_joint_train_step(
            model,
            optimizer,
            chem_data,
            phase2_supervised_weight=phase2_supervised_weight,
            w_element=w_element,
            w_charge=w_charge,
        )
    else:
        phase2_step = make_physics_train_step(
            model,
            optimizer,
            chem_data,
            w_element=w_element,
            w_charge=w_charge,
        )

    history = TrainingHistory(supervised_losses=[], physics_losses=[])

    for _ in range(int(supervised_steps)):
        rng_key, subkey = jax.random.split(rng_key)
        idx = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=n_samples)
        batch = _take_batch(train_data, idx)
        params, opt_state, loss, _ = supervised_step(params, opt_state, batch)
        history.supervised_losses.append(float(jax.device_get(loss)))

    for _ in range(int(physics_steps)):
        rng_key, subkey = jax.random.split(rng_key)
        idx = jax.random.randint(subkey, shape=(batch_size,), minval=0, maxval=n_samples)
        batch = _take_batch(train_data, idx)
        params, opt_state, loss, _ = phase2_step(params, opt_state, batch)
        history.physics_losses.append(float(jax.device_get(loss)))

    return params, {
        "supervised_losses": history.supervised_losses,
        "physics_losses": history.physics_losses,
    }


__all__ = [
    "absolute_abundances_from_ax",
    "sample_thermodynamic_conditions",
    "generate_training_data",
    "create_optimizer",
    "initialize_model_params",
    "supervised_objective",
    "physics_objective",
    "phase2_joint_objective",
    "make_supervised_train_step",
    "make_physics_train_step",
    "make_phase2_joint_train_step",
    "train_pinn",
]
