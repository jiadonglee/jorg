"""
JAXOpt-based single-layer chemical equilibrium solver.

This is intended for benchmarking and numerical comparison against the
SciPy root-based solver, not yet integrated into the full pipeline.
"""

from collections import OrderedDict
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np

import jax
import jax.numpy as jnp
from jaxopt import LevenbergMarquardt

from .species import Species, MAX_ATOMIC_NUMBER
from .korg_chemical_equilibrium import _get_cached_saha_weight_arrays
from ..constants import kboltz_cgs


_LOG10_MIN = -300.0
_LOG10_MAX = 300.0

_JAXOPT_SOLVER_CACHE = OrderedDict()
_JAXOPT_SOLVER_CACHE_MAX = 32
_JAXOPT_LM_RUNNER_CACHE = OrderedDict()
_JAXOPT_LM_RUNNER_CACHE_MAX = 32
_MOLECULE_STRUCTURE_CACHE = OrderedDict()
_MOLECULE_STRUCTURE_CACHE_MAX = 8

_EMPTY_NEUTRAL_STOICH = np.empty((0, MAX_ATOMIC_NUMBER), dtype=np.float64)
_EMPTY_NEUTRAL_LOG_NK = np.empty((0,), dtype=np.float64)
_EMPTY_CHARGED_IDX = np.empty((0,), dtype=np.int32)
_EMPTY_CHARGED_LOG_NK = np.empty((0,), dtype=np.float64)
_EMPTY_CHARGED_STOICH = np.empty((0, MAX_ATOMIC_NUMBER), dtype=np.float64)

_SPECIES_NEUTRAL = tuple(Species.from_atomic_number(Z, 0) for Z in range(1, MAX_ATOMIC_NUMBER + 1))
_SPECIES_ION_1 = tuple(Species.from_atomic_number(Z, 1) for Z in range(1, MAX_ATOMIC_NUMBER + 1))
_SPECIES_ION_2 = tuple(Species.from_atomic_number(Z, 2) for Z in range(1, MAX_ATOMIC_NUMBER + 1))


def _coerce_absolute_abundance_array(absolute_abundances):
    if isinstance(absolute_abundances, dict):
        abs_abund_array = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
        for Z, abund in absolute_abundances.items():
            if 1 <= Z <= MAX_ATOMIC_NUMBER:
                abs_abund_array[Z - 1] = abund
        return abs_abund_array

    abs_abund_array = np.asarray(absolute_abundances, dtype=np.float64)
    if abs_abund_array.shape[0] != MAX_ATOMIC_NUMBER:
        raise ValueError(
            f"absolute_abundances must have length {MAX_ATOMIC_NUMBER} (got {abs_abund_array.shape[0]})."
        )
    return abs_abund_array


def _resolve_lm_solver_sequence(lm_solver: str) -> Tuple[str, ...]:
    lm_solver_norm = str(lm_solver).lower()
    if lm_solver_norm == "auto":
        # Cholesky is fastest on dense CE Jacobians in steady-state CPU runs.
        return ("cholesky", "lu", "qr")
    if lm_solver_norm not in ("cholesky", "lu", "qr", "inv", "svd"):
        raise ValueError(
            f"Unknown lm_solver '{lm_solver}'. Expected one of: auto, cholesky, lu, qr, inv, svd."
        )
    return (lm_solver_norm,)


def _build_cached_molecule_structure(log_equilibrium_constants: Dict):
    neutral_entries = []
    charged_entries = []

    for mol_species, log_K_func in log_equilibrium_constants.items():
        try:
            atoms = tuple(mol_species.get_atoms())
            if len(atoms) == 0:
                continue

            atom_indices = np.asarray([Z - 1 for Z in atoms], dtype=np.int32)
            if np.any((atom_indices < 0) | (atom_indices >= MAX_ATOMIC_NUMBER)):
                continue

            n_atoms_minus_one = float(len(atoms) - 1.0)
            if int(getattr(mol_species, "charge", 0)) == 1:
                if len(atom_indices) < 2:
                    continue
                charged_entries.append(
                    (mol_species, int(atom_indices[0]), int(atom_indices[1]), n_atoms_minus_one, log_K_func)
                )
            else:
                uniq, counts = np.unique(atom_indices, return_counts=True)
                neutral_entries.append(
                    (mol_species, uniq.astype(np.int32), counts.astype(np.float64), n_atoms_minus_one, log_K_func)
                )
        except Exception:
            continue

    if not neutral_entries and not charged_entries:
        return None

    result = {}
    if neutral_entries:
        n_neutral = len(neutral_entries)
        neutral_stoich = np.zeros((n_neutral, MAX_ATOMIC_NUMBER), dtype=np.float64)
        neutral_n_atoms_minus_one = np.zeros(n_neutral, dtype=np.float64)
        neutral_funcs = []
        neutral_species = []
        for i, (species, uniq, counts, n_atoms_minus_one, func) in enumerate(neutral_entries):
            neutral_stoich[i, uniq] = counts
            neutral_n_atoms_minus_one[i] = n_atoms_minus_one
            neutral_funcs.append(func)
            neutral_species.append(species)

        result["neutral_stoich"] = neutral_stoich
        result["neutral_n_atoms_minus_one"] = neutral_n_atoms_minus_one
        result["neutral_funcs"] = tuple(neutral_funcs)
        result["neutral_species"] = tuple(neutral_species)

    if charged_entries:
        n_charged = len(charged_entries)
        charged_idx1 = np.zeros(n_charged, dtype=np.int32)
        charged_idx2 = np.zeros(n_charged, dtype=np.int32)
        charged_stoich = np.zeros((n_charged, MAX_ATOMIC_NUMBER), dtype=np.float64)
        charged_n_atoms_minus_one = np.zeros(n_charged, dtype=np.float64)
        charged_funcs = []
        charged_species = []
        for i, (species, idx1, idx2, n_atoms_minus_one, func) in enumerate(charged_entries):
            charged_idx1[i] = idx1
            charged_idx2[i] = idx2
            charged_stoich[i, idx1] += 1.0
            charged_stoich[i, idx2] += 1.0
            charged_n_atoms_minus_one[i] = n_atoms_minus_one
            charged_funcs.append(func)
            charged_species.append(species)

        result["charged_idx1"] = charged_idx1
        result["charged_idx2"] = charged_idx2
        result["charged_stoich"] = charged_stoich
        result["charged_n_atoms_minus_one"] = charged_n_atoms_minus_one
        result["charged_funcs"] = tuple(charged_funcs)
        result["charged_species"] = tuple(charged_species)

    return result or None


def _get_cached_molecule_structure(log_equilibrium_constants: Dict):
    key = id(log_equilibrium_constants)
    cached = _MOLECULE_STRUCTURE_CACHE.get(key)
    if cached is not None and cached.get("owner") is log_equilibrium_constants:
        _MOLECULE_STRUCTURE_CACHE.move_to_end(key)
        return cached.get("value")

    structure = _build_cached_molecule_structure(log_equilibrium_constants)
    _MOLECULE_STRUCTURE_CACHE[key] = {"owner": log_equilibrium_constants, "value": structure}
    if len(_MOLECULE_STRUCTURE_CACHE) > _MOLECULE_STRUCTURE_CACHE_MAX:
        _MOLECULE_STRUCTURE_CACHE.popitem(last=False)
    return structure


def _prepare_molecule_arrays(temperature: float, log_equilibrium_constants: Optional[Dict]):
    if log_equilibrium_constants is None:
        return None

    structure = _get_cached_molecule_structure(log_equilibrium_constants)
    if structure is None:
        return None

    log_T = float(np.log(temperature))
    log_kT = float(np.log10(kboltz_cgs * temperature))

    result = {}
    neutral_stoich = structure.get("neutral_stoich")
    if neutral_stoich is not None:
        neutral_funcs = structure["neutral_funcs"]
        neutral_n_atoms_minus_one = structure["neutral_n_atoms_minus_one"]
        neutral_log_nK = np.full(len(neutral_funcs), np.inf, dtype=np.float64)
        for i, (log_K_func, n_atoms_minus_one) in enumerate(zip(neutral_funcs, neutral_n_atoms_minus_one)):
            try:
                logKp = float(log_K_func(log_T))  # log10(K_p)
                log_nK = logKp - float(n_atoms_minus_one) * log_kT
                if np.isfinite(log_nK):
                    neutral_log_nK[i] = log_nK
            except Exception:
                continue
        result["neutral_stoich"] = neutral_stoich
        result["neutral_log_nK"] = neutral_log_nK
        result["neutral_species"] = structure["neutral_species"]

    charged_idx1 = structure.get("charged_idx1")
    if charged_idx1 is not None:
        charged_funcs = structure["charged_funcs"]
        charged_n_atoms_minus_one = structure["charged_n_atoms_minus_one"]
        charged_log_nK = np.full(len(charged_funcs), np.inf, dtype=np.float64)
        for i, (log_K_func, n_atoms_minus_one) in enumerate(zip(charged_funcs, charged_n_atoms_minus_one)):
            try:
                logKp = float(log_K_func(log_T))  # log10(K_p)
                log_nK = logKp - float(n_atoms_minus_one) * log_kT
                if np.isfinite(log_nK):
                    charged_log_nK[i] = log_nK
            except Exception:
                continue
        result["charged_idx1"] = charged_idx1
        result["charged_idx2"] = structure["charged_idx2"]
        result["charged_log_nK"] = charged_log_nK
        result["charged_stoich"] = structure["charged_stoich"]
        result["charged_species"] = structure["charged_species"]

    return result or None


def _pack_molecule_arrays(mol_data: Optional[Dict]):
    if mol_data is None:
        return (
            _EMPTY_NEUTRAL_STOICH,
            _EMPTY_NEUTRAL_LOG_NK,
            _EMPTY_CHARGED_IDX,
            _EMPTY_CHARGED_IDX,
            _EMPTY_CHARGED_LOG_NK,
            _EMPTY_CHARGED_STOICH,
        )

    neutral_stoich = mol_data.get("neutral_stoich")
    neutral_log_nK = mol_data.get("neutral_log_nK")
    charged_idx1 = mol_data.get("charged_idx1")
    charged_idx2 = mol_data.get("charged_idx2")
    charged_log_nK = mol_data.get("charged_log_nK")
    charged_stoich = mol_data.get("charged_stoich")

    if neutral_stoich is None:
        neutral_stoich = _EMPTY_NEUTRAL_STOICH
        neutral_log_nK = _EMPTY_NEUTRAL_LOG_NK
    if charged_idx1 is None:
        charged_idx1 = _EMPTY_CHARGED_IDX
        charged_idx2 = _EMPTY_CHARGED_IDX
        charged_log_nK = _EMPTY_CHARGED_LOG_NK
        charged_stoich = _EMPTY_CHARGED_STOICH
    elif charged_stoich is None:
        charged_stoich = _EMPTY_CHARGED_STOICH

    return (
        np.asarray(neutral_stoich, dtype=np.float64),
        np.asarray(neutral_log_nK, dtype=np.float64),
        np.asarray(charged_idx1, dtype=np.int32),
        np.asarray(charged_idx2, dtype=np.int32),
        np.asarray(charged_log_nK, dtype=np.float64),
        np.asarray(charged_stoich, dtype=np.float64),
    )


def _residual_function_generic(
    x,
    n_total_j,
    abs_abund_j,
    wII_ne_j,
    wIII_ne2_j,
    neutral_stoich,
    neutral_log_nK,
    charged_idx1,
    charged_idx2,
    charged_log_nK,
    charged_stoich,
):
    x = jnp.asarray(x, dtype=jnp.float64)
    ne = jnp.maximum(jnp.abs(x[-1]) * n_total_j * 1e-5, 1e-300)

    neutral_fractions = jnp.abs(x[:-1])
    atom_number_densities = abs_abund_j * (n_total_j - ne)
    neutral_number_densities = atom_number_densities * neutral_fractions

    inv_ne = 1.0 / ne
    inv_ne2 = inv_ne * inv_ne
    wII = wII_ne_j * inv_ne
    wIII = wIII_ne2_j * inv_ne2

    R_elem = atom_number_densities - (1.0 + wII + wIII) * neutral_number_densities
    R_charge = jnp.sum((wII + 2.0 * wIII) * neutral_number_densities) - ne

    has_neutral = neutral_stoich.shape[0] > 0
    has_charged = charged_idx1.shape[0] > 0

    if has_neutral or has_charged:
        log_neutral = jnp.log10(jnp.maximum(neutral_number_densities, 1e-100))

    if has_charged:
        log_wII = jnp.log10(jnp.maximum(wII, 1e-300))
        log_n_mol = (
            log_neutral[charged_idx1]
            + log_wII[charged_idx1]
            + log_neutral[charged_idx2]
            - charged_log_nK
        )
        log_n_mol = jnp.clip(log_n_mol, _LOG10_MIN, _LOG10_MAX)
        n_mol = jnp.power(10.0, log_n_mol)
        R_elem = R_elem - jnp.matmul(charged_stoich.T, n_mol)
        R_charge = R_charge + jnp.sum(n_mol)

    if has_neutral:
        log_n_mol = jnp.matmul(neutral_stoich, log_neutral) - neutral_log_nK
        log_n_mol = jnp.clip(log_n_mol, _LOG10_MIN, _LOG10_MAX)
        n_mol = jnp.power(10.0, log_n_mol)
        R_elem = R_elem - jnp.matmul(neutral_stoich.T, n_mol)

    denom_elem = jnp.maximum(atom_number_densities, 1e-100)
    denom_charge = ne * 1e-5

    F_elem = R_elem / denom_elem
    F_charge = R_charge / denom_charge
    return jnp.concatenate((F_elem, jnp.array([F_charge], dtype=jnp.float64)))


def _atomic_residual_function_generic(
    x,
    n_total_j,
    abs_abund_j,
    wII_ne_j,
    wIII_ne2_j,
):
    x = jnp.asarray(x, dtype=jnp.float64)
    ne = jnp.maximum(jnp.abs(x[-1]) * n_total_j * 1e-5, 1e-300)

    neutral_fractions = jnp.abs(x[:-1])
    atom_number_densities = abs_abund_j * (n_total_j - ne)
    neutral_number_densities = atom_number_densities * neutral_fractions

    inv_ne = 1.0 / ne
    inv_ne2 = inv_ne * inv_ne
    wII = wII_ne_j * inv_ne
    wIII = wIII_ne2_j * inv_ne2

    R_elem = atom_number_densities - (1.0 + wII + wIII) * neutral_number_densities
    R_charge = jnp.sum((wII + 2.0 * wIII) * neutral_number_densities) - ne

    denom_elem = jnp.maximum(atom_number_densities, 1e-100)
    denom_charge = ne * 1e-5

    F_elem = R_elem / denom_elem
    F_charge = R_charge / denom_charge
    return jnp.concatenate((F_elem, jnp.array([F_charge], dtype=jnp.float64)))


def _atomic_jacobian_function_generic(
    x,
    n_total_j,
    abs_abund_j,
    wII_ne_j,
    wIII_ne2_j,
):
    x = jnp.asarray(x, dtype=jnp.float64)
    sign_ne = jnp.where(x[-1] == 0, 1.0, jnp.sign(x[-1]))
    sign_f = jnp.where(x[:-1] == 0, 1.0, jnp.sign(x[:-1]))
    ne = jnp.maximum(jnp.abs(x[-1]) * n_total_j * 1e-5, 1e-300)

    neutral_fractions = jnp.abs(x[:-1])
    atom_number_densities = abs_abund_j * (n_total_j - ne)
    neutral_number_densities = atom_number_densities * neutral_fractions

    inv_ne = 1.0 / ne
    inv_ne2 = inv_ne * inv_ne
    wII = wII_ne_j * inv_ne
    wIII = wIII_ne2_j * inv_ne2

    R_elem = atom_number_densities - (1.0 + wII + wIII) * neutral_number_densities
    R_charge = jnp.sum((wII + 2.0 * wIII) * neutral_number_densities) - ne

    denom_elem = jnp.maximum(atom_number_densities, 1e-100)
    denom_charge = ne * 1e-5

    diag_vals = -(1.0 + wII + wIII) * atom_number_densities
    diag_vals = (diag_vals / denom_elem) * sign_f
    J = jnp.zeros((MAX_ATOMIC_NUMBER + 1, MAX_ATOMIC_NUMBER + 1), dtype=jnp.float64)
    J = J.at[:MAX_ATOMIC_NUMBER, :MAX_ATOMIC_NUMBER].set(jnp.diag(diag_vals))

    A = abs_abund_j
    dn_dne = -A * neutral_fractions
    dS_dne = -(wII + 2.0 * wIII) * inv_ne
    dR_dne = (-A) - ((1.0 + wII + wIII) * dn_dne + neutral_number_densities * dS_dne)

    Q = wII + 2.0 * wIII
    dR_charge_df = Q * atom_number_densities
    dQ_dne = -(wII + 4.0 * wIII) * inv_ne
    dR_charge_dne = jnp.sum(Q * dn_dne + neutral_number_densities * dQ_dne) - 1.0

    use_denom = atom_number_densities > 1e-100
    dF_dne = dR_dne / denom_elem
    dF_dne = dF_dne + jnp.where(use_denom, (R_elem * A / (denom_elem * denom_elem)), 0.0)

    dne_dx = sign_ne * n_total_j * 1e-5
    J = J.at[:MAX_ATOMIC_NUMBER, -1].set(dF_dne * dne_dx)

    J = J.at[-1, :MAX_ATOMIC_NUMBER].set((dR_charge_df / denom_charge) * sign_f)
    dF_charge_dne = (dR_charge_dne / denom_charge) - (R_charge * 1e-5) / (denom_charge * denom_charge)
    J = J.at[-1, -1].set(dF_charge_dne * dne_dx)
    return J


def _atomic_newton_solve_impl(
    x0,
    n_total_j,
    abs_abund_j,
    wII_ne_j,
    wIII_ne2_j,
    maxiter_j,
    tol_j,
    damping_j,
):
    eye = jnp.eye(MAX_ATOMIC_NUMBER + 1, dtype=jnp.float64)

    def step(state):
        x, it, _ = state
        F = _atomic_residual_function_generic(x, n_total_j, abs_abund_j, wII_ne_j, wIII_ne2_j)
        J = _atomic_jacobian_function_generic(x, n_total_j, abs_abund_j, wII_ne_j, wIII_ne2_j)
        J = J + damping_j * eye
        delta = jnp.linalg.solve(J, -F)
        x_new = x + delta
        err = jnp.max(jnp.abs(F))
        return (x_new, it + 1, err)

    def cond(state):
        _, it, err = state
        return (it < maxiter_j) & (err > tol_j)

    return jax.lax.while_loop(cond, step, (x0, jnp.asarray(0, dtype=jnp.int32), jnp.inf))


_ATOMIC_NEWTON_SOLVE_JIT = jax.jit(_atomic_newton_solve_impl)


def _run_molecular_newton_line_search(
    x0,
    *,
    residual_stop_tol: float,
    n_total_j,
    abs_abund_j,
    wII_ne_j,
    wIII_ne2_j,
    neutral_stoich_j,
    neutral_log_nK_j,
    charged_idx1_j,
    charged_idx2_j,
    charged_log_nK_j,
    charged_stoich_j,
    maxiter: int,
    damping: float,
    line_search_max_steps: int,
    line_search_backtrack: float,
    line_search_armijo_c: float,
):
    """Refine molecular LM result via damped Newton with residual line search."""
    x = jnp.asarray(x0, dtype=jnp.float64)
    eye = jnp.eye(MAX_ATOMIC_NUMBER + 1, dtype=jnp.float64)

    def residual_fn(xx):
        return _residual_function_generic(
            xx,
            n_total_j,
            abs_abund_j,
            wII_ne_j,
            wIII_ne2_j,
            neutral_stoich_j,
            neutral_log_nK_j,
            charged_idx1_j,
            charged_idx2_j,
            charged_log_nK_j,
            charged_stoich_j,
        )

    jac_fn = jax.jacfwd(residual_fn)
    ls_max = max(int(line_search_max_steps), 1)
    ls_backtrack = float(line_search_backtrack)
    ls_armijo_c = float(line_search_armijo_c)

    nfev = 0
    njev = 0
    accepted_steps = 0
    line_search_evals = 0
    iter_done = 0
    final_residual = np.inf
    converged_residual = False

    for it in range(int(maxiter)):
        F = residual_fn(x)
        nfev += 1
        residual = float(jax.device_get(jnp.max(jnp.abs(F))))
        final_residual = residual
        iter_done = it + 1

        if not np.isfinite(residual):
            break
        if residual <= float(residual_stop_tol):
            converged_residual = True
            break

        J = jac_fn(x)
        njev += 1
        J = J + float(damping) * eye
        delta = jnp.linalg.solve(J, -F)

        if not np.all(np.isfinite(np.asarray(jax.device_get(delta), dtype=np.float64))):
            delta = jnp.matmul(jnp.linalg.pinv(J), -F)
            if not np.all(np.isfinite(np.asarray(jax.device_get(delta), dtype=np.float64))):
                break

        alpha = 1.0
        accepted = False
        best_residual = residual
        best_x = x

        for _ in range(ls_max):
            x_try = x + alpha * delta
            F_try = residual_fn(x_try)
            nfev += 1
            line_search_evals += 1
            residual_try = float(jax.device_get(jnp.max(jnp.abs(F_try))))

            if np.isfinite(residual_try) and residual_try < best_residual:
                best_residual = residual_try
                best_x = x_try

            if np.isfinite(residual_try) and residual_try <= residual * (1.0 - ls_armijo_c * alpha):
                x = x_try
                final_residual = residual_try
                accepted_steps += 1
                accepted = True
                break

            alpha *= ls_backtrack

        if not accepted:
            if best_residual < residual:
                x = best_x
                final_residual = best_residual
            else:
                break

    return np.asarray(jax.device_get(x), dtype=np.float64), {
        "iter_num": int(iter_done),
        "residual_max": float(final_residual),
        "converged_residual": bool(converged_residual),
        "num_fun_eval": int(nfev),
        "num_jac_eval": int(njev),
        "accepted_steps": int(accepted_steps),
        "line_search_evals": int(line_search_evals),
    }


def _get_cached_lm_solver(
    *,
    maxiter: int,
    tol: float,
    jit: bool,
    lm_solver: str,
    materialize_jac: bool,
    implicit_diff: bool,
    neutral_stoich_shape: Tuple[int, int],
    charged_shape: int,
):
    key = (
        int(maxiter),
        float(tol),
        bool(jit),
        str(lm_solver),
        bool(materialize_jac),
        bool(implicit_diff),
        tuple(int(v) for v in neutral_stoich_shape),
        int(charged_shape),
    )
    cached = _JAXOPT_SOLVER_CACHE.get(key)
    if cached is not None:
        _JAXOPT_SOLVER_CACHE.move_to_end(key)
        return cached

    solver = LevenbergMarquardt(
        _residual_function_generic,
        maxiter=maxiter,
        tol=tol,
        solver=lm_solver,
        materialize_jac=materialize_jac,
        implicit_diff=implicit_diff,
        jit=jit,
    )
    _JAXOPT_SOLVER_CACHE[key] = solver
    if len(_JAXOPT_SOLVER_CACHE) > _JAXOPT_SOLVER_CACHE_MAX:
        _JAXOPT_SOLVER_CACHE.popitem(last=False)
    return solver


def _get_cached_lm_runner(cache_key, solver):
    cached = _JAXOPT_LM_RUNNER_CACHE.get(cache_key)
    if cached is not None:
        _JAXOPT_LM_RUNNER_CACHE.move_to_end(cache_key)
        return cached

    maxiter_local = int(solver.maxiter)
    tol_local = float(solver.tol)

    @jax.jit
    def run_with_residual_stop(
        x0,
        residual_stop_tol_j,
        n_total_j,
        abs_abund_j,
        wII_ne_j,
        wIII_ne2_j,
        neutral_stoich_j,
        neutral_log_nK_j,
        charged_idx1_j,
        charged_idx2_j,
        charged_log_nK_j,
        charged_stoich_j,
    ):
        state = solver.init_state(
            x0,
            n_total_j,
            abs_abund_j,
            wII_ne_j,
            wIII_ne2_j,
            neutral_stoich_j,
            neutral_log_nK_j,
            charged_idx1_j,
            charged_idx2_j,
            charged_log_nK_j,
            charged_stoich_j,
        )

        def cond(carry):
            _, state_i = carry
            return (
                (state_i.iter_num < maxiter_local)
                & (state_i.error > tol_local)
                & (jnp.max(jnp.abs(state_i.residual)) > residual_stop_tol_j)
            )

        def body(carry):
            params_i, state_i = carry
            step = solver.update(
                params_i,
                state_i,
                n_total_j,
                abs_abund_j,
                wII_ne_j,
                wIII_ne2_j,
                neutral_stoich_j,
                neutral_log_nK_j,
                charged_idx1_j,
                charged_idx2_j,
                charged_log_nK_j,
                charged_stoich_j,
            )
            return step.params, step.state

        return jax.lax.while_loop(cond, body, (x0, state))

    _JAXOPT_LM_RUNNER_CACHE[cache_key] = run_with_residual_stop
    if len(_JAXOPT_LM_RUNNER_CACHE) > _JAXOPT_LM_RUNNER_CACHE_MAX:
        _JAXOPT_LM_RUNNER_CACHE.popitem(last=False)
    return run_with_residual_stop


def chemical_equilibrium_jaxopt(
    temp: float,
    nt: float,
    model_atm_ne: float,
    absolute_abundances,
    ionization_energies: Dict[int, Tuple],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Dict = None,
    *,
    maxiter: int = 80,
    tol: float = 1e-8,
    method: str = "auto",
    jit: bool = True,
    lm_solver: str = "auto",
    materialize_jac: bool = True,
    implicit_diff: bool = False,
    newton_damping: float = 0.0,
    residual_stop_tol: Optional[float] = None,
    hybrid_newton_maxiter: int = 8,
    hybrid_newton_damping: float = 1e-6,
    hybrid_newton_restarts: int = 2,
    hybrid_newton_damping_growth: float = 100.0,
    hybrid_line_search_max_steps: int = 6,
    hybrid_line_search_backtrack: float = 0.5,
    hybrid_line_search_armijo_c: float = 1e-4,
    hybrid_trigger_residual: float = 1e-1,
    initial_x: Optional[np.ndarray] = None,
    return_species: bool = True,
    return_x: bool = False,
    stats_out: Optional[Dict] = None,
):
    """
    Single-layer CE solve using JAX nonlinear root finding.

    method:
      - "auto": use compiled Newton for atomic-only, LM for molecular.
      - "newton": force Newton (atomic-only).
      - "lm": force Levenberg-Marquardt.
      - "hybrid": LM + Newton(line-search) for molecular fallback/refinement.
    In "auto", hybrid refinement is only triggered when LM residual is large
    (greater than max(residual_stop_tol, hybrid_trigger_residual)).
    lm_solver:
      - "auto": try cholesky, then lu, then qr.
      - explicit choices: cholesky, lu, qr, inv, svd.
    Returns `(ne, species_densities)` by default. If `return_species=False`,
    returns only `ne` (or `(ne, x_solution)` when `return_x=True`).
    """
    abs_abund_array = _coerce_absolute_abundance_array(absolute_abundances)

    temperature = float(temp)
    n_total = float(nt)

    # Reuse cached Saha weights from the validated SciPy path.
    wII_ne, wIII_ne2 = _get_cached_saha_weight_arrays(
        temperature, ionization_energies, partition_funcs
    )

    mol_data = _prepare_molecule_arrays(temperature, log_equilibrium_constants)
    (
        neutral_stoich_np,
        neutral_log_nK_np,
        charged_idx1_np,
        charged_idx2_np,
        charged_log_nK_np,
        charged_stoich_np,
    ) = _pack_molecule_arrays(mol_data)

    method_norm = str(method).lower()
    if method_norm not in ("auto", "newton", "lm", "hybrid"):
        raise ValueError(f"Unknown method '{method}'. Expected one of: auto, newton, lm, hybrid.")
    use_newton = (method_norm == "newton") or (method_norm in ("auto", "hybrid") and mol_data is None)
    if use_newton and mol_data is not None:
        raise ValueError("method='newton' only supports atomic-only equilibrium (no molecules).")

    # Initial guess using model atmosphere ne (same as SciPy)
    ne_guess = max(float(model_atm_ne), 1e-300)
    inv_ne_guess = 1.0 / ne_guess
    inv_ne_guess2 = inv_ne_guess * inv_ne_guess
    wII_guess = wII_ne * inv_ne_guess
    wIII_guess = wIII_ne2 * inv_ne_guess2

    has_energy = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)
    for Z in ionization_energies.keys():
        if 1 <= Z <= MAX_ATOMIC_NUMBER:
            has_energy[Z - 1] = True

    neutral_fraction_guess = np.where(
        has_energy,
        1.0 / (1.0 + wII_guess + wIII_guess),
        0.0,
    )

    default_x0 = np.concatenate([neutral_fraction_guess, [ne_guess / n_total * 1e5]])
    used_initial_x = False
    if initial_x is not None:
        x0 = np.asarray(initial_x, dtype=np.float64)
        if x0.shape != (MAX_ATOMIC_NUMBER + 1,):
            raise ValueError(
                f"initial_x must have shape ({MAX_ATOMIC_NUMBER + 1},), got {x0.shape}."
            )
        if not np.all(np.isfinite(x0)):
            x0 = default_x0
        used_initial_x = True
    else:
        x0 = default_x0

    x0_j = jnp.asarray(x0, dtype=jnp.float64)
    n_total_j = jnp.asarray(n_total, dtype=jnp.float64)
    abs_abund_j = jnp.asarray(abs_abund_array, dtype=jnp.float64)
    wII_ne_j = jnp.asarray(wII_ne, dtype=jnp.float64)
    wIII_ne2_j = jnp.asarray(wIII_ne2, dtype=jnp.float64)
    if use_newton:
        x_j, iter_j, err_j = _ATOMIC_NEWTON_SOLVE_JIT(
            x0_j,
            n_total_j,
            abs_abund_j,
            wII_ne_j,
            wIII_ne2_j,
            jnp.asarray(int(maxiter), dtype=jnp.int32),
            jnp.asarray(float(tol), dtype=jnp.float64),
            jnp.asarray(float(newton_damping), dtype=jnp.float64),
        )
        x = np.asarray(jax.device_get(x_j), dtype=np.float64)
        iter_num = int(jax.device_get(iter_j))
        error = float(jax.device_get(err_j))
        nfev = iter_num + 1
        residual_max = float(error)
        method_used = "newton"
        lm_solver_used = None
        lm_attempts = []
        lm_fallback_used = False
        hybrid_used = False
        hybrid_attempted = False
        hybrid_accepted = False
        hybrid_stats = {}
    else:
        neutral_stoich_j = jnp.asarray(neutral_stoich_np, dtype=jnp.float64)
        neutral_log_nK_j = jnp.asarray(neutral_log_nK_np, dtype=jnp.float64)
        charged_idx1_j = jnp.asarray(charged_idx1_np, dtype=jnp.int32)
        charged_idx2_j = jnp.asarray(charged_idx2_np, dtype=jnp.int32)
        charged_log_nK_j = jnp.asarray(charged_log_nK_np, dtype=jnp.float64)
        charged_stoich_j = jnp.asarray(charged_stoich_np, dtype=jnp.float64)

        residual_stop_tol_val = float(tol) if residual_stop_tol is None else float(residual_stop_tol)
        residual_stop_tol_j = jnp.asarray(residual_stop_tol_val, dtype=jnp.float64)
        hybrid_used = False
        hybrid_attempted = False
        hybrid_accepted = False
        hybrid_stats = {}
        lm_solver_sequence = _resolve_lm_solver_sequence(lm_solver)
        lm_attempts = []
        lm_solver_used = None
        lm_fallback_used = False
        lm_last_exc = None

        for attempt_idx, lm_solver_try in enumerate(lm_solver_sequence):
            try:
                solver = _get_cached_lm_solver(
                    maxiter=maxiter,
                    tol=tol,
                    jit=jit,
                    lm_solver=lm_solver_try,
                    materialize_jac=materialize_jac,
                    implicit_diff=implicit_diff,
                    neutral_stoich_shape=neutral_stoich_np.shape,
                    charged_shape=int(charged_idx1_np.shape[0]),
                )

                if jit:
                    runner = _get_cached_lm_runner(
                        (
                            int(maxiter),
                            float(tol),
                            bool(jit),
                            str(lm_solver_try),
                            bool(materialize_jac),
                            bool(implicit_diff),
                            tuple(int(v) for v in neutral_stoich_np.shape),
                            int(charged_idx1_np.shape[0]),
                        ),
                        solver,
                    )
                    x_j, state = runner(
                        x0_j,
                        residual_stop_tol_j,
                        n_total_j,
                        abs_abund_j,
                        wII_ne_j,
                        wIII_ne2_j,
                        neutral_stoich_j,
                        neutral_log_nK_j,
                        charged_idx1_j,
                        charged_idx2_j,
                        charged_log_nK_j,
                        charged_stoich_j,
                    )
                    x = np.asarray(jax.device_get(x_j), dtype=np.float64)
                    iter_num = int(jax.device_get(state.iter_num))
                    error = float(jax.device_get(state.error))
                    residual_max = float(jax.device_get(jnp.max(jnp.abs(state.residual))))
                    nfev = -1
                else:
                    opt = solver.run(
                        x0_j,
                        n_total_j,
                        abs_abund_j,
                        wII_ne_j,
                        wIII_ne2_j,
                        neutral_stoich_j,
                        neutral_log_nK_j,
                        charged_idx1_j,
                        charged_idx2_j,
                        charged_log_nK_j,
                        charged_stoich_j,
                    )
                    x = np.asarray(jax.device_get(opt.params), dtype=np.float64)
                    iter_num = getattr(opt.state, "iter_num", -1)
                    error = getattr(opt.state, "error", np.nan)
                    nfev = getattr(opt.state, "num_fun_eval", None)
                    residual = getattr(opt.state, "residual", None)
                    try:
                        iter_num = int(jax.device_get(iter_num))
                    except Exception:
                        iter_num = int(iter_num)
                    try:
                        error = float(jax.device_get(error))
                    except Exception:
                        error = float(error)
                    if residual is not None:
                        try:
                            residual_max = float(jax.device_get(jnp.max(jnp.abs(residual))))
                        except Exception:
                            residual_max = float(np.max(np.abs(np.asarray(residual, dtype=np.float64))))
                    else:
                        residual_max = np.nan
                    if nfev is None:
                        nfev = -1
                    else:
                        try:
                            nfev = int(jax.device_get(nfev))
                        except Exception:
                            nfev = int(nfev)

                if np.all(np.isfinite(x)) and np.isfinite(error) and np.isfinite(residual_max):
                    lm_solver_used = lm_solver_try
                    lm_fallback_used = attempt_idx > 0
                    break

                lm_attempts.append(f"{lm_solver_try}: non-finite result")
            except Exception as exc:
                lm_last_exc = exc
                lm_attempts.append(f"{lm_solver_try}: {exc}")

        if lm_solver_used is None:
            details = "; ".join(lm_attempts) if lm_attempts else "no attempt details"
            if lm_last_exc is not None:
                raise RuntimeError(f"LM solve failed ({details})") from lm_last_exc
            raise RuntimeError(f"LM solve failed ({details})")
        method_used = "lm"

        hybrid_trigger = max(float(residual_stop_tol_val), float(hybrid_trigger_residual))
        should_try_hybrid = (
            mol_data is not None
            and method_norm in ("auto", "hybrid")
            and (
                method_norm == "hybrid"
                or (not np.isfinite(residual_max))
                or (float(residual_max) > hybrid_trigger)
            )
        )
        if should_try_hybrid:
            hybrid_attempted = True
            restarts = max(int(hybrid_newton_restarts), 1)
            damping_val = max(float(hybrid_newton_damping), 1e-12)
            damping_growth = max(float(hybrid_newton_damping_growth), 1.0)
            hybrid_runs = []
            x_seed = x
            best_x = x
            best_res = float(residual_max)
            if not np.isfinite(best_res):
                best_res = np.inf
            hybrid_total_iter = 0
            hybrid_total_nfev = 0

            for restart_idx in range(restarts):
                x_hybrid, run_stats = _run_molecular_newton_line_search(
                    x_seed,
                    residual_stop_tol=float(residual_stop_tol_val),
                    n_total_j=n_total_j,
                    abs_abund_j=abs_abund_j,
                    wII_ne_j=wII_ne_j,
                    wIII_ne2_j=wIII_ne2_j,
                    neutral_stoich_j=neutral_stoich_j,
                    neutral_log_nK_j=neutral_log_nK_j,
                    charged_idx1_j=charged_idx1_j,
                    charged_idx2_j=charged_idx2_j,
                    charged_log_nK_j=charged_log_nK_j,
                    charged_stoich_j=charged_stoich_j,
                    maxiter=int(hybrid_newton_maxiter),
                    damping=float(damping_val),
                    line_search_max_steps=int(hybrid_line_search_max_steps),
                    line_search_backtrack=float(hybrid_line_search_backtrack),
                    line_search_armijo_c=float(hybrid_line_search_armijo_c),
                )
                run_stats = dict(run_stats)
                run_stats["restart"] = int(restart_idx + 1)
                run_stats["damping"] = float(damping_val)
                hybrid_runs.append(run_stats)

                run_res = float(run_stats.get("residual_max", np.inf))
                hybrid_total_iter += int(run_stats.get("iter_num", 0))
                hybrid_total_nfev += int(run_stats.get("num_fun_eval", 0))

                if np.isfinite(run_res) and run_res <= best_res:
                    best_res = run_res
                    best_x = x_hybrid
                    x_seed = x_hybrid

                if bool(run_stats.get("converged_residual", False)):
                    break

                damping_val = min(damping_val * damping_growth, 1e2)

            hybrid_stats = {
                "runs": hybrid_runs,
                "iter_num": int(hybrid_total_iter),
                "num_fun_eval": int(hybrid_total_nfev),
                "residual_max": float(best_res),
                "converged_residual": bool(best_res <= float(residual_stop_tol_val)),
            }

            if np.isfinite(best_res) and ((not np.isfinite(residual_max)) or (best_res <= float(residual_max))):
                x = best_x
                residual_max = best_res
                error = best_res if not np.isfinite(error) else min(float(error), best_res)
                iter_num = int(iter_num) + int(hybrid_total_iter)
                if int(nfev) >= 0:
                    nfev = int(nfev) + int(hybrid_total_nfev)
                else:
                    nfev = int(hybrid_total_nfev)
                method_used = "lm+newton_ls"
                hybrid_used = True
                hybrid_accepted = True

    ne = float(abs(x[-1]) * n_total * 1e-5)

    neutral_fractions = np.abs(x[:-1])
    n0 = (n_total - ne) * abs_abund_array * neutral_fractions
    inv_ne = 1.0 / max(float(ne), 1e-300)
    wII_sol = wII_ne * inv_ne
    wIII_sol = wIII_ne2 * (inv_ne * inv_ne)

    species_densities: Optional[Dict[Species, float]] = None
    if return_species:
        species_densities = {}
        for Z in range(1, MAX_ATOMIC_NUMBER + 1):
            n_neutral = float(n0[Z - 1])
            species_densities[_SPECIES_NEUTRAL[Z - 1]] = n_neutral
            if Z in ionization_energies:
                species_densities[_SPECIES_ION_1[Z - 1]] = float(wII_sol[Z - 1] * n_neutral)
                species_densities[_SPECIES_ION_2[Z - 1]] = float(wIII_sol[Z - 1] * n_neutral)

        if mol_data is not None:
            log_n0 = np.log10(np.clip(n0, 1e-300, None))

            neutral_species = tuple(mol_data.get("neutral_species", ()))
            if neutral_species:
                neutral_log_nK = np.asarray(mol_data["neutral_log_nK"], dtype=np.float64)
                neutral_stoich = np.asarray(mol_data["neutral_stoich"], dtype=np.float64)
                neutral_log_n_mol = neutral_stoich @ log_n0 - neutral_log_nK
                finite_mask = np.isfinite(neutral_log_n_mol)
                neutral_vals = np.zeros_like(neutral_log_n_mol)
                neutral_vals[finite_mask] = np.power(
                    10.0,
                    np.clip(neutral_log_n_mol[finite_mask], _LOG10_MIN, _LOG10_MAX),
                )
                for spec, value in zip(neutral_species, neutral_vals):
                    species_densities[spec] = float(value)

            charged_species = tuple(mol_data.get("charged_species", ()))
            if charged_species:
                charged_log_nK = np.asarray(mol_data["charged_log_nK"], dtype=np.float64)
                charged_idx1 = np.asarray(mol_data["charged_idx1"], dtype=np.int32)
                charged_idx2 = np.asarray(mol_data["charged_idx2"], dtype=np.int32)
                log_wII = np.log10(np.clip(wII_sol, 1e-300, None))
                charged_log_n_mol = log_n0[charged_idx1] + log_wII[charged_idx1] + log_n0[charged_idx2] - charged_log_nK
                finite_mask = np.isfinite(charged_log_n_mol)
                charged_vals = np.zeros_like(charged_log_n_mol)
                charged_vals[finite_mask] = np.power(
                    10.0,
                    np.clip(charged_log_n_mol[finite_mask], _LOG10_MIN, _LOG10_MAX),
                )
                for spec, value in zip(charged_species, charged_vals):
                    species_densities[spec] = float(value)

    if isinstance(stats_out, dict):
        stats_out.clear()
        stats_out.update(
            {
                "method": method_used,
                "iter_num": iter_num,
                "error": error,
                "residual_max": residual_max,
                "num_fun_eval": nfev,
                "maxiter": int(maxiter),
                "tol": float(tol),
                "converged": bool(error <= float(tol)),
                "converged_residual": bool(residual_max <= (float(tol) if residual_stop_tol is None else float(residual_stop_tol))),
                "residual_stop_tol": float(tol) if residual_stop_tol is None else float(residual_stop_tol),
                "jit": bool(jit),
                "lm_solver": lm_solver_used if lm_solver_used is not None else str(lm_solver),
                "lm_solver_requested": str(lm_solver),
                "lm_fallback_used": bool(lm_fallback_used),
                "lm_attempts": list(lm_attempts),
                "hybrid_attempted": bool(hybrid_attempted),
                "hybrid_used": bool(hybrid_used),
                "hybrid_accepted": bool(hybrid_accepted),
                "hybrid_stats": dict(hybrid_stats),
                "hybrid_newton_maxiter": int(hybrid_newton_maxiter),
                "hybrid_newton_damping": float(hybrid_newton_damping),
                "hybrid_newton_restarts": int(hybrid_newton_restarts),
                "hybrid_newton_damping_growth": float(hybrid_newton_damping_growth),
                "hybrid_line_search_max_steps": int(hybrid_line_search_max_steps),
                "hybrid_line_search_backtrack": float(hybrid_line_search_backtrack),
                "hybrid_line_search_armijo_c": float(hybrid_line_search_armijo_c),
                "hybrid_trigger_residual": float(hybrid_trigger_residual),
                "materialize_jac": bool(materialize_jac),
                "implicit_diff": bool(implicit_diff),
                "newton_damping": float(newton_damping),
                "used_initial_x": bool(used_initial_x),
            }
        )

    if return_x:
        if return_species:
            return ne, species_densities, x.copy()
        return ne, x.copy()
    if return_species:
        return ne, species_densities
    return ne


def chemical_equilibrium_jaxopt_layers(
    temps: np.ndarray,
    nts: np.ndarray,
    model_atm_nes: np.ndarray,
    absolute_abundances,
    ionization_energies: Dict[int, Tuple],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Dict = None,
    *,
    warm_start: bool = True,
    return_species: bool = False,
    **solver_kwargs,
):
    """
    Solve chemical equilibrium for multiple layers sequentially.

    Parameters
    ----------
    temps, nts, model_atm_nes : array-like
        Layer-wise temperature, total number density, and model-atmosphere
        electron-density guesses.
    warm_start : bool, default=True
        If True, reuse the previous layer solution x as the next initial guess.
    return_species : bool, default=False
        If True, also return per-layer species density dictionaries.
    **solver_kwargs
        Additional kwargs passed through to chemical_equilibrium_jaxopt().

    Returns
    -------
    Tuple
        If return_species=False: (ne_array, x_array)
        If return_species=True:  (ne_array, species_list, x_array)
    """
    temps = np.asarray(temps, dtype=np.float64)
    nts = np.asarray(nts, dtype=np.float64)
    model_atm_nes = np.asarray(model_atm_nes, dtype=np.float64)
    if not (temps.shape == nts.shape == model_atm_nes.shape):
        raise ValueError("temps, nts, and model_atm_nes must have identical shapes.")

    n_layers = int(temps.size)
    abs_abund_array = _coerce_absolute_abundance_array(absolute_abundances)
    ne_array = np.zeros(n_layers, dtype=np.float64)
    x_array = np.zeros((n_layers, MAX_ATOMIC_NUMBER + 1), dtype=np.float64)
    species_list: List[Dict[Species, float]] = [] if return_species else []

    x_prev = None
    for i in range(n_layers):
        kwargs = dict(solver_kwargs)
        kwargs["return_x"] = True
        if warm_start and x_prev is not None and "initial_x" not in kwargs:
            kwargs["initial_x"] = x_prev

        ne_i, species_i, x_i = chemical_equilibrium_jaxopt(
            temp=float(temps[i]),
            nt=float(nts[i]),
            model_atm_ne=float(model_atm_nes[i]),
            absolute_abundances=abs_abund_array,
            ionization_energies=ionization_energies,
            partition_funcs=partition_funcs,
            log_equilibrium_constants=log_equilibrium_constants,
            **kwargs,
        )
        ne_array[i] = float(ne_i)
        x_array[i] = np.asarray(x_i, dtype=np.float64)
        x_prev = x_array[i]
        if return_species:
            species_list.append(species_i)

    if return_species:
        return ne_array, species_list, x_array
    return ne_array, x_array


__all__ = ["chemical_equilibrium_jaxopt", "chemical_equilibrium_jaxopt_layers"]
