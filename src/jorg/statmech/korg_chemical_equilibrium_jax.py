"""
Experimental JAX-based chemical equilibrium solver.

This module provides a vectorized residual function and a Newton solver
implemented with JAX autodiff and lax.while_loop. It is intended for
benchmarking and future integration, not yet used by default.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
except Exception as exc:  # pragma: no cover - handled at runtime
    jax = None
    jnp = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None

from .species import Species, MAX_ATOMIC_NUMBER
from .korg_chemical_equilibrium import _get_cached_saha_weight_arrays
from ..constants import kboltz_cgs


def _require_jax():
    if jax is None:
        raise RuntimeError(f"JAX is required for chemical_equilibrium_jax: {_JAX_IMPORT_ERROR}")


def _prepare_molecule_arrays(
    temperature: float,
    log_equilibrium_constants: Optional[Dict],
):
    if log_equilibrium_constants is None:
        return None

    log_T = float(np.log(temperature))
    log_kT = float(np.log10(kboltz_cgs * temperature))

    neutral_mols = []
    charged_mols = []
    max_unique = 0

    for mol_species, log_K_func in log_equilibrium_constants.items():
        try:
            atoms = tuple(mol_species.get_atoms())
            if len(atoms) == 0:
                continue
            logKp = float(log_K_func(log_T))  # log10(K_p)
            log_nK = logKp - (len(atoms) - 1.0) * log_kT
            if not np.isfinite(log_nK):
                continue

            atom_indices = np.asarray([Z - 1 for Z in atoms], dtype=np.int64)
            if int(getattr(mol_species, "charge", 0)) == 1:
                if len(atom_indices) < 2:
                    continue
                charged_mols.append((int(atom_indices[0]), int(atom_indices[1]), float(log_nK)))
            else:
                uniq, counts = np.unique(atom_indices, return_counts=True)
                max_unique = max(max_unique, int(uniq.size))
                neutral_mols.append((uniq.astype(np.int64), counts.astype(np.float64), float(log_nK)))
        except Exception:
            continue

    result = {}

    if neutral_mols:
        n_neutral = len(neutral_mols)
        idx = -np.ones((n_neutral, max_unique), dtype=np.int32)
        counts = np.zeros((n_neutral, max_unique), dtype=np.float64)
        log_nK = np.zeros(n_neutral, dtype=np.float64)

        for i, (uniq, stoich, log_val) in enumerate(neutral_mols):
            n = int(uniq.size)
            idx[i, :n] = uniq
            counts[i, :n] = stoich
            log_nK[i] = log_val

        result["neutral_idx"] = idx
        result["neutral_counts"] = counts
        result["neutral_log_nK"] = log_nK

    if charged_mols:
        idx1 = np.array([c[0] for c in charged_mols], dtype=np.int32)
        idx2 = np.array([c[1] for c in charged_mols], dtype=np.int32)
        log_nK = np.array([c[2] for c in charged_mols], dtype=np.float64)
        result["charged_idx1"] = idx1
        result["charged_idx2"] = idx2
        result["charged_log_nK"] = log_nK

    return result or None


def _build_residual_function(
    n_total: float,
    abs_abund_array: np.ndarray,
    wII_ne: np.ndarray,
    wIII_ne2: np.ndarray,
    mol_data: Optional[Dict],
):
    _require_jax()

    abs_abund_j = jnp.asarray(abs_abund_array, dtype=jnp.float64)
    wII_ne_j = jnp.asarray(wII_ne, dtype=jnp.float64)
    wIII_ne2_j = jnp.asarray(wIII_ne2, dtype=jnp.float64)
    n_total_j = jnp.asarray(float(n_total), dtype=jnp.float64)

    has_neutral = bool(mol_data and mol_data.get("neutral_idx") is not None)
    has_charged = bool(mol_data and mol_data.get("charged_idx1") is not None)

    if has_neutral:
        neutral_idx = jnp.asarray(mol_data["neutral_idx"], dtype=jnp.int32)
        neutral_counts = jnp.asarray(mol_data["neutral_counts"], dtype=jnp.float64)
        neutral_log_nK = jnp.asarray(mol_data["neutral_log_nK"], dtype=jnp.float64)
    if has_charged:
        charged_idx1 = jnp.asarray(mol_data["charged_idx1"], dtype=jnp.int32)
        charged_idx2 = jnp.asarray(mol_data["charged_idx2"], dtype=jnp.int32)
        charged_log_nK = jnp.asarray(mol_data["charged_log_nK"], dtype=jnp.float64)

    def residuals(x):
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
            n_mol = jnp.power(10.0, log_n_mol)
            R_elem = R_elem.at[charged_idx1].add(-n_mol)
            R_elem = R_elem.at[charged_idx2].add(-n_mol)
            R_charge = R_charge + jnp.sum(n_mol)

        if has_neutral:
            idx_clip = jnp.where(neutral_idx >= 0, neutral_idx, 0)
            log_sel = log_neutral[idx_clip]
            mask = neutral_idx >= 0
            log_sel = jnp.where(mask, log_sel, 0.0)
            log_n_mol = jnp.sum(neutral_counts * log_sel, axis=1) - neutral_log_nK
            n_mol = jnp.power(10.0, log_n_mol)

            contrib = -neutral_counts * n_mol[:, None]
            contrib = jnp.where(mask, contrib, 0.0)
            idx_flat = idx_clip.reshape(-1)
            contrib_flat = contrib.reshape(-1)
            R_elem = R_elem.at[idx_flat].add(contrib_flat)

        denom_elem = jnp.maximum(atom_number_densities, 1e-100)
        denom_charge = ne * 1e-5

        F_elem = R_elem / denom_elem
        F_charge = R_charge / denom_charge
        return jnp.concatenate((F_elem, jnp.array([F_charge], dtype=jnp.float64)))

    return residuals


def _build_residual_and_jacobian_atomic(
    n_total: float,
    abs_abund_array: np.ndarray,
    wII_ne: np.ndarray,
    wIII_ne2: np.ndarray,
):
    _require_jax()

    abs_abund_j = jnp.asarray(abs_abund_array, dtype=jnp.float64)
    wII_ne_j = jnp.asarray(wII_ne, dtype=jnp.float64)
    wIII_ne2_j = jnp.asarray(wIII_ne2, dtype=jnp.float64)
    n_total_j = jnp.asarray(float(n_total), dtype=jnp.float64)

    def _sign_no_zero(x):
        return jnp.where(x == 0, 1.0, jnp.sign(x))

    def residuals(x):
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

    def jacobian(x):
        x = jnp.asarray(x, dtype=jnp.float64)
        sign_ne = _sign_no_zero(x[-1])
        sign_f = _sign_no_zero(x[:-1])
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

    return residuals, jacobian


def _newton_solve(residuals, x0, maxiter: int, tol: float, damping: float, jac_fn=None):
    _require_jax()

    if jac_fn is None:
        jac_fn = jax.jacfwd(residuals)
    n_dim = int(x0.shape[0])
    eye = jnp.eye(n_dim, dtype=jnp.float64)

    def step(state):
        x, it, _ = state
        F = residuals(x)
        J = jac_fn(x)
        if damping > 0.0:
            J = J + damping * eye
        delta = jnp.linalg.solve(J, -F)
        x_new = x + delta
        err = jnp.max(jnp.abs(F))
        return (x_new, it + 1, err)

    def cond(state):
        _, it, err = state
        return (it < maxiter) & (err > tol)

    return jax.lax.while_loop(cond, step, (x0, 0, jnp.inf))


def chemical_equilibrium_jax(
    temp: float,
    nt: float,
    model_atm_ne: float,
    absolute_abundances: Dict[int, float],
    ionization_energies: Dict[int, Tuple],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Dict = None,
    *,
    maxiter: int = 60,
    tol: float = 1e-8,
    damping: float = 0.0,
    jit: bool = False,
    analytic_jacobian: bool = False,
    stats_out: Optional[Dict] = None,
    initial_ne: Optional[float] = None,
):
    """
    JAX Newton solver for a single-layer chemical equilibrium solve.

    Returns (ne, species_densities). This is experimental and not part of the
    default pipeline.
    """
    _require_jax()

    if isinstance(absolute_abundances, dict):
        abs_abund_array = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
        for Z, abund in absolute_abundances.items():
            if 1 <= Z <= MAX_ATOMIC_NUMBER:
                abs_abund_array[Z - 1] = abund
    else:
        abs_abund_array = np.asarray(absolute_abundances, dtype=np.float64)
        if abs_abund_array.shape[0] != MAX_ATOMIC_NUMBER:
            raise ValueError(
                f"absolute_abundances must have length {MAX_ATOMIC_NUMBER} (got {abs_abund_array.shape[0]})."
            )

    temperature = float(temp)
    n_total = float(nt)

    wII_ne, wIII_ne2 = _get_cached_saha_weight_arrays(
        temperature, ionization_energies, partition_funcs
    )

    mol_data = _prepare_molecule_arrays(temperature, log_equilibrium_constants)
    use_analytic = bool(analytic_jacobian and mol_data is None)
    if use_analytic:
        residuals, jac_fn = _build_residual_and_jacobian_atomic(
            n_total, abs_abund_array, wII_ne, wIII_ne2
        )
    else:
        residuals = _build_residual_function(n_total, abs_abund_array, wII_ne, wIII_ne2, mol_data)
        jac_fn = None

    ne_guess = max(float(model_atm_ne), 1e-300)
    ne_initial = ne_guess if initial_ne is None else max(float(initial_ne), 1e-300)
    inv_ne = 1.0 / ne_initial
    inv_ne2 = inv_ne * inv_ne
    wII = wII_ne * inv_ne
    wIII = wIII_ne2 * inv_ne2
    has_energy = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)
    for Z in ionization_energies.keys():
        if 1 <= Z <= MAX_ATOMIC_NUMBER:
            has_energy[Z - 1] = True
    neutral_fraction_guess = np.where(
        has_energy,
        1.0 / (1.0 + wII + wIII),
        0.0,
    )
    x0 = np.concatenate([neutral_fraction_guess, [ne_initial / n_total * 1e5]])

    x0_j = jnp.asarray(x0, dtype=jnp.float64)

    solve_fn = _newton_solve
    if jit:
        solve_fn = jax.jit(_newton_solve, static_argnums=(0, 2, 3, 4, 5))

    x_final, iter_num, err = solve_fn(
        residuals, x0_j, int(maxiter), float(tol), float(damping), jac_fn
    )
    x_final = np.asarray(jax.device_get(x_final), dtype=np.float64)

    if isinstance(stats_out, dict):
        stats_out.clear()
        stats_out.update(
            {
                "iter_num": int(jax.device_get(iter_num)),
                "error": float(jax.device_get(err)),
                "maxiter": int(maxiter),
                "tol": float(tol),
                "damping": float(damping),
                "jit": bool(jit),
                "analytic_jacobian": bool(use_analytic),
                "converged": float(jax.device_get(err)) <= float(tol),
            }
        )

    neutral_fractions = np.abs(x_final[:-1])
    ne = float(abs(x_final[-1]) * n_total * 1e-5)

    species_densities: Dict[Species, float] = {}
    n0 = (n_total - ne) * abs_abund_array * neutral_fractions
    inv_ne_sol = 1.0 / max(float(ne), 1e-300)
    wII_sol = wII_ne * inv_ne_sol
    wIII_sol = wIII_ne2 * (inv_ne_sol * inv_ne_sol)

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        n_neutral = float(n0[Z - 1])
        species_densities[Species.from_atomic_number(Z, 0)] = n_neutral
        if Z in ionization_energies:
            species_densities[Species.from_atomic_number(Z, 1)] = float(wII_sol[Z - 1] * n_neutral)
            species_densities[Species.from_atomic_number(Z, 2)] = float(wIII_sol[Z - 1] * n_neutral)

    if log_equilibrium_constants is not None:
        log_T = np.log(temperature)
        for mol_species, log_K_func in log_equilibrium_constants.items():
            try:
                log_K_partial = log_K_func(log_T)  # log10(K_p)
                n_atoms = len(mol_species.formula.atoms)
                log_nK = log_K_partial - (n_atoms - 1) * np.log10(kboltz_cgs * temperature)

                atoms = list(mol_species.get_atoms())

                if mol_species.charge == 1:
                    Z1, Z2 = atoms[0], atoms[1]
                    n1_II = species_densities[Species.from_atomic_number(Z1, 1)]
                    n2_I = species_densities[Species.from_atomic_number(Z2, 0)]

                    if n1_II > 0 and n2_I > 0:
                        n_mol = 10 ** (np.log10(n1_II) + np.log10(n2_I) - log_nK)
                        species_densities[mol_species] = n_mol
                else:
                    element_log_ns = [
                        np.log10(species_densities[Species.from_atomic_number(Z, 0)]) for Z in atoms
                    ]
                    if all(np.isfinite(element_log_ns)):
                        n_mol = 10 ** (sum(element_log_ns) - log_nK)
                        species_densities[mol_species] = n_mol
            except (KeyError, ValueError):
                continue

    return ne, species_densities


__all__ = ["chemical_equilibrium_jax"]
