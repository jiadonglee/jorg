"""
Complete Korg.jl Chemical Equilibrium Solver
============================================

Direct port of Korg.jl's chemical_equilibrium function from src/statmech.jl:120-343

This is the COMPLETE solver that simultaneously solves:
- Element conservation equations (92 elements)
- Charge neutrality constraint
- Saha ionization equilibrium
- Molecular equilibrium constants

Source: Korg.jl/src/statmech.jl:120-343
"""

import numpy as np
from scipy.optimize import root
from typing import Dict, Tuple, Callable
from collections import OrderedDict
import warnings

from .species import Species, MAX_ATOMIC_NUMBER
from ..constants import kboltz_eV, kboltz_cgs, me_cgs, hplanck_cgs


_SAHA_WEIGHT_CACHE = OrderedDict()
_SAHA_WEIGHT_CACHE_MAX = 256


def _sign_no_zero(x):
    """sign(x) but treat 0 as +1 to keep derivatives well-defined."""
    s = np.sign(x)
    if np.isscalar(s):
        return 1.0 if s == 0 else float(s)
    s = s.astype(np.float64, copy=False)
    s[s == 0] = 1.0
    return s


def translational_U(mass_cgs: float, temperature: float) -> float:
    """
    Translational partition function contribution for free particles

    Source: Korg.jl/src/statmech.jl:48-52

    Parameters
    ----------
    mass_cgs : float
        Particle mass in grams
    temperature : float
        Temperature in K

    Returns
    -------
    float
        Translational partition function factor
    """
    return (2 * np.pi * mass_cgs * kboltz_cgs * temperature / hplanck_cgs**2)**1.5


def _compute_saha_weight_arrays(
    temperature: float,
    ionization_energies: Dict[int, Tuple[float, float, float]],
    partition_funcs: Dict[Species, Callable],
) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute wII*ne and wIII*ne^2 arrays (ne-independent)."""
    wII_ne = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
    wIII_ne2 = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)

    log_T = float(np.log(temperature))
    log_trans_U = float(np.log(translational_U(me_cgs, temperature)))
    inv_kT = 1.0 / (kboltz_eV * temperature)

    pf_cache = {}

    def _get_U(species: Species) -> float:
        if species in pf_cache:
            return pf_cache[species]
        val = float(partition_funcs[species](log_T))
        pf_cache[species] = val
        return val

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z not in ionization_energies:
            continue
        chi_I, chi_II, _ = ionization_energies[Z]

        species_I = Species.from_atomic_number(Z, 0)
        species_II = Species.from_atomic_number(Z, 1)

        U_I = _get_U(species_I)
        U_II = _get_U(species_II)
        log_U_I = np.log(max(U_I, 1e-300))
        log_U_II = np.log(max(U_II, 1e-300))

        log_wII_ne = (
            np.log(2.0)
            + log_U_II
            - log_U_I
            + log_trans_U
            - chi_I * inv_kT
        )
        wII_ne_val = float(np.exp(log_wII_ne))
        wII_ne[Z - 1] = wII_ne_val

        if Z == 1:
            continue
        species_III = Species.from_atomic_number(Z, 2)
        if species_III in partition_funcs:
            U_III = _get_U(species_III)
            log_U_III = np.log(max(U_III, 1e-300))
            log_wIII_ne2 = (
                log_wII_ne
                + np.log(2.0)
                + log_U_III
                - log_U_II
                + log_trans_U
                - chi_II * inv_kT
            )
            wIII_ne2[Z - 1] = float(np.exp(log_wIII_ne2))

    return wII_ne, wIII_ne2


def _get_cached_saha_weight_arrays(
    temperature: float,
    ionization_energies: Dict[int, Tuple[float, float, float]],
    partition_funcs: Dict[Species, Callable],
) -> Tuple[np.ndarray, np.ndarray]:
    key = (float(temperature), id(ionization_energies), id(partition_funcs))
    cached = _SAHA_WEIGHT_CACHE.get(key)
    if cached is not None:
        _SAHA_WEIGHT_CACHE.move_to_end(key)
        return cached

    wII_ne, wIII_ne2 = _compute_saha_weight_arrays(temperature, ionization_energies, partition_funcs)
    _SAHA_WEIGHT_CACHE[key] = (wII_ne, wIII_ne2)
    if len(_SAHA_WEIGHT_CACHE) > _SAHA_WEIGHT_CACHE_MAX:
        _SAHA_WEIGHT_CACHE.popitem(last=False)
    return wII_ne, wIII_ne2


def saha_ion_weights(temperature: float, ne: float, atomic_number: int,
                     ionization_energies: Dict[int, Tuple[float, float, float]],
                     partition_funcs: Dict[Species, Callable]) -> Tuple[float, float]:
    """
    Calculate Saha ionization weights for an element

    Returns (wII, wIII) where wII = n(X II)/n(X I) and wIII = n(X III)/n(X I)

    Source: Korg.jl/src/statmech.jl:18-35

    Parameters
    ----------
    temperature : float
        Temperature in K
    ne : float
        Electron number density in cm^-3
    atomic_number : int
        Atomic number Z
    ionization_energies : Dict
        Ionization energies in eV
    partition_funcs : Dict
        Partition function interpolators

    Returns
    -------
    Tuple[float, float]
        (wII, wIII) ionization weight factors
    """
    chi_I, chi_II, chi_III = ionization_energies[atomic_number]

    log_T = np.log(temperature)

    # Get partition functions (interpolators return U, not log(U))
    species_I = Species.from_atomic_number(atomic_number, 0)
    species_II = Species.from_atomic_number(atomic_number, 1)

    U_I = float(partition_funcs[species_I](log_T))
    U_II = float(partition_funcs[species_II](log_T))
    log_U_I = np.log(max(U_I, 1e-300))
    log_U_II = np.log(max(U_II, 1e-300))

    k = kboltz_eV
    trans_U = translational_U(me_cgs, temperature)

    # Saha equation for first ionization (in log space to avoid overflow)
    # wII = (n_II / n_I) = (2 * U_II / U_I) * (trans_U / ne) * exp(-χ_I / kT)
    # log(wII) = log(2) + log_U_II - log_U_I + log(trans_U) - log(ne) - χ_I/(kT)
    log_wII = (np.log(2.0) + log_U_II - log_U_I + np.log(trans_U) -
               np.log(ne) - chi_I / (k * temperature))
    wII = np.exp(log_wII)

    # Second ionization (skip for hydrogen)
    if atomic_number == 1:
        wIII = 0.0
    else:
        species_III = Species.from_atomic_number(atomic_number, 2)
        if species_III in partition_funcs:
            U_III = float(partition_funcs[species_III](log_T))
            log_U_III = np.log(max(U_III, 1e-300))
            # wIII = wII * (2 * U_III / U_II) * (trans_U / ne) * exp(-χ_II / kT)
            log_wIII = (log_wII + np.log(2.0) + log_U_III - log_U_II +
                       np.log(trans_U) - np.log(ne) - chi_II / (k * temperature))
            wIII = np.exp(log_wIII)
        else:
            wIII = 0.0

    return wII, wIII


def setup_chemical_equilibrium_residuals(
    temperature: float,
    n_total: float,
    absolute_abundances: np.ndarray,
    ionization_energies: Dict[int, Tuple],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Dict = None,
    *,
    wII_ne_precomputed: np.ndarray = None,
    wIII_ne2_precomputed: np.ndarray = None,
):
    """
    Setup residual function for chemical equilibrium nonlinear system

    Source: Korg.jl/src/statmech.jl:272-343

    The system of equations:
    - x[0:92] = neutral fraction for each element
    - x[92] = electron density / n_total * 1e5 (scaled for numerical stability)

    Residuals:
    - F[0:92] = element conservation equations
    - F[92] = charge neutrality equation

    Parameters
    ----------
    temperature : float
        Temperature in K
    n_total : float
        Total number density in cm^-3
    absolute_abundances : np.ndarray
        Element abundances (92 elements)
    ionization_energies : Dict
        Ionization energies
    partition_funcs : Dict
        Partition functions
    log_equilibrium_constants : Dict, optional
        Molecular equilibrium constants

    Returns
    -------
    Callable
        Residual function for scipy.optimize.root
    """

    if wII_ne_precomputed is not None and wIII_ne2_precomputed is not None:
        wII_ne = np.asarray(wII_ne_precomputed, dtype=np.float64)
        wIII_ne2 = np.asarray(wIII_ne2_precomputed, dtype=np.float64)
        if wII_ne.shape != (MAX_ATOMIC_NUMBER,) or wIII_ne2.shape != (MAX_ATOMIC_NUMBER,):
            raise ValueError("Precomputed Saha weight arrays must have shape (92,).")
    else:
        # Precompute Saha weights with ne=1 (will scale later)
        wII_ne = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
        wIII_ne2 = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)

        for Z in range(1, MAX_ATOMIC_NUMBER + 1):
            if Z in ionization_energies:
                wII, wIII = saha_ion_weights(temperature, 1.0, Z, ionization_energies, partition_funcs)
                wII_ne[Z - 1] = wII
                wIII_ne2[Z - 1] = wIII

    # Preprocess molecules for faster evaluation (and Jacobian assembly)
    molecules = None
    molecules_neutral = None
    molecules_charged = None
    if log_equilibrium_constants is not None:
        molecules = []
        for mol_species, log_K_func in log_equilibrium_constants.items():
            try:
                atoms = tuple(mol_species.get_atoms())
                if len(atoms) == 0:
                    continue
                atom_indices = np.asarray([Z - 1 for Z in atoms], dtype=np.int64)
                uniq, counts = np.unique(atom_indices, return_counts=True)
                molecules.append(
                    {
                        "species": mol_species,
                        "logK": log_K_func,
                        "uniq": uniq,
                        "counts": counts.astype(np.float64),
                        "n_atoms": float(len(atoms)),
                        "charge": int(getattr(mol_species, "charge", 0)),
                        # Charged diatomic: first atom ionized, second neutral (matches existing code)
                        "idx1": int(atom_indices[0]) if len(atom_indices) >= 1 else None,
                        "idx2": int(atom_indices[1]) if len(atom_indices) >= 2 else None,
                    }
                )
            except Exception:
                continue

    if molecules:
        log_T = float(np.log(temperature))
        log_kT = float(np.log10(kboltz_cgs * temperature))
        molecules_with_constants = []
        for mol in molecules:
            try:
                logKp = float(mol["logK"](log_T))  # log10(K_p)
                log_nK = logKp - (mol["n_atoms"] - 1.0) * log_kT
                if not np.isfinite(log_nK):
                    continue
                mol = dict(mol)
                mol["log_nK"] = float(log_nK)
                molecules_with_constants.append(mol)
            except Exception:
                continue
        molecules = molecules_with_constants or None

    if molecules:
        molecules_charged = [mol for mol in molecules if mol["charge"] == 1]
        molecules_neutral = [mol for mol in molecules if mol["charge"] != 1]

    cache = {"x_f": None, "F": None, "common": None, "x_j": None, "J": None}

    def _compute_common(x):
        x = np.asarray(x, dtype=np.float64)
        sign_ne = _sign_no_zero(x[-1])
        ne = abs(x[-1]) * n_total * 1e-5
        ne = max(float(ne), 1e-300)

        sign_f = _sign_no_zero(x[:-1])
        neutral_fractions = np.abs(x[:-1])
        atom_number_densities = absolute_abundances * (n_total - ne)
        neutral_number_densities = atom_number_densities * neutral_fractions

        inv_ne = 1.0 / ne
        inv_ne2 = inv_ne * inv_ne
        wII = wII_ne * inv_ne
        wIII = wIII_ne2 * inv_ne2

        R_elem = atom_number_densities - (1.0 + wII + wIII) * neutral_number_densities
        R_charge = float(np.sum((wII + 2.0 * wIII) * neutral_number_densities) - ne)

        log_neutral = None
        active = None
        n_tot_minus_ne = None

        if molecules:
            log_neutral = np.log10(np.maximum(neutral_number_densities, 1e-100))
            active = neutral_number_densities > 1e-100
            n_tot_minus_ne = float(n_total - ne)
            if abs(n_tot_minus_ne) < 1e-300:
                n_tot_minus_ne = 1e-300 if n_tot_minus_ne >= 0 else -1e-300

            for mol in (molecules_charged or []):
                try:
                    log_nK = mol["log_nK"]
                    idx1 = mol["idx1"]
                    idx2 = mol["idx2"]
                    if idx1 is None or idx2 is None:
                        continue
                    wII_1 = wII[idx1]
                    if wII_1 <= 0.0:
                        continue
                    log_n_mol = log_neutral[idx1] + np.log10(wII_1) + log_neutral[idx2] - log_nK
                    n_mol = float(10.0 ** log_n_mol)
                    if not np.isfinite(n_mol) or n_mol == 0.0:
                        continue
                    R_elem[idx1] -= n_mol
                    R_elem[idx2] -= n_mol
                    R_charge += n_mol
                except Exception:
                    continue

            for mol in (molecules_neutral or []):
                try:
                    log_nK = mol["log_nK"]
                    idx = mol["uniq"]
                    stoich = mol["counts"]
                    if idx.size == 0:
                        continue
                    log_n_mol = float(np.dot(stoich, log_neutral[idx]) - log_nK)
                    n_mol = float(10.0 ** log_n_mol)
                    if not np.isfinite(n_mol) or n_mol == 0.0:
                        continue
                    R_elem[idx] -= stoich * n_mol
                except Exception:
                    continue

        denom_elem = np.maximum(atom_number_densities, 1e-100)
        denom_charge = ne * 1e-5

        F = np.zeros_like(x)
        F[:-1] = R_elem / denom_elem
        F[-1] = R_charge / denom_charge

        return {
            "x": x,
            "sign_f": sign_f,
            "sign_ne": sign_ne,
            "neutral_fractions": neutral_fractions,
            "atom_number_densities": atom_number_densities,
            "neutral_number_densities": neutral_number_densities,
            "wII": wII,
            "wIII": wIII,
            "inv_ne": inv_ne,
            "R_elem": R_elem,
            "R_charge": R_charge,
            "denom_elem": denom_elem,
            "denom_charge": denom_charge,
            "log_neutral": log_neutral,
            "active": active,
            "n_tot_minus_ne": n_tot_minus_ne,
            "F": F,
        }

    def residuals(x):
        x = np.asarray(x, dtype=np.float64)
        if cache["x_f"] is not None and np.array_equal(x, cache["x_f"]):
            return cache["F"]
        common = _compute_common(x)
        cache["x_f"] = common["x"].copy()
        cache["F"] = common["F"]
        cache["common"] = common
        return common["F"]

    def jacobian(x):
        x = np.asarray(x, dtype=np.float64)
        if cache["x_j"] is not None and np.array_equal(x, cache["x_j"]):
            return cache["J"]

        if cache["x_f"] is not None and np.array_equal(x, cache["x_f"]) and cache["common"] is not None:
            common = cache["common"]
        else:
            common = _compute_common(x)

        neutral_fractions = common["neutral_fractions"]
        atom_number_densities = common["atom_number_densities"]
        neutral_number_densities = common["neutral_number_densities"]
        wII = common["wII"]
        wIII = common["wIII"]
        inv_ne = common["inv_ne"]

        dR_df = np.zeros((MAX_ATOMIC_NUMBER, MAX_ATOMIC_NUMBER), dtype=np.float64)
        np.fill_diagonal(dR_df, -(1.0 + wII + wIII) * atom_number_densities)

        A = absolute_abundances
        dn_dne = -A * neutral_fractions
        dS_dne = -(wII + 2.0 * wIII) * inv_ne
        dR_dne = (-A) - ((1.0 + wII + wIII) * dn_dne + neutral_number_densities * dS_dne)

        Q = wII + 2.0 * wIII
        dR_charge_df = Q * atom_number_densities
        dQ_dne = -(wII + 4.0 * wIII) * inv_ne
        dR_charge_dne = float(np.sum(Q * dn_dne + neutral_number_densities * dQ_dne) - 1.0)

        if molecules and common["log_neutral"] is not None:
            log_neutral = common["log_neutral"]
            active = common["active"]
            n_tot_minus_ne = common["n_tot_minus_ne"]

            for mol in (molecules_charged or []):
                try:
                    log_nK = mol["log_nK"]
                    idx1 = mol["idx1"]
                    idx2 = mol["idx2"]
                    if idx1 is None or idx2 is None:
                        continue
                    wII_1 = wII[idx1]
                    if wII_1 <= 0.0:
                        continue
                    log_n_mol = log_neutral[idx1] + np.log10(wII_1) + log_neutral[idx2] - log_nK
                    n_mol = float(10.0 ** log_n_mol)
                    if not np.isfinite(n_mol) or n_mol == 0.0:
                        continue

                    f1 = max(float(neutral_fractions[idx1]), 1e-300)
                    f2 = max(float(neutral_fractions[idx2]), 1e-300)
                    dnmol_df1 = (n_mol / f1) if active[idx1] else 0.0
                    dnmol_df2 = (n_mol / f2) if active[idx2] else 0.0

                    dR_df[idx1, idx1] -= dnmol_df1
                    dR_df[idx1, idx2] -= dnmol_df2
                    dR_df[idx2, idx1] -= dnmol_df1
                    dR_df[idx2, idx2] -= dnmol_df2

                    dR_charge_df[idx1] += dnmol_df1
                    dR_charge_df[idx2] += dnmol_df2

                    active_count = float(active[idx1]) + float(active[idx2])
                    dnmol_dne = n_mol * (-inv_ne - active_count / n_tot_minus_ne)
                    dR_dne[idx1] -= dnmol_dne
                    dR_dne[idx2] -= dnmol_dne
                    dR_charge_dne += dnmol_dne
                except Exception:
                    continue

            for mol in (molecules_neutral or []):
                try:
                    log_nK = mol["log_nK"]
                    idx = mol["uniq"]
                    stoich = mol["counts"]
                    if idx.size == 0:
                        continue
                    log_n_mol = float(np.dot(stoich, log_neutral[idx]) - log_nK)
                    n_mol = float(10.0 ** log_n_mol)
                    if not np.isfinite(n_mol) or n_mol == 0.0:
                        continue

                    active_idx = active[idx]
                    if np.any(active_idx):
                        f_u = np.maximum(neutral_fractions[idx], 1e-300)
                        dnmol_df = np.zeros_like(stoich)
                        dnmol_df[active_idx] = stoich[active_idx] * n_mol / f_u[active_idx]
                        dR_df[np.ix_(idx, idx)] -= stoich[:, None] * dnmol_df[None, :]

                    n_active = float(np.sum(stoich[active_idx])) if np.any(active_idx) else 0.0
                    if n_active != 0.0:
                        dnmol_dne = -n_active * n_mol / n_tot_minus_ne
                        dR_dne[idx] -= stoich * dnmol_dne
                except Exception:
                    continue

        denom_elem = common["denom_elem"]
        denom_charge = common["denom_charge"]
        R_elem = common["R_elem"]
        R_charge = common["R_charge"]
        sign_f = common["sign_f"]
        sign_ne = common["sign_ne"]

        J = np.zeros((MAX_ATOMIC_NUMBER + 1, MAX_ATOMIC_NUMBER + 1), dtype=np.float64)
        J[:MAX_ATOMIC_NUMBER, :MAX_ATOMIC_NUMBER] = (dR_df / denom_elem[:, None]) * sign_f[None, :]

        use_denom = atom_number_densities > 1e-100
        dF_dne = dR_dne / denom_elem
        if np.any(use_denom):
            dF_dne = dF_dne + (R_elem * A / (denom_elem * denom_elem)) * use_denom

        dne_dx = sign_ne * n_total * 1e-5
        J[:MAX_ATOMIC_NUMBER, -1] = dF_dne * dne_dx

        J[-1, :MAX_ATOMIC_NUMBER] = (dR_charge_df / denom_charge) * sign_f
        dF_charge_dne = (dR_charge_dne / denom_charge) - (R_charge * 1e-5) / (denom_charge * denom_charge)
        J[-1, -1] = dF_charge_dne * dne_dx

        cache["x_j"] = common["x"].copy()
        cache["J"] = J
        return J

    residuals.jacobian = jacobian
    residuals.molecules = molecules
    return residuals


def chemical_equilibrium(temp: float, nt: float, model_atm_ne: float,
                        absolute_abundances: Dict[int, float],
                        ionization_energies: Dict[int, Tuple],
                        partition_funcs: Dict[Species, Callable],
                        log_equilibrium_constants: Dict = None,
                        electron_number_density_warn_threshold: float = 0.1,
                        **kwargs) -> Tuple[float, Dict[Species, float]]:
    """
    Complete chemical equilibrium solver matching Korg.jl

    Source: Korg.jl/src/statmech.jl:120-165

    Solves the complete system of chemical equilibrium equations including:
    - Element conservation for all 92 elements
    - Charge neutrality constraint
    - Saha ionization equilibrium
    - Molecular equilibrium constants

    Parameters
    ----------
    temperature : float
        Temperature in K
    n_total : float
        Total number density in cm^-3
    model_atm_ne : float
        Model atmosphere electron density (initial guess) in cm^-3
    absolute_abundances : Dict[int, float] or array-like
        Element abundances N_X/N_total (92-element array or dict)
    ionization_energies : Dict
        Ionization energies in eV
    partition_funcs : Dict[Species, Callable]
        Partition function interpolators
    log_equilibrium_constants : Dict, optional
        Molecular equilibrium constants
    electron_number_density_warn_threshold : float
        Warning threshold for ne discrepancy
    stats_out : dict, optional (kwarg)
        If provided, populated with solver diagnostics (attempts, nfev, njev).
    initial_ne : float, optional (kwarg)
        Optional initial electron density guess for the solver (cm^-3). The
        model_atm_ne is still used for warning comparisons.
    warn_on_ne_discrepancy : bool, optional (kwarg)
        If False, suppress warnings about ne discrepancy.

    Returns
    -------
    Tuple[float, Dict[Species, float]]
        (electron_density, species_densities)
    """

    stats_out = kwargs.pop("stats_out", None)
    initial_ne = kwargs.pop("initial_ne", None)
    warn_on_ne_discrepancy = kwargs.pop("warn_on_ne_discrepancy", True)
    record_stats = isinstance(stats_out, dict)
    attempts = [] if record_stats else None
    fallback_used = False

    def _record_attempt(sol, attempt_idx, method_name):
        if not record_stats:
            return
        try:
            attempts.append(
                {
                    "attempt": int(attempt_idx),
                    "method": str(method_name),
                    "success": bool(getattr(sol, "success", False)),
                    "status": int(getattr(sol, "status", -1)) if hasattr(sol, "status") else None,
                    "message": str(getattr(sol, "message", "")),
                    "nfev": int(getattr(sol, "nfev", -1)) if hasattr(sol, "nfev") else None,
                    "njev": int(getattr(sol, "njev", -1)) if hasattr(sol, "njev") else None,
                }
            )
        except Exception:
            pass

    # Rename parameters to match internal variable names
    temperature = temp
    n_total = nt

    # Convert abundances to array (accept dict or array-like)
    if isinstance(absolute_abundances, dict):
        abs_abund_array = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
        for Z, abund in absolute_abundances.items():
            if 1 <= Z <= MAX_ATOMIC_NUMBER:
                abs_abund_array[Z-1] = abund
    else:
        abs_abund_array = np.asarray(absolute_abundances, dtype=np.float64)
        if abs_abund_array.shape[0] != MAX_ATOMIC_NUMBER:
            raise ValueError(
                f"absolute_abundances must have length {MAX_ATOMIC_NUMBER} (got {abs_abund_array.shape[0]})."
            )

    # Compute initial guess by neglecting molecules
    # Source: Korg.jl/src/statmech.jl:124-128
    neutral_fraction_guess = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
    wII_ne, wIII_ne2 = _get_cached_saha_weight_arrays(
        temperature, ionization_energies, partition_funcs
    )

    ne_guess = max(float(model_atm_ne), 1e-300)
    ne_initial = ne_guess if initial_ne is None else max(float(initial_ne), 1e-300)
    inv_ne_initial = 1.0 / ne_initial
    inv_ne_initial2 = inv_ne_initial * inv_ne_initial

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        if Z in ionization_energies:
            wII = wII_ne[Z - 1] * inv_ne_initial
            wIII = wIII_ne2[Z - 1] * inv_ne_initial2
            neutral_fraction_guess[Z - 1] = 1.0 / (1.0 + wII + wIII)

    # Initial guess: [neutral_fractions, ne/n_total*1e5]
    x0 = np.concatenate([neutral_fraction_guess, [ne_initial / n_total * 1e5]])

    # Setup residual function
    residuals_func = setup_chemical_equilibrium_residuals(
        temperature,
        n_total,
        abs_abund_array,
        ionization_energies,
        partition_funcs,
        log_equilibrium_constants,
        wII_ne_precomputed=wII_ne,
        wIII_ne2_precomputed=wIII_ne2,
    )

    # Solve nonlinear system
    # Source baseline: Korg.jl uses Newton with autodiff. Here we use SciPy root and
    # add robust fallbacks for cool-star outer layers where HYBRD can stagnate.
    try:
        jacobian_func = getattr(residuals_func, "jacobian", None)
        attempt_plan = [
            {
                "method": "hybr",
                "x0": x0.copy(),
                "jac": jacobian_func,
                "options": {"xtol": 1e-8, "maxfev": 1000},
            },
            {
                "method": "hybr",
                "x0": np.concatenate([neutral_fraction_guess, [1e-5]]),
                "jac": jacobian_func,
                "options": {"xtol": 1e-8, "maxfev": 1000},
            },
            {
                "method": "lm",
                "x0": x0.copy(),
                "jac": None,
                "options": {"xtol": 1e-10, "ftol": 1e-10, "maxiter": 5000},
            },
        ]

        sol = None
        for attempt_idx, attempt in enumerate(attempt_plan, start=1):
            method_name = attempt["method"]
            use_fallback = attempt_idx > 1
            if use_fallback:
                fallback_used = True

            sol = root(
                residuals_func,
                attempt["x0"],
                method=method_name,
                jac=attempt["jac"],
                options=attempt["options"],
            )
            _record_attempt(sol, attempt_idx, method_name)
            if sol.success:
                break

        if sol is None or not sol.success:
            if record_stats:
                stats_out.clear()
                stats_out.update(
                    {
                        "method": "hybr",
                        "xtol": 1e-8,
                        "maxfev": 1000,
                        "fallback_used": fallback_used,
                        "attempt_count": len(attempts),
                        "attempts": attempts,
                    }
                )
            msg = "unknown failure" if sol is None else str(getattr(sol, "message", "unknown failure"))
            raise RuntimeError(f"Chemical equilibrium solver failed: {msg}")
    except Exception as e:
        if record_stats and not stats_out:
            stats_out.clear()
            stats_out.update(
                {
                    "method": "hybr",
                    "xtol": 1e-8,
                    "maxfev": 1000,
                    "fallback_used": fallback_used,
                    "attempt_count": len(attempts),
                    "attempts": attempts,
                }
            )
        raise RuntimeError(f"Chemical equilibrium solver failed: {e}")

    if record_stats:
        stats_out.clear()
        stats_out.update(
            {
                "method": "hybr",
                "xtol": 1e-8,
                "maxfev": 1000,
                "fallback_used": fallback_used,
                "attempt_count": len(attempts),
                "attempts": attempts,
                "final_success": bool(getattr(sol, "success", False)),
                "final_status": int(getattr(sol, "status", -1)) if hasattr(sol, "status") else None,
            }
        )

    # Extract solution
    neutral_fractions = np.abs(sol.x[:-1])
    ne = abs(sol.x[-1]) * n_total * 1e-5

    # Check convergence warning
    if (warn_on_ne_discrepancy and (ne / n_total > 1e-4) and
        (abs((ne - model_atm_ne) / model_atm_ne) > electron_number_density_warn_threshold)):
        warnings.warn(
            f"Electron number density differs from model atmosphere by "
            f"{abs((ne - model_atm_ne) / model_atm_ne)*100:.1f}% "
            f"(calculated ne = {ne:.3e}, model atmosphere ne = {model_atm_ne:.3e})"
        )

    # Build species densities dict
    # Source: Korg.jl/src/statmech.jl:141-162
    species_densities: Dict[Species, float] = {}

    n0 = (n_total - ne) * abs_abund_array * neutral_fractions
    inv_ne = 1.0 / max(float(ne), 1e-300)
    wII_sol = wII_ne * inv_ne
    wIII_sol = wIII_ne2 * (inv_ne * inv_ne)

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        n_neutral = float(n0[Z - 1])
        species_densities[Species.from_atomic_number(Z, 0)] = n_neutral
        if Z in ionization_energies:
            species_densities[Species.from_atomic_number(Z, 1)] = float(wII_sol[Z - 1] * n_neutral)
            species_densities[Species.from_atomic_number(Z, 2)] = float(wIII_sol[Z - 1] * n_neutral)

    # Molecular species (reuse precomputed equilibrium constants from the residual function)
    if log_equilibrium_constants is not None:
        mols = getattr(residuals_func, "molecules", None)
        if mols:
            n1 = wII_sol * n0
            with np.errstate(divide="ignore", invalid="ignore"):
                log_n0 = np.log10(n0)
                log_n1 = np.log10(n1)

            for mol in mols:
                try:
                    log_nK = float(mol["log_nK"])
                    mol_species = mol["species"]
                    if mol["charge"] == 1:
                        idx1 = mol["idx1"]
                        idx2 = mol["idx2"]
                        if idx1 is None or idx2 is None:
                            continue
                        log_n_mol = float(log_n1[idx1] + log_n0[idx2] - log_nK)
                        if not np.isfinite(log_n_mol):
                            continue
                        species_densities[mol_species] = float(10.0 ** log_n_mol)
                    else:
                        idx = mol["uniq"]
                        stoich = mol["counts"]
                        if idx.size == 0:
                            continue
                        log_n_mol = float(np.dot(stoich, log_n0[idx]) - log_nK)
                        if not np.isfinite(log_n_mol):
                            continue
                        species_densities[mol_species] = float(10.0 ** log_n_mol)
                except Exception:
                    continue
        else:
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


__all__ = ['chemical_equilibrium', 'saha_ion_weights', 'translational_U']
