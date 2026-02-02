"""
Korg.jl Molecular Equilibrium Constants and Partition Functions
==============================================================

Loads Barklem & Collet 2016 diatomic constants and ExoMol polyatomic
partition functions, matching Korg.jl's default equilibrium constants.
"""

from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import h5py
import numpy as np
from scipy.interpolate import CubicSpline

from ..constants import kboltz_cgs, kboltz_eV, hplanck_cgs
from ..data import get_data_path
from ..data.isotopic_nuclear_spin_degeneracies import ISOTOPIC_NUCLEAR_SPIN_DEGENERACIES
from .species import Species, Formula, ATOMIC_MASSES
from .korg_exact_partition_functions import get_korg_exact_partition_functions
from ..lines.atomic_data import ISOTOPIC_ABUNDANCES


def _resolve_data_path(*parts: str) -> Path:
    if parts and parts[0] == "data":
        parts = parts[1:]
    return get_data_path(*parts)


@lru_cache(maxsize=1)
def _load_isotopic_nuclear_spin_degeneracies() -> Dict[int, Dict[int, int]]:
    """
    Load isotopic nuclear spin degeneracies (pure Python data).
    """
    return ISOTOPIC_NUCLEAR_SPIN_DEGENERACIES


def _get_most_abundant_isotope(atomic_number: int) -> int:
    isotopes = ISOTOPIC_ABUNDANCES.get(atomic_number, {})
    if not isotopes:
        return atomic_number
    return max(isotopes, key=isotopes.get)


def _total_nuclear_spin_degeneracy(spec: Species) -> float:
    """
    Compute total nuclear spin degeneracy for the most abundant isotopologue.
    """
    degeneracies = _load_isotopic_nuclear_spin_degeneracies()
    total = 1.0
    for Z in spec.get_atoms():
        most_abundant = _get_most_abundant_isotope(Z)
        g_ns = degeneracies.get(Z, {}).get(most_abundant, 1)
        total *= g_ns
    return total


@lru_cache(maxsize=1)
def load_barklem_collet_molecular_partition_functions() -> Dict[Species, Callable]:
    """
    Load Barklem & Collet 2016 diatomic molecular partition functions.
    """
    path = _resolve_data_path(
        "data", "barklem_collet_2016", "BarklemCollet2016-molecular_partition.dat"
    )
    temperatures = []
    data_pairs = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "T [K]" in line:
                temps = [float(tok) for tok in line.split() if _is_float(tok)]
                temperatures.extend(temps)
                continue
            if line.startswith("#"):
                continue
            parts = line.split()
            species_code = parts[0]
            if species_code.startswith("D_"):
                continue
            values = [float(x) for x in parts[1:]]
            data_pairs.append((species_code, values))

    temps_array = np.array(temperatures, dtype=float)
    temps_mask = np.isfinite(temps_array) & (temps_array > 0)
    log_temps = np.log(temps_array[temps_mask])
    partition_funcs: Dict[Species, Callable] = {}
    for species_code, vals in data_pairs:
        try:
            species = Species.from_string(species_code)
        except Exception:
            continue
        vals_array = np.array(vals, dtype=float)
        if len(vals_array) != len(temps_array):
            continue
        vals_array = vals_array[temps_mask]
        if len(vals_array) == 0:
            continue
        partition_funcs[species] = CubicSpline(log_temps, vals_array, extrapolate=True)

    return partition_funcs


@lru_cache(maxsize=1)
def load_exomol_partition_functions() -> Dict[Species, Callable]:
    """
    Load ExoMol polyatomic partition functions, matching Korg.jl normalization.
    """
    path = _resolve_data_path(
        "data", "polyatomic_partition_funcs", "polyatomic_partition_funcs.h5"
    )
    partition_funcs: Dict[Species, Callable] = {}
    with h5py.File(path, "r") as f:
        for group_name in f.keys():
            try:
                species = Species.from_string(group_name)
            except Exception:
                continue
            temps = np.array(f[group_name]["temp"], dtype=float)
            values = np.array(f[group_name]["partition_function"], dtype=float)
            g_ns = _total_nuclear_spin_degeneracy(species)
            values = values / max(g_ns, 1.0)
            mask = np.isfinite(temps) & (temps > 0) & np.isfinite(values)
            if np.count_nonzero(mask) < 2:
                continue
            partition_funcs[species] = CubicSpline(np.log(temps[mask]), values[mask], extrapolate=True)
    return partition_funcs


@lru_cache(maxsize=1)
def create_default_partition_functions_korg() -> Dict[Species, Callable]:
    """
    Create Korg-compatible partition functions (atomic + diatomic + polyatomic).
    """
    pf_system = get_korg_exact_partition_functions()
    partition_funcs = {spec: pf_system[spec] for spec in pf_system.partition_funcs}
    partition_funcs.update(load_barklem_collet_molecular_partition_functions())
    partition_funcs.update(load_exomol_partition_functions())
    return partition_funcs


def load_barklem_collet_logKs() -> Dict[Species, Callable]:
    """
    Load Barklem & Collet log10(K) splines from HDF5 (diatomic molecules).
    """
    path = _resolve_data_path("data", "barklem_collet_2016", "barklem_collet_ks.h5")
    with h5py.File(path, "r") as f:
        mols_raw = f["mols"][:]
        lnTs = np.array(f["lnTs"][:])
        logKs = np.array(f["logKs"][:])

    mols = []
    for m in mols_raw:
        if isinstance(m, (bytes, bytearray)):
            mols.append(m.decode())
        else:
            mols.append(str(m))

    # Apply C2 correction (Visser+ 2019) matching Korg.jl
    c2_idx = None
    for idx, mol in enumerate(mols):
        try:
            spec = Species.from_string(mol)
        except Exception:
            continue
        if spec.charge == 0 and spec.formula == Formula.from_string("C2"):
            c2_idx = idx
            break

    if c2_idx is not None:
        bc_c2_e0 = 6.371
        visser_c2_e0 = 6.24
        correction = (
            np.log10(np.e)
            / (kboltz_eV * np.exp(lnTs[:, c2_idx]))
            * (visser_c2_e0 - bc_c2_e0)
        )
        logKs[:, c2_idx] = logKs[:, c2_idx] + correction

    logKs_dict: Dict[Species, Callable] = {}
    for idx, mol in enumerate(mols):
        try:
            species = Species.from_string(mol)
        except Exception:
            continue
        lnT_col = lnTs[:, idx]
        logK_col = logKs[:, idx]
        mask = np.isfinite(lnT_col) & np.isfinite(logK_col)
        if not np.any(mask):
            continue
        logKs_dict[species] = CubicSpline(lnT_col[mask], logK_col[mask], extrapolate=True)

    return logKs_dict


def create_polyatomic_log_equilibrium_constants(
    partition_funcs: Dict[Species, Callable]
) -> Dict[Species, Callable]:
    """
    Compute polyatomic log10(K) splines using atomization energies (Korg.jl logic).
    """
    path = _resolve_data_path("data", "polyatomic_partition_funcs", "atomization_energies.csv")
    results: Dict[Species, Callable] = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            spec_str = row.get("spec")
            if not spec_str:
                continue
            try:
                spec = Species.from_string(spec_str)
            except Exception:
                continue
            try:
                d00_kj = float(row["energy"])
            except (KeyError, ValueError):
                continue
            d00_eV = d00_kj * 0.01036

            if spec not in partition_funcs:
                # Skip if molecule partition function is unavailable
                continue

            def logK(logT, spec=spec, d00_eV=d00_eV):
                Zs = spec.get_atoms()
                u_atoms = [
                    max(float(partition_funcs[Species.from_atomic_number(Z, 0)](logT)), 1e-300)
                    for Z in Zs
                ]
                u_mol = max(float(partition_funcs[spec](logT)), 1e-300)
                log_us_ratio = np.log10(np.prod(u_atoms) / u_mol)
                log_masses_ratio = (
                    sum(np.log10(ATOMIC_MASSES[Z - 1]) for Z in Zs)
                    - np.log10(spec.formula.mass)
                )
                T = np.exp(logT)
                log_trans_u = 1.5 * np.log10(2 * np.pi * kboltz_cgs * T / hplanck_cgs**2)
                log_nK = (
                    (len(Zs) - 1) * log_trans_u
                    + 1.5 * log_masses_ratio
                    + log_us_ratio
                    - d00_eV / (kboltz_eV * T * np.log(10))
                )
                return log_nK + (len(Zs) - 1) * np.log10(kboltz_cgs * T)

            results[spec] = logK
    return results


@lru_cache(maxsize=1)
def create_default_log_equilibrium_constants_korg() -> Dict[Species, Callable]:
    """
    Korg-compatible log10 equilibrium constants (partial pressure form).
    """
    logKs = load_barklem_collet_logKs()
    partition_funcs = create_default_partition_functions_korg()
    polyatomic = create_polyatomic_log_equilibrium_constants(partition_funcs)

    # Merge without losing type information
    equilibrium_constants: Dict[Species, Callable] = {}
    equilibrium_constants.update(logKs)
    equilibrium_constants.update(polyatomic)
    return equilibrium_constants


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


__all__ = [
    "create_default_partition_functions_korg",
    "create_default_log_equilibrium_constants_korg",
    "load_barklem_collet_molecular_partition_functions",
    "load_exomol_partition_functions",
    "load_barklem_collet_logKs",
]
