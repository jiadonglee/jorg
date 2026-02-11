"""
JAX-native chemical equilibrium solver with implicit differentiation.

This module replaces the SciPy/NumPy root solver with a fully JAX-friendly
implementation using JAXopt + implicit differentiation (custom_root).

Key design points:
- Unconstrained parameterization: f = sigmoid(y_f), ne = n_total * sigmoid(y_e)
- Saha weights in log space for numerical stability
- Optional molecular contributions in log10 space
- Custom VJP via implicit differentiation (no unrolled iteration backprop)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Sequence

import numpy as np

try:
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.scipy as jsp
    from jaxopt import Broyden
    from jaxopt.implicit_diff import custom_root
except Exception as exc:  # pragma: no cover - handled at runtime
    jax = None
    jnp = None
    jsp = None
    Broyden = None
    custom_root = None
    _JAX_IMPORT_ERROR = exc
else:
    _JAX_IMPORT_ERROR = None

from .species import Species, MAX_ATOMIC_NUMBER
from ..constants import kboltz_eV, kboltz_cgs, me_cgs, hplanck_cgs


def _require_jax():
    if jax is None:
        raise RuntimeError(f"JAX is required for chem_eq_jax: {_JAX_IMPORT_ERROR}")


@dataclass(frozen=True)
class IonizationData:
    chi1: np.ndarray
    chi2: np.ndarray
    mask1: np.ndarray
    mask2: np.ndarray


@dataclass(frozen=True)
class PartitionFunctionTable:
    logT_grid: np.ndarray
    logU_I_table: np.ndarray
    logU_II_table: np.ndarray
    logU_III_table: np.ndarray
    cubic_logT_grid: Optional[np.ndarray]
    U_I_coeffs: Optional[np.ndarray]
    U_II_coeffs: Optional[np.ndarray]
    U_III_coeffs: Optional[np.ndarray]
    cubic_mask_I: Optional[np.ndarray]
    cubic_mask_II: Optional[np.ndarray]
    cubic_mask_III: Optional[np.ndarray]
    mask_I: np.ndarray
    mask_II: np.ndarray
    mask_III: np.ndarray


@dataclass(frozen=True)
class MoleculeData:
    logT_grid: np.ndarray
    neutral_idx: Optional[np.ndarray]
    neutral_counts: Optional[np.ndarray]
    neutral_logKp_table: Optional[np.ndarray]
    neutral_n_atoms: Optional[np.ndarray]
    neutral_species: Optional[Sequence[Species]]
    charged_idx1: Optional[np.ndarray]
    charged_idx2: Optional[np.ndarray]
    charged_logKp_table: Optional[np.ndarray]
    charged_n_atoms: Optional[np.ndarray]
    charged_species: Optional[Sequence[Species]]


@dataclass(frozen=True)
class ChemEqData:
    ion: IonizationData
    pf: PartitionFunctionTable
    mol: Optional[MoleculeData]


_LOG_MIN = -700.0
# Keep exp(log_w) * n0 safely below float64 overflow (log ~ 709).
_LOG_MAX = 600.0
_LOG10_MIN = -300.0
# Keep 10**log10 * n0 below float64 overflow.
_LOG10_MAX = 250.0


def _logit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _infer_logT_grid(partition_funcs: Dict[Species, Callable]) -> np.ndarray:
    for func in partition_funcs.values():
        grid = getattr(func, "x", None)
        if grid is not None:
            grid = np.asarray(grid, dtype=float)
            if grid.ndim == 1 and grid.size >= 2:
                return grid
    raise ValueError("Could not infer logT grid from partition functions. Provide logT_grid.")


def _unwrap_cubic_spline(func: Callable):
    # Try direct CubicSpline-like object.
    if hasattr(func, "c") and hasattr(func, "x"):
        return func
    # Try to unwrap KorgExactPartitionFunctions lambda.
    closure = getattr(func, "__closure__", None)
    freevars = getattr(func, "__code__", None)
    if closure and freevars:
        names = func.__code__.co_freevars
        cells = {name: cell.cell_contents for name, cell in zip(names, closure)}
        self_obj = cells.get("self")
        species = cells.get("species")
        if self_obj is not None and species is not None:
            pf = getattr(self_obj, "partition_funcs", None)
            if pf is not None and species in pf:
                spline = pf[species]
                if hasattr(spline, "c") and hasattr(spline, "x"):
                    return spline
    return None


def _infer_cubic_logT_grid(partition_funcs: Dict[Species, Callable]) -> Optional[np.ndarray]:
    for func in partition_funcs.values():
        spline = _unwrap_cubic_spline(func)
        if spline is not None:
            grid = np.asarray(spline.x, dtype=np.float64)
            if grid.ndim == 1 and grid.size >= 2:
                return grid
    return None


def prepare_ionization_arrays(
    ionization_energies: Dict[int, Tuple[float, float, float]]
) -> IonizationData:
    chi1 = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
    chi2 = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
    mask1 = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)
    mask2 = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)

    for Z, energies in ionization_energies.items():
        if not (1 <= Z <= MAX_ATOMIC_NUMBER):
            continue
        chi_I, chi_II, _ = energies
        if np.isfinite(chi_I) and chi_I > 0:
            chi1[Z - 1] = float(chi_I)
            mask1[Z - 1] = True
        if Z > 1 and np.isfinite(chi_II) and chi_II > 0:
            chi2[Z - 1] = float(chi_II)
            mask2[Z - 1] = True

    return IonizationData(chi1=chi1, chi2=chi2, mask1=mask1, mask2=mask2)


def prepare_partition_function_tables(
    partition_funcs: Dict[Species, Callable],
    *,
    logT_grid: Optional[np.ndarray] = None,
    oversample: int = 10,
) -> PartitionFunctionTable:
    if hasattr(partition_funcs, "partition_funcs"):
        partition_funcs = partition_funcs.partition_funcs

    cubic_logT_grid = _infer_cubic_logT_grid(partition_funcs)

    if logT_grid is None:
        if cubic_logT_grid is not None:
            logT_grid = cubic_logT_grid
        else:
            logT_grid = _infer_logT_grid(partition_funcs)
    logT_grid = np.asarray(logT_grid, dtype=np.float64)
    if cubic_logT_grid is None and oversample and oversample > 1:
        n = logT_grid.size
        logT_grid = np.linspace(logT_grid[0], logT_grid[-1], (n - 1) * oversample + 1)
    # Partition functions are tabulated on this grid and later interpolated in JAX.

    n_T = logT_grid.size
    logU_I = np.zeros((MAX_ATOMIC_NUMBER, n_T), dtype=np.float64)
    logU_II = np.zeros((MAX_ATOMIC_NUMBER, n_T), dtype=np.float64)
    logU_III = np.zeros((MAX_ATOMIC_NUMBER, n_T), dtype=np.float64)
    mask_I = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)
    mask_II = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)
    mask_III = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)

    cubic_mask_I = None
    cubic_mask_II = None
    cubic_mask_III = None
    U_I_coeffs = None
    U_II_coeffs = None
    U_III_coeffs = None

    if cubic_logT_grid is not None:
        n_c = cubic_logT_grid.size
        U_I_coeffs = np.zeros((4, MAX_ATOMIC_NUMBER, n_c - 1), dtype=np.float64)
        U_II_coeffs = np.zeros((4, MAX_ATOMIC_NUMBER, n_c - 1), dtype=np.float64)
        U_III_coeffs = np.zeros((4, MAX_ATOMIC_NUMBER, n_c - 1), dtype=np.float64)
        cubic_mask_I = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)
        cubic_mask_II = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)
        cubic_mask_III = np.zeros(MAX_ATOMIC_NUMBER, dtype=bool)

    for Z in range(1, MAX_ATOMIC_NUMBER + 1):
        for stage, logU_table, mask in (
            (0, logU_I, mask_I),
            (1, logU_II, mask_II),
            (2, logU_III, mask_III),
        ):
            species = Species.from_atomic_number(Z, stage)
            func = partition_funcs.get(species)
            if func is None:
                continue
            if cubic_logT_grid is not None:
                spline = _unwrap_cubic_spline(func)
                if spline is not None:
                    grid = np.asarray(spline.x, dtype=np.float64)
                    if grid.shape == cubic_logT_grid.shape and np.allclose(grid, cubic_logT_grid):
                        coeffs = np.asarray(spline.c, dtype=np.float64)
                        if coeffs.shape[0] == 4 and coeffs.shape[1] == grid.size - 1:
                            if stage == 0:
                                U_I_coeffs[:, Z - 1, :] = coeffs
                                cubic_mask_I[Z - 1] = True
                            elif stage == 1:
                                U_II_coeffs[:, Z - 1, :] = coeffs
                                cubic_mask_II[Z - 1] = True
                            else:
                                U_III_coeffs[:, Z - 1, :] = coeffs
                                cubic_mask_III[Z - 1] = True
            try:
                vals = np.asarray(func(logT_grid), dtype=np.float64)
            except Exception:
                try:
                    vals = np.array([float(func(logT)) for logT in logT_grid], dtype=np.float64)
                except Exception:
                    continue
            vals = np.where(np.isfinite(vals), vals, 1.0)
            vals = np.clip(vals, 1e-300, None)
            logU_table[Z - 1, :] = np.log(vals)
            mask[Z - 1] = True

    return PartitionFunctionTable(
        logT_grid=logT_grid,
        logU_I_table=logU_I,
        logU_II_table=logU_II,
        logU_III_table=logU_III,
        cubic_logT_grid=cubic_logT_grid,
        U_I_coeffs=U_I_coeffs,
        U_II_coeffs=U_II_coeffs,
        U_III_coeffs=U_III_coeffs,
        cubic_mask_I=cubic_mask_I,
        cubic_mask_II=cubic_mask_II,
        cubic_mask_III=cubic_mask_III,
        mask_I=mask_I,
        mask_II=mask_II,
        mask_III=mask_III,
    )


def prepare_molecule_data(
    log_equilibrium_constants: Optional[Dict],
    logT_grid: np.ndarray,
) -> Optional[MoleculeData]:
    if log_equilibrium_constants is None:
        return None

    neutral_species = []
    neutral_idx = []
    neutral_counts = []
    neutral_logKp_table = []
    neutral_n_atoms = []

    charged_species = []
    charged_idx1 = []
    charged_idx2 = []
    charged_logKp_table = []
    charged_n_atoms = []

    for mol_species, log_K_func in log_equilibrium_constants.items():
        try:
            atoms = tuple(mol_species.get_atoms())
            if len(atoms) == 0:
                continue
            logKp_vals = np.array([float(log_K_func(logT)) for logT in logT_grid], dtype=np.float64)
            if not np.all(np.isfinite(logKp_vals)):
                continue
            atom_indices = np.asarray([Z - 1 for Z in atoms], dtype=np.int64)
            charge = int(getattr(mol_species, "charge", 0))
            if charge == 1 and len(atom_indices) >= 2:
                charged_species.append(mol_species)
                charged_idx1.append(int(atom_indices[0]))
                charged_idx2.append(int(atom_indices[1]))
                charged_logKp_table.append(logKp_vals)
                charged_n_atoms.append(float(len(atoms)))
            else:
                uniq, counts = np.unique(atom_indices, return_counts=True)
                neutral_species.append(mol_species)
                neutral_idx.append(uniq.astype(np.int32))
                neutral_counts.append(counts.astype(np.float64))
                neutral_logKp_table.append(logKp_vals)
                neutral_n_atoms.append(float(len(atoms)))
        except Exception:
            continue

    if not neutral_species and not charged_species:
        return None

    def _pad_ragged(idx_list, counts_list):
        max_len = max((arr.size for arr in idx_list), default=0)
        if max_len == 0:
            return None, None
        n = len(idx_list)
        idx = -np.ones((n, max_len), dtype=np.int32)
        counts = np.zeros((n, max_len), dtype=np.float64)
        for i, (u, c) in enumerate(zip(idx_list, counts_list)):
            m = int(u.size)
            idx[i, :m] = u
            counts[i, :m] = c
        return idx, counts

    neutral_idx_arr, neutral_counts_arr = _pad_ragged(neutral_idx, neutral_counts)
    neutral_logKp = np.stack(neutral_logKp_table, axis=0) if neutral_logKp_table else None
    neutral_n_atoms_arr = np.asarray(neutral_n_atoms, dtype=np.float64) if neutral_n_atoms else None

    charged_idx1_arr = np.asarray(charged_idx1, dtype=np.int32) if charged_idx1 else None
    charged_idx2_arr = np.asarray(charged_idx2, dtype=np.int32) if charged_idx2 else None
    charged_logKp = np.stack(charged_logKp_table, axis=0) if charged_logKp_table else None
    charged_n_atoms_arr = np.asarray(charged_n_atoms, dtype=np.float64) if charged_n_atoms else None

    return MoleculeData(
        logT_grid=np.asarray(logT_grid, dtype=np.float64),
        neutral_idx=neutral_idx_arr,
        neutral_counts=neutral_counts_arr,
        neutral_logKp_table=neutral_logKp,
        neutral_n_atoms=neutral_n_atoms_arr,
        neutral_species=tuple(neutral_species) if neutral_species else None,
        charged_idx1=charged_idx1_arr,
        charged_idx2=charged_idx2_arr,
        charged_logKp_table=charged_logKp,
        charged_n_atoms=charged_n_atoms_arr,
        charged_species=tuple(charged_species) if charged_species else None,
    )


def prepare_chem_eq_data(
    ionization_energies: Dict[int, Tuple[float, float, float]],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Optional[Dict] = None,
    *,
    logT_grid: Optional[np.ndarray] = None,
    oversample: int = 10,
) -> ChemEqData:
    ion = prepare_ionization_arrays(ionization_energies)
    pf = prepare_partition_function_tables(partition_funcs, logT_grid=logT_grid, oversample=oversample)
    mol = prepare_molecule_data(log_equilibrium_constants, pf.logT_grid)
    return ChemEqData(ion=ion, pf=pf, mol=mol)


def _interp_table_np(x: float, grid: np.ndarray, table: np.ndarray) -> np.ndarray:
    x = np.clip(x, grid[0], grid[-1])
    idx = np.searchsorted(grid, x) - 1
    idx = np.clip(idx, 0, grid.size - 2)
    x0 = grid[idx]
    x1 = grid[idx + 1]
    t = (x - x0) / (x1 - x0)
    y0 = table[..., idx]
    y1 = table[..., idx + 1]
    return y0 + t * (y1 - y0)


def _eval_cubic_np(x: float, grid: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    x = np.clip(x, grid[0], grid[-1])
    idx = np.searchsorted(grid, x) - 1
    idx = np.clip(idx, 0, grid.size - 2)
    dx = x - grid[idx]
    c0 = coeffs[0, :, idx]
    c1 = coeffs[1, :, idx]
    c2 = coeffs[2, :, idx]
    c3 = coeffs[3, :, idx]
    return ((c0 * dx + c1) * dx + c2) * dx + c3


def _interp_table_jax(x, grid, table):
    x = jnp.clip(x, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, x, side="right") - 1
    idx = jnp.clip(idx, 0, grid.shape[0] - 2)
    x0 = grid[idx]
    x1 = grid[idx + 1]
    t = (x - x0) / (x1 - x0)
    y0 = jnp.take(table, idx, axis=-1)
    y1 = jnp.take(table, idx + 1, axis=-1)
    return y0 + t * (y1 - y0)


def _eval_cubic_jax(x, grid, coeffs):
    x = jnp.clip(x, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, x, side="right") - 1
    idx = jnp.clip(idx, 0, grid.shape[0] - 2)
    dx = x - grid[idx]
    c0 = jnp.take(coeffs[0], idx, axis=-1)
    c1 = jnp.take(coeffs[1], idx, axis=-1)
    c2 = jnp.take(coeffs[2], idx, axis=-1)
    c3 = jnp.take(coeffs[3], idx, axis=-1)
    return ((c0 * dx + c1) * dx + c2) * dx + c3


def make_optimality_fun(
    chem_data: ChemEqData,
    *,
    eps: float = 1e-100,
) -> Callable:
    _require_jax()

    chi1 = jnp.asarray(chem_data.ion.chi1)
    chi2 = jnp.asarray(chem_data.ion.chi2)
    mask1 = jnp.asarray(chem_data.ion.mask1)
    mask2 = jnp.asarray(chem_data.ion.mask2)

    logT_grid = jnp.asarray(chem_data.pf.logT_grid)
    logU_I_table = jnp.asarray(chem_data.pf.logU_I_table)
    logU_II_table = jnp.asarray(chem_data.pf.logU_II_table)
    logU_III_table = jnp.asarray(chem_data.pf.logU_III_table)

    cubic_logT_grid = None
    U_I_coeffs = None
    U_II_coeffs = None
    U_III_coeffs = None
    cubic_mask_I = None
    cubic_mask_II = None
    cubic_mask_III = None

    if chem_data.pf.cubic_logT_grid is not None and chem_data.pf.U_I_coeffs is not None:
        cubic_logT_grid = jnp.asarray(chem_data.pf.cubic_logT_grid)
        U_I_coeffs = jnp.asarray(chem_data.pf.U_I_coeffs)
        U_II_coeffs = jnp.asarray(chem_data.pf.U_II_coeffs)
        U_III_coeffs = jnp.asarray(chem_data.pf.U_III_coeffs)
        if chem_data.pf.cubic_mask_I is not None:
            cubic_mask_I = jnp.asarray(chem_data.pf.cubic_mask_I)
            cubic_mask_II = jnp.asarray(chem_data.pf.cubic_mask_II)
            cubic_mask_III = jnp.asarray(chem_data.pf.cubic_mask_III)

    cubic_logT_grid = None
    U_I_coeffs = None
    U_II_coeffs = None
    U_III_coeffs = None
    cubic_mask_I = None
    cubic_mask_II = None
    cubic_mask_III = None

    if chem_data.pf.cubic_logT_grid is not None and chem_data.pf.U_I_coeffs is not None:
        cubic_logT_grid = jnp.asarray(chem_data.pf.cubic_logT_grid)
        U_I_coeffs = jnp.asarray(chem_data.pf.U_I_coeffs)
        U_II_coeffs = jnp.asarray(chem_data.pf.U_II_coeffs)
        U_III_coeffs = jnp.asarray(chem_data.pf.U_III_coeffs)
        if chem_data.pf.cubic_mask_I is not None:
            cubic_mask_I = jnp.asarray(chem_data.pf.cubic_mask_I)
            cubic_mask_II = jnp.asarray(chem_data.pf.cubic_mask_II)
            cubic_mask_III = jnp.asarray(chem_data.pf.cubic_mask_III)

    mask_pf_I = jnp.asarray(chem_data.pf.mask_I)
    mask_pf_II = jnp.asarray(chem_data.pf.mask_II)
    mask_pf_III = jnp.asarray(chem_data.pf.mask_III)

    # Only allow ionization if both energy and PFs are present.
    mask1 = mask1 & mask_pf_I & mask_pf_II
    mask2 = mask2 & mask_pf_II & mask_pf_III

    has_mol = chem_data.mol is not None
    if has_mol:
        mol = chem_data.mol
        has_neutral = mol.neutral_idx is not None
        has_charged = mol.charged_idx1 is not None
        if has_neutral:
            neutral_idx = jnp.asarray(mol.neutral_idx)
            neutral_counts = jnp.asarray(mol.neutral_counts)
            neutral_logKp_table = jnp.asarray(mol.neutral_logKp_table)
            neutral_n_atoms = jnp.asarray(mol.neutral_n_atoms)
        if has_charged:
            charged_idx1 = jnp.asarray(mol.charged_idx1)
            charged_idx2 = jnp.asarray(mol.charged_idx2)
            charged_logKp_table = jnp.asarray(mol.charged_logKp_table)
            charged_n_atoms = jnp.asarray(mol.charged_n_atoms)

    log2 = jnp.log(2.0)

    def _logU_arrays(logT):
        logU_I_lin = _interp_table_jax(logT, logT_grid, logU_I_table)
        logU_II_lin = _interp_table_jax(logT, logT_grid, logU_II_table)
        logU_III_lin = _interp_table_jax(logT, logT_grid, logU_III_table)

        if cubic_logT_grid is not None and U_I_coeffs is not None:
            U_I = _eval_cubic_jax(logT, cubic_logT_grid, U_I_coeffs)
            U_II = _eval_cubic_jax(logT, cubic_logT_grid, U_II_coeffs)
            U_III = _eval_cubic_jax(logT, cubic_logT_grid, U_III_coeffs)
            logU_I_c = jnp.log(jnp.clip(U_I, 1e-300, None))
            logU_II_c = jnp.log(jnp.clip(U_II, 1e-300, None))
            logU_III_c = jnp.log(jnp.clip(U_III, 1e-300, None))
            if cubic_mask_I is not None:
                logU_I = jnp.where(cubic_mask_I, logU_I_c, logU_I_lin)
                logU_II = jnp.where(cubic_mask_II, logU_II_c, logU_II_lin)
                logU_III = jnp.where(cubic_mask_III, logU_III_c, logU_III_lin)
            else:
                logU_I = logU_I_c
                logU_II = logU_II_c
                logU_III = logU_III_c
        else:
            logU_I = logU_I_lin
            logU_II = logU_II_lin
            logU_III = logU_III_lin

        return (logU_I, logU_II, logU_III)

    def _compute_wII_wIII(T, ne):
        logT = jnp.log(T)
        logU_I, logU_II, logU_III = _logU_arrays(logT)
        log_trans_U = 1.5 * jnp.log(2.0 * jnp.pi * me_cgs * kboltz_cgs * T / (hplanck_cgs ** 2))
        inv_kT = 1.0 / (kboltz_eV * T)

        logC1 = log2 + logU_II - logU_I + log_trans_U - chi1 * inv_kT
        logC2 = logC1 + log2 + logU_III - logU_II + log_trans_U - chi2 * inv_kT
        logC1 = jnp.where(mask1, logC1, -jnp.inf)
        logC2 = jnp.where(mask2, logC2, -jnp.inf)

        log_ne = jnp.log(ne)
        # Soft clamp to prevent exp overflow while keeping gradients smooth.
        log_wII = jnp.clip(logC1 - log_ne, _LOG_MIN, _LOG_MAX)
        log_wIII = jnp.clip(logC2 - 2.0 * log_ne, _LOG_MIN, _LOG_MAX)

        wII = jnp.where(mask1, jnp.exp(log_wII), 0.0)
        wIII = jnp.where(mask2, jnp.exp(log_wIII), 0.0)
        return wII, wIII

    def optimality_fun(y, T, n_total, abund):
        y = jnp.asarray(y)
        T = jnp.asarray(T)
        n_total = jnp.asarray(n_total)
        abund = jnp.asarray(abund)

        f = jax.nn.sigmoid(y[:-1])
        ne = n_total * jax.nn.sigmoid(y[-1])

        n_nuc = abund * (n_total - ne)
        n0 = n_nuc * f

        wII, wIII = _compute_wII_wIII(T, ne)

        R_elem = n_nuc - (1.0 + wII + wIII) * n0
        R_charge = jnp.sum((wII + 2.0 * wIII) * n0) - ne

        if has_mol:
            log_n0 = jnp.log10(jnp.clip(n0, 1e-300, None))
            log_kT = jnp.log10(kboltz_cgs * T)

            if has_charged:
                logKp = _interp_table_jax(jnp.log(T), logT_grid, charged_logKp_table)
                log_nK = logKp - (charged_n_atoms - 1.0) * log_kT
                log_wII = jnp.log10(jnp.clip(wII, 1e-300, None))
                log_n_mol = log_n0[charged_idx1] + log_wII[charged_idx1] + log_n0[charged_idx2] - log_nK
                # Soft clamp in log10 space to avoid 10** overflow.
                log_n_mol = jnp.clip(log_n_mol, _LOG10_MIN, _LOG10_MAX)
                n_mol = jnp.power(10.0, log_n_mol)
                R_elem = R_elem.at[charged_idx1].add(-n_mol)
                R_elem = R_elem.at[charged_idx2].add(-n_mol)
                R_charge = R_charge + jnp.sum(n_mol)

            if has_neutral:
                logKp = _interp_table_jax(jnp.log(T), logT_grid, neutral_logKp_table)
                log_nK = logKp - (neutral_n_atoms - 1.0) * log_kT
                idx_clip = jnp.where(neutral_idx >= 0, neutral_idx, 0)
                mask = neutral_idx >= 0
                log_sel = log_n0[idx_clip]
                log_sel = jnp.where(mask, log_sel, 0.0)
                log_n_mol = jnp.sum(neutral_counts * log_sel, axis=1) - log_nK
                # Soft clamp in log10 space to avoid 10** overflow.
                log_n_mol = jnp.clip(log_n_mol, _LOG10_MIN, _LOG10_MAX)
                n_mol = jnp.power(10.0, log_n_mol)

                contrib = -neutral_counts * n_mol[:, None]
                contrib = jnp.where(mask, contrib, 0.0)
                idx_flat = idx_clip.reshape(-1)
                contrib_flat = contrib.reshape(-1)
                R_elem = R_elem.at[idx_flat].add(contrib_flat)

        denom_elem = jnp.maximum(n_nuc, eps)
        denom_charge = jnp.maximum(ne, eps)

        F_elem = R_elem / denom_elem
        F_charge = R_charge / denom_charge
        return jnp.concatenate((F_elem, jnp.array([F_charge], dtype=jnp.float64)))

    return optimality_fun


def make_solver(
    optimality_fun: Callable,
    *,
    method: str = "levenberg_marquardt",
    maxiter: int = 300,
    tol: float = 1e-8,
    cg_tol: float = 1e-10,
    cg_maxiter: int = 200,
    jit: bool = True,
) -> Callable:
    _require_jax()

    def tangent_solve(matvec, b):
        # Solve (dF/dy)^T v = b using CG with matrix-free matvec.
        sol, _ = jsp.sparse.linalg.cg(matvec, b, tol=cg_tol, maxiter=cg_maxiter)
        return sol

    method = method.lower()
    if method in ("broyden", "broyden1"):
        solver = Broyden(
            optimality_fun,
            maxiter=maxiter,
            tol=tol,
            implicit_diff=False,
            jit=jit,
        )
    elif method in ("levenberg_marquardt", "lm"):
        from jaxopt import LevenbergMarquardt

        solver = LevenbergMarquardt(
            optimality_fun,
            maxiter=maxiter,
            tol=tol,
            solver="lu",
            materialize_jac=True,
            implicit_diff=False,
            jit=jit,
        )
    else:
        raise ValueError(f"Unknown solver method: {method}")

    def _solve(y0, T, n_total, abund):
        out = solver.run(y0, T, n_total, abund)
        return out.params

    return custom_root(optimality_fun, solve=tangent_solve)(_solve)


def initial_guess_y(
    temperature: float,
    n_total: float,
    model_ne: float,
    chem_data: ChemEqData,
    *,
    ne_floor: float = 1e-30,
) -> np.ndarray:
    logT = float(np.log(temperature))
    logU_I = _interp_table_np(logT, chem_data.pf.logT_grid, chem_data.pf.logU_I_table)
    logU_II = _interp_table_np(logT, chem_data.pf.logT_grid, chem_data.pf.logU_II_table)
    logU_III = _interp_table_np(logT, chem_data.pf.logT_grid, chem_data.pf.logU_III_table)

    if chem_data.pf.cubic_logT_grid is not None and chem_data.pf.U_I_coeffs is not None:
        c_grid = chem_data.pf.cubic_logT_grid
        U_I = _eval_cubic_np(logT, c_grid, chem_data.pf.U_I_coeffs)
        U_II = _eval_cubic_np(logT, c_grid, chem_data.pf.U_II_coeffs)
        U_III = _eval_cubic_np(logT, c_grid, chem_data.pf.U_III_coeffs)
        logU_I_c = np.log(np.clip(U_I, 1e-300, None))
        logU_II_c = np.log(np.clip(U_II, 1e-300, None))
        logU_III_c = np.log(np.clip(U_III, 1e-300, None))
        if chem_data.pf.cubic_mask_I is not None:
            logU_I = np.where(chem_data.pf.cubic_mask_I, logU_I_c, logU_I)
            logU_II = np.where(chem_data.pf.cubic_mask_II, logU_II_c, logU_II)
            logU_III = np.where(chem_data.pf.cubic_mask_III, logU_III_c, logU_III)
        else:
            logU_I = logU_I_c
            logU_II = logU_II_c
            logU_III = logU_III_c

    chi1 = chem_data.ion.chi1
    chi2 = chem_data.ion.chi2
    mask1 = chem_data.ion.mask1 & chem_data.pf.mask_I & chem_data.pf.mask_II
    mask2 = chem_data.ion.mask2 & chem_data.pf.mask_II & chem_data.pf.mask_III

    log_trans_U = 1.5 * np.log(2.0 * np.pi * me_cgs * kboltz_cgs * temperature / (hplanck_cgs ** 2))
    inv_kT = 1.0 / (kboltz_eV * temperature)

    logC1 = np.log(2.0) + logU_II - logU_I + log_trans_U - chi1 * inv_kT
    logC2 = logC1 + np.log(2.0) + logU_III - logU_II + log_trans_U - chi2 * inv_kT
    logC1 = np.where(mask1, logC1, -np.inf)
    logC2 = np.where(mask2, logC2, -np.inf)

    ne_guess = max(float(model_ne), ne_floor)
    log_ne = np.log(ne_guess)
    log_wII = np.clip(logC1 - log_ne, _LOG_MIN, _LOG_MAX)
    log_wIII = np.clip(logC2 - 2.0 * log_ne, _LOG_MIN, _LOG_MAX)
    wII = np.where(mask1, np.exp(log_wII), 0.0)
    wIII = np.where(mask2, np.exp(log_wIII), 0.0)

    f_guess = np.where(mask1, 1.0 / (1.0 + wII + wIII), 1.0 - 1e-6)

    y_f = _logit(f_guess)
    ne_frac = ne_guess / n_total
    y_e = _logit(np.array(ne_frac, dtype=np.float64))
    return np.concatenate([y_f, [y_e]]).astype(np.float64)


def unpack_solution(y: jnp.ndarray, n_total: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    f = jax.nn.sigmoid(y[:-1])
    ne = n_total * jax.nn.sigmoid(y[-1])
    return f, ne


def compute_species_densities_arrays(
    y: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abund: jnp.ndarray,
    chem_data: ChemEqData,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Optional[jnp.ndarray], Optional[jnp.ndarray]]:

    logT_grid = jnp.asarray(chem_data.pf.logT_grid)
    logU_I_table = jnp.asarray(chem_data.pf.logU_I_table)
    logU_II_table = jnp.asarray(chem_data.pf.logU_II_table)
    logU_III_table = jnp.asarray(chem_data.pf.logU_III_table)

    chi1 = jnp.asarray(chem_data.ion.chi1)
    chi2 = jnp.asarray(chem_data.ion.chi2)
    mask1 = jnp.asarray(chem_data.ion.mask1) & jnp.asarray(chem_data.pf.mask_I) & jnp.asarray(chem_data.pf.mask_II)
    mask2 = jnp.asarray(chem_data.ion.mask2) & jnp.asarray(chem_data.pf.mask_II) & jnp.asarray(chem_data.pf.mask_III)

    def _logU_arrays(logT):
        logU_I_lin = _interp_table_jax(logT, logT_grid, logU_I_table)
        logU_II_lin = _interp_table_jax(logT, logT_grid, logU_II_table)
        logU_III_lin = _interp_table_jax(logT, logT_grid, logU_III_table)

        if cubic_logT_grid is not None and U_I_coeffs is not None:
            U_I = _eval_cubic_jax(logT, cubic_logT_grid, U_I_coeffs)
            U_II = _eval_cubic_jax(logT, cubic_logT_grid, U_II_coeffs)
            U_III = _eval_cubic_jax(logT, cubic_logT_grid, U_III_coeffs)
            logU_I_c = jnp.log(jnp.clip(U_I, 1e-300, None))
            logU_II_c = jnp.log(jnp.clip(U_II, 1e-300, None))
            logU_III_c = jnp.log(jnp.clip(U_III, 1e-300, None))
            if cubic_mask_I is not None:
                logU_I = jnp.where(cubic_mask_I, logU_I_c, logU_I_lin)
                logU_II = jnp.where(cubic_mask_II, logU_II_c, logU_II_lin)
                logU_III = jnp.where(cubic_mask_III, logU_III_c, logU_III_lin)
            else:
                logU_I = logU_I_c
                logU_II = logU_II_c
                logU_III = logU_III_c
        else:
            logU_I = logU_I_lin
            logU_II = logU_II_lin
            logU_III = logU_III_lin

        return (logU_I, logU_II, logU_III)

    def _compute_wII_wIII(T, ne):
        logT = jnp.log(T)
        logU_I, logU_II, logU_III = _logU_arrays(logT)
        log_trans_U = 1.5 * jnp.log(2.0 * jnp.pi * me_cgs * kboltz_cgs * T / (hplanck_cgs ** 2))
        inv_kT = 1.0 / (kboltz_eV * T)
        logC1 = jnp.log(2.0) + logU_II - logU_I + log_trans_U - chi1 * inv_kT
        logC2 = logC1 + jnp.log(2.0) + logU_III - logU_II + log_trans_U - chi2 * inv_kT
        logC1 = jnp.where(mask1, logC1, -jnp.inf)
        logC2 = jnp.where(mask2, logC2, -jnp.inf)
        log_ne = jnp.log(ne)
        log_wII = jnp.clip(logC1 - log_ne, _LOG_MIN, _LOG_MAX)
        log_wIII = jnp.clip(logC2 - 2.0 * log_ne, _LOG_MIN, _LOG_MAX)
        wII = jnp.where(mask1, jnp.exp(log_wII), 0.0)
        wIII = jnp.where(mask2, jnp.exp(log_wIII), 0.0)
        return wII, wIII

    f, ne = unpack_solution(y, n_total)
    n_nuc = abund * (n_total - ne)
    n0 = n_nuc * f
    wII, wIII = _compute_wII_wIII(T, ne)
    n1 = wII * n0
    n2 = wIII * n0

    n_mol_neutral = None
    n_mol_charged = None
    if chem_data.mol is not None:
        mol = chem_data.mol
        log_n0 = jnp.log10(jnp.clip(n0, 1e-300, None))
        log_kT = jnp.log10(kboltz_cgs * T)

        if mol.neutral_idx is not None:
            neutral_idx = jnp.asarray(mol.neutral_idx)
            neutral_counts = jnp.asarray(mol.neutral_counts)
            neutral_logKp_table = jnp.asarray(mol.neutral_logKp_table)
            neutral_n_atoms = jnp.asarray(mol.neutral_n_atoms)
            logKp = _interp_table_jax(jnp.log(T), logT_grid, neutral_logKp_table)
            log_nK = logKp - (neutral_n_atoms - 1.0) * log_kT
            idx_clip = jnp.where(neutral_idx >= 0, neutral_idx, 0)
            mask = neutral_idx >= 0
            log_sel = log_n0[idx_clip]
            log_sel = jnp.where(mask, log_sel, 0.0)
            log_n_mol = jnp.sum(neutral_counts * log_sel, axis=1) - log_nK
            log_n_mol = jnp.clip(log_n_mol, _LOG10_MIN, _LOG10_MAX)
            n_mol_neutral = jnp.power(10.0, log_n_mol)

        if mol.charged_idx1 is not None:
            charged_idx1 = jnp.asarray(mol.charged_idx1)
            charged_idx2 = jnp.asarray(mol.charged_idx2)
            charged_logKp_table = jnp.asarray(mol.charged_logKp_table)
            charged_n_atoms = jnp.asarray(mol.charged_n_atoms)
            logKp = _interp_table_jax(jnp.log(T), logT_grid, charged_logKp_table)
            log_nK = logKp - (charged_n_atoms - 1.0) * log_kT
            log_wII = jnp.log10(jnp.clip(wII, 1e-300, None))
            log_n_mol = log_n0[charged_idx1] + log_wII[charged_idx1] + log_n0[charged_idx2] - log_nK
            log_n_mol = jnp.clip(log_n_mol, _LOG10_MIN, _LOG10_MAX)
            n_mol_charged = jnp.power(10.0, log_n_mol)

    return ne, n0, n1, n2, n_mol_neutral, n_mol_charged


def solve_equilibrium(
    y0: jnp.ndarray,
    T: jnp.ndarray,
    n_total: jnp.ndarray,
    abund: jnp.ndarray,
    chem_data: ChemEqData,
    *,
    method: str = "levenberg_marquardt",
    maxiter: int = 300,
    tol: float = 1e-8,
    cg_tol: float = 1e-10,
    cg_maxiter: int = 200,
    jit: bool = True,
) -> jnp.ndarray:
    optimality_fun = make_optimality_fun(chem_data)
    solve = make_solver(
        optimality_fun,
        method=method,
        maxiter=maxiter,
        tol=tol,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        jit=jit,
    )
    return solve(y0, T, n_total, abund)


def chemical_equilibrium_jax(
    temp: float,
    nt: float,
    model_atm_ne: float,
    absolute_abundances: Dict[int, float] | np.ndarray,
    ionization_energies: Dict[int, Tuple[float, float, float]],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Optional[Dict] = None,
    *,
    chem_data: Optional[ChemEqData] = None,
    method: str = "levenberg_marquardt",
    maxiter: int = 300,
    tol: float = 1e-8,
    cg_tol: float = 1e-10,
    cg_maxiter: int = 200,
    jit: bool = True,
    fallback: bool = True,
    return_species: bool = False,
    return_y: bool = False,
) -> Tuple[float, np.ndarray] | Tuple[float, np.ndarray, Dict[Species, float]] | Tuple[float, np.ndarray, np.ndarray]:
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

    if chem_data is None:
        chem_data = prepare_chem_eq_data(
            ionization_energies,
            partition_funcs,
            log_equilibrium_constants,
        )

    y0 = initial_guess_y(
        float(temp),
        float(nt),
        float(model_atm_ne),
        chem_data,
    )

    optimality_fun = make_optimality_fun(chem_data)
    solve = make_solver(
        optimality_fun,
        method=method,
        maxiter=maxiter,
        tol=tol,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        jit=jit,
    )

    y = solve(jnp.asarray(y0), jnp.asarray(temp), jnp.asarray(nt), jnp.asarray(abs_abund_array))
    y_np = np.asarray(jax.device_get(y), dtype=np.float64)

    if fallback:
        # Fallback with very small ne fraction if residuals remain large.
        F = optimality_fun(jnp.asarray(y_np), jnp.asarray(temp), jnp.asarray(nt), jnp.asarray(abs_abund_array))
        err = np.max(np.abs(np.asarray(jax.device_get(F))))
        if not np.isfinite(err) or err > 10.0 * float(tol):
            y0[-1] = _logit(np.array(1e-12, dtype=np.float64))
            y = solve(jnp.asarray(y0), jnp.asarray(temp), jnp.asarray(nt), jnp.asarray(abs_abund_array))
            y_np = np.asarray(jax.device_get(y), dtype=np.float64)

    f = 1.0 / (1.0 + np.exp(-y_np[:-1]))
    ne = float(nt) * (1.0 / (1.0 + np.exp(-y_np[-1])))

    if return_y:
        return ne, f, y_np

    if return_species:
        # Reconstruct species densities outside JIT for convenience.
        T_j = jnp.asarray(temp)
        nt_j = jnp.asarray(nt)
        abund_j = jnp.asarray(abs_abund_array)
        _, n0, n1, n2, n_mol_neutral, n_mol_charged = compute_species_densities_arrays(
            jnp.asarray(y_np), T_j, nt_j, abund_j, chem_data
        )
        n0 = np.asarray(jax.device_get(n0), dtype=np.float64)
        n1 = np.asarray(jax.device_get(n1), dtype=np.float64)
        n2 = np.asarray(jax.device_get(n2), dtype=np.float64)

        species_densities: Dict[Species, float] = {}
        for Z in range(1, MAX_ATOMIC_NUMBER + 1):
            species_densities[Species.from_atomic_number(Z, 0)] = float(n0[Z - 1])
            species_densities[Species.from_atomic_number(Z, 1)] = float(n1[Z - 1])
            species_densities[Species.from_atomic_number(Z, 2)] = float(n2[Z - 1])

        if chem_data.mol is not None:
            mol = chem_data.mol
            if n_mol_neutral is not None and mol.neutral_species is not None:
                vals = np.asarray(jax.device_get(n_mol_neutral), dtype=np.float64)
                for spec, val in zip(mol.neutral_species, vals):
                    species_densities[spec] = float(val)
            if n_mol_charged is not None and mol.charged_species is not None:
                vals = np.asarray(jax.device_get(n_mol_charged), dtype=np.float64)
                for spec, val in zip(mol.charged_species, vals):
                    species_densities[spec] = float(val)

        return ne, f, species_densities

    return ne, f


__all__ = [
    "ChemEqData",
    "IonizationData",
    "PartitionFunctionTable",
    "MoleculeData",
    "prepare_chem_eq_data",
    "prepare_ionization_arrays",
    "prepare_partition_function_tables",
    "prepare_molecule_data",
    "make_optimality_fun",
    "make_solver",
    "initial_guess_y",
    "unpack_solution",
    "compute_species_densities_arrays",
    "solve_equilibrium",
    "chemical_equilibrium_jax",
]
