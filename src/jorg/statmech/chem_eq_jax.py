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
import time
from typing import Callable, Dict, Optional, Tuple, Sequence, List

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
_DEFAULT_RESIDUAL_ACCEPT_TOL = 1e-6
_HYBRID_BACKEND_DEFAULT_MAXITER = 80
_ADDITIONAL_NE_SEED_TRIGGER_RESIDUAL = 1e-1
_REFERENCE_REFINE_MAX_STEPS = 3
_REFERENCE_REFINE_PLATEAU_RESIDUAL = 5e-6
_REFERENCE_REFINE_MIN_IMPROVEMENT_FRAC = 0.05


_CHEM_DATA_CACHE: Dict[Tuple[int, int, int], "ChemEqData"] = {}
_SOLVER_CACHE: Dict[Tuple[int, str, int, float, float, int, bool], Tuple[Callable, Callable]] = {}
_SCAN_SOLVER_CACHE: Dict[Tuple[int, bool], Callable] = {}


def clear_solver_caches() -> None:
    """
    Clear cached solver callables that may retain trace-time closures.
    """
    _SOLVER_CACHE.clear()
    _SCAN_SOLVER_CACHE.clear()


def _logit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x / (1.0 - x))


def _set_ne_frac_seed(y: np.ndarray, ne_frac: float) -> np.ndarray:
    y_seed = np.asarray(y, dtype=np.float64).copy()
    y_seed[-1] = float(_logit(np.array(ne_frac, dtype=np.float64)))
    return y_seed


def _append_unique_seed(
    seeds: List[Tuple[str, np.ndarray]],
    label: str,
    y_seed: np.ndarray,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    y_seed = np.asarray(y_seed, dtype=np.float64)
    for _, existing in seeds:
        if np.allclose(existing, y_seed, atol=atol, rtol=rtol):
            return
    seeds.append((label, y_seed))


def _get_cached_chem_data(
    ionization_energies: Dict[int, Tuple[float, float, float]],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Optional[Dict],
) -> ChemEqData:
    key = (id(ionization_energies), id(partition_funcs), id(log_equilibrium_constants))
    cached = _CHEM_DATA_CACHE.get(key)
    if cached is not None:
        return cached
    chem_data = prepare_chem_eq_data(
        ionization_energies=ionization_energies,
        partition_funcs=partition_funcs,
        log_equilibrium_constants=log_equilibrium_constants,
    )
    _CHEM_DATA_CACHE[key] = chem_data
    return chem_data


def _get_cached_optimality_and_solver(
    chem_data: ChemEqData,
    *,
    method: str,
    maxiter: int,
    tol: float,
    cg_tol: float,
    cg_maxiter: int,
    jit: bool,
) -> Tuple[Callable, Callable]:
    key = (
        id(chem_data),
        str(method).lower(),
        int(maxiter),
        float(tol),
        float(cg_tol),
        int(cg_maxiter),
        bool(jit),
    )
    cached = _SOLVER_CACHE.get(key)
    if cached is not None:
        return cached

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
    _SOLVER_CACHE[key] = (optimality_fun, solve)
    return optimality_fun, solve


def _get_cached_scan_solver(solve: Callable, *, jit: bool) -> Callable:
    key = (id(solve), bool(jit))
    cached = _SCAN_SOLVER_CACHE.get(key)
    if cached is not None:
        return cached

    def _scan_solve(temps_x, nts_x, y_guess_x, y_init, abs_abund_x, warm_start_flag):
        def _body(y_prev, payload):
            T_i, nt_i, y_guess_i = payload
            y_seed = jax.lax.cond(
                warm_start_flag,
                lambda _: y_prev,
                lambda _: y_guess_i,
                operand=None,
            )
            y_i = solve(y_seed, T_i, nt_i, abs_abund_x)
            return y_i, y_i

        _, y_hist = jax.lax.scan(_body, y_init, (temps_x, nts_x, y_guess_x))
        return y_hist

    scan_solver = jax.jit(_scan_solve) if jit else _scan_solve
    _SCAN_SOLVER_CACHE[key] = scan_solver
    return scan_solver


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

    def _tabulate_logK(log_K_func: Callable, grid: np.ndarray) -> Optional[np.ndarray]:
        """
        Evaluate a molecule logK spline on the shared JAX grid.

        Korg evaluates each molecule only at the requested temperature. The
        JAX path pre-tabulates onto a common `logT_grid`, so we need to be more
        permissive at the edges: Barklem/Collet splines often have slightly
        narrower support than the atomic partition-function grid. In that case
        we clip to the spline domain instead of dropping the molecule entirely.
        """
        vals: Optional[np.ndarray] = None

        try:
            raw = log_K_func(grid)
            vals = np.asarray(raw, dtype=np.float64)
        except Exception:
            vals = None

        if vals is None or vals.shape != grid.shape or not np.all(np.isfinite(vals)):
            knot_grid = getattr(log_K_func, "x", None)
            if knot_grid is None:
                knot_grid = getattr(log_K_func, "_x", None)
            if knot_grid is not None:
                knot_grid = np.asarray(knot_grid, dtype=np.float64)
                if knot_grid.ndim == 1 and knot_grid.size >= 2:
                    clipped = np.clip(grid, knot_grid[0], knot_grid[-1])
                    try:
                        raw = log_K_func(clipped)
                        vals = np.asarray(raw, dtype=np.float64)
                    except Exception:
                        vals = None

        if vals is None or vals.shape != grid.shape:
            sampled = []
            for logT in grid:
                try:
                    sampled.append(float(log_K_func(float(logT))))
                except Exception:
                    sampled.append(np.nan)
            vals = np.asarray(sampled, dtype=np.float64)

        finite = np.isfinite(vals)
        if not np.any(finite):
            return None
        if not np.all(finite):
            finite_idx = np.flatnonzero(finite)
            if finite_idx.size == 1:
                vals = np.full_like(vals, vals[finite_idx[0]], dtype=np.float64)
            else:
                vals = np.interp(grid, grid[finite_idx], vals[finite_idx])
        return vals

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
            logKp_vals = _tabulate_logK(log_K_func, logT_grid)
            if logKp_vals is None:
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


def _batch_species_order(chem_data: ChemEqData) -> List[Species]:
    """
    Stable species order for dense batch CE outputs.

    Preserve the historical atomic block layout `[neutral, ion1, ion2]` and
    append molecular species afterward, matching Korg's "solve atomic system,
    then reconstruct molecules" flow.
    """
    ordered: List[Species] = []
    for charge in range(3):
        for z in range(1, MAX_ATOMIC_NUMBER + 1):
            ordered.append(Species.from_atomic_number(z, charge))

    if chem_data.mol is not None:
        if chem_data.mol.neutral_species is not None:
            ordered.extend(list(chem_data.mol.neutral_species))
        if chem_data.mol.charged_species is not None:
            ordered.extend(list(chem_data.mol.charged_species))

    return ordered


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


def _evaluate_residual_quality(
    optimality_fun: Callable,
    y: np.ndarray,
    temp: float,
    nt: float,
    abs_abund_array: np.ndarray,
    *,
    temp_j=None,
    nt_j=None,
    abs_abund_j=None,
    include_median: bool = True,
) -> Dict[str, float]:
    F = optimality_fun(
        jnp.asarray(y, dtype=jnp.float64),
        jnp.asarray(temp, dtype=jnp.float64) if temp_j is None else temp_j,
        jnp.asarray(nt, dtype=jnp.float64) if nt_j is None else nt_j,
        jnp.asarray(abs_abund_array, dtype=jnp.float64) if abs_abund_j is None else abs_abund_j,
    )
    F_np = np.asarray(jax.device_get(F), dtype=np.float64)
    abs_F = np.abs(F_np)
    out = {
        "max_abs_residual": float(np.max(abs_F)),
        "charge_residual": float(F_np[-1]),
    }
    if include_median:
        out["median_abs_residual"] = float(np.median(abs_F))
    return out


def _y_from_reference_solution(
    ne: float,
    nt: float,
    abs_abund_array: np.ndarray,
    species_densities: Dict[Species, float],
) -> np.ndarray:
    n_nuc = np.asarray(abs_abund_array, dtype=np.float64) * max(float(nt) - float(ne), 1e-300)
    n0 = np.array(
        [float(species_densities.get(Species.from_atomic_number(Z, 0), 0.0)) for Z in range(1, MAX_ATOMIC_NUMBER + 1)],
        dtype=np.float64,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(n_nuc > 0.0, n0 / n_nuc, 0.0)
    y_f = _logit(f)
    y_e = _logit(np.array(float(ne) / max(float(nt), 1e-300), dtype=np.float64))
    return np.concatenate([y_f, [y_e]]).astype(np.float64)


def _y_from_jaxopt_solution_x(x: np.ndarray, nt: float) -> np.ndarray:
    x_np = np.asarray(x, dtype=np.float64)
    f = np.clip(np.abs(x_np[:-1]), 1e-12, 1.0 - 1e-12)
    ne = max(float(np.abs(x_np[-1]) * float(nt) * 1e-5), 1e-300)
    y_f = _logit(f)
    y_e = _logit(np.array(ne / max(float(nt), 1e-300), dtype=np.float64))
    return np.concatenate([y_f, [y_e]]).astype(np.float64)


def _f_from_jaxopt_solution_x(x: np.ndarray) -> np.ndarray:
    x_np = np.asarray(x, dtype=np.float64)
    return np.clip(np.abs(x_np[:-1]), 0.0, 1.0)


def _resolve_chemical_equilibrium_solution(
    *,
    solve: Callable,
    optimality_fun: Callable,
    temp: float,
    nt: float,
    model_atm_ne: float,
    abs_abund_array: np.ndarray,
    ionization_energies: Dict[int, Tuple[float, float, float]],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Optional[Dict],
    base_y0: np.ndarray,
    residual_accept_tol: float,
    fallback: bool,
    first_label: str,
    prev_y: Optional[np.ndarray] = None,
    temp_j=None,
    nt_j=None,
    abs_abund_j=None,
    record_attempts: bool = True,
) -> Tuple[np.ndarray, Dict[str, object]]:
    t_resolution_start = time.perf_counter()
    attempt_info: List[Dict[str, object]] = []
    attempt_count = 0
    best_y: Optional[np.ndarray] = None
    best_quality: Optional[Dict[str, float]] = None
    best_label: Optional[str] = None
    timings_s = {
        "candidate_solve_s": 0.0,
        "residual_eval_s": 0.0,
        "reference_scipy_s": 0.0,
        "reference_refine_s": 0.0,
    }

    seen_seeds: List[np.ndarray] = []

    def _run_attempt(label: str, y_seed: np.ndarray):
        nonlocal attempt_count, best_y, best_quality, best_label
        t_solve = time.perf_counter()
        y_try = solve(
            jnp.asarray(y_seed, dtype=jnp.float64),
            jnp.asarray(temp, dtype=jnp.float64) if temp_j is None else temp_j,
            jnp.asarray(nt, dtype=jnp.float64) if nt_j is None else nt_j,
            jnp.asarray(abs_abund_array, dtype=jnp.float64) if abs_abund_j is None else abs_abund_j,
        )
        timings_s["candidate_solve_s"] += time.perf_counter() - t_solve
        y_try_np = np.asarray(jax.device_get(y_try), dtype=np.float64)
        t_quality = time.perf_counter()
        quality = _evaluate_residual_quality(
            optimality_fun,
            y_try_np,
            temp,
            nt,
            abs_abund_array,
            temp_j=temp_j,
            nt_j=nt_j,
            abs_abund_j=abs_abund_j,
            include_median=record_attempts,
        )
        timings_s["residual_eval_s"] += time.perf_counter() - t_quality
        attempt_count += 1
        if record_attempts:
            attempt_info.append(
                {
                    "solver": label,
                    "max_abs_residual": quality["max_abs_residual"],
                    "median_abs_residual": quality.get("median_abs_residual", np.nan),
                    "charge_residual": quality["charge_residual"],
                }
            )
        if best_quality is None or quality["max_abs_residual"] < best_quality["max_abs_residual"]:
            best_y = y_try_np
            best_quality = quality
            best_label = label
        if np.isfinite(quality["max_abs_residual"]) and quality["max_abs_residual"] <= float(residual_accept_tol):
            return y_try_np, {
                "max_abs_residual": quality["max_abs_residual"],
                "accepted_solver": label,
                "fallback_used": label != first_label,
                "attempt_count": attempt_count,
                "reference_fallback_used": False,
                "attempts": tuple(attempt_info),
                "timings_s": {
                    **timings_s,
                    "resolution_total_s": time.perf_counter() - t_resolution_start,
                },
            }
        return None

    first_seed = np.asarray(base_y0, dtype=np.float64)
    seen_seeds.append(first_seed)
    accepted = _run_attempt(first_label, first_seed)
    if accepted is not None:
        return accepted

    extra_candidate_seeds: List[Tuple[str, np.ndarray]] = []
    if prev_y is not None:
        _append_unique_seed(extra_candidate_seeds, "jax:warm_start_prev", np.asarray(prev_y, dtype=np.float64))
    first_max_residual = np.inf if best_quality is None else float(best_quality["max_abs_residual"])
    if (not np.isfinite(first_max_residual)) or first_max_residual > _ADDITIONAL_NE_SEED_TRIGGER_RESIDUAL:
        _append_unique_seed(extra_candidate_seeds, "jax:ne_frac_1e-12", _set_ne_frac_seed(base_y0, 1e-12))
        _append_unique_seed(extra_candidate_seeds, "jax:ne_frac_1e-8", _set_ne_frac_seed(base_y0, 1e-8))

    for label, y_seed in extra_candidate_seeds:
        if any(np.array_equal(y_seed, seen) for seen in seen_seeds):
            continue
        seen_seeds.append(y_seed)
        accepted = _run_attempt(label, y_seed)
        if accepted is not None:
            return accepted

    if fallback:
        from .korg_chemical_equilibrium import chemical_equilibrium as chemical_equilibrium_scipy

        t_reference = time.perf_counter()
        ne_ref, species_ref = chemical_equilibrium_scipy(
            float(temp),
            float(nt),
            float(model_atm_ne),
            np.asarray(abs_abund_array, dtype=np.float64),
            ionization_energies,
            partition_funcs,
            log_equilibrium_constants,
            warn_on_ne_discrepancy=False,
        )
        timings_s["reference_scipy_s"] += time.perf_counter() - t_reference
        y_ref = _y_from_reference_solution(ne_ref, nt, abs_abund_array, species_ref)
        t_quality = time.perf_counter()
        quality = _evaluate_residual_quality(
            optimality_fun,
            y_ref,
            temp,
            nt,
            abs_abund_array,
            temp_j=temp_j,
            nt_j=nt_j,
            abs_abund_j=abs_abund_j,
            include_median=record_attempts,
        )
        timings_s["residual_eval_s"] += time.perf_counter() - t_quality
        attempt_count += 1
        if record_attempts:
            attempt_info.append(
                {
                    "solver": "reference_scipy",
                    "max_abs_residual": quality["max_abs_residual"],
                    "median_abs_residual": quality.get("median_abs_residual", np.nan),
                    "charge_residual": quality["charge_residual"],
                }
            )
        if best_quality is None or quality["max_abs_residual"] < best_quality["max_abs_residual"]:
            best_y = y_ref
            best_quality = quality
            best_label = "reference_scipy"
        if np.isfinite(quality["max_abs_residual"]) and quality["max_abs_residual"] <= float(residual_accept_tol):
            return y_ref, {
                "max_abs_residual": quality["max_abs_residual"],
                "accepted_solver": "reference_scipy",
                "fallback_used": True,
                "attempt_count": attempt_count,
                "reference_fallback_used": True,
                "attempts": tuple(attempt_info),
                "timings_s": {
                    **timings_s,
                    "resolution_total_s": time.perf_counter() - t_resolution_start,
                },
            }

        y_refine_seed = np.asarray(y_ref, dtype=np.float64)
        prev_refine_quality = quality
        for refine_idx in range(_REFERENCE_REFINE_MAX_STEPS):
            solver_label = "jax:from_reference" if refine_idx == 0 else f"jax:from_reference:{refine_idx + 1}"
            t_refine = time.perf_counter()
            y_refined = solve(
                jnp.asarray(y_refine_seed, dtype=jnp.float64),
                jnp.asarray(temp, dtype=jnp.float64) if temp_j is None else temp_j,
                jnp.asarray(nt, dtype=jnp.float64) if nt_j is None else nt_j,
                jnp.asarray(abs_abund_array, dtype=jnp.float64) if abs_abund_j is None else abs_abund_j,
            )
            timings_s["reference_refine_s"] += time.perf_counter() - t_refine
            y_refined_np = np.asarray(jax.device_get(y_refined), dtype=np.float64)
            t_quality = time.perf_counter()
            refined_quality = _evaluate_residual_quality(
                optimality_fun,
                y_refined_np,
                temp,
                nt,
                abs_abund_array,
                temp_j=temp_j,
                nt_j=nt_j,
                abs_abund_j=abs_abund_j,
                include_median=record_attempts,
            )
            timings_s["residual_eval_s"] += time.perf_counter() - t_quality
            attempt_count += 1
            if record_attempts:
                attempt_info.append(
                    {
                        "solver": solver_label,
                        "max_abs_residual": refined_quality["max_abs_residual"],
                        "median_abs_residual": refined_quality.get("median_abs_residual", np.nan),
                        "charge_residual": refined_quality["charge_residual"],
                    }
                )
            if best_quality is None or refined_quality["max_abs_residual"] < best_quality["max_abs_residual"]:
                best_y = y_refined_np
                best_quality = refined_quality
                best_label = solver_label
            if np.isfinite(refined_quality["max_abs_residual"]) and refined_quality["max_abs_residual"] <= float(
                residual_accept_tol
            ):
                return y_refined_np, {
                    "max_abs_residual": refined_quality["max_abs_residual"],
                    "accepted_solver": solver_label,
                    "fallback_used": True,
                    "attempt_count": attempt_count,
                    "reference_fallback_used": True,
                    "attempts": tuple(attempt_info),
                    "timings_s": {
                        **timings_s,
                        "resolution_total_s": time.perf_counter() - t_resolution_start,
                    },
                }
            if np.isfinite(refined_quality["max_abs_residual"]):
                if refined_quality["max_abs_residual"] <= _REFERENCE_REFINE_PLATEAU_RESIDUAL:
                    break
                prev_val = float(prev_refine_quality["max_abs_residual"])
                cur_val = float(refined_quality["max_abs_residual"])
                if np.isfinite(prev_val) and prev_val > 0.0:
                    improvement_frac = (prev_val - cur_val) / prev_val
                    if improvement_frac <= _REFERENCE_REFINE_MIN_IMPROVEMENT_FRAC:
                        break
            prev_refine_quality = refined_quality
            y_refine_seed = y_refined_np

    assert best_y is not None
    assert best_quality is not None
    return best_y, {
        "max_abs_residual": best_quality["max_abs_residual"],
        "accepted_solver": best_label or "jax:unknown",
        "fallback_used": attempt_count > 1,
        "attempt_count": attempt_count,
        "reference_fallback_used": False,
        "attempts": tuple(attempt_info),
        "timings_s": {
            **timings_s,
            "resolution_total_s": time.perf_counter() - t_resolution_start,
        },
    }


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


def _coerce_absolute_abundance_array(
    absolute_abundances: Dict[int, float] | np.ndarray,
) -> jnp.ndarray:
    if isinstance(absolute_abundances, dict):
        abs_abund_array = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
        for Z, abund in absolute_abundances.items():
            if 1 <= Z <= MAX_ATOMIC_NUMBER:
                abs_abund_array[Z - 1] = abund
        return jnp.asarray(abs_abund_array, dtype=jnp.float64)

    abs_abund_array = jnp.asarray(absolute_abundances, dtype=jnp.float64)
    if abs_abund_array.ndim != 1 or abs_abund_array.shape[0] != MAX_ATOMIC_NUMBER:
        raise ValueError(
            f"absolute_abundances must be a length-{MAX_ATOMIC_NUMBER} vector."
        )
    return abs_abund_array


def chemical_equilibrium_jax_layers(
    temps: np.ndarray,
    nts: np.ndarray,
    model_atm_nes: np.ndarray,
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
    warm_start: bool = True,
    fallback: bool = True,
    residual_accept_tol: float = _DEFAULT_RESIDUAL_ACCEPT_TOL,
    return_info: bool = False,
):
    """
    Solve CE for all layers with JAX scan + warm-start.

    Returns
    -------
    tuple
        (electron_density, number_density_dense, y_solutions, species_layout)
        where number_density_dense has shape [n_layers, n_species] with
        atomic block columns `[neutral, ion1, ion2]` followed by any molecular
        species included in `log_equilibrium_constants`.
    """
    _require_jax()

    from ..core.state_jax import DenseSpeciesLayout

    temps_j = jnp.asarray(temps, dtype=jnp.float64)
    nts_j = jnp.asarray(nts, dtype=jnp.float64)
    model_atm_nes_j = jnp.asarray(model_atm_nes, dtype=jnp.float64)
    if not (temps_j.shape == nts_j.shape == model_atm_nes_j.shape):
        raise ValueError("temps, nts, and model_atm_nes must have identical shapes.")
    if temps_j.size == 0:
        raise ValueError("temps/nts/model_atm_nes must be non-empty.")

    abs_abund_array = _coerce_absolute_abundance_array(absolute_abundances)

    if chem_data is None:
        chem_data = _get_cached_chem_data(
            ionization_energies=ionization_energies,
            partition_funcs=partition_funcs,
            log_equilibrium_constants=log_equilibrium_constants,
        )

    optimality_fun, solve = _get_cached_optimality_and_solver(
        chem_data,
        method=method,
        maxiter=maxiter,
        tol=tol,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        jit=jit,
    )

    abs_abund_j = jnp.asarray(abs_abund_array, dtype=jnp.float64)

    # Tracer-safe initial guess: start from abundance ratios and model ne fraction.
    log_abund = jnp.log(jnp.maximum(abs_abund_j, 1e-300))
    y_f0 = log_abund - jnp.max(log_abund)
    ne_frac0 = jnp.clip(
        model_atm_nes_j[0] / jnp.maximum(nts_j[0], 1e-300),
        1e-12,
        1.0 - 1e-12,
    )
    y_e0 = jnp.log(ne_frac0 / (1.0 - ne_frac0))
    y0_first = jnp.concatenate((y_f0, jnp.asarray([y_e0], dtype=jnp.float64)), axis=0)

    # Per-layer fallback guesses for non-warm-start mode.
    ne_frac_guess = jnp.clip(
        model_atm_nes_j / jnp.maximum(nts_j, 1e-300),
        1e-12,
        1.0 - 1e-12,
    )
    y_e_guess = jnp.log(ne_frac_guess / (1.0 - ne_frac_guess))
    y_f_guess = jnp.broadcast_to(y0_first[:-1], (temps_j.shape[0], MAX_ATOMIC_NUMBER))
    y_guess_all = jnp.concatenate((y_f_guess, y_e_guess[:, None]), axis=1)
    warm_start_flag = jnp.asarray(bool(warm_start))
    scan_solver = _get_cached_scan_solver(solve, jit=jit)
    y_solutions = scan_solver(
        temps_j,
        nts_j,
        y_guess_all,
        y0_first,
        abs_abund_j,
        warm_start_flag,
    )
    y_solutions_np = np.asarray(jax.device_get(y_solutions), dtype=np.float64)
    temps_np = np.asarray(jax.device_get(temps_j), dtype=np.float64)
    nts_np = np.asarray(jax.device_get(nts_j), dtype=np.float64)
    model_nes_np = np.asarray(jax.device_get(model_atm_nes_j), dtype=np.float64)
    abs_abund_np = np.asarray(jax.device_get(abs_abund_j), dtype=np.float64)

    final_y = np.array(y_solutions_np, copy=True)
    layer_info: List[Dict[str, object]] = []
    for i in range(final_y.shape[0]):
        prev_y = final_y[i - 1] if warm_start and i > 0 else None
        resolved_y, info = _resolve_chemical_equilibrium_solution(
            solve=solve,
            optimality_fun=optimality_fun,
            temp=float(temps_np[i]),
            nt=float(nts_np[i]),
            model_atm_ne=float(model_nes_np[i]),
            abs_abund_array=abs_abund_np,
            ionization_energies=ionization_energies,
            partition_funcs=partition_funcs,
            log_equilibrium_constants=log_equilibrium_constants,
            base_y0=final_y[i],
            residual_accept_tol=residual_accept_tol,
            fallback=fallback,
            first_label="jax:scan",
            prev_y=prev_y,
        )
        final_y[i] = resolved_y
        layer_info.append(info)

    y_solutions = jnp.asarray(final_y, dtype=jnp.float64)

    def _layer_outputs(y_i, T_i, nt_i):
        ne_i, n0_i, n1_i, n2_i, n_mol_neutral_i, n_mol_charged_i = compute_species_densities_arrays(
            y_i, T_i, nt_i, abs_abund_j, chem_data
        )
        dense_parts = [n0_i, n1_i, n2_i]
        if n_mol_neutral_i is not None:
            dense_parts.append(n_mol_neutral_i)
        if n_mol_charged_i is not None:
            dense_parts.append(n_mol_charged_i)
        dense_i = jnp.concatenate(tuple(dense_parts), axis=0)
        return ne_i, dense_i

    ne_layers, number_density_dense = jax.vmap(_layer_outputs)(y_solutions, temps_j, nts_j)

    species_layout = DenseSpeciesLayout.from_species(_batch_species_order(chem_data))
    if return_info:
        info = {
            "max_abs_residual": np.asarray([float(item["max_abs_residual"]) for item in layer_info], dtype=np.float64),
            "accepted_solver": tuple(str(item["accepted_solver"]) for item in layer_info),
            "fallback_used": np.asarray([bool(item["fallback_used"]) for item in layer_info], dtype=bool),
            "attempt_count": np.asarray([int(item["attempt_count"]) for item in layer_info], dtype=np.int32),
            "reference_fallback_used": np.asarray(
                [bool(item["reference_fallback_used"]) for item in layer_info], dtype=bool
            ),
            "attempts": tuple(item["attempts"] for item in layer_info),
        }
        return ne_layers, number_density_dense, y_solutions, species_layout, info

    return ne_layers, number_density_dense, y_solutions, species_layout


def _species_densities_from_y(
    *,
    y_np: np.ndarray,
    temp: float,
    nt: float,
    abs_abund_array: np.ndarray,
    chem_data: ChemEqData,
) -> Dict[Species, float]:
    T_j = jnp.asarray(temp, dtype=jnp.float64)
    nt_j = jnp.asarray(nt, dtype=jnp.float64)
    abund_j = jnp.asarray(abs_abund_array, dtype=jnp.float64)
    _, n0, n1, n2, n_mol_neutral, n_mol_charged = compute_species_densities_arrays(
        jnp.asarray(y_np, dtype=jnp.float64),
        T_j,
        nt_j,
        abund_j,
        chem_data,
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

    return species_densities


def _finalize_single_case_outputs(
    *,
    ne: float,
    y_np: np.ndarray,
    info: Dict[str, object],
    temp: float,
    nt: float,
    abs_abund_array: np.ndarray,
    chem_data: ChemEqData,
    return_species: bool,
    return_y: bool,
    return_info: bool,
    species_densities: Optional[Dict[Species, float]] = None,
):
    f = 1.0 / (1.0 + np.exp(-y_np[:-1]))

    if return_y:
        if return_info:
            return ne, f, y_np, info
        return ne, f, y_np

    if return_species:
        if species_densities is None:
            species_t0 = time.perf_counter()
            species_densities = _species_densities_from_y(
                y_np=y_np,
                temp=temp,
                nt=nt,
                abs_abund_array=abs_abund_array,
                chem_data=chem_data,
            )
            if return_info:
                timings = dict(info.get("timings_s", {}))
                timings["species_rebuild_s"] = float(timings.get("species_rebuild_s", 0.0)) + (
                    time.perf_counter() - species_t0
                )
                info = dict(info)
                info["timings_s"] = timings
        if return_info:
            return ne, f, species_densities, info
        return ne, f, species_densities

    if return_info:
        return ne, f, info
    return ne, f


def _chemical_equilibrium_jax_hybrid(
    *,
    temp: float,
    nt: float,
    model_atm_ne: float,
    abs_abund_array: np.ndarray,
    ionization_energies: Dict[int, Tuple[float, float, float]],
    partition_funcs: Dict[Species, Callable],
    log_equilibrium_constants: Optional[Dict],
    return_species: bool,
    return_y: bool,
    return_info: bool,
    fallback: bool,
    residual_accept_tol: float,
    maxiter: int,
    tol: float,
    jit: bool,
):
    t_total_start = time.perf_counter()
    from .korg_chemical_equilibrium_jaxopt import chemical_equilibrium_jaxopt

    t_cache = time.perf_counter()
    chem_data = _get_cached_chem_data(ionization_energies, partition_funcs, log_equilibrium_constants)
    cache_lookup_s = time.perf_counter() - t_cache
    method = "auto" if log_equilibrium_constants is None else "hybrid"
    stats_out: Dict[str, object] = {}
    try:
        t_solve = time.perf_counter()
        if return_species:
            ne, species_densities, x_solution = chemical_equilibrium_jaxopt(
                temp=float(temp),
                nt=float(nt),
                model_atm_ne=float(model_atm_ne),
                absolute_abundances=abs_abund_array,
                ionization_energies=ionization_energies,
                partition_funcs=partition_funcs,
                log_equilibrium_constants=log_equilibrium_constants,
                maxiter=maxiter,
                tol=tol,
                method=method,
                jit=jit,
                return_species=True,
                return_x=True,
                stats_out=stats_out,
            )
        else:
            ne, x_solution = chemical_equilibrium_jaxopt(
                temp=float(temp),
                nt=float(nt),
                model_atm_ne=float(model_atm_ne),
                absolute_abundances=abs_abund_array,
                ionization_energies=ionization_energies,
                partition_funcs=partition_funcs,
                log_equilibrium_constants=log_equilibrium_constants,
                maxiter=maxiter,
                tol=tol,
                method=method,
                jit=jit,
                return_species=False,
                return_x=True,
                stats_out=stats_out,
            )
            species_densities = None
        hybrid_solve_s = time.perf_counter() - t_solve
    except Exception:
        if not fallback:
            raise
        return None

    residual_max = float(stats_out.get("residual_max", np.nan))
    if not np.isfinite(residual_max) or residual_max > float(residual_accept_tol):
        if not fallback:
            raise RuntimeError(
                f"Hybrid backend residual {residual_max!r} exceeds acceptance tolerance {residual_accept_tol}."
            )
        return None

    if not return_species and not return_y and not return_info:
        return float(ne), _f_from_jaxopt_solution_x(x_solution)

    if species_densities is not None:
        y_np = _y_from_reference_solution(ne, nt, abs_abund_array, species_densities)
    else:
        y_np = _y_from_jaxopt_solution_x(x_solution, nt)
    attempt_count = 1
    lm_attempts = stats_out.get("lm_attempts", [])
    if isinstance(lm_attempts, list):
        attempt_count += len(lm_attempts)
    hybrid_stats = stats_out.get("hybrid_stats", {})
    if isinstance(hybrid_stats, dict):
        hybrid_runs = hybrid_stats.get("runs", [])
        if isinstance(hybrid_runs, list):
            attempt_count += len(hybrid_runs)

    info = {
        "max_abs_residual": residual_max,
        "accepted_solver": f"hybrid:{stats_out.get('method', method)}",
        "fallback_used": bool(stats_out.get("lm_fallback_used", False) or stats_out.get("hybrid_used", False)),
        "attempt_count": int(max(attempt_count, 1)),
        "reference_fallback_used": False,
        "attempts": tuple(),
        "backend": "hybrid",
        "timings_s": {
            "cache_lookup_s": float(cache_lookup_s),
            "hybrid_solve_s": float(hybrid_solve_s),
            "species_rebuild_s": 0.0,
            "total_s": float(time.perf_counter() - t_total_start),
        },
    }
    return _finalize_single_case_outputs(
        ne=float(ne),
        y_np=np.asarray(y_np, dtype=np.float64),
        info=info,
        temp=float(temp),
        nt=float(nt),
        abs_abund_array=np.asarray(abs_abund_array, dtype=np.float64),
        chem_data=chem_data,
        return_species=return_species,
        return_y=return_y,
        return_info=return_info,
        species_densities=species_densities,
    )


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
    residual_accept_tol: float = _DEFAULT_RESIDUAL_ACCEPT_TOL,
    return_info: bool = False,
    backend: str = "stable",
) -> Tuple[float, np.ndarray] | Tuple[float, np.ndarray, Dict[Species, float]] | Tuple[float, np.ndarray, np.ndarray]:
    _require_jax()
    t_total_start = time.perf_counter()

    abs_abund_array = _coerce_absolute_abundance_array(absolute_abundances)
    backend_norm = str(backend).lower()
    if backend_norm not in ("stable", "hybrid"):
        raise ValueError(f"backend must be 'stable' or 'hybrid', got {backend!r}.")

    if backend_norm == "hybrid":
        hybrid_maxiter = _HYBRID_BACKEND_DEFAULT_MAXITER if int(maxiter) == 300 else int(maxiter)
        hybrid_result = _chemical_equilibrium_jax_hybrid(
            temp=float(temp),
            nt=float(nt),
            model_atm_ne=float(model_atm_ne),
            abs_abund_array=np.asarray(abs_abund_array, dtype=np.float64),
            ionization_energies=ionization_energies,
            partition_funcs=partition_funcs,
            log_equilibrium_constants=log_equilibrium_constants,
            return_species=return_species,
            return_y=return_y,
            return_info=return_info,
            fallback=fallback,
            residual_accept_tol=residual_accept_tol,
            maxiter=hybrid_maxiter,
            tol=tol,
            jit=jit,
        )
        if hybrid_result is not None:
            return hybrid_result

    t_cache = time.perf_counter()
    if chem_data is None:
        chem_data = _get_cached_chem_data(
            ionization_energies,
            partition_funcs,
            log_equilibrium_constants,
        )
    cache_lookup_s = time.perf_counter() - t_cache

    t_initial_guess = time.perf_counter()
    y0 = initial_guess_y(
        float(temp),
        float(nt),
        float(model_atm_ne),
        chem_data,
    )
    initial_guess_s = time.perf_counter() - t_initial_guess

    t_solver_setup = time.perf_counter()
    optimality_fun, solve = _get_cached_optimality_and_solver(
        chem_data,
        method=method,
        maxiter=maxiter,
        tol=tol,
        cg_tol=cg_tol,
        cg_maxiter=cg_maxiter,
        jit=jit,
    )
    solver_setup_s = time.perf_counter() - t_solver_setup

    t_device_args = time.perf_counter()
    temp_j = jnp.asarray(temp, dtype=jnp.float64)
    nt_j = jnp.asarray(nt, dtype=jnp.float64)
    abs_abund_j = jnp.asarray(abs_abund_array, dtype=jnp.float64)
    device_args_s = time.perf_counter() - t_device_args

    y_np, info = _resolve_chemical_equilibrium_solution(
        solve=solve,
        optimality_fun=optimality_fun,
        temp=float(temp),
        nt=float(nt),
        model_atm_ne=float(model_atm_ne),
        abs_abund_array=np.asarray(abs_abund_array, dtype=np.float64),
        ionization_energies=ionization_energies,
        partition_funcs=partition_funcs,
        log_equilibrium_constants=log_equilibrium_constants,
        base_y0=np.asarray(y0, dtype=np.float64),
        residual_accept_tol=residual_accept_tol,
        fallback=fallback,
        first_label="jax:initial",
        temp_j=temp_j,
        nt_j=nt_j,
        abs_abund_j=abs_abund_j,
        record_attempts=return_info,
    )
    info = dict(info)
    timings = dict(info.get("timings_s", {}))
    timings.setdefault("species_rebuild_s", 0.0)
    timings.update(
        {
            "cache_lookup_s": float(cache_lookup_s),
            "initial_guess_s": float(initial_guess_s),
            "solver_setup_s": float(solver_setup_s),
            "device_args_s": float(device_args_s),
            "total_s": float(time.perf_counter() - t_total_start),
        }
    )
    info["timings_s"] = timings
    info["backend"] = "stable"
    ne = float(nt) * (1.0 / (1.0 + np.exp(-y_np[-1])))
    return _finalize_single_case_outputs(
        ne=ne,
        y_np=np.asarray(y_np, dtype=np.float64),
        info=info,
        temp=float(temp),
        nt=float(nt),
        abs_abund_array=np.asarray(abs_abund_array, dtype=np.float64),
        chem_data=chem_data,
        return_species=return_species,
        return_y=return_y,
        return_info=return_info,
    )


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
    "chemical_equilibrium_jax_layers",
    "chemical_equilibrium_jax",
]
