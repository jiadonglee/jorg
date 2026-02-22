#!/usr/bin/env python3
"""PINN vs SciPy CE pipeline comparison against Korg outputs.

Outputs are saved into:
  output/benchmarks/pinn_korg_jorg_compare/<run_id>/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _ensure_src_on_path() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        src = parent / "src"
        if (src / "jorg").exists():
            sys.path.insert(0, str(src))
            data_dir = parent / "data"
            if data_dir.exists():
                os.environ.setdefault("JORG_DATA_DIR", str(data_dir))
            return parent
    raise RuntimeError("Could not locate repo root with src/jorg.")


REPO_ROOT = _ensure_src_on_path()
DEFAULT_PINN_MODEL = (
    REPO_ROOT / "output" / "pinn" / "joint_phase2_stable_from_benchmark" / "chem_eq_pinn_model.npz"
)
DEFAULT_KORG_SCRIPT = REPO_ROOT / "examples" / "korg_compare_params.jl"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "benchmarks" / "pinn_korg_jorg_compare"
DEFAULT_COMPARISON_DIRS = [
    REPO_ROOT.parent / "comparison_outputs",
    REPO_ROOT / "examples" / "comparison_outputs",
]
DEFAULT_SINGLE_RANGE = (5000.0, 5002.0)
DEFAULT_GRID_RANGE = (5000.0, 5020.0)

SINGLE_CASE = {"Teff": 5772.0, "logg": 4.44, "m_H": 0.0}
GRID_CASES = [
    {"Teff": 5771.0, "logg": 4.44, "m_H": 0.0},
    {"Teff": 4250.0, "logg": 1.40, "m_H": -0.5},
    {"Teff": 4500.0, "logg": 1.50, "m_H": -2.0},  # clipped from -2.5
    {"Teff": 5000.0, "logg": 4.00, "m_H": -0.5},
    {"Teff": 6000.0, "logg": 4.50, "m_H": 0.2},
]


from jorg.atmosphere import interpolate_marcs
from jorg.data import get_data_path
from jorg.lines.linelist import read_linelist
from jorg.synthesis import create_korg_compatible_abundance_array, synthesize
from jorg.statmech import create_default_ionization_energies, create_default_partition_functions
from jorg.statmech.chem_eq_jax import prepare_chem_eq_data
from jorg.statmech.chem_eq_pinn_inference import (
    build_atomic_ce_source_from_solver,
    load_pinn_solver_from_checkpoint,
)


class KorgSynthesisError(RuntimeError):
    """Raised when Korg auto-run is requested but execution fails."""


@dataclass(frozen=True)
class Case:
    Teff: float
    logg: float
    m_H: float

    @property
    def as_dict(self) -> Dict[str, float]:
        return {"Teff": self.Teff, "logg": self.logg, "m_H": self.m_H}


def _json_default(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable.")


def _dump_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def _fmt_token(x: float) -> str:
    s = f"{float(x):.4f}".rstrip("0").rstrip(".")
    if s == "-0":
        s = "0"
    return s.replace("-", "m").replace(".", "p")


def build_case_tag(case: Case, wl_min: float, wl_max: float) -> str:
    return (
        f"Teff{_fmt_token(case.Teff)}_g{_fmt_token(case.logg)}_mH{_fmt_token(case.m_H)}"
        f"_wl{_fmt_token(wl_min)}m{_fmt_token(wl_max)}"
    )


def normalize_absolute_abundances(a_x: np.ndarray) -> np.ndarray:
    abs_abund = np.power(10.0, np.asarray(a_x, dtype=np.float64) - 12.0)
    abs_abund = np.clip(abs_abund, 0.0, None)
    total = float(np.sum(abs_abund))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid abundance vector after A(X) conversion.")
    return abs_abund / total


def extract_layer_state(atm) -> Tuple[np.ndarray, np.ndarray]:
    if not hasattr(atm, "layers"):
        raise TypeError("Expected atmosphere object with .layers.")
    temperatures = np.asarray([layer.temp for layer in atm.layers], dtype=np.float64)
    n_totals = np.asarray([layer.number_density for layer in atm.layers], dtype=np.float64)
    return temperatures, n_totals


def _coerce_spectrum_table(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < 4:
        raise ValueError(f"Spectrum table must have >=4 columns, got {arr.shape}.")
    return arr[:, :4]


def _pairwise_rect_metrics(
    wl_a: np.ndarray,
    rect_a: np.ndarray,
    wl_b: np.ndarray,
    rect_b: np.ndarray,
) -> Dict[str, float]:
    wl_a = np.asarray(wl_a, dtype=np.float64)
    wl_b = np.asarray(wl_b, dtype=np.float64)
    rect_a = np.asarray(rect_a, dtype=np.float64)
    rect_b = np.asarray(rect_b, dtype=np.float64)

    order_a = np.argsort(wl_a)
    order_b = np.argsort(wl_b)
    wl_a = wl_a[order_a]
    rect_a = rect_a[order_a]
    wl_b = wl_b[order_b]
    rect_b = rect_b[order_b]

    lo = max(float(wl_a[0]), float(wl_b[0]))
    hi = min(float(wl_a[-1]), float(wl_b[-1]))
    if hi <= lo:
        raise RuntimeError("No overlapping wavelength domain between compared spectra.")

    mask = (wl_a >= lo) & (wl_a <= hi)
    wl_common = wl_a[mask]
    rect_a_common = rect_a[mask]
    if wl_common.size < 2:
        raise RuntimeError("Insufficient overlap points for pairwise metric calculation.")

    rect_b_interp = np.interp(wl_common, wl_b, rect_b)
    diff = np.abs(rect_a_common - rect_b_interp)
    return {
        "rect_mad": float(np.mean(diff)),
        "rect_p95_abs": float(np.percentile(diff, 95)),
        "rect_max_abs": float(np.max(diff)),
        "n_overlap_points": int(wl_common.size),
        "wavelength_alignment": "interpolate_to_reference_intersection",
        "overlap_wl_min": float(wl_common[0]),
        "overlap_wl_max": float(wl_common[-1]),
    }


def load_linelist():
    linelist_path = get_data_path("vald_extract_stellar_solar_threshold001.vald", must_exist=True)
    return read_linelist(str(linelist_path), format="vald")


def load_or_generate_korg_outputs(
    *,
    case: Case,
    wl_min: float,
    wl_max: float,
    korg_script_path: Path,
    julia_bin: str,
    auto_run_korg: bool,
    comparison_dirs: Iterable[Path],
) -> Dict[str, object]:
    tag = build_case_tag(case, wl_min, wl_max)
    rel_lines = f"korg_{tag}_spectrum_with_lines.txt"
    rel_cntm = f"korg_{tag}_spectrum_continuum.txt"

    for base in comparison_dirs:
        lines_path = base / rel_lines
        cntm_path = base / rel_cntm
        if lines_path.exists() and cntm_path.exists():
            k_lines = _coerce_spectrum_table(np.loadtxt(lines_path))
            k_cntm = _coerce_spectrum_table(np.loadtxt(cntm_path))
            return {
                "tag": tag,
                "with_lines_path": lines_path,
                "continuum_path": cntm_path,
                "lines": k_lines,
                "cntm": k_cntm,
                "generated": False,
            }

    if not auto_run_korg:
        raise FileNotFoundError(
            f"Missing Korg files for tag={tag} in {[str(p) for p in comparison_dirs]} and auto-run disabled."
        )

    if not korg_script_path.exists():
        raise FileNotFoundError(f"Korg script not found: {korg_script_path}")

    out_dir = next(iter(comparison_dirs))
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("JORG_DATA_DIR", str(REPO_ROOT / "data"))
    cmd = [
        julia_bin,
        f"--project={korg_script_path.parent}",
        str(korg_script_path),
        str(case.Teff),
        str(case.logg),
        str(case.m_H),
        str(wl_min),
        str(wl_max),
        tag,
        str(out_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise KorgSynthesisError(
            "Korg auto-run failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )

    lines_path = out_dir / rel_lines
    cntm_path = out_dir / rel_cntm
    if not (lines_path.exists() and cntm_path.exists()):
        raise KorgSynthesisError(
            "Korg run reported success but output files are missing.\n"
            f"expected: {lines_path}, {cntm_path}"
        )

    k_lines = _coerce_spectrum_table(np.loadtxt(lines_path))
    k_cntm = _coerce_spectrum_table(np.loadtxt(cntm_path))
    return {
        "tag": tag,
        "with_lines_path": lines_path,
        "continuum_path": cntm_path,
        "lines": k_lines,
        "cntm": k_cntm,
        "generated": True,
    }


def run_case_three_way(
    *,
    case: Case,
    wl_min: float,
    wl_max: float,
    linelist,
    pinn_solver,
    chem_data,
    ce_source_cache: Dict[Tuple[float, float, float], Dict[str, object]] | None,
    korg_script_path: Path,
    julia_bin: str,
    auto_run_korg: bool,
    comparison_dirs: Iterable[Path],
) -> Dict[str, object]:
    atm = interpolate_marcs(case.Teff, case.logg, case.m_H)
    a_x = create_korg_compatible_abundance_array(case.m_H)
    abs_abund = normalize_absolute_abundances(a_x)

    t0 = time.perf_counter()
    scipy_res = synthesize(
        atm,
        linelist,
        a_x,
        wavelengths=(wl_min, wl_max),
        hydrogen_lines=False,
        verbose=False,
        logg=case.logg,
        cntm_step=1.0,
    )
    scipy_synthesis_s = time.perf_counter() - t0

    temperatures, n_totals = extract_layer_state(atm)
    cache_key = (float(case.Teff), float(case.logg), float(case.m_H))
    pinn_ce_cache_hit = False

    t1 = time.perf_counter()
    if ce_source_cache is not None and cache_key in ce_source_cache:
        ce_source = ce_source_cache[cache_key]
        pinn_ce_cache_hit = True
    else:
        ce_source = build_atomic_ce_source_from_solver(
            solver=pinn_solver,
            temperatures=temperatures,
            n_totals=n_totals,
            abundances=abs_abund,
            chem_data=chem_data,
        )
        if ce_source_cache is not None:
            ce_source_cache[cache_key] = ce_source
    pinn_ce_build_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    pinn_res = synthesize(
        atm,
        linelist,
        a_x,
        wavelengths=(wl_min, wl_max),
        hydrogen_lines=False,
        verbose=False,
        logg=case.logg,
        cntm_step=1.0,
        use_chemical_equilibrium_from=ce_source,
    )
    pinn_synthesis_s = time.perf_counter() - t2
    pinn_total_s = pinn_ce_build_s + pinn_synthesis_s

    korg = load_or_generate_korg_outputs(
        case=case,
        wl_min=wl_min,
        wl_max=wl_max,
        korg_script_path=korg_script_path,
        julia_bin=julia_bin,
        auto_run_korg=auto_run_korg,
        comparison_dirs=comparison_dirs,
    )

    scipy_wl = np.asarray(scipy_res.wavelengths, dtype=np.float64)
    pinn_wl = np.asarray(pinn_res.wavelengths, dtype=np.float64)
    korg_wl = np.asarray(korg["lines"][:, 0], dtype=np.float64)

    scipy_rect = np.asarray(scipy_res.flux, dtype=np.float64) / np.asarray(scipy_res.cntm, dtype=np.float64)
    pinn_rect = np.asarray(pinn_res.flux, dtype=np.float64) / np.asarray(pinn_res.cntm, dtype=np.float64)
    korg_rect = np.asarray(korg["lines"][:, 3], dtype=np.float64)

    pairwise = {
        "scipy_vs_pinn": _pairwise_rect_metrics(scipy_wl, scipy_rect, pinn_wl, pinn_rect),
        "scipy_vs_korg": _pairwise_rect_metrics(scipy_wl, scipy_rect, korg_wl, korg_rect),
        "pinn_vs_korg": _pairwise_rect_metrics(pinn_wl, pinn_rect, korg_wl, korg_rect),
    }

    scipy_ne = np.asarray(scipy_res.electron_number_density, dtype=np.float64)
    pinn_ne = np.asarray(ce_source["electron_densities"], dtype=np.float64)
    ne_rel = np.abs(pinn_ne - scipy_ne) / np.maximum(np.abs(scipy_ne), 1e-30)
    ne_metrics = {
        "ne_rel_mean": float(np.mean(ne_rel)),
        "ne_rel_p95": float(np.percentile(ne_rel, 95)),
    }

    korg_flux_interp = np.interp(scipy_wl, korg_wl, np.asarray(korg["lines"][:, 1], dtype=np.float64))
    korg_cntm_interp = np.interp(scipy_wl, korg_wl, np.asarray(korg["lines"][:, 2], dtype=np.float64))
    korg_rect_interp = np.interp(scipy_wl, korg_wl, korg_rect)

    return {
        "case": case.as_dict,
        "wl_range": [float(wl_min), float(wl_max)],
        "timings": {
            "scipy_synthesis_s": float(scipy_synthesis_s),
            "pinn_ce_build_s": float(pinn_ce_build_s),
            "pinn_synthesis_s": float(pinn_synthesis_s),
            "pinn_total_s": float(pinn_total_s),
            "pinn_ce_cache_hit": bool(pinn_ce_cache_hit),
            "pinn_speedup_total_vs_scipy": float(scipy_synthesis_s / max(pinn_total_s, 1e-12)),
            "pinn_speedup_synthesis_vs_scipy": float(scipy_synthesis_s / max(pinn_synthesis_s, 1e-12)),
        },
        "pairwise_metrics": pairwise,
        "ne_metrics_pinn_vs_scipy": ne_metrics,
        "korg": {
            "tag": korg["tag"],
            "generated": bool(korg["generated"]),
            "with_lines_path": str(korg["with_lines_path"]),
            "continuum_path": str(korg["continuum_path"]),
        },
        "spectra": {
            "scipy_wavelength": scipy_wl,
            "scipy_flux": np.asarray(scipy_res.flux, dtype=np.float64),
            "scipy_cntm": np.asarray(scipy_res.cntm, dtype=np.float64),
            "scipy_rect": scipy_rect,
            "pinn_wavelength": pinn_wl,
            "pinn_flux": np.asarray(pinn_res.flux, dtype=np.float64),
            "pinn_cntm": np.asarray(pinn_res.cntm, dtype=np.float64),
            "pinn_rect": pinn_rect,
            "korg_wavelength_raw": korg_wl,
            "korg_flux_raw": np.asarray(korg["lines"][:, 1], dtype=np.float64),
            "korg_cntm_raw": np.asarray(korg["lines"][:, 2], dtype=np.float64),
            "korg_rect_raw": korg_rect,
            "korg_flux_interp_to_scipy": korg_flux_interp,
            "korg_cntm_interp_to_scipy": korg_cntm_interp,
            "korg_rect_interp_to_scipy": korg_rect_interp,
            "scipy_ne": scipy_ne,
            "pinn_ne": pinn_ne,
        },
    }


def write_grid_csv(path: Path, rows: List[Dict[str, object]]):
    fieldnames = [
        "case_id",
        "status",
        "error",
        "Teff",
        "logg",
        "m_H",
        "wl_min",
        "wl_max",
        "scipy_synthesis_s",
        "pinn_ce_build_s",
        "pinn_ce_cache_hit",
        "pinn_synthesis_s",
        "pinn_total_s",
        "pinn_speedup_total_vs_scipy",
        "pinn_speedup_synthesis_vs_scipy",
        "ne_rel_mean",
        "ne_rel_p95",
        "rect_mad_scipy_vs_pinn",
        "rect_p95_abs_scipy_vs_pinn",
        "rect_max_abs_scipy_vs_pinn",
        "rect_mad_scipy_vs_korg",
        "rect_p95_abs_scipy_vs_korg",
        "rect_max_abs_scipy_vs_korg",
        "rect_mad_pinn_vs_korg",
        "rect_p95_abs_pinn_vs_korg",
        "rect_max_abs_pinn_vs_korg",
        "korg_generated",
        "korg_tag",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def aggregate_grid(rows: List[Dict[str, object]]) -> Dict[str, object]:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    agg: Dict[str, object] = {
        "total_cases": len(rows),
        "ok_cases": len(ok_rows),
        "failed_cases": len(rows) - len(ok_rows),
    }
    if not ok_rows:
        return agg

    def _collect(key: str) -> np.ndarray:
        vals = [float(r[key]) for r in ok_rows if r.get(key, "") not in ("", None)]
        return np.asarray(vals, dtype=np.float64)

    for metric_key in [
        "pinn_speedup_total_vs_scipy",
        "pinn_speedup_synthesis_vs_scipy",
        "ne_rel_mean",
        "ne_rel_p95",
        "rect_mad_scipy_vs_pinn",
        "rect_mad_scipy_vs_korg",
        "rect_mad_pinn_vs_korg",
    ]:
        vals = _collect(metric_key)
        if vals.size:
            agg[metric_key] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p95": float(np.percentile(vals, 95)),
            }

    agg["cases_where_pinn_faster_total"] = [
        r["case_id"] for r in ok_rows if float(r["pinn_speedup_total_vs_scipy"]) > 1.0
    ]
    agg["cases_where_pinn_closer_to_korg_than_scipy"] = [
        r["case_id"]
        for r in ok_rows
        if float(r["rect_mad_pinn_vs_korg"]) < float(r["rect_mad_scipy_vs_korg"])
    ]
    return agg


def _resolve_range(args, default_range: Tuple[float, float]) -> Tuple[float, float]:
    if args.wl_min is None and args.wl_max is None:
        return default_range
    if args.wl_min is None or args.wl_max is None:
        raise ValueError("Both --wl-min and --wl-max must be provided together.")
    if args.wl_max <= args.wl_min:
        raise ValueError("--wl-max must be greater than --wl-min.")
    return float(args.wl_min), float(args.wl_max)


def _parse_extra_bands(text: str) -> List[Tuple[float, float]]:
    """Parse bands like: '5002:5004,5004:5006'."""
    if not text:
        return []
    bands: List[Tuple[float, float]] = []
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid band token '{token}'. Expected 'min:max'.")
        lo_s, hi_s = token.split(":", 1)
        lo = float(lo_s)
        hi = float(hi_s)
        if hi <= lo:
            raise ValueError(f"Invalid band token '{token}': max must be > min.")
        bands.append((lo, hi))
    return bands


def warmup_pinn_solver(pinn_solver) -> float:
    """Run one tiny forward pass to amortize initial JAX overhead."""
    a_x = create_korg_compatible_abundance_array(0.0)
    abs_abund = normalize_absolute_abundances(a_x)
    t0 = time.perf_counter()
    _ = pinn_solver.solve_batch(
        np.asarray([5772.0], dtype=np.float64),
        np.asarray([1e16], dtype=np.float64),
        np.asarray([abs_abund], dtype=np.float64),
    )
    return float(time.perf_counter() - t0)


def _case_from_dict(d: Dict[str, float]) -> Case:
    return Case(Teff=float(d["Teff"]), logg=float(d["logg"]), m_H=float(d["m_H"]))


def _single_row(case_id: str, result: Dict[str, object]) -> Dict[str, object]:
    pair = result["pairwise_metrics"]
    timings = result["timings"]
    ne_metrics = result["ne_metrics_pinn_vs_scipy"]
    case = result["case"]
    return {
        "case_id": case_id,
        "status": "ok",
        "error": "",
        "Teff": case["Teff"],
        "logg": case["logg"],
        "m_H": case["m_H"],
        "wl_min": result["wl_range"][0],
        "wl_max": result["wl_range"][1],
        "scipy_synthesis_s": timings["scipy_synthesis_s"],
        "pinn_ce_build_s": timings["pinn_ce_build_s"],
        "pinn_ce_cache_hit": timings.get("pinn_ce_cache_hit", False),
        "pinn_synthesis_s": timings["pinn_synthesis_s"],
        "pinn_total_s": timings["pinn_total_s"],
        "pinn_speedup_total_vs_scipy": timings["pinn_speedup_total_vs_scipy"],
        "pinn_speedup_synthesis_vs_scipy": timings["pinn_speedup_synthesis_vs_scipy"],
        "ne_rel_mean": ne_metrics["ne_rel_mean"],
        "ne_rel_p95": ne_metrics["ne_rel_p95"],
        "rect_mad_scipy_vs_pinn": pair["scipy_vs_pinn"]["rect_mad"],
        "rect_p95_abs_scipy_vs_pinn": pair["scipy_vs_pinn"]["rect_p95_abs"],
        "rect_max_abs_scipy_vs_pinn": pair["scipy_vs_pinn"]["rect_max_abs"],
        "rect_mad_scipy_vs_korg": pair["scipy_vs_korg"]["rect_mad"],
        "rect_p95_abs_scipy_vs_korg": pair["scipy_vs_korg"]["rect_p95_abs"],
        "rect_max_abs_scipy_vs_korg": pair["scipy_vs_korg"]["rect_max_abs"],
        "rect_mad_pinn_vs_korg": pair["pinn_vs_korg"]["rect_mad"],
        "rect_p95_abs_pinn_vs_korg": pair["pinn_vs_korg"]["rect_p95_abs"],
        "rect_max_abs_pinn_vs_korg": pair["pinn_vs_korg"]["rect_max_abs"],
        "korg_generated": result["korg"]["generated"],
        "korg_tag": result["korg"]["tag"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Three-way comparison: Jorg-Scipy CE vs Jorg-PINN CE vs Korg."
    )
    parser.add_argument("--mode", choices=("single", "grid", "all"), default="all")
    parser.add_argument("--pinn-model-path", default=str(DEFAULT_PINN_MODEL))
    parser.add_argument("--korg-script-path", default=str(DEFAULT_KORG_SCRIPT))
    parser.add_argument("--julia-bin", default=os.environ.get("JULIA_BIN", "julia"))
    parser.add_argument("--wl-min", type=float, default=None)
    parser.add_argument("--wl-max", type=float, default=None)
    parser.add_argument(
        "--single-extra-bands",
        default="",
        help="Optional extra single-case bands as 'min:max,min:max' for CE-reuse sweeps.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--no-pinn-warmup",
        action="store_true",
        help="Skip startup warmup forward pass for the PINN solver.",
    )
    parser.add_argument(
        "--auto-run-korg",
        dest="auto_run_korg",
        action="store_true",
        default=True,
        help="Auto-run Korg script when outputs are missing (default: true).",
    )
    parser.add_argument(
        "--no-auto-run-korg",
        dest="auto_run_korg",
        action="store_false",
        help="Do not run Korg automatically when outputs are missing.",
    )
    args = parser.parse_args()

    pinn_model_path = Path(args.pinn_model_path).expanduser().resolve()
    if not pinn_model_path.exists():
        raise FileNotFoundError(f"PINN checkpoint not found: {pinn_model_path}")

    korg_script_path = Path(args.korg_script_path).expanduser().resolve()
    comparison_dirs = [p.expanduser().resolve() for p in DEFAULT_COMPARISON_DIRS]

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root).expanduser().resolve()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] repo_root: {REPO_ROOT}")
    print(f"[info] run_dir: {run_dir}")
    print(f"[info] mode: {args.mode}")

    ionization_energies = create_default_ionization_energies()
    partition_funcs = create_default_partition_functions()
    chem_data = prepare_chem_eq_data(
        ionization_energies,
        partition_funcs,
        log_equilibrium_constants=None,
    )
    pinn_solver = load_pinn_solver_from_checkpoint(pinn_model_path, chem_data)
    warmup_s = 0.0
    if not args.no_pinn_warmup:
        warmup_s = warmup_pinn_solver(pinn_solver)

    ce_source_cache: Dict[Tuple[float, float, float], Dict[str, object]] = {}
    linelist = load_linelist()

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "config": {
            "pinn_model_path": str(pinn_model_path),
            "korg_script_path": str(korg_script_path),
            "julia_bin": args.julia_bin,
            "auto_run_korg": bool(args.auto_run_korg),
            "comparison_dirs": [str(p) for p in comparison_dirs],
            "pinn_warmup_enabled": not args.no_pinn_warmup,
            "pinn_warmup_s": warmup_s,
            "single_extra_bands": args.single_extra_bands,
        },
        "outputs": {},
    }

    if args.mode in ("single", "all"):
        wl_min, wl_max = _resolve_range(args, DEFAULT_SINGLE_RANGE)
        extra_bands = _parse_extra_bands(args.single_extra_bands)
        band_ranges = [(wl_min, wl_max)] + extra_bands
        single_case = _case_from_dict(SINGLE_CASE)
        print(
            f"[info] running single case: {single_case.as_dict}, "
            f"bands={[(round(a, 4), round(b, 4)) for a, b in band_ranges]}"
        )
        single_results: List[Dict[str, object]] = []
        for i, (band_min, band_max) in enumerate(band_ranges, start=1):
            print(f"[info] single band {i}/{len(band_ranges)}: wl=({band_min}, {band_max})")
            single_result = run_case_three_way(
                case=single_case,
                wl_min=band_min,
                wl_max=band_max,
                linelist=linelist,
                pinn_solver=pinn_solver,
                chem_data=chem_data,
                ce_source_cache=ce_source_cache,
                korg_script_path=korg_script_path,
                julia_bin=args.julia_bin,
                auto_run_korg=args.auto_run_korg,
                comparison_dirs=comparison_dirs,
            )
            single_results.append(single_result)

        single_result = single_results[0]
        single_metrics_path = run_dir / "single_case_metrics.json"
        _dump_json(single_metrics_path, {k: v for k, v in single_result.items() if k != "spectra"})
        spectra_path = run_dir / "single_case_spectra.npz"
        np.savez_compressed(spectra_path, **single_result["spectra"])
        manifest["outputs"]["single_case_metrics"] = str(single_metrics_path)
        manifest["outputs"]["single_case_spectra"] = str(spectra_path)
        print(f"[info] wrote: {single_metrics_path.name}, {spectra_path.name}")

        if len(single_results) > 1:
            multi_rows = []
            for idx, r in enumerate(single_results, start=1):
                row = _single_row(f"single_band_{idx}", r)
                row["pinn_ce_cache_hit"] = r["timings"].get("pinn_ce_cache_hit", False)
                multi_rows.append(row)
            total_scipy = float(sum(r["timings"]["scipy_synthesis_s"] for r in single_results))
            total_pinn = float(sum(r["timings"]["pinn_total_s"] for r in single_results))
            multi_payload = {
                "bands": [r["wl_range"] for r in single_results],
                "rows": multi_rows,
                "aggregate": {
                    "n_bands": len(single_results),
                    "total_scipy_s": total_scipy,
                    "total_pinn_s": total_pinn,
                    "overall_speedup_total_vs_scipy": float(total_scipy / max(total_pinn, 1e-12)),
                    "ce_cache_hits": int(sum(bool(r["timings"].get("pinn_ce_cache_hit")) for r in single_results)),
                },
            }
            multi_path = run_dir / "single_case_multiband_metrics.json"
            _dump_json(multi_path, multi_payload)
            manifest["outputs"]["single_case_multiband_metrics"] = str(multi_path)
            print(f"[info] wrote: {multi_path.name}")

    if args.mode in ("grid", "all"):
        wl_min, wl_max = _resolve_range(args, DEFAULT_GRID_RANGE)
        print(f"[info] running grid cases, wl=({wl_min}, {wl_max})")
        grid_rows: List[Dict[str, object]] = []
        for i, case_cfg in enumerate(GRID_CASES, start=1):
            case = _case_from_dict(case_cfg)
            case_id = f"case_{i}_{build_case_tag(case, wl_min, wl_max)}"
            print(f"[info] grid {i}/{len(GRID_CASES)}: {case.as_dict}")
            try:
                result = run_case_three_way(
                    case=case,
                    wl_min=wl_min,
                    wl_max=wl_max,
                    linelist=linelist,
                    pinn_solver=pinn_solver,
                    chem_data=chem_data,
                    ce_source_cache=ce_source_cache,
                    korg_script_path=korg_script_path,
                    julia_bin=args.julia_bin,
                    auto_run_korg=args.auto_run_korg,
                    comparison_dirs=comparison_dirs,
                )
            except KorgSynthesisError:
                raise
            except Exception as exc:
                grid_rows.append(
                    {
                        "case_id": case_id,
                        "status": "failed",
                        "error": str(exc),
                        "Teff": case.Teff,
                        "logg": case.logg,
                        "m_H": case.m_H,
                        "wl_min": wl_min,
                        "wl_max": wl_max,
                    }
                )
                continue

            grid_rows.append(_single_row(case_id, result))

        grid_csv_path = run_dir / "grid_case_metrics.csv"
        write_grid_csv(grid_csv_path, grid_rows)
        agg_path = run_dir / "grid_aggregate_metrics.json"
        _dump_json(agg_path, aggregate_grid(grid_rows))
        manifest["outputs"]["grid_case_metrics_csv"] = str(grid_csv_path)
        manifest["outputs"]["grid_aggregate_metrics"] = str(agg_path)
        print(f"[info] wrote: {grid_csv_path.name}, {agg_path.name}")

    manifest_path = run_dir / "manifest.json"
    _dump_json(manifest_path, manifest)
    print(f"[info] wrote: {manifest_path.name}")


if __name__ == "__main__":
    main()
