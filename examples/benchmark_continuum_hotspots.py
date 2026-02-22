#!/usr/bin/env python3
"""Continuum hotspot benchmark for PINN+Jorg vs Korg step timings."""

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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    raise RuntimeError("Could not locate repo root containing src/jorg.")


REPO_ROOT = _ensure_src_on_path()

from jorg.atmosphere import interpolate_marcs
from jorg.continuum.exact_physics_continuum import ContinuumTableCache
from jorg.data import get_data_path
from jorg.lines.linelist import read_linelist
from jorg.opacity.layer_processor import LayerProcessor
from jorg.radiative_transfer_exact import generate_mu_grid
from jorg.statmech import (
    create_default_ionization_energies,
    create_default_log_equilibrium_constants,
    create_default_partition_functions,
)
from jorg.statmech.chem_eq_jax import prepare_chem_eq_data
from jorg.statmech.chem_eq_pinn_inference import load_pinn_solver_from_checkpoint
from jorg.statmech.chem_eq_pinn_loss import compute_atomic_species_densities
from jorg.statmech.species import MAX_ATOMIC_NUMBER
from jorg.synthesis import (
    _calculate_line_opacity_multilayer,
    _calculate_radiative_transfer,
    create_korg_compatible_abundance_array,
)


DEFAULT_PINN_MODEL = (
    REPO_ROOT / "output" / "pinn" / "joint_phase2_stable_from_benchmark" / "chem_eq_pinn_model.npz"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "output" / "benchmarks" / "continuum_accel_baseline"
DEFAULT_KORG_STEP_SCRIPT = REPO_ROOT / "examples" / "tmp" / "korg_step_benchmark.jl"

DEFAULT_CASE_GRID = [
    {"Teff": 5771.0, "logg": 4.44, "m_H": 0.0},
    {"Teff": 4250.0, "logg": 1.40, "m_H": -0.5},
    {"Teff": 4500.0, "logg": 1.50, "m_H": -2.0},
    {"Teff": 5000.0, "logg": 4.00, "m_H": -0.5},
    {"Teff": 6000.0, "logg": 4.50, "m_H": 0.2},
]
DEFAULT_PIXEL_LIST = [200, 1000, 2001]
DEFAULT_SPEC_LIST = [1, 3]

PARTITION_FUNCS = create_default_partition_functions()
IONIZATION_ENERGIES = create_default_ionization_energies()
LOG_EQUILIBRIUM_CONSTANTS = create_default_log_equilibrium_constants()
MU_VALUES, _MU_WEIGHTS = generate_mu_grid(20)
MU_GRID = list(np.asarray(MU_VALUES))


@dataclass(frozen=True)
class Case:
    Teff: float
    logg: float
    m_H: float


class TimedLayerProcessor(LayerProcessor):
    """LayerProcessor with per-component timing counters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing = {
            "chem_eq_s": 0.0,
            "continuum_s": 0.0,
            "line_s": 0.0,
        }

    def _calculate_chemical_equilibrium(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super()._calculate_chemical_equilibrium(*args, **kwargs)
        self.timing["chem_eq_s"] += time.perf_counter() - t0
        return out

    def _calculate_continuum_opacity(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super()._calculate_continuum_opacity(*args, **kwargs)
        self.timing["continuum_s"] += time.perf_counter() - t0
        return out

    def _calculate_continuum_opacity_batch(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super()._calculate_continuum_opacity_batch(*args, **kwargs)
        self.timing["continuum_s"] += time.perf_counter() - t0
        return out

    def _calculate_line_opacity(self, *args, **kwargs):
        t0 = time.perf_counter()
        out = super()._calculate_line_opacity(*args, **kwargs)
        self.timing["line_s"] += time.perf_counter() - t0
        return out


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)!r}")


def _dump_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default))


def parse_int_list(text: str) -> List[int]:
    return [int(tok.strip()) for tok in text.split(",") if tok.strip()]


def build_cases(n: int) -> List[Case]:
    out = []
    for i in range(n):
        row = DEFAULT_CASE_GRID[i % len(DEFAULT_CASE_GRID)]
        out.append(Case(**row))
    return out


def normalize_abundances(a_x: np.ndarray) -> np.ndarray:
    abs_abund = 10.0 ** (np.asarray(a_x, dtype=np.float64) - 12.0)
    abs_abund = np.clip(abs_abund, 0.0, None)
    total = float(np.sum(abs_abund))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Invalid abundance vector.")
    return abs_abund / total


def atmosphere_to_dict(atm) -> Dict[str, np.ndarray]:
    if hasattr(atm, "layers"):
        out = {
            "temperature": np.array([layer.temp for layer in atm.layers], dtype=np.float64),
            "electron_density": np.array([layer.electron_number_density for layer in atm.layers], dtype=np.float64),
            "number_density": np.array([layer.number_density for layer in atm.layers], dtype=np.float64),
            "tau_5000": np.array([layer.tau_5000 for layer in atm.layers], dtype=np.float64),
            "height": np.array([layer.z for layer in atm.layers], dtype=np.float64),
        }
        out["pressure"] = out["number_density"] * 1.380649e-16 * out["temperature"]
        return out
    return atm


def load_linelist():
    path = get_data_path("vald_extract_stellar_solar_threshold001.vald", must_exist=True)
    return read_linelist(str(path), format="vald")


def measure_pinn_ce_source_breakdown(
    case: Case,
    model_path: Path,
) -> Dict[str, float]:
    atm = interpolate_marcs(case.Teff, case.logg, case.m_H)
    temps = np.array([layer.temp for layer in atm.layers], dtype=np.float64)
    n_totals = np.array([layer.number_density for layer in atm.layers], dtype=np.float64)

    a_x = create_korg_compatible_abundance_array(case.m_H)
    abs_abund = normalize_abundances(a_x)
    abund_layers = np.broadcast_to(abs_abund[None, :], (len(temps), MAX_ATOMIC_NUMBER)).copy()

    chem_data = prepare_chem_eq_data(IONIZATION_ENERGIES, PARTITION_FUNCS, LOG_EQUILIBRIUM_CONSTANTS)
    solver = load_pinn_solver_from_checkpoint(model_path=model_path, chem_data=chem_data)

    t0 = time.perf_counter()
    ne_pred, neutral_fractions = solver.solve_batch(temps, n_totals, abund_layers)
    nn_forward_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    _, n0, n1, n2 = compute_atomic_species_densities(
        neutral_fractions,
        ne_pred,
        temps,
        n_totals,
        abund_layers,
        chem_data,
    )
    saha_density_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    number_densities = {}
    from jorg.statmech.species import Species

    for z in range(1, MAX_ATOMIC_NUMBER + 1):
        idx = z - 1
        number_densities[Species.from_atomic_number(z, 0)] = np.asarray(n0[:, idx], dtype=np.float64)
        number_densities[Species.from_atomic_number(z, 1)] = np.asarray(n1[:, idx], dtype=np.float64)
        number_densities[Species.from_atomic_number(z, 2)] = np.asarray(n2[:, idx], dtype=np.float64)
    dict_pack_s = time.perf_counter() - t2

    total = nn_forward_s + saha_density_s + dict_pack_s
    return {
        "n_layers": int(len(temps)),
        "nn_forward_s": float(nn_forward_s),
        "saha_and_density_build_s": float(saha_density_s),
        "dict_pack_s": float(dict_pack_s),
        "total_build_s": float(total),
        "model_path": str(model_path),
    }


def run_jorg_single_case(
    case: Case,
    wl_array: np.ndarray,
    linelist,
    continuum_backend: str,
    enable_batch_continuum: bool = False,
) -> Dict[str, float]:
    t0 = time.perf_counter()
    atm = interpolate_marcs(case.Teff, case.logg, case.m_H)
    t_atm = time.perf_counter() - t0
    atm_dict = atmosphere_to_dict(atm)

    a_x = create_korg_compatible_abundance_array(case.m_H)
    abs_abund = normalize_abundances(a_x)

    cache = ContinuumTableCache(partition_funcs=PARTITION_FUNCS) if continuum_backend == "jax_fast" else None
    layer_processor = TimedLayerProcessor(
        ionization_energies=IONIZATION_ENERGIES,
        partition_funcs=PARTITION_FUNCS,
        log_equilibrium_constants=LOG_EQUILIBRIUM_CONSTANTS,
        line_cutoff_threshold=3e-4,
        verbose=False,
        warn_on_ne_discrepancy=False,
        print_ne_comparison=False,
        continuum_backend=continuum_backend,
        continuum_cache=cache,
        enable_batch_continuum=enable_batch_continuum,
    )

    t1 = time.perf_counter()
    alpha_continuum, all_number_densities, all_electron_densities = layer_processor.process_all_layers(
        atm=atm_dict,
        abs_abundances=abs_abund,
        wl_array=wl_array,
        linelist=None,
        line_buffer=10.0,
        hydrogen_lines=False,
        vmic=1.0,
        use_chemical_equilibrium_from=None,
        log_g=case.logg,
        cntm_step=1.0,
    )
    t_ce_cntm = time.perf_counter() - t1

    t2 = time.perf_counter()
    line_opacity = _calculate_line_opacity_multilayer(
        wl_array=wl_array,
        temps=np.asarray(atm_dict["temperature"], dtype=np.float64),
        electron_densities=all_electron_densities,
        number_densities=all_number_densities,
        partition_funcs=PARTITION_FUNCS,
        linelist=linelist,
        line_buffer=10.0,
        microturbulence_kms=1.0,
        continuum_opacity=alpha_continuum,
        cutoff_threshold=3e-4,
        verbose=False,
    )
    t_line = time.perf_counter() - t2
    alpha_total = alpha_continuum + line_opacity

    t3 = time.perf_counter()
    _calculate_radiative_transfer(
        alpha_total,
        atm_dict,
        wl_array,
        MU_GRID,
        "linear_flux_only",
        True,
        a_x,
        layer_processor,
        linelist,
        10.0,
        False,
        1.0,
        abs_abund,
        None,
        case.logg,
        False,
        rt_method="korg_default",
        verbose=False,
        alpha_continuum=alpha_continuum,
        line_cutoff_threshold=3e-4,
    )
    t_rt = time.perf_counter() - t3

    total = t_atm + t_ce_cntm + t_line + t_rt
    return {
        "atmosphere": float(t_atm),
        "chem_eq_continuum": float(t_ce_cntm),
        "line_opacity": float(t_line),
        "radiative_transfer": float(t_rt),
        "total": float(total),
        "chem_eq_only": float(layer_processor.timing["chem_eq_s"]),
        "continuum_only": float(layer_processor.timing["continuum_s"]),
        "line_only_internal": float(layer_processor.timing["line_s"]),
    }


def run_jorg_step_aggregates(
    pixel_list: Sequence[int],
    spec_list: Sequence[int],
    wl_min: float,
    wl_max: float,
    linelist,
    continuum_backend: str,
    enable_batch_continuum: bool = False,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for n_pix in pixel_list:
        wl_array = np.linspace(wl_min, wl_max, int(n_pix), dtype=np.float64)
        for n_spec in spec_list:
            cases = build_cases(int(n_spec))
            compile_ref = cases[0]
            t_compile = time.perf_counter()
            _ = run_jorg_single_case(
                compile_ref,
                wl_array,
                linelist,
                continuum_backend=continuum_backend,
                enable_batch_continuum=enable_batch_continuum,
            )
            compile_time_s = time.perf_counter() - t_compile

            totals = {
                "atmosphere": 0.0,
                "chem_eq_continuum": 0.0,
                "line_opacity": 0.0,
                "radiative_transfer": 0.0,
                "total": 0.0,
                "chem_eq_only": 0.0,
                "continuum_only": 0.0,
                "line_only_internal": 0.0,
            }
            for case in cases:
                out = run_jorg_single_case(
                    case,
                    wl_array,
                    linelist,
                    continuum_backend=continuum_backend,
                    enable_batch_continuum=enable_batch_continuum,
                )
                for k in totals:
                    totals[k] += out[k]
            pix_total = int(n_pix) * int(n_spec)
            rows.append(
                {
                    "n_pix": int(n_pix),
                    "n_spec": int(n_spec),
                    "pix_total": pix_total,
                    "compile_time_s": float(compile_time_s),
                    "steady_state_s": float(totals["total"]),
                    "J_atm": totals["atmosphere"],
                    "J_ce": totals["chem_eq_continuum"],
                    "J_line": totals["line_opacity"],
                    "J_rt": totals["radiative_transfer"],
                    "J_tot": totals["total"],
                    "J_ce_only": totals["chem_eq_only"],
                    "J_cntm_only": totals["continuum_only"],
                }
            )
    return rows


def parse_korg_step_output(stdout_text: str) -> Dict[Tuple[int, int], Dict[str, float]]:
    rows: Dict[Tuple[int, int], Dict[str, float]] = {}
    for raw in stdout_text.splitlines():
        line = raw.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        # Expected columns:
        # n_pix n_spec pix_total atm ce+cntm line rt total [pix/s]
        if len(parts) < 8:
            continue
        n_pix = int(parts[0])
        n_spec = int(parts[1])
        rows[(n_pix, n_spec)] = {
            "K_atm": float(parts[3]),
            "K_ce": float(parts[4]),
            "K_line": float(parts[5]),
            "K_rt": float(parts[6]),
            "K_tot": float(parts[7]),
        }
    return rows


def run_korg_step_script(
    julia_bin: str,
    korg_script_path: Path,
    julia_project: Optional[Path] = None,
) -> Optional[Dict[Tuple[int, int], Dict[str, float]]]:
    if not korg_script_path.exists():
        return None
    cmd = [julia_bin]

    if julia_project is not None:
        cmd.append(f"--project={julia_project}")
    else:
        inferred_project = korg_script_path.parent
        if (inferred_project / "Project.toml").exists():
            cmd.append(f"--project={inferred_project}")

    cmd.append(str(korg_script_path))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "Korg step benchmark failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    rows = parse_korg_step_output(proc.stdout)
    if not rows:
        preview = "\n".join(proc.stdout.splitlines()[:40])
        raise RuntimeError(
            "Korg step benchmark completed but no parseable timing rows were found.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout (first lines):\n{preview}\n"
        )
    return rows


def write_summary_csv(path: Path, rows: Iterable[Dict[str, float]]):
    fieldnames = [
        "n_pix",
        "n_spec",
        "pix_total",
        "compile_time_s",
        "steady_state_s",
        "J_atm",
        "K_atm",
        "J_ce",
        "K_ce",
        "J_line",
        "K_line",
        "J_rt",
        "K_rt",
        "J_tot",
        "K_tot",
        "J_ce_only",
        "J_cntm_only",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Continuum hotspot baseline benchmark.")
    parser.add_argument("--pinn-model-path", type=Path, default=DEFAULT_PINN_MODEL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--wl-min", type=float, default=5000.0)
    parser.add_argument("--wl-max", type=float, default=5020.0)
    parser.add_argument("--pixel-list", type=str, default="200,1000,2001")
    parser.add_argument("--spec-list", type=str, default="1,3")
    parser.add_argument("--julia-bin", type=str, default="julia")
    parser.add_argument("--korg-step-script", type=Path, default=DEFAULT_KORG_STEP_SCRIPT)
    parser.add_argument(
        "--julia-project",
        type=Path,
        default=None,
        help="Optional Julia project path for Korg step script. If omitted, auto-detect from script dir.",
    )
    parser.add_argument("--skip-korg", action="store_true")
    parser.add_argument("--continuum-backend", choices=["legacy", "jax_fast"], default="jax_fast")
    parser.add_argument("--enable-batch-continuum", action="store_true")
    args = parser.parse_args()

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_list = parse_int_list(args.pixel_list)
    spec_list = parse_int_list(args.spec_list)
    linelist = load_linelist()

    solar_case = Case(Teff=5772.0, logg=4.44, m_H=0.0)
    pinn_breakdown = measure_pinn_ce_source_breakdown(solar_case, args.pinn_model_path)
    _dump_json(out_dir / "pinn_ce_source_breakdown.json", pinn_breakdown)

    jorg_rows = run_jorg_step_aggregates(
        pixel_list=pixel_list,
        spec_list=spec_list,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        linelist=linelist,
        continuum_backend=args.continuum_backend,
        enable_batch_continuum=args.enable_batch_continuum,
    )

    korg_map = None
    if not args.skip_korg:
        korg_map = run_korg_step_script(
            args.julia_bin,
            args.korg_step_script,
            julia_project=args.julia_project,
        )

    merged_rows = []
    missing_korg_keys: List[Tuple[int, int]] = []
    for row in jorg_rows:
        key = (int(row["n_pix"]), int(row["n_spec"]))
        kvals = (korg_map or {}).get(key, {})
        if (not args.skip_korg) and not kvals:
            missing_korg_keys.append(key)
        merged = dict(row)
        merged["K_atm"] = float(kvals.get("K_atm", np.nan))
        merged["K_ce"] = float(kvals.get("K_ce", np.nan))
        merged["K_line"] = float(kvals.get("K_line", np.nan))
        merged["K_rt"] = float(kvals.get("K_rt", np.nan))
        merged["K_tot"] = float(kvals.get("K_tot", np.nan))
        merged_rows.append(merged)

    if (not args.skip_korg) and missing_korg_keys:
        raise RuntimeError(
            "Korg step output is missing requested (n_pix, n_spec) rows: "
            f"{missing_korg_keys}. "
            "Update korg_step_benchmark.jl grid or rerun with --skip-korg."
        )

    csv_path = out_dir / "jorg_vs_korg_per_step_timing_table.csv"
    write_summary_csv(csv_path, merged_rows)

    layer_breakdown = {}
    for row in merged_rows:
        key = f"n_pix={row['n_pix']},n_spec={row['n_spec']}"
        layer_breakdown[key] = {
            "J_ce_plus_cntm_s": float(row["J_ce"]),
            "J_ce_only_s": float(row["J_ce_only"]),
            "J_cntm_only_s": float(row["J_cntm_only"]),
            "J_line_s": float(row["J_line"]),
            "J_rt_s": float(row["J_rt"]),
            "J_total_s": float(row["J_tot"]),
            "compile_time_s": float(row["compile_time_s"]),
            "steady_state_s": float(row["steady_state_s"]),
            "K_total_s": float(row["K_tot"]) if np.isfinite(row["K_tot"]) else None,
        }
    _dump_json(out_dir / "layer_processor_breakdown.json", layer_breakdown)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "continuum_backend": args.continuum_backend,
        "enable_batch_continuum": bool(args.enable_batch_continuum),
        "wl_min": float(args.wl_min),
        "wl_max": float(args.wl_max),
        "pixel_list": pixel_list,
        "spec_list": spec_list,
        "pinn_model_path": str(args.pinn_model_path),
        "korg_step_script": str(args.korg_step_script),
        "julia_project": str(args.julia_project) if args.julia_project is not None else None,
        "korg_enabled": bool(not args.skip_korg),
        "files": {
            "pinn_ce_source_breakdown": str(out_dir / "pinn_ce_source_breakdown.json"),
            "layer_processor_breakdown": str(out_dir / "layer_processor_breakdown.json"),
            "jorg_vs_korg_per_step_timing_table": str(csv_path),
        },
    }
    _dump_json(out_dir / "manifest.json", manifest)

    print(f"Run directory: {out_dir}")
    print(f"Saved: {out_dir / 'pinn_ce_source_breakdown.json'}")
    print(f"Saved: {out_dir / 'layer_processor_breakdown.json'}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
