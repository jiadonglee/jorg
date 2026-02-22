#!/usr/bin/env python3
import argparse
import io
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np


def _ensure_src_on_path() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        src_path = parent / "src"
        if (src_path / "jorg").exists():
            sys.path.insert(0, str(src_path))
            data_dir = parent / "data"
            if data_dir.exists():
                os.environ.setdefault("JORG_DATA_DIR", str(data_dir))
            return parent
    raise RuntimeError("Could not locate src/jorg relative to this script")


REPO_ROOT = _ensure_src_on_path()

from jorg.atmosphere import interpolate_marcs
from jorg.lines.linelist_data import get_VALD_solar_linelist
from jorg.opacity.layer_processor import LayerProcessor
from jorg.radiative_transfer_exact import generate_mu_grid
from jorg.statmech import (
    create_default_ionization_energies,
    create_default_log_equilibrium_constants,
    create_default_partition_functions,
)
from jorg.synthesis import (
    _calculate_line_opacity_multilayer,
    _calculate_radiative_transfer,
    create_korg_compatible_abundance_array,
)


PARAM_GRID = [
    {"Teff": 5771.0, "logg": 4.44, "m_H": 0.0},
    {"Teff": 4250.0, "logg": 1.40, "m_H": -0.5},
    {"Teff": 4500.0, "logg": 1.50, "m_H": -2.5},
    {"Teff": 5000.0, "logg": 4.00, "m_H": -0.5},
    {"Teff": 6000.0, "logg": 4.50, "m_H": 0.2},
]


# Shared atomic data (avoid recomputing for each call)
PARTITION_FUNCS = create_default_partition_functions()
IONIZATION_ENERGIES = create_default_ionization_energies()
LOG_EQUILIBRIUM_CONSTANTS = create_default_log_equilibrium_constants()
MU_VALUES, MU_WEIGHTS = generate_mu_grid(20)
MU_GRID = list(np.array(MU_VALUES))


def build_spectra_params(n):
    params = []
    for i in range(n):
        params.append(PARAM_GRID[i % len(PARAM_GRID)])
    return params


def build_atm_dict(atm):
    if hasattr(atm, "layers"):
        atm_dict = {
            "temperature": np.array([layer.temp for layer in atm.layers]),
            "electron_density": np.array([layer.electron_number_density for layer in atm.layers]),
            "number_density": np.array([layer.number_density for layer in atm.layers]),
            "tau_5000": np.array([layer.tau_5000 for layer in atm.layers]),
            "height": np.array([layer.z for layer in atm.layers]),
        }
        atm_dict["pressure"] = atm_dict["number_density"] * 1.380649e-16 * atm_dict["temperature"]
        return atm_dict
    return atm


def run_jorg_pipeline(
    Teff,
    logg,
    m_H,
    wl_array,
    linelist,
    hydrogen_lines=False,
    cntm_step=1.0,
    vmic=1.0,
    line_buffer=10.0,
    rt_method="korg_default",
    compute_continuum=False,
    collect_ce_stats=False,
    use_prev_ne_initial=False,
):
    timings = {}

    t0 = time.perf_counter()
    atm = interpolate_marcs(Teff, logg, m_H)
    timings["atmosphere"] = time.perf_counter() - t0

    atm_dict = build_atm_dict(atm)
    A_X = create_korg_compatible_abundance_array(m_H)

    abs_abundances = 10 ** (A_X - 12)
    abs_abundances = abs_abundances / np.sum(abs_abundances)

    layer_processor = LayerProcessor(
        ionization_energies=IONIZATION_ENERGIES,
        partition_funcs=PARTITION_FUNCS,
        log_equilibrium_constants=LOG_EQUILIBRIUM_CONSTANTS,
        line_cutoff_threshold=3e-4,
        verbose=False,
        warn_on_ne_discrepancy=False,
        print_ne_comparison=False,
        collect_ce_stats=collect_ce_stats,
        use_prev_ne_initial=use_prev_ne_initial,
    )

    stage1_kwargs = dict(
        atm=atm_dict,
        abs_abundances=abs_abundances,
        wl_array=wl_array,
        linelist=None,
        line_buffer=line_buffer,
        hydrogen_lines=hydrogen_lines,
        vmic=vmic,
        log_g=logg,
        cntm_step=cntm_step,
    )

    t1 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        alpha_continuum, all_number_densities, all_electron_densities = layer_processor.process_all_layers(
            **stage1_kwargs,
            use_chemical_equilibrium_from=None,
        )
    timings["chem_eq_continuum"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    if linelist is None:
        line_opacity = np.zeros_like(alpha_continuum)
    else:
        line_opacity = _calculate_line_opacity_multilayer(
            wl_array=wl_array,
            temps=np.array(atm_dict["temperature"]),
            electron_densities=all_electron_densities,
            number_densities=all_number_densities,
            partition_funcs=PARTITION_FUNCS,
            linelist=linelist,
            line_buffer=line_buffer,
            microturbulence_kms=vmic,
            continuum_opacity=alpha_continuum,
            cutoff_threshold=3e-4,
            verbose=False,
        )
    timings["line_opacity"] = time.perf_counter() - t2

    alpha_total = alpha_continuum + line_opacity

    t3 = time.perf_counter()
    flux, continuum, intensity, source = _calculate_radiative_transfer(
        alpha_total,
        atm_dict,
        wl_array,
        MU_GRID,
        "linear_flux_only",
        True,
        A_X,
        layer_processor,
        linelist,
        line_buffer,
        hydrogen_lines,
        vmic,
        abs_abundances,
        None,
        logg,
        False,
        rt_method=rt_method,
        verbose=False,
        alpha_continuum=alpha_continuum,
        line_cutoff_threshold=3e-4,
    )
    timings["radiative_transfer"] = time.perf_counter() - t3

    continuum_flux = None
    continuum_cont = None
    if compute_continuum:
        t4 = time.perf_counter()
        continuum_flux, continuum_cont, _, _ = _calculate_radiative_transfer(
            alpha_continuum,
            atm_dict,
            wl_array,
            MU_GRID,
            "linear_flux_only",
            True,
            A_X,
            layer_processor,
            [],
            line_buffer,
            hydrogen_lines,
            vmic,
            abs_abundances,
            None,
            logg,
            False,
            rt_method=rt_method,
            verbose=False,
            alpha_continuum=alpha_continuum,
            line_cutoff_threshold=3e-4,
        )
        timings["radiative_transfer_continuum"] = time.perf_counter() - t4

    return {
        "atm": atm,
        "atm_dict": atm_dict,
        "alpha_continuum": alpha_continuum,
        "line_opacity": line_opacity,
        "alpha_total": alpha_total,
        "flux": flux,
        "continuum": continuum,
        "continuum_flux": continuum_flux,
        "continuum_cont": continuum_cont,
        "timings": timings,
        "ce_stats": layer_processor.stats if collect_ce_stats else None,
    }


def run_steps(
    wl_array,
    linelist,
    n_spec,
    hydrogen_lines=False,
    cntm_step=1.0,
    vmic=1.0,
    line_buffer=10.0,
    rt_method="korg_default",
    warmup=True,
    collect_ce_stats=False,
    use_prev_ne_initial=False,
):
    params_list = build_spectra_params(n_spec)
    compile_ref = params_list[0]

    # Cold run metric: includes JAX compile + first execution.
    t_compile = time.perf_counter()
    _ = run_jorg_pipeline(
        compile_ref["Teff"],
        compile_ref["logg"],
        compile_ref["m_H"],
        wl_array,
        linelist,
        hydrogen_lines=hydrogen_lines,
        cntm_step=cntm_step,
        vmic=vmic,
        line_buffer=line_buffer,
        rt_method=rt_method,
        compute_continuum=False,
        collect_ce_stats=False,
        use_prev_ne_initial=use_prev_ne_initial,
    )
    compile_time_s = time.perf_counter() - t_compile

    if warmup:
        wl_warm = np.linspace(wl_array[0], wl_array[-1], 200)
        _ = run_jorg_pipeline(
            PARAM_GRID[0]["Teff"],
            PARAM_GRID[0]["logg"],
            PARAM_GRID[0]["m_H"],
            wl_warm,
            linelist,
            hydrogen_lines=hydrogen_lines,
            cntm_step=cntm_step,
            compute_continuum=False,
            collect_ce_stats=False,
            use_prev_ne_initial=use_prev_ne_initial,
        )

    totals = {
        "atmosphere": 0.0,
        "chem_eq_continuum": 0.0,
        "line_opacity": 0.0,
        "radiative_transfer": 0.0,
    }
    per_spec = []
    ce_stats = None
    for p in params_list:
        res = run_jorg_pipeline(
            p["Teff"],
            p["logg"],
            p["m_H"],
            wl_array,
            linelist,
            hydrogen_lines=hydrogen_lines,
            cntm_step=cntm_step,
            vmic=vmic,
            line_buffer=line_buffer,
            rt_method=rt_method,
            compute_continuum=False,
            collect_ce_stats=collect_ce_stats,
            use_prev_ne_initial=use_prev_ne_initial,
        )
        per_spec.append(res["timings"])
        if collect_ce_stats:
            if ce_stats is None:
                ce_stats = dict(res.get("ce_stats") or {})
            else:
                for key, val in (res.get("ce_stats") or {}).items():
                    if isinstance(val, (int, float)):
                        ce_stats[key] = ce_stats.get(key, 0) + val
        for key in totals:
            totals[key] += res["timings"][key]

    total_time = sum(totals.values())
    pixels_total = len(wl_array) * n_spec
    result = {
        "n_pix": len(wl_array),
        "n_spec": n_spec,
        "pixels_total": pixels_total,
        "compile_time_s": float(compile_time_s),
        "steady_state_s": float(total_time),
        **totals,
        "total": total_time,
        "pixels_per_sec": pixels_total / max(total_time, 1e-12),
        "per_spec_timings": per_spec,
    }
    if collect_ce_stats and ce_stats:
        result["ce_stats"] = ce_stats
    return result


def run_profile(
    wl_array,
    linelist,
    profile_params,
    hydrogen_lines=False,
    cntm_step=1.0,
    vmic=1.0,
    line_buffer=10.0,
    rt_method="korg_default",
    warmup=True,
    use_prev_ne_initial=False,
):
    import cProfile
    import pstats
    if warmup:
        wl_warm = np.linspace(wl_array[0], wl_array[-1], 200)
        _ = run_jorg_pipeline(
            profile_params["Teff"],
            profile_params["logg"],
            profile_params["m_H"],
            wl_warm,
            linelist,
            hydrogen_lines=hydrogen_lines,
            cntm_step=cntm_step,
            compute_continuum=False,
            use_prev_ne_initial=use_prev_ne_initial,
        )

    prof = cProfile.Profile()
    prof.enable()
    _ = run_jorg_pipeline(
        profile_params["Teff"],
        profile_params["logg"],
        profile_params["m_H"],
        wl_array,
        linelist,
        hydrogen_lines=hydrogen_lines,
        cntm_step=cntm_step,
        vmic=vmic,
        line_buffer=line_buffer,
        rt_method=rt_method,
        compute_continuum=False,
        use_prev_ne_initial=use_prev_ne_initial,
    )
    prof.disable()

    s = io.StringIO()
    ps = pstats.Stats(prof, stream=s).strip_dirs().sort_stats("cumtime")
    ps.print_stats(30)
    return s.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="Standalone 5.5 Jorg profiling + step timing (matches jorg_tutorial.ipynb)."
    )
    parser.add_argument("--n-pix", type=int, default=1000, help="Number of wavelength points")
    parser.add_argument("--wl-min", type=float, default=5000.0, help="Minimum wavelength")
    parser.add_argument("--wl-max", type=float, default=5020.0, help="Maximum wavelength")
    parser.add_argument("--n-spec", type=int, default=1, help="Number of spectra to run")
    parser.add_argument("--profile", action="store_true", help="Run cProfile and print top 30")
    parser.add_argument("--steps", action="store_true", help="Run per-step timing")
    parser.add_argument(
        "--ce-stats",
        action="store_true",
        help="Collect and print chemical equilibrium solver stats",
    )
    parser.add_argument(
        "--ce-stats-out",
        nargs="?",
        const="__default__",
        default=None,
        help="Optional path to save CE stats JSON (default: output/benchmarks/jorg_ce_stats.json)",
    )
    parser.add_argument(
        "--ce-initial-prev",
        action="store_true",
        help="Use previous-layer ne as initial guess for CE solver (experimental)",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable warmup run for per-step timing",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save JSON results",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Save JSON to output/benchmarks/jorg_baseline.json",
    )

    args = parser.parse_args()

    if not args.profile and not args.steps:
        args.steps = True
    if (args.ce_stats or args.ce_stats_out is not None) and not args.steps:
        args.steps = True

    try:
        import jax

        backend = jax.default_backend()
    except Exception:
        backend = "unknown"

    linelist = get_VALD_solar_linelist()
    wl_array = np.linspace(args.wl_min, args.wl_max, args.n_pix)

    print("")
    print("Jorg 5.5 profiling / step timing")
    print(f"repo_root: {REPO_ROOT}")
    print(f"jax.default_backend(): {backend}")
    print(f"wl_range: ({args.wl_min}, {args.wl_max})")
    print(f"n_pix: {args.n_pix}, n_spec: {args.n_spec}")

    results = {
        "repo_root": str(REPO_ROOT),
        "backend": backend,
        "wl_range": [args.wl_min, args.wl_max],
        "n_pix": args.n_pix,
        "n_spec": args.n_spec,
    }

    if args.steps:
        step_result = run_steps(
            wl_array,
            linelist,
            args.n_spec,
            warmup=not args.no_warmup,
            collect_ce_stats=bool(args.ce_stats or args.ce_stats_out is not None),
            use_prev_ne_initial=args.ce_initial_prev,
        )
        results["steps"] = step_result

        print("")
        print("Jorg per-step timing (seconds):")
        print(
            f"{'n_pix':>6} {'n_spec':>6} {'pix_total':>10} "
            f"{'atm':>8} {'ce+cntm':>10} {'line':>8} {'rt':>8} {'total':>9} {'pix/s':>10}"
        )
        print("-" * 85)
        print(
            f"{step_result['n_pix']:>6} {step_result['n_spec']:>6} {step_result['pixels_total']:>10} "
            f"{step_result['atmosphere']:>8.2f} {step_result['chem_eq_continuum']:>10.2f} "
            f"{step_result['line_opacity']:>8.2f} {step_result['radiative_transfer']:>8.2f} "
            f"{step_result['total']:>9.2f} {step_result['pixels_per_sec']:>10.0f}"
        )
        if args.ce_stats and "ce_stats" in step_result:
            ce = step_result["ce_stats"]
            ce_calls = int(ce.get("ce_calls", 0) or 0)
            ce_successes = int(ce.get("ce_successes", 0) or 0)
            ce_nfev_total = int(ce.get("ce_nfev_total", 0) or 0)
            ce_njev_total = int(ce.get("ce_njev_total", 0) or 0)
            ce_attempts_total = int(ce.get("ce_attempts_total", 0) or 0)
            avg_nfev = ce_nfev_total / max(ce_calls, 1)
            avg_njev = ce_njev_total / max(ce_calls, 1)
            avg_attempts = ce_attempts_total / max(ce_calls, 1)
            success_rate = ce_successes / max(ce_calls, 1)

            print("")
            print("CE solver stats (aggregate):")
            print(f"  ce_calls: {ce_calls}  ce_successes: {ce_successes}  success_rate: {success_rate:.2%}")
            print(f"  ce_nfev_total: {ce_nfev_total}  avg_nfev: {avg_nfev:.2f}")
            print(f"  ce_njev_total: {ce_njev_total}  avg_njev: {avg_njev:.2f}")
            print(f"  ce_attempts_total: {ce_attempts_total}  avg_attempts: {avg_attempts:.2f}")

    if args.profile:
        profile_params = PARAM_GRID[0]
        profile_output = run_profile(
            wl_array,
            linelist,
            profile_params,
            warmup=not args.no_warmup,
            use_prev_ne_initial=args.ce_initial_prev,
        )
        results["profile"] = {
            "params": profile_params,
            "top_30_cumtime": profile_output,
        }

        print("")
        print(profile_output)

    json_out = args.json_out
    if args.save_baseline:
        json_out = str(REPO_ROOT / "output" / "benchmarks" / "jorg_baseline.json")

    if json_out:
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, indent=2))
        print("")
        print(f"Saved JSON: {json_path}")

    if args.ce_stats_out is not None:
        ce_stats_path = args.ce_stats_out
        if ce_stats_path == "__default__":
            ce_stats_path = str(REPO_ROOT / "output" / "benchmarks" / "jorg_ce_stats.json")
        ce_stats = results.get("steps", {}).get("ce_stats")
        if ce_stats:
            ce_path = Path(ce_stats_path)
            ce_path.parent.mkdir(parents=True, exist_ok=True)
            ce_path.write_text(json.dumps(ce_stats, indent=2))
            print("")
            print(f"Saved CE stats JSON: {ce_path}")
        else:
            print("")
            print("CE stats requested but unavailable (run with --steps).")


if __name__ == "__main__":
    main()
