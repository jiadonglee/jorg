#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

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
from jorg.statmech import (
    create_default_ionization_energies,
    create_default_log_equilibrium_constants,
    create_default_partition_functions,
)
from jorg.statmech.korg_chemical_equilibrium import chemical_equilibrium
from jorg.synthesis import create_korg_compatible_abundance_array


def _select_layer(atm, layer_index: Optional[int], tau_target: float) -> int:
    if layer_index is not None:
        if layer_index < 0 or layer_index >= len(atm.layers):
            raise ValueError(f"layer_index {layer_index} out of bounds (0..{len(atm.layers) - 1}).")
        return int(layer_index)

    if not np.isfinite(tau_target) or tau_target <= 0.0:
        tau_target = 1.0

    tau = np.array([layer.tau_5000 for layer in atm.layers], dtype=float)
    tau = np.where(tau > 0.0, tau, np.nan)
    if np.all(np.isnan(tau)):
        return 0

    log_tau = np.log10(tau)
    idx = int(np.nanargmin(np.abs(log_tau - np.log10(tau_target))))
    return idx


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a single chemical-equilibrium solve and save a reproducible case JSON."
    )
    parser.add_argument("--teff", type=float, default=5771.0, help="Effective temperature (K)")
    parser.add_argument("--logg", type=float, default=4.44, help="log(g)")
    parser.add_argument("--m-h", dest="m_h", type=float, default=0.0, help="[M/H]")
    parser.add_argument("--layer-index", type=int, default=None, help="Select an explicit layer index")
    parser.add_argument("--tau-target", type=float, default=1.0, help="Pick layer with tau_5000 near this")
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save JSON (default: output/benchmarks/ce_case.json)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write JSON output",
    )
    parser.add_argument(
        "--compare-jaxopt",
        action="store_true",
        help="Run JAXOpt solver and compare against SciPy",
    )
    parser.add_argument(
        "--compare-jax",
        action="store_true",
        help="Run JAX Newton solver and compare against SciPy",
    )
    parser.add_argument(
        "--atomic-only",
        action="store_true",
        help="Disable molecular equilibrium constants for both solvers",
    )
    parser.add_argument(
        "--jaxopt-maxiter",
        type=int,
        default=80,
        help="Max iterations for JAXOpt solver (comparison only)",
    )
    parser.add_argument(
        "--jaxopt-tol",
        type=float,
        default=1e-6,
        help="Tolerance for JAXOpt solver (comparison only)",
    )
    parser.add_argument(
        "--jax-maxiter",
        type=int,
        default=60,
        help="Max iterations for JAX Newton solver (comparison only)",
    )
    parser.add_argument(
        "--jax-tol",
        type=float,
        default=1e-8,
        help="Tolerance for JAX Newton solver (comparison only)",
    )
    parser.add_argument(
        "--jax-damping",
        type=float,
        default=0.0,
        help="Diagonal damping for JAX Newton solver (comparison only)",
    )
    parser.add_argument(
        "--jax-jit",
        action="store_true",
        help="JIT-compile the JAX Newton solver (includes compile time)",
    )
    parser.add_argument(
        "--jax-analytic",
        action="store_true",
        help="Use analytic Jacobian for JAX Newton solver (atomic-only only)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each solver N times to amortize overhead (same layer)",
    )
    parser.add_argument(
        "--jax-warmup",
        action="store_true",
        help="Warm up JAX solver before repeat timing (excludes compile time)",
    )

    args = parser.parse_args()

    try:
        import jax

        backend = jax.default_backend()
    except Exception:
        backend = "unknown"

    atm = interpolate_marcs(args.teff, args.logg, args.m_h)
    layer_idx = _select_layer(atm, args.layer_index, args.tau_target)
    layer = atm.layers[layer_idx]

    A_X = create_korg_compatible_abundance_array(args.m_h)
    abs_abundances = 10 ** (A_X - 12.0)
    abs_abundances = abs_abundances / np.sum(abs_abundances)
    abs_abundances_dict = {Z: abs_abundances[Z - 1] for Z in range(1, 93)}

    partition_funcs = create_default_partition_functions()
    ionization_energies = create_default_ionization_energies()
    log_equilibrium_constants = None if args.atomic_only else create_default_log_equilibrium_constants()

    stats = {}
    t0 = time.perf_counter()
    ne, scipy_species = chemical_equilibrium(
        temp=float(layer.temp),
        nt=float(layer.number_density),
        model_atm_ne=float(layer.electron_number_density),
        absolute_abundances=abs_abundances_dict,
        ionization_energies=ionization_energies,
        partition_funcs=partition_funcs,
        log_equilibrium_constants=log_equilibrium_constants,
        stats_out=stats,
        warn_on_ne_discrepancy=False,
    )
    elapsed = time.perf_counter() - t0

    result = {
        "repo_root": str(REPO_ROOT),
        "backend": backend,
        "case": {
            "teff": float(args.teff),
            "logg": float(args.logg),
            "m_h": float(args.m_h),
            "layer_index": int(layer_idx),
            "tau_5000": float(layer.tau_5000),
            "temperature": float(layer.temp),
            "number_density": float(layer.number_density),
            "electron_density_guess": float(layer.electron_number_density),
            "abundances": abs_abundances.tolist(),
        },
        "solver": stats,
        "results": {
            "electron_density": float(ne),
            "elapsed_sec": float(elapsed),
            "ne_to_guess_ratio": float(ne / max(float(layer.electron_number_density), 1e-300)),
            "ne_to_nt_ratio": float(ne / max(float(layer.number_density), 1e-300)),
        },
    }

    print("")
    print("Chemical equilibrium benchmark")
    print(f"repo_root: {REPO_ROOT}")
    print(f"jax.default_backend(): {backend}")
    print(
        f"layer: idx={layer_idx} tau_5000={layer.tau_5000:.3e} "
        f"T={layer.temp:.1f}K nt={layer.number_density:.3e} ne_guess={layer.electron_number_density:.3e}"
    )
    if stats:
        final = stats.get("attempts", stats)
        if isinstance(final, list) and final:
            final = final[-1]
        print(
            "solver: success={success} status={status} nfev={nfev} njev={njev} fallback={fallback}".format(
                success=final.get("success"),
                status=final.get("status"),
                nfev=final.get("nfev"),
                njev=final.get("njev"),
                fallback=stats.get("fallback_used"),
            )
        )
    print(f"ne: {ne:.3e} cm^-3  elapsed: {elapsed:.3f} s")

    if args.repeat > 1:
        t_repeat = time.perf_counter()
        for _ in range(args.repeat):
            _ = chemical_equilibrium(
                temp=float(layer.temp),
                nt=float(layer.number_density),
                model_atm_ne=float(layer.electron_number_density),
                absolute_abundances=abs_abundances_dict,
                ionization_energies=ionization_energies,
                partition_funcs=partition_funcs,
                log_equilibrium_constants=log_equilibrium_constants,
                warn_on_ne_discrepancy=False,
            )
        repeat_elapsed = time.perf_counter() - t_repeat
        avg_repeat = repeat_elapsed / max(args.repeat, 1)
        print("")
        print(
            f"Repeat SciPy ({args.repeat} runs): total={repeat_elapsed:.3f}s avg={avg_repeat:.4f}s"
        )
        result["repeat"] = {
            "runs": int(args.repeat),
            "scipy_total_sec": float(repeat_elapsed),
            "scipy_avg_sec": float(avg_repeat),
        }

    if args.compare_jaxopt:
        try:
            from jorg.statmech.korg_chemical_equilibrium_jaxopt import chemical_equilibrium_jaxopt
        except Exception as exc:
            print(f"jaxopt unavailable: {exc}")
        else:
            jax_stats = {}
            t1 = time.perf_counter()
            ne_jax, jax_species = chemical_equilibrium_jaxopt(
                temp=float(layer.temp),
                nt=float(layer.number_density),
                model_atm_ne=float(layer.electron_number_density),
                absolute_abundances=abs_abundances_dict,
                ionization_energies=ionization_energies,
                partition_funcs=partition_funcs,
                log_equilibrium_constants=log_equilibrium_constants,
                stats_out=jax_stats,
                maxiter=args.jaxopt_maxiter,
                tol=args.jaxopt_tol,
            )
            elapsed_jax = time.perf_counter() - t1

            rel_ne = abs(ne_jax - ne) / max(abs(ne), 1e-300)
            print("")
            print("JAXOpt comparison:")
            print(f"  ne_jax: {ne_jax:.3e} cm^-3  rel_diff={rel_ne:.3e}  elapsed={elapsed_jax:.3f}s")
            if stats:
                last = (stats.get("attempts") or [{}])[-1]
                print(
                    f"  scipy nfev={last.get('nfev')} njev={last.get('njev')}"
                )
            if jax_stats:
                print(
                    f"  jaxopt num_fun_eval={jax_stats.get('num_fun_eval')} iter={jax_stats.get('iter_num')}"
                )
                print(
                    f"  jaxopt error={jax_stats.get('error')}"
                )

            # Compare a few key species densities
            from jorg.statmech.species import Species as _Species

            compare_species = [
                _Species.from_atomic_number(1, 0),  # H I
                _Species.from_atomic_number(1, 1),  # H II
                _Species.from_atomic_number(26, 0),  # Fe I
                _Species.from_atomic_number(26, 1),  # Fe II
            ]
            for spec in compare_species:
                if spec in scipy_species and spec in jax_species:
                    v_ref = scipy_species[spec]
                    v_jax = jax_species[spec]
                    rel = abs(v_jax - v_ref) / max(abs(v_ref), 1e-300)
                    print(f"  {spec}: rel_diff={rel:.3e}")

    if args.compare_jax:
        try:
            from jorg.statmech.korg_chemical_equilibrium_jax import chemical_equilibrium_jax
        except Exception as exc:
            print(f"jax newton unavailable: {exc}")
        else:
            jax_stats = {}
            t1 = time.perf_counter()
            ne_jax, jax_species = chemical_equilibrium_jax(
                temp=float(layer.temp),
                nt=float(layer.number_density),
                model_atm_ne=float(layer.electron_number_density),
                absolute_abundances=abs_abundances_dict,
                ionization_energies=ionization_energies,
                partition_funcs=partition_funcs,
                log_equilibrium_constants=log_equilibrium_constants,
                stats_out=jax_stats,
                maxiter=args.jax_maxiter,
                tol=args.jax_tol,
                damping=args.jax_damping,
                jit=args.jax_jit,
                analytic_jacobian=args.jax_analytic,
            )
            elapsed_jax = time.perf_counter() - t1

            rel_ne = abs(ne_jax - ne) / max(abs(ne), 1e-300)
            print("")
            print("JAX Newton comparison:")
            print(f"  ne_jax: {ne_jax:.3e} cm^-3  rel_diff={rel_ne:.3e}  elapsed={elapsed_jax:.3f}s")
            if jax_stats:
                print(
                    "  iter={iter_num} error={error:.3e} converged={conv} jit={jit}".format(
                        iter_num=jax_stats.get("iter_num"),
                        error=jax_stats.get("error", float("nan")),
                        conv=jax_stats.get("converged"),
                        jit=jax_stats.get("jit"),
                    )
                )

            from jorg.statmech.species import Species as _Species

            compare_species = [
                _Species.from_atomic_number(1, 0),  # H I
                _Species.from_atomic_number(1, 1),  # H II
                _Species.from_atomic_number(26, 0),  # Fe I
                _Species.from_atomic_number(26, 1),  # Fe II
            ]
            for spec in compare_species:
                if spec in scipy_species and spec in jax_species:
                    v_ref = scipy_species[spec]
                    v_jax = jax_species[spec]
                    rel = abs(v_jax - v_ref) / max(abs(v_ref), 1e-300)
                    print(f"  {spec}: rel_diff={rel:.3e}")

            if args.repeat > 1:
                if args.jax_warmup:
                    _ = chemical_equilibrium_jax(
                        temp=float(layer.temp),
                        nt=float(layer.number_density),
                        model_atm_ne=float(layer.electron_number_density),
                        absolute_abundances=abs_abundances_dict,
                        ionization_energies=ionization_energies,
                        partition_funcs=partition_funcs,
                        log_equilibrium_constants=log_equilibrium_constants,
                        maxiter=args.jax_maxiter,
                        tol=args.jax_tol,
                        damping=args.jax_damping,
                        jit=args.jax_jit,
                        analytic_jacobian=args.jax_analytic,
                    )
                t_repeat = time.perf_counter()
                for _ in range(args.repeat):
                    _ = chemical_equilibrium_jax(
                        temp=float(layer.temp),
                        nt=float(layer.number_density),
                        model_atm_ne=float(layer.electron_number_density),
                        absolute_abundances=abs_abundances_dict,
                        ionization_energies=ionization_energies,
                        partition_funcs=partition_funcs,
                        log_equilibrium_constants=log_equilibrium_constants,
                        maxiter=args.jax_maxiter,
                        tol=args.jax_tol,
                        damping=args.jax_damping,
                        jit=args.jax_jit,
                        analytic_jacobian=args.jax_analytic,
                    )
                repeat_elapsed = time.perf_counter() - t_repeat
                avg_repeat = repeat_elapsed / max(args.repeat, 1)
                print(
                    f"Repeat JAX ({args.repeat} runs): total={repeat_elapsed:.3f}s avg={avg_repeat:.4f}s"
                )
                if "repeat" not in result:
                    result["repeat"] = {"runs": int(args.repeat)}
                result["repeat"].update(
                    {
                        "jax_total_sec": float(repeat_elapsed),
                        "jax_avg_sec": float(avg_repeat),
                        "jax_jit": bool(args.jax_jit),
                        "jax_warmup": bool(args.jax_warmup),
                        "jax_analytic": bool(args.jax_analytic),
                    }
                )

    if not args.no_save:
        json_out = args.json_out or str(REPO_ROOT / "output" / "benchmarks" / "ce_case.json")
        json_path = Path(json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, indent=2))
        print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
