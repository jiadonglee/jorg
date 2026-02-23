#!/usr/bin/env python3
"""
Collect autodiff baseline metrics for the JAX synthesis pipeline.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SRC = ROOT / "src"
if SRC.exists():
    sys.path.insert(0, str(SRC))
DATA_DIR = ROOT / "data"
if DATA_DIR.exists():
    os.environ.setdefault("JORG_DATA_DIR", str(DATA_DIR))

from jorg.atmosphere import interpolate_marcs
from jorg.synthesis import create_korg_compatible_abundance_array, synthesize, synthesize_jax

OUTPUT_DIR = ROOT / "output" / "autodiff_baseline"
JSON_PATH = OUTPUT_DIR / "baseline.json"
MD_PATH = OUTPUT_DIR / "baseline.md"


def _relative_stats(reference: np.ndarray, candidate: np.ndarray) -> dict:
    rel = np.abs(candidate - reference) / np.maximum(np.abs(reference), 1e-30)
    return {
        "median": float(np.median(rel)),
        "p95": float(np.percentile(rel, 95.0)),
        "max": float(np.max(rel)),
    }


def _collect() -> dict:
    atm = interpolate_marcs(5771.0, 4.44, 0.0)
    base = jnp.asarray(create_korg_compatible_abundance_array(0.0), dtype=jnp.float64)
    wavelengths = np.linspace(5000.0, 5000.5, 64)

    t0 = time.perf_counter()
    state = synthesize_jax(
        atm=atm,
        linelist=[],
        A_X=base,
        wavelengths=wavelengths,
        rectify=False,
        verbose=False,
    )
    jax_runtime_s = time.perf_counter() - t0

    def loss(abundance_val):
        A_X = base.at[25].set(abundance_val)
        cur_state = synthesize_jax(
            atm=atm,
            linelist=[],
            A_X=A_X,
            wavelengths=wavelengths,
            rectify=False,
            verbose=False,
        )
        return jnp.sum(cur_state.flux)

    grad_val = float(jax.grad(loss)(base[25]))

    t1 = time.perf_counter()
    legacy = synthesize(
        atm=atm,
        linelist=[],
        A_X=np.asarray(base, dtype=np.float64),
        wavelengths=wavelengths,
        engine="legacy",
        ce_solver="jax",
        verbose=False,
    )
    legacy_runtime_s = time.perf_counter() - t1

    flux_jax = np.asarray(jax.device_get(state.flux), dtype=np.float64)
    flux_legacy = np.asarray(legacy.flux, dtype=np.float64)
    flux_stats = _relative_stats(flux_legacy, flux_jax)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": {
            "teff": 5771.0,
            "logg": 4.44,
            "mh": 0.0,
            "wavelength_start_angstrom": 5000.0,
            "wavelength_end_angstrom": 5000.5,
            "n_wavelengths": int(wavelengths.size),
            "linelist": "[]",
        },
        "jax_continuum": {
            "runtime_seconds": float(jax_runtime_s),
            "flux_min": float(np.min(flux_jax)),
            "flux_max": float(np.max(flux_jax)),
        },
        "legacy_continuum": {
            "runtime_seconds": float(legacy_runtime_s),
            "flux_min": float(np.min(flux_legacy)),
            "flux_max": float(np.max(flux_legacy)),
        },
        "gradient": {
            "parameter": "A_X[25]",
            "value": grad_val,
            "is_finite": bool(np.isfinite(grad_val)),
        },
        "legacy_vs_jax_flux_relative": flux_stats,
    }


def _render_markdown(report: dict) -> str:
    rel = report["legacy_vs_jax_flux_relative"]
    grad = report["gradient"]
    jax_block = report["jax_continuum"]
    legacy_block = report["legacy_continuum"]
    return (
        "# Autodiff Baseline\n\n"
        f"- Generated (UTC): {report['generated_at_utc']}\n"
        f"- Scenario: Teff={report['scenario']['teff']}, logg={report['scenario']['logg']}, "
        f"[M/H]={report['scenario']['mh']}, "
        f"wl={report['scenario']['wavelength_start_angstrom']}-"
        f"{report['scenario']['wavelength_end_angstrom']} A, "
        f"N={report['scenario']['n_wavelengths']}\n\n"
        "## Gradient\n\n"
        f"- Parameter: `{grad['parameter']}`\n"
        f"- Value: `{grad['value']:.6e}`\n"
        f"- Finite: `{grad['is_finite']}`\n\n"
        "## Runtime\n\n"
        f"- JAX continuum runtime: `{jax_block['runtime_seconds']:.4f}` s\n"
        f"- Legacy continuum runtime: `{legacy_block['runtime_seconds']:.4f}` s\n\n"
        "## Legacy vs JAX (continuum flux)\n\n"
        f"- Median relative error: `{rel['median']:.6e}`\n"
        f"- P95 relative error: `{rel['p95']:.6e}`\n"
        f"- Max relative error: `{rel['max']:.6e}`\n"
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = _collect()
    JSON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    MD_PATH.write_text(_render_markdown(report), encoding="utf-8")
    print(f"Wrote {JSON_PATH}")
    print(f"Wrote {MD_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

