#!/usr/bin/env python3
"""
Jorg Unit Test 7: Korg vs Jorg parameter comparison
Validates Arcturus-like and metal-poor K-giant deltas stay within tolerance.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
JORG_SRC = REPO_ROOT / "jorg" / "src"
sys.path.insert(0, str(JORG_SRC))

from jorg.synthesis import synthesize, create_korg_compatible_abundance_array
from jorg.lines.linelist import read_linelist
from jorg.atmosphere import interpolate_marcs

LINELIST_PATH = REPO_ROOT / "data" / "linelists" / "vald_extract_stellar_solar_threshold001.vald"
KORG_SCRIPT = REPO_ROOT / "jorg" / "test" / "unit" / "korg_compare_params.jl"

LINE_ALPHA_REL_TOL = 5e-3
RECTIFIED_MEAN_ABS_TOL = 5e-3
FLUX_SCALE_TOL = 2e-2


def run_jorg_case(teff, logg, m_h, wl_min, wl_max, tag, out_dir):
    linelist = read_linelist(str(LINELIST_PATH), format="vald")
    atm = interpolate_marcs(teff, logg, m_h)
    A_X = create_korg_compatible_abundance_array(m_h)

    result_lines = synthesize(
        atm,
        linelist,
        A_X,
        wavelengths=(wl_min, wl_max),
        hydrogen_lines=False,
        verbose=False,
        logg=logg,
    )
    result_cntm = synthesize(
        atm,
        [],
        A_X,
        wavelengths=(wl_min, wl_max),
        hydrogen_lines=False,
        verbose=False,
        logg=logg,
    )

    wl = np.asarray(result_lines.wavelengths)
    flux = np.asarray(result_lines.flux)
    cntm = np.asarray(result_lines.cntm)
    rect = flux / cntm

    wl_c = np.asarray(result_cntm.wavelengths)
    flux_c = np.asarray(result_cntm.flux)
    cntm_c = np.asarray(result_cntm.cntm)
    rect_c = flux_c / cntm_c

    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / f"jorg_{tag}_spectrum_with_lines.txt", np.column_stack([wl, flux, cntm, rect]))
    np.savetxt(out_dir / f"jorg_{tag}_spectrum_continuum.txt", np.column_stack([wl_c, flux_c, cntm_c, rect_c]))

    np.savetxt(out_dir / f"jorg_{tag}_alpha_with_lines.txt", np.asarray(result_lines.alpha))
    np.savetxt(out_dir / f"jorg_{tag}_alpha_continuum.txt", np.asarray(result_cntm.alpha))


def run_korg_case(teff, logg, m_h, wl_min, wl_max, tag, out_dir):
    cmd = [
        "julia",
        "--project=.",
        str(KORG_SCRIPT),
        str(teff),
        str(logg),
        str(m_h),
        str(wl_min),
        str(wl_max),
        tag,
        str(out_dir),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def load_case(prefix, tag, out_dir):
    spectrum = np.loadtxt(out_dir / f"{prefix}_{tag}_spectrum_with_lines.txt")
    alpha_lines = np.loadtxt(out_dir / f"{prefix}_{tag}_alpha_with_lines.txt")
    alpha_cntm = np.loadtxt(out_dir / f"{prefix}_{tag}_alpha_continuum.txt")
    return spectrum, alpha_lines, alpha_cntm


def compute_metrics(k_spectrum, k_alpha_lines, k_alpha_cntm, j_spectrum, j_alpha_lines, j_alpha_cntm):
    n_layers = min(k_alpha_lines.shape[0], j_alpha_lines.shape[0])
    n_wl = min(k_alpha_lines.shape[1], j_alpha_lines.shape[1])

    k_alpha_lines = k_alpha_lines[:n_layers, :n_wl]
    k_alpha_cntm = k_alpha_cntm[:n_layers, :n_wl]
    j_alpha_lines = j_alpha_lines[:n_layers, :n_wl]
    j_alpha_cntm = j_alpha_cntm[:n_layers, :n_wl]

    k_line_only = k_alpha_lines - k_alpha_cntm
    j_line_only = j_alpha_lines - j_alpha_cntm

    k_line_mean = np.mean(np.abs(k_line_only), axis=1)
    j_line_mean = np.mean(np.abs(j_line_only), axis=1)

    line_rel_diff = np.mean(np.abs(j_line_mean - k_line_mean) / np.maximum(k_line_mean, 1e-40))

    n_flux = min(k_spectrum.shape[0], j_spectrum.shape[0])
    k_flux = k_spectrum[:n_flux, 1]
    j_flux = j_spectrum[:n_flux, 1]
    k_rect = k_spectrum[:n_flux, 3]
    j_rect = j_spectrum[:n_flux, 3]

    flux_scale = np.mean(j_flux) / np.mean(k_flux)
    rect_mean_abs = np.mean(np.abs(j_rect - k_rect))

    return line_rel_diff, flux_scale, rect_mean_abs


def run_case(teff, logg, m_h, tag):
    wl_min = 5000.0
    wl_max = 5020.0

    with tempfile.TemporaryDirectory(prefix="korg_jorg_compare_") as tmpdir:
        out_dir = Path(tmpdir)

        run_jorg_case(teff, logg, m_h, wl_min, wl_max, tag, out_dir)
        run_korg_case(teff, logg, m_h, wl_min, wl_max, tag, out_dir)

        k_spectrum, k_alpha_lines, k_alpha_cntm = load_case("korg", tag, out_dir)
        j_spectrum, j_alpha_lines, j_alpha_cntm = load_case("jorg", tag, out_dir)

        line_rel_diff, flux_scale, rect_mean_abs = compute_metrics(
            k_spectrum, k_alpha_lines, k_alpha_cntm,
            j_spectrum, j_alpha_lines, j_alpha_cntm
        )

    print(f"Case {tag}:")
    print(f"  mean |line alpha| relative diff: {line_rel_diff:.3e}")
    print(f"  flux scale factor (Jorg/Korg): {flux_scale:.6f}")
    print(f"  rectified mean abs diff: {rect_mean_abs:.6f}")

    assert line_rel_diff < LINE_ALPHA_REL_TOL, (
        f"Line alpha relative diff {line_rel_diff:.3e} exceeds tolerance {LINE_ALPHA_REL_TOL:.3e}"
    )
    assert abs(flux_scale - 1.0) < FLUX_SCALE_TOL, (
        f"Flux scale {flux_scale:.6f} exceeds tolerance {FLUX_SCALE_TOL:.3e}"
    )
    assert rect_mean_abs < RECTIFIED_MEAN_ABS_TOL, (
        f"Rectified mean abs diff {rect_mean_abs:.6f} exceeds tolerance {RECTIFIED_MEAN_ABS_TOL:.3e}"
    )


def main():
    cases = [
        (4286.0, 1.66, -0.5, "arcturus_like"),
        (4286.0, 1.66, -2.0, "metal_poor_kgiant"),
    ]

    for teff, logg, m_h, tag in cases:
        run_case(teff, logg, m_h, tag)

    print("\n✅ Korg vs Jorg parameter comparison: PASS")


if __name__ == "__main__":
    main()
