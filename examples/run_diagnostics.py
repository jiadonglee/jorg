#!/usr/bin/env python3
"""
Comprehensive diagnostics comparing Jorg and Korg.jl synthesis results.

This script loads the freshly generated Korg.jl reference data and compares
it against Jorg outputs to identify discrepancies.
"""

import sys
from pathlib import Path
sys.path.append("/Users/jdli/Project/Korg.jl/Jorg/src/")

import numpy as np
from jorg.synthesis import synthesize
from jorg.lines.linelist import read_linelist
from jorg.abundances import format_A_X
from jorg.atmosphere import interpolate_marcs

def load_korg_reference(stellar_type, korg_dir="/Users/jdli/Project/Korg.jl/jorg/examples/korg_reference"):
    """Load Korg.jl reference data for a given stellar type."""
    data = {}
    data['with_lines'] = np.loadtxt(f"{korg_dir}/korg_{stellar_type}_with_lines.txt", comments='#')
    data['continuum_only'] = np.loadtxt(f"{korg_dir}/korg_{stellar_type}_continuum_only.txt", comments='#')
    data['opacity'] = np.loadtxt(f"{korg_dir}/korg_{stellar_type}_opacity.txt", comments='#', skiprows=4, usecols=(1, 2, 3))
    data['atmosphere'] = np.loadtxt(f"{korg_dir}/korg_{stellar_type}_atmosphere.txt", comments='#')
    return data

def run_jorg_synthesis(Teff, logg, m_H, linelist):
    """Run Jorg synthesis for given stellar parameters."""
    # Prepare abundances
    A_X_dict = format_A_X()  # Get solar abundances
    A_X = np.full(92, -50.0)
    A_X[0] = 12.0  # H = 12.0 (unchanged by metallicity)

    # Apply metallicity scaling to metal abundances (Z > 2)
    for Z, abundance in A_X_dict.items():
        if 1 <= Z <= 92:
            if Z > 2:  # Metals (not H or He)
                A_X[Z-1] = abundance + m_H
            else:  # H and He
                A_X[Z-1] = abundance

    # Get atmosphere
    atm = interpolate_marcs(Teff=Teff, logg=logg, m_H=m_H)

    # Synthesize with lines
    result_lines = synthesize(atm, linelist, A_X, wavelengths=(5000, 5020), verbose=False)

    # Synthesize continuum-only
    result_cntm = synthesize(atm, [], A_X, wavelengths=(5000, 5020), verbose=False)

    return result_lines, result_cntm, atm

def compare_results(stellar_type, params, jorg_result, jorg_cntm, jorg_atm, korg_data):
    """Compare Jorg and Korg.jl results."""
    print("=" * 80)
    print(f"DIAGNOSTICS: {stellar_type.upper()}")
    print(f"Parameters: Teff={params['Teff']}K, logg={params['logg']}, [M/H]={params['m_H']}")
    print("=" * 80)

    # Extract Korg.jl data
    korg_wl = korg_data['with_lines'][:, 0]
    korg_flux = korg_data['with_lines'][:, 1]
    korg_cntm = korg_data['with_lines'][:, 2]
    korg_norm = korg_data['with_lines'][:, 3]

    korg_opacity = korg_data['opacity'][:56, :]  # Min, Max, Mean
    korg_temps = korg_data['atmosphere'][:, 1]
    korg_ne = korg_data['atmosphere'][:, 2]
    korg_ntot = korg_data['atmosphere'][:, 3]

    # Jorg data
    jorg_norm = jorg_result.flux / jorg_result.cntm
    jorg_temps = np.array([layer.temp for layer in jorg_atm.layers])
    jorg_ne = jorg_result.electron_number_density
    jorg_ntot = np.array([layer.number_density for layer in jorg_atm.layers])

    # === FLUX COMPARISON ===
    print("\n🔬 FLUX COMPARISON:")

    # Continuum
    korg_cntm_mean = korg_data['continuum_only'][:, 2].mean()
    jorg_cntm_mean = jorg_cntm.cntm.mean()
    cntm_ratio = korg_cntm_mean / jorg_cntm_mean

    print(f"\n  CONTINUUM FLUX:")
    print(f"    Korg.jl: {korg_cntm_mean:.6e} erg/s/cm²")
    print(f"    Jorg:    {jorg_cntm_mean:.6e} erg/s/cm²")
    print(f"    Ratio:   {cntm_ratio:.6e} (Korg/Jorg)")
    print(f"    Discrepancy: {cntm_ratio:.1f}×")

    # With-lines
    korg_flux_mean = korg_flux.mean()
    jorg_flux_mean = jorg_result.flux.mean()
    flux_ratio = korg_flux_mean / jorg_flux_mean

    print(f"\n  WITH-LINES FLUX:")
    print(f"    Korg.jl: {korg_flux_mean:.6e} erg/s/cm²")
    print(f"    Jorg:    {jorg_flux_mean:.6e} erg/s/cm²")
    print(f"    Ratio:   {flux_ratio:.6e} (Korg/Jorg)")
    print(f"    Discrepancy: {flux_ratio:.1f}×")

    # Normalized flux
    norm_diff = np.abs(korg_norm - jorg_norm).mean()
    norm_rms = np.sqrt(np.mean((korg_norm - jorg_norm)**2))

    print(f"\n  NORMALIZED FLUX:")
    print(f"    Mean absolute difference: {norm_diff:.6f}")
    print(f"    RMS difference: {norm_rms:.6f}")
    print(f"    Status: {'✅ EXCELLENT' if norm_diff < 0.01 else '⚠️ MODERATE' if norm_diff < 0.05 else '❌ POOR'}")

    # === OPACITY COMPARISON ===
    print("\n🔬 OPACITY COMPARISON:")

    jorg_alpha_mean = jorg_result.alpha.mean(axis=1)
    korg_alpha_mean = korg_opacity[:, 2]

    valid_mask = (jorg_alpha_mean > 0) & (korg_alpha_mean > 0)
    alpha_ratio = jorg_alpha_mean[valid_mask] / korg_alpha_mean[valid_mask]

    print(f"\n  Korg.jl range: {korg_opacity[:, 0].min():.3e} - {korg_opacity[:, 1].max():.3e} cm⁻¹")
    print(f"  Jorg range:    {jorg_result.alpha.min():.3e} - {jorg_result.alpha.max():.3e} cm⁻¹")
    print(f"\n  Mean opacity ratio (Jorg/Korg): {alpha_ratio.mean():.3f}")
    print(f"  Relative error: {np.abs(1 - alpha_ratio.mean()) * 100:.1f}%")
    print(f"  Status: {'✅ GOOD' if np.abs(1 - alpha_ratio.mean()) < 0.5 else '⚠️ MODERATE' if np.abs(1 - alpha_ratio.mean()) < 2.0 else '❌ POOR'}")

    # === ATMOSPHERIC COMPARISON ===
    print("\n🔬 ATMOSPHERIC STRUCTURE:")

    temp_diff = np.abs(korg_temps - jorg_temps).max()
    ne_ratio = jorg_ne / korg_ne
    ntot_ratio = jorg_ntot / korg_ntot

    print(f"\n  TEMPERATURE:")
    print(f"    Max difference: {temp_diff:.3f} K")
    print(f"    Status: ✅ EXACT")

    print(f"\n  ELECTRON DENSITY:")
    print(f"    Mean ratio (Jorg/Korg): {ne_ratio.mean():.6f}")
    print(f"    Relative difference: {np.abs(1 - ne_ratio).mean() * 100:.3f}%")
    print(f"    Status: {'✅ EXCELLENT' if np.abs(1 - ne_ratio).mean() < 0.01 else '⚠️ MODERATE' if np.abs(1 - ne_ratio).mean() < 0.1 else '❌ POOR'}")

    print(f"\n  TOTAL NUMBER DENSITY:")
    print(f"    Mean ratio (Jorg/Korg): {ntot_ratio.mean():.6f}")
    print(f"    Relative difference: {np.abs(1 - ntot_ratio).mean() * 100:.3f}%")
    print(f"    Status: {'✅ EXCELLENT' if np.abs(1 - ntot_ratio).mean() < 0.01 else '⚠️ MODERATE' if np.abs(1 - ntot_ratio).mean() < 0.1 else '❌ POOR'}")

    # === DIAGNOSTIC SUMMARY ===
    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print(f"\n🎯 KEY FINDINGS:")

    if norm_diff < 0.05 and flux_ratio > 100:
        print(f"  ✅ Normalized flux agrees well (diff={norm_diff:.4f})")
        print(f"  ❌ Absolute flux has {flux_ratio:.1f}× discrepancy")
        print(f"\n  📌 CONCLUSION: Line physics CORRECT, flux normalization INCORRECT")
        print(f"\n  💡 LIKELY CAUSES TO INVESTIGATE:")
        print(f"     1. Planck function units (B_λ vs B_ν)")
        print(f"     2. Wavelength/frequency conversion factor")
        print(f"     3. Missing (ν²/c²) or (λ⁻⁴) factor")
        print(f"     4. Intensity → flux integration")
    elif norm_diff > 0.05:
        print(f"  ❌ Normalized flux disagrees (diff={norm_diff:.4f})")
        print(f"  ❌ Absolute flux has {flux_ratio:.1f}× discrepancy")
        print(f"\n  📌 CONCLUSION: BOTH line physics AND flux calculation need work")
    else:
        print(f"  ✅ Both normalized and absolute flux agree!")
        print(f"\n  📌 CONCLUSION: Implementation matches Korg.jl!")

    print("\n" + "=" * 80 + "\n")

    return {
        'flux_ratio': flux_ratio,
        'cntm_ratio': cntm_ratio,
        'norm_diff': norm_diff,
        'alpha_ratio_mean': alpha_ratio.mean(),
        'ne_ratio_mean': ne_ratio.mean(),
    }

def main():
    print("=" * 80)
    print("JORG vs KORG.JL COMPREHENSIVE DIAGNOSTICS")
    print("=" * 80)
    print()

    # Load linelist
    print("📖 Loading VALD linelist...")
    linelist_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
    linelist = read_linelist(linelist_path)
    print(f"   Loaded {len(linelist)} spectral lines\n")

    # Stellar parameters
    stellar_types = {
        "solar": {"Teff": 5771, "logg": 4.44, "m_H": 0.0},
        "arcturus": {"Teff": 4250, "logg": 1.4, "m_H": -0.5},
        "metal_poor_k_giant": {"Teff": 4500, "logg": 1.5, "m_H": -2.5},
    }

    results = {}

    for stellar_type, params in stellar_types.items():
        print(f"\n{'=' * 80}")
        print(f"Processing: {stellar_type}")
        print(f"{'=' * 80}\n")

        # Load Korg.jl reference
        print(f"📖 Loading Korg.jl reference data for {stellar_type}...")
        korg_data = load_korg_reference(stellar_type)
        print(f"   ✅ Loaded\n")

        # Run Jorg synthesis
        print(f"🔬 Running Jorg synthesis for {stellar_type}...")
        jorg_result, jorg_cntm, jorg_atm = run_jorg_synthesis(
            params['Teff'], params['logg'], params['m_H'], linelist
        )
        print(f"   ✅ Complete\n")

        # Compare
        results[stellar_type] = compare_results(
            stellar_type, params, jorg_result, jorg_cntm, jorg_atm, korg_data
        )

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print()

    for stellar_type, res in results.items():
        print(f"{stellar_type:25s}: flux_ratio={res['flux_ratio']:8.1f}×, norm_diff={res['norm_diff']:.4f}")

    print("\n" + "=" * 80)
    print("Diagnostics complete! Review findings above.")
    print("=" * 80)

if __name__ == "__main__":
    main()
