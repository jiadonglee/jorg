"""
Line-by-line comparison: Which lines are Jorg including that Korg isn't?

Strategy:
1. Run both Jorg and Korg.jl on same parameters
2. Identify pixels where Jorg has strong absorption but Korg doesn't
3. Trace back to which lines are causing the extra absorption

Author: Claude Code Assistant
Date: December 2024
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jorg.synthesis import synthesize_korg_compatible
from jorg.abundances import format_abundances
from jorg.atmosphere import interpolate_marcs
from jorg.lines.linelist import read_linelist

print("=" * 80)
print("LINE-BY-LINE COMPARISON: Jorg vs Korg.jl")
print("=" * 80)

# Parameters
Teff = 5000
logg = 4.0
m_H = -0.5
wl_range = (5000, 6000)  # Original problematic range

print(f"\nParameters:")
print(f"  Teff: {Teff} K")
print(f"  logg: {logg}")
print(f"  [M/H]: {m_H}")
print(f"  Wavelength: {wl_range[0]}-{wl_range[1]} Å")

# Load linelist
linelist_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
linelist = read_linelist(str(linelist_path))

# Get atmosphere and abundances
A_X = format_abundances(default_metals_H=m_H, default_alpha_H=m_H, abundances={})
atm = interpolate_marcs(Teff=Teff, logg=logg, m_H=m_H)

print(f"\n{'='*80}")
print("RUNNING JORG WITH DETAILED LINE DIAGNOSTICS")
print('='*80)

# Patch KorgLineProcessor to capture which lines contribute
from jorg.opacity.korg_line_processor import KorgLineProcessor

line_contributions = {}  # {wavelength: [(line_wl, species, amplitude), ...]}

original_process_lines = KorgLineProcessor.process_lines

def traced_process_lines(self, wl_array_cm, temps, electron_densities, n_densities,
                        partition_fns, linelist, microturbulence_cm_s,
                        continuum_opacity_fn, cutoff_threshold):
    """Capture which lines contribute to each wavelength"""

    # Call original
    result = original_process_lines(
        self, wl_array_cm, temps, electron_densities, n_densities,
        partition_fns, linelist, microturbulence_cm_s,
        continuum_opacity_fn, cutoff_threshold
    )

    # Store line info
    for line in linelist:
        line_wl_angstrom = line.wavelength * 1e8
        # Store line info for later analysis
        if line_wl_angstrom not in line_contributions:
            line_contributions[line_wl_angstrom] = []

        # Get species and log_gf for identification
        line_contributions[line_wl_angstrom].append({
            'species': str(line.species),
            'log_gf': line.log_gf,
            'E_lower': line.E_lower
        })

    return result

# Monkey patch
KorgLineProcessor.process_lines = traced_process_lines

print("\nRunning Jorg synthesis...")
result = synthesize_korg_compatible(
    atm=atm,
    linelist=linelist,
    A_X=A_X,
    wavelengths=wl_range,
    logg=logg,
    line_cutoff_threshold=3e-4,
    return_cntm=True,
    rectify=True,
    verbose=False
)

flux_jorg = result.flux / result.cntm
wl_jorg = result.wavelengths

print(f"\nJorg Results:")
print(f"  Wavelength points: {len(wl_jorg)}")
print(f"  Flux range: {flux_jorg.min():.6f} - {flux_jorg.max():.6f}")
print(f"  Strong lines (flux < 0.8): {np.sum(flux_jorg < 0.8)} pixels ({np.sum(flux_jorg < 0.8)/len(flux_jorg)*100:.1f}%)")

# Load Korg.jl results
print(f"\n{'='*80}")
print("LOADING KORG.JL REFERENCE")
print('='*80)

korg_file = Path("/Users/jdli/Project/Korg.jl/jorg/korg_output.txt")
if not korg_file.exists():
    print(f"\n⚠️  Running Korg.jl to generate reference...")
    import subprocess
    korg_script = Path("/Users/jdli/Project/Korg.jl/jorg/korg_script/compare_jorg_korg.jl")

    # Update Julia script with new parameters
    script_content = f'''
using Korg
using Printf

# Parameters
Teff = {Teff}
logg = {logg}
m_H = {m_H}
wl_min = {wl_range[0]}
wl_max = {wl_range[1]}

# Get atmosphere
atm = interpolate_marcs(Teff, logg, m_H)

# Format abundances (Korg.jl uses solar_relative instead of m_H)
A_X = format_A_X(m_H, m_H)

# Load linelist
linelist_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
linelist = read_linelist(linelist_path, format="vald")

# Synthesize
wl_range = (wl_min, wl_max)
result = synthesize(atm, linelist, A_X, wl_range)

# Get rectified flux
wl = result.wavelengths
flux = result.flux
cntm = result.cntm
rect_flux = flux ./ cntm

# Save results
output_file = "/Users/jdli/Project/Korg.jl/jorg/korg_output.txt"
open(output_file, "w") do io
    println(io, "# wavelength(A) flux continuum rectified_flux")
    for i in 1:length(wl)
        @printf(io, "%.6f %.12e %.12e %.12e\\n", wl[i], flux[i], cntm[i], rect_flux[i])
    end
end

println("Saved Korg.jl output to ", output_file)
println("Wavelength points: ", length(wl))
println("Rectified flux range: ", minimum(rect_flux), " - ", maximum(rect_flux))
strong_count = sum(rect_flux .< 0.8)
println("Strong lines (flux < 0.8): ", strong_count, " pixels (", 100*strong_count/length(rect_flux), "%)")
'''

    korg_script.write_text(script_content)

    result = subprocess.run(
        ["julia", "--project=.", str(korg_script)],
        cwd="/Users/jdli/Project/Korg.jl",
        capture_output=True,
        text=True,
        timeout=120
    )

    if result.returncode != 0:
        print(f"❌ Korg.jl failed:")
        print(result.stderr)
        sys.exit(1)

    print(result.stdout)

# Load Korg.jl data
korg_data = np.loadtxt(korg_file, comments='#')
wl_korg = korg_data[:, 0]
flux_korg_rect = korg_data[:, 3]

print(f"\nKorg.jl Results:")
print(f"  Wavelength points: {len(wl_korg)}")
print(f"  Flux range: {flux_korg_rect.min():.6f} - {flux_korg_rect.max():.6f}")
print(f"  Strong lines (flux < 0.8): {np.sum(flux_korg_rect < 0.8)} pixels ({np.sum(flux_korg_rect < 0.8)/len(flux_korg_rect)*100:.1f}%)")

# Interpolate Korg onto Jorg grid for direct comparison
flux_korg_interp = np.interp(wl_jorg, wl_korg, flux_korg_rect)

print(f"\n{'='*80}")
print("PIXEL-BY-PIXEL ANALYSIS")
print('='*80)

# Find pixels where Jorg has strong absorption but Korg doesn't
jorg_strong = flux_jorg < 0.8
korg_strong = flux_korg_interp < 0.8

jorg_only_strong = jorg_strong & ~korg_strong
korg_only_strong = korg_strong & ~jorg_strong
both_strong = jorg_strong & korg_strong

print(f"\nStrong absorption pixels:")
print(f"  Both Jorg & Korg: {np.sum(both_strong)} pixels")
print(f"  Jorg only: {np.sum(jorg_only_strong)} pixels")
print(f"  Korg only: {np.sum(korg_only_strong)} pixels")

if np.sum(jorg_only_strong) > 0:
    print(f"\n{'='*80}")
    print("PIXELS WHERE JORG HAS EXTRA ABSORPTION")
    print('='*80)

    jorg_only_indices = np.where(jorg_only_strong)[0]
    print(f"\nFound {len(jorg_only_indices)} pixels where Jorg < 0.8 but Korg ≥ 0.8")

    for i in jorg_only_indices[:10]:  # Show first 10
        wl = wl_jorg[i]
        flux_j = flux_jorg[i]
        flux_k = flux_korg_interp[i]
        diff = flux_j - flux_k

        print(f"\nPixel {i}: {wl:.4f} Å")
        print(f"  Jorg flux: {flux_j:.6f}")
        print(f"  Korg flux: {flux_k:.6f}")
        print(f"  Difference: {diff:.6f} (Jorg {abs(diff)*100:.1f}% stronger absorption)")

        # Find nearest lines
        nearby_lines = []
        for line_wl, line_info_list in line_contributions.items():
            if abs(line_wl - wl) < 1.0:  # Within 1 Å
                for line_info in line_info_list:
                    nearby_lines.append((line_wl, line_info))

        if nearby_lines:
            nearby_lines.sort(key=lambda x: abs(x[0] - wl))
            print(f"  Nearby lines:")
            for line_wl, line_info in nearby_lines[:5]:
                print(f"    {line_info['species']} at {line_wl:.4f} Å (Δ={abs(line_wl-wl):.4f} Å)")
                print(f"      log_gf={line_info['log_gf']:.2f}, E_lower={line_info['E_lower']:.2f} eV")

print(f"\n{'='*80}")
print("SUMMARY")
print('='*80)
print(f"\nJorg has {np.sum(jorg_only_strong)} extra strong absorption pixels")
print(f"This represents {np.sum(jorg_only_strong)/len(flux_jorg)*100:.1f}% of the spectrum")
print(f"\nKorg has {np.sum(korg_only_strong)} strong absorption pixels that Jorg misses")
print(f"This represents {np.sum(korg_only_strong)/len(flux_jorg)*100:.1f}% of the spectrum")

diff = flux_jorg - flux_korg_interp
rms_diff = np.sqrt(np.mean(diff**2))
print(f"\nRMS difference: {rms_diff:.6f}")
print(f"Mean absolute difference: {np.mean(np.abs(diff)):.6f}")
print(f"Max absolute difference: {np.max(np.abs(diff)):.6f}")

print("\n" + "=" * 80)
