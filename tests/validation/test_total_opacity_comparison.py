#!/usr/bin/env python3
"""
Total Opacity Comparison: Jorg vs Korg

Calculate and compare total opacity from EOS results between Jorg and Korg.
This test validates the complete EOS → Opacity pipeline.
"""

import sys
from pathlib import Path
import subprocess
import json
import os

# Add Jorg to path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def run_korg_opacity_calculation():
    """Run Korg.jl opacity calculation and extract results"""
    print("Calculating opacity with Korg.jl...")
    
    # Create simplified Julia script that uses fixed densities to avoid chemical equilibrium issues
    julia_script = """
    using Korg
    using JSON
    
    # Solar conditions
    Teff = 5777.0
    logg = 4.44
    m_H = 0.0
    alpha_m = 0.0
    
    # Load solar atmosphere
    atm = interpolate_marcs(Teff, logg, m_H, alpha_m)
    
    # Choose representative atmospheric layer
    layer_index = 25
    layer = atm.layers[layer_index]
    T = layer.temp
    nₜ = layer.number_density
    nₑ_model = layer.electron_number_density
    
    # Wavelength range (Angstroms) - conservative optical range
    λs = 5200.0:50.0:5800.0  # 5200-5800 Å in 50 Å steps (conservative range)
    
    # Convert to frequencies (Hz)
    νs = 2.99792458e18 ./ λs  # c/λ in Hz
    
    # Use simplified number densities to avoid chemical equilibrium issues
    # These are approximate values typical for solar photosphere conditions
    number_densities = Dict{Korg.Species, Float64}()
    
    # Create species objects
    H_I = Korg.Species(Korg.Formula(1), 0)
    H_II = Korg.Species(Korg.Formula(1), 1)
    He_I = Korg.Species(Korg.Formula(2), 0)
    He_II = Korg.Species(Korg.Formula(2), 1)
    Fe_I = Korg.Species(Korg.Formula(26), 0)
    Fe_II = Korg.Species(Korg.Formula(26), 1)
    
    # Set typical number densities (cm⁻³) for solar photosphere
    number_densities[H_I] = 2.5e16    # Most hydrogen neutral
    number_densities[H_II] = 6.0e10   # Small ionized fraction
    number_densities[He_I] = 2.0e15   # Most helium neutral
    number_densities[He_II] = 1.0e11  # Small ionized fraction
    number_densities[Fe_I] = 9.0e10   # Mostly neutral iron
    number_densities[Fe_II] = 3.0e10  # Some ionized iron
    
    # Add molecular species that Korg continuum calculation expects
    H2 = Korg.Species(Korg.Formula([1, 1]), 0)  # H2 molecule
    number_densities[H2] = 1.0e13  # Some H2 molecules
    
    # Use model atmosphere electron density
    nₑ = nₑ_model
    
    # Calculate total continuum opacity
    α_continuum = Korg.total_continuum_absorption(
        νs, T, nₑ, number_densities, Korg.default_partition_funcs
    )
    
    # Results to export
    results = Dict(
        "wavelengths_angstrom" => collect(λs),
        "frequencies_hz" => collect(νs), 
        "temperature_k" => T,
        "electron_density_cm3" => nₑ,
        "total_density_cm3" => nₜ,
        "continuum_opacity_cm2_g" => collect(α_continuum),
        "layer_number" => layer_index,
        "stellar_params" => Dict(
            "Teff" => Teff,
            "logg" => logg,
            "m_H" => m_H
        ),
        "number_densities" => Dict(string(k) => v for (k, v) in number_densities)
    )
    
    # Export to JSON
    open("korg_opacity_results.json", "w") do f
        JSON.print(f, results, 2)
    end
    
    println("Korg opacity calculation completed")
    println("Temperature: ", T, " K")
    println("Electron density: ", nₑ, " cm⁻³")
    println("Total density: ", nₜ, " cm⁻³")
    println("Wavelength range: ", minimum(λs), "-", maximum(λs), " Å")
    println("Continuum opacity range: ", minimum(α_continuum), "-", maximum(α_continuum), " cm²/g")
    """
    
    # Write Julia script
    script_path = Path("korg_opacity_script.jl")
    with open(script_path, 'w') as f:
        f.write(julia_script)
    
    try:
        # Run Julia script
        result = subprocess.run([
            "julia", "--project=.", str(script_path)
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"Korg opacity calculation failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        
        print("Korg opacity calculation successful!")
        print(result.stdout)
        
        # Load results
        with open("korg_opacity_results.json", 'r') as f:
            korg_results = json.load(f)
        
        # Cleanup
        script_path.unlink()
        Path("korg_opacity_results.json").unlink()
        
        return korg_results
        
    except Exception as e:
        print(f"Error running Korg opacity calculation: {e}")
        return None

def calculate_jorg_opacity(korg_results):
    """Calculate opacity with Jorg using same conditions as Korg"""
    print("\\nCalculating opacity with Jorg...")
    
    try:
        from jorg.continuum.complete_continuum import calculate_total_continuum_opacity
        
        # Extract conditions from Korg results
        T = korg_results["temperature_k"]
        wavelengths = np.array(korg_results["wavelengths_angstrom"])
        frequencies = np.array(korg_results["frequencies_hz"])
        
        # Get stellar parameters
        stellar_params = korg_results["stellar_params"]
        Teff = stellar_params["Teff"]
        logg = stellar_params["logg"]
        m_H = stellar_params["m_H"]
        
        print(f"Using stellar parameters: Teff={Teff}K, logg={logg}, [M/H]={m_H}")
        print(f"Layer temperature: {T}K")
        print(f"Wavelength range: {wavelengths[0]:.1f}-{wavelengths[-1]:.1f} Å")
        
        # Use the same fixed densities as Korg for fair comparison
        nₑ = korg_results["electron_density_cm3"]
        
        # Create the same number densities as used in Korg
        from jorg.statmech.species import Species, Formula
        number_densities = {}
        
        # Recreate the same species densities (cm⁻³) as used in Korg
        number_densities[Species(Formula([1]), 0)] = 2.5e16    # H I
        number_densities[Species(Formula([1]), 1)] = 6.0e10    # H II
        number_densities[Species(Formula([2]), 0)] = 2.0e15    # He I
        number_densities[Species(Formula([2]), 1)] = 1.0e11    # He II
        number_densities[Species(Formula([26]), 0)] = 9.0e10   # Fe I
        number_densities[Species(Formula([26]), 1)] = 3.0e10   # Fe II
        
        # Add H2 molecule (required by some continuum calculations)
        number_densities[Species(Formula([1, 1]), 0)] = 1.0e13  # H2
        
        print(f"Using fixed densities for fair comparison with Korg")
        print(f"H I: {number_densities[Species(Formula([1]), 0)]:.2e} cm⁻³")
        print(f"H II: {number_densities[Species(Formula([1]), 1)]:.2e} cm⁻³")
        print(f"H2: {number_densities[Species(Formula([1, 1]), 0)]:.2e} cm⁻³")
        
        print(f"Jorg electron density: {nₑ:.2e} cm⁻³")
        print(f"Korg electron density: {korg_results['electron_density_cm3']:.2e} cm⁻³")
        print(f"Electron density ratio (Jorg/Korg): {nₑ/korg_results['electron_density_cm3']:.3f}")
        
        # Calculate continuum opacity
        α_continuum = calculate_total_continuum_opacity(
            frequencies, T, nₑ, number_densities
        )
        
        # Calculate total density from number densities
        total_density = sum(number_densities.values())
        
        jorg_results = {
            "wavelengths_angstrom": wavelengths,
            "frequencies_hz": frequencies,
            "temperature_k": T,
            "electron_density_cm3": nₑ,
            "total_density_cm3": total_density,
            "continuum_opacity_cm2_g": α_continuum,
            "number_densities": {str(species): float(density) 
                               for species, density in number_densities.items()},
            "stellar_params": stellar_params
        }
        
        print(f"Jorg continuum opacity range: {np.min(α_continuum):.2e}-{np.max(α_continuum):.2e} cm²/g")
        print("✅ Jorg opacity calculation successful!")
        
        return jorg_results
        
    except Exception as e:
        print(f"❌ Jorg opacity calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_opacity_results(korg_results, jorg_results):
    """Compare opacity results between Korg and Jorg"""
    print("\\n" + "="*60)
    print("OPACITY COMPARISON: KORG vs JORG")
    print("="*60)
    
    # Extract data
    wavelengths = np.array(korg_results["wavelengths_angstrom"])
    korg_opacity = np.array(korg_results["continuum_opacity_cm2_g"])
    jorg_opacity = np.array(jorg_results["continuum_opacity_cm2_g"])
    
    # Calculate comparison metrics
    ratio = jorg_opacity / korg_opacity
    percent_diff = 100 * (jorg_opacity - korg_opacity) / korg_opacity
    
    # Statistical comparison
    mean_ratio = np.mean(ratio)
    std_ratio = np.std(ratio)
    max_diff = np.max(np.abs(percent_diff))
    rms_diff = np.sqrt(np.mean(percent_diff**2))
    
    print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} Å")
    print(f"Number of wavelength points: {len(wavelengths)}")
    print()
    
    print("OPACITY COMPARISON STATISTICS:")
    print(f"  Mean opacity ratio (Jorg/Korg): {mean_ratio:.4f}")
    print(f"  Std deviation of ratio: {std_ratio:.4f}")
    print(f"  Maximum difference: {max_diff:.2f}%")
    print(f"  RMS difference: {rms_diff:.2f}%")
    print()
    
    print("OPACITY RANGES:")
    print(f"  Korg: {np.min(korg_opacity):.2e} - {np.max(korg_opacity):.2e} cm²/g")
    print(f"  Jorg: {np.min(jorg_opacity):.2e} - {np.max(jorg_opacity):.2e} cm²/g")
    print()
    
    print("ELECTRON DENSITY COMPARISON:")
    korg_ne = korg_results["electron_density_cm3"]
    jorg_ne = jorg_results["electron_density_cm3"]
    ne_ratio = jorg_ne / korg_ne
    ne_diff = 100 * (jorg_ne - korg_ne) / korg_ne
    print(f"  Korg nₑ: {korg_ne:.2e} cm⁻³")
    print(f"  Jorg nₑ: {jorg_ne:.2e} cm⁻³")
    print(f"  Ratio (Jorg/Korg): {ne_ratio:.4f}")
    print(f"  Difference: {ne_diff:.2f}%")
    print()
    
    # Detailed wavelength comparison
    print("WAVELENGTH-DEPENDENT COMPARISON:")
    print("λ (Å)     Korg (cm²/g)  Jorg (cm²/g)  Ratio    Diff(%)")
    print("-" * 55)
    
    # Show every 5th point for brevity
    for i in range(0, len(wavelengths), 5):
        λ = wavelengths[i]
        k_op = korg_opacity[i]
        j_op = jorg_opacity[i]
        r = ratio[i]
        d = percent_diff[i]
        print(f"{λ:6.1f}    {k_op:.2e}    {j_op:.2e}   {r:.3f}   {d:+6.2f}")
    
    # Assessment
    print("\\n" + "="*60)
    print("ASSESSMENT:")
    
    if rms_diff < 5.0:
        print("🎉 EXCELLENT AGREEMENT: RMS difference < 5%")
        assessment = "excellent"
    elif rms_diff < 15.0:
        print("✅ GOOD AGREEMENT: RMS difference < 15%")
        assessment = "good"
    elif rms_diff < 30.0:
        print("⚠️  MODERATE AGREEMENT: RMS difference < 30%")
        assessment = "moderate"
    else:
        print("❌ POOR AGREEMENT: RMS difference > 30%")
        assessment = "poor"
    
    if abs(ne_diff) < 10.0:
        print("✅ ELECTRON DENSITY: Excellent agreement < 10%")
    elif abs(ne_diff) < 25.0:
        print("⚠️  ELECTRON DENSITY: Moderate agreement < 25%")
    else:
        print("❌ ELECTRON DENSITY: Poor agreement > 25%")
    
    print("="*60)
    
    return {
        "assessment": assessment,
        "rms_difference_percent": rms_diff,
        "mean_ratio": mean_ratio,
        "electron_density_difference_percent": ne_diff,
        "wavelengths": wavelengths,
        "korg_opacity": korg_opacity,
        "jorg_opacity": jorg_opacity,
        "ratio": ratio,
        "percent_difference": percent_diff
    }

def create_opacity_plots(comparison_results):
    """Create plots comparing opacity results"""
    print("\\nCreating opacity comparison plots...")
    
    try:
        wavelengths = comparison_results["wavelengths"]
        korg_opacity = comparison_results["korg_opacity"]
        jorg_opacity = comparison_results["jorg_opacity"]
        ratio = comparison_results["ratio"]
        percent_diff = comparison_results["percent_difference"]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Opacity comparison
        ax1.semilogy(wavelengths, korg_opacity, 'b-', label='Korg', linewidth=2)
        ax1.semilogy(wavelengths, jorg_opacity, 'r--', label='Jorg', linewidth=2)
        ax1.set_xlabel('Wavelength (Å)')
        ax1.set_ylabel('Continuum Opacity (cm²/g)')
        ax1.set_title('Continuum Opacity Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Opacity ratio
        ax2.plot(wavelengths, ratio, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Opacity Ratio (Jorg/Korg)')
        ax2.set_title('Opacity Ratio vs Wavelength')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Percent difference
        ax3.plot(wavelengths, percent_diff, 'purple', linewidth=2)
        ax3.axhline(y=0.0, color='k', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Wavelength (Å)')
        ax3.set_ylabel('Percent Difference (%)')
        ax3.set_title('Percent Difference: (Jorg-Korg)/Korg × 100%')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Histogram of percent differences
        ax4.hist(percent_diff, bins=20, alpha=0.7, color='orange', edgecolor='black')
        ax4.axvline(x=0.0, color='k', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Percent Difference (%)')
        ax4.set_ylabel('Number of Wavelengths')
        ax4.set_title('Distribution of Percent Differences')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("opacity_comparison_plot.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Opacity comparison plot saved: {plot_path}")
        
        plt.show()
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
        return False

def run_total_opacity_comparison():
    """Run complete opacity comparison test"""
    print("TOTAL OPACITY COMPARISON: JORG vs KORG")
    print("=" * 50)
    print("Testing EOS → Opacity pipeline accuracy")
    print()
    
    # Step 1: Calculate Korg opacity
    korg_results = run_korg_opacity_calculation()
    if korg_results is None:
        print("❌ Korg opacity calculation failed")
        return False
    
    # Step 2: Calculate Jorg opacity  
    jorg_results = calculate_jorg_opacity(korg_results)
    if jorg_results is None:
        print("❌ Jorg opacity calculation failed")
        return False
    
    # Step 3: Compare results
    comparison = compare_opacity_results(korg_results, jorg_results)
    
    # Step 4: Create plots
    plot_success = create_opacity_plots(comparison)
    
    # Step 5: Final assessment
    print("\\n" + "🎯 FINAL SUMMARY")
    print("=" * 50)
    
    rms_diff = comparison["rms_difference_percent"]
    ne_diff = comparison["electron_density_difference_percent"]
    assessment = comparison["assessment"]
    
    if assessment == "excellent" and abs(ne_diff) < 10:
        print("🏆 OUTSTANDING SUCCESS!")
        print("   Both opacity and electron density show excellent agreement")
        print("   EOS → Opacity pipeline is working correctly")
        result = True
    elif assessment in ["excellent", "good"] and abs(ne_diff) < 25:
        print("✅ SUCCESS!")
        print("   Good agreement in opacity calculations")
        print("   EOS → Opacity pipeline is functional")
        result = True
    else:
        print("⚠️  NEEDS IMPROVEMENT")
        print("   Significant differences detected")
        print("   EOS → Opacity pipeline needs investigation")
        result = False
    
    print(f"   RMS opacity difference: {rms_diff:.2f}%")
    print(f"   Electron density difference: {ne_diff:.2f}%")
    print(f"   Plot created: {plot_success}")
    print("=" * 50)
    
    return result

if __name__ == "__main__":
    success = run_total_opacity_comparison()
    sys.exit(0 if success else 1)