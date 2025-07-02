#!/usr/bin/env python3
"""
Component-wise comparison between Korg and Jorg to identify 4% difference source
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

# Add Jorg to path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg')

from jorg.continuum.hydrogen import (
    h_i_bf_absorption, h_minus_bf_absorption, h_minus_ff_absorption
)
from jorg.continuum.scattering import thomson_scattering, rayleigh_scattering
from jorg.continuum.main import total_continuum_absorption

def compare_components():
    """Compare individual components between Korg and Jorg"""
    
    print("=== COMPONENT-WISE KORG VS JORG COMPARISON ===")
    
    # Load Korg reference data
    with open('/Users/jdli/Project/Korg.jl/Jorg/korg_reference_data.json', 'r') as f:
        korg_data = json.load(f)
    
    frequencies = np.array(korg_data['frequencies'])
    temperature = korg_data['temperature']
    electron_density = korg_data['electron_density']
    number_densities = korg_data['number_densities']
    korg_total = np.array(korg_data['alpha_total'])
    
    print(f"Test conditions:")
    print(f"  Temperature: {temperature} K")
    print(f"  Electron density: {electron_density:.3e} cm^-3")
    print(f"  H I density: {number_densities['H_I']:.3e} cm^-3")
    print()
    
    # Use exact partition functions from debug
    with open('/Users/jdli/Project/Korg.jl/partition_function_debug.json', 'r') as f:
        pf_data = json.load(f)
    
    U_H_I = pf_data['exact_values']['U_H_I']
    U_He_I = pf_data['exact_values']['U_He_I']
    
    print(f"Using exact partition functions:")
    print(f"  U(H I) = {U_H_I}")
    print(f"  U(He I) = {U_He_I}")
    print()
    
    # Calculate individual Jorg components using exact values
    freq_ascending = frequencies[::-1]  # Jorg expects ascending order
    
    # H I densities
    n_h_i = number_densities['H_I']
    n_h_i_div_u = n_h_i / U_H_I
    
    print("Calculating individual Jorg components...")
    
    # 1. H I bound-free
    jorg_h_i_bf = h_i_bf_absorption(
        freq_ascending, temperature, n_h_i_div_u, 0.0, electron_density, 1.0/U_H_I
    )[::-1]  # Reverse back to Korg order
    
    # 2. H^- bound-free  
    jorg_h_minus_bf = h_minus_bf_absorption(
        freq_ascending, temperature, n_h_i_div_u, electron_density
    )[::-1]
    
    # 3. H^- free-free
    jorg_h_minus_ff = h_minus_ff_absorption(
        freq_ascending, temperature, n_h_i_div_u, electron_density
    )[::-1]
    
    # 4. Thomson scattering
    jorg_thomson = np.full_like(freq_ascending, thomson_scattering(electron_density))[::-1]
    
    # 5. Rayleigh scattering (H I)
    jorg_rayleigh = rayleigh_scattering(
        freq_ascending, n_h_i_div_u * U_H_I, 0.0, number_densities.get('H2', 0.0)
    )[::-1]
    
    # Total Jorg 
    jorg_total_components = (jorg_h_i_bf + jorg_h_minus_bf + jorg_h_minus_ff + 
                           jorg_thomson + jorg_rayleigh)
    
    # Also get total from main function
    jorg_total_main = total_continuum_absorption(
        freq_ascending, temperature, electron_density,
        number_densities, {
            'H_I': lambda log_t: U_H_I,
            'He_I': lambda log_t: U_He_I,
            'H_II': lambda log_t: 1.0,
            'H2': lambda log_t: 1.0
        }, True
    )[::-1]
    
    # Convert to wavelengths for analysis
    c_cgs = 2.99792458e10
    wavelengths = c_cgs * 1e8 / frequencies
    
    print("\\nComponent analysis:")
    
    # Calculate ratios for each component
    mid_idx = len(wavelengths) // 2  # ~5500 Å
    
    print(f"Values at {wavelengths[mid_idx]:.0f} Å:")
    print(f"  H I bf:       Jorg={jorg_h_i_bf[mid_idx]:.3e}, Korg=(need reference)")
    print(f"  H^- bf:       Jorg={jorg_h_minus_bf[mid_idx]:.3e}")
    print(f"  H^- ff:       Jorg={jorg_h_minus_ff[mid_idx]:.3e}")
    print(f"  Thomson:      Jorg={jorg_thomson[mid_idx]:.3e}")
    print(f"  Rayleigh:     Jorg={jorg_rayleigh[mid_idx]:.3e}")
    print(f"  Total (sum):  Jorg={jorg_total_components[mid_idx]:.3e}")
    print(f"  Total (main): Jorg={jorg_total_main[mid_idx]:.3e}")
    print(f"  Korg total:   {korg_total[mid_idx]:.3e}")
    print()
    
    # Component contributions
    contributions = {
        'H I bf': 100 * np.mean(jorg_h_i_bf / jorg_total_components),
        'H^- bf': 100 * np.mean(jorg_h_minus_bf / jorg_total_components),
        'H^- ff': 100 * np.mean(jorg_h_minus_ff / jorg_total_components),
        'Thomson': 100 * np.mean(jorg_thomson / jorg_total_components),
        'Rayleigh': 100 * np.mean(jorg_rayleigh / jorg_total_components)
    }
    
    print("Component contributions (% of total):")
    for comp, pct in contributions.items():
        print(f"  {comp:<12}: {pct:6.2f}%")
    print()
    
    # Compare totals
    ratio_components = jorg_total_components / korg_total
    ratio_main = jorg_total_main / korg_total
    
    print("Total comparison:")
    print(f"  Components sum vs Korg: {np.mean(ratio_components):.6f} ± {np.std(ratio_components):.6f}")
    print(f"  Main function vs Korg:  {np.mean(ratio_main):.6f} ± {np.std(ratio_main):.6f}")
    print(f"  Components vs Main:     {np.mean(jorg_total_components/jorg_total_main):.6f}")
    
    # Create detailed comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Individual components
    ax1 = axes[0, 0]
    ax1.semilogy(wavelengths, jorg_h_i_bf, 'b-', label='H I bf', linewidth=2)
    ax1.semilogy(wavelengths, jorg_h_minus_bf, 'r-', label='H^- bf', linewidth=2)
    ax1.semilogy(wavelengths, jorg_h_minus_ff, 'g-', label='H^- ff', linewidth=2)
    ax1.semilogy(wavelengths, jorg_thomson, 'm--', label='Thomson', linewidth=1)
    ax1.semilogy(wavelengths, jorg_rayleigh, 'c--', label='Rayleigh', linewidth=1)
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Absorption (cm⁻¹)')
    ax1.set_title('Jorg Individual Components')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Total comparison
    ax2 = axes[0, 1]
    ax2.semilogy(wavelengths, korg_total, 'b-', linewidth=2, label='Korg')
    ax2.semilogy(wavelengths, jorg_total_main, 'r--', linewidth=2, label='Jorg (main)')
    ax2.semilogy(wavelengths, jorg_total_components, 'g:', linewidth=2, label='Jorg (sum)')
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Total Absorption (cm⁻¹)')
    ax2.set_title('Total Continuum Absorption')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Ratios
    ax3 = axes[1, 0]
    ax3.plot(wavelengths, ratio_main, 'r-', linewidth=2, label='Jorg/Korg (main)')
    ax3.plot(wavelengths, ratio_components, 'g--', linewidth=2, label='Jorg/Korg (sum)')
    ax3.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Wavelength (Å)')
    ax3.set_ylabel('Ratio')
    ax3.set_title('Jorg/Korg Ratios')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Percent differences
    ax4 = axes[1, 1]
    percent_diff_main = 100 * (jorg_total_main - korg_total) / korg_total
    percent_diff_comp = 100 * (jorg_total_components - korg_total) / korg_total
    ax4.plot(wavelengths, percent_diff_main, 'r-', linewidth=2, label='Main function')
    ax4.plot(wavelengths, percent_diff_comp, 'g--', linewidth=2, label='Components sum')
    ax4.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Wavelength (Å)')
    ax4.set_ylabel('Percent Difference (%)')
    ax4.set_title('Percent Difference from Korg')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_fig/component_wise_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save component data for further analysis
    component_data = {
        'wavelengths': wavelengths.tolist(),
        'frequencies': frequencies.tolist(),
        'korg_total': korg_total.tolist(),
        'jorg_components': {
            'h_i_bf': [float(x) for x in jorg_h_i_bf],
            'h_minus_bf': [float(x) for x in jorg_h_minus_bf],
            'h_minus_ff': [float(x) for x in jorg_h_minus_ff],
            'thomson': [float(x) for x in jorg_thomson],
            'rayleigh': [float(x) for x in jorg_rayleigh],
            'total_sum': [float(x) for x in jorg_total_components],
            'total_main': [float(x) for x in jorg_total_main]
        },
        'contributions_percent': contributions,
        'comparison_stats': {
            'ratio_main_mean': float(np.mean(ratio_main)),
            'ratio_main_std': float(np.std(ratio_main)),
            'ratio_components_mean': float(np.mean(ratio_components)),
            'ratio_components_std': float(np.std(ratio_components)),
            'percent_diff_main_mean': float(np.mean(np.abs(percent_diff_main))),
            'percent_diff_comp_mean': float(np.mean(np.abs(percent_diff_comp)))
        }
    }
    
    with open('test_fig/component_analysis.json', 'w') as f:
        json.dump(component_data, f, indent=2)
    
    print("\\nComponent analysis complete!")
    print("Data saved to test_fig/component_analysis.json")
    print("Plot saved to test_fig/component_wise_comparison.png")

if __name__ == "__main__":
    compare_components()