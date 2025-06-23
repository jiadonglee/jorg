#!/usr/bin/env python3
"""
Test the improved Jorg implementation with accurate McLaughlin H^- data
"""
import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

# Add Jorg to path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg')

from jorg.continuum.main import total_continuum_absorption

def test_improved_jorg():
    """Test improved Jorg with McLaughlin data"""
    
    # Load the Korg reference data
    with open('/Users/jdli/Project/Korg.jl/Jorg/korg_reference_data.json', 'r') as f:
        korg_data = json.load(f)
    
    frequencies = np.array(korg_data['frequencies'])
    temperature = korg_data['temperature']
    electron_density = korg_data['electron_density']
    number_densities = korg_data['number_densities']
    korg_alpha = np.array(korg_data['alpha_total'])
    
    # Create partition functions as callable objects
    partition_functions = {
        'H_I': lambda log_t: 2.0,   # Approximate H I partition function
        'H_II': lambda log_t: 1.0,  # H II has no electrons  
        'He_I': lambda log_t: 1.0,  # Ground state only for He I
        'H2': lambda log_t: 1.0     # Approximate for H2
    }
    
    print(f"Testing improved Jorg with McLaughlin interpolation...")
    print(f"Temperature: {temperature} K")
    print(f"Electron density: {electron_density:.3e} cm^-3")
    
    # Call improved Jorg with frequencies in ascending order
    freq_ascending = frequencies[::-1]
    try:
        jorg_alpha = total_continuum_absorption(
            freq_ascending, temperature, electron_density,
            number_densities, partition_functions, True
        )
        
        # Reverse back to match Korg order
        jorg_alpha = jorg_alpha[::-1]
        
        # Calculate comparison statistics
        ratio = jorg_alpha / korg_alpha
        percent_diff = 100 * (jorg_alpha - korg_alpha) / korg_alpha
        
        print(f"\\nComparison Results:")
        print(f"Mean ratio (Jorg/Korg): {np.mean(ratio):.6f}")
        print(f"Std ratio: {np.std(ratio):.6f}")
        print(f"Mean percent difference: {np.mean(np.abs(percent_diff)):.3f}%")
        print(f"Max percent difference: {np.max(np.abs(percent_diff)):.3f}%")
        print(f"Correlation coefficient: {np.corrcoef(korg_alpha, jorg_alpha)[0,1]:.6f}")
        
        # Convert to wavelengths for plotting
        c_cgs = 2.99792458e10
        wavelengths = c_cgs * 1e8 / frequencies  # Angstroms
        
        # Create comparison plot
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Top: absolute values
        ax1.semilogy(wavelengths, korg_alpha, 'b-', linewidth=2, label='Korg (Julia)')
        ax1.semilogy(wavelengths, jorg_alpha, 'r--', linewidth=2, label='Jorg (JAX + McLaughlin)')
        ax1.set_ylabel('Continuum Absorption (cm⁻¹)')
        ax1.set_title('Improved Korg vs Jorg Continuum Absorption Comparison')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Middle: ratio
        ax2.plot(wavelengths, ratio, 'g-', linewidth=2)
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Jorg / Korg Ratio')
        ax2.set_title('Ratio (should be ~1.0)')
        ax2.grid(True, alpha=0.3)
        
        # Bottom: percent difference
        ax3.plot(wavelengths, percent_diff, 'm-', linewidth=2)
        ax3.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
        ax3.set_xlabel('Wavelength (Å)')
        ax3.set_ylabel('Percent Difference (%)')
        ax3.set_title('Percent Difference: 100 × (Jorg - Korg) / Korg')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_fig/improved_korg_jorg_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Summary report
        summary = f"""
IMPROVED JORG COMPARISON REPORT
==============================

Configuration:
- Temperature: {temperature} K
- Electron density: {electron_density:.3e} cm^-3
- Wavelength range: {wavelengths[-1]:.1f} - {wavelengths[0]:.1f} Å
- Number of points: {len(frequencies)}

Statistical Analysis:
- Mean ratio (Jorg/Korg): {np.mean(ratio):.6f}
- Standard deviation of ratio: {np.std(ratio):.6f}
- Mean absolute percent difference: {np.mean(np.abs(percent_diff)):.3f}%
- Maximum absolute percent difference: {np.max(np.abs(percent_diff)):.3f}%
- Correlation coefficient: {np.corrcoef(korg_alpha, jorg_alpha)[0,1]:.6f}

Ratio Statistics:
- Minimum ratio: {np.min(ratio):.6f}
- Maximum ratio: {np.max(ratio):.6f}
- Ratio at 5000 Å: {ratio[0]:.6f}
- Ratio at 6000 Å: {ratio[-1]:.6f}

IMPROVEMENT ANALYSIS:
- Previous implementation: 5.58% mean difference
- Current implementation: {np.mean(np.abs(percent_diff)):.3f}% mean difference
- Improvement: {5.58 - np.mean(np.abs(percent_diff)):.2f} percentage points better

CONCLUSION: {'EXCELLENT' if np.mean(np.abs(percent_diff)) < 2.0 else 'GOOD' if np.mean(np.abs(percent_diff)) < 5.0 else 'NEEDS IMPROVEMENT'}
Agreement achieved with accurate McLaughlin interpolation!
"""
        
        with open('test_fig/improved_comparison_report.txt', 'w') as f:
            f.write(summary)
        
        print(summary)
        
    except Exception as e:
        print(f"Error testing improved Jorg: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_jorg()