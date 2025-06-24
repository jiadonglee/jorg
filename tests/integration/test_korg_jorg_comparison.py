#!/usr/bin/env python3
"""
Comprehensive comparison between Korg (Julia) and Jorg (JAX) continuum absorption

This script loads Korg reference data and compares it with Jorg calculations,
creating detailed comparison plots and statistical analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path
import os

# Add jorg to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    
    from jorg.continuum import total_continuum_absorption, thomson_scattering
    from jorg.constants import c_cgs
    JAX_AVAILABLE = True
    print("✅ JAX and Jorg successfully imported")
except ImportError as e:
    print(f"❌ JAX import error: {e}")
    JAX_AVAILABLE = False


def load_korg_reference_data(filepath="../korg_reference_data.json"):
    """Load Korg reference data from JSON file"""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded Korg reference data from {filepath}")
        return data
    except FileNotFoundError:
        print(f"❌ Could not find Korg reference data at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON file: {e}")
        return None


def run_jorg_calculation(korg_data):
    """Run Jorg calculation with same parameters as Korg"""
    
    if not JAX_AVAILABLE:
        print("❌ Cannot run Jorg calculation - JAX not available")
        return None
    
    print("🔄 Running Jorg calculation with Korg parameters...")
    
    # Extract parameters from Korg data
    frequencies = np.array(korg_data['frequencies'])
    temperature = korg_data['temperature']
    electron_density = korg_data['electron_density']
    
    # Number densities
    number_densities = {
        'H_I': korg_data['number_densities']['H_I'],
        'H_II': korg_data['number_densities']['H_II'],
        'He_I': korg_data['number_densities']['He_I'],
        'H2': korg_data['number_densities']['H2']
    }
    
    # Partition functions (matching Korg's simple functions)
    partition_functions = {
        'H_I': lambda log_T: 2.0,
        'He_I': lambda log_T: 1.0
    }
    
    print(f"  Temperature: {temperature} K")
    print(f"  Electron density: {electron_density:.2e} cm^-3")
    print(f"  H I density: {number_densities['H_I']:.2e} cm^-3")
    print(f"  Frequency range: {frequencies[0]:.2e} to {frequencies[-1]:.2e} Hz")
    
    # Calculate continuum absorption
    alpha_total = total_continuum_absorption(
        frequencies, temperature, electron_density,
        number_densities, partition_functions, True
    )
    
    # Calculate Thomson scattering
    alpha_thomson = thomson_scattering(electron_density)
    
    print("✅ Jorg calculation completed")
    
    return {
        'frequencies': frequencies,
        'alpha_total': np.array(alpha_total),
        'alpha_thomson': float(alpha_thomson),
        'temperature': temperature,
        'electron_density': electron_density,
        'number_densities': number_densities
    }


def compare_results(korg_data, jorg_data):
    """Compare Korg and Jorg results and compute statistics"""
    
    print("📊 Comparing Korg and Jorg results...")
    
    # Extract data
    korg_alpha = np.array(korg_data['alpha_total'])
    jorg_alpha = np.array(jorg_data['alpha_total'])
    wavelengths = np.array(korg_data['wavelengths_angstrom'])
    
    # Check array lengths match
    if len(korg_alpha) != len(jorg_alpha):
        print(f"⚠️  Array length mismatch: Korg {len(korg_alpha)}, Jorg {len(jorg_alpha)}")
        min_len = min(len(korg_alpha), len(jorg_alpha))
        korg_alpha = korg_alpha[:min_len]
        jorg_alpha = jorg_alpha[:min_len]
        wavelengths = wavelengths[:min_len]
    
    # Calculate differences
    abs_diff = np.abs(korg_alpha - jorg_alpha)
    rel_diff = abs_diff / (np.abs(korg_alpha) + 1e-30)  # Avoid division by zero
    
    # Calculate statistics
    stats = {
        'max_abs_diff': np.max(abs_diff),
        'mean_abs_diff': np.mean(abs_diff),
        'rms_abs_diff': np.sqrt(np.mean(abs_diff**2)),
        'max_rel_diff': np.max(rel_diff),
        'mean_rel_diff': np.mean(rel_diff),
        'rms_rel_diff': np.sqrt(np.mean(rel_diff**2)),
        'correlation': np.corrcoef(korg_alpha, jorg_alpha)[0, 1],
        'korg_range': [np.min(korg_alpha), np.max(korg_alpha)],
        'jorg_range': [np.min(jorg_alpha), np.max(jorg_alpha)]
    }
    
    # Thomson scattering comparison
    thomson_abs_diff = abs(korg_data['alpha_thomson'] - jorg_data['alpha_thomson'])
    thomson_rel_diff = thomson_abs_diff / korg_data['alpha_thomson']
    
    stats['thomson_abs_diff'] = thomson_abs_diff
    stats['thomson_rel_diff'] = thomson_rel_diff
    
    print("📈 Comparison Statistics:")
    print(f"  Maximum absolute difference: {stats['max_abs_diff']:.2e} cm^-1")
    print(f"  Mean absolute difference: {stats['mean_abs_diff']:.2e} cm^-1")
    print(f"  Maximum relative difference: {stats['max_rel_diff']*100:.3f}%")
    print(f"  Mean relative difference: {stats['mean_rel_diff']*100:.3f}%")
    print(f"  Correlation coefficient: {stats['correlation']:.6f}")
    print(f"  Thomson scattering difference: {thomson_rel_diff*100:.6f}%")
    
    return stats, wavelengths, korg_alpha, jorg_alpha, abs_diff, rel_diff


def create_comparison_plots(wavelengths, korg_alpha, jorg_alpha, abs_diff, rel_diff, 
                          korg_data, jorg_data, stats, save_dir="test_fig"):
    """Create comprehensive comparison plots"""
    
    print(f"🎨 Creating comparison plots in {save_dir}/...")
    
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up matplotlib parameters for publication-quality plots
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 150
    })
    
    # Plot 1: Direct comparison of absorption coefficients
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Main comparison plot
    ax1.loglog(wavelengths, korg_alpha, 'b-', linewidth=2.5, label='Korg (Julia)', alpha=0.8)
    ax1.loglog(wavelengths, jorg_alpha, 'r--', linewidth=2.5, label='Jorg (JAX)', alpha=0.8)
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Absorption Coefficient (cm⁻¹)')
    ax1.set_title('Continuum Absorption: Korg vs Jorg')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(wavelengths[0], wavelengths[-1])
    
    # Add Thomson scattering line
    thomson_level = korg_data['alpha_thomson']
    ax1.axhline(thomson_level, color='gray', linestyle=':', linewidth=1.5, 
                label=f'Thomson scattering ({thomson_level:.2e})', alpha=0.7)
    
    # Relative difference plot
    ax2.semilogx(wavelengths, rel_diff * 100, 'g-', linewidth=2)
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Relative Difference (%)')
    ax2.set_title('Relative Difference: |Korg - Jorg| / Korg')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(wavelengths[0], wavelengths[-1])
    
    # Add statistics text
    stats_text = f"""Max: {stats['max_rel_diff']*100:.3f}%
Mean: {stats['mean_rel_diff']*100:.3f}%
RMS: {stats['rms_rel_diff']*100:.3f}%"""
    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Absolute difference plot
    ax3.loglog(wavelengths, abs_diff, 'm-', linewidth=2)
    ax3.set_xlabel('Wavelength (Å)')
    ax3.set_ylabel('Absolute Difference (cm⁻¹)')
    ax3.set_title('Absolute Difference: |Korg - Jorg|')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(wavelengths[0], wavelengths[-1])
    
    # Scatter plot comparison
    ax4.loglog(korg_alpha, jorg_alpha, 'ko', markersize=4, alpha=0.7)
    min_val = min(np.min(korg_alpha), np.min(jorg_alpha))
    max_val = max(np.max(korg_alpha), np.max(jorg_alpha))
    ax4.loglog([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
               label='Perfect agreement')
    ax4.set_xlabel('Korg α (cm⁻¹)')
    ax4.set_ylabel('Jorg α (cm⁻¹)')
    ax4.set_title(f'Scatter Plot (r = {stats["correlation"]:.4f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/korg_jorg_comparison.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{save_dir}/korg_jorg_comparison.pdf', bbox_inches='tight')
    print(f"  📋 Main comparison plot saved: {save_dir}/korg_jorg_comparison.png")
    
    # Plot 2: Detailed analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals plot
    residuals = (jorg_alpha - korg_alpha) / korg_alpha * 100
    ax1.plot(wavelengths, residuals, 'b-', linewidth=2, alpha=0.8)
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.axhline(np.mean(residuals), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {np.mean(residuals):.3f}%')
    ax1.fill_between(wavelengths, 
                     np.mean(residuals) - np.std(residuals),
                     np.mean(residuals) + np.std(residuals),
                     alpha=0.2, color='red', 
                     label=f'±1σ: {np.std(residuals):.3f}%')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Residuals: (Jorg - Korg)/Korg (%)')
    ax1.set_title('Residual Analysis')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Histogram of relative differences
    ax2.hist(rel_diff * 100, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.axvline(stats['mean_rel_diff'] * 100, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {stats["mean_rel_diff"]*100:.3f}%')
    ax2.set_xlabel('Relative Difference (%)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Relative Differences')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/korg_jorg_analysis.png', dpi=150, bbox_inches='tight')
    print(f"  📊 Analysis plot saved: {save_dir}/korg_jorg_analysis.png")
    
    # Plot 3: Wavelength-dependent ratio
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    ratio = jorg_alpha / korg_alpha
    ax.semilogx(wavelengths, ratio, 'b-', linewidth=2.5, alpha=0.8)
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Perfect agreement')
    ax.fill_between(wavelengths, 0.99, 1.01, alpha=0.2, color='green', label='±1% agreement')
    ax.fill_between(wavelengths, 0.95, 1.05, alpha=0.1, color='yellow', label='±5% agreement')
    
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Ratio: Jorg / Korg')
    ax.set_title('Wavelength-dependent Agreement Between Jorg and Korg')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(wavelengths[0], wavelengths[-1])
    ax.set_ylim(0.95, 1.05)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/korg_jorg_ratio.png', dpi=150, bbox_inches='tight')
    print(f"  📈 Ratio plot saved: {save_dir}/korg_jorg_ratio.png")
    
    plt.close('all')  # Close all figures to free memory


def generate_comparison_report(korg_data, jorg_data, stats, save_dir="test_fig"):
    """Generate a detailed comparison report"""
    
    report_file = f"{save_dir}/comparison_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("KORG vs JORG CONTINUUM ABSORPTION COMPARISON REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("CALCULATION PARAMETERS:\n")
        f.write(f"  Temperature: {korg_data['temperature']} K\n")
        f.write(f"  Electron density: {korg_data['electron_density']:.2e} cm^-3\n")
        f.write(f"  H I density: {korg_data['number_densities']['H_I']:.2e} cm^-3\n")
        f.write(f"  H II density: {korg_data['number_densities']['H_II']:.2e} cm^-3\n")
        f.write(f"  He I density: {korg_data['number_densities']['He_I']:.2e} cm^-3\n")
        f.write(f"  H2 density: {korg_data['number_densities']['H2']:.2e} cm^-3\n")
        f.write(f"  Wavelength range: {korg_data['wavelengths_angstrom'][0]}-{korg_data['wavelengths_angstrom'][-1]} Å\n")
        f.write(f"  Number of points: {len(korg_data['wavelengths_angstrom'])}\n\n")
        
        f.write("ABSORPTION COEFFICIENT RANGES:\n")
        f.write(f"  Korg: {stats['korg_range'][0]:.2e} to {stats['korg_range'][1]:.2e} cm^-1\n")
        f.write(f"  Jorg: {stats['jorg_range'][0]:.2e} to {stats['jorg_range'][1]:.2e} cm^-1\n\n")
        
        f.write("COMPARISON STATISTICS:\n")
        f.write(f"  Maximum absolute difference: {stats['max_abs_diff']:.2e} cm^-1\n")
        f.write(f"  Mean absolute difference: {stats['mean_abs_diff']:.2e} cm^-1\n")
        f.write(f"  RMS absolute difference: {stats['rms_abs_diff']:.2e} cm^-1\n")
        f.write(f"  Maximum relative difference: {stats['max_rel_diff']*100:.3f}%\n")
        f.write(f"  Mean relative difference: {stats['mean_rel_diff']*100:.3f}%\n")
        f.write(f"  RMS relative difference: {stats['rms_rel_diff']*100:.3f}%\n")
        f.write(f"  Correlation coefficient: {stats['correlation']:.6f}\n\n")
        
        f.write("THOMSON SCATTERING COMPARISON:\n")
        f.write(f"  Korg: {korg_data['alpha_thomson']:.2e} cm^-1\n")
        f.write(f"  Jorg: {jorg_data['alpha_thomson']:.2e} cm^-1\n")
        f.write(f"  Absolute difference: {stats['thomson_abs_diff']:.2e} cm^-1\n")
        f.write(f"  Relative difference: {stats['thomson_rel_diff']*100:.6f}%\n\n")
        
        f.write("ASSESSMENT:\n")
        if stats['max_rel_diff'] < 0.01:
            f.write("  ✅ EXCELLENT: Differences less than 1%\n")
        elif stats['max_rel_diff'] < 0.05:
            f.write("  ✅ GOOD: Differences less than 5%\n")
        elif stats['max_rel_diff'] < 0.1:
            f.write("  ⚠️  ACCEPTABLE: Differences less than 10%\n")
        else:
            f.write("  ❌ POOR: Significant differences detected\n")
        
        f.write(f"\nCORRELATION ASSESSMENT:\n")
        if stats['correlation'] > 0.999:
            f.write("  ✅ EXCELLENT: Very high correlation\n")
        elif stats['correlation'] > 0.99:
            f.write("  ✅ GOOD: High correlation\n")
        elif stats['correlation'] > 0.9:
            f.write("  ⚠️  ACCEPTABLE: Moderate correlation\n")
        else:
            f.write("  ❌ POOR: Low correlation\n")
    
    print(f"📄 Detailed report saved: {report_file}")


def main():
    """Main comparison function"""
    
    print("🌟 Korg vs Jorg Continuum Absorption Comparison")
    print("=" * 60)
    
    # Load Korg reference data
    korg_data = load_korg_reference_data()
    if korg_data is None:
        print("❌ Cannot proceed without Korg reference data")
        return
    
    # Run Jorg calculation
    jorg_data = run_jorg_calculation(korg_data)
    if jorg_data is None:
        print("❌ Cannot proceed without Jorg calculation")
        return
    
    # Compare results
    stats, wavelengths, korg_alpha, jorg_alpha, abs_diff, rel_diff = compare_results(korg_data, jorg_data)
    
    # Create comparison plots
    create_comparison_plots(wavelengths, korg_alpha, jorg_alpha, abs_diff, rel_diff,
                          korg_data, jorg_data, stats)
    
    # Generate detailed report
    generate_comparison_report(korg_data, jorg_data, stats)
    
    print("\n" + "=" * 60)
    print("🎯 SUMMARY:")
    print(f"   Maximum relative difference: {stats['max_rel_diff']*100:.3f}%")
    print(f"   Mean relative difference: {stats['mean_rel_diff']*100:.3f}%")
    print(f"   Correlation coefficient: {stats['correlation']:.6f}")
    
    if stats['max_rel_diff'] < 0.01:
        print("   ✅ EXCELLENT agreement between Korg and Jorg!")
    elif stats['max_rel_diff'] < 0.05:
        print("   ✅ GOOD agreement between Korg and Jorg!")
    elif stats['max_rel_diff'] < 0.1:
        print("   ⚠️  ACCEPTABLE agreement between Korg and Jorg")
    else:
        print("   ❌ Significant differences between Korg and Jorg")
    
    print(f"\n📁 All results saved in: test_fig/")
    print("   - Comparison plots: korg_jorg_comparison.png")
    print("   - Analysis plots: korg_jorg_analysis.png")
    print("   - Ratio plot: korg_jorg_ratio.png")
    print("   - Detailed report: comparison_report.txt")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()