"""
Generate comparison plots for all 10 UVES-POP stars
====================================================
"""

import sys
sys.path.insert(0, 'jorg/src')

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from jorg.uves_pop_comparison import compare_star

# Create output directory
output_dir = Path('/Users/jdli/Project/Korg.jl/jorg/valid/plots')
output_dir.mkdir(exist_ok=True)

# Define wavelength ranges for different spectral features
ranges = {
    'H_alpha': (6550, 6580),
    'H_beta': (4850, 4880),
    'Mg_triplet': (5150, 5200),
    'visible': (4800, 6800),
}

print("="*70)
print("Generating UVES-POP vs Jorg Comparison Plots")
print("="*70)

# Compare each star
for star_id in range(1, 11):
    print(f"\n{'='*70}")
    print(f"Star {star_id}")
    print(f"{'='*70}")

    try:
        # Use a medium wavelength range for the plot
        result = compare_star(
            star_id=star_id,
            wavelength_range=(4800, 6800),
            resolution=20000,
            verbose=True
        )

        star = result['star']
        wl_obs = result['observed']['wavelength']
        flux_obs = result['observed']['flux_normalized']
        flux_syn = result['interpolated']['flux']
        metrics = result['metrics']

        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f"{star['name']}: {star['regime']}\n"
                     f"Teff={star['teff']:.0f} K, logg={star['logg']:.2f}, "
                     f"[Fe/H]={star['feh']:.2f}, [α/Fe]={star['afeh']:.2f}",
                     fontsize=12)

        # Plot 1: Full comparison (4800-6800 Å)
        ax = axes[0]
        ax.plot(wl_obs, flux_obs, 'k-', alpha=0.7, linewidth=0.5, label='Observed')
        ax.plot(wl_obs, flux_syn, 'r-', alpha=0.7, linewidth=0.5, label='Synthetic')
        ax.set_ylabel('Normalized Flux')
        ax.set_title('Full Range (4800-6800 Å)', fontsize=10)
        ax.legend(loc='upper right')
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

        # Plot 2: Zoom on H-beta region
        ax = axes[1]
        mask_hb = (wl_obs >= 4850) & (wl_obs <= 4880)
        ax.plot(wl_obs[mask_hb], flux_obs[mask_hb], 'k-', alpha=0.8, linewidth=1, label='Observed')
        ax.plot(wl_obs[mask_hb], flux_syn[mask_hb], 'r-', alpha=0.8, linewidth=1, label='Synthetic')
        ax.set_ylabel('Normalized Flux')
        ax.set_title('H-β Region (4850-4880 Å)', fontsize=10)
        ax.legend(loc='upper right')
        ax.invert_yaxis()
        ax.grid(alpha=0.3)

        # Plot 3: Residuals
        ax = axes[2]
        finite = result['interpolated']['finite_mask']
        residuals = flux_obs - flux_syn
        ax.plot(wl_obs[finite], residuals[finite], 'b-', alpha=0.5, linewidth=0.5)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Residual (Obs - Syn)')
        ax.set_title(f'Residuals (RMS={metrics["rms"]:.3f}, Rel RMS={metrics["relative_rms"]*100:.1f}%)', fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()

        # Save figure
        safe_name = star['name'].replace('*', 'star').replace(' ', '_')
        output_path = output_dir / f'star_{star_id:02d}_{safe_name}_comparison.png'
        plt.savefig(output_path, dpi=150)
        print(f"\nSaved plot to: {output_path}")
        plt.close()

    except Exception as e:
        print(f"\nERROR: Failed to generate plot for star {star_id}: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*70}")
print("Done! Plots saved to: {output_dir}")
print(f"{'='*70}")
