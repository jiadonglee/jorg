#!/usr/bin/env python3
"""
Stellar Type Comparison with Jorg (Python/JAX)
==============================================

Compare synthesis for different stellar types:
1. Solar-like star: Teff=5771K, logg=4.44, [M/H]=0.0
2. Arcturus-like star: Teff=4250K, logg=1.4, [Fe/H]=-0.5
3. Metal-poor K giant: Teff=4500K, logg=1.5, [Fe/H]=-2.5
"""

import sys
import os
import numpy as np
import time
from datetime import datetime

# Add Jorg to path
sys.path.insert(0, '../../src')

from jorg.synthesis import synth
from jorg.lines.linelist import read_linelist

# Import plotting libraries
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

print("🌟 STELLAR TYPE COMPARISON - JORG (Python/JAX)")
print("=" * 55)

# Define stellar parameters
stellar_types = [
    {
        'name': 'Solar-like',
        'Teff': 5771,
        'logg': 4.44,
        'm_H': 0.0,
        'description': 'G2V dwarf (Sun-like)'
    },
    {
        'name': 'Arcturus-like', 
        'Teff': 4250,
        'logg': 1.4,
        'm_H': -0.5,
        'description': 'K1.5 III giant'
    },
    {
        'name': 'Metal-poor K giant',
        'Teff': 4500,
        'logg': 1.5, 
        'm_H': -2.5,
        'description': 'K2 III halo giant'
    }
]

# Synthesis parameters
wavelength_range = (5000, 5200)  # Å
vmic = 1.0  # km/s

print(f"Synthesis parameters:")
print(f"  Wavelength range: {wavelength_range} Å")
print(f"  Microturbulence: {vmic} km/s")
print(f"  Rectification: enabled")
print()

# Load line list
print("📋 Loading VALD line list...")
try:
    linelist_path = "../../../data/linelists/vald_extract_stellar_solar_threshold001.vald"
    if not os.path.exists(linelist_path):
        linelist_path = "../../../misc/Tutorial notebooks/basics/linelist.vald"
    
    if os.path.exists(linelist_path):
        linelist = read_linelist(linelist_path)
        print(f"✅ Loaded line list: {len(linelist)} lines")
    else:
        print("⚠️  No line list found, using continuum-only synthesis")
        linelist = None
except Exception as e:
    print(f"⚠️  Line list loading failed: {e}")
    print("   Using continuum-only synthesis")
    linelist = None

print()

# Store results
results = {}

# Synthesize each stellar type
for i, star in enumerate(stellar_types, 1):
    print(f"{i}. {star['name']} ({star['description']})")
    print(f"   Teff={star['Teff']}K, logg={star['logg']}, [M/H]={star['m_H']}")
    
    try:
        # Run synthesis
        start_time = time.time()
        
        wavelengths = np.arange(wavelength_range[0], wavelength_range[1], 0.01)
        wl, flux, cntm = synth(
            Teff=star['Teff'],
            logg=star['logg'], 
            m_H=star['m_H'],
            wavelengths=wavelengths,
            linelist=linelist,
            rectify=True
        )
        
        elapsed = time.time() - start_time
        
        # Store results
        results[star['name']] = {
            'wavelengths': wl,
            'flux': flux,
            'continuum': cntm,
            'parameters': star,
            'synthesis_time': elapsed
        }
        
        # Analyze spectrum
        min_flux = np.min(flux)
        max_absorption = (1.0 - min_flux) * 100
        line_features = np.sum(flux < 0.95)
        
        print(f"   ✅ Synthesis complete: {elapsed:.2f}s")
        print(f"      Flux range: {min_flux:.3f} - {np.max(flux):.3f}")
        print(f"      Deepest line: {max_absorption:.1f}% absorption")
        print(f"      Line features: {line_features} pixels >5% depth")
        
    except Exception as e:
        print(f"   ❌ Synthesis failed: {e}")
        results[star['name']] = None
    
    print()

# Save results to files
print("💾 SAVING SYNTHESIS RESULTS")
print("-" * 35)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

for star_name, result in results.items():
    if result is None:
        continue
        
    # Create safe filename
    safe_name = star_name.lower().replace(' ', '_').replace('-', '_')
    spectrum_file = f"jorg_{safe_name}_spectrum_{timestamp}.txt"
    
    # Save spectrum
    with open(spectrum_file, 'w') as f:
        f.write(f"# Jorg (Python/JAX) Stellar Synthesis - {star_name}\n")
        f.write(f"# Generated: {datetime.now()}\n")
        f.write(f"# Parameters: Teff={result['parameters']['Teff']}K, ")
        f.write(f"logg={result['parameters']['logg']}, [M/H]={result['parameters']['m_H']}\n")
        f.write(f"# Synthesis time: {result['synthesis_time']:.2f} seconds\n")
        f.write(f"# Line list: {len(linelist) if linelist else 0} lines\n")
        f.write(f"#\n")
        f.write(f"# Wavelength(Å)    NormalizedFlux\n")
        
        for wl, flux in zip(result['wavelengths'], result['flux']):
            f.write(f"{wl:12.4f}  {flux:12.6f}\n")
    
    file_size = os.path.getsize(spectrum_file) / 1024
    print(f"✅ {star_name}: {spectrum_file} ({file_size:.1f} KB)")

# Create comparison file
comparison_file = f"jorg_stellar_comparison_{timestamp}.txt"
with open(comparison_file, 'w') as f:
    f.write(f"# Jorg Stellar Type Comparison Summary\n")
    f.write(f"# Generated: {datetime.now()}\n")
    f.write(f"#\n")
    f.write(f"# Star Type           Teff(K)  logg   [M/H]  SynthTime(s)  MinFlux  MaxAbsorption(%)\n")
    
    for star_name, result in results.items():
        if result is None:
            continue
            
        params = result['parameters']
        min_flux = np.min(result['flux'])
        max_abs = (1.0 - min_flux) * 100
        
        f.write(f"{star_name:18s}  {params['Teff']:5d}  {params['logg']:5.2f}  ")
        f.write(f"{params['m_H']:5.1f}  {result['synthesis_time']:10.2f}  ")
        f.write(f"{min_flux:7.3f}  {max_abs:13.1f}\n")

print(f"✅ Comparison summary: {comparison_file}")

print()
print("📊 STELLAR TYPE ANALYSIS")
print("-" * 25)

# Compare spectral properties
if len([r for r in results.values() if r is not None]) >= 2:
    print("Relative spectral differences:")
    
    # Find common wavelength range
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) >= 2:
        star_names = list(valid_results.keys())
        
        for i in range(len(star_names)):
            for j in range(i+1, len(star_names)):
                star1, star2 = star_names[i], star_names[j]
                result1, result2 = valid_results[star1], valid_results[star2]
                
                # Interpolate to common grid for comparison
                from scipy.interpolate import interp1d
                
                # Use star1's wavelength grid as reference
                interp_func = interp1d(result2['wavelengths'], result2['flux'], 
                                     bounds_error=False, fill_value=np.nan)
                flux2_interp = interp_func(result1['wavelengths'])
                
                # Calculate differences where both are valid
                valid_mask = ~np.isnan(flux2_interp)
                if np.sum(valid_mask) > 100:  # Need reasonable overlap
                    flux_diff = np.abs(result1['flux'][valid_mask] - flux2_interp[valid_mask])
                    mean_diff = np.mean(flux_diff)
                    max_diff = np.max(flux_diff)
                    rms_diff = np.sqrt(np.mean(flux_diff**2))
                    
                    print(f"  {star1} vs {star2}:")
                    print(f"    Mean difference: {mean_diff:.4f}")
                    print(f"    Max difference: {max_diff:.4f}")  
                    print(f"    RMS difference: {rms_diff:.4f}")
                    print(f"    Valid points: {np.sum(valid_mask)}")

# Create plots if we have results
if len([r for r in results.values() if r is not None]) >= 2:
    print()
    print("📈 CREATING COMPARISON PLOTS")
    print("-" * 30)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot 1: Full wavelength range
    ax1.set_title('Jorg Stellar Type Comparison - Full Spectrum', fontsize=14, fontweight='bold')
    
    for i, (star_name, result) in enumerate(results.items()):
        if result is None:
            continue
        
        params = result['parameters']
        label = f"{star_name}\n(T={params['Teff']}K, logg={params['logg']}, [M/H]={params['m_H']})"
        
        ax1.plot(result['wavelengths'], result['flux'], 
                label=label, linewidth=1.5, color=colors[i % len(colors)])
    
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Normalized Flux')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.05)
    
    # Plot 2: Zoomed section showing differences
    zoom_start, zoom_end = 5050, 5150
    ax2.set_title(f'Zoomed Comparison ({zoom_start}-{zoom_end} Å)', fontsize=14, fontweight='bold')
    
    for i, (star_name, result) in enumerate(results.items()):
        if result is None:
            continue
            
        # Find wavelength mask for zoom region
        wl_mask = (result['wavelengths'] >= zoom_start) & (result['wavelengths'] <= zoom_end)
        
        params = result['parameters']
        label = f"{star_name} (T={params['Teff']}K)"
        
        ax2.plot(result['wavelengths'][wl_mask], result['flux'][wl_mask],
                label=label, linewidth=2, color=colors[i % len(colors)])
    
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Normalized Flux')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.0, 1.05)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = f"jorg_stellar_comparison_plot_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    file_size = os.path.getsize(plot_file) / 1024
    print(f"✅ Comparison plot saved: {plot_file} ({file_size:.1f} KB)")
    
    # Create individual stellar type comparison plot
    if len([r for r in results.values() if r is not None]) == 3:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        
        star_names = list(results.keys())
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        # Calculate spectral differences between stellar types
        differences = {}
        for i, star1 in enumerate(star_names):
            for j, star2 in enumerate(star_names):
                if i < j and star1 in valid_results and star2 in valid_results:
                    result1, result2 = valid_results[star1], valid_results[star2]
                    
                    # Interpolate to common grid
                    interp_func = interp1d(result2['wavelengths'], result2['flux'],
                                         bounds_error=False, fill_value=np.nan)
                    flux2_interp = interp_func(result1['wavelengths'])
                    
                    # Calculate difference
                    valid_mask = ~np.isnan(flux2_interp)
                    flux_diff = result1['flux'] - flux2_interp
                    differences[f"{star1} - {star2}"] = {
                        'wavelengths': result1['wavelengths'][valid_mask],
                        'difference': flux_diff[valid_mask]
                    }
        
        # Plot differences
        for i, (diff_name, diff_data) in enumerate(differences.items()):
            ax.plot(diff_data['wavelengths'], diff_data['difference'],
                   label=diff_name, linewidth=1.5, color=colors[i % len(colors)])
        
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Flux Difference')
        ax.set_title('Stellar Type Spectral Differences (Jorg)', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save difference plot
        diff_plot_file = f"jorg_stellar_differences_plot_{timestamp}.png"
        plt.savefig(diff_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        file_size = os.path.getsize(diff_plot_file) / 1024
        print(f"✅ Difference plot saved: {diff_plot_file} ({file_size:.1f} KB)")

print()
print("🎉 JORG STELLAR TYPE COMPARISON COMPLETE!")
print(f"   Results saved with timestamp: {timestamp}")
print(f"   Ready for comparison with Korg.jl results")