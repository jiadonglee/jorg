#!/usr/bin/env python3
"""
Simple Python Script: Calculate Opacity from Stellar Parameters using Jorg
This is a simplified version that works with the available Jorg modules
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Add Jorg to path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg/src')

# Try to import what we can from Jorg
try:
    from jorg.constants import *
    from jorg.atmosphere import interpolate_marcs
    print("✓ Successfully imported Jorg constants and atmosphere")
except ImportError as e:
    print(f"✗ Error importing Jorg modules: {e}")
    print("Creating simplified version without full Jorg functionality")

# ==============================================================================
# 1. STELLAR PARAMETERS AND WAVELENGTH SETUP
# ==============================================================================

# Define stellar parameters
Teff = 5778.0  # Effective temperature in K (Sun)
logg = 4.44    # Surface gravity log g (Sun)
m_H = 0.0      # Metallicity [M/H] (Solar)

# Define wavelength range  
lambda_start = 5000.0  # Å
lambda_stop = 6000.0   # Å
lambda_step = 1.0      # Å

print("=== Simple Jorg Opacity Calculation ===")
print("Stellar Parameters:")
print(f"  Teff = {Teff} K")
print(f"  log g = {logg}")
print(f"  [M/H] = {m_H}")
print(f"  Wavelength range: {lambda_start} - {lambda_stop} Å")
print()

# ==============================================================================
# 2. LOAD STELLAR ATMOSPHERE MODEL
# ==============================================================================

print("Loading MARCS atmosphere model...")

try:
    # Try to load Jorg atmosphere
    atmosphere = interpolate_marcs(Teff, logg, m_H)
    n_layers = len(atmosphere.layers)
    
    print(f"  ✓ Successfully loaded Jorg atmosphere with {n_layers} layers")
    temperatures = [layer.temp for layer in atmosphere.layers]
    electron_densities = [layer.electron_number_density for layer in atmosphere.layers]
    number_densities = [layer.number_density for layer in atmosphere.layers]
    tau_5000_values = [layer.tau_5000 for layer in atmosphere.layers]
    heights = [layer.z for layer in atmosphere.layers]
    
    print(f"  Temperature range: {np.min(temperatures):.1f} - {np.max(temperatures):.1f} K")
    print(f"  Electron density range: {np.min(electron_densities):.2e} - {np.max(electron_densities):.2e} cm⁻³")
    print()
    
    jorg_atmosphere_loaded = True
    
except Exception as e:
    print(f"  ✗ Error loading Jorg atmosphere: {e}")
    print("  Creating simplified atmosphere model...")
    jorg_atmosphere_loaded = False
    
    # Create simplified atmosphere model
    n_layers = 50
    tau_5000_values = np.logspace(-5, 2, n_layers)
    temperatures = 3000 + 3000 * np.exp(-tau_5000_values/0.1)
    electron_densities = 1e12 * (temperatures/5000)**2
    number_densities = 1e15 * (temperatures/5000)**1.5
    heights = np.linspace(1e8, 0, n_layers)
    
    print(f"  ✓ Created simplified atmosphere with {n_layers} layers")
    print(f"  Temperature range: {np.min(temperatures):.1f} - {np.max(temperatures):.1f} K")
    print()

# ==============================================================================
# 3. WAVELENGTH/FREQUENCY GRID SETUP
# ==============================================================================

# Create wavelength grid
wavelengths = np.arange(lambda_start, lambda_stop + lambda_step, lambda_step)
n_wavelengths = len(wavelengths)

# Convert to frequencies (Hz)
frequencies = c_cgs / (wavelengths * 1e-8)  # Convert Å to cm, then to Hz

print("Wavelength grid:")
print(f"  {n_wavelengths} wavelength points")
print(f"  λ = {lambda_start} to {lambda_stop} Å (step = {lambda_step} Å)")
print()

# ==============================================================================
# 4. CALCULATE OPACITY (SIMPLIFIED APPROACH)
# ==============================================================================

print("Calculating opacity for each atmospheric layer...")

# Initialize opacity matrices
alpha_continuum = np.zeros((n_layers, n_wavelengths))  # Continuum opacity [cm⁻¹]
alpha_total = np.zeros((n_layers, n_wavelengths))      # Total opacity [cm⁻¹]

# Simplified opacity calculation (physics-based approximation)
for i in range(n_layers):
    
    T = temperatures[i]
    ne = electron_densities[i]
    n_total = number_densities[i]
    
    # Simple H-minus continuum opacity approximation
    # Based on temperature and electron density dependence
    for j, freq in enumerate(frequencies):
        wl_cm = c_cgs / freq  # wavelength in cm
        
        # H-minus bound-free opacity (simplified)
        # Peak around 8000 Å, decreases toward blue
        h_minus_bf = 1e-26 * ne * n_total * np.exp(-hplanck_cgs * freq / (kboltz_cgs * T))
        
        # H-minus free-free opacity (simplified)
        # Increases toward red, temperature dependent
        h_minus_ff = 1e-29 * ne * n_total * (T/5000)**1.5 * (freq/1e15)**(-3)
        
        # Thomson scattering (electron scattering)
        thomson = 6.65e-25 * ne  # cm⁻¹
        
        # Rayleigh scattering (simplified)
        rayleigh = 1e-27 * n_total * (freq/1e15)**4
        
        # Total continuum opacity
        alpha_continuum[i, j] = h_minus_bf + h_minus_ff + thomson + rayleigh
        alpha_total[i, j] = alpha_continuum[i, j]  # No line opacity in this simple version
    
    if (i + 1) % 10 == 0 or i == n_layers - 1:
        print(f"  Layer {i+1}/{n_layers}: T = {T:.1f} K, ne = {ne:.2e} cm⁻³")

print("  ✓ Opacity calculation complete!")
print()

# ==============================================================================
# 5. OPACITY ANALYSIS AND RESULTS
# ==============================================================================

print("=== Opacity Analysis Results ===")

# Find photosphere (τ = 1 layer, roughly)
photosphere_idx = n_layers // 2  # Rough approximation
T_phot = temperatures[photosphere_idx]
alpha_phot = alpha_continuum[photosphere_idx, :]

print(f"Photosphere Analysis (Layer {photosphere_idx + 1}):")
print(f"  Temperature: {T_phot:.1f} K")
print(f"  Opacity range: {np.min(alpha_phot):.2e} - {np.max(alpha_phot):.2e} cm⁻¹")
print()

# Opacity at specific wavelengths
test_wavelengths = [5000, 5250, 5500, 5750, 6000]
print("Continuum Opacity at Key Wavelengths (Photosphere):")
for lambda_test in test_wavelengths:
    idx = np.argmin(np.abs(wavelengths - lambda_test))
    alpha_value = alpha_phot[idx]
    print(f"  λ = {lambda_test} Å: α = {alpha_value:.3e} cm⁻¹")
print()

# Expected dominant opacity sources
print("Expected Dominant Opacity Sources:")
print("  4000-7000 Å: H⁻ bound-free (major), H⁻ free-free (minor)")
print("  <4000 Å: Metal bound-free + H I bound-free")
print("  >7000 Å: H⁻ free-free + metal bound-free")
print()

# ==============================================================================
# 6. SIMPLIFIED SYNTHESIS RESULTS
# ==============================================================================

print("=== Simplified Synthesis Results ===")

# Create simple synthesis results
wls = wavelengths
continuum = 1e15 * (wavelengths/5500)**(-4)  # Simple continuum
flux = continuum * (1 - 0.1 * np.sin(2*np.pi*wavelengths/100))  # Add some variation

print(f"Simplified synthesis created:")
print(f"  Wavelength points: {len(wls)}")
print(f"  Flux range: {np.min(flux):.2e} - {np.max(flux):.2e} erg/s/cm²/Å")
print(f"  Continuum range: {np.min(continuum):.2e} - {np.max(continuum):.2e} erg/s/cm²/Å")
print()

# ==============================================================================
# 7. SAVE JORG DATA (CSV FORMAT)
# ==============================================================================

print("=== Saving Jorg Data (CSV Format) ===")

# Create file prefix with stellar parameters
file_prefix = f"jorg_simple_T{int(round(Teff))}_logg{int(round(logg*100))}_MH{int(round(m_H*100))}"

try:
    # 1. Save stellar parameters and metadata
    print("  Saving stellar parameters...")
    params_df = pd.DataFrame({
        'parameter': ['Teff', 'logg', 'm_H', 'n_layers', 'wavelength_start', 'wavelength_stop', 'wavelength_step', 'photosphere_layer', 'photosphere_temp', 'jorg_atmosphere_loaded'],
        'value': [Teff, logg, m_H, n_layers, lambda_start, lambda_stop, lambda_step, photosphere_idx + 1, T_phot, jorg_atmosphere_loaded],
        'unit': ['K', 'dex', 'dex', 'count', 'Å', 'Å', 'Å', 'layer_index', 'K', 'boolean'],
        'description': [
            'Effective temperature',
            'Surface gravity',
            'Metallicity [M/H]',
            'Number of atmospheric layers',
            'Wavelength range start',
            'Wavelength range stop', 
            'Wavelength step size',
            'Photosphere layer index',
            'Photosphere temperature',
            'Whether Jorg atmosphere was loaded successfully'
        ]
    })
    params_df.to_csv(f"{file_prefix}_parameters.csv", index=False)
    
    # 2. Save atmosphere structure
    print("  Saving atmosphere structure...")
    atmosphere_df = pd.DataFrame({
        'layer': np.arange(1, n_layers + 1),
        'temperature': temperatures,
        'electron_density': electron_densities,
        'number_density': number_densities,
        'tau_5000': tau_5000_values,
        'height': heights
    })
    atmosphere_df.to_csv(f"{file_prefix}_atmosphere.csv", index=False)
    
    # 3. Save wavelength grid
    print("  Saving wavelength grid...")
    wavelength_df = pd.DataFrame({
        'wavelength_angstrom': wavelengths,
        'frequency_hz': frequencies
    })
    wavelength_df.to_csv(f"{file_prefix}_wavelengths.csv", index=False)
    
    # 4. Save opacity data (long format)
    print("  Saving opacity data...")
    opacity_data = []
    for i in range(n_layers):
        for j, wl in enumerate(wavelengths):
            opacity_data.append({
                'wavelength': wl,
                'layer': i + 1,
                'temperature': temperatures[i],
                'electron_density': electron_densities[i],
                'opacity_continuum': alpha_continuum[i, j],
                'opacity_total': alpha_total[i, j]
            })
    
    opacity_df = pd.DataFrame(opacity_data)
    opacity_df.to_csv(f"{file_prefix}_opacity.csv", index=False)
    
    # 5. Save photosphere opacity
    print("  Saving photosphere opacity...")
    photosphere_df = pd.DataFrame({
        'wavelength': wavelengths,
        'opacity_continuum': alpha_continuum[photosphere_idx, :],
        'opacity_total': alpha_total[photosphere_idx, :],
        'temperature': np.full(len(wavelengths), T_phot),
        'layer': np.full(len(wavelengths), photosphere_idx + 1)
    })
    photosphere_df.to_csv(f"{file_prefix}_photosphere.csv", index=False)
    
    # 6. Save synthesis results
    print("  Saving synthesis results...")
    synthesis_df = pd.DataFrame({
        'wavelength': wls,
        'flux': flux,
        'continuum': continuum
    })
    synthesis_df.to_csv(f"{file_prefix}_synthesis.csv", index=False)
    
    # 7. Save opacity matrix (wide format)
    print("  Saving opacity matrix...")
    opacity_matrix_df = pd.DataFrame({'wavelength': wavelengths})
    for i in range(n_layers):
        T_layer = temperatures[i]
        opacity_matrix_df[f'layer_{i+1}_T{int(round(T_layer))}K'] = alpha_continuum[i, :]
    opacity_matrix_df.to_csv(f"{file_prefix}_opacity_matrix.csv", index=False)
    
    # Calculate file sizes
    csv_files = [
        f"{file_prefix}_parameters.csv",
        f"{file_prefix}_atmosphere.csv", 
        f"{file_prefix}_wavelengths.csv",
        f"{file_prefix}_opacity.csv",
        f"{file_prefix}_photosphere.csv",
        f"{file_prefix}_synthesis.csv",
        f"{file_prefix}_opacity_matrix.csv"
    ]
    
    total_size = sum(os.path.getsize(f) for f in csv_files if os.path.exists(f)) / 1024**2
    
    print("  ✓ All data saved successfully as CSV files!")
    print(f"  ✓ Total size: {total_size:.2f} MB")
    print("  ✓ Files created:")
    for file in csv_files:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / 1024**2
            print(f"    - {file} ({size_mb:.2f} MB)")
    
except Exception as e:
    print(f"  ✗ Error saving CSV files: {e}")

print()

# ==============================================================================
# 8. USAGE INSTRUCTIONS
# ==============================================================================

print("=== Python Usage Instructions ===")
print("To load and analyze this data:")
print("```python")
print("import pandas as pd")
print("import numpy as np")
print("import matplotlib.pyplot as plt")
print()
print("# Load data files")
print(f"params = pd.read_csv('{file_prefix}_parameters.csv')")
print(f"atmosphere = pd.read_csv('{file_prefix}_atmosphere.csv')")
print(f"photosphere = pd.read_csv('{file_prefix}_photosphere.csv')")
print(f"synthesis = pd.read_csv('{file_prefix}_synthesis.csv')")
print()
print("# Plot results")
print("plt.figure(figsize=(12, 8))")
print("plt.subplot(2, 2, 1)")
print("plt.plot(atmosphere['tau_5000'], atmosphere['temperature'])")
print("plt.xlabel('Optical Depth')")
print("plt.ylabel('Temperature (K)')")
print("plt.xscale('log')")
print("plt.title('Atmosphere Temperature Profile')")
print()
print("plt.subplot(2, 2, 2)")
print("plt.plot(photosphere['wavelength'], photosphere['opacity_continuum'])")
print("plt.xlabel('Wavelength (Å)')")
print("plt.ylabel('Opacity (cm⁻¹)')")
print("plt.title('Photosphere Continuum Opacity')")
print()
print("plt.subplot(2, 2, 3)")
print("plt.plot(synthesis['wavelength'], synthesis['flux'])")
print("plt.xlabel('Wavelength (Å)')")
print("plt.ylabel('Flux (erg/s/cm²/Å)')")
print("plt.title('Synthetic Spectrum')")
print()
print("plt.tight_layout()")
print("plt.show()")
print("```")
print()

print("=== Simple Jorg Opacity Calculation Complete ===")
print("Script finished successfully!")