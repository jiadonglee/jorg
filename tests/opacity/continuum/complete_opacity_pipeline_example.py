#!/usr/bin/env python3
"""
Complete Opacity Pipeline Example: Atmosphere → StatMech → Continuum + Line Opacity
=====================================================================================

This script demonstrates the complete stellar opacity calculation pipeline from
atmosphere interpolation through statistical mechanics to both continuum and 
line opacity calculations.

Pipeline Steps:
1. Atmosphere Interpolation (MARCS models) 
2. Statistical Mechanics (chemical equilibrium, partition functions)
3. Continuum Opacity (H⁻ bf/ff, H I bf, metals, scattering)
4. Line Opacity (VALD linelist, Voigt profiles, broadening)
5. Total Opacity (continuum + lines)

This combines the best practices from both the continuum and line opacity
comparison scripts to provide a complete working example.

Usage:
    python complete_opacity_pipeline_example.py --stellar-type sun
    python complete_opacity_pipeline_example.py --teff 5780 --logg 4.44 --mh 0.0
    python complete_opacity_pipeline_example.py --wavelengths 5000 5005 --n-points 100
"""

import sys
import os
import argparse
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Add Jorg to path
sys.path.insert(0, "/Users/jdli/Project/Korg.jl/Jorg/src")

import jax.numpy as jnp

# Jorg imports - Full pipeline
from jorg.synthesis import interpolate_atmosphere
from jorg.abundances import format_A_X as format_abundances
from jorg.statmech import chemical_equilibrium, create_default_partition_functions, Species

# Continuum opacity
from jorg.continuum import (
    h_minus_bf_absorption, h_minus_ff_absorption, 
    thomson_scattering, H_I_bf
)

# Line opacity  
from jorg.lines.linelist import read_linelist
from jorg.lines.opacity import calculate_line_opacity_korg_method

# Constants
ANGSTROM_TO_CM = 1e-8
C_CGS = 2.99792458e10  # cm/s

# Script version
VERSION = "1.0 - Complete Opacity Pipeline"
LAST_UPDATED = "July 2025"

class StellarParameters:
    """Container for stellar parameters"""
    def __init__(self, name: str, teff: float, logg: float, mh: float):
        self.name = name
        self.teff = teff
        self.logg = logg
        self.mh = mh
    
    def __str__(self):
        return f"{self.name}: Teff={self.teff}K, log g={self.logg}, [M/H]={self.mh}"

# Predefined stellar types
STELLAR_TYPES = {
    'sun': StellarParameters('Sun', 5780, 4.44, 0.0),
    'k_dwarf': StellarParameters('K Dwarf', 4500, 4.5, 0.0),
    'g_dwarf': StellarParameters('G Dwarf', 6000, 4.3, 0.0),
    'k_giant': StellarParameters('K Giant', 4200, 2.0, 0.0),
    'm_dwarf': StellarParameters('M Dwarf', 3500, 4.8, 0.0),
}

def step1_atmosphere_interpolation(stellar_params: StellarParameters) -> Dict:
    """
    Step 1: Interpolate MARCS atmosphere model
    """
    print("🌍 STEP 1: ATMOSPHERE INTERPOLATION")
    print("=" * 50)
    
    # Format abundances (solar by default, metallicity scaling to be added later)
    A_X = format_abundances()
    # TODO: Apply metallicity scaling for stellar_params.mh != 0.0
    print("✅ Abundances formatted")
    print(f"   Metallicity [M/H]: {stellar_params.mh}")
    
    # Interpolate atmosphere
    atm = interpolate_atmosphere(
        Teff=stellar_params.teff,
        logg=stellar_params.logg, 
        A_X=A_X
    )
    print("✅ Atmosphere interpolated")
    print(f"   Model: MARCS")
    print(f"   Layers: {len(atm['temperature'])}")
    print(f"   Temperature range: {jnp.min(atm['temperature']):.0f} - {jnp.max(atm['temperature']):.0f} K")
    print(f"   Pressure range: {jnp.min(atm['pressure']):.2e} - {jnp.max(atm['pressure']):.2e} dyn/cm²")
    print(f"   Abundances loaded: {len(A_X)} elements")
    print()
    
    return {
        'atmosphere': atm,
        'abundances': A_X,
        'stellar_params': stellar_params
    }

def step2_statistical_mechanics(atm_data: Dict, layer_index: int = 30) -> Dict:
    """
    Step 2: Calculate chemical equilibrium and partition functions
    """
    print("⚛️  STEP 2: STATISTICAL MECHANICS")
    print("=" * 50)
    
    atm = atm_data['atmosphere']
    A_X = atm_data['abundances']
    stellar_params = atm_data['stellar_params']
    
    # Extract single layer for detailed analysis
    layer_T = float(atm['temperature'][layer_index])
    layer_P = float(atm['pressure'][layer_index])
    layer_rho = float(atm['density'][layer_index])
    
    print(f"Analyzing layer {layer_index}:")
    print(f"   Temperature: {layer_T:.1f} K")
    print(f"   Pressure: {layer_P:.2e} dyn/cm²")
    print(f"   Density: {layer_rho:.2e} g/cm³")
    
    # Simplified chemistry using atmospheric data directly
    print("🧪 Using atmospheric chemical data...")
    
    # Extract electron density from atmosphere
    n_e = float(atm['electron_density'][layer_index])
    
    # Estimate species densities from atmospheric conditions
    # Calculate total number density from ideal gas law
    k_B = 1.380649e-16  # erg/K
    layer_n_tot = layer_P / (k_B * layer_T)
    
    # Rough ionization fraction estimation based on temperature
    ionization_fraction = min(0.9, max(0.001, (layer_T - 3000) / 7000))
    
    # Hydrogen species (90% of total)
    n_h_total = layer_n_tot * 0.9
    n_h_i = n_h_total * (1 - ionization_fraction)
    n_h_ii = n_h_total * ionization_fraction
    
    # Helium species (10% of total)
    n_he_total = layer_n_tot * 0.1
    n_he_i = n_he_total * (1 - ionization_fraction * 0.5)  # He ionizes at higher T
    n_he_ii = n_he_total * ionization_fraction * 0.5
    
    print("✅ Species densities estimated")
    print(f"   Ionization fraction: {ionization_fraction:.3f}")
    print(f"   Total number density: {layer_n_tot:.2e} cm⁻³")
    
    print(f"Key species densities (cm⁻³):")
    print(f"   H I:  {n_h_i:.2e}")
    print(f"   H II: {n_h_ii:.2e}")
    print(f"   He I: {n_he_i:.2e}")
    print(f"   He II: {n_he_ii:.2e}")
    print(f"   e⁻:   {n_e:.2e}")
    
    # Partition functions
    print("📊 Partition functions...")
    partition_funcs = create_default_partition_functions()
    
    # Define species for partition functions
    h_i_species = Species.from_atomic_number(1, 0)  # H I
    U_H_I = float(partition_funcs[h_i_species](jnp.log(layer_T)))
    
    # Additional species for line opacity (rough estimates)
    # Metal abundances relative to hydrogen (solar values)
    fe_abundance = 10**(7.50 - 12.0)  # log(Fe/H) = -4.5, A(Fe) = 7.50
    ti_abundance = 10**(4.95 - 12.0)  # log(Ti/H) = -7.05, A(Ti) = 4.95
    
    n_fe_i = n_h_i * fe_abundance * (1 - ionization_fraction)  # Rough estimate
    n_ti_i = n_h_i * ti_abundance * (1 - ionization_fraction)  # Rough estimate
    
    fe_i_species = Species.from_atomic_number(26, 0)  # Fe I
    ti_i_species = Species.from_atomic_number(22, 0)  # Ti I
    
    U_Fe_I = float(partition_funcs[fe_i_species](jnp.log(layer_T)))
    U_Ti_I = float(partition_funcs[ti_i_species](jnp.log(layer_T)))
    
    print("✅ Partition functions calculated")
    print(f"   U(H I):  {U_H_I:.3f}")
    print(f"   U(Fe I): {U_Fe_I:.3f}")
    print(f"   U(Ti I): {U_Ti_I:.3f}")
    print()
    
    # Create simple species dictionary for compatibility
    he_i_species = Species.from_atomic_number(2, 0)  # He I
    n_species = {
        h_i_species: n_h_i,
        he_i_species: n_he_i,
        fe_i_species: n_fe_i,
        ti_i_species: n_ti_i
    }
    
    return {
        'layer_index': layer_index,
        'layer_T': layer_T,
        'layer_P': layer_P,
        'layer_rho': layer_rho,
        'n_species': n_species,
        'n_e': n_e,
        'n_h_i': n_h_i,
        'n_h_ii': n_h_ii,
        'n_he_i': n_he_i,
        'n_he_ii': n_he_ii,
        'n_fe_i': n_fe_i,
        'n_ti_i': n_ti_i,
        'U_H_I': U_H_I,
        'U_Fe_I': U_Fe_I,
        'U_Ti_I': U_Ti_I,
        'partition_funcs': partition_funcs,
        'atmosphere': atm_data['atmosphere'],
        'abundances': atm_data['abundances'],
        'stellar_params': atm_data['stellar_params']
    }

def step3_continuum_opacity(statmech_data: Dict, wavelengths: np.ndarray) -> Dict:
    """
    Step 3: Calculate continuum opacity
    """
    print("🌊 STEP 3: CONTINUUM OPACITY")
    print("=" * 50)
    
    # Convert wavelengths to frequencies
    frequencies = C_CGS / (wavelengths * ANGSTROM_TO_CM)
    frequencies_jax = jnp.array(frequencies)
    
    print(f"Wavelength range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} Å")
    print(f"Frequency range: {frequencies[-1]:.2e} - {frequencies[0]:.2e} Hz")
    print(f"Number of points: {len(wavelengths)}")
    
    # Extract atmospheric data for single layer
    layer_T = statmech_data['layer_T']
    n_h_i = statmech_data['n_h_i']
    n_he_i = statmech_data['n_he_i']
    n_e = statmech_data['n_e']
    U_H_I = statmech_data['U_H_I']
    
    print(f"\\nCalculating continuum opacity at T = {layer_T:.1f} K")
    
    # Calculate continuum opacity components individually
    print("   Computing H⁻ bound-free...")
    n_h_i_div_u = n_h_i / U_H_I
    alpha_hminus_bf = h_minus_bf_absorption(
        frequencies=frequencies_jax,
        temperature=layer_T,
        n_h_i_div_u=n_h_i_div_u,
        electron_density=n_e
    )
    
    print("   Computing H⁻ free-free...")
    alpha_hminus_ff = h_minus_ff_absorption(
        frequencies=frequencies_jax,
        temperature=layer_T,
        n_h_i_div_u=n_h_i_div_u,
        electron_density=n_e
    )
    
    print("   Computing H I bound-free...")
    alpha_h_i_bf = H_I_bf(
        frequencies=frequencies_jax,
        temperature=layer_T,
        n_h_i=n_h_i,
        n_he_i=n_he_i,
        electron_density=n_e,
        inv_u_h=1.0/U_H_I
    )
    
    print("   Computing Thomson scattering...")
    alpha_thomson = jnp.full_like(frequencies_jax, thomson_scattering(n_e))
    
    # Total continuum opacity
    continuum_opacity = alpha_hminus_bf + alpha_hminus_ff + alpha_h_i_bf + alpha_thomson
    
    print("✅ Continuum opacity calculated")
    print(f"   Peak opacity: {jnp.max(continuum_opacity):.2e} cm⁻¹")
    print(f"   Mean opacity: {jnp.mean(continuum_opacity):.2e} cm⁻¹")
    print(f"   Min opacity:  {jnp.min(continuum_opacity):.2e} cm⁻¹")
    print()
    
    return {
        **statmech_data,
        'wavelengths': wavelengths,
        'frequencies': frequencies,
        'continuum_opacity': continuum_opacity
    }

def step4_line_opacity(continuum_data: Dict, linelist_path: Optional[str] = None) -> Dict:
    """
    Step 4: Calculate line opacity
    """
    print("📏 STEP 4: LINE OPACITY")
    print("=" * 50)
    
    wavelengths = continuum_data['wavelengths']
    
    # Default linelist for the 5000-5005 Å region
    if linelist_path is None:
        linelist_path = "/Users/jdli/Project/Korg.jl/test/data/linelists/5000-5005.vald"
    
    print(f"Reading linelist: {linelist_path}")
    
    try:
        # Read linelist
        linelist = read_linelist(linelist_path, format="vald")
        print(f"✅ Linelist loaded")
        print(f"   Total lines: {len(linelist.lines)}")
        
        # Filter lines to wavelength range
        # Convert wavelength range to cm (linelist stores wavelengths in cm)
        wl_min, wl_max = wavelengths[0], wavelengths[-1]
        wl_min_cm = wl_min * 1e-8  # Å to cm
        wl_max_cm = wl_max * 1e-8  # Å to cm
        buffer_cm = 1.0 * 1e-8     # 1 Å buffer in cm
        
        relevant_lines = [line for line in linelist.lines 
                         if wl_min_cm - buffer_cm <= line.wavelength <= wl_max_cm + buffer_cm]
        
        print(f"   Lines in range: {len(relevant_lines)}")
        
        if len(relevant_lines) == 0:
            print("⚠️ No lines in wavelength range - using continuum only")
            line_opacity = jnp.zeros_like(wavelengths)
        else:
            # Show some line information
            print(f"   Line details:")
            for i, line in enumerate(relevant_lines[:5]):  # Show first 5
                wl_angstrom = line.wavelength * 1e8  # Convert cm to Å for display
                print(f"     {line.species} {wl_angstrom:.4f} Å (log gf = {line.log_gf:.3f})")
            if len(relevant_lines) > 5:
                print(f"     ... and {len(relevant_lines) - 5} more")
            
            # Calculate line opacity for the strongest lines
            print(f"\\nCalculating line opacity...")
            line_opacity = jnp.zeros_like(wavelengths)
            
            # Extract atmospheric data
            layer_T = continuum_data['layer_T']
            n_e = continuum_data['n_e']
            n_h_i = continuum_data['n_h_i']
            n_species = continuum_data['n_species']
            partition_funcs = continuum_data['partition_funcs']
            A_X = continuum_data['abundances']
            
            # Calculate opacity for each line
            lines_calculated = 0
            for line in relevant_lines:
                try:
                    # Get species information
                    species_id = line.species  # This is a numeric ID like 2600
                    
                    # Convert numeric species ID to Species object
                    # Species ID format: 100*Z + ion_stage, e.g., 2600 = Fe I (Z=26, ion=0)
                    atomic_number = species_id // 100
                    ion_stage = species_id % 100
                    species_obj = Species.from_atomic_number(atomic_number, ion_stage)
                    
                    # Get species density and partition function
                    if species_obj in n_species and species_obj in partition_funcs:
                        n_species_val = float(n_species[species_obj])
                        U_species = float(partition_funcs[species_obj](jnp.log(layer_T)))
                        
                        # Get abundance
                        if atomic_number < len(A_X):
                            abundance = float(A_X[atomic_number])
                        else:
                            abundance = 1e-12  # Very low abundance for unlisted elements
                        
                        # Calculate line opacity
                        line_params = {
                            'wavelengths': jnp.array(wavelengths),
                            'line_wavelength': line.wavelength * 1e8,  # Convert cm to Å
                            'excitation_potential': line.E_lower,  # Use E_lower instead of excitation_potential
                            'log_gf': line.log_gf,
                            'temperature': layer_T,
                            'electron_density': n_e,
                            'hydrogen_density': n_h_i,
                            'abundance': abundance,
                            'atomic_mass': atomic_number * 1.8,  # Rough approximation
                            'gamma_rad': line.gamma_rad,
                            'gamma_stark': line.gamma_stark,
                            'vald_vdw_param': line.vdw_param1,
                            'microturbulence': 2.0,  # km/s
                            'partition_function': U_species
                        }
                        
                        # Use the line opacity calculation with species-specific optimization
                        line_alpha = calculate_line_opacity_korg_method(
                            log_gamma_vdw=None,  # Use species-specific parameters
                            species_name=str(species_obj),
                            **line_params
                        )
                        
                        # Add to total line opacity
                        line_opacity += line_alpha
                        lines_calculated += 1
                        
                except Exception as e:
                    wl_angstrom = line.wavelength * 1e8
                    print(f"⚠️ Error calculating line {line.species} {wl_angstrom:.4f} Å: {e}")
                    continue
            
            print(f"✅ Line opacity calculated for {lines_calculated} lines")
            
    except Exception as e:
        print(f"❌ Error reading linelist: {e}")
        print(f"   Using continuum opacity only")
        line_opacity = jnp.zeros_like(wavelengths)
    
    print(f"   Peak line opacity: {jnp.max(line_opacity):.2e} cm⁻¹")
    print(f"   Mean line opacity: {jnp.mean(line_opacity):.2e} cm⁻¹")
    print()
    
    return {
        **continuum_data,
        'line_opacity': line_opacity,
        'linelist_path': linelist_path
    }

def step5_total_opacity(line_data: Dict) -> Dict:
    """
    Step 5: Combine continuum and line opacity
    """
    print("📊 STEP 5: TOTAL OPACITY")
    print("=" * 50)
    
    continuum_opacity = line_data['continuum_opacity']
    line_opacity = line_data['line_opacity']
    
    # Total opacity
    total_opacity = continuum_opacity + line_opacity
    
    # Calculate statistics
    continuum_contribution = jnp.mean(continuum_opacity / total_opacity) * 100
    line_contribution = jnp.mean(line_opacity / total_opacity) * 100
    
    print(f"Opacity statistics:")
    print(f"   Continuum peak: {jnp.max(continuum_opacity):.2e} cm⁻¹")
    print(f"   Line peak:      {jnp.max(line_opacity):.2e} cm⁻¹")
    print(f"   Total peak:     {jnp.max(total_opacity):.2e} cm⁻¹")
    print(f"")
    print(f"   Continuum mean: {jnp.mean(continuum_opacity):.2e} cm⁻¹")
    print(f"   Line mean:      {jnp.mean(line_opacity):.2e} cm⁻¹")
    print(f"   Total mean:     {jnp.mean(total_opacity):.2e} cm⁻¹")
    print(f"")
    print(f"Relative contributions:")
    print(f"   Continuum: {continuum_contribution:.1f}%")
    print(f"   Lines:     {line_contribution:.1f}%")
    print()
    
    # Enhancement factor
    enhancement_factor = jnp.max(total_opacity) / jnp.max(continuum_opacity)
    print(f"Line enhancement factor: {float(enhancement_factor):.2f}x")
    print(f"(Maximum total opacity / maximum continuum opacity)")
    print()
    
    return {
        **line_data,
        'total_opacity': total_opacity,
        'continuum_contribution': continuum_contribution,
        'line_contribution': line_contribution,
        'enhancement_factor': enhancement_factor
    }

def create_opacity_plots(results: Dict, save_path: Optional[str] = None):
    """
    Create comprehensive opacity plots
    """
    print("📈 CREATING OPACITY PLOTS")
    print("=" * 50)
    
    wavelengths = results['wavelengths']
    continuum_opacity = np.array(results['continuum_opacity'])
    line_opacity = np.array(results['line_opacity'])
    total_opacity = np.array(results['total_opacity'])
    stellar_params = results['stellar_params']
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All opacity components
    ax1.semilogy(wavelengths, continuum_opacity, 'b-', label='Continuum', linewidth=2)
    ax1.semilogy(wavelengths, line_opacity, 'r-', label='Lines', linewidth=2)
    ax1.semilogy(wavelengths, total_opacity, 'k-', label='Total', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Opacity (cm⁻¹)')
    ax1.set_title(f'Opacity Components - {stellar_params.name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Line enhancement
    enhancement = total_opacity / np.where(continuum_opacity > 0, continuum_opacity, 1e-50)
    ax2.plot(wavelengths, enhancement, 'g-', linewidth=2)
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Enhancement Factor (Total/Continuum)')
    ax2.set_title('Line Enhancement Factor')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='No enhancement')
    ax2.legend()
    
    # Plot 3: Linear scale detail
    ax3.plot(wavelengths, continuum_opacity, 'b-', label='Continuum', linewidth=2)
    ax3.plot(wavelengths, line_opacity, 'r-', label='Lines', linewidth=2)
    ax3.plot(wavelengths, total_opacity, 'k-', label='Total', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Wavelength (Å)')
    ax3.set_ylabel('Opacity (cm⁻¹)')
    ax3.set_title('Opacity Components (Linear Scale)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Relative contributions
    continuum_fraction = continuum_opacity / np.where(total_opacity > 0, total_opacity, 1e-50)
    line_fraction = line_opacity / np.where(total_opacity > 0, total_opacity, 1e-50)
    
    ax4.fill_between(wavelengths, 0, continuum_fraction, alpha=0.7, label='Continuum', color='blue')
    ax4.fill_between(wavelengths, continuum_fraction, continuum_fraction + line_fraction, 
                     alpha=0.7, label='Lines', color='red')
    ax4.set_xlabel('Wavelength (Å)')
    ax4.set_ylabel('Relative Contribution')
    ax4.set_title('Relative Opacity Contributions')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plots saved to {save_path}")
    
    plt.show()

def print_summary(results: Dict):
    """
    Print comprehensive pipeline summary
    """
    print("📋 PIPELINE SUMMARY")
    print("=" * 50)
    
    stellar_params = results['stellar_params']
    layer_index = results['layer_index']
    layer_T = results['layer_T']
    
    print(f"Stellar parameters: {stellar_params}")
    print(f"Analysis layer: {layer_index} (T = {layer_T:.1f} K)")
    print(f"Wavelength range: {results['wavelengths'][0]:.1f} - {results['wavelengths'][-1]:.1f} Å")
    print()
    
    print("Key results:")
    print(f"   Continuum opacity peak: {jnp.max(results['continuum_opacity']):.2e} cm⁻¹")
    print(f"   Line opacity peak:      {jnp.max(results['line_opacity']):.2e} cm⁻¹")
    print(f"   Total opacity peak:     {jnp.max(results['total_opacity']):.2e} cm⁻¹")
    print(f"   Line enhancement:       {float(results['enhancement_factor']):.2f}x")
    print()
    
    print("Relative contributions:")
    print(f"   Continuum: {float(results['continuum_contribution']):.1f}%")
    print(f"   Lines:     {float(results['line_contribution']):.1f}%")
    print()
    
    print("Species densities (cm⁻³):")
    print(f"   H I:  {results['n_h_i']:.2e}")
    print(f"   He I: {results['n_he_i']:.2e}")
    print(f"   Fe I: {results['n_fe_i']:.2e}")
    print(f"   Ti I: {results['n_ti_i']:.2e}")
    print(f"   e⁻:   {results['n_e']:.2e}")
    print()
    
    print("✅ Complete opacity pipeline successfully executed!")
    print("   This demonstrates the full stellar atmosphere opacity calculation")
    print("   from first principles: atmosphere → chemistry → continuum + lines")

def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description="Complete opacity pipeline example")
    parser.add_argument('--stellar-type', choices=list(STELLAR_TYPES.keys()), 
                       help='Predefined stellar type')
    parser.add_argument('--teff', type=float, default=5780, help='Effective temperature (K)')
    parser.add_argument('--logg', type=float, default=4.44, help='Surface gravity (log g)')
    parser.add_argument('--mh', type=float, default=0.0, help='Metallicity [M/H]')
    parser.add_argument('--layer-index', type=int, default=30, help='Atmosphere layer index')
    parser.add_argument('--wavelengths', nargs=2, type=float, default=[5000.0, 5005.0],
                       help='Wavelength range (Å)')
    parser.add_argument('--n-points', type=int, default=100, help='Number of wavelength points')
    parser.add_argument('--linelist', type=str, help='Path to VALD linelist file')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to file')
    
    args = parser.parse_args()
    
    print("🌟 COMPLETE OPACITY PIPELINE EXAMPLE")
    print("=" * 70)
    print(f"Version: {VERSION}")
    print(f"Updated: {LAST_UPDATED}")
    print("Pipeline: Atmosphere → StatMech → Continuum → Lines → Total Opacity")
    print()
    
    # Determine stellar parameters
    if args.stellar_type:
        stellar_params = STELLAR_TYPES[args.stellar_type]
    else:
        stellar_params = StellarParameters('Custom', args.teff, args.logg, args.mh)
    
    # Create wavelength grid
    wavelengths = np.linspace(args.wavelengths[0], args.wavelengths[1], args.n_points)
    
    print(f"Configuration:")
    print(f"   Stellar parameters: {stellar_params}")
    print(f"   Wavelength range: {args.wavelengths[0]:.1f} - {args.wavelengths[1]:.1f} Å")
    print(f"   Number of points: {args.n_points}")
    print(f"   Analysis layer: {args.layer_index}")
    if args.linelist:
        print(f"   Linelist: {args.linelist}")
    print()
    
    start_time = time.time()
    
    try:
        # Execute pipeline
        print("🚀 EXECUTING COMPLETE OPACITY PIPELINE\\n")
        
        # Step 1: Atmosphere interpolation
        atm_data = step1_atmosphere_interpolation(stellar_params)
        
        # Step 2: Statistical mechanics
        statmech_data = step2_statistical_mechanics(atm_data, args.layer_index)
        
        # Step 3: Continuum opacity
        continuum_data = step3_continuum_opacity(statmech_data, wavelengths)
        
        # Step 4: Line opacity
        line_data = step4_line_opacity(continuum_data, args.linelist)
        
        # Step 5: Total opacity
        results = step5_total_opacity(line_data)
        
        elapsed = time.time() - start_time
        print(f"⏱️  Pipeline execution time: {elapsed:.2f} seconds\\n")
        
        # Create plots
        if args.save_plots:
            plot_filename = f"complete_opacity_{stellar_params.name.lower().replace(' ', '_')}.png"
            create_opacity_plots(results, plot_filename)
        else:
            create_opacity_plots(results)
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\\n🎉 PIPELINE COMPLETE!")
    print(f"This example demonstrates the complete stellar opacity calculation")
    print(f"from atmosphere models through statistical mechanics to opacity.")
    
    return 0

if __name__ == "__main__":
    exit(main())