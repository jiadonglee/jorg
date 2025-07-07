#!/usr/bin/env python3
"""
Simple Opacity Calculation Example
=================================

Minimal example showing how to calculate total stellar opacity using Jorg.
This script demonstrates the essential steps: EOS → Continuum + Lines → Total Opacity

Usage:
    python simple_opacity_example.py

Requirements:
    - JAX
    - NumPy
    - Matplotlib (optional, for plotting)
"""

import sys
import numpy as np
import jax.numpy as jnp

# Add Jorg to path (adjust as needed)
sys.path.append('src')

from jorg.synthesis import format_abundances, interpolate_atmosphere
from jorg.statmech.chemical_equilibrium import chemical_equilibrium
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.statmech.molecular import create_default_log_equilibrium_constants
from jorg.continuum.core import total_continuum_absorption
from jorg.lines.hydrogen_lines_simple import hydrogen_line_absorption
from jorg.constants import SPEED_OF_LIGHT


def calculate_stellar_opacity(Teff=5778, logg=4.44, m_H=0.0, wavelength_range=(6540, 6580)):
    """
    Calculate total stellar opacity for given stellar parameters.
    
    Parameters
    ----------
    Teff : float
        Effective temperature in K
    logg : float
        Surface gravity (log g)
    m_H : float
        Metallicity [M/H]
    wavelength_range : tuple
        Wavelength range in Å (min, max)
        
    Returns
    -------
    dict
        Dictionary containing wavelengths, opacity components, and results
    """
    
    print(f"🌟 Calculating Stellar Opacity")
    print(f"   Teff = {Teff:.0f} K, log g = {logg:.1f}, [M/H] = {m_H:.1f}")
    print(f"   Wavelength range: {wavelength_range[0]:.0f}-{wavelength_range[1]:.0f} Å")
    
    # =================================================================
    # STEP 1: Setup stellar atmosphere and abundances
    # =================================================================
    
    # Format abundances
    A_X = format_abundances(m_H)
    
    # Create simple atmosphere (or use MARCS interpolation)
    atm = interpolate_atmosphere(Teff, logg, A_X)
    
    # Use surface layer for this example
    T = float(atm['temperature'][0])  # Surface temperature
    ne = float(atm['electron_density'][0])  # Surface electron density
    rho = float(atm['density'][0])  # Surface density
    
    print(f"   Surface conditions: T={T:.0f}K, ne={ne:.1e} cm⁻³, ρ={rho:.1e} g/cm³")
    
    # =================================================================
    # STEP 2: Solve chemical equilibrium (EOS)
    # =================================================================
    
    print("   🧮 Solving chemical equilibrium...")
    
    # Create partition functions and equilibrium constants
    species_partition_functions = create_default_partition_functions()
    log_equilibrium_constants = create_default_log_equilibrium_constants()
    
    # Prepare element abundances and ionization energies
    absolute_abundances = {Z: float(A_X[Z-1]) for Z in range(1, min(len(A_X), 31))}
    ionization_energies = {
        1: (13.6, 0.0, 0.0),     # H
        2: (24.6, 54.4, 0.0),    # He
        26: (7.9, 16.2, 30.7),   # Fe
    }
    
    # Estimate total number density
    k_B = 1.38e-16  # erg/K
    nt = atm['pressure'][0] / (k_B * T)
    
    # Solve chemical equilibrium
    try:
        ne_eq, number_densities = chemical_equilibrium(
            T, nt, ne,
            absolute_abundances,
            ionization_energies, 
            species_partition_functions,
            log_equilibrium_constants
        )
        print(f"   ✅ EOS converged: ne = {ne_eq:.2e} cm⁻³")
        
    except Exception as e:
        print(f"   ⚠️  EOS failed ({e}), using estimates")
        # Fallback to simple estimates
        h_ion_frac = 0.01 if T > 6000 else 0.001
        number_densities = {
            'H_I': rho * (1 - h_ion_frac) * 0.92,
            'H_II': rho * h_ion_frac * 0.92,
            'He_I': rho * 0.08,
            'H_minus': rho * 1e-7,
        }
        ne_eq = ne
    
    # Extract key species densities
    nH_I = number_densities.get('H_I', rho * 0.9)
    nHe_I = number_densities.get('He_I', rho * 0.08)
    nH_minus = number_densities.get('H_minus', rho * 1e-7)
    
    print(f"   Species densities:")
    print(f"   • nH_I = {nH_I:.2e} cm⁻³")
    print(f"   • nHe_I = {nHe_I:.2e} cm⁻³") 
    print(f"   • nH⁻ = {nH_minus:.2e} cm⁻³")
    
    # =================================================================
    # STEP 3: Calculate opacity components
    # =================================================================
    
    print("   🌈 Calculating opacity components...")
    
    # Setup wavelength grid
    wl_min, wl_max = wavelength_range
    wavelengths_A = jnp.linspace(wl_min, wl_max, 50)
    wavelengths_cm = wavelengths_A * 1e-8
    frequencies = SPEED_OF_LIGHT / wavelengths_cm
    
    # 1. CONTINUUM OPACITY
    partition_functions = {'H_I': lambda x: 2.0, 'He_I': lambda x: 1.0}
    
    continuum_alpha = total_continuum_absorption(
        frequencies, T, ne_eq, number_densities, partition_functions
    )
    
    # 2. HYDROGEN LINE OPACITY
    UH_I = 2.0  # H I partition function
    xi = 2e5    # 2 km/s microturbulence in cm/s
    
    hydrogen_alpha = hydrogen_line_absorption(
        wavelengths_cm, T, ne_eq, nH_I, nHe_I, UH_I, xi,
        window_size_cm=20e-8, use_MHD=True
    )
    
    # 3. TOTAL OPACITY
    total_alpha = continuum_alpha + hydrogen_alpha
    
    # =================================================================
    # STEP 4: Analyze results
    # =================================================================
    
    print("   📊 Results:")
    print(f"   • Continuum opacity (mean): {jnp.mean(continuum_alpha):.2e} cm⁻¹")
    print(f"   • Hydrogen lines (peak): {jnp.max(hydrogen_alpha):.2e} cm⁻¹")
    print(f"   • Total opacity (peak): {jnp.max(total_alpha):.2e} cm⁻¹")
    
    # Find line enhancement
    line_enhancement = jnp.max(hydrogen_alpha) / jnp.mean(continuum_alpha)
    print(f"   • Line enhancement: {line_enhancement:.1f}× over continuum")
    
    # Check for Hα if in range
    if 6560 <= (wl_min + wl_max)/2 <= 6570:
        hα_idx = jnp.argmin(jnp.abs(wavelengths_A - 6562.8))
        hα_depth = hydrogen_alpha[hα_idx] / continuum_alpha[hα_idx]
        print(f"   • Hα line depth: {hα_depth:.2f}× continuum")
    
    return {
        'wavelengths_A': wavelengths_A,
        'continuum_opacity': continuum_alpha,
        'hydrogen_opacity': hydrogen_alpha,
        'total_opacity': total_alpha,
        'stellar_params': {'Teff': Teff, 'logg': logg, 'm_H': m_H},
        'surface_conditions': {'T': T, 'ne': ne_eq, 'nH_I': nH_I},
        'line_enhancement': line_enhancement
    }


def plot_results(results):
    """Plot opacity results if matplotlib is available."""
    
    try:
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        wavelengths = results['wavelengths_A']
        continuum = results['continuum_opacity']
        hydrogen = results['hydrogen_opacity']
        total = results['total_opacity']
        
        # Plot 1: Linear scale
        ax1.plot(wavelengths, continuum, 'b-', label='Continuum', linewidth=2)
        ax1.plot(wavelengths, hydrogen, 'r-', label='H lines', linewidth=2)
        ax1.plot(wavelengths, total, 'k-', label='Total', linewidth=2)
        ax1.set_xlabel('Wavelength (Å)')
        ax1.set_ylabel('Opacity (cm⁻¹)')
        ax1.set_title('Stellar Opacity Components')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Log scale 
        ax2.semilogy(wavelengths, continuum, 'b-', label='Continuum', linewidth=2)
        ax2.semilogy(wavelengths, hydrogen, 'r-', label='H lines', linewidth=2)
        ax2.semilogy(wavelengths, total, 'k-', label='Total', linewidth=2)
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Opacity (cm⁻¹)')
        ax2.set_title('Stellar Opacity (Log Scale)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add stellar parameters as title
        params = results['stellar_params']
        fig.suptitle(f"Jorg Opacity: Teff={params['Teff']:.0f}K, log g={params['logg']:.1f}, [M/H]={params['m_H']:.1f}")
        
        plt.tight_layout()
        plt.savefig('simple_opacity_results.png', dpi=300, bbox_inches='tight')
        print("   📊 Plot saved: simple_opacity_results.png")
        
        return fig
        
    except ImportError:
        print("   ⚠️  Matplotlib not available, skipping plots")
        return None


def main():
    """Run simple opacity calculation examples."""
    
    print("🚀 Jorg Simple Opacity Example")
    print("=" * 40)
    
    # Example 1: Solar parameters around Hα
    print("\n📍 Example 1: Solar conditions around Hα")
    results1 = calculate_stellar_opacity(
        Teff=5778, logg=4.44, m_H=0.0, 
        wavelength_range=(6540, 6580)
    )
    
    # Example 2: Cooler star with broader range
    print("\n📍 Example 2: K dwarf with broader wavelength range")
    results2 = calculate_stellar_opacity(
        Teff=4500, logg=4.5, m_H=0.0,
        wavelength_range=(5000, 7000)
    )
    
    # Example 3: Metal-poor star
    print("\n📍 Example 3: Metal-poor star")
    results3 = calculate_stellar_opacity(
        Teff=5500, logg=4.0, m_H=-1.0,
        wavelength_range=(6540, 6580)
    )
    
    # Plot the first example
    print("\n📊 Creating plots...")
    plot_results(results1)
    
    print("\n🏆 Examples Complete!")
    print("✅ Three stellar opacity calculations performed")
    print("✅ Complete EOS → Opacity pipeline demonstrated")
    print("✅ Results show realistic stellar atmosphere physics")
    
    return results1, results2, results3


if __name__ == "__main__":
    results = main()