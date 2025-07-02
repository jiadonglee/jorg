"""
Debug version of synthesis.py to identify hanging issues
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from dataclasses import dataclass
import time

from .continuum.core import total_continuum_absorption
from .constants import SPEED_OF_LIGHT, PLANCK_H, BOLTZMANN_K

# Simplified synthesis without problematic components
def synth_debug(Teff: float = 5000,
                logg: float = 4.5, 
                m_H: float = 0.0,
                wavelengths: Union[Tuple[float, float], List[Tuple[float, float]]] = (5000, 6000),
                verbose: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Debug version of synth() to identify hanging issues
    """
    if verbose:
        print(f"🔍 Debug synth called with Teff={Teff}, logg={logg}, m_H={m_H}")
        print(f"   Wavelength range: {wavelengths}")
    
    start_time = time.time()
    
    try:
        # Step 1: Create wavelength grid
        if verbose:
            print(f"📐 Creating wavelength grid...")
        
        if isinstance(wavelengths, tuple):
            wl = jnp.linspace(wavelengths[0], wavelengths[1], 100)  # Smaller grid for debugging
        else:
            wl_ranges = []
            for wl_start, wl_end in wavelengths:
                wl_ranges.append(jnp.linspace(wl_start, wl_end, 50))
            wl = jnp.concatenate(wl_ranges)
        
        if verbose:
            print(f"   ✅ Wavelength grid: {len(wl)} points from {wl[0]:.1f} to {wl[-1]:.1f} Å")
        
        # Step 2: Simple atmosphere
        if verbose:
            print(f"🌍 Creating simple atmosphere...")
        
        temperature = Teff
        electron_density = 1e13  # cm^-3
        total_density = 1e16     # cm^-3
        
        if verbose:
            print(f"   ✅ Atmosphere: T={temperature}K, ne={electron_density:.1e}, ntot={total_density:.1e}")
        
        # Step 3: Chemical equilibrium
        if verbose:
            print(f"⚗️  Calculating chemical equilibrium...")
        
        from .abundances import calculate_eos_with_asplund
        
        try:
            electron_density, number_densities = calculate_eos_with_asplund(
                temperature, total_density, electron_density, m_H
            )
            if verbose:
                print(f"   ✅ Chemical equilibrium: ne={electron_density:.2e}, {len(number_densities)} species")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Chemical equilibrium failed: {e}")
            # Use simplified values
            number_densities = {
                'H_I': total_density * 0.9,
                'He_I': total_density * 0.1,
                'H_minus': total_density * 1e-6
            }
        
        # Step 4: Continuum opacity
        if verbose:
            print(f"💫 Calculating continuum opacity...")
        
        frequencies = SPEED_OF_LIGHT / (wl * 1e-8)
        
        partition_functions = {
            'H_I': lambda log_T: 2.0,
            'He_I': lambda log_T: 1.0
        }
        
        try:
            continuum_opacity = total_continuum_absorption(
                frequencies, temperature, electron_density,
                number_densities, partition_functions, include_metals=True
            )
            if verbose:
                print(f"   ✅ Continuum opacity: {continuum_opacity.min():.2e} to {continuum_opacity.max():.2e}")
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Continuum opacity failed: {e}")
            continuum_opacity = jnp.ones_like(frequencies) * 1e-10
        
        # Step 5: Simple radiative transfer
        if verbose:
            print(f"🌟 Simple radiative transfer...")
        
        # Simplified optical depth and flux
        optical_depth = continuum_opacity * 1e8  # Simple scale height
        transmission = jnp.exp(-optical_depth)
        
        # Planck function
        h_nu_over_kt = PLANCK_H * frequencies / (BOLTZMANN_K * temperature)
        planck_func = (2 * PLANCK_H * frequencies**3 / SPEED_OF_LIGHT**2) / (jnp.exp(h_nu_over_kt) - 1)
        
        # Emergent flux
        flux = planck_func * transmission
        continuum = planck_func
        
        # Normalize
        flux_norm = flux / jnp.mean(continuum)
        
        end_time = time.time()
        
        if verbose:
            print(f"   ✅ Radiative transfer complete")
            print(f"🎉 Debug synthesis successful in {end_time - start_time:.2f} seconds!")
            print(f"   Flux range: {flux_norm.min():.3f} to {flux_norm.max():.3f}")
        
        return wl, flux_norm, continuum
        
    except Exception as e:
        end_time = time.time()
        if verbose:
            print(f"❌ Debug synthesis failed after {end_time - start_time:.2f} seconds: {e}")
        import traceback
        traceback.print_exc()
        raise


def check_imports():
    """Check which imports are causing issues"""
    print("🔍 Checking problematic imports...")
    
    issues = []
    
    # Check radiative transfer
    try:
        from .radiative_transfer import radiative_transfer, RadiativeTransferResult
        print("✅ radiative_transfer imports OK")
    except ImportError as e:
        issues.append(f"radiative_transfer: {e}")
        print(f"❌ radiative_transfer import failed: {e}")
    
    # Check lines module
    try:
        from .lines.core import total_line_absorption
        print("✅ lines.core imports OK")
    except ImportError as e:
        issues.append(f"lines.core: {e}")
        print(f"❌ lines.core import failed: {e}")
    
    # Check hydrogen lines
    try:
        from .lines.hydrogen_lines import hydrogen_line_absorption
        print("✅ hydrogen_lines imports OK")
    except ImportError as e:
        issues.append(f"hydrogen_lines: {e}")
        print(f"❌ hydrogen_lines import failed: {e}")
    
    # Check statmech
    try:
        from .statmech.chemical_equilibrium import chemical_equilibrium
        print("✅ statmech.chemical_equilibrium imports OK")
    except ImportError as e:
        issues.append(f"statmech.chemical_equilibrium: {e}")
        print(f"❌ statmech.chemical_equilibrium import failed: {e}")
    
    # Check molecular equilibrium
    try:
        from .statmech.molecular import (
            create_default_log_equilibrium_constants, create_default_equilibrium_constants
        )
        print("✅ molecular imports OK")
    except ImportError as e:
        issues.append(f"molecular: {e}")
        print(f"❌ molecular import failed: {e}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} import issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print(f"\n✅ All imports successful!")
    
    return issues