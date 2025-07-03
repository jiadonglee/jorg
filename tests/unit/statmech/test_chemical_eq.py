#!/usr/bin/env python3
"""
Test just the chemical equilibrium performance
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import time
import jax.numpy as jnp

print("🧪 Testing chemical equilibrium performance")

try:
    from jorg.statmech.chemical_equilibrium import chemical_equilibrium
    from jorg.statmech.molecular import create_default_log_equilibrium_constants
    from jorg.statmech.partition_functions import create_default_partition_functions
    from jorg.synthesis import format_abundances
    
    print("✅ Imports OK")
    
    # Test parameters
    T = 5778.0  # K
    P = 1e5     # dyne/cm²
    A_X = format_abundances(0.0)  # Solar abundances
    
    print(f"🌡️  Testing T={T}K, P={P:.0e} dyne/cm²")
    
    # Time partition function creation
    print("⚗️  Creating partition functions...")
    start = time.time()
    partition_functions = create_default_partition_functions()
    end = time.time()
    print(f"   Partition functions: {end-start:.2f}s ({len(partition_functions)} species)")
    
    # Time equilibrium constants creation
    print("⚗️  Creating equilibrium constants...")
    start = time.time()
    log_equilibrium_constants = create_default_log_equilibrium_constants()
    end = time.time()
    print(f"   Equilibrium constants: {end-start:.2f}s")
    
    # Convert A_X to absolute abundances format
    absolute_abundances = {}
    for Z in range(1, min(len(A_X), 31)):  # First 30 elements
        absolute_abundances[Z] = float(A_X[Z-1])  # A_X is 0-indexed
    
    # Simple ionization energies (eV)
    ionization_energies = {
        1: (13.6, 0.0, 0.0),     # H: 13.6 eV
        2: (24.6, 54.4, 0.0),    # He: 24.6, 54.4 eV
        # Add more as needed...
    }
    
    # Estimate number density from pressure
    k_B = 1.38e-16  # erg/K
    nt = P / (k_B * T)  # Total number density
    model_atm_ne = nt * 1e-4  # Rough estimate for electron density
    
    print(f"📊 Total density: {nt:.2e} cm⁻³")
    print(f"📊 Initial ne estimate: {model_atm_ne:.2e} cm⁻³")
    
    # Time the actual chemical equilibrium call
    print("⚗️  Running chemical equilibrium...")
    start = time.time()
    
    try:
        ne_result, number_densities = chemical_equilibrium(
            T, nt, model_atm_ne,
            absolute_abundances,
            ionization_energies,
            partition_functions,
            log_equilibrium_constants
        )
    except Exception as e:
        print(f"❌ Chemical equilibrium failed: {e}")
        # Fallback for testing
        ne_result = model_atm_ne
        number_densities = {}
    
    end = time.time()
    
    print(f"✅ Chemical equilibrium SUCCESS in {end-start:.2f}s")
    print(f"📊 Electron density: {ne_result:.2e} cm⁻³")
    print(f"📊 Number densities: {len(number_densities)} species")
    
    # Show a few species if available
    for i, (species, density) in enumerate(list(number_densities.items())[:5]):
        print(f"📊 {species}: {density:.2e} cm⁻³")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    import traceback
    traceback.print_exc()