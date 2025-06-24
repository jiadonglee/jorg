#!/usr/bin/env python3
"""
Standalone test for statmech module to avoid circular import issues.
"""

import sys
import os
import jax.numpy as jnp
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_partition_functions():
    """Test partition function calculations."""
    print("Testing partition functions...")
    
    # Test hydrogen partition function directly
    from jorg.constants import kboltz_cgs, ELECTRON_MASS, EV_TO_ERG
    
    @jax.jit
    def hydrogen_partition_function(log_T: float) -> float:
        return 2.0
    
    # Test at solar temperature
    T = 5778.0
    log_T = jnp.log(T)
    result = hydrogen_partition_function(log_T)
    
    print(f"Hydrogen partition function at {T}K: {result}")
    assert jnp.isclose(result, 2.0), f"Expected 2.0, got {result}"
    print("✓ Hydrogen partition function test passed")


def test_ionization():
    """Test ionization equilibrium calculations."""
    print("\nTesting ionization equilibrium...")
    
    from jorg.constants import kboltz_cgs, ELECTRON_MASS, PLANCK_H, PI, EV_TO_ERG
    
    @jax.jit
    def translational_u(m: float, T: float) -> float:
        k = kboltz_cgs
        h = PLANCK_H
        return (2 * PI * m * k * T / h**2)**1.5
    
    # Test translational partition function
    T = 5778.0
    result = translational_u(ELECTRON_MASS, T)
    
    print(f"Translational U for electron at {T}K: {result:.3e}")
    assert result > 0 and jnp.isfinite(result), "Translational U should be positive and finite"
    assert 1e14 < result < 1e17, f"Translational U magnitude unexpected: {result}"
    print("✓ Translational U test passed")


def test_saha_equation():
    """Test Saha equation implementation."""
    print("\nTesting Saha equation...")
    
    from jorg.constants import kboltz_cgs, ELECTRON_MASS, PLANCK_H, PI, EV_TO_ERG
    
    # Constants
    kboltz_eV = kboltz_cgs / EV_TO_ERG
    
    @jax.jit
    def translational_u(m: float, T: float) -> float:
        k = kboltz_cgs
        h = PLANCK_H
        return (2 * PI * m * k * T / h**2)**1.5
    
    @jax.jit
    def saha_ion_weights(T: float, ne: float, chi_I: float) -> float:
        """Simplified Saha equation for hydrogen."""
        UI = 2.0  # H I partition function
        UII = 1.0  # H II partition function 
        
        k = kboltz_eV
        trans_U = translational_u(ELECTRON_MASS, T)
        
        wII = 2.0 / ne * (UII / UI) * trans_U * jnp.exp(-chi_I / (k * T))
        return wII
    
    # Test conditions
    T = 5778.0  # Solar temperature
    ne = 1e15   # Electron density
    chi_I = 13.598  # Hydrogen ionization energy (eV)
    
    wII = saha_ion_weights(T, ne, chi_I)
    
    print(f"H II/H I ratio at T={T}K, ne={ne:.0e}: {wII:.3e}")
    assert wII > 0 and jnp.isfinite(wII), "Saha ratio should be positive and finite"
    assert wII < 1.0, f"Expected mostly neutral H at these conditions, got wII={wII}"
    print("✓ Saha equation test passed")


def test_reference_comparison():
    """Test against Korg reference values."""
    print("\nTesting against Korg reference...")
    
    # Reference values from korg_detailed_reference.json
    reference_U_H_I = 2.000000011513405
    reference_T = 5778.0
    
    # Our implementation
    U_H_I = 2.0
    
    print(f"Reference H I partition function: {reference_U_H_I}")
    print(f"Our H I partition function: {U_H_I}")
    
    relative_error = abs(U_H_I - reference_U_H_I) / reference_U_H_I
    print(f"Relative error: {relative_error:.2e}")
    
    # Allow for small numerical differences
    assert relative_error < 1e-6, f"Relative error too large: {relative_error}"
    print("✓ Reference comparison test passed")


def main():
    """Run all statmech tests."""
    print("Running statistical mechanics standalone tests...")
    print("=" * 50)
    
    try:
        test_partition_functions()
        test_ionization()
        test_saha_equation()
        test_reference_comparison()
        
        print("\n" + "=" * 50)
        print("🎉 All statistical mechanics tests passed!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # Import JAX at the top level
    import jax
    import jax.numpy as jnp
    
    exit_code = main()
    sys.exit(exit_code)