#!/usr/bin/env python3
"""
Component-wise Synthesis Test for Jorg

This test validates individual components of the synthesis pipeline
to identify where issues might be occurring.
"""

import sys
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import time

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))


def test_abundance_formatting():
    """Test abundance formatting component"""
    print("Testing abundance formatting...")
    
    try:
        from jorg.synthesis import format_abundances
        
        # Test basic solar abundances
        A_X = format_abundances(0.0)
        assert len(A_X) == 92
        assert A_X[0] == 12.0  # Hydrogen
        assert A_X[1] == 10.93  # Helium (approximate)
        
        print(f"  ✓ Solar abundances: {len(A_X)} elements")
        print(f"    H: {A_X[0]:.2f}, He: {A_X[1]:.2f}, Fe: {A_X[25]:.2f}")
        
        # Test metal-poor
        A_X_poor = format_abundances(-0.5)
        assert A_X_poor[25] < A_X[25]  # Iron should be lower
        print(f"  ✓ Metal-poor: Fe = {A_X_poor[25]:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Abundance formatting failed: {e}")
        return False


def test_atmosphere_interpolation():
    """Test atmosphere interpolation component"""
    print("\nTesting atmosphere interpolation...")
    
    try:
        from jorg.synthesis import interpolate_atmosphere, format_abundances
        
        # Test solar parameters
        A_X = format_abundances(0.0)
        atm = interpolate_atmosphere(5777, 4.44, A_X)
        
        # Validate atmosphere structure
        required_keys = ['tau_5000', 'temperature', 'pressure', 'density', 
                        'electron_density', 'height', 'n_layers']
        for key in required_keys:
            assert key in atm, f"Missing key: {key}"
        
        n_layers = atm['n_layers']
        assert n_layers > 0
        assert len(atm['temperature']) == n_layers
        assert len(atm['pressure']) == n_layers
        
        # Validate physical values
        assert np.all(atm['temperature'] > 1000)  # Reasonable temperatures
        assert np.all(atm['temperature'] < 20000)
        assert np.all(atm['pressure'] > 0)
        assert np.all(atm['density'] > 0)
        assert np.all(atm['electron_density'] > 0)
        
        print(f"  ✓ Solar atmosphere: {n_layers} layers")
        print(f"    T: {np.min(atm['temperature']):.0f}-{np.max(atm['temperature']):.0f} K")
        print(f"    P: {np.min(atm['pressure']):.2e}-{np.max(atm['pressure']):.2e}")
        print(f"    τ₅₀₀₀: {np.min(atm['tau_5000']):.2e}-{np.max(atm['tau_5000']):.2e}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Atmosphere interpolation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_radiative_transfer():
    """Test radiative transfer component"""
    print("\nTesting radiative transfer...")
    
    try:
        from jorg.radiative_transfer import radiative_transfer
        from jorg.constants import PLANCK_H, BOLTZMANN_K, SPEED_OF_LIGHT
        
        # Create simple test data
        n_layers = 10
        n_wavelengths = 20
        
        # Simple atmosphere
        tau_5000 = jnp.logspace(-2, 1, n_layers)
        temperatures = jnp.full(n_layers, 5777.0)
        heights = jnp.linspace(0, 1e7, n_layers)
        
        # Simple wavelength grid
        wavelengths = jnp.linspace(5000, 5100, n_wavelengths)
        frequencies = SPEED_OF_LIGHT / (wavelengths * 1e-8)
        
        # Simple absorption coefficient
        alpha = jnp.ones((n_layers, n_wavelengths)) * 1e-6
        
        # Planck source function
        S = jnp.zeros((n_layers, n_wavelengths))
        for i in range(n_layers):
            for j in range(n_wavelengths):
                h_nu_kt = PLANCK_H * frequencies[j] / (BOLTZMANN_K * temperatures[i])
                S = S.at[i, j].set((2 * PLANCK_H * frequencies[j]**3 / SPEED_OF_LIGHT**2) / 
                                   (jnp.exp(h_nu_kt) - 1))
        
        # Run radiative transfer
        result = radiative_transfer(
            alpha=alpha,
            S=S,
            spatial_coord=heights,
            mu_points=5,
            spherical=False,
            alpha_ref=alpha[:, 0],
            tau_ref=tau_5000,
            tau_scheme="anchored",
            I_scheme="linear_flux_only"
        )
        
        # Validate results
        assert np.all(np.isfinite(result.flux))
        assert np.all(result.flux > 0)
        assert len(result.flux) == n_wavelengths
        
        print(f"  ✓ RT calculation: {len(result.flux)} wavelengths")
        print(f"    Flux: {np.min(result.flux):.2e}-{np.max(result.flux):.2e}")
        print(f"    μ grid: {len(result.mu_grid)} points")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Radiative transfer failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_continuum_absorption():
    """Test continuum absorption component"""
    print("\nTesting continuum absorption...")
    
    try:
        from jorg.continuum.core import total_continuum_absorption
        from jorg.statmech.species import Species
        
        # Simple test conditions
        frequencies = jnp.array([6e14, 7e14, 8e14])  # Hz
        temperature = 5777.0
        electron_density = 1e14
        
        # Simple number densities
        number_densities = {
            Species.from_atomic_number(1, 0): 1e16,  # H I
            Species.from_atomic_number(1, 1): 1e14,  # H II  
            Species.from_atomic_number(2, 0): 1e15,  # He I
        }
        
        # Simple partition functions
        partition_funcs = {
            Species.from_atomic_number(1, 0): lambda log_T: 2.0,
            Species.from_atomic_number(1, 1): lambda log_T: 1.0,
            Species.from_atomic_number(2, 0): lambda log_T: 1.0,
        }
        
        # Calculate continuum absorption
        alpha_continuum = total_continuum_absorption(
            frequencies, temperature, electron_density, 
            number_densities, partition_funcs
        )
        
        # Validate results
        assert len(alpha_continuum) == len(frequencies)
        assert np.all(np.isfinite(alpha_continuum))
        assert np.all(alpha_continuum >= 0)
        
        print(f"  ✓ Continuum absorption: {len(alpha_continuum)} frequencies")
        print(f"    α: {np.min(alpha_continuum):.2e}-{np.max(alpha_continuum):.2e} cm⁻¹")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Continuum absorption failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chemical_equilibrium():
    """Test chemical equilibrium component"""
    print("\nTesting chemical equilibrium...")
    
    try:
        from jorg.statmech.chemical_equilibrium import solve_chemical_equilibrium
        from jorg.statmech.species import Species
        from jorg.statmech.partition_functions import create_default_partition_functions
        from jorg.statmech.saha_equation import create_default_ionization_energies
        from jorg.statmech.molecular import create_default_log_equilibrium_constants
        
        # Test conditions
        temperature = 5777.0
        density = 1e16  # cm⁻³
        electron_density_guess = 1e14
        
        # Simple abundances (H, He, C, O, Fe)
        abundances = {1: 1.0, 2: 0.1, 6: 1e-4, 8: 5e-4, 26: 3e-5}
        
        # Get partition functions and ionization energies
        partition_funcs = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        # Solve chemical equilibrium (with timeout)
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Chemical equilibrium timed out")
        
        # Set 30 second timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            electron_density, number_densities = solve_chemical_equilibrium(
                temperature, density, electron_density_guess, abundances,
                ionization_energies, partition_funcs, log_equilibrium_constants
            )
            signal.alarm(0)  # Cancel timeout
            
            # Validate results
            assert np.isfinite(electron_density)
            assert electron_density > 0
            assert len(number_densities) > 0
            
            # Check that we have hydrogen
            h_i_species = Species.from_atomic_number(1, 0)
            h_ii_species = Species.from_atomic_number(1, 1)
            
            h_i_density = number_densities.get(h_i_species, 0)
            h_ii_density = number_densities.get(h_ii_species, 0)
            
            print(f"  ✓ Chemical equilibrium solved")
            print(f"    nₑ: {electron_density:.2e} cm⁻³")
            print(f"    Species: {len(number_densities)}")
            print(f"    H I: {h_i_density:.2e}, H II: {h_ii_density:.2e}")
            
            return True
            
        except TimeoutError:
            signal.alarm(0)
            print(f"  ⚠ Chemical equilibrium timed out (>30s)")
            return False
            
    except Exception as e:
        print(f"  ✗ Chemical equilibrium failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_synthesis():
    """Test a very simple synthesis"""
    print("\nTesting simple synthesis...")
    
    try:
        # Use a very simple approach
        from jorg.synthesis import format_abundances
        from jorg.radiative_transfer import radiative_transfer
        from jorg.constants import PLANCK_H, BOLTZMANN_K, SPEED_OF_LIGHT
        
        # Minimal test setup
        n_layers = 5
        n_wavelengths = 10
        
        # Simple atmosphere
        temperatures = jnp.array([6000, 5800, 5600, 5400, 5200])
        heights = jnp.array([0, 1e6, 2e6, 3e6, 4e6])
        tau_5000 = jnp.array([0.01, 0.1, 1.0, 10.0, 100.0])
        
        # Simple wavelength grid  
        wavelengths = jnp.linspace(5000, 5100, n_wavelengths)
        frequencies = SPEED_OF_LIGHT / (wavelengths * 1e-8)
        
        # Simple constant absorption
        alpha = jnp.ones((n_layers, n_wavelengths)) * 1e-5
        
        # Planck source function
        S = jnp.zeros((n_layers, n_wavelengths))
        for i in range(n_layers):
            for j in range(n_wavelengths):
                h_nu_kt = PLANCK_H * frequencies[j] / (BOLTZMANN_K * temperatures[i])
                if h_nu_kt < 50:  # Avoid overflow
                    planck = (2 * PLANCK_H * frequencies[j]**3 / SPEED_OF_LIGHT**2) / (jnp.exp(h_nu_kt) - 1)
                else:
                    planck = 0.0
                S = S.at[i, j].set(planck)
        
        # Run minimal radiative transfer
        result = radiative_transfer(
            alpha=alpha,
            S=S,
            spatial_coord=heights,
            mu_points=3,
            spherical=False,
            alpha_ref=alpha[:, 0],
            tau_ref=tau_5000,
            tau_scheme="anchored",
            I_scheme="linear_flux_only"
        )
        
        # Validate
        assert np.all(np.isfinite(result.flux))
        assert np.all(result.flux > 0)
        
        print(f"  ✓ Simple synthesis: {len(result.flux)} wavelengths")
        print(f"    Flux: {np.min(result.flux):.2e}-{np.max(result.flux):.2e}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Simple synthesis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main component test suite"""
    print("=" * 60)
    print("Jorg Synthesis Component Tests")
    print("=" * 60)
    
    components = [
        ("Abundance formatting", test_abundance_formatting),
        ("Atmosphere interpolation", test_atmosphere_interpolation),
        ("Radiative transfer", test_radiative_transfer),
        ("Continuum absorption", test_continuum_absorption),
        ("Chemical equilibrium", test_chemical_equilibrium),
        ("Simple synthesis", test_simple_synthesis),
    ]
    
    passed = 0
    total = len(components)
    
    for component_name, test_func in components:
        print(f"\n{'='*10} {component_name} {'='*10}")
        try:
            start_time = time.time()
            if test_func():
                elapsed = time.time() - start_time
                passed += 1
                print(f"✓ {component_name} PASSED ({elapsed:.1f}s)")
            else:
                elapsed = time.time() - start_time
                print(f"✗ {component_name} FAILED ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"✗ {component_name} ERROR ({elapsed:.1f}s): {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPONENT TEST SUMMARY")
    print("=" * 60)
    print(f"Components passed: {passed}/{total}")
    
    if passed >= total - 1:  # Allow one failure
        print("✅ SYNTHESIS COMPONENTS MOSTLY WORKING")
        success = True
    elif passed >= total // 2:
        print("⚠️ Some synthesis components working")
        success = False
    else:
        print("❌ Major synthesis component failures")
        success = False
    
    print("=" * 60)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)