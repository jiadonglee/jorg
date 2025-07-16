#!/usr/bin/env python3
"""
Debug NaN flux values in optimized synthesis
"""

import sys
import time
import numpy as np
import jax.numpy as jnp
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

def debug_synthesis_step_by_step():
    """Debug synthesis step by step to find NaN source"""
    print("🔍 Debugging NaN Flux Values")
    print("=" * 50)
    
    try:
        from jorg.synthesis_optimized import synthesize_fast
        from jorg.synthesis import format_abundances, interpolate_atmosphere
        
        # Setup basic parameters
        Teff, logg, m_H = 5777, 4.44, 0.0
        
        print("Step 1: Format abundances...")
        A_X = format_abundances(m_H)
        print(f"  ✓ A_X shape: {A_X.shape}")
        print(f"  ✓ A_X range: {np.min(A_X):.2f} - {np.max(A_X):.2f}")
        print(f"  ✓ A_X finite: {np.all(np.isfinite(A_X))}")
        
        print("\nStep 2: Interpolate atmosphere...")
        atm = interpolate_atmosphere(Teff, logg, A_X)
        print(f"  ✓ Layers: {atm['n_layers']}")
        print(f"  ✓ Temperature: {np.min(atm['temperature']):.0f}-{np.max(atm['temperature']):.0f}K")
        print(f"  ✓ Temperature finite: {np.all(np.isfinite(atm['temperature']))}")
        print(f"  ✓ Density: {np.min(atm['density']):.2e}-{np.max(atm['density']):.2e}")
        print(f"  ✓ Density finite: {np.all(np.isfinite(atm['density']))}")
        print(f"  ✓ Electron density: {np.min(atm['electron_density']):.2e}-{np.max(atm['electron_density']):.2e}")
        print(f"  ✓ Electron density finite: {np.all(np.isfinite(atm['electron_density']))}")
        
        print("\nStep 3: Create wavelength grid...")
        wavelengths = jnp.linspace(5000, 5020, 20)  # Very small for debugging
        print(f"  ✓ Wavelengths: {len(wavelengths)} points")
        print(f"  ✓ Range: {np.min(wavelengths):.1f}-{np.max(wavelengths):.1f} Å")
        
        print("\nStep 4: Test synthesis with minimal parameters...")
        
        # Call synthesis with debug parameters
        result = synthesize_fast(
            atm=atm,
            linelist=None,  # No lines
            A_X=A_X,
            wavelengths=wavelengths,
            vmic=1.0,
            hydrogen_lines=False,  # Disable H lines
            mu_values=3,  # Very few mu points
            verbose=True
        )
        
        print(f"\nStep 5: Check synthesis results...")
        print(f"  ✓ Flux shape: {result.flux.shape}")
        print(f"  ✓ Flux finite: {np.all(np.isfinite(result.flux))}")
        if not np.all(np.isfinite(result.flux)):
            nan_count = np.sum(~np.isfinite(result.flux))
            print(f"  ❌ NaN/Inf flux values: {nan_count}")
            print(f"  ❌ Flux range: {np.nanmin(result.flux):.3e} - {np.nanmax(result.flux):.3e}")
        else:
            print(f"  ✓ Flux range: {np.min(result.flux):.3e} - {np.max(result.flux):.3e}")
        
        print(f"  ✓ Continuum shape: {result.cntm.shape}")
        print(f"  ✓ Continuum finite: {np.all(np.isfinite(result.cntm))}")
        if not np.all(np.isfinite(result.cntm)):
            nan_count = np.sum(~np.isfinite(result.cntm))
            print(f"  ❌ NaN/Inf continuum values: {nan_count}")
        else:
            print(f"  ✓ Continuum range: {np.min(result.cntm):.3e} - {np.max(result.cntm):.3e}")
        
        print(f"  ✓ Alpha shape: {result.alpha.shape}")
        print(f"  ✓ Alpha finite: {np.all(np.isfinite(result.alpha))}")
        if not np.all(np.isfinite(result.alpha)):
            nan_count = np.sum(~np.isfinite(result.alpha))
            print(f"  ❌ NaN/Inf alpha values: {nan_count}")
            print(f"  ❌ Alpha range: {np.nanmin(result.alpha):.3e} - {np.nanmax(result.alpha):.3e}")
        else:
            print(f"  ✓ Alpha range: {np.min(result.alpha):.3e} - {np.max(result.alpha):.3e}")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_radiative_transfer_directly():
    """Test radiative transfer directly to isolate issues"""
    print("\n🔬 Testing Radiative Transfer Directly")
    print("=" * 50)
    
    try:
        from jorg.radiative_transfer import radiative_transfer
        from jorg.constants import SPEED_OF_LIGHT, PLANCK_H, BOLTZMANN_K
        
        # Create simple test data
        n_layers = 10
        n_wl = 20
        wavelengths = jnp.linspace(5000, 5020, n_wl)
        frequencies = SPEED_OF_LIGHT / (wavelengths * 1e-8)
        
        # Simple atmosphere
        tau_5000 = jnp.logspace(-3, 1, n_layers)
        temperature = jnp.linspace(6000, 4000, n_layers)
        height = jnp.linspace(0, 1e7, n_layers)
        
        print(f"  Test setup: {n_layers} layers × {n_wl} wavelengths")
        print(f"  Temperature: {np.min(temperature):.0f}-{np.max(temperature):.0f}K")
        print(f"  Tau: {np.min(tau_5000):.3e}-{np.max(tau_5000):.3e}")
        
        # Simple opacity and source function
        alpha = jnp.ones((n_layers, n_wl)) * 1e-6
        print(f"  Alpha: constant {1e-6}")
        
        # Planck source function
        h_nu_over_kt = PLANCK_H * frequencies[None, :] / (BOLTZMANN_K * temperature[:, None])
        source_function = (2 * PLANCK_H * frequencies[None, :]**3 / SPEED_OF_LIGHT**2 / 
                          (jnp.exp(h_nu_over_kt) - 1))
        source_function = source_function * SPEED_OF_LIGHT / (wavelengths[None, :] * 1e-8)**2
        
        print(f"  Source function range: {np.min(source_function):.3e}-{np.max(source_function):.3e}")
        print(f"  Source function finite: {np.all(np.isfinite(source_function))}")
        
        # Test radiative transfer
        print("  Running radiative transfer...")
        rt_result = radiative_transfer(
            alpha=alpha,
            S=source_function,
            spatial_coord=height,
            mu_points=3,
            spherical=False,
            include_inward_rays=False,
            alpha_ref=alpha[:, 0],
            tau_ref=tau_5000,
            tau_scheme='anchored',
            I_scheme='linear_flux_only'
        )
        
        print(f"  ✓ RT completed")
        print(f"  ✓ Flux shape: {rt_result.flux.shape}")
        print(f"  ✓ Flux finite: {np.all(np.isfinite(rt_result.flux))}")
        if not np.all(np.isfinite(rt_result.flux)):
            nan_count = np.sum(~np.isfinite(rt_result.flux))
            print(f"  ❌ NaN/Inf flux values: {nan_count}")
        else:
            print(f"  ✓ Flux range: {np.min(rt_result.flux):.3e} - {np.max(rt_result.flux):.3e}")
        
        return rt_result
        
    except Exception as e:
        print(f"\n❌ RT test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_continuum_directly():
    """Test continuum calculation directly"""
    print("\n🧪 Testing Continuum Calculation Directly")
    print("=" * 50)
    
    try:
        from jorg.continuum.core import total_continuum_absorption
        from jorg.constants import SPEED_OF_LIGHT
        
        # Simple test parameters
        wavelengths = jnp.linspace(5000, 5020, 20)
        frequencies = SPEED_OF_LIGHT / (wavelengths * 1e-8)
        
        # Simple atmospheric layer
        T = 5777.0
        ne = 1e14
        number_densities = {
            'H_I': 1e16,
            'H_II': 1e13,
            'He_I': 1e15,
            'H_minus': 1e10,
            'H2': 1e8
        }
        partition_functions = {
            'H_I': lambda log_T: 2.0,
            'He_I': lambda log_T: 1.0
        }
        
        print(f"  Test parameters:")
        print(f"    T = {T}K")
        print(f"    ne = {ne:.1e} cm⁻³")
        print(f"    Wavelengths: {len(wavelengths)} points")
        
        # Test continuum calculation
        print("  Running continuum calculation...")
        cntm_alpha = total_continuum_absorption(
            frequencies, T, ne, number_densities, partition_functions
        )
        
        print(f"  ✓ Continuum completed")
        print(f"  ✓ Shape: {cntm_alpha.shape}")
        print(f"  ✓ Finite: {np.all(np.isfinite(cntm_alpha))}")
        if not np.all(np.isfinite(cntm_alpha)):
            nan_count = np.sum(~np.isfinite(cntm_alpha))
            print(f"  ❌ NaN/Inf values: {nan_count}")
        else:
            print(f"  ✓ Range: {np.min(cntm_alpha):.3e} - {np.max(cntm_alpha):.3e}")
        
        return cntm_alpha
        
    except Exception as e:
        print(f"\n❌ Continuum test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main debug function"""
    print("🔍 Debugging NaN Flux in Optimized Synthesis")
    print("=" * 60)
    
    # Test individual components
    continuum_result = test_continuum_directly()
    rt_result = test_radiative_transfer_directly()
    
    # Test full synthesis step by step
    synthesis_result = debug_synthesis_step_by_step()
    
    # Summary
    print("\n" + "=" * 60)
    print("DEBUG SUMMARY")
    print("=" * 60)
    
    continuum_ok = continuum_result is not None and np.all(np.isfinite(continuum_result))
    rt_ok = rt_result is not None and np.all(np.isfinite(rt_result.flux))
    synthesis_ok = synthesis_result is not None and np.all(np.isfinite(synthesis_result.flux))
    
    print(f"Continuum calculation: {'✅ OK' if continuum_ok else '❌ NaN'}")
    print(f"Radiative transfer: {'✅ OK' if rt_ok else '❌ NaN'}")
    print(f"Full synthesis: {'✅ OK' if synthesis_ok else '❌ NaN'}")
    
    if continuum_ok and rt_ok and not synthesis_ok:
        print("\n🔍 DIAGNOSIS: Issue is in synthesis integration, not individual components")
        print("Likely causes:")
        print("  • Chemical equilibrium producing invalid densities")
        print("  • Atmosphere interpolation issues")
        print("  • Opacity matrix construction")
    elif not continuum_ok:
        print("\n🔍 DIAGNOSIS: Issue is in continuum calculation")
    elif not rt_ok:
        print("\n🔍 DIAGNOSIS: Issue is in radiative transfer")
    elif synthesis_ok:
        print("\n🎉 SUCCESS: All components working correctly!")
    
    print("=" * 60)
    
    return synthesis_ok


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)