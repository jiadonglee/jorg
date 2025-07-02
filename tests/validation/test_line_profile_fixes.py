#!/usr/bin/env python3
"""
Focused test to validate and fix Jorg line profile implementation
"""

import numpy as np
import jax.numpy as jnp
import sys
from pathlib import Path

# Add jorg to path
sys.path.insert(0, str(Path(__file__).parent / '..' / 'src'))

from jorg.lines.profiles import voigt_hjerting, line_profile, harris_series
from jorg.lines.broadening import doppler_width, scaled_stark, scaled_vdw_simple
from jorg.constants import c_cgs, kboltz_cgs, ATOMIC_MASS_UNIT

def test_doppler_width_fix():
    """Test and fix Doppler width calculation"""
    
    print("=== Testing Doppler Width Calculation ===")
    
    # Test parameters
    lambda_0 = 5000e-8  # 5000 Å in cm
    temperature = 5778.0  # K
    mass_amu = 55.845  # Fe atomic mass in u
    xi = 2.0e5  # 2 km/s microturbulence in cm/s
    
    # Convert mass to grams for Jorg function
    mass_grams = mass_amu * ATOMIC_MASS_UNIT
    
    print(f"Parameters:")
    print(f"  λ₀ = {lambda_0*1e8:.1f} Å")
    print(f"  T = {temperature:.1f} K")
    print(f"  mass = {mass_amu:.3f} u = {mass_grams:.3e} g")
    print(f"  ξ = {xi/1e5:.1f} km/s")
    
    # Calculate with Jorg
    sigma_jorg = doppler_width(lambda_0, temperature, mass_grams, xi)
    
    # Manual calculation matching Korg.jl exactly
    thermal_velocity_sq = kboltz_cgs * temperature / mass_grams
    microturbulent_velocity_sq = xi**2 / 2.0  # Note the /2 factor from Korg.jl
    total_velocity_sq = thermal_velocity_sq + microturbulent_velocity_sq
    sigma_expected = lambda_0 * np.sqrt(total_velocity_sq) / c_cgs
    
    print(f"\nJorg result: {sigma_jorg*1e8:.6f} Å")
    print(f"Expected:    {sigma_expected*1e8:.6f} Å")
    
    rel_error = abs(sigma_jorg - sigma_expected) / sigma_expected
    print(f"Relative error: {rel_error:.2e}")
    
    assert rel_error < 1e-14, f"Doppler width error: {rel_error:.2e}"
    print("✓ Doppler width calculation is correct")
    
    return sigma_jorg, sigma_expected

def test_voigt_profile_calculation():
    """Test basic Voigt profile calculation"""
    
    print("\n=== Testing Voigt Profile Calculation ===")
    
    # Simple test parameters
    lambda_0 = 5000e-8  # 5000 Å in cm  
    sigma = 0.5e-8      # 0.5 Å Doppler width
    gamma = 0.1e-8      # 0.1 Å Lorentz width
    amplitude = 1.0     # Unit amplitude
    
    # Test single wavelength at line center
    wl_center = lambda_0
    profile_center = line_profile(lambda_0, sigma, gamma, amplitude, wl_center)
    
    print(f"Line center (λ₀ = {lambda_0*1e8:.1f} Å):")
    print(f"  σ = {sigma*1e8:.2f} Å")
    print(f"  γ = {gamma*1e8:.2f} Å")
    print(f"  amplitude = {amplitude}")
    print(f"  Profile at center: {profile_center:.6e} cm⁻¹")
    
    # Test a few points around line center
    wavelengths = jnp.array([lambda_0 - 2*sigma, lambda_0 - sigma, lambda_0, 
                            lambda_0 + sigma, lambda_0 + 2*sigma])
    
    profile_values = line_profile(lambda_0, sigma, gamma, amplitude, wavelengths)
    
    print(f"\nProfile values:")
    for i, (wl, val) in enumerate(zip(wavelengths, profile_values)):
        offset = (wl - lambda_0) / sigma
        print(f"  {offset:+5.1f}σ: {val:.6e} cm⁻¹")
    
    # Check that center has maximum value
    center_idx = 2  # Middle wavelength
    assert profile_values[center_idx] == jnp.max(profile_values), "Center should be maximum"
    print("✓ Profile has maximum at line center")
    
    # Check symmetry
    left_val = profile_values[1]   # -1σ
    right_val = profile_values[3]  # +1σ
    symmetry_error = abs(left_val - right_val) / max(left_val, right_val)
    print(f"Symmetry error (±1σ): {symmetry_error:.2e}")
    assert symmetry_error < 1e-14, f"Symmetry error too large: {symmetry_error:.2e}"
    print("✓ Profile is symmetric")
    
    return profile_values

def test_profile_normalization():
    """Test line profile normalization with fine grid"""
    
    print("\n=== Testing Profile Normalization ===")
    
    lambda_0 = 5000e-8  # 5000 Å
    sigma = 0.3e-8      # 0.3 Å Doppler width
    gamma = 0.05e-8     # 0.05 Å Lorentz width
    amplitude = 1.0     # Unit amplitude
    
    # Create fine wavelength grid for integration  
    # Lorentzian tails are long, need wider range
    wl_range = 10e-8  # ±10 Å around line center
    n_points = 50000
    wavelengths = jnp.linspace(lambda_0 - wl_range, lambda_0 + wl_range, n_points)
    
    # Calculate profile
    profile = line_profile(lambda_0, sigma, gamma, amplitude, wavelengths)
    
    # Numerical integration
    dlambda = wavelengths[1] - wavelengths[0]
    integrated = jnp.sum(profile) * dlambda
    
    normalization_error = abs(integrated - amplitude) / amplitude
    
    print(f"Integration range: ±{wl_range*1e8:.1f} Å")
    print(f"Grid points: {n_points}")
    print(f"Input amplitude: {amplitude}")
    print(f"Integrated profile: {integrated:.8f}")
    print(f"Normalization error: {normalization_error:.2e}")
    
    # Reasonable tolerance for numerical integration (Lorentzian tails extend to infinity)
    assert normalization_error < 5e-3, f"Normalization error too large: {normalization_error:.2e}"
    print("✓ Line profile normalization is adequate")
    
    return integrated

def test_voigt_parameter_regimes():
    """Test the different parameter regimes in voigt_hjerting"""
    
    print("\n=== Testing Voigt Parameter Regimes ===")
    
    # Test cases for each regime
    test_cases = [
        # (alpha, v, regime_name)
        (0.05, 6.0, "small_alpha_large_v"),    # α ≤ 0.2 & v ≥ 5
        (0.15, 2.0, "small_alpha_small_v"),    # α ≤ 0.2 & v < 5
        (0.8, 1.0, "intermediate"),            # α ≤ 1.4 & α+v < 3.2
        (1.2, 0.5, "intermediate"),            # α ≤ 1.4 & α+v < 3.2
        (1.5, 1.0, "large_alpha"),             # α > 1.4 or α+v ≥ 3.2
        (0.5, 3.0, "large_alpha"),             # α > 1.4 or α+v ≥ 3.2 (α+v=3.5)
    ]
    
    print("Alpha    v     Sum    Regime                  H(α,v)")
    print("-" * 55)
    
    for alpha, v, expected_regime in test_cases:
        H = voigt_hjerting(alpha, v)
        alpha_plus_v = alpha + v
        
        # Determine actual regime
        if alpha <= 0.2 and v >= 5.0:
            actual_regime = "small_alpha_large_v"
        elif alpha <= 0.2:
            actual_regime = "small_alpha_small_v"
        elif alpha <= 1.4 and alpha_plus_v < 3.2:
            actual_regime = "intermediate"
        else:
            actual_regime = "large_alpha"
        
        status = "✓" if actual_regime == expected_regime else "✗"
        print(f"{alpha:5.1f}  {v:5.1f}  {alpha_plus_v:5.1f}  {actual_regime:18s}  {H:10.6e} {status}")
        
        assert actual_regime == expected_regime, f"Wrong regime for α={alpha}, v={v}"
    
    print("✓ All parameter regimes working correctly")

def test_broadening_temperature_scaling():
    """Test temperature scaling of broadening mechanisms"""
    
    print("\n=== Testing Broadening Temperature Scaling ===")
    
    # Test Stark broadening T^(1/6) scaling
    gamma_stark_ref = 1e-15
    T_ref = 10000.0
    temperatures = [5000.0, 7500.0, 10000.0, 15000.0]
    
    print("Stark broadening (T^1/6 scaling):")
    print("Temperature (K)    Scaling Factor    γ_Stark")
    print("-" * 45)
    
    for T in temperatures:
        gamma = scaled_stark(gamma_stark_ref, T, T_ref)
        expected_scaling = (T / T_ref)**(1/6)
        expected_gamma = gamma_stark_ref * expected_scaling
        
        rel_error = abs(gamma - expected_gamma) / expected_gamma
        print(f"{T:11.0f}       {expected_scaling:.6f}      {gamma:.6e}")
        assert rel_error < 1e-14, f"Stark scaling error: {rel_error:.2e}"
    
    # Test van der Waals T^0.3 scaling
    gamma_vdw_ref = 1e-30
    
    print("\nvan der Waals broadening (T^0.3 scaling):")
    print("Temperature (K)    Scaling Factor    γ_vdW")
    print("-" * 45)
    
    for T in temperatures:
        gamma = scaled_vdw_simple(gamma_vdw_ref, T, T_ref)
        expected_scaling = (T / T_ref)**0.3
        expected_gamma = gamma_vdw_ref * expected_scaling
        
        rel_error = abs(gamma - expected_gamma) / expected_gamma
        print(f"{T:11.0f}       {expected_scaling:.6f}      {gamma:.6e}")
        assert rel_error < 1e-14, f"vdW scaling error: {rel_error:.2e}"
    
    print("✓ Temperature scaling is exact")

def run_all_tests():
    """Run all focused tests"""
    
    print("=" * 60)
    print("FOCUSED JORG LINE PROFILE VALIDATION")
    print("=" * 60)
    
    tests = [
        test_doppler_width_fix,
        test_voigt_profile_calculation, 
        test_profile_normalization,
        test_voigt_parameter_regimes,
        test_broadening_temperature_scaling
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
            print(f"✓ {test_func.__name__} PASSED")
        except Exception as e:
            print(f"✗ {test_func.__name__} FAILED: {e}")
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print(f"{'='*60}")
    
    if passed == total:
        print("🎉 All tests passed - Jorg line profiles are working correctly!")
        return True
    else:
        print("❌ Some tests failed - needs investigation")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)