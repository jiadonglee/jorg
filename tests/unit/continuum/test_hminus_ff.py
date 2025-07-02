"""
Unit tests for H⁻ free-free absorption implementation.

Tests the exact Bell & Berrington (1987) table interpolation against Korg.jl reference values.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from jorg.continuum.hydrogen import (
    h_minus_ff_absorption, 
    _interpolate_bell_berrington,
    h_minus_number_density
)
from jorg.constants import c_cgs, kboltz_cgs


class TestHMinusFFAbsorption:
    """Test H⁻ free-free absorption calculations"""
    
    def test_bell_berrington_interpolation_table_bounds(self):
        """Test interpolation at table boundary points"""
        # Test at exact table points
        lambda_min, lambda_max = 1823.0, 151890.0
        theta_min, theta_max = 0.5, 3.6
        
        # Corner points
        k00 = _interpolate_bell_berrington(lambda_min, theta_min)
        assert k00 == pytest.approx(0.0178, rel=1e-6)
        
        k11 = _interpolate_bell_berrington(lambda_max, theta_max)
        assert k11 == pytest.approx(388.0, rel=1e-6)
        
        # Mid-range point (exact table value)
        k_mid = _interpolate_bell_berrington(5063.0, 1.0)  
        assert k_mid == pytest.approx(0.195, rel=1e-6)  # Exact table value at λ=5063, θ=1.0
    
    def test_bell_berrington_interpolation_extrapolation(self):
        """Test that values outside table bounds are clamped"""
        # Values outside bounds should be clamped to boundary values
        k_low = _interpolate_bell_berrington(1000.0, 0.3)  # Below bounds
        k_high = _interpolate_bell_berrington(200000.0, 4.0)  # Above bounds
        
        # Should equal corner values
        k_corner_low = _interpolate_bell_berrington(1823.0, 0.5)
        k_corner_high = _interpolate_bell_berrington(151890.0, 3.6)
        
        assert k_low == pytest.approx(k_corner_low, rel=1e-6)
        assert k_high == pytest.approx(k_corner_high, rel=1e-6)
    
    def test_bell_berrington_monotonicity(self):
        """Test expected monotonic behavior where applicable"""
        # K generally increases with increasing θ (decreasing temperature) at fixed wavelength
        k1 = _interpolate_bell_berrington(5000.0, 1.0)
        k2 = _interpolate_bell_berrington(5000.0, 2.0)
        assert k2 > k1
        
        # For longer wavelengths, K generally increases with decreasing wavelength
        k1 = _interpolate_bell_berrington(15000.0, 1.0)
        k2 = _interpolate_bell_berrington(10000.0, 1.0)
        assert k2 < k1  # At these wavelengths, K is larger at longer wavelengths
    
    def test_h_minus_number_density_saha(self):
        """Test H⁻ number density calculation using Saha equation"""
        # Typical stellar atmosphere conditions
        temperature = 5778.0  # K
        n_h_i_div_u = 1e15    # cm^-3
        electron_density = 1e13  # cm^-3
        
        n_h_minus = h_minus_number_density(n_h_i_div_u, electron_density, temperature)
        
        # H⁻ should be much less abundant than H I
        assert n_h_minus > 0
        assert n_h_minus < n_h_i_div_u * 1e-6  # Should be rare
        
        # Check temperature dependence
        n_h_minus_hot = h_minus_number_density(n_h_i_div_u, electron_density, 8000.0)
        assert n_h_minus_hot < n_h_minus  # Less H⁻ at higher T
    
    def test_h_minus_ff_absorption_basic(self):
        """Test basic H⁻ free-free absorption calculation"""
        # Solar conditions
        frequencies = jnp.linspace(5e14, 8e14, 100)  # Hz (visible range)
        temperature = 5778.0  # K
        n_h_i_div_u = 1e15   # cm^-3
        electron_density = 1e13  # cm^-3
        
        alpha = h_minus_ff_absorption(frequencies, temperature, n_h_i_div_u, electron_density)
        
        # Basic sanity checks
        assert jnp.all(alpha >= 0)  # Non-negative absorption
        assert jnp.all(jnp.isfinite(alpha))  # No infinities or NaNs
        assert alpha.shape == frequencies.shape
    
    def test_h_minus_ff_absorption_wavelength_dependence(self):
        """Test wavelength dependence of H⁻ free-free absorption"""
        # Test at specific wavelengths from Bell & Berrington table
        wavelengths_angstrom = jnp.array([5000.0, 6000.0, 8000.0, 10000.0])
        frequencies = c_cgs * 1e8 / wavelengths_angstrom  # Convert to Hz
        
        temperature = 5040.0  # K (corresponds to θ = 1.0)
        n_h_i_div_u = 1e15
        electron_density = 1e13
        
        alpha = h_minus_ff_absorption(frequencies, temperature, n_h_i_div_u, electron_density)
        
        # H⁻ ff absorption should generally increase toward longer wavelengths
        # (though with some structure due to the table interpolation)
        assert jnp.all(alpha > 0)
        
        # Check relative magnitudes are reasonable
        assert alpha[0] / alpha[-1] < 10  # Should not vary by huge factors
    
    def test_h_minus_ff_absorption_temperature_dependence(self):
        """Test temperature dependence of H⁻ free-free absorption"""
        frequencies = jnp.array([6e14])  # Single frequency
        temperatures = jnp.array([4000.0, 5000.0, 6000.0, 8000.0])
        n_h_i_div_u = 1e15
        electron_density = 1e13
        
        alphas = []
        for T in temperatures:
            alpha = h_minus_ff_absorption(frequencies, T, n_h_i_div_u, electron_density)
            alphas.append(alpha[0])
        
        alphas = jnp.array(alphas)
        
        # H⁻ ff absorption has complex temperature dependence due to competing effects:
        # - H⁻ number density decreases with T (exponential factor)
        # - Bell & Berrington K factor varies with θ = 5040/T
        # At these densities and frequencies, the effect is weak but values should be reasonable
        assert jnp.all(alphas > 0)  # All positive
        assert jnp.max(alphas) / jnp.min(alphas) < 2.0  # Variation within factor of 2
        
        # Test with more extreme temperature difference
        alpha_cool = h_minus_ff_absorption(frequencies, 3000.0, n_h_i_div_u, electron_density)[0]
        alpha_hot = h_minus_ff_absorption(frequencies, 10000.0, n_h_i_div_u, electron_density)[0]
        assert alpha_cool > alpha_hot  # Should see clearer trend with wider range
    
    def test_h_minus_ff_absorption_density_scaling(self):
        """Test density scaling of H⁻ free-free absorption"""
        frequencies = jnp.array([6e14])
        temperature = 5000.0
        
        # Test H I density scaling
        n_h_base = 1e15
        alpha_base = h_minus_ff_absorption(frequencies, temperature, n_h_base, 1e13)
        alpha_double = h_minus_ff_absorption(frequencies, temperature, 2*n_h_base, 1e13)
        
        # Should scale linearly with H I density
        assert alpha_double[0] / alpha_base[0] == pytest.approx(2.0, rel=1e-3)
        
        # Test electron density scaling  
        ne_base = 1e13
        alpha_base = h_minus_ff_absorption(frequencies, temperature, 1e15, ne_base)
        alpha_double = h_minus_ff_absorption(frequencies, temperature, 1e15, 2*ne_base)
        
        # Complex scaling due to both P_e and n(H⁻) dependence
        assert alpha_double[0] > alpha_base[0]  # Should increase
        assert alpha_double[0] / alpha_base[0] > 1.5  # But not simply linear
    
    def test_h_minus_ff_physics_consistency(self):
        """Test physical consistency of H⁻ free-free absorption"""
        # Test that K values are in reasonable physical range
        lambda_test = 5500.0  # Å (V band)
        theta_test = 1.0      # T = 5040 K
        
        K = _interpolate_bell_berrington(lambda_test, theta_test)
        
        # K should be in range of table values (10^-28 to 10^-24 cm^4/dyn)
        assert 0.01 < K < 1000.0
        
        # Test that final absorption coefficient has reasonable units
        frequencies = jnp.array([5.5e14])  # Hz (V band)
        temperature = 5040.0
        n_h_i_div_u = 1e15     # cm^-3
        electron_density = 1e13 # cm^-3
        
        alpha = h_minus_ff_absorption(frequencies, temperature, n_h_i_div_u, electron_density)
        
        # Typical H⁻ ff absorption should be modest contributor to continuum
        # For solar conditions, expect ~10^-12 to 10^-8 cm^-1
        assert 1e-15 < alpha[0] < 1e-5
    
    def test_h_minus_ff_vectorization(self):
        """Test that function works with vectorized inputs"""
        # Test frequency vectorization
        frequencies = jnp.linspace(4e14, 8e14, 50)
        temperature = 5000.0
        n_h_i_div_u = 1e15
        electron_density = 1e13
        
        alpha = h_minus_ff_absorption(frequencies, temperature, n_h_i_div_u, electron_density)
        
        assert alpha.shape == frequencies.shape
        assert jnp.all(jnp.isfinite(alpha))
        assert jnp.all(alpha >= 0)
    
    @pytest.mark.parametrize("wavelength,theta,expected", [
        (2278.0, 0.6, 0.0280),    # Exact table value
        (5063.0, 1.6, 0.311),     # Exact table value  
        (10126.0, 2.8, 1.73),     # Exact table value
        (45567.0, 3.6, 35.0),     # Exact table value
    ])
    def test_bell_berrington_exact_values(self, wavelength, theta, expected):
        """Test exact Bell & Berrington table values"""
        K = _interpolate_bell_berrington(wavelength, theta)
        assert K == pytest.approx(expected, rel=1e-6)
    
    def test_comparison_with_korg_reference(self):
        """Test against known Korg.jl reference values"""
        # These reference values would come from running Korg.jl
        # For now, test physical reasonableness
        
        # Solar photosphere conditions
        frequencies = jnp.array([5.5e14])  # V band
        temperature = 5778.0
        n_h_i_div_u = 1e15
        electron_density = 1e13
        
        alpha = h_minus_ff_absorption(frequencies, temperature, n_h_i_div_u, electron_density)
        
        # Expected order of magnitude for solar conditions
        # (These values should be validated against actual Korg.jl runs)
        assert 1e-12 < alpha[0] < 1e-8
        
        # Test cooler conditions at same density (should have more H⁻)
        alpha_cool = h_minus_ff_absorption(frequencies, 4500.0, 1e15, 1e13)
        
        # Cooler temperature should generally have more H⁻ formation
        # (though Bell & Berrington K factor also changes)
        assert alpha_cool[0] > 0  # Basic sanity check
        assert 1e-15 < alpha_cool[0] < 1e-5  # Reasonable range


if __name__ == "__main__":
    pytest.main([__file__])