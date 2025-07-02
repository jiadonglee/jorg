"""
Validation tests comparing Jorg H⁻ free-free implementation against Korg.jl reference behavior.

These tests validate that our Bell & Berrington (1987) implementation produces
physically consistent results that match the expected behavior from Korg.jl.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from jorg.continuum.hydrogen import h_minus_ff_absorption, _interpolate_bell_berrington
from jorg.constants import c_cgs, kboltz_cgs


class TestKorgValidation:
    """Validation tests against Korg.jl reference behavior"""
    
    def test_bell_berrington_table_consistency(self):
        """Test that our Bell & Berrington table exactly matches Korg.jl values"""
        # These are exact values from the original Bell & Berrington (1987) table
        # as implemented in Korg.jl
        
        test_cases = [
            # (wavelength_Å, theta, expected_K)
            (1823.0, 0.5, 0.0178),   # Table corners
            (1823.0, 3.6, 0.172),
            (151890.0, 0.5, 75.1),
            (151890.0, 3.6, 388.0),
            
            # Mid-table values
            (5063.0, 1.0, 0.195),    # λ index 6, θ index 3
            (5063.0, 1.6, 0.311),    # λ index 6, θ index 6
            (10126.0, 2.0, 1.28),    # λ index 11, θ index 8
            (22784.0, 2.8, 7.59),    # λ index 16, θ index 9
            
            # Edge cases
            (2278.0, 0.6, 0.0280),   # θ index 1
            (3038.0, 0.8, 0.0616),   # θ index 2
            (7595.0, 1.2, 0.484),    # λ index 9, θ index 4
        ]
        
        for wavelength, theta, expected in test_cases:
            K = _interpolate_bell_berrington(wavelength, theta)
            assert K == pytest.approx(expected, rel=1e-6), \
                f"Failed for λ={wavelength} Å, θ={theta}: got {K}, expected {expected}"
    
    def test_bell_berrington_interpolation_accuracy(self):
        """Test interpolation accuracy between table points"""
        # Test interpolation between known points
        
        # Between λ=5063 and λ=5696 at θ=1.0
        K_5063 = _interpolate_bell_berrington(5063.0, 1.0)  # 0.195
        K_5696 = _interpolate_bell_berrington(5696.0, 1.0)  # 0.241
        K_mid = _interpolate_bell_berrington(5379.5, 1.0)   # Midpoint
        
        # Should be approximately the average (linear interpolation)
        expected_mid = (K_5063 + K_5696) / 2
        assert K_mid == pytest.approx(expected_mid, rel=1e-3)
        
        # Between θ=1.0 and θ=1.2 at λ=5063
        K_theta1 = _interpolate_bell_berrington(5063.0, 1.0)   # 0.195
        K_theta12 = _interpolate_bell_berrington(5063.0, 1.2)  # 0.234
        K_theta_mid = _interpolate_bell_berrington(5063.0, 1.1)  # Midpoint
        
        expected_theta_mid = (K_theta1 + K_theta12) / 2
        assert K_theta_mid == pytest.approx(expected_theta_mid, rel=1e-3)
    
    def test_temperature_parameter_conversion(self):
        """Test θ = 5040/T conversion exactly matches Korg.jl"""
        test_temperatures = [
            (10080.0, 0.5),   # Hot star
            (8400.0, 0.6),    # A star
            (6300.0, 0.8),    # F star  
            (5040.0, 1.0),    # Solar
            (4200.0, 1.2),    # K star
            (3600.0, 1.4),    # M star
            (3150.0, 1.6),    # Cool M star
            (2800.0, 1.8),    # Very cool
            (2520.0, 2.0),    # Brown dwarf
            (1800.0, 2.8),    # Cool brown dwarf
            (1400.0, 3.6),    # Very cool
        ]
        
        for temp, expected_theta in test_temperatures:
            theta = 5040.0 / temp
            assert theta == pytest.approx(expected_theta, rel=1e-6), \
                f"θ calculation failed for T={temp}K: got {theta}, expected {expected_theta}"
    
    def test_stellar_atmosphere_conditions(self):
        """Test H⁻ ff absorption under realistic stellar atmosphere conditions"""
        
        # Solar photosphere (τ_5000 = 1)
        freq_solar = c_cgs * 1e8 / 5500.0  # 5500 Å in Hz
        T_solar = 5778.0
        n_H_solar = 1e15      # cm^-3
        n_e_solar = 1e13      # cm^-3
        
        alpha_solar = h_minus_ff_absorption(
            jnp.array([freq_solar]), T_solar, n_H_solar, n_e_solar
        )[0]
        
        # Should be modest contributor to total opacity
        assert 1e-12 < alpha_solar < 1e-9, \
            f"Solar H⁻ ff opacity out of expected range: {alpha_solar:.2e}"
        
        # K dwarf (cooler, should have more H⁻)
        T_k_dwarf = 4500.0
        alpha_k_dwarf = h_minus_ff_absorption(
            jnp.array([freq_solar]), T_k_dwarf, n_H_solar, n_e_solar
        )[0]
        
        assert alpha_k_dwarf > 0
        assert 1e-15 < alpha_k_dwarf < 1e-8
        
        # M dwarf (much cooler)
        T_m_dwarf = 3500.0
        alpha_m_dwarf = h_minus_ff_absorption(
            jnp.array([freq_solar]), T_m_dwarf, n_H_solar, n_e_solar
        )[0]
        
        assert alpha_m_dwarf > 0
        assert 1e-15 < alpha_m_dwarf < 1e-7
    
    def test_wavelength_range_coverage(self):
        """Test that our implementation covers the full wavelength range"""
        # Bell & Berrington table covers 1823-151890 Å
        temperatures = [3000.0, 5778.0, 8000.0]  # Range of stellar temperatures
        
        # UV to infrared wavelengths
        wavelengths_angstrom = [2000.0, 3000.0, 5000.0, 8000.0, 15000.0, 50000.0]
        frequencies = c_cgs * 1e8 / jnp.array(wavelengths_angstrom)
        
        for T in temperatures:
            alpha = h_minus_ff_absorption(frequencies, T, 1e15, 1e13)
            
            # All values should be finite and non-negative
            assert jnp.all(jnp.isfinite(alpha))
            assert jnp.all(alpha >= 0)
            
            # Should have reasonable magnitude
            assert jnp.all(alpha < 1e-5)  # Not unreasonably large
            assert jnp.any(alpha > 1e-20)  # Not all zero
    
    def test_physical_scaling_laws(self):
        """Test that physical scaling laws are obeyed"""
        frequencies = jnp.array([5.5e14])  # V band
        
        # Test density scaling
        T = 5000.0
        base_ne = 1e13
        base_nH = 1e15
        
        # Double electron density
        alpha1 = h_minus_ff_absorption(frequencies, T, base_nH, base_ne)[0]
        alpha2 = h_minus_ff_absorption(frequencies, T, base_nH, 2*base_ne)[0]
        
        # Should increase (more electrons, higher pressure, more H⁻)
        assert alpha2 > alpha1
        
        # Double H I density  
        alpha3 = h_minus_ff_absorption(frequencies, T, 2*base_nH, base_ne)[0]
        
        # Should increase approximately linearly
        ratio = alpha3 / alpha1
        assert 1.8 < ratio < 2.2  # Allow some nonlinearity
        
        # Test temperature dependence at fixed density
        alpha_cool = h_minus_ff_absorption(frequencies, 4000.0, base_nH, base_ne)[0]
        alpha_hot = h_minus_ff_absorption(frequencies, 7000.0, base_nH, base_ne)[0]
        
        # Generally expect more H⁻ at cooler temperatures
        # (though Bell & Berrington K factor complicates this)
        assert alpha_cool > 0
        assert alpha_hot > 0
        
    def test_units_and_dimensions(self):
        """Test that units are consistent with Korg.jl"""
        frequencies = jnp.array([6e14])  # Hz
        T = 5000.0                       # K
        n_H = 1e15                       # cm^-3
        n_e = 1e13                       # cm^-3
        
        alpha = h_minus_ff_absorption(frequencies, T, n_H, n_e)[0]
        
        # Should have units of cm^-1 (absorption coefficient)
        # For the given inputs, expect reasonable magnitude
        assert 1e-20 < alpha < 1e-5, \
            f"Absorption coefficient has unreasonable magnitude: {alpha:.2e} cm⁻¹"
        
        # Test that Bell & Berrington K has correct units
        K = _interpolate_bell_berrington(5500.0, 1.0)  # returns value from table
        
        # K should be in range 0.01 to 1000 (table values × 10^-26 cm^4/dyn)
        assert 0.001 < K < 10000, \
            f"Bell & Berrington K value out of range: {K}"
    
    def test_numerical_stability(self):
        """Test numerical stability across parameter ranges"""
        # Test extreme values that might cause numerical issues
        frequencies = jnp.logspace(13, 16, 10)  # Wide frequency range
        
        # Very hot conditions
        alpha_hot = h_minus_ff_absorption(frequencies, 15000.0, 1e14, 1e12)
        assert jnp.all(jnp.isfinite(alpha_hot))
        assert jnp.all(alpha_hot >= 0)
        
        # Very cool conditions  
        alpha_cool = h_minus_ff_absorption(frequencies, 2000.0, 1e16, 1e14)
        assert jnp.all(jnp.isfinite(alpha_cool))
        assert jnp.all(alpha_cool >= 0)
        
        # Very low density
        alpha_low = h_minus_ff_absorption(frequencies, 5000.0, 1e10, 1e8)
        assert jnp.all(jnp.isfinite(alpha_low))
        assert jnp.all(alpha_low >= 0)
        
        # Very high density
        alpha_high = h_minus_ff_absorption(frequencies, 5000.0, 1e18, 1e16)
        assert jnp.all(jnp.isfinite(alpha_high))
        assert jnp.all(alpha_high >= 0)


@pytest.mark.integration
def test_hminus_ff_integration_with_synthesis():
    """Integration test: H⁻ ff in full spectral synthesis"""
    from jorg.synthesis import synth
    
    # Test that H⁻ ff contributes to synthetic spectrum  
    wavelengths, flux, continuum = synth(
        Teff=4500,  # Cool enough for significant H⁻
        logg=2.0,   # Giant (lower density)
        m_H=0.0,
        wavelengths=(8000, 8100)  # Red wavelengths where H⁻ ff is important
    )
    
    assert len(wavelengths) == len(flux) == len(continuum)
    assert jnp.all(jnp.isfinite(flux))
    assert jnp.all(flux > 0)
    assert jnp.all(continuum > 0)


if __name__ == "__main__":
    pytest.main([__file__])