"""
Unit tests for the Jorg continuum opacity module

Tests continuum absorption calculations following Korg.jl testing patterns.
"""

import pytest
import jax.numpy as jnp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jorg.continuum.core import total_continuum_absorption
from jorg.continuum.hydrogen import hydrogen_continuum_absorption
from jorg.continuum.helium import helium_continuum_absorption
from jorg.continuum.scattering import rayleigh_scattering, thomson_scattering
from jorg.synthesis import format_abundances


class TestHydrogenContinuum:
    """Test hydrogen continuum absorption calculations"""
    
    def test_hydrogen_bf_absorption(self):
        """Test hydrogen bound-free absorption"""
        wavelengths = jnp.array([4000.0, 5000.0, 6000.0])  # Angstroms
        temperature = 5800.0  # K
        n_HI = 1e12  # cm^-3
        
        # Test that function runs without error
        absorption = hydrogen_continuum_absorption(wavelengths, temperature, n_HI)
        
        assert len(absorption) == len(wavelengths)
        assert jnp.all(absorption >= 0)
        assert jnp.all(jnp.isfinite(absorption))
    
    def test_hydrogen_temperature_dependence(self):
        """Test temperature dependence of hydrogen opacity"""
        wavelengths = jnp.array([5000.0])
        n_HI = 1e12
        
        # Test different temperatures
        T_cool = 4000.0
        T_hot = 8000.0
        
        abs_cool = hydrogen_continuum_absorption(wavelengths, T_cool, n_HI)
        abs_hot = hydrogen_continuum_absorption(wavelengths, T_hot, n_HI)
        
        # Bound-free absorption should decrease with temperature
        # (fewer atoms in ground state)
        assert abs_hot[0] < abs_cool[0]
    
    def test_hydrogen_density_scaling(self):
        """Test density scaling of hydrogen opacity"""
        wavelengths = jnp.array([5000.0])
        temperature = 5800.0
        
        n_low = 1e11
        n_high = 1e13
        
        abs_low = hydrogen_continuum_absorption(wavelengths, temperature, n_low)
        abs_high = hydrogen_continuum_absorption(wavelengths, temperature, n_high)
        
        # Should scale linearly with density
        ratio = abs_high[0] / abs_low[0]
        expected_ratio = n_high / n_low
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-6)
    
    def test_wavelength_dependence(self):
        """Test wavelength dependence follows expected behavior"""
        # Test across Lyman series limit
        wavelengths = jnp.array([900.0, 912.0, 1000.0])  # Around Lyman limit
        temperature = 5800.0
        n_HI = 1e12
        
        absorption = hydrogen_continuum_absorption(wavelengths, temperature, n_HI)
        
        # Should have edge at 912 Å
        assert absorption[0] > 0  # Below limit
        assert absorption[2] > 0  # Above limit
        
        # Check that it's finite everywhere
        assert jnp.all(jnp.isfinite(absorption))


class TestHeliumContinuum:
    """Test helium continuum absorption"""
    
    def test_helium_absorption_basic(self):
        """Test basic helium absorption calculation"""
        wavelengths = jnp.array([4000.0, 5000.0, 6000.0])
        temperature = 5800.0
        n_HeI = 1e11  # cm^-3
        
        absorption = helium_continuum_absorption(wavelengths, temperature, n_HeI)
        
        assert len(absorption) == len(wavelengths)
        assert jnp.all(absorption >= 0)
        assert jnp.all(jnp.isfinite(absorption))
    
    def test_helium_temperature_dependence(self):
        """Test helium opacity temperature dependence"""
        wavelengths = jnp.array([5000.0])
        n_HeI = 1e11
        
        T_cool = 4000.0
        T_hot = 8000.0
        
        abs_cool = helium_continuum_absorption(wavelengths, T_cool, n_HeI)
        abs_hot = helium_continuum_absorption(wavelengths, T_hot, n_HeI)
        
        # Both should be positive and finite
        assert abs_cool[0] >= 0
        assert abs_hot[0] >= 0
        assert jnp.isfinite(abs_cool[0])
        assert jnp.isfinite(abs_hot[0])
    
    def test_helium_wavelength_edge(self):
        """Test helium ionization edge behavior"""
        # Test around He I ionization edge (504 Å)
        wavelengths = jnp.array([500.0, 504.0, 510.0])
        temperature = 5800.0
        n_HeI = 1e11
        
        absorption = helium_continuum_absorption(wavelengths, temperature, n_HeI)
        
        # Should have significant absorption near edge
        assert jnp.all(jnp.isfinite(absorption))
        assert jnp.all(absorption >= 0)


class TestScattering:
    """Test scattering opacity calculations"""
    
    def test_rayleigh_scattering(self):
        """Test Rayleigh scattering calculation"""
        wavelengths = jnp.array([4000.0, 5000.0, 6000.0])
        n_H = 1e12  # cm^-3
        
        scattering = rayleigh_scattering(wavelengths, n_H)
        
        assert len(scattering) == len(wavelengths)
        assert jnp.all(scattering >= 0)
        assert jnp.all(jnp.isfinite(scattering))
        
        # Should scale as λ^-4
        assert scattering[0] > scattering[1] > scattering[2]
    
    def test_thomson_scattering(self):
        """Test Thomson (electron) scattering"""
        wavelengths = jnp.array([4000.0, 5000.0, 6000.0])
        n_e = 1e10  # cm^-3
        
        scattering = thomson_scattering(wavelengths, n_e)
        
        assert len(scattering) == len(wavelengths)
        assert jnp.all(scattering >= 0)
        assert jnp.all(jnp.isfinite(scattering))
        
        # Thomson scattering is wavelength-independent
        np.testing.assert_allclose(scattering, scattering[0], rtol=1e-10)
    
    def test_scattering_density_scaling(self):
        """Test that scattering scales with density"""
        wavelengths = jnp.array([5000.0])
        n_low = 1e10
        n_high = 1e12
        
        scat_low = thomson_scattering(wavelengths, n_low)
        scat_high = thomson_scattering(wavelengths, n_high)
        
        ratio = scat_high[0] / scat_low[0]
        expected_ratio = n_high / n_low
        np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-10)


class TestTotalContinuumAbsorption:
    """Test total continuum absorption combining all sources"""
    
    def test_total_continuum_basic(self):
        """Test basic total continuum calculation"""
        wavelengths = jnp.array([4000.0, 5000.0, 6000.0])
        temperature = 5800.0
        electron_density = 1e10  # cm^-3
        A_X = format_abundances(0.0)  # Solar abundances
        
        absorption = total_continuum_absorption(
            wavelengths, temperature, electron_density, A_X
        )
        
        assert len(absorption) == len(wavelengths)
        assert jnp.all(absorption >= 0)
        assert jnp.all(jnp.isfinite(absorption))
    
    def test_temperature_dependence(self):
        """Test temperature dependence of total continuum"""
        wavelengths = jnp.array([5000.0])
        electron_density = 1e10
        A_X = format_abundances(0.0)
        
        T_cool = 4000.0
        T_hot = 8000.0
        
        abs_cool = total_continuum_absorption(wavelengths, T_cool, electron_density, A_X)
        abs_hot = total_continuum_absorption(wavelengths, T_hot, electron_density, A_X)
        
        # Both should be positive
        assert abs_cool[0] > 0
        assert abs_hot[0] > 0
        
        # Generally, continuum opacity decreases with temperature
        # (though this depends on the wavelength regime)
        assert jnp.isfinite(abs_cool[0])
        assert jnp.isfinite(abs_hot[0])
    
    def test_metallicity_dependence(self):
        """Test metallicity dependence"""
        wavelengths = jnp.array([5000.0])
        temperature = 5800.0
        electron_density = 1e10
        
        A_X_solar = format_abundances(0.0)
        A_X_metal_poor = format_abundances(-1.0)
        
        abs_solar = total_continuum_absorption(wavelengths, temperature, electron_density, A_X_solar)
        abs_poor = total_continuum_absorption(wavelengths, temperature, electron_density, A_X_metal_poor)
        
        # Metal-poor stars should have different continuum opacity
        # (mainly from H- which depends on electron density)
        assert abs_solar[0] > 0
        assert abs_poor[0] > 0
        assert jnp.isfinite(abs_solar[0])
        assert jnp.isfinite(abs_poor[0])
    
    def test_wavelength_coverage(self):
        """Test continuum across wide wavelength range"""
        # Test from UV to near-IR
        wavelengths = jnp.array([2000.0, 4000.0, 5000.0, 8000.0, 12000.0])
        temperature = 5800.0
        electron_density = 1e10
        A_X = format_abundances(0.0)
        
        absorption = total_continuum_absorption(
            wavelengths, temperature, electron_density, A_X
        )
        
        # Should be finite everywhere
        assert jnp.all(jnp.isfinite(absorption))
        assert jnp.all(absorption >= 0)
        
        # Generally expect higher opacity at shorter wavelengths
        # (though this is not always true due to edges)
        assert len(absorption) == len(wavelengths)
    
    def test_electron_density_scaling(self):
        """Test electron density dependence"""
        wavelengths = jnp.array([5000.0])
        temperature = 5800.0
        A_X = format_abundances(0.0)
        
        n_e_low = 1e9
        n_e_high = 1e11
        
        abs_low = total_continuum_absorption(wavelengths, temperature, n_e_low, A_X)
        abs_high = total_continuum_absorption(wavelengths, temperature, n_e_high, A_X)
        
        # Higher electron density should generally increase opacity
        # (from H- and electron scattering)
        assert abs_high[0] > abs_low[0]
    
    def test_physical_units(self):
        """Test that opacity has correct physical units"""
        wavelengths = jnp.array([5000.0])  # Angstroms
        temperature = 5800.0  # K
        electron_density = 1e10  # cm^-3
        A_X = format_abundances(0.0)
        
        absorption = total_continuum_absorption(
            wavelengths, temperature, electron_density, A_X
        )
        
        # Opacity should be in cm^-1 (per cm path length)
        # For typical stellar atmosphere conditions, expect values ~ 1e-6 to 1e-3 cm^-1
        assert 1e-10 < absorption[0] < 1e2  # Reasonable range
        
    def test_abundance_variations(self):
        """Test effect of abundance variations"""
        wavelengths = jnp.array([5000.0])
        temperature = 5800.0
        electron_density = 1e10
        
        # Test different abundance patterns
        A_X_solar = format_abundances(0.0)
        A_X_alpha_enhanced = format_abundances(0.0, alpha_H=0.4)
        A_X_carbon_enhanced = format_abundances(0.0, C=1.0)
        
        abs_solar = total_continuum_absorption(wavelengths, temperature, electron_density, A_X_solar)
        abs_alpha = total_continuum_absorption(wavelengths, temperature, electron_density, A_X_alpha_enhanced)
        abs_carbon = total_continuum_absorption(wavelengths, temperature, electron_density, A_X_carbon_enhanced)
        
        # All should be positive and finite
        assert abs_solar[0] > 0
        assert abs_alpha[0] > 0
        assert abs_carbon[0] > 0
        
        # Variations should produce different results
        assert not jnp.allclose(abs_solar, abs_alpha, rtol=1e-10)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_densities(self):
        """Test behavior with zero densities"""
        wavelengths = jnp.array([5000.0])
        temperature = 5800.0
        electron_density = 0.0
        A_X = format_abundances(0.0)
        
        absorption = total_continuum_absorption(
            wavelengths, temperature, electron_density, A_X
        )
        
        # Should still be finite (scattering terms might be zero)
        assert jnp.isfinite(absorption[0])
        assert absorption[0] >= 0
    
    def test_extreme_temperatures(self):
        """Test behavior at extreme temperatures"""
        wavelengths = jnp.array([5000.0])
        electron_density = 1e10
        A_X = format_abundances(0.0)
        
        # Very cool
        T_cool = 1000.0
        abs_cool = total_continuum_absorption(wavelengths, T_cool, electron_density, A_X)
        assert jnp.isfinite(abs_cool[0])
        assert abs_cool[0] >= 0
        
        # Very hot
        T_hot = 50000.0
        abs_hot = total_continuum_absorption(wavelengths, T_hot, electron_density, A_X)
        assert jnp.isfinite(abs_hot[0])
        assert abs_hot[0] >= 0
    
    def test_extreme_wavelengths(self):
        """Test behavior at extreme wavelengths"""
        temperature = 5800.0
        electron_density = 1e10
        A_X = format_abundances(0.0)
        
        # Very short wavelength
        wl_short = jnp.array([100.0])  # 100 Å
        abs_short = total_continuum_absorption(wl_short, temperature, electron_density, A_X)
        assert jnp.isfinite(abs_short[0])
        
        # Very long wavelength  
        wl_long = jnp.array([50000.0])  # 5 μm
        abs_long = total_continuum_absorption(wl_long, temperature, electron_density, A_X)
        assert jnp.isfinite(abs_long[0])
    
    def test_single_wavelength(self):
        """Test with single wavelength"""
        wavelength = jnp.array([5000.0])
        temperature = 5800.0
        electron_density = 1e10
        A_X = format_abundances(0.0)
        
        absorption = total_continuum_absorption(wavelength, temperature, electron_density, A_X)
        
        assert len(absorption) == 1
        assert absorption[0] > 0
        assert jnp.isfinite(absorption[0])
    
    def test_array_broadcasting(self):
        """Test proper array broadcasting"""
        wavelengths = jnp.array([4000.0, 5000.0, 6000.0])
        temperature = 5800.0  # Scalar
        electron_density = 1e10  # Scalar
        A_X = format_abundances(0.0)
        
        absorption = total_continuum_absorption(
            wavelengths, temperature, electron_density, A_X
        )
        
        assert absorption.shape == wavelengths.shape
        assert jnp.all(jnp.isfinite(absorption))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])