"""
Unit tests for the Jorg lines module

Tests line absorption calculations, profiles, and broadening mechanisms.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jorg.lines.core import total_line_absorption, line_absorption_single
from jorg.lines.profiles import voigt_profile, gaussian_profile, lorentzian_profile
from jorg.lines.broadening import natural_broadening, stark_broadening, vdw_broadening
from jorg.lines.datatypes import Line, Species
from jorg.lines.opacity import line_opacity_coefficient
from jorg.synthesis import format_abundances, interpolate_atmosphere


class TestLineProfiles:
    """Test line profile calculations"""
    
    def test_gaussian_profile(self):
        """Test Gaussian line profile"""
        wavelengths = jnp.linspace(5000, 6000, 1000)
        line_center = 5500.0
        sigma = 0.1  # Angstroms
        
        profile = gaussian_profile(wavelengths, line_center, sigma)
        
        # Check normalization (approximately)
        dw = wavelengths[1] - wavelengths[0]
        integral = jnp.sum(profile) * dw
        np.testing.assert_allclose(integral, 1.0, rtol=1e-2)
        
        # Check peak is at center
        peak_idx = jnp.argmax(profile)
        assert abs(wavelengths[peak_idx] - line_center) < dw
        
        # Check symmetry (approximately)
        center_idx = jnp.argmin(jnp.abs(wavelengths - line_center))
        left_val = profile[center_idx - 10]
        right_val = profile[center_idx + 10]
        np.testing.assert_allclose(left_val, right_val, rtol=1e-10)
    
    def test_lorentzian_profile(self):
        """Test Lorentzian line profile"""
        wavelengths = jnp.linspace(5000, 6000, 1000)
        line_center = 5500.0
        gamma = 0.05  # Angstroms
        
        profile = lorentzian_profile(wavelengths, line_center, gamma)
        
        # Check peak is at center
        peak_idx = jnp.argmax(profile)
        dw = wavelengths[1] - wavelengths[0]
        assert abs(wavelengths[peak_idx] - line_center) < dw
        
        # Check that profile is positive
        assert jnp.all(profile >= 0)
        
        # Check FWHM relationship
        half_max = jnp.max(profile) / 2
        indices = jnp.where(profile >= half_max)[0]
        fwhm_measured = wavelengths[indices[-1]] - wavelengths[indices[0]]
        fwhm_expected = 2 * gamma
        np.testing.assert_allclose(fwhm_measured, fwhm_expected, rtol=0.1)
    
    def test_voigt_profile(self):
        """Test Voigt line profile (Gaussian + Lorentzian convolution)"""
        wavelengths = jnp.linspace(5499, 5501, 200)
        line_center = 5500.0
        sigma = 0.05  # Gaussian width
        gamma = 0.03  # Lorentzian width
        
        profile = voigt_profile(wavelengths, line_center, sigma, gamma)
        
        # Check normalization (approximately)
        dw = wavelengths[1] - wavelengths[0]
        integral = jnp.sum(profile) * dw
        np.testing.assert_allclose(integral, 1.0, rtol=1e-2)
        
        # Check peak is at center
        peak_idx = jnp.argmax(profile)
        assert abs(wavelengths[peak_idx] - line_center) < dw
        
        # Voigt should be between pure Gaussian and pure Lorentzian
        gaussian = gaussian_profile(wavelengths, line_center, sigma)
        lorentzian = lorentzian_profile(wavelengths, line_center, gamma)
        
        # At line center, Voigt should be between the two
        center_idx = jnp.argmin(jnp.abs(wavelengths - line_center))
        assert min(gaussian[center_idx], lorentzian[center_idx]) <= profile[center_idx] <= max(gaussian[center_idx], lorentzian[center_idx])
    
    def test_profile_broadening_limits(self):
        """Test limiting cases of profile broadening"""
        wavelengths = jnp.linspace(5499, 5501, 200)
        line_center = 5500.0
        
        # Pure Gaussian limit (gamma → 0)
        profile_gaussian = voigt_profile(wavelengths, line_center, 0.05, 1e-10)
        profile_gaussian_pure = gaussian_profile(wavelengths, line_center, 0.05)
        np.testing.assert_allclose(profile_gaussian, profile_gaussian_pure, rtol=1e-6)
        
        # Pure Lorentzian limit (sigma → 0)  
        profile_lorentzian = voigt_profile(wavelengths, line_center, 1e-10, 0.03)
        profile_lorentzian_pure = lorentzian_profile(wavelengths, line_center, 0.03)
        np.testing.assert_allclose(profile_lorentzian, profile_lorentzian_pure, rtol=1e-6)


class TestBroadening:
    """Test broadening mechanism calculations"""
    
    def test_natural_broadening(self):
        """Test natural (radiative) broadening"""
        wavelength = 5500.0  # Angstroms
        oscillator_strength = 1.0
        
        gamma_rad = natural_broadening(wavelength, oscillator_strength)
        
        # Should be positive
        assert gamma_rad > 0
        
        # Should scale with oscillator strength
        gamma_rad_strong = natural_broadening(wavelength, 2.0)
        assert gamma_rad_strong > gamma_rad
        
        # Should scale with wavelength (λ^-2 dependence)
        gamma_rad_short = natural_broadening(4000.0, oscillator_strength)
        gamma_rad_long = natural_broadening(7000.0, oscillator_strength)
        assert gamma_rad_short > gamma_rad_long
    
    def test_stark_broadening(self):
        """Test Stark (pressure) broadening"""
        wavelength = 5500.0
        temperature = 5800.0
        electron_density = 1e10  # cm^-3
        stark_constant = 1e-15  # Example value
        
        gamma_stark = stark_broadening(wavelength, temperature, electron_density, stark_constant)
        
        # Should be positive
        assert gamma_stark >= 0
        
        # Should scale with electron density
        gamma_stark_dense = stark_broadening(wavelength, temperature, 1e12, stark_constant)
        assert gamma_stark_dense > gamma_stark
        
        # Should have temperature dependence
        gamma_stark_hot = stark_broadening(wavelength, 8000.0, electron_density, stark_constant)
        # Stark broadening typically increases with temperature
        assert jnp.isfinite(gamma_stark_hot)
    
    def test_vdw_broadening(self):
        """Test van der Waals broadening"""
        wavelength = 5500.0
        temperature = 5800.0
        neutral_density = 1e12  # cm^-3
        vdw_constant = 1e-30  # Example value
        
        gamma_vdw = vdw_broadening(wavelength, temperature, neutral_density, vdw_constant)
        
        # Should be positive
        assert gamma_vdw >= 0
        
        # Should scale with neutral density
        gamma_vdw_dense = vdw_broadening(wavelength, temperature, 1e14, vdw_constant)
        assert gamma_vdw_dense > gamma_vdw
        
        # Should have temperature dependence (typically T^0.3)
        gamma_vdw_hot = vdw_broadening(wavelength, 8000.0, neutral_density, vdw_constant)
        assert gamma_vdw_hot > gamma_vdw  # Higher T gives more broadening
    
    def test_broadening_combinations(self):
        """Test combination of broadening mechanisms"""
        wavelength = 5500.0
        temperature = 5800.0
        electron_density = 1e10
        neutral_density = 1e12
        
        gamma_nat = natural_broadening(wavelength, 1.0)
        gamma_stark = stark_broadening(wavelength, temperature, electron_density, 1e-15)
        gamma_vdw = vdw_broadening(wavelength, temperature, neutral_density, 1e-30)
        
        # Total broadening (quadrature sum for Voigt profile)
        gamma_total = jnp.sqrt(gamma_nat**2 + gamma_stark**2 + gamma_vdw**2)
        
        # Total should be larger than any individual component
        assert gamma_total >= gamma_nat
        assert gamma_total >= gamma_stark
        assert gamma_total >= gamma_vdw


class TestLineDataTypes:
    """Test line and species data structures"""
    
    def test_species_creation(self):
        """Test Species data structure"""
        # Neutral iron
        fe_i = Species(element=26, ionization=0)
        assert fe_i.element == 26
        assert fe_i.ionization == 0
        
        # Singly ionized calcium
        ca_ii = Species(element=20, ionization=1)
        assert ca_ii.element == 20
        assert ca_ii.ionization == 1
    
    def test_line_creation(self):
        """Test Line data structure"""
        species = Species(element=26, ionization=0)  # Fe I
        
        line = Line(
            wavelength=5500.0,
            species=species,
            excitation_potential=2.5,  # eV
            log_gf=-1.0,
            radiative_damping=1e8,
            stark_damping=1e-15,
            vdw_damping=1e-30
        )
        
        assert line.wavelength == 5500.0
        assert line.species.element == 26
        assert line.excitation_potential == 2.5
        assert line.log_gf == -1.0
        
    def test_line_strength_calculation(self):
        """Test line strength calculations"""
        species = Species(element=26, ionization=0)
        line = Line(
            wavelength=5500.0,
            species=species,
            excitation_potential=2.0,
            log_gf=-1.0,
            radiative_damping=1e8,
            stark_damping=1e-15,
            vdw_damping=1e-30
        )
        
        temperature = 5800.0
        partition_function = 100.0
        abundance = 1e-5  # N/N_total
        
        # Test that we can calculate line opacity coefficient
        opacity_coeff = line_opacity_coefficient(
            line, temperature, partition_function, abundance
        )
        
        assert opacity_coeff > 0
        assert jnp.isfinite(opacity_coeff)


class TestLineAbsorption:
    """Test line absorption calculations"""
    
    def test_single_line_absorption(self):
        """Test absorption from a single line"""
        wavelengths = jnp.linspace(5499, 5501, 100)
        
        # Create test line
        species = Species(element=26, ionization=0)
        line = Line(
            wavelength=5500.0,
            species=species,
            excitation_potential=2.0,
            log_gf=-1.0,
            radiative_damping=1e8,
            stark_damping=1e-15,
            vdw_damping=1e-30
        )
        
        # Test atmosphere layer
        temperature = 5800.0
        electron_density = 1e10
        neutral_density = 1e12
        microturbulence = 1.0  # km/s
        
        absorption = line_absorption_single(
            wavelengths, line, temperature, electron_density, 
            neutral_density, microturbulence
        )
        
        # Check basic properties
        assert len(absorption) == len(wavelengths)
        assert jnp.all(absorption >= 0)
        assert jnp.all(jnp.isfinite(absorption))
        
        # Peak should be near line center
        peak_idx = jnp.argmax(absorption) 
        peak_wavelength = wavelengths[peak_idx]
        assert abs(peak_wavelength - line.wavelength) < 0.1
    
    def test_total_line_absorption(self):
        """Test total absorption from multiple lines"""
        wavelengths = jnp.linspace(5000, 6000, 1000)
        
        # Create test atmosphere
        A_X = format_abundances(0.0)
        atm = interpolate_atmosphere(5800, 4.0, A_X)
        
        # Create test linelist
        lines = []
        for wl in [5200.0, 5400.0, 5600.0, 5800.0]:
            species = Species(element=26, ionization=0)
            line = Line(
                wavelength=wl,
                species=species,
                excitation_potential=2.0,
                log_gf=-1.0,
                radiative_damping=1e8,
                stark_damping=1e-15,
                vdw_damping=1e-30
            )
            lines.append(line)
        
        microturbulence = 1.0
        
        absorption = total_line_absorption(
            wavelengths, lines, atm, A_X, microturbulence
        )
        
        # Check shape
        assert absorption.shape == (atm['n_layers'], len(wavelengths))
        assert jnp.all(absorption >= 0)
        assert jnp.all(jnp.isfinite(absorption))
        
        # Should have peaks near line centers
        total_abs = jnp.sum(absorption, axis=0)
        for line in lines:
            line_idx = jnp.argmin(jnp.abs(wavelengths - line.wavelength))
            # Check that there's significant absorption near line center
            assert total_abs[line_idx] > jnp.mean(total_abs) * 0.1
    
    def test_line_absorption_temperature_dependence(self):
        """Test temperature dependence of line absorption"""
        wavelengths = jnp.linspace(5499, 5501, 50)
        
        species = Species(element=26, ionization=0)
        line = Line(
            wavelength=5500.0,
            species=species,
            excitation_potential=3.0,  # Higher excitation
            log_gf=-1.0,
            radiative_damping=1e8,
            stark_damping=1e-15,
            vdw_damping=1e-30
        )
        
        electron_density = 1e10
        neutral_density = 1e12
        microturbulence = 1.0
        
        # Test different temperatures
        T_cool = 4000.0
        T_hot = 8000.0
        
        abs_cool = line_absorption_single(
            wavelengths, line, T_cool, electron_density, neutral_density, microturbulence
        )
        abs_hot = line_absorption_single(
            wavelengths, line, T_hot, electron_density, neutral_density, microturbulence
        )
        
        # High excitation lines should be stronger at higher temperatures
        assert jnp.max(abs_hot) > jnp.max(abs_cool)
    
    def test_line_absorption_microturbulence(self):
        """Test microturbulence effect on line profiles"""
        wavelengths = jnp.linspace(5499, 5501, 100)
        
        species = Species(element=26, ionization=0)
        line = Line(
            wavelength=5500.0,
            species=species,
            excitation_potential=2.0,
            log_gf=-1.0,
            radiative_damping=1e8,
            stark_damning=1e-15,
            vdw_damping=1e-30
        )
        
        temperature = 5800.0
        electron_density = 1e10
        neutral_density = 1e12
        
        # Test different microturbulence values
        vmic_low = 0.5  # km/s
        vmic_high = 2.0  # km/s
        
        abs_low = line_absorption_single(
            wavelengths, line, temperature, electron_density, neutral_density, vmic_low
        )
        abs_high = line_absorption_single(
            wavelengths, line, temperature, electron_density, neutral_density, vmic_high
        )
        
        # Higher microturbulence should broaden lines
        # Peak absorption should be lower, but integrated strength similar
        assert jnp.max(abs_high) < jnp.max(abs_low)
        
        # Integrated strength should be similar
        dw = wavelengths[1] - wavelengths[0]
        strength_low = jnp.sum(abs_low) * dw
        strength_high = jnp.sum(abs_high) * dw
        np.testing.assert_allclose(strength_low, strength_high, rtol=0.1)


class TestLineListOperations:
    """Test operations on line lists"""
    
    def test_line_selection_by_wavelength(self):
        """Test selecting lines within wavelength range"""
        # Create test linelist
        lines = []
        wavelengths = [4500, 5000, 5500, 6000, 6500]
        for wl in wavelengths:
            species = Species(element=26, ionization=0)
            line = Line(
                wavelength=wl,
                species=species,
                excitation_potential=2.0,
                log_gf=-1.0,
                radiative_damping=1e8,
                stark_damping=1e-15,
                vdw_damping=1e-30
            )
            lines.append(line)
        
        # Select lines in range
        wl_min, wl_max = 5200, 5800
        selected_lines = [line for line in lines 
                         if wl_min <= line.wavelength <= wl_max]
        
        assert len(selected_lines) == 1  # Only 5500 Å line
        assert selected_lines[0].wavelength == 5500.0
    
    def test_line_strength_filtering(self):
        """Test filtering lines by strength"""
        lines = []
        log_gf_values = [-3.0, -2.0, -1.0, 0.0, 1.0]
        
        for log_gf in log_gf_values:
            species = Species(element=26, ionization=0)
            line = Line(
                wavelength=5500.0,
                species=species,
                excitation_potential=2.0,
                log_gf=log_gf,
                radiative_damping=1e8,
                stark_damping=1e-15,
                vdw_damping=1e-30
            )
            lines.append(line)
        
        # Filter strong lines
        strong_lines = [line for line in lines if line.log_gf > -2.0]
        assert len(strong_lines) == 3  # log_gf = -1, 0, 1
    
    def test_species_filtering(self):
        """Test filtering lines by species"""
        lines = []
        elements = [26, 20, 12, 1]  # Fe, Ca, Mg, H
        
        for element in elements:
            species = Species(element=element, ionization=0)
            line = Line(
                wavelength=5500.0,
                species=species,
                excitation_potential=2.0,
                log_gf=-1.0,
                radiative_damping=1e8,
                stark_damping=1e-15,
                vdw_damping=1e-30
            )
            lines.append(line)
        
        # Filter iron lines
        fe_lines = [line for line in lines if line.species.element == 26]
        assert len(fe_lines) == 1
        assert fe_lines[0].species.element == 26


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_zero_abundance(self):
        """Test behavior with zero abundance"""
        wavelengths = jnp.linspace(5499, 5501, 50)
        
        species = Species(element=26, ionization=0)
        line = Line(
            wavelength=5500.0,
            species=species,
            excitation_potential=2.0,
            log_gf=-1.0,
            radiative_damping=1e8,
            stark_damping=1e-15,
            vdw_damping=1e-30
        )
        
        temperature = 5800.0
        electron_density = 1e10
        neutral_density = 0.0  # Zero abundance
        microturbulence = 1.0
        
        absorption = line_absorption_single(
            wavelengths, line, temperature, electron_density, 
            neutral_density, microturbulence
        )
        
        # Should be zero or very small
        assert jnp.all(absorption >= 0)
        assert jnp.max(absorption) < 1e-10
    
    def test_extreme_temperatures(self):
        """Test behavior at extreme temperatures"""
        wavelengths = jnp.array([5500.0])
        
        species = Species(element=26, ionization=0)
        line = Line(
            wavelength=5500.0,
            species=species,
            excitation_potential=2.0,
            log_gf=-1.0,
            radiative_damping=1e8,
            stark_damping=1e-15,
            vdw_damping=1e-30
        )
        
        electron_density = 1e10
        neutral_density = 1e12
        microturbulence = 1.0
        
        # Very cool temperature
        abs_cool = line_absorption_single(
            wavelengths, line, 1000.0, electron_density, neutral_density, microturbulence
        )
        assert jnp.isfinite(abs_cool[0])
        
        # Very hot temperature
        abs_hot = line_absorption_single(
            wavelengths, line, 50000.0, electron_density, neutral_density, microturbulence
        )
        assert jnp.isfinite(abs_hot[0])
    
    def test_very_weak_lines(self):
        """Test very weak lines"""
        wavelengths = jnp.linspace(5499, 5501, 50)
        
        species = Species(element=26, ionization=0)
        line = Line(
            wavelength=5500.0,
            species=species,
            excitation_potential=10.0,  # Very high excitation
            log_gf=-5.0,  # Very weak
            radiative_damping=1e8,
            stark_damping=1e-15,
            vdw_damping=1e-30
        )
        
        temperature = 5800.0
        electron_density = 1e10
        neutral_density = 1e12
        microturbulence = 1.0
        
        absorption = line_absorption_single(
            wavelengths, line, temperature, electron_density, 
            neutral_density, microturbulence
        )
        
        # Should be very small but finite
        assert jnp.all(jnp.isfinite(absorption))
        assert jnp.all(absorption >= 0)
        assert jnp.max(absorption) < 1e-5


if __name__ == '__main__':
    pytest.main([__file__, '-v'])