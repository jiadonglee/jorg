#!/usr/bin/env python3
"""
Comprehensive validation tests for Jorg hydrogen line consistency with Korg.jl

This test suite validates that Jorg's hydrogen line implementation produces
results consistent with Korg.jl across a range of stellar conditions.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jorg.lines.hydrogen_lines import (
    hummer_mihalas_w, sigma_line, brackett_oscillator_strength, 
    griem_1960_Knm, hydrogen_line_absorption, load_stark_profiles
)
from jorg.constants import (
    kboltz_eV, RydbergH_eV, hplanck_eV, c_cgs, 
    bohr_radius_cgs, electron_charge_cgs, electron_mass_cgs
)

class TestPhysicalConstants:
    """Test that all physical constants exactly match Korg.jl values."""
    
    def test_kboltz_eV(self):
        """Test Boltzmann constant in eV/K"""
        expected = 8.617333262145e-5
        assert abs(kboltz_eV - expected) < 1e-15
        
    def test_rydberg_eV(self):
        """Test Rydberg constant in eV"""
        expected = 13.598287264
        assert abs(RydbergH_eV - expected) < 1e-9
        
    def test_hplanck_eV(self):
        """Test Planck constant in eV*s"""
        expected = 4.135667696e-15
        assert abs(hplanck_eV - expected) < 1e-21
        
    def test_bohr_radius(self):
        """Test Bohr radius in cm"""
        expected = 5.29177210903e-9
        assert abs(bohr_radius_cgs - expected) < 1e-19
        
    def test_electron_mass(self):
        """Test electron mass in g"""
        expected = 9.1093897e-28
        assert abs(electron_mass_cgs - expected) < 1e-34


class TestMHDFormalism:
    """Test MHD occupation probability formalism against known values."""
    
    @pytest.mark.parametrize("n_eff,expected", [
        (1.0, 1.000000),
        (2.0, 0.999996), 
        (3.0, 0.999971),
        (10.0, 0.969852),
        (20.0, 0.114613)
    ])
    def test_mhd_solar_photosphere(self, n_eff, expected):
        """Test MHD w values for solar photosphere conditions"""
        T = 5778.0
        ne = 1e13
        nH = 1e16  
        nHe = 1e15
        
        w = hummer_mihalas_w(T, n_eff, nH, nHe, ne)
        assert abs(w - expected) < 2e-5, f"n={n_eff}: got {w}, expected {expected}"
        
    def test_mhd_pressure_effects(self):
        """Test pressure ionization effects across stellar conditions"""
        T = 5778.0
        nH = 1e16
        nHe = 1e15
        n_eff = 20.0
        
        test_cases = [
            (1e11, 0.174372),  # Low density photosphere
            (1e13, 0.114613),  # Solar photosphere  
            (1e15, 0.000000),  # Deep atmosphere
            (1e17, 0.000000)   # Extreme pressure
        ]
        
        for ne, expected in test_cases:
            w = hummer_mihalas_w(T, n_eff, nH, nHe, ne)
            assert abs(w - expected) < 2e-5, f"ne={ne:.0e}: got {w}, expected {expected}"


class TestSigmaLine:
    """Test sigma_line cross-section calculations."""
    
    def test_sigma_line_calculation(self):
        """Test sigma_line gives correct quantum mechanical cross-section"""
        # Test for Hα wavelength
        lambda_Ha = 6.563e-5  # cm
        sigma = sigma_line(lambda_Ha)
        
        # Expected value from quantum mechanics: σ = (π * e^2 * λ^2) / (m_e * c^2)
        expected = (jnp.pi * electron_charge_cgs**2 * lambda_Ha**2 / 
                   (electron_mass_cgs * c_cgs**2))
        
        assert abs(sigma - expected) < 1e-30
        
    def test_sigma_line_wavelength_scaling(self):
        """Test that sigma_line scales as λ^2"""
        lambda1 = 4000e-8  # 4000 Å
        lambda2 = 8000e-8  # 8000 Å
        
        sigma1 = sigma_line(lambda1)
        sigma2 = sigma_line(lambda2)
        
        # Should scale as λ^2
        ratio = sigma2 / sigma1
        expected_ratio = (lambda2 / lambda1)**2
        
        assert abs(ratio - expected_ratio) < 1e-12


class TestBrackettOscillatorStrength:
    """Test Brackett oscillator strength calculations."""
    
    @pytest.mark.parametrize("n,m,expected", [
        (4, 5, 1.0377),
        (4, 6, 0.1794),
        (4, 7, 0.0655),
        (4, 10, 0.0119)
    ])
    def test_brackett_values(self, n, m, expected):
        """Test specific Brackett oscillator strength values"""
        f_nm = brackett_oscillator_strength(n, m)
        assert abs(f_nm - expected) < 1e-4
        
    def test_brackett_series_consistency(self):
        """Test that Brackett series follows expected patterns"""
        n = 4
        oscillator_strengths = []
        
        for m in range(5, 20):
            f_nm = brackett_oscillator_strength(n, m)
            oscillator_strengths.append(f_nm)
            
        # Oscillator strengths should generally decrease with increasing m
        for i in range(len(oscillator_strengths) - 1):
            assert oscillator_strengths[i] > oscillator_strengths[i + 1]


class TestGriemConstants:
    """Test Griem 1960 Knm constants."""
    
    @pytest.mark.parametrize("n,m,expected", [
        (1, 2, 0.0001716),
        (1, 3, 0.0005235),
        (1, 4, 0.0008912),
        (2, 3, 0.0090190),
        (2, 4, 0.0177200),
        (3, 4, 0.1001000)
    ])
    def test_tabulated_knm_values(self, n, m, expected):
        """Test tabulated Griem Knm constants"""
        knm = griem_1960_Knm(n, m)
        assert abs(knm - expected) < 1e-7
        
    def test_analytical_knm_formula(self):
        """Test analytical Knm formula for high quantum numbers"""
        n, m = 4, 15
        knm = griem_1960_Knm(n, m)
        
        # Should use analytical formula: 5.5e-5 * n^4 * m^4 / (m^2 - n^2) / (1 + 0.13 / (m - n))
        expected = 5.5e-5 * n**4 * m**4 / (m**2 - n**2) / (1 + 0.13 / (m - n))
        assert abs(knm - expected) < 1e-8


class TestStarkProfiles:
    """Test Stark profile loading and functionality."""
    
    def test_stark_profile_loading(self):
        """Test that Stark profiles can be loaded without errors"""
        profiles = load_stark_profiles()
        # Should not crash - may return empty dict if data file not available
        assert isinstance(profiles, dict)
        
    def test_stark_profile_structure(self):
        """Test Stark profile data structure when available"""
        profiles = load_stark_profiles()
        
        if profiles:  # Only test if profiles were successfully loaded
            for key, profile_data in profiles.items():
                required_keys = ['temps', 'electron_number_densities', 'profile', 
                               'lower', 'upper', 'Kalpha', 'log_gf', 'lambda0']
                for req_key in required_keys:
                    assert req_key in profile_data, f"Missing key {req_key} in profile {key}"


class TestHydrogenLineAbsorption:
    """Test full hydrogen line absorption calculations."""
    
    def test_solar_photosphere_conditions(self):
        """Test hydrogen line absorption for solar photosphere"""
        # Solar photosphere conditions
        T = 5778.0
        ne = 1e13
        nH_I = 1e16
        nHe_I = 1e15  
        UH_I = 2.0
        xi = 1e5  # 1 km/s in cm/s
        
        # Wavelength grid around Hα
        wavelengths = jnp.linspace(6560e-8, 6570e-8, 100)
        
        absorption = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi
        )
        
        # Should have non-zero absorption
        assert jnp.max(absorption) > 0
        
        # Should have peak around Hα center (6563 Å)
        peak_idx = jnp.argmax(absorption)
        peak_wavelength = wavelengths[peak_idx]
        assert abs(peak_wavelength - 6563e-8) < 5e-8  # Within 0.5 Å
        
    def test_mhd_effect_on_absorption(self):
        """Test that MHD formalism affects high-n transitions"""
        T = 5778.0
        ne = 1e15  # High density to see MHD effects
        nH_I = 1e16
        nHe_I = 1e15
        UH_I = 2.0
        xi = 1e5
        
        wavelengths = jnp.linspace(1.8e-4, 2.0e-4, 50)  # Brackett range
        
        # Calculate with and without MHD
        absorption_mhd = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, use_MHD=True, n_max=30
        )
        
        absorption_no_mhd = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, use_MHD=False, n_max=30
        )
        
        # MHD should reduce absorption for high-n transitions at high density
        assert jnp.max(absorption_mhd) < jnp.max(absorption_no_mhd)
        
    def test_temperature_dependence(self):
        """Test temperature dependence of line absorption"""
        ne = 1e13
        nH_I = 1e16
        nHe_I = 1e15
        UH_I = 2.0
        xi = 1e5
        
        wavelengths = jnp.linspace(6560e-8, 6570e-8, 50)
        
        temperatures = [4000, 5000, 6000, 7000]
        max_absorptions = []
        
        for T in temperatures:
            absorption = hydrogen_line_absorption(
                wavelengths, T, ne, nH_I, nHe_I, UH_I, xi
            )
            max_absorptions.append(float(jnp.max(absorption)))
            
        # Absorption should vary with temperature (complex dependence)
        assert len(set(max_absorptions)) == len(max_absorptions)  # All different


class TestConsistencyWithKorg:
    """High-level consistency tests mimicking Korg.jl behavior."""
    
    def test_hydrogen_line_series_coverage(self):
        """Test that major hydrogen series are covered"""
        T = 5778.0
        ne = 1e13
        nH_I = 1e16
        nHe_I = 1e15
        UH_I = 2.0
        xi = 1e5
        
        # Test different spectral regions
        test_ranges = [
            (1000e-8, 2000e-8),   # UV - Lyman series
            (3000e-8, 7000e-8),   # Optical - Balmer series  
            (15000e-8, 25000e-8)  # IR - Brackett series
        ]
        
        for wl_start, wl_end in test_ranges:
            wavelengths = jnp.linspace(wl_start, wl_end, 200)
            absorption = hydrogen_line_absorption(
                wavelengths, T, ne, nH_I, nHe_I, UH_I, xi, n_max=20
            )
            
            # Should have some absorption in each range
            assert jnp.max(absorption) > 0, f"No absorption in range {wl_start:.0e}-{wl_end:.0e}"
            
    def test_physical_reasonableness(self):
        """Test that results are physically reasonable"""
        T = 5778.0
        ne = 1e13
        nH_I = 1e16
        nHe_I = 1e15
        UH_I = 2.0
        xi = 1e5
        
        wavelengths = jnp.linspace(6560e-8, 6570e-8, 100)
        absorption = hydrogen_line_absorption(
            wavelengths, T, ne, nH_I, nHe_I, UH_I, xi
        )
        
        # Check physical constraints
        assert jnp.all(absorption >= 0), "Negative absorption coefficients"
        assert jnp.all(jnp.isfinite(absorption)), "Non-finite absorption coefficients"
        assert jnp.max(absorption) < 1e5, "Unreasonably large absorption"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])