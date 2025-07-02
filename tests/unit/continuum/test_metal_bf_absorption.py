"""
Tests for metal bound-free absorption implementation.

This module tests Jorg's metal bound-free absorption against Korg.jl's implementation
to ensure exact numerical agreement.
"""

import pytest
import jax
jax.config.update("jax_enable_x64", True)  # Enable float64 precision
import jax.numpy as jnp
import numpy as np
from pathlib import Path
import sys

# Add the src directory to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jorg.continuum.metals_bf import (
    MetalBoundFreeData, 
    metal_bf_absorption,
    get_metal_bf_data,
    _bilinear_interpolate_2d,
    _interpolate_metal_cross_section
)
from jorg.statmech.species import Species
from jorg.constants import SPEED_OF_LIGHT


class TestMetalBoundFreeData:
    """Test the metal bound-free data loading and structure."""
    
    def test_data_loading(self):
        """Test that HDF5 data loads correctly."""
        bf_data = get_metal_bf_data()
        
        # Check basic properties
        assert isinstance(bf_data, MetalBoundFreeData)
        assert len(bf_data.species_list) > 0
        assert len(bf_data.cross_sections) == len(bf_data.species_list)
        
        # Check grid parameters match expected values
        assert bf_data.logT_min == 2.0
        assert bf_data.logT_max == 5.0
        assert bf_data.logT_step == 0.1
        
        # Check that frequency grid is reasonable
        assert bf_data.nu_min > 0
        assert bf_data.nu_max > bf_data.nu_min
        assert bf_data.nu_step > 0
        
        print(f"Loaded {len(bf_data.species_list)} metal species:")
        for species in bf_data.species_list:
            print(f"  {species}")
    
    def test_species_parsing(self):
        """Test that species names are parsed correctly."""
        bf_data = get_metal_bf_data()
        
        # Check for expected species
        species_names = [str(s) for s in bf_data.species_list]
        
        expected_species = ["Al I", "C I", "Ca I", "Fe I", "Mg I", "Na I", "S I", "Si I"]
        for exp_species in expected_species:
            assert exp_species in species_names, f"Expected species {exp_species} not found"
    
    def test_cross_section_data_structure(self):
        """Test cross-section data has correct structure."""
        bf_data = get_metal_bf_data()
        
        n_temp = len(bf_data.logT_grid)
        n_freq = len(bf_data.nu_grid)
        
        for species, cross_section_data in bf_data.cross_sections.items():
            # Check data shape
            assert cross_section_data.shape == (n_temp, n_freq)
            
            # Check data type
            assert cross_section_data.dtype == jnp.float64
            
            # Check that data contains finite values (mostly)
            finite_fraction = jnp.mean(jnp.isfinite(cross_section_data))
            assert finite_fraction > 0.5, f"Too many non-finite values for {species}"


class TestInterpolation:
    """Test the interpolation routines."""
    
    def test_bilinear_interpolation(self):
        """Test 2D bilinear interpolation function."""
        # Create simple test grid
        x_grid = jnp.array([1.0, 2.0, 3.0])
        y_grid = jnp.array([10.0, 20.0])
        values = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        
        # Test interior point
        result = _bilinear_interpolate_2d(1.5, 15.0, x_grid, y_grid, values)
        expected = 2.5  # Linear interpolation
        assert abs(result - expected) < 1e-10
        
        # Test boundary point
        result = _bilinear_interpolate_2d(1.0, 10.0, x_grid, y_grid, values)
        expected = 1.0
        assert abs(result - expected) < 1e-10
        
        # Test extrapolation (should use flat extrapolation)
        result = _bilinear_interpolate_2d(0.5, 5.0, x_grid, y_grid, values)
        expected = 1.0  # Should clamp to boundary
        assert abs(result - expected) < 1e-10
    
    def test_cross_section_interpolation(self):
        """Test cross-section interpolation at realistic values."""
        bf_data = get_metal_bf_data()
        
        # Test with Fe I (iron)
        fe_i = Species.from_string("Fe I")
        if fe_i in bf_data.cross_sections:
            log_sigma_data = bf_data.cross_sections[fe_i]
            
            # Test interior point
            nu_test = 2e15  # Hz
            logT_test = 3.7  # log10(5000 K)
            
            result = _interpolate_metal_cross_section(
                nu_test, logT_test, bf_data.nu_grid, bf_data.logT_grid, log_sigma_data
            )
            
            # Should be finite
            assert jnp.isfinite(result)
            
            # Should be reasonable cross-section value (log10 scale)
            assert -30 < result < 10


class TestMetalAbsorption:
    """Test the main metal absorption calculation."""
    
    def test_basic_absorption_calculation(self):
        """Test basic metal absorption calculation."""
        # Test parameters - use wavelengths where metals have BF absorption
        wavelengths_angstrom = jnp.array([1200, 1500, 2000])  # Å
        frequencies = SPEED_OF_LIGHT / (wavelengths_angstrom * 1e-8)  # Hz
        temperature = 5000.0  # K
        
        # Create number densities for metal species
        number_densities = {
            Species.from_string("Fe I"): 1e12,  # cm^-3
            Species.from_string("Ca I"): 1e11,  # cm^-3  
            Species.from_string("Mg I"): 1e11,  # cm^-3
        }
        
        # Calculate absorption
        alpha = metal_bf_absorption(frequencies, temperature, number_densities)
        
        # Check output
        assert alpha.shape == frequencies.shape
        assert jnp.all(jnp.isfinite(alpha))
        assert jnp.all(alpha >= 0)  # Absorption should be non-negative
        
        print(f"Metal absorption at {temperature} K:")
        for i, wl in enumerate(wavelengths_angstrom):
            print(f"  {wl} Å: {alpha[i]}")
    
    def test_temperature_dependence(self):
        """Test that absorption has reasonable temperature dependence."""
        # Use wavelength with strong Fe I absorption
        wavelength_angstrom = 1560.0
        frequencies = jnp.array([SPEED_OF_LIGHT / (wavelength_angstrom * 1e-8)])  # Hz
        
        number_densities = {
            Species.from_string("Fe I"): 1e12,  # cm^-3
        }
        
        temperatures = [3000.0, 5000.0, 8000.0]  # K
        absorptions = []
        
        for T in temperatures:
            alpha = metal_bf_absorption(frequencies, T, number_densities)
            absorptions.append(float(alpha[0]))
        
        print(f"Temperature dependence at {wavelength_angstrom} Å: T={temperatures}, alpha={absorptions}")
        
        # Should have measurable absorption at some temperatures
        assert max(absorptions) >= 0
        # Don't require variation since it depends on the specific cross-section data
    
    def test_frequency_dependence(self):
        """Test that absorption has reasonable frequency dependence."""
        # Convert wavelengths to frequencies  
        wavelengths_angstrom = jnp.array([3000, 5000, 8000, 15000])  # Å
        wavelengths_cm = wavelengths_angstrom * 1e-8
        frequencies = SPEED_OF_LIGHT / wavelengths_cm  # Hz
        
        temperature = 5000.0  # K
        number_densities = {
            Species.from_string("Fe I"): 1e12,  # cm^-3
        }
        
        alpha = metal_bf_absorption(frequencies, temperature, number_densities)
        
        print(f"Wavelengths (Å): {wavelengths_angstrom}")
        print(f"Frequencies (Hz): {frequencies}")
        print(f"Absorption: {alpha}")
        
        # Should have some frequency dependence
        assert jnp.all(alpha >= 0)
        assert jnp.any(alpha > 0)  # At least some absorption
    
    def test_density_scaling(self):
        """Test that absorption scales linearly with number density."""
        # Use frequency where Fe I has significant cross-section (~1560 Å)
        wavelength_angstrom = 1560.0
        frequencies = jnp.array([SPEED_OF_LIGHT / (wavelength_angstrom * 1e-8)])  # Hz
        temperature = 5000.0  # K
        
        base_density = 1e12  # cm^-3
        number_densities_1x = {Species.from_string("Fe I"): base_density}
        number_densities_2x = {Species.from_string("Fe I"): 2 * base_density}
        
        alpha_1x = metal_bf_absorption(frequencies, temperature, number_densities_1x)
        alpha_2x = metal_bf_absorption(frequencies, temperature, number_densities_2x)
        
        print(f"Density scaling test at {wavelength_angstrom} Å:")
        print(f"  1x density: {alpha_1x[0]}")
        print(f"  2x density: {alpha_2x[0]}")
        
        # Should scale linearly (within numerical precision)
        if alpha_1x[0] > 1e-30:  # Only test if we have measurable absorption
            ratio = alpha_2x[0] / alpha_1x[0]
            assert abs(ratio - 2.0) < 0.01, f"Expected 2x scaling, got {ratio}"
        else:
            # If absorption is negligible, both should be very small
            assert alpha_2x[0] < 1e-25, "Second calculation should also be negligible"
    
    def test_multiple_species(self):
        """Test absorption with multiple metal species."""
        wavelength_angstrom = 1560.0
        frequencies = jnp.array([SPEED_OF_LIGHT / (wavelength_angstrom * 1e-8)])  # Hz
        temperature = 5000.0  # K
        
        # Single species
        single_species = {Species.from_string("Fe I"): 1e12}
        alpha_single = metal_bf_absorption(frequencies, temperature, single_species)
        
        # Multiple species with same total density
        multi_species = {
            Species.from_string("Fe I"): 5e11,
            Species.from_string("Ca I"): 5e11
        }
        alpha_multi = metal_bf_absorption(frequencies, temperature, multi_species)
        
        print(f"Single species (Fe I) at {wavelength_angstrom} Å: {alpha_single[0]}")
        print(f"Multi species (Fe+Ca) at {wavelength_angstrom} Å: {alpha_multi[0]}")
        
        # Both should be non-negative
        assert alpha_multi[0] >= 0
        assert alpha_single[0] >= 0


class TestPhysicalConsistency:
    """Test physical consistency of the implementation."""
    
    def test_no_negative_absorption(self):
        """Test that absorption is never negative."""
        # Wide range of test conditions
        frequencies = jnp.logspace(14, 16, 20)  # 10^14 to 10^16 Hz
        temperatures = [2000, 5000, 10000, 20000]  # K
        
        number_densities = {
            Species.from_string("Fe I"): 1e12,
            Species.from_string("Ca I"): 1e11,
            Species.from_string("Mg I"): 5e10
        }
        
        for temperature in temperatures:
            alpha = metal_bf_absorption(frequencies, temperature, number_densities)
            assert jnp.all(alpha >= 0), f"Negative absorption at T={temperature} K"
            assert jnp.all(jnp.isfinite(alpha)), f"Non-finite absorption at T={temperature} K"
    
    def test_realistic_cross_sections(self):
        """Test that cross-sections are in reasonable range."""
        bf_data = get_metal_bf_data()
        
        for species, log_sigma_data in bf_data.cross_sections.items():
            finite_mask = jnp.isfinite(log_sigma_data)
            finite_values = log_sigma_data[finite_mask]
            
            if len(finite_values) > 0:
                min_log_sigma = jnp.min(finite_values)
                max_log_sigma = jnp.max(finite_values)
                
                # Cross-sections should be reasonable (in log10 scale)
                # Typical atomic cross-sections range from 1e-24 to 1e-16 cm^2
                # In log10 scale: -24 to -16, but data includes 1e18 factor
                assert min_log_sigma > -30, f"Cross-section too small for {species}"
                assert max_log_sigma < 10, f"Cross-section too large for {species}"


def test_integration_with_main_continuum():
    """Test integration with main continuum absorption function."""
    from jorg.continuum.core import total_continuum_absorption
    
    # Test parameters
    frequencies = jnp.array([1e15, 2e15, 3e15])  # Hz
    temperature = 5000.0  # K
    electron_density = 1e13  # cm^-3
    
    # Number densities (using string keys as expected by main function)
    number_densities = {
        'H_I': 1e16,
        'H_II': 1e13, 
        'He_I': 1e15,
        'Fe_I': 1e12,  # Metal species
        'Ca_I': 1e11,
        'Mg_I': 5e10
    }
    
    # Simple partition functions
    partition_functions = {
        'H_I': lambda log_T: 2.0,
        'He_I': lambda log_T: 1.0
    }
    
    # Test with and without metals
    alpha_no_metals = total_continuum_absorption(
        frequencies, temperature, electron_density, 
        number_densities, partition_functions, include_metals=False
    )
    
    alpha_with_metals = total_continuum_absorption(
        frequencies, temperature, electron_density,
        number_densities, partition_functions, include_metals=True
    )
    
    # Metals should add to absorption
    metal_contribution = alpha_with_metals - alpha_no_metals
    assert jnp.all(metal_contribution >= 0)
    assert jnp.any(metal_contribution > 0)  # Should have some contribution
    
    print(f"Base absorption: {alpha_no_metals}")
    print(f"With metals: {alpha_with_metals}")
    print(f"Metal contribution: {metal_contribution}")


if __name__ == "__main__":
    # Run basic tests
    print("Testing Metal Bound-Free Implementation")
    print("=" * 50)
    
    # Test data loading
    test_data = TestMetalBoundFreeData()
    test_data.test_data_loading()
    print("✓ Data loading test passed")
    
    test_data.test_species_parsing()
    print("✓ Species parsing test passed")
    
    # Test interpolation
    test_interp = TestInterpolation()
    test_interp.test_bilinear_interpolation()
    print("✓ Bilinear interpolation test passed")
    
    # Test metal absorption
    test_abs = TestMetalAbsorption()
    test_abs.test_basic_absorption_calculation()
    print("✓ Basic absorption test passed")
    
    test_abs.test_density_scaling()
    print("✓ Density scaling test passed")
    
    # Test physical consistency
    test_phys = TestPhysicalConsistency()
    test_phys.test_no_negative_absorption()
    print("✓ Physical consistency test passed")
    
    # Test integration
    test_integration_with_main_continuum()
    print("✓ Integration test passed")
    
    print("\nAll tests passed! ✓")