"""
Unit tests for ionization equilibrium calculations.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from jorg.statmech.ionization import (
    translational_u, 
    saha_ion_weights,
    create_default_ionization_energies,
    DEFAULT_IONIZATION_ENERGIES
)
from jorg.statmech.partition_functions import create_partition_function_dict


class TestTranslationalU:
    """Test translational partition function calculations."""
    
    def test_translational_u_electron(self):
        """Test translational U for electron mass."""
        from jorg.constants import ELECTRON_MASS
        
        T = 5778.0  # Solar temperature
        result = translational_u(ELECTRON_MASS, T)
        
        # Should be a positive finite number
        assert jnp.isfinite(result)
        assert result > 0
        
        # Rough magnitude check (should be around 10^15-10^16 for these conditions)
        assert 1e14 < result < 1e17
    
    def test_translational_u_temperature_scaling(self):
        """Test temperature scaling of translational U."""
        from jorg.constants import ELECTRON_MASS
        
        T1, T2 = 3000.0, 6000.0
        result1 = translational_u(ELECTRON_MASS, T1)
        result2 = translational_u(ELECTRON_MASS, T2)
        
        # Should scale as T^1.5
        expected_ratio = (T2 / T1)**1.5
        actual_ratio = result2 / result1
        
        assert jnp.isclose(actual_ratio, expected_ratio, rtol=1e-6), \
            f"Temperature scaling failed: expected {expected_ratio}, got {actual_ratio}"
    
    def test_translational_u_mass_scaling(self):
        """Test mass scaling of translational U."""
        from jorg.constants import ELECTRON_MASS, PROTON_MASS
        
        T = 5778.0
        result_electron = translational_u(ELECTRON_MASS, T)
        result_proton = translational_u(PROTON_MASS, T)
        
        # Should scale as m^1.5
        expected_ratio = (PROTON_MASS / ELECTRON_MASS)**1.5
        actual_ratio = result_proton / result_electron
        
        assert jnp.isclose(actual_ratio, expected_ratio, rtol=1e-6), \
            f"Mass scaling failed: expected {expected_ratio}, got {actual_ratio}"


class TestSahaIonWeights:
    """Test Saha equation calculations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.T = 5778.0  # Solar temperature
        self.ne = 1e15   # Electron density cm^-3
        self.ionization_energies = create_default_ionization_energies()
        self.partition_funcs = create_partition_function_dict()
    
    def test_saha_hydrogen(self):
        """Test Saha equation for hydrogen."""
        atom = 1  # Hydrogen
        wII, wIII = saha_ion_weights(self.T, self.ne, atom, 
                                    self.ionization_energies, self.partition_funcs)
        
        # Check that results are finite and positive
        assert jnp.isfinite(wII) and wII >= 0
        assert jnp.isfinite(wIII) and wIII >= 0
        
        # For hydrogen, wIII should be 0 (no second ionization in stellar conditions)
        assert jnp.isclose(wIII, 0.0), f"Expected wIII=0 for hydrogen, got {wIII}"
        
        # wII should be small for these conditions (mostly neutral)
        assert wII < 1.0, f"Expected wII < 1 for these conditions, got {wII}"
    
    def test_saha_helium(self):
        """Test Saha equation for helium."""
        atom = 2  # Helium
        wII, wIII = saha_ion_weights(self.T, self.ne, atom,
                                    self.ionization_energies, self.partition_funcs)
        
        # Check that results are finite and positive
        assert jnp.isfinite(wII) and wII >= 0
        assert jnp.isfinite(wIII) and wIII >= 0
        
        # Both ionizations should contribute
        assert wII > 0, "Expected some He I ionization"
        # He II should be small at solar temperature
        assert wIII < wII, "Expected He III << He II at solar temperature"
    
    def test_saha_temperature_dependence(self):
        """Test temperature dependence of Saha equation."""
        atom = 1  # Hydrogen
        temperatures = [4000.0, 6000.0, 8000.0]
        
        w_values = []
        for T in temperatures:
            wII, wIII = saha_ion_weights(T, self.ne, atom,
                                       self.ionization_energies, self.partition_funcs)
            w_values.append(wII)
        
        # Ionization should increase with temperature
        assert w_values[1] > w_values[0], "Ionization should increase with temperature"
        assert w_values[2] > w_values[1], "Ionization should increase with temperature"
    
    def test_saha_density_dependence(self):
        """Test electron density dependence of Saha equation."""
        atom = 1  # Hydrogen
        densities = [1e14, 1e15, 1e16]
        
        w_values = []
        for ne in densities:
            wII, wIII = saha_ion_weights(self.T, ne, atom,
                                       self.ionization_energies, self.partition_funcs)
            w_values.append(wII)
        
        # Ionization fraction should decrease with increasing electron density
        assert w_values[1] < w_values[0], "Ionization should decrease with density"
        assert w_values[2] < w_values[1], "Ionization should decrease with density"
    
    def test_saha_conservation(self):
        """Test that ionization fractions make physical sense."""
        atom = 6  # Carbon
        wII, wIII = saha_ion_weights(self.T, self.ne, atom,
                                    self.ionization_energies, self.partition_funcs)
        
        # Calculate actual fractions
        total = 1.0 + wII + wIII
        fI = 1.0 / total
        fII = wII / total  
        fIII = wIII / total
        
        # All fractions should be between 0 and 1
        assert 0 <= fI <= 1, f"Neutral fraction out of range: {fI}"
        assert 0 <= fII <= 1, f"Single ion fraction out of range: {fII}"
        assert 0 <= fIII <= 1, f"Double ion fraction out of range: {fIII}"
        
        # Should sum to 1
        assert jnp.isclose(fI + fII + fIII, 1.0), \
            f"Fractions don't sum to 1: {fI + fII + fIII}"


class TestIonizationEnergies:
    """Test ionization energy data."""
    
    def test_default_ionization_energies_structure(self):
        """Test structure of default ionization energies."""
        energies = create_default_ionization_energies()
        
        assert isinstance(energies, dict)
        assert len(energies) > 0
        
        # Check hydrogen
        assert 1 in energies
        chi_I, chi_II, chi_III = energies[1]
        assert jnp.isclose(chi_I, 13.598, rtol=1e-3), f"H ionization energy: {chi_I}"
        assert chi_II == 0.0, "H has no second ionization in this model"
        assert chi_III == 0.0, "H has no third ionization"
    
    def test_default_ionization_energies_values(self):
        """Test some known ionization energy values."""
        energies = DEFAULT_IONIZATION_ENERGIES
        
        # Hydrogen
        assert jnp.isclose(energies[1][0], 13.598, rtol=1e-3)
        
        # Helium - check that values are reasonable
        he_first, he_second, _ = energies[2]
        assert 20 < he_first < 30, f"He I ionization energy seems wrong: {he_first}"
        assert 50 < he_second < 60, f"He II ionization energy seems wrong: {he_second}"
        
        # Check that all values are positive
        for element, (chi_I, chi_II, chi_III) in energies.items():
            assert chi_I > 0, f"Element {element} has non-positive first ionization"
            assert chi_II >= 0, f"Element {element} has negative second ionization"
            assert chi_III >= 0, f"Element {element} has negative third ionization"


if __name__ == "__main__":
    pytest.main([__file__])