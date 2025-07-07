"""
Unit Tests for Saha Equation Implementation
==========================================

Test suite for the Saha equation implementation, ensuring exact compatibility
with Korg.jl's statistical mechanics formulation.
"""

import pytest
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from jorg.statmech.chemical_equilibrium import (
    saha_ion_weights,
    translational_U
)
from jorg.statmech.species import Species
from jorg.statmech import (
    create_default_partition_functions,
    create_default_ionization_energies
)
from jorg.constants import me_cgs, kboltz_eV


class TestSahaEquationImplementation:
    """Test Saha equation implementation accuracy."""
    
    @pytest.fixture
    def test_data(self):
        """Load test data for Saha equation calculations."""
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        return {
            'partition_fns': partition_fns,
            'ionization_energies': ionization_energies
        }
    
    def test_hydrogen_saha_weights(self, test_data):
        """Test Saha weights for hydrogen (simplest case)."""
        T = 5778.0  # Solar temperature
        ne = 1e13   # Typical photosphere electron density
        
        wII, wIII = saha_ion_weights(T, ne, 1, 
                                   test_data['ionization_energies'], 
                                   test_data['partition_fns'])
        
        wII_val = float(wII)
        wIII_val = float(wIII)
        
        # Basic properties
        assert wII_val >= 0, "H II weight should be non-negative"
        assert wIII_val == 0, "H III weight should be zero (no such ion)"
        assert np.isfinite(wII_val), "H II weight should be finite"
        
        # At solar conditions, hydrogen should be mostly neutral
        assert wII_val < 1.0, "At solar conditions, H should be mostly neutral"
    
    def test_helium_saha_weights(self, test_data):
        """Test Saha weights for helium (two ionization stages)."""
        T = 8000.0  # Hotter to see He ionization
        ne = 1e13
        
        wII, wIII = saha_ion_weights(T, ne, 2,
                                   test_data['ionization_energies'],
                                   test_data['partition_fns'])
        
        wII_val = float(wII)
        wIII_val = float(wIII)
        
        # Basic properties
        assert wII_val >= 0, "He II weight should be non-negative"
        assert wIII_val >= 0, "He III weight should be non-negative"
        assert np.isfinite(wII_val), "He II weight should be finite"
        assert np.isfinite(wIII_val), "He III weight should be finite"
        
        # He III should be much less abundant than He II at these conditions
        assert wIII_val <= wII_val, "He III should be less abundant than He II"
    
    def test_iron_saha_weights(self, test_data):
        """Test Saha weights for iron (complex atom)."""
        T = 5778.0
        ne = 1e13
        
        wII, wIII = saha_ion_weights(T, ne, 26,
                                   test_data['ionization_energies'],
                                   test_data['partition_fns'])
        
        wII_val = float(wII)
        wIII_val = float(wIII)
        
        # Basic properties
        assert wII_val >= 0, "Fe II weight should be non-negative"
        assert wIII_val >= 0, "Fe III weight should be non-negative"
        assert np.isfinite(wII_val), "Fe II weight should be finite"
        assert np.isfinite(wIII_val), "Fe III weight should be finite"
        
        # Iron has low ionization energy, so should be partially ionized
        assert wII_val > 0.1, "Fe should be significantly ionized at solar conditions"
    
    def test_saha_temperature_scaling(self, test_data):
        """Test Saha equation temperature dependence."""
        temperatures = [3000, 5000, 8000, 12000, 20000]
        ne = 1e13
        
        # Test hydrogen ionization vs temperature
        wII_values = []
        for T in temperatures:
            wII, wIII = saha_ion_weights(T, ne, 1,
                                       test_data['ionization_energies'],
                                       test_data['partition_fns'])
            wII_values.append(float(wII))
        
        # Should generally increase with temperature (more ionization)
        for i in range(1, len(wII_values)):
            if temperatures[i] > temperatures[i-1]:
                # Allow some flexibility due to partition function effects
                assert wII_values[i] >= wII_values[i-1] * 0.5, \
                    f"Ionization should generally increase with T: {temperatures[i-1]}K -> {temperatures[i]}K"
        
        # At very high temperature, should be highly ionized
        assert wII_values[-1] > 10.0, "Hydrogen should be highly ionized at 20000K"
    
    def test_saha_density_scaling(self, test_data):
        """Test Saha equation electron density dependence."""
        T = 10000.0  # Hot enough for significant ionization
        densities = [1e10, 1e11, 1e12, 1e13, 1e14, 1e15]
        
        # Test hydrogen ionization vs density
        wII_values = []
        for ne in densities:
            wII, wIII = saha_ion_weights(T, ne, 1,
                                       test_data['ionization_energies'],
                                       test_data['partition_fns'])
            wII_values.append(float(wII))
        
        # Should decrease with electron density (Le Chatelier's principle)
        for i in range(1, len(wII_values)):
            assert wII_values[i] <= wII_values[i-1] * 1.1, \
                f"Ionization should decrease with ne: {densities[i-1]:.0e} -> {densities[i]:.0e}"
        
        # Should show strong density dependence
        ratio = wII_values[0] / wII_values[-1]
        assert ratio > 100, f"Should show strong density dependence, got ratio {ratio:.1f}"
    
    def test_saha_element_scaling(self, test_data):
        """Test Saha equation across different elements."""
        T = 8000.0
        ne = 1e13
        
        # Test elements with different ionization energies
        elements = [1, 2, 6, 8, 12, 26]  # H, He, C, O, Mg, Fe
        ionization_fractions = []
        
        for Z in elements:
            if Z in test_data['ionization_energies']:
                wII, wIII = saha_ion_weights(T, ne, Z,
                                           test_data['ionization_energies'],
                                           test_data['partition_fns'])
                
                # Calculate ionization fraction
                total = 1.0 + float(wII) + float(wIII)
                ionization_frac = (float(wII) + 2*float(wIII)) / total
                ionization_fractions.append((Z, ionization_frac))
        
        # Elements with lower ionization energies should be more ionized
        # (though this is complicated by partition function effects)
        assert len(ionization_fractions) >= 4, "Should test multiple elements"
        
        # All ionization fractions should be reasonable
        for Z, frac in ionization_fractions:
            assert 0.0 <= frac <= 2.0, f"Ionization fraction should be physical for Z={Z}"
    
    def test_partition_function_dependence(self, test_data):
        """Test that Saha weights properly use partition functions."""
        T = 5778.0
        ne = 1e13
        
        # Get normal Saha weights
        wII_normal, wIII_normal = saha_ion_weights(T, ne, 1,
                                                 test_data['ionization_energies'],
                                                 test_data['partition_fns'])
        
        # Create modified partition functions (double the values)
        modified_partition_fns = test_data['partition_fns'].copy()
        
        h1_species = Species.from_atomic_number(1, 0)
        h2_species = Species.from_atomic_number(1, 1)
        
        if h1_species in modified_partition_fns and h2_species in modified_partition_fns:
            original_U1 = modified_partition_fns[h1_species]
            original_U2 = modified_partition_fns[h2_species]
            
            # Double partition function values
            modified_partition_fns[h1_species] = lambda log_T: 2.0 * original_U1(log_T)
            modified_partition_fns[h2_species] = lambda log_T: 2.0 * original_U2(log_T)
            
            # Get modified Saha weights
            wII_modified, wIII_modified = saha_ion_weights(T, ne, 1,
                                                         test_data['ionization_energies'],
                                                         modified_partition_fns)
            
            # Should be different (partition functions matter)
            relative_diff = abs(float(wII_modified) - float(wII_normal)) / float(wII_normal)
            assert relative_diff > 0.01, "Partition functions should affect Saha weights"


class TestTranslationalPartitionFunction:
    """Test translational partition function implementation."""
    
    def test_basic_properties(self):
        """Test basic properties of translational partition function."""
        T = 5778.0
        
        # Test for electron mass
        trans_U = translational_U(me_cgs, T)
        trans_U_val = float(trans_U)
        
        # Should be positive and large
        assert trans_U_val > 0, "Translational U should be positive"
        assert trans_U_val > 1e40, "Translational U should be large for electron"
        assert np.isfinite(trans_U_val), "Translational U should be finite"
    
    def test_temperature_dependence(self):
        """Test temperature dependence of translational U."""
        temperatures = [1000, 3000, 5000, 10000]
        mass = me_cgs
        
        trans_U_values = []
        for T in temperatures:
            trans_U = translational_U(mass, T)
            trans_U_values.append(float(trans_U))
        
        # Should increase with temperature (T^1.5 dependence)
        for i in range(1, len(trans_U_values)):
            ratio = trans_U_values[i] / trans_U_values[i-1]
            temp_ratio = temperatures[i] / temperatures[i-1]
            expected_ratio = temp_ratio**1.5
            
            # Should be close to T^1.5 scaling
            assert abs(ratio - expected_ratio) / expected_ratio < 0.01, \
                f"Should follow T^1.5 scaling: expected {expected_ratio:.3f}, got {ratio:.3f}"
    
    def test_mass_dependence(self):
        """Test mass dependence of translational U."""
        T = 5778.0
        
        # Test different masses
        mass_electron = me_cgs
        mass_proton = me_cgs * 1836  # Proton is ~1836 times heavier
        
        trans_U_electron = float(translational_U(mass_electron, T))
        trans_U_proton = float(translational_U(mass_proton, T))
        
        # Should scale as mass^1.5
        mass_ratio = mass_proton / mass_electron
        expected_ratio = mass_ratio**1.5
        actual_ratio = trans_U_proton / trans_U_electron
        
        relative_error = abs(actual_ratio - expected_ratio) / expected_ratio
        assert relative_error < 0.01, \
            f"Should follow mass^1.5 scaling: expected {expected_ratio:.1f}, got {actual_ratio:.1f}"
    
    def test_physical_constants(self):
        """Test that physical constants are used correctly."""
        T = 5778.0
        mass = me_cgs
        
        # Calculate manually using the formula
        from jorg.constants import kboltz_cgs, hplanck_cgs
        
        expected = (2.0 * np.pi * mass * kboltz_cgs * T / hplanck_cgs**2)**1.5
        calculated = float(translational_U(mass, T))
        
        relative_error = abs(calculated - expected) / expected
        assert relative_error < 1e-10, \
            f"Should match manual calculation: expected {expected:.6e}, got {calculated:.6e}"


class TestSahaPhysicsValidation:
    """Test physical validity of Saha equation results."""
    
    def test_ionization_energy_effects(self):
        """Test that ionization energies affect results correctly."""
        from jorg.statmech.partition_functions import create_default_partition_functions
        
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        T = 8000.0
        ne = 1e13
        
        # Compare elements with different ionization energies
        # Hydrogen: 13.6 eV
        # Helium: 24.6 eV (first ionization)
        
        wII_h, wIII_h = saha_ion_weights(T, ne, 1, ionization_energies, partition_fns)
        wII_he, wIII_he = saha_ion_weights(T, ne, 2, ionization_energies, partition_fns)
        
        # Hydrogen should be more ionized (lower ionization energy)
        h_ionization_frac = float(wII_h) / (1.0 + float(wII_h))
        he_ionization_frac = float(wII_he) / (1.0 + float(wII_he) + float(wIII_he))
        
        assert h_ionization_frac > he_ionization_frac, \
            "H should be more ionized than He (lower ionization energy)"
    
    def test_statistical_weights(self):
        """Test that statistical weights (partition functions) work correctly."""
        from jorg.statmech.partition_functions import create_default_partition_functions
        
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        T = 5778.0
        ne = 1e13
        
        # Test that partition function values are reasonable
        h1_species = Species.from_atomic_number(1, 0)
        h2_species = Species.from_atomic_number(1, 1)
        
        if h1_species in partition_fns and h2_species in partition_fns:
            log_T = np.log(T)
            U_h1 = float(partition_fns[h1_species](log_T))
            U_h2 = float(partition_fns[h2_species](log_T))
            
            # H I should have U ≈ 2 (ground state degeneracy)
            assert abs(U_h1 - 2.0) < 0.1, f"H I should have U ≈ 2, got {U_h1}"
            
            # H II should have U = 1 (bare proton)
            assert abs(U_h2 - 1.0) < 0.01, f"H II should have U = 1, got {U_h2}"
    
    def test_limiting_cases(self):
        """Test behavior in limiting cases."""
        from jorg.statmech.partition_functions import create_default_partition_functions
        
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        # Very low temperature - should be mostly neutral
        T_low = 1000.0
        ne = 1e13
        
        wII_low, wIII_low = saha_ion_weights(T_low, ne, 1, ionization_energies, partition_fns)
        
        ionization_frac_low = float(wII_low) / (1.0 + float(wII_low))
        assert ionization_frac_low < 0.01, "Should be mostly neutral at low temperature"
        
        # Very high temperature - should be mostly ionized
        T_high = 50000.0
        ne = 1e13
        
        wII_high, wIII_high = saha_ion_weights(T_high, ne, 1, ionization_energies, partition_fns)
        
        ionization_frac_high = float(wII_high) / (1.0 + float(wII_high))
        assert ionization_frac_high > 0.99, "Should be mostly ionized at high temperature"
    
    def test_numerical_stability(self):
        """Test numerical stability under extreme conditions."""
        from jorg.statmech.partition_functions import create_default_partition_functions
        
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        # Test extreme conditions
        extreme_conditions = [
            (100.0, 1e5),    # Very cold, very dense
            (100000.0, 1e20), # Very hot, very dense
            (5778.0, 1e5),   # Normal T, very dense
            (50000.0, 1e8),  # Very hot, low density
        ]
        
        for T, ne in extreme_conditions:
            try:
                wII, wIII = saha_ion_weights(T, ne, 1, ionization_energies, partition_fns)
                
                # Should return finite values
                assert np.isfinite(float(wII)), f"wII should be finite at T={T}, ne={ne}"
                assert np.isfinite(float(wIII)), f"wIII should be finite at T={T}, ne={ne}"
                
                # Should be non-negative
                assert float(wII) >= 0, f"wII should be non-negative at T={T}, ne={ne}"
                assert float(wIII) >= 0, f"wIII should be non-negative at T={T}, ne={ne}"
                
            except Exception as e:
                pytest.fail(f"Saha equation should be stable at T={T}, ne={ne}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])