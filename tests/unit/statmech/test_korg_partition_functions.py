"""
Unit Tests for Korg.jl Partition Functions
==========================================

Test suite for the exact Korg.jl partition function implementation,
ensuring perfect compatibility and accuracy.
"""

import pytest
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from jorg.statmech.korg_partition_functions import (
    load_korg_partition_data,
    create_korg_partition_functions,
    validate_against_korg_values
)
from jorg.statmech.species import Species


class TestKorgPartitionFunctions:
    """Test exact Korg.jl partition functions."""
    
    @pytest.fixture
    def partition_funcs(self):
        """Load partition functions for testing."""
        return create_korg_partition_functions()
    
    def test_partition_data_loading(self):
        """Test that Korg.jl partition function data loads correctly."""
        interpolators = load_korg_partition_data()
        
        # Should have loaded many species
        assert len(interpolators) > 200, f"Expected >200 species, got {len(interpolators)}"
        
        # Check that key species are present
        h1_species = Species.from_atomic_number(1, 0)
        he1_species = Species.from_atomic_number(2, 0)
        fe1_species = Species.from_atomic_number(26, 0)
        
        assert h1_species in interpolators, "H I not found in partition data"
        assert he1_species in interpolators, "He I not found in partition data"
        assert fe1_species in interpolators, "Fe I not found in partition data"
    
    def test_hydrogen_partition_functions(self, partition_funcs):
        """Test hydrogen partition function values."""
        solar_logT = np.log(5778.0)
        
        # H I should be exactly 2.0 (ground state degeneracy)
        h1_species = Species.from_atomic_number(1, 0)
        U_h1 = partition_funcs[h1_species](solar_logT)
        assert abs(float(U_h1) - 2.0) < 1e-6, f"H I partition function should be 2.0, got {U_h1}"
        
        # H II should be exactly 1.0 (bare proton)
        h2_species = Species.from_atomic_number(1, 1)
        U_h2 = partition_funcs[h2_species](solar_logT)
        assert abs(float(U_h2) - 1.0) < 1e-6, f"H II partition function should be 1.0, got {U_h2}"
    
    def test_helium_partition_functions(self, partition_funcs):
        """Test helium partition function values."""
        solar_logT = np.log(5778.0)
        
        # He I should be 1.0 (ground state)
        he1_species = Species.from_atomic_number(2, 0)
        U_he1 = partition_funcs[he1_species](solar_logT)
        assert abs(float(U_he1) - 1.0) < 1e-6, f"He I partition function should be 1.0, got {U_he1}"
        
        # He II should be 2.0 (hydrogen-like)
        he2_species = Species.from_atomic_number(2, 1)
        U_he2 = partition_funcs[he2_species](solar_logT)
        assert abs(float(U_he2) - 2.0) < 1e-6, f"He II partition function should be 2.0, got {U_he2}"
    
    def test_iron_partition_functions(self, partition_funcs):
        """Test iron partition function values (complex atom)."""
        solar_logT = np.log(5778.0)
        
        # Fe I should be around 30.784 (from Korg.jl)
        fe1_species = Species.from_atomic_number(26, 0)
        U_fe1 = partition_funcs[fe1_species](solar_logT)
        expected_fe1 = 30.784
        error = abs(float(U_fe1) - expected_fe1) / expected_fe1
        assert error < 0.001, f"Fe I partition function error {error*100:.3f}% > 0.1%"
        
        # Fe II should be around 46.634 (from Korg.jl)
        fe2_species = Species.from_atomic_number(26, 1)
        U_fe2 = partition_funcs[fe2_species](solar_logT)
        expected_fe2 = 46.634
        error = abs(float(U_fe2) - expected_fe2) / expected_fe2
        assert error < 0.001, f"Fe II partition function error {error*100:.3f}% > 0.1%"
    
    def test_temperature_dependence(self, partition_funcs):
        """Test partition function temperature dependence."""
        temperatures = [3000, 5778, 10000]  # Cool, solar, hot
        
        fe1_species = Species.from_atomic_number(26, 0)
        
        U_values = []
        for T in temperatures:
            log_T = np.log(T)
            U = float(partition_funcs[fe1_species](log_T))
            U_values.append(U)
            
            # Partition functions should be positive
            assert U > 0, f"Partition function should be positive, got {U} at T={T}K"
        
        # Fe I partition function should generally increase with temperature
        # (more excited states become accessible)
        assert U_values[2] > U_values[0], "Partition function should increase with temperature"
    
    def test_interpolation_accuracy(self, partition_funcs):
        """Test interpolation accuracy at intermediate temperatures."""
        # Test at intermediate temperatures not in original grid
        test_temperatures = [4500, 6500, 7500]
        
        h1_species = Species.from_atomic_number(1, 0)
        
        for T in test_temperatures:
            log_T = np.log(T)
            U = float(partition_funcs[h1_species](log_T))
            
            # H I should remain close to 2.0 at all reasonable temperatures
            assert 1.9 < U < 2.1, f"H I partition function should be ~2.0, got {U} at T={T}K"
    
    def test_species_coverage(self, partition_funcs):
        """Test that we have good coverage of elements and ionization states."""
        elements_covered = set()
        ionization_states = {}
        
        for species in partition_funcs.keys():
            Z = species.formula.atoms[0]  # Atomic number
            charge = species.charge
            
            elements_covered.add(Z)
            if Z not in ionization_states:
                ionization_states[Z] = set()
            ionization_states[Z].add(charge)
        
        # Should cover many elements
        assert len(elements_covered) >= 80, f"Expected ≥80 elements, got {len(elements_covered)}"
        
        # Should have multiple ionization states for most elements
        multi_ion_elements = sum(1 for Z in ionization_states if len(ionization_states[Z]) >= 2)
        assert multi_ion_elements >= 70, f"Expected ≥70 elements with multiple ionization states"
        
        # Key elements should have at least 2 ionization states
        key_elements = [1, 2, 6, 8, 12, 14, 26, 28]  # H, He, C, O, Mg, Si, Fe, Ni
        for Z in key_elements:
            if Z in ionization_states:
                assert len(ionization_states[Z]) >= 2, f"Element Z={Z} should have ≥2 ionization states"
    
    def test_partition_function_physics(self, partition_funcs):
        """Test basic physics constraints on partition functions."""
        solar_logT = np.log(5778.0)
        
        # Test several elements
        test_elements = [(1, 0), (2, 0), (6, 0), (8, 0), (26, 0)]
        
        for Z, charge in test_elements:
            species = Species.from_atomic_number(Z, charge)
            if species in partition_funcs:
                U = float(partition_funcs[species](solar_logT))
                
                # Basic physics constraints
                assert U >= 1.0, f"Partition function should be ≥1, got {U} for Z={Z}, q={charge}"
                assert U < 1000.0, f"Partition function suspiciously large: {U} for Z={Z}, q={charge}"
                
                # Statistical weight of ground state should be ≤ partition function
                if Z == 1 and charge == 0:  # H I
                    assert abs(U - 2.0) < 0.01, "H I should have U ≈ 2"
                elif Z == 2 and charge == 0:  # He I
                    assert abs(U - 1.0) < 0.01, "He I should have U ≈ 1"


class TestPartitionFunctionFallback:
    """Test partition function fallback system."""
    
    def test_fallback_to_simplified(self):
        """Test fallback to simplified partition functions."""
        from jorg.statmech.partition_functions import create_simplified_partition_functions
        
        simple_funcs = create_simplified_partition_functions()
        
        # Should have created functions for many species
        assert len(simple_funcs) > 200, f"Expected >200 simplified functions, got {len(simple_funcs)}"
        
        # Test basic functionality
        solar_logT = np.log(5778.0)
        h1_species = Species.from_atomic_number(1, 0)
        
        if h1_species in simple_funcs:
            U = float(simple_funcs[h1_species](solar_logT))
            assert 1.5 < U < 2.5, f"Simplified H I function should be reasonable, got {U}"
    
    def test_default_function_creation(self):
        """Test default partition function creation."""
        from jorg.statmech.partition_functions import create_default_partition_functions
        
        # This should prefer exact Korg.jl functions but fall back gracefully
        default_funcs = create_default_partition_functions()
        
        # Should have many functions
        assert len(default_funcs) > 200, f"Expected >200 default functions, got {len(default_funcs)}"
        
        # Test that key species work
        solar_logT = np.log(5778.0)
        key_species = [
            Species.from_atomic_number(1, 0),  # H I
            Species.from_atomic_number(2, 0),  # He I
            Species.from_atomic_number(26, 0), # Fe I
        ]
        
        for species in key_species:
            if species in default_funcs:
                U = float(default_funcs[species](solar_logT))
                assert U > 0, f"Default function should give positive U, got {U} for {species}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])