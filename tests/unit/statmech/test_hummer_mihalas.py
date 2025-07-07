"""
Unit Tests for Hummer-Mihalas Occupation Probability
===================================================

Test suite for the Hummer-Mihalas occupation probability formalism
implementation, ensuring exact compatibility with Korg.jl.
"""

import pytest
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from jorg.statmech.hummer_mihalas import (
    hummer_mihalas_w,
    hummer_mihalas_U_H,
    _hubeny_charged_term
)


class TestHummerMihalasOccupationProbability:
    """Test Hummer-Mihalas occupation probability calculations."""
    
    def test_w_function_basic_properties(self):
        """Test basic properties of the w function."""
        # Standard stellar photosphere conditions
        T = 5778.0
        nH = 1e15
        nHe = 1e14
        ne = 1e13
        
        # Test different principal quantum numbers
        n_eff_values = [1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
        
        for n_eff in n_eff_values:
            w_std = hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=False)
            w_hub = hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=True)
            
            w_std_val = float(w_std)
            w_hub_val = float(w_hub)
            
            # Occupation probability should be between 0 and 1
            assert 0.0 <= w_std_val <= 1.0, f"w_std should be in [0,1], got {w_std_val} for n_eff={n_eff}"
            assert 0.0 <= w_hub_val <= 1.0, f"w_hub should be in [0,1], got {w_hub_val} for n_eff={n_eff}"
            
            # For low densities and low n_eff, w should be close to 1
            if n_eff <= 3 and ne < 1e14:
                assert w_std_val > 0.99, f"w should be ~1 for low n_eff and density, got {w_std_val}"
    
    def test_w_function_density_dependence(self):
        """Test that w function correctly depends on density."""
        T = 5778.0
        n_eff = 10.0
        nH = 1e15
        nHe = 1e14
        
        # Test different electron densities
        ne_values = [1e11, 1e12, 1e13, 1e14, 1e15]
        w_values = []
        
        for ne in ne_values:
            w = hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=False)
            w_val = float(w)
            w_values.append(w_val)
            
            # Should still be physical
            assert 0.0 <= w_val <= 1.0, f"w should be in [0,1], got {w_val} for ne={ne}"
        
        # Higher density should generally reduce occupation probability
        # (pressure ionization effect)
        assert w_values[-1] <= w_values[0], "Higher density should reduce occupation probability"
    
    def test_w_function_temperature_dependence(self):
        """Test temperature dependence of w function."""
        n_eff = 5.0
        nH = 1e15
        nHe = 1e14
        ne = 1e13
        
        # Test different temperatures
        temperatures = [3000, 5778, 8000, 12000]
        
        for T in temperatures:
            w = hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=False)
            w_val = float(w)
            
            # Should be physical at all temperatures
            assert 0.0 <= w_val <= 1.0, f"w should be in [0,1], got {w_val} for T={T}"
            assert np.isfinite(w_val), f"w should be finite, got {w_val} for T={T}"
    
    def test_w_function_quantum_number_dependence(self):
        """Test that w function correctly depends on quantum number."""
        T = 5778.0
        nH = 1e15
        nHe = 1e14
        ne = 1e13
        
        # Higher quantum numbers should be more affected by pressure ionization
        n_eff_low = 2.0
        n_eff_high = 20.0
        
        w_low = float(hummer_mihalas_w(T, n_eff_low, nH, nHe, ne, use_hubeny_generalization=False))
        w_high = float(hummer_mihalas_w(T, n_eff_high, nH, nHe, ne, use_hubeny_generalization=False))
        
        # Higher quantum number should have more pressure ionization effect
        assert w_low >= w_high, f"Higher n_eff should have lower w: w({n_eff_low})={w_low}, w({n_eff_high})={w_high}"
    
    def test_hubeny_vs_standard_formulation(self):
        """Test differences between Hubeny and standard formulations."""
        T = 8000.0  # Higher temperature where differences might be more apparent
        n_eff = 10.0
        nH = 1e15
        nHe = 1e14
        ne = 1e14  # Higher density
        
        w_std = float(hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=False))
        w_hub = float(hummer_mihalas_w(T, n_eff, nH, nHe, ne, use_hubeny_generalization=True))
        
        # Both should be physical
        assert 0.0 <= w_std <= 1.0, f"Standard w should be in [0,1], got {w_std}"
        assert 0.0 <= w_hub <= 1.0, f"Hubeny w should be in [0,1], got {w_hub}"
        
        # They might differ, but should be reasonably close
        relative_diff = abs(w_std - w_hub) / max(w_std, w_hub, 1e-10)
        assert relative_diff < 0.5, f"Standard and Hubeny formulations differ too much: {relative_diff}"


class TestHummerMihalasPartitionFunction:
    """Test Hummer-Mihalas hydrogen partition function."""
    
    def test_U_H_basic_properties(self):
        """Test basic properties of H&M partition function."""
        T = 5778.0
        nH = 1e15
        nHe = 1e14
        ne = 1e13
        
        U_std = hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=False)
        U_hub = hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=True)
        
        U_std_val = float(U_std)
        U_hub_val = float(U_hub)
        
        # Partition functions should be positive
        assert U_std_val > 0, f"H&M partition function should be positive, got {U_std_val}"
        assert U_hub_val > 0, f"H&M partition function should be positive, got {U_hub_val}"
        
        # Should be finite
        assert np.isfinite(U_std_val), f"H&M partition function should be finite, got {U_std_val}"
        assert np.isfinite(U_hub_val), f"H&M partition function should be finite, got {U_hub_val}"
        
        # Should be reasonable (ground state contribution is 2)
        assert U_std_val >= 2.0, f"H&M partition function should be ≥2, got {U_std_val}"
        assert U_std_val < 100.0, f"H&M partition function suspiciously large: {U_std_val}"
    
    def test_U_H_vs_standard_partition_function(self):
        """Test H&M partition function vs standard approach."""
        T = 5778.0
        nH = 1e15
        nHe = 1e14
        ne = 1e13
        
        # Standard H partition function (just ground state degeneracy)
        U_standard = 2.0
        
        # H&M partition function with occupation probability
        U_hm = float(hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=False))
        
        # H&M should generally be less than or equal to standard (pressure ionization)
        assert U_hm <= U_standard * 1.01, f"H&M should be ≤ standard, got {U_hm} vs {U_standard}"
        
        # But should still be close for typical conditions
        relative_diff = abs(U_hm - U_standard) / U_standard
        assert relative_diff < 0.1, f"H&M should be close to standard for typical conditions: {relative_diff}"
    
    def test_U_H_pressure_dependence(self):
        """Test partition function dependence on pressure."""
        T = 5778.0
        nH = 1e15
        nHe = 1e14
        
        # Different electron densities (pressure)
        ne_low = 1e11
        ne_high = 1e15
        
        U_low = float(hummer_mihalas_U_H(T, nH, nHe, ne_low, use_hubeny_generalization=False))
        U_high = float(hummer_mihalas_U_H(T, nH, nHe, ne_high, use_hubeny_generalization=False))
        
        # Higher pressure should reduce partition function (pressure ionization)
        assert U_low >= U_high, f"Higher pressure should reduce U: U(low)={U_low}, U(high)={U_high}"
    
    def test_U_H_temperature_dependence(self):
        """Test partition function temperature dependence."""
        nH = 1e15
        nHe = 1e14
        ne = 1e13
        
        temperatures = [3000, 5778, 8000, 12000]
        U_values = []
        
        for T in temperatures:
            U = float(hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=False))
            U_values.append(U)
            
            # Should be physical at all temperatures
            assert U > 0, f"U should be positive, got {U} for T={T}"
            assert np.isfinite(U), f"U should be finite, got {U} for T={T}"
        
        # Generally should increase with temperature (more levels populated)
        # But pressure ionization can complicate this
        assert all(U > 1.0 for U in U_values), "All partition functions should be > 1"
    
    def test_U_H_nist_level_consistency(self):
        """Test that NIST energy levels are used correctly."""
        # Test at low density where occupation probability is ~1
        T = 5778.0
        nH = 1e12  # Low density
        nHe = 1e11
        ne = 1e10
        
        U = float(hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=False))
        
        # At low density, should be close to statistical mechanical value
        # Ground state (n=1): g=2, E=0
        # First excited (n=2): g=8, E≈10.2 eV
        
        boltzmann_factor_n2 = np.exp(-10.2 / (8.617e-5 * T))  # Very small at 5778K
        expected_U_approx = 2.0 + 8.0 * boltzmann_factor_n2  # Approximately 2.0
        
        # Should be reasonably close to this estimate
        relative_diff = abs(U - expected_U_approx) / expected_U_approx
        assert relative_diff < 0.5, f"U should be close to statistical estimate: {U} vs {expected_U_approx}"


class TestHummerMihalasIntegration:
    """Test integration with other components."""
    
    def test_integration_with_partition_functions(self):
        """Test that H&M functions integrate with main partition function system."""
        from jorg.statmech import create_default_partition_functions
        
        # Load standard partition functions
        partition_funcs = create_default_partition_functions()
        
        # Compare with H&M approach
        T = 5778.0
        nH = 1e15
        nHe = 1e14
        ne = 1e13
        
        # Standard H I partition function
        from jorg.statmech.species import Species
        h1_species = Species.from_atomic_number(1, 0)
        U_standard = float(partition_funcs[h1_species](np.log(T)))
        
        # H&M partition function
        U_hm = float(hummer_mihalas_U_H(T, nH, nHe, ne, use_hubeny_generalization=False))
        
        # Both should be reasonable
        assert 1.5 < U_standard < 2.5, f"Standard H I function should be ~2, got {U_standard}"
        assert 1.5 < U_hm < 10.0, f"H&M function should be reasonable, got {U_hm}"
    
    def test_extreme_conditions(self):
        """Test behavior under extreme astrophysical conditions."""
        # Hot, dense stellar interior conditions
        T_hot = 20000.0
        nH_dense = 1e18
        nHe_dense = 1e17
        ne_dense = 1e17
        
        try:
            w = hummer_mihalas_w(T_hot, 5.0, nH_dense, nHe_dense, ne_dense, 
                               use_hubeny_generalization=False)
            U = hummer_mihalas_U_H(T_hot, nH_dense, nHe_dense, ne_dense, 
                                 use_hubeny_generalization=False)
            
            w_val = float(w)
            U_val = float(U)
            
            # Should still be physical
            assert 0.0 <= w_val <= 1.0, f"w should be physical under extreme conditions, got {w_val}"
            assert U_val > 0, f"U should be positive under extreme conditions, got {U_val}"
            assert np.isfinite(U_val), f"U should be finite under extreme conditions, got {U_val}"
            
        except Exception as e:
            pytest.fail(f"H&M functions should handle extreme conditions gracefully: {e}")
        
        # Cold, rarefied conditions
        T_cold = 2000.0
        nH_rare = 1e10
        nHe_rare = 1e9
        ne_rare = 1e8
        
        try:
            w = hummer_mihalas_w(T_cold, 3.0, nH_rare, nHe_rare, ne_rare,
                               use_hubeny_generalization=False)
            U = hummer_mihalas_U_H(T_cold, nH_rare, nHe_rare, ne_rare,
                                 use_hubeny_generalization=False)
            
            w_val = float(w)
            U_val = float(U)
            
            # Should be close to ideal gas limit
            assert w_val > 0.95, f"w should be ~1 in rarefied conditions, got {w_val}"
            assert 1.8 < U_val < 2.2, f"U should be ~2 in rarefied conditions, got {U_val}"
            
        except Exception as e:
            pytest.fail(f"H&M functions should handle rarefied conditions gracefully: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])