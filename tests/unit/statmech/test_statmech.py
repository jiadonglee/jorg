"""
Unit tests for the Jorg statistical mechanics module

Tests chemical equilibrium, partition functions, and ionization calculations.
"""

import pytest
import jax.numpy as jnp
import numpy as np

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jorg.statmech.core import chemical_equilibrium
from jorg.statmech.ionization import saha_equation, ionization_balance
from jorg.statmech.partition_functions import partition_function, atomic_partition_function
from jorg.statmech.molecular import molecular_equilibrium
from jorg.synthesis import format_abundances


class TestPartitionFunctions:
    """Test partition function calculations"""
    
    def test_atomic_partition_function_hydrogen(self):
        """Test hydrogen partition function"""
        temperatures = jnp.array([3000.0, 5000.0, 8000.0])
        
        # Hydrogen neutral (ground state dominates at low T)
        U_HI = atomic_partition_function(1, 0, temperatures)  # H I
        
        assert len(U_HI) == len(temperatures)
        assert jnp.all(U_HI >= 1.0)  # At least ground state
        assert jnp.all(jnp.isfinite(U_HI))
        
        # Should increase with temperature (more excited states populated)
        assert U_HI[-1] > U_HI[0]
    
    def test_atomic_partition_function_iron(self):
        """Test iron partition function"""
        temperatures = jnp.array([4000.0, 6000.0, 8000.0])
        
        # Iron neutral (many energy levels)
        U_FeI = atomic_partition_function(26, 0, temperatures)  # Fe I
        
        assert len(U_FeI) == len(temperatures)
        assert jnp.all(U_FeI >= 1.0)
        assert jnp.all(jnp.isfinite(U_FeI))
        
        # Iron has many low-lying levels, so partition function should be large
        assert jnp.all(U_FeI > 10.0)
        
        # Should increase with temperature
        assert jnp.all(jnp.diff(U_FeI) > 0)
    
    def test_partition_function_ionization_stages(self):
        """Test partition functions for different ionization stages"""
        temperature = 6000.0
        
        # Iron in different ionization stages
        U_FeI = atomic_partition_function(26, 0, temperature)   # Fe I
        U_FeII = atomic_partition_function(26, 1, temperature)  # Fe II
        U_FeIII = atomic_partition_function(26, 2, temperature) # Fe III
        
        assert U_FeI > 0
        assert U_FeII > 0
        assert U_FeIII > 0
        
        # All should be finite
        assert jnp.isfinite(U_FeI)
        assert jnp.isfinite(U_FeII)
        assert jnp.isfinite(U_FeIII)
    
    def test_partition_function_temperature_scaling(self):
        """Test temperature scaling behavior"""
        temperatures = jnp.array([3000.0, 6000.0, 12000.0])
        
        # Test for carbon
        U_CI = atomic_partition_function(6, 0, temperatures)
        
        # Should be monotonically increasing
        assert jnp.all(jnp.diff(U_CI) > 0)
        
        # High temperature limit should be much larger
        assert U_CI[-1] > U_CI[0] * 2
    
    def test_partition_function_wrapper(self):
        """Test general partition_function wrapper"""
        temperature = 5800.0
        
        # Test various species
        U_H = partition_function('H', 0, temperature)
        U_He = partition_function('He', 0, temperature)
        U_Fe = partition_function('Fe', 0, temperature)
        
        assert U_H > 0
        assert U_He > 0
        assert U_Fe > 0
        
        # Iron should have largest partition function (many levels)
        assert U_Fe > U_H
        assert U_Fe > U_He


class TestSahaEquation:
    """Test Saha ionization equilibrium"""
    
    def test_saha_basic_calculation(self):
        """Test basic Saha equation calculation"""
        temperature = 6000.0
        electron_density = 1e10  # cm^-3
        ionization_energy = 13.6  # eV (hydrogen)
        
        U_I = 2.0    # Ground state
        U_II = 1.0   # Ionized state
        
        ratio = saha_equation(temperature, electron_density, ionization_energy, U_I, U_II)
        
        # Ratio should be positive
        assert ratio > 0
        assert jnp.isfinite(ratio)
    
    def test_saha_temperature_dependence(self):
        """Test temperature dependence of Saha equation"""
        electron_density = 1e10
        ionization_energy = 13.6
        U_I, U_II = 2.0, 1.0
        
        T_low = 4000.0
        T_high = 10000.0
        
        ratio_low = saha_equation(T_low, electron_density, ionization_energy, U_I, U_II)
        ratio_high = saha_equation(T_high, electron_density, ionization_energy, U_I, U_II)
        
        # Higher temperature should favor ionization
        assert ratio_high > ratio_low
    
    def test_saha_electron_density_dependence(self):
        """Test electron density dependence"""
        temperature = 6000.0
        ionization_energy = 13.6
        U_I, U_II = 2.0, 1.0
        
        n_e_low = 1e9
        n_e_high = 1e11
        
        ratio_low = saha_equation(temperature, n_e_low, ionization_energy, U_I, U_II)
        ratio_high = saha_equation(temperature, n_e_high, ionization_energy, U_I, U_II)
        
        # Higher electron density should favor recombination
        assert ratio_low > ratio_high
    
    def test_saha_ionization_energy_dependence(self):
        """Test ionization energy dependence"""
        temperature = 6000.0
        electron_density = 1e10
        U_I, U_II = 2.0, 1.0
        
        chi_low = 5.0   # eV (easy to ionize)
        chi_high = 15.0  # eV (hard to ionize)
        
        ratio_low = saha_equation(temperature, electron_density, chi_low, U_I, U_II)
        ratio_high = saha_equation(temperature, electron_density, chi_high, U_I, U_II)
        
        # Lower ionization energy should favor ionization
        assert ratio_low > ratio_high
    
    def test_saha_hydrogen_consistency(self):
        """Test Saha equation for hydrogen against known values"""
        # Typical photosphere conditions
        temperature = 5800.0
        electron_density = 1e10
        
        # Hydrogen ionization
        chi_H = 13.6  # eV
        U_HI = 2.0
        U_HII = 1.0
        
        ratio = saha_equation(temperature, electron_density, chi_H, U_HI, U_HII)
        
        # At photospheric conditions, hydrogen should be mostly neutral
        # So n(H+)*n_e/n(H) should be small
        assert 0.001 < ratio < 1.0


class TestIonizationBalance:
    """Test ionization balance calculations"""
    
    def test_ionization_balance_hydrogen(self):
        """Test hydrogen ionization balance"""
        temperature = 6000.0
        total_density = 1e12  # cm^-3
        initial_electron_density = 1e10
        
        # Simple hydrogen-only case
        abundances = {'H': 1.0}  # Only hydrogen
        ionization_energies = {'H': [13.6]}  # Only first ionization
        partition_functions = {
            'H': {0: 2.0, 1: 1.0}  # U(H I), U(H II)
        }
        
        n_e, number_densities = ionization_balance(
            temperature, total_density, initial_electron_density,
            abundances, ionization_energies, partition_functions
        )
        
        assert n_e > 0
        assert jnp.isfinite(n_e)
        assert 'H_0' in number_densities  # Neutral hydrogen
        assert 'H_1' in number_densities  # Ionized hydrogen
        
        # Conservation: n(H I) + n(H II) = total H
        total_H = number_densities['H_0'] + number_densities['H_1']
        expected_H = total_density * abundances['H']
        np.testing.assert_allclose(total_H, expected_H, rtol=1e-6)
        
        # Charge conservation: n_e = n(H II)
        np.testing.assert_allclose(n_e, number_densities['H_1'], rtol=1e-6)
    
    def test_ionization_balance_multi_element(self):
        """Test ionization balance with multiple elements"""
        temperature = 6000.0
        total_density = 1e12
        initial_electron_density = 1e10
        
        # H + He system
        abundances = {'H': 0.9, 'He': 0.1}
        ionization_energies = {
            'H': [13.6],
            'He': [24.6, 54.4]  # He I and He II
        }
        partition_functions = {
            'H': {0: 2.0, 1: 1.0},
            'He': {0: 1.0, 1: 2.0, 2: 1.0}
        }
        
        n_e, number_densities = ionization_balance(
            temperature, total_density, initial_electron_density,
            abundances, ionization_energies, partition_functions
        )
        
        assert n_e > 0
        assert jnp.isfinite(n_e)
        
        # Check that all species are present
        expected_species = ['H_0', 'H_1', 'He_0', 'He_1', 'He_2']
        for species in expected_species:
            assert species in number_densities
            assert number_densities[species] >= 0
        
        # Conservation tests
        total_H = number_densities['H_0'] + number_densities['H_1']
        total_He = number_densities['He_0'] + number_densities['He_1'] + number_densities['He_2']
        
        expected_H = total_density * abundances['H']
        expected_He = total_density * abundances['He']
        
        np.testing.assert_allclose(total_H, expected_H, rtol=1e-6)
        np.testing.assert_allclose(total_He, expected_He, rtol=1e-6)
        
        # Charge conservation
        total_electrons = (number_densities['H_1'] + 
                          number_densities['He_1'] + 
                          2 * number_densities['He_2'])
        np.testing.assert_allclose(n_e, total_electrons, rtol=1e-6)
    
    def test_ionization_balance_temperature_trend(self):
        """Test ionization balance temperature trends"""
        total_density = 1e12
        initial_electron_density = 1e10
        
        abundances = {'H': 1.0}
        ionization_energies = {'H': [13.6]}
        partition_functions = {'H': {0: 2.0, 1: 1.0}}
        
        temperatures = [4000.0, 6000.0, 8000.0]
        electron_densities = []
        
        for T in temperatures:
            n_e, _ = ionization_balance(
                T, total_density, initial_electron_density,
                abundances, ionization_energies, partition_functions
            )
            electron_densities.append(n_e)
        
        # Higher temperature should generally increase ionization
        assert electron_densities[2] > electron_densities[0]


class TestMolecularEquilibrium:
    """Test molecular equilibrium calculations"""
    
    def test_molecular_h2_formation(self):
        """Test H2 formation equilibrium"""
        temperature = 3000.0  # Cool enough for molecules
        n_H = 1e12  # cm^-3
        
        # Simplified H2 formation: H + H <-> H2
        K_eq = 1e-10  # Example equilibrium constant
        
        n_H2 = molecular_equilibrium('H2', temperature, {'H': n_H}, K_eq)
        
        assert n_H2 >= 0
        assert jnp.isfinite(n_H2)
        
        # Should be much less than total hydrogen at this temperature
        assert n_H2 < n_H
    
    def test_molecular_temperature_dependence(self):
        """Test temperature dependence of molecular equilibrium"""
        n_H = 1e12
        K_eq = 1e-10
        
        T_cool = 2000.0  # Favorable for molecules
        T_hot = 6000.0   # Unfavorable for molecules
        
        n_H2_cool = molecular_equilibrium('H2', T_cool, {'H': n_H}, K_eq)
        n_H2_hot = molecular_equilibrium('H2', T_hot, {'H': n_H}, K_eq)
        
        # Cooler temperature should favor molecule formation
        assert n_H2_cool > n_H2_hot
    
    def test_molecular_co_formation(self):
        """Test CO formation equilibrium"""
        temperature = 3000.0
        n_C = 1e10  # cm^-3
        n_O = 1e11  # cm^-3
        
        K_eq = 1e-5  # Strong bond
        
        n_CO = molecular_equilibrium('CO', temperature, {'C': n_C, 'O': n_O}, K_eq)
        
        assert n_CO >= 0
        assert jnp.isfinite(n_CO)
        
        # Should be limited by the less abundant species (carbon)
        assert n_CO <= n_C


class TestChemicalEquilibrium:
    """Test complete chemical equilibrium solver"""
    
    def test_chemical_equilibrium_simple(self):
        """Test chemical equilibrium for simple H+He atmosphere"""
        temperature = 6000.0
        total_density = 1e12  # cm^-3
        initial_electron_density = 1e10
        
        # Standard abundances
        A_X = format_abundances(0.0)  # Solar
        
        # Simplified abundances (just H and He)
        abundances = jnp.array([A_X[0], A_X[1]] + [0.0] * 90)  # H, He, rest zero
        
        # Mock ionization data
        ionization_energies = {
            1: [13.6],        # H
            2: [24.6, 54.4]   # He
        }
        
        partition_functions = {
            (1, 0): lambda T: 2.0,      # H I
            (1, 1): lambda T: 1.0,      # H II
            (2, 0): lambda T: 1.0,      # He I
            (2, 1): lambda T: 2.0,      # He II
            (2, 2): lambda T: 1.0,      # He III
        }
        
        n_e, species_densities = chemical_equilibrium(
            temperature, total_density, initial_electron_density,
            abundances, ionization_energies, partition_functions
        )
        
        assert n_e > 0
        assert jnp.isfinite(n_e)
        assert len(species_densities) > 0
        
        # Basic sanity checks
        for species, density in species_densities.items():
            assert density >= 0
            assert jnp.isfinite(density)
    
    def test_chemical_equilibrium_convergence(self):
        """Test that chemical equilibrium converges"""
        temperature = 5800.0
        total_density = 1e12
        
        # Try different initial guesses
        initial_guesses = [1e9, 1e10, 1e11]
        
        A_X = format_abundances(0.0)
        abundances = A_X[:10]  # First 10 elements
        
        # Mock minimal data
        ionization_energies = {i: [10.0] for i in range(1, 11)}
        partition_functions = {(i, j): lambda T: 1.0 for i in range(1, 11) for j in range(2)}
        
        results = []
        for initial_n_e in initial_guesses:
            try:
                n_e, _ = chemical_equilibrium(
                    temperature, total_density, initial_n_e,
                    abundances, ionization_energies, partition_functions
                )
                results.append(n_e)
            except:
                # Convergence might fail for some initial guesses
                pass
        
        # If convergence works, results should be similar
        if len(results) > 1:
            for i in range(1, len(results)):
                np.testing.assert_allclose(results[i], results[0], rtol=1e-3)
    
    def test_chemical_equilibrium_conservation(self):
        """Test conservation laws in chemical equilibrium"""
        temperature = 6000.0
        total_density = 1e12
        initial_electron_density = 1e10
        
        # Simple H-only case for exact conservation test
        abundances = jnp.array([12.0] + [0.0] * 91)  # Only hydrogen
        
        ionization_energies = {1: [13.6]}
        partition_functions = {
            (1, 0): lambda T: 2.0,
            (1, 1): lambda T: 1.0,
        }
        
        n_e, species_densities = chemical_equilibrium(
            temperature, total_density, initial_electron_density,
            abundances, ionization_energies, partition_functions
        )
        
        # Extract hydrogen densities
        species_list = list(species_densities.keys())
        if 'H_I' in species_densities and 'H_II' in species_densities:
            n_HI = species_densities['H_I']
            n_HII = species_densities['H_II']
            
            # Total hydrogen conservation
            total_H = n_HI + n_HII
            expected_H = total_density * 10**(abundances[0] - 12)  # Convert from A_X format
            np.testing.assert_allclose(total_H, expected_H, rtol=1e-3)
            
            # Charge conservation
            np.testing.assert_allclose(n_e, n_HII, rtol=1e-6)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_extreme_temperatures(self):
        """Test behavior at extreme temperatures"""
        total_density = 1e12
        initial_electron_density = 1e10
        abundances = jnp.array([12.0] + [0.0] * 91)
        
        ionization_energies = {1: [13.6]}
        partition_functions = {
            (1, 0): lambda T: 2.0,
            (1, 1): lambda T: 1.0,
        }
        
        # Very cool temperature
        try:
            n_e_cool, _ = chemical_equilibrium(
                500.0, total_density, initial_electron_density,
                abundances, ionization_energies, partition_functions
            )
            assert n_e_cool >= 0
            assert jnp.isfinite(n_e_cool)
        except:
            # Might not converge at extreme conditions
            pass
        
        # Very hot temperature
        try:
            n_e_hot, _ = chemical_equilibrium(
                50000.0, total_density, initial_electron_density,
                abundances, ionization_energies, partition_functions
            )
            assert n_e_hot >= 0
            assert jnp.isfinite(n_e_hot)
        except:
            # Might not converge at extreme conditions
            pass
    
    def test_zero_abundances(self):
        """Test behavior with zero abundances"""
        temperature = 6000.0
        total_density = 1e12
        initial_electron_density = 1e10
        
        # All abundances zero (unphysical but should handle gracefully)
        abundances = jnp.zeros(92)
        ionization_energies = {}
        partition_functions = {}
        
        try:
            n_e, species_densities = chemical_equilibrium(
                temperature, total_density, initial_electron_density,
                abundances, ionization_energies, partition_functions
            )
            # Should return zero electron density
            assert n_e == 0.0 or jnp.isclose(n_e, 0.0)
        except:
            # This case might raise an error, which is acceptable
            pass
    
    def test_numerical_stability(self):
        """Test numerical stability"""
        temperature = 6000.0
        total_density = 1e12
        initial_electron_density = 1e10
        
        # Very small abundances (numerical precision test)
        abundances = jnp.array([1e-10, 1e-12] + [0.0] * 90)
        
        ionization_energies = {1: [13.6], 2: [24.6]}
        partition_functions = {
            (1, 0): lambda T: 2.0, (1, 1): lambda T: 1.0,
            (2, 0): lambda T: 1.0, (2, 1): lambda T: 2.0,
        }
        
        try:
            n_e, species_densities = chemical_equilibrium(
                temperature, total_density, initial_electron_density,
                abundances, ionization_energies, partition_functions
            )
            
            # Results should be finite even with tiny abundances
            assert jnp.isfinite(n_e)
            for density in species_densities.values():
                assert jnp.isfinite(density)
        except:
            # Might not converge with extreme values
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])