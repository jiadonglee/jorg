"""
Unit Tests for Enhanced Chemical Equilibrium Solver
==================================================

Test suite for the JAX-enhanced chemical equilibrium solver with automatic
differentiation, ensuring compatibility with Korg.jl and superior performance.
"""

import pytest
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from jorg.statmech.chemical_equilibrium import (
    solve_chemical_equilibrium,
    chemical_equilibrium,
    saha_ion_weights,
    translational_U,
    get_log_nK,
    ChemicalEquilibriumError
)
from jorg.statmech.species import Species
from jorg.statmech import (
    create_default_partition_functions,
    create_default_ionization_energies,
    create_default_log_equilibrium_constants
)


class TestChemicalEquilibriumSolver:
    """Test enhanced chemical equilibrium solver functionality."""
    
    @pytest.fixture
    def test_data(self):
        """Load test data for chemical equilibrium calculations."""
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        return {
            'partition_fns': partition_fns,
            'ionization_energies': ionization_energies,
            'log_equilibrium_constants': log_equilibrium_constants
        }
    
    def test_basic_chemical_equilibrium(self, test_data):
        """Test basic chemical equilibrium calculation."""
        # Solar photosphere conditions
        temp = 5778.0
        nt = 1e17
        model_atm_ne = 1e13
        
        # Simple abundances (H and He only)
        abundances = {1: 0.92, 2: 0.08}
        
        # Test both solvers
        ne_jax, densities_jax = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances,
            test_data['ionization_energies'],
            test_data['partition_fns'],
            test_data['log_equilibrium_constants'],
            use_jax_solver=True
        )
        
        ne_scipy, densities_scipy = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances,
            test_data['ionization_energies'],
            test_data['partition_fns'],
            test_data['log_equilibrium_constants'],
            use_jax_solver=False
        )
        
        # Basic validation
        assert ne_jax > 0, "JAX solver should return positive electron density"
        assert ne_scipy > 0, "Scipy solver should return positive electron density"
        assert len(densities_jax) > 0, "JAX solver should return species densities"
        assert len(densities_scipy) > 0, "Scipy solver should return species densities"
        
        # Solvers should agree reasonably well
        relative_diff = abs(ne_jax - ne_scipy) / ne_scipy
        assert relative_diff < 0.1, f"Solvers should agree within 10%, got {relative_diff*100:.1f}%"
    
    def test_convergence_across_temperatures(self, test_data):
        """Test chemical equilibrium convergence across temperature range."""
        temperatures = [3000, 5000, 7000, 10000]
        nt = 1e17
        model_atm_ne = 1e13
        abundances = {1: 0.92, 2: 0.08}
        
        for temp in temperatures:
            ne, densities = chemical_equilibrium(
                temp, nt, model_atm_ne, abundances,
                test_data['ionization_energies'],
                test_data['partition_fns'],
                test_data['log_equilibrium_constants'],
                use_jax_solver=True
            )
            
            # Should converge at all temperatures
            assert np.isfinite(ne), f"Should converge at T={temp}K"
            assert ne > 0, f"Electron density should be positive at T={temp}K"
            
            # Key species should exist
            h1_species = Species.from_atomic_number(1, 0)
            he1_species = Species.from_atomic_number(2, 0)
            
            assert h1_species in densities, f"H I should exist at T={temp}K"
            assert he1_species in densities, f"He I should exist at T={temp}K"
            assert densities[h1_species] > 0, f"H I density should be positive at T={temp}K"
    
    def test_ionization_physics(self, test_data):
        """Test that ionization physics behaves correctly."""
        nt = 1e17
        model_atm_ne = 1e13
        abundances = {1: 0.92}  # Pure hydrogen
        
        # Cool temperature - mostly neutral
        temp_cool = 3000.0
        ne_cool, densities_cool = chemical_equilibrium(
            temp_cool, nt, model_atm_ne, abundances,
            test_data['ionization_energies'],
            test_data['partition_fns'],
            test_data['log_equilibrium_constants']
        )
        
        # Hot temperature - mostly ionized
        temp_hot = 15000.0
        ne_hot, densities_hot = chemical_equilibrium(
            temp_hot, nt, model_atm_ne, abundances,
            test_data['ionization_energies'],
            test_data['partition_fns'],
            test_data['log_equilibrium_constants']
        )
        
        h1_species = Species.from_atomic_number(1, 0)
        h2_species = Species.from_atomic_number(1, 1)
        
        # At cool temperature, more H I than H II
        if h1_species in densities_cool and h2_species in densities_cool:
            assert densities_cool[h1_species] > densities_cool[h2_species], \
                "Cool temperature should favor neutral hydrogen"
        
        # At hot temperature, more H II than H I
        if h1_species in densities_hot and h2_species in densities_hot:
            assert densities_hot[h2_species] > densities_hot[h1_species], \
                "Hot temperature should favor ionized hydrogen"
        
        # Electron density should increase with temperature
        assert ne_hot > ne_cool, "Electron density should increase with temperature"
    
    def test_abundance_effects(self, test_data):
        """Test chemical equilibrium response to abundance changes."""
        temp = 5778.0
        nt = 1e17
        model_atm_ne = 1e13
        
        # Hydrogen-rich case
        abundances_h_rich = {1: 0.99, 2: 0.01}
        ne_h_rich, densities_h_rich = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances_h_rich,
            test_data['ionization_energies'],
            test_data['partition_fns'],
            test_data['log_equilibrium_constants']
        )
        
        # Helium-rich case
        abundances_he_rich = {1: 0.01, 2: 0.99}
        ne_he_rich, densities_he_rich = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances_he_rich,
            test_data['ionization_energies'],
            test_data['partition_fns'],
            test_data['log_equilibrium_constants']
        )
        
        h1_species = Species.from_atomic_number(1, 0)
        he1_species = Species.from_atomic_number(2, 0)
        
        # Hydrogen-rich case should have more hydrogen
        if h1_species in densities_h_rich and h1_species in densities_he_rich:
            assert densities_h_rich[h1_species] > densities_he_rich[h1_species], \
                "H-rich case should have more hydrogen"
        
        # Helium-rich case should have more helium
        if he1_species in densities_h_rich and he1_species in densities_he_rich:
            assert densities_he_rich[he1_species] > densities_h_rich[he1_species], \
                "He-rich case should have more helium"
    
    def test_solver_robustness(self, test_data):
        """Test solver robustness under challenging conditions."""
        # Very low density
        temp = 5778.0
        nt_low = 1e12
        model_atm_ne = 1e8
        abundances = {1: 0.92, 2: 0.08}
        
        try:
            ne_low, densities_low = chemical_equilibrium(
                temp, nt_low, model_atm_ne, abundances,
                test_data['ionization_energies'],
                test_data['partition_fns'],
                test_data['log_equilibrium_constants']
            )
            assert ne_low > 0, "Should handle low density conditions"
            
        except Exception as e:
            pytest.fail(f"Solver should handle low density gracefully: {e}")
        
        # Very high density
        nt_high = 1e20
        model_atm_ne = 1e16
        
        try:
            ne_high, densities_high = chemical_equilibrium(
                temp, nt_high, model_atm_ne, abundances,
                test_data['ionization_energies'],
                test_data['partition_fns'],
                test_data['log_equilibrium_constants']
            )
            assert ne_high > 0, "Should handle high density conditions"
            
        except Exception as e:
            pytest.fail(f"Solver should handle high density gracefully: {e}")
    
    def test_charge_conservation(self, test_data):
        """Test that charge conservation is maintained."""
        temp = 5778.0
        nt = 1e17
        model_atm_ne = 1e13
        abundances = {1: 0.92, 2: 0.08}
        
        ne, densities = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances,
            test_data['ionization_energies'],
            test_data['partition_fns'],
            test_data['log_equilibrium_constants']
        )
        
        # Calculate total positive charge from ions
        total_positive_charge = 0
        for species, density in densities.items():
            if species.charge > 0:
                total_positive_charge += species.charge * density
        
        # Should approximately equal electron density
        charge_balance_error = abs(total_positive_charge - ne) / ne
        assert charge_balance_error < 0.01, \
            f"Charge conservation error {charge_balance_error*100:.2f}% > 1%"
    
    def test_mass_conservation(self, test_data):
        """Test that mass conservation is maintained."""
        temp = 5778.0
        nt = 1e17
        model_atm_ne = 1e13
        abundances = {1: 0.92, 2: 0.08}
        
        ne, densities = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances,
            test_data['ionization_energies'],
            test_data['partition_fns'],
            test_data['log_equilibrium_constants']
        )
        
        # Calculate total number density of nuclei
        total_nuclei = 0
        for species, density in densities.items():
            if hasattr(species, 'formula') and len(species.formula.atoms) == 1:
                # Atomic species
                total_nuclei += density
        
        # Should approximately equal nt - ne
        expected_nuclei = nt - ne
        mass_error = abs(total_nuclei - expected_nuclei) / expected_nuclei
        assert mass_error < 0.1, \
            f"Mass conservation error {mass_error*100:.2f}% > 10%"


class TestSahaEquationAccuracy:
    """Test Saha equation implementation accuracy."""
    
    def test_saha_weights_basic_properties(self):
        """Test basic properties of Saha weights."""
        from jorg.statmech.partition_functions import create_default_partition_functions
        
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        T = 5778.0
        ne = 1e13
        
        # Test hydrogen
        wII, wIII = saha_ion_weights(T, ne, 1, ionization_energies, partition_fns)
        
        # Weights should be positive
        assert float(wII) >= 0, "Saha weight wII should be non-negative"
        assert float(wIII) >= 0, "Saha weight wIII should be non-negative"
        
        # For hydrogen, wIII should be 0 (no He III)
        assert float(wIII) == 0, "Hydrogen should have wIII = 0"
        
        # Test helium
        wII_he, wIII_he = saha_ion_weights(T, ne, 2, ionization_energies, partition_fns)
        
        assert float(wII_he) >= 0, "He Saha weight wII should be non-negative"
        assert float(wIII_he) >= 0, "He Saha weight wIII should be non-negative"
    
    def test_saha_temperature_dependence(self):
        """Test Saha equation temperature dependence."""
        from jorg.statmech.partition_functions import create_default_partition_functions
        
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        ne = 1e13
        temperatures = [3000, 5000, 8000, 12000]
        
        # Test hydrogen ionization vs temperature
        wII_values = []
        for T in temperatures:
            wII, wIII = saha_ion_weights(T, ne, 1, ionization_energies, partition_fns)
            wII_values.append(float(wII))
        
        # Higher temperature should generally increase ionization
        # (though this can be complicated by partition function effects)
        assert wII_values[-1] > wII_values[0], \
            "Higher temperature should increase ionization fraction"
    
    def test_saha_density_dependence(self):
        """Test Saha equation electron density dependence."""
        from jorg.statmech.partition_functions import create_default_partition_functions
        
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        T = 8000.0  # Hot enough for significant ionization
        densities = [1e11, 1e12, 1e13, 1e14]
        
        # Test hydrogen ionization vs density
        wII_values = []
        for ne in densities:
            wII, wIII = saha_ion_weights(T, ne, 1, ionization_energies, partition_fns)
            wII_values.append(float(wII))
        
        # Higher electron density should decrease ionization weights
        # (Le Chatelier's principle)
        assert wII_values[-1] < wII_values[0], \
            "Higher electron density should decrease ionization weights"
    
    def test_translational_partition_function(self):
        """Test translational partition function calculation."""
        from jorg.constants import me_cgs
        
        T = 5778.0
        trans_U = translational_U(me_cgs, T)
        
        # Should be positive and reasonable
        trans_U_val = float(trans_U)
        assert trans_U_val > 0, "Translational U should be positive"
        assert 1e40 < trans_U_val < 1e50, f"Translational U should be reasonable, got {trans_U_val:.2e}"
        
        # Should increase with temperature
        T_high = 10000.0
        trans_U_high = float(translational_U(me_cgs, T_high))
        assert trans_U_high > trans_U_val, "Translational U should increase with temperature"


class TestMolecularEquilibrium:
    """Test molecular equilibrium functionality."""
    
    def test_molecular_equilibrium_constants(self):
        """Test molecular equilibrium constant calculations."""
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        # Should have loaded molecular data
        assert len(log_equilibrium_constants) > 0, "Should have molecular equilibrium constants"
        
        # Test getting log nK for a molecule
        T = 5778.0
        
        for mol in list(log_equilibrium_constants.keys())[:3]:  # Test first few
            try:
                log_nK = get_log_nK(mol, T, log_equilibrium_constants)
                log_nK_val = float(log_nK)
                
                # Should be finite
                assert np.isfinite(log_nK_val), f"log_nK should be finite for {mol}"
                
                # Should be reasonable (molecular equilibrium constants vary widely)
                assert -50 < log_nK_val < 50, f"log_nK should be reasonable for {mol}, got {log_nK_val}"
                
            except Exception as e:
                pytest.fail(f"Failed to calculate log_nK for {mol}: {e}")


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_data_handling(self):
        """Test handling of missing partition functions or ionization energies."""
        # Create minimal test data
        partition_fns = {Species.from_atomic_number(1, 0): lambda log_T: 2.0}
        ionization_energies = {1: (13.6, 0.0, 0.0)}  # Only hydrogen
        log_equilibrium_constants = {}
        
        temp = 5778.0
        nt = 1e17
        model_atm_ne = 1e13
        abundances = {1: 1.0}  # Pure hydrogen
        
        # Should handle minimal data gracefully
        try:
            ne, densities = chemical_equilibrium(
                temp, nt, model_atm_ne, abundances,
                ionization_energies, partition_fns, log_equilibrium_constants
            )
            assert ne > 0, "Should work with minimal data"
            
        except Exception as e:
            pytest.fail(f"Should handle minimal data gracefully: {e}")
    
    def test_invalid_input_handling(self):
        """Test handling of invalid inputs."""
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        # Test negative temperature
        with pytest.raises((ValueError, ChemicalEquilibriumError, Exception)):
            chemical_equilibrium(
                -1000, 1e17, 1e13, {1: 1.0},
                ionization_energies, partition_fns, log_equilibrium_constants
            )
        
        # Test zero density
        with pytest.raises((ValueError, ChemicalEquilibriumError, Exception)):
            chemical_equilibrium(
                5778, 0, 1e13, {1: 1.0},
                ionization_energies, partition_fns, log_equilibrium_constants
            )
        
        # Test empty abundances
        with pytest.raises((ValueError, ChemicalEquilibriumError, Exception)):
            chemical_equilibrium(
                5778, 1e17, 1e13, {},
                ionization_energies, partition_fns, log_equilibrium_constants
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])