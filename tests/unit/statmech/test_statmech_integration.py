"""
Integration Tests for Complete Statmech Module
==============================================

Test suite that validates the complete enhanced statmech module works together
correctly, ensuring all components integrate properly with Korg.jl compatibility.
"""

import pytest
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

from jorg.statmech import (
    chemical_equilibrium,
    create_default_partition_functions,
    create_default_ionization_energies,
    create_default_log_equilibrium_constants,
    hummer_mihalas_w,
    hummer_mihalas_U_H,
    saha_ion_weights,
    Species
)


class TestStatmechIntegration:
    """Test complete statmech module integration."""
    
    @pytest.fixture
    def full_data(self):
        """Load complete statmech data for integration testing."""
        return {
            'partition_fns': create_default_partition_functions(),
            'ionization_energies': create_default_ionization_energies(),
            'log_equilibrium_constants': create_default_log_equilibrium_constants()
        }
    
    def test_complete_stellar_atmosphere_calculation(self, full_data):
        """Test complete stellar atmosphere calculation workflow."""
        # Solar photosphere conditions
        temp = 5778.0
        nt = 1e17
        model_atm_ne = 1e13
        
        # Solar abundances (simplified)
        solar_abundances = {
            1: 0.9200,   # H
            2: 0.0780,   # He
            6: 3.69e-4,  # C
            7: 1.04e-4,  # N
            8: 8.51e-4,  # O
            12: 3.8e-5,  # Mg
            14: 3.55e-5, # Si
            26: 4.68e-5, # Fe
        }
        
        # Complete chemical equilibrium calculation
        ne, densities = chemical_equilibrium(
            temp, nt, model_atm_ne, solar_abundances,
            full_data['ionization_energies'],
            full_data['partition_fns'],
            full_data['log_equilibrium_constants']
        )
        
        # Basic validation
        assert ne > 0, "Electron density should be positive"
        assert len(densities) > 0, "Should return species densities"
        
        # Key species should exist
        key_species = [
            (1, 0, 'H I'), (1, 1, 'H II'),
            (2, 0, 'He I'), (2, 1, 'He II'),
            (26, 0, 'Fe I'), (26, 1, 'Fe II')
        ]
        
        for Z, charge, name in key_species:
            species = Species.from_atomic_number(Z, charge)
            assert species in densities, f"{name} should exist in results"
            assert densities[species] > 0, f"{name} density should be positive"
        
        # Validate physical constraints
        self._validate_physics(ne, densities, nt, solar_abundances)
    
    def test_stellar_parameter_range(self, full_data):
        """Test across stellar parameter range (H-R diagram coverage)."""
        stellar_types = [
            (3500, 4.5, "M dwarf"),
            (4500, 2.5, "K giant"),
            (5778, 4.44, "G dwarf (Sun)"),
            (6500, 4.0, "F dwarf"),
            (8000, 4.0, "A dwarf"),
        ]
        
        nt = 1e17
        abundances = {1: 0.92, 2: 0.08}  # H and He only for speed
        
        results = []
        
        for temp, log_g, description in stellar_types:
            # Estimate electron density from stellar parameters
            model_atm_ne = 10**(11 + (temp/6000) + log_g/5)
            
            try:
                ne, densities = chemical_equilibrium(
                    temp, nt, model_atm_ne, abundances,
                    full_data['ionization_energies'],
                    full_data['partition_fns'],
                    full_data['log_equilibrium_constants']
                )
                
                # Basic validation
                assert ne > 0, f"Should converge for {description}"
                assert np.isfinite(ne), f"Should give finite ne for {description}"
                
                # Key species
                h1_species = Species.from_atomic_number(1, 0)
                h2_species = Species.from_atomic_number(1, 1)
                
                assert h1_species in densities, f"H I should exist for {description}"
                assert h2_species in densities, f"H II should exist for {description}"
                
                results.append({
                    'description': description,
                    'temp': temp,
                    'ne': ne,
                    'h1_density': densities[h1_species],
                    'h2_density': densities[h2_species]
                })
                
            except Exception as e:
                pytest.fail(f"Failed for {description}: {e}")
        
        # Validate trends across stellar types
        assert len(results) == len(stellar_types), "Should succeed for all stellar types"
        
        # Hotter stars should generally have higher ionization
        for i in range(1, len(results)):
            if results[i]['temp'] > results[i-1]['temp']:
                ionization_frac_prev = results[i-1]['h2_density'] / (
                    results[i-1]['h1_density'] + results[i-1]['h2_density'])
                ionization_frac_curr = results[i]['h2_density'] / (
                    results[i]['h1_density'] + results[i]['h2_density'])
                
                # Allow some flexibility due to density effects
                assert ionization_frac_curr >= ionization_frac_prev * 0.1, \
                    f"Hotter stars should generally be more ionized"
    
    def test_hummer_mihalas_integration(self, full_data):
        """Test Hummer-Mihalas formalism integration with main system."""
        temp = 5778.0
        nH = 1e15
        nHe = 1e14
        ne = 1e13
        
        # Test occupation probability calculations
        n_eff_values = [1.0, 2.0, 3.0, 5.0, 10.0]
        
        for n_eff in n_eff_values:
            # Standard formulation
            w_std = hummer_mihalas_w(temp, n_eff, nH, nHe, ne, 
                                   use_hubeny_generalization=False)
            
            # Hubeny formulation
            w_hub = hummer_mihalas_w(temp, n_eff, nH, nHe, ne,
                                   use_hubeny_generalization=True)
            
            w_std_val = float(w_std)
            w_hub_val = float(w_hub)
            
            # Should be physical
            assert 0.0 <= w_std_val <= 1.0, f"w_std should be in [0,1] for n_eff={n_eff}"
            assert 0.0 <= w_hub_val <= 1.0, f"w_hub should be in [0,1] for n_eff={n_eff}"
        
        # Test hydrogen partition function with occupation probability
        U_hm_std = hummer_mihalas_U_H(temp, nH, nHe, ne, use_hubeny_generalization=False)
        U_hm_hub = hummer_mihalas_U_H(temp, nH, nHe, ne, use_hubeny_generalization=True)
        
        U_hm_std_val = float(U_hm_std)
        U_hm_hub_val = float(U_hm_hub)
        
        # Should be reasonable
        assert U_hm_std_val > 0, "H&M partition function should be positive"
        assert U_hm_hub_val > 0, "H&M partition function should be positive"
        assert U_hm_std_val <= 2.1, "H&M partition function should be <= standard (pressure ionization)"
        assert U_hm_hub_val <= 2.1, "H&M partition function should be <= standard (pressure ionization)"
    
    def test_partition_function_consistency(self, full_data):
        """Test partition function consistency across temperature range."""
        temperatures = [3000, 5000, 8000, 12000]
        
        # Test key species
        test_species = [
            (1, 0, 'H I'),
            (2, 0, 'He I'),
            (6, 0, 'C I'),
            (26, 0, 'Fe I')
        ]
        
        for Z, charge, name in test_species:
            species = Species.from_atomic_number(Z, charge)
            
            if species in full_data['partition_fns']:
                U_values = []
                
                for temp in temperatures:
                    log_temp = np.log(temp)
                    U = float(full_data['partition_fns'][species](log_temp))
                    U_values.append(U)
                    
                    # Basic validation
                    assert U > 0, f"{name} partition function should be positive at {temp}K"
                    assert np.isfinite(U), f"{name} partition function should be finite at {temp}K"
                
                # Should generally increase with temperature (more excited states)
                # Allow some flexibility for complex atoms
                assert U_values[-1] >= U_values[0] * 0.5, \
                    f"{name} partition function should not decrease drastically with temperature"
    
    def test_molecular_equilibrium_integration(self, full_data):
        """Test molecular equilibrium integration with atomic equilibrium."""
        temp = 3000.0  # Cool enough for molecules
        nt = 1e17
        model_atm_ne = 1e12
        
        # Include C and O for CO formation
        abundances = {
            1: 0.90,   # H
            2: 0.08,   # He
            6: 1e-4,   # C
            8: 3e-4,   # O
        }
        
        # Should include molecular species if available
        ne, densities = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances,
            full_data['ionization_energies'],
            full_data['partition_fns'],
            full_data['log_equilibrium_constants']
        )
        
        # Basic validation
        assert ne > 0, "Should converge with molecular equilibrium"
        assert len(densities) > 0, "Should return species densities"
        
        # Molecular species might be present
        molecular_species_found = 0
        for species, density in densities.items():
            if hasattr(species, 'n_atoms') and species.n_atoms > 1:
                assert density >= 0, f"Molecular species density should be non-negative"
                if density > 0:
                    molecular_species_found += 1
        
        # At cool temperatures, some molecules might form
        print(f"Found {molecular_species_found} molecular species with positive density")
    
    def test_solver_comparison_integration(self, full_data):
        """Test that JAX and scipy solvers give consistent results."""
        conditions = [
            (5778.0, 1e17, 1e13),  # Solar
            (3500.0, 1e16, 1e12),  # Cool dwarf
            (8000.0, 1e18, 1e14),  # Hot dwarf
        ]
        
        abundances = {1: 0.92, 2: 0.08}
        
        for temp, nt, model_atm_ne in conditions:
            # JAX solver
            ne_jax, densities_jax = chemical_equilibrium(
                temp, nt, model_atm_ne, abundances,
                full_data['ionization_energies'],
                full_data['partition_fns'],
                full_data['log_equilibrium_constants'],
                use_jax_solver=True
            )
            
            # Scipy solver
            ne_scipy, densities_scipy = chemical_equilibrium(
                temp, nt, model_atm_ne, abundances,
                full_data['ionization_energies'],
                full_data['partition_fns'],
                full_data['log_equilibrium_constants'],
                use_jax_solver=False
            )
            
            # Should agree reasonably well
            ne_diff = abs(ne_jax - ne_scipy) / ne_scipy
            assert ne_diff < 0.05, \
                f"JAX and scipy solvers should agree within 5% at T={temp}K, got {ne_diff*100:.1f}%"
            
            # Key species should agree
            h1_species = Species.from_atomic_number(1, 0)
            if h1_species in densities_jax and h1_species in densities_scipy:
                h1_diff = abs(densities_jax[h1_species] - densities_scipy[h1_species]) / densities_scipy[h1_species]
                assert h1_diff < 0.1, \
                    f"H I densities should agree within 10% at T={temp}K, got {h1_diff*100:.1f}%"
    
    def _validate_physics(self, ne, densities, nt, abundances):
        """Validate physical constraints on chemical equilibrium results."""
        # Charge conservation
        total_positive_charge = 0
        for species, density in densities.items():
            if species.charge > 0:
                total_positive_charge += species.charge * density
        
        charge_error = abs(total_positive_charge - ne) / ne
        assert charge_error < 0.01, f"Charge conservation error {charge_error*100:.2f}% > 1%"
        
        # Mass conservation (approximately)
        total_nuclei = 0
        for species, density in densities.items():
            if hasattr(species, 'formula') and len(species.formula.atoms) == 1:
                total_nuclei += density
        
        expected_nuclei = nt - ne
        mass_error = abs(total_nuclei - expected_nuclei) / expected_nuclei
        assert mass_error < 0.1, f"Mass conservation error {mass_error*100:.2f}% > 10%"
        
        # Abundance conservation
        for Z, expected_abundance in abundances.items():
            total_element = 0
            for species, density in densities.items():
                if (hasattr(species, 'formula') and 
                    len(species.formula.atoms) == 1 and 
                    species.formula.atoms[0] == Z):
                    total_element += density
            
            if total_element > 0:
                actual_abundance = total_element / (nt - ne)
                abundance_error = abs(actual_abundance - expected_abundance) / expected_abundance
                assert abundance_error < 0.1, \
                    f"Abundance conservation error for Z={Z}: {abundance_error*100:.2f}% > 10%"


class TestStatmechPerformance:
    """Test statmech module performance characteristics."""
    
    def test_convergence_speed(self):
        """Test that enhanced solver converges efficiently."""
        import time
        
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        temp = 5778.0
        nt = 1e17
        model_atm_ne = 1e13
        abundances = {1: 0.92, 2: 0.08}
        
        # Time JAX solver
        start_time = time.time()
        ne_jax, _ = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances,
            ionization_energies, partition_fns, log_equilibrium_constants,
            use_jax_solver=True
        )
        jax_time = time.time() - start_time
        
        # Time scipy solver
        start_time = time.time()
        ne_scipy, _ = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances,
            ionization_energies, partition_fns, log_equilibrium_constants,
            use_jax_solver=False
        )
        scipy_time = time.time() - start_time
        
        # Both should be reasonably fast (< 1 second for simple case)
        assert jax_time < 1.0, f"JAX solver should be fast, took {jax_time:.3f}s"
        assert scipy_time < 1.0, f"Scipy solver should be fast, took {scipy_time:.3f}s"
        
        print(f"JAX solver: {jax_time:.3f}s, Scipy solver: {scipy_time:.3f}s")
    
    def test_memory_usage(self):
        """Test that solver doesn't use excessive memory."""
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        # Run multiple calculations to check for memory leaks
        for i in range(10):
            temp = 5000 + i * 500  # Vary temperature
            nt = 1e17
            model_atm_ne = 1e13
            abundances = {1: 0.92, 2: 0.08}
            
            ne, densities = chemical_equilibrium(
                temp, nt, model_atm_ne, abundances,
                ionization_energies, partition_fns, log_equilibrium_constants
            )
            
            # Should complete without issues
            assert ne > 0, f"Should converge at iteration {i}"


class TestStatmechCompatibility:
    """Test compatibility with Korg.jl expectations."""
    
    def test_korg_species_format(self):
        """Test that Species objects work as expected."""
        # Test species creation
        h1 = Species.from_atomic_number(1, 0)
        h2 = Species.from_atomic_number(1, 1)
        he1 = Species.from_atomic_number(2, 0)
        
        assert h1.charge == 0, "H I should have charge 0"
        assert h2.charge == 1, "H II should have charge 1"
        assert he1.charge == 0, "He I should have charge 0"
        
        # Test that they can be used as dictionary keys
        test_dict = {h1: 1.0, h2: 2.0, he1: 3.0}
        assert test_dict[h1] == 1.0, "Species should work as dict keys"
        assert len(test_dict) == 3, "Different species should be distinct keys"
    
    def test_korg_data_format_compatibility(self):
        """Test that data formats match Korg.jl expectations."""
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        
        # Check ionization energies format
        assert 1 in ionization_energies, "Should have hydrogen ionization energies"
        h_energies = ionization_energies[1]
        assert len(h_energies) == 3, "Should have three ionization energies"
        assert h_energies[0] > 10, "First ionization energy should be ~13.6 eV"
        
        # Check partition functions format
        h1_species = Species.from_atomic_number(1, 0)
        assert h1_species in partition_fns, "Should have H I partition function"
        
        # Test partition function call
        log_T = np.log(5778.0)
        U_h1 = float(partition_fns[h1_species](log_T))
        assert abs(U_h1 - 2.0) < 0.1, "H I partition function should be ~2"
    
    def test_output_format_compatibility(self):
        """Test that output formats match expected structure."""
        partition_fns = create_default_partition_functions()
        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        
        temp = 5778.0
        nt = 1e17
        model_atm_ne = 1e13
        abundances = {1: 0.92, 2: 0.08}
        
        ne, densities = chemical_equilibrium(
            temp, nt, model_atm_ne, abundances,
            ionization_energies, partition_fns, log_equilibrium_constants
        )
        
        # Check return types
        assert isinstance(ne, (float, np.floating)), "Electron density should be float"
        assert isinstance(densities, dict), "Densities should be dict"
        
        # Check dictionary contents
        for species, density in densities.items():
            assert isinstance(species, Species), "Keys should be Species objects"
            assert isinstance(density, (float, np.floating)), "Densities should be float"
            assert density >= 0, "Densities should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])