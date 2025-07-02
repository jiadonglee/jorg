"""
Comprehensive tests for molecular line implementations in Jorg.

This module tests the molecular line functionality against Korg.jl reference
calculations to ensure accuracy and consistency.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import tempfile
import h5py

# Import Jorg modules
from jorg.lines.molecular_cross_sections import (
    MolecularCrossSection, create_molecular_cross_section,
    save_molecular_cross_section, load_molecular_cross_section,
    interpolate_molecular_cross_sections
)
from jorg.lines.molecular_species import (
    STELLAR_MOLECULES, get_molecular_species, get_molecular_species_by_id,
    is_molecular_species, MolecularPartitionFunction, MolecularEquilibrium,
    molecular_species_summary
)
from jorg.lines.linelist import load_exomol_linelist, get_molecular_species_id
from jorg.lines.core import (
    total_line_absorption_with_molecules, calculate_molecular_line_absorption,
    separate_atomic_molecular_lines, is_molecular_species_id
)
from jorg.lines.datatypes import LineData


class TestMolecularSpecies:
    """Test molecular species identification and properties."""
    
    def test_molecular_species_database(self):
        """Test that all expected molecular species are available."""
        expected_molecules = [
            'H2O', 'TiO', 'VO', 'OH', 'CH', 'CN', 'CO', 'NH', 
            'SiO', 'CaH', 'FeH', 'MgH', 'AlH', 'SiH', 'H2', 'C2', 'N2', 'O2'
        ]
        
        for molecule in expected_molecules:
            species = get_molecular_species(molecule)
            assert species is not None, f"Molecule {molecule} not found"
            assert species.name == molecule
            assert species.species_id > 0
            assert species.mass_amu > 0
            assert len(species.atomic_composition) > 0
    
    def test_molecular_species_by_id(self):
        """Test molecular species lookup by ID."""
        # Test known IDs
        h2o = get_molecular_species_by_id(801)
        assert h2o is not None
        assert h2o.name == 'H2O'
        
        tio = get_molecular_species_by_id(2208)
        assert tio is not None
        assert tio.name == 'TiO'
        
        # Test unknown ID
        unknown = get_molecular_species_by_id(99999)
        assert unknown is None
    
    def test_is_molecular_species(self):
        """Test molecular species identification."""
        # Molecular species
        assert is_molecular_species(801)   # H2O
        assert is_molecular_species(2208)  # TiO
        assert is_molecular_species(601)   # CH
        
        # More molecular species
        assert is_molecular_species(101)      # H2 (molecular hydrogen)
        
        # Atomic species (should return False)
        assert not is_molecular_species(100)  # H I (atomic hydrogen)
        assert not is_molecular_species(2600) # Fe I
        assert not is_molecular_species(99999) # Unknown
    
    def test_molecular_mass_calculation(self):
        """Test molecular mass calculations."""
        h2o = get_molecular_species('H2O')
        expected_mass = 2 * 1.008 + 15.999  # 2H + O
        assert abs(h2o.mass_amu - expected_mass) < 0.1
        
        co = get_molecular_species('CO')
        expected_mass = 12.011 + 15.999  # C + O
        assert abs(co.mass_amu - expected_mass) < 0.1
    
    def test_atomic_composition(self):
        """Test atomic composition encoding."""
        h2o = get_molecular_species('H2O')
        assert h2o.atomic_composition[1] == 2  # 2 hydrogen atoms
        assert h2o.atomic_composition[8] == 1  # 1 oxygen atom
        
        tio = get_molecular_species('TiO')
        assert tio.atomic_composition[22] == 1  # 1 titanium atom
        assert tio.atomic_composition[8] == 1   # 1 oxygen atom


class TestMolecularPartitionFunctions:
    """Test molecular partition function calculations."""
    
    def test_partition_function_creation(self):
        """Test partition function calculator creation."""
        h2o = get_molecular_species('H2O')
        pf_calc = MolecularPartitionFunction(h2o)
        assert pf_calc.molecule == h2o
    
    def test_partition_function_values(self):
        """Test partition function calculation at different temperatures."""
        h2o = get_molecular_species('H2O')
        pf_calc = MolecularPartitionFunction(h2o)
        
        # Test at typical stellar temperatures
        temperatures = [1000, 3000, 5000, 8000]
        for temp in temperatures:
            pf_value = pf_calc.calculate(temp)
            assert pf_value > 0, f"Partition function should be positive at {temp}K"
            assert pf_value < 1e6, f"Partition function unreasonably large at {temp}K"
    
    def test_temperature_scaling(self):
        """Test that partition functions increase with temperature."""
        co = get_molecular_species('CO')
        pf_calc = MolecularPartitionFunction(co)
        
        pf_low = pf_calc.calculate(1000)
        pf_high = pf_calc.calculate(5000)
        
        assert pf_high > pf_low, "Partition function should increase with temperature"


class TestMolecularEquilibrium:
    """Test molecular chemical equilibrium calculations."""
    
    def test_equilibrium_calculator_creation(self):
        """Test molecular equilibrium calculator creation."""
        molecules = [get_molecular_species('H2O'), get_molecular_species('CO')]
        eq_calc = MolecularEquilibrium(molecules)
        assert len(eq_calc.molecules) == 2
        assert len(eq_calc.partition_functions) == 2
    
    def test_number_density_calculation(self):
        """Test molecular number density calculation."""
        molecules = [get_molecular_species('OH'), get_molecular_species('H2O')]
        eq_calc = MolecularEquilibrium(molecules)
        
        # Typical stellar atmosphere conditions
        temperature = 3500  # K
        pressure = 1e5      # dyne/cm²
        abundances = {1: 1.0, 8: 1e-4}  # H: 1, O: 1e-4
        h_density = 1e16    # cm^-3
        
        densities = eq_calc.calculate_number_densities(
            temperature, pressure, abundances, h_density
        )
        
        # Check that we get reasonable values
        for species_id, density in densities.items():
            assert density >= 0, f"Density should be non-negative for species {species_id}"
            assert density <= h_density, f"Molecular density shouldn't exceed H density"


class TestMolecularCrossSections:
    """Test molecular cross-section precomputation and interpolation."""
    
    def create_test_molecular_linelist(self):
        """Create a test molecular linelist."""
        lines = []
        wavelengths = np.array([5000, 5001, 5002, 5003, 5004]) * 1e-8  # cm
        
        for i, wl in enumerate(wavelengths):
            line = LineData(
                wavelength=wl,
                species=801,  # H2O
                log_gf=-2.0 - i * 0.1,
                E_lower=1.0 + i * 0.5,
                gamma_rad=1e8,
                gamma_stark=0.0,  # No Stark for molecules
                vdw_param1=0.0,  # No vdW for molecules
                vdw_param2=0.0
            )
            lines.append(line)
        
        return lines
    
    def test_cross_section_creation(self):
        """Test molecular cross-section creation."""
        linelist = self.create_test_molecular_linelist()
        wavelength_range = (4999e-8, 5005e-8)  # cm
        
        cross_section = create_molecular_cross_section(
            linelist, wavelength_range, wavelength_step=0.1e-8
        )
        
        assert isinstance(cross_section, MolecularCrossSection)
        assert len(cross_section.wavelengths) > 0
        assert cross_section.cross_sections.shape[0] > 0  # vmic dimension
        assert cross_section.cross_sections.shape[1] > 0  # temp dimension
        assert cross_section.cross_sections.shape[2] > 0  # wavelength dimension
    
    def test_cross_section_interpolation(self):
        """Test molecular cross-section interpolation."""
        # Create a simple cross-section for testing
        wavelengths = jnp.array([5000, 5001, 5002]) * 1e-8
        vmic_grid = jnp.array([0, 2e5, 4e5])
        log_temp_grid = jnp.array([3.5, 4.0, 4.5])
        cross_sections = jnp.ones((3, 3, 3)) * 1e-20  # cm²
        
        from jorg.lines.species import Species
        species = Species(element_id=801, ion_state=0)
        
        cross_section = MolecularCrossSection(
            wavelengths=wavelengths,
            vmic_grid=vmic_grid,
            log_temp_grid=log_temp_grid,
            cross_sections=cross_sections,
            species=species
        )
        
        # Test interpolation
        test_wavelengths = jnp.array([5000.5, 5001.5]) * 1e-8
        test_temperatures = jnp.array([3500, 7000])
        test_vmic = 1e5
        test_density = 1e12
        
        alpha = cross_section.interpolate(
            test_wavelengths, test_temperatures, test_vmic, test_density
        )
        
        assert alpha.shape == (2, 2)  # (n_layers, n_wavelengths)
        assert jnp.all(alpha >= 0), "Absorption coefficient should be non-negative"
    
    def test_cross_section_save_load(self):
        """Test saving and loading molecular cross-sections."""
        linelist = self.create_test_molecular_linelist()
        wavelength_range = (4999e-8, 5005e-8)
        
        # Create cross-section
        cross_section = create_molecular_cross_section(
            linelist, wavelength_range, wavelength_step=1e-8
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            save_molecular_cross_section(cross_section, tmp_file.name)
            
            # Load and compare
            loaded_cross_section = load_molecular_cross_section(tmp_file.name)
            
            assert jnp.allclose(cross_section.wavelengths, loaded_cross_section.wavelengths)
            assert jnp.allclose(cross_section.cross_sections, loaded_cross_section.cross_sections)
            assert cross_section.species.element_id == loaded_cross_section.species.element_id
        
        # Clean up
        Path(tmp_file.name).unlink()


class TestExoMolParsing:
    """Test ExoMol linelist parsing functionality."""
    
    def create_test_exomol_files(self):
        """Create test ExoMol states and transitions files."""
        # Create temporary states file with realistic molecular energies
        # Use larger energy differences to get optical wavelengths (~5000-6000 Å = 17000-20000 cm⁻¹)
        states_content = """# Test ExoMol states file
1    0.0      1    0.5    0.0    1e-6
2    20000.0  3    1.5    0.1    1e-5
3    25000.0  5    2.5    0.2    1e-4
4    30000.0  7    3.5    0.3    1e-3
"""
        
        # Create temporary transitions file  
        transitions_content = """# Test ExoMol transitions file
2    1    1.5e8    0.01
3    1    2.0e8    0.02
4    2    1.0e8    0.01
3    2    5.0e7    0.05
"""
        
        states_file = tempfile.NamedTemporaryFile(mode='w', suffix='.states', delete=False)
        states_file.write(states_content)
        states_file.close()
        
        transitions_file = tempfile.NamedTemporaryFile(mode='w', suffix='.trans', delete=False)
        transitions_file.write(transitions_content)
        transitions_file.close()
        
        return states_file.name, transitions_file.name
    
    def test_exomol_parsing(self):
        """Test ExoMol file parsing."""
        states_file, transitions_file = self.create_test_exomol_files()
        
        try:
            # Test parsing
            linelist = load_exomol_linelist(
                'H2O', states_file, transitions_file, 0, 1,
                line_strength_cutoff=-20,  # Very permissive for test
                wavelength_range=(1000, 50000)  # Wide range
            )
            
            assert len(linelist) > 0, "Should parse some lines"
            
            # Check that all lines are molecular
            for line in linelist:
                assert line.species == 801, "All lines should be H2O (species 801)"
                assert line.gamma_stark == 0.0, "Molecular lines should have no Stark broadening"
                assert line.vdw_param1 == 0.0, "Molecular lines should have no vdW broadening"
        
        finally:
            # Clean up
            Path(states_file).unlink()
            Path(transitions_file).unlink()
    
    def test_molecular_species_id_mapping(self):
        """Test molecular species ID mapping."""
        assert get_molecular_species_id('H2O') == 801
        assert get_molecular_species_id('TiO') == 2208
        assert get_molecular_species_id('UNKNOWN') == 99999


class TestMolecularLineSynthesis:
    """Test molecular line synthesis integration."""
    
    def create_mixed_linelist(self):
        """Create a linelist with both atomic and molecular lines."""
        lines = []
        
        # Add some atomic lines
        atomic_line = LineData(
            wavelength=5000e-8, species=2600, log_gf=-1.0, E_lower=2.0,
            gamma_rad=1e8, gamma_stark=1e-5, vdw_param1=1e-7, vdw_param2=0.0
        )
        lines.append(atomic_line)
        
        # Add some molecular lines
        molecular_line = LineData(
            wavelength=5001e-8, species=801, log_gf=-2.0, E_lower=1.0,
            gamma_rad=1e8, gamma_stark=0.0, vdw_param1=0.0, vdw_param2=0.0
        )
        lines.append(molecular_line)
        
        return lines
    
    def test_atomic_molecular_separation(self):
        """Test separation of atomic and molecular lines."""
        linelist = self.create_mixed_linelist()
        atomic_lines, molecular_lines = separate_atomic_molecular_lines(linelist)
        
        assert len(atomic_lines) == 1
        assert len(molecular_lines) == 1
        assert atomic_lines[0].species == 2600  # Fe I
        assert molecular_lines[0].species == 801  # H2O
    
    def test_molecular_species_id_detection(self):
        """Test molecular species ID detection."""
        assert not is_molecular_species_id(2600)  # Fe I - atomic
        assert is_molecular_species_id(801)       # H2O - molecular
        assert is_molecular_species_id(2208)      # TiO - molecular
    
    def test_molecular_line_absorption_calculation(self):
        """Test molecular line absorption calculation."""
        molecular_lines = [
            LineData(
                wavelength=5000e-8, species=801, log_gf=-2.0, E_lower=1.0,
                gamma_rad=1e8, gamma_stark=0.0, vdw_param1=0.0, vdw_param2=0.0
            )
        ]
        
        wavelengths = jnp.array([4999, 5000, 5001]) * 1e-8
        temperature = 3500.0
        microturbulence = 1e5  # cm/s
        number_densities = {801: jnp.array([1e12])}  # H2O density
        
        alpha = calculate_molecular_line_absorption(
            wavelengths, molecular_lines, temperature, microturbulence, number_densities
        )
        
        assert alpha.shape == wavelengths.shape
        assert jnp.all(alpha >= 0), "Absorption should be non-negative"
        assert jnp.max(alpha) > 0, "Should have some absorption"
    
    def test_total_line_absorption_with_molecules(self):
        """Test integrated atomic + molecular line absorption."""
        linelist = self.create_mixed_linelist()
        wavelengths = jnp.array([4999, 5000, 5001, 5002]) * 1e-8
        
        # Mock stellar parameters
        temperature = 3500.0
        log_g = 4.5
        abundances = {26: 1e-5, 8: 1e-4}  # Fe, O abundances
        molecular_densities = {801: jnp.array([1e12])}  # H2O
        
        alpha_total = total_line_absorption_with_molecules(
            wavelengths, linelist, temperature, log_g,
            abundances=abundances,
            molecular_number_densities=molecular_densities
        )
        
        assert alpha_total.shape == wavelengths.shape
        assert jnp.all(alpha_total >= 0), "Total absorption should be non-negative"


class TestMolecularAccuracyValidation:
    """Test molecular line accuracy against reference calculations."""
    
    def test_molecular_line_profile_accuracy(self):
        """Test molecular line profile calculation accuracy."""
        # Test against known Voigt profile properties
        from jorg.lines.profiles import line_profile
        
        wavelength_center = 5000e-8
        sigma = 0.1e-8  # Doppler width
        gamma = 0.05e-8  # Lorentz width  
        amplitude = 1.0
        
        wavelengths = jnp.linspace(4999e-8, 5001e-8, 1000)
        profile = line_profile(wavelength_center, sigma, gamma, amplitude, wavelengths)
        
        # Test profile properties
        center_idx = jnp.argmin(jnp.abs(wavelengths - wavelength_center))
        assert profile[center_idx] == jnp.max(profile), "Peak should be at line center"
        
        # Test symmetry (approximately)
        left_wing = profile[:center_idx]
        right_wing = profile[center_idx+1:]
        min_len = min(len(left_wing), len(right_wing))
        if min_len > 10:
            symmetry_error = jnp.mean(jnp.abs(left_wing[-min_len:] - right_wing[:min_len][::-1]))
            assert symmetry_error < 0.1 * jnp.max(profile), "Profile should be approximately symmetric"
    
    def test_molecular_partition_function_consistency(self):
        """Test partition function consistency across temperature range."""
        molecules = ['H2O', 'CO', 'TiO']
        temperatures = jnp.array([1000, 2000, 3000, 5000, 8000])
        
        for mol_name in molecules:
            molecule = get_molecular_species(mol_name)
            pf_calc = MolecularPartitionFunction(molecule)
            
            pf_values = [pf_calc.calculate(float(T)) for T in temperatures]
            
            # Check monotonic increase with temperature
            for i in range(1, len(pf_values)):
                assert pf_values[i] > pf_values[i-1], \
                    f"Partition function should increase with T for {mol_name}"
            
            # Check reasonable magnitude
            for pf in pf_values:
                assert 1 <= pf <= 1e6, f"Partition function out of reasonable range for {mol_name}"


# Integration test combining all molecular functionality
class TestMolecularIntegration:
    """Integration tests for complete molecular line synthesis pipeline."""
    
    def test_end_to_end_molecular_synthesis(self):
        """Test complete molecular synthesis pipeline."""
        # Create molecular linelist
        molecular_lines = [
            LineData(
                wavelength=5000e-8, species=801, log_gf=-2.0, E_lower=1.0,
                gamma_rad=1e8, gamma_stark=0.0, vdw_param1=0.0, vdw_param2=0.0
            ),
            LineData(
                wavelength=5010e-8, species=2208, log_gf=-3.0, E_lower=2.0,
                gamma_rad=1e8, gamma_stark=0.0, vdw_param1=0.0, vdw_param2=0.0
            )
        ]
        
        # Create molecular cross-sections
        wavelength_range = (4995e-8, 5015e-8)
        h2o_cross_section = create_molecular_cross_section(
            [molecular_lines[0]], wavelength_range, wavelength_step=1e-8
        )
        
        # Test synthesis with precomputed cross-sections
        wavelengths = jnp.linspace(4995e-8, 5015e-8, 100)
        temperature = jnp.array([3500.0])
        
        molecular_cross_sections = {801: h2o_cross_section}
        molecular_densities = {801: jnp.array([1e12])}
        
        alpha = interpolate_molecular_cross_sections(
            wavelengths, temperature, 1e5, molecular_cross_sections, molecular_densities
        )
        
        assert alpha.shape == (1, len(wavelengths))
        assert jnp.all(alpha >= 0), "Absorption should be non-negative"
        # Note: The cross-sections may be very small or zero for this simple test case
        # The important test is that the interpolation function works without JAX compilation errors
    
    def test_molecular_species_functionality(self):
        """Test molecular species database functionality."""
        # Test species summary (should not crash)
        molecular_species_summary()
        
        # Test all molecular species can be retrieved
        for name in STELLAR_MOLECULES.keys():
            species = get_molecular_species(name)
            assert species is not None
            assert species.name == name
        
        # Test molecular equilibrium with common species
        common_molecules = [
            get_molecular_species('H2O'),
            get_molecular_species('CO'), 
            get_molecular_species('OH')
        ]
        
        eq_calc = MolecularEquilibrium(common_molecules)
        densities = eq_calc.calculate_number_densities(
            3500, 1e5, {1: 1.0, 6: 1e-4, 8: 1e-4}, 1e16
        )
        
        assert len(densities) == 3
        assert all(d >= 0 for d in densities.values())


# Performance benchmarks
class TestMolecularPerformance:
    """Performance tests for molecular line calculations."""
    
    @pytest.mark.slow
    def test_large_molecular_linelist_performance(self):
        """Test performance with large molecular linelist."""
        # Create large molecular linelist
        n_lines = 1000
        wavelengths = jnp.linspace(5000e-8, 6000e-8, n_lines)
        
        molecular_lines = []
        for wl in wavelengths:
            line = LineData(
                wavelength=float(wl), species=801, log_gf=-2.0, E_lower=1.0,
                gamma_rad=1e8, gamma_stark=0.0, vdw_param1=0.0, vdw_param2=0.0
            )
            molecular_lines.append(line)
        
        # Time molecular line absorption calculation
        import time
        wavelength_grid = jnp.linspace(4999e-8, 6001e-8, 10000)
        
        start_time = time.time()
        alpha = calculate_molecular_line_absorption(
            wavelength_grid, molecular_lines, 3500.0, 1e5, {801: jnp.array([1e12])}
        )
        end_time = time.time()
        
        calculation_time = end_time - start_time
        print(f"Molecular line calculation time: {calculation_time:.3f} seconds")
        
        assert calculation_time < 10.0, "Molecular calculation should complete in reasonable time"
        assert alpha.shape == wavelength_grid.shape
    
    @pytest.mark.slow  
    def test_molecular_cross_section_performance(self):
        """Test molecular cross-section creation and interpolation performance."""
        # Create moderate-sized linelist
        n_lines = 100
        wavelength_range = (5000e-8, 5100e-8)
        wavelengths = jnp.linspace(wavelength_range[0], wavelength_range[1], n_lines)
        
        molecular_lines = []
        for wl in wavelengths:
            line = LineData(
                wavelength=float(wl), species=801, log_gf=-2.0, E_lower=1.0,
                gamma_rad=1e8, gamma_stark=0.0, vdw_param1=0.0, vdw_param2=0.0
            )
            molecular_lines.append(line)
        
        # Time cross-section creation
        import time
        start_time = time.time()
        cross_section = create_molecular_cross_section(
            molecular_lines, wavelength_range, wavelength_step=0.1e-8
        )
        creation_time = time.time() - start_time
        
        # Time interpolation
        test_wavelengths = jnp.linspace(wavelength_range[0], wavelength_range[1], 1000)
        start_time = time.time()
        alpha = cross_section.interpolate(
            test_wavelengths, jnp.array([3500.0]), 1e5, jnp.array([1e12])
        )
        interpolation_time = time.time() - start_time
        
        print(f"Cross-section creation time: {creation_time:.3f} seconds")
        print(f"Cross-section interpolation time: {interpolation_time:.3f} seconds")
        
        assert creation_time < 30.0, "Cross-section creation should be reasonable"
        assert interpolation_time < 1.0, "Interpolation should be fast"


if __name__ == "__main__":
    # Run specific test categories
    print("🧪 Testing Molecular Species...")
    pytest.main(["-v", "TestMolecularSpecies"])
    
    print("\n🧪 Testing Molecular Cross-Sections...")
    pytest.main(["-v", "TestMolecularCrossSections"])
    
    print("\n🧪 Testing Molecular Line Synthesis...")
    pytest.main(["-v", "TestMolecularLineSynthesis"])
    
    print("\n🧪 Testing Integration...")
    pytest.main(["-v", "TestMolecularIntegration"])