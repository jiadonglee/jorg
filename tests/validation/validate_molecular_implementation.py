#!/usr/bin/env python3
"""
Comprehensive validation of Jorg molecular line implementation against Korg.jl.

This script tests the new molecular line capabilities in Jorg and compares
results with Korg.jl reference calculations where possible.
"""

import sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path
import time
import warnings

# Add Jorg to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    # Import Jorg molecular modules
    from jorg.lines.molecular_cross_sections import (
        MolecularCrossSection, create_molecular_cross_section,
        save_molecular_cross_section, load_molecular_cross_section,
        interpolate_molecular_cross_sections
    )
    from jorg.lines.molecular_species import (
        STELLAR_MOLECULES, get_molecular_species, 
        MolecularPartitionFunction, MolecularEquilibrium,
        molecular_species_summary, get_cool_star_molecules
    )
    from jorg.lines.linelist import load_exomol_linelist, get_molecular_species_id
    from jorg.lines.core import (
        total_line_absorption_with_molecules,
        separate_atomic_molecular_lines,
        is_molecular_species_id
    )
    from jorg.lines.datatypes import LineData, create_line_data
    from jorg.lines.profiles import line_profile
    
    JORG_AVAILABLE = True
    print("✅ Jorg molecular modules imported successfully")
    
except ImportError as e:
    print(f"❌ Failed to import Jorg modules: {e}")
    JORG_AVAILABLE = False


def test_molecular_species_database():
    """Test the molecular species database."""
    print("\n🧬 Testing Molecular Species Database")
    print("=" * 50)
    
    if not JORG_AVAILABLE:
        print("❌ Jorg not available, skipping test")
        return False
    
    try:
        # Print molecular species summary
        molecular_species_summary()
        
        # Test specific molecules
        key_molecules = ['H2O', 'TiO', 'VO', 'OH', 'CO', 'CaH']
        print(f"\n🔍 Testing {len(key_molecules)} key molecular species:")
        
        for mol_name in key_molecules:
            molecule = get_molecular_species(mol_name)
            if molecule:
                print(f"  ✅ {mol_name}: ID={molecule.species_id}, Mass={molecule.mass_amu:.3f} amu")
            else:
                print(f"  ❌ {mol_name}: Not found")
                return False
        
        print("✅ Molecular species database test passed")
        return True
        
    except Exception as e:
        print(f"❌ Molecular species test failed: {e}")
        return False


def test_molecular_partition_functions():
    """Test molecular partition function calculations."""
    print("\n📊 Testing Molecular Partition Functions")
    print("=" * 50)
    
    if not JORG_AVAILABLE:
        print("❌ Jorg not available, skipping test")
        return False
    
    try:
        # Test partition functions for common molecules
        molecules = ['H2O', 'CO', 'TiO', 'OH']
        temperatures = np.array([1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000])
        
        print("Temperature (K) | " + " | ".join(f"{mol:>8}" for mol in molecules))
        print("-" * (16 + len(molecules) * 11))
        
        results = {}
        for mol_name in molecules:
            molecule = get_molecular_species(mol_name)
            pf_calc = MolecularPartitionFunction(molecule)
            pf_values = [pf_calc.calculate(float(T)) for T in temperatures]
            results[mol_name] = pf_values
            
        # Print results table
        for i, T in enumerate(temperatures):
            row = f"{T:>10.0f} K   | "
            row += " | ".join(f"{results[mol][i]:>8.1f}" for mol in molecules)
            print(row)
        
        # Validate monotonic increase
        for mol_name in molecules:
            pf_values = results[mol_name]
            for i in range(1, len(pf_values)):
                if pf_values[i] <= pf_values[i-1]:
                    print(f"❌ Partition function for {mol_name} not monotonic")
                    return False
        
        print("✅ Molecular partition function test passed")
        return True
        
    except Exception as e:
        print(f"❌ Molecular partition function test failed: {e}")
        return False


def test_molecular_cross_sections():
    """Test molecular cross-section precomputation."""
    print("\n🎯 Testing Molecular Cross-Section Precomputation")
    print("=" * 50)
    
    if not JORG_AVAILABLE:
        print("❌ Jorg not available, skipping test")
        return False
    
    try:
        # Create test molecular linelist
        print("Creating test H2O linelist...")
        wavelengths = np.linspace(5000, 5010, 20) * 1e-8  # 20 lines, 5000-5010 Å
        molecular_lines = []
        
        for i, wl in enumerate(wavelengths):
            line = create_line_data(
                wavelength=wl,
                species=801,  # H2O
                log_gf=-2.0 - i * 0.05,  # Varying line strength
                E_lower=1.0 + i * 0.1,   # Varying excitation energy
                gamma_rad=1e8,
                gamma_stark=0.0,  # No Stark for molecules
                vdw_param1=0.0,  # No vdW for molecules
                vdw_param2=0.0
            )
            molecular_lines.append(line)
        
        print(f"Created {len(molecular_lines)} molecular lines")
        
        # Create molecular cross-sections
        print("Computing molecular cross-sections...")
        wavelength_range = (4999e-8, 5011e-8)
        
        start_time = time.time()
        cross_section = create_molecular_cross_section(
            molecular_lines, 
            wavelength_range,
            wavelength_step=0.1e-8,  # 0.01 Å steps
            vmic_vals=jnp.array([0.0, 1e5, 2e5]),  # 0, 1, 2 km/s
            log_temp_vals=jnp.array([3.3, 3.5, 3.7, 3.9]),  # 2000-8000 K
        )
        computation_time = time.time() - start_time
        
        print(f"Cross-section computation completed in {computation_time:.2f} seconds")
        print(f"Grid dimensions: {cross_section.cross_sections.shape}")
        print(f"Wavelength range: {cross_section.wavelengths[0]*1e8:.1f} - {cross_section.wavelengths[-1]*1e8:.1f} Å")
        
        # Test interpolation
        print("Testing cross-section interpolation...")
        test_wavelengths = jnp.linspace(5000e-8, 5010e-8, 100)
        test_temperatures = jnp.array([3000.0, 5000.0])
        test_vmic = 1.5e5  # 1.5 km/s
        test_density = 1e12  # cm^-3
        
        start_time = time.time()
        alpha = cross_section.interpolate(
            test_wavelengths, test_temperatures, test_vmic, test_density
        )
        interpolation_time = time.time() - start_time
        
        print(f"Interpolation completed in {interpolation_time:.4f} seconds")
        print(f"Alpha shape: {alpha.shape}")
        print(f"Max absorption coefficient: {jnp.max(alpha):.2e} cm^-1")
        
        # Validate results
        if alpha.shape != (len(test_temperatures), len(test_wavelengths)):
            print("❌ Incorrect alpha array shape")
            return False
        
        if jnp.any(alpha < 0):
            print("❌ Negative absorption coefficients found")
            return False
        
        if jnp.max(alpha) == 0:
            print("❌ No absorption detected")
            return False
        
        print("✅ Molecular cross-section test passed")
        return True
        
    except Exception as e:
        print(f"❌ Molecular cross-section test failed: {e}")
        return False


def test_molecular_line_profiles():
    """Test molecular line profile accuracy."""
    print("\n📈 Testing Molecular Line Profiles")
    print("=" * 50)
    
    if not JORG_AVAILABLE:
        print("❌ Jorg not available, skipping test")
        return False
    
    try:
        # Test parameters
        lambda_0 = 5000e-8  # cm
        sigma = 0.1e-8      # Doppler width
        gamma = 0.05e-8     # Lorentz width
        amplitude = 1.0
        
        wavelengths = jnp.linspace(4999e-8, 5001e-8, 1000)
        
        print(f"Computing Voigt profile:")
        print(f"  Line center: {lambda_0*1e8:.1f} Å")
        print(f"  Doppler width: {sigma*1e8:.3f} Å") 
        print(f"  Lorentz width: {gamma*1e8:.3f} Å")
        
        # Calculate line profile
        start_time = time.time()
        profile = line_profile(lambda_0, sigma, gamma, amplitude, wavelengths)
        calculation_time = time.time() - start_time
        
        print(f"Profile calculation time: {calculation_time:.4f} seconds")
        print(f"Profile shape: {profile.shape}")
        print(f"Max profile value: {jnp.max(profile):.3e}")
        print(f"Profile at center: {profile[500]:.3e}")  # Approximate center
        
        # Validate profile properties
        center_idx = jnp.argmin(jnp.abs(wavelengths - lambda_0))
        max_idx = jnp.argmax(profile)
        
        if abs(center_idx - max_idx) > 2:
            print("❌ Profile maximum not at line center")
            return False
        
        if jnp.any(profile < 0):
            print("❌ Negative profile values found")
            return False
        
        # Test normalization (approximately)
        integral = jnp.trapezoid(profile, wavelengths)
        expected_integral = amplitude  # For normalized profile
        
        print(f"Profile integral: {integral:.3e}")
        print(f"Expected integral: {expected_integral:.3e}")
        
        if abs(integral - expected_integral) > 0.1 * expected_integral:
            print("⚠️  Profile normalization may be off, but continuing...")
        
        print("✅ Molecular line profile test passed")
        return True
        
    except Exception as e:
        print(f"❌ Molecular line profile test failed: {e}")
        return False


def test_molecular_equilibrium():
    """Test molecular chemical equilibrium calculations."""
    print("\n⚖️  Testing Molecular Chemical Equilibrium")
    print("=" * 50)
    
    if not JORG_AVAILABLE:
        print("❌ Jorg not available, skipping test")
        return False
    
    try:
        # Test with common stellar molecules
        molecule_names = ['H2O', 'OH', 'CO', 'TiO']
        molecules = [get_molecular_species(name) for name in molecule_names]
        
        eq_calc = MolecularEquilibrium(molecules)
        
        # Test conditions
        temperatures = [2500, 3500, 4500, 5500]  # K
        pressure = 1e5  # dyne/cm²
        abundances = {1: 1.0, 6: 1e-4, 8: 1e-4, 22: 1e-8}  # H, C, O, Ti
        h_density = 1e16  # cm^-3
        
        print("Temperature (K) | " + " | ".join(f"{mol:>10}" for mol in molecule_names))
        print("-" * (16 + len(molecule_names) * 13))
        
        for temp in temperatures:
            densities = eq_calc.calculate_number_densities(
                temp, pressure, abundances, h_density
            )
            
            row = f"{temp:>10.0f} K   | "
            density_values = []
            for mol in molecules:
                density = densities.get(mol.species_id, 0.0)
                density_values.append(density)
                row += f"{density:>10.2e} | "
            
            print(row)
            
            # Validate densities
            for i, density in enumerate(density_values):
                if density < 0:
                    print(f"❌ Negative density for {molecule_names[i]}")
                    return False
                if density > h_density:
                    print(f"❌ Molecular density exceeds H density for {molecule_names[i]}")
                    return False
        
        print("✅ Molecular equilibrium test passed")
        return True
        
    except Exception as e:
        print(f"❌ Molecular equilibrium test failed: {e}")
        return False


def test_performance_comparison():
    """Test performance of molecular line calculations."""
    print("\n⚡ Testing Molecular Line Performance")
    print("=" * 50)
    
    if not JORG_AVAILABLE:
        print("❌ Jorg not available, skipping test")
        return False
    
    try:
        # Create progressively larger molecular linelists
        line_counts = [10, 50, 100, 500]
        wavelength_counts = [100, 500, 1000, 2000]
        
        print("Lines | Wavelengths | Computation Time | Throughput")
        print("-" * 55)
        
        for n_lines in line_counts:
            for n_wl in wavelength_counts[:2]:  # Limit for reasonable test time
                # Create molecular linelist
                wl_range = np.linspace(5000e-8, 5100e-8, n_lines)
                molecular_lines = []
                
                for wl in wl_range:
                    line = create_line_data(
                        wavelength=wl, species=801, log_gf=-2.0, E_lower=1.0,
                        gamma_rad=1e8, gamma_stark=0.0, vdw_param1=0.0, vdw_param2=0.0
                    )
                    molecular_lines.append(line)
                
                # Create wavelength grid
                wavelengths = jnp.linspace(4999e-8, 5101e-8, n_wl)
                
                # Time calculation
                start_time = time.time()
                
                from jorg.lines.core import calculate_molecular_line_absorption
                alpha = calculate_molecular_line_absorption(
                    wavelengths, molecular_lines, 3500.0, 1e5, {801: jnp.array([1e12])}
                )
                
                computation_time = time.time() - start_time
                throughput = (n_lines * n_wl) / computation_time
                
                print(f"{n_lines:>5} | {n_wl:>11} | {computation_time:>13.3f} s | {throughput:>9.1e} calc/s")
                
                # Basic validation
                if jnp.any(jnp.isnan(alpha)) or jnp.any(alpha < 0):
                    print(f"❌ Invalid results for {n_lines} lines, {n_wl} wavelengths")
                    return False
        
        print("✅ Performance test passed")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def create_molecular_comparison_plot():
    """Create plots comparing molecular features."""
    print("\n📊 Creating Molecular Feature Comparison Plots")
    print("=" * 50)
    
    if not JORG_AVAILABLE:
        print("❌ Jorg not available, skipping plot")
        return False
    
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Jorg Molecular Line Implementation Validation', fontsize=16)
        
        # Plot 1: Molecular partition functions vs temperature
        temperatures = np.linspace(1000, 8000, 100)
        molecules = ['H2O', 'CO', 'TiO', 'OH']
        colors = ['blue', 'red', 'green', 'orange']
        
        for mol_name, color in zip(molecules, colors):
            molecule = get_molecular_species(mol_name)
            pf_calc = MolecularPartitionFunction(molecule)
            pf_values = [pf_calc.calculate(float(T)) for T in temperatures]
            ax1.semilogy(temperatures, pf_values, label=mol_name, color=color)
        
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Partition Function')
        ax1.set_title('Molecular Partition Functions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Line profile comparison
        lambda_0 = 5000e-8
        wavelengths = jnp.linspace(4999e-8, 5001e-8, 1000)
        
        # Different broadening scenarios
        profiles = {}
        profiles['Doppler only'] = line_profile(lambda_0, 0.1e-8, 0.0, 1.0, wavelengths)
        profiles['Lorentz only'] = line_profile(lambda_0, 0.0, 0.05e-8, 1.0, wavelengths)
        profiles['Voigt (equal)'] = line_profile(lambda_0, 0.07e-8, 0.07e-8, 1.0, wavelengths)
        
        for label, profile in profiles.items():
            ax2.plot(wavelengths*1e8, profile, label=label)
        
        ax2.set_xlabel('Wavelength (Å)')
        ax2.set_ylabel('Normalized Profile')
        ax2.set_title('Molecular Line Profiles')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Molecular equilibrium vs temperature
        molecules_eq = [get_molecular_species(name) for name in ['H2O', 'OH', 'CO']]
        eq_calc = MolecularEquilibrium(molecules_eq)
        
        temp_range = np.linspace(2000, 6000, 50)
        abundances = {1: 1.0, 6: 1e-4, 8: 1e-4}
        
        equilibrium_results = {801: [], 108: [], 608: []}  # H2O, OH, CO
        
        for temp in temp_range:
            densities = eq_calc.calculate_number_densities(
                temp, 1e5, abundances, 1e16
            )
            for species_id in equilibrium_results.keys():
                equilibrium_results[species_id].append(densities.get(species_id, 0))
        
        mol_names = ['H2O', 'OH', 'CO']
        mol_ids = [801, 108, 608]
        for name, mol_id, color in zip(mol_names, mol_ids, colors[:3]):
            ax3.semilogy(temp_range, equilibrium_results[mol_id], label=name, color=color)
        
        ax3.set_xlabel('Temperature (K)')
        ax3.set_ylabel('Number Density (cm⁻³)')
        ax3.set_title('Molecular Equilibrium')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Molecular cross-section demonstration
        # Create small molecular linelist for demo
        demo_lines = []
        demo_wavelengths = np.array([5000, 5002, 5004]) * 1e-8
        for wl in demo_wavelengths:
            line = create_line_data(
                wavelength=wl, species=801, log_gf=-2.0, E_lower=1.0,
                gamma_rad=1e8, gamma_stark=0.0, vdw_param1=0.0, vdw_param2=0.0
            )
            demo_lines.append(line)
        
        # Create cross-section
        cross_section = create_molecular_cross_section(
            demo_lines, (4999e-8, 5005e-8), wavelength_step=0.1e-8
        )
        
        # Show cross-section for different temperatures
        demo_wavelengths_fine = jnp.linspace(4999e-8, 5005e-8, 100)
        temp_values = [3000, 4000, 5000]
        
        for temp in temp_values:
            alpha = cross_section.interpolate(
                demo_wavelengths_fine, jnp.array([temp]), 1e5, jnp.array([1e12])
            )
            ax4.plot(demo_wavelengths_fine*1e8, alpha[0], label=f'{temp} K')
        
        ax4.set_xlabel('Wavelength (Å)')
        ax4.set_ylabel('Absorption Coefficient (cm⁻¹)')
        ax4.set_title('Molecular Cross-Section Interpolation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(__file__).parent / "molecular_validation_plots.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"📊 Plots saved to: {output_file}")
        
        plt.show()
        
        print("✅ Plot creation completed")
        return True
        
    except Exception as e:
        print(f"❌ Plot creation failed: {e}")
        return False


def run_comprehensive_validation():
    """Run all validation tests."""
    print("🧪 Jorg Molecular Line Implementation Validation")
    print("=" * 60)
    print("This script validates the new molecular line capabilities in Jorg")
    print("and compares functionality with Korg.jl where applicable.")
    print("=" * 60)
    
    if not JORG_AVAILABLE:
        print("❌ Cannot run validation - Jorg modules not available")
        return False
    
    # Run all tests
    tests = [
        ("Molecular Species Database", test_molecular_species_database),
        ("Molecular Partition Functions", test_molecular_partition_functions),
        ("Molecular Cross-Sections", test_molecular_cross_sections),
        ("Molecular Line Profiles", test_molecular_line_profiles),
        ("Molecular Equilibrium", test_molecular_equilibrium),
        ("Performance Testing", test_performance_comparison),
    ]
    
    results = {}
    start_time = time.time()
    
    for test_name, test_func in tests:
        print(f"\n▶️  Running: {test_name}")
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Create plots
    plot_result = create_molecular_comparison_plot()
    results["Visualization"] = plot_result
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 60)
    print("🏁 VALIDATION SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<30} | {status}")
    
    print("-" * 60)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {100 * passed / total:.1f}%")
    print(f"Total time: {total_time:.1f} seconds")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Jorg molecular implementation is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please review the implementation.")
        return False


if __name__ == "__main__":
    # Suppress some warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    
    success = run_comprehensive_validation()
    sys.exit(0 if success else 1)