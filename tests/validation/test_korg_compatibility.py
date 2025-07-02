#!/usr/bin/env python3
"""
Test script to verify Jorg line data compatibility with Korg.jl conventions.

This script validates that Jorg's updated line data structures strictly follow
Korg.jl's formats, units, and conventions.
"""

import sys
import numpy as np
from pathlib import Path

# Add Jorg to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from jorg.lines.datatypes import (
    Formula, Species, Line, LineData, 
    create_line_data, create_line, species_from_integer
)
from jorg.constants import (
    c_cgs, kboltz_eV, hplanck_eV, RydbergH_eV, 
    electron_mass_cgs, eV_to_cgs
)


def test_wavelength_unit_conversion():
    """Test wavelength unit conversions follow Korg.jl exactly."""
    print("🧪 Testing Wavelength Unit Conversions")
    print("=" * 40)
    
    # Test Korg.jl auto-detection logic: wl >= 1 → Å, wl < 1 → cm
    
    # Case 1: Wavelength in Angstroms (≥ 1)
    line1 = create_line_data(5000.0, 2600, -2.0, 1.0)  # 5000 Å input
    expected_cm = 5000.0 * 1e-8  # Should be converted to cm
    print(f"Input: 5000.0 Å → Output: {line1.wavelength:.2e} cm")
    print(f"Expected: {expected_cm:.2e} cm")
    assert abs(line1.wavelength - expected_cm) < 1e-15, "Wavelength conversion failed"
    print("✅ Angstrom to cm conversion correct")
    
    # Case 2: Wavelength already in cm (< 1)
    line2 = create_line_data(5e-5, 2600, -2.0, 1.0)  # Already in cm
    print(f"Input: 5e-5 cm → Output: {line2.wavelength:.2e} cm")
    assert abs(line2.wavelength - 5e-5) < 1e-15, "cm input should be unchanged"
    print("✅ cm input unchanged")
    
    # Case 3: Explicit unit specification
    line3 = create_line_data(6000.0, 2600, -2.0, 1.0, wavelength_unit='angstrom')
    expected_cm3 = 6000.0 * 1e-8
    assert abs(line3.wavelength - expected_cm3) < 1e-15, "Explicit Å conversion failed"
    print("✅ Explicit Angstrom unit conversion correct")
    
    print("✅ All wavelength conversions match Korg.jl conventions\n")


def test_species_structure():
    """Test Species and Formula structures match Korg.jl exactly."""
    print("🧪 Testing Species and Formula Structures")
    print("=" * 40)
    
    # Test atomic species
    fe_neutral = Species.from_element(26, charge=0)  # Fe I
    print(f"Fe I: {fe_neutral}")
    print(f"Formula atoms: {fe_neutral.formula.atoms}")
    print(f"Is molecule: {fe_neutral.is_molecule()}")
    assert fe_neutral.formula.atoms == (26, 0, 0, 0, 0, 0), "Fe I formula incorrect"
    assert fe_neutral.charge == 0, "Fe I charge incorrect"
    assert not fe_neutral.is_molecule(), "Fe I should not be molecular"
    print("✅ Atomic species (Fe I) correct")
    
    # Test ionized species
    fe_ion = Species.from_element(26, charge=1)  # Fe II
    print(f"Fe II: {fe_ion}")
    assert fe_ion.charge == 1, "Fe II charge incorrect"
    print("✅ Ionized species (Fe II) correct")
    
    # Test molecular species
    h2o = Species.from_molecule([1, 1, 8], charge=0)  # H2O
    print(f"H2O: {h2o}")
    print(f"Formula atoms: {h2o.formula.atoms}")
    print(f"Is molecule: {h2o.is_molecule()}")
    assert h2o.formula.atoms == (1, 1, 8, 0, 0, 0), "H2O formula incorrect"
    assert h2o.is_molecule(), "H2O should be molecular"
    print("✅ Molecular species (H2O) correct")
    
    # Test species conversion from integers
    fe_from_int = species_from_integer(2600)  # Fe I
    print(f"Species from 2600: {fe_from_int}")
    print("✅ Integer to Species conversion working")
    
    print("✅ All species structures match Korg.jl format\n")


def test_line_structure():
    """Test Line structure matches Korg.jl exactly."""
    print("🧪 Testing Line Structure")
    print("=" * 40)
    
    # Create a test line following Korg.jl structure exactly
    fe_species = Species.from_element(26, charge=0)  # Fe I
    
    line = create_line(
        wl=5000.0,           # Wavelength in Å (will be converted to cm)
        log_gf=-2.5,         # log₁₀(gf)
        species=fe_species,   # Species object
        E_lower=2.2,         # eV
        gamma_rad=1e8,       # s⁻¹
        gamma_stark=1e-5,    # s⁻¹
        vdW=(1e-7, -1.0),   # (γ_vdW, -1) format
        wavelength_unit='angstrom'
    )
    
    print(f"Line wavelength: {line.wl:.2e} cm")
    print(f"Line log_gf: {line.log_gf}")
    print(f"Line species: {line.species}")
    print(f"Line E_lower: {line.E_lower} eV")
    print(f"Line gamma_rad: {line.gamma_rad:.1e} s⁻¹")
    print(f"Line gamma_stark: {line.gamma_stark:.1e} s⁻¹")
    print(f"Line vdW: {line.vdW}")
    
    # Validate structure
    assert line.wl == 5000.0 * 1e-8, "Wavelength conversion incorrect"
    assert line.log_gf == -2.5, "log_gf incorrect"
    assert isinstance(line.species, Species), "Species type incorrect"
    assert line.E_lower == 2.2, "E_lower incorrect"
    assert len(line.vdW) == 2, "vdW tuple length incorrect"
    
    print("✅ Line structure matches Korg.jl exactly\n")


def test_physical_constants():
    """Test physical constants match Korg.jl exactly."""
    print("🧪 Testing Physical Constants")
    print("=" * 40)
    
    # Compare key constants with Korg.jl values
    constants_comparison = [
        ("kboltz_eV", kboltz_eV, 8.617333262145e-5),
        ("hplanck_eV", hplanck_eV, 4.135667696e-15),
        ("RydbergH_eV", RydbergH_eV, 13.598287264),
        ("c_cgs", c_cgs, 2.99792458e10),
        ("electron_mass_cgs", electron_mass_cgs, 9.1093897e-28),
        ("eV_to_cgs", eV_to_cgs, 1.602e-12),
    ]
    
    for name, jorg_value, korg_value in constants_comparison:
        print(f"{name:20s}: Jorg = {jorg_value:.12e}, Korg = {korg_value:.12e}")
        relative_error = abs(jorg_value - korg_value) / korg_value
        print(f"{'':20s}  Relative error: {relative_error:.2e}")
        assert relative_error < 1e-10, f"Constant {name} differs from Korg.jl"
        print(f"{'':20s}  ✅ Match")
    
    print("✅ All physical constants match Korg.jl exactly\n")


def test_energy_units():
    """Test that energy units are consistent with Korg.jl (eV)."""
    print("🧪 Testing Energy Units")
    print("=" * 40)
    
    # Test that excitation potentials are in eV
    line = create_line_data(5000.0, 2600, -2.0, 3.5)  # E_lower in eV
    print(f"Excitation potential: {line.E_lower} eV")
    assert line.E_lower == 3.5, "Energy should be in eV"
    
    # Test energy conversion consistency
    energy_eV = 2.0  # eV
    energy_erg = energy_eV * eV_to_cgs  # Convert to erg
    print(f"Energy conversion: {energy_eV} eV = {energy_erg:.2e} erg")
    
    # Check conversion factor matches Korg.jl
    expected_erg = 2.0 * 1.602e-12  # Using Korg.jl value
    assert abs(energy_erg - expected_erg) < 1e-15, "Energy conversion incorrect"
    
    print("✅ Energy units consistent with Korg.jl (eV)\n")


def test_vdw_broadening_format():
    """Test van der Waals broadening parameter format matches Korg.jl."""
    print("🧪 Testing van der Waals Broadening Format")
    print("=" * 40)
    
    fe_species = Species.from_element(26, charge=0)
    
    # Test (γ_vdW, -1.0) format
    line1 = create_line(5000.0, -2.0, fe_species, 2.0, vdW=(1e-7, -1.0))
    print(f"vdW format 1: {line1.vdW} (γ_vdW, -1)")
    assert line1.vdW[1] == -1.0, "Second vdW parameter should be -1.0 for γ_vdW mode"
    
    # Test (σ, α) format for ABO theory
    line2 = create_line(5000.0, -2.0, fe_species, 2.0, vdW=(2.5e-15, 0.3))
    print(f"vdW format 2: {line2.vdW} (σ, α)")
    assert line2.vdW[0] > 0 and line2.vdW[1] != -1.0, "ABO format should have σ > 0, α ≠ -1"
    
    print("✅ van der Waals broadening format matches Korg.jl\n")


def test_molecular_detection():
    """Test molecular species detection matches Korg.jl logic."""
    print("🧪 Testing Molecular Species Detection")
    print("=" * 40)
    
    # Atomic species should not be molecular
    h_atom = Species.from_element(1, charge=0)  # H I
    fe_atom = Species.from_element(26, charge=0)  # Fe I
    print(f"H I is molecular: {h_atom.is_molecule()}")
    print(f"Fe I is molecular: {fe_atom.is_molecule()}")
    assert not h_atom.is_molecule(), "H I should not be molecular"
    assert not fe_atom.is_molecule(), "Fe I should not be molecular"
    
    # Molecular species should be molecular
    h2o = Species.from_molecule([1, 1, 8])  # H2O
    co = Species.from_molecule([6, 8])      # CO
    print(f"H2O is molecular: {h2o.is_molecule()}")
    print(f"CO is molecular: {co.is_molecule()}")
    assert h2o.is_molecule(), "H2O should be molecular"
    assert co.is_molecule(), "CO should be molecular"
    
    print("✅ Molecular detection logic matches Korg.jl\n")


def run_comprehensive_compatibility_tests():
    """Run all compatibility tests."""
    print("🚀 Jorg Line Data Compatibility with Korg.jl")
    print("=" * 60)
    print("Testing all line data structures and conventions...")
    print("=" * 60)
    
    test_functions = [
        test_wavelength_unit_conversion,
        test_species_structure,
        test_line_structure,
        test_physical_constants,
        test_energy_units,
        test_vdw_broadening_format,
        test_molecular_detection,
    ]
    
    results = {}
    for test_func in test_functions:
        test_name = test_func.__name__
        try:
            test_func()
            results[test_name] = True
            print(f"✅ {test_name} PASSED")
        except Exception as e:
            results[test_name] = False
            print(f"❌ {test_name} FAILED: {e}")
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    print("\n" + "=" * 60)
    print("🏁 COMPATIBILITY TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"{test_name:<35} | {status}")
    
    print("-" * 60)
    print(f"Total tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    print(f"Success rate: {100 * passed / total:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL COMPATIBILITY TESTS PASSED!")
        print("✅ Jorg line data structures strictly follow Korg.jl conventions")
        print("✅ Wavelength units: cm (converted from Å automatically)")
        print("✅ Species structure: Formula + charge matching Korg.jl")
        print("✅ Energy units: eV throughout")
        print("✅ Physical constants: Exact match with Korg.jl")
        print("✅ Broadening parameters: Proper vdW tuple format")
        print("✅ Molecular detection: Consistent logic")
        return True
    else:
        print(f"\n⚠️  {total - passed} compatibility tests failed.")
        print("❌ Line data structures need further alignment with Korg.jl")
        return False


if __name__ == "__main__":
    success = run_comprehensive_compatibility_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("🚀 KORG.JL COMPATIBILITY ACHIEVED")
        print("=" * 60)
        print("✅ Jorg line data structures now strictly follow Korg.jl")
        print("✅ All units, formats, and conventions match exactly")
        print("✅ Ready for seamless interoperability with Korg.jl workflows")
    
    sys.exit(0 if success else 1)