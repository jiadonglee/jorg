# Jorg Precision Regression Test Report

Generated: 2025-08-07 13:54:48

## Overall Results

- **Overall Precision**: 100.000%
- **Target Precision**: <0.1% RMS
- **Production Ready**: ❌ NO
- **Critical Failures**: 5

## Component Precision Summary

- **wavelength_grid**: 100.000% ✅

## Test Case Results

### Solar Reference

- Status: ERROR
- Precision: 0.00%
- Runtime: 0.0s

### Hot A-type

- Status: ERROR
- Precision: 0.00%
- Runtime: 0.0s

### Cool K-dwarf

- Status: ERROR
- Precision: 0.00%
- Runtime: 0.0s

### Metal-poor Giant

- Status: ERROR
- Precision: 0.00%
- Runtime: 0.0s

### Metal-rich Dwarf

- Status: ERROR
- Precision: 0.00%
- Runtime: 0.0s

### Wavelength Precision

- Status: PASS
- Precision: 100.00%
- Runtime: 0.0s
- Components:
  - wavelength_grid: 100.00%

## Critical Failures

- Solar Reference: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
- Hot A-type: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
- Cool K-dwarf: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
- Metal-poor Giant: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
- Metal-rich Dwarf: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'

## Recommendations

⚠️ **System needs improvement before production use**

Priority issues to address:
- Solar Reference: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
- Hot A-type: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
- Cool K-dwarf: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
- Metal-poor Giant: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
- Metal-rich Dwarf: ERROR - 'ChemicalEquilibriumValidator' object has no attribute 'validate_chemical_equilibrium_precision'
