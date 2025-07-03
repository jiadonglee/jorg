# Statistical Mechanics Unit Tests

This directory contains comprehensive unit tests for Jorg's statistical mechanics module, which handles chemical equilibrium calculations, ionization balance, and partition functions for stellar atmosphere modeling.

## Test Files Overview

### `test_statmech.py` (541 lines)
**Main comprehensive test suite**
- **TestPartitionFunctions**: Tests atomic partition functions for H, Fe, and temperature scaling
- **TestSahaEquation**: Tests Saha ionization equilibrium across temperature/density ranges  
- **TestIonizationBalance**: Tests multi-element ionization balance and conservation laws
- **TestMolecularEquilibrium**: Tests H₂ and CO molecular formation equilibrium
- **TestChemicalEquilibrium**: Tests complete chemical equilibrium solver convergence
- **TestEdgeCases**: Tests extreme conditions and numerical stability

### `test_core.py` (249 lines)
**Core chemical equilibrium functionality**
- **TestChemicalEquilibrium**: Basic chemical equilibrium calculations with H+He
- **TestChemicalEquilibriumReference**: Validation against Korg reference conditions
- **TestChemicalEquilibriumErrors**: Error handling for invalid inputs
- Tests neutral fraction guesses, residuals computation, and conservation

### `test_ionization.py` (195 lines) 
**Ionization physics and Saha equation**
- **TestTranslationalU**: Translational partition function calculations and scaling laws
- **TestSahaIonWeights**: Saha equation for H, He, and multi-element systems
- **TestIonizationEnergies**: Validation of ionization energy database
- Tests temperature/density dependencies and conservation of ionization fractions

### `test_partition_functions.py` (126 lines)
**Partition function calculations**
- **TestPartitionFunctions**: H, He, and other element partition functions
- **TestPartitionFunctionReference**: Validation against Korg reference values
- Tests partition function dictionary creation and temperature scaling
- Validates H partition function matches Korg (2.000000011513405 vs 2.0)

## Key Test Categories

### Physics Validation
- **Saha equation**: Temperature and density dependencies match expected behavior
- **Conservation laws**: Mass and charge conservation in chemical equilibrium
- **Ionization trends**: Higher T increases ionization, higher ne decreases it
- **Partition functions**: Proper temperature scaling and reference value matching

### Numerical Testing  
- **Convergence**: Chemical equilibrium solver converges from different initial conditions
- **Stability**: Handles extreme temperatures (500K - 50000K) and tiny abundances
- **Precision**: Maintains accuracy for solar photosphere conditions
- **Edge cases**: Zero abundances and extreme density ranges

### Reference Validation
- **Korg compatibility**: Direct comparison with Korg.jl reference values
- **Literature values**: Solar hydrogen ionization (~1.5×10⁻⁴) and iron ionization (~93%)
- **Physical reasonableness**: Results fall within expected stellar atmosphere ranges

## Running Tests

Run all statmech unit tests:
```bash
cd /Users/jdli/Project/Korg.jl/Jorg
python -m pytest tests/unit/statmech/ -v
```

Run specific test file:
```bash
python -m pytest tests/unit/statmech/test_core.py -v
```

Run with coverage:
```bash
python -m pytest tests/unit/statmech/ --cov=jorg.statmech --cov-report=term-missing
```

## Test Dependencies

- **JAX/NumPy**: Array operations and automatic differentiation
- **pytest**: Test framework and fixtures
- **jorg.statmech**: Core statistical mechanics modules being tested
- **jorg.constants**: Physical constants (kboltz, electron mass, etc.)

## Expected Results

All tests should pass with the optimized chemical equilibrium implementation. Key validations:

- **Hydrogen**: Mostly neutral at solar conditions (fᵢ > 0.5)
- **Temperature scaling**: Higher T increases ionization for all elements  
- **Conservation**: Mass and charge conserved to <1e-6 relative error
- **Korg matching**: Partition functions match Korg reference to <1e-6
- **Performance**: Functions execute in μs-ms range for stellar conditions

The test suite ensures Jorg's statistical mechanics provides accurate, Korg-compatible results for stellar atmosphere synthesis while maintaining computational efficiency.