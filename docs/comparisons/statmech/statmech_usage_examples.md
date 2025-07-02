# Statistical Mechanics Module Usage Examples

This document provides comprehensive examples of using the `jorg.statmech` module for stellar atmosphere calculations. The statistical mechanics module is the foundation of stellar spectroscopy calculations, determining the number densities of all atomic and molecular species at given temperature, pressure, and abundance conditions.

## Overview

The `jorg.statmech` module provides:

- **Species and Formula types** for representing atoms and molecules
- **Chemical equilibrium solver** using the Saha equation and molecular equilibrium
- **Partition functions** for atoms and ions at various temperatures
- **Ionization balance calculations** using Barklem & Collet 2016 data
- **Molecular equilibrium** for common stellar atmosphere molecules

## Module Structure

```python
from jorg.statmech import (
    # Core chemical equilibrium
    chemical_equilibrium,
    
    # Species representation
    Species, Formula,
    
    # Data creation functions
    create_default_partition_functions,
    create_default_ionization_energies,
    create_default_log_equilibrium_constants,
    
    # Low-level functions
    saha_ion_weights,
    translational_U,
    get_log_nK
)
```

## 1. Working with Species and Formulas

### Creating Species Objects

The `Species` class represents chemical species (atoms or molecules with specific charge states):

```python
from jorg.statmech import Species, Formula

# Atomic species using different notation styles
h_neutral = Species.from_string("H I")        # Roman numeral notation
h_ion = Species.from_string("H II")           # Singly ionized hydrogen
h_plus = Species.from_string("H+")            # Alternative notation
h_minus = Species.from_string("H-")           # Hydrogen anion

# Iron species
fe_neutral = Species.from_string("Fe I")      # Neutral iron
fe_ion = Species.from_string("Fe II")         # Singly ionized iron

# Molecular species
co = Species.from_string("CO")                # Carbon monoxide
h2o = Species.from_string("H2O")              # Water
oh = Species.from_string("OH")                # Hydroxyl radical

# MOOG-style numeric codes
co_moog = Species.from_string("0608")         # CO (C=6, O=8)

print(f"Hydrogen neutral: {h_neutral}")
print(f"Iron ion: {fe_ion}")
print(f"Carbon monoxide: {co}")
print(f"Water: {h2o}")
```

### Species Properties

```python
# Examine species properties
species = Species.from_string("Fe II")

print(f"Species: {species}")
print(f"Is atom: {species.is_atom}")
print(f"Is molecule: {species.is_molecule}")  
print(f"Is neutral: {species.is_neutral}")
print(f"Is ion: {species.is_ion}")
print(f"Charge: {species.charge}")
print(f"Mass (AMU): {species.mass}")
print(f"Number of atoms: {species.n_atoms}")
print(f"Atomic composition: {species.get_atoms()}")

# For molecules
h2o = Species.from_string("H2O")
print(f"\nWater molecule:")
print(f"  Formula: {h2o.formula}")
print(f"  Mass: {h2o.mass:.3f} AMU")
print(f"  Atoms: {h2o.get_atoms()}")  # (1, 1, 8) - two H, one O
```

### Creating Formulas

The `Formula` class represents chemical formulas without charge information:

```python
# Create formulas directly
h_formula = Formula.from_atomic_number(1)     # Hydrogen
co_formula = Formula.from_string("CO")        # Carbon monoxide
h2o_formula = Formula.from_string("H2O")      # Water

# Use formulas to create species
h_neutral = Species(h_formula, 0)             # H I
h_ion = Species(h_formula, 1)                 # H II

print(f"Hydrogen formula: {h_formula}")
print(f"CO formula: {co_formula}")
print(f"H2O atoms: {h2o_formula.get_atoms()}")
```

## 2. Loading Statistical Mechanics Data

### Default Data Sets

```python
from jorg.statmech import (
    create_default_partition_functions,
    create_default_ionization_energies,
    create_default_log_equilibrium_constants
)

# Load default data sets
partition_funcs = create_default_partition_functions()
ionization_energies = create_default_ionization_energies()
log_equilibrium_constants = create_default_log_equilibrium_constants()

print(f"Loaded partition functions for {len(partition_funcs)} species")
print(f"Loaded ionization energies for {len(ionization_energies)} elements")
print(f"Loaded equilibrium constants for {len(log_equilibrium_constants)} molecules")
```

### Examining the Data

```python
# Check ionization energies (in eV)
print("Ionization energies (eV):")
for element in [1, 2, 6, 8, 26]:  # H, He, C, O, Fe
    if element in ionization_energies:
        chi_I, chi_II, chi_III = ionization_energies[element]
        print(f"  Z={element}: χI={chi_I:.3f}, χII={chi_II:.3f}, χIII={chi_III:.3f}")

# Check partition functions
h_i = Species.from_string("H I")
fe_i = Species.from_string("Fe I")
fe_ii = Species.from_string("Fe II")

T = 5778.0  # Solar temperature
log_T = jnp.log(T)

print(f"\nPartition functions at T={T}K:")
print(f"  H I: {partition_funcs[h_i](log_T):.2f}")
print(f"  Fe I: {partition_funcs[fe_i](log_T):.2f}")
print(f"  Fe II: {partition_funcs[fe_ii](log_T):.2f}")

# Check molecular equilibrium constants
co = Species.from_string("CO")
oh = Species.from_string("OH")

print(f"\nLog equilibrium constants at T={T}K:")
if co in log_equilibrium_constants:
    print(f"  CO: {log_equilibrium_constants[co](log_T):.2f}")
if oh in log_equilibrium_constants:
    print(f"  OH: {log_equilibrium_constants[oh](log_T):.2f}")
```

## 3. Chemical Equilibrium Calculations

### Basic Setup

```python
import numpy as np
from jorg.statmech import chemical_equilibrium
from jorg.abundances import format_A_X

# Define stellar atmosphere conditions
T = 5778.0      # Temperature (K) - solar photosphere
nt = 1e15       # Total number density (cm⁻³)
model_atm_ne = 1e12  # Initial electron density guess (cm⁻³)

# Get solar abundances and convert to absolute scale
abundances = format_A_X()  # Returns log abundances relative to H=12
absolute_abundances = {}

for Z, log_abundance in abundances.items():
    absolute_abundances[Z] = 10**(log_abundance - 12.0)

# Normalize abundances so sum(N_X/N_total) = 1
total_abundance = sum(absolute_abundances.values())
for Z in absolute_abundances:
    absolute_abundances[Z] /= total_abundance

print(f"Solar abundances loaded for {len(absolute_abundances)} elements")
print(f"Hydrogen abundance: {absolute_abundances[1]:.3e}")
print(f"Helium abundance: {absolute_abundances[2]:.3e}")
print(f"Iron abundance: {absolute_abundances[26]:.3e}")
```

### Running Chemical Equilibrium

```python
# Calculate chemical equilibrium
ne, number_densities = chemical_equilibrium(
    T, nt, model_atm_ne, absolute_abundances,
    ionization_energies, partition_funcs, log_equilibrium_constants
)

print(f"\nChemical equilibrium results:")
print(f"  Temperature: {T} K")
print(f"  Total density: {nt:.2e} cm⁻³")
print(f"  Electron density: {ne:.2e} cm⁻³")
print(f"  Species calculated: {len(number_densities)}")
```

### Examining Results

```python
# Key atomic species
key_species = [
    ("H I", "H II", "H-"),           # Hydrogen
    ("He I", "He II"),               # Helium  
    ("C I", "C II"),                 # Carbon
    ("O I", "O II"),                 # Oxygen
    ("Fe I", "Fe II"),               # Iron
]

print("\nAtomic species number densities:")
for species_group in key_species:
    print(f"  {species_group[0][:-2]}:")  # Element name
    for species_name in species_group:
        species = Species.from_string(species_name)
        density = number_densities.get(species, 0.0)
        print(f"    {species_name}: {density:.3e} cm⁻³")

# Molecular species
molecular_species = ["CO", "OH", "H2", "H2O"]
print("\nMolecular species number densities:")
for mol_name in molecular_species:
    try:
        species = Species.from_string(mol_name)
        density = number_densities.get(species, 0.0)
        print(f"  {mol_name}: {density:.3e} cm⁻³")
    except:
        print(f"  {mol_name}: not found")
```

### Ionization Fractions

```python
# Calculate ionization fractions for key elements
elements = [1, 2, 6, 8, 26]  # H, He, C, O, Fe
element_names = ["H", "He", "C", "O", "Fe"]

print("\nIonization fractions:")
for Z, name in zip(elements, element_names):
    # Get number densities for different ionization states
    neutral = Species.from_atomic_number(Z, 0)
    ion1 = Species.from_atomic_number(Z, 1)  
    ion2 = Species.from_atomic_number(Z, 2)
    
    n_neutral = number_densities.get(neutral, 0.0)
    n_ion1 = number_densities.get(ion1, 0.0)
    n_ion2 = number_densities.get(ion2, 0.0)
    
    n_total = n_neutral + n_ion1 + n_ion2
    
    if n_total > 0:
        f_neutral = n_neutral / n_total
        f_ion1 = n_ion1 / n_total
        f_ion2 = n_ion2 / n_total
        
        print(f"  {name}: I={f_neutral:.3f}, II={f_ion1:.3f}, III={f_ion2:.3f}")
```

## 4. Temperature and Density Studies

### Temperature Dependence

```python
import matplotlib.pyplot as plt

# Study ionization vs temperature
temperatures = np.linspace(3000, 10000, 50)  # K
ionization_fractions = []

print("Calculating temperature dependence...")
for T in temperatures:
    # Recalculate equilibrium at each temperature
    ne, densities = chemical_equilibrium(
        T, nt, model_atm_ne, absolute_abundances,
        ionization_energies, partition_funcs, log_equilibrium_constants
    )
    
    # Calculate hydrogen ionization fraction
    h_neutral = Species.from_string("H I")
    h_ion = Species.from_string("H II")
    
    n_hi = densities.get(h_neutral, 0.0)
    n_hii = densities.get(h_ion, 0.0)
    n_total = n_hi + n_hii
    
    ion_frac = n_hii / n_total if n_total > 0 else 0.0
    ionization_fractions.append(ion_frac)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(temperatures, ionization_fractions, 'b-', linewidth=2)
plt.xlabel('Temperature (K)')
plt.ylabel('H II / (H I + H II)')
plt.title('Hydrogen Ionization vs Temperature')
plt.grid(True, alpha=0.3)
plt.xlim(3000, 10000)
plt.ylim(0, 1)
plt.show()
```

### Density Dependence

```python
# Study electron density vs total density
total_densities = np.logspace(12, 18, 30)  # cm⁻³
electron_densities = []

print("Calculating density dependence...")
for nt_test in total_densities:
    ne, _ = chemical_equilibrium(
        5778.0, nt_test, nt_test * 1e-3, absolute_abundances,
        ionization_energies, partition_funcs, log_equilibrium_constants
    )
    electron_densities.append(ne)

# Plot results
plt.figure(figsize=(10, 6))
plt.loglog(total_densities, electron_densities, 'r-', linewidth=2)
plt.xlabel('Total Number Density (cm⁻³)')
plt.ylabel('Electron Number Density (cm⁻³)')
plt.title('Electron Density vs Total Density (T=5778K)')
plt.grid(True, alpha=0.3)
plt.show()
```

## 5. Advanced Usage

### Custom Abundances

```python
# Create custom abundance pattern (e.g., metal-poor star)
custom_abundances = {
    1: 0.9,      # H - 90% by number
    2: 0.09,     # He - 9% by number  
    6: 1e-5,     # C - reduced
    8: 1e-4,     # O - reduced
    26: 1e-6,    # Fe - very reduced (metal-poor)
}

# Normalize
total = sum(custom_abundances.values())
for Z in custom_abundances:
    custom_abundances[Z] /= total

# Calculate equilibrium with custom abundances
ne_custom, densities_custom = chemical_equilibrium(
    5778.0, 1e15, 1e12, custom_abundances,
    ionization_energies, partition_funcs, log_equilibrium_constants
)

print(f"\nMetal-poor star results:")
print(f"  Electron density: {ne_custom:.2e} cm⁻³")

# Compare iron ionization fractions
fe_i = Species.from_string("Fe I")
fe_ii = Species.from_string("Fe II")

# Solar case
n_fei_solar = number_densities.get(fe_i, 0.0)
n_feii_solar = number_densities.get(fe_ii, 0.0)
frac_solar = n_feii_solar / (n_fei_solar + n_feii_solar) if (n_fei_solar + n_feii_solar) > 0 else 0

# Metal-poor case  
n_fei_poor = densities_custom.get(fe_i, 0.0)
n_feii_poor = densities_custom.get(fe_ii, 0.0)
frac_poor = n_feii_poor / (n_fei_poor + n_feii_poor) if (n_fei_poor + n_feii_poor) > 0 else 0

print(f"  Fe II fraction (solar): {frac_solar:.3f}")
print(f"  Fe II fraction (metal-poor): {frac_poor:.3f}")
```

### Low-Level Functions

```python
from jorg.statmech import saha_ion_weights, translational_U

# Direct Saha equation calculation
T = 5778.0
ne = 1e12
atom = 26  # Iron

# Calculate ionization weights directly
wII, wIII = saha_ion_weights(T, ne, atom, ionization_energies, partition_funcs)

print(f"\nDirect Saha calculation for Fe at T={T}K, ne={ne:.0e}:")
print(f"  n(Fe II)/n(Fe I) = {wII:.3e}")
print(f"  n(Fe III)/n(Fe I) = {wIII:.3e}")

# Translational partition function
electron_mass = 9.1093897e-28  # g
trans_U = translational_U(electron_mass, T)
print(f"  Translational U = {trans_U:.3e}")
```

## 6. Performance and Validation

### Benchmarking

```python
import time

# Benchmark chemical equilibrium calculation
print("Benchmarking chemical equilibrium solver...")

start_time = time.time()
for i in range(10):
    ne, densities = chemical_equilibrium(
        5778.0, 1e15, 1e12, absolute_abundances,
        ionization_energies, partition_funcs, log_equilibrium_constants
    )
end_time = time.time()

avg_time = (end_time - start_time) / 10
print(f"Average time per calculation: {avg_time:.3f} seconds")
print(f"Species calculated: {len(densities)}")
```

### Validation Against Known Results

```python
# Test hydrogen ionization in solar photosphere
# Expected: very low ionization fraction

h_neutral = Species.from_string("H I")
h_ion = Species.from_string("H II")

n_hi = number_densities[h_neutral]
n_hii = number_densities[h_ion]
ionization_frac = n_hii / (n_hi + n_hii)

print(f"\nValidation tests:")
print(f"  Solar H ionization fraction: {ionization_frac:.6f}")
print(f"  Expected: ~1e-4 to 1e-3 (PASS)" if 1e-5 < ionization_frac < 1e-2 else "  FAIL")

# Test electron density conservation
calculated_ne = sum(densities.get(Species.from_atomic_number(Z, 1), 0.0) + 
                   2*densities.get(Species.from_atomic_number(Z, 2), 0.0)
                   for Z in range(1, 93))

print(f"  Calculated ne: {ne:.2e}")
print(f"  Sum of ions: {calculated_ne:.2e}")
print(f"  Difference: {abs(ne - calculated_ne)/ne:.2e}")
print(f"  Electron conservation: PASS" if abs(ne - calculated_ne)/ne < 0.01 else "  FAIL")
```

## Troubleshooting

### Common Issues

1. **Convergence Problems**: If the solver fails to converge, try:
   - Adjusting the initial electron density guess
   - Reducing the temperature range
   - Checking abundance normalization

2. **Missing Species**: Some species may not be included in the default data:
   - Check if the species exists in the data dictionaries
   - Add custom data if needed for specialized applications

3. **Performance**: For large grids of calculations:
   - Pre-compile JAX functions where possible
   - Consider vectorizing over temperature/density arrays
   - Cache partition function evaluations

### Debug Mode

```python
# Enable detailed output for debugging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data consistency
print("Data consistency checks:")
print(f"  Partition functions: {len(partition_funcs)} species")
print(f"  Ionization energies: {len(ionization_energies)} elements")
print(f"  Equilibrium constants: {len(log_equilibrium_constants)} molecules")

# Validate input abundances
abundance_sum = sum(absolute_abundances.values())
print(f"  Abundance sum: {abundance_sum:.6f} (should be 1.0)")

if abs(abundance_sum - 1.0) > 1e-6:
    print("  WARNING: Abundances not properly normalized!")
```

This comprehensive tutorial demonstrates the full capabilities of the `jorg.statmech` module for stellar atmosphere calculations. The module provides a robust, high-precision implementation of chemical equilibrium that matches Korg.jl's accuracy while leveraging Python's ecosystem for analysis and visualization.