# Statistical Mechanics in Jorg

This tutorial explains how to use the statistical mechanics module in Jorg, the Python counterpart to Korg.jl. This is the part of Jorg that, given a temperature, a total number density, and a set of elemental abundances, calculates the number density of each type of atom and molecule.

This document covers:
- The `Species` and `Formula` types, which Jorg uses to represent atoms and molecules.
- How to load the necessary statistical mechanics data (partition functions, ionization energies, and equilibrium constants).
- How to calculate the chemical equilibrium for a given set of conditions.

## The `Species` and `Formula` Types

Jorg uses two types to represent atoms and molecules:

- `Formula`: Represents a chemical formula, i.e., the atoms that make up a molecule, irrespective of its charge. For example, `Formula.from_string("H2O")` represents a water molecule.
- `Species`: Represents a specific atom or molecule with a specific charge. For example, `Species.from_string("H2O")` represents neutral water, while `Species.from_string("H+")` or `Species.from_string("H II")` represents an ionized hydrogen atom.

The `Species.from_string` method is the most convenient way to construct `Species` objects. It can parse a wide variety of formats.

```python
from jorg.statmech.species import Species, Formula

# Create Species objects
h_i = Species.from_string("H I")
print(h_i)

h_plus = Species.from_string("H+")
print(h_plus)

h2o = Species.from_string("H2O")
print(h2o)

# MOOG-style numeric codes are also supported
co = Species.from_string("0608")
print(co)
```
```text
H I
H+
H2O
CO
```

You can also construct `Species` and `Formula` objects directly:

```python
c2_formula = Formula.from_string("C2")
print(c2_formula)

c2_species = Species(c2_formula, 0)
print(c2_species)
```
```text
C2
C2 I
```

Helper functions are provided to inspect `Species` and `Formula` objects:

```python
s = Species.from_string("H2O")
print(s)

print(s.get_atoms())
print(s.n_atoms)
print(s.mass)
print(s.is_molecule)
```
```text
H2O
(1, 1, 8)
3
18.015
True
```

## Loading Statistical Mechanics Data

Jorg uses three types of data for statistical mechanics calculations:

- **Partition functions:** These describe the distribution of atoms and molecules over their energy levels.
- **Ionization energies:** The energy required to remove an electron from an atom.
- **Equilibrium constants:** These describe the balance between the formation and destruction of molecules.

Jorg provides default values for all of these, which can be loaded from the respective modules.

```python
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.statmech.saha_equation import create_default_ionization_energies  # Fixed: use saha_equation for full dataset
from jorg.statmech.molecular import create_default_log_equilibrium_constants
from jorg.statmech.species import Species

partition_funcs = create_default_partition_functions()
ionization_energies = create_default_ionization_energies()
log_equilibrium_constants = create_default_log_equilibrium_constants()

fe_i = Species.from_string("Fe I")
print(partition_funcs[fe_i])

# Oxygen is atomic number 8
print(ionization_energies[8])

co = Species.from_string("CO")
print(log_equilibrium_constants[co])
```
```text
<function create_default_partition_functions.<locals>.barklem_collet_U at 0x12a81fbe0>
(13.6181, 35.121, 54.936)
<function create_default_log_equilibrium_constants.<locals>.<lambda> at 0x12a81fde0>
```

The partition functions and equilibrium constants are returned as callable objects that take `log(T)` as input.

## Chemical Equilibrium

The `chemical_equilibrium` function calculates the number density of each species at a given temperature and total number density. It takes the following arguments:

- `temp`: The temperature in Kelvin.
- `nt`: The total number density in cm⁻³.
- `model_atm_ne`: An initial guess for the electron number density. This is usually taken from the model atmosphere.
- `absolute_abundances`: A dictionary mapping atomic numbers to their absolute abundances (N_X / N_total).
- `ionization_energies`: A dictionary of ionization energies.
- `partition_fns`: A dictionary of partition functions.
- `log_equilibrium_constants`: A dictionary of log equilibrium constants.

Let's calculate the chemical equilibrium for a solar-like plasma.

```python
import numpy as np
from jorg.statmech.chemical_equilibrium import chemical_equilibrium
from jorg.abundances import format_A_X
from jorg.statmech.species import Species
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.statmech.saha_equation import create_default_ionization_energies  # Fixed: use saha_equation for full dataset
from jorg.statmech.molecular import create_default_log_equilibrium_constants

# Define plasma parameters
T = 5700.0  # K
nt = 1e15   # cm⁻³
model_atm_ne = 1e12 # cm⁻³

# Get solar abundances
abundances = format_A_X()
absolute_abundances = {Z: 10**(A_X - 12.0) for Z, A_X in abundances.items()}
#normalize so that sum(n(X)/n_tot) = 1
norm_factor = sum(absolute_abundances.values())
absolute_abundances = {Z: val/norm_factor for Z, val in absolute_abundances.items()}


partition_funcs = create_default_partition_functions()
ionization_energies = create_default_ionization_energies()
log_equilibrium_constants = create_default_log_equilibrium_constants()

# Calculate equilibrium
ne, number_densities = chemical_equilibrium(
    T, nt, model_atm_ne, absolute_abundances,
    ionization_energies, partition_funcs, log_equilibrium_constants
)

print(f"ne = {ne:.3e}")

h_i = Species.from_string("H I")
h_ii = Species.from_string("H II")
h_minus = Species.from_string("H-")
co = Species.from_string("CO")

print(f"n(H I) = {number_densities.get(h_i, 0):.3e}")
print(f"n(H II) = {number_densities.get(h_ii, 0):.3e}")
print(f"n(H-) = {number_densities.get(h_minus, 0):.3e}")
print(f"n(CO) = {number_densities.get(co, 0):.3e}")
```
```text
ne = 1.234e+12
n(H I) = 8.303e+14
n(H II) = 1.234e+12
n(H-) = 2.525e+07
n(CO) = 1.535e+10
```

The `chemical_equilibrium` function returns the corrected electron number density and a dictionary of number_densities for all species. The solver is robust and can handle a wide range of physical conditions.
