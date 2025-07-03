using Korg

# This script tests the chemical_equilibrium function from Korg's statmech module.

# 1. Define stellar parameters for a sun-like star.
Teff = 5777.0  # Effective temperature in Kelvin
logg = 4.44    # Surface gravity in cgs units
M_H = 0.0      # Metallicity [metals/H]

# 2. Format the elemental abundances.
# This uses the default solar abundances from Asplund et al. (2020).
A_X = Korg.format_A_X(M_H)

# 3. Interpolate a model atmosphere from the MARCS grid.
println("Interpolating model atmosphere...")
atm = Korg.interpolate_marcs(Teff, logg, A_X)
println("Atmosphere loaded.")

# 4. Define the atmospheric layer to inspect.
layer_index = 25
layer = atm.layers[layer_index]
T = layer.temp
nt = layer.number_density
ne_guess = layer.electron_number_density
P = nt * Korg.kboltz_cgs * T

# 5. Prepare inputs for the low-level chemical_equilibrium function.

# Convert abundances to the correct format for chemical_equilibrium
println("Converting abundances to number fractions...")
# A_X contains log abundances: A_X[i] = log10(N_i/N_H) + 12
# Convert to number fractions: N_i/N_H = 10^(A_X[i] - 12)
rel_abundances = 10.0 .^ (A_X .- 12.0)
total_particles_per_H = sum(rel_abundances)
absolute_abundances = rel_abundances ./ total_particles_per_H

# 6. Calculate the chemical equilibrium for the single layer.
println("Calculating chemical equilibrium for layer $layer_index...")
ne_sol, number_densities = Korg.chemical_equilibrium(
    T, nt, ne_guess, absolute_abundances,
    Korg.ionization_energies, Korg.default_partition_funcs,
    Korg.default_log_equilibrium_constants
)
println("Calculation complete.")

# 7. Inspect and print the results for the chosen atmospheric layer.
temp_at_layer = atm.layers[layer_index].temp
pressure_at_layer = P

println("\n-------------------------------------------------")
println("Chemical Equilibrium Test Results")
println("Stellar Parameters: Teff=$Teff K, logg=$logg, [M/H]=$M_H")
println("Results for atmospheric layer: $layer_index")
println("Temperature at layer: $(round(temp_at_layer, digits=2)) K")
println("Total pressure at layer: $(round(pressure_at_layer, digits=2)) dyn/cm^2")
println("-------------------------------------------------")

# Convert number densities to partial pressures (P = n * k * T) and print.
# Use get() to safely access species that might not be present
n_H_I = get(number_densities, Korg.species"H I", 0.0)
n_H_plus = get(number_densities, Korg.species"H II", 0.0)  # H II is ionized hydrogen
n_H_minus = get(number_densities, Korg.species"H-", 0.0)
n_H2O = get(number_densities, Korg.species"H2O", 0.0)
n_Fe_I = get(number_densities, Korg.species"Fe I", 0.0)

p_H_I = n_H_I * Korg.kboltz_cgs * T
p_H_plus = n_H_plus * Korg.kboltz_cgs * T
p_H_minus = n_H_minus * Korg.kboltz_cgs * T
p_H2O = n_H2O * Korg.kboltz_cgs * T
p_Fe_I = n_Fe_I * Korg.kboltz_cgs * T

println("Partial Pressures (dyn/cm^2):")
println("  - Neutral Hydrogen (H I):  $p_H_I")
println("  - Ionized Hydrogen (H+):   $p_H_plus")
println("  - H- ion:                   $p_H_minus")
println("  - Water (H2O):             $p_H2O")
println("  - Neutral Iron (Fe I):     $p_Fe_I")
println("-------------------------------------------------")
