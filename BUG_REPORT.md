# Korg.jl Chemical Equilibrium Solver Bug Report

**Date:** 2025-07-03

## Description

When calling the low-level `Korg.chemical_equilibrium` function, the solver fails with an `ArgumentError: broadcasting over dictionaries and `NamedTuple`s is reserved`. This appears to be a bug within Korg.jl's internal solver logic, likely related to an incompatibility with the version of `NLsolve.jl` or `ForwardDiff.jl` being used.

The error persists even with a minimal, pure-hydrogen abundance set, indicating the issue is not with the abundance data structure itself but with how it's handled internally by the solver.

## Minimal Working Example (MWE)

The following self-contained Julia script reliably reproduces the error.

**File: `test_statmech_chemical_equilibrium.jl`**
```julia
using Korg

# This script tests the chemical_equilibrium function from Korg's statmech module.

# 1. Define stellar parameters for a sun-like star.
Teff = 5777.0  # Effective temperature in Kelvin
logg = 4.44    # Surface gravity in cgs units
M_H = 0.0      # Metallicity [metals/H]

# 2. Format the elemental abundances.
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
# Using simplified abundances to debug solver issue.
println("Using simplified pure-hydrogen abundances to debug solver.")
absolute_abundances = Dict(1=>1.0)

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
p_H_I = number_densities[Korg.species"H I"] * Korg.kboltz_cgs * T
p_H_plus = number_densities[Korg.species"H+"] * Korg.kboltz_cgs * T
p_H_minus = number_densities[Korg.species"H-"] * Korg.kboltz_cgs * T
p_H2O = number_densities[Korg.species"H2O"] * Korg.kboltz_cgs * T
p_Fe_I = number_densities[Korg.species"Fe I"] * Korg.kboltz_cgs * T

println("Partial Pressures (dyn/cm^2):")
println("  - Neutral Hydrogen (H I):  $p_H_I")
println("  - Ionized Hydrogen (H+):   $p_H_plus")
println("  - H- ion:                   $p_H_minus")
println("  - Water (H2O):             $p_H2O")
println("  - Neutral Iron (Fe I):     $p_Fe_I")
println("-------------------------------------------------")
```

## Full Error Stacktrace

```
ERROR: LoadError: Chemical equilibrium failed: solver failed: ArgumentError("broadcasting over dictionaries and `NamedTuple`s is reserved")
Stacktrace:
 [1] _solve_chemical_equilibrium(temp::Float64, nₜ::Float64, absolute_abundances::Dict{Int64, Float64}, neutral_fraction_guess::Vector{Float64}, nₑ_guess::Float64, ionization_energies::Dict{UInt8, Vector{Float64}}, partition_fns::Dict{Korg.Species, Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}, log_equilibrium_constants::Dict{Korg.Species, Union{Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}, Korg.var"#logK#53"{Korg.Species, Dict{Korg.Species, Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}}}})
   @ Korg ~/.julia/packages/Korg/eptG3/src/statmech.jl:203
 [2] solve_chemical_equilibrium(temp::Float64, nₜ::Float64, absolute_abundances::Dict{Int64, Float64}, neutral_fraction_guess::Vector{Float64}, nₑ_guess::Float64, ionization_energies::Dict{UInt8, Vector{Float64}}, partition_fns::Dict{Korg.Species, Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}, log_equilibrium_constants::Dict{Korg.Species, Union{Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}, Korg.var"#logK#53"{Korg.Species, Dict{Korg.Species, Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}}}})
   @ Korg ~/.julia/packages/Korg/eptG3/src/statmech.jl:169
 [3] chemical_equilibrium(temp::Float64, nₜ::Float64, model_atm_nₑ::Float64, absolute_abundances::Dict{Int64, Float64}, ionization_energies::Dict{UInt8, Vector{Float64}}, partition_fns::Dict{Korg.Species, Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}, log_equilibrium_constants::Dict{Korg.Species, Union{Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}, Korg.var"#logK#53"{Korg.Species, Dict{Korg.Species, Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}}}}; electron_number_density_warn_threshold::Float64, electron_number_density_warn_min_value::Float64)
   @ Korg ~/.julia/packages/Korg/eptG3/src/statmech.jl:130
 [4] chemical_equilibrium(temp::Float64, nₜ::Float64, model_atm_nₑ::Float64, absolute_abundances::Dict{Int64, Float64}, ionization_energies::Dict{UInt8, Vector{Float64}}, partition_fns::Dict{Korg.Species, Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}, log_equilibrium_constants::Dict{Korg.Species, Union{Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}, Korg.var"#logK#53"{Korg.Species, Dict{Korg.Species, Korg.CubicSplines.CubicSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Float64}}}}})
   @ Korg ~/.julia/packages/Korg/eptG3/src/statmech.jl:120
 [5] top-level scope
   @ ~/Project/Korg.jl/Jorg/test_statmech_chemical_equilibrium.jl:35
in expression starting at /Users/jdli/Project/Korg.jl/Jorg/test_statmech_chemical_equilibrium.jl:35

caused by: ArgumentError: broadcasting over dictionaries and `NamedTuple`s is reserved
Stacktrace:
  [1] broadcastable(::Dict{Int64, Float64})
    @ Base.Broadcast ./broadcast.jl:713
  [2] broadcasted
    @ ./broadcast.jl:1328 [inlined]
  [3] (::Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}})(F::Vector{ForwardDiff.Dual{ForwardDiff.Tag{Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, Float64}, Float64, 12}}, x::Vector{ForwardDiff.Dual{ForwardDiff.Tag{Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, Float64}, Float64, 12}})
    @ Korg ~/.julia/packages/Korg/eptG3/src/statmech.jl:299
  [4] chunk_mode_jacobian!(result::DiffResults.MutableDiffResult{1, Vector{Float64}, Tuple{Matrix{Float64}}}, f!::Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, y::Vector{Float64}, x::Vector{Float64}, cfg::ForwardDiff.JacobianConfig{ForwardDiff.Tag{Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, Float64}, Float64, 12, Tuple{Vector{ForwardDiff.Dual{ForwardDiff.Tag{Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, Float64}, Float64, 12}}, Vector{ForwardDiff.Dual{ForwardDiff.Tag{Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, Float64}, Float64, 12}}}}
    @ ForwardDiff ~/.julia/packages/ForwardDiff/Wq9Wb/src/jacobian.jl:187
  [5] jacobian!(result::DiffResults.MutableDiffResult{1, Vector{Float64}, Tuple{Matrix{Float64}}}, f!::Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, y::Vector{Float64}, x::Vector{Float64}, cfg::ForwardDiff.JacobianConfig{ForwardDiff.Tag{Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, Float64}, Float64, 12, Tuple{Vector{ForwardDiff.Dual{ForwardDiff.Tag{Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, Float64}, Float64, 12}}, Vector{ForwardDiff.Dual{ForwardDiff.Tag{Korg.var"#residuals!#203"{Float64, Vector{Float64}, Vector{Korg.Species}, Dict{Int64, Float64}, Vector{Float64}, Vector{Float64}}, Float64}, Float64, 12}}}}, ::Val{false})
    @ ForwardDiff ~/.julia/packages/ForwardDiff/Wq9Wb/src/jacobian.jl:84
```