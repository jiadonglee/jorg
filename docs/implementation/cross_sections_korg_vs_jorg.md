# Korg.jl Cross-Sections and Jorg Parity

This note explains, from zero order, how Korg.jl turns atomic and molecular physics into opacity, where "cross-section" enters, and how the current Jorg implementation compares.

It is based on direct reading of:

- `Korg.jl-1.0.1/src/synthesize.jl`
- `Korg.jl-1.0.1/src/line_absorption.jl`
- `Korg.jl-1.0.1/src/ContinuumAbsorption/ContinuumAbsorption.jl`
- `Korg.jl-1.0.1/src/molecular_cross_sections.jl`
- `jorg/src/jorg/synthesis.py`
- `jorg/src/jorg/opacity/korg_line_processor.py`
- `jorg/src/jorg/continuum/exact_physics_continuum.py`
- `jorg/src/jorg/lines/molecular_cross_sections.py`

## 1. Zero-order picture

The synthetic-spectrum problem is:

1. For each atmospheric layer, figure out how many absorbers exist.
2. For each absorber and wavelength, compute how strongly it absorbs.
3. Sum all absorption into a linear absorption coefficient `alpha(lambda)` in units of `cm^-1`.
4. Solve radiative transfer through the atmosphere.

The most useful distinction is:

- Cross-section `sigma`: per-particle absorbing power, usually `cm^2`.
- Number density `n`: particles per volume, `cm^-3`.
- Linear absorption coefficient `alpha = n * sigma`: absorption per path length, `cm^-1`.

For spectral lines, the "cross-section" is not a flat number. It is a line profile spread around a line center. So in practice:

`alpha_line(lambda) = amplitude * profile(lambda)`

where:

- `amplitude` contains oscillator strength, level populations, and the line cross-section scale.
- `profile` is usually Voigt-like: Doppler core plus Lorentz wings.

For the continuum, there is no isolated line center. Instead Korg sums many physical processes:

- H I bound-free
- H- bound-free
- H- free-free
- H2+ bound-free and free-free
- He- free-free
- positive-ion free-free
- metal bound-free
- Thomson scattering
- Rayleigh scattering

## 2. What Korg.jl means by "cross-section"

Korg uses "cross-section" in three related but different senses:

### 2.1 Line cross-section scale

In `src/line_absorption.jl`, Korg defines:

`sigma_line(lambda) = (pi e^2 / (m_e c)) * (lambda^2 / c)`

This is the wavelength-space line cross-section per `gf`, not yet including the lower/upper level population factor and not yet multiplied by number density.

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:215`

### 2.2 Continuum photoionization/free-free/scattering cross-sections

In `src/ContinuumAbsorption/`, Korg loads or evaluates physical cross-sections for each continuum process, then converts them into a total continuum `alpha`.

Code:

- `Korg.jl-1.0.1/src/ContinuumAbsorption/ContinuumAbsorption.jl:39`

### 2.3 Precomputed molecular cross-sections

Korg can precompute a molecular species' total opacity per molecule on a 3D grid:

- microturbulence
- temperature
- wavelength

That table is later interpolated and multiplied by the molecule number density to get `alpha`.

Code:

- `Korg.jl-1.0.1/src/molecular_cross_sections.jl:48`
- `Korg.jl-1.0.1/src/molecular_cross_sections.jl:95`

## 3. Korg.jl synthesis pipeline

The main pipeline lives in `src/synthesize.jl`.

### 3.1 Wavelength grid and units

Korg converts the working wavelength grid to cgs (`cm`), not Angstrom.

It also builds a coarser continuum grid `cntm_wls` that extends beyond the requested synthesis window by `line_buffer + cntm_step`.

Code:

- `Korg.jl-1.0.1/src/synthesize.jl:177`
- `Korg.jl-1.0.1/src/synthesize.jl:181`

### 3.2 Abundances to absolute fractions

Korg expects a 92-element abundance vector `A_X` with `A(H) = 12`, then converts to:

`abs_abundances = 10^(A_X - 12)`

and renormalizes so the total fraction sums to 1.

Code:

- `Korg.jl-1.0.1/src/synthesize.jl:202`
- `Korg.jl-1.0.1/src/synthesize.jl:206`

### 3.3 Chemical equilibrium

For each atmospheric layer, Korg solves chemical equilibrium and gets:

- electron density `n_e`
- number density dictionary `n_dict[species]`

These densities are what later turn per-particle cross-sections into `alpha`.

Code:

- `Korg.jl-1.0.1/src/synthesize.jl:216`

### 3.4 Continuum absorption

Korg computes continuum first, layer by layer:

`alpha_cntm_vals = total_continuum_absorption(freqs, T, n_e, n_dict, partition_funcs)`

Then it interpolates that continuum onto the final wavelength grid.

For the anchored optical-depth scheme, it also computes `alpha_ref` at the atmosphere reference wavelength.

Code:

- `Korg.jl-1.0.1/src/synthesize.jl:231`
- `Korg.jl-1.0.1/src/synthesize.jl:236`

The continuum contributors are summed in:

- `Korg.jl-1.0.1/src/ContinuumAbsorption/ContinuumAbsorption.jl:50`

Specifically:

- `H_I_bf`
- `Hminus_bf`
- `Hminus_ff`
- `H2plus_bf_and_ff`
- `Heminus_ff`
- `positive_ion_ff_absorption!`
- `metal_bf_absorption!`
- `electron_scattering`
- `rayleigh`

### 3.5 Hydrogen lines are special

Korg does not want ordinary H I lines in the user linelist. Hydrogen lines are handled separately by `hydrogen_line_absorption!`.

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:44`
- `Korg.jl-1.0.1/src/synthesize.jl:282`

### 3.6 Generic line absorption in Korg

This is the core line-opacity algorithm.

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:32`

For each line and each layer:

#### Step A: thermal and microturbulent broadening

Korg computes Doppler width:

`sigma_D = lambda0 * sqrt(k T / m + xi^2 / 2) / c`

where `xi` is microturbulence in `cm/s`.

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:176`

#### Step B: damping constant

It starts from radiative damping, then for atoms adds:

- Stark broadening from `n_e`
- van der Waals broadening from neutral hydrogen

Then it converts the total damping from angular-frequency width to wavelength-space Lorentz HWHM:

`gamma_lambda = Gamma * lambda0^2 / (4 pi c)`

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:76`
- `Korg.jl-1.0.1/src/line_absorption.jl:85`

#### Step C: level population factor

Korg forms:

`levels_factor = exp(-beta E_lower) - exp(-beta E_upper)`

with:

`beta = 1 / (k_B_eV T)`

and:

`E_upper = E_lower + h c / lambda0`

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:38`
- `Korg.jl-1.0.1/src/line_absorption.jl:87`

This is the LTE absorption-minus-stimulated-emission factor.

#### Step D: number density divided by partition function

Korg precomputes:

`n_div_U[species] = n_species / U_species(T)`

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:40`

#### Step E: integrated line amplitude

Korg then forms the wavelength-integrated line coefficient:

`amplitude = gf * sigma_line(lambda0) * levels_factor * (n / U)`

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:90`

This is the key place where the line cross-section enters.

#### Step F: Korg's continuum-relative windowing

Korg does not evaluate every line on the full wavelength grid.

Instead it asks:

"How far from line center do I need to go before the line falls below
`cutoff_threshold * alpha_continuum(line_center)`?"

It computes a critical density:

`rho_crit = alpha_cntm(line_center) * cutoff_threshold / amplitude`

Then it finds:

- the Gaussian distance where the Doppler core falls to `rho_crit`
- the Lorentz distance where the wings fall to `rho_crit`

and combines them:

`window_size = sqrt(doppler_window^2 + lorentz_window^2)`

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:94`
- `Korg.jl-1.0.1/src/line_absorption.jl:95`
- `Korg.jl-1.0.1/src/line_absorption.jl:97`
- `Korg.jl-1.0.1/src/line_absorption.jl:99`

This is a major performance feature. It is also physically important because the cutoff is tied to the local continuum scale, not an arbitrary fixed width.

#### Step G: line profile

Inside the chosen window, Korg adds a Voigt profile:

`alpha += line_profile(lambda0, sigma_D, gamma_lambda, amplitude, lambda)`

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:107`

The profile implementation uses the Hjerting function approximation.

Code:

- `Korg.jl-1.0.1/src/line_absorption.jl:224`

## 4. Korg.jl molecular cross-sections

This part is easy to miss, but it matters a lot for big molecular linelists.

### 4.1 What Korg precomputes

Korg's `MolecularCrossSection` is not a single-line object. It is a table:

`sigma_mol(vmic, log10(T), lambda)`

for one molecular species.

Code:

- `Korg.jl-1.0.1/src/molecular_cross_sections.jl:4`

### 4.2 How Korg builds that table

Korg uses the ordinary `line_absorption!` machinery itself, but in a clever normalized setup:

- set continuum to 1
- set cutoff threshold to 1
- set molecule number density to `1 / cutoff_alpha`
- compute alpha
- multiply the result back by `cutoff_alpha`

The result is effectively a per-molecule cross-section table.

Code:

- `Korg.jl-1.0.1/src/molecular_cross_sections.jl:60`
- `Korg.jl-1.0.1/src/molecular_cross_sections.jl:71`
- `Korg.jl-1.0.1/src/molecular_cross_sections.jl:76`

### 4.3 How Korg uses that table in synthesis

During synthesis, Korg interpolates the precomputed table and multiplies by the actual molecule number density in each layer:

`alpha[i, :] += sigma_itp(vmic, log10(T_i), lambda) * n_molecule[i]`

Code:

- `Korg.jl-1.0.1/src/molecular_cross_sections.jl:95`
- `Korg.jl-1.0.1/src/molecular_cross_sections.jl:107`

Critically, Korg injects molecular cross-sections in two places:

- into `alpha_ref` for the anchored reference wavelength
- into the full `alpha` grid for the actual synthesis window

Code:

- `Korg.jl-1.0.1/src/synthesize.jl:266`
- `Korg.jl-1.0.1/src/synthesize.jl:296`

## 5. So what does Korg actually "calculate"?

The clean summary is:

1. Korg does not compute "one cross-section".
2. It computes continuum `alpha_continuum(lambda)` from many continuum cross-sections.
3. It computes line amplitudes from the line cross-section scale `sigma_line(lambda0)`.
4. It broadens those lines with Doppler + Lorentz physics.
5. It optionally interpolates precomputed molecular per-particle cross-section tables.
6. It sums everything into total `alpha(lambda, layer)`.

So if you ask "how does Korg calculate cross-sections?", the practical answer is:

- For ordinary atomic and molecular lines: by the `sigma_line + LTE level factor + Voigt profile` route.
- For continuum: by summing bound-free, free-free, and scattering processes.
- For precomputed molecular opacity tables: by tabulating the line-opacity machinery itself over `(vmic, logT, lambda)`.

## 6. Does Jorg have the same functions?

Short answer:

- For the main atomic-line path: mostly yes, and intentionally modeled on Korg.
- For continuum: largely yes in physics coverage, though not every path is guaranteed line-for-line identical.
- For precomputed molecular cross-sections in production synthesis: not yet fully.

## 7. Where Jorg matches Korg well

### 7.1 Same high-level two-stage architecture

Jorg's main synthesis path computes:

1. continuum-only opacity
2. line opacity added on top
3. hydrogen lines separately
4. radiative transfer

Code:

- `jorg/src/jorg/synthesis.py:582`
- `jorg/src/jorg/synthesis.py:840`
- `jorg/src/jorg/synthesis.py:862`
- `jorg/src/jorg/synthesis.py:880`

That is structurally the same idea as Korg's `synthesize`.

### 7.2 Jorg's active line-opacity engine mirrors Korg's formulas

The active Korg-compatible line path in Jorg is `KorgLineProcessor`, not the older helper utilities.

Its core formulas match Korg:

- `beta = 1 / (k_B_eV T)`
- Doppler width with microturbulence
- radiative + Stark + vdW damping
- `gamma = Gamma * lambda^2 / (4 pi c)`
- `levels_factor = exp(-beta E_lower) - exp(-beta E_upper)`
- `cross_section = (pi e^2 / m_e / c^2) * lambda^2`
- `amplitude = gf * cross_section * levels_factor * n_div_U`
- continuum-relative line windowing from `rho_crit`

Code:

- `jorg/src/jorg/opacity/korg_line_processor.py:695`
- `jorg/src/jorg/opacity/korg_line_processor.py:706`
- `jorg/src/jorg/opacity/korg_line_processor.py:738`
- `jorg/src/jorg/opacity/korg_line_processor.py:745`
- `jorg/src/jorg/opacity/korg_line_processor.py:750`
- `jorg/src/jorg/opacity/korg_line_processor.py:756`
- `jorg/src/jorg/opacity/korg_line_processor.py:783`
- `jorg/src/jorg/opacity/korg_line_processor.py:799`

Jorg also routes line opacity through this processor in the main synthesis path:

- `jorg/src/jorg/synthesis.py:1514`
- `jorg/src/jorg/synthesis.py:1559`

### 7.3 Jorg continuum coverage is designed to mirror Korg's component list

Jorg's exact continuum path explicitly includes:

- H- bound-free
- H- free-free
- H2+ bf+ff
- He- free-free
- positive-ion free-free
- metal bound-free
- Thomson scattering
- H I bound-free

Code:

- `jorg/src/jorg/continuum/exact_physics_continuum.py:177`
- `jorg/src/jorg/continuum/exact_physics_continuum.py:195`
- `jorg/src/jorg/continuum/exact_physics_continuum.py:292`
- `jorg/src/jorg/continuum/exact_physics_continuum.py:309`
- `jorg/src/jorg/continuum/exact_physics_continuum.py:325`
- `jorg/src/jorg/continuum/exact_physics_continuum.py:340`
- `jorg/src/jorg/continuum/exact_physics_continuum.py:355`

So in intent and component coverage, Jorg is clearly aiming at Korg parity.

## 8. Where Jorg does not yet match Korg

### 8.1 The `molecular_cross_sections` argument is exposed but not wired into the main synthesis path

Jorg's `synthesize` signature includes `molecular_cross_sections`, but the main path shown in `synthesis.py` computes:

- `alpha_continuum`
- `line_opacity`
- `alpha_matrix = alpha_continuum + line_opacity`
- optional hydrogen lines

There is no corresponding call that injects molecular cross-sections into either:

- the anchored reference opacity equivalent to `alpha_ref`
- the full synthesis opacity matrix

Code:

- `jorg/src/jorg/synthesis.py:582`
- `jorg/src/jorg/synthesis.py:849`
- `jorg/src/jorg/synthesis.py:864`
- `jorg/src/jorg/synthesis.py:878`

By contrast, Korg explicitly adds them in both places:

- `Korg.jl-1.0.1/src/synthesize.jl:266`
- `Korg.jl-1.0.1/src/synthesize.jl:296`

This is the most important parity gap if your question is specifically about precomputed molecular cross-sections.

### 8.2 Jorg has a molecular-cross-section module, but it is not the same implementation strategy as Korg's production one

Jorg does have:

- a `MolecularCrossSection` class
- interpolation
- save/load helpers

Code:

- `jorg/src/jorg/lines/molecular_cross_sections.py:25`
- `jorg/src/jorg/lines/molecular_cross_sections.py:67`
- `jorg/src/jorg/lines/molecular_cross_sections.py:120`

But its construction path is explicitly simplified:

- it loops over lines directly
- it says "Use simplified approach"
- it does not reuse the exact active Korg-compatible `line_absorption` pipeline in the same normalized way Korg does

Code:

- `jorg/src/jorg/lines/molecular_cross_sections.py:195`
- `jorg/src/jorg/lines/molecular_cross_sections.py:196`

So today Jorg's molecular-cross-section module is conceptually similar, but not equivalent to Korg's mature production path.

### 8.3 There is at least one legacy helper in Jorg with a different line-cross-section formula

In `jorg/src/jorg/lines/utils.py`, `sigma_line` is written as:

`prefactor * wl^2 / (4 pi)`

Code:

- `jorg/src/jorg/lines/utils.py:70`

That is not the same normalization as Korg's active formula in `line_absorption.jl`.

Important caveat:

- this does not appear to be the active Korg-compatible synthesis path
- the active synthesis path uses `KorgLineProcessor`, which does use the Korg-style constant

So the practical conclusion is not "Jorg is wrong everywhere".
It is:

- the production synthesis path is much closer to Korg than some older helper modules are
- not every helper utility in the repository is guaranteed to be parity-clean

## 9. Final verdict

If you want the plain answer:

### Korg.jl

Korg calculates opacity by:

1. solving chemical equilibrium per layer
2. computing continuum absorption from physical continuum cross-sections
3. computing line amplitudes from a wavelength-space line cross-section scale
4. broadening lines with Doppler and Lorentz physics
5. truncating each line where it becomes negligible relative to the continuum
6. optionally adding precomputed molecular cross-sections
7. summing all of that into total `alpha(layer, lambda)`

### Jorg

Jorg already has:

- the same broad synthesis architecture
- a Korg-style line-opacity engine
- similar continuum physics coverage
- a separate molecular-cross-section module

But Jorg does not yet fully match Korg in one important place:

- precomputed `molecular_cross_sections` are not currently integrated into the main synthesis path the way Korg does

So the honest answer is:

- Jorg has the same main atomic-line idea and much of the same continuum machinery
- Jorg does not yet have the same end-to-end production molecular-cross-section behavior as Korg.jl

## 10. If you want to think about it in one equation

The full synthesis engine is basically:

`alpha_total = alpha_continuum + alpha_hydrogen_lines + alpha_other_lines + alpha_precomputed_molecules`

where:

- `alpha_continuum` comes from bf/ff/scattering cross-sections
- `alpha_other_lines` comes from `gf * sigma_line * LTE_population_factor * Voigt_profile`
- `alpha_precomputed_molecules` comes from interpolated molecular cross-section tables times molecule number density

That is the simplest mental model that is still faithful to the code.

## 11. Recommended next step for Jorg

If the goal is true Korg parity, the next implementation target should be:

1. wire `molecular_cross_sections` into `jorg.synthesize`
2. inject them both into the reference opacity path and the full opacity matrix
3. make Jorg's molecular-cross-section builder reuse the same active Korg-compatible line-opacity kernel, instead of the current simplified path

That would close the largest remaining cross-section gap.
