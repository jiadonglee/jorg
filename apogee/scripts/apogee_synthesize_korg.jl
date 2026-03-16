using Pkg

if length(ARGS) != 16
    error("Usage: apogee_synthesize_korg.jl <korg_root> <output_path> <teff> <logg> <m_h> <alpha_h> <c_h> <vmic> <mu_values> <water_sigma_path> <use_exomol_aug> <manifest|- > <start_A> <stop_A> <step_A> <resolution>")
end

korg_root = ARGS[1]
output_path = ARGS[2]
teff = parse(Float64, ARGS[3])
logg = parse(Float64, ARGS[4])
m_h = parse(Float64, ARGS[5])
alpha_h = parse(Float64, ARGS[6])
c_h = parse(Float64, ARGS[7])
vmic = parse(Float64, ARGS[8])
mu_values = parse(Int, ARGS[9])
water_sigma_path = ARGS[10]
use_exomol_aug = parse(Int, ARGS[11]) == 1
manifest_path = ARGS[12]
synth_start_A = parse(Float64, ARGS[13])
synth_stop_A = parse(Float64, ARGS[14])
synth_step_A = parse(Float64, ARGS[15])
resolution = parse(Int, ARGS[16])

Pkg.activate(korg_root)

using HDF5
using Korg

function load_exomol_manifest!(linelist, manifest_path)
    if manifest_path == "-" || !isfile(manifest_path)
        return linelist
    end
    for row in eachline(manifest_path)
        isempty(strip(row)) && continue
        species_name, states_path, trans_path, ll, ul, cutoff, temp = split(row, '\t')
        exomol_lines = Korg.load_ExoMol_linelist(
            species_name,
            states_path,
            trans_path,
            parse(Float64, ll),
            parse(Float64, ul);
            line_strength_cutoff=parse(Float64, cutoff),
            T_line_strength=parse(Float64, temp),
            verbose=false,
        )
        append!(linelist, exomol_lines)
    end
    sort!(linelist; by=l -> l.wl)
    return linelist
end

water = Korg.read_molecular_cross_section(water_sigma_path)
A_X = Korg.format_A_X(m_h, alpha_h, Dict("C" => c_h))
atm = Korg.interpolate_marcs(teff, logg, A_X)
synthesis_wavelengths = synth_start_A:synth_step_A:synth_stop_A
apogee_wavelengths = 10 .^ range(; start=log10(15100.802), step=6e-6, length=8575)
lsf = Korg.compute_LSF_matrix(synthesis_wavelengths, apogee_wavelengths, resolution; verbose=false)

linelist = Korg.get_APOGEE_DR17_linelist(; include_water=false)
if use_exomol_aug
    load_exomol_manifest!(linelist, manifest_path)
end

solution = Korg.synthesize(
    atm,
    linelist,
    A_X,
    synthesis_wavelengths;
    vmic=vmic,
    use_MHD_for_hydrogen_lines=false,
    molecular_cross_sections=[water],
    mu_values=mu_values,
    I_scheme="linear_flux_only",
    tau_scheme="anchored",
)

continuum_solution = Korg.synthesize(
    atm,
    eltype(linelist)[],
    A_X,
    synthesis_wavelengths;
    vmic=vmic,
    hydrogen_lines=false,
    use_MHD_for_hydrogen_lines=false,
    mu_values=mu_values,
    I_scheme="linear_flux_only",
    tau_scheme="anchored",
    use_chemical_equilibrium_from=solution,
)

mu_grid = [item[1] for item in solution.mu_grid]
mu_weights = [item[2] for item in solution.mu_grid]
n_mu = length(mu_grid)
n_pix = length(apogee_wavelengths)

intensity = Matrix{Float64}(undef, n_mu, n_pix)
continuum_intensity = Matrix{Float64}(undef, n_mu, n_pix)
for i in 1:n_mu
    intensity[i, :] = lsf * vec(solution.intensity[i, :])
    continuum_intensity[i, :] = lsf * vec(continuum_solution.intensity[i, :])
end

flux = lsf * solution.flux
continuum_flux = lsf * solution.cntm

h5open(output_path, "w") do file
    file["wavelengths"] = apogee_wavelengths
    file["mu_values"] = mu_grid
    file["mu_weights"] = mu_weights
    file["intensity"] = intensity
    file["continuum_intensity"] = continuum_intensity
    file["flux"] = flux
    file["continuum_flux"] = continuum_flux
    attrs(file)["metadata"] = "{}"
end

println("Saved APOGEE synthesis to $(output_path)")
