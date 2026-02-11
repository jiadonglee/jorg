
using Korg
using Interpolations: linear_interpolation
using Korg.ContinuumAbsorption: total_continuum_absorption
using Korg.RadiativeTransfer
using Printf

function run_korg_pipeline(Teff, logg, m_H; wl_range=(5000.0, 5020.0), line_buffer=10.0, cntm_step=1.0,
                           hydrogen_lines=false, mu_values=20, line_cutoff_threshold=3e-4, vmic=1.0)
    timings = Dict{String,Float64}()

    A_X = Korg.format_A_X(m_H)
    t_atm = @elapsed atm = Korg.interpolate_marcs(Teff, logg, A_X; clamp_abundances=true)
    timings["atmosphere"] = t_atm

    wls = Korg.Wavelengths(wl_range)
    linelist = Korg.get_VALD_solar_linelist()
    if !issorted(linelist; by=l -> l.wl)
        linelist = sort(linelist; by=l -> l.wl)
    end

    cntm_step_cm = cntm_step * 1e-8
    line_buffer_cm = line_buffer * 1e-8

    cntm_windows = map(Korg.eachwindow(wls)) do (λstart, λstop)
        (λstart - line_buffer_cm - cntm_step_cm, λstop + line_buffer_cm + cntm_step_cm)
    end
    cntm_windows, _ = Korg.merge_bounds(cntm_windows)
    cntm_wls = Korg.Wavelengths([w[1]:cntm_step_cm:w[2] for w in cntm_windows])

    linelist5 = Korg.get_reference_wavelength_linelist(linelist, atm.reference_wavelength; use_internal_reference_linelist=true)
    linelist = Korg.filter_linelist(linelist, wls, line_buffer_cm)

    abs_abundances = 10 .^(A_X .- 12)
    abs_abundances ./= sum(abs_abundances)

    n_layers = length(atm.layers)
    n_wls = length(wls)
    α = Matrix{Float64}(undef, n_layers, n_wls)
    α_ref = Vector{Float64}(undef, n_layers)
    nₑs = Vector{Float64}(undef, n_layers)
    n_dicts = Vector{Dict}(undef, n_layers)
    α_cntm = Vector{Any}(undef, n_layers)

    t_ce = @elapsed begin
        for (i, layer) in enumerate(atm.layers)
            nₑ, n_dict = Korg.chemical_equilibrium(layer.temp, layer.number_density,
                                                   layer.electron_number_density,
                                                   abs_abundances,
                                                   Korg.ionization_energies,
                                                   Korg.default_partition_funcs,
                                                   Korg.default_log_equilibrium_constants;
                                                   electron_number_density_warn_threshold=Inf)
            α_cntm_vals = reverse(total_continuum_absorption(Korg.eachfreq(cntm_wls), layer.temp, nₑ, n_dict,
                                                             Korg.default_partition_funcs))
            α_cntm_layer = linear_interpolation(cntm_wls, α_cntm_vals)
            α[i, :] .= α_cntm_layer(wls)
            α_ref[i] = total_continuum_absorption([Korg.c_cgs / atm.reference_wavelength], layer.temp,
                                                  nₑ, n_dict, Korg.default_partition_funcs)[1]
            nₑs[i] = nₑ
            n_dicts[i] = n_dict
            α_cntm[i] = α_cntm_layer
        end
    end
    timings["chem_eq_continuum"] = t_ce

    number_densities = Dict([spec => [n[spec] for n in n_dicts]
                             for spec in keys(n_dicts[1])
                             if spec != Korg.species"H III"])

    α_ref_mat = reshape(α_ref, :, 1)
    α_cntm_ref = [_ -> a for a in copy(α_ref)]
    Korg.line_absorption!(α_ref_mat,
                          linelist5,
                          Korg.Wavelengths([atm.reference_wavelength * 1e8]),
                          Korg.get_temps(atm),
                          nₑs,
                          number_densities,
                          Korg.default_partition_funcs,
                          vmic * 1e5,
                          α_cntm_ref;
                          cutoff_threshold=line_cutoff_threshold)
    Korg.interpolate_molecular_cross_sections!(α_ref_mat,
                                                Korg.MolecularCrossSection[],
                                                Korg.Wavelengths([atm.reference_wavelength * 1e8]),
                                                Korg.get_temps(atm),
                                                vmic,
                                                number_densities)

    t_line = @elapsed begin
        if hydrogen_lines
            for (i, (layer, n_dict, nₑ)) in enumerate(zip(atm.layers, n_dicts, nₑs))
                nH_I = n_dict[Korg.species"H I"]
                nHe_I = n_dict[Korg.species"He I"]
                U_H_I = Korg.default_partition_funcs[Korg.species"H I"](log(layer.temp))
                ξ = (isa(vmic, Number) ? vmic : vmic[i]) * 1e5
                Korg.hydrogen_line_absorption!(view(α, i, :), wls, layer.temp, nₑ, nH_I, nHe_I,
                                               U_H_I, ξ, 150 * 1e-8;
                                               use_MHD=true)
            end
        end
        Korg.line_absorption!(α, linelist, wls, Korg.get_temps(atm), nₑs, number_densities,
                              Korg.default_partition_funcs, vmic * 1e5, α_cntm;
                              cutoff_threshold=line_cutoff_threshold)
        Korg.interpolate_molecular_cross_sections!(α, Korg.MolecularCrossSection[], wls,
                                                    Korg.get_temps(atm), vmic, number_densities)
    end
    timings["line_opacity"] = t_line

    source_fn = Korg.blackbody.((l -> l.temp).(atm.layers), wls')
    t_rt = @elapsed begin
        flux, intensity, μ_grid, μ_weights = Korg.RadiativeTransfer.radiative_transfer(
            atm, α, source_fn, mu_values; α_ref=α_ref, I_scheme="linear_flux_only", τ_scheme="anchored")
    end
    timings["radiative_transfer"] = t_rt

    return timings
end

BENCH_WL_RANGE = (5000.0, 5020.0)
PIXEL_LIST = [200, 1000, 2001]
SPECTRA_LIST = [1, 3]

PARAM_GRID = [
    (Teff=5771.0, logg=4.44, m_H=0.0),
    (Teff=4250.0, logg=1.40, m_H=-0.5),
    (Teff=4500.0, logg=1.50, m_H=-2.5),
    (Teff=5000.0, logg=4.00, m_H=-0.5),
    (Teff=6000.0, logg=4.50, m_H=0.2),
]

function build_params(n)
    params = []
    for i in 1:n
        push!(params, PARAM_GRID[(i - 1) % length(PARAM_GRID) + 1])
    end
    return params
end

wl_warm = range(BENCH_WL_RANGE[1], BENCH_WL_RANGE[2]; length=200)
_ = run_korg_pipeline(PARAM_GRID[1].Teff, PARAM_GRID[1].logg, PARAM_GRID[1].m_H;
                      wl_range=(first(wl_warm), last(wl_warm)))

println("
Korg per-step timing (seconds):")
println(rpad("n_pix",6), rpad("n_spec",8), rpad("pix_total",12), rpad("atm",10),
        rpad("ce+cntm",12), rpad("line",10), rpad("rt",10), rpad("total",10), rpad("pix/s",10))
println(repeat("-", 88))

for n_pix in PIXEL_LIST
    wl_array = range(BENCH_WL_RANGE[1], BENCH_WL_RANGE[2]; length=n_pix)
    for n_spec in SPECTRA_LIST
        params_list = build_params(n_spec)
        totals = Dict("atmosphere"=>0.0, "chem_eq_continuum"=>0.0, "line_opacity"=>0.0, "radiative_transfer"=>0.0)
        for p in params_list
            times = run_korg_pipeline(p.Teff, p.logg, p.m_H; wl_range=(first(wl_array), last(wl_array)))
            for (k, v) in times
                totals[k] += v
            end
        end
        total_time = sum(values(totals))
        pix_total = n_pix * n_spec
        pix_per_sec = pix_total / max(total_time, 1e-12)
        println(rpad(string(n_pix),6), rpad(string(n_spec),8), rpad(string(pix_total),12),
                rpad(@sprintf("%.2f", totals["atmosphere"]),10),
                rpad(@sprintf("%.2f", totals["chem_eq_continuum"]),12),
                rpad(@sprintf("%.2f", totals["line_opacity"]),10),
                rpad(@sprintf("%.2f", totals["radiative_transfer"]),10),
                rpad(@sprintf("%.2f", total_time),10),
                rpad(@sprintf("%.0f", pix_per_sec),10))
    end
end
