using Korg
using DelimitedFiles
using Printf

function run_case(teff, logg, m_H, wl_min, wl_max, tag, out_dir)
    root = abspath(joinpath(@__DIR__, "../../.."))
    linelist_path = joinpath(root, "data/linelists/vald_extract_stellar_solar_threshold001.vald")
    linelist = read_linelist(linelist_path, format="vald")

    atm = interpolate_marcs(teff, logg, m_H)
    A_X = format_A_X(m_H)

    wl_range = (wl_min, wl_max)
    result_lines = synthesize(atm, linelist, A_X, wl_range; hydrogen_lines=false, verbose=false)
    result_cntm = synthesize(atm, [], A_X, wl_range; hydrogen_lines=false, verbose=false)

    wl = result_lines.wavelengths
    flux = result_lines.flux
    cntm = result_lines.cntm
    rect = flux ./ cntm

    wl_c = result_cntm.wavelengths
    flux_c = result_cntm.flux
    cntm_c = result_cntm.cntm
    rect_c = flux_c ./ cntm_c

    mkpath(out_dir)

    open(joinpath(out_dir, "korg_$(tag)_spectrum_with_lines.txt"), "w") do io
        for i in 1:length(wl)
            @printf(io, "%.6f %.12e %.12e %.12e\n", wl[i], flux[i], cntm[i], rect[i])
        end
    end

    open(joinpath(out_dir, "korg_$(tag)_spectrum_continuum.txt"), "w") do io
        for i in 1:length(wl_c)
            @printf(io, "%.6f %.12e %.12e %.12e\n", wl_c[i], flux_c[i], cntm_c[i], rect_c[i])
        end
    end

    writedlm(joinpath(out_dir, "korg_$(tag)_alpha_with_lines.txt"), result_lines.alpha)
    writedlm(joinpath(out_dir, "korg_$(tag)_alpha_continuum.txt"), result_cntm.alpha)
end

teff = parse(Float64, ARGS[1])
logg = parse(Float64, ARGS[2])
m_H = parse(Float64, ARGS[3])
wl_min = parse(Float64, ARGS[4])
wl_max = parse(Float64, ARGS[5])
tag = ARGS[6]
out_dir = ARGS[7]

run_case(teff, logg, m_H, wl_min, wl_max, tag, out_dir)
