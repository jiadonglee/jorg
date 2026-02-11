using Korg
using DelimitedFiles

if length(ARGS) < 7
    println("Usage: korg_compare_params.jl Teff logg m_H wl_min wl_max tag out_dir")
    exit(1)
end

Teff = parse(Float64, ARGS[1])
logg = parse(Float64, ARGS[2])
m_H = parse(Float64, ARGS[3])
wl_min = parse(Float64, ARGS[4])
wl_max = parse(Float64, ARGS[5])
tag = ARGS[6]
out_dir = ARGS[7]

jorg_data = get(ENV, "JORG_DATA_DIR", "")
if isempty(jorg_data)
    error("JORG_DATA_DIR is not set. Point it to the Jorg data directory.")
end

linelist_path = joinpath(jorg_data, "vald_extract_stellar_solar_threshold001.vald")
if !isfile(linelist_path)
    error("VALD linelist not found at $(linelist_path)")
end

mkpath(out_dir)

linelist = Korg.read_linelist(linelist_path; format="vald")
A_X = Korg.format_A_X(m_H)
atm = Korg.interpolate_marcs(Teff, logg, m_H)

res_lines = Korg.synthesize(atm, linelist, A_X, (wl_min, wl_max))

wl = res_lines.wavelengths
flux = res_lines.flux
cntm = res_lines.cntm
rect = flux ./ cntm

lines_path = joinpath(out_dir, "korg_$(tag)_spectrum_with_lines.txt")
writedlm(lines_path, hcat(wl, flux, cntm, rect))

# Continuum-only synthesis (no lines, no hydrogen lines)
empty_linelist = empty(linelist)
res_cntm = Korg.synthesize(atm, empty_linelist, A_X, (wl_min, wl_max); hydrogen_lines=false)
wl_c = res_cntm.wavelengths
flux_c = res_cntm.flux
cntm_c = res_cntm.cntm
rect_c = flux_c ./ cntm_c
cntm_path = joinpath(out_dir, "korg_$(tag)_spectrum_continuum.txt")
writedlm(cntm_path, hcat(wl_c, flux_c, cntm_c, rect_c))

println("Wrote:")
println("  ", lines_path)
println("  ", cntm_path)
