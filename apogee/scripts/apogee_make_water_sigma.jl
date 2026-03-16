using Pkg

if length(ARGS) != 5
    error("Usage: apogee_make_water_sigma.jl <korg_root> <output_path> <start_A> <stop_A> <step_A>")
end

korg_root = ARGS[1]
output_path = ARGS[2]
start_A = parse(Float64, ARGS[3])
stop_A = parse(Float64, ARGS[4])
step_A = parse(Float64, ARGS[5])

Pkg.activate(korg_root)

using HDF5
using Korg

waterfile = joinpath(korg_root, "data", "linelists", "APOGEE_DR17", "pokazatel_water_lines.h5")
water_lines = h5open(waterfile, "r") do file
    Korg.Line.(
        Float64.(read(file["wl"])),
        Float64.(read(file["log_gf"])),
        Ref(Korg.Species("H2O")),
        Float64.(read(file["E_lower"])),
        Float64.(read(file["gamma_rad"])),
    )
end

wavelengths = start_A:step_A:stop_A
sigma = Korg.MolecularCrossSection(water_lines, wavelengths)
Korg.save_molecular_cross_section(output_path, sigma)
println("Saved APOGEE H2O molecular cross-section to $(output_path)")
