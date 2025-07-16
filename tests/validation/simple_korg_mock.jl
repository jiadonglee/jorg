
# Simple Korg test
using Printf

println("🌟 Simple Korg Test")
println("=" ^ 30)

# Basic parameter test 
Teff = 5777
logg = 4.44
m_H = 0.0
wl_range = (5000, 5030)

println("Solar case:")
println("  Teff=$(Teff)K, logg=$(logg), [M/H]=$(m_H)")  
println("  Wavelengths: $(wl_range[1])-$(wl_range[2]) Å")

# Mock result for comparison
n_points = 60
wavelengths = collect(range(wl_range[1], wl_range[2], length=n_points))
flux_mock = ones(n_points) .* (0.95 .+ 0.05 .* sin.(2π .* (wavelengths .- 5000) ./ 10))
continuum_mock = ones(n_points) .* 3.2e13

println("")
println("✅ Mock Korg Results:")
println("  Wavelengths: $(length(wavelengths)) points")
println("  Flux range: $(minimum(flux_mock):.3f) - $(maximum(flux_mock):.3f)")
println("  Flux mean: $(sum(flux_mock)/length(flux_mock):.3f)")
println("  Continuum mean: $(sum(continuum_mock)/length(continuum_mock):.2e)")

println("")
println("Note: This is a mock comparison due to Korg setup complexity")
println("Real comparison would require full Korg.jl installation")
