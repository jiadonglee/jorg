using Korg

# This script inspects the fields of a PlanarAtmosphereLayer object.

# 1. Create a dummy atmosphere to get a layer object.
Teff = 5777.0
logg = 4.44
A_X = Korg.format_A_X(0.0)
atm = Korg.interpolate_marcs(Teff, logg, A_X)

# 2. Get the first layer from the atmosphere.
layer = atm.layers[1]

# 3. Print the field names of the layer object.
println("Fields for PlanarAtmosphereLayer:")
println(fieldnames(typeof(layer)))
