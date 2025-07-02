"""
Physical constants for stellar spectroscopy calculations
"""

import numpy as np

# Physical constants matching Korg.jl exactly for strict compatibility
# All values taken directly from Korg.jl src/constants.jl

# Primary constants (CGS units) - exact values from Korg.jl
kboltz_cgs = 1.380649e-16       # erg/K
hplanck_cgs = 6.62607015e-27    # erg*s
c_cgs = 2.99792458e10           # cm/s
electron_mass_cgs = 9.1093897e-28  # g
electron_charge_cgs = 4.80320425e-10  # statcoulomb or cm^3/2 * g^1/2 / s
amu_cgs = 1.6605402e-24         # g
bohr_radius_cgs = 5.29177210903e-9  # cm 2018 CODATA recommended value

# Astronomical constants
solar_mass_cgs = 1.9884e33      # g
G_cgs = 6.67430e-8             # cm^3 g^-1 s^-2

# Energy conversion
eV_to_cgs = 1.602e-12          # ergs per eV (exact from Korg.jl)
EV_TO_ERG = eV_to_cgs          # Alias for backward compatibility

# Energy units constants
kboltz_eV = 8.617333262145e-5   # eV/K (exact from Korg.jl)
hplanck_eV = 4.135667696e-15    # eV*s (exact from Korg.jl)
RydbergH_eV = 13.598287264      # eV (exact from Korg.jl)
Rydberg_eV = 13.605693122994    # eV 2018 CODATA via wikipedia

# Legacy aliases for backward compatibility
SPEED_OF_LIGHT = c_cgs
PLANCK_H = hplanck_cgs
BOLTZMANN_K = kboltz_cgs
ELEMENTARY_CHARGE = electron_charge_cgs
ELECTRON_MASS = electron_mass_cgs
ATOMIC_MASS_UNIT = amu_cgs
PI = np.pi

# Conversion factors (removed duplicate)
ANGSTROM_TO_CM = 1e-8
CM_TO_ANGSTROM = 1e8

# Additional astronomical constants (not in Korg.jl but useful)
SOLAR_RADIUS = 6.96e10      # cm
SOLAR_LUMINOSITY = 3.828e33 # erg/s
AU = 1.496e13               # cm (astronomical unit)
PARSEC = 3.086e18           # cm

# Spectroscopic constants
THOMSON_SCATTERING = 6.6524e-25  # cm^2, Thomson scattering cross section
sigma_thomson = THOMSON_SCATTERING

# Commonly used derived constants
pi = PI
me_cgs = electron_mass_cgs
e_cgs = electron_charge_cgs

# Derived constants for backward compatibility
RYDBERG = electron_mass_cgs * electron_charge_cgs**4 / (2 * hplanck_cgs**2)  # erg
FINE_STRUCTURE = electron_charge_cgs**2 / (hplanck_cgs * c_cgs)  # Fine structure constant
BOHR_RADIUS = bohr_radius_cgs  # cm

# Legacy constants for existing code compatibility
PROTON_MASS = 1.67262192369e-24  # g
AVOGADRO = 6.02214076e23  # mol^-1
VACUUM_PERMEABILITY = 4e-7 * np.pi  # Henry/m (SI), kept for compatibility

# Useful factors for opacity calculations
bf_sigma_const = 2.815e29  # constant for hydrogenic bf cross sections
H_minus_ion_energy = 0.754204  # eV, H^- ionization energy

# Atomic data
ionization_energies = {
    1: [13.598434005136, 0.0],  # H I, H II (eV)
    2: [24.587387936, 54.417763],  # He I, He II (eV)
}