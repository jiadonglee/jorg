"""
Physical constants used in stellar spectral synthesis

Constants are defined in CGS units to match Korg.jl conventions
"""

import jax.numpy as jnp

# Fundamental constants (CGS units)
c_cgs = 2.99792458e10  # cm s^-1, speed of light
hplanck_cgs = 6.6260755e-27  # erg s, Planck constant
kboltz_cgs = 1.380658e-16  # erg K^-1, Boltzmann constant
me_cgs = 9.1093897e-28  # g, electron mass
mp_cgs = 1.6726231e-24  # g, proton mass
e_cgs = 4.8032068e-10  # statcoulomb, elementary charge
sigma_thomson = 6.6524e-25  # cm^2, Thomson scattering cross section

# Derived constants
hplanck_eV = hplanck_cgs / 1.602176634e-12  # eV s
kboltz_eV = kboltz_cgs / 1.602176634e-12  # eV K^-1
RydbergH_eV = 13.605693009  # eV, Rydberg constant for hydrogen

# Atomic data
ionization_energies = {
    1: [13.598434005136, 0.0],  # H I, H II (eV)
    2: [24.587387936, 54.417763],  # He I, He II (eV)
}

# Mathematical constants
pi = jnp.pi

# Useful factors
bf_sigma_const = 2.815e29  # constant for hydrogenic bf cross sections
H_minus_ion_energy = 0.754204  # eV, H^- ionization energy