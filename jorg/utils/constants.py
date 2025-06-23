"""
Physical constants for stellar spectroscopy calculations
"""

import numpy as np

# Fundamental physical constants (CGS units)
SPEED_OF_LIGHT = 2.99792458e10  # cm/s
PLANCK_H = 6.62607015e-27       # erg·s
BOLTZMANN_K = 1.380649e-16      # erg/K
ELEMENTARY_CHARGE = 4.80320425e-10  # esu (CGS)
ELECTRON_MASS = 9.1093837015e-28    # g
PROTON_MASS = 1.67262192369e-24     # g
ATOMIC_MASS_UNIT = 1.66053906660e-24  # g
AVOGADRO = 6.02214076e23        # mol^-1
VACUUM_PERMEABILITY = 4e-7 * np.pi  # Henry/m (SI), converted as needed
PI = np.pi

# Derived constants
FINE_STRUCTURE = ELEMENTARY_CHARGE**2 / (PLANCK_H * SPEED_OF_LIGHT)  # Fine structure constant
BOHR_RADIUS = PLANCK_H**2 / (4 * PI**2 * ELECTRON_MASS * ELEMENTARY_CHARGE**2)  # cm
RYDBERG = ELECTRON_MASS * ELEMENTARY_CHARGE**4 / (2 * PLANCK_H**2)  # erg

# Conversion factors
EV_TO_ERG = 1.602176634e-12  # eV to erg
ANGSTROM_TO_CM = 1e-8        # Angstrom to cm
CM_TO_ANGSTROM = 1e8         # cm to Angstrom

# Astronomical constants
SOLAR_RADIUS = 6.96e10      # cm
SOLAR_MASS = 1.989e33       # g
SOLAR_LUMINOSITY = 3.828e33 # erg/s
AU = 1.496e13               # cm (astronomical unit)
PARSEC = 3.086e18           # cm

# Spectroscopic constants
DOPPLER_CONSTANT = SPEED_OF_LIGHT / np.sqrt(2 * BOLTZMANN_K)  # For thermal Doppler broadening