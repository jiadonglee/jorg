"""
Peach1970 departure coefficient corrections for free-free absorption.

This module contains interpolators of the tabulated ff departure coefficients from
Peach+ 1970 (https://ui.adsabs.harvard.edu/abs/1970MmRAS..73....1P/abstract), which we use to
correct the hydrogenic ff absorption coefficient for He II ff, C II ff, Si II ff, and Mg II ff.

The free-free absorption coefficient (including stimulated emission) is given by:

α_ff = α_hydrogenic,ff(ν, T, n_i, n_e; Z) * (1 + D(T, σ))

where:
- α_hydrogenic,ff(ν, T, n_i, n_e; Z) should include the correction for stimulated emission
- n_i is the number density of the ion species that participates in the interaction
- n_e is the number density of free electrons  
- D(T, σ) is the departure coefficient from the tabulated values in Table III of Peach (1970)
- σ denotes the energy of the photon in units of RydbergH*Zeff²

Outside the regime in which Peach 1970 provides data, the interpolators return 0, falling back to
the hydrogenic approximation.
"""

import jax.numpy as jnp
from jax import jit
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from typing import Dict, Optional
# Species import not needed for this module


class Peach1970Interpolator:
    """Interpolator for Peach 1970 departure coefficients."""
    
    def __init__(self, T_vals: np.ndarray, sigma_vals: np.ndarray, table_vals: np.ndarray):
        """
        Initialize interpolator.
        
        Args:
            T_vals: Temperature values in K
            sigma_vals: Photon energy values in units of RydbergH*Zeff²
            table_vals: Departure coefficient values (unitless)
        """
        self.T_vals = T_vals
        self.sigma_vals = sigma_vals
        self.interpolator = RegularGridInterpolator(
            (T_vals, sigma_vals), 
            table_vals, 
            bounds_error=False, 
            fill_value=0.0
        )
    
    def __call__(self, T: float, sigma: float) -> float:
        """
        Evaluate departure coefficient at given temperature and photon energy.
        
        Args:
            T: Temperature in K
            sigma: Photon energy in units of RydbergH*Zeff²
            
        Returns:
            Departure coefficient (unitless)
        """
        # Ensure inputs are within reasonable bounds
        T = jnp.clip(T, self.T_vals.min(), self.T_vals.max())
        sigma = jnp.clip(sigma, self.sigma_vals.min(), self.sigma_vals.max())
        
        # Use scipy interpolator (will be converted to JAX-compatible form)
        return self.interpolator(jnp.array([T, sigma]))[0]


def _create_he_ii_interpolator() -> Peach1970Interpolator:
    """Create He II departure coefficient interpolator."""
    # From table III of Peach 1970 for neutral Helium
    
    # σ denotes the energy of the photon (in units of RydbergH*Zeff², where Zeff is the net charge
    # of the species participating in the interaction
    sigma_vals = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    
    # The temperature (in K)
    T_vals = np.array([10000.0, 11000.0, 12000.0, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0, 18000.0,
        19000.0, 20000.0, 21000.0, 22000.0, 23000.0, 24000.0, 25000.0, 26000.0, 27000.0,
        28000.0, 29000.0, 30000.0, 32000.0, 34000.0, 36000.0, 38000.0, 40000.0, 42000.0,
        44000.0, 46000.0, 48000.0])
    
    # the (unitless) departure term
    table_vals = np.array([
        [0.016, 0.039, 0.069, 0.100, 0.135, 0.169],
        [0.018, 0.041, 0.071, 0.103, 0.137, 0.172],
        [0.020, 0.043, 0.073, 0.105, 0.139, 0.174],
        [0.022, 0.045, 0.075, 0.107, 0.142, 0.176],
        [0.024, 0.047, 0.078, 0.109, 0.144, 0.179],
        [0.026, 0.050, 0.080, 0.112, 0.146, 0.181],
        [0.028, 0.052, 0.082, 0.114, 0.148, 0.183],
        [0.029, 0.054, 0.084, 0.116, 0.151, 0.185],
        [0.031, 0.056, 0.086, 0.118, 0.153, 0.187],
        [0.033, 0.058, 0.088, 0.120, 0.155, 0.190],
        [0.035, 0.060, 0.090, 0.122, 0.157, 0.192],
        [0.037, 0.062, 0.092, 0.125, 0.159, 0.194],
        [0.039, 0.064, 0.095, 0.127, 0.162, 0.196],
        [0.041, 0.066, 0.097, 0.129, 0.164, 0.198],
        [0.043, 0.068, 0.099, 0.131, 0.166, 0.201],
        [0.045, 0.070, 0.101, 0.133, 0.168, 0.203],
        [0.047, 0.072, 0.103, 0.135, 0.170, 0.205],
        [0.049, 0.074, 0.105, 0.138, 0.173, 0.207],
        [0.050, 0.076, 0.107, 0.140, 0.175, 0.209],
        [0.052, 0.079, 0.109, 0.142, 0.177, 0.211],
        [0.054, 0.081, 0.111, 0.144, 0.179, 0.214],
        [0.058, 0.085, 0.115, 0.148, 0.183, 0.218],
        [0.062, 0.089, 0.119, 0.153, 0.188, 0.222],
        [0.065, 0.093, 0.124, 0.157, 0.192, 0.226],
        [0.069, 0.096, 0.128, 0.161, 0.196, 0.230],
        [0.072, 0.100, 0.132, 0.165, 0.200, 0.235],
        [0.076, 0.104, 0.135, 0.169, 0.204, 0.239],
        [0.079, 0.108, 0.139, 0.173, 0.208, 0.243],
        [0.082, 0.111, 0.143, 0.177, 0.212, 0.247],
        [0.085, 0.115, 0.147, 0.181, 0.216, 0.251]
    ])
    
    return Peach1970Interpolator(T_vals, sigma_vals, table_vals)


def _create_c_ii_interpolator() -> Peach1970Interpolator:
    """Create C II departure coefficient interpolator."""
    # From table III of Peach 1970 for neutral Carbon
    
    sigma_vals = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    
    T_vals = np.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
        14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
        23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 32000.0,
        34000.0, 36000.0])
    
    table_vals = np.array([
        [-0.145, -0.144, -0.068, 0.054, 0.200, 0.394],
        [-0.132, -0.124, -0.045, 0.077, 0.222, 0.415],
        [-0.121, -0.109, -0.027, 0.097, 0.244, 0.438],
        [-0.112, -0.095, -0.010, 0.115, 0.264, 0.461],
        [-0.104, -0.082, 0.005, 0.133, 0.284, 0.484],
        [-0.095, -0.070, 0.020, 0.150, 0.303, 0.507],
        [-0.087, -0.058, 0.034, 0.166, 0.321, 0.529],
        [-0.079, -0.047, 0.048, 0.181, 0.339, 0.550],
        [-0.071, -0.036, 0.061, 0.196, 0.356, 0.570],
        [-0.063, -0.025, 0.074, 0.210, 0.372, 0.590],
        [-0.055, -0.015, 0.086, 0.223, 0.388, 0.609],
        [-0.047, -0.005, 0.098, 0.237, 0.403, 0.628],
        [-0.040, 0.005, 0.109, 0.249, 0.418, 0.646],
        [-0.032, 0.015, 0.120, 0.261, 0.432, 0.664],
        [-0.025, 0.024, 0.131, 0.273, 0.446, 0.680],
        [-0.017, 0.034, 0.141, 0.285, 0.459, 0.697],
        [-0.010, 0.043, 0.152, 0.296, 0.472, 0.713],
        [-0.003, 0.051, 0.161, 0.307, 0.485, 0.728],
        [0.004, 0.060, 0.171, 0.317, 0.497, 0.744],
        [0.011, 0.069, 0.181, 0.327, 0.509, 0.758],
        [0.018, 0.077, 0.190, 0.337, 0.521, 0.773],
        [0.025, 0.085, 0.199, 0.347, 0.532, 0.787],
        [0.032, 0.093, 0.208, 0.356, 0.543, 0.800],
        [0.039, 0.101, 0.216, 0.365, 0.554, 0.814],
        [0.046, 0.109, 0.225, 0.374, 0.564, 0.827],
        [0.052, 0.117, 0.233, 0.383, 0.574, 0.839],
        [0.059, 0.124, 0.241, 0.391, 0.585, 0.852],
        [0.072, 0.139, 0.257, 0.408, 0.604, 0.876],
        [0.085, 0.154, 0.273, 0.424, 0.623, 0.900],
        [0.097, 0.168, 0.288, 0.439, 0.641, 0.923]
    ])
    
    return Peach1970Interpolator(T_vals, sigma_vals, table_vals)


def _create_si_ii_interpolator() -> Peach1970Interpolator:
    """Create Si II departure coefficient interpolator."""
    # From table III of Peach 1970 for neutral Silicon
    
    sigma_vals = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    
    T_vals = np.array([4000.0, 5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 10000.0, 11000.0, 12000.0, 13000.0,
        14000.0, 15000.0, 16000.0, 17000.0, 18000.0, 19000.0, 20000.0, 21000.0, 22000.0,
        23000.0, 24000.0, 25000.0, 26000.0, 27000.0, 28000.0, 29000.0, 30000.0, 32000.0,
        34000.0, 36000.0])
    
    table_vals = np.array([
        [-0.079, 0.033, 0.214, 0.434, 0.650, 0.973],
        [-0.066, 0.042, 0.216, 0.429, 0.642, 0.962],
        [-0.056, 0.050, 0.220, 0.430, 0.643, 0.965],
        [-0.048, 0.057, 0.224, 0.433, 0.648, 0.974],
        [-0.040, 0.063, 0.229, 0.436, 0.653, 0.981],
        [-0.033, 0.069, 0.233, 0.440, 0.659, 0.995],
        [-0.027, 0.074, 0.238, 0.444, 0.666, 1.007],
        [-0.021, 0.080, 0.242, 0.448, 0.672, 1.019],
        [-0.015, 0.085, 0.246, 0.452, 0.679, 1.031],
        [-0.010, 0.089, 0.250, 0.456, 0.685, 1.042],
        [-0.004, 0.094, 0.254, 0.459, 0.692, 1.054],
        [0.001, 0.099, 0.258, 0.463, 0.698, 1.065],
        [0.006, 0.103, 0.262, 0.467, 0.705, 1.076],
        [0.011, 0.107, 0.265, 0.471, 0.711, 1.087],
        [0.016, 0.112, 0.269, 0.474, 0.717, 1.097],
        [0.021, 0.116, 0.273, 0.478, 0.724, 1.108],
        [0.026, 0.120, 0.277, 0.482, 0.730, 1.118],
        [0.030, 0.125, 0.281, 0.486, 0.736, 1.127],
        [0.035, 0.129, 0.285, 0.490, 0.742, 1.137],
        [0.040, 0.134, 0.289, 0.493, 0.747, 1.146],
        [0.045, 0.138, 0.293, 0.497, 0.753, 1.155],
        [0.050, 0.143, 0.297, 0.501, 0.759, 1.164],
        [0.055, 0.147, 0.301, 0.505, 0.765, 1.173],
        [0.060, 0.152, 0.305, 0.509, 0.770, 1.181],
        [0.065, 0.156, 0.310, 0.513, 0.776, 1.189],
        [0.071, 0.161, 0.314, 0.517, 0.781, 1.197],
        [0.076, 0.166, 0.318, 0.520, 0.787, 1.205],
        [0.087, 0.176, 0.328, 0.528, 0.798, 1.221],
        [0.098, 0.186, 0.337, 0.537, 0.809, 1.236],
        [0.109, 0.196, 0.346, 0.545, 0.819, 1.251]
    ])
    
    return Peach1970Interpolator(T_vals, sigma_vals, table_vals)


def _create_mg_ii_interpolator() -> Peach1970Interpolator:
    """Create Mg II departure coefficient interpolator."""
    # From table III of Peach 1970 for neutral Magnesium
    
    sigma_vals = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30])
    
    T_vals = np.array([4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
        17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000,
        29000, 30000, 32000, 34000])
    
    table_vals = np.array([
        [-0.070, 0.008, 0.121, 0.221, 0.274, 0.356],
        [-0.067, 0.003, 0.104, 0.195, 0.244, 0.325],
        [-0.066, -0.002, 0.091, 0.175, 0.221, 0.302],
        [-0.065, -0.007, 0.080, 0.157, 0.201, 0.282],
        [-0.065, -0.012, 0.069, 0.141, 0.183, 0.264],
        [-0.065, -0.016, 0.059, 0.126, 0.166, 0.248],
        [-0.065, -0.020, 0.049, 0.113, 0.151, 0.232],
        [-0.066, -0.024, 0.040, 0.100, 0.137, 0.218],
        [-0.066, -0.028, 0.032, 0.088, 0.124, 0.205],
        [-0.066, -0.032, 0.025, 0.077, 0.112, 0.194],
        [-0.066, -0.035, 0.018, 0.067, 0.101, 0.183],
        [-0.066, -0.037, 0.012, 0.058, 0.091, 0.173],
        [-0.066, -0.040, 0.006, 0.049, 0.082, 0.164],
        [-0.066, -0.042, 0.001, 0.042, 0.074, 0.157],
        [-0.066, -0.044, -0.004, 0.036, 0.067, 0.150],
        [-0.065, -0.045, -0.007, 0.030, 0.061, 0.144],
        [-0.064, -0.046, -0.011, 0.025, 0.056, 0.139],
        [-0.063, -0.047, -0.014, 0.020, 0.051, 0.135],
        [-0.062, -0.048, -0.016, 0.017, 0.048, 0.131],
        [-0.061, -0.048, -0.018, 0.014, 0.045, 0.128],
        [-0.059, -0.047, -0.019, 0.011, 0.042, 0.126],
        [-0.057, -0.047, -0.020, 0.009, 0.040, 0.124],
        [-0.055, -0.046, -0.020, 0.008, 0.039, 0.123],
        [-0.053, -0.045, -0.021, 0.007, 0.038, 0.123],
        [-0.051, -0.044, -0.020, 0.006, 0.038, 0.123],
        [-0.048, -0.042, -0.020, 0.006, 0.038, 0.123],
        [-0.045, -0.040, -0.019, 0.006, 0.039, 0.124],
        [-0.039, -0.035, -0.016, 0.008, 0.042, 0.128],
        [-0.032, -0.030, -0.012, 0.011, 0.046, 0.133]
    ])
    
    return Peach1970Interpolator(T_vals, sigma_vals, table_vals)


# Create global interpolators
_INTERPOLATORS: Dict[str, Peach1970Interpolator] = {
    'He II': _create_he_ii_interpolator(),
    'C II': _create_c_ii_interpolator(),
    'Si II': _create_si_ii_interpolator(),
    'Mg II': _create_mg_ii_interpolator()
}


def get_departure_coefficient(species: str, T: float, sigma: float) -> float:
    """
    Get departure coefficient for a given species.
    
    Args:
        species: Species name (e.g., 'He II', 'C II', 'Si II', 'Mg II')
        T: Temperature in K
        sigma: Photon energy in units of RydbergH*Zeff²
        
    Returns:
        Departure coefficient (unitless). Returns 0.0 if species not found.
    """
    if species in _INTERPOLATORS:
        return _INTERPOLATORS[species](T, sigma)
    else:
        return 0.0


def get_all_departure_coefficients() -> Dict[str, Peach1970Interpolator]:
    """
    Get all departure coefficient interpolators.
    
    Returns:
        Dictionary mapping species names to interpolators
    """
    return _INTERPOLATORS.copy()