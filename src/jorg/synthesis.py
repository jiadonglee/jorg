"""
High-level stellar spectral synthesis interface for Jorg

This module provides the main user-facing API for stellar spectral synthesis,
similar to Korg.jl's synth() and synthesize() functions.
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .continuum.core import total_continuum_absorption
from .lines.core import total_line_absorption
from .constants import SPEED_OF_LIGHT, PLANCK_H, BOLTZMANN_K

# Export main synthesis functions
__all__ = ['synth', 'synthesize', 'SynthesisResult']


class SynthesisResult:
    """
    Container for detailed synthesis results
    
    Similar to Korg.jl's SynthesisResult structure, containing not just
    the final spectrum but also intermediate quantities for diagnostics.
    """
    
    def __init__(self, wavelengths: jnp.ndarray, flux: jnp.ndarray, 
                 continuum: jnp.ndarray, **kwargs):
        self.wavelengths = wavelengths
        self.flux = flux  
        self.continuum = continuum
        
        # Store any additional diagnostic information
        for key, value in kwargs.items():
            setattr(self, key, value)


def synth(wavelengths: Union[np.ndarray, jnp.ndarray],
          temperature: float,
          log_g: float, 
          metallicity: float,
          abundances: Optional[Dict[int, float]] = None,
          linelist: Optional[List] = None,
          microturbulence: float = 0.0,
          **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    High-level stellar spectrum synthesis function
    
    This is the main user-facing function for stellar spectrum synthesis,
    similar to Korg.jl's synth() function. It returns wavelengths, flux,
    and continuum as a simple tuple.
    
    Parameters
    ----------
    wavelengths : array
        Wavelength grid in Angstroms
    temperature : float
        Effective temperature in K
    log_g : float
        Surface gravity (log10(g) where g is in cm/s²)
    metallicity : float
        Metallicity [M/H] in dex
    abundances : dict, optional
        Element abundances {element_id: abundance}
    linelist : list, optional
        List of spectral lines to include
    microturbulence : float
        Microturbulent velocity in km/s
    **kwargs
        Additional parameters passed to lower-level functions
        
    Returns
    -------
    wavelengths : array
        Wavelength grid in Angstroms  
    flux : array
        Synthetic flux (normalized continuum)
    continuum : array
        Continuum flux
    """
    # Convert to JAX arrays
    wl = jnp.asarray(wavelengths)
    
    # For now, implement simplified synthesis
    # In full implementation, this would:
    # 1. Interpolate or load stellar atmosphere model
    # 2. Calculate continuum opacity at each atmospheric layer
    # 3. Calculate line opacity
    # 4. Solve radiative transfer equation
    # 5. Return emergent spectrum
    
    # Placeholder implementation
    continuum_flux = jnp.ones_like(wl)
    
    if linelist is not None and len(linelist) > 0:
        # Calculate line absorption
        line_opacity = total_line_absorption(
            wl, linelist, temperature, log_g, abundances,
            microturbulence=microturbulence, **kwargs
        )
        flux = continuum_flux * jnp.exp(-line_opacity)
    else:
        flux = continuum_flux
    
    return wl, flux, continuum_flux


def synthesize(wavelengths: Union[np.ndarray, jnp.ndarray],
               temperature: float,
               log_g: float,
               metallicity: float,
               abundances: Optional[Dict[int, float]] = None,
               linelist: Optional[List] = None,
               microturbulence: float = 0.0,
               return_diagnostics: bool = True,
               **kwargs) -> SynthesisResult:
    """
    Detailed stellar spectrum synthesis function
    
    Similar to Korg.jl's synthesize() function, this returns a detailed
    SynthesisResult object containing the spectrum and diagnostic information.
    
    Parameters
    ----------
    wavelengths : array
        Wavelength grid in Angstroms
    temperature : float
        Effective temperature in K
    log_g : float
        Surface gravity (log10(g) where g is in cm/s²)
    metallicity : float
        Metallicity [M/H] in dex
    abundances : dict, optional
        Element abundances {element_id: abundance}
    linelist : list, optional
        List of spectral lines to include
    microturbulence : float
        Microturbulent velocity in km/s
    return_diagnostics : bool
        Whether to return detailed diagnostic information
    **kwargs
        Additional parameters
        
    Returns
    -------
    SynthesisResult
        Object containing wavelengths, flux, continuum, and diagnostics
    """
    # Call main synthesis function
    wl, flux, continuum = synth(
        wavelengths, temperature, log_g, metallicity,
        abundances=abundances, linelist=linelist,
        microturbulence=microturbulence, **kwargs
    )
    
    # Prepare diagnostic information
    diagnostics = {}
    if return_diagnostics:
        diagnostics['temperature'] = temperature
        diagnostics['log_g'] = log_g
        diagnostics['metallicity'] = metallicity
        diagnostics['microturbulence'] = microturbulence
    
    return SynthesisResult(wl, flux, continuum, **diagnostics)


# Vectorized synthesis for batch processing
@jax.jit
def batch_synth(wavelengths: jnp.ndarray,
                temperatures: jnp.ndarray,
                log_gs: jnp.ndarray, 
                metallicities: jnp.ndarray,
                **kwargs) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Vectorized synthesis for multiple stellar parameters
    
    This function leverages JAX's vectorization capabilities to efficiently
    compute spectra for multiple sets of stellar parameters.
    
    Parameters
    ----------
    wavelengths : array
        Wavelength grid in Angstroms
    temperatures : array
        Array of effective temperatures in K
    log_gs : array
        Array of surface gravities
    metallicities : array
        Array of metallicities [M/H]
    **kwargs
        Additional parameters
        
    Returns
    -------
    wavelengths : array
        Wavelength grid
    fluxes : array
        Array of synthetic fluxes, shape (n_stars, n_wavelengths)
    """
    # Vectorize the synthesis function
    vectorized_synth = jax.vmap(
        lambda T, logg, mh: synth(wavelengths, T, logg, mh, **kwargs)[1],
        in_axes=(0, 0, 0)
    )
    
    fluxes = vectorized_synth(temperatures, log_gs, metallicities)
    
    return wavelengths, fluxes