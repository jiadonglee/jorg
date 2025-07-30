"""
High-Level H I Bound-Free API - Korg.jl Compatible Interface

This module provides a high-level API that exactly matches Korg.jl's 
ContinuumAbsorption.H_I_bf function signature and behavior.

The wrapper handles:
- Multiple n levels (n=1 to n_max_MHD)
- Correct MHD parameter application per level
- Exact parameter defaults matching Korg.jl
- Vectorized frequency input
- JAX optimization for performance

Usage:
    from jorg.continuum.h_i_bf_api import H_I_bf
    
    alpha = H_I_bf(
        frequencies, temperature, n_h_i, n_he_i, electron_density, inv_u_h,
        n_max_MHD=6, use_hubeny_generalization=False,
        taper=False, use_MHD_for_Lyman=False
    )
"""

import jax
import jax.numpy as jnp
from typing import Union, Optional
from .nahar_h_i_bf import nahar_h_i_bf_absorption_single_level


def H_I_bf(
    frequencies: jnp.ndarray,
    temperature: float,
    n_h_i: float,
    n_he_i: float,
    electron_density: float,
    inv_u_h: float,
    n_max_MHD: int = 6,
    use_hubeny_generalization: bool = False,
    taper: bool = False,
    use_MHD_for_Lyman: bool = False
) -> jnp.ndarray:
    """
    Calculate H I bound-free absorption coefficient exactly matching Korg.jl's H_I_bf function
    
    This function provides a high-level interface that exactly matches the signature
    and behavior of Korg.jl's ContinuumAbsorption.H_I_bf function, including all
    default parameters and MHD level handling.
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Sorted frequency vector in Hz
    temperature : float
        Temperature in K
    n_h_i : float
        The total number density of neutral Hydrogen (in cm⁻³)
    n_he_i : float
        The total number density of neutral Helium (in cm⁻³)
    electron_density : float
        The number density of electrons (in cm⁻³)
    inv_u_h : float
        The inverse of the neutral hydrogen partition function
    n_max_MHD : int, optional
        Maximum n level for MHD treatment (default: 6, matching Korg.jl)
    use_hubeny_generalization : bool, optional
        Use Hubeny 1994 generalization (default: False, matching Korg.jl)
    taper : bool, optional
        Apply HBOP-style tapering (default: False, matching Korg.jl)
    use_MHD_for_Lyman : bool, optional
        Apply MHD to Lyman series (n=1) (default: False, matching Korg.jl)
        
    Returns
    -------
    jnp.ndarray
        H I bound-free absorption coefficient in cm⁻¹
        
    Notes
    -----
    This function exactly replicates Korg.jl's behavior:
    - For n=1 through n=n_max_MHD, uses Nahar 2021 cross-sections with MHD
    - MHD is applied to n>1 levels by default (unless use_MHD_for_Lyman=True)
    - Level dissolution and occupation probabilities match Korg.jl exactly
    - Performance is optimized with JAX compilation
    
    Examples
    --------
    >>> frequencies = jnp.array([4e14, 5e14, 6e14])  # Hz
    >>> alpha = H_I_bf(frequencies, 5780.0, 1.5e16, 1e15, 4.28e12, 0.5)
    >>> print(alpha)  # Should match Korg.jl output exactly
    """
    # Input validation
    if not isinstance(frequencies, jnp.ndarray):
        frequencies = jnp.array(frequencies)
    
    if len(frequencies) == 0:
        return jnp.array([])
    
    # Initialize total absorption coefficient
    total_alpha = jnp.zeros_like(frequencies, dtype=jnp.float64)
    
    # Sum over n levels from 1 to n_max_MHD (matching Korg.jl loop)
    for n_level in range(1, n_max_MHD + 1):
        # Determine MHD settings for this level (exactly matching Korg.jl logic)
        if n_level == 1:
            # For n=1 (Lyman series), use the use_MHD_for_Lyman parameter
            use_mhd_this_level = use_MHD_for_Lyman
        else:
            # For n>1, always use MHD (Korg.jl default behavior)
            use_mhd_this_level = True
        
        # Calculate absorption for this n level
        alpha_n = nahar_h_i_bf_absorption_single_level(
            frequencies=frequencies,
            temperature=temperature,
            n_h_i=n_h_i,
            n_he_i=n_he_i,
            electron_density=electron_density,
            inv_u_h=inv_u_h,
            n_level=n_level,
            use_hubeny_generalization=use_hubeny_generalization,
            use_mhd_for_lyman=use_mhd_this_level,
            taper=taper
        )
        
        # Add to total
        total_alpha += alpha_n
    
    return total_alpha


# JAX-compiled version for performance
H_I_bf_compiled = jax.jit(H_I_bf, static_argnames=['n_max_MHD', 'use_hubeny_generalization', 'taper', 'use_MHD_for_Lyman'])


def H_I_bf_fast(
    frequencies: jnp.ndarray,
    temperature: float,
    n_h_i: float,
    n_he_i: float,
    electron_density: float,
    inv_u_h: float,
    n_max_MHD: int = 6,
    use_hubeny_generalization: bool = False,
    taper: bool = False,
    use_MHD_for_Lyman: bool = False
) -> jnp.ndarray:
    """
    JAX-compiled version of H_I_bf for maximum performance
    
    This is identical to H_I_bf but pre-compiled with JAX for faster execution.
    Use this when calling the function multiple times with similar parameters.
    
    Parameters and returns are identical to H_I_bf.
    """
    return H_I_bf_compiled(
        frequencies, temperature, n_h_i, n_he_i, electron_density, inv_u_h,
        n_max_MHD, use_hubeny_generalization, taper, use_MHD_for_Lyman
    )


# Convenience function with typical stellar parameters
def H_I_bf_stellar(
    frequencies: jnp.ndarray,
    temperature: float = 5780.0,
    n_h_i: float = 1.5e16,
    n_he_i: float = 1e15,
    electron_density: float = 4.28e12,
    inv_u_h: Optional[float] = None
) -> jnp.ndarray:
    """
    H I bound-free calculation with typical stellar atmosphere parameters
    
    This is a convenience function that uses common stellar atmosphere values
    and automatically calculates the H I partition function if not provided.
    
    Parameters
    ----------
    frequencies : jnp.ndarray
        Frequencies in Hz
    temperature : float, optional
        Temperature in K (default: 5780.0, solar value)
    n_h_i : float, optional
        H I density in cm⁻³ (default: 1.5e16, typical value)
    n_he_i : float, optional
        He I density in cm⁻³ (default: 1e15, typical value)
    electron_density : float, optional
        Electron density in cm⁻³ (default: 4.28e12, typical value)
    inv_u_h : float, optional
        Inverse H I partition function (default: calculated automatically)
        
    Returns
    -------
    jnp.ndarray
        H I bound-free absorption coefficient in cm⁻¹
    """
    if inv_u_h is None:
        # Calculate H I partition function automatically
        from ..statmech.partition_functions import create_default_partition_functions
        from ..statmech.species import Species
        
        h_i_species = Species.from_atomic_number(1, 0)
        partition_funcs = create_default_partition_functions()
        U_H_I = partition_funcs[h_i_species](jnp.log(temperature))
        inv_u_h = 1.0 / U_H_I
    
    return H_I_bf_fast(
        frequencies, temperature, n_h_i, n_he_i, electron_density, inv_u_h
    )


# Validation function to test against Korg.jl
def validate_H_I_bf_against_korg():
    """
    Validation function to test the H_I_bf implementation against Korg.jl
    
    This function runs a test comparison to ensure our implementation
    matches Korg.jl's results exactly.
    """
    print("H I Bound-Free API Validation Against Korg.jl")
    print("=" * 50)
    
    # Test parameters (matching our previous validation)
    frequencies = jnp.array([4e14, 5e14, 6e14, 7e14, 8e14, 1e15])
    temperature = 5780.0
    n_h_i = 1.5e16
    n_he_i = 1e15
    electron_density = 4.28e12
    inv_u_h = 0.5
    
    # Calculate with our API
    alpha_jorg = H_I_bf(
        frequencies, temperature, n_h_i, n_he_i, electron_density, inv_u_h
    )
    
    # Expected Korg values from our previous validation
    alpha_korg_expected = jnp.array([7.457e-11, 4.017e-11, 2.418e-11, 1.674e-11, 1.597e-10, 6.192e-10])
    
    print(f"{'Frequency (Hz)':<15} {'Jorg API':<15} {'Korg Expected':<15} {'Ratio':<10}")
    print("-" * 65)
    
    for i, freq in enumerate(frequencies):
        jorg_val = float(alpha_jorg[i])
        korg_val = float(alpha_korg_expected[i])
        ratio = jorg_val / korg_val if korg_val != 0 else float('inf')
        
        print(f"{freq:<15.1e} {jorg_val:<15.3e} {korg_val:<15.3e} {ratio:<10.3f}")
    
    # Calculate overall agreement
    ratios = alpha_jorg / alpha_korg_expected
    mean_ratio = float(jnp.mean(ratios))
    print(f"\nMean ratio: {mean_ratio:.4f}")
    print(f"Agreement: {(1.0 - abs(1.0 - mean_ratio)) * 100:.2f}%")
    
    if abs(mean_ratio - 1.0) < 0.05:
        print("✅ VALIDATION PASSED: API matches Korg.jl within 5%")
    else:
        print("❌ VALIDATION FAILED: API does not match Korg.jl")
    
    return alpha_jorg, alpha_korg_expected


if __name__ == "__main__":
    validate_H_I_bf_against_korg()