"""
Line profile calculations for stellar spectral synthesis
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Union
from ..constants import c_cgs, hplanck_eV, pi


@jax.jit
def harris_series(v: float) -> jnp.ndarray:
    """
    Compute Harris series coefficients H₀, H₁, H₂ for Voigt profile calculation.
    
    This is the exact implementation from Korg.jl following Hunger 1965.
    Assumes v < 5.
    
    Parameters
    ----------
    v : float
        Dimensionless frequency parameter
        
    Returns
    -------
    jnp.ndarray
        Array [H0, H1, H2] of Harris series coefficients
    """
    v2 = v * v
    H0 = jnp.exp(-v2)
    
    # H1 calculation with piecewise polynomials exactly as in Korg.jl
    def h1_case1():  # v < 1.3
        return (-1.12470432 + (-0.15516677 + (3.288675912 + (-2.34357915 + 0.42139162 * v) * v) * v) * v)
    
    def h1_case2():  # 1.3 <= v < 2.4
        return (-4.48480194 + (9.39456063 + (-6.61487486 + (1.98919585 - 0.22041650 * v) * v) * v) * v)
    
    def h1_case3():  # 2.4 <= v < 5
        return ((0.554153432 + 
                (0.278711796 + (-0.1883256872 + (0.042991293 - 0.003278278 * v) * v) * v) * v) /
                (v2 - 1.5))
    
    H1 = jnp.where(
        v < 1.3,
        h1_case1(),
        jnp.where(
            v < 2.4,
            h1_case2(),
            h1_case3()
        )
    )
    
    # H2 calculation exactly as in Korg.jl
    H2 = (1.0 - 2.0 * v2) * H0
    
    return jnp.array([H0, H1, H2])


@jax.jit
def voigt_hjerting(alpha: float, v: float) -> float:
    """
    Compute the Hjerting function H(α, v) for Voigt profile calculation.
    
    This is the exact implementation from Korg.jl following Hunger 1965.
    Approximation from Hunger 1965: https://ui.adsabs.harvard.edu/abs/1956ZA.....39...36H/abstract
    
    Parameters
    ----------
    alpha : float
        Damping parameter (ratio of Lorentz to Doppler width)
    v : float
        Dimensionless frequency offset from line center
        
    Returns
    -------
    float
        Hjerting function value H(α, v)
    """
    v2 = v * v
    sqrt_pi = jnp.sqrt(pi)
    
    def case_small_alpha_large_v():
        # α <= 0.2 && v >= 5
        invv2 = 1.0 / v2
        return (alpha / sqrt_pi * invv2) * (1.0 + 1.5 * invv2 + 3.75 * invv2**2)
    
    def case_small_alpha_small_v():
        # α <= 0.2 && v < 5
        H = harris_series(v)
        H0, H1, H2 = H[0], H[1], H[2]
        return H0 + (H1 + H2 * alpha) * alpha
    
    def case_intermediate():
        # α <= 1.4 && α + v < 3.2
        # Modified Harris series: M_i is H'_i in the source text
        H = harris_series(v)
        H0, H1, H2 = H[0], H[1], H[2]
        
        M0 = H0
        M1 = H1 + 2.0 / sqrt_pi * M0
        M2 = H2 - M0 + 2.0 / sqrt_pi * M1
        M3 = 2.0 / (3.0 * sqrt_pi) * (1.0 - H2) - (2.0 / 3.0) * v2 * M1 + 2.0 / sqrt_pi * M2
        M4 = 2.0 / 3.0 * v2 * v2 * M0 - 2.0 / (3.0 * sqrt_pi) * M1 + 2.0 / sqrt_pi * M3
        
        psi = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * alpha) * alpha) * alpha
        return psi * (M0 + (M1 + (M2 + (M3 + M4 * alpha) * alpha) * alpha) * alpha)
    
    def case_large_alpha():
        # α > 1.4 or (α > 0.2 and α + v > 3.2)
        alpha2 = alpha * alpha
        r2 = v2 / alpha2
        alpha_invu = 1.0 / (jnp.sqrt(2.0) * ((r2 + 1.0) * alpha))
        alpha2_invu2 = alpha_invu * alpha_invu
        
        return (jnp.sqrt(2.0 / pi) * alpha_invu * 
                (1.0 + (3.0 * r2 - 1.0 + ((r2 - 2.0) * 15.0 * r2 + 2.0) * alpha2_invu2) * alpha2_invu2))
    
    # Apply the exact same logic as Korg.jl
    return jnp.where(
        (alpha <= 0.2) & (v >= 5.0),
        case_small_alpha_large_v(),
        jnp.where(
            alpha <= 0.2,  # v < 5
            case_small_alpha_small_v(),
            jnp.where(
                (alpha <= 1.4) & (alpha + v < 3.2),
                case_intermediate(),
                case_large_alpha()
            )
        )
    )


@jax.jit
def line_profile(lambda_0: float, 
                sigma: float, 
                gamma: float, 
                amplitude: float,
                wavelengths: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a Voigt profile centered on λ₀ with Doppler width σ and Lorentz HWHM γ.
    
    This is the exact implementation matching Korg.jl line 229-233:
    ```julia
    function line_profile(λ₀::Real, σ::Real, γ::Real, amplitude::Real, λ::Real)
        inv_σsqrt2 = 1 / (σ * sqrt(2))
        scaling = inv_σsqrt2 / sqrt(π) * amplitude  
        voigt_hjerting(γ * inv_σsqrt2, abs(λ - λ₀) * inv_σsqrt2) * scaling
    end
    ```
    
    Parameters
    ----------
    lambda_0 : float
        Central wavelength of the line in cm
    sigma : float
        Doppler broadening width (standard deviation) in cm
    gamma : float  
        Lorentz broadening parameter (HWHM) in cm
    amplitude : float
        Total line strength (integrated absorption coefficient)
    wavelengths : jnp.ndarray
        Array of wavelengths at which to evaluate the profile in cm
        
    Returns
    -------
    jnp.ndarray
        Absorption coefficient values in cm⁻¹
    """
    
    # Exact translation of Korg.jl implementation
    inv_sigma_sqrt2 = 1.0 / (sigma * jnp.sqrt(2.0))
    scaling = inv_sigma_sqrt2 / jnp.sqrt(pi) * amplitude
    
    # Calculate α and v parameters
    alpha = gamma * inv_sigma_sqrt2
    v = jnp.abs(wavelengths - lambda_0) * inv_sigma_sqrt2
    
    # Handle both scalar and array wavelength inputs
    if jnp.ndim(v) == 0:
        # Scalar case
        voigt_values = voigt_hjerting(alpha, v)
    else:
        # Array case - use vectorized Hjerting function
        voigt_values = jax.vmap(voigt_hjerting, in_axes=(None, 0))(alpha, v)
    
    return voigt_values * scaling


@jax.jit  
def exponential_integral_1(x: float) -> float:
    """
    Compute the first exponential integral E₁(x).
    
    This is a rough approximation lifted from Kurucz's VCSE1F.
    Used in specialized line profile calculations (e.g., Brackett lines).
    
    Parameters
    ----------
    x : float
        Input value
        
    Returns
    -------  
    float
        E₁(x) value
    """
    
    def case_small():
        """x <= 0.01"""
        return -jnp.log(x) - 0.577215 + x
    
    def case_medium():
        """0.01 < x <= 1.0"""
        return (-jnp.log(x) - 0.57721566 + 
                x * (0.99999193 + x * (-0.24991055 + x * (0.05519968 + 
                x * (-0.00976004 + x * 0.00107857)))))
    
    def case_large():
        """1.0 < x <= 30.0"""
        numerator = x * (x + 2.334733) + 0.25062
        denominator = x * (x + 3.330657) + 1.681534
        return numerator / (denominator * x) * jnp.exp(-x)
    
    return jnp.where(
        x <= 0.0,
        0.0,
        jnp.where(
            x <= 0.01,
            case_small(),
            jnp.where(
                x <= 1.0,
                case_medium(),
                jnp.where(
                    x <= 30.0,
                    case_large(),
                    0.0
                )
            )
        )
    )


@jax.jit
def gaussian_profile(lambda_0: float,
                    sigma: float,
                    amplitude: float, 
                    wavelengths: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a pure Gaussian (Doppler-only) line profile.
    
    Parameters
    ----------
    lambda_0 : float
        Central wavelength in cm
    sigma : float
        Doppler width (standard deviation) in cm
    amplitude : float
        Total line strength
    wavelengths : jnp.ndarray
        Wavelengths to evaluate at in cm
        
    Returns
    -------
    jnp.ndarray
        Absorption coefficient values in cm⁻¹
    """
    delta_lambda = wavelengths - lambda_0
    normalization = amplitude / (sigma * jnp.sqrt(2.0 * pi))
    
    return normalization * jnp.exp(-0.5 * (delta_lambda / sigma)**2)


@jax.jit
def doppler_width(lambda_0: float, T: float, mass: float, xi: float = 0.0) -> float:
    """
    Calculate Doppler width for thermal and microturbulent broadening.
    
    Parameters
    ----------
    lambda_0 : float
        Rest wavelength in cm
    T : float
        Temperature in K
    mass : float
        Atomic mass in grams
    xi : float
        Microturbulent velocity in cm/s
        
    Returns
    -------
    float
        Doppler width (standard deviation) in cm
    """
    from ..constants import kboltz_cgs, c_cgs
    
    # Thermal velocity
    v_thermal = jnp.sqrt(2.0 * kboltz_cgs * T / mass)
    
    # Total velocity including microturbulence
    v_total = jnp.sqrt(v_thermal**2 + xi**2)
    
    # Doppler width in wavelength units
    return lambda_0 * v_total / c_cgs


# Simplified functions for testing compatibility
@jax.jit
def gaussian_profile(wavelengths: jnp.ndarray, line_center: float, sigma: float) -> jnp.ndarray:
    """
    Simple Gaussian profile for testing
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Wavelength grid in Angstroms
    line_center : float
        Line center in Angstroms
    sigma : float
        Line width in Angstroms
        
    Returns
    -------
    jnp.ndarray
        Normalized Gaussian profile
    """
    delta = wavelengths - line_center
    profile = jnp.exp(-0.5 * (delta / sigma)**2)
    # Normalize approximately
    return profile / (sigma * jnp.sqrt(2 * jnp.pi))


@jax.jit
def lorentzian_profile(wavelengths: jnp.ndarray, line_center: float, gamma: float) -> jnp.ndarray:
    """
    Simple Lorentzian profile for testing
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Wavelength grid in Angstroms
    line_center : float
        Line center in Angstroms
    gamma : float
        Line width parameter in Angstroms
        
    Returns
    -------
    jnp.ndarray
        Normalized Lorentzian profile
    """
    delta = wavelengths - line_center
    profile = gamma / (jnp.pi * (delta**2 + gamma**2))
    return profile


@jax.jit
def voigt_profile(wavelengths: jnp.ndarray, line_center: float, sigma: float, gamma: float) -> jnp.ndarray:
    """
    Simplified Voigt profile (approximate)
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Wavelength grid in Angstroms
    line_center : float
        Line center in Angstroms
    sigma : float
        Gaussian width in Angstroms
    gamma : float
        Lorentzian width in Angstroms
        
    Returns
    -------
    jnp.ndarray
        Approximate Voigt profile
    """
    # Simple approximation: use pseudo-Voigt (linear combination)
    gaussian = gaussian_profile(wavelengths, line_center, sigma)
    lorentzian = lorentzian_profile(wavelengths, line_center, gamma)
    
    # Mixing parameter (simplified)
    eta = gamma / (sigma + gamma)
    
    return (1 - eta) * gaussian + eta * lorentzian


@jax.jit
def lorentzian_profile(lambda_0: float,
                      gamma: float,
                      amplitude: float,
                      wavelengths: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a pure Lorentzian line profile.
    
    Parameters
    ----------
    lambda_0 : float
        Central wavelength in cm
    gamma : float
        Lorentz width (HWHM) in cm
    amplitude : float
        Total line strength
    wavelengths : jnp.ndarray
        Wavelengths to evaluate at in cm
        
    Returns
    -------
    jnp.ndarray
        Absorption coefficient values in cm⁻¹
    """
    delta_lambda = wavelengths - lambda_0
    normalization = amplitude / (pi * gamma)
    
    return normalization / (1.0 + (delta_lambda / gamma)**2)


@jax.jit
def voigt_profile(wavelengths: jnp.ndarray, 
                 line_center: float, 
                 doppler_width: float, 
                 lorentz_width: float) -> jnp.ndarray:
    """
    Compute a normalized Voigt profile.
    
    This function provides the interface expected by core.py.
    
    Parameters
    ----------
    wavelengths : jnp.ndarray
        Array of wavelengths at which to evaluate the profile
    line_center : float
        Central wavelength of the line
    doppler_width : float
        Doppler broadening width (standard deviation)
    lorentz_width : float
        Lorentz broadening parameter (HWHM)
        
    Returns
    -------
    jnp.ndarray
        Normalized profile values
    """
    # Use amplitude=1 for normalized profile
    return line_profile(line_center, doppler_width, lorentz_width, 1.0, wavelengths)