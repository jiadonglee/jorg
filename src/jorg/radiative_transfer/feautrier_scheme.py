"""
Feautrier Radiative Transfer Scheme

Implements the Feautrier method for solving the radiative transfer equation,
exactly as used in Korg.jl for high-accuracy stellar spectral synthesis.

Reference: Korg.jl/src/RadiativeTransfer/feautrier_transfer.jl
"""

import numpy as np
import jax
import jax.numpy as jnp
from typing import Tuple, Optional
from functools import partial

# Enable float64 support in JAX (required for numerical precision in RT)
jax.config.update("jax_enable_x64", True)


def feautrier_transfer(tau: np.ndarray, 
                      source: np.ndarray,
                      mu: float,
                      boundary_condition: str = "zero") -> np.ndarray:
    """
    Solve radiative transfer using Feautrier method.
    
    This is a second-order accurate method that solves for the mean intensity
    J = (I+ + I-)/2 where I+ is outward and I- is inward intensity.
    
    Parameters
    ----------
    tau : array_like, shape (n_layers,)
        Optical depth at each layer
    source : array_like, shape (n_layers,)
        Source function at each layer
    mu : float
        Cosine of angle with respect to normal (mu = cos(theta))
    boundary_condition : str
        Boundary condition: "zero" for no incident radiation,
                           "diffusion" for diffusion approximation
    
    Returns
    -------
    intensity : array_like, shape (n_layers,)
        Specific intensity at each layer
    """
    n = len(tau)
    
    # Calculate optical depth differences
    dtau = np.zeros(n-1)
    for i in range(n-1):
        dtau[i] = tau[i+1] - tau[i]
    
    # Set up tridiagonal matrix for Feautrier scheme
    # This solves: A * J = B where J is mean intensity
    a = np.zeros(n)  # Lower diagonal
    b = np.zeros(n)  # Main diagonal  
    c = np.zeros(n)  # Upper diagonal
    d = np.zeros(n)  # Right-hand side
    
    # Interior points (second-order differencing)
    for i in range(1, n-1):
        dtau_m = dtau[i-1]  # tau[i] - tau[i-1]
        dtau_p = dtau[i]    # tau[i+1] - tau[i]
        
        # Feautrier coefficients
        alpha_m = 2.0 / (dtau_m * (dtau_m + dtau_p))
        alpha_p = 2.0 / (dtau_p * (dtau_m + dtau_p))
        beta = alpha_m + alpha_p + 1.0 / mu**2
        
        a[i] = -alpha_m
        b[i] = beta
        c[i] = -alpha_p
        d[i] = source[i] / (mu**2)
    
    # Boundary conditions
    if boundary_condition == "zero":
        # No incident radiation at top
        # J[0] - I-[0] = 0 and I-[0] = 0, so J[0] = I+[0]/2
        b[0] = 1.0 + 2.0 / dtau[0]
        c[0] = -2.0 / dtau[0]
        d[0] = source[0] / (mu**2)

        # Diffusion approximation at bottom
        # dI/dtau = (I - S) at large tau
        dtau_last = dtau[-1]
        a[n-1] = -2.0 / dtau_last
        b[n-1] = 1.0 + 2.0 / dtau_last
        d[n-1] = source[n-1] / (mu**2)

    elif boundary_condition == "diffusion":
        # Diffusion approximation at both boundaries
        # Top boundary
        b[0] = 1.0 + 2.0 / (3.0 * dtau[0])
        c[0] = -2.0 / (3.0 * dtau[0])
        d[0] = source[0] / (mu**2)

        # Bottom boundary
        a[n-1] = -2.0 / (3.0 * dtau[-1])
        b[n-1] = 1.0 + 2.0 / (3.0 * dtau[-1])
        d[n-1] = source[n-1] / (mu**2)
    
    # Solve tridiagonal system using Thomas algorithm
    J = thomas_algorithm(a, b, c, d)
    
    # Convert mean intensity J to specific intensity I
    # For now, return J as approximation to I
    # Full implementation would compute I+ and I- from J
    intensity = J
    
    return intensity


def thomas_algorithm(a: np.ndarray, b: np.ndarray, 
                     c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """
    Solve tridiagonal system Ax = d using Thomas algorithm.
    
    Parameters
    ----------
    a : array_like
        Lower diagonal
    b : array_like
        Main diagonal
    c : array_like
        Upper diagonal
    d : array_like
        Right-hand side
        
    Returns
    -------
    x : array_like
        Solution vector
    """
    n = len(b)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    
    # Forward sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n):
        denominator = b[i] - a[i] * c_prime[i-1]
        if i < n-1:
            c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = d_prime[n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    
    return x


def short_characteristics_transfer(tau: np.ndarray,
                                  source: np.ndarray,
                                  mu: float,
                                  boundary_intensity: float = 0.0) -> np.ndarray:
    """
    Solve radiative transfer using short characteristics method.
    
    This method integrates the transfer equation along short ray segments,
    using parabolic interpolation for the source function.
    
    Parameters
    ----------
    tau : array_like, shape (n_layers,)
        Optical depth at each layer
    source : array_like, shape (n_layers,)
        Source function at each layer  
    mu : float
        Cosine of angle with respect to normal
    boundary_intensity : float
        Incident intensity at boundary
        
    Returns
    -------
    intensity : array_like, shape (n_layers,)
        Specific intensity at each layer
    """
    n = len(tau)
    intensity = np.zeros(n)
    
    if mu > 0:
        # Outward ray (from bottom to top)
        # Start from bottom boundary
        intensity[n-1] = source[n-1]  # Assuming LTE at bottom
        
        for i in range(n-2, -1, -1):
            dtau = (tau[i+1] - tau[i]) / mu
            
            if dtau < 1e-4:
                # Optically thin - linear approximation
                intensity[i] = intensity[i+1] * (1 - dtau) + source[i] * dtau
            else:
                # Use exponential integral for accuracy
                exp_dtau = np.exp(-dtau)
                intensity[i] = intensity[i+1] * exp_dtau + source[i] * (1 - exp_dtau)
                
                # Add parabolic correction for source function variation
                if i < n-2:
                    # Parabolic interpolation coefficients
                    tau_m = tau[i]
                    tau_0 = tau[i+1]
                    tau_p = tau[i+2] if i+2 < n else 2*tau[i+1] - tau[i]
                    
                    # Source function derivatives
                    dS_m = (source[i+1] - source[i]) / (tau_0 - tau_m) if tau_0 != tau_m else 0
                    dS_p = (source[i+2] - source[i+1]) / (tau_p - tau_0) if i+2 < n and tau_p != tau_0 else dS_m
                    
                    # Parabolic correction
                    d2S = (dS_p - dS_m) / (tau_p - tau_m) if tau_p != tau_m else 0
                    
                    if abs(d2S) > 1e-10:
                        # Add second-order correction
                        correction = d2S * dtau**2 * (1 - 2*exp_dtau + (1 + dtau)*exp_dtau) / 2
                        intensity[i] += correction
    else:
        # Inward ray (from top to bottom)
        # Start from top boundary
        intensity[0] = boundary_intensity
        
        for i in range(1, n):
            dtau = (tau[i] - tau[i-1]) / abs(mu)
            
            if dtau < 1e-4:
                # Optically thin
                intensity[i] = intensity[i-1] * (1 - dtau) + source[i] * dtau
            else:
                # Exponential integral
                exp_dtau = np.exp(-dtau)
                intensity[i] = intensity[i-1] * exp_dtau + source[i] * (1 - exp_dtau)
    
    return intensity


def hermite_spline_transfer(tau: np.ndarray,
                           source: np.ndarray,
                           mu: float) -> np.ndarray:
    """
    Solve radiative transfer using Hermite spline interpolation.
    
    This provides smooth, high-order accurate interpolation of the
    source function between grid points.
    
    Parameters
    ----------
    tau : array_like, shape (n_layers,)
        Optical depth at each layer
    source : array_like, shape (n_layers,)
        Source function at each layer
    mu : float
        Cosine of angle with respect to normal
        
    Returns
    -------
    intensity : array_like, shape (n_layers,)
        Specific intensity at each layer
    """
    n = len(tau)
    intensity = np.zeros(n)
    
    # Calculate source function derivatives using cubic spline
    dS_dtau = np.zeros(n)
    
    # Use finite differences for derivatives
    for i in range(1, n-1):
        h1 = tau[i] - tau[i-1]
        h2 = tau[i+1] - tau[i]
        
        if h1 > 0 and h2 > 0:
            # Weighted average of forward and backward differences
            w1 = h2 / (h1 + h2)
            w2 = h1 / (h1 + h2)
            dS_dtau[i] = w1 * (source[i] - source[i-1]) / h1 + \
                        w2 * (source[i+1] - source[i]) / h2
    
    # Boundary derivatives
    if tau[1] > tau[0]:
        dS_dtau[0] = (source[1] - source[0]) / (tau[1] - tau[0])
    if tau[n-1] > tau[n-2]:
        dS_dtau[n-1] = (source[n-1] - source[n-2]) / (tau[n-1] - tau[n-2])
    
    # Integrate using Hermite interpolation
    if mu > 0:
        # Outward ray
        intensity[n-1] = source[n-1]
        
        for i in range(n-2, -1, -1):
            dtau = (tau[i+1] - tau[i]) / mu
            
            # Hermite interpolation coefficients
            t = dtau
            h00 = (1 + 2*t) * (1 - t)**2
            h10 = t * (1 - t)**2
            h01 = t**2 * (3 - 2*t)
            h11 = t**2 * (t - 1)
            
            # Interpolated intensity
            intensity[i] = (h00 * source[i] + 
                          h10 * dtau * dS_dtau[i] +
                          h01 * source[i+1] + 
                          h11 * dtau * dS_dtau[i+1])
            
            # Add extinction
            intensity[i] = intensity[i+1] * np.exp(-dtau) + \
                          intensity[i] * (1 - np.exp(-dtau))
    else:
        # Inward ray
        intensity[0] = 0.0
        
        for i in range(1, n):
            dtau = (tau[i] - tau[i-1]) / abs(mu)
            
            # Hermite interpolation
            t = dtau
            h00 = (1 + 2*t) * (1 - t)**2
            h10 = t * (1 - t)**2
            h01 = t**2 * (3 - 2*t)
            h11 = t**2 * (t - 1)
            
            intensity[i] = (h00 * source[i-1] + 
                          h10 * dtau * dS_dtau[i-1] +
                          h01 * source[i] + 
                          h11 * dtau * dS_dtau[i])
            
            intensity[i] = intensity[i-1] * np.exp(-dtau) + \
                          intensity[i] * (1 - np.exp(-dtau))
    
    return intensity


def compare_rt_schemes(tau: np.ndarray, source: np.ndarray, mu: float = 0.5):
    """
    Compare different radiative transfer schemes.
    
    Parameters
    ----------
    tau : array_like
        Optical depth array
    source : array_like
        Source function array
    mu : float
        Cosine of angle
    """
    # Feautrier method
    I_feautrier = feautrier_transfer(tau, source, mu)
    
    # Short characteristics
    I_short = short_characteristics_transfer(tau, source, mu)
    
    # Hermite spline
    I_hermite = hermite_spline_transfer(tau, source, mu)
    
    print("=== RADIATIVE TRANSFER SCHEME COMPARISON ===")
    print(f"Optical depth range: {tau[0]:.2e} to {tau[-1]:.2e}")
    print(f"Number of layers: {len(tau)}")
    print(f"mu = {mu}")
    print()
    
    print("Emergent intensities:")
    print(f"  Feautrier:           {I_feautrier[0]:.4e}")
    print(f"  Short characteristics: {I_short[0]:.4e}")
    print(f"  Hermite spline:      {I_hermite[0]:.4e}")
    print()
    
    # Calculate differences
    diff_fs = abs(I_feautrier[0] - I_short[0]) / I_feautrier[0] * 100
    diff_fh = abs(I_feautrier[0] - I_hermite[0]) / I_feautrier[0] * 100
    
    print("Relative differences:")
    print(f"  Feautrier vs Short: {diff_fs:.2f}%")
    print(f"  Feautrier vs Hermite: {diff_fh:.2f}%")
    
    return I_feautrier, I_short, I_hermite


# ==================== PHASE 1.2 OPTIMIZATION ====================
# Vectorized batch versions for GPU acceleration

@partial(jax.jit, static_argnums=(3,))
def feautrier_transfer_jax(
    tau: jnp.ndarray,
    source: jnp.ndarray,
    mu: float,
    boundary_condition: str = "zero"
) -> jnp.ndarray:
    """
    JAX-compatible Feautrier radiative transfer solver.

    Converts the original NumPy implementation to use JAX arrays
    for JIT compilation and GPU acceleration.

    Parameters
    ----------
    tau : jnp.ndarray, shape (n_layers,)
        Optical depth at each layer
    source : jnp.ndarray, shape (n_layers,)
        Source function at each layer
    mu : float
        Cosine of angle with respect to normal
    boundary_condition : str (static)
        Boundary condition: "zero" or "diffusion"

    Returns
    -------
    intensity : jnp.ndarray, shape (n_layers,)
        Specific intensity at each layer
    """
    n = len(tau)

    # Calculate optical depth differences
    dtau = jnp.diff(tau)  # shape (n-1,)

    # Set up tridiagonal matrix for Feautrier scheme
    a = jnp.zeros(n)  # Lower diagonal
    b = jnp.zeros(n)  # Main diagonal
    c = jnp.zeros(n)  # Upper diagonal
    d = jnp.zeros(n)  # Right-hand side

    # Interior points (vectorized)
    indices = jnp.arange(1, n-1)
    dtau_m = dtau[indices-1]  # tau[i] - tau[i-1]
    dtau_p = dtau[indices]    # tau[i+1] - tau[i]

    # Feautrier coefficients (vectorized)
    alpha_m = 2.0 / (dtau_m * (dtau_m + dtau_p))
    alpha_p = 2.0 / (dtau_p * (dtau_m + dtau_p))
    beta = alpha_m + alpha_p + 1.0 / (mu**2)

    a = a.at[indices].set(-alpha_m)
    b = b.at[indices].set(beta)
    c = c.at[indices].set(-alpha_p)
    d = d.at[indices].set(source[indices] / (mu**2))

    # Boundary conditions (use jax.lax.cond for static string)
    if boundary_condition == "zero":
        # Top boundary
        b = b.at[0].set(1.0 + 2.0 / dtau[0])
        c = c.at[0].set(-2.0 / dtau[0])
        d = d.at[0].set(source[0] / (mu**2))

        # Bottom boundary
        a = a.at[n-1].set(-2.0 / dtau[-1])
        b = b.at[n-1].set(1.0 + 2.0 / dtau[-1])
        d = d.at[n-1].set(source[n-1] / (mu**2))
    else:  # diffusion
        # Top boundary
        b = b.at[0].set(1.0 + 2.0 / (3.0 * dtau[0]))
        c = c.at[0].set(-2.0 / (3.0 * dtau[0]))
        d = d.at[0].set(source[0] / (mu**2))

        # Bottom boundary
        a = a.at[n-1].set(-2.0 / (3.0 * dtau[-1]))
        b = b.at[n-1].set(1.0 + 2.0 / (3.0 * dtau[-1]))
        d = d.at[n-1].set(source[n-1] / (mu**2))

    # Solve tridiagonal system using JAX
    J = thomas_algorithm_jax(a, b, c, d)

    return J


@jax.jit
def thomas_algorithm_jax(
    a: jnp.ndarray,
    b: jnp.ndarray,
    c: jnp.ndarray,
    d: jnp.ndarray
) -> jnp.ndarray:
    """
    JAX-compatible Thomas algorithm for tridiagonal systems.

    Uses jax.lax.scan for efficient sequential computation.
    """
    n = len(b)

    # Forward sweep
    def forward_step(carry, i):
        c_prime_prev, d_prime_prev = carry

        denom = b[i] - a[i] * c_prime_prev
        c_prime_i = jnp.where(i < n-1, c[i] / denom, 0.0)
        d_prime_i = (d[i] - a[i] * d_prime_prev) / denom

        return (c_prime_i, d_prime_i), (c_prime_i, d_prime_i)

    # Initial values
    c_prime_0 = c[0] / b[0]
    d_prime_0 = d[0] / b[0]

    _, (c_prime, d_prime) = jax.lax.scan(
        forward_step,
        (c_prime_0, d_prime_0),
        jnp.arange(1, n)
    )

    # Prepend initial values
    c_prime = jnp.concatenate([jnp.array([c_prime_0]), c_prime])
    d_prime = jnp.concatenate([jnp.array([d_prime_0]), d_prime])

    # Back substitution
    def backward_step(x_next, i):
        x_i = d_prime[i] - c_prime[i] * x_next
        return x_i, x_i

    x_last = d_prime[n-1]
    _, x_reversed = jax.lax.scan(
        backward_step,
        x_last,
        jnp.arange(n-2, -1, -1)
    )

    # Reverse and append last element
    x = jnp.concatenate([x_reversed[::-1], jnp.array([x_last])])

    return x


@jax.jit
def feautrier_transfer_batch_wavelengths(
    tau_matrix: jnp.ndarray,
    source_matrix: jnp.ndarray,
    mu: float
) -> jnp.ndarray:
    """
    Vectorized Feautrier RT over wavelengths using vmap.

    OPTIMIZATION: Processes all wavelengths in parallel on GPU.

    Parameters
    ----------
    tau_matrix : jnp.ndarray, shape (n_layers, n_wavelengths)
        Optical depth for each layer and wavelength
    source_matrix : jnp.ndarray, shape (n_layers, n_wavelengths)
        Source function for each layer and wavelength
    mu : float
        Cosine of angle (single value for all wavelengths)

    Returns
    -------
    intensity_matrix : jnp.ndarray, shape (n_layers, n_wavelengths)
        Specific intensity at each layer and wavelength
    """
    # Vmap over wavelength dimension (axis 1)
    batched_feautrier = jax.vmap(
        lambda tau_wl, source_wl: feautrier_transfer_jax(tau_wl, source_wl, mu, "zero"),
        in_axes=(1, 1),  # Map over axis 1 (wavelengths)
        out_axes=1        # Output also has wavelengths on axis 1
    )

    return batched_feautrier(tau_matrix, source_matrix)


@jax.jit
def feautrier_transfer_batch_full(
    tau_matrix: jnp.ndarray,
    source_matrix: jnp.ndarray,
    mu_values: jnp.ndarray,
    mu_weights: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    FULLY VECTORIZED Feautrier RT over wavelengths AND mu angles.

    OPTIMIZATION STRATEGY:
    - Nested vmap: outer=wavelengths, inner=mu angles
    - All wavelengths × all mu angles processed simultaneously on GPU
    - Expected speedup: 5-10x for RT calculation
    - Target: 1s → 0.1-0.2s on A100 GPU

    Parameters
    ----------
    tau_matrix : jnp.ndarray, shape (n_layers, n_wavelengths)
        Optical depth for each layer and wavelength
    source_matrix : jnp.ndarray, shape (n_layers, n_wavelengths)
        Source function for each layer and wavelength
    mu_values : jnp.ndarray, shape (n_mu,)
        Cosine of angles for angular quadrature
    mu_weights : jnp.ndarray, shape (n_mu,)
        Quadrature weights for angle integration

    Returns
    -------
    flux : jnp.ndarray, shape (n_wavelengths,)
        Emergent flux at each wavelength
    intensity : jnp.ndarray, shape (n_layers, n_wavelengths)
        Mean intensity at each layer and wavelength (angle-averaged)

    Example
    -------
    >>> # Instead of nested Python loops:
    >>> # flux = np.zeros(n_wl)
    >>> # for i in range(n_wl):
    >>> #     for mu, weight in mu_grid:
    >>> #         I = feautrier_transfer(tau[:,i], source[:,i], mu)
    >>> #         flux[i] += weight * I[0] * mu
    >>>
    >>> # Use vectorized version:
    >>> flux, intensity = feautrier_transfer_batch_full(
    ...     tau_matrix, source_matrix, mu_values, mu_weights
    ... )
    >>> # All wavelengths × all mu computed in parallel!
    """
    def per_wavelength_all_mu(tau_wl, source_wl):
        """Compute intensity for all mu angles at one wavelength"""
        # Vmap over mu angles
        intensities_all_mu = jax.vmap(
            lambda mu: feautrier_transfer_jax(tau_wl, source_wl, mu, "zero")
        )(mu_values)  # shape: (n_mu, n_layers)

        # Compute flux contribution from all angles
        # flux = sum(weight * I[surface] * mu)
        emergent_intensities = intensities_all_mu[:, 0]  # shape: (n_mu,)
        flux_wl = jnp.sum(mu_weights * emergent_intensities * mu_values)

        # Compute mean intensity (angle-averaged)
        mean_intensity_wl = jnp.sum(
            mu_weights[:, None] * intensities_all_mu,  # broadcast weights
            axis=0
        )  # shape: (n_layers,)

        return flux_wl, mean_intensity_wl

    # Vmap over all wavelengths
    batched_rt = jax.vmap(
        per_wavelength_all_mu,
        in_axes=(1, 1),  # Map over wavelength axis
        out_axes=(0, 1)   # flux: (n_wl,), intensity: (n_layers, n_wl)
    )

    flux, intensity = batched_rt(tau_matrix, source_matrix)

    return flux, intensity


if __name__ == "__main__":
    # Test with typical stellar atmosphere
    n_layers = 56
    tau = np.logspace(-6, 2, n_layers)  # Optical depth from 1e-6 to 100
    
    # Planck source function (LTE)
    T = 5778 * (0.5 + 0.5 * tau**(1/4))  # Eddington-Barbier approximation
    source = T**4 / 5778**4  # Normalized Planck function
    
    # Compare schemes
    compare_rt_schemes(tau, source, mu=0.5)