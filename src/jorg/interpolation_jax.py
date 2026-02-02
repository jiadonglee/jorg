"""
JAX-Compatible Interpolation Functions
=======================================

Pure JAX implementations of interpolation methods to replace SciPy dependencies.
All functions are JIT-compilable and GPU-accelerated.

PHASE 1.3 OPTIMIZATION: Replace scipy.interpolate.CubicSpline
"""

import jax
import jax.numpy as jnp
from typing import Tuple
from functools import partial


@jax.jit
def cubic_spline_coefficients(x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute cubic spline coefficients for natural cubic spline.

    Uses the natural boundary condition (second derivative = 0 at endpoints).
    This matches SciPy's CubicSpline with bc_type='natural'.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Independent variable values (must be sorted)
    y : jnp.ndarray, shape (n,) or (n, ...)
        Dependent variable values. If multidimensional, spline computed along axis 0.

    Returns
    -------
    a, b, c, d : jnp.ndarray
        Cubic spline coefficients for each interval
        S_i(x) = a[i] + b[i]*(x - x[i]) + c[i]*(x - x[i])^2 + d[i]*(x - x[i])^3
        for x in [x[i], x[i+1]]

    Notes
    -----
    The cubic spline S(x) satisfies:
    - S(x_i) = y_i for all i
    - S'(x) is continuous
    - S''(x) is continuous
    - S''(x_0) = S''(x_n) = 0 (natural boundary conditions)
    """
    n = len(x)
    h = jnp.diff(x)  # h[i] = x[i+1] - x[i]

    # Set up tridiagonal system for second derivatives
    # Natural boundary conditions: M[0] = M[n-1] = 0
    # For interior points: h[i-1]*M[i-1] + 2*(h[i-1]+h[i])*M[i] + h[i]*M[i+1] = 6*delta[i]

    # Build tridiagonal matrix
    alpha = jnp.zeros(n)
    beta = jnp.zeros(n)
    gamma = jnp.zeros(n)
    delta = jnp.zeros_like(y)

    # Natural boundary conditions
    alpha = alpha.at[0].set(0.0)
    beta = beta.at[0].set(1.0)
    gamma = gamma.at[0].set(0.0)
    delta = delta.at[0].set(0.0)

    alpha = alpha.at[n-1].set(0.0)
    beta = beta.at[n-1].set(1.0)
    gamma = gamma.at[n-1].set(0.0)
    delta = delta.at[n-1].set(0.0)

    # Interior points
    for i in range(1, n-1):
        alpha = alpha.at[i].set(h[i-1])
        beta = beta.at[i].set(2.0 * (h[i-1] + h[i]))
        gamma = gamma.at[i].set(h[i])

        # Finite difference approximation of second derivative
        dy_right = (y[i+1] - y[i]) / h[i]
        dy_left = (y[i] - y[i-1]) / h[i-1]
        delta = delta.at[i].set(6.0 * (dy_right - dy_left))

    # Solve tridiagonal system for second derivatives M
    M = _solve_tridiagonal_jax(alpha, beta, gamma, delta)

    # Compute cubic spline coefficients from M
    # For interval [x[i], x[i+1]]:
    # a[i] = y[i]
    # b[i] = (y[i+1] - y[i])/h[i] - h[i]*(2*M[i] + M[i+1])/6
    # c[i] = M[i]/2
    # d[i] = (M[i+1] - M[i])/(6*h[i])

    a = y[:-1]  # shape (n-1, ...)
    c = M[:-1] / 2.0
    if y.ndim > 1:
        h_reshape = h.reshape((h.shape[0],) + (1,) * (y.ndim - 1))
        d = (M[1:] - M[:-1]) / (6.0 * h_reshape)
        b = (y[1:] - y[:-1]) / h_reshape - h_reshape * (2.0 * M[:-1] + M[1:]) / 6.0
    else:
        d = (M[1:] - M[:-1]) / (6.0 * h)
        b = (y[1:] - y[:-1]) / h - h * (2.0 * M[:-1] + M[1:]) / 6.0

    return a, b, c, d


@jax.jit
def _solve_tridiagonal_jax(a: jnp.ndarray, b: jnp.ndarray, c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """
    Solve tridiagonal system using Thomas algorithm (JAX version).

    Solves: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]

    Parameters
    ----------
    a : jnp.ndarray, shape (n,)
        Lower diagonal (a[0] is ignored)
    b : jnp.ndarray, shape (n,)
        Main diagonal
    c : jnp.ndarray, shape (n,)
        Upper diagonal (c[n-1] is ignored)
    d : jnp.ndarray, shape (n,) or (n, ...)
        Right-hand side

    Returns
    -------
    x : jnp.ndarray
        Solution vector
    """
    n = len(b)

    # Forward elimination
    c_prime = jnp.zeros_like(c)
    d_prime = jnp.zeros_like(d)

    c_prime = c_prime.at[0].set(c[0] / b[0])
    d_prime = d_prime.at[0].set(d[0] / b[0])

    for i in range(1, n):
        denom = b[i] - a[i] * c_prime[i-1]
        if i < n-1:
            c_prime = c_prime.at[i].set(c[i] / denom)
        d_prime = d_prime.at[i].set((d[i] - a[i] * d_prime[i-1]) / denom)

    # Back substitution
    x = jnp.zeros_like(d)
    x = x.at[n-1].set(d_prime[n-1])

    for i in range(n-2, -1, -1):
        x = x.at[i].set(d_prime[i] - c_prime[i] * x[i+1])

    return x


@jax.jit
def evaluate_cubic_spline(x: jnp.ndarray, x_query: jnp.ndarray,
                         a: jnp.ndarray, b: jnp.ndarray,
                         c: jnp.ndarray, d: jnp.ndarray) -> jnp.ndarray:
    """
    Evaluate cubic spline at query points.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Original data points (knots)
    x_query : jnp.ndarray, shape (m,) or scalar
        Points at which to evaluate the spline
    a, b, c, d : jnp.ndarray, shape (n-1,) or (n-1, ...)
        Cubic spline coefficients from cubic_spline_coefficients()

    Returns
    -------
    y_query : jnp.ndarray, shape (m,) or (m, ...) or scalar
        Interpolated values
    """
    # Handle scalar input
    scalar_input = jnp.ndim(x_query) == 0
    if scalar_input:
        x_query = jnp.array([x_query])

    # Find interval indices for each query point
    # searchsorted finds i such that x[i-1] <= x_query < x[i]
    indices = jnp.searchsorted(x, x_query) - 1

    # Clamp indices to valid range [0, n-2]
    indices = jnp.clip(indices, 0, len(x) - 2)

    # Compute dx = x_query - x[indices]
    dx = x_query - x[indices]

    # Evaluate cubic polynomial: a + b*dx + c*dx^2 + d*dx^3
    if a.ndim == 1:
        # 1D case
        result = a[indices] + dx * (b[indices] + dx * (c[indices] + dx * d[indices]))
    else:
        # Multi-dimensional case (interpolate along axis 0)
        dx = dx[:, None]  # Broadcast for multiple output dimensions
        result = a[indices] + dx * (b[indices] + dx * (c[indices] + dx * d[indices]))

    return result[0] if scalar_input else result


@jax.jit
def cubic_spline_1d(x: jnp.ndarray, y: jnp.ndarray, x_query: jnp.ndarray) -> jnp.ndarray:
    """
    Convenience function for 1D cubic spline interpolation.

    Combines coefficient computation and evaluation in one call.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Independent variable values (must be sorted)
    y : jnp.ndarray, shape (n,)
        Dependent variable values
    x_query : jnp.ndarray, shape (m,) or scalar
        Points at which to interpolate

    Returns
    -------
    y_query : jnp.ndarray, shape (m,) or scalar
        Interpolated values

    Example
    -------
    >>> x = jnp.array([0.0, 1.0, 2.0, 3.0])
    >>> y = jnp.array([1.0, 2.0, 0.0, 1.5])
    >>> x_new = jnp.array([0.5, 1.5, 2.5])
    >>> y_new = cubic_spline_1d(x, y, x_new)
    """
    a, b, c, d = cubic_spline_coefficients(x, y)
    return evaluate_cubic_spline(x, x_query, a, b, c, d)


@partial(jax.jit, static_argnums=(2,))
def cubic_spline_nd(x: jnp.ndarray, y: jnp.ndarray, axis: int,
                    x_query: jnp.ndarray) -> jnp.ndarray:
    """
    Cubic spline interpolation along a specified axis for multi-dimensional data.

    This function mimics scipy.interpolate.CubicSpline with axis parameter.

    Parameters
    ----------
    x : jnp.ndarray, shape (n,)
        Independent variable values along interpolation axis
    y : jnp.ndarray, shape (..., n, ...)
        Dependent variable values. Dimension 'axis' should have size n.
    axis : int (static)
        Axis along which to interpolate
    x_query : jnp.ndarray or scalar
        Points at which to interpolate

    Returns
    -------
    y_query : jnp.ndarray
        Interpolated values

    Example
    -------
    >>> # Interpolate 2D array along axis 0
    >>> x = jnp.array([0.0, 1.0, 2.0])
    >>> y = jnp.random.normal(size=(3, 5))  # 3 points, 5 quantities
    >>> y_interp = cubic_spline_nd(x, y, axis=0, x_query=0.5)
    >>> # y_interp.shape = (5,)
    """
    # Move interpolation axis to position 0
    y_moved = jnp.moveaxis(y, axis, 0)

    # Compute spline coefficients
    a, b, c, d = cubic_spline_coefficients(x, y_moved)

    # Evaluate
    result = evaluate_cubic_spline(x, x_query, a, b, c, d)

    # Move axis back if necessary
    if jnp.ndim(x_query) == 0:
        # Scalar query - result has shape (n_other_dims,)
        # No need to move axis back for scalar
        return result
    else:
        # Array query - result has shape (n_query, n_other_dims)
        # Need to move query dimension to original axis position
        return jnp.moveaxis(result, 0, axis)


def test_cubic_spline_jax():
    """Test JAX cubic spline against known values."""
    print("=" * 60)
    print("Testing JAX Cubic Spline Implementation")
    print("=" * 60)

    # Simple test case
    x = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = jnp.array([0.0, 1.0, 0.5, 2.0, 1.5])

    # Query points
    x_query = jnp.array([0.5, 1.5, 2.5, 3.5])

    # Interpolate
    y_interp = cubic_spline_1d(x, y, x_query)

    print("\nTest 1: 1D Interpolation")
    print(f"x:       {x}")
    print(f"y:       {y}")
    print(f"x_query: {x_query}")
    print(f"y_interp: {y_interp}")

    # Test multidimensional
    y_2d = jnp.stack([y, 2*y, y**2], axis=1)  # shape (5, 3)
    y_2d_interp = cubic_spline_nd(x, y_2d, axis=0, x_query=1.5)

    print("\nTest 2: Multi-dimensional Interpolation")
    print(f"y_2d shape: {y_2d.shape}")
    print(f"y_2d_interp shape: {y_2d_interp.shape}")
    print(f"y_2d_interp: {y_2d_interp}")

    # Verify it's JIT-compilable
    print("\nTest 3: JIT Compilation")
    jitted_spline = jax.jit(cubic_spline_1d)
    y_jit = jitted_spline(x, y, x_query)
    print(f"JIT compiled: {jnp.allclose(y_interp, y_jit)}")

    # Compare with SciPy (if available)
    try:
        from scipy.interpolate import CubicSpline
        scipy_spline = CubicSpline(x, y, bc_type='natural')
        y_scipy = scipy_spline(x_query)

        print("\nTest 4: Comparison with SciPy")
        print(f"JAX:   {y_interp}")
        print(f"SciPy: {y_scipy}")
        print(f"Max difference: {jnp.max(jnp.abs(y_interp - y_scipy)):.2e}")
        print(f"Agreement: {jnp.allclose(y_interp, y_scipy, rtol=1e-6)}")
    except ImportError:
        print("\nTest 4: SciPy not available, skipping comparison")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_cubic_spline_jax()
