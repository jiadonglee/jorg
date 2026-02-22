"""
JAX-based interpolation helpers for continuum opacity tables.

Boundary behavior names mirror Korg-style usage:
- "flat": clamp outside the tabulated domain to edge values
- "line": linear extrapolation using edge segments
- "zero": return zero outside the tabulated domain
"""

from functools import partial
from typing import Literal

import jax
import jax.numpy as jnp

BoundaryMode = Literal["flat", "line", "zero"]

_BOUNDARY_CODE = {
    "flat": 0,
    "line": 1,
    "zero": 2,
}


def _mode_code(mode: BoundaryMode) -> int:
    if mode not in _BOUNDARY_CODE:
        raise ValueError(f"Unsupported boundary mode: {mode}")
    return _BOUNDARY_CODE[mode]


@partial(jax.jit, static_argnums=(3,))
def _interp1_linear_impl(
    x: jnp.ndarray,
    grid: jnp.ndarray,
    values: jnp.ndarray,
    mode_code: int,
) -> jnp.ndarray:
    x_arr = jnp.asarray(x, dtype=jnp.float64)
    grid_arr = jnp.asarray(grid, dtype=jnp.float64)
    value_arr = jnp.asarray(values, dtype=jnp.float64)

    idx = jnp.searchsorted(grid_arr, x_arr, side="right") - 1
    idx = jnp.clip(idx, 0, grid_arr.shape[0] - 2)

    x0 = grid_arr[idx]
    x1 = grid_arr[idx + 1]
    y0 = value_arr[idx]
    y1 = value_arr[idx + 1]
    frac = (x_arr - x0) / (x1 - x0)

    # flat/zero clamp in-domain blend fraction; line keeps linear extrapolation
    if mode_code != _BOUNDARY_CODE["line"]:
        frac = jnp.clip(frac, 0.0, 1.0)

    out = y0 + frac * (y1 - y0)

    if mode_code == _BOUNDARY_CODE["zero"]:
        out_of_bounds = (x_arr < grid_arr[0]) | (x_arr > grid_arr[-1])
        out = jnp.where(out_of_bounds, 0.0, out)

    return out


def interp1_linear_clamped(
    x: jnp.ndarray,
    grid: jnp.ndarray,
    values: jnp.ndarray,
    mode: BoundaryMode = "flat",
) -> jnp.ndarray:
    """1D linear interpolation with explicit boundary behavior."""
    return _interp1_linear_impl(x, grid, values, _mode_code(mode))


@partial(jax.jit, static_argnums=(5, 6))
def _interp2_linear_impl(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    values: jnp.ndarray,
    x_mode_code: int,
    y_mode_code: int,
) -> jnp.ndarray:
    x_arr, y_arr = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(y, dtype=jnp.float64),
    )
    x_grid_arr = jnp.asarray(x_grid, dtype=jnp.float64)
    y_grid_arr = jnp.asarray(y_grid, dtype=jnp.float64)
    value_arr = jnp.asarray(values, dtype=jnp.float64)

    ix = jnp.searchsorted(x_grid_arr, x_arr, side="right") - 1
    iy = jnp.searchsorted(y_grid_arr, y_arr, side="right") - 1
    ix = jnp.clip(ix, 0, x_grid_arr.shape[0] - 2)
    iy = jnp.clip(iy, 0, y_grid_arr.shape[0] - 2)

    x0 = x_grid_arr[ix]
    x1 = x_grid_arr[ix + 1]
    y0 = y_grid_arr[iy]
    y1 = y_grid_arr[iy + 1]

    fx = (x_arr - x0) / (x1 - x0)
    fy = (y_arr - y0) / (y1 - y0)

    if x_mode_code != _BOUNDARY_CODE["line"]:
        fx = jnp.clip(fx, 0.0, 1.0)
    if y_mode_code != _BOUNDARY_CODE["line"]:
        fy = jnp.clip(fy, 0.0, 1.0)

    v00 = value_arr[ix, iy]
    v10 = value_arr[ix + 1, iy]
    v01 = value_arr[ix, iy + 1]
    v11 = value_arr[ix + 1, iy + 1]

    vx0 = v00 * (1.0 - fx) + v10 * fx
    vx1 = v01 * (1.0 - fx) + v11 * fx
    out = vx0 * (1.0 - fy) + vx1 * fy

    if x_mode_code == _BOUNDARY_CODE["zero"]:
        out = jnp.where((x_arr < x_grid_arr[0]) | (x_arr > x_grid_arr[-1]), 0.0, out)
    if y_mode_code == _BOUNDARY_CODE["zero"]:
        out = jnp.where((y_arr < y_grid_arr[0]) | (y_arr > y_grid_arr[-1]), 0.0, out)

    return out


def interp2_linear_clamped(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    values: jnp.ndarray,
    x_mode: BoundaryMode = "flat",
    y_mode: BoundaryMode = "flat",
) -> jnp.ndarray:
    """2D bilinear interpolation with per-axis boundary behavior."""
    return _interp2_linear_impl(
        x,
        y,
        x_grid,
        y_grid,
        values,
        _mode_code(x_mode),
        _mode_code(y_mode),
    )


def interp2_linear_extrap_line(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_grid: jnp.ndarray,
    y_grid: jnp.ndarray,
    values: jnp.ndarray,
) -> jnp.ndarray:
    """2D bilinear interpolation with linear extrapolation on both axes."""
    return interp2_linear_clamped(
        x,
        y,
        x_grid,
        y_grid,
        values,
        x_mode="line",
        y_mode="line",
    )

