"""
Jorg GPU Configuration and Utilities
=====================================

This module provides GPU/device management for JAX-based computations.
It handles device detection, precision configuration, and timing utilities.

Usage:
    from jorg.gpu import init_jax, get_device_info, timed_block

    init_jax()  # Initialize and log device info

    with timed_block("synthesis"):
        result = synthesize(...)
"""

import jax
import jax.numpy as jnp
import numpy as np
from contextlib import contextmanager
import time
from typing import Optional, Tuple, Any
from functools import wraps

# Global configuration
_GPU_CONFIG = {
    'initialized': False,
    'device': None,
    'dtype': jnp.float64,
    'use_gpu': False,
}


def init_jax(use_float32_on_gpu: bool = True, verbose: bool = True) -> dict:
    """
    Initialize JAX and detect available devices.

    Parameters
    ----------
    use_float32_on_gpu : bool, default=True
        Use float32 on GPU for better performance, float64 on CPU for accuracy.
    verbose : bool, default=True
        Print device and configuration information.

    Returns
    -------
    dict
        Device configuration information.
    """
    global _GPU_CONFIG

    if _GPU_CONFIG['initialized']:
        return get_device_info()

    # Get JAX version info
    jax_version = jax.__version__
    try:
        import jaxlib
        jaxlib_version = jaxlib.__version__
    except ImportError:
        jaxlib_version = "unknown"

    # Detect devices
    devices = jax.devices()
    device = devices[0] if devices else None
    device_kind = getattr(device, "device_kind", "unknown")
    device_platform = getattr(device, "platform", "unknown")
    default_backend = jax.default_backend()

    # Determine if GPU is available (use platform, not device_kind model string)
    use_gpu = any(getattr(d, "platform", "") in ("gpu", "tpu") for d in devices)

    # Set precision based on device
    if use_gpu and use_float32_on_gpu:
        dtype = jnp.float32
        # Enable float32 matrix multiplication precision for better performance
        jax.config.update("jax_default_matmul_precision", "float32")
    else:
        dtype = jnp.float64
        # Ensure float64 is available
        jax.config.update("jax_enable_x64", True)

    _GPU_CONFIG.update({
        'initialized': True,
        'device': device,
        'dtype': dtype,
        'use_gpu': use_gpu,
        'jax_version': jax_version,
        'jaxlib_version': jaxlib_version,
        'device_kind': device_kind,
        'device_platform': device_platform,
        'default_backend': default_backend,
        'num_devices': len(devices),
    })

    if verbose:
        print(f"JAX {jax_version} / jaxlib {jaxlib_version}")
        print(f"Device: {device_kind} ({device_platform}, {len(devices)} device(s))")
        print(f"Precision: {'float32' if dtype == jnp.float32 else 'float64'}")
        if use_gpu:
            print("GPU acceleration: ENABLED")
        else:
            print("GPU acceleration: DISABLED (CPU mode)")

    return get_device_info()


def get_device_info() -> dict:
    """Get current device configuration."""
    if not _GPU_CONFIG['initialized']:
        init_jax(verbose=False)
    return dict(_GPU_CONFIG)


def get_dtype():
    """Get the configured JAX dtype (float32 for GPU, float64 for CPU)."""
    if not _GPU_CONFIG['initialized']:
        init_jax(verbose=False)
    return _GPU_CONFIG['dtype']


def is_gpu_available() -> bool:
    """Check if GPU is available and being used."""
    if not _GPU_CONFIG['initialized']:
        init_jax(verbose=False)
    return _GPU_CONFIG['use_gpu']


def ensure_array(x, dtype=None) -> jnp.ndarray:
    """
    Convert input to JAX array with appropriate dtype.

    Parameters
    ----------
    x : array-like
        Input array (numpy, list, or JAX array).
    dtype : dtype, optional
        Override dtype. If None, uses configured GPU/CPU dtype.

    Returns
    -------
    jnp.ndarray
        JAX array on the appropriate device.
    """
    if dtype is None:
        dtype = get_dtype()
    return jnp.asarray(x, dtype=dtype)


def to_numpy(x) -> np.ndarray:
    """
    Convert JAX array to NumPy, blocking until computation is complete.

    Parameters
    ----------
    x : jnp.ndarray
        JAX array.

    Returns
    -------
    np.ndarray
        NumPy array with results.
    """
    # Block until ready for accurate timing
    x = jax.device_get(x)
    return np.asarray(x)


def block_until_ready(x):
    """
    Block until JAX computation is complete.

    This is essential for accurate timing of GPU computations
    since JAX operations are asynchronous.

    Parameters
    ----------
    x : jnp.ndarray or pytree
        JAX array or pytree of arrays.

    Returns
    -------
    Same as input, but guaranteed to be computed.
    """
    if isinstance(x, jnp.ndarray):
        return x.block_until_ready()
    elif isinstance(x, (tuple, list)):
        return type(x)(block_until_ready(xi) for xi in x)
    elif isinstance(x, dict):
        return {k: block_until_ready(v) for k, v in x.items()}
    return x


@contextmanager
def timed_block(name: str = "", verbose: bool = True):
    """
    Context manager for timing code blocks with proper GPU synchronization.

    Parameters
    ----------
    name : str
        Name of the block for logging.
    verbose : bool
        Whether to print timing info.

    Yields
    ------
    dict
        Dictionary that will contain 'elapsed' time after block completes.

    Example
    -------
    >>> with timed_block("synthesis") as t:
    ...     result = synthesize(...)
    >>> print(f"Elapsed: {t['elapsed']:.2f}s")
    """
    result = {'elapsed': 0.0, 'name': name}

    # Synchronize before starting
    jax.random.PRNGKey(0).block_until_ready()

    start = time.perf_counter()
    try:
        yield result
    finally:
        # Block for any pending operations
        jax.random.PRNGKey(0).block_until_ready()
        end = time.perf_counter()
        result['elapsed'] = end - start

        if verbose and name:
            print(f"[{name}] {result['elapsed']:.3f}s")


def timed(func):
    """
    Decorator for timing functions with GPU synchronization.

    Example
    -------
    >>> @timed
    ... def my_function(x):
    ...     return jnp.sum(x ** 2)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        with timed_block(func.__name__, verbose=True) as t:
            result = func(*args, **kwargs)
            # Ensure result is computed before timing ends
            if isinstance(result, jnp.ndarray):
                result.block_until_ready()
        return result
    return wrapper


def compare_results(cpu_result, gpu_result, rtol=1e-5, atol=1e-8, name="result"):
    """
    Compare CPU and GPU results for numerical accuracy.

    Parameters
    ----------
    cpu_result : array-like
        Result from CPU computation.
    gpu_result : array-like
        Result from GPU computation.
    rtol : float
        Relative tolerance.
    atol : float
        Absolute tolerance.
    name : str
        Name for logging.

    Returns
    -------
    bool
        True if results match within tolerance.
    """
    cpu_arr = np.asarray(cpu_result)
    gpu_arr = np.asarray(gpu_result)

    if cpu_arr.shape != gpu_arr.shape:
        print(f"[{name}] Shape mismatch: CPU {cpu_arr.shape} vs GPU {gpu_arr.shape}")
        return False

    max_abs_diff = np.max(np.abs(cpu_arr - gpu_arr))
    max_rel_diff = np.max(np.abs(cpu_arr - gpu_arr) / (np.abs(cpu_arr) + atol))

    matches = np.allclose(cpu_arr, gpu_arr, rtol=rtol, atol=atol)

    if matches:
        print(f"[{name}] CPU/GPU match: max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e}")
    else:
        print(f"[{name}] CPU/GPU MISMATCH: max_abs={max_abs_diff:.2e}, max_rel={max_rel_diff:.2e}")

    return matches


# Convenience function for JIT with device placement
def jit_gpu(func, static_argnums=None, donate_argnums=None):
    """
    JIT compile a function with GPU-optimized settings.

    This is a convenience wrapper around jax.jit that applies
    GPU-friendly configurations.
    """
    return jax.jit(func, static_argnums=static_argnums, donate_argnums=donate_argnums)
