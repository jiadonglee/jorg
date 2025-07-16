"""
Bounds checking and domain validation for continuum absorption functions.

This module provides utilities for validating input parameters and handling
out-of-bounds conditions in continuum absorption calculations.
"""

import jax.numpy as jnp
from jax import jit
import numpy as np
from typing import Callable, Union, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class Interval:
    """Represents an exclusive interval (lower, upper)."""
    lower: float
    upper: float
    exclusive_lower: bool = True
    exclusive_upper: bool = True
    
    def __post_init__(self):
        """Validate interval bounds."""
        if self.lower >= self.upper:
            raise ValueError("Upper bound must exceed lower bound")
        
        # Handle infinite bounds
        if not np.isfinite(self.lower) and self.exclusive_lower:
            self.exclusive_lower = True
        if not np.isfinite(self.upper) and self.exclusive_upper:
            self.exclusive_upper = True
    
    def contains(self, value: float) -> bool:
        """Check if value is contained in the interval."""
        if self.exclusive_lower and self.exclusive_upper:
            return self.lower < value < self.upper
        elif self.exclusive_lower and not self.exclusive_upper:
            return self.lower < value <= self.upper
        elif not self.exclusive_lower and self.exclusive_upper:
            return self.lower <= value < self.upper
        else:
            return self.lower <= value <= self.upper


def closed_interval(lower: float, upper: float) -> Interval:
    """Create a closed interval [lower, upper]."""
    return Interval(lower, upper, exclusive_lower=False, exclusive_upper=False)


def contained(value: float, interval: Interval) -> bool:
    """Check if value is contained by interval."""
    return interval.contains(value)


def contained_slice(values: np.ndarray, interval: Interval) -> slice:
    """
    Find slice of values that are contained by interval.
    
    Args:
        values: Sorted array of values
        interval: Interval to check against
        
    Returns:
        Slice object for indices within the interval
    """
    # Find first index where value > interval.lower
    start_idx = np.searchsorted(values, interval.lower, side='right')
    
    # Find last index where value < interval.upper
    end_idx = np.searchsorted(values, interval.upper, side='left')
    
    # Return slice (may be empty if start_idx >= end_idx)
    return slice(start_idx, end_idx)


class BoundsError(Exception):
    """Exception raised when function arguments are out of bounds."""
    
    def __init__(self, value: float, bounds: Interval, parameter: str, function_name: str):
        self.value = value
        self.bounds = bounds
        self.parameter = parameter
        self.function_name = function_name
        
        # Format bounds display
        lower_str = f"-∞" if np.isneginf(bounds.lower) else f"{bounds.lower}"
        upper_str = f"+∞" if np.isposinf(bounds.upper) else f"{bounds.upper}"
        
        message = (f"{function_name}: invalid {parameter} = {value}. "
                  f"It should lie between {lower_str} and {upper_str}")
        
        super().__init__(message)


def _prepare_bounds_error(frequencies: np.ndarray, temperature: float, 
                         valid_indices: slice, function_name: str,
                         freq_bound: Interval, temp_bound: Interval) -> BoundsError:
    """Prepare bounds error with appropriate message."""
    
    if not contained(temperature, temp_bound):
        return BoundsError(temperature, temp_bound, "temperature", function_name)
    else:
        # Find the first out-of-bounds frequency
        if valid_indices.start > 0:
            bad_freq = frequencies[valid_indices.start - 1]
        else:
            bad_freq = frequencies[valid_indices.stop]
        
        return BoundsError(bad_freq, freq_bound, "frequency", function_name)


def bounds_checked_absorption(func: Callable, 
                            freq_bound: Interval = None,
                            temp_bound: Interval = None,
                            function_name: str = None) -> Callable:
    """
    Create a wrapper function that implements bounds checking and extrapolation.
    
    Args:
        func: Function with signature f(freq, temp, *args) -> absorption
        freq_bound: Valid frequency interval (Hz)
        temp_bound: Valid temperature interval (K)
        function_name: Name for error messages
        
    Returns:
        Wrapped function with bounds checking
    """
    
    if freq_bound is None:
        freq_bound = Interval(0.0, np.inf)
    if temp_bound is None:
        temp_bound = Interval(0.0, np.inf)
    if function_name is None:
        function_name = func.__name__
    
    def wrapped_func(frequencies: np.ndarray, temperature: float, *args,
                    error_oobounds: bool = False,
                    out_alpha: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Wrapped absorption function with bounds checking.
        
        Args:
            frequencies: Sorted array of frequencies (Hz)
            temperature: Temperature (K)
            *args: Additional arguments to pass to func
            error_oobounds: If True, raise error for out-of-bounds. If False, return 0.
            out_alpha: Pre-allocated output array (optional)
            
        Returns:
            Absorption coefficients (same shape as frequencies)
        """
        
        # Validate input
        if not np.all(np.diff(frequencies) >= 0):
            raise ValueError("Frequencies should be sorted in ascending order")
        
        # Determine output array type and allocate if needed
        if out_alpha is None:
            out_alpha = np.zeros_like(frequencies, dtype=np.float64)
        else:
            if len(out_alpha) != len(frequencies):
                raise ValueError("out_alpha must have same length as frequencies")
        
        # Find valid indices
        if not contained(temperature, temp_bound):
            valid_indices = slice(0, 0)  # Empty slice
        else:
            valid_indices = contained_slice(frequencies, freq_bound)
        
        # Handle out-of-bounds
        if (valid_indices.start != 0 or valid_indices.stop != len(frequencies)):
            if error_oobounds:
                raise _prepare_bounds_error(frequencies, temperature, valid_indices, 
                                          function_name, freq_bound, temp_bound)
            # Otherwise, out-of-bounds regions remain zero (default behavior)
        
        # Compute absorption for valid indices
        if valid_indices.start < valid_indices.stop:
            valid_freqs = frequencies[valid_indices]
            valid_absorption = func(valid_freqs, temperature, *args)
            out_alpha[valid_indices] += valid_absorption
        
        return out_alpha
    
    return wrapped_func


@jit
def validate_temperature_bounds(temperature: float, min_temp: float, max_temp: float) -> float:
    """
    Validate temperature is within bounds, clipping if necessary.
    
    Args:
        temperature: Temperature to validate
        min_temp: Minimum allowed temperature
        max_temp: Maximum allowed temperature
        
    Returns:
        Validated (possibly clipped) temperature
    """
    return jnp.clip(temperature, min_temp, max_temp)


@jit
def validate_frequency_bounds(frequencies: jnp.ndarray, min_freq: float, max_freq: float) -> jnp.ndarray:
    """
    Validate frequencies are within bounds, clipping if necessary.
    
    Args:
        frequencies: Frequencies to validate
        min_freq: Minimum allowed frequency
        max_freq: Maximum allowed frequency
        
    Returns:
        Validated (possibly clipped) frequencies
    """
    return jnp.clip(frequencies, min_freq, max_freq)


def create_safe_interpolator(func: Callable, 
                           freq_range: Tuple[float, float],
                           temp_range: Tuple[float, float],
                           default_value: float = 0.0) -> Callable:
    """
    Create a safe interpolator that returns default_value outside valid ranges.
    
    Args:
        func: Original function to wrap
        freq_range: (min_freq, max_freq) valid frequency range
        temp_range: (min_temp, max_temp) valid temperature range
        default_value: Value to return outside valid ranges
        
    Returns:
        Safe interpolator function
    """
    
    min_freq, max_freq = freq_range
    min_temp, max_temp = temp_range
    
    @jit
    def safe_func(frequencies: jnp.ndarray, temperature: float, *args) -> jnp.ndarray:
        """Safe interpolator with bounds checking."""
        
        # Check temperature bounds
        temp_valid = jnp.logical_and(temperature >= min_temp, temperature <= max_temp)
        
        # Check frequency bounds
        freq_valid = jnp.logical_and(frequencies >= min_freq, frequencies <= max_freq)
        
        # Compute function only where both are valid
        valid_mask = jnp.logical_and(freq_valid, temp_valid)
        
        # Initialize output with default value
        result = jnp.full_like(frequencies, default_value, dtype=jnp.float64)
        
        # Apply function only to valid regions
        result = jnp.where(valid_mask, func(frequencies, temperature, *args), result)
        
        return result
    
    return safe_func


def check_physical_consistency(temperature: float, electron_density: float, 
                             ion_density: float, neutral_density: float) -> None:
    """
    Check physical consistency of input parameters.
    
    Args:
        temperature: Temperature in K
        electron_density: Electron density in cm⁻³
        ion_density: Ion density in cm⁻³
        neutral_density: Neutral density in cm⁻³
        
    Raises:
        ValueError: If parameters are unphysical
    """
    
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature} K")
    
    if electron_density < 0:
        raise ValueError(f"Electron density must be non-negative, got {electron_density} cm⁻³")
    
    if ion_density < 0:
        raise ValueError(f"Ion density must be non-negative, got {ion_density} cm⁻³")
    
    if neutral_density < 0:
        raise ValueError(f"Neutral density must be non-negative, got {neutral_density} cm⁻³")
    
    # Check charge neutrality (allowing for small numerical errors)
    total_charge = electron_density - ion_density
    if abs(total_charge) > 1e-10 * max(electron_density, ion_density, 1e-10):
        raise ValueError(f"Charge neutrality violated: n_e - n_i = {total_charge:.2e} cm⁻³")


def validate_wavelength_range(wavelengths: np.ndarray, min_wavelength: float = 1000.0,
                             max_wavelength: float = 100000.0) -> np.ndarray:
    """
    Validate wavelength range for stellar spectroscopy.
    
    Args:
        wavelengths: Wavelength array in Å
        min_wavelength: Minimum allowed wavelength in Å
        max_wavelength: Maximum allowed wavelength in Å
        
    Returns:
        Validated wavelength array
        
    Raises:
        ValueError: If wavelengths are outside valid range
    """
    
    if np.any(wavelengths < min_wavelength):
        bad_wl = wavelengths[wavelengths < min_wavelength][0]
        raise ValueError(f"Wavelength {bad_wl:.1f} Å is below minimum {min_wavelength:.1f} Å")
    
    if np.any(wavelengths > max_wavelength):
        bad_wl = wavelengths[wavelengths > max_wavelength][0]
        raise ValueError(f"Wavelength {bad_wl:.1f} Å is above maximum {max_wavelength:.1f} Å")
    
    return wavelengths


# Commonly used intervals
STELLAR_TEMPERATURE_RANGE = Interval(1000.0, 100000.0)  # 1000K to 100,000K
OPTICAL_FREQUENCY_RANGE = Interval(3e14, 1e15)          # ~3000-10000 Å
UV_FREQUENCY_RANGE = Interval(1e15, 3e16)               # ~100-3000 Å
IR_FREQUENCY_RANGE = Interval(3e13, 3e14)               # ~10000-100000 Å
STELLAR_DENSITY_RANGE = Interval(1e6, 1e20)             # 10⁶ to 10²⁰ cm⁻³