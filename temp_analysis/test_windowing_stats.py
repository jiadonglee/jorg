#!/usr/bin/env python3
"""
Quick test to show line windowing statistics
"""

import sys
import os
import numpy as np

# Add Jorg source to path  
sys.path.append("/Users/jdli/Project/Korg.jl/Jorg/src/")

from jorg.opacity.korg_line_processor import KorgLineProcessor
from jorg.lines.linelist import read_linelist
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.statmech.species import Species

# Load linelist
linelist = read_linelist("/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald")
print(f"Total lines loaded: {len(linelist)}")

# Filter to wavelength range 5000-5020
lines_in_range = [line for line in linelist 
                  if 5000e-8 <= line.wavelength <= 5020e-8]
print(f"Lines in 5000-5020 Å range: {len(lines_in_range)}")

# Set up test conditions (photosphere)
T = 5780.0  # Surface temperature
ne = 1e14   # Electron density
wl_array_cm = np.linspace(5000e-8, 5020e-8, 4001)

# Simple number densities
n_densities = {
    Species.from_atomic_number(1, 0): 1e17,  # H I
    Species.from_atomic_number(26, 0): 1e11, # Fe I
}

# Create processor
processor = KorgLineProcessor(verbose=True)

# Test with realistic continuum opacity
def continuum_opacity_fn(wl_cm):
    # Typical solar photosphere continuum opacity
    return 1e-6  # cm^-1

print("\nRunning line processor with realistic continuum opacity...")
result = processor.process_lines(
    wl_array_cm=wl_array_cm,
    temps=np.array([T]),
    electron_densities=np.array([ne]),
    n_densities=n_densities,
    partition_fns=create_default_partition_functions(),
    linelist=lines_in_range,
    microturbulence_cm_s=1e5,  # 1 km/s
    continuum_opacity_fn=continuum_opacity_fn,
    cutoff_threshold=3e-4
)

print(f"\nFINAL WINDOWING STATISTICS:")
print(f"Lines processed: {result.lines_processed}")
print(f"Lines contributing: {result.lines_windowed}")
print(f"Reduction factor: {100 * (1 - result.lines_windowed/len(lines_in_range)):.1f}%")