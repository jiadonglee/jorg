# Jorg Opacity Calculation Tutorial

This tutorial shows how to calculate stellar opacity using Jorg with MARCS models and linelists.

## Core Components

### 1. Load MARCS Atmosphere Model

```python
import json
import numpy as np
import sys
import os

# Add Jorg to path
sys.path.insert(0, 'Jorg/src')
from jorg.statmech.eos import gas_pressure, total_pressure
from jorg.lines.opacity import calculate_line_opacity_korg_method

def load_marcs_model(atmosphere_file):
    """Load MARCS atmosphere data"""
    with open(atmosphere_file, 'r') as f:
        atmosphere_data = json.load(f)
    
    # Get layer closest to tau_5000 = 1 (photosphere)
    layer = min(atmosphere_data['atmosphere']['layers'], 
                key=lambda x: abs(x['tau_5000'] - 1.0))
    
    return {
        'temperature': layer['temperature'],
        'electron_density': layer['electron_density'],
        'number_density': layer['number_density']
    }
```

### 2. Calculate Equation of State (EOS)

```python
def calculate_eos(atmosphere_params):
    """Calculate gas and total pressure using Jorg EOS"""
    temp = atmosphere_params['temperature']
    n_density = atmosphere_params['number_density']
    e_density = atmosphere_params['electron_density']
    
    gas_press = gas_pressure(n_density, temp)
    total_press = total_pressure(n_density, e_density, temp)
    
    return {
        'gas_pressure': gas_press,
        'total_pressure': total_press
    }
```

### 3. Load Linelist

```python
def load_linelist(linelist_file):
    """Load atomic/molecular line data"""
    with open(linelist_file, 'r') as f:
        return json.load(f)
```

### 4. Calculate Total Opacity

```python
def calculate_total_opacity(atmosphere_file, linelist_file, wavelength):
    """
    Calculate total opacity at given wavelength
    
    Args:
        atmosphere_file: Path to MARCS atmosphere JSON
        linelist_file: Path to linelist JSON  
        wavelength: Wavelength in Angstroms
        
    Returns:
        float: Total opacity in cm^-1
    """
    # Load data
    atm_params = load_marcs_model(atmosphere_file)
    eos_params = calculate_eos(atm_params)
    linelist = load_linelist(linelist_file)
    
    # Atmospheric parameters
    temperature = atm_params['temperature']
    electron_density = atm_params['electron_density']
    hydrogen_density = atm_params['number_density'] * 0.9  # ~90% H
    
    # Line parameters (example Fe line)
    excitation_potential = 3.0  # eV
    log_gf = -2.0              # oscillator strength
    abundance = 7.5e-5         # Fe/H ratio
    microturbulence = 2.0      # km/s
    
    # Calculate opacity over small wavelength range
    wavelengths = np.linspace(wavelength - 1.0, wavelength + 1.0, 100)
    
    opacity = calculate_line_opacity_korg_method(
        wavelengths=wavelengths,
        line_wavelength=wavelength,
        excitation_potential=excitation_potential,
        log_gf=log_gf,
        temperature=temperature,
        electron_density=electron_density,
        hydrogen_density=hydrogen_density,
        abundance=abundance,
        microturbulence=microturbulence
    )
    
    # Return central wavelength opacity
    return opacity[len(wavelengths) // 2]
```

## Complete Example

```python
# File paths
atmosphere_file = 'jorg_related_files/marcs_data_for_jorg.json'
linelist_file = 'jorg_related_files/galah_linelist_for_jorg.json'
wavelength = 5000.0  # Angstroms

# Calculate opacity
try:
    total_opacity = calculate_total_opacity(atmosphere_file, linelist_file, wavelength)
    print(f"Total opacity at {wavelength} Å: {total_opacity:.3e} cm^-1")
except Exception as e:
    print(f"Error: {e}")
```

## Key Functions Summary

- `load_marcs_model()`: Extract atmospheric parameters from MARCS model
- `calculate_eos()`: Compute gas/total pressure using Jorg EOS
- `load_linelist()`: Load atomic line data
- `calculate_total_opacity()`: Main function returning opacity at wavelength

This provides the minimal core functionality for Jorg opacity calculations.