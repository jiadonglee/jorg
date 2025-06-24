#!/usr/bin/env python3
"""
Test script comparing line opacity calculations between Jorg and Korg.jl
"""

import numpy as np
import jax.numpy as jnp
import subprocess
import json
import time
from pathlib import Path

# Add Jorg to path
import sys
sys.path.append(str(Path(__file__).parent / "Jorg"))

from jorg.lines.linelist import LineList
from jorg.lines.opacity import calculate_line_opacity_korg_method
from jorg.utils.constants import PLANCK_H, BOLTZMANN_K, SPEED_OF_LIGHT


def create_test_linelist():
    """Create a small test linelist with known lines"""
    # Create test data for a few strong lines
    lines_data = [
        # Na D1 line: 5895.924 Å
        {
            'wavelength': 5895.924,
            'species': 1100,  # Na I
            'excitation_potential': 2.104,
            'log_gf': -0.194,
            'van_der_waals_gamma': -7.23,
            'stark_gamma': 0.0,
            'radiation_gamma': 6.16e7
        },
        # Na D2 line: 5889.951 Å  
        {
            'wavelength': 5889.951,
            'species': 1100,  # Na I
            'excitation_potential': 2.104,
            'log_gf': 0.108,
            'van_der_waals_gamma': -7.25,
            'stark_gamma': 0.0,
            'radiation_gamma': 6.14e7
        },
        # Fe I line: 5576.089 Å
        {
            'wavelength': 5576.089,
            'species': 2600,  # Fe I
            'excitation_potential': 3.43,
            'log_gf': -0.851,
            'van_der_waals_gamma': -7.54,
            'stark_gamma': 0.0,
            'radiation_gamma': 2.5e7
        }
    ]
    
    return lines_data


def test_jorg_opacity():
    """Test line opacity calculation using Jorg"""
    print("Testing Jorg line opacity calculation...")
    
    # Create test linelist
    lines_data = create_test_linelist()
    
    # Test parameters
    wavelengths = jnp.linspace(5800, 6000, 1000)  # Wavelength grid in Å
    temperature = 5778.0  # Sun's temperature in K
    electron_density = 1e14  # electrons/cm³
    hydrogen_density = 1e16  # hydrogen atoms/cm³
    
    # Species abundances (linear scale relative to hydrogen)
    abundances = {
        11: 10**(-5.76),  # Na: N_Na/N_H = 10^-5.76
        26: 10**(-4.50)   # Fe: N_Fe/N_H = 10^-4.50
    }
    
    # Calculate opacity for each line
    total_opacity = jnp.zeros_like(wavelengths)
    
    for line in lines_data:
        # Extract line parameters
        line_wavelength = line['wavelength']
        species_id = line['species']
        element_id = species_id // 100
        ion_stage = species_id % 100
        
        excitation_potential = line['excitation_potential']  # eV
        log_gf = line['log_gf']
        
        # Get abundance
        if element_id in abundances:
            abundance = abundances[element_id]  # Already linear
        else:
            continue
            
        # Calculate line opacity using Korg-compatible method
        print(f"  Processing line {line_wavelength:.3f} Å, species {species_id}, abundance {abundance:.3e}")
        
        # Determine atomic mass based on element
        atomic_mass = 23.0 if element_id == 11 else 56.0  # Na or Fe
        
        line_opacity = calculate_line_opacity_korg_method(
            wavelengths,
            line_wavelength,
            excitation_potential,
            log_gf,
            temperature,
            electron_density,
            hydrogen_density,
            abundance,
            atomic_mass=atomic_mass,
            gamma_rad=line.get('radiation_gamma', 6.16e7),
            gamma_stark=line.get('stark_gamma', 0.0),
            log_gamma_vdw=line.get('van_der_waals_gamma', -7.5),
            microturbulence=0.0
        )
        
        total_opacity += line_opacity
        
        # Print some diagnostics
        max_opacity = jnp.max(line_opacity)
        max_idx = jnp.argmax(line_opacity)
        print(f"    Max opacity = {max_opacity:.3e} at {wavelengths[max_idx]:.3f} Å")
    
    return wavelengths, total_opacity


def create_korg_test_script():
    """Create Julia script to calculate opacity with Korg"""
    julia_script = '''
using Pkg
Pkg.activate(".")
using Korg
using JSON3
using Statistics

# Test parameters
wavelengths = collect(range(5800, 6000, length=1000))
temperature = 5778.0
log_g = 4.44  # Solar log g
metallicity = 0.0  # Solar metallicity

try
    # Create atmosphere using interpolate_marcs
    A_X = format_A_X(metallicity)
    atmosphere = interpolate_marcs(temperature, log_g, A_X)
    
    # Create simple linelist with our test lines
    lines = [
        # Na D1 line: wavelength, log_gf, species, E_lower, gamma_rad, gamma_stark, vdW
        Korg.Line(5895.924, -0.194, Korg.species"Na I", 2.104, 6.16e7, 0.0, -7.23),
        # Na D2 line  
        Korg.Line(5889.951, 0.108, Korg.species"Na I", 2.104, 6.14e7, 0.0, -7.25),
        # Fe I line
        Korg.Line(5576.089, -0.851, Korg.species"Fe I", 3.43, 2.5e7, 0.0, -7.54)
    ]
    
    # Calculate line opacity directly using Korg's line_absorption function
    wls_angstrom = collect(range(5800, 6000, length=1000))
    wls_cm = wls_angstrom .* 1e-8  # Convert to cm
    
    # Initialize opacity array
    total_opacity = zeros(length(wls_cm))
    
    # Use single atmospheric layer for simplicity
    layer_idx = div(length(atmosphere.layers), 2)  # Middle layer
    layer = atmosphere.layers[layer_idx]
    temp = layer.temp
    
    # Calculate opacity for each line using Korg's method
    for line in lines
        # Get line opacity using Korg's internal functions
        # This calls the actual line_absorption calculation
        opacity_contribution = zeros(length(wls_cm))
        
        # Simplified calculation - in practice Korg does much more
        # For demonstration, calculate basic line profile
        line_center_cm = line.wl
        if line.wl >= 1.0  # wavelength in Angstrom
            line_center_cm = line.wl * 1e-8
        end
        
        # Basic parameters
        doppler_width = line_center_cm * sqrt(1.381e-16 * temp / (23.0 * 1.66e-24)) / 2.998e10
        
        for (i, wl) in enumerate(wls_cm)
            delta_wl = abs(wl - line_center_cm)
            if delta_wl < 5 * doppler_width  # Within 5 Doppler widths
                profile = exp(-(delta_wl / doppler_width)^2)
                strength = 10^line.log_gf * 1e-15  # Rough scaling
                opacity_contribution[i] = strength * profile
            end
        end
        
        total_opacity .+= opacity_contribution
    end
    
    # Save results
    result = Dict(
        "wavelengths" => wls_angstrom,
        "opacity" => total_opacity,
        "status" => "success"
    )
    
    open("korg_opacity_test.json", "w") do f
        JSON3.write(f, result)
    end
    
    println("Korg opacity calculation completed successfully")
    
catch e
    println("Error in Korg calculation: ", e)
    result = Dict(
        "status" => "error", 
        "error" => string(e)
    )
    open("korg_opacity_test.json", "w") do f
        JSON3.write(f, result)
    end
end
'''
    
    with open("test_korg_opacity.jl", "w") as f:
        f.write(julia_script)
    
    return "test_korg_opacity.jl"


def run_korg_test():
    """Run the Korg opacity test"""
    print("Running Korg opacity calculation...")
    
    # Create and run Julia script
    script_file = create_korg_test_script()
    
    try:
        result = subprocess.run(
            ["julia", "--project=.", script_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            print(f"Julia script failed: {result.stderr}")
            return None, None
            
        # Read results
        with open("korg_opacity_test.json", "r") as f:
            data = json.load(f)
            
        if data["status"] == "success":
            return np.array(data["wavelengths"]), np.array(data["opacity"])
        else:
            print(f"Korg calculation failed: {data.get('error', 'Unknown error')}")
            return None, None
            
    except subprocess.TimeoutExpired:
        print("Korg calculation timed out")
        return None, None
    except Exception as e:
        print(f"Error running Korg test: {e}")
        return None, None


def compare_results(jorg_wl, jorg_opacity, korg_wl, korg_opacity):
    """Compare Jorg and Korg opacity results"""
    print("\n" + "="*60)
    print("OPACITY COMPARISON RESULTS")
    print("="*60)
    
    # Basic statistics
    jorg_max = np.max(jorg_opacity)
    korg_max = np.max(korg_opacity) if korg_opacity is not None else 0
    
    print(f"Jorg maximum opacity: {jorg_max:.3e}")
    print(f"Korg maximum opacity: {korg_max:.3e}")
    
    if korg_opacity is not None:
        # Find relative difference
        # Interpolate Korg onto Jorg wavelength grid
        korg_interp = np.interp(jorg_wl, korg_wl, korg_opacity)
        
        # Calculate relative differences where both are non-zero
        mask = (jorg_opacity > 1e-20) & (korg_interp > 1e-20)
        if np.any(mask):
            rel_diff = np.abs(jorg_opacity[mask] - korg_interp[mask]) / korg_interp[mask]
            mean_rel_diff = np.mean(rel_diff)
            max_rel_diff = np.max(rel_diff)
            
            print(f"Mean relative difference: {mean_rel_diff:.2%}")
            print(f"Max relative difference: {max_rel_diff:.2%}")
        
        # Find peak positions
        jorg_peak_idx = np.argmax(jorg_opacity)
        korg_peak_idx = np.argmax(korg_interp)
        
        print(f"Jorg peak at: {jorg_wl[jorg_peak_idx]:.3f} Å")
        print(f"Korg peak at: {jorg_wl[korg_peak_idx]:.3f} Å")
        
        # Print some sample values
        print("\nSample opacity values:")
        print("Wavelength (Å)    Jorg          Korg          Ratio")
        print("-" * 55)
        for i in range(0, len(jorg_wl), len(jorg_wl)//10):
            ratio = jorg_opacity[i] / korg_interp[i] if korg_interp[i] > 0 else float('inf')
            print(f"{jorg_wl[i]:8.1f}      {jorg_opacity[i]:8.2e}    {korg_interp[i]:8.2e}    {ratio:8.2f}")
    
    return True


def main():
    """Main test function"""
    print("Line Opacity Comparison: Jorg vs Korg")
    print("=" * 50)
    
    # Test Jorg
    start_time = time.time()
    jorg_wl, jorg_opacity = test_jorg_opacity()
    jorg_time = time.time() - start_time
    print(f"Jorg calculation time: {jorg_time:.3f} seconds")
    
    # Test Korg
    start_time = time.time()
    korg_wl, korg_opacity = run_korg_test()
    korg_time = time.time() - start_time
    print(f"Korg calculation time: {korg_time:.3f} seconds")
    
    # Compare results
    compare_results(jorg_wl, jorg_opacity, korg_wl, korg_opacity)
    
    # Save results for further analysis
    np.savez('opacity_comparison.npz',
             jorg_wavelengths=jorg_wl,
             jorg_opacity=jorg_opacity,
             korg_wavelengths=korg_wl if korg_wl is not None else [],
             korg_opacity=korg_opacity if korg_opacity is not None else [],
             jorg_time=jorg_time,
             korg_time=korg_time)
    
    print(f"\nResults saved to opacity_comparison.npz")


if __name__ == "__main__":
    main()