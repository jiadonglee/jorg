#!/usr/bin/env python3
"""
Debug MARCS Grid Structure
=========================

Understand the exact structure and content of MARCS grids to fix JAX implementation.
"""

import h5py
import numpy as np
import sys
from pathlib import Path

# Add Jorg to path
sys.path.insert(0, 'Jorg/src')

def debug_grid_interpolation():
    """Debug the grid interpolation to understand structure"""
    
    print("DEBUGGING MARCS GRID INTERPOLATION")
    print("=" * 50)
    
    # Load SDSS grid
    grid_path = "Jorg/data/marcs_grids/SDSS_MARCS_atmospheres.h5"
    
    with h5py.File(grid_path, 'r') as f:
        grid = f['grid'][:]
        nodes = []
        for i in range(1, 6):  # grid_values/1 through grid_values/5
            nodes.append(f[f'grid_values/{i}'][:])
        
        param_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                      for name in f['grid_parameter_names'][:]]
    
    print(f"Grid shape: {grid.shape}")
    print(f"Parameters: {param_names}")
    print(f"Node counts: {[len(n) for n in nodes]}")
    
    # Test solar parameters
    Teff, logg, m_H, alpha_m, C_m = 5777.0, 4.44, 0.0, 0.0, 0.0
    params = [Teff, logg, m_H, alpha_m, C_m]
    
    print(f"\nTesting parameters: {params}")
    
    # Find grid indices manually
    indices = []
    for i, (param, node_array) in enumerate(zip(params, nodes)):
        idx = np.argmin(np.abs(node_array - param))
        indices.append(idx)
        print(f"{param_names[i]}: {param} -> index {idx} (node value {node_array[idx]})")
    
    # Extract atmosphere directly
    # Grid order: [carbon, alpha, metallicity, logg, Teff, quantities, layers]
    # Parameter order: [Teff, logg, metallicity, alpha, carbon]
    Teff_idx, logg_idx, mH_idx, alpha_idx, C_idx = indices
    
    print(f"\nExtracting grid[{C_idx}, {alpha_idx}, {mH_idx}, {logg_idx}, {Teff_idx}, :, :]")
    atm_data = grid[C_idx, alpha_idx, mH_idx, logg_idx, Teff_idx, :, :]
    print(f"Atmosphere data shape: {atm_data.shape}")
    
    # Transpose to get [layers, quantities]
    atm_data_transposed = atm_data.T  # Now [56 layers, 5 quantities]
    print(f"Transposed shape: {atm_data_transposed.shape}")
    
    # Analyze the quantities
    quantity_names = ["tau_5000", "sinh_z", "temp", "log_ne", "log_nt"]
    
    print(f"\nQuantity analysis (layer 25):")
    for i, name in enumerate(quantity_names):
        value = atm_data_transposed[25, i]
        print(f"  {name}: {value}")
    
    # Check for reasonable values
    tau_5000 = atm_data_transposed[:, 0]
    temp = atm_data_transposed[:, 2]
    
    print(f"\nPhysical checks:")
    print(f"  tau_5000 range: {np.min(tau_5000):.2e} to {np.max(tau_5000):.2e}")
    print(f"  Temperature range: {np.min(temp):.1f} to {np.max(temp):.1f} K")
    print(f"  Valid tau mask: {np.sum(tau_5000 > 0)}/{len(tau_5000)} layers")
    
    # Check what Korg produces for comparison
    from jorg.atmosphere import call_korg_interpolation
    korg_atm = call_korg_interpolation(Teff, logg, m_H)
    
    print(f"\nKorg comparison:")
    print(f"  Korg layers: {len(korg_atm.layers)}")
    print(f"  Layer 25 - Korg: T={korg_atm.layers[25].temp:.1f}K, tau={korg_atm.layers[25].tau_5000:.2e}")
    print(f"  Layer 25 - Grid: T={temp[25]:.1f}K, tau={tau_5000[25]:.2e}")
    
    return atm_data_transposed

def test_corrected_interpolation():
    """Test interpolation with corrected understanding"""
    
    print(f"\n\nTESTING CORRECTED INTERPOLATION")
    print("=" * 50)
    
    # Import and test the corrected implementation
    try:
        from jorg.atmosphere_jax import interpolate_marcs_jax
        
        print("Testing JAX implementation...")
        jax_atm = interpolate_marcs_jax(5777.0, 4.44, 0.0)
        print(f"JAX result: {len(jax_atm.layers)} layers")
        
        if len(jax_atm.layers) > 25:
            layer = jax_atm.layers[25]
            print(f"Layer 25: T={layer.temp:.1f}K, tau={layer.tau_5000:.2e}")
        
    except Exception as e:
        print(f"JAX implementation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    atm_data = debug_grid_interpolation()
    test_corrected_interpolation()