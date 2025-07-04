#!/usr/bin/env python3
"""
Simple Grid Extraction Test
===========================

Test direct grid extraction to understand the structure and verify it works.
"""

import sys
sys.path.insert(0, 'Jorg/src')

import h5py
import numpy as np
import jax.numpy as jnp
from jorg.atmosphere_jax import create_atmosphere_from_quantities
from jorg.atmosphere import call_korg_interpolation

def test_direct_grid_extraction():
    """Test extracting atmosphere directly from grid without interpolation"""
    
    print("TESTING DIRECT GRID EXTRACTION")
    print("=" * 40)
    
    # Load SDSS grid
    with h5py.File('Jorg/data/marcs_grids/SDSS_MARCS_atmospheres.h5', 'r') as f:
        grid = f['grid'][:]
        nodes = []
        for i in range(1, 6):  # grid_values/1 through grid_values/5
            nodes.append(f[f'grid_values/{i}'][:])
        
        param_names = [name.decode('utf-8') if isinstance(name, bytes) else name 
                      for name in f['grid_parameter_names'][:]]
    
    print(f"Grid shape: {grid.shape}")
    print(f"Parameters: {param_names}")
    
    # Test solar parameters - find exact grid point
    Teff, logg, m_H, alpha_m, C_m = 5750.0, 4.5, 0.0, 0.0, 0.0  # Use exact grid values
    
    # Find exact indices
    Teff_idx = np.where(nodes[0] == Teff)[0][0]
    logg_idx = np.where(nodes[1] == logg)[0][0] 
    mH_idx = np.where(nodes[2] == m_H)[0][0]
    alpha_idx = np.where(nodes[3] == alpha_m)[0][0]
    C_idx = np.where(nodes[4] == C_m)[0][0]
    
    print(f"\nExact grid indices:")
    print(f"Teff={Teff}: idx {Teff_idx}")
    print(f"logg={logg}: idx {logg_idx}")
    print(f"mH={m_H}: idx {mH_idx}")
    print(f"alpha={alpha_m}: idx {alpha_idx}")
    print(f"C={C_m}: idx {C_idx}")
    
    # Extract atmosphere data
    # Grid order: [carbon, alpha, metallicity, logg, Teff, quantities, layers]
    atm_data = grid[C_idx, alpha_idx, mH_idx, logg_idx, Teff_idx, :, :]
    print(f"\nExtracted atmosphere shape: {atm_data.shape}")
    
    # Transpose to [layers, quantities]
    atm_data_t = atm_data.T
    print(f"Transposed shape: {atm_data_t.shape}")
    
    # Convert to JAX array and create atmosphere
    atm_quants = jnp.array(atm_data_t)
    jax_atm = create_atmosphere_from_quantities(atm_quants, spherical=False, logg=logg)
    
    print(f"\nJAX atmosphere: {len(jax_atm.layers)} layers")
    
    # Get Korg reference for comparison
    korg_atm = call_korg_interpolation(Teff, logg, m_H)
    print(f"Korg atmosphere: {len(korg_atm.layers)} layers")
    
    # Compare a few layers
    print(f"\nComparison (first few layers):")
    for i in [10, 25, 40]:
        if i < len(jax_atm.layers) and i < len(korg_atm.layers):
            j_layer = jax_atm.layers[i]
            k_layer = korg_atm.layers[i]
            
            temp_diff = abs(j_layer.temp - k_layer.temp) / k_layer.temp * 100
            tau_diff = abs(j_layer.tau_5000 - k_layer.tau_5000) / k_layer.tau_5000 * 100
            
            print(f"Layer {i}: JAX T={j_layer.temp:.1f}K tau={j_layer.tau_5000:.3e}")
            print(f"        Korg T={k_layer.temp:.1f}K tau={k_layer.tau_5000:.3e}")
            print(f"        Diff: T={temp_diff:.2f}% tau={tau_diff:.2f}%")
    
    return jax_atm, korg_atm

if __name__ == "__main__":
    jax_atm, korg_atm = test_direct_grid_extraction()