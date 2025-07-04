#!/usr/bin/env python3
"""
Examine MARCS Grid Structure
===========================

Inspect the structure and content of MARCS atmosphere grid files
to understand their format for JAX implementation.
"""

import h5py
import numpy as np
from pathlib import Path

def examine_grid_file(filepath):
    """Examine the structure of an HDF5 grid file"""
    
    print(f"\n{'='*60}")
    print(f"EXAMINING: {Path(filepath).name}")
    print(f"{'='*60}")
    
    with h5py.File(filepath, 'r') as f:
        print(f"File size: {Path(filepath).stat().st_size / 1024**2:.1f} MB")
        
        # List all keys
        print(f"\nTop-level keys:")
        for key in f.keys():
            print(f"  - {key}")
        
        # Examine grid
        if 'grid' in f:
            grid = f['grid']
            print(f"\nGrid dataset:")
            print(f"  Shape: {grid.shape}")
            print(f"  Dtype: {grid.dtype}")
            print(f"  Size: {grid.size * grid.dtype.itemsize / 1024**2:.1f} MB")
            
            # Sample some values
            if grid.size > 0:
                print(f"  Value range: {np.nanmin(grid[:])} to {np.nanmax(grid[:])}")
                print(f"  NaN count: {np.sum(np.isnan(grid[:]))}")
        
        # Examine parameter names
        if 'grid_parameter_names' in f:
            param_names = f['grid_parameter_names'][:]
            print(f"\nParameter names:")
            for i, name in enumerate(param_names):
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                print(f"  {i}: {name}")
        
        # Examine grid values
        print(f"\nGrid node values:")
        i = 1
        while f'grid_values/{i}' in f:
            values = f[f'grid_values/{i}'][:]
            param_name = param_names[i-1] if 'grid_parameter_names' in f else f"param_{i}"
            if isinstance(param_name, bytes):
                param_name = param_name.decode('utf-8')
            print(f"  {param_name}: {len(values)} nodes, range [{values[0]:.3f}, {values[-1]:.3f}]")
            i += 1
        
        # Additional datasets
        other_keys = [key for key in f.keys() if key not in ['grid', 'grid_parameter_names'] and not key.startswith('grid_values')]
        if other_keys:
            print(f"\nOther datasets:")
            for key in other_keys:
                dataset = f[key]
                if hasattr(dataset, 'shape'):
                    print(f"  {key}: shape {dataset.shape}, dtype {dataset.dtype}")
                else:
                    print(f"  {key}: {type(dataset)}")


def main():
    """Examine all MARCS grid files"""
    
    print("MARCS ATMOSPHERE GRID EXAMINATION")
    print("="*60)
    
    grid_dir = Path("Jorg/data/marcs_grids")
    
    if not grid_dir.exists():
        print(f"❌ Grid directory not found: {grid_dir}")
        return
    
    grid_files = [
        "SDSS_MARCS_atmospheres.h5",
        "MARCS_metal_poor_atmospheres.h5", 
        "resampled_cool_dwarf_atmospheres.h5"
    ]
    
    for filename in grid_files:
        filepath = grid_dir / filename
        if filepath.exists():
            try:
                examine_grid_file(filepath)
            except Exception as e:
                print(f"❌ Error examining {filename}: {e}")
        else:
            print(f"❌ File not found: {filename}")
    
    print(f"\n{'='*60}")
    print("EXAMINATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()