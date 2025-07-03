#!/usr/bin/env python3
"""
Minimal test to isolate synthesis issues
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import jax.numpy as jnp

def test_imports():
    print("Testing imports...")
    try:
        from jorg.synthesis import format_abundances, interpolate_atmosphere
        print("✅ Core functions imported")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_abundance_formatting():
    print("Testing abundance formatting...")
    try:
        from jorg.synthesis import format_abundances
        A_X = format_abundances(m_H=0.0)
        print(f"✅ Abundances: {len(A_X)} elements")
        return True, A_X
    except Exception as e:
        print(f"❌ Abundance formatting failed: {e}")
        return False, None

def test_atmosphere():
    print("Testing atmosphere interpolation...")
    try:
        from jorg.synthesis import format_abundances, interpolate_atmosphere
        A_X = format_abundances(m_H=0.0)
        atm = interpolate_atmosphere(5778, 4.44, A_X)
        print(f"✅ Atmosphere: {atm['n_layers']} layers")
        return True, atm
    except Exception as e:
        print(f"❌ Atmosphere failed: {e}")
        return False, None

if __name__ == "__main__":
    print("🔍 Minimal synthesis diagnostics")
    
    # Test 1: Imports
    if not test_imports():
        sys.exit(1)
    
    # Test 2: Abundances  
    success, A_X = test_abundance_formatting()
    if not success:
        sys.exit(1)
    
    # Test 3: Atmosphere
    success, atm = test_atmosphere()
    if not success:
        sys.exit(1)
        
    print("🎉 Core components working!")
    print("Issue may be in synthesis or continuum calculation")