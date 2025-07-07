#!/usr/bin/env python3
"""
Debug H I partition function to fix the factor 53 error in H⁻ free-free
"""

import sys
from pathlib import Path
jorg_src = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_src))

import numpy as np
from jorg.statmech.partition_functions import partition_function

print("H I PARTITION FUNCTION DEBUG")
print("=" * 32)

# Test conditions
T = 4838.3  # K

print(f"Temperature: {T} K")
print()

# Calculate H I partition function
try:
    # H I has Z=1, charge=0
    U_H_I = partition_function(1, 0, T)  # (Z, charge, T)
    print(f"H I partition function: {U_H_I:.3f}")
    
    # What Korg uses vs what we should use
    korg_assumes_U = 2.0
    actual_U = U_H_I
    
    print(f"Korg typically assumes: {korg_assumes_U}")
    print(f"Our calculated U_H_I: {actual_U:.3f}")
    print(f"Ratio (our/Korg): {actual_U / korg_assumes_U:.3f}")
    print()
    
    # This means our nH_I_div_partition should be:
    n_HI = 2.5e16  # cm⁻³ (test value)
    nH_I_div_partition_correct = n_HI / actual_U
    nH_I_div_partition_assumed = n_HI / korg_assumes_U
    
    print(f"PARTITION FUNCTION CORRECTION:")
    print(f"Total H I density: {n_HI:.2e} cm⁻³")
    print(f"nH_I_div_partition (correct): {nH_I_div_partition_correct:.2e} cm⁻³")
    print(f"nH_I_div_partition (assumed): {nH_I_div_partition_assumed:.2e} cm⁻³")
    print()
    
    # Ground state densities
    nHI_gs_correct = 2 * nH_I_div_partition_correct
    nHI_gs_assumed = 2 * nH_I_div_partition_assumed
    
    print(f"Ground state H I (correct): {nHI_gs_correct:.2e} cm⁻³")
    print(f"Ground state H I (assumed): {nHI_gs_assumed:.2e} cm⁻³")
    print(f"Correction factor: {nHI_gs_correct / nHI_gs_assumed:.1f}")
    print()
    
    # Check if this explains our factor 53 discrepancy
    if abs(nHI_gs_correct / nHI_gs_assumed - 53) < 10:
        print("🎯 FOUND THE BUG! Partition function accounts for the factor 53!")
    elif nHI_gs_correct / nHI_gs_assumed > 10:
        print("🔧 MAJOR FACTOR: Partition function is a large part of the discrepancy")
    elif nHI_gs_correct / nHI_gs_assumed > 2:
        print("⚠️  MODERATE FACTOR: Partition function helps but doesn't fully explain it")
    else:
        print("❌ SMALL FACTOR: Partition function doesn't explain the discrepancy")
        
except Exception as e:
    print(f"Error calculating partition function: {e}")
    print("Using approximate values...")
    
    # For H I at ~5000K, partition function is typically 2-8
    estimated_U_values = [2.0, 4.0, 6.0, 8.0, 10.0]
    
    print("ESTIMATED PARTITION FUNCTION EFFECTS:")
    n_HI = 2.5e16
    for U_est in estimated_U_values:
        nH_I_div_partition = n_HI / U_est
        nHI_gs = 2 * nH_I_div_partition
        factor = nHI_gs / n_HI  # vs using n_HI directly
        print(f"  U = {U_est:.1f}: nHI_gs = {nHI_gs:.2e}, factor = {factor:.2f}")
    
    print()
    print("If factor ~25-50 from one of these, that explains our discrepancy!")