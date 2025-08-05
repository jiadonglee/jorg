#!/usr/bin/env python3
"""
Test Line Amplitude Fix

Quick test to see if the line amplitude calculation fixes are working.
"""

import sys
import numpy as np
sys.path.insert(0, 'src')

print("🧪 TESTING LINE AMPLITUDE FIXES")
print("=" * 50)

try:
    from jorg.synthesis import synth
    
    # Simple test with basic parameters
    print("Running simple continuum-only synthesis...")
    wl, flux, cntm = synth(
        Teff=5780, logg=4.44, m_H=0.0,
        wavelengths=(5000, 5010),
        linelist=None,  # No lines - continuum only
        hydrogen_lines=False,
        rectify=False
    )
    
    print(f"✅ Continuum-only synthesis: {len(wl)} points")
    print(f"   Flux range: {flux.min():.2e} - {flux.max():.2e}")
    print(f"   Continuum range: {cntm.min():.2e} - {cntm.max():.2e}")
    print(f"   Flux/continuum variation: {(flux/cntm).std():.6f}")
    
    # Test with VALD lines
    print(f"\nRunning synthesis with VALD lines...")
    from jorg.lines.linelist import read_linelist
    
    vald_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
    linelist = read_linelist(vald_path, format='vald')
    print(f"Loaded {len(linelist)} VALD lines")
    
    # Synthesis with lines
    wl_lines, flux_lines, cntm_lines = synth(
        Teff=5780, logg=4.44, m_H=0.0,
        wavelengths=(5000, 5010),
        linelist=linelist,
        hydrogen_lines=False,
        rectify=False,
        verbose=True  # Enable verbose output to see KorgLineProcessor
    )
    
    print(f"✅ Line synthesis complete: {len(wl_lines)} points")
    print(f"   Flux range: {flux_lines.min():.2e} - {flux_lines.max():.2e}")
    print(f"   Continuum range: {cntm_lines.min():.2e} - {cntm_lines.max():.2e}")
    
    # Calculate line depths
    flux_ratio = flux_lines / np.maximum(cntm_lines, 1e-10)
    max_line_depth = 1.0 - flux_ratio.min()
    points_with_absorption = np.sum(flux_ratio < 0.99)
    strong_absorption = np.sum(flux_ratio < 0.95)
    
    print(f"   Max line depth: {max_line_depth:.1%}")
    print(f"   Points with >1% absorption: {points_with_absorption}")
    print(f"   Points with >5% absorption: {strong_absorption}")
    
    # Success criteria
    if max_line_depth > 0.01:  # At least 1% line depth
        print(f"\n✅ SUCCESS: Line amplitude fixes are working!")
        print(f"   Achieved {max_line_depth:.1%} line depth (target: >1%)")
    elif max_line_depth > 0.001:  # Some improvement
        print(f"\n⚠️ PARTIAL SUCCESS: Some line absorption detected")
        print(f"   Achieved {max_line_depth:.1%} line depth (still need improvement)")
    else:
        print(f"\n❌ FAILURE: Still no significant line absorption")
        print(f"   Max line depth: {max_line_depth:.1%} (target: >1%)")
        print(f"   Need to investigate further...")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 50)
print("LINE AMPLITUDE FIX TEST COMPLETE")