#!/usr/bin/env python3
"""
Jorg Unit Test 5: Line Opacity Calculation
Tests line opacity with VALD linelist using main synthesis.py module.
"""

import sys
import numpy as np
import time
sys.path.insert(0, '../../src')

print("JORG UNIT TEST 5: LINE OPACITY CALCULATION")
print("=" * 50)

# Import required modules
try:
    from jorg.synthesis import synth
    from jorg.lines.linelist import read_linelist
    print("✅ Modules imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Load VALD linelist
vald_path = "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
try:
    start_time = time.time()
    linelist = read_linelist(vald_path, format='vald')
    load_time = time.time() - start_time
    print(f"✅ VALD linelist loaded: {len(linelist)} lines ({load_time:.2f}s)")
except Exception as e:
    print(f"❌ Could not load VALD linelist: {e}")
    linelist = None

# Test parameters
Teff, logg, m_H = 5780.0, 4.44, 0.0
wavelengths = (5000.0, 5010.0)  # 10 Å range
print(f"Test parameters: Teff={Teff}K, logg={logg}, [M/H]={m_H}, λ={wavelengths[0]}-{wavelengths[1]} Å")

# Run synthesis with lines using simple synth() API
if linelist:
    try:
        print("Running synthesis with line opacity...")
        start_time = time.time()
        
        wl, flux, continuum = synth(
            Teff=Teff,
            logg=logg,
            m_H=m_H,
            wavelengths=wavelengths,
            linelist=linelist,
            hydrogen_lines=False,
            verbose=False,
            rectify=False  # Get raw flux for analysis
        )
        
        synth_time = time.time() - start_time
        print(f"✅ Synthesis completed ({synth_time:.2f}s)")
        print(f"   Wavelengths: {len(wl)} points")
        print(f"   Flux range: {flux.min():.2e} - {flux.max():.2e} erg/s/cm²/Å")
        print(f"   Continuum range: {continuum.min():.2e} - {continuum.max():.2e} erg/s/cm²/Å")
        
        # Analyze line depths
        flux_ratio = flux / np.maximum(continuum, 1e-10)
        max_line_depth = 1.0 - flux_ratio.min()
        
        print(f"   Maximum line depth: {max_line_depth:.1%}")
        print(f"   Points with >1% absorption: {np.sum(flux_ratio < 0.99)}")
        
        # Calculate line opacity (difference from continuum)
        line_opacity = continuum - flux
        
        # Save data for comparison
        np.savetxt("jorg_line_opacity_data.txt", np.column_stack([wl, line_opacity]), 
                  header="Wavelength(A) Line_Opacity(erg/s/cm2/A)", fmt="%.6e")
        print(f"   Saved opacity data to jorg_line_opacity_data.txt")
        
        # Validation checks
        checks = [
            ("Flux positive", np.all(flux >= 0)),
            ("Continuum positive", np.all(continuum >= 0)),
            ("Realistic flux", 1e14 <= np.max(flux) <= 1e17),
            ("Line absorption active", max_line_depth > 0.01),
            ("Performance good", synth_time < 15.0)
        ]
        
        print("\nValidation:")
        all_passed = True
        for check_name, passed in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {check_name:20s}: {status}")
            all_passed = all_passed and passed
        
        if all_passed:
            print("\n🎉 LINE OPACITY CALCULATION: PRODUCTION READY")
        else:
            print("\n⚠️  LINE OPACITY CALCULATION: NEEDS ATTENTION")
            
    except Exception as e:
        print(f"❌ Synthesis failed: {e}")
else:
    print("⚠️  Cannot test without VALD linelist")

print("Test complete!")