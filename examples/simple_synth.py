#!/usr/bin/env python3
"""
Simple Jorg Synthesis Script
============================

The most basic way to run stellar synthesis with Jorg.
Perfect for quick tests and getting started.
"""

import sys
from pathlib import Path

# Add Jorg to path  
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jorg.synthesis import synth

def simple_synthesis():
    """Run the simplest possible stellar synthesis"""
    
    print("🚀 Simple Jorg Synthesis")
    print("=" * 30)
    
    # Basic solar synthesis
    wavelengths, flux, continuum = synth(
        Teff=5780,              # Solar temperature
        logg=4.44,              # Solar gravity  
        m_H=0.0,                # Solar metallicity
        wavelengths=(5000, 5100) # 100 Å range
    )
    
    print(f"✅ Synthesis complete!")
    print(f"   Wavelengths: {len(wavelengths)}")
    print(f"   Flux range: {flux.min():.3f} - {flux.max():.3f}")
    print(f"   Continuum range: {continuum.min():.3f} - {continuum.max():.3f}")
    
    # Show some sample values
    print(f"\n📊 Sample spectrum:")
    for i in range(0, len(wavelengths), len(wavelengths)//5):
        wl = wavelengths[i]
        fl = flux[i] 
        ct = continuum[i]
        print(f"   {wl:7.1f} Å: flux={fl:.4f}, continuum={ct:.4f}")
    
    return wavelengths, flux, continuum

if __name__ == "__main__":
    wl, flux, cont = simple_synthesis()
    print(f"\n🎉 Success! Jorg is working perfectly.")