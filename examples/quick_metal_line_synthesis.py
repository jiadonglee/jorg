#!/usr/bin/env python3
"""
Quick Jorg Metal Line Synthesis Demo
====================================

Demonstrates Jorg's metal line synthesis capabilities with realistic line depths.
This script shows the major breakthrough achieved in July 2025.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add Jorg to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jorg.synthesis import synth
from jorg.lines.linelist import read_linelist

def quick_synthesis_demo():
    """Run a quick metal line synthesis demonstration"""
    
    print("🚀 JORG METAL LINE SYNTHESIS DEMO")
    print("=" * 50)
    
    # Solar parameters
    Teff = 5780  # K
    logg = 4.44  # log(g)
    m_H = 0.0    # Solar metallicity
    
    # Metal-rich wavelength region
    wavelength_range = (5000, 5010)  # Angstroms
    
    print(f"📊 Stellar Parameters:")
    print(f"   Teff = {Teff} K")
    print(f"   log g = {logg}")
    print(f"   [M/H] = {m_H}")
    print(f"   Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} Å")
    
    # Synthesis WITHOUT metal lines (continuum only)
    print(f"\n🌟 Synthesis without metal lines...")
    wl_cont, flux_cont, cont_cont = synth(
        Teff=Teff, logg=logg, m_H=m_H,
        wavelengths=wavelength_range,
        linelist=None  # No lines
    )
    
    # Try to load VALD linelist for metal lines
    vald_path = Path(__file__).parent.parent.parent / "data" / "linelists" / "vald_extract_stellar_solar_threshold001.vald"
    
    if vald_path.exists():
        print(f"📖 Loading VALD linelist: {vald_path.name}")
        linelist = read_linelist(str(vald_path))
        
        # Filter to our wavelength range
        lines_in_range = linelist.filter_by_wavelength(wavelength_range[0], wavelength_range[1])
        print(f"   Found {len(lines_in_range.lines)} lines in range")
        
        # Synthesis WITH metal lines
        print(f"\n🌟 Synthesis with metal lines...")
        wl_lines, flux_lines, cont_lines = synth(
            Teff=Teff, logg=logg, m_H=m_H,
            wavelengths=wavelength_range,
            linelist=lines_in_range
        )
        
        # Analyze results
        print(f"\n📊 RESULTS:")
        print(f"   Continuum flux range: {flux_cont.min():.3f} - {flux_cont.max():.3f}")
        print(f"   With lines flux range: {flux_lines.min():.3f} - {flux_lines.max():.3f}")
        
        # Calculate line depths
        line_depths = (cont_lines - flux_lines) / cont_lines * 100
        max_depth = np.max(line_depths)
        max_idx = np.argmax(line_depths)
        strongest_wl = wl_lines[max_idx]
        
        print(f"   Line depths: {line_depths.min():.1f}% - {line_depths.max():.1f}%")
        print(f"   Strongest line: {strongest_wl:.2f} Å with {max_depth:.1f}% depth")
        
        if max_depth > 10:
            print(f"   ✅ STRONG METAL LINES DETECTED!")
        elif max_depth > 1:
            print(f"   ⚠️  Weak but detectable lines")
        else:
            print(f"   ❌ Lines too weak")
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(wl_cont, flux_cont, 'b-', label='Continuum only', linewidth=2)
        plt.plot(wl_lines, flux_lines, 'r-', label='With metal lines', linewidth=1)
        plt.xlabel('Wavelength [Å]')
        plt.ylabel('Normalized Flux')
        plt.title('Jorg Metal Line Synthesis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(wl_lines, line_depths, 'g-', linewidth=2)
        plt.xlabel('Wavelength [Å]')
        plt.ylabel('Line Depth [%]')
        plt.title('Metal Line Depths')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(__file__).parent / "jorg_metal_lines_demo.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"   📈 Plot saved: {output_path}")
        
        plt.show()
        
    else:
        print(f"⚠️  VALD linelist not found at {vald_path}")
        print(f"   Running continuum-only synthesis")
        
        print(f"\n📊 CONTINUUM RESULTS:")
        print(f"   Wavelengths: {len(wl_cont)}")
        print(f"   Flux range: {flux_cont.min():.3f} - {flux_cont.max():.3f}")
        print(f"   ✅ Continuum synthesis working")
    
    print(f"\n🎉 Jorg synthesis complete!")
    print(f"💡 TIP: Add a VALD linelist to see spectacular metal line absorption!")

if __name__ == "__main__":
    quick_synthesis_demo()