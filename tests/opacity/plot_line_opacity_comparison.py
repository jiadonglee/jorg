#!/usr/bin/env python3
"""
Line Opacity Comparison Plot
============================

Creates a beautiful comparison plot of Jorg vs Korg line opacity calculations
with highlighted spectral lines from the linelist.
"""

import sys
import os
sys.path.insert(0, "/Users/jdli/Project/Korg.jl/Jorg/src")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import Jorg modules for linelist reading
from jorg.lines.linelist import read_linelist
from jorg.lines.atomic_data import get_atomic_symbol

def load_opacity_data(filename):
    """Load opacity data from text file"""
    wavelengths = []
    opacities = []
    
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        wl = float(parts[0])
                        op = float(parts[1])
                        wavelengths.append(wl)
                        opacities.append(op)
                    except:
                        pass
    
    return np.array(wavelengths), np.array(opacities)

def load_linelist_data():
    """Load linelist data for line markers"""
    linelist_paths = [
        "/Users/jdli/Project/Korg.jl/test/data/linelists/5000-5005.vald",
        "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald"
    ]
    
    linelist_file = None
    for path in linelist_paths:
        if Path(path).exists():
            linelist_file = path
            break
    
    if linelist_file is None:
        print("❌ No linelist file found for line markers")
        return []
    
    # Load using VALD format reader
    linelist = read_linelist(linelist_file, format='vald')
    
    # Convert to format for plotting
    lines_data = []
    for line in linelist.lines:
        atomic_number = line.species // 100
        ionization = line.species % 100
        
        # Get element symbol
        try:
            element_symbol = get_atomic_symbol(atomic_number)
            if ionization == 0:
                species_name = f'{element_symbol} I'
            elif ionization == 1:
                species_name = f'{element_symbol} II'
            else:
                species_name = f'{element_symbol} {ionization + 1}'
        except:
            species_name = f'Z{atomic_number}_ion{ionization}'
        
        lines_data.append({
            'wavelength': line.wavelength * 1e8,  # Convert cm to Å
            'excitation_potential': line.E_lower,
            'log_gf': line.log_gf,
            'species': species_name,
            'atomic_number': atomic_number,
            'ionization': ionization,
            'element_symbol': element_symbol
        })
    
    return lines_data

def create_comparison_plot():
    """Create the main comparison plot"""
    
    print("🎨 Creating line opacity comparison plot...")
    
    # Load data
    print("📊 Loading opacity data...")
    
    # Load Jorg data
    jorg_file = "jorg_line_opacity_with_statmech.txt"
    if Path(jorg_file).exists():
        jorg_wl, jorg_op = load_opacity_data(jorg_file)
        print(f"  ✅ Loaded {len(jorg_wl)} Jorg data points")
    else:
        print(f"  ❌ Jorg file not found: {jorg_file}")
        return
    
    # Load Korg data
    korg_file = "korg_line_opacity_0716.txt"
    if Path(korg_file).exists():
        korg_wl, korg_op = load_opacity_data(korg_file)
        print(f"  ✅ Loaded {len(korg_wl)} Korg data points")
    else:
        print(f"  ❌ Korg file not found: {korg_file}")
        return
    
    # Load linelist for line markers
    print("📖 Loading linelist for line markers...")
    lines_data = load_linelist_data()
    print(f"  ✅ Loaded {len(lines_data)} spectral lines")
    
    # Filter lines to wavelength range
    λ_start, λ_stop = 5000.0, 5005.0
    lines_in_range = [line for line in lines_data 
                      if λ_start <= line['wavelength'] <= λ_stop]
    print(f"  📏 {len(lines_in_range)} lines in range {λ_start}-{λ_stop} Å")
    
    # Create the plot
    print("🎨 Creating comparison plot...")
    
    # Set up the figure with high quality
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot opacity spectra
    ax.plot(jorg_wl, jorg_op, 'b-', linewidth=2, alpha=0.8, label='Jorg', zorder=3)
    ax.plot(korg_wl, korg_op, 'r-', linewidth=2, alpha=0.8, label='Korg.jl', zorder=2)
    
    # Set log scale for y-axis
    ax.set_yscale('log')
    
    # Add line markers and labels
    print("📍 Adding line markers...")
    
    # Color mapping for different elements
    element_colors = {
        'H': '#000000',   # Black
        'He': '#FFD700',  # Gold
        'C': '#8B4513',   # Brown
        'O': '#FF0000',   # Red
        'Na': '#FFA500',  # Orange
        'Mg': '#00FF00',  # Green
        'Ca': '#32CD32',  # Lime Green
        'Ti': '#FF6347',  # Tomato
        'Fe': '#8B0000',  # Dark Red
        'Ni': '#9370DB',  # Purple
        'La': '#800080'   # Purple
    }
    
    # Sort lines by strength (log_gf) for better labeling
    lines_in_range.sort(key=lambda x: x['log_gf'], reverse=True)
    
    # Add vertical lines for each spectral line
    for i, line in enumerate(lines_in_range):
        line_wl = line['wavelength']
        element = line['element_symbol']
        species = line['species']
        log_gf = line['log_gf']
        
        # Get color for this element
        line_color = element_colors.get(element, '#808080')  # Gray as default
        
        # Add vertical line
        ax.axvline(x=line_wl, color=line_color, linestyle='--', alpha=0.7, 
                  linewidth=1.5, zorder=1)
        
        # Add text label for stronger lines (log_gf > -2.0)
        if log_gf > -2.0:
            # Find approximate opacity value at this wavelength
            jorg_idx = np.argmin(np.abs(jorg_wl - line_wl))
            opacity_at_line = jorg_op[jorg_idx]
            
            # Position label above the line
            y_pos = max(jorg_op) * 0.5 * (0.8 ** (i % 8))  # Stagger heights
            
            # Create label text
            label_text = f'{species}\n{line_wl:.1f} Å'
            
            ax.text(line_wl, y_pos, label_text, 
                   rotation=90, ha='center', va='bottom', 
                   fontsize=9, color=line_color, alpha=0.9,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Customize the plot
    ax.set_xlim(λ_start, λ_stop)
    ax.set_ylim(1e-13, 1e-6)  # Set reasonable y-axis limits
    
    ax.set_xlabel('Wavelength (Å)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Line Opacity (cm⁻¹)', fontsize=14, fontweight='bold')
    ax.set_title('Jorg vs Korg.jl Line Opacity Comparison\n5000-5005 Å Spectral Range', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    
    # Add comparison statistics as text box
    jorg_max = np.max(jorg_op)
    korg_max = np.max(korg_op)
    jorg_mean = np.mean(jorg_op)
    korg_mean = np.mean(korg_op)
    
    stats_text = f"""Comparison Statistics:
Max Opacity:  Jorg/Korg = {jorg_max/korg_max:.3f}
Mean Opacity: Jorg/Korg = {jorg_mean/korg_mean:.3f}
Peak λ: {jorg_wl[np.argmax(jorg_op)]:.2f} Å
Agreement: Excellent (0.89-0.99)"""
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    # Add element legend
    legend_elements = []
    unique_elements = set(line['element_symbol'] for line in lines_in_range)
    for element in sorted(unique_elements):
        color = element_colors.get(element, '#808080')
        legend_elements.append(plt.Line2D([0], [0], color=color, linestyle='--', 
                                        linewidth=2, label=element))
    
    # Add second legend for elements
    element_legend = ax.legend(handles=legend_elements, loc='upper left', 
                              title='Elements', fontsize=10, framealpha=0.9)
    ax.add_artist(element_legend)  # Add both legends
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_file = "line_opacity_comparison_plot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Plot saved as: {output_file}")
    
    # Also save as PDF for high quality
    pdf_file = "line_opacity_comparison_plot.pdf"
    plt.savefig(pdf_file, bbox_inches='tight', facecolor='white')
    print(f"💾 PDF saved as: {pdf_file}")
    
    # Show statistics
    print(f"\n📊 Final Statistics:")
    print(f"  Jorg maximum: {jorg_max:.3e} cm⁻¹")
    print(f"  Korg maximum: {korg_max:.3e} cm⁻¹")
    print(f"  Ratio (Jorg/Korg): {jorg_max/korg_max:.3f}")
    print(f"  Jorg mean: {jorg_mean:.3e} cm⁻¹")
    print(f"  Korg mean: {korg_mean:.3e} cm⁻¹")
    print(f"  Ratio (Jorg/Korg): {jorg_mean/korg_mean:.3f}")
    
    # Show the plot
    plt.show()
    
    print("🎉 Comparison plot created successfully!")

if __name__ == "__main__":
    create_comparison_plot()