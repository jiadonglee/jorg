"""
Stellar Types Comparison: Jorg vs Korg.jl Continuum Calculations

This script runs comprehensive comparisons between Jorg and Korg.jl continuum
calculations across different stellar types, varying:
- Effective temperature (Teff): 3000K - 10000K  
- Surface gravity (log g): 0.0 - 5.0
- Metallicity ([M/H]): -3.0 to +0.5
- Electron density: Calculated from stellar parameters

The script automatically:
1. Generates Korg.jl reference calculations for each stellar type
2. Calculates Jorg results using the new high-level APIs
3. Compares results and generates detailed reports
4. Creates visualizations showing agreement across stellar parameter space

Usage:
    python stellar_types_comparison.py --mode all
    python stellar_types_comparison.py --mode single --teff 5780 --logg 4.4 --mh 0.0
    python stellar_types_comparison.py --mode grid --save-plots
"""

import sys
import os
import argparse
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add Jorg to path
sys.path.insert(0, "/Users/jdli/Project/Korg.jl/Jorg/src")

import jax.numpy as jnp
from jorg.continuum import H_I_bf
from jorg.continuum.mclaughlin_hminus import mclaughlin_hminus_bf_absorption
from jorg.continuum.hydrogen import h_minus_ff_absorption
from jorg.statmech.partition_functions import create_default_partition_functions
from jorg.statmech.species import Species


class StellarType:
    """Represents a stellar type with physical parameters"""
    def __init__(self, name: str, teff: float, logg: float, mh: float, 
                 n_h_i: Optional[float] = None, n_he_i: Optional[float] = None, 
                 electron_density: Optional[float] = None):
        self.name = name
        self.teff = teff  # K
        self.logg = logg  # log10(g [cm/s²])
        self.mh = mh      # [M/H] metallicity
        
        # Calculate typical densities if not provided
        if n_h_i is None:
            self.n_h_i = self._calculate_h_density()
        else:
            self.n_h_i = n_h_i
            
        if n_he_i is None:
            self.n_he_i = self.n_h_i * 0.1  # Typical He/H ratio
        else:
            self.n_he_i = n_he_i
            
        if electron_density is None:
            self.electron_density = self._calculate_electron_density()
        else:
            self.electron_density = electron_density
    
    def _calculate_h_density(self) -> float:
        """Calculate typical H I density from stellar parameters"""
        # Rough approximation based on stellar atmosphere models
        # Lower temperature and higher gravity → higher density
        log_density = 16.0 + (5780 - self.teff) / 1000.0 + (self.logg - 4.4) * 0.5
        return 10**log_density
    
    def _calculate_electron_density(self) -> float:
        """Calculate electron density from ionization equilibrium"""
        # Rough approximation: higher T → more ionization
        ionization_fraction = min(0.9, max(0.001, (self.teff - 3000) / 7000))
        return self.n_h_i * ionization_fraction
    
    def __str__(self):
        return f"{self.name}: Teff={self.teff}K, log g={self.logg}, [M/H]={self.mh}"


# Predefined stellar types for comprehensive testing
STELLAR_TYPES = {
    'sun': StellarType('Sun', 5780, 4.44, 0.0),
    'k_dwarf': StellarType('K Dwarf', 4500, 4.5, 0.0),
    'g_dwarf': StellarType('G Dwarf', 6000, 4.3, 0.0),
    'f_dwarf': StellarType('F Dwarf', 6750, 4.2, 0.0),
    'm_dwarf': StellarType('M Dwarf', 3500, 4.8, 0.0),
    'a_dwarf': StellarType('A Star', 8000, 4.0, 0.0),
    'k_giant': StellarType('K Giant', 4200, 2.0, 0.0),
    'g_giant': StellarType('G Giant', 5200, 2.5, 0.0),
    'metal_poor': StellarType('Metal Poor', 5780, 4.44, -2.0),
    'metal_rich': StellarType('Metal Rich', 5780, 4.44, +0.3),
    'hot_giant': StellarType('Hot Giant', 7000, 1.5, 0.0),
    'cool_dwarf': StellarType('Cool Dwarf', 3200, 5.0, 0.0)
}

# Subset for quick testing
QUICK_STELLAR_TYPES = {
    'sun': StellarType('Sun', 5780, 4.44, 0.0),
    'k_dwarf': StellarType('K Dwarf', 4500, 4.5, 0.0),
    'g_giant': StellarType('G Giant', 5200, 2.5, 0.0)
}


def run_korg_calculation(stellar_type: StellarType, frequencies: np.ndarray) -> Dict:
    """
    Run Korg.jl continuum calculation for a given stellar type
    """
    julia_code = f"""
    using Korg
    
    # Stellar parameters
    temperature = {stellar_type.teff}
    n_h_i = {stellar_type.n_h_i}
    n_he_i = {stellar_type.n_he_i}
    electron_density = {stellar_type.electron_density}
    
    # Frequencies
    frequencies_hz = {list(frequencies)}
    
    # H I partition function
    U_H_I = Korg.default_partition_funcs[Korg.species"H I"](log(temperature))
    n_h_i_div_u = n_h_i / U_H_I
    inv_u_h = 1.0 / U_H_I
    
    try
        # Calculate individual components
        alpha_hminus_bf = Korg.ContinuumAbsorption.Hminus_bf(
            frequencies_hz, temperature, n_h_i_div_u, electron_density
        )
        
        alpha_hminus_ff = Korg.ContinuumAbsorption.Hminus_ff(
            frequencies_hz, temperature, n_h_i_div_u, electron_density
        )
        
        alpha_h_i_bf = Korg.ContinuumAbsorption.H_I_bf(
            frequencies_hz, temperature, n_h_i, n_he_i, electron_density, inv_u_h
        )
        
        # Calculate total
        alpha_total = alpha_hminus_bf .+ alpha_hminus_ff .+ alpha_h_i_bf
        
        println("KORG_RESULTS_START")
        println("success=true")
        println("U_H_I=", U_H_I)
        println("inv_u_h=", inv_u_h)
        println("alpha_total=", alpha_total)
        println("alpha_hminus_bf=", alpha_hminus_bf)
        println("alpha_hminus_ff=", alpha_hminus_ff)
        println("alpha_h_i_bf=", alpha_h_i_bf)
        println("KORG_RESULTS_END")
        
    catch e
        println("KORG_RESULTS_START")
        println("success=false")
        println("error=", e)
        println("KORG_RESULTS_END")
    end
    """
    
    # Write and execute Julia script
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(julia_code)
        julia_file = f.name
    
    try:
        result = subprocess.run(
            ["julia", "--project=/Users/jdli/Project/Korg.jl", julia_file],
            capture_output=True, text=True, cwd="/Users/jdli/Project/Korg.jl",
            timeout=60  # 60 second timeout
        )
        
        if result.returncode != 0:
            print(f"❌ Julia error for {stellar_type.name}: {result.stderr}")
            return {'success': False, 'error': result.stderr}
        
        # Parse results
        output = result.stdout
        start_idx = output.find("KORG_RESULTS_START")
        end_idx = output.find("KORG_RESULTS_END")
        
        if start_idx == -1 or end_idx == -1:
            print(f"❌ Could not parse Korg output for {stellar_type.name}")
            return {'success': False, 'error': 'Could not parse output'}
        
        results_section = output[start_idx:end_idx]
        components = {}
        
        for line in results_section.split('\n'):
            if '=' in line and not line.startswith('KORG_RESULTS'):
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                if key == 'success':
                    components[key] = value == 'true'
                elif key == 'error':
                    components[key] = value
                elif key in ['U_H_I', 'inv_u_h']:
                    components[key] = float(value)
                else:
                    # Parse array values
                    array_str = value.strip('[]')
                    components[key] = [float(x.strip()) for x in array_str.split(',')]
        
        return components
        
    except subprocess.TimeoutExpired:
        print(f"❌ Korg calculation timeout for {stellar_type.name}")
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        print(f"❌ Korg calculation error for {stellar_type.name}: {e}")
        return {'success': False, 'error': str(e)}
    finally:
        os.unlink(julia_file)


def run_jorg_calculation(stellar_type: StellarType, frequencies: np.ndarray) -> Dict:
    """
    Run Jorg continuum calculation for a given stellar type using new APIs
    """
    try:
        frequencies_jax = jnp.array(frequencies)
        
        # Calculate H I partition function
        h_i_species = Species.from_atomic_number(1, 0)
        partition_funcs = create_default_partition_functions()
        U_H_I = partition_funcs[h_i_species](jnp.log(stellar_type.teff))
        n_h_i_div_u = stellar_type.n_h_i / U_H_I
        inv_u_h = 1.0 / U_H_I
        
        # Calculate individual components
        alpha_hminus_bf = mclaughlin_hminus_bf_absorption(
            frequencies=frequencies_jax,
            temperature=stellar_type.teff,
            n_h_i_div_u=float(n_h_i_div_u),
            electron_density=stellar_type.electron_density,
            include_stimulated_emission=True
        )
        
        alpha_hminus_ff = h_minus_ff_absorption(
            frequencies=frequencies_jax,
            temperature=stellar_type.teff,
            n_h_i_div_u=float(n_h_i_div_u),
            electron_density=stellar_type.electron_density
        )
        
        # Use new high-level API
        alpha_h_i_bf = H_I_bf(
            frequencies=frequencies_jax,
            temperature=stellar_type.teff,
            n_h_i=stellar_type.n_h_i,
            n_he_i=stellar_type.n_he_i,
            electron_density=stellar_type.electron_density,
            inv_u_h=float(inv_u_h)
        )
        
        alpha_total = alpha_hminus_bf + alpha_hminus_ff + alpha_h_i_bf
        
        return {
            'success': True,
            'U_H_I': float(U_H_I),
            'inv_u_h': float(inv_u_h),
            'alpha_total': [float(x) for x in alpha_total],
            'alpha_hminus_bf': [float(x) for x in alpha_hminus_bf],
            'alpha_hminus_ff': [float(x) for x in alpha_hminus_ff],
            'alpha_h_i_bf': [float(x) for x in alpha_h_i_bf]
        }
        
    except Exception as e:
        print(f"❌ Jorg calculation error for {stellar_type.name}: {e}")
        return {'success': False, 'error': str(e)}


def compare_results(korg_results: Dict, jorg_results: Dict, 
                   stellar_type: StellarType, frequencies: np.ndarray) -> Dict:
    """
    Compare Korg and Jorg results and calculate statistics
    """
    if not korg_results.get('success') or not jorg_results.get('success'):
        return {
            'success': False,
            'korg_error': korg_results.get('error'),
            'jorg_error': jorg_results.get('error')
        }
    
    # Calculate ratios and statistics
    korg_total = np.array(korg_results['alpha_total'])
    jorg_total = np.array(jorg_results['alpha_total'])
    
    ratios = jorg_total / np.where(korg_total != 0, korg_total, 1e-100)
    
    # Component comparisons
    components = ['alpha_hminus_bf', 'alpha_hminus_ff', 'alpha_h_i_bf']
    component_stats = {}
    
    for comp in components:
        korg_comp = np.array(korg_results[comp])
        jorg_comp = np.array(jorg_results[comp])
        comp_ratios = jorg_comp / np.where(korg_comp != 0, korg_comp, 1e-100)
        
        component_stats[comp] = {
            'mean_ratio': float(np.mean(comp_ratios)),
            'std_ratio': float(np.std(comp_ratios)),
            'min_ratio': float(np.min(comp_ratios)),
            'max_ratio': float(np.max(comp_ratios))
        }
    
    return {
        'success': True,
        'stellar_type': stellar_type.name,
        'frequencies': frequencies.tolist(),
        'korg_total': korg_total.tolist(),
        'jorg_total': jorg_total.tolist(),
        'ratios': ratios.tolist(),
        'mean_ratio': float(np.mean(ratios)),
        'std_ratio': float(np.std(ratios)),
        'min_ratio': float(np.min(ratios)),
        'max_ratio': float(np.max(ratios)),
        'agreement_percent': float((1.0 - abs(1.0 - np.mean(ratios))) * 100),
        'component_stats': component_stats,
        'partition_function_ratio': float(jorg_results['U_H_I'] / korg_results['U_H_I'])
    }


def run_single_comparison(stellar_type: StellarType, frequencies: np.ndarray, 
                         verbose: bool = True) -> Dict:
    """
    Run comparison for a single stellar type
    """
    if verbose:
        print(f"\n🌟 Testing {stellar_type.name}:")
        print(f"   {stellar_type}")
        print(f"   H I density: {stellar_type.n_h_i:.2e} cm⁻³")
        print(f"   Electron density: {stellar_type.electron_density:.2e} cm⁻³")
    
    start_time = time.time()
    
    # Run Korg calculation
    if verbose:
        print("   Running Korg.jl calculation...")
    korg_results = run_korg_calculation(stellar_type, frequencies)
    
    # Run Jorg calculation  
    if verbose:
        print("   Running Jorg calculation...")
    jorg_results = run_jorg_calculation(stellar_type, frequencies)
    
    # Compare results
    comparison = compare_results(korg_results, jorg_results, stellar_type, frequencies)
    
    elapsed = time.time() - start_time
    
    if comparison.get('success'):
        agreement = comparison['agreement_percent']
        if verbose:
            status = "🎯 EXCELLENT" if agreement > 99 else "✅ GOOD" if agreement > 95 else "⚠️  FAIR" if agreement > 90 else "❌ POOR"
            print(f"   Agreement: {agreement:.1f}% {status}")
            print(f"   Mean ratio: {comparison['mean_ratio']:.4f}")
            print(f"   Time: {elapsed:.2f}s")
        
        comparison['elapsed_time'] = elapsed
        return comparison
    else:
        if verbose:
            print(f"   ❌ FAILED")
            if comparison.get('korg_error'):
                print(f"      Korg error: {comparison['korg_error']}")
            if comparison.get('jorg_error'):
                print(f"      Jorg error: {comparison['jorg_error']}")
        return comparison


def run_stellar_grid(teff_range: Tuple[float, float] = (3000, 10000),
                    logg_range: Tuple[float, float] = (0.0, 5.0),
                    mh_range: Tuple[float, float] = (-2.0, 0.5),
                    n_points: int = 5) -> List[Dict]:
    """
    Run comparison across a grid of stellar parameters
    """
    print(f"\n🌌 STELLAR PARAMETER GRID COMPARISON")
    print(f"   Teff: {teff_range[0]}-{teff_range[1]}K ({n_points} points)")
    print(f"   log g: {logg_range[0]}-{logg_range[1]} ({n_points} points)")
    print(f"   [M/H]: {mh_range[0]}-{mh_range[1]} ({n_points} points)")
    print(f"   Total: {n_points**3} stellar types")
    
    frequencies = np.array([4e14, 5e14, 6e14, 7e14, 8e14, 1e15, 2e15, 3e15, 4e15, 5e15, 6e15])
    
    # Generate parameter grid
    teff_vals = np.linspace(teff_range[0], teff_range[1], n_points)
    logg_vals = np.linspace(logg_range[0], logg_range[1], n_points)
    mh_vals = np.linspace(mh_range[0], mh_range[1], n_points)
    
    results = []
    total_tests = len(teff_vals) * len(logg_vals) * len(mh_vals)
    test_count = 0
    
    for teff in teff_vals:
        for logg in logg_vals:
            for mh in mh_vals:
                test_count += 1
                stellar_type = StellarType(f"Grid_{test_count}", teff, logg, mh)
                
                print(f"\n[{test_count}/{total_tests}] {stellar_type.name}: T={teff:.0f}K, log g={logg:.1f}, [M/H]={mh:.1f}")
                
                result = run_single_comparison(stellar_type, frequencies, verbose=False)
                if result.get('success'):
                    result['teff'] = teff
                    result['logg'] = logg
                    result['mh'] = mh
                    results.append(result)
                    print(f"   ✅ {result['agreement_percent']:.1f}% agreement")
                else:
                    print(f"   ❌ Failed")
    
    return results


def create_comparison_plots(results: List[Dict], save_path: Optional[str] = None):
    """
    Create comprehensive plots showing agreement across stellar types
    """
    if not results:
        print("No results to plot")
        return
    
    # Filter successful results
    successful_results = [r for r in results if r.get('success')]
    if not successful_results:
        print("No successful results to plot")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    agreements = [r['agreement_percent'] for r in successful_results]
    mean_ratios = [r['mean_ratio'] for r in successful_results]
    stellar_names = [r['stellar_type'] for r in successful_results]
    
    # Plot 1: Agreement percentages
    bars1 = ax1.bar(range(len(agreements)), agreements, 
                    color=['green' if a > 99 else 'orange' if a > 95 else 'red' for a in agreements])
    ax1.set_xlabel("Stellar Type")
    ax1.set_ylabel("Agreement (%)")
    ax1.set_title("Agreement Percentage by Stellar Type")
    ax1.set_xticks(range(len(stellar_names)))
    ax1.set_xticklabels(stellar_names, rotation=45, ha='right')
    ax1.axhline(y=99, color='green', linestyle='--', alpha=0.7, label='99% (Excellent)')
    ax1.axhline(y=95, color='orange', linestyle='--', alpha=0.7, label='95% (Good)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, agreement in zip(bars1, agreements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{agreement:.1f}%',
                ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Mean ratios
    bars2 = ax2.bar(range(len(mean_ratios)), mean_ratios,
                    color=['green' if abs(r-1) < 0.01 else 'orange' if abs(r-1) < 0.05 else 'red' for r in mean_ratios])
    ax2.set_xlabel("Stellar Type")
    ax2.set_ylabel("Mean Ratio (Jorg/Korg)")
    ax2.set_title("Mean Ratio by Stellar Type")
    ax2.set_xticks(range(len(stellar_names)))
    ax2.set_xticklabels(stellar_names, rotation=45, ha='right')
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label='Perfect (1.000)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Parameter space (if grid data available)
    grid_results = [r for r in successful_results if 'teff' in r]
    if grid_results:
        teff_vals = [r['teff'] for r in grid_results]
        logg_vals = [r['logg'] for r in grid_results]
        agreement_vals = [r['agreement_percent'] for r in grid_results]
        
        scatter = ax3.scatter(teff_vals, logg_vals, c=agreement_vals, s=50, 
                            cmap='RdYlGn', vmin=90, vmax=100, alpha=0.8)
        ax3.set_xlabel("Effective Temperature (K)")
        ax3.set_ylabel("Surface Gravity (log g)")
        ax3.set_title("Agreement Across Parameter Space")
        plt.colorbar(scatter, ax=ax3, label="Agreement (%)")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "Grid data not available", ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("Parameter Space Analysis")
    
    # Plot 4: Component comparison
    if successful_results:
        sample_result = successful_results[0]
        components = ['alpha_hminus_bf', 'alpha_hminus_ff', 'alpha_h_i_bf']
        component_names = ['H⁻ bf', 'H⁻ ff', 'H I bf']
        component_ratios = [sample_result['component_stats'][comp]['mean_ratio'] for comp in components]
        
        bars4 = ax4.bar(component_names, component_ratios,
                       color=['lightgreen', 'lightblue', 'lightyellow'],
                       edgecolor='black', alpha=0.8)
        ax4.set_ylabel("Mean Ratio (Jorg/Korg)")
        ax4.set_title("Component-by-Component Agreement")
        ax4.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect (1.000)')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, ratio in zip(bars4, component_ratios):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{ratio:.3f}',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plots saved to {save_path}")
    
    plt.show()


def save_results(results: List[Dict], filename: str):
    """
    Save results to JSON file
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to {filename}")


def print_summary(results: List[Dict]):
    """
    Print summary statistics
    """
    successful_results = [r for r in results if r.get('success')]
    
    if not successful_results:
        print("❌ No successful comparisons")
        return
    
    agreements = [r['agreement_percent'] for r in successful_results]
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"   Total tests: {len(results)}")
    print(f"   Successful: {len(successful_results)} ({len(successful_results)/len(results)*100:.1f}%)")
    print(f"   Mean agreement: {np.mean(agreements):.2f}%")
    print(f"   Min agreement: {np.min(agreements):.2f}%")
    print(f"   Max agreement: {np.max(agreements):.2f}%")
    print(f"   Std dev: {np.std(agreements):.2f}%")
    
    excellent = sum(1 for a in agreements if a > 99)
    good = sum(1 for a in agreements if 95 < a <= 99)
    fair = sum(1 for a in agreements if 90 < a <= 95)
    poor = sum(1 for a in agreements if a <= 90)
    
    print(f"\n🎯 AGREEMENT BREAKDOWN:")
    print(f"   Excellent (>99%): {excellent} ({excellent/len(successful_results)*100:.1f}%)")
    print(f"   Good (95-99%): {good} ({good/len(successful_results)*100:.1f}%)")
    print(f"   Fair (90-95%): {fair} ({fair/len(successful_results)*100:.1f}%)")
    print(f"   Poor (<90%): {poor} ({poor/len(successful_results)*100:.1f}%)")


def main():
    """
    Main function with command line interface
    """
    parser = argparse.ArgumentParser(description="Compare Jorg vs Korg.jl across stellar types")
    parser.add_argument('--mode', choices=['all', 'single', 'grid', 'quick'], default='quick',
                       help='Comparison mode: all predefined types, single type, parameter grid, or quick 3-star test')
    parser.add_argument('--teff', type=float, default=5780, help='Effective temperature (K)')
    parser.add_argument('--logg', type=float, default=4.44, help='Surface gravity (log g)')
    parser.add_argument('--mh', type=float, default=0.0, help='Metallicity [M/H]')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to file')
    parser.add_argument('--save-results', action='store_true', help='Save results to JSON')
    parser.add_argument('--output-dir', default='.', help='Output directory for files')
    
    args = parser.parse_args()
    
    print("🌟 JORG vs KORG.JL STELLAR TYPES COMPARISON")
    print("=" * 60)
    
    # Test frequencies (reduced for faster testing)
    frequencies = np.array([4e14, 6e14, 1e15, 3e15, 6e15])
    
    results = []
    
    if args.mode == 'all':
        # Test all predefined stellar types
        print(f"Testing {len(STELLAR_TYPES)} predefined stellar types...")
        for name, stellar_type in STELLAR_TYPES.items():
            result = run_single_comparison(stellar_type, frequencies)
            if result:
                results.append(result)
    
    elif args.mode == 'quick':
        # Test quick subset of stellar types
        print(f"Testing {len(QUICK_STELLAR_TYPES)} stellar types (quick test)...")
        for name, stellar_type in QUICK_STELLAR_TYPES.items():
            result = run_single_comparison(stellar_type, frequencies)
            if result:
                results.append(result)
    
    elif args.mode == 'single':
        # Test single stellar type
        stellar_type = StellarType('Custom', args.teff, args.logg, args.mh)
        result = run_single_comparison(stellar_type, frequencies)
        if result:
            results.append(result)
    
    elif args.mode == 'grid':
        # Test parameter grid
        results = run_stellar_grid()
    
    # Print summary
    print_summary(results)
    
    # Create plots
    if results and (args.save_plots or len(results) > 1):
        plot_filename = os.path.join(args.output_dir, 'stellar_comparison.png') if args.save_plots else None
        create_comparison_plots(results, plot_filename)
    
    # Save results
    if args.save_results and results:
        results_filename = os.path.join(args.output_dir, 'stellar_comparison_results.json')
        save_results(results, results_filename)
    
    print(f"\n✅ Comparison complete!")


if __name__ == "__main__":
    main()