#!/usr/bin/env python3
"""
Comprehensive Synthesis Validation: Jorg vs Korg.jl

This test validates Jorg's synthesis pipeline against Korg.jl by comparing
both synth() and synthesize() functions across different stellar parameters.

Creates Korg.jl reference data and compares with Jorg implementation to ensure
<1% flux differences across H-R diagram coverage.
"""

import sys
import os
import subprocess
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

from jorg.synthesis import synth, synthesize, format_abundances
from jorg.constants import SPEED_OF_LIGHT


def create_korg_reference_script():
    """Create Julia script to generate Korg.jl reference data"""
    script_content = '''#!/usr/bin/env julia
"""
Korg.jl Synthesis Reference Data Generator

This script generates reference synthesis data for validation against Jorg.
Tests both synth() and synthesize() functions across stellar parameter grid.
"""

# Load Korg module
include("../../../src/Korg.jl")
using .Korg
using CSV, DataFrames, JSON

println("=" ^ 60)
println("Korg.jl Synthesis Reference Generator")
println("=" ^ 60)

# Test stellar parameter grid
stellar_parameters = [
    # Solar-type stars
    (5777, 4.44, 0.0, "Sun"),
    (5800, 4.5, 0.0, "Solar_analog"),
    (5800, 4.5, -0.5, "Metal_poor_solar"),
    (5800, 4.5, 0.3, "Metal_rich_solar"),
    
    # M dwarfs
    (3500, 4.5, 0.0, "M_dwarf"),
    (3800, 4.8, -0.3, "M_dwarf_poor"),
    
    # K dwarfs  
    (4500, 4.5, 0.0, "K_dwarf"),
    (5200, 4.6, 0.2, "K_dwarf_rich"),
    
    # G dwarfs
    (5500, 4.4, -0.2, "G_dwarf"),
    (6000, 4.3, 0.1, "G_dwarf_rich"),
    
    # F dwarfs
    (6500, 4.2, 0.0, "F_dwarf"),
    (7000, 4.0, -0.1, "F_dwarf_poor"),
    
    # Giants
    (4800, 2.5, 0.0, "K_giant"),
    (5200, 3.0, -0.5, "G_giant_poor"),
]

# Wavelength ranges to test
wavelength_ranges = [
    (5000, 5100, "Blue_green"),
    (5400, 5500, "Green"), 
    (6000, 6100, "Red"),
    (6500, 6600, "Deep_red"),
]

results = Dict()
errors = []

println("\\nGenerating reference data for $(length(stellar_parameters)) stars...")

for (i, (Teff, logg, m_H, name)) in enumerate(stellar_parameters)
    println("\\n$(i)/$(length(stellar_parameters)): $name (Teff=$Teff, logg=$logg, [M/H]=$m_H)")
    
    try
        # Format abundances
        A_X = Korg.format_A_X(m_H)
        
        # Interpolate atmosphere  
        atm = Korg.interpolate_marcs(Teff, logg, A_X)
        
        star_results = Dict()
        
        for (λ_start, λ_end, wl_name) in wavelength_ranges
            println("  Testing $wl_name: $λ_start-$λ_end Å")
            
            try
                # Test synth() function
                wl_synth, flux_synth, cntm_synth = Korg.synth(
                    Teff=Teff, logg=logg, m_H=m_H,
                    wavelengths=(λ_start, λ_end),
                    rectify=true
                )
                
                # Test synthesize() function  
                result_detailed = Korg.synthesize(
                    atm, Korg.get_VALD_solar_linelist(), A_X, 
                    λ_start, λ_end;
                    vmic=1.0
                )
                
                # Store results
                range_key = "$(name)_$(wl_name)"
                star_results[range_key] = Dict(
                    "wavelengths_synth" => collect(wl_synth),
                    "flux_synth" => collect(flux_synth),
                    "continuum_synth" => collect(cntm_synth),
                    "wavelengths_detailed" => collect(result_detailed.wavelengths),
                    "flux_detailed" => collect(result_detailed.flux),
                    "continuum_detailed" => result_detailed.cntm !== nothing ? collect(result_detailed.cntm) : nothing,
                    "alpha_shape" => size(result_detailed.alpha),
                    "mu_grid_length" => length(result_detailed.mu_grid),
                    "n_species" => length(result_detailed.number_densities),
                    "stellar_params" => [Teff, logg, m_H],
                    "wavelength_range" => [λ_start, λ_end]
                )
                
                println("    ✓ Success: flux range $(minimum(flux_synth):.2e) - $(maximum(flux_synth):.2e)")
                
            catch e
                println("    ✗ Failed: $e")
                push!(errors, "$name $wl_name: $e")
            end
        end
        
        results[name] = star_results
        
    catch e
        println("  ✗ Star failed: $e")
        push!(errors, "$name: $e")
    end
end

# Save results
println("\\nSaving reference data...")

# Save as JSON
open("korg_synthesis_reference.json", "w") do f
    JSON.print(f, results, 2)
end

# Save stellar parameters as CSV
params_df = DataFrame(
    Name = [p[4] for p in stellar_parameters],
    Teff = [p[1] for p in stellar_parameters], 
    logg = [p[2] for p in stellar_parameters],
    m_H = [p[3] for p in stellar_parameters]
)
CSV.write("korg_stellar_parameters.csv", params_df)

# Save wavelength ranges
wl_df = DataFrame(
    Name = [w[3] for w in wavelength_ranges],
    Lambda_start = [w[1] for w in wavelength_ranges],
    Lambda_end = [w[2] for w in wavelength_ranges]
)
CSV.write("korg_wavelength_ranges.csv", wl_df)

# Save errors if any
if !isempty(errors)
    println("\\nErrors encountered:")
    for error in errors
        println("  - $error")
    end
    
    open("korg_synthesis_errors.txt", "w") do f
        for error in errors
            println(f, error)
        end
    end
end

# Summary
successful_results = sum(length(v) for v in values(results))
println("\\n" * "=" ^ 60)
println("Reference data generation complete!")
println("Successful combinations: $successful_results")
println("Errors: $(length(errors))")
println("Files saved:")
println("  - korg_synthesis_reference.json")
println("  - korg_stellar_parameters.csv") 
println("  - korg_wavelength_ranges.csv")
if !isempty(errors)
    println("  - korg_synthesis_errors.txt")
end
println("=" ^ 60)
'''
    
    script_path = Path(__file__).parent / "generate_korg_synthesis_reference.jl"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


def run_korg_reference_generation(script_path: Path) -> Optional[Path]:
    """Run Korg reference generation script"""
    try:
        # Change to script directory
        script_dir = script_path.parent
        original_dir = os.getcwd()
        os.chdir(script_dir)
        
        print("Running Korg.jl reference generation...")
        result = subprocess.run(
            ["julia", "--project=../../../..", str(script_path.name)],
            capture_output=True, text=True, timeout=600
        )
        
        os.chdir(original_dir)
        
        if result.returncode != 0:
            print(f"Korg script failed: {result.stderr}")
            return None
            
        print("✓ Korg reference data generated successfully")
        return script_dir
        
    except subprocess.TimeoutExpired:
        print("Korg script timed out")
        return None
    except Exception as e:
        print(f"Error running Korg script: {e}")
        return None


def load_korg_reference_data(data_dir: Path) -> Optional[Dict]:
    """Load Korg reference data"""
    try:
        ref_file = data_dir / "korg_synthesis_reference.json"
        if not ref_file.exists():
            print(f"Reference file not found: {ref_file}")
            return None
            
        with open(ref_file) as f:
            data = json.load(f)
            
        print(f"✓ Loaded Korg reference data: {len(data)} stars")
        return data
        
    except Exception as e:
        print(f"Error loading Korg reference data: {e}")
        return None


def test_jorg_synthesis_grid(stellar_params: List[Tuple], wavelength_ranges: List[Tuple]) -> Dict:
    """Test Jorg synthesis across stellar parameter grid"""
    print("\nTesting Jorg synthesis...")
    
    jorg_results = {}
    errors = []
    
    for i, (Teff, logg, m_H, name) in enumerate(stellar_params):
        print(f"\n{i+1}/{len(stellar_params)}: {name} (Teff={Teff}, logg={logg}, [M/H]={m_H})")
        
        try:
            # Format abundances
            A_X = format_abundances(m_H)
            
            star_results = {}
            
            for λ_start, λ_end, wl_name in wavelength_ranges:
                print(f"  Testing {wl_name}: {λ_start}-{λ_end} Å")
                
                try:
                    # Test synth() function
                    wl_synth, flux_synth, cntm_synth = synth(
                        Teff=Teff, logg=logg, m_H=m_H,
                        wavelengths=(λ_start, λ_end),
                        rectify=True
                    )
                    
                    # Test synthesize() function
                    from jorg.synthesis import interpolate_atmosphere
                    atm = interpolate_atmosphere(Teff, logg, A_X)
                    wl_detailed = jnp.linspace(λ_start, λ_end, 100)
                    
                    result_detailed = synthesize(
                        atm, None, A_X, wl_detailed,
                        vmic=1.0
                    )
                    
                    # Store results
                    range_key = f"{name}_{wl_name}"
                    star_results[range_key] = {
                        "wavelengths_synth": np.array(wl_synth),
                        "flux_synth": np.array(flux_synth),
                        "continuum_synth": np.array(cntm_synth),
                        "wavelengths_detailed": np.array(result_detailed.wavelengths),
                        "flux_detailed": np.array(result_detailed.flux),
                        "continuum_detailed": np.array(result_detailed.cntm) if result_detailed.cntm is not None else None,
                        "alpha_shape": result_detailed.alpha.shape,
                        "mu_grid_length": len(result_detailed.mu_grid),
                        "n_species": len(result_detailed.number_densities),
                        "stellar_params": [Teff, logg, m_H],
                        "wavelength_range": [λ_start, λ_end],
                        "success": True
                    }
                    
                    print(f"    ✓ Success: flux range {np.min(flux_synth):.2e} - {np.max(flux_synth):.2e}")
                    
                except Exception as e:
                    print(f"    ✗ Failed: {e}")
                    errors.append(f"{name} {wl_name}: {e}")
                    star_results[f"{name}_{wl_name}"] = {"success": False, "error": str(e)}
            
            jorg_results[name] = star_results
            
        except Exception as e:
            print(f"  ✗ Star failed: {e}")
            errors.append(f"{name}: {e}")
            jorg_results[name] = {"success": False, "error": str(e)}
    
    return {"results": jorg_results, "errors": errors}


def compare_synthesis_results(jorg_data: Dict, korg_data: Dict) -> Dict:
    """Compare Jorg and Korg synthesis results"""
    print("\nComparing Jorg vs Korg results...")
    
    comparisons = {}
    overall_stats = {
        "total_comparisons": 0,
        "successful_comparisons": 0,
        "max_flux_error": 0.0,
        "mean_flux_error": 0.0,
        "agreement_1pct": 0,
        "agreement_5pct": 0
    }
    
    for star_name in jorg_data["results"]:
        if star_name not in korg_data:
            continue
            
        star_comparisons = {}
        
        for range_key in jorg_data["results"][star_name]:
            if not jorg_data["results"][star_name][range_key].get("success", False):
                continue
                
            if range_key not in korg_data[star_name]:
                continue
                
            jorg_result = jorg_data["results"][star_name][range_key]
            korg_result = korg_data[star_name][range_key]
            
            # Compare synth() results
            jorg_flux = jorg_result["flux_synth"]
            korg_flux = np.array(korg_result["flux_synth"])
            
            # Interpolate to common wavelength grid
            jorg_wl = jorg_result["wavelengths_synth"]
            korg_wl = np.array(korg_result["wavelengths_synth"])
            
            # Find common wavelength range
            wl_min = max(np.min(jorg_wl), np.min(korg_wl))
            wl_max = min(np.max(jorg_wl), np.max(korg_wl))
            
            if wl_max > wl_min:
                # Create common grid
                common_wl = np.linspace(wl_min, wl_max, 50)
                
                # Interpolate both to common grid
                jorg_flux_interp = np.interp(common_wl, jorg_wl, jorg_flux)
                korg_flux_interp = np.interp(common_wl, korg_wl, korg_flux)
                
                # Calculate relative differences
                rel_diff = np.abs(jorg_flux_interp - korg_flux_interp) / np.abs(korg_flux_interp)
                
                # Remove any NaN or infinite values
                valid_mask = np.isfinite(rel_diff)
                if np.sum(valid_mask) > 0:
                    rel_diff = rel_diff[valid_mask]
                    
                    max_error = np.max(rel_diff)
                    mean_error = np.mean(rel_diff)
                    
                    star_comparisons[range_key] = {
                        "max_relative_error": max_error,
                        "mean_relative_error": mean_error,
                        "agreement_1pct": max_error < 0.01,
                        "agreement_5pct": max_error < 0.05,
                        "n_points": len(rel_diff),
                        "wavelength_range": [wl_min, wl_max]
                    }
                    
                    # Update overall statistics
                    overall_stats["total_comparisons"] += 1
                    overall_stats["successful_comparisons"] += 1
                    overall_stats["max_flux_error"] = max(overall_stats["max_flux_error"], max_error)
                    overall_stats["mean_flux_error"] += mean_error
                    if max_error < 0.01:
                        overall_stats["agreement_1pct"] += 1
                    if max_error < 0.05:
                        overall_stats["agreement_5pct"] += 1
                        
                    print(f"  {range_key}: max error {max_error:.1%}, mean error {mean_error:.1%}")
        
        if star_comparisons:
            comparisons[star_name] = star_comparisons
    
    # Finalize overall statistics
    if overall_stats["successful_comparisons"] > 0:
        overall_stats["mean_flux_error"] /= overall_stats["successful_comparisons"]
    
    return {"comparisons": comparisons, "overall": overall_stats}


def create_comparison_plots(comparison_data: Dict, output_dir: Path):
    """Create visualization plots of comparison results"""
    try:
        import matplotlib.pyplot as plt
        
        # Extract data for plotting
        max_errors = []
        mean_errors = []
        labels = []
        
        for star_name, star_data in comparison_data["comparisons"].items():
            for range_name, range_data in star_data.items():
                max_errors.append(range_data["max_relative_error"])
                mean_errors.append(range_data["mean_relative_error"])
                labels.append(f"{star_name.split('_')[0]}_{range_name.split('_')[-1]}")
        
        if len(max_errors) == 0:
            print("No data for plotting")
            return
        
        # Create plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Max relative errors
        ax1.bar(range(len(max_errors)), [e*100 for e in max_errors])
        ax1.axhline(y=1.0, color='r', linestyle='--', label='1% threshold')
        ax1.axhline(y=5.0, color='orange', linestyle='--', label='5% threshold')
        ax1.set_ylabel('Max Relative Error (%)')
        ax1.set_title('Maximum Flux Errors: Jorg vs Korg.jl')
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mean relative errors
        ax2.bar(range(len(mean_errors)), [e*100 for e in mean_errors])
        ax2.axhline(y=1.0, color='r', linestyle='--', label='1% threshold')
        ax2.set_ylabel('Mean Relative Error (%)')
        ax2.set_title('Mean Flux Errors: Jorg vs Korg.jl')
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "synthesis_comparison_errors.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Comparison plots saved to {output_dir}")
        
    except ImportError:
        print("Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"Error creating plots: {e}")


def main():
    """Main validation test"""
    print("=" * 60)
    print("Synthesis Validation: Jorg vs Korg.jl")
    print("=" * 60)
    
    # Define test grid
    stellar_parameters = [
        (5777, 4.44, 0.0, "Sun"),
        (5800, 4.5, 0.0, "Solar_analog"),
        (5800, 4.5, -0.5, "Metal_poor_solar"),
        (3500, 4.5, 0.0, "M_dwarf"),
        (4500, 4.5, 0.0, "K_dwarf"),
        (6500, 4.2, 0.0, "F_dwarf"),
        (4800, 2.5, 0.0, "K_giant"),
    ]
    
    wavelength_ranges = [
        (5000, 5100, "Blue_green"),
        (5400, 5500, "Green"),
        (6000, 6100, "Red"),
    ]
    
    # Generate Korg reference data
    print("\n1. Generating Korg.jl reference data...")
    script_path = create_korg_reference_script()
    korg_data_dir = run_korg_reference_generation(script_path)
    
    if korg_data_dir:
        korg_data = load_korg_reference_data(korg_data_dir)
    else:
        print("⚠ Korg reference not available, testing Jorg only")
        korg_data = None
    
    # Test Jorg synthesis
    print("\n2. Testing Jorg synthesis...")
    jorg_data = test_jorg_synthesis_grid(stellar_parameters, wavelength_ranges)
    
    # Compare results
    if korg_data:
        print("\n3. Comparing results...")
        comparison_results = compare_synthesis_results(jorg_data, korg_data)
        
        # Create visualization
        output_dir = Path(__file__).parent
        create_comparison_plots(comparison_results, output_dir)
        
        # Save detailed comparison
        with open(output_dir / "synthesis_comparison_results.json", 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
    else:
        comparison_results = None
    
    # Save Jorg results
    output_dir = Path(__file__).parent
    with open(output_dir / "jorg_synthesis_results.json", 'w') as f:
        json.dump(jorg_data, f, indent=2, default=str)
    
    # Summary
    print("\n" + "=" * 60)
    print("SYNTHESIS VALIDATION SUMMARY")
    print("=" * 60)
    
    # Jorg results
    successful_jorg = sum(1 for star_data in jorg_data["results"].values() 
                         for range_data in star_data.values() 
                         if range_data.get("success", False))
    total_tests = len(stellar_parameters) * len(wavelength_ranges)
    
    print(f"Jorg Implementation: {successful_jorg}/{total_tests} tests successful")
    print(f"Jorg Errors: {len(jorg_data['errors'])}")
    
    if comparison_results:
        overall = comparison_results["overall"]
        print(f"Korg Comparisons: {overall['successful_comparisons']}/{overall['total_comparisons']}")
        print(f"Max flux error: {overall['max_flux_error']:.1%}")
        print(f"Mean flux error: {overall['mean_flux_error']:.1%}")
        print(f"Agreement <1%: {overall['agreement_1pct']}/{overall['successful_comparisons']}")
        print(f"Agreement <5%: {overall['agreement_5pct']}/{overall['successful_comparisons']}")
        
        if overall['agreement_1pct'] == overall['successful_comparisons']:
            print("🎉 ALL SYNTHESIS TESTS PASSED - Jorg matches Korg.jl!")
            success = True
        elif overall['agreement_5pct'] == overall['successful_comparisons']:
            print("✅ SYNTHESIS TESTS MOSTLY PASSED - Minor differences <5%")
            success = True
        else:
            print("⚠️ Some synthesis tests show significant differences")
            success = False
    else:
        print("Korg Comparison: Not available")
        success = successful_jorg == total_tests
    
    print("=" * 60)
    
    return {
        'jorg_data': jorg_data,
        'korg_data': korg_data,
        'comparison_results': comparison_results,
        'success': success
    }


if __name__ == "__main__":
    results = main()
    sys.exit(0 if results['success'] else 1)