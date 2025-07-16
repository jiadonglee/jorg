#!/usr/bin/env python3
"""
Jorg synth() Test and Comparison

This script tests Jorg's synth() function with the same parameters as Korg.jl
and compares the results for validation.
"""

import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

def test_jorg_synth_cases():
    """Test Jorg synth() with same parameters as Korg reference"""
    
    # Same test cases as Korg script
    test_cases = [
        {
            "name": "solar_basic",
            "Teff": 5777,
            "logg": 4.44,
            "m_H": 0.0,
            "wavelengths": (5000, 5100),
            "rectify": True,
            "vmic": 1.0
        },
        {
            "name": "solar_metal_poor", 
            "Teff": 5777,
            "logg": 4.44,
            "m_H": -0.5,
            "wavelengths": (5000, 5100),
            "rectify": True,
            "vmic": 1.0
        },
        {
            "name": "m_dwarf",
            "Teff": 3500,
            "logg": 4.5,
            "m_H": 0.0,
            "wavelengths": (5000, 5100),
            "rectify": True,
            "vmic": 1.0
        },
        {
            "name": "k_dwarf",
            "Teff": 4500,
            "logg": 4.5,
            "m_H": 0.0,
            "wavelengths": (5000, 5100),
            "rectify": True,
            "vmic": 1.0
        },
        {
            "name": "f_dwarf",
            "Teff": 6500,
            "logg": 4.2,
            "m_H": 0.0,
            "wavelengths": (5000, 5100),
            "rectify": True,
            "vmic": 1.0
        },
        {
            "name": "solar_blue",
            "Teff": 5777,
            "logg": 4.44,
            "m_H": 0.0,
            "wavelengths": (4500, 4600),
            "rectify": True,
            "vmic": 1.0
        },
        {
            "name": "solar_red",
            "Teff": 5777,
            "logg": 4.44,
            "m_H": 0.0,
            "wavelengths": (6000, 6100),
            "rectify": True,
            "vmic": 1.0
        },
        {
            "name": "solar_unrectified",
            "Teff": 5777,
            "logg": 4.44,
            "m_H": 0.0,
            "wavelengths": (5000, 5100),
            "rectify": False,
            "vmic": 1.0
        },
        {
            "name": "solar_vmic_2",
            "Teff": 5777,
            "logg": 4.44,
            "m_H": 0.0,
            "wavelengths": (5000, 5100),
            "rectify": True,
            "vmic": 2.0
        },
        {
            "name": "solar_fe_poor",
            "Teff": 5777,
            "logg": 4.44,
            "m_H": 0.0,
            "Fe": -0.3,
            "wavelengths": (5000, 5100),
            "rectify": True,
            "vmic": 1.0
        }
    ]
    
    print("=" * 60)
    print("Jorg synth() Test Suite")
    print("=" * 60)
    print(f"Testing {len(test_cases)} cases to compare with Korg.jl...")
    
    results = {}
    errors = []
    
    for i, test_case in enumerate(test_cases):
        name = test_case["name"]
        print(f"\n{i+1}/{len(test_cases)}: Testing {name}")
        
        try:
            # Import here to handle potential issues
            from jorg.synthesis import synth
            
            # Extract parameters
            Teff = test_case["Teff"]
            logg = test_case["logg"]
            m_H = test_case["m_H"]
            wavelengths = test_case["wavelengths"]
            rectify = test_case["rectify"]
            vmic = test_case["vmic"]
            
            # Handle individual element abundances
            abundances = {}
            for key in ["Fe", "C", "O", "Mg", "Si", "Ca", "Ti", "Cr", "Ni"]:
                if key in test_case:
                    abundances[key] = test_case[key]
            
            print(f"  Parameters: Teff={Teff}, logg={logg}, [M/H]={m_H}")
            print(f"  Wavelengths: {wavelengths[0]}-{wavelengths[1]} Å")
            print(f"  Rectify: {rectify}, vmic: {vmic} km/s")
            if abundances:
                print(f"  Individual abundances: {abundances}")
            
            # Run Jorg synth
            start_time = time.time()
            
            wavelengths_out, flux, continuum = synth(
                Teff=Teff,
                logg=logg,
                m_H=m_H,
                wavelengths=wavelengths,
                rectify=rectify,
                vmic=vmic,
                **abundances
            )
            
            elapsed = time.time() - start_time
            
            # Convert to numpy arrays
            wavelengths_out = np.array(wavelengths_out)
            flux = np.array(flux)
            continuum = np.array(continuum)
            
            # Store results
            results[name] = {
                "parameters": test_case,
                "wavelengths": wavelengths_out,
                "flux": flux,
                "continuum": continuum,
                "timing": elapsed,
                "n_points": len(wavelengths_out),
                "flux_stats": {
                    "min": float(np.min(flux)),
                    "max": float(np.max(flux)),
                    "mean": float(np.mean(flux)),
                    "std": float(np.std(flux))
                },
                "continuum_stats": {
                    "min": float(np.min(continuum)),
                    "max": float(np.max(continuum)),
                    "mean": float(np.mean(continuum))
                },
                "success": True
            }
            
            print(f"  ✓ Success in {elapsed:.1f}s")
            print(f"    Points: {len(wavelengths_out)}")
            print(f"    Flux range: {np.min(flux):.3e} - {np.max(flux):.3e}")
            print(f"    Continuum range: {np.min(continuum):.3e} - {np.max(continuum):.3e}")
            
            # Save individual spectrum
            spectrum_df = pd.DataFrame({
                'wavelength': wavelengths_out,
                'flux': flux,
                'continuum': continuum
            })
            spectrum_df.to_csv(f"jorg_synth_{name}.csv", index=False)
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            errors.append(f"{name}: {e}")
            results[name] = {"success": False, "error": str(e)}
            import traceback
            traceback.print_exc()
    
    return results, errors


def load_korg_reference() -> Dict:
    """Load Korg reference data if available"""
    ref_file = Path(__file__).parent / "korg_synth_reference.json"
    
    if ref_file.exists():
        with open(ref_file) as f:
            return json.load(f)
    else:
        print("⚠ Korg reference data not found")
        return {}


def compare_results(jorg_results: Dict, korg_results: Dict) -> Dict:
    """Compare Jorg and Korg results"""
    print("\n" + "=" * 60)
    print("Comparing Jorg vs Korg Results")
    print("=" * 60)
    
    if not korg_results:
        print("No Korg reference data available for comparison")
        return {}
    
    comparisons = {}
    
    for case_name in jorg_results:
        if not jorg_results[case_name].get("success", False):
            continue
            
        if case_name not in korg_results or not korg_results[case_name].get("success", False):
            print(f"\n{case_name}: No Korg reference available")
            continue
        
        print(f"\n{case_name}:")
        
        jorg_data = jorg_results[case_name]
        korg_data = korg_results[case_name]
        
        # Compare basic statistics
        jorg_flux_stats = jorg_data["flux_stats"]
        korg_flux_stats = korg_data["flux_stats"]
        
        # Calculate relative differences in key statistics
        mean_diff = abs(jorg_flux_stats["mean"] - korg_flux_stats["mean"]) / abs(korg_flux_stats["mean"])
        min_diff = abs(jorg_flux_stats["min"] - korg_flux_stats["min"]) / abs(korg_flux_stats["min"])
        max_diff = abs(jorg_flux_stats["max"] - korg_flux_stats["max"]) / abs(korg_flux_stats["max"])
        
        print(f"  Flux statistics comparison:")
        print(f"    Mean: Jorg={jorg_flux_stats['mean']:.3e}, Korg={korg_flux_stats['mean']:.3e} (diff: {mean_diff:.1%})")
        print(f"    Min:  Jorg={jorg_flux_stats['min']:.3e}, Korg={korg_flux_stats['min']:.3e} (diff: {min_diff:.1%})")
        print(f"    Max:  Jorg={jorg_flux_stats['max']:.3e}, Korg={korg_flux_stats['max']:.3e} (diff: {max_diff:.1%})")
        
        # Detailed spectral comparison if arrays are available
        if isinstance(jorg_data["flux"], np.ndarray) and "flux" in korg_data:
            jorg_flux = jorg_data["flux"]
            korg_flux = np.array(korg_data["flux"])
            jorg_wl = jorg_data["wavelengths"]
            korg_wl = np.array(korg_data["wavelengths"])
            
            # Find common wavelength range for interpolation
            wl_min = max(np.min(jorg_wl), np.min(korg_wl))
            wl_max = min(np.max(jorg_wl), np.max(korg_wl))
            
            if wl_max > wl_min and len(jorg_flux) > 10 and len(korg_flux) > 10:
                # Create common wavelength grid
                common_wl = np.linspace(wl_min, wl_max, 50)
                
                # Interpolate both spectra to common grid
                jorg_flux_interp = np.interp(common_wl, jorg_wl, jorg_flux)
                korg_flux_interp = np.interp(common_wl, korg_wl, korg_flux)
                
                # Calculate point-by-point differences
                rel_diff = np.abs(jorg_flux_interp - korg_flux_interp) / np.abs(korg_flux_interp)
                
                # Remove any NaN or infinite values
                valid_mask = np.isfinite(rel_diff)
                if np.sum(valid_mask) > 0:
                    rel_diff = rel_diff[valid_mask]
                    
                    max_error = np.max(rel_diff)
                    mean_error = np.mean(rel_diff)
                    median_error = np.median(rel_diff)
                    
                    print(f"  Spectral comparison ({len(rel_diff)} points):")
                    print(f"    Max relative error: {max_error:.1%}")
                    print(f"    Mean relative error: {mean_error:.1%}")
                    print(f"    Median relative error: {median_error:.1%}")
                    
                    # Assessment
                    if max_error < 0.01:
                        assessment = "Excellent agreement ✓"
                    elif max_error < 0.05:
                        assessment = "Good agreement ✓"
                    elif max_error < 0.1:
                        assessment = "Acceptable agreement ⚠"
                    else:
                        assessment = "Significant differences ✗"
                    
                    print(f"    Assessment: {assessment}")
                    
                    comparisons[case_name] = {
                        "max_error": max_error,
                        "mean_error": mean_error,
                        "median_error": median_error,
                        "n_points": len(rel_diff),
                        "agreement_level": assessment,
                        "flux_stats_diff": {
                            "mean": mean_diff,
                            "min": min_diff,
                            "max": max_diff
                        }
                    }
        
        # Performance comparison
        jorg_time = jorg_data["timing"]
        korg_time = korg_data["timing"]
        time_ratio = jorg_time / korg_time if korg_time > 0 else float('inf')
        
        print(f"  Performance: Jorg={jorg_time:.1f}s, Korg={korg_time:.1f}s (ratio: {time_ratio:.1f}x)")
    
    return comparisons


def create_comparison_plots(jorg_results: Dict, korg_results: Dict, comparisons: Dict):
    """Create visualization plots comparing Jorg and Korg"""
    try:
        import matplotlib.pyplot as plt
        
        # Find a good case for detailed comparison
        plot_case = None
        for case_name in ["solar_basic", "solar_metal_poor", "m_dwarf"]:
            if (case_name in jorg_results and jorg_results[case_name].get("success", False) and
                case_name in korg_results and korg_results[case_name].get("success", False)):
                plot_case = case_name
                break
        
        if not plot_case:
            print("No suitable cases for plotting")
            return
        
        jorg_data = jorg_results[plot_case]
        korg_data = korg_results[plot_case]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Jorg vs Korg Comparison: {plot_case}', fontsize=14)
        
        # Plot 1: Spectrum comparison
        axes[0, 0].plot(jorg_data["wavelengths"], jorg_data["flux"], 'b-', label='Jorg', linewidth=1.5)
        axes[0, 0].plot(korg_data["wavelengths"], korg_data["flux"], 'r--', label='Korg', linewidth=1, alpha=0.8)
        axes[0, 0].set_xlabel('Wavelength (Å)')
        axes[0, 0].set_ylabel('Flux')
        axes[0, 0].set_title('Spectrum Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Residuals
        if len(jorg_data["wavelengths"]) == len(korg_data["wavelengths"]):
            residuals = (jorg_data["flux"] - np.array(korg_data["flux"])) / np.array(korg_data["flux"])
            axes[0, 1].plot(jorg_data["wavelengths"], residuals * 100, 'g-', linewidth=1)
            axes[0, 1].set_xlabel('Wavelength (Å)')
            axes[0, 1].set_ylabel('Relative Difference (%)')
            axes[0, 1].set_title('Flux Residuals')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.5)
        
        # Plot 3: Error summary
        if comparisons:
            case_names = list(comparisons.keys())
            max_errors = [comparisons[name]["max_error"] * 100 for name in case_names]
            mean_errors = [comparisons[name]["mean_error"] * 100 for name in case_names]
            
            x_pos = np.arange(len(case_names))
            axes[1, 0].bar(x_pos - 0.2, max_errors, 0.4, label='Max Error', alpha=0.7)
            axes[1, 0].bar(x_pos + 0.2, mean_errors, 0.4, label='Mean Error', alpha=0.7)
            axes[1, 0].set_xlabel('Test Cases')
            axes[1, 0].set_ylabel('Relative Error (%)')
            axes[1, 0].set_title('Error Summary')
            axes[1, 0].set_xticks(x_pos)
            axes[1, 0].set_xticklabels([name.replace('_', '\n') for name in case_names], rotation=45, ha='right')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance comparison
        jorg_times = [jorg_results[name]["timing"] for name in jorg_results 
                     if jorg_results[name].get("success", False)]
        korg_times = [korg_results[name]["timing"] for name in korg_results 
                     if name in jorg_results and jorg_results[name].get("success", False)]
        
        if jorg_times and korg_times:
            axes[1, 1].scatter(korg_times, jorg_times, alpha=0.7)
            max_time = max(max(jorg_times), max(korg_times))
            axes[1, 1].plot([0, max_time], [0, max_time], 'k--', alpha=0.5, label='Equal time')
            axes[1, 1].set_xlabel('Korg Time (s)')
            axes[1, 1].set_ylabel('Jorg Time (s)')
            axes[1, 1].set_title('Performance Comparison')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = Path(__file__).parent / "jorg_korg_synth_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Comparison plots saved to {output_file}")
        
        try:
            plt.show()
            print("✓ Plots displayed")
        except:
            print("⚠ Cannot display plots (no GUI)")
        
        plt.close()
        
    except ImportError:
        print("⚠ Matplotlib not available, skipping plots")
    except Exception as e:
        print(f"✗ Plotting failed: {e}")


def main():
    """Main comparison test"""
    print("🌟 Jorg vs Korg synth() Function Comparison")
    print("Testing identical parameters to validate implementation")
    
    # Test Jorg synth function
    jorg_results, jorg_errors = test_jorg_synth_cases()
    
    # Load Korg reference data
    korg_results = load_korg_reference()
    
    # Compare results
    comparisons = compare_results(jorg_results, korg_results)
    
    # Create visualizations
    if comparisons:
        create_comparison_plots(jorg_results, korg_results, comparisons)
    
    # Save results
    output_dir = Path(__file__).parent
    
    # Save Jorg results
    jorg_output = {}
    for name, result in jorg_results.items():
        # Convert numpy arrays to lists for JSON serialization
        if result.get("success", False):
            jorg_output[name] = {
                "parameters": result["parameters"],
                "wavelengths": result["wavelengths"].tolist() if isinstance(result["wavelengths"], np.ndarray) else result["wavelengths"],
                "flux": result["flux"].tolist() if isinstance(result["flux"], np.ndarray) else result["flux"],
                "continuum": result["continuum"].tolist() if isinstance(result["continuum"], np.ndarray) else result["continuum"],
                "timing": result["timing"],
                "n_points": result["n_points"],
                "flux_stats": result["flux_stats"],
                "continuum_stats": result["continuum_stats"],
                "success": True
            }
        else:
            jorg_output[name] = result
    
    with open(output_dir / "jorg_synth_results.json", 'w') as f:
        json.dump(jorg_output, f, indent=2)
    
    # Save comparisons
    if comparisons:
        with open(output_dir / "synth_comparison_results.json", 'w') as f:
            json.dump(comparisons, f, indent=2, default=str)
    
    # Final summary
    print("\n" + "=" * 60)
    print("JORG vs KORG synth() COMPARISON SUMMARY")
    print("=" * 60)
    
    jorg_successes = sum(1 for r in jorg_results.values() if r.get("success", False))
    total_tests = len(jorg_results)
    
    print(f"Jorg Tests: {jorg_successes}/{total_tests} successful")
    print(f"Jorg Errors: {len(jorg_errors)}")
    
    if comparisons:
        excellent = sum(1 for c in comparisons.values() if "Excellent" in c.get("agreement_level", ""))
        good = sum(1 for c in comparisons.values() if "Good" in c.get("agreement_level", ""))
        acceptable = sum(1 for c in comparisons.values() if "Acceptable" in c.get("agreement_level", ""))
        total_compared = len(comparisons)
        
        print(f"Korg Comparisons: {total_compared} cases")
        print(f"  Excellent agreement: {excellent}")
        print(f"  Good agreement: {good}")
        print(f"  Acceptable agreement: {acceptable}")
        print(f"  Issues: {total_compared - excellent - good - acceptable}")
        
        if excellent + good >= total_compared * 0.8:
            print("\n🎉 EXCELLENT: Jorg synth() closely matches Korg.jl!")
        elif excellent + good + acceptable >= total_compared * 0.7:
            print("\n✅ GOOD: Jorg synth() shows good agreement with Korg.jl")
        else:
            print("\n⚠️ MIXED: Some differences between Jorg and Korg synth()")
    else:
        print("Korg Comparison: Not available")
    
    print(f"\nFiles saved:")
    print(f"  - jorg_synth_results.json")
    print(f"  - jorg_synth_[case].csv (individual spectra)")
    if comparisons:
        print(f"  - synth_comparison_results.json")
        print(f"  - jorg_korg_synth_comparison.png")
    
    print("=" * 60)
    
    return jorg_successes >= total_tests * 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)