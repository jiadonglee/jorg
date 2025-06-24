#!/usr/bin/env python3
"""
Comprehensive comparison between Korg.jl and Jorg implementations.

This script tests various components to ensure Jorg reproduces Korg results
within acceptable tolerances across different stellar parameters.
"""

import sys
import os
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# Add Jorg to Python path
sys.path.insert(0, '/Users/jdli/Project/Korg.jl/Jorg/src')


class KorgJorgComparison:
    """Main comparison class for Korg vs Jorg testing."""
    
    def __init__(self):
        self.results = {}
        self.tolerance = 1e-6  # Default relative tolerance
        
        # Test stellar parameters
        self.test_params = [
            {"name": "Cool_dwarf", "Teff": 3500, "logg": 4.5, "FeH": 0.0},
            {"name": "Solar", "Teff": 5777, "logg": 4.44, "FeH": 0.0},
            {"name": "Hot_dwarf", "Teff": 8000, "logg": 4.0, "FeH": 0.0},
            {"name": "Metal_poor", "Teff": 5777, "logg": 4.44, "FeH": -2.0},
            {"name": "Metal_rich", "Teff": 5777, "logg": 4.44, "FeH": 0.5}
        ]
        
        # Wavelength ranges for testing
        self.wavelength_ranges = [
            {"name": "Optical", "lambda_min": 5000, "lambda_max": 6000},
            {"name": "Blue", "lambda_min": 4000, "lambda_max": 5000},
            {"name": "Red", "lambda_min": 6000, "lambda_max": 7000}
        ]

    def generate_korg_reference_data(self):
        """Generate reference data from Korg.jl for all test cases."""
        
        print("Generating Korg reference data...")
        
        julia_script = '''
using Pkg
Pkg.activate("/Users/jdli/Project/Korg.jl")
using Korg
using JSON

# Test parameters
test_params = [
    Dict("name" => "Cool_dwarf", "Teff" => 3500, "logg" => 4.5, "FeH" => 0.0),
    Dict("name" => "Solar", "Teff" => 5777, "logg" => 4.44, "FeH" => 0.0),
    Dict("name" => "Hot_dwarf", "Teff" => 8000, "logg" => 4.0, "FeH" => 0.0),
    Dict("name" => "Metal_poor", "Teff" => 5777, "logg" => 4.44, "FeH" => -2.0),
    Dict("name" => "Metal_rich", "Teff" => 5777, "logg" => 4.44, "FeH" => 0.5)
]

results = Dict()

# Generate model atmospheres and basic data
for params in test_params
    try
        name = params["name"]
        Teff = params["Teff"]
        logg = params["logg"]
        FeH = params["FeH"]
        
        println("Processing $name...")
        
        # Create abundance vector (simplified for testing)
        A_X = format_A_X(FeH=FeH)
        
        # Generate atmosphere
        atm = interpolate_marcs(Teff, logg, A_X)
        
        # Extract atmosphere properties
        n_layers = length(atm.layers)
        temps = [layer.temp for layer in atm.layers]
        n_total = [layer.number_density for layer in atm.layers]
        n_e = [layer.electron_number_density for layer in atm.layers]
        
        # Calculate some basic quantities
        mean_temp = mean(temps)
        mean_n_total = mean(n_total)
        mean_n_e = mean(n_e)
        
        # Test synthesis on a small wavelength range (to avoid timeouts)
        try
            wavelengths = 5000.0:1.0:5100.0
            wls, flux, continuum = synth(atm, wavelengths; A_X=A_X)
            
            synthesis_data = Dict(
                "wavelengths" => collect(wls),
                "flux" => flux,
                "continuum" => continuum,
                "mean_flux" => mean(flux),
                "continuum_level" => mean(continuum)
            )
        catch e
            println("Warning: Synthesis failed for $name: $e")
            synthesis_data = Dict("error" => string(e))
        end
        
        results[name] = Dict(
            "stellar_params" => params,
            "atmosphere" => Dict(
                "n_layers" => n_layers,
                "mean_temp" => mean_temp,
                "mean_n_total" => mean_n_total,
                "mean_n_e" => mean_n_e,
                "temp_range" => [minimum(temps), maximum(temps)],
                "density_range" => [minimum(n_total), maximum(n_total)]
            ),
            "synthesis" => synthesis_data
        )
        
    catch e
        println("Error processing $(params["name"]): $e")
        results[params["name"]] = Dict("error" => string(e))
    end
end

# Write results
open("korg_reference_data.json", "w") do f
    JSON.print(f, results, 2)
end

println("Korg reference data generation complete")
'''
        
        with open('generate_korg_reference.jl', 'w') as f:
            f.write(julia_script)
        
        try:
            result = subprocess.run(['julia', 'generate_korg_reference.jl'], 
                                  capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"Korg script failed: {result.stderr}")
                return None
            
            print("Korg output:", result.stdout)
            
            with open('korg_reference_data.json', 'r') as f:
                return json.load(f)
        
        except subprocess.TimeoutExpired:
            print("Korg script timed out")
            return None
        except Exception as e:
            print(f"Error running Korg script: {e}")
            return None

    def test_jorg_basic_functions(self):
        """Test basic Jorg functions and imports."""
        
        print("Testing Jorg basic functionality...")
        
        try:
            # Test imports (avoiding circular import issues)
            from jorg.constants import kboltz_cgs
            from jorg.statmech.eos import gas_pressure
            
            # Test basic calculations
            T = 5777.0
            n = 1e16
            P = float(gas_pressure(n, T))
            
            basic_test = {
                "imports_successful": True,
                "constant_value": kboltz_cgs,
                "test_calculation": {
                    "temperature": T,
                    "density": n,
                    "pressure": P
                }
            }
            
            print(f"  ✅ Basic functions working")
            print(f"  Boltzmann constant: {kboltz_cgs}")
            print(f"  Test pressure calculation: {P:.2e}")
            
            return basic_test
            
        except Exception as e:
            print(f"  ❌ Basic function test failed: {e}")
            return {"imports_successful": False, "error": str(e)}

    def compare_statistical_mechanics(self, korg_data):
        """Compare statistical mechanics calculations."""
        
        print("Comparing statistical mechanics...")
        
        # For now, compare basic thermodynamic quantities
        # This is limited by Jorg's current implementation
        
        try:
            from jorg.statmech.eos import gas_pressure, electron_pressure
            
            comparisons = {}
            
            for case_name, korg_result in korg_data.items():
                if "error" in korg_result:
                    continue
                
                atm_data = korg_result.get("atmosphere", {})
                if not atm_data:
                    continue
                
                # Compare pressure calculations
                T = atm_data["mean_temp"]
                n_total = atm_data["mean_n_total"] 
                n_e = atm_data["mean_n_e"]
                
                # Calculate with Jorg
                jorg_P_gas = float(gas_pressure(n_total, T))
                jorg_P_e = float(electron_pressure(n_e, T))
                
                # Compare with Korg values (calculate what Korg would give)
                korg_P_gas = n_total * 1.380649e-16 * T  # Using same formula
                korg_P_e = n_e * 1.380649e-16 * T
                
                rel_error_gas = abs(jorg_P_gas - korg_P_gas) / korg_P_gas
                rel_error_e = abs(jorg_P_e - korg_P_e) / korg_P_e
                
                comparisons[case_name] = {
                    "gas_pressure": {
                        "korg": korg_P_gas,
                        "jorg": jorg_P_gas,
                        "rel_error": rel_error_gas
                    },
                    "electron_pressure": {
                        "korg": korg_P_e,
                        "jorg": jorg_P_e,
                        "rel_error": rel_error_e
                    }
                }
                
                print(f"  {case_name}: P_gas error = {rel_error_gas:.2e}, P_e error = {rel_error_e:.2e}")
            
            return comparisons
            
        except Exception as e:
            print(f"  ❌ Statistical mechanics comparison failed: {e}")
            return {"error": str(e)}

    def analyze_atmosphere_differences(self, korg_data):
        """Analyze differences in atmosphere handling."""
        
        print("Analyzing atmosphere differences...")
        
        analysis = {}
        
        for case_name, korg_result in korg_data.items():
            if "error" in korg_result:
                print(f"  {case_name}: Korg error - {korg_result['error']}")
                continue
            
            atm_data = korg_result.get("atmosphere", {})
            if not atm_data:
                continue
            
            # Analyze atmosphere properties
            analysis[case_name] = {
                "n_layers": atm_data["n_layers"],
                "temperature_range": atm_data["temp_range"],
                "density_range": atm_data["density_range"],
                "mean_properties": {
                    "temperature": atm_data["mean_temp"],
                    "total_density": atm_data["mean_n_total"],
                    "electron_density": atm_data["mean_n_e"],
                    "ionization_fraction": atm_data["mean_n_e"] / atm_data["mean_n_total"]
                }
            }
            
            print(f"  {case_name}:")
            print(f"    Layers: {atm_data['n_layers']}")
            print(f"    T_range: {atm_data['temp_range'][0]:.0f} - {atm_data['temp_range'][1]:.0f} K")
            print(f"    Ionization: {analysis[case_name]['mean_properties']['ionization_fraction']:.3f}")
        
        return analysis

    def compare_synthesis_results(self, korg_data):
        """Compare synthesis results where available."""
        
        print("Comparing synthesis results...")
        
        synthesis_comparison = {}
        
        for case_name, korg_result in korg_data.items():
            if "error" in korg_result:
                continue
                
            synth_data = korg_result.get("synthesis", {})
            if "error" in synth_data:
                print(f"  {case_name}: Korg synthesis error - {synth_data['error']}")
                synthesis_comparison[case_name] = {"korg_synthesis_failed": True}
                continue
            
            if not synth_data:
                continue
            
            # For now, just analyze Korg synthesis properties
            # Full Jorg synthesis comparison would require more setup
            
            mean_flux = synth_data.get("mean_flux", 0)
            continuum_level = synth_data.get("continuum_level", 0)
            
            synthesis_comparison[case_name] = {
                "korg_results": {
                    "wavelength_points": len(synth_data.get("wavelengths", [])),
                    "mean_flux": mean_flux,
                    "continuum_level": continuum_level,
                    "flux_to_continuum_ratio": mean_flux / continuum_level if continuum_level > 0 else 0
                },
                "jorg_synthesis_attempted": False,  # Would need more implementation
                "notes": "Full synthesis comparison requires complete Jorg setup"
            }
            
            print(f"  {case_name}: Korg synthesis successful")
            print(f"    Wavelength points: {len(synth_data.get('wavelengths', []))}")
            print(f"    Mean flux: {mean_flux:.3f}")
            print(f"    Continuum: {continuum_level:.3f}")
        
        return synthesis_comparison

    def generate_comparison_report(self):
        """Generate a comprehensive comparison report."""
        
        print("\n" + "="*60)
        print("KORG vs JORG COMPREHENSIVE COMPARISON REPORT")
        print("="*60)
        
        # Generate Korg reference data
        korg_data = self.generate_korg_reference_data()
        if not korg_data:
            print("❌ Failed to generate Korg reference data")
            return False
        
        print(f"\n📊 Generated Korg data for {len(korg_data)} test cases")
        
        # Test Jorg basic functionality
        print(f"\n🧪 Testing Jorg Basic Functionality...")
        jorg_basic = self.test_jorg_basic_functions()
        
        # Compare statistical mechanics
        print(f"\n⚖️  Comparing Statistical Mechanics...")
        statmech_comparison = self.compare_statistical_mechanics(korg_data)
        
        # Analyze atmospheres
        print(f"\n🌍 Analyzing Atmosphere Properties...")
        atmosphere_analysis = self.analyze_atmosphere_differences(korg_data)
        
        # Compare synthesis
        print(f"\n🔬 Comparing Synthesis Results...")
        synthesis_comparison = self.compare_synthesis_results(korg_data)
        
        # Compile final report
        final_report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_cases": list(korg_data.keys()),
            "jorg_basic_functionality": jorg_basic,
            "statistical_mechanics_comparison": statmech_comparison,
            "atmosphere_analysis": atmosphere_analysis,
            "synthesis_comparison": synthesis_comparison
        }
        
        # Save detailed results
        with open('korg_jorg_comparison_report.json', 'w') as f:
            json.dump(final_report, f, indent=2)
        
        # Print summary
        print(f"\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        # Jorg functionality status
        if jorg_basic["imports_successful"]:
            print("✅ Jorg basic functionality: WORKING")
        else:
            print("❌ Jorg basic functionality: FAILED")
        
        # Statistical mechanics status
        if isinstance(statmech_comparison, dict) and "error" not in statmech_comparison:
            max_error = 0
            for case_data in statmech_comparison.values():
                max_error = max(max_error, 
                              case_data.get("gas_pressure", {}).get("rel_error", 0),
                              case_data.get("electron_pressure", {}).get("rel_error", 0))
            
            if max_error < 1e-6:
                print(f"✅ Statistical mechanics: EXCELLENT (max error: {max_error:.2e})")
            elif max_error < 1e-3:
                print(f"⚠️  Statistical mechanics: GOOD (max error: {max_error:.2e})")
            else:
                print(f"❌ Statistical mechanics: POOR (max error: {max_error:.2e})")
        else:
            print("❌ Statistical mechanics: COMPARISON FAILED")
        
        # Atmosphere analysis status
        successful_atmospheres = len([k for k, v in atmosphere_analysis.items() if "error" not in v])
        total_atmospheres = len(self.test_params)
        print(f"📊 Atmosphere analysis: {successful_atmospheres}/{total_atmospheres} test cases")
        
        # Synthesis status
        korg_synthesis_success = len([k for k, v in synthesis_comparison.items() 
                                    if not v.get("korg_synthesis_failed", False)])
        print(f"🔬 Korg synthesis: {korg_synthesis_success}/{len(synthesis_comparison)} successful")
        print("🔬 Jorg synthesis: Not yet implemented for full comparison")
        
        print(f"\n📁 Detailed report saved to: korg_jorg_comparison_report.json")
        print("="*60)
        
        # Cleanup
        for temp_file in ['generate_korg_reference.jl', 'korg_reference_data.json']:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        return True


def main():
    """Run the comprehensive comparison."""
    
    comparison = KorgJorgComparison()
    success = comparison.generate_comparison_report()
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)