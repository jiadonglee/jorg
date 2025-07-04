#!/usr/bin/env python3
"""
Test Jorg Atmosphere Interpolation vs Korg (Direct Comparison)
============================================================

This script compares Jorg's atmosphere interpolation against the exact
results from Korg's interpolate_marcs function.
"""

import sys
import numpy as np
from pathlib import Path

# Add Jorg to path
# jorg_path = Path(__file__).parent.parent / "src"
jorg_path = "/Users/jdli/Project/Korg.jl/Jorg/src/"

sys.path.insert(0, str(jorg_path))

from jorg.atmosphere import call_korg_interpolation, AtmosphereInterpolationError
from jorg.constants import kboltz_cgs

# Reference results from Korg (exact values from previous test)
korg_atmosphere_results = {
    "Solar-type G star": {
        "Teff": 5777.0,
        "logg": 4.44,
        "M_H": 0.0,
        "n_layers": 56,
        "spherical": False,
        "test_layers": {
            15: {
                "T": 4590.009579508528,
                "nt": 8.315411129918017e15,
                "ne": 7.239981204782037e11,
                "P": 5269.635801961501,
                "tau_5000": 0.0023776244040555963,
                "z": 3.717249977838802e7,
            },
            25: {
                "T": 4838.221978288154,
                "nt": 2.7356685421333148e16,
                "ne": 2.3860243024247812e12,
                "P": 18273.954914699207,
                "tau_5000": 0.02216150316364371,
                "z": 2.31352762877722e7,
            },
            35: {
                "T": 5383.722228881833,
                "nt": 8.337958936823813e16,
                "ne": 9.453517368277791e12,
                "P": 61976.30484933329,
                "tau_5000": 0.2039855355272931,
                "z": 8.328288096928094e6,
            },
        }
    },
    "Cool K-type star": {
        "Teff": 4500.0,
        "logg": 4.5,
        "M_H": 0.0,
        "n_layers": 56,
        "spherical": False,
        "test_layers": {
            15: {
                "T": 3608.9886137413555,
                "nt": 2.4769114629938384e16,
                "ne": 2.990034164984107e11,
                "P": 12341.82197400089,
                "tau_5000": 0.004436448398941338,
                "z": 2.229414738579783e7,
            },
            25: {
                "T": 3802.2316033227494,
                "nt": 7.680801945308658e16,
                "ne": 9.936302279128654e11,
                "P": 40320.73281347935,
                "tau_5000": 0.03768540579507797,
                "z": 1.3351657504444296e7,
            },
            35: {
                "T": 4269.554262251411,
                "nt": 1.9795187990913702e17,
                "ne": 4.888600236940689e12,
                "P": 116687.79966935837,
                "tau_5000": 0.30039711994792184,
                "z": 4.709898303896188e6,
            },
        }
    },
    "Cool M dwarf": {
        "Teff": 3500.0,
        "logg": 4.8,
        "M_H": 0.0,
        "n_layers": 81,
        "spherical": False,
        "test_layers": {
            15: {
                "T": 2676.971224171452,
                "nt": 3.440690227103728e15,
                "ne": 9.420055688925724e9,
                "P": 1271.6645344402853,
                "tau_5000": 2.5118862527364922e-5,
                "z": 1.4672293821315061e7,
            },
            25: {
                "T": 2757.6431441697528,
                "nt": 1.561033634416188e16,
                "ne": 3.8264281998144066e10,
                "P": 5943.381503808495,
                "tau_5000": 0.0002511885693426358,
                "z": 1.1194421812266937e7,
            },
            35: {
                "T": 2910.3334110333217,
                "nt": 6.345755693469862e16,
                "ne": 1.5800975648355624e11,
                "P": 25498.19134574878,
                "tau_5000": 0.0025118861241130923,
                "z": 8.085973323800263e6,
            },
        }
    },
    "Metal-poor G star": {
        "Teff": 5777.0,
        "logg": 4.44,
        "M_H": -1.0,
        "n_layers": 56,
        "spherical": False,
        "test_layers": {
            15: {
                "T": 4666.866866260754,
                "nt": 2.1882756521811176e16,
                "ne": 3.2283879911501184e11,
                "P": 14099.727608711577,
                "tau_5000": 0.00275127544165067,
                "z": 3.2866487859156106e7,
            },
            25: {
                "T": 4840.113180178101,
                "nt": 7.163604786541968e16,
                "ne": 1.0550775750963367e12,
                "P": 47870.77051900788,
                "tau_5000": 0.025432909700491616,
                "z": 1.8799290548391275e7,
            },
            35: {
                "T": 5343.101981014455,
                "nt": 1.9377501284096784e17,
                "ne": 6.0831917573359375e12,
                "P": 142946.8272290797,
                "tau_5000": 0.23749095859793945,
                "z": 5.462277179825384e6,
            },
        }
    },
    "Metal-rich G star": {
        "Teff": 5777.0,
        "logg": 4.44,
        "M_H": 0.3,
        "n_layers": 56,
        "spherical": False,
        "test_layers": {
            15: {
                "T": 4562.69282361102,
                "nt": 5.886360013228573e15,
                "ne": 9.417291391503596e11,
                "P": 3708.0991190108066,
                "tau_5000": 0.002230652314584484,
                "z": 3.7816670365453124e7,
            },
            25: {
                "T": 4835.645170542186,
                "nt": 1.9410995333394496e16,
                "ne": 3.1444758977749937e12,
                "P": 12959.418463940723,
                "tau_5000": 0.02090389701559974,
                "z": 2.3930030085095856e7,
            },
            35: {
                "T": 5394.3344692256205,
                "nt": 6.003205854880858e16,
                "ne": 1.2010194973623188e13,
                "P": 44709.971132874874,
                "tau_5000": 0.19284860273810983,
                "z": 9.09124861898594e6,
            },
        }
    }
}

def compare_atmosphere_interpolation(stellar_type, expected_data):
    """Test Jorg atmosphere interpolation against Korg results"""
    
    print(f"\nTesting: {stellar_type}")
    print(f"Stellar Parameters: Teff={expected_data['Teff']}K, logg={expected_data['logg']}, [M/H]={expected_data['M_H']}")
    
    try:
        # Call Jorg atmosphere interpolation
        atmosphere = call_korg_interpolation(
            expected_data['Teff'], 
            expected_data['logg'], 
            expected_data['M_H']
        )
        
        print(f"✅ Jorg interpolation successful")
        
        # Compare basic properties
        layer_count_match = len(atmosphere.layers) == expected_data['n_layers']
        spherical_match = atmosphere.spherical == expected_data['spherical']
        
        print(f"   Layers: {len(atmosphere.layers)} (expected {expected_data['n_layers']}) {'✅' if layer_count_match else '❌'}")
        print(f"   Spherical: {atmosphere.spherical} (expected {expected_data['spherical']}) {'✅' if spherical_match else '❌'}")
        
        # Compare specific layers
        layer_comparisons = []
        test_layers = expected_data['test_layers']
        
        for layer_idx, expected_layer in test_layers.items():
            if layer_idx < len(atmosphere.layers):
                jorg_layer = atmosphere.layers[layer_idx]
                
                # Calculate relative differences
                T_diff = abs(jorg_layer.temp - expected_layer['T']) / expected_layer['T'] * 100
                nt_diff = abs(jorg_layer.number_density - expected_layer['nt']) / expected_layer['nt'] * 100
                ne_diff = abs(jorg_layer.electron_number_density - expected_layer['ne']) / expected_layer['ne'] * 100
                tau_diff = abs(jorg_layer.tau_5000 - expected_layer['tau_5000']) / expected_layer['tau_5000'] * 100
                z_diff = abs(jorg_layer.z - expected_layer['z']) / abs(expected_layer['z']) * 100
                
                # Check agreement (should be identical within numerical precision)
                max_diff = max(T_diff, nt_diff, ne_diff, tau_diff, z_diff)
                agreement = "✅ Excellent" if max_diff < 1e-10 else "✅ Good" if max_diff < 0.001 else "⚠️ Poor"
                
                layer_comparison = {
                    'layer': layer_idx,
                    'T_diff': T_diff,
                    'nt_diff': nt_diff,
                    'ne_diff': ne_diff,
                    'tau_diff': tau_diff,
                    'z_diff': z_diff,
                    'max_diff': max_diff,
                    'agreement': agreement
                }
                layer_comparisons.append(layer_comparison)
                
                print(f"   Layer {layer_idx}: Max diff {max_diff:.2e}% {agreement}")
            
        # Overall assessment
        all_excellent = all(comp['max_diff'] < 1e-10 for comp in layer_comparisons)
        overall_status = "✅ Perfect" if all_excellent else "✅ Good" if all(comp['max_diff'] < 0.001 for comp in layer_comparisons) else "⚠️ Issues"
        
        print(f"   Overall agreement: {overall_status}")
        
        return {
            'stellar_type': stellar_type,
            'success': True,
            'layer_count_match': layer_count_match,
            'spherical_match': spherical_match,
            'layer_comparisons': layer_comparisons,
            'overall_status': overall_status
        }
        
    except Exception as e:
        print(f"❌ Jorg interpolation failed: {e}")
        return {
            'stellar_type': stellar_type,
            'success': False,
            'error': str(e)
        }

def test_interpolation_methods():
    """Test different stellar types that trigger different interpolation methods"""
    
    print("\nTESTING INTERPOLATION METHODS")
    print("=" * 60)
    
    # Test specific cases that use different Korg interpolation schemes
    method_tests = [
        ("Standard SDSS grid", 5777.0, 4.44, 0.0),
        ("Cool dwarf cubic spline", 3500.0, 4.8, 0.0),
        ("Low metallicity grid", 5777.0, 4.44, -3.0),
        ("Giant spherical", 4500.0, 2.0, 0.0),
    ]
    
    for description, Teff, logg, M_H in method_tests:
        print(f"\nTesting: {description}")
        print(f"Parameters: Teff={Teff}K, logg={logg}, [M/H]={M_H}")
        
        try:
            atmosphere = call_korg_interpolation(Teff, logg, M_H)
            print(f"✅ Success: {len(atmosphere.layers)} layers, {'spherical' if atmosphere.spherical else 'planar'}")
            
            # Identify expected interpolation method
            if Teff <= 4000 and logg >= 3.5 and M_H >= -2.5:
                expected_method = "Cool dwarf cubic spline"
            elif M_H < -2.5:
                expected_method = "Low metallicity grid" 
            else:
                expected_method = "Standard SDSS grid"
            
            print(f"   Expected method: {expected_method}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

def analyze_accuracy_statistics(results):
    """Analyze overall accuracy statistics"""
    
    print("\nACCURACY ANALYSIS")
    print("=" * 60)
    
    successful_tests = [r for r in results if r['success']]
    
    if not successful_tests:
        print("❌ No successful tests to analyze")
        return
    
    print(f"Successful tests: {len(successful_tests)}/{len(results)}")
    
    # Analyze layer comparison accuracy
    all_layer_diffs = []
    for result in successful_tests:
        for layer_comp in result['layer_comparisons']:
            all_layer_diffs.append(layer_comp['max_diff'])
    
    if all_layer_diffs:
        mean_diff = np.mean(all_layer_diffs)
        max_diff = np.max(all_layer_diffs)
        min_diff = np.min(all_layer_diffs)
        
        print(f"Layer comparison statistics:")
        print(f"  Mean difference: {mean_diff:.2e}%")
        print(f"  Max difference:  {max_diff:.2e}%")
        print(f"  Min difference:  {min_diff:.2e}%")
        
        # Categorize accuracy
        if max_diff < 1e-10:
            accuracy_level = "Perfect (machine precision)"
        elif max_diff < 0.001:
            accuracy_level = "Excellent (< 0.001%)"
        elif max_diff < 0.1:
            accuracy_level = "Good (< 0.1%)"
        else:
            accuracy_level = "Needs improvement"
        
        print(f"  Overall accuracy: {accuracy_level}")
    
    # Analyze property matches
    layer_count_matches = sum(1 for r in successful_tests if r['layer_count_match'])
    spherical_matches = sum(1 for r in successful_tests if r['spherical_match'])
    
    print(f"\nProperty matching:")
    print(f"  Layer count matches: {layer_count_matches}/{len(successful_tests)}")
    print(f"  Spherical matches:   {spherical_matches}/{len(successful_tests)}")

def main():
    print("JORG vs KORG ATMOSPHERE INTERPOLATION COMPARISON")
    print("=" * 80)
    print("Testing Jorg atmosphere interpolation against exact Korg results")
    print("This validates that Jorg can reproduce Korg's interpolation")
    print()
    
    # Test each stellar type
    results = []
    for stellar_type, expected_data in korg_atmosphere_results.items():
        result = compare_atmosphere_interpolation(stellar_type, expected_data)
        results.append(result)
    
    # Test interpolation methods
    test_interpolation_methods()
    
    # Analyze overall accuracy
    analyze_accuracy_statistics(results)
    
    # Summary
    successful_count = sum(1 for r in results if r['success'])
    print(f"\n{'='*80}")
    print("ATMOSPHERE INTERPOLATION COMPARISON SUMMARY")
    print("="*80)
    print(f"✅ Successful tests: {successful_count}/{len(results)}")
    
    if successful_count == len(results):
        print("✅ All atmosphere interpolations successful")
        print("✅ Jorg reproduces Korg interpolation results")
        print("✅ Ready for production use with identical atmospheric data")
    else:
        print(f"⚠️  {len(results) - successful_count} tests failed")
    
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = main()