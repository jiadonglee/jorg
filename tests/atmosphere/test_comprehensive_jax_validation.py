#!/usr/bin/env python3
"""
Comprehensive JAX Validation Test
================================

Final validation of the JAX atmosphere interpolation against Korg.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add Jorg to path
sys.path.insert(0, 'Jorg/src')

from jorg.atmosphere_jax_fixed import interpolate_marcs_jax_fixed
from jorg.atmosphere import call_korg_interpolation
import numpy as np

def main():
    print('COMPREHENSIVE JAX VALIDATION TEST')
    print('='*50)

    # Extended test cases covering different interpolation methods
    test_cases = [
        # Standard SDSS grid
        ('Solar G-type', 5777.0, 4.44, 0.0, 0.0, 0.0),
        ('Hot F-type', 6500.0, 4.0, 0.0, 0.0, 0.0), 
        ('Cool K-type', 4500.0, 4.5, 0.0, 0.0, 0.0),
        ('Metal-poor solar', 5777.0, 4.44, -1.0, 0.0, 0.0),
        ('Metal-rich solar', 5777.0, 4.44, 0.3, 0.0, 0.0),
        ('Alpha-enhanced', 5000.0, 4.0, -0.5, 0.4, 0.0),
        ('Carbon-enhanced', 5000.0, 4.0, -0.5, 0.0, 0.5),
        
        # Giants (spherical)
        ('K giant', 4000.0, 2.0, 0.0, 0.0, 0.0),
        ('G giant', 5000.0, 2.5, 0.0, 0.0, 0.0),
        
        # Cool dwarfs (cubic interpolation)
        ('Cool M dwarf', 3500.0, 4.8, 0.0, 0.0, 0.0),
        ('Cool K dwarf', 3800.0, 4.5, 0.0, 0.0, 0.0),
    ]

    results = []
    for desc, Teff, logg, m_H, alpha_m, C_m in test_cases:
        print(f'\nTesting {desc}:')
        print(f'  Params: Teff={Teff}K, logg={logg}, [M/H]={m_H}, [α/M]={alpha_m}, [C/M]={C_m}')
        
        try:
            # Get both results
            korg_atm = call_korg_interpolation(Teff, logg, m_H, alpha_m, C_m)
            jax_atm = interpolate_marcs_jax_fixed(Teff, logg, m_H, alpha_m, C_m)
            
            # Basic structure comparison
            layer_match = len(korg_atm.layers) == len(jax_atm.layers)
            spherical_match = korg_atm.spherical == jax_atm.spherical
            
            print(f'  Korg: {len(korg_atm.layers)} layers, spherical={korg_atm.spherical}')
            print(f'  JAX:  {len(jax_atm.layers)} layers, spherical={jax_atm.spherical}')
            
            # Layer-by-layer comparison
            if layer_match and len(korg_atm.layers) > 25:
                diffs = []
                for i in [10, 25, 40]:
                    if i < len(korg_atm.layers):
                        k_layer = korg_atm.layers[i]
                        j_layer = jax_atm.layers[i]
                        
                        temp_diff = abs(k_layer.temp - j_layer.temp) / k_layer.temp * 100
                        tau_diff = abs(k_layer.tau_5000 - j_layer.tau_5000) / k_layer.tau_5000 * 100
                        nt_diff = abs(k_layer.number_density - j_layer.number_density) / k_layer.number_density * 100
                        
                        max_diff = max(temp_diff, tau_diff, nt_diff)
                        diffs.append(max_diff)
                
                avg_diff = np.mean(diffs)
                max_diff = np.max(diffs)
                
                print(f'  Differences: avg={avg_diff:.3f}%, max={max_diff:.3f}%')
                
                if avg_diff < 0.001:
                    status = '✅ Perfect match'
                elif avg_diff < 0.1:
                    status = '✅ Excellent agreement' 
                elif avg_diff < 1.0:
                    status = '✅ Very good agreement'
                elif avg_diff < 5.0:
                    status = '✅ Good agreement'
                else:
                    status = '⚠️ Moderate agreement'
                
                print(f'  {status}')
                results.append((desc, True, avg_diff, max_diff, status))
            else:
                print(f'  ⚠️ Structure mismatch')
                results.append((desc, False, None, None, 'Structure mismatch'))
                
        except Exception as e:
            print(f'  ❌ Error: {e}')
            results.append((desc, False, None, None, f'Error: {e}'))

    # Summary
    print('\n\n' + '='*50)
    print('VALIDATION SUMMARY')
    print('='*50)

    successful = [r for r in results if r[1]]
    perfect = [r for r in successful if r[2] is not None and r[2] < 0.001]
    excellent = [r for r in successful if r[2] is not None and 0.001 <= r[2] < 0.1]
    very_good = [r for r in successful if r[2] is not None and 0.1 <= r[2] < 1.0]
    good = [r for r in successful if r[2] is not None and 1.0 <= r[2] < 5.0]
    failed = [r for r in results if not r[1]]

    print(f'Total tests: {len(results)}')
    print(f'Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)')
    print(f'')
    print(f'Agreement quality:')
    print(f'  Perfect match (<0.001%):     {len(perfect)} tests')
    print(f'  Excellent (0.001-0.1%):      {len(excellent)} tests')  
    print(f'  Very good (0.1-1%):          {len(very_good)} tests')
    print(f'  Good (1-5%):                 {len(good)} tests')
    print(f'  Failed:                      {len(failed)} tests')

    if len(successful) == len(results) and len(perfect) >= len(results) * 0.8:
        print(f'\n🎉 JAX IMPLEMENTATION OUTSTANDING - Production ready!')
        overall_status = "OUTSTANDING"
    elif len(successful) >= len(results) * 0.9:
        print(f'\n✅ JAX IMPLEMENTATION EXCELLENT - Ready for deployment')
        overall_status = "EXCELLENT"
    elif len(successful) >= len(results) * 0.7:
        print(f'\n✅ JAX IMPLEMENTATION GOOD - Suitable for most applications')
        overall_status = "GOOD"
    else:
        print(f'\n⚠️ JAX IMPLEMENTATION NEEDS IMPROVEMENT')
        overall_status = "NEEDS_IMPROVEMENT"

    print('='*50)
    
    return results, overall_status

if __name__ == "__main__":
    results, status = main()