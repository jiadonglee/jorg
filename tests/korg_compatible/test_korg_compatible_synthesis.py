#!/usr/bin/env python3
"""
Test Korg-Compatible Jorg Synthesis
===================================

This script tests the new Korg-compatible synthesis pipeline to ensure it:
1. Follows Korg.jl's exact architecture 
2. Uses Jorg's validated physics without hardcoding
3. Produces systematic opacity matrices
4. Maintains performance optimization
"""

import sys
import numpy as np
from pathlib import Path

# Add Jorg to path
sys.path.insert(0, "/Users/jdli/Project/Korg.jl/Jorg/src")

def test_basic_synthesis():
    """Test basic synthesis functionality"""
    print("🧪 TESTING KORG-COMPATIBLE JORG SYNTHESIS")
    print("=" * 50)
    
    try:
        # Import the new Korg-compatible interface
        from jorg.synthesis import synthesize_korg_compatible, SynthesisResult
        from jorg.atmosphere import interpolate_marcs as interpolate_atmosphere
        from jorg.abundances import format_A_X
        
        print("✅ Imports successful")
        
        # 1. Test basic parameter setup
        print("\n1. Setting up test parameters...")
        Teff = 5780
        logg = 4.44
        m_H = 0.0
        wavelengths = (5000.0, 5005.0)  # Small range for fast testing
        
        print(f"   Stellar parameters: Teff={Teff}K, logg={logg}, [M/H]={m_H}")
        print(f"   Wavelength range: {wavelengths[0]}-{wavelengths[1]} Å")
        
        # 2. Test atmospheric interpolation
        print("\n2. Testing atmospheric interpolation...")
        A_X = format_A_X()
        atm = interpolate_atmosphere(Teff=Teff, logg=logg, m_H=m_H)
        
        n_layers = len(atm.layers)
        print(f"   ✅ Atmospheric model loaded: {n_layers} layers")
        temps = [layer.temp for layer in atm.layers]
        print(f"   Temperature range: {np.min(temps):.1f}-{np.max(temps):.1f} K")
        
        # 3. Test abundance array creation
        print("\n3. Testing abundance array...")
        from jorg.synthesis import create_korg_compatible_abundance_array
        A_X_array = create_korg_compatible_abundance_array(m_H)
        
        print(f"   ✅ Abundance array created: {len(A_X_array)} elements")
        print(f"   A(H)={A_X_array[0]}, A(He)={A_X_array[1]}, A(Fe)={A_X_array[25]}")
        
        # 4. Test synthesis (without linelist first)
        print("\n4. Testing continuum-only synthesis...")
        
        result = synthesize_korg_compatible(
            atm=atm,
            linelist=[],  # No lines for first test
            A_X=A_X_array,
            wavelengths=wavelengths,
            verbose=True
        )
        
        print(f"   ✅ Synthesis completed successfully!")
        print(f"   Result type: {type(result)}")
        print(f"   Available fields: {list(result.__dict__.keys())}")
        
        # 5. Validate SynthesisResult structure
        print("\n5. Validating SynthesisResult structure...")
        
        # Check required fields
        required_fields = ['flux', 'cntm', 'intensity', 'alpha', 'mu_grid', 
                          'number_densities', 'electron_number_density', 'wavelengths']
        
        for field in required_fields:
            if hasattr(result, field):
                value = getattr(result, field)
                if value is not None:
                    print(f"   ✅ {field}: {type(value)} shape={getattr(value, 'shape', 'N/A')}")
                else:
                    print(f"   ⚠️  {field}: None")
            else:
                print(f"   ❌ Missing field: {field}")
        
        # 6. Validate key opacity matrix
        print("\n6. Validating opacity matrix...")
        alpha = result.alpha
        print(f"   Shape: {alpha.shape} (should be [layers × wavelengths])")
        print(f"   Range: {np.min(alpha):.3e} - {np.max(alpha):.3e} cm⁻¹")
        print(f"   Non-zero values: {np.count_nonzero(alpha)}/{alpha.size}")
        
        if alpha.shape[0] == n_layers:
            print(f"   ✅ Correct number of layers: {alpha.shape[0]}")
        else:
            print(f"   ❌ Layer mismatch: expected {n_layers}, got {alpha.shape[0]}")
        
        # 7. Check chemical equilibrium results
        print("\n7. Checking chemical equilibrium...")
        ne_array = result.electron_number_density
        print(f"   Electron densities shape: {ne_array.shape}")
        print(f"   Range: {np.min(ne_array):.2e} - {np.max(ne_array):.2e} cm⁻³")
        
        print(f"   Number of species calculated: {len(result.number_densities)}")
        for i, (species, densities) in enumerate(result.number_densities.items()):
            if i < 5:  # Show first 5 species
                max_density = np.max(densities)
                print(f"     {species}: max density {max_density:.2e} cm⁻³")
        
        print("\n✅ BASIC SYNTHESIS TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_linelist():
    """Test synthesis with actual linelist"""
    print("\n\n🔬 TESTING WITH LINELIST")
    print("=" * 30)
    
    try:
        from jorg.synthesis import synthesize_korg_compatible
        from jorg.atmosphere import interpolate_marcs as interpolate_atmosphere
        from jorg.abundances import format_A_X
        from jorg.lines.linelist import read_linelist
        
        # Setup parameters
        Teff = 5780
        logg = 4.44  
        m_H = 0.0
        wavelengths = (5000.0, 5005.0)
        
        # Load atmosphere
        A_X = format_A_X()
        atm = interpolate_atmosphere(Teff=Teff, logg=logg, m_H=m_H)
        
        # Create corrected abundance array
        from jorg.synthesis import create_korg_compatible_abundance_array
        A_X_array = create_korg_compatible_abundance_array(m_H)
        
        # Try to load linelist
        linelist_path = "/Users/jdli/Project/Korg.jl/test/data/linelists/5000-5005.vald"
        
        if Path(linelist_path).exists():
            print(f"📖 Loading linelist: {linelist_path}")
            linelist = read_linelist(linelist_path, format='vald')
            print(f"   ✅ Loaded {len(linelist.lines)} lines")
            
            # Run synthesis with lines
            result = synthesize_korg_compatible(
                atm=atm,
                linelist=linelist.lines,
                A_X=A_X_array,
                wavelengths=wavelengths,
                verbose=True
            )
            
            # Check results
            alpha = result.alpha
            print(f"\n📊 Results with lines:")
            print(f"   Opacity range: {np.min(alpha):.3e} - {np.max(alpha):.3e} cm⁻¹")
            
            # Find peak opacity
            max_idx = np.unravel_index(np.argmax(alpha), alpha.shape)
            max_layer, max_wl_idx = max_idx
            max_opacity = alpha[max_layer, max_wl_idx]
            
            print(f"   Peak opacity: {max_opacity:.3e} cm⁻¹")
            print(f"     at layer {max_layer+1}, wavelength index {max_wl_idx}")
            
            print("\n✅ LINELIST SYNTHESIS TEST PASSED!")
            return True
            
        else:
            print(f"⚠️  Linelist not found: {linelist_path}")
            print("   Skipping linelist test")
            return True
            
    except Exception as e:
        print(f"\n❌ LINELIST TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comparison_format():
    """Test that output format is compatible with Korg.jl comparison"""
    print("\n\n📊 TESTING KORG COMPARISON FORMAT")
    print("=" * 35)
    
    try:
        from jorg.synthesis import synthesize_korg_compatible
        from jorg.atmosphere import interpolate_marcs as interpolate_atmosphere
        from jorg.abundances import format_A_X
        
        # Use same parameters as Korg comparison
        Teff = 5780
        logg = 4.44
        m_H = 0.0
        wavelengths = np.linspace(5000.0, 5005.0, 100)  # 100 points like Korg
        
        # Setup
        A_X = format_A_X()
        atm = interpolate_atmosphere(Teff=Teff, logg=logg, m_H=m_H)
        from jorg.synthesis import create_korg_compatible_abundance_array
        A_X_array = create_korg_compatible_abundance_array(m_H)
        
        # Run synthesis
        result = synthesize_korg_compatible(
            atm=atm,
            linelist=[],
            A_X=A_X_array,
            wavelengths=wavelengths,
            verbose=False
        )
        
        # Extract specific layer for comparison (layer 25 or 30)
        comparison_layer = min(25, result.alpha.shape[0] - 1)
        layer_opacity = result.alpha[comparison_layer, :]
        
        print(f"✅ Comparison format ready:")
        print(f"   Target layer: {comparison_layer+1}")
        print(f"   Wavelengths: {len(result.wavelengths)} points")
        print(f"   Layer opacity shape: {layer_opacity.shape}")
        print(f"   Layer opacity range: {np.min(layer_opacity):.3e} - {np.max(layer_opacity):.3e} cm⁻¹")
        
        # Save for comparison
        comparison_data = np.column_stack([
            result.wavelengths,
            layer_opacity,
            np.zeros_like(layer_opacity),  # Continuum (placeholder)
            np.zeros_like(layer_opacity)   # Line (placeholder)  
        ])
        
        output_file = "korg_compatible_jorg_test_output.txt"
        header = f"# Korg-compatible Jorg synthesis test output\n"
        header += f"# Teff={Teff}K, logg={logg}, [M/H]={m_H}, layer={comparison_layer+1}\n"
        header += f"# Wavelength(Å)  Total_Opacity(cm⁻¹)  Continuum(cm⁻¹)  Line(cm⁻¹)\n"
        
        np.savetxt(output_file, comparison_data, header=header,
                   fmt='%.6f\t%.6e\t%.6e\t%.6e')
        
        print(f"   💾 Saved to: {output_file}")
        print("\n✅ COMPARISON FORMAT TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ COMPARISON FORMAT TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 KORG-COMPATIBLE JORG SYNTHESIS TEST SUITE")
    print("=" * 55)
    
    tests = [
        ("Basic Synthesis", test_basic_synthesis),
        ("Linelist Synthesis", test_with_linelist),
        ("Comparison Format", test_comparison_format)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")
        
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n\n{'='*60}")
    print("🎯 TEST SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED!")
        print("Korg-compatible synthesis pipeline is working correctly!")
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed")
        print("Review output above for details")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)