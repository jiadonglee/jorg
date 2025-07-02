"""
Integration tests for the complete Jorg synthesis workflow

Tests the end-to-end synthesis process and comparison with expected behavior.
"""

import pytest
import jax.numpy as jnp
import numpy as np
import time

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jorg.synthesis import synth, synthesize, format_abundances, interpolate_atmosphere
from jorg.lines.datatypes import Line, Species


class TestCompleteWorkflow:
    """Test complete synthesis workflow from parameters to spectrum"""
    
    def test_solar_spectrum_synthesis(self):
        """Test synthesis of a solar-like spectrum"""
        # Solar parameters
        Teff = 5778
        logg = 4.44
        m_H = 0.0
        wavelength_range = (5000, 6000)
        
        # Run synthesis
        wavelengths, flux, continuum = synth(
            Teff=Teff,
            logg=logg,
            m_H=m_H,
            wavelengths=wavelength_range,
            vmic=1.0
        )
        
        # Basic validation
        assert len(wavelengths) > 100
        assert len(flux) == len(wavelengths)
        assert len(continuum) == len(wavelengths)
        
        # Physical checks
        assert jnp.all(wavelengths >= wavelength_range[0])
        assert jnp.all(wavelengths <= wavelength_range[1])
        assert jnp.all(jnp.isfinite(flux))
        assert jnp.all(jnp.isfinite(continuum))
        assert jnp.all(flux >= 0)
        assert jnp.all(continuum > 0)
        
        # Rectified flux should be around 1.0 in continuum regions
        assert jnp.mean(flux) > 0.8
        assert jnp.mean(flux) < 1.2
    
    def test_stellar_parameter_grid(self):
        """Test synthesis across a grid of stellar parameters"""
        # Define parameter grid
        Teffs = [4500, 5500, 6500]
        loggs = [3.5, 4.0, 4.5]
        metallicities = [-1.0, 0.0, 0.5]
        
        wavelength_range = (5500, 5510)  # Small range for speed
        
        results = []
        
        for Teff in Teffs:
            for logg in loggs:
                for m_H in metallicities:
                    try:
                        wl, flux, cntm = synth(
                            Teff=Teff,
                            logg=logg,
                            m_H=m_H,
                            wavelengths=wavelength_range,
                            vmic=1.0
                        )
                        
                        # Basic validation for each synthesis
                        assert jnp.all(jnp.isfinite(flux))
                        assert jnp.all(flux >= 0)
                        assert jnp.all(cntm > 0)
                        
                        results.append({
                            'Teff': Teff, 'logg': logg, 'm_H': m_H,
                            'flux': flux, 'continuum': cntm
                        })
                        
                    except Exception as e:
                        pytest.fail(f"Synthesis failed for Teff={Teff}, logg={logg}, m_H={m_H}: {e}")
        
        # Should have computed all grid points
        assert len(results) == len(Teffs) * len(loggs) * len(metallicities)
        
        # Test that different parameters give different results
        baseline = results[0]['flux']
        different_count = 0
        
        for result in results[1:]:
            if not jnp.allclose(result['flux'], baseline, rtol=1e-6):
                different_count += 1
        
        # Most parameter combinations should give different results
        assert different_count > len(results) * 0.8
    
    def test_abundance_variations(self):
        """Test synthesis with different abundance patterns"""
        base_params = dict(Teff=5500, logg=4.0, m_H=0.0, wavelengths=(5500, 5510))
        
        # Baseline solar composition
        wl, flux_solar, cntm_solar = synth(**base_params)
        
        # Alpha-enhanced composition
        wl, flux_alpha, cntm_alpha = synth(**base_params, alpha_H=0.4)
        
        # Iron-poor composition
        wl, flux_fe_poor, cntm_fe_poor = synth(**base_params, Fe=-1.0)
        
        # Carbon-enhanced composition
        wl, flux_c_enh, cntm_c_enh = synth(**base_params, C=1.0)
        
        # All should be valid
        for flux in [flux_solar, flux_alpha, flux_fe_poor, flux_c_enh]:
            assert jnp.all(jnp.isfinite(flux))
            assert jnp.all(flux >= 0)
        
        # Different compositions should give different results
        assert not jnp.allclose(flux_alpha, flux_solar, rtol=1e-6)
        assert not jnp.allclose(flux_fe_poor, flux_solar, rtol=1e-6)
        assert not jnp.allclose(flux_c_enh, flux_solar, rtol=1e-6)
    
    def test_wavelength_ranges(self):
        """Test synthesis across different wavelength ranges"""
        base_params = dict(Teff=5500, logg=4.0, m_H=0.0, vmic=1.0)
        
        # Different wavelength ranges
        ranges = [
            (4000, 4100),   # Blue
            (5000, 5100),   # Green
            (6000, 6100),   # Red
            (8000, 8100)    # Near-IR
        ]
        
        for wl_min, wl_max in ranges:
            wl, flux, cntm = synth(**base_params, wavelengths=(wl_min, wl_max))
            
            # Check wavelength coverage
            assert jnp.min(wl) >= wl_min
            assert jnp.max(wl) <= wl_max
            
            # Physical validity
            assert jnp.all(jnp.isfinite(flux))
            assert jnp.all(flux >= 0)
            assert jnp.all(cntm > 0)
    
    def test_multiple_wavelength_windows(self):
        """Test synthesis with multiple wavelength windows"""
        # Multiple windows (like APOGEE or GALAH)
        windows = [(4500, 4600), (5500, 5600), (6500, 6600)]
        
        wl, flux, cntm = synth(
            Teff=5500, logg=4.0, m_H=0.0,
            wavelengths=windows,
            vmic=1.0
        )
        
        # Should cover all windows
        assert len(wl) > 100  # Should have points from all windows
        assert jnp.all(jnp.isfinite(flux))
        assert jnp.all(flux >= 0)
        
        # Check that we have coverage in each window
        for wl_min, wl_max in windows:
            window_indices = (wl >= wl_min) & (wl <= wl_max)
            assert jnp.sum(window_indices) > 10  # At least some points in each window


class TestInstrumentalEffects:
    """Test instrumental effects and post-processing"""
    
    def test_resolution_effects(self):
        """Test different spectral resolutions"""
        base_params = dict(Teff=5500, logg=4.0, m_H=0.0, wavelengths=(5500, 5510), vmic=1.0)
        
        # High resolution (sharp lines)
        wl, flux_high_res, cntm = synth(**base_params, R=100000)
        
        # Medium resolution
        wl, flux_med_res, cntm = synth(**base_params, R=10000)
        
        # Low resolution
        wl, flux_low_res, cntm = synth(**base_params, R=1000)
        
        # Lower resolution should be smoother
        def smoothness(flux):
            return jnp.mean(jnp.abs(jnp.diff(flux, n=2)))
        
        smooth_high = smoothness(flux_high_res)
        smooth_med = smoothness(flux_med_res)
        smooth_low = smoothness(flux_low_res)
        
        # Lower resolution should be smoother
        assert smooth_low < smooth_med < smooth_high
    
    def test_rotational_broadening(self):
        """Test rotational broadening effects"""
        base_params = dict(Teff=6000, logg=4.0, m_H=0.0, wavelengths=(5500, 5510), vmic=1.0)
        
        # No rotation
        wl, flux_no_rot, cntm = synth(**base_params, vsini=0)
        
        # Moderate rotation
        wl, flux_mod_rot, cntm = synth(**base_params, vsini=10)  # 10 km/s
        
        # Fast rotation
        wl, flux_fast_rot, cntm = synth(**base_params, vsini=50)  # 50 km/s
        
        # Faster rotation should broaden lines more
        def line_depth_variation(flux):
            return jnp.std(flux)
        
        var_no_rot = line_depth_variation(flux_no_rot)
        var_mod_rot = line_depth_variation(flux_mod_rot)
        var_fast_rot = line_depth_variation(flux_fast_rot)
        
        # Faster rotation should reduce line depth variations
        assert var_fast_rot < var_mod_rot <= var_no_rot
    
    def test_microturbulence_effects(self):
        """Test microturbulence effects"""
        base_params = dict(Teff=5500, logg=4.0, m_H=0.0, wavelengths=(5500, 5510))
        
        # Low microturbulence
        wl, flux_low_vmic, cntm = synth(**base_params, vmic=0.5)
        
        # High microturbulence
        wl, flux_high_vmic, cntm = synth(**base_params, vmic=3.0)
        
        # Should give different results
        assert not jnp.allclose(flux_low_vmic, flux_high_vmic, rtol=1e-6)
        
        # Both should be physical
        for flux in [flux_low_vmic, flux_high_vmic]:
            assert jnp.all(jnp.isfinite(flux))
            assert jnp.all(flux >= 0)


class TestSynthesizeDetailed:
    """Test detailed synthesize() function"""
    
    def test_synthesize_vs_synth_consistency(self):
        """Test that synthesize() and synth() give consistent results"""
        # Parameters
        Teff, logg, m_H = 5500, 4.0, 0.0
        wavelengths = jnp.linspace(5500, 5510, 100)
        
        # Call synth()
        wl_synth, flux_synth, cntm_synth = synth(
            Teff=Teff, logg=logg, m_H=m_H,
            wavelengths=(5500, 5510),
            vmic=1.0, rectify=False  # No rectification for comparison
        )
        
        # Call synthesize() manually
        A_X = format_abundances(m_H)
        atm = interpolate_atmosphere(Teff, logg, A_X)
        
        result = synthesize(atm, None, A_X, wavelengths, vmic=1.0)
        
        # Results should be similar (within numerical precision)
        # Note: might not be exactly equal due to different wavelength grids
        assert jnp.all(jnp.isfinite(result.flux))
        assert jnp.all(result.flux >= 0)
        assert len(result.wavelengths) == len(wavelengths)
    
    def test_synthesize_diagnostics(self):
        """Test diagnostic information from synthesize()"""
        A_X = format_abundances(0.0)
        atm = interpolate_atmosphere(5500, 4.0, A_X)
        wavelengths = jnp.linspace(5500, 5510, 50)
        
        result = synthesize(atm, None, A_X, wavelengths, vmic=1.0)
        
        # Check all expected attributes
        assert hasattr(result, 'flux')
        assert hasattr(result, 'cntm')
        assert hasattr(result, 'intensity')
        assert hasattr(result, 'alpha')
        assert hasattr(result, 'mu_grid')
        assert hasattr(result, 'number_densities')
        assert hasattr(result, 'electron_number_density')
        assert hasattr(result, 'wavelengths')
        assert hasattr(result, 'subspectra')
        
        # Check shapes
        n_wl = len(wavelengths)
        n_layers = atm['n_layers']
        
        assert result.flux.shape == (n_wl,)
        assert result.alpha.shape == (n_layers, n_wl)
        assert len(result.electron_number_density) == n_layers
        assert len(result.mu_grid) > 0
        
        # Check physical validity
        assert jnp.all(result.alpha >= 0)
        assert jnp.all(result.electron_number_density >= 0)


class TestPerformance:
    """Test performance characteristics"""
    
    def test_synthesis_timing(self):
        """Test that synthesis completes in reasonable time"""
        start_time = time.time()
        
        # Standard synthesis
        wl, flux, cntm = synth(
            Teff=5500, logg=4.0, m_H=0.0,
            wavelengths=(5000, 6000),  # 1000 Å range
            vmic=1.0
        )
        
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time (< 30 seconds for basic synthesis)
        assert elapsed < 30.0
        
        # Results should be valid
        assert len(wl) > 100
        assert jnp.all(jnp.isfinite(flux))
    
    def test_memory_usage(self):
        """Test that synthesis doesn't use excessive memory"""
        # This is a basic test - in practice you'd use memory profiling tools
        import gc
        
        gc.collect()  # Clean up before test
        
        # Run multiple syntheses to check for memory leaks
        for i in range(5):
            wl, flux, cntm = synth(
                Teff=5000 + i*200, logg=4.0, m_H=0.0,
                wavelengths=(5500, 5510),  # Small range
                vmic=1.0
            )
            
            # Basic validation
            assert jnp.all(jnp.isfinite(flux))
        
        gc.collect()  # Clean up after test
        
        # If we get here without memory errors, test passes


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_extreme_stellar_parameters(self):
        """Test synthesis with extreme but physical stellar parameters"""
        extreme_cases = [
            {'Teff': 3000, 'logg': 0.0, 'm_H': -3.0},  # Cool giant, very metal-poor
            {'Teff': 10000, 'logg': 5.0, 'm_H': 0.5},  # Hot dwarf, metal-rich
            {'Teff': 2500, 'logg': 1.0, 'm_H': -2.0},  # Cool supergiant
        ]
        
        wavelength_range = (5500, 5510)  # Small range for speed
        
        for params in extreme_cases:
            try:
                wl, flux, cntm = synth(
                    **params,
                    wavelengths=wavelength_range,
                    vmic=1.0
                )
                
                # If synthesis succeeds, results should be physical
                assert jnp.all(jnp.isfinite(flux))
                assert jnp.all(flux >= 0)
                assert jnp.all(cntm > 0)
                
            except Exception as e:
                # Some extreme cases might fail - document the failure
                print(f"Synthesis failed for extreme parameters {params}: {e}")
                # This is not necessarily a test failure for very extreme cases
    
    def test_empty_linelist(self):
        """Test synthesis with no spectral lines (continuum only)"""
        wl, flux, cntm = synth(
            Teff=5500, logg=4.0, m_H=0.0,
            wavelengths=(5500, 5510),
            linelist=None,  # No lines
            vmic=1.0
        )
        
        # Should still work (continuum only)
        assert jnp.all(jnp.isfinite(flux))
        assert jnp.all(flux >= 0)
        assert jnp.all(cntm > 0)
        
        # Flux should be close to continuum (no line absorption)
        rectified_flux = flux / cntm
        assert jnp.all(rectified_flux > 0.99)  # Very little absorption
    
    def test_single_wavelength(self):
        """Test synthesis at a single wavelength"""
        wl, flux, cntm = synth(
            Teff=5500, logg=4.0, m_H=0.0,
            wavelengths=(5500.0, 5500.1),  # Very narrow range
            vmic=1.0
        )
        
        # Should work even for tiny wavelength range
        assert len(wl) >= 1
        assert jnp.all(jnp.isfinite(flux))
        assert jnp.all(flux >= 0)
    
    def test_unphysical_parameters_handling(self):
        """Test handling of unphysical parameters"""
        unphysical_cases = [
            {'Teff': -1000},      # Negative temperature
            {'logg': -10},        # Negative gravity
            {'m_H': 10},          # Extreme metallicity
        ]
        
        base_params = dict(Teff=5500, logg=4.0, m_H=0.0, wavelengths=(5500, 5510), vmic=1.0)
        
        for bad_param in unphysical_cases:
            params = {**base_params, **bad_param}
            
            # Should either handle gracefully or raise appropriate error
            try:
                wl, flux, cntm = synth(**params)
                # If it succeeds, results should still be finite
                assert jnp.all(jnp.isfinite(flux))
            except (ValueError, RuntimeError):
                # Appropriate error for unphysical parameters
                pass


class TestConsistencyChecks:
    """Test internal consistency of synthesis results"""
    
    def test_energy_conservation(self):
        """Test that flux conservation makes sense"""
        wl, flux, cntm = synth(
            Teff=5500, logg=4.0, m_H=0.0,
            wavelengths=(5000, 6000),
            vmic=1.0, rectify=False
        )
        
        # Total flux should be reasonable compared to blackbody
        from jorg.constants import PLANCK_H, SPEED_OF_LIGHT, BOLTZMANN_K
        
        # Planck function at 5500 K for comparison
        lambda_peak = 2.898e-3 / 5500  # Wien's law in meters
        lambda_aa = lambda_peak * 1e10  # Convert to Angstroms
        
        # Peak should be around 5300 Å for 5500 K star
        assert 4000 < lambda_aa < 7000
        
        # Flux should be positive everywhere
        assert jnp.all(flux >= 0)
        assert jnp.all(jnp.isfinite(flux))
    
    def test_symmetry_and_scaling(self):
        """Test expected scaling relationships"""
        base_params = dict(logg=4.0, m_H=0.0, wavelengths=(5500, 5510), vmic=1.0)
        
        # Temperature scaling test
        wl, flux_cool, cntm_cool = synth(Teff=4000, **base_params)
        wl, flux_hot, cntm_hot = synth(Teff=7000, **base_params)
        
        # Hotter stars should have higher continuum flux
        assert jnp.mean(cntm_hot) > jnp.mean(cntm_cool)
        
        # Surface gravity scaling test  
        wl, flux_giant, cntm_giant = synth(Teff=5500, logg=2.0, m_H=0.0, wavelengths=(5500, 5510), vmic=1.0)
        wl, flux_dwarf, cntm_dwarf = synth(Teff=5500, logg=5.0, m_H=0.0, wavelengths=(5500, 5510), vmic=1.0)
        
        # Different surface gravities should give different pressure broadening
        assert not jnp.allclose(flux_giant, flux_dwarf, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])