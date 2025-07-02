"""
Unit tests for the Jorg synthesis module

Tests the main synth() and synthesize() functions and their supporting utilities,
following the same testing patterns as Korg.jl.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from jorg.synthesis import (
    synth, synthesize, SynthesisResult,
    format_abundances, interpolate_atmosphere,
    apply_LSF, apply_rotation, radiative_transfer
)


class TestFormatAbundances:
    """Test format_abundances function following Korg.jl's format_A_X behavior"""
    
    def test_solar_abundances_default(self):
        """Test default solar abundances"""
        A_X = format_abundances(m_H=0.0, alpha_H=0.0)
        
        # Should return 92-element array
        assert len(A_X) == 92
        
        # Check some known solar values
        assert A_X[0] == 12.00  # H
        assert A_X[25] == 7.50  # Fe
        assert A_X[5] == 8.43   # C
        
    def test_metallicity_scaling(self):
        """Test metallicity scaling behavior"""
        m_H = -1.0
        A_X = format_abundances(m_H=m_H)
        
        # All elements should be scaled by m_H
        assert A_X[25] == 7.50 + m_H  # Fe
        assert A_X[5] == 8.43 + m_H   # C
        
    def test_alpha_enhancement(self):
        """Test alpha element enhancement"""
        m_H = -1.0
        alpha_H = -0.5
        A_X = format_abundances(m_H=m_H, alpha_H=alpha_H)
        
        # Alpha elements should get additional enhancement
        assert A_X[7] == 8.69 + alpha_H   # O (alpha element)
        assert A_X[11] == 7.60 + alpha_H  # Mg (alpha element)
        
        # Non-alpha elements should only get m_H
        assert A_X[25] == 7.50 + m_H      # Fe (not alpha)
        
    def test_individual_abundances(self):
        """Test individual element abundance overrides"""
        A_X = format_abundances(m_H=0.0, Fe=-0.5, C=0.2)
        
        # Individual elements should override scaling
        assert A_X[25] == 7.50 - 0.5  # Fe
        assert A_X[5] == 8.43 + 0.2   # C
        
    def test_alpha_h_defaults_to_m_h(self):
        """Test that alpha_H defaults to m_H when not specified"""
        m_H = -0.5
        A_X1 = format_abundances(m_H=m_H)
        A_X2 = format_abundances(m_H=m_H, alpha_H=m_H)
        
        np.testing.assert_array_equal(A_X1, A_X2)


class TestInterpolateAtmosphere:
    """Test interpolate_atmosphere function"""
    
    def test_atmosphere_structure(self):
        """Test atmosphere structure creation"""
        Teff = 5500
        logg = 4.0
        A_X = format_abundances(0.0)
        
        atm = interpolate_atmosphere(Teff, logg, A_X)
        
        # Check required keys
        required_keys = ['tau_5000', 'temperature', 'pressure', 'density', 
                        'electron_density', 'height', 'n_layers']
        for key in required_keys:
            assert key in atm
            
        # Check dimensions
        n_layers = atm['n_layers']
        assert n_layers == 72  # Standard MARCS
        
        for key in ['tau_5000', 'temperature', 'pressure', 'density', 
                   'electron_density', 'height']:
            assert len(atm[key]) == n_layers
            
    def test_temperature_scaling(self):
        """Test temperature scaling with Teff"""
        A_X = format_abundances(0.0)
        
        atm1 = interpolate_atmosphere(5000, 4.0, A_X)
        atm2 = interpolate_atmosphere(6000, 4.0, A_X)
        
        # Higher Teff should give higher temperatures
        assert jnp.all(atm2['temperature'] > atm1['temperature'])
        
    def test_gravity_scaling(self):
        """Test pressure scaling with surface gravity"""
        A_X = format_abundances(0.0)
        
        atm1 = interpolate_atmosphere(5500, 3.5, A_X)
        atm2 = interpolate_atmosphere(5500, 4.5, A_X)
        
        # Higher logg should give higher pressures
        assert jnp.all(atm2['pressure'] > atm1['pressure'])


class TestSynthesisResult:
    """Test SynthesisResult dataclass"""
    
    def test_synthesis_result_creation(self):
        """Test SynthesisResult initialization"""
        n_wl = 100
        n_layers = 72
        n_mu = 20
        
        wavelengths = jnp.linspace(5000, 6000, n_wl)
        flux = jnp.ones(n_wl)
        continuum = jnp.ones(n_wl)
        intensity = jnp.ones((n_mu, n_layers, n_wl))
        alpha = jnp.ones((n_layers, n_wl))
        mu_grid = [(0.5, 0.1), (1.0, 0.1)]
        number_densities = {'H_I': jnp.ones(n_layers)}
        electron_density = jnp.ones(n_layers)
        subspectra = [range(n_wl)]
        
        result = SynthesisResult(
            flux=flux,
            cntm=continuum,
            intensity=intensity,
            alpha=alpha,
            mu_grid=mu_grid,
            number_densities=number_densities,
            electron_number_density=electron_density,
            wavelengths=wavelengths,
            subspectra=subspectra
        )
        
        # Check all attributes exist
        assert jnp.array_equal(result.flux, flux)
        assert jnp.array_equal(result.cntm, continuum)
        assert jnp.array_equal(result.wavelengths, wavelengths)
        assert result.mu_grid == mu_grid


class TestRadiativeTransfer:
    """Test radiative transfer function"""
    
    def test_radiative_transfer_shapes(self):
        """Test output shapes from radiative transfer"""
        n_layers = 72
        n_wavelengths = 100
        n_mu = 10
        
        alpha = jnp.ones((n_layers, n_wavelengths)) * 0.1
        source_function = jnp.ones((n_layers, n_wavelengths))
        height = jnp.linspace(0, 1e8, n_layers)  
        
        flux, intensity, mu_grid = radiative_transfer(
            alpha, source_function, height, n_mu
        )
        
        # Check shapes
        assert flux.shape == (n_wavelengths,)
        assert intensity.shape == (n_mu, n_layers, n_wavelengths)
        assert len(mu_grid) == n_mu
        
    def test_radiative_transfer_conservation(self):
        """Test basic conservation properties"""
        n_layers = 10
        n_wavelengths = 5
        
        # Optically thin case
        alpha = jnp.ones((n_layers, n_wavelengths)) * 1e-6
        source_function = jnp.ones((n_layers, n_wavelengths))
        height = jnp.linspace(0, 1e6, n_layers)
        
        flux, _, _ = radiative_transfer(alpha, source_function, height, 5)
        
        # Should be close to source function in optically thin limit
        assert jnp.all(flux > 0)
        assert jnp.all(jnp.isfinite(flux))


class TestApplyLSF:
    """Test instrumental line spread function"""
    
    def test_lsf_smoothing(self):
        """Test that LSF smooths the spectrum"""
        wavelengths = jnp.linspace(5000, 6000, 1000)
        
        # Create sharp feature
        flux = jnp.ones_like(wavelengths)
        flux = flux.at[500].set(0.5)  # Sharp absorption line
        
        # Apply LSF
        R = 10000
        smoothed_flux = apply_LSF(flux, wavelengths, R)
        
        # Line should be broadened
        assert smoothed_flux[500] > flux[500]  # Less deep
        assert jnp.sum(jnp.abs(jnp.diff(smoothed_flux))) < jnp.sum(jnp.abs(jnp.diff(flux)))
        
    def test_lsf_callable_resolution(self):
        """Test LSF with wavelength-dependent resolution"""
        wavelengths = jnp.linspace(5000, 6000, 100)
        flux = jnp.ones_like(wavelengths)
        
        # Variable resolution function
        R = lambda wl: 50000 * (wl / 5500)
        
        smoothed_flux = apply_LSF(flux, wavelengths, R)
        assert jnp.all(jnp.isfinite(smoothed_flux))


class TestApplyRotation:
    """Test rotational broadening"""
    
    def test_rotation_broadening(self):
        """Test rotational broadening effect"""
        wavelengths = jnp.linspace(5000, 6000, 1000)
        
        # Create sharp line
        flux = jnp.ones_like(wavelengths)
        flux = flux.at[500].set(0.5)
        
        # Apply rotation
        vsini = 10  # km/s
        broadened_flux = apply_rotation(flux, wavelengths, vsini)
        
        # Should broaden the line
        assert broadened_flux[500] > flux[500]
        
    def test_zero_rotation(self):
        """Test zero rotation leaves spectrum unchanged"""
        wavelengths = jnp.linspace(5000, 6000, 100)
        flux = jnp.sin(wavelengths)  # Some test pattern
        
        # No rotation
        result = apply_rotation(flux, wavelengths, 0.0)
        
        # Should be nearly unchanged (within numerical precision)
        np.testing.assert_allclose(result, flux, rtol=1e-10)


class TestSynth:
    """Test main synth() function"""
    
    @patch('jorg.synthesis.total_continuum_absorption')
    @patch('jorg.synthesis.total_line_absorption')
    def test_synth_basic_call(self, mock_line_abs, mock_cntm_abs):
        """Test basic synth() function call"""
        # Mock the absorption functions
        mock_cntm_abs.return_value = jnp.ones(1000) * 0.1
        mock_line_abs.return_value = jnp.ones((72, 1000)) * 0.01
        
        wavelengths, flux, continuum = synth(
            Teff=5500, logg=4.0, m_H=-0.5
        )
        
        assert len(wavelengths) == 1000  # Default wavelength grid
        assert len(flux) == 1000
        assert len(continuum) == 1000
        assert jnp.all(jnp.isfinite(flux))
        
    def test_synth_wavelength_specification(self):
        """Test different wavelength specifications"""
        # Tuple specification
        wl1, flux1, cntm1 = synth(wavelengths=(5000, 5100))
        assert jnp.min(wl1) >= 5000
        assert jnp.max(wl1) <= 5100
        
        # Multiple ranges
        wl2, flux2, cntm2 = synth(wavelengths=[(5000, 5050), (5950, 6000)])
        assert len(wl2) == 1000  # 500 points per range
        
    def test_synth_abundance_passing(self):
        """Test that abundance parameters are passed correctly"""
        with patch('jorg.synthesis.format_abundances') as mock_format:
            mock_format.return_value = jnp.ones(92)
            
            synth(Teff=5500, m_H=-0.5, alpha_H=-0.2, Fe=-0.8, C=0.1)
            
            # Check that format_abundances was called with correct parameters
            mock_format.assert_called_once()
            args, kwargs = mock_format.call_args
            assert args[0] == -0.5  # m_H
            assert args[1] == -0.2  # alpha_H
            assert 'Fe' in kwargs
            assert 'C' in kwargs
            
    def test_synth_rectification(self):
        """Test continuum rectification option"""
        with patch('jorg.synthesis.synthesize') as mock_synthesize:
            # Mock synthesize result
            mock_result = Mock()
            mock_result.flux = jnp.ones(100) * 2.0
            mock_result.cntm = jnp.ones(100) * 4.0
            mock_result.wavelengths = jnp.linspace(5000, 6000, 100)
            mock_synthesize.return_value = mock_result
            
            # Test rectified (default)
            wl, flux_rect, cntm = synth(rectify=True)
            expected_rect = mock_result.flux / mock_result.cntm
            np.testing.assert_array_equal(flux_rect, expected_rect)
            
            # Test non-rectified
            wl, flux_abs, cntm = synth(rectify=False)
            np.testing.assert_array_equal(flux_abs, mock_result.flux)


class TestSynthesize:
    """Test detailed synthesize() function"""
    
    @patch('jorg.synthesis.total_continuum_absorption')
    @patch('jorg.synthesis.total_line_absorption')
    def test_synthesize_basic_call(self, mock_line_abs, mock_cntm_abs):
        """Test basic synthesize() function call"""
        # Set up mocks
        mock_cntm_abs.return_value = jnp.ones(100) * 0.1
        mock_line_abs.return_value = jnp.ones((72, 100)) * 0.01
        
        # Create test inputs
        atm = interpolate_atmosphere(5500, 4.0, format_abundances(0.0))
        A_X = format_abundances(0.0)
        wavelengths = jnp.linspace(5000, 6000, 100)
        
        result = synthesize(atm, None, A_X, wavelengths)
        
        # Check result type and attributes
        assert isinstance(result, SynthesisResult)
        assert len(result.flux) == 100
        assert len(result.wavelengths) == 100
        
    def test_synthesize_with_linelist(self):
        """Test synthesize with line list"""
        atm = interpolate_atmosphere(5500, 4.0, format_abundances(0.0))
        A_X = format_abundances(0.0)
        wavelengths = jnp.linspace(5000, 6000, 100)
        
        # Mock linelist
        linelist = [{'wavelength': 5500, 'strength': 1.0}]
        
        with patch('jorg.synthesis.total_line_absorption') as mock_line:
            mock_line.return_value = jnp.ones((72, 100)) * 0.1
            
            result = synthesize(atm, linelist, A_X, wavelengths)
            
            # Should call line absorption
            mock_line.assert_called_once()
            
    def test_synthesize_parameter_passing(self):
        """Test that all parameters are handled correctly"""
        atm = interpolate_atmosphere(5500, 4.0, format_abundances(0.0))
        A_X = format_abundances(0.0)
        wavelengths = jnp.linspace(5000, 6000, 50)
        
        # Test with various parameters
        result = synthesize(
            atm, None, A_X, wavelengths,
            vmic=2.0,
            line_cutoff_threshold=1e-3,
            mu_values=15,
            return_cntm=False,
            verbose=True
        )
        
        assert result.cntm is None  # return_cntm=False
        assert len(result.mu_grid) == 15  # mu_values=15


class TestIntegration:
    """Integration tests for the complete synthesis workflow"""
    
    def test_end_to_end_synthesis(self):
        """Test complete synthesis workflow"""
        # This test verifies the entire pipeline works together
        wavelengths, flux, continuum = synth(
            Teff=5800,
            logg=4.4,
            m_H=0.0,
            wavelengths=(5500, 5600),
            vmic=1.0
        )
        
        # Basic sanity checks
        assert len(wavelengths) > 0
        assert len(flux) == len(wavelengths)
        assert len(continuum) == len(wavelengths)
        assert jnp.all(jnp.isfinite(flux))
        assert jnp.all(jnp.isfinite(continuum))
        assert jnp.all(flux >= 0)
        assert jnp.all(continuum > 0)
        
    def test_parameter_variations(self):
        """Test that varying parameters produces expected changes"""
        base_params = dict(Teff=5500, logg=4.0, m_H=0.0, wavelengths=(5500, 5510))
        
        # Get baseline
        wl, flux_base, cntm_base = synth(**base_params)
        
        # Test temperature variation
        wl, flux_hot, cntm_hot = synth(**{**base_params, 'Teff': 6000})
        
        # Hot stars should have different continuum
        assert not jnp.allclose(flux_hot, flux_base)
        
        # Test metallicity variation  
        wl, flux_metal, cntm_metal = synth(**{**base_params, 'm_H': -1.0})
        
        # Different metallicity should affect spectrum
        assert not jnp.allclose(flux_metal, flux_base)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])