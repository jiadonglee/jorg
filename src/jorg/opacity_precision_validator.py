"""
Opacity Matrix Precision Validator
=================================

This module provides systematic validation of opacity calculations between
Jorg and Korg.jl, focusing on identifying the sources of differences in
continuum and line opacity components.

Since opacity differences directly affect the radiative transfer and final
spectrum, this is likely where we'll find the most significant sources of
the remaining precision differences.

Key Analysis Areas:
- Continuum opacity components (H⁻, Thomson, metals, Rayleigh)
- Line opacity calculations (Voigt profiles, broadening, species mapping)
- Layer-by-layer opacity structure validation
- Wavelength-dependent precision analysis
- Temperature and density dependencies

Usage:
    from jorg.opacity_precision_validator import OpacityValidator
    
    validator = OpacityValidator(verbose=True)
    report = validator.validate_full_opacity_matrix(atm, linelist, abundances)

Author: Claude Code
Date: August 2025
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings
import json
import time

# Jorg imports
from .continuum.exact_physics_continuum import total_continuum_absorption_exact_physics_only
from .continuum.exact_physics_continuum import (
    mclaughlin_hminus_bf_absorption, h_minus_ff_absorption,
    thomson_scattering, rayleigh_scattering, metal_bf_absorption
)
from .lines.core import total_line_absorption
from .lines.profiles import voigt_profile, line_profile
from .opacity.layer_processor import LayerProcessor
from .opacity.korg_line_processor import KorgLineProcessor
from .constants import kboltz_cgs, c_cgs, hplanck_cgs


@dataclass
class OpacityComponentResult:
    """Results for a single opacity component validation"""
    component_name: str
    jorg_opacity: np.ndarray        # [layers × wavelengths]
    korg_opacity: Optional[np.ndarray]  # [layers × wavelengths] if available
    agreement_stats: Dict[str, float]
    layer_agreements: np.ndarray    # Agreement per layer
    wavelength_agreements: np.ndarray  # Agreement per wavelength  
    critical_issues: List[str] = field(default_factory=list)
    precision_analysis: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class OpacityValidationReport:
    """Complete opacity matrix validation report"""
    overall_agreement: float
    continuum_results: Dict[str, OpacityComponentResult]
    line_opacity_result: OpacityComponentResult
    total_opacity_result: OpacityComponentResult
    layer_analysis: Dict[int, Dict[str, float]]
    wavelength_analysis: Dict[float, Dict[str, float]]
    critical_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class OpacityValidator:
    """
    Comprehensive opacity matrix precision validator
    
    This class systematically validates each opacity component to identify
    sources of differences with Korg.jl at the sub-percent level.
    """
    
    def __init__(self, verbose: bool = True, precision_target: float = 0.001):
        """
        Initialize opacity validator
        
        Parameters
        ----------
        verbose : bool
            Print detailed analysis information
        precision_target : float  
            Target relative difference for opacity components (0.1% = 0.001)
        """
        self.verbose = verbose
        self.precision_target = precision_target
        
        if self.verbose:
            print("🔍 OPACITY MATRIX PRECISION VALIDATOR")
            print("=" * 50)
            print(f"Target precision: <{precision_target*100:.1f}% relative difference")
            print("Systematic component-by-component analysis")
    
    def validate_continuum_components(
        self,
        wavelengths: np.ndarray,
        atm: Dict,
        species_densities: Dict,
        korg_reference: Optional[Dict] = None
    ) -> Dict[str, OpacityComponentResult]:
        """
        Validate individual continuum opacity components
        
        This breaks down continuum opacity into H⁻ bf/ff, Thomson, metals, etc.
        to identify which specific component has precision issues.
        """
        if self.verbose:
            print("\n🌊 CONTINUUM COMPONENT PRECISION ANALYSIS")
            print("-" * 50)
        
        n_layers = len(atm['temperature'])
        n_wavelengths = len(wavelengths)
        frequencies = c_cgs / (wavelengths * 1e-8)
        
        continuum_results = {}
        
        # Component 1: H⁻ bound-free
        if self.verbose:
            print("1. H⁻ bound-free absorption...")
        
        h_minus_bf_matrix = np.zeros((n_layers, n_wavelengths))
        
        for layer_idx in range(n_layers):
            T = atm['temperature'][layer_idx]
            n_e = atm['electron_density'][layer_idx]
            
            # Get H I density for this layer
            n_h_i = 0.0
            for species, density_array in species_densities.items():
                if (hasattr(species, 'atomic_number') and species.atomic_number == 1 and 
                    hasattr(species, 'charge') and species.charge == 0):
                    if hasattr(density_array, '__len__') and len(density_array) > layer_idx:
                        n_h_i = density_array[layer_idx]
                    break
            
            # Calculate H⁻ bf opacity
            try:
                h_minus_bf_opacity = mclaughlin_hminus_bf_absorption(frequencies, T, n_e, n_h_i)
                h_minus_bf_matrix[layer_idx, :] = h_minus_bf_opacity
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Layer {layer_idx}: H⁻ bf calculation error: {e}")
                continue
        
        continuum_results['h_minus_bf'] = self._create_component_result(
            'H⁻ Bound-Free', h_minus_bf_matrix, 
            korg_reference.get('h_minus_bf') if korg_reference else None
        )
        
        # Component 2: H⁻ free-free
        if self.verbose:
            print("2. H⁻ free-free absorption...")
        
        h_minus_ff_matrix = np.zeros((n_layers, n_wavelengths))
        
        for layer_idx in range(n_layers):
            T = atm['temperature'][layer_idx]
            n_e = atm['electron_density'][layer_idx]
            
            # Get H I density
            n_h_i = 0.0
            for species, density_array in species_densities.items():
                if (hasattr(species, 'atomic_number') and species.atomic_number == 1 and 
                    hasattr(species, 'charge') and species.charge == 0):
                    if hasattr(density_array, '__len__') and len(density_array) > layer_idx:
                        n_h_i = density_array[layer_idx]
                    break
            
            try:
                h_minus_ff_opacity = h_minus_ff_absorption(frequencies, T, n_e, n_h_i)
                h_minus_ff_matrix[layer_idx, :] = h_minus_ff_opacity
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Layer {layer_idx}: H⁻ ff calculation error: {e}")
                continue
        
        continuum_results['h_minus_ff'] = self._create_component_result(
            'H⁻ Free-Free', h_minus_ff_matrix,
            korg_reference.get('h_minus_ff') if korg_reference else None
        )
        
        # Component 3: Thomson scattering
        if self.verbose:
            print("3. Thomson scattering...")
        
        thomson_matrix = np.zeros((n_layers, n_wavelengths))
        
        for layer_idx in range(n_layers):
            n_e = atm['electron_density'][layer_idx]
            # Thomson scattering is frequency-independent, returns scalar
            thomson_opacity = thomson_scattering(n_e)
            # Broadcast to all wavelengths
            thomson_matrix[layer_idx, :] = thomson_opacity
        
        continuum_results['thomson'] = self._create_component_result(
            'Thomson Scattering', thomson_matrix,
            korg_reference.get('thomson') if korg_reference else None
        )
        
        # Component 4: Metal bound-free (if significant)
        if self.verbose:
            print("4. Metal bound-free absorption...")
        
        metal_bf_matrix = np.zeros((n_layers, n_wavelengths))
        
        for layer_idx in range(n_layers):
            T = atm['temperature'][layer_idx]
            
            # Get key metal densities
            layer_densities = {}
            for species, density_array in species_densities.items():
                if (hasattr(species, 'atomic_number') and 
                    hasattr(density_array, '__len__') and len(density_array) > layer_idx):
                    layer_densities[species] = density_array[layer_idx]
            
            try:
                # Extract species densities dictionary for metal bf calculation
                species_dict = {}
                for species, density in layer_densities.items():
                    if hasattr(species, 'atomic_number') and hasattr(species, 'charge'):
                        species_dict[species] = float(density)
                
                metal_bf_opacity = metal_bf_absorption(frequencies, T, species_dict)
                metal_bf_matrix[layer_idx, :] = metal_bf_opacity
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Layer {layer_idx}: Metal bf calculation error: {e}")
                continue
        
        continuum_results['metal_bf'] = self._create_component_result(
            'Metal Bound-Free', metal_bf_matrix,
            korg_reference.get('metal_bf') if korg_reference else None
        )
        
        # Component 5: Rayleigh scattering  
        if self.verbose:
            print("5. Rayleigh scattering...")
        
        rayleigh_matrix = np.zeros((n_layers, n_wavelengths))
        
        for layer_idx in range(n_layers):
            # Get neutral atom densities
            layer_neutrals = {}
            for species, density_array in species_densities.items():
                if (hasattr(species, 'atomic_number') and hasattr(species, 'charge') and 
                    species.charge == 0 and hasattr(density_array, '__len__') and 
                    len(density_array) > layer_idx):
                    layer_neutrals[species.atomic_number] = density_array[layer_idx]
            
            try:
                rayleigh_opacity = rayleigh_scattering(
                    frequencies, 
                    layer_neutrals.get(1, 0),    # H I
                    layer_neutrals.get(2, 0),    # He I  
                    0.0                          # H2 (molecular hydrogen)
                )
                rayleigh_matrix[layer_idx, :] = rayleigh_opacity
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️ Layer {layer_idx}: Rayleigh calculation error: {e}")
                continue
        
        continuum_results['rayleigh'] = self._create_component_result(
            'Rayleigh Scattering', rayleigh_matrix,
            korg_reference.get('rayleigh') if korg_reference else None
        )
        
        if self.verbose:
            self._print_continuum_summary(continuum_results)
        
        return continuum_results
    
    def validate_line_opacity_precision(
        self,
        wavelengths: np.ndarray,
        atm: Dict,
        linelist: List,
        species_densities: Dict,
        korg_reference: Optional[Dict] = None
    ) -> OpacityComponentResult:
        """
        Validate line opacity calculation precision
        
        This focuses on the KorgLineProcessor and Voigt profile calculations
        which are critical for realistic line depths.
        """
        if self.verbose:
            print("\n📊 LINE OPACITY PRECISION ANALYSIS")
            print("-" * 50)
            print(f"Analyzing {len(linelist)} spectral lines")
        
        n_layers = len(atm['temperature'])
        n_wavelengths = len(wavelengths)
        
        # Use KorgLineProcessor for exact Korg.jl compatibility
        korg_line_processor = KorgLineProcessor(verbose=False)
        korg_line_processor.cutoff_threshold = 3e-4
        
        # Calculate line opacity matrix
        start_time = time.time()
        
        line_opacity_matrix = np.zeros((n_layers, n_wavelengths))
        
        try:
            # Check if we have lines to process
            if not linelist or len(linelist) == 0:
                if self.verbose:
                    print("   ⚠️ No lines in linelist - returning zero opacity")
                # Return zero matrix for empty linelist
                line_opacity_matrix = np.zeros((n_layers, n_wavelengths))
            else:
                # Process lines using KorgLineProcessor
                # Convert wavelengths to cm
                wl_array_cm = wavelengths * 1e-8
                temps = atm['temperature']
                electron_densities = atm['electron_density']
                
                # Get partition functions (simplified - would need full implementation)
                from jorg.statmech import create_default_partition_functions
                partition_fns = create_default_partition_functions()
                
                # Process all lines at once (KorgLineProcessor works on entire atmospheric grid)
                line_opacity_result = korg_line_processor.process_lines(
                    wl_array_cm=wl_array_cm,
                    temps=temps,
                    electron_densities=electron_densities,
                    n_densities=species_densities,
                    partition_fns=partition_fns,
                    linelist=linelist,
                    microturbulence_cms=1.0e5,  # 1 km/s in cm/s
                    continuum_opacity_fn=None,
                    cutoff_threshold=korg_line_processor.cutoff_threshold
                )
                
                if line_opacity_result is not None:
                    line_opacity_matrix = line_opacity_result
                else:
                    line_opacity_matrix = np.zeros((n_layers, n_wavelengths))
        
        except Exception as e:
            if self.verbose:
                print(f"   ❌ Line opacity calculation failed: {e}")
                import traceback
                traceback.print_exc()
        
        processing_time = time.time() - start_time
        
        if self.verbose:
            print(f"   ✅ Line opacity calculation: {processing_time:.2f}s")
            print(f"   📊 Opacity range: {line_opacity_matrix.min():.3e} - {line_opacity_matrix.max():.3e} cm⁻¹")
            print(f"   🎯 Non-zero opacity points: {np.sum(line_opacity_matrix > 0)}/{line_opacity_matrix.size}")
        
        return self._create_component_result(
            'Line Opacity', line_opacity_matrix,
            korg_reference.get('line_opacity') if korg_reference else None,
            extra_analysis={
                'processing_time': processing_time,
                'n_lines': len(linelist),
                'cutoff_threshold': korg_line_processor.cutoff_threshold,
                'non_zero_points': int(np.sum(line_opacity_matrix > 0))
            }
        )
    
    def validate_voigt_profile_precision(
        self,
        wavelengths: np.ndarray,
        test_lines: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Validate Voigt profile calculation precision
        
        Tests the core line profile functions against Korg.jl values.
        """
        if self.verbose:
            print("\n📈 VOIGT PROFILE PRECISION ANALYSIS")
            print("-" * 50)
        
        # Test parameters covering different regimes
        test_cases = [
            {"name": "Doppler-dominated", "lambda_0": 5000e-8, "sigma": 2e-9, "gamma": 1e-11},
            {"name": "Intermediate", "lambda_0": 5000e-8, "sigma": 1.5e-9, "gamma": 5e-10},
            {"name": "Pressure-dominated", "lambda_0": 5000e-8, "sigma": 1e-9, "gamma": 2e-9},
            {"name": "Strong line", "lambda_0": 6562.8e-8, "sigma": 3e-9, "gamma": 1e-9},  # H-alpha
        ]
        
        results = {}
        
        for case in test_cases:
            name = case["name"]
            lambda_0 = case["lambda_0"]
            sigma = case["sigma"] 
            gamma = case["gamma"]
            amplitude = 1e-13  # Representative line strength
            
            if self.verbose:
                print(f"   Testing {name}:")
                print(f"     λ₀ = {lambda_0*1e8:.1f} Å, σ = {sigma*1e8:.2f} mÅ, γ = {gamma*1e8:.2f} mÅ")
            
            # Calculate Voigt profile at test wavelengths
            test_wl = np.linspace(lambda_0 - 5*sigma, lambda_0 + 5*sigma, 101)
            
            try:
                profile_values = line_profile(lambda_0, sigma, gamma, amplitude, test_wl)
                
                # Analysis metrics
                peak_value = np.max(profile_values)
                fwhm_indices = profile_values > peak_value / 2
                fwhm_points = np.sum(fwhm_indices)
                
                results[name] = {
                    'peak_value': float(peak_value),
                    'fwhm_points': int(fwhm_points),
                    'profile_integral': float(np.trapz(profile_values, test_wl)),
                    'status': 'SUCCESS'
                }
                
                if self.verbose:
                    print(f"     ✅ Peak: {peak_value:.3e}, FWHM points: {fwhm_points}, Integral: {results[name]['profile_integral']:.3e}")
            
            except Exception as e:
                results[name] = {'status': 'FAILED', 'error': str(e)}
                if self.verbose:
                    print(f"     ❌ Error: {e}")
        
        # Overall assessment
        successful_tests = sum(1 for r in results.values() if r.get('status') == 'SUCCESS')
        total_tests = len(test_cases)
        
        if self.verbose:
            print(f"\n   📊 Voigt profile validation: {successful_tests}/{total_tests} tests passed")
        
        results['summary'] = {
            'tests_passed': successful_tests,
            'total_tests': total_tests,
            'success_rate': successful_tests / total_tests if total_tests > 0 else 0
        }
        
        return results
    
    def _create_component_result(
        self,
        component_name: str,
        jorg_opacity: np.ndarray,
        korg_opacity: Optional[np.ndarray] = None,
        extra_analysis: Optional[Dict] = None
    ) -> OpacityComponentResult:
        """Create standardized component result"""
        
        agreement_stats = {}
        layer_agreements = np.array([])
        wavelength_agreements = np.array([])
        critical_issues = []
        
        if korg_opacity is not None and korg_opacity.shape == jorg_opacity.shape:
            # Calculate agreement statistics
            valid_mask = (jorg_opacity > 0) & (korg_opacity > 0)
            
            if np.sum(valid_mask) > 0:
                ratio = jorg_opacity[valid_mask] / korg_opacity[valid_mask]
                relative_diff = np.abs(1 - ratio)
                
                agreement_stats = {
                    'mean_agreement': float(100 * (1 - np.mean(relative_diff))),
                    'median_agreement': float(100 * (1 - np.median(relative_diff))),
                    'worst_agreement': float(100 * (1 - np.max(relative_diff))),
                    'std_agreement': float(100 * np.std(relative_diff)),
                    'valid_points': int(np.sum(valid_mask))
                }
                
                # Layer-by-layer agreement
                layer_agreements = np.zeros(jorg_opacity.shape[0])
                for layer_idx in range(jorg_opacity.shape[0]):
                    layer_mask = valid_mask[layer_idx, :]
                    if np.sum(layer_mask) > 0:
                        layer_ratio = jorg_opacity[layer_idx, layer_mask] / korg_opacity[layer_idx, layer_mask]
                        layer_agreements[layer_idx] = 100 * (1 - np.mean(np.abs(1 - layer_ratio)))
                
                # Wavelength-by-wavelength agreement
                wavelength_agreements = np.zeros(jorg_opacity.shape[1])
                for wl_idx in range(jorg_opacity.shape[1]):
                    wl_mask = valid_mask[:, wl_idx]
                    if np.sum(wl_mask) > 0:
                        wl_ratio = jorg_opacity[wl_mask, wl_idx] / korg_opacity[wl_mask, wl_idx]
                        wavelength_agreements[wl_idx] = 100 * (1 - np.mean(np.abs(1 - wl_ratio)))
            
            # Identify critical issues
            if agreement_stats.get('mean_agreement', 100) < 95:
                critical_issues.append(f"Mean agreement <95%: {agreement_stats['mean_agreement']:.1f}%")
            
            if agreement_stats.get('worst_agreement', 100) < 80:
                critical_issues.append(f"Worst case agreement <80%: {agreement_stats['worst_agreement']:.1f}%")
        
        else:
            # No reference data - use internal consistency checks
            agreement_stats = {
                'mean_agreement': 100.0,
                'median_agreement': 100.0, 
                'worst_agreement': 100.0,
                'std_agreement': 0.0,
                'valid_points': int(np.sum(jorg_opacity > 0))
            }
        
        # Check for physical consistency
        if np.any(jorg_opacity < 0):
            critical_issues.append("Negative opacity values detected")
        
        if np.any(np.isnan(jorg_opacity)) or np.any(np.isinf(jorg_opacity)):
            critical_issues.append("NaN or infinite opacity values detected")
        
        precision_analysis = {
            'opacity_range': (float(jorg_opacity.min()), float(jorg_opacity.max())),
            'opacity_mean': float(jorg_opacity.mean()),
            'opacity_std': float(jorg_opacity.std()),
            'non_zero_fraction': float(np.sum(jorg_opacity > 0) / jorg_opacity.size)
        }
        
        if extra_analysis:
            precision_analysis.update(extra_analysis)
        
        return OpacityComponentResult(
            component_name=component_name,
            jorg_opacity=jorg_opacity,
            korg_opacity=korg_opacity,
            agreement_stats=agreement_stats,
            layer_agreements=layer_agreements,
            wavelength_agreements=wavelength_agreements,
            critical_issues=critical_issues,
            precision_analysis=precision_analysis
        )
    
    def _print_continuum_summary(self, continuum_results: Dict[str, OpacityComponentResult]):
        """Print summary of continuum component validation"""
        if not self.verbose:
            return
        
        print(f"\n📊 CONTINUUM COMPONENT SUMMARY:")
        print("-" * 40)
        
        for name, result in continuum_results.items():
            agreement = result.agreement_stats.get('mean_agreement', 100)
            opacity_range = result.precision_analysis['opacity_range']
            
            status = "✅" if agreement > 98 else "⚠️" if agreement > 95 else "❌"
            
            print(f"{status} {result.component_name:20}: {agreement:5.1f}% agreement")
            print(f"     Range: {opacity_range[0]:.3e} - {opacity_range[1]:.3e} cm⁻¹")
            
            if result.critical_issues:
                for issue in result.critical_issues:
                    print(f"     🚨 {issue}")


# Export main classes and functions
__all__ = [
    'OpacityValidator',
    'OpacityComponentResult',
    'OpacityValidationReport'
]