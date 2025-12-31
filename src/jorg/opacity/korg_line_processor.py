"""
Korg.jl-Compatible Line Opacity Processor

This module provides a direct implementation of Korg.jl's line_absorption.jl algorithm
to fix fundamental architectural issues in the current line opacity calculation.

ARCHITECTURE: Direct translation of Korg.jl's proven approach:
- Matrix-based calculations [layers × wavelengths]
- Vectorized operations across atmospheric layers
- Direct number density handling from chemical equilibrium
- Built-in line windowing and Voigt profile calculation

REFERENCE: /Users/jdli/Project/Korg.jl/src/line_absorption.jl
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gamma as jax_gamma
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..constants import (
    kboltz_cgs, c_cgs, hplanck_cgs, PI, 
    electron_charge_cgs as ELECTRON_CHARGE, 
    electron_mass_cgs as ELECTRON_MASS,
    kboltz_eV, hplanck_eV, amu_cgs
)
from ..statmech.species import Species


@dataclass
class KorgLineResult:
    """Result from Korg line processing"""
    alpha_matrix: np.ndarray  # [layers × wavelengths] opacity matrix
    lines_processed: int
    lines_windowed: int
    total_amplitude: float


@jax.jit
def _harris_series_vectorized_jax(v: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Vectorized Harris series coefficients for JAX."""
    v2 = v * v
    H0 = jnp.exp(-v2)

    H1_case1 = -1.12470432 + (-0.15516677 + (3.288675912 + (-2.34357915 + 0.42139162 * v) * v) * v) * v
    H1_case2 = -4.48480194 + (9.39456063 + (-6.61487486 + (1.98919585 - 0.22041650 * v) * v) * v) * v
    H1_case3 = ((0.554153432 +
                (0.278711796 + (-0.1883256872 + (0.042991293 - 0.003278278 * v) * v) * v) * v) /
                (v2 - 1.5))

    H1 = jnp.where(v < 1.3, H1_case1, jnp.where(v < 2.4, H1_case2, H1_case3))
    H2 = (1.0 - 2.0 * v2) * H0

    return H0, H1, H2


@jax.jit
def _voigt_hjerting_vectorized_jax(alpha: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Vectorized Hjerting function with Korg.jl regime selection (JAX)."""
    v2 = v * v
    sqrt_pi = jnp.sqrt(PI)

    alpha_safe = jnp.maximum(alpha, 1e-30)
    v2_safe = jnp.maximum(v2, 1e-30)

    mask1 = (alpha <= 0.2) & (v >= 5.0)
    mask2 = (alpha <= 0.2) & (v < 5.0)
    mask3 = (alpha > 0.2) & (alpha <= 1.4) & ((alpha + v) < 3.2)
    mask4 = ~(mask1 | mask2 | mask3)

    invv2 = 1.0 / v2_safe
    result = jnp.zeros_like(v)

    result = jnp.where(
        mask1,
        (alpha / sqrt_pi * invv2) * (1.0 + 1.5 * invv2 + 3.75 * invv2 * invv2),
        result
    )

    H0, H1, H2 = _harris_series_vectorized_jax(v)
    result = jnp.where(
        mask2,
        H0 + (H1 + H2 * alpha) * alpha,
        result
    )

    M0 = H0
    M1 = H1 + 2.0 / sqrt_pi * M0
    M2 = H2 - M0 + 2.0 / sqrt_pi * M1
    M3 = (2.0 / (3.0 * sqrt_pi)) * (1.0 - H2) - (2.0 / 3.0) * v2 * M1 + (2.0 / sqrt_pi) * M2
    M4 = (2.0 / 3.0) * v2 * v2 * M0 - (2.0 / (3.0 * sqrt_pi)) * M1 + (2.0 / sqrt_pi) * M3
    psi = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * alpha) * alpha) * alpha
    result = jnp.where(
        mask3,
        psi * (M0 + (M1 + (M2 + (M3 + M4 * alpha) * alpha) * alpha) * alpha),
        result
    )

    r2 = v2_safe / (alpha_safe * alpha_safe)
    alpha_invu = 1.0 / (jnp.sqrt(2.0) * ((r2 + 1.0) * alpha_safe))
    alpha2_invu2 = alpha_invu * alpha_invu
    result = jnp.where(
        mask4,
        jnp.sqrt(2.0 / PI) * alpha_invu * (
            1.0 + (3.0 * r2 - 1.0 + ((r2 - 2.0) * 15.0 * r2 + 2.0) * alpha2_invu2) * alpha2_invu2
        ),
        result
    )

    return result


class KorgLineProcessor:
    """
    Direct implementation of Korg.jl's line_absorption.jl algorithm
    
    This processor mirrors the exact structure and calculations from Korg.jl
    to ensure perfect compatibility and eliminate architectural issues.
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        
    def process_lines(self, 
                     wl_array_cm: np.ndarray,
                     temps: np.ndarray,
                     electron_densities: np.ndarray,
                     n_densities: Dict[Species, np.ndarray],
                     partition_fns: Dict[Species, Any],
                     linelist: List[Any],
                     microturbulence_cm_s: float,
                     continuum_opacity_fn: Optional[Any] = None,
                     continuum_opacity: Optional[np.ndarray] = None,
                     cutoff_threshold: float = 3e-4,
                     use_jax: bool = True,
                     batch_size: int = 256) -> KorgLineResult:
        """
        Process all lines using exact Korg.jl algorithm
        
        Parameters
        ----------
        wl_array_cm : array_like, shape [n_wavelengths]
            Wavelength array in cm
        temps : array_like, shape [n_layers]  
            Temperature array in K
        electron_densities : array_like, shape [n_layers]
            Electron density array in cm^-3
        n_densities : dict
            Species number densities {Species: array[n_layers]} in cm^-3
        partition_fns : dict
            Partition functions {Species: callable}
        linelist : list
            List of Line objects
        microturbulence_cm_s : float
            Microturbulent velocity in cm/s (NOT km/s!)
        continuum_opacity_fn : callable, optional
            Function returning continuum opacity at wavelength
        continuum_opacity : array_like, optional
            Continuum opacity matrix [layers × wavelengths] for JAX windowing
        cutoff_threshold : float
            Line windowing threshold (default: 3e-4)
        use_jax : bool
            Use JAX-compiled line processing (recommended for speed)
        batch_size : int
            Number of lines to process per JAX batch
            
        Returns
        -------
        KorgLineResult
            Processing results with alpha matrix [layers × wavelengths]
        """
        if len(linelist) == 0:
            return KorgLineResult(
                alpha_matrix=np.zeros((len(temps), len(wl_array_cm))),
                lines_processed=0,
                lines_windowed=0,
                total_amplitude=0.0
            )
        
        n_layers = len(temps)
        n_wavelengths = len(wl_array_cm)
        
        if self.verbose:
            print(f"🔬 KORG LINE PROCESSOR")
            print(f"   Lines to process: {len(linelist)}")
            print(f"   Atmospheric layers: {n_layers}")
            print(f"   Wavelength points: {n_wavelengths}")
            print(f"   Cutoff threshold: {cutoff_threshold:.0e}")
            # Initialize debug counter and warning tracking
            self._debug_line_count = 0
            self._warned_species = set()
        
        # Calculate β = 1/(k*T) for all layers (Korg.jl line 36)
        beta = 1.0 / (kboltz_eV * temps)
        
        # Precompute n_div_U for all species (Korg.jl lines 38-41)
        n_div_U = self._compute_n_div_U(n_densities, partition_fns, temps)

        if use_jax:
            return self._process_lines_jax(
                wl_array_cm=wl_array_cm,
                temps=temps,
                electron_densities=electron_densities,
                n_densities=n_densities,
                n_div_U=n_div_U,
                linelist=linelist,
                microturbulence_cm_s=microturbulence_cm_s,
                continuum_opacity_fn=continuum_opacity_fn,
                continuum_opacity=continuum_opacity,
                cutoff_threshold=cutoff_threshold,
                batch_size=batch_size
            )

        # Initialize alpha matrix [layers × wavelengths]
        alpha_matrix = np.zeros((n_layers, n_wavelengths))
        
        # Process each line (Korg.jl lines 66-106)
        lines_processed = 0
        lines_windowed = 0
        lines_windowed_out = 0  # Track lines completely excluded
        total_amplitude = 0.0
        
        # Track statistics for debugging
        window_sizes = []
        amplitude_values = []
        
        for line_idx, line in enumerate(linelist):
            # DETAILED DEBUGGING: Track every step for first few lines (only if verbose)
            debug_this_line = (self.verbose and line_idx < 5)
            
            if debug_this_line:
                print(f"\n   🔍 DEBUG LINE {line_idx}: {line.wavelength*1e8:.2f} Å, species={line.species}")
                print(f"      log_gf={line.log_gf:.3f}, E_lower={line.E_lower:.3f} eV")
            
            result = self._process_single_line(
                line=line,
                wl_array_cm=wl_array_cm,
                temps=temps,
                electron_densities=electron_densities,
                n_densities=n_densities,
                n_div_U=n_div_U,
                beta=beta,
                microturbulence_cm_s=microturbulence_cm_s,
                continuum_opacity_fn=continuum_opacity_fn,
                cutoff_threshold=cutoff_threshold,
                debug=debug_this_line
            )
            
            if result is not None:
                lb, ub, line_alpha_window, amplitude = result
                if lb < ub:
                    alpha_matrix[:, lb:ub] += line_alpha_window
                total_amplitude += amplitude
                lines_processed += 1
                
                if debug_this_line:
                    max_alpha = np.max(line_alpha_window) if line_alpha_window.size else 0.0
                    print(f"      ✅ ADDED: amplitude={amplitude:.2e}, max_alpha={max_alpha:.2e}")
                
                # Check if line was windowed (truncated)
                if line_alpha_window.size and np.any(line_alpha_window > 0):
                    lines_windowed += 1
                    amplitude_values.append(amplitude)
            else:
                lines_windowed_out += 1
                if debug_this_line:
                    print(f"      ❌ WINDOWED OUT: No contribution")
                elif self.verbose and line_idx < 10:  # Show first 10 windowed lines
                    print(f"     Line {line_idx} at {line.wavelength*1e8:.2f} Å windowed out")
        
        if self.verbose:
            print(f"   ✅ Processed: {lines_processed}/{len(linelist)} lines")
            print(f"   ✅ Contributing: {lines_windowed} lines with non-zero opacity")
            print(f"   ❌ Windowed out: {lines_windowed_out} lines (too weak)")
            print(f"   ✅ Total amplitude: {total_amplitude:.2e}")
            max_opacity = np.max(alpha_matrix)
            print(f"   ✅ Max line opacity: {max_opacity:.2e} cm⁻¹")
            
            # Show reduction factor
            if len(linelist) > 0:
                reduction_factor = lines_windowed_out / len(linelist) * 100
                print(f"   📊 Reduction: {reduction_factor:.1f}% of lines excluded")
                effective_density = lines_windowed / (wl_array_cm[-1] - wl_array_cm[0]) * 1e-8
                print(f"   📊 Effective line density: {effective_density:.1f} lines/Å")
        
        return KorgLineResult(
            alpha_matrix=alpha_matrix,
            lines_processed=lines_processed,
            lines_windowed=lines_windowed,
            total_amplitude=total_amplitude
        )
    
    def _compute_n_div_U(self, n_densities: Dict[Species, np.ndarray], 
                        partition_fns: Dict[Species, Any], 
                        temps: np.ndarray) -> Dict[Species, np.ndarray]:
        """
        Compute number density / partition function (Korg.jl lines 39-41)
        """
        n_div_U = {}
        
        for species in n_densities.keys():
            if species in partition_fns:
                # Calculate partition function at all temperatures
                log_temps = np.log(temps)
                U_values = np.array([partition_fns[species](log_T) for log_T in log_temps])
                
                # Calculate n/U for all layers (avoid divide by zero)
                n_div_U[species] = n_densities[species] / np.maximum(U_values, 1e-50)
            else:
                # Fallback: use simple temperature scaling
                U_fallback = 25.0 * (temps / 5778.0)**0.3
                n_div_U[species] = n_densities[species] / U_fallback
                
        return n_div_U

    def _resolve_continuum_opacity(self, wl_array_cm: np.ndarray,
                                  continuum_opacity: Optional[np.ndarray],
                                  continuum_opacity_fn: Optional[Any],
                                  n_layers: int) -> Optional[np.ndarray]:
        """
        Resolve continuum opacity into a [layers × wavelengths] matrix.
        """
        if continuum_opacity is not None:
            cont = np.asarray(continuum_opacity)
            if cont.ndim == 1:
                cont = cont[None, :]
            return cont

        if continuum_opacity_fn is None:
            return None

        samples = []
        for wl_cm in wl_array_cm:
            value = np.asarray(continuum_opacity_fn(wl_cm))
            if value.ndim == 0:
                value = np.full(n_layers, float(value))
            samples.append(value)

        return np.stack(samples, axis=1)

    def _pack_linelist_arrays(self, linelist: List[Any], species_index: Dict[Species, int],
                             temp_ref: float) -> Optional[Dict[str, np.ndarray]]:
        """
        Pack linelist objects into dense numpy arrays for JAX processing.
        """
        from ..lines.broadening_korg import approximate_vdw_broadening

        wavelengths = []
        log_gf = []
        E_lower = []
        gamma_rad = []
        gamma_stark = []
        vdw_sigma = []
        vdw_alpha = []
        vdw_base_gamma = []
        species_idx = []
        atomic_mass = []
        is_molecule = []

        warned = set()

        for line in linelist:
            species = line.species
            idx = species_index.get(species)
            if idx is None:
                if self.verbose and species not in warned:
                    print(f"       ⚠️  Skipping line species not in n_div_U: {species}")
                    warned.add(species)
                continue

            wl_val = float(line.wavelength)
            log_gf_val = float(line.log_gf)
            E_lower_val = float(line.E_lower)

            if not np.isfinite(wl_val) or wl_val <= 0.0:
                if self.verbose:
                    print("       ⚠️  Skipping line with invalid wavelength")
                continue
            if not np.isfinite(log_gf_val) or not np.isfinite(E_lower_val):
                if self.verbose:
                    print("       ⚠️  Skipping line with invalid log_gf/E_lower")
                continue

            gamma_rad_val = float(getattr(line, 'gamma_rad', 6.16e7))
            if not np.isfinite(gamma_rad_val) or gamma_rad_val < 0:
                gamma_rad_val = 6.16e7

            gamma_stark_val = float(getattr(line, 'gamma_stark', 0.0))
            if not np.isfinite(gamma_stark_val) or gamma_stark_val < 0:
                gamma_stark_val = 0.0

            if hasattr(line, 'vdW'):
                vdW_param = line.vdW
            elif hasattr(line, 'vdw_param1') and hasattr(line, 'vdw_param2'):
                vdW_param = (line.vdw_param1, line.vdw_param2)
            else:
                vdW_param = (0.0, -1)

            sigma, alpha = vdW_param
            sigma = float(sigma)
            alpha = float(alpha)
            if not np.isfinite(sigma):
                sigma = 0.0
            if not np.isfinite(alpha):
                alpha = -1.0
            if sigma < 0.0:
                sigma = 0.0

            if alpha == -2:
                base_gamma = approximate_vdw_broadening(
                    self._map_vald_species_to_jorg(species),
                    E_lower_val,
                    wl_val,
                    float(temp_ref)
                )
            else:
                base_gamma = 1.0

            wavelengths.append(wl_val)
            log_gf.append(log_gf_val)
            E_lower.append(E_lower_val)
            gamma_rad.append(gamma_rad_val)
            gamma_stark.append(gamma_stark_val)
            vdw_sigma.append(sigma)
            vdw_alpha.append(alpha)
            vdw_base_gamma.append(float(base_gamma))
            species_idx.append(int(idx))
            atomic_mass.append(float(self._get_atomic_mass(species)))
            is_molecule.append(bool(self._is_molecule(species)))

        if not wavelengths:
            return None

        return {
            "wavelength": np.asarray(wavelengths, dtype=np.float64),
            "log_gf": np.asarray(log_gf, dtype=np.float64),
            "E_lower": np.asarray(E_lower, dtype=np.float64),
            "gamma_rad": np.asarray(gamma_rad, dtype=np.float64),
            "gamma_stark": np.asarray(gamma_stark, dtype=np.float64),
            "vdw_sigma": np.asarray(vdw_sigma, dtype=np.float64),
            "vdw_alpha": np.asarray(vdw_alpha, dtype=np.float64),
            "vdw_base_gamma": np.asarray(vdw_base_gamma, dtype=np.float64),
            "species_idx": np.asarray(species_idx, dtype=np.int32),
            "atomic_mass": np.asarray(atomic_mass, dtype=np.float64),
            "is_molecule": np.asarray(is_molecule, dtype=bool)
        }

    def _compute_line_windows_numpy(self, wl_array_cm: np.ndarray, temps: np.ndarray,
                                   electron_densities: np.ndarray, n_div_U_array: np.ndarray,
                                   line_arrays: Dict[str, np.ndarray], microturbulence_cm_s: float,
                                   continuum_opacity: Optional[np.ndarray], cutoff_threshold: float,
                                   n_h_neutral: np.ndarray, float_dtype: type) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
        """
        Compute line windows with vectorized numpy (no Python loops).
        """
        from scipy.special import gamma as scipy_gamma

        n_layers = temps.shape[0]
        n_wavelengths = wl_array_cm.shape[0]

        line_wl = line_arrays["wavelength"].astype(float_dtype)
        log_gf = line_arrays["log_gf"].astype(float_dtype)
        E_lower = line_arrays["E_lower"].astype(float_dtype)
        gamma_rad = line_arrays["gamma_rad"].astype(float_dtype)
        gamma_stark = line_arrays["gamma_stark"].astype(float_dtype)
        vdw_sigma = line_arrays["vdw_sigma"].astype(float_dtype)
        vdw_alpha = line_arrays["vdw_alpha"].astype(float_dtype)
        vdw_base_gamma = line_arrays["vdw_base_gamma"].astype(float_dtype)
        species_idx = line_arrays["species_idx"]
        atomic_mass = line_arrays["atomic_mass"].astype(float_dtype)
        is_molecule = line_arrays["is_molecule"]

        temps = temps.astype(float_dtype)
        electron_densities = electron_densities.astype(float_dtype)
        n_h_neutral = n_h_neutral.astype(float_dtype)
        n_div_U_array = n_div_U_array.astype(float_dtype)

        beta = 1.0 / (kboltz_eV * temps)
        inv_mu_const = 1.0 / (1.008 * amu_cgs)

        sigma = line_wl[:, None] * np.sqrt(
            kboltz_cgs * temps[None, :] / atomic_mass[:, None] + (microturbulence_cm_s**2) / 2.0
        ) / c_cgs

        Gamma = gamma_rad[:, None] + temps[None, :] * 0.0

        is_atom = (~is_molecule).astype(float_dtype)
        stark = gamma_stark[:, None] * (temps[None, :] / 10000.0)**(1.0 / 6.0)
        Gamma = Gamma + (electron_densities[None, :] * stark) * is_atom[:, None]

        temp_vdw = (temps[None, :] / 10000.0)**0.3
        vdw_simple = vdw_sigma[:, None] * temp_vdw
        vdw_unsold = vdw_sigma[:, None] * vdw_base_gamma[:, None] * temp_vdw

        inv_mu = inv_mu_const + 1.0 / atomic_mass
        vbar = np.sqrt(8 * kboltz_cgs * temps[None, :] / PI * inv_mu[:, None])
        gamma_factor = scipy_gamma((4.0 - vdw_alpha) / 2.0)
        v0 = 1e6
        vdw_abo = 2.0 * (4.0 / PI)**(vdw_alpha[:, None] / 2.0) * gamma_factor[:, None] * v0 * vdw_sigma[:, None] * (vbar / v0)**(1.0 - vdw_alpha[:, None])

        vdw_gamma = np.where(
            (vdw_alpha == -1)[:, None],
            vdw_simple,
            np.where((vdw_alpha == -2)[:, None], vdw_unsold, vdw_abo)
        )
        Gamma = Gamma + (n_h_neutral[None, :] * vdw_gamma) * is_atom[:, None]

        gamma = Gamma * line_wl[:, None]**2 / (4.0 * PI * c_cgs)

        E_upper = E_lower + hplanck_eV * c_cgs / line_wl
        levels_factor = np.exp(-beta[None, :] * E_lower[:, None]) - np.exp(-beta[None, :] * E_upper[:, None])
        gf = np.power(10.0, log_gf)
        cross_section = (PI * ELECTRON_CHARGE**2 / ELECTRON_MASS / c_cgs**2) * line_wl**2

        n_div_U_lines = n_div_U_array[species_idx]
        amplitude = gf[:, None] * cross_section[:, None] * levels_factor * n_div_U_lines
        amplitude_safe = np.maximum(amplitude, 1e-50)

        if continuum_opacity is None:
            continuum_line = np.full_like(amplitude, 1e-6)
        else:
            continuum_opacity = np.asarray(continuum_opacity, dtype=float_dtype)
            if continuum_opacity.ndim == 1:
                continuum_opacity = continuum_opacity[None, :]
            if n_wavelengths == 1:
                continuum_line = np.tile(continuum_opacity[:, 0], (line_wl.shape[0], 1))
            else:
                idx = np.searchsorted(wl_array_cm, line_wl)
                idx = np.clip(idx, 1, n_wavelengths - 1)
                x0 = wl_array_cm[idx - 1]
                x1 = wl_array_cm[idx]
                dx = x1 - x0
                frac = np.where(dx != 0.0, (line_wl - x0) / dx, 0.0)
                cont0 = np.take(continuum_opacity, idx - 1, axis=1)
                cont1 = np.take(continuum_opacity, idx, axis=1)
                continuum_line = cont0 + (cont1 - cont0) * frac[None, :]
                below = line_wl <= wl_array_cm[0]
                above = line_wl >= wl_array_cm[-1]
                continuum_line = np.where(below[None, :], continuum_opacity[:, 0][:, None], continuum_line)
                continuum_line = np.where(above[None, :], continuum_opacity[:, -1][:, None], continuum_line)
                continuum_line = continuum_line.T

        rho_crit = (continuum_line * cutoff_threshold) / amplitude_safe

        sigma_safe = np.maximum(sigma, 1e-30)
        gamma_safe = np.maximum(gamma, 1e-30)
        sqrt_2pi = np.sqrt(2.0 * PI)

        threshold_g = 1.0 / (sqrt_2pi * sigma_safe)
        safe_g = np.clip(sqrt_2pi * sigma_safe * rho_crit, 1e-300, 1.0)
        doppler_val = sigma_safe * np.sqrt(-2.0 * np.log(safe_g))
        doppler_windows = np.where(rho_crit <= threshold_g, doppler_val, 0.0)

        threshold_l = 1.0 / (PI * gamma_safe)
        safe_rho = np.maximum(rho_crit, 1e-300)
        lorentz_val = np.sqrt(np.maximum(gamma_safe / (PI * safe_rho) - gamma_safe * gamma_safe, 0.0))
        lorentz_windows = np.where(rho_crit <= threshold_l, lorentz_val, 0.0)

        doppler_window = np.max(doppler_windows, axis=1)
        lorentz_window = np.max(lorentz_windows, axis=1)
        window_size = np.sqrt(lorentz_window**2 + doppler_window**2)

        lb = np.searchsorted(wl_array_cm, line_wl - window_size)
        ub = np.searchsorted(wl_array_cm, line_wl + window_size, side='right')
        lb = np.maximum(lb, 0)
        ub = np.minimum(ub, n_wavelengths)

        window_len = np.maximum(ub - lb, 0)
        max_window_pts = int(window_len.max()) if window_len.size else 0
        lines_windowed = int(np.sum(window_len > 0))
        total_amplitude = float(np.sum(amplitude.mean(axis=1)))

        return lb.astype(np.int32), ub.astype(np.int32), max_window_pts, lines_windowed, total_amplitude

    def _process_lines_jax(self, wl_array_cm: np.ndarray, temps: np.ndarray,
                          electron_densities: np.ndarray, n_densities: Dict[Species, np.ndarray],
                          n_div_U: Dict[Species, np.ndarray], linelist: List[Any],
                          microturbulence_cm_s: float, continuum_opacity_fn: Optional[Any],
                          continuum_opacity: Optional[np.ndarray], cutoff_threshold: float,
                          batch_size: int) -> KorgLineResult:
        """
        JAX-compiled line processing with Korg.jl windowing and no Python loops.
        """
        n_layers = len(temps)
        n_wavelengths = len(wl_array_cm)

        continuum_opacity = self._resolve_continuum_opacity(
            wl_array_cm, continuum_opacity, continuum_opacity_fn, n_layers
        )

        species_list = list(n_div_U.keys())
        species_index = {species: idx for idx, species in enumerate(species_list)}
        n_div_U_array = np.stack([n_div_U[species] for species in species_list], axis=0)

        line_arrays = self._pack_linelist_arrays(linelist, species_index, temps[0])
        if line_arrays is None:
            return KorgLineResult(
                alpha_matrix=np.zeros((n_layers, n_wavelengths)),
                lines_processed=0,
                lines_windowed=0,
                total_amplitude=0.0
            )

        h_neutral = Species.from_atomic_number(1, 0)
        n_h_neutral = n_densities.get(h_neutral, np.zeros(n_layers))

        use_x64 = bool(jax.config.read("jax_enable_x64")) if hasattr(jax.config, "read") else False
        float_dtype = np.float64 if use_x64 else np.float32

        lb, ub, max_window_pts, lines_windowed, total_amplitude = self._compute_line_windows_numpy(
            wl_array_cm=wl_array_cm,
            temps=temps,
            electron_densities=electron_densities,
            n_div_U_array=n_div_U_array,
            line_arrays=line_arrays,
            microturbulence_cm_s=microturbulence_cm_s,
            continuum_opacity=continuum_opacity,
            cutoff_threshold=cutoff_threshold,
            n_h_neutral=n_h_neutral,
            float_dtype=float_dtype
        )

        lines_processed = line_arrays["wavelength"].shape[0]
        if max_window_pts <= 0 or lines_processed == 0:
            return KorgLineResult(
                alpha_matrix=np.zeros((n_layers, n_wavelengths)),
                lines_processed=lines_processed,
                lines_windowed=0,
                total_amplitude=total_amplitude
            )

        batch_size = int(max(1, batch_size))
        max_elements = 2_000_000
        batch_limit = max(1, int(max_elements / max(1, n_layers * max_window_pts)))
        batch_size = min(batch_size, lines_processed, batch_limit)

        pad = (-lines_processed) % batch_size
        if pad:
            def _pad(arr, pad_value=0):
                return np.pad(arr, (0, pad), constant_values=pad_value)

            line_arrays = {
                "wavelength": _pad(line_arrays["wavelength"]),
                "log_gf": _pad(line_arrays["log_gf"]),
                "E_lower": _pad(line_arrays["E_lower"]),
                "gamma_rad": _pad(line_arrays["gamma_rad"]),
                "gamma_stark": _pad(line_arrays["gamma_stark"]),
                "vdw_sigma": _pad(line_arrays["vdw_sigma"]),
                "vdw_alpha": _pad(line_arrays["vdw_alpha"]),
                "vdw_base_gamma": _pad(line_arrays["vdw_base_gamma"], pad_value=1.0),
                "species_idx": _pad(line_arrays["species_idx"]),
                "atomic_mass": _pad(line_arrays["atomic_mass"], pad_value=amu_cgs),
                "is_molecule": _pad(line_arrays["is_molecule"], pad_value=False)
            }
            lb = _pad(lb)
            ub = _pad(ub)

        line_mask = np.ones(lines_processed, dtype=float_dtype)
        if pad:
            line_mask = np.pad(line_mask, (0, pad), constant_values=0.0)

        n_lines_padded = line_arrays["wavelength"].shape[0]
        n_batches = n_lines_padded // batch_size

        wl_array_cm_j = jnp.asarray(wl_array_cm)
        temps_j = jnp.asarray(temps)
        electron_densities_j = jnp.asarray(electron_densities)
        n_div_U_array_j = jnp.asarray(n_div_U_array)
        n_h_neutral_j = jnp.asarray(n_h_neutral)

        line_wl_j = jnp.asarray(line_arrays["wavelength"])
        log_gf_j = jnp.asarray(line_arrays["log_gf"])
        E_lower_j = jnp.asarray(line_arrays["E_lower"])
        gamma_rad_j = jnp.asarray(line_arrays["gamma_rad"])
        gamma_stark_j = jnp.asarray(line_arrays["gamma_stark"])
        vdw_sigma_j = jnp.asarray(line_arrays["vdw_sigma"])
        vdw_alpha_j = jnp.asarray(line_arrays["vdw_alpha"])
        vdw_base_gamma_j = jnp.asarray(line_arrays["vdw_base_gamma"])
        species_idx_j = jnp.asarray(line_arrays["species_idx"], dtype=jnp.int32)
        atomic_mass_j = jnp.asarray(line_arrays["atomic_mass"])
        is_molecule_j = jnp.asarray(line_arrays["is_molecule"], dtype=temps_j.dtype)
        lb_j = jnp.asarray(lb, dtype=jnp.int32)
        ub_j = jnp.asarray(ub, dtype=jnp.int32)
        line_mask_j = jnp.asarray(line_mask, dtype=temps_j.dtype)

        beta_j = 1.0 / (kboltz_eV * temps_j)
        temp_stark = (temps_j / 10000.0)**(1.0 / 6.0)
        temp_vdw = (temps_j / 10000.0)**0.3
        temp_vbar = jnp.sqrt(8.0 * kboltz_cgs * temps_j / PI)
        inv_mu_const = 1.0 / (1.008 * amu_cgs)
        sigma_line_const = (PI * ELECTRON_CHARGE**2 / ELECTRON_MASS / c_cgs**2)

        arange_window = jnp.arange(max_window_pts, dtype=jnp.int32)

        def batch_body(i, alpha_matrix):
            start = i * batch_size
            line_wl = jax.lax.dynamic_slice(line_wl_j, (start,), (batch_size,))
            log_gf = jax.lax.dynamic_slice(log_gf_j, (start,), (batch_size,))
            E_lower = jax.lax.dynamic_slice(E_lower_j, (start,), (batch_size,))
            gamma_rad = jax.lax.dynamic_slice(gamma_rad_j, (start,), (batch_size,))
            gamma_stark = jax.lax.dynamic_slice(gamma_stark_j, (start,), (batch_size,))
            vdw_sigma = jax.lax.dynamic_slice(vdw_sigma_j, (start,), (batch_size,))
            vdw_alpha = jax.lax.dynamic_slice(vdw_alpha_j, (start,), (batch_size,))
            vdw_base_gamma = jax.lax.dynamic_slice(vdw_base_gamma_j, (start,), (batch_size,))
            species_idx = jax.lax.dynamic_slice(species_idx_j, (start,), (batch_size,))
            atomic_mass = jax.lax.dynamic_slice(atomic_mass_j, (start,), (batch_size,))
            is_molecule = jax.lax.dynamic_slice(is_molecule_j, (start,), (batch_size,))
            lb_b = jax.lax.dynamic_slice(lb_j, (start,), (batch_size,))
            ub_b = jax.lax.dynamic_slice(ub_j, (start,), (batch_size,))
            line_mask = jax.lax.dynamic_slice(line_mask_j, (start,), (batch_size,))

            sigma = line_wl[:, None] * jnp.sqrt(
                kboltz_cgs * temps_j[None, :] / atomic_mass[:, None] + (microturbulence_cm_s**2) / 2.0
            ) / c_cgs
            sigma = jnp.maximum(sigma, 1e-30)

            is_atom = 1.0 - is_molecule
            Gamma = gamma_rad[:, None] + is_atom[:, None] * (electron_densities_j[None, :] * (gamma_stark[:, None] * temp_stark[None, :]))

            inv_mu = inv_mu_const + 1.0 / atomic_mass
            vbar = temp_vbar[None, :] * jnp.sqrt(inv_mu[:, None])
            gamma_factor = jax_gamma((4.0 - vdw_alpha) / 2.0)
            v0 = 1e6
            vdw_abo = 2.0 * (4.0 / PI)**(vdw_alpha[:, None] / 2.0) * gamma_factor[:, None] * v0 * vdw_sigma[:, None] * (vbar / v0)**(1.0 - vdw_alpha[:, None])
            vdw_simple = vdw_sigma[:, None] * temp_vdw[None, :]
            vdw_unsold = vdw_sigma[:, None] * vdw_base_gamma[:, None] * temp_vdw[None, :]
            vdw_gamma = jnp.where(vdw_alpha[:, None] == -1.0, vdw_simple,
                                  jnp.where(vdw_alpha[:, None] == -2.0, vdw_unsold, vdw_abo))
            Gamma = Gamma + is_atom[:, None] * (n_h_neutral_j[None, :] * vdw_gamma)

            gamma = Gamma * line_wl[:, None]**2 / (4.0 * PI * c_cgs)

            E_upper = E_lower + hplanck_eV * c_cgs / line_wl
            levels_factor = jnp.exp(-beta_j[None, :] * E_lower[:, None]) - jnp.exp(-beta_j[None, :] * E_upper[:, None])
            gf = jnp.power(10.0, log_gf)
            cross_section = sigma_line_const * line_wl**2
            n_div_U_line = n_div_U_array_j[species_idx]
            amplitude = gf[:, None] * cross_section[:, None] * levels_factor * n_div_U_line
            amplitude = amplitude * line_mask[:, None]

            inv_sigma_sqrt2 = 1.0 / (sigma * jnp.sqrt(2.0))
            scaling = inv_sigma_sqrt2 / jnp.sqrt(PI) * amplitude
            alpha = gamma * inv_sigma_sqrt2

            window_idx = lb_b[:, None] + arange_window[None, :]
            window_mask = window_idx < ub_b[:, None]
            window_idx_clipped = jnp.clip(window_idx, 0, n_wavelengths - 1)
            wl_window = jnp.take(wl_array_cm_j, window_idx_clipped)

            v = jnp.abs(wl_window[:, None, :] - line_wl[:, None, None]) * inv_sigma_sqrt2[:, :, None]
            voigt_values = _voigt_hjerting_vectorized_jax(alpha[:, :, None], v)
            line_alpha = voigt_values * scaling[:, :, None]
            line_alpha = jnp.nan_to_num(line_alpha, nan=0.0, posinf=0.0, neginf=0.0)

            line_alpha = line_alpha * window_mask[:, None, :] * line_mask[:, None, None]
            alpha_matrix = alpha_matrix.at[:, window_idx_clipped].add(jnp.transpose(line_alpha, (1, 0, 2)))

            return alpha_matrix

        def run_batches(alpha_init):
            return jax.lax.fori_loop(0, n_batches, batch_body, alpha_init)

        alpha_init = jnp.zeros((n_layers, n_wavelengths), dtype=temps_j.dtype)
        alpha_matrix_j = jax.jit(run_batches)(alpha_init)

        alpha_matrix = np.asarray(alpha_matrix_j)

        if self.verbose:
            max_opacity = np.max(alpha_matrix) if alpha_matrix.size else 0.0
            print(f"   ✅ JAX processed: {lines_processed} lines")
            print(f"   ✅ JAX contributing: {lines_windowed} lines")
            print(f"   ✅ JAX max line opacity: {max_opacity:.2e} cm⁻¹")

        return KorgLineResult(
            alpha_matrix=alpha_matrix,
            lines_processed=lines_processed,
            lines_windowed=lines_windowed,
            total_amplitude=total_amplitude
        )
    
    def _process_single_line(self, line, wl_array_cm, temps, electron_densities, 
                           n_densities, n_div_U, beta, microturbulence_cm_s,
                           continuum_opacity_fn, cutoff_threshold, debug=False) -> Optional[Tuple[int, int, np.ndarray, float]]:
        """
        Process a single line following Korg.jl algorithm exactly (lines 66-106)
        
        Returns
        -------
        tuple or None
            (line_alpha_matrix, amplitude) if line contributes, None if windowed out
        """
        n_layers = len(temps)
        n_wavelengths = len(wl_array_cm)
        
        # Get atomic mass (Korg.jl line 67)
        atomic_mass = self._get_atomic_mass(line.species)
        if debug:
            print(f"        atomic_mass = {atomic_mass/amu_cgs:.1f} amu ({atomic_mass:.2e} g)")
        
        # Calculate Doppler width σ for all layers (Korg.jl line 70)
        sigma = self._doppler_width(line.wavelength, temps, atomic_mass, microturbulence_cm_s)
        if debug:
            print(f"        sigma (Doppler) = {np.mean(sigma):.2e} cm (mean)")
        
        # Calculate broadening parameters (Korg.jl lines 74-83)
        gamma = self._calculate_lorentz_broadening(line, temps, electron_densities, 
                                                  n_densities, atomic_mass)
        if debug:
            print(f"        gamma (Lorentz) = {np.mean(gamma):.2e} cm (mean)")
        
        # Calculate level population factor (Korg.jl lines 85-86)
        E_upper = line.E_lower + hplanck_eV * c_cgs / line.wavelength
        levels_factor = np.exp(-beta * line.E_lower) - np.exp(-beta * E_upper)
        if debug:
            print(f"        E_upper = {E_upper:.3f} eV, levels_factor = {np.mean(levels_factor):.2e} (mean)")
        
        # Calculate line amplitude for all layers (Korg.jl lines 89-90)
        gf = 10.0**line.log_gf
        cross_section = self._sigma_line(line.wavelength)
        if debug:
            print(f"        log_gf = {line.log_gf:.3f}")
            print(f"        gf = 10^{line.log_gf:.3f} = {gf:.3e}")
            print(f"        cross_section (σ_line) = {cross_section:.3e} cm²")
            print(f"        wavelength = {line.wavelength*1e8:.2f} Å = {line.wavelength:.3e} cm")
        
        # Get number density / partition function for this species  
        # VALD linelist species are already Jorg Species objects
        species_for_lookup = line.species
        if debug:
            print(f"        Species: {species_for_lookup} (type: {type(species_for_lookup)})")
        
        if species_for_lookup in n_div_U:
            n_div_U_species = n_div_U[species_for_lookup]
            if debug:
                print(f"        n_div_U found: mean = {np.mean(n_div_U_species):.2e}")
        else:
            # CRITICAL FIX: Skip lines for species not in chemical equilibrium
            # Using fallback values causes HUGE windows for rare earth elements
            # because tiny n_div_U → tiny amplitude → tiny rho_crit → huge window
            if debug or (self.verbose and hasattr(self, '_warned_species')):
                if not hasattr(self, '_warned_species'):
                    self._warned_species = set()
                if species_for_lookup not in self._warned_species:
                    if debug:
                        print(f"        ⚠️  WARNING: Species {species_for_lookup} not in n_div_U")
                        print(f"            SKIPPING this line (not in chemical equilibrium)")
                    self._warned_species.add(species_for_lookup)
            # Return None to skip this line
            return None
            
        amplitude = gf * cross_section * levels_factor * n_div_U_species
        if debug:
            print(f"        amplitude = {np.mean(amplitude):.2e} cm⁻¹ (mean)")
        
        # Apply line windowing algorithm (Korg.jl lines 92-105)
        return self._apply_windowing(
            line=line,
            wl_array_cm=wl_array_cm,
            sigma=sigma,
            gamma=gamma, 
            amplitude=amplitude,
            continuum_opacity_fn=continuum_opacity_fn,
            cutoff_threshold=cutoff_threshold,
            debug=debug
        )
    
    
    def _map_vald_species_to_jorg(self, vald_species) -> Species:
        """
        Map VALD species codes to Jorg Species objects
        
        VALD uses numerical codes like:
        - 2600: Fe I (iron neutral)
        - 2601: Fe II (iron singly ionized)  
        - 6001: C I (carbon neutral)
        - 601: C II (carbon singly ionized)
        - 699: C(-1) (carbon negative ion) - special case
        - etc.
        
        Jorg uses Species objects created from atomic number and ionization.
        """
        if isinstance(vald_species, (int, str)):
            # Convert string to int if needed
            species_code = int(vald_species) if isinstance(vald_species, str) else vald_species
            
            # Handle special cases for negative ions (codes < 100)
            if species_code < 100:
                # These are typically negative ions with special encoding
                # e.g., 699 could be C(-1), 199 could be H(-1), etc.
                if species_code == 699 or species_code == 99:
                    # C(-1) negative ion
                    return Species.from_atomic_number(6, -1)
                elif species_code == 199:
                    # H(-1) negative ion
                    return Species.from_atomic_number(1, -1)
                else:
                    # Other special codes - use neutral carbon as fallback
                    return Species.from_atomic_number(6, 0)
            
            # Standard VALD codes (>= 100)
            element_id = species_code // 100  # First 1-2 digits
            ionization = species_code % 100   # Last 2 digits
            
            # Handle negative ions indicated by high ionization numbers
            if ionization == 99:
                # Convention for negative ions in some VALD files
                ionization = -1
            
            # Create Jorg Species object
            try:
                return Species.from_atomic_number(element_id, ionization)
            except Exception:
                # Fallback to iron neutral if species creation fails
                return Species.from_atomic_number(26, 0)
        elif hasattr(vald_species, 'get_atom'):
            # Already a Jorg Species object
            return vald_species
        else:
            # Unknown species type, default to iron neutral
            return Species.from_atomic_number(26, 0)
    
    def _get_atomic_mass(self, species) -> float:
        """Get atomic mass for species (Korg.jl get_mass function)"""
        # Species from VALD linelist are already Jorg Species objects
        if hasattr(species, 'mass'):
            return float(species.mass) * amu_cgs
        
        # If for some reason we get a string/int, try to map it  
        if isinstance(species, (int, str)):
            jorg_species = self._map_vald_species_to_jorg(species)
            if hasattr(jorg_species, 'mass'):
                return float(jorg_species.mass) * amu_cgs
        
        # Final fallback: default to iron mass
        return 55.845 * amu_cgs
    
    def _doppler_width(self, wavelength_cm: float, temps: np.ndarray, 
                      atomic_mass: float, microturbulence_cm_s: float) -> np.ndarray:
        """
        Calculate Doppler width σ (NOT √2σ) - Korg.jl line 174
        
        doppler_width(λ₀, T, m, ξ) = λ₀ * sqrt(kboltz_cgs * T / m + (ξ^2) / 2) / c_cgs
        """
        thermal_velocity_sq = kboltz_cgs * temps / atomic_mass
        micro_velocity_sq = (microturbulence_cm_s**2) / 2
        return wavelength_cm * np.sqrt(thermal_velocity_sq + micro_velocity_sq) / c_cgs
    
    def _calculate_lorentz_broadening(self, line, temps, electron_densities, 
                                    n_densities, atomic_mass) -> np.ndarray:
        """
        Calculate Lorentz broadening γ (Korg.jl lines 74-83)
        """
        # Start with radiative broadening (Korg.jl line 74)
        gamma_rad = getattr(line, 'gamma_rad', 6.16e7)  # Default from Korg.jl
        # VALD uses negative values to indicate "not available" - use default
        if gamma_rad < 0:
            gamma_rad = 6.16e7  # Korg.jl default
        Gamma = np.full_like(temps, gamma_rad)
        
        # Add Stark broadening for non-molecules (Korg.jl line 76)
        if not self._is_molecule(line.species):
            gamma_stark = getattr(line, 'gamma_stark', 0.0)
            # VALD uses negative values to indicate "not available" - treat as zero
            if gamma_stark < 0:
                gamma_stark = 0.0
            Gamma += electron_densities * self._scaled_stark(gamma_stark, temps)
            
            # Add van der Waals broadening (Korg.jl line 77)
            # Create H I species for lookup
            try:
                h_neutral = Species.from_atomic_number(1, 0)  # H I
                if h_neutral in n_densities:
                    # Get vdW parameters from line data
                    if hasattr(line, 'vdW'):
                        vdW_param = line.vdW
                    elif hasattr(line, 'vdw_param1') and hasattr(line, 'vdw_param2'):
                        vdW_param = (line.vdw_param1, line.vdw_param2)
                    else:
                        vdW_param = (0.0, -1)  # Default: no enhancement
                    
                    # Store line info for _scaled_vdW to access
                    self.species = line.species
                    self.E_lower = line.E_lower
                    self.wavelength = line.wavelength
                    
                    hydrogen_densities = n_densities[h_neutral]
                    Gamma += hydrogen_densities * self._scaled_vdW(vdW_param, atomic_mass, temps)
            except Exception as e:
                if self.verbose:
                    print(f"       Warning: vdW broadening failed: {e}")
                pass  # Skip vdW broadening if calculation fails
        
        # Convert to wavelength HWHM (Korg.jl line 83)
        # γ = Γ * λ²/(4π*c)
        gamma = Gamma * line.wavelength**2 / (4 * PI * c_cgs)
        
        return gamma
    
    def _is_molecule(self, species: Species) -> bool:
        """Check if species is a molecule"""
        # Prefer explicit molecule flags when available
        if hasattr(species, 'is_molecule'):
            return bool(species.is_molecule)
        if hasattr(species, 'formula') and hasattr(species.formula, 'is_molecule'):
            return bool(species.formula.is_molecule)
        # Check atomic number - molecules typically have atomic number > 92 in some systems
        if hasattr(species, 'element') and species.element > 92:
            return True
        return False
    
    def _scaled_stark(self, gamma_stark: float, temps: np.ndarray, T0: float = 10000.0) -> np.ndarray:
        """Stark broadening temperature scaling (Korg.jl line 179)"""
        return gamma_stark * (temps / T0)**(1.0/6.0)
    
    def _scaled_vdW(self, vdW_param: Tuple[float, float], atomic_mass: float, 
                   temps: np.ndarray) -> np.ndarray:
        """
        van der Waals broadening (Korg.jl lines 192-204)
        
        Handles different vdW parameter formats:
        - (value, -1): log10 enhancement factor
        - (value, -2): Unsöld fudge factor  
        - (σ, α) where α >= 0: ABO theory parameters
        """
        sigma, alpha = vdW_param
        
        if alpha == -1:
            # gamma_vdW evaluated at 10,000 K (Korg.jl: scaled_vdW with vdW[2] == -1)
            return sigma * np.power(temps / 10000.0, 0.3)
        elif alpha == -2:
            # Unsöld fudge factor (0 < value < 20)
            # Similar to above but with fudge factor multiplier
            from ..lines.broadening_korg import approximate_vdw_broadening
            # Map VALD species to Jorg species for broadening calculation
            jorg_species = self._map_vald_species_to_jorg(self.species)
            base_gamma = approximate_vdw_broadening(jorg_species, self.E_lower,
                                                    self.wavelength, temps[0])
            # Return array with proper shape for all temperature layers
            return sigma * base_gamma * np.power(temps / 10000.0, 0.3)
        else:
            # ABO theory (σ in cm², α dimensionless)
            v0 = 1e6  # Reference velocity cm/s
            inv_mu = 1.0 / (1.008 * amu_cgs) + 1.0 / atomic_mass  # Inverse reduced mass
            vbar = np.sqrt(8 * kboltz_cgs * temps / PI * inv_mu)  # Mean relative velocity
            
            from scipy.special import gamma as gamma_func
            gamma_factor = gamma_func((4 - alpha) / 2)
            
            # ABO formula from Anstee & O'Mara (1995)
            return 2 * (4/PI)**(alpha/2) * gamma_factor * v0 * sigma * (vbar/v0)**(1-alpha)
    
    def _sigma_line(self, wavelength_cm: float) -> float:
        """
        Line cross-section calculation (Korg.jl lines 213-221)
        
        REVERTED: Back to original formula to find actual root cause
        """
        return (PI * ELECTRON_CHARGE**2 / ELECTRON_MASS / c_cgs**2) * wavelength_cm**2
    
    def _apply_windowing(self, line, wl_array_cm, sigma, gamma, amplitude, 
                        continuum_opacity_fn, cutoff_threshold, debug=False) -> Optional[Tuple[int, int, np.ndarray, float]]:
        """
        Apply Korg.jl line windowing algorithm (lines 92-105)
        """
        n_layers, n_wavelengths = len(sigma), len(wl_array_cm)
        
        # Calculate ρ_crit for each layer (Korg.jl line 92)
        if continuum_opacity_fn is not None:
            continuum_opacity = continuum_opacity_fn(line.wavelength)
        else:
            continuum_opacity = 1e-6  # Default estimate
            
        # Handle potential division by zero in amplitude
        # Ensure amplitude has proper shape for element-wise operations
        amplitude_safe = np.maximum(amplitude, 1e-50)
        rho_crit = (continuum_opacity * cutoff_threshold) / amplitude_safe
        
        if debug:
            print(f"        continuum_opacity = {continuum_opacity:.2e} cm⁻¹")
            print(f"        cutoff_threshold = {cutoff_threshold:.0e}")
            print(f"        rho_crit = {np.mean(rho_crit):.2e} (mean)")
        
        # Calculate window sizes (Korg.jl lines 93-97)
        doppler_windows = self._inverse_gaussian_density_vec(rho_crit, sigma)
        lorentz_windows = self._inverse_lorentz_density_vec(rho_crit, gamma)
        
        # Combined window size (Korg.jl line 97)
        doppler_window = np.max(doppler_windows)
        lorentz_window = np.max(lorentz_windows)
        window_size = np.sqrt(lorentz_window**2 + doppler_window**2)
        
        # Find wavelength bounds (Korg.jl lines 98-103)
        lb = np.searchsorted(wl_array_cm, line.wavelength - window_size)
        ub = np.searchsorted(wl_array_cm, line.wavelength + window_size, side='right')

        if debug:
            print(f"        window_size = {window_size*1e8:.3f} Å")
            print(f"        line range: {(line.wavelength - window_size)*1e8:.2f} - {(line.wavelength + window_size)*1e8:.2f} Å")
            print(f"        grid range: {wl_array_cm[0]*1e8:.2f} - {wl_array_cm[-1]*1e8:.2f} Å")
            print(f"        lb={lb}, ub={ub} (of {n_wavelengths} points)")

        # EXACT Korg.jl behavior (line 101-103): if lb > ub, skip the line
        if lb >= ub:
            if debug:
                print(f"        ❌ REJECTED: Invalid bounds lb={lb} >= ub={ub}")
            return None

        # Ensure bounds are within array limits
        lb = max(0, lb)
        ub = min(n_wavelengths, ub)
        
        if debug:
            print(f"        ✅ ACCEPTED: Will compute profile for {ub-lb} wavelength points")
        
        # Calculate line profiles (Korg.jl line 105)
        wl_window = wl_array_cm[lb:ub]
        line_alpha_window = self._line_profile_matrix(
            line.wavelength, sigma, gamma, amplitude, wl_window
        )
        
        return lb, ub, line_alpha_window, np.mean(amplitude)
    
    def _inverse_gaussian_density(self, rho: float, sigma: float) -> float:
        """
        Inverse Gaussian density function (Korg.jl lines 124-129)

        Returns x such that ρ = exp(-0.5 x²/σ²) / √(2π)

        CRITICAL: Must return 0.0 for weak lines to match Korg.jl exactly
        """
        sqrt_2pi = np.sqrt(2 * PI)

        if rho > 1.0 / (sqrt_2pi * sigma):
            return 0.0  # EXACT Korg.jl behavior: exclude weak lines
        else:
            return sigma * np.sqrt(-2 * np.log(sqrt_2pi * sigma * rho))
    
    def _inverse_lorentz_density(self, rho: float, gamma: float) -> float:
        """
        Inverse Lorentz density function (Korg.jl lines 140-145)

        Returns x such that ρ = 1 / (π γ (1 + x²/γ²))

        CRITICAL: Must return 0.0 for weak lines to match Korg.jl exactly
        """
        if rho > 1.0 / (PI * gamma):
            return 0.0  # EXACT Korg.jl behavior: exclude weak lines
        else:
            return np.sqrt(gamma / (PI * rho) - gamma * gamma)

    def _inverse_gaussian_density_vec(self, rho: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Vectorized inverse Gaussian density function.
        """
        sqrt_2pi = np.sqrt(2 * PI)
        threshold = 1.0 / (sqrt_2pi * sigma)
        out = np.zeros_like(rho)
        mask = rho <= threshold
        safe = sqrt_2pi * sigma[mask] * rho[mask]
        out[mask] = sigma[mask] * np.sqrt(-2 * np.log(safe))
        return out

    def _inverse_lorentz_density_vec(self, rho: np.ndarray, gamma: np.ndarray) -> np.ndarray:
        """
        Vectorized inverse Lorentz density function.
        """
        threshold = 1.0 / (PI * gamma)
        out = np.zeros_like(rho)
        mask = rho <= threshold
        out[mask] = np.sqrt(gamma[mask] / (PI * rho[mask]) - gamma[mask] * gamma[mask])
        return out

    def _line_profile_matrix(self, lambda0: float, sigma: np.ndarray, gamma: np.ndarray,
                             amplitude: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """
        Vectorized Voigt line profile over all layers and wavelength points.
        """
        inv_sigma_sqrt2 = 1.0 / (sigma * np.sqrt(2.0))
        scaling = inv_sigma_sqrt2 / np.sqrt(PI) * amplitude
        alpha = gamma * inv_sigma_sqrt2
        v = np.abs(wavelengths[None, :] - lambda0) * inv_sigma_sqrt2[:, None]
        voigt_values = self._voigt_hjerting_vectorized(alpha[:, None], v)
        return voigt_values * scaling[:, None]
    
    def _line_profile(self, lambda0: float, sigma: float, gamma: float, 
                     amplitude: float, wavelength: float) -> float:
        """
        Voigt line profile (Korg.jl lines 229-233)
        
        Returns line opacity at given wavelength
        """
        inv_sigma_sqrt2 = 1.0 / (sigma * np.sqrt(2))
        scaling = inv_sigma_sqrt2 / np.sqrt(PI) * amplitude
        
        alpha = gamma * inv_sigma_sqrt2
        v = abs(wavelength - lambda0) * inv_sigma_sqrt2
        
        voigt_value = self._voigt_hjerting(alpha, v)
        return voigt_value * scaling

    def _voigt_hjerting_vectorized(self, alpha: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Vectorized Voigt-Hjerting function with Korg.jl regime selection.
        """
        if alpha.shape != v.shape:
            alpha = np.broadcast_to(alpha, v.shape)
        v2 = v * v
        sqrt_pi = np.sqrt(PI)

        result = np.zeros_like(v)

        mask1 = (alpha <= 0.2) & (v >= 5.0)
        if np.any(mask1):
            invv2 = 1.0 / v2[mask1]
            result[mask1] = (alpha[mask1] / sqrt_pi * invv2) * (
                1.0 + 1.5 * invv2 + 3.75 * invv2 * invv2
            )

        mask2 = (alpha <= 0.2) & (v < 5.0)
        mask3 = (alpha > 0.2) & (alpha <= 1.4) & ((alpha + v) < 3.2)
        if np.any(mask2) or np.any(mask3):
            H0, H1, H2 = self._harris_series_vectorized(v)
            if np.any(mask2):
                result[mask2] = H0[mask2] + (H1[mask2] + H2[mask2] * alpha[mask2]) * alpha[mask2]
            if np.any(mask3):
                M0 = H0
                M1 = H1 + 2.0 / sqrt_pi * M0
                M2 = H2 - M0 + 2.0 / sqrt_pi * M1
                M3 = (2.0 / (3.0 * sqrt_pi)) * (1.0 - H2) - (2.0 / 3.0) * v2 * M1 + (2.0 / sqrt_pi) * M2
                M4 = (2.0 / 3.0) * v2 * v2 * M0 - (2.0 / (3.0 * sqrt_pi)) * M1 + (2.0 / sqrt_pi) * M3

                psi = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * alpha) * alpha) * alpha
                result[mask3] = psi[mask3] * (
                    M0[mask3] + (M1[mask3] + (M2[mask3] + (M3[mask3] + M4[mask3] * alpha[mask3]) * alpha[mask3]) * alpha[mask3]) * alpha[mask3]
                )

        mask4 = ~(mask1 | mask2 | mask3)
        if np.any(mask4):
            r2 = v2[mask4] / (alpha[mask4] * alpha[mask4])
            alpha_invu = 1.0 / (np.sqrt(2.0) * ((r2 + 1.0) * alpha[mask4]))
            alpha2_invu2 = alpha_invu * alpha_invu
            result[mask4] = (np.sqrt(2.0 / PI) * alpha_invu * (
                1.0 + (3.0 * r2 - 1.0 + ((r2 - 2.0) * 15.0 * r2 + 2.0) * alpha2_invu2) * alpha2_invu2
            ))

        return result

    def _harris_series_vectorized(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized Harris series coefficients.
        """
        v2 = v * v
        H0 = np.exp(-v2)

        H1_case1 = -1.12470432 + (-0.15516677 + (3.288675912 + (-2.34357915 + 0.42139162 * v) * v) * v) * v
        H1_case2 = -4.48480194 + (9.39456063 + (-6.61487486 + (1.98919585 - 0.22041650 * v) * v) * v) * v
        H1_case3 = ((0.554153432 +
                    (0.278711796 + (-0.1883256872 + (0.042991293 - 0.003278278 * v) * v) * v) * v) /
                    (v2 - 3.0 / 2.0))

        H1 = np.where(v < 1.3, H1_case1, np.where(v < 2.4, H1_case2, H1_case3))
        H2 = (1.0 - 2.0 * v2) * H0

        return H0, H1, H2
    
    def _voigt_hjerting(self, alpha: float, v: float) -> float:
        """
        Voigt-Hjerting function (Korg.jl lines 270-294)
        
        Exact implementation matching Korg.jl with all four regimes:
        1. Small α, large v (asymptotic)
        2. Small α, small v (Harris series)
        3. Intermediate (modified Harris series)
        4. Large α or α + v > 3.2 (general case)
        """
        v2 = v * v
        sqrt_pi = np.sqrt(PI)
        
        if alpha <= 0.2 and v >= 5:
            # Regime 1: Asymptotic case
            inv_v2 = 1.0 / v2
            return (alpha / sqrt_pi * inv_v2) * (1 + 1.5*inv_v2 + 3.75*inv_v2*inv_v2)
            
        elif alpha <= 0.2:  # v < 5
            # Regime 2: Harris series
            H0, H1, H2 = self._harris_series(v)
            return H0 + (H1 + H2 * alpha) * alpha
            
        elif alpha <= 1.4 and alpha + v < 3.2:
            # Regime 3: Modified Harris series
            H0, H1, H2 = self._harris_series(v)
            
            # Modified Harris coefficients (M_i is H'_i in source)
            M0 = H0
            M1 = H1 + 2.0 / sqrt_pi * M0
            M2 = H2 - M0 + 2.0 / sqrt_pi * M1
            M3 = (2.0 / (3.0 * sqrt_pi)) * (1.0 - H2) - (2.0 / 3.0) * v2 * M1 + (2.0 / sqrt_pi) * M2
            M4 = (2.0 / 3.0) * v2 * v2 * M0 - (2.0 / (3.0 * sqrt_pi)) * M1 + (2.0 / sqrt_pi) * M3
            
            # Exact Korg.jl polynomial for ψ
            psi = 0.979895023 + (-0.962846325 + (0.532770573 - 0.122727278 * alpha) * alpha) * alpha
            return psi * (M0 + (M1 + (M2 + (M3 + M4 * alpha) * alpha) * alpha) * alpha)
            
        else:
            # Regime 4: Large α or (α > 0.2 and α + v > 3.2)
            r2 = v2 / (alpha * alpha)
            alpha_invu = 1.0 / (np.sqrt(2) * ((r2 + 1) * alpha))
            alpha2_invu2 = alpha_invu * alpha_invu
            
            return (np.sqrt(2.0 / PI) * alpha_invu * 
                    (1.0 + (3.0 * r2 - 1.0 + ((r2 - 2.0) * 15.0 * r2 + 2.0) * alpha2_invu2) * alpha2_invu2))
    
    def _harris_series(self, v: float) -> Tuple[float, float, float]:
        """
        Harris series coefficients (Korg.jl lines 235-249)
        
        Returns H0, H1, H2 for v < 5
        """
        v2 = v * v
        H0 = np.exp(-v2)
        
        if v < 1.3:
            H1 = -1.12470432 + (-0.15516677 + (3.288675912 + (-2.34357915 + 0.42139162 * v) * v) * v) * v
        elif v < 2.4:
            H1 = -4.48480194 + (9.39456063 + (-6.61487486 + (1.98919585 - 0.22041650 * v) * v) * v) * v
        else:  # v < 5
            H1 = ((0.554153432 + 
                   (0.278711796 + (-0.1883256872 + (0.042991293 - 0.003278278 * v) * v) * v) * v) /
                  (v2 - 3.0 / 2.0))
        
        H2 = (1.0 - 2.0 * v2) * H0
        
        return H0, H1, H2
