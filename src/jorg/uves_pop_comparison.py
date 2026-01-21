"""
UVES-POP Observed vs Synthetic Spectrum Comparison
===================================================

Computes Jorg synthetic spectra for UVES-POP stars and compares with
observed spectra.

Workflow:
1. Load star parameters from UVES-POP catalog
2. Download observed spectrum
3. Compute Jorg synthetic spectrum
4. Interpolate to common wavelength grid
5. Compute comparison metrics (RMS, chi-squared)

Reference:
    - UVES-POP: J_ApJS_266_11
    - Jorg: Python/JAX implementation of Korg.jl synthesis
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .uves_pop_stars import get_all_stars
from .data.uves_pop_spectra import download_and_read, get_spectrum_info


def compare_star(
    star_id: int,
    wavelength_range: Tuple[float, float] = (4800, 6800),
    resolution: float = 20000,
    cache_dir: Optional[Path] = None,
    vmic: float = 1.0,
    verbose: bool = True
) -> Dict:
    """
    Compare observed UVES-POP spectrum with Jorg synthetic spectrum for a single star.

    Parameters
    ----------
    star_id : int
        Star ID from 1 to 10
    wavelength_range : tuple, optional
        Wavelength range (Å) for comparison (default: 4800-6800 Å)
    resolution : float, optional
        Resolution for synthetic spectrum (default: R=20000)
    cache_dir : Path, optional
        Directory for cached spectra
    vmic : float, optional
        Microturbulence (km/s) for synthetic spectrum
    verbose : bool, optional
        Print progress information

    Returns
    -------
    Dict
        Comparison results with keys:
        - star: star parameters
        - observed: observed spectrum (wl, flux)
        - synthetic: synthetic spectrum (wl, flux, continuum)
        - interpolated: synthetic interpolated to observed wavelengths
        - metrics: comparison metrics (RMS, chi2)
    """
    # Import synth here to avoid circular imports
    from jorg.synthesis import synth
    from jorg.lines.linelist_data import get_VALD_solar_linelist

    # Load star parameters
    stars = get_all_stars()
    star = stars[star_id - 1]

    if verbose:
        print(f"\n{'='*60}")
        print(f"Comparing Star {star_id}: {star['name']}")
        print(f"{'='*60}")
        print(f"  Regime: {star['regime']}")
        print(f"  Teff: {star['teff']:.0f} K")
        print(f"  logg: {star['logg']:.2f}")
        print(f"  [Fe/H]: {star['feh']:.2f}")
        print(f"  [α/Fe]: {star['afeh']:.2f}")

    # Download observed spectrum
    if verbose:
        print(f"\n[1/4] Downloading observed spectrum...")

    spec_obs = download_and_read(
        star['filename'],
        resolution='R20k',
        cache_dir=cache_dir
    )
    info_obs = get_spectrum_info(spec_obs)

    if verbose:
        print(f"  Wavelength: {info_obs['wavelength_min']:.1f} - {info_obs['wavelength_max']:.1f} Å")
        print(f"  Pixels: {info_obs['n_pixels']:,}")

    # Compute synthetic spectrum with Jorg
    if verbose:
        print(f"\n[2/4] Computing synthetic spectrum with Jorg...")

    # Convert [Fe/H] and [α/Fe] to m_H and alpha_H
    m_H = star['mh']      # [M/H] ≈ [Fe/H]
    alpha_H = star['alpha_h']  # [α/H] = [Fe/H] + [α/Fe]

    if verbose:
        print(f"  Using m_H={m_H:.2f}, alpha_H={alpha_H:.2f}")

    # Get linelist
    linelist = get_VALD_solar_linelist()

    # Run synthesis
    wl_syn, flux_syn, cntm_syn = synth(
        Teff=star['teff'],
        logg=star['logg'],
        m_H=m_H,
        alpha_H=alpha_H,
        wavelengths=wavelength_range,
        linelist=linelist,
        rectify=True,
        R=resolution,
        vmic=vmic,
        verbose=False
    )

    if verbose:
        print(f"  Synthetic: {len(wl_syn):,} wavelength points")
        print(f"  Flux range: {flux_syn.min():.3f} - {flux_syn.max():.3f}")

    # Match wavelength grids
    if verbose:
        print(f"\n[3/4] Interpolating to observed wavelength grid...")

    from scipy.interpolate import interp1d

    # Create interpolation function (extrapolation=False for safety)
    f = interp1d(wl_syn, flux_syn, kind='linear', bounds_error=False, fill_value=np.nan)

    # Interpolate synthetic spectrum to observed wavelengths
    # Only interpolate within the wavelength range
    wl_obs = spec_obs['wavelength']
    flux_obs = spec_obs['flux']

    # Filter to wavelength range
    mask = (wl_obs >= wavelength_range[0]) & (wl_obs <= wavelength_range[1])
    wl_obs_range = wl_obs[mask]
    flux_obs_range = flux_obs[mask]

    # Interpolate
    flux_syn_interp = f(wl_obs_range)

    # Normalize observed spectrum to match synthetic (pseudo-continuum normalization)
    # Both should be on similar scale for meaningful comparison
    if np.any(np.isfinite(flux_obs_range)):
        # Use percentile-based normalization for observed spectrum
        obs_percentile = np.percentile(flux_obs_range[np.isfinite(flux_obs_range)], 99)
        if obs_percentile > 0:
            flux_obs_normalized = flux_obs_range / obs_percentile
            # Also normalize synthetic to match scale
            syn_percentile = np.percentile(flux_syn_interp[np.isfinite(flux_syn_interp)], 99)
            if syn_percentile > 0:
                flux_syn_interp = flux_syn_interp / syn_percentile
        else:
            flux_obs_normalized = flux_obs_range
    else:
        flux_obs_normalized = flux_obs_range

    # Remove NaN values from interpolation
    finite = np.isfinite(flux_syn_interp) & np.isfinite(flux_obs_normalized)

    if verbose:
        n_valid = finite.sum()
        print(f"  Valid pixels: {n_valid:,} / {len(wl_obs_range):,}")

    # Compute metrics
    if verbose:
        print(f"\n[4/4] Computing comparison metrics...")

    if n_valid > 0:
        residuals = flux_obs_normalized[finite] - flux_syn_interp[finite]

        rms = np.sqrt(np.mean(residuals**2))
        median_residual = np.median(residuals)

        # Normalize by flux level for relative error
        flux_level = np.median(flux_obs_normalized[finite])
        relative_rms = rms / flux_level if flux_level > 0 else np.nan

        if verbose:
            print(f"  RMS residual: {rms:.3f}")
            print(f"  Median residual: {median_residual:.3f}")
            print(f"  Relative RMS: {relative_rms*100:.1f}%")
    else:
        rms = np.nan
        median_residual = np.nan
        relative_rms = np.nan

        if verbose:
            print("  Warning: No valid pixels for comparison!")

    return {
        'star': star,
        'observed': {
            'wavelength': wl_obs_range,
            'flux': flux_obs_range,
            'flux_normalized': flux_obs_normalized,
            'info': info_obs
        },
        'synthetic': {
            'wavelength': wl_syn,
            'flux': flux_syn,
            'continuum': cntm_syn
        },
        'interpolated': {
            'wavelength': wl_obs_range,
            'flux': flux_syn_interp,
            'flux_obs': flux_obs_normalized,
            'finite_mask': finite
        },
        'metrics': {
            'rms': rms,
            'median_residual': median_residual,
            'relative_rms': relative_rms,
            'n_valid': int(finite.sum()) if finite is not None else 0
        }
    }


def compare_all_stars(
    star_ids: Optional[List[int]] = None,
    wavelength_range: Tuple[float, float] = (4800, 6800),
    resolution: float = 20000,
    cache_dir: Optional[Path] = None,
    vmic: float = 1.0,
    verbose: bool = True
) -> List[Dict]:
    """
    Compare multiple UVES-POP stars with synthetic spectra.

    Parameters
    ----------
    star_ids : list of int, optional
        Star IDs to compare (default: all 10)
    wavelength_range : tuple, optional
        Wavelength range (Å) for comparison
    resolution : float, optional
        Resolution for synthetic spectrum
    cache_dir : Path, optional
        Directory for cached spectra
    vmic : float, optional
        Microturbulence (km/s)
    verbose : bool, optional
        Print progress information

    Returns
    -------
    List[Dict]
        List of comparison results for each star
    """
    if star_ids is None:
        star_ids = list(range(1, 11))

    results = []

    for star_id in star_ids:
        try:
            result = compare_star(
                star_id=star_id,
                wavelength_range=wavelength_range,
                resolution=resolution,
                cache_dir=cache_dir,
                vmic=vmic,
                verbose=verbose
            )
            results.append(result)
        except Exception as e:
            print(f"\nWarning: Failed to compare star {star_id}: {e}")
            import traceback
            traceback.print_exc()

    return results


def summarize_comparison_results(results: List[Dict]):
    """
    Print a summary of comparison results.

    Parameters
    ----------
    results : list of dict
        Results from compare_all_stars()
    """
    print("\n" + "="*70)
    print("COMPARISON SUMMARY: Observed vs Synthetic Spectra")
    print("="*70)

    print(f"\n{'Star':>15} {'Teff':>7} {'logg':>6} {'[Fe/H]':>7} {'RMS':>7} {'Rel RMS':>9}")
    print("-"*70)

    for r in results:
        star = r['star']
        metrics = r['metrics']

        print(f"{star['name']:>15} {star['teff']:>7.0f} {star['logg']:>6.2f} {star['feh']:>7.2f} "
              f"{metrics['rms']:>7.3f} {metrics['relative_rms']*100:>8.1f}%")

    print("-"*70)

    # Calculate statistics
    valid_results = [r for r in results if np.isfinite(r['metrics']['rms'])]

    if valid_results:
        mean_rms = np.mean([r['metrics']['rms'] for r in valid_results])
        mean_rel_rms = np.mean([r['metrics']['relative_rms'] for r in valid_results])

        print(f"\nOverall statistics ({len(valid_results)} stars):")
        print(f"  Mean RMS: {mean_rms:.3f}")
        print(f"  Mean relative RMS: {mean_rel_rms*100:.1f}%")


if __name__ == "__main__":
    # Test: compare one star
    print("UVES-POP Comparison Module")
    print("=" * 50)

    result = compare_star(
        star_id=3,  # Solar analog HD 59468
        wavelength_range=(5000, 6000),
        resolution=20000,
        verbose=True
    )

    print(f"\nResult keys: {list(result.keys())}")
