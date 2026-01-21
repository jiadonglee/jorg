"""
UVES-POP Spectrum Download and Reading Utilities
================================================

Provides functions to download and read UVES-POP stellar spectra from the
online archive.

UVES-POP spectra are high-resolution (R=80,000) flux-calibrated stellar
spectra covering 320-1025 nm (optical to near-infrared).

Reference:
    - Base URL: https://data.voxastro.org/uves-pop/model_spec/fit_res/v221115/
    - Format: FITS files (gzip compressed)
    - Resolution: R=20,000 (low-res) and R=80,000 (high-res)
"""

import gzip
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from astropy.io import fits


# Base URL for UVES-POP spectra
BASE_URL_R20 = "https://data.voxastro.org/uves-pop/model_spec/fit_res/v221115/"
BASE_URL_R80 = "https://data.voxastro.org/uves-pop/model_spec/merged_221115/"


def download_spectrum(
    filename: str,
    resolution: str = "R20k",
    output_dir: Optional[Path] = None,
    overwrite: bool = False
) -> Path:
    """
    Download a UVES-POP spectrum FITS file.

    Parameters
    ----------
    filename : str
        FITS filename (e.g., "Arcturus.fits", "HD59468.fits")
    resolution : str, optional
        Resolution of spectrum: "R20k" (R=20,000) or "R80k" (R=80,000)
    output_dir : Path, optional
        Directory to save downloaded files. Defaults to "uves_pop_spectra" in data dir.
    overwrite : bool, optional
        Overwrite existing file if True

    Returns
    -------
    Path
        Path to downloaded file

    Raises
    ------
    ValueError
        If resolution is not "R20k" or "R80k"
    urllib.error.URLError
        If download fails
    """
    if resolution == "R20k":
        base_url = BASE_URL_R20
        url = f"{base_url}{filename}.gz"
    elif resolution == "R80k":
        base_url = BASE_URL_R80
        url = f"{base_url}{filename.replace('.fits', '_R80k.fits')}.gz"
    else:
        raise ValueError(f"resolution must be 'R20k' or 'R80k', got {resolution}")

    if output_dir is None:
        output_dir = Path.cwd() / "uves_pop_spectra"

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    # Check if file exists
    if output_path.exists() and not overwrite:
        print(f"File already exists: {output_path}")
        return output_path

    # Download with progress bar
    print(f"Downloading {url}...")
    print(f"Saving to {output_path}...")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 / total_size) if total_size > 0 else 0
        print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, output_path, reporthook=progress_hook)
    print()  # New line after progress

    return output_path


def read_spectrum(
    filepath: Path,
    extract_from_gz: bool = True,
    verify_fix: bool = True
) -> Dict[str, np.ndarray]:
    """
    Read a UVES-POP spectrum FITS file.

    Parameters
    ----------
    filepath : Path
        Path to FITS file (can be .gz compressed)
    extract_from_gz : bool, optional
        If True, read from .gz file directly
    verify_fix : bool, optional
        If True, fix FITS header issues automatically

    Returns
    -------
    Dict
        Dictionary with keys:
        - wavelength: wavelength array (Å)
        - flux: flux array (erg/s/cm²/Å or normalized)
        - header: FITS header dictionary

    Raises
    ------
    IOError
        If file cannot be read
    ValueError
        If FITS format is unrecognized
    """
    filepath = Path(filepath)

    # Handle gzip compressed files
    if str(filepath).endswith('.gz'):
        if extract_from_gz:
            # Read directly from gzip
            with gzip.open(filepath, 'rb') as gz:
                mode = 'fix' if verify_fix else 'exception'
                with fits.open(gz) as hdul:
                    hdul.verify(mode)
                    return _parse_fits(hdul)
        else:
            # Extract first
            import shutil
            unzipped = filepath.with_suffix('')  # Remove .gz
            if not unzipped.exists():
                print(f"Extracting {filepath}...")
                with gzip.open(filepath, 'rb') as f_in:
                    with open(unzipped, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            filepath = unzipped

    # Read regular FITS
    mode = 'fix' if verify_fix else 'exception'
    with fits.open(filepath) as hdul:
        hdul.verify(mode)
        return _parse_fits(hdul)


def _parse_fits(hdul: fits.HDUList) -> Dict[str, np.ndarray]:
    """
    Parse UVES-POP FITS HDU list to extract wavelength and flux.

    UVES-POP FITS format varies. This function tries multiple common formats:
    1. Binary table with 'WAVELENGTH' and 'FLUX' columns
    2. Primary HDU with NAXIS1 data
    3. Multiple extensions

    Parameters
    ----------
    hdul : fits.HDUList
        FITS HDU list from astropy

    Returns
    -------
    Dict
        Dictionary with wavelength, flux, and header
    """
    # Try to read header first (handle unparsable cards)
    header = {}
    for key in hdul[0].header.keys():
        try:
            header[key] = hdul[0].header[key]
        except:
            # Skip unparsable cards
            header[key] = None

    # Method 1: Binary table extension (most common for UVES-POP)
    for hdu_idx in range(len(hdul)):
        if hasattr(hdul[hdu_idx], 'data') and hdul[hdu_idx].data is not None:
            data = hdul[hdu_idx].data

            # Check if it's a FITS recarray (table) with columns
            if hasattr(data, 'columns') and hasattr(data.columns, 'names'):
                colnames = data.columns.names

                # Look for wavelength and flux columns
                wl_col = None
                flux_col = None

                # Prioritize exact matches for wavelength
                for col in colnames:
                    col_upper = col.upper()
                    if col_upper == 'WAVELENGTH' or col_upper == 'WAVE':
                        wl_col = col
                        break  # Exact match, use it

                # If no exact match, look for partial matches
                if wl_col is None:
                    for col in colnames:
                        col_upper = col.upper()
                        # Match WAVE or WAVELENGTH but exclude WLR, SWLVEC, etc.
                        if ('WAVE' in col_upper or 'LAMBDA' in col_upper) and col_upper not in ['WLR', 'SWLVEC', 'LSFVEC']:
                            wl_col = col
                            break

                for col in colnames:
                    col_upper = col.upper()
                    if col_upper == 'FLUX':
                        flux_col = col
                        break  # Exact match

                if flux_col is None:
                    for col in colnames:
                        col_upper = col.upper()
                        if 'FLUX' in col_upper or 'SPECTRUM' in col_upper:
                            flux_col = col
                            break

                if wl_col is not None and flux_col is not None:
                    # Handle single-row table (UVES-POP format)
                    if data.shape == (1,):
                        # Access the first row, then the column
                        wavelength = np.asarray(data[0][wl_col]).flatten()
                        flux = np.asarray(data[0][flux_col]).flatten()
                    else:
                        wavelength = np.asarray(data[wl_col]).flatten()
                        flux = np.asarray(data[flux_col]).flatten()

                    # Use the header from this HDU
                    hdu_header = {}
                    for key in hdul[hdu_idx].header.keys():
                        try:
                            hdu_header[key] = hdul[hdu_idx].header[key]
                        except:
                            hdu_header[key] = None

                    return {
                        'wavelength': wavelength,
                        'flux': flux,
                        'header': hdu_header
                    }

    # Method 2: Image data in primary HDU or extension
    for hdu_idx in [0, 1]:
        if hdu_idx < len(hdul) and hasattr(hdul[hdu_idx], 'data') and hdul[hdu_idx].data is not None:
            data = np.asarray(hdul[hdu_idx].data)

            # Skip if data is too small
            if data.size < 100:
                continue

            # Check if data is 2D with wavelength in first axis
            if data.ndim >= 1:
                # Try to get wavelength from header CRVAL1/CDELT1
                if 'CRVAL1' in header and 'CDELT1' in header:
                    crval1 = header['CRVAL1']
                    cdelt1 = header['CDELT1']
                    naxis1 = header.get('NAXIS1', data.shape[0])
                    wavelength = crval1 + cdelt1 * np.arange(naxis1)
                else:
                    # No wavelength info - use indices
                    wavelength = np.arange(data.shape[0])

                # Extract flux (handle 2D arrays)
                if data.ndim == 1:
                    flux = data
                elif data.ndim == 2:
                    # If first dimension is small, assume it's the flux axis
                    if data.shape[0] == 1:
                        flux = data[0]
                    elif data.shape[1] == 1:
                        flux = data[:, 0]
                    else:
                        # Assume flux is in the longer dimension
                        flux = data[0] if data.shape[0] < data.shape[1] else data[:, 0]
                else:
                    flux = data.flatten()

                return {
                    'wavelength': wavelength,
                    'flux': flux,
                    'header': header
                }

    # Method 3: Last resort - check all extensions
    for i, hdu in enumerate(hdul):
        if hasattr(hdu, 'data') and hdu.data is not None:
            try:
                data = np.asarray(hdu.data)
                if data.ndim == 1 and len(data) > 100:
                    # Assume it's flux
                    return {
                        'wavelength': np.arange(len(data)),
                        'flux': data,
                        'header': dict(hdu.header)
                    }
            except:
                continue

    raise ValueError(f"Could not parse FITS format. Available extensions: {len(hdul)}")


def download_and_read(
    filename: str,
    resolution: str = "R20k",
    cache_dir: Optional[Path] = None,
    overwrite: bool = False
) -> Dict[str, np.ndarray]:
    """
    Download and read a UVES-POP spectrum in one step.

    Parameters
    ----------
    filename : str
        FITS filename (e.g., "Arcturus.fits")
    resolution : str, optional
        Resolution: "R20k" or "R80k"
    cache_dir : Path, optional
        Directory for cached files
    overwrite : bool, optional
        Re-download even if cached

    Returns
    -------
    Dict
        Dictionary with wavelength, flux, and header
    """
    filepath = download_spectrum(filename, resolution, cache_dir, overwrite)
    return read_spectrum(filepath)


def get_spectrum_info(spec_dict: Dict) -> Dict:
    """
    Get basic information about a spectrum.

    Parameters
    ----------
    spec_dict : Dict
        Spectrum dictionary from read_spectrum()

    Returns
    -------
    Dict
        Information about wavelength range, flux range, etc.
    """
    wl = spec_dict['wavelength']
    flux = spec_dict['flux']

    # Handle NaN values in flux
    finite_mask = np.isfinite(flux)
    flux_finite = flux[finite_mask]

    if len(flux_finite) == 0:
        flux_min = float('nan')
        flux_max = float('nan')
        flux_median = float('nan')
        flux_std = float('nan')
    else:
        flux_min = float(flux_finite.min())
        flux_max = float(flux_finite.max())
        flux_median = float(np.median(flux_finite))
        flux_std = float(flux_finite.std())

    # Get resolution from header or data
    resolution = spec_dict['header'].get('RESOLUT', spec_dict['header'].get('RESOLUTION', 'unknown'))

    return {
        'wavelength_min': float(wl.min()),
        'wavelength_max': float(wl.max()),
        'wavelength_unit': spec_dict['header'].get('WAVEUNIT', 'Å'),
        'n_pixels': len(wl),
        'flux_min': flux_min,
        'flux_max': flux_max,
        'flux_median': flux_median,
        'flux_std': flux_std,
        'n_finite_pixels': len(flux_finite),
        'resolution': resolution,
    }


if __name__ == "__main__":
    # Test: download one spectrum
    print("UVES-POP Spectrum Utilities")
    print("=" * 50)

    # Example: Download Arcturus spectrum
    test_file = "Arcturus.fits"
    print(f"Testing download of {test_file}...")

    try:
        spec = download_and_read(test_file, resolution="R20k")
        info = get_spectrum_info(spec)

        print("\nSpectrum Info:")
        print(f"  Wavelength range: {info['wavelength_min']:.1f} - {info['wavelength_max']:.1f}")
        print(f"  Pixels: {info['n_pixels']}")
        print(f"  Flux range: {info['flux_min']:.2e} - {info['flux_max']:.2e}")
        print(f"  Flux median: {info['flux_median']:.2e}")

    except Exception as e:
        print(f"Download test failed: {e}")
        print("(This is expected if running without internet)")
