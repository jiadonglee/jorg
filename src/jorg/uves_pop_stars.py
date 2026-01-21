"""
UVES-POP Stellar Sample for Jorg Synthetic Spectrum Comparison
================================================================

Provides a curated sample of 10 representative stars from the UVES-POP catalog
covering the full stellar parameter space (Teff, logg, [Fe/H], [α/Fe]).

Reference:
    - UVES-POP catalog: J_ApJS_266_11_table6.dat.fits
    - Base URL: https://data.voxastro.org/uves-pop/model_spec/fit_res/v221115/

Selected Stars:
    - Teff range: 3,728 - 18,859 K
    - logg range: 1.85 - 4.95
    - [Fe/H] range: -2.07 - +0.52
    - [α/Fe] range: -0.22 - +0.68
"""

import json
from pathlib import Path
from typing import Dict, List, Optional


# Package data directory
# __file__ is jorg/src/jorg/uves_pop_stars.py
# We want jorg/valid/ which is at the same level as jorg/src/
SRC_DIR = Path(__file__).parent.parent
PACKAGE_DIR = SRC_DIR.parent
VALID_DIR = PACKAGE_DIR / "valid"
DATA_DIR = PACKAGE_DIR / "data"


# Default star selection file
DEFAULT_STARS_FILE = VALID_DIR / "selected_stars.json"


def load_selected_stars(stars_file: Optional[Path] = None) -> List[Dict]:
    """
    Load the selected UVES-POP stars from JSON file.

    Parameters
    ----------
    stars_file : Path, optional
        Path to stars JSON file. Defaults to selected_stars.json in jorg/valid/.

    Returns
    -------
    List[Dict]
        List of star dictionaries with keys:
        - id: star index (1-10)
        - regime: stellar type description
        - name: star name
        - hd: HD number
        - teff: effective temperature (K)
        - logg: surface gravity
        - feh: [Fe/H] metallicity
        - afeh: [α/Fe] alpha enhancement
        - mh: [M/H] total metallicity (≈ [Fe/H])
        - alpha_h: [α/H] alpha abundance (= [Fe/H] + [α/Fe])
        - url_r20: URL to R=20,000 spectrum
        - filename: FITS filename
    """
    if stars_file is None:
        stars_file = DEFAULT_STARS_FILE

    with open(stars_file, 'r') as f:
        data = json.load(f)

    return data['stars']


def get_star_by_id(star_id: int, stars_file: Optional[Path] = None) -> Dict:
    """
    Get a specific star by ID (1-10).

    Parameters
    ----------
    star_id : int
        Star ID from 1 to 10
    stars_file : Path, optional
        Path to stars JSON file

    Returns
    -------
    Dict
        Star parameter dictionary

    Raises
    ------
    ValueError
        If star_id is not between 1 and 10
    """
    if not 1 <= star_id <= 10:
        raise ValueError(f"star_id must be between 1 and 10, got {star_id}")

    stars = load_selected_stars(stars_file)
    return stars[star_id - 1]


def get_star_by_name(name: str, stars_file: Optional[Path] = None) -> Optional[Dict]:
    """
    Find a star by name (partial match allowed).

    Parameters
    ----------
    name : str
        Star name or partial name (e.g., "Arcturus", "HD 59468")
    stars_file : Path, optional
        Path to stars JSON file

    Returns
    -------
    Dict or None
        Star parameter dictionary if found, None otherwise
    """
    stars = load_selected_stars(stars_file)
    name_lower = name.lower()

    for star in stars:
        if name_lower in star['name'].lower() or name_lower == star['hd'].lower():
            return star

    return None


def get_stars_by_regime(regime: str, stars_file: Optional[Path] = None) -> List[Dict]:
    """
    Get all stars matching a stellar regime.

    Parameters
    ----------
    regime : str
        Regime name (e.g., "Cool M-dwarf", "Solar analog", "Alpha-enhanced giant")
    stars_file : Path, optional
        Path to stars JSON file

    Returns
    -------
    List[Dict]
        List of matching stars
    """
    stars = load_selected_stars(stars_file)
    regime_lower = regime.lower()

    return [s for s in stars if regime_lower in s['regime'].lower()]


def get_parameter_coverage(stars_file: Optional[Path] = None) -> Dict:
    """
    Get the parameter coverage of the selected star sample.

    Parameters
    ----------
    stars_file : Path, optional
        Path to stars JSON file

    Returns
    -------
    Dict
        Dictionary with min/max values for Teff, logg, [Fe/H], [α/Fe]
    """
    stars_file = stars_file or DEFAULT_STARS_FILE

    with open(stars_file, 'r') as f:
        data = json.load(f)

    return data['parameter_coverage']


def summarize_sample(stars_file: Optional[Path] = None):
    """
    Print a summary of the selected star sample.

    Parameters
    ----------
    stars_file : Path, optional
        Path to stars JSON file
    """
    stars = load_selected_stars(stars_file)
    coverage = get_parameter_coverage(stars_file)

    print("UVES-POP Stellar Sample for Synthetic Spectrum Comparison")
    print("=" * 65)
    print(f"Number of stars: {len(stars)}")
    print()
    print("Parameter Coverage:")
    print(f"  Teff:  {coverage['teff_min']:.0f} - {coverage['teff_max']:.0f} K")
    print(f"  logg:  {coverage['logg_min']:.2f} - {coverage['logg_max']:.2f}")
    print(f"  [Fe/H]: {coverage['feh_min']:.2f} - {coverage['feh_max']:.2f}")
    print(f"  [α/Fe]: {coverage['afeh_min']:.2f} - {coverage['afeh_max']:.2f}")
    print()
    print("Selected Stars:")
    print("-" * 65)
    print(f"{'#':>3} {'Regime':>25} {'Name':>15} {'Teff':>7} {'logg':>6} {'[Fe/H]':>7} {'[α/Fe]':>7}")
    print("-" * 65)

    for star in stars:
        print(f"{star['id']:>3} {star['regime']:>25} {star['name']:>15} "
              f"{star['teff']:>7.0f} {star['logg']:>6.2f} {star['feh']:>7.2f} {star['afeh']:>7.2f}")


# Convenience: Pre-loaded star list for quick access
_selected_stars_cache = None


def get_all_stars() -> List[Dict]:
    """
    Get all selected stars (cached for performance).

    Returns
    -------
    List[Dict]
        List of all 10 star parameter dictionaries
    """
    global _selected_stars_cache

    if _selected_stars_cache is None:
        _selected_stars_cache = load_selected_stars()

    return _selected_stars_cache


if __name__ == "__main__":
    # Print summary when run directly
    summarize_sample()
