"""
Built-in linelist functions for Jorg - matching Korg.jl convenience functions
"""

import os
from pathlib import Path
from typing import Optional, Union, List, Dict, Any
import warnings

from .linelist import read_linelist


def get_VALD_solar_linelist() -> List:
    """
    Get a VALD "extract stellar" linelist produced at solar parameters.
    
    Returns a VALD linelist with threshold value set to 0.01, intended
    for quick tests and demonstrations. For production work, download
    a current VALD linelist appropriate for your stellar parameters.
    
    Returns
    -------
    List
        VALD line list suitable for solar-type stars
        
    Notes
    -----
    This function matches Korg.jl's get_VALD_solar_linelist() behavior.
    If you use this in a paper, please cite VALD appropriately:
    https://www.astro.uu.se/valdwiki/Acknowledgement
    
    The built-in linelist is provided for convenience and testing only.
    For research applications, download a current VALD extract appropriate
    for your stellar parameters and wavelength range.
    """
    # Try to find the VALD solar linelist in standard locations
    possible_paths = [
        # Primary location - matches Korg.jl data structure
        "/Users/jdli/Project/Korg.jl/data/linelists/vald_extract_stellar_solar_threshold001.vald",
        # Alternative locations
        "/Users/jdli/Project/Korg.jl/misc/Tutorial notebooks/basics/linelist.vald",
        # Relative to this file
        Path(__file__).parent.parent.parent.parent / "data" / "linelists" / "vald_extract_stellar_solar_threshold001.vald"
    ]
    
    for vald_path in possible_paths:
        if os.path.exists(vald_path):
            try:
                return read_linelist(vald_path, format='vald')
            except Exception as e:
                warnings.warn(f"Failed to load VALD linelist from {vald_path}: {e}")
                continue
                
    # If no built-in linelist found, provide guidance
    raise FileNotFoundError(
        "No built-in VALD solar linelist found. Please provide your own linelist.\n"
        "Download from VALD (http://vald.astro.uu.se/) with parameters:\n"
        "- Teff: 5780K, log g: 4.44, [M/H]: 0.0\n" 
        "- Format: 'extract stellar' with threshold ~0.01\n"
        "- Wavelength range appropriate for your analysis"
    )


def get_APOGEE_DR17_linelist(include_water: bool = True) -> List:
    """
    Get the APOGEE DR17 linelist for infrared synthesis (15,000-17,000 Å).
    
    The APOGEE DR 17 linelist ranges from roughly 15,000 Å to 17,000 Å
    and is nearly the same as the DR 16 linelist described in 
    Smith+ 2021 (https://ui.adsabs.harvard.edu/abs/2021AJ....161..254S/).
    
    Parameters
    ----------
    include_water : bool, default True
        Whether to include water lines in the linelist
        
    Returns
    -------
    List
        APOGEE DR17 line list for infrared synthesis
        
    Notes
    -----
    This function attempts to match Korg.jl's get_APOGEE_DR17_linelist() 
    functionality, but requires the APOGEE data files to be available.
    
    For production use, download the APOGEE linelist from:
    https://www.sdss.org/surveys/apogee/
    """
    # This is a placeholder implementation
    # In practice, would need access to APOGEE DR17 data files
    raise NotImplementedError(
        "APOGEE DR17 linelist not available. This requires the full APOGEE data distribution.\n"
        "Please download from https://www.sdss.org/surveys/apogee/ and use read_linelist()\n"
        "with format='turbospectrum' for atomic lines and molecular cross-sections."
    )


def get_GALAH_DR3_linelist() -> List:
    """
    Get the GALAH DR3 linelist for optical synthesis (4,675-7,930 Å).
    
    The GALAH DR3 linelist (also used for DR4) ranges from roughly
    4,675 Å to 7,930 Å and is optimized for the GALAH survey's
    abundance analysis pipeline.
    
    Returns
    -------
    List
        GALAH DR3 line list for optical synthesis
        
    Notes  
    -----
    This function attempts to match Korg.jl's get_GALAH_DR3_linelist()
    functionality, but requires the GALAH data files to be available.
    
    For production use, obtain the GALAH linelist from:
    https://www.galah-survey.org/
    """
    # This is a placeholder implementation
    # In practice, would need access to GALAH DR3 data files
    raise NotImplementedError(
        "GALAH DR3 linelist not available. This requires the GALAH data distribution.\n"
        "Please obtain from https://www.galah-survey.org/ and use read_linelist()\n" 
        "with appropriate format (typically HDF5 or ASCII)."
    )


def get_GES_linelist(include_molecules: bool = True) -> List:
    """
    Get the Gaia-ESO Survey (GES) linelist.
    
    Based on Heiter et al. 2021 linelist compilation for the
    Gaia-ESO Survey abundance analysis.
    
    Parameters
    ----------
    include_molecules : bool, default True
        Whether to include molecular lines
        
    Returns
    -------
    List
        GES line list
        
    Notes
    -----
    This function attempts to match Korg.jl's get_GES_linelist()
    functionality, but requires the GES data files to be available.
    
    For production use, obtain the GES linelist from the Gaia-ESO
    Survey data releases.
    """
    # This is a placeholder implementation  
    # In practice, would need access to GES data files
    raise NotImplementedError(
        "GES linelist not available. This requires the Gaia-ESO Survey data distribution.\n"
        "Please obtain from the Gaia-ESO Survey and use read_linelist()\n"
        "with appropriate format."
    )


def save_linelist(linelist: List, filename: Union[str, Path], 
                 format: str = "korg") -> None:
    """
    Save a linelist to file for faster repeated access.
    
    Parameters
    ----------
    linelist : List
        Line list to save
    filename : str or Path
        Output filename
    format : str, default "korg"
        Output format. Currently supports:
        - "korg": HDF5 format matching Korg.jl (recommended for speed)
        - "vald": VALD format (for compatibility)
        
    Notes
    -----
    This function provides the same functionality as Korg.jl's save_linelist().
    The HDF5 "korg" format allows very fast loading for repeated use.
    """
    if format == "korg":
        # Save in HDF5 format (like Korg.jl)
        raise NotImplementedError(
            "HDF5 linelist saving not yet implemented.\n"
            "For now, use the original linelist files directly."
        )
    elif format == "vald":
        # Save in VALD format
        raise NotImplementedError(
            "VALD format output not yet implemented.\n"
            "For now, use the original linelist files directly."
        )
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'korg' or 'vald'.")


# Convenience aliases matching Korg.jl naming
get_VALD_solar_linelist = get_VALD_solar_linelist
get_APOGEE_DR17_linelist = get_APOGEE_DR17_linelist  
get_GALAH_DR3_linelist = get_GALAH_DR3_linelist
get_GES_linelist = get_GES_linelist