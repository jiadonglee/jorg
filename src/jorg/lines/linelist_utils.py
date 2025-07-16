"""
Linelist utilities for advanced line management and processing.

This module provides utilities for merging, filtering, and managing linelists
following Korg.jl conventions.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import warnings
from collections import defaultdict

from .datatypes import LineData, Line, create_line, species_from_integer
from .linelist import LineList
from .atomic_data import get_atomic_symbol, format_species_name, get_atomic_mass
from .broadening_korg import approximate_line_strength, get_default_broadening_parameters


def merge_linelists(linelists: List[LineList], 
                   remove_duplicates: bool = True,
                   wavelength_tolerance: float = 0.001e-8) -> LineList:
    """
    Merge multiple linelists into a single linelist.
    
    Parameters
    ----------
    linelists : List[LineList]
        List of LineList objects to merge
    remove_duplicates : bool
        Remove duplicate lines based on wavelength and species
    wavelength_tolerance : float
        Tolerance for considering lines as duplicates (cm)
        
    Returns
    -------
    LineList
        Merged linelist
    """
    if not linelists:
        return LineList([])
    
    # Combine all lines
    all_lines = []
    combined_metadata = {}
    
    for linelist in linelists:
        all_lines.extend(linelist.lines)
        # Combine metadata
        for key, value in linelist.metadata.items():
            if key not in combined_metadata:
                combined_metadata[key] = value
            elif key == 'n_lines':
                combined_metadata[key] = combined_metadata.get(key, 0) + value
    
    # Remove duplicates if requested
    if remove_duplicates:
        all_lines = remove_duplicate_lines(all_lines, wavelength_tolerance)
    
    # Sort by wavelength
    all_lines.sort(key=lambda x: x.wavelength)
    
    # Update metadata
    combined_metadata['n_lines'] = len(all_lines)
    combined_metadata['merged_from'] = len(linelists)
    
    return LineList(all_lines, combined_metadata)


def remove_duplicate_lines(lines: List[LineData], 
                          wavelength_tolerance: float = 0.001e-8) -> List[LineData]:
    """
    Remove duplicate lines based on wavelength and species.
    
    Parameters
    ----------
    lines : List[LineData]
        List of lines to deduplicate
    wavelength_tolerance : float
        Tolerance for considering lines as duplicates (cm)
        
    Returns
    -------
    List[LineData]
        List with duplicates removed
    """
    if not lines:
        return lines
    
    # Group lines by species
    species_groups = defaultdict(list)
    for line in lines:
        species_groups[line.species].append(line)
    
    unique_lines = []
    
    for species_id, species_lines in species_groups.items():
        # Sort by wavelength
        species_lines.sort(key=lambda x: x.wavelength)
        
        # Remove duplicates within this species
        if not species_lines:
            continue
            
        unique_species_lines = [species_lines[0]]
        
        for line in species_lines[1:]:
            # Check if this line is a duplicate of the previous one
            prev_line = unique_species_lines[-1]
            if abs(line.wavelength - prev_line.wavelength) > wavelength_tolerance:
                unique_species_lines.append(line)
            else:
                # Duplicate found - keep the one with higher log_gf
                if line.log_gf > prev_line.log_gf:
                    unique_species_lines[-1] = line
        
        unique_lines.extend(unique_species_lines)
    
    return unique_lines


def filter_lines_by_strength(lines: List[LineData], 
                           temperature: float = 5000.0,
                           log_strength_threshold: float = -5.0) -> List[LineData]:
    """
    Filter lines by approximate line strength at given temperature.
    
    Parameters
    ----------
    lines : List[LineData]
        List of lines to filter
    temperature : float
        Temperature for line strength calculation (K)
    log_strength_threshold : float
        Minimum log₁₀(line strength) to keep
        
    Returns
    -------
    List[LineData]
        Filtered list of lines
    """
    filtered_lines = []
    
    for line in lines:
        # Calculate approximate line strength
        line_strength = approximate_line_strength(line.wavelength, line.log_gf, 
                                                line.E_lower, temperature)
        
        if line_strength >= log_strength_threshold:
            filtered_lines.append(line)
    
    return filtered_lines


def filter_lines_by_species(lines: List[LineData], 
                           species_list: List[Union[int, str]],
                           exclude: bool = False) -> List[LineData]:
    """
    Filter lines by species.
    
    Parameters
    ----------
    lines : List[LineData]
        List of lines to filter
    species_list : List[Union[int, str]]
        List of species to include/exclude (can be species IDs or names)
    exclude : bool
        If True, exclude the specified species; if False, include only these species
        
    Returns
    -------
    List[LineData]
        Filtered list of lines
    """
    # Convert species names to IDs if necessary
    species_ids = set()
    for species in species_list:
        if isinstance(species, str):
            # Parse species name to get ID
            from .species import parse_species
            species_ids.add(parse_species(species))
        else:
            species_ids.add(species)
    
    filtered_lines = []
    
    for line in lines:
        if exclude:
            # Exclude specified species
            if line.species not in species_ids:
                filtered_lines.append(line)
        else:
            # Include only specified species
            if line.species in species_ids:
                filtered_lines.append(line)
    
    return filtered_lines


def prune_weak_lines(lines: List[LineData], 
                    log_gf_threshold: float = -6.0,
                    wavelength_range: Optional[Tuple[float, float]] = None) -> List[LineData]:
    """
    Remove weak lines and optionally filter by wavelength range.
    
    Parameters
    ----------
    lines : List[LineData]
        List of lines to prune
    log_gf_threshold : float
        Minimum log(gf) to keep
    wavelength_range : Tuple[float, float], optional
        Wavelength range to keep (min, max) in cm
        
    Returns
    -------
    List[LineData]
        Pruned list of lines
    """
    pruned_lines = []
    
    for line in lines:
        # Check log_gf threshold
        if line.log_gf < log_gf_threshold:
            continue
            
        # Check wavelength range if specified
        if wavelength_range is not None:
            wl_min, wl_max = wavelength_range
            if line.wavelength < wl_min or line.wavelength > wl_max:
                continue
        
        pruned_lines.append(line)
    
    return pruned_lines


def get_linelist_statistics(linelist: LineList) -> Dict:
    """
    Get comprehensive statistics about a linelist.
    
    Parameters
    ----------
    linelist : LineList
        Linelist to analyze
        
    Returns
    -------
    Dict
        Dictionary with statistics
    """
    if not linelist.lines:
        return {'n_lines': 0, 'species_count': 0, 'wavelength_range': (0, 0)}
    
    # Basic statistics
    n_lines = len(linelist.lines)
    wavelengths = [line.wavelength for line in linelist.lines]
    wl_min, wl_max = min(wavelengths), max(wavelengths)
    
    # Species statistics
    species_counts = defaultdict(int)
    log_gf_values = []
    E_lower_values = []
    
    for line in linelist.lines:
        species_counts[line.species] += 1
        log_gf_values.append(line.log_gf)
        E_lower_values.append(line.E_lower)
    
    # Get species names
    species_names = {}
    for species_id in species_counts.keys():
        try:
            species_obj = species_from_integer(species_id)
            species_names[species_id] = str(species_obj)
        except:
            species_names[species_id] = f"Species_{species_id}"
    
    # Most common species
    most_common_species = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    stats = {
        'n_lines': n_lines,
        'species_count': len(species_counts),
        'wavelength_range_cm': (wl_min, wl_max),
        'wavelength_range_angstrom': (wl_min * 1e8, wl_max * 1e8),
        'log_gf_range': (min(log_gf_values), max(log_gf_values)),
        'E_lower_range': (min(E_lower_values), max(E_lower_values)),
        'most_common_species': [(species_names[sid], count) for sid, count in most_common_species],
        'species_distribution': {species_names[sid]: count for sid, count in species_counts.items()},
        'metadata': linelist.metadata
    }
    
    return stats


def print_linelist_summary(linelist: LineList):
    """
    Print a human-readable summary of a linelist.
    
    Parameters
    ----------
    linelist : LineList
        Linelist to summarize
    """
    stats = get_linelist_statistics(linelist)
    
    print(f"📊 LINELIST SUMMARY")
    print(f"{'='*50}")
    print(f"Total lines: {stats['n_lines']:,}")
    print(f"Species count: {stats['species_count']}")
    print(f"Wavelength range: {stats['wavelength_range_angstrom'][0]:.1f} - {stats['wavelength_range_angstrom'][1]:.1f} Å")
    print(f"Log(gf) range: {stats['log_gf_range'][0]:.2f} - {stats['log_gf_range'][1]:.2f}")
    print(f"E_lower range: {stats['E_lower_range'][0]:.2f} - {stats['E_lower_range'][1]:.2f} eV")
    
    print(f"\n🔬 MOST COMMON SPECIES:")
    for species_name, count in stats['most_common_species']:
        print(f"  {species_name}: {count:,} lines")
    
    if stats['metadata']:
        print(f"\n📋 METADATA:")
        for key, value in stats['metadata'].items():
            print(f"  {key}: {value}")


def create_line_window(central_wavelength: float, 
                      window_width: float,
                      linelist: LineList) -> LineList:
    """
    Extract lines within a wavelength window around a central wavelength.
    
    Parameters
    ----------
    central_wavelength : float
        Central wavelength in Angstroms
    window_width : float
        Width of window in Angstroms
    linelist : LineList
        Source linelist
        
    Returns
    -------
    LineList
        Lines within the specified window
    """
    # Convert to cm
    central_wl_cm = central_wavelength * 1e-8
    window_width_cm = window_width * 1e-8
    
    wl_min = central_wl_cm - window_width_cm / 2
    wl_max = central_wl_cm + window_width_cm / 2
    
    return linelist.filter_by_wavelength(wl_min, wl_max, unit='cm')


def validate_linelist_physics(linelist: LineList) -> Dict:
    """
    Validate linelist for physical reasonableness.
    
    Parameters
    ----------
    linelist : LineList
        Linelist to validate
        
    Returns
    -------
    Dict
        Validation results
    """
    issues = []
    warnings_count = 0
    
    for i, line in enumerate(linelist.lines):
        # Check wavelength
        if line.wavelength <= 0:
            issues.append(f"Line {i}: Non-positive wavelength {line.wavelength}")
        
        # Check log_gf
        if line.log_gf < -10 or line.log_gf > 2:
            issues.append(f"Line {i}: Extreme log_gf value {line.log_gf}")
            warnings_count += 1
        
        # Check E_lower
        if line.E_lower < 0 or line.E_lower > 50:
            issues.append(f"Line {i}: Extreme E_lower value {line.E_lower} eV")
            warnings_count += 1
        
        # Check broadening parameters
        if hasattr(line, 'gamma_rad') and line.gamma_rad < 0:
            issues.append(f"Line {i}: Negative radiative damping {line.gamma_rad}")
        
        if hasattr(line, 'gamma_stark') and line.gamma_stark < 0:
            issues.append(f"Line {i}: Negative Stark damping {line.gamma_stark}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings_count': warnings_count,
        'total_lines': len(linelist.lines)
    }


def convert_linelist_to_korg_format(linelist: LineList) -> List[Line]:
    """
    Convert LineList to Korg.jl-compatible Line objects.
    
    Parameters
    ----------
    linelist : LineList
        Linelist to convert
        
    Returns
    -------
    List[Line]
        List of Korg.jl-compatible Line objects
    """
    korg_lines = []
    
    for line_data in linelist.lines:
        # Convert species ID to Species object
        species = species_from_integer(line_data.species)
        
        # Create vdW tuple
        vdw_tuple = (line_data.vdw_param1, line_data.vdw_param2)
        
        # Create Line object
        korg_line = create_line(
            wl=line_data.wavelength,
            log_gf=line_data.log_gf,
            species=species,
            E_lower=line_data.E_lower,
            gamma_rad=line_data.gamma_rad,
            gamma_stark=line_data.gamma_stark,
            vdW=vdw_tuple,
            wavelength_unit='cm'
        )
        
        korg_lines.append(korg_line)
    
    return korg_lines


def split_linelist_by_species(linelist: LineList) -> Dict[str, LineList]:
    """
    Split linelist into separate linelists by species.
    
    Parameters
    ----------
    linelist : LineList
        Linelist to split
        
    Returns
    -------
    Dict[str, LineList]
        Dictionary mapping species names to their linelists
    """
    species_lines = defaultdict(list)
    
    for line in linelist.lines:
        species_obj = species_from_integer(line.species)
        species_name = str(species_obj)
        species_lines[species_name].append(line)
    
    # Create LineList objects for each species
    species_linelists = {}
    for species_name, lines in species_lines.items():
        metadata = linelist.metadata.copy()
        metadata['species'] = species_name
        metadata['n_lines'] = len(lines)
        species_linelists[species_name] = LineList(lines, metadata)
    
    return species_linelists


def estimate_memory_usage(linelist: LineList) -> Dict:
    """
    Estimate memory usage of a linelist.
    
    Parameters
    ----------
    linelist : LineList
        Linelist to analyze
        
    Returns
    -------
    Dict
        Memory usage estimates
    """
    if not linelist.lines:
        return {'total_mb': 0, 'per_line_bytes': 0}
    
    # Estimate bytes per line (rough approximation)
    # Each LineData has ~8 floats (8 bytes each) + 1 int (4 bytes) = 68 bytes
    bytes_per_line = 68
    
    total_bytes = len(linelist.lines) * bytes_per_line
    total_mb = total_bytes / (1024 * 1024)
    
    return {
        'total_mb': total_mb,
        'per_line_bytes': bytes_per_line,
        'total_lines': len(linelist.lines),
        'estimated_total_bytes': total_bytes
    }