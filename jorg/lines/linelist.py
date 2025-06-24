"""
Linelist reading and parsing for stellar spectral synthesis

This module provides comprehensive linelist reading capabilities compatible
with major stellar spectroscopy formats, matching Korg.jl functionality.
"""

import numpy as np
import pandas as pd
import h5py
import re
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
import warnings

from .main import LineData, create_line_data
from .species import parse_species, Species
from .wavelength_utils import air_to_vacuum, vacuum_to_air, detect_wavelength_unit


class LineList:
    """Container for stellar linelist data with utilities"""
    
    def __init__(self, lines: List[LineData], metadata: Optional[Dict] = None):
        self.lines = lines
        self.metadata = metadata or {}
        self._wavelengths = None
    
    def __len__(self):
        return len(self.lines)
    
    def __getitem__(self, idx):
        return self.lines[idx]
    
    def __iter__(self):
        return iter(self.lines)
    
    @property
    def wavelengths(self):
        """Get array of wavelengths in cm"""
        if self._wavelengths is None:
            self._wavelengths = np.array([line.wavelength for line in self.lines])
        return self._wavelengths
    
    def wavelengths_angstrom(self):
        """Get wavelengths in Angstroms"""
        return self.wavelengths * 1e8
    
    def filter_by_wavelength(self, wl_min: float, wl_max: float, unit: str = 'angstrom'):
        """Filter linelist by wavelength range"""
        if unit == 'angstrom':
            wl_min_cm = wl_min * 1e-8
            wl_max_cm = wl_max * 1e-8
        else:
            wl_min_cm = wl_min
            wl_max_cm = wl_max
        
        filtered_lines = [
            line for line in self.lines
            if wl_min_cm <= line.wavelength <= wl_max_cm
        ]
        
        return LineList(filtered_lines, self.metadata)
    
    def filter_by_species(self, species_ids: List[int]):
        """Filter linelist by species"""
        filtered_lines = [
            line for line in self.lines
            if line.species_id in species_ids
        ]
        
        return LineList(filtered_lines, self.metadata)
    
    def sort_by_wavelength(self):
        """Sort linelist by wavelength"""
        sorted_lines = sorted(self.lines, key=lambda x: x.wavelength)
        return LineList(sorted_lines, self.metadata)
    
    def prune_weak_lines(self, log_gf_threshold: float = -6.0):
        """Remove very weak lines"""
        strong_lines = [
            line for line in self.lines
            if line.log_gf >= log_gf_threshold
        ]
        
        return LineList(strong_lines, self.metadata)


def read_linelist(filename: Union[str, Path], 
                 format: str = "auto",
                 wavelength_unit: str = "auto",
                 isotopic_abundances: Optional[Dict] = None) -> LineList:
    """
    Read linelist from various formats
    
    Parameters:
    -----------
    filename : str or Path
        Path to linelist file
    format : str
        Format type: "auto", "vald", "kurucz", "moog", "turbospectrum", "korg"
    wavelength_unit : str  
        "auto", "angstrom", "cm", "air", "vacuum"
    isotopic_abundances : dict, optional
        Custom isotopic abundances
        
    Returns:
    --------
    LineList
        Parsed linelist
    """
    
    filename = Path(filename)
    
    # Auto-detect format
    if format == "auto":
        format = detect_format(filename)
    
    print(f"📖 Reading linelist: {filename.name}")
    print(f"   Format: {format}")
    print(f"   Wavelength unit: {wavelength_unit}")
    
    # Parse based on format
    if format == "vald":
        return parse_vald_linelist(filename, wavelength_unit, isotopic_abundances)
    elif format == "kurucz":
        return parse_kurucz_linelist(filename, wavelength_unit)
    elif format == "moog":
        return parse_moog_linelist(filename, wavelength_unit)
    elif format == "turbospectrum":
        return parse_turbospectrum_linelist(filename, wavelength_unit)
    elif format == "korg":
        return load_korg_linelist(filename)
    else:
        raise ValueError(f"Unsupported format: {format}")


def detect_format(filename: Path) -> str:
    """Auto-detect linelist format from filename and content"""
    
    # Check file extension
    if filename.suffix.lower() == '.h5':
        return 'korg'
    
    # Read first few lines to detect format
    try:
        with open(filename, 'r') as f:
            lines = [f.readline().strip() for _ in range(10)]
    except UnicodeDecodeError:
        # Binary file, assume HDF5
        return 'korg'
    
    # VALD detection
    if any('VALD' in line for line in lines):
        return 'vald'
    
    # Look for format indicators
    for line in lines:
        if not line or line.startswith('#'):
            continue
            
        # Try to parse as numbers
        try:
            parts = line.split()
            if len(parts) >= 4:
                float(parts[0])  # wavelength
                float(parts[1])  # log_gf or species
                
                # Kurucz format has specific structure
                if len(parts) >= 6 and '.' in parts[1]:
                    return 'kurucz'
                    
                # MOOG format often has fewer columns
                if len(parts) <= 6:
                    return 'moog'
                    
                # Default to turbospectrum for complex formats
                return 'turbospectrum'
                
        except ValueError:
            continue
    
    # Default fallback
    warnings.warn(f"Could not detect format for {filename}, assuming VALD")
    return 'vald'


def parse_vald_linelist(filename: Path, 
                       wavelength_unit: str = "auto",
                       isotopic_abundances: Optional[Dict] = None) -> LineList:
    """Parse VALD format linelist"""
    
    lines = []
    metadata = {'format': 'vald', 'filename': str(filename)}
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Skip header until we find the data
    data_lines = []
    in_data = False
    
    for line in content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        # Start of data (after headers)
        if not in_data:
            if (line.startswith('\'') or 
                (len(line.split()) >= 4 and not line.startswith('#') and not line.startswith('VALD'))):
                in_data = True
            else:
                continue
        
        if in_data:
            data_lines.append(line)
    
    print(f"   Found {len(data_lines)} data lines")
    
    # Parse each line
    for line_text in data_lines:
        try:
            line_data = parse_vald_line(line_text, wavelength_unit, isotopic_abundances)
            if line_data:
                lines.append(line_data)
        except Exception as e:
            warnings.warn(f"Could not parse line: {line_text[:50]}... Error: {e}")
            continue
    
    metadata['n_lines'] = len(lines)
    metadata['n_parsed'] = len(data_lines)
    
    print(f"   Successfully parsed {len(lines)} lines")
    
    return LineList(lines, metadata).sort_by_wavelength()


def parse_vald_line(line_text: str, 
                   wavelength_unit: str,
                   isotopic_abundances: Optional[Dict]) -> Optional[LineData]:
    """Parse a single VALD format line"""
    
    # Remove quotes and clean up
    line_text = line_text.replace("'", "").strip()
    
    # Split by comma or whitespace
    if ',' in line_text:
        parts = [p.strip() for p in line_text.split(',')]
    else:
        parts = line_text.split()
    
    # Remove empty parts
    parts = [p for p in parts if p]
    
    if len(parts) < 4:
        return None
    
    try:
        # Basic fields
        wavelength_str = parts[0]
        log_gf = float(parts[1])
        E_lower_str = parts[2]
        species_str = parts[3]
        
        # Parse wavelength
        wavelength = float(wavelength_str)
        if wavelength_unit == "auto":
            if wavelength > 1000:  # Assume Angstroms
                wavelength_cm = wavelength * 1e-8
            else:
                wavelength_cm = wavelength  # Assume cm
        elif wavelength_unit == "angstrom":
            wavelength_cm = wavelength * 1e-8
        else:
            wavelength_cm = wavelength
        
        # Convert air to vacuum if needed (VALD typically gives air wavelengths)
        wavelength_cm = air_to_vacuum(wavelength_cm)
        
        # Parse energy (could be in cm^-1 or eV)
        E_lower = float(E_lower_str)
        if E_lower > 100:  # Assume cm^-1, convert to eV
            E_lower = E_lower * 1.24e-4  # cm^-1 to eV conversion
        
        # Parse species
        species_id = parse_species(species_str)
        
        # Damping parameters (if available)
        gamma_rad = 0.0
        gamma_stark = 0.0
        vdw_param1 = 0.0
        vdw_param2 = 0.0
        
        if len(parts) > 4:
            # Look for damping parameters
            if len(parts) >= 8:
                try:
                    gamma_rad = float(parts[4])
                    gamma_stark = float(parts[5])
                    vdw_param1 = float(parts[6])
                    if len(parts) > 7:
                        vdw_param2 = float(parts[7])
                except ValueError:
                    pass
        
        # Apply default broadening if not provided
        if gamma_rad == 0.0:
            gamma_rad = approximate_radiative_gamma(log_gf, wavelength_cm)
        if gamma_stark == 0.0:
            gamma_stark = approximate_stark_gamma(species_id)
        if vdw_param1 == 0.0:
            vdw_param1 = approximate_vdw_gamma(species_id)
        
        return create_line_data(
            wavelength_cm=wavelength_cm,
            log_gf=log_gf,
            E_lower_eV=E_lower,
            species_id=species_id,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
            vdw_param1=vdw_param1,
            vdw_param2=vdw_param2
        )
        
    except (ValueError, IndexError) as e:
        return None


def parse_kurucz_linelist(filename: Path, wavelength_unit: str = "auto") -> LineList:
    """Parse Kurucz format linelist"""
    
    lines = []
    metadata = {'format': 'kurucz', 'filename': str(filename)}
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                line_data = parse_kurucz_line(line, wavelength_unit)
                if line_data:
                    lines.append(line_data)
            except Exception as e:
                warnings.warn(f"Line {line_num}: Could not parse {line[:30]}... Error: {e}")
                continue
    
    metadata['n_lines'] = len(lines)
    print(f"   Successfully parsed {len(lines)} Kurucz lines")
    
    return LineList(lines, metadata).sort_by_wavelength()


def parse_kurucz_line(line_text: str, wavelength_unit: str) -> Optional[LineData]:
    """Parse a single Kurucz format line"""
    
    # Kurucz format: wavelength, species.ion, log_gf, E_lower, J_lower, J_upper
    parts = line_text.split()
    
    if len(parts) < 4:
        return None
    
    try:
        wavelength = float(parts[0])
        species_ion = float(parts[1])
        log_gf = float(parts[2])
        E_lower = float(parts[3])
        
        # Convert wavelength
        if wavelength_unit == "auto":
            if wavelength > 1000:
                wavelength_cm = wavelength * 1e-8
            else:
                wavelength_cm = wavelength
        elif wavelength_unit == "angstrom":
            wavelength_cm = wavelength * 1e-8
        else:
            wavelength_cm = wavelength
        
        # Parse species (format: element.ionization)
        element_id = int(species_ion)
        ion_state = int((species_ion - element_id) * 100 + 0.5)
        species_id = element_id * 100 + ion_state
        
        # Energy typically in cm^-1, convert to eV
        if E_lower > 100:
            E_lower_eV = E_lower * 1.24e-4
        else:
            E_lower_eV = E_lower
        
        # Default broadening parameters
        gamma_rad = approximate_radiative_gamma(log_gf, wavelength_cm)
        gamma_stark = approximate_stark_gamma(species_id)
        vdw_param1 = approximate_vdw_gamma(species_id)
        
        return create_line_data(
            wavelength_cm=wavelength_cm,
            log_gf=log_gf,
            E_lower_eV=E_lower_eV,
            species_id=species_id,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
            vdw_param1=vdw_param1,
            vdw_param2=0.0
        )
        
    except (ValueError, IndexError):
        return None


def parse_moog_linelist(filename: Path, wavelength_unit: str = "auto") -> LineList:
    """Parse MOOG format linelist"""
    
    lines = []
    metadata = {'format': 'moog', 'filename': str(filename)}
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                line_data = parse_moog_line(line, wavelength_unit)
                if line_data:
                    lines.append(line_data)
            except Exception as e:
                warnings.warn(f"Line {line_num}: Could not parse {line[:30]}... Error: {e}")
                continue
    
    metadata['n_lines'] = len(lines)
    print(f"   Successfully parsed {len(lines)} MOOG lines")
    
    return LineList(lines, metadata).sort_by_wavelength()


def parse_moog_line(line_text: str, wavelength_unit: str) -> Optional[LineData]:
    """Parse a single MOOG format line"""
    
    # MOOG format: wavelength, species, log_gf, E_lower, [vdW damping]
    parts = line_text.split()
    
    if len(parts) < 4:
        return None
    
    try:
        wavelength = float(parts[0])
        species_code = float(parts[1])
        log_gf = float(parts[2])
        E_lower = float(parts[3])
        
        # Convert wavelength
        if wavelength_unit == "auto":
            if wavelength > 1000:
                wavelength_cm = wavelength * 1e-8
            else:
                wavelength_cm = wavelength
        elif wavelength_unit == "angstrom":
            wavelength_cm = wavelength * 1e-8
        else:
            wavelength_cm = wavelength
        
        # Parse species code (format: element.ion)
        element_id = int(species_code)
        ion_state = int((species_code - element_id) * 10 + 0.5)
        species_id = element_id * 100 + ion_state
        
        # Energy in eV
        E_lower_eV = E_lower
        
        # vdW damping if provided
        vdw_param1 = 0.0
        if len(parts) > 4:
            try:
                vdw_param1 = float(parts[4])
            except ValueError:
                pass
        
        if vdw_param1 == 0.0:
            vdw_param1 = approximate_vdw_gamma(species_id)
        
        # Default broadening parameters
        gamma_rad = approximate_radiative_gamma(log_gf, wavelength_cm)
        gamma_stark = approximate_stark_gamma(species_id)
        
        return create_line_data(
            wavelength_cm=wavelength_cm,
            log_gf=log_gf,
            E_lower_eV=E_lower_eV,
            species_id=species_id,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
            vdw_param1=vdw_param1,
            vdw_param2=0.0
        )
        
    except (ValueError, IndexError):
        return None


def parse_turbospectrum_linelist(filename: Path, wavelength_unit: str = "auto") -> LineList:
    """Parse Turbospectrum format linelist"""
    
    lines = []
    metadata = {'format': 'turbospectrum', 'filename': str(filename)}
    
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                line_data = parse_turbospectrum_line(line, wavelength_unit)
                if line_data:
                    lines.append(line_data)
            except Exception as e:
                warnings.warn(f"Line {line_num}: Could not parse {line[:30]}... Error: {e}")
                continue
    
    metadata['n_lines'] = len(lines)
    print(f"   Successfully parsed {len(lines)} Turbospectrum lines")
    
    return LineList(lines, metadata).sort_by_wavelength()


def parse_turbospectrum_line(line_text: str, wavelength_unit: str) -> Optional[LineData]:
    """Parse a single Turbospectrum format line"""
    
    # Turbospectrum can have various formats, try to parse flexibly
    parts = line_text.split()
    
    if len(parts) < 4:
        return None
    
    try:
        wavelength = float(parts[0])
        species_id = int(float(parts[1]))
        log_gf = float(parts[2])
        E_lower = float(parts[3])
        
        # Convert wavelength
        if wavelength_unit == "auto":
            if wavelength > 1000:
                wavelength_cm = wavelength * 1e-8
            else:
                wavelength_cm = wavelength
        elif wavelength_unit == "angstrom":
            wavelength_cm = wavelength * 1e-8
        else:
            wavelength_cm = wavelength
        
        # Energy conversion
        if E_lower > 100:
            E_lower_eV = E_lower * 1.24e-4
        else:
            E_lower_eV = E_lower
        
        # Broadening parameters (if available)
        gamma_rad = 0.0
        gamma_stark = 0.0
        vdw_param1 = 0.0
        vdw_param2 = 0.0
        
        if len(parts) > 4:
            try:
                gamma_rad = float(parts[4])
                if len(parts) > 5:
                    gamma_stark = float(parts[5])
                if len(parts) > 6:
                    vdw_param1 = float(parts[6])
                if len(parts) > 7:
                    vdw_param2 = float(parts[7])
            except ValueError:
                pass
        
        # Apply defaults if needed
        if gamma_rad == 0.0:
            gamma_rad = approximate_radiative_gamma(log_gf, wavelength_cm)
        if gamma_stark == 0.0:
            gamma_stark = approximate_stark_gamma(species_id)
        if vdw_param1 == 0.0:
            vdw_param1 = approximate_vdw_gamma(species_id)
        
        return create_line_data(
            wavelength_cm=wavelength_cm,
            log_gf=log_gf,
            E_lower_eV=E_lower_eV,
            species_id=species_id,
            gamma_rad=gamma_rad,
            gamma_stark=gamma_stark,
            vdw_param1=vdw_param1,
            vdw_param2=vdw_param2
        )
        
    except (ValueError, IndexError):
        return None


def load_korg_linelist(filename: Path) -> LineList:
    """Load Korg native HDF5 format linelist"""
    
    with h5py.File(filename, 'r') as f:
        lines = []
        
        # Read arrays
        wavelengths = f['wavelength'][:]
        log_gfs = f['log_gf'][:]
        E_lowers = f['E_lower'][:]
        species_ids = f['species_id'][:]
        gamma_rads = f['gamma_rad'][:]
        gamma_starks = f['gamma_stark'][:]
        vdw_param1s = f['vdw_param1'][:]
        vdw_param2s = f['vdw_param2'][:]
        
        # Create LineData objects
        for i in range(len(wavelengths)):
            line = create_line_data(
                wavelength_cm=wavelengths[i],
                log_gf=log_gfs[i],
                E_lower_eV=E_lowers[i],
                species_id=species_ids[i],
                gamma_rad=gamma_rads[i],
                gamma_stark=gamma_starks[i],
                vdw_param1=vdw_param1s[i],
                vdw_param2=vdw_param2s[i]
            )
            lines.append(line)
        
        # Read metadata
        metadata = dict(f.attrs)
        metadata['format'] = 'korg'
        metadata['filename'] = str(filename)
    
    print(f"   Loaded {len(lines)} lines from Korg HDF5 format")
    
    return LineList(lines, metadata)


def save_linelist(filename: Union[str, Path], linelist: LineList):
    """Save linelist in Korg native HDF5 format"""
    
    filename = Path(filename)
    
    with h5py.File(filename, 'w') as f:
        # Create arrays
        n_lines = len(linelist)
        wavelengths = np.zeros(n_lines)
        log_gfs = np.zeros(n_lines)
        E_lowers = np.zeros(n_lines)
        species_ids = np.zeros(n_lines, dtype=int)
        gamma_rads = np.zeros(n_lines)
        gamma_starks = np.zeros(n_lines)
        vdw_param1s = np.zeros(n_lines)
        vdw_param2s = np.zeros(n_lines)
        
        # Fill arrays
        for i, line in enumerate(linelist):
            wavelengths[i] = line.wavelength
            log_gfs[i] = line.log_gf
            E_lowers[i] = line.E_lower
            species_ids[i] = line.species_id
            gamma_rads[i] = line.gamma_rad
            gamma_starks[i] = line.gamma_stark
            vdw_param1s[i] = line.vdw_param1
            vdw_param2s[i] = line.vdw_param2
        
        # Save arrays
        f.create_dataset('wavelength', data=wavelengths)
        f.create_dataset('log_gf', data=log_gfs)
        f.create_dataset('E_lower', data=E_lowers)
        f.create_dataset('species_id', data=species_ids)
        f.create_dataset('gamma_rad', data=gamma_rads)
        f.create_dataset('gamma_stark', data=gamma_starks)
        f.create_dataset('vdw_param1', data=vdw_param1s)
        f.create_dataset('vdw_param2', data=vdw_param2s)
        
        # Save metadata
        for key, value in linelist.metadata.items():
            if isinstance(value, (int, float, str)):
                f.attrs[key] = value
    
    print(f"💾 Saved {len(linelist)} lines to {filename}")


# Approximation functions for missing broadening parameters

def approximate_radiative_gamma(log_gf: float, wavelength_cm: float) -> float:
    """Approximate radiative damping parameter"""
    # Use Unsoeld (1955) approximation
    f_value = 10**log_gf
    freq_hz = 2.998e10 / wavelength_cm  # c/λ
    gamma_rad = 2.47e-9 * f_value * freq_hz**2  # s^-1
    return gamma_rad


def approximate_stark_gamma(species_id: int) -> float:
    """Approximate Stark broadening parameter"""
    # Very crude approximation - use Cowley (1971) scaling
    element_id = species_id // 100
    ion_state = species_id % 100
    
    # Hydrogen has special treatment
    if element_id == 1:
        return 1e-5
    
    # Rough scaling with ionization potential
    if ion_state == 0:  # Neutral
        return 1e-6
    elif ion_state == 1:  # Singly ionized
        return 5e-6
    else:  # Multiply ionized
        return 1e-5


def approximate_vdw_gamma(species_id: int) -> float:
    """Approximate van der Waals broadening parameter"""
    # Use Unsoeld approximation
    element_id = species_id // 100
    ion_state = species_id % 100
    
    # Rough scaling based on atomic size
    if element_id <= 2:  # H, He
        return 1e-8
    elif element_id <= 10:  # Light elements
        return 5e-8
    elif element_id <= 20:  # Medium elements
        return 1e-7
    else:  # Heavy elements
        return 2e-7