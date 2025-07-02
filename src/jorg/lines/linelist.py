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

from .datatypes import LineData, create_line_data
from .species import parse_species, Species
from ..utils.wavelength_utils import air_to_vacuum, vacuum_to_air, detect_wavelength_unit


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
        Format type: "auto", "vald", "kurucz", "moog", "turbospectrum", "korg", "exomol"
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
    elif format == "exomol":
        raise ValueError("ExoMol format requires separate states and transitions files. Use load_exomol_linelist() directly.")
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
            wavelength=wavelength_cm,
            species=species_id,
            log_gf=log_gf,
            E_lower=E_lower,
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
            wavelength=wavelength_cm,
            species=species_id,
            log_gf=log_gf,
            E_lower=E_lower_eV,
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
            wavelength=wavelength_cm,
            species=species_id,
            log_gf=log_gf,
            E_lower=E_lower_eV,
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
            wavelength=wavelength_cm,
            species=species_id,
            log_gf=log_gf,
            E_lower=E_lower_eV,
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
                wavelength=wavelengths[i],
                species=species_ids[i],
                log_gf=log_gfs[i],
                E_lower=E_lowers[i],
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


# ExoMol Linelist Parsing

def load_exomol_linelist(
    species_name: str,
    states_file: Union[str, Path],
    transitions_file: Union[str, Path], 
    lower_level: int,
    upper_level: int,
    line_strength_cutoff: float = -15.0,
    temperature_line_strength: float = 3500.0,
    wavelength_range: Optional[Tuple[float, float]] = None
) -> LineList:
    """
    Load ExoMol format molecular linelist.
    
    This function matches Korg.jl's load_ExoMol_linelist functionality,
    parsing ExoMol states and transitions files to create a molecular linelist.
    
    Parameters
    ----------
    species_name : str
        Molecular species name (e.g., 'H2O', 'TiO', 'CaH')
    states_file : Path
        Path to ExoMol .states file
    transitions_file : Path
        Path to ExoMol .trans file
    lower_level : int
        Lower electronic state
    upper_level : int
        Upper electronic state  
    line_strength_cutoff : float
        Minimum log10(line strength) threshold (default: -15)
    temperature_line_strength : float
        Temperature for line strength evaluation in K (default: 3500K)
    wavelength_range : Tuple[float, float], optional
        Wavelength range in Angstroms (min, max)
        
    Returns
    -------
    LineList
        Parsed molecular linelist
    """
    print(f"🔬 Loading ExoMol linelist for {species_name}")
    print(f"   States file: {states_file}")
    print(f"   Transitions file: {transitions_file}")
    
    # Load states data
    print("   Loading states...")
    states_df = pd.read_csv(
        states_file,
        delim_whitespace=True,
        names=['state_id', 'energy', 'degeneracy', 'J', 'uncertainty', 'lifetime'],
        comment='#'
    )
    
    # Load transitions data
    print("   Loading transitions...")
    transitions_df = pd.read_csv(
        transitions_file,
        delim_whitespace=True,
        names=['upper_state', 'lower_state', 'A_ul', 'uncertainty'],
        comment='#'
    )
    
    print(f"   Found {len(states_df)} states and {len(transitions_df)} transitions")
    
    # Join transitions with state information
    print("   Joining transitions with states...")
    
    # Get upper state info
    transitions_df = transitions_df.merge(
        states_df[['state_id', 'energy', 'degeneracy', 'J']],
        left_on='upper_state',
        right_on='state_id',
        suffixes=('', '_upper')
    )
    
    # Get lower state info
    transitions_df = transitions_df.merge(
        states_df[['state_id', 'energy', 'degeneracy', 'J']],
        left_on='lower_state', 
        right_on='state_id',
        suffixes=('_upper', '_lower')
    )
    
    # Calculate transition properties
    print("   Calculating transition properties...")
    
    # Energy difference in cm^-1
    wavenumber = transitions_df['energy_upper'] - transitions_df['energy_lower']
    
    # Convert to wavelengths in Angstroms (vacuum)
    wavelength_angstrom = 1e8 / wavenumber  # cm^-1 to Å
    wavelength_cm = wavelength_angstrom * 1e-8
    
    # Calculate oscillator strength using Gray equation 11.12
    # f_ul = (8π²me c) / (3h e² λ) * (g_l/g_u) * A_ul
    import scipy.constants as const
    
    # Physical constants in CGS
    me_cgs = const.m_e * 1000  # g
    c_cgs = const.c * 100      # cm/s
    h_cgs = const.h * 1e7      # erg⋅s
    e_cgs = const.e * const.c * 10  # statcoulomb
    
    # Calculate f-values
    wavelength_m = wavelength_angstrom * 1e-10
    prefactor = (8 * np.pi**2 * me_cgs * c_cgs) / (3 * h_cgs * e_cgs**2)
    
    g_ratio = transitions_df['degeneracy_lower'] / transitions_df['degeneracy_upper']
    f_values = prefactor * wavelength_m * g_ratio * transitions_df['A_ul']
    log_gf = np.log10(transitions_df['degeneracy_lower'] * f_values)
    
    # Apply isotopic correction for most abundant isotopologue
    # (This is simplified - would need actual isotopic data)
    isotopic_correction = 1.0
    log_gf += np.log10(isotopic_correction)
    
    # Calculate lower level energy in eV
    hc_eV_cm = const.h * const.c / const.eV * 100  # eV⋅cm
    E_lower_eV = transitions_df['energy_lower'] * hc_eV_cm
    
    # Approximate line strength for filtering
    # S(T) ≈ gf * exp(-E_lower/kT) for simple estimate
    kT_eV = const.k * temperature_line_strength / const.eV
    line_strength_approx = f_values * np.exp(-E_lower_eV / kT_eV)
    log_line_strength = np.log10(line_strength_approx)
    
    # Apply filters
    print("   Applying filters...")
    mask = np.ones(len(transitions_df), dtype=bool)
    
    # Line strength cutoff
    mask &= (log_line_strength >= line_strength_cutoff)
    
    # Wavelength range filter
    if wavelength_range is not None:
        wl_min, wl_max = wavelength_range
        mask &= (wavelength_angstrom >= wl_min) & (wavelength_angstrom <= wl_max)
    
    # Positive wavelengths only
    mask &= (wavelength_angstrom > 0)
    
    print(f"   After filtering: {mask.sum()} lines remain")
    
    # Create molecular species identifier
    molecular_species = get_molecular_species_id(species_name)
    
    # Create LineData objects
    lines = []
    for i in np.where(mask)[0]:
        # Molecular lines have only radiative damping
        gamma_rad = approximate_radiative_gamma(log_gf[i], wavelength_cm[i])
        
        line = create_line_data(
            wavelength=wavelength_cm[i],
            species=molecular_species,
            log_gf=log_gf[i],
            E_lower=E_lower_eV[i],
            gamma_rad=gamma_rad,
            gamma_stark=0.0,  # No Stark broadening for molecules
            vdw_param1=0.0,  # No vdW broadening for molecules
            vdw_param2=0.0
        )
        lines.append(line)
    
    # Create metadata
    metadata = {
        'format': 'exomol',
        'species': species_name,
        'states_file': str(states_file),
        'transitions_file': str(transitions_file), 
        'lower_level': lower_level,
        'upper_level': upper_level,
        'line_strength_cutoff': line_strength_cutoff,
        'temperature_line_strength': temperature_line_strength,
        'n_lines_total': len(transitions_df),
        'n_lines_filtered': len(lines)
    }
    
    print(f"   ✅ Successfully loaded {len(lines)} ExoMol lines")
    
    return LineList(lines, metadata).sort_by_wavelength()


def get_molecular_species_id(species_name: str) -> int:
    """
    Get species ID for common molecules.
    
    This maps molecular names to species codes compatible with Korg.jl.
    
    Parameters
    ----------
    species_name : str
        Molecular species name
        
    Returns
    -------
    int
        Species identifier
    """
    molecular_ids = {
        'H2O': 801,    # Water
        'TiO': 2208,   # Titanium oxide
        'VO': 2308,    # Vanadium oxide
        'OH': 108,     # Hydroxyl
        'CH': 601,     # Methylidyne  
        'CN': 607,     # Cyanogen
        'CO': 608,     # Carbon monoxide
        'NH': 701,     # Imidogen
        'SiO': 1408,   # Silicon monoxide
        'CaH': 2001,   # Calcium hydride
        'FeH': 2601,   # Iron hydride
        'MgH': 1201,   # Magnesium hydride
        'AlH': 1301,   # Aluminum hydride
        'SiH': 1401,   # Silicon hydride
        'H2': 101,     # Molecular hydrogen
        'C2': 606,     # Diatomic carbon
        'N2': 707,     # Molecular nitrogen
        'O2': 808,     # Molecular oxygen
    }
    
    return molecular_ids.get(species_name, 99999)  # Default unknown molecule


def approximate_line_strength(
    log_gf: float,
    E_lower_eV: float, 
    temperature: float = 3500.0
) -> float:
    """
    Approximate line strength at given temperature.
    
    Uses simplified Boltzmann factor for quick filtering.
    
    Parameters
    ----------
    log_gf : float
        Log oscillator strength
    E_lower_eV : float
        Lower level energy in eV
    temperature : float
        Temperature in K
        
    Returns
    -------
    float
        Log line strength estimate
    """
    import scipy.constants as const
    
    kT_eV = const.k * temperature / const.eV
    boltzmann_factor = -E_lower_eV / kT_eV
    
    return log_gf + boltzmann_factor / np.log(10)