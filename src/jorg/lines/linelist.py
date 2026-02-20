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

from .datatypes import LineData, create_line_data, Line, create_line, species_from_integer
from .species import parse_species, Species
from .atomic_data import (get_atomic_symbol, get_atomic_mass, get_isotopic_abundance,
                         get_abundances_dict, get_atomic_masses_dict, get_atomic_numbers_dict,
                         ISOTOPIC_ABUNDANCES)
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
            if getattr(line, 'species_id', line.species) in species_ids
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
        Format type: "auto", "vald", "kurucz", "moog", "turbospectrum", "korg", "alpha_5000", "exomol"
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
    elif format == "galah_dr3":
        return load_galah_dr3_linelist(filename)
    elif format == "alpha_5000":
        return parse_alpha_5000_linelist(filename, wavelength_unit)
    elif format == "exomol":
        raise ValueError("ExoMol format requires separate states and transitions files. Use load_exomol_linelist() directly.")
    else:
        raise ValueError(f"Unsupported format: {format}")


def detect_format(filename: Path) -> str:
    """Auto-detect linelist format from filename and content"""
    
    # Check file extension
    if filename.suffix.lower() == '.h5':
        # For HDF5 files, need to check internal structure
        try:
            import h5py
            with h5py.File(filename, 'r') as f:
                keys = list(f.keys())
                # GALAH DR3 uses 'wl' for wavelength, standard Korg uses 'wavelength'
                if 'wl' in keys and 'E_lo' in keys and 'formula' in keys:
                    return 'galah_dr3'
                elif 'wavelength' in keys and 'E_lower' in keys and 'species_id' in keys:
                    return 'korg'
                else:
                    # Default to korg if can't distinguish
                    return 'korg'
        except ImportError:
            print("Warning: h5py not available, assuming standard Korg format")
            return 'korg'
        except Exception:
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
    
    # Alpha_5000 CSV detection
    if filename.suffix.lower() == '.csv':
        # Check if it has alpha_5000 header structure
        if lines and 'wl,log_gf,species,E_lower,gamma_rad,gamma_stark,vdW' in lines[0]:
            return 'alpha_5000'
        # Check for alpha_5000 in filename
        if 'alpha_5000' in filename.name.lower():
            return 'alpha_5000'
    
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
    """Parse VALD format linelist with full Korg.jl compatibility"""
    
    lines = []
    metadata = {'format': 'vald', 'filename': str(filename)}
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Import default isotopic abundances if not provided
    if isotopic_abundances is None:
        from .atomic_data import ISOTOPIC_ABUNDANCES
        isotopic_abundances = ISOTOPIC_ABUNDANCES
    
    # Detect format details from header
    header_lines = []
    data_lines = []
    in_data = False
    is_extract_stellar = False
    is_short_format = True
    scale_isotopes = False
    air_wavelengths = False
    
    for line in content.split('\n'):
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        # Check for format indicators in header
        if not in_data:
            header_lines.append(line)
            
            # Check for extract stellar format
            if 'lines selected' in line:
                is_extract_stellar = True
            
            # Check for isotopic scaling message
            if 'oscillator strengths were NOT scaled' in line:
                scale_isotopes = True
            elif 'oscillator strengths were scaled' in line:
                scale_isotopes = False
            
            # CRITICAL: Check for air vs vacuum wavelengths in header
            if 'WL_air' in line or 'air' in line.lower():
                air_wavelengths = True
                print(f"   Detected air wavelengths - will convert to vacuum")
            elif 'WL_vac' in line or 'vac' in line.lower():
                air_wavelengths = False
                print(f"   Detected vacuum wavelengths - no conversion needed")
                
            # Start of data - look for lines with single quotes
            if line_stripped.startswith('\''):
                in_data = True
                # Check if next data line indicates long format
                # Long format has lines starting with quotes followed by space
        
        if in_data:
            # Only process lines that start with quotes (actual data lines)
            if line_stripped.startswith('\''):
                data_lines.append(line)
    
    print(f"   Found {len(data_lines)} data lines")
    
    # Parse each line with reference string for isotopic corrections
    filtered_rare_earth = 0
    filtered_molecular = 0
    
    for i, line_text in enumerate(data_lines):
        # For short format, reference string is at end of line
        # For long format, it would be on separate lines (not fully implemented)
        reference_str = ""
        if "'" in line_text:
            # Extract reference string (last quoted section)
            # Use regex to find all quoted sections
            import re
            quoted_sections = re.findall(r"'([^']*)'", line_text)
            if quoted_sections and len(quoted_sections) > 1:
                # The last quoted section is the reference string
                reference_str = quoted_sections[-1]
        
        try:
            line_data = parse_vald_line(line_text, wavelength_unit, isotopic_abundances, 
                                       reference_str, scale_isotopes, air_wavelengths)
            if line_data:
                lines.append(line_data)
            # Statistics tracking removed since we're not filtering anymore
        except Exception as e:
            warnings.warn(f"Could not parse line: {line_text[:50]}... Error: {e}")
            continue
    
    metadata['n_lines'] = len(lines)
    metadata['n_parsed'] = len(data_lines)
    
    print(f"   Successfully parsed {len(lines)} lines")
    if filtered_rare_earth > 0:
        print(f"   Filtered out {filtered_rare_earth} rare earth/heavy element lines")
    if filtered_molecular > 0:
        print(f"   Filtered out {filtered_molecular} molecular lines")
    
    # Apply Korg.jl-compatible line filtering BEFORE creating LineList
    # Filter: (0 <= line.species.charge <= 2) && (line.species != species"H_I")
    filtered_lines = []
    for line in lines:
        # Extract charge from species ID (species_id = atomic_number * 100 + charge)
        if hasattr(line, 'species') and isinstance(line.species, int):
            atomic_number = line.species // 100
            charge = line.species % 100
            
            # Filter triply+ ionized lines (charge > 2)
            if charge > 2:
                continue
                
            # Filter hydrogen lines (atomic_number = 1)
            if atomic_number == 1:
                continue
                
        filtered_lines.append(line)
    
    print(f"   After Korg.jl filtering: {len(filtered_lines)} lines (removed {len(lines) - len(filtered_lines)} lines)")
    
    return LineList(filtered_lines, metadata).sort_by_wavelength()


def parse_vald_line(line_text: str, 
                   wavelength_unit: str,
                   isotopic_abundances: Optional[Dict],
                   reference_str: str = "",
                   scale_isotopes: bool = False,
                   air_wavelengths: bool = False) -> Optional[LineData]:
    """Parse a single VALD format line"""
    
    # Remove quotes and clean up
    line_text = line_text.replace("'", "").strip()
    
    # Split by comma and clean up
    if ',' in line_text:
        parts = [p.strip() for p in line_text.split(',')]
    else:
        parts = line_text.split()
    
    # Remove empty parts
    parts = [p for p in parts if p]
    
    if len(parts) < 8:  # Need at least: species, wavelength, E_lower, vmic, log_gf, gamma_rad, gamma_stark, vdw
        return None
    
    try:
        # VALD format: species, wavelength, E_lower, vmic, log_gf, gamma_rad, gamma_stark, vdw, lande, depth, reference
        species_str = parts[0]
        wavelength_str = parts[1]
        E_lower_str = parts[2]
        vmic = float(parts[3])  # microturbulence (not used in line data)
        log_gf = float(parts[4])
        
        # Parse species
        species_id = parse_species(species_str)
        
        # REMOVED: Overly aggressive filtering that doesn't match Korg.jl
        # Korg.jl includes molecular lines and rare earth elements, so Jorg should too
        # to maintain compatibility. Any filtering should be done at synthesis time.
        #
        # Previous code filtered:
        # - Molecular lines (species_id > 100000)
        # - Rare earth elements (Z=57-71, lanthanides)
        # - Actinides (Z=89-103)
        # - Heavy elements (Z>56)
        #
        # This caused Jorg to parse 3,656 fewer lines than Korg.jl (15,580 vs 19,236)
        
        # Apply isotopic abundance correction if needed
        isotopic_correction = 0.0
        if scale_isotopes and reference_str and isotopic_abundances:
            isotopic_correction = calculate_isotopic_correction(reference_str, isotopic_abundances)
        
        # Parse wavelength - VALD format uses Angstroms
        wavelength = float(wavelength_str)
        if wavelength_unit == "auto":
            if wavelength < 100:  # Assume cm, convert to Angstroms
                wavelength_angstroms = wavelength * 1e8  # Convert cm to Angstroms
            else:  # Assume Angstroms (VALD typical range 1000-25000 Å)
                wavelength_angstroms = wavelength
        elif wavelength_unit == "angstrom":
            wavelength_angstroms = wavelength
        else:
            wavelength_angstroms = wavelength * 1e8  # Convert cm to Angstroms
        
        # CRITICAL FIX: Convert air to vacuum if detected in header (matching Korg.jl behavior)
        if air_wavelengths:
            wavelength_angstroms = air_to_vacuum(wavelength_angstroms)
        
        # Parse energy (in eV)
        E_lower = float(E_lower_str)
        
        # Damping parameters
        gamma_rad = 0.0
        gamma_stark = 0.0
        vdw_param1 = 0.0
        vdw_param2 = 0.0
        
        if len(parts) > 5:
            try:
                # CRITICAL FIX: VALD gamma values are in log₁₀ form, need 10^x conversion
                gamma_rad_log = float(parts[5])
                gamma_stark_log = float(parts[6])
                
                # Convert from log₁₀ to linear scale (as Korg.jl does with tentotheOrMissing)
                # Korg.jl: tentotheOrMissing(x) = x == 0 ? missing : 10^x
                if gamma_rad_log == 0:
                    gamma_rad = 0.0  # Will use default approximation
                else:
                    # Prevent overflow for very large gamma_rad values
                    # Some molecular lines have huge gamma_rad values (e.g., 100+)
                    # Cap at reasonable physical value (10^20 Hz is already enormous)
                    if gamma_rad_log > 20:
                        gamma_rad_log = 20.0
                    gamma_rad = 10.0**gamma_rad_log  # VALD format: log₁₀(gamma) in s⁻¹
                    
                if gamma_stark_log == 0:
                    gamma_stark = 0.0  # Will use default approximation
                else:
                    # Also cap gamma_stark to prevent overflow
                    if gamma_stark_log > 20:
                        gamma_stark_log = 20.0
                    elif gamma_stark_log < -20:
                        gamma_stark_log = -20.0
                    gamma_stark = 10.0**gamma_stark_log  # Convert even if negative!
                
                # CRITICAL FIX: VALD vdW parameter interpretation (following Korg.jl)
                vdw_raw = float(parts[7])
                
                # Interpret vdW based on value range (Korg.jl linelist.jl logic):
                if vdw_raw == 0:
                    # No vdW broadening
                    vdw_param1 = 0.0
                    vdw_param2 = 0.0
                elif vdw_raw < 0:
                    # Negative: log10(γ_vdW) evaluated at 10,000 K (Korg.jl behavior)
                    vdw_param1 = 10.0**vdw_raw
                    vdw_param2 = -1.0  # Marker for gamma_vdW at 10k
                elif 0 < vdw_raw < 20:
                    # Between 0 and 20: Unsöld approximation fudge factor
                    vdw_param1 = vdw_raw
                    vdw_param2 = -2.0  # Marker for Unsöld fudge factor
                else:
                    # >= 20: Packed ABO parameters (σ, α)
                    # σ = floor(vdW) * bohr_radius²
                    # α = vdW - floor(vdW)
                    import numpy as np
                    bohr_radius_cgs = 5.29177210903e-9  # cm
                    vdw_param1 = np.floor(vdw_raw) * bohr_radius_cgs * bohr_radius_cgs  # σ in cm²
                    vdw_param2 = vdw_raw - np.floor(vdw_raw)  # α (dimensionless)
                
                if len(parts) > 8:
                    # Lande factor is in parts[8], depth in parts[9]
                    pass
            except ValueError:
                pass
        
        # Apply default broadening if not provided
        if gamma_rad == 0.0:
            gamma_rad = approximate_radiative_gamma(log_gf, wavelength_angstroms * 1e-8)
        if gamma_stark == 0.0:
            gamma_stark = approximate_stark_gamma(species_id)
        if vdw_param1 == 0.0:
            vdw_param1 = approximate_vdw_gamma(species_id)
            vdw_param2 = -1.0
        
        # Convert species integer to proper Species object
        species_obj = species_from_integer(species_id)
        
        return create_line_data(
            wavelength=wavelength_angstroms,
            species=species_obj,
            log_gf=log_gf + isotopic_correction,  # Apply isotopic correction to log_gf
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
        
        # Convert species integer to proper Species object
        species_obj = species_from_integer(species_id)
        
        return create_line_data(
            wavelength=wavelength_cm,
            species=species_obj,
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
        
        # Convert species integer to proper Species object
        species_obj = species_from_integer(species_id)
        
        return create_line_data(
            wavelength=wavelength_cm,
            species=species_obj,
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
            gamma_rad = approximate_radiative_gamma(log_gf, wavelength_angstroms * 1e-8)
        if gamma_stark == 0.0:
            gamma_stark = approximate_stark_gamma(species_id)
        if vdw_param1 == 0.0:
            vdw_param1 = approximate_vdw_gamma(species_id)
        
        # Convert species integer to proper Species object
        species_obj = species_from_integer(species_id)
        
        return create_line_data(
            wavelength=wavelength_cm,
            species=species_obj,
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
            # Convert species integer to proper Species object
            species_obj = species_from_integer(int(species_ids[i]))
            
            line = create_line_data(
                wavelength=wavelengths[i],
                species=species_obj,
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


def load_galah_dr3_linelist(filename: Path) -> LineList:
    """Load GALAH DR3 format HDF5 linelist"""
    
    import h5py
    import numpy as np
    
    with h5py.File(filename, 'r') as f:
        lines = []
        
        # Read arrays with GALAH DR3 naming convention
        wavelengths = f['wl'][:]  # Wavelength in Angstroms
        log_gfs = f['log_gf'][:]
        E_lowers = f['E_lo'][:]  # Lower energy in eV
        ionizations = f['ionization'][:]  # Ionization stage
        formulas = f['formula'][:]  # [atomic_number, ?, ?]
        gamma_rads = f['gamma_rad'][:]
        gamma_starks = f['gamma_stark'][:]
        vdw_params = f['vdW'][:]
        
        # Convert wavelengths from Angstroms to cm
        wavelengths_cm = wavelengths * 1e-8
        
        # Create species IDs from atomic number and ionization
        # GALAH uses 1-based ionization (1=neutral, 2=singly ionized, etc.)
        # Convert to 0-based for compatibility (0=neutral, 1=singly ionized, etc.)
        atomic_numbers = formulas[:, 0].astype(np.int32)  # Convert to int32 to avoid overflow
        ionization_stages = (ionizations - 1).astype(np.int32)  # Convert to 0-based
        species_ids = atomic_numbers * 100 + ionization_stages
        
        # Create LineData objects
        for i in range(len(wavelengths)):
            # Convert species integer to proper Species object
            species_obj = species_from_integer(int(species_ids[i]))
            
            line = create_line_data(
                wavelength=wavelengths_cm[i],
                species=species_obj,
                log_gf=log_gfs[i],
                E_lower=E_lowers[i],
                gamma_rad=gamma_rads[i] if gamma_rads[i] != 0 else 6.16e7,  # Default if zero
                gamma_stark=gamma_starks[i],
                vdw_param1=vdw_params[i] if vdw_params[i] != 0 else -7.5,  # Default if zero
                vdw_param2=0.0
            )
            lines.append(line)
        
        # Read metadata
        metadata = dict(f.attrs) if f.attrs else {}
        metadata['format'] = 'galah_dr3'
        metadata['filename'] = str(filename)
        metadata['n_lines'] = len(lines)
    
    print(f"   Loaded {len(lines)} lines from GALAH DR3 HDF5 format")
    
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


def parse_alpha_5000_linelist(filename: Union[str, Path], wavelength_unit: str = "auto") -> LineList:
    """Parse Korg.jl alpha_5000 CSV format linelist"""
    
    import csv
    
    filename = Path(filename)
    
    print(f"📖 Reading alpha_5000 linelist: {filename.name}")
    
    lines = []
    
    def parse_korg_species(species_str):
        """Parse Korg.jl species string like 'Fe I' or 'OZr' into Species."""
        from ..statmech.species import Species as ChemSpecies
        return ChemSpecies.from_string(species_str)

    def parse_vdw_parameter(vdw_str):
        """Parse vdW parameter string from Korg.jl format."""
        vdw_str = vdw_str.strip()
        
        if vdw_str.startswith('(') and vdw_str.endswith(')'):
            # ABO parameters: (sigma, alpha)
            try:
                # Remove parentheses and split by comma
                inner = vdw_str[1:-1]
                parts = inner.split(',')
                sigma = float(parts[0].strip())
                alpha = float(parts[1].strip())
                return (sigma, alpha)
            except:
                return (0.0, 0.0)
        else:
            # Simple gamma_vdW value
            try:
                return float(vdw_str)
            except:
                return 0.0

    with open(filename, 'r') as f:
        csv_reader = csv.DictReader(f)
        
        for row in csv_reader:
            try:
                # Parse wavelength (Korg.jl stores in cm, convert to Å)
                wl_cm = float(row['wl'])  # Already in cm
                wl_angstrom = wl_cm * 1e8  # Convert cm to Å
                
                # Parse other parameters
                log_gf = float(row['log_gf'])
                species_obj = parse_korg_species(row['species'])
                E_lower = float(row['E_lower'])  # eV
                
                # Parse broadening parameters
                gamma_rad = float(row['gamma_rad']) if row['gamma_rad'] else 0.0
                gamma_stark = float(row['gamma_stark']) if row['gamma_stark'] else 0.0
                vdw_param = parse_vdw_parameter(row['vdW'])
                
                # Extract vdW parameters for VALD format
                if isinstance(vdw_param, tuple) and len(vdw_param) == 2:
                    vdw_param1, vdw_param2 = vdw_param
                else:
                    vdw_param1, vdw_param2 = float(vdw_param), -1.0
                
                line = create_line_data(
                    wavelength=wl_angstrom,  # Keep in Å 
                    species=species_obj,
                    log_gf=log_gf,
                    E_lower=E_lower,
                    gamma_rad=gamma_rad,
                    gamma_stark=gamma_stark,
                    vdw_param1=vdw_param1,
                    vdw_param2=vdw_param2,
                    wavelength_unit='angstrom'
                )
                
                lines.append(line)
                
            except Exception as e:
                print(f"Warning: Failed to parse line: {row} - {e}")
                continue
    
    print(f"   Format: alpha_5000")
    print(f"   Successfully parsed {len(lines)} lines")
    
    metadata = {
        'format': 'alpha_5000',
        'source': 'Korg.jl internal linelist',
        'wavelength_unit': 'angstrom',
        'filename': str(filename)
    }
    
    return LineList(lines, metadata)


def calculate_isotopic_correction(reference_str: str, isotopic_abundances: Dict) -> float:
    """
    Calculate isotopic abundance correction from VALD reference string.
    
    Matches Korg.jl logic from linelist.jl lines 501-510.
    Looks for patterns like (48)Ti, (56)Fe, etc. in reference string.
    
    Parameters
    ----------
    reference_str : str
        VALD reference string containing isotope information
    isotopic_abundances : dict
        Dictionary of isotopic abundances
        
    Returns
    -------
    float
        Log10 correction to apply to log_gf
    """
    import re
    import numpy as np
    from .atomic_data import ATOMIC_NUMBERS
    
    if not reference_str:
        return 0.0
    
    # Pattern to find isotope information like (48)Ti, (56)Fe
    # Matches: (isotope_mass)element_symbol
    pattern = r'\((\d{1,3})\)([A-Z][a-z]?)'
    
    matches = re.findall(pattern, reference_str)
    
    if not matches:
        return 0.0
    
    total_correction = 0.0
    
    for mass_str, element_symbol in matches:
        mass_number = int(mass_str)
        
        # Get atomic number from element symbol
        atomic_number = ATOMIC_NUMBERS.get(element_symbol, 0)
        
        if atomic_number > 0 and atomic_number in isotopic_abundances:
            isotopes = isotopic_abundances[atomic_number]
            
            if mass_number in isotopes:
                abundance = isotopes[mass_number]
                # Add log10 of abundance fraction
                total_correction += np.log10(abundance)
            else:
                # Isotope not found, assume it's rare
                print(f"Warning: Isotope {element_symbol}-{mass_number} not in database")
                # Don't add correction for unknown isotopes
    
    return total_correction


# Approximation functions for missing broadening parameters

def approximate_radiative_gamma(log_gf: float, wavelength_cm: float) -> float:
    """Approximate radiative damping parameter using Korg.jl method"""
    from .broadening_korg import approximate_radiative_gamma
    return approximate_radiative_gamma(wavelength_cm, log_gf)


def approximate_stark_gamma(species_id: int) -> float:
    """Approximate Stark broadening parameter using Korg.jl method"""
    # Convert species ID to Species object
    species = species_from_integer(species_id)
    
    # Use a representative wavelength and energy for approximation
    wl_cm = 5000e-8  # 5000 Angstroms in cm
    E_lower = 1.0    # Representative energy in eV
    
    from .broadening_korg import approximate_stark_broadening
    return approximate_stark_broadening(species, E_lower, wl_cm)


def approximate_vdw_gamma(species_id: int) -> float:
    """Approximate van der Waals broadening parameter using Korg.jl method"""
    # Convert species ID to Species object
    species = species_from_integer(species_id)
    
    # Use a representative wavelength and energy for approximation
    wl_cm = 5000e-8  # 5000 Angstroms in cm
    E_lower = 1.0    # Representative energy in eV
    
    from .broadening_korg import approximate_vdw_broadening
    return approximate_vdw_broadening(species, E_lower, wl_cm)


# ExoMol Linelist Parsing

# Constants used in Korg.jl ExoMol conversion logic (CGS/eV units)
_ELECTRON_MASS_CGS = 9.1093837015e-28
_C_CGS = 2.99792458e10
_ELECTRON_CHARGE_CGS = 4.803204712570263e-10
_HPLANCK_EV_S = 4.135667696e-15
_KBOLTZ_EV = 8.617333262145e-5
_LOG10_E = np.log10(np.e)


def _strip_exomol_isotopes(species_name: str) -> str:
    """
    Convert ExoMol isotopologue labels to plain molecular formulae.

    Examples
    --------
    40Ca-1H -> CaH
    1H2-16O -> H2O
    """
    name = species_name.strip()
    if "-" not in name:
        return name

    formula_parts = []
    for token in name.split("-"):
        token = token.strip()
        match = re.fullmatch(r"(\d+)([A-Z][a-z]?)(\d*)", token)
        if not match:
            # Fall back to a conservative normalization:
            # remove isotope numbers and separators.
            return re.sub(r"[^A-Za-z0-9]", "", re.sub(r"\d", "", name))
        symbol = match.group(2)
        count = match.group(3)
        formula_parts.append(f"{symbol}{count}" if count else symbol)
    return "".join(formula_parts)


def _resolve_exomol_species(species_name: str):
    """Parse species as the same Species class used in line opacity/statmech."""
    from ..statmech.species import Species as ChemSpecies

    normalized = _strip_exomol_isotopes(species_name)
    try:
        return ChemSpecies.from_string(normalized)
    except Exception:
        # Fall back to linelist species parser for compatibility with existing codes.
        species_id = parse_species(normalized)
        return species_from_integer(species_id)


def _default_exomol_isotopes(species_obj, isotopic_abundances: Dict[int, Dict[int, float]]):
    """Pick the most abundant isotope for each atom in species."""
    isotopes = []
    for z in species_obj.get_atoms():
        z_int = int(z)
        abundances = isotopic_abundances.get(z_int)
        if not abundances:
            continue
        isotope = max(abundances, key=abundances.get)
        isotopes.append((z_int, int(isotope)))
    return isotopes


def _compute_exomol_isotopic_correction(
    isotopes: List[Tuple[int, int]],
    isotopic_abundances: Dict[int, Dict[int, float]],
    isotopic_nuclear_spin_degeneracies: Optional[Dict[int, Dict[int, float]]] = None,
) -> float:
    """
    Compute log10 correction applied to log_gf from isotopic abundances.

    Korg.jl also subtracts a nuclear-spin degeneracy factor; this is optional here.
    """
    if not isotopes:
        return 0.0

    correction = 0.0
    for z, iso in isotopes:
        abundance = isotopic_abundances.get(z, {}).get(iso)
        if abundance is None or abundance <= 0:
            raise KeyError(
                f"Missing isotopic abundance for Z={z}, isotope={iso}. "
                "Pass isotopic_abundances to override defaults."
            )
        correction += np.log10(abundance)

        if isotopic_nuclear_spin_degeneracies is not None:
            degeneracy = isotopic_nuclear_spin_degeneracies.get(z, {}).get(iso, 1.0)
            if degeneracy is not None and degeneracy > 0:
                correction -= np.log10(degeneracy)
    return correction


def _resolve_exomol_wavelength_bounds(
    lower_wavelength: Optional[float],
    upper_wavelength: Optional[float],
    lower_level: Optional[float],
    upper_level: Optional[float],
    wavelength_range: Optional[Tuple[float, float]],
) -> Tuple[float, float]:
    """
    Resolve wavelength bounds in Angstrom.

    lower_level/upper_level are legacy names from an incomplete earlier draft;
    keep them for compatibility and treat as wavelength bounds.
    """
    if wavelength_range is not None:
        if len(wavelength_range) != 2:
            raise ValueError("wavelength_range must be a 2-tuple (lower, upper) in Angstrom.")
        if (
            lower_wavelength is not None
            or upper_wavelength is not None
            or lower_level is not None
            or upper_level is not None
        ):
            warnings.warn(
                "wavelength_range provided; ignoring lower_wavelength/upper_wavelength/"
                "lower_level/upper_level.",
                RuntimeWarning,
            )
        ll, ul = float(wavelength_range[0]), float(wavelength_range[1])
    elif lower_wavelength is not None and upper_wavelength is not None:
        ll, ul = float(lower_wavelength), float(upper_wavelength)
    elif lower_level is not None and upper_level is not None:
        warnings.warn(
            "lower_level/upper_level are deprecated for ExoMol loading; use "
            "lower_wavelength/upper_wavelength. Treating them as Angstrom bounds.",
            DeprecationWarning,
        )
        ll, ul = float(lower_level), float(upper_level)
    else:
        raise ValueError(
            "Provide either (lower_wavelength, upper_wavelength) or wavelength_range."
        )

    if ll >= ul:
        raise ValueError("lower wavelength bound must be smaller than upper bound.")
    return ll, ul


def load_exomol_linelist(
    species_name: str,
    states_file: Union[str, Path],
    transitions_file: Union[str, Path],
    lower_wavelength: Optional[float] = None,
    upper_wavelength: Optional[float] = None,
    isotopes: Optional[List[Tuple[int, int]]] = None,
    isotopic_abundances: Optional[Dict[int, Dict[int, float]]] = None,
    isotopic_nuclear_spin_degeneracies: Optional[Dict[int, Dict[int, float]]] = None,
    line_strength_cutoff: float = -15.0,
    temperature_line_strength: float = 3500.0,
    verbose: bool = True,
    lower_level: Optional[float] = None,
    upper_level: Optional[float] = None,
    wavelength_range: Optional[Tuple[float, float]] = None,
) -> LineList:
    """
    Load ExoMol molecular linelist, analogous to Korg.load_ExoMol_linelist.

    Parameters
    ----------
    species_name : str
        Molecular species (e.g. "CaH", "40Ca-1H", "H2O").
    states_file : str or Path
        ExoMol .states (or .states.bz2/.gz) file.
    transitions_file : str or Path
        ExoMol .trans (or .trans.bz2/.gz) file.
    lower_wavelength, upper_wavelength : float, optional
        Wavelength bounds in Angstrom.
    isotopes : list[(int, int)], optional
        Explicit isotopes as (atomic_number, mass_number).
    isotopic_abundances : dict, optional
        Isotopic abundance table, defaults to atomic_data.ISOTOPIC_ABUNDANCES.
    isotopic_nuclear_spin_degeneracies : dict, optional
        Optional nuclear-spin degeneracy table for Korg-like correction.
    line_strength_cutoff : float
        Threshold on approximate line strength, default -15.
    temperature_line_strength : float
        Temperature for line-strength approximation in K, default 3500.
    verbose : bool
        Print progress messages.
    lower_level, upper_level : float, optional
        Deprecated legacy aliases; interpreted as wavelength bounds in Angstrom.
    wavelength_range : tuple(float, float), optional
        Alternate wavelength bounds in Angstrom.

    Returns
    -------
    LineList
        ExoMol transitions converted into Jorg LineData entries.
    """
    states_path = Path(states_file)
    transitions_path = Path(transitions_file)
    ll_angstrom, ul_angstrom = _resolve_exomol_wavelength_bounds(
        lower_wavelength,
        upper_wavelength,
        lower_level,
        upper_level,
        wavelength_range,
    )

    if isotopic_abundances is None:
        isotopic_abundances = ISOTOPIC_ABUNDANCES

    species_obj = _resolve_exomol_species(species_name)

    if verbose:
        print(
            f"🔬 Loading ExoMol linelist from {states_path} and {transitions_path}. "
            "This functionality is experimental."
        )
    if "states" not in states_path.name or "trans" not in transitions_path.name:
        warnings.warn(
            f"Input file names ({states_path.name}, {transitions_path.name}) do not "
            "look like states/trans files; verify they are not swapped.",
            RuntimeWarning,
        )

    # Read only required columns to keep memory use manageable.
    raw_transitions = pd.read_csv(
        transitions_path,
        sep=r"\s+",
        header=None,
        comment="#",
        usecols=[0, 1, 2],
        names=["id_upper", "id_lower", "A"],
        compression="infer",
    )
    states = pd.read_csv(
        states_path,
        sep=r"\s+",
        header=None,
        comment="#",
        usecols=[0, 1, 2],
        names=["id", "E_wavenumber", "g"],
        compression="infer",
    )

    upper_states = states.rename(
        columns={"id": "id_upper", "E_wavenumber": "wavenumber_upper", "g": "g_upper"}
    )
    lower_states = states.rename(
        columns={"id": "id_lower", "E_wavenumber": "wavenumber_lower", "g": "g_lower"}
    )

    transitions = (
        raw_transitions.merge(upper_states, on="id_upper", how="left")
        .merge(lower_states, on="id_lower", how="left")
    )

    if transitions["g_lower"].isna().any() or transitions["g_upper"].isna().any():
        raise ValueError(
            f"Some transitions in {transitions_path} could not be mapped to "
            f"states in {states_path}."
        )

    wavenumber = transitions["wavenumber_upper"].to_numpy() - transitions[
        "wavenumber_lower"
    ].to_numpy()
    A_values = transitions["A"].to_numpy()
    g_lower = transitions["g_lower"].to_numpy()
    g_upper = transitions["g_upper"].to_numpy()
    wavenumber_lower = transitions["wavenumber_lower"].to_numpy()

    valid_mask = (
        np.isfinite(wavenumber)
        & np.isfinite(A_values)
        & np.isfinite(g_lower)
        & np.isfinite(g_upper)
        & (wavenumber > 0)
        & (g_lower > 0)
        & (g_upper > 0)
        & (A_values > 0)
    )

    if verbose:
        n_invalid = int((~valid_mask).sum())
        if n_invalid > 0:
            print(f"   Dropping {n_invalid} invalid/non-physical transitions before filtering.")

    wavenumber = wavenumber[valid_mask]
    A_values = A_values[valid_mask]
    g_lower = g_lower[valid_mask]
    g_upper = g_upper[valid_mask]
    wavenumber_lower = wavenumber_lower[valid_mask]

    prefactor = (_ELECTRON_MASS_CGS * _C_CGS) / (
        8.0 * np.pi * np.pi * _ELECTRON_CHARGE_CGS * _ELECTRON_CHARGE_CGS
    )
    f_values = A_values * prefactor * g_upper / (g_lower * (wavenumber * wavenumber))
    log_gf = np.log10(g_lower * f_values)

    if isotopes is None:
        isotopes = _default_exomol_isotopes(species_obj, isotopic_abundances)
        if verbose and isotopes:
            print("   Assuming most abundant isotopes:", isotopes)

    isotopic_correction = _compute_exomol_isotopic_correction(
        isotopes,
        isotopic_abundances,
        isotopic_nuclear_spin_degeneracies,
    )
    log_gf = log_gf + isotopic_correction

    E_lower = _HPLANCK_EV_S * wavenumber_lower * _C_CGS
    wavelength_cm = 1.0 / wavenumber
    wavelength_angstrom = wavelength_cm * 1e8

    region_mask = (wavelength_angstrom > ll_angstrom) & (wavelength_angstrom < ul_angstrom)
    wavelength_cm = wavelength_cm[region_mask]
    wavelength_angstrom = wavelength_angstrom[region_mask]
    log_gf = log_gf[region_mask]
    E_lower = E_lower[region_mask]

    if wavelength_cm.size == 0:
        metadata = {
            "format": "exomol",
            "species": species_name,
            "n_lines": 0,
            "n_parsed": 0,
            "states_file": str(states_path),
            "transitions_file": str(transitions_path),
            "wavelength_bounds_angstrom": (ll_angstrom, ul_angstrom),
        }
        return LineList([], metadata)

    # Korg-like strength estimate:
    # line.log_gf + log10(line.wl) - log10(e) * E_lower / (k_B * T)
    approx_strength = (
        log_gf
        + np.log10(wavelength_cm)
        - _LOG10_E * E_lower / (_KBOLTZ_EV * temperature_line_strength)
    )
    strength_mask = approx_strength > line_strength_cutoff

    if verbose:
        removed = int((~strength_mask).sum())
        total = int(strength_mask.size)
        percent = int(round(100 * removed / total)) if total > 0 else 0
        print(
            f"   Removed {removed} lines with strength below {line_strength_cutoff} "
            f"at T={temperature_line_strength} K out of {total} total ({percent}%)."
        )

    wavelength_cm = wavelength_cm[strength_mask]
    wavelength_angstrom = wavelength_angstrom[strength_mask]
    log_gf = log_gf[strength_mask]
    E_lower = E_lower[strength_mask]

    lines = [
        create_line_data(
            wavelength=float(wl_cm),
            species=species_obj,
            log_gf=float(lgf),
            E_lower=float(elow),
            wavelength_unit="cm",
        )
        for wl_cm, lgf, elow in zip(wavelength_cm, log_gf, E_lower)
    ]
    lines.sort(key=lambda line: line.wavelength)

    metadata = {
        "format": "exomol",
        "species": species_name,
        "n_lines": len(lines),
        "n_parsed": len(lines),
        "states_file": str(states_path),
        "transitions_file": str(transitions_path),
        "wavelength_bounds_angstrom": (ll_angstrom, ul_angstrom),
        "line_strength_cutoff": line_strength_cutoff,
        "temperature_line_strength": temperature_line_strength,
    }
    if verbose:
        print(
            f"   Loaded {len(lines)} ExoMol lines for {species_name} in "
            f"{ll_angstrom:.2f}-{ul_angstrom:.2f} A."
        )
    return LineList(lines, metadata)
   
