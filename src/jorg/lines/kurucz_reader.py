"""
Kurucz Line List Reader

Implements support for reading Kurucz atomic and molecular line lists,
matching Korg.jl's functionality for broader line list compatibility.

Reference: Korg.jl/src/linelist.jl - read_kurucz_linelist()
Format documentation: http://kurucz.harvard.edu/linelists.html
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import re

from .datatypes import Line
from ..statmech.species import Species


class KuruczLineReader:
    """
    Reader for Kurucz format line lists.
    
    Kurucz format is widely used in stellar spectroscopy, especially
    for comprehensive atomic line lists and molecular bands.
    """
    
    # Kurucz element codes (periodic table position)
    ELEMENT_CODES = {
        1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
        9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti',
        23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu',
        30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr',
        37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc',
        44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La',
        58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd',
        65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu',
        72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt',
        79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
        86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U'
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize Kurucz line reader.
        
        Parameters
        ----------
        verbose : bool
            Print parsing information
        """
        self.verbose = verbose
        
    def read_linelist(self, 
                     filename: str,
                     wavelength_range: Optional[Tuple[float, float]] = None,
                     strength_threshold: float = -10.0) -> List[Line]:
        """
        Read Kurucz format line list.
        
        Parameters
        ----------
        filename : str
            Path to Kurucz line list file
        wavelength_range : tuple, optional
            (lambda_min, lambda_max) in Angstroms to filter lines
        strength_threshold : float
            Minimum log(gf) to include (default: -10.0)
            
        Returns
        -------
        list
            List of Line objects
        """
        if self.verbose:
            print(f"Reading Kurucz line list: {filename}")
            
        lines = []
        n_read = 0
        n_skipped = 0
        
        with open(filename, 'r') as f:
            for line_str in f:
                # Skip comments and empty lines
                if line_str.startswith('#') or line_str.strip() == '':
                    continue
                    
                try:
                    line = self._parse_kurucz_line(line_str)
                    
                    if line is not None:
                        n_read += 1
                        
                        # Apply filters
                        if wavelength_range is not None:
                            wl_ang = line.wl * 1e8  # Convert cm to Angstrom
                            if wl_ang < wavelength_range[0] or wl_ang > wavelength_range[1]:
                                n_skipped += 1
                                continue
                                
                        if line.log_gf < strength_threshold:
                            n_skipped += 1
                            continue
                            
                        lines.append(line)
                        
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Failed to parse line: {e}")
                    n_skipped += 1
                    
        if self.verbose:
            print(f"  Read {n_read} lines, kept {len(lines)}, skipped {n_skipped}")
            
        return lines
    
    def _parse_kurucz_line(self, line_str: str) -> Optional[Line]:
        """
        Parse a single Kurucz format line.
        
        Kurucz format (columns):
        1-11:   Wavelength in Angstroms (air or vacuum)
        12-18:  log(gf)
        19-24:  Element code (XX.YY where XX=element, YY=ionization*100)
        25-36:  E_lower in cm^-1
        37-48:  E_upper in cm^-1
        49-54:  J_lower
        55-60:  J_upper
        61-66:  Lande g factor (lower)
        67-72:  Lande g factor (upper)
        73-79:  Radiative damping (log gamma_rad)
        80-86:  Stark damping (log gamma_stark)
        87-93:  van der Waals damping (log gamma_vdW)
        
        Parameters
        ----------
        line_str : str
            Line string from Kurucz file
            
        Returns
        -------
        Line or None
            Parsed Line object or None if parsing fails
        """
        # Ensure line is long enough
        if len(line_str) < 48:
            return None
            
        try:
            # Parse wavelength (columns 1-11)
            wavelength_str = line_str[0:11].strip()
            if not wavelength_str:
                return None
            wavelength_ang = float(wavelength_str)
            
            # Convert to cm (Kurucz uses Angstroms)
            wavelength_cm = wavelength_ang * 1e-8
            
            # Parse log(gf) (columns 12-18)
            log_gf_str = line_str[11:18].strip()
            if not log_gf_str:
                return None
            log_gf = float(log_gf_str)
            
            # Parse element code (columns 19-24)
            element_str = line_str[18:24].strip()
            if not element_str:
                return None
            element_code = float(element_str)
            
            # Decode element and ionization
            element_num = int(element_code)
            ionization = int((element_code - element_num) * 100 + 0.5)
            
            # Handle molecules (element_num > 100)
            if element_num > 100:
                # Molecular line - skip for now
                # TODO: Implement molecular line parsing
                return None
                
            # Create Species object
            species = Species.from_atomic_number(element_num, ionization)
            
            # Parse energy levels (columns 25-48)
            E_lower_str = line_str[24:36].strip()
            E_upper_str = line_str[36:48].strip()
            
            if not E_lower_str or not E_upper_str:
                return None
                
            E_lower_cm = float(E_lower_str)  # in cm^-1
            E_upper_cm = float(E_upper_str)  # in cm^-1
            
            # Convert to eV
            cm_to_eV = 1.23984198e-4  # hc in eV*cm
            E_lower_eV = E_lower_cm * cm_to_eV
            
            # Parse damping parameters if available
            gamma_rad = 0.0
            gamma_stark = 0.0
            vdW_param = (0.0, -1)  # Default: no enhancement
            
            if len(line_str) >= 79:
                # Radiative damping (columns 73-79)
                gamma_rad_str = line_str[72:79].strip()
                if gamma_rad_str:
                    try:
                        log_gamma_rad = float(gamma_rad_str)
                        gamma_rad = 10**log_gamma_rad
                    except:
                        gamma_rad = 1e8  # Default ~10^8 s^-1
                        
            if len(line_str) >= 86:
                # Stark damping (columns 80-86)
                gamma_stark_str = line_str[79:86].strip()
                if gamma_stark_str:
                    try:
                        log_gamma_stark = float(gamma_stark_str)
                        # Kurucz gives log(gamma/N_e) at 10000K, 1e16 cm^-3
                        gamma_stark = 10**(log_gamma_stark - 16)
                    except:
                        gamma_stark = 0.0
                        
            if len(line_str) >= 93:
                # van der Waals damping (columns 87-93)
                vdW_str = line_str[86:93].strip()
                if vdW_str:
                    try:
                        log_gamma_vdW = float(vdW_str)
                        # Convert to enhancement factor format
                        vdW_param = (log_gamma_vdW + 6.0, -1)  # Kurucz uses log(gamma/N_H)
                    except:
                        vdW_param = (0.0, -1)
            
            # Create Line object matching Korg.jl structure
            line = Line(
                wl=wavelength_cm,
                log_gf=log_gf,
                species=species,
                E_lower=E_lower_eV,
                gamma_rad=gamma_rad,
                gamma_stark=gamma_stark,
                vdW=vdW_param
            )
            
            return line
            
        except Exception as e:
            if self.verbose:
                print(f"    Parse error: {e}")
            return None
    
    def read_molecular_linelist(self, filename: str) -> List[Line]:
        """
        Read Kurucz molecular line list.
        
        Molecular lines have different format and require special handling.
        
        Parameters
        ----------
        filename : str
            Path to molecular line list
            
        Returns
        -------
        list
            List of molecular Line objects
        """
        # TODO: Implement molecular line parsing
        # Kurucz molecular format is complex with band heads,
        # rotational structure, etc.
        if self.verbose:
            print(f"Molecular line reading not yet implemented for: {filename}")
        return []


def read_kurucz_linelist(filename: str,
                        wavelength_range: Optional[Tuple[float, float]] = None,
                        verbose: bool = False) -> List[Line]:
    """
    Convenience function to read Kurucz line list.
    
    Parameters
    ----------
    filename : str
        Path to Kurucz line list file
    wavelength_range : tuple, optional
        (lambda_min, lambda_max) in Angstroms
    verbose : bool
        Print parsing information
        
    Returns
    -------
    list
        List of Line objects
    """
    reader = KuruczLineReader(verbose=verbose)
    return reader.read_linelist(filename, wavelength_range)


def compare_line_formats():
    """
    Compare VALD vs Kurucz line formats.
    """
    print("=== LINE LIST FORMAT COMPARISON ===")
    print()
    
    print("VALD Format:")
    print("  - Human-readable with metadata headers")
    print("  - Wavelengths in Angstroms (air/vacuum)")
    print("  - Species as text (e.g., 'Fe 1', 'Fe 2')")
    print("  - Extensive damping parameters")
    print("  - Isotope information")
    print("  - References for each line")
    print()
    
    print("Kurucz Format:")
    print("  - Fixed-column format")
    print("  - Wavelengths in Angstroms")
    print("  - Species as numeric codes (26.00 = Fe I, 26.01 = Fe II)")
    print("  - Energy levels in cm^-1")
    print("  - Damping parameters in log scale")
    print("  - Compact but less metadata")
    print()
    
    print("Example Kurucz line (Fe I at 5000 Å):")
    print(" 5000.0000 -1.234 26.00  12345.67  32345.67  4.5  3.5  1.23  1.23  8.50 -6.00 -7.50")
    print("  ^         ^      ^     ^         ^         ^    ^    ^     ^     ^     ^     ^")
    print("  |         |      |     |         |         |    |    |     |     |     |     |")
    print("  wl      log_gf species E_low    E_upp    J_low J_upp g_low g_upp rad  stark vdW")
    print()
    
    print("Advantages of Kurucz format:")
    print("  ✓ Compact file size")
    print("  ✓ Fast parsing")
    print("  ✓ Extensive atomic line coverage")
    print("  ✓ Standard in many stellar codes")
    print()
    
    print("Advantages of VALD format:")
    print("  ✓ More complete metadata")
    print("  ✓ Better documented parameters")
    print("  ✓ Isotope-specific lines")
    print("  ✓ Quality indicators")


if __name__ == "__main__":
    compare_line_formats()
    
    # Test parsing with synthetic Kurucz line
    print("\n=== TEST KURUCZ PARSING ===")
    
    # Create synthetic Kurucz line
    test_line = " 5000.0000 -1.234 26.00  12345.67  32345.67  4.5  3.5  1.23  1.23  8.50 -6.00 -7.50"
    
    reader = KuruczLineReader(verbose=True)
    parsed = reader._parse_kurucz_line(test_line)
    
    if parsed:
        print(f"Successfully parsed:")
        print(f"  Wavelength: {parsed.wl * 1e8:.2f} Å")
        print(f"  log(gf): {parsed.log_gf:.3f}")
        print(f"  Species: {parsed.species}")
        print(f"  E_lower: {parsed.E_lower:.3f} eV")
    else:
        print("Failed to parse test line")