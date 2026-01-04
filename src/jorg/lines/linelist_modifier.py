"""
Log(gf) modification for spectral linelists.

This module provides interactive tools for adjusting oscillator strengths
(log gf values) in spectral linelists while preserving all other line properties.
"""

import copy
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from pathlib import Path
from dataclasses import replace

from .datatypes import Line, LineData, Species
from .linelist import LineList, read_linelist
from ..utils.wavelength_utils import air_to_vacuum, vacuum_to_air


class LogGFModifier:
    """
    Interactive log(gf) modification for spectral lines.

    This class allows you to adjust oscillator strengths for individual lines
    while keeping the original linelist immutable. All modifications are tracked
    and can be reset or exported.

    Parameters
    ----------
    linelist : List[Line] or List[LineData] or LineList
        The original linelist to modify
    wavelength_tolerance : float, optional
        Tolerance in Angstroms for matching lines by wavelength (default: 0.01)
    vacuum_wavelengths : bool, optional
        If True, treat input wavelengths as vacuum wavelengths. If False (default),
        treat as air wavelengths and convert internally.

    Examples
    --------
    >>> from jorg.lines.linelist import read_linelist
    >>> from jorg.lines.linelist_modifier import LogGFModifier
    >>>
    >>> linelist = read_linelist('lines.vald', format='vald')
    >>> modifier = LogGFModifier(linelist)
    >>>
    >>> # Adjust a single line by 0.1 dex
    >>> modifier.adjust_line(5001.2, delta_loggf=0.1)
    >>>
    >>> # Set absolute log(gf) value
    >>> modifier.set_line(5001.2, new_loggf=-1.5)
    >>>
    >>> # Get modified linelist
    >>> modified = modifier.apply_modifications()
    >>>
    >>> # Save to file
    >>> modifier.to_file('modified_lines.vald')

    Notes
    -----
    - Wavelengths are assumed to be in air (observed) wavelengths by default
    - Set vacuum_wavelengths=True if working with vacuum wavelengths
    - The original linelist is never modified
    - Modifications can be reset using reset_line() or reset_all()
    """

    def __init__(
        self,
        linelist: Union[List[Line], List[LineData], LineList],
        wavelength_tolerance: float = 0.01,
        vacuum_wavelengths: bool = False
    ):
        # Store original linelist as immutable copy
        if isinstance(linelist, LineList):
            self._original_lines = copy.deepcopy(linelist.lines)
            self._metadata = linelist.metadata
        else:
            self._original_lines = copy.deepcopy(linelist)
            self._metadata = {}

        self._wavelength_tolerance = wavelength_tolerance
        self._vacuum_wavelengths = vacuum_wavelengths

        # Track modifications: {index_in_original: (old_loggf, new_loggf)}
        self._modifications: Dict[int, Tuple[float, float]] = {}

        # Build wavelength index for fast lookup
        self._build_wavelength_index()

    def _build_wavelength_index(self):
        """Build index mapping wavelengths to line indices."""
        self._wl_index: Dict[float, int] = {}

        for i, line in enumerate(self._original_lines):
            # Convert to Angstroms (stored in cm)
            if isinstance(line, Line):
                wl_angstrom = line.wl * 1e8
            else:  # LineData
                wl_angstrom = line.wavelength * 1e8

            # Convert to air if vacuum
            if self._vacuum_wavelengths:
                wl_air = vacuum_to_air(wl_angstrom)
            else:
                wl_air = wl_angstrom

            # Store with this wavelength (warn if duplicate)
            if wl_air in self._wl_index:
                existing_idx = self._wl_index[wl_air]
                other_wl = self._original_lines[existing_idx].wavelength * 1e8 if isinstance(
                    self._original_lines[existing_idx], Line) else self._original_lines[
                    existing_idx].wavelength * 1e8
                # Only warn if actually different lines
                if abs(wl_air - other_wl) > 1e-4:
                    import warnings
                    warnings.warn(f"Multiple lines at {wl_air:.3f} Å - using last occurrence")
            self._wl_index[wl_air] = i

    def _find_line_index(self, wavelength: float) -> Optional[int]:
        """
        Find line index by wavelength.

        Parameters
        ----------
        wavelength : float
            Wavelength in Angstroms (air unless vacuum_wavelengths=True)

        Returns
        -------
        int or None
            Index of line, or None if not found within tolerance
        """
        # Direct match
        if wavelength in self._wl_index:
            return self._wl_index[wavelength]

        # Search within tolerance
        for wl, idx in self._wl_index.items():
            if abs(wl - wavelength) <= self._wavelength_tolerance:
                return idx

        return None

    def _get_line_wavelength(self, line: Union[Line, LineData]) -> float:
        """Get line wavelength in Angstroms (air)."""
        if isinstance(line, Line):
            wl_angstrom = line.wl * 1e8
        else:
            wl_angstrom = line.wavelength * 1e8

        if self._vacuum_wavelengths:
            return vacuum_to_air(wl_angstrom)
        return wl_angstrom

    def _get_line_species_str(self, line: Union[Line, LineData]) -> str:
        """Get species string for a line."""
        if isinstance(line, Line):
            return str(line.species)
        else:
            # For LineData, decode species integer
            species_id = line.species
            if not isinstance(species_id, (int, np.integer)):
                return str(species_id)
            if species_id < 100:
                from .atomic_data import get_atomic_symbol
                return f"{get_atomic_symbol(species_id)} I"
            else:
                element_id = species_id // 100
                ionization = species_id % 100
                from .atomic_data import get_atomic_symbol
                symbol = get_atomic_symbol(element_id)
                return f"{symbol} {ionization + 1}"

    def adjust_line(
        self,
        wavelength: float,
        delta_loggf: float,
        relative: bool = True
    ) -> Optional[Union[Line, LineData]]:
        """
        Adjust log(gf) for a single line by wavelength.

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms (air unless vacuum_wavelengths=True)
        delta_loggf : float
            Amount to adjust log(gf). Can be positive or negative.
        relative : bool, optional
            If True (default), delta_loggf is added to current value.
            If False, delta_loggf is multiplied (e.g., 1.1 = +10%).

        Returns
        -------
        Line or LineData or None
            The modified line object, or None if line not found

        Raises
        ------
        ValueError
            If wavelength is not found in linelist

        Examples
        --------
        >>> # Increase log(gf) by 0.1 dex (26% stronger line)
        >>> modifier.adjust_line(5001.2, delta_loggf=0.1)
        >>>
        >>> # Decrease by 0.05 dex
        >>> modifier.adjust_line(5001.2, delta_loggf=-0.05)
        >>>
        >>> # Multiply by factor (10% stronger)
        >>> modifier.adjust_line(5001.2, delta_loggf=1.1, relative=False)
        """
        idx = self._find_line_index(wavelength)

        if idx is None:
            raise ValueError(
                f"No line found at {wavelength:.3f} Å "
                f"(tolerance: ±{self._wavelength_tolerance:.3f} Å)"
            )

        line = self._original_lines[idx]
        old_loggf = line.log_gf if isinstance(line, Line) else line.log_gf

        if relative:
            new_loggf = old_loggf + delta_loggf
        else:
            # Multiplicative adjustment on gf (not log_gf)
            # gf_new = gf_old * delta_loggf
            # log_gf_new = log10(gf_old * delta_loggf)
            new_loggf = np.log10(10**old_loggf * delta_loggf)

        # Store modification
        if idx in self._modifications:
            # If already modified, use original value
            old_loggf = self._modifications[idx][0]

        self._modifications[idx] = (old_loggf, new_loggf)

        # Return modified line
        return self._get_modified_line(idx)

    def set_line(self, wavelength: float, new_loggf: float) -> Optional[Union[Line, LineData]]:
        """
        Set absolute log(gf) value for a line.

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms
        new_loggf : float
            New log(gf) value to set

        Returns
        -------
        Line or LineData or None
            The modified line object, or None if line not found

        Examples
        --------
        >>> modifier.set_line(5001.2, -1.45)
        """
        idx = self._find_line_index(wavelength)

        if idx is None:
            raise ValueError(
                f"No line found at {wavelength:.3f} Å "
                f"(tolerance: ±{self._wavelength_tolerance:.3f} Å)"
            )

        line = self._original_lines[idx]
        old_loggf = line.log_gf if isinstance(line, Line) else line.log_gf

        self._modifications[idx] = (old_loggf, new_loggf)

        return self._get_modified_line(idx)

    def reset_line(self, wavelength: float) -> bool:
        """
        Reset a line to its original log(gf) value.

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms

        Returns
        -------
        bool
            True if line was reset, False if line not found or not modified

        Examples
        --------
        >>> modifier.reset_line(5001.2)
        """
        idx = self._find_line_index(wavelength)

        if idx is None or idx not in self._modifications:
            return False

        del self._modifications[idx]
        return True

    def reset_all(self):
        """Reset all modifications."""
        self._modifications.clear()

    def get_line(self, wavelength: float) -> Optional[Union[Line, LineData]]:
        """
        Get a line (modified if applicable, otherwise original).

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms

        Returns
        -------
        Line or LineData or None
            The line object (with modifications applied), or None if not found
        """
        idx = self._find_line_index(wavelength)

        if idx is None:
            return None

        return self._get_modified_line(idx)

    def get_original_line(self, wavelength: float) -> Optional[Union[Line, LineData]]:
        """
        Get original line (without modifications).

        Parameters
        ----------
        wavelength : float
            Line wavelength in Angstroms

        Returns
        -------
        Line or LineData or None
            The original unmodified line, or None if not found
        """
        idx = self._find_line_index(wavelength)

        if idx is None:
            return None

        return self._original_lines[idx]

    def _get_modified_line(self, idx: int) -> Union[Line, LineData]:
        """Get modified line at index."""
        line = self._original_lines[idx]

        if idx not in self._modifications:
            return line

        new_loggf = self._modifications[idx][1]

        if isinstance(line, Line):
            return replace(line, log_gf=new_loggf)
        else:
            # LineData is immutable, create new instance
            return line._replace(log_gf=new_loggf)

    def get_modifications(self) -> Dict[float, Dict[str, float]]:
        """
        Get all modifications made.

        Returns
        -------
        dict
            Dictionary mapping wavelengths to modification details:
            {wavelength: {'old': old_loggf, 'new': new_loggf, 'delta': delta}}

        Examples
        --------
        >>> mods = modifier.get_modifications()
        >>> for wl, details in mods.items():
        ...     print(f"{wl:.2f} Å: {details['old']:.3f} → {details['new']:.3f}")
        """
        result = {}

        for idx, (old_val, new_val) in self._modifications.items():
            line = self._original_lines[idx]
            wl = self._get_line_wavelength(line)

            result[wl] = {
                'old': old_val,
                'new': new_val,
                'delta': new_val - old_val
            }

        return result

    def apply_modifications(self) -> List[Union[Line, LineData]]:
        """
        Apply all modifications and return modified linelist.

        Returns
        -------
        list of Line or LineData
            Modified linelist with all adjustments applied

        Examples
        --------
        >>> modified_linelist = modifier.apply_modifications()
        >>> result = synthesize(atm, modified_linelist, A_X, wavelengths)
        """
        modified = []

        for i in range(len(self._original_lines)):
            modified.append(self._get_modified_line(i))

        return modified

    def as_linelist(self) -> LineList:
        """
        Return modified linelist as a LineList object.

        Returns
        -------
        LineList
            Modified linelist wrapped in LineList container

        Examples
        --------
        >>> modified_ll = modifier.as_linelist()
        >>> filtered = modified_ll.filter_by_wavelength(5000, 5010)
        """
        return LineList(self.apply_modifications(), self._metadata)

    def adjust_by_element(
        self,
        element: str,
        ion: int = 1,
        delta_loggf: float = 0.0,
        min_wavelength: Optional[float] = None,
        max_wavelength: Optional[float] = None
    ) -> int:
        """
        Adjust all lines of a specific element/ion.

        Parameters
        ----------
        element : str
            Element symbol (e.g., 'Fe', 'Mg', 'Si')
        ion : int, optional
            Ionization state (1=neutral, 2=singly ionized, etc.). Default is 1.
        delta_loggf : float, optional
            Amount to adjust log(gf). Default is 0 (no change).
        min_wavelength : float, optional
            Minimum wavelength in Angstroms to filter lines
        max_wavelength : float, optional
            Maximum wavelength in Angstroms to filter lines

        Returns
        -------
        int
            Number of lines modified

        Examples
        --------
        >>> # Increase all Fe I lines by 0.05 dex
        >>> n_modified = modifier.adjust_by_element('Fe', ion=1, delta_loggf=0.05)
        >>> print(f"Modified {n_modified} Fe I lines")
        >>>
        >>> # Adjust only in specific range
        >>> modifier.adjust_by_element('Mg', ion=1, delta_loggf=0.1,
        ...                           min_wavelength=5000, max_wavelength=5200)
        """
        count = 0

        for i, line in enumerate(self._original_lines):
            # Check wavelength range
            wl = self._get_line_wavelength(line)
            if min_wavelength is not None and wl < min_wavelength:
                continue
            if max_wavelength is not None and wl > max_wavelength:
                continue

            # Check species
            species_str = self._get_line_species_str(line)
            target_species = f"{element} {ion}"

            if target_species in species_str:
                old_loggf = line.log_gf if isinstance(line, Line) else line.log_gf
                new_loggf = old_loggf + delta_loggf

                # Only track if actually changed
                if i not in self._modifications:
                    self._modifications[i] = (old_loggf, new_loggf)
                else:
                    # Use original value if already modified
                    orig_loggf = self._modifications[i][0]
                    self._modifications[i] = (orig_loggf, orig_loggf + delta_loggf)

                count += 1

        return count

    def filter_by_wavelength(self, wl_min: float, wl_max: float) -> 'LogGFModifier':
        """
        Create a new modifier with only lines in wavelength range.

        Parameters
        ----------
        wl_min : float
            Minimum wavelength in Angstroms
        wl_max : float
            Maximum wavelength in Angstroms

        Returns
        -------
        LogGFModifier
            New modifier with filtered linelist

        Examples
        --------
        >>> # Work only with 5000-5010 Å range
        >>> modifier_subset = modifier.filter_by_wavelength(5000, 5010)
        """
        filtered_lines = []

        for line in self._original_lines:
            wl = self._get_line_wavelength(line)
            if wl_min <= wl <= wl_max:
                filtered_lines.append(line)

        # Create new modifier with filtered lines
        new_modifier = LogGFModifier(
            filtered_lines,
            wavelength_tolerance=self._wavelength_tolerance,
            vacuum_wavelengths=self._vacuum_wavelengths
        )

        return new_modifier

    def to_file(
        self,
        filename: Union[str, Path],
        format: str = 'vald',
        include_metadata: bool = True
    ):
        """
        Save modified linelist to file.

        Parameters
        ----------
        filename : str or Path
            Output file path
        format : str, optional
            Output format ('vald', 'kurucz', 'moog'). Default is 'vald'.
        include_metadata : bool, optional
            Include modification metadata in header. Default is True.

        Examples
        --------
        >>> modifier.to_file('modified_lines.vald', format='vald')
        """
        filename = Path(filename)
        modified_lines = self.apply_modifications()

        if format.lower() == 'vald':
            self._write_vald(filename, modified_lines, include_metadata)
        elif format.lower() == 'kurucz':
            self._write_kurucz(filename, modified_lines, include_metadata)
        elif format.lower() == 'moog':
            self._write_moog(filename, modified_lines, include_metadata)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _write_vald(
        self,
        filename: Path,
        lines: List[Union[Line, LineData]],
        include_metadata: bool
    ):
        """Write VALD format linelist."""
        with open(filename, 'w') as f:
            # Write header
            f.write("' Format: VALD extracted stellar\n")

            if include_metadata and self._modifications:
                f.write("' Modified by LogGFModifier\n")
                f.write(f"' Number of modified lines: {len(self._modifications)}\n")

            for line in lines:
                # Convert to VALD format
                if isinstance(line, Line):
                    wl_angstrom = line.wl * 1e8
                    species_str = str(line.species)
                    log_gf = line.log_gf
                    e_lower = line.E_lower
                    gamma_rad = line.gamma_rad
                    gamma_stark = line.gamma_stark
                    vdw1, vdw2 = line.vdW
                else:
                    wl_angstrom = line.wavelength * 1e8
                    # Need to convert species int to string
                    species_str = self._get_line_species_str(line)
                    log_gf = line.log_gf
                    e_lower = line.E_lower
                    gamma_rad = getattr(line, 'gamma_rad', 0.0)
                    gamma_stark = getattr(line, 'gamma_stark', 0.0)
                    vdw1 = getattr(line, 'vdw_param1', 0.0)
                    vdw2 = getattr(line, 'vdw_param2', -1.0)

                # VALD format (simplified - adjust as needed for actual format)
                f.write(f"' {wl_angstrom:.4f}, {species_str}, {log_gf:.3f}, "
                       f"{e_lower:.4f}, {gamma_rad:.3e}, {gamma_stark:.3e}, "
                       f"{vdw1:.3e}, {vdw2:.1f}\n")

    def _write_kurucz(
        self,
        filename: Path,
        lines: List[Union[Line, LineData]],
        include_metadata: bool
    ):
        """Write Kurucz format linelist."""
        with open(filename, 'w') as f:
            if include_metadata and self._modifications:
                f.write(f"# Modified by LogGFModifier: {len(self._modifications)} lines\n")

            for line in lines:
                if isinstance(line, Line):
                    wl_angstrom = line.wl * 1e8
                    log_gf = line.log_gf
                    e_lower = line.E_lower
                else:
                    wl_angstrom = line.wavelength * 1e8
                    log_gf = line.log_gf
                    e_lower = line.E_lower

                # Kurucz format: wavelength, log_gf, element_code, excitation_potential
                f.write(f"{wl_angstrom:.3f} {log_gf:6.3f} {e_lower:6.3f}\n")

    def _write_moog(
        self,
        filename: Path,
        lines: List[Union[Line, LineData]],
        include_metadata: bool
    ):
        """Write MOOG format linelist."""
        with open(filename, 'w') as f:
            if include_metadata and self._modifications:
                f.write(f" modified by LogGFModifier: {len(self._modifications)} lines\n")

            for line in lines:
                if isinstance(line, Line):
                    wl_angstrom = line.wl * 1e8
                    log_gf = line.log_gf
                    e_lower = line.E_lower
                    species_str = str(line.species)
                else:
                    wl_angstrom = line.wavelength * 1e8
                    log_gf = line.log_gf
                    e_lower = line.E_lower
                    species_str = self._get_line_species_str(line)

                # MOOG format
                f.write(f"{wl_angstrom:.3f} {log_gf:6.3f} {e_lower:6.3f} {species_str}\n")

    def summary(self) -> str:
        """
        Get a summary of modifications.

        Returns
        -------
        str
            Human-readable summary of all modifications

        Examples
        --------
        >>> print(modifier.summary())
        """
        if not self._modifications:
            return "No modifications applied."

        lines = []
        lines.append(f"Total modifications: {len(self._modifications)}")
        lines.append("-" * 60)

        for wl, details in sorted(self.get_modifications().items()):
            lines.append(
                f"{wl:8.2f} Å: log(gf) {details['old']:7.3f} → "
                f"{details['new']:7.3f} (Δ{details['delta']:+6.3f})"
            )

        return "\n".join(lines)

    def __len__(self) -> int:
        """Return number of lines in linelist."""
        return len(self._original_lines)

    def __repr__(self) -> str:
        return (f"LogGFModifier(n_lines={len(self)}, "
                f"n_modified={len(self._modifications)})")


def modify_log_gf(
    linelist: Union[List[Line], List[LineData], LineList],
    adjustments: Dict[float, float],
    wavelength_tolerance: float = 0.01
) -> List[Union[Line, LineData]]:
    """
    Convenience function to apply multiple log(gf) adjustments at once.

    Parameters
    ----------
    linelist : list or LineList
        Original linelist
    adjustments : dict
        Dictionary of {wavelength: delta_loggf} adjustments
    wavelength_tolerance : float, optional
        Tolerance for matching wavelengths in Angstroms

    Returns
    -------
    list
        Modified linelist

    Examples
    --------
    >>> from jorg.lines.linelist_modifier import modify_log_gf
    >>>
    >>> modified = modify_log_gf(
    ...     linelist,
    ...     adjustments={5001.2: 0.1, 5005.8: -0.05, 5010.3: 0.2}
    ... )
    """
    modifier = LogGFModifier(linelist, wavelength_tolerance=wavelength_tolerance)

    for wl, delta in adjustments.items():
        try:
            modifier.adjust_line(wl, delta)
        except ValueError:
            # Line not found - skip
            import warnings
            warnings.warn(f"Line at {wl} Å not found, skipping")

    return modifier.apply_modifications()


def scale_log_gf(
    linelist: Union[List[Line], List[LineData], LineList],
    scaling_factor: float,
    species_filter: Optional[str] = None
) -> List[Union[Line, LineData]]:
    """
    Scale log(gf) values for all lines or filtered by species.

    Parameters
    ----------
    linelist : list or LineList
        Original linelist
    scaling_factor : float
        Factor to multiply gf values (not log_gf directly)
        E.g., 1.1 = +10% on gf, ~+0.04 dex on log(gf)
    species_filter : str, optional
        Only scale lines of this species (e.g., 'Fe I')

    Returns
    -------
    list
        Modified linelist

    Examples
    --------
    >>> # Make all lines 10% stronger
    >>> modified = scale_log_gf(linelist, 1.1)
    >>>
    >>> # Make Fe I lines 20% weaker
    >>> modified = scale_log_gf(linelist, 0.8, species_filter='Fe I')
    """
    modifier = LogGFModifier(linelist)

    if species_filter is None:
        # Scale all lines
        for i, line in enumerate(modifier._original_lines):
            old_loggf = line.log_gf if isinstance(line, Line) else line.log_gf
            # gf_new = gf_old * scaling_factor
            # log_gf_new = log10(gf_old * scaling_factor)
            new_loggf = np.log10(10**old_loggf * scaling_factor)
            modifier._modifications[i] = (old_loggf, new_loggf)
    else:
        # Parse species filter (e.g., "Fe I" -> element="Fe", ion=1)
        parts = species_filter.split()
        element = parts[0]
        ion = int(parts[1]) if len(parts) > 1 else 1

        # Convert Roman numerals to integer if needed
        roman_to_int = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
        if parts[1] in roman_to_int:
            ion = roman_to_int[parts[1]]

        modifier.adjust_by_element(element, ion=ion, delta_loggf=np.log10(scaling_factor))

    return modifier.apply_modifications()
