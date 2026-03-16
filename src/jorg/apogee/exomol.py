"""ExoMol asset registry and download helpers for APOGEE augmentation."""

from __future__ import annotations

import bz2
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import urllib.request

from ..lines.linelist import load_exomol_linelist


@dataclass(frozen=True)
class ExoMolAsset:
    """Definition of one ExoMol molecule line-list asset."""

    species_name: str
    states_url: str
    transitions_url: str
    states_filename: str
    transitions_filename: str
    lower_wavelength: float = 15100.0
    upper_wavelength: float = 17000.0
    line_strength_cutoff: float = -15.0
    temperature_line_strength: float = 3500.0

    def local_paths(self, root: Path) -> tuple[Path, Path]:
        species_root = root / self.species_name
        return _korg_ready_path(species_root / self.states_filename), _korg_ready_path(
            species_root / self.transitions_filename
        )

    def archive_paths(self, root: Path) -> tuple[Path, Path]:
        species_root = root / self.species_name
        return species_root / self.states_filename, species_root / self.transitions_filename


def _korg_ready_path(path: Path) -> Path:
    if path.suffix == ".bz2":
        return path.with_suffix("")
    return path


def default_exomol_added_lines_path(root: Path) -> Path:
    """Return the default CSV report path for ExoMol-only augmentation lines."""
    root = Path(root)
    if root.name == "exomol" and root.parent.name == "external":
        workspace_root = root.parent.parent
        if workspace_root.name == "apogee":
            return workspace_root / "output" / "exomol_aug_added_lines.csv"
        return workspace_root / "exomol_aug_added_lines.csv"
    return root / "exomol_aug_added_lines.csv"


def default_exomol_assets() -> dict[str, ExoMolAsset]:
    """Return the default FeH/CaH augmentation assets."""
    return {
        "CaH": ExoMolAsset(
            species_name="CaH",
            states_url="https://exomol.com/db/CaH/40Ca-1H/XAB/40Ca-1H__XAB.states.bz2",
            transitions_url="https://exomol.com/db/CaH/40Ca-1H/XAB/40Ca-1H__XAB.trans.bz2",
            states_filename="40Ca-1H__XAB.states.bz2",
            transitions_filename="40Ca-1H__XAB.trans.bz2",
        ),
        "FeH": ExoMolAsset(
            species_name="FeH",
            states_url="https://exomol.com/db/FeH/56Fe-1H/MoLLIST/56Fe-1H__MoLLIST.states.bz2",
            transitions_url="https://exomol.com/db/FeH/56Fe-1H/MoLLIST/56Fe-1H__MoLLIST.trans.bz2",
            states_filename="56Fe-1H__MoLLIST.states.bz2",
            transitions_filename="56Fe-1H__MoLLIST.trans.bz2",
        ),
    }


def download_file(url: str, destination: Path, overwrite: bool = False) -> Path:
    """Download one remote file to the requested destination path."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    with urllib.request.urlopen(url, timeout=120) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    return destination


def _prepare_korg_text_file(source_path: Path, destination: Path, overwrite: bool = False) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination
    if source_path == destination:
        return destination
    if source_path.suffix == ".bz2":
        with bz2.open(source_path, "rb") as source, destination.open("wb") as handle:
            shutil.copyfileobj(source, handle)
        return destination
    shutil.copyfile(source_path, destination)
    return destination


def ensure_exomol_asset(asset: ExoMolAsset, root: Path, overwrite: bool = False) -> tuple[Path, Path]:
    """Ensure one ExoMol asset exists on disk and return local file paths."""
    archive_states_path, archive_transitions_path = asset.archive_paths(root)
    download_file(asset.states_url, archive_states_path, overwrite=overwrite)
    download_file(asset.transitions_url, archive_transitions_path, overwrite=overwrite)
    states_path, transitions_path = asset.local_paths(root)
    _prepare_korg_text_file(archive_states_path, states_path, overwrite=overwrite)
    _prepare_korg_text_file(archive_transitions_path, transitions_path, overwrite=overwrite)
    return states_path, transitions_path


def write_exomol_added_lines_report(
    assets: dict[str, ExoMolAsset],
    local_paths: dict[str, tuple[Path, Path]],
    report_path: Path | str,
) -> Path:
    """Write a CSV report listing every line added by the ExoMol augmentation."""
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "species",
        "wavelength_angstrom",
        "wavelength_cm",
        "log_gf",
        "E_lower_eV",
        "states_file",
        "transitions_file",
        "line_strength_cutoff",
        "temperature_line_strength",
        "lower_wavelength_angstrom",
        "upper_wavelength_angstrom",
    ]
    rows: list[dict[str, object]] = []

    for name, asset in assets.items():
        states_path, transitions_path = local_paths[name]
        linelist = load_exomol_linelist(
            asset.species_name,
            states_path,
            transitions_path,
            lower_wavelength=asset.lower_wavelength,
            upper_wavelength=asset.upper_wavelength,
            line_strength_cutoff=asset.line_strength_cutoff,
            temperature_line_strength=asset.temperature_line_strength,
            verbose=False,
        )
        for line in linelist.lines:
            wavelength_cm = float(line.wavelength)
            rows.append(
                {
                    "species": asset.species_name,
                    "wavelength_angstrom": f"{wavelength_cm * 1e8:.6f}",
                    "wavelength_cm": f"{wavelength_cm:.12e}",
                    "log_gf": f"{float(line.log_gf):.6f}",
                    "E_lower_eV": f"{float(line.E_lower):.6f}",
                    "states_file": str(states_path),
                    "transitions_file": str(transitions_path),
                    "line_strength_cutoff": f"{asset.line_strength_cutoff:.1f}",
                    "temperature_line_strength": f"{asset.temperature_line_strength:.1f}",
                    "lower_wavelength_angstrom": f"{asset.lower_wavelength:.1f}",
                    "upper_wavelength_angstrom": f"{asset.upper_wavelength:.1f}",
                }
            )

    rows.sort(
        key=lambda row: (
            float(row["wavelength_angstrom"]),
            str(row["species"]),
            float(row["log_gf"]),
            float(row["E_lower_eV"]),
        )
    )

    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return report_path


def ensure_default_exomol_assets(
    root: Path,
    overwrite: bool = False,
    *,
    report_path: Path | str | None = None,
) -> dict[str, tuple[Path, Path]]:
    """Download the default augmentation assets, refresh the added-lines CSV, and return local paths."""
    assets = default_exomol_assets()
    local_paths = {
        name: ensure_exomol_asset(asset, root, overwrite=overwrite)
        for name, asset in assets.items()
    }
    write_exomol_added_lines_report(
        assets,
        local_paths,
        report_path or default_exomol_added_lines_path(root),
    )
    return local_paths
