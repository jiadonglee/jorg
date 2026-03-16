"""Bridges between Python and the local Korg.jl APOGEE synthesis workflow."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Iterable, Sequence

import h5py
import numpy as np

from .constants import SynthesisGrid, apogee_root, default_korg_root, repo_root
from .contracts import SyntheticDatasetMetadata, save_synthetic_dataset
from .sampling import StageDefinition, sample_smoke_points, sample_stage_definition


@dataclass(frozen=True)
class KorgExoMolLineList:
    """Local ExoMol file pair passed to the Julia Korg synthesizer."""

    species_name: str
    states_path: Path
    transitions_path: Path
    lower_wavelength: float = 15100.0
    upper_wavelength: float = 17000.0
    line_strength_cutoff: float = -15.0
    temperature_line_strength: float = 3500.0


@dataclass(frozen=True)
class KorgSynthesisRequest:
    """One APOGEE synthesis request executed through Julia/Korg."""

    teff: float
    logg: float
    m_h: float
    alpha_fe: float
    c_fe: float
    vmic: float
    water_sigma_path: Path
    exomol_linelist: tuple[KorgExoMolLineList, ...] = ()
    synthesis_grid: SynthesisGrid = SynthesisGrid()
    mu_values: int = 20
    use_exomol_aug: bool = False

    @property
    def alpha_h(self) -> float:
        return self.m_h + self.alpha_fe

    @property
    def c_h(self) -> float:
        return self.m_h + self.c_fe


@dataclass
class KorgSynthesisResult:
    """Decoded outputs from the Julia/Korg APOGEE synthesizer."""

    wavelengths: np.ndarray
    mu_values: np.ndarray
    mu_weights: np.ndarray
    intensity: np.ndarray
    continuum_intensity: np.ndarray
    flux: np.ndarray
    continuum_flux: np.ndarray
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_mu(self) -> int:
        return int(self.mu_values.shape[0])


def _run_subprocess(cmd: Sequence[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        check=True,
        text=True,
        capture_output=True,
    )


def ensure_korg_environment(korg_root: Path | None = None, julia_bin: str = "julia") -> None:
    """Instantiate the local Korg Julia environment once before synthesis."""
    korg_root = Path(korg_root or default_korg_root())
    _run_subprocess(
        [
            julia_bin,
            f"--project={korg_root}",
            "-e",
            "using Pkg; Pkg.instantiate()",
        ]
    )


def _write_exomol_manifest(lines: Sequence[KorgExoMolLineList], path: Path) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(
                "\t".join(
                    [
                        line.species_name,
                        str(line.states_path),
                        str(line.transitions_path),
                        str(line.lower_wavelength),
                        str(line.upper_wavelength),
                        str(line.line_strength_cutoff),
                        str(line.temperature_line_strength),
                    ]
                )
                + "\n"
            )
    return path


def ensure_apogee_water_sigma(
    output_path: Path | str | None = None,
    *,
    korg_root: Path | None = None,
    julia_bin: str = "julia",
    synthesis_grid: SynthesisGrid = SynthesisGrid(),
) -> Path:
    """Generate the missing APOGEE H2O molecular cross-section from local line data."""
    korg_root = Path(korg_root or default_korg_root())
    output_path = Path(output_path or (apogee_root() / "output" / "apogee_water_sigma.h5"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        return output_path

    script_path = apogee_root() / "scripts" / "apogee_make_water_sigma.jl"
    cmd = [
        julia_bin,
        f"--project={korg_root}",
        str(script_path),
        str(korg_root),
        str(output_path),
        str(synthesis_grid.synthesis_start),
        str(synthesis_grid.synthesis_stop),
        str(synthesis_grid.synthesis_step),
    ]
    _run_subprocess(cmd, cwd=repo_root())
    return output_path


def run_korg_apogee_synthesis(
    request: KorgSynthesisRequest,
    *,
    korg_root: Path | None = None,
    julia_bin: str = "julia",
) -> KorgSynthesisResult:
    """Execute one APOGEE synthesis through the Julia Korg bridge."""
    korg_root = Path(korg_root or default_korg_root())
    script_path = apogee_root() / "scripts" / "apogee_synthesize_korg.jl"
    with tempfile.TemporaryDirectory(prefix="jorg-apogee-korg-") as tmpdir:
        tmpdir_path = Path(tmpdir)
        output_path = tmpdir_path / "synthesis_output.h5"
        manifest_path = tmpdir_path / "exomol_manifest.tsv"
        if request.exomol_linelist:
            _write_exomol_manifest(request.exomol_linelist, manifest_path)
            manifest_arg = str(manifest_path)
        else:
            manifest_arg = "-"

        cmd = [
            julia_bin,
            f"--project={korg_root}",
            str(script_path),
            str(korg_root),
            str(output_path),
            str(request.teff),
            str(request.logg),
            str(request.m_h),
            str(request.alpha_h),
            str(request.c_h),
            str(request.vmic),
            str(request.mu_values),
            str(request.water_sigma_path),
            "1" if request.use_exomol_aug else "0",
            manifest_arg,
            str(request.synthesis_grid.synthesis_start),
            str(request.synthesis_grid.synthesis_stop),
            str(request.synthesis_grid.synthesis_step),
            str(request.synthesis_grid.resolution),
        ]
        completed = _run_subprocess(cmd, cwd=repo_root())
        result = _read_synthesis_hdf5(output_path)
        result.metadata["stdout"] = completed.stdout
        result.metadata["stderr"] = completed.stderr
        return result


def _read_synthesis_hdf5(path: Path) -> KorgSynthesisResult:
    with h5py.File(path, "r") as handle:
        metadata = json.loads(handle.attrs.get("metadata", "{}"))
        return KorgSynthesisResult(
            wavelengths=np.asarray(handle["wavelengths"], dtype=np.float64),
            mu_values=np.asarray(handle["mu_values"], dtype=np.float64),
            mu_weights=np.asarray(handle["mu_weights"], dtype=np.float64),
            intensity=np.asarray(handle["intensity"], dtype=np.float32),
            continuum_intensity=np.asarray(handle["continuum_intensity"], dtype=np.float32),
            flux=np.asarray(handle["flux"], dtype=np.float32),
            continuum_flux=np.asarray(handle["continuum_flux"], dtype=np.float32),
            metadata=metadata,
        )


def flatten_mu_targets(
    result: KorgSynthesisResult,
    parent_labels: np.ndarray,
    parent_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand one parent stellar point into per-mu samples for TP training."""
    compact = np.asarray(parent_labels, dtype=np.float32)
    if compact.shape != (6,):
        raise ValueError(f"Expected parent_labels with shape (6,), got {compact.shape}.")
    n_mu = result.n_mu
    sample_labels = np.repeat(compact[None, :], n_mu, axis=0)
    sample_labels = np.concatenate(
        [sample_labels, result.mu_values.astype(np.float32)[:, None]],
        axis=1,
    )
    targets = np.stack([result.intensity, result.continuum_intensity], axis=-1).astype(np.float32)
    parents = np.full(n_mu, int(parent_index), dtype=np.int64)
    return targets, sample_labels, parents


def _korg_version(korg_root: Path) -> str | None:
    project_path = korg_root / "Project.toml"
    if not project_path.exists():
        return None
    for line in project_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("version ="):
            return line.split("=", 1)[1].strip().strip('"')
    return None


def _korg_git_commit(korg_root: Path) -> str | None:
    if not (korg_root / ".git").exists():
        return None
    try:
        completed = _run_subprocess(["git", "rev-parse", "HEAD"], cwd=korg_root)
    except subprocess.CalledProcessError:
        return None
    return completed.stdout.strip() or None


def _expand_parent_split(parent_split: dict[str, list[int]], n_mu_per_parent: int) -> dict[str, list[int]]:
    sample_split: dict[str, list[int]] = {}
    for split_name, parent_indices in parent_split.items():
        sample_indices: list[int] = []
        for parent_idx in parent_indices:
            start = parent_idx * n_mu_per_parent
            sample_indices.extend(range(start, start + n_mu_per_parent))
        sample_split[split_name] = sample_indices
    return sample_split


def build_synthetic_dataset(
    output_root: Path | str,
    *,
    dataset_name: str,
    variant: str,
    water_sigma_path: Path,
    requests: Sequence[KorgSynthesisRequest],
    split: dict[str, list[int]],
    korg_root: Path | None = None,
    julia_bin: str = "julia",
    notes: dict[str, object] | None = None,
) -> Path:
    """Run multiple Korg requests and persist a synthetic dataset shard."""
    korg_root = Path(korg_root or default_korg_root())
    shard_root = Path(output_root)

    all_targets: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_parent_index: list[np.ndarray] = []
    wavelengths: np.ndarray | None = None
    n_mu_per_parent: int | None = None
    exomol_species: set[str] = set()

    for parent_idx, request in enumerate(requests):
        result = run_korg_apogee_synthesis(request, korg_root=korg_root, julia_bin=julia_bin)
        if wavelengths is None:
            wavelengths = result.wavelengths
            n_mu_per_parent = result.n_mu
        elif not np.allclose(wavelengths, result.wavelengths):
            raise ValueError("All Korg synthesis requests must share the same APOGEE wavelength grid.")
        if result.n_mu != n_mu_per_parent:
            raise ValueError("Mixed mu-grid sizes are not supported in one dataset shard.")

        parent_labels = np.array(
            [
                request.teff,
                request.logg,
                request.m_h,
                request.alpha_fe,
                request.c_fe,
                request.vmic,
            ],
            dtype=np.float32,
        )
        targets, labels, parent_ids = flatten_mu_targets(result, parent_labels, parent_idx)
        all_targets.append(targets)
        all_labels.append(labels)
        all_parent_index.append(parent_ids)
        exomol_species.update(line.species_name for line in request.exomol_linelist)

    assert wavelengths is not None
    assert n_mu_per_parent is not None
    targets = np.concatenate(all_targets, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    parent_index = np.concatenate(all_parent_index, axis=0)
    sample_split = _expand_parent_split(split, n_mu_per_parent)
    metadata = SyntheticDatasetMetadata(
        dataset_name=dataset_name,
        variant=variant,
        stage=Path(output_root).name,
        linelist_mode=variant,
        exomol_species=sorted(exomol_species),
        korg_root=str(korg_root),
        korg_version=_korg_version(korg_root),
        korg_git_commit=_korg_git_commit(korg_root),
        synthesis_resolution=requests[0].synthesis_grid.resolution if requests else None,
        notes=notes or {},
    )
    save_synthetic_dataset(shard_root, wavelengths, targets, labels, parent_index, sample_split, metadata)
    return shard_root


def build_smoke_requests(
    water_sigma_path: Path,
    *,
    n_points: int = 256,
    seed: int = 0,
    synthesis_grid: SynthesisGrid = SynthesisGrid(),
) -> list[KorgSynthesisRequest]:
    """Create the smoke-grid Korg requests for local end-to-end validation."""
    smoke_points = sample_smoke_points(n_points=n_points, seed=seed)
    return [
        KorgSynthesisRequest(
            teff=float(point[0]),
            logg=float(point[1]),
            m_h=float(point[2]),
            alpha_fe=float(point[3]),
            c_fe=float(point[4]),
            vmic=float(point[5]),
            water_sigma_path=water_sigma_path,
            synthesis_grid=synthesis_grid,
        )
        for point in smoke_points
    ]


def build_stage_requests(
    stage: StageDefinition,
    water_sigma_path: Path,
    *,
    seed: int,
    synthesis_grid: SynthesisGrid = SynthesisGrid(),
) -> tuple[list[KorgSynthesisRequest], dict[str, list[int]], np.ndarray]:
    """Sample one stage and convert it to Korg synthesis requests."""
    points, split = sample_stage_definition(stage, seed=seed)
    requests = [
        KorgSynthesisRequest(
            teff=float(point[0]),
            logg=float(point[1]),
            m_h=float(point[2]),
            alpha_fe=float(point[3]),
            c_fe=float(point[4]),
            vmic=float(point[5]),
            water_sigma_path=water_sigma_path,
            synthesis_grid=synthesis_grid,
        )
        for point in points
    ]
    return requests, split, points
