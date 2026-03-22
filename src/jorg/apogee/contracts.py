"""Dataset contracts for APOGEE synthetic and observational benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping

import joblib
import numpy as np

from .constants import TP_COMPACT_LABEL_NAMES


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable")


@dataclass(frozen=True)
class SyntheticDatasetPaths:
    """Resolved file paths for a synthetic APOGEE dataset shard."""

    root: Path
    wavelengths: Path = field(init=False)
    targets: Path = field(init=False)
    labels: Path = field(init=False)
    parent_index: Path = field(init=False)
    split: Path = field(init=False)
    metadata: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "wavelengths", self.root / "wavelengths.npy")
        object.__setattr__(self, "targets", self.root / "targets.joblib")
        object.__setattr__(self, "labels", self.root / "labels.joblib")
        object.__setattr__(self, "parent_index", self.root / "parent_index.npy")
        object.__setattr__(self, "split", self.root / "split.json")
        object.__setattr__(self, "metadata", self.root / "metadata.json")

    def ensure_parent(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ObservationDatasetPaths:
    """Resolved file paths for an APOGEE observation benchmark shard."""

    root: Path
    flux: Path = field(init=False)
    ivar: Path = field(init=False)
    mask: Path = field(init=False)
    labels: Path = field(init=False)
    source_ids: Path = field(init=False)
    metadata: Path = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "flux", self.root / "flux.npy")
        object.__setattr__(self, "ivar", self.root / "ivar.npy")
        object.__setattr__(self, "mask", self.root / "mask.npy")
        object.__setattr__(self, "labels", self.root / "labels.npy")
        object.__setattr__(self, "source_ids", self.root / "source_ids.npy")
        object.__setattr__(self, "metadata", self.root / "metadata.json")

    def ensure_parent(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)


@dataclass
class SyntheticDatasetMetadata:
    """Metadata persisted alongside each synthetic dataset shard."""

    dataset_name: str
    variant: str
    stage: str
    label_names: tuple[str, ...] = TP_COMPACT_LABEL_NAMES
    linelist_mode: str = "baseline"
    exomol_species: list[str] = field(default_factory=list)
    korg_root: str | None = None
    korg_version: str | None = None
    korg_git_commit: str | None = None
    synthesis_resolution: int | None = None
    notes: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObservationDatasetMetadata:
    """Metadata persisted alongside each observation benchmark shard."""

    dataset_name: str
    product: str
    selection: dict[str, Any]
    source_table: str
    normalization: dict[str, Any]
    notes: dict[str, Any] = field(default_factory=dict)


def save_synthetic_dataset(
    root: Path | str,
    wavelengths: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    parent_index: np.ndarray,
    split: Mapping[str, list[int] | np.ndarray],
    metadata: SyntheticDatasetMetadata | Mapping[str, Any],
) -> SyntheticDatasetPaths:
    """Persist a synthetic dataset shard according to the APOGEE TP contract."""
    paths = SyntheticDatasetPaths(Path(root))
    paths.ensure_parent()
    np.save(paths.wavelengths, np.asarray(wavelengths, dtype=np.float64))
    joblib.dump(np.asarray(targets, dtype=np.float32), paths.targets)
    joblib.dump(np.asarray(labels, dtype=np.float32), paths.labels)
    np.save(paths.parent_index, np.asarray(parent_index, dtype=np.int64))
    split_payload = {key: np.asarray(value, dtype=np.int64).tolist() for key, value in split.items()}
    with paths.split.open("w", encoding="utf-8") as handle:
        json.dump(split_payload, handle, indent=2, sort_keys=True)
    payload = asdict(metadata) if isinstance(metadata, SyntheticDatasetMetadata) else dict(metadata)
    with paths.metadata.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    return paths


def load_synthetic_dataset(root: Path | str) -> dict[str, Any]:
    """Load a persisted synthetic dataset shard."""
    paths = SyntheticDatasetPaths(Path(root))
    with paths.split.open("r", encoding="utf-8") as handle:
        split = json.load(handle)
    with paths.metadata.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "wavelengths": np.load(paths.wavelengths),
        "targets": joblib.load(paths.targets),
        "labels": joblib.load(paths.labels),
        "parent_index": np.load(paths.parent_index),
        "split": split,
        "metadata": metadata,
    }


def save_observation_dataset(
    root: Path | str,
    flux: np.ndarray,
    ivar: np.ndarray,
    mask: np.ndarray,
    labels: np.ndarray,
    source_ids: np.ndarray,
    metadata: ObservationDatasetMetadata | Mapping[str, Any],
) -> ObservationDatasetPaths:
    """Persist an observational benchmark shard."""
    paths = ObservationDatasetPaths(Path(root))
    paths.ensure_parent()
    np.save(paths.flux, np.asarray(flux, dtype=np.float32))
    np.save(paths.ivar, np.asarray(ivar, dtype=np.float32))
    np.save(paths.mask, np.asarray(mask, dtype=bool))
    np.save(paths.labels, np.asarray(labels, dtype=np.float32))
    np.save(paths.source_ids, np.asarray(source_ids))
    payload = asdict(metadata) if isinstance(metadata, ObservationDatasetMetadata) else dict(metadata)
    with paths.metadata.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
    return paths


def load_observation_dataset(root: Path | str) -> dict[str, Any]:
    """Load a persisted observation benchmark shard."""
    paths = ObservationDatasetPaths(Path(root))
    with paths.metadata.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    return {
        "flux": np.load(paths.flux),
        "ivar": np.load(paths.ivar),
        "mask": np.load(paths.mask),
        "labels": np.load(paths.labels),
        "source_ids": np.load(paths.source_ids),
        "metadata": metadata,
    }
