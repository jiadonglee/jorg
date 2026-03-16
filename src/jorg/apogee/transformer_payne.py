"""TransformerPayne integration helpers for compact APOGEE label handling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import urllib.request

import joblib
import numpy as np

from .constants import TP_ALPHA_ELEMENTS, TP_COMPACT_LABEL_NAMES

DEFAULT_TPAYNE_CHECKPOINT_URL = (
    "https://huggingface.co/RozanskiT/transformer_payne/resolve/main/"
    "TransformerPayneIntensities_v2.pkl"
)


@dataclass(frozen=True)
class TransformerPayneDefinition:
    """Serializable subset of the original TransformerPayne model definition."""

    spectral_parameters: tuple[str, ...]
    solar_parameters: np.ndarray
    min_spectral_parameters: np.ndarray
    max_spectral_parameters: np.ndarray
    abundance_parameters: np.ndarray
    architecture: str | None = None
    architecture_parameters: dict[str, Any] | None = None
    emulator_weights: str | None = None

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "TransformerPayneDefinition":
        payload = joblib.load(checkpoint_path)
        return cls(
            spectral_parameters=tuple(payload["spectral_parameters"]),
            solar_parameters=np.asarray(payload["solar_parameters"], dtype=np.float64),
            min_spectral_parameters=np.asarray(payload["min_spectral_parameters"], dtype=np.float64),
            max_spectral_parameters=np.asarray(payload["max_spectral_parameters"], dtype=np.float64),
            abundance_parameters=np.asarray(payload["abundance_parameters"], dtype=bool),
            architecture=payload.get("architecture"),
            architecture_parameters=payload.get("architecture_parameters"),
            emulator_weights=payload.get("emulator_weights"),
        )


def ensure_default_checkpoint(destination: str | Path) -> Path:
    """Download the default TransformerPayne checkpoint if needed."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    urllib.request.urlretrieve(DEFAULT_TPAYNE_CHECKPOINT_URL, destination)
    return destination


@dataclass
class TPCompactLabelAdapter:
    """Map compact APOGEE labels onto the original 95-D TransformerPayne vector."""

    definition: TransformerPayneDefinition

    def __post_init__(self) -> None:
        self.parameter_index = {
            name: idx for idx, name in enumerate(self.definition.spectral_parameters)
        }

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "TPCompactLabelAdapter":
        return cls(TransformerPayneDefinition.from_checkpoint(checkpoint_path))

    @property
    def compact_label_names(self) -> tuple[str, ...]:
        return TP_COMPACT_LABEL_NAMES

    def solar_vector(self) -> np.ndarray:
        return self.definition.solar_parameters.copy()

    def compact_to_tp(self, compact_labels: np.ndarray) -> np.ndarray:
        """Expand compact APOGEE labels to the 95-D TP label vector."""
        compact = np.asarray(compact_labels, dtype=np.float64)
        if compact.ndim == 1:
            compact = compact[None, :]
        if compact.shape[1] != len(TP_COMPACT_LABEL_NAMES):
            raise ValueError(
                f"Expected compact labels with {len(TP_COMPACT_LABEL_NAMES)} columns, "
                f"got shape {compact.shape}."
            )

        out = np.repeat(self.definition.solar_parameters[None, :], compact.shape[0], axis=0)
        teff, logg, m_h, alpha_fe, c_fe, vmic, mu = compact.T

        out[:, self.parameter_index["logteff"]] = np.log10(teff)
        out[:, self.parameter_index["logg"]] = logg
        out[:, self.parameter_index["vmic"]] = vmic
        out[:, self.parameter_index["mu"]] = mu

        for idx, name in enumerate(self.definition.spectral_parameters):
            if name in {"logteff", "logg", "vmic", "mu"}:
                continue
            if not self.definition.abundance_parameters[idx]:
                continue
            out[:, idx] = self.definition.solar_parameters[idx] + m_h

        fe_idx = self.parameter_index.get("Fe")
        if fe_idx is None:
            raise KeyError("TransformerPayne checkpoint is missing the Fe abundance label.")
        out[:, fe_idx] = self.definition.solar_parameters[fe_idx] + m_h

        c_idx = self.parameter_index.get("C")
        if c_idx is not None:
            out[:, c_idx] = out[:, fe_idx] + (
                self.definition.solar_parameters[c_idx] - self.definition.solar_parameters[fe_idx]
            ) + c_fe

        for element in TP_ALPHA_ELEMENTS:
            idx = self.parameter_index.get(element)
            if idx is None:
                continue
            out[:, idx] = out[:, fe_idx] + (
                self.definition.solar_parameters[idx] - self.definition.solar_parameters[fe_idx]
            ) + alpha_fe

        return out


def _flatten_tree(tree: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    flat: dict[tuple[str, ...], Any] = {}
    for key, value in tree.items():
        path = prefix + (str(key),)
        if isinstance(value, Mapping):
            flat.update(_flatten_tree(value, path))
        else:
            flat[path] = value
    return flat


def _unflatten_tree(flat: Mapping[tuple[str, ...], Any]) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for path, value in flat.items():
        cursor = root
        for key in path[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[path[-1]] = value
    return root


def transfer_matching_parameters(
    source_params: Mapping[str, Any],
    target_params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Transfer parameters by exact tree path and shape, returning stats."""
    source_flat = _flatten_tree(source_params)
    target_flat = _flatten_tree(target_params)
    result_flat: dict[tuple[str, ...], Any] = {}

    copied = 0
    skipped_missing = 0
    skipped_shape = 0
    copied_elements = 0

    for path, target_value in target_flat.items():
        if path not in source_flat:
            result_flat[path] = target_value
            skipped_missing += 1
            continue
        source_value = source_flat[path]
        if np.shape(source_value) != np.shape(target_value):
            result_flat[path] = target_value
            skipped_shape += 1
            continue
        result_flat[path] = np.array(source_value, copy=True)
        copied += 1
        copied_elements += int(np.size(source_value))

    stats = {
        "copied_tensors": copied,
        "copied_elements": copied_elements,
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
        "target_tensors": len(target_flat),
    }
    return _unflatten_tree(result_flat), stats
