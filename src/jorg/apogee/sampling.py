"""Parameter-space sampling helpers for APOGEE synthetic grid generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import qmc


@dataclass(frozen=True)
class ParameterRange:
    """Closed interval for a single stellar label."""

    name: str
    lower: float
    upper: float

    def scale(self, u: np.ndarray) -> np.ndarray:
        return self.lower + (self.upper - self.lower) * u


@dataclass(frozen=True)
class ParameterBox:
    """Multi-dimensional box for Latin-hypercube sampling."""

    ranges: tuple[ParameterRange, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.ranges)

    @property
    def lower(self) -> np.ndarray:
        return np.array([item.lower for item in self.ranges], dtype=np.float64)

    @property
    def upper(self) -> np.ndarray:
        return np.array([item.upper for item in self.ranges], dtype=np.float64)

    def sample(self, n_samples: int, seed: int | None = None) -> np.ndarray:
        sampler = qmc.LatinHypercube(d=len(self.ranges), seed=seed)
        unit = sampler.random(n_samples)
        return qmc.scale(unit, self.lower, self.upper)


@dataclass(frozen=True)
class StageDefinition:
    """Synthetic grid definition for one training stage."""

    name: str
    bounds: ParameterBox
    n_train: int
    n_val: int
    n_test: int = 0

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test


def smoke_box() -> ParameterBox:
    return ParameterBox(
        (
            ParameterRange("Teff", 3000.0, 4500.0),
            ParameterRange("logg", 0.0, 5.0),
            ParameterRange("m_h", -1.5, 0.5),
            ParameterRange("alpha_fe", -0.2, 0.5),
            ParameterRange("c_fe", -0.8, 0.4),
            ParameterRange("vmic", 0.5, 2.5),
        )
    )


def default_stage_definitions() -> tuple[StageDefinition, ...]:
    """Return the default stage layout from the implementation plan."""
    return (
        StageDefinition(
            name="stage1_overlap",
            bounds=ParameterBox(
                (
                    ParameterRange("Teff", 4000.0, 4500.0),
                    ParameterRange("logg", 2.0, 5.0),
                    ParameterRange("m_h", -1.5, 0.5),
                    ParameterRange("alpha_fe", -0.2, 0.5),
                    ParameterRange("c_fe", -0.8, 0.4),
                    ParameterRange("vmic", 0.5, 2.5),
                )
            ),
            n_train=2048,
            n_val=256,
        ),
        StageDefinition(
            name="stage2_expand_lowT",
            bounds=ParameterBox(
                (
                    ParameterRange("Teff", 3000.0, 4000.0),
                    ParameterRange("logg", 2.0, 5.0),
                    ParameterRange("m_h", -1.5, 0.5),
                    ParameterRange("alpha_fe", -0.2, 0.5),
                    ParameterRange("c_fe", -0.8, 0.4),
                    ParameterRange("vmic", 0.5, 2.5),
                )
            ),
            n_train=2048,
            n_val=256,
        ),
        StageDefinition(
            name="stage2_expand_lowg",
            bounds=ParameterBox(
                (
                    ParameterRange("Teff", 3000.0, 4500.0),
                    ParameterRange("logg", 0.0, 2.0),
                    ParameterRange("m_h", -1.5, 0.5),
                    ParameterRange("alpha_fe", -0.2, 0.5),
                    ParameterRange("c_fe", -0.8, 0.4),
                    ParameterRange("vmic", 0.5, 2.5),
                )
            ),
            n_train=2048,
            n_val=256,
        ),
    )


def make_split_indices(n_train: int, n_val: int, n_test: int = 0) -> dict[str, list[int]]:
    """Return contiguous split indices for a single dataset shard."""
    train_end = n_train
    val_end = train_end + n_val
    test_end = val_end + n_test
    return {
        "train": list(range(0, train_end)),
        "val": list(range(train_end, val_end)),
        "test": list(range(val_end, test_end)),
        "smoke": [],
    }


def sample_stage_definition(stage: StageDefinition, seed: int) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Sample one stage and return dense values plus split indices."""
    points = stage.bounds.sample(stage.n_total, seed=seed)
    split = make_split_indices(stage.n_train, stage.n_val, stage.n_test)
    return points, split


def sample_smoke_points(n_points: int = 256, seed: int = 0) -> np.ndarray:
    """Sample the smoke grid used for local end-to-end validation."""
    return smoke_box().sample(n_points, seed=seed)
