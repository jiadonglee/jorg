"""
Data path helpers for Jorg.

Use JORG_DATA_DIR to point at external data bundles (MARCS grids, linelists, etc.).
Packaged data files live inside this module for lightweight defaults.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path
from typing import Optional, Union


def _env_path(var: str) -> Optional[Path]:
    value = os.environ.get(var)
    if not value:
        return None
    return Path(value).expanduser()


def package_data_root() -> Path:
    """Return the on-disk path to packaged Jorg data files."""
    return Path(resources.files(__name__))


def get_data_root(data_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve the base data directory.

    Priority: explicit data_dir -> JORG_DATA_DIR -> repo-root data -> packaged data.
    """
    if data_dir is not None:
        return Path(data_dir).expanduser()
    env_dir = _env_path("JORG_DATA_DIR")
    if env_dir is not None:
        return env_dir
    repo_root = Path(__file__).resolve().parents[3]
    repo_data = repo_root / "data"
    if repo_data.is_dir():
        return repo_data
    return package_data_root()


def get_data_path(
    *parts: str,
    data_dir: Optional[Union[str, Path]] = None,
    must_exist: bool = True,
) -> Path:
    """Resolve a data file path relative to the data root."""
    candidate = get_data_root(data_dir).joinpath(*parts)
    if must_exist and not candidate.exists():
        raise FileNotFoundError(
            f"Jorg data file not found: {candidate}. "
            "Set JORG_DATA_DIR or pass data_dir to point at your data bundle."
        )
    return candidate
