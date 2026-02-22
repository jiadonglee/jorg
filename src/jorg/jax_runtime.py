"""
JAX runtime configuration helpers for Jorg.

This module centralizes optional runtime settings that reduce cold-start
compilation overhead across separate Python processes.
"""

from __future__ import annotations

import os
from pathlib import Path


_CONFIGURED = False


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def configure_jax_runtime() -> None:
    """
    Configure JAX persistent compilation cache once per process.

    Environment variables:
    - JORG_ENABLE_JAX_PERSISTENT_CACHE: default true
    - JORG_JAX_CACHE_DIR: default ~/.cache/jorg/jax_compilation_cache
    - JORG_JAX_CACHE_MIN_COMPILE_SECS: default 0
    - JORG_JAX_CACHE_MIN_ENTRY_BYTES: default 0
    """

    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    if not _env_flag("JORG_ENABLE_JAX_PERSISTENT_CACHE", True):
        return

    try:
        import jax
    except Exception:
        return

    cache_dir = Path(
        os.environ.get(
            "JORG_JAX_CACHE_DIR",
            str(Path.home() / ".cache" / "jorg" / "jax_compilation_cache"),
        )
    ).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)

    min_compile_secs = float(os.environ.get("JORG_JAX_CACHE_MIN_COMPILE_SECS", "0"))
    min_entry_bytes = int(os.environ.get("JORG_JAX_CACHE_MIN_ENTRY_BYTES", "0"))

    config_updates = {
        "jax_enable_compilation_cache": True,
        "jax_compilation_cache_dir": str(cache_dir),
        "jax_persistent_cache_min_compile_time_secs": min_compile_secs,
        "jax_persistent_cache_min_entry_size_bytes": min_entry_bytes,
    }
    for key, value in config_updates.items():
        try:
            jax.config.update(key, value)
        except Exception:
            pass

