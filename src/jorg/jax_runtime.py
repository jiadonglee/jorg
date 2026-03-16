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
    - JORG_ENABLE_JAX_X64: default true
    - JORG_ENABLE_JAX_PERSISTENT_CACHE: default true
    - JORG_JAX_CACHE_DIR: default ~/.cache/jorg/jax_compilation_cache
    - JORG_JAX_CACHE_MIN_COMPILE_SECS: default 0
    - JORG_JAX_CACHE_MIN_ENTRY_BYTES: default 0
    """

    global _CONFIGURED
    if _CONFIGURED:
        return
    _CONFIGURED = True

    try:
        import jax
    except Exception:
        return

    _patch_flax_jax_compat(jax)

    if _env_flag("JORG_ENABLE_JAX_X64", True):
        try:
            jax.config.update("jax_enable_x64", True)
        except Exception:
            pass

    if not _env_flag("JORG_ENABLE_JAX_PERSISTENT_CACHE", True):
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


def _patch_flax_jax_compat(jax_module) -> None:
    """
    Patch older Flax releases that still expect ``jax.config.define_bool_state``.

    Some local environments carry a JAX build where the helper lives under
    ``jax._src.config`` but is no longer attached to the public config object.
    Adding the attribute back keeps Flax importable without changing runtime
    semantics for supported versions.
    """

    if hasattr(jax_module.config, "define_bool_state"):
        return
    try:
        from jax._src import config as jax_internal_config
    except Exception:
        return
    define_bool_state = getattr(jax_internal_config, "define_bool_state", None)
    if define_bool_state is None:
        return
    try:
        setattr(jax_module.config, "define_bool_state", define_bool_state)
    except Exception:
        pass
