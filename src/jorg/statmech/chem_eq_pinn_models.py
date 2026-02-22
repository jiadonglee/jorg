"""Neural-network models for PINN chemical equilibrium experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import jax
import jax.numpy as jnp

from .species import MAX_ATOMIC_NUMBER

try:  # pragma: no cover - optional runtime dependency
    import flax.linen as nn
except Exception:
    nn = None


def _prepare_inputs(
    log_T: jnp.ndarray,
    log_n_total: jnp.ndarray,
    abundances: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, bool]:
    """Normalize scalar/batched inputs into a consistent batched representation."""

    log_T = jnp.asarray(log_T, dtype=jnp.float64)
    log_n_total = jnp.asarray(log_n_total, dtype=jnp.float64)
    abundances = jnp.asarray(abundances, dtype=jnp.float64)

    input_is_scalar = log_T.ndim == 0 and log_n_total.ndim == 0 and abundances.ndim == 1

    if abundances.ndim == 1:
        abundances = abundances[None, :]
    elif abundances.ndim != 2:
        raise ValueError("abundances must have shape (92,) or (batch, 92).")

    if abundances.shape[-1] != MAX_ATOMIC_NUMBER:
        raise ValueError(
            f"abundances must have {MAX_ATOMIC_NUMBER} entries in the last axis "
            f"(got {abundances.shape[-1]})."
        )

    batch_size = abundances.shape[0]

    if log_T.ndim == 0:
        log_T = jnp.full((batch_size,), log_T)
    elif log_T.ndim != 1 or log_T.shape[0] != batch_size:
        raise ValueError("log_T must be scalar or shape (batch,).")

    if log_n_total.ndim == 0:
        log_n_total = jnp.full((batch_size,), log_n_total)
    elif log_n_total.ndim != 1 or log_n_total.shape[0] != batch_size:
        raise ValueError("log_n_total must be scalar or shape (batch,).")

    return log_T, log_n_total, abundances, input_is_scalar


if nn is not None:

    class ChemicalEquilibriumPINN(nn.Module):
        """Flax MLP model predicting neutral fractions and electron-density logit."""

        hidden_dims: Tuple[int, ...] = (256, 512, 512, 256, 128)
        activation: Callable[[jnp.ndarray], jnp.ndarray] = nn.swish
        neutral_floor: float = 1e-12

        @nn.compact
        def __call__(
            self,
            log_T: jnp.ndarray,
            log_n_total: jnp.ndarray,
            abundances: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            log_T, log_n_total, abundances, input_is_scalar = _prepare_inputs(
                log_T, log_n_total, abundances
            )

            x = jnp.concatenate(
                [log_T[:, None], log_n_total[:, None], abundances],
                axis=-1,
            )

            for hidden_dim in self.hidden_dims:
                x = nn.Dense(hidden_dim)(x)
                x = self.activation(x)

            outputs = nn.Dense(MAX_ATOMIC_NUMBER + 1)(x)
            neutral_logits = outputs[:, :MAX_ATOMIC_NUMBER]
            ne_logit = outputs[:, MAX_ATOMIC_NUMBER]

            neutral_fractions = jax.nn.sigmoid(neutral_logits)
            neutral_fractions = jnp.clip(neutral_fractions, self.neutral_floor, 1.0)

            if input_is_scalar:
                return neutral_fractions[0], ne_logit[0]
            return neutral_fractions, ne_logit

else:

    @dataclass(frozen=True)
    class ChemicalEquilibriumPINN:
        """Pure-JAX fallback when Flax is unavailable or incompatible."""

        hidden_dims: Tuple[int, ...] = (256, 512, 512, 256, 128)
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.swish
        neutral_floor: float = 1e-12

        def init(
            self,
            rng_key: jax.Array,
            log_T: jnp.ndarray,
            log_n_total: jnp.ndarray,
            abundances: jnp.ndarray,
        ):
            _, _, abundances, _ = _prepare_inputs(log_T, log_n_total, abundances)
            input_dim = 2 + abundances.shape[-1]
            layer_dims = (input_dim, *self.hidden_dims, MAX_ATOMIC_NUMBER + 1)

            params = []
            key = rng_key
            for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
                key, k_w, k_b = jax.random.split(key, 3)
                weight_scale = jnp.sqrt(2.0 / float(in_dim))
                w = jax.random.normal(k_w, shape=(in_dim, out_dim), dtype=jnp.float64)
                b = jax.random.normal(k_b, shape=(out_dim,), dtype=jnp.float64) * 1e-2
                params.append({"w": w * weight_scale, "b": b})

            return tuple(params)

        def apply(
            self,
            params,
            log_T: jnp.ndarray,
            log_n_total: jnp.ndarray,
            abundances: jnp.ndarray,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            log_T, log_n_total, abundances, input_is_scalar = _prepare_inputs(
                log_T, log_n_total, abundances
            )

            x = jnp.concatenate([log_T[:, None], log_n_total[:, None], abundances], axis=-1)

            for layer in params[:-1]:
                x = x @ layer["w"] + layer["b"]
                x = self.activation(x)

            out = x @ params[-1]["w"] + params[-1]["b"]
            neutral_logits = out[:, :MAX_ATOMIC_NUMBER]
            ne_logit = out[:, MAX_ATOMIC_NUMBER]

            neutral_fractions = jax.nn.sigmoid(neutral_logits)
            neutral_fractions = jnp.clip(neutral_fractions, self.neutral_floor, 1.0)

            if input_is_scalar:
                return neutral_fractions[0], ne_logit[0]
            return neutral_fractions, ne_logit


USING_FLAX_BACKEND = nn is not None


__all__ = [
    "ChemicalEquilibriumPINN",
    "USING_FLAX_BACKEND",
]
