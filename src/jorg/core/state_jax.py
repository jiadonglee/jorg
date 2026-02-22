"""
JAX synthesis state containers and dense species layout helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..statmech.species import MAX_ATOMIC_NUMBER, Species


@dataclass(frozen=True)
class DenseSpeciesLayout:
    """
    Stable dense layout for species densities.

    `species[i]` maps to `number_density_dense[:, i]`.
    """

    species: Tuple[Any, ...]
    labels: Tuple[str, ...]
    index: Dict[Any, int]

    @classmethod
    def from_species(cls, species: Iterable[Any]) -> "DenseSpeciesLayout":
        sp_tuple = tuple(species)
        return cls(
            species=sp_tuple,
            labels=tuple(str(sp) for sp in sp_tuple),
            index={sp: i for i, sp in enumerate(sp_tuple)},
        )

    @classmethod
    def atomic_ion_layout(cls, max_charge: int = 2) -> "DenseSpeciesLayout":
        ordered: List[Species] = []
        for z in range(1, MAX_ATOMIC_NUMBER + 1):
            for charge in range(max_charge + 1):
                ordered.append(Species.from_atomic_number(z, charge))
        return cls.from_species(ordered)

    def dense_from_stacked_dict(
        self,
        number_densities_stacked: Dict[Any, np.ndarray],
        n_layers: int,
        *,
        dtype: np.dtype = np.float64,
    ) -> np.ndarray:
        dense = np.zeros((int(n_layers), len(self.species)), dtype=dtype)
        for sp, vals in number_densities_stacked.items():
            idx = self.index.get(sp)
            if idx is None:
                continue
            arr = np.asarray(vals, dtype=dtype)
            if arr.shape[0] != n_layers:
                raise ValueError(
                    f"Density vector length mismatch for {sp}: {arr.shape[0]} vs {n_layers}"
                )
            dense[:, idx] = arr
        return dense

    def stacked_dict_from_dense(self, dense: np.ndarray) -> Dict[Any, np.ndarray]:
        dense = np.asarray(dense, dtype=np.float64)
        if dense.ndim != 2:
            raise ValueError("dense must have shape (n_layers, n_species)")
        if dense.shape[1] != len(self.species):
            raise ValueError(
                f"dense species dimension mismatch: {dense.shape[1]} vs {len(self.species)}"
            )
        out: Dict[Any, np.ndarray] = {}
        for i, sp in enumerate(self.species):
            col = dense[:, i]
            if np.any(col > 0.0):
                out[sp] = col
        return out


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SynthesisStateJax:
    """
    End-to-end differentiable synthesis state for the JAX engine.
    """

    flux: jnp.ndarray
    continuum: jnp.ndarray
    alpha_total: jnp.ndarray
    alpha_continuum: jnp.ndarray
    source_function: jnp.ndarray
    electron_density: jnp.ndarray
    number_density_dense: jnp.ndarray
    species_layout: DenseSpeciesLayout
    wavelengths: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.flux,
            self.continuum,
            self.alpha_total,
            self.alpha_continuum,
            self.source_function,
            self.electron_density,
            self.number_density_dense,
            self.wavelengths,
        )
        aux = {"species_layout": self.species_layout}
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            flux,
            continuum,
            alpha_total,
            alpha_continuum,
            source_function,
            electron_density,
            number_density_dense,
            wavelengths,
        ) = children
        return cls(
            flux=flux,
            continuum=continuum,
            alpha_total=alpha_total,
            alpha_continuum=alpha_continuum,
            source_function=source_function,
            electron_density=electron_density,
            number_density_dense=number_density_dense,
            species_layout=aux["species_layout"],
            wavelengths=wavelengths,
        )

