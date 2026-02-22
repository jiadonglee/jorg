"""High-level PINN chemical-equilibrium solver API.

This module provides :class:`ChemicalEquilibriumPINNSolver`, a physics-informed
neural-network (PINN) wrapper around :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN`
that exposes a simple *train / infer / save / load* interface for solving stellar
chemical-equilibrium problems.

Typical usage
-------------
>>> from jorg.statmech.chem_eq_jax import build_chem_data
>>> from jorg.statmech.chem_eq_pinn import ChemicalEquilibriumPINNSolver
>>> chem_data = build_chem_data()
>>> solver = ChemicalEquilibriumPINNSolver(chem_data)
>>> history = solver.train(n_samples=2048, supervised_steps=500, physics_steps=500)
>>> ne, neutral_fracs = solver.solve(5770.0, 1e16, abundances)
>>> solver.save("model_checkpoint")
>>> solver2 = ChemicalEquilibriumPINNSolver.load("model_checkpoint.npz", chem_data=chem_data)

The network takes :math:`(\\log_{10} T,\\, \\log_{10} n_{\\rm tot},\\, A_Z)` as input and
predicts per-element neutral fractions together with the electron-number density
:math:`n_e`.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .chem_eq_jax import ChemEqData
from .chem_eq_pinn_loss import charge_neutrality_residual, element_conservation_residual
from .chem_eq_pinn_models import ChemicalEquilibriumPINN
from .chem_eq_pinn_train import (
    create_optimizer,
    generate_training_data,
    train_pinn,
)
from .species import MAX_ATOMIC_NUMBER


def _serialize_params(params) -> bytes:
    """Serialize a JAX parameter pytree to a raw byte string.

    All device arrays are first moved to CPU via :func:`jax.device_get` and
    converted to :class:`numpy.ndarray` before being pickled with the highest
    available protocol.

    Parameters
    ----------
    params:
        Arbitrary JAX pytree (e.g. a Flax parameter dict or a list of
        ``{"w": ..., "b": ...}`` dicts) that represents the model weights.

    Returns
    -------
    bytes
        Pickle-serialized byte string suitable for storage.
    """
    cpu_params = jax.tree_util.tree_map(lambda x: np.asarray(jax.device_get(x)), params)
    return pickle.dumps(cpu_params, protocol=pickle.HIGHEST_PROTOCOL)


def _deserialize_params(payload: bytes):
    """Deserialize a parameter pytree from a raw byte string.

    The inverse of :func:`_serialize_params`.  The loaded NumPy arrays are
    transferred back to JAX device arrays with ``float64`` precision.

    Parameters
    ----------
    payload:
        Pickle-encoded byte string produced by :func:`_serialize_params`.

    Returns
    -------
    pytree
        JAX parameter pytree with all leaf arrays as ``jnp.float64`` device
        arrays.
    """
    cpu_params = pickle.loads(payload)
    return jax.tree_util.tree_map(lambda x: jnp.asarray(x, dtype=jnp.float64), cpu_params)


def _is_fallback_param_tree(params) -> bool:
    """Check whether *params* follows the lightweight fallback layer format.

    The fallback format is a non-empty sequence of dicts each containing
    exactly the keys ``"w"`` (weight matrix) and ``"b"`` (bias vector).  This
    format is used when the model was saved without a Flax backend.

    Parameters
    ----------
    params:
        Candidate parameter structure to inspect.

    Returns
    -------
    bool
        ``True`` if *params* is a non-empty list/tuple of ``{"w": ..., "b":
        ...}`` dicts; ``False`` otherwise.
    """
    if not isinstance(params, (tuple, list)) or len(params) == 0:
        return False
    for layer in params:
        if not isinstance(layer, dict):
            return False
        if "w" not in layer or "b" not in layer:
            return False
    return True


def _prepare_batched_inputs(
    log_T: jnp.ndarray,
    log_n_total: jnp.ndarray,
    abundances: jnp.ndarray,
):
    """Broadcast and validate inputs for batched model inference.

    Scalar thermodynamic inputs are broadcast to match the batch dimension
    inferred from *abundances*.  A single-atmosphere abundance vector
    ``(MAX_ATOMIC_NUMBER,)`` is promoted to ``(1, MAX_ATOMIC_NUMBER)``.

    Parameters
    ----------
    log_T:
        :math:`\\log_{10}(T / \\mathrm{K})`.  Either a scalar or a 1-D array
        of shape ``(batch,)``.
    log_n_total:
        :math:`\\log_{10}(n_{\\rm tot} / \\mathrm{cm}^{-3})`.  Either a scalar
        or a 1-D array of shape ``(batch,)``.
    abundances:
        Elemental number fractions :math:`A_Z`.  Shape ``(MAX_ATOMIC_NUMBER,)``
        or ``(batch, MAX_ATOMIC_NUMBER)``.

    Returns
    -------
    log_T : jnp.ndarray
        Shape ``(batch,)``, dtype ``float64``.
    log_n_total : jnp.ndarray
        Shape ``(batch,)``, dtype ``float64``.
    abundances : jnp.ndarray
        Shape ``(batch, MAX_ATOMIC_NUMBER)``, dtype ``float64``.
    scalar_input : bool
        ``True`` when the original inputs described a single atmosphere
        (scalar *log_T*, scalar *log_n_total*, 1-D *abundances*).

    Raises
    ------
    ValueError
        If any shape is inconsistent with the inferred batch size.
    """
    log_T = jnp.asarray(log_T, dtype=jnp.float64)
    log_n_total = jnp.asarray(log_n_total, dtype=jnp.float64)
    abundances = jnp.asarray(abundances, dtype=jnp.float64)

    scalar_input = log_T.ndim == 0 and log_n_total.ndim == 0 and abundances.ndim == 1
    if abundances.ndim == 1:
        abundances = abundances[None, :]
    if abundances.ndim != 2 or abundances.shape[1] != MAX_ATOMIC_NUMBER:
        raise ValueError(f"abundances must have shape (92,) or (batch, {MAX_ATOMIC_NUMBER}).")

    batch = abundances.shape[0]

    if log_T.ndim == 0:
        log_T = jnp.full((batch,), log_T)
    elif log_T.ndim != 1 or log_T.shape[0] != batch:
        raise ValueError("log_T must be scalar or shape (batch,).")

    if log_n_total.ndim == 0:
        log_n_total = jnp.full((batch,), log_n_total)
    elif log_n_total.ndim != 1 or log_n_total.shape[0] != batch:
        raise ValueError("log_n_total must be scalar or shape (batch,).")

    return log_T, log_n_total, abundances, scalar_input


class ChemicalEquilibriumPINNSolver:
    """PINN-based chemical-equilibrium solver with a simple train/infer API.

    The solver wraps a :class:`~.chem_eq_pinn_models.ChemicalEquilibriumPINN`
    multi-layer perceptron that maps

    .. math::

        (\\log_{10} T,\\; \\log_{10} n_{\\rm tot},\\; A_Z)
        \\;\\longrightarrow\\;
        (f_{\\rm neutral,\\,Z},\\; n_e)

    where :math:`f_{\\rm neutral,\\,Z}` is the fraction of element :math:`Z` in
    neutral form and :math:`n_e` is the electron number density.

    The network can be either **trained from scratch** via :meth:`train`, or
    **loaded from a checkpoint** via :meth:`load`.  Single-point and batched
    inference are provided by :meth:`solve` and :meth:`solve_batch`
    respectively.  Physics residuals (element conservation and charge
    neutrality) can be evaluated with :meth:`physics_residuals`.

    Attributes
    ----------
    chem_data : ChemEqData
        Pre-computed chemical-equilibrium tables (partition functions, ionisation
        energies, equilibrium constants, etc.).
    model : ChemicalEquilibriumPINN
        The underlying Flax/custom MLP module.
    params :
        Current model parameters as a JAX pytree.
    hidden_dims : tuple of int
        Number of units in each hidden layer.
    rng_key : jax.Array
        PRNG state used for weight initialisation and training.
    """

    def __init__(
        self,
        chem_data: ChemEqData,
        *,
        params=None,
        model_path: Optional[str] = None,
        rng_key: Optional[jax.Array] = None,
        hidden_dims: Tuple[int, ...] = (256, 512, 512, 256, 128),
        neutral_floor: float = 1e-12,
    ) -> None:
        """Initialise the solver.

        Exactly one of the three parameter-supply modes is active at a time:

        1. **Direct params** – pass *params* to use pre-computed weights.
        2. **Load from file** – pass *model_path* to restore a saved checkpoint.
        3. **Random init** – omit both to draw fresh random weights.

        Parameters
        ----------
        chem_data:
            Chemical-equilibrium data tables (partition functions, ionisation
            energies, Saha equilibrium constants, etc.).
        params:
            Pre-built JAX parameter pytree.  Supersedes *model_path* when
            provided.
        model_path:
            Path to an ``.npz`` checkpoint previously written by :meth:`save`.
            Ignored when *params* is given.
        rng_key:
            JAX PRNG key used for random weight initialisation.  Defaults to
            ``jax.random.PRNGKey(0)`` when ``None``.
        hidden_dims:
            Sequence of hidden-layer widths for the MLP.  Default
            ``(256, 512, 512, 256, 128)``.
        neutral_floor:
            Minimum clamp value for predicted neutral fractions, preventing
            log-domain underflow.  Default ``1e-12``.
        """
        self.chem_data = chem_data
        self.hidden_dims = tuple(int(v) for v in hidden_dims)
        self.model = ChemicalEquilibriumPINN(
            hidden_dims=self.hidden_dims,
            neutral_floor=float(neutral_floor),
        )

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        self.rng_key = rng_key

        if params is not None:
            self.params = params
        elif model_path is not None:
            loaded = self.load(model_path, chem_data=chem_data)
            self.params = loaded.params
            self.hidden_dims = loaded.hidden_dims
            self.model = loaded.model
        else:
            self.params = self.model.init(
                self.rng_key,
                jnp.log10(jnp.asarray(5770.0, dtype=jnp.float64)),
                jnp.log10(jnp.asarray(1e16, dtype=jnp.float64)),
                jnp.full((MAX_ATOMIC_NUMBER,), 1.0 / MAX_ATOMIC_NUMBER, dtype=jnp.float64),
            )

    @staticmethod
    def _prepare_abundances(abundances: Dict[int, float] | np.ndarray) -> np.ndarray:
        """Normalise the *abundances* argument into a NumPy array.

        Accepts either a mapping from atomic number :math:`Z` to number
        fraction, or an array already shaped ``(MAX_ATOMIC_NUMBER,)`` /
        ``(batch, MAX_ATOMIC_NUMBER)``.

        Parameters
        ----------
        abundances:
            * ``dict`` – keys are integer atomic numbers :math:`Z \\in [1, 92]`;
              values are elemental number fractions.  Missing elements are set
              to zero.
            * ``np.ndarray`` – must have shape ``(MAX_ATOMIC_NUMBER,)`` or
              ``(batch, MAX_ATOMIC_NUMBER)``.

        Returns
        -------
        np.ndarray
            Float64 array of shape ``(MAX_ATOMIC_NUMBER,)`` or
            ``(batch, MAX_ATOMIC_NUMBER)``.

        Raises
        ------
        ValueError
            If the array shape is incompatible.
        """
        if isinstance(abundances, dict):
            arr = np.zeros(MAX_ATOMIC_NUMBER, dtype=np.float64)
            for Z, value in abundances.items():
                if 1 <= int(Z) <= MAX_ATOMIC_NUMBER:
                    arr[int(Z) - 1] = float(value)
            return arr

        arr = np.asarray(abundances, dtype=np.float64)
        if arr.ndim == 1 and arr.shape[0] == MAX_ATOMIC_NUMBER:
            return arr
        if arr.ndim == 2 and arr.shape[1] == MAX_ATOMIC_NUMBER:
            return arr

        raise ValueError(
            f"abundances must have shape ({MAX_ATOMIC_NUMBER},) or (batch, {MAX_ATOMIC_NUMBER})."
        )

    def _model_apply(self, log_T, log_n_total, abundances):
        """Apply the neural network using whichever backend is active.

        Dispatches between:

        * **Fallback backend** – a plain list/tuple of ``{"w", "b"}`` dicts
          evaluated with explicit matrix multiplications and Swish activations.
        * **Flax backend** – delegated to
          :meth:`ChemicalEquilibriumPINN.apply`.

        Parameters
        ----------
        log_T:
            :math:`\\log_{10}(T / \\mathrm{K})`.  Scalar or shape
            ``(batch,)``.
        log_n_total:
            :math:`\\log_{10}(n_{\\rm tot} / \\mathrm{cm}^{-3})`.  Scalar or
            shape ``(batch,)``.
        abundances:
            Elemental number fractions.  Shape ``(MAX_ATOMIC_NUMBER,)`` or
            ``(batch, MAX_ATOMIC_NUMBER)``.

        Returns
        -------
        neutral_fractions : jnp.ndarray
            Predicted neutral fractions, shape ``(batch, MAX_ATOMIC_NUMBER)``
            or ``(MAX_ATOMIC_NUMBER,)`` for scalar input.
        ne_logit : jnp.ndarray
            Raw logit whose sigmoid gives :math:`n_e / n_{\\rm tot}`.  Shape
            ``(batch,)`` or scalar.
        """
        if _is_fallback_param_tree(self.params):
            log_T_b, log_n_total_b, abund_b, scalar_input = _prepare_batched_inputs(
                log_T, log_n_total, abundances
            )

            x = jnp.concatenate([log_T_b[:, None], log_n_total_b[:, None], abund_b], axis=-1)
            for layer in self.params[:-1]:
                x = x @ layer["w"] + layer["b"]
                x = jax.nn.swish(x)

            out = x @ self.params[-1]["w"] + self.params[-1]["b"]
            neutral_logits = out[:, :MAX_ATOMIC_NUMBER]
            ne_logit = out[:, MAX_ATOMIC_NUMBER]
            neutral_fractions = jnp.clip(jax.nn.sigmoid(neutral_logits), 1e-12, 1.0)

            if scalar_input:
                return neutral_fractions[0], ne_logit[0]
            return neutral_fractions, ne_logit

        return self.model.apply(self.params, log_T, log_n_total, abundances)

    def solve(
        self,
        temperature: float,
        n_total: float,
        abundances: Dict[int, float] | np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """Predict electron density and neutral fractions for one thermodynamic state.

        Parameters
        ----------
        temperature:
            Gas temperature in Kelvin.
        n_total:
            Total number density of all particles (atoms + ions + electrons)
            in cm\ :sup:`-3`.
        abundances:
            Elemental number fractions.  Either a ``dict`` mapping atomic
            number :math:`Z` to fraction, or an array of shape
            ``(MAX_ATOMIC_NUMBER,)``.

        Returns
        -------
        ne : float
            Predicted electron number density in cm\ :sup:`-3`.
        neutral_fractions : np.ndarray
            Float64 array of shape ``(MAX_ATOMIC_NUMBER,)`` with the predicted
            fraction of each element that is in neutral form.
        """

        abundances_arr = self._prepare_abundances(abundances)
        log_T = jnp.log10(jnp.asarray(float(temperature), dtype=jnp.float64))
        log_n_total = jnp.log10(jnp.asarray(float(n_total), dtype=jnp.float64))

        neutral_fractions, ne_logit = self._model_apply(
            log_T,
            log_n_total,
            jnp.asarray(abundances_arr, dtype=jnp.float64),
        )
        ne = float(jax.device_get(jnp.asarray(n_total, dtype=jnp.float64) * jax.nn.sigmoid(ne_logit)))

        return ne, np.asarray(jax.device_get(neutral_fractions), dtype=np.float64)

    def solve_batch(
        self,
        temperatures: np.ndarray,
        n_totals: np.ndarray,
        abundances: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized PINN inference for a batch of thermodynamic states.

        All atmospheres in the batch are processed in a single forward pass
        through the network, making this much more efficient than calling
        :meth:`solve` repeatedly.

        Parameters
        ----------
        temperatures:
            1-D array of gas temperatures in Kelvin, shape ``(batch,)``.
        n_totals:
            1-D array of total number densities in cm\ :sup:`-3`, shape
            ``(batch,)``.
        abundances:
            2-D array of elemental number fractions, shape
            ``(batch, MAX_ATOMIC_NUMBER)``.

        Returns
        -------
        ne : np.ndarray
            Float64 array of predicted electron number densities, shape
            ``(batch,)``.
        neutral_fractions : np.ndarray
            Float64 array of predicted neutral fractions, shape
            ``(batch, MAX_ATOMIC_NUMBER)``.

        Raises
        ------
        ValueError
            If *temperatures*, *n_totals*, or *abundances* have incompatible
            shapes.
        """

        temperatures = np.asarray(temperatures, dtype=np.float64)
        n_totals = np.asarray(n_totals, dtype=np.float64)
        abundances = self._prepare_abundances(abundances)

        if temperatures.ndim != 1 or n_totals.ndim != 1:
            raise ValueError("temperatures and n_totals must be 1D arrays.")
        if temperatures.shape[0] != n_totals.shape[0]:
            raise ValueError("temperatures and n_totals must have equal length.")
        if abundances.ndim != 2 or abundances.shape[0] != temperatures.shape[0]:
            raise ValueError("abundances must have shape (batch, 92).")

        log_T = jnp.log10(jnp.asarray(temperatures, dtype=jnp.float64))
        log_n_total = jnp.log10(jnp.asarray(n_totals, dtype=jnp.float64))
        neutral_fractions, ne_logit = self._model_apply(
            log_T,
            log_n_total,
            jnp.asarray(abundances, dtype=jnp.float64),
        )
        ne = jnp.asarray(n_totals, dtype=jnp.float64) * jax.nn.sigmoid(ne_logit)

        return (
            np.asarray(jax.device_get(ne), dtype=np.float64),
            np.asarray(jax.device_get(neutral_fractions), dtype=np.float64),
        )

    def physics_residuals(
        self,
        temperature: float,
        n_total: float,
        abundances: Dict[int, float] | np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Evaluate elemental conservation and charge-neutrality residuals.

        Runs :meth:`solve` to obtain the current PINN predictions, then
        computes how well they satisfy the physics constraints used during
        training.

        Parameters
        ----------
        temperature:
            Gas temperature in Kelvin.
        n_total:
            Total number density in cm\ :sup:`-3`.
        abundances:
            Elemental number fractions (dict or array; see :meth:`solve`).

        Returns
        -------
        element_residuals : np.ndarray
            Float64 array of shape ``(MAX_ATOMIC_NUMBER,)`` giving the
            fractional residual of the element-conservation equation for each
            element.  Values close to zero indicate a well-satisfied constraint.
        charge_residual : float
            Scalar residual of the charge-neutrality equation.  A value close
            to zero indicates that the predicted :math:`n_e` is consistent with
            the predicted ion populations.
        """

        ne, neutral_fractions = self.solve(temperature, n_total, abundances)
        abundances_arr = self._prepare_abundances(abundances)

        elem = element_conservation_residual(
            neutral_fractions,
            ne,
            temperature,
            n_total,
            abundances_arr,
            self.chem_data,
        )
        charge = charge_neutrality_residual(
            neutral_fractions,
            ne,
            temperature,
            n_total,
            abundances_arr,
            self.chem_data,
        )

        return np.asarray(jax.device_get(elem), dtype=np.float64), float(jax.device_get(charge))

    def train(
        self,
        *,
        train_data: Optional[Dict[str, np.ndarray]] = None,
        n_samples: int = 1024,
        optimizer=None,
        batch_size: int = 128,
        supervised_steps: int = 1_000,
        physics_steps: int = 1_000,
        w_element: float = 1.0,
        w_charge: float = 10.0,
        rng_key: Optional[jax.Array] = None,
    ) -> Dict[str, list]:
        """Train the PINN and update ``self.params`` in-place.

        Training proceeds in two phases:

        1. **Supervised phase** – the network is fitted to pre-computed
           chemical-equilibrium solutions (*train_data*) using a mean-squared
           loss in log-fraction space.
        2. **Physics phase** – the residuals of element conservation and charge
           neutrality are minimised without ground-truth labels.

        The trained parameters are stored in ``self.params`` so that subsequent
        calls to :meth:`solve` / :meth:`solve_batch` use the updated weights.

        Parameters
        ----------
        train_data:
            Pre-computed training dataset as a dict with keys
            ``"log_T"``, ``"log_n_total"``, ``"abundances"``,
            ``"neutral_fractions"``, ``"ne"``.
            If ``None``, a dataset of *n_samples* random atmospheres is
            generated automatically via
            :func:`~.chem_eq_pinn_train.generate_training_data`.
        n_samples:
            Number of random training atmospheres to generate when
            *train_data* is ``None``.  Default ``1024``.
        optimizer:
            An Optax-compatible optimizer instance.  When ``None`` a cosine-
            decay Adam schedule is created automatically.
        batch_size:
            Mini-batch size for gradient updates.  Default ``128``.
        supervised_steps:
            Number of supervised gradient steps.  Default ``1000``.
        physics_steps:
            Number of physics-residual gradient steps.  Default ``1000``.
        w_element:
            Loss weight for element-conservation residuals.  Default ``1.0``.
        w_charge:
            Loss weight for charge-neutrality residual.  Default ``10.0``.
        rng_key:
            JAX PRNG key for training stochasticity.  If ``None``, a new key
            is derived from ``self.rng_key``.

        Returns
        -------
        history : dict
            Training loss history with at least the key ``"loss"`` (list of
            float). May also contain ``"supervised_loss"`` and
            ``"physics_loss"`` depending on the trainer implementation.
        """

        if rng_key is None:
            self.rng_key, rng_key = jax.random.split(self.rng_key)

        if train_data is None:
            train_data = generate_training_data(
                n_samples,
                return_jax_arrays=False,
            )

        if optimizer is None:
            total_steps = max(1, int(supervised_steps) + int(physics_steps))
            optimizer = create_optimizer(total_steps=total_steps)

        params, history = train_pinn(
            self.model,
            self.chem_data,
            train_data,
            rng_key=rng_key,
            params=self.params,
            optimizer=optimizer,
            batch_size=batch_size,
            supervised_steps=supervised_steps,
            physics_steps=physics_steps,
            w_element=w_element,
            w_charge=w_charge,
        )
        self.params = params
        return history

    def save(self, path: str) -> str:
        """Persist model parameters and metadata to a compressed ``.npz`` file.

        The file stores:

        * ``hidden_dims`` – integer array of hidden-layer widths.
        * ``params_bytes`` – pickle-serialized model parameters as a uint8
          byte array.

        The ``.npz`` extension is appended automatically if *path* does not
        already end in it.

        Parameters
        ----------
        path:
            Destination file path (with or without the ``.npz`` extension).

        Returns
        -------
        str
            Absolute path of the written checkpoint file.
        """

        path_obj = Path(path)
        if path_obj.suffix != ".npz":
            path_obj = path_obj.with_suffix(".npz")

        payload = _serialize_params(self.params)
        payload_uint8 = np.frombuffer(payload, dtype=np.uint8)

        np.savez_compressed(
            path_obj,
            hidden_dims=np.asarray(self.hidden_dims, dtype=np.int32),
            params_bytes=payload_uint8,
        )

        return str(path_obj)

    @classmethod
    def load(
        cls,
        path: str,
        *,
        chem_data: ChemEqData,
        rng_key: Optional[jax.Array] = None,
    ) -> "ChemicalEquilibriumPINNSolver":
        """Restore a solver from a checkpoint written by :meth:`save`.

        Reads the ``hidden_dims`` and ``params_bytes`` arrays from the
        ``.npz`` file and reconstructs a fully-configured
        :class:`ChemicalEquilibriumPINNSolver` ready for inference.

        Parameters
        ----------
        path:
            Path to the ``.npz`` file to load.
        chem_data:
            Chemical-equilibrium data tables required by the solver.  Must
            be provided explicitly as it is not stored in the checkpoint.
        rng_key:
            Optional JAX PRNG key for the restored solver.  Defaults to
            ``jax.random.PRNGKey(0)`` when ``None``.

        Returns
        -------
        ChemicalEquilibriumPINNSolver
            A new solver instance with the loaded parameters and architecture.
        """

        path_obj = Path(path)
        with np.load(path_obj, allow_pickle=False) as data:
            hidden_dims = tuple(int(v) for v in np.asarray(data["hidden_dims"]).tolist())
            params_bytes = np.asarray(data["params_bytes"], dtype=np.uint8).tobytes()

        params = _deserialize_params(params_bytes)

        return cls(
            chem_data,
            params=params,
            rng_key=rng_key,
            hidden_dims=hidden_dims,
        )


__all__ = [
    "ChemicalEquilibriumPINNSolver",
]
