"""Fine-tuning utilities for APOGEE synthetic grids and TransformerPayne."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

from flax import traverse_util
from flax.core import freeze, unfreeze
from flax.training import train_state
import jax
import jax.numpy as jnp
import joblib
import numpy as np
import optax

from .contracts import load_synthetic_dataset
from .transformer_payne import (
    TPCompactLabelAdapter,
    TransformerPayneDefinition,
    transfer_matching_parameters,
)

TRANSFORMER_PAYNE_REPO_URL = "https://github.com/jiadonglee/transformer_payne"


@dataclass(frozen=True)
class FinetunePhase:
    """One optimizer phase in the staged APOGEE fine-tuning schedule."""

    name: str
    epochs: int
    base_learning_rate: float
    embedding_multiplier: float = 5.0
    head_multiplier: float = 5.0
    freeze_trunk: bool = False


@dataclass(frozen=True)
class FinetuneConfig:
    """Configuration for original TransformerPayne fine-tuning."""

    checkpoint_path: Path
    checkout_dir: Path
    batch_size: int = 8
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 0
    save_every: int = 1
    stage1_phases: tuple[FinetunePhase, ...] = (
        FinetunePhase(name="stage1_joint", epochs=10, base_learning_rate=2e-5),
    )
    stage2_phases: tuple[FinetunePhase, ...] = (
        FinetunePhase(name="stage2_head_only", epochs=5, base_learning_rate=2e-5, freeze_trunk=True),
        FinetunePhase(name="stage2_joint", epochs=10, base_learning_rate=1e-5),
    )


@dataclass
class TPDataset:
    """Dense in-memory dataset for original TP fine-tuning."""

    log_wavelengths: np.ndarray
    compact_labels: np.ndarray
    tp_labels: np.ndarray
    targets: np.ndarray
    split: dict[str, list[int]]
    metadata: dict[str, Any]


class TrainState(train_state.TrainState):
    """Extended train state with a small metrics history payload."""

    metrics: dict[str, float]


def ensure_transformer_payne_checkout(
    checkout_dir: Path | str,
    *,
    repo_url: str = TRANSFORMER_PAYNE_REPO_URL,
    update: bool = False,
) -> Path:
    """Ensure a local checkout of the target TransformerPayne repo exists."""
    checkout_dir = Path(checkout_dir)
    if checkout_dir.exists():
        if update:
            subprocess.run(["git", "-C", str(checkout_dir), "pull", "--ff-only"], check=True)
        return checkout_dir
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", repo_url, str(checkout_dir)], check=True)
    return checkout_dir


def import_transformer_payne(checkout_dir: Path | str):
    """Import the original TransformerPayne package from a local checkout."""
    checkout_dir = Path(checkout_dir)
    src_dir = checkout_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return importlib.import_module("transformer_payne.transformer_payne")


def load_tp_dataset(
    dataset_root: Path | str,
    checkpoint_path: Path | str,
) -> TPDataset:
    """Load one synthetic contract shard and expand compact labels to TP labels."""
    payload = load_synthetic_dataset(dataset_root)
    adapter = TPCompactLabelAdapter.from_checkpoint(checkpoint_path)
    compact_labels = np.asarray(payload["labels"], dtype=np.float32)
    tp_labels = adapter.compact_to_tp(compact_labels).astype(np.float32)
    wavelengths = np.asarray(payload["wavelengths"], dtype=np.float64)
    return TPDataset(
        log_wavelengths=np.log10(wavelengths),
        compact_labels=compact_labels,
        tp_labels=tp_labels,
        targets=np.asarray(payload["targets"], dtype=np.float32),
        split={key: list(value) for key, value in payload["split"].items()},
        metadata=dict(payload["metadata"]),
    )


def create_finetune_model(
    checkpoint_path: Path | str,
    checkout_dir: Path | str,
):
    """Instantiate the original TP model and return (definition, model, params)."""
    definition = TransformerPayneDefinition.from_checkpoint(checkpoint_path)
    module = import_transformer_payne(checkout_dir)
    model = module.TransformerPayneModel(**definition.architecture_parameters)
    params = freeze(joblib.load(checkpoint_path)["emulator_weights"])
    return definition, model, params


def _param_group_for_path(path: tuple[str, ...]) -> str:
    joined = "/".join(path)
    if "PredictionHead_0" in joined:
        return "head"
    if "ParametersEmbedding_0" in joined:
        return "embedding"
    return "trunk"


def create_group_labels(params: Mapping[str, Any], freeze_trunk: bool = False):
    """Create an Optax label tree for TP parameter groups."""
    flat = traverse_util.flatten_dict(unfreeze(params), keep_empty_nodes=True)
    labels = {}
    for path in flat:
        group = _param_group_for_path(tuple(str(item) for item in path))
        if freeze_trunk and group == "trunk":
            group = "frozen"
        labels[path] = group
    return freeze(traverse_util.unflatten_dict(labels))


def create_optimizer(
    params: Mapping[str, Any],
    phase: FinetunePhase,
    *,
    weight_decay: float,
    gradient_clip: float,
):
    """Build a multi-group optimizer for one fine-tuning phase."""
    transforms = {
        "trunk": optax.adamw(phase.base_learning_rate, weight_decay=weight_decay),
        "embedding": optax.adamw(
            phase.base_learning_rate * phase.embedding_multiplier,
            weight_decay=weight_decay,
        ),
        "head": optax.adamw(
            phase.base_learning_rate * phase.head_multiplier,
            weight_decay=weight_decay,
        ),
        "frozen": optax.set_to_zero(),
    }
    labels = create_group_labels(params, freeze_trunk=phase.freeze_trunk)
    return optax.chain(
        optax.clip_by_global_norm(gradient_clip),
        optax.multi_transform(transforms, labels),
    )


def _make_tx(
    params: Mapping[str, Any],
    phase: FinetunePhase,
    *,
    weight_decay: float,
    gradient_clip: float,
):
    return create_optimizer(
        params,
        phase,
        weight_decay=weight_decay,
        gradient_clip=gradient_clip,
    )


def create_train_state(
    model,
    params,
    phase: FinetunePhase,
    *,
    weight_decay: float,
    gradient_clip: float,
) -> TrainState:
    tx = _make_tx(params, phase, weight_decay=weight_decay, gradient_clip=gradient_clip)
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        metrics={"loss": np.nan, "rmse": np.nan},
    )


def batched_tp_apply(model, params, log_wavelengths: np.ndarray, tp_labels: np.ndarray, *, train: bool):
    """Apply the original TP model over a batch of 95-D labels."""
    log_wavelengths_j = jnp.asarray(log_wavelengths, dtype=jnp.float32)
    labels_j = jnp.asarray(tp_labels, dtype=jnp.float32)

    def apply_one(label_vec):
        return model.apply({"params": params}, (log_wavelengths_j, label_vec), train=train)

    return jax.vmap(apply_one, in_axes=0, out_axes=0)(labels_j)


def _train_step(state: TrainState, model, log_wavelengths, labels, targets):
    targets_j = jnp.asarray(targets, dtype=jnp.float32)

    def loss_fn(params):
        predictions = batched_tp_apply(model, params, log_wavelengths, labels, train=True)
        residual = predictions - targets_j
        loss = jnp.mean(residual * residual)
        rmse = jnp.sqrt(loss)
        return loss, rmse

    (loss, rmse), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(
        metrics={
            "loss": float(np.asarray(loss)),
            "rmse": float(np.asarray(rmse)),
        }
    )
    return state


def _eval_step(params, model, log_wavelengths, labels, targets):
    targets_j = jnp.asarray(targets, dtype=jnp.float32)
    predictions = batched_tp_apply(model, params, log_wavelengths, labels, train=False)
    residual = predictions - targets_j
    loss = jnp.mean(residual * residual)
    rmse = jnp.sqrt(loss)
    mae = jnp.mean(jnp.abs(residual))
    return {
        "loss": float(np.asarray(loss)),
        "rmse": float(np.asarray(rmse)),
        "mae": float(np.asarray(mae)),
    }


def iter_batches(indices: Iterable[int], batch_size: int, *, shuffle: bool, seed: int) -> Iterable[np.ndarray]:
    """Yield contiguous or shuffled mini-batches of indices."""
    indices_arr = np.asarray(list(indices), dtype=np.int64)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices_arr)
    for start in range(0, len(indices_arr), batch_size):
        yield indices_arr[start : start + batch_size]


def _to_host_tree(tree: Any) -> Any:
    if isinstance(tree, Mapping):
        return {key: _to_host_tree(value) for key, value in tree.items()}
    return np.asarray(jax.device_get(tree))


def _save_snapshot(output_dir: Path, name: str, state: TrainState) -> Path:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": _to_host_tree(unfreeze(state.params)),
        "metrics": {key: float(value) for key, value in state.metrics.items()},
    }
    path = checkpoint_dir / f"{name}.joblib"
    joblib.dump(payload, path)
    return path


def run_finetune(
    dataset: TPDataset,
    *,
    output_dir: Path | str,
    config: FinetuneConfig,
    stage: str,
) -> dict[str, Any]:
    """Fine-tune the original TP model on one synthetic shard."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_transformer_payne_checkout(config.checkout_dir)
    definition, model, checkpoint_params = create_finetune_model(
        config.checkpoint_path,
        config.checkout_dir,
    )
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unsupported fine-tuning stage: {stage}")

    model_init = model.init(
        jax.random.PRNGKey(config.seed),
        (
            jnp.asarray(dataset.log_wavelengths, dtype=jnp.float32),
            jnp.asarray(dataset.tp_labels[0], dtype=jnp.float32),
        ),
        train=False,
    )["params"]
    transferred_params, transfer_stats = transfer_matching_parameters(
        unfreeze(checkpoint_params),
        unfreeze(model_init),
    )
    params = freeze(transferred_params)

    phases = config.stage1_phases if stage == "stage1" else config.stage2_phases
    state = create_train_state(
        model,
        params,
        phases[0],
        weight_decay=config.weight_decay,
        gradient_clip=config.gradient_clip,
    )

    history: list[dict[str, Any]] = []
    train_indices = dataset.split.get("train", [])
    val_indices = dataset.split.get("val", [])
    global_epoch = 0

    for phase in phases:
        tx = _make_tx(
            state.params,
            phase,
            weight_decay=config.weight_decay,
            gradient_clip=config.gradient_clip,
        )
        state = state.replace(tx=tx, opt_state=tx.init(state.params))
        for epoch in range(1, phase.epochs + 1):
            global_epoch += 1
            for batch in iter_batches(train_indices, config.batch_size, shuffle=True, seed=config.seed + epoch):
                state = _train_step(
                    state,
                    model,
                    dataset.log_wavelengths,
                    dataset.tp_labels[batch],
                    dataset.targets[batch],
                )
            val_metrics = _eval_step(
                state.params,
                model,
                dataset.log_wavelengths,
                dataset.tp_labels[val_indices],
                dataset.targets[val_indices],
            ) if val_indices else {"loss": np.nan, "rmse": np.nan, "mae": np.nan}
            history.append(
                {
                    "phase": phase.name,
                    "epoch": epoch,
                    "global_epoch": global_epoch,
                    "train_loss": state.metrics["loss"],
                    "train_rmse": state.metrics["rmse"],
                    "val_loss": val_metrics["loss"],
                    "val_rmse": val_metrics["rmse"],
                    "val_mae": val_metrics["mae"],
                }
            )
            if config.save_every > 0 and global_epoch % config.save_every == 0:
                _save_snapshot(output_dir, f"{phase.name}_epoch{epoch:03d}", state)

    with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    with (output_dir / "transfer_stats.json").open("w", encoding="utf-8") as handle:
        json.dump(transfer_stats, handle, indent=2)
    with (output_dir / "finetune_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, default=str)
    _save_snapshot(output_dir, "final", state)
    return {
        "history": history,
        "transfer_stats": transfer_stats,
        "final_metrics": {key: float(value) for key, value in state.metrics.items()},
        "definition": definition,
    }


def overfit_small_subset(
    dataset: TPDataset,
    *,
    output_dir: Path | str,
    config: FinetuneConfig,
    n_samples: int = 32,
) -> dict[str, Any]:
    """Utility used by tests and smoke runs for the 32-sample overfit check."""
    subset_indices = dataset.split.get("train", [])[:n_samples]
    small_dataset = TPDataset(
        log_wavelengths=dataset.log_wavelengths,
        compact_labels=dataset.compact_labels[subset_indices],
        tp_labels=dataset.tp_labels[subset_indices],
        targets=dataset.targets[subset_indices],
        split={"train": list(range(len(subset_indices))), "val": list(range(len(subset_indices))), "test": [], "smoke": []},
        metadata=dict(dataset.metadata),
    )
    return run_finetune(small_dataset, output_dir=output_dir, config=config, stage="stage1")
