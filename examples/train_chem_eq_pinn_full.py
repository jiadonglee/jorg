#!/usr/bin/env python3
"""Full two-phase training pipeline for chemical-equilibrium PINN.

This script runs:
1) data generation (train/validation),
2) supervised pre-training,
3) physics fine-tuning,
4) validation against solver labels and physics residuals,
5) checkpoint + summary export.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def _ensure_src_on_path() -> Path:
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        src_path = parent / "src"
        if (src_path / "jorg").exists():
            sys.path.insert(0, str(src_path))
            data_dir = parent / "data"
            if data_dir.exists():
                os.environ.setdefault("JORG_DATA_DIR", str(data_dir))
            return parent
    raise RuntimeError("Could not locate src/jorg relative to this script")


REPO_ROOT = _ensure_src_on_path()

import jax
import jax.numpy as jnp

from jorg.statmech import create_default_ionization_energies, create_default_partition_functions
from jorg.statmech.chem_eq_jax import prepare_chem_eq_data
from jorg.statmech.chem_eq_pinn import ChemicalEquilibriumPINNSolver
from jorg.statmech.chem_eq_pinn_loss import charge_neutrality_residual, element_conservation_residual
from jorg.statmech.chem_eq_pinn_models import ChemicalEquilibriumPINN, USING_FLAX_BACKEND
from jorg.statmech.chem_eq_pinn_train import (
    create_optimizer,
    generate_training_data,
    initialize_model_params,
    make_phase2_joint_train_step,
    make_physics_train_step,
    make_supervised_train_step,
    phase2_joint_objective,
    physics_objective,
    supervised_objective,
)

DATA_KEYS = (
    "T",
    "log_T",
    "n_total",
    "log_n_total",
    "abundances",
    "neutral_fractions",
    "ne",
)


def _parse_hidden_dims(raw: str) -> Tuple[int, ...]:
    vals = [int(v.strip()) for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("hidden_dims cannot be empty")
    return tuple(vals)


def _to_jax_batch(data: Dict[str, np.ndarray]) -> Dict[str, jnp.ndarray]:
    return {k: jnp.asarray(v, dtype=jnp.float64) for k, v in data.items()}


def _take_batch(data: Dict[str, jnp.ndarray], indices: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    return {k: jnp.take(v, indices, axis=0) for k, v in data.items()}


def _save_dataset(path: Path, data: Dict[str, np.ndarray]) -> None:
    payload = {k: np.asarray(data[k], dtype=np.float64) for k in DATA_KEYS}
    np.savez_compressed(path, **payload)


def _load_dataset(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        missing = [k for k in DATA_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"Dataset file missing keys {missing}: {path}")
        return {k: np.asarray(z[k], dtype=np.float64) for k in DATA_KEYS}


def _compute_validation_metrics(
    solver: ChemicalEquilibriumPINNSolver,
    val_data: Dict[str, np.ndarray],
    chem_data,
) -> Dict[str, float]:
    ne_pred, f_pred = solver.solve_batch(val_data["T"], val_data["n_total"], val_data["abundances"])
    ne_true = val_data["ne"]
    f_true = val_data["neutral_fractions"]
    abund = val_data["abundances"]

    ne_rel = np.abs(ne_pred - ne_true) / np.maximum(ne_true, 1e-30)

    f_abs = np.abs(f_pred - f_true)
    f_rel = f_abs / np.maximum(f_true, 1e-30)
    abundant_mask = abund > 1e-12

    elem_resid = np.asarray(
        jax.device_get(
            element_conservation_residual(
                f_pred,
                ne_pred,
                val_data["T"],
                val_data["n_total"],
                abund,
                chem_data,
            )
        ),
        dtype=np.float64,
    )
    charge_resid = np.asarray(
        jax.device_get(
            charge_neutrality_residual(
                f_pred,
                ne_pred,
                val_data["T"],
                val_data["n_total"],
                abund,
                chem_data,
            )
        ),
        dtype=np.float64,
    )

    metrics = {
        "val_ne_rel_mean": float(np.mean(ne_rel)),
        "val_ne_rel_p95": float(np.percentile(ne_rel, 95.0)),
        "val_ne_rel_max": float(np.max(ne_rel)),
        "val_f_abs_mean": float(np.mean(f_abs)),
        "val_f_abs_p95": float(np.percentile(f_abs, 95.0)),
        "val_f_rel_mean_abundant": float(np.mean(f_rel[abundant_mask])),
        "val_f_rel_p95_abundant": float(np.percentile(f_rel[abundant_mask], 95.0)),
        "val_elem_resid_abs_mean": float(np.mean(np.abs(elem_resid))),
        "val_elem_resid_abs_max": float(np.max(np.abs(elem_resid))),
        "val_charge_resid_abs_mean": float(np.mean(np.abs(charge_resid))),
        "val_charge_resid_abs_max": float(np.max(np.abs(charge_resid))),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Full training for chemical-equilibrium PINN")

    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoint and logs")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory for prepared train/val data npz files")
    parser.add_argument(
        "--init-model-path",
        type=str,
        default=None,
        help="Optional checkpoint (.npz) to initialize model weights (and hidden_dims) before training.",
    )
    parser.add_argument(
        "--prepare-data-only",
        action="store_true",
        help="Only generate and save train/val datasets, then exit.",
    )
    parser.add_argument(
        "--load-data",
        action="store_true",
        help="Load train/val datasets from --data-dir instead of regenerating.",
    )
    parser.add_argument(
        "--save-generated-data",
        action="store_true",
        help="When generating data in training mode, also save train/val npz files.",
    )
    parser.add_argument("--train-samples", type=int, default=10_000)
    parser.add_argument("--val-samples", type=int, default=512)
    parser.add_argument("--solver-backend", type=str, default="jax", choices=["jax", "scipy"])
    parser.add_argument("--max-attempt-factor", type=int, default=8)

    parser.add_argument("--hidden-dims", type=str, default="256,512,512,256,128")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--supervised-steps", type=int, default=50_000)
    parser.add_argument("--physics-steps", type=int, default=50_000)
    parser.add_argument("--w-element", type=float, default=1.0)
    parser.add_argument("--w-charge", type=float, default=10.0)
    parser.add_argument(
        "--w-supervised-phase2",
        type=float,
        default=1.0,
        help="Supervised-loss weight in phase-2 joint objective; set 0 for physics-only phase-2.",
    )

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Global-norm gradient clip for supervised phase optimizer; <=0 disables.",
    )
    parser.add_argument(
        "--phase2-lr-scale",
        type=float,
        default=0.02,
        help="Phase-2 learning-rate scale relative to --learning-rate.",
    )
    parser.add_argument(
        "--phase2-grad-clip",
        type=float,
        default=1.0,
        help="Global-norm gradient clip for phase-2 optimizer; <=0 disables.",
    )
    parser.add_argument(
        "--phase2-early-stop-patience",
        type=int,
        default=10,
        help="Early-stop patience in validation checkpoints during phase-2; <=0 disables.",
    )
    parser.add_argument(
        "--phase2-early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum val-supervised improvement to reset phase-2 early-stop counter.",
    )
    parser.add_argument(
        "--phase2-restore-best",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Restore best phase-2 parameters by validation supervised loss at the end.",
    )

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--report-every", type=int, default=1_000)

    args = parser.parse_args()

    if args.prepare_data_only and args.load_data:
        raise ValueError("--prepare-data-only and --load-data cannot be used together.")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "output" / "pinn" / f"run_{run_stamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir) if args.data_dir else (output_dir / "prepared_data")
    train_data_path = data_dir / "train_data.npz"
    val_data_path = data_dir / "val_data.npz"

    print("=" * 80)
    print("Chemical equilibrium PINN full training")
    print(f"repo_root          : {REPO_ROOT}")
    print(f"output_dir         : {output_dir}")
    print(f"data_dir           : {data_dir}")
    print(f"jax backend        : {jax.default_backend()}")
    print(f"model backend      : {'flax' if USING_FLAX_BACKEND else 'pure-jax-fallback'}")
    print("input log base     : 10")
    print(f"train/val samples  : {args.train_samples}/{args.val_samples}")
    print(f"steps (sup/phys)   : {args.supervised_steps}/{args.physics_steps}")
    print(
        "phase2 objective   : "
        + ("joint(sup+phys)" if args.w_supervised_phase2 > 0.0 else "physics-only")
        + f"  (w_sup2={args.w_supervised_phase2:.3g}, w_elem={args.w_element:.3g}, w_charge={args.w_charge:.3g})"
    )
    print(
        "optimizer settings : "
        f"sup(lr={args.learning_rate:.3e}, clip={args.grad_clip:.3g}) "
        f"phase2(lr_scale={args.phase2_lr_scale:.3g}, clip={args.phase2_grad_clip:.3g})"
    )
    print(
        "phase2 stability   : "
        f"early_stop_patience={args.phase2_early_stop_patience}, "
        f"min_delta={args.phase2_early_stop_min_delta:.3g}, "
        f"restore_best={args.phase2_restore_best}"
    )
    print("=" * 80)

    t0_all = time.perf_counter()

    print("[1/5] Loading chemistry data...")
    partition_funcs = create_default_partition_functions()
    ionization_energies = create_default_ionization_energies()
    chem_data = prepare_chem_eq_data(
        ionization_energies,
        partition_funcs,
        log_equilibrium_constants=None,
    )

    print("[2/5] Preparing train/validation data...")
    t0_data = time.perf_counter()
    data_source = "generated"
    if args.load_data:
        if not train_data_path.exists() or not val_data_path.exists():
            raise FileNotFoundError(
                f"--load-data requested but dataset files are missing:\n"
                f"  {train_data_path}\n  {val_data_path}"
            )
        train_data_np = _load_dataset(train_data_path)
        val_data_np = _load_dataset(val_data_path)
        data_source = "loaded"
        print(f"    loaded train data: {train_data_path}")
        print(f"    loaded val data  : {val_data_path}")
    else:
        train_data_np = generate_training_data(
            args.train_samples,
            ionization_energies=ionization_energies,
            partition_funcs=partition_funcs,
            log_equilibrium_constants=None,
            solver_backend=args.solver_backend,
            rng_seed=args.data_seed,
            max_attempt_factor=args.max_attempt_factor,
            return_jax_arrays=False,
        )
        val_data_np = generate_training_data(
            args.val_samples,
            ionization_energies=ionization_energies,
            partition_funcs=partition_funcs,
            log_equilibrium_constants=None,
            solver_backend=args.solver_backend,
            rng_seed=args.data_seed + 1,
            max_attempt_factor=args.max_attempt_factor,
            return_jax_arrays=False,
        )
        if args.prepare_data_only or args.save_generated_data:
            data_dir.mkdir(parents=True, exist_ok=True)
            _save_dataset(train_data_path, train_data_np)
            _save_dataset(val_data_path, val_data_np)
            meta = {
                "timestamp": datetime.now().isoformat(),
                "solver_backend": args.solver_backend,
                "train_samples": int(train_data_np["T"].shape[0]),
                "val_samples": int(val_data_np["T"].shape[0]),
                "data_seed": int(args.data_seed),
                "max_attempt_factor": int(args.max_attempt_factor),
                "train_data_path": str(train_data_path),
                "val_data_path": str(val_data_path),
            }
            (data_dir / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"    saved train data: {train_data_path}")
            print(f"    saved val data  : {val_data_path}")
            print(f"    saved metadata  : {data_dir / 'dataset_metadata.json'}")

    data_sec = time.perf_counter() - t0_data
    if data_source == "loaded":
        print(f"    data loading done in {data_sec:.1f} s")
    else:
        print(f"    data generation done in {data_sec:.1f} s")

    if args.prepare_data_only:
        print("Prepare-data-only mode complete. Exiting before training.")
        return

    effective_train_samples = int(train_data_np["T"].shape[0])
    effective_val_samples = int(val_data_np["T"].shape[0])
    print(f"    effective train/val samples: {effective_train_samples}/{effective_val_samples}")

    print("[3/5] Building model and optimizer...")
    rng = jax.random.PRNGKey(args.seed)
    loaded_solver = None
    if args.init_model_path:
        init_model_path = Path(args.init_model_path)
        if not init_model_path.exists():
            raise FileNotFoundError(f"--init-model-path does not exist: {init_model_path}")
        loaded_solver = ChemicalEquilibriumPINNSolver.load(str(init_model_path), chem_data=chem_data, rng_key=rng)
        hidden_dims = tuple(int(v) for v in loaded_solver.hidden_dims)
        print(f"    initialized from checkpoint: {init_model_path}")
    else:
        hidden_dims = _parse_hidden_dims(args.hidden_dims)

    model = ChemicalEquilibriumPINN(hidden_dims=hidden_dims)
    rng, init_key = jax.random.split(rng)
    if loaded_solver is not None:
        params = loaded_solver.params
    else:
        params = initialize_model_params(model, init_key)

    # Use separate optimizers for the two phases to prevent phase-2 instability.
    optimizer_sup = create_optimizer(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        total_steps=max(1, int(args.supervised_steps)),
        warmup_steps=args.warmup_steps,
        gradient_clip_norm=args.grad_clip,
    )
    opt_state_sup = optimizer_sup.init(params)
    phase2_learning_rate = float(args.learning_rate) * float(args.phase2_lr_scale)
    phase2_warmup_steps = min(max(0, int(args.warmup_steps) // 10), max(0, int(args.physics_steps) - 1))
    optimizer_phase2 = create_optimizer(
        learning_rate=phase2_learning_rate,
        weight_decay=args.weight_decay,
        total_steps=max(1, int(args.physics_steps)),
        warmup_steps=phase2_warmup_steps,
        gradient_clip_norm=args.phase2_grad_clip,
    )

    train_data = _to_jax_batch(train_data_np)
    val_data = _to_jax_batch(val_data_np)
    n_train = int(train_data["T"].shape[0])

    supervised_step = make_supervised_train_step(model, optimizer_sup)
    phase2_joint = float(args.w_supervised_phase2) > 0.0
    if phase2_joint:
        phase2_step = make_phase2_joint_train_step(
            model,
            optimizer_phase2,
            chem_data,
            phase2_supervised_weight=args.w_supervised_phase2,
            w_element=args.w_element,
            w_charge=args.w_charge,
        )
    else:
        phase2_step = make_physics_train_step(
            model,
            optimizer_phase2,
            chem_data,
            w_element=args.w_element,
            w_charge=args.w_charge,
        )

    report_every = max(1, int(args.report_every))
    batch_size = max(1, int(args.batch_size))

    history_supervised = []
    history_physics = []
    val_supervised = []
    val_physics = []

    print("[4/5] Supervised pre-training...")
    t0_sup = time.perf_counter()
    for step in range(args.supervised_steps):
        rng, sk = jax.random.split(rng)
        idx = jax.random.randint(sk, shape=(batch_size,), minval=0, maxval=n_train)
        batch = _take_batch(train_data, idx)
        params, opt_state_sup, loss, _ = supervised_step(params, opt_state_sup, batch)
        loss_value = float(jax.device_get(loss))
        history_supervised.append(loss_value)

        if step == 0 or (step + 1) % report_every == 0 or (step + 1) == args.supervised_steps:
            val_loss, _ = supervised_objective(params, val_data, model)
            val_loss_value = float(jax.device_get(val_loss))
            val_supervised.append([step + 1, val_loss_value])
            print(
                f"    [sup] step {step + 1:>6d}/{args.supervised_steps:<6d} "
                f"train={loss_value:.3e} val={val_loss_value:.3e}"
            )

    sup_sec = time.perf_counter() - t0_sup

    print("[4/5] Physics fine-tuning...")
    t0_phys = time.perf_counter()
    opt_state_phase2 = optimizer_phase2.init(params)
    phase2_steps_run = 0
    best_phase2_step = 0
    best_val_supervised = float(jax.device_get(supervised_objective(params, val_data, model)[0]))
    best_phase2_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
    stale_phase2_evals = 0
    print(f"    phase2 initial val supervised: {best_val_supervised:.3e}")

    for step in range(args.physics_steps):
        rng, sk = jax.random.split(rng)
        idx = jax.random.randint(sk, shape=(batch_size,), minval=0, maxval=n_train)
        batch = _take_batch(train_data, idx)
        params, opt_state_phase2, loss, metrics = phase2_step(params, opt_state_phase2, batch)
        phase2_steps_run = step + 1

        loss_value = float(jax.device_get(loss))
        elem_value = float(jax.device_get(metrics["element"]))
        charge_value = float(jax.device_get(metrics["charge"]))
        sup2_value = float(jax.device_get(metrics["supervised_total"])) if "supervised_total" in metrics else 0.0
        phys_value = float(jax.device_get(metrics["physics_total"])) if "physics_total" in metrics else loss_value
        history_physics.append([loss_value, elem_value, charge_value, sup2_value, phys_value])

        if step == 0 or (step + 1) % report_every == 0 or (step + 1) == args.physics_steps:
            val_sup_loss, _ = supervised_objective(params, val_data, model)
            val_sup_value = float(jax.device_get(val_sup_loss))
            if phase2_joint:
                val_loss, val_metrics = phase2_joint_objective(
                    params,
                    val_data,
                    chem_data,
                    model,
                    phase2_supervised_weight=args.w_supervised_phase2,
                    w_element=args.w_element,
                    w_charge=args.w_charge,
                )
            else:
                val_loss, val_metrics = physics_objective(
                    params,
                    val_data,
                    chem_data,
                    model,
                    w_element=args.w_element,
                    w_charge=args.w_charge,
                )
            val_loss_value = float(jax.device_get(val_loss))
            val_elem_value = float(jax.device_get(val_metrics["element"]))
            val_charge_value = float(jax.device_get(val_metrics["charge"]))
            val_sup2_value = (
                float(jax.device_get(val_metrics["supervised_total"]))
                if "supervised_total" in val_metrics
                else 0.0
            )
            val_phys_value = (
                float(jax.device_get(val_metrics["physics_total"]))
                if "physics_total" in val_metrics
                else val_loss_value
            )
            val_physics.append(
                [step + 1, val_loss_value, val_elem_value, val_charge_value, val_sup2_value, val_phys_value, val_sup_value]
            )
            improved = val_sup_value < (best_val_supervised - float(args.phase2_early_stop_min_delta))
            if improved:
                best_val_supervised = val_sup_value
                best_phase2_step = step + 1
                best_phase2_params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)
                stale_phase2_evals = 0
            else:
                stale_phase2_evals += 1

            if phase2_joint:
                print(
                    f"    [phy] step {step + 1:>6d}/{args.physics_steps:<6d} "
                    f"train_total={loss_value:.3e} (sup2={sup2_value:.3e}, phys={phys_value:.3e}, "
                    f"e={elem_value:.3e}, c={charge_value:.3e}) "
                    f"val_total={val_loss_value:.3e} (sup2={val_sup2_value:.3e}, phys={val_phys_value:.3e}, "
                    f"e={val_elem_value:.3e}, c={val_charge_value:.3e}, sup={val_sup_value:.3e})"
                )
            else:
                print(
                    f"    [phy] step {step + 1:>6d}/{args.physics_steps:<6d} "
                    f"train={loss_value:.3e} (e={elem_value:.3e}, c={charge_value:.3e}) "
                    f"val={val_loss_value:.3e} (e={val_elem_value:.3e}, c={val_charge_value:.3e}, sup={val_sup_value:.3e})"
                )
            if int(args.phase2_early_stop_patience) > 0 and stale_phase2_evals >= int(args.phase2_early_stop_patience):
                print(
                    f"    [phy] early stop at step {step + 1} after "
                    f"{stale_phase2_evals} non-improving validations; "
                    f"best val supervised={best_val_supervised:.3e} at step {best_phase2_step}"
                )
                break

    phys_sec = time.perf_counter() - t0_phys
    if args.phase2_restore_best and phase2_steps_run > 0:
        params = best_phase2_params
        print(
            f"    restored best phase2 params from step {best_phase2_step} "
            f"(best val supervised={best_val_supervised:.3e})"
        )

    print("[5/5] Validation and checkpoint export...")
    solver = ChemicalEquilibriumPINNSolver(
        chem_data,
        params=params,
        hidden_dims=hidden_dims,
        rng_key=rng,
    )

    model_path = solver.save(str(output_dir / "chem_eq_pinn_model.npz"))
    metrics = _compute_validation_metrics(solver, val_data_np, chem_data)

    np.savez_compressed(
        output_dir / "loss_history.npz",
        supervised=np.asarray(history_supervised, dtype=np.float64),
        physics=np.asarray(history_physics, dtype=np.float64),
        val_supervised=np.asarray(val_supervised, dtype=np.float64),
        val_physics=np.asarray(val_physics, dtype=np.float64),
    )

    total_sec = time.perf_counter() - t0_all

    summary = {
        "timestamp": datetime.now().isoformat(),
        "repo_root": str(REPO_ROOT),
        "output_dir": str(output_dir),
        "model_path": str(model_path),
        "jax_backend": jax.default_backend(),
        "model_backend": "flax" if USING_FLAX_BACKEND else "pure-jax-fallback",
        "config": {
            "input_log_base": 10,
            "data_source": data_source,
            "data_dir": str(data_dir),
            "load_data": bool(args.load_data),
            "save_generated_data": bool(args.save_generated_data),
            "train_samples_requested": int(args.train_samples),
            "val_samples_requested": int(args.val_samples),
            "train_samples_effective": effective_train_samples,
            "val_samples_effective": effective_val_samples,
            "solver_backend": args.solver_backend,
            "max_attempt_factor": int(args.max_attempt_factor),
            "hidden_dims": list(hidden_dims),
            "init_model_path": str(args.init_model_path) if args.init_model_path else None,
            "batch_size": int(args.batch_size),
            "supervised_steps": int(args.supervised_steps),
            "physics_steps": int(args.physics_steps),
            "w_element": float(args.w_element),
            "w_charge": float(args.w_charge),
            "w_supervised_phase2": float(args.w_supervised_phase2),
            "phase2_objective_mode": "joint" if phase2_joint else "physics_only",
            "learning_rate": float(args.learning_rate),
            "phase2_learning_rate": float(phase2_learning_rate),
            "phase2_lr_scale": float(args.phase2_lr_scale),
            "weight_decay": float(args.weight_decay),
            "warmup_steps": int(args.warmup_steps),
            "phase2_warmup_steps": int(phase2_warmup_steps),
            "grad_clip": float(args.grad_clip),
            "phase2_grad_clip": float(args.phase2_grad_clip),
            "phase2_early_stop_patience": int(args.phase2_early_stop_patience),
            "phase2_early_stop_min_delta": float(args.phase2_early_stop_min_delta),
            "phase2_restore_best": bool(args.phase2_restore_best),
            "phase2_steps_run": int(phase2_steps_run),
            "phase2_best_step_by_val_supervised": int(best_phase2_step),
            "phase2_best_val_supervised": float(best_val_supervised),
            "seed": int(args.seed),
            "data_seed": int(args.data_seed),
            "report_every": int(report_every),
        },
        "timing_sec": {
            "data_generation": float(data_sec),
            "supervised_phase": float(sup_sec),
            "physics_phase": float(phys_sec),
            "total": float(total_sec),
        },
        "train_loss": {
            "supervised_initial": float(history_supervised[0]) if history_supervised else None,
            "supervised_final": float(history_supervised[-1]) if history_supervised else None,
            "physics_initial": float(history_physics[0][0]) if history_physics else None,
            "physics_final": float(history_physics[-1][0]) if history_physics else None,
            "physics_element_final": float(history_physics[-1][1]) if history_physics else None,
            "physics_charge_final": float(history_physics[-1][2]) if history_physics else None,
            "phase2_supervised_final": float(history_physics[-1][3]) if history_physics else None,
            "phase2_physics_total_final": float(history_physics[-1][4]) if history_physics else None,
        },
        "validation": metrics,
    }

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"  model checkpoint : {model_path}")
    print(f"  loss history     : {output_dir / 'loss_history.npz'}")
    print(f"  summary          : {summary_path}")
    print("Validation snapshot:")
    print(
        "  ne_rel_mean={val_ne_rel_mean:.3e}, ne_rel_p95={val_ne_rel_p95:.3e}, "
        "elem_resid_max={val_elem_resid_abs_max:.3e}, charge_resid_max={val_charge_resid_abs_max:.3e}".format(
            **metrics
        )
    )


if __name__ == "__main__":
    main()
