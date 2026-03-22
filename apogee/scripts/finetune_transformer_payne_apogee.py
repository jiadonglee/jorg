#!/usr/bin/env python3
"""Fine-tune the original TransformerPayne intensity model on APOGEE synthetic data."""

from __future__ import annotations

import argparse
from pathlib import Path

from jorg.apogee.finetune import (
    FinetuneConfig,
    load_tp_dataset,
    run_finetune,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--checkout-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_tp_dataset(args.dataset_root, args.checkpoint_path)
    config = FinetuneConfig(
        checkpoint_path=args.checkpoint_path,
        checkout_dir=args.checkout_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    run_finetune(dataset, output_dir=args.output_dir, config=config, stage=args.stage)


if __name__ == "__main__":
    main()
