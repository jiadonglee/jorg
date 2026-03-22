#!/usr/bin/env python3
"""Build one APOGEE synthetic dataset shard for TransformerPayne fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

from jorg.apogee.constants import SynthesisGrid, default_korg_root
from jorg.apogee.exomol import default_exomol_assets, ensure_default_exomol_assets
from jorg.apogee.korg import (
    KorgExoMolLineList,
    build_smoke_requests,
    build_stage_requests,
    build_synthetic_dataset,
    ensure_apogee_water_sigma,
    ensure_korg_environment,
)
from jorg.apogee.sampling import default_stage_definitions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("baseline", "exomol_aug"), default="baseline")
    parser.add_argument("--stage", default="smoke", help="smoke or one of the named training stages")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke-points", type=int, default=256)
    parser.add_argument("--limit-parents", type=int, default=None)
    parser.add_argument("--korg-root", type=Path, default=default_korg_root())
    parser.add_argument("--julia-bin", default="julia")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_korg_environment(args.korg_root, julia_bin=args.julia_bin)
    synthesis_grid = SynthesisGrid()
    water_sigma = ensure_apogee_water_sigma(korg_root=args.korg_root, julia_bin=args.julia_bin, synthesis_grid=synthesis_grid)

    exomol_linelist = ()
    if args.variant == "exomol_aug":
        asset_paths = ensure_default_exomol_assets(args.output_root / "external" / "exomol")
        assets = default_exomol_assets()
        exomol_linelist = tuple(
            KorgExoMolLineList(
                species_name=name,
                states_path=asset_paths[name][0],
                transitions_path=asset_paths[name][1],
                lower_wavelength=asset.lower_wavelength,
                upper_wavelength=asset.upper_wavelength,
                line_strength_cutoff=asset.line_strength_cutoff,
                temperature_line_strength=asset.temperature_line_strength,
            )
            for name, asset in assets.items()
        )

    if args.stage == "smoke":
        requests = build_smoke_requests(water_sigma, n_points=args.smoke_points, seed=args.seed, synthesis_grid=synthesis_grid)
        if args.limit_parents is not None:
            requests = requests[: args.limit_parents]
        for idx, request in enumerate(requests):
            requests[idx] = type(request)(
                **{
                    **request.__dict__,
                    "exomol_linelist": exomol_linelist,
                    "use_exomol_aug": args.variant == "exomol_aug",
                }
            )
        split = {"smoke": list(range(len(requests))), "train": [], "val": [], "test": []}
        dataset_name = "apogee_tp_v1"
    else:
        stages = {stage.name: stage for stage in default_stage_definitions()}
        stage_def = stages[args.stage]
        requests, split, _ = build_stage_requests(stage_def, water_sigma, seed=args.seed, synthesis_grid=synthesis_grid)
        if args.limit_parents is not None:
            requests = requests[: args.limit_parents]
            split = {"train": [i for i in split["train"] if i < len(requests)], "val": [i for i in split["val"] if i < len(requests)], "test": [], "smoke": []}
        for idx, request in enumerate(requests):
            requests[idx] = type(request)(
                **{
                    **request.__dict__,
                    "exomol_linelist": exomol_linelist,
                    "use_exomol_aug": args.variant == "exomol_aug",
                }
            )
        dataset_name = "apogee_tp_v1"

    shard_root = args.output_root / dataset_name / args.variant / args.stage
    build_synthetic_dataset(
        shard_root,
        dataset_name=dataset_name,
        variant=args.variant,
        water_sigma_path=water_sigma,
        requests=requests,
        split=split,
        korg_root=args.korg_root,
        julia_bin=args.julia_bin,
        notes={"limit_parents": args.limit_parents},
    )


if __name__ == "__main__":
    main()
