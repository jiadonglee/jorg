#!/usr/bin/env python3
"""Build an APOGEE quick/formal observation benchmark shard."""

from __future__ import annotations

import argparse
from pathlib import Path

from jorg.apogee.observations import (
    ApogeeObservationConfig,
    build_observation_benchmark,
    filter_quick_benchmark_candidates,
    load_allstar_table,
    select_covering_subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--allstar", default=None, help="Local path or URL to allStarLite/allStar")
    parser.add_argument("--product", choices=("aspcapStar", "apStar"), default="aspcapStar")
    parser.add_argument("--n-stars", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ApogeeObservationConfig(product=args.product, selection_size=args.n_stars)
    allstar_source = args.allstar or config.allstar_url
    allstar_df = load_allstar_table(allstar_source, cache_dir=args.cache_dir)
    candidates = filter_quick_benchmark_candidates(
        allstar_df,
        teff_range=config.teff_range,
        min_snr=config.min_snr,
        require_starflag_zero=config.require_starflag_zero,
        require_aspcapflag_zero=config.require_aspcapflag_zero,
    )
    selected = select_covering_subset(candidates, config.selection_size)
    output_root = args.output_root / "apogee_obs_v1" / args.product
    build_observation_benchmark(
        selected,
        output_root=output_root,
        product=args.product,
        cache_dir=args.cache_dir,
        normalization={
            "percentile": config.percentile,
            "window": config.continuum_window,
            "knot_stride": config.knot_stride,
        },
        apstar_base_url=config.apstar_base_url,
        aspcapstar_base_url=config.aspcapstar_base_url,
    )


if __name__ == "__main__":
    main()
