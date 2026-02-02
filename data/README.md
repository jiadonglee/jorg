# Jorg Data Bundle

This directory is intentionally gitignored. Place external data files here
so Jorg can find them via `JORG_DATA_DIR`.

Recommended layout:
- `vald_extract_stellar_solar_threshold001.vald`
- `korg_partition_functions.json`
- `atomic_partition_funcs/partition_funcs.h5`
- `barklem_collet_2016/`
- `bf_cross-sections/`
- `Stehle-Hutchson-hydrogen-profiles.h5`
- `McLaughlin2017Hminusbf.h5`
- `vanHoof2014-nr-gauntff.dat`
- `marcs_grids/` (large; set `JORG_MARCS_GRID_DIR` if stored elsewhere)

Set environment variables:
```bash
export JORG_DATA_DIR=/path/to/jorg/data
export JORG_MARCS_GRID_DIR=/path/to/marcs_grids  # optional
```
