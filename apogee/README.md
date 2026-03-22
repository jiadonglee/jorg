# APOGEE Workflow

This folder is the top-level workspace for the APOGEE M-star pipeline.

Structure:

- `scripts/`: entry points for Korg preprocessing, synthetic-grid generation, APOGEE benchmark downloads, and TransformerPayne fine-tuning.
- `output/`: default location for generated APOGEE-side artifacts such as `apogee_water_sigma.h5`, synthetic grids, and fine-tuning outputs.
- `external/`: recommended cache location for downloaded third-party assets such as ExoMol line lists and SDSS products.

Python implementation lives in `src/jorg/apogee/`, while this folder keeps the workflow-specific scripts and runtime products grouped under one top-level directory.
