# APOGEE Work Status

Date: 2026-03-16

## Scope

This status file records the current state of the APOGEE M-star workflow under `/Users/jdli/Project/jorg/jorg/apogee`.

The current v1 goal is still:

- build a working APOGEE synthetic-spectrum pipeline around local Korg.jl
- add an optional ExoMol augmentation path
- prepare the code path for APOGEE benchmarking and TransformerPayne fine-tuning

## Current Status

Completed:

- baseline APOGEE synthesis through local Julia/Korg bridge
- APOGEE H2O cross-section generation from local POKAZATEL input
- optional `exomol_aug` path using `CaH` and `FeH`
- automatic ExoMol asset download and decompression to Korg-readable `.states` / `.trans`
- notebook demonstration for `Teff`, `logg`, and `[M/H]` sweeps
- notebook comparison between `baseline` and `exomol_aug`
- automatic CSV report listing all ExoMol-added lines
- TransformerPayne compact-label adapter and fine-tuning code path
- unit tests for APOGEE contracts, observations helpers, TransformerPayne integration, and ExoMol reporting

Implemented but not yet run in full production mode:

- APOGEE observation download and benchmark construction
- full synthetic-grid production for training
- formal TransformerPayne fine-tuning on the APOGEE synthetic grid
- observed-vs-synthetic residual benchmark to decide whether `exomol_aug` should replace `baseline`

Deferred on purpose:

- actual APOGEE spectrum download
- literature-guided optimization of line-list choices, molecular windows, and parameter ranges

## Key Artifacts

Main workflow folder:

- [README.md](/Users/jdli/Project/jorg/jorg/apogee/README.md)

Notebook:

- [apogee-korg-parameter-sweeps.ipynb](/Users/jdli/Project/jorg/jorg/apogee/notebooks/apogee-korg-parameter-sweeps.ipynb)

Generated assets:

- [apogee_water_sigma.h5](/Users/jdli/Project/jorg/jorg/apogee/output/apogee_water_sigma.h5)
- [exomol_aug_added_lines.csv](/Users/jdli/Project/jorg/jorg/apogee/output/exomol_aug_added_lines.csv)

Downloaded ExoMol assets:

- [CaH states/trans](/Users/jdli/Project/jorg/jorg/apogee/external/exomol/CaH)
- [FeH states/trans](/Users/jdli/Project/jorg/jorg/apogee/external/exomol/FeH)

Main code paths:

- [exomol.py](/Users/jdli/Project/jorg/jorg/src/jorg/apogee/exomol.py)
- [korg.py](/Users/jdli/Project/jorg/jorg/src/jorg/apogee/korg.py)
- [transformer_payne.py](/Users/jdli/Project/jorg/jorg/src/jorg/apogee/transformer_payne.py)
- [finetune.py](/Users/jdli/Project/jorg/jorg/src/jorg/apogee/finetune.py)

## ExoMol Augmentation Snapshot

Current default augmentation species:

- `CaH`
- `FeH`

Current added-line report:

- total added lines: `11441`
- `CaH`: `2528`
- `FeH`: `8913`

The canonical machine-readable record of lines added by `exomol_aug` is:

- [exomol_aug_added_lines.csv](/Users/jdli/Project/jorg/jorg/apogee/output/exomol_aug_added_lines.csv)

Each row records:

- species
- wavelength in Angstrom and cm
- `log_gf`
- `E_lower_eV`
- source `states` and `transitions` files
- wavelength bounds
- line-strength cutoff
- line-strength temperature

This CSV is refreshed automatically whenever `ensure_default_exomol_assets(...)` is called.

## Notebook Result Snapshot

The notebook has been executed successfully end-to-end.

Parameter sweeps currently shown:

- `Teff`
- `logg`
- `[M/H]`

`baseline` vs `exomol_aug` comparison currently uses two representative points:

- cool dwarf: `Teff=3400 K`, `logg=5.0`, `[M/H]=0.0`
- cool giant: `Teff=3600 K`, `logg=0.5`, `[M/H]=0.0`

Current comparison summary:

- cool dwarf: `max_abs_delta = 0.0077`, `median_abs_delta = 0.00011`
- cool giant: `max_abs_delta = 0.00278`, `median_abs_delta = 0.00003`
- strongest local difference window in both tested cases: `16245.7-16285.7 A`

Interpretation:

- the ExoMol augmentation is visible, but not large in a full-spectrum average sense for the two tested points
- the effect is localized rather than continuum-wide
- the cool dwarf shows a more noticeable response than the cool giant

## Validation Status

Most recent APOGEE-related test run:

```bash
PYTHONPATH=/Users/jdli/Project/jorg/jorg/src pytest \
  /Users/jdli/Project/jorg/jorg/tests/test_apogee_exomol.py \
  /Users/jdli/Project/jorg/jorg/tests/test_apogee_data_contracts.py \
  /Users/jdli/Project/jorg/jorg/tests/test_apogee_observations.py \
  /Users/jdli/Project/jorg/jorg/tests/test_apogee_transformer_payne.py -q
```

Result:

- `16 passed`

## Known Limitations

- APOGEE observed spectra have not been downloaded yet, so the current ExoMol judgment is still synthetic-only
- only two representative stellar points have been tested for `baseline` vs `exomol_aug`
- current ExoMol settings are still defaults:
  - wavelength range `15100-17000 A`
  - `line_strength_cutoff = -15`
  - `temperature_line_strength = 3500 K`
- no literature-guided tuning has been applied yet to:
  - molecule choice
  - isotopologue choice
  - line-strength thresholds
  - diagnostic wavelength windows
  - stellar-parameter sampling strategy for cool stars

## Recommended Next Step

The next phase should be literature-guided rather than purely implementation-driven.

Recommended order:

1. Review APOGEE cool-star and late-type literature to identify which molecular systems materially improve H-band fits for M dwarfs and M giants.
2. Check whether the literature favors additional molecules, different ExoMol releases, or different filtering thresholds beyond the current `CaH/FeH` default.
3. Identify the specific APOGEE H-band windows most sensitive to cool-star molecules, especially around the currently flagged `16245.7-16285.7 A` region.
4. Expand the synthetic comparison to colder, higher-metallicity, and higher-pressure points where molecular effects are more likely to matter.
5. Only after that, run an observed-spectrum benchmark to decide whether `exomol_aug` should become the default training linelist.

## Immediate Practical Follow-Ups

Good next tasks after the literature review:

- add a literature-notes file under `apogee/`
- build a larger ExoMol sensitivity grid over `Teff/logg/[M/H]`
- benchmark `baseline` vs `exomol_aug` against a small APOGEE cool-star validation sample
- promote the better linelist choice into the synthetic dataset builder and TransformerPayne training run
