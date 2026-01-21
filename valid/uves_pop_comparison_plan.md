# Plan: Compare Jorg Synthetic Spectra with UVES-POP Observations

## Overview
Compare Jorg's synthetic stellar spectra with observed UVES-POP spectra for 10 representative stars across the stellar parameter space (Teff, logg, [Fe/H], [α/Fe]).

## Selected Stars (10 across parameter space)

| # | Regime | Name | HD | Teff | logg | [Fe/H] | [α/Fe] |
|---|--------|------|-----|------|------|--------|--------|
| 1 | Cool M-dwarf | HD 16620 | - | 3728 | 4.95 | 0.04 | 0.10 |
| 2 | Cool K-giant | * alf Cet | 18884 | 4180 | 1.85 | 0.52 | 0.68 |
| 3 | Solar analog (G2V) | HD 59468 | 59468 | 5787 | 4.58 | -0.05 | -0.02 |
| 4 | F-type dwarf | * q01 Eri | 10647 | 6208 | 4.46 | -0.17 | 0.02 |
| 5 | A-type dwarf | * alf Gem | 60178J | 9031 | 3.94 | 0.19 | -0.22 |
| 6 | Subgiant | * bet Hyi | 2151 | 5775 | 3.67 | -0.36 | 0.12 |
| 7 | Alpha-enhanced giant | * alf Boo | 124897 | 4765 | 2.47 | -0.43 | 0.58 |
| 8 | Very metal-poor | HD 84937 | 84937 | 6390 | 3.58 | -2.07 | 0.31 |
| 9 | Hot B-type | * alf Eri | 10144 | 14680 | 3.08 | 0.18 | 0.00 |
| 10 | Metal-rich star | HD 148379 | 148379 | 18859 | 2.01 | 0.50 | 0.00 |

**Parameter coverage**: Teff: 3728-18859 K, logg: 1.85-4.95, [Fe/H]: -2.07 to +0.52, [α/Fe]: -0.22 to +0.68

---

## Implementation Steps

### Step 1: Create star selection module
**File**: `jorg/src/jorg/uves_pop_stars.py`

- Store the 10 selected stars with their parameters
- Include URLs for downloading spectra
- Helper function to load star parameters from catalog

### Step 2: Download and read observed spectra
**File**: `jorg/src/jorg/data/uves_pop_spectra.py` (new module)

1. Download FITS spectra from UVES-POP:
   - Base URL: `https://data.voxastro.org/uves-pop/model_spec/fit_res/v221115/`
   - Format: `{star_name}.fits.gz`
   - Use `astropy.io.fits` to read

2. Extract data from FITS:
   - Wavelength array (verify units: likely Å)
   - Flux array
   - Check for air vs vacuum wavelengths
   - Apply radial velocity correction if needed

### Step 3: Compute synthetic spectra
**Use existing Jorg API** (`jorg/src/jorg/synthesis.py`):

```python
from jorg.synthesis import synth
from jorg.lines.linelist_data import get_VALD_solar_linelist

# For each star:
# Convert [Fe/H] and [α/Fe] to m_H and alpha_H
m_H = star["Fe_H"]           # [Fe/H] ≈ [M/H]
alpha_H = star["Fe_H"] + star["a_Fe"]  # [α/H] = [Fe/H] + [α/Fe]

wl_syn, flux_syn, cntm_syn = synth(
    Teff=star["Teff"],
    logg=star["logg"],
    m_H=m_H,
    alpha_H=alpha_H,
    wavelengths=(4800, 6800),  # UVES optical range
    linelist=get_VALD_solar_linelist(),
    rectify=True,
    R=20000,  # Match UVES R=20k observed spectra
    vmic=1.0
)
```

### Step 4: Match wavelength grids and compare
1. Interpolate synthetic spectrum to observed wavelengths using `scipy.interpolate.interp1d`
2. Apply same resolution convolution (R=20,000) if not already done
3. Compute residuals: `flux_obs - flux_syn`
4. Calculate metrics: RMS, χ², line depth comparisons

### Step 5: Create visualization module
**File**: `jorg/examples/compare_uves_pop.ipynb`

- Plot observed vs synthetic spectra for each star
- Overlay residuals
- Create summary plots across parameter space
- Highlight key spectral regions (H-alpha, Mg triplet, etc.)

---

## Critical Files

| File | Purpose |
|------|---------|
| `jorg/src/jorg/synthesis.py` | Main `synth()` API |
| `jorg/src/jorg/abundances.py` | `format_abundances()` for alpha-enhancement |
| `jorg/src/jorg/atmosphere.py` | `interpolate_marcs()` for model atmosphere |
| `jorg/data/J_ApJS_266_11_table6.dat.fits` | UVES-POP catalog |
| `jorg/src/jorg/lines/linelist_data.py` | `get_VALD_solar_linelist()` |

---

## Alpha-Enhancement Handling

The UVES-POP catalog provides `[Fe/H]` and `[α/Fe]`. Convert to Jorg's format:

```python
# [α/Fe] = +0.3 means alpha elements are 0.3 dex enhanced relative to Fe
# For [Fe/H] = -1.0, [α/Fe] = +0.3:
#   m_H = -1.0 (non-alpha metals)
#   alpha_H = -1.0 + 0.3 = -0.7 (alpha elements: O, Ne, Mg, Si, S, Ar, Ca, Ti)

from jorg.abundances import format_abundances
A_X = format_abundances(
    default_metals_H=feh,      # Non-alpha metals
    default_alpha_H=feh + afeh  # Alpha elements
)
```

---

## Verification Plan

1. **Download verification**: Verify at least 3 spectra download successfully
2. **Synthesis verification**: Run `synth()` for a solar-type star (HD 59468) and verify reasonable flux values
3. **Wavelength matching**: Verify observed and synthetic wavelength ranges overlap
4. **Visual inspection**: Plot observed vs synthetic for 1-2 stars before full run
5. **Metrics calculation**: Compute RMS/χ² for at least 3 stars

**Expected outcome**: Synthetic spectra should match observed spectra within ~5-10% for continuum-normalized regions. Line depths may vary due to:
- Non-LTE effects (not included in Jorg)
- 3D hydrodynamical atmosphere effects (MARCS is 1D)
- Missing line list entries
- Instrumental effects in observed spectra

---

## Potential Issues & Mitigations

| Issue | Mitigation |
|-------|------------|
| FITS format unknown | Inspect one file first with `astropy.io.fits.info()` |
| Wavelength units mismatch | Check header for `WAVEUNIT` keyword |
| Air vs vacuum wavelengths | Apply conversion using existing utilities |
| Radial velocity shift | Cross-correlate to measure and correct |
| Continuum normalization differences | Normalize both to pseudo-continuum |
| MARCS grid limits | Check `interpolate_marcs` bounds before synthesis |
| Missing abundances for hot stars | Use default scaling for Teff > 10000 K |
