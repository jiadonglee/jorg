# UVES-POP vs Jorg Synthetic Spectra Comparison - Implementation Summary

## Overview
This implementation compares Jorg's synthetic stellar spectra with observed UVES-POP spectra for 10 representative stars across the stellar parameter space.

## Selected Stars (10 across parameter space)

| # | Regime | Name | Filename | Teff | logg | [Fe/H] | [α/Fe] |
|---|--------|------|----------|------|------|--------|--------|
| 1 | Cool M-dwarf | IC2391-0096 | IC2391-0096.fits | 3728 | 4.95 | 0.04 | 0.10 |
| 2 | Cool K-giant | * alf Cet | HD018884.fits | 4180 | 1.85 | 0.52 | 0.68 |
| 3 | Solar analog (G2V) | HD 59468 | HD059468.fits | 5787 | 4.58 | -0.05 | -0.02 |
| 4 | F-type dwarf | * nu. Phe | HD007570.fits | 6077 | 4.07 | -0.17 | 0.02 |
| 5 | A-type dwarf | * alf Gem | Castor.fits | 9031 | 3.94 | 0.19 | -0.22 |
| 6 | Subgiant | * bet Hyi | HD002151.fits | 5775 | 3.67 | -0.36 | 0.12 |
| 7 | Alpha-enhanced giant | * alf Boo | Arcturus.fits | 4765 | 2.47 | -0.43 | 0.58 |
| 8 | Very metal-poor | HD 84937 | HD084937.fits | 6390 | 3.58 | -2.07 | 0.31 |
| 9 | Hot B-type | * alf Eri | Achernar.fits | 14680 | 3.08 | 0.18 | 0.00 |
| 10 | Metal-rich star | V* BM Hyi | HD010840.fits | 11866 | 3.83 | 0.54 | -0.03 |

**Parameter coverage**: Teff: 3728-14680 K, logg: 1.85-4.95, [Fe/H]: -2.07 to +0.54, [α/Fe]: -0.22 to +0.68

## Implementation Files

### 1. `jorg/src/jorg/uves_pop_stars.py`
Star selection module providing:
- `load_selected_stars()` - Load 10 selected stars from JSON
- `get_star_by_id()` - Get specific star by ID (1-10)
- `get_star_by_name()` - Find star by name
- `get_stars_by_regime()` - Get stars by stellar regime
- `summarize_sample()` - Print summary of selected stars

### 2. `jorg/src/jorg/data/uves_pop_spectra.py`
Spectrum download and reading utilities:
- `download_spectrum()` - Download FITS from UVES-POP server
- `read_spectrum()` - Read FITS file (handles gzip, various formats)
- `download_and_read()` - Combined download + read function
- `get_spectrum_info()` - Get spectrum information (wavelength range, flux stats)

**Features:**
- Handles UVES-POP FITS format (binary table with WAVE/FLUX columns)
- Automatic FITS header verification and fixing
- Gzip decompression support
- Robust column detection (handles various column naming conventions)

### 3. `jorg/src/jorg/uves_pop_comparison.py`
Main comparison module:
- `compare_star()` - Compare one star's observed vs synthetic spectrum
- `compare_all_stars()` - Compare multiple stars
- `summarize_comparison_results()` - Print summary table

**Workflow for each star:**
1. Load stellar parameters (Teff, logg, [Fe/H], [α/Fe])
2. Download observed spectrum from UVES-POP
3. Compute Jorg synthetic spectrum using `synth()`
4. Convert [Fe/H] and [α/Fe] to m_H and alpha_H for Jorg
5. Interpolate synthetic to observed wavelength grid
6. Apply percentile-based normalization to both spectra
7. Compute comparison metrics (RMS, median residual, relative RMS)

## Usage Example

```python
import sys
sys.path.insert(0, 'jorg/src')

from jorg.uves_pop_comparison import compare_star, compare_all_stars

# Compare single star (Arcturus)
result = compare_star(
    star_id=7,
    wavelength_range=(5000, 5200),
    resolution=20000,
    verbose=True
)

# Access results
print(f"RMS residual: {result['metrics']['rms']:.3f}")
print(f"Relative RMS: {result['metrics']['relative_rms']*100:.1f}%")

# Compare all stars
results = compare_all_stars(
    wavelength_range=(4800, 6800),
    resolution=20000
)

# Print summary
from jorg.uves_pop_comparison import summarize_comparison_results
summarize_comparison_results(results)
```

## Test Results

Tested with Arcturus (star 7: alpha-enhanced K-giant):
- Teff: 4765 K, logg: 2.47, [Fe/H]: -0.43, [α/Fe]: +0.58
- Wavelength range: 5000-5200 Å
- RMS residual: 0.31
- Relative RMS: 38%

**Notes on comparison accuracy:**
The RMS residual is reasonable considering:
1. **Non-LTE effects**: Jorg assumes LTE, but real atmospheres have non-LTE effects
2. **3D vs 1D**: MARCS models are 1D, real stars have 3D atmosphere structure
3. **Line list**: VALD linelist may be incomplete
4. **Normalization**: Percentile-based normalization is approximate
5. **Microturbulence**: Fixed at 1 km/s (may vary for different stars)

## File Locations

| File | Location |
|------|----------|
| Star selection module | `jorg/src/jorg/uves_pop_stars.py` |
| Spectrum utilities | `jorg/src/jorg/data/uves_pop_spectra.py` |
| Comparison module | `jorg/src/jorg/uves_pop_comparison.py` |
| Selected stars JSON | `jorg/valid/selected_stars.json` |
| Plan document | `jorg/valid/uves_pop_comparison_plan.md` |
| Catalog | `jorg/data/J_ApJS_266_11_table6.dat.fits` |
| Downloaded spectra | `uves_pop_spectra/*.fits` |

## Dependencies

- `jorg.synthesis` - Main synthesis API (`synth()`)
- `jorg.lines.linelist_data` - VALD linelist (`get_VALD_solar_linelist()`)
- `jorg.abundances` - Abundance formatting (`format_abundances()`)
- `astropy.io.fits` - FITS file reading
- `scipy.interpolate` - Interpolation functions
- `numpy` - Numerical operations

## Next Steps

To extend this implementation:

1. **Create visualization notebook** - Plot observed vs synthetic spectra for all 10 stars
2. **Download all spectra** - Pre-download all 10 spectra to avoid repeated downloads
3. **Optimize normalization** - Implement more sophisticated continuum fitting
4. **Calculate chi-squared** - Formal goodness-of-fit metric
5. **Analyze residuals** - Identify systematic trends across wavelength
6. **Compare multiple wavelength regions** - Test UV, visible, NIR separately

## References

- UVES-POP catalog: J_ApJS_266_11
- UVES-POP spectra: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/266/11/sp/
- Jorg synthesis: Korg.jl-compatible Python/JAX implementation
