# How to Use Linelists with Jorg Synthesis

This guide shows how to use spectral linelists with Jorg stellar synthesis for realistic stellar spectra.

## Quick Start

### 1. Continuum-Only Synthesis (No Lines)

```python
from src.jorg.synthesis import synth

# Fastest synthesis - continuum only
wavelengths, flux, continuum = synth(
    Teff=5780,              # Temperature [K]
    logg=4.44,              # Surface gravity [cgs]
    m_H=0.0,                # Metallicity [dex]
    wavelengths=(5000, 6000), # Range in Angstroms
    linelist=None,          # No lines = continuum only
    rectify=True            # Normalize to continuum
)
```

### 2. Simple Manual Linelist

```python
from src.jorg.lines.datatypes import LineData
from src.jorg.synthesis import synth

# Create famous sodium D lines manually
na_lines = [
    # Na D2 line (5889.95 Å)
    LineData(
        wavelength=5889.95e-8,  # Wavelength in cm (Å × 1e-8)
        species=11,             # Na I (element 11, neutral)
        log_gf=-0.19,          # log(oscillator strength)
        E_lower=2.10,          # Lower level energy [eV]
        gamma_rad=6.17e8,      # Natural broadening [s⁻¹]
        gamma_stark=0.0,       # Stark broadening [s⁻¹]
        vdw_param1=7.5,        # van der Waals parameter 1
        vdw_param2=8.0         # van der Waals parameter 2
    ),
    # Na D1 line (5895.92 Å)
    LineData(
        wavelength=5895.92e-8,
        species=11,
        log_gf=-0.49,          # Weaker than D2
        E_lower=2.10,
        gamma_rad=6.14e8,
        gamma_stark=0.0,
        vdw_param1=7.5,
        vdw_param2=8.0
    )
]

# Synthesis with lines
wavelengths, flux, continuum = synth(
    Teff=5780, logg=4.44, m_H=0.0,
    wavelengths=(5880, 5900),  # Focus on Na D region
    linelist=na_lines,         # Use our manual linelist
    rectify=True,
    vmic=1.0                   # Microturbulent velocity [km/s]
)
```

### 3. Loading Real Linelists from Files

```python
from src.jorg.lines.linelist import read_linelist
from src.jorg.synthesis import synth

# Load VALD linelist
linelist = read_linelist("linelist.vald", format="vald")
print(f"Loaded {len(linelist)} lines")

# Filter to your wavelength range for speed
filtered = linelist.filter_by_wavelength(5000, 6000, unit='angstrom')
print(f"Filtered to {len(filtered)} lines in range")

# Remove very weak lines
strong_lines = filtered.prune_weak_lines(log_gf_threshold=-3.0)
print(f"Kept {len(strong_lines)} strong lines")

# Synthesis with real linelist
wavelengths, flux, continuum = synth(
    Teff=5780, logg=4.44, m_H=0.0,
    wavelengths=(5000, 6000),
    linelist=strong_lines.lines,  # Use the filtered lines
    rectify=True
)
```

## File Formats Supported

| Format | Extension | Description |
|--------|-----------|-------------|
| VALD | `.vald` | Vienna Atomic Line Database format |
| GALAH | `.h5` | GALAH survey linelist (HDF5) |
| APOGEE | `.h5` | APOGEE survey linelist (HDF5) |
| Kurucz | `.dat` | Kurucz atomic line format |
| MOOG | `.dat` | MOOG stellar synthesis format |
| TurboSpectrum | `.atoms`, `.molec` | TurboSpectrum format |

## Linelist Operations

### Loading and Filtering

```python
from src.jorg.lines.linelist import read_linelist

# Load linelist
linelist = read_linelist("mylines.vald", format="auto")  # Auto-detect format

# Filter by wavelength range
visible = linelist.filter_by_wavelength(3000, 10000, unit='angstrom')

# Filter by element/species
iron_lines = linelist.filter_by_species([26.0, 26.1])  # Fe I and Fe II

# Remove weak lines
strong = linelist.prune_weak_lines(log_gf_threshold=-2.0)

# Sort by wavelength
sorted_lines = linelist.sort_by_wavelength()

# Get statistics
from src.jorg.lines.linelist_utils import print_linelist_summary
print_linelist_summary(linelist)
```

### Species Encoding

Species are encoded as `Z.I` where:
- `Z` = atomic number (1=H, 2=He, 11=Na, 26=Fe, etc.)
- `I` = ionization stage (0=neutral, 1=singly ionized, etc.)

Examples:
- `11` = Na I (neutral sodium)
- `26` = Fe I (neutral iron)  
- `26.1` = Fe II (singly ionized iron)
- `20.1` = Ca II (singly ionized calcium)

## Advanced Usage

### Detailed Synthesis with Diagnostics

```python
from src.jorg.synthesis import synthesize, format_abundances, interpolate_atmosphere
import jax.numpy as jnp

# Setup atmosphere and abundances
A_X = format_abundances(m_H=0.0)  # Solar abundances
atm = interpolate_atmosphere(Teff=5780, logg=4.44, A_X=A_X)
wavelengths = jnp.linspace(5000, 6000, 1000)

# Detailed synthesis
result = synthesize(
    atm=atm,
    linelist=my_lines,
    A_X=A_X,
    wavelengths=wavelengths,
    vmic=1.0,
    return_cntm=True,
    mu_points=5  # Angular quadrature points
)

# Access detailed results
flux = result.flux                    # Emergent flux
continuum = result.cntm              # Continuum flux
intensity = result.intensity         # Intensity at all angles and depths
opacity = result.alpha               # Absorption coefficients [layers, wavelengths]
species_densities = result.number_densities  # Chemical abundances
```

### Comparing Different Stellar Types

```python
stellar_types = [
    {"Teff": 5780, "logg": 4.44, "m_H": 0.0},   # Solar
    {"Teff": 4500, "logg": 2.5, "m_H": 0.0},    # Cool giant
    {"Teff": 7000, "logg": 4.0, "m_H": -1.0}    # Hot metal-poor
]

for params in stellar_types:
    wl, flux, continuum = synth(
        Teff=params["Teff"],
        logg=params["logg"], 
        m_H=params["m_H"],
        wavelengths=(5000, 6000),
        linelist=my_lines,
        rectify=True
    )
    
    # Analyze line strengths
    line_depth = 1 - flux.min()
    print(f"T={params['Teff']}K: deepest line = {line_depth:.3f}")
```

## Performance Tips

### For Maximum Speed:
1. **Filter wavelength range first**: Only include lines in your spectral window
2. **Remove weak lines**: Use `prune_weak_lines(log_gf_threshold=-3.0)`
3. **Reduce angular points**: Use `mu_points=3` instead of default 20
4. **Use continuum-only**: Set `linelist=None` for pure continuum
5. **Smaller wavelength ranges**: 100-1000 Å ranges are faster than full optical

### Performance Comparison:
- **Continuum only**: ~1-2 seconds
- **With 100 lines**: ~5-10 seconds  
- **With 1000+ lines**: ~20-60 seconds
- **Full VALD linelist**: ~5-15 minutes

### Ultra-Fast Synthesis:
```python
from src.jorg.synthesis_ultra_optimized import synth_ultra_fast

# Ultra-fast synthesis with approximations
wl, flux, continuum = synth_ultra_fast(
    Teff=5780, logg=4.44, m_H=0.0,
    wavelengths=(5000, 5100),
    n_points=100,
    mu_points=3,
    n_layers=12,  # Reduced atmospheric layers
    rectify=True
)
# Typical time: ~1-2 seconds including JAX compilation
```

## Common Issues and Solutions

### Issue: "No lines in wavelength range"
**Solution**: Check your wavelength units (Å vs cm) and ensure your linelist covers your spectral range.

### Issue: Very weak or no lines
**Solution**: 
- Check your stellar parameters (hot stars have weaker metal lines)
- Verify line oscillator strengths (`log_gf`)
- Try increasing metallicity (`m_H`) for stronger lines

### Issue: Synthesis takes too long
**Solution**:
- Filter linelist to your wavelength range first
- Remove weak lines with `prune_weak_lines()`
- Use fewer `mu_points` (3-5 instead of 20)
- Use smaller wavelength ranges

### Issue: Line profiles look wrong
**Solution**:
- Check microturbulent velocity (`vmic`) - typical values 1-2 km/s
- Verify broadening parameters in your linelist
- Ensure correct stellar parameters for your star type

## Data Sources

### Where to get linelists:
- **VALD**: [http://vald.astro.uu.se/](http://vald.astro.uu.se/) - Vienna Atomic Line Database
- **NIST**: [https://www.nist.gov/pml/atomic-spectra-database](https://www.nist.gov/pml/atomic-spectra-database) - Atomic spectra
- **Kurucz**: [http://kurucz.harvard.edu/linelists.html](http://kurucz.harvard.edu/linelists.html) - Robert Kurucz's linelists
- **ExoMol**: [https://www.exomol.com/](https://www.exomol.com/) - Molecular linelists
- **Survey linelists**: GALAH, APOGEE, Gaia-ESO have curated linelists

### Included with Korg.jl/Jorg:
- Solar linelist extract from VALD
- GALAH DR3 linelist (if available)
- Selected molecular linelists

## Summary

**Basic workflow:**
1. Load or create linelist
2. Filter to your wavelength range and remove weak lines
3. Run synthesis with `synth()` or `synthesize()`
4. Analyze results

**Key parameters:**
- `Teff`, `logg`, `m_H`: Stellar parameters
- `vmic`: Microturbulent velocity (1-2 km/s typical)
- `wavelengths`: Spectral range in Angstroms
- `rectify=True`: For continuum-normalized flux
- `mu_points`: Angular resolution (3-5 for speed, 20 for accuracy)

**Performance vs Accuracy tradeoff:**
- Fast: Continuum-only, few lines, small wavelength range
- Accurate: Full linelist, many angular points, detailed chemistry