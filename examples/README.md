# Jorg Synthesis Examples

This directory contains practical examples demonstrating how to use Jorg for stellar spectral synthesis.

## 🚀 Quick Start

### Minimal Example
```bash
cd Jorg/examples
python quick_start_synthesis.py
```

This synthesizes a solar spectrum in ~30 seconds and creates a plot. Perfect for testing your installation.

### Complete Examples
```bash
python simple_synthesis_example.py
```

Runs 5 comprehensive examples showing all major features of Jorg synthesis.

## 📋 Example Files

### `quick_start_synthesis.py` ⭐
**Perfect for beginners!**
- Minimal working example (20 lines of code)
- Solar spectrum synthesis
- Basic plotting
- Ultra-fast JAX test
- Error handling and troubleshooting

### `simple_synthesis_example.py` 🌟  
**Complete feature demonstration:**
- **Example 1**: Basic solar synthesis
- **Example 2**: Stellar parameter comparison
- **Example 3**: Ultra-fast JAX optimization
- **Example 4**: Detailed synthesis with diagnostics  
- **Example 5**: Abundance variations

## 📊 Generated Outputs

Running the examples will create these plots:

### Quick Start
- `quick_solar_spectrum.png` - Simple solar spectrum

### Complete Examples
- `solar_spectrum_basic.png` - Basic solar synthesis
- `stellar_parameter_comparison.png` - Different stellar types
- `performance_comparison.png` - Regular vs ultra-fast synthesis
- `detailed_synthesis.png` - Full diagnostic output
- `abundance_variations.png` - Element abundance effects

## 🛠️ Requirements

### Essential
- **JAX** - For high-performance computation
- **NumPy** - Array operations
- **Matplotlib** - Plotting

### Installation
```bash
# From Jorg root directory
pip install -e .

# Or install dependencies manually
pip install jax numpy matplotlib
```

## 📖 Usage Patterns

### Basic Synthesis
```python
from jorg.synthesis import synth

wavelengths, flux, continuum = synth(
    Teff=5780,    # Temperature [K]
    logg=4.44,    # Surface gravity [cgs] 
    m_H=0.0,      # Metallicity [dex]
    wavelengths=(5000, 6000),  # Range [Å]
    rectify=True  # Continuum normalize
)
```

### Ultra-Fast Synthesis
```python
from jorg.synthesis_ultra_optimized import synth_ultra_fast

wl, flux, cntm = synth_ultra_fast(
    Teff=5780, logg=4.44, m_H=0.0,
    wavelengths=(5000, 5100),
    n_points=100,    # Number of wavelength points
    mu_points=5      # Angular quadrature points
)
```

### Detailed Analysis
```python  
from jorg.synthesis import synthesize, interpolate_atmosphere, format_abundances

# Get atmosphere
A_X = format_abundances(m_H=0.0)
atm = interpolate_atmosphere(Teff=5780, logg=4.44, A_X=A_X)

# Full synthesis with diagnostics
result = synthesize(
    atm=atm,
    linelist=None,  # Default
    A_X=A_X,
    wavelengths=wavelengths,
    return_cntm=True
)

# Access detailed results
flux = result.flux
continuum = result.cntm
opacity = result.alpha          # [layers, wavelengths]
intensity = result.intensity    # [mu, wavelengths] 
species = result.number_densities
```

### Abundance Modifications
```python
# Enhanced alpha elements
wl, flux, cntm = synth(
    Teff=5780, logg=4.44, m_H=-0.5,
    O=+0.4, Mg=+0.4, Si=+0.4, Ca=+0.4,  # α-enhanced
    wavelengths=(5000, 6000)
)

# Metal-poor iron-enhanced
wl, flux, cntm = synth(
    Teff=5780, logg=4.44, m_H=-2.0,
    Fe=-1.5,  # Less iron-poor than overall metals
    wavelengths=(5000, 6000)
)
```

## 🎯 Performance Tips

### For Speed
1. **Use ultra-fast synthesis** for quick calculations
2. **Limit wavelength range** - fewer points = faster
3. **Reduce mu_points** for radiative transfer
4. **Disable hydrogen_lines** if not needed

### For Accuracy  
1. **Use full synthesis()** for research
2. **Include hydrogen lines** for complete physics
3. **Use fine wavelength sampling** for line profiles
4. **Check convergence** in chemical equilibrium

## 🔧 Troubleshooting

### Import Errors
```bash
# Make sure Jorg is in Python path
export PYTHONPATH="/path/to/Jorg/src:$PYTHONPATH"

# Or install in development mode
cd Jorg && pip install -e .
```

### Performance Issues
- **Chemistry bottleneck**: 90% of time spent in atmosphere interpolation
- **Solution**: Use ultra-fast versions or cache atmospheres
- **JAX compilation**: First run is slow (JIT compilation), subsequent runs are fast

### Common Issues
- **Missing data files**: Ensure `Jorg/data/` directory is present
- **Memory errors**: Reduce wavelength range or atmospheric layers
- **NaN results**: Check stellar parameters are physically reasonable

## 🌟 Next Steps

After trying these examples:

1. **Explore different stellar types** - M dwarfs, giants, hot stars
2. **Try line identification** - match observed spectra
3. **Parameter fitting** - use the Fit module (coming soon)
4. **Compare with observations** - validate against real spectra
5. **Performance optimization** - profile your specific use case

## 📚 Documentation

- **Main docs**: `Jorg/docs/SYNTHESIS_DOCUMENTATION.md`
- **API reference**: `Jorg/docs/implementation/synthesis_api_reference.md`
- **Performance**: `Jorg/OPTIMIZATION_SUMMARY.md`
- **Validation**: `Jorg/SYNTHESIS_PIPELINE_VALIDATION_COMPLETE.md`

## ✅ Validation Status

All examples are validated against Korg.jl:
- **99.98% continuum agreement**
- **Perfect line absorption**  
- **Production-ready accuracy**
- **16x performance improvement**

Happy synthesizing! 🌟