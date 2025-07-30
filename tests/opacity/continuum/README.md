# Continuum Opacity Validation

This directory contains all files related to continuum opacity validation between Jorg and Korg.jl.

## 🚨 Validation Status: **CRITICAL ISSUE DISCOVERED**
- **Maximum opacity ratio**: 0.002 (Jorg/Korg) - **533x discrepancy!**
- **Mean opacity ratio**: 0.002 (Jorg/Korg) - **catastrophic disagreement**
- **Problem**: Jorg ~3e-12 cm⁻¹ vs Korg ~1.5e-9 cm⁻¹

## 📁 Files

### 🧪 **Test Scripts**
- `korg_continuum_opacity_script.jl` - Korg.jl reference implementation using ContinuumAbsorption module
- `test_jorg_continuum_opacity_with_statmech.py` - Jorg test script (tests 3 interfaces)

### 📊 **Data Files**
- `korg_continuum_opacity_0716.txt` - Korg.jl reference results (5200-5800 Å)
- `jorg_continuum_opacity_with_statmech.txt` - Jorg results showing major discrepancy

### 📈 **Visualization & Analysis**
- `plot_continuum_opacity_comparison.py` - Comparison plot script
- `continuum_opacity_comparison_plot.png/.pdf` - Shows 533x discrepancy clearly
- `continuum_validation_summary.py` - **Comprehensive analysis of the problem**

## 🔴 Critical Issues Identified

### **The Problem**
Jorg's continuum opacity is ~533x smaller than Korg.jl's, indicating fundamental implementation problems.

### **Possible Root Causes**
1. **Missing Physics Components**:
   - H⁻ bound-free/free-free absorption missing or incorrect
   - H I bound-free (Lyman/Balmer/Paschen) scaling wrong  
   - Metal bound-free absorption not working
   - Thomson/Rayleigh scattering missing

2. **Unit/Scaling Issues**:
   - Frequency vs wavelength unit conversion errors
   - CGS vs SI unit mismatches
   - Missing scaling factors (π, 4π factors)
   - Temperature/density scaling incorrect

3. **Chemical Equilibrium Issues**:
   - Missing key species (H⁻, H₂⁺, etc.)
   - Incorrect number densities
   - Wrong partition function usage

4. **Implementation Bugs**:
   - Cross-section calculation errors
   - Missing stimulated emission corrections
   - JAX compilation issues affecting physics

## 🛠️ Systematic Debugging Plan

### **Phase 1: Component Isolation**
Test individual continuum components:
- H⁻ bound-free only
- H⁻ free-free only  
- H I bound-free only
- Metal bound-free only
- Scattering only

### **Phase 2: Unit Verification**
Verify all unit conversions:
- Wavelength (Å) ↔ Frequency (Hz)
- Cross-sections (cm²) scaling
- Number density units (cm⁻³)

### **Phase 3: Parameter Validation**
Compare input parameters:
- Species number densities
- Partition function values  
- Physical constants

### **Phase 4: Reference Implementation**
Component-by-component Korg.jl translation:
- H⁻ absorption (Peach 1970)
- H I bound-free (exact Korg formula)
- Metal bound-free (TOPBase data)

## 🚀 Usage

```bash
# Run Korg.jl reference
julia korg_continuum_opacity_script.jl

# Run Jorg implementation (shows problem)
python test_jorg_continuum_opacity_with_statmech.py

# Create comparison plot (shows 533x discrepancy)
python plot_continuum_opacity_comparison.py

# Generate detailed analysis
python continuum_validation_summary.py
```

## ⚡ URGENT Priority

**🔴 CRITICAL**: Continuum module is NOT production-ready  
**🔴 BLOCKER**: 533x discrepancy prevents synthesis pipeline use  
**🔴 ACTION**: Must debug and fix before any production applications

## 🎯 Success Criteria for Fix

- **Target**: Achieve 0.9-1.1 ratio like line opacity validation
- **Physics**: All major continuum components working correctly  
- **Units**: Proper scaling and unit consistency
- **Validation**: Component-by-component agreement with Korg.jl

**This validation pipeline provides the framework for systematic debugging to achieve the same success as line opacity validation.**