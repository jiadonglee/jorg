# Jorg Opacity Validation Suite

This directory contains comprehensive validation tests for Jorg's opacity calculations against Korg.jl benchmarks.

## 📁 Directory Structure

```
opacity/
├── line/           # ✅ Line opacity validation (SUCCESS)
├── continuum/      # 🔴 Continuum opacity validation (CRITICAL ISSUE)
└── README.md       # This file
```

## 🎯 Validation Overview

### ✅ **Line Opacity: VALIDATION SUCCESS**
- **Status**: Production ready
- **Agreement**: 0.89-0.99 ratios with Korg.jl
- **Achievement**: Perfect wavelength matching, exact physics
- **Files**: Complete validation pipeline in `line/`

### 🔴 **Continuum Opacity: CRITICAL ISSUE DISCOVERED**  
- **Status**: ❌ NOT production ready
- **Problem**: 533x discrepancy with Korg.jl
- **Impact**: Blocks synthesis pipeline use
- **Files**: Debugging framework in `continuum/`

## 🧪 Validation Methodology

Both validations follow the same systematic approach:

1. **🔬 Reference Implementation**: Korg.jl scripts using exact same conditions
2. **🧮 Chemical Equilibrium**: Proper species densities and electron densities  
3. **📊 Data Generation**: Opacity calculations over wavelength ranges
4. **📈 Visualization**: Detailed comparison plots and analysis
5. **📋 Documentation**: Comprehensive reports and summaries

## 🎯 Key Lessons Learned

### **Success Factors (Line Opacity)**
- ✅ Use exact Korg.jl partition functions
- ✅ Proper chemical equilibrium with full species dict  
- ✅ Dynamic species lookup (no hardcoded lists)
- ✅ Careful wavelength/frequency unit handling
- ✅ Component-by-component validation

### **Failure Modes (Continuum Opacity)**
- ❌ Missing or incorrect physics components
- ❌ Unit conversion errors  
- ❌ Scaling factor mistakes
- ❌ Incomplete chemical equilibrium species

## 🚀 Usage

### **Line Opacity Validation** ✅
```bash
cd line/
julia korg_linelist_opacity_script.jl
python test_jorg_line_opacity_with_statmech.py
python plot_detailed_comparison.py
```

### **Continuum Opacity Debugging** 🔴
```bash
cd continuum/
julia korg_continuum_opacity_script.jl  
python test_jorg_continuum_opacity_with_statmech.py
python plot_continuum_opacity_comparison.py
python continuum_validation_summary.py
```

## 📊 Results Summary

| Component | Status | Ratio (Jorg/Korg) | Notes |
|-----------|--------|-------------------|--------|
| **Line Opacity** | ✅ SUCCESS | 0.89-0.99 | Production ready |
| **Continuum Opacity** | 🔴 CRITICAL | 0.002 | 533x discrepancy |

## 🛠️ Next Steps

1. **✅ Line Opacity**: Integrate into synthesis pipeline  
2. **🔴 Continuum Opacity**: **URGENT** systematic debugging required
   - Component isolation  
   - Unit verification
   - Physics validation
   - Reference implementation

## 📋 Technical Documentation

- **Line Success**: See `line/improvements_summary.md`
- **Continuum Issues**: See `continuum/continuum_validation_summary.py`
- **Overall Status**: Updated in `/CLAUDE.md`

---

**This validation suite demonstrates Jorg's line opacity excellence while identifying critical continuum issues that must be resolved before production use.**