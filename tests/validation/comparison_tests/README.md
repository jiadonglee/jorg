# Korg vs Jorg Comparison Tests

This directory contains all the debugging and comparison scripts used to achieve sub-1% accuracy between Korg.jl and Jorg implementations.

## Final Results
- **Mean Difference**: 0.072% (Target: <1% ✅)
- **Max Difference**: 0.176% (Target: <1% ✅)
- **Correlation**: 0.999997 (Target: >0.999 ✅)

## Debug Scripts (Julia)
- `debug_partition_functions.jl` - Verified partition function accuracy
- `detailed_component_debug.jl` - Attempted component-wise debugging
- `final_precision_debug.jl` - Generated final Korg reference data

## Debug Scripts (Python)
- `component_wise_comparison.py` - Comprehensive component analysis
- `debug_h_i_bf.py` - H I bound-free debugging
- `debug_h_i_detailed.py` - Detailed H I bound-free analysis
- `debug_h_minus_ff.py` - H^- free-free debugging (critical fix)
- `debug_hminus_interpolation.py` - H^- cross section interpolation
- `test_h_i_range.py` - H I quantum level contribution testing
- `test_improved_jorg.py` - Testing improved Jorg implementation
- `create_mclaughlin_data.py` - Created McLaughlin H^- data file
- `create_final_comparison_plots.py` - Generated final visualization

## Key Fixes Applied
1. **McLaughlin H^- Data**: Implemented exact interpolation (5.58% → 4.09%)
2. **H^- Free-Free Correction**: Applied Bell & Berrington factor (4.09% → 0.072%)

## Usage
Run comparison scripts from the Korg.jl root directory to ensure proper module imports.

## Results
See `../test_fig/` for all comparison plots and performance statistics.