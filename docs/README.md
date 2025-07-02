# Jorg Documentation

This directory contains comprehensive documentation for the Jorg stellar spectral synthesis package.

## Directory Structure

### 📊 `comparisons/`
Direct comparisons between Jorg and Korg.jl implementations:
- `JORG_KORG_COMPREHENSIVE_COMPARISON.md` - Complete feature and accuracy comparison
- `JORG_KORG_RT_CODE_COMPARISON.md` - Radiative transfer code comparison with outputs
- `JORG_KORG_SYNTHESIS_COMPARISON.md` - Full synthesis pipeline comparison
- `KORG_COMPATIBILITY_REPORT.md` - Compatibility assessment and migration guide

### 🔬 `validation/`
Validation reports demonstrating scientific accuracy:
- `RADIATIVE_TRANSFER_FINAL_VALIDATION.md` - RT algorithm validation
- `JORG_KORG_LINE_PROFILE_VALIDATION_SUMMARY.md` - Line profile accuracy validation
- `RADIATIVE_TRANSFER_COMPARISON_SUMMARY.md` - RT comparison summary
- `hydrogen_validation_summary.md` - Hydrogen line implementation validation

### 🛠️ `implementation/`
Technical implementation details:
- `HYDROGEN_LINES_IMPLEMENTATION.md` - Hydrogen line physics implementation
- `METAL_BOUND_FREE_IMPLEMENTATION.md` - Metal bound-free absorption implementation

### 📚 `tutorials/`
User guides and tutorials:
- `statmech_tutorial.md` - Statistical mechanics usage guide

### 📋 `source/`
Legacy documentation (being reorganized):
- Architecture documents
- Project structure guides
- Implementation summaries

## Quick Navigation

**New Users**: Start with the comparisons to understand Jorg vs Korg.jl differences
**Developers**: Check implementation docs for technical details
**Validators**: Review validation reports for accuracy verification
**Scientists**: Use tutorials for practical applications

## Status

All documentation reflects the current state where Jorg achieves **machine precision agreement** with Korg.jl across all major physics components while providing additional performance and ML capabilities.