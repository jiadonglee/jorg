"""
Performance Benchmark Script for JAX Optimizations

This script benchmarks the performance improvements from the JAX optimizations:
1. JIT compilation caching
2. Alpha5 reference optimization with CE reuse
3. Memory chunking for line window calculation
4. Layer data caching

Usage:
    python -m jorg.benchmark_performance [--quick] [--detailed]
"""

import numpy as np
import time
import tracemalloc
from pathlib import Path
from typing import Dict, Tuple, List
import argparse

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from jorg.opacity.korg_line_processor import KorgLineProcessor, clear_jit_cache
from jorg.alpha5_reference import calculate_alpha5_reference
from jorg.statmech import create_default_partition_functions, create_default_ionization_energies, create_default_log_equilibrium_constants
from jorg.lines.linelist import read_linelist
from jorg.atmosphere import import_marcs_atmosphere
from jorg.synthesis import synthesize


class PerformanceBenchmark:
    """Benchmark suite for JAX performance optimizations."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}

    def print(self, *args, **kwargs):
        """Print only if verbose mode is on."""
        if self.verbose:
            print(*args, **kwargs)

    def benchmark_jit_caching(self, n_iterations: int = 5) -> Dict:
        """
        Benchmark JIT compilation caching.

        Tests the improvement from caching JIT-compiled functions
        instead of recompiling on every call.
        """
        self.print("\n" + "="*80)
        self.print("BENCHMARK 1: JIT Compilation Caching")
        self.print("="*80)

        # Load small linelist for testing
        base_dir = Path(__file__).parent
        linelist_path = base_dir / "data" / "linelists" / "vald_5000_5100.csv"

        if not linelist_path.exists():
            self.print("   ⚠️  Test linelist not found, skipping JIT benchmark")
            return {}

        linelist = read_linelist(linelist_path, format="vald")
        self.print(f"   Loaded {len(linelist.lines)} lines for testing")

        # Create sample atmospheric data
        n_layers = 10
        n_wavelengths = 100
        temps = np.linspace(4000, 7000, n_layers)
        electron_densities = np.linspace(1e13, 1e15, n_layers)
        wl_array_cm = np.linspace(5000e-8, 5100e-8, n_wavelengths)

        # Create mock number densities
        from jorg.statmech.species import Species
        h_neutral = Species.from_atomic_number(1, 0)
        n_densities = {h_neutral: np.linspace(1e16, 1e17, n_layers)}

        partition_funcs = create_default_partition_functions()
        n_div_U = {}
        for species in n_densities.keys():
            if species in partition_funcs:
                log_temps = np.log(temps)
                U_values = np.array([partition_funcs[species](log_T) for log_T in log_temps])
                n_div_U[species] = n_densities[species] / np.maximum(U_values, 1e-50)

        # Clear cache before testing
        clear_jit_cache()

        processor = KorgLineProcessor(verbose=False)

        # First call (compilation)
        self.print("\n   First call (with JIT compilation)...")
        start = time.time()
        result1 = processor.process_lines(
            wl_array_cm=wl_array_cm,
            temps=temps,
            electron_densities=electron_densities,
            n_densities=n_densities,
            partition_fns=partition_funcs,
            linelist=linelist.lines[:100],  # Small subset for speed
            microturbulence_cm_s=1e5,
            cutoff_threshold=3e-4
        )
        first_call_time = time.time() - start
        self.print(f"   Time: {first_call_time:.3f}s")

        # Subsequent calls (cached)
        self.print("\n   Subsequent calls (from cache)...")
        cached_times = []
        for i in range(n_iterations):
            start = time.time()
            result = processor.process_lines(
                wl_array_cm=wl_array_cm,
                temps=temps,
                electron_densities=electron_densities,
                n_densities=n_densities,
                partition_fns=partition_funcs,
                linelist=linelist.lines[:100],
                microturbulence_cm_s=1e5,
                cutoff_threshold=3e-4
            )
            cached_time = time.time() - start
            cached_times.append(cached_time)

            if i < 3 or i == n_iterations - 1:
                self.print(f"   Call {i+1}: {cached_time:.3f}s")

        avg_cached_time = np.mean(cached_times)
        speedup = first_call_time / avg_cached_time

        self.print(f"\n   📊 Results:")
        self.print(f"      First call (compile): {first_call_time:.3f}s")
        self.print(f"      Avg cached call: {avg_cached_time:.3f}s")
        self.print(f"      Speedup: {speedup:.1f}x")

        return {
            'first_call_time': first_call_time,
            'avg_cached_time': avg_cached_time,
            'speedup': speedup
        }

    def benchmark_memory_chunking(self) -> Dict:
        """
        Benchmark memory chunking for window calculation.

        Tests the memory reduction from processing lines in chunks
        instead of all at once.
        """
        self.print("\n" + "="*80)
        self.print("BENCHMARK 2: Memory Chunking")
        self.print("="*80)

        # Generate synthetic line data
        n_lines = 5000
        n_layers = 56
        n_wavelengths = 1000

        self.print(f"\n   Generating synthetic data:")
        self.print(f"      Lines: {n_lines}")
        self.print(f"      Layers: {n_layers}")
        self.print(f"      Wavelengths: {n_wavelengths}")

        # Create synthetic line arrays
        line_arrays = {
            "wavelength": np.random.uniform(5000e-8, 5100e-8, n_lines),
            "log_gf": np.random.uniform(-2, 0, n_lines),
            "E_lower": np.random.uniform(0, 10, n_lines),
            "gamma_rad": np.random.uniform(1e7, 1e8, n_lines),
            "gamma_stark": np.random.uniform(0, 1e6, n_lines),
            "vdw_sigma": np.random.uniform(0, 100, n_lines),
            "vdw_alpha": np.random.uniform(-2, 1, n_lines),
            "vdw_base_gamma": np.ones(n_lines),
            "species_idx": np.zeros(n_lines, dtype=np.int32),
            "atomic_mass": np.full(n_lines, 55.845 * 1.66e-24),  # Iron mass
            "is_molecule": np.zeros(n_lines, dtype=bool)
        }

        temps = np.linspace(4000, 7000, n_layers)
        electron_densities = np.linspace(1e13, 1e15, n_layers)
        n_h_neutral = np.linspace(1e16, 1e17, n_layers)
        wl_array_cm = np.linspace(5000e-8, 5100e-8, n_wavelengths)
        n_div_U_array = np.random.uniform(1e14, 1e16, (10, n_layers))

        processor = KorgLineProcessor(verbose=False)

        # Test without chunking
        self.print("\n   Testing without chunking...")
        tracemalloc.start()
        start = time.time()

        try:
            lb1, ub1, max_pts1, windowed1, amp1 = processor._compute_line_windows_single_chunk(
                wl_array_cm=wl_array_cm,
                temps=temps,
                electron_densities=electron_densities,
                n_div_U_array=n_div_U_array,
                line_arrays=line_arrays,
                microturbulence_cm_s=1e5,
                continuum_opacity=None,
                cutoff_threshold=3e-4,
                n_h_neutral=n_h_neutral,
                float_dtype=np.float64
            )
            time_no_chunk = time.time() - start
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            self.print(f"      Time: {time_no_chunk:.3f}s")
            self.print(f"      Peak memory: {peak / 1024**2:.1f} MB")
        except MemoryError:
            self.print(f"      ❌ MemoryError! Cannot process without chunking")
            tracemalloc.stop()
            time_no_chunk = None
            peak = None

        # Test with chunking
        self.print("\n   Testing with chunking (chunk_size=1000)...")
        tracemalloc.start()
        start = time.time()

        lb2, ub2, max_pts2, windowed2, amp2 = processor._compute_line_windows_numpy(
            wl_array_cm=wl_array_cm,
            temps=temps,
            electron_densities=electron_densities,
            n_div_U_array=n_div_U_array,
            line_arrays=line_arrays,
            microturbulence_cm_s=1e5,
            continuum_opacity=None,
            cutoff_threshold=3e-4,
            n_h_neutral=n_h_neutral,
            float_dtype=np.float64,
            chunk_size=1000
        )

        time_chunked = time.time() - start
        current, peak_chunked = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.print(f"      Time: {time_chunked:.3f}s")
        self.print(f"      Peak memory: {peak_chunked / 1024**2:.1f} MB")

        # Calculate improvements
        if peak is not None:
            memory_reduction = (peak - peak_chunked) / peak * 100
            self.print(f"\n   📊 Results:")
            self.print(f"      Memory reduction: {memory_reduction:.1f}%")
            if time_no_chunk is not None:
                time_overhead = (time_chunked - time_no_chunk) / time_no_chunk * 100
                self.print(f"      Time overhead: {time_overhead:.1f}%")
            else:
                self.print(f"      ✅ Chunking enabled processing that would OOM")

        return {
            'peak_memory_no_chunk': peak,
            'peak_memory_chunked': peak_chunked,
            'time_no_chunk': time_no_chunk,
            'time_chunked': time_chunked
        }

    def benchmark_alpha5_optimization(self) -> Dict:
        """
        Benchmark alpha5 reference optimization with CE reuse.

        Tests the speedup from reusing chemical equilibrium results
        instead of recalculating.
        """
        self.print("\n" + "="*80)
        self.print("BENCHMARK 3: Alpha5 Reference Optimization")
        self.print("="*80)

        # Load atmosphere
        base_dir = Path(__file__).parent
        atm_path = base_dir / "data" / "atmospheres" / "sun_marcs.mod"

        if not atm_path.exists():
            self.print("   ⚠️  Test atmosphere not found, skipping alpha5 benchmark")
            return {}

        self.print("\n   Loading MARCS atmosphere...")
        atm = import_marcs_atmosphere(atm_path)

        # Solar abundances
        A_X = np.array([
            12.00, 10.93, 7.53, 8.69, 4.56, 7.93, 8.51, 6.33, 7.90, 6.30,
            5.05, 6.45, 5.50, 4.00, 5.02, 5.67, 5.39, 6.31, 3.15, 4.95
        ])

        # Test without CE reuse
        self.print("\n   Testing alpha5 WITHOUT CE reuse...")
        start = time.time()
        alpha5_no_reuse = calculate_alpha5_reference(
            atm=atm,
            A_X=A_X,
            linelist=None,
            verbose=False
        )
        time_no_reuse = time.time() - start
        self.print(f"      Time: {time_no_reuse:.3f}s")

        # Prepare CE results for reuse (simulate coming from synthesis)
        from jorg.statmech.korg_chemical_equilibrium import chemical_equilibrium

        self.print("\n   Precomputing chemical equilibrium for reuse...")
        temperatures = np.array(atm['temperature'])
        number_density_layers = np.array(atm['number_density'])
        electron_density_guess = np.array(atm['electron_density'])

        abs_abundances = 10 ** (A_X - 12)
        abs_abundances = abs_abundances / np.sum(abs_abundances)
        abs_abundances = {Z: float(abs_abundances[Z - 1]) for Z in range(1, 93)}

        ionization_energies = create_default_ionization_energies()
        log_equilibrium_constants = create_default_log_equilibrium_constants()
        partition_funcs = create_default_partition_functions()

        all_electron_densities = np.zeros(len(temperatures))
        all_number_densities = {}

        for i in range(len(temperatures)):
            ne, n_dict = chemical_equilibrium(
                temp=float(temperatures[i]),
                nt=float(number_density_layers[i]),
                model_atm_ne=float(electron_density_guess[i]),
                absolute_abundances=abs_abundances,
                ionization_energies=ionization_energies,
                partition_funcs=partition_funcs,
                log_equilibrium_constants=log_equilibrium_constants
            )
            all_electron_densities[i] = ne
            for spec, dens in n_dict.items():
                if spec not in all_number_densities:
                    all_number_densities[spec] = np.zeros(len(temperatures))
                all_number_densities[spec][i] = dens

        ce_results = {
            'electron_densities': all_electron_densities,
            'number_densities': all_number_densities
        }

        # Test with CE reuse
        self.print("\n   Testing alpha5 WITH CE reuse...")
        start = time.time()
        alpha5_reuse = calculate_alpha5_reference(
            atm=atm,
            A_X=A_X,
            linelist=None,
            use_chemical_equilibrium_from=ce_results,
            verbose=False
        )
        time_reuse = time.time() - start
        self.print(f"      Time: {time_reuse:.3f}s")

        # Verify results match
        max_diff = np.max(np.abs(alpha5_no_reuse - alpha5_reuse))
        self.print(f"\n   Max difference: {max_diff:.2e}")

        speedup = time_no_reuse / time_reuse
        self.print(f"\n   📊 Results:")
        self.print(f"      Without CE reuse: {time_no_reuse:.3f}s")
        self.print(f"      With CE reuse: {time_reuse:.3f}s")
        self.print(f"      Speedup: {speedup:.1f}x")

        return {
            'time_no_reuse': time_no_reuse,
            'time_reuse': time_reuse,
            'speedup': speedup,
            'max_diff': max_diff
        }

    def run_all_benchmarks(self, quick: bool = False) -> Dict:
        """Run all benchmarks."""
        self.print("\n" + "="*80)
        self.print("JAX PERFORMANCE OPTIMIZATION BENCHMARK SUITE")
        self.print("="*80)

        results = {}

        # Benchmark 1: JIT caching
        try:
            results['jit_caching'] = self.benchmark_jit_caching()
        except Exception as e:
            self.print(f"\n   ❌ JIT benchmark failed: {e}")

        # Benchmark 2: Memory chunking
        try:
            results['memory_chunking'] = self.benchmark_memory_chunking()
        except Exception as e:
            self.print(f"\n   ❌ Memory chunking benchmark failed: {e}")

        # Benchmark 3: Alpha5 optimization (skip in quick mode)
        if not quick:
            try:
                results['alpha5_optimization'] = self.benchmark_alpha5_optimization()
            except Exception as e:
                self.print(f"\n   ❌ Alpha5 benchmark failed: {e}")

        return results

    def print_summary(self, results: Dict):
        """Print benchmark summary."""
        self.print("\n" + "="*80)
        self.print("BENCHMARK SUMMARY")
        self.print("="*80)

        if 'jit_caching' in results and results['jit_caching']:
            r = results['jit_caching']
            self.print(f"\n1. JIT Compilation Caching:")
            self.print(f"   Speedup: {r['speedup']:.1f}x")
            self.print(f"   Compilation time saved: {r['first_call_time'] - r['avg_cached_time']:.3f}s per call")

        if 'memory_chunking' in results and results['memory_chunking']:
            r = results['memory_chunking']
            if r['peak_memory_no_chunk'] is not None:
                reduction = (r['peak_memory_no_chunk'] - r['peak_memory_chunked']) / r['peak_memory_no_chunk'] * 100
                self.print(f"\n2. Memory Chunking:")
                self.print(f"   Memory reduction: {reduction:.1f}%")
                if r['time_no_chunk'] is not None:
                    overhead = (r['time_chunked'] - r['time_no_chunk']) / r['time_no_chunk'] * 100
                    self.print(f"   Time overhead: {overhead:.1f}%")

        if 'alpha5_optimization' in results and results['alpha5_optimization']:
            r = results['alpha5_optimization']
            self.print(f"\n3. Alpha5 Reference Optimization:")
            self.print(f"   Speedup: {r['speedup']:.1f}x")
            self.print(f"   Time saved: {r['time_no_reuse'] - r['time_reuse']:.3f}s")
            self.print(f"   Accuracy: max diff = {r['max_diff']:.2e}")

        self.print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='JAX Performance Optimization Benchmarks')
    parser.add_argument('--quick', action='store_true', help='Run quick benchmarks (skip alpha5)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    args = parser.parse_args()

    benchmark = PerformanceBenchmark(verbose=not args.quiet)
    results = benchmark.run_all_benchmarks(quick=args.quick)
    benchmark.print_summary(results)


if __name__ == '__main__':
    main()
