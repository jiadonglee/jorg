#!/usr/bin/env python3
"""
Performance benchmark: Korg.jl vs Jorg linelist reading speed

This script compares the performance of reading linelists between
Korg.jl (Julia) and Jorg (Python/JAX) implementations.
"""

import numpy as np
import time
import tempfile
import sys
from pathlib import Path
import subprocess
import json

# Add jorg to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    
    from jorg.lines import read_linelist, save_linelist, LineList
    JAX_AVAILABLE = True
    print("✅ JAX and Jorg.lines successfully imported")
except ImportError as e:
    print(f"❌ JAX import error: {e}")
    JAX_AVAILABLE = False


def create_large_vald_linelist(n_lines: int = 10000) -> str:
    """Create a large VALD format linelist for benchmarking"""
    
    # Template lines with different species
    templates = [
        # Na I lines
        ("5889.9510", "0.108", "0.000", "Na 1", "6.14e7", "2.80e-5", "1.40e-7", "0.30"),
        ("5895.9242", "-0.194", "0.000", "Na 1", "6.14e7", "2.80e-5", "1.40e-7", "0.30"),
        
        # Ca I lines  
        ("5801.3310", "-5.234", "2.710", "Ca 1", "3.00e6", "1.20e-6", "8.50e-8", "0.25"),
        ("5857.4510", "-1.989", "2.933", "Ca 1", "3.00e6", "1.20e-6", "8.50e-8", "0.25"),
        
        # Fe I lines
        ("5857.4760", "-2.158", "4.283", "Fe 1", "2.50e6", "8.40e-7", "1.30e-7", "0.22"),
        ("5862.3530", "-0.058", "4.549", "Fe 1", "2.50e6", "8.40e-7", "1.30e-7", "0.22"),
        
        # Mg I lines
        ("5167.3210", "-0.890", "4.912", "Mg 1", "1.80e6", "5.60e-7", "9.20e-8", "0.18"),
        ("5172.6840", "-0.402", "2.712", "Mg 1", "1.80e6", "5.60e-7", "9.20e-8", "0.18"),
        
        # H I lines
        ("6562.8010", "0.640", "10.199", "H 1", "6.14e7", "2.80e-5", "1.40e-7", "0.30"),
        ("4861.3230", "-0.020", "10.199", "H 1", "6.14e7", "2.80e-5", "1.40e-7", "0.30"),
    ]
    
    content = """# VALD3 Extract All Request 
# Large linelist for performance testing
# Generated for benchmarking
# 
"""
    
    np.random.seed(42)  # Reproducible results
    
    for i in range(n_lines):
        # Choose random template
        template = templates[i % len(templates)]
        
        # Vary wavelength slightly
        base_wl = float(template[0])
        wl_variation = np.random.uniform(-5.0, 5.0)  # ±5 Å variation
        new_wl = base_wl + wl_variation
        
        # Vary log_gf slightly
        base_loggf = float(template[1])
        loggf_variation = np.random.uniform(-0.5, 0.5)
        new_loggf = base_loggf + loggf_variation
        
        # Create line
        line = f"'{new_wl:.4f}', {new_loggf:6.3f}, {template[2]}, '{template[3]}', {template[4]}, {template[5]}, {template[6]}, {template[7]}\n"
        content += line
    
    return content


def create_large_kurucz_linelist(n_lines: int = 10000) -> str:
    """Create a large Kurucz format linelist for benchmarking"""
    
    templates = [
        (5889.951, 11.00, 0.108, 0.000, 0.5, 1.5),
        (5895.924, 11.00, -0.194, 0.000, 0.5, 0.5),
        (6562.801, 1.00, 0.640, 82259.158, 0.5, 2.5),
        (4861.323, 1.00, -0.020, 82259.158, 1.5, 2.5),
        (5167.321, 12.00, -0.890, 39968.140, 0.5, 1.5),
        (5857.476, 26.00, -2.158, 35767.669, 1.5, 2.5),
        (5862.353, 26.00, -0.058, 35767.669, 0.5, 1.5),
    ]
    
    content = ""
    np.random.seed(42)
    
    for i in range(n_lines):
        template = templates[i % len(templates)]
        
        # Vary parameters slightly
        wl = template[0] + np.random.uniform(-5.0, 5.0)
        species = template[1]
        log_gf = template[2] + np.random.uniform(-0.5, 0.5)
        E_lower = template[3] + np.random.uniform(-100, 100)
        J_lower = template[4]
        J_upper = template[5]
        
        line = f"{wl:8.3f} {species:5.2f} {log_gf:6.3f} {E_lower:8.3f} {J_lower:3.1f} {J_upper:3.1f}\n"
        content += line
    
    return content


def benchmark_jorg_reading(filename: str, format_type: str, n_trials: int = 5) -> dict:
    """Benchmark Jorg linelist reading performance"""
    
    if not JAX_AVAILABLE:
        return {"error": "JAX not available"}
    
    times = []
    
    for trial in range(n_trials):
        start_time = time.time()
        
        try:
            linelist = read_linelist(filename, format=format_type)
            n_lines = len(linelist)
            
            # Force evaluation by accessing data
            wavelengths = linelist.wavelengths_angstrom()
            mean_wl = np.mean(wavelengths)
            
        except Exception as e:
            return {"error": str(e)}
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "max_time": np.max(times),
        "n_lines": n_lines,
        "lines_per_second": n_lines / np.mean(times)
    }


def benchmark_korg_reading(filename: str, format_type: str, n_trials: int = 5) -> dict:
    """Benchmark Korg.jl linelist reading performance"""
    
    # Create Julia script for benchmarking
    julia_script = f'''
using Korg
using BenchmarkTools

function benchmark_reading(filename::String, n_trials::Int)
    times = Float64[]
    n_lines = 0
    
    for trial in 1:n_trials
        start_time = time()
        
        try
            if endswith(filename, ".vald")
                # VALD format reading (if supported)
                linelist = Korg.read_linelist(filename)  
            else
                # Default Korg format
                linelist = Korg.read_linelist(filename)
            end
            
            n_lines = length(linelist.wavelength)
            
            # Force evaluation
            mean_wl = mean(linelist.wavelength)
            
        catch e
            return Dict("error" => string(e))
        end
        
        end_time = time()
        push!(times, end_time - start_time)
    end
    
    return Dict(
        "times" => times,
        "mean_time" => mean(times),
        "std_time" => std(times),
        "min_time" => minimum(times),
        "max_time" => maximum(times),
        "n_lines" => n_lines,
        "lines_per_second" => n_lines / mean(times)
    )
end

result = benchmark_reading("{filename}", {n_trials})
println(JSON.json(result))
'''
    
    # Write Julia script to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jl', delete=False) as f:
        f.write(julia_script)
        julia_file = f.name
    
    try:
        # Run Julia script
        result = subprocess.run(
            ['julia', '--project=..', julia_file],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
            timeout=60
        )
        
        if result.returncode == 0:
            # Parse JSON output
            try:
                return json.loads(result.stdout.strip())
            except json.JSONDecodeError:
                return {"error": f"Could not parse Julia output: {result.stdout}"}
        else:
            return {"error": f"Julia error: {result.stderr}"}
            
    except subprocess.TimeoutExpired:
        return {"error": "Julia benchmark timed out"}
    except FileNotFoundError:
        return {"error": "Julia not found - please ensure Julia is installed"}
    finally:
        # Clean up
        Path(julia_file).unlink()


def run_comprehensive_benchmark():
    """Run comprehensive linelist reading speed comparison"""
    
    print("🏁 Comprehensive Linelist Reading Speed Benchmark")
    print("=" * 60)
    print("Comparing Korg.jl vs Jorg performance")
    
    # Test different linelist sizes
    test_sizes = [1000, 5000, 10000, 25000]
    formats = ["vald", "kurucz"]
    
    results = {}
    
    for format_type in formats:
        print(f"\n📊 Testing {format_type.upper()} format")
        print("-" * 40)
        
        results[format_type] = {}
        
        for n_lines in test_sizes:
            print(f"\n🔍 Testing with {n_lines} lines...")
            
            # Create test linelist
            if format_type == "vald":
                content = create_large_vald_linelist(n_lines)
                extension = ".vald"
            else:
                content = create_large_kurucz_linelist(n_lines)
                extension = ".dat"
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix=extension, delete=False) as f:
                f.write(content)
                test_file = f.name
            
            try:
                # Benchmark Jorg
                print("  Testing Jorg...")
                jorg_result = benchmark_jorg_reading(test_file, format_type, n_trials=3)
                
                # Benchmark Korg (if available)
                print("  Testing Korg.jl...")
                korg_result = benchmark_korg_reading(test_file, format_type, n_trials=3)
                
                results[format_type][n_lines] = {
                    "jorg": jorg_result,
                    "korg": korg_result
                }
                
                # Print results
                if "error" not in jorg_result:
                    print(f"    Jorg: {jorg_result['mean_time']:.3f}s ({jorg_result['lines_per_second']:.0f} lines/s)")
                else:
                    print(f"    Jorg: Error - {jorg_result['error']}")
                    
                if "error" not in korg_result:
                    print(f"    Korg: {korg_result['mean_time']:.3f}s ({korg_result['lines_per_second']:.0f} lines/s)")
                    
                    # Calculate speedup
                    if "error" not in jorg_result:
                        speedup = korg_result['mean_time'] / jorg_result['mean_time']
                        faster = "Jorg" if speedup > 1 else "Korg"
                        ratio = max(speedup, 1/speedup)
                        print(f"    → {faster} is {ratio:.2f}x faster")
                else:
                    print(f"    Korg: Error - {korg_result['error']}")
                
            finally:
                # Clean up
                Path(test_file).unlink()
    
    return results


def print_benchmark_summary(results: dict):
    """Print comprehensive benchmark summary"""
    
    print("\n" + "=" * 60)
    print("📈 BENCHMARK SUMMARY")
    print("=" * 60)
    
    for format_type, format_results in results.items():
        print(f"\n🔸 {format_type.upper()} Format Results:")
        print("-" * 30)
        
        print(f"{'Lines':<8} {'Jorg (s)':<10} {'Korg (s)':<10} {'Speedup':<10}")
        print("-" * 40)
        
        for n_lines, test_results in format_results.items():
            jorg_time = "ERROR" if "error" in test_results["jorg"] else f"{test_results['jorg']['mean_time']:.3f}"
            korg_time = "ERROR" if "error" in test_results["korg"] else f"{test_results['korg']['mean_time']:.3f}"
            
            if "error" not in test_results["jorg"] and "error" not in test_results["korg"]:
                speedup = test_results["korg"]["mean_time"] / test_results["jorg"]["mean_time"]
                speedup_str = f"{speedup:.2f}x" if speedup > 1 else f"{1/speedup:.2f}x K"
            else:
                speedup_str = "N/A"
            
            print(f"{n_lines:<8} {jorg_time:<10} {korg_time:<10} {speedup_str:<10}")
    
    print(f"\n📋 Key Findings:")
    
    # Analyze results
    jorg_working = False
    korg_working = False
    
    for format_results in results.values():
        for test_results in format_results.values():
            if "error" not in test_results["jorg"]:
                jorg_working = True
            if "error" not in test_results["korg"]:
                korg_working = True
    
    if jorg_working and korg_working:
        print("   ✅ Both Jorg and Korg.jl linelist reading are functional")
        print("   📊 Performance comparison shows relative speeds")
        print("   🎯 Choose implementation based on ecosystem needs")
    elif jorg_working:
        print("   ✅ Jorg linelist reading is working correctly")
        print("   ❌ Korg.jl testing encountered issues")
        print("   🔧 May need Korg.jl setup or different linelist format")
    elif korg_working:
        print("   ❌ Jorg linelist reading encountered issues")  
        print("   ✅ Korg.jl linelist reading is working correctly")
    else:
        print("   ❌ Both implementations encountered issues")
        print("   🔧 Need to debug linelist reading setup")
    
    print(f"\n🚀 Implementation Notes:")
    print(f"   • Jorg uses Python/JAX with numpy/pandas parsing")
    print(f"   • Korg.jl uses native Julia with optimized I/O")
    print(f"   • File format and size significantly affect performance")
    print(f"   • HDF5 format (Korg native) typically fastest for large linelists")


def test_hdf5_performance():
    """Test HDF5 format performance specifically"""
    
    print(f"\n💾 Testing HDF5 Format Performance")
    print("=" * 40)
    
    if not JAX_AVAILABLE:
        print("❌ Cannot test HDF5 - JAX not available")
        return
    
    # Create test linelist
    print("Creating test VALD linelist...")
    vald_content = create_large_vald_linelist(10000)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.vald', delete=False) as f:
        f.write(vald_content)
        vald_file = f.name
    
    with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as f:
        h5_file = f.name
    
    try:
        # Read VALD and convert to HDF5
        print("Converting VALD to HDF5...")
        start_time = time.time()
        linelist = read_linelist(vald_file, format="vald")
        save_linelist(h5_file, linelist)
        conversion_time = time.time() - start_time
        
        print(f"  Conversion time: {conversion_time:.3f}s")
        print(f"  Lines converted: {len(linelist)}")
        
        # Test HDF5 reading speed
        print("\nTesting HDF5 reading speed...")
        h5_result = benchmark_jorg_reading(h5_file, "korg", n_trials=5)
        
        # Test VALD reading speed for comparison
        print("Testing VALD reading speed...")
        vald_result = benchmark_jorg_reading(vald_file, "vald", n_trials=5)
        
        if "error" not in h5_result and "error" not in vald_result:
            print(f"\n📊 HDF5 vs VALD Performance:")
            print(f"  HDF5:  {h5_result['mean_time']:.3f}s ({h5_result['lines_per_second']:.0f} lines/s)")
            print(f"  VALD:  {vald_result['mean_time']:.3f}s ({vald_result['lines_per_second']:.0f} lines/s)")
            
            speedup = vald_result['mean_time'] / h5_result['mean_time']
            print(f"  → HDF5 is {speedup:.1f}x faster than VALD")
            
        else:
            if "error" in h5_result:
                print(f"❌ HDF5 error: {h5_result['error']}")
            if "error" in vald_result:
                print(f"❌ VALD error: {vald_result['error']}")
    
    finally:
        # Clean up
        Path(vald_file).unlink()
        Path(h5_file).unlink()


def main():
    """Main benchmark function"""
    
    print("⚡ Linelist Reading Speed Test: Korg.jl vs Jorg")
    print("=" * 60)
    
    if not JAX_AVAILABLE:
        print("❌ Cannot run Jorg tests - JAX not available")
        print("Please install JAX: pip install jax jaxlib")
        return
    
    try:
        # Run comprehensive benchmark
        results = run_comprehensive_benchmark()
        
        # Print summary
        print_benchmark_summary(results)
        
        # Test HDF5 performance
        test_hdf5_performance()
        
        print(f"\n✨ Speed test completed!")
        print(f"   Results show comparative performance between implementations")
        print(f"   Choose the best tool for your specific use case")
        
    except Exception as e:
        print(f"❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()