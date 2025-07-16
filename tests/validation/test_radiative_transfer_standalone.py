#!/usr/bin/env python3
"""
Radiative Transfer Standalone Test

This test validates Jorg's radiative transfer implementation by running
comprehensive tests across different schemes and atmospheric configurations.
"""

import sys
import numpy as np
import jax.numpy as jnp
from pathlib import Path

# Add Jorg to path
jorg_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(jorg_path))

from jorg.radiative_transfer import (
    radiative_transfer, 
    generate_mu_grid,
    calculate_rays,
    compute_tau_anchored,
    compute_I_linear_flux_only,
    exponential_integral_2
)
from jorg.constants import PLANCK_H, BOLTZMANN_K, SPEED_OF_LIGHT


def test_mu_grid_generation():
    """Test angular quadrature grid generation"""
    print("Testing μ grid generation...")
    
    # Test Gauss-Legendre grid
    mu_grid, mu_weights = generate_mu_grid(5)
    assert len(mu_grid) == 5
    assert len(mu_weights) == 5
    assert np.abs(np.sum(mu_weights) - 1.0) < 1e-10
    assert np.all(mu_grid >= 0) and np.all(mu_grid <= 1)
    print("  ✓ Gauss-Legendre grid generation")
    
    # Test custom grid
    custom_mu = jnp.array([0.1, 0.5, 1.0])
    mu_grid, mu_weights = generate_mu_grid(custom_mu)
    assert len(mu_grid) == 3
    assert np.allclose(mu_grid, custom_mu)
    print("  ✓ Custom μ grid generation")
    
    # Test single point
    mu_grid, mu_weights = generate_mu_grid(jnp.array([1.0]))
    assert len(mu_grid) == 1
    assert mu_weights[0] == 1.0
    print("  ✓ Single point grid")


def test_ray_calculation():
    """Test ray path calculation"""
    print("\nTesting ray calculation...")
    
    # Plane-parallel atmosphere
    mu_surface = jnp.array([0.5, 1.0])
    heights = jnp.array([0, 1e6, 2e6, 3e6])  # cm
    
    rays = calculate_rays(mu_surface, heights, spherical=False)
    assert len(rays) == 2
    
    # Check path scaling
    path1, dsdz1 = rays[0]  # μ = 0.5
    path2, dsdz2 = rays[1]  # μ = 1.0
    assert np.allclose(path1, heights / 0.5)
    assert np.allclose(path2, heights / 1.0)
    print("  ✓ Plane-parallel rays")
    
    # Spherical atmosphere (simplified test)
    radii = jnp.array([1e11, 1.1e11, 1.2e11])  # cm
    rays_sph = calculate_rays(jnp.array([1.0]), radii, spherical=True)
    assert len(rays_sph) == 1
    print("  ✓ Spherical rays")


def test_optical_depth_schemes():
    """Test optical depth calculation schemes"""
    print("\nTesting optical depth schemes...")
    
    # Test data
    alpha = jnp.array([1e-6, 2e-6, 5e-6, 1e-5])
    tau_ref = jnp.array([1e-3, 1e-2, 1e-1, 1.0])
    integrand_factor = jnp.ones(4) * 1e6
    log_tau_ref = jnp.log(tau_ref)
    
    # Anchored scheme
    tau = compute_tau_anchored(alpha, integrand_factor, log_tau_ref)
    assert len(tau) == len(alpha)
    assert tau[0] == 0.0  # Boundary condition
    assert np.all(tau[1:] > tau[:-1])  # Monotonic increase
    print("  ✓ Anchored optical depth")


def test_intensity_schemes():
    """Test intensity calculation schemes"""
    print("\nTesting intensity schemes...")
    
    # Test data
    tau = jnp.array([0.0, 0.1, 0.5, 1.0, 2.0])
    S = jnp.array([1.0, 1.1, 1.2, 1.3, 1.4])  # Increasing source function
    
    # Linear flux-only scheme
    I_surface = compute_I_linear_flux_only(tau, S)
    assert np.isfinite(I_surface)
    assert I_surface > 0
    print("  ✓ Linear flux-only intensity")


def test_exponential_integral():
    """Test exponential integral approximation"""
    print("\nTesting exponential integral...")
    
    # Test specific values
    x_values = jnp.array([0.1, 1.0, 2.0, 5.0, 10.0])
    E2_values = exponential_integral_2(x_values)
    
    assert np.all(np.isfinite(E2_values))
    assert np.all(E2_values > 0)
    
    # Check boundary conditions
    assert exponential_integral_2(jnp.array([0.0])) == 1.0
    print("  ✓ Exponential integral E₂(x)")


def create_realistic_test_case():
    """Create realistic stellar atmosphere test case"""
    # Solar-type atmosphere
    n_layers = 15
    tau_5000 = np.logspace(-3, 1, n_layers)  # τ₅₀₀₀ from 0.001 to 10
    temperatures = 5800 - 1000 * np.log10(tau_5000 + 0.1)  # Realistic T(τ) relation
    heights = -np.log(tau_5000) * 150e5  # Height scale ~150 km
    
    # Wavelength grid
    wavelengths = np.linspace(5000, 6000, 20)  # Å
    frequencies = SPEED_OF_LIGHT / (wavelengths * 1e-8)
    
    # Realistic absorption coefficient
    alpha = np.zeros((n_layers, len(wavelengths)))
    for i in range(n_layers):
        for j in range(len(wavelengths)):
            # Simple continuum model
            base_alpha = 1e-6 * tau_5000[i]**0.5
            wavelength_dep = (wavelengths[j] / 5500)**(-3)
            alpha[i, j] = base_alpha * wavelength_dep
    
    # Planck source function
    S = np.zeros((n_layers, len(wavelengths)))
    for i in range(n_layers):
        T = temperatures[i]
        for j in range(len(wavelengths)):
            nu = frequencies[j]
            h_nu_kt = PLANCK_H * nu / (BOLTZMANN_K * T)
            # Avoid overflow for large h_nu_kt
            if h_nu_kt > 50:
                S[i, j] = 0.0
            else:
                S[i, j] = (2 * PLANCK_H * nu**3 / SPEED_OF_LIGHT**2) / (np.exp(h_nu_kt) - 1)
    
    return {
        'alpha': jnp.array(alpha),
        'S': jnp.array(S),
        'spatial_coord': jnp.array(heights),
        'tau_ref': jnp.array(tau_5000),
        'wavelengths': wavelengths,
        'temperatures': temperatures
    }


def test_full_radiative_transfer():
    """Test complete radiative transfer calculation"""
    print("\nTesting full radiative transfer...")
    
    # Create test case
    test_case = create_realistic_test_case()
    
    schemes_to_test = [
        ("anchored", "linear_flux_only"),
        ("anchored", "linear"),
        ("bezier", "linear")
    ]
    
    for tau_scheme, I_scheme in schemes_to_test:
        print(f"  Testing {tau_scheme} + {I_scheme}...")
        
        # Set up reference arrays
        alpha_ref = test_case['alpha'][:, 0] if tau_scheme == "anchored" else None
        tau_ref = test_case['tau_ref'] if tau_scheme == "anchored" else None
        
        # Run radiative transfer
        result = radiative_transfer(
            alpha=test_case['alpha'],
            S=test_case['S'],
            spatial_coord=test_case['spatial_coord'],
            mu_points=5,
            spherical=False,
            alpha_ref=alpha_ref,
            tau_ref=tau_ref,
            tau_scheme=tau_scheme,
            I_scheme=I_scheme
        )
        
        # Validate results
        assert np.all(np.isfinite(result.flux))
        assert np.all(result.flux > 0)
        assert len(result.flux) == len(test_case['wavelengths'])
        assert len(result.mu_grid) == 5
        assert np.abs(np.sum(result.mu_weights) - 1.0) < 1e-10
        
        print(f"    ✓ Flux range: {np.min(result.flux):.2e} - {np.max(result.flux):.2e}")


def test_spherical_vs_planar():
    """Test spherical vs plane-parallel geometries"""
    print("\nTesting spherical vs plane-parallel geometries...")
    
    test_case = create_realistic_test_case()
    
    # Convert heights to radii for spherical test
    R_star = 7e10  # cm (solar radius)
    radii = R_star + test_case['spatial_coord']
    
    # Plane-parallel
    result_planar = radiative_transfer(
        alpha=test_case['alpha'],
        S=test_case['S'],
        spatial_coord=test_case['spatial_coord'],
        mu_points=3,
        spherical=False,
        alpha_ref=test_case['alpha'][:, 0],
        tau_ref=test_case['tau_ref'],
        tau_scheme="anchored",
        I_scheme="linear_flux_only"
    )
    
    # Spherical
    result_spherical = radiative_transfer(
        alpha=test_case['alpha'],
        S=test_case['S'],
        spatial_coord=radii,
        mu_points=3,
        spherical=True,
        alpha_ref=test_case['alpha'][:, 0],
        tau_ref=test_case['tau_ref'],
        tau_scheme="anchored",
        I_scheme="linear_flux_only"
    )
    
    # Both should produce valid results
    assert np.all(np.isfinite(result_planar.flux))
    assert np.all(np.isfinite(result_spherical.flux))
    
    # Spherical should generally produce different flux
    relative_diff = np.abs(result_spherical.flux - result_planar.flux) / result_planar.flux
    print(f"  Max relative difference: {np.max(relative_diff):.1%}")
    print("  ✓ Both geometries working")


def main():
    """Main test suite"""
    print("=" * 60)
    print("Jorg Radiative Transfer Standalone Test Suite")
    print("=" * 60)
    
    try:
        # Unit tests
        test_mu_grid_generation()
        test_ray_calculation()
        test_optical_depth_schemes()
        test_intensity_schemes()
        test_exponential_integral()
        
        # Integration tests
        test_full_radiative_transfer()
        test_spherical_vs_planar()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED")
        print("Jorg Radiative Transfer implementation is fully functional!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)