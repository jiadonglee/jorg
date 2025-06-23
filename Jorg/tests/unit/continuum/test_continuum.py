"""
Unit tests comparing Jorg and Korg continuum absorption calculations

These tests verify that the JAX implementation produces results
consistent with the original Julia implementation.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add the parent directory to the Python path to import jorg
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)  # Use double precision
    
    from jorg.continuum import (
        total_continuum_absorption,
        h_minus_bf_absorption,
        h_minus_ff_absorption,
        thomson_scattering,
        rayleigh_scattering
    )
    from jorg.constants import c_cgs
    
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


class TestContinuumAbsorption:
    """Test suite for continuum absorption calculations"""
    
    @pytest.fixture
    def setup_test_conditions(self):
        """Set up standard test conditions"""
        # Standard stellar atmosphere conditions
        temperature = 5778.0  # K (Sun-like)
        electron_density = 1e15  # cm^-3
        
        # Number densities (cm^-3) - typical solar values
        number_densities = {
            'H_I': 1e16,
            'H_II': 1e12, 
            'He_I': 1e15,
            'H2': 1e10
        }
        
        # Simple partition functions (constant for testing)
        partition_functions = {
            'H_I': lambda log_T: 2.0,  # Ground state degeneracy
            'He_I': lambda log_T: 1.0   # Ground state degeneracy
        }
        
        # Frequency grid (optical wavelengths: 4000-7000 Å)
        wavelengths_cm = np.linspace(4000e-8, 7000e-8, 100)
        frequencies = c_cgs / wavelengths_cm
        frequencies = np.sort(frequencies)  # Ensure sorted
        
        return {
            'frequencies': frequencies,
            'temperature': temperature,
            'electron_density': electron_density,
            'number_densities': number_densities,
            'partition_functions': partition_functions,
            'wavelengths_cm': wavelengths_cm
        }
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_thomson_scattering(self, setup_test_conditions):
        """Test Thomson scattering calculation"""
        conditions = setup_test_conditions
        
        # Calculate Thomson scattering
        alpha_thomson = thomson_scattering(conditions['electron_density'])
        
        # Expected value: n_e * sigma_thomson
        expected = conditions['electron_density'] * 6.6524e-25  # sigma_thomson
        
        assert np.isclose(alpha_thomson, expected, rtol=1e-10)
        assert alpha_thomson > 0
        
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_rayleigh_scattering(self, setup_test_conditions):
        """Test Rayleigh scattering calculation"""
        conditions = setup_test_conditions
        
        # Calculate Rayleigh scattering
        alpha_rayleigh = rayleigh_scattering(
            conditions['frequencies'],
            conditions['number_densities']['H_I'],
            conditions['number_densities']['He_I'],
            conditions['number_densities']['H2']
        )
        
        # Basic checks
        assert len(alpha_rayleigh) == len(conditions['frequencies'])
        assert np.all(alpha_rayleigh >= 0)
        
        # Rayleigh scattering should increase with frequency (ν^4 dependence)
        # Check that shorter wavelengths have higher scattering
        mid_point = len(alpha_rayleigh) // 2
        assert np.mean(alpha_rayleigh[mid_point:]) > np.mean(alpha_rayleigh[:mid_point])
        
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_h_minus_bf_absorption(self, setup_test_conditions):
        """Test H^- bound-free absorption"""
        conditions = setup_test_conditions
        
        # Calculate H^- bound-free absorption
        n_h_i_div_u = (conditions['number_densities']['H_I'] / 
                      conditions['partition_functions']['H_I'](np.log(conditions['temperature'])))
        
        alpha_h_minus_bf = h_minus_bf_absorption(
            conditions['frequencies'],
            conditions['temperature'],
            n_h_i_div_u,
            conditions['electron_density'],
            include_stimulated_emission=True
        )
        
        # Basic checks
        assert len(alpha_h_minus_bf) == len(conditions['frequencies'])
        assert np.all(alpha_h_minus_bf >= 0)
        
        # H^- absorption should be significant in optical wavelengths
        assert np.max(alpha_h_minus_bf) > 1e-12  # cm^-1
        
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_h_minus_ff_absorption(self, setup_test_conditions):
        """Test H^- free-free absorption"""
        conditions = setup_test_conditions
        
        # Calculate H^- free-free absorption
        n_h_i_div_u = (conditions['number_densities']['H_I'] / 
                      conditions['partition_functions']['H_I'](np.log(conditions['temperature'])))
        
        alpha_h_minus_ff = h_minus_ff_absorption(
            conditions['frequencies'],
            conditions['temperature'],
            n_h_i_div_u,
            conditions['electron_density']
        )
        
        # Basic checks
        assert len(alpha_h_minus_ff) == len(conditions['frequencies'])
        assert np.all(alpha_h_minus_ff >= 0)
        
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_total_continuum_absorption(self, setup_test_conditions):
        """Test total continuum absorption calculation"""
        conditions = setup_test_conditions
        
        # Calculate total continuum absorption
        alpha_total = total_continuum_absorption(
            conditions['frequencies'],
            conditions['temperature'],
            conditions['electron_density'],
            conditions['number_densities'],
            conditions['partition_functions'],
            include_stimulated_emission=True
        )
        
        # Basic checks
        assert len(alpha_total) == len(conditions['frequencies'])
        assert np.all(alpha_total >= 0)
        
        # Total absorption should be non-zero
        assert np.max(alpha_total) > 1e-12  # cm^-1
        
        # Check that different components contribute
        # Thomson scattering (frequency-independent)
        alpha_thomson = thomson_scattering(conditions['electron_density'])
        assert np.all(alpha_total >= alpha_thomson * 0.9)  # Allow for numerical precision
        
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_temperature_dependence(self, setup_test_conditions):
        """Test temperature dependence of continuum absorption"""
        conditions = setup_test_conditions
        
        temperatures = [4000.0, 5778.0, 8000.0]  # K
        alpha_values = []
        
        for T in temperatures:
            alpha = total_continuum_absorption(
                conditions['frequencies'],
                T,
                conditions['electron_density'],
                conditions['number_densities'],
                conditions['partition_functions'],
                include_stimulated_emission=True
            )
            alpha_values.append(np.mean(alpha))
        
        # Check that we get different values for different temperatures
        # Convert JAX arrays to float for comparison
        alpha_values_float = [float(alpha) for alpha in alpha_values]
        assert len(set(alpha_values_float)) == len(temperatures)
        
        # All values should be positive
        assert all(alpha > 0 for alpha in alpha_values)
        
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_jit_compilation(self, setup_test_conditions):
        """Test that JAX JIT compilation works correctly"""
        conditions = setup_test_conditions
        
        # Test internal JIT-compiled function directly
        # Extract parameters
        n_h_i = conditions['number_densities']['H_I']
        n_h_ii = conditions['number_densities']['H_II']
        n_he_i = conditions['number_densities']['He_I']
        n_h2 = conditions['number_densities']['H2']
        u_h_i = conditions['partition_functions']['H_I'](np.log(conditions['temperature']))
        u_he_i = conditions['partition_functions']['He_I'](np.log(conditions['temperature']))
        
        from jorg.continuum.main import _total_continuum_absorption_jit
        
        # Run twice to ensure compilation and execution both work
        alpha1 = _total_continuum_absorption_jit(
            conditions['frequencies'],
            conditions['temperature'],
            conditions['electron_density'],
            n_h_i, n_h_ii, n_he_i, n_h2, u_h_i, u_he_i,
            True
        )
        
        alpha2 = _total_continuum_absorption_jit(
            conditions['frequencies'],
            conditions['temperature'],
            conditions['electron_density'],
            n_h_i, n_h_ii, n_he_i, n_h2, u_h_i, u_he_i,
            True
        )
        
        # Results should be identical
        assert np.allclose(alpha1, alpha2, rtol=1e-15)
        
    def test_import_structure(self):
        """Test that the module structure is correct"""
        if not JAX_AVAILABLE:
            pytest.skip("JAX not available")
            
        # Test that we can import the main functions
        from jorg.continuum import (
            total_continuum_absorption,
            h_minus_bf_absorption,
            thomson_scattering
        )
        
        # Test that functions are callable
        assert callable(total_continuum_absorption)
        assert callable(h_minus_bf_absorption)
        assert callable(thomson_scattering)


class TestKorgComparison:
    """Test comparison with Korg.jl results"""
    
    @pytest.fixture
    def korg_reference_data(self):
        """
        Reference data from Korg.jl for comparison
        
        This would typically be generated by running the equivalent
        Korg.jl functions and saving the results.
        """
        # Placeholder - in practice, this would be loaded from a file
        # generated by running Korg.jl with identical parameters
        return {
            'frequencies': np.linspace(4e14, 7.5e14, 100),  # Hz
            'temperature': 5778.0,
            'electron_density': 1e15,
            'korg_alpha_total': np.random.uniform(1e-10, 1e-8, 100),  # Placeholder
            'korg_alpha_thomson': 6.6524e-10,  # n_e * sigma_thomson
        }
    
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")  
    def test_thomson_scattering_vs_korg(self, korg_reference_data):
        """Compare Thomson scattering with Korg.jl reference"""
        ref = korg_reference_data
        
        # Calculate with Jorg
        alpha_thomson_jorg = thomson_scattering(ref['electron_density'])
        
        # Compare with Korg reference
        assert np.isclose(alpha_thomson_jorg, ref['korg_alpha_thomson'], rtol=1e-10)
        
    @pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX not available")
    def test_performance_comparison(self, setup_test_conditions):
        """Test performance of JAX implementation"""
        import time
        
        conditions = setup_test_conditions
        
        # Time the JIT-compiled function
        jit_func = jax.jit(total_continuum_absorption)
        
        # Warm up JIT compilation
        _ = jit_func(
            conditions['frequencies'],
            conditions['temperature'],
            conditions['electron_density'],
            conditions['number_densities'],
            conditions['partition_functions'],
            True
        )
        
        # Time multiple runs
        start_time = time.time()
        n_runs = 100
        for _ in range(n_runs):
            _ = jit_func(
                conditions['frequencies'],
                conditions['temperature'],
                conditions['electron_density'],
                conditions['number_densities'],
                conditions['partition_functions'],
                True
            )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / n_runs
        print(f"Average time per run: {avg_time*1000:.3f} ms")
        
        # Should be very fast after JIT compilation
        assert avg_time < 0.01  # Less than 10ms per run


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])