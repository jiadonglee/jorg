"""
Unit tests for partition functions module.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from jorg.statmech.partition_functions import (
    hydrogen_partition_function, 
    simple_partition_function,
    create_partition_function_dict
)


class TestPartitionFunctions:
    """Test partition function calculations."""
    
    def test_hydrogen_partition_function(self):
        """Test hydrogen partition function returns expected value."""
        log_T = jnp.log(5778.0)  # Solar temperature
        result = hydrogen_partition_function(log_T)
        
        # Hydrogen ground state has statistical weight of 2
        assert jnp.isclose(result, 2.0), f"Expected 2.0, got {result}"
    
    def test_hydrogen_partition_function_temperature_independence(self):
        """Test that H partition function is temperature independent in this regime."""
        temperatures = [3000, 5000, 8000]
        results = []
        
        for T in temperatures:
            log_T = jnp.log(float(T))
            result = hydrogen_partition_function(log_T)
            results.append(result)
        
        # All should be 2.0 for this simple implementation
        for result in results:
            assert jnp.isclose(result, 2.0)
    
    def test_simple_partition_function_hydrogen(self):
        """Test simple partition function for hydrogen."""
        log_T = jnp.log(5778.0)
        result = simple_partition_function(1, log_T)  # Z=1 for hydrogen
        
        assert jnp.isclose(result, 2.0), f"Expected 2.0 for hydrogen, got {result}"
    
    def test_simple_partition_function_helium(self):
        """Test simple partition function for helium."""
        log_T = jnp.log(5778.0)
        result = simple_partition_function(2, log_T)  # Z=2 for helium
        
        assert jnp.isclose(result, 1.0), f"Expected 1.0 for helium, got {result}"
    
    def test_simple_partition_function_other_elements(self):
        """Test simple partition function for other elements."""
        log_T = jnp.log(5778.0)
        
        for Z in [3, 6, 8, 26]:  # Li, C, O, Fe
            result = simple_partition_function(Z, log_T)
            assert jnp.isclose(result, 1.0), f"Expected 1.0 for Z={Z}, got {result}"
    
    def test_create_partition_function_dict(self):
        """Test creation of partition function dictionary."""
        partition_funcs = create_partition_function_dict()
        
        # Check that dictionary is created
        assert isinstance(partition_funcs, dict)
        assert len(partition_funcs) > 0
        
        # Check that we have entries for different elements and charges
        expected_keys = ['1_0', '1_1', '2_0', '2_1', '6_0', '6_1']  # H I, H II, He I, He II, C I, C II
        
        for key in expected_keys:
            assert key in partition_funcs, f"Missing key {key} in partition function dict"
    
    def test_partition_function_callable(self):
        """Test that partition functions in dictionary are callable."""
        partition_funcs = create_partition_function_dict()
        log_T = jnp.log(5778.0)
        
        # Test a few functions
        for key in ['1_0', '2_0', '6_0']:
            func = partition_funcs[key]
            result = func(log_T)
            
            # Should return a finite number
            assert jnp.isfinite(result), f"Partition function {key} returned non-finite value"
            assert result > 0, f"Partition function {key} returned non-positive value"


class TestPartitionFunctionReference:
    """Test against reference values from Korg."""
    
    def test_hydrogen_reference_value(self):
        """Test hydrogen partition function against Korg reference."""
        # From korg_detailed_reference.json
        T = 5778.0
        log_T = jnp.log(T)
        
        result = hydrogen_partition_function(log_T)
        reference = 2.000000011513405  # From reference JSON
        
        # Allow for small numerical differences
        assert jnp.isclose(result, reference, rtol=1e-6), \
            f"H partition function: expected {reference}, got {result}"
    
    def test_temperature_scaling(self):
        """Test partition function scaling with temperature."""
        temperatures = jnp.array([3000.0, 5000.0, 8000.0])
        log_temperatures = jnp.log(temperatures)
        
        results = []
        for log_T in log_temperatures:
            result = hydrogen_partition_function(log_T)
            results.append(result)
        
        results = jnp.array(results)
        
        # For this simple implementation, all should be 2.0
        expected = jnp.full_like(results, 2.0)
        assert jnp.allclose(results, expected), \
            f"Temperature scaling failed: {results} vs {expected}"


if __name__ == "__main__":
    pytest.main([__file__])