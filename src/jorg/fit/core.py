"""
Core Parameter Fitting Functions
===============================

This module provides the main interfaces for parameter fitting in Jorg,
including spectral fitting and parameter validation.

Key Features:
- JAX-optimized spectral fitting with automatic differentiation
- Robust parameter validation and bounds checking
- GPU-accelerated optimization
- Integration with enhanced continuum opacity
- Flexible fitting windows and LSF convolution

Main Functions:
- fit_spectrum: Primary spectral fitting interface
- validate_fit_parameters: Parameter validation and setup
- FitResult: Dataclass for fitting results
- FitParameters: Dataclass for fitting parameters
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass
import numpy as np

# Jorg imports
from ..synthesis import synthesize_spectrum
from ..atmosphere import interpolate_marcs_atmosphere
from ..lines.datatypes import Line
from ..statmech.species import Species
from ..utils.wavelength_utils import create_wavelength_grid

# Fitting module imports
from .parameter_scaling import transform_parameters, inverse_transform_parameters, get_parameter_bounds
from .optimization import create_optimizer, chi_squared_objective
from .lsf import compute_lsf_matrix, apply_lsf_convolution
from .utils import validate_observed_spectrum, setup_wavelength_windows, FittingError

@dataclass
class FitParameters:
    """
    Container for fitting parameters with bounds and metadata
    """
    # Stellar parameters
    Teff: float = 5780.0
    logg: float = 4.4
    m_H: float = 0.0
    alpha_H: Optional[float] = None
    vmic: float = 1.0
    
    # Rotational parameters
    vsini: float = 0.0
    epsilon: float = 0.6  # Limb darkening coefficient
    
    # Continuum parameters
    cntm_offset: float = 0.0
    cntm_slope: float = 0.0
    
    # Abundance parameters (element symbol -> abundance)
    abundances: Dict[str, float] = None
    
    def __post_init__(self):
        """Initialize abundances dict if not provided"""
        if self.abundances is None:
            self.abundances = {}
        
        # Set alpha_H to m_H if not specified
        if self.alpha_H is None:
            self.alpha_H = self.m_H
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for optimization"""
        params = {
            'Teff': self.Teff,
            'logg': self.logg,
            'm_H': self.m_H,
            'alpha_H': self.alpha_H,
            'vmic': self.vmic,
            'vsini': self.vsini,
            'epsilon': self.epsilon,
            'cntm_offset': self.cntm_offset,
            'cntm_slope': self.cntm_slope
        }
        
        # Add abundances
        params.update(self.abundances)
        
        return params
    
    @classmethod
    def from_dict(cls, params: Dict[str, float]) -> 'FitParameters':
        """Create from dictionary"""
        # Extract standard parameters
        standard_params = {
            'Teff': params.get('Teff', 5780.0),
            'logg': params.get('logg', 4.4),
            'm_H': params.get('m_H', 0.0),
            'alpha_H': params.get('alpha_H'),
            'vmic': params.get('vmic', 1.0),
            'vsini': params.get('vsini', 0.0),
            'epsilon': params.get('epsilon', 0.6),
            'cntm_offset': params.get('cntm_offset', 0.0),
            'cntm_slope': params.get('cntm_slope', 0.0)
        }
        
        # Extract abundance parameters
        from ..constants import ELEMENT_SYMBOLS
        abundances = {
            symbol: params[symbol] 
            for symbol in ELEMENT_SYMBOLS 
            if symbol in params
        }
        
        return cls(abundances=abundances, **standard_params)

@dataclass
class FitResult:
    """
    Container for fitting results with diagnostics and uncertainties
    """
    # Best-fit parameters
    best_fit_params: FitParameters
    
    # Best-fit synthetic spectrum
    best_fit_flux: jnp.ndarray
    
    # Fitting diagnostics
    chi_squared: float
    reduced_chi_squared: float
    n_data_points: int
    n_parameters: int
    
    # Optimization results
    converged: bool
    n_iterations: int
    optimization_time: float
    
    # Parameter uncertainties (1-sigma)
    parameter_uncertainties: Dict[str, float]
    
    # Covariance matrix
    covariance_matrix: jnp.ndarray
    parameter_names: List[str]
    
    # Wavelength mask used in fitting
    wavelength_mask: jnp.ndarray
    
    # Fitting residuals
    residuals: jnp.ndarray
    normalized_residuals: jnp.ndarray
    
    # Additional diagnostics
    final_gradient_norm: float
    condition_number: float
    
    def summary(self) -> str:
        """Create a summary string of the fit results"""
        summary = f"""
Fitting Results Summary
=====================

Best-fit Parameters:
  Teff = {self.best_fit_params.Teff:.0f} ± {self.parameter_uncertainties.get('Teff', 0.0):.0f} K
  logg = {self.best_fit_params.logg:.2f} ± {self.parameter_uncertainties.get('logg', 0.0):.2f}
  [m/H] = {self.best_fit_params.m_H:.2f} ± {self.parameter_uncertainties.get('m_H', 0.0):.2f}
  vmic = {self.best_fit_params.vmic:.2f} ± {self.parameter_uncertainties.get('vmic', 0.0):.2f} km/s
  vsini = {self.best_fit_params.vsini:.2f} ± {self.parameter_uncertainties.get('vsini', 0.0):.2f} km/s

Fit Quality:
  χ² = {self.chi_squared:.2f}
  Reduced χ² = {self.reduced_chi_squared:.2f}
  Data points = {self.n_data_points}
  Parameters = {self.n_parameters}
  
Optimization:
  Converged = {self.converged}
  Iterations = {self.n_iterations}
  Time = {self.optimization_time:.2f} s
  Final gradient norm = {self.final_gradient_norm:.2e}
  Condition number = {self.condition_number:.2e}
        """
        
        if self.best_fit_params.abundances:
            summary += "\nAbundances:\n"
            for element, abundance in self.best_fit_params.abundances.items():
                uncertainty = self.parameter_uncertainties.get(element, 0.0)
                summary += f"  [{element}/H] = {abundance:.3f} ± {uncertainty:.3f}\n"
        
        return summary

def validate_fit_parameters(
    initial_params: Union[Dict[str, float], FitParameters],
    fixed_params: Union[Dict[str, float], FitParameters] = None,
    parameter_bounds: Dict[str, Tuple[float, float]] = None
) -> Tuple[FitParameters, FitParameters, Dict[str, Tuple[float, float]]]:
    """
    Validate and setup fitting parameters
    
    Parameters:
    -----------
    initial_params : Dict or FitParameters
        Initial parameter guesses for optimization
    fixed_params : Dict or FitParameters, optional
        Parameters to hold fixed during fitting
    parameter_bounds : Dict, optional
        Parameter bounds for optimization
        
    Returns:
    --------
    initial_params : FitParameters
        Validated initial parameters
    fixed_params : FitParameters
        Validated fixed parameters
    bounds : Dict
        Parameter bounds
    """
    # Convert to FitParameters if needed
    if isinstance(initial_params, dict):
        initial_params = FitParameters.from_dict(initial_params)
    
    if fixed_params is None:
        fixed_params = FitParameters()
    elif isinstance(fixed_params, dict):
        fixed_params = FitParameters.from_dict(fixed_params)
    
    # Get default bounds if not provided
    if parameter_bounds is None:
        parameter_bounds = get_parameter_bounds()
    
    # Validate required parameters
    required_params = ['Teff', 'logg']
    all_params = set(initial_params.to_dict().keys()) | set(fixed_params.to_dict().keys())
    
    for param in required_params:
        if param not in all_params:
            raise ValueError(f"Required parameter '{param}' not found in initial_params or fixed_params")
    
    # Check for parameter conflicts
    initial_dict = initial_params.to_dict()
    fixed_dict = fixed_params.to_dict()
    
    conflicts = set(initial_dict.keys()) & set(fixed_dict.keys())
    if conflicts:
        raise ValueError(f"Parameters specified in both initial_params and fixed_params: {conflicts}")
    
    # Validate parameter bounds
    for param, value in {**initial_dict, **fixed_dict}.items():
        if param in parameter_bounds:
            lower, upper = parameter_bounds[param]
            if not (lower <= value <= upper):
                raise ValueError(f"Parameter '{param}' value {value} outside bounds [{lower}, {upper}]")
    
    return initial_params, fixed_params, parameter_bounds

def fit_spectrum(
    obs_wavelengths: jnp.ndarray,
    obs_flux: jnp.ndarray,
    obs_error: jnp.ndarray,
    linelist: List[Line],
    initial_params: Union[Dict[str, float], FitParameters],
    fixed_params: Union[Dict[str, float], FitParameters] = None,
    R: Union[float, Callable[[jnp.ndarray], jnp.ndarray]] = None,
    lsf_matrix: Optional[jnp.ndarray] = None,
    synthesis_wavelengths: Optional[jnp.ndarray] = None,
    windows: Optional[List[Tuple[float, float]]] = None,
    wavelength_buffer: float = 1.0,
    optimizer: str = 'bfgs',
    max_iterations: int = 1000,
    tolerance: float = 1e-4,
    time_limit: float = 10000.0,
    adjust_continuum: bool = False,
    verbose: bool = False,
    **synthesis_kwargs
) -> FitResult:
    """
    Fit stellar parameters and abundances to observed spectrum
    
    Parameters:
    -----------
    obs_wavelengths : jnp.ndarray
        Observed wavelengths in Angstroms (vacuum)
    obs_flux : jnp.ndarray
        Observed flux (rectified/normalized)
    obs_error : jnp.ndarray
        Uncertainties in observed flux
    linelist : List[Line]
        Line list for synthesis
    initial_params : Dict or FitParameters
        Initial parameter guesses
    fixed_params : Dict or FitParameters, optional
        Parameters to hold fixed
    R : float or callable, optional
        Spectral resolution (constant or wavelength-dependent)
    lsf_matrix : jnp.ndarray, optional
        Pre-computed LSF matrix
    synthesis_wavelengths : jnp.ndarray, optional
        Wavelengths for synthesis (if LSF matrix provided)
    windows : List[Tuple], optional
        Wavelength windows for fitting
    wavelength_buffer : float, optional
        Buffer around windows for synthesis
    optimizer : str, optional
        Optimization algorithm ('bfgs', 'adam', 'lbfgs')
    max_iterations : int, optional
        Maximum optimization iterations
    tolerance : float, optional
        Convergence tolerance
    time_limit : float, optional
        Time limit in seconds
    adjust_continuum : bool, optional
        Apply linear continuum adjustment
    verbose : bool, optional
        Print optimization progress
    **synthesis_kwargs
        Additional arguments for synthesis
        
    Returns:
    --------
    fit_result : FitResult
        Comprehensive fitting results
    """
    import time
    start_time = time.time()
    
    # Validate inputs
    validate_observed_spectrum(obs_wavelengths, obs_flux, obs_error)
    initial_params, fixed_params, bounds = validate_fit_parameters(
        initial_params, fixed_params
    )
    
    # Setup wavelengths and LSF
    if lsf_matrix is None:
        if R is None:
            raise ValueError("Must provide either R or lsf_matrix")
        
        synthesis_wavelengths, wavelength_mask, lsf_matrix = setup_wavelength_windows(
            obs_wavelengths, windows, wavelength_buffer, R
        )
    else:
        if synthesis_wavelengths is None:
            raise ValueError("Must provide synthesis_wavelengths with lsf_matrix")
        wavelength_mask = jnp.ones(len(obs_wavelengths), dtype=bool)
    
    # Extract parameters to fit
    initial_dict = initial_params.to_dict()
    fixed_dict = fixed_params.to_dict()
    
    params_to_fit = {k: v for k, v in initial_dict.items() if k not in fixed_dict}
    
    if len(params_to_fit) == 0:
        raise ValueError("No parameters to fit - all parameters are fixed")
    
    # Transform parameters to unbounded space
    param_names = list(params_to_fit.keys())
    param_bounds = {name: bounds[name] for name in param_names}
    scaled_initial = transform_parameters(params_to_fit, param_bounds)
    
    # Create objective function
    def objective(scaled_params: jnp.ndarray) -> float:
        return chi_squared_objective(
            scaled_params=scaled_params,
            param_names=param_names,
            param_bounds=param_bounds,
            fixed_params=fixed_dict,
            obs_flux=obs_flux[wavelength_mask],
            obs_error=obs_error[wavelength_mask],
            synthesis_wavelengths=synthesis_wavelengths,
            linelist=linelist,
            lsf_matrix=lsf_matrix,
            adjust_continuum=adjust_continuum,
            **synthesis_kwargs
        )
    
    # Run optimization
    optimizer_func = create_optimizer(
        optimizer,
        max_iterations=max_iterations,
        tolerance=tolerance,
        time_limit=time_limit,
        verbose=verbose
    )
    
    try:
        result = optimizer_func(objective, scaled_initial)
        converged = result.success if hasattr(result, 'success') else True
        n_iterations = result.nit if hasattr(result, 'nit') else 0
        final_gradient_norm = jnp.linalg.norm(result.jac) if hasattr(result, 'jac') else 0.0
        
    except Exception as e:
        raise FittingError(f"Optimization failed: {str(e)}")
    
    # Extract best-fit parameters
    best_fit_scaled = result.x
    best_fit_unscaled = inverse_transform_parameters(best_fit_scaled, param_names, param_bounds)
    
    # Combine with fixed parameters
    all_params = {**fixed_dict, **best_fit_unscaled}
    best_fit_params = FitParameters.from_dict(all_params)
    
    # Generate best-fit spectrum
    best_fit_flux = _generate_synthetic_spectrum(
        synthesis_wavelengths, best_fit_params, linelist, lsf_matrix, 
        adjust_continuum, **synthesis_kwargs
    )
    
    # Calculate diagnostics
    residuals = best_fit_flux - obs_flux[wavelength_mask]
    normalized_residuals = residuals / obs_error[wavelength_mask]
    chi_squared = jnp.sum(normalized_residuals**2)
    n_data = len(obs_flux[wavelength_mask])
    n_params = len(param_names)
    reduced_chi_squared = chi_squared / (n_data - n_params)
    
    # Calculate parameter uncertainties
    try:
        hessian = jax.hessian(objective)(best_fit_scaled)
        covariance_scaled = jnp.linalg.inv(hessian)
        condition_number = jnp.linalg.cond(hessian)
        
        # Transform covariance to physical parameter space
        jacobian = jax.jacobian(
            lambda p: inverse_transform_parameters(p, param_names, param_bounds)
        )(best_fit_scaled)
        
        # Convert jacobian dict to array
        jacobian_array = jnp.stack([jacobian[name] for name in param_names])
        covariance_physical = jacobian_array.T @ covariance_scaled @ jacobian_array
        
        uncertainties = {
            name: jnp.sqrt(jnp.diag(covariance_physical)[i])
            for i, name in enumerate(param_names)
        }
        
    except Exception:
        # Fallback to identity covariance if Hessian fails
        uncertainties = {name: 0.0 for name in param_names}
        covariance_physical = jnp.eye(len(param_names))
        condition_number = jnp.inf
    
    # Create result
    fit_result = FitResult(
        best_fit_params=best_fit_params,
        best_fit_flux=best_fit_flux,
        chi_squared=float(chi_squared),
        reduced_chi_squared=float(reduced_chi_squared),
        n_data_points=n_data,
        n_parameters=n_params,
        converged=converged,
        n_iterations=n_iterations,
        optimization_time=time.time() - start_time,
        parameter_uncertainties=uncertainties,
        covariance_matrix=covariance_physical,
        parameter_names=param_names,
        wavelength_mask=wavelength_mask,
        residuals=residuals,
        normalized_residuals=normalized_residuals,
        final_gradient_norm=float(final_gradient_norm),
        condition_number=float(condition_number)
    )
    
    return fit_result

def _generate_synthetic_spectrum(
    wavelengths: jnp.ndarray,
    params: FitParameters,
    linelist: List[Line],
    lsf_matrix: jnp.ndarray,
    adjust_continuum: bool = False,
    **synthesis_kwargs
) -> jnp.ndarray:
    """
    Generate synthetic spectrum for fitting
    
    This is an internal function used by fit_spectrum to generate
    synthetic spectra during optimization.
    """
    # Create atmosphere
    atmosphere = interpolate_marcs_atmosphere(
        teff=params.Teff,
        logg=params.logg,
        metallicity=params.m_H,
        alpha_enhancement=params.alpha_H
    )
    
    # Prepare abundances
    abundances = params.abundances.copy()
    
    # Synthesize spectrum
    flux, continuum = synthesize_spectrum(
        wavelengths=wavelengths,
        atmosphere=atmosphere,
        linelist=linelist,
        abundances=abundances,
        vmic=params.vmic,
        continuum_method='enhanced',  # Use 102.5% agreement continuum
        **synthesis_kwargs
    )
    
    # Apply continuum adjustment if requested
    if adjust_continuum or params.cntm_offset != 0.0 or params.cntm_slope != 0.0:
        central_wavelength = (wavelengths[0] + wavelengths[-1]) / 2
        continuum_adjustment = (
            1.0 - params.cntm_offset - 
            params.cntm_slope * (wavelengths - central_wavelength)
        )
        flux = flux / (continuum * continuum_adjustment)
    else:
        flux = flux / continuum
    
    # Apply rotational broadening if requested
    if params.vsini > 0.0:
        flux = _apply_rotational_broadening(
            flux, wavelengths, params.vsini, params.epsilon
        )
    
    # Apply LSF convolution
    return lsf_matrix @ flux

def _apply_rotational_broadening(
    flux: jnp.ndarray, 
    wavelengths: jnp.ndarray, 
    vsini: float, 
    epsilon: float
) -> jnp.ndarray:
    """
    Apply rotational broadening to spectrum
    
    This is a placeholder implementation - the actual rotational
    broadening function would need to be implemented properly.
    """
    # TODO: Implement proper rotational broadening
    # For now, return unchanged flux
    return flux