"""
Parameter input handling for GWEMFISH package.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ParameterBounds:
    """Parameter bounds for optimization and sampling."""
    lower: float
    upper: float
    
    def __post_init__(self):
        """Validate bounds."""
        if self.lower >= self.upper:
            raise ValueError("Lower bound must be less than upper bound")


@dataclass
class ParameterPrior:
    """Parameter prior distribution."""
    distribution: str  # 'uniform', 'normal', 'lognormal', 'truncated_normal'
    parameters: Dict[str, float]  # Distribution parameters
    
    def __post_init__(self):
        """Validate prior distribution."""
        valid_distributions = ['uniform', 'normal', 'lognormal', 'truncated_normal']
        if self.distribution not in valid_distributions:
            raise ValueError(f"Distribution must be one of {valid_distributions}")


@dataclass
class ParameterDefinition:
    """Definition for a single parameter."""
    name: str
    value: float
    bounds: Optional[ParameterBounds] = None
    prior: Optional[ParameterPrior] = None
    is_fixed: bool = False
    description: str = ""
    
    def __post_init__(self):
        """Validate parameter definition."""
        if not self.name:
            raise ValueError("Parameter name cannot be empty")
        
        if self.is_fixed and self.bounds is not None:
            print(f"Warning: Fixed parameter '{self.name}' has bounds defined")


class ParameterInput:
    """
    Parameter input handler for GWEMFISH package.
    
    This class manages the input and definition of all parameters
    used in gravitational lensing analysis.
    """
    
    def __init__(self):
        """Initialize the parameter input handler."""
        self.parameters: Dict[str, ParameterDefinition] = {}
        self.parameter_groups: Dict[str, List[str]] = {
            'cosmology': [],
            'lens': [],
            'source': [],
            'lens_light': [],
            'image_positions': [],
            'noise': [],
            'redshifts': []
        }
    
    def add_parameter(self, 
                     name: str,
                     value: float,
                     bounds: Optional[Tuple[float, float]] = None,
                     prior: Optional[Dict[str, Any]] = None,
                     is_fixed: bool = False,
                     description: str = "",
                     group: str = "lens") -> None:
        """
        Add a parameter definition.
        
        Parameters
        ----------
        name : str
            Parameter name
        value : float
            Parameter value
        bounds : tuple, optional
            (lower, upper) bounds for the parameter
        prior : dict, optional
            Prior distribution parameters
        is_fixed : bool
            Whether the parameter is fixed
        description : str
            Parameter description
        group : str
            Parameter group (cosmology, lens, source, etc.)
        """
        # Create bounds object if provided
        bounds_obj = None
        if bounds is not None:
            bounds_obj = ParameterBounds(lower=bounds[0], upper=bounds[1])
        
        # Create prior object if provided
        prior_obj = None
        if prior is not None:
            prior_obj = ParameterPrior(
                distribution=prior['distribution'],
                parameters=prior['parameters']
            )
        
        # Create parameter definition
        param_def = ParameterDefinition(
            name=name,
            value=value,
            bounds=bounds_obj,
            prior=prior_obj,
            is_fixed=is_fixed,
            description=description
        )
        
        self.parameters[name] = param_def
        
        # Add to parameter group
        if group in self.parameter_groups:
            self.parameter_groups[group].append(name)
        else:
            self.parameter_groups[group] = [name]
    
    def add_cosmology_parameters(self, 
                                H0: float = 67.3,
                                Om0: float = 0.316,
                                use_as_parameters: bool = True) -> None:
        """
        Add cosmology parameters.
        
        Parameters
        ----------
        H0 : float
            Hubble constant in km/s/Mpc
        Om0 : float
            Matter density parameter
        use_as_parameters : bool
            Whether to treat as free parameters
        """
        if use_as_parameters:
            self.add_parameter(
                name="H0",
                value=H0,
                bounds=(50.0, 100.0),
                prior={'distribution': 'uniform', 'parameters': {'low': 50.0, 'high': 100.0}},
                description="Hubble constant (km/s/Mpc)",
                group="cosmology"
            )
            
            self.add_parameter(
                name="Om0",
                value=Om0,
                bounds=(0.1, 0.5),
                prior={'distribution': 'uniform', 'parameters': {'low': 0.1, 'high': 0.5}},
                description="Matter density parameter",
                group="cosmology"
            )
        else:
            self.add_parameter(
                name="H0",
                value=H0,
                is_fixed=True,
                description="Hubble constant (km/s/Mpc)",
                group="cosmology"
            )
            
            self.add_parameter(
                name="Om0",
                value=Om0,
                is_fixed=True,
                description="Matter density parameter",
                group="cosmology"
            )
    
    def add_lens_parameters(self, 
                           model_type: str,
                           parameters: Dict[str, float],
                           fixed_parameters: Optional[Dict[str, float]] = None,
                           model_index: int = 0) -> None:
        """
        Add lens model parameters.
        
        Parameters
        ----------
        model_type : str
            Type of lens model
        parameters : dict
            Free parameters
        fixed_parameters : dict, optional
            Fixed parameters
        model_index : int
            Index of the lens model (for multiple models)
        """
        if fixed_parameters is None:
            fixed_parameters = {}
        
        # Add free parameters
        for param_name, value in parameters.items():
            full_name = f"lens_{model_index}_{param_name}"
            
            # Get default bounds and prior based on parameter type
            bounds, prior = self._get_default_bounds_and_prior(param_name, model_type)
            
            self.add_parameter(
                name=full_name,
                value=value,
                bounds=bounds,
                prior=prior,
                description=f"{model_type} {param_name}",
                group="lens"
            )
        
        # Add fixed parameters
        for param_name, value in fixed_parameters.items():
            full_name = f"lens_{model_index}_{param_name}"
            
            self.add_parameter(
                name=full_name,
                value=value,
                is_fixed=True,
                description=f"{model_type} {param_name} (fixed)",
                group="lens"
            )
    
    def add_image_position_parameters(self, 
                                    positions: List[Tuple[float, float]],
                                    bounds: Tuple[float, float] = (-10.0, 10.0)) -> None:
        """
        Add image position parameters.
        
        Parameters
        ----------
        positions : list
            List of (x, y) image positions
        bounds : tuple
            Bounds for position parameters
        """
        for i, (x, y) in enumerate(positions):
            self.add_parameter(
                name=f"image_x{i+1}",
                value=x,
                bounds=bounds,
                prior={'distribution': 'uniform', 'parameters': {'low': bounds[0], 'high': bounds[1]}},
                description=f"Image {i+1} x position (arcsec)",
                group="image_positions"
            )
            
            self.add_parameter(
                name=f"image_y{i+1}",
                value=y,
                bounds=bounds,
                prior={'distribution': 'uniform', 'parameters': {'low': bounds[0], 'high': bounds[1]}},
                description=f"Image {i+1} y position (arcsec)",
                group="image_positions"
            )
    
    def add_source_parameters(self, 
                             model_type: str,
                             parameters: Dict[str, float],
                             fixed_parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Add source model parameters.
        
        Parameters
        ----------
        model_type : str
            Type of source model
        parameters : dict
            Free parameters
        fixed_parameters : dict, optional
            Fixed parameters
        """
        if fixed_parameters is None:
            fixed_parameters = {}
        
        # Add free parameters
        for param_name, value in parameters.items():
            full_name = f"source_{param_name}"
            
            # Get default bounds and prior based on parameter type
            bounds, prior = self._get_default_bounds_and_prior(param_name, model_type)
            
            self.add_parameter(
                name=full_name,
                value=value,
                bounds=bounds,
                prior=prior,
                description=f"Source {param_name}",
                group="source"
            )
        
        # Add fixed parameters
        for param_name, value in fixed_parameters.items():
            full_name = f"source_{param_name}"
            
            self.add_parameter(
                name=full_name,
                value=value,
                is_fixed=True,
                description=f"Source {param_name} (fixed)",
                group="source"
            )
    
    def add_lens_light_parameters(self, 
                                 model_type: str,
                                 parameters: Dict[str, float],
                                 fixed_parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Add lens light model parameters.
        
        Parameters
        ----------
        model_type : str
            Type of lens light model
        parameters : dict
            Free parameters
        fixed_parameters : dict, optional
            Fixed parameters
        """
        if fixed_parameters is None:
            fixed_parameters = {}
        
        # Add free parameters
        for param_name, value in parameters.items():
            full_name = f"light_{param_name}"
            
            # Get default bounds and prior based on parameter type
            bounds, prior = self._get_default_bounds_and_prior(param_name, model_type)
            
            self.add_parameter(
                name=full_name,
                value=value,
                bounds=bounds,
                prior=prior,
                description=f"Lens light {param_name}",
                group="lens_light"
            )
        
        # Add fixed parameters
        for param_name, value in fixed_parameters.items():
            full_name = f"light_{param_name}"
            
            self.add_parameter(
                name=full_name,
                value=value,
                is_fixed=True,
                description=f"Lens light {param_name} (fixed)",
                group="lens_light"
            )
    
    def add_redshift_parameters(self, 
                               z_lens: float = 0.5,
                               z_source: float = 2.0) -> None:
        """
        Add redshift parameters.
        
        Parameters
        ----------
        z_lens : float
            Lens redshift
        z_source : float
            Source redshift
        """
        self.add_parameter(
            name="zl",
            value=z_lens,
            bounds=(0.0, 5.0),
            prior={'distribution': 'uniform', 'parameters': {'low': 0.0, 'high': 5.0}},
            description="Lens redshift",
            group="redshifts"
        )
        
        self.add_parameter(
            name="zs",
            value=z_source,
            bounds=(1.0, 10.0),
            prior={'distribution': 'uniform', 'parameters': {'low': 1.0, 'high': 10.0}},
            description="Source redshift",
            group="redshifts"
        )
    
    def add_noise_parameter(self, 
                           background_rms: float = 0.01,
                           bounds: Tuple[float, float] = (1e-3, 1e-1)) -> None:
        """
        Add noise parameter.
        
        Parameters
        ----------
        background_rms : float
            Background RMS noise
        bounds : tuple
            Bounds for noise parameter
        """
        self.add_parameter(
            name="noise_sigma_bkg",
            value=background_rms,
            bounds=bounds,
            prior={'distribution': 'uniform', 'parameters': {'low': bounds[0], 'high': bounds[1]}},
            description="Background noise RMS",
            group="noise"
        )
    
    def _get_default_bounds_and_prior(self, param_name: str, model_type: str) -> Tuple[Optional[Tuple[float, float]], Optional[Dict[str, Any]]]:
        """Get default bounds and prior for a parameter."""
        # Default bounds and priors based on parameter type
        defaults = {
            'theta_E': ((0.1, 10.0), {'distribution': 'normal', 'parameters': {'loc': 1.5, 'scale': 0.1}}),
            'e1': ((-0.3, 0.3), {'distribution': 'truncated_normal', 'parameters': {'loc': 0.0, 'scale': 0.05, 'low': -0.3, 'high': 0.3}}),
            'e2': ((-0.3, 0.3), {'distribution': 'truncated_normal', 'parameters': {'loc': 0.0, 'scale': 0.05, 'low': -0.3, 'high': 0.3}}),
            'center_x': ((-1.0, 1.0), {'distribution': 'normal', 'parameters': {'loc': 0.0, 'scale': 0.1}}),
            'center_y': ((-1.0, 1.0), {'distribution': 'normal', 'parameters': {'loc': 0.0, 'scale': 0.1}}),
            'gamma1': ((-0.3, 0.3), {'distribution': 'truncated_normal', 'parameters': {'loc': 0.0, 'scale': 0.1, 'low': -0.3, 'high': 0.3}}),
            'gamma2': ((-0.3, 0.3), {'distribution': 'truncated_normal', 'parameters': {'loc': 0.0, 'scale': 0.1, 'low': -0.3, 'high': 0.3}}),
            'amp': ((0.1, 100.0), {'distribution': 'lognormal', 'parameters': {'loc': 1.0, 'scale': 0.1}}),
            'R_sersic': ((0.05, 5.0), {'distribution': 'truncated_normal', 'parameters': {'loc': 0.5, 'scale': 0.1, 'low': 0.05, 'high': 5.0}}),
            'n_sersic': ((1.0, 3.0), {'distribution': 'uniform', 'parameters': {'low': 1.0, 'high': 3.0}}),
        }
        
        return defaults.get(param_name, (None, None))
    
    def get_parameter_values(self) -> Dict[str, float]:
        """Get all parameter values."""
        return {name: param.value for name, param in self.parameters.items()}
    
    def get_free_parameter_values(self) -> Dict[str, float]:
        """Get only free parameter values."""
        return {name: param.value for name, param in self.parameters.items() if not param.is_fixed}
    
    def get_fixed_parameter_values(self) -> Dict[str, float]:
        """Get only fixed parameter values."""
        return {name: param.value for name, param in self.parameters.items() if param.is_fixed}
    
    def get_parameter_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get parameter bounds."""
        bounds = {}
        for name, param in self.parameters.items():
            if param.bounds is not None:
                bounds[name] = (param.bounds.lower, param.bounds.upper)
        return bounds
    
    def get_parameter_priors(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter priors."""
        priors = {}
        for name, param in self.parameters.items():
            if param.prior is not None:
                priors[name] = {
                    'distribution': param.prior.distribution,
                    'parameters': param.prior.parameters
                }
        return priors
    
    def get_parameters_by_group(self, group: str) -> List[str]:
        """Get parameter names in a specific group."""
        return self.parameter_groups.get(group, [])
    
    def print_summary(self) -> None:
        """Print a summary of all parameters."""
        print("GWEMFISH Parameter Input Summary")
        print("=" * 40)
        
        for group, param_names in self.parameter_groups.items():
            if param_names:
                print(f"\n{group.title()} Parameters:")
                for name in param_names:
                    if name in self.parameters:
                        param = self.parameters[name]
                        status = "FIXED" if param.is_fixed else "FREE"
                        bounds_str = f" [{param.bounds.lower:.3f}, {param.bounds.upper:.3f}]" if param.bounds else ""
                        print(f"  {name}: {param.value:.6f} ({status}){bounds_str}")
                        if param.description:
                            print(f"    {param.description}")
        
        print(f"\nTotal Parameters: {len(self.parameters)}")
        print(f"Free Parameters: {len(self.get_free_parameter_values())}")
        print(f"Fixed Parameters: {len(self.get_fixed_parameter_values())}")
