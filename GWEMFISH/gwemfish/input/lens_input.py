"""
Lens model input handling for GWEMFISH package.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LensModelInput:
    """Input for a single lens model component."""
    model_type: str
    parameters: Dict[str, float]
    fixed_parameters: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the lens model input."""
        if not self.model_type:
            raise ValueError("model_type cannot be empty")
        if not self.parameters:
            raise ValueError("parameters cannot be empty")
    
    def get_all_parameters(self) -> Dict[str, float]:
        """Get all parameters (both free and fixed)."""
        all_params = self.parameters.copy()
        all_params.update(self.fixed_parameters)
        return all_params
    
    def get_free_parameters(self) -> Dict[str, float]:
        """Get only free parameters."""
        return self.parameters.copy()
    
    def get_fixed_parameters(self) -> Dict[str, float]:
        """Get only fixed parameters."""
        return self.fixed_parameters.copy()


@dataclass
class SourceInput:
    """Input for source model."""
    model_type: str = "SersicElliptic"
    parameters: Dict[str, float] = field(default_factory=dict)
    fixed_parameters: Dict[str, float] = field(default_factory=dict)
    position: Tuple[float, float] = (0.3, 0.2)  # (y1, y2) in arcsec
    redshift: float = 2.0
    
    def get_all_parameters(self) -> Dict[str, float]:
        """Get all source parameters."""
        all_params = self.parameters.copy()
        all_params.update(self.fixed_parameters)
        return all_params
    
    def get_free_parameters(self) -> Dict[str, float]:
        """Get only free source parameters."""
        return self.parameters.copy()
    
    def get_fixed_parameters(self) -> Dict[str, float]:
        """Get only fixed source parameters."""
        return self.fixed_parameters.copy()


@dataclass
class LensLightInput:
    """Input for lens light model."""
    model_type: str = "SersicElliptic"
    parameters: Dict[str, float] = field(default_factory=dict)
    fixed_parameters: Dict[str, float] = field(default_factory=dict)
    
    def get_all_parameters(self) -> Dict[str, float]:
        """Get all lens light parameters."""
        all_params = self.parameters.copy()
        all_params.update(self.fixed_parameters)
        return all_params
    
    def get_free_parameters(self) -> Dict[str, float]:
        """Get only free lens light parameters."""
        return self.parameters.copy()
    
    def get_fixed_parameters(self) -> Dict[str, float]:
        """Get only fixed lens light parameters."""
        return self.fixed_parameters.copy()


@dataclass
class CosmologyInput:
    """Input for cosmology parameters."""
    H0: float = 67.3  # km/s/Mpc
    Om0: float = 0.316
    Ode0: Optional[float] = None  # If None, assumes flat universe
    use_as_parameters: bool = True  # Whether to treat H0, Om0 as free parameters
    
    def __post_init__(self):
        """Set Ode0 if not provided."""
        if self.Ode0 is None:
            self.Ode0 = 1.0 - self.Om0


@dataclass
class ObservationInput:
    """Input for observation parameters."""
    pixel_scale: float = 0.08  # arcsec/pixel
    image_size: Tuple[int, int] = (200, 200)  # (nx, ny)
    psf_fwhm: float = 0.3  # arcsec
    background_rms: float = 0.01
    exposure_time: float = 1000.0


class LensInput:
    """
    Main class for handling lens model input and parameter definition.
    
    This class manages the input of lens models, source models, and all
    associated parameters for gravitational lensing analysis.
    """
    
    def __init__(self):
        """Initialize the lens input handler."""
        self.lens_models: List[LensModelInput] = []
        self.source: Optional[SourceInput] = None
        self.lens_light: Optional[LensLightInput] = None
        self.cosmology: Optional[CosmologyInput] = None
        self.observation: Optional[ObservationInput] = None
        
        # Image positions (computed from source position)
        self.image_positions: List[Tuple[float, float]] = []
        self.n_images: int = 0
    
    def add_lens_model(self, 
                      model_type: str, 
                      parameters: Dict[str, float],
                      fixed_parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Add a lens model component.
        
        Parameters
        ----------
        model_type : str
            Type of lens model (e.g., 'SIE', 'SIS', 'NFW', 'SHEAR')
        parameters : dict
            Free parameters for the model
        fixed_parameters : dict, optional
            Fixed parameters for the model
        """
        if fixed_parameters is None:
            fixed_parameters = {}
        
        lens_model = LensModelInput(
            model_type=model_type,
            parameters=parameters,
            fixed_parameters=fixed_parameters
        )
        
        self.lens_models.append(lens_model)
    
    def set_source(self, 
                   model_type: str = "SersicElliptic",
                   parameters: Optional[Dict[str, float]] = None,
                   fixed_parameters: Optional[Dict[str, float]] = None,
                   position: Tuple[float, float] = (0.3, 0.2),
                   redshift: float = 2.0) -> None:
        """
        Set the source model.
        
        Parameters
        ----------
        model_type : str
            Type of source model
        parameters : dict, optional
            Free parameters for the source
        fixed_parameters : dict, optional
            Fixed parameters for the source
        position : tuple
            Source position (y1, y2) in arcsec
        redshift : float
            Source redshift
        """
        if parameters is None:
            parameters = {}
        if fixed_parameters is None:
            fixed_parameters = {}
        
        self.source = SourceInput(
            model_type=model_type,
            parameters=parameters,
            fixed_parameters=fixed_parameters,
            position=position,
            redshift=redshift
        )
    
    def set_lens_light(self, 
                      model_type: str = "SersicElliptic",
                      parameters: Optional[Dict[str, float]] = None,
                      fixed_parameters: Optional[Dict[str, float]] = None) -> None:
        """
        Set the lens light model.
        
        Parameters
        ----------
        model_type : str
            Type of lens light model
        parameters : dict, optional
            Free parameters for the lens light
        fixed_parameters : dict, optional
            Fixed parameters for the lens light
        """
        if parameters is None:
            parameters = {}
        if fixed_parameters is None:
            fixed_parameters = {}
        
        self.lens_light = LensLightInput(
            model_type=model_type,
            parameters=parameters,
            fixed_parameters=fixed_parameters
        )
    
    def set_cosmology(self, 
                     H0: float = 67.3,
                     Om0: float = 0.316,
                     Ode0: Optional[float] = None,
                     use_as_parameters: bool = True) -> None:
        """
        Set cosmology parameters.
        
        Parameters
        ----------
        H0 : float
            Hubble constant in km/s/Mpc
        Om0 : float
            Matter density parameter
        Ode0 : float, optional
            Dark energy density parameter
        use_as_parameters : bool
            Whether to treat H0, Om0 as free parameters
        """
        self.cosmology = CosmologyInput(
            H0=H0,
            Om0=Om0,
            Ode0=Ode0,
            use_as_parameters=use_as_parameters
        )
    
    def set_observation(self, 
                       pixel_scale: float = 0.08,
                       image_size: Tuple[int, int] = (200, 200),
                       psf_fwhm: float = 0.3,
                       background_rms: float = 0.01,
                       exposure_time: float = 1000.0) -> None:
        """
        Set observation parameters.
        
        Parameters
        ----------
        pixel_scale : float
            Pixel scale in arcsec/pixel
        image_size : tuple
            Image size (nx, ny)
        psf_fwhm : float
            PSF FWHM in arcsec
        background_rms : float
            Background RMS noise
        exposure_time : float
            Exposure time
        """
        self.observation = ObservationInput(
            pixel_scale=pixel_scale,
            image_size=image_size,
            psf_fwhm=psf_fwhm,
            background_rms=background_rms,
            exposure_time=exposure_time
        )
    
    def set_image_positions(self, positions: List[Tuple[float, float]]) -> None:
        """
        Set image positions (computed from source position via lens equation).
        
        Parameters
        ----------
        positions : list
            List of (x, y) image positions in arcsec
        """
        self.image_positions = positions
        self.n_images = len(positions)
    
    def get_all_parameter_names(self) -> List[str]:
        """Get all parameter names in the model."""
        param_names = []
        
        # Add cosmology parameters if they're treated as free
        if self.cosmology and self.cosmology.use_as_parameters:
            param_names.extend(["H0", "Om0"])
        
        # Add lens model parameters
        for i, lens_model in enumerate(self.lens_models):
            for param_name in lens_model.get_free_parameters().keys():
                param_names.append(f"lens_{i}_{param_name}")
        
        # Add image position parameters
        for i in range(self.n_images):
            param_names.extend([f"image_x{i+1}", f"image_y{i+1}"])
        
        # Add source parameters
        if self.source:
            for param_name in self.source.get_free_parameters().keys():
                param_names.append(f"source_{param_name}")
        
        # Add lens light parameters
        if self.lens_light:
            for param_name in self.lens_light.get_free_parameters().keys():
                param_names.append(f"light_{param_name}")
        
        # Add noise parameter
        param_names.append("noise_sigma_bkg")
        
        # Add redshift parameters
        param_names.extend(["zl", "zs"])
        
        return param_names
    
    def get_fixed_parameters(self) -> Dict[str, float]:
        """Get all fixed parameters."""
        fixed_params = {}
        
        # Add fixed lens parameters
        for i, lens_model in enumerate(self.lens_models):
            for param_name, value in lens_model.get_fixed_parameters().items():
                fixed_params[f"lens_{i}_{param_name}"] = value
        
        # Add fixed source parameters
        if self.source:
            for param_name, value in self.source.get_fixed_parameters().items():
                fixed_params[f"source_{param_name}"] = value
        
        # Add fixed lens light parameters
        if self.lens_light:
            for param_name, value in self.lens_light.get_fixed_parameters().items():
                fixed_params[f"light_{param_name}"] = value
        
        # Add fixed cosmology parameters
        if self.cosmology and not self.cosmology.use_as_parameters:
            fixed_params["H0"] = self.cosmology.H0
            fixed_params["Om0"] = self.cosmology.Om0
        
        return fixed_params
    
    def get_lens_model_list(self) -> List[str]:
        """Get list of lens model types."""
        return [model.model_type for model in self.lens_models]
    
    def get_lens_kwargs(self) -> List[Dict[str, float]]:
        """Get lens model parameters in herculens format."""
        kwargs_list = []
        for lens_model in self.lens_models:
            kwargs_list.append(lens_model.get_all_parameters())
        return kwargs_list
    
    def get_source_kwargs(self) -> Dict[str, float]:
        """Get source model parameters."""
        if self.source is None:
            return {}
        return self.source.get_all_parameters()
    
    def get_lens_light_kwargs(self) -> Dict[str, float]:
        """Get lens light model parameters."""
        if self.lens_light is None:
            return {}
        return self.lens_light.get_all_parameters()
    
    def validate_input(self) -> bool:
        """
        Validate the input configuration.
        
        Returns
        -------
        bool
            True if input is valid
        """
        if not self.lens_models:
            raise ValueError("At least one lens model must be specified")
        
        if self.source is None:
            raise ValueError("Source model must be specified")
        
        if self.cosmology is None:
            raise ValueError("Cosmology parameters must be specified")
        
        if self.observation is None:
            raise ValueError("Observation parameters must be specified")
        
        if self.n_images == 0:
            raise ValueError("Image positions must be set")
        
        return True
    
    def print_summary(self) -> None:
        """Print a summary of the input configuration."""
        print("GWEMFISH Lens Input Summary")
        print("=" * 40)
        
        print(f"Lens Models ({len(self.lens_models)}):")
        for i, model in enumerate(self.lens_models):
            print(f"  {i+1}. {model.model_type}")
            print(f"     Free parameters: {list(model.get_free_parameters().keys())}")
            if model.get_fixed_parameters():
                print(f"     Fixed parameters: {list(model.get_fixed_parameters().keys())}")
        
        if self.source:
            print(f"\nSource Model: {self.source.model_type}")
            print(f"  Position: {self.source.position} arcsec")
            print(f"  Redshift: {self.source.redshift}")
            print(f"  Free parameters: {list(self.source.get_free_parameters().keys())}")
            if self.source.get_fixed_parameters():
                print(f"  Fixed parameters: {list(self.source.get_fixed_parameters().keys())}")
        
        if self.lens_light:
            print(f"\nLens Light Model: {self.lens_light.model_type}")
            print(f"  Free parameters: {list(self.lens_light.get_free_parameters().keys())}")
            if self.lens_light.get_fixed_parameters():
                print(f"  Fixed parameters: {list(self.lens_light.get_fixed_parameters().keys())}")
        
        if self.cosmology:
            print(f"\nCosmology:")
            print(f"  H0: {self.cosmology.H0} km/s/Mpc")
            print(f"  Om0: {self.cosmology.Om0}")
            print(f"  Ode0: {self.cosmology.Ode0}")
            print(f"  As free parameters: {self.cosmology.use_as_parameters}")
        
        if self.observation:
            print(f"\nObservation:")
            print(f"  Pixel scale: {self.observation.pixel_scale} arcsec/pixel")
            print(f"  Image size: {self.observation.image_size}")
            print(f"  PSF FWHM: {self.observation.psf_fwhm} arcsec")
            print(f"  Background RMS: {self.observation.background_rms}")
        
        print(f"\nImage Positions ({self.n_images}):")
        for i, (x, y) in enumerate(self.image_positions):
            print(f"  Image {i+1}: ({x:.3f}, {y:.3f}) arcsec")
        
        print(f"\nTotal Parameters: {len(self.get_all_parameter_names())}")
        print(f"Fixed Parameters: {len(self.get_fixed_parameters())}")
        print(f"Free Parameters: {len(self.get_all_parameter_names()) - len(self.get_fixed_parameters())}")
