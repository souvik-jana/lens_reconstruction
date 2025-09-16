"""
Supported lens models for GWEMFISH package.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class LensModelInfo:
    """Information about a lens model."""
    name: str
    herculens_name: str
    parameters: List[str]
    required_parameters: List[str]
    optional_parameters: List[str]
    description: str


class SupportedLensModels:
    """
    Registry of supported lens models for GWEMFISH.
    
    This class provides information about all supported lens models
    and their parameter requirements.
    """
    
    # Define supported lens models
    MODELS = {
        'SIE': LensModelInfo(
            name='SIE',
            herculens_name='SIE',
            parameters=['theta_E', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['theta_E'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Singular Isothermal Ellipsoid'
        ),
        'SIS': LensModelInfo(
            name='SIS',
            herculens_name='SIS',
            parameters=['theta_E', 'center_x', 'center_y'],
            required_parameters=['theta_E'],
            optional_parameters=['center_x', 'center_y'],
            description='Singular Isothermal Sphere'
        ),
        'NFW': LensModelInfo(
            name='NFW',
            herculens_name='NFW',
            parameters=['Rs', 'alpha_Rs', 'center_x', 'center_y'],
            required_parameters=['Rs', 'alpha_Rs'],
            optional_parameters=['center_x', 'center_y'],
            description='Navarro-Frenk-White profile'
        ),
        'SHEAR': LensModelInfo(
            name='SHEAR',
            herculens_name='SHEAR',
            parameters=['gamma1', 'gamma2', 'ra_0', 'dec_0'],
            required_parameters=['gamma1', 'gamma2'],
            optional_parameters=['ra_0', 'dec_0'],
            description='External shear'
        ),
        'CONVERGENCE': LensModelInfo(
            name='CONVERGENCE',
            herculens_name='CONVERGENCE',
            parameters=['kappa_ext', 'ra_0', 'dec_0'],
            required_parameters=['kappa_ext'],
            optional_parameters=['ra_0', 'dec_0'],
            description='External convergence'
        ),
        'POINT_MASS': LensModelInfo(
            name='POINT_MASS',
            herculens_name='POINT_MASS',
            parameters=['theta_E', 'center_x', 'center_y'],
            required_parameters=['theta_E'],
            optional_parameters=['center_x', 'center_y'],
            description='Point mass'
        ),
        'SPEMD': LensModelInfo(
            name='SPEMD',
            herculens_name='SPEMD',
            parameters=['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['theta_E', 'gamma'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Singular Power-law Elliptical Mass Distribution'
        ),
        'SPEP': LensModelInfo(
            name='SPEP',
            herculens_name='SPEP',
            parameters=['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['theta_E', 'gamma'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Singular Power-law Elliptical Potential'
        ),
        'CHAMELEON': LensModelInfo(
            name='CHAMELEON',
            herculens_name='CHAMELEON',
            parameters=['theta_E', 'w_c', 'w_t', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['theta_E', 'w_c', 'w_t'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Chameleon profile'
        ),
        'DPL': LensModelInfo(
            name='DPL',
            herculens_name='DPL',
            parameters=['alpha_Rs', 'Rs', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['alpha_Rs', 'Rs'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Dual Pseudo Isothermal Elliptical'
        ),
        'GAUSSIAN': LensModelInfo(
            name='GAUSSIAN',
            herculens_name='GAUSSIAN',
            parameters=['sigma', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['sigma'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Gaussian profile'
        ),
        'HERNQUIST': LensModelInfo(
            name='HERNQUIST',
            herculens_name='HERNQUIST',
            parameters=['sigma0', 'Rs', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['sigma0', 'Rs'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Hernquist profile'
        ),
        'JAFFE': LensModelInfo(
            name='JAFFE',
            herculens_name='JAFFE',
            parameters=['sigma0', 'Rs', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['sigma0', 'Rs'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Jaffe profile'
        ),
        'MULTI_GAUSSIAN': LensModelInfo(
            name='MULTI_GAUSSIAN',
            herculens_name='MULTI_GAUSSIAN',
            parameters=['sigma', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['sigma'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Multi-Gaussian profile'
        ),
        'PJAFFE': LensModelInfo(
            name='PJAFFE',
            herculens_name='PJAFFE',
            parameters=['sigma0', 'Ra', 'Rs', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['sigma0', 'Ra', 'Rs'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Projected Jaffe profile'
        ),
        'PJAFFE_ELLIPSE': LensModelInfo(
            name='PJAFFE_ELLIPSE',
            herculens_name='PJAFFE_ELLIPSE',
            parameters=['sigma0', 'Ra', 'Rs', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['sigma0', 'Ra', 'Rs'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Projected Jaffe Elliptical profile'
        ),
        'POWER_LAW': LensModelInfo(
            name='POWER_LAW',
            herculens_name='POWER_LAW',
            parameters=['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['theta_E', 'gamma'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Power-law profile'
        ),
        'POWER_LAW_ELLIPSE': LensModelInfo(
            name='POWER_LAW_ELLIPSE',
            herculens_name='POWER_LAW_ELLIPSE',
            parameters=['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['theta_E', 'gamma'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Power-law Elliptical profile'
        ),
        'POWER_LAW_ELLIPSE_CORE': LensModelInfo(
            name='POWER_LAW_ELLIPSE_CORE',
            herculens_name='POWER_LAW_ELLIPSE_CORE',
            parameters=['theta_E', 'gamma', 's_core', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['theta_E', 'gamma', 's_core'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Power-law Elliptical Core profile'
        ),
        'SERSIC': LensModelInfo(
            name='SERSIC',
            herculens_name='SERSIC',
            parameters=['k_eff', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['k_eff', 'R_sersic', 'n_sersic'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Sersic profile'
        ),
        'SERSIC_ELLIPSE': LensModelInfo(
            name='SERSIC_ELLIPSE',
            herculens_name='SERSIC_ELLIPSE',
            parameters=['k_eff', 'R_sersic', 'n_sersic', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['k_eff', 'R_sersic', 'n_sersic'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Sersic Elliptical profile'
        ),
        'SERSIC_ELLIPSE_CORE': LensModelInfo(
            name='SERSIC_ELLIPSE_CORE',
            herculens_name='SERSIC_ELLIPSE_CORE',
            parameters=['k_eff', 'R_sersic', 'n_sersic', 's_core', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['k_eff', 'R_sersic', 'n_sersic', 's_core'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Sersic Elliptical Core profile'
        ),
        'SPP': LensModelInfo(
            name='SPP',
            herculens_name='SPP',
            parameters=['theta_E', 'gamma', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['theta_E', 'gamma'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Singular Power-law Potential'
        ),
        'TNFW': LensModelInfo(
            name='TNFW',
            herculens_name='TNFW',
            parameters=['Rs', 'alpha_Rs', 'r_trunc', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['Rs', 'alpha_Rs', 'r_trunc'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Truncated Navarro-Frenk-White profile'
        ),
        'TNFW_ELLIPSE': LensModelInfo(
            name='TNFW_ELLIPSE',
            herculens_name='TNFW_ELLIPSE',
            parameters=['Rs', 'alpha_Rs', 'r_trunc', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['Rs', 'alpha_Rs', 'r_trunc'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Truncated Navarro-Frenk-White Elliptical profile'
        ),
        'TNFW_ELLIPSE_CORE': LensModelInfo(
            name='TNFW_ELLIPSE_CORE',
            herculens_name='TNFW_ELLIPSE_CORE',
            parameters=['Rs', 'alpha_Rs', 'r_trunc', 's_core', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['Rs', 'alpha_Rs', 'r_trunc', 's_core'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Truncated Navarro-Frenk-White Elliptical Core profile'
        ),
        'UNIFORM': LensModelInfo(
            name='UNIFORM',
            herculens_name='UNIFORM',
            parameters=['kappa', 'e1', 'e2', 'center_x', 'center_y'],
            required_parameters=['kappa'],
            optional_parameters=['e1', 'e2', 'center_x', 'center_y'],
            description='Uniform convergence'
        )
    }
    
    @classmethod
    def get_model_info(cls, model_name: str) -> LensModelInfo:
        """
        Get information about a specific lens model.
        
        Parameters
        ----------
        model_name : str
            Name of the lens model
            
        Returns
        -------
        LensModelInfo
            Information about the lens model
        """
        if model_name not in cls.MODELS:
            raise ValueError(f"Unknown lens model: {model_name}. Available models: {list(cls.MODELS.keys())}")
        
        return cls.MODELS[model_name]
    
    @classmethod
    def get_supported_models(cls) -> List[str]:
        """Get list of supported lens model names."""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_parameters(cls, model_name: str) -> List[str]:
        """Get parameters for a specific lens model."""
        model_info = cls.get_model_info(model_name)
        return model_info.parameters
    
    @classmethod
    def get_required_parameters(cls, model_name: str) -> List[str]:
        """Get required parameters for a specific lens model."""
        model_info = cls.get_model_info(model_name)
        return model_info.required_parameters
    
    @classmethod
    def get_optional_parameters(cls, model_name: str) -> List[str]:
        """Get optional parameters for a specific lens model."""
        model_info = cls.get_model_info(model_name)
        return model_info.optional_parameters
    
    @classmethod
    def validate_parameters(cls, model_name: str, parameters: Dict[str, Any]) -> bool:
        """
        Validate parameters for a specific lens model.
        
        Parameters
        ----------
        model_name : str
            Name of the lens model
        parameters : dict
            Parameters to validate
            
        Returns
        -------
        bool
            True if parameters are valid
        """
        model_info = cls.get_model_info(model_name)
        
        # Check that all required parameters are present
        for param in model_info.required_parameters:
            if param not in parameters:
                raise ValueError(f"Required parameter '{param}' missing for {model_name}")
        
        # Check that all provided parameters are valid
        for param in parameters.keys():
            if param not in model_info.parameters:
                raise ValueError(f"Unknown parameter '{param}' for {model_name}. Valid parameters: {model_info.parameters}")
        
        return True
    
    @classmethod
    def print_supported_models(cls) -> None:
        """Print information about all supported models."""
        print("Supported Lens Models for GWEMFISH")
        print("=" * 50)
        
        for name, info in cls.MODELS.items():
            print(f"\n{name}:")
            print(f"  Description: {info.description}")
            print(f"  Herculens name: {info.herculens_name}")
            print(f"  Required parameters: {info.required_parameters}")
            print(f"  Optional parameters: {info.optional_parameters}")
            print(f"  All parameters: {info.parameters}")
    
    @classmethod
    def print_model_info(cls, model_name: str) -> None:
        """Print detailed information about a specific model."""
        info = cls.get_model_info(model_name)
        
        print(f"Lens Model: {info.name}")
        print("=" * 30)
        print(f"Description: {info.description}")
        print(f"Herculens name: {info.herculens_name}")
        print(f"Required parameters: {info.required_parameters}")
        print(f"Optional parameters: {info.optional_parameters}")
        print(f"All parameters: {info.parameters}")
