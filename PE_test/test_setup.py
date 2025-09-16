#!/usr/bin/env python3
"""
Test script to verify the setup and dependencies for the PE analysis
"""

import sys
import importlib

def test_imports():
    """Test if all required packages can be imported"""
    
    required_packages = [
        'numpy',
        'scipy', 
        'matplotlib',
        'jax',
        'jax.numpy',
        'optax',
        'numpyro',
        'blackjax',
        'herculens',
        'corner',
        'tqdm'
    ]
    
    print("Testing package imports...")
    print("=" * 50)
    
    failed_imports = []
    
    for package in required_packages:
        try:
            if '.' in package:
                # Handle submodules like jax.numpy
                module_parts = package.split('.')
                module = importlib.import_module(module_parts[0])
                for part in module_parts[1:]:
                    module = getattr(module, part)
            else:
                module = importlib.import_module(package)
            
            print(f"✓ {package}")
            
        except ImportError as e:
            print(f"✗ {package} - {e}")
            failed_imports.append(package)
    
    print("=" * 50)
    
    if failed_imports:
        print(f"Failed to import {len(failed_imports)} packages:")
        for pkg in failed_imports:
            print(f"  - {pkg}")
        print("\nPlease install missing packages with:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("All packages imported successfully!")
        return True

def test_jax_config():
    """Test JAX configuration"""
    print("\nTesting JAX configuration...")
    print("=" * 50)
    
    import jax
    import jax.numpy as jnp
    
    # Test basic JAX functionality
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x)
    
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}")
    print(f"Test computation: sum([1,2,3]) = {y}")
    
    # Test double precision
    if jax.config.jax_enable_x64:
        print("✓ Double precision enabled")
    else:
        print("⚠ Double precision disabled (may cause numerical issues)")
    
    return True

def test_herculens():
    """Test Herculens functionality"""
    print("\nTesting Herculens...")
    print("=" * 50)
    
    try:
        import herculens as hcl
        from herculens.Util import param_util, plot_util
        
        # Test basic functionality
        print(f"Herculens version: {hcl.__version__}")
        
        # Test parameter utilities
        e1, e2 = param_util.phi_q2_ellipticity(0.1, 0.8)
        print(f"Parameter utility test: phi_q2_ellipticity(0.1, 0.8) = ({e1:.3f}, {e2:.3f})")
        
        print("✓ Herculens working correctly")
        return True
        
    except Exception as e:
        print(f"✗ Herculens error: {e}")
        return False

def main():
    """Run all tests"""
    print("GRAVITATIONAL LENSING PE ANALYSIS - SETUP TEST")
    print("=" * 60)
    
    # Test imports
    imports_ok = test_imports()
    
    if not imports_ok:
        print("\n❌ Setup test failed - missing dependencies")
        sys.exit(1)
    
    # Test JAX
    jax_ok = test_jax_config()
    
    # Test Herculens
    herculens_ok = test_herculens()
    
    print("\n" + "=" * 60)
    if imports_ok and jax_ok and herculens_ok:
        print("✅ All tests passed! Ready to run the analysis.")
        print("\nTo run the full analysis:")
        print("  python herculens_pe_analysis.py")
    else:
        print("❌ Some tests failed. Please check the setup.")
        sys.exit(1)

if __name__ == "__main__":
    main()
