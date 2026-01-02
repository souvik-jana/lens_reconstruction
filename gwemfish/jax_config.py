"""
JAX configuration utilities for GWEMFISH pipeline.

This module provides functions to configure JAX for optimal performance,
including CPU/GPU settings and thread management.
"""

import os
# Don't import jax at module level - import it inside setup_jax() instead
# This allows setup_jax() to configure JAX before it's initialized


def setup_jax(ncpus=None, enable_x64=True, platform='auto', verbose=True):
    """Setup JAX configuration for optimal performance.
    
    Args:
        ncpus: Number of CPU cores to use. If None, uses all available cores.
               Also sets thread environment variables (OMP, MKL, OPENBLAS, NUMEXPR).
        enable_x64: Enable 64-bit precision (default: True)
        platform: JAX platform to use. Options: 'auto', 'cpu', 'gpu', 'cuda'.
                  If 'auto', will use GPU if available, otherwise CPU.
        verbose: Print configuration information (default: True)
    
    Returns:
        dict: Configuration summary
    """
    # Set thread environment variables
    if ncpus is None:
        ncpus = os.cpu_count()
    
    os.environ['OMP_NUM_THREADS'] = str(ncpus)
    os.environ['MKL_NUM_THREADS'] = str(ncpus)
    os.environ['OPENBLAS_NUM_THREADS'] = str(ncpus)
    os.environ['NUMEXPR_NUM_THREADS'] = str(ncpus)
    
    # CRITICAL: Set ALL JAX environment variables BEFORE importing JAX
    # These must be set before JAX is imported, not just before it's initialized
    
    # Set JAX precision via environment variable
    if enable_x64:
        os.environ['JAX_ENABLE_X64'] = 'True'
    else:
        os.environ['JAX_ENABLE_X64'] = 'False'
    
    # Determine platform without initializing JAX backends
    # For 'auto', we'll default to 'cpu' and let JAX detect later
    if platform == 'auto':
            platform = 'cpu'
    
    os.environ['JAX_PLATFORM_NAME'] = platform
    
    # CRITICAL: Set XLA_FLAGS for CPU device count BEFORE importing JAX
    if platform in ['cpu', 'auto']:
        os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={ncpus}'
    
    # NOW import JAX - environment variables are already set
    import jax
    
    # Check if JAX backends are already initialized BEFORE we do anything
    # We can't directly check this without initializing, but we can try to set
    # a config and see if it fails with the initialization error
    jax_already_initialized = False
    try:
        # Try to access a config that requires backends to be uninitialized
        # If this fails, backends might already be initialized
        _ = jax.config.jax_num_cpu_devices
        # If we can read it, try to see if we can write to it
        # Actually, we can't check this without trying to set it
    except:
        pass
    
    # Update JAX configs (these should work now since env vars are set)
    if enable_x64:
        jax.config.update('jax_enable_x64', True)
    else:
        jax.config.update('jax_enable_x64', False)
    
    # Try to set jax_num_cpu_devices (should work if JAX wasn't initialized before)
    # Note: XLA_FLAGS environment variable is already set, so CPU device count should work
    # even if we can't update this config
    if platform in ['cpu', 'auto']:
        try:
        jax.config.update('jax_num_cpu_devices', ncpus)
        except (RuntimeError, AttributeError):
            # Silently fail - XLA_FLAGS environment variable is already set,
            # so the configuration will still work
            pass
    
    # Update platform name
    try:
        jax.config.update('jax_platform_name', platform)
    except (RuntimeError, AttributeError):
        if verbose:
            print("Warning: JAX platform already initialized, skipping platform name update")
    
    # Now we can safely detect the actual platform (this initializes backends)
    if platform == 'auto' or verbose:
        try:
            devices = jax.devices()
            actual_platform = devices[0].platform if devices else 'cpu'
            if platform == 'auto':
                platform = actual_platform
        except:
            pass
    
    # Print configuration summary
    if verbose:
        print("=" * 60)
        print("JAX Configuration")
        print("=" * 60)
        print(f"Requested CPU cores: {ncpus}")
        print(f"Thread limits: OMP={os.environ['OMP_NUM_THREADS']}, "
              f"MKL={os.environ['MKL_NUM_THREADS']}, "
              f"OPENBLAS={os.environ['OPENBLAS_NUM_THREADS']}")
        print(f"JAX precision: X64={enable_x64}")
        print(f"JAX platform: {platform}")
        print(f"Actual available CPU count: {os.cpu_count()}")
        print(f"JAX device count: {jax.device_count()}")
        print(f"JAX local device count: {jax.local_device_count()}")
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        print("=" * 60)
    
    return {
        'ncpus': ncpus,
        'enable_x64': enable_x64,
        'platform': platform,
        'device_count': jax.device_count(),
        'devices': jax.devices()
    }

