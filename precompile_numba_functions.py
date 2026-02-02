"""
Precompile all numba functions to initialize and cache them.

This script precompiles all numba JIT-compiled functions in the galaxyGeniusMSA package
to ensure they are cached and ready for use. Running this script will create cache files
in the __pycache__ directory, speeding up subsequent imports and function calls.
"""

import numba
import numpy as np
from numba.typed import Dict
from numba import types

# Import classes and functions
from galaxyGeniusMSA.preprocess import _angular_momentum
from galaxyGeniusMSA.postprocess import (
    _trapezoid_numba,
    _integrate_counts_numba,
    _count_rate_numba,
    _numba_convolve_fft_static,
    integrate_bandpass,
    _add_noise_numba,
    _rebin_fluxes_and_noise_numba,
    _overlap_counts_numba
)
from galaxyGeniusMSA.properties import (
    rotate_coordinates,
    calc_los_velocity,
    calc_rotational_velocity,
    transform,
    binned_statistic_2d_numba
)


def precompile_function(func, *args, func_name=None):
    """Precompile a numba function with given arguments."""
    if func_name is None:
        func_name = func.__name__
    print(f'Precompiling {func_name}...')
    try:
        result = func(*args)
        print(f'  ✓ Successfully precompiled {func_name}')
        return result
    except Exception as e:
        print(f'  ✗ Error precompiling {func_name}: {e}')
        raise


def create_throughput_dict(wavelengths, throughput):
    """Create a numba-compatible typed dictionary for throughput."""
    throughput_dict = Dict.empty(
        key_type=types.unicode_type,
        value_type=types.float32[:]
    )
    throughput_dict['wavelengths'] = wavelengths.astype(np.float32)
    throughput_dict['throughput'] = throughput.astype(np.float32)
    return throughput_dict


if __name__ == '__main__':
    
    print("=" * 60)
    print("Precompiling Numba Functions for galaxyGeniusMSA")
    print("=" * 60)
    print()
    
    # ========================================================================
    # PreProcess functions
    # ========================================================================
    print("PreProcess functions:")
    print("-" * 60)
    
    # _angular_momentum
    dummy_pos = np.ones((10, 3), dtype=np.float32)
    dummy_vel = np.ones((10, 3), dtype=np.float32)
    dummy_mass = np.ones(10, dtype=np.float32)
    precompile_function(_angular_momentum, dummy_pos, dummy_vel, dummy_mass)
    print()
    
    # ========================================================================
    # Properties functions
    # ========================================================================
    print("Properties functions:")
    print("-" * 60)
    
    # rotate_coordinates
    dummy_coords = np.ones((10, 3), dtype=np.float32)
    dummy_inclination = np.float32(45.0)
    dummy_azimuth = np.float32(30.0)
    precompile_function(rotate_coordinates, dummy_coords, dummy_inclination, dummy_azimuth)
    
    # calc_los_velocity
    dummy_velocities = np.ones((10, 3), dtype=np.float32)
    dummy_inclination = np.float32(45.0)
    dummy_azimuth = np.float32(30.0)
    precompile_function(calc_los_velocity, dummy_velocities, dummy_inclination, dummy_azimuth)
    
    # calc_rotational_velocity
    dummy_coords = np.ones((10, 3), dtype=np.float32)
    dummy_velocities = np.ones((10, 3), dtype=np.float32)
    dummy_inclination = np.float32(45.0)
    dummy_azimuth = np.float32(30.0)
    dummy_bulk_velocity = np.zeros(3, dtype=np.float32)
    precompile_function(calc_rotational_velocity, dummy_coords, dummy_velocities, 
                       dummy_inclination, dummy_azimuth, dummy_bulk_velocity)
    
    # transform
    dummy_coords = np.ones((10, 3), dtype=np.float32)
    dummy_rotate = np.float32(30.0)
    dummy_shift_perp = np.float32(1.0)
    dummy_shift_para = np.float32(1.0)
    precompile_function(transform, dummy_coords, dummy_rotate, dummy_shift_perp, dummy_shift_para)
    
    # binned_statistic_2d_numba
    dummy_x = np.random.rand(100).astype(np.float32)
    dummy_y = np.random.rand(100).astype(np.float32)
    dummy_values = np.random.rand(100).astype(np.float32)
    dummy_bins_perp = np.linspace(0, 1, 11, dtype=np.float32)
    dummy_bins_para = np.linspace(0, 1, 11, dtype=np.float32)
    precompile_function(binned_statistic_2d_numba, dummy_x, dummy_y, dummy_values,
                       dummy_bins_perp, dummy_bins_para, 'mean')
    precompile_function(binned_statistic_2d_numba, dummy_x, dummy_y, dummy_values,
                       dummy_bins_perp, dummy_bins_para, 'sum')
    precompile_function(binned_statistic_2d_numba, dummy_x, dummy_y, dummy_values,
                       dummy_bins_perp, dummy_bins_para, 'std')
    precompile_function(binned_statistic_2d_numba, dummy_x, dummy_y, None,
                       dummy_bins_perp, dummy_bins_para, 'count')
    print()
    
    # ========================================================================
    # PostProcess functions
    # ========================================================================
    print("PostProcess functions:")
    print("-" * 60)
    
    # _trapezoid_numba
    dummy_y = np.linspace(0, 10, 10, dtype=np.float32)
    dummy_x = np.linspace(0, 10, 10, dtype=np.float32)
    precompile_function(_trapezoid_numba, dummy_y, dummy_x)
    
    # _integrate_counts_numba
    dummy_values = np.ones(10, dtype=np.float32)
    dummy_wave_bins = np.linspace(1000, 2000, 10, dtype=np.float32)
    dummy_aperture = np.float32(2.5)
    dummy_pixel_scale = np.float32(0.1)
    dummy_wavelengths = np.linspace(1000, 2000, 10, dtype=np.float32)
    dummy_throughput = np.ones(10, dtype=np.float32)
    dummy_throughput_dict = create_throughput_dict(dummy_wavelengths, dummy_throughput)
    dummy_count_constants = np.float32(1e-10)
    precompile_function(_integrate_counts_numba, dummy_values, dummy_wave_bins,
                       dummy_aperture, dummy_pixel_scale, dummy_throughput_dict, dummy_count_constants)
    
    # _count_rate_numba
    dummy_wavelength = np.linspace(1000, 2000, 10, dtype=np.float32)
    dummy_spectrum = np.ones(10, dtype=np.float32)
    dummy_waves_at_pixel = np.linspace(1000, 2000, 11, dtype=np.float32)  # edges, so 11 points
    dummy_throughput_dict = create_throughput_dict(dummy_wavelength, dummy_spectrum)
    dummy_count_constants = np.float32(1e-10)
    dummy_numSpecWaveBins = np.int32(5)
    dummy_aperture = np.float32(2.5)
    dummy_pixel_scale = np.float32(0.1)
    precompile_function(_count_rate_numba, dummy_wavelength, dummy_spectrum, dummy_waves_at_pixel,
                       dummy_throughput_dict, dummy_count_constants, dummy_numSpecWaveBins,
                       dummy_aperture, dummy_pixel_scale)
    
    # _numba_convolve_fft_static
    dummy_cube = np.random.rand(5, 10, 10).astype(np.float32)
    dummy_psf = np.random.rand(5, 5, 5).astype(np.float32)
    precompile_function(_numba_convolve_fft_static, dummy_cube, dummy_psf)
    
    # integrate_bandpass
    dummy_img = np.random.rand(10, 10, 10).astype(np.float32)
    dummy_tran = np.ones(10, dtype=np.float32)
    dummy_wave = np.linspace(1000, 2000, 10, dtype=np.float32)
    precompile_function(integrate_bandpass, dummy_img, dummy_tran, dummy_wave)
    
    # _add_noise_numba
    dummy_counts = np.ones(10, dtype=np.float32) * 100.0
    dummy_bkg_counts = np.ones(10, dtype=np.float32) * 10.0
    dummy_dark_counts = np.ones(10, dtype=np.float32) * 5.0
    dummy_readout = np.ones(10, dtype=np.float32) * 2.0
    dummy_n_exposure = np.int32(3)
    precompile_function(_add_noise_numba, dummy_counts, dummy_bkg_counts, dummy_dark_counts,
                       dummy_readout, dummy_n_exposure)
    
    # _rebin_fluxes_and_noise_numba
    dummy_wavelengths_in = np.linspace(1000, 2000, 20, dtype=np.float32)
    dummy_fluxes_in = np.ones(19, dtype=np.float32)  # one less than wavelengths_in
    dummy_noise_in_squared = np.ones(19, dtype=np.float32) * 0.1
    dummy_sed_in = np.ones(19, dtype=np.float32)
    dummy_wavelengths_out = np.linspace(1000, 2000, 11, dtype=np.float32)  # edges
    precompile_function(_rebin_fluxes_and_noise_numba, dummy_wavelengths_in, dummy_fluxes_in,
                       dummy_noise_in_squared, dummy_sed_in, dummy_wavelengths_out)
    
    # _overlap_counts_numba
    dummy_counts_array = np.random.rand(4, 6, 10).astype(np.float32)
    dummy_noise_counts_array = np.random.rand(4, 6, 10).astype(np.float32) * 0.1
    dummy_sed_counts_array = np.random.rand(4, 6, 10).astype(np.float32)
    dummy_n_pixels_slitlet_parallel = np.int32(2)
    precompile_function(_overlap_counts_numba, dummy_counts_array, dummy_noise_counts_array,
                       dummy_sed_counts_array, dummy_n_pixels_slitlet_parallel)
    
    print()
    print("=" * 60)
    print("All numba functions have been precompiled and cached!")
    print("=" * 60)
