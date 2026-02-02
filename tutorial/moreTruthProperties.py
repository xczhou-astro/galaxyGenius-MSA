"""

Pseduo code tutorial for incorporating more true properties

"""

import sys
sys.path.append('..')

from galaxyGeniusMSA.utils import Units
from galaxyGeniusMSA.properties import (
    rotate_coordinates, transform, calc_stats, 
    age_interp)
from galaxyGeniusMSA.postprocess import PostProcess

import numpy as np
import h5py
from astropy.cosmology import Planck15
import astropy.units as u

units = Units(snapRedshift=1.0, cosmology=Planck15)

def property_xx(
    particle_file: str, inclination: np.float32, azimuth: np.float32, 
    bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
    transformation: dict, subhalo_info: dict, configs: dict) -> tuple[np.ndarray, str]:
    
    """
    The template function for calculating specific property. 
    Note that the inner operation can be customized, but the parameters should be kept the same.
    The pipeline is as follows:
    1. Read the particle file and obtain the properties;
    2. Rotate the coordinates to the observing direction;
    3. Apply the transformation defined in config;
    4. Calculate the statistics for considered properties (e.g. count, mean, std, sum);
    5. Return the statistics and unit.
    
    - particle_file: 
        The particle file to be used for property. 
        If postprocess.input_particle_file() is not called, the particle file saved in preprocess will be used.  
    - inclination and azimuth:
        The inclination and azimuth used in SKIRT, used to rotate the coordinates of particles to the observing direction.
    - bins_perpendicular and bins_parallel:
        The bins for perpendicular and parallel directions to match the slit stepping and dither. 
        Note that the units are in kpc. 
    - transformation:
        A dict storing rotate angle, shift in perpendicular and parallel directions.
    - subhalo_info:
        A dict storing the subhalo information, including SFR, total stellar, DM mass, etc.
    - configs:
        A dict storing the settings from config.toml and config_MSA.toml.
    """
    
    # if need age information
    fage = age_interp(configs['cosmology']) 
    
    particles = {}
    
    # particle file is in h5 format, 
    # either use the one saved in preprocess or manually input.
    with h5py.File(particle_file, 'r') as file:
        
        # coordinates for stellar particles
        # conversion to np.float32 is necessary for numba function to work directly without recompliation
        particles['Coordinates'] = file['PartType4']['Coordinates'][:].astype(np.float32)
        
        # note that for coordinates, the unit should be kpc
        # if you use downloaded particle file from TNG, the default unit is ckpc/h 
        # units.position is ckpc/h, and need to convert to kpc
        particles['Coordinates'] = (particles['Coordinates'] * units.position).to_value(u.kpc)
        
        # if input_particle_file is not called, you use the particle file saved in preprocess,
        # the unit is already in kpc and saved as file['partType4']['Coordinates].attrs['unit']
        # so you can directly use particles['Coordinates'][:].astype(np.float32), or use
        # unit = file['partType4']['Coordinates'].attrs['unit']
        # particles['Coordinates] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
        
        # consider properties 
        particles['Metallicity'] = file['PartType4']['GFM_metallicity'][:].astype(np.float32)
        # include other properties as wish, be careful with the unit!!!
        
    # rotate to observing direction
    coords = rotate_coordinates(
        coordinates=particles['Coordinates'], 
        inclination=inclination, 
        azimuth=azimuth
    )
    
    # apply transformation defined in config
    # if rotate and shifts in config are 0, these lines can be removed
    coords = transform(coordinates=coords, 
                       rotate=transformation['rotate'], 
                       shiftPerpendicular=transformation['shiftPerpendicular'],
                       shiftParallel=transformation['shiftParallel'])
    
    # calculate the statistics for considered properties
    stats = calc_stats(
        coords=coords, 
        values=None, # None if for "count", otherwise values should be provided
        bins_perpendicular=bins_perpendicular, 
        bins_parallel=bins_parallel, 
        statistic='count') # statistic type, can be 'count', 'mean', 'std', 'sum'
    
    # stats = calc_stats(
    #     coords=coords,
    #     values=particles['Metallicity'],
    #     bins_perpendicular=bins_perpendicular,
    #     bins_parallel=bins_parallel,
    #     statistic='mean'
    # )
    
    unit = '1' # string for unit, if no unit, set to '1'
    
    return stats, unit

postprocess = PostProcess(subhaloID=1234)

dataCube_path = f'dataCubes/Subhalo_1234/skirt_view_00_total.fits'
postprocess.input_dataCube(dataCube_path)
postprocess.create_MSA_dataTensor()
postprocess.illustrate_MSA()

# keep_defaults: if true keep the default properties, otherwise, only use the user-defined property functions
keep_defaults = False

# define the names and corresponding functions for properties to be calculated
properties = ['property_xx']
functions = [property_xx]

# you can use the original particle file downloaded from TNG
# if this method is not used, the particle file saved in preprocess will be used
particle_file = f'../../snapshot-99/subhalo-1234/subhalo_1234_particles.h5'
postprocess.input_particle_file(particle_file)

postprocess.get_truth_properties(
    keep_defaults=keep_defaults, 
    properties=properties, 
    functions=functions)