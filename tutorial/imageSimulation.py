"""
This script is used to simulate the image by JWST NIRCam (0.031 arcsec/pixel) of a subhalo using galaxyGenius.
Please be sure that the galaxyGenius is downloaded. 
galaxyGenius repo: https://github.com/xczhou-astro/galaxyGenius

"""

import sys
sys.path.append('../galaxyGenius')

from galaxyGenius.config import Configuration
from galaxyGenius.generation import DataGeneration
from galaxyGenius.preprocess import PreProcess
from galaxyGenius.postprocess import PostProcess
from galaxyGenius.utils import Units

import os
import json
import astropy.units as u

os.environ['GALAXYGENIUS_DATA_DIR'] = '/home/xczhou/nis/xczhou/msa3d/galaxyGenius/Data'

config = Configuration()
config.add_survey('JWST')

conf = config.get_config()

conf['includeDust'] = True
conf['simulationMode'] = 'ExtinctionOnly'
conf['numPackets'] = 1e6
conf['snapNum'] = 50
conf['snapRedshift'] = 1.0
conf['viewRedshift'] = 1.0
conf['boxLengthScale'] = 40
conf['minWavelength'] = 0.09 * u.um
conf['maxWavelength'] = 2.5 * u.um
conf['numWavelengths'] = 500
conf['numThreads'] = 24

conf['faceAndEdge'] = True

conf['filters_JWST'] = ["F070W", "F150W", "F200W"]
conf['pixelScales_JWST'] = 0.031 * u.arcsec
conf['includePSF_JWST'] = False
conf['includeBkg_JWST'] = False
conf['RGBImg_JWST'] = True
conf['RGBFilters_JWST'] = ["F070W", "F150W", "F200W"]

config.save_config(conf)

units = Units(snapRedshift=conf['snapRedshift'])

preprocess = PreProcess(conf)

subhaloIDs = [0, 1, 70415]

for subhaloID in subhaloIDs:

    subhalo_file = os.path.join(f'../data/snapshots-{conf["snapNum"]}', f'subhalo-{subhaloID}/subhalo_{subhaloID}_particles.h5')

    with open(os.path.join(f'../data/snapshots-{conf["snapNum"]}', f'subhalo-{subhaloID}/subhalo_{subhaloID}.json'), 'r') as f:
                subhalo_info = json.load(f)
    
    subhaloInfo = {}
    subhaloInfo['SubhaloID'] = subhalo_info['id']
    subhaloInfo['stellarMass'] = subhalo_info['mass_stars'] * units.mass
    subhaloInfo['halfStellarMassRadius'] = subhalo_info['halfmassrad_stars'] * units.distance

    pos_x = subhalo_info['pos_x']
    pos_y = subhalo_info['pos_y']
    pos_z = subhalo_info['pos_z']
    centerPosition = [pos_x, pos_y, pos_z] * units.position

    preprocess.inputSubhaloParticleFile(subhalo_file, subhaloInfo, centerPosition)
    
    arguments = {}
    
    if subhaloID == 0:
        
        arguments['faceAndEdge'] = False
        arguments['numViews'] = 1
        arguments['randomViews'] = False
        arguments['inclinations'] = [0]
        arguments['azimuths'] = [0]
    
    preprocess.prepare(arguments)
    
    dataGeneration = DataGeneration(conf)
    dataGeneration.runSKIRT()

    postprocess = PostProcess(subhaloID)
    postprocess.runPostprocess()