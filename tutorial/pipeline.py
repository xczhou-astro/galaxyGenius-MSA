import sys
sys.path.append('..')

from galaxyGeniusMSA.config import Configuration
from galaxyGeniusMSA.generation import DataGeneration
from galaxyGeniusMSA.preprocess import PreProcess
from galaxyGeniusMSA.postprocess import PostProcess
from galaxyGeniusMSA.utils import Units

import os
import astropy.units as u

os.environ['GALAXYGENIUS_DATA_DIR'] = '/home/zhouxingchen/JWST_MSA/galaxyGenius-MSA/Data'
os.environ['STPSF_PATH'] = '/home/zhouxingchen/JWST_MSA/stpsf-data'

config = Configuration()
conf = config.get_config()

conf['includeDust'] = True
conf['simulationMode'] = 'ExtinctionOnly'
conf['includeVelocity'] = True
conf['snapNum'] = 50
conf['snapRedshift'] = 1.0
conf['viewRedshift'] = 1.0
conf['oversample'] = 4

conf['starformingSEDFamily'] = 'MAPPINGS'
conf['minWaveRT'] = 0.1 * u.um
conf['maxWaveRT'] = 2.0 * u.um
conf['numWaveRT'] = 3000

conf['minWaveOutput'] = 0.97 * u.um
conf['maxWaveOutput'] = 1.82 * u.um
conf['numWaveOutput'] = 3000

config.save_config(conf)

units = Units(snapRedshift=conf['snapRedshift'])

preprocess = PreProcess(conf)

subhaloIDs = [0, 1, 70415]

for subhaloID in subhaloIDs:
    
    subhalo_file = os.path.join(
        f'../../snapshot-{conf["snapNum"]}',
        f'subhalo-{subhaloID}/subhalo_{subhaloID}_particles.h5'
    )
    
    subhalo_info_file = os.path.join(
        f'../../snapshot-{conf["snapNum"]}',
        f'subhalo-{subhaloID}/subhalo_{subhaloID}.json'
    )
    
    arguments = {}
    if subhaloID == 0:
        arguments['faceAndEdge'] = False
        arguments['numViews'] = 1
        arguments['randomViews'] = False
        arguments['inclinations'] = [0]
        arguments['azimuths'] = [0]
    
    preprocess.inputSubhaloParticleFile(subhalo_file, subhaloInfoFile=subhalo_info_file)
    preprocess.prepare(arguments=arguments)

    dataGeneration = DataGeneration(conf)
    dataGeneration.runSKIRT()

    dataCube_path = f'dataCubes/Subhalo_{subhaloID}/skirt_view_00_total.fits'

    postprocess = PostProcess(subhaloID)
    postprocess.input_dataCube(dataCube_path)
    postprocess.create_MSA_dataTensor()
    postprocess.illustrate_MSA()
    postprocess.get_truth_properties()