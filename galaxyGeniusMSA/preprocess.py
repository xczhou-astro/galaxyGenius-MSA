import numpy as np
import h5py
import os
import illustris_python as ill
from astropy.cosmology import Planck15
import astropy.units as u
import astropy.constants as const
import requests
import json
import tempfile
from scipy.interpolate import interp1d
from typing import Union
import sys
import time
import numba
from typing import Union, Optional

from .utils import Units, u2temp, custom_serialier, setup_logging, galaxygenius_data_dir, lookup_table, to_json_safe

@numba.njit(fastmath=True, cache=True, parallel=True)
def _angular_momentum(
    coords: np.ndarray, 
    vels: np.ndarray, 
    masses: np.ndarray) -> tuple[np.float32, np.float32, np.float32]:
        
    Lx = np.float32(0.0)
    Ly = np.float32(0.0)
    Lz = np.float32(0.0)
    
    n = coords.shape[0]
    
    for i in numba.prange(n):
        # Extract components for readability
        rx, ry, rz = coords[i, 0], coords[i, 1], coords[i, 2]
        vx, vy, vz = vels[i, 0], vels[i, 1], vels[i, 2]
        m = masses[i]
        
        # Cross Product (r x v) * m and accumulate
        Lx += (ry * vz - rz * vy) * m
        Ly += (rz * vx - rx * vz) * m
        Lz += (rx * vy - ry * vx) * m
        
    return Lx, Ly, Lz

class PreProcess:
    
    def __init__(self, config: dict):
        """
        The PreProcess class handles the preprocessing of simulation data according to the provided configuration.
        It generates necessary particle files and SKIRT .ski file for the preparation of radiative transfer simulation.
        
        Parameters
        ----------
        config : dict
            The configuration dictionary from the Configuration class in config.py
        """
        self.config = config
        self.config_ori = config.copy() # set aside original config
        self.workingDir = self.config['workingDir']
        os.makedirs(self.workingDir, exist_ok=True)
        self.logger = setup_logging(os.path.join(os.getcwd(), 'galaxyGeniusMSA.log'))
        self.logger.info(f'Initializing PreProcess class.')
        
        self.inputMethod = 'snapshot' # subhaloFile, snapshot, provided
        
        self.__init()
        # self.__precompile_numba()
        
        
    def __fage(self) -> interp1d:
        z = np.linspace(0, 4, 1000)
        t = self.cosmology.age(z).to(u.Myr).value
        fage = interp1d(z, t, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')  # type: ignore[arg-type]
        return fage
        
    def __init(self):
        
        self.snapRedshift = self.config['snapRedshift']
        self.viewRedshift = self.config['viewRedshift']
        
        # to avoid potential error in SKIRT execution
        if np.isclose(self.viewRedshift, 0, atol=0.005):
            self.logger.info('View redshift cannot be 0, setting to 0.005.')
            self.viewRedshift = 0.005
        
        if Units._instance is None:
            # if not defined, use Planck15
            self.cosmology = Planck15
            self.units = Units(cosmology=self.cosmology, 
                                snapRedshift=self.snapRedshift)
        else:
            # if defined, use cosmology from the units
            self.units = Units()
            self.cosmology = self.units.get_cosmology()
        
        self.logger.info(f'Use cosmology {self.cosmology.name} at redshift {self.snapRedshift}')
        
        self.h = self.cosmology.h
        self.a = 1 / (1 + self.snapRedshift)
        
        self.fage = self.__fage()
        
        self.dataDir = galaxygenius_data_dir() 
        self.simulation = self.config['simulation']
        self.snapnum = self.config['snapNum']
    
        self.__init_params()
        self.table, self.table_units = lookup_table(self.units)
        
        if 'TNG' in self.config['simulation']:
            name = 'tng'
        else:
            name = None
            self.logger.warning('Simulation name is unrecognized. Please manually input data by calling inputParticles() or inputs().')
            
        if self.config['useRequests']:
            self.base_url = f"https://www.{name}-project.org/api/"
            self.headers = {'api-key': self.config['apiKey']}
            
    def __make_request_with_retry(self, url: str, headers: Union[dict, None] = None, 
                                params: Union[dict, None] = None, max_retries: int = 5) -> requests.Response:
        
        start_time = time.time()
        content = None
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()  # Raises an HTTPError for bad responses
                content = response
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:  # Last attempt
                    self.logger.info(f"Error: Failed to make request after {max_retries} attempts.")
                    self.logger.info(f"URL: {url}")
                    self.logger.error(f"Error message: {str(e)}")
                else:
                    self.logger.info(f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff        
                    
        end_time = time.time()
        self.logger.info(f"Requesting time taken: {end_time - start_time:.2f} seconds")
        
        if content is None:
            sys.exit(1)
        
        return content
    
    def __read_subhalos(self) -> dict:
        snap_subhalos = ill.groupcat.loadSubhalos(self.config['TNGPath'], self.config['snapNum'])
        return snap_subhalos
    
    def __retrieve_subhalos_with_requests(self) -> list:
        
        cache_dir = os.path.join('.', 'cache')
        os.makedirs(cache_dir, exist_ok=True)
    
        minStellarMass = self.config['minStellarMass'].value / 10**10 * self.h
        maxStellarMass = self.config['maxStellarMass'].value / 10**10 * self.h
        
        snapnum = self.config['snapNum']
        
        if maxStellarMass == np.inf:
            url = f'{self.base_url}{self.simulation}/snapshots/{snapnum}' \
                + f'/subhalos/?mass_stars__gt={minStellarMass}&subhaloflag=1&limit=1000000'
            
            filename = f'subhalos_snap_{snapnum}_mass_gt_{minStellarMass}_1e10_Msun_over_h_subhaloflag1.json'
        else:
            url = f'{self.base_url}{self.simulation}/snapshots/{snapnum}' \
                + f'/subhalos/?mass_stars__gt={minStellarMass}&mass_stars' \
                + f'__lt={maxStellarMass}&subhaloflag=1&limit=1000000'
                
            filename = f'subhalos_snap_{snapnum}_mass_gt_{minStellarMass}' \
                + f'_lt_{maxStellarMass}_1e10_Msun_over_h_subhaloflag1.json'
        
        cache_file = os.path.join(cache_dir, filename)
        
        response = self.__make_request_with_retry(url, headers=self.headers)
        data = response.json()
        results = data.get('results', [])
        
        with open(cache_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        return results
    
    def get_subhalos(self) -> dict:
        
        """
        Retrieve subhalos from TNG snapshot or from requests.

        Returns:
            dict: A dictionary containing basic information about subhalos: subhaloIDs, SFRs.
        """
        
        
        minStellarMass = self.config['minStellarMass'].to(u.Msun).value
        maxStellarMass = self.config['maxStellarMass'].to(u.Msun).value
        
        minStellarMass_in_10 = np.around(np.log10(minStellarMass), 2)
        maxStellarMass_in_10 = np.around(np.log10(maxStellarMass), 2)
        
        subhalos = {}
        
        if not self.config['useRequests']:
            
            snap_subhalos = self.__read_subhalos()
            stellarMass = (snap_subhalos['SubhaloMassType'][:, 4] * self.units.mass).to(u.Msun)
            
            subhalo_indices = np.where((stellarMass.value > minStellarMass) \
                                        & (stellarMass.value < maxStellarMass) \
                                        & (snap_subhalos['SubhaloFlag'] == 1))[0]
            
            self.subhaloIDs = subhalo_indices
            self.subhaloInfos = {'SubhaloID': subhalo_indices.tolist()}
            self.subhaloInfos.update({key: snap_subhalos[key][subhalo_indices] * unit
                                 for key, unit in self.table_units.items()})
            
            # self.subhaloIDs = subhalo_indices # indices are same as subhaloIDs
            # self.stellarMasses = stellarMass[self.subhaloIDs] # with u.Msun
            # halfMassRad = snap_subhalos['SubhaloHalfmassRadType'][:, 4] * self.units.distance
            # self.halfMassRadii = halfMassRad[self.subhaloIDs] # with u.kpc
            # subhaloNum = self.subhaloIDs.shape[0]
            # subhaloPos = snap_subhalos['SubhaloPos'] * self.units.distance
            # self.subhaloPos = subhaloPos[self.subhaloIDs]
            # self.subhaloSFR = snap_subhalos['SubhaloSFR'][self.subhaloIDs] * self.units.sfr
            # self.subhaloVel = snap_subhalos['Velocities'][self.subhaloIDs] * self.units.velocity
            
            subhalos['subhaloNum'] = len(subhalo_indices)
            subhalos['subhaloIDs'] = self.subhaloIDs
            subhalos['subhaloSFR'] = self.subhaloInfos['SubhaloSFR'].to(u.Msun/u.yr)
            
            # subhalos['subhaloNum'] = subhaloNum
            # subhalos['subhaloIDs'] = self.subhaloIDs
            # subhalos['subhaloSFR'] = self.subhaloSFR.to(u.Msun/u.yr)
            
            # subhalos['units'] = ['1', '1', 'Msun/yr']
            
            if maxStellarMass == np.inf:
                print_info = f'{subhaloNum} subhalos in snapshot {self.snapnum} ' \
                    f'in stellar mass higher than 10^{minStellarMass_in_10:.2f} [M_sun]'
            else:
                print_info = f'{subhaloNum} subhalos in snapshot {self.snapnum} ' \
                    f'in stellar mass from 10^{minStellarMass_in_10:.2f} to 10^{maxStellarMass_in_10:.2f} [M_sun]'
            
        else:
            
            self.logger.info('Using Web-based API to retrieve data.')
            
            results = self.__retrieve_subhalos_with_requests()
            self.results = results # to access in following functions
            
            subhaloNum = len(results)
            self.subhaloIDs = [result['id'] for result in results]
            self.subhaloSFR = [result['sfr'] for result in results] * self.units.sfr
            
            subhalos['subhaloNum'] = subhaloNum
            subhalos['subhaloIDs'] = self.subhaloIDs
            subhalos['subhaloSFR'] = self.subhaloSFR.to(u.Msun/u.yr)
            
            # subhalos['units'] = ['1', '1', 'Msun/yr']
            
            if maxStellarMass == np.inf:
                print_info = f'{subhaloNum} subhalos in snapshot {self.snapnum} ' \
                    f'in stellar mass higher than 10^{minStellarMass_in_10:.2f} [M_sun]'
            else:
                print_info = f'{subhaloNum} subhalos in snapshot {self.snapnum} ' \
                    f'in stellar mass from 10^{minStellarMass_in_10:.2f} to 10^{maxStellarMass_in_10:.2f} [M_sun]'
        
        self.logger.info(print_info)
        
        return subhalos
    
    def subhalo(self, subhaloID: int):
        
        '''
        Specify one subhalo to be processed
        
        Args:
            subhaloID: subhaloID of the subhalo to be processed
        '''
        
        if not self.config['useRequests']:
            idx = list(self.subhaloIDs).index(subhaloID)
            mass = self.subhaloInfos['SubhaloMass'][idx]
            mass_in_10 = np.log10(mass.to_value(u.Msun))
            sfr = self.subhaloInfos['SubhaloSFR'].to_value(u.Msun/u.yr)[idx]
            
            subhalo_info = {}
            subhalo_info['SubhaloID'] = subhaloID
            subhalo_info['snapNum'] = self.config['snapNum']
            subhalo_info['snapRedshift'] = self.config['snapRedshift']
            subhalo_info['viewRedshift'] = self.config['viewRedshift']
            
            for key in self.subhaloInfos.keys():
                subhalo_info[key] = self.subhaloInfos[key][idx]
            
            with open(os.path.join(self.workingDir, f'Subhalo_{subhaloID}.json'), 'w') as f:
                json.dump(subhalo_info, f, indent=4, default=custom_serialier)
            
            # make it accessible in following functions
            self.subhalo_info = subhalo_info
            
            # self.mass = self.stellarMasses[idx].to(u.Msun)
            # self.radius = self.halfMassRadii[idx].to(u.kpc)
            # self.pos = self.subhaloPos[idx].to(u.kpc)
            # self.sfr = self.subhaloSFR[idx].to(u.Msun/u.yr)
            # self.vel = self.subhaloVel[idx].to(u.km/u.s)

            # mass_in_10 = np.log10(self.mass.to(u.Msun).value)
            
            self.logger.info(f'Stellar Mass of Subhalo {subhaloID} is 10^{mass_in_10:.2f} [M_sun].')
            self.logger.info(f'SFR of Subhalo {subhaloID} is {sfr:.2f} [M_sun/yr].')
            
        else:
            subhaloIDs = [result['id'] for result in self.results]
            idx = list(subhaloIDs).index(subhaloID)
            subhalo = self.results[idx]
            subhalo_url = subhalo['url']
            self.logger.info(f'Subhalo URL: {subhalo_url}')
            subhalo_response = self.__make_request_with_retry(subhalo_url, headers=self.headers)
            self.data = subhalo_response.json()
            
            subhalo_info = {}
            subhalo_info['SubhaloID'] = subhaloID
            subhalo_info['snapNum'] = self.config['snapNum']
            subhalo_info['snapRedshift'] = self.config['snapRedshift']
            subhalo_info['viewRedshift'] = self.config['viewRedshift']
            
            for key in self.table.keys():
                if isinstance(self.table[key], list):
                    subhalo_info[key] = []
                    for info_key in self.table[key]:
                        if info_key is not None:
                            subhalo_info[key].append(self.data[info_key])
                        else:
                            subhalo_info[key].append(0)
                    try:
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(self.table_units[key].unit)
                    except:
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(u.Unit(self.table_units[key]))
                    
                else:
                    info_key = self.table[key]
                    try:
                        subhalo_info[key] = u.Quantity(self.data[info_key], self.table_units[key]).to(self.table_units[key].unit)
                    except:
                        subhalo_info[key] = u.Quantity(self.data[info_key], self.table_units[key]).to(u.Unit(self.table_units[key]))

            with open(os.path.join(self.workingDir, f'Subhalo_{subhaloID}.json'), 'w') as f:
                json.dump(subhalo_info, f, indent=4, default=custom_serialier)
                
            self.subhalo_info = subhalo_info
                
            sfr = subhalo_info['SubhaloSFR'].to(u.Msun/u.yr)
            
            # self.mass = (self.data['mass_stars'] * self.units.mass).to(u.Msun)
            # self.radius = (self.data['halfmassrad_stars'] * self.units.distance).to(u.kpc)
            # self.pos_x = (self.data['pos_x'] * self.units.distance).to(u.kpc)
            # self.pos_y = (self.data['pos_y'] * self.units.distance).to(u.kpc)
            # self.pos_z = (self.data['pos_z'] * self.units.distance).to(u.kpc)
            # self.pos = u.Quantity([self.pos_x, self.pos_y, self.pos_z]).to(u.kpc)
            # self.sfr = (self.data['sfr'] * self.units.sfr).to(u.Msun/u.yr)
            # self.vel_x = (self.data['vel_x'] * self.units.velocity).to(u.km/u.s)
            # self.vel_y = (self.data['vel_y'] * self.units.velocity).to(u.km/u.s)
            # self.vel_z = (self.data['vel_z'] * self.units.velocity).to(u.km/u.s)
            # self.vel = u.Quantity([self.vel_x, self.vel_y, self.vel_z]).to(u.km/u.s)
            
            mass = subhalo_info['SubhaloMassType'][4].to(u.Msun)
            mass_in_10 = np.log10(mass.value)
            
            self.logger.info(f'Stellar Mass of Subhalo {subhaloID} is 10^{mass_in_10:.2f} [M_sun].')
            self.logger.info(f'SFR of Subhalo {subhaloID} is {sfr:.2f} [M_sun/yr].')

        radius = subhalo_info['SubhaloHalfmassRadType'][4].to(u.kpc)
        self.partRegion = (self.config['boxLengthScale'] * radius).to(u.kpc) # boxlength, aperture
        self.partRegion = np.min([self.partRegion.value, self.config['maxBoxLength'].value])
        self.partRegion = self.partRegion * u.kpc
        
        return subhalo_info

    
    def __calculate_angular_momentum_and_angles(self) -> tuple:
        
        self.logger.info('------Calculating face-on and edge-on viewing angles------')
        
        if self.inputMethod == 'input':
            particle_file = os.path.join(self.config['workingDir'], 
                                         'stars.txt')
            
            headers = []
            with open(particle_file, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        if 'column' in line.lower():
                            headers.append(line.strip().lower())
                    else:
                        break
                    
            idx_coordinates = [i for i, header in enumerate(headers) if 'coordinate' in header]
            idx_velocities = [i for i, header in enumerate(headers) if 'velocity' in header]
            idx_masses = [i for i, header in enumerate(headers) if 'mass' in header and 'initial' not in header]
            
            idx_columns = idx_coordinates + idx_velocities + idx_masses
            
            # print(idx_columns)
            
            particles = np.loadtxt(particle_file, usecols=idx_columns, dtype=np.float32)
            positions = particles[:, :3]
            velocities = particles[:, 3:6]
            masses = particles[:, 6]
        else:
            positions = self.starPart['Coordinates'].to(u.kpc).value
            velocities = self.starPart['Velocities'].to(u.km/u.s).value
            masses = self.starPart['Masses'].to(u.Msun).value
        
        
        mask = np.where((np.abs(positions[:, 0]) < 20) \
                    & (np.abs(positions[:, 1]) < 20) \
                    & (np.abs(positions[:, 2]) < 20))[0]
        
        positions = positions[mask]
        velocities = velocities[mask]
        masses = masses[mask]

        positions = positions.astype(np.float32)
        velocities = velocities.astype(np.float32)
        masses = masses.astype(np.float32)

        Lx, Ly, Lz = _angular_momentum(positions, velocities, masses)
        L_norm = np.sqrt(Lx**2 + Ly**2 + Lz**2)
        
        Lx /= L_norm
        Ly /= L_norm
        Lz /= L_norm
        
        inc_rad = np.arccos(Lz)
        azi_rad = np.arctan2(Ly, Lx)
        
        face_inc = np.degrees(inc_rad)
        face_azi = np.degrees(azi_rad)
        
        edge_inc_raw = face_inc + 90.0
        edge_azi_raw = face_azi
        
        if edge_inc_raw > 180.0:
            edge_inc = 360.0 - edge_inc_raw
            edge_azi = edge_azi_raw + 180.0
        else:
            edge_inc = edge_inc_raw
            edge_azi = edge_azi_raw
        
        # edge_azi = (edge_azi + 180) % 360 - 180
        
        incs = [face_inc, edge_inc]
        azis = [face_azi, edge_azi]
            
        self.logger.info(f'Face-on angle (inc, azi): {incs[0]:.2f} deg, {azis[0]:.2f} deg')
        self.logger.info(f'Edge-on angle (inc, azi): {incs[1]:.2f} deg, {azis[1]:.2f} deg')
        
        return incs, azis
    
    def __read_temp_file(self, response_content, partType='star', params=None):

        try:
            with tempfile.NamedTemporaryFile(suffix='.hdf5', dir=self.workingDir, delete=True) as tmp:
                tmp.write(response_content)
                tmp.flush()
                
                with h5py.File(tmp.name, 'r') as f:
                    
                    if partType == 'star':
                        particle_data = {key: f[f'PartType4/{key}'][:] for key in params}
                    elif partType == 'gas':
                        particle_data = {key: f[f'PartType0/{key}'][:] for key in params}
                    
            particle_data['count'] = len(particle_data['Coordinates'])
                    
        except Exception as e:
            raise Exception(f'Error reading temporary file: {e}')
            
        return particle_data
    
    def __init_params(self):
        
        self.starParams = ['GFM_InitialMass', 'GFM_Metallicity', 'GFM_StellarFormationTime',
                    'Coordinates', 'Velocities', 'Masses', 'StellarHsml']
        self.starParamsUnits = [self.units.mass, self.units.dimless, self.units.dimless, 
                                self.units.distance, self.units.velocity, self.units.mass, self.units.distance]

        
        if self.config['includeDust']:
            self.gasParams = ['GFM_Metallicity', 'Coordinates', 'Masses', 
                      'InternalEnergy', 'StarFormationRate', 'ElectronAbundance', 
                      'Velocities', 'Density']
            self.gasParamsUnits = [self.units.dimless, self.units.distance, self.units.mass, 
                                   self.units.energy, self.units.sfr, self.units.dimless, 
                                   self.units.velocity, self.units.density]
            
    def __align_param_units(self, params, paramsUnits):
        
        seen = set()
        aligned_params = []
        aligned_paramsUnits = []
        for param, unit in zip(params, paramsUnits):
            if param not in seen:
                aligned_params.append(param)
                aligned_paramsUnits.append(u.Unit(unit)) # convert to astropy.unit.Unit
                seen.add(param)
                
        return aligned_params, aligned_paramsUnits
    
    def get_default_params(self):
        
        """
        Returns default parameters and their units for stars and gas (if includeDust is True).

        Parameters
        ----------
        None

        Returns
        -------
        returns : tuple
            Returns a tuple, where the first element is a dictionary of star parameters and their units,
            and the second element is a dictionary of gas parameters and their units (if includeDust is True).
            If includeDust is False, then the tuple contains only the star parameters and their units.
            The format of the dictionaries is {parameter: unit}.
        """
        
        
        starParamsUnits = [str(paramUnit) for paramUnit in self.starParamsUnits]
        starParamsUnit_dict = dict(zip(self.starParams, starParamsUnits))
        if self.config['includeDust']:
            gasParamsUnits = [str(paramUnit) for paramUnit in self.gasParamsUnits]             
            gasParamsUnit_dict = dict(zip(self.gasParams, gasParamsUnits))
            returns = (starParamsUnit_dict, gasParamsUnit_dict)
        else:
            returns = (starParamsUnit_dict)
        
        return returns
    
    def includeParams(self, starParams: Optional[list | dict] = None,
                          gasParams: Optional[list | dict] = None, **kwargs):
        """
        Adds parameters to the list of parameters to include in the data.

        Parameters
        ----------
        starParams : list or dict, optional
            List of parameters to include for stars. If a list is given, it should contain the
            parameter names as strings, and the units are deduced from the parameter names. 
            If a dict is given, it should be in the format {parameter: unit}. 
        gasParams : list or dict, optional
            List of parameters to include for gas particles. If a list is given, it should contain
            the parameter names as strings, and the units are deduced from the parameter names. 
            If a dict is given, it should be in the format {parameter: unit}. 
        **kwargs
            Additional keyword arguments to include other parameters for other particle types.

        Returns
        -------
        returns : tuple
            Returns a tuple, where the first element is a dictionary of star parameters and their units,
            and the second element is a dictionary of gas parameters and their units (if includeDust is True).
            If includeDust is False, then the tuple contains only the star parameters and their units.
            The format of the dictionaries is {parameter: unit}.
        """

        if starParams is not None:
            
            if isinstance(starParams, list):
                
                self.starParams = starParams
                self.starParamsUnits = [self.__deduce_unit(param) for param in starParams]
                
            if isinstance(starParams, dict):
                
                self.starParams = list(starParams.keys())
                self.starParamsUnits = list(starParams.values())
            
            self.starParams, self.starParamsUnits =\
                self.__align_param_units(self.starParams, self.starParamsUnits)
                
            starParamsUnits = [str(paramUnit) for paramUnit in self.starParamsUnits]
            starParamsUnit_dict = dict(zip(self.starParams, starParamsUnits))
            returns = (starParamsUnit_dict)
            
            self.logger.info(','.join(self.starParams) + ' for star particles included.')
        
        if self.config['includeDust']:
            if gasParams is not None:
                
                if isinstance(gasParams, list):
                    
                    self.gasParams = gasParams
                    self.gasParamsUnits = [self.__deduce_unit(param) for param in gasParams]
                    
                elif isinstance(gasParams, dict):
                    self.gasParams = list(gasParams.keys())
                    self.gasParamsUnits = list(gasParams.values())
            
                self.gasParams, self.gasParamsUnits =\
                    self.__align_param_units(self.gasParams, self.gasParamsUnits)
                    
                gasParamsUnits = [str(paramUnit) for paramUnit in self.gasParamsUnits]             
                gasParamsUnit_dict = dict(zip(self.gasParams, gasParamsUnits))
                returns = (starParamsUnit_dict, gasParamsUnit_dict)

            self.logger.info(','.join(self.gasParams) + ' for gas particles included.')
            
        return returns
    
    def __add_unit(self, parts, units):
        
        for i, param in enumerate(parts.keys()):
            
            parts[param] = np.array(parts[param]) * units[i]
            
        return parts

    def inputSubhaloParticleFile(self, subhaloParticleFile: str, 
                                 subhaloInfo: Union[dict, None]=None,
                                 subhaloInfoFile: Union[str, None]=None):
        """
        Load a subhalo particle file in HDF5 format and associated subhalo metadata.

        This function initializes the subhalo context for further processing by ingesting particle data
        provided in `subhaloParticleFile`, and reading or constructing subhalo properties from either a
        dictionary (`subhaloInfo`) or a JSON info file (`subhaloInfoFile`).

        Args:
            subhaloParticleFile (str): 
                Path to the subhalo particle file in HDF5 (.h5) format.
            subhaloInfo (dict, optional): 
                Dictionary with subhalo properties ('id', 'halfmassrad_stars', 'pos_x', 'pos_y', 'pos_z' are required, other parameters are optional).
                Required if `subhaloInfoFile` is not provided.
            subhaloInfoFile (str, optional): 
                Path to a JSON file containing the subhalo information.
                Have higher priority than `subhaloInfo`.

        Raises:
            FileNotFoundError: If the specified files do not exist.
            ValueError: If neither `subhaloInfo` nor `subhaloInfoFile` is provided.
        """
        
        os.makedirs(self.workingDir, exist_ok=True)
        
        self.inputMethod = 'subhaloFile'
        
        self.logger.info('------Inputting subhalo particle file------')
        
        if not os.path.exists(subhaloParticleFile):
            raise FileNotFoundError(f'Subhalo particle file {subhaloParticleFile} does not exist.')

        if subhaloInfo is None and subhaloInfoFile is None:
            raise ValueError('Either subhaloInfo or subhaloInfoFile must be provided.')
        
        if subhaloInfoFile is not None:
            if not os.path.exists(subhaloInfoFile):
                raise FileNotFoundError(f'Subhalo info file {subhaloInfoFile} does not exist.')

            with open(subhaloInfoFile, 'r') as f:
                subhaloInfo = json.load(f)
        
            subhalo_info = {}
            subhalo_info['SubhaloID'] = subhaloInfo['id']
            subhalo_info['snapNum'] = self.config['snapNum']
            subhalo_info['snapRedshift'] = self.config['snapRedshift']
            subhalo_info['viewRedshift'] = self.config['viewRedshift']
            
            for key in self.table.keys():
                if isinstance(self.table[key], list):
                    subhalo_info[key] = []
                    for info_key in self.table[key]:
                        if info_key is not None:
                            subhalo_info[key].append(subhaloInfo[info_key])
                        else:
                            subhalo_info[key].append(np.nan)
                        
                    if isinstance(self.table_units[key], u.CompositeUnit):
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(u.Unit(self.table_units[key]))
                    else:
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(self.table_units[key].unit)
                    
                else:
                    info_key = self.table[key]
    
                    if isinstance(self.table_units[key], u.CompositeUnit):
                        subhalo_info[key] = u.Quantity(subhaloInfo[info_key], self.table_units[key]).to(u.Unit(self.table_units[key]))
                    else:
                        subhalo_info[key] = u.Quantity(subhaloInfo[info_key], self.table_units[key]).to(self.table_units[key].unit)
                    
            
        elif subhaloInfo is not None and subhaloInfoFile is None:
            
            required_params = ['id', 'halfmassrad_stars', 
                               'pos_x', 'pos_y', 'pos_z']
            
            for par in required_params:
                if par not in subhaloInfo.keys():
                    raise KeyError(f'Key {par} not found in subhalo info.')
            
            subhalo_info = {}
            subhalo_info['SubhaloID'] = subhaloInfo['id']
            subhalo_info['snapNum'] = self.config['snapNum']
            subhalo_info['snapRedshift'] = self.config['snapRedshift']
            subhalo_info['viewRedshift'] = self.config['viewRedshift']
            
            for key in self.table.keys():
                unit = self.table_units[key]
                if isinstance(self.table[key], list):
                    subhalo_info[key] = []
                    for info_key in self.table[key]:
                        if info_key is not None:
                            if info_key in subhaloInfo.keys():
                                subhalo_info[key].append(u.Quantity(subhaloInfo[info_key]))
                            else:
                                subhalo_info[key].append(u.Quantity(np.nan, unit))
                        else:
                            subhalo_info[key].append(u.Quantity(np.nan, unit))
                    
                else:
                    if key in subhaloInfo.keys():
                        subhalo_info[key] = u.Quantity(subhaloInfo[key])
                    else:
                        subhalo_info[key] = u.Quantity(np.nan, unit)
            
            for key in subhalo_info.keys():
                if key != 'SubhaloID' and key != 'snapNum' and key != 'snapRedshift' and key != 'viewRedshift':
                    
                    if isinstance(self.table_units[key], u.CompositeUnit):
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(u.Unit(self.table_units[key]))
                    else:
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(self.table_units[key].unit)
                
        with open(os.path.join(self.workingDir, f'Subhalo_{subhalo_info["SubhaloID"]}.json'), 'w') as f:
            json.dump(to_json_safe(subhalo_info), f, indent=4)
        
        self.logger.info(f'Processing subhalo {subhalo_info["SubhaloID"]}...')
        
        self.subhalo_info = subhalo_info
            
        centerPosition = subhalo_info['SubhaloPos'].to(u.kpc)
        radius = subhalo_info['SubhaloHalfmassRadType'][4].to(u.kpc)
        
        # self.id = subhaloInfo['SubhaloID']
        # self.mass = (subhaloInfo['stellarMass']).to(u.Msun)
        # self.radius = (subhaloInfo['halfStellarMassRadius']).to(u.kpc)
        # centerPosition = (subhaloInfo['centerPosition']).to(u.kpc)
        # self.vel = (subhaloInfo['velocity']).to(u.km/u.s)

        self.partRegion = (self.config['boxLengthScale'] * radius).to(u.kpc)
        self.partRegion = np.min([self.partRegion.value, self.config['maxBoxLength'].to(u.kpc).value])
        self.partRegion = self.partRegion * u.kpc
        
        with h5py.File(subhaloParticleFile, 'r') as file:
            for par in self.starParams:
                if f'PartType4/{par}' not in file:
                    raise KeyError(f'Key PartType4/{par} not found in subhalo particle file.')
                
            self.starPart = {key: file[f'PartType4/{key}'][:] for key in self.starParams}
            self.starPart = self.__add_unit(self.starPart, self.starParamsUnits)
            
            if self.config['includeDust']:
                for par in self.gasParams:
                    if f'PartType0/{par}' not in file:
                        raise KeyError(f'Key PartType0/{par} not found in subhalo particle file.')
                    
                self.gasPart = {key: file[f'PartType0/{key}'][:] for key in self.gasParams}
                self.gasPart = self.__add_unit(self.gasPart, self.gasParamsUnits)
                
            if hasattr(self, 'dmParams'):
                for par in self.dmParams:
                    if f'PartType1/{par}' not in file:
                        raise KeyError(f'Key PartType1/{par} not found in subhalo particle file.')
                    
                self.dmPart = {key: file[f'PartType1/{key}'][:] for key in self.dmParams}
                self.dmPart = self.__add_unit(self.dmPart, self.dmParamsUnits)
                
        if hasattr(self, 'bhParams'):
            for par in self.bhParams:
                if f'PartType5/{par}' not in file:
                    raise KeyError(f'Key PartType5/{par} not found in subhalo particle file.')
                    
                self.bhPart = {key: file[f'PartType5/{key}'][:] for key in self.bhParams}
                self.bhPart = self.__add_unit(self.bhPart, self.bhParamsUnits)
                
        self.starPart = self.__in_box_mask(self.starPart, centerPosition)
            
        if self.config['includeDust']:
            self.gasPart = self.__in_box_mask(self.gasPart, centerPosition)
            
        if hasattr(self, 'dmParams'):
            self.dmPart = self.__in_box_mask(self.dmPart, centerPosition)
            
        if hasattr(self, 'bhParams'):
            self.bhPart = self.__in_box_mask(self.bhPart, centerPosition)
        
    def __retrieveParts(self, partType, params):
        
        if not self.config['requests']:
            Part = ill.snapshot.loadSubhalo(
                self.config['TNGPath'], self.config['snapNum'],
                self.id, partType, params
            )
        else:
            cutout_url = f"{self.base_url}{self.simulation}/snapshots/{self.config['snapNum']}/subhalos/{self.id}/cutout.hdf5"
            self.logger.info(f'Retrieving {partType} particles.')
            read_params = params.copy()
            if partType == 'star':
                key = 'stars'
            elif partType == 'gas':
                key = 'gas'
            params = {f'{key}': ','.join(params)}
            
            response = self.__make_request_with_retry(cutout_url, headers=self.headers, params=params)
            
            Part = self.__read_temp_file(response.content, partType=partType, params=read_params)

        if 'count' in Part.keys():
            Part.pop('count')
        
        return Part
    
    def __get_particles(self):
        
        self.logger.info('Retrieving particles.')
        # if not self.input_subhalo_filename or not hasattr(self, 'input_subhalo_filename'):
        
        if self.inputMethod == 'snapshot':
            
            self.starPart = self.__retrieveParts('star', self.starParams)
            self.starPart = self.__add_unit(self.starPart, self.starParamsUnits)
            self.starPart = self.__in_box_mask(self.starPart, self.pos)
            
            if self.config['includeDust']:
                self.gasPart = self.__retrieveParts('gas', self.gasParams)
                self.gasPart = self.__add_unit(self.gasPart, self.gasParamsUnits)
                self.gasPart = self.__in_box_mask(self.gasPart, self.pos)
                
            if hasattr(self, 'dmParams'):
                self.dmPart = self.__retrieveParts('dm', self.dmParams)
                self.dmPart = self.__add_unit(self.dmPart, self.dmParamsUnits)
                self.dmPart = self.__in_box_mask(self.dmPart, self.pos)
                
            if hasattr(self, 'bhParams'):
                self.bhPart = self.__retrieveParts('bh', self.bhParams)
                self.bhPart = self.__add_unit(self.bhPart, self.bhParamsUnits)
                self.bhPart = self.__in_box_mask(self.bhPart, self.pos)
                
    # def __get_smoothingLength(self):
        
    #     # from https://www.tng-project.org/about/
    #     if 'TNG50' in self.simulation:
    #         tng_smoothLength = 288 # in pc
    #     elif 'TNG100' in self.simulation:
    #         tng_smoothLength = 740 # in pc
    #     elif 'TNG300' in self.simulation:
    #         tng_smoothLength = 1480 # in pc
            
    #     # from https://www.tng-project.org/data/forum/topic/265/spatial-resolution-softening-length-mum-length/
    #     if self.snapRedshift <= 1:
    #         smoothLength = (tng_smoothLength * 10**-3 / self.a * self.h * self.units.distance).to(u.kpc)
    #         # smoothLength = (tng_smoothLength / self.a * self.h).to(u.kpc)
    #     else:
    #         smoothLength = (tng_smoothLength * 10**-3 / 0.5 * self.h * self.units.distance).to(u.kpc)
    #         # smoothLength = (tng_smoothLength / 0.5 * self.h).to(u.kpc)
        
    #     return smoothLength
    
    def __in_box_mask(self, particles, centerPosition=None):
        
        if centerPosition is not None:
            particles['Coordinates'] = (particles['Coordinates'].to(u.kpc) - centerPosition.to(u.kpc)).to(u.kpc)
        
        mask = np.where((np.abs(particles['Coordinates'][:, 0]) < self.partRegion / 2)\
                        & (np.abs(particles['Coordinates'][:, 1]) < self.partRegion / 2)\
                        & (np.abs(particles['Coordinates'][:, 2]) < self.partRegion / 2))[0]
        
        if 'count' in particles.keys():
            particles.pop('count')
        
        for key in particles.keys():
            particles[key] = particles[key][mask]
        
        return particles
    
    def __processing_particles(self):
        
        def starFunction(particles):
            
            # smoothLength = self.__get_smoothingLength() # with kpc
            
            starFormationMask = np.where(particles['GFM_StellarFormationTime'] > 0)[0]
            snapshotAge = self.fage(self.snapRedshift) * u.Myr
            particles['age'] = snapshotAge - self.fage(1/particles['GFM_StellarFormationTime'] - 1) * u.Myr
            ageMask = np.where(particles['age'] > self.config['ageThreshold'])[0]
            
            idx = np.intersect1d(starFormationMask, ageMask)
            
            self.logger.info(f'Star particles: {len(idx)}')
            
            if len(idx) == 0:
                self.logger.info('No star particles found.')
                return {}
            
            properties = {}
            properties['x-coordinate'] = particles['Coordinates'][:, 0][idx]
            properties['y-coordinate'] = particles['Coordinates'][:, 1][idx]
            properties['z-coordinate'] = particles['Coordinates'][:, 2][idx]
            properties['smoothing length'] = particles['StellarHsml'][idx]
            properties['initial mass'] = particles['GFM_InitialMass'][idx]
            properties['metallicity'] = particles['GFM_Metallicity'][idx]
            properties['age'] = particles['age'][idx]
            properties['x-velocity'] = particles['Velocities'][:, 0][idx]
            properties['y-velocity'] = particles['Velocities'][:, 1][idx]
            properties['z-velocity'] = particles['Velocities'][:, 2][idx]
            properties['mass'] = particles['Masses'][idx]
            
            return properties
        
        if self.config['includeVelocity']:
        
            paramNames = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'smoothing length', 
                        'x-velocity', 'y-velocity', 'z-velocity',
                        'initial mass', 'metallicity', 'age', 'mass']
        else:
            
            paramNames = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'smoothing length', 
                        'initial mass', 'metallicity', 'age', 'mass', 
                        'x-velocity', 'y-velocity', 'z-velocity'] # for momentum calculation
        
        self.createFile(paramNames, 'star', 
                        os.path.join(self.workingDir, 'stars.txt'), 
                        starFunction)
        
        def starFormingFunction(particles):
            
            # smoothLength = self.__get_smoothingLength()
            
            starFormationMask = np.where(particles['GFM_StellarFormationTime'] > 0)[0]
            snapshotAge = self.fage(self.snapRedshift) * u.Myr
            particles['age'] = snapshotAge - self.fage(1/particles['GFM_StellarFormationTime'] - 1) * u.Myr
            ageMask = np.where(particles['age'] < self.config['ageThreshold'])[0]
            
            idx = np.intersect1d(starFormationMask, ageMask)
            
            self.logger.info(f'Star forming regions: {len(idx)}')
            
            if len(idx) == 0:
                self.logger.info('No star forming regions found.')
                return {}
            
            properties = {}
            properties['x-coordinate'] = particles['Coordinates'][:, 0][idx]
            properties['y-coordinate'] = particles['Coordinates'][:, 1][idx]
            properties['z-coordinate'] = particles['Coordinates'][:, 2][idx]
            properties['smoothing length'] = particles['StellarHsml'][idx]
            properties['star formation rate'] = (particles['GFM_InitialMass'][idx] / self.config['ageThreshold']).to(u.Msun / u.yr)
            properties['metallicity'] = particles['GFM_Metallicity'][idx]
            # from Kapoor et al. 2021
            
            # random_seed = 42
            # print(random_seed)
            # np.random.seed(random_seed)
            properties['compactness'] = np.random.normal(loc=self.config['logCompactnessMean'], 
                                                        scale=self.config['logCompactnessStd'], size=idx.shape[0]) * u.dimensionless_unscaled
            
            pressure = (10**self.config['logPressure'] * const.k_B * u.K * u.cm**-3).to(u.J / u.m**3) # J / m**3 == Pa
            properties['pressure'] = np.full(idx.shape[0], pressure.value) * pressure.unit
            if self.config['constantCoveringFactor']:
                properties['covering factor'] = np.full(idx.shape[0], self.config['coveringFactor']) * u.dimensionless_unscaled
            else:
                # from Baes, M., et al. 2024
                properties['covering factor'] = np.exp(-particles['age'][idx] / self.config['PDRClearingTimescale']) * u.dimensionless_unscaled

            properties['x-velocity'] = particles['Velocities'][:, 0][idx] # in km/s
            properties['y-velocity'] = particles['Velocities'][:, 1][idx] # in km/s
            properties['z-velocity'] = particles['Velocities'][:, 2][idx] # in km/s
            properties['mass'] = particles['Masses'][idx] # in Msun
            
            return properties
        
        
        def starFormingFunctionTODDLERS(particles):
            
            # https://skirt.ugent.be/skirt9/class_toddlers_s_e_d_family.html
            
            # smoothLength = self.__get_smoothingLength()
            starFormationMask = np.where(particles['GFM_StellarFormationTime'] > 0)[0]
            snapshotAge = self.fage(self.snapRedshift) * u.Myr
            particles['age'] = snapshotAge - self.fage(1/particles['GFM_StellarFormationTime'] - 1) * u.Myr
            ageMask = np.where(particles['age'] < self.config['ageThreshold'])[0]
            
            idx = np.intersect1d(starFormationMask, ageMask)
            
            self.logger.info(f'Star forming regions: {len(idx)}')
            
            if len(idx) == 0:
                self.logger.info('No star forming regions found.')
                return {}
            
            properties = {}
            properties['x-coordinate'] = particles['Coordinates'][:, 0][idx]
            properties['y-coordinate'] = particles['Coordinates'][:, 1][idx]
            properties['z-coordinate'] = particles['Coordinates'][:, 2][idx]
            properties['x-velocity'] = particles['Velocities'][:, 0][idx] # in km/s
            properties['y-velocity'] = particles['Velocities'][:, 1][idx] # in km/s
            properties['z-velocity'] = particles['Velocities'][:, 2][idx] # in km/s
            
            properties['smoothing length'] = particles['StellarHsml'][idx]
            properties['metallicity'] = particles['GFM_Metallicity'][idx]
            
            starFormationEfficiency = np.full(idx.shape[0], self.config['starFormationEfficiency']) * u.dimensionless_unscaled
            properties['star formation efficiency'] = starFormationEfficiency

            cloudNumberDensity = self.config['cloudNumberDensity']
            cloudNumberDensity = np.full(idx.shape[0], cloudNumberDensity.value) * cloudNumberDensity.unit
            properties['cloud number density'] = cloudNumberDensity
            
            if self.config['sedMode'] == 'SFRNormalized':
                properties['star formation rate'] = (particles['GFM_InitialMass'][idx] / self.config['ageThreshold']).to(u.Msun / u.yr)
            elif self.config['sedMode'] == 'Cloud':
                properties['age'] = particles['age'][idx]
                
                def sample_cloud_mass(n_samples, alpha=-1.8, m_min=1e5, m_max=10**6.75):
                    # Calculate the exponent for the CDF (integral of PDF)
                    p = alpha + 1  # This is -0.8
                    
                    # Inverse Transform Sampling
                    u = np.random.random(n_samples)
                    term1 = m_max**p - m_min**p
                    masses = (term1 * u + m_min**p)**(1/p)
                    
                    return masses
                
                alpha = self.config['alpha']
                properties['cloud mass'] = sample_cloud_mass(idx.shape[0], alpha=alpha) * u.Msun
                properties['scaling'] = np.full(idx.shape[0], self.config['scaling']) * u.dimensionless_unscaled
                
            return properties
        
        if self.config['starformingSEDFamily'] == 'MAPPINGS':
            
            if self.config['includeVelocity']:
        
                paramNames = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'smoothing length',
                            'x-velocity', 'y-velocity', 'z-velocity',
                            'star formation rate', 'metallicity', 'compactness', 'pressure', 'covering factor',
                            'mass']
        
            else:
                
                paramNames = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'smoothing length',
                            'star formation rate', 'metallicity', 'compactness', 'pressure', 'covering factor',
                            'mass']
                
            self.createFile(paramNames, 'starforming', 
                            os.path.join(self.workingDir, 'starforming_regions.txt'), 
                            starFormingFunction)
            
        elif self.config['starformingSEDFamily'] == 'TODDLERS':
            
            if self.config['includeVelocity']:
                paramNames = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'smoothing length', 
                              'x-velocity', 'y-velocity', 'z-velocity']
                
            else:
                paramNames = ['x-coordinate', 'y-coordinate', 'z-coordinate', 'smoothing length']
            
            if self.config['sedMode'] == 'SFRNormalized':
                
                extensions = ['metallicity', 'star formation efficiency', 'cloud number density', 
                             'star formation rate']
                
            elif self.config['sedMode'] == 'Cloud':
                
                extensions = ['age', 'metallicity', 'star formation efficiency', 
                             'cloud number density', 'cloud mass', 'scaling']
                
            paramNames.extend(extensions)
            
            self.createFile(paramNames, 'starforming', 
                            os.path.join(self.workingDir, 'starforming_regions.txt'), 
                            starFormingFunctionTODDLERS)
        
        if self.config['includeDust']:
            def dustFunction(particles):
                
                particles['Temperature'] = u2temp(particles['InternalEnergy'],
                                                particles['ElectronAbundance']) * u.K
                if self.config['DISMModel'] == 'Camps_2016':
                    temperatureThreshold = self.config['temperatureThreshold']
                    idx = np.where((particles['StarFormationRate'] > 0) \
                        | (particles['Temperature'] < temperatureThreshold))[0]
                elif self.config['DISMModel'] == 'Torrey_2012':
                    left = np.log10(particles['Temperature'].to(u.K).value)
                    right_log = np.log10(particles['Density'].to(10**10 * self.h**2 * u.Msun * u.kpc**-3).value)
                    right = 6 + 0.25 * right_log
                    idx = np.where(left < right)[0]
                
                self.logger.info(f'Dust: {len(idx)}')
                
                if len(idx) == 0:
                    self.logger.info('No dust found.')
                    return {}
                
                properties = {}
                properties['x-coordinate'] = particles['Coordinates'][:, 0][idx]
                properties['y-coordinate'] = particles['Coordinates'][:, 1][idx]
                properties['z-coordinate'] = particles['Coordinates'][:, 2][idx]
                properties['mass'] = particles['Masses'][idx]
                properties['metallicity'] = particles['GFM_Metallicity'][idx]
                properties['temperature'] = particles['Temperature'][idx]
                properties['x-velocity'] = particles['Velocities'][:, 0][idx]
                properties['y-velocity'] = particles['Velocities'][:, 1][idx]
                properties['z-velocity'] = particles['Velocities'][:, 2][idx]
            
                return properties
        
            paramNames = ['x-coordinate', 'y-coordinate', 'z-coordinate',
                          'mass', 'metallicity', 'temperature', 
                          'x-velocity', 'y-velocity', 'z-velocity']
            
            self.createFile(paramNames, 'dust', 
                            os.path.join(self.workingDir, 'dusts.txt'), 
                            dustFunction)

    
    def __save_particles_to_h5(self):
        
        filename = os.path.join(self.workingDir, f'Subhalo_{self.subhalo_info["SubhaloID"]}_particles.h5')
        
        starPart = {}
        with h5py.File(filename, 'w') as h5f:
            if hasattr(self, 'starPart') and self.starPart is not None:
                for key, val in self.starPart.items():
                    starPart[key] = val
                grp = h5f.create_group('PartType4')
                for key, val in starPart.items():
                    # Cast to numpy array if it is a quantity/object, else save as is
                    if hasattr(val, 'value'):
                        arr = val.value
                        unit = str(val.unit)
                        if not unit:
                            unit = "1"
                    else:
                        arr = val
                        unit = "1"  # No unit for non-quantity
                    dset = grp.create_dataset(key, data=arr)
                    dset.attrs['unit'] = unit
            if hasattr(self, 'gasPart') and self.gasPart is not None:
                grp = h5f.create_group('PartType0')
                for key, val in self.gasPart.items():
                    if hasattr(val, 'value'):
                        arr = val.value
                        unit = str(val.unit)
                        if not unit:
                            unit = "1"
                    else:
                        arr = val
                        unit = "1"
                    dset = grp.create_dataset(key, data=arr)
                    dset.attrs['unit'] = unit
        
            
    def createFile(self, paramNames: list, partType: str, saveFilename: str, function: callable):
        
        """
        Create a file containing the parameters of particles of a given type.

        Parameters
        ----------
        paramNames : list
            List of parameter names.
        partType : str
            Type of particles.
        saveFilename : str
            Path to save the file.
        function : callable
            A function that processes particles and returns a dictionary of properties.
            The values of the dictionary should be astropy.units.Quantity objects.
        """
        if partType in ['star', 'stars', 'stellar', 'starforming', 'starformingRegions']:
            particles = self.starPart
        elif partType in ['gas', 'gases', 'dust']:
            particles = self.gasPart
        elif partType in ['dm', 'darkMatter']:
            if hasattr(self, 'dmPart'):
                particles = self.dmPart
            else:
                raise ValueError('DM particles not included.')
        elif partType in ['bh', 'bhs', 'blackhole', 'blackholes']:
            if hasattr(self, 'bhPart'):
                particles = self.bhPart
            else:
                raise ValueError('BH particles not included.')
        
        properties = function(particles)
        
        if len(properties) == 0:
            np.savetxt(f'{saveFilename}', [])
            return
                
        paramUnits = []
        for name in paramNames:
            unit = properties[name].unit
            paramUnits.append(unit)
            properties[name] = properties[name].to(unit).value
            size = properties[name].shape[0]

        header = f'{os.path.basename(saveFilename).split(".")[0]}\n'
        for i, (key, unit) in enumerate(zip(paramNames, paramUnits)):
            if unit == u.dimensionless_unscaled:
                unit = '1'
            elif 'solMass' in str(unit):
                unit = str(unit).replace('solMass', 'Msun')
            unit = str(unit).replace(' ', '') # remove empty space
            header = header + f'\ncolumn {i + 1}: {key} ({unit})'
        
        if len(properties) == 0:
            arr_size = (0, len(paramNames))
        else:
            arr_size = (size, len(paramNames))
            
        info_array = np.zeros(arr_size)
        
        for i, key in enumerate(paramNames):
            info_array[:, i] = properties[key]
        
        np.savetxt(f'{saveFilename}', info_array, header=header)
        
    def __get_properties(self) -> dict:
        
        properties = {}
        
        for key, value in self.config.items():
            properties[key] = value
        
        properties['snapRedshift'] = self.snapRedshift
        properties['snapNum'] = self.snapnum
        properties['subhaloID'] = self.subhalo_info['SubhaloID']
        properties['stellarMass'] = self.subhalo_info['SubhaloMassType'][4].to(u.Msun)
        properties['radius'] = self.subhalo_info['SubhaloHalfmassRadType'][4].to(u.kpc)
        
        
        if self.config['inLocal']:
            distance = properties['viewDistance']
            # properties['cosmology'] = 'LocalUniverseCosmology'
            properties['viewRedshift'] = 0
        else:
            distance = self.cosmology.luminosity_distance(self.viewRedshift)
            # properties['cosmology'] = 'FlatUniverseCosmology'
            properties['viewRedshift'] = self.viewRedshift
        
        properties['cosmology'] = self.cosmology.to_format('mapping')
        properties['lumiDis'] = distance
        
        faceAndEdge = properties['faceAndEdge']
        inclinations = properties['inclinations']
        azimuths = properties['azimuths']
        numViews = properties['numViews']
        
        if faceAndEdge:
            
            inclinations, azimuths = self.__calculate_angular_momentum_and_angles()
            numViews = 2
            properties['numViews'] = numViews
            
            self.logger.info('Using calculated face-on and edge-on angles')
        
        if not faceAndEdge and properties['randomViews']:
            
            inclinations = np.random.uniform(0, 180, numViews)
            azimuths = np.random.uniform(-360, 360, numViews)
            numViews = len(inclinations)
            
            self.logger.info(f'Using {numViews} random views')
            
        elif not faceAndEdge and not properties['randomViews']:
            
            inclinations = properties['inclinations']
            azimuths = properties['azimuths']
            numViews = len(inclinations)
            
            self.logger.info(f'Using {numViews} specified views')
            
        properties['inclinations'] = inclinations
        properties['azimuths'] = azimuths
        properties['numViews'] = numViews
        
        for i, inc, azi in zip(range(numViews), inclinations, azimuths):
            self.logger.info(f'View {i}: Inclination = {inc:.2f} deg, Azimuth = {azi:.2f} deg')
        
        properties['numViews'] = numViews
        properties['inclinations'] = inclinations
        properties['azimuths'] = azimuths
        
        properties['boxLength'] = self.partRegion
        properties['boxlength_in_arcsec'] = (self.partRegion / distance * u.rad).to(u.arcsec)
        # properties['boxlength_in_arcsec'] = np.rad2deg(boxLength / (distance * 10**6)) * 3600 # in arcsec
        
        return properties
    
    def __create_ski(self):
        
        self.logger.info('Creating .ski file.')
        
        config = self.__get_properties()
        
        mode = config['simulationMode']
        
        ski_file = os.path.join(self.dataDir,  f'ski_templates/{mode}_template.ski')
        
        with open(ski_file, 'r') as file:
            data = file.read()
            
        if mode in ['DustEmission', 'ExtinctionOnly']:
            
            begin_str = '<VoronoiMeshMedium'
            end_str = '</VoronoiMeshMedium>'
            offset = len(end_str)
            
            idx_begin = data.index(begin_str)
            idx_end = data.index(end_str) + offset
            
            voronoiMeshMediumInfo = data[idx_begin: idx_end]
            remainingInfo = data[idx_end:]
            
            if config['hydrodynamicSolver'] == 'smoothParticle':
                replace_str = '<ParticleMedium filename="dusts.txt" '
                replace_str += 'massType="Mass" massFraction="0.3" '
                replace_str += 'importMetallicity="true" importTemperature="true" '
                replace_str += 'maxTemperature="0 K" importVelocity="false" importMagneticField="false" importVariableMixParams="false" useColumns="">\n'
                replace_str += '<smoothingKernel type="SmoothingKernel">\n'
                replace_str += '<CubicSplineSmoothingKernel/>\n'
                replace_str += '</smoothingKernel>\n'
                replace_str += '<materialMix type="MaterialMix">\n'
                replace_str += '<ZubkoDustMix numSilicateSizes="15" numGraphiteSizes="15" numPAHSizes="15"/>\n'
                replace_str += '</materialMix>\n'
                replace_str += '</ParticleMedium>\n'

                data = data.replace(voronoiMeshMediumInfo, replace_str)
                
        begin_str = '<cosmology'
        end_str = '</cosmology>'
        offset = len(end_str)
        
        idx_begin = data.index(begin_str)
        idx_end = data.index(end_str) + offset
        
        cosmologyInfo = data[idx_begin: idx_end]
        remainingInfo = data[idx_end:]
        
        if config['inLocal']:
            replace_str = '<cosmology type="Cosmology">\n'
            replace_str += '<LocalUniverseCosmology/>\n'
            replace_str += '</cosmology>\n'
            
            data = data.replace(cosmologyInfo, replace_str)
        else:
            data = data.replace('redshift="0.008"', f'redshift="{self.viewRedshift}"')
        
        starSEDFamily = {'BC03': 'BruzualCharlotSEDFamily',
                     'FSPS': 'FSPSSEDFamily'}
        starSEDFamily = starSEDFamily[config['starSEDFamily']]
        
        if starSEDFamily == 'FSPSSEDFamily':
            data = data.replace('resolution="High"', '')
        
        if mode in ['DustEmission', 'ExtinctionOnly']:
            dustEmissionType = config['dustEmissionType']
            data = data.replace('dustEmissionType="Equilibrium"',
                                f'dustEmissionType="{dustEmissionType}"')
        
        initialMassFunction = config['initialMassFunction']
        
        data = data.replace('BruzualCharlotSEDFamily', starSEDFamily)
        data = data.replace('Chabrier', initialMassFunction)
        
        sfSEDFamily = '<MappingsSEDFamily/>'
        if config['starformingSEDFamily'] == 'TODDLERS':
            replace_str = '<ToddlersSEDFamily sedMode="Cloud" stellarTemplate="BPASSChab100Bin" includeDust="true" resolution="High" sfrPeriod="Period10Myr"/>'
        
            sedMode = config['sedMode']
            replace_str = replace_str.replace('sedMode="Cloud"', f'sedMode="{sedMode}"')
            stellarTemplate = config['stellarTemplate']
            replace_str = replace_str.replace('stellarTemplate="BPASSChab100Bin"', f'stellarTemplate="{stellarTemplate}"')
            if config['includeDust']:
                includeDust = 'true'
            else:
                includeDust = 'false'
            replace_str = replace_str.replace('includeDust="true"', f'includeDust="{includeDust}"')
            sfrPeriod = int(config['sfrPeriod'])
            replace_str = replace_str.replace('sfrPeriod="Period10Myr"', f'sfrPeriod="Period{sfrPeriod}Myr"')

            data = data.replace(sfSEDFamily, replace_str)
            
        elif config['starformingSEDFamily'] == 'MAPPINGS':
            pass
        
        numPackets = config['numPackets']
        data = data.replace('numPackets="1e7"', f'numPackets="{numPackets}"')
        
        includeVelocity = config['includeVelocity']
        if includeVelocity:
            data = data.replace('importVelocity="false"', 'importVelocity="true"')
        
        minWaveRT = config['minWaveRT'].to(u.um)
        maxWaveRT = config['maxWaveRT'].to(u.um)
        
        minWaveOutput = config['minWaveOutput'].to(u.um)
        maxWaveOutput = config['maxWaveOutput'].to(u.um)
        
        data = data.replace('minWavelength="0.01 micron"', f'minWavelength="{minWaveRT.value} micron"')
        data = data.replace('maxWavelength="1.2 micron"', f'maxWavelength="{maxWaveRT.value} micron"')
        
        numWaveRT = config['numWaveRT']
        data = data.replace('numWavelengths="1000"', f'numWavelengths="{numWaveRT}"')
        
        waveGridRT = config['waveGridRT']
        grid_type = {'Linear': 'LinWavelengthGrid',
                'Log': 'LogWavelengthGrid'}
        grid_type = grid_type[waveGridRT]
        data = data.replace('LinWavelengthGrid', grid_type)
        
        massFraction = config['massFraction']
        data = data.replace('massFraction="0.3"', f'massFraction="{massFraction}"')
        
        dustConfig = '<ZubkoDustMix numSilicateSizes="15" numGraphiteSizes="15" numPAHSizes="15"/>'
        dustModel = config['dustModel']
        
        numSilicateSizes = np.int32(config['numSilicateSizes'])
        numGraphiteSizes = np.int32(config['numGraphiteSizes'])
        numPAHSizes = np.int32(config['numPAHSizes'])
        numHydrocarbonSizes = np.int32(config['numHydrocarbonSizes'])
        
        if dustModel == 'ThemisDustMix':
            dustConfigNew = f'<{dustModel} numHydrocarbonSizes="{numHydrocarbonSizes}" numSilicateSizes="{numSilicateSizes}"/>'
        else:
            dustConfigNew = f'<{dustModel} numSilicateSizes="{numSilicateSizes}" numGraphiteSizes="{numGraphiteSizes}" numPAHSizes="{numPAHSizes}"/>'
            
        data = data.replace(dustConfig, dustConfigNew)
        
        minLevel = np.int32(config['minLevel'])
        maxLevel = np.int32(config['maxLevel'])
        data = data.replace('minLevel="8"', f'minLevel="{minLevel}"')
        data = data.replace('maxLevel="12"', f'maxLevel="{maxLevel}"')
        
        spatialRange = self.partRegion.to(u.pc).value / 2
        data = data.replace('minX="-5e4 pc"', f'minX="{-spatialRange} pc"')
        data = data.replace('maxX="5e4 pc"', f'maxX="{spatialRange} pc"')
        data = data.replace('minY="-5e4 pc"', f'minY="{-spatialRange} pc"')
        data = data.replace('maxY="5e4 pc"', f'maxY="{spatialRange} pc"')
        data = data.replace('minZ="-5e4 pc"', f'minZ="{-spatialRange} pc"')
        data = data.replace('maxZ="5e4 pc"', f'maxZ="{spatialRange} pc"')

        grid_type = {'Linear': 'LinWavelengthGrid',
                'Log': 'LogWavelengthGrid'}
        waveGridOutput = config['waveGridOutput']
        grid_typeOutput = grid_type[waveGridOutput]
        
        begin_str = '<defaultWavelengthGrid'
        end_str = '</defaultWavelengthGrid>'
        offset = len(end_str)
        
        idx_begin = data.index(begin_str)
        idx_end = data.index(end_str) + offset
        
        default_instrument = data[idx_begin: idx_end]
        
        replace_str = data[idx_begin: idx_end]
        
        replace_str = replace_str.replace(f'minWavelength="{minWaveRT.value} micron"', 
                                          f'minWavelength="{minWaveOutput.value} micron"')
        replace_str = replace_str.replace(f'maxWavelength="{maxWaveRT.value} micron"', 
                                          f'maxWavelength="{maxWaveOutput.value} micron"')
        replace_str = replace_str.replace(f'{grid_type}', grid_typeOutput)
        
        data = data.replace(default_instrument, replace_str)
        
        begin_str = '<FullInstrument'
        end_str = '</FullInstrument>'
        offset = len(end_str)
        
        idx_begin = data.index(begin_str)
        idx_end = data.index(end_str) + offset
        
        instrumentInfo = data[idx_begin: idx_end]
        remainingInfo = data[idx_end:]
        
        
        data = data.replace(instrumentInfo, '')
        
        distance = config['lumiDis']
        
        fieldOfView = config['fieldOfView'] # in arcsec
        
        if fieldOfView == 0:
            fovSize = self.partRegion.to(u.pc) # in pc
            fieldOfView = (fovSize / distance * u.rad).to(u.arcsec) # in arcsec
        else:
            fovSize = (distance * fieldOfView.to(u.rad).value).to(u.pc) # in pc
        
        config['fieldOfView'] = fieldOfView
        config['fovSize'] = fovSize
        config['resolution'] = (distance * config['pixelScale'].to(u.rad).value).to(u.pc)
        
        numWaveOutput = config['numWaveOutput']
        
        instrumentInfo = instrumentInfo.replace(f'minWavelength="{minWaveRT.value} micron"', 
                                                f'minWavelength="{minWaveOutput.value} micron"')
        instrumentInfo = instrumentInfo.replace(f'maxWavelength="{maxWaveRT.value} micron"', 
                                                f'maxWavelength="{maxWaveOutput.value} micron"')
        instrumentInfo = instrumentInfo.replace(f'numWavelengths="{numWaveRT}"', 
                                                f'numWavelengths="{numWaveOutput}"')
        
        instrumentInfo = '\n' + instrumentInfo + '\n'
        
        inclinations = config['inclinations']
        azimuths = config['azimuths']
        
        numViews = config['numViews']
        
        pixelScale = config['pixelScale']
        oversamp = config['oversample']
    
        # oversample the pixel scale
        numPixels_original = int(fieldOfView / pixelScale)
        numPixels = numPixels_original * oversamp
        
        config['numPixels_original'] = numPixels_original
        config['numPixels'] = numPixels
        
        insert_begin_idx = idx_begin
        for i, (inclination, azimuth) in enumerate(zip(inclinations, azimuths)):
            info = instrumentInfo.replace('view', f'view_{i:02d}')
            info = info.replace('inclination="0 deg"', f'inclination="{inclination} deg"')
            info = info.replace('azimuth="0 deg"', f'azimuth="{azimuth} deg"')
            info = info.replace('fieldOfViewX="1e5 pc"', f'fieldOfViewX="{fovSize.value} pc"')
            info = info.replace('fieldOfViewY="1e5 pc"', f'fieldOfViewY="{fovSize.value} pc"')
            info = info.replace('numPixelsX="1000"', f'numPixelsX="{numPixels}"')
            info = info.replace('numPixelsY="1000"', f'numPixelsY="{numPixels}"')
            
            grid_type = {'Linear': 'LinWavelengthGrid',
                    'Log': 'LogWavelengthGrid'}
            waveGridOutput = config['waveGridOutput']
            grid_type = grid_type[waveGridOutput]
            info = info.replace('LinWavelengthGrid', grid_type)
            
            data = data[:insert_begin_idx] + info
            insert_begin_idx = insert_begin_idx + len(info)
            
        if config['inLocal']:
            data = data.replace('distance="0 Mpc"', f'distance="{distance.to_value(u.Mpc)} Mpc"')

        data = data + remainingInfo
        
        with open(self.workingDir + '/skirt.ski', 'w') as file:
            file.write(data)
        
        with open(self.workingDir + '/config.json', 'w') as file:
            json.dump(to_json_safe(config), file, indent=4)

        self.logger.info('------estimate memory usage------')
        self.logger.info(f'numViews: {numViews}')
        self.logger.info(f'numSpatialPixels: {numPixels}')
        self.logger.info(f'numWavelengthPixels: {numWaveOutput}')
        
        numPixels = np.int64(numPixels) # avoid overflow
        dataCubeSize = np.int64(numPixels ** 2 * numWaveOutput * numViews)
        dataSize_in_GB = np.around(dataCubeSize * 8 * 1e-9, 3)
        self.logger.info(f'Estimated memory usage: {dataSize_in_GB} GB')
        
    def inputParticles(self, partType: str, particles: dict, 
                       subhaloInfo: dict=None, subhaloInfoFile: str=None,
                       centralized: bool=False):
        """
        Input particle information for a given particle type (e.g., stars, gas, etc.) into the PreProcess object.

        Parameters
        ----------
        partType : str
            The type of particle being input (e.g., 'star', 'gas', 'dm', etc.).
        particles : dict
            Dictionary containing particle properties in astropy.units.Quantity objects.
        subhaloInfo : dict, optional
            Dictionary containing subhalo properties. Required if subhaloInfoFile is not provided. 'id', 'halfmassrad_stars', 'pos_x', 'pos_y', 'pos_z' are required.
        subhaloInfoFile : str, optional
            Path to a JSON file containing subhalo information. Have higher priority compared to `subhaloInfo`.
        centralized : bool, default False
            If True, recenters the particle positions according to the subhalo information. Default is False.

        Raises
        ------
        ValueError
            If neither subhaloInfo nor subhaloInfoFile is provided.
        FileNotFoundError
            If subhaloInfoFile is specified but does not exist.
        KeyError
            If required parameters are missing from subhaloInfo.
        """
        
        os.makedirs(self.workingDir, exist_ok=True)
        
        self.inputMethod = 'partInput'
        self.logger.info(f'------Inputing {partType} particles------')
        
        if not subhaloInfo and not subhaloInfoFile:
            raise ValueError('Either subhaloInfo or subhaloInfoFile must be provided.')
        
        if subhaloInfoFile is not None:
            if not os.path.exists(subhaloInfoFile):
                raise FileNotFoundError(f'Subhalo info file {subhaloInfoFile} does not exist.')
            
            with open(subhaloInfoFile, 'r') as f:
                subhaloInfo = json.load(f)
                
            subhalo_info = {}
            subhalo_info['SubhaloID'] = subhaloInfo['id']
            subhalo_info['snapNum'] = self.config['snapNum']
            subhalo_info['snapRedshift'] = self.config['snapRedshift']
            subhalo_info['viewRedshift'] = self.config['viewRedshift']
            
            for key in self.table.keys():
                if isinstance(self.table[key], list):
                    subhalo_info[key] = []
                    for info_key in self.table[key]:
                        if info_key is not None:
                            subhalo_info[key].append(subhaloInfo[info_key])
                        else:
                            subhalo_info[key].append(np.nan)
                    if isinstance(self.table_units[key], u.CompositeUnit):
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(u.Unit(self.table_units[key]))
                    else:
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(self.table_units[key].unit)
                        
                else:
                    info_key = self.table[key]
                    if isinstance(self.table_units[key], u.CompositeUnit):
                        subhalo_info[key] = u.Quantity(subhaloInfo[info_key], self.table_units[key]).to(u.Unit(self.table_units[key]))
                    else:
                        subhalo_info[key] = u.Quantity(subhaloInfo[info_key], self.table_units[key]).to(self.table_units[key].unit)
                        

        elif subhaloInfo is not None and subhaloInfoFile is None:
            
            required_params = ['id', 'halfmassrad_stars', 
                               'pos_x', 'pos_y', 'pos_z']
            
            for par in required_params:
                if par not in subhaloInfo.keys():
                    raise KeyError(f'Key {par} not found in subhalo info.')
            
            subhalo_info = {}
            subhalo_info['SubhaloID'] = subhaloInfo['id']
            subhalo_info['snapNum'] = self.config['snapNum']
            subhalo_info['snapRedshift'] = self.config['snapRedshift']
            subhalo_info['viewRedshift'] = self.config['viewRedshift']
            
            for key in self.table.keys():
                unit = self.table_units[key]
                if isinstance(self.table[key], list):
                    subhalo_info[key] = []
                    for info_key in self.table[key]:
                        if info_key is not None:
                            if info_key in subhaloInfo.keys():
                                subhalo_info[key].append(u.Quantity(subhaloInfo[info_key]))
                            else:
                                subhalo_info[key].append(u.Quantity(np.nan, unit))
                        else:
                            subhalo_info[key].append(u.Quantity(np.nan, unit))
                    
                else:
                    if key in subhaloInfo.keys():
                        subhalo_info[key] = u.Quantity(subhaloInfo[key])
                    else:
                        subhalo_info[key] = u.Quantity(np.nan, unit)
            
            for key in subhalo_info.keys():
                if key != 'SubhaloID' and key != 'snapNum' and key != 'snapRedshift' and key != 'viewRedshift':
                    if isinstance(self.table_units[key], u.CompositeUnit):
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(u.Unit(self.table_units[key]))
                    else:
                        subhalo_info[key] = u.Quantity(subhalo_info[key], self.table_units[key]).to(self.table_units[key].unit)
                
        
        with open(os.path.join(self.workingDir, f'Subhalo_{subhalo_info["SubhaloID"]}.json'), 'w') as f:
            json.dump(to_json_safe(subhalo_info), f, indent=4)
            
        self.subhalo_info = subhalo_info
        
        # self.id = subhaloInfo['SubhaloID']
        # self.mass = (subhaloInfo['stellarMass']).to(u.Msun)
        # self.radius = (subhaloInfo['halfStellarMassRadius']).to(u.kpc)
        # self.vel = (subhaloInfo['velocity']).to(u.km/u.s)
        
        if not centralized:
            centerPosition = (subhalo_info['SubhaloPos']).to(u.kpc)
        else:
            centerPosition = None
            
        radius = subhalo_info['SubhaloHalfmassRadType'][4].to(u.kpc)
        
        self.partRegion = (self.config['boxLengthScale'] * radius).to(u.kpc)
        self.partRegion = np.min([self.partRegion.value, self.config['maxBoxLength'].to(u.kpc).value])
        self.partRegion = self.partRegion * u.kpc
        
        for key, value in particles.items():
            try:
                unit = value.unit
                particles[key] = np.array(value) * unit # convert to numpy array with unit
            except:
                raise ValueError(f'{key} data is not a u.Quantity object.')

        if partType in ['star', 'stars']:
            self.starPart = particles
            self.starPart = self.__in_box_mask(self.starPart, centerPosition)
            
        if self.config['includeDust']:
            if partType in ['gas', 'gases', 'dust', 'dusts']:
                self.gasPart = particles
                self.gasPart = self.__in_box_mask(self.gasPart, centerPosition)
    
    def modify_configs(self, arguments: dict):
        
        """
        Modify the current configuration (`self.config`) based on a provided dictionary of arguments.

        Parameters
        ----------
        arguments : dict
            A dictionary where keys correspond to configuration options and values are the new settings
            to apply. If the configuration value is an astropy Quantity, the provided value is converted
            to the same unit.
        """
        
        exist_keys = self.config.keys()
        for key, value in arguments.items():
            if key in exist_keys:
                original_value = self.config[key]
                if isinstance(original_value, u.Quantity):
                    try:
                        value = u.Quantity(value)
                        unit = original_value.unit
                        value = value.to(unit)
                    except:
                        raise ValueError(f'{value} cannot be converted to u.Quantity with the same unit as {original_value}')
                self.config[key] = value
                self.logger.info(f'Changed {key} from {original_value} to {value}')
    
    def inputs(self, data: dict):
        """
        Accepts input data as a dictionary and sets the necessary configuration and physical properties.
        
        This method allows the user to provide essential subhalo and physical properties manually, such as
        stellar mass, half mass radius, and velocity. It modifies the configuration accordingly, performs type
        and unit checks, and sets up the `partRegion` for further processing. This method is intended for cases
        when inputs are not loaded from pre-defined data files but are instead provided directly by the user.
        
        Parameters
        ----------
        data : dict
            A dictionary containing at least the following keys:
            
                - 'SubhaloID' (int): Identifier for the subhalo.
                - 'stellarMass' (Quantity): Stellar mass with an astropy unit (preferably Msun).
                - 'halfStellarMassRadius' (Quantity): Half-mass radius with an astropy unit (preferably kpc).
                - 'velocity' (Quantity): Velocity with an astropy unit (preferably km/s).
            
            Additional configuration parameters can also be included. These will override existing configuration settings.
        
        Raises
        ------
        ValueError
            If required parameters are missing or have incorrect types/units.
        FileNotFoundError
            If required particle files are not found in the working directory.
        """
        
        os.makedirs(self.workingDir, exist_ok=True)
        self.inputMethod = 'input'
        
        # in case changing configs
        # exist_keys = self.config.keys()
        # for key, value in data.items():
        #     if key in exist_keys:
        #         self.config[key] = value
        
        self.modify_configs(data)
        
        required_params = ['SubhaloID', 'stellarMass', 'halfStellarMassRadius', 
                           'velocity']
        for par in required_params:
            if par not in data.keys():
                raise ValueError(f'{par} is required but not found in the input data.')
            
            if par == 'SubhaloID':
                self.id = data['SubhaloID']
            
            if par == 'stellarMass':
                try:
                    self.mass = u.Quantity(data['stellarMass']).to(u.Msun)
                except:
                    raise ValueError('stellarMass is not with astropy Unit.')
                    
            if par == 'halfStellarMassRadius':
                try:
                    self.radius = u.Quantity(data['halfStellarMassRadius']).to(u.kpc)
                except:
                    raise ValueError('halfStellarMassRadius is not with astropy Unit.')
                
            if par == 'velocity':
                try:
                    self.vel = u.Quantity(data['velocity']).to(u.km/u.s)
                except:
                    raise ValueError('velocities is not with astropy Unit.')
                
        self.partRegion = (self.config['boxLengthScale'] * self.radius).to(u.kpc)
        self.partRegion = np.min([self.partRegion.value, self.config['maxBoxLength'].value])
        self.partRegion = self.partRegion * u.kpc
        
        if not os.path.join(self.workingDir, 'stars.txt'):
            raise FileNotFoundError(f'stars.txt not found in {self.workingDir}.')
        
        if not os.path.join(self.workingDir, 'starforming_regions.txt'):
            raise FileNotFoundError(f'starforming_regions.txt not found in {self.workingDir}.')
        
        if self.config['includeDust']:
            if not os.path.join(self.workingDir, 'dusts.txt'):
                raise FileNotFoundError(f'dusts.txt not found in {self.workingDir}.')
            
    def prepare(self, arguments: Union[dict, None]=None):
        """
        Prepare and preprocess data by modifying configuration parameters, initializing or updating class attributes,
        and processing particle input and files based on the current input method.

        Parameters
        ----------
        arguments : dict or None, optional
            Dictionary of new configuration arguments to modify or update in the class instance. Default is None.

        Notes
        -----
        This function performs the following operations:
          1. Modifies the current configuration if `arguments` are provided.
          2. Calls class initialization.
          3. Depending on `self.inputMethod`, it acquires, processes, or loads particles from snapshots,
             direct input, subhalo files, or user-provided files.
          4. Saves processed particle data to HDF5 and creates a .ski configuration file.
        """
        
        os.makedirs(self.workingDir, exist_ok=True)
        
        if arguments is not None:
            self.modify_configs(arguments) 
        
        self.__init()
        if self.inputMethod == 'snapshot':
            self.__get_particles()
            self.__processing_particles()
        elif self.inputMethod == 'partInput':
            data = {'SubhaloID': self.id,
                    'stellarMass': self.mass,
                    'halfStellarMassRadius': self.radius, 
                    'velocity': self.vel}
            self.inputs(data)
        elif self.inputMethod == 'subhaloFile':
            self.__processing_particles()
        elif self.inputMethod == 'input':
            pass
        
        self.__save_particles_to_h5()
        self.__create_ski()
        
        self.config = self.config_ori.copy() # reset config to original