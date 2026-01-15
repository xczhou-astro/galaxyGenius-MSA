import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Planck15, Cosmology
import sys
import os
import shutil
from astropy.visualization import ManualInterval, LogStretch, make_rgb
from typing import Union
from types import NoneType
import logging
import inspect
import json
import re
import math
from scipy.interpolate import interp1d
import importlib

class Units:
    _instance = None
    _initialized = False
    
    def __new__(cls, cosmology=None, snapRedshift=None, reinitialize=False):
        if cls._instance is None:
            cls._instance = super(Units, cls).__new__(cls)
        return cls._instance
        
    def __init__(self, cosmology=None, snapRedshift=None, reinitialize=False):
        if not self._initialized or reinitialize:
            
            self.cosmology = cosmology if cosmology is not None else Planck15
            self.snapRedshift = snapRedshift if snapRedshift is not None else 0.
            
            self.a = 1 / (1 + self.snapRedshift)
            self.h = self.cosmology.h
            
            self.distance = u.kpc * self.a / self.h
            self.position = u.kpc * self.a / self.h
            self.density = (10**10 * u.Msun / self.h) / (u.kpc * self.a / self.h)**3
            self.mass = 10**10 * u.Msun / self.h
            self.sfr = u.Msun / u.yr
            self.velocity = u.km / u.s * np.sqrt(self.a)
            self.potential = (u.km / u.s)**2 / self.a
            self.temperature = u.K
            self.energy = (u.km / u.s)**2
            self.dimless = u.dimensionless_unscaled
            
            # mostly used: ckpc
            self.ckpc = u.kpc * self.a
            self.cMpc = u.Mpc * self.a
            self.pkpc = u.kpc
            self.pMpc = u.Mpc
            
            self.ckpc_over_h = u.kpc * self.a / self.h
            self.cMpc_over_h = u.Mpc * self.a / self.h
            self.pc_over_h = u.pc * self.a / self.h
            
            self._initialized = True
    
    def get_cosmology(self):
        return self.cosmology
    
    def get_snapRedshift(self):
        return self.snapRedshift
    
    def unit_convention(self):
        print(f'Current unit convention is in redshift {self.snapRedshift} for cosmology {self.cosmology.name}')
                
    def explain(self):
        """
        Default unit conventions are used by TNG simulation:
        
        - distance: ckpc / h
        - density: (10**10 * Msun) / (ckpc / h)^3
        - mass: 10**10 * Msun
        - sfr: Msun / yr
        - velocity: km * sqrt(a) / s
        - potential: (km / s)^2 / a
        - temperature: K
        - energy: (km / s)^2
        
        comoving quantities can be converted to physical ones by multiplying for the 
        appropriate power of the scale factor a. For instance, to convert a length 
        in physical units it is sufficient to multiply it by a, volumes need a factor a^3,
        densities a^-3 and so on. 
        
        When using other simulations, the unit convention can be re-defined. 
        """
        print(self.explain.__doc__)
        
def setup_logging(log_file="galaxygenius.log", log_level=logging.INFO, force_reconfigure=False):
    """Set up logging configuration that can be used across multiple modules
    
    Args:
        log_file: Path to the log file
        log_level: Logging level (default: logging.INFO)
        force_reconfigure: If True, forces reconfiguration even if handlers exist.
                          Use this after moving log files to update file handlers.
    """
    
    root_logger = logging.getLogger()
    
    # Always remove existing file handlers to prevent accumulation
    # Keep console handlers, but replace file handlers
    file_handlers_to_remove = []
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            file_handlers_to_remove.append(handler)
    
    # Close and remove file handlers
    for handler in file_handlers_to_remove:
        try:
            handler.flush()
            handler.close()
        except Exception:
            pass
        root_logger.removeHandler(handler)
    
    # Check if we need to add a console handler
    has_console_handler = any(isinstance(h, logging.StreamHandler) 
                              for h in root_logger.handlers)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Console handler (only add once)
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler (always create new one for the specified file)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, mode='a')  # Append mode
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set log level
    root_logger.setLevel(log_level)
    
    # Return logger for the calling module
    caller_module = inspect.currentframe().f_back.f_globals['__name__']
    return logging.getLogger(caller_module)

def galaxygenius_data_dir():
    
    if os.environ.get('GALAXYGENIUS_DATA_DIR') is not None:
        dataDir = os.environ.get('GALAXYGENIUS_DATA_DIR').split(':')[0]
        if not os.path.exists(dataDir):
            raise FileNotFoundError('Data directory not found. Please set GALAXYGENIUS_DATA_DIR environment variable.')
    else:
        print('GALAXYGENIUS_DATA_DIR not set in environment variables. ' + 'Data directory falling to default path: ../Data')
        dataDir = '../Data'
        if not os.path.exists(dataDir):
            raise FileNotFoundError('Data directory not found. Please set GALAXYGENIUS_DATA_DIR environment variable.')
    
    return dataDir

def u2temp(u_energy: Union[float, u.Quantity], x_e: float) -> float:
    '''
    u_energy: InternelEnergy
    x_e: ElectronAbundance
    
    return:
    T: temperatrure in K
    '''
    
    if isinstance(u_energy, u.Quantity):
        u_energy = u_energy.value
    
    X_H = 0.76
    u_energy = u_energy * (u.km/u.s)**2
    mu = 4 / (1 + 3 * X_H + 4 * X_H * x_e) * const.m_p
    T = ((5/3 - 1) * u_energy/const.k_B * mu).to(u.K)
    T = T.value
    return T

def convert_to_rgb(bandpassImage: Union[np.ndarray, list], idx: list=[2, 3, 5]) -> np.ndarray:
    
    '''
    Convert the bandpass image to RGB image
    
    Args:
        bandpassImage: bandpass image
        idx: index of the bandpass image used to create the RGB image
        
    Returns:
        rgb: RGB image
    '''

    img_red = bandpassImage[idx[2]]
    img_green = bandpassImage[idx[1]]
    img_blue = bandpassImage[idx[0]]
    
    pctl = 99.5
    maxv = 0
    for img in [img_red, img_green, img_blue]:
        val = np.percentile(img, pctl)
        if val > maxv:
            maxv = val

    rgb = make_rgb(img_red, img_green, img_blue, interval=ManualInterval(vmin=0, vmax=maxv),
                   stretch=LogStretch(a=1000)) 
    
    return rgb
        
def split(string: str, castType: Union[type, NoneType]=None) -> list:
    
    '''
    Split the string into a list
    
    Args:
        string: string to be split
        castType: type of the elements in the list
        
    Returns:
        splits: list of the split string
    '''
    
    splits = [inc for inc in "".join(string.split()).split(',')]
    if castType is not None:
        splits = [castType(sp) for sp in splits]
        
    return splits

def get_wavelength_scale(filename: str) -> float:
    
    '''
    Get the wavelength scale of the filter
    
    Args:
        filename: filename of the filter
        
    Returns:
        wavelength_scale: wavelength scale of the filter
    '''
    
    with open(filename) as file:
        header = file.readline()
        
    if 'angstrom' in header or 'AA' in header:
        wavelength_scale = 1
    elif 'nm' in header:
        wavelength_scale = 10
    elif 'um' in header or 'micron' in header:
        wavelength_scale = 10 * 10**3
    else:
        # default consider as AA
        wavelength_scale = 1
    return wavelength_scale

def get_wavelength_unit(filename: str) -> u.Unit:
    
    with open(filename) as file:
        header = file.readline()
        
    if 'angstrom' in header or 'AA' in header:
        wave_unit = u.angstrom
    elif 'nm' in header:
        wave_unit = u.nm
    elif 'um' in header or 'micron' in header:
        wave_unit = u.um
    elif 'cm' in header:
        wave_unit = u.cm
    elif 'm' in header:
        wave_unit = u.m
    else:
        # default consider as AA
        wave_unit = u.angstrom
    return wave_unit
    

def calc_pivot(dataDir: str, survey: str, filter: str) -> float:
    
    '''
    Calculate the pivot wavelength of the filter
    
    Args:
        dataDir: directory of the data
        survey: survey name
        filter: filter name
        
    Returns:
        pivot: pivot wavelength of the filter
    '''
    
    filterDir = f'{dataDir}/filters/{survey}'
    filterLs = os.listdir(filterDir)
    filterNames = [name.split('.')[0] for name in filterLs]
    filename = filterLs[filterNames.index(filter)]
    filterName = os.path.join(filterDir, filename)
        
    wavelength_scale = get_wavelength_scale(filterName)
    
    try:
        transmission = np.loadtxt(filterName)
    except:
        transmission = np.load(filterName)
    
    transmission[:, 0] = transmission[:, 0] * wavelength_scale
    
    numerator = np.trapz(transmission[:, 1], transmission[:, 0])
    denomerator = np.trapz(transmission[:, 1] * transmission[:, 0]**-2,
                               transmission[:, 0])
    pivot = np.sqrt(numerator/denomerator)
    
    return pivot

def copyfile(src: str, tar: str):
    if os.path.exists(src) and not os.path.exists(tar):
        shutil.copyfile(src, tar)

def extend(values: Union[int, float, list, u.Quantity], nums: int) -> list:
    
    '''
    Extend the values to list with size consistent with nums
    
    Args:
        values: values to be extended
        nums: number of values to be extended
        
    Returns:
        values: extended values
    '''
    
    # if the values is a scaler
    if isinstance(values, (int, float, np.int32, np.int64, np.float32, np.float64)):
        values = u.Quantity([values] * nums)
    # elif the values is a list
    elif isinstance(values, list):
        if len(values) == nums:
            try:
                values = u.Quantity(values)
            except Exception as e:
                raise e
        else:
            raise ValueError(f'length of values is {len(values)}, which is not consistent with nums {nums}.')
    elif isinstance(values, u.Quantity):
        if values.size == 1:
            values = u.Quantity([values] * nums)
        elif values.size == nums:
            values = u.Quantity(values)
        else:
            raise ValueError(f'length of values is {len(values)}, which is not consistent with nums {nums}.')
    return values

def to_json_safe(obj):
    # ---- None ----
    if obj is None:
        return None

    # ---- Python float ----
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    # ---- NumPy floating ----
    if isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)

    # ---- NumPy integer (no NaN in real ints, but safe) ----
    if isinstance(obj, np.integer):
        return int(obj)

    # ---- NumPy masked values ----
    if np.ma.is_masked(obj):
        return None
    
    # ---- Class type ----
    if isinstance(obj, type):
        return {
            "__type__": "class",
            "module": obj.__module__,
            "name": obj.__name__
        }

    # ---- Astropy Quantity ----
    if isinstance(obj, u.Quantity):
        value = obj.value
        unit = str(obj.unit)

        return {
            "value": to_json_safe(value),
            "unit": unit
        }

    # ---- NumPy array ----
    if isinstance(obj, np.ndarray):
        return [to_json_safe(x) for x in obj.tolist()]

    # ---- dict ----
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}

    # ---- list / tuple ----
    if isinstance(obj, (list, tuple)):
        return [to_json_safe(x) for x in obj]

    # ---- fallback ----
    return obj

def custom_serialier(obj):
    
    '''
    Custom serializer for json dump
    
    Args:
        obj: object to be serialized
        
    Returns:
        obj: serialized object
    '''
    
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, float) and np.isinf(obj):
        return 'Infinity'
    elif type(obj) == u.Quantity:
        value = obj.value.tolist()
        if value == np.inf:
            value = 'Infinity'
        unit = str(obj.unit)
        return {
            'value': value,
            'unit': unit
        }
    elif type(obj) == np.ndarray:
        return obj.tolist()
    
# def _assign(value, unit):
#     if u.Quantity(value).unit == unit:
#         value = u.Quantity(value)
#     else:
#         value = u.Quantity(value) * unit
#     return value

def _assign(value, unit):
    try:
        value = u.Quantity(value)
        value = value.to(unit)
    except:
        value = u.Quantity(value) * unit
        
    return value
    
def check_exist_and_assign_unit(conf, key, unit):
    if key in conf:
        conf[key] = _assign(conf[key], unit)
        
    return conf
        
def assign_unit(conf):
    
    conf['minWaveRT'] = _assign(conf['minWaveRT'], u.um)
    conf['maxWaveRT'] = _assign(conf['maxWaveRT'], u.um)
    conf['minWaveOutput'] = _assign(conf['minWaveOutput'], u.um)
    conf['maxWaveOutput'] = _assign(conf['maxWaveOutput'], u.um)
    conf['maxBoxLength'] = _assign(conf['maxBoxLength'], u.kpc)
    conf['fieldOfView'] = _assign(conf['fieldOfView'], u.arcsec)
    conf['minStellarMass'] = _assign(conf['minStellarMass'], u.Msun)
    conf['maxStellarMass'] = _assign(conf['maxStellarMass'], u.Msun)
    conf['pixelScale'] = _assign(conf['pixelScale'], u.arcsec)
    conf['viewDistance'] = _assign(conf['viewDistance'], u.Mpc)
    conf['ageThreshold'] = _assign(conf['ageThreshold'], u.Myr)
    conf['PDRClearingTimescale'] = _assign(conf['PDRClearingTimescale'], u.Myr)
    conf['temperatureThreshold'] = _assign(conf['temperatureThreshold'], u.K)
    conf['cloudNumberDensity'] = _assign(conf['cloudNumberDensity'], u.cm**-3)
    
    conf['offsetParallel'] = _assign(conf['offsetParallel'], u.arcsec)
    conf['offsetPerpendicular'] = _assign(conf['offsetPerpendicular'], u.arcsec)
    conf['slitletSizePerpendicular'] = _assign(conf['slitletSizePerpendicular'], u.arcsec)
    conf['slitletSizeParallel'] = _assign(conf['slitletSizeParallel'], u.arcsec)
    conf['supportBarSize'] = _assign(conf['supportBarSize'], u.arcsec)
    conf['ditherSize'] = _assign(conf['ditherSize'], u.arcsec)
    conf['exposureTime'] = _assign(conf['exposureTime'], u.s)
    conf['rotate'] = _assign(conf['rotate'], u.deg)
    conf['obsRA'] = _assign(conf['obsRA'], u.deg)
    conf['obsDec'] = _assign(conf['obsDec'], u.deg)
    conf['darkCurrent'] = _assign(conf['darkCurrent'], u.s**-1)
    conf['aperture'] = _assign(conf['aperture'], u.m)
    conf['displayBoxSize'] = _assign(conf['displayBoxSize'], u.arcsec)
    
    return conf

def to_quantity_list(value):
    match = re.match(r"\[([^\]]+)\]\s*(.+)", value)    
    if match:
        values_str, unit_str = match.groups()
    else:
        raise Exception("Input format not recognized")

    # Parse numbers (split by spaces)
    values = np.fromstring(values_str, sep=" ")
    # return a list
    return values.tolist()

def read_properties(workingDir: str) -> dict:
        with open(workingDir + '/properties.json', 'r') as file:
            properties = json.load(file)
            
        for key in properties.keys():
            if isinstance(properties[key], dict):
                subkeys = properties[key].keys()
                if 'value' in subkeys and 'unit' in subkeys:
                    properties[key] = u.Quantity(properties[key]['value']).astype(np.float32) * u.Unit(properties[key]['unit'])
                
        return properties
    
def read_config(directory: str) -> dict:
    
    with open(directory + '/config.json', 'r') as file:
        configs = json.load(file)
    
    for key in configs.keys():
        if isinstance(configs[key], dict) and key != 'cosmology':
            subkeys = configs[key].keys()
            if 'value' in subkeys and 'unit' in subkeys:
                value = configs[key]['value']
                unit = configs[key]['unit']
                
                if value == 'Infinity':
                    value = np.inf
                elif value is None:
                    value = np.nan
                    
                configs[key] = u.Quantity(value, unit).astype(np.float32)
        else:
            configs[key] = configs[key]

    module = configs['cosmology']['cosmology']['module']
    module = importlib.import_module(module)
    cosmo = getattr(module, configs['cosmology']['cosmology']['name'])
    configs['cosmology']['cosmology'] = cosmo
    
    for key, value in configs['cosmology'].items():
        if isinstance(value, dict):
            subkeys = value.keys()
            if 'value' in subkeys and 'unit' in subkeys:
                value = configs['cosmology'][key]['value']
                unit = configs['cosmology'][key]['unit']
                value = u.Quantity(value, unit).astype(np.float32)
                configs['cosmology'][key] = value
                
    return configs

def read_config_corr(directory: str) -> dict:
    import importlib
    
    with open(directory + '/config.json', 'r') as file:
        configs = json.load(file)
    
    # Process non-cosmology Quantity objects
    for key in configs.keys():
        print(key)
        
        if isinstance(configs[key], dict) and key != 'cosmology':
            subkeys = configs[key].keys()
            if 'value' in subkeys and 'unit' in subkeys:
                value = configs[key]['value']
                unit = configs[key]['unit']
                
                if value is None:
                    value = np.nan
                
                if value == 'Infinity':
                    value = np.inf
                    
                configs[key] = u.Quantity(value, u.Unit(unit)).astype(np.float32)
    
    # Process cosmology: reconstruct class type
    if 'cosmology' in configs and isinstance(configs['cosmology'], dict):
        cosmo_dict = configs['cosmology']
        
        # Check if there's a nested 'cosmology' key with class info
        if 'cosmology' in cosmo_dict and isinstance(cosmo_dict['cosmology'], dict):
            class_info = cosmo_dict['cosmology']
            if '__type__' in class_info and class_info['__type__'] == 'class':
                module = importlib.import_module(class_info['module'])
                cosmo_class = getattr(module, class_info['name'])
                cosmo_dict['cosmology'] = cosmo_class
        
        # Recursively convert Quantity objects in cosmology dict
        def convert_quantities(obj):
            """Recursively convert Quantity dicts to Quantity objects."""
            if isinstance(obj, dict):
                # Check if it's a Quantity dict
                if 'value' in obj and 'unit' in obj and len(obj) == 2:
                    value = obj['value']
                    unit = obj['unit']
                    
                    # Handle array values
                    if isinstance(value, list):
                        value = np.array(value)
                    
                    return u.Quantity(value, unit)
                else:
                    # Recursively process nested dicts
                    return {k: convert_quantities(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_quantities(item) for item in obj]
            else:
                return obj
        
        # Convert all Quantity objects in cosmology
        configs['cosmology'] = convert_quantities(cosmo_dict)
    
    return configs

def read_json(file: str) -> dict:
    
    with open(file, 'r') as file:
        data = json.load(file)
    
    for key in data.keys():
        if isinstance(data[key], dict):
            subkeys = data[key].keys()
            if 'value' in subkeys and 'unit' in subkeys:
                value = data[key]['value']
                unit = data[key]['unit']
                
                if isinstance(value, list):
                    value_converted = []
                    for val in value:
                        if val is None:
                            value_converted.append(np.nan)
                        else:
                            value_converted.append(val)
                else:
                    if value is None:
                        value_converted = np.nan
                
                data[key] = u.Quantity(value_converted, u.Unit(unit)).astype(np.float32)
        else:
            if data[key] is None:
                data[key] = np.nan
            
    return data

def get_wave_for_emission_line(s):
    
    s = s.replace(" ", "")
    
    m = re.match(r"(.+?)([\d.]+)$", s)
    if not m:
        raise ValueError(f"Cannot parse '{s}'")
    name, wl = m.groups()
    return name, float(wl)

def lookup_table(units: Units):
    
    table = {
        "SubhaloCM": ["cm_x", "cm_y", "cm_z"], 
        "SubhaloGasMetallicity": "gasmetallicity",
        "SubhaloHalfmassRad": "halfmassrad",
        "SubhaloHalfmassRadType": ["halfmassrad_gas", "halfmassrad_dm", None, 
                                   None, "halfmassrad_stars", "halfmassrad_bhs"], 
        "SubhaloMass": "mass", 
        "SubhaloMassType": ["mass_gas", "mass_dm", None, None, "mass_stars", "mass_bhs"], 
        "SubhaloMassInHalfRad": "massinhalfrad",
        "SubhaloMassInHalfRadType": ["massinhalfrad_gas", "massinhalfrad_dm", None, 
                                    None, "massinhalfrad_stars", "massinhalfrad_bhs"],
        "SubhaloPos": ["pos_x", "pos_y", "pos_z"],
        "SubhaloSFR": "sfr",
        "SubhaloStarMetallicity": "starmetallicity",
        "SubhaloVel": ["vel_x", "vel_y", "vel_z"],
        "SubhaloVelDisp": "veldisp",
    }
    
    table_units = {
        "SubhaloCM": units.position, 
        "SubhaloGasMetallicity": u.dimensionless_unscaled,
        "SubhaloHalfmassRad": units.distance,
        "SubhaloHalfmassRadType": units.distance, 
        "SubhaloMass": units.mass, 
        "SubhaloMassType": units.mass, 
        "SubhaloMassInHalfRad": units.mass,
        "SubhaloMassInHalfRadType": units.mass,
        "SubhaloPos": units.position,
        "SubhaloSFR": units.sfr,
        "SubhaloStarMetallicity": u.dimensionless_unscaled,
        "SubhaloVel": u.Unit("km/s"),
        "SubhaloVelDisp": u.Unit("km/s"),
    }
    
    return table, table_units

def fage(cosmology: Cosmology) -> interp1d:
    z = np.linspace(0, 4, 1000)
    t = cosmology.age(z).to(u.Myr).value
    fage = interp1d(z, t, kind='cubic', 
                    bounds_error=False, fill_value='extrapolate')
    return fage