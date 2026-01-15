import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Cosmology
from scipy.interpolate import interp1d
import os
from skimage.transform import rescale, rotate, downscale_local_mean
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension
import numba
from .utils import read_config, galaxygenius_data_dir, setup_logging
from .utils import get_wave_for_emission_line, Units, read_json, fage
from itertools import product
from multiprocessing import shared_memory
import multiprocessing as mp
import time
from typing import Union
import h5py
from scipy.stats import binned_statistic_2d

try:
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
except RuntimeError:
    pass

class ParsecDimension(_Dimension):
    def __init__(self):
        super().__init__('pc')
        self.add_units('kpc', 1000)
    
class AngleDimension(_Dimension):
    def __init__(self):
        super().__init__(r'${^{\prime\prime}}$')
        self.add_units(r'${^{\prime}}$', 60)

class PostProcess:
    
    def __init__(self, subhaloID: int):
        
        self.subhaloID = subhaloID
        self.dataDir = galaxygenius_data_dir()
        # self.save_path = f'MSA_mock/Subhalo_{self.subhaloID}'
        # os.makedirs(self.save_path, exist_ok=True)
        
    
    def __init_paths(self):

        self.config = read_config(self.dataCubeDir)
        self.cosmology = Cosmology.from_format(self.config['cosmology'], format='mapping')
        self.config['cosmology'] = self.cosmology # convert to astropy cosmology object
        
        # limit the number of threads for numba
        numba.set_num_threads(self.config['numThreads'])
        
        # limit the number of threads for numpy
        os.environ['OMP_NUM_THREADS'] = str(self.config['numThreads'])
        os.environ['MKL_NUM_THREADS'] = str(self.config['numThreads'])
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.config['numThreads'])
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.config['numThreads'])
        
        self.subhalo_info = read_json(
            os.path.join(self.dataCubeDir, f'Subhalo_{self.subhaloID}.json')
        )
        self.logger = setup_logging(os.path.join(os.getcwd(), 'galaxyGeniusMSA.log'))
        
        self.cache_path = './cache'
        os.makedirs(self.cache_path, exist_ok=True)
        
        self.save_path = self.dataCubeDir.replace('dataCubes', 'MSA_mock')
        os.makedirs(self.save_path, exist_ok=True)
        
    def __init_stpsf(self):

        try:
            import stpsf
            import poppy
            
            # speed up PSF generation by using multiprocessing and fftw
            poppy.conf.use_multiprocessing = True
            poppy.conf.n_processes = self.config['numThreads']
            
            stpsf.setup_logging(level='DEBUG', filename=os.path.join(self.cache_path, 'stpsf.log'))
            
            nrs = stpsf.NIRSpec()
            nrs.image_mask = 'Single MSA open shutter' # MSA use step and dithers intead of multiple slits
            nrs.mode = 'imaging'
            nrs.disperser = self.config['disperser']
            nrs.filter = self.config['filter']
        
        except Exception as e:
            
            raise ValueError(f'Error getting PSF cube from STPSF: {e}')
        
        return nrs
        
    def __init_consts(self):
        
        pixel_scale = self.config['pixelScale'].to(u.arcsec)
        
        # in pixels, center offset for center
        self.shift_perpendicular = int(self.config['offsetPerpendicular'].to(u.arcsec) / pixel_scale)
        self.shift_parallel = int(self.config['offsetParallel'].to(u.arcsec) / pixel_scale)
        
        self.rescale_ratio_perpendicular = (pixel_scale / self.config['ditherSize'].to(u.arcsec)).value
        self.rescale_ratio_parallel = 1.
        
        # in arcsec
        self.size_perpendicular = (self.config['nDithers'] - 1) * self.config['ditherSize'] + \
            self.config['nSlitlets'] * self.config['slitletSizePerpendicular'] + \
                (self.config['nSlitlets'] - 1) * self.config['supportBarSize']
        self.size_parallel = self.config['nSteps'] * self.config['slitletSizeParallel']
        
        # in pixels ditherSize is 0.075, changed from original 0.1 arcsec
        self.n_pixels_perpendicular = int(self.size_perpendicular / self.config['ditherSize']) # dither size is pixel scale
        # considered pixel in parallel direction
        self.n_pixels_parallel = int(self.size_parallel / self.config['pixelScale'])
        
        # number of pixel in one slitlet in parallel direction
        self.n_pixels_slitlet_parallel = int((self.config['slitletSizeParallel'] \
            / self.config['pixelScale']).value)
        
        self.n_pixels_output_parallel = self.n_pixels_parallel // self.n_pixels_slitlet_parallel
        
        # 40, 18
        self.n_exposure_array = np.ones((
            self.n_pixels_perpendicular, 
            self.n_pixels_parallel,
        ), dtype=np.int32) * (self.config['nDithers'] - 1)
        
        
        # modify the exposure array for each dither
        for i in range(self.config['nDithers']):
            self.n_exposure_array[i, :] = i
            self.n_exposure_array[-i - 1, :] = i
            
        self.n_exposure_output = np.ones((
            self.n_pixels_perpendicular,
            self.n_pixels_output_parallel,
        ), dtype=np.int32) * (self.config['nDithers'] - 1)
        
        for i in range(self.config['nDithers']):
            self.n_exposure_output[i, :] = i
            self.n_exposure_output[-i - 1, :] = i
        
        self.begin_wave = self.config['minWaveOutput'].to(u.angstrom).value
        self.begin_wave = np.around(self.begin_wave, 2)
        
        self.end_wave = self.config['maxWaveOutput'].to(u.angstrom).value
        self.end_wave = np.around(self.end_wave, 2)
        
        # considered box length in pixels
        self.boxlength_in_pixel = int((self.config['displayBoxSize'] / self.config['pixelScale']).value)
        
        
    def __init_data(self):
        
        dispersion = os.path.join(
            self.dataDir, 
            f'nirspec/jwst_nirspec_{self.config["disperser"].lower()}_disp.fits'
        )
        
        with fits.open(dispersion) as f:
            
            wavelengths = f[1].data['wavelength'] * 10**4
            dlds = f[1].data['dlds'] * 10**4
            resolutions = f[1].data['r']
            
        self.interp_dispersion = interp1d(
            wavelengths,
            dlds, 
            kind='linear'
        )
        
        self.interp_resolution = interp1d(
            wavelengths,
            resolutions,
            kind='linear'
        )
        
        throughput = os.path.join(
            self.dataDir, 
            f'nirspec/jwst_nirspec_{self.config["filter"].lower()}_trans.fits'
        )
        
        with fits.open(throughput) as f:
            
            wavelengths = f[1].data['wavelength'] * 10**4
            throughputs = f[1].data['throughput']
            
        self.interp_throughput = interp1d(
            wavelengths,
            throughputs,
            kind='linear'
        )
        
        
    def input_bkg_interp(self, interp_bkg: interp1d):
        
        """
        Attach a background interpolation function to the PostProcess instance.

        This method allows the user to supply a precomputed background interpolation.

        Parameters
        ----------
        interp_bkg : interp1d
            Interpolated function of background vs. wavelength (e.g. created using scipy.interpolate.interp1d).
            Should accept an array of wavelengths (in angstrom) and return the background value(s) in MJy/sr.
        """
        
        self.input_bkg_called = True
        self.interp_bkg = interp_bkg
        
    def input_psf(self, psf_cube: np.ndarray):
        
        """
        This method allows the user to supply a precomputed PSF cube with the same oversample in the config.
        PSF cube should be in a 3-dimensional array with dimensions (number of wavelengths, number of pixels, number of pixels).
        The number of wavelengths should be the same as the number of wavelengths in the data cube.
        
        Parameters
        ----------
        psf_cube : np.ndarray
            PSF cube
        """
        
        self.input_psf_called = True
        self.input_psf_cube = psf_cube
        
    def input_wavelengths(self, wavelengths: np.ndarray):
        # for debug
        self.input_wavelengths_called = True
        self.input_waves = wavelengths
        
    def input_dataCube(self, dataCube_path: str, 
                       viewing_angle: Union[tuple[float, float], list[float, float], None]=None):
        
        """
        Load a data cube file and initialize PostProcess attributes.

        This method sets up the PostProcess instance to use a specific dataCube file,
        extracting metadata, initializing relevant paths, configuration,
        constants, and data required for further processing and analysis.

        Parameters
        ----------
        dataCube_path : str
            The file path to the data cube (output by SKIRT, in .fits format) to be loaded.
        viewing_angle : tuple[float, float] | list[float, float] | None, optional
            The inclination (in degree) and azimuth (in degree) viewing angles to be associated with this data cube.
            If None, the angles will be extracted from the filename and config.
        """
        
        self.dataCubeDir = os.path.dirname(dataCube_path)
        
        self.__init_paths()
        self.__init_consts()
        self.__init_data()
        
        if not os.path.exists(dataCube_path):
            raise FileNotFoundError(f'Data cube file {dataCube_path} does not exist.')
        
        if viewing_angle is not None:
            assert len(viewing_angle) == 2, \
                f'viewing angle should be in [inclination, azimuth] pair'
            
            self.viewing_angle = list(viewing_angle)
        else:
            basename = os.path.basename(dataCube_path)
            view_idx = int(basename.split('_')[2])
            self.viewing_angle = [self.config['inclinations'][view_idx],
                                  self.config['azimuths'][view_idx]]
            
        self.logger.info(f'Viewing at inclination: {self.viewing_angle[0]:.2f} deg, azimuth: {self.viewing_angle[1]:.2f} deg')
        
        with fits.open(dataCube_path) as f:

            self.dataCube = f[0].data
            
            if hasattr(self, 'input_wavelengths_called') and self.input_wavelengths_called:
                self.wavelengths = self.input_waves
            else:
                self.wavelengths = f[1].data['grid_points'] * 10**4 # in angstrom
        
        # slightly extend to avoid error on interp
        begin_wave = self.begin_wave - 20
        end_wave = self.end_wave + 20
        
        idx = np.where((self.wavelengths > begin_wave) & (self.wavelengths < end_wave))[0]
        
        self.wavelengths = self.wavelengths[idx]
        
        self.dataCube = self.dataCube[idx]
        
    def input_subhalo_info(self, subhalo_info: dict):
        
        """
        Add or update subhalo information.

        Parameters
        ----------
        subhalo_info : dict
            A dictionary containing subhalo-related properties or metadata to be associated
            with the object. These may include information such as mass_stars, vel_x, vel_y, vel_z, etc.
            
        """
        
        self.subhalo_info.update(subhalo_info)

    def __save_PSF_cube(self, psf_cube: np.ndarray, savefilename: str):
        
        
        hdulist = fits.HDUList()
        primary_hdu = fits.PrimaryHDU(data=psf_cube)
        header = primary_hdu.header
        header['oversamp'] = (self.config['oversample'], 'Oversampling factor')
        header['pixelsca'] = (self.config['pixelScale'].to(u.arcsec).value / self.config['oversample'],
                              'Pixel scale after oversampling, in arcsec')
        hdulist.append(primary_hdu)

        col = fits.Column(name='wavelengths', array=self.wavelengths, format='D')
        hdu_wavelengths = fits.BinTableHDU.from_columns([col])
        header = hdu_wavelengths.header
        header['unit'] = ('angstrom', 'Wavelength unit')
        hdulist.append(hdu_wavelengths)
        
        hdulist.writeto(savefilename, overwrite=True)
        
    def __save_PSF_bandpass(self, psf_bandpass: np.ndarray, savefilename: str):
        # un-used, bandpass image is not used
        hdulist = fits.HDUList()
        primary_hdu = fits.PrimaryHDU(data=psf_bandpass)
        header = primary_hdu.header
        header['filter'] = (self.config['displayBackgroundFilter'], 'Filter')
        header['oversamp'] = (self.config['oversample'], 'Oversampling factor')
        header['pixelsca'] = (self.config['pixelScale'].to(u.arcsec).value / self.config['oversample'],
                              'Pixel scale after oversampling, in arcsec')
        header['comment'] = ('PSF is rescaled to match dataCube', '')
        
        hdulist.append(primary_hdu)
        
        hdulist.writeto(savefilename, overwrite=True)
        
    def __psf_bandpass_from_stpsf(self) -> np.ndarray:
        # un-used, bandpass image is not used
        _, nrc = self.__init_stpsf()
        
        bandpass_psf = nrc.calc_psf(oversample=self.config['oversample'])
        
        psf_array = bandpass_psf[0].data
        psf_header = bandpass_psf[0].header
        psf_pixel_scale = psf_header['pixelscl'] # after oversampling
        
        # cube pixel scale after oversampling
        cube_pixel_scale = (self.config['pixelScale'].value / self.config['oversample'])
        ratio = psf_pixel_scale / cube_pixel_scale
        
        # rescale to match pixel scale with cube
        psf_array = rescale(psf_array, (ratio, ratio))
        
        return psf_array

    def __psf_from_stpsf(self) -> np.ndarray:
        
        nrs = self.__init_stpsf()
        
        waves = (self.wavelengths * u.angstrom).to_value(u.m)
        psf_cube = nrs.calc_datacube_fast(waves, oversample=self.config['oversample'])
        psf_cube = psf_cube[0].data
        
        return psf_cube
    
    def _check_size(self):
        
        """
        Check the size of the input PSF cube.

        This method verifies that the input PSF cube has the same number of wavelengths as the data cube.
        """
        
        assert len(self.input_psf_cube) == len(self.wavelengths), \
            f'PSF cube and wavelengths size do not match, {len(self.input_psf_cube)} vs. {len(self.wavelengths)}'
            
    def _check_existing_psf(self, wavelengths: np.ndarray, oversample: int):
        
        """
        Check if an existing PSF cube matches the required wavelengths and oversample.

        Parameters
        ----------
        wavelengths : np.ndarray
            The array of wavelengths from the existing PSF cube.
        oversample : int
            The oversample factor from the existing PSF cube.

        Returns
        -------
        bool
            True if the PSF cube matches the current configuration (wavelengths and oversample); False otherwise.
        """
        
        cond_1 = len(wavelengths) == len(self.wavelengths)
        cond_2 = np.allclose(wavelengths, self.wavelengths)
        cond_3 = oversample == self.config['oversample']
        
        if cond_1 and cond_2 and cond_3:
            return True
        else:
            return False    
    
    def _get_PSF_cube(self) -> np.ndarray:
        
        """
        Retrieve the PSF (Point Spread Function) cube.

        This method attempts to provide the PSF cube necessary for convolution or observation simulation.
        It checks, in order:
            1. If an input PSF cube was explicitly provided via input_psf (and validated).
            2. If a cached PSF cube matching the current configuration.
            3. Otherwise, generates a new PSF cube via the STPSF tool and saves it to the cache directory.

        Returns
        -------
        np.ndarray
            The PSF cube as a NumPy array matching the current wavelength grid and oversample.
        """
        
        if hasattr(self, 'input_psf_called') and self.input_psf_called:
            self._check_size()
            self.logger.info('Use input PSF cube.')
            return self.input_psf_cube
        
        psf_cube_path = os.path.join(self.cache_path, 'psf_cube.fits')
        if os.path.exists(psf_cube_path):
            with fits.open(psf_cube_path) as file:
                oversample = file[0].header['oversamp']
                wavelengths = file[1].data['wavelengths']
                
                if self._check_existing_psf(wavelengths, oversample):
                    self.logger.info(f'Use stored PSF cube in {psf_cube_path}.')
                    psf_cube = file[0].data
                else:
                    self.logger.info('PSF cube in cache does not match the configuration, getting new PSF cube from STPSF')
                    psf_cube = self.__psf_from_stpsf()
                    self.__save_PSF_cube(psf_cube, psf_cube_path)
        else:
            self.logger.info('No PSF cube in cache, getting new PSF cube from STPSF')
            psf_cube = self.__psf_from_stpsf()
            self.__save_PSF_cube(psf_cube, psf_cube_path)
        
        return psf_cube
    
    def _get_PSF_bandpass(self) -> np.ndarray:
        # un-used, bandpass image is not used
        if hasattr(self, 'input_psf_called') and self.input_psf_called and self.psf_bandpass is not None:
            self.logger.info('Use input bandpass PSF.')
            return self.psf_bandpass
        
        psf_bandpass_path = os.path.join(self.cache_path, 'psf_bandpass.fits')
        if os.path.exists(psf_bandpass_path):
            self.logger.info('Use cached bandpass PSF.')
            with fits.open(psf_bandpass_path, memmap=False) as file:
                psf_bandpass = file[0].data
        else:
            self.logger.info('Getting bandpass PSF from STPSF')
            psf_bandpass = self.__psf_bandpass_from_stpsf()
            self.__save_PSF_bandpass(psf_bandpass, psf_bandpass_path)
            
        return psf_bandpass
    
    @staticmethod
    def _shift_image_static(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
        
        """
        Shift an image by a given number of pixels in the y and x directions.

        Parameters
        ----------
        img : np.ndarray
            The image array to be shifted.
        dy : int
            The number of pixels to shift in the y direction.
        dx : int
            The number of pixels to shift in the x direction.

        Returns
        -------
        np.ndarray
            The shifted image array.
        """
        
        h, w = img.shape[:2]
        result = np.zeros_like(img)

        y1 = max(0, dy)
        y2 = h + min(0, dy)
        x1 = max(0, dx)
        x2 = w + min(0, dx)

        result[y1:y2, x1:x2] = img[y1 - dy:y2 - dy, x1 - dx:x2 - dx]
        return result
    
    @staticmethod
    def _transform_slice_static(
        cube_slice: np.ndarray, info_dict: dict) -> np.ndarray:
        """
        Transform a spatial slice of the data cube.

        Parameters
        ----------
        cube_slice : np.ndarray
            The spatial slice of the data cube to be transformed.
        info_dict : dict
            A dictionary containing information about the transformation.
        
        Returns
        -------
        np.ndarray
            The transformed spatial slice of the data cube.
        """
        n_pixels_perpendicular = info_dict['n_pixels_perpendicular']
        n_pixels_parallel = info_dict['n_pixels_parallel']
        rotation_angle = info_dict['rotation_angle']
        shift_perpendicular = info_dict['shift_perpendicular']
        shift_parallel = info_dict['shift_parallel']
        rescale_ratio_perpendicular = info_dict['rescale_ratio_perpendicular']
        rescale_ratio_parallel = info_dict['rescale_ratio_parallel']
        
        # make same odd or even
        if n_pixels_perpendicular % 2 != 0:
            if cube_slice.shape[0] % 2 == 0:
                cube_slice = cube_slice[:-1, :] # to odd
        else:
            if cube_slice.shape[0] % 2 != 0:
                cube_slice = cube_slice[:-1, :] # to even
            
        if n_pixels_parallel % 2 != 0:
            if cube_slice.shape[1] % 2 == 0:
                cube_slice = cube_slice[:, :-1] # to odd
        else:    
            if cube_slice.shape[1] % 2 != 0:
                cube_slice = cube_slice[:, :-1] # to even
                
        if rotation_angle != 0:
            try:
                
                cube_slice = rotate(
                    cube_slice, 
                    -rotation_angle,
                    resize=True,
                    mode='constant',
                    cval=0,
                )
            except:
                cube_slice = cube_slice.astype(cube_slice.dtype.newbyteorder('='))
                cube_slice = rotate(
                    cube_slice, 
                    -rotation_angle,
                    resize=True,
                    mode='constant',
                    cval=0,
                )
                
        if shift_perpendicular != 0 or shift_parallel != 0:
            cube_slice = PostProcess._shift_image_static(
                cube_slice, 
                -shift_perpendicular,
                -shift_parallel
            )
            
        cube_slice = rescale(
            cube_slice,
            (rescale_ratio_perpendicular, rescale_ratio_parallel), # y, x
            anti_aliasing=True,
        )
        
        return cube_slice
    
    @staticmethod
    def _process_single_spatial_slice_static(args):
        
        """
        Static method used for multiprocessing: processes a single spatial slice of the data cube.

        This method is designed to be called in parallel on multiple processes. It retrieves a data cube slice
        and the corresponding PSF slice from shared memory, applies the PSF convolution, downsamples and pads the slice,
        and performs geometric transformations (e.g., rotation, shift, rescale) specified by the info_dict.

        Parameters
        ----------
        args : tuple
            Contains all parameters required for processing:
                - i: The index of the spatial (wavelength) slice to process.
                - shm_cube_name: Name of the shared memory block for the data cube.
                - cube_shape: Shape of the data cube array.
                - cube_dtype: Data type of the data cube.
                - shm_psf_name: Name of the shared memory block for the PSF cube.
                - psf_shape: Shape of the PSF cube array.
                - psf_dtype: Data type of the PSF cube.
                - info_dict: Dictionary containing processing information, including oversample, 
                             n_pixels_perpendicular, n_pixels_parallel, and geometric transformation info.

        Returns
        -------
        np.ndarray
            The processed spatial slice cropped to the target size (n_pixels_perpendicular, n_pixels_parallel).
        """
        
        (i,
         shm_cube_name, cube_shape, cube_dtype,
         shm_psf_name, psf_shape, psf_dtype, 
         info_dict
         ) = args
        
        shm_cube = shared_memory.SharedMemory(name=shm_cube_name)
        cube_array = np.ndarray(cube_shape, dtype=cube_dtype, buffer=shm_cube.buf)
        
        shm_psf = shared_memory.SharedMemory(name=shm_psf_name)
        psf_array = np.ndarray(psf_shape, dtype=psf_dtype, buffer=shm_psf.buf)
        
        cube_slice = cube_array[i]
        psf_slice = psf_array[i]
    
        cube_slice = PostProcess._apply_PSF_slice_static(cube_slice, psf_slice, info_dict['num_threads'])
        cube_slice = PostProcess._downsample_static(cube_slice, info_dict['oversample'])
        cube_slice = PostProcess._padding_static(cube_slice, (100, 100))
        
        cube_slice = PostProcess._transform_slice_static(
            cube_slice, info_dict
        )
        
        n_pixels_perpendicular = info_dict['n_pixels_perpendicular']
        n_pixels_parallel = info_dict['n_pixels_parallel']
        
        slice_center_idx_perpendicular = cube_slice.shape[0] // 2
        slice_center_idx_parallel = cube_slice.shape[1] // 2
        
        slice_perpendicular = slice(
            slice_center_idx_perpendicular - n_pixels_perpendicular // 2,
            slice_center_idx_perpendicular + n_pixels_perpendicular // 2
        )
        
        slice_parallel = slice(
            slice_center_idx_parallel - n_pixels_parallel // 2,
            slice_center_idx_parallel + n_pixels_parallel // 2
        )
        container = cube_slice[slice_perpendicular, slice_parallel]
        return container
    
    @staticmethod
    @numba.njit(cache=True, fastmath=True)
    def _trapezoid_numba(y: np.ndarray, x: np.ndarray):
        
        """
        Implementation of the trapezoidal integration using Numba.

        Parameters
        ----------
        y : np.ndarray
            The values to be integrated.
        x : np.ndarray
            The sample points corresponding to the y values.

        Returns
        -------
        float
            The result of the trapezoidal integration.
        """
        
        result = 0.0
        for i in range(1, len(x)):
            dx = x[i] - x[i - 1]
            result += (y[i - 1] + y[i]) * dx / 2.0
        return result
    
    def _integrate_counts(self, values: np.ndarray, wave_bins: np.ndarray) -> u.Quantity:
        
        """
        Integrate the photon count rate over a specified wavelength bin.

        This method computes the number of detected electrons per second (count rate) 
        for a wavelength bin defined by `wave_bins` using the trapezoidal rule. 
        It converts the input surface brightness (in MJy/sr) to specific flux per unit wavelength 
        (in physical units), applies system throughput, then performs the integral:

            count_rate ∝ ∫ λ * Fλ * T(λ) dλ

        where Fλ is the spectral flux density per unit wavelength (converted to 
        erg/cm²/s/Å/sr), T(λ) is the system throughput, and λ is in Å.

        Returns
        -------
        astropy.units.Quantity
            Integrated photon count rate (in 1/s) for the wavelength bin.
        """
        
        # A_aper * l_p^2 / (h * c) * int_wavemin^wave_max lambda * flux * throughput * dlambda
        
        f_lam = values * u.MJy / u.sr * const.c / (wave_bins * u.angstrom)**2
        f_lam = f_lam.to(u.erg / u.cm**2 / u.s / u.angstrom / u.sr)
        f_lam_unit = f_lam.unit
        f_lam_val = f_lam.value
        
        throughput = self.interp_throughput(wave_bins)
        
        const_factor = np.pi * (self.config['aperture'] / 2)**2 * self.config['pixelScale']**2 / (const.h * const.c)
        # without t_exp and N_exp
        count_rate_value = self._trapezoid_numba(wave_bins * f_lam_val * throughput, wave_bins)
        count_rate = count_rate_value * const_factor * f_lam_unit * u.angstrom**2
        count_rate = count_rate.to(u.s**-1)
        return count_rate
    
    @staticmethod
    def _integrate_counts_static(values: np.ndarray, wave_bins: np.ndarray, 
                                 interp_throughput: interp1d,
                                 aperture: u.Quantity, pixel_scale: u.Quantity) -> u.Quantity:
        
        """
        Static method to perform the integration of photon counts over a given wavelength bin using the trapezoidal rule.

        Parameters
        ----------
        values : np.ndarray
            The flux values to be integrated, in MJy/sr.
        wave_bins : np.ndarray
            The wavelength bin edges in Angstroms.
        interp_throughput : scipy.interpolate.interp1d
            Interpolator for the system throughput as a function of wavelength.
        aperture : astropy.units.Quantity
            The aperture diameter (with units compatible with cm or m).
        pixel_scale : astropy.units.Quantity
            The pixel scale (with units compatible with arcsec or other length).

        Returns
        -------
        astropy.units.Quantity
            The integrated photon count rate in units of 1/s (per pixel).
        """
        
        # A_aper * l_p^2 / (h * c) * int_wavemin^wave_max lambda * flux * throughput * dlambda
        
        f_lam = values * u.MJy / u.sr * const.c / (wave_bins * u.angstrom)**2
        f_lam = f_lam.to(u.erg / u.cm**2 / u.s / u.angstrom / u.sr)
        f_lam_unit = f_lam.unit
        f_lam_val = f_lam.value
        
        throughput = interp_throughput(wave_bins)
        
        const_factor = np.pi * (aperture / 2)**2 * pixel_scale**2 / (const.h * const.c)
        # without t_exp and N_exp
        count_rate_value = PostProcess._trapezoid_numba(wave_bins * f_lam_val * throughput, wave_bins)
        count_rate = count_rate_value * const_factor * f_lam_unit * u.angstrom**2
        count_rate = count_rate.to(u.s**-1)
        return count_rate
    
    @staticmethod
    @numba.njit(
    "float32[:, :](float32[:, :, :], float32[:], float32[:])", 
    parallel=True, cache=True, fastmath=True)
    def integrate_bandpass(img, tran, wave):
        # not-used
        n = len(wave)
        h, w = img.shape[1], img.shape[2]
        out = np.zeros((h, w), dtype=np.float32)
        for i in numba.prange(h):
            for j in range(w):
                integral = 0.0
                for k in range(1, n):
                    y1 = img[k-1, i, j] * tran[k-1] * wave[k-1]
                    y2 = img[k, i, j] * tran[k] * wave[k]
                    dx = wave[k] - wave[k-1]
                    integral += (y1 + y2) / 2.0 * dx
                out[i, j] = integral
        return out
    
    # def _get_bandpass_image(self, wavelengths: np.ndarray, dataCube: np.ndarray, 
    #                         psf_bandpass: Union[np.ndarray, None]=None):
        
    #     filter = self.config['displayBackgroundFilter']
    #     throughput_file = os.path.join(self.dataDir, 'filters/JWST', f'{filter}.fil')
    #     if not os.path.exists(throughput_file):
    #         raise FileNotFoundError(f'{throughput_file} not found.')
    
    #     throughput = np.loadtxt(throughput_file)
    #     with open(throughput_file, 'r') as f:
    #         header = f.readline()
    #     if 'micron' in header or 'um' in header:
    #         throughput[:, 0] = throughput[:, 0] * 10**4 # in angstrom
    #     elif 'angstrom' in header:
    #         pass
    #     interp_bandpass_throughput = interp1d(
    #         throughput[:, 0], throughput[:, 1], kind='linear'
    #     )
    #     min_wave = np.min(throughput[:, 0])
    #     max_wave = np.max(throughput[:, 0])
    #     idx = np.where((wavelengths >= min_wave) & (wavelengths <= max_wave))[0]
        
    #     data_in = dataCube[idx] * u.MJy / u.sr
    #     wave_in = wavelengths[idx]
    #     trans_in = interp_bandpass_throughput(wave_in)
        
    #     f_lam = (data_in * const.c / (wave_in.reshape(-1, 1, 1) * u.angstrom)**2)
    #     f_lam = f_lam.to(u.erg / u.cm**2 / u.s / u.angstrom / u.sr)
    #     f_lam_unit = f_lam.unit
        
    #     exposureTime = self.config['exposureTime']
    #     numExp = 1
    #     areaMirror = np.pi * (self.config['aperture'] / 2)**2
        
    #     const_factor = exposureTime * numExp * areaMirror / (const.h * const.c)
    #     integral = self.integrate_bandpass(f_lam.value, trans_in, wave_in) * f_lam_unit * u.angstrom**2
    #     image_in_count = const_factor * integral * (self.config['pixelScale'] / self.config['oversample'])**2
    #     image_in_count = image_in_count.value
        
    #     pivot_numerator = self._trapezoid_numba(trans_in, wave_in) * u.angstrom
    #     pivot_denominator = self._trapezoid_numba(trans_in * wave_in ** -2, wave_in) * u.angstrom**-1
    #     pivot = np.sqrt(pivot_numerator / pivot_denominator)
        
    #     Jy_converter = self._trapezoid_numba(trans_in * wave_in, wave_in) * u.angstrom**2
    #     Jy_converter = Jy_converter / (const.h * const.c) * areaMirror\
    #                     * numExp * exposureTime
    #     Jy_converter = pivot**2 / const.c / Jy_converter
    #     Jy_converter = Jy_converter.to(u.Jy)
        
    #     if psf_bandpass is not None:
    #         psf_bandpass = psf_bandpass / np.sum(psf_bandpass)
    #         bandpass_image = convolve_fft(image_in_count, psf_bandpass)
    

    def _get_waves_by_resolution(self):
        
        """
        Generate an array of wavelengths corresponding to the instrument's spectral resolution.

        Returns
        -------
        np.ndarray
            Array of wavelength bin edges according to instrumental resolution, 
            spanning from self.begin_wave to self.end_wave. Each bin width is computed as
            delta_lambda = wavelength / resolution(wavelength), producing variable-width bins.

        Notes
        -----
        Used to bin spectra according to the spectral resolution as a function of wavelength.
        """
        
        # get waves corresponding to the resolution (output wavelengths for spectra)
        
        waves_corresponding_to_resolution = []
        begin_wave = self.begin_wave
        end_wave = self.end_wave
        
        waves_corresponding_to_resolution.append(begin_wave)

        while True:
            begin_wave_resol = begin_wave
            delta_wave = begin_wave_resol / self.interp_resolution(begin_wave_resol)
            end_wave_resol = begin_wave_resol + delta_wave
            
            waves_corresponding_to_resolution.append(end_wave_resol)
            
            if end_wave_resol > end_wave:
                break
            
            begin_wave = end_wave_resol
        
        waves_corresponding_to_resolution = np.array(waves_corresponding_to_resolution)
        return waves_corresponding_to_resolution
        
    def _get_waves_at_pixel(self):
        
        """
        Generate an array of wavelength bin edges corresponding to the detector pixels along the dispersion direction.

        Returns
        -------
        np.ndarray
            Array of wavelength bin edges in Angstroms, where each bin width is determined by the pixel dispersion
            at that wavelength according to the instrument configuration.
        """
        
        waves_at_pixel = []

        begin_wave = self.begin_wave
        end_wave = self.end_wave
        waves_at_pixel.append(begin_wave)

        while True:
            begin_wave_pixel = begin_wave
            delta_wave = self.interp_dispersion(begin_wave_pixel)
            end_wave_pixel = begin_wave_pixel + delta_wave
            
            waves_at_pixel.append(end_wave_pixel)
            
            if end_wave_pixel > end_wave:
                break
            
            begin_wave = end_wave_pixel

        waves_at_pixel = np.array(waves_at_pixel)
        return waves_at_pixel
    
    def _get_count_rates(self, wavelengths: np.ndarray, 
                         interp: interp1d) -> u.Quantity:
        
        """
        Compute the count rates (in electrons per second) for each wavelength bin, 
        using the given flux interpolator.

        For each wavelength bin defined by `wavelengths`, the method evaluates the 
        input interpolator (`interp`) over a set of wavelength samples within the bin, 
        then integrates to obtain the corresponding count rate using the detector and 
        instrument characteristics defined in `self`.

        Parameters
        ----------
        wavelengths : np.ndarray
            Array of wavelength bin edges in Angstroms. Each count rate will correspond 
            to a bin [wavelengths[i], wavelengths[i+1]].
        interp : scipy.interpolate.interp1d
            Interpolator function returning flux (in MJy/sr) for arbitrary wavelengths.

        Returns
        -------
        astropy.units.Quantity
            Array of count rates (electrons/s) for each bin. Units are 1/s.
        """
        
        count_rates = []
        for begin, end in zip(wavelengths[:-1], wavelengths[1:]):
            
            wave_rebins = np.linspace(begin, end, num=self.config['numSpecWaveBins'])
            
            flux = interp(wave_rebins)
            
            count_rate = self._integrate_counts(flux, wave_rebins)
            count_rates.append(count_rate)
            
        count_rates = u.Quantity(count_rates)
        return count_rates
    
    @staticmethod
    def _apply_PSF_slice_static(cube_slice: np.ndarray, psf_slice: np.ndarray, 
                                num_threads: int) -> np.ndarray:
        
        """
        Apply a PSF (Point Spread Function) convolution to a 2D spatial slice of the data cube (static method).
        If FFTW is available, use FFTW for faster convolution, otherwise use built-in convolution.

        Parameters
        ----------
        cube_slice : np.ndarray
            2D array representing a single spatial slice of the data cube.
        psf_slice : np.ndarray
            2D PSF kernel to convolve with the data. Does not need to have odd dimensions.
        num_threads : int
            Number of threads to use for the convolution.

        Returns
        -------
        np.ndarray
            The result of the 2D convolution (same shape as the input slice).
        """
        
        psf_slice = psf_slice / np.sum(psf_slice)
        
        try:
            import pyfftw
            from pyfftw.interfaces.numpy_fft import fftn, ifftn
            
            pyfftw.interfaces.cache.enable()
            pyfftw.config.NUM_THREADS = num_threads
            HAS_FFTW = True
        except:
            HAS_FFTW = False
            
        if HAS_FFTW:
            cube_slice = convolve_fft(cube_slice, psf_slice, fftn=fftn, ifftn=ifftn)
        else:
            cube_slice = convolve_fft(cube_slice, psf_slice)
        return cube_slice
    
    def _get_jwst_background(self, background_path: str):
        """
        Retrieve and interpolate the JWST background emission spectrum for the target coordinates.

        This method downloads or loads the background emission data (in MJy/sr vs. wavelength) 
        from the JWST Background Tool (JBT) for the RA, Dec, threshold, and thisday specified in self.config. 
        The background data is saved to a file at background_path and also loaded into 
        self.interp_bkg as an interpolator (wavelength in angstrom).

        Parameters
        ----------
        background_path : str
            The path where the background data file will be saved or loaded from.

        Raises
        ------
        ValueError
            If there is a problem downloading or parsing the background file via JBT.
        """
        try:
        
            from jwst_backgrounds import jbt
            
            ra = self.config['obsRA'].to(u.deg).value
            dec = self.config['obsDec'].to(u.deg).value
            wave = 4.4
            threshold = self.config['threshold']
            thisday = self.config['thisDay']
            
            jbt.get_background(ra, dec, wave, thresh=threshold, thisday=thisday, plot_background=False, 
                            plot_bathtub=False, write_bathtub=False,
                            write_background=True, background_file=background_path)
            
            bkg = np.loadtxt(background_path)
            wavelength = bkg[:, 0] * 10**4 # in angstrom
            flux = bkg[:, 1] # in MJy/sr
            
            self.interp_bkg = interp1d(
                wavelength, flux, kind='linear'
            )
        except Exception as e:
            raise ValueError(f'Error getting background emission file from JWST Background Tool (JBT): {e}')
    
    def _calc_bkg(self, wavelengths: np.ndarray) -> np.ndarray:
        """
        Calculate the background count rates for the given array of wavelengths.

        Parameters
        ----------
        wavelengths : np.ndarray
            Array of wavelengths (in Angstrom).

        Returns
        -------
        np.ndarray
            Array of background count rates corresponding to input wavelengths.
        """
        bkg_count_rates = []
        
        if self.config['useJBT']:
            
            suffix = f"{self.config['obsRA'].to_value(u.deg):.3f}_{self.config['obsDec'].to_value(u.deg):.3f}"
            suffix += f"_{self.config['threshold']:.3f}_{self.config['thisDay']}.txt"
            
            background_path = os.path.join(
                self.cache_path, 'background' + f'_{suffix}'
            )
            if os.path.exists(background_path):
                self.logger.info('Use cached background emission file.')
                bkg = np.loadtxt(background_path)
                wavelength = bkg[:, 0] * 10**4 # in angstrom
                flux = bkg[:, 1]
                self.interp_bkg = interp1d(
                    wavelength, flux, kind='linear'
                )
                
            else:
                self.logger.info('Getting background emission file from JWST Background Tool (JBT).')
                self._get_jwst_background(background_path)
        
        else:
            if hasattr(self, 'interp_bkg_called') and self.input_bkg_called:
                pass
            else:
                raise ValueError('Please call PostProcess.input_bkg() to provide a background interpolation.')
        
        bkg_count_rates = self._get_count_rates(wavelengths, self.interp_bkg)
        return bkg_count_rates
    
    @staticmethod
    @numba.njit(cache=True, fastmath=True, parallel=True)
    def _add_noise_numba(
        counts: np.ndarray,
        bkg_counts: np.ndarray,
        dark_counts: np.ndarray,
        read_out_noise: float,
        n_exposure: int
    ):
        
        """
        Add noise to simulated spectral data using Numba.

        Applies Poisson noise (from source + background + dark current) and Gaussian read-out noise
        to simulate detector noise over multiple exposures.

        Parameters
        ----------
        counts : np.ndarray
            Array of source counts per pixel.
        bkg_counts : np.ndarray
            Array of background counts per pixel.
        dark_counts : np.ndarray
            Array of dark current counts per pixel.
        read_out_noise : float
            Standard deviation of the read-out (Gaussian) noise per exposure.
        n_exposure : int
            Number of exposures.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Tuple containing:
                - counts with noise applied (np.ndarray)
                - estimated noise standard deviation for each pixel (np.ndarray)
        """
        
        ideal_counts = counts.copy()
        
        mean_noise_counts = np.zeros_like(counts)
        for i in numba.prange(len(counts)):
            mean_noise_counts[i] = bkg_counts[i] + dark_counts[i]
        
        # Poisson noise
        for i in numba.prange(len(counts)):
            counts[i] = np.random.poisson(counts[i])
        
        counts = counts + mean_noise_counts
        
        # Read noise for each exposure
        for exp in range(n_exposure):
            for i in numba.prange(len(counts)):
                read_noise = np.random.normal(0.0, read_out_noise)
                read_noise = np.round(read_noise)
                counts[i] += read_noise
        
        counts = counts - mean_noise_counts
        
        # Calculate noise counts
        noise_counts = np.zeros_like(counts)
        for i in numba.prange(len(counts)):
            noise_counts[i] = np.sqrt(
                ideal_counts[i] + bkg_counts[i] + dark_counts[i] + n_exposure * read_out_noise**2
            )
        
        return counts, noise_counts
    
    @staticmethod
    def _add_noise_static(spectrum_count_rates: u.Quantity, bkg_count_rates: u.Quantity,
                          n_exposure: int, exposure_time: u.Quantity,
                          dark_current: u.Quantity, read_out: float) -> tuple[np.ndarray, np.ndarray]:
        
        """
        Add noise to a spectrum given count rates, background, and detector parameters.

        This static method computes the total counts (including source, background, and dark current),
        applies Poisson (shot) noise and additive Gaussian read-out noise for each exposure, then
        returns the noisy counts along with an estimation of the noise for each pixel.

        Parameters
        ----------
        spectrum_count_rates : u.Quantity
            The signal count rates from the spectrum, with unit electrons/s.
        bkg_count_rates : u.Quantity
            The background count rates, with unit electrons/s.
        n_exposure : int
            Number of exposures.
        exposure_time : u.Quantity
            Duration of a single exposure, with unit seconds.
        dark_current : u.Quantity
            Dark current rate, with unit electrons/s.
        read_out : float
            Standard deviation of read-out (Gaussian) noise per exposure, typically in electrons.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            Tuple containing:
                - counts with noise applied (np.ndarray)
                - estimated noise standard deviation for each pixel (np.ndarray)
        """
        
        counts = spectrum_count_rates * n_exposure * exposure_time
        bkg_counts = bkg_count_rates * n_exposure * exposure_time
        dark_counts = dark_current * n_exposure * exposure_time
        
        dark_counts = np.full(bkg_counts.shape, 
                              dark_counts.value) * dark_counts.unit
        
        counts_val = counts.value
        bkg_counts_val = bkg_counts.value
        dark_counts_val = dark_counts.value
        readout_val = read_out
        
        counts_val, noise_counts_val = PostProcess._add_noise_numba(
            counts_val, bkg_counts_val, dark_counts_val,
            readout_val, n_exposure
        )
        
        return counts_val, noise_counts_val
    
    @staticmethod
    @numba.njit(cache=True, fastmath=True, parallel=True)
    def _rebin_fluxes_and_noise_numba(
        wavelengths_in: np.ndarray,
        fluxes_in: np.ndarray,
        noise_in_squared: np.ndarray,
        sed_in: np.ndarray,
        wavelengths_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Rebin flux and noise arrays (in quadrature) from the input wavelength bins to the output wavelength bins.

        Parameters
        ----------
        wavelengths_in : np.ndarray
            Input wavelength bin edges, shape (N+1,).
        fluxes_in : np.ndarray
            Flux per input bin, shape (N,).
        noise_in_squared : np.ndarray
            Noise variance per input bin, shape (N,).
        sed_in : np.ndarray
            SED counts per input bin, shape (N,).
        wavelengths_out : np.ndarray
            Output wavelength bin edges, shape (M+1,).

        Returns
        -------
        fluxes_out : np.ndarray
            Rebinned flux per output bin, shape (M,).
        noise_out : np.ndarray
            Rebinned noise per output bin, shape (M,).
        sed_out : np.ndarray
            Rebinned SED counts per output bin, shape (M,).
        """
        
        # wavelengths_out is wave edges
        fluxes_out = np.zeros(len(wavelengths_out) - 1, dtype=fluxes_in.dtype)
        noise_out = np.zeros(len(wavelengths_out) - 1, dtype=noise_in_squared.dtype)
        sed_out = np.zeros(len(wavelengths_out) - 1, dtype=sed_in.dtype)
        
        for i in numba.prange(len(wavelengths_out) - 1):
            w_start = wavelengths_out[i]
            w_end = wavelengths_out[i + 1]

            idx_start = np.searchsorted(wavelengths_in, w_start, side='right') - 1
            idx_end = np.searchsorted(wavelengths_in, w_end, side='left')
            
            idx_start = max(0, idx_start)
            idx_end = min(len(fluxes_in), idx_end)
            
            widths_in = w_end - w_start
            
            for j in range(idx_start, idx_end):
                
                p_start = wavelengths_in[j]
                p_end = wavelengths_in[j + 1]
                
                overlap_start = max(w_start, p_start)
                overlap_end = min(w_end, p_end)
                overlap_width = overlap_end - overlap_start
                
                if overlap_width > 0:
                    frac = overlap_width / widths_in
                    fluxes_out[i] += fluxes_in[j] * frac
                    noise_out[i] += noise_in_squared[j] * frac
                    sed_out[i] += sed_in[j] * frac
                    
        for i in numba.prange(len(noise_out)):
            noise_out[i] = np.sqrt(noise_out[i])
        
        return fluxes_out, noise_out, sed_out
    
    def _rebin_fluxes_and_noise_array(
        self,
        wavelengths_in: np.ndarray,
        fluxes_in_array: np.ndarray,
        noise_in_array: np.ndarray,
        sed_in_array: np.ndarray,
        wavelengths_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Rebins an array of fluxes and noise to new wavelength bins.

        Parameters
        ----------
        wavelengths_in : np.ndarray
            Input array of wavelength bin edges, shape (N+1,).
        fluxes_in_array : np.ndarray
            Input flux array, shape (A, B, N).
        noise_in_array : np.ndarray
            Input noise array, shape (A, B, N).
        sed_in_array : np.ndarray
            Input SED counts array, shape (A, B, N).
        wavelengths_out : np.ndarray
            Output array of wavelength bin edges, shape (M+1,).

        Returns
        -------
        fluxes_out_array : np.ndarray
            Rebinned flux, shape (A, B, M).
        noise_out_array : np.ndarray
            Rebinned noise, shape (A, B, M).
        sed_out_array : np.ndarray
            Rebinned SED counts, shape (A, B, M).
        """
        
        new_shape = (fluxes_in_array.shape[0], fluxes_in_array.shape[1], len(wavelengths_out) - 1)
        
        fluxes_out_array = np.zeros(new_shape, dtype=fluxes_in_array.dtype)
        noise_out_array = np.zeros(new_shape, dtype=noise_in_array.dtype)
        sed_out_array = np.zeros(new_shape, dtype=sed_in_array.dtype)
        
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                
                fluxes_out, noise_out, sed_out = self._rebin_fluxes_and_noise_numba(
                    wavelengths_in, fluxes_in_array[i, j], noise_in_array[i, j]**2,
                    sed_in_array[i, j], wavelengths_out)
                fluxes_out_array[i, j] = fluxes_out
                noise_out_array[i, j] = noise_out
                sed_out_array[i, j] = sed_out
        return fluxes_out_array, noise_out_array, sed_out_array

    
    def _convert_to_flux(self, wavelengths: np.ndarray, counts: np.ndarray, noise_counts: np.ndarray,
                         sed_counts: np.ndarray, n_exposure: int, exposure_time: u.Quantity):
        
        """
        Converts electron counts and their noise to physical flux units.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength bin edges, shape (N+1,).
        counts : np.ndarray
            Electron counts per bin, shape (N,).
        noise_counts : np.ndarray
            Noise in electron counts per bin, shape (N,).
        sed_counts : np.ndarray
            SED counts per bin, shape (N,).
        n_exposure : int
            Number of exposures for this pixel.
        exposure_time : astropy.units.Quantity
            Exposure time per exposure.

        Returns
        -------
        fluxes : np.ndarray
            Flux per bin in the desired unit flux_lambda or flux_nu, shape (N,).
        noise_fluxes : np.ndarray
            Noise per bin in the desired unit flux_lambda or flux_nu, shape (N,).
        sed_fluxes : np.ndarray
            SED flux per bin in the desired unit, shape (N,).
        """
        
        delta_waves = np.diff(wavelengths)
        
        middle_waves = (wavelengths[1:] + wavelengths[:-1]) / 2
        efficiency = self.interp_throughput(middle_waves)
        
        energy_per_photon = (const.h * const.c) / (middle_waves * u.angstrom)
        
        sensitivity = np.pi * (self.config['aperture'] / 2)**2 * n_exposure * exposure_time *\
            delta_waves * u.angstrom * efficiency
        
        flux_lam = (counts * energy_per_photon) / sensitivity
        noise_flux_lam = (noise_counts * energy_per_photon) / sensitivity
        sed_flux_lam = (sed_counts * energy_per_photon) / sensitivity
        
        if self.config['unit'] == 'flux_lambda':
            
            flux_lam = flux_lam.to(u.erg / u.cm**2 / u.s / u.angstrom)
            noise_flux_lam = noise_flux_lam.to(u.erg / u.cm**2 / u.s / u.angstrom)
            sed_flux_lam = sed_flux_lam.to(u.erg / u.cm**2 / u.s / u.angstrom)
            
            return flux_lam.value, noise_flux_lam.value, sed_flux_lam.value
        
        elif self.config['unit'] == 'flux_nu':
            
            flux_nu = (middle_waves * u.angstrom)**2 * flux_lam / const.c
            noise_flux_nu = (middle_waves * u.angstrom)**2 * noise_flux_lam / const.c
            sed_flux_nu = (middle_waves * u.angstrom)**2 * sed_flux_lam / const.c
            
            flux_nu = flux_nu.to(u.Jy)
            noise_flux_nu = noise_flux_nu.to(u.Jy)
            sed_flux_nu = sed_flux_nu.to(u.Jy)
            
            return flux_nu.value, noise_flux_nu.value, sed_flux_nu.value
        
    def _convert_to_flux_array(self, wavelengths: np.ndarray, counts_array: np.ndarray,
                               noise_counts_array: np.ndarray, sed_counts_array: np.ndarray):
        
        """
        Converts an array of electron counts and noise counts to fluxes and noise fluxes per bin
        using the instrument and observation parameters.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength edges per bin, shape (N+1,).
        counts_array : np.ndarray
            Electron counts per bin, shape (M, K, N).
        noise_counts_array : np.ndarray
            Noise in electron counts per bin, shape (M, K, N).
        sed_counts_array : np.ndarray
            SED counts per bin, shape (M, K, N).
            
        Returns
        -------
        fluxes_array : np.ndarray
            Flux per bin in the desired unit, shape (M, K, N-1).
        noise_fluxes_array : np.ndarray
            Noise per bin in the desired unit, shape (M, K, N-1).
        sed_fluxes_array : np.ndarray
            SED flux per bin in the desired unit, shape (M, K, N-1).
        """
        
        fluxes_array = np.zeros_like(counts_array)
        noise_fluxes_array = np.zeros_like(noise_counts_array)
        sed_fluxes_array = np.zeros_like(sed_counts_array)
        
        for i, j in product(range(counts_array.shape[0]), range(counts_array.shape[1])):
            
            n_exposure = self.n_exposure_array[i, j]
            
            if n_exposure == 0:
                fluxes = np.zeros_like(counts_array[i, j])
                noise_fluxes = np.ones_like(counts_array[i, j]) * np.nan
                fluxes_array[i, j] = fluxes
                noise_fluxes_array[i, j] = noise_fluxes
                sed_fluxes = np.zeros_like(counts_array[i, j])
                sed_fluxes_array[i, j] = sed_fluxes
                continue
            
            exposure_time = self.config['exposureTime']
            fluxes, noise_fluxes, sed_fluxes = self._convert_to_flux(
                wavelengths, counts_array[i, j], noise_counts_array[i, j], sed_counts_array[i, j],
                n_exposure, exposure_time
            )
            fluxes_array[i, j] = fluxes
            noise_fluxes_array[i, j] = noise_fluxes
            sed_fluxes_array[i, j] = sed_fluxes
        
        return fluxes_array, noise_fluxes_array, sed_fluxes_array
        
    def _save_dataTensor(self, dataTensor: np.ndarray, savefilename: str):
        
        """
        Save the given data tensor as a FITS file with relevant metadata in the header.

        Parameters
        ----------
        dataTensor : np.ndarray
            The data array to be saved, typically in unit of flux or electron counts.
        savefilename : str
            The filename to save the FITS file to.
        """
        
        hdulist = fits.HDUList()
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        unit_type = self.config['unit']
        unit_comment_dict = {
            'electron': 'Unit of image, in electron counts',
            'flux_lambda': 'Unit of image, in erg / cm^2 / s / angstrom',
            'flux_nu': 'Unit of image, in Jy',
        }
        unit_comment = unit_comment_dict[unit_type]
        
        header = fits.Header()
        header['SNAPNUM'] = (self.config['snapNum'], 'Snapshot ID')
        header['ID'] = (self.config['subhaloID'], 'Subhalo ID')
        header['MASS'] = (np.log10(self.config['stellarMass'].value), 
                          'Subhalo stellar mass, in log10 scale (Msun)')
        header['INSTRU'] = ('NIRSpec MSA', 'Instrument')
        header['DISP'] = (self.config['disperser'], 'Disperser')
        header['FILTER'] = (self.config['filter'], 'Filter')
        header['APERTURE'] = (self.config['aperture'].value, 'Aperture size, in meter')
        header['UNIT'] = (self.config['unit'], unit_comment)
        header['REDSHIFT'] = (self.config['viewRedshift'], 'Viewing redshift')
        if self.config['inLocal']:
            cosmology = 'LocalUniverseCosmology'
        else:
            cosmology = 'Planck15'
        header['COSMO'] = (cosmology, 'Cosmology')
        header['BOXSIZE'] = (self.config['boxLength'].to(u.pc).value, 'Box size, in pc')
        header['FOV'] = (self.config['fieldOfView'].to(u.arcsec).value, 'Field of view, in arcsec')
        header['PHYFOV'] = (self.config['fovSize'].to(u.pc).value, 'Physical field of view, in pc')
        header['LUMIDIS'] = (self.config['lumiDis'].to(u.Mpc).value, 'Luminosity distance, in Mpc')
        header['PIXELSC'] = (self.config['pixelScale'].to(u.arcsec).value, 'Detector pixel scale, in arcsec')
        slitsize = f'{self.config["slitletSizeParallel"].value:.2f},{self.config["slitletSizePerpendicular"].value:.2f}'
        header['SLITSIZE'] = (slitsize, 'slitlet size, in arcsec (para, perp)')
        header['BARSIZE'] = (self.config['supportBarSize'].to(u.arcsec).value, 'Support bar size, in arcsec')
        header['DITHERSI'] = (self.config['ditherSize'].to(u.arcsec).value, 'Dither size, in arcsec')
        header['NDITHERS'] = (self.config['nDithers'], 'Number of dithers')
        header['NSLITLET'] = (self.config['nSlitlets'], 'Number of slitlets')
        header['NSTEPS'] = (self.config['nSteps'], 'Number of steps')
        header['ROTATE'] = (self.config['rotate'].to(u.deg).value, 'Rotation angle, in deg')
        offset = f'{self.config["offsetParallel"].value:.2f},{self.config["offsetPerpendicular"].value:.2f}'
        header['OFFSET'] = (offset, 'Offset, in arcsec (para, perp)')
        hdu_data = fits.ImageHDU(data=dataTensor, header=header)
        hdulist.append(hdu_data)
        
        hdu_exposure = fits.ImageHDU(data=self.n_exposure_array)
        hdulist.append(hdu_exposure)
        
        hdulist.writeto(savefilename, overwrite=True)
        
    def _assemble_data(self, wavelengths: np.ndarray,
                       signal_array: np.ndarray,
                       noise_array: np.ndarray,
                       sed_array: np.ndarray):
        
        """
        Assemble all spectral data to a data tensor.
        
        Parameters
        ----------
        wavelengths : ndarray
            Array of wavelength bin edges, with shape (N+1,).
        signal_array : ndarray
            Measured signal for each slitlet.
        noise_array : ndarray
            Corresponding noise array, same shape as signal_array.
        sed_array : ndarray
            Corresponding SED counts array, same shape as signal_array.
        Returns
        -------
        dataTensor : ndarray
            Data array with shape (n_slitlet_y, n_slitlet_x, 3, n_bins), where:
              - data[...,0,:] = wavelengths (middle points)
              - data[...,1,:] = signal (flux or electron counts)
              - data[...,2,:] = noise (flux or electron counts)
              - data[...,3,:] = SED counts (flux or electron counts)
        """
        
        # middle wavelength of the bin
        wavelengths = (wavelengths[1:] + wavelengths[:-1]) / 2
        
        dataTensor = np.zeros((
            signal_array.shape[0],
            signal_array.shape[1],
            4, 
            signal_array.shape[-1]
        ))
        
        for i in range(signal_array.shape[0]):
            for j in range(signal_array.shape[1]):
                dataTensor[i, j, 0] = wavelengths
                dataTensor[i, j, 1] = signal_array[i, j]
                dataTensor[i, j, 2] = noise_array[i, j]
                dataTensor[i, j, 3] = sed_array[i, j]
                
        return dataTensor
    
    @staticmethod
    def _process_single_spectrum_worker(args):
        
        """
        Worker function for multiprocessing: processing a single 1D spectrum from the MSA datacube.
        
        This method is designed to be called in parallel on multiple processes. It retrieves a single 1D spectrum
        from shared memory, integrates the counts over the wavelength bins, adds noise, and returns the counts and noise.
        
        Parameters
        ----------
        args : tuple
            Tuple containing all necessary parameters and shared memory references for 
            processing an individual spectrum. Expected elements:
                - i : int
                    Y-index in the MSA grid.
                - j : int
                    X-index in the MSA grid.
                - shm_msa_name : str
                    Shared memory name for the MSA datacube.
                - msa_shape : tuple
                    Shape of the MSA datacube array.
                - msa_dtype : dtype
                    Data type of MSA datacube array.
                - shm_exp_name : str
                    Shared memory name for the exposure map.
                - exp_shape : tuple
                    Shape of the exposure array.
                - exp_dtype : dtype
                    Data type of the exposure array.
                - bkg_count_rates : np.ndarray
                    Array of background count rates.
                - wavelengths : np.ndarray
                    Array of wavelengths of the source spectrum.
                - interp_throughput : function
                    Interpolation function for throughput vs wavelength.
                - waves_at_pixel : np.ndarray
                    Edges of wavelength bins for rebinned spectral pixels.
                - config_dict : dict
                    Configuration parameters (exposure time, aperture, pixel scale, etc).

        Returns
        -------
        tuple
            (i, j, spectrum_counts, noise_counts, sed_counts)
            where spectrum_counts and noise_counts are arrays per rebinned pixel.
        """
        
        (i, j, 
         shm_msa_name, msa_shape, msa_dtype,
         shm_exp_name, exp_shape, exp_dtype,
         bkg_count_rates, wavelengths, interp_throughput,
         waves_at_pixel, config_dict) = args
        
        shm_msa = shared_memory.SharedMemory(name=shm_msa_name)
        msa_array = np.ndarray(msa_shape, dtype=msa_dtype, buffer=shm_msa.buf)
        
        shm_exp = shared_memory.SharedMemory(name=shm_exp_name)
        n_exposure_array = np.ndarray(exp_shape, dtype=exp_dtype, buffer=shm_exp.buf)
        
        spectrum = msa_array[:, i, j]
        n_exposure = n_exposure_array[i, j]
        
        if n_exposure == 0:
            # signal and noise are both 0
            spectrum_counts = np.zeros_like(bkg_count_rates)
            noise_counts = np.zeros_like(bkg_count_rates)
            sed_counts = np.zeros_like(bkg_count_rates)
            return i, j, spectrum_counts, noise_counts, sed_counts
            
        
        exposure_time = u.Quantity(config_dict['exposureTime']['value'], 
                                   config_dict['exposureTime']['unit'])
        
        interp_spectrum = interp1d(wavelengths, spectrum, kind='linear', fill_value='extrapolate')
        
        aperture = u.Quantity(config_dict['aperture']['value'], 
                              config_dict['aperture']['unit'])
        pixel_scale = u.Quantity(config_dict['pixelScale']['value'], 
                                 config_dict['pixelScale']['unit'])
        
        count_rates = []
        for begin, end in zip(waves_at_pixel[:-1], waves_at_pixel[1:]):
            wave_rebins = np.linspace(begin, end, config_dict['numSpecWaveBins'])
            
            flux = interp_spectrum(wave_rebins)
            count_rate = PostProcess._integrate_counts_static(flux, wave_rebins, 
                                                              interp_throughput,
                                                              aperture, pixel_scale)
            count_rates.append(count_rate)
            
        count_rates = u.Quantity(count_rates)
        spectrum_count_rates = count_rates

        idx = spectrum_count_rates < 0
        spectrum_count_rates[idx] = 0 * u.s**-1
        
        # ideal SED counts
        sed_counts = spectrum_count_rates * n_exposure * exposure_time
        sed_counts = sed_counts.value
        
        dark_current = u.Quantity(config_dict['darkCurrent']['value'], 
                                  config_dict['darkCurrent']['unit'])
        read_out = config_dict['readOut']
        
        spectrum_counts, noise_counts = PostProcess._add_noise_static(
            spectrum_count_rates, bkg_count_rates, 
            n_exposure, exposure_time,
            dark_current, read_out
        )
        
        return i, j, spectrum_counts, noise_counts, sed_counts
    

    def _display_exposures(self, displaySpectrumAtPixel: list, save_path: str):
        
        """
        Display the 2D exposure map and highlight the pixels specified in 'displaySpectrumAtPixel'.
        This function visualizes the number of exposures for each pixel using a colored grid,
        and marks selected pixels specified by 'displaySpectrumAtPixel' with scatter dots.
        The plot is saved as 'MSA_exposures.png' in 'save_path'.

        Args:
            displaySpectrumAtPixel (list): List of (y, x) tuples indicating pixels to highlight.
            save_path (str): Directory path where the generated plot will be saved.

        """
        
        ny, nx = self.n_exposure_output.shape
        fig, ax = plt.subplots(figsize=(6, 6))
        
        norm = colors.Normalize(vmin=0, vmax=6)
        cmap = cm.viridis
        
        for y in range(ny):
            for x in range(nx):
                n_exposure = self.n_exposure_output[y, x]
                facecolor = cmap(norm(n_exposure))
                
                rect = patches.Rectangle(
                    (x, y),
                    width=1,
                    height=1,
                    facecolor=facecolor,
                    edgecolor="k",
                    linewidth=0.5
                )
                ax.add_patch(rect)
                
        ax.set_xlim(0, nx)
        ax.set_ylim(0, ny)
        ax.set_aspect("auto")
        
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)
        fig.colorbar(sm, ax=ax, label=r'$n_{\rm exp}$')
        
        for i, (y_pix, x_pix) in enumerate(displaySpectrumAtPixel):
            x = ((x_pix + 1) + x_pix) / 2
            y = ((y_pix + 1) + y_pix) / 2
            ax.scatter(x, y, s=20)
        
        ax.set_xticks([x + 0.5 for x in range(nx)])
        
        y_tick_indices = np.linspace(0, ny - 1, num=10, dtype=int)
        ax.set_yticks([y + 0.5 for y in y_tick_indices])
        
        ax.set_xticklabels([str(x) for x in range(nx)])
        ax.set_yticklabels([str(y) for y in y_tick_indices])
        
        filename = os.path.join(save_path, 'MSA_exposures.png')
        plt.savefig(filename)
        plt.close()
                    
    
    def _display_spectra(self, dataTensor: np.ndarray,
                         displaySpectrumAtPixel: list,
                         displayEmissionLines: list, 
                         save_path: str):
        
        """
        Display and save emission spectra at selected MSA pixels.
        Each selected pixel's spectrum will be shown with flux and noise, 
        with the specified emission lines marked, and the plot saved as 'MSA_spectra.png' in 'save_path'.

        Parameters
        ----------
        dataTensor : np.ndarray
            Observation data of shape [ny, nx, 3] where the last dimension 
            represents (wavelength, flux, noise) arrays at each pixel.
        displaySpectrumAtPixel : list
            List of (y, x) pixel tuples to display spectra for.
        displayEmissionLines : list
            List of emission line identifiers to mark on spectra.
        save_path : str
            Directory where the output plots will be saved.

        """
        
        el_names = []
        el_waves = []
        for el in displayEmissionLines:
            name, wave = get_wave_for_emission_line(el)
            wave = wave * (1 + self.config['viewRedshift'])
            if wave < self.begin_wave or wave > self.end_wave:
                self.logger.warning(f'Emission line {el} is out of the wavelength range.')
                continue
            el_names.append(name)
            el_waves.append(wave)
        
        num_spectra = len(displaySpectrumAtPixel)
        
        unit_dict = {'electron': 'electron',
                     'flux_lambda': 'erg / cm^2 / s / angstrom',
                     'flux_nu': 'Jy'}
        unit = unit_dict[self.config['unit']]
        
        fig, axs = plt.subplots(num_spectra, 1, figsize=(8, 4 * num_spectra))
        for i, position in enumerate(displaySpectrumAtPixel):
            
            y_pix = position[0]
            x_pix = position[1]
            n_exp = self.n_exposure_output[y_pix, x_pix]
            
            wave = dataTensor[y_pix, x_pix, 0]
            wave_begin = np.min(wave)
            wave_end = np.max(wave)
            
            axs[i].plot(
                dataTensor[y_pix, x_pix, 0], 
                dataTensor[y_pix, x_pix, 1], 
                label='Spectrum'
                )
            # axs[i].plot(
            #     dataTensor[y_pix, x_pix, 0],
            #     dataTensor[y_pix, x_pix, 3],
            #     label='SED',
            #     linestyle='--'
            # )
            # Plot error on a secondary y-axis on the right
            ax_err = axs[i].twinx()
            ax_err.plot(
                dataTensor[y_pix, x_pix, 0],
                dataTensor[y_pix, x_pix, 2],
                color='tab:orange',
                label='Error'
            )
            ax_err.set_ylabel(fr'Error [${unit}$]', color='tab:orange')
            ax_err.tick_params(axis='y', labelcolor='tab:orange')
            
            y_min_spec, y_max_spec = axs[i].get_ylim()
            y_min_err, y_max_err = ax_err.get_ylim()
            
            y_min = min(y_min_spec, y_min_err)
            y_max = max(y_max_spec, y_max_err)
            
            y_pos = y_max - 0.1 * (y_max - y_min)
            for name, wave in zip(el_names, el_waves):
                axs[i].axvline(wave, color='r', linestyle='--')
                axs[i].text(wave + 50, y_pos, name, 
                            rotation=90, ha='left', va='bottom')
            
            axs[i].set_xlabel(r'Wavelength [$\AA$]')
            axs[i].set_ylabel(fr'flux [${unit}$]')
            axs[i].set_xlim(wave_begin, wave_end)
            axs[i].set_ylim(0, None)
            t_exp = self.config['exposureTime'].to_value(u.minute)
            axs[i].set_title(fr'Spectrum {i} at pixel ({y_pix}, {x_pix}) ($t_{{\rm exp}}$ = ${n_exp} \times {t_exp}$ min)')
            # axs[i].legend(frameon=False)
        
        plt.tight_layout()
        filename = os.path.join(
            save_path, 'MSA_spectra.png'
        )
        plt.savefig(filename)
        plt.close()
                    
    
    def _display_MSA_obs(self, dataTensor: np.ndarray,
                         cube_display: np.ndarray, 
                         slice_wavelength: np.ndarray,
                         displaySpectrumAtPixel: list,
                         save_path: str):
        
        """
        Displays the illstration of the mock observation for MSA. 
        The function overlays markers on the cube_display image to show where the spectra in displaySpectrumAtPixel
        were extracted from, and labels them for reference. The image is saved to the specified save_path.
    
        Parameters
        ----------
        dataTensor : np.ndarray
            Observation data of shape [ny, nx, 3] where the last dimension 
            represents (wavelength, flux, noise) arrays at each pixel.
        cube_display : np.ndarray
            Background image to be displayed.
        slice_wavelength : np.ndarray
            Wavelength of the slice to display.
        displaySpectrumAtPixel : list
            List of (y, x) pixel tuples to display spectra for.
        save_path : str
            Directory where the output plots will be saved.

        """
        
        y_center_tensor = dataTensor.shape[0] // 2
        x_center_tensor = dataTensor.shape[1] // 2
        
        y_deviations = []
        x_deviations = []
        
        for i, position in enumerate(displaySpectrumAtPixel):
            
            y_pix = position[0]
            x_pix = position[1]
            
            y_deviations.append(y_pix - y_center_tensor)
            x_deviations.append(x_pix - x_center_tensor)
            
        y_deviations = np.array(y_deviations)
        x_deviations = np.array(x_deviations)
        
        y_deviations = y_deviations * (self.config['ditherSize'] / self.config['pixelScale']).value
        x_deviations = x_deviations * self.n_pixels_slitlet_parallel
        
        y_center_cube_slice = cube_display.shape[0] // 2
        x_center_cube_slice = cube_display.shape[1] // 2
        
        y_coordinates = y_center_cube_slice + y_deviations
        x_coordinates = x_center_cube_slice + x_deviations
        
        num_pixels = cube_display.shape[1]
        
        # rotate first and them shift
        # shift the plane instead of the slitlets
        cube_display = rotate(cube_display.astype(cube_display.dtype.newbyteorder('=')),
                              -self.config['rotate'].to_value(u.deg), 
                              resize=False, mode='constant', cval=np.nan)
        cube_display = PostProcess._shift_image_static(
            cube_display, 
            -self.shift_perpendicular,
            -self.shift_parallel)
        
        if self.config['viewRedshift'] is not None:
            z = str(np.around(self.config['viewRedshift'], 2))
        else:
            z = '0.00'
            
        if 'stellarMass' in self.config and self.config['stellarMass'] is not None:
            logM = str(np.around(np.log10(self.config['stellarMass'].value), 1))
        else:
            logM = 'N/A'
        pixelscale = self.config['pixelScale'].to(u.arcsec).value
        resolution = self.config['resolution'].to(u.pc).value # the physical resol should also be added

        scalebarSize = 0.25 * num_pixels * resolution
        if scalebarSize > 1000:
            scalebarUnit = 'kpc'
            scalebarSize = np.around(scalebarSize / 1000, 2)
            ps = resolution / 1000
        else:
            scalebarUnit = 'pc'
            scalebarSize = np.around(scalebarSize, 2)
            ps = resolution
            
        pc_dim = ParsecDimension()
        scalebar = ScaleBar(ps, scalebarUnit, dimension=pc_dim, 
                            fixed_value=scalebarSize, fixed_units=scalebarUnit, frameon=False,
                            location='lower right', scale_loc='top',
                            color='red', font_properties={'size': 12})
        
        fig, ax = plt.subplots(figsize=(6, 6))
        im = ax.imshow(np.arcsinh(cube_display), cmap='gray_r', origin='lower')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, label='Asinh(Flux) [MJy/sr]')

        for i, (y_coord, x_coord) in enumerate(zip(y_coordinates, x_coordinates)):
            ax.scatter(x_coord, y_coord, s=10, label=f'Spectrum {i}')
            
        ax.legend(frameon=True, loc='upper right')

        tick_min = -self.config['displayBoxSize'].value / 2
        tick_max = self.config['displayBoxSize'].value / 2
        ax.set_xticks(np.linspace(0.5, cube_display.shape[1] - 0.5, num=11))
        ax.set_yticks(np.linspace(0.5, cube_display.shape[0] - 0.5, num=11))
        ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(tick_min, tick_max, 11)])
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(tick_min, tick_max, 11)])
        ax.set_xlabel("Parallel direction (arcsec)")
        ax.set_ylabel("Perpendicular direction (arcsec)")
        ax.set_xlim(0, cube_display.shape[1] - 1)
        ax.set_ylim(0, cube_display.shape[0] - 1)

        ax.add_artist(scalebar)
        ax.set_title(fr'Data cube slice at $\lambda$ = {slice_wavelength:.1f} $\AA$')
        ax.text(x=0.05, y=0.15, s=fr'${{\rm log}}M_{{\star}} = {logM}$',
                            fontsize=12, transform=ax.transAxes, color='blue')
        ax.text(x=0.05, y=0.1, s=fr'$z$ = {z}', fontsize=12,
                transform=ax.transAxes, color='blue')
        ax.text(x=0.05, y=0.05, s=f'ID:{self.config["subhaloID"]}', fontsize=12,
                transform=ax.transAxes, color='blue')

        cube_slice = cube_display

        arrow_length = min(cube_slice.shape) * 0.15  # 15% of image size
        arrow_width = max(1, int(cube_slice.shape[0] * 0.01))  # minimum width is 1
        perp_length = int(cube_slice.shape[0] * 0.3)  # 30% of image height
        para_length = int(cube_slice.shape[1] * 0.3)  # 30% of image width
        # Position arrows in upper left corner (in data coordinates)
        arrow_x_start = cube_slice.shape[1] * 0.05
        arrow_y_start = cube_slice.shape[0] * 0.9

        # Perpendicular arrow (vertical, pointing up)
        perp_arrow = patches.FancyArrowPatch(
            (arrow_x_start, arrow_y_start),
            (arrow_x_start, arrow_y_start - perp_length),
            arrowstyle='<->', 
            mutation_scale=int(cube_slice.shape[0] * 0.1),
            linewidth=arrow_width,
            color='m',
            linestyle='--',
            zorder=10
        )
        ax.add_patch(perp_arrow)

        # Parallel arrow (horizontal, pointing right)
        para_arrow = patches.FancyArrowPatch(
            (arrow_x_start, arrow_y_start),
            (arrow_x_start + para_length, arrow_y_start),
            arrowstyle='<->',
            mutation_scale=int(cube_slice.shape[1] * 0.1),
            linewidth=arrow_width,
            color='m',
            linestyle='--',
            zorder=10
        )
        ax.add_patch(para_arrow)

        ax.text(arrow_x_start - arrow_length * 0.2, arrow_y_start - perp_length * 0.5,
                'Perpendicular', fontsize=10, color='purple', 
                rotation=90, rotation_mode='anchor', ha='center', va='center')
        ax.text(arrow_x_start + para_length * 0.5, arrow_y_start + para_length * 0.1,
                'Parallel', fontsize=10, color='purple', 
                rotation=0, rotation_mode='anchor', ha='center', va='center')

        width = int(self.config['slitletSizeParallel'] / self.config['pixelScale'])
        height = int(self.config['slitletSizePerpendicular'] / self.config['pixelScale'])
        
        all_width = int(self.size_parallel / self.config['pixelScale'])
        all_height = int(self.size_perpendicular / self.config['pixelScale'])

        center_x = cube_slice.shape[1] // 2
        center_y = cube_slice.shape[0] // 2

        # Translation offsets
        offset_x = self.shift_parallel  # pixels to move right (negative for left)
        offset_y = self.shift_perpendicular  # pixels to move down (negative for up)

        # Apply translation to center
        translated_center_x = center_x + offset_x
        translated_center_y = center_y + offset_y

        # Number of rectangles above and below center

        n_upper = (self.config['nSlitlets'] - 1) // 2
        n_lower = (self.config['nSlitlets'] - 1) // 2

        if self.config['nSlitlets'] % 2 == 0:
                # 0, 1, 2, 3, 4, 5 (6 sitlets) -> center at 2, n_lower = 2, n_upper = 3
            n_upper += 1

        # Gap between rectangles (in pixels)
        gap = 0

        # Rotation angle in degrees
        rotation_angle = self.config['rotate'].to_value(u.deg)

        # Create all rectangles using translated center
        rectangles = []
        
        # observable area
        # obs_rect = patches.Rectangle(
        #     (translated_center_x - all_width / 2, translated_center_y - all_height / 2),
        #     all_width,
        #     all_height,
        #     edgecolor='green',
        #     facecolor='none',
        # )
        # rectangles.append(obs_rect)

        # Center rectangle
        center_rect = patches.Rectangle(
            (translated_center_x - width / 2, translated_center_y - height / 2),
            width, 
            height, 
            edgecolor='r',
            facecolor='none',
        )
        rectangles.append(center_rect)

        # Create upper rectangles
        for i in range(1, n_upper + 1):
            upper_rect = patches.Rectangle(
            (translated_center_x - width / 2, translated_center_y - height / 2 - i * (height + gap)),
            width,
            height,
            edgecolor='r',
            facecolor='none',
            )
            rectangles.append(upper_rect)

        # Create lower rectangles
        for i in range(1, n_lower + 1):
            lower_rect = patches.Rectangle(
            (translated_center_x - width / 2, translated_center_y + height / 2 + (i - 1) * (height + gap) + gap),
            width,
            height,
            edgecolor='r',
            facecolor='none',
            )
            rectangles.append(lower_rect)

        # Apply rotation transform around the translated center point
        # transform = Affine2D().rotate_deg_around(
        #     translated_center_x, translated_center_y, rotation_angle
        # ) + ax.transData

        # Add all rectangles with the rotation transform
        for rect in rectangles:
            # rect.set_transform(transform)
            ax.add_patch(rect)
                
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'MSA_slitlets.png'))
        plt.close()
        
        
    @staticmethod
    @numba.njit(cache=True, fastmath=True, parallel=True)
    def _overlap_counts_numba(counts_array: np.ndarray,
                              noise_counts_array: np.ndarray,
                              sed_counts_array: np.ndarray,
                              n_pixels_slitlet_parallel: int):
        
        """
        Combine pixels along the parallel axis to simulate slitlet overlap.
        
        Parameters
        ----------
        counts_array : np.ndarray
            The input counts array with shape (n_perpendicular, n_parallel, n_wave).
        noise_counts_array : np.ndarray
            The input noise array with the same shape as counts_array.
        sed_counts_array : np.ndarray
            The input SED counts array with the same shape as counts_array.
        n_pixels_slitlet_parallel : int
            Number of pixels comprising a slitlet in the parallel dispersion direction.
        
        Returns
        -------
        overlapped_counts_array : np.ndarray
            Counts array after combining pixels, shape (n_perpendicular, n_slitlets, n_wave).
        overlapped_noise_counts_array : np.ndarray
            Noise array after combining pixels, same shape as above.
        """
        
        # counts_array: 40, 18, num_wave
        num_pixels_on_detector = counts_array.shape[-1]
        size_on_detector = num_pixels_on_detector + n_pixels_slitlet_parallel - 1
        
        # overlapped_counts_array: 40, 9, num_wave
        overlapped_shape = (
            counts_array.shape[0],
            counts_array.shape[1] // n_pixels_slitlet_parallel,
            counts_array.shape[-1]
        )
        
        overlapped_counts_array = np.zeros((
            overlapped_shape
        ))
        
        overlapped_noise_counts_array = np.zeros((
            overlapped_shape
        ))
        
        overlapped_sed_counts_array = np.zeros((
            overlapped_shape
        ))
        
        offset = n_pixels_slitlet_parallel // 2
        
        for i in numba.prange(overlapped_shape[0]):
            for j in range(overlapped_shape[1]):
                
                line = np.zeros((size_on_detector))
                
                for k in range(n_pixels_slitlet_parallel):
                    line[k: k + num_pixels_on_detector] += \
                        counts_array[i, j * n_pixels_slitlet_parallel + k]
                        
                extracted_counts = line[offset:]
                overlapped_counts_array[i, j] = extracted_counts
                
                noise = np.zeros((size_on_detector))
                
                for k in range(n_pixels_slitlet_parallel):
                    noise[k: k + num_pixels_on_detector] += \
                        noise_counts_array[i, j * n_pixels_slitlet_parallel + k]**2
                        
                extracted_noise = np.sqrt(noise[offset:])
                overlapped_noise_counts_array[i, j] = extracted_noise
                        
                sed = np.zeros((size_on_detector))
                
                for k in range(n_pixels_slitlet_parallel):
                    sed[k: k + num_pixels_on_detector] += \
                        sed_counts_array[i, j * n_pixels_slitlet_parallel + k]
                        
                extracted_sed = sed[offset:]
                overlapped_sed_counts_array[i, j] = extracted_sed
                
        return overlapped_counts_array, overlapped_noise_counts_array, overlapped_sed_counts_array
    
    @staticmethod
    def _downsample_static(cube_slice: np.ndarray, oversample: int) -> np.ndarray:
        
        """
        Downsample an array by a factor of 'oversample' using skimage.transform.downscale_local_mean.

        Parameters
        ----------
        cube_slice : np.ndarray
            Input array to be downsampled. Can be 2D or 3D. If 3D, 
            downsampling is performed on the last two axes (spatial).
        oversample : int
            Downsampling factor. The array will be downsampled by this 
            factor along each spatial dimension.

        Returns
        -------
        np.ndarray
            Downsampled array.
        """
        
        if len(cube_slice.shape) == 2:
            down_factor = (oversample, oversample)
        elif len(cube_slice.shape) == 3:
            down_factor = (1, oversample, oversample)
            
        cube_slice = downscale_local_mean(cube_slice, down_factor)
        return cube_slice
    
    @staticmethod
    def _padding_static(cube_slice: np.ndarray, target_shape: tuple) -> np.ndarray:
        
        """
        Pad a numpy array to a target shape with zeros.

        Parameters
        ----------
        cube_slice : np.ndarray
            Array to pad. Can be 2D or 3D.
        target_shape : tuple
            Desired final shape (must match length of cube_slice.shape).

        Returns
        -------
        np.ndarray
            Padded array with shape equal to target_shape.
        """
        
        if cube_slice.ndim != len(target_shape):
            raise ValueError("size must have the same length as cube_slice.shape")
        
        
        padding_size = []
        for i in range(len(cube_slice.shape)):
            if cube_slice.shape[i] < target_shape[i]:
                padding_needed = target_shape[i] - cube_slice.shape[i]
                before_padding_size = padding_needed // 2
                after_padding_size = padding_needed - before_padding_size
            else:
                before_padding_size = 0
                after_padding_size = 0
            
            padding_size.append((before_padding_size, after_padding_size))
        
        padding_size = tuple(padding_size)
        padded_cube_slice = np.pad(
            cube_slice, padding_size, mode='constant', constant_values=0
        )
        return padded_cube_slice
    
    
    def _standardize(self, cube_slice: np.ndarray, target_size: tuple) -> np.ndarray:
        
        """
        Standardize the size of a given cube_slice array to the target_size.

        This method first pads the input array as needed to reach at least the target size, and then
        center-crops it to make sure the final output is exactly the target_size.

        Parameters
        ----------
        cube_slice : np.ndarray
            Input array to be standardized in size. Can be 2D or 3D.
        target_size : tuple
            Desired shape (height, width) or (depth, height, width).

        Returns
        -------
        np.ndarray
            Output array with shape equal to target_size, with original data centered.
        """
        
        padded_cube_slice = self._padding_static(cube_slice, target_size)
        cropped_cube_slice = self._center_crop(
            padded_cube_slice, target_size
        )

        return cropped_cube_slice
    
    @staticmethod
    def _center_crop(cube_slice: np.ndarray, target_shape: tuple) -> np.ndarray:
        
        """
        Center-crops the input array to the specified target shape.

        Parameters
        ----------
        cube_slice : np.ndarray
            The input array to crop. Should be 2D or 3D.
        target_shape : tuple
            The desired output shape, matching the number of dimensions of cube_slice.

        Returns
        -------
        np.ndarray
            The center-cropped array with the specified target_shape.
        """

        if cube_slice.ndim != len(target_shape):
            raise ValueError("target_shape must have the same length as arr.ndim")

        slices = []
        for dim, target in zip(cube_slice.shape, target_shape):
            if dim <= target:
                slices.append(slice(0, dim))
            else:
                start = (dim - target) // 2
                end = start + target
                slices.append(slice(start, end))

        return cube_slice[tuple(slices)]    
    
    @staticmethod
    @numba.njit(cache=True, fastmath=True)
    def _rotate_coordinates(coordinates: np.ndarray,
                            inclination: float, azimuth: float) -> np.ndarray:
        
        """
        Rotates input 3D coordinates by inclination and azimuth angles.

        Parameters
        ----------
        coordinates : np.ndarray
            Input array of shape (N, 3), where N is the number of points.
        inclination : float
            Inclination angle (degrees). Rotation around Y axis.
        azimuth : float
            Azimuth angle (degrees). Rotation around Z axis.

        Returns
        -------
        np.ndarray
            Rotated coordinates array (N, 3).
        """
        
        # 0. Convert degrees to radians
        inclination = np.deg2rad(inclination)
        azimuth = np.deg2rad(azimuth)
        
        # 1. We need to rotate by NEGATIVE azimuth to undo the twist around Z
        theta = -azimuth
        R_azi = np.array([
            [np.cos(theta), -np.sin(theta), 0.],
            [np.sin(theta), np.cos(theta), 0.],
            [0., 0., 1.]
        ], dtype=np.float32)
        
        # 2. We need to rotate by NEGATIVE inclination to undo the tilt around Y
        # Note: Depending on your definition, if 0 is face-on, use -inclination.
        # If 90 is face-on (equator), you might need -(inclination - pi/2).
        # Assuming standard astronomy where i=0 is face-on:
        alpha = -inclination
        R_inc = np.array([
            [np.cos(alpha), 0., np.sin(alpha)],
            [0., 1., 0.],
            [-np.sin(alpha), 0., np.cos(alpha)]
        ], dtype=np.float32)
        
        # The Order of Operations for De-projection:
        # First apply Azimuth undo (R_azi), THEN Inclination undo (R_inc).
        # Matrix math: R_total = R_inc @ R_azi
        # Vector math: v_new = R_inc @ (R_azi @ v_old)
        R = R_inc @ R_azi
        
        # Apply to coordinates
        # (N,3) @ (3,3).T is the correct way to apply matrix R to row vectors
        rotated_coords = coordinates @ R.T
        
        return rotated_coords
    
    
    @staticmethod
    @numba.njit(cache=True, fastmath=True)
    def _get_los_velocity(velocities: np.ndarray, 
                        inclination: float, azimuth: float) -> np.ndarray:
        
        """
        Compute the line-of-sight velocity for each particle given their velocity vectors
        and the observer's viewing angles.

        Parameters
        ----------
        velocities : np.ndarray
            Velocity array of shape (N, 3), where each row is (vx, vy, vz).
        inclination : float
            Inclination angle in degrees (rotation around Y axis).
        azimuth : float
            Azimuth angle in degrees (rotation around Z axis).

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing the line-of-sight velocity for each particle,
            projected according to the specified viewing angles.
        """

        
        inclination = np.deg2rad(inclination)
        azimuth = np.deg2rad(azimuth)
        
        # Construct the unit vector pointing towards the observer
        # Based on standard spherical coordinates:
        # x = sin(inc) * cos(azi)
        # y = sin(inc) * sin(azi)
        # z = cos(inc)
        
        los_vector = np.array([
            np.sin(inclination) * np.cos(azimuth),
            np.sin(inclination) * np.sin(azimuth),
            np.cos(inclination)
        ], dtype=np.float32)
        
        # Calculate dot product: v_los = v . n_los
        # velocities is (N, 3), los_vector is (3,)
        # The result is (N,)
        los_velocity = np.dot(velocities, los_vector)
        
        return los_velocity
    
    @staticmethod
    @numba.njit(cache=True, fastmath=True)
    def _get_rotational_velocity(coordinates: np.ndarray, 
                             velocities: np.ndarray, 
                             inclination: float, 
                             azimuth: float, 
                             bulk_velocity: np.ndarray) -> np.ndarray:

        """
        Compute the rotational (tangential) velocity for each particle in the galaxy, as observed
        from a given inclination and azimuthal angle.

        Parameters
        ----------
        coordinates : np.ndarray
            Array of particle coordinates, shape (N, 3).
        velocities : np.ndarray
            Array of particle velocities, shape (N, 3).
        inclination : float
            Inclination angle in degrees (rotation around Y axis).
        azimuth : float
            Azimuth angle in degrees (rotation around Z axis).

        Returns
        -------
        np.ndarray
            Array of shape (N,) containing the rotational (tangential) component of velocity
            for each particle, projected in the disk plane after deprojection to face-on.
        """

        # 0. Convert degrees to radians
        inclination = np.deg2rad(inclination)
        azimuth = np.deg2rad(azimuth)
        
        # 1. Define Rotation Matrices for De-projection (Inverse Rotation)
        # We use negative angles to rotate from "Observed" -> "Face-on"
        theta = -azimuth
        alpha = -inclination
        
        # Rotation around Z (Undo Azimuth)
        R_azi = np.array([
            [np.cos(theta), -np.sin(theta), 0.],
            [np.sin(theta),  np.cos(theta), 0.],
            [0., 0., 1.]
        ], dtype=np.float32)
        
        # Rotation around Y (Undo Inclination)
        R_inc = np.array([
            [np.cos(alpha), 0., np.sin(alpha)],
            [0., 1., 0.],
            [-np.sin(alpha), 0., np.cos(alpha)]
        ], dtype=np.float32)
        
        # Combined Rotation Matrix: Apply Azi first, then Inc
        R = R_inc @ R_azi
        
        # subtract bulk motion of the subhalo
        velocities -= bulk_velocity
        
        # 2. Rotate both Positions and Velocities
        # We transform the entire frame so the galaxy lies in the XY plane
        pos_rot = coordinates @ R.T
        vel_rot = velocities @ R.T
        
        # 3. Subtract center-of-mass velocity in the rotated frame
        # This ensures particles near the center have zero mean velocity
        # com_vx = np.mean(vel_rot[:, 0])
        # com_vy = np.mean(vel_rot[:, 1])
        # vel_rot[:, 0] -= com_vx
        # vel_rot[:, 1] -= com_vy
        
        # 4. Extract In-Plane Components
        # In the face-on frame, the LOS is the Z-axis.
        # The "perpendicular to LOS" components are just x and y.
        x = pos_rot[:, 0]
        y = pos_rot[:, 1]
        vx = vel_rot[:, 0]
        vy = vel_rot[:, 1]
        
        # 5. Calculate Planar Radius
        # Avoid division by zero for the exact center
        R_plane = np.sqrt(x**2 + y**2)
        
        # 6. Calculate Tangential/Rotational Velocity
        # Formula: Cross product magnitude in 2D / Radius
        # v_rot = (r x v) / |r|  -> (x*vy - y*vx) / R
        
        # with np.errstate(divide='ignore', invalid='ignore'):
        #     v_rot = (x * vy - y * vx) / R_plane
        
        v_rot = np.zeros_like(R_plane)
        for i in range(len(R_plane)):
            v_rot[i] = (x[i] * vy[i] - y[i] * vx[i]) / R_plane[i]
        
        # Handle the center point (where R_plane is 0)
        v_rot = np.nan_to_num(v_rot)
        
        return v_rot
    
    @staticmethod
    @numba.njit(cache=True, fastmath=True)
    def _transform(coords: np.ndarray, rotate: float, 
                   shiftPerpendicular: float, shiftParallel: float):
        
        """
        Apply a 2D affine transformation to 3D coordinates in the galaxy plane.
        
        This method rotates the input coordinates around the z-axis by the given angle `rotate`
        (in degrees), then shifts the coordinates by `shiftPerpendicular` along the y-axis
        and by `shiftParallel` along the x-axis, both in the rotated plane.
        
        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N, 3), the (x, y, z) coordinates to transform.
        rotate : float
            Rotation angle in degrees (counter-clockwise; negative value rotates clockwise).
        shiftPerpendicular : float
            Amount to shift along the y-axis (perpendicular direction) after rotation.
        shiftParallel : float
            Amount to shift along the x-axis (parallel direction) after rotation.
        
        Returns
        -------
        shifted_coords : np.ndarray
            Array of shape (N, 3), the transformed coordinates.
        """
        
        # rotate the plane instead of the slitlets
        rotate = np.radians(-rotate)
        
        c, s = np.cos(rotate), np.sin(rotate)
        rotation_matrix = np.array([
            [c, -s, 0.],
            [s, c, 0.],
            [0., 0., 1.]
        ], dtype=np.float32)
        rotated_coords = coords @ rotation_matrix.T
        
        # shift perpendicular and parallel are in y and x direction respectively
        # similarly, shift the plane instead of the slitlets
        shift_vector = np.array([-shiftPerpendicular, -shiftParallel, 0.], dtype=np.float32)
        shifted_coords = rotated_coords + shift_vector
        
        return shifted_coords
        
    
    @staticmethod
    def _calc_stats(coords: np.ndarray, values: np.ndarray,
                    bins_perpendicular: np.ndarray,
                    bins_parallel: np.ndarray,
                    statistic: str) -> np.ndarray:
        
        """
        Calculate binned statistics (e.g., mean, median, std) of `values` as a function
        of 2D projected coordinates provided by `coords` using bins defined by
        `bins_perpendicular` and `bins_parallel`.

        Parameters
        ----------
        coords : np.ndarray
            Array of shape (N, 2), columns are (perpendicular, parallel) coordinates for each particle.
        values : np.ndarray
            1D array with N elements, values to compute statistic of.
        bins_perpendicular : np.ndarray
            Bin edges for the perpendicular direction.
        bins_parallel : np.ndarray
            Bin edges for the parallel direction.
        statistic : str
            Statistic to compute ('mean', 'median', 'std', etc).

        Returns
        -------
        stats : np.ndarray
            2D array of statistic values for each (perpendicular, parallel) bin,
            shape (len(bins_perpendicular)-1, len(bins_parallel)-1).
        
        Notes
        -----
        Uses scipy.stats.binned_statistic_2d for the calculation.
        """
        
        stats, _, _, _ = binned_statistic_2d(
            coords[:, 1], coords[:, 0], values, 
            statistic=statistic, bins=[bins_perpendicular, bins_parallel]
        )
        return stats
    
    def input_particle_file(self, particle_file: str):
        
        """
        Set the filename for the input particle file. This method updates the instance so that
        future operations use the provided particle file instead of a default path. 
        
        Parameters
        ----------
        particle_file : str
            Path to the particle file to use as input for this object.
        """
        
        self.input_particle_file_called = True
        self.particle_file = particle_file
    
    def _save_properties(self, properties_array: dict,
                         properties_units: dict, save_path: str):
        
        """
        Save the provided properties arrays and their units to a FITS file.

        Parameters
        ----------
        properties_array : dict
            A dictionary where keys are property names and values are 2D numpy arrays
            containing the property values mapped to the grid (e.g., projected binned map).
        properties_units : dict
            A dictionary mapping property names to unit strings or astropy units.
        save_path : str
            The directory to which the FITS file will be saved. Filename fixed as 'properties.fits'.

        This method generates a multi-extension FITS file with one extension for each property.
        Each extension contains the data array for a property with the property name and unit
        set in the FITS header for metadata clarity. The primary HDU is left empty.
        """
        
        hdulist = fits.HDUList()
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        for prop, array in properties_array.items():
            hdu = fits.ImageHDU(array)
            # Add property name and unit to header
            hdu.header['PROP'] = prop
            unit = str(properties_units.get(prop, ''))
            hdu.header['UNIT'] = unit
            hdulist.append(hdu)
        
        filename = os.path.join(save_path, f'properties.fits')
        hdulist.writeto(filename, overwrite=True)
        
    def _display_properties(self, properties_array: dict,
                            properties_units: dict, save_path: str):
        
        """
        Display the provided 2D property maps as images and save the resulting figure.
        The function creates a multi-panel plot, one subplot per property, with proper
        axis scaling, units in titles, colorbars, and axis labels matching the projected
        spatial coordinates. The resulting figure is saved as 'properties.png' in save_path.

        Parameters
        ----------
        properties_array : dict
            Dictionary where keys are property names and values are 2D numpy arrays
            representing the value of that property projected onto the MSA grid.
        properties_units : dict
            Dictionary mapping property names to the unit for that property (str or astropy unit).
        save_path : str
            Directory where the output image will be saved.
        """
        
        fig, ax = plt.subplots(1, len(properties_array),
                               figsize=(5 * len(properties_array), 4))
        
        for i, prop in enumerate(properties_array.keys()):
            array = properties_array[prop]
            unit = properties_units[prop]
            
            y_coverage = self.config['ditherSize'].to_value(u.arcsec) * array.shape[0]
            x_coverage = self.config['slitletSizeParallel'].to_value(u.arcsec) * array.shape[1]
            
            x_tick_min = -x_coverage / 2
            x_tick_max = x_coverage / 2
            
            y_tick_min = -y_coverage / 2
            y_tick_max = y_coverage / 2
            
            if unit == '':
                unit = '1'
            elif unit == 'solMass':
                unit = 'Msun'
            ax[i].imshow(array, origin='lower', aspect='auto')
            cbar = plt.colorbar(ax[i].images[0], ax=ax[i])
            ax[i].set_title(f'{prop} ({unit})')
            
            ax[i].set_xticks(np.linspace(0, array.shape[1] - 1, num=11))
            ax[i].set_yticks(np.linspace(0, array.shape[0] - 1, num=11))
            ax[i].set_xticklabels([f"{v:.1f}" for v in np.linspace(x_tick_min, x_tick_max, 11)])
            ax[i].set_yticklabels([f"{v:.1f}" for v in np.linspace(y_tick_min, y_tick_max, 11)])
            ax[i].set_xlabel("Parallel direction (arcsec)")
            if i == 0:
                ax[i].set_ylabel("Perpendicular direction (arcsec)")
            ax[i].set_xlim(0, array.shape[1] - 1)
            ax[i].set_ylim(0, array.shape[0] - 1)
        
        plt.tight_layout()
        savefilename = os.path.join(save_path, f'True_properties.png')
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_properties(self, properties: list[str]=None,
                             functions: Union[None, list[callable]]=[None]):
        
        """
        Retrieve or compute physical properties projected onto the MSA grid.

        This function loads particle data from the associated file and projects specified physical
        properties (e.g., mass, star formation rate) onto the grid defined by the instrument's
        perpendicular and parallel axes, taking into account cosmological and observational parameters.

        Parameters
        ----------
        properties : list of str, optional
            A list of property names (such as ['mass', 'sfr', ...]) to extract, matching columns in the particle file.
            If None, defaults to extracting 'mass'.
        functions : list of callable or None, optional
            A list of functions to apply to each property (e.g. [np.log10, None]), or [None] for no modification.
            The length should match 'properties', or [None] will be applied to each property by default.

        Returns
        -------
        properties_array : dict
            Dictionary mapping property names to 2D numpy arrays (MSA-grid projections).
        properties_units : dict
            Dictionary mapping property names to their physical units (astropy.unit or str).

        Notes
        -----
        - Uses viewing angles, distances, offsets, and pixel grid setup from self.config and the current PostProcess object.
        - May apply provided functions to post-process resulting property arrays.
        """
        
        if hasattr(self, 'input_particle_file_called') \
            and self.input_particle_file_called \
                and os.path.exists(self.particle_file):
            particle_file = self.particle_file
            self.logger.info(f'Use input particle file: {particle_file}.')
        else:
            particle_file = os.path.join(
                self.dataCubeDir, f'Subhalo_{self.subhaloID}_particles.h5'
            )
            self.logger.info(f'Use particle file saved in preprocessing: {particle_file}.')

            if not os.path.exists(particle_file):
                self.logger.error(f'Particle file not found: {particle_file}.')
                raise FileNotFoundError(f'Particle file not found: {particle_file}.')
            
        inclination = np.float32(self.viewing_angle[0])
        azimuth = np.float32(self.viewing_angle[1])
        
        phyRes_perpendicular = self.config['lumiDis'] * self.config['ditherSize'].to_value(u.rad)
        phyRes_perpendicular = phyRes_perpendicular.to_value(u.kpc)
        
        phyRes_parallel = self.config['lumiDis'] * self.config['slitletSizeParallel'].to_value(u.rad)
        phyRes_parallel = phyRes_parallel.to_value(u.kpc)
        
        bins_perpendicular = np.linspace(-phyRes_perpendicular * (self.n_pixels_perpendicular // 2),
                                         phyRes_perpendicular * (self.n_pixels_perpendicular // 2), 
                                         self.n_pixels_perpendicular + 1)
        
        if self.n_pixels_output_parallel % 2 == 0:
            bins_parallel = np.linspace(-phyRes_parallel * (self.n_pixels_output_parallel // 2),
                                        phyRes_parallel * (self.n_pixels_output_parallel // 2), 
                                        self.n_pixels_output_parallel + 1)
        else:
            half = phyRes_parallel / 2
            bins_parallel = np.linspace(-phyRes_parallel * (self.n_pixels_output_parallel // 2) - half, 
                                        phyRes_parallel * (self.n_pixels_output_parallel // 2) + half,
                                        self.n_pixels_output_parallel + 1)
        
        # in kpc
        shift_perpendicular = self.config['lumiDis'] * self.config['offsetPerpendicular'].to_value(u.rad)
        shift_perpendicular = np.float32(shift_perpendicular.to_value(u.kpc))
        shift_parallel = self.config['lumiDis'] * self.config['offsetParallel'].to_value(u.rad)
        shift_parallel = np.float32(shift_parallel.to_value(u.kpc))
        
        rotate = np.float32(self.config['rotate'].to_value(u.deg))
        
        params = {
            'particle_file': particle_file,
            'inclination': inclination,
            'azimuth': azimuth,
            'bins_perpendicular': bins_perpendicular,
            'bins_parallel': bins_parallel,
            'transformation': {'rotate': rotate, 
                               'shiftParallel': shift_parallel,
                               'shiftPerpendicular': shift_perpendicular},
            'subhalo_info': self.subhalo_info,
            'configs': self.config,
        }
        
        properties_array = {}
        properties_units = {}
        for i, prop in enumerate(properties):
            try:
                func = functions[i]
                stats, unit = func(**params)
                properties_array[prop] = stats
                properties_units[prop] = unit
            except Exception as e:
                self.logger.error(f'Error calculating property {prop}: {e}')
                continue
        
        return properties_array, properties_units
    
    @staticmethod
    def _age_interp(cosmology: Cosmology) -> interp1d:
        z = np.linspace(0, 4, 1000)
        t = cosmology.age(z).to(u.Myr).value
        fage = interp1d(z, t, kind='cubic',
                        bounds_error=False, fill_value='extrapolate')
        return fage
    
    @staticmethod
    def get_metallicity_for_starforming_region(
        particle_file: str, inclination: float, azimuth: float, 
        bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
        transformation: dict, subhalo_info: dict, configs: dict):
        
        fage = PostProcess._age_interp(configs['cosmology'])
        
        particles = {}
        with h5py.File(particle_file, 'r') as file:
            
            particles['GFM_StellarFormationTime'] = file['PartType4']['GFM_StellarFormationTime'][:].astype(np.float32)
            snapshot_age = fage(configs['snapRedshift']) # in Myr
            particles['age'] = snapshot_age - fage(1/particles['GFM_StellarFormationTime'] - 1)
            
            # star formation region mask
            mask = np.where((particles['age'] < 10) & (particles['GFM_StellarFormationTime'] > 0))
            
            particles['Coordinates'] = file['PartType4']['Coordinates'][mask].astype(np.float32)
            unit = file['PartType4']['Coordinates'].attrs['unit']
            particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
            
            particles['GFM_Metallicity'] = file['PartType4']['GFM_Metallicity'][mask].astype(np.float32)
            unit_metallicity = "1"

        # rotate to observing direction
        coords = PostProcess._rotate_coordinates(
            particles['Coordinates'], inclination, azimuth)
        
        # apply transformation for coords in x-y plane if rotate and shifts in config is not 0
        coords = PostProcess._transform(coords, transformation['rotate'], 
                                        transformation['shiftPerpendicular'],
                                        transformation['shiftParallel'])
        
        metallicity = particles['GFM_Metallicity']
        stats = PostProcess._calc_stats(
            coords, metallicity, 
            bins_perpendicular, bins_parallel, statistic='mean')
        
        return stats, unit_metallicity
    
    @staticmethod
    def get_gas_metallicity(particle_file: str, inclination: float, azimuth: float, 
                            bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
                            transformation: dict, subhalo_info: dict, configs: dict):
        
        particles = {}
        with h5py.File(particle_file, 'r') as file:
            # gas in PartType0, make sure the gases are included in preprocessing
            particles['Coordinates'] = file['PartType0']['Coordinates'][:].astype(np.float32)
            unit = file['PartType0']['Coordinates'].attrs['unit']
            particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
            
            particles['GFM_Metallicity'] = file['PartType0']['GFM_Metallicity'][:].astype(np.float32)
            unit_metallicity = "1"
            
        coords = PostProcess._rotate_coordinates(
            particles['Coordinates'], inclination, azimuth)
        
        coords = PostProcess._transform(coords, transformation['rotate'], 
                                        transformation['shiftPerpendicular'],
                                        transformation['shiftParallel'])
        
        metallicity = particles['GFM_Metallicity']
        stats = PostProcess._calc_stats(
            coords, metallicity, 
            bins_perpendicular, bins_parallel, statistic='mean')
        
        return stats, unit_metallicity
    
    # @staticmethod
    # def get_rotational_velocity(
    #     particle_file: str, inclination: float, azimuth: float, 
    #     bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
    #     bulk_velocity: Union[None, np.ndarray] = None):
        
    #     particles = {}
    #     with h5py.File(particle_file, 'r') as file:
    #         particles['Coordinates'] = file['PartType4']['Coordinates'][:].astype(np.float32)
    #         unit = file['PartType4']['Coordinates'].attrs['unit']
    #         particles['Coordinates'] = particles['Coordinates'] * u.Unit(unit).to(u.kpc)
            
    #         particles['Velocities'] = file['PartType4']['Velocities'][:].astype(np.float32)
    #         unit_velocities = file['PartType4']['Velocities'].attrs['unit']
            
    #     coords = PostProcess._rotate_coordinates(
    #         particles['Coordinates'], inclination, azimuth)
    #     rotational_velocity = PostProcess._get_rotational_velocity(
    #         particles['Coordinates'], particles['Velocities'], inclination, azimuth, 
    #         bulk_velocity=bulk_velocity)
        
    #     stats = PostProcess._calc_stats(
    #         coords, rotational_velocity, 
    #         bins_perpendicular, bins_parallel, statistic='mean')
        
    #     return stats, unit_velocities
        
    @staticmethod
    def get_los_velocity(
        particle_file: str, inclination: float, azimuth: float, 
        bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
        transformation: dict, subhalo_info: dict, configs: dict):
        
        particles = {}
        with h5py.File(particle_file, 'r') as file:
            particles['Coordinates'] = file['PartType4']['Coordinates'][:].astype(np.float32)
            unit = file['PartType4']['Coordinates'].attrs['unit']
            particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
            
            particles['Velocities'] = file['PartType4']['Velocities'][:].astype(np.float32)
            unit_velocities = file['PartType4']['Velocities'].attrs['unit']
            
        coords = PostProcess._rotate_coordinates(
            particles['Coordinates'], inclination, azimuth)
        
        coords = PostProcess._transform(coords, transformation['rotate'], 
                                        transformation['shiftPerpendicular'],
                                        transformation['shiftParallel'])
        
        los_velocity = PostProcess._get_los_velocity(
            particles['Velocities'], inclination, azimuth)
            
        stats = PostProcess._calc_stats(
            coords, los_velocity, 
            bins_perpendicular, bins_parallel, statistic='mean')
            
        return stats, unit_velocities
    
    @staticmethod
    def get_velocity_dispersion(
        particle_file: str, inclination: float, azimuth: float, 
        bins_perpendicular: np.ndarray, bins_parallel: np.ndarray,
        transformation: dict, subhalo_info: dict, configs: dict):
        
        particles = {}
        with h5py.File(particle_file, 'r') as file:
            particles['Coordinates'] = file['PartType4']['Coordinates'][:].astype(np.float32)
            unit = file['PartType4']['Coordinates'].attrs['unit']
            particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
            
            particles['Velocities'] = file['PartType4']['Velocities'][:].astype(np.float32)
            unit_velocities = file['PartType4']['Velocities'].attrs['unit']
            
        coords = PostProcess._rotate_coordinates(
            particles['Coordinates'], inclination, azimuth)
        
        coords = PostProcess._transform(coords, transformation['rotate'], 
                                        transformation['shiftPerpendicular'],
                                        transformation['shiftParallel'])
        
        los_velocity = PostProcess._get_los_velocity(
            particles['Velocities'], inclination, azimuth)
            
        stats = PostProcess._calc_stats(
            coords, los_velocity, 
            bins_perpendicular, bins_parallel, statistic='std')
            
        return stats, unit_velocities
    
    @staticmethod
    def get_mass(
        particle_file: str, inclination: float, azimuth: float, 
        bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
        transformation: dict, subhalo_info: dict, configs: dict):
        
        particles = {}
        with h5py.File(particle_file, 'r') as file:
            particles['Coordinates'] = file['PartType4']['Coordinates'][:].astype(np.float32)
            unit = file['PartType4']['Coordinates'].attrs['unit']
            particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
            
            particles['Masses'] = file['PartType4']['Masses'][:].astype(np.float32)
            unit_masses = file['PartType4']['Masses'].attrs['unit']
            
        coords = PostProcess._rotate_coordinates(
            particles['Coordinates'], inclination, azimuth)
        
        coords = PostProcess._transform(coords, transformation['rotate'], 
                                        transformation['shiftPerpendicular'],
                                        transformation['shiftParallel'])
        
        masses = particles['Masses']
        
        stats = PostProcess._calc_stats(
            coords, masses, 
            bins_perpendicular, bins_parallel, statistic='sum')
        
        return stats, unit_masses
        
    def create_MSA_dataTensor(self):
        
        """
        Create the data tensor for the MSA simulation, including signal and noise arrays.

        This method orchestrates the full creation process for the MSA (Multi-Shutter Array) data tensor.
        It performs the following major steps:
            - Define the wavelength sampling based on instrument resolution
            - Retrieve or generate the relevant PSF cube
            - Simulate detector sampling over the spatial and spectral grids
            - Compute signal and noise arrays for each slitlet
            - Assemble all outputs into a single, well-formatted data tensor

        Returns
        -------
        dataTensor : np.ndarray
            The full (n_slitlet_y, n_slitlet_x, 3, n_bins) array, ready for export or further use.
              - data[...,0,:] = wavelengths (middle points)
              - data[...,1,:] = signal (flux or electron counts)
              - data[...,2,:] = noise (flux or electron counts)
        """

        
        waves_by_resolution = self._get_waves_by_resolution()
        waves_at_pixel = self._get_waves_at_pixel()
        
        self.logger.info('Getting PSF cubes...')
        psf_cube = self._get_PSF_cube()
        
        self.logger.info('Creating MSA array...')
        
        start_time = time.time()
        
        shm_cube = shared_memory.SharedMemory(create=True, size=self.dataCube.nbytes)
        shm_cube_array = np.ndarray(self.dataCube.shape, dtype=self.dataCube.dtype, buffer=shm_cube.buf)
        shm_cube_array[:] = self.dataCube[:]
        
        shm_psf = shared_memory.SharedMemory(create=True, size=psf_cube.nbytes)
        shm_psf_array = np.ndarray(psf_cube.shape, dtype=psf_cube.dtype, buffer=shm_psf.buf)
        shm_psf_array[:] = psf_cube[:]
        
        info_dict = {
            'n_pixels_perpendicular': self.n_pixels_perpendicular,
            'n_pixels_parallel': self.n_pixels_parallel,
            'rotation_angle': self.config['rotate'].to_value(u.deg),
            'shift_perpendicular': self.shift_perpendicular,
            'shift_parallel': self.shift_parallel,
            'rescale_ratio_perpendicular': self.rescale_ratio_perpendicular,
            'rescale_ratio_parallel': self.rescale_ratio_parallel,
            'oversample': self.config['oversample'],
            'num_threads': self.config['numThreads']
        }
        
        args_list = [
            (i,
             shm_cube.name, shm_cube_array.shape, shm_cube_array.dtype,
             shm_psf.name, shm_psf_array.shape, shm_psf_array.dtype, 
             info_dict
             )
            for i in range(len(self.wavelengths))
        ]
        
        # with mp.Pool(processes=self.config['nJobs']) as pool:
        #     results = pool.map(self._process_single_spatial_slice_static, args_list)
            
        results = []
        for args in args_list:
            result = self._process_single_spatial_slice_static(args)
            results.append(result)
        
        shm_cube.close()
        shm_cube.unlink()
        shm_psf.close()
        shm_psf.unlink()
        
        msa_array = np.array(results)
            
        end_time = time.time()
        self.logger.info(f'Time taken: {end_time - start_time:.2f} seconds')
        
        # stach slice, (n_wavelengths, n_pixels_perpendicular, n_pixels_parallel)
        msa_array = np.array(msa_array)
        
        self.logger.info('Calculating background count rates...')
        bkg_count_rates = self._calc_bkg(waves_at_pixel)
        
        self.logger.info('Processing MSA spectra...')
        start_time = time.time()
        
        shm_msa = shared_memory.SharedMemory(create=True, size=msa_array.nbytes)
        shm_msa_array = np.ndarray(msa_array.shape, dtype=msa_array.dtype, buffer=shm_msa.buf)
        shm_msa_array[:] = msa_array[:]
        
        shm_exp = shared_memory.SharedMemory(create=True, size=self.n_exposure_array.nbytes)
        shm_exp_array = np.ndarray(self.n_exposure_array.shape, dtype=self.n_exposure_array.dtype, buffer=shm_exp.buf)
        shm_exp_array[:] = self.n_exposure_array[:]
        
        # Prepare config dict (convert astropy Quantity to dict with value and unit)
        config_dict = {}
        for key, value in self.config.items():
            if isinstance(value, u.Quantity):
                config_dict[key] = {
                    'value': value.value,
                    'unit': str(value.unit)
                }
            else:
                config_dict[key] = value
        
        # Prepare arguments for workers (only small data needs to be pickled)
        args_list = [
            (i, j,
             shm_msa.name, msa_array.shape, msa_array.dtype,
             shm_exp.name, self.n_exposure_array.shape, self.n_exposure_array.dtype,
             bkg_count_rates, self.wavelengths, self.interp_throughput,
             waves_at_pixel, config_dict)
            for i, j in product(range(self.n_pixels_perpendicular),
                                range(self.n_pixels_parallel))
        ]
        
        with mp.Pool(processes=self.config['numThreads']) as pool:
            results = pool.map(self._process_single_spectrum_worker, args_list)
            
        # results = []
        # for args in args_list:
        #     result = self._process_single_spectrum_worker(args)
        #     results.append(result)
        
        # Clean up shared memory
        shm_msa.close()
        shm_msa.unlink()
        shm_exp.close()
        shm_exp.unlink()
        
        end_time = time.time()
        self.logger.info(f'Time taken: {end_time - start_time:.2f} seconds')
        
        i, j, counts_val, noise_counts_val, sed_counts_val = results[0]
        
        # 40, 18, num_wave
        counts_array = np.zeros((
            self.n_pixels_perpendicular, 
            self.n_pixels_parallel,
            counts_val.shape[0]
        ))
        
        # 40, 18, num_wave
        noise_counts_array = np.zeros((
            self.n_pixels_perpendicular, 
            self.n_pixels_parallel,
            noise_counts_val.shape[0]
        ))
        
        sed_counts_array = np.zeros((
            self.n_pixels_perpendicular, 
            self.n_pixels_parallel,
            sed_counts_val.shape[0]
        ))
        
        for result in results:
            i, j, counts_val, noise_counts_val, sed_counts_val = result
            counts_array[i, j] = counts_val
            noise_counts_array[i, j] = noise_counts_val
            sed_counts_array[i, j] = sed_counts_val

        # input: 40, 18, num_wave, output: 40, 9, num_wave
        counts_array, noise_counts_array, sed_counts_array = self._overlap_counts_numba(
            counts_array, noise_counts_array, sed_counts_array, self.n_pixels_slitlet_parallel
        )

        if self.config['unit'] != 'electron':
            
            signal_array, noise_array, sed_array = self._convert_to_flux_array(
                waves_at_pixel, counts_array, noise_counts_array, sed_counts_array
            )
        else:
            signal_array = counts_array
            noise_array = noise_counts_array
            sed_array = sed_counts_array
        
        # 40, 9, num_wave
        signal_array, noise_array, sed_array = self._rebin_fluxes_and_noise_array(
            waves_at_pixel, signal_array, noise_array, sed_array, waves_by_resolution
        )
        
        dataTensor = self._assemble_data(
            waves_by_resolution, signal_array, noise_array, sed_array
        )
        
        self.logger.info('Saving data tensor...')
        savefilename = os.path.join(self.save_path, f'MSA_dataTensor.fits')
        self._save_dataTensor(dataTensor, savefilename)
        
    def illustrate_MSA(self):
        
        """
        Illustrate the current state of the MSA simulation. This method reads the saved data tensor FITS,
        pulls the appropriate wavelength slice for a display emission line, and displays multiple diagnostic
        visualizations such as the exposure map, spectra at specific pixels, and the full observed MSA stamp.
        It also retrieves and saves particle property maps and generates property visualization panels.
        """
        
        self.logger.info('Illustrating MSA...')
        
        dataTensor_filename = os.path.join(self.save_path, f'MSA_dataTensor.fits')
        if not os.path.exists(dataTensor_filename):
            self.logger.error(f'Data Tensor file {dataTensor_filename} does not exist.')
            return
        
        with fits.open(dataTensor_filename, memmap=False) as f:
            dataTensor = f[1].data
        
        emission_line, wave = get_wave_for_emission_line(self.config['displayAround'])
        target_wave = wave * (1 + self.config['viewRedshift'])
        
        if target_wave < self.begin_wave or target_wave > self.end_wave:
            self.logger.warning(f'{emission_line}{wave:.2f} is out of range for begin_wave {self.begin_wave:.2f} and end_wave {self.end_wave:.2f}.')
            self.logger.info(f'Falling back to Ha6563')
            
            emission_line = 'Ha'
            wave = 6563
            target_wave = wave * (1 + self.config['viewRedshift'])
        
        idx = (np.abs(self.wavelengths - target_wave)).argmin()
        slice_wavelength = self.wavelengths[idx]
        
        extend = 50
        # for situation when the line is at edges
        if idx - 50 < 0 or idx + 50 > len(self.wavelengths):
            extend = min(idx, len(self.wavelengths) - idx)
        
        # increase the signal by summing over the surrounding wavelengths
        cube_display = np.sum(self.dataCube[idx - extend: idx + extend], axis=0)
        cube_display = self._downsample_static(cube_display, self.config['oversample'])
        cube_display = self._standardize(
            cube_display, (self.boxlength_in_pixel, self.boxlength_in_pixel)
        )
        displaySpectrumAtPixel = self.config['displaySpectrumAtPixel']
        self._display_exposures(displaySpectrumAtPixel, self.save_path)
        displayEmissionLines = self.config['displayEmissionLines']
        self._display_spectra(dataTensor, displaySpectrumAtPixel, 
                              displayEmissionLines, self.save_path)
        self._display_MSA_obs(
            dataTensor, cube_display, slice_wavelength, 
            displaySpectrumAtPixel, self.save_path)
    
    def get_truth_properties(self, keep_defaults: bool=True, properties: list[str]=None, 
                             functions: Union[None, list[callable]]=[None]):
        
        """
        Retrieve and calculate ground-truth properties for the dataset.

        This method computes and saves various "truth" properties—such as metallicity, velocities,
        or mass maps—using supplied property-function pairs (or the defaults). It can be extended to
        include custom properties and functions by the user.

        Parameters
        ----------
        keep_defaults : bool, optional
            If True, include the default set of properties and functions in addition to any provided ones.
            If False, only use the user-supplied properties/functions. Default is True.
        properties : list of str, optional
            A list of property names to compute (each must have an associated callable).
        functions : list of callables or None, optional
            A list of functions to compute the corresponding properties. Each function should accept the required
            arguments to compute the property for the dataset. Defaults to [None], falling back to defaults.

        Returns
        -------
        None
            The calculated properties are saved by side effect, as FITS maps and visualization panels.
        """
        
        self.logger.info('Getting truth properties...')
        
        # default_properties = ["Metallicity", "RotationalVelocity",
        #                     "LOSVelocity", "Mass", "VelDisp"]
        # default_functions = [PostProcess.get_metallicity, PostProcess.get_rotational_velocity, 
        #              PostProcess.get_los_velocity, PostProcess.get_velocity_dispersion, 
        #              PostProcess.get_mass]
        
        if keep_defaults:
            
            self.logger.info('Calculating default properties...')
            
            default_properties = ['MetallicityStarFormingRegion', 'MetallicityGas', 'LOSVelocity', 
                                  'Mass', 'VelDisp']
            default_functions = [PostProcess.get_metallicity_for_starforming_region, 
                                PostProcess.get_gas_metallicity, PostProcess.get_los_velocity, 
                                PostProcess.get_mass, PostProcess.get_velocity_dispersion]
            
            prop_to_func = {prop: func for prop, func in 
                            zip(default_properties, default_functions)}
            
            if properties is not None and functions is not None:
                assert len(properties) == len(functions), \
                    "Number of properties and functions must be the same."
                
                prop_to_func.update({prop: func for prop, func in 
                                    zip(properties, functions)})
            
        else:
            
            prop_to_func = {}
            
            if properties is not None and functions is not None:
                assert len(properties) == len(functions), \
                    "Number of properties and functions must be the same."
                prop_to_func = {prop: func for prop, func in 
                                zip(properties, functions)}
            else:
                pass

        if len(prop_to_func) == 0:
            self.logger.warning('No properties and functions provided, skip true property calculation.')

        else:
            self.logger.info(f'Record properties: {list(prop_to_func.keys())}')
            
            properties = list(prop_to_func.keys())
            functions = [prop_to_func[prop] for prop in properties]
            
            properties_array, properties_units = self._get_properties(
                properties, functions)
            
            self._save_properties(properties_array, properties_units, self.save_path)
            self._display_properties(properties_array, properties_units, self.save_path)