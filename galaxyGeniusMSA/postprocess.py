import numpy as np
from astropy.io import fits
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import Cosmology
from scipy.interpolate import interp1d
import os
from skimage.transform import rescale, rotate, downscale_local_mean
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_scalebar.dimension import _Dimension
import numba
from numba.typed import Dict
from numba import types
from itertools import product
import time
from typing import Union
import rocket_fft

from .utils import read_config, galaxygenius_data_dir, setup_logging
from .utils import get_wave_for_emission_line, read_json
from .properties import *

class ParsecDimension(_Dimension):
    def __init__(self):
        super().__init__('pc')
        self.add_units('kpc', 1000)
    
class AngleDimension(_Dimension):
    def __init__(self):
        super().__init__(r'${^{\prime\prime}}$')
        self.add_units(r'${^{\prime}}$', 60)


@numba.njit(cache=True, fastmath=True)
def _trapezoid_numba(y: np.ndarray, x: np.ndarray) -> np.float32:
    
    result = np.float32(0.0)
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        result += (y[i - 1] + y[i]) * dx / np.float32(2.0)
    return result

@numba.njit(cache=True, fastmath=True)
def _integrate_counts_numba(values: np.ndarray, wave_bins: np.ndarray, 
                                aperture: np.float32, pixel_scale: np.float32,
                                throughput_dict: Dict, count_constants: np.float32) -> np.float32:
    
    wavelength = throughput_dict['wavelengths']
    throughput = throughput_dict['throughput']
    
    values = values / wave_bins**2
    
    thr = np.interp(wave_bins, wavelength, throughput)

    const_factor = np.float32(np.pi) * (aperture / np.float32(2.0))**2 * pixel_scale**2
    
    count_rate = _trapezoid_numba(wave_bins * values * thr, wave_bins)
    count_rate = count_rate * count_constants * const_factor
    
    return count_rate


@numba.njit(cache=True, fastmath=True, parallel=True)
def _count_rate_numba(wavelength: np.ndarray, spectrum: np.ndarray, waves_at_pixel: np.ndarray,
                        throughput_dict: Dict, count_constants: np.float32, numSpecWaveBins: np.int32, 
                        aperture: np.float32, pixel_scale: np.float32) -> np.ndarray:
    
    count_rates = np.zeros(len(waves_at_pixel) - 1, dtype=np.float32)
    
    for i in numba.prange(len(waves_at_pixel) - 1):
        wave_begin = waves_at_pixel[i]
        wave_end = waves_at_pixel[i + 1]
    
        wave_bins = np.linspace(wave_begin, wave_end, numSpecWaveBins).astype(np.float32)
        
        flux = np.interp(wave_bins, wavelength, spectrum)
        count_rate = _integrate_counts_numba(
            flux, wave_bins, aperture, pixel_scale, throughput_dict, count_constants
        )
        count_rates[i] = count_rate
    
    return count_rates

@staticmethod
@numba.njit(cache=True, fastmath=True, parallel=True)
def _numba_convolve_fft_static(cube: np.ndarray, psf: np.ndarray) -> np.ndarray:
    n_wave, h_img, w_img = cube.shape
    _, h_ker, w_ker = psf.shape
    
    pad_h = h_img + h_ker - 1
    pad_w = w_img + w_ker - 1
    
    start_y = (h_ker - 1) // 2
    start_x = (w_ker - 1) // 2
    
    # Pre-allocate output to avoid memory fragmentation
    output = np.empty((n_wave, h_img, w_img), dtype=cube.dtype)
    
    for i in numba.prange(n_wave):
        # Perform 2D convolution for the single slice
        img_slice = cube[i]
        ker_slice = psf[i]
        
        sl_freq = np.fft.rfft2(img_slice, s=(pad_h, pad_w))
        kf_freq = np.fft.rfft2(ker_slice, s=(pad_h, pad_w))
        
        res_full = np.fft.irfft2(sl_freq * kf_freq, s=(pad_h, pad_w))
        
        output[i] = res_full[start_y : start_y + h_img, start_x : start_x + w_img]
        
    return output

@staticmethod
@numba.njit(
"float32[:, :](float32[:, :, :], float32[:], float32[:])", 
parallel=True, cache=True, fastmath=True)
def integrate_bandpass(img: np.ndarray, tran: np.ndarray, wave: np.ndarray) -> np.ndarray:
    # not-used
    n = len(wave)
    h, w = img.shape[1], img.shape[2]
    out = np.zeros((h, w), dtype=np.float32)
    for i in numba.prange(h):
        for j in range(w):
            integral = np.float32(0.0)
            for k in range(1, n):
                y1 = img[k-1, i, j] * tran[k-1] * wave[k-1]
                y2 = img[k, i, j] * tran[k] * wave[k]
                dx = wave[k] - wave[k-1]
                integral += (y1 + y2) / np.float32(2.0) * dx
            out[i, j] = integral
    return out

@staticmethod
@numba.njit(cache=True, fastmath=True, parallel=True)
def _add_noise_numba(
    counts: np.ndarray,
    bkg_counts: np.ndarray,
    dark_counts: np.ndarray,
    readout: np.ndarray,
    n_exposure: np.int32
) -> tuple[np.ndarray, np.ndarray]:
    
    ideal_counts = counts.copy()
    
    mean_noise_counts = np.zeros_like(counts)
    for i in numba.prange(len(counts)):
        mean_noise_counts[i] = bkg_counts[i] + dark_counts[i]
    
    counts = counts + mean_noise_counts
    
    # Poisson noise
    for i in numba.prange(len(counts)):
        counts[i] = np.random.poisson(counts[i])
    
    # Read noise for each exposure
    for exp in range(n_exposure):
        for i in numba.prange(len(counts)):
            read_noise = np.random.normal(np.float32(0.0), readout[i])
            read_noise = np.round(read_noise)
            counts[i] += read_noise
    
    counts = counts - mean_noise_counts
    
    # Calculate noise counts
    noise_counts = np.zeros_like(counts)
    for i in numba.prange(len(counts)):
        noise_counts[i] = np.sqrt(
            ideal_counts[i] + bkg_counts[i] + dark_counts[i] + n_exposure * readout[i]**2
        )
    
    return counts, noise_counts

@staticmethod
@numba.njit(cache=True, fastmath=True, parallel=True)
def _rebin_fluxes_and_noise_numba(
    wavelengths_in: np.ndarray,
    fluxes_in: np.ndarray,
    noise_in_squared: np.ndarray,
    sed_in: np.ndarray,
    wavelengths_out: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    # wavelengths_out is wave edges
    fluxes_out = np.zeros(len(wavelengths_out) - 1, dtype=fluxes_in.dtype)
    noise_out = np.zeros(len(wavelengths_out) - 1, dtype=noise_in_squared.dtype)
    sed_out = np.zeros(len(wavelengths_out) - 1, dtype=sed_in.dtype)
    
    for i in numba.prange(len(wavelengths_out) - 1):
        w_start = wavelengths_out[i]
        w_end = wavelengths_out[i + 1]

        idx_start = np.searchsorted(wavelengths_in, w_start, side='right') - 1
        idx_end = np.searchsorted(wavelengths_in, w_end, side='left')
        
        idx_start = max(np.int32(0), idx_start)
        idx_end = min(len(fluxes_in), idx_end)
        
        widths_in = w_end - w_start
        
        for j in range(idx_start, idx_end):
            
            p_start = wavelengths_in[j]
            p_end = wavelengths_in[j + 1]
            
            overlap_start = max(w_start, p_start)
            overlap_end = min(w_end, p_end)
            overlap_width = overlap_end - overlap_start
            
            if overlap_width > np.float32(0.0):
                frac = overlap_width / widths_in
                fluxes_out[i] += fluxes_in[j] * frac
                noise_out[i] += noise_in_squared[j] * frac
                sed_out[i] += sed_in[j] * frac
                
    for i in numba.prange(len(noise_out)):
        noise_out[i] = np.sqrt(noise_out[i])
    
    return fluxes_out, noise_out, sed_out

@staticmethod
@numba.njit(cache=True, fastmath=True, parallel=True)
def _overlap_counts_numba(counts_array: np.ndarray,
                            noise_counts_array: np.ndarray,
                            sed_counts_array: np.ndarray,
                            n_pixels_slitlet_parallel: np.int32) \
                                -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    
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


class PostProcess:
    
    def __init__(self, subhaloID: int):
        
        """
        PostProcess class for handling post-processing of JWST MSA-3D simulation.
        It loads the data cube and other relevant data, applying astrophysical and instrumental transformations,
        and producing synthetic observables compatible with JWST/NIRSpec MSA observations. 
        Finally, it saves the resulting MSA data tensor and other relevant files to the result directory.

        Parameters
        ----------
        subhaloID : int
            The identifier of the target subhalo to be post-processed.
        """
        
        self.subhaloID = subhaloID
        self.dataDir = galaxygenius_data_dir()
        self.__init_count_constants()
        
    def __init_count_constants(self):
        
        # dimension conversion without real values
        
        f_lam_unit = u.MJy / u.sr * const.c / u.angstrom**2
        f_lam_unit = f_lam_unit.to(u.erg / u.s / u.cm**2 / u.angstrom / u.sr)
        
        # const_factor = np.pi * (u.m / 2)**2 * u.arcsec**2 / (const.h * const.c)
        const_factor = u.m**2 * u.arcsec**2 / (const.h * const.c)
        
        count_constants = f_lam_unit * const_factor * u.angstrom**2
        
        count_constants = count_constants.to_value(u.s**-1)
        
        self.count_constants = count_constants
    
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
        
        # 40, 9
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
            
            wavelengths = wavelengths.astype(np.float32)
            dlds = dlds.astype(np.float32)
            resolutions = resolutions.astype(np.float32)
            
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
            
            wavelengths = wavelengths.astype(np.float32)
            throughputs = throughputs.astype(np.float32)
        
        self.throughput_dict = Dict.empty(
            key_type=types.unicode_type,
            value_type=types.float32[:]
        )
        
        # numba compatible dict
        self.throughput_dict['wavelengths'] = wavelengths
        self.throughput_dict['throughput'] = throughputs
        
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
            If None, the angles will be extracted from the dataCube_path.
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
            self.wavelengths = f[1].data['grid_points'] * 10**4 # in angstrom
        
        # slightly extend to avoid error on interp
        begin_wave = self.begin_wave - 20
        end_wave = self.end_wave + 20
        
        idx = np.where((self.wavelengths > begin_wave) & (self.wavelengths < end_wave))[0]
        
        self.wavelengths = self.wavelengths[idx]
        
        self.dataCube = self.dataCube.astype(np.float32)
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
        psf_array = psf_array.astype(np.float32)
        
        return psf_array

    def __psf_from_stpsf(self) -> np.ndarray:
        
        nrs = self.__init_stpsf()
        
        waves = (self.wavelengths * u.angstrom).to_value(u.m)
        psf_cube = nrs.calc_datacube_fast(waves, oversample=self.config['oversample'])
        psf_cube = psf_cube[0].data
        
        psf_cube = psf_cube.astype(np.float32)
        
        return psf_cube
    
    def _check_size(self):
        
        assert len(self.input_psf_cube) == len(self.wavelengths), \
            f'PSF cube and wavelengths size do not match, {len(self.input_psf_cube)} vs. {len(self.wavelengths)}'
            
    def _check_existing_psf(self, wavelengths: np.ndarray, oversample: int):
        
        cond_1 = len(wavelengths) == len(self.wavelengths)
        cond_2 = np.allclose(wavelengths, self.wavelengths)
        cond_3 = oversample == self.config['oversample']
        
        if cond_1 and cond_2 and cond_3:
            return True
        else:
            return False    
    
    def _get_PSF_cube(self) -> np.ndarray:
        
        if hasattr(self, 'input_psf_called') and self.input_psf_called:
            self._check_size()
            self.logger.info('Use input PSF cube.')
            self.psf_cube = self.input_psf_cube.astype(np.float32)
            return
        
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
        
        psf_cube = psf_cube.astype(np.float32)
        self.psf_cube = psf_cube
    
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
            
        self.psf_bandpass = psf_bandpass
    
    
    def _shift_cube(self, cube: np.ndarray, dy: int, dx: int) -> np.ndarray:
        
        h, w = cube.shape[1:]
        result = np.zeros_like(cube) # 3d array
        
        y1 = max(0, dy)
        y2 = h + min(0, dy)
        x1 = max(0, dx)
        x2 = w + min(0, dx)
        
        result[:, y1:y2, x1:x2] = cube[:, y1 - dy:y2 - dy, x1 - dx:x2 - dx]
        return result

    def _rotate_cube(self, cube: np.ndarray, angle: float) -> np.ndarray:
        
        # We rotate each slice (cube[z, :, :]) in xy spatial plane by the given angle
        rotated_cube = np.empty_like(cube)
        for i in range(cube.shape[0]):
            rotated_cube[i] = rotate(
                cube[i], 
                angle, 
                resize=False, 
                mode='constant', 
                cval=np.nan, 
                preserve_range=True
            )
        return rotated_cube
        
    def _transform_cube(self, cube: np.ndarray) -> np.ndarray:
        
        # match the shape of the cube to the number of pixels
        if self.n_pixels_perpendicular % 2 != 0:
            if cube.shape[1] % 2 == 0:
                cube = cube[:, :-1, :] # to odd
        else:
            if cube.shape[1] % 2 != 0:
                cube = cube[:, :-1, :] # to even
            
        if self.n_pixels_parallel % 2 != 0:
            if cube.shape[2] % 2 == 0:
                cube = cube[:, :, :-1] # to odd
        else:    
            if cube.shape[2] % 2 != 0:
                cube = cube[:, :, :-1] # to even
        
        # rotate
        angle = self.config['rotate'].to_value(u.deg)
        if angle != 0:
            cube = cube.astype(cube.dtype.newbyteorder('='))
            cube = self._rotate_cube(cube, angle)
            
        # shift
        if self.shift_perpendicular != 0 or self.shift_parallel != 0:
            cube = self._shift_cube(cube, self.shift_perpendicular, self.shift_parallel)
        
        # rescale
        cube = rescale(
            cube, 
            (1, self.rescale_ratio_perpendicular, self.rescale_ratio_parallel),
            anti_aliasing=True,
        )
            
        return cube
    
    def _process_spatial_dimension(self) -> np.ndarray:
        
        cube = _numba_convolve_fft_static(
            self.dataCube, self.psf_cube
        )
        cube = self._downsample_static(cube, self.config['oversample'])
        
        cube = self._padding_cube(cube, (None, 100, 100))
        cube = self._transform_cube(cube)
        
        slice_center_idx_perpendicular = cube.shape[1] // 2
        slice_center_idx_parallel = cube.shape[2] // 2
        
        slice_perpendicular = slice(
            slice_center_idx_perpendicular - self.n_pixels_perpendicular // 2,
            slice_center_idx_perpendicular + self.n_pixels_perpendicular // 2
        )
        
        slice_parallel = slice(
            slice_center_idx_parallel - self.n_pixels_parallel // 2,
            slice_center_idx_parallel + self.n_pixels_parallel // 2
        )
        
        return cube[:, slice_perpendicular, slice_parallel]

    def _get_waves_by_resolution(self):
        
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
        
        waves_corresponding_to_resolution = np.array(waves_corresponding_to_resolution, 
                                                     dtype=np.float32)
        
        self.waves_by_resolution = waves_corresponding_to_resolution
        self.logger.info(f'Number of wavelengths by resolution: {len(self.waves_by_resolution) - 1}')
        
    def _get_waves_at_pixel(self):
        
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

        waves_at_pixel = np.array(waves_at_pixel, dtype=np.float32)
        
        self.waves_at_pixel = waves_at_pixel
        self.logger.info(f'Number of wavelengths by pixel: {len(self.waves_at_pixel) - 1}')
    
    def _get_jwst_background(self, background_path: str):

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
            
            wavelength = wavelength.astype(np.float32)
            flux = flux.astype(np.float32)
            
            self.interp_bkg = interp1d(
                wavelength, flux, kind='linear'
            )
        except Exception as e:
            raise ValueError(f'Error getting background emission file from JWST Background Tool (JBT): {e}')
    
    def _get_bkg(self):
        
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
                
                wavelength = wavelength.astype(np.float32)
                flux = flux.astype(np.float32)
                
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
    
    @staticmethod
    def _add_noise_static(spectrum_count_rates: np.ndarray, bkg_count_rates: np.ndarray,
                          n_exposure: np.int32, exposure_time: np.float32,
                          dark_current: np.float32, read_out: np.float32) -> tuple[np.ndarray, np.ndarray]:
        
        counts = spectrum_count_rates * n_exposure * exposure_time
        bkg_counts = bkg_count_rates * n_exposure * exposure_time
        dark_counts = dark_current * n_exposure * exposure_time
        
        dark_counts = np.full(counts.shape, dark_counts)
        readout = np.full(counts.shape, read_out)
        
        counts_val, noise_counts_val = _add_noise_numba(
            counts, bkg_counts, dark_counts,
            readout, n_exposure
        )
        
        return counts_val, noise_counts_val
    
    def _rebin_fluxes_and_noise_array(
        self,
        wavelengths_in: np.ndarray,
        fluxes_in_array: np.ndarray,
        noise_in_array: np.ndarray,
        sed_in_array: np.ndarray,
        wavelengths_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        new_shape = (fluxes_in_array.shape[0], fluxes_in_array.shape[1], len(wavelengths_out) - 1)
        
        fluxes_out_array = np.zeros(new_shape, dtype=fluxes_in_array.dtype)
        noise_out_array = np.zeros(new_shape, dtype=noise_in_array.dtype)
        sed_out_array = np.zeros(new_shape, dtype=sed_in_array.dtype)
        
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                
                fluxes_out, noise_out, sed_out = _rebin_fluxes_and_noise_numba(
                    wavelengths_in, fluxes_in_array[i, j], noise_in_array[i, j]**2,
                    sed_in_array[i, j], wavelengths_out)
                fluxes_out_array[i, j] = fluxes_out
                noise_out_array[i, j] = noise_out
                sed_out_array[i, j] = sed_out
                
        return fluxes_out_array, noise_out_array, sed_out_array

    
    def _convert_to_flux(self, wavelengths: np.ndarray, counts: np.ndarray, noise_counts: np.ndarray,
                         sed_counts: np.ndarray, n_exposure: int, exposure_time: u.Quantity):
        
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
    
    def _process_spectral_dimension(
        self, 
        msa_array: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        
        spectrum_counts_array = np.zeros((
            self.n_pixels_perpendicular, 
            self.n_pixels_parallel, 
            self.waves_at_pixel.shape[0] - 1
        ))
        noise_counts_array = np.zeros((
            self.n_pixels_perpendicular, 
            self.n_pixels_parallel, 
            self.waves_at_pixel.shape[0] - 1
        ))
        sed_counts_array = np.zeros((
            self.n_pixels_perpendicular, 
            self.n_pixels_parallel, 
            self.waves_at_pixel.shape[0] - 1
        ))
        
        exposure_time = self.config['exposureTime'].to_value(u.s)
        aperture = self.config['aperture'].to_value(u.m)
        pixel_scale = self.config['pixelScale'].to_value(u.arcsec)
        numSpecWaveBins = self.config['numSpecWaveBins']
        
        flux_bkg = self.interp_bkg(self.wavelengths)
        bkg_count_rates = _count_rate_numba(
            self.wavelengths, flux_bkg, self.waves_at_pixel,
            self.throughput_dict, self.count_constants, numSpecWaveBins,
            aperture, pixel_scale
        )
        
        for i, j in product(range(self.n_pixels_perpendicular), 
                            range(self.n_pixels_parallel)):
            
            n_exposure = self.n_exposure_array[i, j]
            
            # print(f'{i}, {j}: {n_exposure}')
            
            if n_exposure == 0:
                spectrum_counts_array[i, j] = np.zeros_like(bkg_count_rates)
                noise_counts_array[i, j] = np.zeros_like(bkg_count_rates)
                sed_counts_array[i, j] = np.zeros_like(bkg_count_rates)
                continue
        
            spectrum_count_rates = _count_rate_numba(
                self.wavelengths, msa_array[:, i, j], self.waves_at_pixel,
                self.throughput_dict, self.count_constants, numSpecWaveBins, 
                aperture, pixel_scale
            )
            
            idx = spectrum_count_rates < 0
            spectrum_count_rates[idx] = 0
            
            sed_counts = spectrum_count_rates * n_exposure * exposure_time
            
            dark_current = self.config['darkCurrent'].to_value(u.s**-1)
            read_out = self.config['readOut']
            
            spectrum_counts, noise_counts = self._add_noise_static(
                spectrum_count_rates, bkg_count_rates,
                n_exposure, exposure_time,
                dark_current, read_out
            )
            
            spectrum_counts_array[i, j] = spectrum_counts
            noise_counts_array[i, j] = noise_counts
            sed_counts_array[i, j] = sed_counts
        
        return spectrum_counts_array, noise_counts_array, sed_counts_array
    

    def _display_exposures(self, displaySpectrumAtPixel: list, save_path: str):
        
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
    
    def _display_observation(self, dataTensor: np.ndarray,
                             displaySpectrumAtPixel: list, save_path: str):
        
        ny, nx = dataTensor.shape[0], dataTensor.shape[1]
        
        
        # total flux over all wavelengths
        display_array = np.sum(
            dataTensor[:, :, 1], axis=2)
        
        unit_type = self.config['unit']
        unit_comment_dict = {
            'electron': 'electrons',
            'flux_lambda': 'erg / cm^2 / s / angstrom',
            'flux_nu': 'Jy',
        }
        
        # Calculate coverage in arcsec
        y_coverage = self.config['ditherSize'].to_value(u.arcsec) * ny
        x_coverage = self.config['slitletSizeParallel'].to_value(u.arcsec) * nx
        
        x_tick_min = -x_coverage / 2
        x_tick_max = x_coverage / 2
        
        y_tick_min = -y_coverage / 2
        y_tick_max = y_coverage / 2
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        im = ax.imshow(display_array, origin='lower', cmap='gray_r')
        ax.set_xlim(0, nx - 1)
        ax.set_ylim(0, ny - 1)
        ax.set_title(f'Accumulated flux over all wavelengths')
        ax.set_aspect('auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)
        fig.colorbar(im, cax=cax, label=f'Flux [{unit_comment_dict[unit_type]}]')
        
        # Set ticks and labels in arcsec
        ax.set_xticks(np.linspace(0, nx - 1, num=11))
        ax.set_yticks(np.linspace(0, ny - 1, num=11))
        ax.set_xticklabels([f"{v:.1f}" for v in np.linspace(x_tick_min, x_tick_max, 11)])
        ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(y_tick_min, y_tick_max, 11)])
        ax.set_xlabel("Parallel direction (arcsec)")
        ax.set_ylabel("Perpendicular direction (arcsec)")
        
        for i, (y_pix, x_pix) in enumerate(displaySpectrumAtPixel):
            # x = ((x_pix + 1) + x_pix) / 2
            # y = ((y_pix + 1) + y_pix) / 2
            ax.scatter(x_pix, y_pix, s=10, label=f'Spectrum {i}')
        
        ax.legend()
        plt.tight_layout()
        savefilename = os.path.join(save_path, 'MSA_observation.png')
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
        plt.close()
        
    
    def _display_MSA_obs(self, dataTensor: np.ndarray,
                         cube_display: np.ndarray, 
                         slice_wavelength: np.ndarray,
                         displaySpectrumAtPixel: list,
                         save_path: str):
        
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
        cube_display = self._shift_image_static(
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
    def _shift_image_static(cube_slice: np.ndarray, dy: int, dx: int) -> np.ndarray:
        
        # 2d image shift
        
        h, w = cube_slice.shape
        result = np.zeros_like(cube_slice) # 2d array
        
        y1 = max(0, dy)
        y2 = h + min(0, dy)
        x1 = max(0, dx)
        x2 = w + min(0, dx)
        
        result[y1:y2, x1:x2] = cube_slice[y1 - dy:y2 - dy, x1 - dx:x2 - dx]
        return result
    
    @staticmethod
    def _downsample_static(cube_slice: np.ndarray, oversample: int) -> np.ndarray:
        
        
        if len(cube_slice.shape) == 2:
            down_factor = (oversample, oversample)
        elif len(cube_slice.shape) == 3:
            down_factor = (1, oversample, oversample)
            
        cube_slice = downscale_local_mean(cube_slice, down_factor)
        return cube_slice
    
    def _padding_cube(self, cube: np.ndarray, target_shape: tuple) -> np.ndarray:
        
        if cube.ndim != len(target_shape):
            raise ValueError("size must have the same length as cube.shape")
        
        padding_size = []
        for i in range(len(cube.shape)):
            if target_shape[i] is None:
                padding_size.append((0, 0))
            else:
                
                if cube.shape[i] < target_shape[i]:
                
                    padding_needed = target_shape[i] - cube.shape[i]
                    before_padding_size = padding_needed // 2
                    after_padding_size = padding_needed - before_padding_size
                
                else:
                    
                    before_padding_size = 0
                    after_padding_size = 0
                    
                padding_size.append((before_padding_size, after_padding_size))
        
        padding_size = tuple(padding_size)
        padded_cube = np.pad(cube, padding_size, mode='constant', constant_values=0)
        return padded_cube
    
    @staticmethod
    def _padding_static(cube_slice: np.ndarray, target_shape: tuple) -> np.ndarray:
        
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
        
        
        padded_cube_slice = self._padding_static(cube_slice, target_size)
        cropped_cube_slice = self._center_crop(
            padded_cube_slice, target_size
        )

        return cropped_cube_slice
    
    @staticmethod
    def _center_crop(cube_slice: np.ndarray, target_shape: tuple) -> np.ndarray:
        
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
        
        n_pixel_perpendicular = self.n_pixels_perpendicular
        n_pixel_parallel = self.n_pixels_output_parallel
        
        hdulist = fits.HDUList()
        primary_hdu = fits.PrimaryHDU()
        hdulist.append(primary_hdu)
        
        for prop, array in properties_array.items():
            
            # only save central part 
            
            y_center_array = array.shape[0] // 2
            x_center_array = array.shape[1] // 2
            
            # For extracting n elements centered around center:
            # Start: center - n//2, End: center + (n+1)//2
            # This works correctly for both odd and even n
            perpendicular_slice = slice(
                y_center_array - n_pixel_perpendicular // 2,
                y_center_array + (n_pixel_perpendicular + 1) // 2
            )
            
            parallel_slice = slice(
                x_center_array - n_pixel_parallel // 2,
                x_center_array + (n_pixel_parallel + 1) // 2
            )
            
            array = array[perpendicular_slice, parallel_slice]
            
            hdu = fits.ImageHDU(array)
            # Add property name and unit to header
            hdu.header['PROP'] = prop
            unit = str(properties_units.get(prop, ''))
            hdu.header['UNIT'] = unit
            hdulist.append(hdu)
        
        filename = os.path.join(save_path, f'True_properties.fits')
        hdulist.writeto(filename, overwrite=True)
        
    def _display_properties(self, properties_array: dict,
                            properties_units: dict, save_path: str):
        
        fig, ax = plt.subplots(1, len(properties_array),
                               figsize=(5 * len(properties_array), 4))
        
        if len(properties_array) == 1:
            ax = [ax]
        
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
            ax[i].imshow(array, origin='lower', aspect='auto',
                         cmap='gray_r')
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
            
            # Draw rectangle with specified size in arcsec
            # Rectangle bounds in arcsec
            rect_x_min_arcsec = -self.size_parallel.to_value(u.arcsec) / 2
            rect_x_max_arcsec = self.size_parallel.to_value(u.arcsec) / 2
            rect_y_min_arcsec = -self.size_perpendicular.to_value(u.arcsec) / 2
            rect_y_max_arcsec = self.size_perpendicular.to_value(u.arcsec) / 2
            
            # Convert arcsec to pixel coordinates
            x_pixel_min = (rect_x_min_arcsec - x_tick_min) / (x_tick_max - x_tick_min) * (array.shape[1] - 1)
            x_pixel_max = (rect_x_max_arcsec - x_tick_min) / (x_tick_max - x_tick_min) * (array.shape[1] - 1)
            y_pixel_min = (rect_y_min_arcsec - y_tick_min) / (y_tick_max - y_tick_min) * (array.shape[0] - 1)
            y_pixel_max = (rect_y_max_arcsec - y_tick_min) / (y_tick_max - y_tick_min) * (array.shape[0] - 1)
            
            # Create rectangle patch
            rect_width = x_pixel_max - x_pixel_min
            rect_height = y_pixel_max - y_pixel_min
            rectangle = patches.Rectangle(
                (x_pixel_min, y_pixel_min),
                rect_width,
                rect_height,
                linewidth=2,
                edgecolor='red',
                linestyle='--',
                alpha=0.5,
                facecolor='none'
            )
            ax[i].add_patch(rectangle)
        
        plt.tight_layout()
        savefilename = os.path.join(save_path, f'True_properties.png')
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_properties(self, properties: list[str]=None,
                             functions: Union[None, list[callable]]=[None]):
        
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
        
        n_pixels_perpendicular_display = int(self.config['displayBoxSize'] / self.config['ditherSize'])
        
        bins_perpendicular = np.linspace(-phyRes_perpendicular * (n_pixels_perpendicular_display // 2),
                                         phyRes_perpendicular * (n_pixels_perpendicular_display // 2), 
                                         n_pixels_perpendicular_display + 1)
        
        n_pixels_parallel_display = int(self.config['displayBoxSize'] / self.config['slitletSizeParallel'])
        
        if n_pixels_parallel_display % 2 == 0:
            bins_parallel = np.linspace(-phyRes_parallel * (n_pixels_parallel_display // 2),
                                        phyRes_parallel * (n_pixels_parallel_display // 2), 
                                        n_pixels_parallel_display + 1)
        else:
            half = phyRes_parallel / 2
            bins_parallel = np.linspace(-phyRes_parallel * (n_pixels_parallel_display // 2) - half, 
                                        phyRes_parallel * (n_pixels_parallel_display // 2) + half,
                                        n_pixels_parallel_display + 1)
        
        # for debug
        # bins_perpendicular = np.linspace(-40, 40, 100) # kpc
        # bins_parallel = np.linspace(-40, 40, 100) # kpc
        
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
        
    def create_MSA_dataTensor(self):
        
        """
        Create the data tensor for the MSA simulation, including signal, noise, and sed arrays.

        This method orchestrates the full creation process for the MSA (Multi-Shutter Array) data tensor.
        It performs the following major steps:

        - Define the wavelength sampling based on instrument resolution
        - Retrieve or generate the relevant PSF cube
        - Simulate detector sampling over the spatial and spectral grids
        - Compute signal, noise, and sed arrays for each slitlet
        - Assemble all outputs into a single, well-formatted data tensor (4 dimensions)

        Returns
        -------
        dataTensor : np.ndarray
            The full (n_slitlet_y, n_slitlet_x, 4, n_bins) array, ready for export or further use.
            
            Array structure:
            
            - data[...,0,:] = wavelengths (middle points)
            - data[...,1,:] = signal (flux or electron counts)
            - data[...,2,:] = noise (flux or electron counts)
            - data[...,3,:] = sed (flux or electron counts)
        """

        self.logger.info('MSA-3D simulation starts...')
        
        self._get_waves_by_resolution()
        self._get_waves_at_pixel()
        
        self.logger.info(f'Step 1: Processing spatial dimension (PSF convolution and transformation)')
        
        start_time = time.time()
        
        self.logger.info('Getting PSF cubes...')
        self._get_PSF_cube()
        
        self.logger.info(f'PSF cube shape: {self.psf_cube.shape}')
        self.logger.info(f'Data cube shape: {self.dataCube.shape}')
        
        msa_array = self._process_spatial_dimension()
        self.logger.info(f'MSA cube shape after spatial processing: {msa_array.shape}')
        
        end_time = time.time()
        self.logger.info(f'Time taken for step 1: {end_time - start_time:.2f} seconds')
        
        self.logger.info(f'Step 2: Processing spectral dimension (Simulation of spectra)')
        
        self.logger.info('Getting JWST background...')
        self._get_bkg()
        
        start_time = time.time()
        
        spectrum_counts_array, noise_counts_array, sed_counts_array =\
            self._process_spectral_dimension(msa_array)
        
        # input: 40, 18, num_wave, output: 40, 9, num_wave
        spectrum_counts_array, noise_counts_array, sed_counts_array = _overlap_counts_numba(
            spectrum_counts_array, noise_counts_array, sed_counts_array, self.n_pixels_slitlet_parallel
        )

        if self.config['unit'] != 'electron':
            
            self.logger.info(f'Convert unit to {self.config["unit"]}...')
            
            signal_array, noise_array, sed_array = self._convert_to_flux_array(
                self.waves_at_pixel, spectrum_counts_array, noise_counts_array, sed_counts_array
            )
        else:
            signal_array = spectrum_counts_array
            noise_array = noise_counts_array
            sed_array = sed_counts_array
        
        # 40, 9, num_wave
        signal_array, noise_array, sed_array = self._rebin_fluxes_and_noise_array(
            self.waves_at_pixel, signal_array, noise_array, sed_array, self.waves_by_resolution
        )
        
        dataTensor = self._assemble_data(
            self.waves_by_resolution, signal_array, noise_array, sed_array
        )
        
        end_time = time.time()
        self.logger.info(f'Time taken for step 2: {end_time - start_time:.2f} seconds')
        
        self.logger.info(f'MSA-3D simulation completed. Output shape: {dataTensor.shape}')
        
        savefilename = os.path.join(self.save_path, f'MSA_dataTensor.fits')
        self.logger.info(f'Saving data tensor to {savefilename}...')
        self._save_dataTensor(dataTensor, savefilename)
        
    def illustrate_MSA(self, dataTensor_path: Union[None, str]=None):
        
        """
        Illustrate the MSA (Micro-Shutter Array) data.

        This method generates a series of visualizations to illustrate the simulated MSA data,
        including exposure maps, spectral properties, emission line maps, and observed mosaics.
        The method can take an optional path to a data tensor FITS file; if none is provided, it
        loads the default output in the save path.

        Parameters
        ----------
        dataTensor_path : str or None, optional
            The path to the data tensor FITS file. If None, uses default output location.

        """
        
        self.logger.info('Illustrating MSA...')
        
        if dataTensor_path is not None:
            dataTensor_filename = dataTensor_path
        else:
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
        self._display_observation(dataTensor, displaySpectrumAtPixel, 
                                  self.save_path)
    
    def get_truth_properties(self, keep_defaults: bool=True, properties: list[str]=None, 
                             functions: Union[None, list[callable]]=[None]):
        
        """
        Retrieve, calculate and save ground-truth properties for the simulation.

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
                                  'Mass', 'VelDisp', 'NumberStarformingRegion']
            default_functions = [property_metallicity_for_starforming_region, 
                                property_gas_metallicity, property_los_velocity, 
                                property_total_stellar_mass, property_velocity_dispersion, 
                                property_number_starforming_region]
            
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