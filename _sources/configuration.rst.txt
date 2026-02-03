Configuration
=============

This page documents all configuration options available in ``config.toml`` and ``config_MSA.toml`` files.

Main Config
-----------

The main configuration file (``config.toml``) contains settings for data retrieval, SKIRT simulation, and general processing parameters.

Data Retrieval
~~~~~~~~~~~~~~

``useRequests`` (bool):
    Whether to process subhalos on the fly using Web-based API requests.

``apiKey`` (str):
    Your API key for TNG simulations. Required if ``useRequests`` is ``true``.

``simulation`` (str):
    Simulation name. Options: ``TNG50-1``, ``TNG100-1``, ``TNG300-1``, etc.

``hydrodynamicSolver`` (str):
    Hydrodynamic solver for gas particles. Can be ``VoronoiMesh`` (for TNG) or ``smoothParticle`` (for EAGLE).

``TNGPath`` (str):
    Home path for TNG simulation data on local filesystem.

``postprocessingPath`` (str):
    Home path for postprocessing data of TNG simulations on local filesystem. Both ``TNGPath`` and ``postprocessingPath`` are needed when reading data locally.

SKIRT Simulation Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``workingDir`` (str):
    Working directory for running SKIRT code.

``simulationMode`` (str):
    Simulation mode for SKIRT. Options:
    
    * ``ExtinctionOnly``: Used when extinction and scattering should be considered
    * ``DustEmission``: Used when extinction and scattering along with secondary emission should be considered
    * ``NoMedium``: An ideal case with no dust elements
    
``includeDust`` (bool):
    Whether dust particles are considered. Should be ``true`` if ``simulationMode`` is ``ExtinctionOnly`` or ``DustEmission``.

``dustEmissionType`` (str):
    Dust emission type. Options: ``Equilibrium`` or ``Stochastic``. Only used if ``simulationMode`` is ``DustEmission``.

``dustModel`` (str):
    Dust model. Options: ``ZubkoDustMix``, ``DraineLiDustMix``, or ``ThemisDustMix``.

``includeVelocity`` (bool):
    Whether to include velocity information for particle data.

Wavelength Settings
~~~~~~~~~~~~~~~~~~~

``minWaveRT`` (str):
    Minimum wavelength for radiative transfer, *not* in rest-frame. Must include units (e.g., ``"0.1 um"``). 

``maxWaveRT`` (str):
    Maximum wavelength for radiative transfer, *not* in rest-frame. Must include units (e.g., ``"2 um"``).

``numWaveRT`` (int):
    Number of wavelength bins for radiative transfer.

``waveGridRT`` (str):
    Wavelength grid type for radiative transfer. Options: ``Linear`` or ``Log``.

``minWaveOutput`` (str):
    Minimum wavelength for output, *not* in rest-frame. Must include units (e.g., ``"0.97 um"``).

``maxWaveOutput`` (str):
    Maximum wavelength for output, *not* in rest-frame. Must include units (e.g., ``"1.82 um"``).

``numWaveOutput`` (int):
    Number of wavelength bins for output.

``waveGridOutput`` (str):
    Wavelength grid type for output. Options: ``Linear`` or ``Log``.

Spatial Settings
~~~~~~~~~~~~~~~~

``boxLengthScale`` (float):
    Particles are selected in a box with length = ``halfStellarMassRadius * boxLengthScale``.

``maxBoxLength`` (str):
    Maximum box length, in kpc. Must include units (e.g., ``"300 kpc"``).

``minLevel`` (int):
    Minimum octree level refinement for dust calculation in SKIRT.

``maxLevel`` (int):
    Maximum octree level refinement for dust calculation in SKIRT.

``numPackets`` (float):
    Number of photon packets. Main parameter affecting the SNR of SKIRT simulation.

Stellar Population Settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``starSEDFamily`` (str):
    SED family for quenched star particles (age > 10 Myr). Options: ``BC03`` or ``FSPS``.

``initialMassFunction`` (str):
    Initial mass function for quenched star particles:
    
    * For ``BC03`` SED family: ``Chabrier`` or ``Salpeter``
    * For ``FSPS`` SED family: ``Chabrier``, ``Kroupa``, or ``Salpeter``
    

``starformingSEDFamily`` (str):
    SED family for star-forming regions. Options: ``MAPPINGS`` or ``TODDLERS``.

``ageThreshold`` (str):
    Age threshold for discriminating star-forming or quenched star particles, in Myr. Must include units (e.g., ``"10 Myr"``).

MAPPINGS Settings (for star-forming regions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``logCompactnessMean`` (float):
    Logarithmic mean of compactness for star-forming star particles. See `Kapoor et al. 2021 <https://academic.oup.com/mnras/article/506/4/5703/6324023>`_.

``logCompactnessStd`` (float):
    Logarithmic standard deviation of compactness for star-forming star particles. See `Kapoor et al. 2021 <https://academic.oup.com/mnras/article/506/4/5703/6324023>`_.

``logPressure`` (float):
    Logarithmic pressure for star-forming star particles. log10[(Pressure/k_B)/cm^-3 K] = logPressure.

``constantCoveringFactor`` (bool):
    Whether to use constant covering factor.

``coveringFactor`` (float):
    Constant covering factor, if ``constantCoveringFactor`` is ``true``. From `Groves et al. 2008 <https://iopscience.iop.org/article/10.1086/528711>`_.

``PDRClearingTimescale`` (str):
    PDR clearing timescale, in Myr. Covering factor is f = e^(-t / PDRClearingTimescale) if ``constantCoveringFactor`` is ``false``. From `Baes et al. 2024 <https://www.aanda.org/articles/aa/full_html/2024/12/aa51207-24/aa51207-24.html>`_. Must include units (e.g., ``"3 Myr"``).

TODDLERS Settings (for star-forming regions)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``pahfraction`` (str):
    PAH fraction. Only used when ``starformingSEDFamily`` is ``TODDLERS``. Options: ``"High"``, ``"Medium"``, ``"Low"``.

``sedMode`` (str):
    SED mode for TODDLERS. Options: ``SFRNormalized`` or ``Cloud``.

``stellarTemplate`` (str):
    Stellar template for TODDLERS. Options: ``SB99Kroupa100Sin``, ``BPASSChab100Bin``, or ``BPASSChab300Bin``.

``sfrPeriod`` (int):
    SFR period in Myr. Only used in ``SFRNormalized`` mode. Can be ``10`` or ``30``.

``starFormationEfficiency`` (float):
    Star formation efficiency for TODDLERS.

``cloudNumberDensity`` (str):
    Cloud number density for TODDLERS, in cm^-3. Must include units (e.g., ``"320 cm^-3"``).

``alpha`` (float):
    Alpha parameter for power-law distribution of cloud mass for TODDLERS.

``scaling`` (float):
    Scaling factor for TODDLERS.

Dust ISM Settings
~~~~~~~~~~~~~~~~~

``temperatureThreshold`` (str):
    Temperature threshold in K. Gas particles with temperature lower than this threshold will be considered as dust. Must include units (e.g., ``"8000 K"``).

``massFraction`` (float):
    Fraction of the metallic gas locked up in dust.

``DISMModel`` (str):
    Dust-containing ISM (DISM) model. Options: ``Camps_2016`` or ``Torrey_2012``.

``numSilicateSizes`` (int):
    Number of bins for silicate dust grains.

``numGraphiteSizes`` (int):
    Number of bins for graphite dust grains.

``numPAHSizes`` (int):
    Number of bins for PAH dust grains.

``numHydrocarbonSizes`` (int):
    Number of bins for hydrocarbon dust grains.

Subhalo Selection
~~~~~~~~~~~~~~~~~

``minStellarMass`` (str):
    Minimum stellar mass for subhalos, in Msun. Must include units (e.g., ``"1e10 Msun"``).

``maxStellarMass`` (float or str):
    Maximum stellar mass for subhalos, in Msun. Use ``inf`` for infinite.

Viewing Angles
~~~~~~~~~~~~~~

``faceAndEdge`` (bool):
    Whether to use face-on and edge-on angles derived by angular momentum.

``numViews`` (int):
    Number of instrument views (observing directions).

``randomViews`` (bool):
    Whether to generate views from uniform distribution.

``inclinations`` (list[int]):
    Inclination angles for instrument views (in degrees). Required when ``randomViews`` is ``false``. Separated by commas.

``azimuths`` (list[int]):
    Azimuth angles for instrument views (in degrees). Required when ``randomViews`` is ``false``. Separated by commas.

Output Settings
~~~~~~~~~~~~~~~

``fieldOfView`` (float):
    Field of view for output, in arcsec. Equals box size if ``0``.

``pixelScale`` (str):
    Pixel scale for detector. Must include units (e.g., ``"0.1 arcsec"``).

``oversample`` (int):
    Oversample factor. Oversamples both dataCube (SKIRT output) and PSF.

``unit`` (str):
    Unit type of image output. Options: ``electron``, ``flux_lambda``, or ``flux_nu``.

    * ``electron``: electrons
    * ``flux_lambda``: erg / cm^2 / s / angstrom
    * ``flux_nu``: Jy

Redshift and Distance
~~~~~~~~~~~~~~~~~~~~~

``snapNum`` (int):
    Snapshot ID.

``snapRedshift`` (float):
    Snapshot redshift. Must be provided if ``inLocal`` is ``false``.

``viewRedshift`` (float):
    Viewing redshift, should be close to ``snapRedshift``. For generating galaxies at continuous redshift with a small offset on ``snapRedshift``. For example, ``viewRedshift`` can be ``0.105`` for snapshot-91 at ``0.1``.

``inLocal`` (bool):
    Whether to view galaxy in local cosmology.

``viewDistance`` (str):
    Viewing distance, in Mpc. Must be provided if ``inLocal`` is ``true``. Must include units (e.g., ``"50 Mpc"``).

Performance Settings
~~~~~~~~~~~~~~~~~~~~

``numThreads`` (int):
    Number of threads. No speedup for threads larger than 24.


MSA Config
----------

The MSA configuration file (``config_MSA.toml``) contains settings specific to JWST/NIRSpec Micro-Shutter Assembly (MSA) observations.

Instrument Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

``disperser`` (str):
    NIRSpec disperser. 

``filter`` (str):
    NIRSpec filter. 

``aperture`` (str):
    Telescope aperture size. Must include units (e.g., ``"6.5 m"``).

Slitlet Configuration
~~~~~~~~~~~~~~~~~~~~~

``offsetParallel`` (float):
    Offset from center in parallel direction (dispersion direction), in arcsec.

``offsetPerpendicular`` (float):
    Offset from center in perpendicular direction (cross-dispersion direction), in arcsec.

``slitletSizePerpendicular`` (str):
    Slitlet size in perpendicular direction (cross-dispersion), in arcsec. Must include units (e.g., ``"0.46 arcsec"``).

``slitletSizeParallel`` (str):
    Slitlet size in parallel direction (dispersion), in arcsec. Must include units (e.g., ``"0.2 arcsec"``).

``supportBarSize`` (str):
    Support bar size between slitlets, in arcsec. Must include units (e.g., ``"0.07 arcsec"``).

``rotate`` (str):
    Rotation of the slitlets, counter-clockwise. Must include units (e.g., ``"0 deg"``).

Observation Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

``nDithers`` (int):
    Number of dither operations.

``ditherSize`` (str):
    Dither size, in arcsec. Must include units (e.g., ``"0.075 arcsec"``).

``nSlitlets`` (int):
    Number of slitlets.

``nSteps`` (int):
    Number of steps in parallel direction.

``exposureTime`` (str):
    Single exposure time. Must include units (e.g., ``"19.5 min"``).

Background Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

``useJBT`` (bool):
    Whether to use background from ``jwst_backgrounds`` package.

``obsRA`` (str):
    Observation right ascension. Must include units (e.g., ``"223.555 deg"``). Required if ``useJBT`` is ``true``.

``obsDec`` (str):
    Observation declination. Must include units (e.g., ``"-54.395 deg"``). Required if ``useJBT`` is ``true``.

``threshold`` (float):
    Threshold parameter for background calculation.

``thisDay`` (int):
    Day of year for background calculation (1-365).

Detector Settings
~~~~~~~~~~~~~~~~~

``darkCurrent`` (str):
    Dark current in electrons per second. Units should be specified as ``"/s"`` (e.g., ``"0.01 /s"``).

``readOut`` (float):
    Readout noise in electrons.

Spectral Settings
~~~~~~~~~~~~~~~~~

``numSpecWaveBins`` (int):
    Number of wavelength bins for spectrum rebinning at each pixel of detector.

Display Settings
~~~~~~~~~~~~~~~~

``displaySpectrumAtPixel`` (list[list[int]]):
    List of pixel coordinates ``[y, x]`` where spectra should be displayed.

``displayBoxSize`` (str):
    Box size for display, in arcsec. Must include units (e.g., ``"4.0 arcsec"``).

``displayEmissionLines`` (list[str]):
    List of emission lines to display. Muts in format "line_name wavelength in angstrom". For example, ``"Ha6563"`` and ``"[SII]6716"``.

``displayAround`` (str):
    Around which overlapping cube slice displays. 