import numpy as np
import h5py
import astropy.units as u
import numba
from astropy.cosmology import Cosmology, Planck15
from scipy.interpolate import interp1d

@numba.njit(cache=True, fastmath=True)
def rotate_coordinates(coordinates: np.ndarray,
                        inclination: np.float32, 
                        azimuth: np.float32) -> np.ndarray:
    
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
    
    # rotated_coords[:, 2] = -rotated_coords[:, 2]
    
    return rotated_coords

@numba.njit(cache=True, fastmath=True)
def calc_los_velocity(velocities: np.ndarray, 
                    inclination: np.float32, azimuth: np.float32) -> np.ndarray:
    
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

@numba.njit(cache=True, fastmath=True)
def calc_rotational_velocity(coordinates: np.ndarray, 
                            velocities: np.ndarray, 
                            inclination: np.float32, 
                            azimuth: np.float32, 
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

@numba.njit(cache=True, fastmath=True)
def transform(coords: np.ndarray, rotate: np.float32, 
                shiftPerpendicular: np.float32, shiftParallel: np.float32):
    
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

def age_interp(cosmology: Cosmology=Planck15) -> interp1d:
    z = np.linspace(0, 4, 1000)
    t = cosmology.age(z).to(u.Myr).value
    fage = interp1d(z, t, kind='cubic',
                    bounds_error=False, fill_value='extrapolate')
    return fage

@numba.njit(cache=True, fastmath=True, parallel=True)
def binned_statistic_2d_numba(
    x, y, values, bins_perpendicular, 
    bins_parallel, statistic='mean'):
    
    # x direction is perpendicular, y direction is parallel
    x = -x
    y = -y
    
    # Extract number of bins from bin edge arrays (length n+1 for n bins)
    nx = int(len(bins_perpendicular) - 1)
    ny = int(len(bins_parallel) - 1)
    
    if statistic in ['mean', 'sum', 'std']:
        if values is None:
            raise ValueError("values must be provided for statistic in ['mean', 'sum', 'std']")
    
    n = len(x)
    
    # Initialize buffers
    # res will store the primary statistic (sum, count, or mean)
    res = np.zeros((nx, ny), dtype=np.float32)
    
    # We need extra buffers for mean and std
    if statistic == 'mean' or statistic == 'std':
        counts = np.zeros((nx, ny), dtype=np.int32)
    if statistic == 'std':
        sum_sq = np.zeros((nx, ny), dtype=np.float32)
        sums = np.zeros((nx, ny), dtype=np.float32)

    for i in numba.prange(n):
        ix = np.searchsorted(bins_perpendicular, x[i], side='right') - 1
        iy = np.searchsorted(bins_parallel, y[i], side='right') - 1
        
        if ix >= nx:
            ix = nx - 1
        elif ix < 0:
            ix = 0
            
        if iy >= ny:
            iy = ny - 1
        elif iy < 0:
            iy = 0
        
        if 0 <= ix < nx and 0 <= iy < ny:
            
            if values is not None:
                val = values[i]
            else:
                val = 1.0
        
            if statistic == 'sum':
                res[ix, iy] += val
            elif statistic == 'count':
                res[ix, iy] += 1
            elif statistic == 'mean':
                res[ix, iy] += val
                counts[ix, iy] += 1
            elif statistic == 'std':
                sums[ix, iy] += val
                sum_sq[ix, iy] += val * val
                counts[ix, iy] += 1
                
    # Post-processing for Mean and Std
    if statistic == 'mean':
        for i in range(nx):
            for j in range(ny):
                if counts[i, j] > 0:
                    res[i, j] /= counts[i, j]
                else:
                    res[i, j] = np.nan
                    
    elif statistic == 'std':
        for i in range(nx):
            for j in range(ny):
                c = counts[i, j]
                if c > 1:
                    # Variance = (SumSq / n) - (Mean^2)
                    mean = sums[i, j] / c
                    variance = (sum_sq[i, j] / c) - (mean ** 2)
                    # Use max(0, var) to prevent tiny negative numbers from float imprecision
                    res[i, j] = np.sqrt(max(0.0, variance))
                elif c == 1:
                    res[i, j] = 0.0 # Std of a single point is 0
                else:
                    res[i, j] = np.nan
                    
    return res

def calc_stats(coords: np.ndarray, values: np.ndarray,
                bins_perpendicular: np.ndarray,
                bins_parallel: np.ndarray,
                statistic: str) -> np.ndarray:
    
    if values is not None:
        values = values.astype(np.float32)
        
    coords = coords.astype(np.float32)
    
    bins_perpendicular = bins_perpendicular.astype(np.float32)
    bins_parallel = bins_parallel.astype(np.float32)
    
    stats = binned_statistic_2d_numba(
        coords[:, 1], coords[:, 0], values, 
        bins_perpendicular=bins_perpendicular,
        bins_parallel=bins_parallel,
        statistic=statistic
    )
    return stats

def property_metallicity_for_starforming_region(
    particle_file: str, inclination: np.float32, azimuth: np.float32, 
    bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
    transformation: dict, subhalo_info: dict, configs: dict):
    
    fage = age_interp(configs['cosmology'])
    
    particles = {}
    with h5py.File(particle_file, 'r') as file:
        
        particles['GFM_StellarFormationTime'] = file['PartType4']['GFM_StellarFormationTime'][:].astype(np.float32)
        snapshot_age = fage(configs['snapRedshift']) # in Myr
        particles['age'] = snapshot_age - fage(1/particles['GFM_StellarFormationTime'] - 1)
        
        # star formation region mask
        mask = np.where((particles['age'] < 10) & (particles['GFM_StellarFormationTime'] > 0))
        
        particles['Coordinates'] = file['PartType4']['Coordinates'][mask][:].astype(np.float32)
        unit = file['PartType4']['Coordinates'].attrs['unit']
        particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
        
        particles['GFM_Metallicity'] = file['PartType4']['GFM_Metallicity'][mask][:].astype(np.float32)
        unit_metallicity = "1"

    # rotate to observing direction
    coords = rotate_coordinates(
        particles['Coordinates'], inclination, azimuth)
    
    # apply transformation for coords in x-y plane if rotate and shifts in config is not 0
    coords = transform(coords, transformation['rotate'], 
                                    transformation['shiftPerpendicular'],
                                    transformation['shiftParallel'])
    
    
    metallicity = particles['GFM_Metallicity']
    stats = calc_stats(
        coords, metallicity, 
        bins_perpendicular, bins_parallel, statistic='mean')
    
    return stats, unit_metallicity

def property_gas_metallicity(particle_file: str, inclination: np.float32, azimuth: np.float32, 
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
        
    coords = rotate_coordinates(
        particles['Coordinates'], inclination, azimuth)
    
    coords = transform(coords, transformation['rotate'], 
                                    transformation['shiftPerpendicular'],
                                    transformation['shiftParallel'])
    
    metallicity = particles['GFM_Metallicity']
    stats = calc_stats(
        coords, metallicity, 
        bins_perpendicular, bins_parallel, statistic='mean')
    
    return stats, unit_metallicity

def property_los_velocity(
    particle_file: str, inclination: np.float32, azimuth: np.float32, 
    bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
    transformation: dict, subhalo_info: dict, configs: dict):
    
    particles = {}
    with h5py.File(particle_file, 'r') as file:
        particles['Coordinates'] = file['PartType4']['Coordinates'][:].astype(np.float32)
        unit = file['PartType4']['Coordinates'].attrs['unit']
        particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
        
        particles['Velocities'] = file['PartType4']['Velocities'][:].astype(np.float32)
        unit_velocities = file['PartType4']['Velocities'].attrs['unit']
        
    coords = rotate_coordinates(
        particles['Coordinates'], inclination, azimuth)
    
    coords = transform(coords, transformation['rotate'], 
                                    transformation['shiftPerpendicular'],
                                    transformation['shiftParallel'])
    
    los_velocity = calc_los_velocity(
        particles['Velocities'], inclination, azimuth)
        
    stats = calc_stats(
        coords, los_velocity, 
        bins_perpendicular, bins_parallel, statistic='mean')
        
    return stats, unit_velocities

def property_velocity_dispersion(
    particle_file: str, inclination: np.float32, azimuth: np.float32, 
    bins_perpendicular: np.ndarray, bins_parallel: np.ndarray,
    transformation: dict, subhalo_info: dict, configs: dict):
    
    particles = {}
    with h5py.File(particle_file, 'r') as file:
        particles['Coordinates'] = file['PartType4']['Coordinates'][:].astype(np.float32)
        unit = file['PartType4']['Coordinates'].attrs['unit']
        particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
        
        particles['Velocities'] = file['PartType4']['Velocities'][:].astype(np.float32)
        unit_velocities = file['PartType4']['Velocities'].attrs['unit']
        
    coords = rotate_coordinates(
        particles['Coordinates'], inclination, azimuth)
    
    coords = transform(coords, transformation['rotate'], 
                                    transformation['shiftPerpendicular'],
                                    transformation['shiftParallel'])
    
    los_velocity = calc_los_velocity(
        particles['Velocities'], inclination, azimuth)
        
    stats = calc_stats(
        coords, los_velocity, 
        bins_perpendicular, bins_parallel, statistic='std')
        
    return stats, unit_velocities

def property_total_stellar_mass(
    particle_file: str, inclination: np.float32, azimuth: np.float32, 
    bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
    transformation: dict, subhalo_info: dict, configs: dict):
    
    particles = {}
    with h5py.File(particle_file, 'r') as file:
        particles['Coordinates'] = file['PartType4']['Coordinates'][:].astype(np.float32)
        unit = file['PartType4']['Coordinates'].attrs['unit']
        particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
        
        particles['Masses'] = file['PartType4']['Masses'][:].astype(np.float32)
        unit_masses = file['PartType4']['Masses'].attrs['unit']
        
    coords = rotate_coordinates(
        particles['Coordinates'], inclination, azimuth)
    
    coords = transform(coords, transformation['rotate'], 
                                    transformation['shiftPerpendicular'],
                                    transformation['shiftParallel'])
    
    masses = particles['Masses']
    
    stats = calc_stats(
        coords, masses, 
        bins_perpendicular, bins_parallel, statistic='sum')
    
    return stats, unit_masses

def property_number_starforming_region(
    particle_file: str, inclination: np.float32, azimuth: np.float32, 
    bins_perpendicular: np.ndarray, bins_parallel: np.ndarray, 
    transformation: dict, subhalo_info: dict, configs: dict):
    
    fage = age_interp(configs['cosmology'])
    
    particles = {}
    with h5py.File(particle_file, 'r') as file:
        
        particles['GFM_StellarFormationTime'] = file['PartType4']['GFM_StellarFormationTime'][:].astype(np.float32)
        snapshot_age = fage(configs['snapRedshift']) # in Myr
        particles['age'] = snapshot_age - fage(1/particles['GFM_StellarFormationTime'] - 1)
        
        # star formation region mask
        mask = np.where((particles['age'] < 10) & (particles['GFM_StellarFormationTime'] > 0))
        
        particles['Coordinates'] = file['PartType4']['Coordinates'][mask][:].astype(np.float32)
        unit = file['PartType4']['Coordinates'].attrs['unit']
        particles['Coordinates'] = (particles['Coordinates'] * u.Unit(unit)).to_value(u.kpc)
        
    coords = rotate_coordinates(
        particles['Coordinates'], inclination, azimuth
    )
    
    coords = transform(coords, transformation['rotate'], 
                                    transformation['shiftPerpendicular'],
                                    transformation['shiftParallel'])
    
    stats = calc_stats(
        coords, None, 
        bins_perpendicular, 
        bins_parallel, 
        statistic='count')
    
    return stats, '1'