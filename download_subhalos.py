"""
This script is for downloading subhalo particles and metadata from TNG simulations.

"""

import os
import numpy as np
import requests
import time
import sys
from astropy.cosmology import Planck15
import json

def make_request(url: str, headers: dict, max_retries: int = 5) -> requests.Response:
    """
    Makes an HTTP GET request to the specified URL with given headers, retrying up to `max_retries` times on failure.

    Parameters
    ----------
    url : str
        The URL to send the GET request to.
    headers : dict
        A dictionary of headers to include in the request.
    max_retries : int, optional
        The maximum number of retries in case of failure (default is 5).

    Returns
    -------
    requests.Response
        The Response object from the successful request.

    Raises
    ------
    SystemExit
        If all retry attempts fail, exits the script after printing error details.
    """
    start_time = time.time()
    content = None
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            content = response
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"Error: Failed to make request after {max_retries} attempts.")
                print(f"URL: {url}")
                print(f"Error message: {str(e)}")
            else:
                print(f"Request failed (attempt {attempt + 1}/{max_retries}). Retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff        
                
    end_time = time.time()
    print(f"Requesting time taken: {end_time - start_time:.2f} seconds")
    
    if content is None:
        sys.exit(1)
        
    return content


def get_snap_info(snapshot_id: int, stellarMass_ranges: tuple, headers: dict,
                  cosmo: object, simulation: str, save_path: str) -> list:
    
    """
    Retrieves snapshot information for a specific simulation with stellar mass filtering, and saves the results to disk.

    Parameters
    ----------
    snapshot_id : int or str
        The ID or label of the simulation snapshot to query.
    stellarMass_ranges : tuple or list-like of float
        A (min, max) range for stellar masses (in solar masses) to filter the subhalos.
    headers : dict
        HTTP request headers (typically includes authentication for the API).
    cosmo : object
        Cosmology object with an attribute 'h' for the Hubble parameter scaling.
    simulation : str
        The simulation name (e.g., 'IllustrisTNG', 'TNG100-1', etc.).
    save_path : str
        The local directory path where the JSON result file is saved.

    Returns
    -------
    list
        A list of dictionaries, each representing a subhalo in the snapshot filtered by stellar mass.

    Notes
    -----
    The function constructs the API endpoint according to stellar mass bounds (converted with the cosmology h parameter),
    retrieves results with HTTP GET, prints the number of subhalos found, saves results to a JSON file, and returns the data.
    """
    
    h = cosmo.h
    minStellarMass = np.float32(stellarMass_ranges[0]) / 10**10 * h
    maxStellarMass = np.float32(stellarMass_ranges[1]) / 10**10 * h
    
    base_url = 'https://www.tng-project.org/api/'
    
    url = f'{base_url}{simulation}/snapshots/{snapshot_id}' \
                + f'/subhalos/?mass_stars__gt={minStellarMass}&subhaloflag=1&limit=1000000'
    print('snapshot url: ', url)
    response = make_request(url, headers=headers)
    data = response.json()
    results = data.get('results', [])
    
    subhaloNum = len(results)
    print('Number of subhalos: ', subhaloNum)
    
    with open(os.path.join(save_path, f'snapshot-{snapshot_id}.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def get_subhalo_info(subhaloID: int, results: list, headers: dict) -> dict:
    
    """
    Retrieves detailed information about a specific subhalo from the simulation results, including its ID, URL, and data.

    Parameters
    ----------
    subhaloID : int
        The ID of the subhalo to retrieve information for.
    results : list
        A list of dictionaries, each representing a subhalo in the simulation.
    headers : dict
        HTTP request headers (typically includes authentication for the API).

    Returns
    -------
    dict
        A dictionary containing the subhalo's information, including its ID, URL, and data.

    Notes
    -----
    The function searches for the subhalo in the results list by its ID, constructs the subhalo's URL,
    makes an HTTP GET request to retrieve the subhalo's data, and returns the data as a dictionary.
    """
    
    subhaloIDs = [result['id'] for result in results]
    idx = list(subhaloIDs).index(subhaloID)
    
    id = subhaloIDs[idx]
    subhalo = results[idx]
    subhalo_url = subhalo['url']
    print('subhalo url: ', subhalo_url)
    response = make_request(subhalo_url, headers=headers)
    data = response.json()
    
    return data

def request_subhalo_particles(cutout_url: str, headers: dict, savefilename: str, max_retries: int = 5):
    """
    Downloads the particle data for a subhalo from the given cutout URL and saves it to a file.

    Parameters
    ----------
    cutout_url : str
        The URL endpoint to request the particle data for the subhalo (expected to return an HDF5 file).
    headers : dict
        HTTP request headers (must include any required authorization, such as an API key).
    savefilename : str
        The path (including file name) where the downloaded subhalo particle data will be saved.
    max_retries : int, optional
        The maximum number of retry attempts in case of failed requests (default is 5).

    Notes
    -----
    This function streams the file download for memory efficiency and handles temporary network or server
    errors by retrying up to `max_retries` times with exponential backoff. Raises an exception if all retries fail.
    """
    for attempt in range(max_retries):
        try:
            with requests.get(cutout_url, headers=headers, stream=True) as response:
                response.raise_for_status()
                with open(savefilename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                break
        except requests.exceptions.RequestException as e:
            print(f'Attempt {attempt + 1}/{max_retries} failed, retrying...')
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    

def get_particles(subhaloID: int, snapshot_id: int, headers: dict, simulation: str, save_path: str):
    """
    Downloads and saves particle data for a specific subhalo using its ID and snapshot.

    Parameters
    ----------
    subhaloID : int
        The ID of the subhalo for which to download the particle data.
    snapshot_id : int
        Snapshot identifier (corresponds to a specific simulation lookback time/redshift).
    headers : dict
        The headers for API authentication (e.g., {'api-key': ...}).
    simulation : str
        Name of the simulation (e.g., 'TNG50-1').
    save_path : str
        The directory path where the downloaded particle file should be saved.

    Notes
    -----
    The particle data is saved in HDF5 format with the filename:
        subhalo_<subhaloID>_particles.h5
    This function uses the IllustrisTNG API to retrieve the particle cutout.
    """
    
    base_url = 'https://www.tng-project.org/api/'
    
    cutout_url = f'{base_url}{simulation}/snapshots/{snapshot_id}/subhalos/{subhaloID}/cutout.hdf5'
    
    print('cutout url: ', cutout_url)
    request_subhalo_particles(cutout_url, headers, 
                              os.path.join(save_path, f'subhalo_{subhaloID}_particles.h5'))
    
    
    
if __name__ == '__main__':
    
    simulation = 'TNG50-1'
    snapshot_ids = [50]
    snapshot_zs = [1.0]
    minStellarMass = 10**10
    maxStellarMass = np.inf
    
    stellarMass_ranges = [minStellarMass, maxStellarMass]
    
    cosmo = Planck15
    api_key = 'your_api_key'
    headers = {'api-key': api_key}

    
    for snap_id, snap_z in zip(snapshot_ids, snapshot_zs):
        
        print('Processing snapshot', snap_id, 'at redshift', snap_z)
        
        save_path = f'data/snapshots-{snap_id}'
        os.makedirs(save_path, exist_ok=True)
        
        results = get_snap_info(snap_id, stellarMass_ranges, headers, cosmo, simulation, save_path)
        
        
        subhaloIDs = [result['id'] for result in results]
        mass_log_msun = [result['mass_log_msun'] for result in results]
        sfr = [result['sfr'] for result in results]
        
        idx_high_mass_subhalos = np.argsort(mass_log_msun)[::-1]
        
        subhaloIDs = [subhaloIDs[idx] for idx in idx_high_mass_subhalos]
        subhaloIDs = [0, 1, 70415]
        
        for subhaloID in subhaloIDs:
            
            save_path = f'data/snapshots-{snap_id}/subhalo-{subhaloID}'
            os.makedirs(save_path, exist_ok=True)

            if os.path.exists(os.path.join(save_path, f'subhalo_{subhaloID}_particles.h5')):
                print(f'Subhalo {subhaloID} already downloaded')
                continue
            
            data = get_subhalo_info(subhaloID, results, headers)
            
            with open(os.path.join(save_path, f'subhalo_{subhaloID}.json'), 'w') as f:
                json.dump(data, f, indent=4)
            
            try:
                get_particles(subhaloID, snap_id, headers, simulation, save_path)
            except Exception as e:
                print('Error getting particles: ', e)
                with open('failed_subhalos.txt', 'a') as f:
                    f.write(f'{snap_id},{subhaloID}\n')
    