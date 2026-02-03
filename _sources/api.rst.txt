API Reference
=============

This page provides detailed information about the galaxyGenius-MSA API.

Core Classes
------------

Configuration
~~~~~~~~~~~~~

.. autoclass:: galaxyGeniusMSA.config.Configuration
   :members:
   :undoc-members:
   :show-inheritance:

   The Configuration class handles all configuration settings for galaxyGenius-MSA, including:
   
   - Main configuration settings
   - MSA-specific configurations
   - Configuration validation and management

PreProcess
~~~~~~~~~~

.. autoclass:: galaxyGeniusMSA.preprocess.PreProcess
   :members:
   :undoc-members:
   :show-inheritance:

   The PreProcess class handles the preparation of data for SKIRT simulation:
   
   - Reading and processing subhalo data
   - Preparing particle data
   - Creating SKIRT input files
   - Managing simulation parameters

DataGeneration
~~~~~~~~~~~~~~

.. autoclass:: galaxyGeniusMSA.generation.DataGeneration
   :members:
   :undoc-members:
   :show-inheritance:

   The DataGeneration class manages the SKIRT radiative transfer simulation process:
   
   - Running SKIRT simulations
   - Managing input/output files
   - Handling data cube generation

PostProcess
~~~~~~~~~~~

.. autoclass:: galaxyGeniusMSA.postprocess.PostProcess
   :members:
   :undoc-members:
   :show-inheritance:

   The PostProcess class handles the post-processing of simulation results:
   
   - Processing data cubes
   - Generating bandpass images
   - Creating SEDs
   - Visualizing results

Utility Functions
-----------------

.. automodule:: galaxyGeniusMSA.utils
   :members:
   :undoc-members:
   :show-inheritance:

   Utility functions for various operations including:
   
   - Data conversion and manipulation
   - File handling
   - Mathematical operations
   - Visualization helpers

Properties Module
-----------------

.. automodule:: galaxyGeniusMSA.properties
   :members:
   :undoc-members:
   :show-inheritance:

   Properties calculation functions for various galaxy properties. 