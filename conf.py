# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os

sys.path.insert(0, os.path.abspath('.'))

# Modules that should exclude private methods (except __init__)
FILTER_PRIVATE_METHODS_MODULES = {
    'galaxyGeniusMSA.config',
    'galaxyGeniusMSA.preprocess',
    'galaxyGeniusMSA.generation',
    'galaxyGeniusMSA.postprocess',
}

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'galaxyGenius-MSA'
copyright = '2024'
author = 'Xingchen Zhou'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx settings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# -- Options for autodoc ----------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

def autodoc_skip_member(app, what, name, obj, skip, options):
    """Skip private methods (except __init__) for specific modules"""
    # Only filter methods (not attributes, functions, etc.)
    if what == 'method':
        # Get the module name from the method/function object
        # For methods, obj.__module__ should contain the module where the class is defined
        module_name = getattr(obj, '__module__', None)
        
        # If module_name is not directly available, try to get it from the class
        if not module_name:
            # Try to get the class from the method
            if hasattr(obj, '__self__'):
                # Bound method
                cls = getattr(obj.__self__, '__class__', None)
                if cls:
                    module_name = getattr(cls, '__module__', None)
            elif hasattr(obj, '__qualname__'):
                # Try to extract module from qualname by importing
                try:
                    qualname = obj.__qualname__
                    parts = qualname.split('.')
                    if len(parts) >= 2:
                        # Try to get module from the object's globals
                        if hasattr(obj, '__globals__'):
                            module_name = obj.__globals__.get('__name__', None)
                except:
                    pass
        
        # Only apply filtering to specified modules
        if module_name and module_name in FILTER_PRIVATE_METHODS_MODULES:
            # Allow __init__ to be shown
            if name == '__init__':
                return False
            
            # Skip methods that start with _ or __
            if name.startswith('_'):
                return True
    
    # For other modules (utils, properties) or other object types, don't skip anything
    return None

def setup(app):
    """Register the skip member callback"""
    app.connect('autodoc-skip-member', autodoc_skip_member)

# Type hints settings
typehints_fully_qualified = False
typehints_document_rtype = True
typehints_use_rtype = False
typehints_defaults = 'comma'
typehints_use_signature = True
typehints_use_signature_return = True
typehints_use_union = True
typehints_use_type_aliases = True
typehints_use_numpy = True
typehints_use_astropy = True
