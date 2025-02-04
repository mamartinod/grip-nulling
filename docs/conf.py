# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys

sys.path.insert(0, os.path.abspath('../grip'))
sys.path.insert(1, os.path.abspath('.'))
sys.path.insert(2, os.path.abspath('../tutorials'))

project = 'grip'
copyright = '2024, mamartinod'
author = 'Marc-Antoine Martinod'
release = '1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx_rtd_theme', 'sphinx.ext.autodoc', 'sphinx.ext.mathjax']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

master_doc = 'index'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

html_theme_options = {
    'navigation_depth': 4,
}

pygments_style = 'sphinx'

autodoc_mock_imports = [
'astropy',
'corner',
'cupy',
'cupyx',
'emcee',
'functools',
'grip',
'h5py',
'itertools',
'matplotlib',
'numdifftools',
'numpy',
'sbi',
'scipy',
'timeit',
'torch',
'tqdm'
]

