# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'physicsModels'
copyright = '2019, Connor Feltman'
author = 'Connor Feltman'

# The full version, including alpha/beta/rc tags
# release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    # 'sphinxarg.ext',
    # 'sphinx_copybutton',
    # 'sphinx.ext.autodoc',
    # 'sphinx.ext.autosummary',
    # 'sphinx.ext.coverage',
    # 'sphinx.ext.doctest',
    # 'sphinx.ext.graphviz',
    # 'sphinx.ext.ifconfig',
    # 'sphinx.ext.imgmath',
    # 'sphinx.ext.intersphinx',
    # 'sphinx.ext.napoleon',
    # 'sphinx.ext.todo',
 ]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['AlfvenWave_Resonance.rst',
                    'invertedV_fitting.rst',
                    'ionosphere.rst',
                    'magnetosphere_Ionosphere.rst']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'pydata_sphinx_theme'

# -- Theme configuration -----------------------------------------------------

# Sidebar configuration
html_sidebars = {
    "**": ["search-field.html", "sidebar-nav-bs.html"],
    'index': []
    }

# General theme options
html_theme_options = {
    # Logo
    'logo': {'text': project},
    # Upper bar icons
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
    # Icon links
    "icon_links": [
        # GitHub of the proyect
        {"name": "GitHub",
         "url": "https://github.com/ecastroth/sphinx-documentation-demo",
         "icon": "fa-brands fa-square-github",
         "type": "fontawesome",}
    ]
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_static_path = []

autodoc_mock_imports = [
         "matplotlib",
         "scipy",
       ]


master_doc = 'index'