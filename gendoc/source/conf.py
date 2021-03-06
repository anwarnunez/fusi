# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sphinx_rtd_theme
import numpydoc

# -- Project information -----------------------------------------------------

project = 'fusi'
copyright = '2022, Anwar Nunez-Elizalde'
author = 'Anwar Nunez-Elizalde'

# The full version, including alpha/beta/rc tags
release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.mathjax',
              'sphinx.ext.coverage',
              'sphinx.ext.inheritance_diagram',
              'sphinx.ext.viewcode',
              'sphinx.ext.napoleon',
              'sphinx.ext.githubpages',
              'sphinx_gallery.gen_gallery',
              'sphinx_rtd_theme',
              'numpydoc',
]

autosummary_generate = True
numpydoc_class_members_toctree = True
numpydoc_show_class_members = True

# # Sphinx-gallery
sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs' : '../../examples',
    # path where to save gallery generated examples
    'gallery_dirs'  : 'auto_examples',
    # which files to execute? only those starting with "plot_"
    'filename_pattern' : '/demo_'
    }


# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#

# RANDOMIZE THEME
import numpy as np
if 1:
    html_theme = 'sphinx_rtd_theme'
    html_theme_options = {'canonical_url': '',
    #                       # 'analytics_id': 'UA-XXXXXXX-1',
    #                       'logo_only': False,
                          'display_version': True,
    #                       'prev_next_buttons_location': 'bottom',
    #                       'style_external_links': False,
    #                       'vcs_pageview_mode': '',
    #                       'style_nav_header_background': 'white',
    #                       # Toc options
                          'collapse_navigation': False,
    #                       'sticky_navigation': True,
                          'navigation_depth': 3,
    #                       'includehidden': True,
    #                       'titles_only': False,
                          }


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
