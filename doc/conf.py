# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
# sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'TritonDSE'
copyright = '2022, Quarkslab'
author = 'Quarkslab'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'breathe',
    'sphinx.ext.intersphinx',
    'sphinx.ext.githubpages',
    "nbsphinx",
    "enum_tools.autoenum"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['figs']

autodoc_default_flags = ['members', 'inherited-members']

autoclass_content = "both"  # Comment class with both class docstring and __init__ docstring

autodoc_typehints = "signature"

autodoc_type_aliases = {
    'PathLike': 'tritondse.types.PathLike',
    'Addr': 'tritondse.types.Addr',
    'rAddr': 'tritondse.types.rAddr',
    'BitSize': 'tritondse.types.BitSize',
    'ByteSize': 'tritondse.types.ByteSize',
    'Input': 'tritondse.types.Input',
    'PathHash': 'tritondse.types.PathHash',
    'AddrCallback': 'tritondse.callbacks.AddrCallback',
    'InstrCallback': 'tritondse.callbacks.InstrCallback',
    'MemReadCallback': 'tritondse.callbacks.MemReadCallback',
    'MemWriteCallback': 'tritondse.callbacks.MemWriteCallback',
    'NewInputCallback': 'tritondse.callbacks.NewInputCallback',
    'RegReadCallback': 'tritondse.callbacks.RegReadCallback',
    'RegWriteCallback': 'tritondse.callbacks.RegWriteCallback',
    'RtnCallback': 'tritondse.callbacks.RtnCallback',
    'SymExCallback': 'tritondse.callbacks.SymExCallback',
    'ThreadCallback': 'tritondse.callbacks.ThreadCallback',
}

# For internationalization
locale_dirs = ['locale/']   # path is example but recommended.
gettext_compact = False     # optional.


# For intersphinx
intersphinx_mapping = {'python': ('https://docs.python.org/3', None),
                       'lief': ('https://lief.quarkslab.com/doc/latest/', None)}
