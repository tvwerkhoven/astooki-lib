#!/usr/bin/env python2.5
# encoding: utf-8
"""
This is the astooki module, supplying most of the functionality for astooki. 
This library is meant to be used through the command line tool astooki, 
although it can also be used directly.

The available modules include:

  - astooki.clibshifts for measuring image shifts
  - astooki.libsdimm for SDIMM+ analysis
  - astooki.libsh for Shack-Hartmann related functions
  - astooki.libtomo for tomographic seeing analysis
  - astooki.libfile for file I/O routines
  - astooki.liblog for logging data 
  - astooki.libplot for specialized plotting routines

The docstrings provide concice runtime information. For more elaborate 
documentation, consult the doxygen documentation.
"""

__all__ = ["astooki"]
__version__ = '0.1.0'

## @mainpage Astooki-lib
# 
# This is the astooki module, supplying most of the functionality for 
# astooki-py25. 
# This library is meant to be used through the command line tool astooki-py25, 
# although it can also be used directly. The latest version is available at
# http://github.com/tvwerkhoven/astooki-lib/
# 
# The available modules include:
# 
#   - astooki.clibshifts for measuring image shifts
#   - astooki.libsdimm for SDIMM+ analysis
#   - astooki.libsh for Shack-Hartmann related functions
#   - astooki.libtomo for tomographic seeing analysis
#   - astooki.libfile for file I/O routines
#   - astooki.liblog for logging data 
#   - astooki.libplot for specialized plotting routines
# 
# The docstrings provide concice runtime information. For more elaborate 
# documentation, consult this Doxygen documentation.

## @package astooki
# @brief Main package for the astronomical toolkit library
# @author Tim van Werkhoven (tim@astro.su.se)
#
# This is the astooki module, supplying most of the functionality for 
# astooki-py25. 
