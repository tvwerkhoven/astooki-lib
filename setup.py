#!/usr/bin/env python2.5
# encoding: utf-8
"""
setup.py -- setup file for the astooki-lib module

Created by Tim van Werkhoven (t.i.m.vanwerkhoven@xs4all.nl) on 2009-05-20.
Copyright (c) 2009 Tim van Werkhoven. All rights reserved.
"""
GITREVISION="v20090626.0-15-g42e482d"
import sys

# Try importing to see if we have NumPy available (we need this)
try:
	import numpy
	from numpy.distutils.core import setup, Extension
	from numpy.distutils.misc_util import Configuration
except:
	print "Could not load NumPy (numpy.distutils.{core,misc_util}), required by this package. Aborting."
	sys.exit(1)

# Setup extension module for C stuff
extlibs = []
extlibs.append(\
										Extension('_libshifts',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '1')],
                    include_dirs = [numpy.get_include()],
										libraries = ["m", "pthread"],
                   	library_dirs = [''],
										extra_compile_args=["-O4", "-ffast-math", "-Wall"],
										extra_link_args=None,
										sources = ['src/libshifts-c.c',
															'src/libshifts-c.h'])\
															)

# Setup 
setup(name = 'astooki',
	version = "0.1.0-%s" % (GITREVISION),
	description = 'Process and analyze astronomical data.',
	author = 'Tim van Werkhoven',
	author_email = 't.i.m.vanwerkhoven@xs4all.nl',
	url = 'http://www.solarphysics.kva.se/~tim/',
	license = "GPL",
	packages = ['astooki'],
	# This is for the C module
	ext_package = 'astooki',
	ext_modules = extlibs)
