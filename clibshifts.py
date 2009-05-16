#!/usr/bin/env python2.5
# encoding: utf-8
"""
@file clibshifts.py
@brief Python wrapper for libshifts-c.c library
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090507

Created by Tim van Werkhoven on 2009-05-07.
Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

## Import C library

import _libshifts
import numpy as N

## Same defines as libshift
# TODO: fix this, read from library *directly*

# Defines for comparison algor#ithms
COMPARE_XCORR = 0					# Direct cross correlation
COMPARE_SQDIFF = 1				# Square difference
COMPARE_ABSDIFFSQ = 2			# Absolute difference squared
COMPARE_FFT = 3						# Fourier method
COMPARE_XCORRSQ = 4				# Direct cross correlation

# Defines for extremum finding algorithms
EXTREMUM_MAXVAL = 0		 		# maximum value (no interpolation)
EXTREMUM_2D9PTSQ = 1			# 2d 9 point parabola interpolation
EXTREMUM_2D5PTSQ = 2			# 2d 5 point parabola interpolation

# Defines for reference usage
REF_BESTRMS = 0         	# Use subimages with best RMS as reference, 
								 					# 'refopt' should be an integer indicating how 
								 					# many references should be used.
REF_STATIC = 1						# Use static reference subapertures, pass a 
													# list to the 'refopt' parameter to specify
													# which subaps should be used.

## Wrapper for _libshifts.calcShifts
def calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, method=COMPARE_ABSDIFFSQ, extremum=EXTREMUM_2D9PTSQ, refmode=REF_BESTRMS, refopt=1, shrange=[3,3], refaps=None, subfields=None, corrmaps=None):
	
	# Make sure shrange is a numpy array
	shrange = N.array(shrange, dtype=N.int32)
	img = img.astype(N.float32)
	saccdpos = saccdpos.astype(N.int32)
	saccdsize = saccdsize.astype(N.int32)
	sfccdpos = sfccdpos.astype(N.int32)
	# TODO: fix this, ugly!
	if (refmode == REF_BESTRMS):
		refopt = N.array([refopt]).flatten()[0]
	else:
		refopt = N.array([refopt]).flatten()[0]
	# Call C library
	ret = _libshifts.calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, shrange, method, extremum, refmode, refopt)
	# Return reference apertures used if requested
	if (refaps is not None):
		refaps.extend(ret['refapts'])
	# Return shifts
	return ret['shifts']
