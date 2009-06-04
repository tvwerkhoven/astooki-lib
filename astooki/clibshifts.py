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
import liblog as log
import numpy as N

## Same defines as libshift
# TODO: fix this, read from C library *directly* (is this even possible?)

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

# Mask the subimage and reference image before correlating
MASK_CIRC = 1							# Use a circular mask

## Wrapper for _libshifts.calcShifts
def calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, method=COMPARE_ABSDIFFSQ, extremum=EXTREMUM_2D9PTSQ, refmode=REF_BESTRMS, refopt=1, shrange=[3,3], mask=None, refaps=None, subfields=None, corrmaps=None):
	
	# Make sure datatypes are correct
	img = img.astype(N.float32)
	saccdpos = saccdpos.astype(N.int32)
	saccdsize = saccdsize.astype(N.int32)
	sfccdpos = sfccdpos.astype(N.int32)
	sfccdsize = sfccdsize.astype(N.int32)
	shrange = N.array(shrange, dtype=N.int32)
	
	# Refopt should *always* be a 1-d list. This trick ensures that
	refopt = N.array([refopt]).flatten()[0]
	
	# Sanity check for subfield windows
	if (N.min(sfccdpos, 0) - shrange < 0).any():
		log.prNot(log.NOTICE, "%d,%d - %d,%d < 0,0" % \
			(tuple(N.min(sfccdpos, 0)) + tuple(shrange)))
		log.prNot(log.ERR, "calcShifts(): Error, subfield position - shift range smaller than 0!")
	if (N.max(sfccdpos, 0) + sfccdsize + shrange > saccdsize).any():
		log.prNot(log.NOTICE, "%d,%d + %d,%d + %d,%d > %d,%d" % \
			(tuple(N.max(sfccdpos, 0)) + tuple(sfccdsize) + tuple(shrange) + \
			 	tuple(saccdsize)))
		log.prNot(log.ERR, "calcShifts(): Error, subfield position + subfield size + shift range bigger than subaperture!")
	
	# Check if we need a mask
	if (mask == MASK_CIRC):
		#log.prNot(log.NOTICE, "calcShifts(): Using a circular mask.")
		maskc = N.indices(sfccdsize) - ((sfccdsize-1)/2.).reshape(2,1,1)
		mask = (N.sum(maskc**2.0, 0) < (sfccdsize[0]/2.0)**2.0).astype(N.int32)
	elif (mask):
		log.prNot(log.ERR, "calcShifts(): Error, unknown mask!")
	else:
		#log.prNot(log.NOTICE, "calcShifts(): Not masking.")
		mask = N.ones(sfccdsize, dtype=N.int32)
		
	# Call C library with the pre-processed parameters
	ret = _libshifts.calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, shrange, mask, method, extremum, refmode, refopt)
	# Return reference apertures used if requested
	if (refaps is not None):
		refaps.extend(ret['refapts'])
	
	# Clip shifts. Use float32 shrange because otherwise shifts is
	# upcasted to float64...
	clrn = shrange.astype(N.float32)
	ret['shifts'] = N.clip(ret['shifts'], -clrn, clrn)
	
	# Give stats
	log.prNot(log.NOTICE, "calcShifts(): shift: (%g,%g) +- (%g,%g)." % \
	 	(tuple(ret['shifts'].reshape(-1,2).mean(0)) + \
	 	tuple(ret['shifts'].reshape(-1,2).std(0))) )
	log.prNot(log.NOTICE, "calcShifts(): refaps used: %s" % str(ret['refapts']))
	log.prNot(log.NOTICE, "calcShifts(): clipped: %d/%d, %g%%." % \
	 	(N.sum(abs(ret['shifts']) >= shrange), ret['shifts'].size, \
	 	100*N.sum(abs(ret['shifts']) >= shrange)/ret['shifts'].size) )
	
	# Return shifts
	return ret['shifts']
