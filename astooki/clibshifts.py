#!/usr/bin/env python2.5
# encoding: utf-8
"""
This is astooki.clibshifts, a Python wrapper for the shift library written in
C. 

This module is aimed at calcuting image shifts between different subfields in 
different subimages, which comes down to processing WFWFS data. Different
comparison algorithms (see COMPARE_*) and subpixel interpolation algorithms 
(see EXTREMUM_*) can be chosen. Also, data can be masked before measuring the 
image shifts and multiple subapertures can be used as a reference. For more 
information, consult the elaborate doxygen documentation.
"""

## @file clibshifts.py
# @author Tim van Werkhoven (tim@astro.su.se)
# @date 20090507
# 
# Created by Tim van Werkhoven on 2009-05-07.
# Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)
# 
# This file is licensed under the Creative Commons Attribution-Share Alike
# license versions 3.0 or higher, see
# http://creativecommons.org/licenses/by-sa/3.0/

## @package astooki.clibshifts
# @brief Python wrapper for libshifts-c.c library
# @author Tim van Werkhoven (tim@astro.su.se)
# @date 20090507
#
# This package calculates image shifts and supsersedes the old Python module 
# astooki.libshifts.
# 
# @section Finding a reference image
# 
# One can define different sources as a reference subimage to compare the 
# other subimages with. This can be tuned with the 'refmode' parameter of 
# calcShifts():
# - Use subimage with best RMS as reference (REF_BESTRMS)
# - Use user-supplied comparison image of same geometry as reference
#   (REF_STATIC)
# 
# @section Comparing images
# 
# There are several algorithms and implementations to compare two
# two-dimensional images. Because it is at this point unclear which method
# performs best (speed and quality-wise) in which situations, a multitide of
# these methods have been implemented.
# 
# The following image comparison methods are available
# - Absolute difference squared
# - Squared difference
# - Direct cross correlation
# - (TODO: Absolute difference?, FFT?)
# 
# @section Finding the subpixel maximum
# 
# The above routines compare the images themselves, but to find the best 
# (sub-pixel) image shift some interpolation is needed. The following methods 
# are available:
# - 2d 9-point parabolic interpolation
# - double 1d 5-point parabolic interpolation
# 
# All these functions expect the maps to have a *maximum*. This is done to 
# provide easier and more general extremum finding when using various 
# comparison methods.

# Import C library
import _libshifts
import liblog as log
import numpy as N

# Same defines as libshift
# @TODO: fix this, read from C library *directly* (is this possible?)

# Defines for comparison algorithms
## @brief Direct cross correlation
COMPARE_XCORR = 0					
## @brief Square difference
COMPARE_SQDIFF = 1				
## @brief Absolute difference squared
COMPARE_ABSDIFFSQ = 2			
## @brief Fourier method
COMPARE_FFT = 3						
## @brief Direct cross correlation squared
COMPARE_XCORRSQ = 4				

# Defines for extremum finding algorithms
## @brief maximum value (no interpolation)
EXTREMUM_MAXVAL = 0		 		
## @brief 2d 9 point parabola interpolation
EXTREMUM_2D9PTSQ = 1			
## @brief 2d 5 point parabola interpolation
EXTREMUM_2D5PTSQ = 2			

# Defines for reference usage
## @brief Use subimages with best RMS as reference
# When using this method, 'refopt' should be an integer indicating how many
# references should be used.
REF_BESTRMS = 0         	
## @brief Use static reference subapertures.
# When using this method, pass a list to the 'refopt' parameter to specify
# which subaps should be used.
REF_STATIC = 1						

# Mask the subimage and reference image before correlating
## @brief Use a circular mask
MASK_CIRC = 1							

## @brief Wrapper for libshifts.calcShifts
#
# This is a Python wrapper for the C library. It checks the parameters and 
# gives some statistics.
# 
# @param img Frame to process
# @param saccdpos Lower-left subimage positions in pixel coordinates
# @param saccdsize Subimage size in pixels
# @param sfccdpos Subfield positions in pixels (relative to saccdpos)
# @param sfccdsize Subfield size in pixels
# @param method Image comparison method to use (see COMPARE_* in astooki.clibshifts)
# @param extremum Subpixel interpolation to use (see EXTREMUM_* in astooki.clibshifts)
# @param refmode Method to choose reference subimages (see REF_* in astooki.clibshifts)
# @param refopt Option for refmode, see REF_BESTRMS and REF_STATIC in astooki.clibshifts
# @param shrange Shiftrange to calculate.
# @param mask Mask to apply before comparing subimages
# @param [out] refaps Pass empty list here to get a list of subapertures used for reference
def calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, method=COMPARE_ABSDIFFSQ, extremum=EXTREMUM_2D9PTSQ, refmode=REF_BESTRMS, refopt=1, shrange=[3,3], mask=None, refaps=None):
	
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
	
	# Check & setup a mask if we need it ("circular" or "none")
	if (mask == MASK_CIRC):
		maskc = N.indices(sfccdsize) - ((sfccdsize-1)/2.).reshape(2,1,1)
		mask = (N.sum(maskc**2.0, 0) < (sfccdsize[0]/2.0)**2.0).astype(N.int32)
	elif (mask):
		log.prNot(log.ERR, "calcShifts(): Error, unknown mask!")
	else:
		mask = N.ones(sfccdsize, dtype=N.int32)
		
	# Call C library with the pre-processed parameters
	ret = _libshifts.calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, shrange, mask, method, extremum, refmode, refopt)
	
	# Return reference apertures used if requested
	if (refaps is not None):
		refaps.extend(ret['refapts'])
	
	# Clip shifts, use float32 shrange, otherwise shifts is upcasted to float64.
	clrn = shrange.astype(N.float32)
	ret['shifts'] = N.clip(ret['shifts'], -clrn, clrn)
	
	# Give stats on the shifts just calculated
	log.prNot(log.INFO, "calcShifts(): shift: (%g,%g) +- (%g,%g)." % \
	 	(tuple(ret['shifts'].reshape(-1,2).mean(0)) + \
	 	tuple(ret['shifts'].reshape(-1,2).std(0))) )
	log.prNot(log.INFO, "calcShifts(): clipped: %d/%d, %g%%, refaps used: %s" %\
	 	(N.sum(abs(ret['shifts']) >= shrange), ret['shifts'].size, \
	 	100*N.sum(abs(ret['shifts']) >= shrange)/ret['shifts'].size, \
		str(ret['refapts'])) )
	
	# Return shifts
	return ret['shifts']
