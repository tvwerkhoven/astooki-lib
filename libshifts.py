#!/usr/bin/env /sw/bin/python2.5
# encoding: utf-8
"""
@file calcshifts.py
@brief Image shift measurement library
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090224

Library to measure image shifts tailored for wavefront sensor use, but general
enough to support other data. The calcShifts() routine is the master routine 
calling various subroutines. CPU intensive routines are written in C and 
inlined in Python using Weave. This approach ensures that the speeds attained 
are competetive with pure C libraries, although there is probably still a 
reasonable speed increase possible by using pure C and possibly even assembly 
with SIMD instructions (such as SSEx).

The following image comparison methods are available:
- Absolute difference squared
- Squared difference
- Direct cross correlation
- (TODO: FFT cross correlation)
- (TODO: Absolute difference)

One can define different sources as a reference image to use the above comparison methods on:
- Use subimage with best RMS as reference
- Use user-supplied comparison image of same geometry as reference

The above routines compare the images themselves, but to find the best 
(sub-pixel) image shift some interpolation is needed. The following methods 
are available:
- 2d 9-point parabola interpolation
- Maximum value (no interpolation)
- (TODO?: Double 1d 3-point parabola interpolation)
- (TODO: 2d 9-point spline interpolation)
- (TODO?: double 1d 3-point spline interpolation)

Created by Tim van Werkhoven on 2009-02-24.
Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/

$Id
"""

#=============================================================================
# Import necessary libraries here
#=============================================================================

import numpy as N
import scipy as S
import scipy.ndimage
import scipy.weave
import scipy.weave.converters

#=============================================================================
# Some static defines
#=============================================================================

# Defines for comparison algorithms
COMPARE_XCORR = 0				# Direct cross correlation
COMPARE_SQDIFF = 1				# Square difference
COMPARE_ABSDIFFSQ = 2			# Absolute difference squared

# Defines for extremum finding algorithms
EXTREMUM_2D9PTSQ = 0			# 2d 9 point parabola interpolation
#EXTREMUM_1D3PTSQ = 1			# double 1d 3 point parabola interpolation
EXTREMUM_MAXVAL = 1				# maximum value (no interpolation)

# Defines for reference usage
REF_BESTRMS = 0					# Use subimage with best RMS as reference
#REF_WHOLEFIELDREF = 1			# Supply a whole image to be used as reference
#REF_NOREF = 2					# Don't use a reference, cross-compare 
								# everything with eachother. This will greatly 
								# increase CPU load to about N! instead of N 
								# comparisons.

#=============================================================================
# Helper routines -- image comparison
#=============================================================================

def crossCorrWeave(img, ref, pos, range):
	"""
	Compare 'img' with 'ref' using the cross correlation method:
		diff = Sum( (img(x,y) * ref(pos_x+x+i,pos_y+y+j)) )
	with 'range' the range for i and j. 'pos' is the lower left position of 
	'img' within 'ref'
	"""
	# TODO: do we use a ref with more pixels than img? Do we crop img when 
	# moving around over ref?
	diffmap = N.empty(range*2+1)
	
	code = """
	#line 96 "libshifts.py" (debugging info for compilation)
	double tmpsum;
	// Loop ranges
	int sh0min = (int) -range(0)+pos(0);
	int sh0max = (int) range(0)+pos(0);
	int sh1min = (int) -range(1)+pos(1);
	int sh1max = (int) range(1)+pos(1);
	
	// Loop over all shifts to be tested
	for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
		for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
			// Loop over all pixels within the img and refimg, and compute the 
			// cross correlation between the two.
			tmpsum = 0;
			for (int i=0; i<Nimg[0]; i++) {
				for (int j=0; j<Nimg[1]; j++) {
					tmpsum += img(i,j) * ref(i+sh0,j+sh1);
				}
			}
			// Store the current correlation value in the map.
			diffmap((sh0-sh0min), (sh1-sh1min)) = tmpsum;
		}
	}
	return_val = 1;
	"""
	one = S.weave.inline(code, \
		['ref', 'img', 'pos', 'range', 'diffmap'], \
		type_converters=S.weave.converters.blitz)
	
	return diffmap


def sqDiffWeave(img, ref, pos, range):
	"""
	Compare 'img' with 'ref' using the square difference method:
		diff = Sum( (img(x,y) - ref(pos_x+x+i,pos_y+y+j))^2 )
	with 'range' the range for i and j. 'pos' is the lower left position of 
	'img' within 'ref'
	"""
	# TODO: do we use a ref with more pixels than img? Do we crop img when 
	# moving around over ref?
	diffmap = N.empty(range*2+1)
	
	code = """
	#line 135 "libshifts.py" (debugging info for compilation)
	double tmpsum, diff;
	
	// Loop ranges
	int sh0min = (int) -range(0)+pos(0);
	int sh0max = (int) range(0)+pos(0);
	int sh1min = (int) -range(1)+pos(1);
	int sh1max = (int) range(1)+pos(1);
	
	// Loop over all shifts to be tested
	for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
		for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
			// Loop over all pixels within the img and refimg, and compute the 
			// cross correlation between the two.
			tmpsum = 0;
			// N<array>[<index>] gives the <index>th size of <array>, i.e. 
			// Nimg[1] gives the second dimension of array 'img'
			for (int i=0; i<Nimg[0]; i++) {
				for (int j=0; j<Nimg[1]; j++) {
					// First get the difference...
					diff = img(i,j) - ref(i+sh0,j+sh1);
					// ...then square this
					tmpsum += diff*diff;
				}
			}
			// Store the current correlation value in the map. Use negative 
			// value to ensure that we get a maximum for best match in diffmap
			diffmap((sh0-sh0min), (sh1-sh1min)) = -tmpsum;
		}
	}
	return_val = 1;
	"""
	one = S.weave.inline(code, \
		['ref', 'img', 'pos', 'range', 'diffmap'], \
		type_converters=S.weave.converters.blitz)
	
	return diffmap


def absDiffSqWeave(img, ref, pos, range):
	"""
	Compare 'img' with 'ref' using the absolute difference squared method:
		diff = Sum( |img(x,y) - ref(x+i,y+j)| )^2
	with 'range' the range for i and j.
	"""
	# TODO: do we use a ref with more pixels than img? Do we crop img when 
	# moving around over ref?
	diffmap = N.empty(range*2+1)
	
	code = """
	#line 190 "libshifts.py" (debugging info for compilation)
	double tmpsum;
	// Loop ranges
	int sh0min = (int) -range(0)+pos(0);
	int sh0max = (int) range(0)+pos(0);
	int sh1min = (int) -range(1)+pos(1);
	int sh1max = (int) range(1)+pos(1);
	
	// Loop over all shifts to be tested
	for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
		for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
			// Loop over all pixels within the img and refimg, and compute the 
			// cross correlation between the two.
			tmpsum = 0.0;
			for (int i=0; i<Nimg[0]; i++) {
				for (int j=0; j<Nimg[1]; j++) {
					tmpsum += fabs(img(i,j) - ref(i+sh0,j+sh1));
				}
			}
			// Store the current correlation value in the map. Use negative 
			// value to ensure that we get a maximum for best match in diffmap
			diffmap((sh0-sh0min), (sh1-sh1min)) = -(tmpsum*tmpsum);
		}
	}
	return_val = 1;
	"""
	one = S.weave.inline(code, \
		['ref', 'img', 'pos', 'range', 'diffmap'], \
		type_converters=S.weave.converters.blitz)
	
	return diffmap


#=============================================================================
# Helper routines -- extremum finding
#=============================================================================

def quadInt2dPython(data):
	"""
	Find the extrema of 'data' using a two-dimensional 9-point quadratic
	interpolation. Use 'start' as initial guess.
	"""
	# Initial guess for the interpolation
	start = N.argwhere(data == data.max())[0]
	# Crop from the full map to interpolate for
	submap = data[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
	
	a2 = 0.5 * (submap[2,1] - submap[0,1])
	a3 = 0.5 * submap[2,1] - submap[1,1] + 0.5 * submap[0,1]
	a4 = 0.5 * (submap[1,2] - submap[1,0])
	a5 = 0.5 * submap[1,2] - submap[1,1] + 0.5 * submap[1,0]
	a6 = 0.25 * (submap[2,2] - submap[2,0] - \
		submap[0,2] + submap[0,0])
	
	return N.array([(2*a2*a5-a4*a6)/(a6*a6-4*a3*a5), \
	 	(2*a3*a4-a2*a6)/(a6*a6-4*a3*a5)]) + start


def quadInt2dWeave(data):
	"""
	Find the extrema of 'data' using a two-dimensional 9-point quadratic
	interpolation. Use 'start' as initial guess.
	"""
	
	# Empty array which will hold the subpixel maximum
	extremum = N.empty(2)
	# Initial guess for the interpolation
	start = N.argwhere(data == data.max())[0]
	# Crop from the full map to interpolate for
	submap = data[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
	code = """
	#line 246 "libshifts.py" (debugging info)
	double a2, a3, a4, a5, a6;
	
	a2 = 0.5 * (submap(2,1) -submap(0,1));
	a3 = 0.5 * submap(2,1) - submap(1,1) + 0.5 * submap(0,1);
	a4 = 0.5 * (submap(1,2) - submap(1,0));
	a5 = 0.5 * submap(1,2) - submap(1,1) + 0.5 * submap(1,0);
	a6 = 0.25 * (submap(2,2) - submap(2,0) - submap(0, 2) + submap(0,0));
	
	extremum(0) = (2*a2*a5-a4*a6)/(a6*a6-4*a3*a5);
	extremum(1) = (2*a3*a4-a2*a6)/(a6*a6-4*a3*a5);
	
	return_val = 1;
	"""
	one = S.weave.inline(code, \
		['submap', 'extremum'], \
		type_converters=S.weave.converters.blitz)
	
	return extremum+start

def maxValPython(data):
	"""
	Return the coordinates of the maximum value in 'data'.
	"""
	return N.argwhere(data == data.max())[0]


#=============================================================================
# Big control routine
#=============================================================================


def calcShifts(img, sapos, sasize, sfpos, sfsize, method=COMPARE_SQDIFF, extremum=EXTREMUM_2D9PTSQ, refmode=REF_BESTRMS, shrange=[3,3], verb=0):
	"""
	Calculate the image shifts for subapertures/subfields in 'img' located at 
	'sapos' and 'sfpos' respectively,
	each with the same 'subsize' pixelsize. Method defines the method to
	compare the subimages (i.e. image cross correlation, fourier cross 
	correlation, squared difference, absolute difference squared, etc), 
	'extremum' defines the method to find the best subpixel shift, i.e. what 
	interpolation should be used (double 1d fitting, 2d fitting, simple 
	maximum finding, etc). 'range' defines the possible shifts to test (actual 
	number of distances checked is 2*range+1)
	
	For regular SH WFS, set sapos to the subaperture positions, sfpos to 
	[[0,0]], and subsize to the subimage pixelsize. For wide-field SH WFS, set 
	sapos similarly, but set sfpos to an array of pixelpositions relative to 
	sapos for the subfields to compare. Set subsize not to the complete 
	subimage size, but to the size of the subfield you want to use.
	"""
	# TODO: generalize maps to have a maximum?
	# TODO: add reference finding routine
	# TODO: remove average shift or not?
	
	# Parse the 'method' argument
	if (method == COMPARE_XCORR):
		if (verb>0): print "calcShifts(): Using direct cross correlation"
		mfunc = crossCorrWeave
	elif (method == COMPARE_SQDIFF):
		if (verb>0): print "calcShifts(): Using square difference"
		mfunc = sqDiffWeave
	elif (method == COMPARE_ABSDIFFSQ):
		if (verb>0): print "calcShifts(): Using absolute difference squared"
		mfunc = absDiffSqWeave
	elif (is_function(method)):
		if (verb>0): print "calcShifts(): Using user supplied image comparison function"
		mfunc = method
	else:
		raise RuntimeError("'method' must be either one of the predefined image comparison methods, or a function doing that.")
	
	# Parse the 'extremum' argument
	if (extremum == EXTREMUM_2D9PTSQ):
		if (verb>0): print "calcShifts(): Using 2d parabola interpolation"
		extfunc = quadInt2dWeave
	elif (extremum == EXTREMUM_MAXVAL):
		if (verb>0): print "calcShifts(): Using maximum value"
		extfunc = maxValPython
	elif (is_function(extremum)):
		if (verb>0): print "calcShifts(): Using user supplied maximum-finding function"
		extfunc = extremum
	else:
		raise RuntimeError("'extremum' must be either one of the predefined extremum finding methods, or a function doing that.")
	
	# Find reference 
	if (refmode == REF_BESTRMS):
		if (verb>0): print "calcShifts(): Using best RMS reference criteria"
		# Find the subimage with the best RMS
		bestrms = 0.0
		for _sapos in sapos:
			# Get subimage
			_subimg = img[_sapos[1]:_sapos[1] + sasize[1], \
				_sapos[0]:_sapos[0] + sasize[0]]
			# Calc RMS^2
			rms = ((_subimg-N.mean(_subimg))**2.0).sum()
			# Comparse against previous best
			if (rms > bestrms):
				# If better, store this rms and reference subimage
				if (verb>0): 
					print "Found new best rms @ (%d,%d), rms^2 is: %.3g, was: %.3g." % (_sapos[0], _sapos[1], rms, bestrms)
				bestrms = rms
				bestpos = _sapos
				ref = _subimg
		# Take root to get real RMS
		bestrms = bestrms**0.5
		if (verb>1):
			print "Best RMS subimage @ (%d,%d) (rms: %.3g)" % (bestpos[0], bestpos[1], bestrms)
	else:
		raise RuntimeError("'refmode' must be one of the predefined reference modes.")
	
	# Init shift vectors
	disps = N.empty((sapos.shape[0], sfpos.shape[0], 2))
	# Loop over the subapertures
	for _sapos in sapos:
		if (verb>1):
			print "Processing subimage @ (%d,%d)" % (_sapos[0], _sapos[1])
		# Loop over the subfields
		for _sfpos in sfpos:
			# Current pixel position
			_pos = _sapos + _sfpos
			_end = _pos + sfsize
			# Get the current subfield (remember, the pixel at (x,y) is
			# img[y,x])
			_subimg = img[_pos[1]:_end[1], _pos[0]:_end[0]]
			# Compare the image with the reference image
			diffmap = mfunc(_subimg, ref, _sfpos, shrange)
			# Find the extremum
			shift = extfunc(diffmap)
		if (verb>1):
			print "Last found shift: (%.3g,%.3g)" % (shift[0], shift[1])
	
	#end	

if __name__ == '__main__':
	pass

