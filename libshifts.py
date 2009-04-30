#!/usr/bin/env /sw/bin/python2.5
# encoding: utf-8
"""
@file libshifts.py
@brief Image shift measurement library for Shack-Hartmann wavefront sensors
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090224

Library to measure image shifts tailored for wavefront sensor use, but general
enough to support other data. The calcShifts() routine is the master routine 
calling various subroutines. CPU intensive routines are written in C and 
inlined in Python using Weave. This approach ensures that the speeds attained 
are competetive with pure C libraries, although there is probably still a 
reasonable speed increase possible by using pure C and possibly even assembly 
with SIMD instructions (such as SSEx).

* Comparing images

There are several algorithms and implementations to compare two
two-dimensional images. Because it is at this point unclear which method
performs best (speed and quality-wise) in which situations, a multitide of
these methods have been implemented.

The naming of these routines follows the scheme of <method><implementation>.
<method> is listed between brackets in the list below, and <implementation> is
one of Weave or Python, corresponding to a C implementation inlined with Weave
or a pure Python function.

These functions (mainly the Weave ones) have a lot of double code, but this is 
done on purpose to provide faster execution (although a proper profiling 
analysis of this code has not been done yet).

The following image comparison methods are available (with function prefixes 
listed between brackets):
- Absolute difference squared (absDiffSq)
- Squared difference (sqDiff)
- Direct cross correlation (crossCorr)
- FFT cross correlation (fft)
- (TODO: Absolute difference)

* Finding a reference image

One can define different sources as a reference image to use the above 
comparison methods on (with the 'method' parameter to pass to calcShift() 
between brackets):
- Use subimage with best RMS as reference (REF_BESTRMS)
- Use user-supplied comparison image of same geometry as reference
  (REF_STATIC)

* Finding the subpixel maximum

The above routines compare the images themselves, but to find the best 
(sub-pixel) image shift some interpolation is needed. The following methods 
are available (function name prefixes listed between brackets):
- 2d 9-point parabola interpolation (quadInt2d)
- Maximum value (no interpolation) (maxVal)
- (TODO: 2d 9-point spline interpolation)
- (TODO?: Double 1d 3-point parabola interpolation)
- (TODO?: double 1d 3-point spline interpolation)

All these functions expect the maps to have a maximum. This is done to provide 
easier and more general extremum finding when using various comparison 
methods.

* Naming/programming scheme

The naming convention in this library for subaperture/subfield position and 
sizes is done as follows:
- Full variable name is: [sa|sf][ccd|ll][pos|size]
- [sa|sf] defines whether it is a subaperture or subfield
- [ccd|ll] defines whether the units are in pixels on the CCD or SI units on 
           the lenslet
- [pos|size] defines whether the quantity is the position or size

Other conventions in this library:
- Coordinates, lengths, sizes are stored as (x,y), slicing arrays/matrices in 
  NumPy must therefore always be done with data[coord[1], coord[0]], because 
  of the way data is ordered in NumPy arrays. Saving images to FITS files 
  works correctly (origin is in the lowerleft corner), but saving files as PNG 
  does not work out of the box (origin is in the top left corner), so the data 
  must be shifted and mirrored first.

Created by Tim van Werkhoven on 2009-02-24.
Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

#=============================================================================
# Import necessary libraries here
#=============================================================================

import numpy as N
import scipy as S
import scipy.ndimage			# For fast ndimage processing
import scipy.weave				# For inlining C
#import scipy.weave.converters	# For inlining C
import scipy.fftpack			# For FFT functions in FFT cross-corr
import scipy.signal				# For hanning/hamming windows
import pyfits				
import timlib					# Some miscellaneous functions
from liblog import *			# Logging / printing functions

import unittest
import time						# For unittest

#=============================================================================
# Some static defines
#=============================================================================

# Defines for comparison algorithms
COMPARE_XCORR = 0				# Direct cross correlation
COMPARE_SQDIFF = 1				# Square difference
COMPARE_ABSDIFFSQ = 2			# Absolute difference squared
COMPARE_FFT = 3					# Fourier method

# Defines for extremum finding algorithms
EXTREMUM_2D9PTSQ = 0			# 2d 9 point parabola interpolation
#EXTREMUM_1D3PTSQ = 1			# double 1d 3 point parabola interpolation
EXTREMUM_MAXVAL = 1				# maximum value (no interpolation)

# Defines for reference usage
REF_BESTRMS = 0					# Use subimages with best RMS as reference, 
								# 'refopt' should be an integer indicating how 
								# many references should be used.
#REF_NOREF = 2					# Don't use a reference, cross-compare 
								# everything with eachother. This will greatly 
								# increase CPU load to about N*N instead of N 
								# comparisons.
REF_STATIC = 4					# Use static reference subapertures, pass a 
								# list to the 'refopt' parameter to specify
								# which subaps should be used.

# Fourier windowing method constants
FFTWINDOW_NONE = 0				# No window
FFTWINDOW_HANN = 1				# Hann window
FFTWINDOW_HAMM = 2				# Hamming window
FFTWINDOW_HANN50 = 3			# Hann window with 50% flat surface
FFTWINDOW_HAMM50 = 4			# Hamming window with 50% flat surface

# Compilation flags
__COMPILE_OPTS = "-O3 -ffast-math -msse -msse2"

#=============================================================================
# Helper routines -- image comparison
#=============================================================================

def crossCorrWeave(img, ref, pos, range):
	"""
	Compare 'img' with 'ref' using the cross correlation method:
		diff = Sum( (img(x,y) * ref(pos_x+x+i,pos_y+y+j)) )
	with 'range' the range for i and j. 'pos' is the lower left position of 
	'img' within 'ref'. 
	
	If 'pos' is non-zero, this routine expects that 'img' fits within 'ref' 
	while moving it around over 'range' pixels. If 'pos' is zero, this routine 
	expects 'ref' and 'img' to be of the same resolution, and will only use 
	the intersection of 'img' and 'ref' when comparing it over a certain 
	shift, using normalization to make the results comparable.
	"""
	# Init the map to store the quality of each measured shift in
	diffmap = N.empty((range[1]*2+1, range[0]*2+1))
	
	# Pre-process data, mean should be the same
	img = img/N.float32(img.mean())
	ref = ref/N.float32(ref.mean())
	# print img.shape
	# print ref.shape
	
	#raise RuntimeWarning("crossCorrWeave() is not really working at the moment, probably some gradient or bias issue.")
	
	code = """
	#line 175 "libshifts.py" (debugging info for compilation)
	// We need minmax functions
	#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
	#endif
	
	#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
	#endif
	
	double tmpsum;
	// Loop ranges
	int sh0min = (int) -range(0)+pos(0);
	int sh0max = (int) range(0)+pos(0);
	int sh1min = (int) -range(1)+pos(1);
	int sh1max = (int) range(1)+pos(1);
	//printf("%d -- %d and %d -- %d\\n", sh0min, sh0max, sh1min, sh1max);
	
	// If sum(pos) is not zero, we expect that ref is large enough to naively 
	// shift img around.
	if (pos(0)+pos(1) != 0) {
		// Loop over all shifts to be tested
		for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
			for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
				// Loop over all pixels within the img and refimg, and compute
				// the cross correlation between the two.
				tmpsum = 0.0;
				for (int i=0; i<Nimg[0]; i++) {
					for (int j=0; j<Nimg[1]; j++) {
						tmpsum += img(i,j) * ref(i+sh1,j+sh0);	
					}
				}
				// Store the current correlation value in the map.
				//printf("diff @ %d,%d = %f\\n", (sh0-sh0min), (sh1-sh1min), tmpsum);
				diffmap((sh1-sh1min), (sh0-sh0min)) = tmpsum;
			}
		}
	}
	// If sum(pos) is zero, we use clipping of ref and img to only use the 
	// intersection of the two datasets for comparison, using normalisation to 
	// make the results consistent.
	else {
		for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
			for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
				tmpsum = 0.0;
				// If the shift to compare is negative, we must make sure 
				// i+sh0 in 'ref' will not be negative, so we start i at -sh0.
				// If the shift is positive, we must make sure that i+sh0 in 
				// 'ref' will not go out of bound, so we stop i at Nimg[0] - 
				// sh0.
				// N<array>[<index>] gives the <index>th size of <array>, i.e. 
				// Nimg[1] gives the second dimension of array 'img'
				for (int i=0 - min(sh1, 0); i<Nimg[0] - max(sh1, 0); i++) {
					for (int j=0 - min(sh0, 0); j<Nimg[1] - max(sh0, 0); j++){
						tmpsum += img(i,j) * ref(i+sh1,j+sh0);
					}
				}
				
				// Scale the value found by dividing it by the number of 
				// pixels we compared.
				//printf("diff @ %d,%d = %f\\n", (sh0-sh0min), (sh1-sh1min), tmpsum / ((Nimg[0]-abs(sh0)) * (Nimg[1]-abs(sh1))));
				diffmap((sh1-sh1min), (sh0-sh0min)) = tmpsum / 
					((Nimg[0]-abs(sh0)) * (Nimg[1]-abs(sh1)));
			}
		}
		//printf("%lf, %d and %d \\n", 
		//	tmpsum, (Nimg[0]), (Nimg[1]));
	}
	return_val = 1;
	"""
	one = S.weave.inline(code, \
		['ref', 'img', 'pos', 'range', 'diffmap'], \
		extra_compile_args= [__COMPILE_OPTS], \
		type_converters=S.weave.converters.blitz)
	
	return diffmap


def sqDiffWeave(img, ref, pos, range):
	"""
	Compare 'img' with 'ref' using the square difference method:
		diff = Sum( (img(x,y) - ref(pos_x+x+i,pos_y+y+j))^2 )
	with 'range' the range for i and j. 'pos' is the lower left position of 
	'img' within 'ref'
	
	If 'pos' is non-zero, this routine expects that 'img' fits within 'ref' 
	while moving it around over 'range' pixels. If 'pos' is zero, this routine 
	expects 'ref' and 'img' to be of the same resolution, and will only use 
	the intersection of 'img' and 'ref' when comparing it over a certain 
	shift, using normalization to make the results comparable.
	"""
	# Init the map to store the quality of each measured shift in
	diffmap = N.empty((range[1]*2+1, range[0]*2+1))
	
	# Pre-process data, mean should be the same
	img = img/N.float32(img.mean())
	ref = ref/N.float32(ref.mean())
	# print img.shape
	# print ref.shape
	
	code = """
	#line 276 "libshifts.py" (debugging info for compilation)
	// We need minmax functions
	#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
	#endif
	
	#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
	#endif
	
	double tmpsum, diff;
	// Loop ranges
	int sh0min = (int) -range(0)+pos(0);
	int sh0max = (int) range(0)+pos(0);
	int sh1min = (int) -range(1)+pos(1);
	int sh1max = (int) range(1)+pos(1);
	
	// If sum(pos) is not zero, we expect that ref is large enough to naively 
	// shift img around.
	if (pos(0)+pos(1) != 0) {
		// Loop over all shifts to be tested
		for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
			for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
				// Loop over all pixels within the img and refimg, and compute
				// the cross correlation between the two.
				tmpsum = 0.0;
				for (int i=0; i<Nimg[0]; i++) {
					for (int j=0; j<Nimg[1]; j++) {
						// First get the difference...
						diff = img(i,j) - ref(i+sh1,j+sh0);
						// ...then square this
						tmpsum += diff*diff;
					}
				}
				// Store the current correlation value in the map. Use
				// negative value to ensure that we get a maximum for best
				// match in diffmap (this allows to use a more general 
				// maximum-finding method, instead of splitting between maxima
				// and minima)
				printf("tmp: %.3f ", -tmpsum);
				diffmap((sh1-sh1min), (sh0-sh0min)) = -tmpsum;
			}
		}
	}
	// If sum(pos) is zero, we use clipping of ref and img to only use the 
	// intersection of the two datasets for comparison, using normalisation to 
	// make the results consistent.
	else {
		for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
			for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
				tmpsum = 0.0;
				// If the shift to compare is negative, we must make sure 
				// i+sh0 in 'ref' will not be negative, so we start i at -sh0.
				// If the shift is positive, we must make sure that i+sh0 in 
				// 'ref' will not go out of bound, so we stop i at Nimg[0] - 
				// sh0.
				// N<array>[<index>] gives the <index>th size of <array>, i.e. 
				// Nimg[1] gives the second dimension of array 'img'
				for (int i=0 - min(sh1, 0); i<Nimg[0] - max(sh1, 0); i++) {
					for (int j=0 - min(sh0, 0); j<Nimg[1] - max(sh0, 0); j++){
						// First get the difference...
						diff = img(i,j) - ref(i+sh1,j+sh0);
						// ...then square this
						tmpsum += diff*diff;
					}
				}
				// Scale the value found by dividing it by the number of 
				// pixels we compared.
				
				diffmap((sh1-sh1min), (sh0-sh0min)) = -tmpsum / 
					((Nimg[0]-abs(sh0)) * (Nimg[1]-abs(sh1)));
			}
		}
	}
	return_val = 1;
	"""
	one = S.weave.inline(code, \
		['ref', 'img', 'pos', 'range', 'diffmap'], \
		extra_compile_args= [__COMPILE_OPTS], \
		type_converters=S.weave.converters.blitz)
	
	return diffmap


def absDiffSqWeave(img, ref, pos, range):
	"""
	Compare 'img' with 'ref' using the absolute difference squared method:
		diff = Sum( |img(x,y) - ref(pos_x+x+i, pos_y+y+j)| )^2
	with 'range' the range for i and j. 'pos' is the lower left position of 
	'img' within 'ref'
	
	If 'pos' is non-zero, this routine expects that 'img' fits within 'ref' 
	while moving it around over 'range' pixels. If 'pos' is zero, this routine 
	expects 'ref' and 'img' to be of the same resolution, and will only use 
	the intersection of 'img' and 'ref' when comparing it over a certain 
	shift, using normalization to make the results comparable.
	"""
	# Init the map to store the quality of each measured shift in
	diffmap = N.empty((range[1]*2+1, range[0]*2+1), dtype=N.float32)
	
	# Pre-process data, mean should be the same
	# img = img/N.float32(img.mean())
	# ref = ref/N.float32(ref.mean())
	# print img.shape
	# print ref.shape
	
	code = """
	#line 383 "libshifts.py" (debugging info for compilation)
	// We need minmax functions
	#ifndef max
	#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
	#endif
	
	#ifndef min
	#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
	#endif
	
	double tmpsum;
	// Loop ranges
	int sh0min = (int) -range(0)+pos(0);
	int sh0max = (int) range(0)+pos(0);
	int sh1min = (int) -range(1)+pos(1);
	int sh1max = (int) range(1)+pos(1);
	// If sum(pos) is not zero, we expect that ref is large enough to naively 
	// shift img around.
	if (pos(0)+pos(1) != 0) {
		// Loop over all shifts to be tested
		for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
			for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
				// Loop over all pixels within the img and refimg, and compute
				// the cross correlation between the two.
				tmpsum = 0.0;
				for (int i=0; i<Nimg[0]; i++) {
					for (int j=0; j<Nimg[1]; j++) {
						// TvW @ 20090429: big bugfix! indices wrong! was:
						//tmpsum += fabs(img(i,j) - ref(i+sh0,j+sh1));						
						tmpsum += fabsf(img(i,j) - ref(i+sh1,j+sh0));
					}
				}
				// Store the current correlation value in the map. Use
				// negative value to ensure that we get a maximum for best
				// match in diffmap
				// TvW @ 20090429: swap this too
				diffmap((sh1-sh1min), (sh0-sh0min)) = -(tmpsum*tmpsum);
			}
		}
	}
	// If sum(pos) is zero, we use clipping of ref and img to only use the 
	// intersection of the two datasets for comparison, using normalisation to 
	// make the results consistent.
	else {
		for (int sh0=sh0min; sh0 <= sh0max; sh0++) {
			for (int sh1=sh1min; sh1 <= sh1max; sh1++) {
				tmpsum = 0.0;
				// If the shift to compare is negative, we must make sure 
				// i+sh0 in 'ref' will not be negative, so we start i at -sh0.
				// If the shift is positive, we must make sure that i+sh0 in 
				// 'ref' will not go out of bound, so we stop i at Nimg[0] - 
				// sh0.
				// N<array>[<index>] gives the <index>th size of <array>, i.e. 
				// Nimg[1] gives the second dimension of array 'img'
				// TODO: CHECK THIS! are sh0, sh1 in the right index? are i and j 
				// right?
				for (int i=0 - min(sh1, 0); i<Nimg[0] - max(sh1, 0); i++) {
					for (int j=0 - min(sh0, 0); j<Nimg[1] - max(sh0, 0); j++){
						tmpsum += fabs(img(i,j) - ref(i+sh1,j+sh0));
					}
				}
				// Scale the value found by dividing it by the number of 
				// pixels we compared.
				diffmap((sh1-sh1min), (sh0-sh0min)) = -(tmpsum*tmpsum) /
					 ((Nimg[0]-abs(sh0)) * (Nimg[1]-abs(sh1)));
			}
		}
	}
	//
	return_val = 1;
	"""
	
	one = S.weave.inline(code, \
		['ref', 'img', 'pos', 'range', 'diffmap'], \
#		compiler='gcc ', \
		extra_compile_args = ["-O3"], \
#		extra_link_args = ["--invalid"], \
		auto_downcast=0, \
		type_converters=S.weave.converters.blitz)
	
	return diffmap


def fftPython(img, ref, pos, range, window=FFTWINDOW_HANN50):
	"""
	Compare 'img' with 'ref' using the Fourier transform method, defined as
	map[ii,jj] = FFT-1[ FFT[w * (img[i,j] - mean(img))] * 
		FFT[w * (img[i+ii, j+jj] - mean(img))]* ]
	with FFT and FFT-1 the forward and backward fourier transforms. 'pos' is
	the lower left position of 'img' within 'ref', 'window' is the apodization 
	method used.
	
	TODO: How to apply Fourier in case that img.shape != ref.shape?
	"""
	res = N.array(img.shape)
	wind = 1			# Default 'window' if none specified
	
	# Pre-process data, mean should be the same
	img = img/N.float32(img.mean())
	ref = ref/N.float32(ref.mean())
	
	# Make the window
	if (window == FFTWINDOW_HANN):
		wind = S.signal.hann(res[0]).reshape(-1,1) * \
			S.signal.hann(res[1])
	elif (window == FFTWINDOW_HAMM):
		wind = S.signal.hamming(res[0]).reshape(-1,1) *	\
		 	S.signal.hamming(res[1])
	elif (window == FFTWINDOW_HANN50):
		# 50% flat surface at the top, meaning sqrt(0.5) flat in each 1D 
		# direction, which is about 0.707. That leaves 1-sqrt(0.5) to get the 
		# apodization at the edges.
		
		# Apodisation takes this many pixels *per edge*
		apod = N.round((res * (1-N.sqrt(0.5)))/2)
		# The number of pixels left for the top is then given by:
		top = res - apod*2
		# Generate the window in the x-direction
		winx = S.signal.hann(apod[0]*2+1)
		winx = N.concatenate( \
			(winx[:apod[0]], \
			N.ones(top[0]), \
			winx[apod[0]+1:])).reshape(-1, 1)
		# Window in y direction
		winy = S.signal.hann(apod[1]*2+1)
		winy = N.concatenate( \
			(winy[:apod[1]], \
			N.ones(top[1]), \
			winy[apod[1]+1:]))
		# Total window
		wind = winx * winy
	elif (window == FFTWINDOW_HAMM50):
		# Similar to FFTWINDOW_HANN50, except with Hamming window
		
		# Apodisation takes this many pixels *per edge*
		apod = N.round((res * (1-N.sqrt(0.5)))/2)
		# The number of pixels left for the top is then given by:
		top = res - apod*2
		# Generate the window in the x-direction
		winx = S.signal.hamming(apod[0]*2+1)
		winx = N.concatenate( \
			(winx[:apod[0]], \
			N.ones(top[0]), \
			winx[apod[0]+1:])).reshape(-1, 1)
		# Window in y direction
		winy = S.signal.hamming(apod[1]*2+1)
		winy = N.concatenate( \
			(winy[:apod[1]], \
			N.ones(top[1]), \
			winy[apod[1]+1:]))
		# Total window
		wind = winx * winy
	else:
		raise ValueError("'window' must be one of the valid FFT windowing functions.")
	
	# Window the data
	imgw = (img - N.mean(img)) * wind
	refw = (ref - N.mean(ref)) * wind
	
	# Get cross-correlation map
	fftmap = S.fftpack.ifft2(S.fftpack.fft2(refw) * \
	 	(S.fftpack.fft2(imgw)).conjugate())
	
	# DEBUG (this should always be false):
	if (abs(fftmap.imag).sum() > 0.0001 * abs(fftmap.real).sum()):
		raise ValueError("FFT returned too much imaginary data (%.3g%, %.3g vs %.3g)" % \
		 	(100 * abs(fftmap.imag).sum() / abs(fftmap.real).sum(), \
		 	fftmap.real.sum(), fftmap.imag.sum()))
	
	# If range is not the full image, we crop the result with the zero shift 
	# at the center, otherwise we merely shift the zero shift to the center.
	raise RuntimeWarning("Warning, FFT method possibly incorrect.")
	# Problem: what does pixel diffmap[i,j] mean? is it the cross-correlation of 
	# shift-vector (i,j) or (j,i)? How do we use range[] with this?
	if (range < res/2).all():
		diffmap = (timlib.shift(fftmap.real, range))\
			[:range[0]*2+1,:range[1]*2+1]
	else:
		raise NotImplemented("range > res/2 is not implemented.")
	
	return diffmap


#=============================================================================
# Helper routines -- extremum finding
#=============================================================================

def quadInt2dPython(data, range, limit=None):
	"""
	Find the extrema of 'data' using a two-dimensional 9-point quadratic
	interpolation (QI formulae by Yi & Molowny Horas (1992, Eq. (10)).). 
	The subpixel maximum will be examined around the coordinate of the pixel 
	with the maximum intensity. 
	
	'range' must be set to the shift-range that the correlation map in 'data' 
	corresponds to to find the pixel corresponding to a shift of 0. Absolute 
	returned value will be cropped at 'limit', if this is set.
	
	This routine is implemented in pure Python.
	"""
	# Initial guess for the interpolation
	start = N.argwhere(data == data.max())[0]
	# Use the shift range measured to get the origin (i.e. what pixel 
	# corresponds to a (0,0) shift?)
	offset = -range[0]
	# Crop from the full map to interpolate for
	submap = data[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
	
	a2 = 0.5 * (submap[2,1] - submap[0,1])
	a3 = 0.5 * submap[2,1] - submap[1,1] + 0.5 * submap[0,1]
	a4 = 0.5 * (submap[1,2] - submap[1,0])
	a5 = 0.5 * submap[1,2] - submap[1,1] + 0.5 * submap[1,0]
	a6 = 0.25 * (submap[2,2] - submap[2,0] - \
		submap[0,2] + submap[0,0])
	
	v = N.array([(2*a2*a5-a4*a6)/(a6*a6-4*a3*a5), \
		 	(2*a3*a4-a2*a6)/(a6*a6-4*a3*a5)]) + start + offset
	
	# Debug
	# if (not N.isfinite(v).all()):
	# 	raise RuntimeError("Not all finite")
	
	# if (limit != None):
	# 	v[v > limit] = limit
	# 	v[v < -limit] = -limit
	
	return v


def quadInt2dWeave(data, range, limit=None):
	"""
	Find the extrema of 'data' using a two-dimensional 9-point quadratic
	interpolation (QI formulae by Yi & Molowny Horas (1992, Eq. (10)).). 
	The subpixel maximum will be examined around the coordinate of the pixel 
	with the maximum intensity. 
	
	'range' must be set to the shift-range that the correlation map in 'data' 
	corresponds to to find the pixel corresponding to a shift of 0. Absolute 
	returned value will be cropped at 'limit', if this is set.
	
	This routine is implemented in C using Weave.
	"""
	
	# Empty array which will hold the subpixel maximum
	extremum = N.empty(2)
	# Initial guess for the interpolation
	start = N.argwhere(data == data.max())[0]
	# Use the shift range measured to get the origin (i.e. what pixel 
	# corresponds to a (0,0) shift?)
	offset = -range[0]
	# Crop from the full map to interpolate for. We don't change indices here 
	# because otherwise we would need to change them again when returning the 
	# coordinate of the maximum.
	submap = data[start[0]-1:start[0]+2, start[1]-1:start[1]+2]
	code = """
	#line 615 "libshifts.py" (debugging info)
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
		extra_compile_args= [__COMPILE_OPTS], \
		type_converters=S.weave.converters.blitz)
	
	v = extremum + start + offset
	
	# Debug
	# if (not N.isfinite(v).all()):
	# 	raise RuntimeError("Not all finite")
	
	# if (limit != None):
	# 	v[v > limit] = limit
	# 	v[v < -limit] = -limit
	
	return v


def maxValPython(data):
	"""
	Return the coordinates of the pixel with maximum intensity in 'data'. 
	Although rather useless, this 'subpixel maximum finding' method provides a 
	sort of reference for other methods.
	"""
	return N.argwhere(data == data.max())[0]

#=============================================================================
# Helper routines -- other
#=============================================================================

def findRefIdx(img, saccdpos, saccdsize, refmode=REF_BESTRMS, refopt=1, storeref=False):
	"""
	Based on 'refmode', return a list with indices of subapertures that will 
	be used as a reference for image shift calculations lateron.
	
	'img' should be a wavefront sensor image with 'saccdpos' a list of 
	lower-left pixel coordinates of the subapertures, each 'saccdsize' big. 
	'refmode' should be a valid reference search strategy (see documentation 
	of this file), and 'refopt' is an optional extra parameter for 'refmode'.
		
	NB: This only supports rectangular subapertures.
	"""
	
	# Select the first 'refopt' subapertures with the best RMS
	if (refmode == REF_BESTRMS):
		# If refopt is not an integer, try to convert it
		if (refopt.__class__ != int):
			try:
				refopt = int(refopt)
			except:
				raise ValueError("'refopt' should be an integer.")
		
		# Get RMS for all subimages
		rmslist = []
		for saidx in range(len(saccdpos)):
			_sapos = saccdpos[saidx]
			# Get subimage (remember the reverse indexing)
			_subimg = img[_sapos[1]:_sapos[1] + saccdsize[1], \
				_sapos[0]:_sapos[0] + saccdsize[0]]
			# Calculate RMS
			rms = (((_subimg)**2.0).sum()/(saccdsize[0]*saccdsize[1]))**(0.5)
			# Store in list, store RMS first, index second such that sorting 
			# will be done on the RMS and not the index
			rmslist.append([rms, saidx])
		
		# Sort rmslist reverse, because we want highest values first.
		rmslist.sort(reverse=True)
		# Return the first 'refopt' best RMS indices. Generate a new list on 
		# the fly where we only select the indices, we don't need to return 
		#  the RMS values itself.
		reflist = [rms[1] for rms in rmslist[:refopt]]
	# Select static subapertures, 
	elif (refmode == REF_STATIC):
		# If refopt is not a list, try to convert it
		if (refopt.__class__ != list):
			try:
				refopt = list(refopt)
			except:
				raise ValueError("'refopt' should be a list of integers.")
		
		# Use static subaps here, return refopt itself
		reflist = refopt
	else:
		raise RuntimeError("'refmode' must be one of the predefined reference modes.")
	
	# if (storeref != False):
	# 	# Store the reference subimage if requested
	# 	pyfits.writeto(storeref, ref)
	return reflist


#=============================================================================
# Big control routine
#=============================================================================


def calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, method=COMPARE_ABSDIFFSQ, extremum=EXTREMUM_2D9PTSQ, refmode=REF_BESTRMS, refopt=None, shrange=[3,3], subfields=None, corrmaps=None):
	"""
	Calculate the image shifts for subapertures/subfields in 'img'. 
	Subapertures must be located at pixel positions 'saccdpos' with sizes 
	saccdsize. The subapertures are then located at 'sfccdpos' (relative to 
	saccdpos), with sizes 'sfccdsize' pixelsize.
	
	'method' defines the method to compare the subimages, 'extremum' defines 
	the method to find the best subpixel shift, i.e. what interpolation should 
	be used. 'shrange' defines the possible shifts to test (actual number of 
	distances checked is 2*shrange+1). 'refmode' sets method to choose a 
	reference subaperture, 'refopt' is the reference subaperture used (index) 
	if 'refmode' is set to REF_STATIC.
	
	If an empty list is passed to 'subfields' and/or 'corrmaps', these will 
	contain the subfields analysed and the correlation maps calculated on 
	return.
	
	For regular (non-wide-field) SH WFS, set 'saccdpos' to the subaperture
	positions, 'sfccdpos' to [[0,0]], and 'sfccdsize' to the subimage 
	pixelsize.
	
	For wide-field SH WFS, set 'saccdpos' similarly, but set 'sfccdpos' to an 
	array of pixel positions relative to 'saccdpos' for the subfields to 
	compare. Set 'sfccdsize' not to the complete subimage size, but to the 
	size of the subfield you want to use.
	"""
	
	#===============
	# Initialisation
	#===============
	
	beg = time.time()
	
	# Parse the 'method' argument
	if (method == COMPARE_XCORR):
		prNot(VERB_DEBUG, "calcShifts(): Using direct cross correlation")
		mfunc = crossCorrWeave
	elif (method == COMPARE_SQDIFF):
		prNot(VERB_DEBUG, "calcShifts(): Using square difference")
		mfunc = sqDiffWeave
	elif (method == COMPARE_ABSDIFFSQ):
		prNot(VERB_DEBUG, "calcShifts(): Using absolute difference squared")
		mfunc = absDiffSqWeave
	elif (hasattr(method, '__call__')):
		prNot(VERB_DEBUG, "calcShifts(): Using custom image comparison")
		mfunc = method
	else:
		raise RuntimeError("'method' must be either one of the predefined image comparison methods, or a function doing that.")
	
	# Parse the 'extremum' argument
	if (extremum == EXTREMUM_2D9PTSQ):
		prNot(VERB_DEBUG, "calcShifts(): Using 2d parabola interpolation")
		extfunc = quadInt2dWeave
	elif (extremum == EXTREMUM_MAXVAL):
		prNot(VERB_DEBUG, "calcShifts(): Using maximum value")
		extfunc = maxValPython
	elif (hasattr(extremum, '__call__')):
		prNot(VERB_DEBUG, "calcShifts(): Using custom interpolation")
		extfunc = extremum
	else:
		raise RuntimeError("'extremum' must be either one of the predefined extremum finding methods, or a function doing that.")
	
	# Find reference subaperture(s)
	reflist = findRefIdx(img, saccdpos, saccdsize, refmode=refmode, \
	 	refopt=refopt)
	
	# Init shift vectors (use a list so we can append())
	# Shape will be: ((len(refopt), saccdpos.shape[0], sfccdpos.shape[0], 2))
	disps = []
	
	shrange = N.array(shrange).astype(N.int32)
	
	#=========================
	# Begin shift measurements
	#=========================
	
	# Loop over the reference subapertures
	#-------------------------------------
	for _refsa in reflist:
		prNot(VERB_DEBUG, "calcShifts(): Using subap #%d as reference [%d:%d, %d:%d]" % \
			(_refsa, saccdpos[_refsa][0], saccdpos[_refsa][0]+saccdsize[0], \
			saccdpos[_refsa][1], saccdpos[_refsa][1]+saccdsize[1]))
		# Cut out the reference subaperture
		ref = img[saccdpos[_refsa][1]:saccdpos[_refsa][1]+saccdsize[1], \
			saccdpos[_refsa][0]:saccdpos[_refsa][0]+saccdsize[0]]
		ref = (ref/N.float32(ref.mean()))
		# Expand lists to store measurements in
		disps.append([])
		if (subfields != None): subfields.append([])
		if (corrmaps != None): corrmaps.append([])
		
		# Loop over the subapertures
		#---------------------------
		for _sapos in saccdpos:
			prNot(VERB_ALL, "calcShifts(): -Subimage @ (%d, %d), (%dx%d)"% \
				 	(_sapos[0], _sapos[1], saccdsize[0], saccdsize[1]))
			
			# Expand lists to store measurements in
			disps[-1].append([])
			if (subfields != None): subfields[-1].append([])
			if (corrmaps != None): corrmaps[-1].append([])
			
			# Cut out subimage
			_subimg = img[_sapos[1]:_sapos[1]+saccdsize[1], \
			 	_sapos[0]:_sapos[0]+saccdsize[0]]
			_subimg = (_subimg/N.float32(_subimg.mean()))
			
			# Loop over the subfields
			#------------------------
			for _sfpos in sfccdpos:
				# Current pixel coordinates
				# _pos = _sapos + _sfpos
				# _end = _pos + sfccdsize
				
				# prNot(VERB_ALL, \
				# 	"calcShifts(): --subfield @ (%d, %d), (%dx%d) [%d:%d, %d:%d]" % \
				# 	 	(_sfpos[0], _sfpos[1], sfccdsize[0], sfccdsize[1], \
				# 		_pos[1], _end[1], _pos[0], _end[0]))
				
				# Get the current subfield (remember, the pixel at (x,y) is
				# img[y,x])
				_subfield = _subimg[_sfpos[1]:_sfpos[1]+sfccdsize[1], \
				 	_sfpos[0]:_sfpos[0]+sfccdsize[0]]
				#_subfield = img[_pos[1]:_end[1], _pos[0]:_end[0]]
				# if (_subfield.shape != _subfieldo.shape):
				# 	print _sfpos, sfccdsize
				# 	print _subfield.shape, _subfieldo.shape
				# 	raise RuntimeError("Shapes wrong")
				# if (not N.allclose(_subfield,_subfieldo)):
				# 	print _sfpos, sfccdsize
				# 	print _pos, _end
				# 	raise RuntimeError("Not close, diff: %g" % \
				# 	 	((_subfieldo-_subfield).sum()))
				# Compare the image with the reference image
				diffmap = mfunc(_subfield, ref, _sfpos, shrange)
				#165, 139
				#prNot(VERB_ALL, "calcShifts(): got map, interpolating maximum")
				# Find the extremum, store to list
				shift = extfunc(diffmap, range=shrange, limit=shrange)
				disps[-1][-1].append(shift[::-1])
				
				# Store subfield and correlation map, if requested
				# if (subfields != None): subfields[-1][-1].append(_subfield)
				# if (corrmaps != None): corrmaps[-1][-1].append([diffmap])
				# 			
				# prNot(VERB_ALL, "calcShifts(): --Shift @ (%d,%d): (%.3g, %.3g)" % \
				# 	 	(_sfpos[0], _sfpos[1], shift[1], shift[0]))
	
	# Reform the shift vectors to an numpy array and return it
	# TODO: is N.float32 sensible?
	dur = time.time() - beg
	prNot(VERB_DEBUG, "calcShifts(): done, took %.3g seconds." % (dur))
	ret = N.array(disps).astype(N.float32)
	return ret


#=============================================================================
# Some basic testing functions go here
#=============================================================================

class libshiftTests(unittest.TestCase):
	def setUp(self):
		# Create a test image, shift it, measure shift.
		# As a test image source we use random noise at a certain resolution 
		# (self.res). We then scale the image up with a factor self.scale to 
		# generate structure in the image. After that we scale the image up 
		# with another factor (self.shscl), in this size we will shift the 
		# image around.
		
		# Debug info or not
		self.debug = 1
		if (self.debug): print "setUp(): setting up."
		
		# The resolution of the source image, after shifting and all
		self.srcres = 32
		# The edge of pixels to reserve for shifting around
		self.edge = 5
		# Scale factors
		self.strscl = 4.
		self.shscl = 8.
		
		# Shift vectors
		self.shvecs = N.array([[0.5, 0.2], [3.2, 0.0], [0.0, 3.4], [2.1, 2.9], [3.2, 1.8]])
		# Maximum shift range to test
		self.shrange = N.array([4, 4])
			
		# number of iterations to use for timing tests
		self.nit = 1000
		
		# Generate image source
		self.src = N.random.random([self.srcres/self.strscl]*2)*255
		if (self.debug):
			print "setUp(): src: %.3g (%dx%d)" % (self.src.mean(), \
		 		self.src.shape[0], self.src.shape[1])
		
		# Blow up to get structure (self.strscl) and to make subpixel shifts 
		# possible (self.shscl)
		self.src = S.ndimage.zoom(self.src, self.strscl*self.shscl)
		if (self.debug):
			print "setUp(): src: %.3g (%dx%d)" % (self.src.mean(), \
		 		self.src.shape[0], self.src.shape[1])
		
		# Get reference from source (the center)
		self.refsame = self.src[\
			self.edge*self.shscl:\
			(self.srcres-self.edge)*self.shscl,\
			self.edge*self.shscl:\
			(self.srcres-self.edge)*self.shscl]
		self.refsame = S.ndimage.zoom(self.refsame, 1.0/self.shscl)
		if (self.debug): print "setUp(): refsame shape:", self.refsame.shape
		# Get full reference (which is bigger)
		self.refbigger = self.src
		self.refbigger = S.ndimage.zoom(self.refbigger, 1.0/self.shscl)
		if (self.debug): print "refbigger shape:", self.refbigger.shape
		
		# Get shifted image(s)
		self.shimgs = []
		# These hold the *actual* shifts
		self.rshvecs = []
		for shvec in self.shvecs:
			# We cannot shift arbitrary vectors, exact shifts depend on 
			# self.shscl as well.
			_shvec = N.round(shvec*self.shscl)/self.shscl
			shimg = self.src[\
				(self.edge + _shvec[1])*self.shscl:\
				(self.srcres - self.edge + _shvec[1])*self.shscl,\
				(self.edge + _shvec[0])*self.shscl:\
				(self.srcres - self.edge + _shvec[0])*self.shscl]
			#if (self.debug): print "setUp(): bigimg shape:", shimg.shape
			shimg = S.ndimage.zoom(shimg, 1.0/self.shscl)
			self.shimgs.append(shimg)
			self.rshvecs.append(_shvec)
			if (self.debug):
				print "setUp(): img (%dx%d) %.3g +- %.3g, shift (%.3g,%.3g)"%\
					(shimg.shape[0], shimg.shape[1], shimg.mean(), \
					shimg.std(), _shvec[0], _shvec[1])
		
		if (self.debug):
			print "setUp(): reference: %.3g (%dx%d)" % (self.refsame.mean(), \
		 		self.refsame.shape[0], self.refsame.shape[1])
		# Done
	
	def runTests(self):
		unittest.TextTestRunner(verbosity=2).run(self.suite())
	
	def suite(self):
		suite = unittest.TestLoader().loadTestsFromTestCase(self)
		return suite
	
	def testCcQi(self):
		print
		# crossCorrWeave + quadInt2dWeave
		for (img, sh) in zip(self.shimgs, self.rshvecs):
			diffmap = crossCorrWeave(img, self.refsame, N.array([0,0]), \
		 		self.shrange)
			if (self.debug):
				print "testCcQi(): map: %.3g +- %.3g. %.3g -- %.3g" % \
			 		(N.mean(diffmap), N.std(diffmap), N.min(diffmap), \
			 		N.max(diffmap))
				print "testCcQi(): map: %dx%d, range: %dx%d" % \
					(diffmap.shape[0], diffmap.shape[1], self.shrange[0], \
					self.shrange[1])
			shift = quadInt2dWeave(diffmap, self.shrange)
			print "testCcQi(): shift: (%.3g, %.3g), diff: (%.3g, %.3g)"%\
				(shift[1], shift[0], shift[1]-sh[0], shift[0]-sh[1])
	
	def testSqdQi(self):
		# sqDiffWeave + quadInt2dWeave
		print
		for (img, sh) in zip(self.shimgs, self.rshvecs):
			diffmap = sqDiffWeave(img, self.refsame, N.array([0,0]), \
		 		self.shrange)
			if (self.debug):
				print "testSqdQi(): map: %.3g +- %.3g. %.3g -- %.3g" % \
			 		(N.mean(diffmap), N.std(diffmap), N.min(diffmap), \
			 		N.max(diffmap))
			shift = quadInt2dWeave(diffmap, self.shrange)
			print "testSqdQi(): shift: (%.3g, %.3g), diff: (%.3g, %.3g)"%\
				(shift[1], shift[0], shift[1]-sh[0], shift[0]-sh[1])
	
	def testAdsQi(self):
		print
		# absDiffSqWeave + quadInt2dWeave
		for (img, sh) in zip(self.shimgs, self.rshvecs):
			diffmap = absDiffSqWeave(img, self.refsame, N.array([0,0]), \
		 		self.shrange)
			if (self.debug):
				print "testAdsQi(): map: %.3g +- %.3g. %.3g -- %.3g" % \
			 		(N.mean(diffmap), N.std(diffmap), N.min(diffmap), \
			 		N.max(diffmap))
			shift = quadInt2dWeave(diffmap, self.shrange)
			print "testAdsQi(): shift: (%.3g, %.3g), diff: (%.3g, %.3g)"%\
				(shift[1], shift[0], shift[1]-sh[0], shift[0]-sh[1])
	
	def testCalcShifts(self):
		# Test the whole calcShifts() function
		pass
	
	def testTiming(self):
		# Time the various methods
		img = self.shimgs[0]
		sh = self.rshvecs[0]
		
		print "testTiming(): Starting benchmarks."
		
		### absDiffSqWeave/quadInt2dWeave
		beg = time.time()
		for i in xrange(self.nit):
			diffmap = absDiffSqWeave(img, self.refsame, N.array([0,0]), \
		 		self.shrange)
			shift = quadInt2dWeave(diffmap, self.shrange)
		end = time.time()
		print "testTiming(): absDiffSqWeave/quadInt2dWeave: %.3gs/it." % \
			((end-beg)/self.nit)
		
		### sqDiffWeave/quadInt2dWeave
		beg = time.time()
		for i in xrange(self.nit):
			diffmap = sqDiffWeave(img, self.refsame, N.array([0,0]), \
		 		self.shrange)
			shift = quadInt2dWeave(diffmap, self.shrange)
		end = time.time()
		print "testTiming(): sqDiffWeave/quadInt2dWeave: %.3gs/it." % \
			((end-beg)/self.nit)
		
		### crossCorrWeave/quadInt2dWeave
		beg = time.time()
		for i in xrange(self.nit):
			diffmap = crossCorrWeave(img, self.refsame, N.array([0,0]), \
		 		self.shrange)
			shift = quadInt2dWeave(diffmap, self.shrange)
		end = time.time()
		print "testTiming(): crossCorrWeave/quadInt2dWeave: %.3gs/it." % \
			((end-beg)/self.nit)
	
# Run tests if we call this library instead of importing it
if __name__ == '__main__':
	print "libshifts.py: Running selftests"
	suite = unittest.TestLoader().loadTestsFromTestCase(libshiftTests)
	unittest.TextTestRunner(verbosity=2).run(suite)


