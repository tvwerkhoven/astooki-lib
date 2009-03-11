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

The naming convention in this library for subaperture/subfield position and sizes is done as follows:
- Full variable name is: [sa|sf][pix|][pos|size]
- [sa|sf] defines whether it is a subaperture or subfield
- [pix|] defines whether the units are in pixels or SI units
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

$Id
"""

#=============================================================================
# Import necessary libraries here
#=============================================================================

import numpy as N
import scipy as S
import scipy.ndimage			# For fast ndimage processing
import scipy.weave				# For inlining C
import scipy.weave.converters	# For inlining C
import scipy.fftpack			# For FFT functions in FFT cross-corr
import scipy.signal				# For hanning/hamming windows
import pyfits
import timlib					# Some miscellaneous functions

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
REF_BESTRMS = 0					# Use subimage with best RMS as reference
#REF_WHOLEFIELDREF = 1			# Supply a whole image to be used as reference
#REF_NOREF = 2					# Don't use a reference, cross-compare 
								# everything with eachother. This will greatly 
								# increase CPU load to about N! instead of N 
								# comparisons.
REF_STATIC = 4					# Use a static reference subaperture, use the 
								# 'refapt' parameter in functions using this
								# to specify which subap should be used

# Fourier windowing method constants
FFTWINDOW_NONE = 0				# No window
FFTWINDOW_HANN = 1				# Hann window
FFTWINDOW_HAMM = 2				# Hamming window
FFTWINDOW_HANN50 = 3			# Hann window with 50% flat surface
FFTWINDOW_HAMM50 = 4			# Hamming window with 50% flat surface

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
	diffmap = N.zeros(range*2+1)
	
	# Cross correlation needs some pre-processing
	tmpimg = img - img.mean()
	tmpref = ref - ref.mean()
	
	raise RuntimeWarning("crossCorrWeave() is not really working at the moment.")
	
	code = """
	#line 159 "libshifts.py" (debugging info for compilation)
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
						tmpsum += img(i,j) * ref(i+sh0,j+sh1);	
					}
				}
				// Store the current correlation value in the map.
				diffmap((sh0-sh0min), (sh1-sh1min)) = tmpsum;
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
				for (int i=0 - min(sh0, 0); i<Nimg[0] - max(sh0, 0); i++) {
					for (int j=0 - min(sh1, 0); j<Nimg[1] - max(sh1, 0); j++){
						tmpsum += img(i,j) * ref(i+sh0,j+sh1);
					}
				}
				
				// Scale the value found by dividing it by the number of 
				// pixels we compared.
				diffmap((sh0-sh0min), (sh1-sh1min)) = tmpsum / 
					((Nimg[0]-abs(sh0)) * (Nimg[1]-abs(sh1)));
			}
		}
		//printf("%lf, %d and %d \\n", 
		//	tmpsum, (Nimg[0]), (Nimg[1]));
	}
	return_val = 1;
	"""
	one = S.weave.inline(code, \
		['tmpref', 'tmpimg', 'pos', 'range', 'diffmap'], \
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
	diffmap = N.empty(range*2+1)
	
	code = """
	#line 251 "libshifts.py" (debugging info for compilation)
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
						diff = img(i,j) - ref(i+sh0,j+sh1);
						// ...then square this
						tmpsum += diff*diff;
					}
				}
				// Store the current correlation value in the map. Use
				// negative value to ensure that we get a maximum for best
				// match in diffmap (this allows to use a more general 
				// maximum-finding method, instead of splitting between maxima
				// and minima)
				diffmap((sh0-sh0min), (sh1-sh1min)) = -tmpsum;
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
				for (int i=0 - min(sh0, 0); i<Nimg[0] - max(sh0, 0); i++) {
					for (int j=0 - min(sh1, 0); j<Nimg[1] - max(sh1, 0); j++){
						// First get the difference...
						diff = img(i,j) - ref(i+sh0,j+sh1);
						// ...then square this
						tmpsum += diff*diff;
					}
				}
				// Scale the value found by dividing it by the number of 
				// pixels we compared.
				diffmap((sh0-sh0min), (sh1-sh1min)) = -tmpsum / 
					((Nimg[0]-abs(sh0)) * (Nimg[1]-abs(sh1)));
			}
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
	diffmap = N.empty(range*2+1)
	
	code = """
	#line 349 "libshifts.py" (debugging info for compilation)
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
						tmpsum += fabs(img(i,j) - ref(i+sh0,j+sh1));
					}
				}
				// Store the current correlation value in the map. Use
				// negative value to ensure that we get a maximum for best
				// match in diffmap
				diffmap((sh0-sh0min), (sh1-sh1min)) = -(tmpsum*tmpsum);
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
				for (int i=0 - min(sh0, 0); i<Nimg[0] - max(sh0, 0); i++) {
					for (int j=0 - min(sh1, 0); j<Nimg[1] - max(sh1, 0); j++){
						tmpsum += fabs(img(i,j) - ref(i+sh0,j+sh1));
					}
				}
				// Scale the value found by dividing it by the number of 
				// pixels we compared.
				diffmap((sh0-sh0min), (sh1-sh1min)) = -(tmpsum*tmpsum) /
					 ((Nimg[0]-abs(sh0)) * (Nimg[1]-abs(sh1)));
			}
		}
	}
	return_val = 1;
	"""
	
	one = S.weave.inline(code, \
		['ref', 'img', 'pos', 'range', 'diffmap'], \
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
	if (range < res/2).all():
		diffmap = (timlib.shift(fftmap.real, range))\
			[:range[0]*2+1,:range[0]*2+1]
	else:
		raise RuntimeError("range > res/2 is not implemented.")
	
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
	if (not N.isfinite(v).all()):
		raise RuntimeError("Not all finite")
	
	if (limit != None):
		v[v > limit] = limit
		v[v < -limit] = -limit
	
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
	#line 583 "libshifts.py" (debugging info)
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
	
	v = extremum + start + offset
	
	# Debug
	# if (not N.isfinite(v).all()):
	# 	raise RuntimeError("Not all finite")
	
	if (limit != None):
		v[v > limit] = limit
		v[v < -limit] = -limit
	
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

def findRef(img, sapos, sasize, refmode=REF_BESTRMS, refapt=None, storeref=False, verb=0):
	"""
	Find a reference subaperture within 'img' using 'refmode' as criteria. 
	'img' is expected to be a wavefront sensor image with 'sapos' a list of 
	lower-left pixel coordinates of the subapertures, each 'sasize' big.
	Returns the cutout reference subaperture as result. Use the 'storeref'
	option to save it to disk, if this is desired, pass a filepath to 
	'storeref'. 
	
	NB: This only supports rectangular subapertures.
	"""
	
	if (refmode == REF_BESTRMS):
		# Find the subimage with the best RMS
		bestrms = 0.0
		for _sapos in sapos:
			# Get subimage
			_subimg = img[_sapos[1]:_sapos[1] + sasize[1], \
				_sapos[0]:_sapos[0] + sasize[0]]
			# Calc RMS^2
			rms = ((_subimg)**2.0).sum()/sasize[0]/sasize[1]
			# Comparse against previous best
			if (rms > bestrms):
				# If better, store this rms and reference subimage
				if (verb>0): 
					print "findRef(): Found new best rms^2 @ (%d,%d), rms^2 is: %.3g, was: %.3g." % (_sapos[0], _sapos[1], rms, bestrms)
				bestrms = rms
				bestpos = _sapos
				ref = _subimg
		# Take root to get real RMS
		bestrms = bestrms**0.5
		if (verb>1):
			print "findRef(): -Best RMS subimage @ (%d,%d) (rms: %.3g)" % (bestpos[0], bestpos[1], bestrms)
	elif (refmode == REF_STATIC):
		# Use static subap here
		_sapos = sapos[refapt]
		ref = img[_sapos[1]:_sapos[1] + sasize[1], \
			_sapos[0]:_sapos[0] + sasize[0]]
		if (verb>1):
			print "findRef(): -Static subap @ (%d,%d) (rms: %.3g)" % (_sapos[0], _sapos[1], (((ref)**2.0).sum()/sasize[0]/sasize[1])**0.5 )
	else:
		raise RuntimeError("'refmode' must be one of the predefined reference modes.")
	
	if (storeref != False):
		# Store the reference subimage if requested
		pyfits.writeto(storeref, ref)
	
	return ref


#=============================================================================
# Big control routine
#=============================================================================


def calcShifts(img, sapos, sasize, sfpos, sfsize, method=COMPARE_ABSDIFFSQ, extremum=EXTREMUM_2D9PTSQ, refmode=REF_BESTRMS, refapt=None, shrange=[3,3], verb=0, subfields=None, corrmaps=None):
	"""
	Calculate the image shifts for subapertures/subfields in 'img' located at 
	pixel positions 'sapos' and 'sfpos' respectively, each with the same
	'subsize' pixelsize. 
	
	'method' defines the method to compare the subimages, 'extremum' defines 
	the method to find the best subpixel shift, i.e. what interpolation should 
	be used. 'srange' defines the possible shifts to test (actual number of 
	distances checked is 2*range+1). 'refmode' sets method to choose a 
	reference subaperture, 'refapt' is the reference subaperture used (index) 
	if 'refmode' is set to REF_STATIC.
	
	If an empty list is passed to 'subfields' and/or 'corrmaps', these will 
	contain the subfields analysed and the correlation maps calculated on 
	return.
	
	For regular (non-wide-field) SH WFS, set 'sapos' to the subaperture
	positions, 'sfpos' to [[0,0]], and 'subsize' to the subimage pixelsize.
	
	For wide-field SH WFS, set 'sapos' similarly, but set 'sfpos' to an array 
	of pixelpositions relative to 'sapos' for the subfields to compare. Set
	'subsize' not to the complete subimage size, but to the size of the
	subfield you want to use.
	"""
	
	#===============
	# Initialisation
	#===============
	
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
	
	# Find reference subaperture
	ref = findRef(img, sapos, sasize, refmode=refmode, refapt=refapt)
	
	# Init shift vectors (use a list so we can append())
	disps = [] # N.zeros((sapos.shape[0], sfpos.shape[0], 2))
	
	#=========================
	# Begin shift measurements
	#=========================
	
	# Loop over the subapertures
	if (verb>0): print "calcShifts(): Processing data"
	
	for _sapos in sapos:
		if (verb>1): print "calcShifts(): -Processing subimage @ (%d, %d), sized (%dx%d)" % \
			 	(_sapos[0], _sapos[1], sasize[0], sasize[1])
		
		# Init lists to store displacements, correlation maps and subfields in
		ldisps = []
		lcmaps = []
		lsubfields = []
		
		# Loop over the subfields
		for _sfpos in sfpos:
			# Current pixel position
			_pos = _sapos + _sfpos
			_end = _pos + sfsize
			
			if (verb>1): print "calcShifts(): --Processing subfield @ (%d, %d), sized (%dx%d)" % \
				 	(_sfpos[0], _sfpos[1], sfsize[0], sfsize[1])
			
			# Get the current subfield (remember, the pixel at (x,y) is
			# img[y,x])
			_subimg = img[_pos[1]:_end[1], _pos[0]:_end[0]]
			
			# Store subfield
			if (subfields != None):
				lsubfields.append(_subimg)
			
			# Compare the image with the reference image
			diffmap = mfunc(_subimg, ref, _sfpos, shrange)
			
			# Store correlation map
			if (corrmaps != None):
				lcmaps.append(diffmap)
			
			# Find the extremum, store to list
			shift = extfunc(diffmap, range=shrange, limit=shrange)
			ldisps.append(shift[::-1])
			
			if (verb>1): print "calcShifts(): --Found shift: (%.3g, %.3g)" % \
				 	(shift[1], shift[0])
		
		# Append the list of subfield data to the list of subimage data (this
		# gives us a 2d list that can be indexed using subimage and subfield
		# indices)
		subfields.append(lsubfields)
		corrmaps.append(lcmaps)
		disps.append(ldisps)
	
	# Reform the shift vectors to an numpy array and return it
	return N.array(disps)

if __name__ == '__main__':
	pass

