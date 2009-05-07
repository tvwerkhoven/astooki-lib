/**
@file libshifts-c.c
@brief Image shift measurement library for Shack-Hartmann wavefront sensors -- fast C version
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090429

Created by Tim van Werkhoven on 2009-04-29.
Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
*/

//
// Headers
//

#ifndef __LIBSHIFTS_C_H__  // __LIBSHIFTS_C_H__
#define __LIBSHIFTS_C_H__


#include <Python.h>				// For python extension
#include <numpy/arrayobject.h> 	// For numpy
//#include <Numeric/arrayobject.h> 	// For numpy
//#include <numarray/arrayobject.h> 	// For numpy
#include <stdint.h>
#include <sys/time.h>			// For timestamps
#include <time.h>					// For timestamps
#include <math.h>					// For pow()

#define NTHREADS 1
//
// Types
//

typedef float float32_t;  // 'Standard' 32 bit float type
typedef double float64_t; // 'Standard' 64 bit float type

// Data 
struct thread_data32 {
	float32_t *img;         // Big image and stride
	int32_t stride;
  float32_t *shifts;
	int32_t (*sapos)[2];	        // Subaperture positions
  int32_t nsa;
	int32_t (*sfpos)[2];	        // Subfield positions
  int32_t nsf;
	float32_t *ref;         // Reference subaperture (already normalized)
  int32_t refsa;
	int32_t *sasize;	    // Subaperture size
	int32_t *sfsize;	    // Subfield size
	int32_t *shran;       // Shift range to test
	int32_t dosa[2];        // Subapertures this thread should process
};

//
// Prototypes
//

// Python-accessible functions
static PyObject * libshifts_calcshifts(PyObject *self, PyObject *args);

// Helper routines
int _findrefidx_float32(float32_t *image, int32_t stride, int32_t sapos[][2], int npos, int32_t sasize[2], int refmode, int refopt, int32_t **list, int32_t *nref);

int _calcshifts_float32(float32_t *image, int32_t stride, int32_t sapos[][2], int nsa, int32_t sasize[2], int32_t sfpos[][2], int nsf, int32_t sfsize[2], int32_t shran[2], int compmeth, int extmeth, int32_t *reflist, int32_t nref, float32_t **shifts);

void *_procsubaps_float32(void* args);

int _sqdiff(float32_t *img, int32_t imgsize[2], int32_t imstride, float32_t *ref, int32_t refsize[2], int32_t refstride, float32_t *diffmap, int32_t pos[2], int32_t range[2]);

int _quadint(float32_t *diffmap, int32_t diffsize[2], float32_t shvec[2], int32_t shran[2]);

// One-liner help functions
int _comp_dbls(const double *a, const double *b) {
  return (int) (*b - *a);
}

//
// Defines
//

#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

// Defines for comparison algorithms
#define COMPARE_XCORR 0				// Direct cross correlation
#define COMPARE_SQDIFF 1			// Square difference
#define COMPARE_ABSDIFFSQ 2		// Absolute difference squared
#define COMPARE_FFT 3					// Fourier method

// Defines for extremum finding algorithms
#define EXTREMUM_2D9PTSQ 0		// 2d 9 point parabola interpolation
#define EXTREMUM_MAXVAL 1		 	// maximum value (no interpolation)

// Defines for reference usage
#define REF_BESTRMS 0         // Use subimages with best RMS as reference, 
								 							// 'refopt' should be an integer indicating how 
								 							// many references should be used.
#define REF_STATIC 4					// Use static reference subapertures, pass a 
															// list to the 'refopt' parameter to specify
															// which subaps should be used.


#endif // __LIBSHIFTS_C_H__