/**
@file libshifts-c.c
@brief Image shift measurement library for Shack-Hartmann wavefront sensors -- fast C version
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090429

Compile with
gcc -shared -O3 -Wall -I/sw/include/python2.5/ -I/sw/lib/python2.5/site-packages/numpy/core/include-c libshifts-c.c

Created by Tim van Werkhoven on 2009-04-29.
Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
*/

// Headers
#include <Python.h>				// For python extension
#include <numpy/arrayobject.h> 	// For numpy
//#include <Numeric/arrayobject.h> 	// For numpy
//#include <numarray/arrayobject.h> 	// For numpy
#include <sys/time.h>			// For timestamps
#include <time.h>				// For timestamps

// 'Standard' floats
typedef float float32_t;
typedef double float64_t;

// Prototypes
static PyObject * libshifts_calcshifts(PyObject *self, PyObject *args);
// def calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, method=COMPARE_ABSDIFFSQ, extremum=EXTREMUM_2D9PTSQ, refmode=REF_BESTRMS, refopt=None, shrange=[3,3], subfields=None, corrmaps=None):
int _findrefidx_float32(float32_t *image, int32_t stride, int32_t pos[][2], int npos, int32_t size[2], int refmode, int refopt, int list[]);
//findRefIdx(img, saccdpos, saccdsize, refmode=REF_BESTRMS, refopt=1, storeref=False):
PyObject *_calcshifts_float32(float32_t *image, int32_t saccdpos[][2], int nsapos, int32_t saccdsize[2], int32_t sfccdpos[][2], int nsfpos, int32_t sfccdsize[2], int32_t shran[2], int compmeth, int extmeth, int refmode, int refopt, int debug);
// Defines

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


// Methods table for this module
static PyMethodDef LibshiftsMethods[] = {
	{"calcShifts",  libshifts_calcshifts, METH_VARARGS, "Calculate image shifts."},
	{NULL, NULL, 0, NULL}        /* Sentinel */
};


// Init module methods
PyMODINIT_FUNC init_libshifts(void) {
	(void) Py_InitModule("_libshifts", LibshiftsMethods);
	// Init numpy usage
	import_array();
}

static PyObject * libshifts_calcshifts(PyObject *self, PyObject *args) {
	// Python function arguments
	PyArrayObject *image, *saccdpos, *saccdsize, *sfccdpos, *sfccdsize, *shrange;
	int compmeth = COMPARE_SQDIFF, extmeth = EXTREMUM_2D9PTSQ;
	int refmode = REF_BESTRMS, refopt = 0;
	int debug=1;
	PyObject *shifts;
	// Generic variables
	int i, j;

	// Parse arguments from Python function
	if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!|iiiii", 
		&PyArray_Type, &image, 								// Image data
		&PyArray_Type, &saccdpos, 						// Subaperture positions
		&PyArray_Type, &saccdsize, 						// SA size
		&PyArray_Type, &sfccdpos, 						// Subfield positions
		&PyArray_Type, &sfccdsize, 						// SF size
		&PyArray_Type, &shrange, 								// Shift range
		&compmeth,
		&extmeth, 
		&refmode,
		&refopt,
		&debug))
		return NULL;
	
	if (debug > 0) printf("libshifts_calcshifts(): img: 0x%p, sapos: 0x%p, sasize: 0x%p, sfpos: 0x%p, sfsize: 0x%p, shran: 0x%p.", image, saccdpos, saccdsize, sfccdpos, sfccdsize, shrange);
	
	// Verify that saccdpos, saccdsize, sfccdpos, sfccdsize, shran are all int32
	if (PyArray_TYPE((PyObject *) saccdpos) != NPY_INT32 ||
		PyArray_TYPE((PyObject *) saccdsize) != NPY_INT32 ||
		PyArray_TYPE((PyObject *) sfccdpos) != NPY_INT32 ||
		PyArray_TYPE((PyObject *) sfccdsize) != NPY_INT32 ||
		PyArray_TYPE((PyObject *) shrange) != NPY_INT32) {
			if (debug > 0) 
				printf("...: coordinates and ranges are not int32.\n");
			PyErr_SetString(PyExc_ValueError, "In calcshifts: coordinates and ranges should be int32.");
			return NULL;
		}
	
	// Refopt should be somewhere between 0 and 100
	if (refopt < 0 || refopt > 100) {
		if (debug > 0) 
			printf("...: refopt value invalid.\n");
		PyErr_SetString(PyExc_ValueError, "In calcshifts: refopt value invalid.");
		return NULL;	
	}
	
	// Get number of subapertures and subfields
	int nsa = (int) PyArray_DIM((PyObject*) saccdpos, 0);
	int nsf = (int) PyArray_DIM((PyObject*) sfccdpos, 0);
	if (debug > 0)
		printf("...: nsa: %d, nsf: %d.\n", nsa, nsf);
	
	// Convert options
	int32_t sapos[nsa][2];
	for (i=0; i<nsa; i++) {
		sapos[i][0] = *((uint32_t *)PyArray_GETPTR2((PyObject *) saccdpos, i, 0));
		sapos[i][1] = *((uint32_t *)PyArray_GETPTR2((PyObject *) saccdpos, i, 1));
		if (debug > 0)
			printf("...: sa %d: %d,%d.\n", i, sapos[i][0], sapos[i][1]);
	}
	int32_t sfpos[nsf][2];
	for (i=0; i<nsf; i++) {
		sfpos[i][0] = *((uint32_t *)PyArray_GETPTR2((PyObject *) sfccdpos, i, 0));
		sfpos[i][1] = *((uint32_t *)PyArray_GETPTR2((PyObject *) sfccdpos, i, 1));
		if (debug > 0)
			printf("...: sf %d: %d,%d.\n", i, sfpos[i][0], sfpos[i][1]);
	}
	int32_t sasize[2];
	sasize[0] = *((uint32_t *) PyArray_GETPTR1((PyObject *) saccdsize, 0));
	sasize[1] = *((uint32_t *) PyArray_GETPTR1((PyObject *) saccdsize, 1));
	if (debug > 0)
		printf("...: sasize: %d,%d.\n", sasize[0], sasize[1]);
	int32_t sfsize[2];
	sfsize[0] = *((uint32_t *) PyArray_GETPTR1((PyObject *) sfccdsize, 0));
	sfsize[1] = *((uint32_t *) PyArray_GETPTR1((PyObject *) sfccdsize, 1));
	if (debug > 0)
		printf("...: sfsize: %d,%d.\n", sfsize[0], sfsize[1]);
	
	int32_t shran[2];
	shran[0] = *((uint32_t *) PyArray_GETPTR1((PyObject *) shrange, 0));
	shran[1] = *((uint32_t *) PyArray_GETPTR1((PyObject *) shrange, 1));
	if (debug > 0)
		printf("...: shran: %d,%d.\n", shran[0], shran[1]);
	
	// Check image datatype
	switch (PyArray_TYPE((PyObject *) image)) {
		case (NPY_FLOAT32): 
			if (debug > 0) 
				printf("...: Found type NPY_FLOAT32\n");
			// Convert options
			float32_t *im32 = (float32_t *) PyArray_DATA((PyObject *) image);
			int32_t *shifts = _calcshifts_float32(im32, sapos, nsa, sasize, sfpos, nsf, sfsize, shran, compmeth, extmeth, refmode, refopt, debug);
			break;
		// case (NPY_FLOAT64): 
		// 	if (debug > 0) 
		// 		printf("libshifts_calcshifts(): Found type NPY_FLOAT64\n");
		// 	break;
		default:
			if (debug > 0) 
				printf("...: unsupported type.\n");
			PyErr_SetString(PyExc_ValueError, "In calcshifts: datatype not supported.");
			return NULL;
	}
	
		
	return Py_BuildValue("i", 1);
}


int _findrefidx_float32(float32_t *image, int32_t stride, int32_t pos[][2], int npos, int32_t size[2], int refmode, int refopt, int list[]) {
	if (debug > 0)
		printf("_findrefidx_float32() im: 0x%p, pos: 0x%p, size: 0x%p.\n", image, pos, size);
	// Find a reference subaperture in 'image'
	int sa, i, j;
	double rmslist[npos];
	for (sa=0; sa < npos; sa++) {
		// Calculate RMS for subaperture 'sa'
		for (i=0; i<size[0]; i++) {
			rmslist[sa] = image*0.5;
		}
		if (debug >0)
			printf("...: rms for sa %d = %.3g\n", sa, rmslist[sa]);
	}
	list[0] = 1;
	
	return 1;
}

int32_t *_calcshifts_float32(float32_t *image, int32_t saccdpos[][2], int nsapos, int32_t saccdsize[2], int32_t sfccdpos[][2], int nsfpos, int32_t sfccdsize[2], int32_t shran[2], int compmeth, int extmeth, int refmode, int refopt, int debug) {
	if (debug > 0) 
		printf("_calcshifts_float32().\n");

	
	// First get the reference subapertures
	int reflist[refopt], ret;
	ret = _findrefidx_float32(image, saccdpos, nsapos, saccdsize, refmode, refopt, reflist);
	
	return NULL;
}
