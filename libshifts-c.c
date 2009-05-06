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
#include <math.h>				// For pow


#ifndef max
#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif


// 'Standard' floats
typedef float float32_t;
typedef double float64_t;

// Prototypes
static PyObject * libshifts_calcshifts(PyObject *self, PyObject *args);
// def calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, method=COMPARE_ABSDIFFSQ, extremum=EXTREMUM_2D9PTSQ, refmode=REF_BESTRMS, refopt=None, shrange=[3,3], subfields=None, corrmaps=None):
int _findrefidx_float32(float32_t *image, int32_t stride, int32_t pos[][2], int npos, int32_t size[2], int refmode, int refopt, int list[], int debug);
//findRefIdx(img, saccdpos, saccdsize, refmode=REF_BESTRMS, refopt=1, storeref=False):
int32_t *_calcshifts_float32(float32_t *image, int32_t stride, int32_t saccdpos[][2], int nsapos, int32_t saccdsize[2], int32_t sfccdpos[][2], int nsfpos, int32_t sfccdsize[2], int32_t shran[2], int compmeth, int extmeth, int refmode, int refopt, int debug);
int _sqdiff(float32_t *img, int32_t imgsize[2], int32_t imstride, float32_t *ref, int32_t refsize[2], int32_t refstride, float32_t *diffmap, int32_t pos[2], int32_t range[2]);
int _quadint(float32_t *diffmap, int32_t diffsize[2], float32_t shvec[2], int32_t shran[2]);
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
	int refmode = REF_BESTRMS, refopt = 1;
	int debug=1;
	//PyObject *shifts;
	// Generic variables
	int32_t stride;
	int i;

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
	float32_t *im32;
	switch (PyArray_TYPE((PyObject *) image)) {
		case (NPY_FLOAT32): {
			if (debug > 0) 
				printf("...: Found type NPY_FLOAT32\n");
			// Convert options
			
			im32 = (float32_t *) PyArray_DATA((PyObject *) image);
			stride = PyArray_STRIDES((PyObject *) image)[0] / PyArray_ITEMSIZE((PyObject *) image);
			if (debug > 0) 
				printf("...: Calling _calcshifts_float32().\n");
			int32_t *shifts = _calcshifts_float32(im32, stride, sapos, nsa, sasize, sfpos, nsf, sfsize, shran, compmeth, extmeth, refmode, refopt, debug);
			break;
		}
		// case (NPY_FLOAT64): {
		// 	if (debug > 0) 
		// 		printf("libshifts_calcshifts(): Found type NPY_FLOAT64\n");
		// 	break;
		// }
		default: {
			if (debug > 0) 
				printf("...: unsupported type.\n");
			PyErr_SetString(PyExc_ValueError, "In calcshifts: datatype not supported.");
			return NULL;
		}
	}
	
	return Py_BuildValue("i", 1);
	
	// Cut out something
	float32_t pix, mean, *tmpdata;
	tmpdata = malloc(sasize[0] * sasize[1] * sizeof(float32_t));
	int j;
	mean=0;
	printf("pos: %d,%d\n", sapos[1][0], sapos[1][1]);
	for (j=0; j<sasize[1]; j++) {
		for (i=0; i<sasize[0]; i++) {
			pix = im32[(sapos[1][1] + j) * stride + sapos[1][0] + i];
			tmpdata[sasize[1] * j + i] = pix;
			mean += pix;
		}
	}
	printf("mean: %.18f\n", mean/(sasize[0] * sasize[1]));

	// Create numpy array
	PyArrayObject *tmpret;
	npy_intp npy_dims[] = {sasize[0], sasize[1]};
	int npy_type = NPY_FLOAT32;

	tmpret = (PyArrayObject*) PyArray_SimpleNewFromData(2, npy_dims,
		npy_type, (void *) tmpdata);
	// Make sure Python owns the data, so it will free the data after use
	PyArray_FLAGS(tmpdata) |= NPY_OWNDATA;
	
	if (!PyArray_CHKFLAGS(tmpdata, NPY_OWNDATA)) {
		PyErr_SetString(PyExc_RuntimeError, "In calcshifts: unable to own the data, will cause memory leak. Aborting");
		return NULL;
	}
	return Py_BuildValue("N", tmpret);
}

int _comp_dbls(const double *a, const double *b) {
  return (int) (*b - *a);
}

int _findrefidx_float32(float32_t *image, int32_t stride, int32_t sapos[][2], int npos, int32_t sasize[2], int refmode, int refopt, int list[], int debug) {
	if (debug > 0)
		printf("_findrefidx_float32() im: 0x%p, pos: 0x%p, size: 0x%p.\n", image, sapos, sasize);
	// Find a reference subaperture in 'image'
	int sa, ssa, i, j;
	double rmslist[npos], rmslists[npos];
	
	// Calculate RMS values
	for (sa=0; sa < npos; sa++) {
		if (debug > 0) printf("...: checking subaperture %d...", sa);
		//if (debug > 0) printf("...: img 0x%p, stride: %d, sapos: (%d,%d)\n", image, stride, sapos[sa][0], sapos[sa][1]);
		rmslist[sa] = 0;
		double mean=0;
		// Calculate mean
		for (j=0; j<sasize[1]; j++)
			for (i=0; i<sasize[0]; i++)
				mean += image[(sapos[sa][1] + j) * stride + sapos[sa][0] + i];
		mean /= (sasize[0] * sasize[1]);
		for (j=0; j<sasize[1]; j++) {
			for (i=0; i<sasize[0]; i++) {
				rmslist[sa] += pow(image[(sapos[sa][1] + j) * stride + sapos[sa][0] + i] - mean, 2.0);
			}
		}
		rmslist[sa] = 100.0*pow(rmslist[sa]/(sasize[0]*sasize[1]), 0.5)/mean;
		rmslists[sa] = rmslist[sa];
		if (debug >0)
			printf(" rms is: %g\n", rmslist[sa]);
	}
	
	// Get first refopt best values
	printf("...: sorting RMS values\n");
	qsort((void*) rmslists, npos, sizeof(double), _comp_dbls);
	
	for (ssa=0; ssa < refopt; ssa++) {
		printf("...: Searching #%d rms: %g... ", ssa, rmslists[ssa]);
		for (sa=0; sa < npos; sa++) {
			if (rmslist[sa] == rmslists[ssa]) {
				printf("found at sa %d, %g == %g\n", sa, rmslists[ssa], rmslist[sa]);
				list[ssa] = sa;
				break;
			}
		}
	}
	//		printf("...: rms for sa %d = %.3g\n", sa, rmslists[sa]);
	return 1;
}

int32_t *_calcshifts_float32(float32_t *image, int32_t stride, int32_t sapos[][2], int nsa, int32_t sasize[2], int32_t sfpos[][2], int nsf, int32_t sfsize[2], int32_t shran[2], int compmeth, int extmeth, int refmode, int refopt, int debug) {
	int refsa, sa, sf, i, j;
	float32_t pix, mean;
	int32_t *refpos;
	float32_t ref[sasize[0]*sasize[1]];				// Reference subap
	float32_t _subimg[sasize[0]*sasize[1]];		// Subap to test
	float32_t *_subfield;											// Pointer to subfield
	int32_t diffsize[] = {(shran[0]*2+1), (shran[1]*2+1)};
	float32_t diffmap[diffsize[0] * diffsize[1]]; // correlation map
	float32_t shvec[2];
	if (debug > 0)
		printf("_calcshifts_float32().\n");

	// First get the reference subapertures
	int reflist[refopt], ret;
	ret = _findrefidx_float32(image, stride, sapos, nsa, sasize, refmode, refopt, reflist, debug);
	
	if (debug > 0)
		printf("...: got reflist, first is: %d.\n", reflist[0]);
	
	for (refsa=0; refsa<refopt; refsa++) {
		refpos = sapos[reflist[refsa]];
		if (debug > 0)
			printf("...: parsing ref %d at (%d,%d).\n", reflist[refsa], refpos[0], refpos[1]);
		// Cut out reference subap, calculate mean
		mean = 0;
		for (j=0; j<sasize[1]; j++) {
			for (i=0; i<sasize[0]; i++) {
				pix = image[(refpos[1] + j) * stride + refpos[0] + i];
				ref[sasize[0] * j + i] = pix;
				mean += pix;
			}
		}
		mean = mean/(sasize[1]*sasize[0]);
		if (debug > 0)
			printf("...: ref mean was: %g.\n", mean);
		// Divide reference subap by mean
		for (j=0; j<sasize[1]; j++)
			for (i=0; i<sasize[0]; i++)
				ref[sasize[0] * j + i] /= mean;

		// Loop over subapertures and subfields
		for (sa=0; sa<nsa; sa++) {
			if (debug > 0)
				printf("...: parsing sa %d at (%d,%d).. ", sa, sapos[sa][0], sapos[sa][1]);
			// Cut out subaperture, calculate mean
			mean = 0;
			for (j=0; j<sasize[1]; j++) {
				for (i=0; i<sasize[0]; i++) {
					pix = image[(sapos[sa][1] + j) * stride + sapos[sa][0] + i];
					_subimg[sasize[0] * j + i] = pix;
					mean += pix;
				}
			}
			mean = mean/(sasize[1]*sasize[0]);
			if (debug > 0)
				printf("mean: %g... ", mean);
			// Divide subap by mean
			for (j=0; j<sasize[1]; j++)
				for (i=0; i<sasize[0]; i++)
					_subimg[sasize[0] * j + i] /= mean;
			
			for (sf=0; sf<nsf; sf++) {
				if (debug > 0)
					printf("sf %d... ", sf);
				// Cut out subfield
				_subfield = _subimg + (sfpos[sf][1] * sasize[0]) + sfpos[sf][0];
				mean = 0.0;
				for (j=0; j<sfsize[1]; j++)
					for (i=0; i<sfsize[0]; i++)
						mean += _subfield[sasize[0] * j + i];
					
				printf("mean: %g... ", mean/(sfsize[1] * sfsize[0]));
				
				// Calculate correlation map
				_sqdiff(_subfield, sfsize, sasize[0], ref, sasize, sasize[0], diffmap, sfpos[sf], shran);
				// Find maximum
				_quadint(diffmap, diffsize, shvec, shran);
				printf("sh: (%.3g, %.3g) ", shvec[0]-shran[0], shvec[1]-shran[1]);
			}
			printf("\n");
		}
	}
	
	return NULL;
}

int _quadint(float32_t *diffmap, int32_t diffsize[2], float32_t shvec[2], int32_t shran[2]) {
	// Find maximum
	float32_t max = diffmap[0], pix;
	int32_t maxidx[] = {0,0};
	int i, j;
	for (j=0; j<diffsize[1]; j++) {
		for (i=0; i<diffsize[0]; i++) {
			pix = diffmap[j * diffsize[0] + i];
			if (pix > max) {
				max = pix;
				maxidx[0] = i;
				maxidx[1] = j;
			}
		}
	}
	printf("max: %g (%d,%d) ", max, maxidx[0], maxidx[1]);
	if (maxidx[0] == 0 || maxidx[0] == diffsize[0]-1 ||
		maxidx[1] == 0 || maxidx[1] == diffsize[1]-1) {
			// Out of bound, interpolation failed
			shvec[0] = maxidx[0];
			shvec[1] = maxidx[1];
			return 0;
	}
	// Now interpolate around the maximum	
	// float32_t a2 = 0.5 * (diffmap[(maxidx[1] + 1) * diffsize[0] + maxidx[0]] - \
	// 	diffmap[(maxidx[1] - 1) * diffsize[0] + maxidx[0]]);
	// float32_t a3 = 0.5 * diffmap[(maxidx[1] + 1) * diffsize[0] + maxidx[0]] - \
	// 	diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]] + \
	// 	0.5 * diffmap[(maxidx[1] - 1) * diffsize[0] + maxidx[0]];
	// float32_t a4 = 0.5 * (diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]+1] -
	// 	diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]-1]);
	// float32_t a5 = 0.5 * diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]+1] -
	// 	diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]] + \
	// 	0.5 * diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]-1];
	// float32_t a6 = 0.25 * (diffmap[(maxidx[1]+1) * diffsize[0] + maxidx[0]+1] -\
	// 	diffmap[(maxidx[1]+1) * diffsize[0] + maxidx[0]-1] - \
	// 	diffmap[(maxidx[1]-1) * diffsize[0] + maxidx[0]+1] + \
	// 	diffmap[(maxidx[1]-1) * diffsize[0] + maxidx[0]-1]);
	float32_t a2 = 0.5 * (diffmap[(maxidx[1]) * diffsize[0] + maxidx[0] + 1] - \
		diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]-1]);
	float32_t a3 = 0.5 * diffmap[(maxidx[1]) * diffsize[0] + maxidx[0] + 1] - \
		diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]] + \
		0.5 * diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]-1];
	float32_t a4 = 0.5 * (diffmap[(maxidx[1] + 1) * diffsize[0] + maxidx[0]] -
		diffmap[(maxidx[1] - 1) * diffsize[0] + maxidx[0]]);
	float32_t a5 = 0.5 * diffmap[(maxidx[1]+1) * diffsize[0] + maxidx[0]] -
		diffmap[(maxidx[1]) * diffsize[0] + maxidx[0]] + \
		0.5 * diffmap[(maxidx[1]-1) * diffsize[0] + maxidx[0]];
	float32_t a6 = 0.25 * (diffmap[(maxidx[1]+1) * diffsize[0] + maxidx[0]+1] -\
		diffmap[(maxidx[1]-1) * diffsize[0] + maxidx[0]+1] - \
		diffmap[(maxidx[1]+1) * diffsize[0] + maxidx[0]-1] + \
		diffmap[(maxidx[1]-1) * diffsize[0] + maxidx[0]-1]);
	
	shvec[0] = maxidx[0] + (2.0*a2*a5-a4*a6)/(a6*a6-4.0*a3*a5);
	shvec[1] = maxidx[1] + (2.0*a3*a4-a2*a6)/(a6*a6-4.0*a3*a5);
	printf("_quadint(): a2 %g a3 %g a4 %g a5 %g a6 %g\n", a2, a3, a4, a5, a6);
	return 1;
}

int _sqdiff(float32_t *img, int32_t imgsize[2], int32_t imstride, float32_t *ref, int32_t refsize[2], int32_t refstride, float32_t *diffmap, int32_t pos[2], int32_t range[2]) {
	double tmpsum, diff;
	// Loop ranges
	int sh0min = -range[0] + pos[0];
	int sh0max =  range[0] + pos[0];
	int sh1min = -range[1] + pos[1];
	int sh1max =  range[1] + pos[1];
	
	// If sum(pos) is not zero, we expect that ref is large enough to naively 
	// shift img around.
	int sh0, sh1, i, j;
	if (pos[0]+pos[1] != 0) {
		// Loop over all shifts to be tested
		for (sh0=sh0min; sh0 <= sh0max; sh0++) {
			for (sh1=sh1min; sh1 <= sh1max; sh1++) {
				// Loop over all pixels within the img and refimg, and compute
				// the cross correlation between the two.
				tmpsum = 0.0;
				for (j=0; j<imgsize[1]; j++) {
					for (i=0; i<imgsize[0]; i++) {
						// First get the difference...
						diff = img[j*imstride + i] - ref[(j+sh0)*refstride + i+sh1];
						// ...then square this
						tmpsum += diff*diff;
					}
				}
				// Store the current correlation value in the map. Use
				// negative value to ensure that we get a maximum for best
				// match in diffmap (this allows to use a more general 
				// maximum-finding method, instead of splitting between maxima
				// and minima)
				diffmap[(sh1-sh1min) * (range[0]*2+1) + (sh0-sh0min)] = -tmpsum;
			}
		}
	}
	// If sum(pos) is zero, we use clipping of ref and img to only use the 
	// intersection of the two datasets for comparison, using normalisation to 
	// make the results consistent.
	else {
		for (sh0=sh0min; sh0 <= sh0max; sh0++) {
			for (sh1=sh1min; sh1 <= sh1max; sh1++) {
				tmpsum = 0.0;
				// If the shift to compare is negative, we must make sure 
				// i+sh0 in 'ref' will not be negative, so we start i at -sh0.
				// If the shift is positive, we must make sure that i+sh0 in 
				// 'ref' will not go out of bound, so we stop i at Nimg[0] - 
				// sh0.
				// N<array>[<index>] gives the <index>th size of <array>, i.e. 
				// Nimg[1] gives the second dimension of array 'img'
				for (j=0 - min(sh0, 0); j<imgsize[1] - max(sh0, 0); j++){
					for (i=0 - min(sh1, 0); i<imgsize[0] - max(sh1, 0); i++) {
						// First get the difference...
						diff = img[j*imstride + i] - ref[(j+sh0)*refstride + i+sh1];
						//diff = img(i,j) - ref(i+sh1,j+sh0);
						// ...then square this
						tmpsum += diff*diff;
					}
				}
				// Scale the value found by dividing it by the number of 
				// pixels we compared.
				diffmap[(sh1-sh1min) * (range[0]*2+1) +(sh0-sh0min)] = -tmpsum /
					((imgsize[0]-abs(sh0)) * (imgsize[1]-abs(sh1)));
			}
		}
	}
	
	return pos[0]+pos[1];
}
