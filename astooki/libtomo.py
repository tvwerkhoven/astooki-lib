#!/usr/bin/env python
# encoding: utf-8
"""
@file libtomo.py
@brief Tomographic wide-field wavefront sensor data analysis tools
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090512

This library provides some routines for tomographically analyzing/processing wide-field Shack-Hartmann wavefront sensor data.

Created by Tim on 2009-05-12.
Copyright (c) 2009 Tim van Werkhoven. All rights reserved.
"""

#=============================================================================
# Import libraries here
#=============================================================================

import liblog as log		# To print & log messages
from libfile import *		# File storing, management and other things
import numpy as N				# Math & calculations
import os
import scipy as S
import scipy.weave				# For inlining C

#=============================================================================
# Some static defines
#=============================================================================
# Compilation flags
__COMPILE_OPTS = "-O3 -ffast-math -msse -msse2"

#=============================================================================
# Routines 
#=============================================================================

def computeFwdMatrix(lh, lsize, lorig, lcells, sasize, sapos, sfang, sffov, matroot='./matrices/', engine='c'):
	"""
	Calculate a model forward matrix giving the WFWFS output for a given
	atmosphere geometry.
	
	The model is constructed as follows: for each subaperture and each subfield, 
	trace a cone through the atmosphere. At each atmospheric layer, calculate 
	the size and position of the meta-pupil of the cone. Using this information, 
	calculate where the cone intersects the atmospheric layer. Since each layer 
	consists of a finite amount of cells, the intersection can be expressed as 
	the fractional area intersected for each cell. These intersections are 
	stored in the matrix such that when multiplying it with a model atmosphere, 
	each subaperture-subfield pair gets a linear sum of the intersected cells in 
	the different atmospheric layers as measurement.
	
	@param lh Height for each layer (km)
	@param lsize Size for each layer (m)
	@param lorig Origin for each layer (m)
	@param lcells Number of cells for each layer
	@param sasize Subaperture size (m)
	@param sapos Subaperture positions (m)
	@param sfang Subfield pointing angle (radian)
	@param sffov Subfield field of view (radian)
	@param engine Code to use for matrix computation, 'c' or 'py' ('c')
	"""
	
	# Store all relevant parameters for this matrix in matconf:
	matconf = {'lh': lh,
		'lsize': lsize,
		'lorig': lorig,
		'lcells': lcells,
		'sasize': sasize,
		'sapos': sapos,
		'sfang': sfang,
		'sffov': sffov}
	# Calculate unique md5sum for this configuration
	mattag = calcTag(matconf)
	# See if this matrix is already stored on disk using the unique tag
	matdir = matroot + mattag
	matfile = 'fwdmatrix'
	# If the file exists, load it from disk and return
	if os.path.exists(os.path.join(matdir, matfile)+'.npy'):
		log.prNot(log.NOTICE, "computeFwdMatrix(): Matrix cached on disk, restoring.")
		fwdmatrix = loadData(os.path.join(matdir, matfile)+'.npy', asnpy=True)
		return (fwdmatrix, mattag)
	
	log.prNot(log.NOTICE, "computeFwdMatrix(): Calculating forward matrix '%s'" % (mattag))
	# Number of subaps, subfields, layers
	nsa = len(sapos)
	nsf = len(sfang)
	nl = len(lh)
	log.prNot(log.INFO, "computeFwdMatrix(): Got %d subaps, %d subfields and %d layers." % (nsa, nsf, nl))
	
	# Matrix width (size of input vector, the atmosphere):
	n = N.product(lcells) * nl
	# Matrix height (size of output vector, the wfwfs data):
	m = nsa * nsf
	fwdmatrix = N.zeros((m, n), dtype=N.float32)
	
	if (engine == 'py'):
		
		# Base positions for different fields of view:
		basecbl = (N.tan(sfang - sffov/2.) * lh.reshape(-1,1,1) - sasize/2. - \
			lorig.reshape(-1,1,2) + lsize.reshape(-1,1,2)) * \
			lcells.reshape(1,1,2) / (lsize.reshape(-1,1,2)*2)
		basecur = (N.tan(sfang + sffov/2.) * lh.reshape(-1,1,1) + sasize/2. - \
			lorig.reshape(-1,1,2) + lsize.reshape(-1,1,2)) * \
			lcells.reshape(1,1,2) / (lsize.reshape(-1,1,2)*2)
	
		# Loop over all subapertures:
		for sa, csapos in zip(range(nsa), sapos):
			# Offset base positions for this subap:
			subcbl = basecbl + (csapos.reshape(1,1,-1) * lcells.reshape(1,1,2) / \
			 	(lsize.reshape(-1,1,2)*2))
			subcur = basecur + (csapos.reshape(1,1,-1) * lcells.reshape(1,1,2) / \
			 	(lsize.reshape(-1,1,2)*2))
			# Make sure all coordinates are within the range [0,0] -- lcells
			blFix = len(subcbl[subcbl < lcells*0])
			urFix = len(subcur[subcur > lcells])
			if (blFix + urFix > 0):
				log.prNot(log.WARNING, "computeFwdMatrix(): Fixing %d and %d coordinates @ sa %d." % (blFix, urFix, sa))
		
			# subcbl[subcbl[:,:,0] > lcells[0], 0] = lcells[0]
			# subcbl[subcbl[:,:,1] > lcells[1], 1] = lcells[1]
			subcbl[subcbl[:,:,0] < 0, 0] = 0
			subcbl[subcbl[:,:,1] < 0, 1] = 0
			subcur[subcur[:,:,0] > lcells[0], 0] = lcells[0]
			subcur[subcur[:,:,1] > lcells[1], 1] = lcells[1]
			# Loop over all subfields:
			for sf, csfang in zip(range(nsf), sfang):
				#csfang = sfang[sf]
				# Offset in the model matrix
				woff = sa * nsf + sf
				# Loop over all atmosphere layers:
				for lay in range(nl):
					# Investigate the which atmospheric cells in layer 'lay'
					# influence subfield 'sf' in subaperture 'sa':
				
					# Old:
					#cpos2 = sasfBounds(csapos, csfang, sasize, sffov, \
					# 	lh[lay], units='cell', rtype='bounds', lorig=lorig[lay], \
					# 	lsize=lsize[lay], lcells=lcells)
					#xr = cpos[:, 0]
					#yr = cpos[:, 1]
				
					xr0 = subcbl[lay,sf,0]
					xr1 = subcur[lay,sf,0]
					yr0 = subcbl[lay,sf,1]
					yr1 = subcur[lay,sf,1]
					off = (lay * lcells[0] * lcells[1])
				
					# Old
					# Calculate the coordinates of the cells intersected and the
					# amount of intersection
					#(isectc, isect) = calcIsect(xr, yr, lcells[0])
				
					# Flatten & insert this in the full matrix
					#fwdmatrix[sa * nsf + sf, \
					#	(isectc + (lay * product(lcells))).reshape(-1).tolist()]=\
					# 	isect.reshape(-1)
					#print int(xr0), int(N.ceil(xr1)), int(yr0), int(N.ceil(yr1))
					for cx in xrange(int(xr0), int(N.ceil(xr1))):
						for cy in xrange(int(yr0), int(N.ceil(yr1))):
							isect = (min(cx+1, xr1) - max(cx, xr0)) * \
								(min(cy+1, yr1) - max(cy, yr0))
							fwdmatrix[woff, off + cy * lcells[0] + cx] = isect
	elif (engine == 'c'):
		code = """
		#line 177 "libtomo.py"
		#ifndef max
		#define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
		#endif
	
		#ifndef min
		#define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
		#endif
	
		int sa, sf, lay;
		int xc, yc, off, woff;
		int cx, cy;
		double xll, yll, xur, yur;
		double xllc, yllc, xurc, yurc;
		double isect;
		int lcellsx = lcells(0);
		int lcellsy = lcells(1);
	
		// Loop over all subapertures
		for (sa=0; sa<nsa; sa++) {
			// Loop over all subfields
			for (sf=0; sf<nsf; sf++) {
				// Loop over all layers
				woff = sa * nsf + sf;
				for (lay=0; lay<nl; lay++) {
					// Calculate intersection of subfield 'sf' in subaperture 'sa' with 
					// atmospheric layer 'lay'
				
					// Lower-left position of the subfield-subaperture pair at this layer:
					xll = tan(sfang(sf, 0) - sffov(0)/2.0) * lh(lay) + sapos(sa, 0) - \\
						sasize(0)/2.0;
					yll = tan(sfang(sf, 1) - sffov(1)/2.0) * lh(lay) + sapos(sa, 1) - \\
						sasize(1)/2.0;
					// Upper-right position:
					xur = tan(sfang(sf, 0) + sffov(0)/2.0) * lh(lay) + sapos(sa, 0) + \\
						sasize(0)/2.0;
					yur = tan(sfang(sf, 1) + sffov(1)/2.0) * lh(lay) + sapos(sa, 1) + \\
						sasize(1)/2.0;
					// Now convert to cell coordinates
					xllc = (xll - lorig(lay, 0) + lsize(lay, 0))/(lsize(lay, 0)*2.0) * \\
					 	lcellsx;
					yllc = (yll - lorig(lay, 1) + lsize(lay, 1))/(lsize(lay, 1)*2.0) * \\
					 	lcellsy;
					xurc = (xur - lorig(lay, 0) + lsize(lay, 0))/(lsize(lay, 0)*2.0) * \\
					 	lcellsx;
					yurc = (yur - lorig(lay, 1) + lsize(lay, 1))/(lsize(lay, 1)*2.0) * \\
					 	lcellsy;
					// Clip ranges to (0,0) -- (lcellsx, lcellsy)
					xllc = max(0, xllc);
					yllc = max(0, yllc);
					xurc = min(lcellsx, xurc);
					yurc = min(lcellsy, yurc);
					// Now we know that our subfield cone intersects the area (xllc, yllc)
					// -- (xurc, yurc) in cell coordinates. Calculate the fractional
					// intersection for each cell in this range.
					off = (lay * (lcellsx) * (lcellsy));
					//printf("off: %d, lsize: %d,%d, sa: %d sf: %d, x: %g--%g, y: %g--%g.\\n", off, lcellsx, lcellsy, sa, sf, xllc, xurc, yllc, yurc);
					for (cx = floor(xllc); cx < ceil(xurc); cx++) {
						for (cy = floor(yllc); cy < ceil(yurc); cy++) {
							isect = (min(cx+1, xurc) - max(cx, xllc)) * (min(cy+1, yurc) - max(cy, yllc));
							fwdmatrix(woff, off + (cy * lcellsx) + cx) = isect;
							//printf("(%d,%d) = %g\\n", woff, off + (cy * lcellsx) + cx, isect);
						}
					}
				}
			}
		}
		"""
		one = S.weave.inline(code, \
			['fwdmatrix', 'nsa', 'nsf', 'nl', 'sfang', 'sffov', 'sapos', 'sasize', 'lh', 'lorig', 'lsize', 'lcells'], \
			extra_compile_args= [__COMPILE_OPTS], \
			type_converters=S.weave.converters.blitz)
	# log.prNot(log.NOTICE, "computeFwdMatrix(): difference between C and Python: sum: %g abs mean: %g allclose: %d" % \
	# 	(N.sum(fwdmatrixC-fwdmatrix), \
	# 	N.mean(abs(fwdmatrixC-fwdmatrix)), \
	# 	N.allclose(fwdmatrix, fwdmatrixC)) )
	
	# Simple sanity checks (elements should be between 0 and 1)
	if N.amax(fwdmatrix) > 1.001: 
		raise ArithmeticError("Maximum in forward matrix greater than 1:", \
		 	amax(fwdmatrix), "at: ", N.where(fwdmatrix == amax(fwdmatrix)))
	if N.amin(fwdmatrix) < 0: 
		raise ArithmeticError("Minimum in forward matrix smaller than 0:", \
			amin(fwdmatrix), "at: ", N.where(fwdmatrix == amin(fwdmatrix)))
	# if not N.allclose(fwdmatrix.sum(1),1):
	# 	raise ArithmeticError("Some rows have sums different than 1.")
	log.prNot(log.INFO, "computeFwdMatrix(): Simple sanity checks passed.")

	# Store the matrix to disk for later use
	try: os.makedirs(matdir)
	except OSError: pass
	except: log.prNot(log.ERR, "computeFwdMatrix(): Error creating directory")
	
	saveData(os.path.join(matdir, matfile), fwdmatrix, asnpy=True)
	saveData(os.path.join(matdir, 'matconf'), matconf, aspickle=True)
	return (fwdmatrix, mattag)


def computeSvd(matrix, mattag=None, matroot='./matrices/', checkSanity=True, eps=0.1):
	"""
	Compute the singular value decomposition of 'matrix' and optionally do
	some sanity checking of the decomposition.
	
	@param matrix Matrix to SVD
	@param mattag Unique tag for 'matrix', can be used to load cache
	@param matroot Directory where to save/load matrix cache
	@param checkSanity Do some sanity checks if set to True
	@param eps Cut-off value for singular values
	
	@return Dict with U, S, 1/S and V^H as values.
	"""
	
	if (mattag is not None):
		matdir = os.path.join(matroot, mattag)
		if os.path.exists(os.path.join(matdir, "fwdmatrix-svd-s.npy")):
			log.prNot(log.NOTICE, "computeSvd(): SVD stored on disk, restoring")
			s = loadData(os.path.join(matdir, "fwdmatrix-svd-s.npy"), asnpy=True)
			s_inv = loadData(os.path.join(matdir, "fwdmatrix-svd-s-inv.npy"), \
			 	asnpy=True)
			u = loadData(os.path.join(matdir, "fwdmatrix-svd-u.npy"), asnpy=True)
			vh = loadData(os.path.join(matdir, "fwdmatrix-svd-vh.npy"), asnpy=True)
			return {'u':u, 
				's':s, 
				's_inv':s_inv, 
				'vh': vh}
	else:
		log.prNot(log.WARNING, "computeSvd(): Please supply the parameter mattag to prevent unnecessary recomputation of the SVD")
	
	# Perform the decomposition, do not use full_matrices, this takes up 
	# *a lot* of memory in certain very non-square matrices
	(u,s,vh) = N.linalg.svd(matrix, full_matrices=False)
	
	# Exclude (potentially) bad singular values when inverting them
	goodVals = N.where(s > eps)
	s_inv = N.zeros(len(s))
	s_inv[goodVals] = 1.0/s[goodVals]
	
	# In numpy:
	#   matrix == dot(u, dot(identity(len(s)) * s, vh))
	#   matrix^-1 == dot(v.T, dot(si, u.T))
	
	if checkSanity:
		# Calculate si = diag(1/s)
		sd = N.identity(len(s)) * s
		si = N.identity(len(s)) * s_inv
		
		log.prNot(log.INFO, "computeSvd(): SVD shapes: u: (%dx%d) s: (%d,%d) vh: (%d,%d)." % (u.shape + si.shape + vh.shape))
		if N.allclose(matrix, N.dot(u, N.dot(sd, vh))):
			log.prNot(log.NOTICE, "computeSvd(): Reconstruction seems to have worked")
		else:
			log.prNot(log.WARNING, "computeSvd(): Reconstruction did not work, got inaccurate SVD.")
		
		# Try to obtain identity matrix through matrix * matrix^-1 using the
		# SVD components to calculate the inverse.
		idresid = N.dot(N.dot(vh.T, N.dot(si, u.T)), matrix) - N.identity(len(s))
		log.prNot(log.NOTICE, "computeSvd(): Reconstruction residual, sum: %0.4g avg: %0.4g +- %0.4g" % (idresid.sum(), idresid.mean(), idresid.std()))
		
	if (mattag is not None):
		# Save the SVD components to disk
		matdir = os.path.join(matroot, mattag)
		try: os.makedirs(matdir)
		except OSError: pass
		except: log.prNot(log.ERR, "computeSvd(): Error creating directory '%s'" % (matdir))
		
		saveData(os.path.join(matdir, "fwdmatrix-svd-s"), s, asnpy=True)
		saveData(os.path.join(matdir, "fwdmatrix-svd-s-inv"), s_inv, asnpy=True)
		saveData(os.path.join(matdir, "fwdmatrix-svd-u"), u, asnpy=True)
		saveData(os.path.join(matdir, "fwdmatrix-svd-vh"), vh, asnpy=True)
	else:
		log.prNot(log.NOTICE, "computeSvd(): Please supply the parameter mattag to computeSvd() to prevent unnecessary recomputation of the SVD.")
		
	# Return SVD components as a dict
	return {'u':u, 
		's':s, 
		's_inv':s_inv, 
		'vh': vh}


def cacheSvd(lhs, lsizes, lorigs, lcells, sasize, sapos, sfang, sffov, matroot='./matrices'):
	"""
	Precompute the reconstruction matrices, which are the singular value 
	decomposed forward matrices.
	
	@param lhs List of layer height configurations
	@param lsizes List of layer sizes
	@param lorigs List of layer origins
	@param lcells List of layer cells
	@param sasize Subaperture size (m)
	@param sapos Subaperture positions (m)
	@param sfang Subfield pointing angle (radian)
	@param sffov Subfield field of view (radian)
	"""
	log.prNot(log.NOTICE, "cacheSvd(): Building up SVD cache.")
	
	# Init memory to hold all SVDs
	svdCache = []
	
	for i in range(len(lhs)):
		log.prNot(log.NOTICE, "cacheSvd(): Computing forward matrix %d/%d" % (i+1, len(lhs)))
		# For each reconstruction geometry, calculate the forwardmatrix:
		(fwdmat, mattag) = computeFwdMatrix(lhs[i], lsizes[i], lorigs[i], \
		 	lcells, sasize, sapos, sfang, sffov, matroot=matroot)
		
		log.prNot(log.NOTICE, "cacheSvd(): Computing SVD for forward matrix...")
		# SVD this forward matrix, and store this
		svdCache.append(computeSvd(fwdmat, mattag=mattag, matroot=matroot))
	
	# Return the SVD cache
	return svdCache


