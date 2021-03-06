#!/usr/bin/env python
# encoding: utf-8
"""
This is astooki.libsdimm, providing SDIMM+ analysis functions

This module takes subimage shifts as input and calculates SDIMM+ covariance
maps as output. This output can then be decomposed in different SDIMM+ basis
functions to determine the seeing at different altitudes in the atmosphere.
"""

##  @file libsdimm.py
# @author Tim van Werkhoven (tim@astro.su.se)
# @date 20090608
# 
# Created by Tim van Werkhoven on 2009-06-08.
# Copyright (c) 2008-2009 Tim van Werkhoven (tim@astro.su.se)
# 
# This file is licensed under the Creative Commons Attribution-Share Alike
# license versions 3.0 or higher, see
# http://creativecommons.org/licenses/by-sa/3.0/

## @package astooki.libsdimm
# @brief Library for SDIMM+ analysis
# @author Tim van Werkhoven (tim@astro.su.se)
# @date 20090608
#
# This package provides routines for SDIMM+ analysis

import sys
import os
import numpy as N
import astooki.liblog as log

## @brief Compilation flags for scipy.weave() code
__COMPILE_OPTS = "-Wall -O3 -ffast-math -msse -msse2"

## @brief Merge row- and column-covariance maps 
#
# @param covmaps List of covariance maps to combine
# @param multmaps List of multiplicty maps to combine
# @param slists List of s coordinates for each map
# @param alists List of a coordinates for each map
def mergeMaps(covmaps, multmaps, slists, alists):
	
	# First get all s's and a's
	slist = []
	alist = []
	for sl in slists:
		slist.extend(sl)
	for al in alists:
		alist.extend(al)
	
	# Get only unique values
	alist = N.lib.arraysetops.unique1d(alist).flatten()
	slist = N.lib.arraysetops.unique1d(slist).flatten()
	
	log.prNot(log.INFO, "mergeMap(): Found unique s: %s" % str(slist))
	log.prNot(log.INFO, "mergeMap(): Found unique a: %s" % str(alist))
	
	# Make a new map big enough to hold all data
	mult = N.zeros((len(slist), len(alist)))
	covmap = N.zeros((covmaps[0].shape[:-2]) + mult.shape)
	
	# Loop over the maps, insert into the bigger map
	for n in range(len(covmaps)):
		cmap = covmaps[n]
		# Loop over this covmap in s-direction
		for _s in range(cmap.shape[-2]):
			# Find the index for this s in the new covmap:
			sidx = int(N.argwhere(slist == slists[n][_s]).flatten())
			# Loop over this covmap in a-direction
			for _a in range(cmap.shape[-1]):
				# Find the index for this a in the new covmap:
				aidx = int(N.argwhere(alist == alists[n][_a]).flatten())
				covmap[...,sidx,aidx] += cmap[...,_s,_a] * multmaps[n][_s,_a]
				mult[sidx,aidx] += multmaps[n][_s,_a]
	
	# Normalize the map
	covmap /= mult.reshape( (1,)*(covmap.ndim-2) + mult.shape )
	
	return (slist, alist, covmap, mult)


## @brief Compute the SDIMM+ covariance maps
#
# Compute the SDIMM+ covariance maps which can consequently be used to compute 
# the atmospheric seeing structure using inversion techniques. The methods we
# use here is described in the paper Scharmer & van Werkhoven and is based on 
# DIMM as described by Sarazin and Roddier.
# 
# This is the optimized weave version of computeSdimmCov(), which should not 
# be used anymore.
# 
# @param shifts The shift measurements for a dataset
# @param sapos The centroid subaperture centroid positions
# @param sfpos The centroid subfield positions
# @param skipsa List of (bad) subapertures to skip
# @param refs Number of references to use from the shift data (0=max)
# @param row Use row-wise comparison of subapertures
# @param col Use column-wise comparison of subapertures
# 
# @returns returns a tuple (slist, alist, Cxy, mult), where slist is the list
# of s values, alist is the list of a value, Cxy is the 'covariance' map,
# which has dimensions (nfiles, 2*(1+nref)+2+4, len(slist), len(alist)), mult
# is the multiplicity of Cxy and gives the number of pairs of 
# subapertures-subfields for each (s,a)-pair. Quantities stored in Cxy are 
# described in pyatk.py and in the function itself.
def computeSdimmCovWeave(shifts, sapos, sfpos, skipsa=[], refs=0, row=True, col=False):
	
	import scipy as S
	import scipy.weave				# For inlining C
	
	# Data dimensions + interpretation
	nfiles = shifts.shape[0]
	nref = shifts.shape[1]
	nsa = shifts.shape[2]
	nsf = shifts.shape[3]
	
	if (refs > nref):
		log.prNot(log.WARNING, "Data only contains %d references, cannot use the requested %d!" % (nref, refs))
		refs = nref
	if (refs == 0): refs = nref
	
	log.prNot(log.INFO, "Using %d references for SDIMM+ calculations." % (refs))
	shifts = shifts[:,0:refs]
		
	# Get the different values of s and a we have to work on:
	slist = getDist(sapos, skip=skipsa, row=row, col=col)
	alist = getDist(sfpos, skip=[], row=row, col=col)
	# Get unique values
	slist = N.unique(N.round(slist, 7))
	alist = N.unique(alist)
	log.prNot(log.INFO, "Got s values: %s" % str(slist))
	log.prNot(log.INFO, "Got a values: %s" % str(alist))
	log.prNot(log.INFO, "Got shifts.shape: %s" % str(shifts.shape))
	log.prNot(log.INFO, "Got sfpos.shape: %s" % str(sfpos.shape))
	log.prNot(log.INFO, "Got sapos.shape: %s" % str(sapos.shape))
	
	# Allocate memory for Cx,y(s,a). x2 for longitudinal and transversal,
	# x(1+nref)+2+4 for every reference *and* the average over the reference
	# subapertures *and* the error bias maps for the average *and* some 
	# crossterms
	Cxy = N.zeros((nfiles, 2*(1+nref)+2+4, len(slist), len(alist)))
	# Multiplicity map, number of (s,a)-pairs.
	mult = N.zeros((len(slist), len(alist)))
	if row:
		sarows = N.unique(sapos[:,1])
		sfrows = N.unique(sfpos[:,1])
		# Loop over all subaperture rows
		for sarowpos in sarows:
			log.prNot(log.NOTICE, "Processing next row now...")
			# Get a list of all subapertures at this row (i.e. same y coordinate)
			salist = N.argwhere(sapos[:,1] == sarowpos).flatten()
			# Exclude bad subaps
			salist = N.lib.arraysetops.setdiff1d(salist, skipsa)
			# Loop over all subapertures in this row
			for rowsa1 in salist:
				othersa = salist[sapos[salist,0] >= sapos[rowsa1,0]]
				for rowsa2 in othersa:
					# Calculate the distance between these two subaps
					# FIXME: Need to round off 's' values because we get numerical errors
					s = N.round(sapos[rowsa2, 0] - sapos[rowsa1, 0], 7)
					sidx = int(N.argwhere(slist == s).flatten())
					# Pre-calculate shift-difference between subapertures
					dx_r = shifts[:, :, rowsa1, :, :] - shifts[:, :, rowsa2, :, :]
					# Subtract average over all files from dx_r
					dx_r_avg = N.mean(dx_r, axis=0)
					dx_ra = dx_r - dx_r_avg.reshape( (1,) + dx_r_avg.shape )
					# Calculate average over reference subapertures
					dx_a = dx_ra.mean(axis=1)
					
					log.prNot(log.NOTICE, "ROW: sa %d @ (%g,%g) <-> sa %d @ (%g,%g)."% \
						((rowsa1, ) + tuple(sapos[rowsa1]) + \
						(rowsa2, ) + tuple(sapos[rowsa2])))
					# Loop over all subfield rows (do this in C)
					code = """
					#line 177 "libsdimm.py"
					int sfrow, rowsf2, rowsf1, fr, aidx, i, r;
					double a, errl, errt;

					for (sfrow=0; sfrow < Nsfrows[0]; sfrow++) {
						// current row is: sfrow @ sfrows[sfrow];
						for (rowsf1=0; rowsf1 < Nsfpos[0]; rowsf1++) {
							// Current subfield is: rowsf1 @ sfpos[rowsf1, 1]
							// Check if this subfield is in the correct row:
							if (sfpos(rowsf1, 1) != sfrows(sfrow)) continue;
							for (rowsf2=0; rowsf2 < Nsfpos[0]; rowsf2++) {
								// Current subfield is: rowsf2 @ sfpos[rowsf2, 1]
								// Check if this subfield is in the correct row:
								if (sfpos(rowsf2, 1) != sfrows(sfrow)) continue;
								// Check if rowsf2 is located right of rowsf1
								if (sfpos(rowsf2, 0) < sfpos(rowsf1, 0)) continue;
								
								a = sfpos(rowsf2, 0) - sfpos(rowsf1, 0);
								for (aidx=0; aidx < Nalist[0]; aidx++)
									if (alist(aidx) == a) break;
								
								// Loop over all frames to calculate the expectation value of
								// the various quantities.
								for (fr=0; fr<Ndx_ra[0]; fr++) {
									// Longitudinal average
									Cxy(fr, 0, sidx, aidx) += \\
										dx_a(fr,rowsf1,0) * dx_a(fr,rowsf2,0);
									// Transversal average
									Cxy(fr, 1, sidx, aidx) += \\
										dx_a(fr,rowsf1,1) * dx_a(fr,rowsf2,1);
									
									errl = errt = 0.0;
									// Loop over all reference subapertures
									for (r=0; r<Ndx_ra[1]; r++)	{
										// Error bias map (long.)
										errl += (dx_ra(fr,r,rowsf1,0) - dx_a(fr,rowsf1,0)) * \\
												(dx_ra(fr,r,rowsf2,0) - dx_a(fr,rowsf2,0));
										// Error bias map (trans.)
										errt += (dx_ra(fr,r,rowsf1,1) - dx_a(fr,rowsf1,1)) * \\
												(dx_ra(fr,r,rowsf2,1) - dx_a(fr,rowsf2,1));
										
										// Longitidunal 
										Cxy(fr, 8 + 2*r + 0, sidx, aidx) += \\
											dx_ra(fr,r,rowsf1,0) * dx_ra(fr,r,rowsf2,0);
										// Transversal
										Cxy(fr, 8 + 2*r + 1, sidx, aidx) += \\
											dx_ra(fr,r,rowsf1,1) * dx_ra(fr,r,rowsf2,1);
									}
									// Normalize error bias map
									Cxy(fr, 2, sidx, aidx) += errl/Ndx_ra[1];
									Cxy(fr, 3, sidx, aidx) += errt/Ndx_ra[1];
									
									// Cross terms, ref1 with ref2 longitudinal
									Cxy(fr, 4, sidx, aidx) += \\
										dx_ra(fr,0,rowsf1,0) * dx_ra(fr,1,rowsf2,0);
									Cxy(fr, 5, sidx, aidx) += \\
										dx_ra(fr,1,rowsf1,0) * dx_ra(fr,0,rowsf2,0);
									
									// Cross terms, ref1 with ref2 transversal
									Cxy(fr, 6, sidx, aidx) += \\
										dx_ra(fr,0,rowsf1,1) * dx_ra(fr,1,rowsf2,1);
									Cxy(fr, 7, sidx, aidx) += \\
										dx_ra(fr,1,rowsf1,1) * dx_ra(fr,0,rowsf2,1);
								}
								
								// Increase multiplicity for this (s, a) pair by one
								mult(sidx, aidx) += 1;
							}
						}
					}
										
					return_val = 1;
					"""
					one = S.weave.inline(code, \
						['Cxy', 'mult', 'sidx', 'sfrows', 'sfpos', 'alist', \
							'dx_a', 'dx_ra'], \
						extra_compile_args= [__COMPILE_OPTS], \
						type_converters=S.weave.converters.blitz)
	if col:
		sacols = N.unique(sapos[:,0])
		sfcols = N.unique(sfpos[:,0])
		# Loop over all subaperture cols
		for sacolpos in sacols:
			log.prNot(log.NOTICE, "Processing next column now...")
			# Get a list of all subapertures at this col (i.e. same x coordinate)
			salist = N.argwhere(sapos[:,0] == sacolpos).flatten()
			# Exclude bad subaps
			salist = N.lib.arraysetops.setdiff1d(salist, skipsa)
			# Loop over all subapertures in this column
			for colsa1 in salist:
				othersa = salist[sapos[salist,1] >= sapos[colsa1,1]]
				for colsa2 in othersa:
					# Calculate the distance between these two subaps
					# FIXME: Need to round off 's' values because we get numerical
					# errors
					s = N.round(sapos[colsa2, 1] - sapos[colsa1, 1], 7)
					sidx = int(N.argwhere(slist == s).flatten())
					
					# Pre-calculate shift-difference between subapertures
					dx_r = shifts[:, :, colsa1, :, :] - shifts[:, :, colsa2, :, :]
					# Subtract average over all files from dx_r
					dx_r_avg = N.mean(dx_r, axis=0)
					dx_ra = dx_r - dx_r_avg.reshape( (1,) + dx_r_avg.shape )
					# Calculate average over reference subapertures
					dx_a = dx_ra.mean(axis=1)
										
					log.prNot(log.NOTICE, "COL: sa %d @ (%g,%g) <-> sa %d @ (%g,%g)."% \
						((colsa1, ) + tuple(sapos[colsa1]) + \
						(colsa2, ) + tuple(sapos[colsa2])))
					# Loop over all subfield cols (do this in C)
					code = """
					#line 289 "libsdimm.py"
					int sfcol, colsf2, colsf1, fr, aidx, i, r;
					double a, errl, errt;

					for (sfcol=0; sfcol < Nsfcols[0]; sfcol++) {
						// current col is: sfcol @ sfcols[sfcol];
						for (colsf1=0; colsf1 < Nsfpos[0]; colsf1++) {
							// Current subfield is: colsf1 @ sfpos[colsf1, 0]
							// Check if this subfield is in the correct col:
							if (sfpos(colsf1, 0) != sfcols(sfcol)) continue;
							for (colsf2=0; colsf2 < Nsfpos[0]; colsf2++) {
								// Current subfield is: colsf2 @ sfpos[colsf2, 0]
								// Check if this subfield is in the correct col:
								if (sfpos(colsf2, 0) != sfcols(sfcol)) continue;
								// Check if colsf2 is located above of colsf1
								if (sfpos(colsf2, 1) < sfpos(colsf1, 1)) continue;
								
								a = sfpos(colsf2, 1) - sfpos(colsf1, 1);
								for (aidx=0; aidx < Nalist[0]; aidx++)
									if (alist(aidx) == a) break;
								
								// Loop over all frames to calculate the expectation value of
								// the various quantities.
								for (fr=0; fr<Ndx_ra[0]; fr++) {
									// Longitudinal average
									Cxy(fr, 0, sidx, aidx) += \\
										dx_a(fr,colsf1,1) * dx_a(fr,colsf2,1);
									// Transversal average
									Cxy(fr, 1, sidx, aidx) += \\
										dx_a(fr,colsf1,0) * dx_a(fr,colsf2,0);
									
									errl = errt = 0.0;
									// Loop over all reference subapertures
									for (r=0; r<Ndx_ra[1]; r++)	{
										// Error bias map (long.)
										errl += (dx_ra(fr,r,colsf1,1) - dx_a(fr,colsf1,1)) * \\
												(dx_ra(fr,r,colsf2,1) - dx_a(fr,colsf2,1));
										// Error bias map (trans.)
										errt += (dx_ra(fr,r,colsf1,0) - dx_a(fr,colsf1,0)) * \\
												(dx_ra(fr,r,colsf2,0) - dx_a(fr,colsf2,0));
										
										// Longitidunal 
										Cxy(fr, 8 + 2*r + 0, sidx, aidx) += \\
											dx_ra(fr,r,colsf1,1) * dx_ra(fr,r,colsf2,1);
										// Transversal
										Cxy(fr, 8 + 2*r + 1, sidx, aidx) += \\
											dx_ra(fr,r,colsf1,0) * dx_ra(fr,r,colsf2,0);
									}
									// Normalize error bias map
									Cxy(fr, 2, sidx, aidx) += errl/Ndx_ra[1];
									Cxy(fr, 3, sidx, aidx) += errt/Ndx_ra[1];
									
									// Cross terms, ref1 with ref2 longitudinal
									Cxy(fr, 4, sidx, aidx) += \\
										dx_ra(fr,0,colsf1,1) * dx_ra(fr,1,colsf2,1);
									Cxy(fr, 5, sidx, aidx) += \\
										dx_ra(fr,1,colsf1,1) * dx_ra(fr,0,colsf2,1);
									
									// Cross terms, ref1 with ref2 transversal
									Cxy(fr, 6, sidx, aidx) += \\
										dx_ra(fr,0,colsf1,0) * dx_ra(fr,1,colsf2,0);
									Cxy(fr, 7, sidx, aidx) += \\
										dx_ra(fr,1,colsf1,0) * dx_ra(fr,0,colsf2,0);
								}
								
								// Increase multiplicity for this (s, a) pair by one
								mult(sidx, aidx) += 1;
							}
						}
					}
					
					return_val = 1;
					"""
					one = S.weave.inline(code, \
						['Cxy', 'mult', 'sidx', 'sfcols', 'sfpos', 'alist', \
							'dx_a', 'dx_ra'], \
						extra_compile_args= [__COMPILE_OPTS], \
						type_converters=S.weave.converters.blitz)
	
	# Normalize the covariance map
	Cxy /= mult.reshape(1, 1, mult.shape[0], mult.shape[1])
	
	return (slist, alist, Cxy, mult)


## @brief Calculate unique distances between positions
#
# Calculate unique distances between positions (typically subapertures and 
# subfields positions) for row-wise and column-wise comparison.
# 
# @param pos The positions to process
# @param skip Optional list of positions (subaps) to skip
# @param row Give row-wise unique distances
# @param col Give column-wise unique distances
# @return List of distances (not unique)
def getDist(pos, skip=[], row=False, col=False):
	# Store distances here
	dlist = []
	
	# Process positions row-wise 
	if row:
		# Get unique row positions
		rows = N.unique(pos[:,1])
		# Loop over all rows
		for rowpos in rows:
			# Get a list of all subapertures at this row (i.e. same y coordinate)
			poslist = N.argwhere(pos[:,1] == rowpos).flatten()
			# Exclude certain positions
			poslist = N.lib.arraysetops.setdiff1d(poslist, skip)
			# Loop over all positions in this row
			for rowpos1 in poslist:
				otherpos = poslist[pos[poslist,0] >= pos[rowpos1,0]]
				for rowpos2 in otherpos:
					# Calculate the distance between these two positions
					dlist.append(pos[rowpos2, 0] - pos[rowpos1, 0])
	# Same for column-wise, flip indices
	if col:
		# Get unique column positions
		cols = N.unique(pos[:,0])
		# Loop over all columns
		for colpos in cols:
			# Get a list of all subapertures at this column (i.e. same x coordinate)
			poslist = N.argwhere(pos[:,0] == colpos).flatten()
			# Exclude certain positions
			poslist = N.lib.arraysetops.setdiff1d(poslist, skip)
			# Loop over all positions in this column
			for colpos1 in poslist:
				otherpos = poslist[pos[poslist,1] >= pos[colpos1,1]]
				for colpos2 in otherpos:
					# Calculate the distance between these two positions
					dlist.append(pos[colpos2, 1] - pos[colpos1, 1])
	
	return dlist

