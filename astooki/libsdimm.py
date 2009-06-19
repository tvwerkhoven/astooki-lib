#!/usr/bin/env python
# encoding: utf-8
"""
libsdimm.py

Created by Tim on 2009-06-08.
Copyright (c) 2009 Tim van Werkhoven. All rights reserved.
"""

import sys
import os
import numpy as N
import astooki.liblog as log

# Compilation flags
__COMPILE_OPTS = "-Wall -O3 -ffast-math -msse -msse2"

def computeSdimmCov(shifts, sapos, sfpos, skipsa=[], row=True, col=False):
	"""
	Compute the SDIMM+ covariance maps which can consequently be used to compute 
	the atmospheric seeing structure using inversion techniques. The methods we
	use here is described in the paper Scharmer & van Werkhoven and is based on 
	DIMM as described by Sarazin and Roddier.
	
	@param [in] shifts The shift measurements for a dataset
	@param [in] sapos The centroid subaperture centroid positions
	@param [in] sfpos The centroid subfield positions
	@param [in] row Use row-wise comparison of subapertures
	@param [in] col Use column-wise comparison of subapertures
	"""
	
	log.prNot(log.ERR, "This function is deprecated, please use computeSdimmCovWeave() intead!")
	
	# Average over number of references
	shifts_a = shifts.mean(axis=1)
	# Take difference if number of references is 2 or more
	if (shifts.shape[1] >= 2):
		shifts_d = (shifts[:,0] - shifts[:,1]) * 0.5
	else:
		shifts_d = None
	
	# These will hold the sdimm correlation values
	sdimm_a = []
		
	### Loop over all *rows*
	### ====================
	if row:
		# Get unique SA row positions
		sarows = N.unique(sapos[:,1])
		# Get unique SF row positions
		sfrows = N.unique(sfpos[:,1])		
		# Loop over all subaperture rows
		for sarowpos in sarows:
			# Get a list of all subapertures at this row (i.e. same y coordinate)
			salist = N.argwhere(sapos[:,1] == sarowpos).flatten()
			# Exclude bad subaps
			salist = N.lib.arraysetops.setdiff1d(salist, skipsa)
			# Take a reference subaperture in this row (the one on the left)
			#refsa = salist[N.argmin(sapos[salist][:,0])]
			# Loop over all subapertures in this row
			for rowsa1 in salist:
				othersa = salist[sapos[salist,0] >= sapos[rowsa1,0]]
				for rowsa2 in othersa:
					#if (rowsa == refsa): continue
					log.prNot(log.NOTICE, "ROW: Comparing subap %d with subap %d." % \
						(rowsa1, rowsa2))
					# Calculate the distance between these two subaps
					s = sapos[rowsa2, 0] - sapos[rowsa1, 0]
					# s = sapos[rowsa, 0] - sapos[refsa, 0]
					# Pre-calculate difference
					dx = shifts_a[:, rowsa1, :, :] - shifts_a[:, rowsa2, :, :]
					dx_d = shifts_d[:, rowsa1, :, :] - shifts_d[:, rowsa2, :, :]
					# Loop over all subfield rows
					for sfrowpos in sfrows:
						# Get a list of all subfields at this row (i.e. same y coordinate)
						sflist = N.argwhere(sfpos[:,1] == sfrowpos).flatten()
						# Take a reference subaperture in this row (the one on the left)
						#rowsf = refsf = sflist[N.argmin(sfpos[sflist][:,0])]
						# Loop over all subfields in this row
						for rowsf1 in sflist:
							othersf = sflist[sfpos[sflist,0] >= \
							 	sfpos[rowsf1,0]]
							# Loop over all other subfields
							for rowsf2 in othersf:
								a = sfpos[rowsf2, 0] - sfpos[rowsf1, 0]
								#dx_s0 = shifts_a[:, rowsa1, rowsf1, :] - \
								#	shifts_a[:, rowsa2, rowsf1, :]
								#dx_s02 = dx[:,rowsf1]
								#dx_sa = shifts_a[:, rowsa1, rowsf2, :] - \
								#	shifts_a[:, rowsa2, rowsf2, :]
								#dx_sa2 = dx[:,rowsf2]
								#print (dx_s02 - dx_s0).sum(), (dx_sa2 - dx_sa).sum()
								C_lsa = (N.cov(dx[:,rowsf1,0], dx[:,rowsf2,0]))[0,1]
								C_tsa = (N.cov(dx[:,rowsf1,1], dx[:,rowsf2,1]))[0,1]
								C_lsa_d = (N.cov(dx_d[:,rowsf1,0], dx_d[:,rowsf2,0]))[0,1]
								C_tsa_d = (N.cov(dx_d[:,rowsf1,1], dx_d[:,rowsf2,1]))[0,1]
								# C_lsa = (N.cov(dx_s0[:,0], dx_sa[:,0]))[0,1]
								# C_tsa = (N.cov(dx_s0[:,1], dx_sa[:,1]))[0,1]
								sdimm_a.append([0, s, a, C_lsa, C_tsa, C_lsa_d, C_tsa_d, \
									rowsa1, rowsa2, rowsf1, rowsf2])
	if col:
		log.prNot(log.WARNING, "Column-wise comparison not implemented.")
	
	return N.array(sdimm_a)


def computeSdimmCovWeave(shifts, sapos, sfpos, skipsa=[], refs=0, row=True, col=False):
	"""
	Compute the SDIMM+ covariance maps which can consequently be used to compute 
	the atmospheric seeing structure using inversion techniques. The methods we
	use here is described in the paper Scharmer & van Werkhoven and is based on 
	DIMM as described by Sarazin and Roddier.
	
	This is the optimized weave version of computeSdimmCov(), which should not 
	be used anymore.
	
	@param [in] shifts The shift measurements for a dataset
	@param [in] sapos The centroid subaperture centroid positions
	@param [in] sfpos The centroid subfield positions
	@param [in] row Use row-wise comparison of subapertures
	@param [in] col Use column-wise comparison of subapertures
	"""
	
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
	
	# Average over number of references
	log.prNot(log.INFO, "Averaging shift over references...")
	shifts_a = shifts.mean(axis=1)
	
	# Take difference if number of references is 2 or more
	# if (shifts.shape[1] >= 2):
	# 	log.prNot(log.INFO, "Calculating difference between references...")
	# 	#shifts_d = (shifts[:,0] - shifts[:,1]) * 0.5
	# else:
	# 	#shifts_d = None
		
	# Get the different values of s and a we have to work on:
	slist = getDist(sapos, skip=skipsa, row=row, col=col)
	alist = getDist(sfpos, skip=[], row=row, col=col)
	# Get unique values
	slist = N.unique(N.round(slist, 7))
	alist = N.unique(alist)
	log.prNot(log.INFO, "Got s values: %s" % str(slist))
	log.prNot(log.INFO, "Got a values: %s" % str(alist))
	
	# Allocate memory (mean shift (2), error analysis (2), ref 0 (2)
	# multiplicity (1))
	sd_rc = N.zeros((2+2+2+1, len(slist), len(alist)))
	
	if row:
		sarows = N.unique(sapos[:,1])
		sfrows = N.unique(sfpos[:,1])
		# Loop over all subaperture rows
		for sarowpos in sarows:
			# Get a list of all subapertures at this row (i.e. same y coordinate)
			salist = N.argwhere(sapos[:,1] == sarowpos).flatten()
			# Exclude bad subaps
			salist = N.lib.arraysetops.setdiff1d(salist, skipsa)
			# Take a reference subaperture in this row (the one on the left)
			#refsa = salist[N.argmin(sapos[salist][:,0])]
			# Loop over all subapertures in this row
			for rowsa1 in salist:
				othersa = salist[sapos[salist,0] >= sapos[rowsa1,0]]
				for rowsa2 in othersa:
					#if (rowsa == refsa): continue
					# Calculate the distance between these two subaps
					# FIXME: Need to round off 's' values because we get numerical errors
					s = N.round(sapos[rowsa2, 0] - sapos[rowsa1, 0], 7)
					sidx = int(N.argwhere(slist == s).flatten())
					# Pre-calculate difference between subapertures
					dx_a = shifts_a[:, rowsa1, :, :] - shifts_a[:, rowsa2, :, :]
					#dx_d = shifts_d[:, rowsa1, :, :] - shifts_d[:, rowsa2, :, :]
					dx_r = shifts[:, :, rowsa1, :, :] - shifts[:, :, rowsa2, :, :]
					log.prNot(log.NOTICE, "ROW: Comparing subap %d with subap %d." % \
						(rowsa1, rowsa2))
					# Loop over all subfield rows (do this in C)
					code = """
					#line 177 "libsdimm.py"
					#define NQUANT 3
					#define NCOV (NQUANT*2)
					int sfrow, rowsf2, rowsf1, fr, aidx, i, r;
					double a;
					struct covar {
						float p;			// E(x * y)
						float x;			// E(x)
						float y;			// E(y)
						float c;			// E(x * y) - E(x) * E(y) = Cov(x,y)
					};
					struct covar cov[NCOV];
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
								
								// Set variables to zero
								for (i=0; i<NCOV; i++)
									cov[i].p = cov[i].x = cov[i].y = cov[i].c = 0.0;
								
								// Loop over all frames to calculate the mean of various 
								// quantities, then calculate the cov. from this
								for (fr=0; fr<Ndx_a[0]; fr++) {
									// SDIMM COVARIANCE OF DATA: ///////////////////////////////
									// Longitidinal covariance for mean over references
									cov[0].p += dx_a(fr,rowsf1,0) * dx_a(fr,rowsf2,0);
									cov[0].x += dx_a(fr,rowsf1,0);
									cov[0].y += dx_a(fr,rowsf2,0);
									
									// Transversal covariance for mean over references
									cov[1].p += dx_a(fr,rowsf1,1) * dx_a(fr,rowsf2,1);
									cov[1].x += dx_a(fr,rowsf1,1);
									cov[1].y += dx_a(fr,rowsf2,1);
									
									// NOISE PROPAGATION ANALYSIS: /////////////////////////////
									for (r=0; r<Ndx_r[1]; r++)	{
										// Longitidinal covariance for *difference* over refs
										cov[2].p += (dx_a(fr,rowsf1,0) - dx_r(fr,r,rowsf1,0)) * \\
											(dx_a(fr,rowsf2,0) - dx_r(fr,r,rowsf2,0));
										cov[2].x += dx_a(fr,rowsf1,0) - dx_r(fr,r,rowsf1,0);
										cov[2].y += dx_a(fr,rowsf2,0) - dx_r(fr,r,rowsf2,0);
									
										// Transversal covariance for *difference* over refs
										cov[3].p += (dx_a(fr,rowsf1,1) - dx_r(fr,r,rowsf1,1)) * \\
											(dx_a(fr,rowsf2,1) - dx_r(fr,r,rowsf2,1));
										cov[3].x += dx_a(fr,rowsf1,1) - dx_r(fr,r,rowsf1,1);
										cov[3].y += dx_a(fr,rowsf2,1) - dx_r(fr,r,rowsf2,1);
									}
									
									// Covariance for first single reference
									cov[4].p += dx_r(fr,0,rowsf1,0) * dx_r(fr,0,rowsf2,0);
									cov[4].x += dx_r(fr,0,rowsf1,0);
									cov[4].y += dx_r(fr,0,rowsf2,0);
									
							    cov[5].p += dx_r(fr,0,rowsf1,1) * dx_r(fr,0,rowsf2,1);
							    cov[5].x += dx_r(fr,0,rowsf1,1);
							    cov[5].y += dx_r(fr,0,rowsf2,1);									
								}
								
								// Normalize noise propagation
								cov[2].p /= Ndx_r[1]; 
								cov[2].x /= Ndx_r[1]; 
								cov[2].y /= Ndx_r[1];
								cov[3].p /= Ndx_r[1]; 
								cov[3].x /= Ndx_r[1]; 
								cov[3].y /= Ndx_r[1];
								
								// Normalize values and calculate covariance
								for (i=0; i<NCOV; i++) {
									cov[i].p /= Ndx_a[0]-1;
									cov[i].x /= Ndx_a[0]-1;
									cov[i].y /= Ndx_a[0]-1; 
									cov[i].c = cov[i].p - (cov[i].x * cov[i].y);
									sd_rc(i, sidx, aidx) += cov[i].c;
								}
								
								// Increase multiplicity for this (s, a) pair by one
								sd_rc(NCOV, sidx, aidx) += 1;
							}
						}
					}
										
					return_val = 1;
					"""
					one = S.weave.inline(code, \
						['sd_rc', 'sidx', 'sfrows', 'sfpos', 'alist', 'dx_a', 'dx_r'], \
						extra_compile_args= [__COMPILE_OPTS], \
						type_converters=S.weave.converters.blitz)
		# Normalize the covariance map
		sd_rc[0:-1] /= sd_rc[-1].reshape(1,sd_rc.shape[1],sd_rc.shape[2])
	
	if col:
		log.prNot(log.WARNING, "Column-wise comparison not yet implemented.")
	
	return (slist, alist, sd_rc)


def computeSdimmCovWeave2(shfiles, sapos, sfpos, path='./', skipsa=[], row=True, col=False):
	"""
	Compute the SDIMM+ covariance maps which can consequently be used to compute 
	the atmospheric seeing structure using inversion techniques. The methods we
	use here is described in the paper Scharmer & van Werkhoven and is based on 
	DIMM as described by Sarazin and Roddier.
	
	This is the optimized weave version of computeSdimmCov(), and reads in the 
	shifts per file instead of as one big file.
	
	@param [in] shfiles Array of filenames to read and process
	@param [in] sapos The centroid subaperture centroid positions
	@param [in] sfpos The centroid subfield positions
	@param [in] path Path for 'shfiles' if not the current working dir
	@param [in] skipsa List of subapertures to skip
	@param [in] row Use row-wise comparison of subapertures
	@param [in] col Use column-wise comparison of subapertures
	"""
	
	import scipy as S
	import scipy.weave				# For inlining C
	
	# Get the different values of s and a we have to work on:
	slist = getDist(sapos, skip=skipsa, row=row, col=col)
	alist = getDist(sfpos, skip=[], row=row, col=col)
	
	# Get unique values
	slist = N.unique(N.round(slist, 7))
	alist = N.unique(alist)
	log.prNot(log.INFO, "Got s values: %s" % str(slist))
	log.prNot(log.INFO, "Got a values: %s" % str(alist))
	
	# Allocate memory for covariance map
	# (mean shift (2), error analysis (2+2), individual references (4)
	# multiplicity (1))
	sd_rc = N.zeros((2+2+2+4+1, len(slist), len(alist), 3))
	
	# Subap/subfield row positions
	sarows = N.unique(sapos[:,1])
	sarowlist = []
	for sarowpos in sarows:
		# Get a list of all subapertures at this row (i.e. same y coordinate)
		salist = N.argwhere(sapos[:,1] == sarowpos).flatten()
		# Exclude bad subaps
		salist = N.lib.arraysetops.setdiff1d(salist, skipsa)
		sarowlist.append([salist[0], salist[-1]])
	sarowlist = N.array(sarowlist)
	sfrows = N.unique(sfpos[:,1])
	
	for f in shfiles:
		# Load shifts from file
		log.prNot(log.INFO, "Processing file %s now." % (f))
		sh = N.load(f)
		print sarows.shape, sfrows.shape
		# Loop over:
		# - All rows of subaps
		# - All subap-pairs in each row
		# - All rows of subfields
		# - All subfield-pairs in each row
		
		# # Get unique SA row positions
		# sarows = N.unique(sapos[:,1])
		# # Get unique SF row positions
		# sfrows = N.unique(sfpos[:,1])		
		# # Loop over all subaperture rows
		# for sarowpos in sarows:
		# 	# Get a list of all subapertures at this row (i.e. same y coordinate)
		# 	salist = N.argwhere(sapos[:,1] == sarowpos).flatten()
		# 	# Exclude bad subaps
		# 	salist = N.lib.arraysetops.setdiff1d(salist, skipsa)
		# 	# Take a reference subaperture in this row (the one on the left)
		# 	#refsa = salist[N.argmin(sapos[salist][:,0])]
		# 	# Loop over all subapertures in this row
		# 	for rowsa1 in salist:
		# 		othersa = salist[sapos[salist,0] >= sapos[rowsa1,0]]
		# 		for rowsa2 in othersa:
		# 			#if (rowsa == refsa): continue
		# 			log.prNot(log.NOTICE, "ROW: Comparing subap %d with subap %d." % \
		# 				(rowsa1, rowsa2))
		# 			# Calculate the distance between these two subaps
		# 			s = sapos[rowsa2, 0] - sapos[rowsa1, 0]
		# 			# s = sapos[rowsa, 0] - sapos[refsa, 0]
		# 			# Pre-calculate difference
		# 			# Loop over all subfield rows
		# 			for sfrowpos in sfrows:
		# 				# Get a list of all subfields at this row (i.e. same y coordinate)
		# 				sflist = N.argwhere(sfpos[:,1] == sfrowpos).flatten()
		# 				# Take a reference subaperture in this row (the one on the left)
		# 				#rowsf = refsf = sflist[N.argmin(sfpos[sflist][:,0])]
		# 				# Loop over all subfields in this row
		# 				for rowsf1 in sflist:
		# 					othersf = sflist[sfpos[sflist,0] >= \
		# 					 	sfpos[rowsf1,0]]
		# 					# Loop over all other subfields
		# 					for rowsf2 in othersf:
		# 						a = sfpos[rowsf2, 0] - sfpos[rowsf1, 0]
		# 						#dx_s0 = shifts_a[:, rowsa1, rowsf1, :] - \
		# 						#	shifts_a[:, rowsa2, rowsf1, :]
		# 						#dx_s02 = dx[:,rowsf1]
		# 						#dx_sa = shifts_a[:, rowsa1, rowsf2, :] - \
		# 						#	shifts_a[:, rowsa2, rowsf2, :]
		# 						#dx_sa2 = dx[:,rowsf2]
		# 						#print (dx_s02 - dx_s0).sum(), (dx_sa2 - dx_sa).sum()
		# 						# C_lsa = (N.cov(dx_s0[:,0], dx_sa[:,0]))[0,1]
		# 						# C_tsa = (N.cov(dx_s0[:,1], dx_sa[:,1]))[0,1]
		
		code = """
		#line 354 "libsdimm.py"
		int sarow, rowsa1, rowsa2, sfrow, rowsf1, rowsf2;
		int aidx=0, sidx=0, ntot=0;
		double a, s;
		
		//printf("diagnostics: a=%d, s=%d\\n", Nalist[0], Nslist[0]);
		// SUBAPERTURE LOOP //////////////////////////////////////////////////////
			for (rowsa1=0; rowsa1 < Nsapos[0]; rowsa1++) {
				for (rowsa2=rowsa1; rowsa2 < Nsapos[0]; rowsa2++) {
					if (sapos(rowsa2, 1) != sapos(rowsa1, 1)) break;
					//if (sapos(rowsa2, 0) < sapos(rowsa1, 0)) continue;
					
					s = roundf((sapos(rowsa2, 0) - sapos(rowsa1, 0)) * \\
						10000000)/10000000;
					// Find index where to put this covariance value
					for (sidx=0; sidx < Nslist[0]; sidx++)
						if (slist(sidx) == s) break;
					
					//for (; ; ) {
					//	if (sidx >= Nslist[0]) sidx=0;
					//	if (slist(sidx) == s) break;
					//	sidx++;
					//}					
					for (rowsf1=0; rowsf1 < Nsfpos[0]; rowsf1++) {
						for (rowsf2=rowsf1; rowsf2 < Nsfpos[0]; rowsf2++) {
							if (sfpos(rowsf2, 1) != sfpos(rowsf1, 1)) continue;
							if (sfpos(rowsf2, 0) < sfpos(rowsf1, 0)) continue;
							
							a = sfpos(rowsf2, 0) - sfpos(rowsf1, 0);
							for (; ; ) {
								if (aidx >= Nalist[0]) aidx=0;
								if (alist(aidx) == a) break;
								aidx++;
							}
							//fprintf(stderr, "%d/%d ", aidx, Nalist[0]);
								
								//exit(-1);
								
								if (alist(aidx) != a) printf("FAIL\\n");
								
								ntot++;
								// Now calculate cov map
								// Longitidinal covariance for references 0
								//sd_rc(0, sidx, aidx, 1) += sh(0, rowsa1, rowsf1, 0) - \\
								//	sh(0, rowsa2, rowsf1, 0);
								//sd_rc(0, sidx, aidx, 2) += sh(0, rowsa1, rowsf2, 0) - \\
								//	sh(0, rowsa2, rowsf2, 0);
								//sd_rc(0, sidx, aidx, 0) += sd_rc(0, sidx, aidx, 1) * \\
								//	sd_rc(0, sidx, aidx, 2);
								//
								//// Transversal covariance for references 0
								//sd_rc(1, sidx, aidx, 1) += sh(0, rowsa1, rowsf1, 1) - \\
								//	sh(0, rowsa2, rowsf1, 1);
								//sd_rc(1, sidx, aidx, 2) += sh(0, rowsa1, rowsf2, 1) - \\
								//	sh(0, rowsa2, rowsf2, 1);
								//sd_rc(1, sidx, aidx, 0) += sd_rc(1, sidx, aidx, 1) * \\
								//	sd_rc(1, sidx, aidx, 2);
							}
						}
					//} // SUBFIELD LOOP
				}
			}
		//} // SUBAPERTURE LOOP
		
		printf("tot=%d\\n", ntot);
		"""
		one = S.weave.inline(code, \
			['sd_rc', 'sh', 'sarows', 'sarowlist', 'sapos', 'sfrows', 'sfpos', 'alist', 'slist'], \
			extra_compile_args= [__COMPILE_OPTS], \
			type_converters=S.weave.converters.blitz)
	# End loop over files, normalize
	sd_rc /= len(shfiles)
	
	return (slist, alist, sd_rc)


def getDist(pos, skip=[], row=False, col=False):
	"""
	Calculate unique distances between positions (typically subapertures and 
	subfields positions) for row-wise and column-wise comparison.
	
	@param [in] pos The positions to process
	@param [in] skip Optional list of positions (subaps) to skip
	@param [in] row Give row-wise unique distances
	@param [in] col Give column-wise unique distances
	@return List of distances (not unique)
	"""
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
				otherpos = poslist[pos[poslist,0] >= pos[colpos1,0]]
				for colpos2 in otherpos:
					# Calculate the distance between these two positions
					dlist.append(pos[colpos2, 0] - pos[colpos1, 0])
	
	return dlist

