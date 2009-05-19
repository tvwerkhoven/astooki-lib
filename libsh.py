#!/usr/bin/env python2.5
# encoding: utf-8
"""
@file libsh.py
@brief Shack-Hartmann routines
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090423

This library provides some routines for analyzing/processing Shack-Hartmann 
wavefront sensor data.

Created by Tim van Werkhoven on 2009-04-23.
Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

#=============================================================================
# Import libraries here
#=============================================================================

import numpy as N				# Math & calculations
import csv
import liblog as log		# To print & log messages
from libfile import *		# File storing, management and other things
import os

#=============================================================================
# Some static defines
#=============================================================================

#=============================================================================
# Routines 
#=============================================================================

def makeSubaptMask(pos, size, res):
	"""
	Make a binary subaperture mask and a map of the subaperture borders. 'pos' 
	should be a list of origin pixel-positions of the subapertures, 'size' 
	should be the pixel-size of all subapertures and 'res' should be a tuple for 
	the ouput pixel-size of the masks.
	
	Returns the tuple (mask, bordermask)
	"""
	
	mask = N.zeros(res,dtype=N.bool)
	maskborder = N.zeros(res,dtype=N.bool)
	for p in pos:
		mask[\
			p[1]:p[1]+size[1], \
			p[0]:p[0]+size[0]] = 1
		maskborder[\
			p[1]-1:p[1]+size[1]+1, \
			p[0]-1:p[0]+size[0]+1] = 1
		maskborder[\
			p[1]:p[1]+size[1], \
			p[0]:p[0]+size[0]] = 0
	
	return (mask, maskborder)


def loadSaSfConf(safile):
	"""
	Try to load 'filename', if it exists. This should hold information on 
	the subaperture or subfield positions and sizes on the CCD. Positions should 
	be the absolute lower-left corner of the subaperture, or relative LL corner 
	of the subfield. Syntax of the file should be:
	1: <INT number of coordinates>
	2: <FLOAT xsize [m]> <FLOAT ysize [m]> <FLOAT xoff [m]> <FLOAT yoff [m]>
	3: <INT xsize [pix]> <INT ysize [pix]> <INT xoff [pix]> <INT yoff [pix]>
	4--: <INT xpos n [pix]> <INT ypos n [pix]>
	
	with 'xoff' and 'yoff' a global offsets for the positions, if necessary.
	
	N.B. Line 2 is currently not used, but provided for backwards compatibility.
	
	@param filename file holding the configuration as CSV.
	
	@return (<# of coords>, <pixelpos on CCD>, <pixelsize on CCD>)
	
	Raises IOError if file could not be found or RuntimeError if parsing did not 
	go as expected.
	"""
	
	if (not os.path.isfile(safile)):
		raise IOError("loadSaSfConf(): File '%s' does not exist." % (safile))
	
	reader = csv.reader(open(safile), delimiter=',')
	
	try: 
		# Number of coordinates [int]
		nsa = int((reader.next())[0])
		# Box size at aperture [float, float]
		line = reader.next()
		sallsize = N.array([float(line[0]), float(line[1])])
		# Try to read the lenslet offset, set to 0 if not present
		try: salloff = N.array([float(line[2]), float(line[3])])
		except: salloff = N.array([0, 0])
		# Box pixel size [int, int] and offset [int, int]
		line = reader.next()
		ccdsize = N.array([float(line[0]), float(line[1])])
		# Try to read the ccd offset, set to 0 if not present
		try: saccdoff = N.array([float(line[2]), float(line[3])])
		except: saccdoff = N.array([0, 0])
	except:
		raise RuntimeError("loadSaSfConf(): Could not parse file header.")
	
	ccdpos = []
	for line in reader:
		try:
			_pos = [float(line[0]), float(line[1])]
			ccdpos.append(_pos)
		except:
			raise RuntimeError("loadSaSfConf(): Could not parse file.")
	
	log.prNot(log.INFO, "loadSaSfConf(): In '%s': found %d coordinates, (expected %d)."% (os.path.split(safile)[1], len(ccdpos), nsa))
	
	if (len(ccdpos) != nsa):
		log.prNot(log.WARNING, "loadSaSfConf(): Found %d coordinates, expected %d. Using all positions found (%d)." % (len(ccdpos), nsa, len(ccdpos)))
		nsa = len(ccdpos)
	
	ccdsize = (N.array(ccdsize)).astype(N.float32)
	ccdpos = (N.array(ccdpos)+saccdoff).astype(N.float32)
	return (nsa, ccdpos, ccdsize)


def saveSaSfConf(sfile, n, llsize, ccdsize, ccdpos):
	"""
	Save the subaperture (sa) or subfield (sf) configuration to 'sfile'. The
	syntax of the subaperture and subfield configuration is the same, so we can 
	use the same function to save the data.
	
	@param sfile File to store configuration to
	@param n Number of sa/sf coordinates to store
	@param llsize The size in SI units of the sa/sf
	@param ccdsize The sa/sf pixelsize on the CCD
	@param ccdpos The sa/sf pixel positions on the CCD
	
	@return Nothing
	"""
	# Convert file to real path
	sfile = os.path.realpath(sfile)
	
	# If it exists, save old files
	if (os.path.isfile(sfile)):
		saveOldFile(sfile, postfix='.old', maxold=5)
	
	# Now open the file for writing
	writer = csv.writer(open(sfile, 'w'), delimiter=',')
	
	# Write 'n'
	writer.writerow([n])
	# Write llsize:
	writer.writerow(list(llsize) + [0.0, 0.0])
	# Write ccdsize
	writer.writerow(list(ccdsize) + [0, 0])
	# Write all ccd positions
	for pos in ccdpos:
		writer.writerow(pos)
	
	# Done
	


def calcSubaptConf(rad, size, pitch, shape='circular', xoff=[0,0.5], disp=(0,0), scl=1.0):
	"""
	Generate subaperture (sa) positions for a given configuration.
	
	@param rad radius of the sa pattern (before scaling) (in pixels)
	@param shape shape of the sa pattern ('circular' or 'square')
	@param size size of the sa's (in pixels)
	@param pitch pitch of the sa's (in pixels)
	@param xoff the horizontal position offset of odd rows (in units of 'size')
	@param disp global displacement of the sa positions (in pixels)
	@param scl global scaling of the sa positions (in pixels)
			
	@return (<# of subaps>, <subap pixelpos on CCD>, <subap pixelsize on CCD>)
	
	Raises ValueError if shape is unknown and RuntimeError if no subapertures we 
	found using the specified configuration.
	"""
	
	disp = N.array(disp)
	
	# (half) width and height of the subaperture array
	sa_arr = (N.ceil(rad/pitch)+1).astype(int)
	# Init empty list to store positions
	pos = []
	# Loop over all possible subapertures and see if they fit inside the
	# aperture shape. We loop y from positive to negative (top to bottom 
	# in image coordinates) and x from negative to positive (left to
	# right)
	for say in range(sa_arr[1], -sa_arr[1]-1, -1):
		for sax in range(-sa_arr[0], sa_arr[0]+1, 1):
			# Centroid coordinate for this possible subaperture is:
			sac = [sax, say] * pitch
						
			# 'say % 2' gives 0 for even rows and 1 for odd rows. Use this 
			# to apply a row-offset to even and odd rows
			# If we're in an odd row, check saccdoddoff
			sac[0] -= xoff[say % 2] * pitch[0]
			
			# Check if we're in the apterture bounds, and store the subapt
			# position in that case
			if (shape == 'circular'):
				if (sum((abs(sac)+size/2.0)**2) < rad**2): 
					pos.append(sac)
					log.prNot(log.INFO, "calcSubaptConf(): adding sa @ (%.3g, %.3g)." % \
					 	(sac[0], sac[1]))
			elif shape == 'square':
				if (abs(sac)+size/2.0 < rad).all:
					pos.append(sac)
					log.prNot(log.INFO, "calcSubaptConf(): adding sa @ (%.3g, %.3g)." % \
					 	(sac[0], sac[1]))
			else:
				raise ValueError("Unknown aperture shape '%s'" % (apts))
	
	if (len(pos) <= 0):
		raise RuntimeError("Didn't find any subapertures for this configuration.")
	
	# Apply scaling and displacement to the pattern before returning
	# NB: pos gives the *centroid* position of the subapertures here
	cpos = (N.array(pos) * scl) + disp
	
	# Convert symmetric centroid positions to origin positions:
	llpos = cpos - size/2.0
	
	nsa = len(llpos)
	log.prNot(log.NOTICE, "calcSubaptConf(): found %d subapertures." % (nsa))
	
	return (nsa, llpos, cpos, size)


def optSubapConf(img, sapos, sasize, saifac):
	"""
	Optimize subaperture mask position using a (flatfield) image.
	
	To optimize the subaperture pattern, take the initial origin positions 
	of subapertures (from 'pos'), find the origin of the subaperture, and cut a 
	horizontal an vertical slice of pixels out the image which are twice 
	as large as the size of the subaperture (from 'size').
	
	These two slices should then give an intensity profile across the 
	subimage, and since there is a dark band between the subimages, the 
	dimensions of each of these can be determined by finding the minimum 
	intensity in the slices.
	
	Positions will be rounded to whole pixels.
	
	N.B.: This methods is slightly sensitive to specks of dust on flatfields.
	
	@param img a 2-d numpy array
	@param sapos a list of lower-left pixel positions of the subapertures
	@param sasize a list of the pixel size of the subapertures
	@param saifac the intensity reduction factor counting as 'dark'
	
	@return (<# of subaps>, <subap pixelpos on CCD>, <subap pixelsize on CCD>)
	"""
	
	# Init optimium position and size variables
	optsapos = []				# Store the optimized subap position
	allsizes = []		# Stores all optimized subaperture sizes
	
	for pos in sapos:
		# Calculate the ranges for the slices (make sure we don't get
		# negative indices and stuff like that). The position (pos)
		# is the origin of the subap. This means we take origin + 
		# width*1.5 pixels to the right and origin - width*0.5 pixels to 
		# the left to get a slice across the subap. Same for height.
		## SAORIGIN
		slxran = N.array([max(0, pos[0]-sasize[0]*0.5), \
			min(img.shape[0], pos[0]+sasize[0]*1.5)])
		slyran = N.array([max(0, pos[1]-sasize[1]*0.5), \
			min(img.shape[1], pos[1]+sasize[1]*1.5)])
		
		# Get two slices in horizontal and vertical direction. Slices are 
		# the same width and height as the subapertures are estimated to
		# be, then averaged down to a one-pixel profile
		# NB: image indexing goes reverse: pixel (x,y) is at data[y,x]
		## SAORIGIN
		xslice = img[pos[1]:pos[1]+sasize[1], slxran[0]:slxran[1]]
		xslice = xslice.mean(axis=0)
		
		yslice = img[slyran[0]:slyran[1], pos[0]:pos[0]+sasize[0]]
		yslice = yslice.mean(axis=1)
		
		# Find the first index where the intensity is lower than saifac 
		# times the maximum intensity in the slices *in slice coordinates*.
		slmax = N.max([xslice.max(), yslice.max()])
		cutoff = slmax * saifac
		cutoff = N.mean(img[pos[1]+0.4*sasize[1]:pos[1]+0.6*sasize[1], \
			pos[0]+0.4*sasize[0]:pos[0]+0.6*sasize[0]]) * saifac
		saxran = N.array([ \
			N.argwhere(xslice[:slxran.ptp()/2.] < cutoff)[-1,0], \
			N.argwhere(xslice[slxran.ptp()/2.:] < cutoff)[0,0] + slxran.ptp()/2. ])
		sayran = N.array([ \
			N.argwhere(yslice[:slyran.ptp()/2.] < cutoff)[-1,0], \
			N.argwhere(yslice[slyran.ptp()/2.:] < cutoff)[0,0] + slyran.ptp()/2. ])
		log.prNot(log.DEBUG, "optSubapConf(): ranges: (%d,%d), (%d,%d) cutoff: %g" % \
			(slxran[0], slxran[1], slyran[0], slyran[1], cutoff))
			
		# The size of the subaperture is sa[x|y]ran[1] - sa[x|y]ran[0]:
		_sass = N.array([saxran.ptp(), sayran.ptp()])
		
		# The final origin pixel position in the large image (img) of 
		# the subaperture is the position we found in the slice
		# (saxran[0], sayran[0]),  plus the coordinate where the slice 
		# began in the big dataset (slxran[0], slyran[0]).
		## SAORIGIN
		# To get the centroid position: add half the size of the subimage
		_sapos = N.array([saxran[0] + slxran[0], \
			sayran[0] + slyran[0]])
		
		log.prNot(log.INFO, \
			"optSubapConf(): subap@(%d, %d), size: (%d, %d), pos: (%d, %d) end: (%d, %d)" % \
			(pos[0], pos[1], _sass[0], _sass[1], _sapos[0], _sapos[1], \
			_sapos[0] + _sass[0], _sapos[1] + _sass[1]))
		
		# The subimage size should be the same for all subimages. Store 
		# all subaperture sizes found during looping and then take the
		# mean afterwards.
		allsizes.append(_sass)
		optsapos.append(_sapos)
	
	# Calculate the average optimal subaperture size in pixels + standard dev
	optsize = N.array(allsizes).mean(axis=0)
	tmpstddev = (N.array(allsizes)).std(axis=0)
	
	optsapos = N.round(optsapos).astype(N.int)
	optsize = N.round(optsize).astype(N.int)
	
	log.prNot(log.NOTICE, "optSubapConf(): subimage size optimized to (%d,%d), stddev: (%.3g, %.3g) (was (%.3g, %.3g))" % \
		(optsize[0], optsize[1], tmpstddev[0], tmpstddev[1], \
		sasize[0], sasize[1]))
	if (tmpstddev.mean() > 1.0):
		log.prNot(log.WARNING, \
			"optSubapConf(): size standarddeviation rather high, check results!")
	
	return (len(optsapos), optsapos, optsize)


def procStatShift(shifts):
	"""
	Process shifts from individual files, return the average offset over all 
	files for each subaperture.
	
	To find the same object in each subimage (i.e. where can we find the same 
	granule in each subimage?), one has to measure image shifts between the 
	complete field of view of all subimages for all files. This will give an N * 
	Nref * Nsa * 2 array, with N the number of files, Nref the number of 
	reference subimages and Nsa the number of subapertures in the lenslet array.
	
	We expect that the average over N and Nref will give Nsa offset vectors 
	which give the location of the same pointing on the sun for each subimage.
	
	@param shifts Shifts measured over complete subimages for many files
	
	@return An array of vectors which gives the offset for each subimage.
	"""
	
	# First average over Nref
	s_ref = N.mean(shifts, axis=1)
	# Now make sure the average *per frame* is zero
	s_avgfr = N.mean(s_ref, axis=1)
	s_norm = s_ref - s_avgfr.reshape(-1,1,2)
	# Now average over all frames to get the offset. Also calculate the error
	s_off = N.mean(s_norm, axis=0)
	s_off_err = (N.var(s_norm, axis=0))**0.5
	# Done, return as tuple
	return (s_off, s_off_err)
	


def calcWfsDimmR0(shifts, sapos, sadiam, angscl, mind=2.0, wavelen=550e-9):
	"""
	Calculate the Friedmann parameter ($r_0$) from shifts measured in a 
	Shack-Hartmann wavefront sensor. 'shifts' should be a set of image shifts 
	with dimension N * SA * 2. N should be sufficiently large (~500 and more) 
	to allow for a statistical analysis. 'sapos' should be the centroid 
	positions of the subapertures (or subfields), 'sadiam' should be the 
	diameter of the	subapertures. The units should be some consistent set, if 
	the units are in SI, the output $r_0$ will be in centimeters. 'angscl' 
	should be the angular image scale at the CCD in radians per pixel.
	
	This function is based on the method described in the 'The ESO
	differential image motion monitor' by M. Sarazin and F. Roddier (1989).
	
	Returns a list with elements formatted as:
	[r_0_long, r_0_trans, distance, sa1, sa2]
	
	'mind' is the minimum distance in units of 'sasize' to use for the 
	analysis. The DIMM method is only valid for distances larger than twice 
	the subaperture diameter. [2.0]
	'wavelen' is the wavelength that the shifts are valid for [550e-9]
	"""
	
	# Convert to ndarrays
	shifts = N.array(shifts)
	sapos = N.array(sapos)
	
	# Check data sanity
	if (shifts.ndim != 3):
		log.prNot(log.WARNING, "calcWfsDimmR0(): shifts incorrect shape, should be 3D")
		return False
	if (shifts.shape[1] != sapos.shape[0]):
		log.prNot(log.WARNING, "calcWfsDimmR0(): shifts and sapos shapes do not match")
		return
	
	r0list = []
	# Loop over all subaperture-pairs
	for sa1 in xrange(len(sapos)):
		for sa2 in xrange(sa1+1, len(sapos)):
			### Analyze subaperture pair sa1-sa2 here
			# Calculate distance
			diff = sapos[sa1]-sapos[sa2]
			dist = ((diff**2.0).sum())**0.5
			# Check if distance is > mind * sasize, skip if not
			if (dist < mind * sadiam):
				continue
			
			# Angle between the two points sapos[sa1] and sapos[sa2]
			ang = N.arctan2(diff[1],diff[0])
			# Construct rotation matrix
			rotmat = N.array([	[N.cos(ang), -N.sin(ang)], \
								[N.sin(ang), N.cos(ang)]])
			# Calculate the differential image motion
			dimm = shifts[:, sa1, :] - shifts[:, sa2, :]
			# Rotate the shifts with rotmat
			# TODO: this is probably not optimal
			dimm_r = [N.dot(thisdimm, rotmat) for thisdimm in dimm]
			dimm_r = N.array(dimm_r)
			# Convert pixel shift to radian shift:
			dimm_r *= angscl
			
			# Now calculate the longitudinal [0] and transversal [1] variance
		 	(varl, vart) = N.var(dimm_r, axis=0)
			# Process this according to Eqns. (13) and (14) to get $r_0$
			# TODO: Eqns. might not be rewritten most efficiently
			# SOLVED: 20090407
			r0l = (2.0 * (wavelen**2.0) * \
				(0.179 * sadiam**(-1./3) - 0.0968 * dist**(-1./3)) / varl) \
				**(3./5)
			# r0l = ((varl**2.0 / (2.0*wavelen)) * \
			# 	(1.0 / (0.179*sadiam**(-1.0/3.0) - 0.0968*dist**(-1.0/3.0))))\
			# 		**(-3.0/5.0)
			r0t = (2.0 * (wavelen**2.0) * \
				(0.179 * sadiam**(-1./3) - 0.1450 * dist**(-1./3)) / vart) \
				**(3./5)
			# r0t = ((varl**2.0 / (2.0*wavelen)) * \
			# 	(1.0 / (0.179*sadiam**(-1.0/3.0) - 0.145*dist**(-1.0/3.0))))\
			# 		**(-3.0/5.0)
			r0list.append([dist, r0l, r0t, varl, vart, sa1, sa2])
	
	return N.array(r0list)

