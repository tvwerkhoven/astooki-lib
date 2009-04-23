#!/usr/bin/env /sw/bin/python2.5
# encoding: utf-8
"""
@file libsh.py
@brief Shack-Hartmann routines
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090423

This library provides some routines for analyzing/processing Shack-Hartmann 
wavefront sensor data.

Created by Tim van Werkhoven on 2009-04-23.
Copyright (c) 2008-2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/

$Id$
"""

#=============================================================================
# Import libraries here
#=============================================================================

import numpy as N				# Math & calculations
import csv
from liblog import *		# To print & log messages

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


def loadSubaptConf(filename):
	"""
	Try to load 'filename', if it exists. This should hold information on 
	the subaperture positions and sizes on the CCD. Positions should be 
	the lower-left corner of the subaperture. Syntax of the file should 
	be:
	line 1: <INT number of subapertures>
	line 2: <FLOAT xsize [m]> <FLOAT ysize [m]> (unused ATM)
	line 3: <INT xsize [pix]> <INT ysize [pix]>
	line 4--n: <INT subap n xpos [pix]> <INT subap n ypos [pix]>
	
	@param filename file holding the subaperture configuration as CSV.
	
	@return (<# of subaps>, <subap pixelpos on CCD>, <subap pixelsize on CCD>)
	
	Raises IOError if file could not be found or RuntimeError if parsing did not 
	go as expected.
	"""
	
	if (not os.path.isfile(safile)):
		raise IOError("loadSubaptConf(): File '%s' does not exist." % (filename))
	
	reader = csv.reader(open(safile), delimiter=',')
	
	try: 
		# Number of subapertures [int]
		nsa = int((reader.next())[0])
		# Subaperture size at aperture [float, float]
		line = reader.next()
		sallsize = N.array([float(line[0]), float(line[1])])
		# Subaperture pixel size [int, int]
		line = reader.next()
		saccdsize = N.array([int(line[0]), int(line[1])])
	except:
		raise RuntimeError("loadSubaptConf(): Could not parse file header.")
	
	saccdpos = []
	for line in reader:
		try:
			_pos = [int(line[0]), int(line[1])]
			saccdpos.append(_pos)
		except:
			raise RuntimeError("loadSubaptConf(): Could not parse file.")
	
	prNot(VERB_DEBUG, "loadSubaptConf(): Found %d subaps, (expected %d)."% \
		 (len(saccdpos), nsa))
	
	if (len(saccdpos) != nsa):
		prNot(VERB_WARN, "loadSubaptConf(): Found %d subaps, expected %d. Using all positions found (%d)." % (len(saccdpos), nsa, len(saccdpos)))
		nsa = len(saccdpos)
	
	return (nsa, N.array(saccdpos), N.array(saccdsize))


def calcSubaptConf(rad, size, pitch, shape='circular', xoff=0, disp=(0,0), scl=1.0):
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
					prNot(VERB_DEBUG, "calcSubaptConf(): adding sa @ (%.3g, %.3g)." % \
					 	(sac[0], sac[1]))
			elif shape == 'square':
				if (abs(sac)+size/2.0 < rad).all:
					pos.append(sac)
					prNot(VERB_DEBUG, "calcSubaptConf(): adding sa @ (%.3g, %.3g)." % \
					 	(sac[0], sac[1]))
			else:
				raise ValueError("Unknown aperture shape '%s'" % (apts))
	
	if (len(pos) <= 0):
		raise RuntimeError("Didn't find any subapertures for this configuration.")
	
	# Apply scaling and displacement to the pattern before returning
	# NB: pos gives the *centroid* position of the subapertures here
	pos = (N.array(pos) * scl) + disp
	
	# Convert symmetric centroid positions to CCD positions:
	saccdpos = N.round(pos + rad - size/2.0).astype(N.int)
	
	nsa = len(saccdpos)
	prNot(VERB_INFO, "calcSubaptConf(): found %d subapertures." % (nsa))
	
	return (nsa, sallpos, saccdpos)


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
		# TODO: this should work better, but doesn't. Why?
		#slmin = N.min([xslice.min(), yslice.min()])
		#cutoff = (slmax-slmin)*saifac
		cutoff = slmax * saifac
		saxran = N.array([ \
			N.argwhere(xslice[:slxran.ptp()/2.] < cutoff)[-1,0], \
			N.argwhere(xslice[slxran.ptp()/2.:] < cutoff)[0,0] + slxran.ptp()/2. ])
		sayran = N.array([ \
			N.argwhere(yslice[:slyran.ptp()/2.] < cutoff)[-1,0], \
			N.argwhere(yslice[slyran.ptp()/2.:] < cutoff)[0,0] + slyran.ptp()/2. ])
		
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
		
		prNot(VERB_DEBUG, \
			"optSubapConf(): subap@(%d, %d), size: (%d, %d), pos: (%d, %d)" % \
			(pos[0], pos[1], _sass[0], _sass[1], _sapos[0], _sapos[1]))
		prNot(VERB_DEBUG, "optSubapConf(): ranges: (%d,%d) and (%d,%d)"% \
			(slxran[0], slxran[1], slyran[0], slyran[1]))
		
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
	
	prNot(VERB_INFO, "optSubapConf(): subimage size optimized to (%d,%d), stddev: (%.3g, %.3g) (was (%.3g, %.3g))" % \
		(optsize[0], optsize[1], tmpstddev[0], tmpstddev[1], \
		sasize[0], sasize[1]))
	
	return (len(optpos), optpos, optsize)

