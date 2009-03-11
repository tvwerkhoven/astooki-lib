#!/usr/bin/env /sw/bin/python2.5
# encoding: utf-8
"""
@file tomolib.py
@brief Library for processing WFWFS data
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090218

This library takes WFWFS data as input (images), and processes that to analyse
the atmosphere. It is modular in setup such that different analysis methods 
can be used with the WFWFS data.

The analysis process works more or less like this:
1) Analysis parameters are set (by the user)
2) The WFWFS data is read in and the subapertures are identified.
3) The image displacements of subfields in the subapertures are measured, as 
   dictated by the configuration
4) The image displacements are used for analysis of the seeing, which can be a 
   variety of different methods

These routines are based on the earlier ctomo-py files which simulated the 
atmosphere. Here however, the shifts come from real WFWFS data.

Some conventions used in these (and other) routines include:
 - Naming subap/subfield variables as: [sa|sf][pix|][pos|size]
 - Coordinates, lengths, sizes stored as (x,y), slicing must therefore always 
    be done with data[coord[1], coord[0]]
 ? Position as centroid or LL?

Created by Tim van Werkhoven on 2008-02-18.
Copyright (c) 2008-2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/

$Id
"""

#=============================================================================
# Import libraries here
#=============================================================================

# Math & calculations stuff
import numpy as N
import scipy as S
import ConfigParser			# For parsing the config file
import fnmatch				# For matching the file patterns
import os					# For finding datafiles 
import pyana, pyfits		# For loading images
import cairo				# For PNG output
import time					# For timing measurements

#import scipy.interpolate
#import scipy.lib.blas
#import scipy.sparse

# Other run-time stuff
#import sys
#import os
#import os.path
#import time
#import subprocess
#import pickle

# Data visualisation
#import Gnuplot

#=============================================================================
# Some static defines, used for configuration
#=============================================================================

VERB_SILENT = 0
VERB_INFO = 1
VERB_DEBUG = 2

# Init setting for verbosity:
verb = VERB_DEBUG

#=============================================================================
# wfwfs, used for configuration and processing of wfwfs data
#=============================================================================

class wfwfs():
	"""
	The wide-field wavefront sensor (wfwfs) class is used for configuration of
	the (optical) setup of the wfs on a telescope, after which the wfs data 
	can be processed according to this configuration. The result is a large 
	amount of image displacement measurements between different subfields in 
	subimages.
	"""
	def __init__(self, cfgfile):
		# Default configuration parameters
		self.cfgdef = {'aptr': 0.5, \
			'apts' : 'circular',\
			'fovx' : 50,\
			'fovy' : 55,\
			'sapitchx' : 0.1,\
			'sapitchx' : 0.091,\
			'sapitchx' : 0.09,\
			'sapitchx' : 0.081,\
			'saxoff' : 0.5,\
			'sadispx' : 0.0,\
			'sadispy' : 0.0,\
			'sascl' : 0.95,\
			'sftotx' : 10,\
			'sftoty' : 10,\
			'sfsizex' : 0.1,\
			'sfsizey' : 0.1,\
			'sfoffx' : 0.0,\
			'sfoffy' : 0.0,\
			'sfpitchx' : 0.1,\
			'sfpitchy' : 0.1,\
			'saifac' : 0.9,\
			'datadir' : './rawdata/',\
			'outdir' : './data/',\
			'darkpat' : '*dd*',\
			'flatpat' : '*ff*',\
			'datapat' : '*im*',\
			'dfuse' : 'before'}
		
		# Load configuration from cfgfile
		self.cfg = ConfigParser.ConfigParser()
		self.cfg.read(cfgfile)
		
		# Set variables according to configuration
		self.aptr = self.cfg.getfloat('telescope', 'aptr')
		self.apts = self.cfg.get('telescope', 'apts')
		self.angle = []
		self.resx = self.cfg.getint('telescope', 'resx')
		self.resy = self.cfg.getint('telescope', 'resy')
		self.res = N.array([self.resx, self.resy])
		self.fovx = self.cfg.getfloat('telescope', 'fovx')
		self.fovy = self.cfg.getfloat('telescope', 'fovy')
		self.fov = N.array([self.fovx, self.fovy]) * 1/60*(N.pi/180)
		self.nsa = []
		self.sapitchx = self.cfg.getfloat('telescope', 'sapitchx')
		self.sapitchy = self.cfg.getfloat('telescope', 'sapitchy')
		self.sapitch = N.array([self.sapitchx, self.sapitchy])
		self.sasizex = self.cfg.getfloat('telescope', 'sasizex')
		self.sasizey = self.cfg.getfloat('telescope', 'sasizey')
		self.sasize = N.array([self.sasizex, self.sasizey])
		self.saxoff = self.cfg.getfloat('telescope', 'saxoff')
		self.sadispx = self.cfg.getfloat('telescope', 'sadispx')
		self.sadispy = self.cfg.getfloat('telescope', 'sadispy')
		self.sadisp = N.array([self.sadispx, self.sadispy])
		self.sascl = self.cfg.getfloat('telescope', 'sascl')
		self.sftotx = self.cfg.getfloat('telescope', 'sftotx')
		self.sftoty = self.cfg.getfloat('telescope', 'sftoty')
		self.sftot = N.array([self.sftotx, self.sftoty])
		self.sfsizex = self.cfg.getfloat('telescope', 'sfsizex')
		self.sfsizey = self.cfg.getfloat('telescope', 'sfsizey')
		self._sfsize = N.array([self.sfsizex, self.sfsizey])
		self.sfoffx = self.cfg.getfloat('telescope', 'sfoffx')
		self.sfoffy = self.cfg.getfloat('telescope', 'sfoffy')
		self._sfoff = N.array([self.sfoffx, self.sfoffy])
		self.sfpitchx = self.cfg.getfloat('telescope', 'sfpitchx')
		self.sfpitchy = self.cfg.getfloat('telescope', 'sfpitchy')
		self._sfpitch = N.array([self.sfpitchx, self.sfpitchy])
		self.nsf = N.product(self.sftot)
		self.saifac = self.cfg.getfloat('telescope', 'saifac')
		
		# Some params about where the data is and what it looks like
		self.datadir = self.cfg.get('data', 'datadir')
		self.outdir = self.cfg.get('data', 'outdir')
		self.darkpat = self.cfg.get('data', 'darkpat')
		self.flatpat = self.cfg.get('data', 'flatpat')
		self.rawpat = self.cfg.get('data', 'rawpat')
		self.logpat = self.cfg.get('data', 'logpat')
		self.dfuse = self.cfg.get('data', 'dfuse')
		
		# Check if output dir exists
		if (not os.path.isdir(self.outdir)):
			if (not os.path.exists(self.outdir)):
				os.mkdir(self.outdir)
			else:
				raise IOError("Output dir '%s' does exist, but is not a directory, aborting" % self.outdir)
		# Check if data dir exists
		if (not os.path.isdir(self.datadir)):
			raise IOError("Data dir '%s' does not exist, aborting" % self.datadir)
		
		# Find data
		self.findData()
		# Find logfiles
		self.loadLogs()
		# Generate initial subaperture mask (positions based on config)
		self.genSubaptmask()
		# Calculate initial subfield positions and sizes
		self.calcSubfieldPos()
	
	
	def findData(self):
		"""
		Check if data (raw images, darks, flats) are available in the 
		specified directories and with the specified masks.
		"""
		# Init empty lists to store the files in
		self.rawfiles = []
		self.flatfiles = []
		self.darkfiles = []
		self.logfiles = []
		
		if (verb > 1):
			print "findData(): Checking data in %s, matching raw, flats, darks and logs with %s, %s, %s and %s" % (self.datadir, self.rawpat, self.flatpat, self.darkpat, self.logpat)
		
		# Walk through the directorie(s)
		for (root, dirs, files) in os.walk(self.datadir):
			# Add any raws, flats, darks or logs to the lists
			self.rawfiles.extend(fnmatch.filter(files, self.rawpat))
			self.flatfiles.extend(fnmatch.filter(files, self.flatpat))
			self.darkfiles.extend(fnmatch.filter(files, self.darkpat))
			self.logfiles.extend(fnmatch.filter(files, self.logpat))
		
		# Sort files in alphabetical order
		self.rawfiles.sort()
		self.flatfiles.sort()
		self.darkfiles.sort()
		self.logfiles.sort()
		
		# Check if we actually found any files
		if len(self.rawfiles) == 0:
			raise RuntimeError("Could not find any rawfiles in " + \
			 	self.datadir + " matching the pattern '"+ self.rawpat + "'")
		if len(self.flatfiles) == 0:
			raise RuntimeError("Could not find any flatfields in " + \
			 	self.datadir + " matching the pattern '"+ self.flatpat + "'")
		if len(self.darkfiles) == 0:
			raise RuntimeError("Could not find any darkfields in " + \
			 	self.datadir + " matching the pattern '"+ self.darkpat + "'")
		if len(self.logfiles) == 0:
			raise RuntimeError("Could not find any logfiles in " + \
			 	self.datadir + " matching the pattern '"+ self.logpat + "'")
		
		if (verb > 0): 
			print "findData(): Found %d images, %d flats, %d darks and %d logs" % \
			 	(len(self.rawfiles), \
				len(self.flatfiles), \
				len(self.darkfiles), \
				len(self.logfiles))
		
		if (verb > 1):
			print "findData(): Flat files: ", self.flatfiles
			print "findData(): Dark files: ", self.darkfiles
			print "findData(): Image files: ", self.rawfiles
			print "findData(): Log files: ", self.logfiles
	
	
	def loadLogs(self):
		"""
		Load logfiles listed in self.logfiles to memory. Only works on SST 
		style log-syntax at the moment.
		"""
		self.logdata = {}
		# Loop over all logfiles
		for log in self.logfiles:
			self.logdata[log] = {}
			# For each logfile, read all lines
			f = open(os.path.join(self.datadir, log), 'r')
			lines = f.readlines()
			for entry in lines:
				# Parse each entry, 0:11 is image number, 13:24 time
				# 25:31 and 31:37 resolution, 37:47 exposure, 47:55 sum
				self.logdata[log][int(entry[0:11])] = (int(entry[0:11]), \
					entry[13:25], \
					int(entry[25:31]), \
					int(entry[31:37]), \
					float(entry[37:47]), \
					int(entry[47:55]))
	
	
	def optDarkFlat(self, darkimg, flatimg):
		"""
		Optimize dark- and flatfields for faster processing: convert dark and
		flat to float32, then calculate gain = 1/(flat-dark) so we can skip 
		this explicit calculation lateron.
		"""
		self.darkimg = darkimg.data.astype(N.float32)
		flatfloat = flatimg.data.astype(N.float32)
		self.gainimg = 1./(flatfloat-self.darkimg)
	
	
	def calcSubfieldPos(self):
		"""
		Calculate the lower-left subfield positions and sizes in both pixels 
		and in units of the subaperturesize.
		"""
		# We (should) already have sasize and sapixsize here. We don't need 
		# sasize because for the subfields we are not too interested in the 
		# absolute size, but would rather like to know the size in units of 
		# subaperture size instead, or in pixels.
		
		# sfsize, sfpitch and sfoff can be given in both pixels (if negative) 
		# or in units of the subaperture size (if positive). Fix this first.
		if ((self._sfsize > 0).all()):
			self.sfsize = self._sfsize
			self.sfpixsize = self._sfsize * self.sapixsize
		elif ((self._sfsize < 0).all()):
			self.sfpixsize = -1*self._sfsize
			self.sfsize = self.sfpixsize / self.sapixsize
		else:
			raise ValueError("calcSubfieldPos(): sfsize is invalid. Check configuration file.")
		
		if ((self._sfpitch > 0).all()):
			self.sfpitch = self._sfpitch
			self.sfpixpitch = self._sfpitch * self.sapixsize
		elif ((self._sfpitch < 0).all()):
			self.sfpixpitch = -1*self._sfpitch
			self.sfpitch = self.sfpixpitch / self.sapixsize
		else:
			raise ValueError("calcSubfieldPos(): sfpitch is invalid. Check configuration file.")
		
		if ((self._sfoff >= 0).all()):
			self.sfoff = self._sfoff
			self.sfpixoff = self._sfoff * self.sapixsize
		elif ((self._sfoff < 0).all()):
			self.sfpixoff = -1*self._sfoff
			self.sfoff = self.sfpixoff / self.sapixsize
		else:
			raise ValueError("calcSubfieldPos(): sfoff is invalid. Check configuration file.")
		
		if (verb>0):
			print "calcSubfieldPos(): sfsize: (%.3g,%.3g), sfpixsize: (%d,%d))"% \
				(self.sfsize[0], self.sfsize[1], self.sfpixsize[0], \
				 self.sfpixsize[1])
			print "calcSubfieldPos(): sfoff: (%.3g,%.3g), sfpixoff: (%d,%d))"% \
				(self.sfoff[0], self.sfoff[1], self.sfpixoff[0], \
				 self.sfpixoff[1])
			print "calcSubfieldPos(): sfpitch: (%.3g,%.3g), sfpixpitch: (%d,%d))"% \
				(self.sfpitch[0], self.sfpitch[1], self.sfpixpitch[0], \
				 self.sfpixpitch[1])
		
		# Now calculate the positions of the subfields.
		self.sfpos = self.sfoff + \
			N.indices(self.sftot, dtype=N.float32).reshape(2,-1).T * \
			self.sfpitch
		self.sfpixpos = self.sfpos * self.sapixsize
	
	
	def genSubaptmask(self):
		"""
		Generate a subaperture mask based on a previously initialized  
		telescope configuration. This returns a tuple with the centroid
		subaperture positions in SI units and pixels.
		"""
		
		# (half) width and height of the subaperture array
		sa_arr = (N.ceil(self.aptr/self.sapitch)+1).astype(int)
		# Init empty list to store positions
		pos = []
		# Loop over all possible subapertures and see if they fit inside the
		# aperture shape. We loop y from positive to negative (top to bottom 
		# in image coordinates) and x from negative to positive (left to
		# right)
		for say in range(sa_arr[1], -sa_arr[1]-1, -1):
			for sax in range(-sa_arr[0], sa_arr[0]+1, 1):
				# Lower-left coordinate for this possible subaperture is
				sac = [sax, say] * self.sapitch
				
				# If we're in an odd row, check saxoffset
				if (say % 2 != 0):
					sac[0] -= self.saxoff * self.sapitch[0]
				
				# Check if we're in the apterture bounds, and store the subapt
				# position in that case
				if (self.apts == 'circular'):
					if (sum(sac**2) < self.aptr**2): 
						pos.append(sac)
				elif self.apts == 'square':
					if (sac < self.aptr).all: 
						pos.append(sac)
				else:
					raise ValueError("Unknown aperture shape", self.apts, \
						"(should be 'circular' or 'square')")
		
		# Apply scaling and displacement to the pattern before returning
		self.csapos = (N.array(pos) * self.sascl) + self.sadisp
		
		# Convert position to pixels
		self.csapixpos = N.round(\
			(self.csapos + self.aptr) / \
				(2*self.aptr) * self.res ).astype(N.int)
		# Convert subimage size to pixels as well
		self.sapixsize = self.sasize * self.res / (2*self.aptr)
		
		# Count subimages
		self.nsa = len(self.csapos)
		if (verb >0):
			print "genSubaptmask(): found %d subapertures." % (self.nsa)
		
		return (self.csapos, self.csapixpos)
	
	
	def optMask(self, img):
		"""
		Optimize subaperture mask position using a flatfield image.
		
		To optimize the subaperture pattern, take the initial positions given 
		by genSubaptmask(), and cut a horizontal an vertical slice of pixels
		out the image which are twice as large as the size of the subaperture.
		
		These two slices should then give an intensity profile across the 
		subimage, and since there is a dark band between the subimages, the 
		dimensions of each of these can be determined by finding the minimum 
		intensity in the slices.
		
		This methods is slightly sensitive to specks of dust on flatfields.
		"""
		
		# Init optimium position and size variables
		optsapos = []			# Store the optimized subap position
		optsapixpos = []		# Store the optimized subap position in pixels
		allsapixsize = []		# Stores all optimized subaperture sizes
		
		# Check if the image is a flatfield
		if (img.type != 'flat'):
			raise RuntimeWarning("Optmizing the subimage mask works best with flatfields")
		
		# Loop over all subapertures
		for pos in self.csapixpos:
			# Calculate the ranges for the slices (make sure we don't get
			# negative indices and stuff like that)
			slxran = N.array([max(0, pos[0]-self.sapixsize[0]), \
				min(self.res[0], pos[0]+self.sapixsize[0])])
			slyran = N.array([max(0, pos[1]-self.sapixsize[1]), \
				min(self.res[1], pos[1]+self.sapixsize[1])])
			
			# Get two slices in horizontal and vertical direction
			# NB: image indexing goes reverse: pixel (x,y) is at data[y,x]
			xslice = img.data[pos[1], 				slxran[0]:slxran[1]]
			yslice = img.data[slyran[0]:slyran[1], 	pos[0]]
			
			# Find the first index where the intensity is lower than 0.8 times
			# the maximum intensity in the slices *in slice coordinates*.
			# TODO: this only uses a one-pixel slice, should use more than 
			# that, maybe 30 -- 50 or something
			slmax = N.max([xslice.max(), yslice.max()])
			saxran = N.array([ \
				N.argwhere(xslice[:slxran.ptp()/2.] < \
				 	slmax*self.saifac)[-1,0], \
				N.argwhere(xslice[slxran.ptp()/2.:] < \
					slmax*self.saifac)[0,0] + \
				 	slxran.ptp()/2. ])
			sayran = N.array([ \
				N.argwhere(yslice[:slyran.ptp()/2.] < \
					slmax*self.saifac)[-1,0], \
				N.argwhere(yslice[slyran.ptp()/2.:] < \
					slmax*self.saifac)[0,0] + \
					slyran.ptp()/2. ])
			
			# The final centroid pixel position in the large image (img.data)
			# is the position we found in the slice (saxran[0], sayran[0]), 
			# plus the coordinate where the slice began in the big dataset 
			# (slxran[0], slyran[0]), and then add half the size of the 
			# subimage
			_optsapixsize = N.array([saxran.ptp(), sayran.ptp()])
			_optsapixpos = N.array([saxran[0] + slxran[0], \
				sayran[0] + slyran[0]])# + optsasize/2.
			_optsapos = (_optsapixpos/self.res) * 2 * self.aptr - self.aptr
			
			if (verb > 1):
				print "optMask(): processing subap at (%d, %d), found size (%d, %d) at position (%d, %d)" % (pos[0], pos[1], _optsapixsize[0], _optsapixsize[1], _optsapixpos[0], _optsapixpos[1])
				print "optMask(): used ranges: (%d,%d) and (%d,%d)" % (slxran[0], slxran[1], slyran[0], slyran[1])
			
			# Save positions as pixel and real coordinates
			optsapixpos.append(_optsapixpos)
			optsapos.append(_optsapos)
			
			# The subimage size should be the same for all subimages. Store 
			# all subaperture sizes found during looping and then take the
			# mean afterwards.
			allsapixsize.append(_optsapixsize)
		
		optsapixsize = N.array(allsapixsize).mean(axis=0)
		
		# Calculate optimum size in SI units as well
		optsasize = optsapixsize * (2*self.aptr) / self.res
		
		# Round the pixel position and sizes
		optsapixsize = N.round(optsapixsize).astype(N.int)
		optsapixpos = N.round(optsapixpos).astype(N.int)
				
		tmpstddev = (N.array(allsapixsize)).std(axis=0)
		if (verb > 0):
			print "optMask(): subimage size optimized to (%d,%d), stddev: (%.3g, %.3g) (was (%.3g, %.3g))" % \
				(optsapixsize[0], optsapixsize[1], \
				tmpstddev[0], tmpstddev[1], \
				self.sapixsize[0], self.sapixsize[1])
		
		# Save the new values globally
		self.sapixsize = N.array(optsapixsize)
		self.sasize = N.array(optsasize)
		self.sapixpos = N.array(optsapixpos)
		self.sapos = N.array(optsapos)
		
		# Init a binary mask that will show where the subimages are
		self.mask = N.zeros(self.res, dtype=N.uint8)
		self.gridmask = N.zeros(self.res, dtype=N.uint8)
		
		for optpos in self.sapixpos:	
			# Now make a mask (0/1) for all subapertures (again, remember 
			#image indexing is 'the wrong' way around in NumPy (pixel (x,y) is 
			# at img[y,x]))
			self.mask[ \
				optpos[1]: \
			 	optpos[1]+self.sapixsize[1], \
				optpos[0]: \
				optpos[0]+self.sapixsize[0]] = 1
			# The gridmask is only 1 at the edges...
			self.gridmask[\
				optpos[1]: \
			 	optpos[1]+self.sapixsize[1], \
				optpos[0]: \
				optpos[0]+self.sapixsize[0]] = 1
			# ... and zero everywhere else
			self.gridmask[\
				optpos[1]+1: \
			 	optpos[1]+self.sapixsize[1]-1, \
				optpos[0]+1: \
				optpos[0]+self.sapixsize[0]-1] = 0
		
		# Convert mask to bool for easier array indexing lateron
		self.mask = self.mask.astype(N.bool)
	
	# Mask modes for overlayMask()
	MASK_NONE = 0
	MASK_ALL = 1
	MASK_CIRCULAR = 2
	def overlayMask(self, img, filename='./test', number=False, coord=False, markCorner=True, maskOutside=MASK_NONE):
		"""
		Generate a 'fancy' image from 'img' with an overlay of the subaperture
		mask, and possibly with indices and coordinates at the subapertures if 
		requested.
		"""
		
		# Apply mask first
		# TODO: not complete yet
		if (maskOutside == self.MASK_ALL):
			masked = img[self.mask]
		else:
			masked = img
		
		# Get extrema
		maxval = N.max(masked)
		minval = N.min(masked)
		# Overlay the gridmask on a sample image
		masked[self.gridmask.astype(N.bool)] = maxval
		# Scale the values to 0-255
		masked = (255*(masked - minval)/(maxval - minval)) 
		
		# Mark the 'lower left corner' in numpy coordinate space
		if markCorner:
			masked[0:10,0:20] = 255
		
		# Init a new empty Cairo surface as target surface
		caidata = N.empty(masked.shape, dtype=N.uint8)
		destsurf = cairo.ImageSurface.create_for_data(caidata, \
		 	cairo.FORMAT_A8, caidata.shape[0], caidata.shape[1])
		
		# Create a context from the empty surface
		ctx = cairo.Context(destsurf)
		
		# Init a new Cairo surface from the masked imaage
		imgsurf = cairo.ImageSurface.create_for_data(masked.astype(N.uint8), \
		 	cairo.FORMAT_A8, masked.shape[0], masked.shape[1])
		
		# Mirror the image vertically, so we use a FITS-like origin (first 
		# quadrant of a graph) instead of an image like origin (where we see
		# the 2nd  quadrant of a graph)
		ctx.save()
		mat = cairo.Matrix(1, 0, 0, -1, 0, imgsurf.get_height())
		ctx.transform(mat)
		
		# Use the image as source, paint it
		ctx.set_source_surface(imgsurf, 0,0)
		ctx.paint()
		ctx.restore()
		
		# Choose a font
		ctx.set_font_size(12)
		ctx.select_font_face('Serif', cairo.FONT_SLANT_NORMAL, \
		 	cairo.FONT_WEIGHT_NORMAL)
		
		# Loop over the subapertures and put some text there
		sanum = 0
		for sapos in self.sapixpos:
			# Move the 'cursor', show some text
			# NOTE: we have to perform the position transform ourselves here, 
			# because if we would use ctx.transform(), the text would be 
			# transformed as well (which we do not want)
			ctx.move_to(sapos[0] +1, imgsurf.get_height()- (sapos[1]+1))
			txt = ''
			if (number != False):
				txt += '%d' % (sanum)
				sanum += 1
			if (coord != False):
				txt += ' (%d,%d)' % (sapos[0], sapos[1])
			
			ctx.show_text(txt)
		
		# Done, save as PNG
		destsurf.write_to_png(filename + '.png')
		
		# And as FITS file
		pyfits.writeto(filename + '.fits', masked, clobber=True)
		
		if (verb > 0):
			print "overlayMask(): done, wrote debug image as fits and png."
	
	
	def visCorrMaps(self, maps, res, sapos, sasize, sfpos, sfsize, shifts=None, filename='./debug/corrmaps', outpdf=False, outfits=False):
		"""
		Visualize the correlation maps generated and the shifts measured.
		"""
		
		if (outfits):
			raise RuntimeWarning("FITS output is not implemented yet.")
			return
		
		if ((not outfits) & (not outpdf)):
			raise RuntimeWarning("outfits and outpdf both false, no output will be generated, returning.")
			return
		
		# Globally scale correlation maps to 0-255, convert to uint8
		scmaps = 255*(maps - maps.min())/(maps.max() - maps.min())
		scmaps = scmaps.astype(N.uint8)
		
		# Init cairo for PDF if necessary
		# TODO: 72x72 pt (1x1 inch) reasonable?
		if (outpdf):
			pdfsurf = cairo.PDFSurface(filename+'.pdf', 72.0, 72.0)
			# Create context
			ctx = cairo.Context(pdfsurf)
			
			# Set coordinate system to 'res', origin at lower-left
			ctx.translate(0, 72.0)
			ctx.scale(72.0/res[0], -72.0/res[1])
		
		# Map/shvec counter
		saidx = 0
		msize = N.array(maps[0].shape)
		
		# Loop over the subapertures
		for _sapos in sapos:
			sfidx=0
			# Loop over the subfields
			for _sfpos in sfpos:
				# Current lower-left pixel position of this subfield
				_pos = _sapos + _sfpos
				_end = _pos + sfsize
				_cent = _pos + sfsize/2
				
				# Put the current correlation map at _cent
				if (outpdf):
					# Create outline of the subfield
					ctx.set_source_rgb(0.0, 0.0, 0.0)
					ctx.move_to(_pos[0], _pos[1])
					ctx.rel_line_to(sfsize[0], 0)
					ctx.rel_line_to(0, sfsize[1])
					ctx.rel_line_to(-sfsize[0], 0)
					ctx.rel_line_to(0, -sfsize[1])
					ctx.stroke()
					
					# Create a surface from the data
					tmpmap = scmaps[saidx, sfidx]
					tmps = scmaps[saidx, sfidx].shape
					# Create slightly larger but proper strided array
					tmpmap2 = N.zeros( \
						(N.ceil(tmps[0]/4.0)*4, \
						N.ceil(tmps[1]/4.0)*4), \
						dtype=N.uint8)
					tmpmap2[:tmps[0], :tmps[1]] = tmpmap
					# Create a cairo surface from this buffer
					surf = cairo.ImageSurface.create_for_data(\
					 	tmpmap2, \
					 	cairo.FORMAT_A8, \
					 	tmpmap2.shape[0], \
					 	tmpmap2.shape[1])
					# Set it as source
					# TODO: surf can be as much as 3 pixels larger than the 
					# map, because of striding problems. Therefore, we maybe 
					# should clip the data before paint()ing or fill()ing it.
					# - msize[0]/2
					ctx.set_source_surface(surf, _cent[0], \
					 	_cent[1])
					#  - msize[1]/2
					# Paint it
					ctx.paint()
					# If shifts are give, draw lines
					if (shifts != None):
						# Set to black
						ctx.set_source_rgb(0.0, 0.0, 0.0)
						# Move cursor to the center
						ctx.move_to(_cent[0], _cent[1])
						# Draw a shift vector
						ctx.rel_line_to(shifts[saidx][sfidx][0], \
						 	shifts[saidx][sfidx][1])
						ctx.stroke()
						# Move cursor to lower-left
						ctx.move_to(_pos[0]+5, _pos[1]+5)
						ctx.set_font_size(10) # in pixels?
						ctx.select_font_face('Serif', \
							cairo.FONT_SLANT_NORMAL, \
							cairo.FONT_WEIGHT_NORMAL)
						# Reset scaling
						ctx.scale(1,-1)
						ctx.show_text('%d, %d: (%.3g,%.3g)' % \
						 	(saidx, sfidx, shifts[saidx][sfidx][0], \
						 	shifts[saidx][sfidx][1]))
						ctx.scale(1,-1)
				sfidx += 1
			saidx += 1
		
		if (outpdf):
			pdfsurf.finish()
		# Done


class wfwfsImg():
	"""
	Class to hold information on a WFWFS image (dark, flat, raw or processed)
	"""
	def __init__(self, wfwfs, name, imgtype, format):
		# Link to parent wfwfs class
		self.wfwfs = wfwfs
		# The filename
		self.name = name
		# Path where the file is stored
		self.uri = os.path.join(self.wfwfs.datadir, name)
		# Type of data (dark, flat, raw, corrected, cropped)
		self.type = imgtype
		# Format of the data on disk (ana, fits)
		self.format = format
		# Location to the data itself
		self.data = []
		# Resolution
		self.res = self.wfwfs.res
		# Bits per pixel
		self.bpp = -1
		# Only for darks and flats, see load()
		self.info = None
		
		# Some class specific variables
		# Supported file formats
		self._formats = ['ana', 'fits']
		# Methods to load the file formats and file meta data
		self._formatload = {'ana': self._anaload, 'fits': self._fitsload}
		self._formatinfo = {'ana': self._anainfo, 'fits': self._fitsinfo}
		
		# Get the image index (simply a counter)
		self.getIndex()
		
		# Finally load the image
		self.load()
	
	
	def load(self):
		"""
		Load an image file into memory
		"""
		if (verb > 1):
			print "load() Trying to load file %s" % self.uri
			beg = time.time()
		
		# Check if file exists
		if (not os.path.isfile(self.uri)):
			raise IOError("Cannot find file")
		
		# Check if format is supported, and load if possible
		if (self.format in self._formats):
			self._formatload[self.format](self.uri)
		else:
			raise TypeError("File format not supported")
		
		# If dark or flat, also check logfile
		if (self.type in ['dark', 'flat']):
			self._formatinfo[self.format]()
			# If dark or flat, divide data by number of images summed
			self.data /= self.info.N
		
		if (verb > 1):
			print "load() Duration:", time.time() - beg
				
	def getIndex(self):
		"""
		Get the index or sequence number from a filename of an image
		"""
		self.idx = int(self.name.split('.')[1])
	
	def darkFlatField(self, dark, gain=None, flat=None):
		"""
		Dark and flatfield an image.
		"""
		if (gain != None):
			self.data = (self.data.astype(N.float32) - dark) * gain
		elif (flat != None):
			self.data = (self.data.astype(N.float32) - dark) / (flat-dark)
		else:
			raise RuntimeError("Cannot dark/flatfield without (darkfield && (flatfield || gainfield))")
		
		self.type = 'corrected'
	
	def maskSubimg(self, mask, whitebg=False):
		"""
		Mask out everything but the subimages by multiplying the data with a 
		binary mask, then scaling the pixels to 0--1 within the data that is 
		left.
		"""
		mdata = self.data[mask]
		offset = mdata.min()
		gain = 1./(mdata.max()-offset)
		self.data = (self.data-offset) * gain * mask
		
		# If we want a white background, we don't set the other parts to 0,
		# but the the maximum intensity in the image
		if (whitebg):
			self.data += (mask == 0) * self.data.max()
		
		self.type = 'masked'
	
	def getStats(self):
		"""
		Get min, max and RMS from data
		"""
		tmp = self.data[self.wfwfs.mask]
		dmin = N.min(tmp)
		dmax = N.max(tmp)
		drms = (N.mean(tmp**2.0))**0.5
		return (dmin, dmax, drms)
	
	def computeShifts(self, nsa, saref, sfpos, sfsize, usesf=None):
		"""
		Compute image shifts for the wfwfs subimages with saref as a reference 
		subaperture
		"""
		# Init shift vectors
		# TODO: this is not finished
		self.disp = N.empty(nsa, )
		refimg = self.subimgs[saref]
		
	
	def fitsSave(self, filename, dtype=None):
		"""
		Save an image as fits file
		"""
		if (verb > 1):
			print "fitsSave(): Trying to save image as FITS file %s" % filename
			beg = time.time()
		
		if (dtype):
			hdu = pyfits.PrimaryHDU(self.data.astype(dtype))
		else:
			hdu = pyfits.PrimaryHDU(self.data.astype(dtype))
		
		hdu.header.update('origin', 'WFWFS Data')
		hdu.header.update('origname', self.name)
		hdu.header.update('type', self.type)
		hdu.writeto(os.path.join(self.wfwfs.outdir, filename))
		if (verb > 1):
			print "fitsSave(): Duration:", time.time() - beg
	
	def pngSave(self, filename):
		"""
		Save image as PNG, always rescaling the data to 0--255 and saving it
		as grayscale image.
		"""
		if (verb > 1):
			print "pngSave(): Trying to save image as PNG file %s" % filename
			beg = time.time()
		
		scldat = (self.data - self.data.min())*255 / \
			(self.data.max() - self.data.min())
		surf = cairo.ImageSurface.create_for_data(scldat.astype(N.uint8), \
		 	cairo.FORMAT_A8, self.res[0], self.res[0])
		cairo.ImageSurface.write_to_png(surf, \
			os.path.join(self.wfwfs.outdir, filename))
		
		if (verb > 1):
			print "pngSave(): Duration:", time.time() - beg
	
	def _anaload(self, filename):
		"""
		Wrapper for loading ana files
		"""
		if (verb > 1):
			print "_anaload(): Trying to load ana file %s" % filename
		
		anafile = pyana.load(filename)
		self.data = anafile['data']		
		if (self.res != anafile['header']['dims']).any():
			raise ValueError("Resolution in config file (%dx%d) not the same as file (%s) just loaded (%dx%d)" % \
				(self.res[0], self.res[1], filename, \
				 anafile['header']['dims'][0], anafile['header']['dims'][1]))
		self.bpp = self.data.nbytes / N.product(self.res)
	
	def _anainfo(self):
		"""
		Get metadata on darks and flats, i.e. how many frames were summed etc
		"""
		# Loop over all logfiles
		info = None
		for log in self.wfwfs.logdata:
			# Check if this log file has our file
			# TODO: idx is not unique! idx count is reset each day, so we 
			# should check for date instead...
			try:
				info = self.wfwfs.logdata[log][self.idx]
			except KeyError:
				pass
		
		if (info == None):
			raise RuntimeError("Could not find logfile information for image "+self.name +' or '+self.idx)
		elif (verb >1):
			print "_anainfo(): Read in %sfield, info: " % self.type
			print info
		
		# Parsing info in a standard format
		self.info = imageInfo(idx=info[0], time=info[1], \
			res=(info[2], info[3]), exp=info[4], N=info[5])
	
	
	def _fitsload(self, filename):
		"""
		Wrapper for loading fits files
		"""
		if (verb > 1):
			print "_fitsload(): Trying to load fits file %s" % filename
		self.data = pyfits.getdata(filename)
		self.res = self.data.shape
		self.bpp = self.data.nbytes / product(self.res)
	
	def _fitsinfo(self):
		raise RuntimeError("Fits info not implemented!")
	



class imageInfo():
	"""
	Information about an image
	"""
	def __init__(self, idx=-1, time='00:00:00', res=(-1,-1), exp=0.0, N=1):
		self.idx = idx
		self.time = time
		self.res = res
		self.exp = exp
		self.N = N
	


def main():
	pass

if __name__ == '__main__':
	main()


