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

These routines are based on the earlier ctomo-py files which simulated the atmosphere. Here however, the shifts come from real WFWFS data.

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

verb = 1
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
		# Default configuration paramters
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
			'sfindx' : 10,\
			'sfindy' : 10,\
			'sftotx' : 19,\
			'sftoty' : 19,\
			'saifac' : 0.9,\
			'datadir' : './rawdata',\
			'outdir' : './data',\
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
		self.sfindx = self.cfg.getfloat('telescope', 'sfindx')
		self.sfindy = self.cfg.getfloat('telescope', 'sfindy')
		self.sfind = N.array([self.sfindx, self.sfindy])
		self.sftotx = self.cfg.getfloat('telescope', 'sftotx')
		self.sftoty = self.cfg.getfloat('telescope', 'sftoty')
		self.sftot = N.array([self.sftotx, self.sftoty])
		self.nsf = product(self.sftot)
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
		# Calculate relative subfield positions
		self.calcSubfieldPos()
		
	
	def findData(self):
		"""
		Check if data (raw images, darks, flats) are available
		"""
		self.rawfiles = []
		self.flatfiles = []
		self.darkfiles = []
		self.logfiles = []
		if (verb > 1):
			print "Checking data in %s, matching raw, flats and darks with %s, %s and %s" % (self.datadir, self.rawpat, self.flatpat, self.darkpat)
		
		for root, dirs, files in os.walk(self.datadir):
			self.rawfiles.extend(fnmatch.filter(files, self.rawpat))
			self.flatfiles.extend(fnmatch.filter(files, self.flatpat))
			self.darkfiles.extend(fnmatch.filter(files, self.darkpat))
			self.logfiles.extend(fnmatch.filter(files, self.logpat))
		
		# Sort files in alphabetical order
		self.rawfiles.sort()
		self.flatfiles.sort()
		self.darkfiles.sort()
		self.logfiles.sort()
		
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
			print "Found %d images, %d flats, %d darks and %d logs" % \
			 	(len(self.rawfiles), \
				len(self.flatfiles), \
				len(self.darkfiles), \
				len(self.logfiles))
		if (verb > 1):
			print "Flat files: ", self.flatfiles
			print "Dark files: ", self.darkfiles
			print "Image files: ", self.rawfiles
			print "Log files: ", self.logfiles
	
	def loadLogs(self):
		"""
		Load logfiles listed in self.logfiles to memory
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
		Calculate the relative normalized centroid subfield positions and
		sizes.
		"""
		self.sfpos = indices(self.sftot, dtype=N.float32).reshape(2,-1).T / \
			(self.sftot) + 1./(2*self.sftot)
		self.sfsize = 1./self.sfind
		
	
	def genSubaptmask(self):
		"""
		Generate a subaperture mask based on a previously initialized 
		telescope configuration. This returns a list with the centroid
		subaperture positions.
		"""
		# (half) width and height of the subaperture array
		sa_arr = (N.ceil(self.aptr/self.sapitch)+1).astype(int)
		# Init empty list to store positions
		pos = []
		# Loop over all possible subapertures and see if they fit inside the
		# aperture shape
		for sax in range(-sa_arr[0], sa_arr[0]+1):
			for say in range(-sa_arr[1], sa_arr[1]+1):
				# Coordinate for this possible subaperture is
				sac = [sax, say] * self.sapitch
				
				# If we're in an odd row, check saxoffset
				if say % 2 != 0: sac[0] -= self.saxoff * self.sapitch[0]
				
				# Check if we're in the apterture bounds, and store the subapt
				# position in that case
				if self.apts == 'circular':
					if sum(sac**2) < self.aptr**2: pos.append(sac)
				elif self.apts == 'square':
					if (sac < self.aptr).all: pos.append(sac)
				else:
					raise ValueError("Unknown aperture shape", self.apts, \
						"(should be 'circular' or 'square')")

		# Apply scaling and displacement to the pattern before returning
		self.sapos = (N.array(pos)*self.sascl) + self.sadisp
		# Convert position to pixels
		self.sapospix = (self.sapos + self.aptr)/(2*self.aptr) * self.res
		# Count subimages
		self.nsa = len(self.sapos)
	
	def optMask(self, img):
		"""
		Optimize subaperture mask position using a sample image, preferably a
		flatfield.
		
		To optimize the pattern, take the initial positions given by
		genSubaptmask(), and cut a horizontal an vertical slice of pixels out
		the image which are twice as large as the size of the subaperture.
		These two slices should then give an intensity profile across the 
		subimage, and since there is a dark band between the subimages, the 
		dimensions of each of these can be determined by finding the minimum 
		intensity in the slices.
		"""
		# Init optimium position lists
		self.optsapos = []
		self.optsapospix = []
		self.optsasize = N.array([0,0])
		
		# Check if the image is a flatfield
		if (img.type != 'flat'):
			raise RuntimeWarning("Optmizing the subimage mask works best with flatfields")
		
		# The subaperture size in pixels is given by:
		self.pixsize = self.sasize/(2*self.aptr) * self.res
		
		# Loop over all subapertures
		for pos in self.sapospix:
			# Calculate the ranges for the slices (make sure we don't get
			# negative indices and stuff like that)
			slxran = N.array([max(0, pos[0]-self.pixsize[0]), \
				min(self.res[0], pos[0]+self.pixsize[0])])
			slyran = N.array([max(0, pos[1]-self.pixsize[1]), \
				min(self.res[1], pos[1]+self.pixsize[1])])

			# Get two slices in horizontal and vertical direction
			# NB: image indexing goes reverse: pixel (x,y) is at data[y,x]
			xslice = img.data[pos[1], 				slxran[0]:slxran[1]]
			yslice = img.data[slyran[0]:slyran[1], 	pos[0]]
			
			# Find the first index where the intensity is lower than 0.8 times
			# the maximum intensity in the slices *in slice coordinates*.
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
			optsasize = N.array([saxran.ptp(), sayran.ptp()])
			optpixpos = N.array([saxran[0] + slxran[0], \
				sayran[0] + slyran[0]]) + optsasize/2.
			
			# if (verb > 1):
			# 	print "Xran: ", saxran, " Yran: ", sayran
			# 	print "Sapos (%.4g,%.4g) -> (%.4g,%.4g), sasize (%.4g,%.4g) -> (%.4g,%.4g)" % (pos[0], pos[1], optpixpos[0], optpixpos[1], self.pixsize[0], self.pixsize[1], optsasize[0], optsasize[1])
			
			
			# Save positions as pixel and real coordinates
			self.optsapospix.append(optpixpos)
			self.optsapos.append( (optpixpos/self.res) * 2 * self.aptr - \
			 	self.aptr )
			# The subimage size should be the same for all subimages. Enforce
			# this by setting the subimage size to the average size for all 
			# subimages. Do this by summing all sizes, and then dividing by
			# the # of subimages
			self.optsasize += optsasize
		
		self.optsasize /= self.nsa
		# Init the binary mask that will show where the subimages are
		self.mask = N.zeros(self.res, dtype=N.uint8)
		
		if (verb > 0):
			print "Subimage size optimized to (%.3g,%.3g) (was (%.3g,%.3g))"%\
				(self.optsasize[0], self.optsasize[1], \
				self.pixsize[0], self.pixsize[1])
		
		for optpos in self.optsapospix:	
			# Now make a mask (0/1) for all subapertures (again, remember 
			#image indexing is 'the wrong' way around in NumPy (pixel (x,y) is 
			# at img[y,x]))
			self.mask[ \
				optpos[1]-self.optsasize[1]/2: \
			 	optpos[1]+self.optsasize[1]/2, \
				optpos[0]-self.optsasize[0]/2: \
				optpos[0]+self.optsasize[0]/2] = 1

		# Convert mask to bool for easier array indexing
		self.mask = self.mask.astype(N.bool)

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
			print "Trying to load file %s" % self.uri
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
			print "Duration:", time.time() - beg
				
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
	
	def maskSubimg(self, mask):
		"""
		Mask out everything but the subimages by multiplying the data with a 
		binary mask, then scaling the pixels to 0--1 within the data that is 
		left.
		"""
		mdata = self.data[mask]
		offset = mdata.min()
		gain = 1./(mdata.max()-offset)
		self.data = (self.data-offset) * gain * mask
		
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
	
	def cropSfSa(self, nsf, nsa, sapos, sasize, sfpos, sfsize):
		"""
		Crop all subfields in all subimages out of the full frame and store
		the cropped data in an nsa*nsf*x*y datacube, with x,y the the subfield
		resolution, nsf the number of subfields per subimage and nsa
		the number of subapertures
		"""
		
		# Create empty datacube to hold the subimages
		self.sfsubimgs = N.empty((nsa, nsf, sfsize[1], sfsize[0]), \
		 	dtype=self.data.dtype)
		
		# Loop over the subimages and fill the cube with data
		for (subimg, pos) in zip(self.sfsubimgs, sapos):
			for (subfield, _pos) in zip(subimg, sfpos):
				TODO
				# TODO: this is not finished yet, do we really want to solve
				# the problem like this? Maybe direct slicing and cutting and
				# simultaneous x-correlation in C is better/faster? Do we need 
				# to access the subfields/subimages here?
				subfield = self.data[ \
					pos[1]-sasize[1]/2 + _sfpos[1] * sasize[1] - : \
					pos[1]-sasize[1]/2 + _sfpos[1] * sasize[1]
					
					, :pos[1]+sasize[1]/2, \
					pos[0]-sasize[0]/2:pos[0]+sasize[0]/2]
		
		# Done
	
	def computeShifts(self, nsa, saref, sfpos, sfsize, usesf=None):
		"""
		Compute image shifts for the wfwfs subimages with saref as a reference 
		subaperture
		"""
		# Init shift vectors
		self.disp = N.empty(nsa, )
		refimg = self.subimgs[saref]
		
	
	def fitsSave(self, filename, dtype=None):
		"""
		Save an image as fits file
		"""
		if (verb > 1):
			print "Trying to save image as FITS file %s" % filename
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
			print "Duration:", time.time() - beg
	
	def pngSave(self, filename):
		"""
		Save image as PNG, always rescaling the data to 0--255 and saving it
		as grayscale image.
		"""
		if (verb > 1):
			print "Trying to save image as PNG file %s" % filename
			beg = time.time()
		
		scldat = (self.data - self.data.min())*255 / \
			(self.data.max() - self.data.min())
		surf = cairo.ImageSurface.create_for_data(scldat.astype(N.uint8), \
		 	cairo.FORMAT_A8, self.res[0], self.res[0])
		cairo.ImageSurface.write_to_png(surf, \
			os.path.join(self.wfwfs.outdir, filename))
		
		if (verb > 1):
			print "Duration:", time.time() - beg
	
	def _anaload(self, filename):
		"""
		Wrapper for loading ana files
		"""
		if (verb > 1):
			print "Trying to load ana file %s" % filename
		
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
			try:
				info = self.wfwfs.logdata[log][self.idx]
			except KeyError:
				pass
		
		if (info == None):
			raise RuntimeError("Could not find logfile information for image "+self.name +' or '+self.idx)
		elif (verb >1):
			print "Read in %sfield, info: " % self.type
			print info
		
		# Parsing info in a standard format
		self.info = imageInfo(idx=info[0], time=info[1], \
			res=(info[2], info[3]), exp=info[4], N=info[5])
	
	
	def _fitsload(self, filename):
		"""
		Wrapper for loading fits files
		"""
		if (verb > 1):
			print "Trying to load fits file %s" % filename
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


