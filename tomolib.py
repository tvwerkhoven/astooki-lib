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

verb = 2
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
			'datadir' : './data',\
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
		self.datadir = self.cfg.get('telescope', 'datadir')
		self.darkpat = self.cfg.get('telescope', 'darkpat')
		self.flatpat = self.cfg.get('telescope', 'flatpat')
		self.rawpat = self.cfg.get('telescope', 'rawpat')
		self.logpat = self.cfg.get('telescope', 'logpat')
		self.dfuse = self.cfg.get('telescope', 'dfuse')
		
		# Check presence of data
		#self.dataCheck()
		
		# Use the configuration to generate a subaperture mask
		#self.sapos = self.genSubaptmask()
	
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
	
	def optMask(self, img):
		"""
		Optimize subaperture mask position using a sample image
		"""
		self.optsapos = []
		self.optsapospix = []
		if (img.type != 'flat'):
			raise RuntimeWarning("Optmizing the subimage mask works best with flatfields")
		
		self.pixsize = self.sasize/(2*self.aptr) * self.res
		self.max = N.zeros(self.res)
		# Loop over all subapertures
		for pos in self.sapospix:
			# Get two slices in horizontal and vertical direction
			# NB: image slicing goes reverse, i.e. pixel x,y is at data[y,x]
			# TODO: cropping causes a problem: slices no longer 2*pixsize long
			self.xslice = img.data[\
				max([0,pos[1]-self.pixsize[1]]): \
			 	min(self.res[1], pos[1]+self.pixsize[1]), \
			 	pos[0]]
			self.yslice = img.data[pos[1], \
				max(0,pos[0]-self.pixsize[0]): \
				min(self.res[0], pos[0]+self.pixsize[0]) ]
			# Find lowest points on both sides of the slice *in slice
			# coordinates*
			self.xran = [N.argmin(self.xslice[:self.pixsize[0]]), \
			 	N.argmin(self.xslice[self.pixsize[0]:]) + self.pixsize[0]]
			self.yran = [N.argmin(self.yslice[:self.pixsize[1]]), \
			 	N.argmin(self.yslice[self.pixsize[1]:]) + self.pixsize[1]]
		
			optpixpos = ([self.xran + pos[0] - self.pixsize[0],\
			 	self.yran + pos[1] - self.pixsize[1]])
			self.optsapos.append( (optpixpos/self.res) * 2 * self.aptr - \
			 	self.aptr )
			self.optsapospix.append(optpixpos)
			# Find maximum and minimum *within* the ranges in the xslice
			#self.smax = N.max(self.xslice[self.xran[0]:self.xran[1]])
			#self.smin = N.min(self.xslice[self.xran[0]:self.xran[1]])
		
			# Find the pixel that is at 80% of the maximum
			#self.xslice[self.xran[0]:self.xran[1]]
			
			# Now make a mask (0/1) for these subapertures
			self.mask[]
			
			

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
		# Type of data (dark, flat, raw, corr)
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
	
				
	def getIndex(self):
		"""
		Get the index or sequence number from a filename of an image
		"""
		self.idx = int(self.name.split('.')[1])
	
	def darkFlatField(self, dark, flat, hist=.90):
		"""
		Dark and flatfield an image. Will also provide a histogram filtering
		if hist is set (default 0.90). 
		"""
		self.data = (self.data*1.0 - dark.data)/ \
			(flat.data-dark.data)
		#self.data
		#histogram(a, bins=10, range=None, normed=False, weights=None, new=None)
		self.type = 'corrected'
	
	def fitsSave(self, filename):
		"""
		Save an image as fits file
		"""
		if (verb > 1):
			print "Trying to save image as fits file %s" % filename

		hdu = pyfits.PrimaryHDU(self.data)
		hdu.header.update('origin', 'WFWFS Data')
		hdu.header.update('origname', self.name)
		hdu.header.update('type', self.type)
		hdu.writeto(filename)
	
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


