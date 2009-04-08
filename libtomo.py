#!/usr/bin/env /sw/bin/python2.5
# encoding: utf-8
"""
@file libtomo.py
@brief Library for processing (WF)WFS data
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090218

This library supplies various routines to process and analyse WFWFS data. This
analysis is goverened by a configuration file indicating various properties of
the data. See tomoconf-sst.cfg for examples.

There are three classes exported by this library:
 - WfsSetup(), a class to characterize a WFWFS setup and process the data 
   accordingly,
 - WfsData(), a class which monitors directories for suitable data, identifies 
   raw images, darks and flats, and tries to find metadata,
 - WfwfsImg(), a class to load/process/save Shack-Hartmann images.

These routines are based on the earlier ctomo-py files which simulated the 
atmosphere. Here however, the shifts come from real WFWFS data.

Some conventions used in these (and other) routines include:
 - Naming subap/subfield variables as: [sa|sf][pix|][pos|size]
 - Coordinates, lengths, sizes stored as (x,y), slicing in numpy must 
    therefore always be done with data[coord[1], coord[0]]

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
import libshifts			# To calculate image shifts
import libplot				# To process and plot results
from liblog import *		# To print & log messages

# File parsing/loading/IO
import ConfigParser			# For parsing the config file
import fnmatch				# For matching the file patterns
import pyana, pyfits		# For loading images
import os					# For various OS functions (chdir, mkdir, etc)
import csv					# For parsing CSV files

# Data visualisation & imaging
import cairo				# For PNG output (pngSave)
import pylab

# Other run-time stuff
import cPickle				# For general data stuff I/O
import sys					# For sys.exit()
import time					# For timing measurements
import re					# For parsing strings/filenames

# Debugging
import guppy
# import gc
# gc.set_debug(gc.DEBUG_LEAK)

#=============================================================================
# WfsData(), used for finding, loading and pre-processing WFS data
#=============================================================================

class WfsData():
	"""
	This class is used to find, load and pre-process WFS data. 
	
	Initialize by passing a suitable configuration file as argument.
	"""
	
	def __init__(self, cfgfile, wfssetup):
		# Some static configuration options
		# Telescope setup
		self.SETUP_SST = 'sst'
		self.SETUP_DST = 'dst'
		self.SETUP_SUPPORTED =  [self.SETUP_SST, self.SETUP_DST]
		# Various strategies for matching dark and flatfields to raw files
		self.DFUSE_BEFORE = 'before'
		self.DFUSE_AFTER = 'after'
		self.DFUSE_CLOSEST = 'closest'
		self.DFUSE_SUPPORTED = [self.DFUSE_BEFORE, self.DFUSE_AFTER, \
		 	self.DFUSE_CLOSEST]
		
		# Default configuration parameters
		self.cfgdef = { \
			'datadir' : './rawdata/',\
			'outdir' : './data/',\
			'format' : 'ana', \
			'darkpat' : '*dd*',\
			'flatpat' : '*ff*',\
			'rawpat' : '*im*',\
			'dfuse' : self.DFUSE_BEFORE, \
			'procdelay' : 1.0, \
			'setup' : self.SETUP_SST, \
			'runintval' : 10.0, \
			'runminframes' : 100, \
			'dodarkflat' : True, \
			'dooptgrid' : True, \
			'dostatic' : True, \
			'dosubfield' : True \
			}
		
		prNot(VERB_INFO, "Initializing WfsData()")
		
		# Tasks that can be performed on data (should be MPI.INT's!)
		self.TASK_SUBIMG = 1				# Subimage shift measurement
		self.TASK_SUBFIELD = 2  			# Subfield shift measurement
		# Settings that can be set
		self.TASK_SETSASIZE = 10			# Set subaperture size
		self.TASK_SETSAPOS = 11				# Set subaperture positions
		self.TASK_SETSFSIZE = 12			# Set subfield size
		self.TASK_SETSFPOS = 13				# Set subfield positions
		self.TASK_SETNSAREF = 14			# Set number of reference
		 									# subimages to use
		
		# Load configuration from cfgfile
		self.cfg = ConfigParser.SafeConfigParser(self.cfgdef)
		self.cfg.read(cfgfile)
		
		# Save config filename
		self.cfgfile = cfgfile
		
		# Change directory to that of the configfile
		self.curdir = os.path.realpath(os.path.curdir)
		os.chdir(os.path.realpath(os.path.dirname(cfgfile)))
		
		# Parse data format and structure variables
		self.datadir = os.path.realpath(self.cfg.get('data', 'datadir'))
		self.outdir = os.path.realpath(self.cfg.get('data', 'outdir'))
		self.format = self.cfg.get('data', 'format')
		self.darkpat = self.cfg.get('data', 'darkpat')
		self.flatpat = self.cfg.get('data', 'flatpat')
		self.rawpat = self.cfg.get('data', 'rawpat')
		self.dfuse = self.cfg.get('data', 'dfuse')
		self.procdelay = self.cfg.getfloat('data', 'procdelay')
		self.runintval = self.cfg.getfloat('data', 'runintval')
		self.runminframes = self.cfg.getint('data', 'runminframes')
		self.parsedelay = self.cfg.getfloat('data', 'parsedelay')
		
		# Some parameters on the metadata structure
		self.setup = self.cfg.get('metadata', 'setup')
		self.logpat = self.cfg.get('metadata', 'logpat')
		
		# Parse process parameters, what do we need to process?
		self.dodarkflat = self.cfg.getboolean('process', 'dodarkflat')
		self.dooptgrid = self.cfg.getboolean('process', 'dooptgrid')
		self.dostatic = self.cfg.getboolean('process', 'dostatic')
		self.dosubfield = self.cfg.getboolean('process', 'dosubfield')
		
		# Link to a WfsSetup() instance
		self.ws = wfssetup
		
		# Check configuration file for sanity
		self.checkSetupSanity()	
		
		# Change directory back
		os.chdir(self.curdir)
	
	
	def checkSetupSanity(self):
		"""
		This function checks the given configuration and does some basic tests
		to see if the setup is sane at all.
		"""
		# Check if output dir exists
		if (not os.path.isdir(self.outdir)):
			if (not os.path.exists(self.outdir)):
				os.mkdir(self.outdir)
			else:
				raise IOError("Output dir '%s' does exist, but is not a directory, aborting" % self.outdir)
		
		# Check if data dir exists
		if (not os.path.isdir(self.datadir)):
			raise IOError("Data dir '%s' does not exist, aborting" % self.datadir)
		
		# Check supported setups
		if (not self.setup in self.SETUP_SUPPORTED):
			raise RuntimeError("Setup '%s' not supported at this point." % \
				(self.setup))
		
		# Check dark/flat-field usage types:
		if (not self.dfuse in self.DFUSE_SUPPORTED):
			raise RuntimeError("Dark/flat-field strategy '%s' not supported at this point." % \
				(self.dfuse))
		
		# Check if when dodarkflat is false, dooptgrid should probably also be
		# false, because we need a flatfield to optimize
		if (self.dooptgrid and not self.dodarkflat):
			raise RuntimeWarning("dooptgrid is true, but dodarkflat is false. Flatfields are required for dooptgrid, I hope they're present.")
		
		# Check if filetype is supported
		# if (not self.format in SUPPORTED_FORMATS):
		# 	raise RuntimeError("File format '%s' not supported." % \
		# 		(self.format))
	
	
	def watchDog(self, args):
		"""
		This routine checks the datadirectory for new files continuously. 
		'args' should be a tuple consisting of two callback functions for
		parseRun().
		"""
		# Raw files are stored per directory and per run in a nested dict
		self.rawfiles = {}
		# Dark- and flatfields are stored in one big list because we need to
		# search the whole list for each raw file to get a matching dark/flat.
		# If available, load previous progress. 
		self.flatfiles = loadData(self.outdir, \
		 	'flatfile-info', aspickle=True)
		if (self.flatfiles is False): self.flatfiles = []
		self.darkfiles = loadData(self.outdir, \
		 	'darkfile-info', aspickle=True)
		if (self.darkfiles is False): self.darkfiles = []
		
		prNot(VERB_INFO, "watchDog(): starting monitoring '%s'." % \
			(self.datadir))
		
		# Start watching forever
		while (True):
			# Loop over the directories in datadir
			for (root, dirs, files) in os.walk(self.datadir):
				# Get subdirectory we are now parsing
				subdir = os.path.normpath(root.replace(self.datadir, ''))
				# Make sure the subdir does *not* start with '/' or something
				if (subdir[0] == os.path.sep): subdir=subdir.replace('/','',1)
				resultdir = os.path.normpath(os.path.join(self.outdir,subdir))
				
				prNot(VERB_INFO, "watchDog(): parsing directory '%s'." % \
				 	(subdir))
				
				# If we already processed this dir, skip it
				if (self.rawfiles.has_key(root)):
					prNot(VERB_INFO, "watchDog(): Directory '%s' already parsed, skipping." % (subdir))
					continue
				# If the directory is not clean, skip it
				if (not self.checkDataSanity(root, files)):
					prNot(VERB_INFO, "watchDog(): Directory '%s' not clean, skipping." % (subdir))
					continue
				# If the data is too new, skip it
				if (self.getAge(root, files) < self.procdelay):
					prNot(VERB_INFO, "watchDog(): Directory newer than %.3g days (age is: %.3g), skipping." % (self.procdelay, self.getAge(root, files)))
					continue
				
				# Check previous progress made in this dir
				self.rawfiles = loadData(resultdir, \
					'rawfile-info', aspickle=True)
				if (self.rawfiles is not False):
					prNot(VERB_INFO, "watchDog(): Found pre-parsed cache.")
					prNot(VERB_INFO, "watchDog(): %d runs:" % \
						(len(self.rawfiles[root].keys())))
					for runid in self.rawfiles[root]:
						prNot(VERB_DEBUG, "watchDog(): %s: %d files." % \
							(runid, self.rawfiles[root][runid]['nfiles']))
				else:
					self.rawfiles = {}
					# Check for files in the currect directory
					tmpraw = fnmatch.filter(files, self.rawpat)
					# NOTE: we store flat and darkfiles as complete path, 
					# because they do not have to be in the same directory 
					# (and in fact they are likely to be not)					
					tmpflat = fnmatch.filter(files, self.flatpat)
					tmpflat = [os.path.join(root, flat) for flat in tmpflat]
					tmpdark = fnmatch.filter(files, self.darkpat)
					tmpdark = [os.path.join(root, dark) for dark in tmpdark]
					
					# Get metadata for files
					metaraw = self.getMetaData(root, tmpraw, files)
					metaflat = self.getMetaData(root, tmpflat, files)
					metadark = self.getMetaData(root, tmpdark, files)
				
					# Store flat/dark meta info in a big list. Use extend to
					# get a simple flat list instead of a list nested per day.
					self.flatfiles.extend(metaflat)
					self.darkfiles.extend(metadark)
				
					# Split files up in runs
					self.rawfiles[root] = self.splitRuns(metaraw)
					
					# Add directory information
					for runid in self.rawfiles[root]:
						# The general output directory
						self.rawfiles[root][runid]['outdir'] = self.outdir
						# The directory with data currently begin parsed
						self.rawfiles[root][runid]['datadir'] = root
						# The subdirectory of self.datadir we are parsing now
						self.rawfiles[root][runid]['datasubdir'] = subdir
						# The corresponding output directory
						self.rawfiles[root][runid]['resultdir'] = resultdir
						# Directory for processing cache
						self.rawfiles[root][runid]['cachedir'] = \
							os.path.join(resultdir, runid, 'cache')
						# Directory for plots and all
						self.rawfiles[root][runid]['plotdir'] = \
						 	os.path.join(resultdir, runid, 'plots')
					
					# Find suitable dark- and flatfields for each run
					self.rawfiles[root] = \
					 	self.matchDarkFlat(self.rawfiles[root], \
					 	self.flatfiles, self.darkfiles)
				
					# Cache progress to disk
					saveData(self.rawfiles[root][runid]['resultdir'], \
					 	'rawfile-info', self.rawfiles, aspickle=True)
					saveData(self.outdir, 'darkfile-info', \
						self.darkfiles, aspickle=True)
					saveData(self.outdir, 'flatfile-info', \
						self.flatfiles, aspickle=True)
				
				# Distribute work
				for runid in self.rawfiles[root]:
					prNot(VERB_INFO, \
						"watchDog(): Submitting run '%s' to queue." % (runid))
					self.parseRun(self.rawfiles[root][runid], args)
				
				prNot(VERB_INFO, \
					"watchDog(): Succesfully parsed files in '%s'." % (root))
					
			
			# Give some info
			prNot(VERB_INFO, "watchDog(): Directory parse run complete.")
			for root in self.rawfiles:
				prNot(VERB_INFO, "watchDog(): Found %d runs in '%s':" % \
				 	(len(self.rawfiles[root]), root))
				for runid in self.rawfiles[root]:
					prNot(VERB_INFO, "watchDog(): Run %s has %d files." % \
						(runid, self.rawfiles[root][runid]['nfiles']))
			
			# Wait 60 seconds before next check
			sys.exit(0)
			prNot(VERB_INFO, "watchDog(): Sleeping %d seconds." % \
			 	(self.parsedelay))
			time.sleep(self.parsedelay)
	
	
	def checkDataSanity(self, rootdir, allfiles):
		"""
		Check if the directory can be processed and that nothing is missing.
		"""
		
		# For the SST, we require:
		# - exactly one logfile in the directory
		# - the directory is formatted as %Y-%m-%d
		# - directory date == logfile date
		# - there is at least one raw image
		if (self.setup == self.SETUP_SST):
			# Check logfiles
			logfiles = fnmatch.filter(allfiles, self.logpat)
			if (not len(logfiles) == 1):
				prNot(VERB_DEBUG, "checkDataSanity(): Cannot process '%s': number of logfiles is not 1 (found %d)" % (rootdir, len(logfiles)))
				return False
			
			# Get the last subdirectory
			subdir = os.path.basename(os.path.normpath(rootdir))
			# Check directory format, must be formatted as a date
			# TODO: check this format!
			try:
				dirdate = time.strptime(subdir, "%Y-%m-%d")
			except ValueError:
				prNot(VERB_DEBUG, "checkDataSanity(): Cannot process '%s': directory not formatted as %%Y-%%m-%%d" % (subdir))
				return False
			
			# Check the logfile date
			fnamere = \
				re.compile("^.*_lg(\d+[a-zA-Z]+\d{4})\.?(\d*)$")
			filedate = fnamere.match(logfiles[0])
			logdate = time.strptime(filedate.group(1), "%d%b%Y")
			if (logdate != dirdate):
				prNot(VERB_DEBUG, "checkDataSanity(): Cannot process '%s': directory date and logfile date differ (%.5g)" % (dirdate-logdate))
				return False
			
			# Check raw files
			rawfiles = fnmatch.filter(allfiles, self.rawpat)
			if (len(rawfiles) <= 0):
				prNot(VERB_DEBUG, "checkDataSanity(): Cannot process '%s': no rawfiles found matching pattern '%s'." % (rootdir, self.rawpat))
				return False
			
			# If we got to here, we're ok
			return True	
		# For the DST we don't require anything
		elif (self.setup == self.SETUP_DST):
			return True
	
	
	def getAge(self, rootdir, allfiles):
		"""
		Get the age of a directory in days so we know whether we should
		process it or not. Processing data which is still being
		written/appended is not a good idea.
		
		Note: this function has only limited precision. It uses the date of 
		the directory to calculate the age, such that the age is a step 
		function in days.
		"""
		
		# For the SST, we get the date from the directory
		if (self.setup == self.SETUP_SST):
			# Date from directory:
			# TODO: check this format!
			subdir = os.path.basename(os.path.normpath(rootdir))
			dirdate = time.strptime(subdir, "%Y-%m-%d")
			dirdate = time.mktime(dirdate)
			
			# Calculate age in days
			now = time.time()
			days = (now-dirdate)/3600./24.
			return days
		# For the DST, we use static dates
		elif (self.setup == self.SETUP_DST):
			if (rootdir.find('DST_080653NN') > 0):
				return time.mktime(time.strptime("2008-06-01", "%Y-%m-%d"))
				#return 10.0
			elif (rootdir.find('DST_111339NN') > 0):
				return time.mktime(time.strptime("2008-06-10", "%Y-%m-%d"))
				#return 15.0
			else:
				return -1
	
	
	def getMetaData(self, rootdir, datafiles, allfiles):
		"""
		Get meta data in for files 'datafiles' in directory 'rootdir'. 
		'allfiles' is a list of all files available in 'rootdir' and will be 
		used to find logfiles in certain cases.
		
		The list 'fileinfo' returned will be a list with metadata attached for
		each entry in 'datafiles', structured as:
		[datafile, [index, ctime, resx, resy, exposure, quality, comment]]
		"""
		
		# SST specific meta-data search: use logfiles as source.
		if (self.setup == self.SETUP_SST):
			# Get the logfiles from all files
			logfiles = fnmatch.filter(allfiles, self.logpat)
			# Raise error when there is not exactly one logfile
			if (len(logfiles) != 1):
				raise RuntimeError("Found %d logfiles instead of 1." % (len(logfiles)))
			
			logfile = logfiles[0]
			
			# Regexp to get the date and index from the filename
			fnamere = \
			 	re.compile("^.*_(lg|im|dd|ff)(\d+[a-zA-Z]+\d{4})\.?(\d*)$")
			
			# Find the date in the filename
			date = fnamere.match(logfile)
			if (date == None):
				raise RuntimeError("Could not find date in filename '%s'. Was looking for something like '(\d+[a-zA-Z]+\d{4})')" % (logfile))
			
			# The date will be stored in DDMMMYYYY format (or similar) in
			# date.group(2). Parse this to a epoch time
			datestr = date.group(2)
			datectime = time.strptime(datestr, "%d%b%Y")
			datectime = time.mktime(datectime)
			
			prNot(VERB_DEBUG, "getMetaData(): logfile: '%s' -> date: '%s' -> ctime: '%f'" % (logfile, datestr, datectime))
			
			logdata = {}
			
			# Open the file, loop over the lines
			fd = open(os.path.join(rootdir, logfile), 'r')
			for line in fd.readlines():
				# Parse each entry. 0:11 is image number, 13:24 time 25:31 and
				# 31:37 resolution, 37:47 exposure, 47:55 sum for DF/FF or the
				# 'quality' of the image, 69:71 'DD' or 'FF' for darks and
				# flats, '' for raw images. Linelength 56 means raw image, 72
				# means dark or flat
				
				# TODO: this conversion to seconds is ugly!
				ts = time.strptime(line[13:21], "%H:%M:%S")
				meta = [int(line[0:11]), \
					datectime + ts.tm_hour*3600 + ts.tm_min*60 + ts.tm_sec + \
					 	float(line[21:25]), \
					int(line[25:31]), \
					int(line[31:37]), \
					float(line[37:47]), \
					int(line[47:55]), \
					line[69:71]]
				# Store in dict/tuples for easy access
				logdata[meta[0]] = meta
			
			fileinfo = []
			# Now process the files
			for dfile in datafiles:
				# Get index, should be the last 7 bytes of the filename
				idx = int(dfile[-7:])
				# Get meta info
				try:
					fileinfo.append([dfile, logdata[idx]])
				except KeyError:
					raise RuntimeError("Error while getting metadata for file '%s'. Could not get metadata while using logfile '%s'." % (dfile, logfile))
		
		# DST meta-search: fake metadata
		elif (self.setup == self.SETUP_DST):
			fileinfo = []
			idx = 0
			for dfile in datafiles:
				fileinfo.append([dfile, \
					[idx, 1234000000+idx, 512, 512, 2.0, 1000, '']])
				idx += 1
			
		return fileinfo
	
	
	def makeFieldCache(self, files):
		"""
		Given a list with entries like [filename, [idx, ctime, ...]], generate
		a numpy.ndarray with entries [idx, ctime]. This can then be used to 
		easily look for the best dark/flatfield (see matchDarkFlat()).
		"""
		cache = N.zeros(len(files))
		i = 0
		for i in xrange(len(files)):
			cache[i] = files[i][1][1]
		
		return cache
	
	
	def splitRuns(self, rawfiles):
		"""
		Split up the files in 'rawfiles' in different 'runs'. A run is defined 
		as a sequence of at least self.runminframes frames where the cadence 
		between the frames is less than self.runintval in seconds. These runs 
		are interpreted as different 'measurements' of the atmospheric seeing.
		
		Each entry of 'rawfiles' should have the structure:
		[filename, [index, ctime, resx, resy, exposure, quality, comment]]
		which is exactly the result returned by getMetaData()
		
		Return value will be a dict with a unique key for each run. The value 
		will be a dict as well with the structure:
		run[runid]['files'] holds a list of raw files
		run[runid]['begin'] begin time of the run (epoch)
		run[runid]['end'] end time of the run (epoch)
		"""
		
		prNot(VERB_INFO, "splitRuns(): Splitting %d rawfiles." % \
		 	(len(rawfiles)))
		# Init empty dict for the various runs
		runs = {}
		
		# Loop over files, split in runs. 
		nrun = 0
		totfile = 0
		tmprun = []
		for (fname, meta) in rawfiles:
			# First run is special
			if (len(tmprun) == 0):
				first = [fname, meta]
				prev = meta[1]
				tmprun.append(fname)
				prNot(VERB_DEBUG, "splitRuns(): found first file '%s' date '%f'." % (fname, meta[1]))
			else:
				# If the current frame is within self.runintval of the 
				# previous frame, add it to the current run
				if (meta[1] - prev < self.runintval):
					#prNot(VERB_DEBUG, "splitRuns(): adding file '%s' to run." % (fname))
					last = [fname, meta]
					prev = meta[1]
					tmprun.append(fname)
				# Otherwise, check whether the previous images up till now are 
				# enough to be qualify as a 'run'
				else:
					if (len(tmprun) > self.runminframes):
						# Enough frames, add the run with some meta info
						nfile = len(tmprun)
						runid = 'run%02d' % (nrun)
						runs[runid] = {'begin': first[1][1], \
							'end': last[1][1], \
							'duration': last[1][1] - first[1][1], \
							'runid': runid, \
							'nfiles' : nfile, \
							'files' : tmprun}
						nrun += 1
						totfile += nfile
						prNot(VERB_INFO, "splitRuns(): Adding new run '%s' with %d frames, from %s to %s." % \
							(runid, \
							len(tmprun), \
							time.ctime(first[1][1]), \
							time.ctime(last[1][1])))
						# Clean list
						tmprun = []
					else:
						# Not enough frames, start over
						prNot(VERB_DEBUG, "splitRuns(): Discarding run with only %d frames: too short." % (len(tmprun)))
						first = [fname, meta]
						prev = meta[1]
						tmprun = []
						tmprun.append(fname)
		
		# Check the last tmprun after the for loop
		if (len(tmprun) > self.runminframes):
			# Enough frames, add the run with some meta info
			nfile = len(tmprun)
			runid = 'run%02d' % (nrun)
			runs[runid] = {'begin': first[1][1], \
				'end': last[1][1], \
				'duration': last[1][1] - first[1][1], \
				'runid': runid, \
				'nfiles' : nfile, \
				'files' : tmprun}
			nrun += 1
			totfile += nfile
			prNot(VERB_INFO, "splitRuns(): Adding new run '%s' with %d frames, from %s to %s." % \
				(runid, \
				len(tmprun), \
				time.ctime(first[1][1]), \
				time.ctime(last[1][1])))
		else:
			# Not enough frames, start over
			prNot(VERB_DEBUG, "splitRuns(): Discarding run with only %d frames: too short." % (len(tmprun)))
		
		# Output stats
		prNot(VERB_INFO, "splitRuns(): Found %d runs in total." % (nrun))
		# Return result
		return runs
	
	
	def matchDarkFlat(self, rawfiles, flatfiles, darkfiles):
		"""
		Given a raw file 'rawfile', find a darkfield and flatfield that this 
		rawfile needs to be corrected with using 'dfuse' (or self.dfuse) as a 
		guideline.
		
		Each entry of 'rawfiles', 'flatfiles' and 'darkfiles' should have the
		structure:
		[filename, [index, ctime, resx, resy, exposure, quality, comment]]
		
		This function adds the dict keys 'dark' and 'flat' to each 
		rawfiles[runid], with a structure like:
		rawfiles[runid]['flat'] = [[flatfile, age [s], multiplicity], 
			[flatfile, age [s], multiplicity], ...]
		and the same for 'dark'. Each filename is a complete path in this 
		case.
		"""
		
		# Check if we need to do darkfielding at all
		if (self.dodarkflat is False):
			prNot(VERB_INFO, "matchDarkFlat(): Skipping dark-/flatfield matching, not required.")
			for runid in rawfiles:
				rawfiles[runid]['flat'] = [[None]]
				rawfiles[runid]['dark'] = [[None]]
			return rawfiles
		
		# Check if we have flatfiles and darkfiles at all
		if (len(flatfiles) == 0 or len(darkfiles) == 0):
			raise RuntimeError("Cannot match dark and flatfiles")
		
		# Make cache for faster matching
		flatcache = self.makeFieldCache(self.flatfiles)
		darkcache = self.makeFieldCache(self.darkfiles)
		
		for runid in rawfiles:
			beg = rawfiles[runid]['begin']
			end = rawfiles[runid]['end']
			
			if (self.dfuse == self.DFUSE_BEFORE):
				# Find flat and dark just *before* the run:
				diff_ff = flatcache - beg
				diff_df = darkcache - beg
				diff_ff.sort()
				diff_df.sort()
				# For all values that are negative (i.e. before the run), 
				# take the 5 closest ones, reverse the order
				best_ffidx = N.where(diff_ff <= 0)[0][-5:][::-1]
				best_dfidx = N.where(diff_df <= 0)[0][-5:][::-1]
			elif (self.dfuse == self.DFUSE_AFTER):
				# Find flat and dark just *after* the run:
				diff_ff = flatcache - end
				diff_df = darkcache - end
				diff_ff.sort()
				diff_df.sort()
				# For all values that are positiev (i.e. after the run), 
				# take the 5 closest ones
				best_ffidx = N.where(diff_ff >= 0)[0][:5]
				best_dfidx = N.where(diff_df >= 0)[0][:5]
			if (len(best_ffidx) == 0 or \
				len(best_dfidx) == 0 or \
				self.dfuse == self.DFUSE_CLOSEST):
				if (len(best_ffidx) == 0 or len(best_dfidx) == 0):
					prNot(VERB_SILENT, "matchDarkFlat(): could not find darks/flats for run '%s' using strategy '%s'. Falling back to '%s'." % \
						(runid, self.dfuse, self.DFUSE_CLOSEST))
				
				# Find flat and dark around the run
				diff_ff = flatcache - (beg+end)/2
				diff_df = darkcache - (beg+end)/2
				# Take absolute value, we don't care about the sign
				diff_ff = N.abs(diff_ff)
				diff_df = N.abs(diff_df)
				# Sort in chronological order
				diff_ff.sort()
				diff_df.sort()
				# Take first five values, these represent the lowest time
				best_ffidx = range(5)
				best_dfidx = range(5)
				
				
			# Get the filenames and age for these indices
			best_ff = []
			best_df = []
			for ffidx in best_ffidx:
				best_ff.append([flatfiles[ffidx][0], diff_ff[ffidx], \
				 	flatfiles[ffidx][1][5]])
			for dfidx in best_dfidx:
				best_df.append([darkfiles[dfidx][0], diff_df[dfidx], \
					darkfiles[ffidx][1][5]])
			
			# Add to run data
			rawfiles[runid]['flat'] = best_ff
			rawfiles[runid]['dark'] = best_df
			
			# Give info
			prNot(VERB_DEBUG, "matchDarkFlat(): For runid '%s' found flats:" % (runid))
			prNot(VERB_DEBUG, best_ff)
			prNot(VERB_DEBUG, "matchDarkFlat(): For runid '%s' found darks:" % (runid))
			prNot(VERB_DEBUG, best_df)
			
		return rawfiles
	
	
	def parseRun(self, runfiles, args):
		"""
		Process a run of datafiles, as structured by watchDog(). This function 
		will be called from the watchdog process once it ordered a group of 
		files in a 'run'. The filenames for these raw filenames are stored in 
		'runfiles'.
		
		'args' should be a tuple of 'submitCB', 'resultCB' and 'broadcastCB'. 
		These three functions should take care of submitting work to workers, 
		getting results from workers and broadcasting settings to workers 
		respectively.
		
		'submitCB' should return True when submission was successful, or 
		non-true when results are ready to be read out. In the latter case, 
		'resultCB' should be called to clear a part of the queue.
		"""
		
		prNot(VERB_INFO, "parseRun(): parsing files from one run now.")
		# Parse 'args', which should hold callbacks to submit a job, get 
		# results from the job-queue and broadcast settings to the clients:
		submitCB, resultCB, broadcastCB = args
		
		# This dict will store the location of various files
		progress = {}
		progress['plotdir'] = runfiles['plotdir']
		progress['cachedir'] = runfiles['cachedir']
		
		# Phase 0, local initialisation
		# =============================
		
		# Make output directories
		if (not os.path.isdir(runfiles['plotdir'])):
			os.makedirs(runfiles['plotdir'])
		if (not os.path.isdir(runfiles['cachedir'])):
			os.makedirs(runfiles['cachedir'])
		
		if (self.dooptgrid or self.dodarkflat): 
			prNot(VERB_INFO, "parseRun(): optimizing grid with flatfield.")
			flatfield = WfwfsImg(runfiles['flat'][0][0], imgtype='flat', \
			 	format=self.format)
			# TODO: is this clean enough? Maybe do this in WfwfsImg()?
			flatfield.data /= 1.0*runfiles['flat'][0][2]
			
			libplot.overlayMask(flatfield.data, self.ws.saccdpos, \
			 	self.ws.saccdsize, libplot.mkPlName(runfiles, \
				'overlay-ff-raw'), norm=False)
			# libplot.overlayMask(flatfield.data, self.ws.saccdpox, \
			#  	self.ws.saccdsize, os.path.join(runfiles['plotdir'], \
			#  	'overlaymask-flatfield-nonopt-crop'), crop=True)
			
			# Optmize mask using a flatfield
			(self.ws.nsa, self.ws.sallpos, self.ws.saccdpos, self.ws.sasize, \
			 	self.ws.saccdsize) = self.ws.optSubapConf(\
				flatfield, self.ws.saccdpos, self.ws.saccdsize, \
				self.ws.saifac, self.ws.ccdscale, self.ws.aptr)
			
			# Plot subaperture mask over flatfield image here
			prNot(VERB_INFO, "parseRun(): making flatfield overlaymask.")
			libplot.overlayMask(flatfield.data, self.ws.saccdpos, \
			 	self.ws.saccdsize, libplot.mkPlName(runfiles, \
			 	'overlay-ff'), norm=False)
			# libplot.overlayMask(flatfield.data, self.ws.saccdpos, \
			#  	self.ws.saccdsize, os.path.join(runfiles['plotdir'], \
			#  	'overlaymask-flatfield-crop'), crop=True)
			
			# TODO: output some text file to the cache or result directory
		
		if (self.dodarkflat):
			prNot(VERB_INFO, "parseRun(): loading dark- and flatfield.")
			# We need darks and flats
			# TODO: add a quality check for the darks and flats here			
			darkfield = WfwfsImg(runfiles['dark'][0][0], imgtype='dark', \
			 	format=self.format)
			# TODO: is this clean enough? Maybe do this in WfwfsImg()?
			darkfield.data /= 1.0*runfiles['dark'][0][2]
			
			# Optimize dark and flat for faster processing. darkimg and 
			# flatimg are simple 2d ndarrays with no metainfo.
			(darkimg, gainimg) = \
				self.ws.optDarkFlat(darkfield, flatfield)
		
		# Save the (optimized) subaperture positions and sizes
		files = saveData(runfiles['cachedir'], 'subap-ccdpos', \
		 	self.ws.saccdpos, asnpy=True, ascsv=True, csvfmt='%d')
		progress['saccdpos'] = files
		files = saveData(runfiles['cachedir'], 'subap-ccdsize', \
		 	self.ws.saccdsize, asnpy=True, ascsv=True, csvfmt='%d')
		progress['saccdsize'] = files
		files = saveData(runfiles['cachedir'], 'subap-llpos', \
		 	self.ws.sallpos, asnpy=True)
		progress['sallpos'] = files
		files = saveData(runfiles['cachedir'], 'subap-llsize', \
		 	self.ws.sallsize, asnpy=True)
		progress['sallsize'] = files
		# Plot subaperture mask over a regular image here
		rawimg = WfwfsImg(os.path.join(runfiles['datadir'], \
		 	runfiles['files'][0]), imgtype='raw', format=self.format)
		if (self.dodarkflat): rawimg.darkFlatField(dark=darkimg, gain=gainimg)
		libplot.overlayMask(rawimg.data, self.ws.saccdpos, \
		 	self.ws.saccdsize, libplot.mkPlName(runfiles, \
		 	'overlay-rawimg'), norm=False)
		# libplot.overlayMask(rawimg.data, self.ws.saccdpos, \
		#  	self.ws.saccdsize, os.path.join(runfiles['plotdir'], \
		#  	'overlaymask-rawimg-crop'), crop=True)
		
		# Plot subaperture layout here
		libplot.showSaSfLayout(libplot.mkPlName(runfiles, \
	 		'sa-layout.eps'), self.ws.saccdpos, self.ws.saccdsize)
		# Plot subap _and_ subfield layout here
		libplot.showSaSfLayout(libplot.mkPlName(runfiles, \
	 		'sa-sf-layout.eps'), self.ws.saccdpos, \
			self.ws.saccdsize, self.ws.sfccdpos, self.ws.sfccdsize)
		
		# Phase 1, static correction
		# ==========================
				
		# Shape of the data we'll receive.
		dshape = (len(runfiles['files']), ) + self.ws.saccdpos.shape
		
		# Check if there are already results available and check if the shape 
		# matches what we expect it to be.
		statshift = loadData(runfiles['cachedir'], 
			'static-shifts', shape=dshape, asnpy=True)
		
		# This is the data-analysis part, run this if we want static shifts 
		# and loading cached data fails.
		if (self.dostatic and statshift is False):
			prNot(VERB_INFO, "parseRun(): measuring static subimg shift.")
			
			# Allocate buffer
			statshift = N.zeros(dshape, dtype=N.float32)
			
			# First broadcast subaperture positions to all workers
			broadcastCB(self.ws.saccdsize, self.TASK_SETSASIZE)
			broadcastCB(self.ws.saccdpos, self.TASK_SETSAPOS)
			broadcastCB(N.array([1]), self.TASK_SETNSAREF)
			
			# DEBUG
			# Get heap information
			hp = guppy.hpy()
			cheap = hp.heap()
			print cheap
			
			frame = -1
			for raw in runfiles['files']:
				frame += 1
				# Print progress
				prNot(VERB_INFO, "parseRun(): P: 1, R: %s, F: %d/%d, " % \
				 	(runfiles['runid'], frame, runfiles['nfiles']))
				prNot(VERB_INFO, "#"*70)
				
				# See if data was already processed:
				statsh = loadData(runfiles['cachedir'], \
					raw + '-static', shape=statshift[frame].shape, \
					asnpy=True)
				if (statsh is not False):
					prNot(VERB_INFO, "parseRun(): Found cache, skipping file")
					statshift[frame] = statsh
					continue
				
				fileuri = os.path.join(runfiles['datadir'], raw)
				#_img = pyana.getdata(fileuri)
				img = WfwfsImg(fileuri, imgtype='raw', format=self.format)
				
				# Check if we need to do darks/flats
				if (self.dodarkflat):
					prNot(VERB_DEBUG, "parseRun(): doing dark-/flatfield.")
					img.darkFlatField(dark=darkimg, gain=gainimg)
					
				prNot(VERB_DEBUG, "parseRun() image stats:")
				prNot(VERB_DEBUG, img.getStats())
				
				# Recv buffer is a slice of the bigger buffer allocated before
				rbuf = statshift[frame]
				sub = submitCB(img.data, raw, self.TASK_SUBIMG, rbuf)
				# If the function did not return True, the job could not be 
				# submitted to a worker and there is probably data to be read
				if (sub != True):
					prNot(VERB_DEBUG, \
						"parseRun(): submit queue full, getting results.")
					(rawid, results) = resultCB()
					prNot(VERB_INFO, "parseRun(): got results '%s'" % (rawid))
					if (rawid is False): break
					prNot(VERB_DEBUG, results[:8])
					saveData(runfiles['cachedir'], rawid + '-static', \
				 		results, asnpy=True)
					# Retry submit, this time it must work
					sub = submitCB(img.data, raw, self.TASK_SUBIMG, rbuf)
			
			
			# Wait for all workers to finish
			prNot(VERB_INFO, "parseRun(): flushing result buffer.")
			while (True):
				(rawid, results) = resultCB()
				if (rawid is False): break
				prNot(VERB_INFO, "parseRun(): got results '%s'" % (rawid))
				prNot(VERB_DEBUG, results[:8])
				saveData(runfiles['cachedir'], rawid + '-static', \
				 	results, asnpy=True)
		
		# This is the data-plotting part, run this if we want static shifts 
		# and we have data (either cached or calculated)
		if (self.dostatic):
			# Process static measurements locally
			prNot(VERB_INFO, "parseRun(): Phase 1 completed, saving data.")
			files = saveData(runfiles['cachedir'], 'static-shifts', \
			 	statshift, asnpy=True)
			progress['statshift'] = files
			prNot(VERB_INFO, statshift.mean(0)[:10])
			prNot(VERB_INFO, statshift.std(0)[:10])
		
			# Make a (nice) plot from the static shifts
			pltit = 'Subimage-shifts for %s \@ %s (zoom: %.3g, #: %d).' % \
		 		(runfiles['datasubdir'], time.ctime(runfiles['begin']), 15, \
		 		runfiles['nfiles'])
			libplot.procStatShift(\
				libplot.mkPlName(runfiles, 'statshift.eps'),\
				statshift, self.ws.saccdpos, self.ws.saccdsize, \
				plrange=self.ws.ccdres, mag=15, allsh=True, title=pltit)
			
			# Analyze shifts and calculate r_0
			statdimmr0 = calcWfsDimmR0(statshift, self.ws.sallpos, \
			 	self.ws.salldiam, self.ws.ccdangscale, mind=2.0, \
			 	wavelen=self.ws.wavelen)
			files = saveData(runfiles['cachedir'], \
			 	'static-shifts-dimm-r0', statdimmr0, asnpy=True)
			progress['statdimmr0'] = files
		
		# Phase 2, subfield measurement
		# =============================
		
		if (self.dosubfield):
			prNot(VERB_DEBUG, "parseRun(): measuring subfield shifts.")
			
			# Optmize subfield positions using the static offsets
			#optSubfPos()
			
			# Loop over files, measure (differential) subfield/subimage shifts
			for raw in runfiles['raw']:
				fileuri = os.path.join(runfiles['datadir'], raw)
				img = WfwfsImg(fileuri, imgtype='raw', format=self.format)
			
			# wait for all workers to finish
			#waitFunc()
			
			# process data
			# procSubfield()
			
		
		# Save list of data files to disk
		saveData(runfiles['resultdir'], runfiles['runid']+'-files', \
			progress, aspickle=True)
		# Done, return
		return True
	
	
	def procFrame(self, args):
		"""
		Given an image frame 'frame' with unique id 'frameid', perform 'task' 
		on it and return the results. This function will run on worker threads 
		and get its data through MPI communication.
		"""
		getfunc, sendfunc = args
		while (True):
			### Get data from master MPI thread
			(data, task) = getfunc()
			
			### These are calculation tasks
			### ===========================
			if (task == self.TASK_SUBIMG):
				prNot(VERB_INFO, "procFrame(): calculating subimage shifts.")
				shvec = libshifts.calcShifts(img=data, \
				 	saccdpos=self.ws.saccdpos, \
					saccdsize=self.ws.saccdsize, \
				 	sfccdpos=N.array([[0.0, 0.0]]), \
					sfccdsize=self.ws.saccdsize, \
					method=libshifts.COMPARE_ABSDIFFSQ, \
					extremum=libshifts.EXTREMUM_2D9PTSQ, \
					refmode=libshifts.REF_BESTRMS, \
					refopt=self.ws.nref, \
					shrange=N.array([7,7]), \
					subfields=None, \
					corrmaps=None)
				#shvec = N.arange(N.product(self.ws.saccdpos.shape), \
				# 	dtype=N.float32)
				shvec.shape = self.ws.saccdpos.shape
				
				# Send out data back to master MPI thread
				sendfunc(shvec)
			
			elif (task == self.TASK_SUBFIELD):
				prNot(VERB_INFO, "procFrame(): calculating subfield shifts.")
				pass
			
			### These are configuration tasks
			### =============================
			
			elif (task == self.TASK_SETSAPOS):
				prNot(VERB_INFO, "procFrame(): setting saccdpos:")
				prNot(VERB_DEBUG, data[:8])
				self.ws.saccdpos = data
			elif (task == self.TASK_SETSASIZE):
				prNot(VERB_INFO, "procFrame(): setting saccdsize:")
				prNot(VERB_DEBUG, data)
				self.ws.saccdsize = data
			elif (task == self.TASK_SETNSAREF):
				prNot(VERB_INFO, "procFrame(): setting # of references:")
				prNot(VERB_DEBUG, data)
				self.ws.nref = data
	
	


#=============================================================================
# WfsSetup(), used for configuration and processing of wfwfs data
#=============================================================================

class WfsSetup():
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
			'sageomfile' : None,\
			'sfgeomfile' : None,\
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
			'saifac' : 0.9
			}
		
		prNot(VERB_INFO, "Initializing WfsSetup()")
		
		# Load configuration from cfgfile
		self.cfg = ConfigParser.SafeConfigParser(self.cfgdef)
		self.cfg.read(cfgfile)
		
		# Save config filename
		self.cfgfile = cfgfile
		
		# Change directory to that of the configfile
		self.curdir = os.path.realpath(os.path.curdir)
		os.chdir(os.path.realpath(os.path.dirname(cfgfile)))
			
		# Parse telescope parameters
		self.aptr = self.cfg.getfloat('telescope', 'aptr')
		self.apts = self.cfg.get('telescope', 'apts')
		self.angle = []
		_ccdresx = self.cfg.getint('telescope', 'ccdresx')
		_ccdresy = self.cfg.getint('telescope', 'ccdresy')
		self.ccdres = N.array([_ccdresx, _ccdresy])
		_fovx = self.cfg.getfloat('telescope', 'fovx')
		_fovy = self.cfg.getfloat('telescope', 'fovy')
		self.fov = N.array([_fovx, _fovy]) * 1/60./60.*(N.pi/180)
		self.ccdscale = self.cfg.getfloat('telescope', 'ccdscale')
		self.ccdangscale = self.cfg.getfloat('telescope', 'ccdangscale')
		# Convert angscale to radian/pixel:
		self.ccdangscale /= 206264.8
		self.wavelen = self.cfg.getfloat('telescope', 'wavelen')
		
		# Parse Shack-Hartmann wavefront sensor parameters
		# Subaperture configuration parameters:
		self.nsa = []
		self.safile = self.cfg.get('shackhartmann', 'safile')
		self.safile = os.path.realpath(self.safile)
		_sallsizex = self.cfg.getfloat('shackhartmann', 'sallsizex')
		_sallsizey = self.cfg.getfloat('shackhartmann', 'sallsizey')
		self.sallsize = N.array([_sallsizex, _sallsizey])
		# Set subaperture diameter as the average diameter
		self.salldiam = (_sallsizex+_sallsizey)/2.0
		_saccdpitchx = self.cfg.getfloat('shackhartmann', 'saccdpitchx')
		_saccdpitchy = self.cfg.getfloat('shackhartmann', 'saccdpitchy')
		self.saccdpitch = N.array([_saccdpitchx, _saccdpitchy])
		_saccdsizex = self.cfg.getfloat('shackhartmann', 'saccdsizex')
		_saccdsizey = self.cfg.getfloat('shackhartmann', 'saccdsizey')
		self.saccdsize = N.array([_saccdsizex, _saccdsizey])
		_saccdevenxoff = self.cfg.getfloat('shackhartmann', 'saccdevenxoff')
		_saccdoddxoff = self.cfg.getfloat('shackhartmann', 'saccdoddxoff')
		self.saccdoff = N.array([_saccdevenxoff, _saccdoddxoff])
		_saccddispx = self.cfg.getfloat('shackhartmann', 'saccddispx')
		_saccddispy = self.cfg.getfloat('shackhartmann', 'saccddispy')
		self.saccddisp = N.array([_saccddispx, _saccddispy])
		self.saccdscl = self.cfg.getfloat('shackhartmann', 'saccdscl')
		self.saifac = self.cfg.getfloat('shackhartmann', 'saifac')
		
		# Subfield configuration parameters:
		_sffile = self.cfg.get('shackhartmann', 'sffile')
		self.sffile = os.path.realpath(_sffile)
		_sftotx = self.cfg.getfloat('shackhartmann', 'sftotx')
		_sftoty = self.cfg.getfloat('shackhartmann', 'sftoty')
		self.sftot = N.array([_sftotx, _sftoty])
		_sfsizex = self.cfg.getfloat('shackhartmann', 'sfsizex')
		_sfsizey = self.cfg.getfloat('shackhartmann', 'sfsizey')
		self.sfsize = N.array([_sfsizex, _sfsizey])
		_sfoffx = self.cfg.getfloat('shackhartmann', 'sfoffx')
		_sfoffy = self.cfg.getfloat('shackhartmann', 'sfoffy')
		self.sfoff = N.array([_sfoffx, _sfoffy])
		_sfpitchx = self.cfg.getfloat('shackhartmann', 'sfpitchx')
		_sfpitchy = self.cfg.getfloat('shackhartmann', 'sfpitchy')
		self.sfpitch = N.array([_sfpitchx, _sfpitchy])
		self.nsf = N.product(self.sftot)
		
		# Check configuration file for sanity
		self.checkSetupSanity()
		
		# Generate initial subaperture mask (positions based on config)
		(self.nsa, self.sallpos, self.saccdpos, self.sallsize, \
			self.saccdsize) = self.initSubaptConf()
		
		# Calculate initial subfield positions and sizes
		(self.nsf, self.sfllpos, self.sfccdpos, self.sfllsize, \
			self.sfccdsize) = self.initSubfieldConf()
		
		# Change directory back
		os.chdir(self.curdir)
	
	
	def checkSetupSanity(self):
		"""
		Sanity check for WfsSetup.
		"""
		
		pass
	
	
	def optDarkFlat(self, darkimg, flatimg):
		"""
		Optimize dark- and flatfields for faster processing: convert dark and
		flat to float32, then calculate gain = 1/(flat-dark) so we can skip 
		this explicit calculation lateron.
		"""
		self.darkimg = darkimg.data.astype(N.float32)
		flatfloat = flatimg.data.astype(N.float32)
		self.gainimg = 1./(flatfloat-self.darkimg)
		
		return (self.darkimg, self.gainimg)
	
	
	def initSubfieldConf(self):
		"""
		Wrapper for calcSubfieldConf() an loadSubfieldConf(). If 
		loadSubfieldConf(safile) returns something (i.e. not False), it will 
		return that output. Otherwise it will return the output of 
		calcSubfieldConf().
		"""
		# Try to load the settings from disk
		rettuple = self.loadSubfieldConf(self.sffile, self.saccdsize, \
			self.ccdscale)
		
		if (rettuple != False):
			prNot(VERB_INFO, "initSubfieldConf(): Using subaperture configuration from file.")
			return rettuple
		else:
			prNot(VERB_INFO, "initSubfieldConf(): Calculating subaperture configuration.")
			rettuple = self.calcSubfieldConf(self.sftot, self.sfsize, \
			 	self.sfpitch, self.sfoff, self.saccdsize, self.ccdscale)
			return rettuple
	
	
	def calcSubfieldConf(self, tot, sfsize, sfpitch, sfoff, saccdsize, ccdscale):
		"""
		Calculate the lower-left subfield (sf) positions and sizes pixels.
		
		Returns:
		(<# of subfields>, <subf pos in SI>, <subf pos in pixels>, 
			<subf size in SI>, <subf size in pixels>)
		
		Parameters:
		'tot': total number of sf's in x and y
		'sfsize': size of the sf's (*binary)
		'sfpitch:' pitch of the sf's (*binary)
		'sfoff': offset of the sf pattern (*binary)
		'saccdsize': the size of a subaperture on the CCD (pixels)
		'ccdscale': the pixel scale of the CCD (meter/pixel)
		
		*binary units mean that if the quantity is positive, it is in units of 
		'saccdsize', if it is positive it is in pixels.
		"""
		
		# sfsize, sfpitch and sfoff can be given in both pixels (if negative) 
		# or in units of the subaperture size (if positive). Fix this first.
		if ((sfsize >= 0).all()):
			sfccdsize = N.round(sfsize * saccdsize).astype(N.int)
		elif ((sfsize < 0).all()):
			sfccdsize = N.round(-1*sfsize).astype(N.int)
		else:
			raise ValueError("calcSubfieldConf(): sfsize is invalid. Check configuration file.")
		
		if ((sfpitch >= 0).all()):
			sfccdpitch = N.round(sfpitch * saccdsize).astype(N.int)
		elif ((sfpitch < 0).all()):
			sfccdpitch = N.round(-1*sfpitch).astype(N.int)
		else:
			raise ValueError("calcSubfieldConf(): sfpitch is invalid. Check configuration file.")
		
		if ((sfoff >= 0).all()):
			sfccdoff = N.round(sfoff * saccdsize).astype(N.int)
		elif ((sfoff < 0).all()):
			sfccdoff = N.round(-1*sfoff).astype(N.int)
		else:
			raise ValueError("calcSubfieldConf(): sfoff is invalid. Check configuration file.")		
		
		# Now calculate the positions of the subfields on the CCD, relative to 
		# the subaperture origin
		sfccdpos = sfccdoff + \
			N.indices(self.sftot, dtype=N.float32).reshape(2,-1).T * \
			sfccdpitch
		sfccdpos = N.round(sfccdpos).astype(N.int)
		# Convert lower-left pixel coordinate to centroid lenslet coordinates
		sfllpos = (sfccdpos - sfccdsize/2.0) * ccdscale
		# Calculate the size of the subfields at the lenslets
		sfllsize = sfccdsize * ccdscale
		# Total number of subfields:
		nsf = len(sfllpos)
		
		prNot(VERB_INFO, \
			"calcSubfieldConf(): sfllsize: (%.3g,%.3g), sfccdsize: (%d,%d)"% \
			(sfllsize[0], sfllsize[1], sfccdsize[0], sfccdsize[1]))
		prNot(VERB_INFO, \
			"calcSubfieldConf(): sfccdoff: (%d,%d)" % \
			(sfccdoff[0], sfccdoff[1]))
		prNot(VERB_INFO, \
			"calcSubfieldConf(): sfccdpitch: (%d,%d)"% \
			(sfccdpitch[0], sfccdpitch[1]))
		
		return (nsf, sfllpos, sfccdpos, sfllsize, sfccdsize)
	
	
	def loadSubfieldConf(self, sffile, saccdsize, ccdscale):
		"""
		Try to load 'sffile', if it exists. This should hold information on 
		the subfield positions and sizes on the CCD (in pixels). Positions 
		should be the *lower-left corner* of the subfields. Syntax of the file 
		should be:
		line 1: <INT number of subfields>
		line 2: <FLOAT xsize [m]> <FLOAT ysize [m]>
		line 3: <INT xsize [pix]> <INT ysize [pix]>
		line 4--n: <INT subfield n xpos [pix]> <INT subfields n ypos [pix]>
				
		The pixel position and sizes will be converted to units of subaperture 
		size using 'saccdsize' and 'ccdscale'.
		
		Returns:
		If something goes wrong (file does not exist, parsing does not work, 
		something else), this function will return False. Otherwise it will 
		return the tuple:
		(<# of subfields>, <subf pos in SI>, <subf pos in pixels>, 
			<subf size in SI>, <subf size in pixels>)
		"""
		
		# See if the file exists
		if (not os.path.isfile(sffile)):
			prNot(VERB_WARN, \
				"loadSubfieldConf(): File '%s' does not exist." % (sffile))
			return False
		
		# Try to load and parse the file
		reader = csv.reader(open(sffile), delimiter=',')
		
		try: 
			# Number of subapertures [int]
			line = reader.next()
			nsf = int(line[0])
			# Subfield size at the aperture [float, float]
			line = reader.next()
			sfllsize = N.array([float(line[0]), float(line[1])])			
			# Subfield pixel size should be [int, int]
			line = reader.next()
			sfccdsize = N.array([int(line[0]), int(line[1])])
		except: 
			# If *something* went wrong, abort
			prNot(VERB_WARN, \
				"loadSubfieldConf(): Could not parse file header.")
			return False
		
		# Init sapos list
		sfccdpos = []
		# Try to read the the subfield positions
		for line in reader:
			try:
				sfpos = [int(line[0]), int(line[1])]
				sfccdpos.append(sfpos)
			except:
				prNot(VERB_WARN, "loadSubfieldConf(): Could not parse file")
				return False
		
		prNot(VERB_DEBUG, \
			"loadSubfieldConf(): Found %d subfields, (expected %d)."% \
			 (len(sfccdpos), nsf))
		
		if (len(sfccdpos) != nsf):
			prNot(VERB_WARN, "loadSubfieldConf(): Found %d subfields, expected %d. Using all positions found (%d)." % (len(sfccdpos), nsf, len(sfccdpos)))
			nsf = len(sfccdpos)
		
		# Convert to numpy array
		sfccdpos = N.array(sfccdpos)
		
		# Convert ccd pixel positions to positions at the aperture
		sfllsize = sfccdsize * ccdscale
		sfllpos = (sfccdpos + saccdsize/2.0) * ccdscale
				
		return (nsf, sfllpos, sfccdpos, sfllsize, sfccdsize)
	
	
	def saveSaSfConf(self, sfile, n, llsize, ccdsize, ccdpos):
		"""
		Save the subaperture or subfield configuration to 'sfile'. The syntax 
		of the subaperture and subfield configuration is the same, so we can 
		use the same function to save the data.
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
		writer.writerow(llsize)
		# Write ccdsize
		writer.writerow(ccdsize)
		# Write all ccd positions
		for pos in ccdpos:
			writer.writerow(pos)
		
		# Done
	
		
	def initSubaptConf(self):
		"""
		Wrapper for calcSubaptConf() and loadSubaptConf(). If 
		loadSubaptConf(self.safile) returns something (i.e. not False), it 
		will return that output. Otherwise it will return the output of 
		calcSubaptConf(). Both return the following tuple:
		(<# of subaps>, \
			<subap pos on lenslet in SI>, <subap pos on CCD in pixels>, 
			<subap size on lenslet in SI>, <subap size on CCD in pixels>)
		"""
		
		# Try to load the settings from disk
		rettuple = self.loadSubaptConf(self.safile, self.ccdscale, self.aptr)
		
		if (rettuple is not False):
			prNot(VERB_INFO, "initSubaptConf(): Using subaperture configuration from file.")
			return rettuple
		else:
			prNot(VERB_INFO, "initSubaptConf(): Calculating subaperture configuration.")
			(nsa, sallpos, saccdpos) = self.calcSubaptConf(\
				self.ccdres[0]/2., self.apts, self.saccdsize, \
				self.saccdpitch, self.saccdoff, self.saccddisp, \
				 self.saccdscl, self.ccdscale)
			return (nsa, sallpos, saccdpos, self.sallsize, \
			 	N.round(self.saccdsize).astype(N.int))
	
	
	def loadSubaptConf(self, safile, ccdscale, aptr):
		"""
		Try to load 'safile', if it exists. This should hold information on 
		the subaperture positions and sizes on the CCD. Positions should be 
		the lower-left corner of the subaperture. Syntax of the file should 
		be:
		line 1: <INT number of subapertures>
		line 2: <FLOAT xsize [m]> <FLOAT ysize [m]>
		line 3: <INT xsize [pix]> <INT ysize [pix]>
		line 4--n: <INT subap n xpos [pix]> <INT subap n ypos [pix]>
		
		To calculate the SI centroid positions from the pixel positions, this 
		function uses 'ccdscale' which should be the pixel scale in [meter at 
		the aperture/pixel].
		
		If something goes wrong (file does not exist, parsing does not work, 
		something else), this function will return False. Otherwise it will 
		return
		(<# of subaps>, \
			<subap pos on lenslet in SI>, <subap pos on CCD in pixels>, 
			<subap diameter at pupil in SI>, <subap size on CCD in pixels>)
		"""
		
		# See if the file exists
		if (not os.path.isfile(safile)):
			prNot(VERB_WARN, "loadSubaptConf(): File '%s' does not exist." %\
				(safile))
			return False
		
		# Try to load and parse the file
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
			# If *something* went wrong, abort
			prNot(VERB_WARN, "loadSubaptConf(): Could not parse file header.")
			return False
		
		# Init sapos list
		saccdpos = []
		# Try to read the the subaperture positions
		for line in reader:
			try:
				_pos = [int(line[0]), int(line[1])]
				saccdpos.append(_pos)
			except:
				prNot(VERB_WARN, "loadSubaptConf(): Could not parse file.")
				return False
		
		prNot(VERB_DEBUG, \
			"loadSubaptConf(): Found %d subaps, (expected %d)."% \
			 (len(saccdpos), nsa))
		
		if (len(saccdpos) != nsa):
			prNot(VERB_WARN, "loadSubaptConf(): Found %d subaps, expected %d. Using all positions found (%d)." % (len(saccdpos), nsa, len(saccdpos)))
			nsa = len(saccdpos)
		
		# Convert to numpy array
		saccdpos = N.array(saccdpos)
		
		# Convert to aperture positions as centroid coordinates:
		sallpos = (saccdpos + saccdsize/2) * ccdscale + aptr
		# Calculate subaperture diameter:
		sadiam = sallsize.mean()
		
		return (nsa, sallpos, saccdpos, sallpos, saccdsize)
	
	
	def calcSubaptConf(self, rad, shape, size, pitch, xoff, disp, scl, pixscl):
		"""
		Generate positions for a subaperture (sa) mask. 
		
		Parameters:
		'rad': radius of the sa pattern (before scaling) (in pixels)
		'shape': shape of the sa pattern
		'size': size of the sa's (in pixels)
		'pitch': pitch of the sa's (in pixels)
		'xoff': the horizontal position offset of odd rows (in units of
			'size')
		'disp': global displacement of the sa positions (in pixels)
		'scl': global scaling of the sa positions (in pixels)
		'pixscl': pixel scale on the CCD (in meters/pixel)
				
		Returns:
		A tuple with the following values:
		(<# of subaps>, \
			<subap pos on lenslet>, <subap pos on CCD>, 
			<subap size on lenslet>, <subap size on CCD>)
			
		The CCD position and size are in pixels. The lenslet position and size 
		are in meters.
		
		Raises ValueError if shape is unknown.
		"""
		
		print "rad: %g" % rad
		
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
				
				#print "Win position (%.3g, %.3g)" % (sac[0], sac[1])
				
				# 'say % 2' gives 0 for even rows and 1 for odd rows. Use this 
				# to apply a row-offset to even and odd rows
				# If we're in an odd row, check saccdoddoff
				sac[0] -= xoff[say % 2] * pitch[0]
				
				# Check if we're in the apterture bounds, and store the subapt
				# position in that case
				if (shape == 'circular'):
					if (sum((abs(sac)+size/2.0)**2) < rad**2): 
						pos.append(sac)
						print "Win position (%.3g, %.3g)" % (sac[0], sac[1])
				elif shape == 'square':
					if (abs(sac)+size/2.0 < rad).all:
						pos.append(sac)
						print "Win position (%.3g, %.3g)" % (sac[0], sac[1])
				else:
					raise ValueError("Unknown aperture shape", apts, \
						"(should be 'circular' or 'square')")
		
		if (len(pos) <= 0):
			raise RuntimeError("Didn't find any subapertures for this configuration.")
		
		# Apply scaling and displacement to the pattern before returning
		# NB: pos gives the *centroid* position of the subapertures
		pos = (N.array(pos) * scl) + disp
		
		# Convert symmetric centroid positions to CCD positions:
		saccdpos = N.round(pos + rad - size/2.0).astype(N.int)
		# Convert centroid positions in pixels to meters:
		sallpos = pos * pixscl
		
		nsa = len(saccdpos)
		prNot(VERB_INFO, "calcSubaptConf(): found %d subapertures." % (nsa))
		
		return (nsa, sallpos, saccdpos)
	
	
	def optSubapConf(self, img, saccdpos, saccdsize, saifac, ccdscale, aptr):
		"""
		Optimize subaperture mask position using a flatfield image.
		
		To optimize the subaperture pattern, take the initial origin positions 
		of subapertures, find the origin of the subaperture, and cut a 
		horizontal an vertical slice of pixels out the image which are twice 
		as large as the size of the subaperture.
		
		These two slices should then give an intensity profile across the 
		subimage, and since there is a dark band between the subimages, the 
		dimensions of each of these can be determined by finding the minimum 
		intensity in the slices.
		
		'img' should be of the type WfwfsImg().
		'saccdpos' should be a list of lower-left pixel positions of the 
			subapertures
		'saccdsize' should be a list of the pixel size of the subapertures
		'saifac' should be the intensity reduction factor counting as 'dark'
		'ccdscale' should be the CCD image scale in [meter/pixel]
		'aptr' should be the radius of the aperture in meter
		
		This function returns a tuple like calcSubaptConf() does:
		(<# of subaps>, \
			<subap pos on lenslet in SI>, <subap pos on CCD in pixels>, 
			<subap size on lenslet in SI>, <subap size on CCD in pixels>)
		
		This methods is slightly sensitive to specks of dust on flatfields.
		"""
		
		# Init optimium position and size variables
		optsapos = []			# Store the optimized subap position
		optsaccdpos = []		# Store the optimized subap position in pixels
		allsaccdsize = []		# Stores all optimized subaperture sizes
		
		prNot(VERB_INFO, "optSubapConf(): Optimizing subap mask.")
		
		# Check if the image is a flatfield
		if (img.imgtype != 'flat'):
			raise RuntimeWarning("Optmizing the subimage mask works best with flatfields")
		
		# Loop over all subapertures
		for pos in saccdpos:
			# Calculate the ranges for the slices (make sure we don't get
			# negative indices and stuff like that). The position (saccdsize)
			# is the origin of the subap. This means we take origin + 
			# width*1.5 pixels to the right and origin - width*0.5 pixels to 
			# the left to get a slice across the subap. Same for height.
			## SAORIGIN
			slxran = N.array([max(0, pos[0]-saccdsize[0]*0.5), \
				min(img.res[0], pos[0]+saccdsize[0]*1.5)])
			slyran = N.array([max(0, pos[1]-saccdsize[1]*0.5), \
				min(img.res[1], pos[1]+saccdsize[1]*1.5)])
			
			# Get two slices in horizontal and vertical direction. Slices are 
			# the same width and height as the subapertures are estimated to
			# be, then averaged down to a one-pixel profile
			# NB: image indexing goes reverse: pixel (x,y) is at data[y,x]
			## SAORIGIN
			xslice = img.data[pos[1]:pos[1]+saccdsize[1], \
				slxran[0]:slxran[1]]
			xslice = xslice.mean(axis=0)
			
			yslice = img.data[slyran[0]:slyran[1], \
				pos[0]:pos[0]+saccdsize[0]]
			yslice = yslice.mean(axis=1)
			# Find the first index where the intensity is lower than saifac 
			# times the (maximum intensity - minimum intensity) in the slices 
			# *in slice coordinates*.
			slmax = N.max([xslice.max(), yslice.max()])
			# TODO: this should work better, but doesn't. Why?
			#slmin = N.min([xslice.min(), yslice.min()])
			#cutoff = (slmax-slmin)*saifac
			cutoff = slmax * saifac
			saxran = N.array([ \
				N.argwhere(xslice[:slxran.ptp()/2.] < \
				 	cutoff)[-1,0], \
				N.argwhere(xslice[slxran.ptp()/2.:] < \
					cutoff)[0,0] + \
				 	slxran.ptp()/2. ])
			sayran = N.array([ \
				N.argwhere(yslice[:slyran.ptp()/2.] < \
					cutoff)[-1,0], \
				N.argwhere(yslice[slyran.ptp()/2.:] < \
					cutoff)[0,0] + \
					slyran.ptp()/2. ])
			
			# The size of the subaperture is sa[x|y]ran[1] - sa[x|y]ran[0]:
			_sass = N.array([saxran.ptp(), sayran.ptp()])
			
			# The final origin pixel position in the large image (img.data) of 
			# the subaperture is the position we found in the slice
			# (saxran[0], sayran[0]),  plus the coordinate where the slice 
			# began in the big dataset (slxran[0], slyran[0]).
			## SAORIGIN
			# To get the centroid position: add half the size of the subimage
			_saccdpos = N.array([saxran[0] + slxran[0], \
				sayran[0] + slyran[0]])
			_sapos = (_saccdpos + _sass/2.) * ccdscale - aptr
			
			prNot(VERB_DEBUG, "optSubapConf(): subap@(%d, %d), size: (%d, %d), pos: (%d, %d)" % \
				(pos[0], pos[1], _sass[0], _sass[1], _saccdpos[0], \
				 _saccdpos[1]))
			prNot(VERB_DEBUG, "optSubapConf(): ranges: (%d,%d) and (%d,%d)"% \
				(slxran[0], slxran[1], slyran[0], slyran[1]))
			
			# Save positions as pixel and real coordinates
			optsaccdpos.append(_saccdpos)
			optsapos.append(_sapos)
			
			# The subimage size should be the same for all subimages. Store 
			# all subaperture sizes found during looping and then take the
			# mean afterwards.
			allsaccdsize.append(_sass)
		
		# Calculate the average optimal subaperture size in pixels
		optsaccdsize = N.array(allsaccdsize).mean(axis=0)
		# And the standard deviation
		tmpstddev = (N.array(allsaccdsize)).std(axis=0)
		
		# Calculate optimum size in SI units as well
		optsasize = optsaccdsize * ccdscale
		optsapos = N.array(optsapos)
		
		# Round the pixel position and sizes
		optsaccdsize = N.round(optsaccdsize).astype(N.int)
		optsaccdpos = N.round(optsaccdpos).astype(N.int)
		
		prNot(VERB_INFO, "optSubapConf(): subimage size optimized to (%d,%d), stddev: (%.3g, %.3g) (was (%.3g, %.3g))" % \
			(optsaccdsize[0], optsaccdsize[1], tmpstddev[0], tmpstddev[1], \
			self.saccdsize[0], self.saccdsize[1]))
		
		# Init a binary mask that will show where the subimages are
		self.mask = N.zeros(self.ccdres, dtype=N.uint8)
		self.gridmask = N.zeros(self.ccdres, dtype=N.uint8)
		
		for optpos in optsaccdpos:	
			# Now make a mask (0/1) for all subapertures (again, remember 
			#image indexing is 'the wrong' way around in NumPy (pixel (x,y) is 
			# at img[y,x]))
			## SAORIGIN
			self.mask[ \
				optpos[1]: \
			 	optpos[1]+optsaccdsize[1], \
				optpos[0]: \
				optpos[0]+optsaccdsize[0]] = 1
			# The gridmask is only 1 at the edges...
			self.gridmask[\
				optpos[1]: \
			 	optpos[1]+optsaccdsize[1], \
				optpos[0]: \
				optpos[0]+optsaccdsize[0]] = 1
			# ... and zero everywhere else
			self.gridmask[\
				optpos[1]+1: \
			 	optpos[1]+optsaccdsize[1]-1, \
				optpos[0]+1: \
				optpos[0]+optsaccdsize[0]-1] = 0
		
		# Convert mask to bool for easier array indexing lateron
		self.mask = self.mask.astype(N.bool)
		
		# Return a tuple like calcSubaptConf() does:
		return (len(optsaccdpos), optsapos, optsaccdpos, \
			optsasize,  optsaccdsize)
	
	


#=============================================================================
# WfwfsImg(), used for loading and (pre-)processing WFWFS images
#=============================================================================

class WfwfsImg():
	"""
	Class to hold information on a WFWFS image (dark, flat, raw or processed). 
	Has methods to load and save files to various formats, and to do some 
	basic processing.
	"""
	
	def __init__(self, filepath, imgtype, format):
		# The filename
		self.filename = os.path.basename(filepath)
		# Complete path where the file is stored
		self.uri = os.path.normpath(filepath)
		# Directory containing the file
		self.dir = os.path.dirname(filepath)
		
		# Type of data (dark, flat, raw, corrected, cropped)
		self.imgtype = imgtype
		# Format of the data on disk (ana, fits)
		self.format = format
		# Location to the data itself
		self.data = []
		# Resolution
		self.res = ()
		# Bits per pixel
		self.bpp = -1
		
		# Some class specific variables
		# Supported file formats
		self._formats = ['ana', 'fits']
		# Methods to load the file formats and file meta data
		self._formatload = {'ana': self._anaload, 'fits': self._fitsload}
		# Supported image types
		self._imgtypes = ['raw', 'dark', 'flat', 'corrected']
		
		# Finally load the image
		self.load()
	
	
	def load(self):
		"""
		Load an image file from disk into memory.
		"""
		
		prNot(VERB_DEBUG, "load(): Trying to load file '%s'" % (self.uri))
		
		# Check if file exists
		if (not os.path.isfile(self.uri)):
			raise IOError("Cannot find file '%s'" % (self.uri))
		
		# Check if imagetype is ok
		if (not self.imgtype in self._imgtypes):
			raise RuntimeError("Image type '%s' not supported" % \
			 	(self.imgtype))
		
		# Check if format is supported, and load if possible
		if (self.format in self._formats):
			self._formatload[self.format](self.uri)
		
		else:
			raise TypeError("File format '%s' not supported" % (self.format))
		
	
	
	def darkFlatField(self, dark, gain=None, flat=None):
		"""
		Dark and flatfield an image. This function can use either a 
		combination of dark and gain, or a dark and flatfield. Dark and gain 
		is recommended as it is faster.
		"""
		if (gain != None):
			self.data = (self.data.astype(N.float32) - dark) * gain
		elif (flat != None):
			self.data = (self.data.astype(N.float32) - dark) / (flat-dark)
		else:
			raise RuntimeError("Cannot dark/flatfield without (darkfield && (flatfield || gainfield))")
		
		self.imgtype = 'corrected'
	
	
	def maskSubimg(self, mask, whitebg=False):
		"""
		Mask out everything but the subimages by multiplying the data with a 
		binary mask, then scaling the pixels to 0--1 within the data that is 
		left.
		"""
		raise RuntimeError("Deprecated")
		
		mdata = self.data[mask]
		offset = mdata.min()
		gain = 1./(mdata.max()-offset)
		self.data = (self.data-offset) * gain * mask
		
		# If we want a white background, we don't set the other parts to 0,
		# but the the maximum intensity in the image
		if (whitebg):
			self.data += (mask == 0) * self.data.max()
		
		self.imgtype = 'masked'
	
	
	def getStats(self):
		"""
		Get min, max and RMS from data
		"""
		
		dmin = N.min(self.data)
		dmax = N.max(self.data)
		drms = (N.mean(self.data**2.0))**0.5
		return (dmin, dmax, drms)
	
	
	def fitsSave(self, filename, dtype=N.float32):
		"""
		Save an image as fits file. File will be stored in the same directory 
		as where the file was read. 'dtype' can be used to force a datatype to 
		store the file in, will default to N.float32.
		"""
		
		prNot(VERB_DEBUG, "fitsSave(): Trying to save image as FITS file")
		
		# Init the header with basic metadata
		hdu = pyfits.PrimaryHDU(self.data.astype(dtype))
		
		# Add some header info related to what we're doing
		hdu.header.update('origin', 'WFWFS Data')
		hdu.header.update('origname', self.filename)
		hdu.header.update('origdir', self.dir)
		hdu.header.update('type', self.imgtype)
		
		# Write to disk
		hdu.writeto(os.path.join(self.wfwfs.outdir, filename))
		
		# Flush the header thing to be sure that we've written everything.
		hdu.flush()
	
	
	def pngSave(self, filename):
		"""
		Save image as PNG, always rescaling the data to 0--255 and saving it
		as grayscale image.
		"""
		prNot(VERB_DEBUG, "pngSave(): Trying to save image as PNG file.")
		
		scldat = (self.data - self.data.min())*255 / \
			(self.data.max() - self.data.min())
		surf = cairo.ImageSurface.create_for_data(scldat.astype(N.uint8), \
		 	cairo.FORMAT_A8, self.res[0], self.res[0])
		cairo.ImageSurface.write_to_png(surf, \
			os.path.join(self.dir, filename))
	
	
	def _anaload(self, filename):
		"""
		Wrapper for loading ana f0/fz files.
		"""
		
		prNot(VERB_DEBUG, "_anaload(): Trying to load ana file.")
		if (VERBOSITY >= VERB_DEBUG): db=1
		else: db=0
		self.data = pyana.getdata(filename, debug=db)
		self.res = self.data.shape
		self.bpp = self.data.nbytes / N.product(self.res)
			
		prNot(VERB_DEBUG, "_anaload(): %d bytes, %d-d, %d elem." % \
			(self.data.nbytes, len(self.data.shape), N.product(self.res)))
	
	
	def _fitsload(self, filename):
		"""
		Wrapper for loading fits files
		"""
		prNot(VERB_DEBUG, "_fitsload(): Trying to load fits file")
		
		# Load data
		self.data = pyfits.getdata(filename)
		# Convert datatype to nice numpy.<type> datatypes
		self.data = N.array(self.data, dtype=self.data.dtype.type)
		# Fix some data properties
		self.res = self.data.shape
		self.bpp = self.data.nbytes / N.product(self.res)
		
		prNot(VERB_DEBUG, "_fitsload(): %d bytes, %d-d, %d elem." % \
			(self.data.nbytes, len(self.data.shape), N.product(self.res)))
	
	
	def __del__(self):
		"""
		'Destructor', trying to fix memory leaks here :P
		"""
		
		del self.data
	


#=============================================================================
# Helper routines, not associated with any class
#=============================================================================

def saveOldFile(uri, postfix='.old', maxold=5):
	"""
	If 'uri' is present, rename the file to prevent it being overwritten. 
	The filename will be to 'uri' + postfix + the lowest integer that 
	constitutes a non-existing file. 'maxold' specifies how many old files 
	we should keep.
	"""
	
	if (os.path.exists(uri)):
		app = 0
		while (os.path.exists(uri + postfix + str(app))):
			app += 1
			if (app >= maxold-1): break
			
		# Now rename uri+postfix+str(app-1) -> uri+postfix+str(app)
		# NB: range(app-1, -1, -1) for app = 5 gives [4, 3, 2, 1, 0]
		for i in range(app-1, -1, -1):
			os.rename(uri+postfix+str(i), uri+postfix+str(i+1))
			
		# Now rename the original file to originalfile + '.old0':
		os.rename(uri, uri+postfix+str(0))
		
		# the file 'uri' is now free
		prNot(VERB_DEBUG, \
			"saveOldFile(): renaming file to prevent overwriting")


def loadData(path, id, asnpy=False, aspickle=False, shape=None):
	"""
	Reverse function of saveData(): load data stored on disk to prevent 
	re-computation of the analysis. Formats supported are numpy arrays 
	(enable with 'asnpy') and pickled files (enable with 'aspickle'). File
	URI will be dirname + file + '.npy'/'.pickle'. If both 'asnpy' and 
	'aspickle' are True, numpy will be preferred.
	
	If shape is set, it will be verified that the returned array is indeed 
	that shape.
	
	Return value is a tuple of (flag, data), with the bool 'flag' 
	indicating whether or not data has been found and 'data' carrying the 
	data -- which can be of any type. If 'flag' is False, data is 
	set to False.
	
	TODO: add ascsv
	"""
	
	prNot(VERB_DEBUG, "loadData(): id '%s' asnpy: %d aspickle: %d" % \
		(id, asnpy, aspickle))
	
	# Make sure there is only one setting true
	if (asnpy and aspickle):
		aspickle = False
		prNot(VERB_WARNING, "loadData(): Cannot load more than one format at a time, disabling pickle")
	elif (not asnpy and not aspickle):
		asnpy = True
		prNot(VERB_WARNING, "loadData(): No format selected, enabling npy")
	
	if (asnpy):
		uri = os.path.join(path, id) + '.npy'
		# Check if file exists
		if (not os.path.isfile(uri)):
			prNot(VERB_WARN, \
				"loadData(): numpy file does not exists, continuing.")
			return False
			
		# Load results
		results = N.load(uri)
	
	if (aspickle):
		uri = os.path.join(path, id) + '.pickle'
		# Check if file exists
		if (not os.path.isfile(uri)):
			prNot(VERB_WARN, \
				"loadData(): pickle file does not exists, continuing.")
			return False
			
		# Load results
		results = N.load(uri)
	
	# Check if shape matches
	if (shape != None and results.shape != shape):
		prNot(VERB_WARN, "loadData(): shapes do not match, continuing.")
		return False
		
	return results


def saveData(path, id, data, asnpy=False, aspickle=False, ascsv=False, csvfmt='%.18e'):
	"""
	Save (intermediate) results to 'path' with file ID 'id'. Data can be 
	stored as numpy array (if asnpy is True), csv file (if ascsv is True)		
	and/or pickled format (if aspickle is True). The final path will be
	os.path.join(path,id) + '.npy'/'.csv'/'.pickle', for the different
	formats.
	
	Parameters:
	'csvfmt': the format for storing data as CSV ['%.18e']
	
	Returns:
	A dict of files the data was saved to when successful. The keys will 
	be one or more of 'npy', 'pickle' or 'csv' and the values will be the 
	full file paths. Returns False when something failed.
	"""
	# Init empty list
	flist = {}
	
	prNot(VERB_DEBUG, "saveData(): id '%s' npy: %d pickle: %d, csv: %d" %\
		(id, asnpy, aspickle, ascsv))
	
	# Make dir if necessary
	if (not os.path.isdir(path)):
		os.makedirs(path)
	
	# If everything is False, enable asnpy
	if (not asnpy and not aspickle and not ascsv):
		asnpy = True
	
	if (asnpy):
		# Save data in numpy format
		uri = os.path.join(path, id) + '.npy'
		# Save old file, if present
		saveOldFile(uri, postfix='.old', maxold=3)
		try:
			N.save(uri, data)
			flist['npy'] = uri
		except:
			return False
	if (ascsv):
		# Save data in numpy format
		uri = os.path.join(path, id) + '.csv'
		# Save old file, if present
		saveOldFile(uri, postfix='.old', maxold=3)
		try:
			N.savetxt(uri, data, fmt=csvfmt)
			flist['csv'] = uri
		except:
			return False			
	if (aspickle):
		uri = os.path.join(path, id) + '.pickle'
		# Save old file, if present
		saveOldFile(uri, postfix='.old', maxold=3)
		try:
			cPickle.dump(data, file(uri, 'w'))
			flist['pickle'] = uri
		except:
			return False
	
	return flist
	# done


#=============================================================================
# Prototype functions, functions without a clear home
#=============================================================================

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
		prNot(VERB_WARN, "calcDimmR0(): shifts incorrect shape, should be 3D")
		return False
	if (shifts.shape[1] != sapos.shape[0]):
		prNot(VERB_WARN, "calcDimmR0(): shifts and sapos shapes do not match")
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


def restoreCache(path, metafile):
	"""
	Load 'metafile', which should hold a dict. For all entries in the dict 
	with a 'npy' key, load the value of that key.
	"""
	# This will hold the data
	data = {}
	# Load the metafile
	filelist = loadData(path, metafile, aspickle=True)
	if (filelist is False):
		prNot(VERB_WARN, "restoreCache(): Failed to load filelist.")
		return False
	
	# Loop over all quantities in filelist
	for quant in filelist:
		# Try to load the file stored at the 'npy' key in the dict, if it 
		# exists
		try:
			uri = filelist[quant]['npy']
			data[quant] = N.load(uri)
		except:
			pass
	
	return (filelist, data)

