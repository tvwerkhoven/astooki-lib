#!/usr/bin/env python2.5
# encoding: utf-8
"""
@file libfile.py
@brief Library for file I/O
@author Tim van Werkhoven (tim@astrou.su.se)
@date 20090424

Created by Tim van Werkhoven on 2009-04-24.
Copyright (c) 2008-2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""

#=============================================================================
# Import libraries here
#=============================================================================

import os
import liblog as log
import time

#=============================================================================
# Local helper functions
#=============================================================================

def _anaload(path):
	"""
	Load ana file using pyana.
	@param path File to load
	@return Data if successful, False otherwise.
	"""
	import pyana
	try: data = pyana.getdata(path)
	except: return False
	
	return data

def _fitsload(path):
	"""
	Load fits file using pyfits.
	@param path File to load
	@return Data if successful, False otherwise.
	"""
	import pyfits
	try: data = pyfits.getdata(path)
	except: return False
	return data

def _npyload(path):
	"""
	Load npy file.
	@param path File to load
	@return Data if successful, False otherwise.
	"""
	import numpy
	try: data = numpy.load(path)
	except: return False
	return data

def _pickleload(path):
	"""
	Load pickled file.
	@param path File to load
	@return Data if successful, False otherwise.
	"""
	import cPickle
	try: data = cPickle.load(open(path))
	except: return False
	return data


#=============================================================================
# Definitions necessary for processing
#=============================================================================

_FORMAT_ANA = 'ana'
_FORMAT_FITS = 'fits'
_FORMAT_PNG = 'png'
_FORMAT_NPY = 'npy'
_FORMAT_CSV = 'csv'
_FORMAT_PICKLE = 'pickle'
_FORMATS_LOAD = [_FORMAT_PICKLE, _FORMAT_FITS, _FORMAT_ANA, _FORMAT_NPY]
_FORMATS_LOAD_F = {_FORMAT_ANA: _anaload, _FORMAT_FITS: _fitsload, _FORMAT_PICKLE: _pickleload, _FORMAT_NPY: _npyload}

#=============================================================================
# Data storage functions
#=============================================================================

def saveOldFile(uri, postfix='.old', maxold=5):
	"""
	If 'uri' is present, rename the file to prevent it being overwritten. 
	The filename will be to 'uri' + postfix + the lowest integer that 
	constitutes a non-existing file. 'maxold' specifies how many old files 
	we should keep.
	
	@param uri Filepath to prevent overwriting to
	@param postfix Suffix to add to the file before a counter
	@param maxold Keep this many old files maximum
	
	@return Nothing
	"""
	
	if (maxold == 0): return 
	
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
		log.prNot(log.INFO, "saveOldFile(): renamed file (%s) to prevent overwriting" % (uri))

	
def loadData(path, asnpy=False, aspickle=False, ascsv=False, auto=False, shape=None):
	"""
	Reverse function of saveData(): load data stored on disk. Formats supported 
	are numpy arrays, pickled files and csv files. If multiple formats are 
	enabled, numpy will be preferred. If shape is set, it will be verified that
	the returned array is indeed that shape. If not, it will return False.
	
	@param path Path to the datafile
	@param asnpy Load data as numpy format
	@param aspickle Load data as pickle format
	@param ascsv Load data as csv format
	@param auto Try to guess the filetype from extension + contents (TODO)
	@param shape Shape that the data stored in the file should have
	
	@return Data array, or False when loading failed.
	"""
	import numpy as N
	log.prNot(log.INFO, "loadData(): asnpy: %d aspickle: %d, ascsv: %d" % \
		(asnpy, aspickle, ascsv))
	
	if (N.sum([asnpy, aspickle, ascsv]) > 0 and auto):
		log.prNot(log.WARNING, "loadData(): auto-guessing set but .")
	# Make sure there is only one setting true
	if (N.sum([asnpy, aspickle, ascsv]) > 1):
		aspickle = False
		ascsv = False
		log.prNot(log.WARNING, "loadData(): Cannot load more than one format at a time, disabling pickle.")
	elif (N.sum([asnpy, aspickle, ascsv]) < 1 and not auto):
		asnpy = True
		log.prNot(log.WARNING, "loadData(): No format selected, enabling npy.")
	
	if (asnpy):
		import numpy as N
		#uri = path + '.npy'
		# Check if file exists
		if (not os.path.isfile(path)):
			log.prNot(log.WARNING, "loadData(): numpy file '%s' does not exists."%\
					(os.path.split(path)[1]))
			return False
			
		# Load results
		results = N.load(path)
	
	if (aspickle):
		import cPickle as pickle
		
		#uri = path + '.pickle'
		# Check if file exists
		if (not os.path.isfile(path)):
			log.prNot(log.WARNING, "loadData(): pickle file '%s' does not exists."%\
					(os.path.split(path)[1]))
			return False
			
		# Load results
		results = pickle.load(open(path))
	
	if (ascsv):
		#uri = path + '.pickle'
		# Check if file exists
		if (not os.path.isfile(path)):
			log.prNot(log.WARNING, "loadData(): pickle file '%s' does not exists."%\
					(os.path.split(path)[1]))
			return False
			
		# Load results as csv
		results = N.loadtxt(path, delimiter=',')
	
	# Check if shape matches
	if (shape is not None and results.shape != shape):
		log.prNot(log.WARNING, "loadData(): shapes do not match.")
		return False
		
	return results


def saveData(path, data, asnpy=False, aspickle=False, asfits=False, ascsv=False, explicit=False, csvfmt='%g', csvhdr=None, old=3):
	"""
	Save (intermediate) results to 'path'. Data can be 
	stored as numpy array (if asnpy is True), csv file (if ascsv is True), FITS 
	file (asfits) and/or pickled format (if aspickle is True). The final path 
	will be path + '.npy'/'.csv'/'.fits'/'.pickle', for the different formats.
	
	@param path Base path to store files to. Should be dir+filename
	@param data Some numpy formatted data.
	@param asnpy Store data in numpy format to path+'.npy' [False]
	@param aspickle Store data in pickle format to path+'.pickle' [False]
	@param asfits Store data in FITS format to path+'.fits' [False]
	@param ascsv Store data in csv format to path+'.csv' [False]
	@param explicit Store the data exactly to 'path', don't append suffix(es)
	@param csvfmt The format for storing data as CSV ['%.18e']
	@param csvhdr A header to write to the head of the file. Header should nbe a 
	              list with the same number of elements as elements in 'data' 
	              [None]
	@param old Backup this many pre-existing files at maximum
	
	@return A dict of files the data was saved to when successful. The keys will 
	be one or more of 'npy', 'pickle' or 'csv' and the values will be the 
	full file paths. Returns False when something failed.
	"""
	
	# Init empty list
	flist = {}
	
	# Expand path
	path = os.path.realpath(path)
	
	log.prNot(log.INFO, "saveData(): file '%s', npy: %d pickle: %d, csv: %d" %\
		(os.path.basename(path), asnpy, aspickle, ascsv))
	
	# Make dir if necessary
	outdir = os.path.dirname(path)
	if (not os.path.isdir(outdir)):
		log.prNot(log.INFO, "saveData(): making directory '%s'" % (outdir))
		os.makedirs(outdir)
	
	# If everything is False, enable asnpy
	if (not asnpy and not aspickle and not ascsv):
		asnpy = True
	
	if (asnpy):
		import numpy as N
		# Save data in numpy format
		if (explicit): uri = path
		else: uri = path + '.npy'
		log.prNot(log.INFO, "saveData(): storing numpy to '%s'" % (uri))
		# Save old file, if present
		saveOldFile(uri, postfix='.old', maxold=old)
		N.save(uri, data)
		flist['npy'] = os.path.basename(uri)
	
	if (ascsv):
		import numpy as N
		# Save data in csv format
		if (explicit): uri = path
		else: uri = path + '.csv'
		log.prNot(log.INFO, "saveData(): storing csv to '%s'" % (uri))
		# Save old file, if present
		saveOldFile(uri, postfix='.old', maxold=old)
		if (csvhdr is not None): data = N.vstack((N.array(csvhdr), data))
		N.savetxt(uri, data, fmt=csvfmt, delimiter=', ')
		flist['csv'] = os.path.basename(uri)
	
	if (aspickle):
		import cPickle as pickle
		if (explicit): uri = path
		else: uri = path + '.pickle'
		# Save old file, if present
		log.prNot(log.INFO, "saveData(): storing pickle to '%s'" % (uri))
		saveOldFile(uri, postfix='.old', maxold=old)
		pickle.dump(data, file(uri, 'w'))
		flist['pickle'] = os.path.basename(uri)
	
	if (asfits):
		import pyfits
		if (explicit): uri = path
		else: uri = path + '.fits'
		# Save old file, if present
		log.prNot(log.INFO, "saveData(): storing fits to '%s'" % (uri))
		saveOldFile(uri, postfix='.old', maxold=old)
		pyfits.writeto(uri, data)
		flist['fits'] = os.path.basename(uri)
	
	return flist
	# done


def restoreData(path):
	"""
	Restore files to memory. 'path' should be a pickled file holding a dict with 
	data identifiers as keys. Each entry should again be a dict with data types 
	as keys, and filenames as values. For example:
	
	meta = {'data': {'fits': 'data-in-fitsformat.fits', npy:
	  'data-in-numpyformat.npy'}}
	
	This function will load the file, parse the contents, and return a dict with 
	the data for each data id. If one data id has more datatypes, the first one
	will be loaded.
	
	@param path Pickled file holding the metadata.
	
	@return A dict with dataid as keys, and the data as values.
	"""
	
	import cPickle as pickle
	
	meta = pickle.load(open(path))
	
	ret = {}
	files_used = []
		
	# Loop over the data IDs
	for (did, dtype) in meta.items():
		# Loop over the data types for each dataid
		if (did in ['path', 'base']): continue
		for (dtype, dfile) in dtype.items():
			# If the filetype is supported, load it
			if (dtype in _FORMATS_LOAD): 
				ret[did] = _FORMATS_LOAD_F[dtype](dfile)
				files_used.append(dfile)
				# If we succesfully loaded the file using this datatype, skip the 
				# other datatypes.
				if (ret[did] is not False): continue
			else:
				ret[did] = None			
	
	# Process meta-data first, remove from dict.
	ret['path'] = meta.pop('path', os.path.dirname(os.path.realpath('.')))
	ret['base'] = meta.pop('base', os.path.commonprefix(files_used)[:-1])
	ret['base'] = os.path.basename(ret['base'])
	if (len(ret['base']) < 4): 
		log.prNot(log.WARNING, "restoreData(): Warning, base very short, adding timestamp.")
		ret['base'] += str(int(time.time()))
	# Re-insert in meta after sanitizing
	meta['base'] = ret['base']
	meta['path'] = ret['path']
	
	return (ret, meta)

