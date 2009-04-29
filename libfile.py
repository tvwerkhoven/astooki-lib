#!/usr/bin/env /sw/bin/python2.5
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
import os
from liblog import *
import cPickle as pickle

def saveOldFile(uri, postfix='.old', maxold=5):
	"""
	If 'uri' is present, rename the file to prevent it being overwritten. 
	The filename will be to 'uri' + postfix + the lowest integer that 
	constitutes a non-existing file. 'maxold' specifies how many old files 
	we should keep.
	
	@param uri Filepath to prevent overwriting to
	@param postfix Suffix to add to the file before a counter
	@param maxold Keep this many old files maximum
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
		prNot(VERB_DEBUG, "saveOldFile(): renaming file to prevent overwriting")


def loadData(path, asnpy=False, aspickle=False, ascsv=False, shape=None):
	"""
	Reverse function of saveData(): load data stored on disk to prevent 
	re-computation of the analysis. Formats supported are numpy arrays 
	(enable with 'asnpy') and pickled files (enable with 'aspickle'). File
	URI will be path + '.npy'/'.pickle'. If both 'asnpy' and 'aspickle' are 
	True, numpy will be preferred.
	
	If shape is set, it will be verified that the returned array is indeed 
	that shape.
	
	Return value is a tuple of (flag, data), with the bool 'flag' 
	indicating whether or not data has been found and 'data' carrying the 
	data -- which can be of any type. Returns False when loading failed.
	
	TODO: add 'ascsv' option
	"""
	import numpy as N
	
	prNot(VERB_DEBUG, "loadData(): id '%s' asnpy: %d aspickle: %d" % \
		(id, asnpy, aspickle))
	
	# Make sure there is only one setting true
	if (N.sum([asnpy, aspickle, ascsv]) > 1):
		aspickle = False
		ascsv = False
		prNot(VERB_WARNING, "loadData(): Cannot load more than one format at a time, disabling pickle.")
	elif (N.sum([asnpy, aspickle, ascsv]) < 1):
		asnpy = True
		prNot(VERB_WARNING, "loadData(): No format selected, enabling npy.")
	
	if (asnpy):
		#uri = path + '.npy'
		# Check if file exists
		if (not os.path.isfile(path)):
			prNot(VERB_WARN, "loadData(): numpy file '%s' does not exists." % \
					(os.path.split(path)[1]))
			return False
			
		# Load results
		results = N.load(path)
	
	if (aspickle):
		#uri = path + '.pickle'
		# Check if file exists
		if (not os.path.isfile(path)):
			prNot(VERB_WARN, "loadData(): pickle file '%s' does not exists." % \
					(os.path.split(path)[1]))
			return False
			
		# Load results
		results = pickle.load(open(path))
	
	if (ascsv):
		#uri = path + '.pickle'
		# Check if file exists
		if (not os.path.isfile(path)):
			prNot(VERB_WARN, "loadData(): pickle file '%s' does not exists." % \
					(os.path.split(path)[1]))
			return False
			
		# Load results as csv
		results = N.loadtxt(path, delimiter=',')
	
	# Check if shape matches
	if (shape is not None and results.shape != shape):
		prNot(VERB_WARN, "loadData(): shapes do not match.")
		return False
		
	return results


def saveData(path, data, asnpy=False, aspickle=False, ascsv=False, csvfmt='%.18e', csvhdr=None, old=0):
	"""
	Save (intermediate) results to 'path'. Data can be 
	stored as numpy array (if asnpy is True), csv file (if ascsv is True)		
	and/or pickled format (if aspickle is True). The final path will be
	path + '.npy'/'.csv'/'.pickle', for the different formats.
	
	@param path Base path to store files to. Should be dir+filename
	@param data Some numpy formatted data.
	@param asnpy Store data in numpy format to path+'.npy' [False]
	@param aspickle Store data in pickle format to path+'.pickle' [False]
	@param ascsv Store data in csv format to path+'.csv' [False]
	@param csvfmt The format for storing data as CSV ['%.18e']
	@param csvhdr A header to write to the head of the file. Header should nbe a 
	              list with the same number of elements as elements in 'data' 
	              [None]
	@param old Backup this many pre-existing files at maximum
	
	
	@return A dict of files the data was saved to when successful. The keys will 
	be one or more of 'npy', 'pickle' or 'csv' and the values will be the 
	full file paths. Returns False when something failed.
	"""
	import numpy as N
	
	# Init empty list
	flist = {}
	
	prNot(VERB_DEBUG, "saveData(): file '%s', npy: %d pickle: %d, csv: %d" %\
		(os.path.basename(path), asnpy, aspickle, ascsv))
	
	# Make dir if necessary
	outdir = os.path.dirname(path)
	if (not os.path.isdir(outdir)):
		prNot(VERB_DEBUG, "saveData(): making directory '%s'" % (outdir))
		os.makedirs(outdir)
	
	# If everything is False, enable asnpy
	if (not asnpy and not aspickle and not ascsv):
		asnpy = True
	
	if (asnpy):
		# Save data in numpy format
		uri = path + '.npy'
		prNot(VERB_DEBUG, "saveData(): storing numpy to '%s'" % (uri))
		# Save old file, if present
		saveOldFile(uri, postfix='.old', maxold=old)
		N.save(uri, data)
		flist['npy'] = os.path.basename(uri)
	
	if (ascsv):
		# Save data in csv format
		uri = path + '.csv'
		prNot(VERB_DEBUG, "saveData(): storing csv to '%s'" % (uri))
		# Save old file, if present
		saveOldFile(uri, postfix='.old', maxold=old)
		if (csvhdr is not None): data = N.vstack((N.array(csvhdr), data))
		N.savetxt(uri, data, fmt=csvfmt, delimiter=', ')
		flist['csv'] = os.path.basename(uri)
	
	if (aspickle):
		uri = path + '.pickle'
		# Save old file, if present
		prNot(VERB_DEBUG, "saveData(): storing pickle to '%s'" % (uri))
		saveOldFile(uri, postfix='.old', maxold=old)
		pickle.dump(data, file(uri, 'w'))
		flist['pickle'] = os.path.basename(uri)
	
	return flist
	# done
