#!/usr/bin/env python2.5
# encoding: utf-8
"""
This is astooki.libplot, providing some specialized plotting functions.
"""

##  @file libplot.py
# @author Tim van Werkhoven (tim@astro.su.se)
# @date 20090327
# 
# Created by Tim van Werkhoven on 2009-03-27.
# Copyright (c) 2008-2009 Tim van Werkhoven (tim@astro.su.se)
# 
# This file is licensed under the Creative Commons Attribution-Share Alike
# license versions 3.0 or higher, see
# http://creativecommons.org/licenses/by-sa/3.0/

## @package astooki.libplot
# @brief Library for specialized plotting
# @author Tim van Werkhoven (tim@astro.su.se)
# @date 20090327
#
# This package provides some routines for making WFWFS related plots

#=============================================================================
# Import libraries here
#=============================================================================

import numpy as N				# Processing numerical data
import liblog as log		# Logging stuff
import Gnuplot					# For making nice plots
import time
import os
import subprocess				# For running eps2pdf in a shell

#=============================================================================
# Data processing & plotting functions
#=============================================================================

def plotShifts(filebase, shifts, sapos, sasize, sfpos, sfsize, plrange=None, mag=1.0, allsh=False, allshstyle=2, showvecs=True, showsa=True, showsf=True, title=None, legend=True, cleanup=False, swapxy=False):
	"""
	Plot 'shifts', which should be a N_files * N_ref * N_sa * N_sf * 2 array.
  Array will be averaged over N_ref before processing. 'filebase' should be a
  complete uri which will be appended with eps, pdf and txt for various output
  formats. 'sapos', 'sasize', 'sfpos' and 'sfsize' should hold subaperture
  position and -size and subfield position and -size, respectively.
	
	"""
	
	if (swapxy):
		# Swap x and y position everywhere. Simply re-slice the arrays like this:
		shifts = shifts[...,::-1]
		sapos = sapos[...,::-1]
		sasize = sasize[...,::-1]
		sfpos = sfpos[...,::-1]
		sfsize = sfsize[...,::-1]
	
	# Output filenames
	plotfile = filebase + '.eps'
	infofile =  filebase + '-info.txt'
	datafile = filebase + '-data.dat'
	alldatafile = filebase + '-alldata.dat'
	gnuplotfile = filebase + '-plot.plt'
	
	# Dimensions
	nfile = shifts.shape[0]
	nref = shifts.shape[1]
	nsa = shifts.shape[2]
	nsf = shifts.shape[3]
	# Average over Nref, the number of references we have
	shifts = shifts.mean(axis=1)
	
	# Reform data, save to file. Use 4 column x, y, dx, dy syntax.
	allcpos = (sapos.reshape(-1,1,2) + sfpos.reshape(1,-1,2)) + sfsize/2.0
	allcpos = allcpos.reshape(-1,2)
	shiftsplall = (shifts.reshape(nfile,-1,2) * mag) + allcpos.reshape(1,-1, 2)
	shiftsplall = shiftsplall.reshape(-1, 2)
	shiftspl = (N.mean(shifts, axis=0)).reshape(-1,2) * mag
	N.savetxt(datafile, N.concatenate((allcpos, shiftspl), axis=1))
	
	# Init plotting
	gp = Gnuplot.Gnuplot()
	gnuplotInit(gp, hardcopy=plotfile, rmfile=True)
	
	# OLD
	# if (plorigin and plrange):
	# 	gp('set xrange [%d:%d]' % (plorigin[0], plrange[0]))
	# 	gp('set yrange [%d:%d]' % (plorigin[1], plrange[1]))
	# 	plsize = N.array(plrange)-N.array(plorigin)
	# 	gp('set size ratio %f' % (1.0*plsize[1]/plsize[0]))
	
	# Set plotting range
	if plrange == None:
		xran = (min(sapos[:,0]) + N.floor(N.min(shifts[...,0])) - 0.5*sasize[0], \
			max(sapos[:,0]) + N.ceil(N.max(shifts[...,0])) + 1.5*sasize[0] + 1)
		yran = (min(sapos[:,1]) + N.floor(N.min(shifts[...,1])) - 0.5*sasize[1], \
			max(sapos[:,1]) + N.ceil(N.max(shifts[...,1])) + 1.5*sasize[1])
	else:
		xran = tuple(N.array(plrange[0])*1.0)
		yran = tuple(N.array(plrange[1])*1.0)
	
	plsize = N.array([xran[1] - xran[0], yran[1] - yran[0]])
	
	gp('set xrange [%f:%f]' % xran)	
	gp('set yrange [%f:%f]' % yran)
	
	# Aspect ratio square
	ar = (yran[1] - yran[0])/(xran[1] - xran[0])
	gp('set size ratio %f' % (ar))
	
	gp('set xlabel "x [pixel]"')
	gp('set ylabel "y [pixel]"')
	gp('set key off')
	
	if (title != None):
		title = addGpSlashes(title)
		gp('set title "%s"' % (title))
	
	# Loop over sapos and sfpos
	box = 1
	if (showsa == True):
		for _sa in xrange(len(sapos)):
			_sapos = sapos[_sa]
			# Draw subaperture boxes
			gp('set obj %d rect from %d,%d to %d,%d fs empty lw 0.4' % \
				(box, _sapos[0], _sapos[1], _sapos[0]+sasize[0], _sapos[1]+sasize[1]))
			box += 1
			
			if (showsf == True):
				for _sf in xrange(len(sfpos)):
					_sfpos = _sapos + sfpos[_sf]
					# Draw subfield boxes
					gp('set obj %d rect from %d,%d to %d,%d fs empty lw 0.2' % \
						(box, _sfpos[0], _sfpos[1], \
						_sfpos[0]+sfsize[0], _sfpos[1]+sfsize[1]))
					box += 1
	
	if (legend is True):
		# Draw vector corresponding with the overall average shift
		avgshift = N.mean((shifts[...,0]**2.0 + shifts[...,1]**2.0)**0.5)
		if (avgshift < 1.0): avgshift = 1.0
		avgshift = N.round(avgshift)
		legsize = avgshift * mag
		# Position at 5% of the plot
		legpos = N.round(plsize * 0.05)
		txtpos = N.round(plsize * [0.05, 0.02])
		gp('set style line 1 lt 1 lw 1.0 lc rgb "red"')
		gp('set arrow from %d,%d rto %d,%d ls 1' % \
			(legpos[0], legpos[1], legsize, 0))
		gp('set label "%d-pix shift" at %d,%d font "Palatino,4"' % \
			(avgshift, txtpos[0], txtpos[1]))
	
	# If allsh is True, add dots for all individual measured shifts
	gp('set style line 1 lt 1 lw 0.6 lc rgb "red"')
	gp('set style line 2 pt 0 ps 0.6 lt 0 lw 0.6 lc rgb "blue"')
	plotcmd = ''
	prefix = "plot "
	
	# Save all data, we might plot it but otherwise we can use it elsewhere
	N.savetxt(alldatafile, shiftsplall)
	if (allsh == True):
		plotcmd += prefix + '"%s" with dots ls %d title "All WFWFS shifts"' % (alldatafile, allshstyle)
		prefix = ", "
	if (showvecs == True):
		plotcmd += prefix +'"%s" using 1:2:3:4 with vec nohead ls 1 title "WFWFS shifts"' % (datafile)
		prefix = ", "
	
	if (plotcmd != ''): 
		gp(plotcmd)
	
	# Save gnuplot output for further reference.
	gp.save(gnuplotfile)
	
	_waitForFile(plotfile)
	_convertPdf(plotfile)
	
	# Now print some useful info to a file
	f = open(infofile,'w')
	tmp = shifts.reshape(-1,2)
	avgsh = N.mean(tmp, axis=0)
	avgsh_var = N.std(tmp, axis=0)
	minsh = N.min(tmp, axis=0)
	maxsh = N.max(tmp, axis=0)
	tmp = tmp.flatten()
	clip = 100.0 * N.argwhere(abs(tmp) == tmp.max()).size / tmp.size
	
	print >> f, "frames: %d, subapertures: %d, subfields: %d" % \
	 	shifts.shape[0:3]
	print >> f, "avg shift: (%.5g, %.5g) +- (%.5g, %.5g)" % \
	 	(avgsh[0], avgsh[1], avgsh_var[0], avgsh_var[1])
	print >> f, "min: (%.5g, %.5g), max: (%.5g, %.5g)" % \
	 	(minsh[0], minsh[1], maxsh[0], maxsh[1])
	print >> f, "percentage at maximum (clipped): %-3.2f" % (clip)
	f.close()
		
	# Done


def overlayMask(img, saccdpos, saccdsize, filename, number=True, coord=True, norm=0.5, crop=False, border=True):
	"""
	Generate a 'fancy' image from 'img' with an overlay of the subaperture
	mask (positions in 'saccdpos' and size in 'saccdsize').
	
	Optional parameters include:
	'number' to number the subapertures [True]
	'coord' to show the subaperture coordinates [True]
	'crop' sets everything outside the subapertures to white [False]
	'norm' scale all the data outside the subapertures to the top 'norm' part 
	of the range of the data within the subapertures. [0.5]
	"""
	
	log.prNot(log.INFO, "overlayMask(): rendering subap mask over image.")
	
	### Process data
	### ============
	
	# Image should be float during processing
	img = img.astype(N.float)
	
	# Make a mask first
	mask = N.zeros((img.shape), dtype=N.bool)
	maskborder = N.zeros((img.shape), dtype=N.bool)
	for pos in saccdpos:
		# Again remember the reverse indexing of numpy arrays:
		mask[\
			pos[1]:pos[1]+saccdsize[1], \
			pos[0]:pos[0]+saccdsize[0]] = 1
		maskborder[\
			pos[1]-1:pos[1]+saccdsize[1]+1, \
			pos[0]-1:pos[0]+saccdsize[0]+1] = 1
		maskborder[\
			pos[1]:pos[1]+saccdsize[1], \
			pos[0]:pos[0]+saccdsize[0]] = 0
	
	# Get the range of the interesting part:
	maxval = N.max(img[mask])
	minval = N.min(img[mask])
	# Copy the image
	masked = img
	
	# If we're cropping, set everything outside the subapertures to the 
	# maximum value found inside the subapertures, 'maxval' (making it white)
	if (crop):
		masked[mask == False] = maxval
	# Normalize the data outside the subapertures if we're not cropping:
	elif (norm is not False):
		# Set everything *outside* the mask to the range of everything inside 
		# the maxed, but scaled down by a factor 'norm', using the upper part 
		# of the range. If the interesting data has a range 0-1, and norm is 
		# 0.7, everthing outside the subapertures will be scaled to 0.3--1.0
		
		# Absolute dynamic range for outside the subaps
		crange = (maxval-minval) * norm
		# Minimum value (offset)
		cmin = minval + ((maxval-minval) * (1-norm))
		# Old minimum and maximum
		omin = N.min(masked[mask == False])
		omax = N.max(masked[mask == False])
		# Scale data
		masked[mask == False] = ((masked[mask == False] - omin) * crange / \
		 	(omax - omin)) + cmin
	else:
		# If norm is False, use the whole image: don't crop and don't rescale
		maxval = N.max(img)
		minval = N.min(img)
		
	# Draw white borders around the subapertures
	if (border):
		masked[maskborder] = maxval
	
	### Use Cairo to make the image
	### ===========================
	import cairo				# For making nice PNG stuff
	
	# Scale the values to 0-255
	masked = (255*(masked - minval)/(maxval - minval)) 
		
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
	for pos in saccdpos:
		# Move the 'cursor', show some text
		# NOTE: we have to perform the position transform ourselves here, 
		# because if we would use ctx.transform(), the text would be 
		# transformed as well (which we do not want)
		ctx.move_to(pos[0] +1, imgsurf.get_height()- (pos[1]+1))
		txt = ''
		if (number is True):
			txt += '%d' % (sanum)
			sanum += 1
		if (coord is True):
			txt += ' @ (%d,%d)' % (pos[0], pos[1])
		
		ctx.show_text(txt)
	
	# Done, save as PNG
	destsurf.write_to_png(filename + '.png')
	
	# And as FITS file
	import pyfits
	pyfits.writeto(filename + '.fits', masked, clobber=True)
	
	log.prNot(log.INFO, "overlayMask(): done, wrote image as fits and png.")


def visCorrMaps(maps, res, sapos, sasize, sfpos, sfsize, filename, shifts=None, mapscale=1.0, text=None, outpdf=False, outfits=False):
	"""
	Visualize the correlation maps generated and the shifts measured.
	
	Using the correlation maps in 'maps' and the meta-information on these 
	maps in 'sapos' (subaperture position), 'sasize' (subap size), 'sfpos' 
	(relative subfield position), 'sfsize' (relative sf size), make a nice 
	plot to visualize these correlation maps.
	
	If 'shifts' is not None, lines originating from the center of each 
	subaperture-subfield pair with 'shifts' as lengths will be drawn over 
	the correlation maps.
	
	The layout of 'maps' and 'shifts' should be such that maps[sai][sfi] 
	should correspond to the map for subaperture sai and subfield sfi.
	
	'filename' should be a path to a basename to be used for PDF output 
	(enabled with 'outpdf') and FITS output (enabled with 'outfits'). If 
	'mapscale' is not eqaul to 1, the correlation maps will be scaled 
	according to this factor.
	"""
	
	if (outfits):
		raise NotImplemented("FITS output is not implemented yet.")
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
		import cairo				# For making nice PNG stuff
		pdfsurf = cairo.PDFSurface(filename+'.pdf', 72.0, 72.0)
		# Create context
		ctx = cairo.Context(pdfsurf)
		
		# Set coordinate system to 'res', origin at lower-left
		ctx.translate(0, 72.0)
		ctx.scale(72.0/res[0], -72.0/res[1])
	
	# Map/shvec counter
	saidx = 0
	# Take the size of the first correlation map as standard (should all 
	# be the same)
	msize = N.array(maps[0,0].shape)
	
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
				# Create outline of the subfield in black
				ctx.set_source_rgb(0.0, 0.0, 0.0)
				ctx.set_line_width(0.75)
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
				ctx.scale(mapscale, mapscale)
				ctx.set_source_surface(surf, _cent[0]/mapscale - \
				 	msize[0]/2, \
					_cent[1]/mapscale - msize[1]/2)
				# Paint it
				ctx.paint()					
				# Reset scaling
				ctx.scale(1./mapscale, 1./mapscale)
				# If shifts are give, draw lines
				if (shifts != None):
					# Set to red, make a thin line (0.5 pixel)
					ctx.set_source_rgb(1, 0, 0)
					ctx.set_line_width(0.5 * mapscale)
					ctx.set_line_cap(cairo.LINE_CAP_ROUND)
					# Move cursor to the center
					ctx.move_to(_cent[0], _cent[1])
					# Draw a shift vector
					ctx.rel_line_to(shifts[saidx][sfidx][0] * mapscale, \
					 	shifts[saidx][sfidx][1] * mapscale)
					ctx.stroke()
					
					# Move cursor to lower-left
					ctx.move_to(_pos[0]+5, _pos[1]+5)
					ctx.set_font_size(20) # in pixels?
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
		# Write some text in the lower-left corner
		if (text != None):
			ctx.move_to(10, 10)
			ctx.scale(1,-1)
			ctx.show_text(text)
			ctx.scale(1,-1)
		pdfsurf.finish()
	# Done
	


def showSaSfLayout(outfile, sapos, sasize, sfpos=[], sfsize=[], method='ccd', coord=True, number=True, plrange=None, aptr=None, aptp=None):
	"""
	Show the subaperture/subfield layout in Gnuplot.
	
	'outfile' will be the output EPS file of Gnuplot
	'sapos' is the subaperture positions in some units
	'sasize' is the subaperture size in the same units
	'sfpos' is the subfield positions in the same units
	'sfsize' is the subfield size in the same units
	'method' is the meaning of the positions: 'ccd' or 'll'
	'coord' if True, add coordinates of the subapertures
	'number' if True, add numbers to the subapertures	
	"""	
	
	# TODO: implement 'll' method
	if (method != 'ccd'):
		raise NotImplemented("Not implemented yet")
	
	log.prNot(log.INFO, "showSaSfLayout(): Saving plot to '%s'." % (outfile))
	# Initiate Gnuplot
	gp = Gnuplot.Gnuplot()
	
	# Make a nice path
	outfile = os.path.realpath(outfile)
	
	# Set default settings
	gnuplotInit(gp, hardcopy=outfile, rmfile=True)
	
	# Set plotting range
	if plrange == None:
		xran = (min(sapos[:,0]) - 0.5*sasize[0], \
			max(sapos[:,0]) + 1.5*sasize[0])
		yran = (min(sapos[:,1]) - 0.5*sasize[1], \
			max(sapos[:,1]) + 1.5*sasize[1])
	else:
		xran = tuple(N.array(plrange[0])*1.0)
		yran = tuple(N.array(plrange[1])*1.0)
	
	gp('set xrange [%f:%f]' % xran)	
	gp('set yrange [%f:%f]' % yran)
	
	# Aspect ratio square
	ar = (yran[1] - yran[0])/(xran[1] - xran[0])
	gp('set size ratio %f' % (ar))
	
	# No legend
	gp('set key off')
	
	# Object counter
	obj = 0
	
	# # Draw circular aperture if wanted
	# if aptr != None and aptp != None:
	# 	gp('set obj %d circle at %f,%f size %f back' % \
	# 		(obj, aptp[0], aptp[1], aptr))
	# 	obj += 1
	
	# Loop over the subapertures
	sanum = -1
	for sa in sapos:
		obj += 1
		sanum += 1
		# Set subimage box
		gp('set obj %d rect from %f,%f to %f,%f fs empty lw 0.8' % \
			(obj, sa[0], sa[1], sa[0]+sasize[0], sa[1]+sasize[1]))
		
		caption = ''
		# Add number, if requested
		if (number):
			caption += '#%d' % (sanum)
		# Add coordinate, if requested
		if (coord):
			caption += ' (%.4g,%.4g)' % (sa[0], sa[1])
		# Set caption
		gp('set label %d at %f,%f "%s" font "Palatino,2.5"' % \
			(obj, sa[0] + sasize[0]*0.06, sa[1] + sasize[1]*0.1, caption))
		
		# Add subfields
		for sf in sfpos:
			_sf = sa + sf
			obj += 1
			gp('set obj %d rect from %f,%f to %f,%f fs empty lw 0.4' % \
				(obj, _sf[0], _sf[1], _sf[0]+sfsize[0], _sf[1]+sfsize[1]))
	
	# Finish the plot (TODO: ugly hack, how to do this nicer?)
	gp('plot -99999')
	
	_waitForFile(outfile)
	_convertPdf(outfile)


#=============================================================================
# Helper routines
#=============================================================================

def _convertPdf(file):
	"""
	Convert an EPS file to PDF using epstopdf.
	"""
	pdffile = os.path.splitext(file)[0]+'.pdf'
	ret = subprocess.call(["epstopdf %s -o=%s" % (file, pdffile)], shell=True)


def mkPlName(runfiles, appendix, sep='_'):
	"""
	Make a filename for a plot. Convenience function for elaborate filenames.
	"""
	
	# Construct filename
	filename = runfiles['datasubdir'].replace(os.path.sep, sep) + sep +\
	 	runfiles['runid'] + sep + appendix
	# Add directory and return
	return os.path.join(runfiles['plotdir'], filename)


def _waitForFile(fname, delay=0.3, maxw=5):
	"""
	Wait until 'fname' exists. Sort of a solution for the asynchronous Gnuplot 
	calls, we never know when it's finished so we can't convert it to PDF and 
	all.
	
	'delay' is the check-interval in seconds
	'maxw' is the maximum time this function will wait
	"""
	
	while (not os.path.exists(fname) and maxw >= 0 ):
		log.prNot(log.INFO, "waitForFile(): Waiting for '%s'" % (fname))
		maxw -= delay
		time.sleep(delay)
	
	time.sleep(delay)


def addGpSlashes(txt):
	"""
	Add Gnuplot slashes to 'txt' such that it is parsed literally. Used mainly
	for titles and text in plots in Gnuplot. Adds four slashes in total, the 
	first factor two to survive Python, the second factor two to survive 
	Gnuplot itself.
	"""
	escapechars = ['_', '^', '$']
	for c in escapechars:
		txt = txt.replace(c,'\\\\'+c)
	return txt


def rmFiles(filelist):
	"""
	Remove files in 'filelist', if they exist.
	"""
	for _f in filelist:
		if os.path.exists(_f):
			log.prNot(log.INFO, "rmFiles(): Removing %s" % (_f))
			os.remove(_f)


def gnuplotInit(gp, hardcopy=False, verb=False, rmfile=False):
	"""
	Set some default gnuplot options for Gnuplot instance 'gp'. If 'hardcopy'
	is set, an EPS file will be written to that location instead of display
	the plot to screen. If 'verb' is True, display some debug information. If 
	'rmfile' is True, delete the hardcopy file.
	"""
	
	# First reset gnuplot completely
	gp.reset()
	
	# If we want a hardcopy, do so
	if (hardcopy is not False):
		hc = os.path.realpath(hardcopy)
		log.prNot(log.INFO, "Saving hardcopy to '%s'" % (hc))
		# Make sure the directory exists
		if (not os.path.exists(os.path.dirname(hc))):
			os.makedirs(os.path.dirname(hc))
		if (rmfile == True and os.path.isfile(hc)):
			os.remove(hc)
		gp('set terminal postscript eps enhanced color size 8.8cm,5.44cm "Palatino-Roman" 10')
		gp('set output "%s"' % (hardcopy))
	
	gp('set key on top left box spacing 2 samplen 6')
	gp('set bmargin 3.5')
	gp('set rmargin 2')
	gp('set style line 1 lt 1 lw 2.2 lc rgb "red"')
	gp('set style line 2 lt 2 lw 2.2 lc rgb "blue"')
	gp('set style line 3 lt 3 lw 2.2 lc rgb "purple"')
	gp('set style line 4 lt 4 lw 2.2 lc rgb "cyan"')
	gp('set style line 5 lt 5 lw 2.2 lc rgb "orange"')
	gp('set style line 6 lt 6 lw 2.2 lc rgb "black"')
	gp('set style line 7 lt 7 lw 2.2 lc rgb "green"')
	gp('set style line 8 lt 8 lw 2.2 lc rgb "brown"')

