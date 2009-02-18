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
import ConfigParser
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
# Physics-related routines go here
#=============================================================================

ConfigParser.ConfigParser()
config.read('example.cfg')


class teleConfig():
	"""
	Telescope info
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
			'saoff' : 0.5,\
			'sadispx' : 0.0,\
			'sadispy' : 0.0,\
			'sascl' : 0.95,\
			'sfindx' : 10,\
			'sfindy' : 10}
		# Load configuration from cfgfile
		self.cfg = ConfigParser.ConfigParser()
		self.cfg.read(cfgfile)
		try:
			self.cfglist = cfg.items('telescope')
		except:
			raise IOError("Cannot find 'telescope' section in config file")
		
		# Set variables according to configuration
		self.aptr = self.cfg.getfloat('telescope', 'aptr')
		self.apts = self.cfg.get('telescope', 'apts')
		self.angle = []
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
		self.saoff = self.cfg.getfloat('telescope', 'sapoff')
		self.sadispx = self.cfg.getfloat('telescope', 'sadispx')
		self.sadispy = self.cfg.getfloat('telescope', 'sadispy')
		self.sadisp = N.array([self.sadispx, self.sadispy])
		self.sascl = self.cfg.getfloat('telescope', 'sascl')
		self.sfindx = self.cfg.getfloat('telescope', 'safindx')
		self.sfindy = self.cfg.getfloat('telescope', 'safindy')
		self.sfind = N.array([self.sfindx, self.sfindy])
		self.sftotx = self.cfg.getfloat('telescope', 'sftotx')
		self.sftoty = self.cfg.getfloat('telescope', 'sftoty')
		self.sftot = N.array([self.sftotx, self.sftoty])
		
	def genSubaptmask(self):
		""" Generate a subaperture mask given a pre-setup telescope
		configurations with at least the parameters aptr, sapitch, saxoff and
		aptr. This returns a list with the centroid subaperture positions.
		"""
		# (half) width and height of the subaperture array
		sa_arr = (ceil(self.aptr/self.sapitch)+1).astype(int)
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
		
		return array(pos)
	

def parseSetup(tele, atmos, recon, seed=-1):
	""" Takes a freshly setup reconstruction configuration and fills in the
	blanks that are necessary to perform the reconstruction.
	"""
	# Parse telescope parameters first
	# --------------------------------
	
	# Get the subaperture positions and count these
	tele['sapos'] = genSubaptmask(tele)
	tele['nsa'] = len(tele['sapos'])
	# Subfield FoV
	tele['sffov'] = tele['fov']/tele['sfind']
	# Calculate subfield viewing direction here
	sfang = indices(tele['sftot'], dtype=float).reshape(2,-1).T / \
		(tele['sftot']+1)
	tele['sfang'] = sfang * tele['fov'] - tele['fov']/2 + tele['sffov']/2
	# Number of subfields
	tele['nsf'] = product(tele['sftot'])
	
	# Next parse the atmosphere parameters
	# ------------------------------------
	
	if atmos['lh'] == -1:
		# Atmosphere layer heights (equi-distant in height^2)
		atmos['lh'] = atmos['maxh'] * (arange(atmos['n']) / \
		 	(atmos['n']-1.0))**2.0
	
	# Cut-off points around the layers. These will be the integration
	# lengths to obtain r_0 values from the C_n^2 profile. Obtain these values
	# by fitting a + b*x + c*x^2 to the heights as function of layer index,
	# and take the heights of half-valued indices (0.5, 1.5, 2.5 etc)
	
	# Setup the functions (1, x and x^2)
	funcs = ones((atmos['n'], 3), dtype=float)
	funcs[:,1] = arange(atmos['n'], dtype=float)
	funcs[:,2] = arange(atmos['n'], dtype=float)**2.0
	# Fit the data
	(p, residuals, rank, s) = linalg.lstsq(funcs, atmos['lh'])
	# Find the mid-layer-heights
	idx = (r_[0.0, arange(atmos['n'])+0.5])
	atmos['dh'] = p[0] + p[1] * idx + p[2] * (idx**2.0)
	
	# Calculate layer origins
	atmos['lorig'] = atmos['lh'] * tan(tele['angle']).reshape(2,1)
	
	# Calculate layer sizes
	atmos['lsize'] = tele['aptr'] + atmos['lh'].reshape(-1,1) * \
		tan(0.5 * tele['fov'])
	# Take into account the fill factor
	atmos['lsize'] *= atmos['fill']
	
	# Calculate r_0 values from C_n^2 profile and the corresponding wavefront 
	# variances
	atmos['r0'] = calcR0(atmos['cn2'], atmos['dh'], tele['angle'])
	atmos['vars'] = 1.03 * \
		( atmos['lsize']*2 / tile(atmos['r0'], (2,1)).T )**(5./3.)
	
	# Finally parse the reconstruction parameters
	# -------------------------------------------
	
	# These variables will hold all height geometries
	recon['cheights'] = zeros((product(recon['vary']), recon['n']))
	#recon['corigs'] = zeros((product(recon['vary']), recon['n'], 2))
	#recon['csizes'] = zeros((product(recon['vary']), recon['n'], 2))
	# Generate layer configurations here
	for l in range(recon['n']):
		# This layer should vary recon['vary'][l] times between heights 
		# recon['hbounds'][l][0] and recon['hbounds'][l][1]
		thisHeights = linspace(recon['hbounds'][l,0], recon['hbounds'][l,1], \
		 	recon['vary'][l])
		# Now tile and repeat these heights
		recon['cheights'][:, l] = tile( \
			repeat(thisHeights, product(recon['vary'][l+1:])), \
			recon['vary'][:l])
	
	recon['corigs'] = recon['cheights'].reshape(product(recon['vary']), \
	 	recon['n'], 1) * tan(tele['angle']).reshape(1,1,2)
	recon['csizes'] = tele['aptr'] + \
			recon['cheights'].reshape(product(recon['vary']), recon['n'], 1)*\
			tan(0.5 * tele['fov']).reshape(1,1,2)
			#tan(tele['angle']).reshape(1,1,2)
	
	# Setup global configurations
	# ---------------------------
	
	# Setup random number seeding if seed is not auto
	if seed != -1:
		random.seed(seed)
	else:
		random.seed(random.randint(2L**31-1))
	
	# Combine configuration settings in one dict
	conf = {
		'tele': tele,
		'atmos': atmos,
		'recon': recon,
		'seed': seed,
		'tag': -1
	}
	
	# Generate a tag (hash) for the data
	conf["tag"] = calcTag(conf)
			
	# Save the configuration to a configuration-specific directory
	try: os.mkdir("./data/")
	except OSError: pass
	except: prNot("Could not create directory './data/")
	
	try: os.mkdir("./data/" + conf["tag"]) 
	except OSError: pass
	except: prNot("Could not create directory './data/" + conf["tag"] + "'")
	
	save('./data/' + conf["tag"] + '/conf.npy', conf)
	pickleSave('./data/' + conf["tag"] + '/conf.pickle', conf)
	
	print "*** Successfully saved configuration to ./data/" + conf["tag"] + \
	 	"/conf.npy"
	
	return conf


def genSubaptmask(tele):
	""" Generate a subaperture mask given a pre-setup telescope configurations
	with at least the parameters aptr, sapitch, saxoff and aptr. This returns
	a list with the centroid subaperture positions.
	"""
	# (half) width and height of the subaperture array
	sa_arr = (ceil(tele['aptr']/tele['sapitch'])+1).astype(int)
	# Init empty list to store positions
	pos = []
	# Loop over all possible subapertures and see if they fit inside the shape
	for sax in range(-sa_arr[0], sa_arr[0]+1):
		for say in range(-sa_arr[1], sa_arr[1]+1):
			# Coordinate for this possible subaperture is
			sac = [sax, say] * tele['sapitch']
			
			# If we're in an odd row, check saxoffset
			if say % 2 != 0: sac[0] -= tele['saxoff']*tele['sapitch'][0]
			
			# Check if we're in the apterture bounds, and store the subapt
			# position in that case
			if tele['apts'] == 'circular':
				if sum(sac**2) < tele['aptr']**2: pos.append(sac)
			elif tele['apts'] == 'square':
				if (sac < tele['aptr']).all: pos.append(sac)
			else:
				raise ValueError("Unknown aperture shape", tele['apts'], \
					"(should be 'circular' or 'square')")
	
	return array(pos)


def genCn2(maxh=25000, mode=0, n=5000):
	""" Generate a C_n^2 profile up to height 'maxh' in 'n' steps
	gen_cn2(maxh=25000, mode=0, n=5000)
	"""
	# Generate height array and empty C_n^2 profile
	height = linspace(0.0, maxh, n)
	cn2 = zeros(len(height))
	# Generate C_n^2 values
	if mode == 0:
		# Generate H-V 5/7 profile (Tyson p10)
		W = 21.
		A = 1.7e-14
		height /= 1000.0
		cn2 = 5.94e-23 * height**10 * (W/27)**2 * exp(-height) + \
			2.7e-16 * exp(-2 * height / 3) + A * exp(-10 * height)
		height *= 1000.
	else:
		raise ValueError('Unknown mode (should be 0)')
	
	# Return the C_n^2 profile with associated heights
	return array([height, cn2])


def calcR0(cn2, dh, ang=r_[0,0], wavelength=500e-9):
	""" Calculate r_0 values given a C_n^2 profile and height boundaries
	"""
	# The height resolution in the C_n^2 profile 
	ddh = (cn2[0,1] - cn2[0,0])
	
	# Integrate each portion of the C_n^2 profile as dictated by dh
	ints = zeros(len(dh)-1)
	
	for nl in range(len(dh)-1):
		idx = (cn2[0,:] > dh[nl]) & (cn2[0,:] < dh[nl+1])
		ints[nl] = sum(cn2[1, idx]) * ddh
	
	# Total angle given the x and y pointing angle
	totAng = (sum( tan(ang)**2 ))**0.5
	# This is Eqn. 2.13 in Roddier
	r0 = (0.423 * (2 * pi/wavelength)**2. * (cos(totAng)**(-1.)) * \
	 	ints)**(-3./5)
	
	return r0


def readRadialKl(filename, nModes=500, verb=False):
	"""Read in the radial KL profiles from disk, limiting the number of modes
	read in to 'nModes'
	"""
	fd = open(filename, 'r')
	
	n_e = int(fd.readline()) # Number of 'raw' KL modes (unique q's)
	n_r = int(fd.readline()) # Number of radial points per profile
	# Estimate the number of modes present if not set
	n_m = nModes if (nModes) else 2*n_e
	
	# Skip four lines
	for i in range(4):
		fd.next()
	
	# Allocate memory for the various KL quantities
	kl_p = zeros(n_m, int)
	kl_q = zeros(n_m, int)
	kl_e = zeros(n_m, float)
	# Memory for the radial coordinates
	kl_r = zeros(n_r, float)
	# Memory for the radial KL modes
	kl = zeros((n_r, n_m))
	
	# Read in radial coordinates
	for i in range(n_r):
		kl_r[i] = float(fd.next())
	
	# Read in KL modes. File starts with mode 2 (tip), piston is not included.
	jj = 1
	nold = 0
	if verb: prNot("Starting reading in KL modes at ", fd.tell())
	for i in range(n_e):
		# Use the line listing the KL mode number as consistency check
		n = int(fd.next())
		if (n != nold+1):
			raise IOError(-1, ("Reading in KL modes from " + filename + \
				" failed, KL modes do not increment correctly."))
		nold = n
		
		# Read some KL mode properties
		kl_e[jj] = float(fd.next())
		kl_p[jj] = int(fd.next())
		kl_q[jj] = int(fd.next())
		
		# Read in the radial values of the KL mode
		for r in range(n_r):
			kl[r,jj] = float(fd.next())
		
		# Increase the number of KL modes read, stop if we have enough
		jj += 1
		if jj >= n_m: break
		
		# If kl_q is not zero, we can re-use this base-mode
		if kl_q[jj-1] != 0:
			kl_e[jj] = kl_e[jj-1]
			kl_p[jj] = kl_p[jj-1]
			kl_q[jj] = -kl_q[jj-1]
			kl[:,jj] = kl[:,jj-1]
			jj += 1
			if jj >= n_m: break
		
	
	if verb: prNot("Read ", jj, " modes and ", fd.tell()/1000, " kilobytes")
	fd.close()
	
	# Sanity checking here
	if not (alltrue(isfinite(kl)) and alltrue(kl_e >= 0)):
		raise ValueError("Error reading KL modes, some values are not finite")
	
	return {
		'r': kl_r, 
		'e': kl_e, 
		'p': kl_p, 
		'q': kl_q, 
		'kl': kl
	}


def computeFwdMatrix(lh, lsize, lorig, lcells, sasize, sapos, sfang, sffov, matroot='./matrices/', verb=False):
	"""Calculate a forward matrix giving the WFWFS output for a given
	atmosphere geometry.
	"""
	
	# Store all relevant parameters for this matrix in matconf:
	matconf = {'lh': lh,
		'lsize': lsize,
		'lorig': lorig,
		'lcells': lcells,
		'sasize': sasize,
		'sapos': sapos,
		'sfang': sfang,
		'sffov': sffov}	
	# Calculate tag for this configuration
	mattag = calcTag(matconf)
	# See if this matrix is already stored on disk
	matdir = matroot + mattag + '/'
	matfile = 'fwdmatrix.npy'
	prNot("Calculating forward matrix ", mattag)
	# If the file exists, load it from disk and return
	if os.path.exists(matdir + matfile):
		if verb: prNot("Matrix stored on disk, restoring")
		fwdmatrix = load(matdir + matfile)
		return (fwdmatrix, mattag)
	
	# Number of subaps, subfields, layers
	nsa = len(sapos)
	nsf = len(sfang)
	nl = len(lh)
	
	# Matrix width (size of input vector, the atmosphere):
	n = product(lcells) * nl
	# Matrix height (size of output vector, the wfwfs data):
	m = nsa * nsf
	# Init the sparse matrix
	fwdmatrix = zeros((m, n), dtype=float32)
	
	# Base positions for different fields of view:
	basecbl = (tan(sfang - sffov/2.) * lh.reshape(-1,1,1) - sasize/2. - \
		lorig.reshape(-1,1,2) + lsize.reshape(-1,1,2)) * \
		lcells.reshape(1,1,2) / (lsize.reshape(-1,1,2)*2)
	basecur = (tan(sfang + sffov/2.) * lh.reshape(-1,1,1) + sasize/2. - \
		lorig.reshape(-1,1,2) + lsize.reshape(-1,1,2)) * \
		lcells.reshape(1,1,2) / (lsize.reshape(-1,1,2)*2)
	
	# Loop over all subapertures:
	for sa, csapos in zip(range(nsa), sapos):
		# Offset base positions for this subap:
		subcbl = basecbl + (csapos.reshape(1,1,-1) * lcells.reshape(1,1,2) / \
		 	(lsize.reshape(-1,1,2)*2))
		subcur = basecur + (csapos.reshape(1,1,-1) * lcells.reshape(1,1,2) / \
		 	(lsize.reshape(-1,1,2)*2))
		# Make sure all coordinates are within [0,0] to lcells
		blFix = len(subcbl[subcbl > lcells])
		urFix = len(subcur[subcur > lcells])
		if (blFix + urFix > 0):
			prNot("Fixing ", blFix, " and ", urFix, " coordinates")
		
		subcbl[subcbl[:,:,0] > lcells[0], 0] = lcells[0]
		subcbl[subcbl[:,:,1] > lcells[1], 1] = lcells[1]
		subcur[subcur[:,:,0] > lcells[0], 0] = lcells[0]
		subcur[subcur[:,:,1] > lcells[1], 1] = lcells[1]
		# Loop over all subfields:
		for sf, csfang in zip(range(nsf), sfang):
			#csfang = sfang[sf]
			woff = sa * nsf + sf
			# Loop over all atmosphere layers:
			for lay in range(nl):
				# Investigate the which atmospheric cells in layer 'lay'
				# influence subfield 'sf' in subaperture 'sa':
				
				# Old:
				#cpos2 = sasfBounds(csapos, csfang, sasize, sffov, \
				# 	lh[lay], units='cell', rtype='bounds', lorig=lorig[lay], \
				# 	lsize=lsize[lay], lcells=lcells)
				#xr = cpos[:, 0]
				#yr = cpos[:, 1]
				
				xr0 = subcbl[lay,sf,0]
				xr1 = subcur[lay,sf,0]
				yr0 = subcbl[lay,sf,1]
				yr1 = subcur[lay,sf,1]
				off = (lay * lcells[0] * lcells[1])
				
				# Old
				# Calculate the coordinates of the cells intersected and the
				# amount of intersection
				#(isectc, isect) = calcIsect(xr, yr, lcells[0])
				
				# Flatten & insert this in the full matrix
				#fwdmatrix[sa * nsf + sf, \
				#	(isectc + (lay * product(lcells))).reshape(-1).tolist()]=\
				# 	isect.reshape(-1)
				
				for cx in range(int(xr0), int(ceil(xr1))):
					for cy in range(int(yr0), int(ceil(yr1))):
						isect = (min(cx+1, xr1) - max(cx, xr0)) * \
							(min(cy+1, yr1) - max(cy, yr0))
						
						fwdmatrix[woff, off + \
							cy * lcells[0] + cx] = isect
								
	
	# Simple sanity checks (elements should be between 0 and 1), sum for each
	# row must be 1:
	if amax(fwdmatrix) > 1.001: 
		raise ArithmeticError("Maximum in forward matrix greater than 1:", \
		 	amax(fwdmatrix), "at: ", where(fwdmatrix == amax(fwdmatrix)))
	if amin(fwdmatrix) < 0: 
		raise ArithmeticError("Minimum in forward matrix smaller than 0:", \
			amin(fwdmatrix), "at: ", where(fwdmatrix == amin(fwdmatrix)))
	if allclose(fwdmatrix.sum(1),1): 
		raise ArithmeticError("Some rows have sums different than 1.")
	if verb:
		prNot("Simple sanity checks passed.")
	
	# Store the matrix to disk for later use
	try: os.mkdir(matroot)
	except OSError: pass
	except: prErr("Error creating directory")
	
	try: os.mkdir(matdir ) 
	except OSError: pass
	except: prErr("Error creating directory")
	
	save(matdir + matfile, fwdmatrix)
	pickleSave(matdir + 'matconf.pickle', matconf)
	return (fwdmatrix, mattag)


def calcIsect(xr, yr, stride):
	# Calculate the intersection factor for each atmospheric cell
	# Use floor(x0) for the beginning of the range, but ceil(x1-1)
	# for the end of the range such that ranges like [7.3, 8] are
	# chopped off to [7,7]
	# Horizontal intersection:
	hra = arange(floor(xr[0]), ceil(xr[1]-1), dtype=float32)+1
	hv = r_[hra, xr[1]] - r_[xr[0], hra]
	
	# Vertical intersection
	vra = arange(floor(yr[0]), ceil(yr[1]-1), dtype=float32)+1
	vv = r_[vra, yr[1]] - r_[yr[0], vra]
	
	# Cross product gives the intersection of these cells, and
	# normalize by the total surface
	isect = (hv.reshape(1,-1) * vv.reshape(-1,1)) / \
		((xr[1]-xr[0])*(yr[1]-yr[0]))
	
	# isect is only a subsection of the complete layer. The
	# one-dimensional coordinates for each element are given by:
	isectc = arange(floor(xr[0]), ceil(xr[1])).reshape(-1,1) + \
		(arange(floor(yr[0]), ceil(yr[1])) * stride)
		
	return (isectc, isect)


def sasfBounds(sapos, sfang, sasize, sffov, lh, units='real', rtype='bounds', lorig=[], lsize=[], lcells=[]):
	"""Calculate the bounds of a subfield at a certain height lh. If units is
	set to 'real', return real coordinates, if it is set to 'cell' return cell
	coordinates. In the last case, lorig, lsize, lcells must also be 
	specified. If rtype is 'bounds', return [lowerleft, upperright], if set to 
	'range', return [origin, size]"""
	
	# Bottom left corner of the bounds
	bl = sapos - sasize/2. + tan(sfang - sffov/2.) * lh
	# Upper right corner
	ur = sapos + sasize/2. + tan(sfang + sffov/2.) * lh
	
	if units == 'cell':
		# Return coordinates in cell units
		cbl = (bl - lorig + lsize) * lcells / (lsize*2)
		cur = (ur - lorig + lsize) * lcells / (lsize*2)
		if rtype == 'bounds': 
			return r_[[cbl], [cur]]
		elif rtype == 'range':
			return r_[[cbl], [cur-cbl]]
		else: 
			raise ValueError("Unknown rtype", rtype, \
				" (should be 'bounds' or 'range')")
	elif units == 'real':
		# Return the coordinates meter units
		if rtype == 'bounds': 
			return r_[[bl], [ur]]
		elif rtype == 'range':
			return r_[[bl], [ur-bl]]
		else: 
			raise ValueError("Unknown rtype", rtype, \
				" (should be 'bounds' or 'range')")
	else:
		raise ValueError("Unknown units", units, \
			" (should be 'real' or 'cell')")


def computeSvd(matrix, mattag=None, matroot='./matrices/', checkSanity=True, eps=0.1, verb=True):
	"""Compute the singular value decomposition of matrix and optionally do
	some sanity checking of the decomposition"""
	
	if mattag != None:
		#Try to load the SVD from disk
		matdir = matroot + mattag + '/'
		if os.path.exists(matdir + "fwdmatrix-svd-s.npy"):
			if verb: prNot("SVD stored on disk, restoring")
			s = load(matdir + "fwdmatrix-svd-s.npy")
			s_inv = load(matdir + "fwdmatrix-svd-s-inv.npy")
			u = load(matdir + "fwdmatrix-svd-u.npy")
			vh = load(matdir + "fwdmatrix-svd-vh.npy")
			return {'u':u, 
				's':s, 
				's_inv':s_inv, 
				'vh': vh}
	else:
		prWarn("Please supply the parameter mattag to computeSvd() to prevent unnecessary recomputation of the SVD")
	
	# Perform the decomposition, do not use full_matrices, this takes up 
	# *a lot* of memory in certain very non-square matrices
	(u,s,vh) = linalg.svd(matrix, full_matrices=False)
	
	# Exclude (potentially) bad singular values when inverting them
	goodVals = where(s > eps)
	s_inv = zeros(len(s))
	s_inv[goodVals] = 1.0/s[goodVals]
	
	# In numpy:
	#   matrix == dot(u, dot(identity(len(s)) * s, vh))
	#   matrix^-1 == dot(v.T, dot(si, u.T))
	
	if checkSanity:
		# Calculate si = diag(1/s)
		sd = identity(len(s)) * s
		si = identity(len(s)) * s_inv
		
		prNot("SVD shapes: u: ", u.shape, " s: ", s.shape, " vh: ", vh.shape)
		if allclose(matrix, dot(u, dot(sd, vh))):
			prNot("Reconstruction seems to have worked")
		# Try to obtain identity matrix through matrix * matrix^-1 using the
		# SVD components to calculate the inverse.
		idresid = dot(dot(vh.T, dot(si, u.T)), matrix) - identity(len(s))
		prNot("Reconstruction residual, sum: %0.4g avg: %0.4g +- %0.4g" % \
		 	(idresid.sum(), idresid.mean(), idresid.std()))
	
	if mattag != None:
		# Save the SVD components to disk
		matdir = matroot + mattag + '/'
		try: os.mkdir(matroot)
		except OSError: pass
		except: prErr("Error creating directory '" + matroot + "'")
		
		try: os.mkdir(matdir) 
		except OSError: pass
		except: prErr("Error creating directory '" + matdir + "'")
		
		save(matdir + "fwdmatrix-svd-s.npy", s)
		save(matdir + "fwdmatrix-svd-s-inv.npy", s_inv)
		save(matdir + "fwdmatrix-svd-u.npy", u)
		save(matdir + "fwdmatrix-svd-vh.npy", vh)
	else:
		prWarn("Please supply the parameter mattag to computeSvd() to prevent unnecessary recomputation of the SVD")
	
	# Return SVD components as a dict
	return {'u':u, 
		's':s, 
		's_inv':s_inv, 
		'vh': vh}


def cacheSvd(lhs, lsizes, lorigs, lcells, sasize, sapos, sfang, sffov, verb=False):
	"""Precompute the reconstruction matrices, which are the singular value 
	decomposed forward matrices.
	"""
	
	prNot("Building up SVD cache, might take some time...")
	
	# Init memory to hold all SVDs
	svdCache = []
	
	for i in range(len(lhs)):
		prNot("Computing forward matrix ", i, " of ", len(lhs))
		# For each reconstruction geometry, calculate the forwardmatrix:
		print lhs[i]
		print lsizes[i]
		print lorigs[i]
		(fwdmat, mattag) = computeFwdMatrix(lhs[i], lsizes[i], lorigs[i], \
		 	lcells, sasize, sapos, sfang, sffov, verb=True)
		
		prNot("Computing SVD")
		# SVD this forward matrix, and store this
		svdCache.append(computeSvd(fwdmat, mattag=mattag))
	
	# Return the SVD cache
	return svdCache


def cacheKl(nModes=500, res=256, verb=False):
	"""Setup a cache for a number of fully-expanded Karhuhen-Loeve modes
	using the radial profiles as input.
	"""
	# Read in radial KL profiles here
	kl = readRadialKl('output1800.dat', nModes=nModes, verb=verb)
	
	# Make a theta-mask
	thmap = indices((res*2, res*2)) - res
	thmap = arctan2(thmap[0], thmap[1])
	
	# Get a matrix with the normalized distance to the center as values
	rdin = unitRadius(res*2)
	# Make a circular aperture mask here
	support = where(shift(dist(res*2), (res, res)) > res, 0, 1)
	
	# Get the radial values we need to interpolate for
	s_val = (rdin.reshape(-1))
	
	# Init memory for the expanded KL modes, use float32 to save space
	if (res*2)*(res*2)*(nModes)*4 > 2**30: 
		prErr("using more than 1 GiB for klcache,", \
			(res*2)*(res*2)*(nModes)*4 * (2**(-30)), "GiB")
	
	klModes = zeros((nModes+1, res*2, res*2), float32)
	
	if verb: prNot("Constructing ", nModes, " KL modes now, using ", \
	 	(res*2)*(res*2)*(nModes)*4 * (2**(-20)), "MiB")
	
	# Prepare radial KL components
	# TODO: modes 2 and 3 (indices 1 and 2) are tip-and tilt, might exclude
	# those
	# -> 20090114 @ 11:11: only calculated once, no big deal
	for j in range(1, nModes):
		if verb: 
			if j % (nModes/10) == 0: 
				prNot((j*100/nModes), "%")
		# Interpolate the radial KL mode for the values s_val
		kl_int = S.interpolate.interp1d(kl["r"], kl["kl"][:, j])
		# TODO 20081226 @ 16:04
		# This solution is sub-optimal, interpolation is done for 20 times 
		# more points than necessary (there are only about 5% unique points in
		# s_val)
		i_val = kl_int(s_val)
		
		# And insert these in the matrix rd
		rd = (i_val.reshape(res*2, res*2)) * support
		
		# Apply a azimuthal function for q != 0
		#if verb: prNot("Q is %f" % kl["q"][j])
		if kl["q"][j] == 0:
			c = 1.0
		else:
			c = sin(kl["q"][j]*thmap) if (kl["q"][j] > 0) \
				else cos(kl["q"][j]*thmap)
		
		# Store the current mode (j) in the cache memory for later usage
		klModes[j, :, :] = rd * c * sqrt(kl["e"][j])
	
	if not (alltrue(isfinite(klModes))):
		raise ValueError("Error reading KL modes, some values are not finite")
	
	klCache = {
		'n': nModes,
		'th': thmap, 
		'rdin': rdin, 
		'kl': klModes
	}
	
	return klCache


def genPhases(variance, klCache=-1, nModes=500, res=256, seed=-1, verb=False, doRot=False):
	"""For each element in the array variance, generate a phasescreens with
	nModes KL modes with resolution res that with the variance set to that
	element of variance.
	genPhases(variance, klCache=-1, nModes=500, res=256, seed=-1, verb=False):
	"""
	
	# If the seed parameter is set, (re-)seed the RNG for reproducible results
	if seed != -1:
		if verb: prNot("Random seeding with %f" % seed)
		random.seed(seed)
	
	if verb: 
		prNot ('Generating %d KL phase-screens' % len(variance))
	if not (alltrue(variance[:,0] == variance[:,1])):
		raise ValueError("x- and y-variance unequal, not supported (sorry)")
	
	# Verify we have klCache available, load otherwise
	try:
		if klCache["n"] < nModes:
			raise Exception("klCache available but not", \
				"sufficient to construct", nModes, "modes")
	except:
		prNot("klCache not available, loading now for ", nModes)
		klCache = cacheKl(nModes=nModes, res=res, verb=verb)
	
	# Init output variable, which will hold several phase-screens. Use float32
	# to save space and time.
	# TODO: check efficiency of index ordering
	# -> 20090112 @ 13:01: Last indices change fastest (like C)
	klscreens = zeros((len(variance), res*2, res*2), float32)
	
	for l in range(len(variance)):
		for j in range(0, nModes):
			rnd = random.randn()
			#if verb: prNot ("Layer %d, Mode %d, rand: %f" % (l, j, rnd))
			klscreens[l,:,:] += (klCache["kl"][j,:,:]) * rnd
		
		#if verb: 
		#	prNot("Applying variance %f to layer %d" % (variance[l,0], l))
		klscreens[l,:,:] *= variance[l,0]
		# Rotate to check the problem of the inconsistencies
		# TODO: why do we need copy() here? What am I missing?
		if doRot:
			if verb: prNot("Rotating KL screen layer %d" % l)
			tmp = rot90(klscreens[l,:,:])
			klscreens[l,:,:] = tmp.copy()
			
	if verb: 
		prNot("KL phase-screens completed")
	
	return klscreens


def genSlopes(inPhase, cells, klPhase=True, verb=False):
	"""Generate slopes from the continous input phase screens."""
	
	# Number of layers, resolution
	nl = (shape(inPhase))[0]
	res = (shape(inPhase))[1:3]
	# Input pixels per output cell
	ppc = res/cells
	# fitX is used for fitting the phase to a straight line
	fitX = range(ppc[0])
	fitY = range(ppc[1])
	
	# Check if the phasescreen is square
	if (res[0] != res[1]):
		raise ValueError("Phase screens not square, unsupported")
	
	# In case inPhase are KL modes (klPhase is true), cut off some parts of
	# the phase-screen because KL modes are circular.
	# TODO 20090112 @ 10:10
	
	# Init output data
	slopePhase = empty(r_[nl, cells, 2])
	
#	prNot("Calculating slopes, nl: ", nl, " res ", res, " ppc ", ppc, \
#		" fitX/Y ", fitX, fitY)
	
	# Loop over all layers
	for lay in range(nl):
		# Loop over all output cells
		for cx in range(cells[0]):
			for cy in range(cells[1]):
				# Take a crop of the continuous phasescreen layer 
				# if lay == 0:
				# 	prNot("crop: ", cx*ppc[0], "-", (cx+1)*ppc[0], " and ", \
				# 		cy*ppc[0], "-", (cy+1)*ppc[0])
				
				crop = inPhase[lay, 
					cx*ppc[0]:(cx+1)*ppc[0],
					cy*ppc[0]:(cy+1)*ppc[0]]
					
				# Slope in x-direction:
				(slopePhase[lay, cx, cy, 0],b) = polyfit(fitX, crop.sum(0), 1)
				# Slope in y-direction:
				(slopePhase[lay, cx, cy, 1],b) = polyfit(fitY, crop.sum(1), 1)
				# if lay == 0:
				# 	prNot("slopes: ", slopePhase[lay, cx, cy, 0], " & ", \
				# 		slopePhase[lay, cx, cy, 1])
	
	return slopePhase


#=============================================================================
# Generic functions not necessarily related to the ctomo problem start here
#=============================================================================

def calcTag(data):
	"""Calculate a unique tag for 'data'"""
	save('/tmp/calcTagData.npy', data)
	tag = subprocess.Popen(["md5sum /tmp/calcTagData.npy | cut -c 1-32"], \
		stdout=subprocess.PIPE, shell=True).communicate()[0].rstrip()
	return tag


def unitRadius(n):
	"""Construct a square matrix of size n by n with value 0 for elements
	outside a circle with radius n, and the normalized distance to the center
	of the matrix for the other elements, yielding a matrix with a unit-radius
	circle.
	"""
	# Use dist() as source, shift it to get the distance to the center
	rd = shift(dist(n), (n/2, n/2))
	# Normalize the distances
	rd = rd / (n/2)
	# Set the values larger than one to zero
	mask = where(rd>1.0, 0, 1)
	rd = rd * mask
	
	return rd


def pickleSave(file, data):
	"""Save a data object to file using pickling while taking care of file
	operations"""
	
	fd = open(file, 'wb')
	pickle.dump(data, fd)
	fd.close()


def pickleLoad(file):
	"""Load pickled data from a file while taking care of file operations"""
	
	fd = open(file, 'rb')
	data = pickle.load(fd)
	fd.close()
	return data


def prNot(*args):
	"""Print a notice somewhere (logfile, screen)"""
	print '*** ' + ''.join(map(str, args))


def prWarn(*args):
	"""Print a warning somewhere (logfile, screen)"""
	print '!!! ' + ''.join(map(str, args))


def prErr(*args):
	"""Print an error somewhere (logfile, screen)"""
	print 'ERR ' + ''.join(map(str, args))


def rms(data):
	"""Calculate the RMS of the data"""
	return sqrt(mean(data**2.0))


def saveData(config, results, datadir='./', verb=False):
	"""Save results to disk, both in one big blob as well as separete files
	of particular interest"""
	
	if verb:
		prNot("Saving data to " + datadir)
	
	# Save in one big blob
	save(datadir + 'results.npy', results)
	
	# In some ascii files, as function of iteration
	savetxt(datadir + 'inrms-iter.txt', \
		results["inrms"], fmt='%0.6g', delimiter=', ')
	savetxt(datadir + 'recrms-iter.txt', \
		results["recrms"].mean(1), fmt='%0.6g', delimiter=', ')
	# (This is the RMS of the difference between input and reconstruction)
	savetxt(datadir + 'diffrms-iter.txt', \
		results["diffrms"].mean(1), fmt='%0.6g', delimiter=', ')
	# (This is the difference of the input RMS and the reconstruction RMS)
	savetxt(datadir + 'rmsdiff-iter.txt', \
		results["inrms"]-results["recrms"].mean(1), \
		fmt='%0.6g', delimiter=', ')
	
	# In some ascii files, as function of geometry
	savetxt(datadir + 'recrms-geom.txt', \
		results["recrms"].mean(0), fmt='%0.6g', delimiter=', ')
	savetxt(datadir + 'diffrms-geom.txt', \
		results["diffrms"].mean(0), fmt='%0.6g', delimiter=', ')
		
	# Save in and output atmospheres
	


def getData(datadir, conffile='conf.npy', resultsfile='results.npy', verb=False):
	# Try to read in the configuration and results
	if verb:
		prNot("Trying to read in configuration and results.")
	
	# Init return variables to empty tuples
	conf = ()
	results = ()
	
	# Load & reformat the data, it's not immediately ready for use
	if (conffile != None):
		confin = load(datadir + conffile)
		conf = confin.flatten()[0]
	if (resultsfile != None):
		resultsin = load(datadir + resultsfile)
		results = resultsin.flatten()[0]
	
	# Return the data
	return (conf, results)


def plotData(datadir, conffile='conf.npy', resultsfile='results.npy', verb=False):
	"""Make plots for the results stored in datadir"""
	
	# Getting data
	(conf, results) = getData(datadir, conffile, resultsfile, verb=verb)
	
	# Next, plot and analyse the results:
	gp = Gnuplot.Gnuplot(persist = 1)
	
	if verb:
		prNot("Making plots of the data.")
	
	# Plot input rms as function of iteration
	gnuplotInit(gp, hardcopy=datadir+'./plot-wfwfsrms-iter.eps')
	data = results["inrms"]
	gp("set xlabel 'Sample'")
	gp("set ylabel 'RMS [a.u.]'")
	gp("set title 'Input RMS - %.4g +- %.4g'" % \
	 	(data.mean(), data.std()))
	gp('plot "'+datadir + '/inrms-iter.txt" ' + \
		'using ($1) with lines ls 1 title " x-RMS", '+ \
		'"'+datadir + '/inrms-iter.txt" ' + \
		'using ($2) with lines ls 2 title " y-RMS", ' + \
		str(data[:,0].mean()) + 'title "" with lines ls 1, ' + \
		str(data[:,1].mean()) + 'title "" with lines ls 2')
	
	# Plot reconstruction RMS as function of iteration
	gnuplotInit(gp, hardcopy=datadir+'./plot-reconrms-iter.eps')
	data = results["recrms"].mean(1)
	gp("set xlabel 'Sample'")
	gp("set ylabel 'RMS [a.u.]'")
	gp("set title 'Reconstruction RMS - %.4g +- %.4g'" % \
	 	(data.mean(), data.std()))
	gp('plot "'+datadir + '/recrms-iter.txt" ' + \
		'using ($1) with lines ls 1 title " x-RMS", '+ \
		'"'+datadir + '/recrms-iter.txt" ' + \
		'using ($2) with lines ls 2 title " y-RMS", ' + \
		str(data[:,0].mean()) + 'title "" with lines ls 1, ' + \
		str(data[:,1].mean()) + 'title "" with lines ls 2')
	
	# Plot RMS difference as function of iteration
	gnuplotInit(gp, hardcopy=datadir+'./plot-rmsdiff-iter.eps')
	data = results["inrms"]-results["recrms"].mean(1)
	gp("set xlabel 'Sample'")
	gp("set ylabel 'RMS [a.u.]'")
	gp("set title 'Input RMS - Reconstruction RMS - %.4g +- %.4g'" % \
	 	(data.mean(), data.std()))
	gp('plot "'+datadir + '/rmsdiff-iter.txt" ' + \
		'using ($1) with lines ls 1 title " x-RMS", '+ \
		'"'+datadir + '/rmsdiff-iter.txt" ' + \
		'using ($2) with lines ls 2 title " y-RMS", ' + \
		str(data[:,0].mean()) + 'title "" with lines ls 1, ' + \
		str(data[:,1].mean()) + 'title "" with lines ls 2')
	
	# Plot difference RMS as function of iteration
	gnuplotInit(gp, hardcopy=datadir+'./plot-diffrms-iter.eps')
	data = results["diffrms"].mean(1)
	gp("set xlabel 'Sample'")
	gp("set ylabel 'RMS [a.u.]'")
	gp("set title 'RMS of measured data - reconstructed data - %.4g +- %.4g'" % \
	 	(data.mean(), data.std()))
	gp('plot "'+datadir + '/diffrms-iter.txt" ' + \
		'using ($1) with lines ls 1 title " x-RMS", '+ \
		'"'+datadir + '/diffrms-iter.txt" ' + \
		'using ($2) with lines ls 2 title " y-RMS", ' + \
		str(data[:,0].mean()) + 'title "" with lines ls 1, ' + \
		str(data[:,1].mean()) + 'title "" with lines ls 2')
	
	# Plot (averaged) difference RMS as function of geometry
	gnuplotInit(gp, hardcopy=datadir+'./plot-diffrms-geom.eps')
	data = results["diffrms"].mean(0)
	gp("set xlabel 'Sample'")
	gp("set ylabel 'Geometry'")
	gp("set title 'RMS of measured data - reconstructed data - %.4g +- %.4g'" % \
	 	(data.mean(), data.std()))
	gp('plot "'+datadir + '/diffrms-geom.txt" ' + \
		'using ($1) with linespoints ls 1 title " x-RMS", '+ \
		'"'+datadir + '/diffrms-geom.txt" ' + \
		'using ($2) with linespoints ls 2 title " y-RMS"')
	
	# Plot (averaged) reconstruction RMS as function of geometry
	gnuplotInit(gp, hardcopy=datadir+'./plot-recrms-geom.eps')
	data = results["recrms"].mean(0)
	gp("set xlabel 'Sample'")
	gp("set ylabel 'Geometry'")
	gp("set title 'RMS of reconstructed data - %.4g +- %.4g'" % \
	 	(data.mean(), data.std()))
	gp('plot "'+datadir + '/recrms-geom.txt" ' + \
		'using ($1) with linespoints ls 1 title " x-RMS", '+ \
		'"'+datadir + '/recrms-geom.txt" ' + \
		'using ($2) with linespoints ls 2 title " y-RMS"')
	
	# Wait until gp is complete by checking the existence of the eps files
	# TODO: bad method, improve this
	while not os.path.exists(datadir+'./plot-recrms-geom.eps'):
		time.sleep(0.1)
	
	# Just to be sure that a previously called gp() didn't finish later...
	time.sleep(0.2)
	
	if verb:
		prNot("Converting EPS files to PDF files.")
	
	# Convert EPS files to PDF files using eps2pdf
	ret = subprocess.call(["cd " + datadir + \
		"; for i in `ls *eps`; do epstopdf $i; done; cd -"], shell=True)
	if ret != 0:
		raise IOError(-1, ("Converting eps files to PDF failed"))
	
	# Combine some plots
	ret = subprocess.call(["cd " + datadir + \
		"; pdftk plot-wfwfsrms-iter.pdf plot-rmsdiff-iter.pdf  plot-diffrms-geom.pdf plot-recrms-geom.pdf cat output plots-combined.pdf"], \
		shell=True)
	if ret != 0:
		raise IOError(-1, ("Converting eps files to PDF failed"))
		


def gnuplotInit(gp, hardcopy=False, verb=False):
	"""Set some default gnuplot options for Gnuplot instance gp"""
	
	if verb: prNot("Initializing gnuplot settings")
	# First reset gnuplot completely
	gp.reset()
	
	# If we want a hardcopy, do so
	if hardcopy:
		if verb: prNot("Saving hardcopy to " + hardcopy)
		gp('set terminal postscript eps enhanced color size 8.8cm,5.44cm "Palatino-Roman" 10')
		gp('set output "' + hardcopy + '"')
	
	gp('set key on box spacing 2 samplen 6')
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


def main():
	pass


if __name__ == '__main__':
	main()


