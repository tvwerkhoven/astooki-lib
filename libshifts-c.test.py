#!/usr/bin/env python2.5
# encoding: utf-8

### Test libshifts-c library
import sys
import pyana
import libsh
import numpy as N
import scipy
import _libshifts			# C version of libshifts
import libshifts			# Python version of libshifts
import time						# For timing

def main(realdata=True):
	if (realdata):
		ddir = '../data/2009.04.28-run01/'
		pdir = ddir + 'proc/'
		imfile = 'wfwfs_test_im28Apr2009.0000100'
	
		# load SA / SF config
		(nsa, saccdpos, saccdsize) = libsh.loadSaSfConf(pdir+'2009.04.28-mask.csv')
		(nsf, sfccdpos, sfccdsize) = \
			libsh.loadSaSfConf(pdir+'2009.04.28-subfield-big.csv')
	
		nsa = len(saccdpos)
		img = pyana.getdata(ddir+imfile)
		img = img.astype(N.float32)
	else:
		# Make fake image with gaussian
		im1 = mk2dgauss((512,512), (153.5, 150.6))
		im2 = mk2dgauss((512,512), (300, 150.))
		im3 = mk2dgauss((512,512), (151.2, 303.3))
		im4 = mk2dgauss((512,512), (302.4, 300.3))
		img = (im1+im2+im3+im4).astype(N.float32)
		saccdpos = N.array([[150, 150], [300, 150], [150, 300], [300, 300]], dtype=N.int32) - N.int32([32,32])
		saccdsize = N.array([64,64], dtype=N.int32)
		sfccdpos = N.array([[40, 40]], dtype=N.int32)
		sfccdsize = N.array([12, 12], dtype=N.int32)
		#saccdpos = saccdpos[:2]
		nsa = len(saccdpos)
	
	#
	# C libshift code
	#
	
	beg1 = time.time()
	datr = _libshifts.calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, N.array([7,7]))
	end1 = time.time()
	
	# 
	# Python libshift code
	#
	
	# beg2 = time.time()
	# pyshift = libshifts.calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, method=libshifts.COMPARE_SQDIFF, extremum=libshifts.EXTREMUM_2D9PTSQ, refmode=libshifts.REF_BESTRMS, refopt=1, shrange=[7,7])
	# end2 = time.time()
	
	refsa = datr['refapts'][0]
	ref = img[saccdpos[refsa][1]:saccdpos[refsa][1]+saccdsize[1], saccdpos[refsa][0]:saccdpos[refsa][0]+saccdsize[0]].astype(N.float32)
	ref = ref/N.float32(ref.mean())
	pyshift = []
	for sa in xrange(nsa):
		pos = saccdpos[sa]
		c = img[pos[1]:pos[1]+saccdsize[1], pos[0]:pos[0]+saccdsize[0]]
		print "py: sa %d @ (%d,%d), mean: %g... " % (sa, pos[0], pos[1], c.mean()),
		c = c / c.mean()
		_subfield = c[sfccdpos[0][1]:sfccdpos[0][1]+sfccdsize[1], \
			sfccdpos[0][0]:sfccdpos[0][0]+sfccdsize[0]].astype(N.float32)
		diffmap = libshifts.sqDiffWeave(_subfield, ref, sfccdpos[0], N.array([7,7]))
		print "sf 0 @ mean: %g... " % (_subfield.mean()),	
		diffmap = diffmap.astype(N.float32)
		print "max:  %g..." % (diffmap.max()),
		pyshift.append(libshifts.quadInt2dWeave(diffmap, range=N.array([7,7]), limit=N.array([7,7])))
		print "sh: (%g, %g)." % (pyshift[-1][0], pyshift[-1][1])
	pyshift = N.array(pyshift)
	end2 = time.time()
	return
	print datr['shifts'].shape, datr['shifts'][0,:,0,:].mean(axis=0).reshape(1,2)
	print pyshift.shape, pyshift[0,:,0,:].mean(axis=0).reshape(1,2)
	datr['shifts'][0,:,0,:] -= datr['shifts'][0,:,0,:].mean(axis=0).reshape(1,2)
	pyshift[0,:,0,:] -= pyshift[0,:,0,:].mean(axis=0).reshape(1,2)
	for sa in xrange(nsa):
		print "diff @ sa %d:" % (sa), datr['shifts'][0][sa] - pyshift[0][sa], datr['shifts'][0][sa], pyshift[0][sa]
	
	print "C took: %g sec, Python took %g." % (end1-beg1, end2-beg2)
	# tdiff = N.sum(abs((datr['shifts'][0][:][0] - pyshift[:])))
	# maxdiff = N.max(abs((datr['shifts'][0][:][0] - pyshift[:])))
	# print "Total difference: %g. Max: %g" % (tdiff, maxdiff)

def mk2dgauss(size, orig):
	"""Function to make nice Gaussian shapes"""
	im = N.indices(size)
	dat1 = N.exp(-(im[0]-orig[1])**2.0/300.)
	dat2 = N.exp(-(im[1]-orig[0])**2.0/300.)
	dat = (dat1*dat2)/(dat1*dat2).max()
	return dat


if __name__ == "__main__":
	sys.exit(main(realdata=True))
