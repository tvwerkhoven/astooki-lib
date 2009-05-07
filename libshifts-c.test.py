#!/usr/bin/env python2.5
# encoding: utf-8

### Test libshifts-c library
import sys
import pyana
import pyfits
import libsh
import numpy as N
import scipy as S
import clibshifts			# C version of libshifts
import clibshifts as cs
import libshifts			# Python version of libshifts
import time						# For timing

def main(realdata='real'):
	if (realdata == 'wfwfs'):
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
	elif (realdata == 'fake'):
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
	elif (realdata == 'real'):
		# Use a big image as source
		imfile = '../poster-crisp-intensity.fits'
		img = pyfits.getdata(imfile)
		# Crop about 400 by 400 pixels
		crop = img[400:600,0:200]
		# Blow up the crop by a factor of 2
		crop = S.ndimage.zoom(crop, 2)
		# Take whole image as reference
		ref = crop.copy()
		# List of shift vectors
		#shvec = N.array([[0,0.5], [1,1], [0,0], [0.5,0.5], [3.5, 1.2]])
		shx = N.arange(20.)/10.
		shy = N.arange(10.)/3.0
		shvec = N.array([ [i, y] for y in shx for i in shy ])
		imlist = []
		# Shift images around, and downscale by a factor of 10
		for sh in shvec:
			#print "Shvec: (%g,%g)." % (sh[0], sh[1])
			sh *= 10.
			im = crop[40+sh[1]:-40+sh[1],40+sh[0]:-40+sh[0]].copy()
			im = S.ndimage.zoom(im, 1.0/10.)
			imlist.append(im)
			sh /= 10.
		refsm = S.ndimage.zoom(ref, 1.0/10.)
		# print refsm.shape
		# print im.shape
		
		# Put data in big image:
		nx = N.ceil(N.sqrt(len(shvec)+1))
		bigimg = N.empty((nx*40, nx*40))
		# Put reference frame in
		bigimg[0:40, 0:40] = refsm
		# Put other images in the big image
		x=1
		y=0
		saccdpos = [[4,4]]
		saccdsize = N.array([32,32])
		for im in imlist:
			saccdpos.append([x*40+4,y*40+4])
			bigimg[y*40+4:y*40+4+32, x*40+4:x*40+4+32] = im
			x+=1
			if (x >= nx):
				x=0
				y+=1
			if (y >= nx):
				print "Oops, shouldnt happen :P"
				break
		saccdpos = N.array(saccdpos)
		# Static 'subfield' positions
		sfccdpos = N.array([4,4])
		sfccdsize = N.array([24,24])
		
		# Now process the data!
		diff = {}
		for meth in [cs.COMPARE_ABSDIFFSQ, cs.COMPARE_SQDIFF, cs.COMPARE_XCORR]:
			for intpl in [cs.EXTREMUM_2D9PTSQ, cs.EXTREMUM_2D5PTSQ]:
				cshift = cs.calcShifts(bigimg.astype(N.float32), saccdpos, saccdsize, sfccdpos, sfccdsize, shrange=[4,4], method=meth, extremum=intpl, refmode=cs.REF_STATIC, refopt=[0])
				print shvec[12], cshift[0,1:,0][12]
				diff['meth:%d-int:%d' % (meth, intpl)] = shvec - cshift[0,1:,0]
		
		print "Meth ADSQ: %d, SQD: %d, XCORR: %d" % (cs.COMPARE_ABSDIFFSQ, cs.COMPARE_SQDIFF, cs.COMPARE_XCORR)
		print "Int 9p: %d, 5p: %d" % (cs.EXTREMUM_2D9PTSQ, cs.EXTREMUM_2D5PTSQ)
		
		for key, val in diff.items():
			print "%s:" % (key),
			print "%g, %g, %g, %g" % (N.sum(N.abs(val)), N.max(N.abs(val)), N.mean(N.abs(val)), N.var(N.abs(val))**0.5)
		
		# for (sh, recsh) in zip(shvec, cshift[0,1:,0]):
		# 	print "shift was: %g,%g, found: %g,%g" % \
		# 		(sh[0]/10., sh[1]/10., recsh[0], recsh[1])
		return -1
		
	else:
		print "Data '%s' not supported" % (realdata)
		return -1
		
			
		

		src = N.random.random((8,8))
		src = S.ndimage.zoom(src, 8)
		#img = src[]
	#
	# C libshift code
	#
	
	# beg1 = time.time()
	# datr = _libshifts.calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, N.array([7,7]), 2, 0, 0, 2)
	# # comp, int, refmode, refopt
	# end1 = time.time()
	beg1 = time.time()
	creflist = []
	cshift = clibshifts.calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, shrange=[7,7], method=clibshifts.COMPARE_ABSDIFFSQ, extremum=clibshifts.EXTREMUM_2D9PTSQ, refmode=clibshifts.REF_BESTRMS, refopt=2, refaps=creflist)
	# comp, int, refmode, refopt
	end1 = time.time()
	
	# 
	# Python libshift code
	#
	
	beg2 = time.time()
	
	pyreflist = []
	pyshift = libshifts.calcShifts(img, saccdpos, saccdsize, sfccdpos, sfccdsize, method=libshifts.COMPARE_ABSDIFFSQ, extremum=libshifts.EXTREMUM_2D9PTSQ, refmode=libshifts.REF_BESTRMS, refopt=2, shrange=[7,7], refaps = pyreflist)
	end2 = time.time()
	
	
	# refsa = datr['refapts'][0]
	# ref = img[saccdpos[refsa][1]:saccdpos[refsa][1]+saccdsize[1], saccdpos[refsa][0]:saccdpos[refsa][0]+saccdsize[0]].astype(N.float32)
	# ref = ref/N.float32(ref.mean())
	# pyshift = []
	# for sa in xrange(nsa):
	# 	pos = saccdpos[sa]
	# 	c = img[pos[1]:pos[1]+saccdsize[1], pos[0]:pos[0]+saccdsize[0]]
	# 	print "py: sa %d @ (%d,%d), mean: %g... " % (sa, pos[0], pos[1], c.mean()),
	# 	c = c / c.mean()
	# 	_subfield = c[sfccdpos[0][1]:sfccdpos[0][1]+sfccdsize[1], \
	# 		sfccdpos[0][0]:sfccdpos[0][0]+sfccdsize[0]].astype(N.float32)
	# 	#diffmap = libshifts.sqDiffWeave(_subfield, ref, sfccdpos[0], N.array([7,7]))
	# 	diffmap = libshifts.absDiffSqWeave(_subfield, ref, sfccdpos[0], N.array([7,7]))
	# 	print "sf 0 @ mean: %g... " % (_subfield.mean()),	
	# 	diffmap = diffmap.astype(N.float32)
	# 	print "max:  %g..." % (diffmap.max()),
	# 	pyshift.append(libshifts.quadInt2dWeave(diffmap, range=N.array([7,7]), limit=N.array([7,7])))
	# 	print "sh: (%g, %g)." % (pyshift[-1][0], pyshift[-1][1])
	# pyshift = N.array(pyshift)
	# end2 = time.time()
	
	for sa in xrange(nsa):
		print "diff @ sa %d:" % (sa), cshift[0][sa] - pyshift[0][sa], cshift[0][sa], pyshift[0][sa]
	
	print creflist
	print pyreflist
	
	print cshift.shape, cshift[0,:,0,:].mean(axis=0).reshape(1,2)
	print pyshift.shape, pyshift[0,:,0,:].mean(axis=0).reshape(1,2)

	print "C took: %g sec, Python took %g." % (end1-beg1, end2-beg2)
	
	# datr['shifts'][0,:,0,:] -= datr['shifts'][0,:,0,:].mean(axis=0).reshape(1,2)
	# pyshift[0,:,0,:] -= pyshift[0,:,0,:].mean(axis=0).reshape(1,2)
	
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
	sys.exit(main(realdata='real'))
