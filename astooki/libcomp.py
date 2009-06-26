#!/usr/bin/env python2.5
# encoding: utf-8
##
# @file libcomp.py
# @brief (Grid/Parallel/Distributed) computation library
# @author Tim van Werkhoven (tim@astro.su.se)
# @date 20090320
# 
# This library provides functions to distribute computation over different 
# MPI threads. This package is currently deprecated.
# 
# Created by Tim van Werkhoven on 2009-03-20.
# Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)
# 
# This file is licensed under the Creative Commons Attribution-Share Alike
# license versions 3.0 or higher, see
# http://creativecommons.org/licenses/by-sa/3.0/

#=============================================================================
# Import libraries here
#=============================================================================

import liblog as log		# To print & log messages
from mpi4py import MPI
import numpy as N			# for creating and modifying numpy buffers
import array				# for creating python buffers
import socket				# For getting IP addresses
import ConfigParser			# For parsing the config file
import os					# For some dir manipulation

#=============================================================================
# CompGrid(), computation distribution class
#=============================================================================

class CompGrid():
	"""
	This class provides a framework to distribute work over multiple MPI
	threads and computers using a generic setup such that any data can be 
	processed. It uses MPI through mpi4py to distribute the work.
	"""
	def __init__(self, cfgfile):
		# Static configuration options
		# Various communication tags
		self.COMM_WD_W = 1				# watchdog to worker
		self.COMM_W_WD = 2				# worker to watchdog
		
		self.cfgdef = { \
			'masterhost' : 'dhcp-179.astro.su.se',\
			}
		
		log.prNot(log.NOTICE, "Initializing CompGrid()")
		# Load configuration from cfgfile
		self.cfg = ConfigParser.SafeConfigParser(self.cfgdef)
		self.cfg.read(cfgfile)
		
		# Save config filename
		self.cfgfile = cfgfile
		
		# Change directory to that of the configfile
		self.curdir = os.path.realpath(os.path.curdir)
		os.chdir(os.path.realpath(os.path.dirname(cfgfile)))
		
		# Parse data format and structure variables
		self.masterhost = self.cfg.get('compgrid', 'masterhost')
		self.sbuflen = self.cfg.get('compgrid', 'sbuflen')
		self.rbuflen = self.cfg.get('compgrid', 'rbuflen')
				
		# These will hold the various different processes
		self.procs = []						# List for all MPI threads
		self.watchdog = -1					# watchdog MPI rank
		self.workers = []					# list of all worker MPI threads
		self.iworkers = []					# list of idle worker MPI threads
		# Send & receive requests buffer
		self.sreqbuf = {}
		self.rreqbuf = {}
		# TODO: doc
		#self.metabuf = {}
		# Send & receive data buffers
		self.sendbuf = {}
		self.recvbuf = {}
		# Buffer for keeping track of the data ID (i.e. what data is stored in 
		# each buffer?)
		self.idbuf = {}
		
		self.comm = MPI.COMM_WORLD			# world communicator
		self.size = self.comm.Get_size()	# number of slaves
		self.rank = self.comm.Get_rank()	# current rank
		
		# Only do checks in one MPI thread
		if (self.rank == 0):
			# Check configuration file for sanity
			self.checkSetupSanity()
		
		# Change directory back
		os.chdir(self.curdir)
		
		# Setup datatype conversion
		self.setupTypes()
		
		# Divide tasks over MPI threads
		self.divideTasks()
	
	
	def checkSetupSanity(self):
		"""
		Check setup. Currently just a placeholder.
		"""
		
		pass
	
	
	def linkFuncs(self, wdfunc, wdargs, wfunc, wargs):
		"""
		Link callback functions to real functions.
		"""
		# Semi-static configuration. These are the functions and arguments 
		# that the various tasks will call.
		self.watchdogfunc = wdfunc
		self.watchdogargs = wdargs
		self.workerfunc = wfunc
		self.workerargs = wargs
	
	
	def divideTasks(self):
		"""
		Divide the tasks over various MPI threads.
		"""
		log.prNot(log.NOTICE, "divideTasks(): dividing tasks over threads.")
		
		# Everyone sends their ip to the master thread
		locip = socket.gethostbyname(socket.gethostname())
		sreq = self.comm.Isend([locip, MPI.CHARACTER], dest=0)
		# Store groups per IP here
		# NB: although the workers are still sorted per IP, this is not used 
		# for anything. Sort-of deprecated I guess.
		groups = {}
		
		# All other threads besides rank 0 do not continue.
		if (self.rank != 0):
			# wait until we're sure the message was received
			log.prNot(log.INFO, "divideTasks(): waiting for send")
			sreq.Wait()
			log.prNot(log.INFO, "divideTasks(): success.")
			return
		
		# Only MPI thread rank 0 remains here:
		for proc in xrange(self.size):
			# Receive ip from all MPI threads, store with thread ID ('rank')
			buf = array.array('c', '0'*15)
			stat = MPI.Status()
			ret = self.comm.Recv([buf, MPI.CHARACTER], source=proc, \
			 	status=stat)
			ip = buf.tostring()
			self.procs.append([proc, ip])
			log.prNot(log.INFO, "divideTasks(): received '%s' from %d." % \
				(ip, proc))
			
			# Now make groups per IP
			if (not groups.has_key(ip)):
				groups[ip] = []
			groups[ip].append(proc)
		
		log.prNot(log.INFO, "divideTasks(): making task groups.")
		foundmaster = False
		
		# Loop over groups, divide tasks
		for ip in groups:
			if (ip == self.masterhost):
				log.prNot(log.INFO, "divideTasks(): found master ip.")
				foundmaster = True
				
				# We need at least one MPI threads for the watchdog
				if (len(groups[ip]) < 1):
					raise RuntimeError("Master machine must have at least one MPI threads!")
					
				# Sort the group. The lowest ranks on the master host will be 
				# watchdog.
				groups[ip].sort()
				self.watchdog = groups[ip][0]
				self.workers.extend(groups[ip][1:])
				self.iworkers.extend(groups[ip][1:])
			else:
				self.workers.extend(groups[ip])
				self.iworkers.extend(groups[ip])
			log.prNot(log.INFO, "divideTasks(): group %s has %d nodes." % \
				(ip, len(groups[ip])))
		
		if (not foundmaster):
			raise RuntimeError("Master machine '%d' not found!" % \
			 	self.masterhost)
		
		log.prNot(log.NOTICE, "divideTasks(): watchdog @ %d, workers:" % \
		 	(self.watchdog))
		log.prNot(log.NOTICE, self.workers)
	
	
	def deploy(self, ack=None):
		"""
		Commit various MPI threads to their respective tasks as defined 
		earlier.
		"""
		if (ack != None):
			log.prNot(log.NOTICE, "deploy(): %s" % (ack))
		
		if (self.rank == self.watchdog):
			log.prNot(log.NOTICE, "deploy(): MPI thread %d running watchdog." %\
				(self.rank))
			self.watchdogfunc(self.watchdogargs)
		else:
			log.prNot(log.NOTICE, "deploy(): MPI thread %d running worker." %\
				(self.rank))
			self.workerfunc(self.workerargs)
		
		log.prNot(log.NOTICE, "deploy(): MPI thread %d done working." %\
			(self.rank))
	
	
	def setupTypes(self):
		"""
		Setup dicts to convert datatypes between Numpy <-> MPI <-> Meta type.
		self._tmpitonp is MPI -> NumPy
		self._tnptompi is NumPy -> MPI
		self._tnptometa is NumPy -> Meta type
		self._tnptometa is Meta type -> NumPy
		"""
		
		# MPI types are not hashable, so we use a less than optimal 
		# list-approach. Should work fine, just not very elegant.
		self._tmpiandnp = [[MPI.CHAR, N.int8], \
			[MPI.SHORT, N.int16], \
			[MPI.INT, N.int32], \
			[MPI.LONG, N.int64], \
			[MPI.UNSIGNED_CHAR, N.uint8], \
			[MPI.UNSIGNED_SHORT, N.uint16], \
			[MPI.UNSIGNED, N.uint32], \
			[MPI.UNSIGNED_LONG, N.uint64], \
			[MPI.FLOAT, N.float32], \
			[MPI.DOUBLE, N.float64], \
			[MPI.DOUBLE, N.float], \
			[MPI.LONG_DOUBLE, N.float128], \
			[MPI.BYTE, N.byte], \
			[MPI.PACKED, False]]
		
		# Internal types for MPI metadata
		self.int8 = 1
		self.uint8 = 2
		self.int16 = 3
		self.uint16 = 4
		self.int32 = 5
		self.uint32 = 6
		self.int64 = 7
		self.uint64 = 8
		self.float32 = 9
		self.float64 = 10
		self.float128 = 11
		self.byte = 12
			
		# Numpy to Meta type
		self._tnptometa = {\
			N.int8: self.int8,\
			N.uint8: self.uint8,\
			N.int16: self.int16,\
			N.uint16: self.uint16,\
			N.int32: self.int32,\
			N.int: self.int32,\
			N.uint32: self.uint32,\
			N.int64: self.int64,\
			N.uint64: self.uint64,\
			N.float32: self.float32,\
			N.float64: self.float64,\
			N.float: self.float64,\
			N.float128: self.float128,\
			N.byte: self.byte,\
			}
		# Reverse, Meta to NumPy:
		self._tmetatonp = {}
		for key in self._tnptometa.keys():
			self._tmetatonp[self._tnptometa[key]] = key
	
	
	def tmpitonp(self, dtype):
		"""
		Wrapper for self._tmpiandnp
		"""
		for (tmpi, tnp) in self._tmpiandnp:
			if (tmpi == dtype): return tnp
		return False
	
	
	def tnptompi(self, dtype):
		"""
		Wrapper for self._tmpiandnp
		"""
		for (tmpi, tnp) in self._tmpiandnp:
			if (tnp == dtype): return tmpi
		return False
	
	
	def tnptometa(self, dtype):
		"""
		Wrapper for self._tnptometa
		"""
		return self._tnptometa[dtype]
	
	
	def tmetatonp(self, dtype):
		"""
		Wrapper for self._tmpiandnp
		"""
		return self._tmetatonp[dtype]
	
	
	def broadcastWorkers(self, data, task):
		"""
		Broadcast something to all workers. Will block until messages are 
		received.
		"""
		
		log.prNot(log.INFO, "broadcastWorkers(): Broadcasting data.")
		
		# Send metadata first, don't track the req because we don't need to
		meta = self.getMeta(data, task)
		
		sreqs = []
		for worker in self.workers:
			self.comm.Isend([meta, MPI.INT], dest=worker, tag=self.COMM_WD_W)
			sreq = self.comm.Isend([data, self.tnptompi(data.dtype.type)], \
			 	dest=worker, tag=self.COMM_WD_W)
			sreqs.append(sreq)
		
		# Wait for send to complete
		MPI.Request.Waitall(sreqs)
	
	
	def sendToWorker(self, data, dataid, task, rbuf):
		"""
		This function sends a frame out to an idle worker thread. This should 
		be called from the watchDog MPI thread parsing the files. 'task' 
		should be a valid MPI tag (integer) and describe one of the tasks that
		a worker can perform on a frame. The frame is sent with 'task' as tag, 
		and a receive with the same task is scheduled. 'rbuf' should be a 
		receive buffer for the data expected.
		"""
			
		# If there are no idle workers, everyone is busy and we should get 
		# results back first.
		if (len(self.iworkers) == 0):
			return False
		
		# If the receive buffer is full, we should wait for data first
		if (len(self.recvbuf) >= self.rbuflen):
			return False
		
		# If the send buffer is full, wait until a send operation has 
		# completed before continuing
		if (len(self.sendbuf) >= self.sbuflen):
			log.prNot(log.INFO, "sendToWorker(): %d: sendbuf full (%d elems)" %\
			 	(rank, len(self.sendbuf)))
			sstat = MPI.Status()
			wreq = MPI.Request.Waitany(self.sreqbuf.values(), status=sstat)
			# At least one request must be done now, test all to see which 
			# requests that are
			for sreq in self.sreqbuf:
				if MPI.Request.Test(self.sreqbuf[sreq]):
					log.prNot(log.INFO,  "sendToWorker(): %d: found '%d' is done, removing" % (rank, sreq))
					del self.sreqbuf[sreq]
					del self.sendbuf[sreq]
					#del self.metabuf[sreq]
					break
		
		# Get an idle worker
		w = self.iworkers.pop(0)
		
		# Send metadata first, don't track the req because we don't need to
		meta = self.getMeta(data, task)
		log.prNot(log.INFO, "sendToWorker(): Sending out data.")
		self.comm.Isend([meta, MPI.INT], dest=w, tag=self.COMM_WD_W)
		# Request async send for data
		sreq = self.comm.Isend([data, self.tnptompi(data.dtype.type)], \
		 	dest=w, tag=self.COMM_WD_W)
		# Store request and data in buffers
		self.sreqbuf[w] = sreq
		self.sendbuf[w] = data
		#self.metabuf[w] = meta
		
		# Request async receive from worker
		rdtype = self.tnptompi(rbuf.dtype.type)
		rreq = self.comm.Irecv([rbuf, rdtype], source=w, tag=self.COMM_W_WD)
		# Keep track of what we're receiving and what buffers we use
		self.rreqbuf[w] = rreq
		self.recvbuf[w] = rbuf
		self.idbuf[w] = dataid
		
		# Submit was successful, return True		
		log.prNot(log.INFO, "sendToWorker(): Success.")
		return True
	
	
	def getFromWorker(self):
		"""
		This function is meant to receive results from worker threads.
		"""
		# TODO: Document
		# TODO: fix ugly (False, False) return values
		log.prNot(log.INFO, "getFromWorker(): Getting data from worker.")
		
		# Check if there are requests left on the buffer:
		if (len(self.rreqbuf) == 0):
			# No requests left, so we don't expect data to return
			return (False, False)
			
		# Wait for data
		rstat = MPI.Status()
		wreq = MPI.Request.Waitany(self.rreqbuf.values(), status=rstat)
		log.prNot(log.INFO, "getFromWorker(): worker %d done crunching" % \
			(rstat.source))
		
		# Construct return value
		retval = (self.idbuf[rstat.source], self.recvbuf[rstat.source])
		
		# Remove entries from buffer lists
		del self.recvbuf[rstat.source]
		del self.idbuf[rstat.source]
		del self.rreqbuf[rstat.source]
		
		# Add idle worker back to list
		self.iworkers.append(rstat.source)
		
		# Return results
		return retval
	
	
	def getFromMaster(self):
		"""
		Receive data from the master to the worker MPI thread.
		"""
		log.prNot(log.INFO, "getFromMaster(): waiting for metadata.")
		
		# First we receive some metadata on the data we're getting after this.
		meta = N.empty((32), dtype=N.int32)
		recv = self.comm.Recv([meta, MPI.INT], source=0, tag=self.COMM_WD_W)
		
		# Buf now has the following structure:
		# meta[0]: number of dimensions of the data coming
		# meta[1:9]: size of each dimension. Only the first meta[0] elements
	 	#            of this slice are valid and should be used
		# meta[10]: datatype.
		
		# Now we set up the buffer to receive the actual data
		databuf = N.empty(tuple(meta[1:1+meta[0]]), \
		 	dtype=self.tmetatonp(meta[10]))
		
		log.prNot(log.INFO, "getFromMaster(): got metadata.")
		# And block to receive the data
		self.comm.Recv([databuf, self.tnptompi(databuf.dtype.type)], \
		 	source=0, tag=self.COMM_WD_W)
		
		return (databuf, meta[11])
	
	
	def sendToMaster(self, result):
		"""
		Send results to the master MPI thread. This function blocks until the 
		send is complete.
		"""
		self.comm.Send([result, self.tnptompi(result.dtype.type)], dest=0, \
		 	tag=self.COMM_W_WD)
	
	
	def getMeta(self, data, task):
		"""
		Generate meta-data list to send somewhere. Will generate a list 'meta'
		of 32 32-bit integers with the following syntax:
		
		meta[0]: number of dimensions (max 8)
		meta[1:9]: size of each dimension
		meta[10]: datatype
		meta[11]: task
		meta[12:]: unused/reserved
		"""
		
		if (len(data.shape) > 8):
			raise RuntimeError("Number of dimensions higher than 8 not supported.")
		
		# Dimension info
		meta = N.zeros((32), dtype=N.int32)
		meta[0] = len(data.shape)
		for dim in range(meta[0]):
			meta[dim+1] = data.shape[dim]
		
		# Datatype info
		meta[10] = self.tnptometa(data.dtype.type)
		# Task info
		meta[11] = task
		
		# Done, return the list
		return meta
	

