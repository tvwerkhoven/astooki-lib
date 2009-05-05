#!/usr/bin/env python
# encoding: utf-8
"""
liblog.py

Some routines for printing and logging messages to screen, disk or whatever.

Created by Tim van Werkhoven on 2009-03-30.
Copyright (c) 2009 Tim van Werkhoven (tim@astro.su.se)

This file is licensed under the Creative Commons Attribution-Share Alike
license versions 3.0 or higher, see
http://creativecommons.org/licenses/by-sa/3.0/
"""
#=============================================================================
# Import libraries here
#=============================================================================

import sys
import time

#=============================================================================
# Defines
#=============================================================================

# # Various levels of verbosity (make sure these increment nicely)
# VERB_ERROR = -2				# Print fatal problems with this code, and exit
# VERB_WARN = -1				# Print non-fatal problems with this code
# VERB_SILENT = 0				# Always print these messages
# VERB_INFO = 1					# Print as info (useful runtime info)
# VERB_DEBUG = 2				# Print as debug (useful debug info)
# VERB_ALL = 3					# Print lots of debug (useless debug info)

# Shorter synonyms of the above, use these instead
# EMERG = 0
# ALERT = 1
# CRIT = 2
ERR = 3
WARNING = 4
NOTICE = 5
INFO = 6
DEBUG = 7

# Exit with this code
EXIT = -1

LOGFILE = None
LOGFD = 0
LOGLASTDAY = 0

VERBOSITY = 4

# Some colors
RESETCL = "\033[0m"
# Debug is blue foregroud
DEBUGCL = "\033[34m"
# Warning color is yellow in black bg
WARNCL = "\033[33;40m"
# Warning color is white in red bg
ERRORCL = "\033[37;41m"

#=============================================================================
# Routines
#=============================================================================

def initLogFile(logfile):
	"""
	(Re-)initialize logfile.
	"""
	import libfile
	libfile.saveOldFile(logfile)
	global LOGFD
	if (not LOGFD):
		LOGFD = open(logfile, "a+")
	else:
		LOGFD.close()
		LOGFD = open(logfile, "a+")

	
def prNot(verb, msg, err=EXIT):
	"""
	Print a message to the screen, depending on how 'verb' and the global 
	'VERBOSITY' relate.
	"""
	
	if (VERBOSITY >= verb):
		if (verb >= INFO):
			sys.stdout.write(DEBUGCL)
			print "debug:", msg, RESETCL
		elif (verb == WARNING):
			sys.stdout.write(WARNCL)
			print "WARNING!", msg, RESETCL
		elif (verb == ERR):
			sys.stdout.write(ERRORCL)
			print "ERROR!", msg, RESETCL
			sys.exit(err)
		else:
			print msg
	if (verb < DEBUG and LOGFD):
		tm = time.localtime()
		global LOGLASTDAY
		if (LOGLASTDAY != tm[2]):
			print >> LOGFD, "-"*20, time.asctime(tm), "-"*20
			LOGLASTDAY = tm[2]
		print >> LOGFD, time.strftime("%H:%M:%S", tm), verb, msg
		LOGFD.flush()
		
