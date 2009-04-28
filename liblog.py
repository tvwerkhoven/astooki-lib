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
# Defines
#=============================================================================

# Various levels of verbosity (make sure these increment nicely)
VERB_WARN = -2				# Print fatal problems with this code, and exit
VERB_WARN = -1				# Print non-fatal problems with this code
VERB_SILENT = 0				# Always print these messages
VERB_INFO = 1				# Print as info (useful runtime info)
VERB_DEBUG = 2				# Print as debug (useful debug info)
VERB_ALL = 3				# Print lots of debug (useless debug info)

# Shorter synonyms of the above, use these instead
ERROR = -2
WARN = -1
SILENT = 0
INFO = 1
DEBUG = 2
ALL = 3

# Exit with this code
EXIT = -1


VERBOSITY = 0
#=============================================================================
# Routines
#=============================================================================

def prNot(verb, msg, todisk=DEBUG, err=EXIT):
	"""
	Print a message to the screen, depending on how 'verb' and the global 
	'VERBOSITY' relate. If 'todisk' is set, log the message to disk if 
	'todisk' >= 'verb'.
	"""
	import sys
	
	resetcl = "\033[0m"
	# Debug is blue foregroud
	debugcl = "\033[34m"
	# Warning color is yellow in black bg
	warncl = "\033[33;40m"
	# Warning color is white in red bg
	errorcl = "\033[37;41m"
	if (VERBOSITY >= verb):
		if (verb >= DEBUG):
			sys.stdout.write(debugcl)
			print "---", msg, resetcl
		elif (verb == WARN):
			sys.stdout.write(warncl)
			print "!!!", msg, resetcl
		elif (verb == ERROR):
			sys.stdout.write(errorcl)
			print "---", msg, resetcl
			sys.exit(err)
		else:
			print "***", msg
	#if (todisk >= verb):


