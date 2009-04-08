#!/usr/bin/env python
# encoding: utf-8
"""
liblog.py

Some routines for printing and logging messages to screen, disk or whatever.

Created by Tim on 2009-03-30.
Copyright (c) 2009 Tim van Werkhoven. All rights reserved.

$Id$
"""

#=============================================================================
# Defines
#=============================================================================

# Various levels of verbosity (make sure these increment nicely)
VERB_WARN = -1				# Print non-fatal problems with this code
VERB_SILENT = 0				# Always print these messages
VERB_INFO = 1				# Print as info (useful runtime info)
VERB_DEBUG = 2				# Print as debug (useful debug info)
VERB_ALL = 3				# Print lots of debug (useless debug info)

# Init setting for verbosity:
VERBOSITY = VERB_DEBUG

#=============================================================================
# Routines
#=============================================================================

def prNot(verb, msg, todisk=VERB_DEBUG):
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
	if (VERBOSITY >= verb):
		if (verb == VERB_DEBUG):
			sys.stdout.write(debugcl)
			print "---", msg, resetcl
		elif (verb == VERB_WARN):
			sys.stdout.write(warncl)
			print "!!!", msg, resetcl
		else:
			print "***", msg
	#if (todisk >= verb):


