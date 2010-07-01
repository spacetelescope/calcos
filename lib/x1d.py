#! /usr/bin/env python

from __future__ import division         # confidence high
import os
import sys
import string
import getopt

import numpy as np
import pyfits

import calcosparam                        # parameter definitions
import cosutil
import getinfo
import extract
import timetag

__version__ = calcosparam.CALCOS_VERSION_NUMBER
__vdate__   = calcosparam.CALCOS_VERSION_DATE

def main (args):
    """This is a driver to perform 1-D extraction for one file.

    The input is a corrtag file name (the complete name, e.g.
    rootname_corrtag_a.fits, not just a rootname).  One or more input
    files may be specified.
    The suffixes _corrtag, _flt and _counts are assumed by the code,
    so the files on disk must have these suffixes.
    """

    if len (args) < 1:
        print "Specify one or more input corrtag file names."
        prtOptions()
        raise RuntimeError

    try:
        (options, pargs) = getopt.getopt (args, "qvuo:",
                                          ["find", "location=", "extrsize="])
    except Exception, error:
        print error
        prtOptions()
        raise RuntimeError

    if len (options) == 0:
        for i in range (len (pargs)):
            if pargs[i][0] == '-':
                prtOptions()
                raise RuntimeError, \
                "Command-line options must precede the input file name(s)."

    outdir = None               # output directory name
    find_target = False
    location = None
    extrsize = None
    for i in range (len (options)):
        if options[i][0] == "-q":
            cosutil.setVerbosity (calcosparam.QUIET)
        elif options[i][0] == "-v":
            cosutil.setVerbosity (calcosparam.VERY_VERBOSE)

        elif options[i][0] == "-o":
            outdir = options[i][1]

        elif options[i][0] == "--find":
            find_target = True

        elif options[i][0] == "--location":
            values = options[i][1].split()
            location = []
            for i in range (len (values)):
                if values[i].lower() == "none":
                    location.append (None)
                else:
                    location.append (float (values[i]))

        elif options[i][0] == "--extrsize":
            values = options[i][1].split()
            extrsize = []
            for i in range (len (values)):
                if values[i].lower() == "none":
                    extrsize.append (None)
                else:
                    extrsize.append (int (values[i]))

    extractSpec (pargs, outdir, location, extrsize, find_target)

def extractSpec (inlist=[], outdir=None,
                 location=None, extrsize=None, find_target=False,
                 verbosity=None):
    """Extract a 1-D spectrum from each set of flt and counts images.

    @param inlist: names of input corrtag files
    @type inlist: list of strings
    @param outdir: name of output directory, or None
    @type outdir: string or None
    @param location: location(s) at which to extract spectrum or spectra
    @type location: float, or a list of floats
    @param extrsize: extraction height(s) for spectra
    @type extrsize: integer, or a list of integers
    @param find_target: if True, search for the target spectrum in the
        Y direction, rather than relying on the wavecal offset (shift2)
    @type find_target: boolean
    @param verbosity: if not None, set verbosity to this level (0, 1, 2)
    @type verbosity: int or None
    """

    if verbosity is not None:
        if verbosity < 0 or verbosity > 2:
            raise RuntimeError, \
                "Verbosity %d is out of range (0, 1, or 2)" % verbosity
        cosutil.setVerbosity (verbosity)

    cal_ver = calcosparam.CALCOS_VERSION

    if outdir:
        outdir = os.path.expandvars (outdir)
        if not os.path.isdir (outdir):
            raise RuntimeError, \
                "The specified output directory doesn't exist:  %s" % outdir
    else:
        outdir = ""

    # Get the names of the input (corrtag), intermediate (flt, counts) and
    # output (x1d) files.
    corrtag_files = []
    flt_files = []
    counts_files = []
    x1d_files = []
    for i in range (len (inlist)):
        input = inlist[i]
        input = os.path.expandvars (input)
        (corrtag, flt, counts, x1d) = makeFileNames (input, outdir)
        if x1d is None:
            print "input = '%s' will be skipped; " \
                  "expected a corrtag file name" % input
            continue
        corrtag_files.append (corrtag)
        flt_files.append (flt)
        counts_files.append (counts)
        x1d_files.append (x1d)

    concatenate_these = getNamesToConcatenate (x1d_files)

    # Now check whether any input file is missing or any intermediate or
    # output file already exists.
    missing = checkMissing (corrtag_files)
    already_exists = checkExists (flt_files, counts_files, x1d_files,
                                  concatenate_these)
    if missing or already_exists:
        raise IOError

    for i in range (len (x1d_files)):
        is_wavecal = makeReqdFiles (cal_ver, corrtag_files[i],
                                 flt_files[i], counts_files[i])
        extract.extract1D (flt_files[i], counts_files[i], x1d_files[i],
                           location=location, extrsize=extrsize,
                           find_target=find_target)
        if is_wavecal:
            extract.recomputeWavelengths (x1d_files[i])

    # If any input files are for FUV, merge the x1d_a.fits and x1d_b.fits
    # files to x1d.fits.
    concatenateSegments (concatenate_these)

    # Update segment-specific keywords in x1d_a.fits, x1d_b.fits, and in
    # flt and counts files.
    updateSomeKeywords (concatenate_these,
                        x1d_files, flt_files, counts_files)

def checkMissing (corrtag_files):
    """Check for missing input files.

    @param corrtag_files: names of input corrtag files
    @type corrtag_files: list

    @return: list of input corrtag files that are not present
    @rtype: list
    """

    missing = []
    for i in range (len (corrtag_files)):
        if not os.access (corrtag_files[i], os.R_OK):
            missing.append (corrtag_files[i])
    if missing:
        if len (missing) == 1:
            cosutil.printError ("The following input file is missing:")
            cosutil.printContinuation (missing[0])
        else:
            cosutil.printError ("The following input files are missing:")
            for filename in missing:
                cosutil.printContinuation (filename)

    return missing

def checkExists (flt_files, counts_files, x1d_files, concatenate_these):
    """Check for output files that already exist.

    @param flt_files: names of flt files
    @type flt_files: list
    @param counts_files: names of counts files
    @type counts_files: list
    @param x1d_files: names of output x1d.fits, x1d_a.fits, or x1d_b.fits files
    @type x1d_files: list
    @param concatenate_these: keys are x1d.fits file names, value for each is
        a list of x1d_a.fits and/or x1d_b.fits file names
    @type concatenate_these: dictionary

    @return: list of output files that already exist
    @rtype: list
    """

    already_exists = []

    for i in range (len (x1d_files)):
        if os.access (flt_files[i], os.R_OK):
            already_exists.append (flt_files[i])
        if os.access (counts_files[i], os.R_OK):
            already_exists.append (counts_files[i])
        if os.access (x1d_files[i], os.R_OK):
            already_exists.append (x1d_files[i])
    keys = concatenate_these.keys()
    keys.sort()
    for x1d in keys:
        if os.access (x1d, os.R_OK):
            already_exists.append (x1d)

    if already_exists:
        if len (already_exists) == 1:
            cosutil.printError ("The following output file already exists:")
            cosutil.printContinuation (already_exists[0])
        else:
            cosutil.printError ("The following output files already exist:")
            for filename in already_exists:
                cosutil.printContinuation (filename)
        cosutil.printError ("Output files will not be overwritten.")

    return already_exists

def makeFileNames (input, outdir=""):
    """Replace suffixes to make the names of all files that we will need.

    @param input: name of the input corrtag file
    @type input: string
    @param outdir: name of the directory for output files, or ""
    @type outdir: string

    @return: names of corrtag, flt, counts, and x1d files; the flt, counts
        and x1d names will be in the output directory
    @rtype: tuple of strings
    """

    # This is the input file name, but in the output directory.
    outdir_input = os.path.join (outdir, os.path.basename (input))

    if input.endswith ("_corrtag.fits") or \
       input.endswith ("_corrtag_a.fits") or \
       input.endswith ("_corrtag_b.fits"):
        corrtag = input
        flt = replaceSuffix (outdir_input, "_corrtag", "_flt")
        counts = replaceSuffix (outdir_input, "_corrtag", "_counts")
        x1d = replaceSuffix (outdir_input, "_corrtag", "_x1d")
    else:
        corrtag, flt, counts, x1d = None, None, None, None

    return (corrtag, flt, counts, x1d)

def makeReqdFiles (cal_ver, corrtag, flt, counts):
    """Create the flt and counts files.

    @param cal_ver: calcos version number and date
    @type cal_ver: string
    @param corrtag: name of corrected events table
    @type corrtag: string
    @param flt: name of effective count rate file
    @type flt: string
    @param counts: name of count rate file
    @type counts: string

    @return: True if the current exposure is a wavecal, based on EXPTYPE
    @rtype: boolean
    """

    fd = pyfits.open (corrtag, mode="readonly")
    phdr = fd[0].header
    phdr.update ("cal_ver", cal_ver)

    detector = phdr.get ("detector")
    headers = [phdr]
    for i in range (3):
        headers.append (fd[1].header)
    exptime = headers[1].get ("exptime")

    is_wavecal = phdr["exptype"].find ("WAVE") >= 0

    events = fd[1].data
    x = events.field ("XFULL")
    y = events.field ("YFULL")
    epsilon = events.field ("EPSILON")
    dq = events.field ("DQ")
    if detector == "FUV":
        npix = (calcosparam.FUV_Y, calcosparam.FUV_EXTENDED_X)
        x_offset = calcosparam.FUV_X_OFFSET
    else:
        npix = (calcosparam.NUV_Y, calcosparam.NUV_EXTENDED_X)
        x_offset = calcosparam.NUV_X_OFFSET

    # Create the data quality array.
    info = getinfo.getGeneralInfo (phdr, headers[1])
    info["corrtag_input"] = True
    switches = getinfo.getSwitchValues (phdr)
    reffiles = getinfo.getRefFileNames (phdr)
    timetag.setActiveArea (events, info, reffiles["brftab"])
    minmax_shift_dict = timetag.getWavecalOffsets (events,
                                info, reffiles["xtractab"])
    dq_array = timetag.doDqicorr (events, corrtag, info, switches, reffiles,
                                  phdr, headers[1], minmax_shift_dict)

    timetag.writeImages (x, y, epsilon, dq,
                         phdr, headers, dq_array, npix, x_offset, exptime,
                         counts, flt)

    fd.close()

    return is_wavecal

def getNamesToConcatenate (x1d_files):
    """Get the names of x1d_a and x1d_b file names to be concatenated.

    If every input file is for NUV, the output dictionary will be empty.
    Each entry in the output dictionary will be a list of either one or two
    file names, the rootname_x1d_a.fits and/or rootname_x1d_b.fits file.
    The key will be the rootname_x1d.fits file that will be the result of
    concatenating the x1d_a and x1d_b files.

    @param x1d_files: names of the x1d files that have just been created
    @type x1d_files: list of strings

    @return: keys are x1d.fits file names, value for each is a list of
        x1d_a.fits and/or x1d_b.fits file names
    @rtype: dictionary
    """

    concatenate_these = {}
    fuv_files = []
    for x1d in x1d_files:
        if x1d.endswith ("_x1d_a.fits") or x1d.endswith ("_x1d_b.fits"):
            fuv_files.append (x1d)

    nfiles = len (fuv_files)
    if nfiles == 0:
        return concatenate_these        # empty dictionary

    fuv_files.sort()

    done = False
    i = 0
    while not done:
        x1d_ab = fuv_files[i]           # root_x1d_a.fits or root_x1d_b.fits
        k = len (x1d_ab)
        x1d = x1d_ab[:k-7] + ".fits"    # root_x1d.fits

        if x1d.endswith ("_x1d_b.fits"):
            concatenate_these[x1d] = [x1d_ab]
        else:
            # this must be an _x1d_a.fits file; check the next one in the list
            if i >= nfiles-1:
                concatenate_these[x1d] = [x1d_ab]
            else:
                next_x1d_ab = fuv_files[i+1]
                k = len (x1d_ab)
                name_to_compare = x1d_ab[:k-7] + "_b.fits"
                if next_x1d_ab == name_to_compare:
                    concatenate_these[x1d] = [x1d_ab, next_x1d_ab]
                    i += 1
                else:
                    concatenate_these[x1d] = [x1d_ab]
        i += 1
        if i >= nfiles:
            done = True

    return concatenate_these

def concatenateSegments (concatenate_these):
    """Concatenate the 1-D spectra for the two FUV segments into one file.

    Each entry in the input dictionary should be a list of either one or two
    file names, the rootname_x1d_a.fits and/or rootname_x1d_b.fits file.
    The key is the rootname_x1d.fits file that will be the result of
    concatenating the x1d_a and x1d_b files (or renaming the file, if there
    is just one in the list).

    @param concatenate_these: keys are x1d.fits file names, value for each is
        a list of x1d_a.fits and/or x1d_b.fits file names
    @type concatenate_these: dictionary
    """

    keys = concatenate_these.keys()
    keys.sort()
    for x1d in keys:
        file_list = concatenate_these[x1d]
        x1d_ab = file_list[0]           # root_x1d_a.fits or root_x1d_b.fits
        if len (file_list) == 1 and x1d_ab != x1d:
            cosutil.renameFile (x1d_ab, x1d)
        else:
            extract.concatenateFUVSegments (file_list, x1d)

def updateSomeKeywords (concatenate_these,
                        x1d_files, flt_files, counts_files):

    # If there are any FUV files, copy keywords from x1d.fits to
    # x1d_a.fits and/or x1d_b.fits.
    keys = concatenate_these.keys()
    keys.sort()
    for x1d in keys:
        file_list = concatenate_these[x1d]
        copyKeywords (x1d, file_list)

def copyKeywords (x1d, file_list):
    """Copy extraction location keywords to other headers.

    @param x1d: name of the x1d.fits file (containing both segments, if FUV)
    @type x1d: string
    @param file_list: names of the files (e.g. x1d_a.fits, x1d_b.fits)
        in which header keywords should be updated
    @type file_list: list of strings
    """

    fd1 = pyfits.open (x1d, mode="readonly")

    if fd1[0].header["detector"] == "FUV":
        keywords = ["sp_loc_a", "sp_loc_b",
                    "sp_off_a", "sp_off_b",
                    "sp_nom_a", "sp_nom_b",
                    "sp_slp_a", "sp_slp_b",
                    "sp_hgt",
                    "b_bkg1_a", "b_bkg1_b",
                    "b_bkg2_a", "b_bkg2_b",
                    "b_hgt1_a", "b_hgt1_b",
                    "b_hgt2_a", "b_hgt2_b"]
    else:
        keywords = ["sp_loc_a", "sp_loc_b", "sp_loc_c",
                    "sp_off_a", "sp_off_b", "sp_off_c",
                    "sp_nom_a", "sp_nom_b", "sp_nom_c",
                    "sp_slp_a", "sp_slp_b", "sp_slp_c",
                    "sp_hgt",
                    "b_bkg1_a", "b_bkg1_b", "b_bkg1_c",
                    "b_bkg2_a", "b_bkg2_b", "b_bkg2_c",
                    "b_hgt1_a", "b_hgt1_b", "b_hgt1_c",
                    "b_hgt2_a", "b_hgt2_b", "b_hgt2_c"]

    for filename in file_list:
        # check that the file still exists (might have been renamed)
        if os.access (filename, os.R_OK):
            fd2 = pyfits.open (filename, mode="update")
            for key in keywords:
                value = fd1[1].header.get (key, -999.)
                fd2[1].header.update (key, value)
            fd2.close()

    fd1.close()

def prtOptions():
    """Print a list of command-line options and arguments."""

    print "The command-line arguments and options are:"
    print "  -q (quiet)"
    print "  -v (very verbose)"
    print "  -u (update keywords in input flt and counts files)"
    print "  -o outdir (output directory name)"
    print "  --find (find Y location of spectrum)"
    print "  --location Y location(s) at which to extract spectra"
    print "  --extrsize height(s) of extraction region(s)"
    print "  one or more corrtag file names"

def replaceSuffix (rawname, suffix, new_suffix):
    """Replace the suffix in a raw file name.

    @param rawname: a file name
    @type rawname: string
    @param suffix: suffix part of rootname to be replaced
    @type suffix: string
    @param new_suffix: the string to replace suffix in rawname
    @type new_suffix: string

    @return: rawname with suffix replaced by new_suffix
    @rtype: string
    """

    lenraw = len (rawname)
    lensuffix = len (suffix)
    i = rawname.rfind (suffix)
    if i >= 0:
        newname = rawname[0:i] + new_suffix + rawname[i+lensuffix:]
    else:
        raise RuntimeError, \
            "File name " + rawname + " was expected to have suffix " + suffix

    return newname

if __name__ == "__main__":

    main (sys.argv[1:])
