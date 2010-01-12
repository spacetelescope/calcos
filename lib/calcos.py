#! /usr/bin/env python

from __future__ import division         # confidence high
import sys
import os
import shutil
import time
import getopt
import glob
import copy

import numpy
import pyfits
import accum
import average
import cosutil
import extract
import fpavg
import getinfo
import timetag
import wavecal
from calcosparam import *       # parameter definitions

# These values for Observation.exp_type are used in this file only.
EXP_UNKNOWN     = 0
EXP_SCIENCE     = 1
EXP_WAVECAL     = 2
EXP_CALIBRATION = 3     # tagflash, but exptype not EXTERNAL/SCI
EXP_TARGET_ACQ  = 4
EXP_ACQ_IMAGE   = 5
EXP_ENGINEERING = 6

# calcos returns this if there are no files to calibrate (e.g. rawacq but
# not acq/image).
NO_DATA_TO_CALIBRATE = 5

# If the input is a raw file rather than an association file, this flag
# will be set to True.
raw_input_trailer = False

def main (args):
    """Check arguments and call calcos.

    This driver interprets command-line arguments and calls calcos for
    each association file or raw file specified on the command line.

    The command-line options are:
        -q (quiet)
        -v (very verbose)
        -s (save temporary files)
        -o outdir (output directory name)
        --csum (create csum image)
        --compress parameters (compress csum image)
        --binx X_bin_factor (csum binning in X)
        --biny Y_bin_factor (csum binning in Y)
        --shift filename (file to specify shift values)
        --stim filename (append stim locations to filename)
        --live filename (append livetime factors to filename)
        --burst filename (append burst info to filename)
    Following the command-line options, there should be a list of one
    or more association files or raw files, specified by rootname with
    "_asn" or "_raw".
    """

    if len (args) < 1:
        cosutil.printError (
        "An association file name or observation rootname must be specified.")
        prtOptions()
        sys.exit()

    try:
        (options, pargs) = getopt.getopt (args, "qvso:",
                           ["csum", "compress=", "binx=", "biny=",
                            "shift=", "stim=", "live=", "burst="])
    except Exception, error:
        cosutil.printError (str (error))
        prtOptions()
        sys.exit()

    if len (options) == 0:
        for i in range (len (pargs)):
            if pargs[i][0] == '-':
                cosutil.printError (
                "Command-line options must precede the association file name.")
                prtOptions()
                sys.exit()

    # default values
    cosutil.setVerbosity (VERBOSE)
    # parameters pertaining to the "calcos sum" file
    create_csum_image = False
    binx = 1
    biny = 1
    compress_csum = False
    compression_parameters = "gzip,-0.01"
    # user-supplied text file to specify shift1 and shift2
    shift_file = None
    save_temp_files = False
    stimfile = None
    livetimefile = None
    burstfile = None
    outdir = None

    for i in range (len (options)):
        if options[i][0] == "-q":
            cosutil.setVerbosity (QUIET)
        elif options[i][0] == "-v":
            cosutil.setVerbosity (VERY_VERBOSE)
        elif options[i][0] == "-s":
            save_temp_files = True
        elif options[i][0] == "-o":
            outdir = options[i][1]
        elif options[i][0] == "--csum":
            create_csum_image = True
        elif options[i][0] == "--compress":
            compress_csum = True
            compression_parameters = options[i][1]
        elif options[i][0] == "--binx":
            binx = int (options[i][1])
        elif options[i][0] == "--biny":
            biny = int (options[i][1])
        elif options[i][0] == "--shift":
            shift_file = options[i][1]
        elif options[i][0] == "--stim":
            stimfile = options[i][1]
        elif options[i][0] == "--live":
            livetimefile = options[i][1]
        elif options[i][0] == "--burst":
            burstfile = options[i][1]

    if outdir:
        outdir = os.path.expandvars (outdir)
        if not os.path.isdir (outdir):
            cosutil.printError (
                "The specified output directory doesn't exist:  %s" % outdir)
            sys.exit()

    infiles = uniqueInput (pargs)       # remove duplicate names from list

    status = 0
    for i in range (len (infiles)):
        status = calcos (infiles[i], outdir=outdir, verbosity=None,
                         create_csum_image=create_csum_image,
                         binx=binx, biny=biny,
                         compress_csum=compress_csum,
                         compression_parameters=compression_parameters,
                         shift_file=shift_file,
                         save_temp_files=save_temp_files,
                         stimfile=stimfile, livetimefile=livetimefile,
                         burstfile=burstfile)
    if status != 0:
        sys.exit (status)

def prtOptions():
    """Print a list of command-line options and arguments."""

    cosutil.printMsg ("The command-line options are:")
    cosutil.printMsg ("  -q (quiet)")
    cosutil.printMsg ("  -v (very verbose)")
    cosutil.printMsg ("  -s (save temporary files)")
    cosutil.printMsg ("  -o outdir (output directory name)")
    cosutil.printMsg ("  --csum (create 'calcos sum' image)")
    cosutil.printMsg ("  --compress parameters (compress csum image)")
    cosutil.printMsg ("  --binx X_bin_factor (csum bin factor in X)")
    cosutil.printMsg ("  --biny Y_bin_factor (csum bin factor in Y)")
    cosutil.printMsg ("  --shift filename (file to specify shift values)")
    cosutil.printMsg ("  --stim filename (append stim locations to filename)")
    cosutil.printMsg ("  --live filename (append livetime factors to filename)")
    cosutil.printMsg ("  --burst filename (append burst info to filename)")
    cosutil.printMsg ("")
    cosutil.printMsg ("Following the options, list one or more association")
    cosutil.printMsg ("files (rootname_asn) or raw files (rootname_raw).")

def uniqueInput (infiles):
    """Remove effective duplicates from list of files to process.

    @param infiles: list of input files
    @type infiles: list

    @return: the list of input files but with duplicates removed
    @rtype: list
    """

    if len (infiles) <= 1:
        return infiles

    inlist = copy.copy (infiles)
    inlist.sort()

    newlist = [inlist[0]]
    for i in range (1, len (inlist)):
        n = len (inlist[i])
        if inlist[i].endswith ("_a.fits"):
            n -= 7
        elif inlist[i].endswith ("_b.fits"):
            n -= 7
        elif inlist[i].endswith (".fits"):
            n -= 5
        if inlist[i][:n] != inlist[i-1][:n]:
            newlist.append (inlist[i])

    unique_files = []
    for input in infiles:
        if input in newlist and \
           input not in unique_files:
            unique_files.append (input)

    return unique_files

def checkNumerix():
    """Check whether the environment variable NUMERIX is set to numpy."""

    if os.environ.has_key ("NUMERIX") and os.environ["NUMERIX"] != "numpy":
        cosutil.printWarning ("NUMERIX is set to '%s', should be 'numpy'" % \
                              os.environ["NUMERIX"])

def calcos (asntable, outdir=None, verbosity=None,
            create_csum_image=False,
            binx=None, biny=None,
            compress_csum=False, compression_parameters="gzip,-0.01",
            shift_file=None,
            save_temp_files=False,
            stimfile=None, livetimefile=None, burstfile=None):
    """Calibrate COS data.

    @param asntable: the rootname (with "_asn") of an association file, or
        the rootname (with "_raw") of a raw file (or pair of files if FUV)
    @type asntable: string

    @param outdir: name of output directory, or None
    @type outdir: string

    @param verbosity: if not None, set verbosity to this level (0, 1, 2)
    @type verbosity: int or None

    @param create_csum_image: if True, write an image that reflects the
        counts detected at each pixel (includes deadcorr but not flatcorr),
        for OPUS to add to cumulative image
    @type create_csum_image: boolean

    @param binx: binning factor for the X axis (or None, which means that
        the default binning should be used)
    @type binx: int or None

    @param biny: binning factor for the Y axis (or None, which means that
        the default binning should be used)
    @type biny: int or None

    @param compress_csum: if True, compress the "calcos sum" image
    @type compress_csum: boolean

    @param compression_parameters: two values separated by a comma; the first
        is the compression type (rice, gzip or hcompress), and the second is
        the quantization level
    @type compression_parameters: string

    @param shift_file: if specified, this text file contains values of
        shift1 (and possibly shift2) to override the values found via
        wavecal processing
    @type shift_file: string

    @param save_temp_files: By default, the _x1d_a.fits and _x1d_b.fits files
        (if FUV) will be deleted after concatenating to the _x1d.fits file.
        Specify save_temp_files=True to keep these files.
    @type save_temp_files: boolean

    @param stimfile: if specified, the stim positions will be written to
        (or appended to) a text file with this name
    @type stimfile: string

    @param livetimefile: if specified, the livetime factors will be written
        to (or appended to) a text file with this name
    @type livetimefile: string

    @param burstfile: if specified, burst information will be written to
        (or appended to) a text file with this name
    @type burstfile: string
    """

    t0 = time.time()
    # If asntable is a raw file, open a trailer for it.
    openTrailerForRawInput (asntable, outdir)

    cosutil.printMsg ("CALCOS version " + CALCOS_VERSION)
    cosutil.printMsg ("numpy version " + numpy.__version__)
    cosutil.printMsg ("pyfits version " + pyfits.__version__)
    cosutil.printMsg ("Begin " + cosutil.returnTime(), VERBOSE)

    # Check that NUMERIX is set to numpy (or is not set at all).
    checkNumerix()

    if verbosity is not None:
        cosutil.setVerbosity (verbosity)

    # some of the command-line arguments
    cl_args = {"create_csum_image": create_csum_image,
               "binx": binx,
               "biny": biny,
               "compress_csum": compress_csum,
               "compression_parameters": compression_parameters,
               "shift_file": shift_file,
               "save_temp_files": save_temp_files,
               "stimfile": stimfile,
               "livetimefile": livetimefile,
               "burstfile": burstfile}

    assoc = Association (asntable, outdir, cl_args)
    if len (assoc.obs) == 0:
        return NO_DATA_TO_CALIBRATE
    if not assoc.isAnySwitchSet():
        cosutil.printMsg ("Nothing to do; all calibration switches are OMIT.")
        return 0

    cal = Calibration (assoc)

    cal.allWavecals()

    cal.allScience()

    cal.mergeKeywords()
    cal.combineToProduct()

    assoc.updateMempresent()
    assoc.copySptFile()

    cosutil.printMsg ("End   " + cosutil.returnTime(), VERBOSE)

    t1 = time.time()
    cosutil.printMsg ("elapsed time = %.1f sec. = %.2f min." % \
                                (t1-t0, (t1-t0)/60.), VERY_VERBOSE)
    closeTrailerForRawInput()

    return 0

def openTrailerForRawInput (input, outdir):
    """Open the trailer file for this file."""

    global raw_input_trailer

    if input.endswith ("_asn") or input.endswith ("_asn.fits"):
        return
    if raw_input_trailer:               # already open?
        return

    input = os.path.expandvars (input)
    input = os.path.basename (input)
    rootname = getRootname (input, "_raw")
    if outdir:
        outdir = expandDirectory (outdir)
        trailer = os.path.join (outdir, rootname) + ".tra"
    else:
        trailer = rootname + ".tra"

    cosutil.openTrailer (trailer)
    raw_input_trailer = True

def closeTrailerForRawInput():
    """Close the trailer file for this file."""

    global raw_input_trailer

    cosutil.closeTrailer()
    raw_input_trailer = False

def expandDirectory (dirname):
    """Get the real directory name.

    @param dirname: a directory name
    @type dirname: string

    @return: the real directory name
    @rtype: string
    """

    indir = dirname
    done = False
    count = 0
    MAX_COUNT = 100
    while not done:
        temp = os.path.expandvars (indir)       # $stuff/dir
        count += 1
        if temp == indir:
            done = True
        indir = temp
        if count >= MAX_COUNT:
            break
    if not done:
        cosutil.printWarning ("%d iterations exceeded while expanding " \
        "variables in directory %s" % (MAX_COUNT, dirname))
    indir = os.path.abspath (indir)             # ../dir
    indir = os.path.expanduser (indir)          # ~/dir
    directory_name = os.path.normpath (indir)   # remove redundant strings

    return directory_name

def replaceSuffix (rawname, suffix, new_suffix):
    """Replace the suffix in a raw file name.

    @param rawname: name of a raw input file
    @type rawname: string

    @param suffix: suffix (last part of root name) that is expected to be
        found in the raw file name
    @type suffix: string

    @param new_suffix: string to replace 'suffix' to create an output file name
    @type new_suffix: string

    @return: 'rawname' with 'suffix' replaced by 'new_suffix'
    @rtype: string

    >>> print replaceSuffix ("rootname_rawtag.fits", "_rawtag", "_flt")
    rootname_flt.fits
    >>> print replaceSuffix ("rootname_rawtag_a.fits", "_rawtag", "_flt")
    rootname_flt_a.fits
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

def getRootname (input, suffix):
    """Return the root of a file name.

    If 'suffix' is found in 'input', return the portion of input
    that precedes 'suffix'.  Otherwise, if 'input' ends in ".fits",
    return everything from 'input' that precedes ".fits".

    @param input: name of a raw input file
    @type input: string

    @param suffix: suffix that might be found in 'input'
    @type suffix: string

    @return: 'input' truncated before 'suffix', or truncated before ".fits"
        if 'suffix' is not found
    @rtype: string

    >>> print getRootname ("abc_asn.fits", "_asn")
    abc
    >>> print getRootname ("abc_rawtag_b.fits", "_asn")
    abc_rawtag_b
    >>> print getRootname ("abc_rawtag_b.fits", "_raw")
    abc
    >>> print getRootname ("abc_rawtag_b.fits", "_rawtag")
    abc
    >>> print getRootname ("abc_rawtag_b.fits", "_rawtag_b")
    abc
    >>> print getRootname ("abc", "_asn")
    abc
    """

    # Allow corrtag as input.
    if input.find (suffix) < 0:
        suffix = "_corr"

    pieces = input.split (suffix)
    if len (pieces) > 1:
        root = suffix.join (pieces[:-1])
    elif input.endswith (".fits"):
        extn = input.rfind (".fits")
        root = input[:extn]
    else:
        root = input[:]

    return root

class Association (object):
    """Read and interpret the association table.

    Some of the attributes are:
        asntable           full name of the association file, or None if the
                             name (or rootname) of a raw file was specified
        cl_args            some of the command-line arguments
        asn_info           a dictionary of the contents of the association table
        indir              name of input directory, or an empty string; if a
                             directory was specified, it will be added as a
                             prefix to the memnames in asn_info
        outdir             name of output directory, or an empty string
        combine            a dictionary of lists of file names to be
                             averaged (i.e. individual repeatobs or fp-pos
                             exposures)
        concat             pairs of files of 1-D extracted FUV spectra for
                             segments A and B that need to be concatenated;
                             this is a list of dictionaries with info about
                             files to be concatenated
        merge_kwds         list of pairs of file names for FUV flt or counts
                             files; segment-specific keywords will be copied
                             from one to the other so all will be populated
        product            rootname of the product (rootname portion is lower
                             case; includes outdir)
        product_type       memtype of the product (case unchanged from asn
                             table)
        global_switches    dictionary of the global calibration switches
        rawfiles           a list of all rawtag or rawaccum files
        obs                a list of Observation instances, one for each raw
                             file
        first_science_tuple indexes of first science observations in obs list
                             (the first is time-tag and the second is accum)
        first_science      index of first science observation (either time-tag
                             or accum, whichever is not None)
    """

    def __init__ (self, asntable, outdir, cl_args):

        """Constructor.

        @param asntable: the rootname (with "_asn") of an association file, or
            the rootname (with "_raw") of a raw file (or pair of files if FUV)
        @type asntable: string

        @param outdir: name of output directory, or None
        @type outdir: string

        @param cl_args: some of the command-line arguments, or their defaults
        @type cl_args: dictionary
        """

        self.asntable = None
        self.cl_args = None
        self.indir = ""
        self.outdir = ""
        self.asn_info = {}          # association table info
        self.combine = {}           # files to combine
        self.concat = []            # list of dictionaries of concat info
        self.merge_kwds = []        # list of lists of _a and _b pairs
        self.product = None         # rootname of product
        self.product_type = None    # memtype of product
        self.global_switches = {}   # global calibration switches
        self.rawfiles = []          # list of all raw input files
        self.obs = []               # list of Observations
        self.first_science_tuple = None # (i_timetag, i_accum)
        self.first_science = None   # an integer, either i_timetag or i_accum,
                                    # whichever is not None

        # Copy command-line options to attributes.
        self.cl_args = cl_args

        asntable = os.path.expandvars (asntable)
        self.indir = os.path.dirname (asntable)
        if outdir:
            self.outdir = outdir
        else:
            self.outdir = ""

        # Open the association table and read its contents into asn_info.
        if asntable.endswith ("_asn"):
            self.asntable = asntable + ".fits"
            self.readAsnTable()
        elif asntable.endswith ("_asn.fits"):
            self.asntable = asntable
            self.readAsnTable()
        else:
            # asntable is a raw file name; construct a one-row asn_info.
            self.asntable = None
            self.dummyAsnTable (asntable)

        memname = self.asn_info["memname"]
        memtype = self.asn_info["memtype"]
        mempresent = self.asn_info["mempresent"]

        for i in range (len (memname)):

            if memtype[i].find ("PROD") >= 0:
                continue

            if mempresent[i]:

                basic_info = self.initialInfo (memname[i])
                if basic_info is None:
                    # Missing raw data, or error in association table.
                    continue            # don't process this exposure
                self.rawfiles.extend (basic_info["rawfiles"])
                concat_info = {}                # will be one element of concat
                concat_these = []               # x1d_a and x1d_b names
                concat_info_flash = {}          # another element of concat
                concat_these_flash = []         # flash_a and flash_b names
                # merge_flt and merge_counts (only used for FUV) are pairs of
                # names between which segment-specific keywords will be copied
                merge_flt = []                  # flt_a and flt_b names
                merge_counts = []               # counts_a and counts_b names
                first = True                    # first of a pair for FUV
                for input in basic_info["rawfiles"]:    # one (NUV) or two (FUV)
                    obs = initObservation (input, self.outdir, memtype[i],
                          basic_info["detector"], basic_info["obsmode"], first)
                    self.obs.append (obs)
                    if basic_info["detector"] == "FUV":
                        concat_these.append (obs.filenames["x1d_x"])
                        if obs.info["tagflash"]:
                            concat_these_flash.append (obs.filenames["flash_x"])
                        merge_flt.append (obs.filenames["flt"])
                        merge_counts.append (obs.filenames["counts"])
                    if first:
                        if obs.exp_type == EXP_SCIENCE or \
                           obs.exp_type == EXP_CALIBRATION:
                            concat_info["type"] = "science"
                        elif obs.exp_type == EXP_WAVECAL:
                            concat_info["type"] = "wavecal"
                        else:
                            concat_info["type"] = "unknown"
                        concat_info["output"] = obs.filenames["x1d"]
                        if obs.info["tagflash"]:
                            concat_info_flash["type"] = "tagflash"
                            concat_info_flash["output"] = obs.filenames["flash"]
                    if obs.exp_type == EXP_SCIENCE and self.product is not None:
                        # only for imaging data
                        self.updateCombineFlt (obs.filenames,
                                               obs.info["obstype"])
                        if first:
                            self.updateCombineX1d (obs.filenames,
                                  obs.info["fppos"], obs.info["obstype"])
                    first = False

                if concat_these:
                    concat_info["input"] = concat_these
                    self.concat.append (concat_info)
                if concat_these_flash:
                    concat_info_flash["input"] = concat_these_flash
                    self.concat.append (concat_info_flash)
                if len (merge_flt) == 2:
                    self.merge_kwds.append (merge_flt)
                    self.merge_kwds.append (merge_counts)

        if len (self.obs) == 0:
            return

        cosutil.printMsg ("combine = " + repr (self.combine), VERY_VERBOSE)
        cosutil.printMsg ("concat = " + repr (self.concat), VERY_VERBOSE)

        # Find the first science obs (i.e. non-wavecal, if possible).
        self.first_science_tuple = self.findFirstScience()
        (i, j) = self.first_science_tuple
        if i is None:
            i = j
        self.first_science = i

        self.compareConfig()
        self.compareRefFiles()
        self.compareSwitches()
        self.missingRefFiles()
        self.globalSwitches()
        self.checkOutputExists()
        self.stimfileSanityCheck()

    def readAsnTable (self):
        """Read an association table into memory, and get product info."""

        cosutil.printMsg ("Association file = " + self.asntable, VERBOSE)

        fd = pyfits.open (self.asntable, mode="readonly")
        asn_data = fd[1].data
        nrows = asn_data.shape[0]
        if nrows <= 0:
            fd.close()
            raise RuntimeError, "The association table is empty."

        self.asn_info["memname"] = []
        self.asn_info["memtype"] = []
        self.asn_info["mempresent"] = []

        # Convert the memnames to lower case (unless a full file name was
        # given), and prefix them with the input directory name.
        asn_memname = asn_data.field ("memname")
        asn_memtype = asn_data.field ("memtype")
        asn_memprsnt = asn_data.field ("memprsnt")
        for i in range (nrows):
            if not asn_memname[i].endswith (".fits"):
                asn_memname[i] = asn_memname[i].lower()
            self.asn_info["memname"].append ( \
                        os.path.join (self.indir, asn_memname[i]))
            self.asn_info["memtype"].append (asn_memtype[i])
            self.asn_info["mempresent"].append (asn_memprsnt[i])

        fd.close()

        self.product = None
        for i in range (nrows):
            if asn_memtype[i].find ("PROD") >= 0:
                if self.product is not None:
                    raise RuntimeError, \
                    "The association table may list no more than one product."
                self.product = asn_memname[i].lower()
                self.product_type = asn_memtype[i]

        if self.product is not None:
            self.product = os.path.join (self.outdir, self.product)
            cosutil.printMsg ("product = " + self.product, VERY_VERBOSE)

        # Enable writing to trailer files.
        # cosutil.setWriteToTrailer (True)

    def dummyAsnTable (self, asntable):
        """Construct a recarray corresponding to an association table.

        This function will be called for the case that the user specified
        the name of a raw (or corrtag) file instead of an association table.
        The 'asntable' argument is the name as given by the user; we will
        assign that full name (not just the rootname) to asn_info["memname"].
        There will only be this one row; product will be set to None.  The
        memtype will be set to "none", even though it might actually be a
        wavecal.

        @param asntable: the name of an input raw file (not really an
            association table name)
        @type asntable: string
        """

        cosutil.printMsg ("Input file = " + asntable, VERBOSE)

        # asntable is not an association table name, it's an actual file
        # name.  If a complete file name was specified, and if that file
        # exists, save the full name as memname; otherwise, extract the
        # root name and save that.
        if os.access (asntable, os.R_OK):
            self.asn_info["memname"] = [asntable]
        else:
            rootname = getRootname (asntable, "_raw")
            self.asn_info["memname"] = [rootname]

        self.asn_info["memtype"] = ["none"]
        self.asn_info["mempresent"] = [True]    # yes, it is present

        # Because the input is not an association, there is no product.
        self.product = None
        self.product_type = None

        # Disable writing to the trailer file.
        # cosutil.setWriteToTrailer (False)

    def initialInfo (self, memname):
        """Get preliminary information from an input file.

        This gets the names of the raw files, and from the first of those
        files, reads the primary header and calls a function to get
        DETECTOR, OBSMODE, and EXPTYPE.  In addition, this function checks
        that the suffixes are as expected for the DETECTOR and OBSMODE
        keywords.

        If the input is a complete file name, and the file exists, then the
        dictionary of keywords and values will be returned without using
        wildcards or other checks on suffix.

        @param memname: a value in the MEMNAME (member name) column of an
            association table, converted to lower case; if the user specified
            an explicit file name rather than an association table name,
            memname should be the full file name
        @type memname: string

        @return: dictionary of keywords and values; the value will be None
            if there are no files that match the template, or if the input
            is an ACQ other than ACQ/IMAGE.
        @rtype: dictionary, or None
        """

        # Did the user specify a particular input file?  If so, this is
        # all we need to do.
        if os.access (memname, os.R_OK):
            basic_info = getinfo.initialInfo (memname)
            if memname.endswith ("rawacq.fits") and \
               basic_info["exptype"] != "ACQ/IMAGE":
                cosutil.printWarning ("File %s will be skipped because " \
                        "it is not an ACQ/IMAGE" % memname)
                return None
            rawfiles = [memname]
            if basic_info["detector"] == "FUV":
                if memname.endswith ("_a.fits"):
                    other_segment = memname[:-7] + "_b.fits"
                elif memname.endswith ("_b.fits"):
                    other_segment = memname[:-7] + "_a.fits"
                else:
                    other_segment = None
                if other_segment is not None and \
                   os.access (other_segment, os.R_OK):
                    rawfiles.append (other_segment)
                    rawfiles.sort()
            basic_info["rawfiles"] = rawfiles

            return basic_info

        # First find out whether we've got time-tag or accum, FUV or NUV.
        # Look for both rawaccum and rawimage.
        all_rawfiles = []
        raw = glob.glob (memname + "_rawtag*.fits")
        all_rawfiles.extend (raw)
        raw = glob.glob (memname + "_rawaccum*.fits")
        all_rawfiles.extend (raw)
        raw = glob.glob (memname + "_rawimage*.fits")
        all_rawfiles.extend (raw)

        # The input should not include both science files and acq files,
        # but if it does, make sure the first raw file (see below) is a
        # science file rather than an acq.  If the only file is an acq,
        # however, it must be the first raw file, and that's OK.
        if len (all_rawfiles) > 0:
            all_rawfiles.sort()
        raw = glob.glob (memname + "_rawacq.fits")
        if raw:
            if self.isAcqImage (raw[0]):
                all_rawfiles.extend (raw)
            else:
                cosutil.printWarning (
            "File %s will be skipped because it is not an ACQ/IMAGE" % raw[0])

        if not all_rawfiles:
            cosutil.printWarning (
                "There are no files to calibrate for rootname '%s'" % memname)
            return None

        # Get info from the first raw file with the specified rootname.
        initial_basic_info = getinfo.initialInfo (all_rawfiles[0])
        detector = initial_basic_info["detector"]
        obsmode = initial_basic_info["obsmode"]
        exptype = initial_basic_info["exptype"]
        if exptype[0:3] == "ACQ" and exptype != "ACQ/IMAGE":
            cosutil.printWarning (
                "Rootname `%s' is an %s, which cannot be processed." %
                    (memname, exptype))
            return None

        # Find the raw files that we expect to have.
        if detector == "FUV":
            tail = "_[ab].fits"
        else:
            tail = ".fits"
        if obsmode == "TIME-TAG":
            rawfiles = glob.glob (memname + "_rawtag" + tail)
        elif obsmode == "ACCUM":
            # first look for rawaccum
            rawfiles = glob.glob (memname + "_rawaccum" + tail)
            if len (rawfiles) < 1:
                # rawaccum not found, so look for rawimage
                rawfiles = glob.glob (memname + "_rawimage" + tail)
        else:
            raise RuntimeError, \
                  "unexpected OBSMODE `%s' in `%s'" % obsmode, all_rawfiles[0]
        if len (rawfiles) > 0:
            rawfiles.sort()
        rawfiles.extend (glob.glob (memname + "_rawacq.fits"))

        nfiles = len (rawfiles)
        if nfiles == 0:
            raise RuntimeError, \
                "Keywords and filenames are inconsistent for rootname `%s'" \
                        % memname

        # Read the first raw file with the specified rootname.
        basic_info = getinfo.initialInfo (rawfiles[0])

        if len (rawfiles) < len (all_rawfiles):
            cosutil.printWarning ("There are more raw files than we expected:")
            cosutil.printContinuation ("we expected " + repr (rawfiles))
            cosutil.printContinuation ("but we found " + repr (all_rawfiles))

        basic_info["rawfiles"] = rawfiles

        return basic_info

    def isAcqImage (self, rawacq):
        """Check whether rawacq is an ACQ/IMAGE.

        @param rawacq: name of an acq file
        @type rawacq: string

        @return: True if exptype for rawacq is "ACQ/IMAGE", False otherwise
        @rtype: boolean
        """

        fd = pyfits.open (rawacq, mode="readonly")
        exptype = fd[0].header.get ("exptype", "not found")
        fd.close()

        if exptype == "ACQ/IMAGE":
            return True
        else:
            return False

    def updateCombineFlt (self, filenames, obstype):
        """Add the flt name to the input lists in self.combine.

        @param filenames: dictionary of input and output file names
        @type filenames: dictionary

        @param obstype: observation type, "SPECTROSCOPIC" or "IMAGING"
        @type obstype: string
        """

        if obstype != "IMAGING":
            return

        if not self.combine.has_key ("flt"):
            self.combine["flt"] = []

        flt = filenames["flt"]
        self.combine["flt"].append (flt)

    def updateCombineX1d (self, filenames, fppos, obstype):
        """Add the x1d name and fppos index to 'combine'.

        @param filenames: dictionary of input and output file names
        @type filenames: dictionary

        @param fppos: focal plane position index (1, 2, 3, or 4)
        @type fppos: integer

        @param obstype: observation type, "SPECTROSCOPIC" or "IMAGING"
        @type obstype: string
        """

        if obstype != "SPECTROSCOPIC":
            return

        if not self.combine.has_key ("x1d"):
            self.combine["x1d"] = []
        self.combine["x1d"].append (filenames["x1d"])

        if not self.combine.has_key ("fppos"):
            self.combine["fppos"] = []
        self.combine["fppos"].append (fppos)

    def findFirstScience (self):
        """Find the first science file in the list.

        This function returns the indexes of the first science observations.
        The return value is a tuple of two integers.  The first number is the
        index of the first time-tag science observation, or None.  The second
        number is the index of the first accum science observation, or None.
        If there are no science observations, this returns (i, None), where i
        is the index of the first wavecal (which are assumed to always be
        time-tag).  If there are no wavecals either, this returns either
        (0, None) or (None, 0), depending on whether the first observation
        (of any kind) is time-tag or accum, respectively.

        @return: indexes of the first time-tag and the first accum science
            observations
        @rtype: tuple of two integers
        """

        i_timetag = None
        i_accum = None
        foundit_timetag = False
        foundit_accum = False

        # look for a time-tag science observation
        for i in range (len (self.obs)):
            obs = self.obs[i]
            if obs.exp_type == EXP_SCIENCE and \
               obs.info["obsmode"] == "TIME-TAG":
                foundit_timetag = True
                i_timetag = i
                break
        # look for an accum science observation
        for i in range (len (self.obs)):
            obs = self.obs[i]
            if obs.exp_type == EXP_SCIENCE and \
               obs.info["obsmode"] == "ACCUM":
                foundit_accum = True
                i_accum = i
                break

        if not foundit_timetag:
            # No time-tag science observation; find the first wavecal, if any.
            for i in range (len (self.obs)):
                if self.obs[i].exp_type == EXP_WAVECAL or \
                   self.obs[i].exp_type == EXP_CALIBRATION:
                    foundit_timetag = True
                    i_timetag = i
                    break
        if not foundit_timetag and not foundit_accum:
            # No wavecal observation; take the first observation of any kind.
            i = 0
            if self.obs[i].info["obsmode"] == "TIME-TAG":
                i_timetag = i
            else:
                i_accum = i

        return (i_timetag, i_accum)

    def compareConfig (self):
        """Compare detector, opt_elem, and cenwave.

        All the files in an association must have been taken with the same
        detector and grating (or mirror).  For spectroscopic observations,
        the central wavelength must also be the same.
        """

        if len (self.obs) < 2:
            return

        # Take the reference configuration from the first observation
        # of any kind.
        refinfo = self.obs[0].info
        detector = refinfo["detector"]
        opt_elem = refinfo["opt_elem"]
        cenwave = refinfo["cenwave"]            # 0 for imaging type

        for obs in self.obs:
            if obs.info["detector"] != detector or \
               obs.info["opt_elem"] != opt_elem or \
               obs.info["cenwave"] != cenwave:
                cosutil.printError (obs.filenames["raw"])
                errmess = "All files must be for the same detector"
                if obs.info["obstype"] == "SPECTROSCOPIC":
                    errmess += ", opt_elem and cenwave."
                else:
                    errmess += " and opt_elem."
                raise RuntimeError, errmess

    def compareRefFiles (self):
        """Compare reference file names.

        This function compares the values of the reference file keywords in
        the observation list.  If there is any mismatch, a warning will be
        printed giving the values in the first science observation (can be
        different for time-tag vs accum) and in the current observation.
        """

        # Take the reference file list from the first segment for the first
        # science observation.
        if self.first_science_tuple[0] is None:
            reffiles_timetag = None
        else:
            reffiles_timetag = self.obs[self.first_science_tuple[0]].reffiles
        if self.first_science_tuple[1] is None:
            reffiles_accum = None
        else:
            reffiles_accum = self.obs[self.first_science_tuple[1]].reffiles

        # Now do the comparisons.  'a_file' is the value of a reference file
        # keyword, and 'compare' is the value in the first science observation.
        message_printed = False
        for obs in self.obs:
            obs.openTrailer()
            if obs.info["obsmode"] == "TIME-TAG":
                reffiles = reffiles_timetag
            else:
                reffiles = reffiles_accum
            keys = reffiles.keys()
            keys.sort()
            for key in keys:
                if key.find ("_hdr") >= 0:
                    continue
                compare = reffiles[key].strip()
                a_file = obs.reffiles[key].strip()
                if a_file != compare:
                    compare_hdr = reffiles[key+"_hdr"]
                    a_file_hdr = obs.reffiles[key+"_hdr"]
                    if not message_printed:
                        cosutil.printWarning ( \
                                "Inconsistent reference file names:")
                        message_printed = True
                    if len (compare) == 0:
                        compare = "(blank)"
                    if len (a_file) == 0:
                        a_file = "(blank)"
                    cosutil.printMsg (obs.input + ":  " + key + " = " + \
                            a_file_hdr + " vs. " + compare_hdr)
            obs.closeTrailer()

    def compareSwitches (self):
        """Compare switches.

        This function compares the values of the calibration switch keywords
        in the observation list.  If there is any mismatch, a warning will
        be printed giving the values in the first science observation (can
        be different for time-tag vs accum) and in the current observation.
        """

        # Take the list of switches from the first segment for the first
        # science observation.
        if self.first_science_tuple[0] is None:
            switches_timetag = None
        else:
            switches_timetag = self.obs[self.first_science_tuple[0]].switches
        if self.first_science_tuple[1] is None:
            switches_accum = None
        else:
            switches_accum = self.obs[self.first_science_tuple[1]].switches

        # Do the comparisons.  'sw' is the value of a calibration keyword,
        # and 'compare' is the value in the first science observation.
        message_printed = False
        for obs in self.obs:
            obs.openTrailer()
            if obs.info["obsmode"] == "TIME-TAG":
                switches = switches_timetag
            else:
                switches = switches_accum
            keys = switches.keys()
            keys.sort()
            for key in keys:
                compare = switches[key].strip()
                sw = obs.switches[key].strip()
                if sw != compare:
                    if obs.exp_type == EXP_WAVECAL:
                        if key in ["wavecorr", "doppcorr",
                                   "helcorr", "fluxcorr", "tdscorr"]:
                            continue
                    if not message_printed:
                        cosutil.printWarning (
                                "Inconsistent calibration switches:")
                        message_printed = True
                    if len (compare) == 0:
                        compare = "(blank)"
                    if len (sw) == 0:
                        sw = "(blank)"
                    cosutil.printMsg (obs.input + ":  " + key + " = " + \
                            sw + " vs. " + compare)
            obs.closeTrailer()

    def missingRefFiles (self):
        """Check for missing reference files.

        This function opens each of the required reference files, gets the
        FILETYPE keyword from the primary header, and compares that value
        with the expected value.  It is an error if any of the reference
        files can't be opened, or if the value of FILETYPE doesn't match.

        Note that the minimum reference file version is specified here, for
        each reference file.  This is the min_ver value.
        """

        # Take info from both time-tag and accum data, if we have both.
        (i, j) = self.first_science_tuple
        if i is None:
            i = j
            j = None
        switches = copy.copy (self.obs[i].switches)
        reffiles = copy.copy (self.obs[i].reffiles)
        if j is not None:
            j_switches = copy.copy (self.obs[j].switches)
            j_reffiles = copy.copy (self.obs[j].reffiles)
            for key in switches.keys():
                if switches[key] != "PERFORM" and j_switches[key] == "PERFORM":
                    switches[key] = "PERFORM"
            for key in reffiles.keys():
                if reffiles[key] == NOT_APPLICABLE:
                    reffiles[key] = j_reffiles[key]

        missing = {}            # reference file is not accessible
        wrong_filetype = {}     # wrong FILETYPE
        bad_version = {}        # inconsistent version strings

        # temp is a temporary dictionary with just min_ver and filetype,
        # for readability; these and other values will be copied to ref,
        # which is used for the argument to findRefFile.
        temp = {
            "flatfile": ["2.0", "FLAT FIELD REFERENCE IMAGE"],
            "badttab":  ["2.0", "BAD TIME INTERVALS TABLE"],
            "bpixtab":  ["2.0", "DATA QUALITY INITIALIZATION TABLE"],
            "deadtab":  ["2.0", "DEADTIME REFERENCE TABLE"],
            "brftab":   ["2.0", "BASELINE REFERENCE FRAME TABLE"],
            "phatab":   ["2.0", "PULSE HEIGHT PARAMETERS REFERENCE TABLE"],
            "geofile":  ["2.0", "GEOMETRIC DISTORTION REFERENCE IMAGE"],
            "lamptab":  ["2.0", "TEMPLATE CAL LAMP SPECTRA TABLE"],
            "wcptab":   ["2.0", "WAVECAL PARAMETERS REFERENCE TABLE"],
            "xtractab": ["2.0", "1-D EXTRACTION PARAMETERS TABLE"],
            "disptab":  ["2.0", "DISPERSION RELATION REFERENCE TABLE"],
            "fluxtab":  ["2.0", "PHOTOMETRIC SENSITIVITY REFERENCE TABLE"],
            "imphttab": ["2.0", "IMAGING PHOTOMETRIC TABLE"],
            "tdstab":   ["2.0", "TIME DEPENDENT SENSITIVITY TABLE"],
            "brsttab":  ["2.0", "BURST PARAMETERS TABLE"]
        }
        # The contents of these dictionaries must agree with what
        # cosutil.findRefFile expects.
        ref = {}
        for keyword in temp:
            value = {"keyword":    keyword,
                     "filename":   reffiles[keyword],
                     "calcos_ver": CALCOS_VERSION_NUMBER,
                     "min_ver":    temp[keyword][0],
                     "filetype":   temp[keyword][1]}
            ref[keyword] = value

        if switches["flatcorr"] == "PERFORM":
            cosutil.findRefFile (ref["flatfile"],
                    missing, wrong_filetype, bad_version)
        if switches["brstcorr"] == "PERFORM":
            cosutil.findRefFile (ref["brsttab"],
                    missing, wrong_filetype, bad_version)

        if switches["badtcorr"] == "PERFORM":
            cosutil.findRefFile (ref["badttab"],
                    missing, wrong_filetype, bad_version)

        if switches["dqicorr"] == "PERFORM":
            cosutil.findRefFile (ref["bpixtab"],
                    missing, wrong_filetype, bad_version)

        if switches["deadcorr"] == "PERFORM":
            cosutil.findRefFile (ref["deadtab"],
                    missing, wrong_filetype, bad_version)

        if switches["tempcorr"] == "PERFORM":
            cosutil.findRefFile (ref["brftab"],
                    missing, wrong_filetype, bad_version)

        if switches["phacorr"] == "PERFORM":
            cosutil.findRefFile (ref["phatab"],
                    missing, wrong_filetype, bad_version)

        if switches["geocorr"] == "PERFORM":
            cosutil.findRefFile (ref["geofile"],
                    missing, wrong_filetype, bad_version)

        if switches["wavecorr"] == "PERFORM":
            if self.obs[i].info["obstype"] != "IMAGING":
                cosutil.findRefFile (ref["lamptab"],
                        missing, wrong_filetype, bad_version)
                cosutil.findRefFile (ref["wcptab"],
                        missing, wrong_filetype, bad_version)

        if switches["x1dcorr"] == "PERFORM":
            cosutil.findRefFile (ref["xtractab"],
                    missing, wrong_filetype, bad_version)
            cosutil.findRefFile (ref["disptab"],
                    missing, wrong_filetype, bad_version)

        if switches["fluxcorr"] == "PERFORM":
            cosutil.findRefFile (ref["fluxtab"],
                    missing, wrong_filetype, bad_version)
            if switches["tdscorr"] == "PERFORM":
                cosutil.findRefFile (ref["tdstab"],
                        missing, wrong_filetype, bad_version)

        if switches["photcorr"] == "PERFORM":
            # xxx commented out because we don't have this table yet
            # cosutil.findRefFile (ref["imphttab"],
            #         missing, wrong_filetype, bad_version)
            pass

        if len (missing) > 0:
            msg = "The following reference file"
            if len (missing) > 1:
                msg += "s are missing:"
            else:
                msg += " is missing:"
            cosutil.printError (msg)
            keywords = missing.keys()
            keywords.sort()
            for key in keywords:
                cosutil.printMsg (key + "=" + missing[key])

        if len (wrong_filetype) > 0:
            cosutil.printError ("Wrong FILETYPE; expected the following:")
            keywords = wrong_filetype.keys()
            keywords.sort()
            for key in keywords:
                cosutil.printMsg (key + " = " + wrong_filetype[key][0])
                cosutil.printMsg (
                    "  filetype should be " + wrong_filetype[key][1])

        if len (bad_version) > 0:
            cosutil.printError (
                "Version incompatibility between CALCOS and reference file:")
            keywords = bad_version.keys()
            keywords.sort()
            for key in keywords:
                cosutil.printMsg (key + " = " + bad_version[key][0])
                cosutil.printMsg (bad_version[key][1])

        if len (missing) > 0 or len (wrong_filetype) > 0 or \
           len (bad_version) > 0:
            raise RuntimeError

    def globalSwitches (self):
        """Set global switches.

        The global switches are "any", "science" and "wavecal".
        Their values are either "PERFORM" or "OMIT", though "science" and
        "wavecal" indicate the presence of one or more files of that type
        rather than an actual calibration switch.  "wavecal" refers to
        separate wavecal files, not concurrent science and wavecal data
        (tagflash); the latter would be included in "science".  "any" is
        "PERFORM" if any of the calibration steps other than wavecorr is
        "PERFORM".
        """

        # Take the calibration switch list from the first science observation.
        i = self.first_science
        switches = self.obs[i].switches

        # There are not the only calibration switches, but these are the
        # ones that are independent of others.  For example, it wouldn't
        # matter if fluxcorr were set to perform if x1dcorr were omit.
        self.global_switches["any"] = "OMIT"            # default value
        if self.cl_args["create_csum_image"]:
            self.global_switches["any"] = "PERFORM"
        for key in ["badtcorr", "brstcorr", "deadcorr", "doppcorr",
                    "dqicorr",  "flatcorr", "geocorr",  "helcorr",
                    "phacorr",  "randcorr", "tempcorr", "x1dcorr",
                    "wavecorr"]:
            if switches[key] == "PERFORM":
                self.global_switches["any"] = "PERFORM"
                break

        # This indicates the presence of a separate wavecal file, not
        # tagflash wavecals.
        self.global_switches["wavecal"] = "OMIT"        # default value
        for obs in self.obs:
            if obs.exp_type == EXP_WAVECAL:
                self.global_switches["wavecal"] = "PERFORM"
                break

        # This indicates the presence of a science or calibration observation.
        self.global_switches["science"] = "OMIT"       # default value
        for obs in self.obs:
            if obs.exp_type == EXP_SCIENCE or \
               obs.exp_type == EXP_CALIBRATION or \
               obs.exp_type == EXP_ACQ_IMAGE:
                self.global_switches["science"] = "PERFORM"
                break

        if self.product is not None:
            if self.combine.has_key ("x1d"):
                ncombine = len (self.combine["x1d"])
            elif self.combine.has_key ("flt"):
                ncombine = len (self.combine["flt"])
            else:
                ncombine_a = 0
                ncombine_b = 0
                if self.combine.has_key ("flt_a"):
                    ncombine_a = len (self.combine["flt_a"])
                if self.combine.has_key ("flt_b"):
                    ncombine_b = len (self.combine["flt_b"])
                ncombine = max (ncombine_a, ncombine_b)

    def isAnySwitchSet (self):
        """Return 1 if any calibration switch is PERFORM, 0 otherwise.

        The test is made using the global switches, because the test on
        individual switches is done in the method that sets the global
        switches.
        """

        if self.global_switches["any"] == "PERFORM" or \
           self.global_switches["wavecal"] == "PERFORM":
            return 1
        else:
            return 0

    def checkOutputExists (self):
        """Check whether output files already exist.

        This routine checks for the existence of any output file for each of
        the input files.  Wavecal files and science files are treated
        differently, because wavecal files (conventional, not tagflash) can
        be common to more than one association.  If one or more calibrated
        files are found, a message will be printed giving their names.
        Existing calibrated wavecal files will be deleted.  If calibrated
        science files are found, however, an exception will be raised.
        """

        detector = self.obs[0].info["detector"]

        already_exists = []
        wavecal_exists = []
        for obs in self.obs:
            if obs.exp_type == EXP_WAVECAL:
                self.checkExists (obs.filenames["corrtag"], wavecal_exists)
                self.checkExists (obs.filenames["flt"], wavecal_exists)
                self.checkExists (obs.filenames["counts"], wavecal_exists)
                self.checkExists (obs.filenames["x1d_x"], wavecal_exists)
                if obs.filenames["x1d"] != obs.filenames["x1d_x"]:
                    self.checkExists (obs.filenames["x1d"], wavecal_exists)
                if self.cl_args["create_csum_image"]:
                    self.checkExists (obs.filenames["csum"], wavecal_exists)
            else:
                self.checkExists (obs.filenames["corrtag"], already_exists)
                self.checkExists (obs.filenames["flt"], already_exists)
                self.checkExists (obs.filenames["counts"], already_exists)
                if obs.info["obsmode"] == "TIME-TAG":
                    self.checkExists (obs.filenames["flash_x"], already_exists)
                    self.checkExists (obs.filenames["flash"], already_exists)
                if obs.switches["x1dcorr"] == "PERFORM":
                    self.checkExists (obs.filenames["x1d_x"], already_exists)
                    if obs.filenames["x1d"] != obs.filenames["x1d_x"]:
                        self.checkExists (obs.filenames["x1d"], already_exists)

        if self.product is not None:
            self.checkExists (self.product + "_fltsum.fits", already_exists)
            self.checkExists (self.product + "_x1dsum.fits", already_exists)
            self.checkExists (self.product + "_x1dsum1.fits", already_exists)
            self.checkExists (self.product + "_x1dsum2.fits", already_exists)
            self.checkExists (self.product + "_x1dsum3.fits", already_exists)
            self.checkExists (self.product + "_x1dsum4.fits", already_exists)

        # Remove duplicates.
        for i in range (len(already_exists)-1, 0, -1):
            fname = already_exists[i]
            if fname in already_exists[0:i]:
                del (already_exists[i])
        for i in range (len(wavecal_exists)-1, 0, -1):
            fname = wavecal_exists[i]
            if fname in wavecal_exists[0:i]:
                del (wavecal_exists[i])

        if already_exists:
            if len (already_exists) == 1:
                errmess = "output file already exists"
            else:
                errmess = "output files already exist"
            cosutil.printError (errmess + ":")
            for fname in already_exists:
                cosutil.printError ("  %s" % fname)
            raise RuntimeError, errmess

        if wavecal_exists:
            if len (wavecal_exists) == 1:
                msg = "Calibrated wavecal file already exists"
            else:
                msg = "Calibrated wavecal files already exist"
            msg += ", will be deleted:"
            cosutil.printWarning (msg)
            for fname in wavecal_exists:
                os.remove (fname)
                cosutil.printWarning ("  %s deleted" % fname)

    def checkExists (self, fname, already_exists):
        """If fname exists, append the name to already_exists.

        @param fname: the name of the file
        @type fname: string

        @param already_exists: a list of names of files that currently
            exist; may be modified in-place by appending 'fname'
        @type already_exists: list
        """

        if os.access (fname, os.R_OK):
            already_exists.append (fname)

    def stimfileSanityCheck (self):
        """Ignore stimfile if detector is not FUV.

        Only the FUV detector has stims.  If a file was specified for
        saving measured stim locations (--stim stimfile), the name will
        be reset to None if the detector was not FUV.
        """

        i = self.first_science
        if self.cl_args["stimfile"] is not None and \
           self.obs[i].info["detector"] != "FUV":
            self.cl_args["stimfile"] = None
            cosutil.printWarning (
                "stimfile reset to None because detector is NUV.")

    def updateMempresent (self):
        """Update the ASN_PROD keyword and MEMPRSNT column."""

        if self.asntable is None or self.product is None:
            return
        cosutil.printMsg ("updateMempresent", VERY_VERBOSE)

        # Modify the association table in-place.
        fd = pyfits.open (self.asntable, mode="update")

        # Set ASN_PROD to true to indicate that a product has been created.
        fd[0].header.update ("asn_prod", True)

        asn = fd[1].data
        nrows = asn.shape[0]
        memtype = asn.field ("MEMTYPE")
        mempresent = asn.field ("MEMPRSNT")

        for i in range (nrows):
            if memtype[i].find ("PROD") >= 0:
                mempresent[i] = True
                break

        fd.close()

    def copySptFile (self):
        """Copy an spt file to the association product name."""

        if self.asntable is None or self.product is None:
            return
        cosutil.printMsg ("copySptFile", VERY_VERBOSE)

        # Find the first science observation that has a support file.
        sptfile = None
        fallback = None         # use this if no suitable spt file available
        for i in range (len (self.obs)):
            obs = self.obs[i]
            if not os.access (obs.filenames["spt"], os.R_OK):
                continue
            if fallback is None:
                fallback = obs.filenames["spt"]
            if obs.exp_type == EXP_SCIENCE:
                sptfile = obs.filenames["spt"]
                break
            if obs.exp_type == EXP_CALIBRATION:
                # use this in preference to fallback
                sptfile = obs.filenames["spt"]

        if sptfile is None:
            if fallback is None:
                cosutil.printWarning (
                "spt file not found, so not copied to product", VERBOSE)
                return
            else:
                sptfile = fallback

        # Change the suffix to "jnk" so that if the user gets this file it
        # will be clear that it should be ignored.
        product_spt_file = self.product + "_jnk.fits"
        cosutil.printMsg ("copy %s to %s" % (sptfile, product_spt_file),
                          VERY_VERBOSE)

        # Copy the spt file to the "product spt" file.
        shutil.copy (sptfile, product_spt_file)

        # Update keywords in the "product spt" file.
        fd = pyfits.open (product_spt_file, mode="update")

        phdr = fd[0].header
        product = os.path.basename (self.product)

        cosutil.updateFilename (phdr, product_spt_file)
        phdr.update ("rootname", product)
        phdr.update ("obset_id", product[4:6])
        phdr.update ("observtn", product[-3:].upper())
        phdr.update ("asn_mtyp", self.product_type)     # do we need this?
        phdr.add_comment (
        "Please ignore this file, which is a copy of an input spt file.")
        phdr.add_comment (
        "This file is used by the archive to obtain certain keywords.")

        for i in range (1, len (fd)):
            fd[i].header.update ("rootname", product)

        fd.close()

def initObservation (input, outdir, memtype, detector, obsmode, first=False):
    """Construct an Observation object for the current mode.

    @param input: the name of an input raw file
    @type input: string
    @param outdir: either an empty string or the name of the output directory
    @type outdir: string
    @param memtype: from association table; used to distinguish between
        wavecal and science observation
    @type memtype: string
    @param detector: FUV or NUV
    @type detector: string
    @param obsmode: TIME-TAG or ACCUM
    @type obsmode: string
    @param first: True if the current file is the first for a given rootname
        (this is for writing the calcos version string to the trailer, so
        that it won't be written for both FUV segments A and B)
    @type first: boolean

    @return: an Observation object
    @rtype: instance
    """

    if detector == "FUV":
        if obsmode == "TIME-TAG":
            obs = FUVTimetagObs (input, outdir, memtype, first)
        else:
            obs = FUVAccumObs (input, outdir, memtype, first)
    else:
        if obsmode == "TIME-TAG":
            obs = NUVTimetagObs (input, outdir, memtype, first)
        else:
            obs = NUVAccumObs (input, outdir, memtype, first)

    return obs

class Observation (object):
    """Get information about an observation from its headers.

    This base class is not directly used; one of its subclasses will
    be invoked, depending on DETECTOR and OBSMODE.
    """

    def __init__ (self, input, outdir, memtype, suffix, first):
        """Invoked by a subclass.

        @param input: the name of an input raw file
        @type input: string
        @param outdir: an empty string or the name of the output directory
        @type outdir: string
        @param memtype: from association table; used to distinguish between
            wavecal and science observation
        @type memtype: string
        @param suffix: suffix to the rootname, but just "_rawtag" or
            "_rawaccum" (i.e. excluding "_a" or "_b" if the data were taken
            with the FUV detector); this can be reset to "_corrtag" or
            "_rawimage" or "_rawacq"
        @type suffix: string
        @param first: True if the current file is the first of two for FUV
        @type first: boolean
        """

        self.input = input              # name of a raw input file
        self.exp_type = EXP_SCIENCE     # science, wavecal, target acq
        self.filenames = {}             # input and output file names
        self.info = {}                  # detector, opt_elem, etc.
        self.switches = {}              # calibration switch values
        self.reffiles = {}              # reference file names

        indir = os.path.dirname (input)
        input_directory = expandDirectory (indir)
        if outdir:
            output_directory = expandDirectory (outdir)
        else:
            output_directory = os.path.realpath (os.curdir)

        self.getHeaderInfo()
        if (input_directory == output_directory) and self.info["corrtag_input"]:
            raise RuntimeError, "For corrtag input," \
                    " the input and output directories must not be the same."

        if self.info["corrtag_input"]:
            suffix = "_corrtag"
        else:
            # For ACCUM data, allow suffix to be "_rawaccum", "_rawimage" or
            # "_rawacq".
            if input.find (suffix) < 0:
                suffix = "_rawimage"
            if input.find (suffix) < 0:
                suffix = "_rawacq"
        if input.find (suffix) < 0:
            raise RuntimeError, "can't find suffix %s in %s" % \
                    (suffix, input)

        self.filenames = self.makeFileNames (suffix, outdir)
        # This value of info["root"] is based on the filename on disk, which
        # could differ from the value of the rootname keyword.
        self.info["root"] = self.filenames["root"]
        self.openTrailer (first)    # open the trailer file for this input file
        self.sanityCheck()

        # Determine what type of observation this is.

        # Assign an initial value for self.exp_type based on keyword exptype.
        if self.info["exptype"] == "WAVECAL":
            self.exp_type = EXP_WAVECAL
        elif self.info["exptype"] == "EXTERNAL/SCI" or \
             self.info["exptype"] == "EXTERNAL/CAL" or \
             self.info["exptype"] == "DARK" or \
             self.info["exptype"] == "FLAT":
            self.exp_type = EXP_SCIENCE
        elif self.info["exptype"] == "ACQ/IMAGE":
            self.exp_type = EXP_ACQ_IMAGE
        elif self.info["exptype"] == "ACQ/SEARCH" or \
             self.info["exptype"] == "ACQ/PEAKD" or \
             self.info["exptype"] == "ACQ/PEAKXD":
            self.exp_type = EXP_TARGET_ACQ
        elif self.info["exptype"] == "ENG DIAG" or \
             self.info["exptype"] == "MEMORY DUMP" or \
             self.info["exptype"] == "PHA":
            cosutil.printWarning ("EXPTYPE = `%s' in %s;" \
                    % (self.info["exptype"], input))
            cosutil.printContinuation ("can't calibrate this exposure type.")
            self.exp_type = EXP_ENGINEERING
        else:
            cosutil.printWarning ("EXPTYPE = `%s' in %s;" \
                    % (self.info["exptype"], input))
            cosutil.printContinuation ("don't recognize this exposure type.")
            self.exp_type = EXP_UNKNOWN

        if memtype != "none":           # is there an association table?
            conflict = False
            memtype_wavecal = memtype.endswith ("WAVE")
            exptype_wavecal = self.info["exptype"] == "WAVECAL"
            if memtype_wavecal and not exptype_wavecal:
                conflict = True
            if exptype_wavecal and not memtype_wavecal:
                conflict = True
            if conflict:
                raise RuntimeError, "MEMTYPE = %s but EXPTYPE = %s for %s" % \
                        (memtype, self.info["exptype"], self.input)

        if self.info["obstype"] == "SPECTROSCOPIC":
            if self.info["tagflash"]:
                if self.info["exptype"] == "EXTERNAL/SCI":
                    pass                        # no change needed
                elif self.info["exptype"] == "WAVECAL":
                    cosutil.printWarning ("EXPTYPE = WAVECAL but TAGFLASH " \
                        "!= NONE for %s;" % self.input)
                    cosutil.printContinuation (
                        "EXPTYPE will be changed to EXTERNAL/CAL.")
                    self.info["exptype"] = "EXTERNAL/CAL"
                    self.exp_type = EXP_CALIBRATION
                    # or should we use:  self.exp_type = EXP_SCIENCE
                else:
                    cosutil.printWarning ("EXPTYPE = %s and TAGFLASH = %s " \
                        "for %s;" % (self.info["exptype"],
                                    self.info["tagflash"], self.input))
                    cosutil.printContinuation (
                        "EXPTYPE will be changed to EXTERNAL/CAL.")
                    self.info["exptype"] = "EXTERNAL/CAL"
                    self.exp_type = EXP_CALIBRATION

            else:
                # just listing cases
                if self.info["exptype"] == "EXTERNAL/SCI":
                    pass                        # no change needed
                elif self.info["exptype"] == "WAVECAL":
                    pass                        # no change needed

        else:                   # obstype = IMAGING

            if self.info["tagflash"]:
                # semi-supported as of version 2.8.5
                if self.info["exptype"] == "EXTERNAL/SCI":
                    pass
                else:
                    # don't need to change exptype, just exp_type
                    self.exp_type = EXP_CALIBRATION
            elif self.info["exptype"] == "WAVECAL":
                cosutil.printWarning ("EXPTYPE = %s and OBSTYPE = %s " \
                        "for %s;" % (self.info["exptype"],
                                    self.info["obstype"], self.input))
                cosutil.printContinuation (
                        "EXPTYPE will be changed to EXTERNAL/CAL.")
                self.info["exptype"] = "EXTERNAL/CAL"
                self.exp_type = EXP_CALIBRATION
            else:
                pass                            # no change needed

        if self.exp_type == EXP_WAVECAL and self.info["aperture"] != "WCA":
            cosutil.printWarning (
            "APERTURE = %s for a wavecal; this could be a serious error" \
                                % self.info["aperture"])

        self.checkSwitches()
        self.closeTrailer()

    def openTrailer (self, first=False):
        """Open the trailer file for this file."""

        global raw_input_trailer

        if raw_input_trailer:           # handled separately
            return

        cosutil.openTrailer (self.filenames["trl"])
        if first:
            cosutil.writeVersionToTrailer()

    def closeTrailer (self):
        """Close the trailer file for this file."""

        global raw_input_trailer

        if raw_input_trailer:           # handled separately
            return

        cosutil.closeTrailer()

    def getHeaderInfo (self):
        """Read keyword values.

        This routine gets general info from both the primary and EVENTS or SCI
        extension headers, and it gets calibration switches and reference
        file names from the primary header.

        This function also adds keys "corrtag_input" and "cal_ver" to the
        info dictionary.
        """

        fd = pyfits.open (self.input, mode="readonly")
        phdr = fd[0].header
        try:
            hdr = fd["EVENTS"].header
        except:
            hdr = fd[("SCI",1)].header

        fd.close()

        # Each of these is a dictionary with (lower case) header keywords
        # as the keys.
        self.info = getinfo.getGeneralInfo (phdr, hdr)
        self.switches = getinfo.getSwitchValues (phdr)
        self.reffiles = getinfo.getRefFileNames (phdr)

        # check for ref file name "N/A"
        getinfo.resetSwitches (self.switches, self.reffiles)

        # Is the input a corrtag file?
        self.info["corrtag_input"] = cosutil.isCorrtag (self.input)

        self.info["cal_ver"] = CALCOS_VERSION

    def sanityCheck (self):
        """Check some keywords to make sure they're reasonable.

        For thermal vac data, this also updates opt_elem, cenwave and fpoffset
        in the info dictionary, if necessary.  Other keywords that may be
        reset are obstype and dispaxis.
        """

        info = self.info
        warn = 0                # initial values
        bad = 0

        if self.info["coscoord"] != USER_COORDINATES:
            bad = 1
            cosutil.printError ("Wrong coordinates for this version of calcos")
            cosutil.printContinuation ("for %s" % self.input)
            raise RuntimeError

        # Replace RelMvReq in opt_elem or aperture keywords, etc.
        self.fixRelMvReq (self.filenames["spt"], info)

        # check SEGMENT
        if info["detector"] == "FUV" and \
                (info["segment"] != "FUVA" and info["segment"] != "FUVB"):
            bad = 1
            cosutil.printError ("SEGMENT = `%s' is invalid" % info["segment"])

        # check EXPTIME, EXPSTART, EXPEND
        if info["exptime"] < 0.:
            bad = 1
            cosutil.printError ("EXPTIME = %g is invalid" % info["exptime"])
        # add 0.5 for roundoff error
        if (info["expend"] - info["expstart"]) * SEC_PER_DAY + 0.5 < \
                info["exptime"]:
            warn = 1
            cosutil.printWarning ("(EXPEND - EXPSTART) is less than EXPTIME")

        # check OBSTYPE
        if info["obstype"] == "ACQUISITION":
            warn = 1
            cosutil.printWarning (
            "OBSTYPE = ACQUISITION, will be reset to IMAGING")
            info["obstype"] = "IMAGING"
        if info["obstype"] == "SPECTROSCOPIC" and \
               (info["opt_elem"][0:6] == "MIRROR" or
                info["opt_elem"][0:3] == "TA1"):
            bad = 1
            cosutil.printError (
            "OBSTYPE = SPECTROSCOPIC and OPT_ELEM = %s is invalid"
                 % info["opt_elem"])
        if info["obstype"] == "IMAGING":
            if info["dispaxis"] != 0:
                # not a fatal error
                warn = 1
                cosutil.printWarning (
                "DISPAXIS = %d, will be reset to 0 for imaging data" \
                             % info["dispaxis"])
                info["dispaxis"] = 0
        elif info["obstype"] == "SPECTROSCOPIC":
            if info["dispaxis"] == 2:
                warn = 1
                cosutil.printWarning ("DISPAXIS = 2")
        else:
            bad = 1
            cosutil.printError (
            "OBSTYPE = `%s'; should be IMAGING or SPECTROSCOPIC" \
                         % info["obstype"])

        # check OBSMODE
        if info["obsmode"] != "TIME-TAG" and info["obsmode"] != "ACCUM":
            bad = 1
            cosutil.printError ("OBSMODE = `%s'; should be TIME-TAG or ACCUM" \
                         % info["obsmode"])

        # check OPT_ELEM
        if info["obstype"] == "SPECTROSCOPIC":
            opt_elem = info["opt_elem"]
            if info["detector"] == "FUV" and (opt_elem != "G130M" and \
                    opt_elem != "G160M" and opt_elem != "G140L" and \
                    opt_elem != "NCM1"):
                bad = 1
                cosutil.printError ("OPT_ELEM = `%s' is invalid for FUV" \
                         % info["opt_elem"])
            elif info["detector"] == "NUV" and \
                   (opt_elem != "G185M" and opt_elem != "G225M" and \
                    opt_elem != "G285M" and opt_elem != "G230L"):
                bad = 1
                cosutil.printError ("OPT_ELEM = `%s' is invalid for NUV" \
                         % info["opt_elem"])

        # check APERTURE
        if info["aperture"] != "PSA" and info["aperture"] != "BOA" and \
           info["aperture"] != "WCA" and info["aperture"] != "FCA":
            bad = 1
            cosutil.printError ("APERTURE = `%s' is not valid" \
                    % info["aperture"])

        if warn or bad:
            cosutil.printContinuation ("for %s" % self.input)
        if bad:
            raise RuntimeError

    def fixRelMvReq (self, sptfile, info):
        """Replace RelMvReq in keywords with values based on OSM position.

        For thermal vac data (only), this function determines the values of
        OPT_ELEM, CENWAVE and FPOFFSET based on the OSM1 or OSM2 positions
        as given by LOM1STP or LOM2STP respectively, in the support file
        header.  If OPT_ELEM is "RelMvReq", then OPT_ELEM, CENWAVE and
        FPOFFSET will be silently replaced by the correct values.  If CENWAVE
        is unreasonably small (< 1000), it will be replaced.  Otherwise,
        these three keywords will be compared with the values determined from
        the OSM positions, and discrepancies will be noted and corrected.

        @param sptfile: name of support file
        @type sptfile: string

        @param info: dictionary of keywords and values; values may be
            updated in-place by this function
        @type info: dictionary
        """

        if info["targname"] != "Thermal_Vac":
            return

        # dictionaries with OSM step position as key and tuple of
        # optical element, central wavelength, and FP offset as value;
        # also defines date ranges for NUV and OSM range for TA1Image.
        import osmstep

        try:
            fd = pyfits.open (sptfile, mode="readonly")
            rootname = fd[0].header.get ("rootname", default=NOT_APPLICABLE)
            lom1stp = int (fd[2].header.get ("lom1stp", -1))
            lom2stp = int (fd[2].header.get ("lom2stp", -1))
            fd.close()
        except IOError:
            cosutil.printWarning ("spt file %s not found" % sptfile)
            cosutil.printContinuation ("can't check OPT_ELEM, CENWAVE, FPOFFSET")
            return

        if info["detector"] == "FUV":
            osm_dict = osmstep.fuv_osm1_dict
            if osm_dict.has_key (lom1stp):
                (opt_elem_osm, cenwave_osm, fpoffset_osm) = osm_dict[lom1stp]
            else:
                cosutil.printWarning ("%s has invalid LOM1STP %d" % \
                        (sptfile, lom1stp))
                return
        else:
            # extract the day number from the rootname
            day_num = rootname[12:15]
            if len (day_num) < 1:
                cosutil.printWarning ( \
            "%s has invalid ROOTNAME %s for TV data" % (sptfile, rootname))
                return
            day_num = int (day_num)
            if day_num < osmstep.nuv_tv_dayrange[0]:
                osm_dict = osmstep.nuv_osm2_dict_early
            elif day_num >= osmstep.nuv_tv_dayrange[0] and \
                 day_num <= osmstep.nuv_tv_dayrange[1]:
                osm_dict = osmstep.nuv_osm2_dict_middle
            else:
                osm_dict = osmstep.nuv_osm2_dict_late

            if lom2stp >= osmstep.ta1image_range[0] and \
               lom2stp <= osmstep.ta1image_range[1]:
                opt_elem_osm = "TA1Image"
                cenwave_osm = 0
                fpoffset_osm = 0
            elif lom2stp >= osmstep.ta1bright_range[0] and \
                 lom2stp <= osmstep.ta1bright_range[1]:
                opt_elem_osm = "TA1Brght"
                cenwave_osm = 0
                fpoffset_osm = 0
            else:
                if osm_dict.has_key (lom2stp):
                    (opt_elem_osm, cenwave_osm, fpoffset_osm) = \
                        osm_dict[lom2stp]
                else:
                    cosutil.printWarning ("%s has invalid LOM2STP %d" % \
                            (sptfile, lom2stp))
                    return

        self.compareKeywords_TV (info, opt_elem_osm, cenwave_osm, fpoffset_osm)

    def compareKeywords_TV (self,
            info, opt_elem_osm, cenwave_osm, fpoffset_osm):
        """Update keyword values in info dictionary.

        @param info: dictionary of keywords and values, modified in-place
        @type info: dictionary

        @param opt_elem_osm: value of OPM_ELEM as determined from OSM position
        @type opt_elem_osm: string

        @param cenwave_osm: value of CENWAVE as determined from OSM position
        @type cenwave_osm: integer

        @param fpoffset_osm: value of FPOFFSET as determined from OSM position
        @type fpoffset_osm: integer
        """

        if info["opt_elem"] == "RelMvReq":
            info["opt_elem"] = opt_elem_osm
            info["cenwave"]  = cenwave_osm
            info["fpoffset"] = fpoffset_osm

        if info["cenwave"] < 1000:
            info["cenwave"] = cenwave_osm

        if info["opt_elem"] != opt_elem_osm:
            cosutil.printWarning (
    "OPT_ELEM = %s; will be replaced with %s, based on OSM position" % \
                    (info["opt_elem"], opt_elem_osm))
            info["opt_elem"] = opt_elem_osm

        if info["cenwave"] != cenwave_osm:
            cosutil.printWarning (
    "CENWAVE = %d; will be replaced with %d, based on OSM position" % \
                    (info["cenwave"], cenwave_osm))
            info["cenwave"] = cenwave_osm

        if info["fpoffset"] != fpoffset_osm:
            cosutil.printWarning (
    "FPOFFSET = %d; will be replaced with %d, based on OSM position" % \
                    (info["fpoffset"], fpoffset_osm))
            info["fpoffset"] = fpoffset_osm

    def makeFileNames (self, suffix, outdir):
        """Create names of input and output files from input raw file names.

        @param suffix: an obsmode-specific string, either "_rawtag" or
            "_rawaccum" (or "_rawimage"); note that 'suffix' excludes
            "_a" or "_b", in the case that we have FUV data
        @type suffix: string

        @param outdir: the name of the output directory (or an empty string)
        @type outdir: string

        @return: dictionary of the input and output names
        @rtype: dictionary

        These are the keys for the dictionary of file names:

          root     rootname (not including suffix or directory); note that this
                     is from the file name, not the header keyword
          trl      name (including output directory) of the trailer file
          raw      name of input (raw) file (including directory)
          spt      name of input support file
          pha      input pulse-height histogram (for FUV accum)
          corrtag  output corrected event list (for time-tag)
          flt      output effective count rate image, from accum or
                     corrected time-tag
          counts   output count rate image
          x1d_x    output 1-D extracted spectrum for one segment
                     (or all 3 NUV stripes)
          x1d      output 1-D extracted spectrum (the file that includes
                     all segments or stripes)
          flash_x  output 1-D extracted tagflash wavecal spectrum for one
                     segment (or for all 3 NUV stripes)
          flash    output 1-D extracted tagflash wavecal spectrum (the file
                     that includes all segments or stripes)
          csum     output image for OPUS to add to cumulative image
        """

        input = os.path.basename (self.input)
        # This is the input file name, but in the output directory.
        output = os.path.join (outdir, input)

        rootname = getRootname (input, "_raw")

        trailer = os.path.join (outdir, rootname) + ".tra"

        x1d_x = replaceSuffix (output, suffix, "_x1d")
        flash_x = replaceSuffix (output, suffix, "_lampflash")

        # Remove the "_a" or "_b" from the x1d_x and flash_x suffixes.
        if x1d_x.endswith ("_a.fits"):
            find_this = "_a.fits"
        elif x1d_x.endswith ("_b.fits"):
            find_this = "_b.fits"
        else:
            find_this = None
            x1d = x1d_x[:]
            flash = flash_x[:]
        if find_this is not None:
            i = x1d_x.rfind (find_this)
            x1d = x1d_x[0:i] + ".fits"
            i = flash_x.rfind (find_this)
            flash = flash_x[0:i] + ".fits"

        filenames = {}
        filenames["root"]    = rootname
        filenames["trl"]     = trailer
        filenames["raw"]     = self.input
        filenames["pha"]     = replaceSuffix (self.input, suffix, "_pha")
        filenames["corrtag"] = replaceSuffix (output, suffix, "_corrtag")
        filenames["flt"]     = replaceSuffix (output, suffix, "_flt")
        filenames["counts"]  = replaceSuffix (output, suffix, "_counts")
        filenames["x1d_x"]   = x1d_x
        filenames["x1d"]     = x1d
        filenames["flash_x"] = flash_x
        filenames["flash"]   = flash
        filenames["csum"]    = replaceSuffix (output, suffix, "_csum")

        filenames["spt"]     = getRootname (self.input, "_raw") + "_spt.fits"

        return filenames

    def checkImSpecSwitches (self):
        """Turn off switches depending on obstype.

        This routine resets switches to OMIT or PERFORM, depending
        primarily on whether the observation is imaging or spectroscopic.
        A list of messages (or an empty list) regarding which switch
        values were overridden will be returned.  Other routines may
        reset switches depending on time-tag vs accum, or FUV vs NUV,
        and they would append their messages to the list returned by
        this function, and then print the complete list.
        """

        messages = []

        if self.exp_type == EXP_WAVECAL:
            # Silently set these switches.
            self.switches["doppcorr"] = "OMIT"
            self.switches["helcorr"] = "OMIT"
            self.switches["fluxcorr"] = "OMIT"
            self.switches["tdscorr"] = "OMIT"

        if self.info["aperture"] == "WCA" or self.info["aperture"] == "FCA":
            self.overrideSwitch ("photcorr", messages)

        if self.info["obstype"] == "IMAGING" or \
           self.exp_type == EXP_TARGET_ACQ or \
           self.exp_type == EXP_ACQ_IMAGE:

            self.overrideSwitch ("doppcorr", messages)
            self.overrideSwitch ("helcorr", messages)
            self.overrideSwitch ("backcorr", messages)
            self.overrideSwitch ("fluxcorr", messages)

        else:                                   # spectroscopic

            if self.info["obsmode"] == "TIME-TAG" and \
               self.info["doppmagv"] == 0.:
                self.overrideSwitch ("doppcorr", messages)

            if self.info["obsmode"] == "ACCUM" and \
               self.info["dopmagt"] == 0:
                self.overrideSwitch ("doppcorr", messages)

            if self.info["ra_targ"] < 0.:
                self.overrideSwitch ("helcorr", messages)

            if self.switches["x1dcorr"] != "PERFORM":
                # Can't do backcorr or fluxcorr without 1-D extraction.
                self.overrideSwitch ("backcorr", messages)
                self.overrideSwitch ("fluxcorr", messages)
                self.overrideSwitch ("tdscorr", messages)

        return messages

    def overrideSwitch (self, keyword, messages, reset_to="OMIT"):
        """If switch for keyword is "PERFORM", reset it to "OMIT".

        @param keyword: a calibration switch keyword
        @type keyword: string

        @param messages: tells what keywords have been changed; modified
            in-place
        @type messages: string

        @param reset_to: value to assign to keyword (e.g. "OMIT" or "SKIPPED")
        @type reset_to: string
        """

        key_lower = keyword.lower()
        if self.switches.has_key (key_lower):
            if self.switches[key_lower] == "PERFORM":
                self.switches[key_lower] = reset_to
                messages.append (keyword.upper() + " reset to " + reset_to)
        else:
            self.switches[key_lower] = reset_to

    def printSwitchMessages (self, messages, input):
        """Print info about which calibration switches are being reset.

        @param messages: tells what keywords have been changed
        @type messages: string

        @param input: name of an input file (to be included in the text
            that is printed)
        @type input: string
        """

        if len (messages) > 0:
            msg = "Warning:  The following calibration switch"
            if len (messages) > 1:
                msg += "es"
            msg += " will be reset as shown,"
            cosutil.printMsg (msg, VERBOSE)
            cosutil.printMsg ("  for file " + input, VERBOSE)
            for msg in messages:
                cosutil.printMsg (msg, VERBOSE)

class FUVTimetagObs (Observation):

    def __init__ (self, input, outdir, memtype, first=False):

        Observation.__init__ (self, input, outdir, memtype, "_rawtag", first)

    def checkSwitches (self):

        messages = self.checkImSpecSwitches()

        self.printSwitchMessages (messages, self.input)

class FUVAccumObs (Observation):

    def __init__ (self, input, outdir, memtype, first=False):

        Observation.__init__ (self, input, outdir, memtype, "_rawaccum", first)

    def checkSwitches (self):

        messages = self.checkImSpecSwitches()

        # Note that this tests on DOPPONT, while the generic test in
        # checkImSpecSwitches uses DOPPMAGV (for time-tag) or
        # DOPMAGT (for accum).
        if not self.info["doppont"]:
            self.overrideSwitch ("doppcorr", messages, reset_to="SKIPPED")

        self.printSwitchMessages (messages, self.input)

class NUVTimetagObs (Observation):

    def __init__ (self, input, outdir, memtype, first=False):

        Observation.__init__ (self, input, outdir, memtype, "_rawtag", first)

    def checkSwitches (self):

        messages = self.checkImSpecSwitches()

        self.overrideSwitch ("tempcorr", messages)
        self.overrideSwitch ("geocorr", messages)
        self.overrideSwitch ("igeocorr", messages)
        self.overrideSwitch ("randcorr", messages)
        self.overrideSwitch ("phacorr", messages)

        self.printSwitchMessages (messages, self.input)

class NUVAccumObs (Observation):

    def __init__ (self, input, outdir, memtype, first=False):

        Observation.__init__ (self, input, outdir, memtype, "_rawaccum", first)

    def checkSwitches (self):

        messages = self.checkImSpecSwitches()

        self.overrideSwitch ("tempcorr", messages)
        self.overrideSwitch ("geocorr", messages)
        self.overrideSwitch ("igeocorr", messages)
        self.overrideSwitch ("randcorr", messages)
        self.overrideSwitch ("phacorr", messages)
        if not self.info["doppont"]:
            self.overrideSwitch ("doppcorr", messages, reset_to="SKIPPED")

        self.printSwitchMessages (messages, self.input)

class Calibration (object):
    """Calibrate COS data.

    The attributes are:
        assoc              the Association instance
        wavecal_info       list of dictionaries, each of which contains the
                             following:
                               time (MJD of middle of exposure)
                               fpoffset (header keyword fpoffset)
                               shift dictionary:  keys are shift1a, shift1b,
                                 and (if NUV) shift1c; value is the shift
                                 that was determined, in pixels; positive
                                 shift means that features in the spectrum
                                 were found at larger pixel number than the
                                 nominal location
                               rootname
        wcp_info           matching row (just one) from the wavecal
                             parameters table
    """

    def __init__ (self, assoc):
        """Constructor

        @param assoc: an Association object
        @type assoc: instance
        """

        self.assoc = assoc
        self.wavecal_info = []
        self.wcp_info = None

    def basicCal (self, filenames, info, switches, reffiles):
        """Do the "basic" calibration.

        @param filenames: input and output file names
        @type filenames: dictionary

        @param info: values of header keywords for general information
        @type info: dictionary

        @param switches: values of header keywords for calibration switches
        @type switches: dictionary

        @param reffiles: values of header keywords for reference file names
        @type reffiles: dictionary
        """

        input = filenames["raw"]
        inpha = filenames["pha"]
        output = filenames["flt"]
        outtag = filenames["corrtag"]
        outcounts = filenames["counts"]
        if info["tagflash"]:
            outflash = filenames["flash_x"]
        else:
            outflash = None
        if self.assoc.cl_args["create_csum_image"]:
            outcsum = filenames["csum"]
        else:
            outcsum = None
        if info["obsmode"] == "TIME-TAG":
            status = timetag.timetagBasicCalibration (input, None, outtag,
                        output, outcounts, outflash, outcsum,
                        self.assoc.cl_args,
                        info, switches, reffiles,
                        self.wavecal_info)
        else:
            status = accum.accumBasicCalibration (input, inpha, outtag,
                        output, outcounts, outcsum,
                        self.assoc.cl_args,
                        info, switches, reffiles,
                        self.wavecal_info)

    def allWavecals (self):
        """Process all the wavecal observations in the association."""

        if self.assoc.global_switches["wavecal"] != "PERFORM":
            return

        cosutil.printMsg ("Begin calibration of wavecals.", VERY_VERBOSE)

        # First calibrate all the wavecals.
        for obs in self.assoc.obs:
            if obs.exp_type == EXP_WAVECAL:
                obs.openTrailer()
                if self.wcp_info is None:
                    # Read info from wavecal parameters table.
                    wcp_info = cosutil.getTable (obs.reffiles["wcptab"],
                               filter={"opt_elem": obs.info["opt_elem"]},
                               exactly_one=True)
                    self.wcp_info = wcp_info[0]
                self.basicCal (obs.filenames,
                        obs.info, obs.switches, obs.reffiles)
                # Find spectrum in cross-dispersion direction.
                # (xd_shifts and xd_locns are ignored.)
                (shift2, xd_shifts, xd_locns, lamp_is_on) = \
                wavecal.findWavecalSpectrum (obs.filenames["corrtag"],
                                             obs.info, obs.reffiles)
                # Update shift2[a-c] keywords, and possibly lampused.
                self.setSpectrumOffset (obs.filenames,
                        obs.info["segment"], shift2, lamp_is_on)
                self.extractSpectrum (obs.filenames)
                obs.closeTrailer()

        self.concatenateSpectra ("wavecal")

        # Now find the shift of each wavecal.
        self.processWavecal()

        # Set the shift keywords in the corrtag, flt, and counts headers
        # (already set in x1d header) for each wavecal observation.
        # Compute wavelengths and assign to the wavelength column in the
        # corrtag tables.
        for obs in self.assoc.obs:
            if obs.exp_type == EXP_WAVECAL:
                self.setWavecalShift (obs.filenames)
                self.corrtagWavelengths (obs.filenames["corrtag"],
                                         obs.info, obs.reffiles)

        cosutil.printMsg ("wavecal_info = " + repr (self.wavecal_info),
                VERY_VERBOSE)

        # Update the wavelength column in the x1d table to take account of
        # the shift in the dispersion direction.
        previous_x1d_file = " "
        for obs in self.assoc.obs:
            x1d_file = obs.filenames["x1d"]
            # For FUV, we expect duplicate x1d file names in the obs list.
            if x1d_file == previous_x1d_file:
                continue
            previous_x1d_file = x1d_file
            if obs.exp_type == EXP_WAVECAL:
                obs.openTrailer()
                extract.recomputeWavelengths (x1d_file)
                obs.closeTrailer()

    def allScience (self):
        """Process all the science observations in the association."""

        if self.assoc.global_switches["science"] != "PERFORM":
            return

        cosutil.printMsg ("Begin calibration of science data.", VERY_VERBOSE)
        # initial values
        any_x1dcorr = "omit"
        any_wavecorr = "omit"
        for obs in self.assoc.obs:
            if obs.exp_type == EXP_SCIENCE or \
               obs.exp_type == EXP_CALIBRATION or \
               obs.exp_type == EXP_ACQ_IMAGE:
                obs.openTrailer()
                self.basicCal (obs.filenames,
                        obs.info, obs.switches, obs.reffiles)
                self.updateShift (obs.filenames, obs.switches["wavecorr"],
                            obs.info)
                if obs.switches["x1dcorr"] == "PERFORM":
                    self.extractSpectrum (obs.filenames)
                    any_x1dcorr = "PERFORM"
                elif obs.info["obstype"] == "SPECTROSCOPIC":
                    cosutil.printSwitch ("X1DCORR", obs.switches)
                if obs.info["tagflash"]:
                    if obs.switches["wavecorr"] == "PERFORM" or \
                       obs.switches["wavecorr"] == "COMPLETE":
                        any_wavecorr = "PERFORM"
                obs.closeTrailer()

        if any_x1dcorr == "PERFORM":
            self.concatenateSpectra ("science")

        if any_wavecorr == "PERFORM":
            self.concatenateSpectra ("tagflash")

    def extractSpectrum (self, filenames):
        """Extract a 1-D spectrum from corrtag table or from 2-D images.

        @param filenames: input and output file names
        @type filenames: dictionary

        The 1-D spectrum will be extracted from the 2-D flt and counts images.
        """

        input = filenames["flt"]
        incounts = filenames["counts"]
        output = filenames["x1d_x"]

        extract.extract1D (input, incounts, output)

    def processWavecal (self):
        """Determine shift from wavecal observation.

        The shift and related info will be appended to the wavecal_info list.
        """
        cosutil.printSwitch ("WAVECORR", {"wavecorr": "PERFORM"})

        previous_x1d_file = " "
        first = True
        for obs in self.assoc.obs:

            x1d_file = obs.filenames["x1d"]
            # For FUV, we expect duplicate x1d file names in the obs list.
            if x1d_file == previous_x1d_file:
                continue
            previous_x1d_file = x1d_file

            if obs.exp_type == EXP_WAVECAL:

                obs.openTrailer()
                if first:
                    wavecal.printWavecalRef (obs.reffiles)
                    first = False
                cosutil.printFilenames ([("Input", x1d_file)])
                shift_dict = wavecal.findWavecalShift (x1d_file,
                                self.assoc.cl_args["shift_file"], obs.info,
                                self.wcp_info)

                if shift_dict is not None:
                    # time is the MJD at the midpoint of the exposure.
                    time = cosutil.timeAtMidpoint (obs.info)
                    wavecal.storeWavecalInfo (self.wavecal_info,
                            time, obs.info["fpoffset"],
                            shift_dict, obs.filenames["root"],
                            obs.filenames["raw"])
                obs.closeTrailer()

    def updateShift (self, filenames, wavecorr, info):
        """Update the shift keywords in corrtag, flt, counts headers.

        This function is only relevant for ACCUM mode data.
        The shift for the two segments (or three NUV stripes) will be copied
        (or interpolated) from the list of wavecal information to the
        keywords SHIFT1A, SHIFT1B, SHIFT1C, SHIFT2A, SHIFT2B, SHIFT2C.

        @param filenames: input and output file names
        @type filenames: dictionary

        @param wavecorr: "PERFORM" if wavecal processing is being done
        @type wavecorr: string

        @param info: values of header keywords for general information
        @type info: dictionary
        """

        if info["obsmode"] == "TIME-TAG":
            return
        if info["exptype"] == "ACQ/IMAGE":
            return

        if wavecorr != "PERFORM" and wavecorr != "COMPLETE":
            shift_dict = None
        elif len (self.wavecal_info) < 1:
            shift_dict = None
        else:
            shift_dict = None                           # replaced below
            time = cosutil.timeAtMidpoint (info)        # MJD
            shift_info = wavecal.returnWavecalShift (self.wavecal_info,
                         self.wcp_info, info["fpoffset"], time)
            if len (self.wavecal_info) > 0 and shift_info is not None:
                # only the shift will be used, not the slope or the file name
                (shift_dict, slope_dict, wavecal_filename) = shift_info
                cosutil.printSwitch ("WAVECORR", {"wavecorr": "PERFORM"})
                if cosutil.checkVerbosity (VERY_VERBOSE):
                    keywords = shift_dict.keys()
                    keywords.sort()
                    for key in keywords:
                        cosutil.printMsg (
                            "  %s = %.4f" % (key.upper(), shift_dict[key]),
                            VERY_VERBOSE)

        if shift_dict is None:
            cosutil.printMsg (
                "Warning:  No wavecal info; shift assumed to be 0.", VERBOSE)

        # corrtag is in this list because there might be an output file for
        # the pseudo-corrtag table.
        for fname in [filenames["corrtag"], \
                      filenames["flt"], filenames["counts"]]:
            if os.access (fname, os.R_OK):
                fd = pyfits.open (fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd[("SCI",1)].header
                if wavecorr == "PERFORM" and len (self.wavecal_info) > 0:
                    phdr.update ("WAVECORR", "COMPLETE")
                hdr.update ("DPIXEL1A", 0.)     # dpixel1 not used for ACCUM
                hdr.update ("DPIXEL1B", 0.)
                if info["detector"] == "NUV":
                    hdr.update ("DPIXEL1C", 0.)
                if shift_dict is None:
                    hdr.update ("SHIFT1A", 0.)
                    hdr.update ("SHIFT1B", 0.)
                    hdr.update ("SHIFT2A", 0.)
                    hdr.update ("SHIFT2B", 0.)
                    if info["detector"] == "NUV":
                        hdr.update ("SHIFT1C", 0.)
                        hdr.update ("SHIFT2C", 0.)
                else:
                    for key in shift_dict.keys():
                        shift = shift_dict[key]
                        shift = round (shift, 4)        # round to four places
                        hdr.update (key, shift)
                fd.close()

    def setSpectrumOffset (self, filenames, segment, shift2, lamp_is_on):
        """Update the shift2 keywords in corrtag, flt, counts headers.

        This function is called only for a wavecal, not for a science
        observation.  (For science data, the shift2 keyword(s) will be
        updated by updateShift.)

        @param filenames: input and output file names
        @type filenames: dictionary
        @param segment: FUV segment name or NUV stripe name
        @type segment: string
        @param shift2: offset in cross-dispersion direction, as determined
            from (conventional) wavecal data
        @type shift2: float
        @param lamp_is_on: True if the wavecal lamp was actually on
        @type lamp_is_on: boolean
        """

        if segment[0:3] == "FUV":
            keywords = ["SHIFT2"+segment[-1]]
        else:
            keywords = ["SHIFT2A", "SHIFT2B", "SHIFT2C"]
        # flags to control printing of messages regarding keyword lampused
        print_msg1 = False
        print_msg2 = False
        print_msg3 = False
        print_msg4 = False
        for fname in [filenames["corrtag"], \
                      filenames["flt"], filenames["counts"]]:
            if os.access (fname, os.R_OK):
                fd = pyfits.open (fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd[("SCI",1)].header
                for keyword in keywords:
                    hdr.update (keyword, round (shift2, 4))
                lampused = phdr.get ("lampused", "missing")
                lampplan = phdr.get ("lampplan", "missing")
                if lamp_is_on and lampused == "NONE":
                    if lampplan == "missing":
                        print_msg1 = True
                    else:
                        print_msg2 = True
                        phdr["lampused"] = lampplan
                if not lamp_is_on:
                    print_msg3 = True
                    if lampused != "NONE":
                        print_msg4 = True
                        phdr["lampused"] = "NONE"
                fd.close()
        if print_msg1:
            cosutil.printWarning ("The wavecal lamp was on, but LAMPUSED = " \
                                  "%s and LAMPPLAN is missing." % \
                                  lampused, level=VERBOSE)
        if print_msg2:
            cosutil.printMsg ("LAMPUSED = %s, which is incorrect; " \
                              "the value will be reset to %s." % \
                              (lampused, lampplan), level=VERBOSE)
        if print_msg3:
            cosutil.printWarning ("The wavecal lamp was off for a wavecal!", \
                                  level=VERBOSE)
        if print_msg4:
            cosutil.printMsg ("LAMPUSED = %s, and it will be reset to NONE." \
                              % lampused, level=VERBOSE)

    def setWavecalShift (self, filenames):
        """Update the shift keywords in corrtag, flt, counts headers.

        This function is called only for a wavecal, not for a science
        observation.  There must be an exact match with the rootname of the
        observation.

        @param filenames: input and output file names
        @type filenames: dictionary
        """

        shift_dict = wavecal.returnExactMatch (self.wavecal_info,
                             filenames["root"])
        if shift_dict is None:
            return

        for fname in [filenames["corrtag"], \
                      filenames["flt"], filenames["counts"]]:
            if os.access (fname, os.R_OK):
                fd = pyfits.open (fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd[("SCI",1)].header
                phdr.update ("WAVECORR", "COMPLETE")
                for keyword in shift_dict.keys():
                    shift = shift_dict[keyword]
                    hdr.update (keyword, round (shift, 4))
                fd.close()

    def corrtagWavelengths (self, corrtag, info, reffiles):
        """Compute and assign wavelengths in the corrtag table.

        This function is only called for a wavecal (auto or GO).  For science
        exposures, the wavelengths are assigned during time-tag processing.

        @param filenames: input and output file names
        @type filenames: dictionary
        """

        if os.access (corrtag, os.R_OK):
            fd = pyfits.open (corrtag, mode="update")
            events = fd["EVENTS"].data
            hdr = fd["EVENTS"].header
            timetag.computeWavelengths (events, info, reffiles, hdr=hdr)
            fd.close()

    def mergeKeywords (self):
        """Copy segment-specific keywords between FUV pairs of files.

        Keywords that have different names for the A and B segments will be
        copied from flt_a to flt_b and vice versa, and similarly for the
        counts files.
        """

        # The strings in this list use "X" as a character to be replaced
        # by "a" or "b" to get lists a_kwds and b_kwds respectively.
        incl_wildcard = ["stimX_lx", "stimX_ly", "stimX_rx", "stimX_ry",
                "stimX0lx", "stimX0ly", "stimX0rx", "stimX0ry",
                "stimXslx", "stimXsly", "stimXsrx", "stimXsry",
                "npha_X", "phalowrX", "phaupprX",
                "tbrst_X", "nbrst_X", "tbadt_X", "nbadt_X",
                "nout_X",
                "globrt_X",
                "deadrt_X", "deadmt_X", "livetm_X",
                "sp_loc_X", "sp_slp_X",
                "b_bkg1_X", "b_bkg2_X",
                "b_hgt1_X", "b_hgt2_X",
                "shift1X", "shift2X", "dpixel1X",
                "chi_sq_X", "ndf_X"]
        a_kwds = []
        b_kwds = []
        for keyword in incl_wildcard:
            a_kwds.append (keyword.replace ("X", "a"))
            b_kwds.append (keyword.replace ("X", "b"))

        for files in self.assoc.merge_kwds:
            assert len (files) == 2
            files.sort()
            fd_a = pyfits.open (files[0], mode="update")
            fd_b = pyfits.open (files[1], mode="update")
            hdr_a = fd_a[1].header
            hdr_b = fd_b[1].header
            for i in range (len (a_kwds)):
                keyword_a = a_kwds[i]
                keyword_b = b_kwds[i]
                if hdr_a.has_key (keyword_a):
                    hdr_b.update (keyword_a, hdr_a[keyword_a])
                if hdr_b.has_key (keyword_b):
                    hdr_a.update (keyword_b, hdr_b[keyword_b])
            fd_a.close()
            fd_b.close()

    def concatenateSpectra (self, type):
        """Concatenate two 1-D FUV spectra into one spectrum.

        If type="wavecal", this routine will concatenate pairs of wavecal
        files; if type="science", this routine will concatenate pairs of
        science files.  The input _x1d_a and _x1d_b will then be deleted,
        if save_temp_files = False.

        @param type: "science", "wavecal" or "tagflash" (ignore if "unknown")
        @type type: string
        """

        for one_set in self.assoc.concat:

            if one_set["type"] == type:

                infiles = one_set["input"]
                output = one_set["output"]
                if len (infiles) < 1:
                    continue

                if len (infiles) == 1:
                    if os.access (infiles[0], os.R_OK) and infiles[0] != output:
                        cosutil.renameFile (infiles[0], output)
                    else:
                        continue
                else:
                    extract.concatenateFUVSegments (infiles, output)
                    if not self.assoc.cl_args["save_temp_files"]:
                        # Delete the _x1d_a.fits and _x1d_b.fits files.
                        for file in infiles:
                            if os.access (file, os.R_OK):
                                os.remove (file)

    def combineToProduct (self):
        """Average the calibrated files, producing the product files."""

        if self.assoc.product is None:
            return

        combine = self.assoc.combine

        if combine.has_key ("flt"):
            self.combineFlt()

        i = self.assoc.first_science
        if self.assoc.obs[i].switches["x1dcorr"] != "PERFORM":
            return

        # Average all the x1d spectroscopic exposures in the association.
        self.combineAllX1D()

        # If we have more than one spectroscopic exposure in the association,
        # average x1d files that have the same fppos index.
        if combine.has_key ("x1d"):
            x1d_list = combine["x1d"]
            fppos_list = combine["fppos"]
            fppos_list_copy = copy.copy (fppos_list)
            fppos_list_copy.sort()
            fppos_max = fppos_list_copy[-1]
            for fppos in range (1, fppos_max+1):
                # extract subset for current osm position
                x1d_subset = []
                for i in range (len (x1d_list)):
                    if fppos_list[i] == fppos:
                        x1d_subset.append (x1d_list[i])
                if len (x1d_subset) > 0:
                    self.combineX1Di (x1d_subset, fppos)

    def combineFlt (self):
        """Average image mode data."""

        combine = self.assoc.combine

        if combine.has_key ("flt"):
            output = self.fltProductName()
            average.avgImage (combine["flt"], output)

    def combineAllX1D (self):
        """Average x1d data for all OSM positions."""

        combine = self.assoc.combine

        if combine.has_key ("x1d"):
            output = self.x1dProductName (0)
            fpavg.fpAvgSpec (combine["x1d"], output)

    def combineX1Di (self, input, fppos):
        """Average the x1d data for one specified FPPOS position.

        @param input: name of input file
        @type input: string

        @param fppos: value of header keyword FPPOS
        @type fppos: integer
        """

        output = self.x1dProductName (fppos)

        fpavg.fpAvgSpec (input, output)

    def fltProductName (self):
        """Construct the product name for the flt file.

        @return: name of output flt file
        @rtype: string
        """

        output = self.assoc.product + "_fltsum.fits"

        return output

    def x1dProductName (self, fppos=0):
        """Construct the product name for the x1d file.

        If fppos is greater than zero, then the output file name will be of
        the form "rootname_x1dsum1.fits", where the number appended to "x1dsum"
        will be the value of fppos.

        @param fppos: FPPOS index (0 or 1-4)
        @type fppos: integer

        @return: name of output x1d file
        @rtype: string
        """

        output = self.assoc.product + "_x1dsum"
        if fppos > 0:
            output += str (fppos) + ".fits"
        else:
            output += ".fits"

        return output

if __name__ == "__main__":

    main (sys.argv[1:])
