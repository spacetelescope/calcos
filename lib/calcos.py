#! /usr/bin/env python

import sys
import os
import time
import getopt
import glob
import copy

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
EXP_TARGET_ACQ  = 3
EXP_ACQ_IMAGE   = 4
EXP_ENGINEERING = 5

def main (args):
    """Check arguments and call calcos.

    This driver interprets command-line arguments and calls calcos for
    each association file or raw file specified on the command line.

    The command-line options are:
        -q (quiet)
        -v (very verbose)
        -s (save temporary files)
        -o outdir (output directory name)
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
                           ["stim=", "live=", "burst="])
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

    for i in range (len (infiles)):
        calcos (infiles[i], outdir, save_temp_files,
                stimfile, livetimefile, burstfile)

def prtOptions():
    """Print a list of command-line options and arguments."""

    cosutil.printMsg ("The command-line options are:")
    cosutil.printMsg ("  -q (quiet)")
    cosutil.printMsg ("  -v (very verbose)")
    cosutil.printMsg ("  -s (save temporary files)")
    cosutil.printMsg ("  -o outdir (output directory name)")
    cosutil.printMsg ("  --stim filename (append stim locations to filename)")
    cosutil.printMsg ("  --live filename (append livetime factors to filename)")
    cosutil.printMsg ("  --burst filename (append burst info to filename)")
    cosutil.printMsg ("")
    cosutil.printMsg ("Following the options, list one or more association")
    cosutil.printMsg ("files (rootname_asn) or raw files (rootname_raw).")

def uniqueInput (infiles):
    """Remove effective duplicates from list of files to process.

    argument:
    infiles           list of input files

    The function argument is the list of input files but with
    duplicates removed.
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
    """Check that the environment variable NUMERIX is set to numpy."""

    if not os.environ.has_key ("NUMERIX"):
        cosutil.printWarning (
                "environment variable NUMERIX should be set to numpy")
    elif os.environ["NUMERIX"] != "numpy":
        cosutil.printWarning ("NUMERIX is set to '%s', should be numpy" % \
                              os.environ["NUMERIX"])

def calcos (asntable, outdir=None, quiet=False, save_temp_files=False,
            stimfile=None, livetimefile=None, burstfile=None):
    """Calibrate COS data.

    The arguments are:
    asntable         the rootname (with "_asn") of an association file, or
                     the rootname (with "_raw") of a raw file (or pair of
                     files if FUV)
    outdir           name of output directory, or None
    quiet            if quiet=True, set verbosity to QUIET
    save_temp_files  By default, the files containing the count rate image
                     will be deleted after extracting 1-D spectra; also,
                     the _x1d_a.fits and _x1d_b.fits files (if FUV) will
                     be deleted after concatenating to the _x1d.fits file.
                     Specify save_temp_files=True to keep these files.
    stimfile         If specified, the stim positions will be written to
                     (or appended to) a text file with this name.
    livetimefile     If specified, the livetime factors will be written to
                     (or appended to) a text file with this name.
    burstfile        If specified, burst information will be written to
                     (or appended to) a text file with this name.
    """

    t0 = time.time()
    cosutil.printMsg ("CALCOS version " + CALCOS_VERSION)
    cosutil.printMsg ("Begin " + cosutil.returnTime(), VERBOSE)

    # check that NUMERIX is set to numpy
    checkNumerix()

    if quiet:
        cosutil.setVerbosity (QUIET)
    assoc = Association (asntable, outdir, save_temp_files,
                         stimfile, livetimefile, burstfile)
    if len (assoc.obs) == 0:
        cosutil.printMsg ("The association table is empty.")
        return
    if not assoc.isAnySwitchSet():
        cosutil.printMsg ("Nothing to do; all calibration switches are OMIT.")
        return

    cal = Calibration (assoc)

    cal.allWavecals()

    cal.allScience()

    cal.combineToProduct()

    assoc.updateMempresent()

    cosutil.printMsg ("End   " + cosutil.returnTime(), VERBOSE)

    t1 = time.time()
    cosutil.printMsg ("elapsed time = %.1f sec. = %.2f min." % \
                                (t1-t0, (t1-t0)/60.), VERY_VERBOSE)

def replaceSuffix (rawname, suffix, new_suffix):
    """Replace the suffix in a raw file name.

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

    pieces = input.split (suffix)
    if len (pieces) > 1:
        root = suffix.join (pieces[:-1])
    elif input.endswith (".fits"):
        extn = input.rfind (".fits")
        root = input[:extn]
    else:
        root = input[:]

    return root

class Association:
    """Read and interpret the association table.

    The attributes are:
        asntable           full name of the association file, or None if the
                             name (or rootname) of a raw file was specified
        asn_info           a dictionary of the contents of the association table
        indir              name of input directory, or an empty string; if a
                             directory was specified, it will be added as a
                             prefix to the memnames in asn_info
        outdir             name of output directory, or an empty string
        combine            a dictionary of lists of file names to be
                             averaged (i.e. individual repeatobs or fp-split
                             exposures)
        concat             pairs of files of 1-D extracted FUV spectra for
                             segments A and B need to be concatenated; this
                             is a list of dictionaries with info about files
                             to be concatenated
        product            rootname of the product (rootname portion is lower
                             case; includes outdir)
        product_type       memtype of the product (case unchanged from asn
                             table)
        global_switches    dictionary of the global calibration switches
        rawfiles           a list of all rawtag or rawimage files
        obs                a list of Observation instances, one for each raw
                             file
        first_science      index of first science observation in obs list

     the following come from command-line options:
        save_temp_files    true if the count-rate files should not be
                             deleted
        stimfile           name of file for stim positions, or None
        livetimefile       name of file for livetime information, or None
        burstfile          name of file for burst information, or None
    """

    def __init__ (self, asntable, outdir, save_temp_files,
                  stimfile, livetimefile, burstfile):

        self.asn_info = {}          # association table info
        self.combine = {}           # files to combine
        self.concat = []            # list of dictionaries of concat info
        self.product = None         # rootname of product
        self.product_type = None    # memtype of product
        self.global_switches = {}   # global calibration switches
        self.rawfiles = []          # list of all raw input files
        self.obs = []               # list of Observations
        self.first_science = 0      # defaults to first file

        # Copy command-line options to attributes.
        self.asntable = None                    # initial value
        self.save_temp_files = save_temp_files
        self.stimfile = stimfile
        self.livetimefile = livetimefile
        self.burstfile = burstfile

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
                self.rawfiles.extend (basic_info["rawfiles"])
                concat_info = {}                # will be one element of concat
                concat_these = []               # x1d_a and x1d_b names
                concat_info_flash = {}          # another element of concat
                concat_these_flash = []         # flash_a and flash_b names
                first = True                    # first of a pair for FUV
                for input in basic_info["rawfiles"]:    # one (NUV) or two (FUV)
                    obs = initObservation (input, self.outdir, memtype[i],
                          basic_info["detector"], basic_info["obsmode"])
                    self.obs.append (obs)
                    if basic_info["detector"] == "FUV":
                        concat_these.append (obs.filenames["x1d_x"])
                        if obs.info["tagflash"]:
                            concat_these_flash.append (obs.filenames["flash_x"])
                    if first:
                        if obs.exp_type == EXP_SCIENCE:
                            concat_info["type"] = "science"
                            if obs.info["tagflash"]:
                                concat_info_flash["type"] = "tagflash"
                        elif obs.exp_type == EXP_WAVECAL:
                            concat_info["type"] = "wavecal"
                        else:
                            concat_info["type"] = "unknown"
                        concat_info["output"] = obs.filenames["x1d"]
                        if obs.info["tagflash"]:
                            concat_info_flash["output"] = obs.filenames["flash"]
                    if obs.exp_type == EXP_SCIENCE:
                        self.updateCombineFlt (obs.filenames,
                                  basic_info["detector"])
                        if first:
                            self.updateCombineX1d (obs.filenames,
                                  obs.info["fppos"])
                    first = False

                if concat_these:
                    concat_info["input"] = concat_these
                    self.concat.append (concat_info)
                if concat_these_flash:
                    concat_info_flash["input"] = concat_these_flash
                    self.concat.append (concat_info_flash)

        if len (self.obs) == 0:
            return

        cosutil.printMsg ("combine = " + repr (self.combine), VERY_VERBOSE)
        cosutil.printMsg ("concat = " + repr (self.concat), VERY_VERBOSE)

        # Find the first science obs (non-wavecal and non-TA1, if possible).
        self.first_science = self.findFirstScience()

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

        # Convert the memnames to lower case, and prefix them with the
        # input directory name.
        asn_memname = asn_data.field ("memname")
        asn_memtype = asn_data.field ("memtype")
        asn_memprsnt = asn_data.field ("memprsnt")
        for i in range (nrows):
            self.asn_info["memname"].append ( \
                        os.path.join (self.indir, asn_memname[i].lower()))
            self.asn_info["memtype"].append (asn_memtype[i])
            self.asn_info["mempresent"].append (asn_memprsnt[i])

        fd.close()

        self.product = None
        for i in range (nrows):
            if asn_memtype[i].find ("PROD") >= 0:
                if self.product is not None:
                    raise RuntimeError, \
                    "The association table may list no more than one product."
                self.product = asn_memname[i]
                self.product_type = asn_memtype[i]

        if self.product is not None:
            self.product = os.path.join (self.outdir, self.product)
            cosutil.printMsg ("product = " + self.product, VERY_VERBOSE)

    def dummyAsnTable (self, asntable):
        """Construct a recarray corresponding to an association table.

        This function will be called for the case that the user specified
        the name of a raw file instead of an association table.  The
        'asntable' argument is the name as given by the user; we will
        extract the root that precedes "_raw*.fits", i.e. excluding the
        suffix.  A recarray object will be created that has the appropriate
        column names for an association table, and the "row" of data will be
        assigned the specified rootname.  There will only be this one row;
        product will be set to None.  The memtype will be set to "none",
        even though it might actually be a wavecal.
        """

        cosutil.printMsg ("Input file = " + asntable, VERBOSE)

        rootname = getRootname (asntable, "_raw")

        self.asn_info["memname"] = [rootname]
        self.asn_info["memtype"] = ["none"]
        self.asn_info["mempresent"] = [True]    # yes, it is present

        # Because the input is not an association, there is no product.
        self.product = None
        self.product_type = None

    def initialInfo (self, memname):
        """Get preliminary information from an input file.

        This gets the names of the raw files, and from the first of those
        files, reads the primary header and calls a function to get
        DETECTOR, OBSMODE, and EXPTYPE.  In addition, this function checks
        that the suffixes are as expected for the DETECTOR and OBSMODE
        keywords.
        """

        # Find the names of all raw files with the specified rootname.
        rawfiles = []

        # The number of files of each type, based on suffix.
        num = {}

        raw = glob.glob (memname + "_rawtag_[ab].fits")
        rawfiles.extend (raw)
        num["fuv_timetag"] = len (raw)

        raw = glob.glob (memname + "_rawtag.fits")
        rawfiles.extend (raw)
        num["nuv_timetag"] = len (raw)

        raw = glob.glob (memname + "_rawimage_[ab].fits")
        rawfiles.extend (raw)
        num["fuv_accum"] = len (raw)

        raw = glob.glob (memname + "_rawimage.fits")
        rawfiles.extend (raw)
        num["nuv_accum"] = len (raw)

        nfiles = len (rawfiles)
        if nfiles == 0:
            raise RuntimeError, \
                  "There are no raw files for rootname `%s'" % memname
        rawfiles.sort()

        # Read the first raw file with the specified rootname.
        filename = rawfiles[0]

        basic_info = getinfo.initialInfo (filename)

        # initial value; one element will be deleted
        check = ["fuv_timetag", "fuv_accum", "nuv_timetag", "nuv_accum"]
        if basic_info["detector"] == "FUV":
            if basic_info["obsmode"] == "TIME-TAG":
                del check[0]                    # FUV time-tag
            else:
                del check[1]                    # FUV accum
        else:
            if basic_info["obsmode"] == "TIME-TAG":
                del check[2]                    # NUV time-tag
            else:
                del check[3]                    # NUV accum

        bad_name = False
        for key in check:
            if num[key] > 0:
                bad_name = True
                break
        if bad_name:
            cosutil.printError (
    "Filename suffixes are not as expected, based on DETECTOR and OBSMODE:")
            cosutil.printContinuation ("filenames are " + repr (rawfiles))
            raise RuntimeError

        basic_info["rawfiles"] = rawfiles

        return basic_info

    def updateCombineFlt (self, filenames, detector):
        """Add the flt name to the input lists in 'combine'."""

        if detector == "FUV":
            if not self.combine.has_key ("flt_a"):
                self.combine["flt_a"] = []
            if not self.combine.has_key ("flt_b"):
                self.combine["flt_b"] = []
        else:
            if not self.combine.has_key ("flt"):
                self.combine["flt"] = []

        flt = filenames["flt"]
        if detector == "FUV":
            if flt.endswith ("_a.fits"):
                self.combine["flt_a"].append (flt)
            elif flt.endswith ("_b.fits"):
                self.combine["flt_b"].append (flt)
            else:
                # can't happen
                raise RuntimeError, "Internal error in updateCombineFlt"
        else:
            self.combine["flt"].append (flt)

    def updateCombineX1d (self, filenames, fppos):
        """Add the x1d name and osm index to 'combine'."""

        if not self.combine.has_key ("x1d"):
            self.combine["x1d"] = []
        self.combine["x1d"].append (filenames["x1d"])

        if not self.combine.has_key ("fppos"):
            self.combine["fppos"] = []
        self.combine["fppos"].append (fppos)

    def findFirstScience (self):
        """Find the first science file in the list.

        This is a utility function to return the index of the first science
        observation.  If there are no science observations, this returns the
        index of the first wavecal.  If there are no wavecals either, this
        returns zero (the first observation could be a TA1).
        """

        if len (self.obs) < 2:
            return 0

        foundit = False
        for i in range (len (self.obs)):
            obs = self.obs[i]
            if obs.exp_type == EXP_SCIENCE:
                foundit = True
                break
        if not foundit:
            # No science observation; find the first wavecal, if any.
            for i in range (len (self.obs)):
                obs = self.obs[i]
                if obs.exp_type == EXP_WAVECAL:
                    foundit = True
                    break
        if not foundit:
            # No wavecal observation; take the first observation of any kind.
            i = 0

        return i

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
        the observation list.  If there is any mismatch, the file name in
        the first science observation will take precedence.
        """

        # Take the reference file list from the first segment for the first
        # science observation.
        reffiles = self.obs[self.first_science].reffiles

        # Now do the comparisons.  'a_file' is the value of a reference file
        # keyword, and 'compare' is the value in the first science observation.
        message_printed = False
        for obs in self.obs:
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
                            a_file_hdr + " reset to " + compare_hdr)
                    obs.reffiles[key] = compare
                    obs.reffiles[key+"_hdr"] = compare_hdr

    def compareSwitches (self):
        """Compare switches.

        This function compares the values of the calibration switch keywords
        in the observation list.  If there is a mismatch, the value in the
        first first science observation will take precedence.  Allowance is
        made for differences in the type of observation, i.e. science,
        wavecal, or TA1.
        """

        # Take the list of switches from the first segment for the first
        # science observation.
        switches = self.obs[self.first_science].switches

        # Do the comparisons.  'sw' is the value of a calibration keyword,
        # and 'compare' is the value in the first science observation.
        message_printed = False
        for obs in self.obs:
            keys = switches.keys()
            keys.sort()
            for key in keys:
                compare = switches[key].strip()
                sw = obs.switches[key].strip()
                if sw != compare:
                    if obs.exp_type == EXP_WAVECAL:
                        if key in ["wavecorr", "rptcorr",
                                   "doppcorr", "helcorr", "fluxcorr"]:
                            continue
                    if not message_printed:
                        cosutil.printWarning (
                                "Inconsistent calibration switches:")
                        message_printed = True
                    if sw == "COMPLETE" or sw == "SKIPPED":
                        cosutil.printMsg (obs.input + ":  " + key + " = " + \
                                sw + " ... NOT reset")
                    else:
                        if len (compare) == 0:
                            compare = "(blank)"
                        if len (sw) == 0:
                            sw = "(blank)"
                        cosutil.printMsg (obs.input + ":  " + key + " = " + \
                                sw + " reset to " + compare)
                        obs.switches[key] = compare

    def missingRefFiles (self):
        """Check for missing reference files.

        This function opens each of the required reference files, gets the
        FILETYPE keyword from the primary header, and compares that value
        with the expected value.  It is an error if any of the reference
        files can't be opened, or if the value of FILETYPE doesn't match.

        Note that the minimum reference file version is specified here, for
        each reference file.  This is the min_ver value.
        """

        # Take info from the first science observation.
        i = self.first_science
        switches = self.obs[i].switches
        reffiles = self.obs[i].reffiles

        missing = {}            # reference file is not accessible
        wrong_filetype = {}     # wrong FILETYPE
        bad_version = {}        # inconsistent version strings

        # temp is a temporary dictionary with just min_ver and filetype,
        # for readability; these and other values will be copied to ref,
        # which is used for the argument to findRefFile.
        temp = {
            "flatfile": ["1.0", "FLAT FIELD REFERENCE IMAGE"],
            "badttab":  ["1.0", "BAD TIME INTERVALS TABLE"],
            "bpixtab":  ["1.0", "DATA QUALITY INITIALIZATION TABLE"],
            "deadtab":  ["1.0", "DEADTIME REFERENCE TABLE"],
            "brftab":   ["1.0", "BASELINE REFERENCE FRAME TABLE"],
            "phatab":   ["1.0", "PULSE HEIGHT PARAMETERS REFERENCE TABLE"],
            "geofile":  ["1.0", "GEOMETRIC DISTORTION REFERENCE IMAGE"],
            "lamptab":  ["1.0", "TEMPLATE CAL LAMP SPECTRA TABLE"],
            "wcptab":   ["1.0", "WAVECAL PARAMETERS REFERENCE TABLE"],
            "xtractab": ["1.0", "1-D EXTRACTION PARAMETERS TABLE"],
            "disptab":  ["1.0", "DISPERSION RELATION REFERENCE TABLE"],
            "phottab":  ["1.0", "PHOTOMETRIC SENSITIVITY REFERENCE TABLE"],
            "tdstab":   ["1.0", "TIME DEPENDENT SENSITIVITY TABLE"],
            "brsttab":  ["1.0", "BURST PARAMETERS TABLE"]
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
            cosutil.findRefFile (ref["phottab"],
                    missing, wrong_filetype, bad_version)
            if reffiles["tdstab"] != NOT_APPLICABLE:
                cosutil.findRefFile (ref["tdstab"],
                        missing, wrong_filetype, bad_version)

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

        The global switches are "any", "science", "wavecal", and "repeat".
        Their values are either "PERFORM" or "OMIT", though "science" and
        "wavecal" indicate the presence of one or more files of that type
        rather than an actual calibration switch.  "wavecal" refers to
        separate wavecal files, not concurrent science and wavecal data
        (tagflash); the latter would be included in "science".  "any" is
        "PERFORM" if any of the calibration steps other than wavecorr is
        "PERFORM".

        "repeat" depends on both the rptcorr switch and on the product row
        in the association table.  If there's no product, "repeat" will be
        "OMIT".  If there is a product, "repeat" will be "PERFORM" if the
        rptcorr switch is "PERFORM", or if there is only one calibrated
        file of a given type; in the latter case the files will just be
        renamed to the product name.
        """

        # Take the calibration switch list from the first science observation.
        switches = self.obs[self.first_science].switches

        # There are not the only calibration switches, but these are the
        # ones that are independent of others.  For example, it wouldn't
        # matter if fluxcorr were set to perform if x1dcorr were omit.
        self.global_switches["any"] = "OMIT"            # default value
        for key in ["dqicorr", "deadcorr", "tempcorr", "geocorr",
                    "phacorr", "flatcorr", "brstcorr", "doppcorr",
                    "helcorr", "x1dcorr", "rptcorr"]:
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
               obs.exp_type == EXP_ACQ_IMAGE:
                self.global_switches["science"] = "PERFORM"
                break

        if self.product is None:
            self.global_switches["repeat"] = "OMIT"
        else:
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
            if ncombine < 1:
                self.global_switches["repeat"] = "OMIT"
            elif ncombine == 1:
                # We'll actually just rename the file; the rptcorr switch
                # from the input header is irrelevant.
                self.global_switches["repeat"] = "PERFORM"
            else:
                self.global_switches["repeat"] = switches["rptcorr"]

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

        This routine checks for the existence of any output file for
        each of the input files.  If any are found it prints their names
        and raises an exception.
        """

        detector = self.obs[0].info["detector"]

        already_exists = []
        for obs in self.obs:
            if obs.info["obsmode"] == "TIME-TAG":
                self.checkExists (obs.filenames["corrtag"], already_exists)
                self.checkExists (obs.filenames["flash_x"], already_exists)
                self.checkExists (obs.filenames["flash"], already_exists)
            self.checkExists (obs.filenames["flt"], already_exists)
            self.checkExists (obs.filenames["counts"], already_exists)
            if obs.switches["x1dcorr"] == "PERFORM" or \
               obs.exp_type == EXP_WAVECAL:
                self.checkExists (obs.filenames["x1d_x"], already_exists)
                if obs.filenames["x1d"] != obs.filenames["x1d_x"]:
                    self.checkExists (obs.filenames["x1d"], already_exists)

        if self.global_switches["repeat"] == "PERFORM":
            product = self.product
            if detector == "FUV":
                self.checkExists (product + "_fltsum_a.fits", already_exists)
                self.checkExists (product + "_fltsum_b.fits", already_exists)
            else:
                self.checkExists (product + "_fltsum.fits", already_exists)
            self.checkExists (product + "_x1dsum.fits", already_exists)
            self.checkExists (product + "_x1dsum1.fits", already_exists)
            self.checkExists (product + "_x1dsum2.fits", already_exists)
            self.checkExists (product + "_x1dsum3.fits", already_exists)
            self.checkExists (product + "_x1dsum4.fits", already_exists)

        # Remove duplicates.
        for i in range (len (already_exists) - 1, 0, -1):
            fname = already_exists[i]
            if fname in already_exists[0:i]:
                del (already_exists[i])

        if already_exists:
            if len (already_exists) == 1:
                errmess = "output file already exists"
            else:
                errmess = "output files already exist"
            cosutil.printError (errmess + ":")
            for fname in already_exists:
                cosutil.printError ("  %s" % fname)
            raise RuntimeError, errmess

    def checkExists (self, fname, already_exists):
        """If fname exists, append the name to already_exists.

        arguments:
        fname             the name of the file
        already_exists    a list of file names; fname may be appended
        """

        if os.access (fname, os.R_OK):
            already_exists.append (fname)

    def stimfileSanityCheck (self):
        """Ignore stimfile if detector is not FUV.

        Only the FUV detector has stims.  If a file was specified for
        saving measured stim locations (--stim stimfile), the name will
        be reset to None if the detector was not FUV.
        """

        if self.stimfile is not None and \
           self.obs[self.first_science].info["detector"] != "FUV":
            self.stimfile = None
            cosutil.printWarning (
                "stimfile reset to None because detector is NUV.")

    def updateMempresent (self):
        """Update the MEMPRSNT flag in the association table."""

        if self.asntable is None or self.product is None:
            return
        cosutil.printMsg ("updateMempresent", VERY_VERBOSE)

        fd = pyfits.open (self.asntable, mode="update")

        asn = fd[1].data
        nrows = asn.shape[0]
        memtype = asn.field ("MEMTYPE")
        mempresent = asn.field ("MEMPRSNT")

        for i in range (nrows):
            if memtype[i].find ("PROD") >= 0:
                mempresent[i] = True
                break

        fd.close()

def initObservation (input, outdir, memtype, detector, obsmode):
    """Construct an Observation object for the current mode.

    arguments:
    input       the name of an input raw file
    outdir      either an empty string or the name of the output directory
    memtype     from association table; used to distinguish between
                  wavecal and science obervation
    detector    FUV or NUV
    obsmode     TIME-TAG or ACCUM
    """

    if detector == "FUV":
        if obsmode == "TIME-TAG":
            obs = FUVTimetagObs (input, outdir, memtype)
        else:
            obs = FUVAccumObs (input, outdir, memtype)
    else:
        if obsmode == "TIME-TAG":
            obs = NUVTimetagObs (input, outdir, memtype)
        else:
            obs = NUVAccumObs (input, outdir, memtype)

    return obs

class Observation:
    """Get information about an observation from its headers.

    This base class is not directly used; one of its subclasses will
    be invoked, depending on DETECTOR and OBSMODE.
    """

    def __init__ (self, input, outdir, memtype, suffix):
        """Invoked by a subclass.

        arguments:
        input       name of input raw file
        outdir      either an empty string or the name of the output directory
        memtype     from association table; used to distinguish between
                      wavecal and science obervation
        suffix      just "_rawtag" or "_rawimage" (i.e. excluding "_a" or
                      "_b" if the data were taken with the FUV detector)
        """

        self.input = input              # name of a raw input file
        self.exp_type = EXP_SCIENCE     # science, wavecal, target acq
        self.filenames = {}             # input and output file names
        self.info = {}                  # detector, opt_elem, etc.
        self.switches = {}              # calibration switch values
        self.reffiles = {}              # reference file names

        self.getHeaderInfo()
        self.filenames = self.makeFileNames (suffix, outdir)
        self.sanityCheck()

        # Determine what type of observation this is.

        # initial value based on keyword exptype
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

        # memtype will be "none" if there is no association table.
        if memtype != "none":
            if memtype.find ("WAVE") >= 0:
                self.exp_type = EXP_WAVECAL
                if self.info["exptype"].find ("WAVE") < 0:
                    cosutil.printWarning ("EXPTYPE = %s in %s," \
                                % (self.info["exptype"], input))
                    cosutil.printContinuation (
                            "but MEMTYPE = %s in the association table;" \
                                % memtype)
                    cosutil.printContinuation (
                    "this will be processed as a wavecal because of MEMTYPE.")

            elif memtype.find ("ACQ") >= 0 or memtype.find ("PKP") >= 0:
                self.exp_type = EXP_TARGET_ACQ

            else:
                self.exp_type = EXP_SCIENCE
                if self.info["exptype"].find ("WAVE") >= 0:
                    cosutil.printWarning ("EXPTYPE = %s in %s," \
                                % (self.info["exptype"], input))
                    cosutil.printContinuation (
                            "but MEMTYPE = %s in the association table;" \
                                % memtype)
                    cosutil.printContinuation (
        "this will be processed as a science observation because of MEMTYPE.")

        if self.exp_type == EXP_WAVECAL and self.info["aperture"] != "WCA":
            cosutil.printWarning (
            "APERTURE = %s for a wavecal; this could be a serious error" \
                                % self.info["aperture"])

        self.checkSwitches()

    def getHeaderInfo (self):
        """Read keyword values.

        This routine gets general info from both the primary and EVENTS or SCI
        extension headers, and it gets calibration switches and reference
        file names from the primary header.
        """
        fd = pyfits.open (self.input, mode="readonly")
        phdr = fd[0].header
        try:
            hdr = fd["EVENTS"].header
        except:
            hdr = fd["SCI"].header

        # Each of these is a dictionary with (lower case) header keywords
        # as the keys.
        self.info = getinfo.getGeneralInfo (phdr, hdr)
        self.switches = getinfo.getSwitchValues (phdr)
        self.reffiles = getinfo.getRefFileNames (phdr)

        self.info["cal_ver"] = CALCOS_VERSION

        fd.close()

    def sanityCheck (self):
        """Check some keywords to make sure they're reasonable.

        For thermal vac data, this also updates opt_elem, cenwave and fpoffset
        in the info dictionary, if necessary.  Other keywords that may be
        reset are obstype, exptype and dispaxis.
        """

        info = self.info
        warn = 0                # initial values
        bad = 0

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
           info["opt_elem"][0:3] == "TA1":
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
            if info["exptype"] == "WAVECAL":
                bad = 1
                cosutil.printError (
                "OBSTYPE = IMAGING and EXPTYPE = WAVECAL is invalid")
        elif info["obstype"] == "SPECTROSCOPIC":
            if info["dispaxis"] == 1:
                if info["detector"] == "NUV":
                    warn = 1
                    cosutil.printWarning ("DISPAXIS = 1 for NUV data")
            elif info["dispaxis"] == 2:
                if info["detector"] == "FUV":
                    warn = 1
                    cosutil.printWarning ("DISPAXIS = 2 for FUV data")
            else:
                bad = 1
                cosutil.printError (
            "DISPAXIS = %d, but it should be 1 or 2 for spectroscopic data" \
                         % info["dispaxis"])
        else:
            bad = 1
            cosutil.printError (
            "OBSTYPE = `%s'; should be IMAGING or SPECTROSCOPIC" \
                         % info["obstype"])

        # check for TAGFLASH with conventional wavecal
        if info["tagflash"] and info["exptype"] == "WAVECAL":
            warn = 1
            cosutil.printWarning (
            "EXPTYPE = WAVECAL but TAGFLASH = T; TAGFLASH will be reset to F")
            info["tagflash"] = 0

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
        """Replace RelMvReq in keywords with values based on OSM step position.

        For thermal vac data (only), this function determines the values of
        OPT_ELEM, CENWAVE and FPOFFSET based on the OSM1 or OSM2 step positions
        as given by LOM1STP or LOM2STP respectively, in the support file header.
        If OPT_ELEM is "RelMvReq", then OPT_ELEM, CENWAVE and FPOFFSET will be
        silently replaced by the correct values.  If CENWAVE is unreasonably
        small (< 1000), it will be replaced.  Otherwise, these three keywords
        will be compared with the values determined from the OSM step positions,
        and discrepancies will be noted and corrected.

        arguments:
        sptfile           i: name of support file
        info              io: dictionary of keywords and values; values may be
                              updated by this function
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

        arguments:
        info              io: dictionary of keywords and values
        opt_elem_osm      i: value of OPM_ELEM as determined from OSM position
        cenwave_osm       i: value of CENWAVE as determined from OSM position
        fpoffset_osm      i: value of FPOFFSET as determined from OSM position
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

        'suffix' is an obsmode-specific string, either
        "_rawtag" or "_rawimage"; note that 'suffix' excludes "_a" or "_b",
        in the case that we have FUV data.  'outdir' is either an empty string
        or the name of the output directory.  A dictionary of the input and
        output names will be constructed and returned.  These are the keys
        for each such dictionary:

          root     rootname (not including suffix or directory); note that this
                     is from the file name, not the header keyword
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
                     segment (or all 3 NUV stripes)
          flash    output 1-D extracted tagflash wavecal spectrum (the file
                     that includes all segments or stripes)
        """

        input = os.path.basename (self.input)
        # This is the input file name, but in the output directory.
        output = os.path.join (outdir, input)

        rootname = getRootname (input, "_raw")

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
        filenames["raw"]     = self.input
        filenames["pha"]     = replaceSuffix (self.input, suffix, "_pha")
        filenames["corrtag"] = replaceSuffix (output, suffix, "_corrtag")
        filenames["flt"]     = replaceSuffix (output, suffix, "_flt")
        filenames["counts"]  = replaceSuffix (output, suffix, "_counts")
        filenames["x1d_x"]   = x1d_x
        filenames["x1d"]     = x1d
        filenames["flash_x"] = flash_x
        filenames["flash"]   = flash

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

        if self.info["obstype"] == "IMAGING" or \
           self.exp_type == EXP_TARGET_ACQ or \
           self.exp_type == EXP_ACQ_IMAGE:

            self.overrideSwitch ("doppcorr", messages)
            self.overrideSwitch ("helcorr", messages)
            self.overrideSwitch ("x1dcorr", messages)
            self.overrideSwitch ("wavecorr", messages)
            self.overrideSwitch ("backcorr", messages)
            self.overrideSwitch ("fluxcorr", messages)

        else:                                   # spectroscopic

            if self.exp_type == EXP_WAVECAL:
                # Silently set these switches.
                self.switches["wavecorr"] = "PERFORM"
                self.switches["x1dcorr"] = "PERFORM"
                self.switches["doppcorr"] = "OMIT"
                self.switches["helcorr"] = "OMIT"
                self.switches["fluxcorr"] = "OMIT"

            if self.info["doppmag"] == 0.:
                self.overrideSwitch ("doppcorr", messages)
            if self.info["ra_targ"] < 0.:
                self.overrideSwitch ("helcorr", messages)

            if self.switches["x1dcorr"] != "PERFORM":
                # Can't do backcorr or fluxcorr without 1-D extraction.
                self.overrideSwitch ("backcorr", messages)
                self.overrideSwitch ("fluxcorr", messages)

        return messages

    def overrideSwitch (self, keyword, messages):
        """If switch for keyword is "PERFORM", reset it to "OMIT"."""

        key_lower = keyword.lower()
        if self.switches.has_key (key_lower):
            if self.switches[key_lower] == "PERFORM":
                self.switches[key_lower] = "OMIT"
                messages.append (keyword.upper() + " reset to OMIT")
        else:
            self.switches[key_lower] = "OMIT"

    def printSwitchMessages (self, messages, input):
        """Print info about which calibration switches are being reset."""

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

    def __init__ (self, input, outdir, memtype):

        Observation.__init__ (self, input, outdir, memtype, "_rawtag")

    def checkSwitches (self):

        messages = self.checkImSpecSwitches()

        self.printSwitchMessages (messages, self.input)

class FUVAccumObs (Observation):

    def __init__ (self, input, outdir, memtype):

        Observation.__init__ (self, input, outdir, memtype, "_rawimage")

    def checkSwitches (self):

        messages = self.checkImSpecSwitches()

        # Note that this tests on DOPPON, while the generic test in
        # checkImSpecSwitches uses DOPPMAG.
        if not self.info["doppon"]:
            self.overrideSwitch ("doppcorr", messages)

        self.printSwitchMessages (messages, self.input)

class NUVTimetagObs (Observation):

    def __init__ (self, input, outdir, memtype):

        Observation.__init__ (self, input, outdir, memtype, "_rawtag")

    def checkSwitches (self):

        messages = self.checkImSpecSwitches()

        self.overrideSwitch ("tempcorr", messages)
        self.overrideSwitch ("geocorr", messages)
        self.overrideSwitch ("igeocorr", messages)
        self.overrideSwitch ("randcorr", messages)
        self.overrideSwitch ("phacorr", messages)

        self.printSwitchMessages (messages, self.input)

class NUVAccumObs (Observation):

    def __init__ (self, input, outdir, memtype):

        Observation.__init__ (self, input, outdir, memtype, "_rawimage")

    def checkSwitches (self):

        messages = self.checkImSpecSwitches()

        self.overrideSwitch ("tempcorr", messages)
        self.overrideSwitch ("geocorr", messages)
        self.overrideSwitch ("igeocorr", messages)
        self.overrideSwitch ("randcorr", messages)
        self.overrideSwitch ("phacorr", messages)
        if not self.info["doppon"]:
            self.overrideSwitch ("doppcorr", messages)

        self.printSwitchMessages (messages, self.input)

class Calibration:
    """Calibrate COS data.

    The attributes are:
        assoc              the Association instance
        wavecal_info       list of lists, each of which contains the
                             following:
                               time (MJD of middle of exposure)
                               fpoffset (header keyword fpoffset)
                               shift dictionary:  keys are pshifta, pshiftb,
                                 and (if NUV) pshiftc; value is the shift
                                 that was determined, in pixels; positive
                                 shift means that features in the spectrum
                                 were found at larger pixel number than the
                                 nominal location
                               rootname
        wcp_info           matching row (just one) from the wavecal
                             parameters table
    """

    def __init__ (self, assoc):

        self.assoc = assoc
        self.wavecal_info = []
        self.wcp_info = None

    def basicCal (self, filenames, info, switches, reffiles):

        input = filenames["raw"]
        inpha = filenames["pha"]
        output = filenames["flt"]
        outtag = filenames["corrtag"]
        outcounts = filenames["counts"]
        if info["tagflash"]:
            outflash = filenames["flash_x"]
        else:
            outflash = None
        if info["obsmode"] == "TIME-TAG":
            status = timetag.timetagBasicCalibration (input, outtag,
                        output, outcounts, outflash,
                        info, switches, reffiles,
                        self.assoc.stimfile, self.assoc.livetimefile,
                        self.assoc.burstfile)
        else:
            status = accum.accumBasicCalibration (input, inpha,
                        output, outcounts,
                        info, switches, reffiles,
                        self.assoc.stimfile, self.assoc.livetimefile)

    def allWavecals (self):
        """Process all the wavecal observations in the association."""

        if self.assoc.global_switches["wavecal"] != "PERFORM":
            return

        cosutil.printMsg ("Begin calibration of wavecals.", VERY_VERBOSE)

        # First calibrate all the wavecals.
        for obs in self.assoc.obs:
            if obs.exp_type == EXP_WAVECAL:
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
                (shift2, xd_shifts, xd_locns) = wavecal.findWavecalSpectrum \
                        (obs.filenames["corrtag"], obs.info, obs.reffiles)
                # Update shift2[a-c] keywords.
                self.setSpectrumOffset (obs.filenames,
                        obs.info["segment"], shift2)
                self.extractSpectrum (obs.filenames)
                if not self.assoc.save_temp_files:
                    self.removeCountRateFile (obs.filenames)

        self.concatenateSpectra ("wavecal")

        # Now find the shift of each wavecal.
        self.processWavecal()

        # Set the shift keywords in the corrtag, flt, and counts headers
        # (already set in x1d header) for each wavecal observation.
        for obs in self.assoc.obs:
            if obs.exp_type == EXP_WAVECAL:
                self.setWavecalShift (obs.filenames)

        cosutil.printMsg ("wavecal_info = " + repr (self.wavecal_info),
                VERY_VERBOSE)

    def allScience (self):
        """Process all the science observations in the association."""

        if self.assoc.global_switches["science"] != "PERFORM":
            return

        cosutil.printMsg ("Begin calibration of science data.", VERY_VERBOSE)
        tagflash = False                # initial value
        for obs in self.assoc.obs:
            if obs.exp_type == EXP_SCIENCE or obs.exp_type == EXP_ACQ_IMAGE:
                self.basicCal (obs.filenames,
                        obs.info, obs.switches, obs.reffiles)
                self.updateShift (obs.filenames, obs.switches["wavecorr"],
                            obs.info)
                if obs.switches["x1dcorr"] == "PERFORM":
                    self.extractSpectrum (obs.filenames)
                elif obs.info["obstype"] == "SPECTROSCOPIC":
                    cosutil.printSwitch ("X1DCORR", obs.switches)
                if not self.assoc.save_temp_files:
                    self.removeCountRateFile (obs.filenames)
                if obs.info["tagflash"]:
                    tagflash = True     # there is at least one tagflash obs

        if obs.switches["x1dcorr"] == "PERFORM":
            self.concatenateSpectra ("science")

        if tagflash:
            self.concatenateSpectra ("tagflash")

    def extractSpectrum (self, filenames):
        """Extract a 1-D spectrum from 2-D images."""

        input = filenames["flt"]
        incounts = filenames["counts"]
        output = filenames["x1d_x"]
        extract.extract1D (input, incounts, output)

    def removeCountRateFile (self, filenames):
        """Delete the count rate file (_counts)."""

        cosutil.printMsg ("Remove the count rate file(s):", VERY_VERBOSE)
        outcounts = filenames["counts"]
        if os.access (outcounts, os.R_OK):
            cosutil.printMsg ("  remove " + outcounts, VERY_VERBOSE)
            os.remove (outcounts)

    def processWavecal (self):
        """Determine shift from wavecal observation.

        The shift and related info will be appended to the wavecal_info list.
        """
        cosutil.printSwitch ("WAVECORR", {"wavecorr": "PERFORM"})
        reffiles = self.assoc.obs[self.assoc.first_science].reffiles
        wavecal.printWavecalRef (reffiles)

        previous_x1d_file = " "
        for obs in self.assoc.obs:

            x1d_file = obs.filenames["x1d"]
            # For FUV, we expect duplicate x1d file names in the obs list.
            if x1d_file == previous_x1d_file:
                continue
            previous_x1d_file = x1d_file

            if obs.exp_type == EXP_WAVECAL:

                shift_dict = wavecal.findWavecalShift (x1d_file, self.wcp_info)

                if shift_dict is not None:
                    # time is the MJD at the midpoint of the exposure.
                    time = cosutil.timeAtMidpoint (obs.info)
                    wavecal.storeWavecalInfo (self.wavecal_info,
                            time, obs.info["fpoffset"],
                            shift_dict, obs.filenames["root"])

    def updateShift (self, filenames, wavecorr, info):
        """Update the shift keywords in corrtag, flt, counts headers.

        The shift for the two segments (or three NUV stripes) will be copied
        (or interpolated) from the list of wavecal information to the
        keywords PSHIFTA and PSHIFTB (and PSHIFTC if NUV).
        """

        if info["obstype"] != "SPECTROSCOPIC":
            return
        if info["tagflash"]:
            return

        if wavecorr != "PERFORM":
            shift_dict = None
        elif len (self.wavecal_info) < 1:
            shift_dict = None
        else:
            time = cosutil.timeAtMidpoint (info)        # MJD
            shift_dict = wavecal.returnWavecalShift (self.wavecal_info,
                    self.wcp_info, info["fpoffset"], time)
            if wavecorr == "PERFORM" and len (self.wavecal_info) > 0:
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

        for fname in [filenames["corrtag"], \
                      filenames["flt"], filenames["counts"]]:
            if os.access (fname, os.R_OK):
                fd = pyfits.open (fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd["SCI"].header
                if wavecorr == "PERFORM" and len (self.wavecal_info) > 0:
                    phdr.update ("WAVECORR", "COMPLETE")
                if shift_dict is None:
                    hdr.update ("PSHIFTA", 0.)
                    hdr.update ("PSHIFTB", 0.)
                    hdr.update ("SHIFT2A", 0.)
                    hdr.update ("SHIFT2B", 0.)
                    if info["detector"] == "NUV":
                        hdr.update ("PSHIFTC", 0.)
                        hdr.update ("SHIFT2C", 0.)
                else:
                    for key in shift_dict.keys():
                        shift = shift_dict[key]
                        hdr.update (key, shift)
                fd.close()

    def setSpectrumOffset (self, filenames, segment, shift2):
        """Update the shift2 keywords in corrtag, flt, counts headers.

        This function is called only for a wavecal, not for a science
        observation.  (For science data, the shift2 keyword(s) will be
        updated by updateShift.)
        """

        if segment[0:3] == "FUV":
            keywords = ["SHIFT2"+segment[-1]]
        else:
            keywords = ["SHIFT2A", "SHIFT2B", "SHIFT2C"]
        for fname in [filenames["corrtag"], \
                      filenames["flt"], filenames["counts"]]:
            if os.access (fname, os.R_OK):
                fd = pyfits.open (fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd["SCI"].header
                for keyword in keywords:
                    hdr.update (keyword, shift2)
                fd.close()

    def setWavecalShift (self, filenames):
        """Update the shift keywords in corrtag, flt, counts headers.

        This function is called only for a wavecal, not for a science
        observation.  There must be an exact match with the rootname of the
        observation.
        """

        shift_dict = wavecal.returnExactMatch (self.wavecal_info,
                             filenames["root"])

        for fname in [filenames["corrtag"], \
                      filenames["flt"], filenames["counts"]]:
            if os.access (fname, os.R_OK):
                fd = pyfits.open (fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd["SCI"].header
                phdr.update ("WAVECORR", "COMPLETE")
                for keyword in shift_dict.keys():
                    shift = shift_dict[keyword]
                    hdr.update (keyword, shift)
                fd.close()

    def concatenateSpectra (self, type):
        """Concatenate two 1-D FUV spectra into one spectrum.

        argument:
        type          "science", "wavecal" or "tagflash" (ignore if "unknown")

        If type="wavecal", this routine will concatenate pairs of wavecal
        files; if type="science", this routine will concatenate pairs of
        science files.  The input _x1d_a and _x1d_b will then be deleted,
        if save_temp_files = False.
        """

        for one_set in self.assoc.concat:

            if one_set["type"] == type:

                infiles = one_set["input"]
                output = one_set["output"]
                if len (infiles) < 1:
                    continue

                if len (infiles) == 1:
                    if infiles[0] != output:
                        cosutil.renameFile (infiles[0], output)
                else:
                    extract.concatenateFUVSegments (infiles, output)
                    if not self.assoc.save_temp_files:
                        # Delete the _x1d_a.fits and _x1d_b.fits files.
                        for file in infiles:
                            if os.access (file, os.R_OK):
                                os.remove (file)

    def combineToProduct (self):
        """Average the calibrated files, producing the product files."""

        if self.assoc.global_switches["repeat"] != "PERFORM":
            return

        combine = self.assoc.combine

        # If we have repeatobs data rather than FP-pos, average the flt files.
        if self.assoc.product_type.find ("RPT") >= 0:
            self.combineFlt()

        i = self.assoc.first_science
        if self.assoc.obs[i].switches["x1dcorr"] != "PERFORM":
            return

        self.combineAllX1D()

        # If we have a combination of FP-pos and repeatobs data, average
        # multiple x1d files that have the same OSM index.  If all of them
        # have the same OSM index, however, skip this step.
        if combine.has_key ("x1d"):
            x1d_list = combine["x1d"]
            osm_list = combine["fppos"]
            do_subsets = False                  # initial value
            for i in range (1, len (osm_list)):
                if osm_list[i] != osm_list[0]:
                    do_subsets = True
                    break
            if do_subsets:
                for osm in range (1, 5):
                    # extract subset for current osm position
                    x1d_subset = []
                    for i in range (len (x1d_list)):
                        if osm_list[i] == osm:
                            x1d_subset.append (x1d_list[i])
                    if len (x1d_subset) > 1:
                        self.combineX1Di (x1d_subset, osm)

    def combineFlt (self):
        """Average flat fielded repeatobs data."""

        combine = self.assoc.combine

        if combine.has_key ("flt_a"):
            output = self.fltProductName (combine["flt_a"])
            average.avgImage (combine["flt_a"], output)

        if combine.has_key ("flt_b"):
            output = self.fltProductName (combine["flt_b"])
            average.avgImage (combine["flt_b"], output)

        if combine.has_key ("flt"):
            output = self.fltProductName (combine["flt"])
            average.avgImage (combine["flt"], output)

    def combineAllX1D (self):
        """Average x1d data for all OSM positions."""

        combine = self.assoc.combine

        if combine.has_key ("x1d"):
            output = self.x1dProductName (combine["x1d"], 0)
            fpavg.fpAvgSpec (combine["x1d"], output)

    def combineX1Di (self, input, osm):
        """Average x1d data for one specified OSM position."""

        output = self.x1dProductName (input, osm)

        fpavg.fpAvgSpec (input, output)

    def fltProductName (self, input):
        """Construct the product name for the flt file."""

        multiple = len (input) > 1

        first = input[0]

        if first.endswith ("flt.fits"):
            if multiple:
                output = self.assoc.product + "_fltsum.fits"
            else:
                output = self.assoc.product + "_flt.fits"

        elif first.endswith ("flt_a.fits"):
            if multiple:
                output = self.assoc.product + "_fltsum_a.fits"
            else:
                output = self.assoc.product + "_flt_a.fits"

        elif first.endswith ("flt_b.fits"):
            if multiple:
                output = self.assoc.product + "_fltsum_b.fits"
            else:
                output = self.assoc.product + "_flt_b.fits"

        else:
            raise RuntimeError, "Internal error"

        return output

    def x1dProductName (self, input, osm):
        """Construct the product name for the x1d file.

        If there are multiple files to be combined (i.e. the length of
        'input' is greater than one), and if osm is greater than zero, then
        the output file name will be of the form "rootname_x1dsum1.fits",
        where the number appended to "x1dsum" will be the value of osm.

        arguments:
        input         list of file names (we only need this to check its
                        length)
        osm           OSM index (0 or 1-4)
        """

        if len (input) > 1:
            output = self.assoc.product + "_x1dsum"
            if osm > 0:
                output += str (osm) + ".fits"
            else:
                output += ".fits"
        else:
            output = self.assoc.product + "_x1d.fits"

        return output

if __name__ == "__main__":

    main (sys.argv[1:])
