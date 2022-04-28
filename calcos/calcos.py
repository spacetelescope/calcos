#! /usr/bin/env python

from __future__ import absolute_import, division, print_function # confidence high
import sys
import os
import time
import getopt
import glob
import copy

import numpy
import astropy
import astropy.io.fits as fits
from . import accum
from . import average
from . import cosutil
from . import extract
from . import fpavg
from . import getinfo
from . import shiftfile
from . import spwcs
from . import timetag
from . import trace
from . import wavecal
from .calcosparam import *       # parameter definitions

# These values for Observation.exp_type are used in this file only.
EXP_UNKNOWN     = 0
EXP_SCIENCE     = 1
EXP_WAVECAL     = 2
EXP_CALIBRATION = 3     # tagflash, but exptype not EXTERNAL/SCI
EXP_TARGET_ACQ  = 4
EXP_ACQ_IMAGE   = 5
EXP_ENGINEERING = 6

# calcos will return this if there are no files to calibrate (e.g. rawacq but
# not acq/image).
NO_DATA_TO_CALIBRATE = 5

# calcos can return this value if the APERTURE keyword is not valid or
# if a required row is missing from a reference table.
BAD_APER_MISSING_ROW_EXCEPTION = 16

# If the input is a raw file rather than an association file, this flag
# will be set to True.
raw_input_trailer = False

def main(args=sys.argv[1:]):
    """Check arguments and call calcos.

    This driver interprets command-line arguments and calls calcos for
    each association file or raw file specified on the command line.

    The command-line options are:
        -q (quiet)
        -v (very verbose)
        -s (save temporary files)
        -o outdir (output directory name)
        -r (print version (revision) string and exit)
        --version (print the version number and exit)
        --find yes|no|cutoff (find Y location of spectrum)
        --csum (create csum image)
        --raw (use raw coordinates for csum image)
        --only_csum (create csum image, and do almost nothing else)
        --compress parameters (compress csum image;
                the default value for parameters is 'gzip,-0.01')
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

    if len(args) < 1:
        prtOptions()
        cosutil.printError(
        "An association file name or observation rootname must be specified.")
        sys.exit()

    try:
        (options, pargs) = getopt.getopt(args, "qvrso:",
                           ["find=",
                            "version",
                            "csum", "raw", "only_csum",
                            "compress=", "binx=", "biny=",
                            "shift=", "stim=", "live=", "burst="])
    except Exception as error:
        prtOptions()
        cosutil.printError(str(error))
        sys.exit()

    if len(options) == 0:
        for i in range(len(pargs)):
            if pargs[i][0] == '-':
                prtOptions()
                cosutil.printError(
                "Command-line options must precede the association file name.")
                sys.exit()

    # default values
    cosutil.setVerbosity(VERBOSE)
    # parameters pertaining to the "calcos sum" file
    create_csum_image = False
    raw_csum_coords = False
    only_csum = False
    binx = 1
    biny = 1
    find_target = {"flag": False, "cutoff": None}
    compress_csum = False
    compression_parameters = "gzip,-0.01"
    # user-supplied text file to specify shift1 and shift2
    shift_file = None
    save_temp_files = False
    stimfile = None
    livetimefile = None
    burstfile = None
    outdir = None

    for i in range(len(options)):
        if options[i][0] == "--version":
            print("%s" % CALCOS_VERSION_NUMBER)
            sys.exit(0)
        if options[i][0] == "-r":
            print("%s" % CALCOS_VERSION)
            sys.exit(0)
        if options[i][0] == "-q":
            cosutil.setVerbosity(QUIET)
        elif options[i][0] == "-v":
            cosutil.setVerbosity(VERY_VERBOSE)
        elif options[i][0] == "-s":
            save_temp_files = True
        elif options[i][0] == "-o":
            outdir = options[i][1]
        elif options[i][0] == "--find":
            temp = options[i][1].lower()
            if temp == "yes" or temp == "true":
                find_target["flag"] = True
            elif temp == "no" or temp == "false":
                find_target["flag"] = False
            else:
                try:
                    cutoff = float(temp)
                except ValueError:
                    prtOptions()
                    cosutil.printError("Don't understand '--find %s'" %
                                       options[i][1])
                    sys.exit()
                if cutoff < 0.:
                    prtOptions()
                    cosutil.printError("Cutoff for --find cannot be negative.")
                    sys.exit()
                find_target["flag"] = True
                find_target["cutoff"] = cutoff
        elif options[i][0] == "--nofind":
            find_target = False
        elif options[i][0] == "--csum":
            create_csum_image = True
        elif options[i][0] == "--raw":
            raw_csum_coords = True
        elif options[i][0] == "--only_csum":
            only_csum = True
        elif options[i][0] == "--compress":
            compress_csum = True
            compression_parameters = options[i][1]
        elif options[i][0] == "--binx":
            binx = int(options[i][1])
        elif options[i][0] == "--biny":
            biny = int(options[i][1])
        elif options[i][0] == "--shift":
            shift_file = options[i][1]
        elif options[i][0] == "--stim":
            stimfile = options[i][1]
        elif options[i][0] == "--live":
            livetimefile = options[i][1]
        elif options[i][0] == "--burst":
            burstfile = options[i][1]

    if only_csum:
        create_csum_image = True
        shift_file = None
        stimfile = None
        livetimefile = None
        burstfile = None
    if raw_csum_coords:
        if not create_csum_image:
            cosutil.printWarning("--raw will be ignored because "
                                 "--csum was not specified")
            raw_csum_coords = False
        else:
            raw_csum_coords = True

    infiles = uniqueInput(pargs)        # remove duplicate names from list

    status = 0
    for i in range(len(infiles)):
        stat = calcos(infiles[i], outdir=outdir, verbosity=None,
                      find_target=find_target,
                      create_csum_image=create_csum_image,
                      raw_csum_coords=raw_csum_coords,
                      only_csum=only_csum,
                      binx=binx, biny=biny,
                      compress_csum=compress_csum,
                      compression_parameters=compression_parameters,
                      shift_file=shift_file,
                      save_temp_files=save_temp_files,
                      stimfile=stimfile, livetimefile=livetimefile,
                      burstfile=burstfile)
        status |= stat
    if status != 0:
        sys.exit(status)

def prtOptions():
    """Print a list of command-line options and arguments."""

    cosutil.printMsg("The command-line options are:")
    cosutil.printMsg("  --version (print the version number and exit)")
    cosutil.printMsg("  -r (print the full version string and exit)")
    cosutil.printMsg("  -q (quiet)")
    cosutil.printMsg("  -v (very verbose)")
    cosutil.printMsg("  -s (save temporary files)")
    cosutil.printMsg("  -o outdir (output directory name)")
    cosutil.printMsg("  --find yes (find Y location of spectrum)")
    cosutil.printMsg("  --find no (use Y location of spectrum "
                     "from 1dx file and wavecal)")
    cosutil.printMsg("  --find cutoff (find Y location if sigma <= cutoff)")
    cosutil.printMsg("  --csum (create 'calcos sum' image)")
    cosutil.printMsg("  --only_csum (do little else but create csum)")
    cosutil.printMsg("  --raw (use raw coordinates for csum image)")
    cosutil.printMsg("  --compress parameters (compress csum image)")
    cosutil.printMsg("  --binx X_bin_factor (csum bin factor in X)")
    cosutil.printMsg("  --biny Y_bin_factor (csum bin factor in Y)")
    cosutil.printMsg("  --shift filename (file to specify shift values)")
    cosutil.printMsg("  --stim filename (append stim locations to filename)")
    cosutil.printMsg("  --live filename (append livetime factors to filename)")
    cosutil.printMsg("  --burst filename (append burst info to filename)")
    cosutil.printMsg("")
    cosutil.printMsg("Following the options, list one or more association")
    cosutil.printMsg("files (rootname_asn) or raw files (rootname_raw).")

def uniqueInput(infiles):
    """Remove effective duplicates from list of files to process.

    This function also expands environment variables and wildcards.
    Aside from that, the order of the input file names will be preserved.

    Parameters
    ----------
    infiles: list of strings
        List of input file names.

    Returns
    -------
    unique_files: list of strings
        The list of input files but with duplicates removed.
    """

    MAX_COUNT = 100
    # expand environment variables and wildcards
    allfiles = []
    for file in infiles:
        for i in range(MAX_COUNT):
            template = os.path.expandvars(file)
            if template == file:
                break
            file = template
        files = glob.glob(template)
        files.sort()
        allfiles.extend(files)

    if len(allfiles) <= 1:
        return allfiles

    inlist = copy.copy(allfiles)
    inlist.sort()

    newlist = [inlist[0]]
    for i in range(1, len(inlist)):
        n = len(inlist[i])
        if inlist[i].endswith("_a.fits"):
            n -= 7
        elif inlist[i].endswith("_b.fits"):
            n -= 7
        elif inlist[i].endswith(".fits"):
            n -= 5
        if inlist[i][:n] != inlist[i-1][:n]:
            newlist.append(inlist[i])

    unique_files = []
    for input in allfiles:
        if input in newlist and \
           input not in unique_files:
            unique_files.append(input)

    return unique_files

def calcos(asntable, outdir=None, verbosity=None,
           find_target={"flag": False, "cutoff": None},
           create_csum_image=False,
           raw_csum_coords=False,
           only_csum=False,
           binx=None, biny=None,
           compress_csum=False, compression_parameters="gzip,-0.01",
           shift_file=None,
           save_temp_files=False,
           stimfile=None, livetimefile=None, burstfile=None):
    """Calibrate COS data.

    This is the main module for calibrating COS data.

    Parameters
    ----------
    asntable: str
        The rootname (with "_asn") of an association file, or the rootname
        (with "_raw") of a raw file.  If the value of a raw FUV file is
        specified and files for both segments are present, then both of
        those files will be calibrated (i.e. without having to explicitly
        list both files).

    Returns
    -------
    status: int
        0 is OK; 5 means no file was found that could be calibrated.

    Other parameters
    ----------------
    outdir: str or None, optional
        Name of output directory.

    verbosity: int {0, 1, 2} or None, optional
        If not None, set verbosity to this level.

    find_target: dictionary, optional
        Keys are "flag" and "cutoff".  flag = True means use the location
        of the target in the cross-dispersion direction if the standard
        deviation (pixels) of the location is less than or equal to cutoff
        (if cutoff is positive).  flag = False means use the location
        determined from the wavecal.

    create_csum_image: boolean, optional
        If True, write an image that reflects the counts detected at each
        pixel (includes deadcorr but not flatcorr), for OPUS to add to the
        cumulative image.

    raw_csum_coords: boolean, optional
        If True, use raw pixel coordinates (rather than thermally and
        geometrically corrected) to create the csum image.

    only_csum: boolean, optional
        If True, create a csum image, but most other files will not be
        written.

    binx, biny: int or None, optional
        Binning factor for the X and Y axes, or None, which means that
        the default binning (currently 1) should be used.

    compress_csum: boolean, optional
        If True, compress the "calcos sum" image.

    compression_parameters: string, optional
        Two values separated by a comma; the first is the compression type
        (rice, gzip or hcompress), and the second is the quantization
        level.  The default is "gzip,-0.01".

    shift_file: str, optional
        If specified, this text file contains values of shift1 (and
        possibly shift2) to override the values found via wavecal
        processing.

    save_temp_files: boolean, optional
        By default, the _x1d_a.fits and _x1d_b.fits files (if FUV) will
        be deleted after concatenating to the _x1d.fits file.  Specify
        save_temp_files=True to keep these files.

    stimfile: str, optional
        If specified, the stim positions will be written to (or
        appended to) a text file with this name.

    livetimefile: str, optional
        If specified, the livetime factors will be written to (or
        appended to) a text file with this name.

    burstfile: str, optional
        If specified, burst information will be written to (or appended to)
        a text file with this name.
    """

    t0 = time.time()

    # Create the output directory if it was specified and doesn't exist.
    if outdir:
        outdir = os.path.expandvars(outdir)
    createOutputDirectory(outdir)

    # If asntable is a raw file, open a trailer for it.
    openTrailerForRawInput(asntable, outdir)

    cosutil.printMsg("CALCOS version " + CALCOS_VERSION)
    cosutil.printMsg("numpy version " + numpy.__version__)
    cosutil.printMsg("astropy version " + astropy.__version__)
    cosutil.printMsg("Begin " + cosutil.returnTime(), VERBOSE)

    if verbosity is not None:
        cosutil.setVerbosity(verbosity)

    # some of the command-line arguments
    cl_args = {"find_target": find_target,
               "create_csum_image": create_csum_image,
               "raw_csum_coords": raw_csum_coords,
               "only_csum": only_csum,
               "binx": binx,
               "biny": biny,
               "compress_csum": compress_csum,
               "compression_parameters": compression_parameters,
               "shift_file": shift_file,
               "save_temp_files": save_temp_files,
               "stimfile": stimfile,
               "livetimefile": livetimefile,
               "burstfile": burstfile}

    assoc = Association(asntable, outdir, cl_args)
    if len(assoc.obs) == 0:
        return NO_DATA_TO_CALIBRATE
    if not assoc.isAnySwitchSet():
        cosutil.printMsg("Nothing to do; all calibration switches are OMIT.")
        return 0

    cal = Calibration(assoc)

    wav_status = cal.allWavecals()
    sci_status = cal.allScience()
    if sci_status:                      # bad value for aperture keyword
        return sci_status
    elif wav_status:
        return wav_status

    cal.mergeKeywords()
    cal.combineToProduct()

    assoc.updateMempresent()
    assoc.copySptFile()

    cosutil.printMsg("End   " + cosutil.returnTime(), VERBOSE)

    t1 = time.time()
    cosutil.printMsg("elapsed time = %.1f sec. = %.2f min." % \
                     (t1-t0, (t1-t0)/60.), VERY_VERBOSE)
    closeTrailerForRawInput()

    return 0

def createOutputDirectory(outdir):
    """Check whether outdir exists, and create it if necessary.

    If outdir was specified but doesn't exist, create it.

    Parameters
    ----------
    outdir: str or None
        Name of output directory.
    """

    if outdir:
        full_outdir = expandDirectory(outdir)
        if os.path.lexists(full_outdir):
            if not os.path.isdir(full_outdir):
                raise RuntimeError("'%s' is a file; should be a directory." %
                                   outdir)
        else:
            cosutil.printWarning("Creating output directory '%s'." % outdir)
            os.mkdir(full_outdir)

def openTrailerForRawInput(input, outdir):
    """Open the trailer file for this file."""

    global raw_input_trailer

    if input.endswith("_asn") or input.endswith("_asn.fits"):
        return
    if raw_input_trailer:               # already open?
        return

    input = os.path.expandvars(input)
    input = os.path.basename(input)
    rootname = getRootname(input, "_raw")
    if outdir:
        outdir = expandDirectory(outdir)
        trailer = os.path.join(outdir, rootname) + ".tra"
    else:
        trailer = rootname + ".tra"

    cosutil.openTrailer(trailer)
    raw_input_trailer = True

def closeTrailerForRawInput():
    """Close the trailer file for this file."""

    global raw_input_trailer

    cosutil.closeTrailer()
    raw_input_trailer = False

def expandDirectory(dirname):
    """Get the real directory name.

    Parameters
    ----------
    dirname: str
        A directory name.

    Returns
    -------
    directory_name: str
        The real directory name.
    """

    indir = dirname
    done = False
    count = 0
    MAX_COUNT = 100
    while not done:
        temp = os.path.expandvars(indir)        # $stuff/dir
        count += 1
        if temp == indir:
            done = True
        indir = temp
        if count >= MAX_COUNT:
            break
    if not done:
        cosutil.printWarning("%d iterations exceeded while expanding " \
        "variables in directory %s" % (MAX_COUNT, dirname))
    indir = os.path.abspath(indir)              # ../dir
    indir = os.path.expanduser(indir)           # ~/dir
    directory_name = os.path.normpath(indir)    # remove redundant strings

    return directory_name

def replaceSuffix(rawname, suffix, new_suffix):
    """Replace the suffix in a raw file name.

    Parameters
    ----------
    rawname: str
        Name of a raw input file.

    suffix: str
        Suffix (last part of root name) that is expected to be found in
        the raw file name.

    new_suffix: str
        String to replace suffix to create an output file name.

    Returns
    -------
    newname: str
        rawname with suffix replaced by new_suffix.

    Examples
    --------
    >>> print replaceSuffix("rootname_rawtag.fits", "_rawtag", "_flt")
    rootname_flt.fits

    >>> print replaceSuffix("rootname_rawtag_a.fits", "_rawtag", "_flt")
    rootname_flt_a.fits
    """

    lenraw = len(rawname)
    lensuffix = len(suffix)
    i = rawname.rfind(suffix)
    if i >= 0:
        newname = rawname[0:i] + new_suffix + rawname[i+lensuffix:]
    else:
        raise RuntimeError("File name " + rawname +
                           " was expected to have suffix " + suffix)

    return newname

def getRootname(input, suffix):
    """Return the root of a file name.

    If suffix is found in input, return the portion of input
    that precedes suffix.  Otherwise, if input ends in ".fits",
    return everything from input that precedes ".fits".

    Parameters
    ----------
    input: str
        Name of a raw input file.

    suffix: str
        Suffix that might be found in input.

    Returns
    -------
    root: str
        input truncated before suffix, or truncated before ".fits"
        if suffix is not found.

    Examples
    --------
    >>> print getRootname("abc_asn.fits", "_asn")
    abc

    >>> print getRootname("abc_rawtag_b.fits", "_asn")
    abc_rawtag_b

    >>> print getRootname("abc_rawtag_b.fits", "_raw")
    abc

    >>> print getRootname("abc_rawtag_b.fits", "_rawtag")
    abc

    >>> print getRootname("abc_rawtag_b.fits", "_rawtag_b")
    abc

    >>> print getRootname("abc", "_asn")
    abc
    """

    # Allow corrtag as input.
    if input.find(suffix) < 0:
        suffix = "_corr"

    pieces = input.split(suffix)
    if len(pieces) > 1:
        root = suffix.join(pieces[:-1])
    elif input.endswith(".fits"):
        extn = input.rfind(".fits")
        root = input[:extn]
    else:
        root = input[:]

    return root

class Association(object):
    """Read and interpret the association table.

    Parameters
    ----------
    asntable: str
        The rootname (with "_asn") of an association file, or
        the rootname (with "_raw") of a raw file (or pair of files if FUV).

    outdir: str or None
        Name of output directory.

    cl_args: dictionary
        Some of the command-line arguments, or their defaults.
    """

    def __init__(self, asntable, outdir, cl_args):

        """Constructor."""

        self.asntable = None
        self.asntable_copy = None   # name of association file in output dir
        self.copy_asn = False       # set to True if need to copy asn file
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

        asntable = os.path.expandvars(asntable)
        self.indir = os.path.dirname(asntable)
        if outdir:
            self.outdir = outdir
        else:
            self.outdir = ""

        # Open the association table and read its contents into asn_info.
        if asntable.endswith("_asn"):
            self.asntable = asntable + ".fits"
            self.readAsnTable()
            self.asn_info["exists"] = True
        elif asntable.endswith("_asn.fits"):
            self.asntable = asntable
            self.readAsnTable()
            self.asn_info["exists"] = True
        else:
            # asntable is a raw file name; construct a one-row asn_info.
            self.asntable = None
            self.dummyAsnTable(asntable)
            self.asn_info["exists"] = False

        self.checkNeedToCopyAsn()

        memname = self.asn_info["memname"]
        memtype = self.asn_info["memtype"]
        mempresent = self.asn_info["mempresent"]

        for i in range(len(memname)):

            if memtype[i].find("PROD") >= 0:
                continue

            if mempresent[i]:

                basic_info = self.initialInfo(memname[i])
                if basic_info is None:
                    # Missing raw data, or error in association table.
                    continue            # don't process this exposure
                self.rawfiles.extend(basic_info["rawfiles"])
                concat_info = {}                # will be one element of concat
                concat_these = []               # x1d_a and x1d_b names
                concat_info_flash = {}          # another element of concat
                concat_these_flash = []         # flash_a and flash_b names
                # merge_corrtag, merge_flt, merge_counts (only used for FUV)
                # are pairs of names between which segment-specific keywords
                # will be copied
                merge_corrtag = []              # corrtag_a and corrtag_b names
                merge_flt = []                  # flt_a and flt_b names
                merge_counts = []               # counts_a and counts_b names
                first = True                    # first of a pair for FUV
                for input in basic_info["rawfiles"]:    # one (NUV) or two (FUV)
                    obs = initObservation(input, self.outdir, memtype[i],
                          basic_info["detector"], basic_info["obsmode"],
                          cl_args["shift_file"], first)
                    self.obs.append(obs)
                    if basic_info["detector"] == "FUV":
                        concat_these.append(obs.filenames["x1d_x"])
                        if obs.info["tagflash"]:
                            concat_these_flash.append(obs.filenames["flash_x"])
                        merge_corrtag.append(obs.filenames["corrtag"])
                        merge_flt.append(obs.filenames["flt"])
                        merge_counts.append(obs.filenames["counts"])
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
                        self.updateCombineFlt(obs.filenames,
                                              obs.info["obstype"])
                        if first:
                            self.updateCombineX1d(obs.filenames,
                                                  obs.info["fppos"],
                                                  obs.info["obstype"])
                    first = False

                if concat_these:
                    concat_info["input"] = concat_these
                    self.concat.append(concat_info)
                if concat_these_flash:
                    concat_info_flash["input"] = concat_these_flash
                    self.concat.append(concat_info_flash)
                if len(merge_flt) == 2:
                    self.merge_kwds.append(merge_corrtag)
                    self.merge_kwds.append(merge_flt)
                    self.merge_kwds.append(merge_counts)

        if len(self.obs) == 0:
            return

        cosutil.printMsg("combine = " + repr(self.combine), VERY_VERBOSE)
        cosutil.printMsg("concat = " + repr(self.concat), VERY_VERBOSE)

        # Find the first science obs (i.e. non-wavecal, if possible).
        self.first_science_tuple = self.findFirstScience()
        (i, j) = self.first_science_tuple
        if i is None:
            i = j
        self.first_science = i

        self.checkforWalk()
        self.compareConfig()
        self.resetSwitches()    # set switches to OMIT if only_csum is True
        self.compareRefFiles()
        self.compareSwitches()
        self.missingRefFiles()
        self.checkGeoSwitches()
        self.globalSwitches()
        self.checkOutputExists()
        self.stimfileSanityCheck()

    def checkforWalk(self):
        """Check for the existence of any WALK-related keywords:
        [WALKCORR, WALKTAB] in the primary header of all members
        """
        walkreferrers = []
        for rawfile in self.rawfiles:
            f1 = fits.open(rawfile)
            phdr = f1[0].header
            if 'WALKCORR' in phdr.keys():
                walkreferrers.append((rawfile, 'WALKCORR'))
            if 'WALKTAB' in phdr.keys():
                walkreferrers.append((rawfile, 'WALKTAB'))
            f1.close()

        if len(walkreferrers) > 0:
            errormessage = "Input file(s) contain keywords WALKCORR and/or WALKTAB"
            for referrer in walkreferrers:
                errormessage = "".join([errormessage,"\n   " + referrer[0] + ":  " + referrer[1]])
            errormessage = "".join([errormessage,"\n\nPlease either re-retrieve the data from MAST or"])
            errormessage = "".join([errormessage,"\nreplace these keywords with XWLKCORR+YWLKCORR and"])
            errormessage = "".join([errormessage,"\nXWLKFILE+YWLKFILE"])
            raise RuntimeError(errormessage)

    def readAsnTable(self):
        """Read an association table into memory, and get product info."""

        cosutil.printMsg("Association file = " + self.asntable, VERBOSE)

        fd = fits.open(self.asntable, mode="copyonwrite", memmap=False)
        asn_data = fd[1].data
        nrows = asn_data.shape[0]
        if nrows <= 0:
            fd.close()
            raise RuntimeError("The association table is empty.")

        self.asn_info["memname"] = []
        self.asn_info["memtype"] = []
        self.asn_info["mempresent"] = []

        # Convert the memnames to lower case (unless a full file name was
        # given), and prefix them with the input directory name.
        asn_memname = asn_data.field("memname")
        asn_memtype = asn_data.field("memtype")
        asn_memprsnt = asn_data.field("memprsnt")
        for i in range(nrows):
            if not asn_memname[i].endswith(".fits"):
                asn_memname[i] = asn_memname[i].lower()
            self.asn_info["memname"].append(
                        os.path.join(self.indir, asn_memname[i]))
            self.asn_info["memtype"].append(asn_memtype[i])
            self.asn_info["mempresent"].append(asn_memprsnt[i])

        fd.close()

        self.product = None
        for i in range(nrows):
            if asn_memtype[i].find("PROD") >= 0:
                if self.product is not None:
                    raise RuntimeError("The association table may list "
                                       "no more than one product.")
                self.product = asn_memname[i].lower()
                self.product_type = asn_memtype[i]

        if self.product is not None:
            self.product = os.path.join(self.outdir, self.product)
            cosutil.printMsg("product = " + self.product, VERY_VERBOSE)

        # Enable writing to trailer files.
        # cosutil.setWriteToTrailer(True)

    def dummyAsnTable(self, asntable):
        """Construct a recarray corresponding to an association table.

        This function will be called for the case that the user specified
        the name of a raw (or corrtag) file instead of an association table.
        The asntable argument is the name as given by the user; we will
        assign that full name (not just the rootname) to asn_info["memname"].
        There will only be this one row; product will be set to None.  The
        memtype will be set to "none", even though it might actually be a
        wavecal.

        Parameters
        ----------
        asntable: str
            The name of an input raw file (not really an association table
            name).
        """

        cosutil.printMsg("Input file = " + asntable, VERBOSE)

        # asntable is not an association table name, it's an actual file
        # name.  If a complete file name was specified, and if that file
        # exists, save the full name as memname; otherwise, extract the
        # root name and save that.
        if os.access(asntable, os.R_OK):
            self.asn_info["memname"] = [asntable]
        else:
            rootname = getRootname(asntable, "_raw")
            self.asn_info["memname"] = [rootname]

        self.asn_info["memtype"] = ["none"]
        self.asn_info["mempresent"] = [True]    # yes, it is present

        # Because the input is not an association, there is no product.
        self.product = None
        self.product_type = None

        # Disable writing to the trailer file.
        # cosutil.setWriteToTrailer(False)

    def checkNeedToCopyAsn(self):
        """Check whether the association file should be copied to outdir.

        If there is an association table (rather than a raw file) that
        specifies a "product," and if the input and output directories
        are not the same, this function sets attribute copy_asn to True
        to indicate that calcos should copy the association table to the
        output directory, and that copy is the one that will be modified
        to indicate that the product has been written.
        """

        if self.product is None:
            self.copy_asn = False
        else:
            if self.indir:
                input_directory = self.indir
            else:
                input_directory = os.curdir
            if self.outdir:
                output_directory = expandDirectory(self.outdir)
            else:
                output_directory = os.curdir
            if os.path.samefile(input_directory, output_directory):
                self.copy_asn = False
            else:
                self.copy_asn = True

        if self.copy_asn:
            self.asntable_copy = os.path.join(output_directory,
                                              os.path.basename(self.asntable))
        else:
            self.asntable_copy = self.asntable          # just copy the name

    def initialInfo(self, memname):
        """Get preliminary information from an input file.

        This gets the names of the raw files, and from the first of those
        files, reads the primary header and calls a function to get
        DETECTOR, OBSMODE, and EXPTYPE.  In addition, this function checks
        that the suffixes are as expected for the DETECTOR and OBSMODE
        keywords.

        If the input is a complete file name, and the file exists, then the
        dictionary of keywords and values will be returned without using
        wildcards or other checks on suffix.

        Parameters
        ----------
        memname: str
            A value in the MEMNAME (member name) column of an association
            table, converted to lower case; if the user specified an
            explicit file name rather than an association table name,
            memname should be the full file name.

        Returns
        -------
        basic_info: dictionary or None
            Dictionary of keywords and values; the value will be None if
            there are no files that match the template, or if the input is
            an ACQ other than ACQ/IMAGE.
        """

        # Did the user specify a particular input file?  If so, this is
        # all we need to do.
        if os.access(memname, os.R_OK):
            basic_info = getinfo.initialInfo(memname)
            if memname.endswith("rawacq.fits") and \
               basic_info["exptype"] != "ACQ/IMAGE":
                cosutil.printWarning("File %s will be skipped because " \
                                     "it is not an ACQ/IMAGE" % memname)
                return None
            rawfiles = [memname]
            if basic_info["detector"] == "FUV":
                if memname.endswith("_a.fits"):
                    other_segment = memname[:-7] + "_b.fits"
                elif memname.endswith("_b.fits"):
                    other_segment = memname[:-7] + "_a.fits"
                else:
                    other_segment = None
                if other_segment is not None and \
                   os.access(other_segment, os.R_OK):
                    rawfiles.append(other_segment)
                    rawfiles.sort()
            basic_info["rawfiles"] = rawfiles

            return basic_info

        # First find out whether we've got time-tag or accum, FUV or NUV.
        # Look for both rawaccum and rawimage.
        all_rawfiles = []
        raw = glob.glob(memname + "_rawtag*.fits")
        all_rawfiles.extend(raw)
        raw = glob.glob(memname + "_rawaccum*.fits")
        all_rawfiles.extend(raw)
        raw = glob.glob(memname + "_rawimage*.fits")
        all_rawfiles.extend(raw)

        # The input should not include both science files and acq files,
        # but if it does, make sure the first raw file (see below) is a
        # science file rather than an acq.  If the only file is an acq,
        # however, it must be the first raw file, and that's OK.
        if len(all_rawfiles) > 0:
            all_rawfiles.sort()
        raw = glob.glob(memname + "_rawacq.fits")
        if raw:
            if self.isAcqImage(raw[0]):
                all_rawfiles.extend(raw)
            else:
                cosutil.printWarning(
            "File %s will be skipped because it is not an ACQ/IMAGE" % raw[0])

        if not all_rawfiles:
            cosutil.printWarning(
                "There are no files to calibrate for rootname '%s'" % memname)
            return None

        # Get info from the first raw file with the specified rootname.
        initial_basic_info = getinfo.initialInfo(all_rawfiles[0])
        detector = initial_basic_info["detector"]
        obsmode = initial_basic_info["obsmode"]
        exptype = initial_basic_info["exptype"]
        if exptype[0:3] == "ACQ" and exptype != "ACQ/IMAGE":
            cosutil.printWarning(
                "Rootname '%s' is an %s, which cannot be processed." %
                    (memname, exptype))
            return None

        # Find the raw files that we expect to have.
        if detector == "FUV":
            tail = "_[ab].fits"
        else:
            tail = ".fits"
        if obsmode == "TIME-TAG":
            rawfiles = glob.glob(memname + "_rawtag" + tail)
        elif obsmode == "ACCUM":
            # first look for rawaccum
            rawfiles = glob.glob(memname + "_rawaccum" + tail)
            if len(rawfiles) < 1:
                # rawaccum not found, so look for rawimage
                rawfiles = glob.glob(memname + "_rawimage" + tail)
        else:
            raise RuntimeError("unexpected OBSMODE '%s' in '%s'"
                               % (obsmode, all_rawfiles[0]))
        if len(rawfiles) > 0:
            rawfiles.sort()
        rawfiles.extend(glob.glob(memname + "_rawacq.fits"))

        nfiles = len(rawfiles)
        if nfiles == 0:
            raise RuntimeError("Keywords and filenames are inconsistent "
                               "for rootname '%s'" % memname)

        # Read the first raw file with the specified rootname.
        basic_info = getinfo.initialInfo(rawfiles[0])

        if len(rawfiles) < len(all_rawfiles):
            cosutil.printWarning("There are more raw files than we expected:")
            cosutil.printContinuation("we expected " + repr(rawfiles))
            cosutil.printContinuation("but we found " + repr(all_rawfiles))

        basic_info["rawfiles"] = rawfiles

        return basic_info

    def checkGeoSwitches(self):
        """Check that GEOCORR switches aren't set to 'OMIT' if the DGEOCORR switches are
        set to 'PERFORM'"""
        incompatibleSwitches = []
        for rawfile in self.rawfiles:
            f1 = fits.open(rawfile)
            phdr = f1[0].header
            if 'DGEOCORR' in phdr.keys():
                geocorr = phdr['GEOCORR']
                dgeocorr = phdr['DGEOCORR']
                if dgeocorr == 'PERFORM':
                    if geocorr == 'OMIT':
                        incompatibleSwitches.append((rawfile, geocorr, dgeocorr))
            f1.close()

        if len(incompatibleSwitches) > 0:
            errormessage = "Input file(s) have illegal combination of GEOCORR='OMIT' and DGEOCORR='PERFORM'"
            for badfile in incompatibleSwitches:
                errormessage = "".join([errormessage,"\n   " + badfile[0]])
            errormessage = "".join([errormessage,"\n\nPlease either set the GEOCORR switch to 'PERFORM'"])
            errormessage = "".join([errormessage,"\nif the GEO correction hasn't been applied, or 'COMPLETE'"])
            errormessage = "".join([errormessage,"\nif it has"])
            raise RuntimeError(errormessage)

    def isAcqImage(self, rawacq):
        """Check whether rawacq is an ACQ/IMAGE.

        Parameters
        ----------
        rawacq: str
            Name of an acq file.

        Returns
        -------
        flag: boolean
            True if exptype for rawacq is "ACQ/IMAGE", False otherwise
        """

        fd = fits.open(rawacq, mode="readonly")
        exptype = fd[0].header.get("exptype", "not found")
        fd.close()

        if exptype == "ACQ/IMAGE":
            return True
        else:
            return False

    def updateCombineFlt(self, filenames, obstype):
        """Add the flt name to the input lists in self.combine.

        Parameters
        ----------
        filenames: dictionary
            Dictionary of input and output file names.

        obstype: str {"SPECTROSCOPIC", "IMAGING"}
            Observation type.
        """

        if obstype != "IMAGING":
            return

        if self.cl_args["only_csum"]:           # there won't be any flt files
            return

        if "flt" not in self.combine:
            self.combine["flt"] = []

        flt = filenames["flt"]
        self.combine["flt"].append(flt)

    def updateCombineX1d(self, filenames, fppos, obstype):
        """Add the x1d name and fppos index to 'combine'.

        Parameters
        ----------
        filenames: dictionary
            Dictionary of input and output file names.

        fppos: int {1, 2, 3, 4}
            Focal plane position index.

        obstype: str
            Observation type, "SPECTROSCOPIC" or "IMAGING".
        """

        if obstype != "SPECTROSCOPIC":
            return

        if self.cl_args["only_csum"]:           # there won't be any x1d files
            return

        if "x1d" not in self.combine:
            self.combine["x1d"] = []
        self.combine["x1d"].append(filenames["x1d"])

        if "fppos" not in self.combine:
            self.combine["fppos"] = []
        self.combine["fppos"].append(fppos)

    def findFirstScience(self):
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

        Returns
        -------
        (i_timetag, i_accum): tuple of two integers
            Indexes of the first time-tag and the first accum science
            observations.
        """

        i_timetag = None
        i_accum = None
        foundit_timetag = False
        foundit_accum = False

        # look for a time-tag science observation
        for i in range(len(self.obs)):
            obs = self.obs[i]
            if obs.exp_type == EXP_SCIENCE and \
               obs.info["obsmode"] == "TIME-TAG":
                foundit_timetag = True
                i_timetag = i
                break
        # look for an accum science observation
        for i in range(len(self.obs)):
            obs = self.obs[i]
            if obs.exp_type == EXP_SCIENCE and \
               obs.info["obsmode"] == "ACCUM":
                foundit_accum = True
                i_accum = i
                break

        if not foundit_timetag:
            # No time-tag science observation; find the first wavecal, if any.
            for i in range(len(self.obs)):
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

    def compareConfig(self):
        """Compare detector and opt_elem.

        All the files in an association must have been taken with the same
        detector, grating (or mirror), and cenwave.
        """

        if len(self.obs) < 2:
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
                cosutil.printError(obs.filenames["raw"])
                errmess = "All files must be for the same detector"
                if obs.info["obstype"] == "SPECTROSCOPIC":
                    errmess += ", opt_elem and cenwave."
                else:
                    errmess += " and opt_elem."
                raise RuntimeError(errmess)

    def resetSwitches(self):
        """Reset most/all switches to OMIT if only_csum is True."""

        if not self.cl_args["only_csum"]:
            return

        if self.cl_args["raw_csum_coords"]:
            leave_unchanged = []        # reset all switches to OMIT
        else:
            leave_unchanged = ["tempcorr", "geocorr", "dgeocorr", "igeocorr", "randcorr"]

        for obs in self.obs:
            for key in obs.switches:
                if key not in leave_unchanged and \
                   obs.switches[key] == "PERFORM":
                    obs.switches[key] = "OMIT"
            obs.reffiles["spwcstab"] = "N/A"

    def compareRefFiles(self):
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
            # keys = reffiles.keys()
            # keys.sort()
            keys = sorted(reffiles)
            for key in keys:
                if key.find("_hdr") >= 0:
                    continue
                compare = reffiles[key].strip()
                a_file = obs.reffiles[key].strip()
                if a_file != compare:
                    # Ignore spwcstab between science and wavecal.
                    if key == "spwcstab" and obs.info["exptype"] == "WAVECAL":
                        continue
                    compare_hdr = reffiles[key+"_hdr"]
                    a_file_hdr = obs.reffiles[key+"_hdr"]
                    if not message_printed:
                        cosutil.printWarning(
                                "Inconsistent reference file names:")
                        message_printed = True
                    if len(compare) == 0:
                        compare = "(blank)"
                    if len(a_file) == 0:
                        a_file = "(blank)"
                    cosutil.printMsg(obs.input + ":  " + key + " = " + \
                                     a_file_hdr + " vs. " + compare_hdr)
            obs.closeTrailer()

    def compareSwitches(self):
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
            # keys = switches.keys()
            # keys.sort()
            keys = sorted(switches)
            for key in keys:
                compare = switches[key].strip()
                sw = obs.switches[key].strip()
                if sw != compare:
                    if obs.exp_type == EXP_WAVECAL:
                        if key in ["wavecorr", "doppcorr",
                                   "helcorr", "fluxcorr", "tdscorr"]:
                            continue
                    if not message_printed:
                        cosutil.printWarning(
                                "Inconsistent calibration switches:")
                        message_printed = True
                    if len(compare) == 0:
                        compare = "(blank)"
                    if len(sw) == 0:
                        sw = "(blank)"
                    cosutil.printMsg(obs.input + ":  " + key + " = " + \
                                     sw + " vs. " + compare)
            obs.closeTrailer()

    def missingRefFiles(self):
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
        switches = copy.copy(self.obs[i].switches)
        reffiles = copy.copy(self.obs[i].reffiles)
        info = copy.copy(self.obs[i].info)
        if j is not None:
            j_switches = copy.copy(self.obs[j].switches)
            j_reffiles = copy.copy(self.obs[j].reffiles)
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
            "hvtab":    ["2.0", "FUV HIGH VOLTAGE HISTORY"],
            "flatfile": ["2.0", "FLAT FIELD REFERENCE IMAGE"],
            "badttab":  ["2.0", "BAD TIME INTERVALS TABLE"],
            "bpixtab":  ["2.0", "DATA QUALITY INITIALIZATION TABLE"],
            "gsagtab":  ["2.0", "GAIN SAG REFERENCE TABLE"],
            "deadtab":  ["2.0", "DEADTIME REFERENCE TABLE"],
            "brftab":   ["2.0", "BASELINE REFERENCE FRAME TABLE"],
            "phatab":   ["2.0", "PULSE HEIGHT PARAMETERS REFERENCE TABLE"],
            "phafile":  ["2.0", "PULSE HEIGHT THRESHOLD REFERENCE IMAGE"],
            "geofile":  ["2.0", "GEOMETRIC DISTORTION REFERENCE IMAGE"],
            "dgeofile": ["2.0", "DELTA GEOMETRIC CORRECTION REFERENCE IMAGE"],
            "lamptab":  ["2.0", "TEMPLATE CAL LAMP SPECTRA TABLE"],
            "wcptab":   ["2.0", "WAVECAL PARAMETERS REFERENCE TABLE"],
            "spwcstab": ["2.0", "SPECTROSCOPIC WCS PARAMETERS TABLE"],
            "xtractab": ["2.0", "1-D EXTRACTION PARAMETERS TABLE"],
            "disptab":  ["2.0", "DISPERSION RELATION REFERENCE TABLE"],
            "fluxtab":  ["2.0", "PHOTOMETRIC SENSITIVITY REFERENCE TABLE"],
            "imphttab": ["2.0", "IMAGING PHOTOMETRIC TABLE"],
            "tdstab":   ["2.0", "TIME DEPENDENT SENSITIVITY TABLE"],
            "brsttab":  ["2.0", "BURST PARAMETERS TABLE"],
            "xwlkfile": ["3.1", "X WALK CORRECTION LOOKUP REFERENCE IMAGE"],
            "ywlkfile": ["3.1", "Y WALK CORRECTION LOOKUP REFERENCE IMAGE"],
            "tracetab": ["2.0", "1D SPECTRAL TRACE TABLE"],
            "proftab": ["2.0", "2D SPECTRUM PROFILE TABLE"],
            "twozxtab": ["2.0", "TWO-ZONE SPECTRAL EXTRACTION PARAMETERS TABLE"],
            "spottab": ["2.0", "TRANSIENT BAD PIXEL REFERENCE TABLE"]
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

        if reffiles["hvtab"] != NOT_APPLICABLE:
            cosutil.findRefFile(ref["hvtab"],
                                missing, wrong_filetype, bad_version)

        if switches["flatcorr"] == "PERFORM":
            cosutil.findRefFile(ref["flatfile"],
                                missing, wrong_filetype, bad_version)

        if switches["brstcorr"] == "PERFORM":
            cosutil.findRefFile(ref["brsttab"],
                                missing, wrong_filetype, bad_version)

        if switches["badtcorr"] == "PERFORM":
            cosutil.findRefFile(ref["badttab"],
                                missing, wrong_filetype, bad_version)

        if switches["dqicorr"] == "PERFORM":
            cosutil.findRefFile(ref["bpixtab"],
                                missing, wrong_filetype, bad_version)
            if reffiles["gsagtab"] != NOT_APPLICABLE:
                cosutil.findRefFile(ref["gsagtab"],
                                    missing, wrong_filetype, bad_version)
            if reffiles["spottab"] != NOT_APPLICABLE:
                cosutil.findRefFile(ref["spottab"],
                                    missing, wrong_filetype, bad_version)

        if switches["deadcorr"] == "PERFORM":
            cosutil.findRefFile(ref["deadtab"],
                                missing, wrong_filetype, bad_version)

        if switches["tempcorr"] == "PERFORM":
            cosutil.findRefFile(ref["brftab"],
                                missing, wrong_filetype, bad_version)

        if switches["phacorr"] == "PERFORM":
            (i_pha, j_pha) = self.first_science_tuple
            if i_pha is not None:
                # there is a TIME-TAG exposure
                if reffiles["phafile"] != NOT_APPLICABLE:
                    # phafile was specified
                    cosutil.findRefFile(ref["phafile"],
                                        missing, wrong_filetype, bad_version)
                else:
                    # no phafile; use phatab instead
                    cosutil.findRefFile(ref["phatab"],
                                        missing, wrong_filetype, bad_version)
            if j_pha is not None:
                # there is an ACCUM exposure, so we need phatab
                cosutil.findRefFile(ref["phatab"],
                                    missing, wrong_filetype, bad_version)

        if switches["geocorr"] == "PERFORM":
            cosutil.findRefFile(ref["geofile"],
                                missing, wrong_filetype, bad_version)

        if switches["dgeocorr"] == "PERFORM":
            # check that geocorr is not 'OMIT'
            if switches["geocorr"] == 'OMIT':
                cosutil.printError("DGEOCORR = 'PERFORM' but GEOCORR = 'OMIT'")
                cosutil.printContinuation("This combination is not permitted")
                cosutil.printContinuation("Please change the GEOCORR keyword to 'PERFORM' or 'COMPLETE'")
                raise RuntimeError("Error in GEOCORR switch when DGEOCORR = 'PERFORM'")
            cosutil.findRefFile(ref["dgeofile"],
                                missing, wrong_filetype, bad_version)

        if switches["wavecorr"] == "PERFORM":
            if self.obs[i].info["obstype"] != "IMAGING":
                cosutil.findRefFile(ref["lamptab"],
                                    missing, wrong_filetype, bad_version)
                cosutil.findRefFile(ref["wcptab"],
                                    missing, wrong_filetype, bad_version)

        if switches["x1dcorr"] == "PERFORM":
            cosutil.findRefFile(ref["xtractab"],
                                missing, wrong_filetype, bad_version)
            cosutil.findRefFile(ref["disptab"],
                                missing, wrong_filetype, bad_version)
            if ref["spwcstab"]["filename"] != NOT_APPLICABLE:
                cosutil.findRefFile(ref["spwcstab"],
                                    missing, wrong_filetype, bad_version)

        if switches["fluxcorr"] == "PERFORM":
            cosutil.findRefFile(ref["fluxtab"],
                                missing, wrong_filetype, bad_version)
            if switches["tdscorr"] == "PERFORM":
                cosutil.findRefFile(ref["tdstab"],
                                    missing, wrong_filetype, bad_version)

        if switches["xwlkcorr"] == "PERFORM":
            cosutil.findRefFile(ref["xwlkfile"],
                                missing, wrong_filetype, bad_version)
        if switches["ywlkcorr"] == "PERFORM":
            cosutil.findRefFile(ref["ywlkfile"],
                                missing, wrong_filetype, bad_version)
        if switches["photcorr"] == "PERFORM":
            # xxx commented out because we don't have this table yet
            # cosutil.findRefFile(ref["imphttab"],
            #                     missing, wrong_filetype, bad_version)
            pass

        if switches["trcecorr"] == "PERFORM":
            cosutil.findRefFile(ref["tracetab"],
                                missing, wrong_filetype, bad_version)

        if switches["algncorr"] == "PERFORM":
            cosutil.findRefFile(ref["proftab"],
                                missing, wrong_filetype, bad_version)

        if info["xtrctalg"] == "TWOZONE":
            cosutil.findRefFile(ref["twozxtab"],
                                missing, wrong_filetype, bad_version)
        if len(missing) > 0:
            msg = "The following reference file"
            if len(missing) > 1:
                msg += "s are missing:"
            else:
                msg += " is missing:"
            cosutil.printError(msg)
            # keywords = missing.keys()
            # keywords.sort()
            keywords = sorted(missing)
            for key in keywords:
                cosutil.printMsg(key + "=" + missing[key])

        if len(wrong_filetype) > 0:
            cosutil.printError("Wrong FILETYPE; expected the following:")
            # keywords = wrong_filetype.keys()
            # keywords.sort()
            keywords = sorted(wrong_filetype)
            for key in keywords:
                cosutil.printMsg(key + " = " + wrong_filetype[key][0])
                cosutil.printMsg(
                    "  filetype should be " + wrong_filetype[key][1])

        if len(bad_version) > 0:
            cosutil.printError(
                "Version incompatibility between CALCOS and reference file:")
            # keywords = bad_version.keys()
            # keywords.sort()
            keywords = sorted(bad_version)
            for key in keywords:
                cosutil.printMsg(key + " = " + bad_version[key][0])
                cosutil.printMsg(bad_version[key][1])

        if len(missing) > 0 or len(wrong_filetype) > 0 or \
           len(bad_version) > 0:
            raise RuntimeError()

    def globalSwitches(self):
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
                    "dqicorr",  "flatcorr", "geocorr",
                    "dgeocorr", "helcorr",
                    "phacorr",  "randcorr", "tempcorr", "x1dcorr",
                    "wavecorr", "trcecorr", "algncorr"]:
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
            if "x1d" in self.combine:
                ncombine = len(self.combine["x1d"])
            elif "flt" in self.combine:
                ncombine = len(self.combine["flt"])
            else:
                ncombine_a = 0
                ncombine_b = 0
                if "flt_a" in self.combine:
                    ncombine_a = len(self.combine["flt_a"])
                if "flt_b" in self.combine:
                    ncombine_b = len(self.combine["flt_b"])
                ncombine = max(ncombine_a, ncombine_b)

    def isAnySwitchSet(self):
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

    def checkOutputExists(self):
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
                self.checkExists(obs.filenames["corrtag"], wavecal_exists)
                self.checkExists(obs.filenames["flt"], wavecal_exists)
                self.checkExists(obs.filenames["counts"], wavecal_exists)
                self.checkExists(obs.filenames["x1d_x"], wavecal_exists)
                if obs.filenames["x1d"] != obs.filenames["x1d_x"]:
                    self.checkExists(obs.filenames["x1d"], wavecal_exists)
                if self.cl_args["create_csum_image"]:
                    self.checkExists(obs.filenames["csum"], wavecal_exists)
            else:
                self.checkExists(obs.filenames["corrtag"], already_exists)
                self.checkExists(obs.filenames["flt"], already_exists)
                self.checkExists(obs.filenames["counts"], already_exists)
                if obs.info["obsmode"] == "TIME-TAG":
                    self.checkExists(obs.filenames["flash_x"], already_exists)
                    self.checkExists(obs.filenames["flash"], already_exists)
                if obs.switches["x1dcorr"] == "PERFORM":
                    self.checkExists(obs.filenames["x1d_x"], already_exists)
                    if obs.filenames["x1d"] != obs.filenames["x1d_x"]:
                        self.checkExists(obs.filenames["x1d"], already_exists)

        if self.product is not None:
            self.checkExists(self.product + "_fltsum.fits", already_exists)
            self.checkExists(self.product + "_x1dsum.fits", already_exists)
            self.checkExists(self.product + "_x1dsum1.fits", already_exists)
            self.checkExists(self.product + "_x1dsum2.fits", already_exists)
            self.checkExists(self.product + "_x1dsum3.fits", already_exists)
            self.checkExists(self.product + "_x1dsum4.fits", already_exists)

        # Remove duplicates.
        for i in range(len(already_exists)-1, 0, -1):
            fname = already_exists[i]
            if fname in already_exists[0:i]:
                del(already_exists[i])
        for i in range(len(wavecal_exists)-1, 0, -1):
            fname = wavecal_exists[i]
            if fname in wavecal_exists[0:i]:
                del(wavecal_exists[i])

        if already_exists:
            if len(already_exists) == 1:
                errmess = "output file already exists"
            else:
                errmess = "output files already exist"
            cosutil.printError(errmess + ":")
            for fname in already_exists:
                cosutil.printError("  %s" % fname)
            raise RuntimeError(errmess)

        if wavecal_exists:
            if len(wavecal_exists) == 1:
                msg = "Calibrated wavecal file already exists"
            else:
                msg = "Calibrated wavecal files already exist"
            msg += ", will be deleted:"
            cosutil.printWarning(msg)
            for fname in wavecal_exists:
                os.remove(fname)
                cosutil.printWarning("  %s deleted" % fname)

    def checkExists(self, fname, already_exists):
        """If fname exists, append the name to already_exists.

        Parameters
        ----------
        fname: str
            The name of the file.

        already_exists: list of strings
            A list of names of files that currently exist; may be modified
            in-place by appending fname.
        """

        if os.access(fname, os.R_OK):
            already_exists.append(fname)

    def stimfileSanityCheck(self):
        """Ignore stimfile if detector is not FUV.

        Only the FUV detector has stims.  If a file was specified for
        saving measured stim locations (--stim stimfile), the name will
        be reset to None if the detector was not FUV.
        """

        i = self.first_science
        if self.cl_args["stimfile"] is not None and \
           self.obs[i].info["detector"] != "FUV":
            self.cl_args["stimfile"] = None
            cosutil.printWarning(
                "stimfile reset to None because detector is NUV.")

    def updateMempresent(self):
        """Update the ASN_PROD keyword and MEMPRSNT column."""

        if self.asntable is None or self.product is None:
            return
        cosutil.printMsg("updateMempresent", VERY_VERBOSE)

        if self.copy_asn:
            fd = fits.open(self.asntable)
            fd.writeto(self.asntable_copy)
            fd.close()

        # Modify the association table in-place.
        fd = fits.open(self.asntable_copy, mode="update")

        # Set ASN_PROD to true to indicate that a product has been created.
        fd[0].header["asn_prod"] = True

        asn = fd[1].data
        nrows = asn.shape[0]
        memtype = asn.field("MEMTYPE")
        mempresent = asn.field("MEMPRSNT")

        for i in range(nrows):
            if memtype[i].find("PROD") >= 0:
                mempresent[i] = True
                break

        fd.close()

    def copySptFile(self):
        """Copy an spt file to the association product name."""

        if self.asntable is None or self.product is None:
            return
        cosutil.printMsg("copySptFile", VERY_VERBOSE)

        # Find the first science observation that has a support file.
        sptfile = None
        fallback = None         # use this if no suitable spt file available
        for i in range(len(self.obs)):
            obs = self.obs[i]
            if not os.access(obs.filenames["spt"], os.R_OK):
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
                cosutil.printWarning(
                "spt file not found, so not copied to product", VERBOSE)
                return
            else:
                sptfile = fallback

        # Change the suffix to "jnk" so that if the user gets this file it
        # will be clear that it should be ignored.
        product_spt_file = self.product + "_jnk.fits"
        cosutil.printMsg("copy %s to %s" % (sptfile, product_spt_file),
                          VERY_VERBOSE)

        # Copy the spt file to the "product spt" file.
        fd = fits.open(sptfile)
        fd.writeto(product_spt_file)
        fd.close()

        # Update keywords in the "product spt" file.
        fd = fits.open(product_spt_file, mode="update")

        phdr = fd[0].header
        product = os.path.basename(self.product)

        cosutil.updateFilename(phdr, product_spt_file)
        phdr["rootname"] = product
        phdr["obset_id"] = product[4:6]
        phdr["observtn"] = product[-3:].upper()
        phdr["asn_mtyp"] = self.product_type            # do we need this?
        phdr.add_comment(
        "Please ignore this file, which is a copy of an input spt file.")
        phdr.add_comment(
        "This file is used by the archive to obtain certain keywords.")

        for i in range(1, len(fd)):
            fd[i].header["rootname"] = product

        fd.close()

def initObservation(input, outdir, memtype, detector, obsmode,
                    shift_file, first=False):
    """Construct an Observation object for the current mode.

    Parameters
    ----------
    input: str
        The name of an input raw file.

    outdir: str
        Either an empty string or the name of the output directory.

    memtype: str
        From association table; used to distinguish between wavecal and
        science observation.

    detector: str {"FUV", "NUV"}
        The detector name.

    obsmode: str {"TIME-TAG", "ACCUM"}
        The observation mode.

    shift_file: str or None
        The name of the shift file (command-line argument), if one was
        specified.  This is only used (here) to possibly override keyword
        TAGFLASH.

    first: boolean
        True if the current file is the first for a given rootname (this
        is for writing the calcos version string to the trailer, so that
        it won't be written for both FUV segments A and B).

    Returns
    -------
    obs: instance
        An Observation object.
    """

    if detector == "FUV":
        if obsmode == "TIME-TAG":
            obs = FUVTimetagObs(input, outdir, memtype, shift_file, first)
        else:
            obs = FUVAccumObs(input, outdir, memtype, shift_file, first)
    else:
        if obsmode == "TIME-TAG":
            obs = NUVTimetagObs(input, outdir, memtype, shift_file, first)
        else:
            obs = NUVAccumObs(input, outdir, memtype, shift_file, first)

    return obs

class Observation(object):
    """Get information about an observation from its headers.

    This base class is not directly used; one of its subclasses will
    be invoked, depending on DETECTOR and OBSMODE.

    Parameters
    ----------
    input: str
        The name of an input raw file.

    outdir: str
        An empty string or the name of the output directory.

    memtype: str
        Read from the association table; used to distinguish between
        wavecal and science observation.

    suffix: str
        Suffix to the rootname, but just "_rawtag" or "_rawaccum" (i.e.
        excluding "_a" or "_b" if the data were taken with the FUV
        detector); this may be reset internally to "_corrtag" or
        "_rawimage" or "_rawacq".

    shift_file: str or None
        The name of the shift file (command-line argument), if one was
        specified.

    first: boolean
        True if the current file is the first of two for FUV.
    """

    def __init__(self, input, outdir, memtype, suffix, shift_file, first):
        """Invoked by a subclass."""

        self.input = input              # name of a raw input file
        self.exp_type = EXP_SCIENCE     # science, wavecal, target acq
        self.filenames = {}             # input and output file names
        self.info = {}                  # detector, opt_elem, etc.
        self.switches = {}              # calibration switch values
        self.reffiles = {}              # reference file names
        self.info['addsimulatedwavecal'] = False

        indir = os.path.dirname(input)
        input_directory = expandDirectory(indir)
        if outdir:
            output_directory = expandDirectory(outdir)
        else:
            output_directory = os.path.realpath(os.curdir)

        self.getHeaderInfo()

        if self.info["corrtag_input"]:
            suffix = "_corrtag"
            if input.find(suffix) >= 0:
                if input_directory == output_directory:
                    raise RuntimeError("For corrtag input,"
                    " the input and output directories must not be the same.")
            else:
                suffix = "_rawtag"
        else:
            # For ACCUM data, allow suffix to be "_rawaccum", "_rawimage" or
            # "_rawacq".
            if input.find(suffix) < 0:
                suffix = "_rawimage"
            if input.find(suffix) < 0:
                suffix = "_rawacq"
        if input.find(suffix) < 0:
            raise RuntimeError("can't find suffix %s in %s" % (suffix, input))

        self.filenames = self.makeFileNames(suffix, outdir)
        # This value of info["root"] is based on the filename on disk, which
        # could differ from the value of the rootname keyword.
        self.info["root"] = self.filenames["root"]
        self.openTrailer(first)     # open the trailer file for this input file
        self.sanityCheck()

        # If a shift file was specified but this is not a tagflash exposure
        # or a wavecal, override the tagflash flag so the shift file will
        # actually be used.
        if shift_file is not None and not self.info["tagflash"] and \
           self.info["exptype"] != "WAVECAL":
            # Is the current exposure included in the shift file?
            user_shifts = shiftfile.ShiftFile(shift_file, self.info["root"],
                                              self.info["fpoffset"])
            (shifts, nfound) = user_shifts.getShifts(("any", "any"))
            if nfound > 0:
                self.info["tagflash"] = True    # OK to use the shift file
            del(user_shifts)

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
            cosutil.printWarning("EXPTYPE = '%s' in %s;" \
                                 % (self.info["exptype"], input))
            cosutil.printContinuation("can't calibrate this exposure type.")
            self.exp_type = EXP_ENGINEERING
        else:
            cosutil.printWarning("EXPTYPE = '%s' in %s;" \
                                 % (self.info["exptype"], input))
            cosutil.printContinuation("don't recognize this exposure type.")
            self.exp_type = EXP_UNKNOWN

        if memtype != "none":           # is there an association table?
            conflict = False
            memtype_wavecal = memtype.endswith("WAVE")
            exptype_wavecal = self.info["exptype"] == "WAVECAL"
            if memtype_wavecal and not exptype_wavecal:
                conflict = True
            if exptype_wavecal and not memtype_wavecal:
                conflict = True
            if conflict:
                raise RuntimeError("MEMTYPE = %s but EXPTYPE = %s for %s" %
                                   (memtype, self.info["exptype"], self.input))

        if self.info["obstype"] == "SPECTROSCOPIC":
            if self.info["tagflash"]:
                if self.info["exptype"] == "EXTERNAL/SCI":
                    pass                        # no change needed
                elif self.info["exptype"] == "WAVECAL":
                    cosutil.printWarning("EXPTYPE = WAVECAL but TAGFLASH "
                                         "!= NONE for %s;" % self.input)
                    cosutil.printContinuation(
                        "EXPTYPE will be changed to EXTERNAL/CAL.")
                    self.info["exptype"] = "EXTERNAL/CAL"
                    self.exp_type = EXP_CALIBRATION
                    # or should we use:  self.exp_type = EXP_SCIENCE
                else:
                    cosutil.printWarning("EXPTYPE = %s and TAGFLASH = %s "
                        "for %s;" % (self.info["exptype"],
                                     self.info["tagflash"], self.input))
                    cosutil.printContinuation(
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
                cosutil.printWarning("EXPTYPE = %s and OBSTYPE = %s "
                        "for %s;" % (self.info["exptype"],
                                     self.info["obstype"], self.input))
                cosutil.printContinuation(
                        "EXPTYPE will be changed to EXTERNAL/CAL.")
                self.info["exptype"] = "EXTERNAL/CAL"
                self.exp_type = EXP_CALIBRATION
            else:
                pass                            # no change needed

        if self.exp_type == EXP_WAVECAL and self.info["aperture"] != "WCA":
            cosutil.printWarning(
            "APERTURE = %s for a wavecal; this could be a serious error" \
                                % self.info["aperture"])

        self.checkSwitches()
        self.closeTrailer()

    def openTrailer(self, first=False):
        """Open the trailer file for this file."""

        global raw_input_trailer

        if raw_input_trailer:           # handled separately
            return

        cosutil.openTrailer(self.filenames["trl"])
        if first:
            cosutil.writeVersionToTrailer()

    def closeTrailer(self):
        """Close the trailer file for this file."""

        global raw_input_trailer

        if raw_input_trailer:           # handled separately
            return

        cosutil.closeTrailer()

    def getHeaderInfo(self):
        """Read keyword values.

        This routine gets general info from both the primary and EVENTS or SCI
        extension headers, and it gets calibration switches and reference
        file names from the primary header.

        This function also adds keys "corrtag_input" and "cal_ver" to the
        info dictionary.
        """

        fd = fits.open(self.input, mode="copyonwrite")
        phdr = fd[0].header
        try:
            hdr = fd["EVENTS"].header
        except:
            hdr = fd[("SCI",1)].header

        fd.close()

        # Each of these is a dictionary with (lower case) header keywords
        # as the keys.
        self.info = getinfo.getGeneralInfo(phdr, hdr)
        self.switches = getinfo.getSwitchValues(phdr)
        self.reffiles = getinfo.getRefFileNames(phdr)

        if self.info["life_adj"] == -1 and self.info["life_adj_offset"] != 0.:
            cosutil.printMsg("Info:  aperture plate is offset by %.2f pixels"
                             % self.info["life_adj_offset"])

        # check for ref file name "N/A"
        getinfo.resetSwitches(self.switches, self.reffiles)

        # Is the input a corrtag file?
        self.info["corrtag_input"] = cosutil.isCorrtag(self.input)

        self.info["cal_ver"] = CALCOS_VERSION

    def sanityCheck(self):
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
            cosutil.printError("Wrong coordinates for this version of calcos")
            cosutil.printContinuation("for %s" % self.input)
            raise RuntimeError()

        # Replace RelMvReq in opt_elem or aperture keywords, etc.
        self.fixRelMvReq(self.filenames["spt"], info)

        # check SEGMENT
        if info["detector"] == "FUV" and \
                (info["segment"] != "FUVA" and info["segment"] != "FUVB"):
            bad = 1
            cosutil.printError("SEGMENT = '%s' is invalid" % info["segment"])

        # check EXPTIME, EXPSTART, EXPEND
        if info["exptime"] < 0.:
            bad = 1
            cosutil.printError("EXPTIME = %g is invalid" % info["exptime"])
        # add 0.5 for roundoff error
        if (info["expend"] - info["expstart"]) * SEC_PER_DAY + 0.5 < \
                info["exptime"]:
            warn = 1
            cosutil.printWarning("(EXPEND - EXPSTART) is less than EXPTIME")

        # check OBSTYPE
        if info["obstype"] == "ACQUISITION":
            warn = 1
            cosutil.printWarning(
            "OBSTYPE = ACQUISITION, will be reset to IMAGING")
            info["obstype"] = "IMAGING"
        if info["obstype"] == "SPECTROSCOPIC" and \
               (info["opt_elem"][0:6] == "MIRROR" or
                info["opt_elem"][0:3] == "TA1"):
            bad = 1
            cosutil.printError(
            "OBSTYPE = SPECTROSCOPIC and OPT_ELEM = %s is invalid"
                 % info["opt_elem"])
        if info["obstype"] == "IMAGING":
            if info["dispaxis"] != 0:
                # not a fatal error
                warn = 1
                cosutil.printWarning(
                "DISPAXIS = %d, will be reset to 0 for imaging data" \
                             % info["dispaxis"])
                info["dispaxis"] = 0
        elif info["obstype"] == "SPECTROSCOPIC":
            if info["dispaxis"] == 2:
                warn = 1
                cosutil.printWarning("DISPAXIS = 2")
        else:
            bad = 1
            cosutil.printError(
            "OBSTYPE = '%s'; should be IMAGING or SPECTROSCOPIC" \
                         % info["obstype"])

        # check OBSMODE
        if info["obsmode"] != "TIME-TAG" and info["obsmode"] != "ACCUM":
            bad = 1
            cosutil.printError("OBSMODE = '%s'; should be TIME-TAG or ACCUM" \
                               % info["obsmode"])

        if warn or bad:
            cosutil.printContinuation("for %s" % self.input)
        if bad:
            raise RuntimeError()

    def fixRelMvReq(self, sptfile, info):
        """Replace RelMvReq in keywords with values based on OSM position.

        For thermal vac data (only), this function determines the values of
        OPT_ELEM, CENWAVE and FPOFFSET based on the OSM1 or OSM2 positions
        as given by LOM1STP or LOM2STP respectively, in the support file
        header.  If OPT_ELEM is "RelMvReq", then OPT_ELEM, CENWAVE and
        FPOFFSET will be silently replaced by the correct values.  If CENWAVE
        is unreasonably small (< 1000), it will be replaced.  Otherwise,
        these three keywords will be compared with the values determined from
        the OSM positions, and discrepancies will be noted and corrected.

        Parameters
        ----------
        sptfile: str
            Name of support file.

        info: dictionary
            Header keywords and values; values may be updated in-place by
            this function.
        """

        if info["targname"] != "Thermal_Vac":
            return

        # dictionaries with OSM step position as key and tuple of
        # optical element, central wavelength, and FP offset as value;
        # also defines date ranges for NUV and OSM range for TA1Image.
        from . import osmstep

        try:
            fd = fits.open(sptfile, mode="readonly")
            rootname = fd[0].header.get("rootname", default=NOT_APPLICABLE)
            lom1stp = int(fd[2].header.get("lom1stp", -1))
            lom2stp = int(fd[2].header.get("lom2stp", -1))
            fd.close()
        except IOError:
            cosutil.printWarning("spt file %s not found" % sptfile)
            cosutil.printContinuation("can't check OPT_ELEM, CENWAVE, FPOFFSET")
            return

        if info["detector"] == "FUV":
            osm_dict = osmstep.fuv_osm1_dict
            if lom1stp in osm_dict:
                (opt_elem_osm, cenwave_osm, fpoffset_osm) = osm_dict[lom1stp]
            else:
                cosutil.printWarning("%s has invalid LOM1STP %d" % \
                                     (sptfile, lom1stp))
                return
        else:
            # extract the day number from the rootname
            day_num = rootname[12:15]
            if len(day_num) < 1:
                cosutil.printWarning(
            "%s has invalid ROOTNAME %s for TV data" % (sptfile, rootname))
                return
            day_num = int(day_num)
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
                if lom2stp in osm_dict:
                    (opt_elem_osm, cenwave_osm, fpoffset_osm) = \
                        osm_dict[lom2stp]
                else:
                    cosutil.printWarning("%s has invalid LOM2STP %d" %
                            (sptfile, lom2stp))
                    return

        self.compareKeywords_TV(info, opt_elem_osm, cenwave_osm, fpoffset_osm)

    def compareKeywords_TV(self,
                           info, opt_elem_osm, cenwave_osm, fpoffset_osm):
        """Update keyword values in info dictionary.

        Parameters
        ----------
        info: dictionary
            Header keywords and values, modified in-place.

        opt_elem_osm: str
            Value of OPM_ELEM as determined from OSM position.

        cenwave_osm: int
            Value of CENWAVE as determined from OSM position.

        fpoffset_osm: int
            Value of FPOFFSET as determined from OSM position.
        """

        if info["opt_elem"] == "RelMvReq":
            info["opt_elem"] = opt_elem_osm
            info["cenwave"]  = cenwave_osm
            info["fpoffset"] = fpoffset_osm

        if info["cenwave"] < 1000:
            info["cenwave"] = cenwave_osm

        if info["opt_elem"] != opt_elem_osm:
            cosutil.printWarning(
    "OPT_ELEM = %s; will be replaced with %s, based on OSM position" %
                    (info["opt_elem"], opt_elem_osm))
            info["opt_elem"] = opt_elem_osm

        if info["cenwave"] != cenwave_osm:
            cosutil.printWarning(
    "CENWAVE = %d; will be replaced with %d, based on OSM position" %
                    (info["cenwave"], cenwave_osm))
            info["cenwave"] = cenwave_osm

        if info["fpoffset"] != fpoffset_osm:
            cosutil.printWarning(
    "FPOFFSET = %d; will be replaced with %d, based on OSM position" %
                    (info["fpoffset"], fpoffset_osm))
            info["fpoffset"] = fpoffset_osm

    def makeFileNames(self, suffix, outdir):
        """Create names of input and output files from input raw file names.

        These are the keys for the dictionary of file names:

          root     rootname (not including suffix or directory); note that
                     this is from the file name, not the header keyword
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

        Parameters
        ----------
        suffix: str
            An obsmode-specific string, either "_rawtag" or "_rawaccum"
            (or "_rawimage").  Note that 'suffix' excludes "_a" or "_b",
            in the case that we have FUV data.

        outdir: str
            The name of the output directory (or an empty string).

        Returns
        -------
        filenames: dictionary
            The input and output names.
        """

        input = os.path.basename(self.input)
        # This is the input file name, but in the output directory.
        output = os.path.join(outdir, input)

        rootname = getRootname(input, "_raw")

        trailer = os.path.join(outdir, rootname) + ".tra"

        x1d_x = replaceSuffix(output, suffix, "_x1d")
        flash_x = replaceSuffix(output, suffix, "_lampflash")

        # Remove the "_a" or "_b" from the x1d_x and flash_x suffixes.
        if x1d_x.endswith("_a.fits"):
            find_this = "_a.fits"
        elif x1d_x.endswith("_b.fits"):
            find_this = "_b.fits"
        else:
            find_this = None
            x1d = x1d_x[:]
            flash = flash_x[:]
        if find_this is not None:
            i = x1d_x.rfind(find_this)
            x1d = x1d_x[0:i] + ".fits"
            i = flash_x.rfind(find_this)
            flash = flash_x[0:i] + ".fits"

        filenames = {}
        filenames["root"]    = rootname
        filenames["trl"]     = trailer
        filenames["raw"]     = self.input
        filenames["pha"]     = replaceSuffix(self.input, suffix, "_pha")
        filenames["corrtag"] = replaceSuffix(output, suffix, "_corrtag")
        filenames["flt"]     = replaceSuffix(output, suffix, "_flt")
        filenames["counts"]  = replaceSuffix(output, suffix, "_counts")
        filenames["x1d_x"]   = x1d_x
        filenames["x1d"]     = x1d
        filenames["flash_x"] = flash_x
        filenames["flash"]   = flash
        filenames["csum"]    = replaceSuffix(output, suffix, "_csum")

        filenames["spt"]     = getRootname(self.input, "_raw") + "_spt.fits"

        return filenames

    def checkImSpecSwitches(self):
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
            self.overrideSwitch("photcorr", messages)

        if self.info["obstype"] == "IMAGING" or \
           self.exp_type == EXP_TARGET_ACQ or \
           self.exp_type == EXP_ACQ_IMAGE:

            self.overrideSwitch("doppcorr", messages)
            self.overrideSwitch("helcorr", messages)
            self.overrideSwitch("backcorr", messages)
            self.overrideSwitch("fluxcorr", messages)

        else:                                   # spectroscopic

            if self.info["obsmode"] == "TIME-TAG" and \
               self.info["doppmagv"] == 0.:
                self.overrideSwitch("doppcorr", messages)

            if self.info["obsmode"] == "ACCUM" and \
               self.info["dopmagt"] == 0:
                self.overrideSwitch("doppcorr", messages)

            if self.info["ra_targ"] < 0.:
                self.overrideSwitch("helcorr", messages)

            if self.switches["x1dcorr"] != "PERFORM":
                # Can't do backcorr or fluxcorr without 1-D extraction.
                self.overrideSwitch("backcorr", messages)
                self.overrideSwitch("fluxcorr", messages)
                self.overrideSwitch("tdscorr", messages)

        return messages

    def overrideSwitch(self, keyword, messages, reset_to="OMIT"):
        """If switch for keyword is "PERFORM", reset it to "OMIT".

        Parameters
        ----------
        keyword: str
            A calibration switch keyword.

        messages: str
            Tells what keywords have been changed; modified in-place.

        reset_to: str
            Value to assign to keyword (e.g. "OMIT" or "SKIPPED") in the
            switches attribute.
        """

        key_lower = keyword.lower()
        if key_lower in self.switches:
            if self.switches[key_lower] == "PERFORM":
                self.switches[key_lower] = reset_to
                messages.append(keyword.upper() + " reset to " + reset_to)
        else:
            self.switches[key_lower] = reset_to

    def printSwitchMessages(self, messages, input):
        """Print info about which calibration switches are being reset.

        Parameters
        ----------
        messages: str
            Tells what keywords have been changed.

        input: str
            Name of an input file (to be included in the text that is
            printed).
        """

        if len(messages) > 0:
            msg = "Warning:  The following calibration switch"
            if len(messages) > 1:
                msg += "es"
            msg += " will be reset as shown,"
            cosutil.printMsg(msg, VERBOSE)
            cosutil.printMsg("  for file " + input, VERBOSE)
            for msg in messages:
                cosutil.printMsg(msg, VERBOSE)

    def CheckforAddSimulatedWavecal(self, association, wavecal_info, debug=False):
        """A simulated wavecal entry needs to be inserted if these conditions are met

        Parameters:
        -----------

        association: calcos.Association object
            The association containing the Observation

        wavecal_info: List of dictionaries
            The wavecal info obtained from running calcos.Calibration.allWavecals()

        debug: boolean
            Whether to print diagnostic info as to why a virtual wavecal is or is
            not added

        Returns:
        --------

        boolean
            True if simulated wavecal is to be inserted
        """
        info = self.info
        reffiles = self.reffiles

        if info['detector'] == 'NUV':
            if debug:
                cosutil.printMsg("Don't add simulated wavecal because observation is NUV")
            return False

        if 'EXP-SWAVE' not in association.asn_info['memtype']:
            if debug:
                cosutil.printMsg("Don't add simulated wavecal because association has no exp_swave members")
            return False

        # If the other segment exists in this association and has the addsimulatedwavecal flag set,
        # set it for this obs
        othersegmentobs = self.getOtherSegmentObs(association)
        if othersegmentobs is not None:
            othersegmentinfo = othersegmentobs.info
            if othersegmentinfo['addsimulatedwavecal']:
                if debug:
                    cosutil.printMsg("Add simulated wavecal because other segment is")
                return True

        # Read info from wavecal parameters table.
        wcp_info = cosutil.getTable(reffiles["wcptab"],
                                    filter={"opt_elem": info["opt_elem"]},
                                    exactly_one=True)
        wcp_info = wcp_info[0]

        f1 = fits.open(self.input)
        try:
            events = f1[1].data
            time = events['time']
        except KeyError:
            cosutil.printMsg("Cannot get events from data extension")
            return False
        f1.close()
        if len(events) == 0:
            if debug:
                cosutil.printMsg("Don't add simulated wavecal as no events recorded")
            return False
        events_duration = time[-1] - time[0]
        mintime = self.getMinTime(wcp_info)
        if mintime is None:
            self.checkwcpinfo()
        if events_duration < mintime:
            if debug:
                cosutil.printMsg("Don't add simulated wavecal because events duration < {}".format(mintime))
            return False

        numwavecals = self.getNumWavecals(wavecal_info, wcp_info)
        if numwavecals != 2:
            if debug:
                cosutil.printMsg("Don't add simulated wavecal because #wavecals != 2")
            return False

        # If we get this far, then add simulated wavecal
        # If we set it for this segment, we need to set it for the other
        # segment, if there is one
        if othersegmentobs is not None:
            othersegmentinfo['addsimulatedwavecal'] = True
        return True

    def getOtherSegmentObs(self, association):
        othersegmentobs = None
        info = self.info
        if info["segment"] == 'FUVA':
            othersegmentletter = 'b'
        elif info["segment"] == 'FUVB':
            othersegmentletter = 'a'
        rootname = self.filenames["root"]
        othersegmentending = f"_{othersegmentletter}.fits"
        for otherobs in association.obs:
            for filename in self.filenames:
                if filename.startswith(rootname) and filename.endswith(othersegmentending):
                    othersegmentobs = otherobs
                    break
        return othersegmentobs

    def getMinTime(self, wcp_info):
        """Get the minimum exposure time that a science exposure must have
        to be eligible for a simulated wavecal

        """
        try:
            return wcp_info['min_exptime']
        except KeyError:
            return None

    def checkwcpinfo(self):
        # Report on missing columns in the wcptab reference file and exit if
        # missing columns are found
        wcp_info = cosutil.getTable(self.reffiles["wcptab"],
                                    filter={"opt_elem": self.info["opt_elem"]},
                                    exactly_one=True)
        wcp_info = wcp_info[0]

        necessary_keys = ['TCROSSOVER', 'FRACSHORT', 'FRACLONG', 'OFFSET_SHORT',
                          'OFFSET_LONG', 'MIN_EXPTIME']
        bad_keys = []
        for key in necessary_keys:
            try:
                temp = wcp_info[key]
            except KeyError:
                bad_keys.append(key)

        if len(bad_keys) != 0:
            cosutil.printError("The following columns were missing from the WCPTAB:")
            for bad_key in bad_keys:
                cosutil.printMsg(bad_key)
            raise RuntimeError("Please use an updated WCPTAB reference file")
        else:
            return None

    def getNumWavecals(self, wavecal_info, wcp_info):
        """Get the number of wavecals for this science exposure.

        Parameters:
        -----------

        events: FITS Recarray
            The table of events.  Must have a 'TIME' entry

        info: Dict
            Dictionary of COS exposure information for the science exposure

        Returns:
        --------

        int: number of wavecal exposures for this science exposure

        """
        info = self.info
        tmid = 0.5 * (info["expstart"] + info["expend"])

        shift_info = wavecal.returnWavecalShift(wavecal_info,
                                                wcp_info, info["cenwave"],
                                                info["fpoffset"], tmid)
        if shift_info is not None:
            shift_dict, slope_dict, wavecalfiles = shift_info
        else:
            return 0

        # wavecalfiles is a string containing the name(s) of wavecal files
        # used to calculate the shift and slope dicts.  If more than 1, they
        # are separated by a space
        numfiles = len(wavecalfiles.strip().split(' '))
        return numfiles

class FUVTimetagObs(Observation):

    def __init__(self, input, outdir, memtype, shift_file, first=False):

        Observation.__init__(self, input, outdir, memtype, "_rawtag",
                             shift_file, first)

    def checkSwitches(self):

        messages = self.checkImSpecSwitches()

        self.printSwitchMessages(messages, self.input)

class FUVAccumObs(Observation):

    def __init__(self, input, outdir, memtype, shift_file, first=False):

        Observation.__init__(self, input, outdir, memtype, "_rawaccum",
                             shift_file, first)

    def checkSwitches(self):

        messages = self.checkImSpecSwitches()

        # Note that this tests on DOPPONT, while the generic test in
        # checkImSpecSwitches uses DOPPMAGV (for time-tag) or
        # DOPMAGT (for accum).
        if not self.info["doppont"]:
            self.overrideSwitch("doppcorr", messages, reset_to="SKIPPED")

        self.printSwitchMessages(messages, self.input)

class NUVTimetagObs(Observation):

    def __init__(self, input, outdir, memtype, shift_file, first=False):

        Observation.__init__(self, input, outdir, memtype, "_rawtag",
                             shift_file, first)

    def checkSwitches(self):

        messages = self.checkImSpecSwitches()

        self.overrideSwitch("tempcorr", messages)
        self.overrideSwitch("geocorr", messages)
        self.overrideSwitch("dgeocorr", messages)
        self.overrideSwitch("igeocorr", messages)
        self.overrideSwitch("randcorr", messages)
        self.overrideSwitch("phacorr", messages)

        self.printSwitchMessages(messages, self.input)

class NUVAccumObs(Observation):

    def __init__(self, input, outdir, memtype, shift_file, first=False):

        Observation.__init__(self, input, outdir, memtype, "_rawaccum",
                             shift_file, first)

    def checkSwitches(self):

        messages = self.checkImSpecSwitches()

        self.overrideSwitch("tempcorr", messages)
        self.overrideSwitch("geocorr", messages)
        self.overrideSwitch("dgeocorr", messages)
        self.overrideSwitch("igeocorr", messages)
        self.overrideSwitch("randcorr", messages)
        self.overrideSwitch("phacorr", messages)
        if not self.info["doppont"]:
            self.overrideSwitch("doppcorr", messages, reset_to="SKIPPED")

        self.printSwitchMessages(messages, self.input)

class Calibration(object):
    """Calibrate COS data.

    The attributes are:
        assoc              the Association instance
        wavecal_info       list of dictionaries, each of which contains the
                             following:
                               time (MJD of middle of exposure)
                               cenwave (from the header keyword)
                               fpoffset (from the header keyword)
                               rootname
                               filename
                               shift_dict:  Keys are shift1a, shift1b,
                                 and (if NUV) shift1c; value is the shift
                                 that was determined, in pixels; positive
                                 shift means that features in the spectrum
                                 were found at larger pixel number than the
                                 nominal location.  Other keys as well.
                               fp_dict:  Keys are (segment, fpoffset),
                                 value is fp_pixel_shift
        wcp_info           matching row (just one) from the wavecal
                             parameters table

    Parameters
    ----------
    assoc: instance
        An Association object.
    """

    def __init__(self, assoc):
        """Constructor."""

        self.assoc = assoc
        self.wavecal_info = []
        self.wcp_info = None

    def basicCal(self, filenames, info, switches, reffiles):
        """Do the "basic" calibration.

        Parameters
        ----------
        filenames: dictionary
            Input and output file names.

        info: dictionary
            Values of header keywords for general information.

        switches: dictionary
            Values of header keywords for calibration switches.

        reffiles: dictionary
            Values of header keywords for reference file names.
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
            status = timetag.timetagBasicCalibration(input, None, outtag,
                        output, outcounts, outflash, outcsum,
                        self.assoc.cl_args,
                        info, switches, reffiles,
                        self.wavecal_info)
        else:
            status = accum.accumBasicCalibration(input, inpha, outtag,
                        output, outcounts, outcsum,
                        self.assoc.cl_args,
                        info, switches, reffiles,
                        self.wavecal_info)

    def allWavecals(self):
        """Process all the wavecal observations in the association."""

        status = 0

        if self.assoc.global_switches["wavecal"] != "PERFORM":
            return status

        cosutil.printMsg("Begin calibration of wavecals.", VERY_VERBOSE)

        # initial value
        any_x1dcorr = "omit"
        # First calibrate all the wavecals.
        for obs in self.assoc.obs:
            if obs.exp_type == EXP_WAVECAL:
                obs.openTrailer()
                if self.wcp_info is None:
                    # Read info from wavecal parameters table.
                    wcp_info = cosutil.getTable(obs.reffiles["wcptab"],
                               filter={"opt_elem": obs.info["opt_elem"]},
                               exactly_one=True)
                    self.wcp_info = wcp_info[0]
                try:
                    self.basicCal(obs.filenames,
                                  obs.info, obs.switches, obs.reffiles)
                    if obs.switches["x1dcorr"] == "PERFORM":
                        # Find spectrum in cross-dispersion direction.
                        # (xd_shifts and xd_locns are ignored.)
                        (shift2, xd_shifts, xd_locns, lamp_is_on) = \
                        wavecal.findWavecalSpectrum(obs.filenames["corrtag"],
                                                    obs.info, obs.reffiles)
                        # Update shift2[a-c] keywords, and possibly lampused.
                        self.setSpectrumOffset(obs.filenames,
                                               obs.info["segment"],
                                               shift2, lamp_is_on)
                        self.extractSpectrum(obs.filenames)
                        any_x1dcorr = "PERFORM"
                except (BadApertureError, MissingRowError) as e:
                    cosutil.printError("%s" % e)
                    status = BAD_APER_MISSING_ROW_EXCEPTION
                    continue
                finally:
                    obs.closeTrailer()

        w_status = 0
        if any_x1dcorr == "PERFORM":

            self.concatenateSpectra("wavecal")

            # Now find the shift of each wavecal.
            w_status = self.processWavecal()
            if w_status and not status:
                status = w_status

            # Set the shift keywords in the corrtag, flt, and counts headers
            # (already set in x1d header) for each wavecal observation.
            # Compute wavelengths and assign to the wavelength column in the
            # corrtag tables.
            for obs in self.assoc.obs:
                if obs.exp_type == EXP_WAVECAL:
                    self.setWavecalShift(obs.filenames)
                    self.corrtagWavelengths(obs.filenames["corrtag"],
                                            obs.info, obs.reffiles)

            cosutil.printMsg("wavecal_info = " + repr(self.wavecal_info),
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
                    extract.recomputeWavelengths(x1d_file)
                    obs.closeTrailer()

        if status == BAD_APER_MISSING_ROW_EXCEPTION:
            if self.assoc.asn_info["exists"]:
                cosutil.printWarning("No further auto/GO wavecal processing"
                                     " for this association (status = %d)."
                                     % status)
            else:
                cosutil.printWarning("No further processing of this wavecal"
                                     " will be done (status = %d)." % status)

        return status

    def allScience(self):
        """Process all the science observations in the association."""

        status = 0

        if self.assoc.global_switches["science"] != "PERFORM":
            return status

        cosutil.printMsg("Begin calibration of science data.", VERY_VERBOSE)
        # initial values
        any_x1dcorr = "omit"
        any_wavecorr = "omit"
        any_spectroscopic = "omit"
        spwcstab = NOT_APPLICABLE
        for obs in self.assoc.obs:
            if obs.exp_type == EXP_SCIENCE or \
               obs.exp_type == EXP_CALIBRATION or \
               obs.exp_type == EXP_ACQ_IMAGE:
                obs.openTrailer()
                # Check for whether this exposure meets the criteria for adding a simulated wavecal
                # If so, set the obs.info['addsimulatedwavecal'] entry to True
                if obs.CheckforAddSimulatedWavecal(self.assoc, self.wavecal_info, debug=True):
                    obs.info['addsimulatedwavecal'] = True
                    obs.checkwcpinfo()
                else:
                    obs.info['addsimulatedwavecal'] = False
                if not self.assoc.asn_info['exists'] and not obs.info['tagflash'] and \
                   obs.switches["wavecorr"] == "PERFORM":
                    nowavecalwarning = "\nCAUTION: You are running CalCOS with a "
                    nowavecalwarning += "rawtag or corrtag file\nthat does not "
                    nowavecalwarning += "contain simultaneous lamp data instead "
                    nowavecalwarning += "of using\nan association (asn) file as "
                    nowavecalwarning += "an input. No wavelength correction\nwill "
                    nowavecalwarning += "be applied and your wavelength "
                    nowavecalwarning += "calibration will be wrong.\nIf you wish "
                    nowavecalwarning += "to create a custom asn file for use with "
                    nowavecalwarning += "your data,\nplease refer to Chapter 3 of "
                    nowavecalwarning += "the COS Data Handbook."
                    cosutil.printMsg(nowavecalwarning)
                try:
                    self.basicCal(obs.filenames,
                                  obs.info, obs.switches, obs.reffiles)
                    self.updateShift(obs.filenames, obs.switches["wavecorr"],
                                     obs.info)
                    if obs.switches["x1dcorr"] == "PERFORM":
                        self.extractSpectrum(obs.filenames)
                        any_x1dcorr = "PERFORM"
                        any_spectroscopic = "PERFORM"
                    elif obs.info["obstype"] == "SPECTROSCOPIC":
                        cosutil.printSwitch("X1DCORR", obs.switches)
                        any_spectroscopic = "PERFORM"
                    if obs.info["obstype"] == "SPECTROSCOPIC" and \
                       obs.reffiles["spwcstab"] != NOT_APPLICABLE:
                        spwcstab = obs.reffiles["spwcstab_hdr"]
                    if obs.info["tagflash"] and \
                       obs.switches["wavecorr"] == "PERFORM":
                        any_wavecorr = "PERFORM"
                except (BadApertureError, MissingRowError) as e:
                    cosutil.printError("%s" % e)
                    status = BAD_APER_MISSING_ROW_EXCEPTION
                    continue
                finally:
                    obs.closeTrailer()

        if any_x1dcorr == "omit" and any_wavecorr == "omit" and \
           any_spectroscopic == "omit":
            if self.assoc.asn_info["exists"]:
                cosutil.printWarning("No further processing for this"
                                     " association (status = %d)." % status)
            else:
                cosutil.printWarning("No further processing for this dataset"
                                     " (status = %d)." % status)

        if any_x1dcorr == "PERFORM":
            self.concatenateSpectra("science")

        if any_wavecorr == "PERFORM":
            self.concatenateSpectra("tagflash")

        if any_spectroscopic == "PERFORM":
            if spwcstab != NOT_APPLICABLE:
                updated = False
                for obs in self.assoc.obs:
                    if obs.info["obstype"] == "SPECTROSCOPIC":
                        flag = self.writeWCS(obs.filenames, obs.info,
                                             obs.switches, obs.reffiles)
                        updated = flag or updated
                if updated:
                    cosutil.printRef("spwcstab", {"spwcstab_hdr": spwcstab})
            else:
                cosutil.printWarning("SPWCSTAB = %s, so WCS keywords will"
                        " not be updated" % spwcstab)

        return status

    def extractSpectrum(self, filenames):
        """Extract a 1-D spectrum from corrtag table or from 2-D images.

        The 1-D spectrum will be extracted from the 2-D flt and counts images.

        Parameters
        ----------
        filenames: dictionary
            Input and output file names.
        """

        input = filenames["flt"]
        incounts = filenames["counts"]
        corrtag = filenames["corrtag"]
        output = filenames["x1d_x"]

        find_target = self.assoc.cl_args["find_target"]
        extract.extract1D(input, incounts, output,
                          find_target=find_target)

        # Copy keywords from input (the flt file) to corrtag.
        extract.updateCorrtagKeywords(input, corrtag)

    def writeWCS(self, filenames, info, switches, reffiles):
        """Write the WCS header keywords for spectroscopic data.

        Parameters
        ----------
        filenames: dictionary
            Input and output file names.

        info: dictionary
            Header keywords and values for general information.

        switches: dictionary
            Calibration switches (we need helcorr).

        reffiles: dictionary
            reference file names (we need spwcstab).

        Returns
        -------
        updated: boolean
            True if keywords were actually written.
        """

        helcorr = switches["helcorr"]
        spwcstab = reffiles["spwcstab"]
        xtractab = reffiles["xtractab"]

        updated = False

        wcs = spwcs.SpWcsCorrtag(filenames["corrtag"], info, helcorr,
                                 spwcstab, xtractab)
        flag = wcs.writeWCSKeywords()
        if flag:
            updated = True
            cosutil.printMsg("WCS keywords were written for %s" %
                             (filenames["corrtag"]), VERY_VERBOSE)

        wcs = spwcs.SpWcsImage(filenames["flt"], info, helcorr,
                               spwcstab, xtractab)
        flag = wcs.writeWCSKeywords()
        if flag:
            updated = True
            cosutil.printMsg("WCS keywords were written for %s" %
                             (filenames["flt"]), VERY_VERBOSE)

        wcs = spwcs.SpWcsImage(filenames["counts"], info, helcorr,
                               spwcstab, xtractab)
        flag = wcs.writeWCSKeywords()
        if flag:
            updated = True
            cosutil.printMsg("WCS keywords were written for %s" %
                             (filenames["counts"]), VERY_VERBOSE)

        return updated

    def processWavecal(self):
        """Determine shift from wavecal observation.

        The shift and related info will be appended to the wavecal_info list.

        Returns
        -------
        status: int
            0 if OK; BAD_APER_MISSING_ROW_EXCEPTION if the try/except block
            catches a bad aperture or missing row.
        """

        cosutil.printSwitch("WAVECORR", {"wavecorr": "PERFORM"})
        status = 0

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
                    wavecal.printWavecalRef(obs.reffiles)
                    first = False
                cosutil.printFilenames([("Input", x1d_file)])
                try:
                    wavecal_stuff = wavecal.findWavecalShift(x1d_file,
                                    self.assoc.cl_args["shift_file"],
                                    obs.info, self.wcp_info)
                except (BadApertureError, MissingRowError) as e:
                    cosutil.printError("%s" % e)
                    obs.closeTrailer()
                    status = BAD_APER_MISSING_ROW_EXCEPTION
                    obs.closeTrailer()
                    continue

                if wavecal_stuff is not None:
                    (shift_dict, fp_dict) = wavecal_stuff
                    # time is the MJD at the midpoint of the exposure.
                    time = cosutil.timeAtMidpoint(obs.info)
                    wavecal.storeWavecalInfo(self.wavecal_info,
                            time, obs.info["cenwave"], obs.info["fpoffset"],
                            shift_dict, fp_dict,
                            obs.filenames["root"], obs.filenames["raw"])
                obs.closeTrailer()

        return status

    def updateShift(self, filenames, wavecorr, info):
        """Update the shift keywords in corrtag, flt, counts headers.

        This function is only relevant for ACCUM mode data.
        The shift for the two segments (or three NUV stripes) will be copied
        (or interpolated) from the list of wavecal information to the
        keywords SHIFT1A, SHIFT1B, SHIFT1C, SHIFT2A, SHIFT2B, SHIFT2C.

        Parameters
        ----------
        filenames: dictionary
            Input and output file names.

        wavecorr: str
            "PERFORM" if wavecal processing is being done.

        info: dictionary
            Header keywords and values for general information.
        """

        if info["obsmode"] == "TIME-TAG":
            return
        if info["exptype"] == "ACQ/IMAGE":
            return

        if wavecorr != "PERFORM" and wavecorr != "COMPLETE":
            shift_dict = None
        elif len(self.wavecal_info) < 1:
            shift_dict = None
        else:
            shift_dict = None                           # replaced below
            time = cosutil.timeAtMidpoint(info)         # MJD
            shift_info = wavecal.returnWavecalShift(self.wavecal_info,
                         self.wcp_info, info["cenwave"], info["fpoffset"],
                         time)
            if len(self.wavecal_info) > 0 and shift_info is not None:
                # only the shift will be used, not the slope or the file name
                (shift_dict, slope_dict, wavecal_filename) = shift_info
                cosutil.printSwitch("WAVECORR", {"wavecorr": "PERFORM"})
                if cosutil.checkVerbosity(VERY_VERBOSE):
                    # keywords = shift_dict.keys()
                    # keywords.sort()
                    keywords = sorted(shift_dict)
                    for key in keywords:
                        cosutil.printMsg(
                            "  %s = %.4f" % (key.upper(), shift_dict[key]),
                            VERY_VERBOSE)

        if shift_dict is None:
            cosutil.printMsg(
                "Warning:  No wavecal info; shift assumed to be 0.", VERBOSE)

        # corrtag is in this list because there might be an output file for
        # the pseudo-corrtag table.
        for fname in [filenames["corrtag"], \
                      filenames["flt"], filenames["counts"]]:
            if os.access(fname, os.R_OK):
                fd = fits.open(fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd[("SCI",1)].header
                if wavecorr == "PERFORM" and len(self.wavecal_info) > 0:
                    phdr["WAVECORR"] = "COMPLETE"
                hdr["DPIXEL1A"] = 0.            # dpixel1 not used for ACCUM
                hdr["DPIXEL1B"] = 0.
                if info["detector"] == "NUV":
                    hdr["DPIXEL1C"] = 0.
                if shift_dict is None:
                    hdr["SHIFT1A"] = 0.
                    hdr["SHIFT1B"] = 0.
                    hdr["SHIFT2A"] = 0.
                    hdr["SHIFT2B"] = 0.
                    if info["detector"] == "NUV":
                        hdr["SHIFT1C"] = 0.
                        hdr["SHIFT2C"] = 0.
                else:
                    for key in shift_dict.keys():
                        shift = shift_dict[key]
                        shift = round(shift, 4)
                        hdr[key] = shift
                fd.close()

    def setSpectrumOffset(self, filenames, segment, shift2, lamp_is_on):
        """Update the shift2 keywords in corrtag, flt, counts headers.

        This function is called only for a wavecal, not for a science
        observation.  (For science data, the shift2 keyword(s) will be
        updated by updateShift.)

        Parameters
        ----------
        filenames: dictionary
            Input and output file names.

        segment: str
            FUV segment name or NUV stripe name.

        shift2: float
            Offset in cross-dispersion direction, as determined from
            (conventional) wavecal data.

        lamp_is_on: boolean
            True if the wavecal lamp was actually on.
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
            if os.access(fname, os.R_OK):
                fd = fits.open(fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd[("SCI",1)].header
                for keyword in keywords:
                    hdr[keyword] = round(shift2, 4)
                lampused = phdr.get("lampused", "missing")
                lampplan = phdr.get("lampplan", "missing")
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
            cosutil.printWarning("The wavecal lamp was on, but LAMPUSED = " \
                                 "%s and LAMPPLAN is missing." % \
                                 lampused, level=VERBOSE)
        if print_msg2:
            cosutil.printMsg("LAMPUSED = %s, which is incorrect; " \
                             "the value will be reset to %s." % \
                             (lampused, lampplan), level=VERBOSE)
        if print_msg3:
            cosutil.printWarning("The wavecal lamp was off for a wavecal!", \
                                 level=VERBOSE)
        if print_msg4:
            cosutil.printMsg("LAMPUSED = %s, and it will be reset to NONE." \
                             % lampused, level=VERBOSE)

    def setWavecalShift(self, filenames):
        """Update the shift keywords in corrtag, flt, counts headers.

        This function is called only for a wavecal, not for a science
        observation.  There must be an exact match with the rootname of the
        observation.

        Parameters
        ----------
        filenames: dictionary
            Input and output file names.
        """

        shift_dict = wavecal.returnExactMatch(self.wavecal_info,
                                              filenames["root"])
        if shift_dict is None:
            return

        for fname in [filenames["corrtag"], \
                      filenames["flt"], filenames["counts"]]:
            if os.access(fname, os.R_OK):
                fd = fits.open(fname, mode="update")
                phdr = fd[0].header
                try:
                    hdr = fd["EVENTS"].header
                except:
                    hdr = fd[("SCI",1)].header
                phdr["WAVECORR"] = "COMPLETE"
                for keyword in shift_dict.keys():
                    shift = shift_dict[keyword]
                    hdr[keyword] = round(shift, 4)
                fd.close()

    def corrtagWavelengths(self, corrtag, info, reffiles):
        """Compute and assign wavelengths in the corrtag table.

        This function is only called for a wavecal (auto or GO).  For science
        exposures, the wavelengths are assigned during time-tag processing.

        Parameters
        ----------
        filenames: dictionary
            Input and output file names.
        """

        if os.access(corrtag, os.R_OK):
            fd = fits.open(corrtag, mode="update")
            events = fd["EVENTS"].data
            hdr = fd["EVENTS"].header
            timetag.computeWavelengths(events, info, reffiles,
                                       helcorr="OMIT", hdr=hdr)
            fd.close()

    def mergeKeywords(self):
        """Copy segment-specific keywords between FUV pairs of files.

        Keywords that have different names for the A and B segments will be
        copied from flt_a to flt_b and vice versa, and similarly for the
        counts files.
        """

        # segment_specific_keywords is in calcosparam.py.  The strings in
        # this list use "X" as a character to be replaced by "a" or "b" to
        # get lists a_kwds and b_kwds respectively.
        a_kwds = []
        b_kwds = []
        for keyword in segment_specific_keywords:
            a_kwds.append(keyword.replace("X", "a"))
            b_kwds.append(keyword.replace("X", "b"))

        for files in self.assoc.merge_kwds:
            assert len(files) == 2
            # If either file doesn't exist, there's nothing to do.
            if not os.access(files[0], os.R_OK) or \
               not os.access(files[1], os.R_OK):
                return
            files.sort()
            fd_a = fits.open(files[0], mode="update")
            fd_b = fits.open(files[1], mode="update")
            hdr_a = fd_a[1].header
            hdr_b = fd_b[1].header
            for i in range(len(a_kwds)):
                keyword_a = a_kwds[i]
                keyword_b = b_kwds[i]
                if keyword_a in hdr_a:
                    hdr_b[keyword_a] = hdr_a[keyword_a]
                if keyword_b in hdr_b:
                    hdr_a[keyword_b] = hdr_b[keyword_b]
            # concatenate comments for keyword GSAGTAB
            extract.updateGsagComment(fd_a[0].header, fd_b[0].header,
                                      [fd_a[0].header, fd_b[0].header])
            fd_a.close()
            fd_b.close()

    def concatenateSpectra(self, type):
        """Concatenate two 1-D FUV spectra into one spectrum.

        If type="wavecal", this routine will concatenate pairs of wavecal
        files; if type="science", this routine will concatenate pairs of
        science files.  The input _x1d_a and _x1d_b will then be deleted,
        if save_temp_files = False.

        Parameters
        ----------
        type: str {"science", "wavecal", "tagflash", "unknown"}
            Type of file to be concatenated, used as a dictionary key.
        """

        for one_set in self.assoc.concat:

            if one_set["type"] == type:

                infiles = one_set["input"]
                output = one_set["output"]
                if len(infiles) < 1:
                    continue

                if len(infiles) == 1:
                    if os.access(infiles[0], os.R_OK) and infiles[0] != output:
                        cosutil.renameFile(infiles[0], output)
                    else:
                        continue
                else:
                    extract.concatenateFUVSegments(infiles, output)
                    if not self.assoc.cl_args["save_temp_files"]:
                        # Delete the _x1d_a.fits and _x1d_b.fits files.
                        for file in infiles:
                            if os.access(file, os.R_OK):
                                os.remove(file)

    def combineToProduct(self):
        """Average the calibrated files, producing the product files."""

        if self.assoc.product is None:
            return

        combine = self.assoc.combine

        if "flt" in combine:
            self.combineFlt()

        i = self.assoc.first_science
        if self.assoc.obs[i].switches["x1dcorr"] != "PERFORM":
            return

        # Average all the x1d spectroscopic exposures in the association.
        self.combineAllX1D()

        # If we have more than one spectroscopic exposure in the association,
        # average x1d files that have the same fppos index.
        if "x1d" in combine:
            x1d_list = combine["x1d"]
            fppos_list = combine["fppos"]
            fppos_list_copy = copy.copy(fppos_list)
            fppos_list_copy.sort()
            fppos_max = fppos_list_copy[-1]
            for fppos in range(1, fppos_max+1):
                # extract subset for current osm position
                x1d_subset = []
                for i in range(len(x1d_list)):
                    if fppos_list[i] == fppos:
                        x1d_subset.append(x1d_list[i])
                if len(x1d_subset) > 0:
                    self.combineX1Di(x1d_subset, fppos)

    def combineFlt(self):
        """Average image mode data."""

        combine = self.assoc.combine

        if "flt" in combine:
            output = self.fltProductName()
            average.avgImage(combine["flt"], output)

    def combineAllX1D(self):
        """Average x1d data for all OSM positions."""

        combine = self.assoc.combine

        if "x1d" in combine:
            output = self.x1dProductName(0)
            fpavg.fpAvgSpec(combine["x1d"], output)

    def combineX1Di(self, input, fppos):
        """Average the x1d data for one specified FPPOS position.

        Parameters
        ----------
        input: list of str
            List of names of input files.

        fppos: int {1, 2, 3, 4}
            Value of header keyword FPPOS.
        """

        output = self.x1dProductName(fppos)

        fpavg.fpAvgSpec(input, output)

    def fltProductName(self):
        """Construct the product name for the flt file.

        Returns
        -------
        output: str
            Name of output flt file.
        """

        output = self.assoc.product + "_fltsum.fits"

        return output

    def x1dProductName(self, fppos=0):
        """Construct the product name for the x1d file.

        If fppos is greater than zero, then the output file name will be of
        the form "rootname_x1dsum1.fits", where the number appended to "x1dsum"
        will be the value of fppos.

        Parameters
        ----------
        fppos: int {0, 1, 2, 3, 4}
            Value of header keyword FPPOS.

        Returns
        -------
        output: str
            Name of output x1d file.
        """

        output = self.assoc.product + "_x1dsum"
        if fppos > 0:
            output += str(fppos) + ".fits"
        else:
            output += ".fits"

        return output

if __name__ == "__main__":

    main(sys.argv[1:])
