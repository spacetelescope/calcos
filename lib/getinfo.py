import os
import string
import pyfits
import cosutil
from calcosparam import *

def initialInfo (filename):
    """Get DETECTOR, OBSMODE, and EXPTYPE from the primary header.

    argument:
    filename      name of input file
    """

    fd = pyfits.open (filename, mode="readonly")
    phdr = fd[0].header

    info = {}

    if phdr.has_key ("DETECTOR"):
        detector = phdr["DETECTOR"]
    else:
        raise RuntimeError, \
        "File " + filename + " does not have DETECTOR keyword."

    if phdr.has_key ("OBSMODE"):
        obsmode = phdr["OBSMODE"]
    else:
        raise RuntimeError, \
        "File " + filename + " does not have OBSMODE keyword."

    if phdr.has_key ("EXPTYPE"):
        exptype = phdr["EXPTYPE"]
    else:
        raise RuntimeError, \
        "File " + filename + " does not have EXPTYPE keyword."

    if detector != "FUV" and detector != "NUV":
        raise ValueError, \
        "File " + filename + " has invalid DETECTOR = " + detector

    if obsmode != "TIME-TAG" and obsmode != "ACCUM":
        raise ValueError, \
        "File " + filename + " has invalid OBSMODE = " + obsmode

    info["detector"] = detector
    info["obsmode"] = obsmode
    info["exptype"] = exptype

    fd.close()

    return info

def getGeneralInfo (phdr, hdr):
    """Get keyword values from the primary and extension header.

    The input argument phdr is the primary header, and the second hdr is
    the first extension header, as provided by the pyfits module.  The
    function value is a dictionary of keyword = value pairs.  If a keyword
    is missing from the header, it will still be included in the dictionary,
    but its value will be set to the NOT_APPLICABLE string, or a reasonable
    default for keyword values that are not text strings.  npix (a tuple
    giving the output image size) will also be assigned.  If the data
    portion is empty (based on the NAXIS keyword), the value will be (0,);
    otherwise, the value will be assigned an appropriate value for the
    detector, rather than being read directly from the header.  The
    heliocentric velocity will be initialized to zero.

    arguments:
    phdr          primary header
    hdr           extension header
    """

    info = {}

    # Get keywords from the primary header.

    # FUV or NUV
    info["detector"] = phdr["DETECTOR"]
    # Set output image size (variables defined in calcosparam.py).
    if info["detector"] == "FUV":
        info["npix"] = (FUV_Y, FUV_X)
    else:
        info["npix"] = (NUV_Y, NUV_X)

    # Replace the value for npix if there's no data (based on extension header).
    if hdr["NAXIS"] == 0:
        info["npix"] = (0,)
    elif hdr["NAXIS"] == 2 and hdr["NAXIS2"] == 0:
        info["npix"] = (0,)

    # Assign an initial value for the heliocentric velocity
    info["v_helio"] = 0.

    # This is a list of primary header keywords and default values.
    keylist = {
        "segment":  NOT_APPLICABLE,
        "obstype":  NOT_APPLICABLE,
        "obsmode":  NOT_APPLICABLE,
        "exptype":  NOT_APPLICABLE,
        "opt_elem": NOT_APPLICABLE,
        "targname": NOT_APPLICABLE,
        "tagflash":  False,
        "cenwave":   0,
        "randseed": -1,
        "fppos":     1,
        "fpoffset":  0,
        "coscoord":  DETECTOR_COORDINATES,
        "ra_targ":  -999.,
        "dec_targ": -999.}

    for key in keylist.keys():
        info[key] = phdr.get (key, default=keylist[key])

    info["aperture"] = cosutil.getApertureKeyword (phdr, truncate=1)

    if info["tagflash"] == TAGFLASH_AUTO:
        info["tagflash"] = True
        info["tagflash_type"] = TAGFLASH_TYPE_AUTO
    elif info["tagflash"] == TAGFLASH_UNIFORMLY_SPACED:
        info["tagflash"] = True
        info["tagflash_type"] = TAGFLASH_TYPE_UNIFORMLY_SPACED
    else:
        info["tagflash"] = False
        info["tagflash_type"] = TAGFLASH_TYPE_NONE

    #if info["obstype"] == "SPECTROSCOPY":
    #    info["obstype"] = "SPECTROSCOPIC"
    #    cosutil.printWarning ("OBSTYPE = SPECTROSCOPY" \
    #                          " has been changed to SPECTROSCOPIC")

    # Engineering keywords relevant to deadtime correction.

    if info["detector"] == "FUV":
        if info["segment"] == "FUVA":
            info["countrate"] = phdr.get ("DEVENTA", default=0.)
        else:
            info["countrate"] = phdr.get ("DEVENTB", default=0.)
    else:
        info["countrate"] = phdr.get ("MEVENTS", default=0.)

    # Now get keywords from the extension header.

    if info["detector"] == "FUV":
        info["stimrate"] = hdr.get ("STIMRATE", default=0.)
    else:
        info["stimrate"] = 0.

    # This is a list of extension header keywords and default values.
    keylist = {
        "dispaxis":  0,
        "tc2_2":     1.,        # dispersion for spectroscopic data
        "sdqflags":  32767,
        "numflash":  0,
        "exptime":  -1.,
        "expstart": -1.,
        "expend":   -1.,
        "doppon":    False,
        "doppont":   False,
        "doppmagv": -1.,
        "dopmagt":  -1.,
        "doppzero": -1.,
        "dopzerot": -1.,
        "orbitper": -1.,
        "orbtpert": -1.}

    for key in keylist.keys():
        info[key] = hdr.get (key, default=keylist[key])

    if info["tagflash"] and info["numflash"] < 1:
        info["tagflash"] = False

    return info

def getSwitchValues (phdr):
    """Get calibration switch values from the primary header.

    The input argument phdr is the primary header, as provided by the fits
    module.  The function value is a dictionary of keyword = value pairs.
    Note that the keyword values will be converted to upper case.  If a
    keyword is missing from the header, it will still be included in the
    dictionary, but its value will be set to the NOT_APPLICABLE string.

    argument:
    phdr          primary header
    """

    switches = {}

    for key in ["dqicorr", "randcorr", "tempcorr", "geocorr", "igeocorr",
                "deadcorr", "flatcorr", "doppcorr", "helcorr", "phacorr",
                "brstcorr", "badtcorr", "x1dcorr", "wavecorr", "backcorr",
                "fluxcorr", "tdscorr", "statflag"]:
        switches[key]  = cosutil.getSwitch (phdr, key)

    return switches

def getRefFileNames (phdr):
    """Get reference file names from the primary header.

    The input argument phdr is the primary header, as provided by the pyfits
    module.  The function value is a dictionary of keyword = value pairs.
    If a keyword is missing from the header, it will still be included in
    the dictionary, but its value will be set to the NOT_APPLICABLE string.
    If the name includes an environment variable (Unix-style or IRAF-style),
    the name will be expanded to a complete pathname.  Keys of the form
    "bpixtab_hdr" (for example) are the values read directly from the
    header, while keys of the form "bpixtab" have been translated to full
    path names (operating system dependent).

    argument:
    phdr          primary header
    """

    reffiles = {}

    for key in ["flatfile", "bpixtab", "brftab", "geofile",
                "deadtab", "phatab", "brsttab", "badttab",
                "xtractab", "lamptab", "disptab", "phottab",
                "wcptab", "tdstab"]:
        reffiles[key+"_hdr"] = phdr.get (key, default=NOT_APPLICABLE)
        reffiles[key] = cosutil.expandFileName (reffiles[key+"_hdr"])

    return reffiles

def resetSwitches (switches, reffiles):
    """Reset calibration switches if required reference file is "N/A".

    If a calibration step needs one or more reference files, and if the
    name of any such file is given in the header as "N/A", the calibration
    step cannot be done.  This function checks some steps and resets the
    switch from PERFORM to SKIPPED if a required reference file is "N/A".

    @param switches: keyword and value for calibration switches
    @type switches: dictionary
    @param reffiles: keyword and value for reference file names
    @type reffiles: dictionary
    """

    check_these = {"badtcorr": ["badttab"],
                   "tdscorr": ["tdstab"]}
    #check_these = {"badtcorr": ["badttab"],
    #               "brstcorr": ["brsttab"],
    #               "dqicorr": ["bpixtab"],
    #               "flatcorr": ["flatfile"],
    #               "deadcorr": ["deadtab"],
    #               "geocorr": ["geofile"],
    #               "x1dcorr": ["xtractab", "disptab"],
    #               "fluxcorr": ["phottab"]}

    for switch_key in check_these.keys():
        not_specified = []
        if switches[switch_key] == "PERFORM":
            for reffile_key in check_these[switch_key]:
                if reffiles[reffile_key] == NOT_APPLICABLE:
                    not_specified.append (reffile_key)
        if not_specified:
            switches[switch_key] = "SKIPPED"
            cosutil.printWarning ("%s will be set to SKIPPED because" %
                                  switch_key.upper())
            for (i, reffile_key) in enumerate (not_specified):
                keyword = reffile_key.upper()
                if i == 0:
                    message = "%s = %s" % (keyword, NOT_APPLICABLE)
                else:
                    message += ", %s = %s" % (keyword, NOT_APPLICABLE)
            cosutil.printContinuation (message)
