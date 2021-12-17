from __future__ import absolute_import, division         # confidence high
import math
import os
import numpy as np
from scipy import signal as scipysignal
from scipy import ndimage
import astropy.io.fits as fits
from .calcosparam import *
from . import ccos
from . import cosutil
from . import findshift1
from . import shiftfile

# Each element of wavecal_info is a dictionary with the keys:
# "time", "cenwave", "fpoffset", "shift_dict", "rootname", "filename".
# The variable name wc_dict is used for one element of wavecal_info.

def findWavecalShift(input, shift_file, info, wcp_info):
    """Find the shift from a wavecal image.

    Note that the input file will be opened read-write, and several
    keywords (e.g. SHIFT1A, SHIFT1B, and SHIFT1C if NUV) will be updated
    in the header.

    Parameters
    ----------
    input: str
        Name of an x1d FITS file containing a wavecal observation.

    shift_file: str
        Name of a user-supplied file to override shifts, or None.

    info: dictionary
        Keywords and values from the headers of the input file.

    wcp_info: array_like
        Data (one row) from the wavecal parameters table.

    Returns
    -------
    (shift_dict, fp_dict)
        shift_dict is a dictionary or None.  Keys are header keywords (but
        lower case) for wavecal shift information, values are the values
        for updating the header.  shift_dict will be None if there is no
        data in the first extension of input.
        fp_dict is a dictionary.  Keys are tuples of segment or stripe
        name (upper case) and fpoffset.  Values are fp_pixel_shift.  This
        is only needed for the case that there is a science exposure with
        no wavecal taken at the same fpoffset, but there is a wavecal for
        at least one fpoffset.
    """

    # Note that we open the x1d table read-write, so we can update keywords.
    fd = fits.open(input, mode="update")
    phdr = fd[0].header
    sci_extn = fd["SCI"]
    if sci_extn.data is None or len(sci_extn.data) == 0:
        fd.close()
        return None

    reffiles = {}
    lamptab = cosutil.expandFileName(phdr["lamptab"])
    reffiles["lamptab"] = lamptab

    wcptab = cosutil.expandFileName(phdr["wcptab"])
    reffiles["wcptab"] = wcptab

    x_offset = sci_extn.header.get("x_offset", default=0)
    info["x_offset"] = x_offset

    nrows = sci_extn.data.shape[0]
    gross = sci_extn.data.field("gross")
    nelem = len(gross[0])
    exptime = sci_extn.data.field("exptime")
    segment = sci_extn.data.field("segment")

    detector = info["detector"]
    xc_range = wcp_info.field("xc_range")
    stepsize = wcp_info.field("stepsize")
    try:
        search_offset = wcp_info.field("search_offset")
    except KeyError:
        search_offset = 0.

    # Replace shift values in segment B with values from segment A?
    if "FUVB" in segment:
        override_segment_B = cosutil.checkForNoWavecalData(
                info["opt_elem"], info["cenwave"], "FUVB", lamptab)
    else:
        override_segment_B = False
    segment_A_present = ("FUVA" in segment)

    # segment will be added to the filter in the loop below.
    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"]}
    # If the FPOFFSET column is present in the lamptab, include fpoffset
    # in the filter.  If not, use an offset when doing the cross correlation.
    if cosutil.findColumn(lamptab, "fpoffset"):
        filter["fpoffset"] = info["fpoffset"]
    # initial_offset is used as the center of the search range, when
    # matching the spectrum to the template.
    got_pixel_shift = cosutil.findColumn(lamptab, "fp_pixel_shift")
    if got_pixel_shift:
        initial_offset = search_offset
    else:
        initial_offset = info["fpoffset"] * stepsize + search_offset

    shift_dict = {}

    index = segment.argsort()
    save_spectra = {}
    save_templates = {}
    fp_pixel_shift = {}
    fp_dict = {}

    for row in index:
        filter["segment"] = segment[row]
        lamp_info = cosutil.getTable(lamptab, filter)
        if lamp_info is None:
            raise MissingRowError("Missing row in LAMPTAB; filter = %s" %
                                  str(filter))
        # Save gross, but convert from count rate back to counts.
        save_spectra[segment[row]] = gross[row] * exptime[row]
        raw_template = lamp_info.field("intensity")[0]
        save_templates[segment[row]] = \
                cosutil.getTemplate(raw_template, x_offset, nelem)
        if got_pixel_shift:
            fp_pixel_shift[segment[row]] = \
                    lamp_info.field("fp_pixel_shift")[0]
        else:
            fp_pixel_shift[segment[row]] = 0.
        # Add elements to fp_dict, one for each fpoffset.
        readFpPixelShift(info, lamptab, segment[row], stepsize, fp_dict)

    # find offset in dispersion direction
    fs1 = findshift1.Shift1(save_spectra, save_templates, info, reffiles,
                            xc_range, fp_pixel_shift, initial_offset)
    fs1.findShifts()

    if override_segment_B:
        if segment_A_present:           # copy shift1 from A to B
            fs1.setShift1("FUVB", fs1.getShift1("FUVA"))
        else:
            fs1.setShift1("FUVB", 0., fp=False)

    # Did the user supply a file with overrides for the shifts?
    if shift_file is None:
        user_shifts = None
    else:
        user_shifts = shiftfile.ShiftFile(shift_file,
                                          info["root"], info["fpoffset"])

    if detector == "FUV":
        cosutil.printMsg("segment   shift    err  [orig.]    FP   chi sq (n)",
                         VERBOSE)
        cosutil.printMsg("-------  --------- ---- ------- ------  ----------",
                         VERBOSE)
    else:
        cosutil.printMsg("stripe    shift    err  [orig.]    FP   chi sq (n)",
                         VERBOSE)
        cosutil.printMsg("------   --------- ---- ------- ------  ----------",
                         VERBOSE)

    # Print and save results.
    for row in index:

        # This is the offset in the cross-dispersion direction.
        key = "shift2" + segment[row][-1].lower()
        if override_segment_B and segment[row] == "FUVB":
            if segment_A_present:       # copy shift2 from A to B
                shift_dict[key] = sci_extn.header.get("shift2a", default=0.)
            else:
                shift_dict[key] = 0.
        else:
            shift_dict[key] = sci_extn.header.get(key, default=0.)
        sci_extn.header[key] = shift_dict[key]

        # Zero for dpixel1[a-c] is appropriate for auto/GO wavecals.
        key = "dpixel1" + segment[row][-1].lower()
        sci_extn.header[key] = 0.
        shift_dict[key] = 0.

        key = "shift1" + segment[row][-1].lower()
        user_specified = False
        if user_shifts is not None:
            ((user_shift1, user_shift2), nfound) = \
                        user_shifts.getShifts(("any", segment[row]))
            if user_shift1 is not None:
                fs1.setShift1(segment[row], user_shift1)
                user_specified = True
        shift_segment = fs1.getShift1(segment[row])
        scatter = fs1.getScatter(segment[row])
        measured_shift1 = fs1.getMeasuredShift1(segment[row])
        fp_pixel_shift_seg = fs1.getFpPixelShift(segment[row])
        sci_extn.header[key] = shift_segment
        shift_dict[key] = shift_segment

        message = " %4s    %9.4f %4.2f [%5.1f] %6.1f  %6.1f (%d)" % \
                (segment[row], shift_segment, scatter, measured_shift1,
                 fp_pixel_shift_seg,
                 fs1.getChiSq(segment[row]), fs1.getNdf(segment[row]))
        if user_specified:
            message = message + "  # user-specified"
        elif override_segment_B and segment[row] == "FUVB":
            if segment_A_present:
                message = message + "  # based on FUVA value"
            else:
                message = message + "  # set to default (no FUVA)"
        elif not fs1.getSpecFound(segment[row]):
            message = message + "  # not found"
        cosutil.printMsg(message, VERBOSE)

        # Save Chi square and the number of degrees of freedom in the
        # shift dictionary, and update the keywords in the x1d header.
        key = "chi_sq_" + segment[row][-1].lower()
        shift_dict[key] = round(fs1.getChiSq(segment[row]), 1)
        sci_extn.header[key] = shift_dict[key]
        key = "ndf_" + segment[row][-1].lower()
        shift_dict[key] = fs1.getNdf(segment[row])
        sci_extn.header[key] = shift_dict[key]

    fd.close()

    return (shift_dict, fp_dict)

def readFpPixelShift(info, lamptab, segment, stepsize, fp_dict):
    """Read all four fp_pixel_shift values from the lamptab.

    Parameters
    ----------
    info: dictionary
        Keywords and values from the headers of the input file.

    lamptab: str
        Name of the template lamp table.

    segment: str
        Current segment or stripe name.

    stepsize: int
        Offset in pixels corresponding to one step of the OSM.  This is
        only used if the lamptab does not have FPOFFSET and FP_PIXEL_SHIFT
        columns.

    fp_dict: dictionary
        Updated in-place.  The keys are tuples of (segment, fpoffset),
        the segment or stripe name (upper case) and the integer offset
        (-2, -1, 0, 1) of the OSM.  Entries for the current segment and
        all fpoffset in the lamptab will be added to fp_dict.
    """

    # We expect to get four rows matching this filter, one row for each
    # fpoffset.
    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "segment": segment}
    lamp_info = cosutil.getTable(lamptab, filter)
    if lamp_info is None:
        raise MissingRowError("Missing row in LAMPTAB; filter = %s" %
                              str(filter))

    names = []
    for name in lamp_info.names:
        names.append(name.lower())
    if "fpoffset" in names and "fp_pixel_shift" in names:
        fpoffset = lamp_info.field("fpoffset")
        fp_pixel_shift = lamp_info.field("fp_pixel_shift")
        for i in range(len(fpoffset)):
            fp_dict[(segment, fpoffset[i])] = fp_pixel_shift[i]
    else:
        for i in [-2, -1, 0, 1]:
            fp_dict[(segment, i)] = float(i * stepsize)

def storeWavecalInfo(wavecal_info, time, cenwave, fpoffset,
                     shift_dict, fp_dict,
                     rootname, filename):
    """Append the current info to the wavecal_info list.

    shift_dict can have any of the following keys:
        "shift1a" for FUV segment A or NUV stripe A,
        "shift1b" for FUV segment B or NUV stripe B,
        "shift1c" for NUV stripe C,
        "shift2a", "shift2b", "shift2c",
        "dpixel1a", "dpixel1b", "dpixel1c",
        "chi_sq_a", "chi_sq_b", "chi_sq_c",
        "ndf_a", "ndf_b", "ndf_c".
    shift1[abc] is the measured shift in the dispersion direction, and
    shift2[abc] is the measured shift in the cross-dispersion direction.

    fp_dict has (segment, fpoffset) as key.  segment is either a segment
    or stripe name (upper case), and fpoffset is an int, -2, -1, 0 or 1.
    The values are fp_pixel_shift (from that column in the lamptab).

    The input information will be combined into a dictionary, which will
    then be appended to wavecal_info.  wavecal_info will be sorted in
    increasing order of time.

    Parameters
    ----------
    wavecal_info: list of wavecal information dictionaries
        The other arguments will be used to create a wavecal information
        dictionary, and that will be appended to wavecal_info.
        A wavecal information dictionary has keys "time", "fpoffset",
        "shift_dict", "rootname", and "filename".  The values are the
        arguments with those names, except that if filename includes a
        directory, that will be removed before saving in the dictionary.

    time: float
        Time of observation, MJD at middle of exposure

    cenwave: int
        Central wavelength, used to select entries from wavecal_info

    fpoffset: int
        OSM position, used to select entries from wavecal_info

    shift_dict: dictionary
        A dictionary of keyword names and shifts

    fp_dict: dictionary
        Dictionary of fp_pixel_shift for each key (segment, fpoffset).

    rootname: str
        The rootname (typically lower case) of the observation

    filename: str
        The name of the raw file, including directory; for FUV this will
        typically be the _a.fits name
    """

    wc_dict = {}
    wc_dict["time"]       = time
    wc_dict["cenwave"]    = cenwave
    wc_dict["fpoffset"]   = fpoffset
    wc_dict["shift_dict"] = shift_dict
    wc_dict["fp_dict"]    = fp_dict
    wc_dict["rootname"]   = rootname
    wc_dict["filename"]   = os.path.basename(filename)

    wavecal_info.append(wc_dict)
    if len(wavecal_info) > 1:
        wavecal_info.sort(key=keyTime)

def keyTime(wc_dict):
    """Return the time in a wavecal_info entry.

    Parameters
    ----------
    wc_dict: dictionary
        A wavecal information dictionary (one element of wavecal_info)

    Returns
    -------
    float
        The element of wc_dict with key "time"
    """

    return wc_dict["time"]

def returnWavecalShift(wavecal_info, wcp_info, cenwave, fpoffset, time):
    """Return the matching shift dictionary from wavecal_info.

    The element of wavecal_info that matches cenwave and fpoffset will be
    extracted.  If there are multiple entries that match, we'll find the
    two that are closest to the time of the observation and linearly
    interpolate the shifts at that time.

    If there is no entry in wavecal_info for the current cenwave and
    fpoffset, we'll find the entry that is closest in time to the time of
    the observation.  If the difference in time for that entry is not too
    large (based on info from the wavecal parameters table), we'll use that
    entry and correct the shift in the dispersion direction by adding the
    difference in FP_PIXEL_SHIFT values (from the lamptab).

    None will be returned if wavecal_info is empty.

    Parameters
    ----------
    wavecal_info: list of dictionaries
        List of wavecal information dictionaries.

    wcp_info: astropy.io.fits record object
        Data (one row) from the wavecal parameters table.

    cenwave: int
        Central wavelength, used to select entries from wavecal_info.

    fpoffset: int
        OSM position for the current science exposure, used to select
        entries from wavecal_info.

    time: float
        Time of observation, MJD at middle of exposure.

    Returns
    -------
    tuple, or None
        A pair of dictionaries and a string.  For the dictionaries, the
        key is the keyword name for the shift (shift1a, shift1b, shift1c,
        shift2a, shift2b, shift2c).  For the first dictionary, the value
        is the shift at the specified time.  For the second dictionary,
        the value is the slope (pixels/s).  The string is the name or
        names of the wavecal files (separated by a blank, if there are
        two) that were used to find the shifts.
    """

    if len(wavecal_info) < 1:
        return None

    slope_dict = {}             # initial value

    # Extract those elements of wavecal_info that match cenwave and fpoffset.
    subset_wavecal_info = selectWavecalInfo(wavecal_info, cenwave, fpoffset)

    if len(subset_wavecal_info) == 1:
        shift_dict = subset_wavecal_info[0]["shift_dict"]
        filename = subset_wavecal_info[0]["filename"]
    elif len(subset_wavecal_info) > 1:
        (shift_dict, slope_dict, filename) = \
        interpolateWavecal(subset_wavecal_info, time)
    else:
        # No matching row; find nearest in time.
        wc_dict = minTimeWavecalInfo(wavecal_info, time, cenwave,
                                     wcp_info.field("max_time_diff"))
        if wc_dict is None or cenwave != wc_dict["cenwave"]:
            cosutil.printWarning(
                    "No matching wavecal info; zero shift assumed.")
            shift_dict = None
        else:
            # No matching element in wavecal_info, but wc_dict should
            # match everything except fpoffset.  Apply a correction from
            # the wavecal fpoffset to the science exposure fpoffset.
            # fpoffset is for the current science exposure.
            # wc_dict["fpoffset"] is for the closest wavecal exposure.
            filename = wc_dict["filename"]
            shift_dict = wc_dict["shift_dict"].copy()
            fp_dict = wc_dict["fp_dict"]
            # Get any segment in fp_dict, use to construct segment in loop.
            fp_dict_keys = list(fp_dict.keys())
            (segment, fpoff) = fp_dict_keys.pop()       # ignore fpoff
            # Apply correction to all shift1[abc] values.
            for shift_dict_key in shift_dict:
                if not shift_dict_key.startswith("shift1"):
                    continue
                letter = shift_dict_key[-1].upper()
                key1 = (segment[:-1] + letter, fpoffset)
                key2 = (segment[:-1] + letter, wc_dict["fpoffset"])
                if key1 not in fp_dict or key2 not in fp_dict:
                    raise RuntimeError("fpoffset = %d or %d not found"
                                       " in the lamptab." %
                                       (fpoffset, wc_dict["fpoffset"]))
                # This is a difference in fp_pixel_shift values.
                correction = fp_dict[key1] - fp_dict[key2]
                shift_dict[shift_dict_key] += correction
                cosutil.printMsg("Info:  Keyword %s adjusted by adding %.6g" %
                                 (shift_dict_key, correction), VERBOSE)

    if shift_dict is None:
        return None

    # The default is zero slope.
    if not slope_dict:
        for key in shift_dict.keys():
            slope_dict[key] = 0.

    return (shift_dict, slope_dict, filename)

def returnExactMatch(wavecal_info, rootname):
    """Return the shift dictionary for the specified wavecal rootname.

    The rootname of a wavecal exposure is used to extract one element of
    wavecal_info (there should always be exactly one matching element).
    The shift dictionary from that matchine element is then returned.

    None will be returned if wavecal_info is empty.

    Parameters
    ----------
    wavecal_info: list of dictionaries
        List of wavecal information dictionaries

    rootname: str
        Used to find the appropriate element of wavecal_info

    Returns
    -------
    dictionary or None
        The shift dictionary corresponding to rootname, or None
    """

    if len(wavecal_info) < 1:
        return None

    for wc_dict in wavecal_info:
        if wc_dict["rootname"] == rootname:
            return wc_dict["shift_dict"]

    # We shouldn't get here.
    raise RuntimeError("There should have been a matching element.")

def selectWavecalInfo(wavecal_info, cenwave, fpoffset):
    """Return a list of all matching elements of wavecal_info.

    Parameters
    ----------
    wavecal_info: list of dictionaries
        List of wavecal information dictionaries

    cenwave: int
        Central wavelength, used to select entries from wavecal_info

    fpoffset: int
        Used to find one or more elements of wavecal_info

    Returns
    -------
    list
        List of dictionaries in wavecal_info that match cenwave and
        fpoffset
    """

    subset_wavecal_info = []

    for wc_dict in wavecal_info:
        if wc_dict["cenwave"] == cenwave and wc_dict["fpoffset"] == fpoffset:
            subset_wavecal_info.append(wc_dict)

    return subset_wavecal_info

def minTimeWavecalInfo(wavecal_info, time, cenwave, max_time_diff):
    """Return the element of wavecal_info that is closest to time.

    The element of wavecal_info that is closest in time to the specified
    time will be selected.  If the difference between the time for that
    element and the specified time is less than max_time_diff, that element
    of wavecal_info will be returned; otherwise, None will be returned.

    Associations can include exposures with different values of central
    wavelength, but it would not make sense to use a wavecal taken with
    one value of cenwave for a science exposure taken with a different
    value of cenwave, so elements of wavecal_info that do not match
    cenwave are ignored.

    Parameters
    ----------
    wavecal_info: list of dictionaries
        List of wavecal information dictionaries

    time: float
        Time of a science observation (MJD at middle of exposure)

    cenwave: int
        Central wavelength of the current science exposure, used to reject
        non-matching entries from wavecal_info

    max_time_diff: float
        Cutoff for time difference between time and a wavecal observation

    Returns
    -------
    dictionary
        The shift dictionary closest in time to time, or None if there
        is none that is within max_time_diff
    """

    index = -1
    for i in range(len(wavecal_info)):
        wc_dict = wavecal_info[i]
        if wc_dict["cenwave"] != cenwave:
            continue
        delta_t = abs(time - wc_dict["time"])
        if index < 0 or delta_t < min_time:
            index = i
            min_time = delta_t

    if index < 0:
        wc_dict = None
    else:
        wc_dict = wavecal_info[index]
        if min_time > max_time_diff:
            wc_dict = None

    return wc_dict

def interpolateWavecal(wavecal_info, time):
    """Interpolate to get a shift dictionary at the specified time.

    wavecal_info is assumed to be sorted in increasing order of time.
    If the time of observation is earlier than the first entry or later
    than the last entry in wavecal_info, then the shift dictionary for the
    first or last element respectively of wavecal_info will be returned.
    Otherwise, the pair of entries that bracket the time of observation
    will be selected, and the shifts in the shift dictionaries will be
    linearly interpolated at the time of observation.

    None will be returned if wavecal_info is empty.

    Parameters
    ----------
    wavecal_info: list of dictionaries
        list of wavecal information dictionaries

    time: float
        time of observation, MJD at middle of exposure

    Returns
    -------
    tuple, or None
        A pair of dictionaries and a string.  For the dictionaries, the
        key is the keyword name for the shift (shift1a, shift1b, shift1c,
        shift2a, shift2b, shift2c).  For the first dictionary, the value is
        the shift at the specified time.  For the second dictionary, the
        value is the slope in pixels per second.  The string is the name or
        names of the wavecal files (separated by a blank, if there are two)
        that were used to find the shifts.
    """

    wlen = len(wavecal_info)
    if wlen < 1:
        return None

    slope_dict = {}
    for key in wavecal_info[0]["shift_dict"]:
        slope_dict[key] = 0.

    if time <= wavecal_info[0]["time"]:
        return (wavecal_info[0]["shift_dict"], slope_dict,
                wavecal_info[0]["filename"])

    if time >= wavecal_info[wlen-1]["time"]:
        return (wavecal_info[wlen-1]["shift_dict"], slope_dict,
                wavecal_info[wlen-1]["filename"])

    for i in range(1, wlen):
        wc_i = wavecal_info[i]
        if time <= wavecal_info[i]["time"]:
            t0 = wavecal_info[i-1]["time"]
            t1 = wavecal_info[i]["time"]
            shift_dict0 = wavecal_info[i-1]["shift_dict"].copy()
            filename = wavecal_info[i-1]["filename"]
            if t0 != t1:
                filename = filename + " " + wavecal_info[i]["filename"]
                shift_dict1 = wavecal_info[i]["shift_dict"]
                # for each segment or stripe ...
                for key in shift_dict0.keys():
                    shift0 = shift_dict0[key]
                    if key in shift_dict1:
                        shift1 = shift_dict1[key]
                        p = (time - t0) / (t1 - t0)
                        q = 1. - p
                        shift = q * shift0 + p * shift1
                        shift_dict0[key] = shift
                        slope_dict[key] = (shift1 - shift0) \
                                          / ((t1 - t0) * SEC_PER_DAY)

            return (shift_dict0, slope_dict, filename)

def findWavecalSpectrum(corrtag, info, reffiles):
    """Find the offset of a wavecal spectrum in the cross-dispersion direction.

    Note that it is assumed that all wavecals are taken in time-tag mode.

    Parameters
    ----------
    corrtag: str
        Name of the corrtag FITS file containing a wavecal.

    info: dictionary
        Keywords and values from the headers of the input file.

    reffiles: dictionary
        Reference file names.

    Returns
    -------
    (shift2, xd_shifts, xd_locns, lamp_is_on): tuple of a float, two
    dictionaries, and a boolean flag
        shift2 is the offset (average of those found, if NUV) from
        nominal in the cross-dispersion direction, in pixels; this value
        will be zero if the offset could not be determined.  Dictionaries
        xd_shifts and xd_locns use the segment or stripe name as the
        key; the value for xd_shifts is the shift from nominal, and the
        value for xd_locns is the location where the spectrum was found
        (projected onto the left edge).  lamp_is_on is a flag that
        indicates whether the lamp was actually on.
    """

    fd = fits.open(corrtag, mode="copyonwrite")
    phdr = fd[0].header
    sci_extn = fd["EVENTS"]
    if sci_extn.data is None or len(sci_extn.data) == 0:
        fd.close()
        return (0., {}, {}, False)

    xtractab = reffiles["xtractab"]

    wcp_info = cosutil.getTable(reffiles["wcptab"],
                                filter={"opt_elem": info["opt_elem"]},
                                exactly_one=True)
    xd_range = wcp_info.field("xd_range")[0]
    box = wcp_info.field("box")[0]

    if info["detector"] == "FUV":
        xi = sci_extn.data.field("XCORR")
        eta = sci_extn.data.field("YCORR")
    else:
        xi = sci_extn.data.field("RAWX")
        eta = sci_extn.data.field("RAWY")

    dq = sci_extn.data.field("DQ")

    (shift2, xd_shifts, xd_locns, lamp_is_on) = \
    ttFindWavecalSpectrum(xi, eta, dq, info, xd_range, box, xtractab)

    fd.close()

    cosutil.printMsg("Shift (location) in cross-dispersion direction:")
    # keys = xd_shifts.keys()
    # keys.sort()
    keys = sorted(xd_shifts)
    for key in keys:
        if xd_shifts[key] is None:
            cosutil.printMsg("%4s    ---- (%5.1f)  # not found in XD" \
                             % (key, xd_locns[key]))
        else:
            cosutil.printMsg("%4s  %6.1f (%5.1f)" % \
                             (key, xd_shifts[key], xd_locns[key]))
    cosutil.printMsg("  avg %6.1f" % shift2)

    return (shift2, xd_shifts, xd_locns, lamp_is_on)

def ttFindWavecalSpectrum(xi, eta, dq, info, xd_range, box, xtractab):
    """Find the offset of a wavecal spectrum in cross-dispersion direction.

    Parameters
    ----------
    xi: array_like
        Corrected pixel coordinates in the dispersion direction.

    eta: array_like
        Corrected pixel coordinates in the cross-dispersion direction.

    dq: array_like
        Data quality flags.

    info: dictionary
        Header keywords and values.

    xd_range: int
        Search within + or - xd_range from the nominal location for the
        peak in the cross-dispersion direction.

    box: int
        Smooth the cross-dispersion profile with a box of this width before
        looking for the maximum.

    xtractab: str
        Name of the 1-D extraction parameters table.

    Returns
    -------
    (shift2, xd_shifts, xd_locns, lamp_is_on):  tuple of a float, two
    dictionaries, and a boolean flag
        shift2 is the offset (average of those found, if NUV) from
        nominal in the cross-dispersion direction, in pixels; this value
        will be zero if the offset could not be determined.  Dictionaries
        xd_shifts and xd_locns use the segment or stripe name as the
        key; the value for xd_shifts is the shift from nominal, and the
        value for xd_locns is the location where the spectrum was found
        (projected onto the left edge).  lamp_is_on is a flag that
        indicates whether the lamp was actually on.
    """

    if len(xi) < 1:
        shift2 = 0.
        xd_shifts = {}
        xd_locns = {}
        lamp_is_on = False
        if info["detector"] == "FUV":
            segment_list = [info["segment"]]
        elif info["obstype"] == "IMAGING":
            segment_list = ["NUVA"]
        else:
            segment_list = ["NUVA", "NUVB", "NUVC"]
        for segment in segment_list:
            xd_shifts[segment] = None
            xd_locns[segment] = 0.
        return (shift2, xd_shifts, xd_locns, lamp_is_on)

    filter = {"segment": info["segment"],
              "opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": "WCA"}

    if info["detector"] == "FUV":
        (shift2, xd_shifts, xd_locns) = ttFindFUV(xi, eta, dq,
                info["life_adj_offset"], xd_range, box, filter, xtractab)
    elif info["obstype"] == "IMAGING":
        (shift2, xd_shifts, xd_locns) = ttFindImagingWavecal(xi, eta, dq,
                info["life_adj_offset"], xd_range, box, filter, xtractab)
    else:
        (shift2, xd_shifts, xd_locns) = ttFindNUV(xi, eta, dq,
                info["life_adj_offset"], xd_range, box, filter, xtractab)

    if shift2 is None:
        shift2 = 0.

    # Was the lamp was actually on?  shift2 would have been None if we
    # didn't find the shift, but this test is more quantitative and prints
    # some statistics.
    lamp_is_on = cosutil.isLampOn(xi, eta, dq, info, xtractab, shift2)

    return (shift2, xd_shifts, xd_locns, lamp_is_on)

def ttFindFUV(xi, eta, dq, life_adj_offset, xd_range, box, filter, xtractab):

    xdisp = np.zeros(FUV_Y, dtype=np.float32)

    shift2 = None
    xd_shifts = {}
    xd_locns = {}
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    if xtract_info is not None:
        slope = xtract_info.field("slope")[0]
        # Collapse the data along the dispersion direction, putting the
        # result in xdisp.
        ccos.xy_collapse(xi, eta, dq, slope, xdisp)
        (shift2, y) = ttFindSpec(xdisp, xtract_info,
                                 life_adj_offset, xd_range, box)
        segment = filter["segment"]
        xd_shifts = {segment: shift2}
        xd_locns = {segment: y}

    return (shift2, xd_shifts, xd_locns)

def ttFindImagingWavecal(xi, eta, dq, life_adj_offset, xd_range,
                         box, filter, xtractab):

    xdisp = np.zeros(NUV_Y, dtype=np.float32)

    xd_shifts = {}
    xd_locns = {}

    filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable(xtractab, filter)
    if xtract_info is not None:
        slope = xtract_info.field("slope")[0]
        # Collapse the data along the dispersion direction.
        ccos.xy_collapse(xi, eta, dq, slope, xdisp)
        (shift2, y) = ttFindSpec(xdisp, xtract_info,
                                 life_adj_offset, xd_range, box)
        xd_shifts["NUVA"] = shift2
        xd_locns["NUVA"] = y

    return (shift2, xd_shifts, xd_locns)

def ttFindNUV(xi, eta, dq, life_adj_offset, xd_range, box, filter, xtractab):

    xdisp = np.zeros(NUV_Y, dtype=np.float32)

    xd_shifts = {}
    xd_locns = {}
    got_shift = False

    filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    if xtract_info is not None:
        slope = xtract_info.field("slope")[0]
        outlier_limit = xtract_info.field("height")[0] / 4.
        # Collapse the data along the dispersion direction.
        ccos.xy_collapse(xi, eta, dq, slope, xdisp)
        (shift2, y) = ttFindSpec(xdisp, xtract_info,
                                 life_adj_offset, xd_range, box)
        xd_shifts["NUVA"] = shift2
        xd_locns["NUVA"] = y
        if shift2 is not None:
            got_shift = True

    filter["segment"] = "NUVB"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    if xtract_info is not None:
        slope = xtract_info.field("slope")[0]
        outlier_limit = xtract_info.field("height")[0] / 4.
        ccos.xy_collapse(xi, eta, dq, slope, xdisp)
        (shift2, y) = ttFindSpec(xdisp, xtract_info,
                                 life_adj_offset, xd_range, box)
        xd_shifts["NUVB"] = shift2
        xd_locns["NUVB"] = y
        if shift2 is not None:
            got_shift = True

    filter["segment"] = "NUVC"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    if xtract_info is not None:
        slope = xtract_info.field("slope")[0]
        outlier_limit = xtract_info.field("height")[0] / 4.
        ccos.xy_collapse(xi, eta, dq, slope, xdisp)
        (shift2, y) = ttFindSpec(xdisp, xtract_info,
                                 life_adj_offset, xd_range, box)
        xd_shifts["NUVC"] = shift2
        xd_locns["NUVC"] = y
        if shift2 is not None:
            got_shift = True

    if got_shift:
        # Find the average shift, ignoring outliers (beyond 1/4 the
        # extraction height).
        shifts = []
        for shift in xd_shifts.values():
            if shift is not None:
                shifts.append(shift)
        shifts.sort()
        median_shift = shifts[len(shifts)//2]
        sum_s = 0.
        nsum = 0.
        for shift in shifts:
            if abs(shift - median_shift) < outlier_limit:
                sum_s += shift
                nsum += 1.
        shift2 = sum_s / nsum
    else:
        shift2 = None

    return (shift2, xd_shifts, xd_locns)

def ttFindSpec(xdisp, xtract_info, life_adj_offset, xd_range, box):
    """Find the location in the cross-dispersion direction.

    Parameters
    ----------
    xdisp: array_like
        The cross-dispersion profile, 1-D array of time-tag data collapsed
        along the dispersion axis, but taking into account the tilt of the
        spectrum.

    xtract_info: array_like
        Data block (but just one row) from the xtractab.

    life_adj_offset: float
        Normally this will be 0.  If the LIFE_ADJ keyword is -1, however,
        indicating that the aperture block is not at one of the recognized
        "lifetime positions," life_adj_offset will be the expected offset
        (in pixels) of the wavecal spectrum from lifetime position 1.

    xd_range: int
        Search within + or - xd_range from the nominal location for the
        peak in xdisp.

    box: int
        Smooth xdisp with a box of this width before looking for the
        maximum.

    Returns
    -------
    (shift2, y): tuple of two floats
        shift2 is the shift from nominal in the cross-dispersion
        direction (or None), and y is the location of the spectrum.
        The location is based on fitting a quadratic to points near the
        maximum.  Note that the data were collapsed to the left edge to
        get xdisp, so the location is the intercept on the edge, rather
        than where the spectrum crosses the middle of the detector.
    """

    y_nominal = xtract_info.field("b_spec")[0] + life_adj_offset
    segment = xtract_info.field("segment")[0]   # for possible warning message

    # The values of y_nominal and xd_range should be such that neither
    # y0 nor y1 will be less than zero or greater than 1023.
    y0 = int(round(y_nominal - xd_range))
    y1 = int(round(y_nominal + xd_range)) + 1
    if y0 < 0 or y1 >= len(xdisp):
        cosutil.printWarning("XD_RANGE in WCPTAB is too large.")
        y0 = max(y0, 0)
        y1 = min(y1, len(xdisp) - 1)

    boxcar_kernel = scipysignal.boxcar(box) / box
    xdisp_sm = ndimage.convolve(xdisp, boxcar_kernel, mode="nearest")
    len_xdisp_sm = len(xdisp_sm)

    if y0 >= y1:
        return (None, 0.)
    index = np.argsort(xdisp_sm[y0:y1])
    y = y0 + index[-1]
    signal = xdisp_sm[y]                # value in smoothed array
    # Check for duplicate values.
    y_min = y
    y_max = y
    while y_min > 0 and xdisp_sm[y_min] == signal:
        y_min -= 1
    while y_max < len_xdisp_sm and xdisp_sm[y_max] == signal:
        y_max += 1
    y_float = float(y_min + y_max) / 2.
    y = int(round(y_float))

    # Fit a quadratic to the smoothed curve near the peak.
    fit_range = (y_max - y_min) + box
    if fit_range < xd_range:
        r0 = y - fit_range // 2
        r1 = r0 + fit_range
        r0 = max(r0, 0)
        r1 = min(r1, len_xdisp_sm)
        r0 = r1 - fit_range
        x = np.arange(fit_range, dtype=np.float64)
        (coeff, var) = cosutil.fitQuadratic(x, xdisp_sm[r0:r1])
        (y_temp, y_float_sigma) = cosutil.centerOfQuadratic(coeff, var)
        if y_temp is None:
            return (None, 0.)
        y_float = y_temp + r0

    # Find the background level.
    i = index[(y1-y0)//2]
    background = xdisp_sm[y0+i]         # median of smoothed array

    sigma_s = math.sqrt(signal * box)
    sigma_b = math.sqrt(background * box)
    sigma_s_b = math.sqrt(sigma_s**2 + sigma_b**2)
    if sigma_s_b > 0.:
        signal_to_noise = (signal - background) * box / sigma_s_b
    else:
        signal_to_noise = 0.

    if signal_to_noise >= 5.:
        shift2 = y_float - y_nominal + life_adj_offset
    else:
        shift2 = None

    return (shift2, y_float)

def printWavecalRef(reffiles):
    """Print the names of reference files used for wavecal processing.

    Parameters
    ----------
    reffiles: dictionary
        Reference file names
    """

    cosutil.printRef("wcptab", reffiles)
    cosutil.printRef("lamptab", reffiles)
    cosutil.printRef("xtractab", reffiles)
