from __future__ import division
import math
import os
import numpy as np
from convolve import boxcar
import pyfits
from calcosparam import *
import ccos
import cosutil
import findshift1
import shiftfile

# Each element of wavecal_info is a dictionary with the keys:
# "time", "fpoffset", "shift_dict", "rootname", "filename".  The variable
# name wc_dict is used for one element of wavecal_info.

def findWavecalShift (input, shift_file, info, wcp_info):
    """Find the shift from a wavecal image.

    @param input: name of an x1d FITS file containing a wavecal observation
    @type input: string
    @param shift_file: name of a user-supplied file to override shifts, or None
    @type shift_file: string
    @param info: keywords and values
    @type info: dictionary
    @param wcp_info: data (one row) from the wavecal parameters table
    @type wcp_info: pyfits record

    The function value is a dictionary of shifts, with keys shift1a,
    shift1b, shift1c.  The shift is the value for that segment or stripe
    only.  For FUV, there may be only one element in the dictionary.

    Note that the input file will be opened read-write, the SHIFT1A, SHIFT1B
    (and SHIFT1C if NUV) keywords will be updated with the shift of each
    individual segment or stripe.

    None will be returned if there is no data in the first extension of input.
    """

    # Note that we open the x1d table read-write, so we can update keywords.
    fd = pyfits.open (input, mode="update")
    phdr = fd[0].header
    sci_extn = fd["SCI"]
    if sci_extn.data is None:
        fd.close()
        return None

    reffiles = {}
    lamptab = cosutil.expandFileName (phdr["lamptab"])
    reffiles["lamptab"] = lamptab

    x_offset = sci_extn.header.get ("x_offset", default=0)
    info["x_offset"] = x_offset

    nrows = sci_extn.data.shape[0]
    gross = sci_extn.data.field ("gross")
    nelem = len (gross[0])
    exptime = sci_extn.data.field ("exptime")
    segment = sci_extn.data.field ("segment")

    detector = info["detector"]
    xc_range = wcp_info.field ("xc_range")
    stepsize = wcp_info.field ("stepsize")

    # Replace shift values in segment B with values from segment A?
    override_segment_B = (info["opt_elem"] == "G140L" and "FUVA" in segment)

    # segment will be added to the filter in the loop below.
    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"]}
    # If the FPOFFSET column is present in the lamptab, include fpoffset
    # in the filter.  If not, use an offset when doing the cross correlation.
    if cosutil.findColumn (lamptab, "fpoffset"):
        filter["fpoffset"] = info["fpoffset"]
    # fp is for an initial offset when matching the spectrum to the
    # template.  If we've got fpoffset and fp_pixel_shift columns,
    # the initial offset should be zero.
    got_pixel_shift = cosutil.findColumn (lamptab, "fp_pixel_shift")
    if got_pixel_shift:
        initial_offset = 0
    else:
        initial_offset = info["fpoffset"] * stepsize

    shift_dict = {}

    index = segment.argsort()
    save_spectra = {}
    save_templates = {}
    fp_pixel_shift = {}

    for row in index:
        filter["segment"] = segment[row]
        lamp_info = cosutil.getTable (lamptab, filter)
        if lamp_info is None:
            continue
        # Save gross, but convert from count rate back to counts.
        save_spectra[segment[row]] = gross[row] * exptime[row]
        raw_template = lamp_info.field ("intensity")[0]
        save_templates[segment[row]] = \
                cosutil.getTemplate (raw_template, x_offset, nelem)
        if got_pixel_shift:
            fp_pixel_shift[segment[row]] = \
                    lamp_info.field ("fp_pixel_shift")[0]
        else:
            fp_pixel_shift[segment[row]] = 0.

    # find offset in dispersion direction
    fs1 = findshift1.Shift1 (save_spectra, save_templates, info, reffiles,
                             xc_range, fp_pixel_shift, initial_offset)
    fs1.findShifts()

    if override_segment_B:              # copy shift1 from A to B
        fs1.setShift1 ("FUVB", fs1.getShift1 ("FUVA"))

    # Did the user supply a file with overrides for the shifts?
    if shift_file is None:
        user_shifts = None
    else:
        user_shifts = shiftfile.ShiftFile (shift_file,
                                           info["root"], info["fpoffset"])

    if detector == "FUV":
        cosutil.printMsg ("segment   shift err  [orig.]   chi sq (n)", VERBOSE)
        cosutil.printMsg ("-------  ------ ---- -------   ----------", VERBOSE)
    else:
        cosutil.printMsg ("stripe    shift err  [orig.]   chi sq (n)", VERBOSE)
        cosutil.printMsg ("------   ------ ---- -------   ----------", VERBOSE)

    # Print and save results.
    for row in index:

        # This is the offset in the cross-dispersion direction.
        key = "shift2" + segment[row][-1].lower()
        if override_segment_B and segment[row] == "FUVB":
            shift_dict[key] = sci_extn.header.get ("shift2a", default=0.)
        else:
            shift_dict[key] = sci_extn.header.get (key, default=0.)
        sci_extn.header.update (key, shift_dict[key])

        # Zero for dpixel1[a-c] is appropriate for auto/GO wavecals.
        key = "dpixel1" + segment[row][-1].lower()
        sci_extn.header.update (key, 0.)
        shift_dict[key] = 0.

        key = "shift1" + segment[row][-1].lower()
        user_specified = False
        if user_shifts is not None:
            ((user_shift1, user_shift2), nfound) = \
                        user_shifts.getShifts (("any", segment[row]))
            if user_shift1 is not None:
                fs1.setShift1 (segment[row], user_shift1)
                user_specified = True
        shift_segment = fs1.getShift1 (segment[row])
        scatter = fs1.getScatter (segment[row])
        orig_shift1 = fs1.getOrigShift1 (segment[row])
        sci_extn.header.update (key, shift_segment)
        shift_dict[key] = shift_segment

        message = " %4s    %6.1f %4.2f [%5.1f]  %7.1f (%d)" % \
                (segment[row], shift_segment, scatter, orig_shift1,
                 fs1.getChiSq (segment[row]), fs1.getNdf (segment[row]))
        if user_specified:
            message = message + "  # user-specified"
        elif override_segment_B and segment[row] == "FUVB":
            message = message + "  # based on FUVA value"
        elif not fs1.getSpecFound (segment[row]):
            message = message + "  # not found"
        cosutil.printMsg (message, VERBOSE)

        # Save Chi square and the number of degrees of freedom in the
        # shift dictionary, and update the keywords in the x1d header.
        key = "chi_sq_" + segment[row][-1].lower()
        shift_dict[key] = round (fs1.getChiSq (segment[row]), 1)
        sci_extn.header.update (key, shift_dict[key])
        key = "ndf_" + segment[row][-1].lower()
        shift_dict[key] = fs1.getNdf (segment[row])
        sci_extn.header.update (key, shift_dict[key])

    fd.close()

    return shift_dict

def storeWavecalInfo (wavecal_info, time, fpoffset, shift_dict,
                      rootname, filename):
    """Append the current info to the wavecal_info list.

    @param wavecal_info: list of wavecal information dictionaries; updated
        in-place
    @type wavecal_info: list
    @param time: time of observation, MJD at middle of exposure
    @type time: float
    @param fpoffset: OSM position, used to select entries from wavecal_info
    @type fpoffset: int
    @param shift_dict: a dictionary of keyword names and shifts
    @type shift_dict: dictionary
    @param rootname: the rootname (typically lower case) of the observation
    @type rootname: string
    @param filename: the name of the raw file (this will typically be the
                    _a.fits name for FUV)
    @type filename: string

    shift_dict can have any of the following keys (and others):
    "shift1a" for FUV segment A or NUV stripe A,
    "shift1b" for FUV segment B or NUV stripe B,
    "shift1c" for NUV stripe C
    "shift2a" for FUV segment A or NUV stripe A,
    "shift2b" for FUV segment B or NUV stripe B,
    "shift2c" for NUV stripe C
    For each key, the value is the shift that was measured for that segment
    or stripe in the dispersion direction (shift1[abc]) or in the cross
    dispersion direction (shift2[abc]).

    The input information will be combined into a dictionary, which will then
    be appended to wavecal_info.  wavecal_info will be sorted in increasing
    order of time.
    """

    wc_dict = {}
    wc_dict["time"]       = time
    wc_dict["fpoffset"]   = fpoffset
    wc_dict["shift_dict"] = shift_dict
    wc_dict["rootname"]   = rootname
    wc_dict["filename"]   = os.path.basename (filename)

    wavecal_info.append (wc_dict)
    if len (wavecal_info) > 1:
        wavecal_info.sort (cmpTime)

def cmpTime (wc_dict_a, wc_dict_b):
    """Compare the times in two wavecal_info entries.

    @param wc_dict_a: one wavecal information dictionary (one element of
        wavecal_info)
    @type wc_dict_a: dictionary
    @param wc_dict_b: another wavecal information dictionary
    @type wc_dict_b: dictionary

    @return: 0 if the times are the same,
            -1 if the time in wc_dict_a is smaller than the time in wc_dict_b,
            +1 if the time in wc_dict_a is larger than the time in wc_dict_b
    @rtype: int
    """

    return cmp (wc_dict_a["time"], wc_dict_b["time"])

def returnWavecalShift (wavecal_info, wcp_info, fpoffset, time):
    """Return the shift dictionary from wavecal_info that matches fpoffset.

    @param wavecal_info: list of wavecal information dictionaries
    @type wavecal_info: list of dictionaries
    @param wcp_info: data (one row) from the wavecal parameters table
    @type wcp_info: PyFITS record
    @param fpoffset: OSM position, used to select entries from wavecal_info
    @type fpoffset: int
    @param time: time of observation, MJD at middle of exposure
    @type time: float

    @return: a pair of dictionaries and a string.  For the dictionaries, the
        key is the keyword name for the shift (shift1a, shift1b, shift1c,
        shift2a, shift2b, shift2c).  For the first dictionary, the value is
        the shift at the specified time.  For the second dictionary, the
        value is the slope (pixels/s).  The string is the name or names of
        the wavecal files (separated by a blank, if there are two) that were
        used to find the shifts.
    @rtype: tuple, or None

    The element of wavecal_info that matches fpoffset will be extracted.
    If there are multiple entries that match fpoffset, we'll find the two
    that are closest to the time of the observation and linearly interpolate
    the shifts at that time.

    If there is no entry in wavecal_info for the current fpoffset, we'll
    find the entry that is closest in time to the time of the observation.
    If the difference in time for that entry is not too large (based on info
    from the wavecal parameters table), we'll use that entry and correct for
    the difference in OSM position.

    None will be returned if wavecal_info is empty.
    """

    if len (wavecal_info) < 1:
        return None

    slope_dict = {}             # initial value

    # Extract those elements of wavecal_info that match fpoffset.
    subset_wavecal_info = selectWavecalInfo (wavecal_info, fpoffset)

    if len (subset_wavecal_info) == 1:
        shift_dict = subset_wavecal_info[0]["shift_dict"]
        filename = subset_wavecal_info[0]["filename"]
    elif len (subset_wavecal_info) > 1:
        (shift_dict, slope_dict, filename) = \
        interpolateWavecal (subset_wavecal_info, time)
    else:
        # No matching row; find nearest in time.
        wc_dict = minTimeWavecalInfo (wavecal_info, time,
                         wcp_info.field ("max_time_diff"))
        if wc_dict is None:
            cosutil.printWarning (
                    "No matching wavecal info; zero shift assumed.")
            shift_dict = None
        else:
            # Apply a correction to the current fpoffset.
            correction = \
                (fpoffset - wc_dict["fpoffset"]) * wcp_info.field ("stepsize")
            shift_dict = wc_dict["shift_dict"].copy()
            for key in shift_dict.keys():
                if key.startswith ("shift1"):
                    shift_dict[key] += correction
            filename = wc_dict["filename"]

    if shift_dict is None:
        return None

    # The default is zero slope.
    if not slope_dict:
        for key in shift_dict.keys():
            slope_dict[key] = 0.

    return (shift_dict, slope_dict, filename)

def returnExactMatch (wavecal_info, rootname):
    """Return the shift dictionary for the specified wavecal rootname.

    @param wavecal_info: list of wavecal information dictionaries
    @type wavecal_info: list
    @param rootname: used to find the appropriate element of wavecal_info
    @type rootname: string

    @return: the shift dictionary corresponding to rootname, or None
    @rtype: dictionary

    The rootname of a wavecal exposure is used to extract one element of
    wavecal_info (there should always be exactly one matching element).
    The shift dictionary from that matchine element is then returned.

    None will be returned if wavecal_info is empty.
    """

    if len (wavecal_info) < 1:
        return None

    for wc_dict in wavecal_info:
        if wc_dict["rootname"] == rootname:
            return wc_dict["shift_dict"]

    # We shouldn't get here.
    raise RuntimeError, "There should have been a matching element."

def selectWavecalInfo (wavecal_info, fpoffset):
    """Return a list of all elements of wavecal_info that match fpoffset.

    @param wavecal_info: list of wavecal information dictionaries
    @type wavecal_info: list
    @param fpoffset: used to find one or more elements of wavecal_info
    @type fpoffset: int

    @return: list of dictionaries in wavecal_info that match fpoffset
    @rtype: list
    """

    subset_wavecal_info = []

    for wc_dict in wavecal_info:
        if wc_dict["fpoffset"] == fpoffset:
            subset_wavecal_info.append (wc_dict)

    return subset_wavecal_info

def minTimeWavecalInfo (wavecal_info, time, max_time_diff):
    """Return the element of wavecal_info that is closest to time.

    @param wavecal_info: list of wavecal information dictionaries
    @type wavecal_info: list
    @param time: time of a science observation (MJD at middle of exposure)
    @type time: float
    @param max_time_diff: cutoff for time difference between 'time' and a
        wavecal observation
    @type max_time_diff: float

    @return: the shift dictionary closest in time to 'time', or None if
        there is none that is within max_time_diff
    @rtype: dictionary

    The element of wavecal_info that is closest in time to the specified time
    will be selected.  If the difference between the time for that element
    and the specified time is less than max_time_diff, that element of
    wavecal_info will be returned; otherwise, None will be returned.
    """

    index = -1
    for i in range (len (wavecal_info)):
        wc_dict = wavecal_info[i]
        delta_t = abs (time - wc_dict["time"])
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

def interpolateWavecal (wavecal_info, time):
    """Interpolate to get a shift dictionary at the specified time.

    @param wavecal_info: list of wavecal information dictionaries
    @type wavecal_info: list of dictionaries
    @param time: time of observation, MJD at middle of exposure
    @type time: float

    @return: a pair of dictionaries and a string.  For the dictionaries, the
        key is the keyword name for the shift (shift1a, shift1b, shift1c,
        shift2a, shift2b, shift2c).  For the first dictionary, the value is
        the shift at the specified time.  For the second dictionary, the
        value is the slope in pixels per second.  The string is the name or
        names of the wavecal files (separated by a blank, if there are two)
        that were used to find the shifts.
    @rtype: tuple, or None

    wavecal_info is assumed to be sorted in increasing order of time.
    If the time of observation is earlier than the first entry or later
    than the last entry in wavecal_info, then the shift dictionary for the
    first or last element respectively of wavecal_info will be returned.
    Otherwise, the pair of entries that bracket the time of observation
    will be selected, and the shifts in the shift dictionaries will be
    linearly interpolated at the time of obsrvation.

    None will be returned if wavecal_info is empty.
    """

    wlen = len (wavecal_info)
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

    for i in range (1, wlen):
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
                    if shift_dict1.has_key (key):
                        shift1 = shift_dict1[key]
                        p = (time - t0) / (t1 - t0)
                        q = 1. - p
                        shift = q * shift0 + p * shift1
                        shift_dict0[key] = shift
                        slope_dict[key] = (shift1 - shift0) \
                                          / ((t1 - t0) * SEC_PER_DAY)

            return (shift_dict0, slope_dict, filename)

def findWavecalSpectrum (corrtag, info, reffiles):
    """Find the offset of a wavecal spectrum in the cross-dispersion direction.

    @param corrtag: name of the corrtag FITS file containing a wavecal
    @type corrtag: string
    @param info: header keywords and values
    @type info: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary

    @return: (shift2, xd_shifts, xd_locns, lamp_is_on)
    @rtype: tuple of a float, two dictionaries, and a boolean flag

    The function value is a tuple of four items:  shift2, two dictionaries
    (xc_shifts, xd_locns), and a boolean flag.  shift2 is the offset (average
    of those found, if NUV) from nominal in the cross-dispersion direction,
    in pixels; this value will be zero if the offset could not be determined.
    xd_shifts and xd_locns use the segment or stripe name as the key; the
    value for xd_shifts is the shift from nominal, and the value for xd_locns
    is the location where the spectrum was found (projected onto the left
    edge).  lamp_is_on is a flag that indicates whether the lamp was actually
    on.

    Note that it is assumed that all wavecals are taken in time-tag mode.
    """

    fd = pyfits.open (corrtag, mode="readonly", memmap=0)
    # fd = pyfits.open (corrtag, mode="readonly", memmap=1)
    phdr = fd[0].header
    sci_extn = fd["EVENTS"]
    if sci_extn.data is None:
        fd.close()
        return (0., {}, {})
    xtractab = reffiles["xtractab"]

    wcp_info = cosutil.getTable (reffiles["wcptab"],
                                 filter={"opt_elem": info["opt_elem"]},
                                 exactly_one=True)
    xd_range = wcp_info.field ("xd_range")[0]
    box = wcp_info.field ("box")[0]

    if info["detector"] == "FUV":
        xi = sci_extn.data.field ("XCORR")
        eta = sci_extn.data.field ("YCORR")
    else:
        xi = sci_extn.data.field ("RAWX")
        eta = sci_extn.data.field ("RAWY")

    dq = sci_extn.data.field ("DQ")

    (shift2, xd_shifts, xd_locns, lamp_is_on) = \
    ttFindWavecalSpectrum (xi, eta, dq, info, xd_range, box, xtractab)

    fd.close()

    cosutil.printMsg ("Shift (location) in cross-dispersion direction:")
    keys = xd_shifts.keys()
    keys.sort()
    for key in keys:
        if xd_shifts[key] is None:
            cosutil.printMsg ("%4s    ---- (%5.1f)  # not found in XD" \
                              % (key, xd_locns[key]))
        else:
            cosutil.printMsg ("%4s  %6.1f (%5.1f)" % \
                    (key, xd_shifts[key], xd_locns[key]))
    cosutil.printMsg ("  avg %6.1f" % shift2)

    return (shift2, xd_shifts, xd_locns, lamp_is_on)

def ttFindWavecalSpectrum (xi, eta, dq, info, xd_range, box, xtractab):
    """Find the offset of a wavecal spectrum in cross-dispersion direction.

    @param xi: corrected pixel coordinates in the dispersion direction
    @type xi: numpy array
    @param eta: corrected pixel coords in the cross-dispersion direction
    @type eta: numpy array
    @param dq: data quality flags
    @type dq: numpy array
    @param info: header keywords and values
    @type info: dictionary
    @param xd_range: search within + or - xd_range from the nominal location
        for the peak in the cross-dispersion direction
    @type xd_range: int
    @param box: smooth the cross-dispersion profile with a box of this
        width before looking for the maximum
    @type box: float
    @param xtractab: name of the 1-D extraction parameters table
    @type xtractab: string

    @return: (shift2, xd_shifts, xd_locns, lamp_is_on)
    @rtype: tuple of a float, two dictionaries, and a boolean flag

    The function value is a tuple of four items:  shift2, two dictionaries
    (xc_shifts, xd_locns), and a boolean flag.  shift2 is the offset (average
    of those found, if NUV) from nominal in the cross-dispersion direction,
    in pixels; this value will be zero if the offset could not be determined.
    xd_shifts and xd_locns use the segment or stripe name as the key; the
    value for xd_shifts is the shift from nominal, and the value for xd_locns
    is the location where the spectrum was found.  lamp_is_on is a flag that
    indicates whether the lamp was actually on.
    """

    if len (xi) < 1:
        return (0., {}, {})

    filter = {"segment": info["segment"],
              "opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": "WCA"}

    if info["detector"] == "FUV":
        (shift2, xd_shifts, xd_locns) = ttFindFUV (xi, eta, dq,
                xd_range, box, filter, xtractab)
    elif info["obstype"] == "IMAGING":
        (shift2, xd_shifts, xd_locns) = ttFindImagingWavecal (xi, eta, dq,
                xd_range, box, filter, xtractab)
    else:
        (shift2, xd_shifts, xd_locns) = ttFindNUV (xi, eta, dq,
                xd_range, box, filter, xtractab)

    if shift2 is None:
        shift2 = 0.

    # Was the lamp was actually on?  shift2 would have been None if we
    # didn't find the shift, but this test is more quantitative and prints
    # some statistics.
    lamp_is_on = cosutil.isLampOn (xi, eta, dq, info, xtractab, shift2)

    return (shift2, xd_shifts, xd_locns, lamp_is_on)

def ttFindFUV (xi, eta, dq, xd_range, box, filter, xtractab):

    xdisp = np.zeros (FUV_Y, dtype=np.float32)

    shift2 = None
    xd_shifts = {}
    xd_locns = {}
    xtract_info = cosutil.getTable (xtractab, filter)
    if xtract_info is not None:
        slope = xtract_info.field ("slope")[0]
        # Collapse the data along the dispersion direction, putting the
        # result in xdisp.
        ccos.xy_collapse (xi, eta, dq, slope, xdisp)
        (shift2, y) = ttFindSpec (xdisp, xtract_info, xd_range, box)
        segment = filter["segment"]
        xd_shifts = {segment: shift2}
        xd_locns = {segment: y}

    return (shift2, xd_shifts, xd_locns)

def ttFindImagingWavecal (xi, eta, dq, xd_range, box, filter, xtractab):

    xdisp = np.zeros (NUV_Y, dtype=np.float32)

    xd_shifts = {}
    xd_locns = {}

    filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable (xtractab, filter)
    if xtract_info is not None:
        slope = xtract_info.field ("slope")[0]
        # Collapse the data along the dispersion direction.
        ccos.xy_collapse (xi, eta, dq, slope, xdisp)
        (shift2, y) = (ttFindSpec (xdisp, xtract_info, xd_range, box))
        xd_shifts["NUVA"] = shift2
        xd_locns["NUVA"] = y

    return (shift2, xd_shifts, xd_locns)

def ttFindNUV (xi, eta, dq, xd_range, box, filter, xtractab):

    xdisp = np.zeros (NUV_Y, dtype=np.float32)

    xd_shifts = {}
    xd_locns = {}
    got_shift = False

    filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable (xtractab, filter)
    if xtract_info is not None:
        slope = xtract_info.field ("slope")[0]
        outlier_limit = xtract_info.field ("height")[0] / 4.
        # Collapse the data along the dispersion direction.
        ccos.xy_collapse (xi, eta, dq, slope, xdisp)
        (shift2, y) = (ttFindSpec (xdisp, xtract_info, xd_range, box))
        xd_shifts["NUVA"] = shift2
        xd_locns["NUVA"] = y
        if shift2 is not None:
            got_shift = True

    filter["segment"] = "NUVB"
    xtract_info = cosutil.getTable (xtractab, filter)
    if xtract_info is not None:
        slope = xtract_info.field ("slope")[0]
        outlier_limit = xtract_info.field ("height")[0] / 4.
        ccos.xy_collapse (xi, eta, dq, slope, xdisp)
        (shift2, y) = (ttFindSpec (xdisp, xtract_info, xd_range, box))
        xd_shifts["NUVB"] = shift2
        xd_locns["NUVB"] = y
        if shift2 is not None:
            got_shift = True

    filter["segment"] = "NUVC"
    xtract_info = cosutil.getTable (xtractab, filter)
    if xtract_info is not None:
        slope = xtract_info.field ("slope")[0]
        outlier_limit = xtract_info.field ("height")[0] / 4.
        ccos.xy_collapse (xi, eta, dq, slope, xdisp)
        (shift2, y) = (ttFindSpec (xdisp, xtract_info, xd_range, box))
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
                shifts.append (shift)
        shifts.sort()
        median_shift = shifts[len(shifts)//2]
        sum_s = 0.
        nsum = 0.
        for shift in shifts:
            if abs (shift - median_shift) < outlier_limit:
                sum_s += shift
                nsum += 1.
        shift2 = sum_s / nsum
    else:
        shift2 = None

    return (shift2, xd_shifts, xd_locns)

def ttFindSpec (xdisp, xtract_info, xd_range, box):
    """Find the location in the cross-dispersion direction.

    @param xdisp: 1-D array of time-tag data collapsed along dispersion
        axis, but taking into account the tilt of the spectrum
    @type xdisp: numpy array
    @param xtract_info: data block (but just one row) from the xtractab
    @type xtract_info: pyfits record array
    @param xd_range: search within + or - xd_range from the nominal location
        for the peak in xdisp
    @type xd_range: int
    @param box: smooth xdisp with a box of this width before looking
        for the maximum
    @type box: int

    @return: (shift2, y)
    @rtype: tuple of two floats

    The function value is a tuple of the shift from nominal in the
    cross-dispersion direction and the location of the spectrum.
    The location is based on fitting a quadratic to points near the maximum.
    Note that the data were collapsed to the left edge to get xdisp, so the
    location is the intercept on the edge, rather than where the spectrum
    crosses the middle of the detector.
    """

    y_nominal = xtract_info.field ("b_spec")[0]
    segment = xtract_info.field ("segment")[0]  # for possible warning message

    # The values of y_nominal and xd_range should be such that neither
    # y0 nor y1 will be less than zero or greater than 1023.
    y0 = int (round (y_nominal - xd_range))
    y1 = int (round (y_nominal + xd_range)) + 1
    if y0 < 0 or y1 >= len (xdisp):
        cosutil.printWarning ("XD_RANGE in WCPTAB is too large.")
        y0 = max (y0, 0)
        y1 = min (y1, len (xdisp) - 1)

    xdisp_sm = boxcar (xdisp, (box,), mode="nearest")
    len_xdisp_sm = len (xdisp_sm)

    index = np.argsort (xdisp_sm[y0:y1])
    y = y0 + index[-1]
    signal = xdisp_sm[y]                # value in smoothed array
    # Check for duplicate values.
    y_min = y
    y_max = y
    while y_min > 0 and xdisp_sm[y_min] == signal:
        y_min -= 1
    while y_max < len_xdisp_sm and xdisp_sm[y_max] == signal:
        y_max += 1
    y_float = float (y_min + y_max) / 2.
    y = int (round (y_float))

    # Fit a quadratic to the smoothed curve near the peak.
    fit_range = (y_max - y_min) + box
    if fit_range < xd_range:
        r0 = y - fit_range // 2
        r1 = r0 + fit_range
        r0 = max (r0, 0)
        r1 = min (r1, len_xdisp_sm)
        r0 = r1 - fit_range
        x = np.arange (fit_range, dtype=np.float64)
        (coeff, var) = cosutil.fitQuadratic (x, xdisp_sm[r0:r1])
        (y_temp, y_float_sigma) = cosutil.centerOfQuadratic (coeff, var)
        y_float = y_temp + r0

    # Find the background level.
    i = index[(y1-y0)//2]
    background = xdisp_sm[y0+i]         # median of smoothed array

    sigma_s = math.sqrt (signal * box)
    sigma_b = math.sqrt (background * box)
    sigma_s_b = math.sqrt (sigma_s**2 + sigma_b**2)
    if sigma_s_b > 0.:
        signal_to_noise = (signal - background) * box / sigma_s_b
    else:
        signal_to_noise = 0.

    if signal_to_noise >= 5.:
        shift2 = y_float - y_nominal
    else:
        shift2 = None

    return (shift2, y_float)

def printWavecalRef (reffiles):
    """Print the names of reference files used for wavecal processing.

    @param reffiles: reference file names
    @type reffiles: dictionary
    """

    cosutil.printRef ("wcptab", reffiles)
    cosutil.printRef ("lamptab", reffiles)
    cosutil.printRef ("xtractab", reffiles)
