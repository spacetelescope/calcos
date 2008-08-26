import numpy as N
from convolve import boxcar
import pyfits
from calcosparam import *
import ccos
import cosutil

# Each element of wavecal_info is a dictionary with the keys:
# "time", "fpoffset", "shift_dict", "rootname".  The variable name
# wc_dict is used for one element of wavecal_info.

def findWavecalShift (input, wcp_info):
    """Find the shift from a wavecal image.

    arguments:
    input           name of an x1d FITS file containing a wavecal observation
    wcp_info        data (one row) from the wavecal parameters table

    The function value is a dictionary of shifts, with keys pshifta,
    pshiftb, pshiftc.  The shift is the value for that segment or stripe
    only.  For FUV, there may be only one element in the dictionary.

    Note that the input file will be opened read-write, the PSHIFTA, PSHIFTB
    (and PSHIFTC if NUV) keywords will be updated with the shift of each
    individual segment or stripe, and the primary header keyword WAVECORR
    will be set to "COMPLETE".

    None will be returned if there is no data in the first extension of input.
    """

    # Note that we open the x1d table read-write, so we can update keywords.
    fd = pyfits.open (input, mode="update")
    phdr = fd[0].header
    sci_extn = fd["SCI"]
    if sci_extn.data is None:
        fd.close()
        return None

    nrows = sci_extn.data.shape[0]
    net = sci_extn.data.field ("net")
    nelem = len (net[0])
    segment = sci_extn.data.field ("segment")
    lamptab = phdr["lamptab"]
    lamptab = cosutil.expandFileName (lamptab)

    detector = phdr["detector"]
    maxlag = wcp_info.field ("xc_range")
    fp = -phdr.get ("fpoffset", default=0) * wcp_info.field ("stepsize")

    # segment will be added to the filter in the loop below.
    filter = {"opt_elem": phdr["opt_elem"],
              "cenwave": phdr["cenwave"]}
    # If the FPOFFSET column is present in the lamptab, include fpoffset
    # in the filter.
    if cosutil.findColumn (lamptab, "fpoffset"):
        filter["fpoffset"] = phdr["fpoffset"]

    shift_dict = {}
    if detector == "FUV":
        cosutil.printMsg ("segment   shift   diagnostics", VERBOSE)
        cosutil.printMsg ("-------   -----   -----------", VERBOSE)
    else:
        cosutil.printMsg ("stripe    shift   diagnostics", VERBOSE)
        cosutil.printMsg ("------    -----   -----------", VERBOSE)

    index = segment.argsort()

    if detector == "FUV":
        for row in index:
            filter["segment"] = segment[row]
            lamp_info = cosutil.getTable (lamptab, filter)
            if lamp_info is None:
                continue
            sum_template = lamp_info.field ("intensity")[0].copy()
            sum_net = net[row].copy()

            (shift, n50) = crosscor (sum_net, sum_template, maxlag, fp)

            key = "pshift" + segment[row][-1].lower()
            sci_extn.header.update (key, shift)
            shift_dict[key] = shift
            message = " %4s    %6.2f   " % (segment[row], shift) + str (n50)
            cosutil.printMsg (message, VERBOSE)

            # This is the offset in the cross-dispersion direction.
            key = "shift2" + segment[row][-1].lower()
            shift_dict[key] = sci_extn.header.get (key, default=0.)
    else:
        first = True
        for row in index:
            filter["segment"] = segment[row]
            lamp_info = cosutil.getTable (lamptab, filter)
            if lamp_info is None:
                continue
            if first:
                sum_template = lamp_info.field ("intensity")[0].copy()
                sum_net = net[row].copy()
                first = False
            else:
                sum_template += lamp_info.field ("intensity")[0]
                sum_net += net[row]

        (shift, n50) = crosscor (sum_net, sum_template, maxlag, fp)

        first = True
        for row in index:
            key = "pshift" + segment[row][-1].lower()
            sci_extn.header.update (key, shift)
            shift_dict[key] = shift
            if first:
                message = " %4s    %6.2f   " % (segment[row], shift) + str (n50)
                first = False
            else:
                message = " %4s" % segment[row]
            cosutil.printMsg (message, VERBOSE)

            # This is the offset in the cross-dispersion direction.
            key = "shift2" + segment[row][-1].lower()
            shift_dict[key] = sci_extn.header.get (key, default=0.)

    if nrows > 0:
        phdr.update ("WAVECORR", "COMPLETE")

    fd.close()

    return shift_dict

def storeWavecalInfo (wavecal_info, time, fpoffset, shift_dict, rootname):
    """Append the current info to the wavecal_info list.

    arguments:
    wavecal_info    list of wavecal information dictionaries; updated in-place
    time            time of observation, MJD at middle of exposure
    fpoffset        OSM position, used to select entries from wavecal_info
    shift_dict      a dictionary of keyword names and shifts
    rootname        the rootname (typically lower case) of the observation

    shift_dict can have any of the following keys:
    "pshifta" for FUV segment A or NUV stripe A,
    "pshiftb" for FUV segment B or NUV stripe B,
    "pshiftc" for NUV stripe C
    For each key, the value is the shift that was measured for that segment
    or stripe.

    The input information will be combined into a dictionary, which will then
    be appended to wavecal_info.  wavecal_info will be sorted in increasing
    order of time.
    """

    wc_dict = {}
    wc_dict["time"]       = time
    wc_dict["fpoffset"]   = fpoffset
    wc_dict["shift_dict"] = shift_dict
    wc_dict["rootname"]   = rootname

    wavecal_info.append (wc_dict)
    if len (wavecal_info) > 1:
        wavecal_info.sort (cmpTime)

def cmpTime (wc_dict_a, wc_dict_b):
    """Compare the times in two wavecal_info entries.

    arguments:
    wc_dict_a      one wavecal information dictionary (one element of
                     wavecal_info)
    wc_dict_b      another wavecal information dictionary
    """

    return cmp (wc_dict_a["time"], wc_dict_b["time"])

def returnWavecalShift (wavecal_info, wcp_info, fpoffset, time):
    """Return the shift dictionary from wavecal_info that matches fpoffset.

    @param wavecal_info:   list of wavecal information dictionaries
    @type wavecal_info:   list of dictionaries
    @param wcp_info: data (one row) from the wavecal parameters table
    @type wcp_info: PyFITS record
    @param fpoffset: OSM position, used to select entries from wavecal_info
    @type fpoffset: int
    @param time: time of observation, MJD at middle of exposure
    @type time: float

    @return: a pair of dictionaries, with the keyword name for the shift
        (pshifta, pshiftb, pshiftc, shift2a, shift2b, shift2c) as the key.
        For the first dictionary, the value is the shift at the specified
        time.  For the second dictionary, the value is the slope.
    @rtype: tuple of dictionaries, or None

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
    elif len (subset_wavecal_info) > 1:
        (shift_dict, slope_dict) = \
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
                if key.startswith ("pshift"):
                    shift_dict[key] += correction

    if shift_dict is None:
        return None

    # The default is zero slope.
    if not slope_dict:
        for key in shift_dict.keys():
            slope_dict[key] = 0.

    return (shift_dict, slope_dict)

def returnExactMatch (wavecal_info, rootname):
    """Return the shift dictionary for the specified wavecal rootname.

    arguments:
    wavecal_info    list of wavecal information dictionaries
    rootname        used to find the appropriate element of wavecal_info

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

    arguments:
    wavecal_info    list of wavecal information dictionaries
    fpoffset        used to find one or more elements of wavecal_info
    """

    subset_wavecal_info = []

    for wc_dict in wavecal_info:
        if wc_dict["fpoffset"] == fpoffset:
            subset_wavecal_info.append (wc_dict)

    return subset_wavecal_info

def minTimeWavecalInfo (wavecal_info, time, max_time_diff):
    """Return the element of wavecal_info that is closest to time.

    arguments:
    wavecal_info    list of wavecal information dictionaries
    time            time of a science observation (MJD at middle of exposure)
    max_time_diff   cutoff for time difference between 'time' and a wavecal obs.

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

    @return: a pair of dictionaries, with the keyword name for the shift
        (pshifta, pshiftb, pshiftc, shift2a, shift2b, shift2c) as the key.
        For the first dictionary, the value is the shift at the specified
        time.  For the second dictionary, the value is the slope.
    @rtype: tuple of dictionaries, or None

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
        return (wavecal_info[0]["shift_dict"], slope_dict)

    if time >= wavecal_info[wlen-1]["time"]:
        return (wavecal_info[wlen-1]["shift_dict"], slope_dict)

    for i in range (1, wlen):
        wc_i = wavecal_info[i]
        if time <= wavecal_info[i]["time"]:
            t0 = wavecal_info[i-1]["time"]
            t1 = wavecal_info[i]["time"]
            shift_dict0 = wavecal_info[i-1]["shift_dict"].copy()
            if t0 != t1:
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

            return (shift_dict0, slope_dict)

def crosscor (x, template, maxlag, fp=0):
    """Cross correlate two arrays to find the offset between them.

    arguments:
    x           the observed spectrum
    template    a template wavecal spectrum (same length as x)
    maxlag      the amplitude of the offset for computing cross correlation
    fp          fpoffset in pixels

    The function value is a tuple, consisting of the shift and a diagnostic
    quantity n50.

    >>> x        = N.array ([1., 1., 1., 4., 1., 1., 1., 1., 1.])
    >>> template = N.array ([0., 0., 0., 0., 7., 0., 0., 0., 0.])
    >>> print crosscor (x, template, 2)
    (-1.0, array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    >>> x        = N.array ([1., 1., 1., 1., 1., 4., 1., 1., 1.])
    >>> print crosscor (x, template, 2)
    (1.0, array([1, 1, 1, 1, 1, 1, 1, 1, 1]))
    """

    length = len (x)

    # xc is the cross correlation.
    lenxc = 2*maxlag+1
    xc = N.zeros (lenxc, dtype=N.float32)
    for lag in range (-maxlag, maxlag+1):
        x1 = lag - fp
        x2 = x1 + length
        x1 = max (x1, 0)
        x2 = min (x2, length)
        t1 = -lag + fp
        t2 = t1 + length
        t1 = max (t1, 0)
        t2 = min (t2, length)
        product = x[x1:x2] * template[t1:t2]
        xc[maxlag+lag] = N.sum (product)

    # imax is the index of the maximum.  n50 is for diagnostic purposes.
    (imax, n50) = xcStat (xc)

    i1 = imax - 1
    i1 = max (i1, 0)
    i2 = i1 + 2
    i2 = min (i2, lenxc-1)
    # Unless we're at an endpoint of xc, index is equal to imax, the index
    # of the peak.
    index = i2 - 1
    denominator = xc[index-1] - 2. * xc[index] + xc[index+1]
    if denominator == 0.:
        cosutil.printWarning (
                "Indeterminate location of peak in cross correlation")
        cosutil.printContinuation (
                "for finding the shift from a wavecal observation.")
        cosutil.printContinuation (
                "The shift for this wavecal image will be set to zero.")
        shift = 0.
    else:
        location = (xc[index-1] - xc[index+1]) / (2. * denominator)
        # The peak in xc would be at maxlag (the middle element of xc)
        # if x and template were identical.
        shift = location + index - fp - maxlag

    return (shift, n50)

def xcStat (xc):
    """Find the location of the maximum, and some diagnostic info.

    argument:
    xc          the cross correlation

    The function value is a tuple giving the index of the maximum in xc
    and a diagnostic quantity n50.  n50 is described in the doc string for
    ttFindWavecalShift.
    """

    xc_sort = N.argsort (xc)

    # Find the location of the maximum value, and other stuff.
    imax = xc_sort[-1]
    imin = xc_sort[0]
    maxval = xc[imax]
    minval = xc[imin]
    if imax == 0 or imax == len (xc) - 1:
        cosutil.printWarning (
                "Peak in cross correlation is at an endpoint of the range")
        cosutil.printContinuation (
                "for finding the shift from a wavecal observation;")
        cosutil.printContinuation (
                "the shift is therefore likely to be incorrect.")

    # Find the number of elements that have a value greater than
    # the midpoint of the range.
    diff = maxval - minval
    fractions = N.arange (0.9, 0., -0.1)
    cutoff = [(fraction * diff + minval) for fraction in fractions]
    n50 = N.zeros (len (cutoff), dtype=N.int32)
    for i in range (len (cutoff)):
        gt = xc > cutoff[i]
        # Using sum() here relies on the fact that the values are 0 or 1.
        n = N.sum (gt.astype (N.float64))
        n50[i] = int (round (n))

    return (imax, n50)

def ttFindWavecalShift (net, template, info, wcp_info):
    """Find the shift from a wavecal spectrum.

    arguments:
    net             the 1-D extracted spectrum (may be either net or gross)
    template        template spectrum
    info            dictionary of header keywords and values
    wcp_info        data (one row) from the wavecal parameters table

    The function value is a tuple of the shift in the dispersion direction
    and some diagnostic info n50.  (None, [0.]) will be returned if there
    is nothing in the net array.  n50 is an array of nine elements, giving
    the number of elements in the cross correlation with values greater
    than various fractions of the range from the minimum to maximum values
    of the cross correlation.  The fractions are 0.9, 0.8, ... 0.1.  The
    idea is that if the spectrum and the template are very similar (as we
    would expect), the values in n50 should be small, but they could increase
    sharply toward the end due to noise in the spectrum.  Elements near the
    middle of n50 should be of order twice the size of the resolution element.
    Larger values in n50 indicate a poorer agreement between the spectrum
    and the template.
    """

    nelem = len (net)
    if nelem < 1:
        return (None, [0.])

    maxlag = wcp_info.field ("xc_range")
    fp = -info["fpoffset"] * wcp_info.field ("stepsize")

    (pshift, n50) = crosscor (net, template, maxlag, fp)

    return (pshift, n50)

def findWavecalSpectrum (corrtag, info, reffiles):
    """Find the offset of a wavecal spectrum in the cross-dispersion direction.

    arguments:
    corrtag         name of the corrtag FITS file containing a wavecal
    info            dictionary of header keywords and values
    reffiles        dictionary of reference file names

    The function value is a tuple of three items:  shift2 and two dictionaries,
    xc_shifts, xd_locns.  shift2 is the offset (average of those found, if NUV)
    from nominal in the cross-dispersion direction, in pixels; this value will
    be zero if the offset could not be determined.  xd_shifts and xd_locns use
    the segment or stripe name as the key; the value for xd_shifts is the shift
    from nominal, and the value for xd_locns is the location where the spectrum
    was found (projected onto the left edge if FUV or lower edge if NUV).

    Note that this assumes that all wavecals are taken in time-tag mode.
    """

    fd = pyfits.open (corrtag, mode="readonly", memmap=0)
    # fd = pyfits.open (corrtag, mode="readonly", memmap=1)
    phdr = fd[0].header
    sci_extn = fd["EVENTS"]
    if sci_extn.data is None:
        fd.close()
        return (0., {}, {})
    xtractab = reffiles["xtractab"]

    if info["detector"] == "FUV":
        xi = sci_extn.data.field ("XCORR")
        eta = sci_extn.data.field ("YCORR")
    else:
        xi = sci_extn.data.field ("RAWX")
        eta = sci_extn.data.field ("RAWY")

    dq = sci_extn.data.field ("DQ")

    (shift2, xd_shifts, xd_locns) = \
                ttFindWavecalSpectrum (xi, eta, dq, info, xtractab)

    fd.close()

    cosutil.printMsg ("Shift (location) in cross-dispersion direction:")
    keys = xd_shifts.keys()
    keys.sort()
    for key in keys:
        if xd_shifts[key] is None:
            cosutil.printMsg ("%4s    None (%5.1f)" % (key, xd_locns[key]))
        else:
            cosutil.printMsg ("%4s  %6.1f (%5.1f)" % \
                    (key, xd_shifts[key], xd_locns[key]))
    cosutil.printMsg ("  avg %6.1f" % shift2)

    return (shift2, xd_shifts, xd_locns)

def ttFindWavecalSpectrum (xi, eta, dq, info, xtractab):
    """Find the offset of a wavecal spectrum in cross-dispersion direction.

    arguments:
    xi              corrected pixel coordinates in the dispersion direction
    eta             corrected pixel coords in the cross-dispersion direction
    dq              data quality flags
    info            dictionary of header keywords and values
    xtractab        name of 1-D extraction parameters table

    The function value is a tuple of three items:  shift2 and two dictionaries,
    xc_shifts, xd_locns.  shift2 is the offset (average of those found, if NUV)
    from nominal in the cross-dispersion direction, in pixels; this value will
    be zero if the offset could not be determined.  xd_shifts and xd_locns use
    the segment or stripe name as the key; the value for xd_shifts is the shift
    from nominal, and the value for xd_locns is the location where the spectrum
    was found.
    """

    if len (xi) < 1:
        return (0., {}, {})

    filter = {"segment": info["segment"],
              "opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": "WCA"}

    if info["detector"] == "FUV":
        (shift2, xd_shifts, xd_locns) = ttFindFUV (xi, eta, dq,
                filter, xtractab)
    else:
        (shift2, xd_shifts, xd_locns) = ttFindNUV (xi, eta, dq,
                filter, xtractab)

    if shift2 is None:
        shift2 = 0.

    return (shift2, xd_shifts, xd_locns)

def ttFindFUV (xi, eta, dq, filter, xtractab):

    xdisp = N.zeros (FUV_Y, dtype=N.float32)

    xd_range = 50
    box = 7
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

def ttFindNUV (xi, eta, dq, filter, xtractab):

    xdisp = N.zeros (NUV_Y, dtype=N.float32)

    xd_range = 40
    box = 13
    xd_shifts = {}
    xd_locns = {}
    got_shift = False

    # Note that we allow for the possibility that the slopes of the
    # three stripes might not all be the same.  If they are close enough
    # to ignore the differences, the code could be modified to call
    # xy_collapse only once, using the slope for, say, NUVB.

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

    arguments:
    xdisp           1-D array of time-tag data collapsed along dispersion
                    axis, but taking into account the tilt of the spectrum
    xtract_info     data block (but just one row) from the xtractab
    xd_range        search within + or - xd_range from the nominal location
                    for the peak in xdisp
    box             smooth xdisp with a box of this width before looking
                    for the maximum

    The function value is a tuple of the shift from nominal in the
    cross-dispersion direction and the location of the spectrum.
    The location is an integer, the nearest to the location of the maximum.
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

    xdisp_sm = boxcar (xdisp, (box,), mode="nearest")

    index = N.argsort (xdisp_sm[y0:y1])
    y = y0 + index[-1]
    signal = xdisp[y]                   # actual value, not smoothed
    i = index[(y1-y0)//2]
    background = xdisp_sm[y0+i]         # median of smoothed array

    if signal > 4. * background:
        shift2 = y - y_nominal
    else:
        shift2 = None

    return (shift2, y)

def printWavecalRef (reffiles):
    """Print the names of reference files used for wavecal processing.

    argument:
    reffiles        dictionary of reference file names
    """

    cosutil.printRef ("wcptab", reffiles)
    cosutil.printRef ("lamptab", reffiles)
    cosutil.printRef ("xtractab", reffiles)
