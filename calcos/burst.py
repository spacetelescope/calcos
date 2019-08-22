from __future__ import absolute_import, division         # confidence high
import math
import numpy as np
from .calcosparam import *
from . import cosutil
from . import ccos
LARGE_BURST = -20               # flag value in bkg_counts
SMALL_BURST = -10               # flag value in bkg_counts

def burstFilter(time, y, dq, reffiles, info, burstfile=None,
                high_countrate=False):
    """Flag regions where the count rate is unreasonably high.

    For each burst interval detected, a flag will be set in the
    data quality column for each event within that time interval.
    This is based on cf_screen_burst.c for FUSE.

    Parameters
    ----------
    time: array_like
        The time column in the events table.

    y: array_like
        The column of cross dispersion locations of events.

    dq: array_like
        The data quality column in the events table (updated in-place).

    reffiles: dictionary
        Reference file names.

    info: dictionary
        Header keywords and values.

    burstfile: str or None
        Name of output text file for burst info.

    high_countrate: boolean
        This flag can be set to true to force the time interval to be
        short, as if the data were high count rate.

    Returns
    -------
    bursts: list of two floats
        List of [bad_start, bad_stop] intervals during which a burst
        was detected (seconds since expstart).
    """

    bursts = None

    if info["segment"][:3] != "FUV":
        return None

    if info["exptime"] <= 0.:
        cosutil.printWarning("burstFilter:  Can't screen for bursts because"
                             " exptime = %g" % info["exptime"])
        return None

    cosutil.printMsg("Screen for bursts", VERBOSE)

    # Read parameters from burst reference table.
    (median_n, delta_t, delta_t_high, median_dt, burst_min,
        stdrej, source_frac, max_iter, high_rate) = \
                getBurstParam(reffiles["brsttab"], info["segment"])

    # Read location of active area from baseline reference table and
    # source and background locations from 1-D extraction reference table.
    try:
        (active_low, active_high, src_low, src_high,
         bkg1_low, bkg1_high, bkg2_low, bkg2_high) = \
                getRegionLocations(reffiles, info)
    except:
        cosutil.printWarning("Can't screen for bursts" \
                             " due to missing row in reference table")
        return None

    # rows of extraction aperture / background rows
    bkgsf = float(src_high - src_low + 1) / \
            float(bkg1_high - bkg1_low + bkg2_high - bkg2_low + 2)

    exptime = info["exptime"]
    countrate = float(len(time)) / exptime
    cosutil.printMsg("Total counts = %d, exposure time = %g s,"
            " count rate = %g c/s" % (len(time), exptime, countrate), VERBOSE)

    t0 = time[0]
    last = time[len(time)-1]
    if last <= t0:
        return None

    if countrate > high_rate:
        delta_t = delta_t_high
        cosutil.printMsg("High count rate; time bin set to %g s" % delta_t,
                         VERBOSE)
    elif high_countrate:
        delta_t = delta_t_high
        cosutil.printMsg("time bin set to %g s" % delta_t, VERBOSE)
    nbins = int(math.ceil((last - t0) / delta_t))
    del countrate, last
    if nbins <= 3:
        cosutil.printWarning("There are so few time bins (%d) that "
                "burst detection is not practical." % nbins)
        return None

    printParam(median_n, delta_t, median_dt, burst_min,
               stdrej, source_frac, max_iter, high_rate,
               active_low, active_high, src_low, src_high,
               bkg1_low, bkg1_high, bkg2_low, bkg2_high)

    # istart & istop are arrays of indices for slicing up the time and y
    # columns into intervals of length delta_t seconds.
    istart = np.zeros(nbins, dtype=np.int32)
    istop = np.zeros(nbins, dtype=np.int32)
    bkg_counts = np.zeros(nbins, dtype=np.int32)
    src_counts = np.zeros(nbins, dtype=np.int32)

    # Find istart & istop for each delta_t interval.
    ccos.getstartstop(time, istart, istop, delta_t)

    # find the counts within the background and source regions, for each
    # delta_t interval.
    ccos.getbkgcounts(y, dq, istart, istop,
                      bkg_counts, src_counts,
                      bkg1_low, bkg1_high, bkg2_low, bkg2_high,
                      src_low, src_high, bkgsf)

    bkg_counts_save = bkg_counts.copy()

    # Find the median of the values in the background counts array.
    index = np.argsort(bkg_counts)
    mid = nbins // 2
    median = bkg_counts[index[mid]]
    del index
    if median < 1:
        cosutil.printWarning("median = %d is unreasonable, reset to 1."
                             % median, VERBOSE)
        median = 1
    cutoff = median_n * median
    cosutil.printMsg("Initial check for large bursts,", VERBOSE)
    cosutil.printMsg("  median = %d, cutoff = %g; time interval = %.1f"
                     % (median, cutoff, delta_t), VERBOSE)
    # Identify intervals where the counts are greater than median_n * median.
    b1_flags = bkg_counts > cutoff
    index = np.nonzero(b1_flags)[0]
    nreject = len(index)
    for k in range(nreject):
        i = index[k]
        cosutil.printMsg("large burst at time %d, counts = %d"
                         % (int(time[istart[i]] + delta_t/2.), bkg_counts[i]),
                         VERBOSE)
        # Flag all events in the interval.
        dq[istart[i]:istop[i]] |= DQ_BURST
    if nreject > 0:
        # Set bkg_counts to a negative value for each burst.
        bkg_counts = np.where(b1_flags, LARGE_BURST, bkg_counts)
        cosutil.printMsg("%d large bursts detected." % nreject, VERBOSE)
    else:
        cosutil.printMsg("No large burst detected.", VERBOSE)
    del b1_flags

    # Search for smaller bursts.
    cosutil.printMsg("Check for smaller bursts;", VERBOSE)
    cosutil.printMsg("  median filter over time = %d s" % median_dt, VERBOSE)
    smallest_burst = burst_min * delta_t
    half_block = int(round(median_dt / delta_t)) // 2
    ccos.smallerbursts(time, dq,
                       istart, istop, bkg_counts, src_counts,
                       delta_t, smallest_burst, stdrej, source_frac,
                       half_block, max_iter,
                       LARGE_BURST, SMALL_BURST, DQ_BURST,
                       cosutil.checkVerbosity(VERBOSE))

    if burstfile is not None:
        # Write the time (middle of the interval), the background counts,
        # and whether the interval was flagged as a burst, large or small.
        fd = open(burstfile, "a")
        for i in range(nbins):
            t = t0 + (i+0.5) * delta_t
            fd.write("%.3f %d %d %d\n" %
                     (t, bkg_counts_save[i],
                         bkg_counts[i] == LARGE_BURST,
                         bkg_counts[i] == SMALL_BURST))
        fd.close()

    # Construct the list of start, stop intervals containing bursts.
    bursts = extractIntervals(time, istart, bkg_counts)

    return bursts

def getBurstParam(brsttab, segment):
    """Read parameters from burst reference table.

    Parameters
    ----------
    brsttab: str
        The name of the burst reference table.

    segment: str {"FUVA", "FUVB"}
        FUV segment name.

    Returns
    -------
    tuple
        The parameters read from the brsttab.
    """

    burst_info = cosutil.getTable(brsttab, filter={"segment": segment},
                                  exactly_one=True)

    median_n     = burst_info.field("median_n")[0]
    delta_t      = burst_info.field("delta_t")[0]
    delta_t_high = burst_info.field("delta_t_high")[0]
    median_dt    = burst_info.field("median_dt")[0]
    burst_min    = burst_info.field("burst_min")[0]
    stdrej       = burst_info.field("stdrej")[0]
    source_frac  = burst_info.field("source_frac")[0]
    max_iter     = burst_info.field("max_iter")[0]
    high_rate    = burst_info.field("high_rate")[0]

    return (median_n, delta_t, delta_t_high, median_dt, burst_min,
            stdrej, source_frac, max_iter, high_rate)

def getRegionLocations(reffiles, info):
    """Read region locations from reference tables.

    The lower and upper limits of the active area will be read from the
    baseline reference table.  The location and height of the source
    extraction region will be read from the 1-D extraction parameters
    table, and these will be used to define the source and background
    regions.

    Parameters
    ----------
    reffiles: dictionary
        Reference file names.

    info: dictionary
        Header keywords and values.

    Returns
    -------
    tuple
        A tuple with the lower and upper limits (in the cross-dispersion
        direction) of the active area, the source extraction region, and
        the two background regions.
    """

    (active_low, active_high, active_left, active_right) = \
            cosutil.activeArea(info["segment"], reffiles["brftab"])

    filter = {"segment": info["segment"],
              "opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": info["aperture"]}
    xtract_info = cosutil.getTable(reffiles["xtractab"],
                  filter, exactly_one=True)
    b_spec = xtract_info.field("b_spec")[0]
    b_spec = int(round(b_spec))
    height = xtract_info.field("height")[0]

    src_low = b_spec - height // 2
    src_high = b_spec + height // 2

    bkg1_low = max(0, active_low)
    bkg2_high = min(1023, active_high)
    bkg1_high = src_low - height // 4
    bkg2_low = src_high + height // 4

    if info["tagflash"]:
        # Reset bkg2_low to a point above the wavecal spectrum.
        filter["aperture"] = "WCA"
        xtract_info = cosutil.getTable(reffiles["xtractab"],
                                       filter, exactly_one=True)
        b_spec = xtract_info.field("b_spec")[0]
        b_spec = int(round(b_spec))
        bkg2_low = b_spec + height * 3 // 4

    return (active_low, active_high, src_low, src_high,
            bkg1_low, bkg1_high, bkg2_low, bkg2_high)

def extractIntervals(time, istart, bkg_counts):
    """Construct list of bad time intervals.

    Parameters
    ----------
    time: array_like
        Time column from events table.

    istart: array_like
        Array of indices; time[istart[i]] is the time at the start of bin i.

    bkg_counts: array_like
        Negative values are used to flag bursts (otherwise, this is the
        array of background counts within each time bin).

    Returns
    -------
    list of two-element lists, or None
        List of [bad_start, bad_stop] intervals during which a burst was
        detected (seconds since expstart).  The function value will be
        None of no burst was detected.
    """

    if bkg_counts.min() >= 0:
        return None

    bursts = []
    nbins = len(bkg_counts)

    in_bad_interval = False
    for i in range(nbins):
        if bkg_counts[i] < 0 and not in_bad_interval:
            in_bad_interval = True
            t1 = time[istart[i]]        # time at start of current bin
        elif bkg_counts[i] >= 0 and in_bad_interval:
            in_bad_interval = False
            t2 = time[istart[i]]        # time at end of previous bin
            bursts.append([t1, t2])

    if in_bad_interval:
        bursts.append([t1, time[-1]])

    return bursts

def printParam(median_n, delta_t, median_dt, burst_min,
               stdrej, source_frac, max_iter, high_rate,
               active_low, active_high, src_low, src_high,
               bkg1_low, bkg1_high, bkg2_low, bkg2_high):
    """Print the parameters that will be used."""

    if not cosutil.checkVerbosity(VERY_VERBOSE):
        return

    cosutil.printMsg("The burst parameters are:", VERY_VERBOSE)

    cosutil.printMsg(
"reject counts higher than %.1f times the global median" % median_n,
                VERY_VERBOSE)

    cosutil.printMsg(
"%.1f counts/s is considered to be high count rate" % high_rate,
                VERY_VERBOSE)

    cosutil.printMsg(
"time interval for binning events = %.1f s" % delta_t, VERY_VERBOSE)

    cosutil.printMsg(
"time interval for median filter = %.1f s" % median_dt, VERY_VERBOSE)

    cosutil.printMsg(
"%.1f counts/s is minimum count rate that can be regarded as a burst" %
                burst_min, VERY_VERBOSE)

    cosutil.printMsg(
"reject counts greater than %.1f standard deviations" % stdrej, VERY_VERBOSE)

    cosutil.printMsg(
"burst must exceed %.3f of the source count rate before it is" % source_frac,
                VERY_VERBOSE)

    cosutil.printMsg("  considered to be significant", VERY_VERBOSE)

    cosutil.printMsg("maximum number of iterations = %d" % max_iter,
                VERY_VERBOSE)

    cosutil.printMsg(
"active area is rows %d to %d inclusive" % (active_low, active_high),
                VERY_VERBOSE)

    cosutil.printMsg(
"source extraction region is rows %d to %d inclusive" % (src_low, src_high),
                VERY_VERBOSE)

    cosutil.printMsg(
"background regions are %d to %d and %d to %d inclusive" %
               (bkg1_low, bkg1_high, bkg2_low, bkg2_high), VERY_VERBOSE)
