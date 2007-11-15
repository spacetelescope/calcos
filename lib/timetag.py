import math
import numpy as N
import pyfits
import cosutil
import burst
import ccos
import concurrent
from calcosparam import *       # parameter definitions

# These are column names in the corrtag table.  The default values are
# appropriate for FUV data.

xcorr = "xcorr"
ycorr = "ycorr"
xdopp = "xdopp"
ydopp = "ycorr"
xfull = "xfull"
yfull = "yfull"

def timetagBasicCalibration (input, outtag, output, outcounts, outflash,
                  info, switches, reffiles,
                  stimfile, livetimefile, burstfile):
    """Do the basic processing for time-tag data.

    The function value will be zero if there was no problem,
    and it will be one if there was no input data.

    arguments:
    input         name of the input file
    outtag        name of the output file for corrected time-tag data
    output        name of the output file for flat-fielded count-rate image
    outcounts     name of the output file for count-rate image
    outflash      name of the output file for tagflash wavecal spectra (or None)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    stimfile      name of output text file for stim positions (or None)
    livetimefile  name of output text file for livetime factors (or None)
    burstfile     name of output text file for burst info (or None)
    """

    cosutil.printIntro ("TIME-TAG calibration")
    names = [("Input", input), ("Outtag", outtag),
             ("Output", output), ("Outcounts", outcounts)]
    if outflash is not None:
        names.append (("Outflash", outflash))
    cosutil.printFilenames (names, stimfile=stimfile, livetimefile=livetimefile)
    cosutil.printMode (info)

    # Copy data from the input file to the output.  Then open the output
    # file read/write.
    nrows = cosutil.writeOutputEvents (input, outtag)
    ofd = pyfits.open (outtag, mode="update", memmap=0)
    # ofd = pyfits.open (outtag, mode="update", memmap=1)

    # Get a copy of the primary header.  This copy will be modified and
    # written to the output image files.
    phdr = ofd[0].header

    # Update the switches and reference file names, so the output header
    # will reflect what was actually used.
    cosutil.overrideKeywords (phdr, ofd[1].header, info, switches, reffiles)

    # events_hdu is a complete fits HDU object (i.e., header plus data),
    # while events (assigned below) is just the data, a recarray object.
    events_hdu = ofd["EVENTS"]

    if nrows == 0:
        writeNull (input, output, outcounts, phdr, events_hdu)
        ofd.close()
        return 1

    setCorrColNames (info["detector"], info["tagflash"])

    events = events_hdu.data

    updateGlobrate (events, info, reffiles, events_hdu.header)

    doBurstcorr (events, info, switches, reffiles, phdr, burstfile)

    doBadtcorr (events, info, switches, reffiles, phdr)

    recomputeExptime (input, events, events_hdu.header)

    doPhacorr (events, info, switches, reffiles, phdr, events_hdu.header)

    (stim_param, stim_countrate, stim_livetime) = initTempcorr (events,
            input, info, switches, reffiles, events_hdu.header, stimfile)

    doRandcorr (events, info, switches, phdr)

    doTempcorr (stim_param, events, info, switches, reffiles, phdr)

    doGeocorr (events, info, switches, reffiles, phdr)

    dq_array = doDqicorr (events, info, switches, reffiles,
            phdr, events_hdu.header)

    doDoppcorr (events, info, switches, reffiles, phdr, events_hdu.header)

    doFlatcorr (events, info, switches, reffiles, phdr)

    doDeadcorr (events, input, info, switches, reffiles, phdr,
            stim_countrate, stim_livetime, livetimefile)

    concurrent.processConcurrentWavecal (events, outflash,
                info, switches, reffiles,
                phdr, events_hdu.header)

    writeImages (events.field (xfull), events.field (yfull),
                 events.field ("epsilon"), events.field ("dq"),
                 phdr, events_hdu.header,
                 dq_array, info["npix"], info["exptime"],
                 outcounts, output)

    doStatflag (switches, output, outcounts)

    ofd.close()

    return 0            # 0 is OK


def setCorrColNames (detector, tagflash):
    """Assign column names to global variables.

    argument:
    detector      FUV or NUV
    """

    global xcorr, ycorr, xdopp, ydopp, xfull, yfull

    if detector == "FUV":
        xcorr = "XCORR"
        ycorr = "YCORR"
        xdopp = "XDOPP"
        ydopp = "YCORR"
    else:
        xcorr = "RAWX"
        ycorr = "RAWY"
        xdopp = "RAWX"
        ydopp = "YDOPP"

    if tagflash:
        xfull = "XFULL"
        yfull = "YFULL"
    else:
        xfull = xdopp
        yfull = ydopp

def updateGlobrate (events, info, reffiles, hdr):
    """Update the GLOBRATE keyword in the extension header.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    reffiles      dictionary of reference file names
    hdr           the input events extension header
    """

    if info["detector"] == "FUV":
        eta = events.field ("ycorr")
    else:
        eta = events.field ("rawx")
    globrate = globrate_tt (eta,
                info["exptime"], info["segment"], reffiles["brftab"])
    hdr.update ("globrate", globrate)

def globrate_tt (eta, exptime, segment, brftab):
    """Return the global count rate for time-tag data.

    arguments:
    eta           pixel coordinates in cross-dispersion direction
    exptime       the exposure time
    segment       for finding a row in the brftab
    brftab        name of the baseline reference table

    The function value is the global count rate, counts per second.
    """

    if exptime <= 0.:
        return 0.

    if segment[0] == "N":
        return float (len (eta)) / exptime

    (b_low, b_high) = cosutil.activeArea (segment, brftab)
    flags = N.zeros (len (eta), dtype=N.bool8)
    flags |= N.logical_and (eta > b_low, eta < b_high)

    return N.sum (flags.astype (N.float32)) / exptime

def doBurstcorr (events, info, switches, reffiles, phdr, burstfile):
    """Find bursts, and flag them in the data quality column.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    phdr          the input primary header
    burstfile     name of output text file for burst info (or None)
    """

    if info["segment"][:3] == "FUV":
        # Find and flag regions where the count rate is unreasonably high.
        cosutil.printSwitch ("BRSTCORR", switches)
        if switches["brstcorr"] == "PERFORM":
            cosutil.printRef ("brsttab", reffiles)
            cosutil.printRef ("xtractab", reffiles)
            burst.burstFilter (events.field ("time"), events.field (ycorr),
                               events.field ("dq"), reffiles, info, burstfile)
            phdr.update ("brstcorr", "COMPLETE")

def doBadtcorr (events, info, switches, reffiles, phdr):
    """Flag bad time intervals in the data quality column.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    phdr          the input primary header
    """

    cosutil.printSwitch ("BADTCORR", switches)
    if switches["badtcorr"] == "PERFORM":
        cosutil.printRef ("BADTTAB", reffiles)
        filterByTime (events.field ("time"), events.field ("dq"),
                    reffiles["badttab"], info["expstart"], info["segment"])
        phdr["badtcorr"] = "COMPLETE"

def filterByTime (time, dq, badttab, expstart, segment):
    """Flag bad time intervals in dq.

    For each bad time interval in the badttab, a flag will be set in the
    data quality column for each event within that time interval.

    arguments:
    time          the time column in the events table
    dq            the data quality column in the events table (updated in-place)
    badttab       the name of the bad-time-intervals table
    expstart      the exposure start time (MJD)
    segment       FUVA or FUVB
    """

    # Flag regions listed in the badt table.
    badt = cosutil.getTable (badttab, filter={"segment": segment})

    nrows = badt.shape[0]

    if nrows > 0:
        start = badt.field ("start")
        stop  = badt.field ("stop")

        # Convert from MJD to seconds after expstart.
        for i in range (nrows):
            start[i] = (start[i] - expstart) * SEC_PER_DAY
            stop[i] = (stop[i] - expstart) * SEC_PER_DAY

        # For each time interval in the badttab, flag every event for which
        # the time falls within that interval.
        for i in range (nrows):
            dq |= N.where (N.logical_and \
                      (time >= start[i], time <= stop[i]), DQ_BAD_TIME, 0)

def recomputeExptime (input, events, hdr):
    """Recompute the exposure time and update the keyword.

    arguments:
    input         name of the input file (for getting GTI table)
    events        the data unit containing the events table
    hdr           the events extension header (exptime keyword will be updated)
    """
    time = events.field ("time")
    dq = events.field ("dq")

    exptime = 0.
    gti_list = cosutil.returnGTI (input)
    if len (gti_list) <= 0:
        exptime = time[-1] - time[0] - ccos.getbadtime (time, dq)
    else:
        for gti in gti_list:
            # These are the start and stop times of an interval.
            subexp = gti[1] - gti[0]
            (i0, i1) = ccos.range (time, gti[0], gti[1])
            badtime = ccos.getbadtime (time[i0:i1], dq[i0:i1])
            exptime += (subexp - badtime)

    old_exptime = hdr.get ("exptime", 0.)
    if exptime != old_exptime:
        hdr.update ("exptime", exptime)
        if abs (exptime - old_exptime) > 1.:
            cosutil.printWarning ("exposure time in header was %.3f" % \
                    old_exptime, VERBOSE)
            cosutil.printContinuation ("exptime has been corrected to %.3f" % \
                    exptime, VERBOSE)

def doPhacorr (events, info, switches, reffiles, phdr, hdr):
    """Filter by pulse height.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    phdr          the input primary header
    hdr           the input events extension header
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch ("PHACORR", switches)
        if switches["phacorr"] == "PERFORM":
            cosutil.printRef ("PHATAB", reffiles)
            filterByPulseHeight (events.field ("pha"), events.field ("dq"),
                    reffiles["phatab"], info["segment"], hdr)
            phdr["phacorr"] = "COMPLETE"

def filterByPulseHeight (pha, dq, phatab, segment, hdr):
    """Flag events that have a pulse height outside an allowed range.

    arguments:
    pha        pulse-height column in events table
    dq         data-quality column in events table (modified in-place)
    phatab     name of PHA thresholds table
    segment    segment name (FUVA or FUVB)
    hdr        header for events table extension (keywords for screening
                 limits and number of rejected events will be assigned)
    """

    pha_info = cosutil.getTable (phatab, filter={"segment": segment},
                   exactly_one=True)

    low = pha_info.field ("llt")[0]
    high = pha_info.field ("ult")[0]

    # Flag an event if the pulse height is below the minimum value or
    # above the maximum value that is likely to be encountered from a
    # real photon event.
    dq |= N.where (pha < low, DQ_PH_LOW, 0)
    dq |= N.where (pha > high, DQ_PH_HIGH, 0)

    # Count the number of rejected events.
    rejected = N.nonzero (dq & DQ_PH_LOW)[0]
    nbad_low = len (rejected)
    rejected = N.nonzero (dq & DQ_PH_HIGH)[0]
    nbad_high = len (rejected)
    nbad = nbad_low + nbad_high
    if cosutil.checkVerbosity (VERY_VERBOSE):
        cosutil.printMsg ("Filter by pulse height:", VERY_VERBOSE)
        if nbad_low == 0:
            msg = "  no event was"
        elif nbad_low == 1:
            msg = "  one event was"
        else:
            msg = "  %d events were" % nbad_low
        msg += " rejected because PHA was less than %d" % low
        cosutil.printMsg (msg, VERY_VERBOSE)
        if nbad_high == 0:
            msg = "  no event was"
        elif nbad_high == 1:
            msg = "  one event was"
        else:
            msg = "  %d events were" % nbad_high
        msg += " rejected because PHA was greater than %d" % high
        cosutil.printMsg (msg, VERY_VERBOSE)

    keyword = "PHA_BAD" + segment[-1]
    hdr.update (keyword, nbad)

    # Update the values for the screening limit keywords
    # (low and high are the default values).
    cosutil.updatePulseHeightKeywords (hdr, segment, low, high)

def initTempcorr (events, input, info, switches, reffiles, hdr, stimfile):
    """Compute parameters for thermal distortion.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    hdr           the input events extension header
    stimfile      name of output text file for stim positions (or None)
    """

    if info["detector"] == "FUV" and \
       (switches["tempcorr"] == "PERFORM" or switches["deadcorr"] == "PERFORM"):
        # Compute the parameters (to be used later).
        time = cosutil.getColCopy (data=events, column="time")
        (stim_param, stim_locations, stim_countrate, stim_livetime) = \
         computeThermalParam (time,
            events.field (xcorr), events.field (ycorr), events.field ("dq"),
            reffiles["brftab"],
            info["segment"], info["exptime"], info["stimrate"],
            input, stimfile)
        # Update stim location keywords in extension header.
        stimKeywords (hdr, info["segment"], stim_locations)
    else:
        stim_param = {}
        stim_countrate = 0.
        stim_livetime = 1.

    return (stim_param, stim_countrate, stim_livetime)

def computeThermalParam (time, x, y, dq,
           brftab, segment, exptime, stimrate, input, stimfile):
    """Compute thermal distortion parameters from stim positions.

    This function loops over intervals of time, and within each interval
    calls routines to find the stim locations and compute the thermal
    distortion parameters.

    If a stimfile was specified, it will be opened (append mode), and the
    stim positions for each time interval will be written to the file.
    (The 'input' argument is included in the calling sequence only for the
    purpose of writing its name to the stimfile.)

    arguments:
    time          array of event times
    x, y          arrays of detector X and Y coordinates
    dq            array of data quality flags
    brftab        name of baseline reference data table
    segment       segment name (for FUV)
    exptime       exposure time (for computing livetime)
    stimrate      input count rate for stims (for computing livetime)
    input         name of raw file (for writing to stimfile)
    stimfile      name of text file to which stim locations will be appended

    The function value is a tuple:
      (stim_param, stim_locations, stim_countrate, stim_livetime)
    stim_param is a dictionary of lists:  (i0, i1, x0, xslope, y0, yslope)
    stim_locations is a tuple of stim positions:  (avg_s1, avg_s2)
    stim_countrate is the observed count rate for the stims
    stim_livetime is the live time computed from the input and observed
      stim rate

    For each i:
      i0[i], i1[i] is the slice of indices in 'events' corresponding to the
        ith time interval.  Each such interval is of length dt_thermal in
        duration (except possibly the last, which could be shorter).
      x0[i] and xslope[i] are the intercept and slope respectively of the
        linear correction to the X positions (the more rapidly varying
        direction).
      y0[i] and yslope[i] are the intercept and slope for the linear
        correction to the Y positions.

    avg_s1[0] is the average Y location of the first stim.
    avg_s1[1] is the average X location of the first stim.
    avg_s2[0] is the average Y location of the second stim.
    avg_s2[1] is the average X location of the second stim.
    """

    if stimfile is None:
        fd = None
    else:
        fd = open (stimfile, "a")
        fd.write ("# %s\n" % input)

    nevents = len (time)

    brf_info = cosutil.getTable (brftab, filter={"segment": segment},
                   exactly_one=True)

    # Find stims and compute parameters every dt_thermal seconds.
    fd_brf = pyfits.open (brftab, mode="readonly")
    dt_thermal = fd_brf[1].header["timestep"]
    fd_brf.close()
    cosutil.printMsg (
"Compute thermal corrections from stim positions; timestep is %.6g s:" \
        % dt_thermal, VERY_VERBOSE)

    sx1 = brf_info.field ("sx1")[0]
    sy1 = brf_info.field ("sy1")[0]
    sx2 = brf_info.field ("sx2")[0]
    sy2 = brf_info.field ("sy2")[0]
    xwidth = brf_info.field ("xwidth")[0]
    ywidth = brf_info.field ("ywidth")[0]

    # These are the reference locations of the stims.
    s1_ref = (sy1, sx1)
    s2_ref = (sy2, sx2)

    counts1 = 0.
    counts2 = 0.
    i0 = []
    i1 = []
    x0 = []
    xslope = []
    y0 = []
    yslope = []
    if fd is not None:
        fd.write ("# t0 t1 stim_locations\n")

    t0 = time[0]
    t1 = t0 + dt_thermal
    sumstim = (0, 0., 0., 0, 0., 0.)
    while t0 < time[nevents-1]:

        # time[i:j] matches t0 to t1.
        try:
            (i, j) = ccos.range (time, t0, t1)
        except:
            t0 = t1
            t1 = t0 + dt_thermal
            continue

        (s1, counts1) = findStim (x[i:j], y[i:j], s1_ref, xwidth, ywidth)

        (s2, counts2) = findStim (x[i:j], y[i:j], s2_ref, xwidth, ywidth)

        # Increment sums for averaging the stim positions.
        sumstim = updateStimSum (sumstim, j-i, s1, s2)

        if fd is not None:
            fd.write ("%.0f %.0f" % (t0, t1))
            if s1[0] is None:
                fd.write ("  INDEF INDEF")
            else:
                fd.write ("  %.1f %.1f" % (s1[1], s1[0]))
            if s2[0] is None:
                fd.write ("  INDEF INDEF\n")
            else:
                fd.write ("  %.1f %.1f\n" % (s2[1], s2[0]))
        if s1[0] is None or s2[0] is None:
            msg = "  %9d ... %9d  " % (i, j-1)
            if s1[0] is None:
                msg += "  stim1 not found; "
            else:
                msg += "  %.1f %.1f" % (s1[1], s1[0])
            if s2[0] is None:
                msg += "  stim2 not found"
            else:
                msg += "  %.1f %.1f" % (s2[1], s2[0])
            cosutil.printWarning (msg)
        elif cosutil.checkVerbosity (VERY_VERBOSE):
            msg = "  %9d ... %9d    %.1f %.1f  %.1f %.1f" % \
                     (i, j-1, s1[1], s1[0], s2[1], s2[0])
            cosutil.printMsg (msg, VERY_VERBOSE)

        (x0_n, xslope_n, y0_n, yslope_n) = thermalParam (s1, s2, s1_ref, s2_ref)
        i0.append (i)
        i1.append (j)
        x0.append (x0_n)
        xslope.append (xslope_n)
        y0.append (y0_n)
        yslope.append (yslope_n)
        t0 = t1
        t1 = t0 + dt_thermal

    # Compute the average of the stim positions.
    avg_s1 = [-1., -1.]
    avg_s2 = [-1., -1.]
    if sumstim[0] > 0:
        avg_s1[0] = sumstim[1] / sumstim[0]     # y
        avg_s1[1] = sumstim[2] / sumstim[0]     # x
    if sumstim[3] > 0:
        avg_s2[0] = sumstim[4] / sumstim[3]
        avg_s2[1] = sumstim[5] / sumstim[3]

    countrate = (counts1 + counts2) / (2. * exptime)
    if stimrate > 0. and countrate > 0.:
        livetime = countrate / stimrate
    else:
        livetime = 1.

    if fd is not None:
        fd.close()

    stim_param = {"i0": i0, "i1": i1,
                  "x0": x0, "xslope": xslope,
                  "y0": y0, "yslope": yslope}

    return (stim_param, (avg_s1, avg_s2), countrate, livetime)

def findStim (x, y, stim_ref, xwidth, ywidth):
    """Find one stim in time-tag data.

    arguments:
    x, y          arrays of detector X and Y coordinates
    stim_ref      reference position (y, x) for the stim
    xwidth        half width of the search region in X
    ywidth        half width of the search region in Y
    """

    # This is the search region for finding the stim.
    sxlow  = stim_ref[1] - xwidth
    sxhigh = stim_ref[1] + xwidth
    sylow  = stim_ref[0] - ywidth
    syhigh = stim_ref[0] + ywidth

    # Initial value of mask is 1. (which in this case means "good").
    mask = N.ones (len (x), dtype=N.float32)

    # Now set mask to 0. ("bad") outside the search region.
    mask = N.where (x > sxhigh, 0., mask)
    mask = N.where (x < sxlow,  0., mask)
    mask = N.where (y > syhigh, 0., mask)
    mask = N.where (y < sylow,  0., mask)
    n = N.sum (mask)
    if n > 0.:
        # The stim reference position is subtracted before taking the sum
        # and then added back to the average in order to reduce the
        # possibility of numerical roundoff errors.
        sumx = N.sum ((x-stim_ref[1]) * mask)
        sumy = N.sum ((y-stim_ref[0]) * mask)
        sx = sumx / n + stim_ref[1]
        sy = sumy / n + stim_ref[0]
    else:
        sx = None
        sy = None

    return ((sy, sx), n)

def updateStimSum (sumstim, nevents, s1, s2):
    """Update sums for averages of stim positions.

    arguments:
    sumstim    tuple with current sums:
                 n1     number of events for first stim
                 sum1y  sum for first stim, Y coordinate
                 sum1x  sum for first stim, X coordinate
                 n2     number of events for second stim
                 sum2y  sum for second stim, Y coordinate
                 sum2x  sum for second stim, X coordinate
    nevents    number of events in current time interval
    s1         tuple of (y,x) coordinates of the first stim in current
                 interval
    s2         same as s1, but for the second stim

    The function value is an updated sumstim tuple.

    nevents is used as a weight when incrementing the sums.  n1 and n2 are
    the total number of events with defined first and second stim positions
    respectively.
    """

    (n1, sum1y, sum1x, n2, sum2y, sum2x) = sumstim

    if s1[0] is not None and s1[1] is not None:
        n1 = n1 + nevents
        sum1x = sum1x + s1[1] * nevents
        sum1y = sum1y + s1[0] * nevents

    if s2[0] is not None and s2[1] is not None:
        n2 = n2 + nevents
        sum2x = sum2x + s2[1] * nevents
        sum2y = sum2y + s2[0] * nevents

    return (n1, sum1y, sum1x, n2, sum2y, sum2x)

def stimKeywords (hdr, segment, stim_locations):
    """Update keywords for the locations of the stims.

    arguments:
    hdr             the input events extension header (updated)
    segment         FUVA or FUVB
    stim_locations  a two-element list, one for each stim; each element is
                    the (y, x) coordinates of the stim
    """

    seg = segment[-1]           # "A" or "B"

    # Positions of the first (left) and second (right) stims.
    (s1, s2) = stim_locations

    if s1[0] is None or s1[1] is None:
        hdr["STIM"+seg+"_LX"] = -1.
        hdr["STIM"+seg+"_LY"] = -1.
    else:
        hdr["STIM"+seg+"_LX"] = s1[1]
        hdr["STIM"+seg+"_LY"] = s1[0]

    if s2[0] is None or s2[1] is None:
        hdr["STIM"+seg+"_RX"] = -1.
        hdr["STIM"+seg+"_RY"] = -1.
    else:
        hdr["STIM"+seg+"_RX"] = s2[1]
        hdr["STIM"+seg+"_RY"] = s2[0]

def thermalParam (s1, s2, s1_ref, s2_ref):
    """Compute linear thermal distortion correction from stim positions.

    arguments:
    s1          measured location in raw data of first stim (y, x)
    s2          measured location in raw data of second stim (y, x)
    s1_ref      reference location of first stim (y, x)
    s2_ref      reference location of second stim (y, x)
    """

    if s1[0] is None or s2[0] is None:

        xslope = 1.
        xintercept = 0.
        yslope = 1.
        yintercept = 0.

    else:

        (sy1, sx1) = s1_ref
        (sy2, sx2) = s2_ref

        xslope = (sx2 - sx1) / (s2[1] - s1[1])
        xintercept = sx1 - s1[1] * xslope

        yslope = (sy2 - sy1) / (s2[0] - s1[0])
        yintercept = sy1 - s1[0] * yslope

    return (xintercept, xslope, yintercept, yslope)

def doRandcorr (events, info, switches, phdr):
    """Add pseudo-random numbers to x and y coordinates.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    phdr          the input primary header
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch ("RANDCORR", switches)
        if switches["randcorr"] == "PERFORM":
            cosutil.addRandomNumbers (events.field (xcorr),
                    events.field (ycorr), info["randseed"])
            phdr["randcorr"] = "COMPLETE"

def doTempcorr (stim_param, events, info, switches, reffiles, phdr):
    """Apply thermal distortion correction.

    arguments:
    stim_param    a dictionary of lists, with keys
                    i0, i1, x0, xslope, y0, yslope
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    phdr          the input primary header
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch ("TEMPCORR", switches)
        if switches["tempcorr"] == "PERFORM":
            cosutil.printRef ("BRFTAB", reffiles)
            # The function value is true if a correction was actually applied.
            if thermalDistortion (events.field (xcorr),
                                  events.field (ycorr), stim_param):
                phdr["tempcorr"] = "COMPLETE"
            else:
                phdr["tempcorr"] = "SKIPPED"
                cosutil.printWarning ("TEMPCORR was skipped")

def thermalDistortion (x, y, stim_param):
    """Apply thermal distortion correction to positions in events list.

    arguments:
    x, y          arrays of pixel coordinates of events
    stim_param    a dictionary of lists, with keys
                    i0, i1, x0, xslope, y0, yslope

    The function value will be true if a correction was actually applied.
    No correction is necessary and none will be applied if the slopes are
    all 0 and the intercepts are all 1.
    """

    # These are the parameters found by computeThermalParam.
    x0 = stim_param["x0"]
    xslope = stim_param["xslope"]
    y0 = stim_param["y0"]
    yslope = stim_param["yslope"]

    actually_done = 0

    if stim_param.has_key ("i0"):
        i0 = stim_param["i0"]
        i1 = stim_param["i1"]
    else:
        i0 = [0]
        i1 = [len (x)]

    for n in range (len (i0)):
        i = i0[n]
        j = i1[n]
        if x0[n] != 0. or xslope[n] != 1. or \
           y0[n] != 0. or yslope[n] != 1.:
            x[i:j] = x0[n] + x[i:j] * xslope[n]
            y[i:j] = y0[n] + y[i:j] * yslope[n]
            actually_done = 1

    return actually_done

def doGeocorr (events, info, switches, reffiles, phdr):
    """Apply geometric correction.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    phdr          the input primary header
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch ("GEOCORR", switches)
        if switches["geocorr"] == "PERFORM":
            cosutil.printRef ("GEOFILE", reffiles)
            cosutil.printSwitch ("IGEOCORR", switches)
            cosutil.geometricDistortion (events.field (xcorr),
                    events.field (ycorr), reffiles["geofile"],
                    info["segment"], switches["igeocorr"])
            phdr["geocorr"] = "COMPLETE"
            if switches["igeocorr"] == "PERFORM":
                phdr["igeocorr"] = "COMPLETE"

def doDqicorr (events, info, switches, reffiles, phdr, hdr):
    """Create a data quality array, initialized from the DQI table.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    phdr          the input primary header
    hdr           the input events extension header

    This function applies the data quality initialization table (bpixtab) to
    two arrays, the 2-D DQ image extension and the 1-D DQ events table column.

    The 2-D DQ image array dq_array is created and initialized to zero.
    The 1-D DQ events table column, on the other hand, is not initialized
    because it may already contain meaningful flags from pulse-height or time
    filtering.  Note that flags for pulse-height or time filtering that are
    set in the 1-D DQ table column are _not_ included in the 2-D image array,
    since they would be associated with either specific events or time
    intervals, rather than spatial regions on the detector.

    The function value is the 2-D data quality image array, or None if
    dqicorr is not PERFORM.
    """

    cosutil.printSwitch ("DQICORR", switches)

    if switches["dqicorr"] == "PERFORM":

        cosutil.printRef ("BPIXTAB", reffiles)

        # Update the dq column in the events list with the bpixtab regions.
        dq_info = cosutil.getTable (reffiles["bpixtab"],
                            filter={"segment": info["segment"]})
        if dq_info is not None:
            ccos.applydq (dq_info.field ("lx"), dq_info.field ("ly"),
                          dq_info.field ("dx"), dq_info.field ("dy"),
                          dq_info.field ("dq"),
                          events.field (xcorr), events.field (ycorr),
                          events.field ("dq"))
            del dq_info

        # Create an initially zero 2-D data quality extension array.
        dq_array = N.zeros (info["npix"], dtype=N.int16)

        # Copy values from the bpixtab to the dq_array, applying an offset
        # depending on the Doppler shift.
        cosutil.updateDQArray (reffiles["bpixtab"], info,
                      switches["doppcorr"], dq_array)

        # Flag regions that are outside any subarray as out of bounds.
        cosutil.flagOutOfBounds (phdr, hdr, dq_array)

        phdr["dqicorr"] = "COMPLETE"
    else:
        dq_array = None

    return dq_array

def doDoppcorr (events, info, switches, reffiles, phdr, hdr):
    """Apply Doppler correction to the x and y pixel coordinates.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    phdr          the input primary header
    hdr           the input events extension header
    """

    if info["obstype"] == "SPECTROSCOPIC":
        cosutil.printSwitch ("DOPPCORR", switches)
        cosutil.printSwitch ("HELCORR", switches)

    # xi and eta are the columns of pixel coordinates for the
    # dispersion and cross-dispersion directions respectively.
    # (explicit column names are used here for clarity)
    if info["detector"] == "FUV":
        xi = events.field ("xcorr")
        eta = events.field ("ycorr")
        dopp = events.field ("xdopp")
    else:
        xi = events.field ("rawy")
        eta = events.field ("rawx")
        dopp = events.field ("ydopp")

    if switches["doppcorr"] == "PERFORM" or switches["helcorr"] == "PERFORM":

        cosutil.printRef ("DISPTAB", reffiles)
        if info["detector"] == "FUV":
            cosutil.printRef ("BRFTAB", reffiles)

        # This array of flags indicates which events should be corrected.
        shift_flags = dopplerRegions (eta, info, reffiles)

        if shift_flags is None:
            # Correct all events.
            dopp[:] = dopplerCorrection (hdr, events.field ("time"),
                        xi, info, switches["doppcorr"], switches["helcorr"],
                        reffiles["disptab"])
        else:
            # Apply the orbital and/or heliocentric Doppler correction to
            # the flagged events.
            dopp[:] = N.where (shift_flags, \
                        dopplerCorrection (hdr, events.field ("time"),
                          xi, info, switches["doppcorr"], switches["helcorr"],
                          reffiles["disptab"]),
                        xi)

        if switches["doppcorr"] == "PERFORM":
            phdr["doppcorr"] = "COMPLETE"
        if switches["helcorr"] == "PERFORM":
            phdr["helcorr"] = "COMPLETE"
    else:
        dopp[:] = xi

def dopplerRegions (eta, info, reffiles):
    """Determine the regions over which Doppler shift should be applied.

    arguments:
    eta           pixel coordinates in cross-dispersion direction
    info          dictionary of keywords and values
    reffiles      dictionary of reference file names

    The function value is a Boolean array, true for events that are
    within the region for which it would be reasonable to apply Doppler
    or heliocentric correction.  No mask is needed for NUV if tagflash
    wavecals are not used, and in this case the function value will be None.
    """

    if info["detector"] == "FUV":
        (b_low, b_high) = cosutil.activeArea (info["segment"],
                          reffiles["brftab"])
        shift_flags = N.zeros (len (eta), dtype=N.bool8)
        shift_flags |= N.logical_and (eta >= b_low, eta <= b_high)

    if info["tagflash"]:

        # segment and aperture will be added to the filter below.
        filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"]}

        # The computation of the 'boundary' variable makes assumptions
        # about the relative locations of the PSA and WCA regions on the
        # FUV and NUV detectors.  For FUV, the PSA spectral region is
        # at lower Y pixel numbers; for NUV the PSA region is at higher
        # X pixel numbers.

        if info["detector"] == "FUV":
            filter["segment"] = info["segment"]
        else:
            filter["segment"] = "NUVC"
        filter["aperture"] = "PSA"
        xtract_info = cosutil.getTable (reffiles["xtractab"],
                            filter, exactly_one=True)
        b_spec_psa = xtract_info.field ("b_spec")[0]

        if info["detector"] == "NUV":
            filter["segment"] = "NUVA"
        filter["aperture"] = "WCA"
        xtract_info = cosutil.getTable (reffiles["xtractab"],
                            filter, exactly_one=True)
        b_spec_wca = xtract_info.field ("b_spec")[0]

        boundary = int (round ((b_spec_psa + b_spec_wca) / 2.))

        if info["detector"] == "FUV":
            shift_flags &= (eta < boundary)
        else:
            shift_flags = eta > boundary

    elif info["detector"] == "NUV":

        shift_flags = None

    return shift_flags

def dopplerCorrection (hdr, time, xi, info, doppcorr, helcorr, disptab):
    """Apply orbital and heliocentric Doppler correction.

    arguments:
    hdr        events extension header; v_helio will be assigned
    time       array of event times
    xi         array of detector coordinates in dispersion direction
    info       dictionary of keywords and values
    doppcorr   PERFORM if orbital Doppler correction is to be done
    helcorr    PERFORM if heliocentric Doppler correction is to be done

    The function value is the array of Doppler-corrected X (or Y if NUV)
    pixel coordinates.
    """

    expstart = info["expstart"]
    doppmag  = info["doppmag"]
    orbitper = info["orbitper"]
    doppzero = info["doppzero"]

    # Compute the Doppler corrected pixel positions.
    if doppcorr == "PERFORM":
        xd = orbitalDoppler (time, xi, expstart, doppmag, doppzero, orbitper)
    else:
        xd = xi.copy()

    if helcorr == "PERFORM":

        # get midpoint of exposure, MJD
        t_mid = expstart + (time[0] + time[len(time)-1]) / 2. / SEC_PER_DAY

        # Compute radial velocity and heliocentric correction factor.
        radvel = heliocentricVelocity (t_mid, info["ra_targ"], info["dec_targ"])
        helio_factor = -radvel  / SPEED_OF_LIGHT
        hdr.update ("v_helio", radvel)
        info["v_helio"] = radvel

        # Get the dispersion relation.
        filter = {"opt_elem": info["opt_elem"],
                  "aperture": info["aperture"],
                  "cenwave": info["cenwave"]}
        if info["detector"] == "FUV":
            filter["segment"] = info["segment"]
        else:
            filter["segment"] = "NUVB"
        disp_info = cosutil.getTable (disptab, filter, exactly_one=True)
        ncoeff = disp_info.field ("nelem")[0]
        coeff = disp_info.field ("coeff")[0][0:ncoeff]
        xd = heliocentricDoppler (helio_factor, xd, coeff)

    return xd

def orbitalDoppler (time, xi, expstart, doppmag, doppzero, orbitper):
    """Apply Doppler correction for HST orbital motion.

    arguments:
    time          array of times of events (seconds)
    xi            array of pixel coordinates of events, in the dispersion
                    direction
    expstart      exposure start time (MJD)
    doppmag       magnitude of Doppler shift (pixels)
    doppzero      time when orbital Doppler shift is zero and increasing (MJD)
    orbitper      orbital period of HST (seconds)
    """

    # t is the time of each event in seconds since doppzero.
    t = (expstart - doppzero) * SEC_PER_DAY + time

    # For both FUV and NUV, wavelengths increase toward smaller pixel number.
    shift = doppmag * N.sin (2. * N.pi * t / orbitper)

    return xi + shift

def heliocentricDoppler (helio_factor, xi, coeff):
    """Apply Doppler correction for Earth's heliocentric motion.

    If wl is the observed wavelength and helio_factor is the heliocentric
    correction factor (-radial_velocity / c), then the heliocentric-
    corrected wavelength would be:
        wl * (1 + helio_factor).
    Note that helio_factor has opposite sign from radial velocity.

    arguments:
    helio_factor  heliocentric correction factor
    xi            array of pixel coordinates of events, in the dispersion
                    direction
    coeff         array of polynomial coefficients for the dispersion relation
    """

    # wl[i] and dwl[i] are the wavelength and dispersion at pixel xi[i].
    wl = cosutil.evalDisp (xi, coeff)
    dwl = cosutil.evalDerivDisp (xi, coeff)
    shift = helio_factor * wl / dwl                  # shift in pixels

    return xi + shift

def heliocentricVelocity (t_mid, ra_targ, dec_targ):
    """Compute heliocentric radial velocity.

    arguments:
    t_mid         time (MJD) at the middle of the exposure
    ra_targ       right ascension of the target
    dec_targ      declination of the target
    """

    deg_to_rad = math.pi / 180.
    eps = 23.439 * deg_to_rad           # obliquity of Earth's axis
    orb_v = 29.786                      # speed of Earth around Sun, km/s

    ra  = ra_targ * deg_to_rad
    dec = dec_targ * deg_to_rad

    # target will be a unit vector toward the target;
    # velocity will be Earth's orbital velocity in km/s.
    target = N.zeros (3, dtype=N.float64)
    velocity = N.zeros (3, dtype=N.float64)

    target[0] = math.cos (dec) * math.cos (ra)
    target[1] = math.cos (dec) * math.sin (ra)
    target[2] = math.sin (dec)

    # Earth's ecliptic longitude
    ecl_long = (100.461 + 0.9856474 * (t_mid - 51544.5)) * deg_to_rad

    velocity[0] = -orb_v * math.sin (ecl_long)
    velocity[1] = orb_v * math.cos (ecl_long) * math.cos (eps)
    velocity[2] = orb_v * math.cos (ecl_long) * math.sin (eps)

    radvel = -N.dot (velocity, target)

    return radvel

def doFlatcorr (events, info, switches, reffiles, phdr):
    """Apply flat field correction.

    arguments:
    events        the data unit containing the events table
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    phdr          the input primary header
    """

    cosutil.printSwitch ("FLATCORR", switches)

    if switches["flatcorr"] == "PERFORM":

        cosutil.printRef ("FLATFILE", reffiles)

        fd = pyfits.open (reffiles["flatfile"], mode="readonly", memmap=0)
        # fd = pyfits.open (reffiles["flatfile"], mode="readonly", memmap=1)

        if info["detector"] == "NUV":
            hdu = fd[1]
        else:
            hdu = fd[(info["segment"],1)]

        origin_x = hdu.header.get ("origin_x", 0)
        origin_y = hdu.header.get ("origin_y", 0)

        ccos.applyflat (events.field (xcorr), events.field (ycorr),
                        events.field ("epsilon"), hdu.data, origin_x, origin_y)

        fd.close()

        phdr["flatcorr"] = "COMPLETE"

def doDeadcorr (events, input, info, switches, reffiles, phdr,
            stim_countrate, stim_livetime, livetimefile):
    """Correct for deadtime.

    arguments:
    events          the data unit containing the events table
    input           name of raw file (for writing to livetimefile)
    info            dictionary of header keywords and values
    switches        dictionary of calibration switches
    reffiles        dictionary of reference file names
    phdr            the input primary header
    stim_countrate  the observed count rate for the stims
    stim_livetime   live time computed from the input and observed stim rate
    livetimefile    name of output text file for livetime factors (or None)
    """

    cosutil.printSwitch ("DEADCORR", switches)
    if switches["deadcorr"] == "PERFORM":
        cosutil.printRef ("DEADTAB", reffiles)
        deadtimeCorrection (events, reffiles["deadtab"], info["segment"],
                info["exptime"],
                stim_countrate, stim_livetime, info["countrate"],
                input, livetimefile)
        phdr["deadcorr"] = "COMPLETE"

def deadtimeCorrection (events, deadtab, segment, exptime,
            stim_countrate, stim_livetime, dec_countrate,
            input, livetimefile):
    """Apply livetime factor to correct for dead time.

    arguments:
    events          the data unit containing the events table
    deadtab         reference table of count rates and livetime factors
    segment         segment name (for FUV)
    exptime         exposure time
    stim_countrate  the observed count rate for the stims
    stim_livetime   live time computed from the input and observed stim rate
    dec_countrate   the count rate from the digital event counter
    input           name of input raw file (for writing to livetimefile)
    livetimefile    name of output text file for livetime factors (or None)
    """

    if livetimefile is None:
        fd = None
    else:
        fd = open (livetimefile, "a")
        fd.write ("# %s\n" % input)
        fd.write ("# t0 t1 countrate livetime\n")

    time = cosutil.getColCopy (data=events, column="time")
    epsilon = events.field ("epsilon")
    nevents = len (time)

    live_info = cosutil.getTable (deadtab, filter={"segment": segment},
                at_least_one=True)
    obs_rate = live_info.field ("obs_rate")
    live_factor = live_info.field ("livetime")

    # Use counts over dt_deadtime seconds to compute livetime.
    fd_dead = pyfits.open (deadtab, mode="readonly")
    dt_deadtime = fd_dead[1].header["timestep"]
    fd_dead.close()
    cosutil.printMsg ("Compute livetime factor; timestep is %.6g s:" \
                  % dt_deadtime, VERY_VERBOSE)

    t0 = time[0]
    t1 = t0 + dt_deadtime
    last_time = time[nevents-1]
    while t0 < last_time:

        # time[i:j] matches t0 to t1.
        try:
            (i, j) = ccos.range (time, t0, t1)
        except:
            t0 = t1
            t1 = t0 + dt_deadtime
            continue

        if t1 < last_time:
            countrate = (j - i + 1.) / dt_deadtime
        elif t0 < last_time:
            countrate = (j - i + 1.) / (last_time - t0)
        else:
            countrate = 0.
        livetime = determineLivetime (countrate, obs_rate, live_factor)
        if livetime > 0.:
            epsilon[i:j] = epsilon[i:j] / livetime

        if fd is not None:
            fd.write ("%.0f %.0f %.6g %.6g\n" % (t0, t1, countrate, livetime))

        t0 = t1
        t1 = t0 + dt_deadtime

    # This livetime value is effectively an average over the entire exposure.
    if time[nevents-1] > time[0]:
        countrate = float (nevents) / (time[nevents-1] - time[0])
    elif exptime > 0.:
        countrate = float (nevents) / exptime
    else:
        countrate = 0.
    livetime = determineLivetime (countrate, obs_rate, live_factor)

    # dec_countrate is from DEVENTA, DEVENTB or from MEVENTS.
    dec_livetime = determineLivetime (dec_countrate, obs_rate, live_factor)

    print_details = (cosutil.checkVerbosity (VERY_VERBOSE))     # initial value
    if abs (stim_livetime - livetime) > LIVETIME_CRITERION * livetime or \
        abs (dec_livetime - livetime) > LIVETIME_CRITERION * livetime:
        cosutil.printWarning ("livetime estimates differ.")
        print_details = 1

    if print_details:
        printLiveInfo (segment, countrate, livetime,
                stim_countrate, stim_livetime,
                dec_countrate, dec_livetime)

    if fd is not None:
        printLiveInfo (segment, countrate, livetime,
                stim_countrate, stim_livetime,
                dec_countrate, dec_livetime, fd=fd)
        fd.close()

def printLiveInfo (segment, countrate, livetime,
                stim_countrate, stim_livetime,
                dec_countrate, dec_livetime, fd=None):
    """Print or write information about livetime.

    arguments:
    segment         segment name (for setting keyword name for DEC count rate)
    countrate       observed count rate, from events table
    livetime        livetime factor derived from countrate
    stim_countrate  the observed count rate for the stims
    stim_livetime   livetime factor computed from the input and observed
                      stim rate
    dec_countrate   the count rate from the digital event counter
    dec_livetime    livetime factor computed from dec_countrate
    fd              None if printing to trailer; an fd for printing to a
                      log file
    """

    if segment == "FUVA":
        kwd = "DEVENTA"
    elif segment == "FUVB":
        kwd = "DEVENTB"
    else:
        kwd = "MEVENTS"

    messages = []
    messages.append ("Average event rate and livetime:  %.6g, %6.4f" % \
              (countrate, livetime))
    if segment == "FUVA" or segment == "FUVB":
        messages.append ("Stim countrate and livetime:  %.6g, %6.4f" % \
                  (stim_countrate, stim_livetime))
    messages.append ("Countrate and livetime from %s:  %.6g, %6.4f" % \
              (kwd, dec_countrate, dec_livetime))

    if fd is None:
        for msg in messages:
            cosutil.printMsg (msg)
    else:
        fd.write ("\n")
        for msg in messages:
            fd.write ("# " + msg + "\n")

def determineLivetime (countrate, obs_rate, live_factor):
    """Compute livetime factor from observed count rate.

    This is just linear interpolation in live_factor vs obs_rate.

    arguments:
    countrate     observed count rate
    obs_rate      list of observed count rates, from deadtab reference table
    live_factor   list of livetime factors corresponding to obs_rate, from
                    deadtab

    The function value is the interpolated livetime factor.
    """

    n = len (obs_rate)

    if countrate <= 0.:
        livetime = 1.
    elif n == 1:
        livetime = live_factor[0]
    elif countrate < obs_rate[0]:
        livetime = 1.
    elif countrate >= obs_rate[n-1]:
        livetime = live_factor[n-1]
    else:
        # Find the interval containing the observed count rate, and interpolate.
        for i in range (n-1):
            if countrate < obs_rate[i+1]:
                p = (countrate - obs_rate[i]) / (obs_rate[i+1] - obs_rate[i])
                q = 1. - p
                livetime = live_factor[i] * q + live_factor[i+1] * p
                break

    return livetime

def writeNull (input, output, outcounts, phdr, events_hdu):
    """Write output files; images will have null data portions.

    The outtag file has already been written, so we only need to write
    the output and outcounts files.

    arguments:
    input         name of input file
    output        name of the output file for flat-fielded count-rate image
    outcounts     name of the output file for count-rate image
    phdr          primary header
    events_hdu    hdu for events extension
    """

    cosutil.printWarning ("No data in " + input)
    makeImage (outcounts, phdr, events_hdu.header, None, None, None)
    makeImage (output, phdr, events_hdu.header, None, None, None)

def writeImages (x, y, epsilon, dq,
                 phdr, hdr, dq_array, npix, exptime,
                 outcounts=None, output=None):
    """Bin events to images, and write to output files.

    arguments:
    x, y          arrays of pixel coordinates of events
    epsilon       weight column
    dq            data quality column
    phdr          the input primary header
    hdr           the input events extension header
    dq_array      the data quality array
    npix          the array shape (ny, nx)
    exptime       the exposure time
    outcounts     name of the output file for count-rate image
    output        name of the output file for flat-fielded count-rate image
    """

    # notation:
    # t = exposure time (exptime)
    # C = sum of counts
    # C_rate = C / t
    # E = C / flat_field, i.e. "effective" counts
    # E_rate = E / t
    # the corresponding error arrays are:
    # errC = sqrt (C)
    # errC_rate = sqrt (C) / t
    # errE = sqrt (C) / flat_field = sqrt (C) * (E / C) = E / sqrt (C)
    # errE_rate = errE / t = (E / sqrt (C)) / t
    #           = (E / t) / (sqrt (C) / t) / t
    #           =  E_rate / errC_rate / t

    if outcounts is not None:
        cosutil.printMsg ("writing file %s ..." % outcounts, VERY_VERBOSE)

    # Get the bit mask for "serious" data quality flags from the header,
    # but remove the "near edge" and "out of bounds" bits so that we'll
    # include data over the entire detector.
    sdqflags = hdr.get ("sdqflags", 32767)
    sdqflags -= (DQ_NEAR_EDGE + DQ_OUT_OF_BOUNDS)

    # First make an image array in which each input event counts as one,
    # i.e. ignoring flat field and deadtime corrections.
    C_rate = N.zeros (npix, dtype=N.float32)
    ccos.binevents (x, y, C_rate, dq, sdqflags)

    errC_rate = N.sqrt (C_rate) / exptime

    if outcounts is not None:
        C_rate /= exptime
        makeImage (outcounts, phdr, hdr, C_rate, errC_rate, dq_array)
    del C_rate                          # but we still need errC_rate

    if output is None:
        return                          # nothing further to do

    cosutil.printMsg ("writing file %s ..." % output, VERY_VERBOSE)

    # Make an image array where event number i has weight epsilon[i].
    E_rate = N.zeros (npix, dtype=N.float32)
    ccos.binevents (x, y, E_rate, dq, sdqflags, epsilon)

    # errC_rate will likely have a number of zero values, so we
    # have to set those to one before dividing.
    errC_rate = N.where (errC_rate == 0., 1., errC_rate)

    # convert from counts to count rate
    E_rate /= exptime
    errE_rate = E_rate / errC_rate / exptime
    del errC_rate

    makeImage (output, phdr, hdr, E_rate, errE_rate, dq_array)

def makeImage (outimage, phdr, hdr, sci_array, err_array, dq_array):
    """Write a FITS file, based on headers and data arrays.

    arguments:
    output        name of the output file to be written
    phdr          the input primary header
    hdr           the input events extension header
    sci_array     the science data array (may be None)
    err_array     the error estimates array (may be None)
    dq_array      the data quality array (may be None)
    """

    primary_hdu = pyfits.PrimaryHDU (header=phdr)
    fd = pyfits.HDUList (primary_hdu)
    fd[0].header["nextend"] = 3
    cosutil.updateFilename (fd[0].header, outimage)

    makeImageHDU (fd, hdr, sci_array, name="SCI")
    makeImageHDU (fd, hdr, err_array, name="ERR")
    makeImageHDU (fd, hdr, dq_array, name="DQ")

    fd.writeto (outimage, output_verify='silentfix')

def makeImageHDU (fd, table_hdr, data_array, name="SCI"):
    """Make an image hdu from data and a table header and append to fd.

    arguments:
    fd            pyfits object for FITS file (new hdu will be appended)
    table_hdr     a FITS Header object for a table
    data_array    image data to be appended (may be None)
    name          string to be used for EXTNAME
    """

    # Create an image header from the table header.
    imhdr = cosutil.tableHeaderToImage (table_hdr)
    if name == "DQ":
        imhdr.update ("BUNIT", "UNITLESS")
    else:
        imhdr.update ("BUNIT", "count /s")

    hdu = pyfits.ImageHDU (data=data_array, header=imhdr, name=name)
    fd.append (hdu)

def doStatflag (switches, output, outcounts):
    """Compute statistics and update keywords.

    arguments:
    output        name of the output file for flat-fielded count-rate image
    outcounts     name of the output file for count-rate image
    switches      dictionary of calibration switches
    """

    cosutil.printSwitch ("STATFLAG", switches)
    if switches["statflag"] == "PERFORM":
        cosutil.doImageStat (outcounts)
        cosutil.doImageStat (output)

def flag_gti (time, dq, gti):
    """Flag events in dq that are outside any good time interval.

    xxx This function may be unnecessary, in the sense that there probably
    won't be any events that are outside any good time interval.

    arguments:
    time          the time column in the events table
    dq            the data quality column in the events table (updated in-place)
    gti           list of good time intervals
    """

    SMALL_INCR = 0.02           # smaller than the timestep of 0.032 s

    if len (gti) < 1 or len (time) < 1:
        return

    # Nothing to do if there is only one GTI and it covers the entire
    # time range.
    if len (gti) == 1 and \
      (time[0] >= gti[0][0] and time[-1] <= gti[0][1]):
        return

    dq[:] |= DQ_BAD_TIME

    for (t_start, t_stop) in gti:
        (i0, i1) = ccos.range (time, t_start, t_stop+SMALL_INCR)
        dq[i0:i1] &= ~DQ_BAD_TIME
