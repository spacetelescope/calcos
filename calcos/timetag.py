from __future__ import absolute_import, division         # confidence high
import math
import os
import time
import numpy as np
from numpy import random
import astropy.io.fits as fits

from . import cosutil
from . import burst
from . import ccos
from . import concurrent
from . import dispersion
from . import phot
from . import shiftfile
from . import timeline
from . import wavecal
from . import trace
from .calcosparam import *       # parameter definitions

# These are column names in the corrtag table.  The default values are
# appropriate for FUV data.  These can be reset in setCorrColNames().

xcorr = "xcorr"
ycorr = "ycorr"
xdopp = "xdopp"
ydopp = "ycorr"
xfull = "xfull"
yfull = "yfull"

# This will be a Boolean array, true for events that are within the
# active area.  This is only needed for FUV, but it will also be defined
# for NUV (all True).
active_area = None

# Used as a default value in blurDQ.  The actual value should be
# gotten via keyword WIDEN in the BPIXTAB table header.
# (Should move this to calcosparam as it's in cosutil as well)
PIXEL_FRACTION = 0.25

def timetagBasicCalibration(input, inpha, outtag,
                            output, outcounts, outflash, outcsum,
                            cl_args,
                            info, switches, reffiles,
                            wavecal_info):
    """Do the basic processing for either time-tag or accum data.

    The function value will be zero if there was no problem,
    and it will be one if there was no input data.

    Parameters
    ----------
    input: str
        Name of the input file.

    inpha: str
        Name of the input file containing the pulse height histogram (FUV
        ACCUM only).

    outtag: str
        Name of the output file for corrected time-tag data.

    output: str
        Name of the output file for flat-fielded count-rate image.

    outcounts: str
        Name of the output file for count-rate image.

    outflash: str or None
        Name of the output file for tagflash wavecal spectra.

    outcsum: str or None
        Name of the output image for OPUS to add to cumulative image.

    cl_args: dictionary
        Some of the command-line arguments.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    wavecal_info: list of dictionaries
        When wavecal exposures were processed, the results were stored in
        dictionaries in this list.

    Returns
    -------
    status: int
        0 is OK
        1 means there were no rows in the input table
    """
    input_path = os.path.dirname(input)
    if info["obsmode"] == "TIME-TAG":
        cosutil.printIntro("TIME-TAG calibration")
        names = [("Input", input),
                 ("OutTag", outtag),
                 ("OutFlt", output),
                 ("OutCounts", outcounts)]
        if outflash is not None:
            names.append(("OutFlash", outflash))
        if outcsum is not None:
            names.append(("OutCsum", outcsum))
        cosutil.printFilenames(names,
                               shift_file=cl_args["shift_file"],
                               stimfile=cl_args["stimfile"],
                               livetimefile=cl_args["livetimefile"])
        cosutil.printMode(info)

    # Copy data from the input file to the output.  Then open the output
    # file read/write.
    if info["obsmode"] == "TIME-TAG":
        nrows = cosutil.writeOutputEvents(input, outtag)
    ofd = fits.open(outtag, mode="update")
    if ofd["EVENTS"].data is None:
        nrows = 0
    else:
        nrows = len(ofd["EVENTS"].data)

    # events_hdu is a complete astropy.io.fits HDU object (i.e., header plus data),
    # while events (assigned below) is just the data, a recarray object.
    events_hdu = ofd["EVENTS"]

    if nrows > 0 and info["obsmode"] == "TIME-TAG":
        # Change orig_exptime to be the range of times in the TIME column.
        time64 = events_hdu.data.field("TIME").astype(np.float64)
        if time64[-1] > time64[0]:
            info["orig_exptime"] = time64[-1] - time64[0]
        del time64

    # Get a copy of the primary header.  This copy will be modified and
    # written to the output image files.
    phdr = ofd[0].header
    # This list also includes the primary header, but we'll ignore this
    # copy of the primary header.
    if info["obsmode"] == "ACCUM" and not info["corrtag_input"]:
        headers = cosutil.getHeaders(input)
        # replace the first extension header so the headers of the
        # pseudo-corrtag table will be updated
        headers[1] = events_hdu.header
    else:
        headers = mkHeaders(phdr, events_hdu.header)

    # Update the switches and reference file names, so the output header
    # will reflect what was actually used.
    cosutil.overrideKeywords(phdr, headers[1], info, switches, reffiles)

    # Update keywords for FUV high voltage.
    if info["detector"] == "FUV":
        updateHVKeywords(headers[1], info, reffiles)

    if nrows == 0:
        writeNull(input, ofd, output, outcounts, outcsum,
                  cl_args, info, phdr, headers)
        ofd.close()
        return 1

    setCorrColNames(info["detector"])

    events = events_hdu.data

    # For corrtag input, reinitialize the DQ column if dqicorr is perform.
    if info["corrtag_input"] and switches["dqicorr"] == "PERFORM":
        events.field("dq")[:] = 0.

    # Set active_area, but note that this is preliminary because we haven't
    # done tempcorr or geocorr yet.
    setActiveArea(events, info, reffiles["brftab"])

    doPhotcorr(info, switches, reffiles["imphttab"], phdr, headers[1])

    badt = doBadtcorr(events, info, switches, reffiles, phdr)

    doRandcorr(events, info, switches, reffiles, phdr)

    (stim_param, stim_countrate, stim_livetime) = initTempcorr(events,
            input, info, switches, reffiles, headers[1],
            cl_args["stimfile"])

    doTempcorr(stim_param, events, info, switches, reffiles, phdr)

    doGeocorr(events, info, switches, reffiles, phdr)

    doDgeocorr(events, info, switches, reffiles, phdr)

    # Set active_area based on (xcorr, ycorr) coordinates.
    setActiveArea(events, info, reffiles["brftab"])

    #
    # The X and Y walk correction need to be independent, and applied to the
    # same xcorr/pha values
    if doWalkCorr(switches):

        xcorrection = doXWalkcorr(events, info, switches, reffiles, phdr)
        ycorrection = doYWalkcorr(events, info, switches, reffiles, phdr)
        applyWalkCorrection(events, xcorrection, ycorrection)

    updateGlobrate(info, headers[1])

    # Copy columns to xdopp, xfull, yfull so we'll have default values.
    if not info["corrtag_input"]:
        copyColumns(events)

    initHelcorr(events, info, headers[1])

    doDeadcorr(events, input, info, switches, reffiles, phdr, headers[1],
               stim_countrate, stim_livetime, cl_args["livetimefile"])

    # Write the calcos sum image.
    if info["obsmode"] == "TIME-TAG":
        bursts = None
        (modified, gti) = recomputeExptime(input, bursts, badt, events,
                                           headers[1], info)
        if info["detector"] == "FUV":       # update keywords EXPTIME[AB]
            key = cosutil.segmentSpecificKeyword("exptime", info["segment"])
            headers[1][key] = info["exptime"]
    if outcsum is not None:
        writeCsum(outcsum, events,
                  info["detector"], info["obsmode"],
                  phdr, headers[1],
                  cl_args["raw_csum_coords"],
                  cl_args["binx"], cl_args["biny"],
                  cl_args["compress_csum"],
                  cl_args["compression_parameters"])

    doPhacorr(inpha, events, info, switches, reffiles, phdr, headers[1])

    doDoppcorr(events, info, switches, reffiles, phdr)

    if not (info["aperture"] in APERTURE_NAMES or
            info["targname"] == "DARK" and
            info["aperture"] in OTHER_APERTURE_NAMES):
        ofd.close()
        raise BadApertureError("APERTURE = %s is not a valid aperture name." %
                               info["aperture"])

    if outcsum is not None and cl_args["only_csum"]:
        return 0                        # don't write flt and counts

    doFlatcorr(events, info, switches, reffiles, phdr, headers[1])

    phdr["wavecals"] = ""               # initial value
    if info["tagflash"]:
        cosutil.printSwitch("WAVECORR", switches)
    if switches["wavecorr"] == "PERFORM":
        if info["tagflash"]:
            (tl_time, shift1_vs_time, wavecorr) = \
            concurrent.processConcurrentWavecal(events, \
                        outflash, cl_args["shift_file"],
                        info, switches, reffiles, phdr, headers[1])
            if wavecorr == "COMPLETE":
                filename = os.path.basename(input)
                if cl_args["shift_file"] is not None:
                    filename = filename + " " + cl_args["shift_file"]
                phdr["wavecals"] = filename
        else:
            # Value to assign to keyword in phdr (updateFromWavecal does
            # this), but this value can be overridden by noWavecal.
            wavecorr = "COMPLETE"
            if not wavecal_info:
                # The exposure is not tagflash and there's no auto/GO
                # wavecal, so create wavecal info with a default shift1,
                # or possibly with a value specified by the user.
                (wavecal_info, wavecorr) = noWavecal(input,
                                cl_args["shift_file"],
                                info, switches, reffiles)
            # LP6 FUV data has no tagflash, so wavecal is done using wavecal
            # exposures before and after each science exposure.  Long exposures
            # (>900s) need a simulated wavecal inserted 600s after the beginning of
            # the preceding waecal to model the non-linear
            # behaviour of shift vs time
            (tl_time, shift1_vs_time) = \
            updateFromWavecal(events, wavecal_info, wavecorr,
                              cl_args["shift_file"],
                              info, switches, reffiles, input_path, phdr, headers[1])
        # Compute wavelengths for the wavelength column (except for wavecals).
        if info["obstype"] == "SPECTROSCOPIC" and \
           info["exptype"].find("WAVE") == -1:
            computeWavelengths(events, info, reffiles,
                               helcorr=switches["helcorr"], hdr=None)
    else:
        time = cosutil.getColCopy(data=events, column="time")
        tl_time = cosutil.timelineTimes(time[0], time[-1], dt=1.)
        shift1_vs_time = None
        del time

    if info["obsmode"] == "TIME-TAG":
        bursts = doBurstcorr(events, info, switches, reffiles, phdr,
                             cl_args["burstfile"])
        (modified, gti) = recomputeExptime(input, bursts, badt, events,
                                           headers[1], info)
        if modified:
            saveNewGTI(ofd, gti)
        countBadEvents(events, bursts, badt, info, headers[1])

    if info["detector"] == "FUV":       # update keyword EXPTIMEA or EXPTIMEB
        key = cosutil.segmentSpecificKeyword("exptime", info["segment"])
        headers[1][key] = info["exptime"]
    minmax_shift_dict = getWavecalOffsets(events, info, switches["wavecorr"],
                                          reffiles["xtractab"],
                                          reffiles["brftab"])
    tracemask = createTraceMask(events, info, switches,
                                reffiles['xtractab'], active_area)

    traceprofile = doTraceCorr(events, info, switches, reffiles, phdr,
                               tracemask)

    #
    # Make sure we have a gti variable, and make one if we don't.  None is OK, it will be
    # detected and filled in later if necessary
    try:
        temp_gti = gti
    except NameError:
        gti = None

    align = doProfileAlignmentCorr(events, input, info, switches, reffiles,
                                   phdr, headers[1], minmax_shift_dict,
                                   tracemask, traceprofile, gti)

    dq_array = doDqicorr(events, input, info, switches, reffiles,
                         phdr, headers[1], minmax_shift_dict,
                         traceprofile, gti)


    writeImages(events.field(xfull), events.field(yfull),
                events.field("epsilon"), events.field("dq"),
                phdr, headers,
                dq_array, info["npix"], info["x_offset"], info["exptime"],
                outcounts, output)

    doStatflag(switches, output, outcounts)

    # Create or update a TIMELINE extension.
    timeline.createTimeline(input, ofd, info, reffiles,
                            tl_time, shift1_vs_time,
                            events.field("TIME").astype(np.float64),
                            events.field(xfull), events.field(yfull))

    ofd.close()

    return 0            # 0 is OK


def setCorrColNames(detector):
    """Assign column names to global variables.

    Parameters
    ----------
    detector: {"FUV", "NUV"}
        Detector name.
    """

    global xcorr, ycorr, xdopp, ydopp, xfull, yfull

    xcorr = "XCORR"
    ycorr = "YCORR"
    xdopp = "XDOPP"
    ydopp = "YCORR"

    xfull = "XFULL"
    yfull = "YFULL"

def setActiveArea(events, info, brftab):
    """Assign a value to active_area.

    This function updates the global variable active_area, which is a
    Boolean array with the same number of elements as there are rows in
    the events table.  An element will be True if the corresponding
    event (row in the table) is within the FUV active area.  For NUV
    all elements will be set to True.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table

    info: dictionary
        Header keywords and values

    brftab: str
        Name of the baseline reference table
    """

    global active_area

    xi  = events.field(xcorr)
    eta = events.field(ycorr)
    active_area = np.ones(len(xi), dtype=np.bool8)

    # A value of 1 (True) in active_area means the corresponding event
    # is within the active area.
    if info["detector"] == "FUV":
        (b_low, b_high, b_left, b_right) = \
                cosutil.activeArea(info["segment"], brftab)
        active_area = np.where(xi > b_right, False, active_area)
        active_area = np.where(xi < b_left,  False, active_area)
        active_area = np.where(eta > b_high, False, active_area)
        active_area = np.where(eta < b_low,  False, active_area)
        # Make sure the data type is still boolean.
        active_area = active_area.astype(np.bool8)

def mkHeaders(phdr, events_header, extver=1):
    """Create a list of four headers for creating the flt and counts files.

    The following keywords will be assigned or copied from events_header
    to the ERR extension header:
        EXTNAME
        EXTVER
        ROOTNAME
        EXPNAME
        RA_APER
        DEC_APER
        PA_APER
        DISPAXIS
        NGOODPIX
        GOODMEAN
        GOODMAX
    The following keywords will be assigned or copied from events_header
    to the DQ extension header:
        EXTNAME
        EXTVER
        ROOTNAME
        EXPNAME

    Parameters
    ----------
    phdr: astropy.io.fits Header object
        primary header from input file

    events_header: astropy.io.fits Header object
        EVENTS extension header from input file

    Returns
    -------
    list
        Primary, SCI, ERR and DQ headers
    """

    headers = [phdr]
    # This is a reference, not a copy.  Keywords will be updated (in other
    # functions) in headers[1], and the output corrtag header as well as the
    # flt and counts headers will contain the updated values.
    headers.append(events_header)

    err_hdr = fits.Header()
    dq_hdr = fits.Header()
    err_hdr["extname"] = ("ERR", "extension name")
    dq_hdr["extname"] = ("DQ", "extension name")
    err_hdr["extver"] = (extver, "extension version number")
    dq_hdr["extver"] = (extver, "extension version number")
    if "rootname" in events_header:
        rootname = events_header["rootname"]
        err_hdr["rootname"] = (rootname, "rootname of the observation set")
        dq_hdr["rootname"] = (rootname, "rootname of the observation set")
    if "expname" in events_header:
        expname = events_header["expname"]
        err_hdr["expname"] = (expname, "exposure identifier")
        dq_hdr["expname"] = (expname, "exposure identifier")
    if "ra_aper" in events_header:
        err_hdr["ra_aper"] = (events_header["ra_aper"],
                              "RA of reference aperture center")
    if "dec_aper" in events_header:
        err_hdr["dec_aper"] = (events_header["dec_aper"],
                               "Declination of reference aperture center")
    if "pa_aper" in events_header:
        err_hdr["pa_aper"] = (events_header["pa_aper"],
                        "Position Angle of reference aperture center (de")
    if "dispaxis" in events_header:
        err_hdr["dispaxis"] = (events_header["dispaxis"],
                        "dispersion axis; 1 = axis 1, 2 = axis 2, none")
    if "ngoodpix" in events_header:
        err_hdr["ngoodpix"] = (-999, "number of good pixels")
    if "goodmean" in events_header:
        err_hdr["goodmean"] = (-999., "mean value of good pixels")
    if "goodmax" in events_header:
        err_hdr["goodmax"] = (-999., "maximum value of good pixels")

    headers.append(err_hdr)
    headers.append(dq_hdr)

    return headers

def updateHVKeywords(hdr, info, reffiles):
    """Find the commanded high voltage, and update HV keywords.

    Parameters
    ----------
    hdr: astropy.io.fits Header object
        Header for EVENTS extension.  Keywords "hvlevela" and "hvlevelb"
        will be updated with the commanded high voltage (raw) for segments
        FUVA and FUVB respectively.

    info: dictionary
        Header keywords and values.  Key "hvlevel" will be assigned the
        commanded high voltage (raw) for the current segment.

    reffiles: dictionary
        Contains the name of the table containing high voltage values.
    """

    hvtab = reffiles["hvtab"]

    if hvtab == NOT_APPLICABLE:
        return

    cosutil.printRef("HVTAB", reffiles)

    segment_list = ["FUVA", "FUVB"]     # update keywords for both segments

    fd = fits.open(hvtab, mode="readonly")

    kwd_root = "hvlevel"        # high voltage (commanded, raw)
    expstart = info["expstart"]

    for segment in segment_list:

        hdu = fd[(segment,1)]
        keyword = cosutil.segmentSpecificKeyword(kwd_root, segment)
        start = hdu.data.field("date")
        # The column name for raw HV counts is the same as the keyword name.
        raw = hdu.data.field(keyword)
        # Find the row with closest time before the exposure's expstart.
        t_diff = expstart - start[0]    # initial values
        row_min = 0
        for row in range(len(start)):
            diff = expstart - start[row]
            if diff >= 0. and diff < t_diff:
                t_diff = diff
                row_min = row
        hv_raw = raw[row_min]
        hdr[keyword] = hv_raw
        if segment == info["segment"]:
            info["hvlevel"] = hv_raw

    fd.close()

def doPhotcorr(info, switches, imphttab, phdr, hdr):
    """Update photometry parameter keywords for imaging data.

    Parameters
    ----------
    info: dictionary
        Header keywords and values

    switches: dictionary
        Calibration switches

    imphttab: str
        The name of the imaging photometric parameters table

    phdr: astropy.io.fits Header Object
        The primary header, photcorr keyword updated in-place

    hdr: astropy.io.fits Header Object
        The first extension header, updated in-place
    """

    if info["obstype"] == "IMAGING" and info["detector"] == "NUV":
        cosutil.printSwitch("PHOTCORR", switches)
        if switches["photcorr"] == "PERFORM":
            obsmode = "cos,nuv," + info["opt_elem"] + "," + info["aperture"]
            phot.doPhot(imphttab, obsmode, hdr)
            phdr["photcorr"] = "COMPLETE"

def updateGlobrate(info, hdr):
    """Update the GLOBRATE keyword in the extension header.

    Parameters
    ----------
    info: dictionary
        Header keywords and values

    hdr: astropy.io.fits Header object
        The input events extension header
    """

    globrate = globrate_tt(info["orig_exptime"], info["detector"])
    if info["detector"] == "FUV":
        keyword = "globrt_" + info["segment"][-1]
    else:
        keyword = "globrate"
    globrate = round(globrate, 4)
    hdr[keyword] = globrate

def globrate_tt(exptime, detector):
    """Return the global count rate for time-tag data.

    Parameters
    ----------
    exptime: float
        The exposure time; this is the original value from the header,
        i.e. not corrected for bursts or bad time intervals

    detector: {"FUV", "NUV"}
        Detector name.

    Returns
    -------
    float
        The global count rate, counts per second
    """

    global active_area

    if exptime <= 0.:
        return 0.

    if detector == "NUV":
        return float(len(active_area)) / exptime

    return np.sum(active_area.astype(np.float32)) / exptime

def doBurstcorr(events, info, switches, reffiles, phdr, burstfile):
    """Find bursts, and flag them in the data quality column.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.

    burstfile: str, or None
        Name of output text file for burst info.

    Returns
    -------
    bursts: list of two-element lists, or None
        List of [bad_start, bad_stop] intervals during which a burst was
        detected (seconds since expstart)
    """

    bursts = None
    if info["segment"][:3] == "FUV":
        # Find and flag regions where the count rate is unreasonably high.
        cosutil.printSwitch("BRSTCORR", switches)
        if switches["brstcorr"] == "PERFORM":
            cosutil.printRef("brsttab", reffiles)
            cosutil.printRef("xtractab", reffiles)
            bursts = burst.burstFilter(events.field("time"),
                                       events.field(yfull), events.field("dq"),
                                       reffiles, info, burstfile)
            phdr["brstcorr"] = "COMPLETE"

    return bursts

def doBadtcorr(events, info, switches, reffiles, phdr):
    """Flag bad time intervals in the data quality column.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table

    info: dictionary
        Header keywords and values

    switches: dictionary
        Calibration switches

    reffiles: dictionary
        Reference file names

    phdr: astropy.io.fits Header object
        The input primary header

    Returns
    -------
    badt: list of two-element lists
        List of [bad_start, bad_stop] intervals from the badttab (converted
        to seconds since expstart)
    """

    badt = []

    cosutil.printSwitch("BADTCORR", switches)
    if switches["badtcorr"] == "PERFORM":
        cosutil.printRef("BADTTAB", reffiles)
        badt = filterByTime(events.field("time"), events.field("dq"),
                            reffiles["badttab"],
                            info["expstart"], info["segment"])
        phdr["badtcorr"] = "COMPLETE"

    return badt

def filterByTime(time, dq, badttab, expstart, segment):
    """Flag bad time intervals in dq.

    For each bad time interval in the badttab, a flag will be set in the
    data quality column for each event within that time interval.

    Parameters
    ----------
    time: array_like
        The time column in the events table

    dq: array_like
        The data quality column in the events table (updated in-place)

    badttab: str
        The name of the bad-time-intervals table

    expstart: float
        The exposure start time (MJD)

    segment: str
        Segment or stripe name

    Returns
    -------
    badt: list of two-element lists
        List of [bad_start, bad_stop] intervals from the badttab (converted
        to seconds since expstart)
    """

    # Flag regions listed in the badt table.
    badt_info = cosutil.getTable(badttab, filter={"segment": segment})

    badt = []
    if badt_info is not None:
        nrows = badt_info.shape[0]

        start = badt_info.field("start")
        stop  = badt_info.field("stop")

        # Convert from MJD to seconds after expstart.
        for i in range(nrows):
            start[i] = (start[i] - expstart) * SEC_PER_DAY
            stop[i] = (stop[i] - expstart) * SEC_PER_DAY
            badt.append([start[i], stop[i]])

        # For each time interval in the badttab, flag every event for which
        # the time falls within that interval.
        for i in range(nrows):
            dq |= np.where(np.logical_and(time >= start[i], time <= stop[i]),
                           DQ_BAD_TIME, 0)

    return badt

def countBadEvents(events, bursts, badt, info, hdr):
    """Update keywords for events and time lost.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    bursts: list of two-element lists
        List of [bad_start, bad_stop] intervals during which a burst was
        detected.

    badt: list of two-element lists
        List of [bad_start, bad_stop] intervals from the badttab
        (converted to seconds since expstart).

    info: dictionary
        Keywords and values

    hdr: astropy.io.fits Header object
        The events extension header (keywords will be updated).
    """

    t = events.field("time").astype(np.float64)
    expstart = t[0]             # seconds since exposure start
    expend = t[-1]

    t_burst = 0.
    n_burst = 0
    t_badt = 0.
    n_badt = 0
    n_outside_active_area = 0
    n_bad_pha = 0

    if info["detector"] == "FUV":
        if bursts is not None:
            for burst in bursts:
                t_burst += (burst[1] - burst[0])
                r = ccos.range(t, burst[0], burst[1])
                n_burst += (r[1] - r[0])
        t_key = "tbrst_" + info["segment"][-1]
        n_key = "nbrst_" + info["segment"][-1]
        hdr[t_key] = t_burst
        hdr[n_key] = n_burst

        # The length of t is the total number of events, while the number of
        # True flags is the number of events that are within the active area.
        n_outside_active_area = len(t) - np.sum(active_area.astype(np.int32))
        n_key = "nout_" + info["segment"][-1]
        hdr[n_key] = n_outside_active_area

    for (bad_start, bad_stop) in badt:
        if badt is not None:
            # badt includes all time intervals in the badttab, and many of
            # those intervals may lie outside the time range of the exposure.
            if bad_stop <= expstart:
                continue
            if bad_start >= expend:
                continue
            bad_start = max(bad_start, expstart)
            bad_stop = min(bad_stop, expend)
            t_badt += (bad_stop - bad_start)
            r = ccos.range(t, bad_start, bad_stop)
            n_badt += (r[1] - r[0])
    if info["detector"] == "FUV":
        t_key = "tbadt_" + info["segment"][-1]
        n_key = "nbadt_" + info["segment"][-1]
    else:
        t_key = "tbadt"
        n_key = "nbadt"
    hdr[t_key] = t_badt
    hdr[n_key] = n_badt

    if info["detector"] == "FUV":
        # The keyword for the number of events flagged as bad due to pulse
        # height out of bounds has already been set, so just get the value.
        n_pha_key = "npha_" + info["segment"][-1]
        n_bad_pha = hdr.get(n_pha_key, 0)

    if info["detector"] == "FUV":
        n_key = "nbadevt" + info["segment"][-1]
    else:
        n_key = "nbadevnt"
    hdr[n_key] = n_burst + n_badt + n_outside_active_area + n_bad_pha

def recomputeExptime(input, bursts, badt, events, hdr, info):
    """Recompute the exposure time and update the keyword.

    Parameters
    ----------
    input: str
        Name of the input file (for getting GTI table).

    bursts: list of two-element lists
        List of [bad_start, bad_stop] intervals during which a burst was
        detected.

    badt: list of two-element lists
        List of [bad_start, bad_stop] intervals from the badttab
        (converted to seconds since expstart).

    events: astropy.io.fits record array
        The data unit containing the events table.

    hdr: astropy.io.fits Header object
        The events extension header (keywords will be updated).

    info: dictionary
        Keywords and values (exptime can be updated).

    Returns
    -------
    tuple containing a flag and a list of two-element lists
        modified is a flag indicating whether there was actually any
        change to the list of [start, stop] intervals.
        gti is an updated list of [start, stop] good time intervals
        (seconds since expstart), updated from the GTI table in the raw
        file by excluding bursts and intervals flagged as bad by the
        badttab.
    """

    time = events.field("time")

    modified_0 = False
    gti = cosutil.returnGTI(input)
    if len(gti) <= 0:
        cosutil.printWarning("No GTI table found in raw file.", VERBOSE)
        gti = [[time[0], time[-1]]]
        modified_0 = True

    (modified_1, gti) = recomputeGTI(gti, bursts)
    (modified_2, gti) = recomputeGTI(gti, badt)
    modified = modified_0 or modified_1 or modified_2

    exptime = 0.
    for (start, stop) in gti:
        exptime += (stop - start)

    old_exptime = hdr.get("exptime", 0.)
    if exptime != old_exptime:
        hdr["exptime"] = exptime
        info["exptime"] = exptime
        if abs(exptime - old_exptime) > 1.:
            cosutil.printWarning("exposure time in header was %.3f" %
                                 old_exptime, VERBOSE)
            cosutil.printContinuation("exptime has been corrected to %.3f" %
                                      exptime, VERBOSE)

    return (modified, gti)

def recomputeGTI(gti, badt):
    """Recompute the list of good [start, stop] intervals.

    Parameters
    ----------
    gti: list of two-element lists
        List of [start, stop] good time intervals (times are in seconds
        since EXPSTART).

    badt: list of two-element lists
        List of [bad_start, bad_stop] intervals, e.g. during which there
        was a burst or a bad time interval from the BADTTAB (seconds since
        EXPSTART).

    Returns
    -------
    tuple containing a flag and a list of two-element lists
        modified is a flag indicating whether there was actually any
        change to the list of [start, stop] intervals.
        gti is an updated list of [start, stop] good time intervals
        (seconds since expstart), updated from the GTI table in the raw
        file by excluding bursts and intervals flagged as bad by the
        badttab.
    """

    modified = False                    # initial value
    if not badt:
        return (modified, gti)

    for (bad_start, bad_stop) in badt:
        new_gti = []
        for (start, stop) in gti:
            if bad_start >= stop or bad_stop <= start:
                new_gti.append([start, stop])
            else:
                if bad_start > start:
                    new_gti.append([start, bad_start])
                    modified = True
                if bad_stop < stop:
                    new_gti.append([bad_stop, stop])
                    modified = True
        gti = new_gti

    return (modified, gti)

def saveNewGTI(ofd, gti):
    """Append new GTI information as a BINTABLE extension.

    Parameters
    ----------
    ofd: astropy.io.fits HDUList object
        Output file header/data list.

    gti: list of two-element lists
        An updated list of [start, stop] good time intervals.
    """

    len_gti = len(gti)
    col = []
    col.append(fits.Column(name="START", format="1D", unit="s"))
    col.append(fits.Column(name="STOP", format="1D", unit="s"))
    cd = fits.ColDefs(col)
    hdu = fits.BinTableHDU.from_columns(cd, nrows=len_gti)
    hdu.header["extname"] = "GTI"
    outdata = hdu.data
    startcol = outdata.field("START")
    stopcol = outdata.field("STOP")
    for i in range(len_gti):
        startcol[i] = gti[i][0]
        stopcol[i] = gti[i][1]

    # Set extver for the new GTI table to be larger than extver for any
    # existing GTI table.  We expect only one, the original one, but there
    # could be others.
    last_extver = 0                     # initial value
    for i in range(1, len(ofd)):
        existing_gti = ofd[i]
        extname = existing_gti.header.get("extname", "MISSING")
        extname = extname.upper()
        if extname == "GTI":
            extver = existing_gti.header.get("extver", 1)
            last_extver = max(last_extver, extver)
    hdu.header["extver"] = last_extver + 1

    # Now append the updated GTI table.
    ofd.append(hdu)
    # if we have pyfits 2.1.1dev462 or later, we could insert
    # ofd.insert(2, hdu)

    ofd[0].header["nextend"] = len(ofd) - 1

def doPhacorr(inpha, events, info, switches, reffiles, phdr, hdr):
    """Filter by pulse height.

    For TIME-TAG data, the lower and upper thresholds will be taken from
    the PHAFILE, if that keyword is present and not set to N/A; otherwise,
    the limits will be taken from the PHATAB.

    Parameters
    ----------
    inpha: str
        Name of the input file containing the pulse height histogram.

    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.

    hdr: astropy.io.fits Header object
        The input events extension header.
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch("PHACORR", switches)
        if switches["phacorr"] == "PERFORM":
            if info["obsmode"] == "TIME-TAG":
                if reffiles["phafile"] != NOT_APPLICABLE:
                    cosutil.printRef("PHAFILE", reffiles)
                    filterPHA(events.field(xcorr), events.field(ycorr),
                              events.field("pha"), events.field("dq"),
                              reffiles["phafile"], info, hdr)
                else:
                    cosutil.printRef("PHATAB", reffiles)
                    filterByPulseHeight(events.field("pha"),
                                        events.field("dq"),
                                        reffiles["phatab"], info, hdr)
            else:
                checkPulseHeight(inpha, reffiles["phatab"], info, hdr)
            phdr["phacorr"] = "COMPLETE"

def filterPHA(xcorr, ycorr, pha, dq, phafile, info, hdr):
    """Flag events that have a pulse height outside an allowed range.

    This is only called for TIME-TAG mode data.
    This version uses a pair of images with the cutoff limits.  Since the
    limits can vary with position on the detector, the keywords for the
    screening limits will be set to -999 to indicate that the limits are
    not constant.

    Parameters
    ----------
    xcorr: array_like
        The column of corrected X pixel coordinates from the events table.

    ycorr: array_like
        The column of corrected Y pixel coordinates from the events table.

    pha: array_like
        The column of pulse-height amplitudes from the events table.

    dq: array_like
        The data quality column in the events table (updated in-place).

    phafile: str
        The name of the file containing images of lower and upper cutoff
        limits for the pulse height.

    info: dictionary
        Header keywords and values.

    hdr: astropy.io.fits Header object
        The EVENTS extension header; keywords for number of rejected
        events will be assigned.
    """

    segment = info["segment"]

    # get im_low, im_high from phafile
    fd = fits.open(phafile, mode="copyonwrite")
    hdu_low = fd[(segment,1)]           # hdu with data for lower limits
    hdu_high = fd[(segment,2)]          # hdu with data for upper limits
    im_low = hdu_low.data
    im_high = hdu_high.data
    fd.close()

    counters = ccos.pha_check(xcorr, ycorr, pha.astype(np.int16), dq,
                              im_low, im_high, DQ_PHA_OUT_OF_BOUNDS)
    if counters is None:
        raise RuntimeError("PHACORR:  images in PHAFILE %s are "
                           "not the same shape" % phafile)

    (nbad_low, nbad_high) = counters

    if cosutil.checkVerbosity(VERY_VERBOSE):
        cosutil.printMsg("Filter by pulse height using PHAFILE:",
                         VERY_VERBOSE)
        if nbad_low == 0:
            msg = "  no event was"
        elif nbad_low == 1:
            msg = "  one event was"
        else:
            msg = "  %d events were" % nbad_low
        msg += " rejected because PHA was less than the cutoff"
        cosutil.printMsg(msg, VERY_VERBOSE)
        if nbad_high == 0:
            msg = "  no event was"
        elif nbad_high == 1:
            msg = "  one event was"
        else:
            msg = "  %d events were" % nbad_high
        msg += " rejected because PHA was greater than the cutoff"
        cosutil.printMsg(msg, VERY_VERBOSE)

    keyword = "NPHA_" + segment[-1]
    hdr[keyword] = nbad_low + nbad_high

    # xxx what should we do for keywords for lower and upper limits?
    # The screening limits are not necessarily constant, so the keywords
    # for these limits are no longer useful.
    (low, high) = (-999, -999)
    cosutil.updatePulseHeightKeywords(hdr, segment, low, high)

def filterByPulseHeight(pha, dq, phatab, info, hdr):
    """Flag events that have a pulse height outside an allowed range.

    This is only called for TIME-TAG mode data.
    This version uses a table with the cutoff limits.

    Parameters
    ----------
    pha: array_like
        The column of pulse-height amplitudes from the events table.

    dq: array_like
        The data quality column in the events table (updated in-place).

    phatab: str
        The name of the table containing lower and upper cutoff limits for
        the pulse height.

    info: dictionary
        Header keywords and values.

    hdr: astropy.io.fits Header object
        The EVENTS extension header; keywords for screening limits and
        number of rejected events will be assigned.
    """

    global active_area

    segment = info["segment"]
    filter = {"segment": segment}
    if cosutil.findColumn(phatab, "opt_elem"):
        filter["opt_elem"] = info["opt_elem"]

    pha_info = cosutil.getTable(phatab, filter, exactly_one=True)

    low = pha_info.field("llt")[0]
    high = pha_info.field("ult")[0]

    # Flag an event if the pulse height is below the minimum value or
    # above the maximum value that is likely to be encountered from a
    # real photon event.
    # Restrict this test to the active area.
    test_low = np.logical_and(active_area, pha < low)
    test_high = np.logical_and(active_area, pha > high)

    dq |= np.where(test_low, DQ_PHA_OUT_OF_BOUNDS, 0)
    rejected = np.nonzero(dq & DQ_PHA_OUT_OF_BOUNDS)[0]
    nbad_low = len(rejected)            # number of rejected events, PHA low

    dq |= np.where(test_high, DQ_PHA_OUT_OF_BOUNDS, 0)
    rejected = np.nonzero(dq & DQ_PHA_OUT_OF_BOUNDS)[0]
    nbad = len(rejected)                # total number of rejected events
    nbad_high = nbad - nbad_low         # number of rejected events, PHA high

    if cosutil.checkVerbosity(VERY_VERBOSE):
        cosutil.printMsg("Filter by pulse height using PHATAB:",
                         VERY_VERBOSE)
        if nbad_low == 0:
            msg = "  no event was"
        elif nbad_low == 1:
            msg = "  one event was"
        else:
            msg = "  %d events were" % nbad_low
        msg += " rejected because PHA was less than %d" % low
        cosutil.printMsg(msg, VERY_VERBOSE)
        if nbad_high == 0:
            msg = "  no event was"
        elif nbad_high == 1:
            msg = "  one event was"
        else:
            msg = "  %d events were" % nbad_high
        msg += " rejected because PHA was greater than %d" % high
        cosutil.printMsg(msg, VERY_VERBOSE)

    keyword = "NPHA_" + segment[-1]
    hdr[keyword] = nbad

    # Update the values for the screening limit keywords
    # (low and high are the default values).
    cosutil.updatePulseHeightKeywords(hdr, segment, low, high)

def checkPulseHeight(inpha, phatab, info, hdr):
    """Check that the pulse-height distribution is reasonable.

    This is only called for ACCUM mode data.

    Parameters
    ----------
    inpha: str
        Name of file containing pulse-height distribution.

    phatab: str
        Name of table of pulse-height parameters.

    info: dictionary
        Header keywords and values.

    hdr: astropy.io.fits Header object
        Header for events table extension (keywords for screening limits
        and number of rejected events will be assigned).
    """

    filter = {"segment": info["segment"]}
    if cosutil.findColumn(phatab, "opt_elem"):
        filter["opt_elem"] = info["opt_elem"]

    pha_info = cosutil.getTable(phatab, filter, exactly_one=True)

    low = pha_info.field("llt")[0]
    high = pha_info.field("ult")[0]

    # Update the values for the screening limit keywords
    cosutil.updatePulseHeightKeywords(hdr, info["segment"], low, high)

    # The peak in the pulse-height distribution should be within low and high.
    # Apply a factor to low and high to account for the fact that the
    # histogram is from seven-bit values but the values in the table are
    # for five-bit values (the PHA column in an EVENTS table).
    # The mean should be within the factors min_peak and max_peak of the peak.
    low *= TWO_BITS
    high *= TWO_BITS
    min_peak = pha_info.field("min_peak")[0]
    max_peak = pha_info.field("max_peak")[0]

    # Read the pulse-height histogram.
    fd = fits.open(inpha, mode="readonly", memmap=False)
    pha_data = fd[1].data

    npts = len(pha_data)

    sum = np.sum(np.arange(npts, dtype=np.float32) *
                 pha_data.astype(np.float32))
    sumwgt = np.sum(pha_data.astype(np.float32))
    pha_index = np.argsort(pha_data)
    peak = pha_index[-1]

    if sumwgt == 0.:
        cosutil.printWarning("Histogram is empty.")
        fd.close()
        return

    meanval = sum / sumwgt

    warn = (cosutil.checkVerbosity(VERY_VERBOSE))       # initial value
    if peak <= low:
        cosutil.printWarning("Peak in pulse-height distribution is too low.")
        warn = 1
    if peak >= high:
        cosutil.printWarning("Peak in pulse-height distribution is too high.")
        warn = 1

    if meanval < peak * min_peak:
        cosutil.printWarning("Mean of pulse-height distribution is too low.")
        warn = 1
    if meanval > peak * max_peak:
        cosutil.printWarning("Mean of pulse-height distribution is too high.")
        warn = 1

    if warn:
        cosutil.printMsg(
                "Pulse-height distribution peak = %d, mean = %.6g;" %
                (peak, meanval))
        cosutil.printMsg(
                "the peak should be between %.6g and %.6g," % (low, high))
        cosutil.printMsg(
                "and the mean should be between %.6g and %.6g." %
                (peak * min_peak, peak * max_peak))

    fd.close()

def doRandcorr(events, info, switches, reffiles, phdr):
    """Add pseudo-random numbers to x and y coordinates within the active area.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        Primary header.
    """

    global active_area

    if info["detector"] == "FUV":
        cosutil.printSwitch("RANDCORR", switches)
        if switches["randcorr"] == "PERFORM":
            xi  = events.field(xcorr)
            eta = events.field(ycorr)
            nelem = len(xi)
            if info["randseed"] == -1:
                seed = int(time.time())
                phdr["randseed"] = seed
                msg = "RANDSEED = %d (was -1)" % seed
            else:
                seed = info["randseed"]
                msg = "RANDSEED = %d" % seed
            cosutil.printMsg(msg)
            random.seed(seed)
            rn = random.uniform(-0.5, +0.5, nelem)
            xi[:] = np.where(active_area, xi - rn, xi)
            rn = random.uniform(-0.5, +0.5, nelem)
            eta[:] = np.where(active_area, eta - rn, eta)
            phdr["randcorr"] = "COMPLETE"

def initTempcorr(events, input, info, switches, reffiles, hdr, stimfile):
    """Compute parameters for thermal distortion.

    Parameters
    ----------
    events: astropy.io.fits record array
        the data unit containing the events table

    input: str
        name of raw file (for writing to stimfile)

    info: dictionary
        header keywords and values

    switches: dictionary
        calibration switches

    reffiles: dictionary
        reference file names

    hdr: astropy.io.fits Header object
        the input events extension header

    stimfile: str
        name of output text file for stim positions (or None)

    Returns
    -------
    tuple, (stim_param, stim_countrate, stim_livetime)
        stim_param is a dictionary of lists, with keys
        i0, i1, x0, xslope, y0, yslope;
        stim_countrate and stim_livetime are the count rate of the
        stims and the livetime factor based on that count rate
    """

    if info["detector"] == "FUV" and \
       (switches["tempcorr"] == "PERFORM" or switches["deadcorr"] == "PERFORM"):
        # Compute the parameters (to be used later).
        time = events.field("TIME").astype(np.float64)
        (stim_param, avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref,
         stim_countrate, stim_livetime) = \
         computeThermalParam(time,
                             events.field(xcorr), events.field(ycorr),
                             events.field("dq"),
                             reffiles["brftab"], info["obsmode"],
                             info["segment"], info["orig_exptime"],
                             info["stimrate"], input, stimfile)
        if switches["tempcorr"] == "PERFORM":
            # Update stim location keywords in extension header.
            stimKeywords(hdr, info["segment"], avg_s1, avg_s2,
                         rms_s1, rms_s2, s1_ref, s2_ref)
    else:
        stim_countrate = 0.
        stim_livetime = 1.
        stim_param = {}

    return (stim_param, stim_countrate, stim_livetime)

def computeThermalParam(time, x, y, dq,
                        brftab, obsmode,
                        segment, exptime, stimrate, input, stimfile):
    """Compute thermal distortion parameters from stim positions.

    This function loops over intervals of time, and within each interval
    calls routines to find the stim locations and compute the thermal
    distortion parameters.

    If a stimfile was specified, it will be opened (append mode), and the
    stim positions for each time interval will be written to the file.
    (The 'input' argument is included in the calling sequence only for the
    purpose of writing its name to the stimfile.)

    Parameters
    ----------
    time: array like
        Array of event times.

    x: array like
        Detector X coordinates.

    y: array like
        Detector Y coordinates.

    dq: array like
        Array of data quality flags   (NOTE:  not currently used).

    brftab: str
        Name of baseline reference data table.

    obsmode: str
        TIME-TAG or ACCUM.

    segment: str
        Segment name (for FUV).

    exptime: float
        Exposure time (for computing livetime); this is the original value,
        i.e. not corrected for bursts or bad time intervals.

    stimrate: float
        Input count rate for a stim (for computing livetime).

    input: str
        Name of raw file (for writing to stimfile).

    stimfile: str
        Name of text file to which stim locations will be appended.

    Returns
    -------
    tuple, (stim_param, avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref,
        stim_countrate, stim_livetime)

    stim_param is a dictionary of lists:  (i0, i1, x0, xslope, y0, yslope)

    avg_s1[0] is the average Y location of the first stim.
    avg_s1[1] is the average X location of the first stim.
    avg_s2[0] is the average Y location of the second stim.
    avg_s2[1] is the average X location of the second stim.

    rms_s1[0] is the RMS in Y for the first stim.
    rms_s1[1] is the RMS in X for the first stim.
    rms_s2[0] is the RMS in Y for the second stim.
    rms_s2[1] is the RMS in X for the second stim.

    stim_countrate is the observed count rate for a stim, or None if
      neither stim could be found
    stim_livetime is the live time computed from the input and observed
      stim rate

    For each i:
      i0[i], i1[i] is the slice of indices in 'events' corresponding to
        the ith time interval.  Each such interval is of length dt_thermal
        in duration (except possibly the last, which could be shorter).
      x0[i] and xslope[i] are the intercept and slope respectively of the
        linear correction to the X positions (the more rapidly varying
        direction).
      y0[i] and yslope[i] are the intercept and slope for the linear
        correction to the Y positions.
    """

    if stimfile is None:
        fd = None
    else:
        fd = open(stimfile, "a")
        fd.write("# %s\n" % input)

    nevents = len(time)

    brf_info = cosutil.getTable(brftab, filter={"segment": segment},
                                exactly_one=True)

    # Find stims and compute parameters every dt_thermal seconds.
    if obsmode == "TIME-TAG":
        fd_brf = fits.open(brftab, mode="readonly", memmap=False)
        dt_thermal = fd_brf[1].header["timestep"]
        fd_brf.close()
        cosutil.printMsg(
"Compute thermal corrections from stim positions; timestep is %.6g s:"
            % dt_thermal, VERY_VERBOSE)
    else:
        # For ACCUM data we want just one time interval.
        dt_thermal = time[-1] - time[0] + 1.

    sx1 = brf_info.field("sx1")[0]
    sy1 = brf_info.field("sy1")[0]
    sx2 = brf_info.field("sx2")[0]
    sy2 = brf_info.field("sy2")[0]
    xwidth = brf_info.field("xwidth")[0]
    ywidth = brf_info.field("ywidth")[0]

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
        fd.write("# t0 t1 stim_locations\n")

    t0 = time[0]
    t1 = t0 + dt_thermal
    sumstim = (0, 0., 0., 0., 0., 0, 0., 0., 0., 0.)
    last_s1 = s1_ref            # initial default values
    last_s2 = s2_ref
    while t0 <= time[nevents-1]:

        # time[i:j] matches t0 to t1.
        try:
            (i, j) = ccos.range(time, t0, t1)
        except:
            t0 = t1
            t1 = t0 + dt_thermal
            continue
        if i >= j:              # i and j can be equal due to roundoff
            t0 = t1
            t1 = t0 + dt_thermal
            continue

        (s1, sumsq1, counts1, found_s1) = \
                findStim(x[i:j], y[i:j], s1_ref, xwidth, ywidth)

        (s2, sumsq2, counts2, found_s2) = \
                findStim(x[i:j], y[i:j], s2_ref, xwidth, ywidth)

        # Increment sums for averaging the stim positions.
        sumstim = updateStimSum(sumstim, counts1, s1, sumsq1, found_s1,
                                counts2, s2, sumsq2, found_s2)

        if fd is not None:
            fd.write("%.0f %.0f" % (t0, min(time[nevents-1], t1)))
            if found_s1:
                fd.write("  %.1f %.1f" % (s1[1], s1[0]))
            else:
                fd.write("  INDEF INDEF")
            if found_s2:
                fd.write("  %.1f %.1f\n" % (s2[1], s2[0]))
            else:
                fd.write("  INDEF INDEF\n")
        if found_s1:
            last_s1 = s1        # save current value
        else:
            s1 = last_s1        # use last stim position that was found
        if found_s2:
            last_s2 = s2
        else:
            s2 = last_s2
        if cosutil.checkVerbosity(VERY_VERBOSE) or \
           not (found_s1 and found_s2):
            msg = "  %7d ... %7d" % (i, j-1)
            msg += "  %.1f %.1f" % (s1[1], s1[0])
            if not found_s1:
                msg += " (stim1 not found)"
            msg += "  %.1f %.1f" % (s2[1], s2[0])
            if not found_s2:
                msg += " (stim2 not found)"
            if not (found_s1 and found_s2):
                cosutil.printWarning(msg)
                if time[j-1] - time[i] < dt_thermal and obsmode == "TIME-TAG":
                    cosutil.printContinuation(
                "Note that the time interval is %g s" % (time[j-1] - time[i]))
            else:
                cosutil.printMsg(msg)

        (x0_n, xslope_n, y0_n, yslope_n) = thermalParam(s1, s2, s1_ref, s2_ref)
        i0.append(i)
        i1.append(j)
        x0.append(x0_n)
        xslope.append(xslope_n)
        y0.append(y0_n)
        yslope.append(yslope_n)
        t0 = t1
        t1 = t0 + dt_thermal

    # Compute the average of the stim positions.
    avg_s1 = [-1., -1.]
    avg_s2 = [-1., -1.]
    rms_s1 = [-1., -1.]
    rms_s2 = [-1., -1.]
    total_counts1 = sumstim[0]
    total_counts2 = sumstim[5]
    if total_counts1 > 0:
        avg_s1[0] = sumstim[1] / sumstim[0]             # y
        avg_s1[1] = sumstim[2] / sumstim[0]             # x
        if sumstim[0] > 1:
            rms_s1[0] = math.sqrt(sumstim[3] / (sumstim[0] - 1.))
            rms_s1[1] = math.sqrt(sumstim[4] / (sumstim[0] - 1.))
        else:
            rms_s1[0] = math.sqrt(sumstim[3])
            rms_s1[1] = math.sqrt(sumstim[4])
    if total_counts2 > 0:
        avg_s2[0] = sumstim[6] / sumstim[5]
        avg_s2[1] = sumstim[7] / sumstim[5]
        if sumstim[5] > 1:
            rms_s2[0] = math.sqrt(sumstim[8] / (sumstim[5] - 1.))
            rms_s2[1] = math.sqrt(sumstim[9] / (sumstim[5] - 1.))
        else:
            rms_s2[0] = math.sqrt(sumstim[8])
            rms_s2[1] = math.sqrt(sumstim[9])

    if total_counts1 > 0 and total_counts2 > 0:
        stim_countrate = (total_counts1 + total_counts2) / (2. * exptime)
    elif total_counts1 > 0:
        stim_countrate = total_counts1 / exptime
    elif total_counts2 > 0:
        stim_countrate = total_counts2 / exptime
    else:
        stim_countrate = None
    if stim_countrate is not None and stimrate > 0.:
        stim_livetime = stim_countrate / stimrate
    else:
        stim_livetime = 1.

    if fd is not None:
        fd.close()

    stim_param = {"i0": i0, "i1": i1,
                  "x0": x0, "xslope": xslope,
                  "y0": y0, "yslope": yslope}

    return (stim_param, avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref,
            stim_countrate, stim_livetime)

def findStim(x, y, stim_ref, xwidth, ywidth):
    """Find one stim in time-tag data.

    Parameters
    ----------
    x: array like
        Array of detector X coordinates.

    y: array like
        Array of detector Y coordinates.

    stim_ref: tuple
        Reference position (y, x) for the stim.

    xwidth: int
        Half width of the search region in X.

    ywidth: int
        Half width of the search region in Y.

    Returns
    -------
    tuple, ((sy, sx), (sumysq, sumxsq), n, found_stim)
        (sy, sx) is the stim location (if found, else None),
        (sumysq, sumxsq) is the sum of squared deviations from the mean
        location,
        n is the number of events for this stim within the current time
        interval,
        found_stim will be True if there is at least one count within
        the search region.
    """

    # This is the search region for finding the stim.
    sxlow  = stim_ref[1] - xwidth
    sxhigh = stim_ref[1] + xwidth
    sylow  = stim_ref[0] - ywidth
    syhigh = stim_ref[0] + ywidth

    # Truncate at the lower and upper borders, excluding the first
    # and last lines.
    sylow = max(sylow, 1)
    syhigh = min(syhigh, 1022)

    # Initial value of mask is 1. (which in this case means "good").
    mask = np.ones(len(x), dtype=np.float32)

    # Now set mask to 0. ("bad") outside the search region.
    mask = np.where(x > sxhigh, 0., mask)
    mask = np.where(x < sxlow,  0., mask)
    mask = np.where(y > syhigh, 0., mask)
    mask = np.where(y < sylow,  0., mask)
    n = np.sum(mask)
    if n > 0.:
        # The stim reference position is subtracted before taking the sum
        # and then added back to the average in order to reduce the
        # possibility of numerical roundoff errors.
        sumx = np.sum((x-stim_ref[1]) * mask)
        sumy = np.sum((y-stim_ref[0]) * mask)
        sx = sumx / n + stim_ref[1]
        sy = sumy / n + stim_ref[0]
        # sum of squared deviations, for computing RMS
        sumxsq = np.sum((x-sx)**2 * mask)
        sumysq = np.sum((y-sy)**2 * mask)
        found_stim = True
    else:
        sx = None
        sy = None
        sumxsq = None
        sumysq = None
        found_stim = False

    return ((sy, sx), (sumysq, sumxsq), n, found_stim)

def updateStimSum(sumstim, nevents1, s1, sumsq1, found_s1,
                  nevents2, s2, sumsq2, found_s2):
    """Update sums for averages of stim positions.

    Parameters
    ----------
    sumstim: tuple with current sums
        n1      number of events for first stim
        sum1y   sum for first stim, Y coordinate
        sum1x   sum for first stim, X coordinate
        sumsq1y sum of squares for first stim, Y coordinate
        sumsq1x sum of squares for first stim, X coordinate
        n2      number of events for second stim
        sum2y   sum for second stim, Y coordinate
        sum2x   sum for second stim, X coordinate
        sumsq2y sum of squares for second stim, Y coordinate
        sumsq2x sum of squares for second stim, X coordinate

    nevents1: int
        Number of events for first stim in current time interval.

    s1: tuple of two floats
        Tuple of (y,x) coordinates of the first stim in current interval.

    found_s1: boolean
        True if the first stim was actually found.

    nevents2: int
        Number of events for second stim in current time interval.

    s2: tuple of two floats
        Same as s1, but for the second stim.

    found_s2: boolean
        True if the second stim was actually found.

    Returns
    -------
    tuple
        An updated sumstim tuple.

    nevents1 and nevents2 are used as weights when incrementing the
    sums.  n1 and n2 are the total number of events for the first and
    second stims respectively.
    """

    (n1, sum1y, sum1x, sumsq1y, sumsq1x,
     n2, sum2y, sum2x, sumsq2y, sumsq2x) = sumstim

    if found_s1:
        n1 = n1 + nevents1
        sum1y = sum1y + s1[0] * nevents1
        sum1x = sum1x + s1[1] * nevents1
        sumsq1y = sumsq1y + sumsq1[0]
        sumsq1x = sumsq1x + sumsq1[1]

    if found_s2:
        n2 = n2 + nevents2
        sum2y = sum2y + s2[0] * nevents2
        sum2x = sum2x + s2[1] * nevents2
        sumsq2y = sumsq2y + sumsq2[0]
        sumsq2x = sumsq2x + sumsq2[1]

    return (n1, sum1y, sum1x, sumsq1y, sumsq1x,
            n2, sum2y, sum2x, sumsq2y, sumsq2x)

def stimKeywords(hdr, segment, avg_s1, avg_s2, rms_s1, rms_s2,
                 s1_ref, s2_ref):
    """Update keywords for the locations of the stims.

    hdr: astropy.io.fits Header object
        The input events extension header (updated).

    segment: {"FUVA", "FUVB"}
        Segment name.

    avg_s1:  list of two floats
        avg_s1[0] is the average Y location of the first stim.
        avg_s1[1] is the average X location of the first stim.

    avg_s2:  list of two floats
        avg_s2[0] is the average Y location of the second stim.
        avg_s2[1] is the average X location of the second stim.

    rms_s1:  list of two floats
        rms_s1[0] is the RMS in Y for the first stim.
        rms_s1[1] is the RMS in X for the first stim.

    rms_s2:  list of two floats
        rms_s2[0] is the RMS in Y for the second stim.
        rms_s2[1] is the RMS in X for the second stim.

    s1_ref:  list of two floats
        s1_ref[0] is the Y position of the first stim, from the BRFTAB.
        s1_ref[1] is the X position of the first stim, from the BRFTAB.

    s2_ref:  list of two floats
        s2_ref[0] is the Y position of the second stim, from the BRFTAB.
        s2_ref[1] is the X position of the second stim, from the BRFTAB.
    """

    seg = segment[-1]           # "A" or "B"

    hdr["STIM"+seg+"0LX"] = s1_ref[1]
    hdr["STIM"+seg+"0LY"] = s1_ref[0]
    hdr["STIM"+seg+"0RX"] = s2_ref[1]
    hdr["STIM"+seg+"0RY"] = s2_ref[0]

    if avg_s1[0] is None or avg_s1[1] is None:
        hdr["STIM"+seg+"_LX"] = -1.
        hdr["STIM"+seg+"_LY"] = -1.
    else:
        hdr["STIM"+seg+"_LX"] = round(avg_s1[1], 3)
        hdr["STIM"+seg+"_LY"] = round(avg_s1[0], 3)
        hdr["STIM"+seg+"SLX"] = round(rms_s1[1], 3)
        hdr["STIM"+seg+"SLY"] = round(rms_s1[0], 3)

    if avg_s2[0] is None or avg_s2[1] is None:
        hdr["STIM"+seg+"_RX"] = -1.
        hdr["STIM"+seg+"_RY"] = -1.
    else:
        hdr["STIM"+seg+"_RX"] = round(avg_s2[1], 3)
        hdr["STIM"+seg+"_RY"] = round(avg_s2[0], 3)
        hdr["STIM"+seg+"SRX"] = round(rms_s2[1], 3)
        hdr["STIM"+seg+"SRY"] = round(rms_s2[0], 3)

def thermalParam(s1, s2, s1_ref, s2_ref):
    """Compute linear thermal distortion correction from stim positions.

    Parameters
    ----------
    s1: tuple
        Measured location in raw data of first stim (y, x).

    s2: tuple
        Measured location in raw data of second stim (y, x).

    s1_ref: tuple
        Reference location of first stim (y, x).

    s2_ref: tuple
        Reference location of second stim (y, x).

    Returns
    -------
    tuple of four floats, (xintercept, xslope, yintercept, yslope)
        The values are used as follows to apply the thermal distortion
        correction:
            xcorr = xintercept + xcorr * xslope
            ycorr = yintercept + ycorr * yslope
        where xcorr and ycorr are slices within the XCORR and YCORR
        columns.
    """

    if s1[0] is None or s2[0] is None:

        xslope = 1.
        xintercept = 0.
        yslope = 1.
        yintercept = 0.

    else:

        xslope = (s2_ref[1] - s1_ref[1]) / (s2[1] - s1[1])
        xintercept = s1_ref[1] - s1[1] * xslope

        yslope = (s2_ref[0] - s1_ref[0]) / (s2[0] - s1[0])
        yintercept = s1_ref[0] - s1[0] * yslope

    return (xintercept, xslope, yintercept, yslope)

def doTempcorr(stim_param, events, info, switches, reffiles, phdr):
    """Apply thermal distortion correction.

    Parameters
    ----------
    stim_param: dictionary of lists
        The dictionary has keys i0, i1, x0, xslope, y0, yslope.

    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch("TEMPCORR", switches)
        if switches["tempcorr"] == "PERFORM":
            cosutil.printRef("BRFTAB", reffiles)
            # The function value is true if a correction was actually applied.
            if thermalDistortion(events.field(xcorr),
                                 events.field(ycorr), stim_param):
                phdr["tempcorr"] = "COMPLETE"
            else:
                phdr["tempcorr"] = "SKIPPED"
                cosutil.printWarning("TEMPCORR was skipped")

def thermalDistortion(x, y, stim_param):
    """Apply thermal distortion correction to positions in events list.

    No correction is necessary and none will be applied if the slopes are
    all 0 and the intercepts are all 1.

    Parameters
    ----------
    x: array like
        Array of detector X coordinates.

    y: array like
        Array of detector Y coordinates.

    stim_param: dictionary of lists
        The dictionary has keys i0, i1, x0, xslope, y0, yslope.

    Returns
    -------
    boolean
        True if a correction was actually applied in at least one of
        the time intervals.
    """

    # These are the parameters found by computeThermalParam.
    x0 = stim_param["x0"]
    xslope = stim_param["xslope"]
    y0 = stim_param["y0"]
    yslope = stim_param["yslope"]

    actually_done = False

    if "i0" in stim_param:
        i0 = stim_param["i0"]
        i1 = stim_param["i1"]
    else:
        i0 = [0]
        i1 = [len(x)]

    for n in range(len(i0)):
        i = i0[n]
        j = i1[n]
        if x0[n] != 0. or xslope[n] != 1. or \
           y0[n] != 0. or yslope[n] != 1.:
            x[i:j] = x0[n] + x[i:j] * xslope[n]
            y[i:j] = y0[n] + y[i:j] * yslope[n]
            actually_done = True

    return actually_done

def doGeocorr(events, info, switches, reffiles, phdr):
    """Apply geometric correction.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch("GEOCORR", switches)
        if switches["geocorr"] == "PERFORM":
            cosutil.printRef("GEOFILE", reffiles)
            cosutil.printSwitch("IGEOCORR", switches)
            cosutil.geometricDistortion(events.field(xcorr),
                                        events.field(ycorr),
                                        reffiles["geofile"],
                                        info["segment"], switches["igeocorr"])
            phdr["geocorr"] = "COMPLETE"
            if switches["igeocorr"] == "PERFORM":
                phdr["igeocorr"] = "COMPLETE"

def doDgeocorr(events, info, switches, reffiles, phdr):
    """Apply delta geometric correction.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch("DGEOCORR", switches)
        if switches["dgeocorr"] == "PERFORM":
            cosutil.printRef("DGEOFILE", reffiles)
            cosutil.printSwitch("IGEOCORR", switches)
            cosutil.geometricDistortion(events.field(xcorr),
                                        events.field(ycorr),
                                        reffiles["dgeofile"],
                                        info["segment"], switches["igeocorr"])
            phdr["dgeocorr"] = "COMPLETE"
            if switches["igeocorr"] == "PERFORM":
                phdr["igeocorr"] = "COMPLETE"

def doWalkCorr(switches):
    """Returns True if we are going to do either xwlkcorr or ywlkcorr
    """
    if switches["xwlkcorr"] == "PERFORM" or switches["ywlkcorr"] == "PERFORM":
        return True
    else:
        return False

def doXWalkcorr(events, info, switches, reffiles, phdr):
    """Apply X walk correction.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch("XWLKCORR", switches)
        if switches["xwlkcorr"] == "PERFORM":
            cosutil.printRef("XWLKFILE", reffiles)
            xcorrection = walkCorrection(events.field('xcorr'),
                                         events.field('pha'),
                                         reffiles["xwlkfile"],
                                         info["segment"])
            phdr["xwlkcorr"] = "COMPLETE"
            return xcorrection
        else:
            return None

def doYWalkcorr(events, info, switches, reffiles, phdr):
    """Apply Y walk correction.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch("YWLKCORR", switches)
        if switches["ywlkcorr"] == "PERFORM":
            cosutil.printRef("YWLKFILE", reffiles)
            ycorrection = walkCorrection(events.field('xcorr'),
                                         events.field('pha'),
                                         reffiles["ywlkfile"],
                                         info["segment"])
            phdr["ywlkcorr"] = "COMPLETE"
            return ycorrection
        else:
            return None

def walkCorrection(fastCoordinate, slowCoordinate, reference_file, segment):
    """Apply walk correction
    The same algorithm is used for both.
    slowCoordinate and fastCoordinate are arrays of coordinates that are used
    to look up in the reference image

    Parameters
    ----------
    slowCoordinate: numpy ndarray
        The array of coordinates that is used to look up the correction in the
        slow direction of the reference array

    fastCoordinate: numpy ndarray
        The array of coordinates that is used to look up the correction in the
        fast direction of the reference array

    reference_file: string
        Name of reference file

    segment: string
        FUV segment ("FUVA" or "FUVB")

    Returns:
    --------

    correction: numpy ndarray
        The array of lookups in the reference array

    """
    nevents = len(fastCoordinate)
    fd = fits.open(reference_file)
    for extension in fd[1:]:
        if extension.header['SEGMENT'] == segment:
            reference_array = extension.data
            break
    delta = np.zeros(len(fastCoordinate))
    delta = bilinear_interpolation(fastCoordinate, slowCoordinate,
                           reference_array)
    return delta

def bilinear_interpolation(fastCoordinate, slowCoordinate,
                               reference_array):
    nrows, ncols = reference_array.shape
    extended_ref = np.zeros((nrows+1,ncols+1),
                            dtype=reference_array.dtype)
    extended_ref[:nrows,:ncols] = reference_array
    fastcopy = fastCoordinate.copy()
    slowcopy = slowCoordinate.copy()
    nevents = len(fastCoordinate)
    negx = np.where(fastcopy < 0.0)
    xtoobig = np.where(fastcopy > ncols-1)
    fastcopy[negx] = 0.0
    fastcopy[xtoobig] = ncols - 1.0
    ix = fastcopy.astype(np.int32)
    negy = np.where(slowcopy < 0.0)
    ytoobig = np.where(slowcopy > nrows-1)
    slowcopy[negy] = 0.0
    slowcopy[ytoobig] = nrows-1.0
    iy = slowcopy.astype(np.int32)
    ix1 = ix + 1
    iy1 = iy + 1
    dx1 = fastcopy - ix
    dx2 = 1.0 - dx1
    dy1 = slowcopy - iy
    dy2 = 1.0 - dy1
    flat = extended_ref.ravel()
    f11 = (ncols+1)*iy + ix
    f12 = (ncols+1)*iy1 + ix
    f21 = (ncols+1)*iy + ix1
    f22 = (ncols+1)*iy1 + ix1
    delta = flat[f11]*dx2*dy2 + flat[f12]*dx2*dy1 + \
        flat[f21]*dx1*dy2 + flat[f22]*dx1*dy1
    return delta

def applyWalkCorrection(events, xcorrection, ycorrection):
    """Apply the walk correction
    """
    global active_area
    if xcorrection is not None:
        events['xcorr'] = np.where(active_area, events['xcorr'] - xcorrection,
                                   events['xcorr'])
    if ycorrection is not None:
        events['ycorr'] = np.where(active_area, events['ycorr'] - ycorrection,
                                   events['ycorr'])
    return

def doDqicorr(events, input, info, switches, reffiles,
               phdr, hdr, minmax_shift_dict, traceprofile, gti):
    """Create a data quality array, initialized from the DQI table.

    This function applies the data quality initialization table (bpixtab)
    to two arrays, the 2-D DQ image extension and the 1-D DQ events table
    column.

    The 2-D DQ image array dq_array is created and initialized to zero.
    The 1-D DQ events table column, on the other hand, is not initialized
    because it may already contain meaningful flags from pulse-height or
    time filtering.  Note that flags for pulse-height or time filtering
    that are set in the 1-D DQ table column are _not_ included in the 2-D
    image array, since they would be associated with either specific events
    or time intervals, rather than spatial regions on the detector.

    The function value is the 2-D data quality image array, which may be
    filled with zeros.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    input: str
        Name of raw file, used for getting DQ array for ACCUM data.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.

    hdr: astropy.io.fits Header object
        The input events extension header.

    minmax_shift_dict: dictionary
        Each key is a tuple (lower_y, upper_y), the lower and upper limits
        of a slice in axis 0 within the data quality array.  The value is
        a list [min_shift1, max_shift1, min_shift2, max_shift2] of the
        minimum and maximum offsets (determined via the wavecal) in the
        dispersion direction and in the cross-dispersion direction during
        the exposure.  The idea here is that the wavecal shifts may be
        different for the science spectrum and the wavecal, and for NUV
        data the shifts differ from one stripe to another.

    traceprofile: 1-d array
        The DQ array needs to be shifted in the y direction by an amount equal
        to the trace correction, since that shift was applied to the YFULL
        values of each event

    gti: list
        List of good time intervals.  Used to check for overlap with hotspots.

    Returns
    -------
    array like
        2-D data quality array.
    """

    # temp_switch is only used for printing the DQICORR message
    temp_switch = {}
    if switches["dqicorr"] == "COMPLETE" and info["corrtag_input"]:
        temp_switch["dqicorr"] = "PERFORM (complete, but repeat)"
    else:
        temp_switch["dqicorr"] = switches["dqicorr"]
    cosutil.printSwitch("DQICORR", temp_switch)

    if info["obsmode"] == "TIME-TAG" or info["corrtag_input"]:
        # Create an initially zero 2-D data quality extension array.
        dq_array = np.zeros(info["npix"], dtype=np.int16)
    else:
        # Read the data quality array from the rawaccum file.
        dq_array = cosutil.getInputDQ(input)

    # If the input is a corrtag file and dqicorr was done when that file was
    # created, we should do dqicorr again.
    if switches["dqicorr"] == "PERFORM" or switches["dqicorr"] == "COMPLETE":

        cosutil.printRef("BPIXTAB", reffiles)
        if "gsagtab" in reffiles and reffiles["gsagtab"] != NOT_APPLICABLE:
            cosutil.printRef("GSAGTAB", reffiles)
        if "spottab" in reffiles and reffiles["spottab"] != NOT_APPLICABLE:
            cosutil.printRef("SPOTTAB", reffiles)
            #
            # Check that the header keywords in the SPOTTAB reference file
            # agree with the PHAFILE/PHATAB, and print a warning if they
            # don't
            cosutil.checkSpottabKeywords(reffiles, info)
        # Update the dq column in the events list with the bpixtab regions.
        # This also gets the gsagtab regions and the hotspot regions
        (lx, ly, dx, dy, dq, extn, message) = \
                cosutil.getDQArrays(info, reffiles, gti)
        if message:
            cosutil.printWarning(message)
        if len(lx) > 0:
            pharange = cosutil.getPulseHeightRange(hdr, info["segment"])
            # xxx temporary; eventually select rows based on pharange
            bpixtab = reffiles["bpixtab"]
            ref_pharange = cosutil.tempPulseHeightRange(bpixtab)
            cosutil.comparePulseHeightRanges(pharange, ref_pharange, bpixtab)
            # The location of the flags is based on xcorr & ycorr, because
            # the bad regions depend on the detector, not the science data.
            ccos.applydq(lx, ly, dx, dy, dq,
                         events.field(xcorr), events.field(ycorr),
                         events.field("dq"))

        # Copy values from the bpixtab to the dq_array, applying offsets
        # depending on the wavecal shift and the Doppler shift.
        (doppmag, doppzero, orbitper) = dopplerParam(info,
                                reffiles["disptab"], switches["doppcorr"])
        minmax_doppler = cosutil.minmaxDoppler(info, switches["doppcorr"],
                               doppmag, doppzero, orbitper)
        if info["obstype"] == "SPECTROSCOPIC" and \
           minmax_doppler[0] != 0. and minmax_doppler[1] != 0.:
            doppler_boundary = psaWcaBoundary(info, reffiles["xtractab"])
        else:
            doppler_boundary = -10
        #
        # If the trace correction or alignment correction has been performed,
        # we need to handle
        # the DQ arrays differently - need to apply the trace correction
        # to the DQ array in (XCORR, YCORR) space, then apply the range of
        # shift1 and shift2 values in a bitwise_or 'convolution'
        doBlur = False
        #
        # If the alignment correction has been performed, add this to the
        # trace profile
        alignment_correction = 0.0
        keyword = "SP_OFF_" + info["segment"][-1]
        #
        # Check the SP_OFF_[AB] keyword.  If it exists, and is not -999.0,
        # set doBlur to True.  Otherwise, set keep it False
        try:
            #
            # Since the trace correction is SUBTRACTED, whereas the alignment
            # corrction is ADDED, we need to subtract the alignment correction
            # from the trace correction to ensure they are applied correctly.
            # The header keyword is the NEGATIVE of the alignment correction,
            # so we'll just add this.
            alignment_correction = hdr[keyword]
            if alignment_correction > -998.0:
                doBlur = True
            else:
                alignment_correction = 0.0
        except KeyError:
            alignment_correction = 0.0
        #
        # Check whether the trace correction was done.  If it was, and even if
        # the alignment correction is zero or not done, add the alignment
        # correction and set doBlur to True
        try:
            if phdr['TRCECORR'] == 'COMPLETE':
                #
                # traceprofile is returned by trace.doTrace, and is None if TRCECORR is not
                # 'PERFORM'.  We need to check against this, and if traceprofile is None, we
                # need to reload the traceprofile.  This can happen if calcos is run twice,
                # once up to and including the trace correction (so that TRCECORR becomes
                # 'COMPLETE'), and then again starting at ALGNCORR.  Then TRCECORR is 'COMPLETE',
                # but we don't have a traceprofile from this run...
                if traceprofile is None:
                    traceprofile = trace.getTrace(reffiles['tracetab'], info)
                traceprofile = traceprofile + alignment_correction
                doBlur = True
            #
            # If there's no TRCECORR keyword (e.g. for NUV data),  we don't do the blur
            # correction
        except KeyError:
            doBlur = False

        if doBlur:
            #
            # If we get to this point and the traceprofile is still None, just
            # make a zeroed out traceprofile and add the alignment correction to it
            # This should only happen if TRCECORR is set to 'OMIT' and ALGNCORR is
            # set to PERFORM
            if traceprofile is None:
                cosutil.printWarning("No trace profile, using zero")
                traceprofile = np.zeros((info["npix"][1]), dtype=np.float32)
                traceprofile = traceprofile + alignment_correction
            #
            # First zero out the minmax_doppler dictionary and force the
            # minmax_shift_dict to just be [0, 0, shift2, shift2]
            temp_minmax_shift_dict = minmax_shift_dict.copy()
            temp_minmax_doppler = (0.0, 0.0)
            key = "SHIFT2" + info["segment"][-1]
            shift2 = hdr[key]
            for regionkey in temp_minmax_shift_dict.keys():
                temp_minmax_shift_dict[regionkey] = [0.0, 0.0,
                                                     shift2, shift2]
            #
            # Get the dq_array in (XCORR, YCORR) space
            cosutil.updateDQArray(info, reffiles, dq_array,
                                  temp_minmax_shift_dict,
                                  temp_minmax_doppler, doppler_boundary, gti)
            # Flag regions that are outside any subarray as out of bounds.
            cosutil.flagOutOfBounds(hdr, dq_array, info, switches,
                                    reffiles["brftab"], reffiles["geofile"],
                                    reffiles["dgeofile"],
                                    temp_minmax_shift_dict,
                                    temp_minmax_doppler, doppler_boundary)
            # Flag the region that is outside the active area.
            cosutil.flagOutsideActiveArea(dq_array, info["segment"],
                                          reffiles["brftab"], info["x_offset"],
                                          temp_minmax_shift_dict,
                                          temp_minmax_doppler)
            #
            # Apply the trace correction to the DQ array by shifting it down
            # by the amount in the trace profile.  Don't do any shift in the
            # WCA aperture
            filter = {"segment": info["segment"],
                      "opt_elem": info["opt_elem"],
                      "cenwave": info["cenwave"],
                      "aperture": "WCA"
                      }
            wca_row = cosutil.getTable(reffiles["xtractab"], filter)
            trace_dq = traceShiftDQ(dq_array, traceprofile, wca_row)
            #
            # Now do the bitwise_or blurring
            widen = hdr.get("widen", default=PIXEL_FRACTION)
            #
            # We need to back out SHIFT2 from the minmax_shift_dict
            # as we included it when we made the DQ array
            temp_minmax_shift_dict = minmax_shift_dict.copy()
            for regionkey in temp_minmax_shift_dict.keys():
                [min_shift1, max_shift1, min_shift2, max_shift2] = \
                    temp_minmax_shift_dict[regionkey]
                temp_minmax_shift_dict[regionkey] = [min_shift1,
                                                     max_shift1,
                                                     min_shift2 - shift2,
                                                     max_shift2 - shift2]
            dq_array = blurDQ(trace_dq, temp_minmax_shift_dict, minmax_doppler,
                              doppler_boundary, widen)
        else:
            cosutil.updateDQArray(info, reffiles, dq_array,
                                  minmax_shift_dict,
                                  minmax_doppler, doppler_boundary, gti)
            # Flag regions that are outside any subarray as out of bounds.
            cosutil.flagOutOfBounds(hdr, dq_array, info, switches,
                                    reffiles["brftab"], reffiles["geofile"],
                                    reffiles["dgeofile"],
                                    minmax_shift_dict,
                                    minmax_doppler, doppler_boundary)
            # Flag the region that is outside the active area.
            if info["detector"] == "FUV":
                cosutil.flagOutsideActiveArea(dq_array, info["segment"],
                                              reffiles["brftab"], info["x_offset"],
                                              minmax_shift_dict,
                                              minmax_doppler)

        phdr["dqicorr"] = "COMPLETE"

        if extn is not None:
            if "gsagtab" in phdr:
                # replace the comment, to give the extension number
                segment = info["segment"]
                gsagtab = phdr["gsagtab"]
                if extn > 0:
                    comment = "ext. %d for %s" % (extn, segment)
                else:
                    comment = "no ext. for %s" % segment
                phdr["gsagtab"] = (gsagtab, comment)

    return dq_array

def traceShiftDQ(dq_array, traceprofile, wca_row):
    """Shift the DQ array by the amount in the traceprofile.  This must be
    SUBTRACTED from the row numbers.  Only do this outside the WCA, the same as for
    the SCI data
    For each column, we copy the WCA region to the output, then shift the regions
    below and above the WCA aperture by the trace+aligncorr shift"""
    shifted_dq = dq_array.copy() * 0
    nrows, ncolumns = dq_array.shape
    wca_0 = wca_row["B_SPEC"][0]
    wcaslope = wca_row["SLOPE"][0]
    wcaheight = wca_row["HEIGHT"][0]
    for column in range(ncolumns):
    #
    # Calculate the extent of the WCA aperture
        wcacenter = wca_0 + int(round(column*wcaslope))
        wcastart = wcacenter - wcaheight // 2
        wcastop = wcacenter + wcaheight // 2
        tracevalue = int(round(traceprofile[column]))
        #
        # Put the unshifted (WCA) region into the output array
        shifted_dq[int(wcastart):int(wcastop+1), column] = dq_array[int(wcastart):int(wcastop+1), column]
        #
        # Shift the region below the WCA
        instart = max(0, tracevalue)
        instop = min(wcastart+tracevalue-1, wcastart-1)
        outstart = max(0, -tracevalue)
        outstop = min(wcastart-tracevalue-1, wcastart-1)
        n_in = instop - instart + 1
        n_out = outstop - outstart + 1
        if n_out != n_in:
            cosutil.printWarning("Input and output arrays have different sizes")
        shifted_dq[int(outstart):int(outstop+1), column] = dq_array[int(instart):int(instop+1), column]
        #
        # Now the part above the WCA
        instart = max(wcastop+1, wcastop+1+tracevalue)
        instop = min(nrows+tracevalue-1, nrows-1)
        outstart = max(wcastop+1, wcastop+1-tracevalue)
        outstop = min(nrows-tracevalue-1, nrows-1)
        n_in = instop - instart + 1
        n_out = outstop - outstart + 1
        if n_out != n_in:
            cosutil.printWarning("Input and output arrays have different sizes")
        shifted_dq[int(outstart):int(outstop+1), column] = dq_array[int(instart):int(instop+1), column]
    return shifted_dq

def  blurDQ(trace_dq, minmax_shift_dict, minmax_doppler, doppler_boundary, widen):
    """Blur the DQ array by shifting it by the range of values in
    minmax_shift_dict and minmax_doppler, and then bitwise_OR-ing it with
    the running DQ array"""
    nrows, ncols = trace_dq.shape
    (mindopp, maxdopp) = minmax_doppler
    if doppler_boundary > 0.0:
        #
        # Split into two regions, below the doppler boundary and above the
        # doppler boundary
        key = list(minmax_shift_dict.keys())[0]
        value = minmax_shift_dict[key]
        (lower_y, upper_y) = key
        minmax_dict = {(lower_y, doppler_boundary): value,
                       (doppler_boundary, upper_y): value}
    else:
        cosutil.printWarning("Running blurDQ when doppler boundary <= 0.0")
        minmax_dict = minmax_shift_dict
    #
    # Do the shift and blur.  Do the Y first because the values are the same on either side
    # of the Doppler boundary
    blur_dq = trace_dq.copy() * 0
    yshifts = []
    #
    # For the y shifts we can use the original minmax_shift_dict
    [min_shift1, max_shift1, min_shift2, max_shift2] = \
        list(minmax_shift_dict.values())[0]
    yshifts.append(int(round(max_shift2 + widen)))
    yshifts.append(int(round(min_shift2 - widen)))

    for yshift in range(min(yshifts), max(yshifts)+1):
        y_shifted_dq = arrayShift(trace_dq, yshift, 0, DQ_PIXEL_OUT_OF_BOUNDS)
        #
        # Now do the shift and blur in x
        keys = sorted(minmax_dict)
        for key in keys:
            xshifts = []
            (lower_y, upper_y) = key
            [min_shift1, max_shift1, min_shift2, max_shift2] = \
                minmax_dict[key]
            if doppler_boundary > 0 and ((lower_y + upper_y) // 2 < doppler_boundary):
                xshifts.append(int(round(max_shift1 + maxdopp + widen)))
                xshifts.append(int(round(min_shift1 + mindopp - widen)))
            else:
                xshifts.append(int(round(max_shift1 + widen)))
                xshifts.append(int(round(min_shift1 - widen)))
            for xshift in range(min(xshifts), max(xshifts)+1):
                cosutil.printMsg("Shifting to %d, %d" % (xshift, yshift))
                shifted_dq = arrayShift(y_shifted_dq[int(lower_y):int(upper_y)], 0, xshift,
                                        DQ_PIXEL_OUT_OF_BOUNDS)
                blur_dq[int(lower_y):int(upper_y)] = np.bitwise_or(blur_dq[int(lower_y):int(upper_y)],
                                                                   shifted_dq)

    return blur_dq

def arrayShift(array, yshift, xshift, default):
    """Shift an array by xshift in x and yshift in y"""
    outarray = array.copy() * 0 + default
    nrows, ncols = array.shape
    inxstart = int(max(0, xshift))
    inxstop = int(min(ncols + xshift, ncols))
    outxstart = int(max(0, -xshift))
    outxstop = int(min(ncols - xshift, ncols))
    inystart = int(max(0, yshift))
    inystop = int(min(nrows + yshift, nrows))
    outystart = int(max(0, -yshift))
    outystop = int(min(nrows - yshift, nrows))
    outarray[outystart:outystop,outxstart:outxstop] = array[inystart:inystop, inxstart:inxstop]
    return outarray

def dopplerParam(info, disptab, doppcorr):
    """Return the appropriate set of Doppler keyword values.

    Different keywords will be used depending on whether the data are
    TIME-TAG or ACCUM.

    Parameters
    ----------
    info: dictionary
        Keywords and values.

    disptab: str
        Name of dispersion relation table.

    doppcorr: str
        If Doppler correction is OMIT or SKIPPED, return dummy values.

    Returns
    -------
    tuple, (doppmag, doppzero, orbitper)
        Doppler magnitude in pixels, time (MJD) when the Doppler shift
        is zero and increasing, period (seconds) of HST.
    """

    if doppcorr == "OMIT" or doppcorr == "SKIPPED":
        doppmag  = 0.
        doppzero = info["expstart"]
        orbitper = 5760.
    elif info["obsmode"] == "TIME-TAG":
        # Get the dispersion (for stripe B, if NUV) in order to convert the
        # Doppler magnitude from km/s to pixels.
        filter = {"opt_elem": info["opt_elem"],
                  "cenwave": info["cenwave"],
                  "fpoffset": info["fpoffset"],
                  "aperture": info["aperture"]}
        if info["detector"] == "FUV":
            filter["segment"] = info["segment"]
            middle = float(FUV_X) / 2.
        else:
            filter["segment"] = "NUVB"
            middle = float(NUV_X) / 2.
        disp_rel = dispersion.Dispersion(disptab, filter)
        if not disp_rel.isValid():
            raise MissingRowError("missing row in disptab")
        # get the dispersion (disp) at the middle of the detector
        disp = disp_rel.evalDerivDisp(middle)
        disp_rel.close()
        # Compute the Doppler shift in pixels from the shift in km/s.
        if disp <= 0.:
            doppmag = 0.
        else:
            doppmag = (info["doppmagv"] / SPEED_OF_LIGHT) * \
                      (info["cenwave"] / disp)
        doppzero = info["doppzero"]
        orbitper = info["orbitper"]

    else:               # ACCUM
        doppmag  = info["dopmagt"]
        doppzero = info["dopzerot"]
        orbitper = info["orbtpert"]

    return (doppmag, doppzero, orbitper)

def doDoppcorr(events, info, switches, reffiles, phdr):
    """Apply Doppler correction to the x and y pixel coordinates.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.
    """

    if info["obsmode"] == "ACCUM":              # done on-board
        return

    if info["obstype"] == "SPECTROSCOPIC":
        cosutil.printSwitch("DOPPCORR", switches)

    if switches["doppcorr"] == "PERFORM" or switches["doppcorr"] == "COMPLETE":

        # xi and eta are the columns of pixel coordinates for the
        # dispersion and cross-dispersion directions respectively.
        # (explicit column names are used here for clarity)
        xi = events.field("xcorr")
        eta = events.field("ycorr")
        dopp = events.field("xdopp")
        xi_full  = events.field("xfull")

        cosutil.printRef("XTRACTAB", reffiles)
        cosutil.printRef("DISPTAB", reffiles)
        if info["detector"] == "FUV":
            cosutil.printRef("BRFTAB", reffiles)

        xtractab = reffiles["xtractab"]
        if info["detector"] == "FUV":
            # This array of flags indicates which events should be corrected.
            region_flags = fuvDopplerRegions(eta, info, xtractab)
            # Apply the orbital Doppler correction to the flagged events.
            dopp[:] = np.where(region_flags,
                               dopplerCorrection(events.field("time"),
                                                 xi, info, reffiles),
                               xi)
        else:
            region_flags_dict = nuvPsaRegions(eta, info, xtractab)
            dopp[:] = xi
            for stripe in ["NUVA", "NUVB", "NUVC"]:
                dopp[:] = np.where(region_flags_dict[stripe],
                                   dopplerCorrection(events.field("time"),
                                                     xi, info, reffiles,
                                                     stripe=stripe),
                                   dopp)

        # Copy to xfull if wavecal processing will not be done.
        if switches["wavecorr"] == "OMIT" and not info["corrtag_input"]:
            xi_full[:] = dopp.copy()

        phdr["doppcorr"] = "COMPLETE"

def psaWcaBoundary(info, xtractab):
    """Determine the boundary between PSA and WCA.

    The computation of the 'boundary' variable makes an assumption
    about the relative locations of the PSA and WCA regions on the
    detectors.  The PSA spectral region is at lower Y pixel numbers.

    Parameters
    ----------
    info: dictionary
        Keywords and values.

    xtractab: str
        Name of spectral extraction parameters reference table.

    Returns
    -------
    int
        The YCORR coordinate between the PSA region and the WCA region,
        at the middle column of the detector
    """

    if info["detector"] == "FUV":
        middle = float(FUV_X) / 2.
    else:
        middle = float(NUV_X) / 2.

    # Protect against the possibility that the aperture keyword is "WCA".
    if info["aperture"] == "BOA":
        aperture = "BOA"
    else:
        aperture = "PSA"

    if info["detector"] == "FUV":
        segment = info["segment"]
    else:
        segment = "NUVC"

    filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"],
              "segment": segment, "aperture": aperture}

    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_psa = xtract_info.field("b_spec")[0] + \
                 xtract_info.field("slope")[0] * middle

    filter["aperture"] = "WCA"
    if info["detector"] == "NUV":
        filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_wca = xtract_info.field("b_spec")[0] + \
                 xtract_info.field("slope")[0] * middle

    boundary = int(round((b_spec_psa + b_spec_wca) / 2.))

    return boundary

def fuvDopplerRegions(eta, info, xtractab):
    """Determine the region over which Doppler shift should be applied.

    This version is for FUV data.

    Parameters
    ----------
    eta: array like
        Pixel coordinates in cross-dispersion direction.

    info: dictionary
        Keywords and values.

    xtractab: str
        Name of spectral extraction parameters reference table.

    Returns
    -------
    array like, boolean
        True for events that are within the region for which it would be
        appropriate to apply Doppler correction.  There is one element for
        each event in the table.
    """

    global active_area

    region_flags = active_area.copy()

    boundary = psaWcaBoundary(info, xtractab)

    region_flags &= (eta < boundary)

    return region_flags

def dopplerCorrection(time, xi, info, reffiles, stripe=None):
    """Apply orbital and heliocentric Doppler correction.

    Parameters
    ----------
    time: array like
        Times of events (seconds).

    xi: array like
        Pixel coordinates of events, in dispersion direction.

    info: dictionary
        Keywords and values.

    reffiles: dictionary
        Dictionary of reference file names.

    stripe: str
        Name of NUV stripe ("NUVA", "NUVB", "NUVC"), or None for FUV.

    Returns
    -------
    array like
        Array of Doppler-corrected X pixel coordinates.
    """

    disptab = reffiles["disptab"]

    # Compute the wavelength and dispersion at each pixel.
    if info["aperture"] in APERTURE_NAMES:
        aperture = info["aperture"]
    else:
        cosutil.printWarning("Aperture %s temporarily set to PSA for computing"
                             " Doppler correction." % info["aperture"])
        aperture = "PSA"
    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": aperture,
              "fpoffset": info["fpoffset"]}
    if stripe is None:
        filter["segment"] = info["segment"]
    else:
        filter["segment"] = stripe
    disp_rel = dispersion.Dispersion(disptab, filter, use_fpoffset=True)
    if not disp_rel.isValid():
        disp_rel.close()
        raise MissingRowError("missing row in disptab")

    xi = xi.astype(np.float64)
    fpoffset_present = cosutil.findColumn(disptab, "fpoffset")
    if fpoffset_present:
        # Compute wavelength and dispersion at each element of xi.
        wavelength = disp_rel.evalDisp(xi)
        disp = disp_rel.evalDerivDisp(xi)
    else:
        # Correct for fpoffset when computing wavelength and dispersion
        # (a feature will be at larger pixel number if fpoffset is larger,
        # so the wavelength at a given pixel will be smaller).
        wcp_info = cosutil.getTable(reffiles["wcptab"],
                                    filter={"opt_elem": info["opt_elem"]},
                                    exactly_one=True)
        stepsize = wcp_info.field("stepsize")[0]
        wavelength = disp_rel.evalDisp(xi)
        xi_temp = xi - info["fpoffset"] * stepsize
        wavelength = disp_rel.evalDisp(xi_temp)
        disp = disp_rel.evalDerivDisp(xi_temp)
        del xi_temp, wcp_info
    disp_rel.close()

    # Apply the Doppler correction to the pixel coordinates.
    xd = orbitalDoppler(time, xi, wavelength, disp, info["expstart"],
                        info["doppmagv"], info["doppzero"], info["orbitper"])

    return xd

def orbitalDoppler(time, xi, wavelength, dispersion, expstart,
                   doppmag_v, doppzero, orbitper):
    """Apply Doppler correction for HST orbital motion.

    Parameters
    ----------
    time: array like
        Times of events (seconds).

    xi: array like
        Pixel coordinates of events, in dispersion direction.

    wavelength: array like
        Wavelengths corresponding to xi (Angstroms).

    dispersion: array like
        Dispersion at each element of xi (Angstroms/pixel).

    expstart: float
        Exposure start time (MJD).

    doppmag_v: float
        Magnitude of Doppler shift (km/s).

    doppzero: float
        Time when orbital Doppler shift is zero and increasing (MJD).

    orbitper: float
        Orbital period of HST (seconds).

    Returns
    -------
    array like
        Doppler-corrected xi array.
    """

    # t is the time of each event in seconds since doppzero.
    t = (expstart - doppzero) * SEC_PER_DAY + time.astype(np.float64)

    shift = doppmag_v / SPEED_OF_LIGHT * wavelength / dispersion * \
            np.sin(2. * np.pi * t / orbitper)

    return xi - shift

def initHelcorr(events, info, hdr):
    """Compute the radial velocity and update the V_HELIO keyword.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Dictionary of header keywords and values.

    hdr: astropy.io.fits Header object
        The events extension header.
    """

    if info["obstype"] != "SPECTROSCOPIC":
        return

    # get midpoint of exposure, MJD
    expstart = info["expstart"]
    time = events.field("time")
    t_mid = expstart + (time[0] + time[len(time)-1]) / 2. / SEC_PER_DAY

    # Compute radial velocity and heliocentric correction factor (the latter
    # is actually not used here).
    radvel = heliocentricVelocity(t_mid, info["ra_targ"], info["dec_targ"])
    helio_factor = -radvel  / SPEED_OF_LIGHT
    hdr["v_helio"] = radvel
    info["v_helio"] = radvel

def heliocentricVelocity(t, ra_targ, dec_targ):
    """Compute heliocentric radial velocity.

    This is copied from the code for calstis, except that the target
    coordinates will not be precessed to the time of observation.

    Parameters
    ----------
    t: float
        Time (MJD).

    ra_targ: float
        Right ascension of the target (J2000).

    dec_targ: float
        Declination of the target (J2000).

    Returns
    -------
    float
        The contribution of the Earth's velocity around the Sun to the
        radial velocity of the target, in km/s; if the Earth is approaching
        the target, this will be negative (i.e. the sign convention is that
        radial velocity is positive if the distance between the Earth and
        the target is increasing).
    """

    REFDATE = 51544.5           # MJD for 2000 Jan 1.5 UT, or JD 2451545.0
    KM_AU   = 1.4959787e8       # astronomical unit in kilometers
    SEC_DAY = 86400.            # seconds per day

    deg_to_rad = math.pi / 180.
    eps = 23.439 * deg_to_rad           # obliquity of Earth's axis

    ra  = ra_targ * deg_to_rad
    dec = dec_targ * deg_to_rad

    # target will be a unit vector toward the target;
    # velocity will be Earth's orbital velocity in km/s.
    target = [0., 0., 0.]
    velocity = [0., 0., 0.]

    target[0] = math.cos(dec) * math.cos(ra)
    target[1] = math.cos(dec) * math.sin(ra)
    target[2] = math.sin(dec)

    # Precess the target coordinates to time t.
    # target = cosutil.precess(t, target)         # note, commented out

    dt = t - REFDATE                    # days since 2000 Jan 1, 12h UT

    g_dot = 0.9856003 * deg_to_rad
    l_dot = 0.9856474 * deg_to_rad

    eps = (23.439 - 0.0000004 * dt) * deg_to_rad

    g = mod2pi((357.528 + 0.9856003 * dt) * deg_to_rad)
    l = mod2pi((280.461 + 0.9856474 * dt) * deg_to_rad)

    #       L   1.915 degree             0.02 degree
    elong = l + 0.033423 * math.sin(g) + 0.000349 * math.sin(2.*g)
    elong_dot = l_dot + \
                0.033423 * math.cos(g) * g_dot + \
                0.000349 * math.cos(2.*g) * 2.*g_dot

    radius = 1.00014 - 0.01671 * math.cos(g) - 0.00014 * math.cos(2.*g)
    radius_dot =       0.01671 * math.sin(g) * g_dot + \
                       0.00014 * math.sin(2.*g) * 2.*g_dot

    x_dot = radius_dot * math.cos(elong) - \
                radius * math.sin(elong) * elong_dot

    y_dot = radius_dot * math.cos(eps) * math.sin(elong) + \
                radius * math.cos(eps) * math.cos(elong) * elong_dot

    z_dot = radius_dot * math.sin(eps) * math.sin(elong) + \
                radius * math.sin(eps) * math.cos(elong) * elong_dot

    velocity[0] = -x_dot * KM_AU / SEC_DAY
    velocity[1] = -y_dot * KM_AU / SEC_DAY
    velocity[2] = -z_dot * KM_AU / SEC_DAY

    dot_product = velocity[0] * target[0] + \
                  velocity[1] * target[1] + \
                  velocity[2] * target[2]
    radvel = -dot_product

    return radvel

def mod2pi(x):
    """Return the argument modulo two pi.

    Parameters
    ----------
    x: float
        An angle in radians.

    Returns
    -------
    float
        x modulo 2 * pi.
    """

    (f, i) = math.modf(x / (2.*math.pi))
    if f < 0.:
        f += 1.
    return f * 2. * math.pi

def doFlatcorr(events, info, switches, reffiles, phdr, hdr):
    """Apply flat field correction.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.

    hdr: astropy.io.fits Header object
        The events extension header.
    """

    cosutil.printSwitch("FLATCORR", switches)

    if switches["flatcorr"] == "PERFORM":

        cosutil.printRef("FLATFILE", reffiles)

        fd = fits.open(reffiles["flatfile"], mode="copyonwrite")

        if info["detector"] == "NUV":
            hdu = fd[1]
        else:
            pharange = cosutil.getPulseHeightRange(hdr, info["segment"])
            # xxx this is temporary; eventually select image based on pharange
            ref_pharange = cosutil.tempPulseHeightRange(reffiles["flatfile"])
            cosutil.comparePulseHeightRanges(pharange, ref_pharange,
                                             reffiles["flatfile"])
            hdu = fd[(info["segment"],1)]
        flat = hdu.data

        origin_x = hdu.header.get("origin_x", 0)
        origin_y = hdu.header.get("origin_y", 0)

        if info["obsmode"] == "ACCUM":
            if info["obstype"] == "SPECTROSCOPIC":
                cosutil.printSwitch("DOPPCORR", switches)
            if switches["doppcorr"] == "PERFORM" or \
               switches["doppcorr"] == "COMPLETE":
                convolveFlat(flat, info["dispaxis"], \
                     info["expstart"], info["orig_exptime"],
                     info["dopmagt"], info["dopzerot"], info["orbtpert"])
                phdr["doppcorr"] = "COMPLETE"

        ccos.applyflat(events.field(xcorr), events.field(ycorr),
                       events.field("epsilon"), flat, origin_x, origin_y)

        fd.close()

        phdr["flatcorr"] = "COMPLETE"

def convolveFlat(flat, dispaxis,
                 expstart, exptime, dopmagt, dopzerot, orbtpert):
    """Convolve the flat field file with the Doppler smearing function.

    Parameters
    ----------
    flat: array like
        Flat field data array, modified in-place.

    dispaxis: {1, 2}
        Dispersion axis (value of header keyword DISPAXIS); 1 is the more
        rapidly varying axis (x), 2 is the less rapidly varying axis (y).

    expstart: float
        Exposure start time, MJD.

    exptime: float
        Exposure duration, seconds; this is the original value,
        not the one corrected for bursts and bad time intervals.

    dopmagt: int
        Magnitude of Doppler shift, pixels.

    dopzerot: float
        Time when Doppler shift is zero and increasing.

    orbtpert: float
        orbital period of HST.
    """

    # Round dopmagt up to the next integer; mag is a zero-point offset.
    mag = int(math.ceil(dopmagt))

    # dopp will be the Doppler smoothing function, normalized so its sum is 1.
    dopp = np.zeros(2*mag+1, dtype=np.float32)

    # t is the time in seconds since dopzerot, in one second increments.
    xpts = round(exptime)
    xpts = max(xpts, 1.)
    npts = int(xpts)
    t = np.arange(npts, dtype=np.float32) + \
                (expstart - dopzerot) * SEC_PER_DAY

    # shift is in pixels (wavelengths increase toward larger pixel number).
    shift = -dopmagt * np.sin(2. * np.pi * t / orbtpert)

    # Construct the Doppler smoothing function.
    increment = 1. / xpts
    for i in range(npts):                       # one-second increments
        ishift = int(round(shift[i])) + mag
        assert ishift >= 0 and ishift <= 2*mag
        dopp[ishift] += increment

    # Do the convolution (in-place).
    axis = 2 - dispaxis         # 1 --> 1,  2 --> 0
    ccos.convolve1d(flat, dopp, axis)

def doDeadcorr(events, input, info, switches, reffiles, phdr, hdr,
            stim_countrate, stim_livetime, livetimefile):
    """Correct for deadtime.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    input: str
        Name of raw file (for writing to livetimefile).

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The input primary header.

    hdr: astropy.io.fits Header object
        The input extension header.

    livetimefile: str
        Name of output text file for livetime factors (or None).

    stim_countrate: float
        The observed count rate for a stim (for info).

    stim_livetime: float
        Live time computed from the stim rate.
    """

    cosutil.printSwitch("DEADCORR", switches)
    if switches["deadcorr"] == "PERFORM":
        cosutil.printRef("DEADTAB", reffiles)
        if info["obsmode"] == "TIME-TAG":
            (dead_rate, dead_method, avg_livetime) = \
                deadtimeCorrection(events, reffiles["deadtab"], info,
                                   stim_countrate, stim_livetime,
                                   input, livetimefile)
        else:
            (dead_rate, dead_method, avg_livetime) = \
                deadtimeCorrectionAccum(events, reffiles["deadtab"], info,
                                        stim_countrate, stim_livetime,
                                        input, livetimefile)
        updateDeadtimeKeywords(hdr, info["segment"],
                               dead_rate, dead_method, avg_livetime)
        phdr["deadcorr"] = "COMPLETE"

def updateDeadtimeKeywords(hdr, segment,
                           dead_rate, dead_method, avg_livetime):
    """Assign values to keywords pertaining to the deadtime correction.

    Parameters
    ----------
    hdr: astropy.io.fits Header object
        The first extension header, updated in-place.

    segment: str
        FUVA, FUVB, or N/A for NUV.

    dead_rate: float
        The count rate that was used for determining the livetime factor.

    dead_method: str
        A string that indicates which method was used for determining the
        livetime factor.

    avg_livetime: float
        The livetime factor that was applied to the data, or the average
        of the factors if the actual count rate was used for time-tag data.
    """

    if segment == "FUVA":
        hdr["deadrt_a"] = dead_rate
        hdr["deadmt_a"] = dead_method
        hdr["livetm_a"] = avg_livetime
    elif segment == "FUVB":
        hdr["deadrt_b"] = dead_rate
        hdr["deadmt_b"] = dead_method
        hdr["livetm_b"] = avg_livetime
    else:
        hdr["deadrt"] = dead_rate
        hdr["deadmt"] = dead_method
        hdr["livetm"] = avg_livetime

def deadtimeCorrection(events, deadtab, info,
                       stim_countrate, stim_livetime,
                       input, livetimefile):
    """Compute and apply livetime factor to correct for dead time.

    Calculate one livetime factor from the count rate averaged over
    the whole exposure.
    Calculate another livetime factor from keyword deventa, deventb
    or mevents.
    if there are subarrays:
        if the difference between the two livetime factors is > 10%:
            Use the livetime factor based on the keyword (the same
            factor for all events).
        else:
            Calculate and apply the livetime factor based on the
            actual count rate within each TIMESTEP time interval.
    else no subarrays:
        if the difference between the two livetime factors is > 10%:
            Print a warning.
        Calculate and apply the livetime factor based on the actual
        count rate within each TIMESTEP time interval.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    deadtab: str
        Name of reference table of count rates and livetime factors.

    info: dictionary
        Header keywords and values.

    stim_countrate: float
        The observed count rate for the stims.

    stim_livetime: float
        Livetime computed from the stims.

    input: str
        Name of input raw file (for writing to livetimefile).

    livetimefile: str
        Name of output text file for livetime factors (or None).

    Returns
    -------
    tuple, (dead_rate, dead_method, avg_livetime)
        The count rate used for determining the livetime factor, a
        string that indicates which method was used for determining the
        livetime factor, and the average livetime factor that was used.
    """

    if livetimefile is None:
        fd = None
    else:
        fd = open(livetimefile, "a")

    # dec_countrate is the count rate from the digital event counter.
    segment = info["segment"]
    dec_countrate = info["countrate"]

    time = events.field("TIME").astype(np.float64)
    epsilon = events.field("epsilon")
    nevents = len(time)

    live_info = cosutil.getTable(deadtab, filter={"segment": segment},
                                 at_least_one=True)
    # These are the values in the deadtab table columns.
    obs_rate = live_info.field("obs_rate")
    live_factor = live_info.field("livetime")

    # This livetime value is based on count rate over the entire exposure.
    if time[nevents-1] > time[0]:
        actual_countrate = float(nevents) / (time[nevents-1] - time[0])
    else:
        actual_countrate = 0.
    actual_rate_livetime = cosutil.determineLivetime(actual_countrate,
                                                     obs_rate, live_factor)

    # dec_countrate is from DEVENTA, DEVENTB or from MEVENTS.
    dec_livetime = cosutil.determineLivetime(dec_countrate,
                                             obs_rate, live_factor)

    print_details =(cosutil.checkVerbosity(VERY_VERBOSE))       # initial value
    if abs(dec_livetime - actual_rate_livetime) > \
           LIVETIME_CRITERION * actual_rate_livetime:
        cosutil.printWarning("livetime estimates differ.")
        if info["subarray"] and info["nsubarry"] > 0:   # are there subarrays?
            use_actual_rate = False
        else:
            use_actual_rate = True
        print_details = True
    else:
        use_actual_rate = True

    if use_actual_rate:
        livetime_source = "actual count rate"
        dead_rate = actual_countrate
        dead_method = "DATA"
    else:
        dead_rate = dec_countrate
        if info["detector"] == "FUV":
            dead_method = "DEVENT"
            keyword = "DEVENT" + info["segment"][-1]
        else:
            dead_method = "MEVENTS"
            keyword = "MEVENTS"
        livetime_source = "digital event counter (%s)" % keyword

    if print_details:
        printLiveInfo(segment, stim_countrate, stim_livetime,
                      actual_countrate, actual_rate_livetime,
                      dec_countrate, dec_livetime, livetime_source)
    if fd is not None:
        printLiveInfo(segment, stim_countrate, stim_livetime,
                      actual_countrate, actual_rate_livetime,
                      dec_countrate, dec_livetime, livetime_source, fd=fd)

    if use_actual_rate:

        if fd is not None:
            fd.write("# %s\n" % input)
            fd.write("# t0 t1 countrate livetime\n")

        # Use counts over dt_deadtime seconds to compute livetime.
        fd_dead = fits.open(deadtab, mode="readonly", memmap=False)
        dt_deadtime = fd_dead[1].header["timestep"]
        fd_dead.close()
        cosutil.printMsg("Compute livetime factor; timestep is %.6g s:" \
                      % dt_deadtime, VERY_VERBOSE)

        t0 = time[0]
        t1 = t0 + dt_deadtime
        last_time = time[nevents-1]
        cosutil.printMsg("  time range    rate   livetime", VERY_VERBOSE)
        last_livetime = 1.      # use this for saving previous value
        sum_livetime = 0.       # for computing average livetime factor
        wgt_livetime = 0.
        countrate = 0.
        first = True
        while t0 < last_time:

            # time[i:j] matches t0 to t1.
            try:
                (i, j) = ccos.range(time, t0, t1)
            except:
                t0 = t1
                t1 = t0 + dt_deadtime
                continue
            t1_for_printing = t1        # may be changed below
            if i >= j:          # i and j can be equal due to roundoff
                t0 = t1
                t1 = t0 + dt_deadtime
                continue

            if t1 < last_time:
                countrate = (j - i) / dt_deadtime
                livetime = cosutil.determineLivetime(countrate,
                                                     obs_rate, live_factor)
                sum_livetime += livetime * (t1 - t0)
                wgt_livetime += (t1 - t0)
            elif t0 < last_time:
                t1_for_printing = last_time
                if (last_time - t0) < 0.5 * dt_deadtime and not first:
                    livetime = last_livetime
                    cosutil.printMsg("Last time interval is short (%.6g s),"
                                     " so previous livetime will be used." %
                                     (last_time - t0,))
                else:
                    countrate = (j - i) / (last_time - t0)
                    livetime = cosutil.determineLivetime(countrate,
                                                         obs_rate, live_factor)
                sum_livetime += livetime * (last_time - t0)
                wgt_livetime += (last_time - t0)
            else:
                countrate = 0.
                livetime = 1.
            if livetime > 0.:
                epsilon[i:j] = epsilon[i:j] / livetime
                last_livetime = livetime
            first = False

            if fd is not None:
                fd.write("%.0f %.0f %.6g %.6g\n" %
                         (t0, t1_for_printing, countrate, livetime))
            cosutil.printMsg("%6.1f %6.1f   %.6g %.6g" %
                             (t0, t1_for_printing, countrate, livetime),
                             VERY_VERBOSE)

            t0 = t1
            t1 = t0 + dt_deadtime

        if wgt_livetime > 0.:
            avg_livetime = sum_livetime / wgt_livetime
        else:
            avg_livetime = 1.
    else:
        epsilon[:] = epsilon / dec_livetime
        avg_livetime = dec_livetime

    if fd is not None:
        fd.close()

    return (dead_rate, dead_method, avg_livetime)

def deadtimeCorrectionAccum(events, deadtab, info,
                            stim_countrate, stim_livetime,
                            input, livetimefile):
    """Determine and apply the livetime factor for ACCUM data.

    If there are subarrays, the livetime factor is gotten from the digital
    event counter.  If there are no subarrays, the livetime factor is based
    on the actual count rate.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    deadtab: str
        Name of reference table of count rates and livetime factors.

    info: dictionary
        Header keywords and values.

    stim_countrate: float
        The observed count rate for the stims.

    stim_livetime: float
        Livetime computed from the stims.

    input: str
        Name of input raw file (for writing to livetimefile).

    livetimefile: str
        Name of output text file for livetime factors (or None).

    Returns
    -------
    tuple, (dead_rate, dead_method, livetime)
        The count rate used for determining the livetime factor, a
        string that indicates which method was used for determining the
        livetime factor, and the livetime factor that was used.
    """

    if livetimefile is None:
        fd = None
    else:
        fd = open(livetimefile, "a")
        fd.write("# %s\n" % (input,))

    # This is the column that will be modified in-place.
    epsilon = events.field("epsilon")
    ncounts = len(epsilon)

    live_info = cosutil.getTable(deadtab, filter={"segment": info["segment"]},
                                 at_least_one=True)
    obs_rate = live_info.field("obs_rate")
    live_factor = live_info.field("livetime")

    # keyword used if print_details is true or we're writing to a livetimefile.
    if info["segment"] == "FUVA":
        keyword = "DEVENTA"
    elif info["segment"] == "FUVB":
        keyword = "DEVENTB"
    else:
        keyword = "MEVENTS"

    # Output count rate from digital event counter (DEC), and corresponding
    # livetime factor.
    dec_countrate = info["countrate"]
    dec_livetime = cosutil.determineLivetime(dec_countrate,
                                             obs_rate, live_factor)

    if info["orig_exptime"] <= 0.:
        cosutil.printWarning("Can't do deadcorr, exptime = %.6g." %
                             info["orig_exptime"])
        return (0., "SKIPPED", 1.)
    actual_countrate = float(ncounts) / info["orig_exptime"]
    actual_rate_livetime = cosutil.determineLivetime(actual_countrate,
                                                     obs_rate, live_factor)

    if info["subarray"]:
        livetime_source = "digital event counter (%s)" % keyword
        livetime = dec_livetime
        dead_rate = dec_countrate
        if info["detector"] == "FUV":
            dead_method = "DEVENT"
        else:
            dead_method = "MEVENTS"
    else:
        livetime_source = "actual count rate"
        livetime = actual_rate_livetime
        dead_rate = actual_countrate
        dead_method = "DATA"

    epsilon[:] = epsilon / livetime

    print_details = cosutil.checkVerbosity(VERY_VERBOSE)        # initial value

    if abs(dec_livetime - actual_rate_livetime) > \
           LIVETIME_CRITERION * actual_rate_livetime:
        cosutil.printWarning("livetime estimates differ.")
        print_details = True

    if print_details:
        cosutil.printMsg("  actual countrate and livetime:  %.6g, %6.4f" % \
                         (actual_countrate, actual_rate_livetime))
        cosutil.printMsg("  countrate and livetime from %s:  %.6g, %6.4f" % \
                         (keyword, dec_countrate, dec_livetime))
        cosutil.printMsg("Livetime %6.4f is based on %s." % \
                         (livetime, livetime_source))
        if info["detector"] == "FUV":
            if stim_countrate is None:
                cosutil.printMsg(
                "  stim countrate and livetime could not be determined")
            else:
                cosutil.printMsg(
                "  stim countrate and livetime:  %.6g, %6.4f" % \
                                  (stim_countrate, stim_livetime))

    if fd is not None:
        fd.write("actual countrate and livetime:  %.6g, %6.4f\n" %
                 (actual_countrate, actual_rate_livetime))
        fd.write("countrate and livetime from %s:  %.6g, %6.4f\n" %
                 (keyword, dec_countrate, dec_livetime))
        fd.write("livetime %6.4f is based on %s.\n" % \
                 (livetime, livetime_source))
        if info["detector"] == "FUV":
            if stim_countrate is None:
                fd.write(
                "stim countrate and livetime could not be determined\n")
            else:
                fd.write("stim countrate and livetime:  %.6g, %6.4f\n" %
                         (stim_countrate, stim_livetime))

    if fd is not None:
        fd.close()

    return (dead_rate, dead_method, livetime)

def printLiveInfo(segment, stim_countrate, stim_livetime,
                  actual_countrate, actual_rate_livetime,
                  dec_countrate, dec_livetime, livetime_source, fd=None):
    """Print or write information about livetime.

    Parameters
    ----------
    segment: str
        Segment name (for setting keyword name for DEC count rate).

    stim_countrate: float
        The observed count rate for the stims, or None.

    stim_livetime: float
        Livetime factor computed from the input and observed stim rate.

    actual_countrate: float
        Observed count rate, from events table.

    actual_rate_livetime: float
        Livetime factor derived from countrate.

    dec_countrate: float
        The count rate from the digital event counter.

    dec_livetime: float
        Livetime factor computed from dec_countrate.

    livetime_source: str
        A string saying whether actual rate or DEC was used for computing
        the livetime factor.

    fd: file
        None if printing to trailer; an fd for printing to a log file.
    """

    if segment == "FUVA":
        keyword = "DEVENTA"
    elif segment == "FUVB":
        keyword = "DEVENTB"
    else:
        keyword = "MEVENTS"

    messages = []
    if segment == "FUVA" or segment == "FUVB":
        if stim_countrate is None:
            messages.append(
                "stim countrate and livetime could not be determined")
        else:
            messages.append("stim countrate and livetime:  %.6g, %6.4f" %
                            (stim_countrate, stim_livetime))
    messages.append("actual (average) event rate and livetime:  %.6g, %6.4f" %
                    (actual_countrate, actual_rate_livetime))
    messages.append("countrate and livetime from %s:  %.6g, %6.4f" %
                    (keyword, dec_countrate, dec_livetime))
    messages.append("Livetime is based on %s." % livetime_source)

    if fd is None:
        for msg in messages:
            cosutil.printMsg(msg)
    else:
        fd.write("\n")
        for msg in messages:
            fd.write(msg + "\n")

def writeNull(input, ofd, output, outcounts, outcsum,
              cl_args, info, phdr, headers):
    """Write output files; images will have null data portions.

    The outtag file has already been written, so we only need to write
    the output and outcounts files.

    Parameters
    ----------
    input: str
        Name of the input file.

    ofd: astropy.io.fits HDUList object
        Output file header/data list.

    output: str
        Name of the output file for flat-fielded count-rate image.

    outcounts: str
        Name of the output file for count-rate image.

    outcsum: str
        Name of the output image file for OPUS to add to cumulative image,
        or None.

    cl_args: dictionary
        Some of the command-line arguments.

    info: dictionary
        Header keywords and values.

    phdr: astropy.io.fits Header object
        Primary header.

    headers: list of astropy.io.fits Header objects
        Headers.
    """

    cosutil.printWarning("No data in " + input)
    makeImage(outcounts, phdr, headers, None, None, None)
    makeImage(output, phdr, headers, None, None, None)
    tl_time = cosutil.timelineTimes(None, None)
    timeline.createTimeline(input, ofd, info, reffiles={},
                            tl_time=tl_time, shift1_vs_time=None,
                            time=None, xfull=None, yfull=None)
    if outcsum is not None:
        writeCsum(outcsum, None,
                  info["detector"], info["obsmode"],
                  phdr, headers[1],
                  cl_args["raw_csum_coords"],
                  cl_args["binx"], cl_args["biny"],
                  cl_args["compress_csum"],
                  cl_args["compression_parameters"])

def createTraceMask(events, info, switches, xtractab, active_area):
    """Create a mask for events that will be corrected.  This is events within
    the Active Area, but not including the events in the tagflash region"""
    #
    # Only create the tracemask if we are going to do either the trace
    # correction or the profile alignment correction
    if switches["trcecorr"] == 'PERFORM' or switches["algncorr"] == 'PERFORM':
        filter = {"segment": info["segment"],
                  "opt_elem": info["opt_elem"],
                  "cenwave": info["cenwave"],
                  "aperture": "WCA"}
        mask = active_area.copy()
        xtract_info = cosutil.getTable(xtractab, filter)
        slope = xtract_info['SLOPE'][0]
        intercept = xtract_info['B_SPEC'][0]
        height = xtract_info['HEIGHT'][0]
        ncols = FUV_X
        xfull = events.field('xfull')
        center = intercept + slope*xfull
        rowstart = center - height/2
        rowstop = center + height/2
        yfull = events.field('yfull')
        inside = (yfull - rowstart)*(yfull - rowstop)
        inside_WSA = np.where(inside < 0.0)
        mask[inside_WSA] = False
    else:
        mask = None
    return mask

def doTraceCorr(events, info, switches, reffiles, phdr, tracemask):
    """Do the trace correction.  The trace reference file follows the
    centroid of a point source.  Applying the correction involves subtracting
    the profile of trace vs. xcorr from yfull
    Returns the 1-d array of the trace correction"""
    #
    # If the TRCECORR step is omitted, return None
    result = None
    if switches["trcecorr"] != 'N/A':
        cosutil.printSwitch("TRCECORR", switches)
        if switches["trcecorr"] == "PERFORM":
            cosutil.printRef("TRACETAB", reffiles)
            tracetab = reffiles["tracetab"]
            f1 = fits.open(tracetab)
            ref_life_adj = f1[0].header['LIFE_ADJ']
            if ref_life_adj != info["life_adj"]:
                cosutil.printWarning("LIFE_ADJ values are different for data and TRACETAB")
                cosutil.printWarning("LIFE_ADJ = %d in data" % (info["life_adj"]))
                cosutil.printWarning("LIFE_ADJ = %d in TRACETAB" % (ref_life_adj))
                cosutil.printWarning("If you MUST use this reference file, change one of")
                cosutil.printWarning("the LIFE_ADJ values so they agree")
                raise Exception("Invalid Reference file")
            result = trace.doTrace(events, info, reffiles, tracemask)
            phdr["TRCECORR"] = "COMPLETE"
    return result

def doProfileAlignmentCorr(events, input, info, switches, reffiles, phdr, hdr,
                           minmax_shift_dict, tracemask, traceprofile, gti):
    """Do the profile alignment correction.  This is usually combined with the
    trace correction.  It involves calculating the flux-weighted centroid of a
    reference profile, the flux-weighted centroid of the science data, and
    applying the difference to the YFULL values so that the data and reference
    profile should have the same centroid."""
    if info["detector"] == "FUV":
        cosutil.printSwitch("ALGNCORR", switches)
        if switches["algncorr"] == "PERFORM":
            cosutil.printRef("PROFTAB", reffiles)
            #
            # Need to compute a temporary copy of the DQ array so that the
            # profile alignment step can use it
            dq_array = doDqicorr(events, input, info, switches, reffiles,
                                 phdr, hdr, minmax_shift_dict,
                                 traceprofile, gti)

            result = trace.doProfileAlignment(events, input, info, switches,
                                              reffiles, phdr, hdr, dq_array,
                                              tracemask)
            if result == trace.CENTROID_SET_BY_USER:
                phdr["ALGNCORR"] = "USER-SUPPLIED"
            elif result == trace.CENTROID_OK:
                phdr["ALGNCORR"] = "COMPLETE"
            elif result == trace.NO_CONVERGENCE:
                phdr["ALGNCORR"] = "SKIPPED"
            elif result == trace.CENTROID_ERROR_TOO_LARGE:
                phdr["ALGNCORR"] = "SKIPPED"
        return

def writeImages(x, y, epsilon, dq,
                phdr, headers, dq_array, npix, x_offset, exptime,
                outcounts=None, output=None):
    """Bin events to images, and write to output files.

    Parameters
    ----------
    x: array like
        X pixel coordinates of events.

    y: array like
        Y pixel coordinates of events.

    epsilon: array like
        Weight column.

    dq: array like
        Data quality column.

    phdr: astropy.io.fits Header object
        The input primary header.

    headers: list of astropy.io.fits Header objects
        The input headers.

    dq_array: array like
        The data quality array.

    npix: tuple
        The array shape (ny, nx).

    x_offset: int
        Offset of the detector in a calibrated image.

    exptime: float
        The exposure time.

    outcounts: str
        Name of the output file for count-rate image.

    output: str
        Name of the output file for flat-fielded count-rate image.
    """

    # notation:
    # t = exposure time (exptime)
    # C_counts = sum of counts
    # E_counts = "effective" counts, sum of EPSILON column
    # C_rate = C_counts / t
    # E_rate = E_counts / t
    # reciprocal_flat_field = E_counts / C_counts
    # the corresponding error arrays are:
    # errC_rate = errGehrels(C_counts) / t
    # errE_rate = errC_rate * reciprocal_flat_field

    if outcounts is not None:
        cosutil.printMsg("writing file %s ..." % outcounts, VERY_VERBOSE)

    # First make an image array in which each input event counts as one,
    # i.e. ignoring flat field and deadtime corrections.
    C_counts = np.zeros(npix, dtype=np.float32)

    if exptime <= 0:
        cosutil.printWarning(
                "Exposure time is zero, so output files are dummy.")
        if outcounts is not None:
            makeImage(outcounts, phdr, headers, C_counts, C_counts, dq_array)
        if output is not None:
            makeImage(output, phdr, headers, C_counts, C_counts, dq_array)
        return

    ccos.binevents(x, y, C_counts, x_offset, dq, SERIOUS_DQ_FLAGS)

    # Use the Frequentist variance function.
    err_lower, err_upper = cosutil.errFrequentist(C_counts)
    errC_rate = err_upper / exptime

    if outcounts is not None:
        C_rate = C_counts / exptime
        makeImage(outcounts, phdr, headers, C_rate, errC_rate, dq_array)
    del C_rate

    if output is None:
        return                          # nothing further to do

    cosutil.printMsg("writing file %s ..." % output, VERY_VERBOSE)

    # Make an image array where event number i has weight epsilon[i].
    E_counts = np.zeros(npix, dtype=np.float32)
    ccos.binevents(x, y, E_counts, x_offset, dq, SERIOUS_DQ_FLAGS, epsilon)

    E_rate = E_counts / exptime

    reciprocal_flat = np.where(E_counts == 0., 1., E_counts) / \
                      np.where(C_counts == 0., 1., C_counts)
    del E_counts, C_counts
    errE_rate = errC_rate * reciprocal_flat
    del reciprocal_flat, errC_rate

    makeImage(output, phdr, headers, E_rate, errE_rate, dq_array)

def makeImage(outimage, phdr, headers, sci_array, err_array, dq_array):
    """Write a FITS file, based on headers and data arrays.

    Parameters
    ----------
    output: str
        Name of the output file to be written.

    phdr: astropy.io.fits Header object
        The input primary header.

    headers: list of astropy.io.fits Header objects
        The input headers.

    sci_array: array like
        The science data array (may be None).

    err_array: array like
        The error estimates array (may be None).

    dq_array: array like
        The data quality array (may be None).
    """

    primary_hdu = fits.PrimaryHDU(header=phdr)
    fd = fits.HDUList(primary_hdu)
    fd[0].header["nextend"] = 3
    cosutil.updateFilename(fd[0].header, outimage)

    makeImageHDU(fd, headers[1], sci_array, name="SCI")
    makeImageHDU(fd, headers[2], err_array, name="ERR")
    makeImageHDU(fd, headers[3], dq_array, name="DQ")

    fd.writeto(outimage, output_verify='silentfix')

def makeImageHDU(fd, table_hdr, data_array, name="SCI"):
    """Make an image hdu from data and a table header and append to fd.

    Parameters
    ----------
    fd: astropy.io.fits HDUList object
        astropy.io.fits object for FITS file (new hdu will be appended).

    table_hdr: astropy.io.fits Header object
        Header for the input table.

    data_array: array like
        Image data to be appended (may be None).

    name: str
        Name to be used for EXTNAME.
    """

    # Create an image header from the table header.
    imhdr = cosutil.tableHeaderToImage(table_hdr)
    if name == "DQ":
        imhdr["BUNIT"] = "UNITLESS"
    else:
        imhdr["BUNIT"] = "count /s"

    if data_array is not None:
        if "npix1" in imhdr:
            del(imhdr["npix1"])
        if "npix2" in imhdr:
            del(imhdr["npix2"])
        if "pixvalue" in imhdr:
            del(imhdr["pixvalue"])

    hdu = fits.ImageHDU(data=data_array, header=imhdr, name=name)
    fd.append(hdu)

def writeCsum(outcsum, events,
              detector, obsmode,
              phdr, hdr,
              raw_csum_coords,
              binx=None, biny=None,
              compress_csum=False,
              compression_parameters="gzip,-0.1"):
    """Write the "calcos sum" (csum) image.

    Parameters
    ----------
    outcsum: str
        Name of output "calcos sum" file.

    events: astropy.io.fits record array
        The data unit containing the events table.

    detector: {"FUV", "NUV"}
        Detector name.

    obsmode: str
        TIME-TAG or ACCUM, used for determining whether to write a third
        dimension with PHA for FUV data.

    phdr: astropy.io.fits Header object
        Primary header from input file.

    hdr: astropy.io.fits Header object
        First extension (EVENTS) header from input file.

    raw_csum_coords: boolean
        Use raw pixel coordinates?

    binx: int
        Binning factor in the dispersion direction (or None for the default
        binning).

    biny: int
        Binning factor in the cross-dispersion direction (or None for the
        default binning).

    compress_csum: boolean
        Compress the csum image?

    compression_parameters: str
        compressionType and quantizeLevel (separated by a comma) for the
        call to astropy.io.fits.CompImageHDU; compressionType can be "rice", "gzip",
        or "hcompress", and quantizeLevel can be e.g. -0.1, which means the
        floating point values will be scaled to integers with spacing that
        corresponds to 0.1 dn (see the doc string for fits.CompImageHDU
        for more details).
    """

    # This is the number of possible values for the pulse height amplitude,
    # pha = 0..31.
    PHA_RANGE = 32

    cosutil.printMsg("writing file %s ..." % outcsum, VERY_VERBOSE)

    primary_hdu = fits.PrimaryHDU(header=phdr)
    fd = fits.HDUList(primary_hdu)
    fd[0].header["nextend"] = 1
    fd[0].header["filetype"] = "CALCOS SUM FILE"
    cosutil.updateFilename(fd[0].header, outcsum)

    if events is None or len(events) == 0:
        xcoord = None
        ycoord = None
        epsilon = None
        pha = None
        if raw_csum_coords:
            fd[0].header["coordfrm"] = "raw"
        else:
            fd[0].header["coordfrm"] = "corrected"
    else:
        if raw_csum_coords:
            xcoord = events.field("rawx")
            ycoord = events.field("rawy")
            flagOmit(fd[0].header)      # set some cal. switches to OMIT
            fd[0].header["coordfrm"] = "raw"
        else:
            xcoord = events.field(xcorr)
            ycoord = events.field(ycorr)
            fd[0].header["coordfrm"] = "corrected"
        epsilon = events.field("epsilon")
        if detector == "FUV" and obsmode == "TIME-TAG":
            pha = events.field("pha")
        else:
            pha = None

    if detector == "FUV":
        if binx is None or binx <= 0:
            binx = FUV_BIN_X
        if biny is None or biny <= 0:
            biny = FUV_BIN_Y
        nx = FUV_X // binx
        ny = FUV_Y // biny
    else:
        if binx is None or binx <= 0:
            binx = NUV_BIN_X
        if biny is None or biny <= 0:
            biny = NUV_BIN_Y
        nx = NUV_X // binx
        ny = NUV_Y // biny

    if compress_csum:
        (compType, quantLevel) = compression_parameters.split(",")
        compType = compType.upper() + "_1"
        quantLevel = float(quantLevel)
        if detector == "FUV":
            if obsmode == "ACCUM":
                data = np.zeros((ny, nx), dtype=np.float32)
                if xcoord is not None:
                    ccos.csum_2d(data, xcoord, ycoord, epsilon, binx, biny)
            else:
                data = np.zeros((PHA_RANGE, ny, nx), dtype=np.float32)
                if xcoord is not None:
                    ccos.csum_3d(data, xcoord, ycoord, epsilon,
                                 pha.astype(np.int16), binx, biny)
        else:
            data = np.zeros((ny, nx), dtype=np.float32)
            if xcoord is not None:
                ccos.csum_2d(data, xcoord, ycoord, epsilon, binx, biny)
        fd.append(fits.CompImageHDU(data, header=hdr, name="SCI",
                                    compressionType=compType,
                                    quantizeLevel=quantLevel))
        fd[1].header["counts"] = data.sum(dtype=np.float64)
        #  the arguments to CompImageHDU and their defaults are:
        # compressionType='RICE_1', 'PLIO_1', 'GZIP_1', 'HCOMPRESS_1'
        # tileSize=None,       # shape of tile, default is one row
        # hcompScale=0.,       # unit is RMS of image tile
        # hcompSmooth=0,
        # if quantizeLevel is positive, unit is RMS of image tile; if negative,
        # the quantization level is the absolute value of quantizeLevel
        # quantizeLevel=16.
        del data
    else:
        if detector == "FUV":
            if obsmode == "ACCUM":
                fd.append(fits.ImageHDU(data=np.zeros((ny, nx),
                                                      dtype=np.float32),
                                        header=hdr, name="SCI"))
                if xcoord is not None:
                    ccos.csum_2d(fd[1].data, xcoord, ycoord, epsilon,
                                 binx, biny)
            else:
                fd.append(fits.ImageHDU(data=np.zeros((PHA_RANGE, ny, nx),
                                                      dtype=np.float32),
                                        header=hdr, name="SCI"))
                if xcoord is not None:
                    ccos.csum_3d(fd[1].data, xcoord, ycoord, epsilon,
                                 pha.astype(np.int16), binx, biny)
        else:
            fd.append(fits.ImageHDU(data=np.zeros((ny, nx),
                                                  dtype=np.float32),
                                    header=hdr, name="SCI"))
            if xcoord is not None:
                ccos.csum_2d(fd[1].data, xcoord, ycoord, epsilon, binx, biny)
        fd[1].header["counts"] = fd[1].data.sum(dtype=np.float64)

    if detector == "FUV":
        fd[1].header["fuvbinx"] = binx
        fd[1].header["fuvbiny"] = biny
    else:
        fd[1].header["nuvbinx"] = binx
        fd[1].header["nuvbiny"] = biny

    fd[1].header["BUNIT"] = "count"

    fd.writeto(outcsum, output_verify="silentfix")
    fd.close()

def flagOmit(phdr):
    """Flag certain calibration switches as OMIT.

    This is called for the primary header of the csum file, for the case
    that the csum image will be constructed from raw pixel coordinates.
    These steps may have actually been done, but they don't affect the
    raw coordinates, so it would be misleading to have their values set
    to "COMPLETE" in the csum header.

    Parameters
    ----------
    phdr: astropy.io.fits Header object
        The primary header of the csum file, modified in-place to set some
        calibration switches to OMIT.
    """

    keys = list(phdr.keys())
    omit_switches = ["BRSTCORR",
                     "BADTCORR",
                     "RANDCORR",
                     "TEMPCORR",
                     "GEOCORR",
                     "IGEOCORR",
                     "PHACORR",
                     "DOPPCORR",
                     "HELCORR",
                     "DEADCORR"]
    for key in omit_switches:
        if key in keys:
            phdr[key] = "OMIT"

def doStatflag(switches, output, outcounts):
    """Compute statistics and update keywords.

    Parameters
    ----------
    switches: dictionary
        Calibration switches.

    outflt: str
        Name of the output file for flat-fielded count-rate image.

    outcounts: str
        Name of the output file for count-rate image.
    """

    cosutil.printSwitch("STATFLAG", switches)
    if switches["statflag"] == "PERFORM":
        cosutil.doImageStat(outcounts)
        cosutil.doImageStat(output)

def flag_gti(time, dq, gti):
    """Flag events in dq that are outside any good time interval.

    Parameters
    ----------
    time: array like
        The time column in the events table.

    dq: array like
        The data quality column in the events table (updated in-place).

    gti: list
        List of good time intervals.
    """

    SMALL_INCR = 0.02           # smaller than the timestep of 0.032 s

    if len(gti) < 1 or len(time) < 1:
        return

    # Nothing to do if there is only one GTI and it covers the entire
    # time range.
    if len(gti) == 1 and \
      (time[0] >= gti[0][0] and time[-1] <= gti[0][1]):
        return

    dq[:] |= DQ_BAD_TIME

    for (t_start, t_stop) in gti:
        (i0, i1) = ccos.range(time, t_start, t_stop+SMALL_INCR)
        dq[i0:i1] &= ~DQ_BAD_TIME

def noWavecal(input, shift_file, info, switches, reffiles):
    """Assign a default value for the wavecal shift, from fp_pixel_shift.

    Parameters
    ----------
    input: str
        Name of the input file.

    shift_file: str
        Optional, user-specified values of shift1.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    Returns
    -------
    (wavecal_info, wavecorr)
    wavecal_info is a list of dictionaries.
    wavecorr is the value to assign to keyword WAVECORR.
    """

    # For the present purposes, we don't need the other key values for
    # shift_dict.  Some of these shift1 values may be assigned below.
    shift_dict = {"shift1a": 0., "shift1b": 0., "shift1c": 0.,
                  "shift2a": 0., "shift2b": 0., "shift2c": 0.}

    if info["detector"] == "FUV":
        segment_list = [info["segment"]]
    else:
        segment_list = ["NUVA", "NUVB", "NUVC"]

    user_specified = False              # initial value
    if shift_file is not None:
        user_shifts = shiftfile.ShiftFile(shift_file,
                                          info["root"], info["fpoffset"])
    else:
        user_shifts = None
    fp_dict = {}                        # this is mostly a dummy dictionary
    lamptab = reffiles["lamptab"]
    got_pixel_shift = cosutil.findColumn(lamptab, "fp_pixel_shift")
    for segment in segment_list:
        # get fp_pixel_shift for shift1
        filter_lamp = {"opt_elem": info["opt_elem"],
                       "cenwave": info["cenwave"],
                       "segment": segment}
        if got_pixel_shift:
            filter_lamp["fpoffset"] = info["fpoffset"]
        lamp_info = cosutil.getTable(lamptab, filter_lamp)
        if lamp_info is not None and got_pixel_shift:
            fp_pixel_shift = lamp_info.field("fp_pixel_shift")[0]
        else:
            fp_pixel_shift = 0.
        # This may be replaced below, if the user specified the value.
        shift1 = fp_pixel_shift

        if user_shifts is not None:
            # Check for a user-supplied value for shift1, assuming the
            # user specified the "flash" as the first one (n = 1).
            ((user_shift1, user_shift2), nfound) = \
                user_shifts.getShifts((1, segment))     # 1 --> first flash
            if user_shift1 is not None:
                user_specified = True
                shift1 = user_shift1
        key = "shift1" + segment[-1].lower()
        shift_dict[key] = shift1
        fp_key = (segment, info["fpoffset"])
        fp_dict[fp_key] = fp_pixel_shift

    wavecal_info = []               # updated by storeWavecalInfo
    wavecal.storeWavecalInfo(wavecal_info,
                             (info["expstart"] + info["expend"]) / 2.,
                             info["cenwave"], info["fpoffset"],
                             shift_dict, fp_dict,
                             info["rootname"], input)

    # True if the user specified the shift for any segment/stripe.
    if user_specified:
        # Not skipped because the user specified the shift.
        wavecorr = "COMPLETE"
    else:
        wavecorr = "SKIPPED"

    return (wavecal_info, wavecorr)


def updateFromWavecal(events, wavecal_info, wavecorr,
                      shift_file,
                      info, switches, reffiles, input_path, phdr, hdr):
    """Update XFULL and YFULL based on auto, simulated or GO wavecal info.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.

    wavecal_info: dictionary
        When wavecal exposures were processed, the results
        were stored in this dictionary.

    wavecorr: str
        Value to assign to WAVECORR keyword in phdr.

    shift_file: str
        If not None, this text file may have been used to override shift1;
        it's included here just to append its name to the WAVECALS string
        for the header and trailer.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: astropy.io.fits Header object
        The primary header (WAVECORR and WAVECALS can be updated).

    hdr: astropy.io.fits Header object
        The events extension header (modified in-place).

    Returns
    -------
    (tl_time, shift1_vs_time): tuple of two array like
        tl_time is the array of times at one-second intervals, for the
        timeline table.  shift1_vs_time is the array of corresponding
        values of shift1a or shift1b, or None if the current observation
        is a wavecal or if wavecal processing was not done.
    """

    global xcorr, ycorr, xdopp, ydopp, xfull, yfull
    global active_area

    # Read info from wavecal parameters table.
    wcp_info = cosutil.getTable(reffiles["wcptab"],
                                filter={"opt_elem": info["opt_elem"]},
                                exactly_one=True)
    wcp_info = wcp_info[0]

    time = events.field("TIME")
    # Create an array of times with one-second increments.
    if info["obsmode"] == "ACCUM":
        first_time = 0.
        last_time = info["exptime"]
    else:
        first_time = time[0]
        last_time = time[-1]
    # The name tl_time means timeline time.
    tl_time = cosutil.timelineTimes(first_time, last_time, dt=1.)

    xi  = events.field(xdopp)
    eta = events.field(ydopp)
    xi_full  = events.field(xfull)
    eta_full = events.field(yfull)

    # If the current exposure is a wavecal, or for a science exposure if
    # wavecal processing has not been done, there's nothing to do.
    if info["exptype"].find("WAVE") >= 0 or not wavecal_info:
        return (tl_time, None)

    # Get the shifts in dispersion and cross-dispersion directions at the
    # start of the exposure.  If the science exposure was bracketed by
    # two wavecals, the slope of the shifts can be non-zero.
    shift_info = wavecal.returnWavecalShift(wavecal_info,
                                            wcp_info, info["cenwave"],
                                            info["fpoffset"], info["expstart"])
    if shift_info is None:
        return (tl_time, None)

    (shift_dict, slope_dict, filename) = shift_info

    if info["detector"] == "FUV":
        segment_list = [info["segment"]]
    else:
        segment_list = ["NUVA", "NUVB", "NUVC"]
        psa_region_flags_dict = nuvPsaRegions(eta, info, reffiles["xtractab"])
        wca_region_flags_dict = nuvWcaRegions(eta, info, reffiles["xtractab"])

    t0 = time[0]
    t_mid = (t0 + time[-1]) / 2.

    xi_full[:] = xi.copy()

    for segment in segment_list:

        key = "shift1" + segment[-1].lower()
        goodshift1key = key
        if not (key in shift_dict and key in slope_dict):
            cosutil.printWarning("There is no wavecal for segment %s." % segment)
            if info['detector'] == "FUV":
                othersegment = "FUVA"
                if segment == "FUVA":
                    othersegment = "FUVB"
                otherkey = "shift1" + othersegment[-1].lower()
                if not (otherkey in shift_dict and otherkey in slope_dict):
                    cosutil.printError("No matching wavecal for segment {} either".format(othersegment))
                    return (tl_time, None)
                cosutil.printMsg("Using shift info for segment {}".format(othersegment))
                shift1_zero = shift_dict[otherkey]
                shift1_slope = slope_dict[otherkey]
                goodshift1key = otherkey
            else:
                return (tl_time, None)
        else:
            shift1_zero = shift_dict[key]
            shift1_slope = slope_dict[key]
        if info["detector"] == "FUV":
            if info['addsimulatedwavecal']:
                simulated_wavecal_info = getSimulatedWavecalInfo(info, key, wavecal_info, wcp_info, input_path)
                transition_time, early_slope, early_intercept, late_slope, late_intercept = simulated_wavecal_info
                early_times = np.where(time < transition_time)
                late_times = np.where(time >= transition_time)
                shift1 = np.zeros(xi_full.shape)
                shift1[early_times] = ((time[early_times] - t0) * early_slope + early_intercept)
                shift1[late_times] = ((time[late_times] - t0) * late_slope + late_intercept)
                xi_full[early_times] = xi[early_times] - shift1[early_times]
                xi_full[late_times] = xi[late_times] - shift1[late_times]
                xi_full[:] = np.where(active_area, xi_full, xi)
            else:
                xi_full[:] = np.where(active_area,
                               xi - ((time - t0) * shift1_slope + shift1_zero),
                               xi_full)
        else:
            xi_full[:] = np.where(psa_region_flags_dict[segment],
                           xi - ((time - t0) * shift1_slope + shift1_zero),
                           xi_full)
            xi_full[:] = np.where(wca_region_flags_dict[segment],
                           xi - ((time - t0) * shift1_slope + shift1_zero),
                           xi_full)
        # Calculate the average shift for the simulated wavecal as the
        # average shift of the events in the active Area
        if info['addsimulatedwavecal']:
            avg_shift1 = np.mean(shift1[np.where(active_area)], dtype=np.float64)
        else:
            avg_shift1 = shift1_slope * t_mid + shift1_zero
        key = "SHIFT1" + segment[-1]
        hdr[key] = round(avg_shift1, 4)

    if info["detector"] == "FUV":
        segment = segment_list[0]
    else:
        segment = "NUVB"

    key = "shift2" + segment[-1].lower()
    if (key in shift_dict and key in slope_dict):
        shift2_zero = shift_dict[key]
        shift2_slope = slope_dict[key]
    else:
        otherkey = "shift2" + othersegment[-1].lower()
        shift2_zero = shift_dict[otherkey]
        shift2_slope = slope_dict[otherkey]
    if info["detector"] == "FUV":
        eta_full[:] = np.where(active_area,
                        eta - ((time - t0) * shift2_slope + shift2_zero),
                        eta)
    else:
        # Use the same shift2 for every stripe.
        eta_full[:] = eta - ((time - t0) * shift2_slope + shift2_zero)

    # stripe B for NUV
    key = "shift1" + segment[-1].lower()
    if (key not in shift_dict and key not in slope_dict):
        otherkey = "shift1" + othersegment[-1].lower()
        shift1_zero = shift_dict[otherkey]
        shift1_slope = slope_dict[otherkey]
    else:
        shift1_zero = shift_dict[key]
        shift1_slope = slope_dict[key]
    avg_dy = shift2_slope * t_mid + shift2_zero

    # Create the array of shift1 at the times in tl_time.
    if info['addsimulatedwavecal']:
        early_times = np.where(tl_time < transition_time)
        late_times = np.where(tl_time <= transition_time)
        shift1_vs_time = np.zeros(tl_time.shape, dtype=np.float32)
        shift1_vs_time[early_times] = early_slope * tl_time[early_times] + early_intercept
        shift1_vs_time[late_times] = late_slope * tl_time[late_times] + late_intercept
    else:
        shift1_vs_time = shift1_slope * tl_time + shift1_zero

    # Set the SHIFT2[A-C] keywords to the average offset in the
    # cross-dispersion direction.
    # Set DPIXEL1[A-C] to the average of the difference xfull minus the
    # nearest integer to xfull, where xfull is the column of that name;
    # this will be used when assigning wavelengths in extract.py.

    for segment in segment_list:
        key = "SHIFT2" + segment[-1]
        hdr[key] = round(avg_dy, 4)
        if info["detector"] == "FUV":
            xi_active = xi_full[active_area]
            xi_diff = xi_active - np.around(xi_active)
        else:
            xi_psa = xi_full[psa_region_flags_dict[segment]]
            xi_diff = xi_psa - np.around(xi_psa)
        if len(xi_diff) > 0:
            dpixel1 = xi_diff.mean(dtype=np.float64)
        else:
            dpixel1 = 0.
        key = "DPIXEL1" + segment[-1]
        hdr[key] = round(dpixel1, 4)

    phdr["wavecorr"] = wavecorr
    if wavecorr == "COMPLETE":
        filename = cosutil.changeSegment(filename, info["detector"],
                                         info["segment"])
        if shift_file is not None:
            filename = filename + " " + shift_file
        phdr["wavecals"] = filename
        cosutil.printMsg("Wavecal file(s) '%s'" % filename, VERBOSE)

    return (tl_time, shift1_vs_time)

def getSimulatedWavecalInfo(info, key, wavecal_info, wcp_info, input_path):
    """Calculate the slopes and intercepts for the 2 line segments that
    model the behaviour of the shift1 value as a function of time when
    a virtual wavecal is added

    Parameters:
    -----------

    info: dictionary
        info dictionary for the observation

    key: str
        'shift1a' or 'shift1b'

    wavecal_info: list of dictionaries
        wavecal_info list from calcos.Calibration.allWavecals()

    wcp_info: FITS_rec
        Row of the wcptab matching this observation

    input_path: str
        Directory containing the input file; the wavecals will be
        in the same directory

    Returns:
    --------

    tuple containing 5 floats:

    seconds_since_exposure_start,     # 600s from the start of the preceding wavecal
    early_slope,                      # slope of relation for t < seconds_since_exposure_start
    early_intercept,                  # intercept of relation for t < seconds_since_exposure_start
    late_slope,                       # slope of relation for t > seconds_since_exposure_start
    late_intercept                    # intercept of relation for t > seconds_since_exposure_start
    """

    cosutil.printMsg("Adding simulated wavecal")
    # The model that is used to calculate the simulated wavecal depends on
    # the exposure time
    exptime = info['exptime']
    tcrossover = wcp_info['TCROSSOVER']
    if exptime < tcrossover:
        frac = wcp_info['FRACSHORT']
        offset = wcp_info['OFFSET_SHORT']
    else:
        frac = wcp_info['FRACLONG']
        offset = wcp_info['OFFSET_LONG']

    cosutil.printMsg("For exposure of {:.4f}s, frac={:.4f}, offset={:.4f}".format(exptime, frac, offset))
    # Get the wavecal files that bracket this exposure
    tmid = 0.5 * (info["expstart"] + info["expend"])
    shift_info = wavecal.returnWavecalShift(wavecal_info, wcp_info,
                                            info["cenwave"], info["fpoffset"],
                                            tmid)
    shift_dict, slope_dict, wavecalfiles = shift_info

    # Calculate the time at which the simulated wavecal is to be added
    # Should be 600s after the start of the wavecal preceding this exposure
    preceding_wavecal = wavecalfiles.split(' ')[0]
    # Need to get the directory the wavecal is in
    full_wavecal_path = os.path.join(input_path, preceding_wavecal)
    fw = fits.open(full_wavecal_path)
    start = fw[1].header['expstart']
    fw.close()
    # Simulated wavecal is to go 600s after the start of the preceding wavecal
    tsimulated = start + 600.0 / SEC_PER_DAY
    seconds_since_exposure_start = (tsimulated - info['expstart']) * SEC_PER_DAY
    matching_wavecals = wavecal.selectWavecalInfo(wavecal_info,
                                                  info['cenwave'],
                                                  info['fpoffset'])
    nwavecals = len(matching_wavecals)
    if nwavecals != 2:
        cosutil.printMsg('Expected exactly 2 matching wavecals, got {}'.format(nwavecals))
        return None
    shift1_before = matching_wavecals[0]['shift_dict'][key]
    tbefore = matching_wavecals[0]['time']
    seconds_before = (tbefore - info['expstart']) * SEC_PER_DAY
    shift1_after = matching_wavecals[1]['shift_dict'][key]
    tafter = matching_wavecals[1]['time']
    seconds_after = (tafter - info['expstart']) * SEC_PER_DAY
    cosutil.printMsg("Wavecal before exposure has {}={:.4f} at t={:.4f} ({:.4f}s)".format(key, shift1_before, tbefore, seconds_before))
    cosutil.printMsg("Wavecal after exposure has {}={:.4f} at t={:.4f} ({:.4f}s)".format(key, shift1_after, tafter, seconds_after))
    delta_shift = shift1_after - shift1_before
    simulated_shift = shift1_before + frac * delta_shift + offset
    cosutil.printMsg('Simulated shift of {:.4f} added at t={:.4f}s'.format(simulated_shift, seconds_since_exposure_start))
    early_slope = (simulated_shift - shift1_before) / (tsimulated - tbefore)
    early_slope = early_slope / SEC_PER_DAY
    early_intercept = simulated_shift + (info['expstart'] - tsimulated) * early_slope * SEC_PER_DAY
    late_slope = (shift1_after - simulated_shift) / (tafter - tsimulated)
    late_slope = late_slope / SEC_PER_DAY
    late_intercept = simulated_shift + (info['expstart'] - tsimulated) * late_slope * SEC_PER_DAY
    return (seconds_since_exposure_start, early_slope, early_intercept, late_slope, late_intercept)

def computeWavelengths(events, info, reffiles, helcorr="OMIT", hdr=None):
    """Compute wavelengths for a corrtag table.

    Parameters
    ----------
    events: astropy.io.fits record array, or None
        The data unit containing the events table.

    info: dictionary
        Header keywords and values.

    reffiles: dictionary
        Reference file names.

    helcorr: str
        If helcorr is PERFORM or COMPLETE, wavelengths should be corrected
        for heliocentric velocity (helcorr in header will not be modified,
        however); the default value is appropriate for a wavecal.

    hdr: astropy.io.fits Header object, or None
        If not None, apply shift1[abc] and shift2[abc] to the pixel
        coordinates; this is needed for a wavecal exposure.
    """

    if events is None or len(events) == 0:
        return

    # If the current exposure is a wavecal, we need to apply the shift keyword
    # values to the pixel coordinates.  In that case, hdr is supplied so we
    # can read the keywords.
    use_shift_keywords = (hdr is not None)

    # Heliocentric correction is not relevant for wavecals and won't be done.
    if hdr is None:
        message = "%-9s %s for computing wavelengths for the corrtag table" \
                   % ("HELCORR", helcorr)
        cosutil.printMsg(message, VERBOSE)

    disptab = reffiles["disptab"]
    xtractab = reffiles["xtractab"]
    detector = info["detector"]
    opt_elem = info["opt_elem"]
    cenwave = info["cenwave"]
    if detector == "FUV":
        segment_list = [info["segment"]]
    else:
        segment_list = ["NUVA", "NUVB", "NUVC"]
    shift1_dict = {}
    shift2_dict = {}
    if use_shift_keywords:
        for segment in segment_list:
            keyword = "shift1" + segment[-1]
            shift1 = hdr.get(keyword, default=0.)
            keyword = "shift2" + segment[-1]
            shift2 = hdr.get(keyword, default=0.)
            shift1_dict[segment] = shift1
            shift2_dict[segment] = shift2

    # aperture and segment will be added to the filter within the loop.
    filter = {"opt_elem": opt_elem,
              "cenwave": cenwave,
              "fpoffset": info["fpoffset"]}

    wavelength = events.field("WAVELENGTH")

    # The YFULL position is used to determine which stripe a given
    # event corresponds to.
    if use_shift_keywords:
        xi = events.field(xfull).copy()         # because we need to modify it
        if detector == "FUV":
            shift2b = shift2_dict[segment_list[0]]
        else:
            shift2b = shift2_dict["NUVB"]
        eta = events.field(yfull).copy() - shift2b
    else:
        xi = events.field(xfull)
        eta = events.field(yfull)

    if detector == "FUV":
        setActiveArea(events, info, reffiles["brftab"])
        (psa_region_flags, wca_region_flags) = \
                        fuvPsaWcaRegions(eta, info, xtractab)
    else:
        psa_region_flags_dict = nuvPsaRegions(eta, info, xtractab)
        wca_region_flags_dict = nuvWcaRegions(eta, info, xtractab)

    for segment in segment_list:
        # Compute the wavelengths for the output table.
        filter["segment"] = segment
        filter["aperture"] = "PSA"
        psa_disp_rel = dispersion.Dispersion(disptab, filter)
        filter["aperture"] = "WCA"
        wca_disp_rel = dispersion.Dispersion(disptab, filter)
        if not (psa_disp_rel.isValid() and wca_disp_rel.isValid()):
            cosutil.printError("Matching row in disptab %s was not found" \
                               % disptab)
            cosutil.printContinuation(
                "can't compute wavelengths for corrtag file.")
            continue
        if detector == "FUV":
            if use_shift_keywords:
                xi -= shift1_dict[segment]
            psa_wavelength = psa_disp_rel.evalDisp(xi)
            # "hdr is None" means the current exposure is not a wavecal
            if hdr is None and (helcorr == "PERFORM" or helcorr == "COMPLETE"):
                psa_wavelength += (psa_wavelength *
                                   (-info["v_helio"]) / SPEED_OF_LIGHT)
            wavelength[:] = np.where(psa_region_flags,
                                     psa_wavelength, wavelength)
            del psa_wavelength
            wca_wavelength = wca_disp_rel.evalDisp(xi)
            wavelength[:] = np.where(wca_region_flags,
                                     wca_wavelength, wavelength)
            del wca_wavelength
        else:
            if use_shift_keywords:
                xi_full = xi - shift1_dict[segment]
            else:
                xi_full = xi
            # Update the wavelength array for those events that are within
            # the PSA and WCA regions for the current stripe.
            psa_wavelength = psa_disp_rel.evalDisp(xi_full)
            if hdr is None and (helcorr == "PERFORM" or helcorr == "COMPLETE"):
                psa_wavelength += (psa_wavelength *
                                   (-info["v_helio"]) / SPEED_OF_LIGHT)
            wavelength[:] = np.where(psa_region_flags_dict[segment],
                                     psa_wavelength, wavelength)
            del psa_wavelength
            wca_wavelength = wca_disp_rel.evalDisp(xi_full)
            wavelength[:] = np.where(wca_region_flags_dict[segment],
                                     wca_wavelength, wavelength)
            del wca_wavelength
        psa_disp_rel.close()
        wca_disp_rel.close()

    return

def fuvPsaWcaRegions(eta, info, xtractab):
    """Determine the sets of events within the PSA and WCA.

    This version is for FUV data.

    Parameters
    ----------
    eta: array like
        Pixel coordinates in cross-dispersion direction.

    info: dictionary
        Keywords and values.

    xtractab: str
        Name of spectral extraction parameters reference table.

    Returns
    -------
    tuple of two boolean arrays
        The values in the first array are True for events that are within
        the active area and also within the PSA region (i.e. below the
        midpoint between PSA and WCA).  The second array values are True
        for events that are within the active area and also within the WCA
        region.
    """

    global active_area

    psa_region_flags = active_area.copy()
    wca_region_flags = active_area.copy()

    filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"],
              "segment": info["segment"]}       # aperture added below
    middle = float(FUV_X) / 2.

    # The computation of the 'boundary' variable makes an assumption
    # about the relative locations of the PSA and WCA regions on the
    # detectors.  The PSA spectral region is at lower Y pixel numbers.

    filter["aperture"] = "PSA"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_psa = xtract_info.field("b_spec")[0] + \
                 xtract_info.field("slope")[0] * middle

    filter["aperture"] = "WCA"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_wca = xtract_info.field("b_spec")[0] + \
                 xtract_info.field("slope")[0] * middle

    boundary = int(round((b_spec_psa + b_spec_wca) / 2.))

    psa_region_flags &= (eta < boundary)
    wca_region_flags &= (eta >= boundary)

    return (psa_region_flags, wca_region_flags)

def flagsFromBoundaries(eta, boundaries_dict):
    """Given lower and upper cutoffs, return arrays of Boolean flags.

    Parameters
    ----------
    eta: array like
        Pixel coordinates in cross-dispersion direction.

    boundaries_dict: dictionary
        Key is a stripe name, value is a tuple with the lower and upper Y
        boundaries for that stripe.

    Returns
    -------
    dictionary of boolean arrays
        Dictionary with stripe name ("NUVA", "NUVB", "NUVC") as the key
        and an array of Boolean flags as the value, True for events for
        which the Y coordinate is within the regions for the PSA.
    """

    region_flags_dict = {}
    for key in boundaries_dict:
        (lower, upper) = boundaries_dict[key]
        region_flags_dict[key] = (eta >= lower) & (eta < upper)

    return region_flags_dict

def nuvPsaBoundaries(eta, info, xtractab):
    """Determine the limits in Y for each NUV region for the PSA.

    Parameters
    ----------
    eta: array like
        Pixel coordinates in cross-dispersion direction.

    info: dictionary
        Keywords and values.

    xtractab: str
        Name of spectral extraction parameters reference table.

    Returns
    -------
    dictionary of tuples
        Dictionary with stripe name ("NUVA", "NUVB", "NUVC") as the key
        and a two-element tuple as the value; the elements of the tuple are
        the lower and upper Y coordinates (integers) of the regions for the
        stripe for the PSA.
    """

    # segment will be added to the filter below.
    filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"],
              "aperture": "PSA"}
    middle = float(NUV_X) / 2.

    # b_spec_a, b_spec_b, b_spec_c, are the locations (at the middle of the
    # detector) of stripes A, B, C for the PSA, and b_spec_wca is the location
    # of stripe A for the WCA.

    filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_a = xtract_info.field("b_spec")[0] + \
               xtract_info.field("slope")[0] * middle

    filter["segment"] = "NUVB"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_b = xtract_info.field("b_spec")[0] + \
               xtract_info.field("slope")[0] * middle

    filter["segment"] = "NUVC"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_c = xtract_info.field("b_spec")[0] + \
               xtract_info.field("slope")[0] * middle

    filter["segment"] = "NUVA"
    filter["aperture"] = "WCA"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_wca = xtract_info.field("b_spec")[0] + \
                 xtract_info.field("slope")[0] * middle

    # Set boundaries midway between adjacent stripes.
    boundary_a_b = int(round((b_spec_a + b_spec_b) / 2.))
    boundary_b_c = int(round((b_spec_b + b_spec_c) / 2.))
    boundary_c_wca = int(round((b_spec_c + b_spec_wca) / 2.))

    boundaries_dict = {}
    boundaries_dict["NUVA"] = (0, boundary_a_b)
    boundaries_dict["NUVB"] = (boundary_a_b, boundary_b_c)
    boundaries_dict["NUVC"] = (boundary_b_c, boundary_c_wca)

    return boundaries_dict

def nuvPsaRegions(eta, info, xtractab):
    """Determine the set of events within the NUV regions for the PSA.

    This is only used for NUV data.

    Parameters
    ----------
    eta: array like
        Pixel coordinates in cross-dispersion direction.

    info: dictionary
        Keywords and values.

    xtractab: str
        Name of spectral extraction parameters reference table.

    Returns
    -------
    dictionary of boolean arrays
        Dictionary with stripe name ("NUVA", "NUVB", "NUVC") as the key
        and an array of Boolean flags as the value, true for events for
        which the Y coordinate is within the regions for the PSA.
    """

    boundaries_dict = nuvPsaBoundaries(eta, info, xtractab)

    region_flags_dict = flagsFromBoundaries(eta, boundaries_dict)

    return region_flags_dict

def nuvWcaBoundaries(eta, info, xtractab):
    """Determine the set of events within the NUV regions for the WCA.

    This is only used for NUV data.

    Parameters
    ----------
    eta: array like
        Pixel coordinates in cross-dispersion direction.

    info: dictionary
        Keywords and values.

    xtractab: str
        Name of spectral extraction parameters reference table.

    Returns
    -------
    dictionary of tuples
        Dictionary with stripe name ("NUVA", "NUVB", "NUVC") as the key
        and a two-element tuple as the value; the elements of the tuple are
        the lower and upper Y coordinates (integers) of the regions for the
        stripe for the WCA.
    """

    # aperture and segment will be added to the filter below.
    filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"]}
    middle = float(NUV_X) / 2.

    # b_spec_c is the location (at the middle of the detector) of stripe C
    # for the PSA, and b_spec_wca is the location
    # of stripe A for the WCA.

    filter["segment"] = "NUVC"
    filter["aperture"] = "PSA"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_c = xtract_info.field("b_spec")[0] + \
               xtract_info.field("slope")[0] * middle

    filter["aperture"] = "WCA"

    filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_wca_a = xtract_info.field("b_spec")[0] + \
                   xtract_info.field("slope")[0] * middle

    filter["segment"] = "NUVB"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_wca_b = xtract_info.field("b_spec")[0] + \
                   xtract_info.field("slope")[0] * middle

    filter["segment"] = "NUVC"
    xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
    b_spec_wca_c = xtract_info.field("b_spec")[0] + \
                   xtract_info.field("slope")[0] * middle

    # Set boundaries midway between adjacent stripes.
    boundary_c_wca = int(round((b_spec_c + b_spec_wca_a) / 2.))
    boundary_a_b = int(round((b_spec_wca_a + b_spec_wca_b) / 2.))
    boundary_b_c = int(round((b_spec_wca_b + b_spec_wca_c) / 2.))

    boundaries_dict = {}
    boundaries_dict["NUVA"] = (boundary_c_wca, boundary_a_b)
    boundaries_dict["NUVB"] = (boundary_a_b, boundary_b_c)
    boundaries_dict["NUVC"] = (boundary_b_c, NUV_Y)

    return boundaries_dict

def nuvWcaRegions(eta, info, xtractab):
    """Determine the set of events within the NUV regions for the WCA.

    This is only used for NUV data.

    Parameters
    ----------
    eta: array like
        Pixel coordinates in cross-dispersion direction.

    info: dictionary
        Keywords and values.

    xtractab: str
        Name of spectral extraction parameters reference table.

    Returns
    -------
    dictionary of tuples
        Dictionary with stripe name ("NUVA", "NUVB", "NUVC") as the key
        and an array of Boolean flags as the value, true for events for
        which the Y coordinate is within the regions for the WCA.
    """

    boundaries_dict = nuvWcaBoundaries(eta, info, xtractab)

    region_flags_dict = flagsFromBoundaries(eta, boundaries_dict)

    return region_flags_dict

def getWavecalOffsets(events, info, wavecorr, xtractab, brftab):
    """Get min and max values of shift1 and shift2.

    Parameters
    ----------
    events: astropy.io.fits record array
        the data unit containing the events table

    info: dictionary
        Keywords and values.

    wavecorr: str
        Specifies whether wavecal processing has been done.

    xtractab: str
        Name of spectral extraction parameters reference table.

    brftab: str
        Name of baseline reference table

    Returns
    -------
    dictionary
        Each key is a two-element tuple, the lower and upper limits in Y;
        the value is a list:
        [min_shift1, max_shift1, min_shift2, max_shift2], where
        min_shift1 and max_shift1 are the minimum and maximum values
        of the wavecal shift in the dispersion direction for events with
        lower <= Y < upper (positive means a feature was detected at larger
        pixel coordinate than in the template), and min_shift2 and
        max_shift2 are the corresponding values in the cross-dispersion
        direction.
    """

    global active_area

    minmax_shift_dict = {}

    # wavecorr can be set to SKIPPED for FUVB data if there is no wavecal
    # data and no corresponding segment A data (see concurrent.py).  In
    # this case, the shifts will be non-zero if fpoffset is not zero.
    if info["obstype"] == "IMAGING" or wavecorr == "OMIT":
        if info["detector"] == "NUV":
            minmax_shift_dict[(0, NUV_Y)] = [0., 0., 0., 0.]
        else:
            minmax_shift_dict[(0, FUV_Y)] = [0., 0., 0., 0.]
        return minmax_shift_dict

    #
    # Since we're going to be using the active_area array, make sure it's
    # populated correctly
    setActiveArea(events, info, brftab)

    if active_area.any():
        xi_dopp  = events.field(xdopp)
        eta_corr = events.field(ycorr)
        xi_full  = events.field(xfull)
        eta_full = events.field(yfull)

        xdiff = xi_dopp - xi_full
        ydiff = eta_corr - eta_full

        if info["detector"] == "NUV":
            b_dict_list = [nuvPsaBoundaries(eta_corr, info, xtractab),
                           nuvWcaBoundaries(eta_corr, info, xtractab)]
            # these two could be moved inside the loop, if necessary:
            min_shift2 = ydiff.min()
            max_shift2 = ydiff.max()
            for boundaries_dict in b_dict_list:
                for key in boundaries_dict.keys():
                    (lower, upper) = boundaries_dict[key]
                    flags = (eta_corr >= lower) & (eta_corr < upper)
                    xdiff_subset = xdiff[flags]
                    if len(xdiff_subset) > 0:
                        min_shift1 = xdiff_subset.min()
                        max_shift1 = xdiff_subset.max()
                        # ydiff_subset = ydiff[flags]
                        # min_shift2 = ydiff_subset.min()
                        # max_shift2 = ydiff_subset.max()
                        minmax_shift_dict[(lower, upper)] = \
                            [min_shift1, max_shift1, min_shift2, max_shift2]
                    else:
                        minmax_shift_dict[(lower, upper)] = [0., 0., 0., 0.]
        else:
            xdiff = xdiff[active_area]
            ydiff = ydiff[active_area]

            min_shift1 = xdiff.min()
            max_shift1 = xdiff.max()
            #
            # The difference between the min and max shift1 values should be < 20
            # If it's greater than that, we need to prune the rogue values out
            # This should happen VERY rarely, so we can leave the diagnostic print
            # statements in
            if (max_shift1 - min_shift1) > 20.0:
                # Count the values on either side of the midpoint
                midpoint = 0.5*(min_shift1 + max_shift1)
                lowershift1 = np.where(xdiff < midpoint)[0].size
                highershift1 = np.where(xdiff > midpoint)[0].size
                cosutil.printWarning("Difference between min and max shift1 > 20.0")
                cosutil.printContinuation("Min=%f, max=%f" % (min_shift1, max_shift1))
                cosutil.printContinuation("%d events < midpt, %d events > midpt" % (lowershift1,
                                                                                    highershift1))
                if lowershift1 > highershift1:
                    indices = np.where(xdiff < midpoint)
                else:
                    indices = np.where(xdiff > midpoint)
                min_shift1 = xdiff[indices].min()
                max_shift1 = xdiff[indices].max()
            min_shift2 = ydiff.min()
            max_shift2 = ydiff.max()
            minmax_shift_dict[(0, FUV_Y)] = [min_shift1, max_shift1,
                                             min_shift2, max_shift2]
    else:
        # active_area is all False.
        if info["detector"] == "NUV":
            minmax_shift_dict[(0, NUV_Y)] = [0., 0., 0., 0.]
        else:
            minmax_shift_dict[(0, FUV_Y)] = [0., 0., 0., 0.]

    return minmax_shift_dict

def copyColumns(events):
    """Copy XCORR and YCORR columns to XDOPP, XFULL and YFULL.

    Copy XCORR (RAWX) and YCORR (RAWY) to XDOPP, XFULL, YFULL as initial
    values, in case this is imaging data or wavecal processing will not
    be done.

    Parameters
    ----------
    events: astropy.io.fits record array
        The data unit containing the events table.
    """

    global xcorr, ycorr, xdopp, ydopp, xfull, yfull

    xi  = events.field(xcorr)
    eta = events.field(ycorr)
    xi_dopp  = events.field(xdopp)
    xi_full  = events.field(xfull)
    eta_full = events.field(yfull)

    xi_dopp[:] = xi.copy()
    xi_full[:] = xi.copy()
    eta_full[:] = eta.copy()
