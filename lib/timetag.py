from __future__ import division         # confidence high
import math
import os
import time
import numpy as np
from numpy import random
import pyfits

import cosutil
import burst
import ccos
import concurrent
import dispersion
import phot
import wavecal
from calcosparam import *       # parameter definitions

# This variable gives the data quality flags that will result in events
# not being included when writeImages writes the flt and counts images.
# It will be modified if brstcorr, badtcorr or phacorr is set to perform.
serious_dq_flags = 0

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

def timetagBasicCalibration (input, inpha, outtag,
                  output, outcounts, outflash, outcsum,
                  cl_args,
                  info, switches, reffiles,
                  wavecal_info):
    """Do the basic processing for either time-tag or accum data.

    The function value will be zero if there was no problem,
    and it will be one if there was no input data.

    @param input: name of the input file
    @type input: string
    @param inpha: name of the input file containing the pulse height
        histogram (FUV ACCUM only)
    @type inpha: string
    @param outtag: name of the output file for corrected time-tag data
    @type outtag: string
    @param output: name of the output file for flat-fielded count-rate image
    @type output: string
    @param outcounts: name of the output file for count-rate image
    @type outcounts: string
    @param outflash: name of the output file for tagflash wavecal spectra
        (or None)
    @type outflash: string
    @param outcsum: name of the output image for OPUS to add to cumulative
        image (or None)
    @type outcsum: string
    @param cl_args: some of the command-line arguments
    @type cl_args: dictionary
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param wavecal_info: when wavecal exposures were processed, the results
        were stored in dictionaries in this list
    @type wavecal_info: list of dictionaries
    """

    if info["obsmode"] == "TIME-TAG":
        cosutil.printIntro ("TIME-TAG calibration")
        names = [("Input", input),
                 ("OutTag", outtag),
                 ("OutFlt", output),
                 ("OutCounts", outcounts)]
        if outflash is not None:
            names.append (("OutFlash", outflash))
        if outcsum is not None:
            names.append (("OutCsum", outcsum))
        cosutil.printFilenames (names,
                                shift_file=cl_args["shift_file"],
                                stimfile=cl_args["stimfile"],
                                livetimefile=cl_args["livetimefile"])
        cosutil.printMode (info)

    # Copy data from the input file to the output.  Then open the output
    # file read/write.
    if info["obsmode"] == "TIME-TAG":
        nrows = cosutil.writeOutputEvents (input, outtag)
    ofd = pyfits.open (outtag, mode="update")
    if ofd["EVENTS"].data is None:
        nrows = 0
    else:
        nrows = len (ofd["EVENTS"].data)

    # events_hdu is a complete fits HDU object (i.e., header plus data),
    # while events (assigned below) is just the data, a recarray object.
    events_hdu = ofd["EVENTS"]

    # Get a copy of the primary header.  This copy will be modified and
    # written to the output image files.
    phdr = ofd[0].header
    # This list also includes the primary header, but we'll ignore this
    # copy of the primary header.
    if info["obsmode"] == "ACCUM" and not info["corrtag_input"]:
        headers = cosutil.getHeaders (input)
        # replace the first extension header so the headers of the
        # pseudo-corrtag table will be updated
        headers[1] = events_hdu.header
    else:
        headers = [phdr]
        for i in range (3):
            headers.append (events_hdu.header)

    # Update the switches and reference file names, so the output header
    # will reflect what was actually used.
    cosutil.overrideKeywords (phdr, headers[1], info, switches, reffiles)

    if nrows == 0:
        writeNull (input, output, outcounts, outcsum,
                   cl_args, info, phdr, headers)
        ofd.close()
        return 1

    setCorrColNames (info["detector"])

    events = events_hdu.data

    # For corrtag input, reinitialize the DQ column if dqicorr is perform.
    if info["corrtag_input"] and switches["dqicorr"] == "PERFORM":
        events.field ("dq")[:] = 0.

    setActiveArea (events, info, reffiles["brftab"])

    doPhotcorr (info, switches, reffiles["imphttab"], phdr, headers[1])

    bursts = doBurstcorr (events, info, switches, reffiles, phdr,
                          cl_args["burstfile"])

    badt = doBadtcorr (events, info, switches, reffiles, phdr)

    if info["obsmode"] == "TIME-TAG":
        (modified, gti) = recomputeExptime (input, bursts, badt, events,
                                            headers[1], info)
        if modified:
            saveNewGTI (ofd, gti)

    doRandcorr (events, info, switches, reffiles, phdr)

    (stim_param, stim_countrate, stim_livetime) = initTempcorr (events,
            input, info, switches, reffiles, headers[1],
            cl_args["stimfile"])

    doTempcorr (stim_param, events, info, switches, reffiles, phdr)

    doGeocorr (events, info, switches, reffiles, phdr)

    # Set this array of flags again, after geometric correction.
    setActiveArea (events, info, reffiles["brftab"])

    doPhacorr (inpha, events, info, switches, reffiles, phdr, headers[1])

    updateGlobrate (info, headers[1])

    if info["obsmode"] == "TIME-TAG":
        countBadEvents (events, bursts, badt, info, headers[1])

    # Copy columns to xdopp, xfull, yfull so we'll have default values.
    if not info["corrtag_input"]:
        copyColumns (events)

    doDoppcorr (events, info, switches, reffiles, phdr)
    initHelcorr (events, info, switches, headers[1])

    doDeadcorr (events, input, info, switches, reffiles, phdr, headers[1],
                stim_countrate, stim_livetime, cl_args["livetimefile"])

    # Write the calcos sum image.
    if outcsum is not None:
        if info["detector"] == "FUV" and info["obsmode"] == "TIME-TAG":
            pha = events.field ("pha")
        else:
            pha = None
        writeCsum (outcsum, events.field (xcorr), events.field (ycorr),
                   events.field ("epsilon"), pha,
                   info["detector"], info["subarray"],
                   phdr, headers[1],
                   cl_args["binx"],
                   cl_args["biny"],
                   cl_args["compress_csum"],
                   cl_args["compression_parameters"])

    doFlatcorr (events, info, switches, reffiles, phdr, headers[1])

    if info["tagflash"]:
        cosutil.printSwitch ("WAVECORR", switches)
    if switches["wavecorr"] == "PERFORM" or switches["wavecorr"] == "COMPLETE":
        if info["tagflash"]:
            shift1_vs_time = concurrent.processConcurrentWavecal (events, \
                        outflash, cl_args["shift_file"],
                        info, switches, reffiles, phdr, headers[1])
            filename = os.path.basename (input)
            if cl_args["shift_file"] is not None:
                filename = filename + " " + cl_args["shift_file"]
            phdr.update ("wavecals", filename)
        else:
            shift1_vs_time = updateFromWavecal (events, wavecal_info,
                        cl_args["shift_file"],
                        info, switches, reffiles, phdr, headers[1])
        # Compute wavelengths for the wavelength column (except for wavecals).
	if info["obstype"] == "SPECTROSCOPIC" and \
           info["exptype"].find ("WAVE") == -1:
            computeWavelengths (events, info, reffiles,
                                helcorr=switches["helcorr"], hdr=None)
    else:
        shift1_vs_time = None

    minmax_shifts = getWavecalOffsets (events)

    dq_array = doDqicorr (events, input, info, switches, reffiles,
                          phdr, headers[1], minmax_shifts)

    writeImages (events.field (xfull), events.field (yfull),
                 events.field ("epsilon"), events.field ("dq"),
                 phdr, headers,
                 dq_array, info["npix"], info["x_offset"], info["exptime"],
                 outcounts, output)

    doStatflag (switches, output, outcounts)

    ofd.close()

    # Comment this out for the time being.
    # appendShift1 (outtag, output, outcounts, shift1_vs_time)

    return 0            # 0 is OK


def setCorrColNames (detector):
    """Assign column names to global variables.

    @param detector: FUV or NUV
    @type detector: string
    """

    global xcorr, ycorr, xdopp, ydopp, xfull, yfull

    xcorr = "XCORR"
    ycorr = "YCORR"
    xdopp = "XDOPP"
    ydopp = "YCORR"

    xfull = "XFULL"
    yfull = "YFULL"

def setActiveArea (events, info, brftab):
    """Assign a value to active_area.

    @param events: the data unit containing the events table
    @type events: record array
    @param info: header keywords and values
    @type info: dictionary
    @param brftab: name of the baseline reference table
    @type brftab: string

    This function updates the global variable active_area, which is a
    Boolean array with the same number of elements as there are rows in
    the events table.  An element will be True if the corresponding
    event (row in the table) is within the FUV active area.  For NUV
    all elements will be set to True.
    """

    global active_area

    xi  = events.field (xcorr)
    eta = events.field (ycorr)
    active_area = np.ones (len (xi), dtype=np.bool8)

    # A value of 1 (True) in active_area means the corresponding event
    # is within the active area.
    if info["detector"] == "FUV":
        (b_low, b_high, b_left, b_right) = \
                cosutil.activeArea (info["segment"], brftab)
        active_area = np.where (xi > b_right, False, active_area)
        active_area = np.where (xi < b_left,  False, active_area)
        active_area = np.where (eta > b_high, False, active_area)
        active_area = np.where (eta < b_low,  False, active_area)
        # Make sure the data type is still boolean.
        active_area = active_area.astype (np.bool8)

def doPhotcorr (info, switches, imphttab, phdr, hdr):
    """Update photometry parameter keywords for imaging data.

    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param imphttab: the name of the imaging photometric parameters table
    @type imphttab: string
    @param phdr: the primary header, photcorr keyword updated in-place
    @type phdr: pyfits Header object
    @param hdr: the first extension header, updated in-place
    @type hdr: pyfits Header object
    """

    if info["obstype"] == "IMAGING" and info["detector"] == "NUV":
        cosutil.printSwitch ("PHOTCORR", switches)
        if switches["photcorr"] == "PERFORM":
            obsmode = "cos,nuv," + info["opt_elem"] + "," + info["aperture"]
            obsmode = obsmode.lower()
            phot.doPhot (imphttab, obsmode, hdr)
            phdr.update ("photcorr", "COMPLETE")

def updateGlobrate (info, hdr):
    """Update the GLOBRATE keyword in the extension header.

    @param info: header keywords and values
    @type info: dictionary
    @param hdr: the input events extension header
    @type hdr: pyfits Header object
    """

    globrate = globrate_tt (info["exptime"], info["detector"])
    if info["detector"] == "FUV":
        keyword = "globrt_" + info["segment"][-1]
    else:
        keyword = "globrate"
    globrate = round (globrate, 4)
    hdr.update (keyword, globrate)

def globrate_tt (exptime, detector):
    """Return the global count rate for time-tag data.

    @param exptime: the exposure time
    @type exptime: float
    @param detector: FUV or NUV
    @type detector: string

    @return: the global count rate, counts per second
    @rtype: float
    """

    global active_area

    if exptime <= 0.:
        return 0.

    if detector == "NUV":
        return float (len (active_area)) / exptime

    return np.sum (active_area.astype (np.float32)) / exptime

def doBurstcorr (events, info, switches, reffiles, phdr, burstfile):
    """Find bursts, and flag them in the data quality column.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    @param burstfile: name of output text file for burst info (or None)
    @type burstfile: string

    @return: list of [bad_start, bad_stop] intervals during which a burst
        was detected (seconds since expstart)
    @rtype: list of two-element lists, or None
    """

    global serious_dq_flags

    bursts = None
    if info["segment"][:3] == "FUV":
        # Find and flag regions where the count rate is unreasonably high.
        cosutil.printSwitch ("BRSTCORR", switches)
        if switches["brstcorr"] == "PERFORM":
            serious_dq_flags |= DQ_BURST
            cosutil.printRef ("brsttab", reffiles)
            cosutil.printRef ("xtractab", reffiles)
            bursts = burst.burstFilter (events.field ("time"),
                         events.field (ycorr), events.field ("dq"),
                         reffiles, info, burstfile)
            phdr.update ("brstcorr", "COMPLETE")

    return bursts

def doBadtcorr (events, info, switches, reffiles, phdr):
    """Flag bad time intervals in the data quality column.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object

    @return: list of [bad_start, bad_stop] intervals from the badttab
        (converted to seconds since expstart)
    @rtype: list of two-element lists
    """

    global serious_dq_flags

    badt = []

    cosutil.printSwitch ("BADTCORR", switches)
    if switches["badtcorr"] == "PERFORM":
        serious_dq_flags |= DQ_BAD_TIME
        cosutil.printRef ("BADTTAB", reffiles)
        badt = filterByTime (events.field ("time"), events.field ("dq"),
                    reffiles["badttab"], info["expstart"], info["segment"])
        phdr["badtcorr"] = "COMPLETE"

    return badt

def filterByTime (time, dq, badttab, expstart, segment):
    """Flag bad time intervals in dq.

    For each bad time interval in the badttab, a flag will be set in the
    data quality column for each event within that time interval.

    @param time: the time column in the events table
    @type time: numpy array
    @param dq: the data quality column in the events table (updated in-place)
    @type dq: numpy array
    @param badttab: the name of the bad-time-intervals table
    @type badttab: string
    @param expstart: the exposure start time (MJD)
    @type expstart: float
    @param segment: FUVA or FUVB
    @type segment: string

    @return: list of [bad_start, bad_stop] intervals from the badttab
        (converted to seconds since expstart)
    @rtype: list of two-element lists
    """

    # Flag regions listed in the badt table.
    badt_info = cosutil.getTable (badttab, filter={"segment": segment})

    badt = []
    if badt_info is not None:
        nrows = badt_info.shape[0]

        start = badt_info.field ("start")
        stop  = badt_info.field ("stop")

        # Convert from MJD to seconds after expstart.
        for i in range (nrows):
            start[i] = (start[i] - expstart) * SEC_PER_DAY
            stop[i] = (stop[i] - expstart) * SEC_PER_DAY
            badt.append ([start[i], stop[i]])

        # For each time interval in the badttab, flag every event for which
        # the time falls within that interval.
        for i in range (nrows):
            dq |= np.where (np.logical_and \
                      (time >= start[i], time <= stop[i]), DQ_BAD_TIME, 0)

    return badt

def countBadEvents (events, bursts, badt, info, hdr):
    """Update keywords for events and time lost.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param bursts: list of [bad_start, bad_stop] intervals during which
        a burst was detected
    @type bursts: list of two-element lists
    @param badt: list of [bad_start, bad_stop] intervals from the badttab
        (converted to seconds since expstart)
    @type badt: list of two-element lists
    @param info: keywords and values (exptime can be updated)
    @type info: dictionary
    @param hdr: the events extension header (keywords will be updated)
    @type hdr: pyfits Header object
    """

    t = events.field ("time").astype (np.float64)
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
                r = ccos.range (t, burst[0], burst[1])
                n_burst += (r[1] - r[0])
        t_key = "tbrst_" + info["segment"][-1]
        n_key = "nbrst_" + info["segment"][-1]
        hdr.update (t_key, t_burst)
        hdr.update (n_key, n_burst)

        # The length of t is the total number of events, while the number of
        # True flags is the number of events that are within the active area.
        n_outside_active_area = len (t) - np.sum (active_area.astype (np.int32))
        n_key = "nout_" + info["segment"][-1]
        hdr.update (n_key, n_outside_active_area)

    for (bad_start, bad_stop) in badt:
        if badt is not None:
            # badt includes all time intervals in the badttab, and many of
            # those intervals may lie outside the time range of the exposure.
            if bad_stop <= expstart:
                continue
            if bad_start >= expend:
                continue
            bad_start = max (bad_start, expstart)
            bad_stop = min (bad_stop, expend)
            t_badt += (bad_stop - bad_start)
            r = ccos.range (t, bad_start, bad_stop)
            n_badt += (r[1] - r[0])
    if info["detector"] == "FUV":
        t_key = "tbadt_" + info["segment"][-1]
        n_key = "nbadt_" + info["segment"][-1]
    else:
        t_key = "tbadt"
        n_key = "nbadt"
    hdr.update (t_key, t_badt)
    hdr.update (n_key, n_badt)

    if info["detector"] == "FUV":
        # The keyword for the number of events flagged as bad due to pulse
        # height out of bounds has already been set, so just get the value.
        n_pha_key = "npha_" + info["segment"][-1]
        n_bad_pha = hdr.get (n_pha_key, 0)

    hdr.update ("nbadevnt", n_burst + n_badt +
                 n_outside_active_area + n_bad_pha)

def recomputeExptime (input, bursts, badt, events, hdr, info):
    """Recompute the exposure time and update the keyword.

    @param input: name of the input file (for getting GTI table)
    @type input: string
    @param bursts: list of [bad_start, bad_stop] intervals during which
        a burst was detected
    @type bursts: list of two-element lists
    @param badt: list of [bad_start, bad_stop] intervals from the badttab
        (converted to seconds since expstart)
    @type badt: list of two-element lists
    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param hdr: the events extension header (exptime keyword can be updated)
    @type hdr: pyfits Header object
    @param info: keywords and values (exptime can be updated)
    @type info: dictionary

    @return: a flag indicating whether there was actually any change
        to the list of [start, stop] intervals, and an updated list
        of [start, stop] good time intervals (seconds since expstart),
        updated from the GTI table in the raw file by excluding bursts
        and intervals flagged as bad by the badttab
    @rtype: tuple:  (boolean, list of two-element lists)
    """

    modified_0 = False
    gti = cosutil.returnGTI (input)
    if len (gti) <= 0:
        cosutil.printWarning ("No GTI table found in raw file.", VERBOSE)
        time = events.field ("time")
        gti = [[time[0], time[-1]]]
        modified_0 = True

    (modified_1, gti) = recomputeGTI (gti, bursts)
    (modified_2, gti) = recomputeGTI (gti, badt)
    modified = modified_0 or modified_1 or modified_2

    exptime = 0.
    for (start, stop) in gti:
        exptime += (stop - start)

    old_exptime = hdr.get ("exptime", 0.)
    if exptime != old_exptime:
        hdr.update ("exptime", exptime)
        info["exptime"] = exptime
        if abs (exptime - old_exptime) > 1.:
            cosutil.printWarning ("exposure time in header was %.3f" % \
                    old_exptime, VERBOSE)
            cosutil.printContinuation ("exptime has been corrected to %.3f" % \
                    exptime, VERBOSE)

    return (modified, gti)

def recomputeGTI (gti, badt):
    """Recompute the list of good [start, stop] intervals.

    @param gti: list of [start, stop] good time intervals (times are in
        seconds since EXPSTART)
    @type gti: list of two-element lists
    @param badt: list of [bad_start, bad_stop] intervals, e.g. during which
        there was a burst or a bad time interval from the BADTTAB (seconds
        since EXPSTART)
    @type badt: list of two-element lists

    @return: a flag indicating whether there was actually any change
        to the list of [start, stop] intervals, and an updated list
        of [start, stop] good time intervals
    @rtype: tuple:  (boolean, list of two-element lists)
    """

    modified = False                    # initial value
    if not badt:
        return (modified, gti)

    for (bad_start, bad_stop) in badt:
        new_gti = []
        for (start, stop) in gti:
            if bad_start >= stop or bad_stop <= start:
                new_gti.append ([start, stop])
            else:
                if bad_start > start:
                    new_gti.append ([start, bad_start])
                    modified = True
                if bad_stop < stop:
                    new_gti.append ([bad_stop, stop])
                    modified = True
        gti = new_gti

    return (modified, gti)

def saveNewGTI (ofd, gti):
    """Append new GTI information as a BINTABLE extension.

    @param ofd: output file header/data list
    @type ofd: pyfits HDUList object
    @param gti: an updated list of [start, stop] good time intervals
    @type gti: list of two-element lists
    """

    len_gti = len (gti)
    col = []
    col.append (pyfits.Column (name="START", format="1D", unit="s"))
    col.append (pyfits.Column (name="STOP", format="1D", unit="s"))
    cd = pyfits.ColDefs (col)
    hdu = pyfits.new_table (cd, nrows=len_gti)
    hdu.header.update ("extname", "GTI")
    outdata = hdu.data
    startcol = outdata.field ("START")
    stopcol = outdata.field ("STOP")
    for i in range (len_gti):
        startcol[i] = gti[i][0]
        stopcol[i] = gti[i][1]

    # Set extver for the new GTI table to be larger than extver for any
    # existing GTI table.  We expect only one, the original one, but there
    # could be others.
    last_extver = 0                     # initial value
    for i in range (1, len(ofd)):
        existing_gti = ofd[i]
        extname = existing_gti.header.get ("extname", "MISSING")
        extname = extname.upper()
        if extname == "GTI":
            extver = existing_gti.header.get ("extver", 1)
            last_extver = max (last_extver, extver)
    hdu.header.update ("extver", last_extver+1)

    # Now append the updated GTI table.
    ofd.append (hdu)
    # if we have pyfits 2.1.1dev462 or later, we could insert
    # ofd.insert (2, hdu)

    ofd[0].header.update ("nextend", len(ofd)-1)

def doPhacorr (inpha, events, info, switches, reffiles, phdr, hdr):
    """Filter by pulse height.

    @param inpha: name of the input file containing the pulse height histogram
    @type inpha: string
    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    @param hdr: the input events extension header
    @type hdr: pyfits Header object
    """

    global serious_dq_flags

    if info["detector"] == "FUV":
        cosutil.printSwitch ("PHACORR", switches)
        if switches["phacorr"] == "PERFORM":
            serious_dq_flags |= DQ_PH_LOW
            serious_dq_flags |= DQ_PH_HIGH
            if info["obsmode"] == "TIME-TAG":
                cosutil.printRef ("PHATAB", reffiles)
                filterByPulseHeight (events.field ("pha"), events.field ("dq"),
                        reffiles["phatab"], info["segment"], hdr)
            else:
                checkPulseHeight (inpha, reffiles["phatab"], info, hdr)
            phdr["phacorr"] = "COMPLETE"

def filterByPulseHeight (pha, dq, phatab, segment, hdr):
    """Flag events that have a pulse height outside an allowed range.

    This is only called for TIME-TAG mode data.

    @param pha: pulse-height column in events table
    @type pha: numpy array
    @param dq: data-quality column in events table (modified in-place)
    @type dq: numpy array
    @param phatab: name of PHA thresholds table
    @type phatab: string
    @param segment: segment name (FUVA or FUVB)
    @type segment: string
    @param hdr: header for events table extension (keywords for screening
        limits and number of rejected events will be assigned)
    @type hdr: pyfits Header object
    """

    global active_area

    pha_info = cosutil.getTable (phatab, filter={"segment": segment},
                   exactly_one=True)

    low = pha_info.field ("llt")[0]
    high = pha_info.field ("ult")[0]

    # Flag an event if the pulse height is below the minimum value or
    # above the maximum value that is likely to be encountered from a
    # real photon event.
    # Restrict this test to the active area.
    test_low = np.logical_and (active_area, pha < low)
    test_high = np.logical_and (active_area, pha > high)
    dq |= np.where (test_low, DQ_PH_LOW, 0)
    dq |= np.where (test_high, DQ_PH_HIGH, 0)

    # Count the number of rejected events.
    rejected = np.nonzero (dq & DQ_PH_LOW)[0]
    nbad_low = len (rejected)
    rejected = np.nonzero (dq & DQ_PH_HIGH)[0]
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

    keyword = "NPHA_" + segment[-1]
    hdr.update (keyword, nbad)

    # Update the values for the screening limit keywords
    # (low and high are the default values).
    cosutil.updatePulseHeightKeywords (hdr, segment, low, high)

def checkPulseHeight (inpha, phatab, info, hdr):
    """Check that the pulse-height distribution is reasonable.

    This is only called for ACCUM mode data.

    @param inpha: name of file containing pulse-height distribution
    @type inpha: string
    @param phatab: name of table of pulse-height parameters
    @type phatab: string
    @param info: header keywords and values
    @type info: dictionary
    @param hdr: header for events table extension (keywords for screening
        limits and number of rejected events will be assigned)
    @type hdr: pyfits Header object
    """

    pha_info = cosutil.getTable (phatab, filter={"segment": info["segment"]},
                   exactly_one=True)

    low = pha_info.field ("llt")[0]
    high = pha_info.field ("ult")[0]

    # Update the values for the screening limit keywords
    cosutil.updatePulseHeightKeywords (hdr, info["segment"], low, high)

    # The peak in the pulse-height distribution should be within low and high.
    # Apply a factor to low and high to account for the fact that the
    # histogram is from seven-bit values but the values in the table are
    # for five-bit values (the PHA column in an EVENTS table).
    # The mean should be within the factors min_peak and max_peak of the peak.
    low *= TWO_BITS
    high *= TWO_BITS
    min_peak = pha_info.field ("min_peak")[0]
    max_peak = pha_info.field ("max_peak")[0]

    # Read the pulse-height histogram.
    fd = pyfits.open (inpha, mode="readonly")
    pha_data = fd[1].data

    npts = len (pha_data)

    sum = np.sum (np.arange (npts, dtype=np.float32) *
                  pha_data.astype (np.float32))
    sumwgt = np.sum (pha_data.astype (np.float32))
    pha_index = np.argsort (pha_data)
    peak = pha_index[-1]

    if sumwgt == 0.:
        cosutil.printWarning ("Histogram is empty.")
        fd.close()
        return

    meanval = sum / sumwgt

    warn = (cosutil.checkVerbosity (VERY_VERBOSE))      # initial value
    if peak <= low:
        cosutil.printWarning ("Peak in pulse-height distribution is too low.")
        warn = 1
    if peak >= high:
        cosutil.printWarning ("Peak in pulse-height distribution is too high.")
        warn = 1

    if meanval < peak * min_peak:
        cosutil.printWarning ("Mean of pulse-height distribution is too low.")
        warn = 1
    if meanval > peak * max_peak:
        cosutil.printWarning ("Mean of pulse-height distribution is too high.")
        warn = 1

    if warn:
        cosutil.printMsg (
                "Pulse-height distribution peak = %d, mean = %.6g;" % \
                (peak, meanval))
        cosutil.printMsg (
                "the peak should be between %.6g and %.6g," % (low, high))
        cosutil.printMsg (
                "and the mean should be between %.6g and %.6g." % \
                (peak * min_peak, peak * max_peak))

    fd.close()

def doRandcorr (events, info, switches, reffiles, phdr):
    """Add pseudo-random numbers to x and y coordinates within the active area.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: primary header
    @type phdr: pyfits Header object
    """

    global active_area

    if info["detector"] == "FUV":
        cosutil.printSwitch ("RANDCORR", switches)
        if switches["randcorr"] == "PERFORM":
            xi  = events.field (xcorr)
            eta = events.field (ycorr)
            nelem = len (xi)
            if info["randseed"] == -1:
                seed = int (time.time())
                phdr["randseed"] = seed
                msg = "RANDSEED = %d (was -1)" % seed
            else:
                seed = info["randseed"]
                msg = "RANDSEED = %d" % seed
            cosutil.printMsg (msg)
            random.seed (seed)
            rn = random.uniform (-0.5, +0.5, nelem)
            xi[:] = np.where (active_area, xi - rn, xi)
            rn = random.uniform (-0.5, +0.5, nelem)
            eta[:] = np.where (active_area, eta - rn, eta)
            phdr["randcorr"] = "COMPLETE"

def initTempcorr (events, input, info, switches, reffiles, hdr, stimfile):
    """Compute parameters for thermal distortion.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param input: name of raw file (for writing to stimfile)
    @type input: string
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param hdr: the input events extension header
    @type hdr: pyfits Header object
    @param stimfile: name of output text file for stim positions (or None)
    @type stimfile: string

    @return: (stim_param, stim_countrate, stim_livetime); stim_param is a
        dictionary of lists, with keys i0, i1, x0, xslope, y0, yslope;
        stim_countrate and stim_livetime are the count rate of the stims
        and the livetime factor based on that count rate
    @rtype: tuple
    """

    if info["detector"] == "FUV" and \
       (switches["tempcorr"] == "PERFORM" or switches["deadcorr"] == "PERFORM"):
        # Compute the parameters (to be used later).
        time = cosutil.getColCopy (data=events, column="time")
        (stim_param, avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref,
         stim_countrate, stim_livetime) = \
         computeThermalParam (time,
            events.field (xcorr), events.field (ycorr), events.field ("dq"),
            reffiles["brftab"], info["obsmode"],
            info["segment"], info["exptime"], info["stimrate"],
            input, stimfile)
        if switches["tempcorr"] == "PERFORM":
            # Update stim location keywords in extension header.
            stimKeywords (hdr, info["segment"], avg_s1, avg_s2,
                          rms_s1, rms_s2, s1_ref, s2_ref)
    else:
        stim_countrate = 0.
        stim_livetime = 1.
        stim_param = {}

    return (stim_param, stim_countrate, stim_livetime)

def computeThermalParam (time, x, y, dq,
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

    @param time: array of event times
    @type time: numpy array
    @param x: detector X coordinates
    @type x: numpy array
    @param y: detector Y coordinates
    @type y: numpy array
    @param dq: array of data quality flags   (NOTE:  not currently used)
    @type dq: numpy array
    @param brftab: name of baseline reference data table
    @type brftab: string
    @param obsmode: TIME-TAG or ACCUM
    @type obsmode: string
    @param segment: segment name (for FUV)
    @type segment: string
    @param exptime: exposure time (for computing livetime)
    @type exptime: float
    @param stimrate: input count rate for a stim (for computing livetime)
    @type stimrate: float
    @param input: name of raw file (for writing to stimfile)
    @type input: string
    @param stimfile: name of text file to which stim locations will be appended
    @type stimfile: string

    @return: (stim_param, avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref,
        stim_countrate, stim_livetime)
    @rtype: tuple

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
      i0[i], i1[i] is the slice of indices in 'events' corresponding to the
        ith time interval.  Each such interval is of length dt_thermal in
        duration (except possibly the last, which could be shorter).
      x0[i] and xslope[i] are the intercept and slope respectively of the
        linear correction to the X positions (the more rapidly varying
        direction).
      y0[i] and yslope[i] are the intercept and slope for the linear
        correction to the Y positions.
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
    if obsmode == "TIME-TAG":
        fd_brf = pyfits.open (brftab, mode="readonly")
        dt_thermal = fd_brf[1].header["timestep"]
        fd_brf.close()
        cosutil.printMsg (
"Compute thermal corrections from stim positions; timestep is %.6g s:" \
            % dt_thermal, VERY_VERBOSE)
    else:
        # For ACCUM data we want just one time interval.
        dt_thermal = time[-1] - time[0] + 1.

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
    sumstim = (0, 0., 0., 0., 0., 0, 0., 0., 0., 0.)
    last_s1 = s1_ref            # initial default values
    last_s2 = s2_ref
    while t0 <= time[nevents-1]:

        # time[i:j] matches t0 to t1.
        try:
            (i, j) = ccos.range (time, t0, t1)
        except:
            t0 = t1
            t1 = t0 + dt_thermal
            continue
        if i >= j:              # i and j can be equal due to roundoff
            t0 = t1
            t1 = t0 + dt_thermal
            continue

        (s1, sumsq1, counts1, found_s1) = \
                findStim (x[i:j], y[i:j], s1_ref, xwidth, ywidth)

        (s2, sumsq2, counts2, found_s2) = \
                findStim (x[i:j], y[i:j], s2_ref, xwidth, ywidth)

        # Increment sums for averaging the stim positions.
        sumstim = updateStimSum (sumstim, counts1, s1, sumsq1, found_s1,
                                          counts2, s2, sumsq2, found_s2)

        if fd is not None:
            fd.write ("%.0f %.0f" % (t0, min (time[nevents-1], t1)))
            if found_s1:
                fd.write ("  %.1f %.1f" % (s1[1], s1[0]))
            else:
                fd.write ("  INDEF INDEF")
            if found_s2:
                fd.write ("  %.1f %.1f\n" % (s2[1], s2[0]))
            else:
                fd.write ("  INDEF INDEF\n")
        if found_s1:
            last_s1 = s1        # save current value
        else:
            s1 = last_s1        # use last stim position that was found
        if found_s2:
            last_s2 = s2
        else:
            s2 = last_s2
        if cosutil.checkVerbosity (VERY_VERBOSE) or \
           not (found_s1 and found_s2):
            msg = "  %7d ... %7d" % (i, j-1)
            msg += "  %.1f %.1f" % (s1[1], s1[0])
            if not found_s1:
                msg += " (stim1 not found)"
            msg += "  %.1f %.1f" % (s2[1], s2[0])
            if not found_s2:
                msg += " (stim2 not found)"
            if not (found_s1 and found_s2):
                cosutil.printWarning (msg)
                if time[j-1] - time[i] < dt_thermal and obsmode == "TIME-TAG":
                    cosutil.printContinuation (\
                "Note that the time interval is %g s" % (time[j-1] - time[i]))
            else:
                cosutil.printMsg (msg)

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
    rms_s1 = [-1., -1.]
    rms_s2 = [-1., -1.]
    if sumstim[0] > 0:
        avg_s1[0] = sumstim[1] / sumstim[0]             # y
        avg_s1[1] = sumstim[2] / sumstim[0]             # x
        if sumstim[0] > 1:
            rms_s1[0] = math.sqrt (sumstim[3] / (sumstim[0] - 1.))
            rms_s1[1] = math.sqrt (sumstim[4] / (sumstim[0] - 1.))
        else:
            rms_s1[0] = math.sqrt (sumstim[3])
            rms_s1[1] = math.sqrt (sumstim[4])
    if sumstim[5] > 0:
        avg_s2[0] = sumstim[6] / sumstim[5]
        avg_s2[1] = sumstim[7] / sumstim[5]
        if sumstim[5] > 1:
            rms_s2[0] = math.sqrt (sumstim[8] / (sumstim[5] - 1.))
            rms_s2[1] = math.sqrt (sumstim[9] / (sumstim[5] - 1.))
        else:
            rms_s2[0] = math.sqrt (sumstim[8])
            rms_s2[1] = math.sqrt (sumstim[9])

    if counts1 > 0 and counts2 > 0:
        stim_countrate = (counts1 + counts2) / (2. * exptime)
    elif counts1 > 0:
        stim_countrate = counts1 / exptime
    elif counts2 > 0:
        stim_countrate = counts2 / exptime
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

def findStim (x, y, stim_ref, xwidth, ywidth):
    """Find one stim in time-tag data.

    @param x: array of detector X coordinates
    @type x: array
    @param y: array of detector Y coordinates
    @type y: array
    @param stim_ref: reference position (y, x) for the stim
    @type stim_ref: tuple
    @param xwidth: half width of the search region in X
    @type xwidth: int
    @param ywidth: half width of the search region in Y
    @type ywidth: int

    @return: ((sy, sx), (sumysq, sumxsq), n, found_stim), where (sy, sx) is
        the stim location (if found), (sumysq, sumxsq) is the sum of squared
        deviations from the mean location, n is the number of events for this
        stim within the current time interval, and found_stim is True if the
        stim was actually found (i.e. if n > 0).
    @rtype: tuple
    """

    # This is the search region for finding the stim.
    sxlow  = stim_ref[1] - xwidth
    sxhigh = stim_ref[1] + xwidth
    sylow  = stim_ref[0] - ywidth
    syhigh = stim_ref[0] + ywidth

    # Truncate at the lower and upper borders, excluding the first
    # and last lines.
    sylow = max (sylow, 1)
    syhigh = min (syhigh, 1022)

    # Initial value of mask is 1. (which in this case means "good").
    mask = np.ones (len (x), dtype=np.float32)

    # Now set mask to 0. ("bad") outside the search region.
    mask = np.where (x > sxhigh, 0., mask)
    mask = np.where (x < sxlow,  0., mask)
    mask = np.where (y > syhigh, 0., mask)
    mask = np.where (y < sylow,  0., mask)
    n = np.sum (mask)
    if n > 0.:
        # The stim reference position is subtracted before taking the sum
        # and then added back to the average in order to reduce the
        # possibility of numerical roundoff errors.
        sumx = np.sum ((x-stim_ref[1]) * mask)
        sumy = np.sum ((y-stim_ref[0]) * mask)
        sx = sumx / n + stim_ref[1]
        sy = sumy / n + stim_ref[0]
        # sum of squared deviations, for computing RMS
        sumxsq = np.sum ((x-sx)**2 * mask)
        sumysq = np.sum ((y-sy)**2 * mask)
        found_stim = True
    else:
        sx = None
        sy = None
        sumxsq = None
        sumysq = None
        found_stim = False

    return ((sy, sx), (sumysq, sumxsq), n, found_stim)

def updateStimSum (sumstim, nevents1, s1, sumsq1, found_s1,
                            nevents2, s2, sumsq2, found_s2):
    """Update sums for averages of stim positions.

    @param sumstim: tuple with current sums:
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
    @type sumstim: tuple
    @param nevents1: number of events for first stim in current time interval
    @type nevents1: int
    @param s1: tuple of (y,x) coordinates of the first stim in current interval
    @type s1: tuple
    @param found_s1: True if the first stim was actually found
    @type found_s1: boolean
    @param nevents2: number of events for second stim in current time interval
    @type nevents2: int
    @param s2: same as s1, but for the second stim
    @type s2: tuple
    @param found_s2: True if the second stim was actually found
    @type found_s2: boolean

    @return: an updated sumstim tuple
    @rtype: tuple

    nevents1 and nevents2 are used as weights when incrementing the sums.
    n1 and n2 are the total number of events for the first and second stims
    respectively.
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

def stimKeywords (hdr, segment, avg_s1, avg_s2, rms_s1, rms_s2,
                  s1_ref, s2_ref):
    """Update keywords for the locations of the stims.

    @param hdr: the input events extension header (updated)
    @type hdr: pyfits Header object
    @param segment: FUVA or FUVB
    @type segment: string

    avg_s1[0] is the average Y location of the first stim.
    avg_s1[1] is the average X location of the first stim.
    avg_s2[0] is the average Y location of the second stim.
    avg_s2[1] is the average X location of the second stim.

    rms_s1[0] is the RMS in Y for the first stim.
    rms_s1[1] is the RMS in X for the first stim.
    rms_s2[0] is the RMS in Y for the second stim.
    rms_s2[1] is the RMS in X for the second stim.

    s1_ref[0] is the Y position of the first stim, from the BRFTAB.
    s1_ref[1] is the X position of the first stim, from the BRFTAB.
    s2_ref[0] is the Y position of the second stim, from the BRFTAB.
    s2_ref[1] is the X position of the second stim, from the BRFTAB.
    """

    seg = segment[-1]           # "A" or "B"

    hdr.update ("STIM"+seg+"0LX", s1_ref[1])
    hdr.update ("STIM"+seg+"0LY", s1_ref[0])
    hdr.update ("STIM"+seg+"0RX", s2_ref[1])
    hdr.update ("STIM"+seg+"0RY", s2_ref[0])

    if avg_s1[0] is None or avg_s1[1] is None:
        hdr.update ("STIM"+seg+"_LX", -1.)
        hdr.update ("STIM"+seg+"_LY", -1.)
    else:
        hdr.update ("STIM"+seg+"_LX", round (avg_s1[1], 3))
        hdr.update ("STIM"+seg+"_LY", round (avg_s1[0], 3))
        hdr.update ("STIM"+seg+"SLX", round (rms_s1[1], 3))
        hdr.update ("STIM"+seg+"SLY", round (rms_s1[0], 3))

    if avg_s2[0] is None or avg_s2[1] is None:
        hdr.update ("STIM"+seg+"_RX", -1.)
        hdr.update ("STIM"+seg+"_RY", -1.)
    else:
        hdr.update ("STIM"+seg+"_RX", round (avg_s2[1], 3))
        hdr.update ("STIM"+seg+"_RY", round (avg_s2[0], 3))
        hdr.update ("STIM"+seg+"SRX", round (rms_s2[1], 3))
        hdr.update ("STIM"+seg+"SRY", round (rms_s2[0], 3))

def thermalParam (s1, s2, s1_ref, s2_ref):
    """Compute linear thermal distortion correction from stim positions.

    @param s1: measured location in raw data of first stim (y, x)
    @type s1: tuple
    @param s2: measured location in raw data of second stim (y, x)
    @type s2: tuple
    @param s1_ref: reference location of first stim (y, x)
    @type s1_ref: tuple
    @param s2_ref: reference location of second stim (y, x)
    @type s2_ref: tuple

    @return: (xintercept, xslope, yintercept, yslope)
    @rtype: tuple
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

def doTempcorr (stim_param, events, info, switches, reffiles, phdr):
    """Apply thermal distortion correction.

    @param stim_param: a dictionary of lists, with keys
        i0, i1, x0, xslope, y0, yslope
    @type stim_param: dictionary of lists
    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
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

    @param x: array of detector X coordinates
    @type x: array
    @param y: array of detector Y coordinates
    @type y: array
    @param stim_param: a dictionary of lists, with keys
        i0, i1, x0, xslope, y0, yslope
    @type stim_param: dictionary of lists

    @return: True if a correction was actually applied
    @rtype: boolean

    No correction is necessary and none will be applied if the slopes are
    all 0 and the intercepts are all 1.
    """

    # These are the parameters found by computeThermalParam.
    x0 = stim_param["x0"]
    xslope = stim_param["xslope"]
    y0 = stim_param["y0"]
    yslope = stim_param["yslope"]

    actually_done = False

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
            actually_done = True

    return actually_done

def doGeocorr (events, info, switches, reffiles, phdr):
    """Apply geometric correction.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
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

def doDqicorr (events, input, info, switches, reffiles,
               phdr, hdr, minmax_shifts):
    """Create a data quality array, initialized from the DQI table.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param input: name of raw file, used for getting DQ array for ACCUM data
    @type input: string
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    @param hdr: the input events extension header
    @type hdr: pyfits Header object
    @param minmax_shifts: (min_shift1, max_shift1, min_shift2, max_shift2)
    @type minmax_shifts: tuple

    @return: 2-D data quality array
    @rtype: numpy array

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

    # temp_switch is only used for printing the DQICORR message
    temp_switch = {}
    if switches["dqicorr"] == "COMPLETE" and info["corrtag_input"]:
        temp_switch["dqicorr"] = "PERFORM (complete, but repeat)"
    else:
        temp_switch["dqicorr"] = switches["dqicorr"]
    cosutil.printSwitch ("DQICORR", temp_switch)

    if info["obsmode"] == "TIME-TAG" or info["corrtag_input"]:
        # Create an initially zero 2-D data quality extension array.
        dq_array = np.zeros (info["npix"], dtype=np.int16)
    else:
        # Read the data quality array from the rawaccum file.
        dq_array = cosutil.getInputDQ (input)

    # If the input is a corrtag file and dqicorr was done when that file was
    # created, we should do dqicorr again.
    if switches["dqicorr"] == "PERFORM" or switches["dqicorr"] == "COMPLETE":

        cosutil.printRef ("BPIXTAB", reffiles)
        bpixtab = reffiles["bpixtab"]

        # Update the dq column in the events list with the bpixtab regions.
        dq_info = cosutil.getTable (bpixtab,
                                    filter={"segment": info["segment"]})
        if dq_info is not None:
            pharange = cosutil.getPulseHeightRange (hdr, info["segment"])
            # xxx temporary; eventually select rows based on pharange
            ref_pharange = cosutil.tempPulseHeightRange (bpixtab)
            cosutil.comparePulseHeightRanges (pharange, ref_pharange,
                                              bpixtab)
            ccos.applydq (dq_info.field ("lx"), dq_info.field ("ly"),
                          dq_info.field ("dx"), dq_info.field ("dy"),
                          dq_info.field ("dq"),
                          events.field (xcorr), events.field (ycorr),
                          events.field ("dq"))
            del dq_info

        # Copy values from the bpixtab to the dq_array, applying offsets
        # depending on the wavecal shift and the Doppler shift.
        (doppmag, doppzero, orbitper) = dopplerParam (info,
                                reffiles["disptab"], switches["doppcorr"])
        minmax_doppler = cosutil.minmaxDoppler (info, switches["doppcorr"],
                               doppmag, doppzero, orbitper)
        cosutil.updateDQArray (bpixtab, info, dq_array,
                               minmax_shifts, minmax_doppler)

        # Flag regions that are outside any subarray as out of bounds.
        cosutil.flagOutOfBounds (hdr, dq_array, info, switches,
                                 reffiles["brftab"], reffiles["geofile"],
                                 minmax_shifts, minmax_doppler)

        # Flag the region that is outside the active area.
        if info["detector"] == "FUV":
            cosutil.flagOutsideActiveArea (dq_array,
                        info["segment"], reffiles["brftab"], info["x_offset"],
                        minmax_shifts, minmax_doppler)

        phdr["dqicorr"] = "COMPLETE"

    return dq_array

def dopplerParam (info, disptab, doppcorr):
    """Return the appropriate set of Doppler keyword values.

    @param info: keywords and values
    @type info: dictionary
    @param disptab: name of dispersion relation table
    @type disptab: string
    @param doppcorr: if Doppler correction is OMIT or SKIPPED, return
        dummy values
    @type doppcorr: string

    @return: Doppler magnitude in pixels, time (MJD) when the Doppler shift
        is zero and increasing, period (seconds) of HST; different keywords
        will be used depending on whether the data are TIME-TAG or ACCUM
    @rtype: tuple
    """

    if doppcorr == "OMIT" or doppcorr == "SKIPPED":
        doppmag  = 0.
        doppzero = info["expstart"]
        orbitper = 5760.
    elif info["obsmode"] == "TIME-TAG":
        if info["tc2_2"] == 1.:                 # default value
            # tc2_2 is the dispersion.  If its value is the default,
            # compute the dispersion from the central wavelength and
            # the dispersion coefficients.
            # xxx this section should only be needed temporarily xxx
            cosutil.printWarning ("TC2_2 keyword has the default value.")
            filter = {"opt_elem": info["opt_elem"],
                      "cenwave": info["cenwave"],
                      "aperture": info["aperture"]}
            if info["detector"] == "FUV":
                filter["segment"] = info["segment"]
                middle = float (FUV_X) / 2.
            else:
                filter["segment"] = "NUVB"
                middle = float (NUV_X) / 2.
            disp_rel = dispersion.Dispersion (disptab, filter, False)
            if not disp_rel.isValid():
                raise RuntimeError, "missing row in disptab"
            # get the dispersion (disp) at the middle of the detector
            disp = disp_rel.evalDerivDisp (middle)
            disp_rel.close()
        else:
            disp = info["tc2_2"]
        # Compute the Doppler shift in pixels from the shift in km/s.
        doppmag = (info["doppmagv"] / SPEED_OF_LIGHT) * (info["cenwave"] / disp)
        doppzero = info["doppzero"]
        orbitper = info["orbitper"]

    else:               # ACCUM
        doppmag  = info["dopmagt"]
        doppzero = info["dopzerot"]
        orbitper = info["orbtpert"]

    return (doppmag, doppzero, orbitper)

def doDoppcorr (events, info, switches, reffiles, phdr):
    """Apply Doppler correction to the x and y pixel coordinates.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    """

    if info["obsmode"] == "ACCUM":              # done on-board
        return

    if info["obstype"] == "SPECTROSCOPIC":
        cosutil.printSwitch ("DOPPCORR", switches)

    if switches["doppcorr"] == "PERFORM" or switches["doppcorr"] == "COMPLETE":

        # xi and eta are the columns of pixel coordinates for the
        # dispersion and cross-dispersion directions respectively.
        # (explicit column names are used here for clarity)
        xi = events.field ("xcorr")
        eta = events.field ("ycorr")
        dopp = events.field ("xdopp")
        xi_full  = events.field ("xfull")

        cosutil.printRef ("XTRACTAB", reffiles)
        cosutil.printRef ("DISPTAB", reffiles)
        if info["detector"] == "FUV":
            cosutil.printRef ("BRFTAB", reffiles)

        xtractab = reffiles["xtractab"]
        disptab = reffiles["disptab"]
        wcptab = reffiles["wcptab"]
        if info["detector"] == "FUV":
            # This array of flags indicates which events should be corrected.
            region_flags = fuvDopplerRegions (eta, info, xtractab)
            # Apply the orbital Doppler correction to the flagged events.
            dopp[:] = np.where (region_flags, \
                                dopplerCorrection (events.field ("time"),
                                                   xi, info, reffiles),
                                xi)
        else:
            region_flags_dict = nuvPsaRegions (eta, info, xtractab)
            dopp[:] = xi
            for stripe in ["NUVA", "NUVB", "NUVC"]:
                dopp[:] = np.where (region_flags_dict[stripe], \
                                    dopplerCorrection (events.field ("time"),
                                           xi, info, reffiles, stripe=stripe),
                                    dopp)

        # Copy to xfull if wavecal processing will not be done.
        if switches["wavecorr"] == "OMIT" and not info["corrtag_input"]:
            xi_full[:] = dopp.copy()

        phdr["doppcorr"] = "COMPLETE"

def fuvDopplerRegions (eta, info, xtractab):
    """Determine the region over which Doppler shift should be applied.

    This version is for FUV data.

    @param eta: pixel coordinates in cross-dispersion direction
    @type eta: array
    @param info: keywords and values
    @type info: dictionary
    @param xtractab: name of spectral extraction parameters reference table
    @type xtractab: string

    @return: True for events that are within the region for which it would be
        appropriate to apply Doppler correction
    @rtype: Boolean array
    """

    global active_area

    region_flags = active_area.copy()

    # Protect against the possibility that the aperture keyword is "WCA".
    if info["aperture"] == "BOA":
        aperture = "BOA"
    else:
        aperture = "PSA"
    filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"],
              "segment": info["segment"], "aperture": aperture}
    middle = float (FUV_X) / 2.

    # The computation of the 'boundary' variable makes an assumption
    # about the relative locations of the PSA and WCA regions on the
    # detectors.  The PSA spectral region is at lower Y pixel numbers.

    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_psa = xtract_info.field ("b_spec")[0] + \
                 xtract_info.field ("slope")[0] * middle

    filter["aperture"] = "WCA"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_wca = xtract_info.field ("b_spec")[0] + \
                 xtract_info.field ("slope")[0] * middle

    boundary = int (round ((b_spec_psa + b_spec_wca) / 2.))

    region_flags &= (eta < boundary)

    return region_flags

def dopplerCorrection (time, xi, info, reffiles, stripe=None):
    """Apply orbital and heliocentric Doppler correction.

    @param time: times of events (seconds)
    @type time: numpy array
    @param xi: pixel coordinates of events, in dispersion direction
    @type xi: numpy array
    @param info: keywords and values
    @type info: dictionary
    @param reffiles: dictionary of reference file names
    @type reffiles: dictionary
    @param stripe: name of NUV stripe ("NUVA", "NUVB", "NUVC"), or None for FUV
    @type stripe: string

    @return: array of Doppler-corrected X pixel coordinates
    @rtype: numpy array
    """

    disptab = reffiles["disptab"]

    # Compute the wavelength and dispersion at each pixel.
    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": info["aperture"],
              "fpoffset": info["fpoffset"]}
    if stripe is None:
        filter["segment"] = info["segment"]
    else:
        filter["segment"] = stripe
    disp_rel = dispersion.Dispersion (disptab, filter, use_fpoffset=True)
    if not disp_rel.isValid():
        disp_rel.close()
        raise RuntimeError, "missing row in disptab"

    xi = xi.astype (np.float64)
    fpoffset_present = cosutil.findColumn (disptab, "fpoffset")
    if fpoffset_present:
        # Compute wavelength and dispersion at each element of xi.
        wavelength = disp_rel.evalDisp (xi)
        disp = disp_rel.evalDerivDisp (xi)
    else:
        # Correct for fpoffset when computing wavelength and dispersion
        # (a feature will be at larger pixel number if fpoffset is larger,
        # so the wavelength at a given pixel will be smaller).
        wcp_info = cosutil.getTable (reffiles["wcptab"],
                                     filter={"opt_elem": info["opt_elem"]},
                                     exactly_one=True)
        stepsize = wcp_info.field ("stepsize")[0]
        wavelength = disp_rel.evalDisp (xi)
        xi_temp = xi - info["fpoffset"] * stepsize
        wavelength = disp_rel.evalDisp (xi_temp)
        disp = disp_rel.evalDerivDisp (xi_temp)
        del xi_temp, wcp_info
    disp_rel.close()

    # Apply the Doppler correction to the pixel coordinates.
    xd = orbitalDoppler (time, xi, wavelength, disp, info["expstart"],
                         info["doppmagv"], info["doppzero"], info["orbitper"])

    return xd

def orbitalDoppler (time, xi, wavelength, dispersion, expstart,
                    doppmag_v, doppzero, orbitper):
    """Apply Doppler correction for HST orbital motion.

    @param time: times of events (seconds)
    @type time: numpy array
    @param xi: pixel coordinates of events, in dispersion direction
    @type xi: numpy array
    @param wavelength: wavelengths corresponding to xi (Angstroms)
    @type wavelength: numpy array
    @param dispersion: dispersion at each element of xi (Angstroms/pixel)
    @type dispersion: numpy array
    @param expstart: exposure start time (MJD)
    @type expstart: float
    @param doppmag_v: magnitude of Doppler shift (km/s)
    @type doppmag_v: float
    @param doppzero: time when orbital Doppler shift is zero and increasing
        (MJD)
    @type doppzero: float
    @param orbitper: orbital period of HST (seconds)
    @type orbitper: float
    """

    # t is the time of each event in seconds since doppzero.
    t = (expstart - doppzero) * SEC_PER_DAY + time.astype (np.float64)

    shift = doppmag_v / SPEED_OF_LIGHT * wavelength / dispersion * \
            np.sin (2. * np.pi * t / orbitper)

    return xi - shift

def initHelcorr (events, info, switches, hdr):
    """Compute the radial velocity and update the V_HELIO keyword.

    @param events: the data unit containing the events table
    @type events: record array
    @param info: dictionary of header keywords and values
    @type info: dictionary
    @param reffiles: dictionary of reference file names
    @type reffiles: dictionary
    @param hdr: the events extension header
    @type hdr: pyfits Header object
    """

    if info["obstype"] != "SPECTROSCOPIC":
        return

    # get midpoint of exposure, MJD
    expstart = info["expstart"]
    time = events.field ("time")
    t_mid = expstart + (time[0] + time[len(time)-1]) / 2. / SEC_PER_DAY

    # Compute radial velocity and heliocentric correction factor (the latter
    # is actually not used here).
    radvel = heliocentricVelocity (t_mid, info["ra_targ"], info["dec_targ"])
    helio_factor = -radvel  / SPEED_OF_LIGHT
    hdr.update ("v_helio", radvel)
    info["v_helio"] = radvel

def heliocentricVelocity (t, ra_targ, dec_targ):
    """Compute heliocentric radial velocity.

    This is copied from the code for calstis, except that the target
    coordinates will not be precessed to the time of observation.

    @param t: time (MJD)
    @type t: float
    @param ra_targ: right ascension of the target (J2000)
    @type ra_targ: float
    @param dec_targ: declination of the target (J2000)
    @type dec_targ: float

    @return: the contribution of the Earth's velocity around the Sun to the
        radial velocity of the target, in km/s; if the Earth is approaching
        the target, this will be negative (i.e. the sign convention is that
        radial velocity is positive if the distance between the Earth and
        the target is increasing)
    @rtype: float
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

    target[0] = math.cos (dec) * math.cos (ra)
    target[1] = math.cos (dec) * math.sin (ra)
    target[2] = math.sin (dec)

    # Precess the target coordinates to time t.
    # target = cosutil.precess (t, target)        # note, commented out

    dt = t - REFDATE                    # days since 2000 Jan 1, 12h UT

    g_dot = 0.9856003 * deg_to_rad
    l_dot = 0.9856474 * deg_to_rad

    eps = (23.439 - 0.0000004 * dt) * deg_to_rad

    g = mod2pi ((357.528 + 0.9856003 * dt) * deg_to_rad)
    l = mod2pi ((280.461 + 0.9856474 * dt) * deg_to_rad)

    #       L   1.915 deg                 0.02 deg
    elong = l + 0.033423 * math.sin (g) + 0.000349 * math.sin (2.*g)
    elong_dot = l_dot + \
                0.033423 * math.cos (g) * g_dot + \
                0.000349 * math.cos (2.*g) * 2.*g_dot

    radius = 1.00014 - 0.01671 * math.cos (g) - 0.00014 * math.cos (2.*g)
    radius_dot =       0.01671 * math.sin (g) * g_dot + \
                       0.00014 * math.sin (2.*g) * 2.*g_dot

    x_dot = radius_dot * math.cos (elong) - \
                radius * math.sin (elong) * elong_dot

    y_dot = radius_dot * math.cos (eps) * math.sin (elong) + \
                radius * math.cos (eps) * math.cos (elong) * elong_dot

    z_dot = radius_dot * math.sin (eps) * math.sin (elong) + \
                radius * math.sin (eps) * math.cos (elong) * elong_dot

    velocity[0] = -x_dot * KM_AU / SEC_DAY
    velocity[1] = -y_dot * KM_AU / SEC_DAY
    velocity[2] = -z_dot * KM_AU / SEC_DAY

    dot_product = velocity[0] * target[0] + \
                  velocity[1] * target[1] + \
                  velocity[2] * target[2]
    radvel = -dot_product

    return radvel

def mod2pi (x):
    """Return the argument modulo two pi."""

    (f, i) = math.modf (x / (2.*math.pi))
    if f < 0.:
        f += 1.
    return f * 2. * math.pi

def doFlatcorr (events, info, switches, reffiles, phdr, hdr):
    """Apply flat field correction.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    @param hdr: the events extension header
    @type hdr: pyfits Header object
    """

    cosutil.printSwitch ("FLATCORR", switches)

    if switches["flatcorr"] == "PERFORM":

        cosutil.printRef ("FLATFILE", reffiles)

        fd = pyfits.open (reffiles["flatfile"], mode="readonly")

        if info["detector"] == "NUV":
            hdu = fd[1]
        else:
            pharange = cosutil.getPulseHeightRange (hdr, info["segment"])
            # xxx this is temporary; eventually select image based on pharange
            ref_pharange = cosutil.tempPulseHeightRange (reffiles["flatfile"])
            cosutil.comparePulseHeightRanges (pharange, ref_pharange,
                                              reffiles["flatfile"])
            hdu = fd[(info["segment"],1)]
        flat = hdu.data

        origin_x = hdu.header.get ("origin_x", 0)
        origin_y = hdu.header.get ("origin_y", 0)

        if info["obsmode"] == "ACCUM":
            if info["obstype"] == "SPECTROSCOPIC":
                cosutil.printSwitch ("DOPPCORR", switches)
            if switches["doppcorr"] == "PERFORM" or \
               switches["doppcorr"] == "COMPLETE":
                convolveFlat (flat, info["dispaxis"], \
                     info["expstart"], info["exptime"],
                     info["dopmagt"], info["dopzerot"], info["orbtpert"])
                phdr["doppcorr"] = "COMPLETE"

        ccos.applyflat (events.field (xcorr), events.field (ycorr),
                        events.field ("epsilon"), flat, origin_x, origin_y)

        fd.close()

        phdr["flatcorr"] = "COMPLETE"

def convolveFlat (flat, dispaxis,
                expstart, exptime, dopmagt, dopzerot, orbtpert):
    """Convolve the flat field file with the Doppler smearing function.

    @param flat: flat field data array, modified in-place
    @type flat: numpy array
    @param dispaxis: dispersion axis (1 or 2)
    @type dispaxis: int
    @param expstart: exposure start time, MJD
    @type expstart: float
    @param exptime: exposure duration, seconds
    @type exptime: float
    @param dopmagt: magnitude of Doppler shift, pixels
    @type dopmagt: int
    @param dopzerot: time when Doppler shift is zero and increasing
    @type dopzerot: float
    @param orbtpert: orbital period of HST
    @type orbtpert: float
    """

    # Round dopmagt up to the next integer; mag is a zero-point offset.
    mag = int (math.ceil (dopmagt))

    # dopp will be the Doppler smoothing function, normalized so its sum is 1.
    dopp = np.zeros (2*mag+1, dtype=np.float32)

    # t is the time in seconds since dopzerot, in one second increments.
    t = np.arange (int (round (exptime)), dtype=np.float32) + \
               (expstart - dopzerot) * SEC_PER_DAY

    # shift is in pixels (wavelengths increase toward larger pixel number).
    shift = -dopmagt * np.sin (2. * np.pi * t / orbtpert)

    # Construct the Doppler smoothing function.
    npts = round (exptime)
    increment = 1. / npts
    npts = int (npts)
    for i in range (npts):                      # one-second increments
        ishift = int (round (shift[i])) + mag
        assert ishift >= 0 and ishift <= 2*mag
        dopp[ishift] += increment

    # Do the convolution (in-place).
    axis = 2 - dispaxis         # 1 --> 1,  2 --> 0
    ccos.convolve1d (flat, dopp, axis)

def doDeadcorr (events, input, info, switches, reffiles, phdr, hdr,
            stim_countrate, stim_livetime, livetimefile):
    """Correct for deadtime.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param input: name of raw file (for writing to livetimefile)
    @type input: string
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    @param hdr: the input extension header
    @type hdr: pyfits Header object
    @param livetimefile: name of output text file for livetime factors (or None)
    @type livetimefile: string
    @param stim_countrate: the observed count rate for a stim (for info)
    @type stim_countrate: float
    @param stim_livetime: live time computed from the stim rate
    @type stim_livetime: float
    """

    cosutil.printSwitch ("DEADCORR", switches)
    if switches["deadcorr"] == "PERFORM":
        cosutil.printRef ("DEADTAB", reffiles)
        if info["obsmode"] == "TIME-TAG":
            (dead_rate, dead_method, avg_livetime) = \
                deadtimeCorrection (events, reffiles["deadtab"], info,
                                    stim_countrate, stim_livetime,
                                    input, livetimefile)
        else:
            (dead_rate, dead_method, avg_livetime) = \
                deadtimeCorrectionAccum (events, reffiles["deadtab"], info,
                                         stim_countrate, stim_livetime,
                                         input, livetimefile)
        updateDeadtimeKeywords (hdr, info["segment"],
                                dead_rate, dead_method, avg_livetime)
        phdr["deadcorr"] = "COMPLETE"

def updateDeadtimeKeywords (hdr, segment,
                            dead_rate, dead_method, avg_livetime):
    """Assign values to keywords pertaining to the deadtime correction.

    @param hdr: the first extension header, updated in-place
    @type hdr: pyfits Header object
    @param segment: FUVA, FUVB, or N/A for NUV
    @type segment: string
    @param dead_rate: the count rate that was used for determining the
        livetime factor
    @type dead_rate: float
    @param dead_method: a string that indicates which method was used for
        determining the livetime factor
    @type dead_method: string
    @param avg_livetime: the livetime factor that was applied to the data,
        or the average of the factors if the actual count rate was used
        for time-tag data
    @type avg_livetime: float
    """

    if segment == "FUVA":
        hdr.update ("deadrt_a", dead_rate)
        hdr.update ("deadmt_a", dead_method)
        hdr.update ("livetm_a", avg_livetime)
    elif segment == "FUVB":
        hdr.update ("deadrt_b", dead_rate)
        hdr.update ("deadmt_b", dead_method)
        hdr.update ("livetm_b", avg_livetime)
    else:
        hdr.update ("deadrt", dead_rate)
        hdr.update ("deadmt", dead_method)
        hdr.update ("livetm", avg_livetime)

def deadtimeCorrection (events, deadtab, info,
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

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param deadtab: name of reference table of count rates and livetime factors
    @type deadtab: string
    @param info: header keywords and values
    @type info: dictionary
    @param stim_countrate: the observed count rate for the stims
    @type stim_countrate: float
    @param stim_livetime: livetime computed from the stims
    @type stim_livetime: float
    @param input: name of input raw file (for writing to livetimefile)
    @type input: string
    @param livetimefile: name of output text file for livetime factors (or None)
    @type livetimefile: string

    @return: the count rate used for determining the livetime factor, a
        string that indicates which method was used for determining the
        livetime factor, and the average livetime factor that was used
    @rtype: tuple
    """

    if livetimefile is None:
        fd = None
    else:
        fd = open (livetimefile, "a")

    # dec_countrate is the count rate from the digital event counter.
    segment = info["segment"]
    dec_countrate = info["countrate"]

    time = cosutil.getColCopy (data=events, column="time")
    epsilon = events.field ("epsilon")
    nevents = len (time)

    live_info = cosutil.getTable (deadtab, filter={"segment": segment},
                                  at_least_one=True)
    # These are the values in the deadtab table columns.
    obs_rate = live_info.field ("obs_rate")
    live_factor = live_info.field ("livetime")

    # This livetime value is based on count rate over the entire exposure.
    if time[nevents-1] > time[0]:
        actual_countrate = float (nevents) / (time[nevents-1] - time[0])
    else:
        actual_countrate = 0.
    actual_rate_livetime = cosutil.determineLivetime (actual_countrate,
                                                      obs_rate, live_factor)

    # dec_countrate is from DEVENTA, DEVENTB or from MEVENTS.
    dec_livetime = cosutil.determineLivetime (dec_countrate,
                                              obs_rate, live_factor)

    print_details = (cosutil.checkVerbosity (VERY_VERBOSE))     # initial value
    if abs (dec_livetime - actual_rate_livetime) > \
            LIVETIME_CRITERION * actual_rate_livetime:
        cosutil.printWarning ("livetime estimates differ.")
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
        printLiveInfo (segment, stim_countrate, stim_livetime,
                       actual_countrate, actual_rate_livetime,
                       dec_countrate, dec_livetime, livetime_source)
    if fd is not None:
        printLiveInfo (segment, stim_countrate, stim_livetime,
                       actual_countrate, actual_rate_livetime,
                       dec_countrate, dec_livetime, livetime_source, fd=fd)

    if use_actual_rate:

        if fd is not None:
            fd.write ("# %s\n" % input)
            fd.write ("# t0 t1 countrate livetime\n")

        # Use counts over dt_deadtime seconds to compute livetime.
        fd_dead = pyfits.open (deadtab, mode="readonly")
        dt_deadtime = fd_dead[1].header["timestep"]
        fd_dead.close()
        cosutil.printMsg ("Compute livetime factor; timestep is %.6g s:" \
                      % dt_deadtime, VERY_VERBOSE)

        t0 = time[0]
        t1 = t0 + dt_deadtime
        last_time = time[nevents-1]
        cosutil.printMsg ("  time range    rate   livetime", VERY_VERBOSE)
        last_livetime = 1.      # use this for saving previous value
        sum_livetime = 0.       # for computing average livetime factor
        wgt_livetime = 0.
        countrate = 0.
        first = True
        while t0 < last_time:

            # time[i:j] matches t0 to t1.
            try:
                (i, j) = ccos.range (time, t0, t1)
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
                livetime = cosutil.determineLivetime (countrate,
                                                      obs_rate, live_factor)
                sum_livetime += livetime * (t1 - t0)
                wgt_livetime += (t1 - t0)
            elif t0 < last_time:
                t1_for_printing = last_time
                if (last_time - t0) < 0.5 * dt_deadtime and not first:
                    livetime = last_livetime
                    cosutil.printMsg ("Last time interval is short (%.6g s),"
                                      " so previous livetime will be used." %
                                      (last_time - t0,))
                else:
                    countrate = (j - i) / (last_time - t0)
                    livetime = cosutil.determineLivetime (countrate,
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
                fd.write ("%.0f %.0f %.6g %.6g\n" %
                          (t0, t1_for_printing, countrate, livetime))
            cosutil.printMsg ("%6.1f %6.1f   %.6g %.6g" %
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

def deadtimeCorrectionAccum (events, deadtab, info,
                             stim_countrate, stim_livetime,
                             input, livetimefile):
    """Determine and apply the livetime factor for ACCUM data.

    If there are subarrays, the livetime factor is gotten from the digital
    event counter.  If there are no subarrays, the livetime factor is based
    on the actual count rate.

    @param events: the data unit containing the events table
    @type events: pyfits record array
    @param deadtab: name of reference table of count rates and livetime factors
    @type deadtab: string
    @param info: header keywords and values
    @type info: dictionary
    @param stim_countrate: the observed count rate for the stims
    @type stim_countrate: float
    @param stim_livetime: livetime computed from the stims
    @type stim_livetime: float
    @param input: name of input raw file (for writing to livetimefile)
    @type input: string
    @param livetimefile: name of output text file for livetime factors (or None)
    @type livetimefile: string

    @return: the count rate used for determining the livetime factor, a
        string that indicates which method was used for determining the
        livetime factor, and the livetime factor that was used
    @rtype: tuple
    """

    if livetimefile is None:
        fd = None
    else:
        fd = open (livetimefile, "a")
        fd.write ("# %s\n" % (input,))

    # This is the column that will be modified in-place.
    epsilon = events.field ("epsilon")
    ncounts = len (epsilon)

    live_info = cosutil.getTable (deadtab, filter={"segment": info["segment"]},
                                  at_least_one=True)
    obs_rate = live_info.field ("obs_rate")
    live_factor = live_info.field ("livetime")

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
    dec_livetime = cosutil.determineLivetime (dec_countrate,
                                              obs_rate, live_factor)

    if info["exptime"] <= 0.:
        cosutil.printWarning ("Can't do deadcorr, exptime = %.6g." %
                              info["exptime"])
        return (0., "SKIPPED")
    actual_countrate = float (ncounts) / info["exptime"]
    actual_rate_livetime = cosutil.determineLivetime (actual_countrate,
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

    print_details = (cosutil.checkVerbosity (VERY_VERBOSE))     # initial value

    if abs (dec_livetime - actual_rate_livetime) > \
            LIVETIME_CRITERION * actual_rate_livetime:
        cosutil.printWarning ("livetime estimates differ.")
        print_details = True

    if print_details:
        cosutil.printMsg ("  actual countrate and livetime:  %.6g, %6.4f" % \
                          (actual_countrate, actual_rate_livetime))
        cosutil.printMsg ("  countrate and livetime from %s:  %.6g, %6.4f" % \
                          (keyword, dec_countrate, dec_livetime))
        cosutil.printMsg ("Livetime %6.4f is based on %s." % \
                          (livetime, livetime_source))
        if info["detector"] == "FUV":
            if stim_countrate is None:
                cosutil.printMsg (
                "  stim countrate and livetime could not be determined")
            else:
                cosutil.printMsg (
                "  stim countrate and livetime:  %.6g, %6.4f" % \
                                  (stim_countrate, stim_livetime))

    if fd is not None:
        fd.write ("actual countrate and livetime:  %.6g, %6.4f\n" %
                  (actual_countrate, actual_rate_livetime))
        fd.write ("countrate and livetime from %s:  %.6g, %6.4f\n" %
                  (keyword, dec_countrate, dec_livetime))
        fd.write ("livetime %6.4f is based on %s.\n" % \
                  (livetime, livetime_source))
        if info["detector"] == "FUV":
            if stim_countrate is None:
                fd.write (
                "stim countrate and livetime could not be determined\n")
            else:
                fd.write ("stim countrate and livetime:  %.6g, %6.4f\n" %
                          (stim_countrate, stim_livetime))

    if fd is not None:
        fd.close()

    return (dead_rate, dead_method, livetime)

def printLiveInfo (segment, stim_countrate, stim_livetime,
                   actual_countrate, actual_rate_livetime,
                   dec_countrate, dec_livetime, livetime_source, fd=None):
    """Print or write information about livetime.

    @param segment: segment name (for setting keyword name for DEC count rate)
    @type segment: string
    @param stim_countrate: the observed count rate for the stims, or None
    @type stim_countrate: float
    @param stim_livetime: livetime factor computed from the input and observed
                      stim rate
    @type stim_livetime: float
    @param actual_countrate: observed count rate, from events table
    @type actual_countrate: float
    @param actual_rate_livetime: livetime factor derived from countrate
    @type actual_rate_livetime: float
    @param dec_countrate: the count rate from the digital event counter
    @type dec_countrate: float
    @param dec_livetime: livetime factor computed from dec_countrate
    @type dec_livetime: float
    @param livetime_source: a string saying whether actual rate or DEC was used
                      for computing the livetime factor
    @type livetime_source: string
    @param fd: None if printing to trailer; an fd for printing to a log file
    @type fd: int
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
            messages.append (
                "stim countrate and livetime could not be determined")
        else:
            messages.append ("stim countrate and livetime:  %.6g, %6.4f" %
                             (stim_countrate, stim_livetime))
    messages.append ("actual (average) event rate and livetime:  %.6g, %6.4f" %
                     (actual_countrate, actual_rate_livetime))
    messages.append ("countrate and livetime from %s:  %.6g, %6.4f" %
                     (keyword, dec_countrate, dec_livetime))
    messages.append ("Livetime is based on %s." % livetime_source)

    if fd is None:
        for msg in messages:
            cosutil.printMsg (msg)
    else:
        fd.write ("\n")
        for msg in messages:
            fd.write (msg + "\n")

def writeNull (input, output, outcounts, outcsum,
               cl_args, info, phdr, headers):
    """Write output files; images will have null data portions.

    The outtag file has already been written, so we only need to write
    the output and outcounts files.

    @param input: name of the input file
    @type input: string
    @param output: name of the output file for flat-fielded count-rate image
    @type output: string
    @param outcounts: name of the output file for count-rate image
    @type outcounts: string
    @param outcsum: name of the output image for OPUS to add to cumulative
        image (or None)
    @type outcsum: string
    @param cl_args: some of the command-line arguments
    @type cl_args: dictionary
    @param info: header keywords and values
    @type info: dictionary
    @param phdr: primary header
    @type phdr: pyfits Header object
    @param headers: headers
    @type headers: list of pyfits Header objects
    """

    cosutil.printWarning ("No data in " + input)
    makeImage (outcounts, phdr, headers, None, None, None)
    makeImage (output, phdr, headers, None, None, None)
    if outcsum is not None:
        # pha has to be not None to get the correct dimensions for FUV.
        writeCsum (outcsum, None, None, None, np.zeros (1, dtype=np.int8),
                   info["detector"], info["subarray"],
                   phdr, headers[1],
                   cl_args["binx"],
                   cl_args["biny"],
                   cl_args["compress_csum"],
                   cl_args["compression_parameters"])

def writeImages (x, y, epsilon, dq,
                 phdr, headers, dq_array, npix, x_offset, exptime,
                 outcounts=None, output=None):
    """Bin events to images, and write to output files.

    @param x: X pixel coordinates of events
    @type x: numpy array
    @param y: Y pixel coordinates of events
    @type y: numpy array
    @param epsilon: weight column
    @type epsilon: numpy array
    @param dq: data quality column
    @type dq: numpy array
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    @param headers: the input headers
    @type headers: list of pyfits Header objects
    @param dq_array: the data quality array
    @type dq_array: numpy array
    @param npix: the array shape (ny, nx)
    @type npix: tuple
    @param x_offset: offset of the detector in a calibrated image
    @type x_offset: int
    @param exptime: the exposure time
    @type exptime: float
    @param outcounts: name of the output file for count-rate image
    @type outcounts: string
    @param output: name of the output file for flat-fielded count-rate image
    @type output: string
    """

    global serious_dq_flags

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

    # First make an image array in which each input event counts as one,
    # i.e. ignoring flat field and deadtime corrections.
    C_rate = np.zeros (npix, dtype=np.float32)

    if exptime <= 0:
        cosutil.printWarning (
                "Exposure time is zero, so output files are dummy.")
        E_rate = C_rate.copy()
        errE_rate = C_rate.copy()
        if outcounts is not None:
            makeImage (outcounts, phdr, headers, E_rate, errE_rate, dq_array)
        if output is not None:
            makeImage (output, phdr, headers, E_rate, errE_rate, dq_array)
        return

    ccos.binevents (x, y, C_rate, x_offset, dq, serious_dq_flags)

    errC_rate = np.sqrt (C_rate) / exptime

    if outcounts is not None:
        C_rate /= exptime
        makeImage (outcounts, phdr, headers, C_rate, errC_rate, dq_array)
    del C_rate                          # but we still need errC_rate

    if output is None:
        return                          # nothing further to do

    cosutil.printMsg ("writing file %s ..." % output, VERY_VERBOSE)

    # Make an image array where event number i has weight epsilon[i].
    E_rate = np.zeros (npix, dtype=np.float32)
    ccos.binevents (x, y, E_rate, x_offset, dq, serious_dq_flags, epsilon)

    # errC_rate will likely have a number of zero values, so we
    # have to set those to one before dividing.
    errC_rate = np.where (errC_rate == 0., 1., errC_rate)

    # convert from counts to count rate
    E_rate /= exptime
    errE_rate = E_rate / errC_rate / exptime
    del errC_rate

    makeImage (output, phdr, headers, E_rate, errE_rate, dq_array)

def makeImage (outimage, phdr, headers, sci_array, err_array, dq_array):
    """Write a FITS file, based on headers and data arrays.

    @param output: name of the output file to be written
    @type output: string
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    @param headers: the input headers
    @type headers: list of pyfits Header objects
    @param sci_array: the science data array (may be None)
    @type sci_array: numpy array
    @param err_array: the error estimates array (may be None)
    @type err_array: numpy array
    @param dq_array: the data quality array (may be None)
    @type dq_array: numpy array
    """

    primary_hdu = pyfits.PrimaryHDU (header=phdr)
    fd = pyfits.HDUList (primary_hdu)
    fd[0].header["nextend"] = 3
    cosutil.updateFilename (fd[0].header, outimage)

    makeImageHDU (fd, headers[1], sci_array, name="SCI")
    makeImageHDU (fd, headers[2], err_array, name="ERR")
    makeImageHDU (fd, headers[3], dq_array, name="DQ")

    fd.writeto (outimage, output_verify='silentfix')

def makeImageHDU (fd, table_hdr, data_array, name="SCI"):
    """Make an image hdu from data and a table header and append to fd.

    @param fd: pyfits object for FITS file (new hdu will be appended)
    @type fd: pyfits HDUList object
    @param table_hdr: header for the input table
    @type table_hdr: pyfits Header object
    @param data_array: image data to be appended (may be None)
    @type data_array: numpy array
    @param name: name to be used for EXTNAME
    @type name: string
    """

    # Create an image header from the table header.
    imhdr = cosutil.tableHeaderToImage (table_hdr)
    if name == "DQ":
        imhdr.update ("BUNIT", "UNITLESS")
    else:
        imhdr.update ("BUNIT", "count /s")

    if data_array is not None:
        if imhdr.has_key ("npix1"):
            del (imhdr["npix1"])
        if imhdr.has_key ("npix2"):
            del (imhdr["npix2"])
        if imhdr.has_key ("pixvalue"):
            del (imhdr["pixvalue"])

    hdu = pyfits.ImageHDU (data=data_array, header=imhdr, name=name)
    fd.append (hdu)

def writeCsum (outcsum, xcorr, ycorr, epsilon, pha, detector, subarray,
               phdr, hdr,
               binx=None, biny=None,
               compress_csum=False,
               compression_parameters="gzip,-0.1"):
    """Write the "calcos sum" (csum) image.

    @param outcsum: name of output "calcos sum" file
    @type outcsum: string
    @param xcorr: column for X coordinates of events
    @type xcorr: numpy array
    @param ycorr: column for Y coordinates of events
    @type ycorr: numpy array
    @param epsilon: column of weights for events
    @type epsilon: numpy array
    @param pha: column for pulse height amplitudes, or None if detector is NUV
    @type pha: numpy array
    @param detector: "FUV" or "NUV"
    @type detector: string
    @param subarray: True if the exposure used one or more subarrays
    @type subarray: boolean
    @param phdr: primary header from input file
    @type phdr: pyfits Header object
    @param hdr: first extension (EVENTS) header from input file
    @type hdr: pyfits Header object
    @param binx: binning factor in the dispersion direction (or None for
        the default binning)
    @type binx: int
    @param biny: binning factor in the cross-dispersion direction (or None
        for the default binning)
    @type biny: int
    @param compress_csum: compress the csum image?
    @type compress_csum: boolean
    @param compression_parameters: compressionType and quantizeLevel (separated
        by a comma) for the call to pyfits.CompImageHDU; compressionType can
        be "rice", "gzip", or "hcompress", and quantizeLevel can be e.g. -0.1,
        which means the floating point values will be scaled to integers with
        spacing that corresponds to 0.1 dn (see the doc string for
        pyfits.CompImageHDU for more details)
    @type compression_parameters: string
    """

    # This is the number of possible values for the pulse height amplitude,
    # pha = 0..31.
    PHA_RANGE = 32

    cosutil.printMsg ("writing file %s ..." % outcsum, VERY_VERBOSE)

    primary_hdu = pyfits.PrimaryHDU (header=phdr)
    fd = pyfits.HDUList (primary_hdu)
    fd[0].header.update ("nextend", 1)
    fd[0].header.update ("filetype", "CALCOS SUM FILE")
    cosutil.updateFilename (fd[0].header, outcsum)

    # Copy the exposure time keywords to the output primary header.
    cosutil.copyExptimeKeywords (hdr, fd[0].header)

    # Copy the high-voltage keywords to the output primary header.
    cosutil.copyVoltageKeywords (hdr, fd[0].header, detector)

    # Copy the subarray keywords to the output primary header.
    cosutil.copySubKeywords (hdr, fd[0].header, subarray)

    if detector == "FUV":
        if binx is None or binx <= 0:
            binx = FUV_BIN_X
        if biny is None or biny <= 0:
            biny = FUV_BIN_Y
        nx = FUV_X // binx
        ny = FUV_Y // biny
        fd[0].header.update ("fuvbinx", binx)
        fd[0].header.update ("fuvbiny", biny)
    else:
        if binx is None or binx <= 0:
            binx = NUV_BIN_X
        if biny is None or biny <= 0:
            biny = NUV_BIN_Y
        nx = NUV_X // binx
        ny = NUV_Y // biny
        fd[0].header.update ("nuvbinx", binx)
        fd[0].header.update ("nuvbiny", biny)

    if compress_csum:
        (compType, quantLevel) = compression_parameters.split (",")
        compType = compType.upper() + "_1"
        quantLevel = float (quantLevel)
        if detector == "FUV":
            if pha is None:
                data = np.zeros ((ny, nx), dtype=np.float32)
                if xcorr is not None:
                    ccos.csum_2d (data, xcorr, ycorr, epsilon, binx, biny)
            else:
                data = np.zeros ((PHA_RANGE, ny, nx), dtype=np.float32)
                if xcorr is not None:
                    ccos.csum_3d (data, xcorr, ycorr, epsilon,
                               pha.astype(np.int16), binx, biny)
        else:
            data = np.zeros ((ny, nx), dtype=np.float32)
            if xcorr is not None:
                ccos.csum_2d (data, xcorr, ycorr, epsilon, binx, biny)
        fd[0].header.update ("counts", data.sum(dtype=np.float64))
        fd.append (pyfits.CompImageHDU (data, header=hdr, name="SCI",
                                        compressionType=compType,
                                        quantizeLevel=quantLevel))
        #  the arguments and their defaults are:
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
            if pha is None:
                fd.append (pyfits.ImageHDU (data=np.zeros ((ny, nx),
                                                           dtype=np.float32),
                                            header=hdr, name="SCI"))
                if xcorr is not None:
                    ccos.csum_2d (fd[1].data, xcorr, ycorr, epsilon, binx, biny)
            else:
                fd.append (pyfits.ImageHDU (data=np.zeros ((PHA_RANGE, ny, nx),
                                                           dtype=np.float32),
                                            header=hdr, name="SCI"))
                if xcorr is not None:
                    ccos.csum_3d (fd[1].data, xcorr, ycorr, epsilon,
                                  pha.astype(np.int16), binx, biny)
        else:
            fd.append (pyfits.ImageHDU (data=np.zeros ((ny, nx),
                                                       dtype=np.float32),
                                        header=hdr, name="SCI"))
            if xcorr is not None:
                ccos.csum_2d (fd[1].data, xcorr, ycorr, epsilon, binx, biny)
        fd[0].header.update ("counts", fd[1].data.sum(dtype=np.float64))

    fd[1].header.update ("BUNIT", "count")

    fd.writeto (outcsum, output_verify="silentfix")

def doStatflag (switches, output, outcounts):
    """Compute statistics and update keywords.

    @param switches: calibration switches
    @type switches: dictionary
    @param outflt: name of the output file for flat-fielded count-rate image
    @type outflt: string
    @param outcounts: name of the output file for count-rate image
    @type outcounts: string
    """

    cosutil.printSwitch ("STATFLAG", switches)
    if switches["statflag"] == "PERFORM":
        cosutil.doImageStat (outcounts)
        cosutil.doImageStat (output)

def appendShift1 (outtag, output, outcounts, shift1_vs_time=None):
    """For tagflash data, append a table of shift1 vs time.

    @param outtag: name of the output corrtag table
    @type outtag: string
    @param output: name of the output flat-fielded count rate image file
    @type output: string
    @param outcounts: name of the output count rate image file
    @type outcounts: string
    @param shift1_vs_time: shift in dispersion dir. at one-second intervals
    @type shift1_vs_time: array, or None
    """

    if shift1_vs_time is None or len (shift1_vs_time) < 1:
        return

    col = []
    col.append (pyfits.Column (name="SHIFT1", format="1E", unit="pixel",
                               array=shift1_vs_time))
    cd = pyfits.ColDefs (col)
    hdu = pyfits.new_table (cd)
    hdu.header.update ("EXTNAME", "INFO", after="TFIELDS")
    hdu.header.update ("EXTVER", 1, after="EXTNAME")

    fd = pyfits.open (outtag, mode="update")
    fd.append (hdu)
    fd[0].header.update ("nextend", len(fd)-1)
    fd.close()

    fd = pyfits.open (output, mode="update")
    fd.append (hdu)
    fd[0].header.update ("nextend", len(fd)-1)
    fd.close()

    fd = pyfits.open (outcounts, mode="update")
    fd.append (hdu)
    fd[0].header.update ("nextend", len(fd)-1)
    fd.close()

def flag_gti (time, dq, gti):
    """Flag events in dq that are outside any good time interval.

    @param time: the time column in the events table
    @type time: numpy array
    @param dq: the data quality column in the events table (updated in-place)
    @type dq: numpy array
    @param gti: list of good time intervals
    @type gti: list
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

def updateFromWavecal (events, wavecal_info,
                       shift_file,
                       info, switches, reffiles, phdr, hdr):
    """Update XFULL and YFULL based on auto or GO wavecal info.

    @param events: the data unit containing the events table
    @type events: record array
    @param wavecal_info: when wavecal exposures were processed, the results
        were stored in this dictionary
    @type wavecal_info: dictionary
    @param shift_file: if not None, this text file may have been used to
        override shift1; it's included here just to append its name to
        the WAVECALS string for the header and trailer
    @type shift_file: string
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the primary header (WAVECORR and WAVECALS can be updated)
    @type phdr: PyFITS Header object
    @param hdr: the events extension header (modified in-place)
    @type hdr: PyFITS Header object

    @return: the shifts in the dispersion direction at one-second intervals,
        or None if the current observation is a wavecal or if wavecal
        processing was not done.
    @rtype: array
    """

    global xcorr, ycorr, xdopp, ydopp, xfull, yfull
    global active_area

    # Read info from wavecal parameters table.
    wcp_info = cosutil.getTable (reffiles["wcptab"],
                       filter={"opt_elem": info["opt_elem"]},
                       exactly_one=True)
    wcp_info = wcp_info[0]

    xi  = events.field (xdopp)
    eta = events.field (ydopp)
    xi_full  = events.field (xfull)
    eta_full = events.field (yfull)

    # If the current exposure is a wavecal, or for a science exposure if
    # wavecal processing has not been done, there's nothing to do.
    if info["exptype"].find ("WAVE") >= 0 or not wavecal_info:
        return None

    # Get the shifts in dispersion and cross-dispersion directions at the
    # start of the exposure.  If the science exposure was bracketed by
    # two wavecals, the slope of the shifts can be non-zero.
    shift_info = wavecal.returnWavecalShift (wavecal_info,
                        wcp_info, info["fpoffset"], info["expstart"])
    if shift_info is None:
        return None

    (shift_dict, slope_dict, filename) = shift_info

    if info["detector"] == "FUV":
        segment_list = [info["segment"]]
    else:
        segment_list = ["NUVA", "NUVB", "NUVC"]
        psa_region_flags_dict = nuvPsaRegions (eta, info, reffiles["xtractab"])
        wca_region_flags_dict = nuvWcaRegions (eta, info, reffiles["xtractab"])

    time = events.field ("TIME")
    t0 = time[0]
    t_mid = (t0 + time[-1]) / 2.

    xi_full[:] = xi.copy()

    for segment in segment_list:

        key = "shift1" + segment[-1].lower()
        if not (shift_dict.has_key (key) and slope_dict.has_key (key)):
            cosutil.printError ("There is no wavecal for segment %s." % segment)
            return None
        shift1_zero = shift_dict[key]
        shift1_slope = slope_dict[key]
        if info["detector"] == "FUV":
            xi_full[:] = np.where (active_area,
                           xi - ((time - t0) * shift1_slope + shift1_zero),
                           xi_full)
        else:
            xi_full[:] = np.where (psa_region_flags_dict[segment],
                           xi - ((time - t0) * shift1_slope + shift1_zero),
                           xi_full)
            xi_full[:] = np.where (wca_region_flags_dict[segment],
                           xi - ((time - t0) * shift1_slope + shift1_zero),
                           xi_full)
        avg_shift1 = shift1_slope * t_mid + shift1_zero
        key = "SHIFT1" + segment[-1]
        hdr.update (key, round (avg_shift1, 4))

    if info["detector"] == "FUV":
        segment = segment_list[0]
    else:
        segment = "NUVB"

    key = "shift2" + segment[-1].lower()
    shift2_zero = shift_dict[key]
    shift2_slope = slope_dict[key]
    if info["detector"] == "FUV":
        eta_full[:] = np.where (active_area,
                        eta - ((time - t0) * shift2_slope + shift2_zero),
                        eta)
    else:
        # Use the same shift2 for every stripe.
        eta_full[:] = eta - ((time - t0) * shift2_slope + shift2_zero)

    # stripe B for NUV
    key = "shift1" + segment[-1].lower()
    shift1_zero = shift_dict[key]
    shift1_slope = slope_dict[key]
    avg_dy = shift2_slope * t_mid + shift2_zero

    # These are one-second time bins, so we add 0.5 second to the array t
    # so the values of t will be the times at the middle of each interval.
    nbins = int (math.ceil (time[-1] - time[0]))
    t = np.arange (nbins, dtype=np.float64) + t0 + 0.5
    shift1_vs_time = shift1_slope * t + shift1_zero

    # Set the SHIFT2[A-C] keywords to the average offset in the
    # cross-dispersion direction.
    # Set DPIXEL1[A-C] to the average of the difference xfull minus the
    # nearest integer to xfull, where xfull is the column of that name;
    # this will be used when assigning wavelengths in extract.py.

    for segment in segment_list:
        key = "SHIFT2" + segment[-1]
        hdr.update (key, round (avg_dy, 4))
        if info["detector"] == "FUV":
            xi_active = xi_full[active_area]
            xi_diff = xi_active - np.around (xi_active)
        else:
            xi_psa = xi_full[psa_region_flags_dict[segment]]
            xi_diff = xi_psa - np.around (xi_psa)
        dpixel1 = xi_diff.mean()
        key = "DPIXEL1" + segment[-1]
        hdr.update (key, round (dpixel1, 4))

    phdr["wavecorr"] = "COMPLETE"
    filename = cosutil.changeSegment (filename, info["detector"],
                                      info["segment"])
    if shift_file is not None:
        filename = filename + " " + shift_file
    phdr.update ("wavecals", filename)
    cosutil.printMsg ("Wavecal file(s) '%s'" % filename, VERBOSE)

    return shift1_vs_time

def computeWavelengths (events, info, reffiles, helcorr="OMIT", hdr=None):
    """Compute wavelengths for a corrtag table.

    @param events: the data unit containing the events table
    @type events: record array
    @param info: header keywords and values
    @type info: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param helcorr: if helcorr is PERFORM or COMPLETE, wavelengths should be
        corrected for heliocentric velocity (helcorr in header will not be
        modified, however); the default value is appropriate for a wavecal
    @type helcorr: string
    @param hdr: if not None, apply shift1[abc] and shift2[abc] to the pixel
        coordinates; this is needed for a wavecal exposure
    @type hdr: pyfits Header object, or None
    """

    # If the current exposure is a wavecal, we need to apply the shift keyword
    # values to the pixel coordinates.  In that case, hdr is supplied so we
    # can read the keywords.
    use_shift_keywords = (hdr is not None)

    # Heliocentric correction is not relevant for wavecals and won't be done.
    if hdr is None:
        message = "%-9s %s for computing wavelengths for the corrtag table" \
                   % ("HELCORR", helcorr)
        cosutil.printMsg (message, VERBOSE)

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
            shift1 = hdr.get (keyword, default=0.)
            keyword = "shift2" + segment[-1]
            shift2 = hdr.get (keyword, default=0.)
            shift1_dict[segment] = shift1
            shift2_dict[segment] = shift2

    # aperture and segment will be added to the filter within the loop.
    filter = {"opt_elem": opt_elem,
              "cenwave": cenwave}

    wavelength = events.field ("WAVELENGTH")

    # The YFULL position is used to determine which stripe a given
    # event corresponds to.
    if use_shift_keywords:
        xi = events.field (xfull).copy()        # because we need to modify it
        if detector == "FUV":
            shift2b = shift2_dict[segment_list[0]]
        else:
            shift2b = shift2_dict["NUVB"]
        eta = events.field (yfull).copy() - shift2b
    else:
        xi = events.field (xfull)
        eta = events.field (yfull)

    if detector == "FUV":
        setActiveArea (events, info, reffiles["brftab"])
        (psa_region_flags, wca_region_flags) = \
                        fuvPsaWcaRegions (eta, info, xtractab)
    else:
        psa_region_flags_dict = nuvPsaRegions (eta, info, xtractab)
        wca_region_flags_dict = nuvWcaRegions (eta, info, xtractab)

    for segment in segment_list:
        # Compute the wavelengths for the output table.
        filter["segment"] = segment
        filter["aperture"] = "PSA"
        psa_disp_rel = dispersion.Dispersion (disptab, filter)
        filter["aperture"] = "WCA"
        wca_disp_rel = dispersion.Dispersion (disptab, filter)
        if not (psa_disp_rel.isValid() and wca_disp_rel.isValid()):
            cosutil.printError ("Matching row in disptab %s was not found" \
                                % disptab)
            cosutil.printContinuation (
                "can't compute wavelengths for corrtag file.")
            continue
        if detector == "FUV":
            if use_shift_keywords:
                xi -= shift1_dict[segment]
            psa_wavelength = psa_disp_rel.evalDisp (xi)
            # "hdr is None" means the current exposure is not a wavecal
            if hdr is None and (helcorr == "PERFORM" or helcorr == "COMPLETE"):
                psa_wavelength += (psa_wavelength *
                                   (-info["v_helio"]) / SPEED_OF_LIGHT)
            wavelength[:] = np.where (psa_region_flags,
                                      psa_wavelength, wavelength)
            del psa_wavelength
            wca_wavelength = wca_disp_rel.evalDisp (xi)
            wavelength[:] = np.where (wca_region_flags,
                                      wca_wavelength, wavelength)
            del wca_wavelength
        else:
            if use_shift_keywords:
                xi_full = xi - shift1_dict[segment]
            else:
                xi_full = xi
            # Update the wavelength array for those events that are within
            # the PSA and WCA regions for the current stripe.
            psa_wavelength = psa_disp_rel.evalDisp (xi_full)
            if hdr is None and (helcorr == "PERFORM" or helcorr == "COMPLETE"):
                psa_wavelength += (psa_wavelength *
                                   (-info["v_helio"]) / SPEED_OF_LIGHT)
            wavelength[:] = np.where (psa_region_flags_dict[segment],
                                      psa_wavelength, wavelength)
            del psa_wavelength
            wca_wavelength = wca_disp_rel.evalDisp (xi_full)
            wavelength[:] = np.where (wca_region_flags_dict[segment],
                                      wca_wavelength, wavelength)
            del wca_wavelength
        psa_disp_rel.close()
        wca_disp_rel.close()

    return

def fuvPsaWcaRegions (eta, info, xtractab):
    """Determine the sets of events within the PSA and WCA.

    This version is for FUV data.

    @param eta: pixel coordinates in cross-dispersion direction
    @type eta: array
    @param info: keywords and values
    @type info: dictionary
    @param xtractab: name of spectral extraction parameters reference table
    @type xtractab: string

    @return: The values in the first array are True for events that are within
        the active area and also within the PSA region (i.e. below the midpoint
        between PSA and WCA).  The second array values are True for events
        that are within the active area and also within the WCA region.
    @rtype: tuple of two Boolean arrays
    """

    global active_area

    psa_region_flags = active_area.copy()
    wca_region_flags = active_area.copy()

    filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"],
              "segment": info["segment"]}       # aperture added below
    middle = float (FUV_X) / 2.

    # The computation of the 'boundary' variable makes an assumption
    # about the relative locations of the PSA and WCA regions on the
    # detectors.  The PSA spectral region is at lower Y pixel numbers.

    filter["aperture"] = "PSA"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_psa = xtract_info.field ("b_spec")[0] + \
                 xtract_info.field ("slope")[0] * middle

    filter["aperture"] = "WCA"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_wca = xtract_info.field ("b_spec")[0] + \
                 xtract_info.field ("slope")[0] * middle

    boundary = int (round ((b_spec_psa + b_spec_wca) / 2.))

    psa_region_flags &= (eta < boundary)
    wca_region_flags &= (eta >= boundary)

    return (psa_region_flags, wca_region_flags)

def nuvPsaRegions (eta, info, xtractab):
    """Determine the set of events within the NUV regions for the PSA.

    This is only used for NUV data.

    @param eta: pixel coordinates in cross-dispersion direction
    @type eta: array
    @param info: keywords and values
    @type info: dictionary
    @param xtractab: name of spectral extraction parameters reference table
    @type xtractab: string

    @return: dictionary with stripe name ("NUVA", "NUVB", "NUVC") as the key
        and an array of Boolean flags as the value, true for events for which
        the Y coordinate is within the regions for the PSA.
    @rtype: dictionary of Boolean arrays
    """

    # segment will be added to the filter below.
    filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"],
              "aperture": "PSA"}
    middle = float (NUV_X) / 2.

    # b_spec_a, b_spec_b, b_spec_c, are the locations (at the middle of the
    # detector) of stripes A, B, C for the PSA, and b_spec_wca is the location
    # of stripe A for the WCA.

    filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_a = xtract_info.field ("b_spec")[0] + \
               xtract_info.field ("slope")[0] * middle

    filter["segment"] = "NUVB"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_b = xtract_info.field ("b_spec")[0] + \
               xtract_info.field ("slope")[0] * middle

    filter["segment"] = "NUVC"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_c = xtract_info.field ("b_spec")[0] + \
               xtract_info.field ("slope")[0] * middle

    filter["segment"] = "NUVA"
    filter["aperture"] = "WCA"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_wca = xtract_info.field ("b_spec")[0] + \
                 xtract_info.field ("slope")[0] * middle

    # Set boundaries midway between adjacent stripes.
    boundary_a_b = round ((b_spec_a + b_spec_b) / 2.)
    boundary_b_c = round ((b_spec_b + b_spec_c) / 2.)
    boundary_c_wca = round ((b_spec_c + b_spec_wca) / 2.)

    region_flags_dict = {}
    region_flags_dict["NUVA"] = (eta < boundary_a_b)
    region_flags_dict["NUVB"] = (eta >= boundary_a_b) & (eta < boundary_b_c)
    region_flags_dict["NUVC"] = (eta >= boundary_b_c) & (eta < boundary_c_wca)

    return region_flags_dict

def nuvWcaRegions (eta, info, xtractab):
    """Determine the set of events within the NUV regions for the WCA.

    This is only used for NUV data.

    @param eta: pixel coordinates in cross-dispersion direction
    @type eta: array
    @param info: keywords and values
    @type info: dictionary
    @param xtractab: name of spectral extraction parameters reference table
    @type xtractab: string

    @return: dictionary with stripe name ("NUVA", "NUVB", "NUVC") as the key
        and an array of Boolean flags as the value, true for events for which
        the Y coordinate is within the regions for the WCA.
    @rtype: dictionary of Boolean arrays
    """

    # aperture and segment will be added to the filter below.
    filter = {"opt_elem": info["opt_elem"], "cenwave": info["cenwave"]}
    middle = float (NUV_X) / 2.

    # b_spec_c is the location (at the middle of the detector) of stripe C
    # for the PSA, and b_spec_wca is the location
    # of stripe A for the WCA.

    filter["segment"] = "NUVC"
    filter["aperture"] = "PSA"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_c = xtract_info.field ("b_spec")[0] + \
               xtract_info.field ("slope")[0] * middle

    filter["aperture"] = "WCA"

    filter["segment"] = "NUVA"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_wca_a = xtract_info.field ("b_spec")[0] + \
                   xtract_info.field ("slope")[0] * middle

    filter["segment"] = "NUVB"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_wca_b = xtract_info.field ("b_spec")[0] + \
                   xtract_info.field ("slope")[0] * middle

    filter["segment"] = "NUVC"
    xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
    b_spec_wca_c = xtract_info.field ("b_spec")[0] + \
                   xtract_info.field ("slope")[0] * middle

    # Set boundaries midway between adjacent stripes.
    boundary_c_wca = round ((b_spec_c + b_spec_wca_a) / 2.)
    boundary_a_b = round ((b_spec_wca_a + b_spec_wca_b) / 2.)
    boundary_b_c = round ((b_spec_wca_b + b_spec_wca_c) / 2.)

    region_flags_dict = {}
    region_flags_dict["NUVA"] = (eta >= boundary_c_wca) & (eta < boundary_a_b)
    region_flags_dict["NUVB"] = (eta >= boundary_a_b) & (eta < boundary_b_c)
    region_flags_dict["NUVC"] = (eta >= boundary_b_c)

    return region_flags_dict

def getWavecalOffsets (events):
    """Get min and max values of shift1 and shift2.

    @param events: the data unit containing the events table
    @type events: record array

    @return: (min_shift1, max_shift1, min_shift2, max_shift2), where
        min_shift1 and max_shift1 are the minimum and maximum values
        of the wavecal shift in the dispersion direction during the
        exposure (positive means a feature was detected at larger pixel
        coordinate than in the template); min_shift2 and max_shift2 are
        the corresponding values in the cross-dispersion direction
    @rtype: tuple
    """

    global active_area

    if active_area.any():
        xi  = events.field (xdopp)
        eta = events.field (ycorr)
        xi_full  = events.field (xfull)
        eta_full = events.field (yfull)

        xdiff = xi - xi_full
        ydiff = eta - eta_full
        xdiff = xdiff[active_area]
        ydiff = ydiff[active_area]

        min_shift1 = xdiff.min()
        max_shift1 = xdiff.max()
        min_shift2 = ydiff.min()
        max_shift2 = ydiff.max()
    else:
        # active_area is all False.
        min_shift1 = 0.
        max_shift1 = 0.
        min_shift2 = 0.
        max_shift2 = 0.

    return (min_shift1, max_shift1, min_shift2, max_shift2)

def copyColumns (events):
    """Copy XCORR and YCORR columns to XDOPP, XFULL and YFULL.

    Copy XCORR (RAWX) and YCORR (RAWY) to XDOPP, XFULL, YFULL as initial
    values, in case this is imaging data or wavecal processing will not
    be done.

    @param events: the data unit containing the events table
    @type events: record array
    """

    global xcorr, ycorr, xdopp, ydopp, xfull, yfull

    xi  = events.field (xcorr)
    eta = events.field (ycorr)
    xi_dopp  = events.field (xdopp)
    xi_full  = events.field (xfull)
    eta_full = events.field (yfull)

    xi_dopp[:] = xi.copy()
    xi_full[:] = xi.copy()
    eta_full[:] = eta.copy()
