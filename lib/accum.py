import math
import types
import numpy as N
import pyfits
import cosutil
import ccos
import timetag                  # actually for more generic functions
from calcosparam import *       # parameter definitions

def accumBasicCalibration (input, inpha, output, outcounts,
                  info, switches, reffiles,
                  stimfile, livetimefile):
    """Do the basic processing for accum data.

    The function value will be zero if there was no problem.

    arguments:
    input         name of the input file
    inpha         name of the input file containing the pulse height
                  histogram (FUV only)
    output        name of the output file for flat-fielded count-rate image
    outcounts     name of the output file for count-rate image
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    stimfile      name of output text file for stim positions (or None)
    livetimefile  name of output text file for livetime factors (or None)
    """

    cosutil.printIntro ("ACCUM calibration")
    names = [("Input", input), ("Output", output), ("Outcounts", outcounts)]
    if info["detector"] == "FUV":
        names.insert (1, ("Inpha", inpha))
    cosutil.printFilenames (names, stimfile=stimfile, livetimefile=livetimefile)
    cosutil.printMode (info)

    # Default values.
    stim_countrate = 0.
    stim_livetime = 1.

    # Get all the headers in the input file.  This copy will be
    # modified and written to the output files.
    headers = cosutil.getHeaders (input)
    phdr = headers[0]

    # Update the switches and reference file names, so the output header
    # will reflect what was actually used.
    cosutil.overrideKeywords (phdr, headers[1], info, switches, reffiles)

    # Check for null science data.
    if info["npix"] == (0,):
        writeNull (input, output, outcounts, headers)
        return 1

    doPhacorr (inpha, info, switches, reffiles, headers)

    dq_array = doDqicorr (input, info, switches, reffiles, headers)

    # Open the accum image.
    fd = pyfits.open (input, mode="readonly")
    sci = fd["SCI"].data
    fd.close()

    updateGlobrate (sci, info, reffiles, headers[1])

    # If these calibration steps are done, we will convert the accum image
    # to pseudo-time-tag format (just x & y, no time) and use the same
    # calibration as if we really had time-tag data.  Then we'll convert
    # back to an accum image.  But if none of these steps is done, we
    # don't need to go through the process of converting to/from pseudo
    # time-tag format.
    if info["detector"] == "FUV" and \
        (switches["tempcorr"] == "PERFORM" or
         switches["geocorr"]  == "PERFORM" or
         switches["randcorr"] == "PERFORM" or
         switches["deadcorr"] == "PERFORM"):

        ncounts = getNcounts (sci, info)        # total counts in input image
        if ncounts == 0:
            writeNull (input, output, outcounts, headers)
            info["npix"] = (0,)
            return 1

        # Get the stim locations and livetime factor now, for use later.
        if switches["tempcorr"] == "PERFORM" or \
           switches["deadcorr"] == "PERFORM":
            (s1, s2, s1_ref, s2_ref, stim_countrate, stim_livetime) = \
                  stimInfo (sci, reffiles["brftab"], info)

        # Create pseudo-timetag arrays (x & y, no time) from the raw image.
        x = N.zeros (ncounts, dtype=N.float32)
        y = N.zeros (ncounts, dtype=N.float32)
        ccos.unbinaccum (sci, x, y)

        stim_param = initTempcorr (input, s1, s2, s1_ref, s2_ref,
                         info, switches, stimfile, headers)

        doRandcorr (x, y, info, switches, headers)

        doTempcorr (x, y, stim_param, info, switches, reffiles, headers)

        doGeocorr (x, y, info, switches, reffiles, headers)

        # Now convert back to an accum image, still in counts.
        sci = N.zeros (info["npix"], dtype=N.float32)
        ccos.binevents (x, y, sci)

    err = N.sqrt (sci)

    sci /= info["exptime"]
    err /= info["exptime"]

    doHelcorr (info, switches, headers)

    # Write the count rate image.
    writeImset (outcounts, headers, sci, err, dq_array)

    doDeadcorr (input, sci, err, info, switches, reffiles,
                    stim_countrate, stim_livetime, livetimefile, headers)

    doFlatcorr (sci, err, info, switches, reffiles, headers)

    # Write the effective count rate image.  Note that this will be the same
    # as the count rate image if neither flatcorr nor deadcorr has been done.
    writeImset (output, headers, sci, err, dq_array)

    del sci, err, dq_array

    doStatflag (switches, output, outcounts)

    return 0            # 0 is OK

def updateGlobrate (sci, info, reffiles, hdr):
    """Update the GLOBRATE keyword in the extension header.

    arguments:
    sci           raw sci data array, assumed to be in counts (not counts/s)
    info          dictionary of header keywords and values
    reffiles      dictionary of reference file names
    hdr           the input sci extension header
    """

    globrate = globrate_accum (sci,
                info["exptime"], info["segment"], reffiles["brftab"])
    hdr.update ("globrate", globrate)

def globrate_accum (sci, exptime, segment, brftab):
    """Return the global count rate for accum data.

    arguments:
    sci           raw sci data array, assumed to be in counts (not counts/s)
    exptime       the exposure time
    segment       for finding a row in the brftab
    brftab        name of the baseline reference table

    The function value is the global count rate, counts per second.
    """

    if exptime <= 0.:
        return 0.

    if segment[0] == "N":
        return N.sum (sci.ravel().astype (N.float32)) / exptime

    (b_low, b_high) = cosutil.active_area (segment, brftab)
    temp = sci[b_low:b_high,:].copy()
    return N.sum (temp.ravel().astype (N.float32)) / exptime

def doPhacorr (inpha, info, switches, reffiles, headers):
    """Verify that the pulse height histograms are reasonable.

    arguments:
    inpha         name of the input file containing the pulse height histogram
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    headers       a list of the headers from the input file
    """

    if info["detector"] == "FUV":
        cosutil.printSwitch ("PHACORR", switches)
        if switches["phacorr"] == "PERFORM":
            cosutil.printRef ("PHATAB", reffiles)
            checkPulseHeight (inpha, reffiles["phatab"], info, headers[1])
            headers[0]["phacorr"] = "COMPLETE"

def checkPulseHeight (inpha, phatab, info, hdr):
    """Check that the pulse-height distribution is reasonable.

    arguments:
    inpha         name of file containing pulse-height distribution
    phatab        name of table of pulse-height parameters
    info          dictionary of keywords and values
    hdr           extension header
    """

    pha_info = cosutil.getTable (phatab, filter={"segment": info["segment"]},
                   exactly_one=True)

    # The peak in the pulse-height distribution should be within low and high.
    # The mean should be within the factors min_peak and max_peak of the peak.
    low = pha_info.field ("llt")[0]
    high = pha_info.field ("ult")[0]
    min_peak = pha_info.field ("min_peak")[0]
    max_peak = pha_info.field ("max_peak")[0]

    # Update the values for the screening limit keywords
    # (low and high are the default values).
    cosutil.updatePulseHeightKeywords (hdr, info["segment"], low, high)

    # Read the pulse-height histogram.
    fd = pyfits.open (inpha, mode="readonly")
    pha_data = fd[1].data

    npts = len (pha_data)

    sum = N.sum ( \
              N.arange (npts, dtype=N.float32) * \
              pha_data.astype (N.float32))
    sumwgt = N.sum (pha_data.astype (N.float32))
    pha_index = N.argsort (pha_data)
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

def doDqicorr (input, info, switches, reffiles, headers):
    """Apply the data quality initialization table.

    The data quality extension in the input file will be read into dq_array.
    The flag information in the bpixtab will be combined with dq_array
    in-place via bitwise OR, taking the Doppler shift into account.  Also,
    if the input was a subarray, regions outside the subarray will be
    flagged as out of bounds.

    arguments:
    input         name of the input raw file
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    headers       a list of the headers from the input file

    The function value is dq_array, the updated data quality array.
    """

    # Get the input data quality array (which defaults to an array of zeros).
    dq_array = cosutil.getInputDQ (input)

    cosutil.printSwitch ("DQICORR", switches)

    if switches["dqicorr"] == "PERFORM":

        cosutil.printRef ("BPIXTAB", reffiles)

        # Read values from the bpixtab, and bitwise OR them with the dq_array,
        # taking into account the Doppler shift.
        cosutil.updateDQArray (reffiles["bpixtab"], info,
                      switches["doppcorr"], dq_array)

        # Flag regions that are outside any subarray as out of bounds.
        cosutil.flagOutOfBounds (headers[0], headers[1], dq_array)

        headers[0]["dqicorr"] = "COMPLETE"

    return dq_array

def doHelcorr (info, switches, headers):
    """Compute radial velocity, and assign to v_helio keyword.

    arguments:
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    headers       a list of the headers from the input file
    """

    if info["obstype"] != "SPECTROSCOPIC":
        return

    cosutil.printSwitch ("HELCORR", switches)
    if switches["helcorr"] == "PERFORM":
        t_mid = cosutil.timeAtMidpoint (info)
        radvel = timetag.heliocentricVelocity (t_mid,
                        info["ra_targ"], info["dec_targ"])
        headers[1].update ("v_helio", radvel)
        info["v_helio"] = radvel
        headers[0]["helcorr"] = "COMPLETE"
    else:
        headers[1].update ("v_helio", 0.)

def initTempcorr (input, s1, s2, s1_ref, s2_ref,
                         info, switches, stimfile, headers):
    """Return the stim parameters.

    arguments:
    input         name of the input file
    s1            measured location in raw data of first stim (y, x)
    s2            measured location in raw data of second stim (y, x)
                  (the location will be [None, None] if there are no counts)
    s1_ref        reference location of first stim (y, x)
    s2_ref        reference location of second stim (y, x)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    stimfile      name of output text file for stim positions (or None)
    headers       a list of the headers from the input file
    """

    if switches["tempcorr"] == "PERFORM":
        # Update stim location keywords in extension header.
        timetag.stimKeywords (headers[1], info["segment"], (s1, s2))
        stim_param = computeThermalParamAccum (s1, s2, s1_ref, s2_ref,
                input, stimfile)
    else:
        stim_param = {}

    return stim_param

def doRandcorr (x, y, info, switches, headers):
    """Add pseudo-random numbers to x and y coordinates.

    arguments:
    x, y          1-D arrays of pixel coordinates (modified in-place)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    headers       a list of the headers from the input file
    """

    cosutil.printSwitch ("RANDCORR", switches)
    if switches["randcorr"] == "PERFORM":
        cosutil.addRandomNumbers (x, y, info["randseed"])
        headers[0]["randcorr"] = "COMPLETE"

def doTempcorr (x, y, stim_param, info, switches, reffiles, headers):
    """Apply thermal distortion correction to x and y coordinates.

    arguments:
    x, y          1-D arrays of pixel coordinates (modified in-place)
    stim_param    a dictionary of lists:  (i0, i1, x0, xslope, y0, yslope)
                    (see timetag.computeThermalParam)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    headers       a list of the headers from the input file
    """

    cosutil.printSwitch ("TEMPCORR", switches)
    if switches["tempcorr"] == "PERFORM":
        cosutil.printRef ("BRFTAB", reffiles)
        if timetag.thermalDistortion (x, y, stim_param):
            headers[0]["tempcorr"] = "COMPLETE"
        else:
            headers[0]["tempcorr"] = "SKIPPED"
            cosutil.printMsg ("TEMPCORR was skipped")

def doGeocorr (x, y, info, switches, reffiles, headers):
    """Apply geometric correction to x and y coordinates.

    arguments:
    x, y          1-D arrays of pixel coordinates (modified in-place)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    headers       a list of the headers from the input file
    """

    cosutil.printSwitch ("GEOCORR", switches)
    if switches["geocorr"] == "PERFORM":
        cosutil.printRef ("GEOFILE", reffiles)
        cosutil.printSwitch ("IGEOCORR", switches)
        cosutil.geometricDistortion (x, y,
                reffiles["geofile"], info["segment"], switches["igeocorr"])
        headers[0]["geocorr"] = "COMPLETE"
        if switches["igeocorr"] == "PERFORM":
            headers[0]["igeocorr"] = "COMPLETE"

def doDeadcorr (input, sci, err, info, switches, reffiles,
                    stim_countrate, stim_livetime, livetimefile, headers):
    """Correct sci and err for deadtime.

    arguments:
    input           name of the input file (only used to write to livetimefile)
    sci             sci data array (updated in-place)
    err             error estimates data array (updated in-place)
    info            dictionary of header keywords and values
    switches        dictionary of calibration switches
    reffiles        dictionary of reference file names
    stim_countrate  observed count rate (per second) of the stims
                    (both together)
    stim_livetime   livetime factor based on stim_countrate
    livetimefile    if not None, the name of a log file for livetime factors
    headers         a list of the headers from the input file
    """

    cosutil.printSwitch ("DEADCORR", switches)

    if switches["deadcorr"] == "PERFORM":
        cosutil.printRef ("DEADTAB", reffiles)
        # Determine live time factor.
        livetime = deadtimeCorrectionAccum (reffiles["deadtab"],
                    stim_countrate, stim_livetime, info, input, livetimefile)
        sci /= livetime
        err /= livetime
        headers[0]["deadcorr"] = "COMPLETE"

def deadtimeCorrectionAccum (deadtab,
            stim_countrate, stim_livetime, info, input, livetimefile):
    """Determine livetime factor.

    For FUV the livetime factor is gotten from the count rates for the
    stims; the count rate and corresponding livetime are input parameters.
    The factor is gotten from the digital event counter as well, however,
    and compared with the stim livetime factor.  For NUV the livetime factor
    is gotten from the MAMA event counter.

    arguments:
    deadtab         name of the reference table giving livetime vs count rate
    stim_countrate  observed count rate (per second) of the stims
                    (both together)
    stim_livetime   livetime factor based on stim_countrate
    info            dictionary of header keywords and values
    input           name of the input file (only used to write to livetimefile)
    livetimefile    if not None, the name of a log file for livetime factors

    The function value is the livetime.
    """

    if livetimefile is None:
        fd = None
    else:
        fd = open (livetimefile, "a")
        fd.write ("# %s\n" % (input,))

    live_info = cosutil.getTable (deadtab, filter={"segment": info["segment"]},
                    at_least_one=True)
    obs_rate = live_info.field ("obs_rate")
    live_factor = live_info.field ("livetime")

    # Output count rate from digital event counter (DEC), or from MEVENTS
    # if NUV.
    dec_countrate = info["countrate"]
    # Interpolate to get livetime from DEC.
    dec_livetime = timetag.determineLivetime (dec_countrate,
                     obs_rate, live_factor)

    print_details = (cosutil.checkVerbosity (VERY_VERBOSE))     # initial value
    if info["detector"] == "FUV":
        # Use livetime from stims, instead of from DEC.
        livetime = stim_livetime
        # Compare livetime from stims with livetime from DEC.
        if abs (stim_livetime - dec_livetime) > \
                LIVETIME_CRITERION * stim_livetime:
            cosutil.printWarning ("Livetime estimates differ.")
            print_details = 1
    elif info["detector"] == "NUV":
        livetime = dec_livetime

    # This is used if print_details is true or we're writing to a livetimefile.
    if info["segment"] == "FUVA":
        kwd = "DEVENTA"
    elif info["segment"] == "FUVB":
        kwd = "DEVENTB"
    else:
        kwd = "MEVENTS"

    if print_details:
        if info["detector"] == "FUV":
            cosutil.printMsg ("  stim countrate and livetime:  %.6g, %6.4f" % \
                      (stim_countrate, stim_livetime))
        cosutil.printMsg ("  countrate and livetime from %s:  %.6g, %6.4f" % \
                  (kwd, dec_countrate, dec_livetime))

    if fd is not None:
        if info["detector"] == "FUV":
            fd.write ("# stim countrate and livetime:\n")
            fd.write ("%.6g %6.4f\n" % (stim_countrate, stim_livetime))
        fd.write ("# countrate and livetime from %s:\n" % (kwd,))
        fd.write ("%.6g %6.4f\n" % (dec_countrate, dec_livetime))

    if fd is not None:
        fd.close()

    return livetime

def doFlatcorr (sci, err, info, switches, reffiles, headers):
    """Apply the flat field to sci and err.

    arguments:
    sci           sci data array (updated in-place)
    err           error estimates data array (updated in-place)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    headers       a list of the headers from the input file
    """

    cosutil.printSwitch ("FLATCORR", switches)

    if switches["flatcorr"] == "PERFORM":
        cosutil.printRef ("FLATFILE", reffiles)
        (flat, origin) = getFlatField (reffiles["flatfile"],
                         info["detector"], info["segment"])
        if info["obstype"] == "SPECTROSCOPIC":
            cosutil.printSwitch ("DOPPCORR", switches)
        if switches["doppcorr"] == "PERFORM":
            convolveFlat (flat, info["dispaxis"], \
                 info["expstart"], info["exptime"],
                 info["doppmag"], info["doppzero"], info["orbitper"])
            headers[0]["doppcorr"] = "COMPLETE"
        # Now divide by the flat field image.
        doFlatAccum (sci, flat, origin)
        doFlatAccum (err, flat, origin)
        headers[0]["flatcorr"] = "COMPLETE"

def getFlatField (flatfile, detector, segment):
    """Read flat field file.

    arguments:
    flatfile   name of the flat field reference file
    detector   FUV or NUV
    segment    FUVA or FUVB (not used for NUV)

    the function value is a tuple containing:
    flat       2-D data array read from flat field file (may be a subarray)
    origin     list of Y and X offsets of flat subarray from first pixel
    """

    fd = pyfits.open (flatfile, mode="readonly")

    if detector == "NUV":
        hdu = fd[1]
    else:
        hdu = fd[(segment,1)]

    origin = [0, 0]                             # create the list
    origin[1] = hdu.header.get ("origin_x", 0)
    origin[0] = hdu.header.get ("origin_y", 0)

    flat = hdu.data

    fd.close()

    return (flat, origin)

def convolveFlat (flat, dispaxis,
                expstart, exptime, doppmag, doppzero, orbitper):
    """Convolve the flat field file with the Doppler smearing function.

    arguments:
    flat       flat field data array, modified in-place
    dispaxis   dispersion axis (1 or 2)
    expstart   exposure start time, MJD
    exptime    exposure duration, seconds
    doppmag    magnitude of Doppler shift, pixels
    doppzero   time when Doppler shift is zero and increasing
    orbitper   orbital period of HST
    """

    # Round doppmag up to the next integer; mag is a zero-point offset.
    mag = int (math.ceil (doppmag))

    # dopp will be the Doppler smoothing function, normalized so its sum is 1.
    dopp = N.zeros (2*mag+1, dtype=N.float32)

    # time is the time in seconds since doppzero, in one second increments.
    time = N.arange (exptime, dtype=N.float32) + \
               (expstart - doppzero) * SEC_PER_DAY

    # shift is in pixels (wavelengths decrease toward larger pixel number).
    shift = doppmag * N.sin (2. * N.pi * time / orbitper)

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

def doFlatAccum (raw, flat, origin):
    """Divide the raw image in-place by the flat field (for accum data).

    arguments:
    raw        2-D raw data image array (could be SCI or ERR extension)
    flat       2-D data array read from flat field file (may be a subarray)
    origin     list of Y and X offsets of flat subarray from first pixel of
                 the detector
    """

    # Get the location of the flat subarray within the raw image.
    rawsize = raw.shape
    flatsize = flat.shape
    xlow = origin[1]
    ylow = origin[0]
    xhigh = xlow + flatsize[1]
    yhigh = ylow + flatsize[0]
    if xhigh > rawsize[1] or yhigh > rawsize[0]:
        raise RuntimeError, "ERROR:  FLATFILE size or offset is too large."

    raw[ylow:yhigh,xlow:xhigh] /= flat

def stimInfo (sci, brftab, info):
    """Get positions of stims, and get livetime from stims.

    arguments:
    sci        raw (or possibly geometrically corrected) image data array,
                 in counts
    brftab     name of the baseline reference frame table
    info       dictionary of keywords and values

    The function value is a tuple:
    s1         measured location in raw data of first stim (y, x)
    s2         measured location in raw data of second stim (y, x)
               (the location will be [None, None] if there are no counts)
    s1_ref     reference location of first stim (y, x)
    s2_ref     reference location of second stim (y, x)
    countrate  counts / s
    livetime   the livetime factor (default = 1.) based on the stim rate
    """

    brf_info = cosutil.getTable (brftab, filter={"segment": info["segment"]},
                   exactly_one=True)

    sx1 = brf_info.field ("sx1")[0]
    sy1 = brf_info.field ("sy1")[0]
    sx2 = brf_info.field ("sx2")[0]
    sy2 = brf_info.field ("sy2")[0]
    xwidth = brf_info.field ("xwidth")[0]
    ywidth = brf_info.field ("ywidth")[0]

    # These are the reference locations of the stims.
    s1_ref = (sy1, sx1)
    s2_ref = (sy2, sx2)

    (s1, counts1) = findStimAccum (sci, s1_ref, xwidth, ywidth)
    (s2, counts2) = findStimAccum (sci, s2_ref, xwidth, ywidth)

    if cosutil.checkVerbosity (VERY_VERBOSE):
        if s1[0] is None:
            str1 = "stim1 not found"
        else:
            str1 = "%.2f %.2f" % (s1[1], s1[0])
        if s2[0] is None:
            str2 = "stim2 not found"
        else:
            str2 = "%.2f %.2f" % (s2[1], s2[0])
        msg = "measured stim locations:  " + str1 + "   " + str2
        cosutil.printMsg (msg, VERY_VERBOSE)

    countrate = (counts1 + counts2) / (2. * info["exptime"])

    if info["stimrate"] > 0.:
        livetime = countrate / info["stimrate"]
    else:
        livetime = 1.

    return (s1, s2, s1_ref, s2_ref, countrate, livetime)

def findStimAccum (sci, stim_ref, xwidth, ywidth):
    """Get the location and total counts for one of the stims.

    arguments:
    sci             image data, in counts
    stim_ref        Y and X locations of stim from baseline reference frame
    xwidth, ywidth  search range (size of subarray) for stim

    The function value is a tuple of (s_loc, counts):
    s_loc      measured location of stim:  [y, x]
               (the location will be [None, None] if there are no counts)
    counts     total counts in search region
    """

    sX = int (round (stim_ref[1]))
    sY = int (round (stim_ref[0]))

    # One is added to highX (and highY) because we're going to use
    # a slice, and we want sX + xwidth (and xY + ywidth) to be
    # inclusive upper limits.
    lowX  = sX - xwidth
    highX = sX + xwidth + 1
    lowY  = sY - ywidth
    highY = sY + ywidth + 1

    (sci_ny,sci_nx) = sci.shape
    lowX = max (lowX, 0)
    lowY = max (lowY, 0)
    highX = min (highX, sci_nx)
    highY = min (highY, sci_ny)

    nx = highX - lowX
    ny = highY - lowY

    region = sci[lowY:highY,lowX:highX].copy()

    # sX and sY are subtracted here (and added back in later) to
    # reduce the possibility of numerical roundoff errors.
    x = N.arange (lowX, highX, dtype=N.float32) - sX
    y = N.arange (lowY, highY, dtype=N.float32) - sY
    y.setshape ((ny, 1))

    x_region = region * x
    y_region = region * y

    region.ravel()
    x_region.ravel()
    y_region.ravel()

    counts = N.sum (region)
    s_loc = [None, None]
    if counts > 0:
        s_loc[1] = N.sum (x_region) / float (counts) + sX
        s_loc[0] = N.sum (y_region) / float (counts) + sY

    return (s_loc, counts)

def computeThermalParamAccum (s1, s2, s1_ref, s2_ref, input, stimfile):
    """Compute thermal distortion parameters from stim positions.

    If a stimfile was specified, it will be opened (append mode), and the
    stim positions will be written to the file.  (The 'input' argument is
    included in the calling sequence just for the purpose of writing its
    name to the stimfile.)

    arguments:
    s1           measured location in raw data of first stim (y, x)
    s2           measured location in raw data of second stim (y, x)
                 (the location will be [None, None] if there are no counts)
    s1_ref       reference location of first stim (y, x)
    s2_ref       reference location of second stim (y, x)
    input        name of raw file (for writing to stimfile)
    stimfile     name of text file to which stim locations will be appended

    The function value is a dictionary of lists, with keys "x0", "xslope",
    "y0", "yslope".

    In this version (for accum), each list contains only one element:
      x0[0] and xslope[0] are the intercept and slope respectively of the
        linear correction to the X positions (the more rapidly varying
        direction).
      y0[0] and yslope[0] are the intercept and slope for the linear
        correction to the Y positions.
    """

    if stimfile is None:
        fd = None
    else:
        fd = open (stimfile, "a")
        fd.write ("# %s\n" % (input,))

    if fd is not None:
        fd.write ("# stim_locations\n")
        if s1[0] is None:
            fd.write ("INDEF INDEF")
        else:
            fd.write ("%.1f %.1f" % (s1[0], s1[1]))
        if s2[0] is None:
            fd.write ("  INDEF INDEF\n")
        else:
            fd.write ("  %.1f %.1f\n" % (s2[0], s2[1]))
        fd.close()

    (x0_n, xslope_n, y0_n, yslope_n) = \
                timetag.thermalParam (s1, s2, s1_ref, s2_ref)
    x0 = [x0_n]
    xslope = [xslope_n]
    y0 = [y0_n]
    yslope = [yslope_n]

    cosutil.printMsg (
                "thermal distortion:  %0.4f + %0.10fx;  %0.4f + %0.10fy" % \
                (x0_n, xslope_n, y0_n, yslope_n), VERY_VERBOSE)

    stim_param = {"x0": x0, "xslope": xslope, "y0": y0, "yslope": yslope}

    return stim_param

def getNcounts (sci, info):
    """Return the total number of counts in an array.

    arguments:
    sci        image data array, in counts
    info       dictionary of keywords and values
    """

    ncounts = N.sum (sci.flat)
    if type (ncounts) is types.FloatType:
        ncounts = int (round (ncounts))

    return ncounts

def writeNull (input, output, outcounts, headers):
    """Write output files with null data portions.

    arguments:
    input         name of input file
    output        name of the output file for flat-fielded count-rate image
    outcounts     name of the output file for count-rate image
    headers       list of headers (primary, sci, err, dq)
    """

    cosutil.printWarning ("No data in " + input)
    writeImset (outcounts, headers, None, None, None)
    writeImset (output, headers, None, None, None)

def writeImset (output, headers, sci_array, err_array, dq_array):
    """Write an image set (SCI, ERR, DQ extensions).

    This function writes an output file that consists of a primary
    header (with no data), a SCI extension HDU, an ERR HDU (estimates
    of the errors in the SCI data, and a DQ HDU (data quality flags).

    arguments:
    output        name of output file
    headers       list of headers (primary, sci, err, dq)
    sci_array     data array for SCI extension
    err_array     data array for ERR extension
    dq_array      data array for DQ extension
    """

    cosutil.printMsg ("writing file %s ..." % output, VERY_VERBOSE)

    primary_hdu = pyfits.PrimaryHDU (header=headers[0])
    fd = pyfits.HDUList (primary_hdu)
    fd[0].header["nextend"] = 3
    cosutil.updateFilename (fd[0].header, output)

    hdu = pyfits.ImageHDU (data=sci_array, header=headers[1], name="SCI")
    hdu.header.update ("BUNIT", "count /s")
    fd.append (hdu)

    hdu = pyfits.ImageHDU (data=err_array, header=headers[2], name="ERR")
    hdu.header.update ("BUNIT", "count /s")
    fd.append (hdu)

    hdu = pyfits.ImageHDU (data=dq_array, header=headers[3], name="DQ")
    hdu.header.update ("BUNIT", "UNITLESS")
    fd.append (hdu)

    fd.writeto (output, output_verify="silentfix")

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
