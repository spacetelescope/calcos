import math
import time
import types
import numpy as N
from numpy import random
import pyfits

import cosutil
import ccos
import timetag                  # actually for more generic functions
import wavecal
from calcosparam import *       # parameter definitions

def accumBasicCalibration (input, inpha, output, outcounts, outcsum,
                  info, switches, reffiles,
                  wavecal_info,
                  stimfile, livetimefile):
    """Do the basic processing for accum data.

    The function value will be zero if there was no problem.

    arguments:
    input         name of the input file
    inpha         name of the input file containing the pulse height
                  histogram (FUV only)
    output        name of the output file for flat-fielded count-rate image
    outcounts     name of the output file for count-rate image
    outcsum       name of the output image for OPUS to add to cumulative
                      image (or None)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    wavecal_info  when wavecal exposures were processed, the results
                    were stored in this dictionary
    stimfile      name of output text file for stim positions (or None)
    livetimefile  name of output text file for livetime factors (or None)
    """

    cosutil.printIntro ("ACCUM calibration")
    names = [("Input", input), ("OutFlt", output), ("OutCounts", outcounts)]
    if info["detector"] == "FUV":
        names.insert (1, ("InPha", inpha))
    if outcsum is not None:
        names.append (("OutCsum", outcsum))
    cosutil.printFilenames (names, stimfile=stimfile, livetimefile=livetimefile)
    cosutil.printMode (info)

    # Default values.
    (avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref,
     stim_countrate, stim_livetime) = \
           (None, None, None, None, None, None, 0., 1.)

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

    dq_array = doDqicorr (input, info, switches, reffiles,
                          wavecal_info, headers)

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

        # Create pseudo-timetag arrays (x & y, no time) from the raw image.
        x = N.zeros (ncounts, dtype=N.float32)
        y = N.zeros (ncounts, dtype=N.float32)
        ccos.unbinaccum (sci, x, y)

        doRandcorr (x, y, info, switches, reffiles, headers)

        # Get the stim locations and livetime factor.
        if switches["tempcorr"] == "PERFORM" or \
           switches["deadcorr"] == "PERFORM":
            (avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref,
             stim_countrate, stim_livetime) = \
                  stimInfo (x, y, reffiles["brftab"], info)

        # Compute the linear distortion parameters from the stim locations.
        stim_param = initTempcorr (input, avg_s1, avg_s2, rms_s1, rms_s2,
                         s1_ref, s2_ref, info, switches, stimfile, headers)

        # Do the linear distortion correction based on the stim locations.
        doTempcorr (x, y, stim_param, info, switches, reffiles, headers)

        # Do the nonlinear geometric correction.
        doGeocorr (x, y, info, switches, reffiles, headers)

        # Now convert back to an accum image, still in counts.
        sci = N.zeros (info["npix"], dtype=N.float32)
        ccos.binevents (x, y, sci)

    if sci.dtype != N.float32:
        sci = sci.astype (N.float32)

    sci = clobberBadPixels (sci, dq_array)

    err = N.sqrt (sci)

    sci /= info["exptime"]
    err /= info["exptime"]

    initHelcorr (info, switches, headers)

    # Write the count rate image.
    writeImset (outcounts, headers, sci, err, dq_array)

    doDeadcorr (input, sci, err, info, switches, reffiles,
                    stim_countrate, stim_livetime, livetimefile, headers)

    # Write the calcos sum image.
    if outcsum is not None:
        writeCsum (outcsum, headers, sci, info["exptime"])

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

    (b_low, b_high, b_left, b_right) = cosutil.activeArea (segment, brftab)
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

    sum = N.sum (N.arange (npts, dtype=N.float32) * pha_data.astype (N.float32))
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

def doDqicorr (input, info, switches, reffiles, wavecal_info, headers):
    """Apply the data quality initialization table.

    The data quality extension in the input file will be read into dq_array.
    The flag information in the bpixtab will be combined with dq_array
    in-place via bitwise OR, taking the Doppler shift and wavecal offset
    into account.

    arguments:
    input         name of the input raw file
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    wavecal_info  when wavecal exposures were processed, the results
                    were stored in this dictionary
    headers       a list of the headers from the input file

    The function value is dq_array, the updated data quality array.
    """

    # Get the input data quality array (which defaults to an array of zeros).
    dq_array = cosutil.getInputDQ (input)

    cosutil.printSwitch ("DQICORR", switches)

    if switches["dqicorr"] == "PERFORM":

        cosutil.printRef ("BPIXTAB", reffiles)
        minmax_shifts = getWavecalOffsets (wavecal_info,
                                           info, switches, reffiles)
        if minmax_shifts is not None:
            (min_shift1, max_shift1, min_shift2, max_shift2) = minmax_shifts
            # xxx the following is temporary;
            # xxx remove when the full wavecal shift is to be applied to accum
            dx = (max_shift1 - min_shift1) / 2.
            dy = (max_shift2 - min_shift2) / 2.
            min_shift1 = -dx
            min_shift2 = -dy
            max_shift1 = dx
            max_shift2 = dy
            minmax_shifts = (min_shift1, max_shift1, min_shift2, max_shift2)

        # Read values from the bpixtab, and bitwise OR them with the dq_array,
        # taking into account the Doppler shift and wavecal offset.
        cosutil.updateDQArray (reffiles["bpixtab"], info, switches["doppcorr"],
                      info["dopmagt"], info["dopzerot"], info["orbtpert"],
                      dq_array, minmax_shifts)

        # Flag regions that are outside any subarray as out of bounds.
        #  (OPUS does this already)
        # cosutil.flagOutOfBounds (headers[0], headers[1], dq_array)

        # Flag the region that is outside the active area.
        if info["detector"] == "FUV":
            cosutil.flagOutsideActiveArea (dq_array,
                        info["segment"], reffiles["brftab"])

        headers[0]["dqicorr"] = "COMPLETE"

    return dq_array

def clobberBadPixels (sci, dq_array):
    """Set pixel values to zero where the data quality is "bad".

    @param sci: the data array
    @type sci: numpy array (float32)
    @param dq_array: the data quality array
    @type dq_array: numpy array (int8)

    @return: the data array with bad pixels set to zero
    @rtype: numpy array (float32)
    """

    sdqflags = (DQ_DEAD + DQ_HOT)

    sci = N.where (N.bitwise_and (dq_array, sdqflags), 0., sci)

    return sci

def initHelcorr (info, switches, headers):
    """Compute radial velocity, and assign to v_helio keyword.

    arguments:
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    headers       a list of the headers from the input file
    """

    if info["obstype"] != "SPECTROSCOPIC":
        return

    if switches["helcorr"] == "PERFORM":
        t_mid = cosutil.timeAtMidpoint (info)
        radvel = timetag.heliocentricVelocity (t_mid,
                        info["ra_targ"], info["dec_targ"])
        headers[1].update ("v_helio", radvel)
        info["v_helio"] = radvel
    else:
        headers[1].update ("v_helio", 0.)

def doRandcorr (x, y, info, switches, reffiles, headers):
    """Add pseudo-random numbers to x and y coordinates.

    arguments:
    x, y          1-D arrays of pixel coordinates (modified in-place)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    headers       a list of the headers from the input file
    """

    cosutil.printSwitch ("RANDCORR", switches)
    if switches["randcorr"] == "PERFORM":
        nelem = len (x)
        (b_low, b_high, b_left, b_right) = \
                cosutil.activeArea (info["segment"], reffiles["brftab"])
        # A value of 0 in rand_flags means the corresponding event
        # should not be modified by adding a pseudo-random number.
        rand_flags = N.ones (nelem, dtype=N.bool8)
        rand_flags = N.where (x > b_right, 0, rand_flags)
        rand_flags = N.where (x < b_left,  0, rand_flags)
        rand_flags = N.where (y > b_high,  0, rand_flags)
        rand_flags = N.where (y < b_low,   0, rand_flags)
        if info["randseed"] == -1:
            seed = int (time.time())
            headers[0]["randseed"] = seed
        else:
            seed = info["randseed"]
        random.seed (seed)
        rn = random.uniform (-0.5, +0.5, nelem)
        x[:] = N.where (rand_flags, x - rn, x)
        rn = random.uniform (-0.5, +0.5, nelem)
        y[:] = N.where (rand_flags, y - rn, y)
        headers[0]["randcorr"] = "COMPLETE"

def stimInfo (x, y, brftab, info):
    """Get positions of stims, and get livetime from stims.

    arguments:
    x, y       pseudo-timetag pixel coordinates
    brftab     name of the baseline reference frame table
    info       dictionary of keywords and values

    The function value is a tuple:
    s1         measured location in raw data of first stim (y, x)
    s2         measured location in raw data of second stim (y, x)
               (the location will be [None, None] if there are no counts)
    rms_s1     RMS width in raw data of first stim (y, x)
    rms_s2     RMS width in raw data of second stim (y, x)
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

    (avg_s1, sumsq1, counts1, found_s1) = \
                timetag.findStim (x, y, s1_ref, xwidth, ywidth)
    (avg_s2, sumsq2, counts2, found_s2) = \
                timetag.findStim (x, y, s2_ref, xwidth, ywidth)

    if cosutil.checkVerbosity (VERY_VERBOSE):
        if found_s1:
            str1 = "%.2f %.2f" % (avg_s1[1], avg_s1[0])
        else:
            str1 = "stim1 not found"
        if found_s2:
            str2 = "%.2f %.2f" % (avg_s2[1], avg_s2[0])
        else:
            str2 = "stim2 not found"
        msg = "measured stim locations:  " + str1 + "   " + str2
        cosutil.printMsg (msg, VERY_VERBOSE)
    rms_s1 = [-1., -1.]
    rms_s2 = [-1., -1.]
    if counts1 > 1:
        rms_s1[0] = math.sqrt (sumsq1[0] / (counts1 - 1.))
        rms_s1[1] = math.sqrt (sumsq1[1] / (counts1 - 1.))
    elif counts1 > 0:
        rms_s1[0] = math.sqrt (sumsq1[0])
        rms_s1[1] = math.sqrt (sumsq1[1])
    if counts2 > 1:
        rms_s2[0] = math.sqrt (sumsq2[0] / (counts2 - 1.))
        rms_s2[1] = math.sqrt (sumsq2[1] / (counts2 - 1.))
    elif counts2 > 0:
        rms_s2[0] = math.sqrt (sumsq2[0])
        rms_s2[1] = math.sqrt (sumsq2[1])

    countrate = (counts1 + counts2) / (2. * info["exptime"])

    if info["stimrate"] > 0.:
        livetime = countrate / info["stimrate"]
    else:
        livetime = 1.

    return (avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref,
            countrate, livetime)

def initTempcorr (input, avg_s1, avg_s2, rms_s1, rms_s2,
                  s1_ref, s2_ref, info, switches, stimfile, headers):
    """Return the stim parameters.

    This function computes the stim parameters, and it also calls a
    function (in timetag.py) to write the stim locations and RMS widths
    to the output header.

    arguments:
    input         name of the input file

    avg_s1[0] is the Y location of the first stim.
    avg_s1[1] is the X location of the first stim.
    avg_s2[0] is the Y location of the second stim.
    avg_s2[1] is the X location of the second stim.
                  (the location will be [None, None] if there are no counts)

    rms_s1[0] is the RMS in Y for the first stim.
    rms_s1[1] is the RMS in X for the first stim.
    rms_s2[0] is the RMS in Y for the second stim.
    rms_s2[1] is the RMS in X for the second stim.

    s1_ref        reference location of first stim (y, x)
    s2_ref        reference location of second stim (y, x)
    info          dictionary of header keywords and values
    switches      dictionary of calibration switches
    stimfile      name of output text file for stim positions (or None)
    headers       a list of the headers from the input file
    """

    if switches["tempcorr"] == "PERFORM":
        # Update stim location keywords in extension header.
        timetag.stimKeywords (headers[1], info["segment"],
                              avg_s1, avg_s2, rms_s1, rms_s2, s1_ref, s2_ref)
        stim_param = computeThermalParamAccum (avg_s1, avg_s2, s1_ref, s2_ref,
                input, stimfile)
    else:
        stim_param = {}

    return stim_param

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
                 info["dopmagt"], info["dopzerot"], info["orbtpert"])
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
                expstart, exptime, dopmagt, dopzerot, orbtpert):
    """Convolve the flat field file with the Doppler smearing function.

    arguments:
    flat       flat field data array, modified in-place
    dispaxis   dispersion axis (1 or 2)
    expstart   exposure start time, MJD
    exptime    exposure duration, seconds
    dopmagt    magnitude of Doppler shift, pixels
    dopzerot   time when Doppler shift is zero and increasing
    orbtpert   orbital period of HST
    """

    # Round dopmagt up to the next integer; mag is a zero-point offset.
    mag = int (math.ceil (dopmagt))

    # dopp will be the Doppler smoothing function, normalized so its sum is 1.
    dopp = N.zeros (2*mag+1, dtype=N.float32)

    # t is the time in seconds since dopzerot, in one second increments.
    t = N.arange (int (round (exptime)), dtype=N.float32) + \
               (expstart - dopzerot) * SEC_PER_DAY

    # shift is in pixels (wavelengths increase toward larger pixel number).
    shift = -dopmagt * N.sin (2. * N.pi * t / orbtpert)

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

    sci_hdr = None
    err_hdr = None
    dq_hdr = None
    for i in range (1, len (headers)):
        extname = headers[i].get ("extname", "not found")
        extver  = headers[i].get ("extver", 1)
        if extver != 1:
            cosutil.printWarning ("EXTVER = %d, should be 1" % extver)
        if extname.upper() == "SCI":
            sci_hdr = headers[i]
        elif extname.upper() == "ERR":
            err_hdr = headers[i]
        elif extname.upper() == "DQ":
            dq_hdr = headers[i]

    hdu = pyfits.ImageHDU (data=sci_array, header=sci_hdr, name="SCI")
    hdu.header.update ("BUNIT", "count /s")
    fd.append (hdu)

    hdu = pyfits.ImageHDU (data=err_array, header=err_hdr, name="ERR")
    hdu.header.update ("BUNIT", "count /s")
    fd.append (hdu)

    hdu = pyfits.ImageHDU (data=dq_array, header=dq_hdr, name="DQ")
    hdu.header.update ("BUNIT", "UNITLESS")
    fd.append (hdu)

    fd.writeto (output, output_verify="silentfix")

def writeCsum (outcsum, headers, sci_array, exptime):
    """Write the "calcos sum" (csum) image.

    @param outcsum: name of output calcos sum image file
    @type outcsum: string
    @param headers: list of headers (primary, sci, err, dq)
    @type headers: list
    @param sci_array: data array for SCI extension
    @type sci_array: numpy array
    @param exptime: exposure time (s)
    @type exptime: float
    """

    cosutil.printMsg ("writing file %s ..." % outcsum, VERY_VERBOSE)

    sci_counts = sci_array * exptime

    primary_hdu = pyfits.PrimaryHDU (header=headers[0])
    fd = pyfits.HDUList (primary_hdu)
    fd[0].header.update ("nextend", 1)
    fd[0].header.update ("counts", sci_counts.sum())
    fd[0].header.update ("filetype", "CALCOS SUM FILE")
    cosutil.updateFilename (fd[0].header, outcsum)

    hdu = pyfits.ImageHDU (data=sci_counts, header=headers[1], name="SCI")
    hdu.header.update ("BUNIT", "count")
    fd.append (hdu)

    fd.writeto (outcsum, output_verify="silentfix")

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

def getWavecalOffsets (wavecal_info,
                       info, switches, reffiles):
    """Return offsets based on auto or GO wavecal info.

    @param wavecal_info: when wavecal exposures were processed, the results
        were stored in this dictionary
    @type wavecal_info: dictionary
    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary

    @return: four values:  the min and max offsets in the dispersion
        direction and the min and max offsets in the cross-dispersion
        direction during the exposure; these values will all be zero if
        the current observation is a wavecal or if wavecal processing was
        not done.
    @rtype: tuple
    """

    min_shift1 = 0.
    max_shift1 = 0.
    min_shift2 = 0.
    max_shift2 = 0.

    # Read info from wavecal parameters table.
    wcp_info = cosutil.getTable (reffiles["wcptab"],
                       filter={"opt_elem": info["opt_elem"]},
                       exactly_one=True)
    wcp_info = wcp_info[0]

    # Get the shifts in dispersion and cross-dispersion directions at the
    # start of the exposure.  If the current exposure was bracketed by
    # two wavecals, the slopes of the shifts can be non-zero.
    shift_info = wavecal.returnWavecalShift (wavecal_info,
                        wcp_info, info["fpoffset"], info["expstart"])
    if shift_info is None:
        return (min_shift1, max_shift1, min_shift2, max_shift2)

    (shift_dict, slope_dict) = shift_info

    if info["detector"] == "FUV":
        segment = info["segment"]
    else:
        segment = "NUVB"

    key = "shift1" + segment[-1].lower()
    shift1_zero = shift_dict[key]
    shift1_slope = slope_dict[key]

    key = "shift2" + segment[-1].lower()
    shift2_zero = shift_dict[key]
    shift2_slope = slope_dict[key]

    # the slope of the shift is in pixels per second
    min_shift1 = shift1_zero
    max_shift1 = shift1_zero + info["exptime"] * shift1_slope
    min_shift2 = shift2_zero
    max_shift2 = shift2_zero + info["exptime"] * shift2_slope

    if max_shift1 < min_shift1:
        (min_shift1, max_shift1) = (max_shift1, min_shift1)
    if max_shift2 < min_shift2:
        (min_shift2, max_shift2) = (max_shift2, min_shift2)

    return (min_shift1, max_shift1, min_shift2, max_shift2)
