#
#  This module has all the code for trace profile application
#  and calculation of centroids for reference profile and science
#  data - the TRCECORR and ALGNCORR steps
#

from __future__ import division, absolute_import
import astropy.io.fits as fits
import numpy as np
import math

from . import airglow
from .calcosparam import *
from . import ccos
from . import cosutil
from . import dispersion

TRACE_OK = 1
CENTROID_OK = 1
CENTROID_UNDEFINED = 2
CENTROID_ERROR_TOO_LARGE = 3
CENTROID_SET_BY_USER = 4
NO_CONVERGENCE = 5

#
# Temporarily use one of the unused bits in the DQ array to mark
# airglow lines

DQ_AIRGLOW = 1024

#
# Minimum number of good columns to make a row 'good'

COLMIN = 100

def doTrace(events, info, reffiles, tracemask):
    """Do the trace correction.  The trace reference file follows the
    centroid of a point source.  Applying the correction involves
    subtracting the profile of trace vs. xcorr from yfull"""
    traceprofile = getTrace(reffiles['tracetab'], info)
    #
    # Remove bad values from the trace
    cleanTrace(traceprofile)
    #
    # Apply the trace correction to the yfull values
    applyTrace(events.field('xcorr'), events.field('yfull'), traceprofile,
               tracemask)
    
    return traceprofile

def getTrace(tracetab, info):
    """Get the trace from the tracetab reference file."""
    filter = {"segment": info["segment"],
              "opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": info["aperture"]
              }
    trace_row = cosutil.getTable(tracetab, filter)
    if trace_row is None:
        raise MissingRowError("Missing row in TRACETAB, filter = %s" %  str(filter))
    trace = trace_row["TRACE"][0]
    return trace

def cleanTrace(trace):
    """Clean the trace.  For now, this just means replacing values
    of -999.0 with 0.0"""
    bad = np.where(trace == -999.0)
    trace[bad] = 0.0
    return

def applyTrace(xcorr, yfull, trace, tracemask):
    """Apply the trace correction"""
    nevents = len(xcorr)
    ixcorr = xcorr.astype(np.int32)
    #
    # remainder and delta are 1-d arrays
    remainder = xcorr - ixcorr
    delta = (1.0-remainder)*trace[ixcorr] + remainder*trace[ixcorr+1]
    yfull[tracemask] = yfull[tracemask] - delta[tracemask]
    return

def doProfileAlignment(events, input, info, switches, reffiles, phdr, hdr,
                       dq_array, tracemask):
    #
    # First look for a user-defined value for the alignment correction
    # in the SP_SET_A/B keyword
    try:
        segmentletter = info["segment"][-1]
        key = "SP_SET_" + segmentletter
        user_offset = hdr[key]
    except KeyError:
        user_offset = None
    #
    # Now do the profile correction. First we calculate
    # the centroid of the profile
    cosutil.printMsg("Calculating centroid")
    filter = {"segment": info["segment"],
              "opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": info["aperture"]}
    #
    #   Use the value of the XTRCTALG keyword to decide what to use to
    #   define regions:
    #   BOXCAR:  Use XTRACTAB
    #   TWOZONE: Use TWOZXTAB
    if info["xtrctalg"] == "BOXCAR":
        xtract_info = cosutil.getTable(reffiles["xtractab"], filter)
        if xtract_info is None:
            raise MissingRowError("Missing row in XTRACTAB; filter = %s" %
                                  str(filter))
    else:
        xtract_info = cosutil.getTable(reffiles["twozxtab"], filter)
        if xtract_info is None:
            raise MissingRowError("Missing row in TWOZXTAB; filter = %s" %
                                  str(filter))
        #
        # Make sure the table doesn't have a SLOPE column
        try:
            slope = xtract_info.field("SLOPE")[0]
            cosutil.printWarning("TWOZXTAB file has a SLOPE column")
        except KeyError:
            slope = 0.0
    #
    # Rebin the data.  Output is the flatfielded image, with events binned
    # in (XFULL, YFULL) space
    rebinned_data = rebinData(events, info)
    nrows, ncols = rebinned_data.shape
    #
    # Mask airglow lines in the DQ image
    maskAirglowLines(dq_array, info, reffiles['disptab'], DQ_AIRGLOW)
    #
    # Apply wavelength limits specified in the PROFTAB reference file
    # so we don't centroid over noise
    applyWavelengthLimits(dq_array, info, reffiles['proftab'],
                          reffiles['disptab'])
    #
    # Determine which columns are good by checking the dq flags in the
    # extraction regions of the background and source
    # The bit value for SDQFLAGS comes from the science data header
    sdqflags = hdr["SDQFLAGS"]
    dqexclude = sdqflags | DQ_AIRGLOW
    centroid = None
    goodcolumns = None
    regions = None
    try:
        slope = xtract_info.field('SLOPE')[0]
    except KeyError:
        slope = 0.0
    startcenter = xtract_info.field('B_SPEC')[0] + slope*(ncols / 2)
    #
    # Calculate the centroid in the science data
    status, centroid, goodcolumns, regions = getScienceCentroid(rebinned_data,
                                                                dq_array,
                                                                xtract_info,
                                                                dqexclude,
                                                                centroid)
    #
    # Calculate the error on the centroid using the counts data which
    # will be calculated from the event list
    error = getCentroidError(events, info, goodcolumns, regions)
    if error is not None:
        cosutil.printMsg("Error on centroid = %f" % (error))
    else:
        cosutil.printWarning("Centroid error not defined in science data")
    #
    # Get the YERRMAX parameter from the twozxtab
    try:
        max_err = xtract_info["yerrmax"][0]
    except KeyError:
        raise MissingColumnError("Missing YERRMAX column in TWOZXTAB, filter = %s" %
                                 str(filter))        
    if error is not None and error > max_err:
        cosutil.printWarning("Error on flux-weighted centroid > %f" % max_err)
        status = CENTROID_ERROR_TOO_LARGE
    ref_centroid = getReferenceCentroid(reffiles['proftab'],
                                        info,
                                        goodcolumns,
                                        regions)
    #
    # Calculate the offset
    if (user_offset is None) or (user_offset < -990.0):
        if status == CENTROID_OK:
            offset = ref_centroid - centroid
            cosutil.printMsg("Using calculated offset of %f" % (offset))
        else:
            #
            # If something went wrong with the centroid calculation, use
            # the center from the xtractab or twozxtab in calculating the
            # offset to make the centroid the same as that in the reference
            # profile
            offset = ref_centroid - startcenter
            cosutil.printMsg("Using offset of {:+f} calculated using".format(offset))
            cosutil.printMsg("initial reference file aperture center")
    else:
        offset = -user_offset
        cosutil.printMsg("Using user-supplied offset of %f" % (offset))
        status = CENTROID_SET_BY_USER
    #
    # Apply the offset.  When this is done, the science target and the
    # reference profile will have the same centroid
    # Only do the correction to events inside the active area and outside
    # the WCA aperture
    applyOffset(events.field('yfull'), offset, tracemask)
    updateTraceKeywords(hdr, info["segment"], ref_centroid, offset, error)
    return status

def rebinData(events, info):
    rebinned = np.zeros(info['npix'], dtype=np.float32)
    ccos.binevents(events.field('xfull'),
                   events.field('yfull'),
                   rebinned,
                   info['x_offset'],
                   events.field('dq'),
                   SERIOUS_DQ_FLAGS,
                   events.field('epsilon'))
    return rebinned

def rebinCounts(events, info):
    rebinnedCounts = np.zeros(info['npix'], dtype=np.float32)
    ccos.binevents(events.field('xfull'),
                   events.field('yfull'),
                   rebinnedCounts,
                   info['x_offset'],
                   events.field('dq'),
                   SERIOUS_DQ_FLAGS)
    return rebinnedCounts

def maskAirglowLines(dq_image, info, disptab, airglow_bits):
    segment = info["segment"]
    for airglow_line in airglow.AIRGLOW_WAVELENGTHS:
        limits = airglow.findAirglowLimits(info, segment,disptab,
                                           airglow_line)
        if limits is not None:
            colstart, colstop = limits
            dq_image[:,int(colstart):int(colstop+1)] = \
            np.bitwise_or(dq_image[:,int(colstart):int(colstop+1)], airglow_bits)
    return

def applyWavelengthLimits(dq_image, info, proftab, disptab):
    """Get the min and max wavelength for our setting.  Calculate
    column numbers for those wavelengths and set dq_image to SERIOUS_DQ_FLAGS
    for columns outside those columns"""
    #
    # Select the row on OPT_ELEM, CENWAVE, SEGMENT and APERTURE
    filter = {'OPT_ELEM': info['opt_elem'],
              'CENWAVE': info['cenwave'],
              'SEGMENT': info['segment'],
              'APERTURE': info['aperture']
              }
    prof_row = cosutil.getTable(proftab, filter)
    if prof_row is None:
        raise MissingRowError("Missing row in PROFTAB: filter = %s" %
                              str(filter))
    wmin, wmax = getWavelengthLimits(prof_row)
    #
    # Now get the dispersion information
    disp_rel = dispersion.Dispersion(disptab, filter)
    min_column = float(disp_rel.evalInvDisp(wmin, tiny=1.0e-8))
    max_column = float(disp_rel.evalInvDisp(wmax, tiny=1.0e-8))
    cosutil.printMsg("Lower wavelength limit of %f corresponds to column %d" % \
                         (wmin, int(min_column)))
    cosutil.printMsg("Upper wavelength limit of %f corresponds to column %d" % \
                         (wmax, int(max_column)))
    if min_column >= 0:
        dq_image[:, 0:int(min_column+1)] = DQ_AIRGLOW
    if max_column <= FUV_X:
        dq_image[:, int(max_column):] = DQ_AIRGLOW
    return

def getWavelengthLimits(prof_row):
    return (900.0, 2100.0)

def getGoodRows(dq_array, dqexclude):
    #
    # Get the first and last good rows in the science data
    nrows, ncols = dq_array.shape
    masked = np.bitwise_and(dq_array, dqexclude)
    mask = np.where(masked != 0, 1, 0)
    rowsum = mask.sum(axis=1)
    #
    # Need to have at least COLMIN good columns
    goodrows = np.where(rowsum < (ncols-COLMIN))
    firstgood = goodrows[0][0]
    lastgood = goodrows[0][-1]
    return firstgood, lastgood

def getRegions(dq_array, xtract_info, dqexclude, center):
    regions = {'bg1start': 0,
               'bg1stop': 0,
               'bg2start': 0,
               'bg2stop': 0,
               'specstart': 0,
               'specstop': 0,
               'firstgoodrow' : 0,
               'lastgoodrow' : 0,
               'center' : 0.0
               }
    #
    # First get the range of rows to use
    regions['firstgoodrow'], regions['lastgoodrow'] = getGoodRows(dq_array,
                                                                  dqexclude)
    nrows, ncols = dq_array.shape
    # BHEIGHT is the width for the background extraction regions
    bgheight = xtract_info["BHEIGHT"][0]
    if center is None:
        # Background and source extraction regions in the XTRACTAB and TWOZXTAB
        # are slanted relative to rows at an angle with tangent = SLOPE.
        # We can assume that the trace takes out the overall slope.
        # The limits of the extraction are calculated at the first column,
        # whereas we need them at the center.  Hence we correct by the
        # delta_y parameter below.
        #
        try:
            slope = xtract_info["SLOPE"][0]
        except KeyError:
            slope = 0.0
        delta_y = slope*ncols/2
        center = xtract_info["B_SPEC"][0] + delta_y
    bg1offset = int(round(xtract_info["B_BKG1"][0] - xtract_info["B_SPEC"][0]))
    regions['bg1start'] = int(round(center)) + bg1offset - bgheight//2
    regions['bg1stop'] = int(round(center)) + bg1offset + bgheight//2
    bg2offset = int(round(xtract_info["B_BKG2"][0] - xtract_info["B_SPEC"][0]))
    regions['bg2start'] = int(round(center)) + bg2offset - bgheight//2
    regions['bg2stop'] = int(round(center)) + bg2offset + bgheight//2
    #
    # Make sure the background regions doesn't extend into bad rows
    regions['bg1start'] = max(regions['bg1start'], regions['firstgoodrow'])
    regions['bg2stop'] = min(regions['bg2stop'], regions['lastgoodrow'])
    specheight = xtract_info["HEIGHT"][0]
    regions['specstart'] = int(round(center)) - specheight//2
    regions['specstop'] = int(round(center)) + specheight//2
    regions['center'] = center
    return regions

def getGoodColumns(dq_array, dqexclude_target, dqexclude_background, regions):
    #
    # Only use columns where the DQ values are good for both background
    # regions and the source
    rowstart = regions['bg1start']
    rowstop = regions['bg1stop']
    dqsum1 = np.bitwise_and(dq_array[int(rowstart):int(rowstop+1)],
                            dqexclude_background).sum(axis=0)

    rowstart = regions['bg2start']
    rowstop = regions['bg2stop']
    dqsum2 = np.bitwise_and(dq_array[int(rowstart):int(rowstop+1)],
                            dqexclude_background).sum(axis=0)

    rowstart = regions['specstart']
    rowstop = regions['specstop']
    dqsum3 = np.bitwise_and(dq_array[int(rowstart):int(rowstop+1)],
                            dqexclude_target).sum(axis=0)

    dqsum = dqsum1 + dqsum2 + dqsum3
    goodcolumns = np.where(dqsum == 0)
    return goodcolumns

def getScienceCentroid(rebinned_data, dq_array, xtract_info,
                       dqexclude, centroid):
    n_iterations = 5
    nrows, ncols = rebinned_data.shape
    center = None
    try:
        slope = xtract_info.field('SLOPE')[0]
    except KeyError:
        slope = 0.0
    startcenter = xtract_info.field('B_SPEC')[0] + slope*(ncols / 2)
    cosutil.printMsg('Starting center = %f' % startcenter)
    for iteration in range(n_iterations):
        #
        # Determine which columns to use.  The rows to be used are centered
        # on the value of the variable 'center', which gets updated between
        # iterations, so the good columns can change from one iteration to
        # the next
        regions = getRegions(dq_array, xtract_info, dqexclude, center)
        #
        # Get the  DQ values for target and background regions.  Both
        # use header value of SDQFLAGS or'd with DQ_AIRGLOW
        # with the gain sag hole DQ taken out.  Put this in a couple of
        # functions so we can change if needed
        dqexclude_target = getTargetDQ(dqexclude)
        dqexclude_background = getBackgroundDQ(dqexclude)
        goodcolumns = getGoodColumns(dq_array, dqexclude_target,
                                     dqexclude_background, regions)
        n_goodcolumns = len(goodcolumns[0])
        cosutil.printMsg("%d good columns" % (len(goodcolumns[0])))
        if n_goodcolumns == 0:
            #
            # If this happens, it means the region over which we are calculating the
            # centroid and associated backgrounds is overlapping part of the detector
            # with non-zero DQ (usually the DQ=8/poorly calibrated regions at the top
            # and bottom edges of the active area)
            # In that case, return with a status of CENTROID_UNDEFINED and set the
            # centroid to the center from the reference file and the goodcolumns and
            # regions to what are appropriate for that centroid
            centroid = startcenter
            center = None
            regions = getRegions(dq_array, xtract_info, dqexclude, center)
            goodcolumns = getGoodColumns(dq_array, dqexclude_target,
                                         dqexclude_background, regions)
            status = CENTROID_UNDEFINED
            cosutil.printMsg("No good columns.")
            cosutil.printMsg("Centroid set to input value from reference file: {}".format(centroid))
            return status, centroid, goodcolumns, regions
        #
        # Now calculate the background.  Use the XTRACTAB reference file
        # to determine the background regions but center them on the 'center'
        # parameter
        background = getBackground(rebinned_data, goodcolumns, regions)
        cosutil.printMsg("Adopted background in science data = %f" %
                         (background))
        rowstart = regions['specstart']
        rowstop = regions['specstop']
        centroid = getCentroid(rebinned_data,
                               goodcolumns,
                               rowstart,
                               rowstop,
                               background)
        if centroid is not None:
            cosutil.printMsg("Centroid = %f" % (centroid))
            difference = (centroid - regions['center'])
            if abs(difference) > 20:
                cosutil.printMsg("Centroid shift too big")
                status = CENTROID_UNDEFINED
                break
            elif abs(difference) < 0.005:
                cosutil.printMsg("Centroid calculation converged")
                status = CENTROID_OK
                break
        else:
            cosutil.printWarning("Centroid is not defined in science data")
            status = CENTROID_UNDEFINED
        center = centroid
        if iteration == n_iterations - 1:
            #
            # If we get here, it means we didn't converge after the maximum
            # number of iterations
            cosutil.printWarning("Centroid calculation did not converge")
            cosutil.printWarning("after %d iterations" % (n_iterations))
            status = NO_CONVERGENCE
            centroid = difference
    return status, centroid, goodcolumns, regions

def getBackgroundDQ(dqexclude):
    """Return the DQ value to be used for the background regions.  For now, this
    means removing the bit corresponding to DQ_GAIN_SAG_HOLE.  To unset a bit,
    AND it with NOT"""
    return dqexclude&(~DQ_GAIN_SAG_HOLE)

def getTargetDQ(dqexclude):
    """Return the DQ value to be used for the target region.  For now, this
    means removing the bit corresponding to DQ_GAIN_SAG_HOLE.  To unset a bit,
    AND it with NOT"""
    return dqexclude&(~DQ_GAIN_SAG_HOLE)

def getBackground(data_array, goodcolumns, regions):
    bg1start = regions['bg1start']
    bg1stop = regions['bg1stop']
    bg2start = regions['bg2start']
    bg2stop = regions['bg2stop']
    cosutil.printMsg("Background regions are from %d to %d" % (bg1start,
                                                               bg1stop))
    cosutil.printMsg("and from %d to %d" % (bg2start, bg2stop))
    bkg1 = data_array[int(bg1start):int(bg1stop+1)].mean(axis=0,
                                                         dtype=np.float64)[goodcolumns].mean(dtype=np.float64)
    nbkg1 = bg1stop - bg1start + 1
    bkg2 = data_array[int(bg2start):int(bg2stop+1)].mean(axis=0,
                                                         dtype=np.float64)[goodcolumns].mean(dtype=np.float64)
    nbkg2 = bg2stop - bg2start + 1
    bkg = (nbkg1*bkg1 + nbkg2*bkg2)/float(nbkg1 + nbkg2)
    return bkg

def getCentroid(data_array, goodcolumns, rowstart, rowstop, background=None):
    """Calculate the centroid of data_array."""
    rows = np.arange(rowstart, rowstop+1)
    if background is None: background = 0.0
    i = data_array[int(rowstart):int(rowstop+1)] - background
    y = rows[:,np.newaxis]
    sumy = (i*y).sum(axis=0, dtype=np.float64)[goodcolumns].sum(dtype=np.float64)
    sumi = i.sum(axis=0,dtype=np.float64)[goodcolumns].sum(dtype=np.float64)
    if sumi != 0.0:
        ycent = sumy/sumi
        return ycent
    else:
        return None

def getCentroidError(events, info, goodcolumns, regions):
    #
    # Calculate the error on the centroid.
    # Use counts data, not flatfielded data.
    # Use the same columns as were used to calculate regular centroid
    # Calculate background for counts data
    # Calculate centroid of counts data
    # Calculate error in centroid for counts data
    # Even though the centroid and background may be different, the error should
    # be appropriate
    #
    if len(goodcolumns[0]) == 0:
        cosutil.printMsg("No good columns, cannot calculate centroid error")
        return None
    # First need to rebin evens to counts
    counts_ij = rebinCounts(events, info)
    #
    # Now need to calculate the background
    background = getBackground(counts_ij, goodcolumns, regions)
    #
    # Calculate centroid
    rowstart = regions['specstart']
    rowstop = regions['specstop']
    centroid = getCentroid(counts_ij, goodcolumns, rowstart, rowstop,
                           background)
    error = calculateCentroidError(counts_ij, goodcolumns, regions,
                                   centroid, background=background)
    return error

def calculateCentroidError(data_ij, goodcolumns, regions,
                           centroid, background=0.0):
    rowstart = regions['specstart']
    rowstop = regions['specstop']
    rows = np.arange(rowstart, rowstop+1)
    if background is None: background = 0.0
    i = data_ij[int(rowstart):int(rowstop+1)]
    y = rows[:,np.newaxis]
    sumi = (i - background).sum(axis=0, dtype=np.float64)[goodcolumns].sum(dtype=np.float64)
    sumsq = (i*(y - centroid) * \
                 (y - centroid)).sum(axis=0,
                                     dtype=np.float64)[goodcolumns].sum(dtype=np.float64)
    if sumi != 0.0:
        error = sumsq / sumi / sumi
        if error >= 0.0:
            return math.sqrt(error)
        else:
            return None
    else:
        return None

def getReferenceBackground(profile, goodcolumns, refcenter, regions):
    nrows, ncols = profile.shape
    if refcenter is None:
        refcenter = nrows/2
    offset = int(round(refcenter - regions['center']))
    bg1start = offset + regions['bg1start']
    bg1stop = offset + regions['bg1stop']
    bg2start = offset + regions['bg2start']
    bg2stop = offset + regions['bg2stop']
    if bg1start >= 0:
        bkg = profile[int(bg1start):int(bg1stop+1)].mean(axis=0,
                                                         dtype=np.float64)[goodcolumns].mean(dtype=np.float64)
        wbkg = 1.0
    elif bg1stop > 0:
        cosutil.printMsg("Background region #1 truncated")
        cosutil.printMsg("Reguested region: %d to %d" % (bg1start, bg1stop))
        wbkg = float(bg1stop+1)/(bg1stop+1-bg1start)
        bg1start = 0
        cosutil.printMsg("Using %d to %d, with weight %f" % (bg1start, 
                                                             bg1stop, wbkg))
        bkg = profile[int(bg1start):int(bg1stop+1)].mean(axis=0,
                                                         dtype=np.float64)[goodcolumns].mean(dtype=np.float64)
    else:
        cosutil.printMsg("Unable to extract background region #1 from reference profile")
        cosutil.printMsg("Requested rows: %d to %d" % (bg1start, bg1stop))
        bkg = 0
        wbkg = 0.0
    if bg2stop <= nrows:
        bkg = bkg + profile[int(bg2start):int(bg2stop+1)].mean(axis=0,
                                                               dtype=np.float64)[goodcolumns].mean(dtype=np.float64)
        wbkg2 = 1.0
    elif bg2start <= nrows:
        cosutil.printMsg("Background region #2 truncated")
        cosutil.printMsg("Reguested region: %d to %d" % (bg2start, bg2stop))
        wbkg2 = float(nrows-bg2start+1)/(bg2stop+1-bg2start)
        bg2stop = nrows
        cosutil.printMsg("Using %d to %d, with weight %f" % (bg2start,
                                                             bg2stop, wbkg2))
        bkg = bkg + profile[int(bg2start):int(bg2stop+1)].mean(axis=0,
                                                               dtype=np.float64)[goodcolumns].mean(dtype=np.float64)
    else:
        cosutil.printMsg("Unable to extract background region #2 from" \
                             " reference profile")
        cosutil.printMsg("Requested rows: %d to %d" % (bg2start, bg2stop))
        wbkg2 = 0.0
    wbkg = wbkg + wbkg2
    if wbkg > 0.0:
        cosutil.printMsg("Adopted background regions for reference profile:")
        cosutil.printMsg("Rows %d to %d, and rows %d to %d" % (bg1start,
                                                               bg1stop,
                                                               bg2start,
                                                               bg2stop))
        background = bkg/wbkg
    else:
        cosutil.printMsg("Unable to determine background in reference profile")
        background = 0.0
    cosutil.printMsg("Background = %f" % background)
    return background

def getReferenceCentroid(proftab, info, goodcolumns, regions):
    #
    # Select the row on OPT_ELEM, CENWAVE, SEGMENT and APERTURE
    filter = {'OPT_ELEM': info['opt_elem'],
              'CENWAVE': info['cenwave'],
              'SEGMENT': info['segment'],
              'APERTURE': info['aperture']
              }
    prof_row = cosutil.getTable(proftab, filter)
    if prof_row is None:
        raise MissingRowError("Missing row in PROFTAB: filter = %s" %
                              str(filter))
    else:
        cosutil.printMsg("Using profile reference file %s" % (proftab))
    #
    # Get the profile and center
    profile = prof_row['PROFILE'][0]
    nrows, ncols = profile.shape
    row_0 = prof_row['ROW_0'][0]
    refcenter = prof_row['CENTER'][0]
    n_iterations = 5
    for iteration in range(n_iterations):
        profcenter = refcenter - row_0
        cosutil.printMsg("Input Reference centroid = %f, row_0 = %d" %
                         (refcenter, row_0))
        offset = int(round(profcenter - regions['center']))
        rowstart = offset + regions['specstart']
        rowstop = offset + regions['specstop']
        background = getReferenceBackground(profile, goodcolumns,
                                            profcenter, regions)
        centroid = getCentroid(profile, goodcolumns, rowstart, rowstop, 
                               background=background)
        ref_centroid =  centroid + row_0
        cosutil.printMsg("Measured reference centroid = %f" %
                         (ref_centroid))
        difference = (refcenter - ref_centroid)
        if abs(difference) < 0.005:
            cosutil.printMsg("Reference centroid calculation converged")
            break
        refcenter = ref_centroid
        if iteration == n_iterations -1:
            #
            # If we get here, it means we didn't converge after the maximum
            # number of iterations
            cosutil.printWarning("Reference centroid calculation did not converge")
            cosutil.printWarning("after %d iterations" % (n_iterations))
    return refcenter

def applyOffset(yfull, offset, tracemask):
    yfull[tracemask] = yfull[tracemask] + offset
    return

def updateTraceKeywords(hdr, segment, ref_centroid, offset, error):
    segmentlist = ['A', 'B']
    segment_letter = segment[-1]
    segmentlist.remove(segment_letter)
    othersegment = segmentlist[0]
    key = "SP_LOC_" + segment_letter
    hdr[key] = ref_centroid
    key = "SP_ERR_" + segment_letter
    if error is not None:
        hdr[key] = error
    else:
        hdr[key] = -999.0
    key = "SP_ERR_" + othersegment
    try:
        temp = hdr[key]
    except KeyError:
        hdr[key] = -999.0
    key = "SP_OFF_" + segment_letter
    hdr[key] = -offset
    return
