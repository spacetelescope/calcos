from __future__ import absolute_import, division         # confidence high
import copy
import math
import shutil
import time
import types
import numpy as np
from numpy import random
import astropy.io.fits as fits

from . import cosutil
from . import ccos
from . import phot
from . import timetag                  # actually for more generic functions
from . import wavecal
from .calcosparam import *       # parameter definitions

def accumBasicCalibration(input, inpha, outtag,
                          outflt, outcounts, outcsum,
                          cl_args,
                          info, switches, reffiles,
                          wavecal_info):
    """Do the basic processing for accum data.

    The function value will be zero if there was no problem.

    Parameters
    ----------
    input: str
        Name of the input file.

    inpha: str
        Name of the input file containing the pulse height histogram
        (FUV only).

    outtag: str
        Name of the output file for pseudo time-tag data.

    outflt: str
        Name of the output file for the flat-fielded count-rate image.

    outcounts: str
        Name of the output file for the count-rate image.

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

    cosutil.printIntro("ACCUM calibration")
    if info["exptype"] == "ACQ/IMAGE":
        names = [("Input", input),
                 ("OutFlt", outflt), ("OutCounts", outcounts)]
    else:
        names = [("Input", input), ("pseudo tt", outtag),
                 ("OutFlt", outflt), ("OutCounts", outcounts)]
    if info["detector"] == "FUV":
        names.insert(1, ("InPha", inpha))
    if outcsum is not None:
        names.append(("OutCsum", outcsum))
    cosutil.printFilenames(names,
                           shift_file=cl_args["shift_file"],
                           stimfile=cl_args["stimfile"],
                           livetimefile=cl_args["livetimefile"])
    cosutil.printMode(info)

    if info["corrtag_input"]:
        shutil.copy(input, outtag)
        status = timetag.timetagBasicCalibration(input, inpha, outtag,
                    outflt, outcounts, None, outcsum,
                    cl_args, info, switches, reffiles, wavecal_info)
        return status

    # Get a list of all the headers in the input file.
    headers = cosutil.getHeaders(input)
    phdr = headers[0]
    # Get x_offset now, because overrideKeywords may change it in headers[1].
    x_offset = headers[1].get("x_offset", 0)

    # Update the switches and reference file names, so the output header
    # will reflect what was actually used.
    cosutil.overrideKeywords(phdr, headers[1], info, switches, reffiles)

    # acq/image data are processed differently because they have two imsets.
    if info["exptype"] == "ACQ/IMAGE":
        acqImage(input, outflt, outcounts, outcsum, cl_args,
                 info, switches, reffiles)

    if info["exptype"] == "ACQ/IMAGE":
        return 0

    # Check for null science data.
    if info["npix"] == (0,):
        nrows = 0
    else:
        # Open the accum image.
        # The number of rows in the pseudo time-tag table will be equal to
        # the total number of counts in the input image.
        fd = fits.open(input, mode="copyonwrite")
        sci_hdu = fd[("SCI", 1)]
        sci = fd[("SCI",1)].data
        fd.close()
        nrows = getNcounts(sci)
        if nrows == 0:
            info["npix"] = (0,)

    hdu = cosutil.createCorrtagHDU(nrows, info["detector"], sci_hdu)
    hdu.header["extname"] = "EVENTS"

    if nrows > 0:
        # Create pseudo-timetag arrays (x & y, no time) from the raw image.
        x = np.zeros(nrows, dtype=np.float32)
        y = np.zeros(nrows, dtype=np.float32)
        ccos.unbinaccum(sci, x, y, x_offset)

        # Copy x and y to the pseudo time-tag table.
        outdata = hdu.data
        outdata.field("TIME")[:] = info["exptime"] / 2.
        outdata.field("RAWX")[:] = x
        outdata.field("RAWY")[:] = y
        outdata.field("XCORR")[:] = x
        outdata.field("YCORR")[:] = y
        outdata.field("XDOPP")[:] = x
        outdata.field("XFULL")[:] = x
        outdata.field("YFULL")[:] = y
        outdata.field("WAVELENGTH")[:] = np.zeros(nrows, dtype=np.float32)
        outdata.field ("EPSILON")[:] = np.ones(nrows, dtype=np.float32)
        outdata.field("DQ")[:] = np.zeros(nrows, dtype=np.int16)
        outdata.field("PHA")[:] = 0

    primary_hdu = fits.PrimaryHDU(header=phdr)
    ofd = fits.HDUList(primary_hdu)
    cosutil.updateFilename(ofd[0].header, outtag)
    ofd.append(hdu)
    ofd.append(cosutil.dummyGTI(info["exptime"]))
    ofd[0].header["nextend"] = len(ofd) - 1     # number of extensions

    ofd.writeto(outtag)
    del ofd

    status = timetag.timetagBasicCalibration(input, inpha, outtag,
                    outflt, outcounts, None, outcsum,
                    cl_args, info, switches, reffiles, wavecal_info)

    return status

def acqImage(input, outflt, outcounts, outcsum, cl_args,
             info, switches, reffiles):
    """Do the calibration for ACQ/IMAGE data."""

    livetimefile = cl_args["livetimefile"]

    fd = fits.open(input, mode="copyonwrite")
    nextend = len(fd) - 1
    nimsets = len(fd) // 3
    phdr = fd[0].header
    phdr["cal_ver"] = info["cal_ver"]
    hdr_list = []

    writePrimaryHDU(outcounts, phdr, nextend)
    writePrimaryHDU(outflt, phdr, nextend)
    if outcsum is not None:
        # we'll add the SCI array for each imset to this array
        csum_array = np.zeros((NUV_Y, NUV_X), dtype=np.float32)

    for imset in range(1, nimsets+1):

        sci_hdr = fd[("SCI",imset)].header
        err_hdr = fd[("ERR",imset)].header
        dq_hdr = fd[("DQ",imset)].header
        hdr_list.append(sci_hdr)

        counts_sci = fd[("SCI",imset)].data
        #
        # Allow subsequent floating point processing
        # astype returns a copy by default, not a view
        flt_sci = counts_sci.astype(np.float64)

        dq_array = cosutil.getInputDQ(input, imset)

        doPhotcorr(info, switches, reffiles["imphttab"], phdr, sci_hdr)

        updateGlobrate(sci_hdr, counts_sci, info["exptime"])

        doDqicorr(info, switches, reffiles, phdr, dq_array)

        doDeadcorr(flt_sci, sci_hdr["exptime"], info, switches, reffiles,
                   phdr, sci_hdr, input, livetimefile)

        if outcsum is not None:
            csum_array += flt_sci

        doFlatcorr(flt_sci, switches, reffiles, phdr)

        (C_rate, errC_rate, E_rate, errE_rate) = makeImages(
                        counts_sci, flt_sci, sci_hdr["exptime"])
        appendImset(outcounts, imset, C_rate, errC_rate, dq_array,
                    sci_hdr, err_hdr, dq_hdr)
        appendImset(outflt, imset, E_rate, errE_rate, dq_array,
                    sci_hdr, err_hdr, dq_hdr)
    doStatflag(switches, outflt, outcounts)

    # Calibration switch keywords have been updated in phdr, but the output
    # primary headers have already been written to disk, so those files need
    # to be reopened in order to update these keywords.
    updateSwitches(phdr, outflt, outcounts)

    if outcsum is not None:
        writeCsum(outcsum, phdr, hdr_list, csum_array,
                  cl_args["raw_csum_coords"],
                  cl_args["binx"], cl_args["biny"],
                  cl_args["compress_csum"],
                  cl_args["compression_parameters"])

    fd.close()
    if not (info["aperture"] in APERTURE_NAMES or
            info["targname"] == "DARK" and
            info["aperture"] in OTHER_APERTURE_NAMES):
        raise BadApertureError("APERTURE = %s is not a valid aperture name." %
                               info["aperture"])

def updateGlobrate(hdr, data, exptime):
    """Update the GLOBRATE keyword in the extension header.

    Parameters
    ----------
    hdr: pyfits Header object
        The input events extension header.

    data: array_like
        Data array.

    exptime: float
        Exposure time in seconds.
    """

    if data is None or len(data) == 0 or exptime <= 0.:
        globrate = 0.
    else:
        globrate = data.sum(dtype=np.float64) / exptime

    globrate = round(globrate, 4)
    hdr["globrate"] = globrate

def doPhotcorr(info, switches, imphttab, phdr, hdr):
    """Update photometry parameter keywords for imaging data.

    Parameters
    ----------
    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    imphttab: str
        The name of the imaging photometric parameters table.

    phdr: pyfits Header object
        The primary header, photcorr keyword updated in-place.

    hdr: pyfits Header object
        The first extension header, updated in-place.
    """

    if info["obstype"] == "IMAGING" and info["detector"] == "NUV":
        cosutil.printSwitch("PHOTCORR", switches)
        if switches["photcorr"] == "PERFORM":
            # If aperture is invalid, phot.doPhot will use PSA.
            obsmode = "cos,nuv," + info["opt_elem"] + "," + info["aperture"]
            phot.doPhot(imphttab, obsmode, hdr)
            phdr["photcorr"] = "COMPLETE"

def doDqicorr(info, switches, reffiles, phdr, dq_array):
    """Update the DQ array using the DQI table.

    Parameters
    ----------
    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: pyfits Header object
        Primary header from input file.

    dq_array: array_like
        DQ array from input file, or an array of zeros; this will be
        modified in-place.
    """

    cosutil.printSwitch("DQICORR", switches)

    if switches["dqicorr"] == "PERFORM":

        cosutil.printRef("BPIXTAB", reffiles)

        # update using imaging wavecal xxx
        minmax_shift_dict = {(0, 1024): [0., 0., 0., 0.]}
        minmax_doppler = (0., 0.)
        doppler_boundary = -10          # anywhere below 0
        cosutil.updateDQArray(info, reffiles, dq_array,
                              minmax_shift_dict,
                              minmax_doppler, doppler_boundary, None)

        phdr["dqicorr"] = "COMPLETE"

def doDeadcorr(flt_sci, exptime, info, switches, reffiles,
               phdr, hdr, input, livetimefile):
    """Correct for deadtime."""

    cosutil.printSwitch("DEADCORR", switches)

    if switches["deadcorr"] == "PERFORM":
        cosutil.printRef("DEADTAB", reffiles)
        (dead_rate, dead_method, livetime) = \
                deadtimeCorrection(flt_sci, exptime, reffiles["deadtab"],
                                   info, input, livetimefile)
        hdr["deadrt"] = dead_rate
        hdr["deadmt"] = dead_method
        hdr["livetm"] = livetime
        if dead_method == "SKIPPED":
            # Deadcorr would already be set to COMPLETE if it was done for
            # the first imset, in which case don't change it.  So COMPLETE
            # means it was done for at least one imset.
            if phdr["deadcorr"] != "COMPLETE":
                phdr["deadcorr"] = "SKIPPED"
        else:
            phdr["deadcorr"] = "COMPLETE"

def deadtimeCorrection(flt_sci, exptime, deadtab, info,
                       input, livetimefile):
    """Determine and apply the livetime factor.

    If there are subarrays, the livetime factor is gotten from the digital
    event counter.  If there are no subarrays, the livetime factor is based
    on the actual count rate.

    Parameters
    ----------
    flt_sci: array_like
        The SCI image array, to be corrected in-place.

    exptime: float
        Exposure time for current imset.

    deadtab: str
        Name of reference table of count rates and livetime factors.

    info: dictionary
        Header keywords and values.

    input: str
        Name of input raw file (for writing to livetimefile).

    livetimefile: str or None
        Name of output text file for livetime factors.

    Returns
    -------
    (dead_rate, dead_method, livetime): tuple of float, str, float
        dead_rate is the count rate used for determining the livetime
        factor.  dead_method is a string that indicates which method was
        used for determining the livetime factor; dead_method will be
        "SKIPPED" if the exposure time is zero.  livetime is the livetime
        factor that was used.
    """

    if exptime <= 0.:
        cosutil.printWarning("Can't do deadcorr, exptime = %.6g." % exptime)
        return (0., "SKIPPED", 1.)

    if livetimefile is None:
        fd = None
    else:
        fd = open(livetimefile, "a")
        fd.write("# %s\n" % (input,))

    ncounts = getNcounts(flt_sci)

    live_info = cosutil.getTable(deadtab,
                                 filter={"segment": info["segment"]},
                                 at_least_one=True)
    obs_rate = live_info.field("obs_rate")
    live_factor = live_info.field("livetime")

    # keyword used to print information
    keyword = "MEVENTS"

    # Output count rate from digital event counter (DEC), and corresponding
    # livetime factor.
    dec_countrate = info["countrate"]
    dec_livetime = cosutil.determineLivetime(dec_countrate,
                                             obs_rate, live_factor)
    actual_countrate = float(ncounts) / exptime
    actual_rate_livetime = cosutil.determineLivetime(actual_countrate,
                                                     obs_rate, live_factor)

    if info["subarray"]:
        livetime_source = "digital event counter (%s)" % keyword
        livetime = dec_livetime
        dead_rate = dec_countrate
        dead_method = "MEVENTS"
    else:
        livetime_source = "actual count rate"
        livetime = actual_rate_livetime
        dead_rate = actual_countrate
        dead_method = "DATA"

    flt_sci /= livetime

    print_details = (cosutil.checkVerbosity(VERY_VERBOSE))      # initial value

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

    if fd is not None:
        fd.write("actual countrate and livetime:  %.6g, %6.4f\n" %
                 (actual_countrate, actual_rate_livetime))
        fd.write("countrate and livetime from %s:  %.6g, %6.4f\n" %
                 (keyword, dec_countrate, dec_livetime))
        fd.write("livetime %6.4f is based on %s.\n" % \
                 (livetime, livetime_source))

    if fd is not None:
        fd.close()

    return (dead_rate, dead_method, livetime)

def doFlatcorr(flt_sci, switches, reffiles, phdr):
    """Apply flat field correction.

    Parameters
    ----------
    flt_sci: array_like
        The image array, modified in-place.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: pyfits Header object
        The input primary header.
    """

    cosutil.printSwitch("FLATCORR", switches)

    if switches["flatcorr"] == "PERFORM":

        cosutil.printRef("FLATFILE", reffiles)

        fd = fits.open(reffiles["flatfile"], mode="copyonwrite")
        hdu = fd[1]
        flat = hdu.data
        fd.close()

        (ny, nx) = flat.shape
        x0 = hdu.header.get("origin_x", 0)
        y0 = hdu.header.get("origin_y", 0)

        flt_sci[int(y0):int(y0+ny),int(x0):int(x0+nx)] /= flat

        phdr["flatcorr"] = "COMPLETE"

def doStatflag(switches, outflt, outcounts):
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
        cosutil.doImageStat(outflt)

def updateSwitches(phdr, outflt, outcounts):
    """Update calibration switch keywords in output primary headers.

    Parameters
    ----------
    phdr: pyfits Header object
        The input primary header.

    outflt: str
        Name of the output file for the flat-fielded count-rate image.

    outcounts: str
        Name of the output file for the count-rate image.
    """

    for filename in [outflt, outcounts]:
        fd = fits.open(filename, mode="update")
        for keyword in ["photcorr", "dqicorr", "deadcorr", "flatcorr"]:
            fd[0].header[keyword] = phdr[keyword]
        fd.close()

def makeImages(counts_sci, flt_sci, exptime):
    """Create the count rate and error arrays.

    Parameters
    ----------
    counts_sci: array_like
        The SCI image array, counts.

    flt_sci: array_like
        The SCI image array, after deadcorr and flatcorr.

    exptime: float
        The exposure time.

    Returns
    -------
    (C_rate, errC_rate, E_rate, errE_rate): tuple of floats
        C_rate is the count rate array; errC_rate is the error estimate
        for the count rate array; E_rate is the flat-fielded count rate
        array; errE_rate is the error estimate for the flat fielded
        count-rate array.
    """

    if exptime <= 0:
        cosutil.printWarning(
                "Exposure time is zero, so output files are dummy.")
        C_rate = counts_sci * 0.
        E_rate = C_rate.copy()
        errC_rate = C_rate.copy()
        errE_rate = C_rate.copy()
        return (C_rate, errC_rate, E_rate, errE_rate)

    C_rate = counts_sci / exptime
    E_rate = flt_sci / exptime

    counts_sci_temp = np.where(counts_sci < 0., 0., counts_sci)
    errC_rate = np.sqrt(counts_sci_temp) / exptime
    del counts_sci_temp

    # errC_rate will likely have a number of zero values, so we set those
    # to one before dividing.
    errC_rate_temp = np.where(errC_rate == 0., 1., errC_rate)
    errE_rate = E_rate / errC_rate_temp / exptime
    del errC_rate_temp

    return (C_rate, errC_rate, E_rate, errE_rate)

def writeCsum(outcsum, phdr, hdr_list, csum_array,
              raw_csum_coords,
              binx=None, biny=None,
              compress_csum=False,
              compression_parameters="gzip,-0.1"):
    """Write the "calcos sum" (csum) image.

    Parameters
    ----------
    outcsum: str
        Name of output calcos sum image file.

    phdr: pyfits Header object
        Primary header.

    hdr_list: list of pyfits Header objects
        List of sci extension headers.

    csum_array: array_like
        Data array for SCI extension.

    raw_csum_coords: boolean
        This only affects the COORDFRM keyword value.

    binx: int or None
        Binning factor in the dispersion direction (or None for
        the default binning).

    biny: int or None
        Binning factor in the cross-dispersion direction (or None
        for the default binning).

    compress_csum: boolean
        Compress the csum image?

    compression_parameters: str
        compressionType and quantizeLevel (separated by a comma) for
        the call to fits.CompImageHDU; compressionType can be "rice",
        "gzip", or "hcompress", and quantizeLevel can be e.g. -0.1,
        which means the floating point values will be scaled to integers
        with spacing that corresponds to 0.1 dn (see the doc string for
        fits.CompImageHDU for more details).
    """

    cosutil.printMsg("writing file %s ..." % outcsum, VERY_VERBOSE)

    primary_hdu = fits.PrimaryHDU(header=phdr)
    fd = fits.HDUList(primary_hdu)
    fd[0].header["nextend"] = 1
    fd[0].header["filetype"] = "CALCOS SUM FILE"
    cosutil.updateFilename(fd[0].header, outcsum)
    if raw_csum_coords:
        fd[0].header["coordfrm"] = "raw"
    else:
        fd[0].header["coordfrm"] = "corrected"

    # used later for updating the COUNTS keyword
    counts = csum_array.sum(dtype=np.float64)

    detector = phdr["detector"]

    shape = csum_array.shape
    if binx is None or binx <= 0:
        binx = NUV_BIN_X
    if biny is None or biny <= 0:
        biny = NUV_BIN_Y
    nx = shape[1] // binx
    ny = shape[0] // biny

    # Copy the exposure time keywords to the output headers.
    hdr = hdr_list[0].copy()
    addExptimeKeywords(hdr_list, hdr)

    if binx > 1 or biny > 1:
        binned_array = np.zeros((ny,nx), dtype=np.float32)
        ccos.bin2d(csum_array, binned_array)
    else:
        binned_array = csum_array

    if compress_csum:
        (compType, quantLevel) = compression_parameters.split(",")
        compType = compType.upper() + "_1"
        quantLevel = float(quantLevel)
        fd.append(fits.CompImageHDU(binned_array, header=hdr, name="SCI",
                                    compressionType=compType,
                                    quantizeLevel=quantLevel))
    else:
        fd.append(fits.ImageHDU(data=binned_array, header=hdr, name="SCI"))
    fd[1].header["BUNIT"] = "count"
    fd[1].header["counts"] = counts
    fd[1].header["nuvbinx"] = binx
    fd[1].header["nuvbiny"] = biny

    fd.writeto(outcsum, output_verify="silentfix")

def getNcounts(sci):
    """Return the total number of counts in an array.

    Parameters
    ----------
    sci: array_like
        Image data array, in counts.

    Returns
    -------
    ncounts: int
        Sum of values in sci, rounded to an int.
    """

    ncounts = sci.sum(dtype=np.float64)
    # The value returned by sum() is an "array scalar," so convert it.
    ncounts = float(ncounts)
    ncounts = int(round(ncounts))

    return ncounts

def addExptimeKeywords(hdr_list, hdr):
    """Copy the exposure time keywords to the output headers.

    exptime and rawtime will be the sums of those values from all (both)
    of the input headers, and expstart and expend will be taken from
    the first and last headers.

    Parameters
    ----------
    hdr_list: list of pyfits Header objects
        List of sci extension headers.

    hdr: pyfits Header object
        Output header, modified in-place
    """

    # The headers in hdr_list are assumed to be in chronological order.
    expstart = hdr_list[0].get("expstart", -999.)
    expend = hdr_list[-1].get("expend", -999.)

    exptime = 0.
    rawtime = 0.
    for hdr_i in hdr_list:
        exptime_i = hdr_i.get("exptime", 0.)
        exptime += exptime_i
        rawtime += hdr_i.get("rawtime", exptime_i)

    hdr["expstart"] = expstart
    hdr["expend"] = expend
    hdr["exptime"] = exptime
    hdr["rawtime"] = rawtime

def writePrimaryHDU(output, phdr, nextend):
    """Write an output file containing just a primary header.

    Parameters
    ----------
    output: str
        Name of the output file (flt or counts).

    phdr: pyfits Header object
        Primary header of input file.

    nextend: int
        Number of extensions.
    """

    cosutil.printMsg("writing file %s ..." % output, VERY_VERBOSE)

    primary_hdu = fits.PrimaryHDU(header=phdr)
    fd = fits.HDUList(primary_hdu)
    fd[0].header["nextend"] = nextend
    cosutil.updateFilename(fd[0].header, output)

    fd.writeto(output, output_verify="silentfix")

def appendImset(output, imset, sci_array, err_array, dq_array,
                sci_hdr, err_hdr, dq_hdr):
    """Append an image set (SCI, ERR, DQ extensions).

    This function appends one image set to an output file.

    Parameters
    ----------
    output: str
        Name of the output file (flt or counts).

    imset: int
        Image set number (one indexed, to match EXTVER).

    sci_array: array_like or None
        Data array for the SCI extension.

    err_array: array_like or None
        Data array for the ERR extension.

    dq_array: array_like or None
        Data array for the DQ extension.

    sci_hdr: pyfits Header object
        Header for SCI extension.

    err_hdr: pyfits Header object
        Header for ERR extension.

    dq_hdr: pyfits Header object
        Header for DQ extension.
    """

    fd = fits.open(output, mode="update")

    hdu = fits.ImageHDU(data=sci_array, header=sci_hdr, name="SCI")
    hdu.header["EXTVER"] = imset
    hdu.header["BUNIT"] = "count /s"
    fd.append(hdu)

    hdu = fits.ImageHDU(data=err_array, header=err_hdr, name="ERR")
    hdu.header["EXTVER"] = imset
    hdu.header["BUNIT"] = "count /s"
    fd.append(hdu)

    hdu = fits.ImageHDU(data=dq_array, header=dq_hdr, name="DQ")
    hdu.header["EXTVER"] = imset
    hdu.header["BUNIT"] = "UNITLESS"
    fd.append(hdu)

    fd.close()
