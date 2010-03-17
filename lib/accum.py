from __future__ import division         # confidence high
import copy
import math
import shutil
import time
import types
import numpy as np
from numpy import random
import pyfits

import cosutil
import ccos
import phot
import timetag                  # actually for more generic functions
import wavecal
from calcosparam import *       # parameter definitions

def accumBasicCalibration (input, inpha, outtag,
                  outflt, outcounts, outcsum,
                  cl_args,
                  info, switches, reffiles,
                  wavecal_info):
    """Do the basic processing for accum data.

    The function value will be zero if there was no problem.

    @param input: name of the input file
    @type input: string
    @param inpha: name of the input file containing the pulse height
        histogram (FUV only)
    @type inpha: string
    @param outtag: name of the output file for pseudo time-tag data
    @type outtag: string
    @param outflt: name of the output file for flat-fielded count-rate image
    @type outflt: string
    @param outcounts: name of the output file for count-rate image
    @type outcounts: string
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

    cosutil.printIntro ("ACCUM calibration")
    if info["exptype"] == "ACQ/IMAGE":
        names = [("Input", input),
                 ("OutFlt", outflt), ("OutCounts", outcounts)]
    else:
        names = [("Input", input), ("pseudo tt", outtag),
                 ("OutFlt", outflt), ("OutCounts", outcounts)]
    if info["detector"] == "FUV":
        names.insert (1, ("InPha", inpha))
    if outcsum is not None:
        names.append (("OutCsum", outcsum))
    cosutil.printFilenames (names,
                            shift_file=cl_args["shift_file"],
                            stimfile=cl_args["stimfile"],
                            livetimefile=cl_args["livetimefile"])
    cosutil.printMode (info)

    if info["corrtag_input"]:
        shutil.copy (input, outtag)
        status = timetag.timetagBasicCalibration (input, inpha, outtag,
                    outflt, outcounts, None, outcsum,
                    cl_args, info, switches, reffiles, wavecal_info)
        return status

    # Get a list of all the headers in the input file.
    headers = cosutil.getHeaders (input)
    phdr = headers[0]
    # Get x_offset now, because overrideKeywords may change it in headers[1].
    x_offset = headers[1].get ("x_offset", 0)

    # Update the switches and reference file names, so the output header
    # will reflect what was actually used.
    cosutil.overrideKeywords (phdr, headers[1], info, switches, reffiles)

    # acq/image data are processed differently because they have two imsets.
    if info["exptype"] == "ACQ/IMAGE":
        acqImage (input, outflt, outcounts, outcsum, cl_args,
                  info, switches, reffiles)
        return 0

    # Check for null science data.
    if info["npix"] == (0,):
        writeNull (input, outflt, outcounts, headers)
        return 1

    # Open the accum image.
    fd = pyfits.open (input, mode="readonly")
    sci = fd[("SCI",1)].data
    fd.close()

    # The number of rows in the pseudo time-tag table will be equal to
    # the total number of counts in the input image.
    nrows = getNcounts (sci)

    if nrows == 0:
        writeNull (input, outflt, outcounts, headers)
        info["npix"] = (0,)
        return 1

    # Create pseudo-timetag arrays (x & y, no time) from the raw image.
    x = np.zeros (nrows, dtype=np.float32)
    y = np.zeros (nrows, dtype=np.float32)
    ccos.unbinaccum (sci, x, y, x_offset)

    hdu = cosutil.createCorrtagHDU (nrows, info["detector"], headers[1])
    hdu.header.update ("extname", "EVENTS")
    outdata = hdu.data

    # Copy x and y to the pseudo time-tag table.
    outdata.field ("TIME")[:] = info["exptime"] / 2.
    outdata.field ("RAWX")[:] = x
    outdata.field ("RAWY")[:] = y
    outdata.field ("XCORR")[:] = x
    outdata.field ("YCORR")[:] = y
    outdata.field ("XDOPP")[:] = x
    outdata.field ("XFULL")[:] = x
    outdata.field ("YFULL")[:] = y
    outdata.field ("WAVELENGTH")[:] = np.zeros (nrows, dtype=np.float32)
    outdata.field ("EPSILON")[:] = np.ones (nrows, dtype=np.float32)
    outdata.field ("DQ")[:] = np.zeros (nrows, dtype=np.int16)
    outdata.field ("PHA")[:] = 0

    primary_hdu = pyfits.PrimaryHDU (header=phdr)
    ofd = pyfits.HDUList (primary_hdu)
    cosutil.updateFilename (ofd[0].header, outtag)
    ofd.append (hdu)
    ofd.append (cosutil.dummyGTI (info["exptime"]))
    ofd[0].header.update ("nextend", len (ofd) - 1)     # number of extensions

    ofd.writeto (outtag)
    del ofd

    status = timetag.timetagBasicCalibration (input, inpha, outtag,
                    outflt, outcounts, None, outcsum,
                    cl_args, info, switches, reffiles, wavecal_info)

    return status

def acqImage (input, outflt, outcounts, outcsum, cl_args,
              info, switches, reffiles):
    """Do the calibration for ACQ/IMAGE data."""

    livetimefile = cl_args["livetimefile"]

    fd = pyfits.open (input, mode="readonly")
    nextend = len (fd) - 1
    nimsets = len (fd) // 3
    phdr = fd[0].header
    phdr.update ("cal_ver", info["cal_ver"])
    hdr_list = []

    writePrimaryHDU (outcounts, phdr, nextend)
    writePrimaryHDU (outflt, phdr, nextend)
    if outcsum is not None:
        # we'll add the SCI array for each imset to this array
        csum_array = np.zeros ((NUV_Y, NUV_X), dtype=np.float32)

    for imset in range (1, nimsets+1):

        sci_hdr = fd[("SCI",imset)].header
        err_hdr = fd[("ERR",imset)].header
        dq_hdr = fd[("DQ",imset)].header
        hdr_list.append (sci_hdr)

        counts_sci = fd[("SCI",imset)].data
        flt_sci = counts_sci.copy()

        dq_array = cosutil.getInputDQ (input, imset)

        doPhotcorr (info, switches, reffiles["imphttab"], phdr, sci_hdr)

        updateGlobrate (sci_hdr, counts_sci, info["exptime"])

        doDqicorr (info, switches, reffiles, phdr, dq_array)

        doDeadcorr (flt_sci, sci_hdr["exptime"], info, switches, reffiles,
                    phdr, sci_hdr, input, livetimefile)

        if outcsum is not None:
            csum_array += flt_sci

        doFlatcorr (flt_sci, switches, reffiles, phdr)

        (C_rate, errC_rate, E_rate, errE_rate) = makeImages (
                        counts_sci, flt_sci, sci_hdr["exptime"])
        appendImset (outcounts, imset, C_rate, errC_rate, dq_array,
                     sci_hdr, err_hdr, dq_hdr)
        appendImset (outflt, imset, E_rate, errE_rate, dq_array,
                     sci_hdr, err_hdr, dq_hdr)
    doStatflag (switches, outflt, outcounts)

    if outcsum is not None:
        binx = cl_args["binx"]
        biny = cl_args["biny"]
        compress_csum = cl_args["compress_csum"]
        compression_parameters = cl_args["compression_parameters"]
        writeCsum (outcsum, info["subarray"], phdr, hdr_list, csum_array,
                   binx, biny,
                   compress_csum, compression_parameters)

def updateGlobrate (hdr, data, exptime):
    """Update the GLOBRATE keyword in the extension header.

    @param hdr: the input events extension header
    @type hdr: pyfits Header object
    @param data: data array
    @type data: numpy array
    @param exptime: exposure time in seconds
    @type exptime: float
    """

    if data is None or exptime <= 0.:
        globrate = 0.
    else:
        globrate = data.sum (dtype=np.float64) / exptime

    globrate = round (globrate, 4)
    hdr.update ("globrate", globrate)

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

def doDqicorr (info, switches, reffiles, phdr, dq_array):
    """Update the DQ array using the DQI table.

    @param info: header keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: primary header from input file
    @type phdr: pyfits Header object
    @param dq_array: DQ array from input file, or an array of zeros;
        this will be modified in-place
    @type dq_array: numpy array
    """

    cosutil.printSwitch ("DQICORR", switches)

    if switches["dqicorr"] == "PERFORM":

        cosutil.printRef ("BPIXTAB", reffiles)

        # update using imaging wavecal xxx
        minmax_shift_dict = {(0, 1024): [0., 0., 0., 0.]}
        minmax_doppler = (0., 0.)
        doppler_boundary = -10          # anywhere below 0
        cosutil.updateDQArray (reffiles["bpixtab"], info, dq_array,
                               minmax_shift_dict,
                               minmax_doppler, doppler_boundary)

        phdr["dqicorr"] = "COMPLETE"

def doDeadcorr (flt_sci, exptime, info, switches, reffiles,
                phdr, hdr, input, livetimefile):
    """Correct for deadtime."""

    cosutil.printSwitch ("DEADCORR", switches)

    if switches["deadcorr"] == "PERFORM":
        cosutil.printRef ("DEADTAB", reffiles)
        (dead_rate, dead_method, livetime) = \
                deadtimeCorrection (flt_sci, exptime, reffiles["deadtab"],
                                    info, input, livetimefile)
        hdr.update ("deadrt", dead_rate)
        hdr.update ("deadmt", dead_method)
        hdr.update ("livetm", livetime)
        phdr["deadcorr"] = "COMPLETE"

def deadtimeCorrection (flt_sci, exptime, deadtab, info,
                        input, livetimefile):
    """Determine and apply the livetime factor.

    If there are subarrays, the livetime factor is gotten from the digital
    event counter.  If there are no subarrays, the livetime factor is based
    on the actual count rate.

    @param flt_sci: the SCI image array, to be corrected in-place
    @type flt_sci: numpy array
    @param exptime: exposure time for current imset
    @type exptime: float
    @param deadtab: name of reference table of count rates and livetime factors
    @type deadtab: string
    @param info: header keywords and values
    @type info: dictionary
    @param input: name of input raw file (for writing to livetimefile)
    @type input: string
    @param livetimefile: name of output text file for livetime factors (or None)
    @type livetimefile: string

    @return: the count rate used for determining the livetime factor, a
        string that indicates which method was used for determining the
        livetime factor, and the livetime factor that was used
    @rtype: tuple
    """

    if exptime <= 0.:
        cosutil.printWarning ("Can't do deadcorr, exptime = %.6g." % exptime)
        return (0., "SKIPPED")

    if livetimefile is None:
        fd = None
    else:
        fd = open (livetimefile, "a")
        fd.write ("# %s\n" % (input,))

    ncounts = getNcounts (flt_sci)

    live_info = cosutil.getTable (deadtab,
                                  filter={"segment": info["segment"]},
                                  at_least_one=True)
    obs_rate = live_info.field ("obs_rate")
    live_factor = live_info.field ("livetime")

    # keyword used to print information
    keyword = "MEVENTS"

    # Output count rate from digital event counter (DEC), and corresponding
    # livetime factor.
    dec_countrate = info["countrate"]
    dec_livetime = cosutil.determineLivetime (dec_countrate,
                                              obs_rate, live_factor)
    actual_countrate = float (ncounts) / exptime
    actual_rate_livetime = cosutil.determineLivetime (actual_countrate,
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

    if fd is not None:
        fd.write ("actual countrate and livetime:  %.6g, %6.4f\n" %
                  (actual_countrate, actual_rate_livetime))
        fd.write ("countrate and livetime from %s:  %.6g, %6.4f\n" %
                  (keyword, dec_countrate, dec_livetime))
        fd.write ("livetime %6.4f is based on %s.\n" % \
                  (livetime, livetime_source))

    if fd is not None:
        fd.close()

    return (dead_rate, dead_method, livetime)

def doFlatcorr (flt_sci, switches, reffiles, phdr):
    """Apply flat field correction.

    @param flt_sci: the image array, modified in-place
    @type flt_sci: numpy array
    @param switches: calibration switches
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param phdr: the input primary header
    @type phdr: pyfits Header object
    """

    cosutil.printSwitch ("FLATCORR", switches)

    if switches["flatcorr"] == "PERFORM":

        cosutil.printRef ("FLATFILE", reffiles)

        fd = pyfits.open (reffiles["flatfile"], mode="readonly")
        hdu = fd[1]
        flat = hdu.data
        fd.close()

        (ny, nx) = flat.shape
        x0 = hdu.header.get ("origin_x", 0)
        y0 = hdu.header.get ("origin_y", 0)

        flt_sci[y0:y0+ny,x0:x0+nx] /= flat

        phdr["flatcorr"] = "COMPLETE"

def doStatflag (switches, outflt, outcounts):
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
        cosutil.doImageStat (outflt)

def makeImages (counts_sci, flt_sci, exptime):
    """Create the count rate and error arrays.

    @param counts_sci: the SCI image array, counts
    @type counts_sci: numpy array
    @param flt_sci: the SCI image array, after deadcorr and flatcorr
    @type flt_sci: numpy array
    @param exptime: the exposure time
    @type exptime: float

    @return: count rate array, error estimate for count rate array,
        flat-fielded count rate array, error estimate for flat fielded
        count-rate array
    @rtype: tuple
    """

    if exptime <= 0:
        cosutil.printWarning (
                "Exposure time is zero, so output files are dummy.")
        C_rate = counts_sci * 0.
        E_rate = C_rate.copy()
        errC_rate = C_rate.copy()
        errE_rate = C_rate.copy()
        return (C_rate, errC_rate, E_rate, errE_rate)

    C_rate = counts_sci / exptime
    E_rate = flt_sci / exptime

    counts_sci_temp = np.where (counts_sci < 0., 0., counts_sci)
    errC_rate = np.sqrt (counts_sci_temp) / exptime
    del counts_sci_temp

    # errC_rate will likely have a number of zero values, so we set those
    # to one before dividing.
    errC_rate_temp = np.where (errC_rate == 0., 1., errC_rate)
    errE_rate = E_rate / errC_rate_temp / exptime
    del errC_rate_temp

    return (C_rate, errC_rate, E_rate, errE_rate)

def writeCsum (outcsum, subarray, phdr, hdr_list, csum_array,
               binx=None, biny=None,
               compress_csum=False,
               compression_parameters="gzip,-0.1"):
    """Write the "calcos sum" (csum) image.

    @param outcsum: name of output calcos sum image file
    @type outcsum: string
    @param subarray: True if the exposure used one or more subarrays
    @type subarray: boolean
    @param phdr: primary header
    @type phdr: pyfits Header object
    @param hdr_list: list of sci extension headers
    @type hdr_list: list of pyfits Header objects
    @param csum_array: data array for SCI extension
    @type csum_array: numpy array
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

    cosutil.printMsg ("writing file %s ..." % outcsum, VERY_VERBOSE)

    primary_hdu = pyfits.PrimaryHDU (header=phdr)
    fd = pyfits.HDUList (primary_hdu)
    fd[0].header.update ("nextend", 1)
    fd[0].header.update ("counts", csum_array.sum(dtype=np.float64))
    fd[0].header.update ("filetype", "CALCOS SUM FILE")
    cosutil.updateFilename (fd[0].header, outcsum)
    detector = phdr["detector"]

    shape = csum_array.shape
    if binx is None or binx <= 0:
        binx = NUV_BIN_X
    if biny is None or biny <= 0:
        biny = NUV_BIN_Y
    nx = shape[1] // binx
    ny = shape[0] // biny
    fd[0].header.update ("nuvbinx", binx)
    fd[0].header.update ("nuvbiny", biny)

    # Copy the exposure time keywords to the output headers.
    hdr = hdr_list[0].copy()
    addExptimeKeywords (hdr_list, fd[0].header, hdr)

    # Copy the high-voltage keywords to the output primary header.
    cosutil.copyVoltageKeywords (hdr_list[0], fd[0].header, detector)

    # Copy the subarray keywords to the output primary header.
    cosutil.copySubKeywords (hdr_list[0], fd[0].header, subarray)

    if nx > 1 or ny > 1:
        binned_array = np.zeros ((ny,nx), dtype=np.float32)
        ccos.bin2d (csum_array, binned_array)
    else:
        binned_array = csum_array

    if compress_csum:
        (compType, quantLevel) = compression_parameters.split (",")
        compType = compType.upper() + "_1"
        quantLevel = float (quantLevel)
        fd.append (pyfits.CompImageHDU (binned_array, header=hdr, name="SCI",
                                        compressionType=compType,
                                        quantizeLevel=quantLevel))
    else:
        fd.append (pyfits.ImageHDU (data=binned_array, header=hdr, name="SCI"))
    fd[1].header.update ("BUNIT", "count")

    fd.writeto (outcsum, output_verify="silentfix")

def getNcounts (sci):
    """Return the total number of counts in an array.

    @param sci: image data array, in counts
    @type sci: numpy array

    @return: sum of values in sci, rounded to an int
    @rtype: integer
    """

    ncounts = sci.sum (dtype=np.float64)
    # The value returned by sum() is an "array scalar," so convert it.
    ncounts = float (ncounts)
    ncounts = int (round (ncounts))

    return ncounts

def addExptimeKeywords (hdr_list, phdr, hdr):
    """Copy the exposure time keywords to the output headers.

    exptime and rawtime will be the sums of those values from all (both)
    of the input headers, and expstart and expend will be taken from
    the first and last headers.

    @param hdr_list: list of sci extension headers
    @type hdr_list: list of pyfits Header objects
    @param phdr: output primary header
    @type phdr: pyfits Header object
    @param hdr: output header
    @type hdr: pyfits Header object
    """

    # The headers in hdr_list are assumed to be in chronological order.
    expstart = hdr_list[0].get ("expstart", -999.)
    expend = hdr_list[-1].get ("expend", -999.)

    exptime = 0.
    rawtime = 0.
    for hdr_i in hdr_list:
        exptime_i = hdr_i.get ("exptime", 0.)
        exptime += exptime_i
        rawtime += hdr_i.get ("rawtime", exptime_i)

    phdr.update ("expstart", expstart)
    phdr.update ("expend", expend)
    phdr.update ("exptime", exptime)
    phdr.update ("rawtime", rawtime)

    hdr.update ("expstart", expstart)
    hdr.update ("expend", expend)
    hdr.update ("exptime", exptime)
    hdr.update ("rawtime", rawtime)

def writeNull (input, outflt, outcounts, headers):
    """Write output files with null data blocks.

    @param input: name of the input file
    @type input: string
    @param outflt: name of the output file for flat-fielded count-rate image
    @type outflt: string
    @param outcounts: name of the output file for count-rate image
    @type outcounts: string
    @param headers: list of headers (primary, sci, err, dq)
    @type headers: list of pyfits Header objects
    """

    cosutil.printWarning ("No data in " + input)

    imset = 1
    nextend = 3

    writePrimaryHDU (outcounts, headers[0], nextend)
    appendImset (outcounts, imset, None, None, None,
                 headers[1], headers[2], headers[3])

    writePrimaryHDU (outflt, headers[0], nextend)
    appendImset (outflt, imset, None, None, None,
                 headers[1], headers[2], headers[3])

def writePrimaryHDU (output, phdr, nextend):
    """Write an output file containing just a primary header.

    @param output: name of the output file (flt or counts)
    @type output: string
    @param phdr: primary header of input file
    @type phdr: pyfits Header object
    @param nextend: number of extensions
    @type nextend: int
    """

    cosutil.printMsg ("writing file %s ..." % output, VERY_VERBOSE)

    primary_hdu = pyfits.PrimaryHDU (header=phdr)
    fd = pyfits.HDUList (primary_hdu)
    fd[0].header["nextend"] = nextend
    cosutil.updateFilename (fd[0].header, output)

    fd.writeto (output, output_verify="silentfix")

def appendImset (output, imset, sci_array, err_array, dq_array,
                 sci_hdr, err_hdr, dq_hdr):
    """Append an image set (SCI, ERR, DQ extensions).

    This function appends one image set to an output file.

    @param output: name of the output file (flt or counts)
    @type output: string
    @param imset: image set number (one indexed, to match EXTVER)
    @type imset: int
    @param sci_array: data array for the SCI extension
    @type sci_array: numpy array, or None
    @param err_array: data array for the ERR extension
    @type err_array: numpy array, or None
    @param dq_array: data array for the DQ extension
    @type dq_array: numpy array, or None
    @param sci_hdr: header for SCI extension
    @type sci_hdr: pyfits Header object
    @param err_hdr: header for ERR extension
    @type err_hdr: pyfits Header object
    @param dq_hdr: header for DQ extension
    @type dq_hdr: pyfits Header object
    """

    fd = pyfits.open (output, mode="append")

    hdu = pyfits.ImageHDU (data=sci_array, header=sci_hdr, name="SCI")
    hdu.header.update ("EXTVER", imset)
    hdu.header.update ("BUNIT", "count /s")
    fd.append (hdu)

    hdu = pyfits.ImageHDU (data=err_array, header=err_hdr, name="ERR")
    hdu.header.update ("EXTVER", imset)
    hdu.header.update ("BUNIT", "count /s")
    fd.append (hdu)

    hdu = pyfits.ImageHDU (data=dq_array, header=dq_hdr, name="DQ")
    hdu.header.update ("EXTVER", imset)
    hdu.header.update ("BUNIT", "UNITLESS")
    fd.append (hdu)

    fd.close()
