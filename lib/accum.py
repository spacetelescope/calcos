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

    # Get a list of all the headers in the input file.
    headers = cosutil.getHeaders (input)
    phdr = headers[0]
    # Get x_offset now, because overrideKeywords may change it in headers[1].
    x_offset = headers[1].get ("x_offset", 0)

    # Update the switches and reference file names, so the output header
    # will reflect what was actually used.
    cosutil.overrideKeywords (phdr, headers[1], info, switches, reffiles)

    # Check for null science data.
    if info["npix"] == (0,):
        writeNull (input, outflt, outcounts, headers)
        return 1

    # acq/image data are processed differently because they have two imsets.
    if info["exptype"] == "ACQ/IMAGE":
        acqImage (input, outflt, outcounts, outcsum,
                  info, switches, reffiles, cl_args["livetimefile"])
        return 0

    # Open the accum image.
    fd = pyfits.open (input, mode="readonly")
    sci = fd[("SCI",1)].data
    nimsets = len (fd) // 3
    fd.close()

    # The number of rows in the pseudo time-tag table will be equal to
    # the total number of counts in the input image.
    nrows = getNcounts (sci)

    if nrows == 0:
        writeNull (input, outflt, outcounts, headers)
        info["npix"] = (0,)
        return 1

    # Create pseudo-timetag arrays (x & y, no time) from the raw image.
    x = N.zeros (nrows, dtype=N.float32)
    y = N.zeros (nrows, dtype=N.float32)
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
    outdata.field ("EPSILON")[:] = N.ones (nrows, dtype=N.float32)
    outdata.field ("DQ")[:] = N.zeros (nrows, dtype=N.int16)
    outdata.field ("PHA")[:] = 0

    primary_hdu = pyfits.PrimaryHDU (header=phdr)
    ofd = pyfits.HDUList (primary_hdu)
    cosutil.updateFilename (ofd[0].header, outtag)
    ofd.append (hdu)
    ofd.append (cosutil.dummyGTI (info["exptime"]))

    ofd.writeto (outtag)
    del ofd

    status = timetag.timetagBasicCalibration (input, inpha, outtag,
                    outflt, outcounts, None, outcsum,
                    cl_args,
                    info, switches, reffiles,
                    wavecal_info)

    return status

def acqImage (input, outflt, outcounts, outcsum,
              info, switches, reffiles, livetimefile):
    cosutil.printWarning ("acq data are not supported yet")

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

def getNcounts (sci):
    """Return the total number of counts in an array.

    @param sci: image data array, in counts
    @type sci: numpy array

    @return: sum of values in sci, rounded to an int
    @rtype: integer
    """

    ncounts = sci.sum (dtype=N.float64)
    # The value returned by sum() is an "array scalar," so convert it.
    ncounts = float (ncounts)
    ncounts = int (round (ncounts))

    return ncounts

def writeNull (input, outflt, outcounts, headers):
    """Write output files with null data portions.

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
    writeImset (outcounts, headers, None, None, None)
    writeImset (outflt, headers, None, None, None)

def writeImset (output, headers, sci_array, err_array, dq_array):
    """Write an image set (SCI, ERR, DQ extensions).

    This function writes an output file that consists of a primary
    header (with no data), a SCI extension HDU, an ERR HDU (estimates
    of the errors in the SCI data), and a DQ HDU (data quality flags).

    @param output: name of the output file (flt or counts)
    @type output: string
    @param headers: list of headers (primary, sci, err, dq)
    @type headers: list of pyfits Header objects
    @param sci_array: data array for the SCI extension
    @type sci_array: numpy array, or None
    @param err_array: data array for the ERR extension
    @type err_array: numpy array, or None
    @param dq_array: data array for the DQ extension
    @type dq_array: numpy array, or None
    """

    cosutil.printMsg ("writing file %s ..." % output, VERY_VERBOSE)

    primary_hdu = pyfits.PrimaryHDU (header=headers[0])
    fd = pyfits.HDUList (primary_hdu)
    fd[0].header["nextend"] = len (headers) - 1
    cosutil.updateFilename (fd[0].header, output)

    nimsets = len (headers) // 3

    # xxx currently this writes the same data to every imset xxx
    for imset in range (1, nimsets+1):
        sci_hdr = None
        err_hdr = None
        dq_hdr = None
        for i in range (1, len (headers)):
            extname = headers[i].get ("extname", "not found")
            extver  = headers[i].get ("extver", 1)
            if extver == imset:
                if extname.upper() == "SCI":
                    sci_hdr = headers[i]
                elif extname.upper() == "ERR":
                    err_hdr = headers[i]
                elif extname.upper() == "DQ":
                    dq_hdr = headers[i]

        if sci_hdr is None or err_hdr is None or dq_hdr is None:
            raise RuntimeError, "Could not find data for imset %d" % imset

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

    fd.writeto (output, output_verify="silentfix")
