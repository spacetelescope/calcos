#! /usr/bin/env python

from __future__ import division
import math
import os
import shutil
import sys
import time
import types
import numpy as np
import numpy.linalg as LA
import pyfits
import ccos
from calcosparam import *       # parameter definitions

# initial value
verbosity = VERBOSE

# for appending to a trailer file
fd_trl = None
# if this is False, writing to trailer files will be disabled
write_to_trailer = True

def writeOutputEvents (infile, outfile):
    """
    This function creates a recarray object with the column definitions
    appropriate for a corrected time-tag table, reads an input events table
    into this object, and writes it to the output file.  If the input file
    contains a GTI table, that will be copied unchanged to output.

    If the input is already a corrtag table (if the table in the first
    extension contains the column XFULL), then the file will be copied
    to output without change.

    @param infile: name of the input FITS file containing an EVENTS table
        and optionally a GTI table
    @type infile: string
    @param outfile: name of file for output EVENTS table (and GTI table)
    @type outfile: string
    """

    # ifd = pyfits.open (infile, mode="readonly", memmap=1)
    ifd = pyfits.open (infile, mode="readonly")
    events_extn = ifd["EVENTS"]
    indata = events_extn.data
    if indata is None:
        nrows = 0
    else:
        nrows = indata.shape[0]

    # If the input is already a corrtag file, just copy it.
    if isCorrtag (infile):
        ifd.close()
        shutil.copy (infile, outfile)
        return nrows

    detector = ifd[0].header.get ("detector", "FUV")
    tagflash = (ifd[0].header.get ("tagflash", default="NONE") != "NONE")

    # Create the output events HDU.
    hdu = createCorrtagHDU (nrows, detector, events_extn.header)

    if nrows == 0:
        primary_hdu = pyfits.PrimaryHDU (header=ifd[0].header)
        ofd = pyfits.HDUList (primary_hdu)
        updateFilename (ofd[0].header, outfile)
        ofd.append (hdu)
        if len (ifd) == 3:
            ofd.append (ifd["GTI"])
        ofd.writeto (outfile)
        ifd.close()
        return nrows

    outdata = hdu.data

    # Copy data from the input table to the output HDU object.

    outdata.field ("TIME")[:] = indata.field ("TIME")

    outdata.field ("RAWX")[:] = indata.field ("RAWX")
    outdata.field ("RAWY")[:] = indata.field ("RAWY")
    outdata.field ("XCORR")[:] = indata.field ("RAWX")
    outdata.field ("YCORR")[:] = indata.field ("RAWY")

    outdata.field ("XDOPP")[:] = np.zeros (nrows, dtype=np.float32)
    outdata.field ("XFULL")[:] = np.zeros (nrows, dtype=np.float32)
    outdata.field ("YFULL")[:] = np.zeros (nrows, dtype=np.float32)
    outdata.field ("WAVELENGTH")[:] = np.zeros (nrows, dtype=np.float32)

    outdata.field ("EPSILON")[:] = np.ones (nrows, dtype=np.float32)
    outdata.field ("DQ")[:] = np.zeros (nrows, dtype=np.int16)
    if detector == "FUV":
        outdata.field ("PHA")[:] = indata.field ("PHA")
    else:
        outdata.field ("PHA")[:] = 0

    primary_hdu = pyfits.PrimaryHDU (header=ifd[0].header)
    ofd = pyfits.HDUList (primary_hdu)
    updateFilename (ofd[0].header, outfile)
    ofd.append (hdu)

    # GTI table.
    if len (ifd) == 3:
        ofd.append (ifd["GTI"])

    ofd.writeto (outfile)
    ifd.close()

    return nrows

def isCorrtag (filename):
    """Determine whether 'filename' is a corrtag file.

    A corrtag file contains a table in the first extension, and there
    will be a column with the name "XFULL".

    @param filename: name of a file
    @type filename: string

    @return: True if the first extension of 'filename' is a corrtag table
    @rtype: boolean
    """

    fd = pyfits.open (filename, mode="readonly")
    if len (fd) < 2:                    # no extensions?
        fd.close()
        return False

    # Find an EVENTS table (any one, if there is more than one).
    hdunum = 0
    for i in range (1, len(fd)):
        hdu = fd[i]
        extname = hdu.header.get ("extname", "MISSING")
        if extname.upper() == "EVENTS":
            hdunum = i
            break

    if hdunum < 1:
        fd.close()
        return False

    hdr = fd[hdunum].header
    data = fd[hdunum].data
    got_xfull = False                   # initial value
    if data is None:
        # check each of the TTYPEi keywords
        ncols = hdr.get ("tfields", 0)
        for i in range (1, ncols+1):
            key = "ttype%d" % i
            ttype = hdr.get (key, "missing").lower()
            if ttype == "xfull":
                got_xfull = True
                break
    else:
        got_xfull = findColumn (data, "xfull")
    fd.close()

    return got_xfull

def createCorrtagHDU (nrows, detector, header):
    """Create the output events HDU.

    @param nrows: number of rows to allocate (may be zero)
    @type nrows: int
    @param detector: FUV or NUV
    @type detector: string
    @param header: events extension header
    @type header: pyfits Header object

    @return: header/data unit for a corrtag table
    @rtype: pyfits BinTableHDU object
    """

    col = []
    col.append (pyfits.Column (name="TIME", format="1E", unit="s"))
    col.append (pyfits.Column (name="RAWX", format="1I", unit="pixel"))
    col.append (pyfits.Column (name="RAWY", format="1I", unit="pixel"))
    col.append (pyfits.Column (name="XCORR", format="1E", unit="pixel"))
    col.append (pyfits.Column (name="YCORR", format="1E", unit="pixel"))
    col.append (pyfits.Column (name="XDOPP", format="1E", unit="pixel"))
    col.append (pyfits.Column (name="XFULL", format="1E", unit="pixel"))
    col.append (pyfits.Column (name="YFULL", format="1E", unit="pixel"))
    col.append (pyfits.Column (name="WAVELENGTH", format="1E",
                               unit="angstrom", disp="%9.4f"))
    col.append (pyfits.Column (name="EPSILON", format="1E"))
    col.append (pyfits.Column (name="DQ", format="1I"))
    col.append (pyfits.Column (name="PHA", format="1B"))
    cd = pyfits.ColDefs (col)

    hdu = pyfits.new_table (cd, header=header, nrows=nrows)

    return hdu

def copyExptimeKeywords (inhdr, outhdr):
    """Copy the exposure time keywords from one header to another.

    This is for copying the exposure time keywords from the input extension
    header to the primary header of the csum file.

    @param inhdr: input header
    @type inhdr: pyfits Header object
    @param outhdr: output header
    @type outhdr: pyfits Header object
    """

    outhdr.update ("expstart", inhdr.get ("expstart", -999.))
    outhdr.update ("expend", inhdr.get ("expend", -999.))
    exptime = inhdr.get ("exptime", -999.)
    outhdr.update ("exptime", exptime)
    outhdr.update ("rawtime", inhdr.get ("rawtime", exptime))

def copyVoltageKeywords (inhdr, outhdr, detector):
    """Copy keywords for high voltages from one header to another.

    This is for copying the high voltage keywords from the input extension
    header to the primary header of the csum file.

    @param inhdr: input header
    @type inhdr: pyfits Header object
    @param outhdr: output header
    @type outhdr: pyfits Header object
    @param detector: FUV or NUV
    @type detector: string
    """

    if detector == "FUV":
        outhdr.update ("dethvla", inhdr.get ("dethvla", -999.))
        outhdr.update ("dethvlb", inhdr.get ("dethvlb", -999.))
        outhdr.update ("dethvca", inhdr.get ("dethvca", -999.))
        outhdr.update ("dethvcb", inhdr.get ("dethvcb", -999.))
        outhdr.update ("dethvna", inhdr.get ("dethvna", -999.))
        outhdr.update ("dethvnb", inhdr.get ("dethvnb", -999.))
    elif detector == "NUV":
        outhdr.update ("dethvl", inhdr.get ("dethvl", -999.))
        outhdr.update ("dethvc", inhdr.get ("dethvc", -999.))

def copySubKeywords (inhdr, outhdr, subarray):
    """Copy the subarray keywords from one header to another.

    This is for copying the subarray keywords from the input extension
    header to the primary header of the csum file.

    @param inhdr: input header
    @type inhdr: pyfits Header object
    @param outhdr: output header
    @type outhdr: pyfits Header object
    @param subarray: True if the exposure used one or more subarrays
    @type subarray: boolean
    """

    if subarray:
        outhdr.update ("nsubarry", inhdr.get ("nsubarry", 0))
    else:
        outhdr.update ("nsubarry", 0)
    for i in range (8):
        x_corner_kwd = "corner%1dx" % i
        y_corner_kwd = "corner%1dy" % i
        x_size_kwd = "size%1dx" % i
        y_size_kwd = "size%1dy" % i
        outhdr.update (x_corner_kwd, inhdr.get (x_corner_kwd, -1))
        outhdr.update (y_corner_kwd, inhdr.get (y_corner_kwd, -1))
        outhdr.update (x_size_kwd, inhdr.get (x_size_kwd, -1))
        outhdr.update (y_size_kwd, inhdr.get (y_size_kwd, -1))

def dummyGTI (exptime):
    """Return a GTI table.

    @param exptime: exposure time in seconds
    @type exptime: float

    @return: header/data unit for a GTI table covering the entire exposure
    @rtype: pyfits BinTableHDU object
    """

    col = []
    col.append (pyfits.Column (name="START", format="1D", unit="s"))
    col.append (pyfits.Column (name="STOP", format="1D", unit="s"))
    cd = pyfits.ColDefs (col)
    hdu = pyfits.new_table (cd, nrows=1)
    hdu.header.update ("extname", "GTI")
    outdata = hdu.data
    outdata.field ("START")[:] = 0.
    outdata.field ("STOP")[:] = exptime

    return hdu

def returnGTI (infile):
    """Return a list of (start, stop) good time intervals.

    @param infile: name of the input FITS file containing a GTI table
    @type infile: string
    """

    fd = pyfits.open (infile, mode="readonly")

    # Find the GTI table with the largest value of EXTVER.
    last_extver = 0                     # initial value
    hdunum = 0
    for i in range (1, len(fd)):
        hdu = fd[i]
        extname = hdu.header.get ("extname", "MISSING")
        if extname.upper() == "GTI":
            extver = hdu.header.get ("extver", 1)
            if extver > last_extver:
                last_extver = extver
                hdunum = i

    if hdunum < 1:
        gti = []
    else:
        indata = fd[hdunum].data
        if indata is None:
            gti = []
        else:
            nrows = indata.shape[0]
            start = indata.field ("START")
            stop = indata.field ("STOP")
            gti = [(start[i], stop[i]) for i in range (nrows)]

    return gti

def findColumn (table, colname):
    """Return True if colname is found (case-insensitive) in table.

    @param table: name of table or data block for a FITS table
    @type table: string (if name of table) or FITS record object
    @param colname: name to test for existence in table
    @type colname: string

    @return: True if colname is in the table (without regard to case)
    @rtype: boolean
    """

    if type (table) is str:
        fd = pyfits.open (table, mode="readonly")
        fits_rec = fd[1].data
        fd.close()
    else:
        fits_rec = table

    names = []
    for name in fits_rec.names:
        names.append (name.lower())

    if colname.lower() in names:
        return True
    else:
        return False

def getTable (table, filter, extension=1,
              exactly_one=False, at_least_one=False):
    """Return the data portion of a table.

    All rows that match the filter (a dictionary of column_name = value)
    will be returned.  If the value in the table is STRING_WILDCARD or
    INT_WILDCARD (depending on the data type of the column), that value
    does match the filter for that column.  Also, for a given filter key,
    if the value of the filter is STRING_WILDCARD or NOT_APPLICABLE,
    the test on filter will not be applied for that key (i.e. that filter
    element matches any row).

    It is an error if exactly_one or at_least_one is true but no row
    matches the filter.  A warning will be printed if exactly_one is true
    but more than one row matches the filter.

    @param table: name of the reference table
    @type table: string
    @param filter: dictionary; each key is a column name, and if the value
        in that column matches the filter value for some row, that row will
        be included in the set that is returned
    @type filter: dictionary
    @param extension: identifier for the extension containing the table
    @type extension: tuple, string or integer
    @param exactly_one: true if there must be one and only one matching row
    @type exactly_one: boolean
    @param at_least_one: true if there must be at least one matching row
    @type at_least_one: boolean

    @return: data object containing the selected row(s)
    @rtype: pyfits record array
    """

    # fd = pyfits.open (table, mode="readonly", memmap=1)
    fd = pyfits.open (table, mode="readonly")
    data = fd[extension].data

    # There will be one element of select_arrays for each non-trivial
    # selection criterion.  Each element of select_arrays is an array
    # of flags, true if the row matches the criterion.
    select_arrays = []
    for key in filter.keys():

        if filter[key] == STRING_WILDCARD or \
           filter[key] == NOT_APPLICABLE:
            continue
        column = data.field (key)
        selected = (column == filter[key])

        # Test for for wildcards in the table.
        wild = None
        if isinstance (column, np.chararray):
            wild = (column == STRING_WILDCARD)
        #elif isinstance (column[0], np.integer):
        #    wild = (column == INT_WILDCARD)
        if wild is not None:
            selected = np.logical_or (selected, wild)

        select_arrays.append (selected)

    if len (select_arrays) > 0:
        selected = select_arrays[0]
        for sel_i in select_arrays[1:]:
             selected = np.logical_and (selected, sel_i)
        newdata = data[selected]
    else:
        newdata = data.copy()

    fd.close()

    nselect = len (newdata)
    if nselect < 1:
        newdata = None

    if (exactly_one or at_least_one) and nselect < 1:
        message = "Table has no matching row;\n" + \
                  "table name is " + table + "\n" + \
                  "row selection is " + repr (filter)
        raise RuntimeError, message

    if exactly_one and nselect > 1:
        printWarning ("Table has more than one matching row;")
        printContinuation ("table name is " + table)
        printContinuation ("row selection is " + repr (filter))
        printContinuation ("only the first will be used.")

    return newdata

def getColCopy (filename="", column=None, extension=1, data=None):
    """Return the specified column in native format.

    @param filename: the name of the FITS file
    @type filename: string
    @param column: column name or number
    @type column: string or integer
    @param extension: number of extension containing the table
    @type extension: integer
    @param data: the data portion of a table
    @type data: pyfits record object

    @return: the column data
    @rtype: array

    Specify either the name of the file or the data block, but not both.
    """

    if filename and data is not None:
        raise RuntimeError, "Specify either filename or data, but not both."

    if filename:
        fd = pyfits.open (filename, mode="readonly")
        temp = fd[extension].data.field (column)
        fd.close()
    elif data is not None:
        temp = data.field (column)
    else:
        raise RuntimeError, "Either filename or data must be specified."

    x = np.empty (temp.shape, dtype=temp.dtype.type)
    x[...] = temp

    return x

def getTemplate (raw_template, x_offset, nelem):
    """Return the template spectrum embedded in a possibly larger array.

    @param raw_template: template spectrum as read from the lamptab
    @type raw_template: numpy array
    @param x_offset: offset of raw_template in the extended template
    @type x_offset: int
    @param nelem: length of template spectrum to return
    @type nelem: int
    """

    len_raw = len (raw_template)

    if x_offset == 0 and nelem == len_raw:
        return raw_template.copy()

    template = np.zeros (nelem, dtype=raw_template.dtype)
    template[x_offset:len_raw+x_offset] = raw_template

    return template

def determineLivetime (countrate, obs_rate, live_factor):
    """Compute livetime factor from observed count rate.

    This is just linear interpolation in live_factor vs obs_rate.

    @param countrate: observed count rate
    @type countrate: float
    @param obs_rate: observed count rate column from deadtab
    @type obs_rate: array
    @param live_factor: livetime factor column from deadtab
    @type live_factor: array

    @return: the interpolated livetime factor.
    @rtype: float
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

def isLampOn (xi, eta, dq, info, xtractab, shift2=0.):
    """Test whether a lamp was on.

    This function returns True if a wavecal lamp was on, i.e. if the
    counts through the wavecal aperture were significantly greater than
    the background counts.

    @param xi: pixel coordinates of events, in dispersion direction
    @type xi: numpy array
    @param eta: pixel coordinates of events, in cross-dispersion direction
    @type eta: numpy array
    @param dq: data quality column
    @type dq: numpy array
    @param info: header keywords and values
    @type info: dictionary
    @param xtractab: name of the 1-D extraction parameters table
    @type xtractab: string
    @param shift2: offset of spectrum in cross-dispersion direction
    @type shift2: float

    @return: True if the background-subtracted wavecal source spectrum is
        more than five times the standard deviation of the difference
        between the source counts and the background counts
    @rtype: boolean
    """

    # Use hard-coded numbers for imaging data.  Delete this section if
    # the xtractab actually includes MIRRORA and MIRRORB.
    if info["obstype"] == "IMAGING":
        # note:  xtractab and shift2 are ignored
        len_spectrum = NUV_X
        x_offset = 0
        slope = 0.
        b_spec = 605.
        height = 100
        b_bkg1 = 705.
        b_bkg2 = 505.
        b_hgt = 50
        source = np.zeros ((height, len_spectrum), dtype=np.float64)
        background1 = np.zeros ((b_hgt, len_spectrum), dtype=np.float64)
        background2 = np.zeros ((b_hgt, len_spectrum), dtype=np.float64)
        ccos.xy_extract (xi, eta, source, slope, b_spec, x_offset,
                         dq, info["sdqflags"])
        ccos.xy_extract (xi, eta, background1, slope, b_bkg1, x_offset,
                         dq, info["sdqflags"])
        ccos.xy_extract (xi, eta, background2, slope, b_bkg2, x_offset,
                         dq, info["sdqflags"])
        ns = source.sum (dtype=np.float64)
        nb = background1.sum (dtype=np.float64) + \
             background2.sum (dtype=np.float64)
        sigma_s = math.sqrt (ns)
        sigma_b = math.sqrt (nb)
        printMsg ("Counts from lamp = %.0f, background = %.1f, " \
                  "stddev of difference = %.2f" % \
                  (ns, nb, math.sqrt (sigma_s**2 + sigma_b**2)),
                  level=VERY_VERBOSE)
        sigma_s_b = math.sqrt (sigma_s**2 + sigma_b**2)
        if sigma_s_b > 0.:
            signal_to_noise = (ns - nb) / sigma_s_b
        else:
            signal_to_noise = 0.
        if signal_to_noise > 5.:
            return True
        else:
            return False
    # end of section to delete if xtractab includes MIRRORA and MIRRORB

    x_offset = info["x_offset"]

    if info["detector"] == "FUV":
        len_spectrum = FUV_EXTENDED_X
        segment_list = [info["segment"]]
    else:
        if info["obstype"] == "IMAGING":
            segment_list = ["NUVA"]
        else:
            segment_list = ["NUVB", "NUVA", "NUVC"]     # list NUVB first
        if x_offset <= 0:
            len_spectrum = NUV_X
        else:
            len_spectrum = NUV_EXTENDED_X

    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": "WCA"}

    # Get the background counts.  For NUV the background regions are in
    # nearly the same place for all stripes, so take the background region
    # for just NUVB.
    filter["segment"] = segment_list[0]         # if NUV, use NUVB
    xtract_info = getTable (xtractab, filter)
    if xtract_info is None:
        printWarning ("(isLampOn) matching row not found in xtractab %s" \
                      % xtractab)
        printContinuation ("filter = %s" % str (filter))
        return False

    slope  = xtract_info.field ("slope")[0]
    b_bkg1 = xtract_info.field ("b_bkg1")[0] + shift2
    b_bkg2 = xtract_info.field ("b_bkg2")[0] + shift2
    if findColumn (xtract_info, "b_hgt1"):
        bkg_height1 = xtract_info.field ("b_hgt1")[0]
        bkg_height2 = xtract_info.field ("b_hgt2")[0]
    else:
        bkg_height1 = xtract_info.field ("bheight")[0]
        bkg_height2 = bkg_height1
    background1 = np.zeros ((bkg_height1, len_spectrum), dtype=np.float64)
    background2 = np.zeros ((bkg_height2, len_spectrum), dtype=np.float64)
    ccos.xy_extract (xi, eta, background1, slope, b_bkg1, x_offset,
                     dq, info["sdqflags"])
    ccos.xy_extract (xi, eta, background2, slope, b_bkg2, x_offset,
                     dq, info["sdqflags"])
    # number of background counts
    unscaled_nb = background1.sum (dtype=np.float64) + \
                  background2.sum (dtype=np.float64)
    sum_bkg_height = bkg_height1 + bkg_height2
    del background1, background2

    # Get the source counts.
    ns = 0.                     # number of source counts (incremented in loop)
    sum_height = 0
    for segment in segment_list:
        filter["segment"] = segment
        xtract_info = getTable (xtractab, filter, exactly_one=True)
        slope  = xtract_info.field ("slope")[0]
        b_spec = xtract_info.field ("b_spec")[0] + shift2
        height = xtract_info.field ("height")[0]
        source = np.zeros ((height, len_spectrum), dtype=np.float64)
        ccos.xy_extract (xi, eta, source, slope, b_spec, x_offset,
                         dq, info["sdqflags"])
        ns += source.sum (dtype=np.float64)
        sum_height += height
        del source

    # The heights of the source and background regions differ, so the
    # background counts will be multiplied by this factor.
    normalization = float (sum_height) / float (sum_bkg_height)
    nb = float (unscaled_nb) * normalization
    sigma_s = math.sqrt (ns)
    sigma_b = normalization * math.sqrt (unscaled_nb)

    printMsg ("Counts in wavecal = %.0f, background = %.1f, " \
              "stddev of difference = %.2f" % \
              (ns, nb, math.sqrt (sigma_s**2 + sigma_b**2)),
              level=VERY_VERBOSE)

    sigma_s_b = math.sqrt (sigma_s**2 + sigma_b**2)
    if sigma_s_b > 0.:
        signal_to_noise = (ns - nb) / sigma_s_b
    else:
        signal_to_noise = 0.

    if signal_to_noise > 5.:
        return True
    else:
        return False

def getHeaders (input):
    """Return a list of all the headers in the file.

    argument:
    input       name of an input file
    """

    fd = pyfits.open (input, mode="readonly")

    headers = [hdu.header.copy() for hdu in fd]

    fd.close()

    return headers

def timeAtMidpoint (info):
    """Return the time (MJD) at the midpoint of an exposure.

    argument:
    info          dictionary of header keywords (or could be a Header object)
    """
    return (info["expstart"] + info["expend"]) / 2.

def geometricDistortion (x, y, geofile, segment, igeocorr):
    """Apply geometric (INL) correction.

    x, y          arrays of pixel coordinates of events
    geofile       name of geometric correction reference file
    segment       FUVA or FUVB
    igeocorr      "PERFORM" if interpolation should be used within the geofile
    """

    fd = pyfits.open (geofile, mode="readonly", memmap=0)
    # fd = pyfits.open (geofile, mode="readonly", memmap=1)
    x_hdu = fd[(segment,1)]
    y_hdu = fd[(segment,2)]

    origin_x = x_hdu.header.get ("origin_x", 0)
    origin_y = x_hdu.header.get ("origin_y", 0)

    if origin_x != y_hdu.header.get ("origin_x", 0) or \
       origin_y != y_hdu.header.get ("origin_y", 0):
        raise RuntimeError, "Inconsistent ORIGIN_X or _Y keywords in GEOFILE"

    xbin = x_hdu.header.get ("xbin", 1)
    ybin = x_hdu.header.get ("ybin", 1)
    if xbin != y_hdu.header.get ("xbin", 1) or \
       ybin != y_hdu.header.get ("ybin", 1):
        raise RuntimeError, "Inconsistent XBIN or YBIN keywords in GEOFILE"

    interp_flag = (igeocorr == "PERFORM")
    ccos.geocorrection (x, y, x_hdu.data, y_hdu.data, interp_flag,
                origin_x, origin_y, xbin, ybin)

    fd.close()

def activeArea (segment, brftab):
    """Return the limits of the FUV active area.

    @param segment: for finding the appropriate row in the brftab
    @type segment: string
    @param brftab: name of the baseline reference frame table (ignored for NUV)
    @type brftab: string
    @return: the low and high limits and the left and right limits of the
        active area of the detector.  For NUV this will be (0, 1023, 0, 1023).
    @rtype: tuple
    """

    if segment[0] == "N":
        return (0, NUV_Y-1, 0, NUV_X-1)

    brf_info = getTable (brftab, {"segment": segment}, exactly_one=True)

    a_low = brf_info.field ("a_low")[0]
    a_high = brf_info.field ("a_high")[0]
    a_left = brf_info.field ("a_left")[0]
    a_right = brf_info.field ("a_right")[0]

    return (a_low, a_high, a_left, a_right)

def getInputDQ (input, imset=1):
    """Return the data quality array, or an array of zeros.

    If the data quality extension (EXTNAME = "DQ", EXTVER = imset) actually
    has a non-null data portion, that data array will be returned.  If the
    data portion is null (NAXIS = 0), a constant array will be returned;
    in this case the size will be taken from keywords NPIX1 and NPIX2, and
    the data value will be the value of the PIXVALUE keyword.

    @param input: name of a FITS file containing an image set (SCI, ERR, DQ);
                  only the DQ extension will be read
    @type input: string
    @param imset: image set number (one indexed)
    @type imset: int

    @return: data quality array read from input file, or array of zeros
    @rtype: numpy array
    """

    fd = pyfits.open (input, mode="readonly")

    hdr = fd[("DQ",imset)].header
    detector = fd[0].header["detector"]
    obstype = fd[0].header["obstype"]

    # this section for npix and x_offset is based on getinfo.getGeneralInfo
    if detector == "FUV":
        len_raw = FUV_X
        npix = (FUV_Y, FUV_EXTENDED_X)
        x_offset = FUV_X_OFFSET
    else:
        len_raw = NUV_X
        if obstype == "IMAGING":
            npix = (NUV_Y, NUV_X)
            x_offset = 0
        else:
            npix = (NUV_Y, NUV_EXTENDED_X)
            x_offset = NUV_X_OFFSET

    # Does the data portion exist?
    if hdr["naxis"] > 0:
        if fd[("DQ",imset)].data.shape[1] == npix[1]:
            dq_array = fd[("DQ",imset)].data
            # undo the flagging of regions outside subarrays
            dq_array = np.bitwise_and (dq_array, 16383-(64+128))
        else:
            dq_array = np.zeros (npix, dtype=np.int16)
            dq_array[:,x_offset:len_raw+x_offset] = fd[("DQ",imset)].data
    else:
        dq_array = np.zeros (npix, dtype=np.int16)
        if hdr.has_key ("pixvalue"):
            pixvalue = hdr["pixvalue"]
            if pixvalue != 0:
                dq_array[:,:] = pixvalue

    fd.close()

    return dq_array

def minmaxDoppler (info, doppcorr, doppmag, doppzero, orbitper):
    """Compute the range of Doppler shifts.

    @param info: keywords and values
    @type info: dictionary
    @param doppcorr: if doppcorr = "PERFORM", shift DQ positions to track
        Doppler shift during exposure
    @type doppcorr: string
    @param doppmag: magnitude (pixels) of Doppler shift
    @type doppmag: int or float
    @param doppzero: time (MJD) when Doppler shift is zero and increasing
    @type doppzero: float
    @param orbitper: orbital period (s) of HST
    @type orbitper: float

    @return: minimum and maximum Doppler shifts (will be 0 if doppcorr is omit)
    @rtype: tuple
    """

    if doppcorr == "PERFORM" or doppcorr == "COMPLETE":
        expstart = info["expstart"]
        exptime  = info["exptime"]

        # time is the time in seconds since doppzero.
        nelem = int (round (exptime))           # one element per sec
        nelem = max (nelem, 1)
        time = np.arange (nelem, dtype=np.float64) + \
                   (expstart - doppzero) * SEC_PER_DAY

        # shift is in pixels (wavelengths increase toward larger pixel number).
        shift = -doppmag * np.sin (2. * np.pi * time / orbitper)
        mindopp = shift.min()
        maxdopp = shift.max()
    else:
        mindopp = 0.
        maxdopp = 0.

    return (mindopp, maxdopp)

def updateDQArray (bpixtab, info, dq_array,
                   minmax_shifts, minmax_doppler):
    """Apply the data quality initialization table to DQ array.

    dq_array is a 2-D array, to be written as the DQ extension in an
    ACCUM file (_counts or _flt).  Its contents are assumed to be valid
    on input, since it may have been read from the raw file (if the
    input was an ACCUM image), and it may therefore include flagged
    pixels.  The flag information in the bpixtab will be combined
    (in-place) with dq_array using bitwise OR.

    @param bpixtab: name of the data quality initialization table
    @type bpixtab: string
    @param info: keywords and values
    @type info: dictionary
    @param dq_array: data quality image array (modified in-place)
    @type dq_array: numpy array
    @param minmax_shifts: the min and max offsets in the dispersion direction
        and the min and max offsets in the cross-dispersion direction during
        the exposure
    @type minmax_shifts: tuple
    @param minmax_doppler: minimum and maximum Doppler shifts (will be 0 if
        doppcorr is omit)
    @type minmax_doppler: tuple
    """

    dq_info = getTable (bpixtab, filter={"segment": info["segment"]})
    if dq_info is None:
        return

    (min_shift1, max_shift1, min_shift2, max_shift2) = minmax_shifts
    (mindopp, maxdopp) = minmax_doppler

    # Update the 2-D data quality extension array from the DQI table info.
    lx = dq_info.field ("lx")
    ly = dq_info.field ("ly")
    dx = dq_info.field ("dx")
    dy = dq_info.field ("dy")
    ux = lx + dx - 1
    uy = ly + dy - 1
    lx -= int (round (max_shift1))
    ux -= int (round (min_shift1))
    ly -= int (round (max_shift2))
    uy -= int (round (min_shift2))

    lx += int (round (mindopp))
    ux += int (round (maxdopp))

    ccos.bindq (lx, ly, ux, uy, dq_info.field ("dq"),
                dq_array, info["x_offset"])

def flagOutOfBounds (hdr, dq_array, info, switches,
                     brftab, geofile, minmax_shifts, minmax_doppler):
    """Flag regions that are outside all subarrays (done in-place).

    @param hdr: the extension header
    @type hdr: pyfits Header object
    @param dq_array: data quality image array (modified in-place)
    @type dq_array: numpy array
    @param info: keywords and values
    @type info: dictionary
    @param switches: calibration switches
    @type switches: dictionary
    @param brftab: name of baseline reference table (for active area)
    @type brftab: string
    @param minmax_shifts: the min and max offsets in the dispersion direction
        and the min and max offsets in the cross-dispersion direction during
        the exposure
    @type minmax_shifts: tuple
    @param minmax_doppler: minimum and maximum Doppler shifts (will be 0 if
        doppcorr is omit)
    @type minmax_doppler: tuple
    """

    nsubarrays = info["nsubarry"]
    x_offset = info["x_offset"]
    detector = info["detector"]
    segment = info["segment"]

    if detector == "FUV":
        # Indices 0, 1, 2, 3 are for FUVA, while 4, 5, 6, 7 are for FUVB.
        indices = np.arange (4, dtype=np.int32)
        if segment == "FUVB":
            indices += 4
    else:
        indices = np.arange (nsubarrays, dtype=np.int32)

    temp = dq_array.copy()
    (ny, nx) = dq_array.shape

    # These are for shifting and smearing the out-of-bounds region into
    # the subarray due to the wavecal offset and Doppler shift and their
    # variation during the exposure.
    (min_shift1, max_shift1, min_shift2, max_shift2) = minmax_shifts
    (mindopp, maxdopp) = minmax_doppler

    dx = min_shift1
    dy = min_shift2
    dx -= maxdopp
    dx = int (round (dx))
    dy = int (round (dy))
    xwidth = int (round (max_shift1 - min_shift1 + maxdopp - mindopp))
    ywidth = int (round (max_shift2 - min_shift2))

    # get a list of subarray locations
    subarrays = []
    for i in indices:
        sub = {}
        sub_number = str (i)
        # these keywords are 0-indexed
        x0 = hdr["corner"+sub_number+"x"]
        y0 = hdr["corner"+sub_number+"y"]
        xsize = hdr["size"+sub_number+"x"]
        ysize = hdr["size"+sub_number+"y"]
        if xsize <= 0 or ysize <= 0:
            continue
        if detector == "FUV" and (ysize, xsize) == (FUV_Y, FUV_X):
            continue
        if detector == "NUV" and (ysize, xsize) == (NUV_Y, NUV_X):
            continue
        x1 = x0 + xsize - xwidth
        y1 = y0 + ysize - ywidth
        sub["x0"] = x0
        sub["y0"] = y0
        sub["x1"] = x1
        sub["y1"] = y1
        subarrays.append (sub)
    if not subarrays:
        # Create one full-size "subarray" in order to account for the NUV
        # image being larger than the detector and because of fpoffset.
        sub = {}
        x0 = 0
        y0 = 0
        if detector == "FUV":
            xsize = FUV_X
            ysize = FUV_Y
        else:
            xsize = NUV_X
            ysize = NUV_Y
        x1 = x0 + xsize - xwidth
        y1 = y0 + ysize - ywidth
        sub["x0"] = x0
        sub["y0"] = x0
        sub["x1"] = x1
        sub["y1"] = y1
        subarrays.append (sub)

    # Initially flag the entire image as out of bounds, then remove the
    # flag (set it to zero) for each subarray.
    temp[:,:] = DQ_OUT_OF_BOUNDS
    (ny, nx) = dq_array.shape

    # The test on COMPLETE is for corrtag input.
    if switches["tempcorr"] == "PERFORM" or switches["tempcorr"] == "COMPLETE":

        # Get the parameters found by computeThermalParam.
        seg = segment[-1]           # "A" or "B"
        # reference positions
        sx1r = hdr.get ("STIM"+seg+"0LX", -1.)
        sy1r = hdr.get ("STIM"+seg+"0LY", -1.)
        sx2r = hdr.get ("STIM"+seg+"0RX", -1.)
        sy2r = hdr.get ("STIM"+seg+"0RY", -1.)
        # measured positions of the stims
        sx1 = hdr.get ("STIM"+seg+"_LX", sx1r)
        sy1 = hdr.get ("STIM"+seg+"_LY", sy1r)
        sx2 = hdr.get ("STIM"+seg+"_RX", sx2r)
        sy2 = hdr.get ("STIM"+seg+"_RY", sy2r)
        if sx1 < 0:
            sx1 = sx1r
        if sy1 < 0:
            sy1 = sy1r
        if sx2 < 0:
            sx2 = sx2r
        if sy2 < 0:
            sy2 = sy2r
        if sx1 < 0. or sy1 < 0. or sx2 < 0. or sy2 < 0.:
            xslope = 1.
            xintercept = 0.
            yslope = 1.
            yintercept = 0.
        else:
            xslope = (sx2r - sx1r) / (sx2 - sx1)
            xintercept = sx1r - sx1 * xslope
            yslope = (sy2r - sy1r) / (sy2 - sy1)
            yintercept = sy1r - sy1 * yslope

        # subarrays is a list of dictionaries, each with keys:
        #     "x0", "x1", "y0", "y1"
        # x is the more rapidly varying axis (dispersion direction), and
        # y is the less rapidly varying axis.  The limits can be used as a
        # slice, i.e. x1 and y1 are one larger than the actual upper limits.
        new_subarrays = []
        for sub in subarrays:
            x0 = sub["x0"]
            x1 = sub["x1"]
            y0 = sub["y0"]
            y1 = sub["y1"]
            # apply the correction for thermal distortion
            sub["x0"] = xintercept + x0 * xslope
            sub["y0"] = yintercept + y0 * yslope
            sub["x1"] = xintercept + (x1 - 1.) * xslope + 1.
            sub["y1"] = yintercept + (y1 - 1.) * yslope + 1.
            new_subarrays.append (sub)
        del subarrays
        subarrays = new_subarrays

    # Add shifts, apply geometric correction to the subarray for the
    # source spectrum, and set flags to zero in temp within subarrays.
    (b_low, b_high, b_left, b_right) = activeArea (segment, brftab)
    nfound = 0
    save_sub = None
    for sub in subarrays:
        x0 = sub["x0"]
        x1 = sub["x1"]
        y0 = sub["y0"]
        y1 = sub["y1"]
        # the subarrays for the stims are outside the active area
        if y1 < b_low or y0 > b_high:
            clearSubarray (temp, x0, x1, y0, y1, dx, dy, x_offset)
            continue
        nfound += 1
        # These are arrays of pixel coordinates just inside the borders
        # of the subarray.
        x_lower = np.arange (x0, x1, dtype=np.float32)
        x_upper = np.arange (x0, x1, dtype=np.float32)
        y_left  = np.arange (y0, y1, dtype=np.float32)
        y_right = np.arange (y0, y1, dtype=np.float32)
        y_lower = y0 + 0. * x_lower
        y_upper = (y1 - 1.) + 0. * x_upper
        x_left  = x0 + 0. * y_left
        x_right = (x1 - 1.) + 0. * y_right
        # These are independent variable arrays for interpolation.
        x_lower_uniform = np.arange (nx, dtype=np.float32)
        x_upper_uniform = np.arange (nx, dtype=np.float32)
        y_left_uniform  = np.arange (ny, dtype=np.float32)
        y_right_uniform = np.arange (ny, dtype=np.float32)
        # These will be the arrays of interpolated edge coordinates.
        y_lower_interp = np.arange (nx, dtype=np.float32)
        y_upper_interp = np.arange (nx, dtype=np.float32)
        x_left_interp  = np.arange (ny, dtype=np.float32)
        x_right_interp = np.arange (ny, dtype=np.float32)
        save_sub = (x0, x1, y0, y1)             # in case geocorr is omit
    if nfound == 0:
        printWarning (
        "in flagOutOfBounds, there should be at least one full-size 'subarray'")
    if nfound > 1:
        printWarning ("in flagOutOfBounds, more subarrays than expected")
    # The test on COMPLETE is for corrtag input.
    if switches["geocorr"] == "PERFORM" or switches["geocorr"] == "COMPLETE":
        interp_flag = (switches["igeocorr"] == "PERFORM")
        (x_data, origin_x, xbin, y_data, origin_y, ybin) = \
                        getGeoData (geofile, segment)
        # Undistort x_lower, y_lower, etc., in-place.
        ccos.geocorrection (x_lower, y_lower, x_data, y_data, interp_flag,
                            origin_x, origin_y, xbin, ybin)
        ccos.geocorrection (x_upper, y_upper, x_data, y_data, interp_flag,
                            origin_x, origin_y, xbin, ybin)
        ccos.geocorrection (x_left, y_left, x_data, y_data, interp_flag,
                            origin_x, origin_y, xbin, ybin)
        ccos.geocorrection (x_right, y_right, x_data, y_data, interp_flag,
                            origin_x, origin_y, xbin, ybin)
        del (x_data, y_data)
        # Interpolate to uniform spacing (pixel spacing).
        ccos.interp1d (x_lower, y_lower, x_lower_uniform, y_lower_interp)
        ccos.interp1d (x_upper, y_upper, x_upper_uniform, y_upper_interp)
        ccos.interp1d (y_left,  x_left,  y_left_uniform,  x_left_interp)
        ccos.interp1d (y_right, x_right, y_right_uniform, x_right_interp)
        # Apply offsets for zero point and wavecal shifts, replacing the
        # previous x_lower, y_lower, etc.  The independent variable arrays
        # will now be uniform, and the dependent variable arrays will have
        # been interpolated onto the uniform grid.
        (y_lower, y_upper) = applyOffsets (y_lower_interp, y_upper_interp,
                                           ny, dy)
        (x_left, x_right)  = applyOffsets (x_left_interp, x_right_interp,
                                           nx, dx, x_offset)

        ccos.clear_rows (temp, y_lower, y_upper, x_left, x_right)
    elif save_sub is not None:
        (x0, x1, y0, y1) = save_sub
        clearSubarray (temp, x0, x1, y0, y1, dx, dy, x_offset)

    dq_array[:,:] = np.bitwise_or (dq_array, temp)

def applyOffsets (x_left, x_right, nx, dx, x_offset=0):

    x_left += x_offset
    x_right += x_offset
    x_left -= dx
    x_right -= dx
    x_left = np.where (x_left < 0., 0., x_left)
    x_right = np.where (x_right > nx-1., nx-1., x_right)

    return (x_left, x_right)

def clearSubarray (temp, x0, x1, y0, y1, dx, dy, x_offset):
    """Set the subarray to zero in temp."""

    (ny, nx) = temp.shape
    x0 += x_offset
    x0 -= dx
    y0 -= dy
    x1 += x_offset
    x1 -= dx
    y1 -= dy
    x0 = max (x0, 0)
    y0 = max (y0, 0)
    x1 = min (x1, nx)
    y1 = min (y1, ny)
    temp[y0:y1,x0:x1] = DQ_OK

def flagOutsideActiveArea (dq_array, segment, brftab, x_offset,
                           minmax_shifts, minmax_doppler):
    """Flag the region that is outside the active area.

    This is only relevant for FUV data.

    @param dq_array: 2-D data quality array, modified in-place
    @type dq_array: numpy array
    @param segment: segment name (FUVA or FUVB)
    @type segment: string
    @param brftab: name of baseline reference table
    @type brftab: string
    @param x_offset: offset of raw_template in the extended template
    @type x_offset: int
    @param minmax_shifts: the min and max offsets in the dispersion direction
        and the min and max offsets in the cross-dispersion direction during
        the exposure
    @type minmax_shifts: tuple
    """

    (b_low, b_high, b_left, b_right) = activeArea (segment, brftab)

    # These are for shifting and smearing the out-of-bounds region into
    # the active region due to the wavecal offset and Doppler shift and
    # their variation during the exposure.
    (min_shift1, max_shift1, min_shift2, max_shift2) = minmax_shifts
    (mindopp, maxdopp) = minmax_doppler

    b_left -= int (round (min_shift1))
    b_right -= int (round (max_shift1))
    b_low -= int (round (min_shift2))
    b_high -= int (round (max_shift2))

    b_left += int (round (maxdopp))
    b_right += int (round (mindopp))

    b_left += x_offset
    b_right += x_offset

    (ny, nx) = dq_array.shape

    if b_low >= 0:
        dq_array[0:b_low,:]    |= DQ_OUT_OF_BOUNDS
    if b_high < ny-1:
        dq_array[b_high+1:,:]  |= DQ_OUT_OF_BOUNDS
    if b_left >= 0:
        dq_array[:,0:b_left]   |= DQ_OUT_OF_BOUNDS
    if b_right < nx-1:
        dq_array[:,b_right+1:] |= DQ_OUT_OF_BOUNDS

def getGeoData (geofile, segment):
    """Open and read the geofile.

    @param geofile: name of geometric correction reference file
    @type geofile: string
    @param segment: FUVA or FUVB
    @type segment: string

    @return: the data from the geofile for X and Y, and the offsets;
        x_hdu.data:  array to correct distortion in X
        origin_x:  offset of x_hdu.data within detector coordinates
        xbin:  binning (int) in the X direction
        y_hdu.data:  array to correct distortion in Y
        origin_y:  offset of y_hdu.data within detector coordinates
        ybin:  binning (int) in the Y direction
    @rtype: tuple
    """

    fd = pyfits.open (geofile, mode="readonly", memmap=0)
    x_hdu = fd[(segment,1)]
    y_hdu = fd[(segment,2)]

    # The images in the geofile will typically be smaller than the full
    # detector.  These offsets give the location of geofile pixel [0,0]
    # on the detector.
    origin_x = x_hdu.header.get ("origin_x", 0)
    origin_y = x_hdu.header.get ("origin_y", 0)

    if origin_x != y_hdu.header.get ("origin_x", 0) or \
       origin_y != y_hdu.header.get ("origin_y", 0):
        raise RuntimeError, "Inconsistent ORIGIN_X or _Y keywords in GEOFILE"

    xbin = x_hdu.header.get ("xbin", 1)
    ybin = x_hdu.header.get ("ybin", 1)
    if xbin != y_hdu.header.get ("xbin", 1) or \
       ybin != y_hdu.header.get ("ybin", 1):
        raise RuntimeError, "Inconsistent XBIN or YBIN keywords in GEOFILE"

    # "touch" the data before closing the file.  Is this necessary?
    x_data = x_hdu.data
    y_data = y_hdu.data

    fd.close()

    return (x_data, origin_x, xbin, y_data, origin_y, ybin)

def tableHeaderToImage (thdr):
    """Rename table WCS keywords to image WCS keywords.

    The function returns a copy of the header with table-specific WCS
    keywords renamed to their image-style counterparts, to serve as an
    image header.

    argument:
    thdr          a FITS Header object for a table
    """

    hdr = thdr.copy()

    # These are the world coordinate system keywords in an events table
    # and their corresponding names for an image.  NOTE that this assumes
    # that the XCORR and YCORR columns are 2 and 3 (one indexed).
    tkey = ["TCTYP2", "TCRVL2", "TCRPX2", "TCDLT2", "TCUNI2", "TC2_2", "TC2_3",
            "TCTYP3", "TCRVL3", "TCRPX3", "TCDLT3", "TCUNI3", "TC3_2", "TC3_3"]
    ikey = ["CTYPE1", "CRVAL1", "CRPIX1", "CDELT1", "CUNIT1", "CD1_1", "CD1_2",
            "CTYPE2", "CRVAL2", "CRPIX2", "CDELT2", "CUNIT2", "CD2_1", "CD2_2"]
    # Rename events table WCS keywords to the corresponding image WCS keywords.
    for i in range (len (tkey)):
        if hdr.has_key (tkey[i]):
            if hdr.has_key (ikey[i]):
                printWarning ("Can't rename %s to %s" % (tkey[i], ikey[i]))
                printContinuation ("keyword already exists")
                del (hdr[tkey[i]])
            else:
                hdr.rename_key (tkey[i], ikey[i])

    return hdr

def imageHeaderToTable (imhdr):
    """Modify keywords to turn an image header into a table header.

    The function returns a copy of the header with image-specific world
    coordinate system keywords and BUNIT deleted.

    arguments:
    imhdr         a FITS Header object for an image
    """

    hdr = imhdr.copy()

    ikey = ["CTYPE1", "CRVAL1", "CRPIX1", "CDELT1", "CUNIT1", "CD1_1", "CD1_2",
            "CTYPE2", "CRVAL2", "CRPIX2", "CDELT2", "CUNIT2", "CD2_1", "CD2_2",
            "BUNIT"]
    for keyword in ikey:
        if hdr.has_key (keyword):
            del hdr[keyword]

    return hdr

def delCorrtagWCS (thdr):
    """Delete table WCS keywords.

    The function returns a copy of the header with table-specific WCS keywords
    deleted.  This is appropriate when creating an x1d table from a corrtag
    table.

    argument:
    thdr          a FITS Header object for a table
    """

    hdr = thdr.copy()

    # These are the world coordinate system keywords in an events table.
    # NOTE that this assumes that the XCORR and YCORR columns are 2 and 3
    # (one indexed).
    tkey = ["TCTYP2", "TCRVL2", "TCRPX2", "TCDLT2", "TCUNI2", "TC2_2", "TC2_3",
            "TCTYP3", "TCRVL3", "TCRPX3", "TCDLT3", "TCUNI3", "TC3_2", "TC3_3"]
    for keyword in tkey:
        if hdr.has_key (keyword):
            del hdr[keyword]

    return hdr

def updateFilename (phdr, filename):
    """Update the FILENAME keyword in a primary header.

    This routine will update (or add) the FILENAME keyword.  If filename
    includes a directory, that will not be included in the keyword value.

    arguments:
    phdr        primary header
    filename    may include directory
    """

    phdr.update ("filename", os.path.basename (filename))

def renameFile (infile, outfile):
    """Rename a FITS file, and update the FILENAME keyword.

    @param infile: current name of a FITS file
    @type infile: string
    @param outfile: new name for the file
    @type outfile: string
    """

    printMsg ("rename " + infile + " --> " + outfile, VERY_VERBOSE)

    os.rename (infile, outfile)

    fd = pyfits.open (outfile, mode="update")

    # If the output file name is a product name (ends with '0' before
    # the suffix), change the value of the extension keyword ASN_MTYP.
    if isProduct (outfile):
        asn_mtyp = fd[1].header.get ("asn_mtyp", "missing")
        asn_mtyp = modifyAsnMtyp (asn_mtyp)
        if asn_mtyp != "missing":
            fd[1].header["asn_mtyp"] = asn_mtyp
    updateFilename (fd[0].header, outfile)

    fd.close()

def copyFile (infile, outfile):
    """Copy a FITS file, and update the FILENAME keyword.

    @param infile: name of input FITS file
    @type infile: string
    @param outfile: name of output FITS file
    @type outfile: string
    """

    printMsg ("copy " + infile + " --> " + outfile, VERY_VERBOSE)

    shutil.copy (infile, outfile)

    fd = pyfits.open (outfile, mode="update")

    # If the output file name is a product name (ends with '0' before
    # the suffix), change the value of the extension keyword ASN_MTYP.
    if isProduct (outfile):
        asn_mtyp = fd[1].header.get ("asn_mtyp", "missing")
        asn_mtyp = modifyAsnMtyp (asn_mtyp)
        if asn_mtyp != "missing":
            fd[1].header["asn_mtyp"] = asn_mtyp
    updateFilename (fd[0].header, outfile)

    fd.close()

def isProduct (filename):
    """Return True if 'filename' is a "product" name.

    @param filename: name of an output file
    @type filename: string
    @return: True if the root part (before the suffix) of 'filename'
        ends in '0', implying that it is a product name
    @rtype: boolean
    """

    is_product = False          # may be changed below
    i = filename.rfind ("_")
    if i > 0 and filename[i:] == "_a.fits" or filename[i:] == "_b.fits":
        i = filename[0:i-1].rfind ("_")
    if i > 0 and filename[i-1] == '0':
        is_product = True

    return is_product

def modifyAsnMtyp (asn_mtyp):
    """Replace 'EXP' with 'PROD' in the ASN_MTYP keyword string.

    @param asn_mtyp: value of ASN_MTYP keyword from an input file
    @type asn_mtyp: string
    @return: modified asn_mtyp
    @rtype: string
    """

    if asn_mtyp.startswith ("EXP-") or asn_mtyp.startswith ("EXP_"):
        asn_mtyp = "PROD" + asn_mtyp[3:]

    return asn_mtyp

def doImageStat (input):
    """Compute statistics for an image, and update keywords in header.

    argument:
    input       name of FITS file; keywords in the file will be modified
                in-place
    """

    fd = pyfits.open (input, mode="update")

    if fd[1].data is None:
        fd.close()
        return
    phdr = fd[0].header
    xtractab = expandFileName (phdr.get ("xtractab", ""))
    detector = phdr.get ("detector", "")
    if detector == "FUV":
        fuv_segment = phdr.get ("segment", "")  # not used for NUV
    opt_elem = phdr.get ("opt_elem", "")
    cenwave = phdr.get ("cenwave", 0)
    aperture = getApertureKeyword (phdr, truncate=1)
    exptype = phdr.get ("exptype", "")
    nextend = len (fd) - 1      # number of extensions
    nimsets = nextend // 3      # number of image sets

    for k in range (nimsets):
        extver = k + 1          # extver is one indexed

        hdr = fd[("SCI",extver)].header
        sci = fd[("SCI",extver)].data
        err = fd[("ERR",extver)].data
        dq = fd[("DQ",extver)].data

        dispaxis = hdr.get ("dispaxis", 0)
        exptime = hdr.get ("exptime", 0.)
        sdqflags = hdr.get ("sdqflags", 3832)
        x_offset = hdr.get ("x_offset", 0)

        if exptype == "ACQ/IMAGE":
            dispaxis = 0

        if dispaxis > 0:
            axis = 2 - dispaxis         # 1 --> 1,  2 --> 0
            axis_length = fd[1].data.shape[axis]

        # This will be a list of dictionaries, one for FUV, three for NUV.
        stat_info = []

        if detector == "FUV":
            segment_list = [fuv_segment]            # just one
        elif dispaxis == 0:
            segment_list = ["NUV"]                  # target-acq image
        else:
            segment_list = ["NUVA", "NUVB", "NUVC"]

        for segment in segment_list:

            if dispaxis > 0:
                filter = {"segment": segment,
                          "opt_elem": opt_elem,
                          "cenwave": cenwave,
                          "aperture": aperture}

                xtract_info = getTable (xtractab, filter)
                if xtract_info is None:
                    continue

                slope = xtract_info.field ("slope")[0]
                b_spec = xtract_info.field ("b_spec")[0]
                extr_height = xtract_info.field ("height")[0]

                sci_band = np.zeros ((extr_height, axis_length),
                                     dtype=np.float32)
                ccos.extractband (sci, axis, slope, b_spec, x_offset,
                                  sci_band)

                if err is None:
                    err_band = None
                else:
                    err_band = np.zeros ((extr_height, axis_length),
                                         dtype=np.float32)
                    ccos.extractband (err, axis, slope, b_spec, x_offset,
                                      err_band)

                if dq is None:
                    dq_band = None
                else:
                    dq_band = np.zeros ((extr_height, axis_length),
                                        dtype=np.int16)
                    ccos.extractband (dq, axis, slope, b_spec, x_offset,
                                      dq_band)

                stat_info.append (computeStat (sci_band,
                              err_band, dq_band, sdqflags))

            else:
                # This is presumably a target-acquisition image.  Compute info
                # for the entire image.
                stat_info.append (computeStat (sci, err, dq, sdqflags))

        # Combine the three NUV stripes, or for FUV return the first element.
        stat_avg = combineStat (stat_info)

        sci_hdr = fd[("SCI",extver)].header
        sci_hdr.update ("ngoodpix", stat_avg["ngoodpix"])
        sci_hdr.update ("goodmean", exptime * stat_avg["sci_goodmean"])
        sci_hdr.update ("goodmax", exptime * stat_avg["sci_goodmax"])
        if err is not None:
            err_hdr = fd[("ERR",extver)].header
            err_hdr.update ("ngoodpix", stat_avg["ngoodpix"])
            err_hdr.update ("goodmean", exptime * stat_avg["err_goodmean"])
            err_hdr.update ("goodmax", exptime * stat_avg["err_goodmax"])

    fd.close()

def doSpecStat (input):
    """Compute statistics for a table, and update keywords in header.

    The NET column will be read, and statistics computed for all rows.

    argument:
    input       name of FITS file; keywords in the file will be modified
                in-place
    """

    fd = pyfits.open (input, mode="update")
    try:
        sci_extn = fd["SCI"]
    except KeyError:
        doTagFlashStat (fd)                     # extname is "LAMPFLASH"
        fd.close()
        return

    if sci_extn.data is None:
        fd.close()
        return
    sdqflags = sci_extn.header["sdqflags"]
    outdata = sci_extn.data
    nrows = outdata.shape[0]
    if nrows < 1:
        fd.close()
        return
    exptime_col = outdata.field ("EXPTIME")
    net = outdata.field ("NET")
    error = outdata.field ("ERROR")
    dq = outdata.field ("DQ")

    # This will be a list of dictionaries, one for each segment or stripe.
    # (statistics for the error array are computed but then ignored)
    stat_info = []
    sum_exptime = 0.
    for row in range (nrows):
        sum_exptime += exptime_col[row]
        onestat = computeStat (net[row], error[row], dq[row], sdqflags)
        stat_info.append (onestat)
    exptime = sum_exptime / nrows

    # Combine the segments or stripes.
    stat_avg = combineStat (stat_info)

    sci_extn.header.update ("ngoodpix", stat_avg["ngoodpix"])
    sci_extn.header.update ("goodmean", exptime * stat_avg["sci_goodmean"])
    sci_extn.header.update ("goodmax", exptime * stat_avg["sci_goodmax"])

    fd.close()

def doTagFlashStat (fd):
    """Compute statistics for an (already open) tagflash output file.

    The GROSS column will be read, and statistics computed for all rows.

    argument:
    fd          HDU list for the FITS file (opened by doSpecStat)
    """

    sci_extn = fd["LAMPFLASH"]
    if sci_extn.data is None:
        return

    outdata = sci_extn.data
    nrows = outdata.shape[0]
    if nrows < 1:
        return
    nelem = outdata.field ("NELEM")
    gross = outdata.field ("GROSS")

    sum_gross = 0.
    max_gross = 0.
    n = 0
    for row in range (nrows):
        max_gross = max (max_gross, np.maximum.reduce (gross[row]))
        sum_gross += np.sum (gross[row])
        n += nelem[row]

    sci_extn.header.update ("ngoodpix", n)
    sci_extn.header.update ("goodmean", sum_gross / float (n))
    sci_extn.header.update ("goodmax", max_gross)

def computeStat (sci_band, err_band=None, dq_band=None, sdqflags=3832):
    """Compute statistics.

    The function value is a dictionary with the info.  The keys are the
    keyword names, except that ones that have the same keyword but different
    values in the SCI and ERR extensions (goodmean, goodmax) have
    sci_ or err_ prefixes.

    arguments:
    sci_band       science data array for which statistics are needed
    err_band       error array (but may be None) associated with sci_band
    dq_band        data quality array (may be None) associated with sci_band
    sdqflags       "serious" data quality flags
    """

    # default values:
    stat_info = {"ngoodpix": 0, "sci_goodmax": 0., "sci_goodmean": 0.,
                                "err_goodmax": 0., "err_goodmean": 0.}

    # Don't quit if there are numpy exceptions.
    # xxx np.Error.setMode (all="warn", underflow="ignore")

    # Compute statistics for the sci array.  Note that mask is used
    # for both the sci and err arrays (if there is a dq_band).
    if dq_band is None:
        sci_good = np.ravel (sci_band)
    else:
        serious_dq = dq_band & sdqflags
        # mask = 1 where dq == 0
        mask = np.where (serious_dq == 0)
        sci_good = sci_band[mask]

    ngoodpix = len (sci_good)
    stat_info["ngoodpix"] = ngoodpix
    if ngoodpix > 0:
        stat_info["sci_goodmax"] = np.maximum.reduce (sci_good)
        stat_info["sci_goodmean"] = np.sum (sci_good) / ngoodpix
    del sci_good

    # Compute statistics for the err array.
    if err_band is not None:
        if dq_band is None:
            err_good = np.ravel (err_band)
        else:
            err_good = err_band[mask]
        if ngoodpix > 0:
            stat_info["err_goodmax"] = np.maximum.reduce (err_good)
            stat_info["err_goodmean"] = \
                      np.sum (err_good) / ngoodpix

    return stat_info

def combineStat (stat_info):
    """Combine statistical info for the segments or stripes.

    The input is a list of dictionaries.  The output is one dictionary
    with the same keys and with values that are the averages of the input.

    argument:
    stat_info      list of dictionaries, one for each segment (or stripe)
    """

    if len (stat_info) == 1:
        return stat_info[0]

    # Initialize these variables.
    sum_n = 0
    sci_max = 0.
    sci_sum = 0.
    err_max = 0.
    err_sum = 0.

    for stat in stat_info:
        n = stat["ngoodpix"]
        if n > 0:
            sum_n += n
            sci_max = max (sci_max, stat["sci_goodmax"])
            sci_sum += (n * stat["sci_goodmean"])
            if stat.has_key ("err_goodmax"):
                err_max = max (err_max, stat["err_goodmax"])
                err_sum += (n * stat["err_goodmean"])

    if sum_n > 0:
        sci_sum /= float (sum_n)
        err_sum /= float (sum_n)

    return {"ngoodpix": sum_n,
            "sci_goodmax": sci_max, "sci_goodmean": sci_sum,
            "err_goodmax": err_max, "err_goodmean": err_sum}

def overrideKeywords (phdr, hdr, info, switches, reffiles):
    """Override the calibration switch and reference file keywords.

    The calibration switch and reference file keywords and a few other
    specific keywords will be overridden.  The x_offset keyword will be set.

    arguments:
    phdr          primary header from input
    hdr           extension header from input
    info          dictionary of keywords and values
    switches      dictionary of calibration switches
    reffiles      dictionary of reference file names
    """

    for key in switches.keys():
        if phdr.has_key (key):
            if key == "statflag":
                if switches["statflag"] == "PERFORM":
                    phdr["statflag"] = True
                else:
                    phdr["statflag"] = False
            else:
                phdr[key] = switches[key]

    for key in reffiles.keys():
        # Skip the _hdr keys (they're redundant), and skip any keyword
        # that isn't already in the header.
        if key.find ("_hdr") < 0 and phdr.has_key (key):
            phdr[key] = reffiles[key+"_hdr"]

    for key in ["cal_ver", "opt_elem", "cenwave", "fpoffset", "obstype",
                "exptype"]:
        if phdr.has_key (key):
            phdr[key] = info[key]
    #if phdr.has_key ("exptype"):
    #    # Override exptype, except for the case of an imaging wavecal.
    #    if info["obstype"] != "IMAGING" or info["targname"] != "WAVE":
    #        phdr["exptype"] = info["exptype"]

    if hdr.has_key ("dispaxis"):
        hdr["dispaxis"] = info["dispaxis"]

    hdr.update ("x_offset", info["x_offset"])

def updatePulseHeightKeywords (hdr, segment, low, high):
    """Update the screening limit keywords for pulse height.

    This is only used for FUV data, since NUV doesn't have pulse height info.

    arguments:
    hdr            header with keywords to be modified
    segment        FUVA or FUVB (last character used to construct keyword names)
    low, high      values for PHALOWR[AB] and PHAUPPR[AB] respectively
    """

    key_low  = "PHALOWR" + segment[-1]
    hdr.update (key_low, low)
    key_high = "PHAUPPR" + segment[-1]
    hdr.update (key_high, high)

def getPulseHeightRange (hdr, segment):
    """Get the pulse height range that was used for PHACORR.

    @param hdr: extension header of corrtag, counts, flt, etc
    @type hdr: pyfits Header object
    @param segment: segment name ("FUVA" or "FUVB")
    @type segment: string

    @return: "ll_hh", where ll is the lower limit and hh is the upper limit
    @rtype: string, or None if keyword(s) are missing or less than 0
    """

    if segment[:3] != "FUV":
        return None

    # These keywords were assigned when PHACORR was done.
    key_low  = "PHALOWR" + segment[-1]
    low = hdr.get (key_low, -1)
    key_high = "PHAUPPR" + segment[-1]
    high = hdr.get (key_high, -1)

    if low < 0:
        low = None
    if high < 0:
        high = None

    if low is None or high is None:
        return None

    return "%2d_%2d" % (low, high)

def tempPulseHeightRange (ref):
    """Get keyword PHARANGE from the primary header of a reference file.

    @param ref: name of a reference file
    @type ref: string

    @return: value of keyword PHARANGE, or None if the keyword is missing
    @rtype: string, or None
    """

    fd = pyfits.open (ref, "readonly")
    ref_pharange = fd[0].header.get ("pharange", None)
    fd.close()

    return ref_pharange

def comparePulseHeightRanges (pharange, ref_pharange, refname):
    """Compare pharange with the pulse height range from the phatab.

    @param pharange: pulse height range from the PHATAB, formatted "ll_hh",
        where ll and hh are the lower and upper limits
    @type pharange: string, or None
    @param ref_pharange: pulse height range used when calibrating the data
        used for creating the reference file (refname), formatted "ll_hh",
        where ll and hh are the lower and upper limits
    @type ref_pharange: string, or None
    @param refname: name of reference file for comparing ranges (only used
        for printing a warning message)
    @type refname: string
    """

    if pharange is None or ref_pharange is None:
        return

    words = pharange.split ("_")
    low = int (words[0])
    high = int (words[1])

    ref_words = ref_pharange.split ("_")
    if len (ref_words) != 2:
        printWarning ("Can't compare pulse height ranges for %s; "
                      "PHARANGE = %s" % (refname, ref_pharange))
        return
    ref_low = int (ref_words[0])
    ref_high = int (ref_words[1])
    if ref_low != low or ref_high != high:
        printWarning ("Pulse height ranges for %s don't agree:" % refname)
        printContinuation ("PHATAB limits are %d to %d, "
                           "but PHARANGE limits are %d to %d" %
                           (low, high, ref_low, ref_high))

def getSwitch (phdr, keyword):
    """Get the value of a calibration switch from a primary header.

    The value will be converted to upper case.  If the keyword is STATFLAG,
    the header value T or F will be converted to PERFORM or OMIT
    respectively.

    arguments:
    phdr           primary header
    keyword        name of keyword to get from header
    """

    if phdr.has_key (keyword):
        switch = phdr[keyword]
        if keyword.upper() == "STATFLAG":
            if switch:
                switch = "PERFORM"
            else:
                switch = "OMIT"
        switch = switch.upper()
    else:
        switch = NOT_APPLICABLE

    return switch

def setVerbosity (verbosity_level):
    """Copy verbosity to a variable that is global for this file.

    argument:
    verbosity_level   an integer value indicating the level of verbosity
    """

    global verbosity
    verbosity = verbosity_level

def checkVerbosity (level):
    """Return true if verbosity is at least as great as level.

    >>> setVerbosity (VERBOSE)
    >>> print checkVerbosity (QUIET)
    1
    >>> print checkVerbosity (VERBOSE)
    1
    >>> print checkVerbosity (VERY_VERBOSE)
    0
    """

    return (verbosity >= level)

def setWriteToTrailer (flag=False):
    """Set the flag to indicate whether we should write to trailer files.

    @param flag: if True, write to trailer file(s)
    @type flag: boolean
    """

    global write_to_trailer

    write_to_trailer = flag

def openTrailer (filename):
    """Open the trailer file for 'filename' in append mode.

    @param filename: name of an input (science or wavecal) file
    @type filename: string
    """

    global fd_trl
    global write_to_trailer

    if not write_to_trailer:
        fd_trl = None
        return

    closeTrailer()

    fd_trl = open (filename, 'a')

def writeVersionToTrailer():
    """Write the calcos version string to the trailer file."""

    if fd_trl is not None:
        fd_trl.write ("CALCOS version " + CALCOS_VERSION + "\n")
        fd_trl.flush()

def closeTrailer():
    """Close the trailer file if it is open."""

    global fd_trl

    if fd_trl is not None and not fd_trl.closed:
        fd_trl.close()
    fd_trl = None

def printMsg (message, level=QUIET):
    """Print 'message' if verbosity is at least as great as 'level'.

    >>> setVerbosity (VERBOSE)
    >>> printMsg ("quiet", QUIET)
    quiet
    >>> printMsg ("verbose", VERBOSE)
    verbose
    >>> printMsg ("very verbose", VERY_VERBOSE)
    """

    if verbosity >= level:
        print message
        sys.stdout.flush()
        if fd_trl is not None:
            fd_trl.write (message+"\n")
            fd_trl.flush()

def printIntro (str):
    """Print introductory message.

    argument:
    str            string to be printed
    """

    printMsg ("", VERBOSE)
    printMsg (str + " -- " + returnTime(), VERBOSE)

def printFilenames (names, shift_file=None, stimfile=None, livetimefile=None):
    """Print input and output filenames.

    arguments:
    names         a list of (label, filename) tuples
    shift_file    name of input text file to specify shift1 and shift2
    stimfile      name of output text file for stim positions (or None)
    livetimefile  name of output text file for livetime factors (or None)

    >>> setVerbosity (VERBOSE)
    >>> names = [("Input", "abc_raw.fits"), ("Output", "abc_flt.fits")]
    >>> printFilenames (names)
    Input     abc_raw.fits
    Output    abc_flt.fits
    >>> printFilenames (names, stimfile="stim.txt", livetimefile="live.txt")
    Input     abc_raw.fits
    Output    abc_flt.fits
    stim locations log file   stim.txt
    livetime factors log file live.txt
    """

    for (label, filename) in names:
        printMsg ("%-10s%s" % (label, filename), VERBOSE)

    if shift_file is not None:
        printMsg ("wavecal shifts overridden by file " + shift_file, VERBOSE)
    if stimfile is not None:
        printMsg ("stim locations log file   " + stimfile, VERBOSE)
    if livetimefile is not None:
        printMsg ("livetime factors log file " + livetimefile, VERBOSE)

def printMode (info):
    """Print info about the observation mode.

    argument:
    info          dictionary of header keywords and values
    """

    if info["detector"] == "FUV":
        printMsg ("DETECTOR  FUV, segment " + info["segment"][-1], VERBOSE)
    else:
        printMsg ("DETECTOR  NUV", VERBOSE)
    printMsg ("EXPTYPE   " + info["exptype"], VERBOSE)
    if info["obstype"] == "SPECTROSCOPIC":
        printMsg ("OPT_ELEM  " + info["opt_elem"] + \
              ", CENWAVE " + str (info["cenwave"]) + \
              ", FPOFFSET " + str (info["fpoffset"]), VERBOSE)
    else:
        printMsg ("OPT_ELEM  " + info["opt_elem"], VERBOSE)
    printMsg ("APERTURE  " + info["aperture"], VERBOSE)

    printMsg ("", VERBOSE)

def printSwitch (keyword, switches):
    """Print calibration switch name and value.

    arguments:
    keyword       keyword name of calibration switch (e.g. "flatcorr")
    switches      dictionary of calibration switches

    >>> setVerbosity (VERBOSE)
    >>> switches = {"statflag": "PERFORM", "flatcorr": "PERFORM", "geocorr": "COMPLETE", "randcorr": "SKIPPED"}
    >>> printSwitch ("statflag", switches)
    STATFLAG  T
    >>> printSwitch ("flatcorr", switches)
    FLATCORR  PERFORM
    >>> printSwitch ("geocorr", switches)
    GEOCORR   OMIT (already complete)
    >>> printSwitch ("randcorr", switches)
    RANDCORR  OMIT (skipped)
    """

    key_upper = keyword.upper()
    value = switches[keyword.lower()]
    if key_upper == "STATFLAG":
        if value == "PERFORM":
            message = "STATFLAG  T"
        else:
            message = "STATFLAG  F"
    else:
        if value == "COMPLETE":
            message = "%-9s OMIT (already complete)" % key_upper
        elif value == "SKIPPED":
            message = "%-9s OMIT (skipped)" % key_upper
        else:
            message = "%-9s %s" % (key_upper, value)
    printMsg (message, VERBOSE)

def printRef (keyword, reffiles):
    """Print reference file keyword and file name.

    arguments:
    keyword       keyword name for reference file name (e.g. "flatfile")
    reffiles      dictionary of reference file names

    >>> setVerbosity (VERBOSE)
    >>> reffiles = {"flatfile": "abc_flat.fits", "flatfile_hdr": "lref$abc_flat.fits"}
    >>> printRef ("flatfile", reffiles)
    FLATFILE= lref$abc_flat.fits
    """

    key_upper = keyword.upper()
    key_lower = keyword.lower()
    printMsg ("%-8s= %s" % (key_upper, reffiles[key_lower+"_hdr"]), VERBOSE)

def printWarning (message, level=QUIET):
    """Print a warning message."""

    printMsg ("Warning:  " + message, level)

def printError (message):
    """Print an error message."""

    printMsg ("ERROR:  " + message, level=QUIET)

def printContinuation (message, level=QUIET):
    """Print a continuation line of a warning or error message."""

    printMsg ("    " + message, level)

def returnTime():
    """Return the current date and time, formatted into a string."""

    return time.strftime ("%d-%b-%Y %H:%M:%S %Z", time.localtime (time.time()))

def getPedigree (switch, refkey, filename, level=VERBOSE):
    """Return the value of the PEDIGREE keyword.

    @param switch: keyword name for calibration switch
    @type switch: string
    @param refkey: keyword name for the reference file
    @type refkey: string
    @param filename: name of the reference file
    @type filename: string
    @param level: QUIET, VERBOSE, or VERY_VERBOSE
    @type level: integer

    @return: the value of the PEDIGREE keyword, or "OK" if not found
    @rtype: string
    """

    if filename == "N/A":
        return "OK"

    fd = pyfits.open (filename, mode="readonly")
    pedigree = fd[0].header.get ("pedigree", "OK")
    fd.close()
    if pedigree == "DUMMY":
        printWarning ("%s %s is a dummy file" % (refkey.upper(), filename),
                      level=VERBOSE)
        printContinuation ("so %s will not be done." %
                           switch.upper(), level=VERBOSE)

    return pedigree

def getApertureKeyword (hdr, truncate=1):
    """Get the value of the APERTURE keyword.

    arguments:
    hdr           pyfits Header object
    truncate      if true, strip "-FUV" or "-NUV" from keyword value
    """

    aperture = hdr.get ("aperture", NOT_APPLICABLE)
    if aperture == "RelMvReq":
        aperture = "PSA"
    elif truncate and aperture != NOT_APPLICABLE:
        aperture = aperture[0:3]

    return aperture

def expandFileName (filename):
    """Expand environment variable in a file name.

    If the input file name begins with either a Unix-style or IRAF-style
    environment variable (e.g. $lref/name_dqi.fits or lref$name_dqi.fits
    respectively), this routine expands the variable and returns a complete
    path name for the file.

    argument:
    filename      a file name, possibly including an environment variable
    """

    n = filename.find ("$")
    if n == 0:
        if filename != NOT_APPLICABLE:
            # Unix-style file name.
            filename = os.path.expandvars (filename)
    elif n > 0:
        # IRAF-style file name.
        temp = "$" + filename[0:n] + os.sep + filename[n+1:]
        filename = os.path.expandvars (temp)
        # If filename contains "//", delete one of them.
        double_sep = os.sep + os.sep
        i = filename.find (double_sep)
        if i != -1:
            filename = filename[:i+1] + filename[i+2:]

    return filename

def changeSegment (filename, detector, segment):
    """Replace '_a' with '_b' or vice versa, if appropriate.

    This was written for auto/GO wavecal file names for FUV data.  Wavecals
    are processed from the x1d file, and the name of the raw file is for
    the first segment in the input list (which will be FUVA if both segments
    are present).  When calibrating segment B data, the name or names of
    the wavecal files need to be changed to end in "_b.fits" instead of
    "_a.fits".

    @param filename: one or more file names, separated by spaces
    @type filename: string
    @param detector: FUV or NUV
    @type filename: string
    @param segment: FUVA or FUVB, if detector is FUV
    @type segment: string

    @return: name(s) with '_a' replaced with '_b', or vice versa, or no change
    @rtype: string
    """

    if detector != "FUV":
        return filename

    if segment == "FUVB":
        names = filename.split()
        new_names = []
        for name in names:
            if name.endswith ("_a.fits"):
                n = len (name) - 7
                name = name[:n] + "_b.fits"
            new_names.append (name)
        filename = " ".join(new_names)
    elif segment == "FUVA":
        names = filename.split()
        new_names = []
        for name in names:
            if name.endswith ("_b.fits"):
                n = len (name) - 7
                name = name[:n] + "_a.fits"
            new_names.append (name)
        filename = " ".join(new_names)

    return filename

def findRefFile (ref, missing, wrong_filetype, bad_version):
    """Check for the existence of a reference file.

    arguments:
      (missing, wrong_filetype and bad_version are dictionaries, with the
       reference file keyword as key.)
    ref             a dictionary with the following keys:
                      keyword (e.g. "FLATFILE")
                      filename (name of file)
                      calcos_ver (calcos version number)
                      min_ver (minimum acceptable version number)
                      filetype (e.g. "FLAT FIELD REFERENCE IMAGE")
    missing         messages about missing reference files
    wrong_filetype  messages about wrong FILETYPE keyword in reference files
    bad_version     messages about inconsistent version strings

    If the reference file does not exist, its name is added to the 'missing'
    dictionary.  If the file does exist, open the file and compare
    'filetype' with the value of the FILETYPE keyword in the primary header.
    If they're not the same (unless FILETYPE is "ANY"), then an entry is
    added to the 'wrong_filetype' dictionary.  The VCALCOS keyword is also
    gotten from the primary header (with a default value of "1.0").  If the
    version of the reference file is not consistent with calcos, the
    reference file name and error message will be added to the 'bad_version'
    dictionary.
    """

    keyword    = ref["keyword"]
    filename   = ref["filename"]
    calcos_ver = ref["calcos_ver"]
    min_ver    = ref["min_ver"]
    filetype   = ref["filetype"]

    if os.access (filename, os.R_OK):

        fd = pyfits.open (filename, mode="readonly")
        phdr = fd[0].header

        phdr_filetype = phdr.get ("FILETYPE", "ANY")
        if phdr_filetype != "ANY" and phdr_filetype != filetype:
            wrong_filetype[keyword] = (filename, filetype)

        if min_ver != "ANY":
            phdr_ver = phdr.get ("VCALCOS", "1.0")
            if type (phdr_ver) is not types.StringType:
                phdr_ver = str (phdr_ver)
            compare = cmpVersion (min_ver, phdr_ver, calcos_ver)
            if compare < 0:
                bad_version[keyword] = (filename,
                "  the reference file must be at least version " + min_ver)
            elif compare > 0:
                bad_version[keyword] = (filename,
                "  to use this reference file you must have calcos version " + \
                 phdr_ver + " or later.")

        fd.close()

    else:

        missing[keyword] = filename

def fitQuadratic (x, y):
    """Fit a quadratic to y vs x.

    @param x: array of independent values
    @type x: ndarray
    @param y: array of dependent values
    @type y: ndarray

    @return: (coeff, var), where coeff is an array of the coefficients
        of the fit (coeff[0] + coeff[1]*x + coeff[2]*x**2), and var is an
        array of the corresponding variances; coeff and var will be None if
        there was a LinAlgError.
    @rtype: tuple
    """

    assert len (x) == len (y)
    n = float (len (x))

    y0 = y[0]
    yp = (y - y0)

    sum_x = x.sum (dtype=np.float64).item()
    sum_x2 = (x**2).sum (dtype=np.float64).item()
    sum_x3 = (x**3).sum (dtype=np.float64).item()
    sum_x4 = (x**4).sum (dtype=np.float64).item()
    sum_y = yp.sum (dtype=np.float64).item()
    sum_yx = (yp*x).sum (dtype=np.float64).item()
    sum_yx2 = (yp*x**2).sum (dtype=np.float64).item()

    m = np.array ([[n,      sum_x,  sum_x2],
                   [sum_x,  sum_x2, sum_x3],
                   [sum_x2, sum_x3, sum_x4]])
    v = np.array ([sum_y, sum_yx, sum_yx2])

    succeeded = True
    try:
        coeff = LA.solve (m, v)
        m_inv = LA.inv (m)
    except LA.LinAlgError:
        succeeded = False

    if not succeeded:
        coeff = None
        var = None
    else:
        coeff[0] += y0
        (a0, a1, a2) = coeff
        if len (x) > 3:
            residual = y - (a0 + a1*x + a2*x**2)
            chisq = (residual**2).sum()
            scatter = math.sqrt (chisq / (n - 3.))
        else:
            scatter = 0.
        var = np.array ([m_inv[0,0], m_inv[1,1], m_inv[2,2]]) * scatter

    return (coeff, var)

def centerOfQuadratic (coeff, var):
    """Find the center of a quadratic function from its coefficients.

    @param coeff: the coefficients of the fit (or None if not determined):
           y = coeff[0] + coeff[1]*x + coeff[2]*x**2
    @type coeff: ndarray
    @param var: the variances of the coefficients
    @type var: ndarray

    @return: (x_min, x_min_sigma), where x is value at which y is an extremum,
        and x_min_sigma is the error estimate for x_min, based on the scatter
        of the values around the fitted curve; the values will be (None, 0.)
        if coeff is None or if the second-order coefficient is zero
    @rtype: tuple
    """

    if coeff is None or coeff[2] == 0:
        x_min = None
        x_min_sigma = 0.
    else:
        a1 = coeff[1].item()
        a2 = coeff[2].item()
        var1 = var[1].item()
        var2 = var[2].item()
        x_min = -a1 / (2. * a2)
        x_min_sigma = 0.5 * math.sqrt (var1 / a2**2 + var2 * a1**2 / a2**4)

    return (x_min, x_min_sigma)

def fitQuartic (x, y):
    """not currently used"""

    assert len (x) == len (y)
    n = float (len (x))

    y0 = y[0]
    yp = (y - y0)

    sum_x = x.sum (dtype=np.float64).item()
    sum_x2 = (x**2).sum (dtype=np.float64).item()
    sum_x3 = (x**3).sum (dtype=np.float64).item()
    sum_x4 = (x**4).sum (dtype=np.float64).item()
    sum_x5 = (x**5).sum (dtype=np.float64).item()
    sum_x6 = (x**6).sum (dtype=np.float64).item()
    sum_x7 = (x**7).sum (dtype=np.float64).item()
    sum_x8 = (x**8).sum (dtype=np.float64).item()
    sum_y = yp.sum (dtype=np.float64).item()
    sum_yx = (yp*x).sum (dtype=np.float64).item()
    sum_yx2 = (yp*x**2).sum (dtype=np.float64).item()
    sum_yx3 = (yp*x**3).sum (dtype=np.float64).item()
    sum_yx4 = (yp*x**4).sum (dtype=np.float64).item()

    m = np.array ([[n,      sum_x,  sum_x2,  sum_x3, sum_x4],
                   [sum_x,  sum_x2, sum_x3,  sum_x4, sum_x5],
                   [sum_x2, sum_x3, sum_x4,  sum_x5, sum_x6],
                   [sum_x3, sum_x4, sum_x5,  sum_x6, sum_x7],
                   [sum_x4, sum_x5, sum_x6,  sum_x7, sum_x8]])
    v = np.array ([sum_y, sum_yx, sum_yx2, sum_yx3, sum_yx4])

    try:
        coeff = LA.solve (m, v)
        m_inv = LA.inv (m)
    except LA.LinAlgError:
        succeeded = False

    if not succeeded:
        coeff = None
        var = None
    else:
        (a0, a1, a2, a3, a4) = coeff
        a0 += y0
        if len (x) > 5:
            residual = y - (a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)
            chisq = (residual**2).sum()
            scatter = math.sqrt (chisq / (n - 3.))
        else:
            scatter = 0.
        var = np.array ([m_inv[0,0], m_inv[1,1], m_inv[2,2],
                         m_inv[3,3], m_inv[4,4]]) * scatter

    return (coeff, var)

def centerOfQuartic (x, coeff):
    """Find the center of a quartic function from its coefficients.

    not currently used

    @return: the x value at which y is a minimum, or None if coeff is None
    @rtype: float
    """

    if coeff is None:
        return None

    a0 = coeff[0].item()
    a1 = coeff[1].item()
    a2 = coeff[2].item()
    a3 = coeff[3].item()
    a4 = coeff[4].item()
    yp = a1 + 2.*a2*x + 3.*a3*x**2 + 4.*a4*x**3

    xminmax = []
    for i in range (len (x) - 1):
        # opposite slopes at i and i+1?
        if yp[i] * yp[i+1] < 0.:
            value = x[i] + abs (yp[i] / (yp[i+1] - yp[i]))
            xminmax.append ((i, value))
    if len (xminmax) == 0:
        return None
    x_min = xminmax[0][1]
    if len (xminmax) > 1:
        for (i, value) in enumerate (xminmax):
            if value < minvalue:
                x_min = xminmax[i][1]

    return x_min

def precess (t, target):
    """Precess target to the time of observation.

    This function is currently not used.
    It could be called by timetag.heliocentricVelocity.

    @param t: time (MJD)
    @type t: float
    @param target: unit vector pointing toward the target, J2000 coordinates
    @type target: sequence type

    @return: target coordinates precessed to time t
    @rtype: list
    """

    # 51544.5 is MJD for 2000 Jan 1.5 UT, or JD 2451545.0
    dt = (t - 51544.5) / 36525.
    dt2 = dt * dt
    dt3 = dt * dt * dt

    zeta = 2306.2181 * dt + 0.30188 * dt2 + 0.017998 * dt3

    z = 2306.2181 * dt + 1.09468 * dt2 + 0.018203 * dt3

    theta = 2004.3109 * dt - 0.42665 * dt2 - 0.041833 * dt3

    # convert from arc seconds to radians
    zeta = math.radians (zeta / 3600.)
    z = math.radians (z / 3600.)
    theta = math.radians (theta / 3600.)

    # convert zeta, z, theta to a rotation matrix
    a = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # first row
    a[0] =  math.cos (z) * math.cos (theta) * math.cos (zeta) - \
            math.sin (z) *                    math.sin (zeta)
    a[1] = -math.cos (z) * math.cos (theta) * math.sin (zeta) - \
            math.sin (z) *                    math.cos (zeta)
    a[2] = -math.cos (z) * math.sin (theta)

    # second row
    a[3] =  math.sin (z) * math.cos (theta) * math.cos (zeta) + \
            math.cos (z) *                    math.sin (zeta)
    a[4] = -math.sin (z) * math.cos (theta) * math.sin (zeta) + \
            math.cos (z) *                    math.cos (zeta)
    a[5] = -math.sin (z) * math.sin (theta)

    # third row
    a[6] =                 math.sin (theta) * math.cos (zeta)
    a[7] =                -math.sin (theta) * math.sin (zeta)
    a[8] =                 math.cos (theta)

    # Multiply:  a * target
    targ = [0., 0., 0.]
    targ[0] = a[0] * target[0] + a[1] * target[1] + a[2] * target[2]
    targ[1] = a[3] * target[0] + a[4] * target[1] + a[5] * target[2]
    targ[2] = a[6] * target[0] + a[7] * target[1] + a[8] * target[2]

    return targ

def cmpVersion (min_ver, phdr_ver, calcos_ver):
    """Compare version strings.

    arguments:
    min_ver      calcos requires the reference file to be at least this
                   version
    phdr_ver     version of the reference file, read from its primary header
    calcos_ver   version of calcos

    This function returns 0 if the 'phdr_ver' is compatible with
    'calcos_ver' and 'min_ver', i.e. that the following conditions are met:

        min_ver <= phdr_ver <= calcos_ver

    If min_ver > phdr_ver, this function returns -1.
    If phdr_ver > calcos_ver, this function returns +1.

    Each string is first separated into a list of substrings, splitting
    on ".", and comparisons are made on the substrings one at a time.
    A comparison between min_ver="1a" and phdr_ver="1.0a" will fail,
    for example, because the strings will be separated into parts before
    comparing, and "1a" > "1".

    >>> print cmpVersion ("1", "1", "1.1")
    0
    >>> print cmpVersion ("1", "1.1", "1")
    1
    >>> print cmpVersion ("1.1", "1", "1")
    -1
    >>> print cmpVersion ("1.1", "1.1", "1.2")
    0
    >>> print cmpVersion ("1.1", "1.2", "1.1")
    1
    >>> print cmpVersion ("1.2", "1.1", "1.1")
    -1
    >>> print cmpVersion ("1.0", "1", "1a")
    0
    >>> print cmpVersion ("1.0", "1.0a", "1")
    1
    >>> print cmpVersion ("1.0a", "1", "1")
    -1
    >>> print cmpVersion ("1.0a", "1.0a", "1b")
    0
    >>> print cmpVersion ("1a", "1.0a", "1b")
    -1
    """

    minv = min_ver.split ('.')
    phdrv = phdr_ver.split ('.')
    calv = calcos_ver.split ('.')

    length = min (len (minv), len (phdrv), len (calv))

    # These are initial values.  They'll be reset if either test passes
    # (because of an inequality in a part of the version string), in which
    # case tests on subsequent parts of the version strings will be omitted.
    passed_min_test = 0
    passed_calcos_test = 0

    for i in range (length):
        if not passed_min_test:
            cmp = cmpPart (minv[i], phdrv[i])
            if cmp < 0:
                passed_min_test = 1
            elif cmp > 0:
                return -1
        if not passed_calcos_test:
            cmp = cmpPart (phdrv[i], calv[i])
            if cmp < 0:
                passed_calcos_test = 1
            elif cmp > 0:
                return 1

    if passed_min_test or passed_calcos_test:
        return 0

    if len (minv) > len (phdrv):
        return -1
    if len (phdrv) > len (calv):
        return 1

    return 0

def cmpPart (s1, s2):
    """Compare two strings.

    s1 and s2 are "parts" of version strings, i.e. each is a simple integer,
    possibly with one or more appended letters.  The function value will be
    -1, 0, or +1, depending on whether s1 is less than, equal to, or greater
    than s2 respectively.  Comparison is done first on the numerical part,
    and any appended string is used to break a tie.

    >>> print cmpPart ("1", "01")
    0
    >>> print cmpPart ("14", "104")
    -1
    >>> print cmpPart ("9", "13a")
    -1
    >>> print cmpPart ("13", "13a")
    -1
    >>> print cmpPart ("13a", "14")
    -1
    >>> print cmpPart ("13a", "13b")
    -1
    """

    if s1 == s2:
        return 0

    nine = ord ('9')

    int1 = 0
    str1 = ""
    for i in range (len (s1)):
        ich = ord (s1[i])
        if ich > nine:
            if i > 0:
                int1 = int (s1[0:i])
            str1 = s1[i:]
            break
        int1 = int (s1[0:i+1])

    int2 = 0
    str2 = ""
    for i in range (len (s2)):
        ich = ord (s2[i])
        if ich > nine:
            if i > 0:
                int2 = int (s2[0:i])
            str2 = s2[i:]
            break
        int2 = int (s2[0:i+1])

    if int1 < int2:
        return -1
    elif int1 > int2:
        return 1
    else:
        # The numerical parts are identical; use the letter(s) to break the tie.
        if str1 == str2:
            return 0
        elif str1 == "":
            return -1                   # the first string is "smaller"
        elif str2 == "":
            return 1                    # the first string is "larger"
        else:
            length = min (len (str1), len (str2))
            for i in range (length):
                ich1 = ord (str1[i])
                ich2 = ord (str2[i])
                if ich1 < ich2:
                    return -1
                elif ich1 > ich2:
                    return 1
            if len (str1) < len (str2):
                return -1
            else:
                return 1


def _test():
    import doctest, cosutil
    return doctest.testmod (cosutil)

if __name__ == "__main__":
    _test()
