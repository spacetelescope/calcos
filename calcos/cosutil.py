#! /usr/bin/env python

from __future__ import absolute_import, division         # confidence high
import math
import os
import shutil
import sys
import time
import types
import copy
import numpy as np
import numpy.linalg as LA
import astropy.io.fits as fits
from astropy.stats import poisson_conf_interval
from . import ccos
from .calcosparam import *       # parameter definitions

# initial value
verbosity = VERBOSE

# for appending to a trailer file
fd_trl = None
# if this is False, writing to trailer files will be disabled
write_to_trailer = True

# Used as a default value in updateDQArray.  The actual value should be
# gotten via keyword WIDEN in the BPIXTAB table header.
PIXEL_FRACTION = 0.25

def writeOutputEvents(infile, outfile):
    """
    This function creates a recarray object with the column definitions
    appropriate for a corrected time-tag table, reads an input events table
    into this object, and writes it to the output file.  If the input file
    contains a GTI table, that will be copied unchanged to output.

    If the input is already a corrtag table (if the table in the first
    extension contains the column XFULL), then the file will be copied
    to output without change.

    Parameters
    ----------
    infile: str
        Name of the input FITS file containing an EVENTS table and
        optionally a GTI table.

    outfile: str
        Name of file for output EVENTS table (and GTI table).
    """

    ifd = fits.open(infile, mode="copyonwrite")
    events_extn = ifd["EVENTS"]
    indata = events_extn.data
    if indata is None:
        nrows = 0
    else:
        nrows = len(indata)

    # If the input is already a corrtag file, just copy it.
    if isCorrtag(infile):
        ifd.close()
        shutil.copy(infile, outfile)
        return nrows

    detector = ifd[0].header.get("detector", "FUV")
    tagflash = (ifd[0].header.get("tagflash", default="NONE") != "NONE")

    # Create the output events HDU.
    hdu = createCorrtagHDU(nrows, detector, events_extn)

    if nrows == 0:
        primary_hdu = fits.PrimaryHDU(header=ifd[0].header)
        ofd = fits.HDUList(primary_hdu)
        updateFilename(ofd[0].header, outfile)
        ofd.append(hdu)
        if len(ifd) == 3:
            ofd.append(ifd["GTI"])
        ofd.writeto(outfile)
        ifd.close()
        return nrows

    outdata = hdu.data

    # Copy data from the input table to the output HDU object.

    outdata.field("TIME")[:] = indata.field("TIME")

    outdata.field("RAWX")[:] = indata.field("RAWX")
    outdata.field("RAWY")[:] = indata.field("RAWY")
    outdata.field("XCORR")[:] = indata.field("RAWX")
    outdata.field("YCORR")[:] = indata.field("RAWY")

    outdata.field("XDOPP")[:] = np.zeros(nrows, dtype=np.float32)
    outdata.field("XFULL")[:] = np.zeros(nrows, dtype=np.float32)
    outdata.field("YFULL")[:] = np.zeros(nrows, dtype=np.float32)
    outdata.field("WAVELENGTH")[:] = np.zeros(nrows, dtype=np.float32)

    outdata.field("EPSILON")[:] = np.ones(nrows, dtype=np.float32)
    outdata.field("DQ")[:] = np.zeros(nrows, dtype=np.int16)
    if detector == "FUV":
        outdata.field("PHA")[:] = indata.field("PHA")
    else:
        outdata.field("PHA")[:] = 0

    primary_hdu = fits.PrimaryHDU(header=ifd[0].header)
    ofd = fits.HDUList(primary_hdu)
    updateFilename(ofd[0].header, outfile)
    ofd.append(hdu)

    # GTI table.
    if len(ifd) == 3:
        ofd.append(ifd["GTI"])

    ofd.writeto(outfile)
    ifd.close()

    return nrows

def isCorrtag(filename):
    """Determine whether 'filename' is a corrtag file.

    A corrtag file contains a table in the first extension, and there
    will be a column with the name "XFULL".

    Parameters
    ----------
    filename: str
        Name of a file.

    Returns
    -------
    boolean
        True if the first extension of 'filename' is a corrtag table.
    """

    fd = fits.open(filename, mode="readonly")
    if len(fd) < 2:                     # no extensions?
        fd.close()
        return False

    # Find an EVENTS table (any one, if there is more than one).
    hdunum = 0
    for i in range(1, len(fd)):
        hdu = fd[i]
        extname = hdu.header.get("extname", "MISSING")
        if extname.upper() == "EVENTS":
            hdunum = i
            break

    if hdunum < 1:
        fd.close()
        return False

    hdr = fd[hdunum].header
    data = fd[hdunum].data
    got_xfull = False                   # initial value
    if data is None or len(data) == 0:
        # Check each of the TTYPEi keywords, looking for column XFULL.
        ncols = hdr.get("tfields", 0)
        for i in range(1, ncols+1):
            key = "ttype%d" % i
            ttype = hdr.get(key, "missing").lower()
            if ttype == "xfull":
                got_xfull = True
                break
    else:
        got_xfull = findColumn(data, "xfull")
    fd.close()

    return got_xfull

def createCorrtagHDU(nrows, detector, hdu):
    """Create the output events HDU.

    Parameters
    ----------
    nrows: int
        Number of rows to allocate (may be zero).

    detector: {"FUV", "NUV"}
        Detector name.

    hdu: fits HDU object
        Events extension hdu.

    Returns
    -------
    fits BinTableHDU object
        Header/data unit for a corrtag table.
    """

    col = []
    #
    # Copy over the TIME, RAWX and RAWY columns from the input if possible
    # to preserve column attributes
    try:
        col.append(hdu.column["TIME"])
    except (AttributeError, KeyError):
        col.append(fits.Column(name="TIME", format="1E", unit="s"))
    try:
        col.append(hdu.column["RAWX"])
    except (AttributeError, KeyError):
        col.append(fits.Column(name="RAWX", format="1I", unit="pixel"))
    try:
        col.append(hdu.column["RAWY"])
    except (AttributeError, KeyError):
        col.append(fits.Column(name="RAWY", format="1I", unit="pixel"))
    col.append(fits.Column(name="XCORR", format="1E", unit="pixel"))
    col.append(fits.Column(name="YCORR", format="1E", unit="pixel"))
    col.append(fits.Column(name="XDOPP", format="1E", unit="pixel"))
    col.append(fits.Column(name="XFULL", format="1E", unit="pixel"))
    col.append(fits.Column(name="YFULL", format="1E", unit="pixel"))
    col.append(fits.Column(name="WAVELENGTH", format="1E",
                           unit="angstrom", disp="F9.4"))
    col.append(fits.Column(name="EPSILON", format="1E"))
    col.append(fits.Column(name="DQ", format="1I"))
    col.append(fits.Column(name="PHA", format="1B"))
    cd = fits.ColDefs(col)

    # Rename or delete some image-specific keywords.
    header = imageHeaderToCorrtag(hdu.header)

    newheader = remove_WCS_keywords(header, cd)

    outhdu = fits.BinTableHDU.from_columns(cd, header=newheader, nrows=nrows)

    return outhdu

def remove_WCS_keywords(header, cd):
    """Remove WCS-specific keywords from the header of a table
    They should be column attributes

    Parameters
    ----------
    header: FITS header object
        Header that will be passed to BinTableHDU.from_columns

    cd: FITS column descriptor object
        Column descriptor that should contain the attributes referred to
        by the header keywords
    """
    newheader = header.copy()
    WCS_keywords = {'TCTYP': 'coord_type',
                    'TCUNI': 'coord_unit',
                    'TCRPX': 'coord_ref_point',
                    'TCRVL': 'coord_ref_value',
                    'TCDLT': 'coord_inc',
                    'TRPOS': 'time_ref_pos'}
    for keyword in header.keys():
        if keyword[0:5] in WCS_keywords.keys():
            index = int(keyword[5]) - 1
            keyword_value = header[keyword]
            column_value = cd[index].__getattribute__(WCS_keywords[keyword[0:5]])
            if keyword_value != column_value:
                cd[index].__setattr__(WCS_keywords[keyword[0:5]], keyword_value)
            del newheader[keyword]
    return newheader

def copyExptimeKeywords(inhdr, outhdr):
    """Copy the exposure time keywords from one header to another.

    This is for copying the exposure time keywords from the input extension
    header to the primary header of the csum file.

    Parameters
    ----------
    inhdr: pyfits Header object
        Input header.

    outhdr: pyfits Header object
        Output header.
    """

    outhdr["expstart"] = inhdr.get("expstart", -999.)
    outhdr["expend"] = inhdr.get("expend", -999.)
    exptime = inhdr.get("exptime", -999.)
    outhdr["exptime"] = exptime
    outhdr["rawtime"] = inhdr.get("rawtime", exptime)

def copyVoltageKeywords(inhdr, outhdr, detector):
    """Copy keywords for high voltages from one header to another.

    This is for copying the high voltage keywords from the input extension
    header to the primary header of the csum file.

    Parameters
    ----------
    inhdr: pyfits Header object
        Input header.

    outhdr: pyfits Header object
        Output header.

    detector: {"FUV", "NUV"}
        Detector name.
    """

    if detector == "FUV":
        outhdr["dethvla"] = inhdr.get("dethvla", -999.)
        outhdr["dethvlb"] = inhdr.get("dethvlb", -999.)
        outhdr["dethvca"] = inhdr.get("dethvca", -999.)
        outhdr["dethvcb"] = inhdr.get("dethvcb", -999.)
        outhdr["dethvna"] = inhdr.get("dethvna", -999.)
        outhdr["dethvnb"] = inhdr.get("dethvnb", -999.)
    elif detector == "NUV":
        outhdr["dethvl"] = inhdr.get("dethvl", -999.)
        outhdr["dethvc"] = inhdr.get("dethvc", -999.)

def copySubKeywords(inhdr, outhdr, subarray):
    """Copy the subarray keywords from one header to another.

    This is for copying the subarray keywords from the input extension
    header to the primary header of the csum file.

    Parameters
    ----------
    inhdr: pyfits Header Object
        Input header.

    outhdr: pyfits Header Object
        Output header.

    subarray: boolean
        True if the exposure used one or more subarrays.
    """

    if subarray:
        outhdr["nsubarry"] = inhdr.get("nsubarry", 0)
    else:
        outhdr["nsubarry"] = 0
    for i in range(8):
        x_corner_kwd = "corner%1dx" % i
        y_corner_kwd = "corner%1dy" % i
        x_size_kwd = "size%1dx" % i
        y_size_kwd = "size%1dy" % i
        outhdr[x_corner_kwd] = inhdr.get(x_corner_kwd, -1)
        outhdr[y_corner_kwd] = inhdr.get(y_corner_kwd, -1)
        outhdr[x_size_kwd] = inhdr.get(x_size_kwd, -1)
        outhdr[y_size_kwd] = inhdr.get(y_size_kwd, -1)

def dummyGTI(exptime):
    """Return a GTI table.

    Parameters
    ----------
    exptime: float
        Exposure time in seconds.

    Returns
    -------
    pyfits BinTableHDU object
        Header/data unit for a GTI table covering the entire exposure.
    """

    col = []
    col.append(fits.Column(name="START", format="1D", unit="s"))
    col.append(fits.Column(name="STOP", format="1D", unit="s"))
    cd = fits.ColDefs(col)
    hdu = fits.BinTableHDU.from_columns(cd, nrows=1)
    hdu.header["extname"] = "GTI"
    outdata = hdu.data
    outdata.field("START")[:] = 0.
    outdata.field("STOP")[:] = exptime

    return hdu

def returnGTI(infile):
    """Return a list of (start, stop) good time intervals.

    Parameters
    ----------
    infile: str
        Name of the input FITS file containing a GTI table.

    Returns
    -------
    list of two-element tuples
        Each tuple gives the start and stop times (seconds since the
        start of the exposure).
    """

    fd = fits.open(infile, mode="copyonwrite")

    # Find the GTI table with the largest value of EXTVER.
    last_extver = 0                     # initial value
    hdunum = 0
    for i in range(1, len(fd)):
        hdu = fd[i]
        extname = hdu.header.get("extname", "MISSING")
        if extname.upper() == "GTI":
            extver = hdu.header.get("extver", 1)
            if extver > last_extver:
                last_extver = extver
                hdunum = i

    if hdunum < 1:
        gti = []
    else:
        indata = fd[hdunum].data
        if indata is None or len(indata) == 0:
            gti = []
        else:
            nrows = indata.shape[0]
            start = indata.field("START")
            stop = indata.field("STOP")
            gti = [(start[i], stop[i]) for i in range(nrows)]

    return gti

def findColumn(table, colname):
    """Return True if colname is found (case-insensitive) in table.

    Parameters
    ----------
    table: string (if name of table) or FITS record object
        Name of table or data block for a FITS table.

    colname: str
        Name of column to test for existence in table.

    Returns
    -------
    boolean
        True if colname is in the table (without regard to case).
    """

    if isinstance(table, str):
        fd = fits.open(table, mode="copyonwrite")
        fits_rec = fd[1].data
        fd.close()
    else:
        fits_rec = table

    names = []
    for name in fits_rec.names:
        names.append(name.lower())

    if colname.lower() in names:
        return True
    else:
        return False

def getTable(table, filter, extension=1,
             exactly_one=False, at_least_one=False):
    """Return the data portion of a table.

    All rows that match the filter (a dictionary of column_name = value)
    will be returned.  If the value in the table is STRING_WILDCARD and
    the column contains strings, that value is regarded as matching that
    row.  Also, for a given filter key, if the value of the filter is
    STRING_WILDCARD or NOT_APPLICABLE, the test on filter will not be
    applied for that key (i.e. that filter element matches any row).

    It is an error if exactly_one or at_least_one is true but no row
    matches the filter.  A warning will be printed if exactly_one is true
    but more than one row matches the filter.

    Parameters
    ----------
    table: str
        Name of the reference table.

    filter: dictionary
        Each key/value pair in filter is used to select rows in the
        table.  The key is a column name, and if the filter value for that
        key matches the value in the column for one or more rows, those
        rows will be included in the array of rows that is returned.
        If the filter value is a tuple or list, however, the first element
        is taken to be a numpy function for a relation (such as
        np.greater for "greater than"), and the second element is taken
        to be the value against which the column values should be compared.
        This can be used for cases where rows need to be selected on a
        value, but the relation is not necessarily equality.  The rows
        selected are those for which the value in the table column (the
        column with the same name as the key) has the specified relation
        to the value in the filter.  For example, to select rows for which
        the values in the DATE column are greater than or equal to 55785.,
        use:
            filter = {"date": (np.greater_equal, 55785.)}

    extension: tuple, str, or int
        Identifier for the extension containing the table.

    exactly_one: boolean
        True to indicate that there must be one and only one matching row.

    at_least_one: boolean
        True to indicate that there must be at least one matching row.

    Returns
    -------
    array_like or None
        Pyfits table data object containing the selected row(s).  If the
        input table is empty, or if no rows match the selection criteria,
        None will be returned.
    """

    fd = fits.open(table, mode="copyonwrite")
    data = fd[extension].data
    if data is None or len(data) < 1:
        fd.close()
        return None

    # There will be one element of select_arrays for each non-trivial
    # selection criterion.  Each element of select_arrays is an array
    # of flags, true if the row matches the criterion.
    select_arrays = []
    for key in filter.keys():

        if filter[key] == STRING_WILDCARD or filter[key] == NOT_APPLICABLE:
            continue
        column = data.field(key)
        if isinstance(filter[key], tuple) or isinstance(filter[key], list):
            (relation_fcn, value) = filter[key]
            selected = relation_fcn(column, value)
        else:
            selected = (column == filter[key])

        # Test for for wildcards in the table.
        wild = None
        if isinstance(column, np.chararray):
            wild = (column == STRING_WILDCARD)
        if wild is not None:
            selected = np.logical_or(selected, wild)

        select_arrays.append(selected)

    if len(select_arrays) > 0:
        selected = select_arrays[0]
        for sel_i in select_arrays[1:]:
             selected = np.logical_and(selected, sel_i)
        newdata = data[selected]
    else:
        newdata = data.copy()

    fd.close()

    nselect = len(newdata)
    if nselect < 1:
        newdata = None

    if (exactly_one or at_least_one) and nselect < 1:
        message = "Table has no matching row;\n" + \
                  "table name is " + table + "\n" + \
                  "row selection is " + repr(filter)
        raise MissingRowError(message)

    if exactly_one and nselect > 1:
        printWarning("Table has more than one matching row;")
        printContinuation("table name is " + table)
        printContinuation("row selection is " + repr(filter))
        printContinuation("only the first will be used.")

    return newdata

def getColCopy(filename="", column=None, extension=1, data=None):
    """Return the specified column in native format.

    Specify either the data block (data) or the name of a file
    (filename), but not both.

    Parameters
    ----------
    filename: str
        The name of the FITS file.

    column: str or int
        Column name or number.

    extension: int
        Number of extension containing the table.

    data: array_like
        The data portion of a table.

    Returns
    -------
    array_like
        The column data.
    """

    if filename and data is not None:
        raise RuntimeError("Specify either filename or data, but not both.")

    if filename:
        fd = fits.open(filename, mode="copyonwrite")
        temp = fd[extension].data.field(column)
        fd.close()
    elif data is not None:
        temp = data.field(column)
    else:
        raise RuntimeError("Either filename or data must be specified.")

    x = np.empty(temp.shape, dtype=temp.dtype.type)
    x[...] = temp

    return x

def getTemplate(raw_template, x_offset, nelem):
    """Return the template spectrum embedded in a possibly larger array.

    Parameters
    ----------
    raw_template: array_like
        Template spectrum as read from the lamptab.

    x_offset: int
        Offset of raw_template in the extended template.

    nelem: int
        Length of template spectrum to return.

    Returns
    -------
    array_like
        A copy of raw_template, possibly padded with zeros on the left
        and right.
    """

    len_raw = len(raw_template)

    if x_offset == 0 and nelem == len_raw:
        return raw_template.copy()

    template = np.zeros(nelem, dtype=raw_template.dtype)
    template[x_offset:len_raw+x_offset] = raw_template

    return template

def checkForNoWavecalData(opt_elem, cenwave, segment, lamptab):
    """Read the HAS_LINES column to see whether to override wavecal info.

    For certain FUV modes, there is no detectable wavecal signal on
    segment B.  This is the case for G140L, but also for some blue central
    wavelengths for G130M.  The presence or absence of wavecal signal for
    segment B can be flagged in a HAS_LINES column in the LAMPTAB.  If this
    column is present, its value is taken to indicate whether the wavecal
    shift should be copied from the wavecal shifts for segment A.  If the
    column is not found, the test is based on the grating name;
    specifically, copy segment A wavecal shifts to segment B for G140L
    data.

    Parameters
    ----------
    opt_elem: str
        Grating name.

    cenwave: str
        Central wavelength, for spectroscopic data.

    segment: str
        Segment name.

    lamptab: str
        Wavecal lamp template reference table.

    Returns
    -------
    boolean
        True if there is no wavecal signal for this mode.
    """

    if findColumn(lamptab, "has_lines"):
        filter = {"opt_elem": opt_elem,
                  "cenwave": cenwave,
                  "segment": segment}
        lamp_info = getTable(lamptab, filter, at_least_one=True)
        # True if there are no wavecal lines for this mode.
        override_segment_B = not lamp_info.field("has_lines")[0]
    elif opt_elem == "G140L" and segment == "FUVB":
        override_segment_B = True
    else:
        override_segment_B = False

    return override_segment_B

def determineLivetime(countrate, obs_rate, live_factor):
    """Compute livetime factor from observed count rate.

    This is just linear interpolation in live_factor vs obs_rate.

    Parameters
    ----------
    countrate: float
        Observed count rate.

    obs_rate: array_like
        Observed count rate column from deadtab.

    live_factor: array_like
        Livetime factor column from deadtab.

    Returns
    -------
    float
        The interpolated livetime factor.
    """

    n = len(obs_rate)

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
        for i in range(n-1):
            if countrate < obs_rate[i+1]:
                p = (countrate - obs_rate[i]) / (obs_rate[i+1] - obs_rate[i])
                q = 1. - p
                livetime = live_factor[i] * q + live_factor[i+1] * p
                break

    return livetime

def isLampOn(xi, eta, dq, info, xtractab, shift2=0.):
    """Test whether a lamp was on.

    This function returns True if a wavecal lamp was on, i.e. if the
    counts through the wavecal aperture were significantly greater than
    the background counts.

    Parameters
    ----------
    xi: array_like
        Pixel coordinates of events, in dispersion direction.

    eta: array_like
        Pixel coordinates of events, in cross-dispersion direction.

    dq: array_like
        Data quality column.

    info: dictionary
        Header keywords and values.

    xtractab: str
        Name of the 1-D extraction parameters table.

    shift2: float
        Offset of spectrum in cross-dispersion direction.

    Returns
    -------
    boolean
        True if the background-subtracted wavecal source spectrum is
        more than five times the standard deviation of the difference
        between the source counts and the background counts.
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
        source = np.zeros((height, len_spectrum), dtype=np.float64)
        background1 = np.zeros((b_hgt, len_spectrum), dtype=np.float64)
        background2 = np.zeros((b_hgt, len_spectrum), dtype=np.float64)
        ccos.xy_extract(xi, eta, source, slope, b_spec, x_offset,
                        dq, info["sdqflags"])
        ccos.xy_extract(xi, eta, background1, slope, b_bkg1, x_offset,
                        dq, info["sdqflags"])
        ccos.xy_extract(xi, eta, background2, slope, b_bkg2, x_offset,
                        dq, info["sdqflags"])
        ns = source.sum(dtype=np.float64)
        nb = background1.sum(dtype=np.float64) + \
             background2.sum(dtype=np.float64)
        sigma_s = math.sqrt(ns)
        sigma_b = math.sqrt(nb)
        printMsg("Counts from lamp = %.0f, background = %.1f, " \
                 "stddev of difference = %.2f" % \
                 (ns, nb, math.sqrt(sigma_s**2 + sigma_b**2)),
                 level=VERY_VERBOSE)
        sigma_s_b = math.sqrt(sigma_s**2 + sigma_b**2)
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
    xtract_info = getTable(xtractab, filter)
    if xtract_info is None:
        printWarning("(isLampOn) matching row not found in xtractab %s" \
                     % xtractab)
        printContinuation("filter = %s" % str(filter))
        return False

    slope  = xtract_info.field("slope")[0]
    b_bkg1 = xtract_info.field("b_bkg1")[0] + shift2
    b_bkg2 = xtract_info.field("b_bkg2")[0] + shift2
    if findColumn(xtract_info, "b_hgt1"):
        bkg_height1 = xtract_info.field("b_hgt1")[0]
        bkg_height2 = xtract_info.field("b_hgt2")[0]
    else:
        bkg_height1 = xtract_info.field("bheight")[0]
        bkg_height2 = bkg_height1
    background1 = np.zeros((bkg_height1, len_spectrum), dtype=np.float64)
    background2 = np.zeros((bkg_height2, len_spectrum), dtype=np.float64)
    ccos.xy_extract(xi, eta, background1, slope, b_bkg1, x_offset,
                    dq, info["sdqflags"])
    ccos.xy_extract(xi, eta, background2, slope, b_bkg2, x_offset,
                    dq, info["sdqflags"])
    # number of background counts
    unscaled_nb = background1.sum(dtype=np.float64) + \
                  background2.sum(dtype=np.float64)
    sum_bkg_height = bkg_height1 + bkg_height2
    del background1, background2

    # Get the source counts.
    ns = 0.                     # number of source counts (incremented in loop)
    sum_height = 0
    for segment in segment_list:
        filter["segment"] = segment
        xtract_info = getTable(xtractab, filter, exactly_one=True)
        slope  = xtract_info.field("slope")[0]
        b_spec = xtract_info.field("b_spec")[0] + shift2
        height = xtract_info.field("height")[0]
        source = np.zeros((height, len_spectrum), dtype=np.float64)
        ccos.xy_extract(xi, eta, source, slope, b_spec, x_offset,
                        dq, info["sdqflags"])
        ns += source.sum(dtype=np.float64)
        sum_height += height
        del source

    # The heights of the source and background regions differ, so the
    # background counts will be multiplied by this factor.
    normalization = float(sum_height) / float(sum_bkg_height)
    nb = float(unscaled_nb) * normalization
    sigma_s = math.sqrt(ns)
    sigma_b = normalization * math.sqrt(unscaled_nb)

    printMsg("Counts in wavecal = %.0f, background = %.1f, " \
             "stddev of difference = %.2f" % \
             (ns, nb, math.sqrt(sigma_s**2 + sigma_b**2)),
             level=VERY_VERBOSE)

    sigma_s_b = math.sqrt(sigma_s**2 + sigma_b**2)
    if sigma_s_b > 0.:
        signal_to_noise = (ns - nb) / sigma_s_b
    else:
        signal_to_noise = 0.

    if signal_to_noise > 5.:
        return True
    else:
        return False

def getHeaders(input):
    """Return a list of all the headers in the file.

    Parameters
    ----------
    input: str
        Name of an input file.

    Returns
    -------
    list of pyfits Header objects
        A list of all the headers in the input FITS file.
    """

    fd = fits.open(input, mode="copyonwrite")

    headers = [hdu.header.copy() for hdu in fd]

    fd.close()

    return headers

def timeAtMidpoint(info):
    """Return the time (MJD) at the midpoint of an exposure.

    Parameters
    ----------
    info: dictionary
        Header keywords and values.

    Returns
    -------
    float
        average of expstart and expend
    """

    return (info["expstart"] + info["expend"]) / 2.

def timelineTimes(first_time, last_time, dt=1.):
    """Create an array of times.

    Parameters
    ----------
    first_time: float or None
        The time of the first event.  If this is None, the array that is
        returned will have length 1, and the value will be 0.

    last_time: float
        The time of the last event.

    dt: float
        The time interval for the output array of times.

    Returns
    -------
    array_like
        Array of uniformly spaced times, in seconds, with zero point
        EXPSTART.  The data type is float32.
    """

    if first_time is None or (last_time - first_time) <= 0.:
        tl_time = np.arange(1, dtype=np.float32)
    else:
        # add one so every event will be within the array of times
        nelem = int(round((last_time - first_time) / dt)) + 1
        tl_time = first_time + dt * np.arange(nelem, dtype=np.float32)

    return tl_time

def geometricDistortion(x, y, geofile, segment, igeocorr):
    """Apply geometric (INL) correction.

    Parameters
    ----------
    x: array_like
        Array of X pixel coordinates of events.

    y: array_like
        Array of Y pixel coordinates of events.

    geofile: str
        Name of geometric correction reference file.

    segment: {"FUVA", "FUVB"}
        FUV segment name.

    igeocorr: str
        "PERFORM" if interpolation should be used within the geofile.
    """

    fd = fits.open(geofile, mode="copyonwrite")
    x_hdu = fd[(segment,1)]
    y_hdu = fd[(segment,2)]

    origin_x = x_hdu.header.get("origin_x", 0)
    origin_y = x_hdu.header.get("origin_y", 0)

    if origin_x != y_hdu.header.get("origin_x", 0) or \
       origin_y != y_hdu.header.get("origin_y", 0):
        raise RuntimeError("Inconsistent ORIGIN_X or _Y keywords in GEOFILE")

    xbin = x_hdu.header.get("xbin", 1)
    ybin = x_hdu.header.get("ybin", 1)
    if xbin != y_hdu.header.get("xbin", 1) or \
       ybin != y_hdu.header.get("ybin", 1):
        raise RuntimeError("Inconsistent XBIN or YBIN keywords in GEOFILE")

    interp_flag = (igeocorr == "PERFORM")
    ccos.geocorrection(x, y, x_hdu.data, y_hdu.data, interp_flag,
                       origin_x, origin_y, xbin, ybin)

    fd.close()

def activeArea(segment, brftab):
    """Return the limits of the FUV active area.

    Parameters
    ----------
    segment: {"FUVA", "FUVB", "N/A"}
        Segment name, for finding the appropriate row in the brftab.

    brftab: str
        Name of the baseline reference frame table (ignored for NUV).

    Returns
    -------
    tuple of int
        The low and high limits and the left and right limits of the
        active area of the detector.  For NUV this will be the full
        detector size, (0, 1023, 0, 1023).
    """

    if segment[0] == "N":
        return (0, NUV_Y-1, 0, NUV_X-1)

    brf_info = getTable(brftab, {"segment": segment}, exactly_one=True)

    a_low = brf_info.field("a_low")[0]
    a_high = brf_info.field("a_high")[0]
    a_left = brf_info.field("a_left")[0]
    a_right = brf_info.field("a_right")[0]

    return (a_low, a_high, a_left, a_right)

def getInputDQ(input, imset=1):
    """Return the data quality array, or an array of zeros.

    If the data quality extension (EXTNAME = "DQ", EXTVER = imset) actually
    has a non-null data portion, that data array will be returned.  If the
    data portion is null (NAXIS = 0), a constant array will be returned;
    in this case the size will be taken from keywords NPIX1 and NPIX2, and
    the data value will be the value of the PIXVALUE keyword.

    Parameters
    ----------
    input: str
        Name of a FITS file containing an image set (SCI, ERR, DQ);
        only the DQ extension will be read.

    imset: int
        Image set number (one indexed).

    Returns
    -------
    array_like
        Data quality array read from input file, or array of zeros.
    """

    fd = fits.open(input, mode="copyonwrite")

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
            dq_array = np.bitwise_and(dq_array, 16383-(64+128))
        else:
            dq_array = np.zeros(npix, dtype=np.int16)
            dq_array[:,x_offset:len_raw+x_offset] = fd[("DQ",imset)].data
    else:
        dq_array = np.zeros(npix, dtype=np.int16)
        if "pixvalue" in hdr:
            pixvalue = hdr["pixvalue"]
            if pixvalue != 0:
                dq_array[:,:] = pixvalue

    fd.close()

    return dq_array

def checkSpottabKeywords(reffiles, info):
    """Check the SPOTTAB keywords against the PHA_LOW and PHA_HI values
    from the PHATAB.  If they aren't the same, print a warning.

    Parameters
    ----------
    reffiles: dictionary
        Dictionary of reference files
    """
    
    if reffiles["phafile"] == NOT_APPLICABLE:
        #
        # Skip the check if a PHAFILE is used, the PHA_LOW and PHA_HI
        # aren't fixed in that case so the comparison makes no sense
        phatab = reffiles["phatab"]
        segment = info["segment"]
        filter = {"segment": segment}
        if findColumn(phatab, "opt_elem"):
            filter["opt_elem"] = info["opt_elem"]
        pha_info = getTable(phatab, filter, exactly_one=True)

        low = pha_info.field("llt")[0]
        high = pha_info.field("ult")[0]
        spothdr = fits.open(reffiles["spottab"])[0].header
        try:
            spotlo = spothdr["phamin"]
            spothi = spothdr["phamax"]
            if spotlo != low and spothi != high:
                printWarning("PHAMIN and PHAMAX in SPOTTAB are different")
                printContinuation("from values in PHATAB")
                printContinuation("SPOTTAB: %d %d" % (spotlo, spothi))
                printContinuation("PHATAB: %d %d" % (low, high))
        except KeyError:
            printWarning("PHAMIN and PHAMAX keywords not found in SPOTTAB")
    return

def minmaxDoppler(info, doppcorr, doppmag, doppzero, orbitper):
    """Compute the range of Doppler shifts.

    Parameters
    ----------
    info: dictionary
        Header keywords and values.

    doppcorr: str
        Calibration switch from header; if doppcorr is "PERFORM", the DQ
        positions will be shifted to track the Doppler shift during the
        exposure.

    doppmag: float
        Magnitude (pixels) of the Doppler shift.

    doppzero: float
        Time (MJD) when the Doppler shift is zero and increasing.

    orbitper: float
        Orbital period (s) of HST.

    Returns
    -------
    tuple of two floats
        Minimum and maximum Doppler shifts (will be 0 if doppcorr is omit).
        The sign is such that the Doppler shift should be subtracted from
        XCORR to get XDOPP (the Doppler corrected X pixel coordinate).
    """

    if doppcorr == "PERFORM" or doppcorr == "COMPLETE":
        expstart = info["expstart"]
        exptime  = info["exptime"]

        # time is the time in seconds since doppzero.
        nelem = int(round(exptime))             # one element per sec
        nelem = max(nelem, 1)
        time = np.arange(nelem, dtype=np.float64) + \
                         (expstart - doppzero) * SEC_PER_DAY

        # shift is in pixels (wavelengths increase toward larger pixel number).
        shift = doppmag * np.sin(2. * np.pi * time / orbitper)
        mindopp = shift.min()
        maxdopp = shift.max()
    else:
        mindopp = 0.
        maxdopp = 0.

    return (mindopp, maxdopp)

def updateDQArray(info, reffiles, dq_array,
                  minmax_shift_dict,
                  minmax_doppler, doppler_boundary, gti):
    """Apply the data quality initialization table to DQ array.

    dq_array is a 2-D array, to be written as the DQ extension in an
    ACCUM file (_counts or _flt).  Its contents are assumed to be valid
    on input, since it may have been read from the raw file (if the
    input was an ACCUM image), and it may therefore include flagged
    pixels.  The flag information in the bpixtab will be combined
    (in-place) with dq_array using bitwise OR.

    Parameters
    ----------
    info: dictionary
        Header keywords and values.

    reffiles: dictionary
        Reference file keywords and names.

    dq_array: array_like
        Data quality image array (modified in-place)

    minmax_shift_dict: dictionary
        The min and max offsets in the dispersion direction and the min and
        max offsets in the cross-dispersion direction during the exposure

    minmax_doppler: tuple of two floats
        Minimum and maximum Doppler shifts (will be 0 if doppcorr is omit)

    doppler_boundary: int
        The border between PSA and WCA regions:
            Y < doppler_boundary is the PSA,
            Y >= doppler_boundary is the WCA
    """

    (lx, ly, dx, dy, dq, extn, message) = getDQArrays(info, reffiles, gti)
    if len(lx) < 1:
        return

    dq_shape = dq_array.shape

    # Upper limits (inclusive) of regions to flag.
    ux = lx + dx - 1
    uy = ly + dy - 1

    # A comment on notation for lower and upper limits of regions to flag:
    # suffix _s means shifted, i.e. wavecal and/or Doppler shift
    # suffix _s_s means shifted and relative to the lower limit of a slice

    (mindopp, maxdopp) = minmax_doppler
    if doppler_boundary > 0. and info["detector"] == "FUV":
        # split FUV into two regions, the PSA and the WCA
        key = list(minmax_shift_dict.keys())[0]
        value = minmax_shift_dict[key]
        (lower_y, upper_y) = key
        minmax_dict = {(lower_y, doppler_boundary): value,
                       (doppler_boundary, upper_y): value}
    else:
        minmax_dict = minmax_shift_dict
    fd = fits.open(reffiles["bpixtab"])
    widen = fd[1].header.get("widen", default=PIXEL_FRACTION)
    fd.close()
    # Update the 2-D data quality extension array from the DQI table info
    # for each slice in the minmax_shift_dict.
    # The flagged region (each row in the DQI table) will be expanded:
    #   the maximum shift will be subtracted from the lower limit, and
    #   the minimum shift will be subtracted from the upper limit;
    #   widen will be subtracted from the lower limit and added to
    #   the upper limit.
    # It is explicitly assumed here that the slice is only in the cross-
    # dispersion direction.
    keys = sorted(minmax_dict)
    for key in keys:
        (lower_y, upper_y) = key
        [min_shift1, max_shift1, min_shift2, max_shift2] = \
                minmax_dict[key]

        if doppler_boundary > 0. and \
           ((lower_y + upper_y) // 2 < doppler_boundary):
            lx_s = lx - (max_shift1 + maxdopp) - widen
            ux_s = ux - (min_shift1 + mindopp) + widen
        else:
            lx_s = lx - max_shift1 - widen
            ux_s = ux - min_shift1 + widen
        lx_s = lx_s.round().astype(np.int32)
        ux_s = ux_s.round().astype(np.int32)

        # Correct the Y limits of the regions for the wavecal shift.
        ly_s = (ly.astype(np.float64) - max_shift2 - widen).round()
        uy_s = (uy.astype(np.float64) - min_shift2 + widen).round()

        # These are the limits of a slice, corrected for the wavecal shift.
        lower_y_s = int(round(lower_y - max_shift2))
        upper_y_s = int(round(upper_y - min_shift2))
        if upper_y_s < 0:
            continue
        if lower_y_s >= dq_shape[0]:
            continue
        if lower_y_s < 0.:
            lower_y_s = 0
        if upper_y_s > dq_shape[0]:
            upper_y_s = dq_shape[0]

        # These are the shifted Y region limits relative to the slice.
        ly_s_slice = (ly_s - lower_y_s).astype(np.int32)
        uy_s_slice = (uy_s - lower_y_s).astype(np.int32)

        ccos.bindq(lx_s, ly_s_slice, ux_s, uy_s_slice, dq,
                   dq_array[int(lower_y_s):int(upper_y_s),:], info["x_offset"])

def getDQArrays(info, reffiles, gti):
    """Get DQ info from BPIXTAB and possibly GSAGTAB and SPOTTAB.

    Parameters
    ----------
    info: dictionary
        Header keywords and values.

    reffiles: dictionary
        Reference file keywords and names.

    Returns
    -------
    tuple of five arrays, an int, and a string
        (lx, ly, dx, dy, dq, extn, message)
        lx and ly are arrays (np.int16) of the lower-left corners of
            flagged regions (ly is along axis 0, lx is along axis 1),
        dx and dy are arrays (np.int16) of the width and height of flagged
            regions (dy is along axis 0, dx is along axis 1),
        dq is an array (np.int16) of the data quality value for each
            region,
        extn is the integer extension number from which the gain sag table
            information was read (or None if there is no gain sag table,
            or -1 if no matching extension was found)
        message is a string, which, if not empty, gives a warning message
    """

    # These defaults indicate there is no gain sag table (e.g. for NUV).
    lx = []; ly = []; dx = []; dy = []; dq = []; extn = None; message = ""

    bpixtab = reffiles["bpixtab"]
    dq_info = getTable(bpixtab, filter={"segment": info["segment"]})
    if dq_info is None:
        return (lx, ly, dx, dy, dq, extn, message)

    lx = dq_info.field("lx")
    ly = dq_info.field("ly")
    dx = dq_info.field("dx")
    dy = dq_info.field("dy")
    dq = dq_info.field("dq")

    if info["detector"] == "FUV":
        gsagtab = reffiles["gsagtab"]
        if gsagtab != NOT_APPLICABLE:
            (extn, message) = findGSagExtn(gsagtab, info["hvlevel"],
                                           info["segment"])
            if extn > 0:        # was an appropriate extension found?
                gsag_info = getTable(gsagtab,
                        filter={"date": (np.less_equal, info["expstart"])},
                        extension=extn)
                if gsag_info is not None:
                    gsag_lx = gsag_info.field("lx")
                    gsag_ly = gsag_info.field("ly")
                    gsag_dx = gsag_info.field("dx")
                    gsag_dy = gsag_info.field("dy")
                    gsag_dq = gsag_info.field("dq")

                    lx = concatArrays(lx, gsag_lx)
                    ly = concatArrays(ly, gsag_ly)
                    dx = concatArrays(dx, gsag_dx)
                    dy = concatArrays(dy, gsag_dy)
                    dq = concatArrays(dq, gsag_dq)
        #
        # SPOTTAB processing is similar to GSAGTAB processing
        spottab = reffiles["spottab"]
        if spottab != NOT_APPLICABLE:
            #
            # Loop over good time intervals.  For each interval
            # hotspot overlaps the interval if the start of the spot is
            # before the end of the interval, and the end of the spot
            # is after the beginning of the interval
            # If there's no good time interval list, make one from exptime
            # Remember the times in to good time interval are in seconds since the
            # beginning of the exposure,  whereas the times in the SPOTTAB are in MJD
            if gti is None:
                gticopy = [[0.0, info["exptime"]]]
            else:
                gticopy = copy.deepcopy(gti)
            for gtstart, gtstop in gticopy:
                gti_start_mjd = info["expstart"] + gtstart / 86400.0
                gti_stop_mjd = info["expstart"] + gtstop / 86400.0
                spotfilter = {"segment": info["segment"],
                              "start": (np.less_equal, gti_stop_mjd),
                              "stop": (np.greater_equal, gti_start_mjd)}
                spot_info = getTable(spottab, filter=spotfilter)
                if spot_info is not None:
                    spot_lx = spot_info.field("lx")
                    spot_ly = spot_info.field("ly")
                    spot_dx = spot_info.field("dx")
                    spot_dy = spot_info.field("dy")
                    spot_dq = spot_info.field("dq")
                    
                    lx = concatArrays(lx, spot_lx)
                    ly = concatArrays(ly, spot_ly)
                    dx = concatArrays(dx, spot_dx)
                    dy = concatArrays(dy, spot_dy)
                    dq = concatArrays(dq, spot_dq)
                    # If we find a match for this start/stop time, we don't need to
                    # check any other start/stop time combinations
                    break

    return (lx, ly, dx, dy, dq, extn, message)

def concatArrays(x0, x1):
    """Concatenate two arrays.

    Parameters
    ----------
    x0: array_like
        First array, to be copied to the first part of the output array.

    x1: array_like
        Second array, to be copied to the second part of the output array.

    Returns
    -------
    array
        This is the result of appending x1 to x0.
    """

    length0 = len(x0)
    length1 = len(x1)
    length = length0 + length1
    dtype = x0.dtype
    x = np.zeros(length, dtype=dtype)
    x[0:length0] = x0
    x[length0:length] = x1

    return x

def findGSagExtn(gsagtab, hvlevel, segment):
    """Find the appropriate extension in the gain sag table.

    Parameters
    ----------
    gsagtab: string
        Name of the gain sag table, i.e. listing gain-sagged regions.

    hvlevel: int
        Raw (commanded) FUV detector high voltage level that was used for
        the current exposure.

    segment: string
        Segment name (FUVA or FUVB)

    Returns
    -------
    tuple (extn, message) of an int and a string
        extn is the integer extension number in the gain sag table that
            matches the specified segment name and high voltage.  If there
            is no extension with an exact match to the current high
            voltage, the extension with the closest voltage value greater
            than the current high voltage will be selected.  If no such
            extension was found (e.g. no extension for the current segment,
            or voltages in all extensions were lower than the current high
            voltage), then extn will be set to -1.
        message is a string, either empty or giving a warning message
            regarding hvlevel compared with the nominal value in the
            matching extension.

    """

    kwd_root = "hvlevel"        # high voltage (commanded, raw)
    keyword = segmentSpecificKeyword(kwd_root, segment)

    fd = fits.open(gsagtab)
    extn = -1           # default indicates that no extension matched the HV
    message = ""
    extn_hv_min = None
    hv_diff = None
    for i in range(1, len(fd)):
        hdr = fd[i].header
        segment_i = hdr.get("segment", default="missing")
        if segment_i != segment:
            continue
        hv = hdr.get(keyword, default=-999)
        if hv == hvlevel:
            extn = i
            break
        elif hv > hvlevel:
            # Find the closest voltage value greater than the current
            # commanded value.
            if hv_diff is None or (hv - hvlevel) < hv_diff:
                hv_diff = hv - hvlevel
                extn_hv_min = i

    if extn < 0:
        message = "No matching extension was found in the gain sag table " \
        "for SEGMENT=%s, %s=%d" % (segment, keyword.upper(), hvlevel)
        if extn_hv_min is not None:
            extn = extn_hv_min
            message = message + ";\n  using extension %d instead" % extn

    fd.close()

    return (extn, message)

def flagOutOfBounds(hdr, dq_array, info, switches,
                     brftab, geofile, dgeofile, minmax_shift_dict,
                     minmax_doppler, doppler_boundary):
    """Flag regions that are outside all subarrays (done in-place).

    Parameters
    ----------

    hdr: pyfits Header object
        the EVENTS or SCI extension header

    dq_array: array_like
        Data quality image array (modified in-place)

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switch keywords and values.

    brftab: str
        Name of baseline reference table (for active area)

    geofile: str
        Name of geometric correction reference file

    minmax_shift_dict: dictionary
        The min and max offsets in the dispersion direction and the min and
        max offsets in the cross-dispersion direction during the exposure

    minmax_doppler: tuple of two floats
        Minimum and maximum Doppler shifts (will be 0 if doppcorr is omit)

    doppler_boundary: int
        The border between PSA and WCA regions:
            Y < doppler_boundary is the PSA,
            Y >= doppler_boundary is the WCA
    """

    if info["detector"] == "FUV":
        fuvFlagOutOfBounds(hdr, dq_array, info, switches,
                           brftab, geofile, dgeofile,
                           minmax_shift_dict, minmax_doppler)
    else:
        nuvFlagOutOfBounds(hdr, dq_array, info, switches,
                           minmax_shift_dict,
                           minmax_doppler, doppler_boundary)

def fuvFlagOutOfBounds(hdr, dq_array, info, switches,
                       brftab, geofile, dgeofile, 
                       minmax_shift_dict, minmax_doppler):
    """In FUV data, flag regions that are outside all subarrays (in-place).

    Parameters
    ----------

    hdr: pyfits Header object
        the EVENTS or SCI extension header

    dq_array: array_like
        Data quality image array (modified in-place)

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switch keywords and values.

    brftab: str
        Name of baseline reference table (for active area)

    geofile: str
        Name of geometric correction reference file

    minmax_shift_dict: dictionary
        The min and max offsets in the dispersion direction and the min and
        max offsets in the cross-dispersion direction during the exposure

    minmax_doppler: tuple of two floats
        Minimum and maximum Doppler shifts (will be 0 if doppcorr is omit)
    """

    nsubarrays = info["nsubarry"]
    x_offset = info["x_offset"]
    segment = info["segment"]

    # Indices 0, 1, 2, 3 are for FUVA, while 4, 5, 6, 7 are for FUVB.
    indices = np.arange(4, dtype=np.int32)
    if segment == "FUVB":
        indices += 4

    temp = dq_array.copy()
    (ny, nx) = dq_array.shape

    # there's only one key for FUV
    key = list(minmax_shift_dict.keys())[0]

    # These are for shifting the out-of-bounds region into the subarray
    # due to the wavecal offset and Doppler shift during the exposure.
    [min_shift1, max_shift1, min_shift2, max_shift2] = minmax_shift_dict[key]
    (mindopp, maxdopp) = minmax_doppler
    dx = min_shift1
    dy = min_shift2
    dx += mindopp
    dx = int(round(dx))
    dy = int(round(dy))
    xwidth = int(round(max_shift1 - min_shift1 + maxdopp - mindopp))
    ywidth = int(round(max_shift2 - min_shift2))

    # get a list of subarray locations
    subarrays = []
    for i in indices:
        sub = {}
        sub_number = str(i)
        # these keywords are 0-indexed
        x0 = hdr["corner"+sub_number+"x"]
        y0 = hdr["corner"+sub_number+"y"]
        xsize = hdr["size"+sub_number+"x"]
        ysize = hdr["size"+sub_number+"y"]
        if xsize <= 0 or ysize <= 0:
            continue
        if (ysize, xsize) == (FUV_Y, FUV_X):
            continue
        x1 = x0 + xsize - xwidth
        y1 = y0 + ysize - ywidth
        sub["x0"] = x0
        sub["y0"] = y0
        sub["x1"] = x1
        sub["y1"] = y1
        subarrays.append(sub)
    if not subarrays:
        # Create one full-size "subarray" in order to account for the NUV
        # image being larger than the detector and because of fpoffset.
        sub = {}
        x0 = 0
        y0 = 0
        xsize = FUV_X
        ysize = FUV_Y
        x1 = x0 + xsize - xwidth
        y1 = y0 + ysize - ywidth
        sub["x0"] = x0
        sub["y0"] = x0
        sub["x1"] = x1
        sub["y1"] = y1
        subarrays.append(sub)

    # Initially flag the entire image as out of bounds, then remove the
    # flag (set it to zero) for each subarray.
    temp[:,:] = DQ_PIXEL_OUT_OF_BOUNDS
    (ny, nx) = dq_array.shape

    # The test on COMPLETE is for corrtag input.
    if switches["tempcorr"] == "PERFORM" or switches["tempcorr"] == "COMPLETE":

        # Get the parameters found by computeThermalParam.
        seg = segment[-1]           # "A" or "B"
        # reference positions
        sx1r = hdr.get("STIM"+seg+"0LX", -1.)
        sy1r = hdr.get("STIM"+seg+"0LY", -1.)
        sx2r = hdr.get("STIM"+seg+"0RX", -1.)
        sy2r = hdr.get("STIM"+seg+"0RY", -1.)
        # measured positions of the stims
        sx1 = hdr.get("STIM"+seg+"_LX", sx1r)
        sy1 = hdr.get("STIM"+seg+"_LY", sy1r)
        sx2 = hdr.get("STIM"+seg+"_RX", sx2r)
        sy2 = hdr.get("STIM"+seg+"_RY", sy2r)
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
            new_subarrays.append(sub)
        del subarrays
        subarrays = new_subarrays

    # Add shifts, apply geometric correction to the subarray for the
    # source spectrum, and set flags to zero in temp within subarrays.
    (b_low, b_high, b_left, b_right) = activeArea(segment, brftab)
    nfound = 0
    save_sub = None
    for sub in subarrays:
        x0 = sub["x0"]
        x1 = sub["x1"]
        y0 = sub["y0"]
        y1 = sub["y1"]
        # the subarrays for the stims are outside the active area
        if y1 < b_low or y0 > b_high:
            clearSubarray(temp, x0, x1, y0, y1, dx, dy, x_offset)
            continue
        nfound += 1
        # These are arrays of pixel coordinates just inside the borders
        # of the subarray.
        x_lower = np.arange(x0, x1, dtype=np.float32)
        x_upper = np.arange(x0, x1, dtype=np.float32)
        y_left  = np.arange(y0, y1, dtype=np.float32)
        y_right = np.arange(y0, y1, dtype=np.float32)
        y_lower = y0 + 0. * x_lower
        y_upper = (y1 - 1.) + 0. * x_upper
        x_left  = x0 + 0. * y_left
        x_right = (x1 - 1.) + 0. * y_right
        # These are independent variable arrays for interpolation.
        x_lower_uniform = np.arange(nx, dtype=np.float32)
        x_upper_uniform = np.arange(nx, dtype=np.float32)
        y_left_uniform  = np.arange(ny, dtype=np.float32)
        y_right_uniform = np.arange(ny, dtype=np.float32)
        # These will be the arrays of interpolated edge coordinates.
        y_lower_interp = np.arange(nx, dtype=np.float32)
        y_upper_interp = np.arange(nx, dtype=np.float32)
        x_left_interp  = np.arange(ny, dtype=np.float32)
        x_right_interp = np.arange(ny, dtype=np.float32)
        save_sub = (x0, x1, y0, y1)             # in case geocorr is omit
    if nfound == 0:
        printWarning("in fuvFlagOutOfBounds,"
                     " there should be at least one full-size 'subarray'")
    if nfound > 1:
        printWarning("in fuvFlagOutOfBounds, more subarrays than expected")
    # The test on COMPLETE is for corrtag input.
    if switches["geocorr"] == "PERFORM" or switches["geocorr"] == "COMPLETE":
        interp_flag = (switches["igeocorr"] == "PERFORM")
        (x_data, origin_x, xbin, y_data, origin_y, ybin) = \
                        getGeoData(geofile, segment)
        # Undistort x_lower, y_lower, etc., in-place.
        ccos.geocorrection(x_lower, y_lower, x_data, y_data, interp_flag,
                           origin_x, origin_y, xbin, ybin)
        ccos.geocorrection(x_upper, y_upper, x_data, y_data, interp_flag,
                           origin_x, origin_y, xbin, ybin)
        ccos.geocorrection(x_left, y_left, x_data, y_data, interp_flag,
                           origin_x, origin_y, xbin, ybin)
        ccos.geocorrection(x_right, y_right, x_data, y_data, interp_flag,
                           origin_x, origin_y, xbin, ybin)
        del(x_data, y_data)
        if switches["dgeocorr"] == "PERFORM" or switches["dgeocorr"] == "COMPLETE":
            interp_flag = (switches["igeocorr"] == "PERFORM")
            (x_data, origin_x, xbin, y_data, origin_y, ybin) = \
                getGeoData(dgeofile, segment)
            # Undistort x_lower, y_lower, etc., in-place.
            ccos.geocorrection(x_lower, y_lower, x_data, y_data, interp_flag,
                               origin_x, origin_y, xbin, ybin)
            ccos.geocorrection(x_upper, y_upper, x_data, y_data, interp_flag,
                               origin_x, origin_y, xbin, ybin)
            ccos.geocorrection(x_left, y_left, x_data, y_data, interp_flag,
                               origin_x, origin_y, xbin, ybin)
            ccos.geocorrection(x_right, y_right, x_data, y_data, interp_flag,
                               origin_x, origin_y, xbin, ybin)
            del(x_data, y_data)
        # Interpolate to uniform spacing (pixel spacing).
        ccos.interp1d(x_lower, y_lower, x_lower_uniform, y_lower_interp)
        ccos.interp1d(x_upper, y_upper, x_upper_uniform, y_upper_interp)
        ccos.interp1d(y_left,  x_left,  y_left_uniform,  x_left_interp)
        ccos.interp1d(y_right, x_right, y_right_uniform, x_right_interp)
        # Apply offsets for zero point and wavecal shifts, replacing the
        # previous x_lower, y_lower, etc.  The independent variable arrays
        # will now be uniform, and the dependent variable arrays will have
        # been interpolated onto the uniform grid.
        (y_lower, y_upper) = applyOffsets(y_lower_interp, y_upper_interp,
                                          ny, dy)
        (x_left, x_right)  = applyOffsets(x_left_interp, x_right_interp,
                                          nx, dx, x_offset)

        ccos.clear_rows(temp, y_lower, y_upper, x_left, x_right)
    elif save_sub is not None:
        (x0, x1, y0, y1) = save_sub
        clearSubarray(temp, x0, x1, y0, y1, dx, dy, x_offset)

    dq_array[:,:] = np.bitwise_or(dq_array, temp)

def applyOffsets(x_left, x_right, nx, dx, x_offset=0):

    x_left += x_offset
    x_right += x_offset
    x_left -= dx
    x_right -= dx
    x_left = np.where(x_left < 0., 0., x_left)
    x_right = np.where(x_right > nx-1., nx-1., x_right)

    return (x_left, x_right)

def clearSubarray(temp, x0, x1, y0, y1, dx, dy, x_offset):
    """Set the subarray to zero in temp."""

    (ny, nx) = temp.shape
    x0 += x_offset
    x0 -= dx
    y0 -= dy
    x1 += x_offset
    x1 -= dx
    y1 -= dy
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, nx)
    y1 = min(y1, ny)
    temp[int(y0):int(y1),int(x0):int(x1)] = DQ_OK

def nuvFlagOutOfBounds(hdr, dq_array, info, switches,
                       minmax_shift_dict, minmax_doppler, doppler_boundary):
    """In NUV data, flag regions that are outside all subarrays (in-place).

    Parameters
    ----------

    hdr: pyfits Header object
        the EVENTS or SCI extension header

    dq_array: array_like
        Data quality image array (modified in-place)

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switch keywords and values.

    minmax_shift_dict: dictionary
        The min and max offsets in the dispersion direction and the min and
        max offsets in the cross-dispersion direction during the exposure

    minmax_doppler: tuple of two floats
        Minimum and maximum Doppler shifts (will be 0 if doppcorr is omit)

    doppler_boundary: int
        The border between PSA and WCA regions:
            Y < doppler_boundary is the PSA,
            Y >= doppler_boundary is the WCA
    """

    nsubarrays = info["nsubarry"]
    x_offset = info["x_offset"]
    segment = info["segment"]

    indices = np.arange(nsubarrays, dtype=np.int32)

    # Initially flag the entire image as out of bounds, then remove the
    # flag (set it to zero) for each subarray.
    temp = dq_array.copy()
    temp[:,:] = DQ_PIXEL_OUT_OF_BOUNDS
    (ny, nx) = dq_array.shape

    (mindopp, maxdopp) = minmax_doppler

    for key in minmax_shift_dict.keys():

        (lower_y, upper_y) = key

        # The "good" region (not out of bounds) will be shrunk:
        #   the minimum shift will be subtracted from the lower border, and
        #   the maximum shift will be subtracted from the upper border.

        # These are for shifting the out-of-bounds region into the subarray
        # due to the wavecal offset and Doppler shift during the exposure.
        [min_shift1, max_shift1, min_shift2, max_shift2] = \
                minmax_shift_dict[key]

        # get a list of subarray locations
        subarrays = []
        for i in indices:
            sub = {}
            sub_number = str(i)
            # these keywords are 0-indexed
            x0 = hdr["corner"+sub_number+"x"]
            y0 = hdr["corner"+sub_number+"y"]
            xsize = hdr["size"+sub_number+"x"]
            ysize = hdr["size"+sub_number+"y"]
            x1 = x0 + xsize                     # modified below
            y1 = y0 + ysize                     # modified below
            if xsize <= 0 or ysize <= 0:
                continue
            y0 = max(y0, lower_y)
            y1 = min(y1, upper_y)
            if y0 >= y1:
                continue
            if (lower_y + upper_y) // 2 < doppler_boundary:
                x0 -= int(round(min_shift1 + mindopp))
                x1 -= int(round(max_shift1 + maxdopp))
            else:
                x0 -= int(round(min_shift1))
                x1 -= int(round(max_shift1))
            y0 -= int(round(min_shift2))
            y1 -= int(round(max_shift2))
            sub["x0"] = x0
            sub["y0"] = y0
            sub["x1"] = x1
            sub["y1"] = y1
            subarrays.append(sub)
        if not subarrays:
            # Create one full-size "subarray" in order to account for the NUV
            # image being larger than the detector and because of fpoffset.
            sub = {}
            x0 = 0
            x1 = x0 + NUV_X
            if (lower_y + upper_y) // 2 < doppler_boundary:
                x0 -= int(round(min_shift1 + mindopp))
                x1 -= int(round(max_shift1 + maxdopp))
            else:
                x0 -= int(round(min_shift1))
                x1 -= int(round(max_shift1))
            y0 = lower_y - int(round(min_shift2))
            y1 = upper_y - int(round(max_shift2))
            sub["x0"] = x0
            sub["y0"] = x0
            sub["x1"] = x1
            sub["y1"] = y1
            subarrays.append(sub)

        # Add shifts, and set flags to zero in temp within subarrays.
        for sub in subarrays:
            x0 = sub["x0"] + x_offset
            x1 = sub["x1"] + x_offset
            y0 = sub["y0"]
            y1 = sub["y1"]
            x0 = max(x0, 0)
            y0 = max(y0, 0)
            x1 = min(x1, nx)
            y1 = min(y1, ny)
            temp[int(y0):int(y1),int(x0):int(x1)] = DQ_OK

    dq_array[:,:] = np.bitwise_or(dq_array, temp)

def flagOutsideActiveArea(dq_array, segment, brftab, x_offset,
                          minmax_shift_dict, minmax_doppler):
    """Flag the region that is outside the active area.

    This is only relevant for FUV data.

    Parameters
    ----------

    dq_array: array_like
        Data quality image array (modified in-place)

    segment: str
        Segment name (FUVA or FUVB)

    brftab: str
        Name of baseline reference table (for active area)

    x_offset: int
        Offset of the detector in the image

    minmax_shift_dict: dictionary
        The min and max offsets in the dispersion direction and the min and
        max offsets in the cross-dispersion direction during the exposure

    minmax_doppler: tuple of two floats
        Minimum and maximum Doppler shifts (will be 0 if doppcorr is omit)
    """

    (b_low, b_high, b_left, b_right) = activeArea(segment, brftab)

    # These are for shifting and smearing the out-of-bounds region into
    # the active region due to the wavecal offset and Doppler shift and
    # their variation during the exposure.
    key = list(minmax_shift_dict.keys())[0]
    [min_shift1, max_shift1, min_shift2, max_shift2] = minmax_shift_dict[key]
    (mindopp, maxdopp) = minmax_doppler

    b_left -= int(round(min_shift1))
    b_right -= int(round(max_shift1))
    b_low -= int(round(min_shift2))
    b_high -= int(round(max_shift2))

    b_left -= int(round(mindopp))
    b_right -= int(round(maxdopp))

    b_left += x_offset
    b_right += x_offset

    (ny, nx) = dq_array.shape

    if b_low >= 0:
        dq_array[0:int(b_low),:]    |= DQ_PIXEL_OUT_OF_BOUNDS
    if b_high < ny-1:
        dq_array[int(b_high)+1:,:]  |= DQ_PIXEL_OUT_OF_BOUNDS
    if b_left >= 0:
        dq_array[:,0:int(b_left)]   |= DQ_PIXEL_OUT_OF_BOUNDS
    if b_right < nx-1:
        dq_array[:,int(b_right)+1:] |= DQ_PIXEL_OUT_OF_BOUNDS

def correctTraceAndAlignment(dq_array, info, traceprofile, shift1,
                             alignment_correction):
    """Correct the DQ array for the trace and alignment correction.
    The trace and alignment correction shifts every spectrum to a horizontal
    spectrum in (xfull, yfull) space.  The DQ arrays need to be shifted in the
    same way.

    Parameters
    ----------

    dq_array: array-like
        The input DQ array

    info: dictionary
        Keywords and values

    traceprofile: array-like
        Trace profile, a 1-d vector of delta YFULL values for each XCORR.  This
        was SUBTRACTED from the yfull values, so we need to ADD the NEGATIVE
        of this

    alignment_correction:
        The offset that was added to the YFULL values  to correct the alignment
        
    """
    nrows, ncols = dq_array.shape
    total_correction = None
    if alignment_correction is not None:
        total_correction = alignment_correction
        if traceprofile is not None:
            total_correction = -traceprofile + alignment_correction
    return

def getGeoData(geofile, segment):
    """Open and read the geofile.

    Parameters
    ----------
    geofile: str
        Name of geometric correction reference file

    segment: str
        Segment name (FUVA or FUVB)

    Returns
    -------
    tuple
        The data from the geofile for X and Y, and the offsets;
        x_hdu.data:  array to correct distortion in X
        origin_x:  offset of x_hdu.data within detector coordinates
        xbin:  binning (int) in the X direction
        y_hdu.data:  array to correct distortion in Y
        origin_y:  offset of y_hdu.data within detector coordinates
        ybin:  binning (int) in the Y direction
    """

    fd = fits.open(geofile, mode="copyonwrite")
    x_hdu = fd[(segment,1)]
    y_hdu = fd[(segment,2)]

    # The images in the geofile will typically be smaller than the full
    # detector.  These offsets give the location of geofile pixel [0,0]
    # on the detector.
    origin_x = x_hdu.header.get("origin_x", 0)
    origin_y = x_hdu.header.get("origin_y", 0)

    if origin_x != y_hdu.header.get("origin_x", 0) or \
       origin_y != y_hdu.header.get("origin_y", 0):
        raise RuntimeError("Inconsistent ORIGIN_X or _Y keywords in GEOFILE")

    xbin = x_hdu.header.get("xbin", 1)
    ybin = x_hdu.header.get("ybin", 1)
    if xbin != y_hdu.header.get("xbin", 1) or \
       ybin != y_hdu.header.get("ybin", 1):
        raise RuntimeError("Inconsistent XBIN or YBIN keywords in GEOFILE")

    # "touch" the data before closing the file.  Is this necessary?
    x_data = x_hdu.data
    y_data = y_hdu.data

    fd.close()

    return (x_data, origin_x, xbin, y_data, origin_y, ybin)

def tableHeaderToImage(thdr):
    """Rename table WCS keywords to image WCS keywords.

    The function returns a copy of the header with table-specific WCS
    keywords renamed to their image-style counterparts, to serve as an
    image header.

    Parameters
    ----------
    thdr: pyfits Header object
        A header for a BINTABLE extension.

    Returns
    -------
    hdr: pyfits Header object
        A copy of thdr, with certain table WCS keywords renamed or deleted.
    """

    hdr = thdr.copy()

    # These are the world coordinate system keywords in an events table
    # and their corresponding names for an image.  NOTE that this assumes
    # that the X and Y columns are 2 and 3 (one indexed).
    tkey = ["TCTYP2", "TCRVL2", "TCRPX2", "TCDLT2", "TCUNI2", "TC2_2", "TC2_3",
            "TCTYP3", "TCRVL3", "TCRPX3", "TCDLT3", "TCUNI3", "TC3_2", "TC3_3"]
    ikey = ["CTYPE1", "CRVAL1", "CRPIX1", "CDELT1", "CUNIT1", "CD1_1", "CD1_2",
            "CTYPE2", "CRVAL2", "CRPIX2", "CDELT2", "CUNIT2", "CD2_1", "CD2_2"]
    # Rename events table WCS keywords to the corresponding image WCS keywords.
    for i in range(len(tkey)):
        if tkey[i] in hdr:
            if ikey[i] in hdr:
                printWarning("Can't rename %s to %s" % (tkey[i], ikey[i]))
                printContinuation("keyword already exists")
                del(hdr[tkey[i]])
            else:
                hdr.rename_keyword(tkey[i], ikey[i])

    return hdr

def imageHeaderToCorrtag(imhdr):
    """Modify keywords to turn an image header into an EVENTS table header.

    The function returns a copy of the header with some image-specific
    keywords (e.g. world coordinate system keywords and BUNIT) either
    renamed or deleted.

    Parameters
    ----------
    imhdr: FITS Header object
        A header for a FITS IMAGE HDU.

    Returns
    -------
    hdr: FITS Header object
        A copy of imhdr, with certain image-specific keywords either
        renamed or deleted.
    """

    hdr = imhdr.copy()

    tkey = ["TCTYP2", "TCRVL2", "TCRPX2", "TCDLT2", "TCUNI2", "TC2_2", "TC2_3",
            "TCTYP3", "TCRVL3", "TCRPX3", "TCDLT3", "TCUNI3", "TC3_2", "TC3_3"]
    ikey = ["CTYPE1", "CRVAL1", "CRPIX1", "CDELT1", "CUNIT1", "CD1_1", "CD1_2",
            "CTYPE2", "CRVAL2", "CRPIX2", "CDELT2", "CUNIT2", "CD2_1", "CD2_2"]
    delkey = ["BSCALE", "BZERO", "BUNIT", "DATAMIN", "DATAMAX"]

    # Rename image WCS keywords to the corresponding events table WCS keywords.
    for i in range(len(ikey)):
        if ikey[i] in hdr:
            if tkey[i] in hdr:
                printWarning("Can't rename %s to %s" % (ikey[i], tkey[i]))
                printContinuation("keyword already exists")
                del(hdr[ikey[i]])
            else:
                hdr.rename_keyword(ikey[i], tkey[i])
    for keyword in delkey:
        if keyword in hdr:
            del hdr[keyword]

    return hdr

def imageHeaderToTable(imhdr):
    """Modify keywords to turn an image header into a table header.

    The function returns a copy of the header with some image-specific
    keywords (e.g. world coordinate system keywords and BUNIT) deleted.

    Parameters
    ----------
    imhdr: pyfits Header object
        A header for a FITS IMAGE HDU.

    Returns
    -------
    hdr: pyfits Header object
        A copy of imhdr, with certain image-specific keywords deleted.
    """

    hdr = imhdr.copy()

    for keyword in ["bscale", "bzero", "bunit", "datamin", "datamax"]:
        if keyword in hdr:
            del hdr[keyword]

    keyword_list = []
    for key in ["ctype", "crval", "crpix", "cdelt", "cunit"]:
        for dim in range(1, 4):
            keyword = key + str(dim)
            keyword_list.append(keyword)
    keyword_list.extend(["wcsaxes", "pv1_0", "pv1_1", "pv1_2", "pv1_6",
                         "cd1_1", "cd1_2", "cd2_1", "cd2_2",
                         "pc1_1", "pc1_2", "pc2_1", "pc2_2", "pc3_1", "pc3_2"])

    for alt in ["", "a", "b", "c"]:
        for key in keyword_list:
            keyword = key + alt
            if keyword in hdr:
                del hdr[keyword]

    return hdr

def delCorrtagWCS(thdr):
    """Delete table WCS keywords.

    The function returns a copy of the header with table-specific WCS keywords
    deleted.  This is appropriate when creating an x1d table from a corrtag
    table.

    Parameters
    ----------
    thdr: pyfits Header object
        A header for a BINTABLE extension.

    Returns
    -------
    hdr: pyfits Header object
        A copy of thdr, with certain table WCS keywords deleted.
    """

    hdr = thdr.copy()

    # These are the world coordinate system keywords in an events table.
    # NOTE that this assumes that the X and Y columns are 2 and 3
    # (one indexed).
    tkey = ["TCTYP2", "TCRVL2", "TCRPX2", "TCDLT2", "TCUNI2", "TC2_2", "TC2_3",
            "TCTYP3", "TCRVL3", "TCRPX3", "TCDLT3", "TCUNI3", "TC3_2", "TC3_3"]
    for keyword in tkey:
        if keyword in hdr:
            del hdr[keyword]

    return hdr

def updateFilename(phdr, filename):
    """Update the FILENAME keyword in a primary header.

    This routine will update (or add) the FILENAME keyword.  If filename
    includes a directory, that will not be included in the keyword value.

    Parameters
    ----------
    phdr: pyfits Header object
        A primary header; keyword FILENAME will be modified in-place.

    filename: str
        Name of file, possibly including directory.
    """

    phdr["filename"] = os.path.basename(filename)

def renameFile(infile, outfile):
    """Rename a FITS file, and update the FILENAME keyword.

    Parameters
    ----------
    infile: str
        Current name of a FITS file.

    outfile: str
        New name for the file.
    """

    printMsg("rename " + infile + " --> " + outfile, VERY_VERBOSE)

    os.rename(infile, outfile)

    fd = fits.open(outfile, mode="update")

    # If the output file name is a product name (ends with '0' before
    # the suffix), change the value of the extension keyword ASN_MTYP.
    if isProduct(outfile):
        asn_mtyp = fd[1].header.get("asn_mtyp", "missing")
        asn_mtyp = modifyAsnMtyp(asn_mtyp)
        if asn_mtyp != "missing":
            fd[1].header["asn_mtyp"] = asn_mtyp
    updateFilename(fd[0].header, outfile)

    fd.close()

def copyFile(infile, outfile):
    """Copy a FITS file, and update the FILENAME keyword.

    Parameters
    ----------
    infile: str
        Name of input FITS file.

    outfile: str
        Name of output FITS file.
    """

    printMsg("copy " + infile + " --> " + outfile, VERY_VERBOSE)

    shutil.copy(infile, outfile)

    fd = fits.open(outfile, mode="update")

    # If the output file name is a product name (ends with '0' before
    # the suffix), change the value of the extension keyword ASN_MTYP.
    if isProduct(outfile):
        asn_mtyp = fd[1].header.get("asn_mtyp", "missing")
        asn_mtyp = modifyAsnMtyp(asn_mtyp)
        if asn_mtyp != "missing":
            fd[1].header["asn_mtyp"] = asn_mtyp
    updateFilename(fd[0].header, outfile)

    fd.close()

def isProduct(filename):
    """Return True if 'filename' is a "product" name.

    Parameters
    ----------
    filename: str
        Name of an output file.

    Returns
    -------
    is_product: boolean
        True if the root part (before the suffix) of 'filename' ends in '0',
        implying that it is a product name.
    """

    is_product = False          # may be changed below
    i = filename.rfind("_")
    if i > 0 and filename[i:] == "_a.fits" or filename[i:] == "_b.fits":
        i = filename[0:i-1].rfind("_")
    if i > 0 and filename[i-1] == '0':
        is_product = True

    return is_product

def modifyAsnMtyp(asn_mtyp):
    """Replace 'EXP' with 'PROD' in the ASN_MTYP keyword string.

    Parameters
    ----------
    asn_mtyp: str
        Value of ASN_MTYP keyword from an input file.

    Returns
    -------
    asn_mtyp: str
        asn_mtyp string, but with "EXP" replaced by "PROD".
    """

    if asn_mtyp.startswith("EXP-") or asn_mtyp.startswith("EXP_"):
        asn_mtyp = "PROD" + asn_mtyp[3:]

    return asn_mtyp

def doImageStat(input):
    """Compute statistics for an image, and update keywords in header.

    Parameters
    ----------
    input: str
        Name of FITS file; keywords in the file will be modified in-place.
    """

    fd = fits.open(input, mode="update")

    if fd[1].data is None:
        fd.close()
        return
    phdr = fd[0].header
    xtractab = expandFileName(phdr.get("xtractab", ""))
    detector = phdr.get("detector", "")
    segment = phdr.get("segment", "")           # used for FUV
    opt_elem = phdr.get("opt_elem", "")
    cenwave = phdr.get("cenwave", 0)
    (aperture, message) = getApertureKeyword(phdr)
    exptype = phdr.get("exptype", "")
    nextend = len(fd) - 1       # number of extensions
    nimsets = nextend // 3      # number of image sets

    for k in range(nimsets):
        extver = k + 1          # extver is one indexed

        hdr = fd[("SCI",extver)].header
        sci = fd[("SCI",extver)].data
        err = fd[("ERR",extver)].data
        dq = fd[("DQ",extver)].data

        dispaxis = hdr.get("dispaxis", 0)
        key = segmentSpecificKeyword("exptime", segment)
        exptime = hdr.get(key, 0.)
        sdqflags = hdr.get("sdqflags", 3832)
        x_offset = hdr.get("x_offset", 0)

        if exptype == "ACQ/IMAGE":
            dispaxis = 0

        if dispaxis > 0:
            axis = 2 - dispaxis         # 1 --> 1,  2 --> 0
            axis_length = fd[1].data.shape[axis]

        # This will be a list of dictionaries, one for FUV, three for NUV.
        stat_info = []

        if detector == "FUV":
            segment_list = [segment]                # just one
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

                xtract_info = getTable(xtractab, filter)
                if xtract_info is None:
                    continue

                slope = xtract_info.field("slope")[0]
                b_spec = xtract_info.field("b_spec")[0]
                extr_height = xtract_info.field("height")[0]

                sci_band = np.zeros((extr_height, axis_length),
                                    dtype=np.float32)
                ccos.extractband(sci, axis, slope, b_spec, x_offset,
                                 sci_band)

                if err is None:
                    err_band = None
                else:
                    err_band = np.zeros((extr_height, axis_length),
                                        dtype=np.float32)
                    ccos.extractband(err, axis, slope, b_spec, x_offset,
                                     err_band)

                if dq is None:
                    dq_band = None
                else:
                    dq_band = np.zeros((extr_height, axis_length),
                                       dtype=np.int16)
                    ccos.extractband(dq, axis, slope, b_spec, x_offset,
                                     dq_band)

                stat_info.append(computeStat(sci_band, err_band, dq_band,
                                             sdqflags))

            else:
                # This is presumably a target-acquisition image.  Compute info
                # for the entire image.
                stat_info.append(computeStat(sci, err, dq, sdqflags))

        # Combine the three NUV stripes, or for FUV return the first element.
        stat_avg = combineStat(stat_info)

        sci_hdr = fd[("SCI",extver)].header
        sci_hdr["ngoodpix"] = stat_avg["ngoodpix"]
        sci_hdr["goodmean"] = exptime * stat_avg["sci_goodmean"]
        sci_hdr["goodmax"] = exptime * stat_avg["sci_goodmax"]
        if err is not None:
            err_hdr = fd[("ERR",extver)].header
            err_hdr["ngoodpix"] = stat_avg["ngoodpix"]
            err_hdr["goodmean"] = exptime * stat_avg["err_goodmean"]
            err_hdr["goodmax"] = exptime * stat_avg["err_goodmax"]

    fd.close()

def doSpecStat(input):
    """Compute statistics for a table, and update keywords in header.

    The NET column will be read, and statistics computed for all rows.

    Parameters
    ----------
    input: str
        Name of FITS file; keywords in the file will be modified in-place.
    """

    fd = fits.open(input, mode="update")
    try:
        sci_extn = fd["SCI"]
    except KeyError:
        doTagFlashStat(fd)                      # extname is "LAMPFLASH"
        fd.close()
        return

    if sci_extn.data is None or len(sci_extn.data) == 0:
        fd.close()
        return
    sdqflags = sci_extn.header["sdqflags"]
    outdata = sci_extn.data
    nrows = outdata.shape[0]
    if nrows < 1:
        fd.close()
        return
    exptime_col = outdata.field("EXPTIME")
    net = outdata.field("NET")
    error = outdata.field("ERROR")
    dq = outdata.field("DQ")

    # This will be a list of dictionaries, one for each segment or stripe.
    # (statistics for the error array are computed but then ignored)
    stat_info = []
    sum_exptime = 0.
    for row in range(nrows):
        sum_exptime += exptime_col[row]
        onestat = computeStat(net[row], error[row], dq[row], sdqflags)
        stat_info.append(onestat)
    exptime = sum_exptime / nrows

    # Combine the segments or stripes.
    stat_avg = combineStat(stat_info)

    sci_extn.header["ngoodpix"] = stat_avg["ngoodpix"]
    sci_extn.header["goodmean"] = exptime * stat_avg["sci_goodmean"]
    sci_extn.header["goodmax"] = exptime * stat_avg["sci_goodmax"]

    fd.close()

def doTagFlashStat(fd):
    """Compute statistics for an (already open) tagflash output file.

    The GROSS column will be read, and statistics computed for all rows.

    Parameters
    ----------
    fd: pyfits HDUList object
        HDU list for the FITS file (opened by doSpecStat).
    """

    sci_extn = fd["LAMPFLASH"]
    if sci_extn.data is None or len(sci_extn.data) == 0:
        return

    outdata = sci_extn.data
    nrows = outdata.shape[0]
    if nrows < 1:
        return
    nelem = outdata.field("NELEM")
    gross = outdata.field("GROSS")

    sum_gross = 0.
    max_gross = 0.
    n = 0
    for row in range(nrows):
        max_gross = max(max_gross, np.maximum.reduce(gross[row]))
        sum_gross += np.sum(gross[row])
        n += nelem[row]

    sci_extn.header["ngoodpix"] = n
    sci_extn.header["goodmean"] = sum_gross / float(n)
    sci_extn.header["goodmax"] = max_gross

def computeStat(sci_band, err_band=None, dq_band=None, sdqflags=3832):
    """Compute statistics.

    The function value is a dictionary with the info.  The keys are the
    keyword names, except that ones that have the same keyword but different
    values in the SCI and ERR extensions (goodmean, goodmax) have
    sci_ or err_ prefixes.

    Parameters
    ----------
    sci_band: array_like
        Science data array for which statistics are needed.

    err_band: array_like
        Error array (but may be None) associated with sci_band.

    dq_band: array_like
        Data quality array (but may be None) associated with sci_band.

    sdqflags: int
        "Serious" data quality flags.

    Returns
    -------
    stat_info: dictionary
        Contains values for ngoodpix, sci_goodmax, sci_goodmean,
        err_goodmax, err_goodmean.
    """

    # default values:
    stat_info = {"ngoodpix": 0, "sci_goodmax": 0., "sci_goodmean": 0.,
                                "err_goodmax": 0., "err_goodmean": 0.}

    # Don't quit if there are numpy exceptions.
    # xxx np.Error.setMode(all="warn", underflow="ignore")

    # Compute statistics for the sci array.  Note that mask is used
    # for both the sci and err arrays(if there is a dq_band).
    if dq_band is None:
        sci_good = np.ravel(sci_band)
    else:
        serious_dq = dq_band & sdqflags
        # mask = 1 where dq == 0
        mask = np.where(serious_dq == 0)
        sci_good = sci_band[mask]

    ngoodpix = len(sci_good)
    stat_info["ngoodpix"] = ngoodpix
    if ngoodpix > 0:
        stat_info["sci_goodmax"] = np.maximum.reduce(sci_good)
        stat_info["sci_goodmean"] = np.sum(sci_good) / ngoodpix
    del sci_good

    # Compute statistics for the err array.
    if err_band is not None:
        if dq_band is None:
            err_good = np.ravel(err_band)
        else:
            err_good = err_band[mask]
        if ngoodpix > 0:
            stat_info["err_goodmax"] = np.maximum.reduce(err_good)
            stat_info["err_goodmean"] = \
                      np.sum(err_good) / ngoodpix

    return stat_info

def combineStat(stat_info):
    """Combine statistical info for the segments or stripes.

    The input is a list of dictionaries.  The output is one dictionary
    with the same keys and with values that are the averages of the input.

    Parameters
    ----------
    stat_info: list of dictionaries
        One dictionary for each segment (FUV) or stripe (NUV).

    Returns
    -------
    dictionary (same keys as an element of input list)
        Contains values for ngoodpix, sci_goodmax, sci_goodmean,
        err_goodmax, err_goodmean; these values are the averages of
        the values in the input.
    """

    if len(stat_info) == 1:
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
            sci_max = max(sci_max, stat["sci_goodmax"])
            sci_sum += (n * stat["sci_goodmean"])
            if "err_goodmax" in stat:
                err_max = max(err_max, stat["err_goodmax"])
                err_sum += (n * stat["err_goodmean"])

    if sum_n > 0:
        sci_sum /= float(sum_n)
        err_sum /= float(sum_n)

    return {"ngoodpix": sum_n,
            "sci_goodmax": sci_max, "sci_goodmean": sci_sum,
            "err_goodmax": err_max, "err_goodmean": err_sum}

def overrideKeywords(phdr, hdr, info, switches, reffiles):
    """Override the calibration switch and reference file keywords.

    The calibration switch and reference file keywords will be overridden
    with values from switches and reffiles respectively.  Keywords
    cal_ver, opt_elem, cenwave, fpoffset, obstype, exptype and aperture
    in the primary header, as well as keywords dispaxis and x_offset in
    the extension header, will be overridden from info.

    Parameters
    ----------
    phdr: pyfits Header object
        Primary header from input file.

    hdr: pyfits Header object
        Extension header from input file.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switch keywords and values.

    reffiles: dictionary
        Reference file keywords and names.
    """

    for key in switches.keys():
        if key in phdr:
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
        if key.find("_hdr") < 0 and key in phdr:
            phdr[key] = reffiles[key+"_hdr"]

    for key in ["cal_ver", "opt_elem", "cenwave", "fpoffset", "obstype",
                "exptype", "aperture"]:
        if key in phdr:
            phdr[key] = info[key]

    if "dispaxis" in hdr:
        hdr["dispaxis"] = info["dispaxis"]

    hdr["x_offset"] = info["x_offset"]

def updatePulseHeightKeywords(hdr, segment, low, high):
    """Update the screening limit keywords for pulse height.

    This is only used for FUV data, since NUV doesn't have pulse height info.

    Parameters
    ----------
    hdr: pyfits Header object
        header with keywords to be modified

    segment: str
        FUVA or FUVB (last character used to construct keyword names)

    low: float
        value for PHALOWR[AB]

    high: float
        value for PHAUPPR[AB]
    """

    key_low  = "PHALOWR" + segment[-1]
    hdr[key_low] = low
    key_high = "PHAUPPR" + segment[-1]
    hdr[key_high] = high

def getPulseHeightRange(hdr, segment):
    """Get the pulse height range that was used for PHACORR.

    Parameters
    ----------
    hdr: pyfits Header object
        Extension header of corrtag, counts, flt, etc.

    segment: str
        Segment name ("FUVA" or "FUVB")

    Returns
    -------
    str, or None if keyword(s) are missing or less than 0
        "ll_hh", where ll is the lower limit and hh is the upper limit
    """

    if segment[:3] != "FUV":
        return None

    # These keywords were assigned when PHACORR was done.
    key_low  = "PHALOWR" + segment[-1]
    low = hdr.get(key_low, -1)
    key_high = "PHAUPPR" + segment[-1]
    high = hdr.get(key_high, -1)

    if low < 0:
        low = None
    if high < 0:
        high = None

    if low is None or high is None:
        return None

    return "%2d_%2d" % (low, high)

def tempPulseHeightRange(ref):
    """Get keyword PHARANGE from the primary header of a reference file.

    Parameters
    ----------
    ref: str
        Name of a reference file

    Returns
    -------
    str or None
        Value of keyword PHARANGE, or None if the keyword is missing
    """

    fd = fits.open(ref, "readonly")
    ref_pharange = fd[0].header.get("pharange", None)
    fd.close()

    return ref_pharange

def comparePulseHeightRanges(pharange, ref_pharange, refname):
    """Compare pharange with the pulse height range from the phatab.

    Parameters
    ----------
    pharange: str or None
        pulse height range from the PHATAB, formatted "ll_hh", where
        ll and hh are the lower and upper limits

    ref_pharange: str or None
        pulse height range used when calibrating the data used for
        creating the reference file (refname), formatted "ll_hh", where
        ll and hh are the lower and upper limits

    refname: str
        name of reference file for comparing ranges (only used for
        printing a warning message)
    """

    if pharange is None or ref_pharange is None:
        return

    words = pharange.split("_")
    low = int(words[0])
    high = int(words[1])

    ref_words = ref_pharange.split("_")
    if len(ref_words) != 2:
        printWarning("Can't compare pulse height ranges for %s; "
                     "PHARANGE = %s" % (refname, ref_pharange))
        return
    ref_low = int(ref_words[0])
    ref_high = int(ref_words[1])
    if ref_low != low or ref_high != high:
        printWarning("Pulse height ranges for %s don't agree:" % refname)
        printContinuation("PHATAB limits are %d to %d, "
                          "but PHARANGE limits are %d to %d" %
                          (low, high, ref_low, ref_high))

def getSwitch(phdr, keyword):
    """Get the value of a calibration switch from a primary header.

    The value will be converted to upper case.

    Parameters
    ----------
    phdr: pyfits Header object
        Primary header

    keyword: str
        Name of keyword to get from header

    Returns
    -------
    str
        Value of the keyword keyword, converted to upper case; for
        keyword STATFLAG, the value will be "PERFORM" or "OMIT" if
        statflag is T or F respectively.
    """

    if keyword in phdr:
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

def setVerbosity(verbosity_level):
    """Copy verbosity to a variable that is global for this file.

    Parameters
    ----------
    verbosity_level: int
        The value to assign to the variable verbosity (global to this
        file); possible values (QUIET, VERBOSE, VERY_VERBOSE) are defined
        in calcosparam.py
    """

    global verbosity
    verbosity = verbosity_level

def checkVerbosity(level):
    """Return True if verbosity is at least as great as level.

    Returns
    -------
    boolean
        True if a message with verbosity equal to level would be printed

    Examples
    --------
    >>> setVerbosity(VERBOSE)
    >>> print checkVerbosity(QUIET)
    1
    >>> print checkVerbosity(VERBOSE)
    1
    >>> print checkVerbosity(VERY_VERBOSE)
    0
    """

    return (verbosity >= level)

def setWriteToTrailer(flag=False):
    """Set the flag to indicate whether we should write to trailer files.

    flag: boolean
        Value to assign to the variable write_to_trailer (global to this
        file); if True, the printMsg function will write to the trailer
        file, in addition to the standard output
    """

    global write_to_trailer

    write_to_trailer = flag

def openTrailer(filename):
    """Open the trailer file for filename in append mode.

    Parameters
    ----------
    filename: str
        Name of an input (science or wavecal) file, but including the
        full directory, and with the ".fits" extension replaced by ".tra"
    """

    global fd_trl
    global write_to_trailer

    if not write_to_trailer:
        fd_trl = None
        return

    closeTrailer()

    fd_trl = open(filename, 'a')

def writeVersionToTrailer():
    """Write the calcos version string to the trailer file."""

    if fd_trl is not None:
        fd_trl.write("CALCOS version " + CALCOS_VERSION + "\n")
        fd_trl.flush()

def closeTrailer():
    """Close the trailer file if it is open."""

    global fd_trl

    if fd_trl is not None and not fd_trl.closed:
        fd_trl.close()
    fd_trl = None

def printMsg(message, level=QUIET):
    """Print 'message' if verbosity is at least as great as 'level'.

    Examples
    --------
    >>> setVerbosity(VERBOSE)

    >>> printMsg("quiet", QUIET)
    quiet

    >>> printMsg("verbose", VERBOSE)
    verbose

    >>> printMsg("very verbose", VERY_VERBOSE)
    """

    if verbosity >= level:
        print(message)
        sys.stdout.flush()
        if fd_trl is not None:
            fd_trl.write(message+"\n")
            fd_trl.flush()

def printIntro(str):
    """Print introductory message.

    Parameters
    ----------
    str: str
        String to be printed
    """

    printMsg("", VERBOSE)
    printMsg(str + " -- " + returnTime(), VERBOSE)

def printFilenames(names, shift_file=None, stimfile=None, livetimefile=None):
    """Print input and output filenames.

    Parameters
    ----------
    names: list of tuples
        Each tuple is (label, filename), where label is a short string
        (preferably 10 characters or less) that describes the file, and
        filename is the name of the file; the file may be an existing
        input file or an output file that hasn't been created yet

    shift_file: str or None
        Name of input text file to specify shift1 and shift2 (or None if
        no shift file was specified)

    stimfile: str or None
        Name of output text file for stim positions (or None if no file
        was specified)

    livetimefile: str or None
        Name of output text file for livetime factors (or None if no file
        was specified)

    Examples
    --------
    >>> setVerbosity(VERBOSE)
    >>> names = [("Input", "abc_raw.fits"), ("Output", "abc_flt.fits")]
    >>> printFilenames(names)
    Input     abc_raw.fits
    Output    abc_flt.fits

    >>> printFilenames(names, stimfile="stim.txt", livetimefile="live.txt")
    Input     abc_raw.fits
    Output    abc_flt.fits
    stim locations log file   stim.txt
    livetime factors log file live.txt
    """

    for (label, filename) in names:
        printMsg("%-10s%s" % (label, filename), VERBOSE)

    if shift_file is not None:
        printMsg("wavecal shifts overridden by file " + shift_file, VERBOSE)
    if stimfile is not None:
        printMsg("stim locations log file   " + stimfile, VERBOSE)
    if livetimefile is not None:
        printMsg("livetime factors log file " + livetimefile, VERBOSE)

def printMode(info):
    """Print info about the observation mode.

    Parameters
    ----------
    info: dictionary
        Header keywords and values.
    """

    if info["detector"] == "FUV":
        printMsg("DETECTOR  FUV, segment " + info["segment"][-1], VERBOSE)
    else:
        printMsg("DETECTOR  NUV", VERBOSE)
    printMsg("EXPTYPE   " + info["exptype"], VERBOSE)
    if info["obstype"] == "SPECTROSCOPIC":
        printMsg("OPT_ELEM  " + info["opt_elem"] + \
                 ", CENWAVE " + str(info["cenwave"]) + \
                 ", FPOFFSET " + str(info["fpoffset"]), VERBOSE)
    else:
        printMsg("OPT_ELEM  " + info["opt_elem"], VERBOSE)
    printMsg("APERTURE  " + info["aperture"], VERBOSE)

    printMsg("", VERBOSE)

def printSwitch(keyword, switches):
    """Print calibration switch name and value.

    Parameters
    ----------
    keyword: str
        Keyword name of calibration switch (e.g. "flatcorr")

    switches: dictionary
        Dictionary of calibration switches

    Examples
    --------
    >>> setVerbosity(VERBOSE)
    >>> switches = {"statflag": "PERFORM", "flatcorr": "PERFORM", "geocorr": "COMPLETE", "randcorr": "SKIPPED"}
    >>> printSwitch("statflag", switches)
    STATFLAG  T

    >>> printSwitch("flatcorr", switches)
    FLATCORR  PERFORM

    >>> printSwitch("geocorr", switches)
    GEOCORR   OMIT (already complete)

    >>> printSwitch("randcorr", switches)
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
    printMsg(message, VERBOSE)

def printRef(keyword, reffiles):
    """Print reference file keyword and file name.

    Parameters
    ----------
    keyword: str
        Keyword name for reference file name (e.g. "flatfile")

    reffiles: dictionary
        Dictionary of reference file names

    Examples
    --------
    >>> setVerbosity(VERBOSE)
    >>> reffiles = {"flatfile": "abc_flat.fits", "flatfile_hdr": "lref$abc_flat.fits"}
    >>> printRef("flatfile", reffiles)
    FLATFILE= lref$abc_flat.fits
    """

    key_upper = keyword.upper()
    key_lower = keyword.lower()
    printMsg("%-8s= %s" % (key_upper, reffiles[key_lower+"_hdr"]), VERBOSE)

def printWarning(message, level=QUIET):
    """Print a warning message.

    Parameters
    ----------
    message: str
        Warning message to be printed
    """

    printMsg("Warning:  " + message, level)

def printError(message):
    """Print an error message.

    Parameters
    ----------
    message: str
        Error message to be printed
    """

    printMsg("ERROR:  " + message, level=QUIET)

def printContinuation(message, level=QUIET):
    """Print a continuation line of a warning or error message.

    Parameters
    ----------
    message: str
        Continuation message to be printed
    """

    printMsg("    " + message, level)

def returnTime():
    """Return the current date and time, formatted into a string.

    Returns
    -------
    str
        The current local time, e.g. "20-Oct-2010 16:28:08 EDT"
    """

    return time.strftime("%d-%b-%Y %H:%M:%S %Z", time.localtime(time.time()))

def getPedigree(switch, refkey, filename, level=VERBOSE):
    """Return the value of the PEDIGREE keyword.

    Parameters
    ----------
    switch: str
        Keyword name for calibration switch

    refkey: str
        Keyword name for the reference file

    filename: str
        Name of the reference file

    level: int
        QUIET, VERBOSE, or VERY_VERBOSE (defined in calcosparam.py)

    Returns
    -------
    str
        The value of the PEDIGREE keyword, or "OK" if not found
    """

    if filename == "N/A":
        return "OK"

    fd = fits.open(filename, mode="readonly")
    pedigree = fd[0].header.get("pedigree", "OK")
    fd.close()
    if pedigree == "DUMMY":
        printWarning("%s %s is a dummy file" % (refkey.upper(), filename),
                     level=VERBOSE)
        printContinuation("so %s will not be done." %
                          switch.upper(), level=VERBOSE)

    return pedigree

def getApertureKeyword(hdr):
    """Get the value of the APERTURE keyword.

    The reason for this function is that some thermal-vac data had "-FUV"
    or "-NUV" appended to the aperture name, and in some cases the keyword
    value was "RelMvReq".  This function will strip off "-FUV" or "-NUV"
    (if found), and it will replace "RelMvReq" with the value of PROPAPER.

    Parameters
    ----------
    hdr: pyfits Header object
        Primary header from which to get the APERTURE keyword

    Returns
    -------
    tuple (str, str)
        The value of the APERTURE keyword, corrected if necessary,
        and a message (may be empty) about what was changed.
    """

    message = ""

    aperture = hdr.get("aperture", NOT_APPLICABLE)
    propaper = hdr.get("propaper", NOT_APPLICABLE)
    hdr_aper = aperture                         # save keyword value
    if len(aperture) > 3 and aperture[3] == "-":        # e.g. "PSA-FUV"
        aperture = aperture[0:3]
        message = "APERTURE changed from %s to %s" % (hdr_aper, aperture)
    if aperture == "RelMvReq":
        if propaper in APERTURE_NAMES:
            message = "APERTURE changed from %s to %s " \
                       "(copied from PROPAPER)" % (hdr_aper, propaper)
            aperture = propaper
        else:
            message = "Guessing correct APERTURE ... was %s" % hdr_aper
            shutter = hdr.get("shutter", NOT_APPLICABLE)
            lampused = hdr.get("lampused", NOT_APPLICABLE)
            if shutter.lower() == "closed":
                if lampused[0] == "P":          # Platinum lamp
                    message += ", now set to WCA"
                    aperture = "WCA"
                elif lampused[0] == "D":        # Deuterium lamp
                    message += ", now set to FCA"
                    aperture = "FCA"
                else:           # shutter closed, no lamp --> dark exposure
                    message += ", now set to PSA"
                    aperture = "PSA"
            else:
                life_adj = hdr.get("life_adj", NOT_APPLICABLE)
                aperypos = hdr.get("aperypos", NOT_APPLICABLE)
                aperture = guessAperFromLocn(life_adj, aperypos)
                message += ", now set to PSA"

    return (aperture, message)

def guessAperFromLocn(life_adj, aperypos):
    """Infer which aperture was in use from the aperture position.

    Parameters
    ----------
    life_adj: int or "N/A"
        Integer code for the "lifetime position," or "N/A" if the LIFE_ADJ
        keyword was not found in the header.

    aperypos: float or "N/A"
        Location (in steps) of the aperture block, or "N/A" if the keyword
        was not found in the header.

    Returns
    -------
    aperture: str or None
        An educated guess as to which aperture was in use, either "PSA" or
        "BOA".  None will be returned if the input arguments were not
        sufficient for determining the aperture.
    """

    if life_adj == 1:
        if 116. < aperypos and aperypos < 136.:
            aperture = "PSA"
        elif -163. < aperypos and aperypos < -143.:
            aperture = "BOA"
        else:
            aperture = None
    elif life_adj == 2:
        # -64 steps from life_adj = 1
        if 52. < aperypos and aperypos < 72.:
            aperture = "PSA"
        elif -227. < aperypos and aperypos < -207.:
            aperture = "BOA"
        else:
            aperture = None
    else:
        aperture = None

    return aperture

def computeLifeAdjOffset(info):
    """Compute offset in pixels from LIFE_ADJ = 1.

    Parameters
    ----------
    info: dictionary
        General information.  info["life_adj_offset"] will be updated with
        the offset (may be 0.) from life_adj = 1 to the location of the
        aperture, as determined from aperture and aperypos.
    """

    if info["life_adj"] != -1 or \
       info["aperypos"] == NOT_APPLICABLE or \
       info["opt_elem"] == NOT_APPLICABLE:
        info["life_adj_offset"] = 0.
        return

    aperture = info["aperture"]
    aperypos = info["aperypos"]
    opt_elem = info["opt_elem"]

    # These parameters are in calcosparam.py.

    if aperture in APERTURE_POSN1:
        offset = aperypos - APERTURE_POSN1[aperture]
    else:
        offset = aperypos - APERTURE_POSN1["PSA"]       # reasonable default

    # Positive is toward larger pixel numbers in cross-dispersion.
    offset_arcsec = offset * ARCSEC_PER_XD_APER_STEP
    info["life_adj_offset"] = offset_arcsec * XD_PLATE_SCALE[opt_elem]

def segmentSpecificKeyword(keyword_root, segment):
    """Construct a segment-specific keyword.

    Parameters
    ----------
    keyword_root: str
        The part of the keyword name preceding the segment-specific
        character.

    segment: str
        Segment or stripe name

    Returns
    -------
    str
        Keyword (lower case) for the specified segment.
    """

    keyword_root = keyword_root.lower()
    if segment[0] == "F":
        key = keyword_root + segment[-1].lower()
    else:
        key = keyword_root

    return key

def expandFileName(filename):
    """Expand environment variable in a file name.

    If the input file name begins with either a Unix-style or IRAF-style
    environment variable (e.g. $lref/name_dqi.fits or lref$name_dqi.fits
    respectively), this routine expands the variable and returns a complete
    path name for the file.

    Parameters
    ----------
    filename: str
        A file name, possibly including an environment variable

    Returns
    -------
    filename: str
        The name of the file, with any environment variable expanded
    """

    n = filename.find("$")
    if n == 0:
        if filename != NOT_APPLICABLE:
            # Unix-style file name.
            filename = os.path.expandvars(filename)
    elif n > 0:
        # IRAF-style file name.
        temp = "$" + filename[0:n] + os.sep + filename[n+1:]
        filename = os.path.expandvars(temp)
        # If filename contains "//", delete one of them.
        double_sep = os.sep + os.sep
        i = filename.find(double_sep)
        if i != -1:
            filename = filename[:i+1] + filename[i+2:]

    return filename

def changeSegment(filename, detector, segment):
    """Replace '_a' with '_b' or vice versa, if appropriate.

    This was written for auto/GO wavecal file names for FUV data.  Wavecals
    are processed from the x1d file, and the name of the raw file is for
    the first segment in the input list (which will be FUVA if both segments
    are present).  When calibrating segment B data, the name or names of
    the wavecal files need to be changed to end in "_b.fits" instead of
    "_a.fits".

    Parameters
    ----------
    filename: str
        One or more file names, separated by spaces

    detector: str
        FUV or NUV

    segment: str
        FUVA or FUVB, if detector is FUV

    Returns
    -------
    str
        A copy of the input filename, but with '_a' replaced with '_b'
        or vice versa, or no change if the input name does not end in
        '_a.fits' or '_b.fits'
    """

    if detector != "FUV":
        return filename

    if segment == "FUVB":
        names = filename.split()
        new_names = []
        for name in names:
            if name.endswith("_a.fits"):
                n = len(name) - 7
                name = name[:n] + "_b.fits"
            new_names.append(name)
        filename = " ".join(new_names)
    elif segment == "FUVA":
        names = filename.split()
        new_names = []
        for name in names:
            if name.endswith("_b.fits"):
                n = len(name) - 7
                name = name[:n] + "_a.fits"
            new_names.append(name)
        filename = " ".join(new_names)

    return filename

def findRefFile(ref, missing, wrong_filetype, bad_version):
    """Check for the existence of a reference file.

    If the reference file does not exist, its name is added to the
    'missing' dictionary.  If the file does exist, open the file and
    compare 'filetype' with the value of the FILETYPE keyword in the
    primary header.  If they're not the same (unless FILETYPE is "ANY"),
    then an entry is added to the 'wrong_filetype' dictionary.  The
    VCALCOS keyword is also gotten from the primary header of the
    reference file (with a default value of "1.0").  If the version of
    the reference file is not consistent with calcos, the reference file
    name and error message will be added to the 'bad_version' dictionary.

    Parameters
    ----------
    ref: dictionary
        a dictionary with the following keys:
            reference file keyword (e.g. "FLATFILE")
            filename (name of reference file)
            calcos_ver (calcos version number)
            min_ver (minimum acceptable value of VCALCOS)
            filetype (e.g. "FLAT FIELD REFERENCE IMAGE")

    missing: dictionary
        Messages about missing reference files; the reference file keywords
        are the keys

    wrong_filetype: dictionary
        Messages about wrong FILETYPE keyword in reference files; the
        reference file keywords are the keys

    bad_version: dictionary
        Messages about inconsistent version strings; the reference file
        keywords are the keys
    """

    keyword    = ref["keyword"]
    filename   = ref["filename"]
    calcos_ver = ref["calcos_ver"]
    min_ver    = ref["min_ver"]
    filetype   = ref["filetype"]

    if os.access(filename, os.R_OK):

        fd = fits.open(filename, mode="readonly")
        phdr = fd[0].header

        phdr_filetype = phdr.get("FILETYPE", "ANY")
        if phdr_filetype != "ANY" and phdr_filetype != filetype:
            wrong_filetype[keyword] = (filename, filetype)

        if min_ver != "ANY":
            vcalcos = phdr.get("VCALCOS", "1.0")
            if not isinstance(vcalcos, str):
                vcalcos = str(vcalcos)
            compare = cmpVersion(min_ver, vcalcos, calcos_ver)
            if compare < 0:
                bad_version[keyword] = (filename,
                "  the reference file must be at least version " + min_ver)
            elif compare > 0:
                bad_version[keyword] = (filename,
                "  to use this reference file you must have calcos version " + \
                 vcalcos + " or later.")

        fd.close()

    else:

        missing[keyword] = filename

def fitQuadratic(x, y):
    """Fit a quadratic to y vs x.

    Parameters
    ----------
    x: array_like
        Array of independent variable values
    y: array_like
        Array of dependent variable values

    Returns
    -------
    tuple
        (coeff, var), where coeff is an array of the coefficients of the
        fit (coeff[0] + coeff[1]*x + coeff[2]*x**2), and var is an array of
        the corresponding variances; coeff and var will be None if there
        was a LinAlgError.
    """

    assert len(x) == len(y)
    n = float(len(x))

    y0 = y[0]
    yp = (y - y0)

    sum_x = x.sum(dtype=np.float64).item()
    sum_x2 = (x**2).sum(dtype=np.float64).item()
    sum_x3 = (x**3).sum(dtype=np.float64).item()
    sum_x4 = (x**4).sum(dtype=np.float64).item()
    sum_y = yp.sum(dtype=np.float64).item()
    sum_yx = (yp*x).sum(dtype=np.float64).item()
    sum_yx2 = (yp*x**2).sum(dtype=np.float64).item()

    m = np.array([[n,      sum_x,  sum_x2],
                  [sum_x,  sum_x2, sum_x3],
                  [sum_x2, sum_x3, sum_x4]])
    v = np.array([sum_y, sum_yx, sum_yx2])

    succeeded = True
    try:
        coeff = LA.solve(m, v)
        m_inv = LA.inv(m)
    except LA.LinAlgError:
        succeeded = False

    if not succeeded:
        coeff = None
        var = None
    else:
        coeff[0] += y0
        (a0, a1, a2) = coeff
        if len(x) > 3:
            residual = y - (a0 + a1*x + a2*x**2)
            chisq = (residual**2).sum()
            scatter = math.sqrt(chisq / (n - 3.))
        else:
            scatter = 0.
        var = np.array([m_inv[0,0], m_inv[1,1], m_inv[2,2]]) * scatter

    return (coeff, var)

def centerOfQuadratic(coeff, var):
    """Find the center of a quadratic function from its coefficients.

    Parameters
    ----------
    coeff: array_like or None
        The coefficients of the fit (or None if not determined):
           y = coeff[0] + coeff[1]*x + coeff[2]*x**2

    var: array_like or None
        The variances of the coefficients (or None if not determined)

    Returns
    -------
    tuple
        (x_min, x_min_sigma), where x_min is value at which y is an
        extremum, and x_min_sigma is the error estimate for x_min, based
        on the scatter of the values around the fitted curve; the values
        will be (None, 0.) if coeff is None or if the second-order
        coefficient is zero
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
        x_min_sigma = 0.5 * math.sqrt(var1 / a2**2 + var2 * a1**2 / a2**4)

    return (x_min, x_min_sigma)

def fitQuartic(x, y):
    """Fit a fourth-order polynomial to y vs x.

    not currently used

    Parameters
    ----------
    x: array_like
        Array of independent variable values
    y: array_like
        Array of dependent variable values

    Returns
    -------
    tuple
        (coeff, var), where coeff is an array of the coefficients of the
        fit (coeff[0] + coeff[1]*x + coeff[2]*x**2 + coeff[3]*x**3 +
             coeff[4]*x**4), and var is an array of the corresponding
        variances; coeff and var will be None if there was a LinAlgError.
    """

    assert len(x) == len(y)
    n = float(len(x))

    y0 = y[0]
    yp = (y - y0)

    sum_x = x.sum(dtype=np.float64).item()
    sum_x2 = (x**2).sum(dtype=np.float64).item()
    sum_x3 = (x**3).sum(dtype=np.float64).item()
    sum_x4 = (x**4).sum(dtype=np.float64).item()
    sum_x5 = (x**5).sum(dtype=np.float64).item()
    sum_x6 = (x**6).sum(dtype=np.float64).item()
    sum_x7 = (x**7).sum(dtype=np.float64).item()
    sum_x8 = (x**8).sum(dtype=np.float64).item()
    sum_y = yp.sum(dtype=np.float64).item()
    sum_yx = (yp*x).sum(dtype=np.float64).item()
    sum_yx2 = (yp*x**2).sum(dtype=np.float64).item()
    sum_yx3 = (yp*x**3).sum(dtype=np.float64).item()
    sum_yx4 = (yp*x**4).sum(dtype=np.float64).item()

    m = np.array([[n,      sum_x,  sum_x2,  sum_x3, sum_x4],
                  [sum_x,  sum_x2, sum_x3,  sum_x4, sum_x5],
                  [sum_x2, sum_x3, sum_x4,  sum_x5, sum_x6],
                  [sum_x3, sum_x4, sum_x5,  sum_x6, sum_x7],
                  [sum_x4, sum_x5, sum_x6,  sum_x7, sum_x8]])
    v = np.array([sum_y, sum_yx, sum_yx2, sum_yx3, sum_yx4])

    try:
        coeff = LA.solve(m, v)
        m_inv = LA.inv(m)
    except LA.LinAlgError:
        succeeded = False

    if not succeeded:
        coeff = None
        var = None
    else:
        (a0, a1, a2, a3, a4) = coeff
        a0 += y0
        if len(x) > 5:
            residual = y - (a0 + a1*x + a2*x**2 + a3*x**3 + a4*x**4)
            chisq = (residual**2).sum()
            scatter = math.sqrt(chisq / (n - 3.))
        else:
            scatter = 0.
        var = np.array([m_inv[0,0], m_inv[1,1], m_inv[2,2],
                        m_inv[3,3], m_inv[4,4]]) * scatter

    return (coeff, var)

def centerOfQuartic(x, coeff):
    """Find the center of a fourth-order function from its coefficients.

    not currently used

    Parameters
    ----------
    x: array_like
        Array of independent variable values

    coeff: array_like or None
        The coefficients of the fit (or None if not determined):
           y = coeff[0] + coeff[1]*x + coeff[2]*x**2

    Returns
    -------
    x_min: float, or None
        The value at which y is a minimum, or None if coeff is None
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
    for i in range(len(x) - 1):
        # opposite slopes at i and i+1?
        if yp[i] * yp[i+1] < 0.:
            value = x[i] + abs(yp[i] / (yp[i+1] - yp[i]))
            xminmax.append((i, value))
    if len(xminmax) == 0:
        return None
    x_min = xminmax[0][1]
    if len(xminmax) > 1:
        for (i, value) in enumerate(xminmax):
            if value < minvalue:
                x_min = xminmax[i][1]

    return x_min

def errGehrels(counts):
    """Compute error estimate.

    The error estimate is computed using the Gehrels approximation for the
    upper confidence limit.

    Parameters
    ----------
    counts: array_like or float
        Number of counts (not necessarily integer values).

    Returns
    -------
    tuple of 2 array_like or float
        (The lower error estimate for counts,
         the upper error estimate for counts)
    """
    icounts = (counts + .5).astype(np.int)
    upper = (1. + np.sqrt(icounts + 0.75))
    lower = np.where(icounts > 0., Gehrels_lower(icounts), 0.)
    return (lower.astype(np.float32), upper.astype(np.float32))

def Gehrels_lower(counts):
    return counts - counts * (1.0 - 1.0 / (9.0 * counts) - 1.0 / (3.0 * np.sqrt(counts)))**3

def errFrequentist(counts):
    """Compute errors using the 'frequentist-confidence' option of astropy's poisson_conf_interval

    Parameters
    ----------
    counts: array-like or float
        Number of counts (not necessarily integer values).

    Returns
    -------
    tuple of 2 array-like or float
        (The lower error estimate for counts,
         the upper error estimate for counts)
    """

    lower, upper = poisson_conf_interval(counts, interval='frequentist-confidence')
    err_lower = counts - lower
    err_upper = upper - counts
    return (err_lower.astype(np.float32), err_upper.astype(np.float32))

def precess(t, target):
    """Precess target to the time of observation.

    This function is currently not used.
    It could be called by timetag.heliocentricVelocity.

    Parameters
    ----------
    t: float
        Time (MJD) of observation

    target: array_like
        Unit vector pointing toward the target, J2000 coordinates

    Returns
    -------
    list
        Target coordinates precessed to time t
    """

    # 51544.5 is MJD for 2000 Jan 1.5 UT, or JD 2451545.0
    dt = (t - 51544.5) / 36525.
    dt2 = dt * dt
    dt3 = dt * dt * dt

    zeta = 2306.2181 * dt + 0.30188 * dt2 + 0.017998 * dt3

    z = 2306.2181 * dt + 1.09468 * dt2 + 0.018203 * dt3

    theta = 2004.3109 * dt - 0.42665 * dt2 - 0.041833 * dt3

    # convert from arc seconds to radians
    zeta = math.radians(zeta / 3600.)
    z = math.radians(z / 3600.)
    theta = math.radians(theta / 3600.)

    # convert zeta, z, theta to a rotation matrix
    a = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
    # first row
    a[0] =  math.cos(z) * math.cos(theta) * math.cos(zeta) - \
            math.sin(z) *                   math.sin(zeta)
    a[1] = -math.cos(z) * math.cos(theta) * math.sin(zeta) - \
            math.sin(z) *                   math.cos(zeta)
    a[2] = -math.cos(z) * math.sin(theta)

    # second row
    a[3] =  math.sin(z) * math.cos(theta) * math.cos(zeta) + \
            math.cos(z) *                   math.sin(zeta)
    a[4] = -math.sin(z) * math.cos(theta) * math.sin(zeta) + \
            math.cos(z) *                   math.cos(zeta)
    a[5] = -math.sin(z) * math.sin(theta)

    # third row
    a[6] =                math.sin(theta) * math.cos(zeta)
    a[7] =               -math.sin(theta) * math.sin(zeta)
    a[8] =                math.cos(theta)

    # Multiply:  a * target
    targ = [0., 0., 0.]
    targ[0] = a[0] * target[0] + a[1] * target[1] + a[2] * target[2]
    targ[1] = a[3] * target[0] + a[4] * target[1] + a[5] * target[2]
    targ[2] = a[6] * target[0] + a[7] * target[1] + a[8] * target[2]

    return targ

def cmpVersion(min_ver, vcalcos, calcos_ver):
    """Compare version strings.

    The test passes if min_ver <= vcalcos <= calcos_ver, in which case
    this function will return 0.

    If min_ver > vcalcos, this function returns -1; otherwise,
    if vcalcos > calcos_ver, this function returns +1.

    Each string is first separated into a list of substrings, splitting
    on ".", and then those substrings are split into an integer part and
    a part starting with a letter (if any).  Then comparisons are made on
    the substrings one at a time.

    Parameters
    ----------
    min_ver: str
        calcos requires the reference file to be at least this version

    vcalcos: str
        VCALCOS from the primary header of the reference file

    calcos_ver: str
        Version of calcos

    Returns
    -------
    int
        -1 if min_ver > vcalcos
         0 if min_ver <= vcalcos <= calcos_ver
        +1 if vcalcos > calcos_ver

    Examples
    --------
    >>> print cmpVersion("1", "1", "1.1")
    0

    >>> print cmpVersion("1", "1.1", "1")
    1

    >>> print cmpVersion("1.1", "1", "1")
    -1

    >>> print cmpVersion("1.1", "1.1", "1.2")
    0

    >>> print cmpVersion("1.1", "1.2", "1.1")
    1

    >>> print cmpVersion("1.2", "1.1", "1.1")
    -1

    >>> print cmpVersion("1.0", "1.7", "2.3")
    0

    >>> print cmpVersion("2.7", "2.8", "2.8a")
    0

    >>> print cmpVersion("2.0", "2.13.1", "2.13")
    1

    >>> print cmpVersion("2.9", "2.9", "2.13.1")
    0

    >>> print cmpVersion("2.12d", "2.13b", "2.13a")
    1

    >>> print cmpVersion("2.13d", "2.13b", "2.13a")
    -1

    >>> print cmpVersion("2.13", "2.13b", "2.13c")
    0
    """

    minv = min_ver.split('.')
    phdrv = vcalcos.split('.')
    calv = calcos_ver.split('.')

    # Replace any element that ends with one or more letters by two
    # elements, the integer part and the string part.
    splitIntLetter(minv)
    splitIntLetter(phdrv)
    splitIntLetter(calv)

    len_minv = len(minv)
    len_phdrv = len(phdrv)
    len_calv = len(calv)

    # Pad these lists with "0" to make them all the same length.
    maxlength = max(len(minv), len(phdrv), len(calv))
    for i in range(len_minv, maxlength):
        minv.append("0")
    for i in range(len_phdrv, maxlength):
        phdrv.append("0")
    for i in range(len_calv, maxlength):
        calv.append("0")
    length = min(len(minv), len(phdrv), len(calv))

    # These are initial values.  They'll be reset if either test passes
    # (because of an inequality in a part of the version string), in which
    # case tests on subsequent parts of the version strings will be omitted.
    # For example, "2.3" is later than "1.7" because 2 > 1, and we ignore
    # the fact that the second parts 3 and 7 compare in the opposite sense.
    passed_min_test = False
    passed_calcos_test = False

    for i in range(length):
        if not passed_min_test:
            cmp = cmpPart(minv[i], phdrv[i])
            if cmp < 0:
                passed_min_test = True
            elif cmp > 0:
                return -1
        if not passed_calcos_test:
            cmp = cmpPart(phdrv[i], calv[i])
            if cmp < 0:
                passed_calcos_test = True
            elif cmp > 0:
                return 1

    if passed_min_test or passed_calcos_test:
        return 0

    return 0

def splitIntLetter(x):
    """Split each part of x that has an appended letter or letters.

    Parameters
    ----------
    x: list of strings
        List of the parts of a version string, previously split on '.'.
        This may be modified in-place, by splitting each element that has an
        integer part followed by a string.  For example, x = ["2", "13b"]
        would be modified to x = ["2", "13", "b"].
    """

    nine = ord("9")
    nelem = len(x)
    mixed_parts = []
    breakpoint = []
    for i in range(nelem):
        part = x[i]
        for j in range(len(part)):
            if ord(part[j]) > nine:
                mixed_parts.append(i)
                breakpoint.append(j)
                break

    for n in range(len(mixed_parts) - 1, -1, -1):
        k = mixed_parts[n]
        j = breakpoint[n]
        part = x[k]
        p1 = part[0:j]
        if len(p1) < 1:
            p1 = "0"
        p2 = part[j:]
        x[k] = p2
        x.insert(k, p1)

def cmpPart(s1, s2):
    """Compare two strings.

    s1 and s2 are "parts" of version strings; each is expected to be an
    integer or a letter.  The function value will be -1, 0, or +1,
    depending on whether s1 is less than, equal to, or greater than s2
    respectively.
        +1 --> s1 > s2
         0 --> s1 == s2
        -1 --> s1 < s2
    """

    if s1 == s2:
        return 0

    s1_is_int = True            # initial values
    s2_is_int = True
    try:
        int_s1 = int(s1)
    except ValueError:
        s1_is_int = False
    try:
        int_s2 = int(s2)
    except ValueError:
        s2_is_int = False

    if s1_is_int and s2_is_int:
        # integer comparison
        if int_s1 < int_s2:
            value = -1
        elif int_s1 > int_s2:
            value = 1
        else:
            value = 0
    else:
        # string comparison
        if s1 < s2:
            value = -1
        elif s1 > s2:
            value = 1
        else:
            value = 0
    return value


def _test():
    from . import cosutil
    import doctest
    return doctest.testmod(cosutil)

if __name__ == "__main__":
    _test()
