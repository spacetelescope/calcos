#! /usr/bin/env python

import os
import sys
import time
import types
import numpy as N
import pyfits
import ccos
from calcosparam import *       # parameter definitions

# initial value
verbosity = VERBOSE

def writeOutputEvents (infile, outfile):
    """
    This function creates a recarray object with the column definitions
    appropriate for a corrected time-tag table, reads an input events table
    into this object, and writes it to the output file.  If the input file
    contains a GTI table, that will be copied unchanged to output.

    argument:
    infile         name of the input FITS file containing an EVENTS table
                   and optionally a GTI table
    outfile        name of file for output EVENTS table (and GTI table)
    """

    # ifd = pyfits.open (infile, mode="readonly", memmap=1)
    ifd = pyfits.open (infile, mode="readonly")
    events_extn = ifd["EVENTS"]
    indata = events_extn.data
    if indata is None:
        nrows = 0
    else:
        nrows = indata.shape[0]
    detector = ifd[0].header.get ("detector", "FUV")
    tagflash = (ifd[0].header.get ("tagflash", default="NONE") != "NONE")

    # Check whether the PHA column exists.
    if detector == "FUV":
        pha_exists = 1
        if nrows > 0:
            try:
                pha = indata.field ("PHA")
            except NameError:
                pha_exists = 0
    else:
        pha_exists = 0

    # Create the output events HDU.
    col = []
    col.append (pyfits.Column (name="TIME", format="1E", unit="s"))
    if detector == "FUV":
        col.append (pyfits.Column (name="XCORR", format="1E", unit="pixel"))
        col.append (pyfits.Column (name="XDOPP", format="1E", unit="pixel"))
        col.append (pyfits.Column (name="YCORR", format="1E", unit="pixel"))
    else:
        col.append (pyfits.Column (name="RAWX", format="1I", unit="pixel"))
        col.append (pyfits.Column (name="RAWY", format="1I", unit="pixel"))
        col.append (pyfits.Column (name="YDOPP", format="1E", unit="pixel"))
    if tagflash:
        col.append (pyfits.Column (name="XFULL", format="1E", unit="pixel"))
        col.append (pyfits.Column (name="YFULL", format="1E", unit="pixel"))
    col.append (pyfits.Column (name="EPSILON", format="1E"))
    col.append (pyfits.Column (name="DQ", format="1I"))
    if pha_exists:
        col.append (pyfits.Column (name="PHA", format="1B"))
        cd = pyfits.ColDefs (col)
    else:
        cd = pyfits.ColDefs (col)

    hdu = pyfits.new_table (cd, header=events_extn.header, nrows=nrows)
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

    if detector == "FUV":
        outdata.field ("XCORR")[:] = indata.field ("RAWX")
        outdata.field ("YCORR")[:] = indata.field ("RAWY")
    else:
        outdata.field ("RAWX")[:] = indata.field ("RAWX")
        outdata.field ("RAWY")[:] = indata.field ("RAWY")

    if detector == "FUV":
        outdata.field ("XDOPP")[:] = \
                N.zeros (nrows, dtype=N.float32)
    else:
        outdata.field ("YDOPP")[:] = \
                N.zeros (nrows, dtype=N.float32)

    outdata.field ("EPSILON")[:] = \
            N.ones (nrows, dtype=N.float32)

    outdata.field ("DQ")[:] = N.zeros (nrows, dtype=N.int16)

    if pha_exists:
        outdata.field ("PHA")[:] = indata.field ("PHA")

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

def returnGTI (infile):
    """Return a list of (start, stop) good time intervals.

    arguments:
    infile         name of the input FITS file containing a GTI table
    """

    fd = pyfits.open (infile, mode="readonly")
    if len (fd) != 3:
        fd.close()
        gti = []
        return gti

    indata = fd["GTI"].data
    if indata is None:
        gti = []
    else:
        nrows = indata.shape[0]
        start = indata.field ("START")
        stop = indata.field ("STOP")
        gti = [(start[i], stop[i]) for i in range (nrows)]

    return gti

def getTable (table, filter, exactly_one=False, at_least_one=False):
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

    arguments:
    table          name of the reference table
    filter         dictionary; each key is a column name, and if the value
                   in that column matches the filter value for some row,
                   that row will be included in the set that is returned
    exactly_one    true if there must be one and only one matching row
    at_least_one   true if there must be at least one matching row
    """

    # fd = pyfits.open (table, mode="readonly", memmap=1)
    fd = pyfits.open (table, mode="readonly")
    data = fd[1].data

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
        if isinstance (column, N.chararray):
            wild = (column == STRING_WILDCARD)
        elif isinstance (column[0], int):
            wild = (column == INT_WILDCARD)
        if wild is not None:
            selected = N.logical_or (selected, wild)

        select_arrays.append (selected)

    if len (select_arrays) > 0:
        selected = select_arrays[0]
        for sel_i in select_arrays[1:]:
             selected = N.logical_and (selected, sel_i)
        newdata = data[selected]
    else:
        newdata = fd[1].data.copy()

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

def evalDisp (x, coeff):
    """Evaluate the dispersion relation at x.

    The function value will be the wavelength (or array of wavelengths) at x,
    in Angstroms.

    arguments:
    x             pixel coordinate (or array of coordinates)
    coeff         array of coefficients for the dispersion relation
    """

    ncoeff = len (coeff)
    if ncoeff < 2:
        raise ValueError, "Dispersion relation has too few coefficients"

    sum = coeff[ncoeff-1]
    for i in range (ncoeff-2, -1, -1):
        sum = sum * x + coeff[i]

    return sum

def evalDerivDisp (x, coeff):
    """Evaluate the derivative of the dispersion relation at x.

    The function value will be the slope (or array of slopes) at x,
    in Angstroms per pixel.

    arguments:
    x             pixel coordinate (or array of coordinates)
    coeff         array of coefficients for the dispersion relation
    """

    ncoeff = len (coeff)
    if ncoeff < 2:
        raise ValueError, "Dispersion relation has too few coefficients"

    sum = (ncoeff-1.) * coeff[ncoeff-1]
    for n in range (ncoeff-2, 0, -1):
        sum = sum * x + n * coeff[n]

    return sum

def addRandomNumbers (x, y, seed):
    """Add pseudo-random numbers to the x and y columns.

    arguments:
    x, y          arrays of detector X and Y coordinates
    seed          an integer value to initialize the random-number generator;
                  -1 means use the system clock to generate a starting value
    """

    use_clock = (seed == -1)
    seed = ccos.addrandom (x, seed, use_clock)
    use_clock = 0
    seed = ccos.addrandom (y, seed, use_clock)

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

def getInputDQ (input):
    """Return the data quality array, or an array of zeros.

    If the data quality extension (EXTNAME = "DQ", EXTVER = 1) actually has
    a non-null data portion, that data array will be returned.  If the data
    portion is null (NAXIS = 0), a constant array will be returned; in this
    case the size will be taken from keywords NPIX1 and NPIX2, and the data
    value will be the value of the PIXVALUE keyword.

    argument:
    input         name of a FITS file containing an image set (SCI, ERR, DQ);
                  only the DQ extension will be read
    """

    fd = pyfits.open (input, mode="readonly")

    hdr = fd[("DQ",1)].header

    # Does the data portion exist?
    if hdr["naxis"] > 0:
        dq_array = fd[("DQ",1)].data
    else:
        npix1 = hdr["npix1"]
        npix2 = hdr["npix2"]
        dq_array = N.zeros ((npix2, npix1), dtype=N.int16)
        if hdr.has_key ("pixvalue"):
            pixvalue = hdr["pixvalue"]
            if pixvalue != 0:
                dq_array[:,:] = pixvalue

    fd.close()

    return dq_array

def updateDQArray (bpixtab, info, doppcorr, dq_array):
    """Apply the data quality initialization table to DQ array.

    dq_array is a 2-D array, to be written as the DQ extension in an
    ACCUM file (_counts or _flt).  Its contents are assumed to be valid
    on input, since it may have been read from the raw file (if the
    input was an ACCUM image), and it may therefore include flagged
    pixels.  The flag information in the bpixtab will be combined
    (in-place) with dq_array using bitwise OR.

    arguments:
    bpixtab    name of the data quality initialization table
    info       dictionary of keywords and values (for doppcorr)
    doppcorr   shift DQ positions to track Doppler shift during exposure?
    dq_array   data quality image array (modified in-place)
    """

    dq_info = getTable (bpixtab, filter={"segment": info["segment"]})
    if dq_info is None:
        return

    if doppcorr == "PERFORM":
        expstart = info["expstart"]
        exptime  = info["exptime"]
        doppmag  = info["doppmag"]
        doppzero = info["doppzero"]
        orbitper = info["orbitper"]
        axis = 2 - info["dispaxis"]     # 1 --> 1,  2 --> 0

        # time is the time in seconds since doppzero.
        nelem = int (round (exptime))           # one element per sec
        nelem = max (nelem, 1)
        time = N.arange (nelem, dtype=N.float32) + \
                   (expstart - doppzero) * SEC_PER_DAY

        # shift is in pixels (wavelengths decrease toward larger pixel number).
        shift = doppmag * N.sin (2. * N.pi * time / orbitper)
        mindopp = N.minimum.reduce (shift)
        maxdopp = N.maximum.reduce (shift)
        mindopp = int (round (mindopp))
        maxdopp = int (round (maxdopp))
        printMsg ("DOPPCORR applied to BPIXTAB positions", VERBOSE)
    else:
        axis = -1                       # disable Doppler correction
        mindopp = 0
        maxdopp = 0

    # Update the 2-D data quality extension array from the DQI table info.
    ccos.bindq (dq_info.field ("lx"), dq_info.field ("ly"),
                dq_info.field ("dx"), dq_info.field ("dy"),
                dq_info.field ("dq"), dq_array, axis, mindopp, maxdopp)

def activeArea (segment, brftab):
    """Return the low and high limits of the FUV active area.

    arguments:
    segment       for finding a row in the brftab
    brftab        name of the baseline reference frame table
                  (ignored for NUV)

    The function value is a tuple of the low and high limits of the
    active area of the detector.  For NUV this will be (0, 1023).
    """

    if segment[0] == "N":
        return (0, NUV_Y-1)

    brf_info = getTable (brftab, {"segment": segment}, exactly_one=True)

    try:
        b_low = brf_info.field ("a_low")[0]
        b_high = brf_info.field ("a_high")[0]
    except NameError:
        # These are the cross-dispersion locations of the stims,
        # rounded to the nearest integer.
        sy1 = int (brf_info.field ("sy1")[0] + 0.5)
        sy2 = int (brf_info.field ("sy2")[0] + 0.5)

        y_max = FUV_Y - 1               # because of zero indexing
        b_low = 2 * sy1
        b_high = y_max - 2 * (y_max - sy2)

    return (b_low, b_high)

def flagOutOfBounds (phdr, hdr, dq_array):
    """Flag regions that are outside all subarrays (done in-place).

    arguments:
    phdr          primary header
    hdr           extension header
    dq_array      data quality array
    """

    if not phdr.has_key ("subarray"):
        return
    if not phdr["subarray"]:
        return

    nsubarrays = hdr.get ("nsubarry", 0)
    if nsubarrays < 1:
        return

    temp = dq_array.copy()

    # Initially flag the entire image as out of bounds, then remove the
    # flag (set it to zero) for each subarray.
    temp[:,:] = DQ_OUT_OF_BOUNDS
    for i in range (nsubarrays):
        sub_number = str (i)
        x0 = hdr["corner"+sub_number+"x"]       # these keywords are 0-indexed
        y0 = hdr["corner"+sub_number+"y"]
        xsize = hdr["size"+sub_number+"x"]
        ysize = hdr["size"+sub_number+"y"]
        temp[y0:y0+ysize,x0:x0+xsize] = DQ_OK

    dq_array[:,:] = N.bitwise_or (dq_array, temp)

def tableHeaderToImage (thdr, shape, ndtype):
    """Modify keywords to turn a table header into an image header.

    The function returns a copy of the header with table-specific keywords
    deleted and other keywords modified, to serve as an image header for
    data of the specified shape and type.

    arguments:
    thdr          a FITS Header object for a table
    shape         a list of axis lengths for the associated data array
    ndtype        numpy type, e.g. float32
    """

    hdr = thdr.copy()

    hdr["XTENSION"] = "IMAGE"

    # Set the NAXIS and NAXISi keywords, and delete unused NAXISi keywords.
    if shape == (0,):
        naxis = 0
    else:
        naxis = len (shape)
    MAX_NAXIS = 7
    hdr["NAXIS"] = naxis
    previous = "NAXIS"
    for i in range (naxis):
        j = i + 1                       # j is one-indexed
        key = "NAXIS" + str (j)
        if hdr.has_key (key):
            hdr[key] = shape[naxis-j]
        else:
            hdr.update (key, shape[naxis-j], after=previous)
        previous = key
    for i in range (naxis, MAX_NAXIS):
        j = i + 1
        key = "NAXIS" + str (j)
        if hdr.has_key (key):
            del hdr[key]

    hdr["PCOUNT"] = 0
    hdr["GCOUNT"] = 1

    if ndtype == N.int32:
        hdr["bitpix"] = 32
    elif ndtype == N.int16:
        hdr["bitpix"] = 16
    elif ndtype == N.uint16:
        hdr["bitpix"] = 16
        hdr.update ("BSCALE", 1., comment="scale factor for unsigned data")
        hdr.update ("BZERO", 32768., comment="zero point for unsigned data")
    elif ndtype == N.int8 or ndtype == N.uint8:
        hdr["bitpix"] = 8
    elif ndtype == N.float32:
        hdr["bitpix"] = -32
    elif ndtype == N.float64:
        hdr["bitpix"] = -64
    else:
        raise TypeError, str (ndtype) + " is not a supported data type"

    if hdr.has_key ("TFIELDS"):
        ncols = hdr["TFIELDS"]
        del hdr["TFIELDS"]
    else:
        ncols = 0

    for keyword in ["TTYPE", "TFORM", "TUNIT", "TNULL", "TDISP",
                    "TSCAL", "TZERO", "TALEN"]:
        for i in range (1, ncols+1):    # FITS keywords are one-indexed
            key = keyword + str (i)
            if hdr.has_key (key):
                del hdr[key]

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
    """Rename a FITS file, and update FILENAME keyword.

    arguments:
    infile      current name of a file
    outfile     new name for the file
    """

    printMsg ("rename " + infile + " --> " + outfile, VERY_VERBOSE)

    os.rename (infile, outfile)

    fd = pyfits.open (outfile, mode="update")
    updateFilename (fd[0].header, outfile)
    fd.close()

def doImageStat (input):
    """Compute statistics for an image, and update keywords in header.

    argument:
    input       name of FITS file; keywords in the file will be modified
                in-place
    """

    # Open the file readonly to read the data.
    fd = pyfits.open (input, mode="readonly", memmap=0)
    # fd = pyfits.open (input, mode="readonly", memmap=1)

    if fd[1].data is None:
        fd.close()
        return
    phdr = fd[0].header
    hdr = fd["SCI"].header
    sci = fd["SCI"].data
    err = fd["ERR"].data
    dq = fd["DQ"].data

    xtractab = expandFileName (phdr.get ("xtractab", ""))
    detector = phdr.get ("detector", "")
    if detector == "FUV":
        fuv_segment = phdr.get ("segment", "")  # not used for NUV
    opt_elem = phdr.get ("opt_elem", "")
    cenwave = phdr.get ("cenwave", 0)
    aperture = getApertureKeyword (phdr, truncate=1)
    dispaxis = hdr.get ("dispaxis", 0)
    exptime = hdr.get ("exptime", 0.)
    sdqflags = hdr.get ("sdqflags", 32767)

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

            sci_band = N.zeros ((extr_height, axis_length),
                                   dtype=N.float32)
            ccos.extractband (sci, axis, slope, b_spec, sci_band)

            if err is None:
                err_band = None
            else:
                err_band = N.zeros ((extr_height, axis_length),
                                       dtype=N.float32)
                ccos.extractband (err, axis, slope, b_spec, err_band)

            if dq is None:
                dq_band = None
            else:
                dq_band = N.zeros ((extr_height, axis_length),
                                      dtype=N.int16)
                ccos.extractband (dq, axis, slope, b_spec, dq_band)

            stat_info.append (computeStat (sci_band,
                          err_band, dq_band, sdqflags))

        else:
            # This is presumably a target-acquisition image.  Compute info
            # for the entire image.
            stat_info.append (computeStat (sci, err, dq, sdqflags))

    # Combine the three NUV stripes, or for FUV just return the first element.
    stat_avg = combineStat (stat_info)

    fd.close()

    # Now re-open the file read/write to update the keywords.
    fd = pyfits.open (input, mode="update")
    fd["SCI"].header.update ("ngoodpix", stat_avg["ngoodpix"])
    fd["SCI"].header.update ("goodmean", exptime * stat_avg["sci_goodmean"])
    fd["SCI"].header.update ("goodmax", exptime * stat_avg["sci_goodmax"])
    if err is not None:
        fd["ERR"].header.update ("ngoodpix", stat_avg["ngoodpix"])
        fd["ERR"].header.update ("goodmean", exptime * stat_avg["err_goodmean"])
        fd["ERR"].header.update ("goodmax", exptime * stat_avg["err_goodmax"])
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
    maxdq = outdata.field ("MAXDQ")

    # This will be a list of dictionaries, one for each segment or stripe.
    stat_info = []
    sum_exptime = 0.
    for row in range (nrows):
        #stat_info.append (computeStat (net[row], error[row], maxdq[row],
        #                  sdqflags))
        sum_exptime += exptime_col[row]
        onestat = computeStat (net[row], error[row], maxdq[row], sdqflags)
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
        max_gross = max (max_gross, N.maximum.reduce (gross[row]))
        sum_gross += N.sum (gross[row])
        n += nelem[row]

    sci_extn.header.update ("ngoodpix", n)
    sci_extn.header.update ("goodmean", sum_gross / float (n))
    sci_extn.header.update ("goodmax", max_gross)

def computeStat (sci_band, err_band=None, dq_band=None, sdqflags=32767):
    """Compute statistics.

    The function value is a dictionary with the info.  The keys are the
    keyword names, except that ones that have the same keyword but different
    values in the SCI and ERR extensions (goodmean, goodmax) have
    sci_ or err_ prefixes.

    arguments:
    sci_band       science data array for which statistics are needed
    err_band       error array (but may be None) associated with sci_band
    dq_band        data quality array (may be None) associated with sci_band
    sdqflags       "serious" data quality flags (default includes all flags)
    """

    # default values:
    stat_info = {"ngoodpix": 0, "sci_goodmax": 0., "sci_goodmean": 0.,
                                "err_goodmax": 0., "err_goodmean": 0.}

    # Don't quit if there are numpy exceptions.
    # xxx N.Error.setMode (all="warn", underflow="ignore")

    # Compute statistics for the sci array.  Note that mask is used
    # for both the sci and err arrays (if there is a dq_band).
    if dq_band is None:
        sci_good = N.ravel (sci_band)
    else:
        serious_dq = dq_band & sdqflags
        # mask = 1 where dq == 0
        mask = N.where (serious_dq == 0)
        sci_good = sci_band[mask]

    ngoodpix = len (sci_good)
    stat_info["ngoodpix"] = ngoodpix
    if ngoodpix > 0:
        stat_info["sci_goodmax"] = N.maximum.reduce (sci_good)
        stat_info["sci_goodmean"] = N.sum (sci_good) / ngoodpix
    del sci_good

    # Compute statistics for the err array.
    if err_band is not None:
        if dq_band is None:
            err_good = N.ravel (err_band)
        else:
            err_good = err_band[mask]
        if ngoodpix > 0:
            stat_info["err_goodmax"] = N.maximum.reduce (err_good)
            stat_info["err_goodmean"] = \
                      N.sum (err_good) / ngoodpix

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
    specific keywords will be overridden.

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

    for key in ["cal_ver", "opt_elem", "cenwave", "fpoffset", \
                "obstype", "exptype"]:
        if phdr.has_key (key):
            phdr[key] = info[key]

    if hdr.has_key ("dispaxis"):
        hdr["dispaxis"] = info["dispaxis"]

def updatePulseHeightKeywords (hdr, segment, low, high):
    """Update the screening limit keywords for pulse height.

    This is only used for FUV data, since NUV doesn't have pulse height info.

    arguments:
    hdr            header with keywords to be modified
    segment        FUVA or FUVB (last character used to construct keyword names)
    low, high      default values for PHALOWR[AB] and PHAUPPR[AB] respectively
    """

    # Update the values for the screening limit keywords
    key_low  = "PHALOWR" + segment[-1]
    phalowr = hdr.get (key_low, low)
    if low < phalowr:
        hdr.update (key_low, low)
    key_high = "PHAUPPR" + segment[-1]
    phauppr = hdr.get (key_high, high)
    if high > phauppr:
        hdr.update (key_high, high)

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

def printIntro (str):
    """Print introductory message.

    argument:
    str            string to be printed
    """

    printMsg ("", VERBOSE)
    printMsg (str + " -- " + returnTime(), VERBOSE)

def printFilenames (names, stimfile=None, livetimefile=None):
    """Print input and output filenames.

    arguments:
    names         a list of (label, filename) tuples
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
              ", CENWAVE " + str (info["cenwave"]), VERBOSE)
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
