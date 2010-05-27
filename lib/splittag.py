from __future__ import division         # confidence high
import glob
import os
import numpy
import pyfits
import cosutil
import ccos

NOT_APPLICABLE = "N/A"
SEC_PER_DAY = 86400.            # seconds in a day

def splittag (infiles, outroot, starttime=None, increment=None, endtime=None,
              time_list=None, verbosity=1):
    """Split TIME-TAG files into multiple files.

    @param infiles: name of input file, possibly including a wildcard character
    @type infiles: string
    @param outroot: root name with which to construct output file names
    @type outroot: string
    @param starttime: time at beginning of first interval, or None if
        time_list was specified
    @type starttime: float, or None
    @param increment: length of each time interval, or None if time_list
        was specified
    @type increment: float, or None
    @param endtime: time at end of last interval, or None if time_list is
        specified
    @type endtime: float, or None
    @param time_list: string containing explicit times of beginning of each
        time interval and end of the last interval; the first time may be
        given as "start" (time=0, i.e. at EXPSTART), and the last time may be
        given as either "stop" or "end" (time of last event in TIME column)
    @type time_list: string
    @param verbosity: 0 --> print almost nothing, 1 --> print some info,
        2 --> print more info
    @type verbosity: int

    All times are in seconds, and the zero point is EXPSTART.
    """

    infiles = os.path.expandvars (infiles)
    outroot = os.path.expandvars (outroot)
    inlist = glob.glob (infiles)
    for input in inlist:
        splitOneTag (input, outroot, starttime, increment, endtime,
                     time_list, verbosity)

def splitOneTag (input, outroot, starttime=None, increment=None, endtime=None,
              time_list=None, verbosity=1):
    """Split a TIME-TAG file into multiple files.

    @param input: name of input file
    @type input: string
    @param outroot: root name with which to construct output file names
    @type outroot: string
    @param starttime: time at beginning of first interval, or None
    @type starttime: float, or None
    @param increment: length of each time interval, or None
    @type increment: float, or None
    @param endtime: time at end of last interval, or None
    @type endtime: float, or None
    @param time_list: string containing explicit times of beginning of each
        time interval and end of the last interval
    @type time_list: string
    @param verbosity: 0-2, indicating how much should be printed
    @type verbosity: int
    """

    cosutil.setVerbosity (verbosity)

    if not cosutil.isCorrtag (input):
        raise RuntimeError, "%s is not a corrtag file" % input

    (inroot, suffix) = splitName (input)

    ifd = pyfits.open (input, mode="readonly")
    phdr = ifd[0].header
    try:
        hdr = ifd[("events")].header
    except KeyError:
        ifd.close()
        raise RuntimeError, "%s is not a corrtag file" % input
    data = ifd[("events")].data
    time_col = cosutil.getColCopy (filename="", column="time", data=data)

    info = getInfo (input, phdr, hdr)
    if info["wavecorr"] != "COMPLETE":
        cosutil.printWarning ("WAVECORR was not done for " + input)
    gti_hdu = getGTI (ifd)
    time_list = convertToSlices (time_col,
                                 starttime, increment, endtime, time_list)
    cosutil.printMsg ("time_list = %s" % repr (time_list), 2)

    # define output columns based on input table
    cd = ifd[("events")].columns

    file_index = 1              # one indexing for output file names
    for (t0, t1) in time_list:

        (i, j) = determineSlice (time_col, t0, t1)
        nrows = j - i
        if nrows <= 0:
            cosutil.printWarning ("no rows in increment %.2f to %.2f" %
                                  (t0, t1))
            continue

        filename = constructOutputName (outroot, file_index, suffix)
        ofd = pyfits.HDUList (pyfits.PrimaryHDU (header=phdr))
        hdu = pyfits.new_table (cd, header=hdr, nrows=nrows)
        ofd.append (hdu)

        copyRows (data, ofd, i, j)

        out_gti_hdu = createNewGTI (gti_hdu, t0, t1)
        if out_gti_hdu is not None:
            ofd.append (out_gti_hdu)

        updateKeywords (info, out_gti_hdu, t0, t1, ofd)

        ofd.writeto (filename)
        ofd.close()
        del ofd
        cosutil.printMsg ("%s written" % filename)
        file_index += 1

def splitName (input):
    """Split the input name into rootname and suffix.

    @param input: name of input file
    @type input: string

    @return: root name (up to "_corrtag"), everything past the root name
    @rtype: tuple of two strings

    For example, if input = "xyz_corrtag_a.fits", then the returned tuple
    would be ("xyz", "_corrtag_a.fits").
    """

    i = input.find ("_corrtag")
    if i < 0:
        i = input.find (".fit")

    if i < 0:
        inroot = input
        suffix = ""
    else:
        inroot = input[:i]
        suffix = input[i:]

    return (inroot, suffix)

def getInfo (input, phdr, hdr):
    """Get header information.

    @param input: name of input file
    @type input: string
    @param phdr: primary header of input file
    @type phdr: pyfits Header object
    @param hdr: EVENTS extension header of input file
    @type hdr: pyfits Header object

    @return: keywords and values
    @rtype: dictionary

    The input file name is included in the calling sequence just so it can be
    added to the info dictionary.
    """

    info = {}

    info["input"] = input       # the name of the input file

    # This is a list of primary header keywords and default values.
    keylist = {
        "wavecorr":  "omit",
        "detector":  NOT_APPLICABLE,
        "segment":   NOT_APPLICABLE,
        "obstype":   NOT_APPLICABLE,
        "obsmode":   NOT_APPLICABLE,
        "exptype":   NOT_APPLICABLE,
        "opt_elem":  NOT_APPLICABLE}

    for key in keylist.keys():
        info[key] = phdr.get (key, default=keylist[key])

    # This is a list of extension header keywords and default values.
    keylist = {
        "exptime":  -1.,
        "expstart": -1.,
        "expend":   -1.}

    for key in keylist.keys():
        info[key] = hdr.get (key, default=keylist[key])

    return info

def convertToSlices (time_col, starttime, increment, endtime, time_list):
    """Return a list of two-element tuples, giving time intervals.

    @param time_col: a copy of the TIME column from the input table
    @type time_col: numpy array
    @param starttime: time at beginning of first interval, or None if
        time_list was specified
    @type starttime: float, or None
    @param increment: length of each time interval, or None if time_list
        was specified
    @type increment: float, or None
    @param endtime: time at end of last interval, or None if time_list is
        specified
    @type endtime: float, or None
    @param time_list: the times of the beginning of each interval and the
        time of the end of the last interval; or time_list may already be
        the output format, a list of two-element tuples, in which case it
        will be returned unchanged
    @type time_list: string or list

    @return: each tuple in the list gives the start and end times of an
        interval to be extracted from the input EVENTS table
    @rtype: list of two-element tuples
    """

    if not increment and not time_list:
        raise RuntimeError, "Must specify either increment or time_list."

    if time_list and len (time_list) < 2:
        raise RuntimeError, "time_list must have at least two elements."

    if increment and time_list:
        cosutil.printWarning ("Both increment and time_list were specified;"
                              " time_list will be used.")

    if time_list:

        if isinstance (time_list, str):
            time_list = convertToList (time_list)
        else:
            # If time_list is already a list of two-element tuples (or lists),
            # just return it.
            is_a_list_of_tuples = True          # default
            for value in time_list:
                if not isinstance (value, (list, tuple, numpy.ndarray)):
                    is_a_list_of_tuples = False
                    break
                if len (value) != 2:
                    is_a_list_of_tuples = False
                    break
            if is_a_list_of_tuples:
                return time_list

        # time_list is a list of times; extract the times into a list of
        # two-element tuples.
        new_time_list = []

        if isinstance (time_list[0], str):
            value = time_list[0].lower()
            if value == "start":
                t0 = 0.
            else:
                raise RuntimeError, \
                "First element of time_list is '%s'." % time_list[0]
        else:
            t0 = time_list[0]

        nelem = len (time_list)
        for i in range (1, nelem):
            if isinstance (time_list[i], str):
                value = time_list[i].lower()
                if value == "start":
                    t1 = 0.
                elif value == "stop" or value == "end":
                    t1 = time_col[-1]
                else:
                    raise RuntimeError, \
                    "Don't understand time_list[%d] = '%s'." % \
                        (i, time_list[i])
            else:
                t1 = time_list[i]
            if t1 < t0:
                cosutil.printError ("time_list = %s" % repr (time_list))
                raise RuntimeError, \
                "Values in time_list must be in increasing order."
            elif t0 == t1:
                continue
            new_time_list.append ((t0, t1))
            t0 = t1

    else:
        # Construct a list of two-element tuples from the start time and
        # increment, stopping at the end time.
        new_time_list = []

        if starttime is None:
            starttime = 0.
        t0 = starttime
        if endtime is None:
            endtime = time_col[-1]

        done = False
        while not done:
            if t0 >= endtime:
                done = True
            else:
                t1 = t0 + increment
                if t1 >= endtime:
                    t1 = endtime
                    done = True
                new_time_list.append ((t0, t1))
                t0 = t1

    # Remove intervals that are outside the range of the data.
    trimmed_list = []
    for interval in new_time_list:
        if interval[1] > time_col[0] and interval[0] <= time_col[-1]:
            trimmed_list.append (interval)

    return trimmed_list

def convertToList (time_string):
    """Split a string of times on commas and/or blanks.

    @param time_string: string containing times, separated by blanks and/or
        commas; the times may be integers or floats, but "start", "stop" and
        "end" are also allowed
    @type time_string: string

    @return: list of start and/or stop times of intervals, extracted from
        the input string
    @rtype: list
    """

    if time_string.find (",") >= 0:
        # split the string on commas, then strip whitespace
        words = []
        temp_words = time_string.split (",")
        for word in temp_words:
            words.extend (word.split())
    else:
        # split the string on blanks
        words = time_string.split()

    time_list = []
    for word in words:
        word = word.lower()
        if word == "start" or word == "stop" or word == "end":
            time_list.append (word)
        else:
            try:
                t_value = float (word)
            except ValueError:
                cosutil.printError ("Values in time_list must be a number," \
                                    " or 'start', 'stop', or 'end'.")
                raise
            time_list.append (t_value)

    return time_list

def determineSlice (time_col, t0, t1):
    """Find the indices correspond to the start and end times of an interval.

    @param time_col: a copy of the TIME column from the input table
    @type time_col: numpy array
    @param t0: time at the beginning of an interval
    @type t0: float
    @param t1: time at the end of an interval
    @type t1: float

    @return: the indices of a slice in time_col that will give the elements
        with times >= t0 and < t1
    @rtype: tuple of two integers
    """

    return ccos.range (time_col, t0, t1)

def constructOutputName (outroot, file_index, suffix):
    """Construct an output file name.

    @param outroot: root name with which to construct output file names
    @type outroot: string
    @param file_index: this number, preceded by "_", will be appended to
        outroot
    @type file_index: integer
    @param suffix: all of the input file name that follows the root name
    @type suffix: string

    @return: the name to use for the output file
    @rtype: string

    For example, if outroot="abc", file_index=3, and suffix="_corrtag_a.fits",
    the output file name will be "abc_3_corrtag_a.fits".

    If a file with the constructed name already exists, however, a new name
    will be tried, a name with "_1" appended after the file_index.  If that
    file also already exists, "_2" will be tried instead of "_1", and so on
    until a name is found that doesn't exist.  For the above example, the
    first such name would be "abc_3_1_corrtag_a.fits".
    """

    upper_limit = 10000

    filename = "%s_%d%s" % (outroot, file_index, suffix)
    if os.access (filename, os.R_OK):
        save_filename = filename
        i = 1
        done = False
        while not done:
            filename = "%s_%d_%d%s" % (outroot, file_index, i, suffix)
            if os.access (filename, os.R_OK):
                i += 1
                if i > upper_limit:
                    raise RuntimeError, "Output file already exists," \
                          " and upper limit of names has been exceeded."
            else:
                done = True
        cosutil.printWarning ("Output file %s already exists,"
                              " will use name %s instead." %
                              (save_filename, filename))

    return filename

def copyRows (data, ofd, i, j):
    """Copy rows from the input file to an output file.

    @param data: the data block of the EVENTS table extension
    @type data: numpy record array
    @param ofd: the data block of the EVENTS table extension
    @type ofd: pyfits HDUList object
    @param i: number (zero indexed) of the first row to copy
    @type i: integer
    @param j: one more than the number (zero indexed) of the last row to copy;
        that is, [i:j] is a slice
    @type j: integer
    """

    ofd[1].data = data[i:j]

def getGTI (ifd):
    """Find the most up-to-date GTI table in the input file.

    @param ifd: the pyfits file handle for the input file
    @type ifd: pyfits HDUList object

    @return: the GTI extension from the input file, or None if there isn't one
    @rtype: pyfits BinTableHDU object, or None
    """

    # Find the GTI table with the largest value of EXTVER.
    last_extver = 0                     # initial value
    hdunum = 0
    for i in range (1, len(ifd)):
        hdu = ifd[i]
        extname = hdu.header.get ("extname", "MISSING")
        if extname.upper() == "GTI":
            extver = hdu.header.get ("extver", 1)
            if extver > last_extver:
                last_extver = extver
                hdunum = i

    if hdunum < 1:
        gti_hdu = None
    else:
        gti_hdu = ifd[hdunum]
    return gti_hdu

def createNewGTI (gti_hdu, t0, t1):
    """Create a GTI table for the output table.

    @param info: keywords and values
    @type info: dictionary
    @param gti_hdu: the GTI table from the input file (may be None)
    @type gti_hdu: pyfits BinTableHDU object, or None
    @param t0: time at the start of the interval
    @type t0: float
    @param t1: time at the end of the interval
    @type t1: float

    @return: a GTI table to append to the output file
    @rtype: pyfits BinTableHDU object
    """

    cd = gti_hdu.columns

    if gti_hdu is None or gti_hdu.data is None:
        # No GTI table means the entire exposure is good.
        out_gti_hdu = pyfits.new_table (cd, header=gti_hdu.header, nrows=0)
        out_start_col = out_gti_hdu.data.field ("start")
        out_stop_col = out_gti_hdu.data.field ("stop")
        out_start_col[0] = t0
        out_stop_col[0] = t1
        return out_gti_hdu

    data = gti_hdu.data
    in_nrows = len (data)

    # columns in the GTI table from the input file
    start_col = data.field ("start")
    stop_col = data.field ("stop")

    gti = []                    # list of good (start, stop) intervals
    for i in range (in_nrows):
        start = start_col[i]
        stop = stop_col[i]
        if start >= t1 or stop <= t0:
            continue
        start = max (start, t0)
        stop = min (stop, t1)
        if (stop - start) <= 0.:
            continue
        gti.append ((start, stop))

    if not gti:
        gti.append ((0., 0.))
    out_nrows = len (gti)

    out_gti_hdu = pyfits.new_table (cd, header=gti_hdu.header, nrows=out_nrows)
    out_start_col = out_gti_hdu.data.field ("start")
    out_stop_col = out_gti_hdu.data.field ("stop")
    for i in range (out_nrows):
        (start, stop) = gti[i]
        out_start_col[i] = start
        out_stop_col[i] = stop

    return out_gti_hdu

def updateKeywords (info, out_gti_hdu, t0, t1, ofd):
    """Update keywords in an output file.

    @param info: keywords and values
    @type info: dictionary
    @param out_gti_hdu: header/data unit for the GTI table for the output file
    @type out_gti_hdu: pyfits BinTableHDU object
    @param t0: time at the start of the interval
    @type t0: float
    @param t1: time at the end of the interval
    @type t1: float
    @param ofd: the pyfits file handle for the output file
    @type ofd: pyfits HDUList object

    This function adds two HISTORY records to the output primary header and
    updates EXPTIME, EXPEND and EXPENDJ in the EVENTS extension header.
    """

    phdr = ofd[0].header
    hdr = ofd[1].header

    filename = os.path.basename (info["input"])         # just the file name
    phdr.add_history ("Copied from %s" % filename)
    phdr.add_history ("Time slice from input was %.3f to %.3f" % (t0, t1))

    if out_gti_hdu is None:
        exptime = t1 - t0
    else:
        start_col = out_gti_hdu.data.field ("start")
        stop_col = out_gti_hdu.data.field ("stop")
        exptime = 0.
        n = len (start_col)
        for i in range (n):
            exptime += (stop_col[i] - start_col[i])

    hdr.update ("exptime", exptime)

    expstart = info["expstart"]
    if expstart > 0.:
        expend = expstart + t1/SEC_PER_DAY
        hdr.update ("expend", expend)
        expend_j = expend + 2400000.5
        hdr.update ("expendj", expend_j)
