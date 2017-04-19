from __future__ import absolute_import, division         # confidence high
import glob
import math
import os
import numpy as np
import astropy.io.fits as fits
from . import cosutil
from . import ccos

NOT_APPLICABLE = "N/A"
SEC_PER_DAY = 86400.            # seconds in a day

def splittag(infiles, outroot, starttime=None, increment=None, endtime=None,
             time_list=None, verbosity=1):
    """Split TIME-TAG files into multiple files.

    All times are in seconds, and the zero point is EXPSTART.

    Parameters
    ----------
    infiles: str
        Name of input file, possibly including a wildcard character.

    outroot: str
        Root name with which to construct output file names.

    starttime: float or None
        Time at beginning of first interval, or None if time_list was
        specified.

    increment: float or None
        Length of each time interval, or None if time_list was specified.

    endtime: float or None
        Time at end of last interval, or None if time_list is specified.

    time_list: str
        String containing explicit times of beginning of each time
        interval and end of the last interval; the first time may be given
        as "start" (time=0, i.e. at EXPSTART), and the last time may be
        given as either "stop" or "end" (time of last event in TIME column).

    verbosity: int {0, 1, 2}
        0 --> print almost nothing, 1 --> print some info,
        2 --> print more info
    """

    infiles = os.path.expandvars(infiles)
    outroot = os.path.expandvars(outroot)
    inlist = glob.glob(infiles)
    for input in inlist:
        splitOneTag(input, outroot, starttime, increment, endtime,
                    time_list, verbosity)

def splitOneTag(input, outroot, starttime=None, increment=None, endtime=None,
                time_list=None, verbosity=1):
    """Split a TIME-TAG file into multiple files.

    Parameters
    ----------
    input: str
        Name of input file

    outroot: str
        Root name with which to construct output file names

    starttime: float or None
        Time at beginning of first interval, or None

    increment: float or None
        Length of each time interval, or None

    endtime: float or None
        Time at end of last interval, or None

    time_list: str
        String containing explicit times of beginning of each
        time interval and end of the last interval

    verbosity: int {0, 1, 2}
        Indicates how much should be printed
    """

    cosutil.setVerbosity(verbosity)

    if not cosutil.isCorrtag(input):
        raise RuntimeError("%s is not a corrtag file" % input)

    (inroot, suffix) = splitName(input)

    ifd = fits.open(input, mode="copyonwrite")
    phdr = ifd[0].header
    try:
        hdr = ifd[("events")].header
    except KeyError:
        ifd.close()
        raise RuntimeError("%s is not a corrtag file" % input)
    data = ifd[("events")].data
    time_col = data.field("TIME").astype(np.float64)

    info = getInfo(input, phdr, hdr)
    if info["wavecorr"] != "COMPLETE":
        cosutil.printWarning("WAVECORR was not done for " + input)
    gti_hdu = getGTI(ifd)
    timeline_hdu = getTimeline(ifd)
    time_list = convertToSlices(time_col,
                                starttime, increment, endtime, time_list)
    cosutil.printMsg("time_list = %s" % repr(time_list), 2)

    # define output columns based on input table
    cd = ifd[("events")].columns

    file_index = 1              # one indexing for output file names
    for (t0, t1) in time_list:

        (i, j) = determineSlice(time_col, t0, t1)
        nrows = j - i
        if nrows <= 0:
            cosutil.printWarning("no rows in increment %.2f to %.2f" %
                                 (t0, t1))
            continue

        filename = constructOutputName(outroot, file_index, suffix)
        ofd = fits.HDUList(fits.PrimaryHDU(header=phdr))
        hdu = fits.BinTableHDU.from_columns(cd, header=hdr, nrows=nrows)
        ofd.append(hdu)

        copyRows(data, ofd, i, j)
        nevents = j - i

        out_gti_hdu = createNewGTI(gti_hdu, t0, t1)
        if out_gti_hdu is not None:
            ofd.append(out_gti_hdu)

        out_timeline_hdu = createNewTimeline(timeline_hdu, t0, t1)
        if out_timeline_hdu is not None:
            ofd.append(out_timeline_hdu)

        updateKeywords(info, out_gti_hdu, t0, t1, nevents, ofd)

        ofd.writeto(filename)
        ofd.close()
        del ofd
        cosutil.printMsg("%s written" % filename)
        file_index += 1

def splitName(input):
    """Split the input name into rootname and suffix.

    Parameters
    ----------
    input: str
        Name of input file

    Returns
    -------
    tuple of two str
        The first string is the root name (up to "_corrtag"), and the
        second is everything past the root name.  For example, if
        input = "xyz_corrtag_a.fits", then the returned tuple would be
        ("xyz", "_corrtag_a.fits").
    """

    i = input.find("_corrtag")
    if i < 0:
        i = input.find(".fit")

    if i < 0:
        inroot = input
        suffix = ""
    else:
        inroot = input[:i]
        suffix = input[i:]

    return (inroot, suffix)

def getInfo(input, phdr, hdr):
    """Get header information.

    The input file name is included in the calling sequence just so it
    can be added to the info dictionary.

    Parameters
    ----------
    input: str
        Name of input file

    phdr: pyfits Header object
        Primary header of input file

    hdr: pyfits Header object
        EVENTS extension header of input file

    Returns
    -------
    info: dictionary
        Selected keywords and values from the input headers
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
        info[key] = phdr.get(key, default=keylist[key])

    # This is a list of extension header keywords and default values.
    # (also exptime; see below)
    keylist = {
        "expstart": -1.,
        "expend":   -1.}

    for key in keylist.keys():
        info[key] = hdr.get(key, default=keylist[key])
    exptime_key = cosutil.segmentSpecificKeyword("exptime", info["segment"])
    exptime_default = hdr.get("exptime", default=-1.)
    info["exptime"] = hdr.get(exptime_key, default=exptime_default)

    return info

def convertToSlices(time_col, starttime, increment, endtime, time_list):
    """Return a list of two-element tuples, giving time intervals.

    Parameters
    ----------
    time_col: array_like
        A copy of the TIME column from the input table

    starttime: float or None
        Time at beginning of first interval, or None if time_list was
        specified

    increment: float or None
        Length of each time interval, or None if time_list was specified

    endtime: float or None
        Time at end of last interval, or None if time_list was specified

    time_list: str or list
        The times of the beginning of each interval and the time of the
        end of the last interval; or time_list may already be the output
        format, a list of two-element tuples, in which case it will be
        returned unchanged

    Returns
    -------
    list of two-element tuples
        Each tuple in the list gives the start and end times (float) of an
        interval to be extracted from the input EVENTS table
    """

    if not increment and not time_list:
        raise RuntimeError("Must specify either increment or time_list.")

    if time_list and len(time_list) < 2:
        raise RuntimeError("time_list must have at least two elements.")

    if increment and time_list:
        cosutil.printWarning("Both increment and time_list were specified;"
                             " time_list will be used.")

    if time_list:

        if isinstance(time_list, str):
            time_list = convertToList(time_list)
        else:
            # If time_list is already a list of two-element tuples (or lists),
            # just return it.
            is_a_list_of_tuples = True          # default
            for value in time_list:
                if not isinstance(value, (list, tuple, np.ndarray)):
                    is_a_list_of_tuples = False
                    break
                if len(value) != 2:
                    is_a_list_of_tuples = False
                    break
            if is_a_list_of_tuples:
                return time_list

        # time_list is a list of times; extract the times into a list of
        # two-element tuples.
        new_time_list = []

        if isinstance(time_list[0], str):
            value = time_list[0].lower()
            if value == "start":
                t0 = 0.
            else:
                raise RuntimeError("First element of time_list is '%s'." %
                                   time_list[0])
        else:
            t0 = time_list[0]

        nelem = len(time_list)
        for i in range(1, nelem):
            if isinstance(time_list[i], str):
                value = time_list[i].lower()
                if value == "start":
                    t1 = 0.
                elif value == "stop" or value == "end":
                    t1 = time_col[-1]
                else:
                    raise RuntimeError(
                    "Don't understand time_list[%d] = '%s'." %
                        (i, time_list[i]))
            else:
                t1 = time_list[i]
            if t1 < t0:
                cosutil.printError("time_list = %s" % repr(time_list))
                raise RuntimeError("Values in time_list must be "
                                   "in increasing order.")
            elif t0 == t1:
                continue
            new_time_list.append((t0, t1))
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
                new_time_list.append((t0, t1))
                t0 = t1

    # Remove intervals that are outside the range of the data.
    trimmed_list = []
    for interval in new_time_list:
        if interval[1] > time_col[0] and interval[0] <= time_col[-1]:
            trimmed_list.append(interval)

    return trimmed_list

def convertToList(time_string):
    """Split a string of times on commas and/or blanks.

    Parameters
    ----------
    time_string: str
        String containing times, separated by blanks and/or commas; the
        times may be integers or floats, but "start", "stop" and "end" are
        also allowed

    Returns
    -------
    list of floats
        list of start and/or stop times of intervals, extracted from
        the input string
    """

    if time_string.find(",") >= 0:
        # split the string on commas, then strip whitespace
        words = []
        temp_words = time_string.split(",")
        for word in temp_words:
            words.extend(word.split())
    else:
        # split the string on blanks
        words = time_string.split()

    time_list = []
    for word in words:
        word = word.lower()
        if word == "start" or word == "stop" or word == "end":
            time_list.append(word)
        else:
            try:
                t_value = float(word)
            except ValueError:
                cosutil.printError("Values in time_list must be a number," \
                                   " or 'start', 'stop', or 'end'.")
                raise
            time_list.append(t_value)

    return time_list

def determineSlice(time_col, t0, t1):
    """Find the indices correspond to the start and end times of an interval.

    Parameters
    ----------
    time_col: array_like
        A copy of the TIME column from the input table

    t0: float
        Time at the beginning of an interval

    t1: float
        Time at the end of an interval

    Returns
    -------
    tuple of two integers
        The indices of a slice in time_col that will give the elements
        with times >= t0 and < t1
    """

    return ccos.range(time_col, t0, t1)

def constructOutputName(outroot, file_index, suffix):
    """Construct an output file name.

    For example, if outroot="abc", file_index=3, and suffix="_corrtag_a.fits",
    the output file name will be "abc_3_corrtag_a.fits".

    If a file with the constructed name already exists, however, a new name
    will be tried, a name with "_1" appended after the file_index.  If that
    file also already exists, "_2" will be tried instead of "_1", and so on
    until a name is found that doesn't exist.  For the above example, the
    first such name would be "abc_3_1_corrtag_a.fits".

    Parameters
    ----------
    outroot: str
        Root name with which to construct output file names

    file_index: int
        This number, preceded by "_", will be appended to outroot

    suffix: str
        All of the input file name that follows the root name

    Returns
    -------
    The name to use for the output file
    """

    upper_limit = 10000

    filename = "%s_%d%s" % (outroot, file_index, suffix)
    if os.access(filename, os.R_OK):
        save_filename = filename
        i = 1
        done = False
        while not done:
            filename = "%s_%d_%d%s" % (outroot, file_index, i, suffix)
            if os.access(filename, os.R_OK):
                i += 1
                if i > upper_limit:
                    raise RuntimeError("Output file already exists,"
                          " and upper limit of names has been exceeded.")
            else:
                done = True
        cosutil.printWarning("Output file %s already exists,"
                             " will use name %s instead." %
                             (save_filename, filename))

    return filename

def copyRows(data, ofd, i, j):
    """Copy rows from the input file to an output file.

    Parameters
    ----------
    data: pyfits record array
        The data block of the EVENTS table extension

    ofd: pyfits HDUList object
        The data block of the EVENTS table extension

    i: int
        Number (zero indexed) of the first row to copy

    j: int
        One more than the number (zero indexed) of the last row to copy;
        that is, [i:j] is a slice
    """

    ofd[1].data = data[i:j]

def getGTI(ifd):
    """Find the most up-to-date GTI table in the input file.

    Parameters
    ----------
    ifd: pyfits HDUList object
        The pyfits file handle for the input file

    Returns
    -------
    pyfits BinTableHDU object, or None
        The GTI extension from the input file, or None if there isn't one
    """

    # Find the GTI table with the largest value of EXTVER.
    last_extver = 0                     # initial value
    hdunum = 0
    for i in range(1, len(ifd)):
        hdu = ifd[i]
        extname = hdu.header.get("extname", "MISSING")
        if extname.upper() == "GTI":
            extver = hdu.header.get("extver", 1)
            if extver > last_extver:
                last_extver = extver
                hdunum = i

    if hdunum < 1:
        gti_hdu = None
    else:
        gti_hdu = ifd[hdunum]
    return gti_hdu

def createNewGTI(gti_hdu, t0, t1):
    """Create a GTI table for the output table.

    Parameters
    ----------
    gti_hdu: pyfits BinTableHDU object, or None
        The GTI table from the input file (may be None)

    t0: float
        Time at the start of the interval

    t1: float
        Time at the end of the interval

    Returns
    -------
    pyfits BinTableHDU object
        A GTI table to append to the output file
    """

    if gti_hdu is None or gti_hdu.data is None or len(gti_hdu.data) == 0:
        # No GTI table means the entire exposure is good.
        if gti_hdu is None or gti_hdu.data is None:
            col = []
            col.append(fits.Column(name="START", format="1D", unit="s"))
            col.append(fits.Column(name="STOP", format="1D", unit="s"))
            cd = fits.ColDefs(col)
        else:
            cd = gti_hdu.columns
        out_gti_hdu = fits.BinTableHDU.from_columns(cd, header=gti_hdu.header,
                                                    nrows=1)
        out_start_col = out_gti_hdu.data.field("start")
        out_stop_col = out_gti_hdu.data.field("stop")
        out_start_col[0] = t0
        out_stop_col[0] = t1
        return out_gti_hdu

    cd = gti_hdu.columns

    data = gti_hdu.data
    in_nrows = len(data)

    # columns in the GTI table from the input file
    start_col = data.field("start")
    stop_col = data.field("stop")

    gti = []                    # list of good (start, stop) intervals
    for i in range(in_nrows):
        start = start_col[i]
        stop = stop_col[i]
        if start >= t1 or stop <= t0:
            continue
        start = max(start, t0)
        stop = min(stop, t1)
        if (stop - start) <= 0.:
            continue
        gti.append((start, stop))

    if not gti:
        gti.append((0., 0.))
    out_nrows = len(gti)

    out_gti_hdu = fits.BinTableHDU.from_columns(cd, header=gti_hdu.header,
                                                nrows=out_nrows)
    out_start_col = out_gti_hdu.data.field("start")
    out_stop_col = out_gti_hdu.data.field("stop")
    for i in range(out_nrows):
        (start, stop) = gti[i]
        out_start_col[i] = start
        out_stop_col[i] = stop

    return out_gti_hdu

def getTimeline(ifd):
    """Get the TIMELINE extension (if there is one) from the input file.

    Parameters
    ----------
    ifd: pyfits HDUList object
        The pyfits file handle for the input file.

    Returns
    -------
    pyfits BinTableHDU object, or None
        The TIMELINE extension from the input file, if there is one.
    """

    hdunum = 0
    for i in range(1, len(ifd)):
        hdu = ifd[i]
        extname = hdu.header.get("extname", "MISSING")
        if extname.upper() == "TIMELINE":
            hdunum = i
            break               # assume there's only one

    if hdunum < 1:
        timeline_hdu = None
    else:
        timeline_hdu = ifd[hdunum]
        # Note that this allows timeline_hdu.data to have zero length,
        # which will be checked for elsewhere and is OK.
        if timeline_hdu.data is None:
            timeline_hdu = None

    return timeline_hdu

def createNewTimeline(timeline_hdu, t0, t1):
    """Create a TIMELINE table for the output table.

    Parameters
    ----------
    timeline_hdu: pyfits BinTableHDU object, or None
        The TIMELINE table from the input file (may be None).

    t0: float
        Time at the start of the interval.

    t1: float
        Time at the end of the interval.

    Returns
    -------
    pyfits BinTableHDU object, or None
        A TIMELINE table to append to the output file, or None.  If
        there is no TIMELINE extension in the input file (indicated by
        timeline_hdu being None) or if the time increment is zero or
        negative, None will be returned.  Otherwise, the returned value
        will have the same columns as the input timeline_hdu, but the rows
        will be a subset of timeline_hdu.
    """

    if timeline_hdu is None:
        return None

    cd = timeline_hdu.columns

    in_data = timeline_hdu.data
    if in_data is None:
        in_nrows = 0
    else:
        in_nrows = len(in_data)

    if in_nrows > 0:
        time_col = in_data.field("time").astype(np.float64)
        # "ceil(t1) + 0.1" here is to ensure that the time range
        # (specifically i_end) actually includes all the relevant rows
        # of the input TIMELINE table.
        # This implicitly assumes that the time increment is one second.
        (i_start, i_end)= ccos.range(time_col, t0, math.ceil(t1) + 0.1)
        out_nrows = i_end - i_start
    else:
        out_nrows = 0

    out_timeline_hdu = fits.BinTableHDU.from_columns(cd,
                                                     header=timeline_hdu.header,
                                                     nrows=out_nrows)
    if in_nrows > 0:
        out_data = out_timeline_hdu.data
        i = i_start
        for j in range(out_nrows):
            out_data[j] = in_data[i]
            i += 1

    return out_timeline_hdu

def updateKeywords(info, out_gti_hdu, t0, t1, nevents, ofd):
    """Update keywords in an output file.

    This function adds two HISTORY records to the output primary header and
    updates EXPTIME, EXPTIMEA or EXPTIMEB, NEVENTS, NEVENTSA or NEVENTSB,
    EXPEND and EXPENDJ in the EVENTS extension header.

    Parameters
    ----------
    info: dictionary
        Keywords and values from the input header

    out_gti_hdu: pyfits BinTableHDU object
        Header/data unit for the GTI table for the output file

    t0: float
        Time at the start of the interval

    t1: float
        Time at the end of the interval

    nevents: int
        Number of events in the output EVENTS table

    ofd: pyfits HDUList object
        The pyfits file handle for the output file
    """

    phdr = ofd[0].header
    hdr = ofd[1].header

    filename = os.path.basename(info["input"])          # just the file name
    phdr.add_history("Copied from %s" % filename)
    phdr.add_history("Time slice from input was %.3f to %.3f" % (t0, t1))

    if out_gti_hdu is None:
        exptime = t1 - t0
    else:
        start_col = out_gti_hdu.data.field("start")
        stop_col = out_gti_hdu.data.field("stop")
        exptime = 0.
        n = len(start_col)
        for i in range(n):
            exptime += (stop_col[i] - start_col[i])

    # Modified 2011 May 13 to update exptimea or exptimeb, depending on
    # segment.  Also update nevents and either neventsa or neventsb.
    hdr["exptime"] = exptime
    hdr["nevents"] = nevents
    if info["detector"] == "FUV":
        # first assign default values, so keywords for the "other" segment
        # will have the default
        hdr["exptimea"] = 0.
        hdr["exptimeb"] = 0.
        hdr["neventsa"] = 0
        hdr["neventsb"] = 0
        # "exptimea" or "exptimeb"
        exptime_key = cosutil.segmentSpecificKeyword("exptime",
                                                     info["segment"])
        hdr[exptime_key] = exptime
        nevents_key = "nevents" + info["segment"][-1].lower()
        hdr[nevents_key] = nevents

    expstart = info["expstart"]
    if expstart > 0.:
        expend = expstart + t1/SEC_PER_DAY
        hdr["expend"] = expend
        expend_j = expend + 2400000.5
        hdr["expendj"] = expend_j
