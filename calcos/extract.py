from __future__ import absolute_import, division         # confidence high
import copy
import os
import numpy as np
import astropy.io.fits as fits
from astropy.stats import poisson_conf_interval
from . import cosutil
from . import ccos
from . import dispersion
from . import getinfo
from . import xd_search
from .calcosparam import *       # parameter definitions

def extract1D(input, incounts=None, output=None,
              update_input=True,
              location=None, extrsize=None,
              find_target={"flag": False, "cutoff": None}):
    """Extract 1-D spectrum from 2-D image.

    Parameters
    ----------
    input: str
        Name of either the flat-fielded count-rate image (in which
        case incounts must also be specified) or the corrtag table.

    incounts: str or None
        Name of the file containing the count-rate image, or None if
        input is a corrtag table.

    output: str
        Name of the output file for 1-D extracted spectra.

    update_input: boolean
        True if the input flt and counts files should be modified in-place
        by updating keywords regarding extraction location.

    location: int or float, or list of integers or floats, or None
        The location (or list of three locations for NUV) of the spectrum
        in the cross-dispersion direction, in pixels; this is where the
        spectrum crosses the middle of the detector (index 8192 for FUV,
        512 for NUV).  None means the user did not specify the location.
        If location was specified, that value will be used, regardless
        of the switch in find_target.

    extrsize: int, or list of int, or None
        The height of the extraction box (or list of three heights for NUV)
        in the cross-dispersion direction.  None means the user did not
        specify the extraction height.

    find_target: dictionary
        Keys are "flag" and "cutoff".  flag = True means that we should use
        the location that we find for the target in the cross-dispersion
        direction if the standard deviation (pixels) of the location is
        less than or equal to cutoff (if cutoff is positive).  flag = False
        means we should use the location determined from the wavecal or as
        specified by the user.  find_target["flag"] will locally be set to
        False if location is not None, or if the input is a wavecal.
    """

    cosutil.printIntro("Spectral Extraction")
    names = [("Input", input), ("Incounts", incounts), ("Output", output)]
    cosutil.printFilenames(names)
    cosutil.printMsg("", VERBOSE)

    # Open the input files.
    ifd_e = fits.open(input, mode="copyonwrite")
    if incounts is None:
        ifd_c = None
    else:
        ifd_c = fits.open(incounts, mode="copyonwrite")

    phdr = ifd_e[0].header
    hdr = ifd_e[1].header
    info = getinfo.getGeneralInfo(phdr, hdr)
    switches = getinfo.getSwitchValues(phdr)
    reffiles = getinfo.getRefFileNames(phdr)
    is_wavecal = info["exptype"].find("WAVE") >= 0
    if not is_wavecal and switches["wavecorr"] != "COMPLETE":
        cosutil.printWarning("WAVECORR was not done for " + input)

    local_find_targ = copy.deepcopy(find_target)

    if is_wavecal:
       location = None
       extrsize = None
       local_find_targ["flag"] = False
    if location is None:
        if local_find_targ["flag"]:
            flag = "yes"
        else:
            flag = "no"
        cutoff = local_find_targ["cutoff"]
        cosutil.printMsg("Info:  find-target option = %s" % flag, VERBOSE)
        if find_target["flag"]:
            if cutoff is None or cutoff <= 0.:
                cosutil.printMsg("Info:  cutoff was not specified.",
                                 VERBOSE)
            else:
                cosutil.printMsg("Info:  cutoff = %.4f" % cutoff, VERBOSE)
    else:
        cosutil.printMsg("Info:  Spectrum will be extracted"
                         " at user-specified location.", VERBOSE)
    if extrsize is not None:
        cosutil.printMsg("Info:  User-specified extraction height"
                         " will be used.", VERBOSE)

    # Check data types and lengths (if not scalar), and copy values for

    # location and extrsize to dictionary entries.  Override value of
    # local_find_targ["flag"] (set to False) if location was specified.
    (location, extrsize) = \
        checkLocation(info, location, extrsize, local_find_targ)

    cosutil.printSwitch("X1DCORR", switches)
    cosutil.printMsg("Extraction algorithm = %s" % info["xtrctalg"])
    if info["xtrctalg"] == "TWOZONE":
        cosutil.printRef("TWOZXTAB", reffiles)
        cosutil.printRef("PROFTAB", reffiles)
    else:
        cosutil.printRef("XTRACTAB", reffiles)
    cosutil.printRef("DISPTAB", reffiles)
    cosutil.printSwitch("HELCORR", switches)
    cosutil.printSwitch("BACKCORR", switches)
    cosutil.printSwitch("STATFLAG", switches)
    cosutil.printSwitch("FLUXCORR", switches)
    if switches["fluxcorr"] == "PERFORM":
        cosutil.printRef("fluxtab", reffiles)
        cosutil.printSwitch("TDSCORR", switches)
        if switches["tdscorr"] == "PERFORM":
            cosutil.printRef("tdstab", reffiles)

    # Create the output FITS header/data unit object.
    ofd = fits.HDUList(ifd_e[0])

    # Set the default length of the dispersion axis.
    if info["detector"] == "FUV":
        nelem = FUV_EXTENDED_X
    else:
        nelem = NUV_EXTENDED_X

    if info["npix"] == (0,):
        nrows = 0
    else:
        if info["detector"] == "FUV":
            nrows = FUV_SPECTRA
        else:
            nrows = NUV_SPECTRA
        if ifd_c is not None:
            # get the actual value of naxis (note:  dispaxis is one-indexed)
            dispaxis = max(info["dispaxis"], 1)
            key = "naxis" + str(dispaxis)
            nelem = hdr[key]
    rpt = str(nelem)                            # used for defining columns

    # Define output columns.
    col = []
    col.append(fits.Column(name="SEGMENT", format="4A"))
    col.append(fits.Column(name="EXPTIME", format="1D",
               disp="F8.3", unit="s"))
    col.append(fits.Column(name="NELEM", format="1J", disp="I6"))
    col.append(fits.Column(name="WAVELENGTH", format=rpt+"D",
               unit="angstrom"))
    col.append(fits.Column(name="FLUX", format=rpt+"E",
               unit="erg /s /cm**2 /angstrom"))
    col.append(fits.Column(name="ERROR", format=rpt+"E",
               unit="erg /s /cm**2 /angstrom"))
    col.append(fits.Column(name="ERROR_LOWER", format=rpt+"E",
               unit="erg /s /cm**2 /angstrom"))
    col.append(fits.Column(name="VARIANCE_FLAT", format=rpt+"E"))
    col.append(fits.Column(name="VARIANCE_COUNTS", format=rpt+"E"))
    col.append(fits.Column(name="VARIANCE_BKG", format=rpt+"E"))
    col.append(fits.Column(name="GROSS", format=rpt+"E",
               unit="count /s"))
    col.append(fits.Column(name="GCOUNTS", format=rpt+"E",
               unit="count"))
    col.append(fits.Column(name="NET", format=rpt+"E",
               unit="count /s"))
    # col.append(fits.Column(name="NET_ERROR", format=rpt+"E",        xxx
    #            unit="count /s"))                                      xxx
    col.append(fits.Column(name="BACKGROUND", format=rpt+"E",
               unit="count /s"))
    col.append(fits.Column(name="DQ", format=rpt+"I"))
    col.append(fits.Column(name="DQ_WGT", format=rpt+"E"))
    col.append(fits.Column(name="DQ_OUTER", format=rpt+"I"))
    col.append(fits.Column(name="BACKGROUND_PER_PIXEL", format=rpt+"E",
                           unit="count /s /pixel",))
    col.append(fits.Column(name="NUM_EXTRACT_ROWS", format=rpt+"I"))
    col.append(fits.Column(name="ACTUAL_EE", format=rpt+"D"))
    col.append(fits.Column(name="Y_LOWER_OUTER", format=rpt+"D"))
    col.append(fits.Column(name="Y_UPPER_OUTER", format=rpt+"D"))
    col.append(fits.Column(name="Y_LOWER_INNER", format=rpt+"D"))
    col.append(fits.Column(name="Y_UPPER_INNER", format=rpt+"D"))
    col.append(fits.Column(name="EE_LOWER_OUTER", format=rpt+"D"))
    col.append(fits.Column(name="EE_UPPER_OUTER", format=rpt+"D"))
    col.append(fits.Column(name="EE_LOWER_INNER", format=rpt+"D"))
    col.append(fits.Column(name="EE_UPPER_INNER", format=rpt+"D"))
    cd = fits.ColDefs(col)

    hdu = fits.BinTableHDU.from_columns(cd, header=hdr, nrows=nrows)
    hdu.name = "SCI"
    ofd.append(hdu)

    if nrows > 0:
        if info["detector"] == "FUV":
            segments = [info["segment"]]
        elif info["obstype"] == "IMAGING":
            segments = ["NUVA"]
        else:
            segments = ["NUVA", "NUVB", "NUVC"]
        # Extract the spectrum or spectra.
        doExtract(ifd_e, ifd_c, ofd, nelem,
                  segments, info, switches, reffiles, is_wavecal,
                  location, extrsize, local_find_targ)
        if switches["fluxcorr"] == "PERFORM":
            # Convert net count rate to flux.
            doFluxCorr(ofd, info, reffiles, switches["tdscorr"])
    # Update nrows, in case rows were skipped during 1-D extraction.
    nrows = ofd[1].data.shape[0]

    # Apply heliocentric Doppler correction to the wavelength array.
    if switches["helcorr"] == "PERFORM" or switches["helcorr"] == "COMPLETE":
        wavelength = ofd[1].data.field("WAVELENGTH")
        for row in range(nrows):
            wl_row = wavelength[row]
            wl_row += (wl_row * (-hdr["v_helio"]) / SPEED_OF_LIGHT)
            wavelength[row][:] = wl_row
        phdr["helcorr"] = "COMPLETE"

    # Update the output header.
    ofd[1].header["bitpix"] = 8         # temporary, xxx
    ofd[0].header["nextend"] = 1
    cosutil.updateFilename(ofd[0].header, output)
    if ifd_c is None:                   # ifd_e is a corrtag table
        # Delete table-specific world coordinate system keywords.
        ofd[1].header = cosutil.delCorrtagWCS(ofd[1].header)
    else:                               # ifd_e is an flt image
        # Delete image-specific world coordinate system keywords.
        ofd[1].header = cosutil.imageHeaderToTable(ofd[1].header)
    updateArchiveSearch(ofd)            # update some keywords
    if nrows > 0:
        ofd[0].header["x1dcorr"] = "COMPLETE"
        if switches["backcorr"] == "PERFORM":
            ofd[0].header["backcorr"] = "COMPLETE"
        # FLUXCORR and TDSCORR are updated in doFluxCorr.
    #
    # Remove unwanted columns (comment out this line to retain columns
    # for debug purposes)
    ofd = remove_unwanted_columns(ofd)
    #
    # Add comment for BACKGROUND_PER_PIXEL column
    ofd = add_column_comment(ofd, 'BACKGROUND_PER_PIXEL',
                             'Average background per pixel')
    ofd.writeto(output, output_verify="silentfix")
    del ofd
    ifd_e.close()
    if ifd_c is not None:
        ifd_c.close()

    if update_input and nrows > 0:
        copyKeywordsToInput(output, input, incounts)

    if switches["statflag"] == "PERFORM":
        cosutil.doSpecStat(output)

def remove_unwanted_columns(ofd):
    unwanted_columns = ['EE_LOWER_OUTER', 'EE_LOWER_INNER',
                        'EE_UPPER_INNER', 'EE_UPPER_OUTER']
    newcols = []
    table = ofd[1].data
    columns = table.columns
    for column in columns:
        if column.name in unwanted_columns:
            pass
        else:
            newcols.append(fits.Column(name=column.name,
                                       format=column.format,
                                       unit=column.unit,
                                       disp=column.disp,
                                       array=table[column.name]))
    cd = fits.ColDefs(newcols)
    newhdu = fits.BinTableHDU.from_columns(cd, header=ofd[1].header)
    ofd[1] = newhdu
    return ofd

def add_column_comment(ofd, column_name, comment):
    #
    # columns don't have comments, per se, but you can add a comment to the
    # corresponding TTYPEn keyword
    number = 1
    while (True):
        try:
            keyword = 'TTYPE%s' % str(number)
            if ofd[1].header[keyword] == column_name:
                ofd[1].header.set(keyword, comment=comment)
                break
            else:
                number = number + 1
        except KeyError:
            break
    return ofd

def checkLocation(info, location, extrsize, local_find_targ):
    """Check that location and height were specified correctly.

    Parameters
    ----------
    info: dictionary
        Header keywords and values

    location: integer, float, or array_like
        Location(s) at which to extract spectrum or spectra

    extrsize: integer or array_like
        Extraction height (or heights for NUV)

    local_find_targ: dictionary
        The "flag" value in this dictionary may be modified in-place.  If
        location is None, the flag will not be modified.  If location
        was specified, however, the flag will be set to False.  That is,
        if the user specified both a location for the spectrum and that we
        should search for the location, the location takes precedence.

    Returns
    -------
    tuple of two dictionaries
        Dictionary of locations (key is segment or stripe), dictionary of
        extraction heights (key is segment or stripe).  The entries in the
        dictionaries may be None (i.e. if location is None or extrsize is
        None)
    """

    if location is None:
        if info["detector"] == "FUV":
            location = {info["segment"]: None}
        else:
            location = {"NUVA": None, "NUVB": None, "NUVC": None}
    else:
        local_find_targ["flag"] = False         # override
        if isinstance(location, int) or isinstance(location, float):
            if info["detector"] == "FUV":
                location = {info["segment"]: location}
            else:
                # This doesn't seem like a very useful case.
                location = {"NUVA": location, "NUVB": None, "NUVC": None}
        else:
            try:
                nelem = len(location)
            except TypeError:
                raise TypeError("location must be an int, float, or sequence")
            if info["detector"] == "FUV":
                if nelem == 1:
                    location = {info["segment"]: location[0]}
                else:
                    raise TypeError("for FUV, location may have "
                                    "only one element")
            elif info["detector"] == "NUV":
                if nelem > 3:
                    raise TypeError("location may not have more than "
                                    "three elements")
                segments = ["NUVA", "NUVB", "NUVC"]
                temp = {"NUVA": None, "NUVB": None, "NUVC": None}
                for i in range(nelem):
                    temp[segments[i]] = location[i]
                location = temp

    if extrsize is None:
        if info["detector"] == "FUV":
            extrsize = {info["segment"]: None}
        else:
            extrsize = {"NUVA": None, "NUVB": None, "NUVC": None}
    else:
        if isinstance(extrsize, int):
            if info["detector"] == "FUV":
                extrsize = {info["segment"]: extrsize}
            else:
                extrsize = {"NUVA": extrsize,
                            "NUVB": extrsize,
                            "NUVC": extrsize}
        else:
            try:
                nelem = len(extrsize)
            except TypeError:
                raise TypeError("extrsize must be an integer or sequence")
            if info["detector"] == "FUV":
                if nelem == 1:
                    extrsize = {info["segment"]: extrsize[0]}
                else:
                    raise TypeError("for FUV, extrsize may have "
                                     "only one element")
            elif info["detector"] == "NUV":
                if nelem == 1:
                    extrsize = {"NUVA": extrsize[0],
                                "NUVB": extrsize[0],
                                "NUVC": extrsize[0]}
                else:
                    if nelem > 3:
                        raise TypeError("extrsize may not have more than "
                                        "three elements")
                    segments = ["NUVA", "NUVB", "NUVC"]
                    temp = {"NUVA": None, "NUVB": None, "NUVC": None}
                    for i in range(nelem):
                        temp[segments[i]] = extrsize[i]
                    extrsize = temp

    return (location, extrsize)

def doExtract(ifd_e, ifd_c, ofd, nelem,
              segments, info, switches, reffiles, is_wavecal,
              location, extrsize,
              local_find_targ={"flag": False, "cutoff": None}):
    """Extract either FUV or NUV data.

    This calls a routine to do the extraction for one segment, and it
    assigns the results to one row of the output table.

    Parameters
    ----------
    ifd_e: pyfits ImageHDU object
        Header/data unit for either the effective count-rate image or for
        the corrtag events table

    ifd_c: pyfits ImageHDU object
        Header/data unit for the count-rate image, or None if the input is
        a corrtag events table

    ofd: pyfits HDUList object
        List of header/data units for the output file, modified in-place

    nelem: int
        Number of elements in current segment of output data

    segments: list of strings
        The segment names, one for FUV, three for NUV

    info: dictionary
        Header keywords and values

    switches: dictionary
        Calibration switch values

    reffiles: dictionary
        Reference file names

    is_wavecal: boolean
        True if the observation is a wavecal, based on exptype

    location: dictionary with segment or stripe as key
        Locations of the spectrum in the cross-dispersion direction, in
        pixels; this is where the spectrum crosses the middle of the
        detector (index 8192 for FUV, 512 for NUV).  A key value may be
        None, which means the user did not specify the location.

    extrsize: dictionary with segment or stripe as key
        The height of the extraction box (or list of three heights for NUV)
        in the cross-dispersion direction.  A key value may be None, which
        means the user did not specify the extraction height.

    local_find_targ: dictionary
        Keys are "flag" and "cutoff".  flag = True means that we should use
        the location that we find for the target in the cross-dispersion
        direction if the standard deviation (pixels) of the location is
        less than or equal to cutoff (if cutoff is positive).  flag = False
        means we should use the location determined from the wavecal or as
        specified by the user.
    """

    hdr = ifd_e[1].header
    outdata = ofd[1].data
    try:
        sdqouter = hdr['sdqouter']
    except KeyError:
        cosutil.printWarning("No SDQOUTER keyword, setting to 0")
        sdqouter = 0
    is_corrtag = (ifd_c is None)
    if is_corrtag:              # the input is a corrtag table
        (xi, eta, dq, epsilon) = getColumns(ifd_e, info["detector"])
        if info["detector"] == "FUV":
            axis_height = FUV_Y
            axis_length = FUV_EXTENDED_X
            segment = segments[0]
        else:
            axis_height = NUV_Y
            axis_length = NUV_EXTENDED_X
            segment = "NUVB"
        # populate the DQ array
        # xxx temporary, should be improved
        shift1 = hdr.get("SHIFT1" + segment[-1], 0.)
        shift2 = hdr.get("SHIFT2" + segment[-1], 0.)
        minmax_shift_dict = {}
        minmax_shift_dict[(0, 1024)] = [shift1, shift1, shift2, shift2] # xxx
        minmax_doppler = (0., 0.)       # xxx replace with actual values
        doppler_boundary = 512          # xxx replace with actual value
        dq_array = np.zeros((axis_height,axis_length), dtype=np.int16)
        cosutil.updateDQArray(info, reffiles, dq_array,
                              minmax_shift_dict,
                              minmax_doppler, doppler_boundary, None)

    row = 0
    for segment in segments:

        filter = {"segment": segment,
                  "opt_elem": info["opt_elem"],
                  "cenwave": info["cenwave"],
                  "aperture": info["aperture"]}
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
            # Check that EE boundaries increase monotonically
            lower_outer = xtract_info.field('LOWER_OUTER')[0]
            lower_inner = xtract_info.field('LOWER_INNER')[0]
            upper_inner = xtract_info.field('UPPER_INNER')[0]
            upper_outer = xtract_info.field('UPPER_OUTER')[0]
            if 0 > lower_outer or \
                    lower_outer > lower_inner or \
                    lower_inner > upper_inner or \
                    upper_inner > upper_outer or \
                    upper_outer > 1.0:
                cosutil.printWarning("Zone boundaries invalid:")
                cosutil.printWarning("LOWER_OUTER = %f" % (lower_outer))
                cosutil.printWarning("LOWER_INNER = %f" % (lower_inner))
                cosutil.printWarning("UPPER_INNER = %f" % (upper_inner))
                cosutil.printWarning("UPPER_OUTER = %f" % (upper_outer))
                raise Exception("Invalid EE boundaries in TWOZXTAB reference file")
            proftab_info = cosutil.getTable(reffiles["proftab"], filter)
            if proftab_info is None:
                raise MissingRowError("Missing row in PROFTAB; filter = %s" %
                                      str(filter))

        # Include fpoffset in the filter for disptab.
        filter["fpoffset"] = info["fpoffset"]
        disp_rel = dispersion.Dispersion(reffiles["disptab"], filter, True)
        if not disp_rel.isValid():
            raise MissingRowError("Missing row in DISPTAB; filter = %s" %
                                  str(disp_rel.getFilter()))
        try:
            slope = xtract_info.field("slope")[0]
        except KeyError:
            slope = 0.0

        if is_wavecal:
            dpixel1 = 0.
            key = "shift2" + segment[-1]
            shift2 = hdr.get(key, 0.)
        elif switches["wavecorr"] != "COMPLETE":
            # Without wavecorr, we may need an offset to find the target.
            dpixel1 = 0.
            shift2 = info["life_adj_offset"]
        else:
            key = "dpixel1" + segment[-1]
            dpixel1 = hdr.get(key, 0.)
            shift2 = 0.

        # user_xdisp_locn will be the user-specified location in the
        # cross-dispersion direction (or None, if the user did not specify
        # a value).  xd_locn (assigned later) will be either user_xdisp_locn
        # (if specified), or the value found by searching (if
        # local_find_targ["flag"] is True), or the default value plus shift2.
        if location is None:
            user_xdisp_locn = None
        else:
            user_xdisp_locn = location[segment]
        if extrsize is None:
            user_xdisp_size = None
        else:
            user_xdisp_size = extrsize[segment]

        outdata.field("NELEM")[row] = nelem

        # These are pixel coordinates.
        pixel = np.arange(nelem, dtype=np.float64)

        x_offset = hdr.get("x_offset", 0)

        # Correct for the extra pixels (if any) in the dispersion direction.
        pixel -= x_offset

        pixel += dpixel1                # dpixel1 will be 0 for a wavecal
        wavelength = disp_rel.evalDisp(pixel)
        disp_rel.close()

        # S/N of the flat field
        snr_ff = getSnrFf(switches, reffiles, segment)

        dispaxis = max(info["dispaxis"], 1)
        axis = 2 - dispaxis             # 1 --> 1,  2 --> 0

        # For FUV, the keyword for exposure time depends on segment.
        exptime_key = cosutil.segmentSpecificKeyword("exptime", segment)
        exptime = hdr.get(exptime_key, default=hdr["exptime"])

        if is_corrtag:
            key = "shift1" + segment[-1]
            shift1 = ofd[1].header.get(key, 0.)
            (N_i, ERROR_i, ERROR_LOWER_i, VARIANCE_FLAT_i, VARIANCE_COUNTS_i, VARIANCE_BKG_i,
                 GC_i, GCOUNTS_i, BK_i, DQ_i, DQ_WGT_i, DQ_ALL_i,
                 LOWER_OUTER_i, UPPER_OUTER_i, LOWER_INNER_i, UPPER_INNER_i,
                 ENCLOSED_FRACTION_i, BACKGROUND_PER_ROW_i, EE_LOWER_OUTER_i,
                 EE_LOWER_INNER_i, EE_UPPER_INNER_i, EE_UPPER_OUTER_i
             ) = \
                extractCorrtag(xi, eta, dq, epsilon, dq_array,
                               ofd[1].header, segment, axis_length,
                               x_offset, hdr["sdqflags"], snr_ff,
                               exptime, switches["backcorr"], axis,
                               xtract_info, shift1, shift2,
                               user_xdisp_locn, user_xdisp_size,
                               local_find_targ)
        else:
            if info["xtrctalg"] == 'BOXCAR':
                (N_i, ERROR_i, ERROR_LOWER_i, VARIANCE_FLAT_i, VARIANCE_COUNTS_i, VARIANCE_BKG_i,
                 GC_i, GCOUNTS_i, BK_i,
                 DQ_i, DQ_WGT_i, DQ_ALL_i,
                 LOWER_OUTER_i, UPPER_OUTER_i, LOWER_INNER_i, UPPER_INNER_i,
                 ENCLOSED_FRACTION_i, BACKGROUND_PER_ROW_i, EE_LOWER_OUTER_i,
                 EE_LOWER_INNER_i, EE_UPPER_INNER_i, EE_UPPER_OUTER_i
                 ) = \
                      extractSegmentBoxcar(ifd_e["SCI"].data, ifd_c["SCI"].data,
                                           ifd_e["DQ"].data, ofd[1].header,
                                           segment, x_offset, hdr["sdqflags"],
                                           snr_ff, exptime,
                                           switches["backcorr"], axis,
                                           xtract_info, shift2,
                                           info, wavelength, is_wavecal,
                                           user_xdisp_locn, user_xdisp_size,
                                           local_find_targ)
            elif info["xtrctalg"] == 'TWOZONE':
                (N_i, ERROR_i, ERROR_LOWER_i, VARIANCE_FLAT_i, VARIANCE_COUNTS_i, VARIANCE_BKG_i,
                 GC_i, GCOUNTS_i, BK_i, DQ_i,
                 DQ_WGT_i, DQ_ALL_i,
                 LOWER_OUTER_i, UPPER_OUTER_i, LOWER_INNER_i, UPPER_INNER_i,
                 ENCLOSED_FRACTION_i, BACKGROUND_PER_ROW_i, EE_LOWER_OUTER_i,
                 EE_LOWER_INNER_i, EE_UPPER_INNER_i, EE_UPPER_OUTER_i
                 ) = \
                      extractSegmentTwozone(ifd_e["SCI"].data,
                                            ifd_c["SCI"].data,
                                            ifd_e["DQ"].data, ofd[1].header,
                                            segment, x_offset, hdr["sdqflags"],
                                            sdqouter, snr_ff, exptime,
                                            switches["backcorr"], axis, hdr,
                                            xtract_info, shift2, proftab_info,
                                            info, wavelength, is_wavecal,
                                            user_xdisp_locn, user_xdisp_size,
                                            local_find_targ)
            else:
                cosutil.printMsg("Unknown extraction method, defaulting to", \
                                     " BOXCAR")
                (N_i, ERROR_i, ERROR_LOWER_i, VARIANCE_FLAT_i, VARIANCE_COUNTS_i, VARIANCE_BKG_i,
                 GC_i, GCOUNTS_i, BK_i, DQ_i,
                 DQ_WGT_i, DQ_ALL_i,
                 LOWER_OUTER_i, UPPER_OUTER_i, LOWER_INNER_i, UPPER_INNER_i,
                 ENCLOSED_FRACTION_i, BACKGROUND_PER_ROW_i, EE_LOWER_OUTER_i,
                 EE_LOWER_INNER_i, EE_UPPER_INNER_i, EE_UPPER_OUTER_i
                 ) = \
                      extractSegmentBoxcar(ifd_e["SCI"].data, ifd_c["SCI"].data,
                                           ifd_e["DQ"].data, ofd[1].header,
                                           segment,
                                           x_offset, hdr["sdqflags"], snr_ff,
                                           exptime, switches["backcorr"], axis,
                                           xtract_info, shift2,
                                           info, wavelength, is_wavecal,
                                           user_xdisp_locn, user_xdisp_size,
                                           local_find_targ)
        del xtract_info

        outdata.field("SEGMENT")[row] = segment
        outdata.field("EXPTIME")[row] = exptime
        outdata.field("WAVELENGTH")[row][:] = wavelength.copy()
        outdata.field("FLUX")[row][:] = 0.
        outdata.field("ERROR")[row][:] = ERROR_i.copy()
        outdata.field("ERROR_LOWER")[row][:] = ERROR_LOWER_i.copy()
        outdata.field("VARIANCE_FLAT")[row][:] = VARIANCE_FLAT_i.copy()
        outdata.field("VARIANCE_BKG")[row][:] = VARIANCE_BKG_i.copy()
        outdata.field("GROSS")[row][:] = GC_i.copy()
        outdata.field("GCOUNTS")[row][:] = GCOUNTS_i.copy()
        outdata.field("VARIANCE_COUNTS")[row][:] = VARIANCE_COUNTS_i.copy()
        outdata.field("NET")[row][:] = N_i.copy()
        # outdata.field("NET_ERROR")[row][:] = ERR_i.copy()     xxx
        outdata.field("BACKGROUND")[row][:] = BK_i.copy()
        outdata.field("DQ")[row][:] = DQ_i.copy()
        outdata.field("DQ_WGT")[row][:] = DQ_WGT_i.copy()
        outdata.field("DQ_OUTER")[row][:] = DQ_ALL_i.copy()
        outdata.field("Y_LOWER_OUTER")[row][:] = LOWER_OUTER_i.copy()
        outdata.field("Y_UPPER_OUTER")[row][:] = UPPER_OUTER_i.copy()
        outdata.field("Y_LOWER_INNER")[row][:] = LOWER_INNER_i.copy()
        outdata.field("Y_UPPER_INNER")[row][:] = UPPER_INNER_i.copy()
        outdata.field("ACTUAL_EE")[row][:] = ENCLOSED_FRACTION_i.copy()
        outdata.field("BACKGROUND_PER_PIXEL")[row][:] = \
            BACKGROUND_PER_ROW_i.copy()
        outdata.field("EE_LOWER_OUTER")[row][:] = EE_LOWER_OUTER_i.copy()
        outdata.field("EE_LOWER_INNER")[row][:] = EE_LOWER_INNER_i.copy()
        outdata.field("EE_UPPER_INNER")[row][:] = EE_UPPER_INNER_i.copy()
        outdata.field("EE_UPPER_OUTER")[row][:] = EE_UPPER_OUTER_i.copy()
        NUM_EXTRACT_ROWS = UPPER_OUTER_i - LOWER_OUTER_i + 1
        outdata.field("NUM_EXTRACT_ROWS")[row][:] = NUM_EXTRACT_ROWS.copy()
        row += 1

    # Remove unused rows, if any.
    if row < len(outdata):
        data = outdata[0:row]
        ofd[1].data = data.copy()
        del data

def postargOffset(phdr, dispaxis):
    """Get the offset to shift2 if postarg is non-zero.

    I don't think this should be used, but I'll leave the function here
    for the time being.  If it were to be used, the function value would
    be added to (or subtracted from?) shift2 in doExtract, e.g.:
        shift2 = 0.              # cross-dispersion direction
        shift2 += postargOffset (ifd_e[0].header, hdr["dispaxis"])

    The plate scale should be gotten from a header keyword.
    The sign of the offset needs to be checked.

    Parameters
    ----------
    phdr: pyfits Header object
        Primary header

    dispaxis: int
        Dispersion axis (1 or 2)

    Returns
    -------
    float
        Offset in pixels to be added to cross-dispersion location
    """

    # pixels per arcsecond in the cross-dispersion direction
    plate_scale = {
        "G130M":  9.02,
        "G160M": 10.12,
        "G140L":  9.48,
        "G185M": 41.85,
        "G225M": 41.89,
        "G285M": 41.80,
        "G230L": 42.27}

    if dispaxis == 1:
        postarg_xdisp = phdr.get("postarg2", 0.)
    elif dispaxis == 2:
        postarg_xdisp = phdr.get("postarg1", 0.)
    else:
        return 0.

    opt_elem = phdr["opt_elem"]

    return postarg_xdisp * plate_scale[opt_elem]

def getColumns(ifd_e, detector):
    """Get the appropriate columns from the events table extension.

    The returned columns xi, eta, dq and epsilon are as follows:
        xi is the array of positions in the dispersion direction
        eta is the array of positions in the cross-dispersion direction
        dq is the array of data quality flags
        epsilon is the array of weights (inverse flat field and deadtime
        correction)
    There is one element for each detected photon.

    Parameters
    ----------
    ifd_e: pyfits ImageHDU object
        Header/data unit for the corrtag events table

    detector: str
        Detector name ("FUV" or "NUV")

    Returns
    -------
    tuple of array_like
        Columns from the corrtag table
    """

    data = ifd_e[1].data

    if cosutil.findColumn(data, "xfull"):
        xi = data.field("xfull")
    else:
        xi = data.field("xdopp")
    if cosutil.findColumn(data, "yfull"):
        eta = data.field("yfull")
    else:
        if detector == "FUV":
            eta = data.field("ycorr")
        else:
            eta = data.field("rawy")

    dq = data.field("dq")
    epsilon = data.field("epsilon")

    return (xi, eta, dq, epsilon)

def getSnrFf(switches, reffiles, segment):
    """Get the signal-to-noise ratio of the flat field data.

    If the flat-field correction has been done, this function reads the
    keyword SNR_FF from the appropriate header of the flat field image
    and returns that value; otherwise, this function returns zero.

    Parameters
    ----------
    switches: dictionary
        Calibration switch values

    reffiles: dictionary
        Reference file names

    segment: str
        Segment (or stripe) name

    Returns
    -------
    float
        Signal-to-noise ratio of the flat field
    """

    if switches["flatcorr"] == "COMPLETE":
        fd_flat = fits.open(reffiles["flatfile"], mode="readonly")
        if segment in ["FUVA", "FUVB"]:
            flat_hdr = fd_flat[segment].header
        else:
            flat_hdr = fd_flat[1].header
        snr_ff = flat_hdr.get("snr_ff", 0.)
        fd_flat.close()
        del fd_flat
    else:
        snr_ff = 0.

    return snr_ff

def extractSegmentBoxcar(e_data, c_data, e_dq_data, ofd_header, segment,
                         x_offset, sdqflags, snr_ff,
                         exptime, backcorr, axis,
                         xtract_info, shift2,
                         info, wavelength, is_wavecal,
                         user_xdisp_locn=None, user_xdisp_size=None,
                         find_target={"flag": False, "cutoff": None}):
    """Extract a 1-D spectrum for one segment or stripe.

    This does the actual extraction, returning the results as a tuple.

    An "_ij" suffix indicates a 2-D array; here they will all be sections
    extracted from full images.  An "_i" suffix indicates a 1-D array
    which is the result of summing the 2-D array with the same prefix in
    the cross-dispersion direction.  Variables beginning with a capital
    letter are included in the returned tuple.

      e_i       effective count rate, extracted from ifd_e[1].data
      GC_i      gross count rate, extracted from ifd_c[1].data
      GCOUNTS_i gross counts, extracted from ifd_c[1].data
      BK_i      background count rate
      N_i       net count rate
      eps_i     effective count rate / gross count rate
      ERR_i     error estimate for net count rate
      DQ_i      data quality flags, bitwise OR of input DQ array
      DQ_WGT_i  data quality weight array

    Parameters
    ----------
    e_data: 2-D array
        SCI data from the flt file ('e' for effective count rate)

    c_data: 2-D array
        SCI data from the counts file (count rate)

    e_dq_data: 2-D array
        DQ data from the flt file

    ofd_header: pyfits Header object
        header of the output table (for updating keywords)

    segment: str
        FUVA or FUVB, etc. (only used for updating keywords)

    x_offset: int
        Offset of the detector in the output array

    sdqflags: int
        "Serious" data quality flags

    snr_ff: float
        The signal-to-noise ratio of the flat field reference file (from
        the extension header of the flat field)

    exptime: float
        Exposure time (seconds), from the (corrected) header keyword

    backcorr: int
        "PERFORM" if background subtraction is to be done

    axis: int
        The dispersion axis, 0 (Y) or 1 (X)

    xtract_info: pyfits record object
        One row of the xtractab

    shift2: float
        Offset in the cross-dispersion direction.  This should be zero
        except in two cases, a wavecal exposure or a science exposure
        without a wavecal.  Otherwise, the offset in XD should already
        have been taken into account when binning to the flt and counts
        images.

    info: dictionary
        Header keywords and values

    wavelength: array_like
        Wavelength at each pixel (needed if find_target["flag"] is
        True)

    is_wavecal: boolean
        True if the observation is a wavecal, based on exptype

    user_xdisp_locn: int or float, or None if not specified
        User-specified location in cross-dispersion direction

    user_xdisp_size: int, or None if not specified
        User-specified height of extraction box

    find_target: dictionary
        Keys are "flag" and "cutoff".  flag = True means that we should use
        the location that we find for the target in the cross-dispersion
        direction if the standard deviation (pixels) of the location is
        less than or equal to cutoff (if cutoff is positive).  flag = False
        means we should use the location determined from the wavecal or as
        specified by the user.  find_target["flag"] will locally be set to
        False if the cross-dispersion location was not found.

    Returns
    -------
    tuple of seven 1-D arrays
        net count rate, error estimate, gross count rate, gross counts,
        background count rate, data quality array, data quality weight
        array
    """

    local_find_targ = copy.deepcopy(find_target)

    try:
        slope           = xtract_info.field("slope")[0]
    except KeyError:
        slope = 0.0
    b_spec          = xtract_info.field("b_spec")[0]    # may be changed below
    extr_height     = xtract_info.field("height")[0]    # see user_xdisp_size
    b_bkg1          = xtract_info.field("b_bkg1")[0]
    b_bkg2          = xtract_info.field("b_bkg2")[0]
    if cosutil.findColumn(xtract_info, "b_hgt1"):
        bkg_height1  = xtract_info.field("b_hgt1")[0]
        bkg_height2  = xtract_info.field("b_hgt2")[0]
    else:
        bkg_height1  = xtract_info.field("bheight")[0]
        bkg_height2  = bkg_height1
    bkg_smooth      = xtract_info.field("bwidth")[0]

    axis_length = e_data.shape[axis]

    offset_to_middle = slope * (axis_length // 2 - x_offset)
    # nominal location of spectrum, where it crosses the middle of the
    # flt or counts image
    xd_nominal = b_spec + shift2 + offset_to_middle

    if is_wavecal:
        xd_offset = -999.       # offset in the cross-dispersion direction
        found_locn_sigma = 999.
    else:
        # Search for the target spectrum.
        (xd_offset, found_locn, found_locn_sigma, fwhm) = \
                xd_search.xdSearch(e_data,
                                   e_dq_data, wavelength,
                                   axis, slope, b_spec+shift2,
                                   x_offset, info["detector"])
        # The value of xd_offset returned by xdSearch is the offset from
        # b_spec + shift2, but we need the offset from b_spec.  Note,
        # however, that shift2 should be zero unless there was no wavecal.
        xd_offset += shift2

        if found_locn is None:
            xd_offset = 0.
            # turn off for this segment/stripe
            local_find_targ["flag"] = False
            message = "%s spectrum was not found; nominal y = %.2f" % \
                        (segment, xd_nominal)
        else:
            # offset from found location to nominal location
            message = "%s spectrum was found at y = %.2f" \
                      " vs. nominal y = %.2f" % \
                                (segment,
                                 found_locn + offset_to_middle,
                                 xd_nominal)
        cosutil.printMsg(message, VERBOSE)
        msg1 = "error estimate for y location = %.2f, FWHM = " % \
               found_locn_sigma
        if isinstance(fwhm, int):
            msg2 = "%d" % fwhm
        else:
            msg2 = "%.2f" % fwhm
        cosutil.printContinuation(msg1 + msg2)

    # b_spec and xd_locn are either the user-specified value (if it was
    # specified), or the location where the spectrum was found, or the
    # nominal location based on the xtractab and the wavecal.
    # b_spec is where the spectrum crosses the left edge of the detector
    # (X = x_offset),
    # and xd_locn is where the spectrum crosses the middle of the array.
    if user_xdisp_locn is None:
        use_found_location = local_find_targ["flag"]
        if local_find_targ["cutoff"] is not None and \
           local_find_targ["cutoff"] > 0. and \
           found_locn_sigma > local_find_targ["cutoff"]:
            use_found_location = False
            cosutil.printMsg("%s sigma = %.2f of found location"
                             " is higher than cutoff = %.2f." %
                             (segment, found_locn_sigma,
                              local_find_targ["cutoff"]),
                             VERBOSE)
        if use_found_location:
            b_spec = found_locn
            xd_locn = found_locn + offset_to_middle
        else:
            # add the shift to the nominal location; assign a value to xd_locn
            # (which will be used to update a header keyword)
            b_spec += shift2
            xd_locn = b_spec + slope * (axis_length // 2 - x_offset)
            b_bkg1 += shift2
            b_bkg2 += shift2
    else:
        # use the user-specified value, but convert to b_spec, the Y location
        # of the spectrum at X = x_offset
        b_spec = user_xdisp_locn - slope * (axis_length // 2 - x_offset)
        xd_locn = user_xdisp_locn
    cosutil.printMsg("Spectrum will be extracted at y = %.2f" % xd_locn,
                     VERBOSE)

    if user_xdisp_size is not None:
        extr_height = user_xdisp_size   # use the user-specified value

    # Compute the data quality and data quality weight arrays.
    DQ_i = np.zeros(axis_length, dtype=np.int16)
    if e_dq_data is not None:

        # Get data quality flags within extraction region.
        dq_ij = np.zeros((extr_height, axis_length), dtype=np.int16)
        ccos.extractband(e_dq_data, axis, slope, b_spec, x_offset, dq_ij)
        # For each i, DQ_i[i] will be the bitwise OR of dq_ij[:,i].
        ccos.dq_or(dq_ij, DQ_i)

        # In bad_ij and bad_i, 0 means OK and 1 means bad
        bad_ij = np.zeros((extr_height, axis_length), dtype=np.int32)
        bad_ij[:,:] = np.where(np.bitwise_and(dq_ij, sdqflags), 1, 0)
        bad_i = bad_ij.sum(axis=0)
        # Any bad pixel in extraction region?  DQ_WGT is a weight,
        # so 0 is bad and 1 is good.
        DQ_WGT_i = np.where(bad_i > 0, 0., 1.)
        del dq_ij, bad_ij, bad_i
    else:
        DQ_WGT_i = np.ones(axis_length, dtype=np.float32)

    e_ij = np.zeros((extr_height, axis_length), dtype=np.float32)
    ccos.extractband(e_data, axis, slope, b_spec, x_offset, e_ij)

    GC_ij = np.zeros((extr_height, axis_length), dtype=np.float32)
    ccos.extractband(c_data, axis, slope, b_spec, x_offset, GC_ij)

    e_i  = e_ij.sum(axis=0, dtype=np.float64)
    GC_i = GC_ij.sum(axis=0, dtype=np.float64)
    GCOUNTS_i = GC_i * exptime          # gross counts (not count rate)

    eps_i = e_i / np.where(GC_i <= 0., 1., GC_i)
    # default value when there are no counts
    eps_i = np.where(e_i == 0., 1., eps_i)
    del e_ij, e_i

    bkg_norm = float(extr_height) / (float(bkg_height1 + bkg_height2))
    if backcorr == "PERFORM":
        BK1_ij = np.zeros((bkg_height1, axis_length), dtype=np.float32)
        dq1_ij = np.zeros((bkg_height1, axis_length), dtype=np.int16)
        BK2_ij = np.zeros((bkg_height2, axis_length), dtype=np.float32)
        dq2_ij = np.zeros((bkg_height2, axis_length), dtype=np.int16)
        # Get the background data from the counts image.
        ccos.extractband(c_data, axis, slope, b_bkg1, x_offset, BK1_ij)
        ccos.extractband(c_data, axis, slope, b_bkg2, x_offset, BK2_ij)
        original_BK_i = BK1_ij.sum(axis=0, dtype=np.float64) + \
            BK2_ij.sum(axis=0, dtype=np.float64)
        # Get the data quality array from the flt file.
        ccos.extractband(e_dq_data, axis, slope, b_bkg1, x_offset, dq1_ij)
        ccos.extractband(e_dq_data, axis, slope, b_bkg2, x_offset, dq2_ij)
        good1_ij = dq1_ij.copy()
        good2_ij = dq2_ij.copy()
        # In good[12]_ij, 1 means OK and 0 means bad.
        good1_ij[:,:] = np.where(np.bitwise_and(dq1_ij, sdqflags), 0, 1)
        good2_ij[:,:] = np.where(np.bitwise_and(dq2_ij, sdqflags), 0, 1)
        # Use the good[12]_ij arrays as a mask to exclude bad data in the
        # background regions.
        BK1_ij *= good1_ij
        BK2_ij *= good2_ij
        BK_i = BK1_ij.sum(axis=0, dtype=np.float64) + \
            BK2_ij.sum(axis=0, dtype=np.float64)

        # The sum along axis=0 gives the number of good pixels in each column.
        # Use this sum to correct (rescale) the background to account for
        # pixels that are flagged as bad.
        good_i = good1_ij.sum(axis=0, dtype=np.float64) + \
                 good2_ij.sum(axis=0, dtype=np.float64)
        # If good_i is zero, the background will also be zero, so it doesn't
        # matter what we set good_i to as long as it's not zero (we're going
        # to divide by it).
        good_i_div = np.where(good_i > 0., good_i, 1.)
        # Correct for regions excluded because they're flagged as bad.
        BK_i *= (float(bkg_height1 + bkg_height2)) / good_i_div
        # Scale the background to the spectral extraction height.
        BK_i *= bkg_norm
        original_BK_i *= bkg_norm
        # Restore BK_i where all background pixels were flagged as bad.
        BK_i[:] = np.where(good_i > 0., BK_i, original_BK_i)
        if x_offset > 0:
            # assumes x_offset only for NUV
            key = "shift1" + segment[-1].lower()
            i = x_offset - ofd_header.get(key, 0.)
            i = int(round(i))
            j = i + NUV_X       # assumes x_offset only for NUV
            i = max(i, 0)
            j = min(j, axis_length)     # upper limit of a slice
        else:
            (i, j) = (0, axis_length)
        (i, j) = excludeAllBad(good_i, i, j)
        temp_bk = BK_i[i:j].copy().astype(np.float32)
        ccos.smoothbkg(temp_bk, bkg_smooth)
        BK_i[i:j] = temp_bk.copy()
        del temp_bk
    else:
        BK_i = np.zeros(axis_length, dtype=np.float32)
    # The error in the counts is the sum in quadrature of 3 terms
    N_i = eps_i * (GC_i - BK_i)

    if snr_ff > 0.:
        VARIANCE_FLAT_i = (N_i * exptime / (extr_height * snr_ff))**2
    else:
        VARIANCE_FLAT_i = N_i * 0.
    VARIANCE_COUNTS_i = eps_i**2 * GC_i * exptime
    temp_val = BK_i * exptime * (bkg_norm / float(bkg_smooth))
    VARIANCE_BKG_i = eps_i * eps_i * temp_val

    variance_i = VARIANCE_FLAT_i + VARIANCE_COUNTS_i + VARIANCE_BKG_i
    # Use the frequentist option of the astropy Poisson confidence interval function
    # to calculate errors
    ERROR_LOWER_i, ERROR_i = cosutil.errFrequentist(variance_i)
    # ERR_i is the error in the count RATE
    if exptime > 0.:
        ERROR_LOWER_i /= exptime
        ERROR_i /= exptime
    else:
        ERROR_LOWER_i = N_i * 0.
        ERROR_i = N_i * 0.

    updateExtractionKeywords(ofd_header, segment,
                             slope, extr_height,
                             xd_nominal, xd_locn, found_locn_sigma, xd_offset,
                             b_bkg1, b_bkg2, bkg_height1, bkg_height2)
    #
    # Compute the 'extended' quantities
    DQ_ALL_i = DQ_i
    #
    # Make the INDEXes all vectors that follow the slope
    half_height = extr_height // 2
    y0 = (b_spec - half_height) + slope * np.arange(float(axis_length))
    LOWER_OUTER_INDEX_i = (y0 + 0.5).astype(int)
    UPPER_OUTER_INDEX_i = LOWER_OUTER_INDEX_i + extr_height - 1
    LOWER_INNER_INDEX_i = LOWER_OUTER_INDEX_i.copy()
    UPPER_INNER_INDEX_i = UPPER_OUTER_INDEX_i.copy()
    ENCLOSED_FRACTION_i = N_i*0.0 + 1.0
    AV_E_BKG_i = BK_i / float(extr_height)
    LOWER_OUTER_VALUE_i = N_i*0.0 + 0.0
    LOWER_INNER_VALUE_i = N_i*0.0 + 0.0
    UPPER_INNER_VALUE_i = N_i*0.0 + 1.0
    UPPER_OUTER_VALUE_i = N_i*0.0 + 1.0

    return (N_i, ERROR_i, ERROR_LOWER_i, VARIANCE_FLAT_i, VARIANCE_COUNTS_i, VARIANCE_BKG_i,
            GC_i, GCOUNTS_i, BK_i, DQ_i, DQ_WGT_i,
            DQ_ALL_i, LOWER_OUTER_INDEX_i, UPPER_OUTER_INDEX_i,
            LOWER_INNER_INDEX_i, UPPER_INNER_INDEX_i,
            ENCLOSED_FRACTION_i, AV_E_BKG_i,
            LOWER_OUTER_VALUE_i, LOWER_INNER_VALUE_i,
            UPPER_INNER_VALUE_i, UPPER_OUTER_VALUE_i)

def excludeAllBad(good_i, i, j):
    """Exclude endpoints of BK_i where all pixels are flagged as bad.

    Parameters
    ----------
    good_i: array_like
        1-D array of floats, one for each element along the dispersion
        axis.  The values are 0. or 1., depending on the data quality array
        in the background regions.  If any element in the DQ array in
        column i (in either background region) indicates a bad pixel,
        good_i[i] will be 0 (bad); otherwise, it will be 1 (good).

    i: int
        Lower limit of a slice, i:j excludes the X_OFFSET (as shifted
        depending on the wavecal offset).

    j: int
        Upper limit of a slice.

    Returns
    -------
    tuple of two integers
        New values for i and j, such that good_i[i] and good_i[j-1]
        are both > 0 (if there are any such indices)
    """

    nelem = len(good_i)

    done = False
    ip = i
    while not done:
        if ip >= nelem:
            break
        if good_i[ip] > 0.:
            done = True
            break
        ip += 1
    if not done:        # no point not flagged as bad
        ip = i

    done = False
    # i:j is a slice, so start looking for good data at j-1
    jp = j - 1
    while not done:
        if jp < ip:
            break
        if good_i[jp] > 0.:
            done = True
            break
        jp -= 1
    if not done:
        jp = j
    jp += 1             # jp is the upper limit of a slice

    return (ip, jp)

def extractSegmentTwozone(e_data, c_data, e_dq_data, ofd_header, segment,
                          x_offset, sdqflags, sdqouter, snr_ff,
                          exptime, backcorr, axis, hdr,
                          xtract_info, shift2, proftab_info,
                          info, wavelength, is_wavecal,
                          user_xdisp_locn=None, user_xdisp_size=None,
                          find_target={"flag": False, "cutoff": None}):
    """Do the two-zone extraction
    
     This does the actual extraction, returning the results as a tuple.

    An "_ij" suffix indicates a 2-D array; here they will all be sections
    extracted from full images.  An "_i" suffix indicates a 1-D array
    which is the result of summing the 2-D array with the same prefix in
    the cross-dispersion direction.  Variables beginning with a capital
    letter are included in the returned tuple.
      N_i       net count rate
      ERR_i     error estimate for net count rate
      GC_i      gross count rate, extracted from ifd_c[1].data
      GCOUNTS_i gross counts, extracted from ifd_c[1].data
      SUMMED_BACKGROUND_i  total counts in the background
      DQ_i      data quality flags, bitwise OR of input DQ array in the
                inner zone
      DQ_WGT_i  data quality weight array
      DQ_ALL_i  data quality flags over the whole extraction region
      LOWER_OUTER_INDEX_i
      LOWER_INNER_INDEX_i
      UPPER_INNER_INDEX_i
      UPPER_OUTER_INDEX_i Zone boundaries in the data
      ENCLOSED_FRACTION_i   Actual fraction of flux within outer boundaries
      AV_E_BKG_i     Average background per pixel
      LOWER_OUTER_VALUE_i   Fraction of flux enclosed at and above 
                            row lower_outer_index
      LOWER_INNER_VALUE_i   Fraction of flux enclosed at and above
                            row lower_inner_index
      UPPER_INNER_VALUE_i   Fraction of flux enclosed at and above
                            row upper_inner_index
      UPPER_OUTER_VALUE_i   Fraction of flux enclosed at and above
                            row upper_outer_index
    Parameters
    ----------
    e_data: 2-D array
        SCI data from the flt file ('e' for effective count rate)

    c_data: 2-D array
        SCI data from the counts file (count rate)

    e_dq_data: 2-D array
        DQ data from the flt file

    ofd_header: pyfits Header object
        header of the output table (for updating keywords)

    segment: str
        FUVA or FUVB, etc. (only used for updating keywords)

    x_offset: int
        Offset of the detector in the output array

    sdqflags: int
        "Serious" data quality flags

    sdqouter: int
        "Serious" data quality flags for the outer extraction regions

    snr_ff: float
        The signal-to-noise ratio of the flat field reference file (from
        the extension header of the flat field)

    exptime: float
        Exposure time (seconds), from the (corrected) header keyword

    backcorr: int
        "PERFORM" if background subtraction is to be done

    axis: int
        The dispersion axis, 0 (Y) or 1 (X)

    hdr: pyfits Header object
        Extension header of the flt file

    xtract_info: pyfits record object
        One row of the xtractab

    shift2: float
        Offset in the cross-dispersion direction.  This should be zero
        except in two cases, a wavecal exposure or a science exposure
        without a wavecal.  Otherwise, the offset in XD should already
        have been taken into account when binning to the flt and counts
        images.

    proftab_info: pyfits record array
        Required row of the PROFTAB reference file
    info: dictionary
        Header keywords and values

    wavelength: array_like
        Wavelength at each pixel (needed if find_target["flag"] is
        True)

    is_wavecal: boolean
        True if the observation is a wavecal, based on exptype

    user_xdisp_locn: int or float, or None if not specified
        User-specified location in cross-dispersion direction

    user_xdisp_size: int, or None if not specified
        User-specified height of extraction box

    find_target: dictionary
        Keys are "flag" and "cutoff".  flag = True means that we should use
        the location that we find for the target in the cross-dispersion
        direction if the standard deviation (pixels) of the location is
        less than or equal to cutoff (if cutoff is positive).  flag = False
        means we should use the location determined from the wavecal or as
        specified by the user.  find_target["flag"] will locally be set to
        False if the cross-dispersion location was not found.

    Returns
    -------
    tuple of eighteen 1-D arrays
        net count rate, error estimate, gross count rate, gross counts,
        summed background count rate, data quality array (from inner zone only),
        data quality weight array, data quality for inner and outer zones,
        array of indices for lower outer zone boundary,
        array of indices for upper outer zone boundary,
        array of indices for lower inner zone boundary,
        array of indices for upper inner zone boundary,
        fraction of flux enclosed by outer boundaries,
        average background per pixel,
        fraction of flux enclosed between lower outer zone boundary and
        lower aperture boundary,
        fraction of flux enclosed between lower inner zone boundary and
        lower aperture boundary,
        fraction of flux enclosed between upper inner zone boundary and
        lower aperture boundary,
        fraction of flux enclosed between upper outer zone boundary and
        lower aperture boundary
    """
    cosutil.printMsg("Two-zone extraction method")
    #
    # Check that the sdqouter dq value is in sdqflags, and print a warning
    # if it isn't
    sdqflags_ok = (sdqflags & sdqouter) == sdqouter
    if not sdqflags_ok:
        cosutil.printWarning("SDQOUTER (%d) is not in SDQFLAGS (%d)" % \
                                 (sdqouter, sdqflags))
    #
    # Get the profile array from the PROFTAB reference file record
    profile_ij = proftab_info["profile"][0]
    nrows, ncols = profile_ij.shape
    centroid = getProfileCentroid(hdr, segment)
    if centroid < 0.0:
        # Centroid keyword is still < 0, so ALGNCORR was probably skipped
        cosutil.printWarning("Starting centroid from SP_LOC keyword = %f" % \
                                 centroid)
        cosutil.printWarning("Using b_spec from xtractab/twozxtab as centroid")
        centroid = xtract_info.field("b_spec")[0]
    cosutil.printMsg("Using profile centroid of %f" % (centroid))
    row_0 = proftab_info["ROW_0"][0]
    refcentroid = centroid - row_0
    if backcorr == "PERFORM":
        #
        # Dermine background for reference profile
        # Need a subarray of the original dq array of the same shape as the
        # profile
        cosutil.printMsg("Calculating background for reference profile")
        dq_profile = e_dq_data[int(row_0):int(row_0+nrows)]
        ref_background_i, nrows = getBackground(profile_ij, dq_profile,
                                                xtract_info,
                                                refcentroid, sdqflags)
        #
        # Background subtract the profile.
        profile_ij = profile_ij - ref_background_i
    #
    # Get the percentiles for the extraction zone boundaries
    p1, p2, p3, p4 = getPercentiles(xtract_info)
    cosutil.printMsg("Using extraction percentiles of:")
    cosutil.printMsg("%f and %f (outer region)" % (100*p1, 100*p4))
    cosutil.printMsg("and %f and %f (inner region)" % (100*p2, 100*p3))
    #
    # Normalize the profiles to 1.00 within the area defined by the xtractab
    normalized_profile_ij = NormalizeProfile(profile_ij, xtract_info,
                                             refcentroid)
    height = xtract_info.field("height")[0]
    rowstart = int(round(refcentroid)) - height // 2
    rowstop = int(round(refcentroid)) + height // 2
    cumulative_profile = np.cumsum(normalized_profile_ij[rowstart:rowstop+1],
                                   axis=0, dtype=np.float64)
    cumrows, cumcols = cumulative_profile.shape
    (LOWER_OUTER_INDEX_i, LOWER_INNER_INDEX_i,
     UPPER_INNER_INDEX_i, UPPER_OUTER_INDEX_i,
     ENCLOSED_FRACTION_i, LOWER_OUTER_VALUE_i,
     LOWER_INNER_VALUE_i, UPPER_INNER_VALUE_i,
     UPPER_OUTER_VALUE_i) = getPercentileVectors(cumulative_profile, p1, p2,
                                                 p3, p4)
    #
    # Extend the ends so that the regions where there are no counts have the
    # same values as the first or last good columns
    goodcolumns = np.where(UPPER_OUTER_INDEX_i != 0)
    firstgood = goodcolumns[0][0]
    lastgood = goodcolumns[0][-1]
    LOWER_OUTER_INDEX_i[:firstgood] = LOWER_OUTER_INDEX_i[firstgood]
    LOWER_OUTER_INDEX_i[lastgood+1:] = LOWER_OUTER_INDEX_i[lastgood]
    LOWER_INNER_INDEX_i[:firstgood] = LOWER_INNER_INDEX_i[firstgood]
    LOWER_INNER_INDEX_i[lastgood+1:] = LOWER_INNER_INDEX_i[lastgood]
    UPPER_INNER_INDEX_i[:firstgood] = UPPER_INNER_INDEX_i[firstgood]
    UPPER_INNER_INDEX_i[lastgood+1:] = UPPER_INNER_INDEX_i[lastgood]
    UPPER_OUTER_INDEX_i[:firstgood] = UPPER_OUTER_INDEX_i[firstgood]
    UPPER_OUTER_INDEX_i[lastgood+1:] = UPPER_OUTER_INDEX_i[lastgood]
    LOWER_OUTER_INDEX_i = LOWER_OUTER_INDEX_i + row_0 + rowstart
    LOWER_INNER_INDEX_i = LOWER_INNER_INDEX_i + row_0 + rowstart
    UPPER_INNER_INDEX_i = UPPER_INNER_INDEX_i + row_0 + rowstart
    UPPER_OUTER_INDEX_i = UPPER_OUTER_INDEX_i + row_0 + rowstart
    #
    # Now work on the science data
    nrows, ncols = e_data.shape
    # First need to calculate background vector
    bkg_smooth = xtract_info.field("bwidth")[0]
    if backcorr == "PERFORM":
        #
        # Determine background regions for science data
        #
        AV_E_BKG_i, nrows_e_bkg_i = getBackground(e_data, e_dq_data,
                                                  xtract_info, centroid,
                                                  sdqflags)
        #
        # This for the calculation of the error
        av_c_bkg_i, nrows_c_bkg_i = getBackground(c_data, e_dq_data,
                                                  xtract_info, centroid,
                                                  sdqflags)
    else:
        AV_E_BKG_i = np.zeros(ncols, dtype=np.float32)
        av_c_bkg_i = np.zeros(ncols, dtype=np.float32)
    e_data_sub = e_data - AV_E_BKG_i
    height = xtract_info.field("height")[0]
    rowstart = int(round(centroid)) - height // 2
    rowstop = int(round(centroid)) + height // 2
    #
    # Initialize arrays
    total_ecounts = np.zeros(ncols, dtype=np.float32)
    total_ccounts = np.zeros(ncols, dtype=np.float32)
    DQ_i = np.zeros(ncols, dtype=np.int16)
    DQ_ALL_i = np.zeros(ncols, dtype=np.int16)
    DQ_WGT_i = np.ones(ncols, dtype=np.float32)
    extr_height_i = np.ones(ncols, dtype=np.float32)
    bad_i = np.zeros(ncols, dtype=np.int32)
    lowerbad_i = np.zeros(ncols, dtype=np.int32)
    upperbad_i = np.zeros(ncols, dtype=np.int32)

    #
    # Loop over columns
    for column in range(ncols):
        if UPPER_OUTER_INDEX_i[column] > LOWER_OUTER_INDEX_i[column]:
            lowerstart = LOWER_OUTER_INDEX_i[column]
            lowerstop = LOWER_INNER_INDEX_i[column]
            lower_ecounts = e_data_sub[int(lowerstart):int(lowerstop),
                                       column].sum(dtype=np.float64)
            lower_ccounts = c_data[int(lowerstart):int(lowerstop),
                                   column].sum(dtype=np.float64)
            lowerdq = e_dq_data[int(lowerstart):int(lowerstop), column]
            upperstart = UPPER_INNER_INDEX_i[column] + 1
            upperstop = UPPER_OUTER_INDEX_i[column] + 1
            upper_ecounts = e_data_sub[int(upperstart):int(upperstop),
                                       column].sum(dtype=np.float64)
            upper_ccounts = c_data[int(upperstart):int(upperstop),
                                   column].sum(dtype=np.float64)
            upperdq = e_dq_data[int(upperstart):int(upperstop), column]
            innerstart = lowerstop
            innerstop = upperstart
            inner_ecounts = e_data_sub[int(innerstart):int(innerstop),
                                       column].sum(dtype=np.float64)
            inner_ccounts = c_data[int(innerstart):int(innerstop),
                                   column].sum(dtype=np.float64)
            innerdq = e_dq_data[int(innerstart):int(innerstop), column]
            outerstart = lowerstart
            outerstop = upperstop
            outerdq = e_dq_data[int(outerstart):int(outerstop), column]
            bad_i[column] = np.where(np.bitwise_and(innerdq, sdqflags), 1,
                                     0).sum()
            DQ_WGT_i[column] = 1.0
            if bad_i[column] > 0.0:
                DQ_WGT_i[column] = 0.0
            sdqlower = np.bitwise_and(lowerdq, sdqouter)
            lowerbad_i[column] = np.where(sdqlower, 1, 0).sum()
            if lowerbad_i[column] > 0.0:
                DQ_WGT_i[column] = 0.0
            sdqupper = np.bitwise_and(upperdq, sdqouter)
            upperbad_i[column] = np.where(sdqupper, 1, 0).sum()
            if upperbad_i[column] > 0.0:
                DQ_WGT_i[column] = 0.0
            DQ_i[column] = bitwise_or_vector(innerdq) | \
                bitwise_or_vector(sdqlower) | \
                bitwise_or_vector(sdqupper)
            DQ_ALL_i[column] = bitwise_or_vector(outerdq)
            total_ecounts[column] = lower_ecounts + upper_ecounts + \
                inner_ecounts
            total_ccounts[column] = lower_ccounts + upper_ccounts + \
                inner_ccounts
            extr_height_i[column] = UPPER_OUTER_INDEX_i[column] - \
                LOWER_OUTER_INDEX_i[column] + 1
            if ENCLOSED_FRACTION_i[column] != 0.0:
                total_ecounts[column] = total_ecounts[column] / \
                    ENCLOSED_FRACTION_i[column]
            else:
                if total_ecounts[column] != 0.0:
                    total_ecounts[column] = 0.0
        else:
            DQ_i[column] = DQ_PIXEL_OUT_OF_BOUNDS
            DQ_ALL_i[column] = DQ_PIXEL_OUT_OF_BOUNDS
            DQ_WGT_i[column] = 0.0
    N_i = total_ecounts
    goodcolumns = np.where(nrows_c_bkg_i > 0)
    DQ_WGT_i = np.where(nrows_c_bkg_i > 0, DQ_WGT_i, 0.0)
    flat_correction = total_ecounts / np.where(total_ccounts <= 0.0, 1.0,
                                               total_ccounts)
    flat_correction = np.where(flat_correction == 0.0, 1.0, flat_correction)
#
# Now calculate the error
    VARIANCE_FLAT_i = np.zeros(ncols, dtype=np.float32)
    if snr_ff > 0.0:
        VARIANCE_FLAT_i[goodcolumns] = (N_i[goodcolumns] * exptime / (extr_height_i[goodcolumns] * snr_ff))**2
    else:
        VARIANCE_FLAT_i = N_i * 0.0
    VARIANCE_COUNTS_i = np.zeros(ncols, dtype=np.float32)
    VARIANCE_BKG_i = np.zeros(ncols, dtype=np.float32)
    VARIANCE_COUNTS_i[goodcolumns] = (flat_correction[goodcolumns])**2 * exptime \
        * total_ccounts[goodcolumns]
    VARIANCE_BKG_i[goodcolumns] = (flat_correction[goodcolumns])**2 * exptime \
        * av_c_bkg_i[goodcolumns] * (extr_height_i[goodcolumns])**2 \
               / (nrows_c_bkg_i[goodcolumns] * bkg_smooth)
    if exptime > 0.0:
        # Use the frequentist option of the astropy Poisson confidence interval function
        # to calculate errors
        VARIANCE_i = VARIANCE_FLAT_i + VARIANCE_COUNTS_i + VARIANCE_BKG_i
        ERROR_LOWER_i, ERROR_i = cosutil.errFrequentist(VARIANCE_i)
        ERROR_LOWER_i /= exptime
        ERROR_i /= exptime
    else:
        ERROR_LOWER_i = N_i * 0.0
        ERROR_i = N_i * 0.0
    SUMMED_BACKGROUND_i = AV_E_BKG_i * extr_height_i
    GC_i = total_ccounts
    GCOUNTS_i = GC_i * exptime
    key = "SP_ERR_" + segment[-1]
    try:
        cent_err = hdr[key]
    except KeyError:
        cosutil.printWarning("CENT_ERR keyword not found, setting to -999.0")
        cent_err = -999.0
        hdr[key] = cent_err
    key = "SP_OFF_" + segment[-1]
    offset = hdr[key]
    try:
        slope = xtract_info.field("slope")[0]
    except KeyError:
        slope = 0.0
    b_spec = xtract_info.field("b_spec")[0]
    offset_to_middle = slope * (ncols // 2 - x_offset)
    # nominal location of spectrum, where it crosses the middle of the
    # flt or counts image.  This is copied from boxcar extraction.
    xd_nominal = b_spec + shift2 + offset_to_middle
    b_bkg1, b_bkg2 = getBackgroundCenters(xtract_info, centroid)
    if cosutil.findColumn(xtract_info, "b_hgt1"):
        bkg_height1  = xtract_info.field("b_hgt1")[0]
        bkg_height2  = xtract_info.field("b_hgt2")[0]
    else:
        bkg_height1  = xtract_info.field("bheight")[0]
        bkg_height2  = bkg_height1    
    updateExtractionKeywords(ofd_header, segment,
                             slope, height,
                             xd_nominal, centroid, cent_err, offset,
                             b_bkg1, b_bkg2,
                             bkg_height1, bkg_height2)
    return (N_i, ERROR_i, ERROR_LOWER_i, VARIANCE_FLAT_i, VARIANCE_COUNTS_i, VARIANCE_BKG_i,
            GC_i, GCOUNTS_i, SUMMED_BACKGROUND_i, DQ_i, DQ_WGT_i,
            DQ_ALL_i, LOWER_OUTER_INDEX_i, UPPER_OUTER_INDEX_i,
            LOWER_INNER_INDEX_i, UPPER_INNER_INDEX_i,
            ENCLOSED_FRACTION_i, AV_E_BKG_i,
            LOWER_OUTER_VALUE_i, LOWER_INNER_VALUE_i,
            UPPER_INNER_VALUE_i, UPPER_OUTER_VALUE_i
            )

def getProfileCentroid(phdr, segment):
    key = "SP_LOC_" + segment[-1]
    return phdr[key]

def getBackgroundCenters(xtract_info, centroid):
    #
    # Background regions in reference profile are calculated relative to the
    # centroid
    b_bkg1 = xtract_info.field("b_bkg1")[0]
    b_bkg2 = xtract_info.field("b_bkg2")[0]
    b_spec = xtract_info.field("b_spec")[0]
    offset = int(round(b_bkg1 - b_spec))
    b_bkg1 = int(round(offset + centroid))
    offset = int(round(b_bkg2 - b_spec))
    b_bkg2 = int(round(offset + centroid))
    return b_bkg1, b_bkg2

def getBackgroundRegion(data_ij, b_bkg, bkg_height):
    #
    # Get the background region
    nrows, ncols = data_ij.shape
    #
    # b_bkg is an integer
    rowstart = b_bkg - bkg_height // 2
    rowstop = b_bkg + bkg_height // 2
    if rowstart >= nrows or rowstop < 0:
        cosutil.printWarning("Background region outside array")
        return None
    elif rowstart < 0:
        rowstart = 0
    elif rowstop > nrows:
        rowstop = nrows
    return data_ij[int(rowstart):int(rowstop+1),:]

def getBackground(data_ij, dq_ij, xtract_info, centroid, sdqflags):
    nrows, ncols = data_ij.shape
    ndqrows, ndqcols = dq_ij.shape
    if nrows != ndqrows:
        cosutil.printWarning("Data and dq arrays have unequal rows")
        cosutil.printWarning("Data: %d, dq: %d" % (nrows, ndqrows))
    if ncols != ndqcols:
        cosutil.printWarning("Data and dq arrays have unequal columns")
        cosutil.printWarning("Data: %d, dq: %d" % (ncols, ndqcols))
    bkg_smooth = xtract_info.field("bwidth")[0]
    #
    # b_bkg1, b_bkg2 are integers
    b_bkg1, b_bkg2 = getBackgroundCenters(xtract_info, centroid)
    if cosutil.findColumn(xtract_info, "b_hgt1"):
        bkg_height1  = xtract_info.field("b_hgt1")[0]
        bkg_height2  = xtract_info.field("b_hgt2")[0]
    else:
        bkg_height1  = xtract_info.field("bheight")[0]
        bkg_height2  = bkg_height1

    BK1_ij = getBackgroundRegion(data_ij, b_bkg1, bkg_height1)
    dq1_ij = getBackgroundRegion(dq_ij, b_bkg1, bkg_height1)
    if BK1_ij is not None:
        good1_ij = np.where(np.bitwise_and(dq1_ij, sdqflags), 0.0, 1.0)
        n_bkg1_rows = good1_ij.sum(axis=0)
        BK1_ij = BK1_ij * good1_ij
        bk1_i = BK1_ij.sum(axis=0, dtype=np.float64)
    else:
        n_bkg1_rows = 0
        bk1_i = 0
        cosutil.printWarning("Background region 1 is outside data array")
    BK2_ij = getBackgroundRegion(data_ij, b_bkg2, bkg_height2)
    dq2_ij = getBackgroundRegion(dq_ij, b_bkg2, bkg_height2)
    if BK2_ij is not None:
        good2_ij = np.where(np.bitwise_and(dq2_ij, sdqflags), 0.0, 1.0)
        n_bkg2_rows = good2_ij.sum(axis=0)
        BK2_ij = BK2_ij * good2_ij
        bk2_i = BK2_ij.sum(axis=0, dtype=np.float64)
    else:
        n_bkg2_rows = 0
        bk2_i = 0
        cosutil.printWarning("Background region 2 is outside data array")
    original_BK_i = bk1_i + bk2_i
    BK_i = original_BK_i
    n_bkg_rows = n_bkg1_rows + n_bkg2_rows
    goodrows = np.where(n_bkg_rows > 0)
    av_bkg = np.zeros(ncols, dtype=np.float32)
    av_bkg[goodrows] = BK_i[goodrows] / n_bkg_rows[goodrows]
    temp_bk = av_bkg.copy().astype(np.float32)
    ccos.smoothbkg(temp_bk, bkg_smooth)
    return temp_bk, n_bkg_rows

def getPercentiles(xtract_info):
    p1 = xtract_info.field("LOWER_OUTER")[0]
    p2 = xtract_info.field("LOWER_INNER")[0]
    p3 = xtract_info.field("UPPER_INNER")[0]
    p4 = xtract_info.field("UPPER_OUTER")[0]
    return (p1, p2, p3, p4)

def NormalizeProfile(profile_ij, xtract_info, centroid):
    #
    # Normalize the profile to 1.0 within the area defined by the xtractab
    # Returns the full-sized array
    height = xtract_info.field("height")[0]
    rowstart = int(round(centroid)) - height // 2
    rowstop = int(round(centroid)) + height // 2
    summed = profile_ij[rowstart:rowstop+1,:].sum(axis=0, dtype=np.float64)
    nonzero_cols = np.where(summed != 0.0)
    normalized = profile_ij.copy()
    normalized[:,nonzero_cols] = normalized[:,nonzero_cols] / \
        summed[nonzero_cols]
    return normalized

def getPercentileVectors(cumulative_profile, p1, p2, p3, p4):
    nrows, ncols = cumulative_profile.shape
    lower_outer_index = np.zeros(ncols)
    lower_inner_index = np.zeros(ncols)
    upper_inner_index = np.zeros(ncols)
    upper_outer_index = np.zeros(ncols)
    lower_outer_value = np.zeros(ncols)
    lower_inner_value = np.zeros(ncols)
    upper_inner_value = np.zeros(ncols)
    upper_outer_value = np.zeros(ncols)
    enclosed_fraction = np.zeros(ncols)
    for column in range(ncols):
        if cumulative_profile[-1, column] > 0:
            if p1 == 0.0:
                lower_outer_index[column] = 0
                lower_outer_value[column] = 0.0
            else:
                lessthanp1 =  np.where(cumulative_profile[:, column] < p1)[0]
                if len(lessthanp1) > 0:
                    #
                    # Last row where profile < p1
                    # First row of the lower outer region
                    row = lessthanp1[-1]
                else:
                    row = 0
                lower_outer_index[column] = row
                lower_outer_value[column] = cumulative_profile[row, column]
            if p2 == 0.0:
                lower_inner_index[column] = 0
                lower_inner_value[column] = 0.0
            else:
                lessthanp2 =  np.where(cumulative_profile[:, column] < p2)[0]
                if len(lessthanp2) > 0:
                    #
                    # Last row where profile < p2
                    # First row of the inner region
                    row = lessthanp2[-1]
                else:
                    row = 0
                lower_inner_index[column] = row
                lower_inner_value[column] = cumulative_profile[row, column]
            if p3 == 1.0:
                #
                # The index of the last row is (nrows-1)
                upper_inner_index[column] = nrows - 1
                upper_inner_value[column] = 1.0
            else:
                morethanp3 =  np.where(cumulative_profile[:, column] > p3)[0]
                if len(morethanp3) > 0:
                    #
                    # First row where profile > p3
                    # Last row of the inner region, so we'll need to add
                    # 1 to the slice index
                    row = morethanp3[0]
                else:
                    row = nrows
                upper_inner_index[column] = row
                upper_inner_value[column] = cumulative_profile[row, column]
            if p4 == 1.0:
                #
                # The index of the last row is (nrows-1)
                upper_outer_index[column] = nrows - 1
                upper_outer_value[column] = 1.0
            else:
                morethanp4 =  np.where(cumulative_profile[:, column] > p4)[0]
                if len(morethanp4) > 0:
                    #
                    # First row where profile > p4
                    # Last row of the upper outer region, so we'll add 1 to
                    # the index in the slice
                    row = morethanp4[0]
                else:
                    row = nrows
                upper_outer_index[column] = row
                upper_outer_value[column] =  cumulative_profile[row, column]
            enclosed_fraction[column] = upper_outer_value[column] - \
                lower_outer_value[column]
    return (lower_outer_index, lower_inner_index, upper_inner_index,
            upper_outer_index, enclosed_fraction, lower_outer_value,
            lower_inner_value, upper_inner_value, upper_outer_value)

def bitwise_or_vector(vector):
    length = len(vector)
    if length == 0: return 0
    if length == 1: return vector[0]
    new_length = next_power_of_two(length)
    newvector = np.zeros((new_length),dtype=np.int16)
    newvector[:length] = vector
    while new_length > 1:
        new_length = new_length//2
        result = np.bitwise_or(newvector[:new_length],
                               newvector[new_length:])
        newvector = result
    return result[0]

def next_power_of_two(n):
    """
    Return next power of 2 greater than or equal to n
    """
    n -= 1 # greater than OR EQUAL TO n
    shift = 1
    while (n+1) & n: # n+1 is not a power of 2 yet
        n |= n >> shift
        shift *= 2
    return n + 1

def extractCorrtag(xi, eta, dq, epsilon, dq_array,
                   ofd_header, segment, axis_length,
                   x_offset, sdqflags, snr_ff,
                   exptime, backcorr, axis,
                   xtract_info, shift1, shift2,
                   user_xdisp_locn=None, user_xdisp_size=None,
                   find_target={"flag": False, "cutoff": None}):
    """Extract a 1-D spectrum for one segment or stripe.

    Parameters
    ----------
    xi: array_like
        Column of pixel coordinates in the dispersion direction

    eta: array_like
        Column of pixel coordinates in the cross-dispersion direction

    dq: array_like
        Column of data quality flags

    epsilon: array_like
        Column of weights for flat field or nonlinearity

    dq_array: 2-D array
        DQ array, created from the bad pixel table

    ofd_header: pyfits Header object
        header of the output table (for updating keywords)

    segment: str
        FUVA or FUVB, etc. (only used for updating keywords)

    axis_length: int
        Length of dispersion axis

    x_offset: int
        Offset of the detector in the output array

    sdqflags: int
        "Serious" data quality flags

    snr_ff: float
        The signal-to-noise ratio of the flat field reference file (from
        the extension header of the flat field)

    exptime: float
        Exposure time (seconds), from the (corrected) header keyword

    backcorr: int
        "PERFORM" if background subtraction is to be done

    axis: int
        The dispersion axis, 0 (Y) or 1 (X)

    xtract_info: pyfits record object
        One row of the xtractab

    shift1: float
        Offset in the dispersion direction (used for NUV in the section for
        smoothing the background)

    shift2: float
        Offset in the cross-dispersion direction

    user_xdisp_locn: int or float, or None if not specified
        User-specified location in cross-dispersion direction

    user_xdisp_size: int, or None if not specified
        User-specified height of extraction box

    find_target: dictionary
        Keys are "flag" and "cutoff".  flag = True means that we should use
        the location that we find for the target in the cross-dispersion
        direction if the standard deviation (pixels) of the location is
        less than or equal to cutoff (if cutoff is positive).  flag = False
        means we should use the location determined from the wavecal or as
        specified by the user.

    Returns
    -------
    tuple of seven 1-D arrays
        net count rate, error estimate, gross count rate, gross counts,
        background count rate, data quality array, data quality weight
        array
    """

    local_find_targ = copy.deepcopy(find_target)

    try:
        slope           = xtract_info.field("slope")[0]
    except KeyError:
        slope = 0.0
    b_spec          = xtract_info.field("b_spec")[0]    # see user_xdisp_locn
    extr_height     = xtract_info.field("height")[0]    # see user_xdisp_size
    b_bkg1          = xtract_info.field("b_bkg1")[0]
    b_bkg2          = xtract_info.field("b_bkg2")[0]
    if cosutil.findColumn(xtract_info, "b_hgt1"):
        bkg_height1  = xtract_info.field("b_hgt1")[0]
        bkg_height2  = xtract_info.field("b_hgt2")[0]
    else:
        bkg_height1  = xtract_info.field("bheight")[0]
        bkg_height2  = bkg_height1
    bkg_smooth      = xtract_info.field("bwidth")[0]

    offset_to_middle = slope * (axis_length // 2 - x_offset)
    # nominal location of spectrum, where it crosses the middle of the
    # flt or counts image
    xd_nominal = b_spec + shift2 + offset_to_middle

    # xd_locn is either the user-specified value (if it was specified) or the
    # location based on the wavecal; in either case, it's where the spectrum
    # crosses the middle of the array, not the left edge of the array.
    if user_xdisp_locn is None:
        use_found_location = local_find_targ["flag"]
        # xxx not implemented yet xxx
        #if local_find_targ["cutoff"] is not None and \
        #   found_locn_sigma > local_find_targ["cutoff"]):
        #    use_found_location = False
        if use_found_location:
            b_spec += shift2
            xd_locn = b_spec + slope * (axis_length // 2 - x_offset)
            # xxx not implemented yet xxx
            #y_nominal = b_spec + shift2
            # xxx need different arguments for xdSearch for corrtag xxx
            #(shift2, b_spec, b_spec_sigma, fwhm) = xd_search.xdSearch (e_data,
            #                    e_dq_data, wavelength,
            #                    axis, slope, b_spec,
            #                    x_offset, info["detector"], info["opt_elem"])
            # xxx check whether b_spec is None
            #message = "Spectrum found at y = %.2f (nominal y = %.2f)." % \
            #           (xd_locn, y_nominal + offset_to_middle)
            #cosutil.printMsg (message, VERBOSE)
            #cosutil.printContinuation (
            #"error estimate for y location = %.2f, FWHM = %.2f" % \
            #                           (b_spec_sigma, fwhm))
        else:
            # add the shift to the nominal location; assign a value to xd_locn
            # (which will be used to update a header keyword)
            b_spec += shift2
            xd_locn = b_spec + slope * (axis_length // 2 - x_offset)
            b_bkg1 += shift2
            b_bkg2 += shift2
    else:
        # use the user-specified value, but convert to b_spec, the intersection
        # with the left edge of the array
        b_spec = user_xdisp_locn - slope * (axis_length // 2 - x_offset)
        xd_locn = user_xdisp_locn

    if user_xdisp_size is not None:
        extr_height = user_xdisp_size   # use the user-specified value

    # Compute the data quality and data quality weight arrays.
    DQ_i = np.zeros(axis_length, dtype=np.int16)
    if dq_array is not None:

        # Get data quality flags within extraction region.
        dq_ij = np.zeros((extr_height, axis_length), dtype=np.int16)
        ccos.extractband(dq_array, axis, slope, b_spec, x_offset, dq_ij)
        # For each i, DQ_i[i] will be the bitwise OR of dq_ij[:,i].
        ccos.dq_or(dq_ij, DQ_i)

        # In bad_ij and bad_i, 0 means OK and 1 means bad
        bad_ij = np.zeros((extr_height, axis_length), dtype=np.int32)
        bad_ij[:,:] = np.where(np.bitwise_and(dq_ij, sdqflags), 1, 0)
        bad_i = bad_ij.sum(axis=0)
        # Any bad pixel in extraction region?  DQ_WGT is a weight,
        # so 0 is bad and 1 is good.
        DQ_WGT_i = np.where(bad_i > 0, 0., 1.)
        del dq_ij, bad_ij, bad_i
    else:
        DQ_WGT_i = np.ones(axis_length, dtype=np.float64)

    e_ij = np.zeros((extr_height, axis_length), dtype=np.float64)
    ccos.xy_extract(xi, eta, e_ij, slope, b_spec,
                    x_offset, dq, SERIOUS_DQ_FLAGS, epsilon)

    GC_ij = np.zeros((extr_height, axis_length), dtype=np.float64)
    ccos.xy_extract(xi, eta, GC_ij, slope, b_spec,
                    x_offset, dq, SERIOUS_DQ_FLAGS)

    e_ij /= exptime
    e_i = e_ij.sum(axis=0, dtype=np.float64)
    GCOUNTS_i = GC_ij.sum(axis=0,
                          dtype=np.float64)   # gross counts (not count rate)
    GC_i = GCOUNTS_i / exptime                # gross count rate
    del GC_ij

    eps_i = e_i / np.where(GC_i <= 0., 1., GC_i)
    # default value when there are no counts
    eps_i = np.where(e_i == 0., 1., eps_i)
    del e_ij, e_i

    bkg_norm = float(extr_height) / (float(bkg_height1 + bkg_height2))
    if backcorr == "PERFORM":
        BK1_ij = np.zeros((bkg_height1, axis_length), dtype=np.float64)
        dq1_ij = np.zeros((bkg_height1, axis_length), dtype=np.int16)
        BK2_ij = np.zeros((bkg_height2, axis_length), dtype=np.float64)
        dq2_ij = np.zeros((bkg_height2, axis_length), dtype=np.int16)
        # Get the background data.
        ccos.xy_extract(xi, eta, BK1_ij, slope, b_bkg1, x_offset)
        ccos.xy_extract(xi, eta, BK2_ij, slope, b_bkg2, x_offset)
        # Get the data quality array from the flt file.
        ccos.extractband(dq_array, axis, slope, b_bkg1, x_offset, dq1_ij)
        ccos.extractband(dq_array, axis, slope, b_bkg2, x_offset, dq2_ij)
        good1_ij = dq1_ij.copy()
        good2_ij = dq2_ij.copy()
        # In good[12]_ij, 1 means OK and 0 means bad.
        good1_ij[:,:] = np.where(np.bitwise_and(dq1_ij, sdqflags), 0, 1)
        good2_ij[:,:] = np.where(np.bitwise_and(dq2_ij, sdqflags), 0, 1)
        # Use the good[12]_ij arrays as a mask to exclude bad data in the
        # background regions.
        BK1_ij *= good1_ij
        BK2_ij *= good2_ij
        BK_i = BK1_ij.sum(axis=0, dtype=np.float64) + \
            BK2_ij.sum(axis=0, dtype=np.float64)
        BK_i /= exptime
        original_BK_i = BK_i.copy()

        # The sum along axis=0 gives the number of good pixels in each column.
        # Use this sum to correct (rescale) the background to account for
        # pixels that are flagged as bad.
        good_i = good1_ij.sum(axis=0, dtype=np.float64) + \
                 good2_ij.sum(axis=0, dtype=np.float64)
        # flags will be 0 where good, 1 where bad; this is used for smoothing.
        flags = np.ones(axis_length, dtype=np.int16)
        flags[:] = np.where(good_i > 0.5, 0, 1)
        # If good_i is zero, the background will also be zero, so it doesn't
        # matter what we set good_i to as long as it's not zero (we're going
        # to divide by it).
        good_i_div = np.where(good_i > 0., good_i, 1.)
        # Correct for regions excluded because they're flagged as bad.
        BK_i *= (float(bkg_height1 + bkg_height2)) / good_i_div
        # Scale the background to the spectral extraction height.
        BK_i *= bkg_norm
        original_BK_i *= bkg_norm
        # Restore BK_i where all background pixels were flagged as bad.
        BK_i[:] = np.where(good_i > 0., BK_i, original_BK_i)
        if x_offset > 0:
            i = x_offset - shift1
            i = int(round(i))
            j = i + NUV_X
            i = max(i, 0)
            j = min(j, axis_length)     # upper limit of a slice
        else:
            (i, j) = (0, axis_length)
        (i, j) = excludeAllBad(good_i, i, j)
        bk_i_f32 = BK_i.astype(np.float32)
        temp_bk = bk_i_f32[i:j].copy()
        ccos.smoothbkg(temp_bk, bkg_smooth)
        BK_i[i:j] = temp_bk.copy()
        del temp_bk
    else:
        BK_i = np.zeros(axis_length, dtype=np.float64)

    N_i = eps_i * (GC_i - BK_i)

    if snr_ff > 0.:
        VARIANCE_FLAT_i = (N_i * exptime / (extr_height * snr_ff))**2
    else:
        VARIANCE_FLAT_i = N_i * 0.
    VARIANCE_COUNTS_i = eps_i**2 * exptime * GC_i
    VARIANCE_BKG_i = eps_i**2 * exptime * BK_i * (bkg_norm / float(bkg_smooth))
    if exptime > 0.:
        VARIANCE_i = VARIANCE_FLAT_i + VARIANCE_COUNTS_i + VARIANCE_BKG_i
        VARIANCE_i = np.where(VARIANCE_i > 0, VARIANCE_i, 0.)
        # Use the frequentist option of astropy Poisson Confidence Interval function
        # to calculate errors
        ERROR_LOWER_i, ERROR_i = cosutil.errFrequentist(VARIANCE_i)
        ERROR_LOWER_i /= exptime
        ERROR_i /= exptime
    else:
        ERROR_LOWER_i = N_i * 0.
        ERROR_i = N_i * 0.
    if ofd_header is not None:
        xd_offset = -999.               # not implemented yet
        updateExtractionKeywords(ofd_header, segment,
                                 slope, extr_height,
                                 xd_nominal, xd_locn, 999., xd_offset,
                                 b_bkg1, b_bkg2, bkg_height1, bkg_height2)
    DQ_ALL_i = DQ_i
    LOWER_OUTER_INDEX_i = N_i*0.0 + xd_nominal - extr_height//2
    UPPER_OUTER_INDEX_i = N_i*0.0 + xd_nominal + extr_height//2
    LOWER_INNER_INDEX_i = LOWER_OUTER_INDEX_i.copy()
    UPPER_INNER_INDEX_i = UPPER_OUTER_INDEX_i.copy()
    ENCLOSED_FRACTION_i = N_i*0.0 + 1.0
    AV_E_BKG_i = BK_i / float(bkg_height1 + bkg_height2)
    LOWER_OUTER_VALUE_i = N_i*0.0 + 0.0
    LOWER_INNER_VALUE_i = N_i*0.0 + 0.0
    UPPER_INNER_VALUE_i = N_i*0.0 + 1.0
    UPPER_OUTER_VALUE_i = N_i*0.0 + 1.0

    return (N_i, ERROR_i, ERROR_LOWER_i, 
            VARIANCE_FLAT_i, VARIANCE_COUNTS_i, VARIANCE_BKG_i,
            GC_i, GCOUNTS_i, BK_i, DQ_i, DQ_WGT_i,
            DQ_ALL_i, LOWER_OUTER_INDEX_i, UPPER_OUTER_INDEX_i,
            LOWER_INNER_INDEX_i, UPPER_INNER_INDEX_i,
            ENCLOSED_FRACTION_i, AV_E_BKG_i,
            LOWER_OUTER_VALUE_i, LOWER_INNER_VALUE_i,
            UPPER_INNER_VALUE_i, UPPER_OUTER_VALUE_i)

def doFluxCorr(ofd, info, reffiles, tdscorr):
    """Convert net counts to flux, updating flux and error columns.

    The correction to flux is made by dividing by the appropriate row
    in the fluxtab.  If a time-dependent sensitivity table (tdstab) has
    been specified, the flux and error will be corrected to the time of
    observation.

    Parameters
    ----------
    ofd: pyfits HDUList object
        HDUList for the output table; the primary header will be modified
        to set FLUXCORR to COMPLETE, and TDSCORR may be set to either
        COMPLETE or SKIPPED

    info: dictionary
        Header keywords and values

    reffiles: dictionary
        Reference file names

    tdscorr: str
        Calibration switch, time-dependent sensitivity correction
    """

    outdata = ofd[1].data
    nrows = outdata.shape[0]
    segment = outdata.field("SEGMENT")
    wavelength = outdata.field("WAVELENGTH")
    net = outdata.field("NET")
    flux = outdata.field("FLUX")
    error = outdata.field("ERROR")
    error_lower = outdata.field("ERROR_LOWER")
    fluxtab = reffiles["fluxtab"]

    # segment will be added to filter in the loop
    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "aperture": info["aperture"]}
    # Also select the row on fpoffset, if that column is present in the table.
    if cosutil.findColumn(fluxtab, "fpoffset"):
        filter["fpoffset"] = info["fpoffset"]

    for row in range(nrows):
        pharange = cosutil.getPulseHeightRange(ofd[1].header, segment[row])
        # xxx this is temporary; eventually select the row based on pharange
        ref_pharange = cosutil.tempPulseHeightRange(fluxtab)
        cosutil.comparePulseHeightRanges(pharange, ref_pharange, fluxtab)
        factor = np.zeros(len(flux[row]), dtype=np.float32)
        filter["segment"] = segment[row]
        flux_info = cosutil.getTable(fluxtab, filter, exactly_one=True)
        # Interpolate sensitivity at each wavelength.
        wl_phot = flux_info.field("wavelength")[0]
        sens_phot = flux_info.field("sensitivity")[0]
        ccos.interp1d(wl_phot, sens_phot, wavelength[row], factor)
        factor = np.where(factor <= 0., 1., factor)
        flux[row][:] = net[row] / factor
        error[row][:] = error[row] / factor
        error_lower[row][:] = error_lower[row] / factor
    ofd[0].header["fluxcorr"] = "COMPLETE"

    # Compute an array of time-dependent correction factors (a potentially
    # different value at each wavelength), and divide the flux and error by
    # this array.

    if tdscorr == "PERFORM":
        tdstab = reffiles["tdstab"]
        t_obs = (ofd[1].header["expstart"] + ofd[1].header["expend"]) / 2.
        filter = {"opt_elem": info["opt_elem"],
                  "aperture": info["aperture"],
                  "cenwave": info["cenwave"]}
        # First check for dummy rows in the TDS table.  If there is no
        # pedigree column, assume all rows are good (i.e. not dummy).
        dummy = False           # initial value
        for row in range(nrows):
            pharange = cosutil.getPulseHeightRange(ofd[1].header, segment[row])
            # xxx this is temporary
            ref_pharange = cosutil.tempPulseHeightRange(tdstab)
            cosutil.comparePulseHeightRanges(pharange, ref_pharange, tdstab)
            filter["segment"] = segment[row]
            tds_info = cosutil.getTable(tdstab, filter, exactly_one=True)
            names = []
            for name in tds_info.names:
                names.append(name.lower())
            if "pedigree" not in names:
                break
            pedigree = tds_info.field("pedigree")[0]
            if pedigree == "DUMMY":
                dummy = True
                cosutil.printWarning("Current row in TDSTAB %s is dummy" % \
                                     tdstab, level=VERBOSE)
                cosutil.printContinuation("for filter = %s," % \
                                          str(filter), level=VERBOSE)
                cosutil.printContinuation("so TDSTAB will not be done.", \
                                          level=VERBOSE)
                break
        if dummy:
            ofd[0].header["tdscorr"] = "SKIPPED"
        else:
            printed = False             # used below
            for row in range(nrows):
                filter["segment"] = segment[row]
                # Get an array of factors vs. wavelength at the time of the obs.
                tds_results = getTdsFactors(tdstab, filter, t_obs)
                (wl_tds, factor_tds, extrapolate) = tds_results
                factor = np.zeros(len(flux[row]), dtype=np.float32)
                # Interpolate factor_tds at each wavelength.
                ccos.interp1d(wl_tds, factor_tds, wavelength[row], factor)
                flux[row][:] /= factor
                error[row][:] /= factor
                error_lower[row][:] /= factor
                if extrapolate and not printed:
                    cosutil.printWarning("TDS correction was extrapolated.")
                    printed = True
            ofd[0].header["tdscorr"] = "COMPLETE"

def getTdsFactors(tdstab, filter, t_obs):
    """Get arrays of wavelengths and corresponding TDS factors.

    If the time of observation is outside the range of times in the TDS
    table, the correction factor will be extrapolated using the slope at
    the first or last time in the table respectively.

    Parameters
    ----------
    tdstab: str
        Name of the time-dependent sensitivity reference table

    filter: dictionary
        For selecting a row from tdstab

    t_obs: float
        Time of the observation (MJD)

    Returns
    -------
    tuple or None
        (wl_tds, factor_tds, extrapolate), where wl_tds is the array
        of wavelengths from the TDS table, and factor_tds is the
        corresponding array of time-dependent sensitivity factors,
        evaluated at the time of observation from the slope and
        intercept from the TDS table, and extrapolate will be True if
        the time of observation was outside the range of times in the
        TDS table.
    """

    # Slope and intercept are specified for each of the nt entries in
    # the TIME column and for each of the nwl values in the WAVELENGTH
    # column.  nt and nwl should be at least 1.

    tds_info = cosutil.getTable(tdstab, filter, exactly_one=True)

    fd = fits.open(tdstab, mode="readonly")
    try:
        ref_time = fd[1].header["ref_time"]         # MJD
    except KeyError:
        cosutil.printWarning("REF_TIME keyword missing from TDSTAB data extension header")
        cosutil.printMsg("Setting to 0, will probably make fluxes negative")
        ref_time = 0.0
    fd.close()

    nwl = tds_info.field("nwl")[0]
    nt = tds_info.field("nt")[0]
    wl_tds = tds_info.field("wavelength")[0]            # 1-D array
    time = tds_info.field("time")[0]                    # 1-D array
    slope = tds_info.field("slope")[0]                  # 2-D array
    intercept = tds_info.field("intercept")[0]          # 2-D array

    # temporary, xxx
    # This section is needed because pyfits currently ignores TDIMi.
    maxt = len(time)
    maxwl = len(wl_tds)
    slope = np.reshape(slope, (maxt, maxwl))
    intercept = np.reshape(intercept, (maxt, maxwl))

    extrapolate = (t_obs < time[0] or t_obs >= time[nt-1])

    # Find the time interval that includes the time of observation.
    # The variable i is set here, and it's used below.
    if nt == 1 or t_obs >= time[nt-1]:
        i = nt - 1
    else:
        for i in range(nt-1):
            if t_obs < time[i+1]:
                break

    # The slope in the tdstab is in percent per year.  Convert the time
    # interval to years, and convert the slope to fraction per year.
    # If the time of observation is before the first time in the table or
    # after the last time, the extrapolation will be done using the slope
    # at the first time or the last time respectively.
    delta_t = (t_obs - ref_time) / DAYS_PER_YEAR
    slope[:,:] /= 100.

    # Take the slice [0:nwl] to avoid using elements that may not be valid,
    # and because the array of factors should be the same length as the
    # set of wavelengths that have been specified.
    wl_tds = wl_tds[0:nwl]
    factor_tds = delta_t * slope[i][0:nwl] + intercept[i][0:nwl]

    return (wl_tds, factor_tds, extrapolate)

def updateExtractionKeywords(hdr, segment, slope, height,
                             xd_nominal, xd_locn, found_locn_sigma, xd_offset,
                             b_bkg1, b_bkg2, bkg_height1, bkg_height2):
    """Update keywords giving the locations of extraction regions.

    Parameters
    ----------
    ofd_header: pyfits Header object
        Header of the output table.

    segment: str
        FUVA or FUVB; NUVA, NUVB, or NUVC.

    slope: float
        Slope of spectrum.

    height: int
        Height of extraction box.

    xd_nominal: float
        Expected location of the spectrum in the cross-dispersion
        direction, where it crosses the middle of the image.

    xd_locn: float
        Location of the spectrum in the cross-dispersion direction, where
        it crosses the middle of the image.

    found_locn_sigma: float
        Error estimate for xd_locn, pixels.
        xxx currently ignored xxx

    xd_offset: float
        Difference between where the spectrum was found in the cross-
        dispersion direction and where it was expected, in the sense
        (found - expected).

    b_bkg1: float
        Location of first background region, at left edge, as read from
        the reference table.

    b_bkg2: float
        Location of second background region, at left edge, as read from
        the reference table.

    bkg_height1: int
        Height of first background region.

    bkg_height2: int
        Height of second background region.
    """

    key = "SP_LOC_" + segment[-1]           # SP_LOC_A, SP_LOC_B, SP_LOC_C
    hdr[key] = xd_locn
    if segment[-1] == "A":
        othersegment = "B"
    else:
        othersegment = "A"
    key = "SP_ERR_" + segment[-1]           # SP_ERR_A, SP_ERR_B, SP_ERR_C
    hdr[key] = found_locn_sigma
    key = "SP_ERR_" + othersegment
    hdr[key] = -999.0
    key = "SP_OFF_" + segment[-1]           # SP_OFF_A, SP_OFF_B, SP_OFF_C
    hdr[key] = xd_offset
    key = "SP_NOM_" + segment[-1]           # SP_NOM_A, SP_NOM_B, SP_NOM_C
    hdr[key] = xd_nominal
    key = "SP_SLP_" + segment[-1]           # SP_SLP_A, SP_SLP_B, SP_SLP_C
    hdr[key] = slope
    key = "SP_HGT_" + segment[-1]           # SP_HGT_A, SP_HGT_B, SP_HGT_C
    hdr[key] = height

    # Adjust the values of the background locations to be where the regions
    # cross the middle of the detector.
    if segment[0] == "F":
        tilt_offset = slope * FUV_X / 2.
    else:
        tilt_offset = slope * NUV_X / 2.
    b_bkg1 += tilt_offset
    b_bkg2 += tilt_offset

    key = "B_BKG1_" + segment[-1]
    hdr[key] = b_bkg1
    key = "B_BKG2_" + segment[-1]
    hdr[key] = b_bkg2
    key = "B_HGT1_" + segment[-1]
    hdr[key] = bkg_height1
    key = "B_HGT2_" + segment[-1]
    hdr[key] = bkg_height2

def copyKeywordsToInput(output, input, incounts):
    """Copy extraction location keywords to the input headers.

    Parameters
    ----------
    output: str
        Name of the output file for 1-D extracted spectra

    input: str
        Name of either the flat-fielded count-rate image or the corrtag
        table

    incounts: str or None
        Name of the file containing the count-rate image, or None if input
        is the corrtag table
    """

    ofd = fits.open(output, mode="readonly")
    ifd_e = fits.open(input, mode="update")
    if incounts is not None:
        ifd_c = fits.open(incounts, mode="update")

    if ofd[0].header["detector"] == "FUV":
        keywords = ["sp_loc_a", "sp_loc_b",
                    "sp_off_a", "sp_off_b",
                    "sp_err_a", "sp_err_b",
                    "sp_nom_a", "sp_nom_b",
                    "sp_slp_a", "sp_slp_b",
                    "sp_hgt_a", "sp_hgt_b",
                    "b_bkg1_a", "b_bkg1_b",
                    "b_bkg2_a", "b_bkg2_b",
                    "b_hgt1_a", "b_hgt1_b",
                    "b_hgt2_a", "b_hgt2_b"]
    else:
        keywords = ["sp_loc_a", "sp_loc_b", "sp_loc_c",
                    "sp_off_a", "sp_off_b", "sp_off_c",
                    "sp_nom_a", "sp_nom_b", "sp_nom_c",
                    "sp_slp_a", "sp_slp_b", "sp_slp_c",
                    "sp_hgt_a", "sp_hgt_b", "sp_hgt_c",
                    "b_bkg1_a", "b_bkg1_b", "b_bkg1_c",
                    "b_bkg2_a", "b_bkg2_b", "b_bkg2_c",
                    "b_hgt1_a", "b_hgt1_b", "b_hgt1_c",
                    "b_hgt2_a", "b_hgt2_b", "b_hgt2_c"]

    for key in keywords:
        value = ofd[1].header.get(key, -999.)
        ifd_e[1].header[key] = value
        if incounts is not None:
            ifd_c[1].header[key] = value

    ofd.close()
    ifd_e.close()
    if incounts is not None:
        ifd_c.close()

def updateCorrtagKeywords(flt, corrtag):
    """Update extraction-location keywords in a corrtag file.

    Parameters
    ----------
    flt: str
        Name of an flt file

    corrtag: str
        Name of a corrtag file, to be modified in-place
    """

    ifd = fits.open(flt, mode="readonly")
    ofd = fits.open(corrtag, mode="update")

    iphdr = ifd[0].header
    detector = iphdr.get("detector", "missing")
    if detector == "FUV":
        segment_list = [iphdr["segment"]]
    elif detector == "NUV":
        segment_list = ["NUVA", "NUVB", "NUVC"]
    else:
        segment_list = []

    ihdr = ifd[1].header
    ohdr = ofd[1].header

    for segment in segment_list:
        for key in ["SP_LOC_", "SP_OFF_", "SP_ERR_", "SP_NOM_", "SP_SLP_",
                    "SP_HGT_", "B_BKG1_", "B_BKG2_", "B_HGT1_", "B_HGT2_"]:
            keyword = key + segment[-1]
            if keyword in ihdr:
                ohdr[keyword] = ihdr[keyword]

    ofd.close()
    ifd.close()

def updateArchiveSearch(ofd):
    """Update the keywords giving min & max wavelengths, etc.

    Parameters
    ----------
    ofd: pyfits HDUList object
        Output, primary header will be modified in-place
    """

    phdr = ofd[0].header
    detector = phdr["detector"]
    outdata = ofd[1].data
    nrows = outdata.shape[0]
    segment = outdata.field("SEGMENT")
    wavelength = outdata.field("WAVELENGTH")
    if cosutil.findColumn(outdata, "dq_wgt"):
        dq_wgt = outdata.field("DQ_WGT")
    else:
        dq_wgt = None

    # First update PLATESC and SPECRES (even if there's no data).
    spwcstab = phdr.get("spwcstab", "N/A")
    if spwcstab != "N/A":
        opt_elem = phdr.get("opt_elem", "missing")
        cenwave = phdr.get("cenwave", "missing")
        aperture = phdr.get("aperture", "PSA")
        if aperture not in ["PSA", "BOA"]:
            aperture = "PSA"    # override aperture = "WCA" or "FCA"
        if nrows > 0:
            current_segment = segment[0]
        else:
            if detector == "FUV":
                if phdr["segment"][0:3] == "FUV":
                    current_segment = phdr["segment"]
                else:
                    current_segment = "FUVA"
            else:
                current_segment = "NUVA"
        filter = {"opt_elem": opt_elem,
                  "cenwave":  cenwave,
                  "segment":  current_segment,
                  "aperture": aperture}
        spwcstab = cosutil.expandFileName(spwcstab)
        wcs_info = cosutil.getTable(spwcstab, filter, exactly_one=True)
        cdelt2 = wcs_info.field("cdelt2")[0] * 3600
        if detector == "NUV":
            cdelt3 = wcs_info.field("cdelt3")[0] * 3600
            platesc = (cdelt2 + cdelt3) / 2.
        else:
            platesc = cdelt2
        phdr["platesc"] = platesc
        specres = wcs_info.field("specres")
        phdr["specres"] = specres[0]

    if nrows <= 0 or len(wavelength[0]) < 1:
        return

    nelem = len(wavelength[0])
    # This initial value assumes wavelengths increase with pixel number.
    minwave = wavelength[0][nelem-1]
    maxwave = wavelength[0][0]
    for row in range(nrows):
        if dq_wgt is None:
            good_wl = wavelength[row]
        elif dq_wgt[row].sum(dtype=np.float64) <= 0:
            cosutil.printWarning("DQ_WGT is all 0 for '%s'" % segment[row])
            good_wl = wavelength[row]
        else:
            good_wl = wavelength[row][dq_wgt[row] > 0.]
        minwave_row = good_wl.min()
        minwave = min(minwave, minwave_row)
        maxwave_row = good_wl.max()
        maxwave = max(maxwave, maxwave_row)

    phdr["MINWAVE"] = minwave
    phdr["MAXWAVE"] = maxwave
    phdr["BANDWID"] = maxwave - minwave
    phdr["CENTRWV"] = (maxwave + minwave) / 2.

def concatenateFUVSegments(infiles, output):
    """Concatenate the 1-D spectra for the two FUV segments into one file.

    Parameters
    ----------
    infiles: list
        List of input x1d_a and x1d_b file names

    output: str
        Output x1d file name
    """

    cosutil.printMsg("Concatenate " + repr (infiles) + " --> " + output, \
                     VERY_VERBOSE)

    a_exists = os.access(infiles[0], os.R_OK)
    b_exists = os.access(infiles[1], os.R_OK)
    if not (a_exists or b_exists):
        cosutil.printWarning("Neither %s nor %s exists." %
                             (infiles[0], infiles[1]), VERY_VERBOSE)
        return

    rename_file = ""
    if a_exists and not b_exists:
        rename_file = infiles[0]
        missing = infiles[1]
    if b_exists and not a_exists:
        rename_file = infiles[1]
        missing = infiles[0]
    if rename_file:
        cosutil.printWarning("%s is missing." % missing, VERY_VERBOSE)
        cosutil.renameFile(rename_file, output)
        return

    ifd_0 = fits.open(infiles[0], mode="copyonwrite")
    ifd_1 = fits.open(infiles[1], mode="copyonwrite")

    # Make sure we know which files are for segments A and B respectively,
    # and then use seg_a and seg_b instead of ifd_0 and ifd_1.
    if ifd_0[0].header["segment"] == "FUVA":
        seg_a = ifd_0
    elif ifd_1[0].header["segment"] == "FUVA":
        seg_a = ifd_1
    else:
        seg_a = None
    if ifd_0[0].header["segment"] == "FUVB":
        seg_b = ifd_0
    elif ifd_1[0].header["segment"] == "FUVB":
        seg_b = ifd_1
    else:
        seg_b = None
    if seg_a is None or seg_b is None:
        cosutil.printError("files are " + infiles[0] + " " + infiles[1])
        raise RuntimeError("Files to concatenate must be for "
                           "segments FUVA and FUVB.")

    if seg_a[1].data is None:
        nrows_a = 0
    else:
        nrows_a = seg_a[1].data.shape[0]

    if seg_b[1].data is None:
        nrows_b = 0
    else:
        nrows_b = seg_b[1].data.shape[0]

    # Take output column definitions from input for segment A.
    cd = fits.ColDefs(seg_a[1])
    hdu = fits.BinTableHDU.from_columns(cd, seg_a[1].header,
                                        nrows=nrows_a+nrows_b)

    # Copy data from input to output.
    copySegments(seg_a[1].data, nrows_a, seg_b[1].data, nrows_b, hdu.data)

    # Include segment-specific keywords from segment B.  The strings in
    # segment_specific_keywords (which is defined in calcosparam.py) use "X"
    # as the character to be replaced by "b" to get the actual keyword name
    # for segment B.
    for key in segment_specific_keywords:
        keyword = key.replace("X", "b")
        if keyword in seg_b[1].header:
            hdu.header[keyword] = seg_b[1].header.get(keyword, -1.0)

    exptimea = seg_a[1].header.get("exptimea",
                                   default=seg_a[1].header["exptime"])
    exptimeb = seg_b[1].header.get("exptimeb",
                                   default=seg_b[1].header["exptime"])
    hdu.header["exptime"] = max(exptimea, exptimeb)

    neventsa = seg_a[1].header.get("neventsa",
                                   seg_a[1].header.get("nevents", 0))
    neventsb = seg_b[1].header.get("neventsb",
                                   seg_b[1].header.get("nevents", 0))
    hdu.header["nevents"] = neventsa + neventsb

    # If one of the segments has no data, use the other segment for the
    # primary header.  This is so the calibration switch keywords in the
    # output x1d file will be set appropriately.
    if nrows_a > 0 or nrows_b == 0:
        phdu = seg_a[0]
    else:
        phdu = seg_b[0]
    ofd = fits.HDUList(phdu)
    cosutil.updateFilename(ofd[0].header, output)
    updateGsagComment(seg_a[0].header, seg_b[0].header, [ofd[0].header])
    if a_exists and b_exists and nrows_a > 0 and nrows_b > 0:
        # we now have both segments
        ofd[0].header["segment"] = "BOTH"
    ofd.append(hdu)

    # Update the "archive search" keywords.
    updateArchiveSearch(ofd)

    ofd.writeto(output, output_verify="fix")
    ifd_0.close()
    ifd_1.close()

    if phdu.header["statflag"]:
        cosutil.doSpecStat(output)

def updateGsagComment(phdr0, phdr1, phdr_list):
    """Combine the comments for keyword GSAGTAB.

    Parameters
    ----------
    phdr0: pyfits Header object
        Primary header of the first input file (for segment A)

    phdr1: pyfits Header object
        Primary header of the second input file (for segment B)

    phdr_list: list of pyfits Header objects
        Output primary headers, modified in-place
    """

    # Check which header is for segment A and which is for segment B.
    segment0 = phdr0.get("segment", "missing")
    segment1 = phdr1.get("segment", "missing")
    if segment0 == "FUVA":
        seg_a = phdr0
    elif segment1 == "FUVA":
        seg_a = phdr1
    else:
        seg_a = None
    if segment0 == "FUVB":
        seg_b = phdr0
    elif segment1 == "FUVB":
        seg_b = phdr1
    else:
        seg_b = None

    if seg_a is None or seg_b is None:
        return

    cards_a = seg_a.cards
    cards_b = seg_b.cards
    if "gsagtab" in seg_a:
        comment_a = cards_a["GSAGTAB"].comment
    else:
        comment_a = ""
    if "gsagtab" in seg_b:
        comment_b = cards_b["GSAGTAB"].comment
    else:
        comment_b = ""
    if comment_a.find("ext. ") >= 0 and comment_b.find("ext. ") >= 0:
        comment = comment_a + "; " + comment_b
        for phdr in phdr_list:
            gsagtab = phdr.get("gsagtab", NOT_APPLICABLE)
            phdr["gsagtab"] = (gsagtab, comment)

def copySegments(data_a, nrows_a, data_b, nrows_b, outdata):
    """Copy the two input tables to the output table.

    Parameters
    ----------
    data_a: pyfits recarray object
        Data block for segment A (may have no data)

    nrows_a: int
        Length of data_a (may be zero)

    data_b: pyfits recarray object
        Data block for segment B (may have no data)

    nrows_b: int
        Length of data_b (may be zero)

    outdata: pyfits recarray object
        Data block with nrows_a + nrows_b rows
    """

    n = 0
    for i in range(nrows_a):
        outdata[n] = data_a[i]
        n += 1
    for i in range(nrows_b):
        outdata[n] = data_b[i]
        n += 1

def recomputeWavelengths(input):
    """Update the wavelength column in a wavecal x1d table.

    The values in the wavelength column will be recomputed (updated in-place)
    to include the shift in the dispersion direction, as read from the
    shift1[abc] keywords.

    If wavecorr is already COMPLETE, this function returns without doing
    anything; otherwise, after correcting the wavelengths, wavecorr will be
    set to COMPLETE.  (Wavecal processing was done earlier, but it was not
    really finished until this function was called.)  The check on wavecorr
    is necessary because for FUV data the list of files may include the
    x1d file name twice, once for each segment.

    Parameters
    ----------
    input: str
        Name of an x1d file for a wavecal
    """

    fd = fits.open(input, mode="update")
    phdr = fd[0].header
    hdr = fd[1].header
    if hdr["naxis2"] == 0:
        fd.close()
        return
    cosutil.printMsg("Updating wavelengths in %s" % input, VERY_VERBOSE)

    data = fd[1].data

    info = getinfo.getGeneralInfo(phdr, hdr)
    disptab = cosutil.expandFileName(phdr["disptab"])

    segment_col = data.field("SEGMENT")
    nelem_col = data.field("NELEM")
    wl_col = data.field("WAVELENGTH")

    # To correct for the extra pixels (if any) in the dispersion direction.
    x_offset = hdr.get("x_offset", 0)

    for row in range(len(data)):

        segment = segment_col[row]
        filter = {"segment": segment,
                  "opt_elem": info["opt_elem"],
                  "cenwave": info["cenwave"],
                  "aperture": "WCA",
                  "fpoffset": info["fpoffset"]}
        disp_rel = dispersion.Dispersion(disptab, filter)
        if not disp_rel.isValid():
            raise MissingRowError("Missing row in DISPTAB; filter = %s" %
                                  str(disp_rel.getFilter()))
        key = "shift1" + segment[-1]
        shift1 = hdr.get(key, 0.)

        # 'pixel' is an array of pixel coordinates.
        nelem = nelem_col[row]
        pixel = np.arange(nelem, dtype=np.float64)

        pixel -= shift1
        pixel -= x_offset
        wl_col[row][0:nelem] = disp_rel.evalDisp(pixel)
        disp_rel.close()

    phdr["WAVECORR"] = "COMPLETE"

    fd.close()
