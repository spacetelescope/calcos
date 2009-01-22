import numpy as N
from convolve import boxcar
import pyfits
import cosutil
import ccos
import getinfo
# xxx import xd_search
from calcosparam import *       # parameter definitions

def extract1D (input, incounts=None, output=None,
               location=None, find_target=False):
    """Extract 1-D spectrum from 2-D image.

    @param input: name of either the flat-fielded count-rate image (in which
        case incounts must also be specified) or the corrtag table
    @type input: string
    @param incounts: name of the file containing the count-rate image,
        or None if input is the corrtag table
    @type incounts: string, or None
    @param output: name of the output file for 1-D extracted spectra
    @type output: string
    @param location: the location (or list of three locations for NUV) of
        the spectrum in the cross-dispersion direction, in pixels; this
        is where the spectrum crosses the middle of the detector (index
        8192 for FUV, 512 for NUV).  None means the user did not specify
        the location.  If location was specified, that value will be used,
        regardless of the find_target switch.
    @type location: int or float for FUV, list or tuple for NUV
    @param find_target: True means that we should search for the location of
        the target in the cross-dispersion direction; False means we should
        use the location determined from the wavecal.  find_target will be
        locally set to False if location is not None.
    @type find_target: boolean
    """

    find_target = False         # xxx not implemented yet

    cosutil.printIntro ("Spectral Extraction")
    names = [("Input", input), ("Incounts", incounts), ("Output", output)]
    cosutil.printFilenames (names)
    cosutil.printMsg ("", VERBOSE)

    # Open the input files.
    ifd_e = pyfits.open (input, mode="readonly")
    if incounts is None:
        ifd_c = None
    else:
        ifd_c = pyfits.open (incounts, mode="readonly")

    phdr = ifd_e[0].header
    hdr = ifd_e[1].header
    info = getinfo.getGeneralInfo (phdr, hdr)
    switches = getinfo.getSwitchValues (phdr)
    reffiles = getinfo.getRefFileNames (phdr)
    is_wavecal = info["exptype"].find ("WAVE") >= 0
    if not is_wavecal and switches["wavecorr"] != "COMPLETE":
        cosutil.printWarning ("WAVECORR was not done for " + input)

    if location is not None:
        find_target = False
        if isinstance (location, int) or isinstance (location, float):
            if info["detector"] == "FUV":
                location = [location]
            else:
                location = [location, location, location]
        else:
            try:
                test_type = location[0]
            except TypeError:
                raise TypeError, "location must be an int, float, or sequence"

    cosutil.printSwitch ("X1DCORR", switches)
    cosutil.printRef ("XTRACTAB", reffiles)
    cosutil.printRef ("DISPTAB", reffiles)
    cosutil.printSwitch ("HELCORR", switches)
    cosutil.printSwitch ("BACKCORR", switches)
    cosutil.printSwitch ("STATFLAG", switches)
    cosutil.printSwitch ("FLUXCORR", switches)
    if switches["fluxcorr"] == "PERFORM":
        cosutil.printRef ("phottab", reffiles)
        cosutil.printSwitch ("TDSCORR", switches)
        if switches["tdscorr"] == "PERFORM":
            cosutil.printRef ("tdstab", reffiles)

    # Create the output FITS header/data unit object.
    ofd = pyfits.HDUList (ifd_e[0])

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
            key = "naxis" + str (info["dispaxis"])
            nelem = hdr[key]
    rpt = str (nelem)                           # used for defining columns

    # Define output columns.
    col = []
    col.append (pyfits.Column (name="SEGMENT", format="4A"))
    col.append (pyfits.Column (name="EXPTIME", format="1D",
                disp="F8.3", unit="s"))
    col.append (pyfits.Column (name="NELEM", format="1J", disp="I6"))
    col.append (pyfits.Column (name="WAVELENGTH", format=rpt+"D",
                unit="angstrom"))
    col.append (pyfits.Column (name="FLUX", format=rpt+"E",
                unit="erg /s /cm**2 /angstrom"))
    col.append (pyfits.Column (name="ERROR", format=rpt+"E",
                unit="erg /s /cm**2 /angstrom"))
    col.append (pyfits.Column (name="GROSS", format=rpt+"E",
                unit="count /s"))
    col.append (pyfits.Column (name="NET", format=rpt+"E",
                unit="count /s"))
    col.append (pyfits.Column (name="BACKGROUND", format=rpt+"E",
                unit="count /s"))
    col.append (pyfits.Column (name="DQ", format=rpt+"I"))
    col.append (pyfits.Column (name="DQ_WGT", format=rpt+"E"))
    cd = pyfits.ColDefs (col)

    hdu = pyfits.new_table (cd, header=hdr, nrows=nrows)
    hdu.name = "SCI"
    ofd.append (hdu)

    if nrows > 0:
        if info["detector"] == "FUV":
            segments = [info["segment"]]
        else:
            segments = ["NUVA", "NUVB", "NUVC"]
        # Extract the spectrum or spectra.
        doExtract (ifd_e, ifd_c, ofd, nelem,
                   segments, info, switches, reffiles, is_wavecal,
                   location, find_target)
        if switches["fluxcorr"] == "PERFORM":
            # Convert net count rate to flux.
            doFluxCorr (ofd, info["opt_elem"], info["cenwave"],
                        info["aperture"], switches["tdscorr"], reffiles)
    # Update nrows, in case rows were skipped during 1-D extraction.
    nrows = ofd[1].data.shape[0]

    # Apply heliocentric Doppler correction to the wavelength array.
    if switches["helcorr"] == "PERFORM" or switches["helcorr"] == "COMPLETE":
        wavelength = ofd[1].data.field ("WAVELENGTH")
        for row in range (nrows):
            wl_row = wavelength[row]
            wl_row += (wl_row * (-hdr["v_helio"]) / SPEED_OF_LIGHT)
            wavelength[row][:] = wl_row
        phdr["helcorr"] = "COMPLETE"

    # Update the output header.
    ofd[1].header["bitpix"] = 8         # temporary, xxx
    ofd[0].header.update ("nextend", 1)
    cosutil.updateFilename (ofd[0].header, output)
    if ifd_c is None:                   # ifd_e is a corrtag table
        # Delete table-specific world coordinate system keywords.
        ofd[1].header = cosutil.delCorrtagWCS (ofd[1].header)
    else:                               # ifd_e is an flt image
        # Delete image-specific world coordinate system keywords.
        ofd[1].header = cosutil.imageHeaderToTable (ofd[1].header)
    updateArchiveSearch (ofd)           # update some keywords
    # Fix the aperture keyword, if it's RelMvReq.
    fixApertureKeyword (ofd, info["aperture"], info["detector"])
    if nrows > 0:
        ofd[0].header["x1dcorr"] = "COMPLETE"
        if switches["backcorr"] == "PERFORM":
            ofd[0].header["backcorr"] = "COMPLETE"
        # FLUXCORR and TDSCORR are updated in doFluxCorr.

    ofd.writeto (output, output_verify="silentfix")
    del ofd
    ifd_e.close()
    if ifd_c is not None:
        ifd_c.close()

    if nrows > 0:
        copyKeywordsToInput (output, input, incounts)

    if switches["statflag"] == "PERFORM":
        cosutil.doSpecStat (output)

def doExtract (ifd_e, ifd_c, ofd, nelem,
               segments, info, switches, reffiles, is_wavecal,
               location=None, find_target=False):
    """Extract either FUV or NUV data.

    This calls a routine to do the extraction for one segment, and it
    assigns the results to one row of the output table.

    @param ifd_e: HDUList for either the effective count-rate image or for
        the corrtag events table
    @type ifd_e: PyFITS HDUList object
    @param ifd_c: HDUList for the count-rate image, or None if the input is
        a corrtag events table
    @type ifd_c: PyFITS HDUList object, or None
    @param ofd: HDUList for the output table, modified in-place
    @type ofd: PyFITS HDUList object
    @param nelem: number of elements in current segment of output data
    @type nelem: int
    @param segments: the segment names, one for FUV, three for NUV
    @type segments: list
    @param info: keywords and values
    @type info: dictionary
    @param switches: calibration switch values
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param is_wavecal: true if the observation is a wavecal, based on exptype
    @type is_wavecal: boolean
    @param location: location (if FUV) or locations (if NUV, in the order
        NUVA, NUVB, NUVC), or None
    @type location: sequence of int or float, or None
    @param find_target: True means that we should search for the location of
        the target in the cross-dispersion direction; False means we should
        use the location determined from the wavecal.
    @type find_target: boolean
    """

    hdr = ifd_e[1].header
    outdata = ofd[1].data

    corrtag = (ifd_c is None)

    # Get columns, if the input is a corrtag table.
    if corrtag:
        (xi, eta, dq, epsilon) = getColumns (ifd_e, info["detector"])

    row = 0
    for segment in segments:

        filter = {"segment": segment,
                  "opt_elem": info["opt_elem"],
                  "cenwave": info["cenwave"],
                  "aperture": info["aperture"]}
        xtract_info = cosutil.getTable (reffiles["xtractab"], filter)
        # If the FPOFFSET column is present in the disptab, include fpoffset
        # in the filter.
        if cosutil.findColumn (reffiles["disptab"], "fpoffset"):
            filter["fpoffset"] = info["fpoffset"]
        disp_info = cosutil.getTable (reffiles["disptab"], filter)
        if disp_info is None or xtract_info is None:
            continue
        slope = xtract_info.field ("slope")[0]
        height = xtract_info.field ("height")[0]

        if is_wavecal:
            dpixel1 = 0.
        else:
            key = "dpixel1" + segment[-1]
            dpixel1 = hdr.get (key, 0.)

        shift2 = 0.              # cross-dispersion direction

        # xdisp_locn will be the user-specified location in cross-dispersion
        # direction (or None, if the user did not specify a value).
        # xd_locn (assigned later) will be either xdisp_locn (if specified),
        # or the value found by searching (if find_target is True), or the
        # default value plus shift2.
        if location is None:
            xdisp_locn = None
        else:
            xdisp_locn = location[row]

        outdata.field ("NELEM")[row] = nelem

        # These are pixel coordinates.
        pixel = N.arange (nelem, dtype=N.float64)
        ncoeff = disp_info.field ("nelem")[0]
        coeff = disp_info.field ("coeff")[0][0:ncoeff]
        if cosutil.findColumn (disp_info, "delta"):
            delta = disp_info.field ("delta")[0]
        else:
            delta = 0.

        x_offset = hdr.get ("x_offset", 0)

        # Correct for the extra pixels (if any) in the dispersion direction.
        pixel -= x_offset

        pixel += dpixel1                # shift will be 0 for a wavecal
        wavelength = cosutil.evalDisp (pixel, coeff, delta)
        del disp_info

        # S/N of the flat field
        snr_ff = getSnrFf (switches, reffiles, segment)

        axis = 2 - hdr["dispaxis"]          # 1 --> 1,  2 --> 0

        if corrtag:
            if info["detector"] == "FUV":
                axis_length = FUV_EXTENDED_X 
            else:
                axis_length = NUV_EXTENDED_X 
            (N_i, ERR_i, GC_i, BK_i, DQ_i, DQ_WGT_i, xd_locn) = \
                extractCorrtag (xi, eta, dq, epsilon, hdr["sdqflags"],
                    axis_length, snr_ff, hdr["exptime"], switches["backcorr"],
                    xtract_info, shift2, xdisp_locn)
        else:
            if xdisp_locn is None and find_target:
                y_nominal = xtract_info.field ("b_spec")[0] + shift2
                # search for the target spectrum
                # xxx not finished yet
                #xd_locn = xd_search.xdSearch (ifd_e["SCI"].data,
                #                ifd_e["DQ"].data, wavelength,
                #                axis, slope, y_nominal,
                #                x_offset, info["detector"], info["opt_elem"])
            else:
                xd_locn = xdisp_locn    # updated by extractSegment()
            (N_i, ERR_i, GC_i, BK_i, DQ_i, DQ_WGT_i, xd_locn) = \
             extractSegment (ifd_e["SCI"].data, ifd_c["SCI"].data,
                    ifd_e["DQ"].data, x_offset, hdr["sdqflags"],
                    snr_ff, hdr["exptime"], switches["backcorr"], axis,
                    xtract_info, shift2, xd_locn, find_target)
        updateExtractionKeywords (ofd[1].header, segment,
                                  slope, height, xd_locn)
        del xtract_info

        outdata.field ("SEGMENT")[row] = segment
        outdata.field ("EXPTIME")[row] = hdr["exptime"]
        outdata.field ("WAVELENGTH")[row][:] = wavelength
        outdata.field ("FLUX")[row][:] = 0.
        outdata.field ("ERROR")[row][:] = ERR_i
        outdata.field ("GROSS")[row][:] = GC_i
        outdata.field ("NET")[row][:] = N_i
        outdata.field ("BACKGROUND")[row][:] = BK_i
        outdata.field ("DQ")[row][:] = DQ_i
        outdata.field ("DQ_WGT")[row][:] = DQ_WGT_i

        row += 1

    # Remove unused rows, if any.
    if row < len (outdata):
        data = outdata[0:row]
        ofd[1].data = data.copy()
        del data

def postargOffset (phdr, dispaxis):
    """Get the offset to shift2 if postarg is non-zero.

    I don't think this should be used, but I'll leave the function here
    for the time being.  If it were to be used, the function value would
    be added to (or subtracted from?) shift2 in doExtract, e.g.:
        shift2 = 0.              # cross-dispersion direction
        shift2 += postargOffset (ifd_e[0].header, hdr["dispaxis"])

    @param phdr: primary header
    @type phdr: pyfits Header object
    @param dispaxis: dispersion axis (1 or 2)
    @type dispaxis: int
    @return: offset in pixels to be added to cross-dispersion location
    @rtype: float

    xxx The plate scale should be gotten from a header keyword.
    xxx The sign of the offset needs to be checked.
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
        postarg_xdisp = phdr.get ("postarg2", 0.)
    elif dispaxis == 2:
        postarg_xdisp = phdr.get ("postarg1", 0.)
    else:
        return 0.

    opt_elem = phdr["opt_elem"]

    return postarg_xdisp * plate_scale[opt_elem]

def getColumns (ifd_e, detector):
    """Get the appropriate columns from the events table extension.

    @param ifd_e: HDUList for the corrtag events table
    @type ifd_e: PyFITS HDUList object
    @param detector: detector name ("FUV" or "NUV")
    @type detector: string

    @return: columns from the corrtag table
    @rtype: tuple of arrays

    The returned columns xi, eta, dq and epsilon are as follows:
        xi is the array of positions in the dispersion direction
        eta is the array of positions in the cross-dispersion direction
        dq is the array of data quality flags
        epsilon is the array of weights (inverse flat field and deadtime
          correction)
    There is one element for each detected photon.
    """

    data = ifd_e[1].data

    if cosutil.findColumn (data, "xfull"):
        xi = data.field ("xfull")
    else:
        xi = data.field ("xdopp")
    if cosutil.findColumn (data, "yfull"):
        eta = data.field ("yfull")
    else:
        if detector == "FUV":
            eta = data.field ("ycorr")
        else:
            eta = data.field ("rawy")

    dq = data.field ("dq")
    epsilon = data.field ("epsilon")

    return (xi, eta, dq, epsilon)

def getSnrFf (switches, reffiles, segment):
    """Get the signal-to-noise ratio of the flat field data.

    @param switches: calibration switch values
    @type switches: dictionary
    @param reffiles: reference file names
    @type reffiles: dictionary
    @param segment: segment (or stripe) name
    @type segment: string

    @return: signal-to-noise ratio of the flat field
    @rtype: float

    If the flat-field correction has been done, this function reads the
    keyword SNR_FF from the appropriate header of the flat field image
    and returns that value; otherwise, this function returns zero.
    """

    if switches["flatcorr"] == "COMPLETE":
        fd_flat = pyfits.open (reffiles["flatfile"], mode="readonly")
        if segment in ["FUVA", "FUVB"]:
            flat_hdr = fd_flat[segment].header
        else:
            flat_hdr = fd_flat[1].header
        snr_ff = flat_hdr.get ("snr_ff", 0.)
        fd_flat.close()
        del fd_flat
    else:
        snr_ff = 0.

    return snr_ff

def extractSegment (e_data, c_data, e_dq_data, x_offset, sdqflags,
                snr_ff, exptime, backcorr, axis,
                xtract_info, shift2, xdisp_locn=None, find_target=False):

    """Extract a 1-D spectrum for one segment or stripe.

    This does the actual extraction, returning the results as a tuple.

    @param e_data: SCI data from the flt file ('e' for effective count rate)
    @type e_data: 2-D numpy array
    @param c_data: SCI data from the counts file (count rate)
    @type c_data: 2-D numpy array
    @param e_dq_data: DQ data from the flt file
    @type e_dq_data: 2-D numpy array
    @param x_offset: offset of the detector in the output array
    @type x_offset: int
    @param sdqflags: "serious" data quality flags
    @type sdqflags: int
    @param snr_ff: the signal-to-noise ratio of the flat field reference file
        (from the extension header of the flat field)
    @type snr_ff: float
    @param exptime: exposure time (seconds), from the header keyword
    @type exptime: float
    @param backcorr: "PERFORM" if background subtraction is to be done
    @type backcorr: int
    @param axis: the dispersion axis, 0 (Y) or 1 (X)
    @type axis: int
    @param xtract_info: one row of the xtractab
    @type xtract_info: PyFITS record object
    @param shift2: offset in the cross-dispersion direction
    @type shift2: float
    @param xdisp_locn: user-specified location in cross-dispersion direction
    @type xdisp_locn: int or float, or None if not specified
    @param find_target: search for the cross-disp location of the target?
    @type find_target: boolean

    @return: net count rate, error estimate, gross count rate, background
        count rate, data quality array, data quality weight array, xd_locn
    @rtype: tuple of seven 1-D arrays

    The xd_locn that is returned is either the user-specified value (if
    it was specified) or the location based on the wavecal; in either case,
    it's where the spectrum crosses the middle of the array, not the left
    edge of the array.

    An "_ij" suffix indicates a 2-D array; here they will all be sections
    extracted from full images.  An "_i" suffix indicates a 1-D array
    which is the result of summing the 2-D array with the same prefix in
    the cross-dispersion direction.  Variables beginning with a capital
    letter are included in the returned tuple.

      e_i       effective count rate, extracted from ifd_e[1].data
      GC_i      gross count rate, extracted from ifd_c[1].data
      BK_i      background count rate
      N_i       net count rate
      eps_i     effective count rate / gross count rate
      ERR_i     error estimate for net count rate
      DQ_i      data quality flags, bitwise OR of input DQ array
      DQ_WGT_i  data quality weight array
    """

    slope           = xtract_info.field ("slope")[0]
    b_spec          = xtract_info.field ("b_spec")[0]
    extr_height     = xtract_info.field ("height")[0]
    b_bkg1          = xtract_info.field ("b_bkg1")[0]
    b_bkg2          = xtract_info.field ("b_bkg2")[0]
    bkg_extr_height = xtract_info.field ("bheight")[0]
    bkg_smooth      = xtract_info.field ("bwidth")[0]

    axis_length = e_data.shape[axis]

    if xdisp_locn is None:
        if find_target:
            # (shift2, xd_locn) = xxx not implemented yet xxx
            b_spec = xd_locn - slope * (axis_length // 2)
        else:
            # add the shift to the nominal location; assign a value to xd_locn
            # (which will be returned but not otherwise used)
            b_spec += shift2
            xd_locn = b_spec + slope * (axis_length // 2 - x_offset)
    else:
        # use the user-specified value, but convert to b_spec, the intersection
        # with the left edge of the array
        b_spec = xdisp_locn - slope * (axis_length // 2 - x_offset)
        xd_locn = xdisp_locn

    # Compute the data quality and data quality weight arrays.
    DQ_i = N.zeros (axis_length, dtype=N.int16)
    if e_dq_data is not None:

        # Get data quality flags within extraction region.
        dq_ij = N.zeros ((extr_height, axis_length), dtype=N.int16)
        ccos.extractband (e_dq_data, axis, slope, b_spec, x_offset, dq_ij)
        # xxx replace with a C function
        DQ_i[:] = dq_ij[0].copy()
        for j in range (extr_height-1):
            DQ_i[:] = N.bitwise_or (DQ_i, dq_ij[j+1])

        # In bad_ij and bad_i, 0 means OK and 1 means bad
        bad_ij = N.zeros ((extr_height, axis_length), dtype=N.int32)
        bad_ij[:,:] = N.where (N.bitwise_and (dq_ij, sdqflags), 1, 0)
        bad_i = bad_ij.sum (axis=0)
        # Any bad pixel in extraction region?  DQ_WGT is a weight,
        # so 0 is bad and 1 is good.
        DQ_WGT_i = N.where (bad_i > 0, 0., 1.)
        del dq_ij, bad_ij, bad_i
    else:
        DQ_WGT_i = N.ones (axis_length, dtype=N.float32)

    e_ij = N.zeros ((extr_height, axis_length), dtype=N.float32)
    ccos.extractband (e_data, axis, slope, b_spec, x_offset, e_ij)

    GC_ij = N.zeros ((extr_height, axis_length), dtype=N.float32)
    ccos.extractband (c_data, axis, slope, b_spec, x_offset, GC_ij)

    e_i  = e_ij.sum (axis=0)
    GC_i = GC_ij.sum (axis=0)

    eps_i = e_i / N.where (GC_i <= 0., 1., GC_i)
    # default value when there are no counts
    eps_i = N.where (e_i == 0., 1., eps_i)
    del e_ij, e_i

    bkg_norm = float (extr_height) / (2. * float (bkg_extr_height))
    if backcorr == "PERFORM":
        BK1_ij = N.zeros ((bkg_extr_height, axis_length), dtype=N.float32)
        BK2_ij = N.zeros ((bkg_extr_height, axis_length), dtype=N.float32)
        ccos.extractband (c_data, axis, slope, b_bkg1, x_offset, BK1_ij)
        ccos.extractband (c_data, axis, slope, b_bkg2, x_offset, BK2_ij)
        BK_i = BK1_ij.sum (axis=0) + BK2_ij.sum (axis=0)
        BK_i = BK_i * bkg_norm
        boxcar (BK_i, (bkg_smooth,), output=BK_i, mode='nearest')
    else:
        BK_i = N.zeros (axis_length, dtype=N.float32)

    N_i = eps_i * (GC_i - BK_i)

    if snr_ff > 0.:
        term1_i = (N_i * exptime / (extr_height * snr_ff))**2
    else:
        term1_i = 0.
    term2_i = eps_i**2 * exptime * \
                (GC_i + BK_i * (bkg_norm / float (bkg_smooth)))
    ERR_i = N.sqrt (term1_i + term2_i) / exptime

    return (N_i, ERR_i, GC_i, BK_i, DQ_i, DQ_WGT_i, xd_locn)

def extractCorrtag (xi, eta, dq, epsilon, sdqflags,
                    axis_length, snr_ff, exptime, backcorr,
                    xtract_info, shift2, xdisp_locn=None):
    """Extract a 1-D spectrum for one segment or stripe.

    not finished
    """

    slope           = xtract_info.field ("slope")[0]
    b_spec          = xtract_info.field ("b_spec")[0]
    extr_height     = xtract_info.field ("height")[0]
    b_bkg1          = xtract_info.field ("b_bkg1")[0]
    b_bkg2          = xtract_info.field ("b_bkg2")[0]
    bkg_extr_height = xtract_info.field ("bheight")[0]
    bkg_smooth      = xtract_info.field ("bwidth")[0]

    zero_pixel = 0              # offset into output spectrum

    if xdisp_locn is None:
        # add the shift to the nominal location; assign a value to xd_locn
        b_spec += shift2
        xd_locn = b_spec + slope * (axis_length // 2)
    else:
        # use the user-specified value, but convert to the intersection
        # with the left edge of the array
        b_spec = xdisp_locn - slope * (axis_length // 2)
        xd_locn = xdisp_locn

    e_ij = N.zeros ((extr_height, axis_length), dtype=N.float64)
    ccos.xy_extract (xi, eta, e_ij, slope, b_spec,
                     zero_pixel, dq, sdqflags, epsilon)
    e_ij /= exptime
    e_i = e_ij.sum (axis=0)

    GC_ij = N.zeros ((extr_height, axis_length), dtype=N.float64)
    ccos.xy_extract (xi, eta, GC_ij, slope, b_spec,
                     zero_pixel, dq, sdqflags)
    GC_ij /= exptime
    GC_i = GC_ij.sum (axis=0)

    eps_i = e_i / N.where (GC_i <= 0., 1., GC_i)
    del e_ij, e_i

    bkg_norm = float (extr_height) / (2. * float (bkg_extr_height))
    if backcorr == "PERFORM":
        BK1_ij = N.zeros ((bkg_extr_height, axis_length),
                           dtype=N.float64)
        BK2_ij = N.zeros ((bkg_extr_height, axis_length),
                           dtype=N.float64)
        ccos.xy_extract (xi, eta, BK1_ij, slope, b_bkg1,
                         zero_pixel, dq, sdqflags)
        ccos.xy_extract (xi, eta, BK2_ij, slope, b_bkg2,
                         zero_pixel, dq, sdqflags)
        BK_i = BK1_ij.sum (axis=0) + BK2_ij.sum (axis=0)
        BK_i /= exptime
        BK_i *= bkg_norm
        boxcar (BK_i, (bkg_smooth,), output=BK_i, mode='nearest')
    else:
        BK_i = N.zeros (axis_length, dtype=N.float64)

    N_i = eps_i * (GC_i - BK_i)

    if snr_ff > 0.:
        term1_i = (N_i * exptime / (extr_height * snr_ff))**2
    else:
        term1_i = 0.
    term2_i = eps_i**2 * exptime * \
                (GC_i + BK_i * (bkg_norm / float (bkg_smooth)))
    ERR_i = N.sqrt (term1_i + term2_i) / exptime

    # dummy DQ array
    DQ_i = N.zeros (axis_length, dtype=N.int16)

    # dummy weight array
    DQ_WGT_i = N.ones (axis_length, dtype=N.float32)

    return (N_i, ERR_i, GC_i, BK_i, DQ_i, DQ_WGT_i, xd_locn)

def doFluxCorr (ofd, opt_elem, cenwave, aperture, tdscorr, reffiles):
    """Convert net counts to flux, updating flux and error columns.

    The correction to flux is made by dividing by the appropriate row
    in the phottab.  If a time-dependent sensitivity table (tdstab) has
    been specified, the flux and error will be corrected to the time of
    observation.

    @param ofd: HDUList for the output table; the primary header will be
        modified to set FLUXCORR to COMPLETE, and TDSCORR may be set to
        either COMPLETE or SKIPPED
    @type ofd: PyFITS HDUList object
    @param opt_elem: grating name
    @type opt_elem: string
    @param cenwave: central wavelength
    @type cenwave: integer
    @param aperture: PSA, BOA, WCA
    @type aperture: string
    @param tdscorr: calibration switch, time-dependent sensitivity correction
    @type tdscorr: string
    @param reffiles: dictionary of reference file names
    @type reffiles: dictionary
    """

    outdata = ofd[1].data
    nrows = outdata.shape[0]
    segment = outdata.field ("SEGMENT")
    wavelength = outdata.field ("WAVELENGTH")
    net = outdata.field ("NET")
    flux = outdata.field ("FLUX")
    error = outdata.field ("ERROR")

    # segment will be added to filter in the loop
    filter = {"opt_elem": opt_elem,
              "cenwave": cenwave,
              "aperture": aperture}

    phottab = reffiles["phottab"]
    for row in range (nrows):
        factor = N.zeros (len (flux[row]), dtype=N.float32)
        filter["segment"] = segment[row]
        phot_info = cosutil.getTable (phottab, filter)
        if phot_info is None:
            flux[row][:] = 0.
        else:
            # Interpolate sensitivity at each wavelength.
            wl_phot = phot_info.field ("wavelength")[0]
            sens_phot = phot_info.field ("sensitivity")[0]
            ccos.interp1d (wl_phot, sens_phot, wavelength[row], factor)
            factor = N.where (factor <= 0., 1., factor)
            flux[row][:] = net[row] / factor
            error[row][:] = error[row] / factor
    ofd[0].header["fluxcorr"] = "COMPLETE"

    # Compute an array of time-dependent correction factors (a potentially
    # different value at each wavelength), and divide the flux and error by
    # this array.

    if tdscorr == "PERFORM":
        tdstab = reffiles["tdstab"]
        t_obs = (ofd[1].header["expstart"] + ofd[1].header["expend"]) / 2.
        filter = {"opt_elem": opt_elem,
                  "aperture": aperture}
        # First check for dummy rows in the TDS table.  If there is no
        # pedigree column, assume all rows are good (i.e. not dummy).
        dummy = False           # initial value
        for row in range (nrows):
            filter["segment"] = segment[row]
            tds_info = cosutil.getTable (tdstab, filter, exactly_one=True)
            names = []
            for name in tds_info.names:
                names.append (name.lower())
            if "pedigree" not in names:
                break
            pedigree = tds_info.field ("pedigree")[0]
            if pedigree == "DUMMY":
                dummy = True
                cosutil.printWarning ("Current row in TDSTAB %s is dummy" % \
                                      tdstab, level=VERBOSE)
                cosutil.printContinuation ("for filter = %s," % \
                                           filter, level=VERBOSE)
                cosutil.printContinuation ("so TDSTAB will not be done.", \
                                           level=VERBOSE)
                break
        if dummy:
            ofd[0].header["tdscorr"] = "SKIPPED"
        else:
            for row in range (nrows):
                filter["segment"] = segment[row]
                # Get an array of factors vs. wavelength at the time of the obs.
                try:
                    tds_results = getTdsFactors (tdstab, filter, t_obs)
                except RuntimeError:    # no matching row in table
                    continue
                (wl_tds, factor_tds) = tds_results
                factor = N.zeros (len (flux[row]), dtype=N.float32)
                # Interpolate factor_tds at each wavelength.
                ccos.interp1d (wl_tds, factor_tds, wavelength[row], factor)
                flux[row][:] /= factor
                error[row][:] /= factor
            ofd[0].header["tdscorr"] = "COMPLETE"

def getTdsFactors (tdstab, filter, t_obs):
    """Get arrays of wavelengths and corresponding TDS factors.

    @param tdstab: name of the time-dependent sensitivity reference table
    @type tdstab: string
    @param filter: dictionary for selecting a row from tdstab
    @type filter: dictionary
    @param t_obs: time of the observation (MJD)
    @type t_obs: float

    @return: (wl_tds, factor_tds), where wl_tds is the array of wavelengths
        from the TDS table, and factor_tds is the corresponding array of
        time-dependent sensitivity factors, evaluated at the time of
        observation from the slope and intercept from the TDS table;
        if the time of observation is outside the range of times in the
        table, factor_tds will be independent of time and equal to the
        factor at the first or last time in the table respectively
    @rtype: tuple, or None
    """

    # Slope and intercept are specified for each of the nt entries in
    # the TIME column and for each of the nwl values in the WAVELENGTH
    # column.  nt and nwl should be at least 1.

    tds_info = cosutil.getTable (tdstab, filter, exactly_one=True)

    fd = pyfits.open (tdstab, mode="readonly")
    ref_time = fd[1].header.get ("ref_time", 0.)        # MJD
    fd.close()

    nwl = tds_info.field ("nwl")[0]
    nt = tds_info.field ("nt")[0]
    wl_tds = tds_info.field ("wavelength")[0]           # 1-D array
    time = tds_info.field ("time")[0]                   # 1-D array
    slope = tds_info.field ("slope")[0]                 # 2-D array
    intercept = tds_info.field ("intercept")[0]         # 2-D array

    # temporary, xxx
    # This section is needed because pyfits currently ignores TDIMi.
    maxt = len (time)
    maxwl = len (wl_tds)
    slope = N.reshape (slope, (maxt, maxwl))
    intercept = N.reshape (intercept, (maxt, maxwl))

    # Find the time interval that includes the time of observation.
    if nt == 1 or t_obs >= time[nt-1]:
        i = nt - 1
    else:
        for i in range (nt-1):
            if t_obs < time[i+1]:
                break

    # The slope in the tdstab is in percent per year.  Convert the time
    # interval to years, and convert the slope to fraction per year.
    # If the time of observation is before the first time in the table or
    # after the last time, the correction factor is to be the factor at
    # the first time or the last time respectively.  This is done by setting
    # delta_t to be the difference from the reference time to the first or
    # last time.
    if t_obs < time[0]:
        delta_t = (time[0] - ref_time) / DAYS_PER_YEAR
    elif t_obs > time[nt-1]:
        delta_t = (time[nt-1] - ref_time) / DAYS_PER_YEAR
    else:
        delta_t = (t_obs - ref_time) / DAYS_PER_YEAR
    slope[:] /= 100.

    # Take the slice [0:nwl] to avoid using elements that may not be valid,
    # and because the array of factors should be the same length as the
    # set of wavelengths that have been specified.
    wl_tds = wl_tds[0:nwl]
    factor_tds = delta_t * slope[i][0:nwl] + intercept[i][0:nwl]

    return (wl_tds, factor_tds)

def updateExtractionKeywords (hdr, segment, slope, height, xdisp_locn):
    """Update keywords giving the locations of extraction regions.

    arguments:
    hdr          output bintable hdu header
    segment      FUVA or FUVB; NUVA, NUVB, or NUVC
    slope        slope of spectrum
    height       height of extraction box
    xdisp_locn   location of the spectrum in the cross-dispersion direction
                 (where it crosses the middle of the detector)
    """

    key = "SP_LOC_" + segment[-1]           # SP_LOC_A, SP_LOC_B, SP_LOC_C
    hdr.update (key, xdisp_locn)
    key = "SP_SLP_" + segment[-1]           # SP_SLP_A, SP_SLP_B, SP_SLP_C
    hdr.update (key, slope)
    hdr.update ("SP_WIDTH", height)

def updateArchiveSearch (ofd):
    """Update the keywords giving min & max wavelengths, etc.

    argument:
    ofd         output (FITS HDUList object), table header modified in-place
    """

    phdr = ofd[0].header
    detector = phdr["detector"]
    outdata = ofd[1].data
    nrows = outdata.shape[0]
    wavelength = outdata.field ("WAVELENGTH")

    #phdr.update ("SPECRES", 20000.)
    #if detector == "FUV":
    #    phdr.update ("PLATESC", 0.094)  # arcsec / pixel, cross-disp direction
    #elif detector == "NUV":
    #    phdr.update ("PLATESC", 0.026)

    if nrows <= 0 or len (wavelength[0]) < 1:
        return

    minwave = wavelength[0][0]
    maxwave = wavelength[0][0]
    for row in range (nrows):
        minwave_row = N.minimum.reduce (wavelength[row])
        minwave = min (minwave, minwave_row)
        maxwave_row = N.maximum.reduce (wavelength[row])
        maxwave = max (maxwave, maxwave_row)

    phdr.update ("MINWAVE", minwave)
    phdr.update ("MAXWAVE", maxwave)
    phdr.update ("BANDWID", maxwave - minwave)
    phdr.update ("CENTRWV", (maxwave + minwave) / 2.)

def fixApertureKeyword (ofd, aperture, detector):
    """Replace aperture in output header if aperture is RelMvReq.

    arguments:
    ofd         output (FITS HDUList object), primary header modified in-place
    aperture    correct aperture name, without -FUV or -NUV
    detector    detector name
    """

    aperture_hdr = ofd[0].header.get ("aperture", NOT_APPLICABLE)
    if aperture_hdr == "RelMvReq":
        aperture_fixed = aperture + "-" + detector
        ofd[0].header.update ("aperture", aperture_fixed)
        cosutil.printWarning ("APERTURE reset from %s to %s" % \
                (aperture_hdr, aperture_fixed), level=VERBOSE)

def concatenateFUVSegments (infiles, output):
    """Concatenate the 1-D spectra for the two FUV segments into one file.

    arguments:
    infiles       list of input file names
    output        output file name
    """

    if len (infiles) != 2:
        cosutil.printMsg ("Internal error")
        raise RuntimeError, \
            "There should have been exactly two file names in " + infiles

    cosutil.printMsg ("Concatenate " + repr (infiles) + " --> " + output, \
                VERY_VERBOSE)

    ifd_0 = pyfits.open (infiles[0], mode="readonly")
    ifd_1 = pyfits.open (infiles[1], mode="readonly")

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
        cosutil.printError ("files are " + infiles[0] + infiles[1])
        raise RuntimeError, \
            "Files to concatenate must be for segments FUVA and FUVB."

    if seg_a[1].data is None:
        nrows_a = 0
    else:
        nrows_a = seg_a[1].data.shape[0]

    if seg_b[1].data is None:
        nrows_b = 0
    else:
        nrows_b = seg_b[1].data.shape[0]

    # Take output column definitions from input for segment A.
    cd = pyfits.ColDefs (seg_a[1])
    hdu = pyfits.new_table (cd, seg_a[1].header, nrows=nrows_a+nrows_b)

    # Copy data from input to output.
    copySegments (seg_a[1].data, nrows_a, seg_b[1].data, nrows_b, hdu.data)

    # Include segment-specific keywords from segment B.
    for key in ["stimb_lx", "stimb_ly", "stimb_rx", "stimb_ry",
                "stimb0lx", "stimb0ly", "stimb0rx", "stimb0ry",
                "stimbslx", "stimbsly", "stimbsrx", "stimbsry",
                "pha_badb", "phalowrb", "phaupprb",
                "sp_loc_b", "sp_slp_b",
                "shift1b", "shift2b", "dpixel1b"]:
        if seg_b[1].header.has_key (key):
            hdu.header.update (key, seg_b[1].header.get (key, -1.0))

    # If one of the segments has no data, use the other segment for the
    # primary header.  This is so the calibration switch keywords in the
    # output x1d file will be set appropriately.
    if nrows_a > 0 or nrows_b == 0:
        phdu = seg_a[0]
    else:
        phdu = seg_b[0]
    ofd = pyfits.HDUList (phdu)
    cosutil.updateFilename (ofd[0].header, output)
    if ofd[0].header.has_key ("segment"):
        ofd[0].header["segment"] = NOT_APPLICABLE   # we now have both segments
    ofd.append (hdu)

    # Update the "archive search" keywords.
    updateArchiveSearch (ofd)

    ofd.writeto (output, output_verify="fix")
    ifd_0.close()
    ifd_1.close()

    if phdu.header["statflag"]:
        cosutil.doSpecStat (output)

def copySegments (data_a, nrows_a, data_b, nrows_b, outdata):
    """Copy the two input tables to the output table.

    arguments:
    data_a        recarray object for segment A (may have no data)
    nrows_a       length of data_a (may be zero)
    data_b        recarray object for segment B (may have no data)
    nrows_b       length of data_b (may be zero)
    outdata       a recarray object with nrows_a + nrows_b rows
    """

    n = 0
    for i in range (nrows_a):
        outdata[n] = data_a[i]
        n += 1
    for i in range (nrows_b):
        outdata[n] = data_b[i]
        n += 1

def copyKeywordsToInput (output, input, incounts):
    """Copy extraction location keywords to the input headers.

    @param output: name of the output file for 1-D extracted spectra
    @type output: string
    @param input: name of either the flat-fielded count-rate image, or the
        corrtag table
    @type input: string
    @param incounts: name of the file containing the count-rate image,
        or None if input is the corrtag table
    @type incounts: string, or None
    """

    ofd = pyfits.open (output, mode="readonly")
    ifd_e = pyfits.open (input, mode="update")
    if incounts is not None:
        ifd_c = pyfits.open (incounts, mode="update")

    for key in ["sp_loc_a", "sp_loc_b", "sp_loc_c",
                "sp_slp_a", "sp_slp_b", "sp_slp_c",
                "sp_width"]:
        value = ofd[1].header.get (key, -999.)
        ifd_e[1].header.update (key, value)
        if incounts is not None:
            ifd_c[1].header.update (key, value)

    ofd.close()
    ifd_e.close()
    if incounts is not None:
        ifd_c.close()

def recomputeWavelengths (input):
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

    @param input: name of an x1d file for a wavecal
    @type input: string
    """

    fd = pyfits.open (input, mode="update")
    phdr = fd[0].header
    hdr = fd[1].header
    if hdr["naxis2"] == 0 or phdr["wavecorr"].upper() == "COMPLETE":
        fd.close()
        return
    cosutil.printMsg ("Updating wavelengths in %s" % input, VERY_VERBOSE)

    data = fd[1].data

    info = getinfo.getGeneralInfo (phdr, hdr)
    disptab = cosutil.expandFileName (phdr["disptab"])

    segment_col = data.field ("SEGMENT")
    nelem_col = data.field ("NELEM")
    wl_col = data.field ("WAVELENGTH")

    for row in range (len (data)):

        segment = segment_col[row]
        filter = {"segment": segment,
                  "opt_elem": info["opt_elem"],
                  "cenwave": info["cenwave"],
                  "aperture": "WCA"}
        # If the FPOFFSET column is present, include it in the filter.
        if cosutil.findColumn (disptab, "fpoffset"):
            filter["fpoffset"] = info["fpoffset"]
        disp_info = cosutil.getTable (disptab, filter)
        if disp_info is None:
            continue
        key = "shift1" + segment[-1]
        shift1 = hdr.get (key, 0.)

        # 'pixel' is an array of pixel coordinates.
        nelem = nelem_col[row]
        pixel = N.arange (nelem, dtype=N.float64)
        ncoeff = disp_info.field ("nelem")[0]
        coeff = disp_info.field ("coeff")[0][0:ncoeff]
        if cosutil.findColumn (disp_info, "delta"):
            delta = disp_info.field ("delta")[0]
        else:
            delta = 0.

        pixel -= shift1
        wl_col[row][0:nelem] = cosutil.evalDisp (pixel, coeff, delta)
        del disp_info

    phdr.update ("WAVECORR", "COMPLETE")

    fd.close()
