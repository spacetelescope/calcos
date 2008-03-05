import numpy as N
from convolve import boxcar
import pyfits
import cosutil
import ccos
import getinfo
from calcosparam import *       # parameter definitions

def extract1D (input, incounts, output):
    """Extract 1-D spectrum from 2-D image.

    arguments:
    input         name of the file containing the flat-fielded count-rate image
    incounts      name of the file containing the count-rate image
    output        name of the output file for 1-D extracted spectra
    """

    cosutil.printIntro ("Spectral Extraction")
    names = [("Input", input), ("Incounts", incounts), ("Output", output)]
    cosutil.printFilenames (names)
    cosutil.printMsg ("", VERBOSE)

    # Open the input files.
    ifd_e = pyfits.open (input, mode="readonly", memmap=0)
    ifd_c = pyfits.open (incounts, mode="readonly", memmap=0)
    # ifd_e = pyfits.open (input, mode="readonly", memmap=1)
    # ifd_c = pyfits.open (incounts, mode="readonly", memmap=1)

    phdr = ifd_e[0].header
    hdr = ifd_e[1].header
    info = getinfo.getGeneralInfo (phdr, hdr)
    switches = getinfo.getSwitchValues (phdr)
    reffiles = getinfo.getRefFileNames (phdr)
    is_wavecal = info["exptype"].find ("WAVE") >= 0
    if not is_wavecal and switches["wavecorr"] != "COMPLETE":
        cosutil.printWarning ("WAVECORR was not done for " + input)

    cosutil.printSwitch ("X1DCORR", switches)
    cosutil.printRef ("XTRACTAB", reffiles)
    cosutil.printRef ("DISPTAB", reffiles)
    cosutil.printSwitch ("BACKCORR", switches)
    cosutil.printSwitch ("STATFLAG", switches)
    cosutil.printSwitch ("FLUXCORR", switches)
    if switches["fluxcorr"] == "PERFORM":
        cosutil.printRef ("phottab", reffiles)
        cosutil.printRef ("tdstab", reffiles)

    # Create the output FITS header/data unit object.
    ofd = pyfits.HDUList (ifd_e[0])

    if info["npix"] == (0,):
        nrows = 0
        # Set the nominal length of the dispersion axis.
        if info["detector"] == "FUV":
            nelem = FUV_X
        else:
            nelem = NUV_Y
    else:
        if info["detector"] == "FUV":
            nrows = FUV_SPECTRA
        else:
            nrows = NUV_SPECTRA
        key = "naxis" + str (info["dispaxis"])  # dispaxis is one-indexed
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
    col.append (pyfits.Column (name="MAXDQ", format=rpt+"I"))
    col.append (pyfits.Column (name="AVGDQ", format=rpt+"I"))
    cd = pyfits.ColDefs (col)

    hdu = pyfits.new_table (cd, header=hdr, nrows=nrows)
    ofd.append (hdu)

    if nrows > 0:
        if info["detector"] == "FUV":
            segments = [info["segment"]]
        else:
            segments = ["NUVC", "NUVB", "NUVA"]
        doExtract (ifd_e, ifd_c, ofd, nelem,
                   segments, info, switches, reffiles, is_wavecal)
        if switches["fluxcorr"] == "PERFORM":
            doFluxCorr (ofd, info["opt_elem"], info["cenwave"],
                        info["aperture"], reffiles)

    # Update the output header.
    ofd[1].header["bitpix"] = 8         # temporary, xxx
    ofd[0].header.update ("nextend", 1)
    cosutil.updateFilename (ofd[0].header, output)
    # Delete image-specific world coordinate system keywords.
    ofd[1].header = cosutil.imageHeaderToTable (ofd[1].header)
    # Update the "archive search" keywords.
    updateArchiveSearch (ofd)
    # Fix the aperture keyword, if it's RelMvReq.
    fixApertureKeyword (ofd, info["aperture"], info["detector"])
    # Delete the exposure time keyword, which is now saved in a column.
    del ofd[1].header["exptime"]
    if nrows > 0:
        ofd[0].header["x1dcorr"] = "COMPLETE"
        if switches["backcorr"] == "PERFORM":
            ofd[0].header["backcorr"] = "COMPLETE"
        if switches["fluxcorr"] == "PERFORM":
            ofd[0].header["fluxcorr"] = "COMPLETE"

    ofd.writeto (output, output_verify="silentfix")
    del ofd
    ifd_e.close()
    ifd_c.close()

    if switches["statflag"] == "PERFORM":
        cosutil.doSpecStat (output)

def doExtract (ifd_e, ifd_c, ofd, nelem,
                     segments, info, switches, reffiles, is_wavecal):
    """Extract either FUV or NUV data.

    This calls a routine to do the extraction for one segment, and it
    assigns the results to one row of the output table.

    arguments:
    ifd_e       effective count rate data (FITS HDUList object)
    ifd_c       count rate data (FITS HDUList object)
    ofd         output (FITS HDUList object), modified in-place
    nelem       number of elements in current segment of output data
    row         write spectral data to this row number (zero indexed)
    segments    a list of the segment names, one for FUV, three for NUV
    info        dictionary of keywords and values
    switches    dictionary of calibration switch values
    reffiles    dictionary of reference file names
    is_wavecal  true if the observation is a wavecal, based on exptype
    """

    hdr = ifd_e[1].header
    outdata = ofd[1].data

    row = 0
    for segment in segments:

        filter = {"segment": segment,
                  "opt_elem": info["opt_elem"],
                  "cenwave": info["cenwave"],
                  "aperture": info["aperture"]}
        disp_info = cosutil.getTable (reffiles["disptab"], filter)
        xtract_info = cosutil.getTable (reffiles["xtractab"], filter)
        if disp_info is None or xtract_info is None:
            continue

        if is_wavecal:
            shift = 0.
        else:
            key = "pshift" + segment[-1]
            shift = hdr[key]

        # cross-dispersion direction
        key = "shift2" + segment[-1]
        shift2 = hdr.get (key, 0.)
        shift2 += postargOffset (ifd_e[0].header, hdr["dispaxis"])

        outdata.field ("NELEM")[row] = nelem

        # These are pixel coordinates.
        pixel = N.arange (nelem, dtype=N.float64)
        ncoeff = disp_info.field ("nelem")[0]
        coeff = disp_info.field ("coeff")[0][0:ncoeff]
        pixel -= shift                      # shift will be 0 for a wavecal
        wavelength = cosutil.evalDisp (pixel, coeff)
        wavelength_dq = N.where (wavelength < MIN_WAVELENGTH, \
                        DQ_BAD_WAVELENGTH, 0).astype (N.int16)
        del disp_info

        # Get the signal-to-noise ratio of the flat field data (at the pixel
        # level), if the flat field correction has been done.
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

        if info["obsmode"] == "ACCUM" and switches["helcorr"] == "PERFORM":
            wavelength += (wavelength * (-hdr["v_helio"]) / SPEED_OF_LIGHT)
            # wavelength *= ((1. - hdr["v_helio"]) / SPEED_OF_LIGHT)

        outdata.field ("WAVELENGTH")[row][:] = wavelength

        axis = 2 - hdr["dispaxis"]          # 1 --> 1,  2 --> 0

        (N_i, ERR_i, GC_i, BK_i, AVGDQ_i, MAXDQ_i) = \
            extractSegment (ifd_e["SCI"].data, ifd_c["SCI"].data,
                    ifd_e["DQ"].data, wavelength_dq,
                    snr_ff, hdr["exptime"], switches["backcorr"], axis,
                    xtract_info, shift2)
        updateExtractionKeywords (ofd[1].header, segment, xtract_info, shift2)
        del xtract_info

        outdata.field ("SEGMENT")[row] = segment
        outdata.field ("EXPTIME")[row] = hdr["exptime"]
        outdata.field ("FLUX")[row][:] = 0.
        outdata.field ("ERROR")[row][:] = ERR_i
        outdata.field ("GROSS")[row][:] = GC_i
        outdata.field ("NET")[row][:] = N_i
        outdata.field ("BACKGROUND")[row][:] = BK_i
        outdata.field ("AVGDQ")[row][:] = AVGDQ_i
        outdata.field ("MAXDQ")[row][:] = MAXDQ_i

        row += 1

    # Remove unused rows, if any.
    if row < len (outdata):
        data = outdata[0:row]
        ofd[1].data = data.copy()
        del data

def postargOffset (phdr, dispaxis):
    """Get the offset to shift2 if postarg is non-zero.

    @param phdr: primary header
    @type phdr: pyfits Header object
    @param dispaxis: dispersion axis (1 or 2)
    @type dispaxis: int
    @return: offset in pixels to be added to cross-dispersion location
    @rtype: float

    xxx This will have to be rewritten when the axes are reoriented.
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

def extractSegment (e_data, c_data, e_dq_data, wavelength_dq,
                snr_ff, exptime, backcorr, axis,
                xtract_info, shift2):

    """Extract a 1-D spectrum for one segment or stripe.

    This does the actual extraction, returning the results as a tuple.

    arguments:
    e_data           effective (i.e. flat fielded) count rate data array
    c_data           count rate data array
    e_dq_data        data quality array (associated with the flat-fielded data)
    wavelength_dq    data quality flags for wavelength out of range
    snr_ff           the signal-to-noise ratio of the flat field reference file
                     (from the extension header of the flat field)
    exptime          exposure time (seconds), from the header keyword
    backcorr         PERFORM if background subtraction is to be done
    axis             the dispersion axis, 0 (Y) or 1 (X)
    xtract_info      matching row of the xtractab, giving extraction parameters:
      slope            slope (pixels per pixel) of the extraction region
      b_spec           intercept of the middle of the extraction region
      extr_height      cross-dispersion width of the extraction region
      b_bkg1, b_bkg2   intercept of the background regions
      bkg_extr_height  cross-dispersion width of the background region
      bkg_smooth       width for boxcar smoothing of the background
    shift2           offset in cross-dispersion direction

    An "_ij" suffix indicates a 2-D array; here they will all be sections
    extracted from full images.  An "_i" suffix indicates a 1-D array
    which is the result of summing the 2-D array with the same prefix in
    the cross-dispersion direction.  Variables beginning with a capital
    letter are included in the returned tuple.

      e_i      effective count rate, extracted from ifd_e[1].data
      GC_i     gross count rate, extracted from ifd_c[1].data
      BK_i     background count rate
      N_i      net count rate
      eps_i    effective count rate / gross count rate
      ERR_i    error estimate for net count rate
      AVGDQ_i  average data quality
      MAXDQ_i  maximum data quality
    """

    slope           = xtract_info.field ("slope")[0]
    b_spec          = xtract_info.field ("b_spec")[0]
    extr_height     = xtract_info.field ("height")[0]
    b_bkg1          = xtract_info.field ("b_bkg1")[0]
    b_bkg2          = xtract_info.field ("b_bkg2")[0]
    bkg_extr_height = xtract_info.field ("bheight")[0]
    bkg_smooth      = xtract_info.field ("bwidth")[0]

    b_spec += shift2

    axis_length = e_data.shape[axis]

    e_ij  = N.zeros ((extr_height, axis_length), dtype=N.float32)
    ccos.extractband (e_data, axis, slope, b_spec, e_ij)

    GC_ij  = N.zeros ((extr_height, axis_length), dtype=N.float32)
    ccos.extractband (c_data, axis, slope, b_spec, GC_ij)

    e_i  = N.sum (e_ij, 0)
    GC_i = N.sum (GC_ij, 0)

    eps_i = e_i / N.where (GC_i <= 0., 1., GC_i)
    del e_ij, e_i

    bkg_norm = float (extr_height) / (2. * float (bkg_extr_height))
    if backcorr == "PERFORM":
        BK1_ij  = N.zeros ((bkg_extr_height, axis_length),
                        dtype=N.float32)
        BK2_ij  = N.zeros ((bkg_extr_height, axis_length),
                        dtype=N.float32)
        ccos.extractband (c_data, axis, slope, b_bkg1, BK1_ij)
        ccos.extractband (c_data, axis, slope, b_bkg2, BK2_ij)
        BK_i = N.sum (BK1_ij, 0) + \
               N.sum (BK2_ij, 0)
        BK_i = BK_i * bkg_norm
        boxcar (BK_i, (bkg_smooth,), output=BK_i, mode='nearest')
    else:
        BK_i  = N.zeros (axis_length, dtype=N.float32)

    N_i = eps_i * (GC_i - BK_i)

    if snr_ff > 0.:
        term1_i = (N_i * exptime / (extr_height * snr_ff))**2
    else:
        term1_i = 0.
    term2_i = eps_i**2 * exptime * \
                (GC_i + BK_i * (bkg_norm / float (bkg_smooth)))
    ERR_i = N.sqrt (term1_i + term2_i) / exptime

    DQ_ij  = N.zeros ((extr_height, axis_length), dtype=N.int16)
    if e_dq_data is not None:
        ccos.extractband (e_dq_data, axis, slope, b_spec, DQ_ij)
    # Flag regions where the wavelength is ridiculously low.
    N.bitwise_or (DQ_ij, wavelength_dq, DQ_ij)
    n_ij = GC_ij * exptime                      # counts in each pixel
    weighted_dq = n_ij * DQ_ij
    sum = N.sum (weighted_dq, 0)
    sum_weight = N.sum (n_ij, 0)         # counts in extraction region
    # replace 0 with 1 so we don't divide by zero
    sum_weight = N.where (sum_weight == 0., 1., sum_weight)
    avgdq = sum / sum_weight + 0.49999          # round off
    AVGDQ_i = avgdq.astype (N.int16)
    MAXDQ_i = N.maximum.reduce (DQ_ij, 0)

    return (N_i, ERR_i, GC_i, BK_i, AVGDQ_i, MAXDQ_i)

def doFluxCorr (ofd, opt_elem, cenwave, aperture, reffiles):
    """Convert net counts to flux, updating flux and error columns.

    The correction to flux is made by dividing by the appropriate row
    in the phottab.  If a time-dependent sensitivity table (tdstab) has
    been specified, the flux and error will be corrected to the time of
    observation.

    arguments:
    ofd         output (FITS HDUList object), modified in-place
    opt_elem    grating name
    cenwave     central wavelength (integer)
    aperture    PSA, BOA, WCA
    reffiles    dictionary of reference file names
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

    # Compute an array of time-dependent correction factors (a potentially
    # different value at each wavelength), and divide the flux and error by
    # this array.

    tdstab = reffiles["tdstab"]
    if tdstab != NOT_APPLICABLE:
        t_obs = (ofd[1].header["expstart"] + ofd[1].header["expend"]) / 2.
        filter = {"opt_elem": opt_elem,
                  "aperture": aperture}

        for row in range (nrows):
            filter["segment"] = segment[row]
            # Get factor at each wavelength in the tds table.
            try:
                (wl_tds, factor_tds) = getTdsFactors (tdstab, filter, t_obs)
            except RuntimeError:        # no matching row in table
                continue

            factor = N.zeros (len (flux[row]), dtype=N.float32)
            # Interpolate factor_tds at each wavelength.
            ccos.interp1d (wl_tds, factor_tds, wavelength[row], factor)

            flux[row][:] /= factor
            error[row][:] /= factor

def getTdsFactors (tdstab, filter, t_obs):
    """Get arrays of wavelengths and corresponding TDS factors.

    arguments:
    tdstab      name of the time-dependent sensitivity reference table
    filter      dictionary for selecting a row from tdstab
    t_obs       time of the observation (MJD)

    The function value is the tuple (wl_tds, factor_tds), where wl_tds
    is the array of wavelengths from the TDS table, and factor_tds is the
    corresponding array of time-dependent sensitivity factors, evaluated
    at the time of observation from the slope and intercept from the TDS
    table.
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
    delta_t = (t_obs - ref_time) / DAYS_PER_YEAR
    slope[:] /= 100.

    # Take the slice [0:nwl] to avoid using elements that may not be valid,
    # and because the array of factors should be the same length as the
    # set of wavelengths that have been specified.
    wl_tds = wl_tds[0:nwl]
    factor_tds = delta_t * slope[i][0:nwl] + intercept[i][0:nwl]

    return (wl_tds, factor_tds)

def updateExtractionKeywords (hdr, segment, xtract_info, shift2):
    """Update keywords giving the locations of extraction regions.

    arguments:
    hdr          output bintable hdu header
    segment      FUVA or FUVB; NUVA, NUVB, or NUVC
    xtract_info  extraction information, from xtractab
    shift2       offset in cross-dispersion direction
    """

    slope = xtract_info.field ("slope")[0]

    if segment[0:3] == "FUV":
        half_disp_axis = FUV_X / 2.
    else:
        half_disp_axis = NUV_Y / 2.

    key = "SP_LOC_" + segment[-1]           # SP_LOC_A, SP_LOC_B, SP_LOC_C
    location = xtract_info.field ("b_spec")[0] + shift2 + \
               slope * half_disp_axis
    hdr.update (key, location)
    hdr.update ("SP_SLOPE", slope)
    hdr.update ("SP_WIDTH", xtract_info.field ("height")[0])

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

    minwave = max (minwave, MIN_WAVELENGTH)
    maxwave = max (maxwave, MIN_WAVELENGTH)
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
                "pha_badb", "phalowrb", "phaupprb", "pshiftb",
                "sp_loc_b", "shift2b"]:
        hdu.header.update (key, seg_b[1].header.get (key, 0))

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
