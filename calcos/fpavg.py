from __future__ import absolute_import, division         # confidence high
import copy
import math
import numpy as np
import astropy.io.fits as fits
from astropy.stats import poisson_conf_interval
from . import cosutil
from .calcosparam import *       # parameter definitions

# Extract a slice of this height from the flat field in Spectrum.
XD_WIDTH = 15

def fpAvgSpec(input, output):
    """Average 1-D extracted FP-POS spectra.

    It is assumed that the arrays in all the input tables have the same
    length, but the output spectra will in general be longer than the
    input spectra.  The wavelengths in the input spectra will cover
    different ranges if the inputs are for different FP-POS positions.

    For FP-POS observations, the spectrum is moved over the detector
    from one exposure to another by tilting the grating.  This actually
    changes the dispersion relation, not just the zero point.  We assume,
    however, that the effect is the same as if we moved the detector with
    respect to a fixed spectrum.  That is, we use the same dispersion
    relation for all FP-POS positions, but the pixel numbers are offset
    by an amount that is constant for each FP-POS position.

    The data in the input table are count rates or fluxes.  When
    averaging the input spectra, we therefore weight by the exposure
    time.  For FUV data, the input spectra will be aligned to the
    nearest pixel and averaged without interpolation.  For NUV data,
    the pixel offset between input spectra need not be an integer.
    The output pixel size is the same as the input pixel size, and
    the contribution of a given input pixel to an output pixel is
    proportional to the area of overlap.  This is equivalent to linear
    interpolation, since the area of overlap (actually the length, since
    this is 1-D) is linearly related to the pixel shift.

    Parameters
    ----------
    input: list of str
        Name(s) of the input x1d files.

    output: str
        Name of a file for the averaged spectra.
    """

    nfiles = len(input)

    assert nfiles >= 1

    cosutil.printIntro("Average 1-D spectra")
    names = [("Input", repr(input)), ("Output", output)]
    cosutil.printFilenames(names)

    if nfiles == 1:
        oneInputFile(input[0], output)
    else:
        outspec = OutputX1D(input, output)

def oneInputFile(input, output):
    """Copy input to output, setting values to zero if dq_wgt is zero.

    Parameters
    ----------
    input: str
        Name of the (one) input x1d file

    output: str
        Name of a file for the modified copy of input
    """

    fd = fits.open(input, mode="copyonwrite")
    data = fd[1].data
    if data is None or len(data) == 0:
        fd.close()
        cosutil.copyFile(input, output)
        return

    flux = data.field("flux")
    error = data.field("error")
    gross = data.field("gross")
    error_lower = data.field("error_lower")
    err_frequentist_up = data.field("error")
    net = data.field("net")
    background = data.field("background")
    dq_wgt = data.field("dq_wgt")
    gcounts = data.field("gcounts")
    error = data.field("error")
    variance_flat = data.field("variance_flat")
    variance_counts = data.field("variance_counts")
    variance_bkg = data.field("variance_bkg")

    for row in range(len(data)):
        flux[row,:] = np.where(dq_wgt[row] <= 0., 0., flux[row])
        error[row,:] = np.where(dq_wgt[row] <= 0., 0., error[row])
        gross[row,:] = np.where(dq_wgt[row] <= 0., 0., gross[row])
        gcounts[row,:] = np.where(dq_wgt[row] <= 0., 0., gcounts[row])
        error_lower[row,:] = np.where(dq_wgt[row] <= 0., 0., error_lower[row])
        net[row,:] = np.where(dq_wgt[row] <= 0., 0., net[row])
        background[row,:] = np.where(dq_wgt[row] <= 0., 0., background[row])
        variance_flat[row,:] = np.where(dq_wgt[row] <= 0., 0., variance_flat[row])
        variance_counts[row,:] = np.where(dq_wgt[row] <= 0., 0., variance_counts[row])
        variance_bkg[row,:] = np.where(dq_wgt[row] <= 0., 0., variance_bkg[row])

    cosutil.updateFilename(fd[0].header, output)
    if cosutil.isProduct(output):
        asn_mtyp = fd[1].header.get("asn_mtyp", "missing")
        asn_mtyp = cosutil.modifyAsnMtyp(asn_mtyp)
        if asn_mtyp != "missing":
            fd[1].header["asn_mtyp"] = asn_mtyp
    list_keywords = [("MFPPOS", "fppos", fd[0].header["fppos"]),
                     ("MFPOFSET", "fpoffset", fd[0].header["fpoffset"]),
                     ("MCENWAVE", "cenwave", fd[0].header["cenwave"])]
    for (new_kwd, old_kwd, value) in list_keywords:
        str_value = str(value)
        fd[0].header.set(new_kwd, str_value, after=old_kwd)     # xxx temp
        # xxx fd[0].header.insert(old_kwd, (new_kwd, str_value), after=True)
    del_these = ["segment", "wavecals", "fppos", "fpoffset"]
    for keyword in del_these:
        if keyword in fd[0].header:
            del(fd[0].header[keyword])
    delSomeKeywords(fd[1].header)
    newfd = delExtraColumns(fd[1])
    fd[1] = newfd
    fd.writeto(output)
    fd.close()

def delSomeKeywords(hdr):
    """Delete exposure-specific keywords.

    Parameters
    ----------
    hdr: pyfits Header object
        Extension header to be modified
    """

    # These keywords are exposure-specific and are not relevant
    # to the entire association.
    for key in ["shift1a", "shift1b", "shift1c",
                "shift2a", "shift2b", "shift2c",
                "dpixel1a", "dpixel1b", "dpixel1c"]:
        if key in hdr:
            del(hdr[key])

def delExtraColumns(hdu):
    """Delete extra columns in the output file.

    Parameters
    ----------
    hdu: pyfits HDU object
        HDU object to be modified to be modified
    """

    columns_to_delete = ["DQ_OUTER", "BACKGROUND_PER_ROW", "NUM_EXTRACT_ROWS",
                         "ACTUAL_EE", "Y_LOWER_OUTER", "Y_UPPER_OUTER",
                         "Y_LOWER_INNER", "Y_UPPER_INNER"]

    table = hdu.data
    header = hdu.header
    columns = table.columns
    outcols = []
    for column in columns:
        if column.name in columns_to_delete:
            number = 0
            while(True):
                number = number + 1
                tstring = 'TTYPE' + str(number)
                try:
                    ttype = header[tstring]
                    if ttype in columns_to_delete:
                        del header[tstring]
                        tstring = 'TFORM' + str(number)
                        del header[tstring]
                        tstring = 'TUNIT' + str(number)
                        try:
                            del header[tstring]
                        except KeyError:
                            pass
                        tstring = 'TDISP' + str(number)
                        try:
                            del header[tstring]
                        except KeyError:
                            pass
                except KeyError:
                    break
        else:
            outcols.append(column)
    cd = fits.ColDefs(outcols)
    newhdu = fits.BinTableHDU.from_columns(cd, header=hdu.header)
    return newhdu

def makeStringList(inlist):
    """Construct a comma-separated string from elements of a list.

    Parameters
    ----------
    inlist: list
        Items to be written to a string.

    Returns
    -------
    outstr: str
        The elements of inlist as a string, with a comma and blank
        between each element.
    """

    nelem = len(inlist)
    outstr = ""
    for i in range(nelem):
        if i == 0:
            outstr += "%s" % str(inlist[i])
        else:
            outstr += ", %s" % str(inlist[i])

    return outstr

def pixelsFromWl(input_wavelength, output_wavelength):
    """Find pixel numbers in input corresponding to wavelengths in output.

    This function returns an array of pixel coordinates (floating point)
    in the input spectrum that have the same wavelengths as pixels 0, 1, 2,
    etc., in the output spectrum.

    An example may help.  Suppose pixel k in the output spectrum has
    wavelength wl (i.e. output_wavelength[k] = wl).  Suppose pixel n (an
    integer, just for example) in the input spectrum has the same wavelength
    wl.  Then pixel k in the array returned by this function would have
    value n.

    input_wavelength and output_wavelength do not need to have the same
    length.  As used by fpAvgSpec, there will typically be wavelengths in
    output_wavelength that lie outside the range of input_wavelength.  For
    those points, the element in the returned array can be less than or equal
    to zero, or it could be greater than nelem-1; these values should not be
    counted on to be accurate extrapolations.

    Parameters
    ----------
    input_wavelength: array_like
        Array of wavelengths in input spectrum

    output_wavelength: array_like
        Array of wavelengths in output spectrum

    Returns
    -------
    array_like, same data type as wavelength arrays
        Pixel numbers (but not integer values) in input spectrum
    """

    nelem = len(input_wavelength)

    avgdisp = (input_wavelength[-1] - input_wavelength[0]) / (nelem - 1.)

    # disp will be the dispersion at each pixel of the input wavelengths.
    disp = input_wavelength.copy()
    disp[1:nelem-1] = (input_wavelength[2:nelem] -
                       input_wavelength[0:nelem-2]) / 2.
    disp[0] = input_wavelength[1] - input_wavelength[0]
    disp[nelem-1] = input_wavelength[nelem-1] - input_wavelength[nelem-2]

    # x0 is a first estimate of the pixel numbers.
    x0 = (output_wavelength - input_wavelength[0]) / avgdisp
    x0 = np.where(x0 < 0., 0., x0)
    ix0 = x0.astype(np.int32)
    ix0 = np.where(ix0 > nelem-1, nelem-1, ix0)

    # wavelengths in input at pixels ix0 are input_wavelength[ix0]
    diff = (output_wavelength - input_wavelength[ix0])

    # x1 should be very close to the correct pixel numbers.
    x1 = ix0 + diff / disp[ix0]
    x1 = np.where(x1 < 0., 0., x1)
    ix1 = x1.astype(np.int32)
    ix1 = np.where(ix1 > nelem-1, nelem-1, ix1)

    diff = (output_wavelength - input_wavelength[ix1])
    ipixel = ix1 + diff / disp[ix1]

    return ipixel

class OutputX1D(object):
    """Average 1-D FP-POS spectra.

    Parameters
    ----------
    input: list of str
        Input file names

    output: str
        Output file name

    Attributes
    ----------
    input
    output

    keywords: dictionary
        Relevant keywords and values, e.g. detector

    inspec: list of Spectrum objects
        Input spectra

    segments: list of str
        Segment names found in input x1d tables

    ofd: pyfits HDUList object
        Pyfits object for output file

    nrows: int
        Number of rows to be written to the output table

    output_nelem: int
        Number of elements to use when allocating output arrays

    output_wl_range: dictionary of two-element tuples
        Smallest and largest wavelengths in an output spectrum; key is
        segment or stripe

    output_dispersion: dictionary
        Dispersion (Angstroms per pixel) in an output spectrum, key is
        segment or stripe
    """

    def __init__(self, input, output):
        """Constructor."""

        self.input = input
        self.output = output
        self.keywords = {}
        self.inspec = []
        self.segments = []
        self.ofd = None
        self.nrows = 0
        self.output_nelem = 1
        self.output_wl_range = {}
        self.output_dispersion = {}
        # This is the index of the element of self.inspec that has the
        # maximum value of nelem.  We'll use this spectrum as the template
        # for column definitions for the output table.
        self.index_max_nelem = 0

        # Create a list of Spectrum objects, and get info from headers.
        self.getInputInfo()

        # Check that the data in each Spectrum is comparable to the others.
        self.compareX1d()

        # Compute length of output arrays.
        self.computeOutputInfo()

        # Create ofd, the output pyfits object.
        self.createOutput()

        # Fill in the data in the output table.
        for segment in self.segments:           # for each output row ...
            osp = OutputSpectrum(self.ofd, self.inspec, self.keywords,
                                 segment, self.output_wl_range[segment],
                                 self.output_dispersion[segment])
        if cosutil.isProduct(self.output):
            asn_mtyp = self.ofd[1].header.get("asn_mtyp", "missing")
            asn_mtyp = cosutil.modifyAsnMtyp(asn_mtyp)
            if asn_mtyp != "missing":
                self.ofd[1].header["asn_mtyp"] = asn_mtyp
        self.updateArchiveSearch(self.ofd)      # minwave & maxwave
        newhdu = delExtraColumns(self.ofd[1])
        self.ofd[1] = newhdu
        self.ofd.writeto(self.output)

        if self.keywords["statflag"]:
            cosutil.doSpecStat(self.output)

    def getInputInfo(self):
        """Get info and data from input files.

        This routine creates Spectrum objects (one for each row of each input
        table) and appends them to the inspec list, gets keywords from the
        input headers, and determines the number of rows (nrows) that the
        output table should have.
        """

        # These are for averaging the global count rates.  The elements are
        # for NUV, FUVA and FUVB respectively.
        sum_globrate = [0., 0., 0.]     # incremented for each row in each file
        sum_exptime = [0., 0., 0.]      # exptime from the column
        avg_globrate = [-1., -1., -1.]  # average values for NUV, FUVA, FUVB
        # This is for updating the exptime, exptimea, exptimeb keywords.
        sum_exptime_kwd = [0., 0., 0.]  # exptime from the header keywords

        first = True            # true for first input file
        for input in self.input:
            ifd = fits.open(input, mode="readonly")
            phdr = ifd[0].header
            hdr = ifd[1].header
            # Get keyword values.
            if first:
                detector = phdr["detector"]
                opt_elem = phdr["opt_elem"]
                cenwave_list = [phdr["cenwave"]]    # may have multiple values
                fppos_list = [phdr["fppos"]]        # may have multiple values
                fpoffset_list = [phdr["fpoffset"]]  # may have multiple values
                (aperture, message) = cosutil.getApertureKeyword(phdr)
                statflag = phdr.get("statflag", False)
                sum_plantime = hdr["plantime"]
                expstart = hdr["expstart"]
                expend = hdr["expend"]
                first = False
            else:
                sum_plantime += hdr["plantime"]
                expstart = min(expstart, hdr["expstart"])
                expend = max(expend, hdr["expend"])
                if phdr["cenwave"] not in cenwave_list:
                    cenwave_list.append(phdr["cenwave"])
                if phdr["fppos"] not in fppos_list:
                    fppos_list.append(phdr["fppos"])
                if phdr["fpoffset"] not in fpoffset_list:
                    fpoffset_list.append(phdr["fpoffset"])
            fpoffset = phdr["fpoffset"]

            if ifd[1].data is not None:
                nrows = len(ifd[1].data)
                # for each row in the current input table
                for row in range(nrows):
                    sp = Spectrum(ifd, row, fpoffset)
                    segment = sp.segment
                    if segment not in self.segments:
                        self.segments.append(segment)
                    self.inspec.append(sp)
                    if sp.segment == "FUVA":
                        sum_exptime_kwd[1] += hdr.get("exptimea",
                                              default=hdr.get("exptime", 0.))
                        globrate = hdr.get("globrt_a", -1.)
                        if globrate >= 0.:
                            sum_globrate[1] += (globrate * sp.exptime)
                            sum_exptime[1] += sp.exptime
                    elif sp.segment == "FUVB":
                        sum_exptime_kwd[2] += hdr.get("exptimeb",
                                              default=hdr.get("exptime", 0.))
                        globrate = hdr.get("globrt_b", -1.)
                        if globrate >= 0.:
                            sum_globrate[2] += (globrate * sp.exptime)
                            sum_exptime[2] += sp.exptime
                    elif detector == "NUV" and row == 0:
                        sum_exptime_kwd[0] += hdr["exptime"]
                        globrate = hdr["globrate"]
                        if globrate >= 0.:
                            sum_globrate[0] += (globrate * sp.exptime)
                            sum_exptime[0] += sp.exptime

            ifd.close()

        for i in range(3):
            if sum_exptime[i] > 0.:
                avg_globrate[i] = sum_globrate[i] / sum_exptime[i]

        # number of rows to be written to the output table
        self.nrows = len(self.segments)

        if detector == "NUV":
            exptime = sum_exptime_kwd[0]
        else:
            exptime = max(sum_exptime_kwd[1], sum_exptime_kwd[2])
        self.keywords = {
             "detector": detector,
             "opt_elem": opt_elem,
             "cenwave":  cenwave_list,          # this is a list
             "fppos":    fppos_list,            # this is a list
             "fpoffset": fpoffset_list,         # this is a list
             "aperture": aperture,
             "exptime":  exptime,
             "exptimea": sum_exptime_kwd[1],
             "exptimeb": sum_exptime_kwd[2],
             "expstart": expstart,
             "expend":   expend,
             "expstrtj": expstart + MJD_TO_JD,
             "expendj":  expend + MJD_TO_JD,
             "plantime": sum_plantime,
             "globrate": avg_globrate[0],       # average for NUV spectra
             "globrt_a": avg_globrate[1],       # average for FUVA spectra
             "globrt_b": avg_globrate[2],       # average for FUVB spectra
             "statflag": statflag}

    def compareX1d(self):
        """Check that the rows of two x1d tables contain comparable info.

        Currently, the only check is on the array sizes.
        """

        for sp in self.inspec:
            if sp.nelem != self.inspec[0].nelem:
                raise RuntimeError("x1d tables have different array sizes.")

    def computeOutputInfo(self):
        """Compute length of output arrays, and info for output wavelengths.

        This routine assigns values to the attributes output_nelem,
        index_max_nelem, output_wl_range, and output_dispersion.
        """

        if len(self.inspec) < 1:
            if self.keywords["detector"] == "FUV":
                self.output_nelem = FUV_EXTENDED_X
            elif self.keywords["detector"] == "NUV":
                self.output_nelem = NUV_EXTENDED_X
            else:
                self.output_nelem = 1
            return

        # Find the maximum input nelem, and set output_nelem to that value.
        # The input nelem should really be all the same, but just in case
        # one of them is zero, we need to be able to skip that one.
        # Also set self.index_max_nelem, which we'll use in createOutput
        # for getting the initial column definitions for the output table.
        min_output_nelem = -1
        for (k, sp) in enumerate(self.inspec):
            if sp.nelem > min_output_nelem:
                self.index_max_nelem = k
                min_output_nelem = sp.nelem

        # Find the wavelength and dispersion for each segment.
        self.output_wl_range = {}
        self.output_dispersion = {}
        for segment in self.segments:
            min_wl = 1.e9
            max_wl = 1.
            min_dispersion = 1.e9
            for sp in self.inspec:
                if sp.segment == segment:
                    if sp.nelem < 2:
                        continue
                    min_wl = min(min_wl, sp.wavelength[0])
                    max_wl = max(max_wl, sp.wavelength[-1])
                    min_dispersion = min(min_dispersion,
                        (sp.wavelength[-1] - sp.wavelength[0]) / (sp.nelem - 1))
            self.output_wl_range[segment] = (min_wl, max_wl)
            self.output_dispersion[segment] = min_dispersion

        # Determine the number of elements we will need for the output
        # spectra.  The output array size should be at least as large as the
        # arrays in the input spectra (min_output_nelem)
        output_nelem = min_output_nelem         # initial value
        for segment in self.segments:
            (min_wl, max_wl) = self.output_wl_range[segment]
            dispersion = self.output_dispersion[segment]
            if dispersion <= 0.:
                continue
            nelem = (max_wl - min_wl) / dispersion
            nelem = int(round(math.ceil(nelem)))
            output_nelem = max(output_nelem, nelem)
        if output_nelem > 0:
            self.output_nelem = output_nelem

    def createOutput(self):
        """Create pyfits object for output file."""

        # Get header info from the input.
        ifd = fits.open(self.input[self.index_max_nelem],
                        mode="copyonwrite")
        detector = ifd[0].header["detector"]

        primary_hdu = fits.PrimaryHDU(header=ifd[0].header)
        cosutil.updateFilename(primary_hdu.header, self.output)
        # Add new keywords with comma-separated values.
        list_keywords = [("MFPPOS", "fppos", self.keywords["fppos"]),
                         ("MFPOFSET", "fpoffset", self.keywords["fpoffset"]),
                         ("MCENWAVE", "cenwave", self.keywords["cenwave"])]
        for (new_kwd, old_kwd, list_value) in list_keywords:
            str_value = makeStringList(list_value)
            primary_hdu.header.set(new_kwd, str_value, after=old_kwd)   # xxx
            # xxx primary_hdu.header.insert(old_kwd, (new_kwd, str_value),
            # xxx                           after=True)
        del_these = ["segment", "wavecals", "fppos", "fpoffset"]
        for keyword in del_these:
            if keyword in primary_hdu.header:
                del(primary_hdu.header[keyword])
        if len(self.keywords["cenwave"]) > 1:
            del(primary_hdu.header["cenwave"])
        ofd = fits.HDUList(primary_hdu)

        rpt = str(self.output_nelem)    # used for defining columns

        # Define the columns explicitly, rather than using an input table
        # as a template and then modifying the lengths of arrays (see below),
        # because the modified columns kept reverting to the original length.
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
                   unit="count /s"))
        col.append(fits.Column(name="GROSS", format=rpt+"E",
                   unit="count /s"))
        col.append(fits.Column(name="GCOUNTS", format=rpt+"E",
                   unit="count"))
        col.append(fits.Column(name="VARIANCE_FLAT", format=rpt+"E",
                   unit="count"))
        col.append(fits.Column(name="VARIANCE_COUNTS", format=rpt+"E",
                   unit="count"))
        col.append(fits.Column(name="VARIANCE_BKG", format=rpt+"E",
                   unit="count"))
        col.append(fits.Column(name="NET", format=rpt+"E",
                   unit="count /s"))
        col.append(fits.Column(name="BACKGROUND", format=rpt+"E",
                   unit="count /s"))
        col.append(fits.Column(name="DQ", format=rpt+"I"))
        col.append(fits.Column(name="DQ_WGT", format=rpt+"E"))
        cd = fits.ColDefs(col)

        # Modify some of the output columns.
        #cd = ifd[1].columns             # this is a ColDefs object
        #col_names = cd.names
        #col_formats = cd.formats
        #ncols = len(col_names)
        # xxx x = ifd[1].data                 # xxx touch the data
        #for i in range(ncols):
        #    fmt = col_formats[i]
        #    if fmt[-1] in ["D", "E", "I", "J"] and fmt[0] in "123456789":
        #        x = ifd[1].data         # xxx touch the data
        #        newfmt = rpt + fmt[-1]
        #        cd.change_attrib(col_names[i], "format", newfmt)

        # Create output HDU for the table.
        newhdu = delExtraColumns(ifd[1])
        ifd[1].header = newhdu.header
        hdu = fits.BinTableHDU.from_columns(cd, header=ifd[1].header,
                                            nrows=self.nrows)

        hdu.header["exptime"] = self.keywords["exptime"]
        if detector == "FUV":
            hdu.header["exptimea"] = self.keywords["exptimea"]
            hdu.header["exptimeb"] = self.keywords["exptimeb"]

        hdu.header["expstart"] = self.keywords["expstart"]
        hdu.header["expend"] = self.keywords["expend"]
        hdu.header["expstrtj"] = self.keywords["expstrtj"]
        hdu.header["expendj"] = self.keywords["expendj"]
        hdu.header["plantime"] = self.keywords["plantime"]
        if self.keywords["globrate"] >= 0.:
            hdu.header["globrate"] = round(self.keywords["globrate"], 4)
        if self.keywords["globrt_a"] >= 0.:
            hdu.header["globrt_a"] = round(self.keywords["globrt_a"], 4)
        if self.keywords["globrt_b"] >= 0.:
            hdu.header["globrt_b"] = round(self.keywords["globrt_b"], 4)

        # Delete some keywords because they are specific to one exposure.
        delSomeKeywords(hdu.header)

        ofd.append(hdu)
        self.fpInitData(ofd)            # initialize data in output hdu

        ifd.close()

        self.ofd = ofd

    def fpInitData(self, ofd):
        """Initialize the output data block.

        Two scalar columns, SEGMENT and NELEM, will be set to their actual
        values.  EXPTIME and the array columns will be initialized to zero.
        """

        ofd[1].data.field("segment")[:] = self.segments
        ofd[1].data.field("nelem")[:] = self.output_nelem

        ofd[1].data.field("exptime")[:] = 0.

        ofd[1].data.field("wavelength")[:] = 0.
        ofd[1].data.field("flux")[:] = 0.
        ofd[1].data.field("error")[:] = 0.
        ofd[1].data.field("gross")[:] = 0.
        ofd[1].data.field("error_lower")[:] = 0.
        ofd[1].data.field("VARIANCE_FLAT")[:] = 0.
        ofd[1].data.field("VARIANCE_COUNTS")[:] = 0.
        ofd[1].data.field("VARIANCE_BKG")[:] = 0.
        ofd[1].data.field("gcounts")[:] = 0.
        ofd[1].data.field("net")[:] = 0.
        ofd[1].data.field("background")[:] = 0.
        ofd[1].data.field("dq")[:] = DQ_PIXEL_OUT_OF_BOUNDS
        ofd[1].data.field("dq_wgt")[:] = 0.

    def updateArchiveSearch(self, ofd):
        """Update the keywords giving min & max wavelengths.

        ofd: pyfits HDUList object
            For the output file, primary header modified in-place
        """

        phdr = ofd[0].header
        outdata = ofd[1].data
        nrows = outdata.shape[0]
        wavelength = outdata.field("WAVELENGTH")
        dq_wgt = outdata.field("DQ_WGT")

        if nrows <= 0 or len(wavelength[0]) < 1:
            return

        nelem = len(wavelength[0])
        # This initial value assumes wavelengths increase with pixel number.
        minwave = wavelength[0][nelem-1]
        maxwave = wavelength[0][0]
        for row in range(nrows):
            if dq_wgt[row].sum(dtype=np.float64) <= 0:
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

class Spectrum(object):
    """One row of an input spectrum.

    Parameters
    ----------
    ifd: pyfits HDUList object
        The list of header/data objects for an input file

    row: int
        Row number (zero indexed) in the current input file

    fpoffset: int
        Value of the FPOFFSET keyword for the current input file

    Attributes
    ----------
    exptime: float
        exposure time (seconds) for this input spectrum

    segment: str
        segment or stripe name for the current row

    nelem: int
        number of elements in the arrays

    wavelength: array_like
        wavelengths for the current row

    flux: array_like
        flux values

    error: array_like
        error estimates for the flux

    gross: array_like
        gross values (count rate)

    gcounts: array_like
        gross counts

    net: array_like
        net values

    background: array_like
        background values

    dq: array_like
        data quality flags

    dq_wgt: array_like
        weights to account for pixels excluded due to data quality

    fpoffset
    """

    def __init__(self, ifd, row=0, fpoffset=0):
        """Constructor."""

        self.segment = ifd[1].data.field("segment")[row]
        self.exptime = ifd[1].data.field("exptime")[row]
        self.nelem = ifd[1].data.field("nelem")[row]
        self.wavelength = ifd[1].data.field("wavelength")[row]
        self.flux = ifd[1].data.field("flux")[row]
        self.error = ifd[1].data.field("error")[row]
        self.gross = ifd[1].data.field("gross")[row]
        self.error_lower = ifd[1].data.field("error_lower")[row]
        self.variance_flat = ifd[1].data.field("variance_flat")[row]
        self.variance_counts = ifd[1].data.field("variance_counts")[row]
        self.variance_bkg = ifd[1].data.field("variance_bkg")[row]
        self.gcounts = ifd[1].data.field("gcounts")[row]
        self.net = ifd[1].data.field("net")[row]
        self.background = ifd[1].data.field("background")[row]
        self.dq = ifd[1].data.field("dq")[row]
        self.dq_wgt = ifd[1].data.field("dq_wgt")[row]
        self.fpoffset = fpoffset

        # These are used for weighting by the flat field.
        self.origin_ff = np.zeros(2, dtype=np.int32)
        self.data_ff = None
        self.state_ff = "empty"

        # Read the flat field.
        phdr = ifd[0].header
        obsmode = phdr["obsmode"]
        doppcorr = phdr["doppcorr"]
        detector = phdr["detector"]
        hdr = ifd[1].header
        seg = self.segment[-1].lower()
        if phdr.get("exptype", default="missing").startswith("EXTERNAL"):
            shift1 = hdr["shift1" + seg]
            shift2 = hdr["shift2" + seg]
        else:
            shift1 = 0.
            shift2 = 0.
        sp_loc = hdr["sp_loc_" + seg]
        x_offset = hdr.get("x_offset", default=0)
        expstart = hdr["expstart"]
        expend = hdr["expend"]

        self.getFlatField(ifd)
        if self.data_ff is not None:
            self.collapseFlatField(sp_loc, shift2, width=XD_WIDTH)
            (doppmag, doppzero, orbitper) = \
                        self.getDopplerParam(doppcorr, obsmode,
                                             detector, hdr)
            self.shiftFlatField(shift1, x_offset,
                                doppcorr, doppmag, doppzero, orbitper,
                                expstart, expend)

    def getFlatField(self, ifd):
        """Read the flat field data.

        Values will be assigned to self.origin_ff, self.data_ff, and
        self.state_ff.

        Parameters
        ----------
        ifd: pyfits HDUList object
            The list of header/data objects for the current input file
        """

        phdr = ifd[0].header
        flatcorr = phdr["flatcorr"]
        if flatcorr != "COMPLETE" and flatcorr != "PERFORM":
            return
        flatfile = phdr["flatfile"]
        detector = phdr["detector"]
        segment = self.segment
        reffiles = {}
        reffiles["flatfile"] = cosutil.expandFileName(flatfile)
        fd = fits.open(reffiles["flatfile"])
        if detector == "FUV":
            self.data_ff = fd[(segment,1)].data.copy()
            hdr_ff = fd[(segment,1)].header
        else:
            self.data_ff = fd[1].data.copy()
            hdr_ff = fd[1].header
        self.origin_ff[0] = hdr_ff.get("origin_y", default=0)
        self.origin_ff[1] = hdr_ff.get("origin_x", default=0)
        fd.close()
        self.state_ff = "2-D, as read from reference file"

    def collapseFlatField(self, sp_loc, shift2, width):
        """Average a slice of self.data_ff, replacing full array.

        Parameters
        ----------
        sp_loc: float
            The value of keyword SP_LOC_[ABC], the Y location where the
            spectrum was extracted (where it crossed the middle of the
            detector).

        shift2: float
            The wavecal offset in Y.

        width: float
            The width in Y (i.e. the height) of the slice of the flat
            field to be extracted.
        """

        if self.data_ff is None:
            return
        height_ff = self.data_ff.shape[0]
        if height_ff < 1:
            return

        lowlim = sp_loc - width // 2 + shift2 - self.origin_ff[0]
        lowlim = int(round(lowlim))
        highlim = int(lowlim + width)        # upper limit of a slice
        if highlim < 0 or lowlim >= height_ff:
            cosutil.printWarning("Target is outside range of flat field")
            self.data_ff = self.data_ff.mean(axis=0, dtype=np.float64)
        else:
            lowlim = max(lowlim, 0)
            highlim = min(highlim, height_ff)
            height = highlim - lowlim           # height of slice
            if height > 0:
                self.data_ff = \
                self.data_ff[lowlim:highlim,:].mean(axis=0,
                                                  dtype=np.float64)
            else:
                cosutil.printError("Height of flat field slice is %d" % height)
                self.data_ff = self.data_ff.mean(axis=0, dtype=np.float64)
        self.state_ff = "slice taken, averaged to 1-D"

    def getDopplerParam(self, doppcorr, obsmode, detector, hdr):
        """Read Doppler parameters from the header.

        Parameters
        ----------
        doppcorr: str
            PERFORM, COMPLETE, OMIT, SKIPPED.

        obsmode: str
            TIME-TAG or ACCUM.

        detector: str
            FUV or NUV.

        hdr: pyfits Header object
            The header of the first extension.

        Returns
        -------
        (doppmag, doppzero, orbitper): tuple of three floats
        """

        if doppcorr == "OMIT" or doppcorr == "SKIPPED":
            doppmag  = 0.
            doppzero = 0.
            orbitper = 5760.
        elif obsmode == "TIME-TAG":
            # Get wavelength and dispersion from spectrum near middle.
            nelem = len(self.wavelength)
            if detector == "FUV":
                middle = nelem // 2
            else:
                middle = hdr.get("x_offset", default=0) + NUV_X // 2
            wavelength = self.wavelength[int(middle)]
            low = int(middle - 50)
            high = int(middle + 50)
            if low > 0 and high < nelem:
                disp = (self.wavelength[high] - self.wavelength[low]) / 100.
            else:
                raise RuntimeError("wavelength array is too short, %d" % nelem)
            doppmag = (hdr["doppmagv"] / SPEED_OF_LIGHT) * (wavelength / disp)
            doppzero = hdr["doppzero"]
            orbitper = hdr["orbitper"]

        else:               # ACCUM
            doppmag  = hdr["dopmagt"]
            doppzero = hdr["dopzerot"]
            orbitper = hdr["orbtpert"]

        return (doppmag, doppzero, orbitper)

    def shiftFlatField(self, shift1, x_offset,
                       doppcorr, doppmag, doppzero, orbitper,
                       expstart, expend):
        """

        Parameters
        ----------
        shift1: float
            OSM (wavecal) shift in dispersion direction.

        x_offset: int
            Value of X_OFFSET keyword (buffer to avoid losing data due to
            fpoffset shift).

        doppcorr: str
            PERFORM, OMIT, etc.

        doppmag: float
            Magnitude of Doppler shift in pixels.

        doppzero: float
            Time (MJD) when the orbital Doppler shift is zero and
            increasing.

        orbitper: float
            Orbital period (seconds) of HST.

        expstart: float
            Exposure start time (MJD).

        expend: float
            Exposure end time (MJD).
        """

        if self.data_ff is None:
            return
        nelem = len(self.wavelength)
        if nelem < 1:
            return
        if len(self.data_ff) > nelem:
            cosutil.printWarning("Flat field is wider than data array,"
                                 " %d vs %d;" % (len(self.data_ff), nelem))
            cosutil.printContinuation("flat will not be included in weight.")
            self.data_ff = None
            return

        flat = np.ones(nelem, dtype=np.float64)

        # dispersion direction
        shift1 = round(shift1)
        offset1 = self.origin_ff[1] - shift1 + x_offset
        offset1 = int(round(offset1))
        len_ff = len(self.data_ff)
        # Check for extending off either end of flat.
        if offset1 < 0 or offset1 + len_ff > nelem:
            cosutil.printWarning("Flat field extends beyond the data array;"
                                 " the flat will be truncated.")
            if offset1 < 0:
                trim = -offset1
                flat[:int(offset1+len_ff)] = self.data_ff[int(trim):]
            else:               # offset1 + len_ff > nelem
                trim = (offset1 + len_ff) - nelem
                flat[int(offset1):int(offset1+len_ff-trim)] = self.data_ff[:int(len_ff-trim)]
        else:
            flat[int(offset1):int(offset1+len_ff)] = self.data_ff
        self.data_ff = self.convolveFlat(flat, doppmag, doppzero, orbitper,
                                         expstart, expend)

        self.state_ff = "1-D, matching data"

    def convolveFlat(self, flat, doppmag, doppzero, orbitper,
                     expstart, expend):
        """Convolve the flat with the Doppler smoothing function.

        Based on timetag.convolveFlat.

        Parameters
        ----------
        flat: array_like
            Flat field, collapsed to 1-D, float.

        doppmag: float
            Magnitude of Doppler shift in pixels.

        doppzero: float
            Time (MJD) when the orbital Doppler shift is zero and
            increasing.

        orbitper: float
            Orbital period (seconds) of HST.

        expstart: float
            Exposure start time (MJD).

        expend: float
            Exposure end time (MJD).

        Returns
        -------
            The flat field convolved with the Doppler shift throughout
            the exposure (i.e. from expstart to expend).  If the magnitude
            of the Doppler shift was zero, or expend is not later than
            expstart, the input flat will be returned.
        """

        if doppmag <= 0.:
            return flat

        # Round doppmag up to the next integer; mag is a zero-point offset.
        mag = int(math.ceil(doppmag + 1.))
        lendopp = 2 * mag + 1

        # dopp will be the normalized Doppler smoothing function.
        dopp = np.zeros(lendopp, dtype=np.float64)

        # This spans the exposure, may be greater than actual exposure time.
        exptime = (expend - expstart) * SEC_PER_DAY
        rnd_exptime = int(round(exptime))
        if rnd_exptime <= 0:
            return flat

        # t is the time in seconds since doppzero, in one-second increments.
        t = np.arange(rnd_exptime, dtype=np.float64) + \
                   (expstart - doppzero) * SEC_PER_DAY

        # shift is in pixels (wavelengths increase toward larger pixel number).
        shift = -doppmag * np.sin(2. * np.pi * t / orbitper)

        # Construct the Doppler smoothing function.
        i_npts = rnd_exptime
        i_npts = max(i_npts, 1)
        increment = 1. / float(i_npts)
        for i in range(i_npts):                     # one-second increments
            ishift = int(round(shift[i])) + mag
            dopp[ishift] += increment

        # Do the convolution (in-place).
        nelem = len(flat)
        conv_flat = np.zeros(nelem, dtype=np.float64)
        conv_flat[0:mag] = 1.
        conv_flat[nelem-mag:] = 1.
        for k in range(lendopp):
            low = k
            high = k + nelem - 2 * mag
            conv_flat[mag:nelem-mag] += (dopp[lendopp-1-k] * flat[low:high])

        return conv_flat

class OutputSpectrum(object):
    """An output spectrum.

    The interpolation and averaging for one row are done by invoking this.
    Data for the current output row are computed and assigned to the data
    block in ofd.

    Parameters
    ----------
    ofd: pyfits HDUList object
        For the output file

    inspec: list of Spectrum objects
        For the input tables

    keywords: dictionary
        keywords and values from input headers

    segment: str
        Segment or stripe name for current row

    output_wl_range: float
        Wavelength at first pixel

    output_dispersion: float
        Angstroms per pixel to use for output

    Attributes
    ----------
    ofd
    inspec
    keywords
    segment
    output_wl_range
    output_dispersion
    """

    def __init__(self, ofd, inspec, keywords, segment,
                  output_wl_range, output_dispersion):
        """Constructor."""

        self.ofd = ofd
        self.inspec = inspec
        self.keywords = keywords
        self.segment = segment

        data = self.ofd[1].data

        foundit = False
        for row in range(len(data)):
            if data.field("segment")[row] == self.segment:
                foundit = True
                break
        assert foundit == True

        nelem = data.field("nelem")[row]

        # Allocate space for the sum of weights.
        sumweight = np.zeros(nelem, dtype=np.float64)

        # Assign wavelengths for the current row.
        data.field("wavelength")[row,:] = output_wl_range[0] + \
                output_dispersion * np.arange(nelem, dtype=np.float64)

        for sp in self.inspec:
            if self.segment == sp.segment:
                self.accumulateSums(sp, data[row], sumweight)

        self.normalizeSums(data[row], sumweight)

    def normalizeSums(self, data, sumweight):
        """Divide the sums by the sum of the weights.

        Parameters
        ----------
        data: fits record array
            The current row of the output file

        sumweight: float
            Sum of weights
        """

        verylowweight = np.where(data.field("dq_wgt") < 0.1)
        sumweight = np.where(sumweight == 0., 1., sumweight)
        nonzeroweight = np.where(data.field("dq_wgt") != 0.)
        zeroweight = np.where(data.field("dq_wgt") == 0.)
        data.field("flux")[:] /= sumweight
        data.field("gross")[:] /= sumweight
        data.field("background")[:] /= sumweight
        variance = data.field("variance_flat") + data.field("variance_counts") + data.field("variance_bkg")
        variance = np.where(variance < 0.5, 0.5, variance)
        error_frequentist_lower, error_frequentist_upper = cosutil.errFrequentist(variance)
        error_frequentist_lower[zeroweight] = 0.0
        error_frequentist_upper[zeroweight] = 0.0
        conversion = data.field("flux") / data.field("net")
#       Clean out NaNs from where flux and net are zero
        good = np.where(~np.isnan(conversion))
        bad = np.where(np.isnan(conversion))
        wavelength = data.field("wavelength")
        interpolated_values = np.interp(wavelength[bad], wavelength[good], conversion[good])
        conversion[bad] = interpolated_values
        data.setfield("error_lower", error_frequentist_lower * conversion)
        data.setfield("error", error_frequentist_upper * conversion)
        data.field("net")[:] /= sumweight
        data.field("flux")[:][verylowweight] = 0.
        data.field("gross")[:][verylowweight] = 0.
        data.field("gcounts")[:][verylowweight] = 0.
        data.field("background")[:][verylowweight] = 0.
        data.field("variance_flat")[:][verylowweight] = 0.
        data.field("variance_counts")[:][verylowweight] = 0.
        data.field("variance_bkg")[:][verylowweight] = 0.
        data.field("net")[:][verylowweight] = 0.
        data.field("error_lower")[:][verylowweight] = 0.
        data.field("error")[:][verylowweight] = 0.
        data.field("dq_wgt")[:][verylowweight] = 0.

    def accumulateSums(self, sp, data, sumweight):
        """Add input data to output, weighting by exposure time.

        The values in data and sumweight will be modified in-place.
        This version allows for fractional-pixel offset of the input arrays.

        Parameters
        ----------
        sp: Spectrum object
            Current input spectrum

        data: pyfits record array
            The current row of the output file

        sumweight: float
            Sum of weights
        """

        input_nelem = sp.nelem
        input_wavelength = sp.wavelength
        output_wavelength = data.field("wavelength")

        # Find the pixel numbers (floating point) in the input array
        # corresponding to the pixels in the output array, matching by
        # wavelength.  That is, for integer k:
        #   input_wavelength[ipixel[k]] = output_wavelength[k]
        # where the expression on the left hand side implies interpolation
        # rather than just indexing, since ipixel[k] will not in general
        # be an integer.

        ipixel = pixelsFromWl(input_wavelength, output_wavelength)

        # The output array will typically be longer than any of the input
        # arrays, so we must find the minimum and maximum indices in the
        # output array that map (via wavelength) to points within the current
        # input array.
        flag = np.where(np.logical_and (ipixel >= 0.,
                                        ipixel <= input_nelem-1.))
        min_k = flag[0][0]
        max_k = flag[0][-1]

        # ix is the array of pixel numbers (integer values but floating point
        # data type) in the input array that are actually within the input
        # array.  p and q are weight arrays for linear interpolation.
        ix = np.floor(ipixel[min_k:max_k])
        q = ipixel[min_k:max_k] - ix
        p = 1. - q
        i = ix.astype(np.int32)
        flux = data.field("flux")
        error = data.field("error")
        gross = data.field("gross")
        variance_flat = data.field("variance_flat")
        variance_counts = data.field("variance_counts")
        variance_bkg = data.field("variance_bkg")
        gcounts = data.field("gcounts")
        net = data.field("net")
        background = data.field("background")
        dq = data.field("dq")
        dq_wgt = data.field("dq_wgt")
        first = (data.field("exptime") == 0.)           # used for DQ

        weight1 = sp.dq_wgt[i] * sp.exptime
        weight2 = sp.dq_wgt[i+1] * sp.exptime
        # Also weight by the flat field.
        if sp.data_ff is not None:
            weight1 *= sp.data_ff[i]
            weight2 *= sp.data_ff[i+1]

        data.setfield("exptime", data.field("exptime") + sp.exptime)

        sumweight[min_k:max_k] += (p * weight1 + q * weight2)

        flux[min_k:max_k] += (sp.flux[i]   * p * weight1 +
                              sp.flux[i+1] * q * weight2)
        gross[min_k:max_k] += (sp.gross[i]   * p * weight1 +
                               sp.gross[i+1] * q * weight2)
        net[min_k:max_k] += (sp.net[i]   * p * weight1 +
                             sp.net[i+1] * q * weight2)
        background[min_k:max_k] += (sp.background[i]   * p * weight1 +
                                    sp.background[i+1] * q * weight2)
        temp_dq = sp.dq[i] | sp.dq[i+1]
        if first:
            dq[min_k:max_k] = temp_dq.copy()
        else:
            dq[min_k:max_k] &= temp_dq
        dq_wgt[min_k:max_k] += (sp.dq_wgt[i] * p + sp.dq_wgt[i+1] * q)
        gcounts[min_k:max_k] += (sp.gcounts[i]   * p * sp.dq_wgt[i] +
                                 sp.gcounts[i+1] * q * sp.dq_wgt[i+1])
        variance_flat[min_k:max_k] += (sp.variance_flat[i] * p * sp.dq_wgt[i] +
                                 sp.variance_flat[i+1] * q * sp.dq_wgt[i+1])
        variance_counts[min_k:max_k] += (sp.variance_counts[i] * p * sp.dq_wgt[i] +
                                 sp.variance_counts[i+1] * q * sp.dq_wgt[i+1])
        variance_bkg[min_k:max_k] += (sp.variance_bkg[i] * p * sp.dq_wgt[i] +
                                 sp.variance_bkg[i+1] * q * sp.dq_wgt[i+1])
