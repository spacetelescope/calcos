#! /usr/bin/env python

import math
import numpy as N
import pyfits
import cosutil
from calcosparam import *       # parameter definitions

def fpAvgSpec (input, output):
    """Average 1-D extracted FP-POS spectra.

    arguments:
        input         a list of one or more input x1d file names
        output        name of a file for the averaged spectra

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
    """

    nfiles = len (input)

    assert nfiles >= 1

    cosutil.printIntro ("Average 1-D spectra")
    names = [("Input", repr (input)), ("Output", output)]
    cosutil.printFilenames (names)

    if nfiles == 1:
        cosutil.copyFile (input[0], output)
    else:
        outspec = OutputX1D (input, output)

class OutputX1D (object):

    def __init__ (self, input, output):
        """Average 1-D FP-POS spectra.

        The attributes are:
            input            list of input file names
            output           output file name
            keywords         dictionary of relevant keywords and values,
                             e.g. detector
            inspec           list of Spectrum objects
            smallest_wl      dictionary of (segment: wavelength[0])
            segments         list of segment names found in input x1d tables
            ofd              pyfits object for output file
            nrows            number of rows to be written to the output table
            output_nelem     number of elements to use when allocating output
                             arrays
        """

        self.input = input
        self.output = output
        self.keywords = {}
        self.inspec = []
        self.smallest_wl = {}
        self.segments = []
        self.ofd = None
        self.nrows = 0
        self.output_nelem = 0

        # Create a list of Spectrum objects, and get info from headers.
        self.getInputInfo()

        # Check that the data in each Spectrum is comparable to the others.
        self.compareX1d()

        # Compute output pshift and length of arrays.
        self.computeOutputInfo()

        # Create ofd, the output pyfits object.
        self.createOutput()

        # Fill in the data in the output table.
        for segment in self.segments:           # for each output row ...
            osp = initOutputSpectrum (self.ofd, self.inspec, self.keywords,
                        self.smallest_wl[segment], segment)
        if cosutil.isProduct (self.output):
            asn_mtyp = self.ofd[1].header.get ("asn_mtyp", "missing")
            asn_mtyp = cosutil.modifyAsnMtyp (asn_mtyp)
            if asn_mtyp != "missing":
                self.ofd[1].header["asn_mtyp"] = asn_mtyp
        self.ofd.writeto (self.output)

    def getInputInfo (self):
        """Get info and data from input files.

        This routine creates Spectrum objects (one for each row of each input
        table) and appends them to the inspec list, gets keywords from the
        input headers, and determines the number of rows (nrows) that the
        output table should have.
        """

        first = True            # true for first input file
        sum_globrate = 0.       # incremented for each row in each file
        sum_exptime = 0.        # exptime is the weight for globrate
        first_wavelengths = {}  # dict of (segment: list of wavelength[0])
        for input in self.input:
            ifd = pyfits.open (input, mode="readonly")
            phdr = ifd[0].header
            hdr = ifd[1].header
            # Get keyword values.
            if first:
                detector = phdr["detector"]
                disptab = cosutil.expandFileName (phdr["disptab"])
                opt_elem = phdr["opt_elem"]
                cenwave = phdr["cenwave"]
                aperture = cosutil.getApertureKeyword (phdr, truncate=1)
                statflag = phdr.get ("statflag", False)
                sum_plantime = hdr["plantime"]
                sum_exptime_kwd = hdr["exptime"]
                expstart = hdr["expstart"]
                expend = hdr["expend"]
                first = False
            else:
                sum_plantime += hdr["plantime"]
                # sum of header EXPTIME keywords, not column values
                sum_exptime_kwd += hdr["exptime"]
                expstart = min (expstart, hdr["expstart"])
                expend = max (expend, hdr["expend"])
            fpoffset = phdr["fpoffset"]
            # segment will be added to filter in the loop over rows
            filter = {"opt_elem": opt_elem,
                      "cenwave": cenwave,
                      "aperture": aperture}
            if cosutil.findColumn (disptab, "fpoffset"):
                filter["fpoffset"] = fpoffset
            if ifd[1].data is not None:
                nrows = len (ifd[1].data)
                # for each row in the current input table
                for row in range (nrows):
                    sp = Spectrum (ifd, row, fpoffset)
                    segment = sp.segment
                    if segment not in self.segments:
                        self.segments.append (segment)
                    filter["segment"] = segment
                    disp_info = cosutil.getTable (disptab, filter,
                                exactly_one=True)
                    ncoeff = disp_info.field ("nelem")[0]
                    coeff = disp_info.field ("coeff")[0][0:ncoeff]
                    if cosutil.findColumn (disp_info, "delta"):
                        delta = disp_info.field ("delta")[0]
                    else:
                        delta = 0.
                    sp.setCoeff (coeff, delta)
                    self.inspec.append (sp)
                    if segment in first_wavelengths.keys():
                        first_wavelengths[segment].append (sp.wavelength[0])
                    else:
                        first_wavelengths[segment] = [sp.wavelength[0]]
                    sum_globrate += (hdr["globrate"] * sp.exptime)
                    sum_exptime += sp.exptime

            ifd.close()

        # For each segment/stripe, compute the pixel offset of each input
        # spectrum.  Save this value in Spectrum attribute pshift, later
        # (in OutputSpectrum) called input_pshift.
        for sp in self.inspec:
            segment = sp.segment
            self.smallest_wl[segment] = min (first_wavelengths[segment])
            sp.computePshift (self.smallest_wl[segment])

        if sum_exptime > 0.:
            globrate = sum_globrate / sum_exptime
        else:
            globrate = 0.

        # number of rows to be written to the output table
        self.nrows = len (self.segments)

        self.keywords = {
             "detector": detector,
             "disptab":  disptab,
             "opt_elem": opt_elem,
             "cenwave":  cenwave,
             "aperture": aperture,
             "exptime":  sum_exptime_kwd,
             "expstart": expstart,
             "expend":   expend,
             "expstrtj": expstart + MJD_TO_JD,
             "expendj":  expend + MJD_TO_JD,
             "plantime": sum_plantime,
             "globrate": globrate}

    def compareX1d (self):
        """Check that the rows of two x1d tables contain comparable info.

        Currently, the only check is on the array sizes.
        """

        for sp in self.inspec:
            if sp.nelem != self.inspec[0].nelem:
                raise RuntimeError, "x1d tables have different array sizes."

    def computeOutputInfo (self):
        """Compute length of output arrays.

        This routine assigns a value to the attribute output_nelem.
        """

        if len (self.inspec) < 1:
            self.output_nelem = 0
        else:
            max_pshift = self.inspec[0].pshift
            # locations of first and last pixel, relative to nominal
            min_x = -self.inspec[0].pshift
            max_x = self.inspec[0].nelem - self.inspec[0].pshift - 1.
            for sp in self.inspec:
                max_pshift = max (max_pshift, sp.pshift)
                min_x = min (min_x, -sp.pshift)
                max_x = max (max_x, sp.nelem - sp.pshift - 1.)
            # add almost one rather than exactly one, to allow for error
            # in floating-point computation
            self.output_nelem = int (math.ceil (max_x - min_x + 0.99999))
            if self.keywords["detector"] == "NUV":
                # increase the output array size by one to allow for roundoff
                # error during interpolation (not needed for FUV)
                self.output_nelem += 1

    def createOutput (self):
        """Create pyfits object for output file."""

        # Get header info from the input.
        ifd = pyfits.open (self.input[0], mode="readonly")

        primary_hdu = pyfits.PrimaryHDU (header=ifd[0].header)
        primary_hdu.header.update ("RPTCORR", "COMPLETE")
        cosutil.updateFilename (primary_hdu.header, self.output)
        ofd = pyfits.HDUList (primary_hdu)

        rpt = str (self.output_nelem)           # used for column definitions
        if self.output_nelem < 1:
            tform = ifd[1].header.get ("tform4", "1D")
            i = tform.find ("D")
            rpt = tform[:i]

        # Define output columns.
        col = []
        col.append (pyfits.Column (name="SEGMENT", format="4A"))
        col.append (pyfits.Column (name="EXPTIME", format="1D"))
        col.append (pyfits.Column (name="NELEM", format="1J"))
        col.append (pyfits.Column (name="WAVELENGTH", format=rpt+"D"))
        col.append (pyfits.Column (name="FLUX", format=rpt+"E"))
        col.append (pyfits.Column (name="ERROR", format=rpt+"E"))
        col.append (pyfits.Column (name="GROSS", format=rpt+"E"))
        col.append (pyfits.Column (name="NET", format=rpt+"E"))
        col.append (pyfits.Column (name="BACKGROUND", format=rpt+"E"))
        col.append (pyfits.Column (name="DQ_WGT", format=rpt+"E"))
        col.append (pyfits.Column (name="MAXDQ", format=rpt+"I"))
        col.append (pyfits.Column (name="AVGDQ", format=rpt+"I"))
        cd = pyfits.ColDefs (col)

        hdu = pyfits.new_table (cd, header=ifd[1].header, nrows=self.nrows)
        hdu.header.update ("exptime", self.keywords["exptime"])
        hdu.header.update ("expstart", self.keywords["expstart"])
        hdu.header.update ("expend", self.keywords["expend"])
        hdu.header.update ("expstrtj", self.keywords["expstrtj"])
        hdu.header.update ("expendj", self.keywords["expendj"])
        hdu.header.update ("plantime", self.keywords["plantime"])
        hdu.header.update ("globrate", self.keywords["globrate"])
        if hdu.header.has_key ("pshifta"):
            hdu.header.update ("pshifta", 0.)
        if hdu.header.has_key ("pshiftb"):
            hdu.header.update ("pshiftb", 0.)
        if hdu.header.has_key ("pshiftc"):
            hdu.header.update ("pshiftc", 0.)

        ofd.append (hdu)
        self.fpInitData (ofd)           # initialize data in output hdu

        ifd.close()

        self.ofd = ofd

    def fpInitData (self, ofd):
        """Initialize the output data block.

        Two scalar columns, SEGMENT and NELEM, will be set to their actual
        values.  EXPTIME and the array columns will be initialized to zero.
        """

        ofd[1].data.field ("segment")[:] = self.segments
        ofd[1].data.field ("nelem")[:] = self.output_nelem

        ofd[1].data.field ("exptime")[:] = 0.

        ofd[1].data.field ("wavelength")[:] = 0.
        ofd[1].data.field ("flux")[:] = 0.
        ofd[1].data.field ("error")[:] = 0.
        ofd[1].data.field ("gross")[:] = 0.
        ofd[1].data.field ("net")[:] = 0.
        ofd[1].data.field ("background")[:] = 0.
        ofd[1].data.field ("dq_wgt")[:] = 0.
        ofd[1].data.field ("maxdq")[:] = 0
        ofd[1].data.field ("avgdq")[:] = 0

class Spectrum (object):

    def __init__ (self, ifd, row=0, fpoffset=0):
        """This is one row of an input x1d table.

        The attributes are:
            exptime          exposure time (seconds) for this input spectrum
            segment          segment or stripe name for the current row
            nelem            number of elements in the arrays
            wavelength       array of wavelengths for the current row
            flux             array of flux values
            error            array of error estimates for the flux
            gross            array of gross values
            net              array of net values
            background       array of background values
            maxdq            array of maximum dq values
            avgdq            array of average dq values
            fpoffset         OSM offset in motor steps from nominal
            coeff            coefficients of the dispersion relation
            delta            pixel offset for the dispersion relation
            pshift           pixel shift computed from wavelength[0] with
                                 respect to smallest_wl
        """

        self.segment = ifd[1].data.field ("segment")[row]
        self.exptime = ifd[1].data.field ("exptime")[row]
        self.nelem = ifd[1].data.field ("nelem")[row]
        self.wavelength = ifd[1].data.field ("wavelength")[row]
        self.flux = ifd[1].data.field ("flux")[row]
        self.error = ifd[1].data.field ("error")[row]
        self.gross = ifd[1].data.field ("gross")[row]
        self.net = ifd[1].data.field ("net")[row]
        self.background = ifd[1].data.field ("background")[row]
        self.dq_wgt = ifd[1].data.field ("dq_wgt")[row]
        self.maxdq = ifd[1].data.field ("maxdq")[row]
        self.avgdq = ifd[1].data.field ("avgdq")[row]
        self.fpoffset = fpoffset
        self.coeff = None       # assigned later by setCoeff
        self.delta = None       # assigned later by setCoeff
        self.pshift = 0.        # assigned later by computePshift

    def setCoeff (self, coeff, delta):
        """Assign the coefficient array to an attribute.

        This is not done in init because that would likely require duplicate
        calls to getTable, since there can be multiple table rows (in different
        input files) with the same row-selection criteria, in particular the
        same segment or stripe.
        """

        self.coeff = coeff
        self.delta = delta

    def computePshift (self, smallest_wl):
        """Get the pixel shift from the wavelength at the first pixel.

        Note that setCoeff must have been called first in order to assign
        values to the coeff list.

        @param smallest_wl: the minimum wavelength[0] for all input spectra
        @type smallest_wl: float
        """

        x = 0.

        slope = cosutil.evalDerivDisp (x, self.coeff, self.delta)
        x = (self.wavelength[0] - smallest_wl) / slope

        self.pshift = -x

def initOutputSpectrum (ofd, inspec, keywords, smallest_wl, segment):
    """Construct an OutputSpectrum object, depending on the detector.

    arguments:
        ofd              pyfits object for output file
        inspec           list of Spectrum objects
        keywords         dictionary of keywords and values, e.g. detector
        smallest_wl      wavelength at the first pixel
        segment          segment or stripe name for current row
    """

    if keywords["detector"] == "FUV":
        osp = FUV_OutputSpectrum (ofd, inspec, keywords, smallest_wl, segment)
    else:
        osp = NUV_OutputSpectrum (ofd, inspec, keywords, smallest_wl, segment)

    return osp

class OutputSpectrum (object):

    def __init__ (self, ofd, inspec, keywords, smallest_wl, segment):
        """This is only invoked by a subclass that depends on detector.

        The attributes are:
            ofd              pyfits object for output file
            inspec           list of Spectrum objects for the input tables
            keywords         dictionary of keywords and values from input
                             headers
            smallest_wl      wavelength at the first pixel
            segment          segment or stripe name for current row

        The interpolation and averaging for one row are done by invoking this.
        Data for the current output row are computed and assigned to the data
        block in ofd.
        """

        self.ofd = ofd
        self.inspec = inspec
        self.keywords = keywords
        self.smallest_wl = smallest_wl
        self.segment = segment

        foundit = False
        for row in range (len (self.ofd[1].data)):
            if self.ofd[1].data.field ("segment")[row] == self.segment:
                foundit = True
                break
        assert foundit == True

        # Allocate space for the sum of weights and for the sum of the
        # data qualify flags.  We can't accumulate sums for the latter
        # in-place in the ofd data block because avgdq is an int16, and
        # that could overflow due to weighting by the exposure time.
        sumweight = N.zeros (self.ofd[1].data.field ("nelem")[row],
                        dtype=N.float64)
        sumdq = N.zeros (self.ofd[1].data.field ("nelem")[row],
                        dtype=N.float64)

        min_fpoffset = 999      # initial (invalid) value
        coeff = None
        delta = None
        for sp in self.inspec:
            if self.segment == sp.segment:
                self.accumulateSums (sp, self.ofd[1].data[row],
                        sumdq, sumweight)
                # look for fpoffset = 0 (or closest to 0)
                if abs (sp.fpoffset) < min_fpoffset:
                    min_fpoffset = abs (sp.fpoffset)
                    coeff = sp.coeff
                    delta = sp.delta

        self.normalizeSums (self.ofd[1].data[row], sumdq, sumweight,
                            coeff, delta)

    def normalizeSums (self, data, sumdq, sumweight, coeff, delta):
        """Divide the sums by the sum of the weights.

        arguments:
            data           record array for the current row of the output file
            sumdq          weighted sum of data quality flags
            sumweight      sum of weights
            coeff          coefficients of the dispersion relation
            delta          offset to the pixel coordinates
        """

        maxdq = data.field ("maxdq")
        maxdq = N.where (sumweight == 0., DQ_OUT_OF_BOUNDS, maxdq)

        sumweight = N.where (sumweight == 0., 1., sumweight)

        nelem = len (sumweight)
        x = N.arange (nelem, dtype=N.float64)   # pixel numbers
        # Add an offset so the wavelength at x[0] will be smallest_wl.
        x += cosutil.evalInvDisp (self.smallest_wl, coeff, delta)
        data.field ("wavelength")[:] = cosutil.evalDisp (x, coeff, delta)
        del x

        data.field ("flux")[:] /= sumweight
        data.field ("gross")[:] /= sumweight
        data.field ("net")[:] /= sumweight
        data.field ("background")[:] /= sumweight
        data.field ("avgdq")[:] = N.around (sumdq / sumweight)
        data.field ("maxdq")[:] = maxdq
        data.field ("error")[:] = \
                        N.sqrt (data.field ("error")) / sumweight

class FUV_OutputSpectrum (OutputSpectrum):
    """This is one row of an FUV output x1d table.

    The difference between this and NUV_OutputSpectrum is that the
    accumulateSums method in the latter allows for fractional pixel
    overlap between input spectra, while accumulateSums in this class
    rounds to the nearest pixel.
    """

    def __init__ (self, ofd, inspec, keywords, smallest_wl, segment):

        OutputSpectrum.__init__ (self,
                ofd, inspec, keywords, smallest_wl, segment)

    def accumulateSums (self, sp, data, sumdq, sumweight):
        """Add input data to output, weighting by exposure time.

        The values in data, sumdq, and sumweight will be modified in-place.
        Most of the sums will be accumulated in data; the exception is the
        sum for the average data quality, which will be accumulated in sumdq
        to prevent short-integer overflow.

        arguments:
            sp             current input Spectrum object
            data           record array for the current row of the output file
            sumdq          weighted sum of data quality flags
            sumweight      sum of weights
        """

        input_nelem = sp.nelem
        input_pshift = int (round (sp.pshift))

        flux = data.field ("flux")
        error = data.field ("error")
        gross = data.field ("gross")
        net = data.field ("net")
        background = data.field ("background")
        dq_wgt = data.field ("dq_wgt")
        maxdq = data.field ("maxdq")

        weight = sp.dq_wgt * sp.exptime

        i = -input_pshift
        i = int (round (i))
        j = i + input_nelem

        data.setfield ("exptime", data.field ("exptime") + sp.exptime)
        sumweight[i:j] += weight
        flux[i:j] += (sp.flux * weight)
        gross[i:j] += (sp.gross * weight)
        net[i:j] += (sp.net * weight)
        background[i:j] += (sp.background * weight)
        dq_wgt[i:j] += sp.dq_wgt
        sumdq[i:j] += (sp.avgdq * weight)
        maxdq[i:j] = N.maximum (sp.maxdq, maxdq[i:j])
        error[i:j] += (sp.error * weight)**2

class NUV_OutputSpectrum (OutputSpectrum):
    """This is one row of an NUV output x1d table."""

    def __init__ (self, ofd, inspec, keywords, smallest_wl, segment):

        OutputSpectrum.__init__ (self,
                ofd, inspec, keywords, smallest_wl, segment)

    def accumulateSums (self, sp, data, sumdq, sumweight):
        """Add input data to output, weighting by exposure time.

        The values in data, sumdq, and sumweight will be modified in-place.
        This version allows for fractional-pixel offset of the input arrays.

        arguments:
            sp             current input Spectrum object
            data           record array for the current row of the output file
            sumdq          weighted sum of data quality flags
            sumweight      sum of weights
        """

        input_nelem = sp.nelem
        input_pshift = sp.pshift

        flux = data.field ("flux")
        error = data.field ("error")
        gross = data.field ("gross")
        net = data.field ("net")
        background = data.field ("background")
        dq_wgt = data.field ("dq_wgt")
        maxdq = data.field ("maxdq")

        weight = sp.dq_wgt * sp.exptime

        ix = -input_pshift
        i = int (math.floor (ix))
        j = i + input_nelem
        q = ix - i
        p = 1. - q

        data.setfield ("exptime", data.field ("exptime") + sp.exptime)
        sumweight[i:j] += (weight * p)
        flux[i:j] += (sp.flux * p * weight)
        gross[i:j] += (sp.gross * p * weight)
        net[i:j] += (sp.net * p * weight)
        background[i:j] += (sp.background * p * weight)
        dq_wgt[i:j] += (sp.dq_wgt * p)
        sumdq[i:j] += (sp.avgdq * p * weight)
        maxdq[i:j] = N.maximum (sp.maxdq, maxdq[i:j])
        error[i:j] += (sp.error * p * weight)**2

        if q > 0.:
            i += 1
            j += 1
            sumweight[i:j] += (weight * q)
            flux[i:j] += (sp.flux * q * weight)
            gross[i:j] += (sp.gross * q * weight)
            net[i:j] += (sp.net * q * weight)
            background[i:j] += (sp.background * q * weight)
            dq_wgt[i:j] += (sp.dq_wgt * q)
            sumdq[i:j] += (sp.avgdq * q * weight)
            maxdq[i:j] = N.maximum (sp.maxdq, maxdq[i:j])
            error[i:j] += (sp.error * q * weight)**2
