import math
import numpy as N
import pyfits
import cosutil
import wavecal
import ccos
from calcosparam import *       # parameter definitions

def processConcurrentWavecal (events, outflash,
                info, switches, reffiles, phdr, hdr):
    """Determine shifts from concurrent (tagflash) wavecal exposures.

    If tagflash mode was not used or the wavecorr switch is not
    "PERFORM", this function returns without doing anything.

    @param events: data block for a corrtag table
    @type events: record array

    @param outflash: name of output file for extracted wavecal spectra
    @type outflash: string

    @param info: dictionary of header keywords and values
    @type info: dictionary

    @param switches: dictionary of calibration switch keywords and values
    @type switches: dictionary

    @param reffiles: dictionary of reference file names
    @type reffiles: dictionary

    @param phdr: primary header of corrtag file
    @type phdr: pyfits header object

    @param hdr: events extension header of corrtag file
    @type hdr: pyfits header object
    """

    if not info["tagflash"]:
        return

    cosutil.printSwitch ("WAVECORR", switches)
    if switches["wavecorr"] != "PERFORM":
        return
    cosutil.printMsg ("Process tagflash wavecal")
    wavecal.printWavecalRef (reffiles)
    cosutil.printRef ("disptab", reffiles)

    cw = initWavecal (events, outflash, info, reffiles, phdr, hdr)

    cw.getStartStopTimes()
    if cw.numflash < 1:
        return

    cw.zeroKeywords()
    cw.outFlashSetup()          # create an HDU list for the outflash file

    cw.getWavecalParameters()
    cw.findShifts()
    cw.identifyOutliers()
    cw.applyCorrections()

    if switches["statflag"] == "PERFORM":
        cw.doStat()
    phdr["wavecorr"] = "COMPLETE"

    cw.writeOutFlash()

def initWavecal (events, outflash, info, reffiles, phdr, hdr):
    """Return a ConcurrentWavecal object, depending on detector.

    arguments:
    events      data block (recarray object) for an events table corrtag file
    outflash    name of output file for table of extracted wavecal spectra
    info        dictionary of keywords and values
    reffiles    dictionary of reference file names
    phdr        primary header of corrtag file
    hdr         events extension header
    """

    if info["detector"] == "FUV":
        cw = FUVConcurrentWavecal (events, outflash, info, reffiles, phdr, hdr)
    else:
        cw = NUVConcurrentWavecal (events, outflash, info, reffiles, phdr, hdr)

    return cw

class ConcurrentWavecal:
    """Process wavecals embedded in a science observation (tagflash).

    @ivar events: data block for a corrtag table
    @type events: record array

    @ivar outflash: name of output file for extracted wavecal spectra
    @type outflash: string

    @ivar info: dictionary of header keywords and values
    @type info: dictionary

    @ivar reffiles: dictionary of reference file names
    @type reffiles: dictionary

    @ivar phdr: primary header of corrtag file
    @type phdr: pyfits header object

    @ivar hdr: events extension header of corrtag file
    @type hdr: pyfits header object
    """

    def __init__ (self, events, outflash, info, reffiles, phdr, hdr):

        self.events = events
        self.outflash = outflash
        self.info = info
        self.reffiles = reffiles
        self.phdr = phdr
        self.hdr = hdr

        self.ofd = None                 # HDU list for outflash FITS file

        # lamp_on and lamp_off are in seconds, with the same zero point
        # as the TIME column in the events table
        self.lamp_on = []               # start times of wavecals (flashes)
        self.lamp_off = []              # stop times of wavecals
        self.lamp_median = []           # median times of wavecals
        self.numflash = 0               # number of embedded wavecals

        # for FUV segment_list is just one segment name;
        # for NUV this is all three stripe names ["NUVA", "NUVB", "NUVC"]
        self.segment_list = []          # segment names

        # the key is segment or stripe name, the value is a list, each
        # element of which is a two-element list of the slice limits
        # (in the cross-dispersion direction) within which the shifts
        # should be subtracted from the pixel coordinates
        self.regions = {}               # apply pshift only to these regions

        # each element is a dictionary; the key is segment or stripe name,
        # and the value is the shift in the dispersion direction
        self.pshift = []                # dictionaries of shift in disp dir

        # each element is the shift in the cross-dispersion direction
        # (one value for all stripes)
        self.shift2 = []                # shifts in cross-dispersion direction

        # the relevant row from the wavecal parameters table
        self.wcp_info = None            # matching row (just one) from wcp table

        # times of photon events, in seconds since EXPSTART
        self.time = events.field ("TIME")

        # These five columns are assigned by a subclass, depending on
        # the detector.  xi is in the dispersion direction, and eta is
        # in the cross-dispersion direction.
        self.xi = None
        self.eta = None
        self.dq = None                  # data quality flags; 0 is OK
        self.xi_corr = None             # corrected coords (XFULL or YFULL)
        self.eta_corr = None            # corrected coords (YFULL or XFULL)
        self.spectrum = None            # scratch space for extracted spectrum

    def outFlashSetup (self):
        """Create an HDU list for the outflash FITS table."""

        if not self.outflash:
            self.ofd = None
            return

        # Number of elements in a WAVELENGTH or GROSS array.
        rpt = str (len (self.spectrum))

        col = []
        col.append (pyfits.Column (name="SEGMENT", format="4A"))
        col.append (pyfits.Column (name="TIME", format="1D",
                    disp="F8.3", unit="s"))
        col.append (pyfits.Column (name="EXPTIME", format="1D",
                    disp="F8.3", unit="s"))
        col.append (pyfits.Column (name="LAMP_ON", format="1D",
                    disp="F8.3", unit="s"))
        col.append (pyfits.Column (name="LAMP_OFF", format="1D",
                    disp="F8.3", unit="s"))
        col.append (pyfits.Column (name="NELEM", format="1J", disp="I6"))
        col.append (pyfits.Column (name="WAVELENGTH",
                    format=rpt+"D", unit="angstrom"))
        col.append (pyfits.Column (name="GROSS",
                    format=rpt+"E", unit="count /s"))
        col.append (pyfits.Column (name="SHIFT_DISP",
                    format="1E", unit="pixel"))
        col.append (pyfits.Column (name="SHIFT_XDISP",
                    format="1E", unit="pixel"))
        col.append (pyfits.Column (name="SPEC_FOUND", format="1L"))
        cd = pyfits.ColDefs (col)

        nrows = self.numflash * len (self.segment_list)

        primary_hdu = pyfits.PrimaryHDU (header=self.phdr)
        self.ofd = pyfits.HDUList (primary_hdu)
        hdu = pyfits.new_table (cd, header=self.hdr, nrows=nrows)
        hdu.name = "TAGFLASH"
        self.deleteCoordinateKeywords (hdu)
        self.ofd.append (hdu)

        cosutil.updateFilename (self.ofd[0].header, self.outflash)
        self.ofd[0].header["wavecorr"] = "COMPLETE"

        # We know this value, so assign it now.
        self.ofd[1].data.field ("nelem")[:] = len (self.spectrum)

    def deleteCoordinateKeywords (self, hdu):
        """Delete keywords that are not relevant for extracted spectra.

        @param hdu: HDU for table of extracted wavecal spectra (modified)
        @type hdu: pyfits header/data unit object
        """

        ikey = ["TCTYP2", "TCTYP3", "TCRVL2", "TCRVL3", "TCRPX2", "TCRPX3",
                "TCDLT2", "TCDLT3", "TCUNI2", "TCUNI3",
                "TC2_2",  "TC2_3",  "TC3_2",  "TC3_3",
                "TALEN2", "TALEN3"]
        for keyword in ikey:
            if hdu.header.has_key (keyword):
                del hdu.header[keyword]

    def doStat (self):
        """Compute mean and max of the GROSS column."""

        if self.ofd is not None:
            cosutil.doTagFlashStat (self.ofd)

    def writeOutFlash (self):
        """Write the outflash HDU list to the output file."""

        if self.ofd is not None:
            self.ofd.writeto (self.outflash, output_verify="fix")
            self.ofd.close()

    def copyColumns (self):
        """Copy xi and eta to xi_corr and eta_corr."""

        self.xi_corr[:] = self.xi.copy()
        self.eta_corr[:] = self.eta.copy()

    def getStartStopTimes (self):
        """Get the times when the lamp was turned on or off."""

        # total number of tagflash wavecal exposures, according to keyword
        self.numflash = self.hdr.get ("NUMFLASH", default=0)

        t0 = self.time[0]
        t_last = self.time[-1]
        numflash = 0            # will be actual number of tagflash wavecals
        for n in range (self.numflash):
            key_on = "LMP_ON" + str (n+1)       # one indexed for FITS
            key_off = "LMPOFF" + str (n+1)
            starttime = self.hdr.get (key_on)
            endtime = self.hdr.get (key_off)
            if endtime < t0 or starttime > t_last:
                continue
            starttime = max (starttime, t0)
            endtime = min (endtime, t_last)
            self.lamp_on.append (starttime)
            self.lamp_off.append (endtime)
            numflash += 1
        self.numflash = numflash        # can be modified by findFlash

        # Determine the actual times from the data, and then set the keywords
        # and update the attributes.
        self.findFlash (delta_t=1.0, output=None, update=True)

    def getWavecalParameters (self):
        """Get the matching row from the wavecal parameters table."""

        wcp_info = cosutil.getTable (self.reffiles["wcptab"],
                        filter={"opt_elem": self.info["opt_elem"]},
                        exactly_one=1)
        self.wcp_info = wcp_info[0]

    def findShifts (self):
        """For each wavecal flash, find the shift in each axis."""

        xtractab = self.reffiles["xtractab"]
        lamptab = self.reffiles["lamptab"]
        # segment will be added to the filter within the loop.  Note that
        # the aperture is explicitly set to "WCA", because the aperture
        # keyword will give the aperture used for the science data.
        filter = {"opt_elem": self.info["opt_elem"],
                  "cenwave": self.info["cenwave"],
                  "aperture": "WCA"}

        # Find the offsets in both axes, for each wavecal exposure.
        row = 0                                 # incremented in inner loop
        cosutil.printMsg ("  segment    cross-disp      dispersion direction",
                VERBOSE)
        cosutil.printMsg ("            shift (locn)      shift  diagnostics",
                VERBOSE)
        cosutil.printMsg ("  -------   -------------     -----  -----------",
                VERBOSE)
        for n in range (self.numflash):
            (i0, i1) = ccos.range (self.time, self.lamp_on[n], self.lamp_off[n])

            # Find offset from nominal in cross-dispersion direction.
            (shift2, xd_shifts, xd_locn) = \
                wavecal.ttFindWavecalSpectrum (
                        self.xi[i0:i1], self.eta[i0:i1], self.dq[i0:i1],
                        self.info, "WCA", xtractab)
            if shift2 is None:
                shift2 = 0.
            self.shift2.append (shift2)

            # Extract wavecal spectra from events table, and determine offset
            # in dispersion direction.
            pshift = {}
            for segment in self.segment_list:
                filter["segment"] = segment
                xtract_info = cosutil.getTable (xtractab, filter, exactly_one=1)
                extr_height = xtract_info.field ("height")[0]
                slope       = xtract_info.field ("slope")[0]
                intercept   = xtract_info.field ("b_spec")[0]
                ccos.xy_extract (self.xi[i0:i1], self.eta[i0:i1],
                        self.dq[i0:i1], extr_height, slope, intercept+shift2,
                        self.spectrum)
                if xd_shifts[segment] is not None:
                    # find offset in dispersion direction
                    (pshift[segment], n50) = \
                            wavecal.ttFindWavecalShift (self.spectrum,
                                segment, self.info, lamptab, self.wcp_info)
                    spec_found = True
                    message = "%2d %4s %9.1f (%5.1f) %9.1f  " \
                        % (n+1, segment, xd_shifts[segment], xd_locn[segment],
                        pshift[segment]) + str (n50)
                    cosutil.printMsg (message, VERBOSE)
                else:
                    # ttFindWavecalSpectrum couldn't find the spectrum
                    pshift[segment] = 0.
                    spec_found = False
                    message = "%2d %4s not found (%5.1f)" \
                        % (n+1, segment, xd_locn[segment])
                    cosutil.printMsg (message, VERBOSE)
                # copy to outflash table data
                self.saveSpectrum (self.reffiles["disptab"], filter,
                            n, row, pshift[segment], shift2, spec_found)
                row += 1
            if self.info["detector"] == "NUV":
                cosutil.printMsg ("%2d      avg %5.1f" % (n+1, shift2), VERBOSE)
            self.pshift.append (pshift)

    def saveSpectrum (self, disptab, filter, n, row,
                      pshift, shift2, spec_found):
        """Copy the spectrum to the record array for the outflash table.

        @param disptab: name of the dispersion relation table
        @type disptab: string

        @param filter: for extracting the row from the disptab
        @type filter: dictionary

        @param n: index of current wavecal
        @type n: int

        @param row: row index (zero indexed) in output table
        @type row: int

        @param pshift: shift in dispersion direction
        @type pshift: float

        @param shift2: shift in cross-dispersion direction
        @type shift2: float

        @param spec_found: was the wavecal spectrum actually found?
        @type spec_found: boolean
        """

        if self.ofd is None:
            return

        t0 = self.lamp_on[n]
        t1 = self.lamp_off[n]

        pixel = N.arange (len (self.spectrum), dtype=N.float64)
        disp_info = cosutil.getTable (disptab, filter, exactly_one=1)
        ncoeff = disp_info.field ("nelem")[0]
        coeff = disp_info.field ("coeff")[0][0:ncoeff]
        pixel -= pshift         # correct the wavelengths for the shift
        wavelength = cosutil.evalDisp (pixel, coeff)

        self.ofd[1].data.field ("segment")[row] = filter["segment"]
        self.ofd[1].data.field ("time")[row] = self.lamp_median[n]
        self.ofd[1].data.field ("exptime")[row] = t1 - t0
        self.ofd[1].data.field ("lamp_on")[row] = self.lamp_on[n]
        self.ofd[1].data.field ("lamp_off")[row] = self.lamp_off[n]
        self.ofd[1].data.field ("wavelength")[row] = wavelength
        exptime = t1 - t0
        if exptime <= 0.:
            exptime = 1.
        self.ofd[1].data.field ("gross")[row] = self.spectrum / exptime
        self.ofd[1].data.field ("shift_disp")[row] = pshift
        self.ofd[1].data.field ("shift_xdisp")[row] = shift2
        self.ofd[1].data.field ("spec_found")[row] = spec_found

    def identifyOutliers (self):
        pass                        # xxx

    def applyCorrections (self):
        """Apply the pshift[a-c] and shift2 offsets."""

        nintervals = max (1, self.numflash - 1)

        for n in range (nintervals):
            (i0, i1) = self.getInterval (n)
            self.shift2Corr (n, i0, i1)
            self.pshiftCorr (n, i0, i1)

    def getInterval (self, n):
        """Get the slice for times between wavecal exposures n and n+1."""

        if n == 0:
            t0 = self.time[0]               # extrapolate to the beginning
        else:
            t0 = self.lamp_median[n]

        if self.numflash <= 2 or n == self.numflash - 2:
            # this is the last interval
            nevents = len (self.time)
            t1 = self.time[nevents-1]       # extrapolate to the end
        else:
            t1 = self.lamp_median[n+1]

        return ccos.range (self.time, t0, t1)

    def pshiftCorr (self, n, i0, i1):
        """Correct the pixel coordinates in the dispersion direction.

        @param n: apply pshift for the nth time interval between wavecals
        @type n: int

        @param i0: [i0:i1] is the slice of event numbers to be corrected
        @type i0: int

        @param i1: [i0:i1] is the slice of event numbers to be corrected
        @type i1: int
        """

        for segment in self.segment_list:

            # Restrict the correction to the applicable regions.
            shift_flags = N.zeros (i1 - i0, dtype=N.bool8)
            locn_list = self.regions[segment]
            for region in locn_list:
                shift_flags |= N.logical_and (
                               self.eta_corr[i0:i1] >= region[0],
                               self.eta_corr[i0:i1] < region[1])

            pshift_zero = self.pshift[n][segment]
            if self.numflash == 1:
                self.xi_corr[:] = N.where (shift_flags,
                        self.xi_corr - pshift_zero,
                        self.xi_corr)
            else:
                # Note that i0 & i1 do not necessarily correspond to t0 and t1.
                t0 = self.lamp_median[n]
                t1 = self.lamp_median[n+1]
                if t1 <= t0:
                    slope = 0.
                else:
                    slope = (self.pshift[n+1][segment] -
                             self.pshift[n][segment]) / (t1 - t0)
                    self.xi_corr[i0:i1] = N.where (shift_flags,
                        self.xi_corr[i0:i1] -
                            ((self.time[i0:i1] - t0) * slope + pshift_zero),
                        self.xi_corr[i0:i1])

    def findFlash (self, delta_t=1.0, output=None, update=True):
        """Find tagflash wavecals in science data.

        @param delta_t: time step for binning events into array of count rates
        @type delta_t: float

        @param output: if specified, write array of count rates to this file
            (for testing or debugging)
        @type output: string, or None

        @param update: if true, keywords in input file will be updated
        @type update: boolean
        """

        hdr = self.hdr

        detector = self.info["detector"]
        exptime = self.info["exptime"]

        xtractab = self.reffiles["xtractab"]

        filter = {"opt_elem": self.info["opt_elem"],
                  "cenwave": self.info["cenwave"],
                  "aperture": "WCA"}

        if detector == "FUV":
            filter["segment"] = self.info["segment"]
            xtract_info = cosutil.getTable (xtractab, filter, exactly_one=1)
            b_spec = xtract_info.field ("b_spec")[0]
            height = xtract_info.field ("height")[0]
            src_low  = int (b_spec - height)
            src_high = int (b_spec + height)
            eta = self.events.field ("ycorr")
        else:
            filter["segment"] = "NUVA"
            xtract_info = cosutil.getTable (xtractab, filter, exactly_one=1)
            b_spec = xtract_info.field ("b_spec")[0]
            height = xtract_info.field ("height")[0]
            src_low = 0
            src_high = int (b_spec + height)
            eta = self.events.field ("rawx")

        # Dummy values, so no background counts will be found.
        bkg1_low = -10
        bkg1_high = -20
        bkg2_low = -10
        bkg2_high = -20
        bkgsf = 0.

        time = self.time
        dq = N.zeros (len (time), dtype=N.int16)
        nbins = int (math.ceil ((time[-1] - time[0]) / delta_t))
        istart = N.zeros (nbins, dtype=N.int32)
        istop = N.zeros (nbins, dtype=N.int32)
        src_counts = N.zeros (nbins, dtype=N.int32)
        bkg_counts = N.zeros (nbins, dtype=N.int32)

        ccos.getstartstop (time, eta, dq, istart, istop, delta_t)
        ccos.getbkgcounts (eta, dq, istart, istop, bkg_counts, src_counts,
                    bkg1_low, bkg1_high, bkg2_low, bkg2_high,
                    src_low, src_high, bkgsf)
        del dq, bkg_counts

        # Convert to count rate.
        src_counts = src_counts.astype (N.float64) / delta_t

        if output is not None:
            ofd = open (output, "w")
            for val in src_counts:
                ofd.write ("%g\n" % val)
            ofd.close()

        (hist, step) = self.makeHistogram (src_counts)

        if hist is not None:
            cutoff = self.findCutoff (hist, step)
            self.findLampOn (src_counts, cutoff, time[0], delta_t)
            self.findLampMedian()

        if cosutil.checkVerbosity (VERBOSE):
            self.printInfo()

        if update:
            self.updateHeader (hdr)

    def makeHistogram (self, src_counts):
        """Make a histogram of src_counts.

        The function value is (hist, step), the histogram and step size.
        If the maximum value in src_counts is less than 20, the count
        rate is so low that it would be difficult to determine when the
        lamp turned on or off, and (None, None) will be returned to
        indicate this case.

        @param src_counts: count rate in each delta_t time interval in
            the wavecal region of the detector
        @type src_counts: array

        @return: the histogram (array) and step size (float)
        @rtype: tuple
        """

        maxval = N.maximum.reduce (src_counts)
        if maxval <= 20.:
            return (None, None)

        i_maxval = int (round (maxval))
        nbins = max (20, len (src_counts) // 100)
        step = maxval / float (nbins)
        hist = N.zeros (nbins, dtype=N.int32)
        for src in src_counts:
            i = int (src / step)
            if i < nbins:               # ignore max value
                hist[i] += 1
        cosutil.printMsg ("tagflash histogram = %s" % repr (hist),
                          VERY_VERBOSE)
        cosutil.printMsg ("step size for histogram = %g" % step, VERY_VERBOSE)

        return (hist, step)

    def findCutoff (self, hist, step):
        """Find the count rate above which the lamp is probably on.

        The histogram is likely to have two peaks, one close to zero that
        is due to dark counts and one close to the maximum count rate
        that is due to the wavecal lamp being on.  This function searches
        for the second one by looking for the maximum of the upper half
        (higher count rates) of the histogram.  Then the cutoff is set
        to half the count rate corresponding to that peak.

        @param hist: histogram of count rates src_counts.  hist[i] is the
            number of elements of src_counts with count rates between
            i*step and (i+1)*step.
        @type hist: array

        @param step: step size for histogram, i.e. change in count rate
            from hist[i] to hist[i+1]
        @type step: float

        @return: the cutoff count rate
        @rtype: float
        """

        n = len (hist)
        istart = n // 2
        index = istart
        maxhist = 0
        for i in range (n // 2, n):
            if hist[i] > maxhist:
                maxhist = hist[i]
                index = i

        cutoff = (index / 2.) * step
        cosutil.printMsg ("tagflash cutoff = %.2f counts/s" % cutoff,
                           VERY_VERBOSE)

        return cutoff

    def findLampOn (self, src_counts, cutoff, t0, delta_t):
        """Find the actual times when the lamps turned on and off.

        The nominal lamp turn-on and turn-off times were gotten from
        the EVENTS header and saved as self.lamp_on and self.lamp_off.
        This function gets the actual times, constrained to be within
        the time intervals specified in the header.  The attributes
        lamp_on and lamp_off are then updated.

        @param src_counts: count rate in each delta_t time interval in
            the wavecal region of the detector
        @type src_counts: array

        @param cutoff: a count rate (in src_counts array) greater than this
            indicates that the wavecal lamp was on
        @type cutoff: float

        @param t0: time of first photon event (first element of TIME column)
        @type t0: float

        @param delta_t: time step (seconds) for src_counts array
        @type delta_t: float
        """

        lamp_on = []
        lamp_off = []

        nbins = len (src_counts)
        for i in range (self.numflash):
            lamp_is_on = False
            (k_on, k_off) = self.getIndices (nbins, t0, delta_t,
                                self.lamp_on[i], self.lamp_off[i])
            if k_on is None:
                continue
            # search for the actual lamp turn-on time
            for k in range (k_on, k_off+1):
                if src_counts[k] > cutoff:
                    lamp_is_on = True
                    t = t0 + k * delta_t
                    lamp_on.append (t)
                    break
            if not lamp_is_on:
                continue
            # search for the actual lamp turn-off time
            for k in range (k_off, k_on-1, -1):
                if src_counts[k] > cutoff:
                    t = t0 + (k+1) * delta_t
                    t = min (t, self.time[-1])
                    lamp_off.append (t)
                    break

        if len (lamp_on) != len (lamp_off):
            raise RuntimeError, \
                "Internal error:  len (lamp_on) = %d, len (lamp_off) = %d" % \
                (len (lamp_on), len (lamp_off))

        self.numflash = len (lamp_on)           # update this value

        self.lamp_on = lamp_on
        self.lamp_off = lamp_off

    def findLampMedian (self):
        """Find the median time of each wavecal flash."""

        lamp_median = []
        for i in range (self.numflash):
            (i0, i1) = ccos.range (self.time, self.lamp_on[i], self.lamp_off[i])
            index = (i0 + i1) // 2
            lamp_median.append (self.time[index])

        self.lamp_median = lamp_median

    def getIndices (self, nbins, t0, delta_t, lamp_on_i, lamp_off_i):
        """Compute indices in src_counts corresponding to given times.

        @param nbins: number of elements in src_counts array
        @type nbins: int

        @param t0: time of first photon event
        @type t0: float

        @param delta_t: time step (seconds) for src_counts array
        @type delta_t: float

        @param lamp_on_i: one element of lamp_on array
        @type lamp_on_i: float

        @param lamp_off_i: one element of lamp_off array
        @type lamp_off_i: float
        """

        k_on = (lamp_on_i - t0) / delta_t
        k_off = (lamp_off_i - t0) / delta_t
        k_on = int (math.floor (k_on))
        k_off = int (math.ceil (k_off))
        if k_on >= nbins or k_off < 0:
            return (None, None)
        k_on = max (k_on, 0)
        k_off = min (k_off, nbins-1)

        return (k_on, k_off)

    def printInfo (self):
        """Print time info for each wavecal flash."""

        if self.numflash < 1:
            return

        cosutil.printMsg ("lamp on, off, duration, median time:")
        for i in range (self.numflash):
            cosutil.printMsg ("%d:  %.1f  %.1f  %.1f  %.1f" %
                              (i+1, self.lamp_on[i], self.lamp_off[i],
                              self.lamp_off[i] - self.lamp_on[i],
                              self.lamp_median[i]))

    def updateHeader (self, hdr):
        """Assign extension header keywords with updated info."""

        hdr.update ("NUMFLASH", self.numflash)

        for i in range (self.numflash):
            keyword = "LMP_ON%d" % (i+1)
            hdr.update (keyword, self.lamp_on[i])
            keyword = "LMPOFF%d" % (i+1)
            hdr.update (keyword, self.lamp_off[i])
            keyword = "LMPDUR%d" % (i+1)
            hdr.update (keyword, self.lamp_off[i] - self.lamp_on[i])
            keyword = "LMPMED%d" % (i+1)
            hdr.update (keyword, self.lamp_median[i])

class FUVConcurrentWavecal (ConcurrentWavecal):

    def __init__ (self, events, outflash, info, reffiles, phdr, hdr):

        ConcurrentWavecal.__init__ (self,
                        events, outflash, info, reffiles, phdr, hdr)
        self.xi  = events.field ("XDOPP")
        self.eta = events.field ("YCORR")
        self.dq  = events.field ("DQ")
        self.xi_corr  = events.field ("XFULL")
        self.eta_corr = events.field ("YFULL")
        self.spectrum = N.zeros (FUV_X, dtype=N.float64)
        self.segment_list = [info["segment"]]

        # Copy xi and eta to the columns for corrected values.
        self.copyColumns()

        # The pshift offset should be applied only within this region.
        self.regions[info["segment"]] = \
                [cosutil.activeArea (info["segment"], reffiles["brftab"])]

    def zeroKeywords (self):

        self.hdr.update ("PSHIFTA", 0.)
        self.hdr.update ("PSHIFTB", 0.)
        self.hdr.update ("SHIFT2A", 0.)
        self.hdr.update ("SHIFT2B", 0.)

    def shift2Corr (self, n, i0, i1):
        """Correct the pixel coordinates in the cross-dispersion direction.

        The difference between this version and the one for NUV is that this
        one limits the shift to the active area.

        @param n: apply shift2 for the nth time interval between wavecals
        @type n: int

        @param i0: [i0:i1] is the slice of event numbers to be corrected
        @type i0: int

        @param i1: [i0:i1] is the slice of event numbers to be corrected
        @type i1: int
        """

        # Restrict the correction to the applicable region.  Note that the
        # limits of the region (the active area) are not adjusted by shift2.
        shift_flags = N.zeros (i1 - i0, dtype=N.bool8)
        region = self.regions[self.segment_list[0]][0]
        shift_flags |= N.logical_and (
                       self.eta_corr[i0:i1] >= region[0],
                       self.eta_corr[i0:i1] < region[1])

        shift2_zero = self.shift2[n]
        if self.numflash == 1:
            self.eta_corr[:] = N.where (shift_flags,
                                   self.eta_corr - shift2_zero,
                                   self.eta_corr)
        else:
            # Note that i0 & i1 do not necessarily correspond to t0 and t1,
            # because we can extrapolate to the beginning or end of the array.
            t0 = self.lamp_median[n]
            t1 = self.lamp_median[n+1]
            if t1 <= t0:
                slope = 0.
            else:
                slope = (self.shift2[n+1] - self.shift2[n]) / (t1 - t0)
            self.eta_corr[i0:i1] = N.where (shift_flags,
                        self.eta_corr[i0:i1] -
                            ((self.time[i0:i1] - t0) * slope + shift2_zero),
                        self.eta_corr[i0:i1])

class NUVConcurrentWavecal (ConcurrentWavecal):

    def __init__ (self, events, outflash, info, reffiles, phdr, hdr):

        ConcurrentWavecal.__init__ (self,
                        events, outflash, info, reffiles, phdr, hdr)
        self.xi  = events.field ("YDOPP")
        self.eta = events.field ("RAWX")
        self.dq  = events.field ("DQ")
        self.xi_corr  = events.field ("YFULL")
        self.eta_corr = events.field ("XFULL")
        self.spectrum = N.zeros (NUV_Y, dtype=N.float64)
        self.segment_list = ["NUVA", "NUVB", "NUVC"]

        # Copy xi and eta to the columns for corrected values.
        self.copyColumns()

        # The pshift offset should be applied only within this region.
        self.regions = self.setRegions()

    def zeroKeywords (self):

        self.hdr.update ("PSHIFTA", 0.)
        self.hdr.update ("PSHIFTB", 0.)
        self.hdr.update ("PSHIFTC", 0.)
        self.hdr.update ("SHIFT2A", 0.)
        self.hdr.update ("SHIFT2B", 0.)
        self.hdr.update ("SHIFT2C", 0.)

    def setRegions (self):
        """Determine the regions over which pshift should be applied.

        The function value is a dictionary with nominally three entries.
        Segment name is the key, and each value is a list of the two
        intervals (one for PSA, one for WCA) over which the shift in the
        dispersion direction (pshift) should be applied.  The limits of
        the intervals are the midpoints between b_spec values from the
        xtractab.  Note that shift2 has not been subtracted from these
        intervals, and it should not be, because the cross-dispersion
        positions will be corrected for shift2 before correcting the
        positions in the dispersion direction.
        """

        # segment and aperture will be added to the filter in the loop.
        filter = {"opt_elem": self.info["opt_elem"],
                  "cenwave": self.info["cenwave"]}

        # locations will be a list of tuples, each containing the
        # segment name and nominal location.
        locations = []
        for segment in self.segment_list:

            filter["segment"] = segment

            filter["aperture"] = "WCA"
            xtract_info = cosutil.getTable (self.reffiles["xtractab"],
                                filter, exactly_one=1)
            b_spec = xtract_info.field ("b_spec")[0]
            locations.append ((segment, b_spec))

            filter["aperture"] = "PSA"
            xtract_info = cosutil.getTable (self.reffiles["xtractab"],
                                filter, exactly_one=1)
            b_spec = xtract_info.field ("b_spec")[0]
            locations.append ((segment, b_spec))

        locations.sort (self.r_cmp)     # sort on b_spec, regardless of segment
        len_locn = len (locations)

        # intervals will be the same length as locations.  Each interval
        # [first,last] is the slice over which pshift[a-c] should be
        # applied.  There should be six such intervals, one for each
        # stripe, and for apertures PSA and WCA.
        intervals = []
        first = 0                       # left edge of the detector
        for i in range (len_locn):
            segment = locations[i][0]
            locn = locations[i][1]
            if i == len_locn - 1:
                last = NUV_X            # right edge of the detector
            else:
                next_locn = locations[i+1][1]
                # midpoint between adjacent b_spec values
                last = int (round ((locn + next_locn) / 2.))
            intervals.append ([segment, [first,last]])
            first = last

        regions = {}
        for segment in self.segment_list:
            locn_list = []
            for i in range (len_locn):
                if segment == intervals[i][0]:
                    locn_list.append (intervals[i][1])
            regions[segment] = locn_list

        return regions

    def r_cmp (self, x, y):
        """Comparison function for sorting locations in setRegions().

        The comparison is based entirely on the second element, b_spec.
        """
        if x[1] < y[1]:
            return -1
        elif x[1] > y[1]:
            return 1
        else:
            return 0

    def shift2Corr (self, n, i0, i1):
        """Correct the pixel coordinates in the cross-dispersion direction.

        @param n: apply shift2 for the nth time interval between wavecals
        @type n: int

        @param i0: [i0:i1] is the slice of event numbers to be corrected
        @type i0: int

        @param i1: [i0:i1] is the slice of event numbers to be corrected
        @type i1: int
        """

        shift2_zero = self.shift2[n]
        if self.numflash == 1:
            self.eta_corr[:] -= shift2_zero
        else:
            # Note that i0 & i1 do not necessarily correspond to t0 and t1,
            # because we can extrapolate to the beginning or end of the array.
            t0 = self.lamp_median[n]
            t1 = self.lamp_median[n+1]
            if t1 <= t0:
                slope = 0.
            else:
                slope = (self.shift2[n+1] - self.shift2[n]) / (t1 - t0)
            self.eta_corr[i0:i1] -= \
                        ((self.time[i0:i1] - t0) * slope + shift2_zero)
