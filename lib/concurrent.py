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

    @return: three objects:  the average offset in the X direction, the
        average offset in the Y direction, and an array of the shifts in
        the dispersion direction at one-second intervals; these values
        will be (0., 0., None) if wavecorr is not perform or the input
        data are not tagflash or there are no flashes.
    @rtype: tuple
    """

    if not info["tagflash"]:
        return (0., 0., None)

    cosutil.printSwitch ("WAVECORR", switches)
    if switches["wavecorr"] != "PERFORM":
        cw = initWavecal (events, outflash, info, reffiles, phdr, hdr)
        return (0., 0., None)
    cosutil.printMsg ("Process tagflash wavecal")
    wavecal.printWavecalRef (reffiles)
    cosutil.printRef ("disptab", reffiles)

    cw = initWavecal (events, outflash, info, reffiles, phdr, hdr)

    cw.getStartStopTimes()
    if cw.numflash < 1:
        # write an empty lampflash table
        cw.outFlashSetup()
        cw.writeOutFlash()
        return (0., 0., None)

    cw.outFlashSetup()          # create an HDU list for the outflash file

    cw.getWavecalParameters()
    cw.findShifts()
    cw.applyCorrections()
    cw.setShiftKeywords()

    (avg_dx, avg_dy) = cw.avgShiftXY()
    pshift_vs_time = cw.pshiftVsTime()

    if switches["statflag"] == "PERFORM":
        cw.doStat()
    phdr["wavecorr"] = "COMPLETE"

    cw.writeOutFlash()

    return (avg_dx, avg_dy, pshift_vs_time)

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

class ConcurrentWavecal (object):
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

    def __init__ (self, events, outflash, info, reffiles, phdr, hdr,
                  delta_t=1.0, buffer_on=2.0, buffer_off=4.0):

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

        # will be the count rate in each delta_t time interval in the
        # wavecal region of the detector
        self.src_counts = None
        self.delta_t = delta_t          # bin events in this size interval
        # subtract buffer_on from lamp_on and add buffer_off to lamp_off
        # before searching for actual lamp turn-on and turn-off times
        self.buffer_on = buffer_on
        self.buffer_off = buffer_off

        # for FUV segment_list is just one segment name;
        # for NUV this is all three stripe names ["NUVA", "NUVB", "NUVC"]
        # (except for G230L, 3360).
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
        hdu.name = "LAMPFLASH"
        self.deleteCoordinateKeywords (hdu)
        self.ofd.append (hdu)

        cosutil.updateFilename (self.ofd[0].header, self.outflash)
        self.ofd[0].header["wavecorr"] = "COMPLETE"

        # We know this value, so assign it now.
        self.ofd[1].data.field ("nelem")[:] = len (self.spectrum)

    def deleteCoordinateKeywords (self, hdu):
        """Delete keywords that are not relevant for extracted spectra.

        @param hdu: HDU for table of extracted wavecal spectra (will be
            modified in-place)
        @type hdu: pyfits header/data unit object
        """

        ikey = ["TCTYP2", "TCTYP3", "TCRVL2", "TCRVL3", "TCRPX2", "TCRPX3",
                "TCDLT2", "TCDLT3", "TCUNI2", "TCUNI3",
                "TC2_2",  "TC2_3",  "TC3_2",  "TC3_3",
                "TALEN2", "TALEN3"]
        for keyword in ikey:
            if hdu.header.has_key (keyword):
                del hdu.header[keyword]

        # Set the values of these keywords to zero.
        zkey = ["PSHIFTA", "PSHIFTB", "PSHIFTC"] 
        for keyword in zkey:
            if hdu.header.has_key (keyword):
                hdu.header[keyword] = 0.

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

        # total number of tagflash wavecal exposures (commanded value)
        self.numflash = self.hdr.get ("NUMFLASH", default=0)

        t0 = self.time[0]
        t_last = self.time[-1]
        numflash = 0            # will be actual number of tagflash wavecals

        if self.info["tagflash_type"] == TAGFLASH_TYPE_AUTO:
            # Explicitly get LMP_ONi and LMPOFFi for every flash.
            for n in range (self.numflash):
                key_on = "LMP_ON%d" % (n+1)     # one indexed for FITS
                key_off = "LMPOFF%d" % (n+1)
                starttime = self.hdr.get (key_on)
                endtime = self.hdr.get (key_off)
                if endtime < t0 or starttime > t_last:
                    continue
                starttime = max (starttime, t0)
                endtime = min (endtime, t_last)
                self.lamp_on.append (starttime)
                self.lamp_off.append (endtime)
                numflash += 1
        else:
            # The flashes are uniformly spaced, so calculate the
            # expected on and off times.
            starttime = self.hdr.get ("LMP_ON1")
            starttime = max (starttime, t0)
            endtime = self.hdr.get ("LMPOFF1")
            if self.numflash > 1:
                lmpdelta = self.hdr.get ("LMPOFF2") - endtime
                duration = self.hdr.get ("LMPOFF2") - self.hdr.get ("LMP_ON2")
            else:
                lmpdelta = 1.
                duration = endtime - starttime
            for n in range (self.numflash):
                endtime = starttime + duration
                if starttime > t_last:
                    break
                self.lamp_on.append (starttime)
                self.lamp_off.append (endtime)
                starttime += lmpdelta
                numflash += 1

        self.numflash = numflash        # can be modified by findFlash

        # Determine the actual times from the data, and then set the keywords
        # and update the attributes.
        self.findFlash (output=None, update=True)

    def getWavecalParameters (self):
        """Get the matching row from the wavecal parameters table."""

        wcp_info = cosutil.getTable (self.reffiles["wcptab"],
                        filter={"opt_elem": self.info["opt_elem"]},
                        exactly_one=True)
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
        row = 0         # incremented in the second loop over segments
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
                        self.info, xtractab)
            if shift2 is None:
                shift2 = 0.
            self.shift2.append (shift2)

            # Extract wavecal spectra from events table, and determine offset
            # in dispersion direction.
            zero_pixel = 0      # offset into output spectrum
            sdqflags = (DQ_BURST + DQ_PH_LOW + DQ_PH_HIGH + DQ_BAD_TIME +
                        DQ_DEAD + DQ_HOT)
            pshift = {}
            save_spectra = {}   # save just for second loop over segments
            lamp_found = {}     # for second loop over segments
            first = True
            for segment in self.segment_list:   # first loop over segments
                filter["segment"] = segment
                xtract_info = cosutil.getTable (xtractab, filter)
                # not needed yet; just check whether there is a matching row
                disp_info = cosutil.getTable (self.reffiles["disptab"], filter)
                if xtract_info is None or disp_info is None:
                    lamp_found[segment] = False
                    continue
                extr_height = xtract_info.field ("height")[0]
                slope       = xtract_info.field ("slope")[0]
                intercept   = xtract_info.field ("b_spec")[0]
                # The spectrum will first be extracted into this 2-D band.
                spectrum_band = N.zeros ((extr_height, len (self.spectrum)),
                                         dtype=N.float64)
                ccos.xy_extract (self.xi[i0:i1], self.eta[i0:i1],
                        spectrum_band, slope, intercept+shift2,
                        zero_pixel, self.dq[i0:i1], sdqflags)
                self.spectrum = N.sum (spectrum_band, 0)
                save_spectra[segment] = self.spectrum.copy()
                lamp_found[segment] = True      # default
                if xd_shifts[segment] is not None:
                    lamp_info = cosutil.getTable (lamptab,
                                {"segment": segment,
                                 "opt_elem": self.info["opt_elem"],
                                 "cenwave": self.info["cenwave"]})
                    if lamp_info is None:       # no row matched the filter
                        lamp_found[segment] = False
                        continue
                    if first:
                        sum_spectra = self.spectrum.copy()
                        template = lamp_info.field ("intensity")[0].copy()
                        first = False
                    else:
                        sum_spectra += self.spectrum
                        template += lamp_info.field ("intensity")[0]

            # If first is still True, no spectrum was found.
            if first:
                print "%2d no spectrum found" % (n+1,)
                for segment in self.segment_list:
                    pshift[segment] = 0.
                self.pshift.append (pshift)
                continue

            # find offset in dispersion direction
            (global_shift, n50) = wavecal.ttFindWavecalShift (sum_spectra,
                        template, self.info, self.wcp_info)

            # Print results, and save extracted spectra in lampflash table.
            first = True
            for segment in self.segment_list:   # second loop over segments
                pshift[segment] = global_shift
                filter["segment"] = segment
                if not lamp_found[segment]:
                    # no matching row in table
                    spec_found = False
                    message = \
                        "%2d %4s skipped due to missing reference table row" \
                        % (n+1, segment)
                elif xd_shifts[segment] is not None:
                    spec_found = True
                    if first:
                        message = "%2d %4s %9.1f (%5.1f) %9.1f  " \
                                % (n+1, segment, xd_shifts[segment],
                                   xd_locn[segment], pshift[segment]) \
                                + str (n50)
                        first = False
                    else:
                        message = "%2d %4s %9.1f (%5.1f)" \
                                % (n+1, segment, xd_shifts[segment],
                                   xd_locn[segment])
                else:
                    # ttFindWavecalSpectrum couldn't find the spectrum
                    spec_found = False
                    message = "%2d %4s not found" % (n+1, segment)
                cosutil.printMsg (message, VERBOSE)
                # copy to outflash table data
                if lamp_found[segment]:
                    self.saveSpectrum (self.reffiles["disptab"], filter,
                                       n, row, save_spectra[segment],
                                       pshift[segment], xd_shifts[segment],
                                       spec_found)
                else:
                    self.saveSpectrum (self.reffiles["disptab"], filter,
                                       n, row, None, None, None,
                                       spec_found)
                row += 1
            if self.info["detector"] == "NUV":
                cosutil.printMsg ("%2d      avg %5.1f" % (n+1, shift2), VERBOSE)
            self.pshift.append (pshift)

    def saveSpectrum (self, disptab, filter,
                      n, row, spectrum,
                      pshift, shift2, spec_found):
        """Copy the spectrum to the record array for the outflash table.

        @param disptab: name of the dispersion relation table
        @type disptab: string

        @param filter: for extracting the row from the disptab
        @type filter: dictionary

        @param n: index of current lamp flash
        @type n: int

        @param row: row index (zero indexed) in output table
        @type row: int

        @param spectrum: spectrum for current segment or stripe (may be None)
        @type spectrum: array

        @param pshift: shift in dispersion direction (may be None)
        @type pshift: float

        @param shift2: shift in cross-dispersion direction (may be None)
        @type shift2: float

        @param spec_found: was the wavecal spectrum actually found?
        @type spec_found: boolean
        """

        if self.ofd is None:
            return

        t0 = self.lamp_on[n]
        t1 = self.lamp_off[n]

        if spectrum is not None:
            pixel = N.arange (len (spectrum), dtype=N.float64)
            disp_info = cosutil.getTable (disptab, filter, exactly_one=True)
            ncoeff = disp_info.field ("nelem")[0]
            coeff = disp_info.field ("coeff")[0][0:ncoeff]
            pixel -= pshift         # correct the wavelengths for the shift
            wavelength = cosutil.evalDisp (pixel, coeff)

        self.ofd[1].data.field ("segment")[row] = filter["segment"]
        self.ofd[1].data.field ("time")[row] = self.lamp_median[n]
        self.ofd[1].data.field ("exptime")[row] = t1 - t0
        self.ofd[1].data.field ("lamp_on")[row] = self.lamp_on[n]
        self.ofd[1].data.field ("lamp_off")[row] = self.lamp_off[n]
        if spectrum is None:
            self.ofd[1].data.field ("wavelength")[row][:] = 0.
        else:
            self.ofd[1].data.field ("wavelength")[row] = wavelength
        exptime = t1 - t0
        if exptime <= 0.:
            exptime = 1.
        if spectrum is None:
            self.ofd[1].data.field ("gross")[row][:] = 0.
            self.ofd[1].data.field ("shift_disp")[row] = 0.
        else:
            self.ofd[1].data.field ("gross")[row] = spectrum / exptime
            self.ofd[1].data.field ("shift_disp")[row] = pshift
        if shift2 is None:
            self.ofd[1].data.field ("shift_xdisp")[row] = 0.
        else:
            self.ofd[1].data.field ("shift_xdisp")[row] = shift2
        self.ofd[1].data.field ("spec_found")[row] = spec_found

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

    def findFlash (self, output=None, update=True):
        """Find tagflash wavecals in science data.

        @param output: if specified, write array of count rates to this file
            (for testing or debugging)
        @type output: string, or None

        @param update: if true, keywords in input file will be updated
        @type update: boolean
        """

        detector = self.info["detector"]
        exptime = self.info["exptime"]

        xtractab = self.reffiles["xtractab"]

        filter = {"opt_elem": self.info["opt_elem"],
                  "cenwave": self.info["cenwave"],
                  "aperture": "WCA"}

        if detector == "FUV":
            filter["segment"] = self.info["segment"]
            xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
            b_spec = xtract_info.field ("b_spec")[0]
            height = xtract_info.field ("height")[0]
            src_low  = int (b_spec - height)
            src_high = int (b_spec + height)
            eta = self.events.field ("ycorr")
        else:
            # For NUV use only the data for stripe A.
            filter["segment"] = "NUVA"
            xtract_info = cosutil.getTable (xtractab, filter, exactly_one=True)
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
        nbins = int (math.ceil ((time[-1] - time[0]) / self.delta_t))
        istart = N.zeros (nbins, dtype=N.int32)
        istop = N.zeros (nbins, dtype=N.int32)
        src_counts = N.zeros (nbins, dtype=N.int32)
        bkg_counts = N.zeros (nbins, dtype=N.int32)

        ccos.getstartstop (time, istart, istop, self.delta_t)
        ccos.getbkgcounts (eta, dq, istart, istop,
                           bkg_counts, src_counts,
                           bkg1_low, bkg1_high, bkg2_low, bkg2_high,
                           src_low, src_high, bkgsf)
        del dq, bkg_counts

        # Convert to count rate.
        self.src_counts = src_counts.astype (N.float64) / self.delta_t
        del src_counts

        if output is not None:
            ofd = open (output, "w")
            for val in self.src_counts:
                ofd.write ("%g\n" % val)
            ofd.close()

        (hist, step) = self.makeHistogram()

        if hist is None:
            self.numflash = 0
        else:
            cutoff = self.findCutoff (hist, step)
            self.findLampOn (cutoff, time[0])
            self.findLampMedian()

        if cosutil.checkVerbosity (VERBOSE):
            self.printInfo()

        if update:
            self.updateHeader()

    def makeHistogram (self):
        """Make a histogram of src_counts.

        The function value is (hist, step), the histogram and step size.
        If the maximum value in src_counts is less than 20, the count
        rate is so low that it would be difficult to determine when the
        lamp turned on or off, and (None, None) will be returned to
        indicate this case.

        @return: the histogram (array) and step size (float)
        @rtype: tuple
        """

        maxval = N.maximum.reduce (self.src_counts)
        if maxval <= 20.:
            return (None, None)

        i_maxval = int (round (maxval))
        nbins = max (20, len (self.src_counts) // 100)
        step = maxval / float (nbins)
        hist = N.zeros (nbins, dtype=N.int32)
        for src in self.src_counts:
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

    def findLampOn (self, cutoff, t0):
        """Find the actual times when the lamps turned on and off.

        The nominal lamp turn-on and turn-off times were gotten from
        the EVENTS header and saved as self.lamp_on and self.lamp_off.
        This function gets the actual times, constrained to be within
        the time intervals specified in the header.  The attributes
        lamp_on and lamp_off are then updated.

        @param cutoff: a count rate (in src_counts array) greater than this
            indicates that the wavecal lamp was on
        @type cutoff: float

        @param t0: time of first photon event (first element of TIME column)
        @type t0: float
        """

        lamp_on = []
        lamp_off = []

        nbins = len (self.src_counts)
        for i in range (self.numflash):
            lamp_is_on = False
            lamp_on_i = self.lamp_on[i] - self.buffer_on
            lamp_off_i = self.lamp_off[i] + self.buffer_off
            (k_on, k_off) = self.getIndices (nbins, t0, lamp_on_i, lamp_off_i)
            if k_on is None:
                continue
            # search for the actual lamp turn-on time
            for k in range (k_on, k_off+1):
                if self.src_counts[k] > cutoff:
                    lamp_is_on = True
                    t = t0 + k * self.delta_t
                    lamp_on.append (t)
                    break
            if not lamp_is_on:
                continue
            # search for the actual lamp turn-off time
            for k in range (k_off, k_on-1, -1):
                if self.src_counts[k] > cutoff:
                    t = t0 + (k+1) * self.delta_t
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

    def getIndices (self, nbins, t0, lamp_on_i, lamp_off_i):
        """Compute indices in src_counts corresponding to given times.

        @param nbins: number of elements in src_counts array
        @type nbins: int

        @param t0: time of first photon event
        @type t0: float

        @param lamp_on_i: one element of lamp_on array (minus buffer_on)
        @type lamp_on_i: float

        @param lamp_off_i: one element of lamp_off array (plus buffer_off)
        @type lamp_off_i: float

        @return: the indices corresponding to lamp_on_i and lamp_off_i;
            both values will be None if these times are outside the
            interval (t0, t0+delta_t)
        @rtype: tuple
        """

        k_on = (lamp_on_i - t0) / self.delta_t
        k_off = (lamp_off_i - t0) / self.delta_t
        k_on = int (math.floor (k_on))
        k_off = int (math.ceil (k_off))
        if k_on >= nbins or k_off < 0:
            k_on = None
            k_off = None
        else:
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

    def updateHeader (self):
        """Assign extension header keywords with updated info.

        This function updates keywords with actual values (the input
        header contained commanded values).  The keywords that will be
        updated are NUMFLASH, LMP_ONi, LMPOFFi, LMPDURi, and LMPMEDi;
        i normally runs from 1 to NUMFLASH inclusive, but if NUMFLASH is
        large not all the keywords will be present in the input header,
        and in this case (the test is on LMP_ONi) the remaining values
        will not be written to the header.  Note that the _tagflash.fits
        table does contain these values for every flash.
        """

        self.hdr.update ("NUMFLASH", self.numflash)

        for i in range (self.numflash):
            keyword = "LMP_ON%d" % (i+1)
            if not self.hdr.has_key (keyword):
                break
            self.hdr.update (keyword, self.lamp_on[i])
            keyword = "LMPOFF%d" % (i+1)
            self.hdr.update (keyword, self.lamp_off[i])
            keyword = "LMPDUR%d" % (i+1)
            self.hdr.update (keyword, self.lamp_off[i] - self.lamp_on[i])
            keyword = "LMPMED%d" % (i+1)
            self.hdr.update (keyword, self.lamp_median[i])

    def avgShift (self):
        """Compute the average pshift and shift2 offsets.

        @return: a tuple containing the average shifts in the dispersion
            and cross-dispersion directions
        @rtype: tuple
        """

        if self.numflash < 1 or self.info["exptime"] <= 0.:
            return (0., 0.)

        exptime = self.info["exptime"]
        segment = self.segment_list[0]
        time = self.time

        t_prev = time[0]
        pshift_prev = self.pshift[0][segment]
        shift2_prev = self.shift2[0]
        sum_t = 0.
        sum_pshift = 0.
        sum_shift2 = 0.
        for n in range (self.numflash):
            t = self.lamp_median[n]
            pshift = self.pshift[n][segment]
            shift2 = self.shift2[n]
            sum_pshift += (t - t_prev) * (pshift + pshift_prev) / 2.
            sum_shift2 += (t - t_prev) * (shift2 + shift2_prev) / 2.
            t_prev = t
            pshift_prev = pshift
            shift2_prev = shift2

        if time[-1] > t:
            sum_pshift += (time[-1] - t) * pshift
            sum_shift2 += (time[-1] - t) * shift2

        return (sum_pshift/exptime, sum_shift2/exptime)

    def pshiftVsTime (self):
        """Interpolate pshift at one-second intervals.

        @return: an array of the shifts in the dispersion direction at
            one-second intervals
        @rtype: array, or None if there are either no flashes or no data
        """

        if self.numflash < 1 or self.info["exptime"] <= 0.:
            return None

        time = self.time
        nbins = int (math.ceil (time[-1] - time[0]))
        pshift_vs_time = N.zeros (nbins, dtype=N.float32)

        segment = self.segment_list[0]

        t0 = time[0]
        t_prev = time[0]
        pshift_prev = self.pshift[0][segment]
        i = 0
        for n in range (self.numflash):
            t = self.lamp_median[n]
            pshift_t = self.pshift[n][segment]
            max_k = int (round (t)) - i
            if max_k < 1:
                continue
            if i + max_k > nbins:
                full = True
                max_k = nbins - i
            else:
                full = False
            subset = N.arange (max_k, dtype=N.float32)
            slope = (pshift_t - pshift_prev) / (t - t_prev)
            subset = slope * subset + pshift_prev
            pshift_vs_time[i:i+max_k] = subset
            t_prev = t
            pshift_prev = pshift_t
            i += max_k
            if full:
                break

        max_k = nbins - i
        if time[-1] > t and max_k > 0:
            subset = N.ones (max_k, dtype=N.float32) * pshift_t
            pshift_vs_time[i:i+max_k] = subset

        return pshift_vs_time

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
        (b_low, b_high, b_left, b_right) = \
                cosutil.activeArea (info["segment"], reffiles["brftab"])
        self.regions[info["segment"]] = [(b_low, b_high)]

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

    def setShiftKeywords (self):
        """Set PSHIFTA or PSHIFTB to the average fractional pixel offset."""

        xi_diff = self.xi_corr - N.around (self.xi_corr)
        pshift = -xi_diff.mean()

        for segment in self.segment_list:
            key = "PSHIFT" + segment[-1]
            self.hdr.update (key, pshift)

        self.hdr.update ("SHIFT2A", 0.)
        self.hdr.update ("SHIFT2B", 0.)

    def avgShiftXY (self):
        """Return the average offsets in X and Y."""

        (avg_d_xi, avg_d_eta) = self.avgShift()

        return (avg_d_xi, avg_d_eta)

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
        if phdr.get ("OPT_ELEM", "missing") == "G230L" and \
           phdr.get ("cenwave", 0) == 3360:
            self.segment_list = ["NUVA", "NUVB"]
        else:
            self.segment_list = ["NUVA", "NUVB", "NUVC"]

        # Copy xi and eta to the columns for corrected values.
        self.copyColumns()

        # The pshift offset should be applied only within this region.
        self.regions = self.setRegions()

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
            xtract_info = cosutil.getTable (self.reffiles["xtractab"], filter)
            if xtract_info is None:
                continue
            b_spec = xtract_info.field ("b_spec")[0]
            locations.append ((segment, b_spec))

            filter["aperture"] = "PSA"
            xtract_info = cosutil.getTable (self.reffiles["xtractab"], filter)
            if xtract_info is None:
                continue
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

    def setShiftKeywords (self):
        """Set PSHIFT[ABC] to the average fractional pixel offset."""

        xi_diff = self.xi_corr - N.around (self.xi_corr)
        pshift = -xi_diff.mean()

        for segment in self.segment_list:
            key = "PSHIFT" + segment[-1]
            self.hdr.update (key, pshift)

        self.hdr.update ("SHIFT2A", 0.)
        self.hdr.update ("SHIFT2B", 0.)
        self.hdr.update ("SHIFT2C", 0.)

    def avgShiftXY (self):
        """Return the average offsets in X and Y."""

        (avg_d_xi, avg_d_eta) = self.avgShift()

        return (avg_d_eta, avg_d_xi)
