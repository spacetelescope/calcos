import math
import numpy as N
import pyfits
import cosutil
import dispersion
import shiftfile
import wavecal
import ccos
from calcosparam import *       # parameter definitions
import findshift1

def processConcurrentWavecal (events, outflash, shift_file,
                info, switches, reffiles, phdr, hdr):
    """Determine shifts from concurrent (tagflash) wavecal exposures.

    If tagflash mode was not used or the wavecorr switch is not
    "PERFORM", this function returns without doing anything.

    @param events: data block for a corrtag table
    @type events: record array

    @param outflash: name of output file for extracted wavecal spectra
    @type outflash: string

    @param shift_file: if not None, this text file contains values of
        shift1 (and possibly shift2) to override the values found via
        wavecal processing
    @type shift_file: string

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

    @return: the shifts in the dispersion direction at one-second intervals,
        or None if wavecorr is not perform or the input data are not tagflash
        or there are no flashes.
    @rtype: array
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

    cw = initWavecal (events, outflash, shift_file, info, reffiles, phdr, hdr)

    cw.getStartStopTimes()
    if cw.numflash < 1:
        # write an empty lampflash table
        cosutil.printWarning ("No lamp flash was found.")
        cw.outFlashSetup()
        cw.writeOutFlash()
        return (0., 0., None)

    cw.outFlashSetup()          # create an HDU list for the outflash file

    cw.getWavecalParameters()
    cw.findShifts()
    cw.applyCorrections()

    (avg_dx, avg_dy) = cw.avgShift()
    cw.setShiftKeywords (avg_dx, avg_dy)

    shift1_vs_time = cw.shift1VsTime()

    if switches["statflag"] == "PERFORM":
        cw.doStat()
    phdr["wavecorr"] = "COMPLETE"

    cw.writeOutFlash()

    return shift1_vs_time

def initWavecal (events, outflash, shift_file, info, reffiles, phdr, hdr):
    """Return a ConcurrentWavecal object, depending on detector.

    arguments:
    events      data block (recarray object) for an events table corrtag file
    outflash    name of output file for table of extracted wavecal spectra
    shift_file  name of user-supplied file to override shifts (or None)
    info        dictionary of keywords and values
    reffiles    dictionary of reference file names
    phdr        primary header of corrtag file
    hdr         events extension header
    """

    if info["detector"] == "FUV":
        cw = FUVConcurrentWavecal (events, outflash, shift_file,
                                   info, reffiles, phdr, hdr)
    else:
        cw = NUVConcurrentWavecal (events, outflash, shift_file,
                                   info, reffiles, phdr, hdr)

    return cw

class ConcurrentWavecal (object):
    """Process wavecals embedded in a science observation (tagflash).

    @ivar events: data block for a corrtag table
    @type events: record array

    @ivar outflash: name of output file for extracted wavecal spectra
    @type outflash: string

    @ivar shift_file: name of user-supplied file to override shifts
    @type shift_file: string

    @ivar info: dictionary of header keywords and values
    @type info: dictionary

    @ivar reffiles: dictionary of reference file names
    @type reffiles: dictionary

    @ivar phdr: primary header of corrtag file
    @type phdr: pyfits header object

    @ivar hdr: events extension header of corrtag file
    @type hdr: pyfits header object
    """

    def __init__ (self, events, outflash, shift_file,
                  info, reffiles, phdr, hdr,
                  delta_t=1.0, buffer_on=2.0, buffer_off=4.0):

        self.events = events
        self.outflash = outflash
        self.shift_file = shift_file
        self.info = info
        self.reffiles = reffiles
        self.phdr = phdr
        self.hdr = hdr
        self.user_shifts = None

        self.ofd = None                 # HDU list for outflash FITS file

        # lamp_on and lamp_off are in seconds, with the same zero point
        # as the TIME column in the events table
        self.lamp_on = []               # start times of wavecals (flashes)
        self.lamp_off = []              # stop times of wavecals
        self.lamp_median = []           # median times of wavecals
        self.numflash = 0               # number of embedded wavecals

        # Did the user supply a file with overrides for the shifts?
        if self.shift_file is not None:
            self.user_shifts = shiftfile.ShiftFile (self.shift_file,
                                                    self.info["root"],
                                                    self.info["fpoffset"])
        else:
            self.user_shifts = None

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
        self.regions = {}               # apply shift1 only to these regions

        # the number of elements is the number of flashes:
        # each element is a dictionary; the key is segment or stripe name,
        # and the value is the shift in the dispersion direction
        self.shift1 = []                # dictionaries of shift in disp dir
        self.chi_square = []            # dictionaries of Chi square
        self.n_deg_freedom = []         # dictionaries of num of deg of freedom

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
        self.xi_corr = None             # corrected coords (XFULL)
        self.eta_corr = None            # corrected coords (YFULL)
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
        col.append (pyfits.Column (name="NELEM", format="1J",
                                   disp="I6"))
        col.append (pyfits.Column (name="WAVELENGTH", format=rpt+"D",
                                   unit="angstrom"))
        col.append (pyfits.Column (name="GROSS", format=rpt+"E",
                                   unit="count /s"))
        col.append (pyfits.Column (name="SHIFT_DISP", format="1E",
                                   unit="pixel"))
        col.append (pyfits.Column (name="SHIFT_XDISP", format="1E",
                                   unit="pixel"))
        col.append (pyfits.Column (name="SPEC_FOUND", format="1L"))
        col.append (pyfits.Column (name="CHI_SQUARE", format="1E"))
        col.append (pyfits.Column (name="N_DEG_FREEDOM", format="1J",
                                   disp="I5"))
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
        zkey = ["SHIFT1A", "SHIFT1B", "SHIFT1C"]
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
        disptab = self.reffiles["disptab"]
        lamptab = self.reffiles["lamptab"]
        # segment will be added to the filter within the loop.  Note that
        # the aperture is explicitly set to "WCA", because the aperture
        # keyword will give the aperture used for the science data.
        filter_1dx = {"opt_elem": self.info["opt_elem"],
                      "cenwave": self.info["cenwave"],
                      "aperture": "WCA"}
        filter_disp = {"opt_elem": self.info["opt_elem"],
                       "cenwave": self.info["cenwave"],
                       "aperture": "WCA",
                       "fpoffset": self.info["fpoffset"]}
        filter_lamp = {"opt_elem": self.info["opt_elem"],
                       "cenwave": self.info["cenwave"]}
        # These two flags will be used within a loop below.
        got_fpoffset = cosutil.findColumn (lamptab, "fpoffset")
        got_pixel_shift = cosutil.findColumn (lamptab, "fp_pixel_shift")
        if got_fpoffset:
            filter_lamp["fpoffset"] = self.info["fpoffset"]
        xc_range = self.wcp_info.field ("xc_range")
        stepsize = self.wcp_info.field ("stepsize")
        xd_range = self.wcp_info.field ("xd_range")
        box = self.wcp_info.field ("box")
        # fp is for an initial offset when matching the spectrum to the
        # template.  If we've got fpoffset and fp_pixel_shift columns,
        # the initial offset should be zero.
        if got_pixel_shift:
            fp = 0
        else:
            fp = self.info["fpoffset"]

        # Find the offsets in both axes, for each wavecal exposure.
        row = 0         # incremented in the second loop over segments
        cosutil.printMsg (
"  segment    cross-disp        dispersion direction", VERBOSE)
        cosutil.printMsg (
"            shift (locn)      shift  [orig.]  chi sq (n)", VERBOSE)
        cosutil.printMsg (
"  -------   -------------     --------------------------", VERBOSE)
        for n in range (self.numflash):
            (i0, i1) = ccos.range (self.time, self.lamp_on[n], self.lamp_off[n])

            # Find offset from nominal in cross-dispersion direction.
            (shift2, xd_shifts, xd_locn) = \
                wavecal.ttFindWavecalSpectrum (
                        self.xi[i0:i1], self.eta[i0:i1], self.dq[i0:i1],
                        self.info, xd_range, box, xtractab)
            if shift2 is None:
                shift2 = 0.
            self.shift2.append (shift2)

            # Extract wavecal spectra from events table, and determine offset
            # in dispersion direction.
            x_offset = self.info["x_offset"]    # offset of lamptab in template
            sdqflags = self.info["sdqflags"]
            shift1 = {}                 # to be saved in an attribute
            chi_square = {}             # to be saved in an attribute
            n_deg_freedom = {}          # to be saved in an attribute
            save_spectra = {}   # save, but just within this function
            save_templates = {}
            spec_found = {}     # true if spectrum was found
            lamp_found = {}     # for second loop over segments
            at_least_one_found = False
            ((user_shift1, user_shift2), nfound) = ((None, None), 0)
            for segment in self.segment_list:   # first loop over segments
                # Check the user-supplied shift file for a match with this
                # flash number and segment/stripe; we don't need the value
                # yet, but if there's a match we want to set the flag to
                # say that the shift was found.
                if self.user_shifts is not None:
                    ((user_shift1, user_shift2), nfound) = \
                        self.user_shifts.getShifts ((n, segment))
                spec_found[segment] = False     # initial value
                filter_1dx["segment"] = segment
                # filter_disp["segment"] = segment      # don't need this yet
                filter_lamp["segment"] = segment
                xtract_info = cosutil.getTable (xtractab, filter_1dx)
                if xtract_info is None:
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
                        x_offset, self.dq[i0:i1], sdqflags)
                self.spectrum = N.sum (spectrum_band, 0)
                save_spectra[segment] = self.spectrum.copy()
                lamp_found[segment] = True      # default
                lamp_info = cosutil.getTable (lamptab, filter_lamp)
                if lamp_info is None:           # no row matched the filter
                    lamp_found[segment] = False
                    continue
                raw_template = lamp_info.field ("intensity")[0]
                save_templates[segment] = cosutil.getTemplate (raw_template,
                                          x_offset, len (self.spectrum))
                if user_shift1 is not None or xd_shifts[segment] is not None:
                    spec_found[segment] = True
                    at_least_one_found = True

            if not at_least_one_found:
                print "%2d no spectrum found" % (n+1,)
                for segment in self.segment_list:
                    shift1[segment] = 0.
                self.shift1.append (shift1)
                continue

            # find offset in dispersion direction
            fs1 = findshift1.Shift1 (save_spectra, save_templates,
                                     self.info, self.reffiles,
                                     xc_range, stepsize, fp, spec_found)
            fs1.findShifts()

            # Print results, and save extracted spectra in lampflash table.
            for segment in self.segment_list:   # second loop over segments
                filter_lamp["segment"] = segment
                lamp_info = cosutil.getTable (lamptab, filter_lamp)
                if got_pixel_shift:
                    fp_pixel_shift = lamp_info.field ("fp_pixel_shift")[0]
                else:
                    fp_pixel_shift = 0.
                user_specified = False
                if self.user_shifts is not None:        # override shifts?
                    ((user_shift1, user_shift2), nfound) = \
                        self.user_shifts.getShifts ((n, segment))
                    if user_shift1 is not None:
                        fs1.setShift1 (segment, user_shift1-fp_pixel_shift)
                        user_specified = True
                shift1[segment] = fs1.getShift1 (segment) + fp_pixel_shift
                orig_shift1 = fs1.getOrigShift1 (segment) + fp_pixel_shift
                chi_square[segment] = fs1.getChiSq (segment)
                n_deg_freedom[segment] = fs1.getNdf (segment)
                filter_disp["segment"] = segment
                if not lamp_found[segment]:
                    # no matching row in table
                    foundit = False
                    message = \
                        "%2d %4s skipped due to missing reference table row" \
                        % (n+1, segment)
                else:
                    foundit = fs1.getSpecFound (segment)
                    if xd_shifts[segment] is None:
                        # ttFindWavecalSpectrum couldn't find the spectrum
                        message = \
"%2d %4s      ---- (%5.1f) %9.1f [%6.1f]  %6.1f (%d)  # not found in XD" \
                            % (n+1, segment,
                               xd_locn[segment], shift1[segment], orig_shift1,
                               fs1.getChiSq (segment), fs1.getNdf (segment))
                    else:
                        message = \
"%2d %4s %9.1f (%5.1f) %9.1f [%6.1f]  %6.1f (%d)" \
                            % (n+1, segment, xd_shifts[segment],
                               xd_locn[segment], shift1[segment], orig_shift1,
                               fs1.getChiSq (segment), fs1.getNdf (segment))
                        if not foundit:
                            message = message + "  # not found"
                    if user_specified:
                        message = message + "  # user-specified"
                cosutil.printMsg (message, VERBOSE)
                # copy to outflash table data
                if lamp_found[segment]:
                    self.saveSpectrum (disptab, filter_disp,
                                       n, row, save_spectra[segment],
                                       shift1[segment], xd_shifts[segment],
                                       foundit, chi_square[segment],
                                       n_deg_freedom[segment])
                else:
                    self.saveSpectrum (disptab, filter_disp,
                                       n, row, None, None, None,
                                       foundit, chi_square[segment],
                                       n_deg_freedom[segment])
                row += 1
            if self.info["detector"] == "NUV":
                cosutil.printMsg ("%2d      avg %5.1f" % (n+1, shift2), VERBOSE)
            self.shift1.append (shift1)
            self.chi_square.append (chi_square)
            self.n_deg_freedom.append (n_deg_freedom)

    def saveSpectrum (self, disptab, filter,
                      n, row, spectrum,
                      shift1, shift2, spec_found,
                      chi_square, n_deg_freedom):
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
        @param shift1: shift in dispersion direction (may be None)
        @type shift1: float
        @param shift2: shift in cross-dispersion direction (may be None)
        @type shift2: float
        @param spec_found: was the wavecal spectrum actually found?
        @type spec_found: boolean
        @param chi_square: Chi square for the current flash
        @type chi_square: float
        @param n_deg_freedom: number of degrees of freedom for current flash
        @type n_deg_freedom: int
        """

        if self.ofd is None:
            return

        t0 = self.lamp_on[n]
        t1 = self.lamp_off[n]

        if spectrum is not None:
            pixel = N.arange (len (spectrum), dtype=N.float64)
            disp_rel = dispersion.Dispersion (disptab, filter)
            pixel -= shift1         # correct the wavelengths for the shift
            # Correct for any extra pixels in the dispersion direction.
            pixel -= self.info["x_offset"]
            wavelength = disp_rel.evalDisp (pixel)
            disp_rel.close()

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
            self.ofd[1].data.field ("shift_disp")[row] = shift1
        if shift2 is None:
            self.ofd[1].data.field ("shift_xdisp")[row] = 0.
        else:
            self.ofd[1].data.field ("shift_xdisp")[row] = shift2
        self.ofd[1].data.field ("spec_found")[row] = spec_found
        self.ofd[1].data.field ("chi_square")[row] = chi_square
        self.ofd[1].data.field ("n_deg_freedom")[row] = n_deg_freedom

    def applyCorrections (self):
        """Apply the shift1[a-c] and shift2[a-c] offsets."""

        # There will be at least one flash.
        for n in range (self.numflash):
            (i0, i1, extrapolate) = self.getInterval (n)
            self.shift2Corr (n, i0, i1, extrapolate)
            self.shift1Corr (n, i0, i1, extrapolate)

    def getInterval (self, n):
        """Get the slice for times between wavecal exposures n and n+1.

        @param n: index of flash at left end of time interval; the valid
            range for n is from 0 to the number of flashes minus one, inclusive
        @type n: integer

        @return: (i0, i1, extrapolate), where i0 and i1 are the indices in
            the time array (Python slice indices) over which to apply the
            shift based on flashes n and n+1 (unless n is the last), and
            extrapolate is True if n is the last flash
        @rtype: tuple
        """

        if n == 0:
            t0 = self.time[0]           # extrapolate to the beginning
        else:
            t0 = self.lamp_median[n]

        if n == self.numflash - 1:
            extrapolate = True
            t1 = self.time[-1]
        else:
            extrapolate = False
            t1 = self.lamp_median[n+1]

        (i0, i1) = ccos.range (self.time, t0, t1)

        return (i0, i1, extrapolate)

    def shift1Corr (self, n, i0, i1, extrapolate=False):
        """Correct the pixel coordinates in the dispersion direction.

        @param n: apply shift1 for the nth time interval between wavecals
        @type n: int

        @param i0: [i0:i1] is the slice of event numbers to be corrected
        @type i0: int

        @param i1: [i0:i1] is the slice of event numbers to be corrected
        @type i1: int

        @param extrapolate: True if n is the last flash
        @type extrapolate: boolean
        """

        for segment in self.segment_list:

            # Restrict the correction to the applicable regions.
            shift_flags = N.zeros (i1 - i0, dtype=N.bool8)
            locn_list = self.regions[segment]
            for region in locn_list:
                if region[0] is None:
                    shift_flags |= N.where (
                                   self.eta_corr[i0:i1] < region[1], 1, 0)
                elif region[1] is None:
                    shift_flags |= N.where (
                                   self.eta_corr[i0:i1] >= region[0], 1, 0)
                else:
                    shift_flags |= N.logical_and (
                                   self.eta_corr[i0:i1] >= region[0],
                                   self.eta_corr[i0:i1] < region[1])

            shift1_zero = self.shift1[n][segment]
            if extrapolate:
                self.xi_corr[i0:i1] = N.where (shift_flags,
                        self.xi_corr[i0:i1] - shift1_zero,
                        self.xi_corr[i0:i1])
            else:
                # Note that i0 & i1 do not necessarily correspond to t0 and t1.
                t0 = self.lamp_median[n]
                t1 = self.lamp_median[n+1]
                if t1 <= t0:
                    slope = 0.
                else:
                    slope = (self.shift1[n+1][segment] -
                             self.shift1[n][segment]) / (t1 - t0)
                self.xi_corr[i0:i1] = N.where (shift_flags,
                    self.xi_corr[i0:i1] -
                        ((self.time[i0:i1] - t0) * slope + shift1_zero),
                    self.xi_corr[i0:i1])

    def findFlash (self, output=None, update=True):
        """Find tagflash wavecals in science data.

        @param output: if specified, write array of count rates to this file
            (for testing)
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
            src_low = int (b_spec - height)
            src_high = NUV_Y - 1
            eta = self.events.field ("rawy")

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
        """Compute the shift1 and shift2 offsets averaged over time.

        @return: a tuple of dictionaries (key is segment or stripe name)
            containing the shifts in the dispersion and cross-dispersion
            directions averaged over the exposure
        @rtype: tuple of dictionaries
        """

        avg_dx = {}
        avg_dy = {}
        if self.numflash < 1 or self.info["exptime"] <= 0.:
            return (avg_dx, avg_dy)

        exptime = self.info["exptime"]
        time = self.time

        for segment in self.segment_list:
            t_prev = time[0]
            shift1_prev = self.shift1[0][segment]
            shift2_prev = self.shift2[0]
            sum_t = 0.
            sum_shift1 = 0.
            sum_shift2 = 0.
            for n in range (self.numflash):
                t = self.lamp_median[n]
                shift1 = self.shift1[n][segment]
                shift2 = self.shift2[n]
                sum_shift1 += (t - t_prev) * (shift1 + shift1_prev) / 2.
                sum_shift2 += (t - t_prev) * (shift2 + shift2_prev) / 2.
                t_prev = t
                shift1_prev = shift1
                shift2_prev = shift2

            if time[-1] > t:
                sum_shift1 += (time[-1] - t) * shift1
                sum_shift2 += (time[-1] - t) * shift2
            avg_dx[segment] = sum_shift1/exptime
            avg_dy[segment] = sum_shift2/exptime

        return (avg_dx, avg_dy)

    def setShiftKeywords (self, avg_dx, avg_dy):
        """Assign values to the shift keywords.

        Keywords SHIFT1[ABC] will be set to the average offset in the
        dispersion direction, SHIFT2[ABC] to the average offset in the
        cross-dispersion direction, and DPIXEL1[ABC] to:
            XFULL - (XFULL rounded to an integer)

        @param avg_dx: dictionary of the average shift in the dispersion
            direction
        @type avg_dx: dictionary
        @param avg_dy: dictionary of the average shift in the cross-dispersion
            direction
        @type avg_dy: dictionary
        """

        for segment in self.segment_list:
            key = "SHIFT1" + segment[-1]
            self.hdr.update (key, avg_dx[segment])
            key = "SHIFT2" + segment[-1]
            self.hdr.update (key, avg_dy[segment])
            sum_chisq = 0.
            sum_ndf = 0
            for n in range (self.numflash):
                sum_chisq += self.chi_square[n][segment]
                sum_ndf += (self.n_deg_freedom[n][segment] + 1)
            sum_ndf -= 1
            key = "chi_sq_" + segment[-1]
            self.hdr.update (key, round (sum_chisq, 1))
            key = "ndf_" + segment[-1]
            self.hdr.update (key, sum_ndf)

            # use self.regions for dpixel1[abc]
            shift_flags = N.zeros (len (self.eta_corr), dtype=N.bool8)
            locn_list = self.regions[segment]
            # if NUV, take the region for the PSA (lower pixel numbers)
            region = locn_list[0]
            if region[0] is None:
                shift_flags |= N.where (self.eta_corr < region[1], 1, 0)
            elif region[1] is None:
                shift_flags |= N.where (self.eta_corr >= region[0], 1, 0)
            else:
                shift_flags |= N.logical_and (self.eta_corr >= region[0],
                                              self.eta_corr < region[1])
            xi = self.xi_corr[shift_flags]      # copy out the relevant subset
            xi_diff = xi - N.around (xi)
            dpixel1 = xi_diff.mean()
            key = "DPIXEL1" + segment[-1]
            self.hdr.update (key, dpixel1)

    def shift1VsTime (self):
        """Interpolate shift1 at one-second intervals.

        @return: an array of the shifts in the dispersion direction at
            one-second intervals
        @rtype: array, or None if there are either no flashes or no data
        """

        if self.numflash < 1 or self.info["exptime"] <= 0.:
            return None

        time = self.time
        nbins = int (math.ceil (time[-1] - time[0]))
        shift1_vs_time = N.zeros (nbins, dtype=N.float32)

        segment = self.segment_list[0]

        t0 = time[0]
        t_prev = time[0]
        shift1_prev = self.shift1[0][segment]
        i = 0
        for n in range (self.numflash):
            t = self.lamp_median[n]
            shift1_t = self.shift1[n][segment]
            max_k = int (round (t)) - i
            if max_k < 1:
                continue
            if i + max_k > nbins:
                full = True
                max_k = nbins - i
            else:
                full = False
            subset = N.arange (max_k, dtype=N.float32)
            slope = (shift1_t - shift1_prev) / (t - t_prev)
            subset = slope * subset + shift1_prev
            shift1_vs_time[i:i+max_k] = subset
            t_prev = t
            shift1_prev = shift1_t
            i += max_k
            if full:
                break

        max_k = nbins - i
        if time[-1] > t and max_k > 0:
            subset = N.ones (max_k, dtype=N.float32) * shift1_t
            shift1_vs_time[i:i+max_k] = subset

        return shift1_vs_time

class FUVConcurrentWavecal (ConcurrentWavecal):

    def __init__ (self, events, outflash, shift_file,
                  info, reffiles, phdr, hdr):

        ConcurrentWavecal.__init__ (self, events, outflash, shift_file,
                                    info, reffiles, phdr, hdr)
        self.xi  = events.field ("XDOPP")
        self.eta = events.field ("YCORR")
        self.dq  = events.field ("DQ")
        self.xi_corr  = events.field ("XFULL")
        self.eta_corr = events.field ("YFULL")
        self.spectrum = N.zeros (FUV_EXTENDED_X, dtype=N.float64)
        self.segment_list = [info["segment"]]

        # Copy xi and eta to the columns for corrected values.
        self.copyColumns()

        # The shift1 offset should be applied only within this region.
        (b_low, b_high, b_left, b_right) = \
                cosutil.activeArea (info["segment"], reffiles["brftab"])
        self.regions[info["segment"]] = [(b_low, b_high)]

    def shift2Corr (self, n, i0, i1, extrapolate=False):
        """Correct the pixel coordinates in the cross-dispersion direction.

        The difference between this version and the one for NUV is that this
        one limits the shift to the active area.

        @param n: apply shift2 for the nth time interval between wavecals
        @type n: int

        @param i0: [i0:i1] is the slice of event numbers to be corrected
        @type i0: int

        @param i1: [i0:i1] is the slice of event numbers to be corrected
        @type i1: int

        @param extrapolate: True if n is the last flash
        @type extrapolate: boolean
        """

        # Restrict the correction to the applicable region.  Note that the
        # limits of the region (the active area) are not adjusted by shift2.
        shift_flags = N.zeros (i1 - i0, dtype=N.bool8)
        region = self.regions[self.segment_list[0]][0]
        shift_flags |= N.logical_and (
                       self.eta_corr[i0:i1] >= region[0],
                       self.eta_corr[i0:i1] < region[1])

        shift2_zero = self.shift2[n]
        if extrapolate:
            self.eta_corr[i0:i1] = N.where (shift_flags,
                                   self.eta_corr[i0:i1] - shift2_zero,
                                   self.eta_corr[i0:i1])
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

    def __init__ (self, events, outflash, shift_file,
                  info, reffiles, phdr, hdr):

        ConcurrentWavecal.__init__ (self, events, outflash, shift_file,
                                    info, reffiles, phdr, hdr)
        self.xi  = events.field ("XDOPP")
        self.eta = events.field ("RAWY")
        self.dq  = events.field ("DQ")
        self.xi_corr  = events.field ("XFULL")
        self.eta_corr = events.field ("YFULL")
        self.spectrum = N.zeros (NUV_EXTENDED_X, dtype=N.float64)
        if phdr.get ("OPT_ELEM", "missing") == "G230L" and \
           phdr.get ("cenwave", 0) == 3360:
            self.segment_list = ["NUVA", "NUVB"]
        else:
            self.segment_list = ["NUVA", "NUVB", "NUVC"]

        # Copy xi and eta to the columns for corrected values.
        self.copyColumns()

        # The shift1 offset should be applied only within this region.
        self.regions = self.setRegions()

    def setRegions (self):
        """Determine the regions over which shift1 should be applied.

        The function value is a dictionary with nominally three entries.
        Segment name is the key, and each value is a list of the two
        intervals (one for PSA, one for WCA) over which the shift in the
        dispersion direction (shift1) should be applied.  The limits of
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
        # [first,last] is the slice over which shift1[a-c] should be
        # applied.  There should be six such intervals, one for each
        # stripe, and for apertures PSA and WCA.
        # Use None for the lower and upper cutoffs so every event will be
        # included.
        intervals = []
        first = None                    # no lower cutoff
        for i in range (len_locn):
            segment = locations[i][0]
            locn = locations[i][1]
            if i == len_locn - 1:
                last = None             # no upper cutoff
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

    def shift2Corr (self, n, i0, i1, extrapolate=False):
        """Correct the pixel coordinates in the cross-dispersion direction.

        @param n: apply shift2 for the nth time interval between wavecals
        @type n: int

        @param i0: [i0:i1] is the slice of event numbers to be corrected
        @type i0: int

        @param i1: [i0:i1] is the slice of event numbers to be corrected
        @type i1: int

        @param extrapolate: True if n is the last flash
        @type extrapolate: boolean
        """

        shift2_zero = self.shift2[n]
        if extrapolate:
            self.eta_corr[i0:i1] -= shift2_zero
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
