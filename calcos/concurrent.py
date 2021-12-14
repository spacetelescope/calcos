from __future__ import absolute_import, division         # confidence high
import copy
import math
import os
import numpy as np
import astropy.io.fits as fits
from . import cosutil
from . import dispersion
from . import extract
from . import shiftfile
from . import wavecal
from . import ccos
from .calcosparam import *       # parameter definitions
from . import findshift1

# add DELTA_SHIFT[12] to a shift for segment A to get the shift for B
DELTA_SHIFT1 = 0.0
DELTA_SHIFT2 = 0.0

# This is used in makeHistogram.  If the maximum value of the wavecal
# count rate within the search region for a lamp flash is less than or
# equal to this value, it will be assumed either that the lamp was not on
# or that the search region did not cover the wavecal spectrum.
MIN_COUNT_RATE = 20.

# This is the nominal location of an NUV wavecal image; X0 is in the more
# rapidly varying axis, and Y0 is in the less rapidly varying axis.
# xxx these should be gotten from a reference table
X0 = 600.
Y0 = 606.
# search range in each axis
DX = 50
DY = 50

def processConcurrentWavecal(events, outflash, shift_file,
                             info, switches, reffiles, phdr, hdr):
    """Determine shifts from concurrent (tagflash) wavecal exposures.

    If tagflash mode was not used or the wavecorr switch is "OMIT" or
    "SKIPPED", this function will return without doing anything.

    Parameters
    ----------
    events: array_like
        Data block for a corrtag table.

    outflash: str
        Name of output file for extracted wavecal spectra.

    shift_file: str or None
        If not None, this text file contains values of shift1 (and
        possibly shift2) to override the values found via wavecal
        processing.

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switch keywords and values.

    reffiles: dictionary
        Reference file names.

    phdr: pyfits Header object
        Primary header of corrtag file.

    hdr: pyfits Header object
        Events extension header of corrtag file.

    Returns
    -------
    (tl_time, shift1_vs_time, wavecorr): tuple of three items
        tl_time is the array of times at one-second intervals, for the
        timeline table.  shift1_vs_time is the array of corresponding
        values of shift1a or shift1b, or None if there were no flashes or
        if the exposure time is zero.  wavecorr will normally be "COMPLETE"
        but may be some other value, e.g. "SKIPPED", if wavecal processing
        was not actually done.
    """

    tl_time = np.zeros(1, dtype=np.float32)     # replaced later

    if not info["tagflash"]:
        return (tl_time, None, switches["wavecorr"])

    # This test allows processing to continue if wavecorr is either
    # PERFORM or COMPLETE.
    if switches["wavecorr"] == "OMIT" or switches["wavecorr"] == "SKIPPED":
        cw = initWavecal(events, outflash, shift_file,
                         info, switches, reffiles, phdr, hdr)
        tl_time = cosutil.timelineTimes(self.time[0], self.time[-1], dt=1.)
        return (tl_time, None, switches["wavecorr"])

    cosutil.printMsg("Process tagflash wavecal")
    wavecal.printWavecalRef(reffiles)
    cosutil.printRef("disptab", reffiles)

    cw = initWavecal(events, outflash, shift_file, info, switches, reffiles,
                     phdr, hdr)
    if cw.override_segment_B:
        if cw.segment_A_present:
            # Copy lampflash_a.fits to lampflash_b.fits.
            cw.copySegAtoSegB()
            cw.applyCorrections()
            (avg_dx, avg_dy) = cw.avgShift()
            cw.setShiftKeywords(avg_dx, avg_dy)
            (tl_time, shift1_vs_time) = cw.shift1VsTime()
            cw.closeOutFlash()
        else:
            cw.setSegBtoZero()          # set shifts to fp_pixel_shift
            cw.setUpOutFlash()
            (avg_dx, avg_dy) = cw.miscSegB()
            cw.applyCorrections()
            cw.setShiftKeywords(avg_dx, avg_dy)
            (tl_time, shift1_vs_time) = cw.shift1VsTime()
            cw.writeOutFlash()
        phdr["wavecorr"] = cw.wavecorr
        return (tl_time, shift1_vs_time, cw.wavecorr)

    cw.getStartStopTimes()
    if cw.numflash < 1 and cw.shift_file is not None:
        cw.dummyFlash()
    if cw.numflash < 1:
        # write an empty lampflash table
        cosutil.printWarning("No lamp flash was found.")
        cw.wavecorr = "SKIPPED"
        cw.setUpOutFlash()
        phdr["wavecorr"] = cw.wavecorr
        phdr["lampused"] = "NONE"
        cw.ofd[0].header["lampused"] = "NONE"
        cw.writeOutFlash()
        return (tl_time, None, cw.wavecorr)

    cw.setUpOutFlash()          # create an HDU list for the outflash file

    cw.getWavecalParameters()
    cw.findShifts()
    cw.applyCorrections()

    (avg_dx, avg_dy) = cw.avgShift()
    cw.setShiftKeywords(avg_dx, avg_dy)

    (tl_time, shift1_vs_time) = cw.shift1VsTime()

    if switches["statflag"] == "PERFORM":
        cw.doStat()
    phdr["wavecorr"] = cw.wavecorr

    cw.writeOutFlash()

    return (tl_time, shift1_vs_time, cw.wavecorr)

def r_key(x):
    """Comparison key for sorting locations in setRegions().

    The comparison is based entirely on the second element, b_spec.
    """

    return x[1]

def initWavecal(events, outflash, shift_file, info, switches, reffiles,
                phdr, hdr):
    """Return a ConcurrentWavecal object, depending on detector.

    Parameters
    ----------
    events: array_like
        Data block for an events table corrtag file.

    outflash: str
        Name of output file for table of extracted wavecal spectra.

    shift_file: str or None
        Name of user-supplied file to override shifts (or None).

    info: dictionary
        Header keywords and values.

    switches: dictionary
        Calibration switches.

    reffiles: dictionary
        Reference file names.

    phdr: pyfits Header object
        Primary header of corrtag file.

    hdr: pyfits Header object
        Events extension header.
    """

    if info["detector"] == "FUV":
        cw = FUVConcurrentWavecal(events, outflash, shift_file,
                                  info, switches, reffiles, phdr, hdr)
    elif info["obstype"] == "IMAGING":
        cw = NUVImagingWavecal(events, outflash, shift_file,
                               info, switches, reffiles, phdr, hdr)
    else:
        cw = NUVConcurrentWavecal(events, outflash, shift_file,
                                  info, switches, reffiles, phdr, hdr)

    return cw

class ConcurrentWavecal(object):
    """Process wavecals embedded in a science observation (tagflash).

    @ivar events: data block for a corrtag table
    @type events: record array

    @ivar outflash: name of output file for extracted wavecal spectra
    @type outflash: string

    @ivar shift_file: name of user-supplied file to override shifts
    @type shift_file: string

    @ivar info: dictionary of header keywords and values
    @type info: dictionary

    @ivar switches: calibration switches
    @type switches: dictionary

    @ivar reffiles: dictionary of reference file names
    @type reffiles: dictionary

    @ivar phdr: primary header of corrtag file
    @type phdr: pyfits header object

    @ivar hdr: events extension header of corrtag file
    @type hdr: pyfits header object
    """

    def __init__(self, events, outflash, shift_file,
                 info, switches, reffiles, phdr, hdr,
                 delta_t=0.2, buffer_on=2.0, buffer_off=4.0):

        self.events = events
        self.outflash = outflash
        self.shift_file = shift_file
        self.info = info
        self.switches = switches
        self.reffiles = reffiles
        self.phdr = phdr
        self.hdr = hdr
        self.wavecorr = "COMPLETE"      # possibly reset later
        self.user_shifts = None
        self.override_segment_B = False
        self.segment_A_present = True

        self.ofd = None                 # HDU list for outflash FITS file

        # lamp_on and lamp_off are in seconds, with the same zero point
        # as the TIME column in the events table
        self.lamp_on = []               # start time of each flash
        self.lamp_off = []              # stop time of each flash
        self.lamp_duration = []         # length of each flash
        self.lamp_median = []           # median time of each flash
        self.numflash = 0               # number of flashes
        self.lamp_is_on = False         # true if any flash was actually on

        # Did the user supply a file with overrides for the shifts?
        if self.shift_file is not None:
            self.user_shifts = shiftfile.ShiftFile(self.shift_file,
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
        # for NUV this is all three stripe names ["NUVA", "NUVB", "NUVC"].
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
        self.time = events.field("TIME").astype(np.float64)

        # These five columns are assigned by a subclass, depending on
        # the detector.  xi is in the dispersion direction, and eta is
        # in the cross-dispersion direction.
        self.xi = None
        self.eta = None
        self.dq = None                  # data quality flags; 0 is OK
        self.xi_corr = None             # corrected coords (XFULL)
        self.eta_corr = None            # corrected coords (YFULL)
        self.spectrum = None            # scratch space for extracted spectrum

    def dummyFlash(self):
        """Assign values for one flash, even if there wasn't."""

        self.numflash = 1
        self.lamp_on = [0.]
        self.lamp_off = [1.]
        self.lamp_duration = [1.]
        self.lamp_median = [0.5]

    def setUpOutFlash(self):
        """Create an HDU list for the outflash FITS table."""

        if not self.outflash:
            self.ofd = None
            return

        # Number of elements in a WAVELENGTH or GROSS array.
        rpt = str(len(self.spectrum))

        col = []
        col.append(fits.Column(name="SEGMENT", format="4A"))
        col.append(fits.Column(name="TIME", format="1D",
                               disp="F8.3", unit="s"))
        col.append(fits.Column(name="EXPTIME", format="1D",
                               disp="F8.3", unit="s"))
        col.append(fits.Column(name="LAMP_ON", format="1D",
                               disp="F8.3", unit="s"))
        col.append(fits.Column(name="LAMP_OFF", format="1D",
                               disp="F8.3", unit="s"))
        col.append(fits.Column(name="NELEM", format="1J",
                               disp="I6"))
        col.append(fits.Column(name="WAVELENGTH", format=rpt+"D",
                               unit="angstrom"))
        col.append(fits.Column(name="NET", format=rpt+"E",
                               unit="count /s"))
        col.append(fits.Column(name="GROSS", format=rpt+"E",
                               unit="count /s"))
        col.append(fits.Column(name="BACKGROUND", format=rpt+"E",
                               unit="count /s"))
        col.append(fits.Column(name="SHIFT_DISP", format="1E",
                               unit="pixel"))
        col.append(fits.Column(name="SHIFT_XDISP", format="1E",
                               unit="pixel"))
        col.append(fits.Column(name="SPEC_FOUND", format="1L"))
        col.append(fits.Column(name="CHI_SQUARE", format="1E"))
        col.append(fits.Column(name="N_DEG_FREEDOM", format="1J",
                               disp="I5"))
        cd = fits.ColDefs(col)

        nrows = self.numflash * len(self.segment_list)

        primary_hdu = fits.PrimaryHDU(header=self.phdr)
        self.ofd = fits.HDUList(primary_hdu)

        #
        # The WCS table keywords don't need to be propagated to the lampflash
        # extension, so we can remove them before we create the hdu
        newheader = self.hdr.copy()
        self.deleteCoordinateKeywords(newheader)

        hdu = fits.BinTableHDU.from_columns(cd, header=newheader, nrows=nrows)
        hdu.name = "LAMPFLASH"
        self.ofd.append(hdu)

        cosutil.updateFilename(self.ofd[0].header, self.outflash)
        nextend = len(self.ofd) - 1
        self.ofd[0].header["nextend"] = nextend
        self.ofd[0].header["wavecorr"] = self.wavecorr

        # We know this value, so assign it now.
        self.ofd[1].data.field("nelem")[:] = len(self.spectrum)

    def deleteCoordinateKeywords(self, header):
        """Delete keywords that are not relevant for extracted spectra.

        Parameters
        ----------
        header: FITS header object
            Header for table of extracted wavecal spectra (will be modified
            in-place).
        """

        ikey = ["TCTYP2", "TCTYP3", "TCRVL2", "TCRVL3", "TCRPX2", "TCRPX3",
                "TCDLT2", "TCDLT3", "TCUNI2", "TCUNI3",
                "TC2_2",  "TC2_3",  "TC3_2",  "TC3_3",
                "TALEN2", "TALEN3"]
        for keyword in ikey:
            if keyword in header:
                del header[keyword]

        # Set the values of these keywords to zero.
        zkey = ["SHIFT1A", "SHIFT1B", "SHIFT1C"]
        for keyword in zkey:
            if keyword in header:
                header[keyword] = 0.

    def doStat(self):
        """Compute mean and max of the GROSS column."""

        if self.ofd is not None:
            cosutil.doTagFlashStat(self.ofd)

    def writeOutFlash(self):
        """Write the outflash HDU list to the output file."""

        if self.ofd is not None:
            self.ofd.writeto(self.outflash, output_verify="fix")
            self.ofd.close()

    def copyColumns(self):
        """Copy xi and eta to xi_corr and eta_corr."""

        self.xi_corr[:] = self.xi.copy()
        self.eta_corr[:] = self.eta.copy()

    def getStartStopTimes(self):
        """Get the times when the lamp was turned on or off."""

        # total number of tagflash wavecal exposures (commanded value)
        self.numflash = self.hdr.get("NUMFLASH", default=0)

        t0 = self.time[0]
        t_last = self.time[-1]
        numflash = 0            # will be actual number of tagflash wavecals

        # First assign reasonable initial values, based on header keywords.
        if self.info["tagflash_type"] == TAGFLASH_TYPE_AUTO:
            # Explicitly get LMP_ONi and LMPOFFi for every flash.
            for n in range(self.numflash):
                key_on = "LMP_ON%d" % (n+1)     # one indexed for FITS
                key_off = "LMPOFF%d" % (n+1)
                key_duration = "LMPDUR%d" % (n+1)
                starttime = self.hdr.get(key_on)
                endtime = self.hdr.get(key_off)
                duration = self.hdr.get(key_duration)
                if endtime < t0 or starttime > t_last:
                    continue
                starttime = max(starttime, t0)
                endtime = min(endtime, t_last)
                duration = min(duration, endtime-starttime)
                self.lamp_on.append(starttime)
                self.lamp_off.append(endtime)
                self.lamp_duration.append(duration)
                self.lamp_median.append((starttime + endtime) / 2.)
                numflash += 1
        else:
            # The flashes are uniformly spaced, so calculate the
            # expected on and off times.
            starttime = self.hdr.get("LMP_ON1")
            starttime = max(starttime, t0)
            endtime = self.hdr.get("LMPOFF1")
            duration = self.hdr.get("LMPDUR1")
            if self.numflash > 1:
                spacing = self.hdr.get("LMP_ON2") - self.hdr.get("LMP_ON1")
            else:
                spacing = 1.
            for n in range(self.numflash):
                endtime = starttime + duration
                if starttime > t_last:
                    break
                self.lamp_on.append(starttime)
                self.lamp_off.append(endtime)
                self.lamp_duration.append(duration)
                self.lamp_median.append((starttime + endtime) / 2.)
                starttime += spacing
                numflash += 1

        self.numflash = numflash        # can be modified by findFlash

        # Determine the actual times from the data, and then set the keywords
        # and update the attributes.
        self.findFlash(output=None, update=True)

    def getWavecalParameters(self):
        """Get the matching row from the wavecal parameters table."""

        if self.info["obstype"] == "IMAGING":
            self.wcp_info = None
        else:
            wcp_info = cosutil.getTable(self.reffiles["wcptab"],
                            filter={"opt_elem": self.info["opt_elem"]},
                            exactly_one=True)
            self.wcp_info = wcp_info[0]

    def findShifts(self):
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
        got_fpoffset = cosutil.findColumn(lamptab, "fpoffset")
        got_pixel_shift = cosutil.findColumn(lamptab, "fp_pixel_shift")
        if got_fpoffset:
            filter_lamp["fpoffset"] = self.info["fpoffset"]
        xc_range = self.wcp_info.field("xc_range")
        stepsize = self.wcp_info.field("stepsize")
        xd_range = self.wcp_info.field("xd_range")
        box = self.wcp_info.field("box")
        try:
            search_offset = self.wcp_info.field("search_offset")
        except KeyError:
            search_offset = 0.
        # initial_offset is used as the center of the search range, when
        # matching the spectrum to the template.
        if got_pixel_shift:
            initial_offset = search_offset
        else:
            initial_offset = self.info["fpoffset"] * stepsize + search_offset

        # Create a data quality array, and assign values from the bpixtab.
        if self.info["detector"] == "FUV":
            axis_height = FUV_Y
            axis_length = FUV_EXTENDED_X
        else:
            axis_height = NUV_Y
            axis_length = NUV_EXTENDED_X
        # create and populate a DQ array
        dq_array = np.zeros((axis_height,axis_length), dtype=np.int16)
        cosutil.updateDQArray(self.info, self.reffiles, dq_array,
                              {(0, 1024): [0., 0., 0., 0.]},
                              (0., 0.), -10, None)
        # weights from flat field or nonlinearity
        epsilon = self.events.field("epsilon")

        # Find the offsets in both axes, for each wavecal exposure.
        row = 0         # incremented in the second loop over segments
        cosutil.printMsg(
"  segment    cross-disp           dispersion direction", VERBOSE)
        cosutil.printMsg(
"            shift (locn)      shift err  [orig.]    FP   chi sq (n)", VERBOSE)
        cosutil.printMsg(
"  -------   -------------     -------------------------  ----------", VERBOSE)
        for n in range(self.numflash):
            (i0, i1) = ccos.range(self.time, self.lamp_on[n], self.lamp_off[n])

            # Find offset from nominal in cross-dispersion direction.
            (shift2, xd_shifts, xd_locn, lamp_is_on) = \
                wavecal.ttFindWavecalSpectrum(
                        self.xi[i0:i1], self.eta[i0:i1], self.dq[i0:i1],
                        self.info, xd_range, box, xtractab)
            if shift2 is None:
                shift2 = 0.
            self.shift2.append(shift2)
            if not self.lamp_is_on:
                self.lamp_is_on = lamp_is_on

            # Extract wavecal spectra from events table, and determine offset
            # in dispersion direction.
            x_offset = self.info["x_offset"]    # offset of lamptab in template
            sdqflags = self.info["sdqflags"]
            shift1 = {}                 # to be saved in an attribute
            chi_square = {}             # to be saved in an attribute
            n_deg_freedom = {}          # to be saved in an attribute
            save_spectra = {}   # save, but just within this loop
            save_net = {}       # save net counts, within this loop
            save_bkg = {}       # save background counts, within this loop
            save_templates = {}
            fp_pixel_shift = {}
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
                        self.user_shifts.getShifts((n+1, segment))
                spec_found[segment] = False     # initial value
                filter_1dx["segment"] = segment
                filter_lamp["segment"] = segment
                xtract_info = cosutil.getTable(xtractab, filter_1dx,
                                               at_least_one=True)
                snr_ff = 0.             # ignore error array
                axis = 1                # dispersion is along X axis
                dummy_exptime = 1.
                (N_i, ERR_i, ERR_LOW_i, VARIANCE_FLAT_i, VARIANCE_COUNTS_i, VARIANCE_BKG_i,
            GC_i, GCOUNTS_i, BK_i, DQ_i, DQ_WGT_i,
            DQ_ALL_i, LOWER_OUTER_INDEX_i, UPPER_OUTER_INDEX_i,
            LOWER_INNER_INDEX_i, UPPER_INNER_INDEX_i,
            ENCLOSED_FRACTION_i, AV_E_BKG_i,
            LOWER_OUTER_VALUE_i, LOWER_INNER_VALUE_i,
            UPPER_INNER_VALUE_i, UPPER_OUTER_VALUE_i) = \
                    extract.extractCorrtag(self.xi[i0:i1], self.eta[i0:i1],
                                self.dq[i0:i1], epsilon[i0:i1], dq_array,
                                self.ofd[1].header, segment, axis_length,
                                x_offset, sdqflags, snr_ff,
                                dummy_exptime, self.switches["backcorr"],
                                axis, xtract_info, 0., shift2)
                self.spectrum = GCOUNTS_i       # gross counts
                save_spectra[segment] = self.spectrum.copy()
                save_net[segment] = N_i         # net counts
                save_bkg[segment] = BK_i        # background counts
                lamp_found[segment] = True      # default
                lamp_info = cosutil.getTable(lamptab, filter_lamp,
                                             at_least_one=True)
                raw_template = lamp_info.field("intensity")[0]
                save_templates[segment] = cosutil.getTemplate(raw_template,
                                          x_offset, len(self.spectrum))
                if got_pixel_shift:
                    fp_pixel_shift[segment] = \
                            lamp_info.field("fp_pixel_shift")[0]
                else:
                    fp_pixel_shift[segment] = 0.
                if user_shift1 is not None or xd_shifts[segment] is not None:
                    spec_found[segment] = True
                    at_least_one_found = True

            if not at_least_one_found:
                # flag all segments/stripes as not found
                for segment in self.segment_list:
                    spec_found[segment] = False

            # find offset in dispersion direction
            fs1 = findshift1.Shift1(save_spectra, save_templates,
                                    self.info, self.reffiles,
                                    xc_range, fp_pixel_shift, initial_offset,
                                    spec_found)
            fs1.findShifts()

            # Print results, and save extracted spectra in lampflash table.
            for segment in self.segment_list:   # second loop over segments
                user_specified = False
                if self.user_shifts is not None:        # override shifts?
                    # note that flash number is one indexed
                    ((user_shift1, user_shift2), nfound) = \
                        self.user_shifts.getShifts((n+1, segment))
                    if user_shift1 is not None:
                        fs1.setShift1(segment, user_shift1)
                        user_specified = True
                shift1[segment] = fs1.getShift1(segment)
                measured_shift1 = fs1.getMeasuredShift1(segment)
                fp_pixel_shift_seg = fs1.getFpPixelShift(segment)
                chi_square[segment] = fs1.getChiSq(segment)
                n_deg_freedom[segment] = fs1.getNdf(segment)
                filter_disp["segment"] = segment
                if not lamp_found[segment]:
                    # no matching row in table
                    foundit = False
                    message = \
                        "%2d %4s skipped due to missing reference table row" \
                        % (n+1, segment)
                else:
                    foundit = fs1.getSpecFound(segment)
                    if xd_shifts[segment] is None:
                        # ttFindWavecalSpectrum couldn't find the spectrum
                        message = \
"%2d %4s      ---- (%5.1f) %9.1f %4.2f [%5.1f]        %6.1f (%d)  # not found in XD" \
                            % (n+1, segment,
                               xd_locn[segment], shift1[segment],
                               fs1.getScatter(segment), measured_shift1,
                               chi_square[segment], n_deg_freedom[segment])
                    else:
                        message = \
"%2d %4s %9.1f (%5.1f) %9.1f %4.2f [%5.1f] %6.1f  %6.1f (%d)" \
                            % (n+1, segment, xd_shifts[segment],
                               xd_locn[segment], shift1[segment],
                               fs1.getScatter(segment), measured_shift1,
                               fp_pixel_shift_seg,
                               chi_square[segment], n_deg_freedom[segment])
                        if not foundit:
                            message = message + "  # not found"
                    if user_specified:
                        message = message + "  # user-specified"
                cosutil.printMsg(message, VERBOSE)
                # copy to outflash table data
                if lamp_found[segment]:
                    self.saveSpectrum(disptab, filter_disp, n, row,
                                      save_spectra[segment],
                                      save_net[segment], save_bkg[segment],
                                      shift1[segment], xd_shifts[segment],
                                      foundit, chi_square[segment],
                                      n_deg_freedom[segment])
                else:
                    self.saveSpectrum(disptab, filter_disp,
                                      n, row, None, None, None, None, None,
                                      foundit, chi_square[segment],
                                      n_deg_freedom[segment])
                row += 1
            if self.info["detector"] == "NUV":
                cosutil.printMsg("%2d      avg %5.1f" % (n+1, shift2), VERBOSE)
            self.shift1.append(shift1)
            self.chi_square.append(chi_square)
            self.n_deg_freedom.append(n_deg_freedom)

    def saveSpectrum(self, disptab, filter, n, row,
                     spectrum, net_spectrum, bkg_spectrum,
                     shift1, shift2, spec_found,
                     chi_square, n_deg_freedom):
        """Copy the spectrum to the record array for the outflash table.

        Parameters
        ----------
        disptab: str
            Name of the dispersion relation table.

        filter: dictionary
            For extracting the row from the disptab.

        n: int
            Index of current lamp flash.

        row: int
            Row index (zero indexed) in output table.

        spectrum: array_like or None
            Gross counts for current segment or stripe (may be None).

        net_spectrum: array_like or None
            Net counts for current segment or stripe (may be None).

        bkg_spectrum: array_like or None
            Background counts for current segment or stripe (may be None).

        shift1: float
            Shift in dispersion direction.

        shift2: float or None
            Shift in cross-dispersion direction (may be None).

        spec_found: boolean
            Was the wavecal spectrum actually found?

        chi_square: float
            Chi square for the current flash.

        n_deg_freedom: int
            Number of degrees of freedom for current flash.
        """

        if self.ofd is None:
            return

        t0 = self.lamp_on[n]
        t1 = self.lamp_off[n]

        if spectrum is not None:
            pixel = np.arange(len(spectrum), dtype=np.float64)
            disp_rel = dispersion.Dispersion(disptab, filter)
            pixel -= shift1         # correct the wavelengths for the shift
            # Correct for any extra pixels in the dispersion direction.
            pixel -= self.info["x_offset"]
            wavelength = disp_rel.evalDisp(pixel)
            disp_rel.close()

        self.ofd[1].data.field("segment")[row] = filter["segment"]
        self.ofd[1].data.field("time")[row] = self.lamp_median[n]
        self.ofd[1].data.field("exptime")[row] = t1 - t0
        self.ofd[1].data.field("lamp_on")[row] = self.lamp_on[n]
        self.ofd[1].data.field("lamp_off")[row] = self.lamp_off[n]
        if spectrum is None:
            self.ofd[1].data.field("wavelength")[row][:] = 0.
        else:
            self.ofd[1].data.field("wavelength")[row] = wavelength
        exptime = t1 - t0
        if exptime <= 0.:
            exptime = 1.
        if spectrum is None:
            self.ofd[1].data.field("gross")[row][:] = 0.
            self.ofd[1].data.field("net")[row][:] = 0.
            self.ofd[1].data.field("background")[row][:] = 0.
            self.ofd[1].data.field("shift_disp")[row] = 0.
        else:
            self.ofd[1].data.field("gross")[row] = spectrum / exptime
            self.ofd[1].data.field("net")[row] = net_spectrum / exptime
            self.ofd[1].data.field("background")[row] = bkg_spectrum / exptime
            self.ofd[1].data.field("shift_disp")[row] = shift1
        if shift2 is None:
            self.ofd[1].data.field("shift_xdisp")[row] = 0.
        else:
            self.ofd[1].data.field("shift_xdisp")[row] = shift2
        self.ofd[1].data.field("spec_found")[row] = spec_found
        self.ofd[1].data.field("chi_square")[row] = chi_square
        self.ofd[1].data.field("n_deg_freedom")[row] = n_deg_freedom

    def applyCorrections(self):
        """Apply the shift1[a-c] and shift2[a-c] offsets."""

        # There will be at least one flash.
        for n in range(self.numflash):
            (i0, i1, extrapolate) = self.getInterval(n)
            self.shift2Corr(n, i0, i1, extrapolate)
            self.shift1Corr(n, i0, i1, extrapolate)

    def getInterval(self, n):
        """Get the slice for times between wavecal exposures n and n+1.

        Parameters
        ----------
        n: int
            Index of flash at left end of time interval; the valid range
            for n is from 0 to the number of flashes minus one, inclusive.

        Returns
        -------
        (i0, i1, extrapolate): tuple of int, int, boolean
            i0 and i1 are the indices in the time array (Python slice
            indices) over which to apply the shift based on flashes n and
            n+1 (unless n is the last), and extrapolate is True if
            n is the last flash.
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

        (i0, i1) = ccos.range(self.time, t0, t1)
        if extrapolate:
            i1 = len(self.time)

        return (i0, i1, extrapolate)

    def shift1Corr(self, n, i0, i1, extrapolate=False):
        """Correct the pixel coordinates in the dispersion direction.

        Parameters
        ----------
        n: int
            Apply shift1 for the nth time interval between wavecals.

        i0, i1: int
            [i0:i1] is the slice of event numbers to be corrected.

        extrapolate: boolean
            True if n is the last flash.
        """

        for segment in self.segment_list:

            # Restrict the correction to the applicable regions.
            shift_flags = np.zeros(i1 - i0, dtype=np.bool8)
            locn_list = self.regions[segment]
            for region in locn_list:
                if region[0] is None:
                    shift_flags |= np.where(
                                   self.eta[i0:i1] < region[1], True, False)
                elif region[1] is None:
                    shift_flags |= np.where(
                                   self.eta[i0:i1] >= region[0], True, False)
                else:
                    shift_flags |= np.logical_and(
                                   self.eta[i0:i1] >= region[0],
                                   self.eta[i0:i1] < region[1])

            shift1_zero = self.shift1[n][segment]
            if extrapolate:
                self.xi_corr[i0:i1] = np.where(shift_flags,
                        self.xi_corr[i0:i1] - shift1_zero,
                        self.xi_corr[i0:i1])
            else:
                # Note that i0 & i1 do not necessarily correspond to t0 and t1.
                t0 = self.lamp_median[n]
                t1 = self.lamp_median[n+1]
                if t1 <= t0:
                    slope = 0.
                else:
                    slope =(self.shift1[n+1][segment] -
                            self.shift1[n][segment]) / (t1 - t0)
                self.xi_corr[i0:i1] = np.where(shift_flags,
                    self.xi_corr[i0:i1] -
                        ((self.time[i0:i1] - t0) * slope + shift1_zero),
                    self.xi_corr[i0:i1])

    def findFlash(self, output=None, update=True):
        """Find tagflash wavecals in science data.

        Parameters
        ----------
        output: str or None
            If specified, write array of count rates to this file
            (for testing).

        update: boolean
            If True, keywords in input file will be updated.
        """

        detector = self.info["detector"]
        life_adj_offset = self.info["life_adj_offset"]  # usually 0.


        xtractab = self.reffiles["xtractab"]

        filter = {"opt_elem": self.info["opt_elem"],
                  "cenwave": self.info["cenwave"],
                  "aperture": "WCA"}

        # src_low to src_high is a range for getting the source count rates,
        # which is done by ccos.getbkgcounts.
        if detector == "FUV":
            filter["segment"] = self.info["segment"]
            xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
            b_spec = xtract_info.field("b_spec")[0] + life_adj_offset
            height = xtract_info.field("height")[0]
            src_low = int(round(b_spec - height))
            src_high = int(round(b_spec + height))
            eta = self.events.field("ycorr")
        elif self.info["obstype"] == "IMAGING":
            b_spec = Y0 + life_adj_offset
            height = 2 * DY
            src_low = int(round(b_spec - height))
            src_high = NUV_Y - 1
            eta = self.events.field("rawy")
        else:
            # For NUV use only the data for stripe A.
            filter["segment"] = "NUVA"
            xtract_info = cosutil.getTable(xtractab, filter, exactly_one=True)
            b_spec = xtract_info.field("b_spec")[0] + life_adj_offset
            height = xtract_info.field("height")[0]
            src_low = int(round(b_spec - height))
            # Include source counts for wavecal stripes NUVB and NUVC as well.
            src_high = NUV_Y - 1
            eta = self.events.field("rawy")

        # Dummy values, so no background counts will be found.
        bkg1_low = -10
        bkg1_high = -20
        bkg2_low = -10
        bkg2_high = -20
        bkgsf = 0.

        # Get time as a single precision array, for ccos.getstartstop.
        time = cosutil.getColCopy(data=self.events, column="time")
        dq = np.zeros(len(time), dtype=np.int16)
        nbins = int(math.ceil((time[-1] - time[0]) / self.delta_t))
        istart = np.zeros(nbins, dtype=np.int32)
        istop = np.zeros(nbins, dtype=np.int32)
        src_counts = np.zeros(nbins, dtype=np.int32)
        bkg_counts = np.zeros(nbins, dtype=np.int32)

        ccos.getstartstop(time, istart, istop, self.delta_t)
        ccos.getbkgcounts(eta, dq, istart, istop,
                          bkg_counts, src_counts,
                          bkg1_low, bkg1_high, bkg2_low, bkg2_high,
                          src_low, src_high, bkgsf)
        del dq, bkg_counts

        # Convert to count rate.
        self.src_counts = src_counts.astype(np.float64) / self.delta_t
        del src_counts

        if output is not None:
            ofd = open(output, "w")
            for val in self.src_counts:
                ofd.write("%g\n" % val)
            ofd.close()

        (hist, step) = self.makeHistogram()

        if hist is None:
            self.numflash = 0
        else:
            cutoff = self.findCutoff(hist, step)
            self.findLampOn(cutoff, time[0])
            self.findLampMedian()
        self.lamp_is_on = (self.numflash > 0)

        if cosutil.checkVerbosity(VERBOSE):
            self.printInfo()

        if update:
            self.updateHeader()

    def makeHistogram(self):
        """Make a histogram of src_counts.

        The function value is (hist, step), the histogram and step size.
        If the maximum value in src_counts is less than 20, the count
        rate is so low that it would be difficult to determine when the
        lamp turned on or off, and (None, None) will be returned to
        indicate this case.

        Returns
        -------
        tuple of array_like and float
            The histogram (array) and step size (float).
        """

        maxval = np.maximum.reduce(self.src_counts)
        if maxval <= MIN_COUNT_RATE:
            return (None, None)

        i_maxval = int(round(maxval))
        nbins = max(20, len(self.src_counts) // 100)
        step = maxval / float(nbins)
        hist = np.zeros(nbins, dtype=np.int32)
        for src in self.src_counts:
            i = int(src / step)
            if i < nbins:               # ignore max value
                hist[i] += 1
        cosutil.printMsg("tagflash histogram = %s" % repr(hist),
                         VERY_VERBOSE)
        cosutil.printMsg("step size for histogram = %g" % step, VERY_VERBOSE)

        return (hist, step)

    def findCutoff(self, hist, step):
        """Find the count rate above which the lamp is probably on.

        The histogram is likely to have two peaks, one close to zero that
        is due to dark counts and one close to the maximum count rate
        that is due to the wavecal lamp being on.  This function searches
        for the second one by looking for the maximum of the upper half
        (higher count rates) of the histogram.  Then the cutoff is set
        to half the count rate corresponding to that peak.

        Parameters
        ----------
        hist: array_like
            Histogram of count rates src_counts.  hist[i] is the
            number of elements of src_counts with count rates between
            i*step and (i+1)*step.

        step: float
            Step size for histogram, i.e. change in count rate
            from hist[i] to hist[i+1].

        Returns
        -------
        float
            The cutoff count rate.
        """

        n = len(hist)
        istart = n // 2
        index = istart
        maxhist = 0
        for i in range(n // 2, n):
            if hist[i] > maxhist:
                maxhist = hist[i]
                index = i

        cutoff = (index / 2.) * step
        cosutil.printMsg("tagflash cutoff = %.2f counts/s" % cutoff,
                         VERY_VERBOSE)

        return cutoff

    def findLampOn(self, cutoff, t0):
        """Find the actual times when the lamps turned on and off.

        The nominal lamp turn-on and turn-off times were gotten from
        the EVENTS header and saved as self.lamp_on and self.lamp_off.
        This function gets the actual times, constrained to be within
        the time intervals specified in the header.  The attributes
        lamp_on and lamp_off are then updated.  The number of flashes,
        the numflash attribute, can also be modified.

        Parameters
        ----------
        cutoff: float
            A count rate (in src_counts array) greater than this
            indicates that the wavecal lamp was on.

        t0: float
            Time of first photon event (first element of TIME column).
        """

        nbins = len(self.src_counts)

        # Find the turn-on and turn-off time of each flash, starting with
        # the nominal times in self.lamp_on and self.lamp_off.
        lamp_on = []
        lamp_off = []

        for i in range(self.numflash):
            lamp_is_on = False
            lamp_on_i = self.lamp_on[i] - self.buffer_on
            lamp_off_i = self.lamp_off[i] + self.buffer_off
            (k_on, k_off) = self.getIndices(nbins, t0, lamp_on_i, lamp_off_i)
            if k_on is None:
                continue
            # search for the actual lamp turn-on time
            for k in range(k_on, k_off+1):
                if self.src_counts[k] > cutoff:
                    lamp_is_on = True
                    t = t0 + k * self.delta_t
                    lamp_on.append(t)
                    break
            if not lamp_is_on:
                continue
            # search for the actual lamp turn-off time
            for k in range(k_off, k_on-1, -1):
                if self.src_counts[k] > cutoff:
                    t = t0 + (k+1) * self.delta_t
                    t = min(t, self.time[-1])
                    lamp_off.append(t)
                    break

        if len(lamp_on) != len(lamp_off):
            raise RuntimeError("Internal error:  len(lamp_on) = %d, "
                               "len(lamp_off) = %d" % \
                               (len(lamp_on), len(lamp_off)))

        self.numflash = len(lamp_on)            # the actual number of flashes

        self.lamp_on = lamp_on
        self.lamp_off = lamp_off

    def findLampMedian(self):
        """Find the median time of each wavecal flash."""

        lamp_median = []
        for i in range(self.numflash):
            (i0, i1) = ccos.range(self.time, self.lamp_on[i], self.lamp_off[i])
            index = (i0 + i1) // 2
            lamp_median.append(self.time[index])

        self.lamp_median = lamp_median

    def getIndices(self, nbins, t0, lamp_on_i, lamp_off_i):
        """Compute indices in src_counts corresponding to given times.

        Parameters
        ----------
        nbins: int
            Number of elements in src_counts array.

        t0: float
            Time of first photon event.

        lamp_on_i: float
            One element of lamp_on array (minus buffer_on).

        lamp_off_i: float
            One element of lamp_off array (plus buffer_off).

        Returns
        -------
        tuple of two ints
            The indices corresponding to lamp_on_i and lamp_off_i;
            both values will be None if these times are outside the
            interval (t0, t0+delta_t).
        """

        k_on = (lamp_on_i - t0) / self.delta_t
        k_off = (lamp_off_i - t0) / self.delta_t
        k_on = int(math.floor(k_on))
        k_off = int(math.ceil(k_off))
        if k_on >= nbins or k_off < 0:
            k_on = None
            k_off = None
        else:
            k_on = max(k_on, 0)
            k_off = min(k_off, nbins-1)

        return (k_on, k_off)

    def printInfo(self):
        """Print time info for each wavecal flash."""

        if self.numflash < 1:
            return

        cosutil.printMsg("lamp on, off, duration, median time:")
        for i in range(self.numflash):
            cosutil.printMsg("%d:  %.1f  %.1f  %.1f  %.1f" %
                             (i+1, self.lamp_on[i], self.lamp_off[i],
                             self.lamp_off[i] - self.lamp_on[i],
                             self.lamp_median[i]))

    def updateHeader(self):
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

        self.hdr["NUMFLASH"] = self.numflash

        for i in range(self.numflash):
            keyword = "LMP_ON%d" % (i+1)
            if keyword not in self.hdr:
                break
            self.hdr[keyword] = self.lamp_on[i]
            keyword = "LMPOFF%d" % (i+1)
            self.hdr[keyword] = self.lamp_off[i]
            keyword = "LMPDUR%d" % (i+1)
            self.hdr[keyword] = self.lamp_off[i] - self.lamp_on[i]
            keyword = "LMPMED%d" % (i+1)
            self.hdr[keyword] = self.lamp_median[i]

    def avgShift(self):
        """Compute the shift1 and shift2 offsets averaged over time.

        Returns
        -------
        tuple of two dictionaries
            The key is the segment or stripe name.  The dictionaries
            contain the shifts in the dispersion and cross-dispersion
            directions averaged over the exposure.
        """

        avg_dx = {}
        avg_dy = {}
        if self.numflash < 1 or self.info["exptime"] <= 0.:
            for segment in self.segment_list:
                avg_dx[segment] = 0.
                avg_dy[segment] = 0.
            return (avg_dx, avg_dy)

        time = self.time
        # Note that this is the entire time interval, i.e. including any
        # bad time intervals.
        time_range = time[-1] - time[0]
        if time_range <= 0.:
            time_range = 1.

        for segment in self.segment_list:
            t_prev = time[0]
            shift1_prev = self.shift1[0][segment]
            shift2_prev = self.shift2[0]
            sum_t = 0.
            sum_shift1 = 0.
            sum_shift2 = 0.
            for n in range(self.numflash):
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
            avg_dx[segment] = sum_shift1 / time_range
            avg_dy[segment] = sum_shift2 / time_range

        return (avg_dx, avg_dy)

    def setShiftKeywords(self, avg_dx, avg_dy):
        """Assign values to the shift keywords.

        Keywords SHIFT1[ABC] will be set to the average offset in the
        dispersion direction, SHIFT2[ABC] to the average offset in the
        cross-dispersion direction, and DPIXEL1[ABC] to:
            XFULL - (XFULL rounded to an integer)
        Keyword values will be set in the corrtag header and in the output
        lampflash header.

        LAMPUSED may also be updated.  This keyword can be NONE in the raw
        header if the info for the support file was taken at times when the
        lamp happened to be off.  If the lamp really was on, the keyword
        will be set to the value of LAMPPLAN.

        Parameters
        ----------
        avg_dx: dictionary
            The average shift in the dispersion direction; the key is the
            segment or stripe name.

        avg_dy: dictionary
            The average shift in the cross-dispersion direction; the key is
            the segment or stripe name.
        """

        for segment in self.segment_list:
            key = "SHIFT1" + segment[-1]
            value = round(avg_dx[segment], 4)
            self.hdr[key] = value                               # corrtag
            self.ofd[1].header[key] = value                     # lampflash
            key = "SHIFT2" + segment[-1]
            value = round(avg_dy[segment], 4)
            self.hdr[key] = value
            self.ofd[1].header[key] = value
            sum_chisq = 0.
            sum_ndf = 0
            for n in range(self.numflash):
                sum_chisq += self.chi_square[n][segment]
                sum_ndf += (self.n_deg_freedom[n][segment] + 1)
            sum_ndf -= 1
            if self.override_segment_B:
                sum_ndf = 0
            key = "chi_sq_" + segment[-1]
            self.hdr[key] = round(sum_chisq, 1)
            self.ofd[1].header[key] = round(sum_chisq, 1)
            key = "ndf_" + segment[-1]
            self.hdr[key] = sum_ndf
            self.ofd[1].header[key] = sum_ndf

            # use self.regions for dpixel1[abc]
            shift_flags = np.zeros(len(self.eta), dtype=np.bool8)
            locn_list = self.regions[segment]
            # if NUV, take the region for the PSA (lower pixel numbers)
            region = locn_list[0]
            if region[0] is None:
                shift_flags |= np.where(self.eta < region[1], True, False)
            elif region[1] is None:
                shift_flags |= np.where(self.eta >= region[0], True, False)
            else:
                shift_flags |= np.logical_and(self.eta >= region[0],
                                              self.eta < region[1])
            xi = self.xi_corr[shift_flags]      # copy out the relevant subset
            if len(xi) > 0:
                xi_diff = xi - np.around(xi)
                dpixel1 = xi_diff.mean(dtype=np.float64)
                value = round(dpixel1, 4)
            else:
                value = 0.
            key = "DPIXEL1" + segment[-1]
            self.hdr[key] = value
            self.ofd[1].header[key] = value

        if self.override_segment_B:
            lampused = self.ofd[0].header.get("lampused", "missing")
            if lampused != "missing":
                self.phdr["lampused"] = lampused
        else:
            lampused = self.phdr.get("lampused", "missing")
            lampplan = self.phdr.get("lampplan", "missing")
            if self.lamp_is_on and lampused == "NONE":
                if lampplan == "missing":
                    cosutil.printWarning("The wavecal lamp was on, " \
                                         "but LAMPUSED = %s and LAMPPLAN is " \
                                         "missing." % lampused, level=VERBOSE)
                else:
                    cosutil.printMsg("LAMPUSED = %s, which is incorrect; " \
                                     "the value will be reset to %s." % \
                                     (lampused, lampplan), level=VERBOSE)
                    self.phdr["lampused"] = lampplan
                    self.ofd[0].header["lampused"] = lampplan
            if not self.lamp_is_on:
                cosutil.printWarning("The wavecal lamp was not on " \
                                     "for tagflash data.", level=VERBOSE)
                if lampused != "NONE":
                    cosutil.printMsg("LAMPUSED = %s, and it will be reset " \
                                     "to NONE." % lampused, level=VERBOSE)
                    self.phdr["lampused"] = "NONE"
                    self.ofd[0].header["lampused"] = "NONE"

    def shift1VsTime(self):
        """Interpolate shift1 at one-second intervals.

        Returns
        -------
        (tl_time, shift1_vs_time): tuple of two array like
            tl_time is the array of times at one-second intervals, for
            the timeline table.  shift1_vs_time is the array of
            corresponding values of shift1a or shift1b, or None if there
            were no flashes or if the exposure time is zero.
        """

        if self.numflash < 1 or self.info["exptime"] <= 0.:
            return (cosutil.timelineTimes(None, 0.), None)

        first_time = self.time[0]
        tl_time = cosutil.timelineTimes(self.time[0], self.time[-1], dt=1.)
        nbins = len(tl_time)
        shift1_vs_time = np.zeros(nbins, dtype=np.float32)

        if self.segment_list[0][0] == "F":
            segment = self.segment_list[0]
        elif "NUVB" in self.segment_list:
            segment = "NUVB"
        else:
            segment = self.segment_list[0]

        for n in range(self.numflash):

            t0 = self.lamp_median[n]
            if n == self.numflash - 1:
                t1 = tl_time[-1]
            else:
                t1 = self.lamp_median[n+1]

            shift1_zero = self.shift1[n][segment]
            t0 = self.lamp_median[n]
            i0 = int(round(t0 - first_time))
            i0 = max(i0, 0)

            if self.numflash == 1:
                shift1_vs_time[:] = shift1_zero
            elif n == self.numflash - 1:
                # last flash; extrapolate with zero slope
                shift1_vs_time[i0:] = shift1_zero
            else:
                if n == 0:      # extrapolate with slope of first interval
                    i0 = 0
                t1 = self.lamp_median[n+1]
                if t1 <= t0:
                    slope = 0.
                else:
                    slope = (self.shift1[n+1][segment] -
                             self.shift1[n][segment]) / (t1 - t0)
                i1 = int(round(t1 - first_time))
                i1 = min(i1, nbins-1)
                shift1_vs_time[i0:i1] = \
                        ((tl_time[i0:i1] - t0) * slope + shift1_zero)

        return (tl_time, shift1_vs_time)

class FUVConcurrentWavecal(ConcurrentWavecal):

    def __init__(self, events, outflash, shift_file,
                 info, switches, reffiles, phdr, hdr):

        ConcurrentWavecal.__init__(self, events, outflash, shift_file,
                                   info, switches, reffiles, phdr, hdr)
        self.xi  = events.field("XDOPP")
        self.eta = events.field("YCORR")
        self.dq  = events.field("DQ")
        self.xi_corr  = events.field("XFULL")
        self.eta_corr = events.field("YFULL")
        self.spectrum = np.zeros(FUV_EXTENDED_X, dtype=np.float64)
        self.segment_list = [info["segment"]]
        self.override_segment_B = cosutil.checkForNoWavecalData(
                        info["opt_elem"], info["cenwave"], info["segment"],
                        reffiles["lamptab"])
        if self.override_segment_B:
            if outflash.endswith("_b.fits"):
                index = outflash.rfind("b.fits")
                self.lampflash_a = outflash[0:index] + "a.fits"
                if os.access(self.lampflash_a, os.R_OK):
                    cosutil.printMsg("Info:  No wavecal signal for FUVB, "
                                     " so info will be copied from FUVA.")
                    self.segment_A_present = True
                else:
                    cosutil.printMsg("Info:  No wavecal signal for FUVB,"
                                     " but the file")
                    cosutil.printContinuation(
                        "%s for segment A does not exist,"
                        % self.lampflash_a)
                    cosutil.printContinuation(
                        "so wavecal shifts will be set to the default.")
                    self.segment_A_present = False
            else:
                cosutil.printWarning("No wavecal signal, "
                                     "but don't understand the name")
                cosutil.printContinuation("outflash = %s," % outflash)
                cosutil.printContinuation(
                        "so wavecal shifts will be set to the default.")
                self.segment_A_present = False

        # Copy xi and eta to the columns for corrected values.
        self.copyColumns()

        # The shift1 offset should be applied only within this region.
        (b_low, b_high, b_left, b_right) = \
                cosutil.activeArea(info["segment"], reffiles["brftab"])
        self.regions[info["segment"]] = [(b_low, b_high)]

    def copySegAtoSegB(self):
        """Copy lampflash_a.fits to lampflash_b.fits.

        This method is called for the case of FUVB where there is no
        wavecal data but there is data for segment A.  The lampflash file
        for segment A will be copied to <rootname>_lampflash_b.fits.
        The latter file will be opened (with pyfits), and the "FUVA" in
        the SEGMENT column will be changed to "FUVB"; the file should
        be closed later by calling closeOutFlash().
        """

        # copy rootname_lampflash_a.fits to rootname_lampflash_b.fits
        cosutil.copyFile(self.lampflash_a, self.outflash)

        self.ofd = fits.open(self.outflash, mode="update")
        self.ofd[0].header["segment"] = self.segment_list[0]
        if self.ofd[1].data is None or len(self.ofd[1].data) == 0:
            self.numflash = 0
            cosutil.printWarning("No data in lampflash table.")
            self.phdr["wavecorr"] = "SKIPPED"
            return

        # A_shift1 is the shift1 value for FUVA, and A_spec_found is used
        # for printing "not found in FUVA" if the shift was not found.
        A_shift1 = self.ofd[1].data.field("shift_disp").copy()
        A_shift1[:] = A_shift1 + DELTA_SHIFT1
        A_spec_found = self.ofd[1].data.field("spec_found").copy()

        nrows = len(self.ofd[1].data)
        # This assumes that lampflash_a contains data for only one segment.
        self.numflash = nrows
        self.lamp_median = self.ofd[1].data.field("time")
        self.shift2 = self.ofd[1].data.field("shift_xdisp")
        self.spec_found = self.ofd[1].data.field("spec_found")

        segment_column = self.ofd[1].data.field("segment")
        shift1 = self.ofd[1].data.field("shift_disp")
        chi_square = self.ofd[1].data.field("chi_square")
        n_deg_freedom = self.ofd[1].data.field("n_deg_freedom")

        # Create lists of the appropriate length.
        self.shift1 = [0] * nrows
        self.chi_square = [0] * nrows
        self.n_deg_freedom = [0] * nrows
        # The elements of the lists are dictionaries.
        segment = self.segment_list[0]
        for i in range(len(shift1)):
            self.shift1[i] = {segment: A_shift1[i]}
            self.chi_square[i] = {segment: 0.}
            self.n_deg_freedom[i] = {segment: 0}

        segment_column[:] = self.segment_list[0]    # replace FUVA with FUVB
        self.shift2[:] += DELTA_SHIFT2
        self.spec_found[:] = False

        # Override shifts if specified in shift_file, and print shift info.
        row = 0
        cosutil.printMsg(
"  segment    cross-disp        dispersion direction", VERBOSE)
        cosutil.printMsg(
"            shift (locn)      shift  [orig.]  chi sq (n)", VERBOSE)
        cosutil.printMsg(
"  -------   -------------     --------------------------", VERBOSE)
        for n in range(self.numflash):
            for segment in self.segment_list:
                if segment != segment_column[row]:
                    cosutil.printWarning("Out of synch at row %d in %s!" % \
                                         (row, self.outflash))
                    return
                user_specified = False
                if self.user_shifts is not None:        # override shifts?
                    # note that flash number is one indexed
                    ((user_shift1, user_shift2), nfound) = \
                        self.user_shifts.getShifts((n+1, segment))
                    if user_shift1 is None:
                        user_specified = False
                    else:
                        user_specified = True
                        self.shift1[row] = {segment: user_shift1}
                message = "%2d %4s %9.1f (-999) %9.1f [%5.1f]  %6.1f (%d)" \
                            % (n+1, segment, self.shift2[row],
                               self.shift1[row][segment], A_shift1[row], 0., 0)
                if user_specified:
                    message = message + "  # user-specified"
                elif not A_spec_found[row]:
                    message = message + "  # not found in FUVA"
                else:
                    message = message + "  # based on FUVA value"
                cosutil.printMsg(message, VERBOSE)
                row += 1

        self.phdr["wavecorr"] = "COMPLETE"

    def setSegBtoZero(self):
        """Set the shifts to zero (no segment A data to copy)."""

        cosutil.printMsg(
"  segment    cross-disp           dispersion direction", VERBOSE)
        cosutil.printMsg(
"            shift (locn)      shift err  [orig.]    FP   chi sq (n)", VERBOSE)
        cosutil.printMsg(
"  -------   -------------     -------------------------  ----------", VERBOSE)
        n = 0
        segment = self.segment_list[0]
        user_specified = False          # may be reset below
        self.numflash = 1               # there might not have been any
        # Create one-element lists.
        self.lamp_on = [0.]
        self.lamp_off = [0.]
        self.lamp_duration = [0.]
        self.lamp_median = [0.]
        self.shift2 = [0.]
        self.spec_found = [False]
        self.chi_square = [{segment: 0.}]
        self.n_deg_freedom = [{segment: 0}]

        # get fp_pixel_shift for shift1
        lamptab = self.reffiles["lamptab"]
        filter_lamp = {"opt_elem": self.info["opt_elem"],
                       "cenwave": self.info["cenwave"],
                       "segment": segment}
        got_pixel_shift = cosutil.findColumn(lamptab, "fp_pixel_shift")
        if got_pixel_shift:
            filter_lamp["fpoffset"] = self.info["fpoffset"]
        lamp_info = cosutil.getTable(lamptab, filter_lamp)
        if lamp_info is not None and got_pixel_shift:
            fp_pixel_shift = lamp_info.field("fp_pixel_shift")[0]
        else:
            fp_pixel_shift = 0.
        # this may be replaced below, if the user specified the value
        self.shift1 = [{segment: fp_pixel_shift}]

        if self.user_shifts is not None:
            # Check for a user-supplied value for shift1.
            # Note that flash number is one indexed.
            ((user_shift1, user_shift2), nfound) = \
                self.user_shifts.getShifts((n+1, segment))
            if user_shift1 is None:
                user_specified = False
            else:
                user_specified = True
                self.shift1 = [{segment: user_shift1}]
        message = \
"%2d %4s %9.1f (-9999) %9.1f 0.00 [  0.0] %6.1f     0.0 (0)" \
                    % (n+1, segment, self.shift2[n],
                       self.shift1[n][segment], fp_pixel_shift)
        if user_specified:
            message = message + "  # user-specified"
            # Not skipped because the user specified the shift.
            self.wavecorr = "COMPLETE"
        else:
            message = message + "  # set to 0"
            self.wavecorr = "SKIPPED"
        cosutil.printMsg(message, VERBOSE)

    def miscSegB(self):
        """Miscellaneous stuff for FUVB, with no segment A data."""

        segment = self.segment_list[0]

        # compute wavelengths
        disptab = self.reffiles["disptab"]
        filter_disp = {"opt_elem": self.info["opt_elem"],
                       "cenwave": self.info["cenwave"],
                       "fpoffset": self.info["fpoffset"],
                       "segment": segment,
                       "aperture": "WCA"}
        pixel = np.arange(len(self.spectrum), dtype=np.float64)
        disp_rel = dispersion.Dispersion(disptab, filter_disp)
        wavelength = disp_rel.evalDisp(pixel)
        disp_rel.close()

        # note that the pyfits HDUList object self.ofd must exist at this time
        segment_col = self.ofd[1].data.field("segment")
        wavelength_col = self.ofd[1].data.field("wavelength")
        shift_disp_col = self.ofd[1].data.field("shift_disp")
        spec_found_col = self.ofd[1].data.field("spec_found")
        n = 0
        segment_col[n] = segment
        wavelength_col[n][:] = wavelength
        shift_disp_col[n] = self.shift1[n][segment]
        spec_found_col[n] = False

        avg_dx = {segment: self.shift1[0][segment]}
        avg_dy = {segment: 0.}
        return (avg_dx, avg_dy)

    def closeOutFlash(self):
        """Close the lampflash_b.fits file."""

        self.ofd.close()
        self.ofd = None

    def shift2Corr(self, n, i0, i1, extrapolate=False):
        """Correct the pixel coordinates in the cross-dispersion direction.

        The difference between this version and the one for NUV is that this
        one limits the shift to the active area.

        Parameters
        ----------
        n: int
            Apply shift2 for the nth time interval between wavecals.

        i0: int
            [i0:i1] is the slice of event numbers to be corrected.

        i1: int
            [i0:i1] is the slice of event numbers to be corrected.

        extrapolate: boolean
            True if n is the last flash.
        """

        # Restrict the correction to the applicable region.  Note that the
        # limits of the region (the active area) are not adjusted by shift2.
        shift_flags = np.zeros(i1 - i0, dtype=np.bool8)
        region = self.regions[self.segment_list[0]][0]
        shift_flags |= np.logical_and(self.eta[i0:i1] >= region[0],
                                      self.eta[i0:i1] <= region[1])

        shift2_zero = self.shift2[n]
        if extrapolate:
            self.eta_corr[i0:i1] = np.where(shift_flags,
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
            self.eta_corr[i0:i1] = np.where(shift_flags,
                        self.eta_corr[i0:i1] -
                            ((self.time[i0:i1] - t0) * slope + shift2_zero),
                        self.eta_corr[i0:i1])

class NUVConcurrentWavecal(ConcurrentWavecal):

    def __init__(self, events, outflash, shift_file,
                 info, switches, reffiles, phdr, hdr):

        ConcurrentWavecal.__init__(self, events, outflash, shift_file,
                                   info, switches, reffiles, phdr, hdr)
        self.xi  = events.field("XDOPP")
        self.eta = events.field("RAWY")
        self.dq  = events.field("DQ")
        self.xi_corr  = events.field("XFULL")
        self.eta_corr = events.field("YFULL")
        self.spectrum = np.zeros(NUV_EXTENDED_X, dtype=np.float64)
        self.segment_list = ["NUVA", "NUVB", "NUVC"]

        # Copy xi and eta to the columns for corrected values.
        self.copyColumns()

        # The shift1 offset should be applied only within this region.
        self.regions = self.setRegions()

    def setRegions(self):
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
        middle = float(NUV_X) / 2.
        locations = []
        for segment in self.segment_list:

            filter["segment"] = segment

            filter["aperture"] = "WCA"
            xtract_info = cosutil.getTable(self.reffiles["xtractab"], filter,
                                           at_least_one=True)
            # Note that this does not include life_adj_offset; see the comment
            # in the doc string about shift2.
            b_spec = xtract_info.field("b_spec")[0] + \
                     xtract_info.field("slope")[0] * middle
            locations.append((segment, b_spec))

            filter["aperture"] = "PSA"
            xtract_info = cosutil.getTable(self.reffiles["xtractab"], filter,
                                           at_least_one=True)
            b_spec = xtract_info.field("b_spec")[0] + \
                     xtract_info.field("slope")[0] * middle
            locations.append((segment, b_spec))

        locations.sort(key=r_key)       # sort on b_spec, regardless of segment
        len_locn = len(locations)

        # intervals will be the same length as locations.  Each interval
        # [first,last] is the slice over which shift1[a-c] should be
        # applied.  There should be six such intervals, one for each
        # stripe, and for apertures PSA and WCA.
        # Use None for the lower and upper cutoffs so every event will be
        # included.
        intervals = []
        first = None                    # no lower cutoff
        for i in range(len_locn):
            segment = locations[i][0]
            locn = locations[i][1]
            if i == len_locn - 1:
                last = None             # no upper cutoff
            else:
                next_locn = locations[i+1][1]
                # midpoint between adjacent b_spec values
                last = int(round((locn + next_locn) / 2.))
            intervals.append([segment, [first,last]])
            first = last

        regions = {}
        for segment in self.segment_list:
            locn_list = []
            for i in range(len_locn):
                if segment == intervals[i][0]:
                    locn_list.append(intervals[i][1])
            regions[segment] = locn_list

        return regions

    def shift2Corr(self, n, i0, i1, extrapolate=False):
        """Correct the pixel coordinates in the cross-dispersion direction.

        Parameters
        ----------
        n: int
            Apply shift2 for the nth time interval between wavecals.

        i0: int
            [i0:i1] is the slice of event numbers to be corrected.

        i1: int
            [i0:i1] is the slice of event numbers to be corrected.

        extrapolate: boolean
            True if n is the last flash.
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

class NUVImagingWavecal(ConcurrentWavecal):

    def __init__(self, events, outflash, shift_file,
                 info, switches, reffiles, phdr, hdr):

        info_copy = copy.deepcopy(info)
        info_copy["cenwave"] = 0

        ConcurrentWavecal.__init__(self, events, outflash, shift_file,
                                   info_copy, switches, reffiles, phdr, hdr)
        self.xi  = events.field("XDOPP")
        self.eta = events.field("RAWY")
        self.dq  = events.field("DQ")
        self.xi_corr  = events.field("XFULL")
        self.eta_corr = events.field("YFULL")
        self.spectrum = np.zeros(NUV_X, dtype=np.float64)
        self.segment_list = ["N/A"]

        # Copy xi and eta to the columns for corrected values.
        self.copyColumns()

        # The shift1 offset should be applied over the entire detector.
        self.regions = self.setRegions()

    def setRegions(self):
        """Shift1 should be applied over the full detector.

        The function value is a dictionary with one entry.  The key is the
        "stripe" name "NUVA", and the value (there's only one) is a list of
        lists of the interval (the full detector) over which the shift in the
        dispersion direction (shift1) should be applied.
        """

        regions = {}
        for segment in self.segment_list:
            regions[segment] = [[0, NUV_Y]]

        return regions

    def shift2Corr(self, n, i0, i1, extrapolate=False):
        """Correct the pixel coordinates in the cross-dispersion direction.

        n: int
            Apply shift2 for the nth time interval between wavecals.

        i0: int
            [i0:i1] is the slice of event numbers to be corrected.

        i1: int
            [i0:i1] is the slice of event numbers to be corrected.

        extrapolate: boolean
            True if n is the last flash.
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

    def findShifts(self):
        """For each wavecal flash, find the shift in each axis."""

        global X0, Y0, DX, DY

        fpoffset = self.info["fpoffset"]
        segment = self.segment_list[0]          # don't really need segment

        # Find the offsets in both axes, for each wavecal exposure.
        cosutil.printMsg(
"         Y direction       X direction", VERBOSE)
        cosutil.printMsg(
"        shift (locn)      shift (locn)", VERBOSE)
        cosutil.printMsg(
"        ------------      ------------", VERBOSE)
        row = 0
        for n in range(self.numflash):
            (i0, i1) = ccos.range(self.time, self.lamp_on[n], self.lamp_off[n])

            shift1 = {}                 # to be saved in an attribute
            chi_square = {}             # to be saved in an attribute
            n_deg_freedom = {}          # to be saved in an attribute
            # First find the median of the xi and eta positions.
            nelem = i1 - i0
            index_x = self.xi[i0:i1].argsort()
            index_y = self.eta[i0:i1].argsort()
            x_median = self.xi[i0+index_x[nelem//2]]
            y_median = self.eta[i0+index_y[nelem//2]]

            # Now find the mean values of xi and of eta, but restrict
            # the range to the median plus or minus DX or DY.
            select = np.zeros(len(self.xi), dtype=np.bool8)
            select[i0:i1] = 1
            select = np.where(self.xi < x_median-DX, False, select)
            select = np.where(self.xi > x_median+DX, False, select)
            select = np.where(self.eta < y_median-DY, False, select)
            select = np.where(self.eta > y_median+DY, False, select)
            select = select.astype(np.bool8)

            x = self.xi[select].mean(dtype=np.float64)
            y = self.eta[select].mean(dtype=np.float64)
            shift1[segment] = x - X0
            shift2 = y - Y0
            chi_square[segment] = 0.            # not used
            n_deg_freedom[segment] = 0

            message = "%2d %9.1f (%5.1f) %9.1f (%5.1f)" \
                            % (n+1, shift2, y, shift1[segment], x)
            user_specified = False
            if self.user_shifts is not None:            # override shifts?
                # note that flash number is one indexed
                ((user_shift1, user_shift2), nfound) = \
                    self.user_shifts.getShifts((n+1, segment))
                if user_shift1 is not None:
                    shift1[segment] = user_shift1
                    user_specified = True
                if user_shift2 is not None:
                    shift2 = user_shift2
                    user_specified = True
            if user_specified:
                message = message + "  # user-specified"
            cosutil.printMsg(message, VERBOSE)
            # copy to outflash table data
            self.saveSpectrum(n, row, shift1[segment], shift2, True)
            self.shift1.append(shift1)
            self.shift2.append(shift2)
            self.chi_square.append(chi_square)
            self.n_deg_freedom.append(n_deg_freedom)
            row += 1

    def saveSpectrum(self, n, row, shift1, shift2, spec_found):
        """Copy the spectrum to the record array for the outflash table.

        Parameters
        ----------
        n: int
            Index of current lamp flash.

        row: int
            Row index (zero indexed) in output table.

        shift1: float
            Shift in X direction.

        shift2: float
            Shift in Y direction.

        spec_found: boolean
            Was the wavecal spectrum actually found?
        """

        if self.ofd is None:
            return

        t0 = self.lamp_on[n]
        t1 = self.lamp_off[n]

        self.ofd[1].data.field("time")[row] = self.lamp_median[n]
        self.ofd[1].data.field("exptime")[row] = t1 - t0
        self.ofd[1].data.field("lamp_on")[row] = self.lamp_on[n]
        self.ofd[1].data.field("lamp_off")[row] = self.lamp_off[n]
        self.ofd[1].data.field("shift_disp")[row] = shift1
        self.ofd[1].data.field("shift_xdisp")[row] = shift2
        self.ofd[1].data.field("spec_found")[row] = spec_found

        self.ofd[1].data.field("segment")[row] = "N/A"
        self.ofd[1].data.field("wavelength")[row][:] = 0.
        self.ofd[1].data.field("gross")[row][:] = 0.
        self.ofd[1].data.field("chi_square")[row] = 0.
        self.ofd[1].data.field("n_deg_freedom")[row] = 0
