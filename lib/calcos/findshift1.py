from __future__ import absolute_import, division         # confidence high
import copy
import numpy as np
from .calcosparam import *
from . import cosutil

# If there aren't at least this many counts in a wavecal spectrum, flag
# it as not found.
MIN_NUMBER_OF_COUNTS = 50

# For chi square.
N_SIGMA = 15.

# For comparison of individual shifts with the global shift (NUV only).
SLOP = 25.      # pixels

# The maximum number of pixels to set to zero at an end of a spectrum.
TRIM = 10
NUV_X = 1024

# This is used by findShift.  5 is listed last because that's the number
# of points we'll use for determining the shift; the others are for
# finding the scatter of the shift for these values of npts.
NPTS_RANGE = [3, 4, 6, 7, 5]

class Shift1(object):
    """Find the shift in the dispersion direction.

    fs1 = findshift1.Shift1(spectra, templates, info, reffiles,
                            xc_range, fp_pixel_shift, initial_offset=0,
                            spec_found={})
    The public methods are:
        fs1.findShifts()
        shift1 = fs1.getShift1(key)
        fs1.setShift1(key, shift1, fp=True)
        user_specified = fs1.getUserSpecified(key)
        orig_shift1 = fs1.getOrigShift1(key)
        measured_shift1 = getMeasuredShift1(key)
        fp_pixel_shift = getFpPixelShift(key)
        flag = fs1.getSpecFound(key)
        error_estimate = fs1.getScatter(key)
        chi_square = fs1.getChiSq(key)
        number_of_degrees_of_freedom = fs1.getNdf(key)
    The following may be used for testing/debugging:
        spectrum = fs1.getSpec(key)     # a slice, normalized to the template
        template = fs1.getTmpl(key)     # a slice

    Parameters
    ----------
    spectra: dictionary of array_like
        The 1-D extracted spectra; these should be in counts, not counts/s,
        because Poisson statistics will be assumed.  Key is segment or
        stripe name.

    templates: dictionary of array_like
        Template spectra (same keys as for spectra) from the lamptab.
        Key is segment or stripe name.

    info: dictionary
        Header keywords and values.

    reffiles: dictionary
        Reference file names.

    xc_range: int
        The maximum offset (lag) for the cross correlation; this is from
        column XC_RANGE in the WCPTAB.

    fp_pixel_shift: dictionary
        From the FP_PIXEL_SHIFT column of the lamptab, with an entry (same
        keys as for spectra) for each segment or stripe; if that column
        is not present in the lamptab, the values should be 0.

    initial_offset: float
        If the lamptab contains both the FPOFFSET and FP_PIXEL_SHIFT
        columns (this is the normal case), this argument is the value read
        from column SEARCH_OFFSET in the WCPTAB.  If SEARCH_OFFSET is not
        found in the WCPTAB, initial_offset should be set to 0.
        If FPOFFSET and FP_PIXEL_SHIFT are not in the lamptab,
        initial_offset should be SEARCH_OFFSET + (FPOFFSET * STEPSIZE),
        i.e. the sum of SEARCH_OFFSET from WCPTAB and the nominal offset
        between the wavecal spectrum and the template spectrum from the
        lamptab.

    spec_found: dictionary of boolean flags
        True for each spectrum that was found (same keys as for spectra);
        this is a set of initial values which will be saved as an
        attribute, updated, and may be gotten via method getSpecFound.
    """

    def __init__(self, spectra, templates,
                 info, reffiles,
                 xc_range, fp_pixel_shift, initial_offset=0.,
                 spec_found={}):

        self.spectra = copy.deepcopy(spectra)
        self.templates = copy.deepcopy(templates)
        self.info = info
        self.reffiles = reffiles
        self.xc_range = xc_range
        self.lenxc = 2*xc_range + 1
        self.fp_pixel_shift = fp_pixel_shift
        self.initial_offset = initial_offset

        # These are the results.
        self.spec_found = copy.copy(spec_found)         # may be updated
        self.shift1_dict = {}
        self.orig_shift1_dict = {}      # shift1 even if poorly found
        self.user_specified_dict = {}
        self.chisq_dict = {}
        self.ndf_dict = {}              # number of degrees of freedom
        self.scatter_dict = {}
        # for testing; these are aligned slices
        self.spec_dict = {}
        self.tmpl_dict = {}

        # These are for copying info from computeNormalization to
        # computeChiSquare.
        self.factor = None      # spec = baseline + factor * tmpl + noise
        self.baseline = None
        self.rms = None
        self.spec_slice = None
        self.tmpl_slice = None

        # working parameters
        # keys = self.spectra.keys()
        # keys.sort()
        keys = sorted(self.spectra)
        self.keys = keys
        self.current_key = ""

        self.status = 0                 # currently not used

        # LP4 and later wcptab ref files will include the chi-squared threshold
        # Will read from the wcptab column "N_SIGMA" if it exists
        # If not, will revert to default value
        self.chi_sq_threshold = N_SIGMA

        # Trim bright lines off the ends of the spectra.
        self.trimSpectra()

        if not self.spec_found:
            for key in keys:
                self.spec_found[key] = True

        for key in keys:
            self.shift1_dict[key] = 0.
            self.orig_shift1_dict[key] = 0.
            self.user_specified_dict[key] = False
            self.scatter_dict[key] = 0.
            self.chisq_dict[key] = 0.
            self.ndf_dict[key] = 0

    def trimSpectra(self):
        """Trim the ends of the spectra, if bright lines are found.

        This function checks the left and right edges of NUV wavecal spectra,
        looking for emission lines that are partially truncated by the edge.
        For each truncated line that is found, pixel values near the edge will
        be set to zero to reduce or eliminate the effect of the line when
        comparing the wavecal spectrum with the template.
        """

        if self.info["detector"] == "FUV":
            return

        # binning by three pixels and a cutoff of 3*10 are arbitrary, and
        # they may need to be adjusted
        BIN = 3                 # binning of spectrum to get s_l or s_r

        for key in self.keys:
            nelem = len(self.spectra[key])
            index = np.argsort(self.spectra[key])
            median = self.spectra[key][index[nelem//2]]
            cutoff_brightness = float(BIN * max(1., median))
            cutoff_brightness = max(BIN*10., cutoff_brightness)
            # first and last pixels that are on the detector
            left = int(self.info["x_offset"])
            right = left + NUV_X - 1
            assert left < nelem and right < nelem
            # first and last twelve (detector) pixel values, binned by three
            # (so these are four-element arrays)
            s_l = self.spectra[key][left:left+12:BIN] + \
                  self.spectra[key][left+1:left+13:BIN] + \
                  self.spectra[key][left+2:left+14:BIN]
            s_r = self.spectra[key][right-11:right-1:BIN] + \
                  self.spectra[key][right-10:right:BIN] + \
                  self.spectra[key][right-9:right+1:BIN]
            # mod_left and mod_right are flags that can be set within a loop;
            # if True, a message will be printed (outside the loop) to say
            # that a wavecal spectral line was on the left or right edge and
            # has been clobbered.
            mod_left = False
            mod_right = False
            # Look for values in the binned arrays that are bright (compared
            # with the median) and decreasing in brightness away from the edge.
            if s_l[0] > cutoff_brightness:
                i = 0               # index in s_l
                j = left            # index in spectrum
                while i < 3:
                    if s_l[i] > s_l[i+1]:
                        mod_left = True
                        self.spectra[key][j:j+BIN] = 0.
                        i += 1; j += BIN
                    else:
                        break
            if s_r[3] > cutoff_brightness:
                i = 3; j = right + 1
                while i >= 0:
                    if s_r[i] > s_r[i-1]:
                        mod_right = True
                        self.spectra[key][j-BIN:j] = 0.
                        i -= 1; j -= BIN
                    else:
                        break

            if mod_left:
                cosutil.printMsg("Info:  truncated line removed at left"
                                 " edge of %s" % key)
            if mod_right:
                cosutil.printMsg("Info:  truncated line removed at right"
                                 " edge of %s" % key)

    def getShift1(self, key):
        """Return the shift in the dispersion direction."""

        if key in self.shift1_dict:
            return self.shift1_dict[key] + self.fp_pixel_shift[key]
        else:
            return 0.

    def setShift1(self, key, shift1, fp=True):
        """Set shift1 to the value supplied by the user."""

        if key not in self.shift1_dict:
            return

        spectrum = self.spectra[key]
        template = self.templates[key]
        self.computeNormalization(spectrum, template,
                                  shift1 - self.fp_pixel_shift[key])
        if self.factor is None:
            self.chisq_dict[key] = 0.
            self.ndf_dict[key] = 0
            self.spec_dict[key] = None
            self.tmpl_dict[key] = None
        else:
            (chisq, ndf, spec, tmpl) = \
                        self.computeChiSquare(spectrum, template)
            self.chisq_dict[key] = chisq
            self.ndf_dict[key] = ndf
            self.spec_dict[key] = spec.copy()
            self.tmpl_dict[key] = tmpl.copy()

        # the default is to include the fp_pixel_shift offset
        if fp:
            self.shift1_dict[key] = shift1 - self.fp_pixel_shift[key]
        else:
            self.shift1_dict[key] = shift1
        self.spec_found[key] = True
        self.user_specified_dict[key] = True
        self.scatter_dict[key] = 0.

    def getUserSpecified(self, key):
        """Return True if shift1 was specified by the user."""

        if key in self.user_specified_dict:
            return self.user_specified_dict[key]
        else:
            return False

    def getOrigShift1(self, key):
        """Return the shift1 value even if it was poorly found."""

        if key in self.shift1_dict:
            return self.orig_shift1_dict[key] + self.fp_pixel_shift[key]
        else:
            return 0.

    def getMeasuredShift1(self, key):
        """Return the shift1 value that was directly measured."""

        if key in self.shift1_dict:
            return self.orig_shift1_dict[key]
        else:
            return 0.

    def getFpPixelShift(self, key):
        """Return the fp_pixel_shift value."""

        if key in self.shift1_dict:
            return self.fp_pixel_shift[key]
        else:
            return 0.

    def getSpecFound(self, key):
        """Return a flag indicating whether the spectrum was found."""

        if key in self.spec_found:
            return self.spec_found[key]
        else:
            return False

    def getScatter(self, key):
        """Return an estimate of the uncertainty in shift1."""

        if key in self.scatter_dict:
            return self.scatter_dict[key]
        else:
            return 0.

    def getChiSq(self, key):
        """Return Chi square for the spectrum vs template."""

        if key in self.chisq_dict:
            return self.chisq_dict[key]
        else:
            return -1.

    def getNdf(self, key):
        """Return the number of number of degrees of freedom for Chi square."""

        if key in self.ndf_dict:
            return self.ndf_dict[key]
        else:
            return 0

    def getSpec(self, key):
        """Return the extracted spectrum for testing."""

        if key in self.spec_dict:
            return self.spec_dict[key]
        else:
            return np.zeros(1, dtype=np.float32)

    def getTmpl(self, key):
        """Return the template spectrum for testing."""

        if key in self.tmpl_dict:
            return self.tmpl_dict[key]
        else:
            return np.zeros(1, dtype=np.float32)

    def findShifts(self):
        """Find the shifts in the dispersion direction.

        This function updates:
            self.spec_found
            self.shift1_dict
            self.scatter_dict
            self.chisq_dict
            self.ndf_dict
            self.spec_dict
            self.tmpl_dict
        """

        nelem = len(self.keys)
        if nelem < 1:
            return
        #
        # Try reading the chi square threshold from the wcptab
        opt_elem = self.info["opt_elem"]
        wcp_info = cosutil.getTable(self.reffiles["wcptab"],
                                    filter={"opt_elem": opt_elem},
                                    exactly_one=True)
        try:
            # This field should exist in LP4 and later data
            self.chi_sq_threshold = wcp_info.field("N_SIGMA")
        except KeyError:
            # This could occur if the wcptab doesn't have the
            # N_SIGMA column.
            # We already have the threshold set in the constructor, so
            # we don't need to do anything if we can't get it from the
            # wcptab
            pass
        self.checkCounts()              # flag spectra with negligible counts

        if self.info["detector"] == "FUV":
            self.findShiftsFUV()
        else:
            self.findShiftsNUV()

    def findShiftsFUV(self):
        """Find the shifts in the dispersion direction for FUV data."""

        for key in self.keys:
            if key not in self.templates.keys():
                self.notFound(key)
                continue
            self.current_key = key
            spectrum = self.spectra[key]
            template = self.templates[key]
            (shift, orig_shift1, scatter, foundit) = \
                        self.findShift(spectrum, template)
            self.orig_shift1_dict[key] = orig_shift1
            self.scatter_dict[key] = scatter
            if not foundit:
                self.notFound(key)
                continue

            self.computeNormalization(spectrum, template, shift)
            if self.factor is None:
                self.notFound(key)
                continue
            (chisq, ndf, spec, tmpl) = \
                        self.computeChiSquare(spectrum, template)
            self.chisq_dict[key] = chisq
            self.ndf_dict[key] = ndf
            self.spec_dict[key] = spec.copy()
            self.tmpl_dict[key] = tmpl.copy()
            if ndf > 0:
                ratio = chisq / ndf
            else:
                ratio = chisq
            if shift is None or ratio > self.chi_sq_threshold \
                    or ratio < 1./self.chi_sq_threshold:
                shift = 0.
                self.spec_found[key] = False
            self.shift1_dict[key] = shift

    def findShiftsNUV(self):
        """Find the shifts in the dispersion direction for NUV data."""

        global_shift = self.globalShift()

        for key in self.keys:
            if key not in self.templates.keys():
                self.notFound(key)
                continue
            self.current_key = key
            spectrum = self.spectra[key]
            template = self.templates[key]
            (shift, orig_shift1, scatter, foundit) = \
                        self.findShift(spectrum, template)
            self.orig_shift1_dict[key] = orig_shift1
            self.scatter_dict[key] = scatter
            if not foundit:
                self.spec_found[key] = False
            elif global_shift is not None:
                if abs(shift - global_shift) > SLOP:
                    self.spec_found[key] = False
            self.shift1_dict[key] = shift

            self.computeNormalization(spectrum, template, shift)
            if self.factor is None:
                self.notFound(key)
                continue

            (chisq, ndf, spec, tmpl) = \
                        self.computeChiSquare(spectrum, template)
            self.chisq_dict[key] = chisq
            self.ndf_dict[key] = ndf
            self.spec_dict[key] = spec.copy()
            self.tmpl_dict[key] = tmpl.copy()
            if ndf > 0:
                ratio = chisq / ndf
            else:
                ratio = chisq
            if shift is None or ratio > self.chi_sq_threshold \
                    or ratio < 1./self.chi_sq_threshold:
                self.spec_found[key] = False
                shift = 0.

        self.repairNUV()        # assign best-guess values for bad shifts

    def notFound(self, key):
        self.shift1_dict[key] = 0.
        self.spec_found[key] = False
        self.chisq_dict[key] = 0.
        self.ndf_dict[key] = 0

    def checkCounts(self):
        """Flag data with negligible counts.

        This function updates:
            self.spec_found
        """

        for key in self.keys:
            self.current_key = key
            if self.spectra[key].sum() < MIN_NUMBER_OF_COUNTS:
                self.spec_found[key] = False

    def globalShift(self):
        """Return the shift of the sum of all NUV stripes.

        Returns
        -------
        float
            The shift of the sum of all (nominally three) NUV stripes
        """

        key = self.keys[0]

        # Add spectra together, add templates together, find the shift.
        nelem = len(self.spectra[key])
        sum_spectra = np.zeros(nelem, dtype=np.float64)
        sum_templates = np.zeros(nelem, dtype=np.float64)
        nsum = 0
        for key in self.keys:
            self.current_key = key
            # Skip spectra for which spec_found is already set to False,
            # because they probably have negligible counts.
            if key in self.templates and self.spec_found[key]:
                sum_spectra += self.spectra[key]
                sum_templates += self.templates[key]
                nsum += 1
        if nsum < 1:
            return 0

        self.current_key = "all"
        (global_shift, orig_shift1, scatter, foundit) = \
                self.findShift(sum_spectra, sum_templates)
        if not foundit:
            global_shift = None

        return global_shift

    def findShift(self, spectrum, template):
        """Find a shift in the dispersion direction.

        Parameters
        ----------
        spectrum: array_like
            A 1-D extracted wavecal spectrum

        template: array_like
            Template spectrum

        Returns
        -------
        tuple
            (shift, orig_shift1, scatter, foundit), where shift is the
            shift in the dispersion direction, orig_shift1 is the shift
            even if it wasn't well determined, scatter is an estimate of
            the uncertainty in shift1, and foundit is True if we think the
            shift was actually found
        """

        # xxx need to improve the comments in this function

        GOOD_VALUE = 1
        BAD_VALUE = -1

        lenxc = self.lenxc
        # These arrays are for finding a minimum or maximum.
        # We expect to find a minimum rms.
        # These arrays all have length self.lenxc, and we can
        # use the same index(e.g. imin) in all of these arrays.
        rms = np.zeros(lenxc, dtype=np.float32)
        chisq = np.zeros(lenxc, dtype=np.float32)
        factor = np.zeros(lenxc, dtype=np.float32)
        baseline = np.zeros(lenxc, dtype=np.float32)
        flag = np.zeros(lenxc, dtype=np.int32)
        maxlag = lenxc // 2
        i = 0
        # Assign reduced chi square to chisq.
        for shift in range(-maxlag, maxlag+1):
            shift_x = shift + self.initial_offset
            # compute self.rms and other values
            self.computeNormalization(spectrum, template, shift_x)
            if self.factor is not None:
                (chisq_i, ndf, spec, tmpl) = \
                        self.computeChiSquare(spectrum, template)
                chisq[i] = chisq_i / float(max(ndf, 1))
            if self.factor is None:
                flag[i] = BAD_VALUE
            else:
                flag[i] = GOOD_VALUE
                rms[i] = self.rms
            i += 1
        # Where factor was None, set chisq to a large value.
        max_xc = chisq.max()
        chisq = np.where(flag == GOOD_VALUE, chisq, 2.*max_xc)

        good_values = np.where(flag == GOOD_VALUE)
        # Find all the local minima in the array of RMS values.
        n = lenxc
        # indices of local minima in rms
        local_minima = np.where(np.logical_and(rms[1:n-1] <= rms[0:n-2],
                                               rms[1:n-1] <= rms[2:n]))
        if len(good_values[0]) <= 0 or len(local_minima[0]) <= 0:
            return (0., 0., 0., False)
        # Extract the array of indices, and add one to get indices in rms
        # (because rms[1:n-1] starts with 1).
        local_minima = local_minima[0] + 1

        # Check each local minimum to make sure none of the nearby values
        # is flagged as bad.
        # real_minima, rms_list, and chisq_list will all have the same
        # number of elements.
        real_minima = []        # list of indices (imin) of minima of rms
        chisq_list = []         # chi square at each imin in real_minima
        rms_list = []           # value of rms at each imin in real_minima
        npts = max(NPTS_RANGE)
        for imin in local_minima:
            i1 = imin - npts//2
            i1 = max(i1, 0)
            i2 = i1 + npts
            i2 = min(i2, lenxc)
            i1 = i2 - npts
            bad = (flag[i1:i2] == BAD_VALUE)
            if np.any(bad):
                continue
            real_minima.append(imin)
            rms_list.append(rms[imin])
            chisq_list.append(chisq[imin])
        if len(real_minima) <= 0:
            return (0., 0., 0., False)

        # Pick the location with the smallest RMS.  index_of_min is the
        # index in rms_list that gives the smallest RMS.
        min_rms = None
        index_of_min = 0                # initial values
        for (i, rms_i) in enumerate(rms_list):
            chisq_i = chisq_list[i]

        # The values in rms_index and chisq_index are indices in real_minima,
        # rms_list, and chisq_list.
        # index_of_min (below) will also be an index in real_minima, etc.
        # Example:
        # k = rms_index[0]         the point with the minimum RMS
        # imin = real_minima[k]    the index in rms, factor, baseline, flag
        # rms[imin] and rms_list[k] will be the same
        rms_array = np.array(rms_list)
        rms_index = np.argsort(rms_array)
        chisq_array = np.array(chisq_list)
        chisq_index = np.argsort(chisq_array)

        # Of the PICK_N points with the smallest RMS, skip ones that don't have
        # positive curvature, then select the one with smallest chi square.
        PICK_N = min(10, len(rms_index))
        x = np.arange(5, dtype=np.float64)      # for fitting a quadratic
        index_of_min = None                     # initial value
        for i in range(PICK_N):
            # k is an index in real_minima, rms_list, and chisq_list, while
            # real_minima[k] is an index in rms.
            k = rms_index[i]
            imin = real_minima[k]
            # fit a quadratic to five points centered on imin, to get the
            # curvature
            j1 = imin - 2
            j1 = max(j1, 0)
            j2 = j1 + 5
            j2 = min(j2, lenxc)
            j1 = j2 - 5
            (coeff, var) = cosutil.fitQuadratic(x, rms[j1:j2])
            if coeff is None or coeff[2] <= 0.: # coeff[2] is the curvature
                continue
            if index_of_min is None or chisq_list[k] < min_chisq:
                min_chisq = chisq_list[k]
                index_of_min = k
        if index_of_min is None:
            return (0., 0., 0., False)

        # Fit a quadratic to points near the minimum of chisq.
        imin = real_minima[index_of_min]
        (shift, orig_shift1, scatter) = self.findMinimum(imin, maxlag, chisq)

        return (shift, orig_shift1, scatter, True)

    def findMinimum(self, imin, maxlag, chisq):
        """Fit a quadratic to points near the minimum of chisq.

        Parameters
        ----------
        imin: int
            Index to use as a starting point

        maxlag: int
            Half the search range, i.e. maximum offset from nominal

        chisq: array_like
            Reduced chi square, one for each offset

        Returns
        -------
        tuple of three floats
            (shift, orig_shift1, scatter), where shift is the shift in the
            dispersion direction (or zero if not found), orig_shift1 will
            be the same value if the shift was found (but will be the shift
            based on imin if not found), and scatter is a measure of the
            uncertainty in the shift
        """

        lenxc = self.lenxc
        min_shift = None
        max_shift = None
        for npts in NPTS_RANGE:
            i1 = imin - npts//2
            i1 = max(i1, 0)
            i2 = i1 + npts
            i2 = min(i2, lenxc)
            i1 = i2 - npts
            x = np.arange(npts, dtype=np.float64)
            (coeff, var) = cosutil.fitQuadratic(x, chisq[i1:i2])
            (x_min, sigma_shift) = cosutil.centerOfQuadratic(coeff, var)
            if x_min is None:
                shift = 0.
                orig_shift1 = imin + self.initial_offset - maxlag
                self.spec_found[self.current_key] = False
            else:
                shift = x_min + i1 + self.initial_offset - maxlag
                orig_shift1 = shift
                if min_shift is None or shift < min_shift:
                    min_shift = shift
                if max_shift is None or shift > max_shift:
                    max_shift = shift
        if min_shift is None or max_shift is None:
            scatter = sigma_shift
        else:
            scatter = max(sigma_shift, (max_shift - min_shift) / 2.)

        return (shift, orig_shift1, scatter)

    def computeNormalization(self, spectrum, template, shift):
        """Compute a normalization factor between spectrum and template.

        The following attributes are assigned, to be used by computeChiSquare:
            self.factor
            self.baseline
            self.rms
                where spectrum = self.baseline + self.factor * template + noise
                within an overlap region (see spec_slice and tmpl_slice)
            self.spec_slice: the slice of the spectrum to use
            self.tmpl_slice: the slice of the template to use

        self.factor will be set to None if the normalization could not
        be found due to one of these conditions:
            no non-zero data (overlap region could not be found)
            singularity in the fit between template and spectrum
            factor is zero or negative

        Parameters
        ----------
        spectrum: array_like
            The 1-D extracted wavecal spectrum

        template: array_like
            Template spectrum

        shift: float or None
            The pixel shift in the dispersion direction, or None if the
            shift was not found successfully
        """

        self.factor = None
        self.baseline = 0.
        self.rms = 0.

        if shift is None:
            shift = 0.
        shift = int(round(shift))

        len_spec = len(spectrum)

        # Get the overlap region.
        if shift >= 0:
            s0 = shift
            s1 = len_spec
            t0 = 0
            t1 = len_spec - shift
        else:
            s0 = 0
            s1 = len_spec - (-shift)
            t0 = -shift
            t1 = len_spec

        # Narrow the endpoints to exclude elements that are zero in either
        # spectrum or template.
        done = False
        while not done:
            if spectrum[s0] != 0. and template[t0] != 0.:
                break
            s0 += 1
            t0 += 1
            if s0 >= s1-1:
                done = True
        while not done:
            if spectrum[s1-1] != 0. and template[t1-1] != 0.:
                break
            s1 -= 1
            t1 -= 1
            if s1 <= s0:
                done = True
        if done:
            self.factor = None
            return

        self.spec_slice = (s0, s1)
        self.tmpl_slice = (t0, t1)

        # Fit the template to the spectrum:
        # spec = baseline + factor * tmpl + noise
        n = float(s1 - s0)
        spec = spectrum[s0:s1]
        tmpl = template[t0:t1]
        sum_s = spec.sum(dtype=np.float64)
        sum_t = tmpl.sum(dtype=np.float64)
        sum_t2 = (tmpl**2).sum(dtype=np.float64)
        sum_st = (spec * tmpl).sum(dtype=np.float64)
        denominator = sum_t**2 - n * sum_t2
        if denominator == 0.:
            self.factor = None
            return
        else:
            self.factor = (sum_s * sum_t - n * sum_st) / denominator
            if self.factor <= 0.:
                self.factor = None
                return

        self.baseline = (sum_s - self.factor * sum_t) / n

        diff = spec - (self.baseline + self.factor * tmpl)
        nelem = len(diff)
        if nelem > 1:
            self.rms = float(np.sqrt((diff**2).sum() / (nelem-1.)))

    def computeChiSquare(self, spectrum, template):
        """Compute chi square for spectrum and template.

        This method makes use of the following attributes that were
        assigned by computeNormalization:
            self.factor
            self.baseline
            self.spec_slice
            self.tmpl_slice

        Parameters
        ----------
        spectrum: array_like
            The 1-D extracted wavecal spectrum

        template: array_like
            Template spectrum

        Returns
        -------
        tuple
            Chi square, the number of degrees of freedom, the overlapping
            slice of the spectrum and normalized template
        """

        (s0, s1) = self.spec_slice
        (t0, t1) = self.tmpl_slice
        spec = spectrum[s0:s1]
        tmpl = template[t0:t1]

        # Normalize the spectrum to match the template.
        n_spec = (spec - self.baseline) / self.factor

        # sigma for the template = sqrt(template);
        # sigma for the normalized spectrum = sqrt(spectrum) / factor;
        # variance for the normalized spectrum = spectrum / factor**2
        # add variances (add sigmas in quadrature)
        variance = spec / self.factor**2 + tmpl

        # When computing chi square, include only those elements for which
        # either the spectrum or the template is non-zero.
        # (see both_zero below)
        either_positive = np.logical_or(spec > 0., tmpl > 0.)
        nelem = either_positive.sum(dtype=np.float64)
        ndf = max(0, nelem - 1)                 # number of degrees of freedom
        # v is scratch, just so we can divide by it
        v = np.where(variance > 0., variance, 1.)

        # Compute chi square.
        diff = (n_spec - tmpl)
        a_chisq = diff**2 / v                   # array of values
        # Truncate chi square at 1 if both the spectrum and template are
        # 0 or 1.  This cutoff is pretty arbitrary, just a number that's
        # clearly in the noise.
        both_small = np.logical_and(spec <= 1.5, tmpl <= 1.5)
        a_chisq = np.where(both_small, np.minimum(a_chisq, 1.), a_chisq)
        # If both the spectrum (not scaled) and template are zero,
        # chi square should be zero (regardless of scaling).
        both_zero = np.logical_and(spec == 0., tmpl == 0.)
        a_chisq = np.where(both_zero, 0., a_chisq)

        chisq = float(a_chisq.sum(dtype=np.float64))

        return (chisq, ndf, n_spec, tmpl)

    def repairNUV(self):
        """Assign reasonable values for shifts that weren't found."""

        # This is an estimate of the relative shifts between stripes.
        # (52 pixels per fpoffset step)
        # These polynomial coefficients should be read from a reference table.
        offset = {("NUVA", "G185M"): [-0.5, 0.02307692],
                  ("NUVB", "G185M"): [0.0, 0.0],
                  ("NUVC", "G185M"): [1.0, 0.02307692],
                  ("NUVA", "G225M"): [-0.5, 0.02307692],
                  ("NUVB", "G225M"): [0.0, 0.0],
                  ("NUVC", "G225M"): [1.0, 0.02307692],
                  ("NUVA", "G285M"): [-0.5, 0.02307692],
                  ("NUVB", "G285M"): [0.0, 0.0],
                  ("NUVC", "G285M"): [1.0, 0.02307692],
                  ("NUVA", "G230L"): [0.0, 0.0],
                  ("NUVB", "G230L"): [0.0, 0.0],
                  ("NUVC", "G230L"): [0.0, 0.0]}
        # note:
        # 0.02307692 = 1.2 / 52, i.e. 1.2 pixels per fpoffset step of 52 pixels

        # Compute the average shift of the stripes for which shift1 was
        # found, first applying the offset between the current stripe and
        # NUVB.  The average will then be of NUVB-equivalent shifts.

        # shift_0 (in the loop below) is the shift of the wavecal spectrum
        # from the template at fpoffset = 0.  This is used under the assumption
        # that the relative offsets between stripes depends on this shift.
        # Currently, shift_0 is just the anticipated fpoffset shift, not
        # the total shift1[abc] value.
        # shift_to_nuvb is the relative shift from the current stripe to
        # NUVB, based on shift_0 and the dictionary 'offset'.

        ngood = 0
        sum_shifts = 0.
        for key in self.keys:
            self.current_key = key
            off_key = (key, self.info["opt_elem"])
            if self.spec_found[key]:
                shift_0 = self.initial_offset + self.fp_pixel_shift[key]
                shift_to_nuvb = self.evalPoly(shift_0, offset[off_key])
                sum_shifts += (self.shift1_dict[key] - shift_to_nuvb)
                ngood += 1
        if ngood == 0:
            return              # no data; can't do anything

        # Now use the average NUVB-equivalent shift to replace the shift of
        # any stripe that wasn't found, applying the offset from NUVB to the
        # missing stripe or stripes.
        mean_shift = sum_shifts / ngood
        for key in self.keys:
            self.current_key = key
            off_key = (key, self.info["opt_elem"])
            if not self.spec_found[key]:
                shift_0 = self.initial_offset + self.fp_pixel_shift[key]
                shift_to_nuvb = self.evalPoly(shift_0, offset[off_key])
                self.shift1_dict[key] = mean_shift + shift_to_nuvb

    def evalPoly(self, x, coeff):
        """Evaluate a polynomial in x.

        Parameters
        ----------
        x: float
            Argument

        Returns
        -------
        float
            coeff[0] + coeff[1] * x + coeff[2] * x**2 + ...
        """

        ncoeff = len(coeff)
        sum = coeff[ncoeff-1]
        for i in range(ncoeff-2, -1, -1):
            sum = sum * x + coeff[i]
        return sum
