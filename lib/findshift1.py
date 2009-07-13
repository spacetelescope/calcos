from __future__ import division
import copy
import numpy as N
from calcosparam import *

# Peak in cross correlation is at an endpoint of the range for finding the
# shift from a wavecal observation; the shift is therefore likely to be
# incorrect.
AT_END = 1
# Indeterminate location of peak in cross correlation for finding the shift
# from a wavecal observation.  The shift for this wavecal image will be set
# to zero.
XC_NOT_FOUND = 2                # denominator >= 0

# If there aren't at least this many counts in a wavecal spectrum, flag
# it as not found.
MIN_NUMBER_OF_COUNTS = 50

# For chi square.
N_SIGMA = 6.

# For comparison of individual shifts with the global shift (NUV only).
# SLOP = 15.      # pixels
SLOP = 25.      # pixels

# The number of pixels to set to zero at the ends of the template spectra.
TRIM = 20       # this is currently not used

class Shift1 (object):
    """Find the shift in the dispersion direction.

    fs1 = findshift1.Shift1 (spectra, templates, info, reffiles,
                             xc_range, stepsize, fp=0, spec_found={})
    The public methods are:
        fs1.findShifts()
        shift1 = fs1.getShift1 (key)
        fs1.setShift1 (key, shift1)
        user_specified = fs1.getUserSpecified (key)
        orig_shift1 = fs1.getOrigShift1 (key)
        flag = fs1.getSpecFound (key)
        chi_square = fs1.getChiSq (key)
        number_of_degrees_of_freedom = fs1.getNdf (key)
    The following may be used for testing/debugging:
        spectrum = fs1.getSpec (key)
        template = fs1.getTmpl (key)
        xc_array = fs1.getXc()

    @ivar spectra: the 1-D extracted spectra; these should be in counts,
        not counts/s, because Poisson statistics will be assumed
    @type spectra: dictionary of arrays
    @ivar templates: template spectra (same keys as for spectra)
    @type templates: dictionary of arrays
    @ivar info: keywords and values
    @type info: dictionary
    @ivar reffiles: reference file names
    @type reffiles: dictionary
    @ivar xc_range: the maximum offset (lag) for the cross correlation;
        this is from column XC_RANGE in the WCPTAB
    @type xc_range: int
    @ivar stepsize: approximate number of pixels for each fpoffset step,
        used for an initial offset for the cross correlation; this is from
        column STEPSIZE in the WCPTAB
    @type stepsize: int
    @ivar fp: value of keyword FPOFFSET, or 0 if no initial offset
        should be applied
    @type fp: int
    @ivar spec_found: True for each spectrum that was found (same keys as
        for spectra)
    @type spec_found: dictionary of boolean flags
    """

    def __init__ (self, spectra, templates,
                  info, reffiles,
                  xc_range, stepsize, fp=0, spec_found={}):

        self.spectra = copy.deepcopy (spectra)
        self.templates = copy.deepcopy (templates)
        self.info = info
        self.reffiles = reffiles
        self.xc_range = xc_range
        self.stepsize = stepsize
        self.fp = fp
        self.spec_found = copy.copy (spec_found)

        # These are the results.
        # self.spec_found               # copied from input and updated
        self.shift1_dict = {}
        self.orig_shift1_dict = {}      # shift1 even if poorly found
        self.user_specified_dict = {}
        self.n50_dict = {}
        self.chisq_dict = {}
        self.ndf_dict = {}              # number of degrees of freedom
        # for testing; these are aligned slices
        self.spec_dict = {}
        self.tmpl_dict = {}

        self.NFILES = 0                 # for debugging

        # working parameters
        keys = self.spectra.keys()
        keys.sort()
        self.keys = keys
        self.current_key = ""
        # buffer for cross correlation
        self.xc = N.zeros (2*xc_range + 1, dtype=N.float64)

        self.status = 0                 # currently not used

        # Trim the ends of the template spectra.
        # xxx self.trimTemplates (info["x_offset"], info["detector"])

        if not self.spec_found:
            for key in keys:
                self.spec_found[key] = True

        for key in keys:
            self.orig_shift1_dict[key] = 0.
            self.user_specified_dict[key] = False

    def trimTemplates (self, x_offset, detector):
        """Trim the ends of the template spectra."""

        for key in self.keys:
            if key not in self.templates.keys():
                continue
            nelem = len (self.templates[key])
            if detector == "FUV":
                naxis1 = FUV_X
            else:
                naxis1 = NUV_X
            trim_left = TRIM + x_offset
            trim_right = TRIM + nelem - naxis1 - x_offset
            self.templates[key][0:trim_left] = 0.
            self.templates[key][nelem-trim_right:] = 0.

    def getShift1 (self, key):
        """Return the shift in the dispersion direction."""

        if self.shift1_dict.has_key (key):
            return self.shift1_dict[key]
        else:
            return 0.

    def setShift1 (self, key, shift1):
        """Set shift1 to the value supplied by the user."""

        spectrum = self.spectra[key]
        template = self.templates[key]
        (factor, spec_slice, tmpl_slice) = \
                    self.computeNormalization (spectrum, template, shift1)
        if factor is None:
            self.n50_dict[key] = None
            self.chisq_dict[key] = 0.
            self.ndf_dict[key] = 0
        else:
            (chisq, ndf, spec, tmpl) = \
            self.computeChiSquare (spectrum, template,
                                   factor, spec_slice, tmpl_slice)
            self.chisq_dict[key] = chisq
            self.ndf_dict[key] = ndf
        self.spec_dict[key] = spec.copy()
        self.tmpl_dict[key] = tmpl.copy()

        self.shift1_dict[key] = shift1
        self.spec_found[key] = True
        self.user_specified_dict[key] = True

    def getUserSpecified (self, key):
        """Return True if shift1 was specified by the user."""

        if self.user_specified_dict.has_key (key):
            return self.user_specified_dict[key]
        else:
            return False

    def getOrigShift1 (self, key):
        """Return the shift1 value even if it was poorly found."""

        if self.shift1_dict.has_key (key):
            return self.orig_shift1_dict[key]
        else:
            return 0.

    def getSpecFound (self, key):
        """Return a flag indicating whether the spectrum was found."""

        if self.spec_found.has_key (key):
            return self.spec_found[key]
        else:
            return False

    def getN50 (self, key):
        """Return a diagnostic quantity."""

        if self.n50_dict.has_key (key):
            return self.n50_dict[key]
        else:
            return []

    def getChiSq (self, key):
        """Return Chi square for the spectrum vs template."""

        if self.chisq_dict.has_key (key):
            return self.chisq_dict[key]
        else:
            return -1.

    def getNdf (self, key):
        """Return the number of number of degrees of freedom for Chi square."""

        if self.ndf_dict.has_key (key):
            return self.ndf_dict[key]
        else:
            return 0

    def getSpec (self, key):
        """Return the extracted spectrum for testing."""

        if self.spec_dict.has_key (key):
            return self.spec_dict[key]
        else:
            return N.zeros (1, dtype=N.float32)

    def getTmpl (self, key):
        """Return the template spectrum for testing."""

        if self.tmpl_dict.has_key (key):
            return self.tmpl_dict[key]
        else:
            return N.zeros (1, dtype=N.float32)

    def getXc (self):
        """Return the result of cross correlation for testing."""

        return self.xc

    def findShifts (self):
        """Find the shifts in the dispersion direction.

        This function updates:
            self.spec_found
            self.shift1_dict
            self.n50_dict
            self.chisq_dict
            self.ndf_dict
            self.spec_dict
            self.tmpl_dict
        """

        nelem = len (self.keys)
        if nelem < 1:
            return

        self.checkCounts()              # flag spectra with negligible counts

        if self.info["detector"] == "FUV":
            self.findShiftsFUV()
        else:
            self.findShiftsNUV()

    def findShiftsFUV (self):
        """Find the shifts in the dispersion direction for FUV data."""

        for key in self.keys:
            if key not in self.templates.keys():
                self.notFound (key)
                continue
            self.current_key = key
            spectrum = self.spectra[key]
            template = self.templates[key]
            (shift, orig_shift1, n50) = self.findShift (spectrum, template)
            self.orig_shift1_dict[key] = orig_shift1
            if not self.spec_found[key]:
                self.notFound (key)
                continue

            (factor, spec_slice, tmpl_slice) = \
                    self.computeNormalization (spectrum, template, shift)
            if factor is None:
                self.notFound (key)
                continue
            (chisq, ndf, spec, tmpl) = \
            self.computeChiSquare (spectrum, template,
                                   factor, spec_slice, tmpl_slice)
            self.chisq_dict[key] = chisq
            self.ndf_dict[key] = ndf
            self.spec_dict[key] = spec.copy()
            self.tmpl_dict[key] = tmpl.copy()
            if ndf > 0:
                ratio = chisq / ndf
            else:
                ratio = chisq
            if shift is None or ratio > N_SIGMA or ratio < 1./N_SIGMA:
                shift = 0.
                self.spec_found[key] = False
            self.shift1_dict[key] = shift
            self.n50_dict[key] = n50

    def findShiftsNUV (self):
        """Find the shifts in the dispersion direction for NUV data."""

        global_shift = self.globalShift()

        for key in self.keys:
            if key not in self.templates.keys():
                self.notFound (key)
                continue
            self.current_key = key
            spectrum = self.spectra[key]
            template = self.templates[key]
            (shift, orig_shift1, n50) = self.findShift (spectrum, template)
            self.orig_shift1_dict[key] = orig_shift1
            if global_shift is not None:
                if abs (shift - global_shift) > SLOP:
                    self.spec_found[key] = False
            self.shift1_dict[key] = shift
            self.n50_dict[key] = n50

            (factor, spec_slice, tmpl_slice) = \
                    self.computeNormalization (spectrum, template, shift)
            if factor is None:
                self.notFound (key)
                continue

            (chisq, ndf, spec, tmpl) = \
            self.computeChiSquare (spectrum, template,
                                   factor, spec_slice, tmpl_slice)
            self.chisq_dict[key] = chisq
            self.ndf_dict[key] = ndf
            self.spec_dict[key] = spec.copy()
            self.tmpl_dict[key] = tmpl.copy()
            if ndf > 0:
                ratio = chisq / ndf
            else:
                ratio = chisq
            if shift is None or ratio > N_SIGMA or ratio < 1./N_SIGMA:
                self.spec_found[key] = False
                shift = 0.

        self.repairNUV()        # assign best-guess values for bad shifts

    def notFound (self, key):
        self.shift1_dict[key] = 0.
        self.spec_found[key] = False
        self.n50_dict[key] = None
        self.chisq_dict[key] = 0.
        self.ndf_dict[key] = 0

    def checkCounts (self):
        """Flag data with negligible counts.

        This function updates:
            self.spec_found
        """

        for key in self.keys:
            self.current_key = key
            if self.spectra[key].sum() < MIN_NUMBER_OF_COUNTS:
                self.spec_found[key] = False

    def globalShift (self):
        """Return the shift of the sum of all NUV stripes.

        @return: the shift of the sum of all (nominally three) NUV stripes
        @rtype: float
        """

        key = self.keys[0]

        # Add spectra together, add templates together, find the shift.
        nelem = len (self.spectra[key])
        sum_spectra = N.zeros (nelem, dtype=N.float64)
        sum_templates = N.zeros (nelem, dtype=N.float64)
        nsum = 0
        for key in self.keys:
            self.current_key = key
            # Skip spectra for which spec_found is already set to False,
            # because they probably have negligible counts.
            if self.templates.has_key (key) and self.spec_found[key]:
                sum_spectra += self.spectra[key]
                sum_templates += self.templates[key]
                nsum += 1
        if nsum < 1:
            return 0

        self.current_key = "all"
        (global_shift, orig_shift1, n50) = \
                self.findShift (sum_spectra, sum_templates)

        return global_shift

    def findShift (self, spectrum, template):
        """Find a shift in the dispersion direction.

        @param spectrum: a 1-D extracted spectrum
        @type spectrum: array
        @param template: template spectrum
        @type template: array

        @return: (shift, orig_shift1, n50), where shift is the shift in the
            dispersion direction, orig_shift1 is the shift even if it wasn't
            well determined, and n50 is no longer used
        @rtype: tuple
        """

        initial_offset = self.fp * self.stepsize

        FACTOR_IS_NONE = 1.             # this is a flag value

        lenxc = len (self.xc)
        maxlag = lenxc // 2
        for shift in range (-maxlag, maxlag+1):
            shift_x = shift + initial_offset
            (factor, spec_slice, tmpl_slice) = \
                    self.computeNormalization (spectrum, template, shift_x)
            if factor is None:
                # replace this later
                self.xc[maxlag+shift] = FACTOR_IS_NONE
                continue
            (chisq, ndf, spec, tmpl) = \
            self.computeChiSquare (spectrum, template,
                                   factor, spec_slice, tmpl_slice)
            # we want the minimum, but xcStat finds the maximum, so change sign
            self.xc[maxlag+shift] = -chisq
        # Where factor was None, set xc to a smaller value (larger chisq)
        # than any actual value.
        min_xc = self.xc.min()
        self.xc = N.where (self.xc == FACTOR_IS_NONE, 2.*min_xc, self.xc)

        # imax is the index of the maximum.  n50 is for diagnostic purposes.
        (imax, status, n50) = self.xcStat()

        i1 = imax - 1
        i1 = max (i1, 0)
        i2 = i1 + 2
        i2 = min (i2, lenxc-1)
        # Unless we're at an endpoint of xc, index is equal to imax, the index
        # of the peak.
        index = i2 - 1
        denominator = self.xc[index-1] - 2. * self.xc[index] + self.xc[index+1]
        if denominator >= 0.:
            status |= XC_NOT_FOUND
        else:
            location = (self.xc[index-1] - self.xc[index+1]) \
                       / (2. * denominator)
            # The peak in xc would be at maxlag (the middle element of xc)
            # if x and template were identical.
            shift = location + index + initial_offset - maxlag
            orig_shift1 = shift
        if status:
            shift = 0.
            orig_shift1 = index + initial_offset - maxlag
            self.spec_found[self.current_key] = False

        return (shift, orig_shift1, n50)

    def crosscor (self, spectrum, template, initial_offset):
        """Cross correlate two arrays to find the offset between them.

        example:
        import numpy as N
        xc = N.zeros (9, dtype=N.float64)
        template = N.array ([0.,0.,0.,0.,0.,0.,7.,0.,0.,0.,0.,0.,0.])
        spectrum = N.array ([1.,1.,1.,1.,1.,4.,1.,1.,1.,1.,1.,1.,1.])
        shift1 = -1.0
        spectrum = N.array ([1.,1.,1.,1.,1.,1.,1.,4.,1.,1.,1.,1.,1.])
        shift1 = +1.0

        This function updates:
            self.shift1_dict
            self.n50_dict
        """

        length = len (spectrum)
        lenxc = len (self.xc)
        maxlag = lenxc // 2

        for lag in range (-maxlag, maxlag+1):
            x1 = lag + initial_offset
            x2 = x1 + length
            x1 = max (x1, 0)
            x2 = min (x2, length)
            t1 = -lag - initial_offset
            t2 = t1 + length
            t1 = max (t1, 0)
            t2 = min (t2, length)
            product = spectrum[x1:x2] * template[t1:t2]
            self.xc[maxlag+lag] = product.sum()

        # imax is the index of the maximum.  n50 is for diagnostic purposes.
        (imax, status, n50) = self.xcStat()

        i1 = imax - 1
        i1 = max (i1, 0)
        i2 = i1 + 2
        i2 = min (i2, lenxc-1)
        # Unless we're at an endpoint of xc, index is equal to imax, the index
        # of the peak.
        index = i2 - 1
        denominator = self.xc[index-1] - 2. * self.xc[index] + self.xc[index+1]
        if denominator >= 0.:
            status |= XC_NOT_FOUND
            shift = None
        else:
            location = (self.xc[index-1] - self.xc[index+1]) \
                       / (2. * denominator)
            # The peak in xc would be at maxlag (the middle element of xc)
            # if x and template were identical.
            shift = location + index + initial_offset - maxlag

        return (shift, n50)

    def xcStat (self):
        """Find the location of the maximum, and some diagnostic info.

        @return: the index (int) of the maximum in xc, status (0 is OK), and
            a diagnostic quantity n50 (array)
        @rtype: tuple

        n50 is an array of nine elements, giving the number of elements in
        the cross correlation with values greater than various fractions of
        the range from the minimum to maximum values of the cross correlation.
        The fractions are 0.9, 0.8, ... 0.1.  The idea is that if the spectrum
        and the template are very similar (as we would expect), the values in
        n50 should be small, but they could increase sharply toward the end due
        to noise in the spectrum.  Elements near the middle of n50 should be of
        order twice the size of the resolution element.  Larger values in n50
        indicate a poorer agreement between the spectrum and the template.
        """

        status = 0

        xc_sort = N.argsort (self.xc)

        # Find the location of the maximum value, and other stuff.
        imax = xc_sort[-1]
        imin = xc_sort[0]
        maxval = self.xc[imax]
        minval = self.xc[imin]
        if imax == 0 or imax == len (self.xc) - 1:
            status = AT_END

        # Find the number of elements that have a value greater than
        # the midpoint of the range.
        diff = maxval - minval
        fractions = N.arange (0.9, 0., -0.1)
        cutoff = [(fraction * diff + minval) for fraction in fractions]
        n50 = N.zeros (len (cutoff), dtype=N.int32)
        for i in range (len (cutoff)):
            gt = self.xc > cutoff[i]
            # Using sum() here relies on the fact that the values are 0 or 1.
            n = N.sum (gt.astype (N.float64))
            n50[i] = int (round (n))

        return (imax, status, n50)

    def computeNormalization (self, spectrum, template, shift):
        """Compute a normalization factor between spectrum and template.

        @param spectrum: the 1-D extracted spectrum
        @type spectrum: array
        @param template: template spectrum
        @type template: array
        @param shift: the pixel shift in the dispersion direction
        @type shift: float, or None if shift was not found successfully

        @return: (factor, spec_slice, tmpl_slice), where factor is the ratio
            of the counts in the spectrum to the counts in the template within
            an overlap region; spec_slice is the slice for the spectrum;
            tmpl_slice is the slice for the template
        @rtype: tuple
        """

        if shift is None:
            shift = 0.
        shift = int (round (shift))

        len_spec = len (spectrum)

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
            return (None, (s0, s1), (t0, t1))

        # Truncate the noise to zero, then add up the remaining counts.
        nelem = s1 - s0
        middle = nelem // 2
        spec_cp = spectrum[s0:s1].copy()
        tmpl_cp = template[t0:t1].copy()
        spec_cp.sort()
        tmpl_cp.sort()
        median_spec = spec_cp[middle]
        median_tmpl = tmpl_cp[middle]
        # absolute values of the deviations from the median
        spec_diff = N.abs (spec_cp - median_spec)
        tmpl_diff = N.abs (tmpl_cp - median_tmpl)
        spec_diff.sort()
        tmpl_diff.sort()
        # median of the absolute values of the deviations from the median
        median_spec_diff = spec_diff[middle]
        median_tmpl_diff = tmpl_diff[middle]
        # Cut off the noise.
        cutoff = median_spec + 5. * median_spec_diff
        spec_cp = N.where (spec_cp > cutoff, spec_cp, 0.)
        cutoff = median_tmpl + 5. * median_tmpl_diff
        tmpl_cp = N.where (tmpl_cp > cutoff, tmpl_cp, 0.)

        sum_spec = spec_cp.sum()
        sum_tmpl = tmpl_cp.sum()
        if sum_tmpl <= 0.:
            factor = None
        else:
            factor = sum_spec / sum_tmpl

        return (factor, (s0, s1), (t0, t1))

    def computeChiSquare (self, spectrum, template,
                          factor, spec_slice, tmpl_slice):
        """Compute chi square for spectrum and template.

        @param spectrum: the 1-D extracted spectrum
        @type spectrum: array
        @param template: template spectrum
        @type template: array
        @param factor: the ratio of sum_spec to sum_tmpl
        @type factor: float
        @param spec_slice: the slice to use for the spectrum
        @type spec_slice: int
        @param tmpl_slice: the overlapping slice to use for the template
        @type tmpl_slice: int

        @return: Chi square, the number of degrees of freedom, the overlapping
            slice of the spectrum and normalized template
        @rtype: tuple
        """

        (s0, s1) = spec_slice
        (t0, t1) = tmpl_slice
        spec = spectrum[s0:s1]
        tmpl = template[t0:t1]

        # Normalize the template to match the spectrum.
        n_tmpl = tmpl * factor

        # explanation:
        # sigma for the template = sqrt (template);
        # sigma for the normalized template = sqrt (template) * factor;
        # variance for the normalized template = template * factor**2
        #   = normalized template * factor = n_tmpl * factor.
        variance = spec + n_tmpl * factor       # add sigmas in quadrature

        # self.writeDebug (spec, n_tmpl, variance)

        # When computing chi square, include only those elements for which
        # the variance is non-zero.
        ia = N.where (variance > 0.)
        nelem = len (ia[0])
        ndf = max (0, nelem - 1)                # number of degrees of freedom
        # v is scratch, just so we can divide by it
        v = N.where (variance > 0., variance, 1.)
        chisq = N.where (variance > 0., (spec - n_tmpl)**2 / v, 0.)
        chisq = float (chisq.sum(dtype=N.float64))

        return (chisq, ndf, spec, n_tmpl)

    def repairNUV (self):
        """Assign reasonable values for shifts that weren't found."""

        # This is a rough estimate of the relative shifts between stripes.
        offset = {"NUVA": -1., "NUVB": 0., "NUVC": 1.}

        ngood = 0
        sum_shifts = 0.
        for key in self.keys:
            self.current_key = key
            if self.spec_found[key]:
                sum_shifts += self.shift1_dict[key] + offset[key] * self.fp
                ngood += 1
        if ngood == 0:
            return              # no data; can't do anything

        mean_shift = sum_shifts / ngood
        for key in self.keys:
            self.current_key = key
            if not self.spec_found[key]:
                self.shift1_dict[key] = mean_shift + offset[key] * self.fp

    def writeDebug (self, spec, tmpl, variance):
        """debug"""

        filename = "debug_" + str (self.NFILES) + ".txt"

        fd = open (filename, "w")
        fd.write ("# key = %s\n" % self.current_key)
        for i in range (len (spec)):
            fd.write ("%.8g %.8g %.8g\n" % (spec[i], tmpl[i], variance[i]))
        fd.close()

        self.NFILES += 1
