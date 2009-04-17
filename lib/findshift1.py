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
N_SIGMA = 5.

# For comparison of individual shifts with the global shift (NUV only).
SLOP = 15.      # pixels

# The number of pixels to set to zero at the ends of the template spectra.
TRIM = 20

class Shift1 (object):
    """Find the shift in the dispersion direction.

    @ivar spectra: the 1-D extracted spectra; these should be in counts,
        not counts/s, because Poisson statistics will be assumed
    @type spectra: dictionary of arrays
    @ivar templates: template spectra
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
    @ivar spec_found: True for each spectrum that was found
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
        self.n50_dict = {}
        self.chisq_dict = {}
        self.nelem_dict = {}
        # for testing; these are aligned slices
        self.spec_dict = {}
        self.tmpl_dict = {}

        # working parameters
        keys = self.spectra.keys()
        keys.sort()
        self.keys = keys
        # buffer for cross correlation
        self.xc = N.zeros (2*xc_range + 1, dtype=N.float64)

        self.status = 0                 # currently not used

        # Trim the ends of the template spectra.
        # xxx self.trimTemplates (info["x_offset"], info["detector"])

        if not self.spec_found:
            for key in keys:
                self.spec_found[key] = True

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

        # the number of degrees of freedom is nelem - 1
        if self.nelem_dict.has_key (key):
            return (self.nelem_dict[key] - 1)
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
            self.nelem_dict
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
            spectrum = self.spectra[key]
            template = self.templates[key]
            (shift, n50) = self.findShift (spectrum, template)
            if shift is None:
                self.notFound (key)
                continue

            (factor, spec_slice, tmpl_slice) = \
                    self.computeNormalization (spectrum, template, shift)
            if factor is None:
                self.notFound (key)
                continue
            (chisq, spec, tmpl) = self.computeChiSquare (spectrum, template,
                                  factor, spec_slice, tmpl_slice)
            nelem = len (spec)
            self.chisq_dict[key] = chisq
            self.nelem_dict[key] = nelem
            self.spec_dict[key] = spec.copy()
            self.tmpl_dict[key] = tmpl.copy()
            ratio = chisq / nelem
            if shift is None or ratio > N_SIGMA or ratio < 1./N_SIGMA:
                shift = 0.
                self.spec_found[key] = False
            self.shift1_dict[key] = shift
            self.n50_dict[key] = n50

    def findShiftsNUV (self):
        """Find the shifts in the dispersion direction for NUV data."""

        global_shift = self.globalShift()
        if global_shift is None:
            global_shift = 0.

        factors = []
        for key in self.keys:
            if key not in self.templates.keys():
                self.notFound (key)
                continue
            spectrum = self.spectra[key]
            template = self.templates[key]
            (shift, n50) = self.findShift (spectrum, template)
            if shift is None or abs (shift - global_shift) > SLOP:
                self.spec_found[key] = False
            self.shift1_dict[key] = shift
            self.n50_dict[key] = n50

            (factor, spec_slice, tmpl_slice) = \
                    self.computeNormalization (spectrum, template, shift)
            if factor is None:
                self.notFound (key)
                continue
            if self.spec_found[key]:
                factors.append (factor)

        # Use the median factor in the loop below.
        n_factors = len (factors)
        if n_factors == 0:
            return
        factors.sort()
        n = n_factors // 2
        if n * 2 == n_factors:
            median_factor = (factors[n-1] + factors[n]) / 2.
        else:
            median_factor = factors[n]

        for key in self.keys:
            shift = self.shift1_dict[key]
            spectrum = self.spectra[key]
            template = self.templates[key]
            (factor, spec_slice, tmpl_slice) = \
                    self.computeNormalization (spectrum, template, shift)
            if factor is None:
                self.notFound (key)
                continue
            (chisq, spec, tmpl) = self.computeChiSquare (spectrum, template,
                                  median_factor, spec_slice, tmpl_slice)
            nelem = len (spec)
            self.chisq_dict[key] = chisq
            self.nelem_dict[key] = nelem
            self.spec_dict[key] = spec.copy()
            self.tmpl_dict[key] = tmpl.copy()
            ratio = chisq / nelem
            if shift is None or ratio > N_SIGMA or ratio < 1./N_SIGMA:
                self.spec_found[key] = False
                shift = 0.

        self.repairNUV()        # assign best-guess values for bad shifts

    def notFound (self, key):
        self.shift1_dict[key] = 0.
        self.spec_found[key] = False
        self.n50_dict[key] = None
        self.chisq_dict[key] = 0.
        self.nelem_dict[key] = 0
        self.spec_dict[key] = []
        self.tmpl_dict[key] = []

    def checkCounts (self):
        """Flag data with negligible counts.

        This function updates:
            self.spec_found
        """

        for key in self.keys:
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
        for key in self.keys:
            if self.templates.has_key (key):
                sum_spectra += self.spectra[key]
                sum_templates += self.templates[key]
        (global_shift, n50) = self.findShift (sum_spectra, sum_templates)

        return global_shift

    def findShift (self, spectrum, template):
        """Find a shift in the dispersion direction.

        @param spectrum: a 1-D extracted spectrum
        @type spectrum: array
        @param template: template spectrum
        @type template: array

        @return: the shift in the dispersion direction
        @rtype: float
        """

        initial_offset = -self.fp * self.stepsize

        (shift, n50) = self.crosscor (spectrum, template, initial_offset)

        return (shift, n50)

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
            x1 = lag - initial_offset
            x2 = x1 + length
            x1 = max (x1, 0)
            x2 = min (x2, length)
            t1 = -lag + initial_offset
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
            shift = location + index - initial_offset - maxlag

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

        # Ratio of counts in the spectrum to counts in the template.
        factors = []
        STEP = 100
        nelem = s1 - s0
        si = s0
        ti = t0
        done = False
        while not done:
            sj = si + STEP
            tj = ti + STEP
            sj = min (sj, nelem)
            tj = min (tj, nelem)
            sum_spec = spectrum[si:sj].sum()
            sum_tmpl = template[ti:tj].sum()
            si += STEP
            ti += STEP
            if si > s1 or ti > t1:
                done = True
            if sum_tmpl <= 0.:
                continue
            factors.append (sum_spec / sum_tmpl)
        factors.sort()
        n_factors = len (factors)
        if n_factors == 0:
            return (None, (s0, s1), (t0, t1))
        n = n_factors // 2
        if n * 2 == n_factors:
            factor = (factors[n-1] + factors[n]) / 2.
        else:
            factor = factors[n]

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

        @return: Chi square, the spectrum, and the normalized template
        @rtype: tuple
        """

        (s0, s1) = spec_slice
        (t0, t1) = tmpl_slice
        spec = spectrum[s0:s1]
        tmpl = template[t0:t1]

        # Normalize the template to match the spectrum.
        n_tmpl = tmpl * factor

        # Use mean_0_1 for the variance where the value in the spectrum is 0.
        ia = N.where (spec < 1.5)
        mean_0_1 = spec[ia].mean()
        if mean_0_1 <= 0.:
            mean_0_1 = 1.
        variance = N.where (spec <= 0., mean_0_1, spec)

        chisq = (spec - n_tmpl)**2 / variance
        chisq = float (chisq.sum())

        return (chisq, spec, n_tmpl)

    def repairNUV (self):
        """Assign reasonable values for shifts that weren't found."""

        # This is a rough estimate of the relative shifts between stripes.
        offset = {"NUVA": -1., "NUVB": 0., "NUVC": 1.}

        ngood = 0
        sum_shifts = 0.
        for key in self.keys:
            if self.spec_found[key]:
                sum_shifts += self.shift1_dict[key] + offset[key] * self.fp
                ngood += 1
        if ngood == 0:
            return              # no data; can't do anything

        mean_shift = sum_shifts / ngood
        for key in self.keys:
            if not self.spec_found[key]:
                self.shift1_dict[key] = mean_shift + offset[key] * self.fp
