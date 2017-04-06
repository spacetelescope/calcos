from __future__ import absolute_import, division         # confidence unknown
import numpy as np
from scipy import signal
from scipy import ndimage
from . import cosutil
from .calcosparam import *       # parameter definitions
from . import ccos

MASK_X = 189            # width of region to mask for each geocoronal line
SEARCH_Y = 91           # height of search region

# for comparison between values of fwhm
SIGNFICANTLY_LARGER = 2.        # pixels

# Lyman alpha, oxygen I, oxygen I
AIRGLOW_WAVELENGTHS = [1215.67, 1304., 1356.]

"""Info pertaining to geocoronal features:
For G130M, 1309, PSA, the image of geocoronal LyA is about 184 x 60 pixels.
The PSA and WCA apertures are about 100 pixels apart, for all FUV gratings.
NUVA and NUVB are closer, about 94 pixels for the M gratings.

The plate scale (arcsec per pixel) in the dispersion direction is roughly
the same for all gratings; it should be OK to mask a region 189 pixels wide,
centered on each geocoronal line.

A search range of 91 pixels seems reasonable (-45 to +45 pixels inclusive).

In t9h1220sl_phot.fits for NUV, the minimum wavelength for the M gratings
is 1648.58 Angstroms.  Wavelengths go down to 1213.77 for G230L, 2635,
but the throughput at short wavelengths is very low.
"""

def xdSearch(data, dq_data, wavelength, axis, slope, y_nominal,
             x_offset, detector):
    """Find the cross-dispersion location of the target spectrum.

    Parameters
    ----------
    data: array_like, 2-D
        SCI data from the flt file.

    dq_data: array_like, 2-D
        DQ data from the flt file.

    wavelength: array_like, 1-D
        Wavelength at each pixel (only needed for FUV).

    axis: int
        The dispersion axis, 0 (Y) or 1 (X).

    slope: float
        Slope of spectrum, pixels per pixel.

    y_nominal: float
        Location of spectrum at left edge of detector, i.e. at
        X = x_offset.

    x_offset: intls
        Offset of the detector in the data array.

    detector: str
        Detector name ("FUV" or "NUV").

    Returns
    -------
    (offset2, y_locn, y_locn_sigma, fwhm): tuple of four values
        offset2 is the offset of the spectrum from y_nominal (positive
        if the spectrum was found at a larger Y pixel number).  y_locn
        is the Y pixel number at which the spectrum was found (at pixel
        x_offset from the left edge of data).  y_locn_sigma is the
        error estimate for y_locn.  fwhm is the full-width at half
        maximum of the peak in the cross-dispersion profile; this can be
        either a float or an int.
    """

    (e_j, zero_point) = extractBand(data, dq_data, wavelength,
                                    axis, slope, y_nominal,
                                    x_offset, detector)

    box = 3

    (y_locn, y_locn_sigma, fwhm) = findPeak(e_j, box)

    if y_locn is None:
        offset2 = 0.
    else:
        # Shift y_locn to account for the offset of e_j from Y = 0 in 'data',
        # and shift y_locn to where the spectrum crosses X = x_offset.
        y_locn += zero_point
        y_locn += slope * float(x_offset)
        offset2 = y_locn - y_nominal

    return (offset2, y_locn, y_locn_sigma, fwhm)

def extractBand(data, dq_data, wavelength, axis, slope, y_nominal,
                x_offset, detector):
    """Extract a 2-D stripe centered on the nominal location of the target.

    Parameters
    ----------
    data: array_like
        SCI data from the flt file

    dq_data: array_like
        DQ data from the flt file

    wavelength: array_like
        Wavelength at each pixel (to locate the airglow lines)

    axis: int
        The dispersion axis, 0 (Y) or 1 (X)

    slope: float
        Slope of spectrum, pixels per pixel

    y_nominal: float
        Intercept of spectrum at left edge of detector

    x_offset: int
        Offset of the detector in the data array

    detector: str
        Detector name ("FUV" or "NUV")

    Returns
    -------
    tuple
        (e_j, zero_point), where e_j is a 1-D array containing a section of
        data collapsed along the dispersion direction and zero_point is
        the Y pixel number at the left edge of data corresponding to
        pixel 0 of e_j
    """

    extr_height = SEARCH_Y
    axis_length = data.shape[axis]
    e_ij = np.zeros((extr_height, axis_length), dtype=np.float32)
    ccos.extractband(data, axis, slope, y_nominal, x_offset, e_ij)

    # Clobber any region flagged as bad; note that this won't work well if a
    # flagged region covers part but not all of a spectral feature.
    if dq_data is not None:
        dq_ij = np.zeros((extr_height, axis_length), dtype=np.int16)
        ccos.extractband(dq_data, axis, slope, y_nominal, x_offset, dq_ij)
        dq = np.where(dq_ij == 0, 1, 0)
        e_ij *= dq

    if detector == "FUV":
        # Block out (i.e. set to zero) regions affected by airglow lines.
        for airglow in AIRGLOW_WAVELENGTHS:
            pixel_center = findPixelNumber(wavelength, airglow)
            pixel0 = pixel_center - (MASK_X // 2)
            pixel1 = pixel_center + (MASK_X // 2)
            if pixel1 < 0 or pixel0 >= axis_length:
                continue
            pixel0 = max(pixel0, 0)
            pixel1 = min(pixel1, axis_length-1)
            e_ij[:,int(pixel0):int(pixel1)] = 0.

    # sum the data along the dispersion direction
    e_j = e_ij.sum(axis=1)

    # Y pixel number in data corresponding to e_j[0]
    zero_point = int(round(y_nominal - slope * float(x_offset))) - \
                 SEARCH_Y // 2

    return (e_j, zero_point)

def findPixelNumber(wl, wavelength):
    """Find the nearest pixel to 'wavelength'.

    Parameters
    ----------
    wl: array_like, float64
        Wavelength at each pixel, assumed to be increasing

    wavelength: float
        A particular wavelength

    Returns
    -------
    int
        Pixel number closest to wavelength in the array wl
    """

    nelem = len(wl)

    dispersion = (wl[-1] - wl[0]) / float(nelem)
    if wavelength < wl[0]:
        x = (wavelength - wl[0]) / dispersion
        return int(round(x))
    elif wavelength >= wl[-1]:
        x = (wavelength - wl[-1]) / dispersion + float(nelem) - 1.
        return int(round(x))

    i0 = 0
    i1 = nelem - 1
    while (i1 - i0) >= 5:
        if i0 == i1:
            break
        slope = (wl[i1] - wl[i0]) / (i1 - i0)
        if slope == 0.:
            raise RuntimeError("Bad wavelength array.")
        mid = (i1 + i0) // 2
        x = int(round((wavelength - wl[mid]) / slope)) + mid
        dx = i1 - i0
        i0 = x - dx // 16
        i1 = x + dx // 16
        i0 = max(i0, 0)
        i1 = min(i1, nelem-1)

    x = i0
    diff = abs(wavelength - wl[x])
    for i in range(i0, i1+1):
        if abs(wavelength - wl[i]) < diff:
            x = i
            diff = abs(wavelength - wl[x])

    return x

def findPeak(e_j, box):
    """Find the location of the maximum within the subset.

    Note that the data were collapsed to the left edge to get e_j, so
    the location is the intercept on the edge, rather than where the
    spectrum crosses the middle of the detector or where it crosses
    X = x_offset.
    Also, e_j is not the full height of the detector, just a subset
    centered on the nominal Y location of the spectrum.

    Parameters
    ----------
    e_j: array_like
        1-D array of data collapsed along dispersion axis, taking into
        account the tilt of the spectrum

    box: int
        Smooth e_j with a box of this width before looking for the
        maximum

    Returns
    -------
    tuple
        The location (float) in the cross-dispersion direction relative
        to the first pixel in e_j, an estimate of the uncertainty in
        that location, and the FWHM of the peak in the cross-dispersion
        profile
    """

    boxcar_kernel = signal.boxcar(box) / box
    e_j_sm = ndimage.convolve(e_j, boxcar_kernel, mode="nearest")

    index = np.argsort(e_j_sm)
    ymax = index[-1]

    nelem = len(e_j)

    # This may be done again later, after we have found the location more
    # accurately.
    fwhm = findFwhm(e_j, ymax)

    # fit a quadratic to at least five points centered on ymax
    MIN_NPTS = 5
    npts = int(round(fwhm))
    npts = max(npts, MIN_NPTS)
    if npts // 2 * 2 == npts:
        npts += 1
    x = np.arange(nelem, dtype=np.float64)
    j1 = ymax - npts // 2
    j1 = max(j1, 0)
    j2 = j1 + npts
    if j2 > nelem:
        j2 = nelem
        j1 = j2 - npts
        j1 = max(j1, 0)
    (coeff, var) = cosutil.fitQuadratic(x[j1:j2], e_j_sm[j1:j2])

    (y_locn, y_locn_sigma) = cosutil.centerOfQuadratic(coeff, var)
    if y_locn is None:
        y_locn = ymax
        y_locn_sigma = 999.

    # Find the FWHM again if the location is far from the brightest pixel.
    if abs(y_locn - ymax) > fwhm / 4.:
        fwhm = findFwhm(e_j, y_locn)

    return (y_locn, y_locn_sigma, fwhm)

def findFwhm(e_j, y_locn):
    """Find the FWHM of the cross-dispersion profile of the spectrum.

    Two different approaches are used to find the FWHM.  The first method
    is to count the number of elements in the cross-dispersion profile with
    values above the half-maximum value; this value will be an integer.
    The second method is to follow the profile to the half-maximum value on
    either side of the maximum, using linear interpolation to get a better
    estimate of where the profile cuts across the half-maximum level; this
    value will be a float.  The value from the second method is expected to
    be more accurate if the target was actually found and has good
    signal-to-noise, so normally that value will be returned.  If the first
    method gives a significantly larger value, however, that value will be
    returned because it may indicate that the cross-dispersion profile is
    just noise.

    Parameters
    ----------
    e_j: array_like
        1-D array of data collapsed along dispersion axis

    y_locn: float
        The location in the cross-dispersion direction, relative to the
        first pixel in e_j.

    Returns
    -------
    float or int
        The full width half maximum of the peak in e_j.
    """

    nelem = len(e_j)
    y_locn_nint = int(round(y_locn))
    if y_locn_nint < 0 or y_locn_nint >= nelem:
        return -1.

    e_max = e_j[y_locn_nint]
    if e_max <= 0:
        return -1.

    e_j_sorted = np.sort(e_j)

    third = nelem // 3
    background = e_j_sorted[0:third].mean(dtype=np.float64)

    find_this_level = (e_max - background) / 2. + background

    # first estimate of FWHM
    # Count all elements in the sorted array that are greater than the
    # halfway level.  This will be large if the array is just noise
    # (at least, that's the idea).
    j = nelem - 1
    while j >= 0:
        if e_j_sorted[j] < find_this_level:
            break
        j -= 1
    fwhm_1 = nelem - 1 - j      # this is an int

    # second estimate of FWHM
    # Find where the cross-dispersion profile crosses the halfway level
    # on either side of the maximum.

    j_low = 0                   # initial values
    j_high = nelem - 1

    # first the low side
    j = y_locn_nint
    while j >= 0:
        if e_j[j] < find_this_level:
            j_low = j
            break
        j -= 1

    # Use linear interpolation to find where e_j would equal find_this_level.
    denom = e_j[j_low+1] - e_j[j_low]
    if denom == 0.:
        low = float(j_low) + 0.5        # 0.5 is an estimate
    else:
        low = float(j_low) + (find_this_level - e_j[j_low]) / denom

    # now the high side
    j = y_locn_nint
    while j < nelem:
        if e_j[j] < find_this_level:
            j_high = j
            break
        j += 1

    denom = e_j[j_high] - e_j[j_high-1]
    if denom == 0.:
        high = float(j_high) - 0.5
    else:
        high = float(j_high-1) + (find_this_level - e_j[j_high-1]) / denom

    fwhm_2 = high - low         # this is a float

    if fwhm_1 > fwhm_2 + SIGNFICANTLY_LARGER:
        fwhm = fwhm_1
    else:
        fwhm = fwhm_2

    return fwhm
