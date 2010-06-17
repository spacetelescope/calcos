from __future__ import division         # confidence unknown
import numpy as np
from convolve import boxcar
import cosutil
from calcosparam import *       # parameter definitions
import ccos

MASK_X = 189            # width of region to mask for each geocoronal line
SEARCH_Y = 91           # height of search region

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

def xdSearch (data, dq_data, wavelength, axis, slope, y_nominal,
              x_offset, detector):
    """Find the cross-dispersion location of the target spectrum.

    @param data: SCI data from the flt file
    @type data: 2-D numpy array
    @param dq_data: DQ data from the flt file
    @type dq_data: 2-D numpy array
    @param wavelength: wavelength at each pixel (only needed for FUV)
    @type wavelength: array
    @param axis: the dispersion axis, 0 (Y) or 1 (X)
    @type axis: int
    @param slope: slope of spectrum, pixels per pixel
    @type slope: float
    @param y_nominal: location of spectrum at left edge of detector,
        i.e. at X = x_offset
    @type y_nominal: float
    @param x_offset: offset of the detector in the data array
    @type x_offset: int
    @param detector: detector name ("FUV" or "NUV")
    @type detector: string

    @return: offset of the spectrum from y_nominal (positive if the spectrum
        was found at a larger Y pixel number); the Y pixel number at which
        the spectrum was found (at pixel x_offset from the left edge of
        'data'); the error estimate for y_locn
    @rtype: tuple of three floats
    """

    (e_j, zero_point) = extractBand (data, dq_data, wavelength,
                                     axis, slope, y_nominal,
                                     x_offset, detector)

    box = 3

    (y_locn, y_locn_sigma) = findPeak (e_j, box)

    # Shift y_locn to account for the offset of e_j from Y = 0 in 'data',
    # and shift y_locn to where the spectrum crosses X = x_offset.
    y_locn += zero_point
    y_locn += slope * float (x_offset)

    offset2 = y_locn - y_nominal

    return (offset2, y_locn, y_locn_sigma)

def extractBand (data, dq_data, wavelength, axis, slope, y_nominal,
                 x_offset, detector):
    """Extract a 2-D stripe centered on the nominal location of the target.

    @param data: SCI data from the flt file
    @type data: 2-D numpy array
    @param dq_data: DQ data from the flt file
    @type dq_data: 2-D numpy array
    @param wavelength: wavelength at each pixel (needed if find_target is True)
    @type wavelength: array
    @param axis: the dispersion axis, 0 (Y) or 1 (X)
    @type axis: int
    @param slope: slope of spectrum, pixels per pixel
    @type slope: float
    @param y_nominal: intercept of spectrum at left edge of detector
    @type y_nominal: float
    @param x_offset: offset of the detector in the data array
    @type x_offset: int
    @param detector: detector name ("FUV" or "NUV")
    @type detector: string

    @return: e_j, a 1-D array containing a section of 'data' collapsed along
        the dispersion direction; zero_point, the Y pixel number at the left
        edge of 'data' corresponding to pixel 0 of e_j
    @rtype: tuple
    """

    extr_height = SEARCH_Y
    axis_length = data.shape[axis]
    e_ij = np.zeros ((extr_height, axis_length), dtype=np.float32)
    ccos.extractband (data, axis, slope, y_nominal, x_offset, e_ij)

    # Clobber any region flagged as bad; note that this won't work well if a
    # flagged region covers part but not all of a spectral feature.
    if dq_data is not None:
        dq_ij = np.zeros ((extr_height, axis_length), dtype=np.int16)
        ccos.extractband (dq_data, axis, slope, y_nominal, x_offset, dq_ij)
        dq = np.where (dq_ij == 0, 1, 0)
        e_ij *= dq

    if detector == "FUV":
        # Block out (i.e. set to zero) regions affected by airglow lines.
        for airglow in AIRGLOW_WAVELENGTHS:
            pixel_center = findPixelNumber (wavelength, airglow)
            pixel0 = pixel_center - (MASK_X // 2)
            pixel1 = pixel_center + (MASK_X // 2)
            if pixel1 < 0 or pixel0 >= axis_length:
                continue
            pixel0 = max (pixel0, 0)
            pixel1 = min (pixel1, axis_length-1)
            e_ij[:,pixel0:pixel1] = 0.

    # sum the data along the dispersion direction
    e_j = e_ij.sum (axis=1)

    # Y pixel number in data corresponding to e_j[0]
    zero_point = int (round (y_nominal - slope * float (x_offset))) - \
                 SEARCH_Y // 2

    return (e_j, zero_point)

def findPixelNumber (wl, wavelength):
    """Find the nearest pixel to 'wavelength'.

    @param wl: wavelength at each pixel, assumed to be increasing
    @type wl: array of float64
    @param wavelength: a particular wavelength
    @type wavelength: float

    @return: pixel number closest to 'wavelength' in the array 'wl'
    @rtype: int
    """

    dispersion = (wl[-1] - wl[0]) / float (len (wl))
    if wavelength < wl[0]:
        x = (wavelength - wl[0]) / dispersion
        return int (round (x))
    elif wavelength >= wl[-1]:
        x = (wavelength - wl[-1]) / dispersion + float (len (wl)) - 1.
        return int (round (x))

    i0 = 0
    i1 = len (wl) - 1
    while (i1 - i0) >= 5:
        if i0 == i1:
            break
        slope = (wl[i1] - wl[i0]) / (i1 - i0)
        if slope == 0.:
            raise RuntimeError, "Bad wavelength array."
        mid = (i1 + i0) // 2
        x = int (round ((wavelength - wl[mid]) / slope)) + mid
        dx = i1 - i0
        i0 = x - dx // 8
        i1 = x + dx // 8

    x = i0
    diff = abs (wavelength - wl[x])
    for i in range (i0, i1+1):
        if abs (wavelength - wl[i]) < diff:
            x = i
            diff = abs (wavelength - wl[x])

    return x

def findPeak (e_j, box):
    """Find the location of the maximum within the subset.

    @param e_j: 1-D array of data collapsed along dispersion axis,
        taking into account the tilt of the spectrum
    @type e_j: array
    @param box: smooth e_j with a box of this width before looking
        for the maximum
    @type box: int

    @return: The location (float) in the cross-dispersion direction
        relative to the first pixel in e_j, and an estimate of the
        uncertainty in that location
    @rtype: tuple

    Note that the data were collapsed to the left edge to get e_j, so the
    location is the intercept on the edge, rather than where the spectrum
    crosses the middle of the detector or where it crosses X = x_offset.
    Also, e_j is not the full height of the detector, just a subset centered
    on the nominal Y location of the spectrum.
    """

    e_j_sm = boxcar (e_j, (box,), mode="nearest")

    index = np.argsort (e_j_sm)
    ymax = index[-1]

    nelem = len (e_j)

    # fit a quadratic to five points centered on ymax
    NPTS = 5
    x = np.arange (nelem, dtype=np.float64)
    j1 = ymax - NPTS // 2
    j1 = max (j1, 0)
    j2 = j1 + NPTS
    j2 = min (j2, nelem)
    j1 = j2 - NPTS
    (coeff, var) = cosutil.fitQuadratic (x[j1:j2], e_j_sm[j1:j2])

    (y_locn, y_locn_sigma) = cosutil.centerOfQuadratic (coeff, var)

    return (y_locn, y_locn_sigma)
