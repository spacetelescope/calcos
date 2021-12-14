from __future__ import absolute_import, division         # confidence unknown
import math
import os
import numpy as np
import astropy.io.fits as fits
from . import cosutil
from . import dispersion
from . import orbit
from . import ccos
from .calcosparam import *       # parameter definitions

DIST_SUN  = 149597870.691       # 1 AU in km
RADIUS_EARTH = 6.367456e3       # geometric mean of equatorial and polar, km
SEC_PER_DAY = 86400.
MREF = 2451545. - 2400000.5     # MJD for 2000 Jan 1, 12h

TWOPI     = 2. * math.pi
ASECtoRAD = math.pi / (180. * 3600.)    # radians per arcsecond
DEGtoRAD  = math.pi / 180.              # radians per degree

# Wavelengths in Angstroms of airglow lines Lyman alpha, oxygen I, oxygen I.
# The values in the tuple are the wavelengths of the lines in the multiplet.
AIRGLOW_WAVELENGTHS = {"ly_alpha": (1215.67,),
                       "oi_1304": (1302.2, 1304.9, 1306.0),
                       "oi_1356": (1355.6, 1358.5),
                       "dark": None}

def createTimeline(input, fd, info, reffiles,
                   tl_time, shift1_vs_time,
                   time, xfull, yfull):
    """Create (or update) a timeline table.

    Parameters
    ----------
    input: str
        Name of input file (used only for creating spt file name).

    fd: pyfits HDUList object
        List of HDUs in corrtag file.

    info: dictionary
        Keywords and values.

    reffiles: dictionary
        Reference file keywords and names.

    tl_time: array_like
        The array of times corresponding to shift1_vs_time.

    shift1_vs_time: array_like
        The shifts in the dispersion direction at one-second intervals.
        or None if the current observation is a wavecal or if wavecal
        processing was not done.

    time: array_like
        The array of times of events in the corrtag table.

    xfull: array_like
        The array of fully corrected X positions, from the corrtag table.

    yfull: array_like
        The array of fully corrected Y positions, from the corrtag table.
    """

    # does the timeline extension already exist?
    try:
        hdu = fd[("timeline",1)]
        cosutil.printMsg("Update the TIMELINE extension.", VERBOSE)
    except KeyError:
        # timeline table doesn't exist; create it and append to corrtag file
        hdu = timelineHDU(len(tl_time), fd[1].header)
        cosutil.printMsg("Append a TIMELINE extension.", VERBOSE)
        fd.append(hdu)

    tl_data = hdu.data

    time_col = tl_data.field("time")
    if len(time_col) != len(tl_time):
        cosutil.printWarning("Number of rows in TIMELINE extension "
                "is %d, expected %d," % (len(time_col), len(tl_time)))
        cosutil.printContinuation("so TIMELINE table will not be updated.")
        return
    time_col[:] = tl_time.copy()

    if shift1_vs_time is None:
        median_shift1 = 0.
    else:
        shift1_col = tl_data.field("shift1")
        shift1_col[:] = shift1_vs_time.copy()
        index = shift1_vs_time.argsort()
        nelem = len(shift1_vs_time)
        mid_index = index[nelem // 2]
        median_shift1 = shift1_vs_time.item(mid_index)

    external_target = info["exptype"][:3] == "EXT"

    # columns for airglow count rates and dark counts
    ly_alpha_col = tl_data.field("ly_alpha")
    oi_1304_col = tl_data.field("oi_1304")
    oi_1356_col = tl_data.field("oi_1356")
    darkrate_col = tl_data.field("darkrate")
    # initialize to zero
    ly_alpha_col[:] = 0.
    oi_1304_col[:] = 0.
    oi_1356_col[:] = 0.
    darkrate_col[:] = 0.
    exptime = info["exptime"]
    if external_target and info["aperture"] in ["PSA", "BOA"] and \
       exptime > 0. and xfull is not None and yfull is not None:
        if len(tl_time) > 1:
            dt = tl_time[1] - tl_time[0]
        else:
            dt = 1.
        for key in ["ly_alpha", "oi_1304", "oi_1356", "dark"]:
            wl_airglow = AIRGLOW_WAVELENGTHS[key]
            region = findPixelRegion(info, reffiles["disptab"],
                                     reffiles["xtractab"],
                                     median_shift1, wl_airglow)
            if region is None:
                continue
            (y0, y1, x0, x1) = region
            if key == "ly_alpha" or key == "oi_1304":
                cosutil.printMsg("Airglow region for %s is "
                                 "X: %d to %d, Y: %d to %d" %
                                 (key.upper(), x0, x1+1, y0, y1+1), VERBOSE)
            # A value of 1 (True) in region_flags means the corresponding
            # event is within the area that includes the airglow line.
            region_flags = np.ones(len(xfull), dtype=np.bool8)
            region_flags = np.where(xfull > x1, False, region_flags)
            region_flags = np.where(xfull < x0,  False, region_flags)
            if isinstance(y0, (list, tuple)):
                # for the dark, there are two y regions
                region_flags = np.where(yfull > y1[1], False, region_flags)
                region_flags = np.where(yfull < y0[0],  False, region_flags)
                between = np.logical_and(yfull > y1[0], yfull < y0[1])
                region_flags = np.where(between,  False, region_flags)
                npixels = (x1 - x0) * (y1[0] - y0[0] + y1[1] - y0[1])
            else:
                region_flags = np.where(yfull > y1, False, region_flags)
                region_flags = np.where(yfull < y0,  False, region_flags)
                npixels = 1.
            region_flags = region_flags.astype(np.bool8)
            # scratch array for counts per second within each time bin
            temp = np.zeros(len(tl_time), dtype=np.float32)
            if time[-1] - time[0] < 1.:         # e.g. ACCUM data
                temp[:] = float(region_flags.sum(dtype=np.int32)) / exptime
            else:
                for i in range(len(tl_time)):
                    # jt0 and jt1 are indices in the TIME column, and therefore
                    # also in region_flags.
                    try:
                        (jt0, jt1) = ccos.range(time,
                                                tl_time[i], tl_time[i]+dt)
                        temp[i] = float(
                        (region_flags[jt0:jt1]).sum(dtype=np.int32)) / dt
                    except RuntimeError:
                        temp[i] = 0.
            if key == "ly_alpha":
                ly_alpha_col[:] = temp
            elif key == "oi_1304":
                oi_1304_col[:] = temp
            elif key == "oi_1356":
                oi_1356_col[:] = temp
            elif key == "dark":
                darkrate_col[:] = temp / npixels        # dark rate per pixel

    sptfile = makeSptFileName(input)
    if sptfile == "notfound":
        found_spt = False
    else:
        found_spt = os.access(sptfile, os.R_OK)
    if not found_spt:
        cosutil.printWarning("spt file not found, so TIMELINE extension "
                             "is incomplete")
        return

    orb = orbit.HSTOrbit(sptfile)

    sun_alt_col = tl_data.field("sun_alt")
    sun_zd_col = tl_data.field("sun_zd")
    long_col = tl_data.field("longitude")
    lat_col = tl_data.field("latitude")

    if external_target:
        ra_targ = info["ra_targ"] * DEGtoRAD
        dec_targ = info["dec_targ"] * DEGtoRAD
        rect_targ = sphToRect((1., ra_targ, dec_targ))      # unit vector
        target_alt_col = tl_data.field("target_alt")
        rv_col = tl_data.field("radial_vel")

    for i in range(len(tl_time)):
        mjd = tl_time[i] / SEC_PER_DAY + info["expstart"]
        (rect_hst, vel_hst) = orb.getPos(mjd)
        (r, ra_hst, dec_hst) = rectToSph(rect_hst)
        # Assume that we want geocentric latitude.  The difference from
        # astronomical latitude can be up to about 8.6 arcmin.
        lat_hst = dec_hst
        # Subtract the sidereal time at Greenwich to convert to longitude.
        long_hst = ra_hst - 2. * math.pi * gmst(mjd)
        if long_hst < 0.:
            long_hst += (2. * math.pi)
        long_col[i] = long_hst / DEGtoRAD
        lat_col[i] = lat_hst / DEGtoRAD
        rect_sun = eqSun(mjd)                   # equatorial coords of the Sun
        sun_alt_col[i] = computeAlt(rect_sun, rect_hst, parallax=True)
        sun_zd_col[i] = computeZD(rect_sun, rect_hst)
        if external_target:
            rv_col[i] = -dotProduct(rect_targ, vel_hst)
            target_alt_col[i] = computeAlt(rect_targ, rect_hst,
                                           parallax=False)

def makeSptFileName(input):
    """Construct the spt file name from the corrtag file name.

    Parameters
    ----------
    input: str
        Name of input file

    Returns
    --------
    str
        Name of support file, or "notfound".
    """
    pathname = os.path.dirname(input)
    filename = os.path.basename(input)
    rootname = filename.split("_")[0]
    sptfile = os.path.join(pathname, rootname + "_spt.fits")
    if not os.path.isfile:
        sptfile = "notfound"

    return sptfile

def sphToRect(sph):
    """Convert distance, RA and Dec to rectangular coordinates.

    Parameters
    ----------
    sph: array_like
        Distance (may be 1), longitude, latitude (angles in radians).

    Returns
    --------
    array_like
        Vector in rectangular coordinates.
    """

    (radius, longitude, latitude) = sph
    rect = np.zeros(3, dtype=np.float64)

    rect[0] = radius * math.cos(latitude) * math.cos(longitude)
    rect[1] = radius * math.cos(latitude) * math.sin(longitude)
    rect[2] = radius * math.sin(latitude)

    return rect

def rectToSph(rect):
    """Convert rectangular coordinates to RA and Dec.

    Parameters
    ----------
    rect: array_like
        Vector in rectangular coordinates.

    Returns
    --------
    array_like
        Distance, longitude, latitude (angles in radians).
    """

    # radius, longitude, latitude
    sph = np.zeros(3, dtype=np.float64)

    r2xy = rect[0] * rect[0] + rect[1] * rect[1]
    sph[0] = math.sqrt(r2xy + rect[2] * rect[2])
    rxy = math.sqrt(r2xy)
    sph[2] = math.atan2(rect[2], rxy)
    if r2xy > 0.:
        sph[1] = math.atan2(rect[1], rect[0])
        if sph[1] < 0.:
            sph[1] += (2. * math.pi)
    else:
        sph[1] = 0.

    return sph

def dotProduct(v1, v2):
    product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    return product.item()

def gmst(mjd):
    """Greenwich mean sidereal time at mjd.

    See Arthur N. Cox, Allen's Astrophysical Quantities, 4th ed., p 14.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    float
        The sidereal time at Greenwich, unit is fraction of a day.
    """

    d = mjd - MREF                      # days since MREF
    du = math.floor(mjd) - MREF
    Tu = du / 36525.
    # Greenwich mean sidereal time at 0h UT1, in seconds.
    GMST0 = 24110.54841 + 8640184.812866 * Tu \
                        + 0.093104 * Tu**2 - 6.2e-6 * Tu**3

    T = (mjd - MREF) / 36525.
    rprime = 1.002737909350795 + 5.9006e-11 * T - 5.9e-15 * T**2

    GMST = GMST0 / SEC_PER_DAY + (d - du) * rprime      # days
    t = GMST % 1.

    return t

def timelineHDU(nrows_timeline, hdr):
    """Create the TIMELINE HDU.

    Parameters
    ----------
    nrows_timeline: int
        Number of rows for the timeline table.

    hdr: pyfits Header object
        EVENTS extension header.

    Returns
    -------
    pyfits BinTableHDU object
        Header/data unit for a timeline table.
    """

    col = []
    col.append(fits.Column(name="TIME", format="1E",
                           unit="s", disp="F8.3"))
    col.append(fits.Column(name="LONGITUDE", format="1E",
                           unit="degree", disp="F10.6"))
    col.append(fits.Column(name="LATITUDE", format="1E",
                           unit="degree", disp="F10.6"))
    col.append(fits.Column(name="SUN_ALT", format="1E",
                           unit="degree", disp="F6.2"))
    col.append(fits.Column(name="SUN_ZD", format="1E",
                           unit="degree", disp="F6.2"))
    col.append(fits.Column(name="TARGET_ALT", format="1E",
                           unit="degree", disp="F6.2"))
    col.append(fits.Column(name="RADIAL_VEL", format="1E",
                           unit="km /s", disp="F7.5"))
    col.append(fits.Column(name="SHIFT1", format="1E",
                           unit="pixel", disp="F7.3"))
    col.append(fits.Column(name="LY_ALPHA", format="1E",
                           unit="count /s", disp="G15.6"))
    col.append(fits.Column(name="OI_1304", format="1E",
                           unit="count /s", disp="G15.6"))
    col.append(fits.Column(name="OI_1356", format="1E",
                           unit="count /s", disp="G15.6"))
    col.append(fits.Column(name="DARKRATE", format="1E",
                           unit="count /s /pixel", disp="G15.6"))
    cd = fits.ColDefs(col)

    #
    # Remove WCS keywords from table header
    newheader = cosutil.remove_WCS_keywords(hdr, cd)

    hdu = fits.BinTableHDU.from_columns(cd, header=newheader, nrows=nrows_timeline)

    hdu.header.set("extname", "TIMELINE", after="TFIELDS")      # xxx temp
    hdu.header.set("extver", 1, after="EXTNAME")        # xxx temporary
    # xxx hdu.header.insert("TFIELDS", ("extname", "TIMELINE"), after=True)
    # xxx hdu.header.insert("EXTNAME", ("extver", 1), after=True)

    return hdu

def computeAlt(rect, rect_hst, parallax=False):
    """Compute the altitude of an object, e.g. the Sun.

    This is based on the GC_sun_alt function, gc_sun_alt.cpp in OPUS.

    Parameters
    ----------
    rect: array like
        Rectangular, geocentric coordinates of an object.  If parallax due
        to the orbit of HST is significant, this vector should be in km
        (see parallax); otherwise, this may be a unit vector pointing
        toward the object.

    rect_hst: array like
        Rectangular, geocentric coordinates of HST, in km.

    parallax: boolean
        True if the object is close enough that we should correct for
        the parallax due to HST's offset from the center of the Earth.
        If True, rect should be in km; otherwise, rect may be a
        unit vector.

    Returns
    -------
    float
        Altitude of the object above the horizon, as seen by HST, in
        degrees.
    """

    if parallax:
        target = rect - rect_hst        # shift origin to HST
    else:
        target = rect
    rtarget = math.sqrt(dotProduct(target, target))
    # unit vector pointing from HST toward the object
    utarget = target / rtarget

    rhst = math.sqrt(dotProduct(rect_hst, rect_hst))
    # unit vector pointing from the center of the Earth toward HST
    uhst = rect_hst / rhst

    cz = dotProduct(uhst, utarget)      # cosine zenith distance
    zenith_dist = math.acos(cz)

    # The horizon is more than 90 degrees from the zenith; add this extra
    # angle.
    horizon_corr = math.acos(RADIUS_EARTH / rhst)
    altitude = math.pi / 2 + horizon_corr - zenith_dist

    return altitude / DEGtoRAD

def computeZD(rect_sun, rect_hst):
    """Compute the zenith distance (ignoring parallax) of the Sun.

    Parameters
    ----------
    rect_sun: array like
        Rectangular, geocentric coordinates of an object, in particular,
        the Sun.  The units are arbitrary because the vector will be
        normalized to 1.

    rect_hst: array like
        Rectangular, geocentric coordinates of HST.  The units are
        arbitrary because the vector will be normalized to 1.

    Returns
    -------
    float
        The angle between HST and the Sun, as seen from the center of
        the Earth, in degrees.
    """

    rsun = math.sqrt(dotProduct(rect_sun, rect_sun))
    usun = rect_sun / rsun

    rhst = math.sqrt(dotProduct(rect_hst, rect_hst))
    uhst = rect_hst / rhst

    # zenith_dist is not quite the zenith distance, because parallax
    # is not accounted for.
    cz = dotProduct(uhst, usun)
    zenith_dist = math.acos(cz)

    return zenith_dist / DEGtoRAD

def findPixelRegion(info, disptab, xtractab, median_shift1, wl_airglow):
    """Find the pixel region corresponding to an airglow line (or dark).

    Parameters
    ----------
    info: dictionary
        Keywords and values.

    disptab: str
        Name of reference table for dispersion solution.

    xtractab: str
        Name of extraction parameters reference table.

    median_shift1: float
        Median value in shift1_vs_time (in createTimeline()).  This is
        used when testing whether an airglow line is on the detector, and
        for shifting the region for the dark count rate.

    wl_airglow: tuple of floats, or None
        The elements of the tuple are the wavelengths of the airglow line
        or lines (i.e. in a multiplet), in Angstroms.
        If the region is for the dark count rate, wl_airglow will be None.

    Returns
    -------
    tuple
        (y0, y1, x0, x1)  x0 and x1 are floats; y0 and y1 are floats for an
        airglow line, or lists of two floats (each) for background regions.
        These are inclusive limits (pixels), not the elements of a slice.
        For an airglow line, y0 and y1 are the lower and upper limits of
        the airglow region, x0 and x1 are the left and right limits of the
        airglow region.
        For dark counts, x0 to x1 will cover the full width of the
        detector.  y0 and y1 are two-element lists; y0[0] and y0[1] are the
        lower limits of the two background regions, while y1[0] and y1[1]
        are the upper limits of the two background regions.
    """

    if info["obstype"] == "IMAGING":
        return None

    if info["detector"] == "FUV":
        segment = info["segment"]
        axis_length = FUV_X
        AIRGLOW_WIDTH = 184
        AIRGLOW_HEIGHT = 60
    else:
        segment = "NUVA"
        axis_length = NUV_X
        AIRGLOW_WIDTH = 60
        AIRGLOW_HEIGHT = 60

    x_width = float(AIRGLOW_WIDTH // 2)
    y_width = float(AIRGLOW_HEIGHT // 2)
    # fpoffset will be added to filter for the disptab.
    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "segment": segment,
              "aperture": info["aperture"]}
    xtract_info = cosutil.getTable(xtractab, filter)
    if xtract_info is None:
        cosutil.printWarning("No matching row in the XTRACTAB; filter is:")
        cosutil.printContinuation(str(filter))
        return None

    if wl_airglow is None:
        # Regions for dark count rate.
        x0 = 0. - median_shift1
        x1 = float(axis_length) - median_shift1
        b_bkg1 = xtract_info.field("b_bkg1")[0]
        b_bkg2 = xtract_info.field("b_bkg2")[0]
        if cosutil.findColumn(xtract_info, "b_hgt1"):
            bkg_height1 = xtract_info.field("b_hgt1")[0]
            bkg_height2 = xtract_info.field("b_hgt2")[0]
        else:
            bkg_height1 = xtract_info.field("bheight")[0]
            bkg_height2 = bkg_height1
        y0_low  = b_bkg1 - bkg_height1 // 2
        y0_high = b_bkg1 + bkg_height1 // 2
        y1_low  = b_bkg2 - bkg_height2 // 2
        y1_high = b_bkg2 + bkg_height2 // 2
        y0 = [y0_low, y1_low]
        y1 = [y0_high, y1_high]
        y0.sort()
        y1.sort()
    else:
        # Region for an airglow line.
        filter["fpoffset"] = info["fpoffset"]
        disp_rel = dispersion.Dispersion(disptab, filter)
        min_wl = min(wl_airglow)
        max_wl = max(wl_airglow)
        # First check whether the airglow line is off the detector.
        # NOTE that we assume that wavelength increases with x.
        wl_left_edge = disp_rel.evalDisp(-x_width - median_shift1)
        if max_wl < wl_left_edge:
            disp_rel.close()
            return None
        wl_right_edge = disp_rel.evalDisp(axis_length - 1. +
                                          x_width - median_shift1)
        if min_wl > wl_right_edge:
            disp_rel.close()
            return None
        # x_left and x_right are the pixel coordinates for the minimum
        # and maximum airglow wavelengths respectively.
        x_left = float(disp_rel.evalInvDisp(min_wl, tiny=1.e-8))
        x_right = float(disp_rel.evalInvDisp(max_wl, tiny=1.e-8))
        x0 = x_left - x_width
        x1 = x_right + x_width
        disp_rel.close()
        slope = xtract_info.field("slope").item(0)
        b_spec = xtract_info.field("b_spec").item(0)
        y = b_spec + slope * (x_left + x_right) / 2.
        y0 = y - y_width
        y1 = y + y_width

    return (y0, y1, x0, x1)

def eqSun(mjd):
    """Compute the equatorial coordinates of the Sun.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    rect_sun: array_like
        Three-element vector containing the equatorial rectangular
        coordinates of the Sun, in km.
    """

    ecl_sun = eclSun(mjd)               # ecliptic coordinates of the Sun
    rect_sun = eclToEq(ecl_sun, mjd)    # convert to equatorial coordinates
    rect_sun *= DIST_SUN                # convert distance from AU to km

    return rect_sun

def eclSun(mjd):
    """Compute the ecliptic coordinates of the Sun.

    This is based on Pulkinnen & VanFlandern, ApJ Suppl Series, vol. 41,
    Nov. 1979, p. 391-41.  The accuracy is of order a minute of arc (but
    often much better), as seen from Earth.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    eclcoord: array_like
        Three-element vector containing the ecliptic rectangular
        coordinates of the Sun at mjd, in astronomical units.
    """

    eclcoord = np.zeros(3, dtype=np.float64)

    tJcent = (mjd - MREF) / 36525. + 1.         # Julian centuries

    ls = lsun(mjd)
    gs = gsun(mjd)
    lm = lmoon(mjd)
    gv2 = gvenus(mjd)
    gm4 = gmars(mjd)
    gj5 = gjupiter(mjd)

    sings = math.sin(gs)
    cosgs = math.cos(gs)
    sin2gs = 2. * sings * cosgs
    cos2gs = cosgs * cosgs - sings * sings

    ecllong = ls + (6910. * sings + 72. * sin2gs - 17. * tJcent * sings \
                     - 7. * math.cos(gs - gj5) \
                     + 6. * math.sin(lm - ls) \
                     + 5. * math.sin(4.*gs - 8.*gm4 + 3.*gj5) \
                     - 5. * math.cos(2.*gs - 2.*gv2) \
                     - 4. * math.sin(gs - gv2) \
                     + 4. * math.cos(4.*gs - 8.*gm4 + 3.*gj5) \
                     + 3. * math.sin(2.*gs - 2.*gv2) \
                     - 3. * math.sin(gj5) \
                     - 3. * math.sin(2.*gs - 2.*gj5)) * ASECtoRAD

    dist = 1.00014 - 0.01675 * cosgs - 0.00014 * cos2gs

    eclcoord[0] = dist * math.cos(ecllong)
    eclcoord[1] = dist * math.sin(ecllong)
    eclcoord[2] = 0.

    return eclcoord

def eclToEq(eclcoord, mjd):
    """Convert from ecliptic coordinates to equatorial coordinates.

    Parameters
    ----------
    eclcoord: array_like
        Three-element vector containing ecliptic rectangular coordinates.

    mjd: float
        The Modified Julian Date.

    Returns
    -------
    eqcoord: array_like
        Three-element vector containing the equatorial rectangular
        coordinates corresponding to eclcoord at mjd.
    """

    eqcoord = np.zeros(3, dtype=np.float64)

    tJcent = (mjd - MREF) / 36525. + 1.         # Julian centuries

    omega = momega(mjd)         # longitude of the ascending node of the moon

    eps = (84428. - 47. * tJcent + 9. * math.cos(omega)) * ASECtoRAD
    coseps = math.cos(eps)
    sineps = math.sin(eps)

    # Rotate around x-axis by the obliquity of the ecliptic.
    eqcoord[0] = eclcoord[0]
    eqcoord[1] = coseps * eclcoord[1] - sineps * eclcoord[2]
    eqcoord[2] = sineps * eclcoord[1] + coseps * eclcoord[2]

    return eqcoord

def lsun(mjd):
    """Compute the mean longitude of the Sun.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    float
        The mean longitude of the Sun at mjd, in radians.
    """

    return TWOPI * (0.779072 + 0.00273790931 * (mjd - MREF))

def gsun(mjd):
    """Compute the mean anomaly of the Sun.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    float
        The mean anomaly of the Sun at mjd, in radians.
    """

    return TWOPI * (0.993126 + 0.00273777850 * (mjd - MREF))

def lmoon(mjd):
    """Compute the mean longitude of the Moon.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    float
        The mean longitude of the Moon at mjd, in radians.
    """

    return TWOPI * (0.606434 + 0.03660110129 * (mjd - MREF))

def momega(mjd):
    """Compute L_moon - F_moon for the Moon.

    L_moon is the mean longitude of the Moon, and F_moon is the argument
    of latitude of the Moon.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    float
        L_moon - F_moon at mjd, in radians.
    """

    return TWOPI * (0.347343 - 0.00014709391 * (mjd - MREF))

def gvenus(mjd):
    """Compute the mean anomaly for Venus.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    float
        The mean anomaly for Venus at mjd, in radians.
    """

    return TWOPI * (0.140023 + 0.00445036173 * (mjd - MREF))

def gmars(mjd):
    """Compute the mean anomaly for Mars.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    float
        The mean anomaly for Mars at mjd, in radians.
    """

    return TWOPI * (0.053856 + 0.00145561327 * (mjd - MREF))

def gjupiter(mjd):
    """Compute the mean anomaly for Jupiter.

    Parameters
    ----------
    mjd: float
        The Modified Julian Date.

    Returns
    -------
    float
        The mean anomaly for Jupiter at mjd, in radians.
    """

    return TWOPI * (0.056531 + 0.00023080893 * (mjd - MREF))
