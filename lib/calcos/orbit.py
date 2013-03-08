from __future__ import division         # confidence unknown
import math
import numpy as np
import astropy.io.fits as fits

TWOPI       = 2. * math.pi
SEC_PER_DAY = 86400.0

class HSTOrbit(object):
    """Orbital parameters.

    The public methods are getOrbitper and getPos.

    This was originially written in IDL (hst_pos_mjd.pro) by Tom Ake.

    Parameters
    ----------
    sptfile: str
        The name of the support file (rootname_spt.fits).
    """

    def __init__(self, sptfile):
        """Constructor."""

        # attributes
        self.argperig = 0.  # argument of perigee (revolutions)
        self.cirveloc = 0.  # circular orbit linear velocity (meters/second)
        self.cosincli = 0.  # cosine of inclination
        self.ecbdx3   = 0.  # eccentricity cubed times 3
        self.eccentry = 0.  # eccentricity
        self.eccentx2 = 0.  # eccentricity times 2
        self.ecbdx4d3 = 0.  # eccentricity cubed times 4/3
        self.epchtime = 0.  # epoch time of parameters (secs since 1/1/85)
        self.esqdx5d2 = 0.  # eccentricity squared times 5/2
        self.fdmeanan = 0.  # 1st derivative coef for mean anomly (revs/sec)
        self.hsthorb  = 0.  # half the duration of the ST orbit (seconds)
        self.meananom = 0.  # mean anomaly (radians)
        self.rascascn = 0.  # right ascension of ascending node (revolutions)
        self.rcargper = 0.  # rate change of argument of perigee (revs/sec)
        self.rcascnrv = 0.  # rt chge right ascension ascend node (revs/sec)
        self.sdmeanan = 0.  # 2nd deriv coef for mean anomaly (revs/sec/sec)
        self.semilrec = 0.  # semi-latus rectum (meters)
        self.sineincl = 0.  # sine of inclination

        self._readOrbitalParameters(sptfile)

    def _readOrbitalParameters(self, sptfile):
        """Get the orbital parameters from the spt primary header.

        Parameters
        ----------
        sptfile: str
            The name of the support file.
        """

        fd = fits.open(sptfile, mode="readonly")
        phdr = fd[0].header

        # Orbital elements for HST.
        self.argperig = phdr["argperig"]
        self.cirveloc = phdr["cirveloc"]
        self.cosincli = phdr["cosincli"]
        self.ecbdx3   = phdr["ecbdx3"]
        self.eccentry = phdr["eccentry"]
        self.eccentx2 = phdr["eccentx2"]
        self.ecbdx4d3 = phdr["ecbdx4d3"]
        self.epchtime = phdr["epchtime"]
        self.esqdx5d2 = phdr["esqdx5d2"]
        self.fdmeanan = phdr["fdmeanan"]
        self.hsthorb  = phdr["hsthorb"]
        self.meananom = phdr["meananom"]
        self.rascascn = phdr["rascascn"]
        self.rcargper = phdr["rcargper"]
        self.rcascnrv = phdr["rcascnrv"]
        self.sdmeanan = phdr["sdmeanan"]
        self.semilrec = phdr["semilrec"]
        self.sineincl = phdr["sineincl"]

        fd.close()

    def getOrbitper(self):
        """Return the orbital period.

        Returns
        -------
        float
            The orbital period in seconds.
        """

        return 2. * self.hsthorb

    def getPos(self, mjd):
        """Get position and velocity at a given time.

        # S. Hulbert, Oct 91    Original
        # PEH, 2008 Oct 3       Converted from SPP to Python

        Parameters
        ----------
        mjd: float
            The time (MJD) at which to compute the position and velocity.

        Returns
        -------
        tuple of two array_like
            The first array is the position vector (km), the second array
            is the velocity vector (km/s).
        """

        # These will be returned, after assigning the actual values.
        x_hst = np.zeros(3, dtype=np.float64)
        v_hst = np.zeros(3, dtype=np.float64)

        argperig = self.argperig
        cirveloc = self.cirveloc
        cosincli = self.cosincli
        ecbdx3   = self.ecbdx3
        eccentry = self.eccentry
        eccentx2 = self.eccentx2
        ecbdx4d3 = self.ecbdx4d3
        epchtime = self.epchtime
        esqdx5d2 = self.esqdx5d2
        fdmeanan = self.fdmeanan
        hsthorb  = self.hsthorb
        meananom = self.meananom
        rascascn = self.rascascn
        rcargper = self.rcargper
        rcascnrv = self.rcascnrv
        sdmeanan = self.sdmeanan
        semilrec = self.semilrec
        sineincl = self.sineincl

        # convert time from MJD to seconds since 1985 Jan 1
        sec85 = (mjd - 46066.0) * SEC_PER_DAY

        # calculate time difference between observation and epoch time
        deltim = sec85 - epchtime

        # mean anomaly
        temp2 = fdmeanan * deltim
        temp3 = 0.5 * sdmeanan * deltim*deltim
        m = meananom + TWOPI * (temp2 + temp3)

        sin_m = math.sin(m)
        cos_m = math.cos(m)

        # true anomaly (equation of the center)
        v = m + sin_m * (eccentx2 + ecbdx3 * cos_m * cos_m -
                ecbdx4d3 * sin_m * sin_m + esqdx5d2 * cos_m)
        sin_v = math.sin(v)
        cos_v = math.cos(v)

        # distance
        r = semilrec / (1.0 + eccentry * cos_v)

        # argument of perigee
        wsmall = TWOPI * (argperig + rcargper * deltim)

        # longitude of the ascending node
        wbig = TWOPI * (rascascn + rcascnrv * deltim)
        sin_wbig = math.sin(wbig)
        cos_wbig = math.cos(wbig)

        # calculate the rectangular coordinates
        #  (see Smart, Spherical Astronomy, section 75, page 122-124)

        f = wsmall + v
        sin_f = math.sin(f)
        cos_f = math.cos(f)

        x_hst[0] = r * (cos_wbig * cos_f - cosincli * sin_wbig * sin_f)
        x_hst[1] = r * (sin_wbig * cos_f + cosincli * cos_wbig * sin_f)
        x_hst[2] = r * sineincl * sin_f

        a0 = cirveloc * eccentry * sin_v / r
        a1 = cirveloc * (1.0 + eccentry * cos_v) + \
                TWOPI * rcargper * r
        v_hst[0] = a0 * x_hst[0] - \
                a1 * (cos_wbig * sin_f + cosincli * sin_wbig * cos_f) - \
                TWOPI * rcascnrv * x_hst[1]
        v_hst[1] = a0 * x_hst[1] - \
                a1 * (sin_wbig * sin_f - cosincli * cos_wbig * cos_f) + \
                TWOPI * rcascnrv * x_hst[0]
        v_hst[2] = a0 * x_hst[2] + a1 * sineincl * cos_f

        # Convert from meters to kilometers.
        x_hst /= 1000.0
        v_hst /= 1000.0

        return (x_hst, v_hst)
