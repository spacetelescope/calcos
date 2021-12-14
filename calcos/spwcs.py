from __future__ import absolute_import
import math
import astropy.io.fits as fits
from .calcosparam import *
from . import cosutil

# These are the column numbers (one indexed) for the XFULL and YFULL
# columns in a corrtag file.  See SpWcsCorrtag.__init__().
Xi = 7
Eta = 8

class SpWCS(object):
    """Base class for spectroscopic coordinate parameters.

    usage:

    import spwcs

    wcs = spwcs.SpWcsCorrtag(corrtag_filename, info, helcorr,
                             spwcstab, xtractab)
    flag = wcs.writeWCSKeywords()

    wcs = spwcs.SpWcsImage(image_filename, info, helcorr,
                           spwcstab, xtractab)
    flag = wcs.writeWCSKeywords()

    Parameters
    ----------
    filename: str
        Name of file within which keywords will be updated.

    info: dictionary
        Header keywords and values.

    helcorr: str
        PERFORM or COMPLETE if heliocentric correction should be applied
        to the wavelengths (CRVAL1).

    spwcstab: str
        Name of reference table containing spectroscopic WCS parameters.

    xtractab: str
        Name of reference table for extraction parameters.
    """

    def __init__(self, filename, info, helcorr, spwcstab, xtractab):
        """Constructor."""

        self.filename = filename
        self.info = info
        self.helcorr = helcorr
        self.spwcstab = spwcstab
        self.xtractab = xtractab

        # These four will be assigned in a subclass.  For the dictionaries,
        # the key is a generic form for the keyword (e.g. crval1), and the
        # value is the actual keyword (e.g. crval1, crval1a, tcrvl7, tcrv7a)
        # except that it does not include the trailing letter for the
        # alternate WCS.  The actual keyword will either be taken directly
        # from primary_key_dict, or it will be constructed by appending the
        # alternate WCS letter to alternate_key_dict.
        self.extension = 1      # default
        self.keywords = []
        self.primary_key_dict = {}
        self.alternate_key_dict = {}    # does not include the letter (A, B, C)

        self.detector = info["detector"]
        self.ra_aper  = info["ra_aper"]
        self.dec_aper = info["dec_aper"]
        self.pa_aper  = info["pa_aper"]
        self.x_offset = info["x_offset"]
        self.v_helio = 0.               # assigned later

    def writeWCSKeywords(self):
        """Update keywords in-place in the extension header.

        Returns
        -------
        boolean
            True if keywords were actually written.  False if the file is
            a wavecal or an FCA exposure.
        """

        if self.detector == "FUV":
            segment_list = [self.info["segment"]]
        else:
            # "primary" refers to the primary axis description
            segment_list = ["primary", "NUVA", "NUVB", "NUVC"]

        aperture = self.info["aperture"]
        if aperture not in ["PSA", "BOA"]:
            return False

        fd = fits.open(self.filename, mode="update")
        hdr = fd[self.extension].header
        self.v_helio = hdr.get("v_helio", 0.)

        # Delete some redundant or unnecessary keywords.
        self.deleteKeywords(hdr)

        for segment in segment_list:
            if self.detector == "FUV":
                alt = ""
            else:
                if segment == "primary":
                    segment = "NUVB"    # use NUVB for the primary WCS
                    alt = ""
                else:
                    alt = segment[-1]

            filter = {"opt_elem": self.info["opt_elem"],
                      "cenwave":  self.info["cenwave"],
                      "segment":  segment,
                      "aperture": aperture}
            wcs_info = cosutil.getTable(self.spwcstab, filter,
                                        exactly_one=True)

            wcs_dict = self.computeKeywordValues(wcs_info[0], alt)
            self.addKeywords(hdr, wcs_dict)

        fd.close()
        return True

    def computeKeywordValues(self, wcs_info, alt):
        """Defined in a subclass."""
        pass

    def computeCrpix2(self, wcs_info):
        """Determine the value of the crpix2 keyword.

        crpix2 should be the location of the spectrum, at the point where
        it crosses the middle (crpix1) of the detector.  This depends on
        the segment or stripe.

        Parameters
        ----------
        wcs_info: array_like
            One row from the spwcstab.

        Returns
        -------
        crpix2: float
            The value for the crpix2 keyword (one indexed).
        """

        segment = wcs_info.field("segment")
        filter = {"segment": segment,
                  "opt_elem": self.info["opt_elem"],
                  "cenwave": self.info["cenwave"],
                  "aperture": self.info["aperture"]}
        xtract_info = cosutil.getTable(self.xtractab, filter,
                                       exactly_one=True)
        slope = xtract_info.field("slope")[0]
        b_spec = xtract_info.field("b_spec")[0]

        middle = wcs_info.field("crpix1") - 1. # zero indexing
        crpix2 = b_spec + middle * slope

        return (crpix2 + 1.)                    # one indexing

    def makeKeyword(self, generic_keyword, alt):
        """Construct the actual keyword name.

        Parameters
        ----------
        generic_keyword: str
            Generic WCS keyword (e.g. ctype1)

        alt: str
            Alternate WCS letter, or "" for the primary WCS

        Returns
        -------
        str
            Actual keyword to use in header (e.g. ctype1a, tcty7a)
        """

        if alt and alt != " ":
            keyword = self.alternate_key_dict[generic_keyword] + alt
        else:
            keyword = self.primary_key_dict[generic_keyword]

        return keyword

    def doHelcorr(self, crval1):
        """Apply heliocentric correction (if helcorr is perform) to crval1.

        Parameters
        ----------
        crval1: float
            Wavelength at the reference pixel, as read from sptrctab

        Returns
        -------
        float
            Crval1 with heliocentric velocity correction applied
        """

        if self.helcorr == "PERFORM" or self.helcorr == "COMPLETE":
            crval1 -= (crval1 * self.v_helio / SPEED_OF_LIGHT)
        return crval1

    def deleteKeywords(self, hdr):
        """Defined in a subclass."""
        pass

    def addKeywords(self, hdr, wcs_dict):
        """Add (or update) WCS keywords in the header.

        Parameters
        ----------
        hdr: pyfits Header object
            header to be updated in-place

        wcs_dict: dictionary
            Key is the generic WCS keyword (lower case), value is a tuple
            of the actual keyword (lower case) and the value to assign to
            that keyword in the header
        """

        for generic_keyword in self.keywords:
            (actual_keyword, value) = wcs_dict[generic_keyword]
            if generic_keyword == "wcsaxes":
                # It is a FITS requirement that WCSAXES precede all other
                # WCS keywords in a header.
                if actual_keyword in hdr:
                    hdr[actual_keyword] = value
                else:
                    # GCOUNT is the last of the set of keywords that must be
                    # present at the beginning of an extension header.
                    if actual_keyword == "wcsaxes":
                        hdr.set(actual_keyword, value, after="gcount") # xxx
                        # xxx hdr.insert("gcount", (actual_keyword, value),
                        # xxx            after=True)
                    elif actual_keyword == "wcsaxesa":
                        hdr.set(actual_keyword, value, after="wcsaxes") # xxx
                        # xxx hdr.insert("wcsaxes", (actual_keyword, value),
                        # xxx            after=True)
                    elif actual_keyword == "wcsaxesb":
                        hdr.set(actual_keyword, value, after="wcsaxesa") # xxx
                        # xxx hdr.insert("wcsaxesa", (actual_keyword, value),
                        # xxx            after=True)
                    elif actual_keyword == "wcsaxesc":
                        hdr.set(actual_keyword, value, after="wcsaxesb") # xxx
                        # xxx hdr.insert("wcsaxesb", (actual_keyword, value),
                        # xxx            after=True)
                    else:       # don't really expect anything else
                        hdr.set(actual_keyword, value, after="gcount") # xxx
                        # xxx hdr.insert("gcount", (actual_keyword, value),
                        # xxx            after=True)
            else:
                hdr[actual_keyword] = value

class SpWcsImage(SpWCS):
    """Spectroscopic WCS for image data.

    Parameters
    ----------
    filename: str
        Name of image file.

    info: dictionary
        Header keywords and values.

    helcorr: str
        PERFORM or COMPLETE if heliocentric correction should be applied
        to the wavelengths (CRVAL1).

    spwcstab: str
        Name of reference table containing spectroscopic WCS parameters.

    xtractab: str
        Name of reference table for extraction parameters.
    """

    def __init__(self, filename, info, helcorr, spwcstab, xtractab):
        """Constructor."""

        SpWCS.__init__(self, filename, info, helcorr, spwcstab, xtractab)

        self.extension = ("sci",1)

        # The WCS keywords that we'll update in the header.  This is in
        # a list (copied to a dictionary below) so that the order will be
        # well defined, in case the header doesn't have all these keywords.
        self.keywords = ["wcsaxes",
                         "ctype1", "ctype2", "ctype3",
                         "crpix1", "crpix2",
                         "crval1", "crval2", "crval3",
                         "pc1_1", "pc1_2", "pc2_1", "pc2_2", "pc3_1", "pc3_2",
                         "cdelt1", "cdelt2", "cdelt3",
                         "cunit1",
                         "pv1_0", "pv1_1", "pv1_2", "pv1_6"]

        # Keywords for an image array.
        for key in self.keywords:
            self.primary_key_dict[key] = key
            self.alternate_key_dict[key] = key

    def computeKeywordValues(self, wcs_info, alt):
        """Determine the values of the WCS keywords.

        Parameters
        ----------
        wcs_info: pyfits record object
            One row from the spwcstab

        alt: str
            Alternate WCS letter, or "" for the primary WCS

        Returns
        -------
        dictionary
            Key is the generic WCS keyword (but lower case), value is a
            tuple of the actual keyword (lower case) and the value to
            assign to that keyword in the header
        """

        cos_pa = math.cos(self.pa_aper * math.pi / 180.)
        sin_pa = math.sin(self.pa_aper * math.pi / 180.)
        pc2_1 =  cos_pa
        pc2_2 = -sin_pa
        pc3_1 =  sin_pa
        pc3_2 =  cos_pa

        # The key will be a generic keyword, and the value will be a tuple
        # with the actual keyword and the value to assign for that keyword.
        wcs_dict = {}

        wcs_dict["wcsaxes"] = (self.makeKeyword("wcsaxes", alt), 3)

        wcs_dict["ctype1"] = (self.makeKeyword("ctype1", alt),
                              wcs_info.field("ctype1"))
        wcs_dict["ctype2"] = (self.makeKeyword("ctype2", alt), "RA---TAN")
        wcs_dict["ctype3"] = (self.makeKeyword("ctype3", alt), "DEC--TAN")

        crval1 = self.doHelcorr(wcs_info.field("crval1"))
        wcs_dict["crval1"] = (self.makeKeyword("crval1", alt), crval1)
        wcs_dict["crval2"] = (self.makeKeyword("crval2", alt), self.ra_aper)
        wcs_dict["crval3"] = (self.makeKeyword("crval3", alt), self.dec_aper)

        wcs_dict["cunit1"] = (self.makeKeyword("cunit1", alt), "angstrom")

        wcs_dict["crpix1"] = (self.makeKeyword("crpix1", alt),
                              wcs_info.field("crpix1") + self.x_offset)
        wcs_dict["crpix2"] = (self.makeKeyword("crpix2", alt),
                              self.computeCrpix2(wcs_info))

        wcs_dict["pc1_1"] = (self.makeKeyword("pc1_1", alt), 1.)
        wcs_dict["pc1_2"] = (self.makeKeyword("pc1_2", alt), 0.)
        wcs_dict["pc2_1"] = (self.makeKeyword("pc2_1", alt), pc2_1)
        wcs_dict["pc2_2"] = (self.makeKeyword("pc2_2", alt), pc2_2)
        wcs_dict["pc3_1"] = (self.makeKeyword("pc3_1", alt), pc3_1)
        wcs_dict["pc3_2"] = (self.makeKeyword("pc3_2", alt), pc3_2)

        wcs_dict["cdelt1"] = (self.makeKeyword("cdelt1", alt),
                              wcs_info.field("cdelt1"))
        wcs_dict["cdelt2"] = (self.makeKeyword("cdelt2", alt),
                              wcs_info.field("cdelt2"))
        wcs_dict["cdelt3"] = (self.makeKeyword("cdelt3", alt),
                              wcs_info.field("cdelt3"))

        wcs_dict["pv1_0"] = (self.makeKeyword("pv1_0", alt),
                             wcs_info.field("g"))
        wcs_dict["pv1_1"] = (self.makeKeyword("pv1_1", alt),
                             wcs_info.field("sporder"))
        wcs_dict["pv1_2"] = (self.makeKeyword("pv1_2", alt),
                             wcs_info.field("alpha"))
        wcs_dict["pv1_6"] = (self.makeKeyword("pv1_6", alt),
                             wcs_info.field("theta"))

        return wcs_dict

    def deleteKeywords(self, hdr):
        """Delete some keywords (if they're present) in the header."""

        keyword_list = ["talen2", "talen3", "cunit2",
                        "cd1_1", "cd1_2", "cd2_1", "cd2_2"]
        for keyword in keyword_list:
            if keyword in hdr:
                del hdr[keyword]

        # The following keywords could be left around if the input file
        # was corrtag rather than raw.
        keyword_list = ["tctyp7", "tctyp8", "tcrpx7", "tcrpx8",
                        "tcrvl7", "tcrvl8", "tcdlt7", "tcdlt8",
                        "tpc7_7", "tpc7_8", "tpc8_7", "tpc8_8",
                        "tcuni7", "tcuni8",
                        "tpv7_0", "tpv7_1", "tpv7_2", "tpv7_6"]
        if self.detector == "FUV":
            for keyword in keyword_list:
                if keyword in hdr:
                    del hdr[keyword]
        else:
            for alt in ["", "a", "b", "c"]:
                for key in keyword_list:
                    keyword = key + alt
                    if keyword in hdr:
                        del hdr[keyword]
            more_keywords = ["tcty7", "tcty8", "tcrp7", "tcrp8",
                             "tcrv7", "tcrv8", "tcde7", "tcde8",
                             "tcun7", "tcun8"]
            # These keywords are only used for an alternate WCS, so drop "".
            for alt in ["a", "b", "c"]:
                for key in more_keywords:
                    keyword = key + alt
                    if keyword in hdr:
                        del hdr[keyword]

class SpWcsCorrtag(SpWCS):
    """Spectroscopic WCS for pixel list (corrtag) data.

    Parameters
    ----------
    filename: str
        Name of corrtag file.

    info: dictionary
        Header keywords and values.

    helcorr: str
        PERFORM or COMPLETE if heliocentric correction should
        be applied to the wavelengths (CRVAL1).

    spwcstab: str
        Name of reference table containing spectroscopic WCS
        parameters.

    xtractab: str
        Name of reference table for extraction parameters.
    """

    def __init__(self, filename, info, helcorr, spwcstab, xtractab):
        """Constructor."""

        SpWCS.__init__(self, filename, info, helcorr, spwcstab, xtractab)

        self.extension = ("events",1)

        # These are the generic names for the keywords that we'll update in
        # the header; the actual names are listed below (in the same order!)
        # separately for primary and alternate coordinate axes.
        # This is in a list (copied to a dictionary below) so that the order
        # will be well defined, in case the header doesn't have all these
        # keywords.
        self.keywords = ["ctype1", "ctype2",
                         "crpix1", "crpix2",
                         "crval1", "crval2",
                         "pc1_1",
                         "pc1_2",
                         "pc2_1",
                         "pc2_2",
                         "cdelt1", "cdelt2",
                         "cunit1", "cunit2",
                         "pv1_0", "pv1_1", "pv1_2", "pv1_6"]
        # These are the actual keywords for the primary coordinate system.
        primary_keywords = ["tctyp%d" % Xi, "tctyp%d" % Eta,
                            "tcrpx%d" % Xi, "tcrpx%d" % Eta,
                            "tcrvl%d" % Xi, "tcrvl%d" % Eta,
                            "tpc%d_%d" % (Xi, Xi),
                            "tpc%d_%d" % (Xi, Eta),
                            "tpc%d_%d" % (Eta, Xi),
                            "tpc%d_%d" % (Eta, Eta),
                            "tcdlt%d" % Xi, "tcdlt%d" % Eta,
                            "tcuni%d" % Xi, "tcuni%d" % Eta,
                            "tpv%d_0" % Xi,
                            "tpv%d_1" % Xi,
                            "tpv%d_2" % Xi,
                            "tpv%d_6" % Xi]
        # These are the actual keywords for an alternate coordinate system,
        # except that the letter (A, B, C) indicating the alternate system
        # is not included here.
        alternate_keywords = ["tcty%d" % Xi, "tcty%d" % Eta,
                              "tcrp%d" % Xi, "tcrp%d" % Eta,
                              "tcrv%d" % Xi, "tcrv%d" % Eta,
                              "tpc%d_%d" % (Xi, Xi),
                              "tpc%d_%d" % (Xi, Eta),
                              "tpc%d_%d" % (Eta, Xi),
                              "tpc%d_%d" % (Eta, Eta),
                              "tcde%d" % Xi, "tcde%d" % Eta,
                              "tcun%d" % Xi, "tcun%d" % Eta,
                              "tpv%d_0" % Xi,
                              "tpv%d_1" % Xi,
                              "tpv%d_2" % Xi,
                              "tpv%d_6" % Xi]

        # Copy keywords from the lists to dictionaries.
        for i in range(len(self.keywords)):
            key = self.keywords[i]
            self.primary_key_dict[key] = primary_keywords[i]
            self.alternate_key_dict[key] = alternate_keywords[i]

    def computeKeywordValues(self, wcs_info, alt):
        """Determine the values of the WCS keywords.

        wcs_info: pyfits record object
            One row from the spwcstab

        alt: str
            Alternate WCS letter, or "" for the primary WCS

        dictionary
            Key is the generic WCS keyword (but lower case), value is a
            tuple of the actual keyword (lower case) and the value to
            assign to that keyword in the header
        """

        wcs_dict = {}

        wcs_dict["ctype1"] = (self.makeKeyword("ctype1", alt),
                              wcs_info.field("ctype1"))
        wcs_dict["ctype2"] = (self.makeKeyword("ctype2", alt), "ANGLE")

        crval1 = self.doHelcorr(wcs_info.field("crval1"))
        wcs_dict["crval1"] = (self.makeKeyword("crval1", alt), crval1)
        wcs_dict["crval2"] = (self.makeKeyword("crval2", alt), 0.)

        wcs_dict["cunit1"] = (self.makeKeyword("cunit1", alt), "angstrom")
        wcs_dict["cunit2"] = (self.makeKeyword("cunit2", alt), "deg")

        wcs_dict["crpix1"] = (self.makeKeyword("crpix1", alt),
                              wcs_info.field("crpix1") + self.x_offset)
        wcs_dict["crpix2"] = (self.makeKeyword("crpix2", alt),
                              self.computeCrpix2(wcs_info))

        wcs_dict["pc1_1"] = (self.makeKeyword("pc1_1", alt), 1.)
        wcs_dict["pc1_2"] = (self.makeKeyword("pc1_2", alt), 0.)
        wcs_dict["pc2_1"] = (self.makeKeyword("pc2_1", alt), 0.)
        wcs_dict["pc2_2"] = (self.makeKeyword("pc2_2", alt), 1.)

        wcs_dict["cdelt1"] = (self.makeKeyword("cdelt1", alt),
                              wcs_info.field("cdelt1"))
        # note that the value is cdelt3 from the table, which is the Y axis
        wcs_dict["cdelt2"] = (self.makeKeyword("cdelt2", alt),
                              wcs_info.field("cdelt3"))

        wcs_dict["pv1_0"] = (self.makeKeyword("pv1_0", alt),
                             wcs_info.field("g"))
        wcs_dict["pv1_1"] = (self.makeKeyword("pv1_1", alt),
                             wcs_info.field("sporder"))
        wcs_dict["pv1_2"] = (self.makeKeyword("pv1_2", alt),
                             wcs_info.field("alpha"))
        wcs_dict["pv1_6"] = (self.makeKeyword("pv1_6", alt),
                             wcs_info.field("theta"))

        return wcs_dict

    def deleteKeywords(self, hdr):
        """Delete some keywords (if they're present) in the header."""

        keyword_list = ["tctyp2", "tctyp3", "tcrvl2", "tcrvl3",
                        "tcdlt2", "tcdlt3", "tcrpx2", "tcrpx3",
                        "tc2_2", "tc2_3", "tc3_2", "tc3_3",
                        "tcuni2", "tcuni3"]

        for keyword in keyword_list:
            if keyword in hdr:
                del hdr[keyword]
