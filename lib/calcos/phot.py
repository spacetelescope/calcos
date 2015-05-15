from __future__ import absolute_import, division         # confidence high

from . import cosutil

def doPhot(imphttab, obsmode, hdr):
    """Update photometry parameter keywords for imaging data.

    PHOTFLAM, inverse sensitivity, ergs/s/cm2/Ang per count/s
    PHOTFNU,  inverse sensitivity, ergs/s/cm2/Hz per count/s
    PHOTBW,   RMS bandwidth of filter plus detector (Angstroms)
    PHOTPLAM, Pivot wavelength (Angstroms)
    PHOTZPT = -21.10, ST magnitude system zero point

    Parameters
    ----------
    imphttab: str
        The name of the imaging photometric parameters table.

    obsmode: str
        Observation mode (e.g. "cos,nuv,mirrora,psa").

    hdr: pyfits Header object
        The first extension header, updated in-place.
    """

    (photflam, photfnu, photbw, photplam, photzpt) = \
                readImPhtTab(imphttab, obsmode)

    hdr["photflam"] = photflam
    hdr["photfnu"] = photfnu
    hdr["photbw"] = photbw
    hdr["photplam"] = photplam
    hdr["photzpt"] = photzpt

def readImPhtTab(imphttab, obsmode):
    """Read the photometry parameters for imaging data from the imphttab.

    This version has hardcoded values, since the imphttab hasn't been
    created yet.  The values were determined using pysynphot (and synphot
    for the bandwidth) as follows:
    
    % setenv PYSYN_CDBS <directory>
    % python
    > import pysynphot as S
    > for obsmode in ["cos,nuv,mirrora,psa",
    >             "cos,nuv,mirrora,boa",
    >             "cos,nuv,mirrorb,psa",
    >             "cos,nuv,mirrorb,boa"]:
    >     sp = S.FlatSpectrum(1., fluxunits="flam")
    >     bp = S.ObsBandpass(obsmode)
    >     obs = S.Observation(sp, bp)
    >     print "#", fluxunits, obsmode
    >     print 1. / obs.countrate()          # photflam
    >     print obs.pivot()                   # photplam

    obs.pivot() gave different values for flam vs fnu, so the values
    obtained via bandpar were used instead.  The bandwidth was also
    gotten via bandpar.  Here is an example (showing only the first lines):
    --> bandpar "cos,nuv,mirrora,psa"
               # OBSMODE          URESP          PIVWV          BANDW
     cos,nuv,mirrora,psa     4.8214E-18         2319.7         382.88

    The values of photfnu were gotten from photflam as follows:

    bp = S.ObsBandpass("cos,nuv,mirrora,psa")
    sp = S.FlatSpectrum(4.816554456084e-18, fluxunits="flam")
    print obs.effstim("fnu")
    photfnu = 8.64540709538e-30

    bp = S.ObsBandpass("cos,nuv,mirrora,boa")
    sp = S.FlatSpectrum(1.107251346369e-15, fluxunits="flam")
    photfnu = 1.90968620531e-27

    bp = S.ObsBandpass("cos,nuv,mirrorb,psa")
    sp = S.FlatSpectrum(9.720215320058e-17, fluxunits="flam")
    photfnu = 1.48789056193e-28

    bp = S.ObsBandpass("cos,nuv,mirrorb,boa")
    sp = S.FlatSpectrum(1.866877735677e-14, fluxunits="flam")
    photfnu = 2.68068135014e-26

    Parameters
    ----------
    imphttab: str
        The name of the imaging photometry parameters table.

    obsmode: str
        Observation mode.

    Returns
    -------
    param: tuple of floats
        Photflam, photfnu, photbw, photplam, photpzt.
    """

    # These values are photflam, photfnu, photbw, photplam, photzpt:

    photdict = {
        "mirrora,psa": [4.816554456084e-18,
                        8.64540709538e-30,
                        382.88,
                        2319.7,
                        -21.1],
        "mirrora,boa": [1.107251346369e-15,
                        1.90968620531e-27,
                        370.65,
                        2273.9,
                        -21.1],
        "mirrorb,psa": [9.720215320058e-17,
                        1.48789056193e-28,
                        466.56,
                        2142.4,
                        -21.1],
        "mirrorb,boa": [1.866877735677e-14,
                        2.68068135014e-26,
                        451.56,
                        2075.3,
                        -21.1]
        }

    if obsmode.find(",") >= 0:
        words = obsmode.split(",")
    else:
        words = obsmode.split()
    w = []
    for word in words:
        w.append(word.strip())
    words = w

    keylist = ["dummy", "dummy"]
    for word_orig in words:
        word = word_orig.lower()
        if word == "cos":
            continue
        elif word == "nuv":
            continue
        elif word == "mirrora" or word == "mirrorb":
            keylist[0] = word
        elif word == "psa" or word == "boa":
            keylist[1] = word
        else:
            cosutil.printWarning("Don't recognize obsmode component %s" %
                                 word_orig)

    if keylist[1] == "dummy":
        cosutil.printWarning("No valid aperture found in obsmode %s;"
                             % obsmode)
        cosutil.printContinuation("assuming PSA instead.")
        keylist[1] = "psa"

    key = keylist[0] + "," + keylist[1]

    if key in photdict:
        param = photdict[key]
    else:
        raise RuntimeError("obsmode '%s' not recognized, expected "
                           "'mirrora' or 'mirrorb', 'psa' or 'boa'" % obsmode)

    return param
