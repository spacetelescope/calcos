from __future__ import division

def doPhot (imphttab, obsmode, hdr):
    """Update photometry parameter keywords for imaging data.

    @param imphttab: the name of the imaging photometric parameters table
    @type imphttab: string
    @param obsmode: observation mode (e.g. "cos,nuv,mirrora,psa")
    @type obsmode: string
    @param hdr: the first extension header, updated in-place
    @type hdr: pyfits Header object

    PHOTFLAM, inverse sensitivity, ergs/s/cm2/Ang per count/s
    PHOTFNU,  inverse sensitivity, ergs/s/cm2/Hz per count/s
    PHOTBW,   RMS bandwidth of filter plus detector (Angstroms)
    PHOTPLAM, Pivot wavelength (Angstroms)
    PHOTZPT = -21.10, ST magnitude system zero point
    """

    (photflam, photfnu, photbw, photplam, photzpt) = \
                readImPhtTab (imphttab, obsmode)

    hdr.update ("photflam", photflam)
    hdr.update ("photfnu", photfnu)
    hdr.update ("photbw", photbw)
    hdr.update ("photplam", photplam)
    hdr.update ("photzpt", photzpt)

def readImPhtTab (imphttab, obsmode):
    """Read the photometry parameters for imaging data from the imphttab.

    @param imphttab: the name of the imaging photometry parameters table
    @type imphttab: string
    @param obsmode: observation mode
    @type obsmode: string

    @return: photflam, photfnu, photbw, photplam, photpzt
    @rtype: tuple of floats

    This version has hardcoded values, since the imphttab hasn't been
    created yet.  The values were determined using pysynphot (and synphot
    for the bandwidth) as follows:

    % python
    >>> import pysynphot as S
    >>> sp = S.FlatSpectrum (1., fluxunits="flam")
    >>> sp = S.FlatSpectrum (1., fluxunits="fnu")
    >>> bp = S.ObsBandpass ("cos,nuv,mirrora,psa")
    >>> bp = S.ObsBandpass ("cos,nuv,mirrora,boa")
    >>> bp = S.ObsBandpass ("cos,nuv,mirrorb,psa")
    >>> bp = S.ObsBandpass ("cos,nuv,mirrorb,boa")
    >>> print 1. / obs.countrate()    # photflam or photfnu
    >>> print obs.pivot()             # photplam (pivot wavelength)

    The bandwidth doesn't seem to be available via pysynphot yet, so
    the old synphot was used, e.g.:
     --> bandpar "cos,nuv,mirrora,psa"
                # OBSMODE          URESP          PIVWV          BANDW
      cos,nuv,mirrora,psa     5.3957E-18         2310.1         373.71
    """

    # These values are photflam, photfnu, photbw, photplam, photzpt:
    photdict = {
        "mirrora,psa": [5.43017143551e-18,
                        9.62456797814e-30,
                        373.71,
                        2376.0957106802111,
                        -21.1],
        "mirrora,boa": [1.09115778462e-15,
                        1.85857319356e-27,
                        361.2,
                        2329.2062282964275,
                        -21.1],
        "mirrorb,psa": [6.48688953828e-17,
                        9.81902116625e-29,
                        460.38,
                        2238.2740455962849,
                        -21.1],
        "mirrorb,boa": [1.25491564392e-14,
                        1.78345723897e-26,
                        445.7,
                        2171.5642072889386,
                        -21.1]
        }

    if obsmode.find (",") >= 0:
        words = obsmode.split (",")
    else:
        words = obsmode.split()
    w = []
    for word in words:
        w.append (word.strip().lower())
    words = w

    keylist = ["dummy", "dummy"]
    for word in words:
        if word == "cos":
            continue
        elif word == "nuv":
            continue
        elif word == "mirrora" or word == "mirrorb":
            keylist[0] = word
        elif word == "psa" or word == "boa":
            keylist[1] = word

    key = keylist[0] + "," + keylist[1]

    if photdict.has_key (key):
        param = photdict[key]
    else:
        raise RuntimeError, "obsmode '%s' not recognized, expected " \
                            "'mirrora' or 'mirrorb', 'psa' or 'boa'" % obsmode

    return param
