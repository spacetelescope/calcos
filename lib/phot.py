from __future__ import division         # confidence high

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

    % setenv PYSYN_CDBS <directory>
    % python
    >>> import pysynphot as S
    >>> for obsmode in ["cos,nuv,mirrora,psa",
    >>>             "cos,nuv,mirrora,boa",
    >>>             "cos,nuv,mirrorb,psa",
    >>>             "cos,nuv,mirrorb,boa"]:
    >>>     for fluxunits in ["flam", "fnu"]:
    >>>         sp = S.FlatSpectrum (1., fluxunits=fluxunits)
    >>>         bp = S.ObsBandpass (obsmode)
    >>>         obs = S.Observation (sp, bp)
    >>>         print "#", fluxunits, obsmode
    >>>         print 1. / obs.countrate()      # photflam or photfnu
    >>>         print obs.pivot()               # photplam

    obs.pivot() gave different values for flam vs fnu, so the values
    obtained via bandpar were used instead.  The bandwidth was also
    gotten via bandpar.  Here is an example (showing only the first lines):
    --> bandpar "cos,nuv,mirrora,psa"
               # OBSMODE          URESP          PIVWV          BANDW
     cos,nuv,mirrora,psa     4.8214E-18         2319.7         382.88
    """

    # These values are photflam, photfnu, photbw, photplam, photzpt:
    # photflam for mirrora,boa was 9.84441964002e-16
    # photfnu  for mirrora,boa was 1.67911272761e-27
    # photflam for mirrorb,boa was 1.90534694809e-14
    # photfnu  for mirrorb,boa was 2.71134226806e-26
    #
    # photflam for mirrora,boa = 233. * 4.89549708817e-18
    # photfnu  for mirrora,boa =
    #           1.67911272761e-27 / 9.84441964002e-16 * 1.14065082154e-15
    # photflam for mirrorb,boa = 196. * 9.84213962334e-17
    # photfnu  for mirrorb,boa =
    #           2.71134226806e-26 / 1.90534694809e-14 * 1.92905936617e-14
    photdict = {
        "mirrora,psa": [4.89549708817e-18,
                        8.69157674037e-30,
                        382.88,
                        2319.7,
                        -21.1],
        "mirrora,boa": [1.14065082154e-15,
                        1.94555025308e-27,
                        370.65,
                        2273.9,
                        -21.1],
        "mirrorb,psa": [9.84213962334e-17,
                        1.49222298358e-28,
                        466.56,
                        2142.4,
                        -21.1],
        "mirrorb,boa": [1.92905936617e-14,
                        2.74508545666e-26,
                        451.56,
                        2075.3,
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
