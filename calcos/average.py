from __future__ import absolute_import, division         # confidence high
import numpy as np
import astropy.io.fits as fits
from . import cosutil
from .calcosparam import *       # parameter definitions

def avgImage(input, output):
    """Average 2-D image sets, assumed to be aligned.

    Parameters
    ----------
    input: str
        Name of the input file.

    output: str
        Name of the output file.
    """

    nimages = len(input)

    assert nimages >= 1

    cosutil.printIntro("Average images")
    names = [("Input", repr(input)), ("Output", output)]
    cosutil.printFilenames(names)

    if nimages == 1:
        cosutil.copyFile(input[0], output)
        if cosutil.isProduct(output):
            fd = fits.open(output, mode="update")
            asn_mtyp = fd[1].header.get("asn_mtyp", "missing")
            asn_mtyp = cosutil.modifyAsnMtyp(asn_mtyp)
            if asn_mtyp != "missing":
                fd[1].header["asn_mtyp"] = asn_mtyp
            fd.close()
        return

    # Average the SCI extensions.

    got_data = 0                                # initial values
    sum_exptime = 0.
    sum_plantime = 0.
    sum_globrate = 0.

    # Open the first file just to get some header keywords.
    ifd = fits.open(input[0], mode="copyonwrite")
    phdr = ifd[0].header
    sci_extn = ifd["SCI"]
    statflag = phdr.get("statflag", False)
    if phdr["detector"] == "FUV":
        segment = phdr["segment"]
        globrate_keyword = "globrt_" + segment[-1].lower()
    else:
        globrate_keyword = "globrate"
    expstart = sci_extn.header["expstart"]
    expend = sci_extn.header["expend"]
    ifd.close()

    for i in range(nimages):
        ifd = fits.open(input[i], mode="copyonwrite")
        sci_extn = ifd["SCI"]
        exptime = sci_extn.header["exptime"]
        sum_plantime += sci_extn.header.get("plantime", exptime)
        expstart = min(expstart, sci_extn.header["expstart"])
        expend = max(expend, sci_extn.header["expend"])
        if sci_extn.data is not None:
            if got_data:
                sci_data += (sci_extn.data * exptime)
            else:
                hdr = sci_extn.header
                sci_data = sci_extn.data * exptime
                got_data = 1
            sum_exptime += exptime
            sum_globrate += (sci_extn.header[globrate_keyword] * exptime)
        ifd.close()
    del ifd

    if got_data:
        if sum_exptime <= 0.:
            raise RuntimeError("ERROR in avgImage; invalid EXPTIME.")
        sci_data /= sum_exptime
        globrate = sum_globrate / sum_exptime
    else:
        sci_data = None
        globrate = 0.

    # Create the output file, and write the averaged SCI extension.
    primary_hdu = fits.PrimaryHDU(header=phdr)
    cosutil.updateFilename(primary_hdu.header, output)
    ofd = fits.HDUList(primary_hdu)
    scihdu = fits.ImageHDU(data=sci_data, header=hdr, name="SCI")
    if cosutil.isProduct(output):
        asn_mtyp = scihdu.header.get("asn_mtyp", "missing")
        asn_mtyp = cosutil.modifyAsnMtyp(asn_mtyp)
        if asn_mtyp != "missing":
            scihdu.header["asn_mtyp"] = asn_mtyp
    scihdu.header["exptime"] = sum_exptime
    scihdu.header["expstart"] = expstart
    scihdu.header["expend"] = expend
    scihdu.header["expstrtj"] = expstart + MJD_TO_JD
    scihdu.header["expendj"] = expend + MJD_TO_JD
    scihdu.header["plantime"] = sum_plantime
    scihdu.header[globrate_keyword] = round(globrate, 4)
    ofd.append(scihdu)
    ofd.writeto(output, output_verify='silentfix')
    del ofd, phdr, hdr, primary_hdu, sci_data, scihdu

    # Average the ERR extensions in quadrature.

    got_data = 0
    for i in range(nimages):
        ifd = fits.open(input[i], mode="copyonwrite")
        sci_extn = ifd["SCI"]
        err_extn = ifd["ERR"]
        exptime = sci_extn.header["exptime"]    # exptime is in SCI extension
        if err_extn.data is not None:
            if got_data:
                err_data += (err_extn.data * exptime)**2
            else:
                hdr = err_extn.header
                err_data = (err_extn.data * exptime)**2
                got_data = 1
        elif i == 0:
            hdr = err_extn.header
        ifd.close()
    del ifd

    if got_data:
        np.sqrt(err_data, err_data)
        err_data /= sum_exptime
    else:
        err_data = None

    ofd = fits.open(output, mode="append")
    errhdu = fits.ImageHDU(data=err_data, header=hdr, name="ERR")
    ofd.append(errhdu)
    ofd.close()
    del ofd, hdr, err_data, errhdu

    # Combine the DQ extensions.

    got_data = 0
    for i in range(nimages):
        ifd = fits.open(input[i], mode="copyonwrite")
        dq_extn = ifd["DQ"]
        if dq_extn.data is not None:
            if got_data:
                np.bitwise_or(dq_data, dq_extn.data, dq_data)
            else:
                hdr = dq_extn.header
                dq_data = dq_extn.data
                got_data = 1
        elif i == 0:
            hdr = dq_extn.header
        ifd.close()
    del ifd

    ofd = fits.open(output, mode="append")
    dqhdu = fits.ImageHDU(data=dq_data, header=hdr, name="DQ")
    ofd.append(dqhdu)
    ofd.close()
    del ofd, hdr, dq_data, dqhdu

    if statflag:
        cosutil.doImageStat(output)
