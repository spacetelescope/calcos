import os

import numpy as np
from astropy.io import fits

from calcos import average


def test_avg_image():
    # Setup
    infile = ["test_count1.fits", "test_count2.fits"]
    outfile = "test_output.fits"
    if os.path.exists(outfile):
        os.remove(outfile)  # avoid file exists error
    inhdr1, inhdr2 = fits.getheader(infile[0]), fits.getheader(infile[1])
    # Test
    average.avgImage(infile, outfile)
    out_hdr = fits.getheader(outfile)
    # for i in range(0,160):
    #     print(i," ",inhdr1[i]," ",inhdr2[i]," ",out_hdr[i])
    # Verify
    assert os.path.exists(outfile)
    for (i, j, k) in zip(inhdr1[1].header, inhdr2[1].header, out_hdr[1].header):
        assert i == j == k
    np.testing.assert_array_equal((inhdr1[1].data + inhdr1[1].data) / 2, out_hdr[1].data)
