import os

import numpy as np
from astropy.io import fits

from calcos import average
from generate_tempfiles import create_count_file


def test_avg_image():
    """
    tests avg_image() in average.py
    explanation of the test
    - create temporary count files to be used as inputs
    - expected values in the output file are the average of the input values
    - loop though the values to check if the math holds.
    Returns
    -------
    pass if expected == actual fail otherwise.
    """
    # Setup
    infile = ["test_count1.fits", "test_count2.fits"]
    outfile = "test_output.fits"
    if os.path.exists(outfile):
        os.remove(outfile)  # avoid file exists error
    create_count_file(infile[0])
    create_count_file(infile[1])
    inhdr1, inhdr2 = fits.open(infile[0]), fits.open(infile[1])
    # Test
    average.avgImage(infile, outfile)
    out_hdr = fits.open(outfile)

    # Verify
    assert os.path.exists(outfile)
    for (i, j, k) in zip(inhdr1[1].header, inhdr2[1].header, out_hdr[1].header):
        assert i == j == k
    np.testing.assert_array_equal((inhdr1[1].data + inhdr1[1].data) / 2, out_hdr[1].data)
