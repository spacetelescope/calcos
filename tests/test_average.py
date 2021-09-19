import os

from astropy.io import fits

from calcos import average


def test_avg_image():
    # Setup
    infile = ["lc8803010_fltsum.fits", "lc8803i6q_counts.fits"]
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
    for i in range(len(out_hdr)):
        if i != 6:
            assert inhdr1[i] == out_hdr[i] == inhdr2[i]
