from astropy.io import fits

from calcos import phot


def test_do_phot():
    pass
    # Setup
    obsmode = ["cos,nuv,mirrora,psa", "cos,nuv,mirrora,boa", "cos,nuv,mirrorb,psa", "cos,nuv,mirrorb,boa"]
    hdr = fits.open("51c1638pi_imp.fits")  # fits extension header (that will be updated)
    # imphttab = hdr[1]  # imaging photometric parameters table
    # Test
    # todo use for loop
    # todo test the values of
    #  photflam, photfnu, photbw, photplam, photzpt for a change
    phot.doPhot(obsmode[0], hdr)
    # Verify


def test_read_im_pht_tab():
    pass
    # Setup
    imphttab = ""  # this value is never used in this function however.
    obsmode = ["cos,nuv,mirrora,psa", "cos,nuv,mirrora,boa", "cos,nuv,mirrorb,psa", "cos,nuv,mirrorb,boa"]
    actual_param_values = [[4.816554456084e-18, 8.64540709538e-30, 382.88, 2319.7, -21.1],
                           [1.107251346369e-15, 1.90968620531e-27, 370.65, 2273.9, -21.1],
                           [9.720215320058e-17, 1.48789056193e-28, 466.56, 2142.4, -21.1],
                           [1.866877735677e-14, 2.68068135014e-26, 451.56, 2075.3, -21.1]]
    # Test
    test_param_values = []
    for obs in obsmode:
        test_param_values.append(phot.readImPhtTab(obs))
    # Verify
    for i in range(len(actual_param_values)):
        assert actual_param_values[i] == test_param_values[i]
