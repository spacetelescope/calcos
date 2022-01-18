import generate_tempfiles
from calcos import getinfo
from astropy.io import fits


def test_initial_info():
    """
    Create a temp fits file and cross check the headers from the primary header.
    """
    # Setup
    temp_file = "initial_info_temp.fits"
    hdu = generate_tempfiles.generate_fits_file(temp_file)
    inf = {}
    phdr = hdu[0].header
    inf["detector"] = phdr["DETECTOR"]
    inf["obsmode"] = phdr["OBSMODE"]
    inf["exptype"] = phdr["EXPTYPE"]
    # Test
    test_inf = getinfo.initialInfo(temp_file)
    # Verify
    assert inf["detector"] == test_inf["detector"]
    assert inf["obsmode"] == test_inf["obsmode"]
    assert inf["exptype"] == test_inf["exptype"]


def test_get_general_info():
    assert True


def test_get_switch_values():
    assert True


def test_get_ref_file_names():
    assert True


def test_reset_switches():
    assert True
