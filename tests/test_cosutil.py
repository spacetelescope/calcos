import numpy as np
import pytest
from astropy.io import fits

from calcos import cosutil, MissingRowError
from tests import test_extract


def test_find_column():
    # Setup
    # create a test fits file
    name = "Output/findCol.fts"
    ofd = test_extract.generate_fits_file(name)

    target_col = 'TIME'
    # Test
    col_exists = True
    # Verify
    assert col_exists == cosutil.findColumn(name, target_col)


def test_get_table():
    # Setup
    # create a test fits file
    name = "Output/getTable.fts"
    ofd = test_extract.generate_fits_file(name)
    truth = [tuple(ofd[1].data[3])]
    time = ofd[1].data[3][0]
    # rawx = ofd[1].data[3][1]
    # Test
    dt = list(cosutil.getTable(name, {'TIME': time}, exactly_one=True))
    # Verify
    np.testing.assert_array_equal(truth, dt)


def test_get_table_exceptions():
    # Raise MissingRowError
    name = "Output/getTable.fts"
    ofd = test_extract.generate_fits_file(name)
    # truth = [tuple(ofd[1].data[3])]
    time = np.ones(5)  # non-existent values
    with pytest.raises(MissingRowError):
        cosutil.getTable(name, {'Time': time}, exactly_one=True)


def test_get_col_copy():
    # Setup
    # create a test fits file
    name = "Output/getTable.fits"
    ofd = test_extract.generate_fits_file(name)
    col_name = 'XCORR'
    portion_of_array = ofd[1].data[:]
    truth_values = ofd[1].data.field(col_name)
    # Test (using filename or portion of data)
    test1 = cosutil.getColCopy(name, column=col_name)
    test2 = cosutil.getColCopy(column=col_name, data=portion_of_array)

    # Verify
    np.testing.assert_array_equal(truth_values, test1)
    np.testing.assert_array_equal(truth_values, test2)


def test_get_col_copy_exception():
    # raise RuntimeError error
    with pytest.raises(RuntimeError):
        name = "Output/getTable.fits"
        ofd = test_extract.generate_fits_file(name)
        col_name = 'XCORR'
        portion_of_array = ofd[1].data[:]
        cosutil.getColCopy(filename="Output/getTable.fits", column=col_name, data=portion_of_array)
        cosutil.getColCopy(filename=None, column=col_name, data=None)


def test_get_headers():
    # Setup
    # create a test fits file
    name = "Output/getHeaders.fits"
    ofd = test_extract.generate_fits_file(name)
    true_hdr = ofd[0].header

    # Test
    test_hdr = cosutil.getHeaders(name)

    # Verify
    np.testing.assert_array_equal(true_hdr, test_hdr[0])
    # print(true_hdr[0])
    # print("#" * 100)
    # print(test_hdr[1][0])


def test_write_output_events():
    # Setup
    in_file = "Output/outputEvent.fits"
    out_file = "Output/outputEvents.fits"
    # ofd = test_extract.generate_fits_file(in_file)
    actual_lines = 10
    lines = cosutil.writeOutputEvents(in_file, out_file)
    # assert False
    assert actual_lines == lines


def test_concat_arrays():
    # setup
    arr1 = np.ones(10, dtype=float)
    arr2 = np.zeros(10, dtype=float)
    actual = np.concatenate((arr1, arr2), axis=0)
    # Test
    concat_arrays = cosutil.concatArrays(arr1, arr2)
    # Verify
    np.testing.assert_array_equal(actual, concat_arrays)


def test_update_filename():
    # Setup
    filename = "update_filename"
    test_extract.generate_fits_file("Output/update_filename.fits")
    before_update_hdr = fits.open("Output/update_filename.fits", mode="update")
    # Test
    cosutil.updateFilename(before_update_hdr[0].header, filename)
    before_update_hdr.close()
    after_update_hdr = fits.open("Output/update_filename.fits")
    # Verity
    assert filename == after_update_hdr[0].header["filename"]
    after_update_hdr.close()


def test_copy_file():
    # Setup
    infile = "Output/input.fits"
    test_extract.generate_fits_file(infile)
    outfile = "Output/output.fits"
    # Test
    cosutil.copyFile(infile, outfile)
    # Verify
    inf = fits.open(infile)
    out = fits.open(outfile)
    np.testing.assert_array_equal(inf[1].data, out[1].data)
    np.testing.assert_array_equal(inf[2].data, out[2].data)
    np.testing.assert_array_equal(inf[3].data, out[3].data)


def test_is_product():
    # Setup
    product_file = "Output/my0_product_a.fits"
    raw_file = "Output/my_raw.fits"
    test_extract.generate_fits_file(product_file)
    test_extract.generate_fits_file(raw_file)
    # Test

    # Verify
    assert cosutil.isProduct(product_file)
    assert not cosutil.isProduct(raw_file)


# todo: check if counts can be 0.
def test_gehrels_lower():
    # Setup
    counts = 4.0
    actual = 1.9090363511659807
    # Test
    test_value = cosutil.Gehrels_lower(counts)
    # Verify
    assert actual == test_value


# todo: handle ValueError
def test_cmp_part_exception():
    # test for exception
    with pytest.raises(ValueError) as t_err:
        cosutil.cmpPart(5, "k")


def test_cmp_part():
    # Setup
    str1 = "hello"
    str2 = "hey"
    # Test
    cmp1 = cosutil.cmpPart(str1, str2)
    cmp2 = cosutil.cmpPart(str2, str1)
    cmp3 = cosutil.cmpPart(str1, str1)
    # Verify
    assert cmp1 == -1
    assert cmp2 == 1
    assert cmp3 == 0


def test_split_int_letter():
    # Setup
    test = ["1this", "is", "4a", "15test"]
    truth = ["1", "this", "0", "is", "4", "a", "15", "test"]
    # Test
    cosutil.splitIntLetter(test)
    # Verify
    assert truth == test


def test_create_corrtag_hdu():
    # Setup
    hdu = test_extract.generate_fits_file("corrtag.fits")
    num_of_rows = 10
    # Test
    out_bin_table = cosutil.createCorrtagHDU(num_of_rows, hdu[1])
    assert len(out_bin_table.data) == num_of_rows
    assert out_bin_table != all(hdu[1].data)


def test_remove_wcs_keywords():
    # Setup
    hdu = test_extract.generate_fits_file("removeWCS.fits")
    inhdr = fits.getheader("corrtag.fits", 1)
    cd = hdu[1].data.columns
    WCS_keywords = ['TCTYP*',
                    'TCUNI*',
                    'TCRPX*',
                    'TCRVL*',
                    'TCDLT*',
                    ]
    # Test
    newheader = cosutil.remove_WCS_keywords(inhdr, cd)
    # Verify
    for keys in WCS_keywords:
        assert keys not in newheader
        assert len(inhdr[keys]) > 0


def test_copy_exptime_keywords():
    # Setup
    # create two files
    hdu = test_extract.generate_fits_file("original.fits")
    test_extract.generate_fits_file("copy.fits")
    # get header of the files
    inhdr = fits.getheader("original.fits", 1)
    outhdr = fits.getheader("corrtag.fits", 1)
    print(inhdr.get("expstart", -999.))
    hdu.close()
    # set values to the exposure time
    with open("original.fits", "ab+"):
        fits.setval("original.fits", "expstart*", value=-999, ext=1)
        print(fits.getval("original.fits", "expstart*"))
        print(repr(inhdr))
    # fits.setval("original.fits", "expend", value=-999, ext=1)
    # fits.setval("original.fits", "exptime", value=-999, ext=1)
    # fits.setval("original.fits", "rawtime", value=-999, ext=1)

    # Test
    # Verify


def test_dummy_gti():
    # Setup
    test_exptime_value = 1.423
    # Test
    dummy_hdu = cosutil.dummyGTI(test_exptime_value)
    # Verify
    assert dummy_hdu.data[0][0] == 0.0
    assert dummy_hdu.data[0][1] == test_exptime_value


def test_return_gti():
    # Setup
    hdu = test_extract.generate_fits_file("gti_file.fits")
    # Test
    gti = cosutil.returnGTI("gti_file.fits")
    # Verify
    np.testing.assert_array_equal(list(hdu[2].data), gti)
