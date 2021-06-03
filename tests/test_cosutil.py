import io
import os
import sys
import time

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


def test_gehrels_lower():
    # Setup
    counts = 4.0
    actual = 1.9090363511659807
    # Test
    test_value = cosutil.Gehrels_lower(counts)
    # Verify
    assert actual == test_value


def test_err_gehrels():
    # Setup
    # values to be tested on
    zeros = np.zeros(5)
    ones = np.ones(5)
    random_values = np.array([2.2400559, 0.85776844, 5.31731382, 8.98167105, 7.88191824]).astype(np.float32)
    # Actual results expected
    true_lower1 = np.random.uniform(low=0.0, high=0.0, size=(5,))
    true_upper1 = np.array([1.8660254, 1.8660254, 1.8660254, 1.8660254, 1.8660254]).astype(np.float32)

    true_lower2 = np.array([0.8285322, 0.8285322, 0.8285322, 0.8285322, 0.8285322]).astype(np.float32)
    true_upper2 = np.array([2.3228757, 2.3228757, 2.3228757, 2.3228757, 2.3228757]).astype(np.float32)

    true_lower3 = np.array([1.2879757, 0.8285322, 2.1544096, 2.9387457, 2.7635214]).astype(np.float32)
    true_upper3 = np.array([2.6583123, 2.3228757, 3.3979158, 4.122499, 3.95804]).astype(np.float32)
    # Test
    lower1, upper1 = cosutil.errGehrels(zeros)  # should produce a warning
    lower2, upper2 = cosutil.errGehrels(ones)
    lower3, upper3 = cosutil.errGehrels(random_values)
    # Verify
    np.testing.assert_array_equal(true_lower1, lower1)
    np.testing.assert_array_equal(true_upper1, upper1)
    np.testing.assert_array_equal(true_lower2, lower2)
    np.testing.assert_array_equal(true_upper2, upper2)
    np.testing.assert_array_equal(true_lower3, lower3)
    np.testing.assert_array_equal(true_upper3, upper3)


def test_cmp_part_exception():
    # test for exception
    with pytest.raises(TypeError) as t_err:
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


def test_err_frequentist():
    # Setup
    zeros = np.zeros(5)
    ones = np.ones(5)
    random_values = np.array([2.2400559, 0.85776844, 5.31731382, 8.98167105, 7.88191824]).astype(np.float32)

    # Actual values expected
    true_lower1 = np.zeros(5).astype(np.float32)
    true_upper1 = np.array([1.8410217, 1.8410217, 1.8410217, 1.8410217, 1.8410217]).astype(np.float32)

    true_lower2 = np.array([0.82724625, 0.82724625, 0.82724625, 0.82724625, 0.82724625]).astype(np.float32)
    true_upper2 = np.array([2.2995265, 2.2995265, 2.2995265, 2.2995265, 2.2995265]).astype(np.float32)

    true_lower3 = np.array([1.3812032, 0.74085414, 2.2319276, 2.940346, 2.7469769]).astype(np.float32)
    true_upper3 = np.array([2.7093022, 2.2441676, 3.4480913, 4.1072574, 3.9250364]).astype(np.float32)
    # Test
    lower1, upper1 = cosutil.errFrequentist(zeros)
    lower2, upper2 = cosutil.errFrequentist(ones)
    lower3, upper3 = cosutil.errFrequentist(random_values)

    # Verify
    np.testing.assert_array_equal(true_lower1, lower1)
    np.testing.assert_array_equal(true_upper1, upper1)
    np.testing.assert_array_equal(true_lower2, lower2)
    np.testing.assert_array_equal(true_upper2, upper2)
    np.testing.assert_array_equal(true_lower3, lower3)
    np.testing.assert_array_equal(true_upper3, upper3)


def test_precess():
    # Setup
    time = 55545.270617  # MJD
    target = np.array([3.4, 6.5, 2.5])
    actual_coordinates = [3.3814054391781228, 6.508305447528035, 2.5036088867020565]
    # Test
    test_coordinates = cosutil.precess(time, target)
    # Verify
    np.testing.assert_array_equal(actual_coordinates, test_coordinates)


def test_fit_quartic():
    # Setup
    x = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([2, 4, 6, 8, 10, 12])
    polynom = cosutil.fitQuartic(x, y)
    # Expected
    fitted_polynomial = ([-2.00000000e+00, 2.00000000e+00, 8.46875656e-12, -1.74897134e-12,
                          1.23333442e-13],
                         [4.97529008e-11, 1.33375950e-10, 3.65406961e-11, 1.56942681e-12,
                          7.95573614e-15])
    test_polynomial = cosutil.fitQuartic(x, y)
    # Verify
    np.testing.assert_almost_equal(fitted_polynomial, test_polynomial)


def test_center_of_quadratic():
    # Setup
    # todo add more cases
    coeff = np.array([2, 4, 1])
    var = np.array([0, 2, 4])
    # Expected
    actual_center = (-2.0, 4.06201920231798)
    # Test
    center = cosutil.centerOfQuadratic(coeff, var)
    # Verify
    assert actual_center == center


def test_fit_quadratic():
    # Setup
    x = np.array([1, 2, 4])
    y = np.array([0, 7, 15])
    # Expected
    expected_quadratic = (np.array([-9., 10., -1.]), np.array([0., 0., 0.]))
    # Test
    fitted_quadratic = cosutil.fitQuadratic(x, y)
    # Verify
    # todo: arrays have the same value but fail to assert.
    np.testing.assert_array_almost_equal(expected_quadratic[0], fitted_quadratic[0])
    np.testing.assert_array_almost_equal(expected_quadratic[1], fitted_quadratic[1])


def test_change_segment():
    # Setup
    filename1 = "testfits_a.fits"
    filename2 = "testfits_b.fits"
    filename3 = "testfits.fits"
    # Expected
    fname1 = "testfits_b.fits"
    fname2 = "testfits_a.fits"
    fname3 = "testfits.fits"
    # Test
    test_name1 = cosutil.changeSegment(filename1, "FUV", "FUVB")
    test_name2 = cosutil.changeSegment(filename2, "FUV", "FUVA")
    test_name3 = cosutil.changeSegment(filename3, "NUV", "")
    # Verify
    assert fname1 == test_name1
    assert fname2 == test_name2
    assert fname3 == test_name3


def test_copy_exptime_keywords():
    # Setup
    # create two files
    test_extract.generate_fits_file("original.fits")
    test_extract.generate_fits_file("copy.fits")
    files = ["original.fits", "copy.fits"]
    headers = ["expstart", "expend", "exptime", "rawtime"]
    # set values to the exposure time
    for file in files:
        with open(file, "ab+"):
            for header in headers:
                if file == files[0]:
                    fits.setval(file, header, value=-999, ext=1)
                # set values in the copy to 0 to check if the function is actually changing the values.
                else:
                    fits.setval(file, header, value=0, ext=1)
    # get header of the files
    inhdr = fits.getheader("original.fits", 1)
    outhdr = fits.getheader("copy.fits", 1)
    # Test
    cosutil.copyExptimeKeywords(inhdr, outhdr)
    # Verify
    for header in headers:
        assert inhdr[header] == outhdr[header]


def test_copy_voltage_keywords():
    # Setup
    # create two files each for FUV and NUV
    test_extract.generate_fits_file("originalFUV.fits")
    test_extract.generate_fits_file("originalNUV.fits")
    test_extract.generate_fits_file("copyFUV.fits")
    test_extract.generate_fits_file("copyNUV.fits")
    original = ["originalFUV.fits", "originalNUV.fits"]
    copy = ["copyFUV.fits", "copyNUV.fits"]
    detectors = ["FUV", "NUV"]
    fuv_headers = ["dethvla", "dethvlb", "dethvca", "dethvcb", "dethvna", "dethvnb"]
    nuv_headers = ["dethvl", "dethvc"]
    # set the values to the headers
    index = 0
    for fileName in original:
        detector = detectors[index]
        with open(fileName, "ab+"):
            if detector == "FUV":
                for header in fuv_headers:
                    fits.setval(fileName, header, value=-999., ext=1)
            else:
                for header in nuv_headers:
                    fits.setval(fileName, header, value=-999, ext=1)
        index += 1
    index = 0
    for fileName in copy:
        detector = detectors[index]
        with open(fileName, "ab+"):
            if detector == "FUV":
                for header in fuv_headers:
                    fits.setval(fileName, header, value=0., ext=1)
            else:
                for header in nuv_headers:
                    fits.setval(fileName, header, value=0., ext=1)
        index += 1
    in_FUV_hdr = fits.getheader("originalFUV.fits", 1)
    in_NUV_hdr = fits.getheader("originalNUV.fits", 1)
    out_FUV_hdr = fits.getheader("copyFUV.fits", 1)
    out_NUV_hdr = fits.getheader("copyNUV.fits", 1)
    # Test 1
    cosutil.copyVoltageKeywords(in_FUV_hdr, out_FUV_hdr, detectors[0])
    # Verify 1
    for header in fuv_headers:
        assert in_FUV_hdr[header] == out_FUV_hdr[header]
    # Test 2
    cosutil.copyVoltageKeywords(in_NUV_hdr, out_NUV_hdr, detectors[1])
    # Verify 2
    for header in nuv_headers:
        assert in_NUV_hdr[header] == out_NUV_hdr[header]


def test_copy_sub_keywords():
    # Setup
    test_extract.generate_fits_file("subKeywords.fits")
    test_extract.generate_fits_file("copySubKeywords.fits")
    files = ["subKeywords.fits", "copySubKeywords.fits"]
    headers = ["corner%1dx", "corner%1dy", "size%1dx", "size%1dy"]

    for file in files:
        with open(file, "ab+"):
            for header in headers:
                for i in range(8):
                    if file == files[0]:
                        fits.setval(file, header % i, value=-1, ext=1)
                    else:  # set the value of the copy to 0 to test for change.
                        fits.setval(file, header % i, value=0, ext=1)
    with open(files[0], "ab+"):
        fits.setval(files[0], "nsubarry", value=0, ext=1)
    with open(files[1], "ab+"):
        fits.setval(files[1], "nsubarry", value=-1, ext=1)
    inhdr = fits.getheader(files[0], 1)
    outhdr = fits.getheader(files[1], 1)
    # Test
    cosutil.copySubKeywords(inhdr, outhdr, False)
    # Verify
    for header in headers:
        for i in range(8):
            assert inhdr[header % i] == outhdr[header % i]
    cosutil.copySubKeywords(inhdr, outhdr, True)
    # check if nsubarry has been set to 0
    assert inhdr["nsubarry"] == outhdr["nsubarry"]


def test_modify_asn_mtyp():
    # Setup
    str1 = "EXP-TESTVALUE"
    str2 = "EXP_SECONDVALUE"
    str3 = "VALUE_WITHOUT_PREFIX"
    # Expected vals
    string1 = "PROD-TESTVALUE"
    string2 = "PROD_SECONDVALUE"
    string3 = str3
    # Test
    val1 = cosutil.modifyAsnMtyp(str1)
    val2 = cosutil.modifyAsnMtyp(str2)
    val3 = cosutil.modifyAsnMtyp(str3)
    # Verify
    assert string1 == val1
    assert string2 == val2
    assert string3 == val3


def test_rename_file():
    # Setup
    original_filename1 = "raw-file.fits"
    original_filename2 = "product0_file_a.fits"

    new_filename1 = "renamed_raw_file.fits"
    new_filename2 = "renamed0_file_a.fits"
    # Create the files
    test_extract.generate_fits_file(original_filename1)
    test_extract.generate_fits_file(original_filename2)
    # Test
    cosutil.renameFile(original_filename1, new_filename1)
    cosutil.renameFile(original_filename2, new_filename2)
    # Verify
    assert os.path.exists(new_filename1)
    assert os.path.exists(new_filename2)


def test_del_corrtag_wcs():
    # Setup
    test_extract.generate_fits_file("del_corrtagWCS.fits")
    thdr = fits.getheader("del_corrtagWCS.fits", 3)
    tkey = ["TCTYP2", "TCRVL2", "TCRPX2", "TCDLT2", "TCUNI2", "TC2_2", "TC2_3",
            "TCTYP3", "TCRVL3", "TCRPX3", "TCDLT3", "TCUNI3", "TC3_2", "TC3_3"]
    # Test
    thdr = cosutil.delCorrtagWCS(thdr)
    # Verify
    for key in tkey:
        assert key not in thdr


def test_set_verbosity():
    # Setup
    verbosity = 5
    # Test
    cosutil.setVerbosity(verbosity)
    test_value = cosutil.verbosity
    # Verify
    assert verbosity == test_value


def test_check_verbosity():
    # Setup
    verbosity = 4
    # Test
    cosutil.setVerbosity(3)
    test1 = cosutil.verbosity
    check_verbosity = cosutil.checkVerbosity(verbosity)  # should be False
    cosutil.setVerbosity(verbosity)
    check2 = cosutil.checkVerbosity(verbosity)  # should be True
    # Verify
    assert not check_verbosity
    assert check2


def test_set_write_to_trailer():
    # Setup
    flag = True
    # Test
    cosutil.setWriteToTrailer(flag)  # should change to True
    flag_value = cosutil.write_to_trailer
    cosutil.setWriteToTrailer()  # should change to False
    # Verify
    assert flag_value
    assert not cosutil.write_to_trailer


def test_print_msg():
    # Setup
    captured_msg = io.StringIO()
    sys.stdout = captured_msg  # redirect stdout
    verbosity_level = 4
    test_message = "message from pytest"
    # Test
    cosutil.setVerbosity(verbosity_level)
    cosutil.printMsg(test_message, verbosity_level)
    sys.stdout = sys.__stdout__  # reset the redirect
    # Verify
    print(captured_msg.getvalue()[:-1])
    assert test_message == captured_msg.getvalue()[:-1]  # to remove the newline at the end


def test_return_time():
    t = time.strftime("%d-%b-%Y %H:%M:%S %Z", time.localtime(time.time()))
    get_time = cosutil.returnTime()
    assert t == get_time


def test_print_intro():
    # Setup
    captured_msg = io.StringIO()
    sys.stdout = captured_msg
    verbosity = 4
    message = "Pytest message"
    # Test
    cosutil.setVerbosity(verbosity)
    cosutil.printIntro(message)
    sys.stdout = sys.__stdout__
    # remove the newlines at the beginning and end
    captured_msg = captured_msg.getvalue()[1:-1]
    time = cosutil.returnTime()
    message = message + " -- " + time
    # Verify
    assert captured_msg == message


def test_print_filenames():
    # Setup
    captured_msg = io.StringIO()
    sys.stdout = captured_msg
    verbosity = 4
    names = [("Input", "abc_raw.fits"), ("Output", "abc_flt.fits")]
    stimfile = "stim.txt"
    livetimefile = "live.txt"
    expected_msg = ""
    for (lable, filename) in names:
        expected_msg += "%-10s%s\n" % (lable, filename)
    expected_msg += "stim locations log file   " + stimfile + "\n"
    expected_msg += "livetime factors log file " + livetimefile + "\n"
    # Test
    cosutil.setVerbosity(verbosity)
    cosutil.printFilenames(names, shift_file=None, stimfile=stimfile, livetimefile=livetimefile)
    sys.stdout = sys.__stdout__
    # Verify
    assert expected_msg == captured_msg.getvalue()


def test_print_mode():
    # Setup
    hdu_list = test_extract.generate_fits_file("printMode_test.fits")
    hdr = hdu_list[1]
    info = hdr
    captured_mgs = io.StringIO()
    sys.stdout = captured_mgs
    # Test
    cosutil.printMode(info)
    sys.stdout = sys.__stdout__
    print(captured_mgs.getvalue())
    # Verify
    # assert False


# test_print_mode()


def test_print_warning():
    # Setup
    captured_msg = io.StringIO()
    sys.stdout = captured_msg
    verbosity = 4
    message = "warning message from pytest"
    # Test
    cosutil.setVerbosity(verbosity)
    cosutil.printWarning(message, verbosity)
    sys.stdout = sys.__stdout__
    message = "Warning:  " + message + "\n"
    # Verify
    assert message == captured_msg.getvalue()


def test_print_error():
    # Setup
    captured_msg = io.StringIO()
    sys.stdout = captured_msg
    verbosity = 4
    message = "error message from pytest"
    # Test
    cosutil.setVerbosity(verbosity)
    cosutil.printError(message)
    sys.stdout = sys.__stdout__
    message = "ERROR:  " + message + "\n"
    # Verify
    assert message == captured_msg.getvalue()


def test_print_continuation():
    # Setup
    captured_msg = io.StringIO()
    sys.stdout = captured_msg
    verbosity = 4
    message = "continued message from pytest"
    # Test
    cosutil.setVerbosity(verbosity)
    cosutil.printContinuation(message, level=verbosity)
    sys.stdout = sys.__stdout__
    message = "    " + message + "\n"
    assert message == captured_msg.getvalue()


def test_print_ref():
    # Setup
    captured_msg = io.StringIO()
    sys.stdout = captured_msg
    verbosity = 4
    keyw = "flatfile".upper()
    reffiles = {"flatfile": "abc_flat.fits", "flatfile_hdr": "lref$abc_flat.fits"}
    message = "%-8s= %s" % (keyw, reffiles[keyw.lower() + "_hdr"]) + "\n"
    cosutil.setVerbosity(verbosity)
    # Test
    cosutil.printRef(keyw, reffiles)
    sys.stdout = sys.__stdout__
    print(message)
    # Verify
    assert message == captured_msg.getvalue()


def test_print_switch():
    # Setup
    switches = {"statflag": "PERFORM", "flatcorr": "PERFORM", "geocorr": "COMPLETE", "randcorr": "SKIPPED"}
    keys = ["STATFLAG", "FLATCORR", "GEOCORR", "RANDCORR"]
    captured_msg = io.StringIO()
    sys.stdout = captured_msg
    msg1 = "STATFLAG  T\n"
    msg2 = "FLATCORR  PERFORM\n"
    msg3 = "%-9s OMIT (already complete)" % keys[2].upper() + "\n"
    msg4 = "%-9s OMIT (skipped)" % keys[3].upper()+"\n"
    actual_msg = msg1+msg2+msg3+msg4
    # Test
    for key in keys:
        cosutil.printSwitch(key,switches)
        captured_msg.flush()
    sys.stdout = sys.__stdout__
    # Verify
    assert actual_msg == captured_msg.getvalue()