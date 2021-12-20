import io
import os
import sys
import time

import numpy as np
import pytest
from astropy.io import fits

from calcos import cosutil, MissingRowError
from generate_tempfiles import generate_fits_file


def test_find_column():
    # Setup
    # create a test fits file
    name = "findCol.fits"
    ofd = generate_fits_file(name)

    target_col = 'TIME'
    # Test
    col_exists = True
    # Verify
    assert col_exists == cosutil.findColumn(name, target_col)


def test_get_table():
    # Setup
    # create a test fits file
    name = "getTable.fits"
    ofd = generate_fits_file(name)
    truth = [tuple(ofd[1].data[3])]
    time = ofd[1].data[3][0]
    # Test
    dt = list(cosutil.getTable(name, {'TIME': time}, exactly_one=True))
    # Verify
    np.testing.assert_array_equal(truth, dt)


def test_get_table_exceptions():
    # Raise MissingRowError
    name = "getTable.fits"
    generate_fits_file(name)
    # truth = [tuple(ofd[1].data[3])]
    t = np.ones(5)  # non-existent values
    with pytest.raises(MissingRowError):
        cosutil.getTable(name, {'Time': t}, exactly_one=True)


def test_get_col_copy():
    # Setup
    # create a test fits file
    name = "getTable.fits"
    ofd = generate_fits_file(name)
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
        name = "getTable.fits"
        ofd = generate_fits_file(name)
        col_name = 'XCORR'
        portion_of_array = ofd[1].data[:]
        cosutil.getColCopy(filename="Output/getTable.fits", column=col_name, data=portion_of_array)
        cosutil.getColCopy(filename=None, column=col_name, data=None)


def test_get_headers():
    # Setup
    # create a test fits file
    name = "getHeaders.fits"
    ofd = generate_fits_file(name)
    true_hdr = ofd[0].header

    # Test
    test_hdr = cosutil.getHeaders(name)

    # Verify
    np.testing.assert_array_equal(true_hdr, test_hdr[0])


def test_write_output_events():
    # Setup
    in_file = "outputEvents.fits"
    out_file = "outputEvents_cpy.fits"
    generate_fits_file(in_file)
    actual_lines = 10
    lines = cosutil.writeOutputEvents(in_file, out_file)
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
    generate_fits_file("update_filename.fits")
    before_update_hdr = fits.open("update_filename.fits", mode="update")
    # Test
    cosutil.updateFilename(before_update_hdr[0].header, filename)
    before_update_hdr.close()
    after_update_hdr = fits.open("update_filename.fits")
    # Verity
    assert filename == after_update_hdr[0].header["filename"]
    after_update_hdr.close()


def test_copy_file():
    # Setup
    infile = "input.fits"
    generate_fits_file(infile)
    outfile = "output.fits"
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
    product_file = "my0_product_a.fits"
    raw_file = "my_raw.fits"
    generate_fits_file(product_file)
    generate_fits_file(raw_file)
    # Test

    # NOTE:
    # no test to be done here since we're checking the file if its a product or not
    # the return of the function isProduct() is a boolean hence, assert it directly.

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
    """
    unit test for err_gehrels(counts)
    test ran
    - create 3 arrays, one should be random float values, the other two should be the lower (zero) and upper (one) error estimates
    - following the math for calculating the upper limit by taking the sqrt of counts + 0.5 and then adding 1 to the result.
    - similarly for the lower we add counts + 0.5 and then counts - counts * (1.0 - 1.0 / (9.0 * counts) - 1.0 / (3.0 * np.sqrt(counts))) ** 3
      we will be able to get the lower array.
    - finally assert the upper array and the lower array with the results obtained from err_gehrels().
    """
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
    hdu = generate_fits_file("corrtag.fits")
    num_of_rows = 10
    # Test
    # detector parameter is not needed consider removing it
    out_bin_table = cosutil.createCorrtagHDU(num_of_rows, detector="FUV", hdu=hdu[0])
    assert len(out_bin_table.data) == num_of_rows
    assert all(out_bin_table.header) == all(hdu[0].header)


def test_remove_wcs_keywords():
    # Setup
    hdu = generate_fits_file("removeWCS.fits")
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
    hdu = generate_fits_file("gti_file.fits")
    # Test
    gti = cosutil.returnGTI("gti_file.fits")
    # Verify
    np.testing.assert_array_equal(list(hdu[2].data), gti)


def test_err_frequentist():
    """
    unit test for err_frequentist(counts)
    - create 3 arrays similar to the test in err_gehrels().
    - find the poisson confidence interval for each array.
    - assert the result with the expected err_lower and err_upper.
    """
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
    """
    unit test for precess(t, target)
    - set a time in MJD
    - create a unit vector toward the target
    - calculate the expected coordinates
    - assert expected with the actual.
    """
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
    # Expected
    fitted_polynomial = ([-2.00000000e+00, 2.00000000e+00, 8.46875656e-12, -1.74897134e-12,
                          1.23333442e-13],
                         [4.97529008e-11, 1.33375950e-10, 3.65406961e-11, 1.56942681e-12,
                          7.95573614e-15])
    test_polynomial = cosutil.fitQuartic(x, y)
    # Verify
    np.testing.assert_almost_equal(fitted_polynomial, test_polynomial)


def test_center_of_quadratic():
    """
    unit test for center_of_quadratic(coeff, var)
    - create a randomized coeff and var arrays
    - follow the math to calculate x_min and x_min_sigma aka center of the quadratic
    - x_min = -coeff[1] / (2 * coeff[2])
    - x_min_sigma = 0.5 * math.sqrt(var1 / coeff[2] ** 2 + var2 * coeff[1] ** 2 / coeff[2] ** 4)
    - assert the expected result with the functions return.
    """
    # Setup
    coeff1 = np.array([2, 4, 1])
    coeff2 = np.array([3, 5, 7])
    coeff3 = np.array([2.3, 4.6, 1.3])
    coeff4 = np.zeros(shape=3)

    var1 = np.array([0, 2, 4])
    var2 = np.array([1, 2, 5])
    var3 = np.array([5, 3, 9])
    # Expected
    actual_center1 = (-2.0, 4.06201920231798)
    actual_center2 = (-0.35714285714285715, 0.15237943390885794)
    actual_center3 = (-1.769230769230769, 4.1368310795286165)
    actual_center4 = (None, 0.)
    # Test
    center1 = cosutil.centerOfQuadratic(coeff1, var1)
    center2 = cosutil.centerOfQuadratic(coeff2, var2)
    center3 = cosutil.centerOfQuadratic(coeff3, var3)
    center4 = cosutil.centerOfQuadratic(coeff4, var3)
    # Verify
    assert actual_center1 == center1
    assert actual_center2 == center2
    assert actual_center3 == center3
    assert actual_center4 == center4


def test_fit_quadratic():
    # Setup
    x = np.array([1, 2, 4])
    y = np.array([0, 7, 15])
    # Expected
    expected_quadratic = (np.array([-9., 10., -1.], dtype=np.float64), np.array([0., 0., 0.]))
    # Test
    fitted_quadratic = cosutil.fitQuadratic(x, y)
    # round to 10 decimal places to avoid assertion error due to floating point error.
    fitted_quadratic[0][0] = round(fitted_quadratic[0][0], 10)
    fitted_quadratic[0][1] = round(fitted_quadratic[0][1], 10)
    fitted_quadratic[0][2] = round(fitted_quadratic[0][2], 10)
    # Verify
    np.testing.assert_array_equal(expected_quadratic[0], fitted_quadratic[0])
    np.testing.assert_array_equal(expected_quadratic[1], fitted_quadratic[1])


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
    generate_fits_file("original.fits")
    generate_fits_file("copy.fits")
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
    generate_fits_file("originalFUV.fits")
    generate_fits_file("originalNUV.fits")
    generate_fits_file("copyFUV.fits")
    generate_fits_file("copyNUV.fits")
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
    generate_fits_file("subKeywords.fits")
    generate_fits_file("copySubKeywords.fits")
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
    generate_fits_file(original_filename1)
    generate_fits_file(original_filename2)
    # Test
    cosutil.renameFile(original_filename1, new_filename1)
    cosutil.renameFile(original_filename2, new_filename2)
    # Verify
    assert os.path.exists(new_filename1)
    assert os.path.exists(new_filename2)


def test_del_corrtag_wcs():
    # Setup
    generate_fits_file("del_corrtagWCS.fits")
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
    assert test_message == captured_msg.getvalue()[:-1]  # to remove the newline at the end


def test_return_time():
    t = time.strftime("%d-%b-%Y %H:%M:%S %Z", time.localtime(time.time()))
    get_time = cosutil.returnTime()
    assert t == get_time


"""
Unit tests that have the word "print" in their name using the same algorithm.
- open an IO stream
- initialize the message you want to print
- call the function that is being tested and pass the message string to it.
- redirect the output stream towards the function to catch the printed message
- write the value to a variable
- assert the captured message with the original message.
"""


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
    msg4 = "%-9s OMIT (skipped)" % keys[3].upper() + "\n"
    actual_msg = msg1 + msg2 + msg3 + msg4
    # Test
    for key in keys:
        cosutil.printSwitch(key, switches)
        captured_msg.flush()
    sys.stdout = sys.__stdout__
    # Verify
    assert actual_msg == captured_msg.getvalue()


def test_guess_aper_from_locn():
    """
    unit test for guessAperFromLocn()
    - create lists for LPs and aperture positions (2 in this case).
    - use the ranges provided to guess which aperture is being used
        LP: 1
        (116.0, 135) ---> PSA
        (-163.0, -143.0) ---> BOA
        LP: 2
        (52.0, 72.0) ---> PSA
        (-227.0, -207.0) ---> BOA
        LP: 3 and above
        aperture will be none
    - assert expected positions with the actual position.
    """
    # Setup
    lps = [1, 2, 3]
    aper_pos1 = [120.2, -151.34, 2.34]  # lp 1
    expected_aper1 = ["PSA", "BOA", None]
    aper_pos2 = [63.2, -214.89, 10.67]  # lp 2
    expected_aper2 = ['PSA', 'BOA', None]
    # Test
    test_postitions1 = []
    test_postitions2 = []
    test_postitions3 = []
    for lp in lps:
        if lp == 1:
            for pos in aper_pos1:
                test_postitions1.append(cosutil.guessAperFromLocn(lp, pos))
        elif lp == 2:
            for pos in aper_pos2:
                test_postitions2.append(cosutil.guessAperFromLocn(lp, pos))
        else:
            test_postitions3 = None
    # Verify
    assert expected_aper1 == test_postitions1
    assert expected_aper2 == test_postitions2
    assert test_postitions3 is None


def test_segment_specific_keyword():
    # Setup
    root = "keyword_root_"
    seg = "FUVA"
    seg2 = "NUVB"
    # Test
    key = cosutil.segmentSpecificKeyword(root, seg)  # keyword_root_a
    key2 = cosutil.segmentSpecificKeyword(root, seg2)  # keyword_root_
    # Verify
    root2 = root
    root += "a"
    assert key == root
    assert key2 == root2


def test_find_ref_file():
    # Setup
    generate_fits_file("wrong_file.fits")
    generate_fits_file("test.fits")
    fits.setval("wrong_file.fits", 'FILETYPE', value='FLAT FIELD REFERENCE IMAGE')
    # Missing
    ref1 = {"keyword": "FLATFILE", "filename": "test_flt.fits", "calcos_ver": "3.0",
            "min_ver": "2.2", "filetype": "FLAT FIELD REFERENCE IMAGE"}
    # Bad version
    ref2 = {"keyword": "FLATFILE", "filename": "test.fits", "calcos_ver": "2.21",
            "min_ver": "3.3", "filetype": "FLAT FIELD REFERENCE IMAGE"}
    # Wrong file
    ref3 = {"keyword": "FLATFILE", "filename": "wrong_file.fits", "calcos_ver": "3.0",
            "min_ver": "2.3", "filetype": "IMAGE"}
    missing1 = {}
    missing2 = {}
    missing3 = {}
    wrong_f1 = {}
    wrong_f2 = {}
    wrong_f3 = {}
    bad_ver1 = {}
    bad_ver2 = {}
    bad_ver3 = {}
    # Actual values
    actual_missing = {"FLATFILE": "test_flt.fits"}
    actual_bad_ver = {"FLATFILE": ("test.fits", "  the reference file must be at least version 3.3")}
    actual_wrong_ver = {'FLATFILE': ('wrong_file.fits', 'IMAGE')}
    # Test
    cosutil.findRefFile(ref1, missing1, wrong_f1, bad_ver1)
    cosutil.findRefFile(ref2, missing2, wrong_f2, bad_ver2)
    cosutil.findRefFile(ref3, missing3, wrong_f3, bad_ver3)
    # Verify
    assert actual_missing == missing1
    assert actual_bad_ver == bad_ver2
    assert actual_wrong_ver == wrong_f3


def test_cmp_version():
    # Setup
    min_ver = ["1", "1", "1.1", "1.1", "1.1", "1.2", "1.0", "2.7", "2.0", "2.9", "2.12d", "2.13d", "2.13"]
    vcalcos = ["1", "1.1", "1", "1.1", "1.2", "1.1", "1.7", "2.8", "2.13.1", "2.9", "2.13b", "2.13b", "2.13b"]
    calcos_ver = ["1.1", "1", "1", "1.2", "1.1", "1.1", "2.3", "2.8a", "2.13", "2.13.1", "2.12a", "2.13a", "2.13c"]

    expected_cmp = [0, 1, -1, 0, 1, -1, 0, 0, 1, 0, 1, -1, 0]
    # Test and Verify
    test_cmp = []
    for i in range(len(expected_cmp)):
        test_cmp.append(cosutil.cmpVersion(min_ver[i], vcalcos[i], calcos_ver[i]))
        assert expected_cmp[i] == test_cmp[i]


def test_get_pedigree():
    # Setup
    capture_msg = io.StringIO()
    sys.stdout = capture_msg
    switch = "perform"
    refkey = "statflag"
    filename = "test_flt.file"
    generate_fits_file(filename)
    err_msg = "Warning:  STATFLAG test_flt.file is a dummy file\n" \
              "    so PERFORM will not be done.\n"
    # Test
    pedgr1 = cosutil.getPedigree(switch, refkey, filename)
    fits.setval(filename, "pedigree", value="DUMMY", ext=0)
    pedgr2 = cosutil.getPedigree(switch, refkey, filename)
    sys.stdout = sys.__stdout__
    print(capture_msg.getvalue())

    # Verify
    assert pedgr1 == "OK"
    assert pedgr2 == "DUMMY"
    assert err_msg == capture_msg.getvalue()


def test_get_aperture_keyword():
    # Setup
    generate_fits_file("aperture_test.fits")
    # condition 1
    fits.setval("aperture_test.fits", "aperture", value="PSA-FUV", ext=0)
    fits.setval("aperture_test.fits", "propaper", value="PSA-FUV", ext=0)
    hdr1 = fits.getheader("aperture_test.fits", ext=0)
    # condition 2
    fits.setval("aperture_test.fits", "aperture", value="RelMvReq", ext=0)
    fits.setval("aperture_test.fits", "propaper", value="WCA", ext=0)
    hdr2 = fits.getheader("aperture_test.fits", ext=0)
    # condition 3
    fits.setval("aperture_test.fits", "propaper", value="NA", ext=0)
    fits.setval("aperture_test.fits", "shutter", value="closed", ext=0)
    fits.setval("aperture_test.fits", "lampused", value="P", ext=0)
    hdr3 = fits.getheader("aperture_test.fits", ext=0)
    # condition 4
    fits.setval("aperture_test.fits", "lampused", value="D", ext=0)
    hdr4 = fits.getheader("aperture_test.fits", ext=0)
    # condition 5
    fits.setval("aperture_test.fits", "lampused", value="A", ext=0)
    hdr5 = fits.getheader("aperture_test.fits", ext=0)
    # condition 6
    fits.setval("aperture_test.fits", "shutter", value="open", ext=0)
    fits.setval("aperture_test.fits", "life_adj", value=2, ext=0)
    fits.setval("aperture_test.fits", "aperypos", value=4.56, ext=0)
    hdr6 = fits.getheader("aperture_test.fits", ext=0)

    # Expected values
    rtn1 = ('PSA', 'APERTURE changed from PSA-FUV to PSA')
    rtn2 = ('WCA', 'APERTURE changed from RelMvReq to WCA (copied from PROPAPER)')
    rtn3 = ('WCA', 'Guessing correct APERTURE ... was RelMvReq, now set to WCA')
    rtn4 = ('FCA', 'Guessing correct APERTURE ... was RelMvReq, now set to FCA')
    rtn5 = ('PSA', 'Guessing correct APERTURE ... was RelMvReq, now set to PSA')
    rtn6 = (None, 'Guessing correct APERTURE ... was RelMvReq, now set to PSA')
    # Test
    rtn_value1 = cosutil.getApertureKeyword(hdr1)
    rtn_value2 = cosutil.getApertureKeyword(hdr2)
    rtn_value3 = cosutil.getApertureKeyword(hdr3)
    rtn_value4 = cosutil.getApertureKeyword(hdr4)
    rtn_value5 = cosutil.getApertureKeyword(hdr5)
    rtn_value6 = cosutil.getApertureKeyword(hdr6)
    # Verify
    assert rtn1 == rtn_value1
    assert rtn2 == rtn_value2
    assert rtn3 == rtn_value3
    assert rtn4 == rtn_value4
    assert rtn5 == rtn_value5
    assert rtn6 == rtn_value6


def test_write_version_to_trailer():
    capture_msg = io.StringIO()
    sys.stdout = capture_msg
    generate_fits_file("dummy_file.fits")
    ascii_file = open("ascii.txt", mode="w")
    cosutil.fd_trl = ascii_file
    cosutil.CALCOS_VERSION = '3.1.0'
    cosutil.writeVersionToTrailer()
    sys.stdout = sys.__stdout__
    assert ascii_file == cosutil.fd_trl
    assert capture_msg.getvalue() == ''


def test_get_switch():
    # Setup
    generate_fits_file("switch.fits")
    phdr = fits.getheader("switch.fits", ext=0)
    keyword = "statflag"
    # Test
    switch = cosutil.getSwitch(phdr, keyword)
    phdr.set('STATFLAG', False, 'Calculate statistics')
    switch2 = cosutil.getSwitch(phdr, keyword)
    switch3 = cosutil.getSwitch(phdr, "none")
    # Verify
    assert switch == 'PERFORM'
    assert switch2 == 'OMIT'
    assert switch3 == 'N/A'


def test_temp_pulse_height_range():
    generate_fits_file("pulseHeightRef.fits")
    true_pha_value = 4
    fits.setval("pulseHeightRef.fits", "pharange", value=true_pha_value, ext=0)
    fits.getheader("pulseHeightRef.fits", ext=0)
    pha_value = cosutil.tempPulseHeightRange('pulseHeightRef.fits')
    assert true_pha_value == pha_value


def test_get_pulse_height_range():
    fits.setval("pulseHeightRef.fits", "phalowrA", value=7, ext=0)
    fits.setval("pulseHeightRef.fits", "phaupprA", value=10, ext=0)
    hdu = fits.getheader("pulseHeightRef.fits", ext=0)
    seg = ['FUVA', 'FUVB']
    actual = [' 7_10', None]
    for s, a in zip(seg, actual):
        test_str = cosutil.getPulseHeightRange(hdu, s)
        assert a == test_str


def test_time_at_midpoint():
    # Setup
    generate_fits_file("test_timeAtMidpoint.fits")
    hdr = fits.getheader("test_timeAtMidpoint.fits", ext=1)
    info = {'expstart': hdr['TCRPX7'], 'expend': hdr['TCRVL7']}
    average = 4776.314586556611
    # Test
    test_average = cosutil.timeAtMidpoint(info)
    # Verify
    assert average == test_average


def test_timeline_times():
    # Setup
    first_time = np.array([2.5], dtype=np.float32)
    last_time = np.array([3.8], dtype=np.float32)
    actual_values = [2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    # Test
    tl_time = cosutil.timelineTimes(2.5, 10.8)
    # Verify
    for i in range(len(actual_values)):
        assert actual_values[i] == tl_time[i]


def test_combine_stat():
    # Setup
    stat_info = [{'ngoodpix': 5.8, "sci_goodmax": 2.7, "sci_goodmean": 4.3, "err_goodmax": 1,
                  "err_goodmean": 2}, {'ngoodpix': 10.8, "sci_goodmax": 4.7, "sci_goodmean": 8.3, "err_goodmax": 2,
                                       "err_goodmean": 1}]
    actual = {'ngoodpix': 16.6, 'sci_goodmax': 4.7, 'sci_goodmean': 6.902409638554217, 'err_goodmax': 2,
              'err_goodmean': 1.3493975903614457}
    # Test
    test = cosutil.combineStat(stat_info)
    # Verify
    for key in actual.keys():
        assert actual[key] == test[key]


def test_override_keywords():
    # Setup
    generate_fits_file("overridekeywords.fits")
    info = {"cal_ver": 3.1, "opt_elem": 2, "cenwave": 0.34, "fpoffset": 3.43, "obstype": "FUV",
            "exptype": "N/A", "aperture": "PSA", "x_offset": 1.2, "dispaxis": 2.5}
    switches = {"statflag": "PERFORM", "flatcorr": "PERFORM", "geocorr": "COMPLETE", "randcorr": "SKIPPED"}
    reffiles = {"flatfile": "abc_flat.fits", "flt_hdr": "lref$abc_flat.fits"}
    fits.setval("overridekeywords.fits", "flt_hdr", value="NA", ext=0)
    fits.setval("overridekeywords.fits", "dispaxis", value=2.9, ext=1)
    fits.setval("overridekeywords.fits", "x_offset", value=1.9, ext=1)
    phdr = fits.getheader("overridekeywords.fits", ext=0)
    hdr = fits.getheader("overridekeywords.fits", ext=1)
    # Actual Values
    val1 = True
    val2 = switches.values()
    val3 = phdr.values()
    val4 = reffiles["flt_hdr"]
    # Test
    cosutil.overrideKeywords(phdr, hdr, info, switches, reffiles)
    phdr = fits.getheader("overridekeywords.fits", ext=0)
    # Verify
    assert val1 == phdr["statflag"]
