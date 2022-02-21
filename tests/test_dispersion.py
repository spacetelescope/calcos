import io
import sys

import generate_tempfiles
import numpy as np
from calcos import dispersion


def test_init():
    # setup
    # create a temp disptab
    # this function is only used by the functions in this script.
    tmp_file = "test_disptab_init.fits"
    generate_tempfiles.create_disptab_file(tmp_file)
    t = "49g17153l_disp.fits"
    filters = {'FPOFFSET': 0,
               'NELEM': 2
               }
    disp_obj = dispersion.Dispersion(t, filters, True)

    return disp_obj


def test_info():
    # Setup
    disp_obj = test_init()
    captured_msg = io.StringIO()
    sys.stdout = captured_msg
    expected_str = "filter = {'nelem': 2}\n" \
                   "use_fpoffset = True\n" \
                   "fpoffset = 0\n" \
                   "number of coefficients = 2\n" \
                   "coeff = [1.04502424e+03 9.95559148e-03]\n" \
                   "delta = 0\n" \
                   "number of matching rows = 78\n" \
                   "valid = True\n"
    # Test
    disp_obj.info()
    sys.stdout = sys.__stdout__
    captured_msg = captured_msg.getvalue()
    # Verify
    assert expected_str == captured_msg


def test_isValid():
    # Setup
    disp_obj = test_init()
    expected = True
    # Test
    test = disp_obj.isValid()
    # Verify
    assert expected == test


def test_getNRows():
    # Setup
    disp_obj = test_init()
    expected_val = 78
    # Test
    num_of_rows = disp_obj.getNRows()
    # Verify
    assert expected_val == num_of_rows


def test_getFilter():
    # Setup
    disp_obj = test_init()
    expected_filter = {'nelem': 2}
    # Test
    actual_filter = disp_obj.getFilter()
    # Verify
    assert expected_filter == actual_filter


def test_close():
    # Setup
    disp_obj = test_init()
    expected_filter = {}
    exp_ncoeff = 0
    exp_delta = 0.
    exp_fpoffset = 0
    exp_nrows = 0
    exp_valid = False
    # Test
    disp_obj.close()
    # Verify
    assert expected_filter == disp_obj.getFilter()
    assert exp_ncoeff == disp_obj.ncoeff
    assert exp_delta == disp_obj.delta
    assert exp_fpoffset == disp_obj.fpoffset
    assert exp_nrows == disp_obj.getNRows()
    assert exp_valid == disp_obj.isValid()


def test_evalDisp():
    # Setup
    disp_obj = test_init()
    pix_coord = [3.23, 4.55, 2.0, 1.8]
    expected_sum = [1045.0563993131348, 1045.0695406938933, 1045.04415393561, 1045.0421628173133]
    # Test
    result = disp_obj.evalDisp(pix_coord)
    # Verify
    np.testing.assert_array_equal(expected_sum, result)


def test_evalDerivDisp():
    # Setup
    disp_obj = test_init()
    pix_coord = [3.23, 4.55, 2.0, 1.8]
    actual_derivative = 0.009955591483556625
    # Test
    result = disp_obj.evalDerivDisp(pix_coord)
    # Verify
    assert actual_derivative == result


def test_evalInvDisp():
    # Setup
    disp_obj = test_init()
    wavelegnth = [1200, 1366]
    expected_pixels = [15566.705152910923, 32240.75212181053]
    # Test
    result = disp_obj.evalInvDisp(wavelegnth)
    # Verify
    np.testing.assert_array_equal(expected_pixels, result)
