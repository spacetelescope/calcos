import numpy as np

from calcos import cosutil
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
    # todo: use np.testing
    np.testing.assert_array_equal(truth_values, test1)
    np.testing.assert_array_equal(truth_values, test2)


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


test_get_headers()

def test_write_output_events():
    # Setup
    in_file = "Output/outputEvent.fits"
    out_file = "Output/outputEvents.fits"
    # todo create additional headers in the file to test this function.
    ofd = test_extract.generate_fits_file(in_file)

    # lines = cosutil.writeOutputEvents(in_file, out_file)
    # assert False
    # print("lines = ",lines)


test_write_output_events()
