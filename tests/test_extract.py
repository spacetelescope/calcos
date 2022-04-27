from calcos.x1d import *
import numpy as np
from generate_tempfiles import generate_fits_file
import pytest
import os, glob

def test_get_columns():
    """
    Test if the function is returning the right column fields
    """
    # Setup
    test_data = generate_fits_file("lbgu17qnq_corrtag_a.fits")
    dt = test_data[1].data
    detector = "FUV"

    # Truth actual values
    # Testing for FUV
    xfull = dt.field("xfull")
    if cosutil.findColumn(dt, "yfull"):
        yfull = dt.field("yfull")
    else:
        yfull = dt.field("ycorr")
    dq = dt.field("dq")
    epsilon = dt.field("epsilon")

    # Test function
    (xf, yf, dq2, epsilon2) = extract.getColumns(test_data, detector)

    # Verify
    np.testing.assert_array_equal(xfull, xf)
    np.testing.assert_array_equal(yfull, yf)
    np.testing.assert_array_equal(dq, dq2)
    np.testing.assert_array_equal(epsilon, epsilon2)


def test_remove_unwanted_column():
    """
    Old column length should be equal to new column length + amount of the removed columns
    """
    # Setup
    target_cols = ['XFULL', 'YFULL']
    # Truth
    fd = generate_fits_file("lbgu17qnq_lampflash.fits")
    table = fd[1].data
    cols = table.columns

    # Test
    fd = extract.remove_unwanted_columns(fd)
    new_cols = fd[1].data.columns
    # Verify
    deleted_cols = set(cols) - set(new_cols)
    deleted_cols = np.array(list(deleted_cols))
    temp_cols = [d.name for d in deleted_cols]
    deleted_cols = deleted_cols[np.argsort(temp_cols)]
    # assert target_cols[0] == deleted_cols[0].name
    # assert target_cols[1] == deleted_cols[1].name


def test_next_power_of_two():
    """
    Test the next_power_of_two
    @return: none
    """
    # Truth
    next_power = 8

    # Verify
    assert next_power == extract.next_power_of_two(7)


def test_add_column_comment():
    # verify if entered comment to a header is present in the fits file.
    # Setup
    ofd = generate_fits_file("myFitsFile.fits")
    comment = "This comment is generated by a unit-test."

    # Exercise
    test_table = extract.add_column_comment(ofd, 'TIME', comment)

    # Verify
    assert comment == test_table[1].header.comments['TTYPE1']
