# input files are x1d, flt, corrtag, and counts

from calcos.x1d import *


def generate_fits_file(file):
    """
    Opens a corrtag file for testing.
    @param file: the file path.
    @return: the table (BinTableHDU).
    """
    # define columns with some random values (dummy corrtag file)
    time = fits.Column(name='TIME', format='1E', unit='s', array=np.random.rand(10, 1))
    rawx = fits.Column(name='RAWX', format='1I', unit='pixel', array=np.random.rand(10, 1))
    rawy = fits.Column(name='RAWY', format='1I', unit='pixel', array=np.random.rand(10, 1))
    xcorr = fits.Column(name='XCORR', format='1E', unit='pixel', array=np.random.rand(10, 1))
    ycorr = fits.Column(name='YCORR', format='1E', unit='pixel', array=np.random.rand(10, 1))
    xdopp = fits.Column(name='XDOPP', format='1E', unit='pixel', array=np.random.rand(10, 1))
    xfull = fits.Column(name='XFULL', format='1E', unit='pixel', coord_type='WAVE', coord_unit='angstrom',
                        coord_ref_point=8192.999999999998, coord_ref_value=1359.629173113225,
                        coord_inc=0.009967382065951824, array=np.random.rand(10, 1))
    yfull = fits.Column(name='YFULL', format='1E', unit='pixel', coord_type='ANGLE', coord_unit='deg',
                        coord_ref_point=487.3817151221292, coord_ref_value=0.0, coord_inc=2.777778e-05,
                        array=np.random.rand(10, 1))
    wavelength = fits.Column(name='WAVELENGTH', format='1E', unit='angstrom', disp='F9.4', array=np.random.rand(10, 1))
    epsilon = fits.Column(name='EPSILON', format='1E', array=np.random.rand(10, 1))
    dq = fits.Column(name='DQ', format='1I', array=np.random.rand(10, 1))
    pha = fits.Column(name='PHA', format='1B', array=np.random.rand(10, 1))
    col_defs = fits.ColDefs([time, rawx, rawy, xcorr, ycorr, xdopp, xfull, yfull, wavelength, epsilon, dq, pha])
    hdu = fits.BinTableHDU.from_columns(col_defs)
    hdu.name = "EVENTS"
    hdu.header.set('TIME', '04/09/21')

    prim_hdu = fits.PrimaryHDU()

    # define cols for gti header
    """name = 'START'; format = '1D'; unit = 'seconds'
    name = 'STOP'; format = '1D'; unit = 'seconds'"""
    start = fits.Column('START', format='1D', unit='seconds', array=np.random.rand(10, 1))
    stop = fits.Column('STOP', format='1D', unit='seconds', array=np.random.rand(10, 1))
    col_defs = fits.ColDefs([start,stop])
    gti_hdu = fits.BinTableHDU.from_columns(col_defs)
    gti_hdu.name = "GTI"

    # define cols for timeline header
    time = fits.Column('TIME', format='1E', unit='s', disp='F8.3', array=np.random.rand(10, 1))
    longitude = fits.Column('LONGITUDE', format='1E', unit='degree', disp='F10.6', coord_type='RA---TAN',
                            coord_unit='angstrom', coord_ref_point=1.0, coord_ref_value=-999.0,
                            array=np.random.rand(10, 1))
    latitude = fits.Column('LATITUDE', format='1E', unit='degree', disp='F10.6', coord_type='ANGLE', coord_unit='deg',
                           coord_ref_point=1.0, coord_ref_value=-999.0, array=np.random.rand(10, 1))
    sun_alt = fits.Column('SUN_ALT', format='1E', unit='degree', disp='F6.2', array=np.random.rand(10, 1))
    sun_zd = fits.Column('SUN_ZD', format='1E', unit='degree', disp='F6.2', array=np.random.rand(10, 1))
    target_alt = fits.Column('TARGET_ALT', format='1E', unit='degree', disp='F6.2', array=np.random.rand(10, 1))
    radial_vel = fits.Column('RADIAL_VEL', format='1E', unit='km /s', disp='F7.5', array=np.random.rand(10, 1))
    shift1 = fits.Column('SHIFT1', format='1E', unit='pixel', disp='F7.3', array=np.random.rand(10, 1))
    ly_alpha = fits.Column('LY_ALPHA', format='1E', unit='count /s', disp='G15.6', array=np.random.rand(10, 1))
    oi_1305 = fits.Column('OI_1304', format='1E', unit='count /s', disp='G15.6', array=np.random.rand(10, 1))
    oi_1356 = fits.Column('OI_1356', format='1E', unit='count /s', disp='G15.6', array=np.random.rand(10, 1))
    darkrate = fits.Column('DARKRATE', format='1E', unit='count /s /pixel', disp='G15.6', array=np.random.rand(10, 1))
    col_defs = fits.ColDefs(
        [time, longitude, latitude, sun_alt, sun_zd, target_alt, radial_vel, shift1, ly_alpha, oi_1305, oi_1356,
         darkrate])
    time_hdu = fits.BinTableHDU.from_columns(col_defs)
    time_hdu.name = "TIMELINE"
    # create an hdu list
    hdu_list = fits.HDUList([prim_hdu, hdu, gti_hdu, time_hdu])
    # write the hdu to a fits file
    if os.path.exists(file):
        os.remove(file)
    hdu_list.writeto(file)
    hdu_list = fits.open(file)
    return hdu_list


# generate_fits_file("test.fits")

def test_get_columns():
    """
    Test if the function is returning the right column fields
    """
    # Setup
    test_data = generate_fits_file("Output/lbgu17qnq_corrtag_a.fits")
    dt = test_data[1].data
    detector = "FUV"
    # detector = "NUV"

    # Truth actual values
    # Testing for FUV
    xfull = dt.field("xfull")
    # xfull = dt.field("xdopp") not necessary for current detector
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
    fd = generate_fits_file("Output/lbgu17qnq_lampflash.fits")
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
    assert target_cols[0] == deleted_cols[0].name
    assert target_cols[1] == deleted_cols[1].name


def test_next_power_of_two():
    """
    Test the next_power_of_two
    @return: none
    """
    # Truth
    next_power = 8

    # Verify
    assert next_power == extract.next_power_of_two(7)


# todo *****************************


def test_add_column_comment():
    # Setup
    ofd = generate_fits_file("Output/myFitsFile.fits")
    comment = "Time in seconds"

    # Exercise
    test_table = extract.add_column_comment(ofd, 'TIME', comment)
    # todo: get the column comment that was added.

    # Verify
    # todo: assert
    assert comment == test_table[1].header.comments['TTYPE1']
