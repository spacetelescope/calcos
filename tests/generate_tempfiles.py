from astropy.io import fits
import numpy as np
import os
import random


def generate_fits_file(file):
    """
    Opens a corrtag file for testing.
    @param file: the file path.
    @return: the HDU_List
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
    prim_hdu.header.set('STATFLAG', True, 'Calculate statistics')

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


def create_disptab_file(file=None):
    """
    Parameters
    ----------
    file: str
        name of the temp file to be created.

    Returns
    -------
        name of the temp file created.
    """
    if file is None:
        file = 'temp_disptab.fits'

    prim_hdu = fits.PrimaryHDU()
    prim_hdu.header.set('ORIGIN', 'NOAO-IRAF FITS Image Kernel July 2003', 'FITS file originator')
    prim_hdu.header.set('IRAF-TM', '2013-06-17T20:41:50', 'Time of last modification')
    prim_hdu.header.set('DATE', '2020-09-02T14:59:36.375298', 'Creation UTC (CCCC-MM-DD) date of FITS')
    prim_hdu.header.set('FILETYPE', 'DISPERSION RELATION REFERENCE TABLE')
    prim_hdu.header.set('VCALCOS', '3.3.9')
    prim_hdu.header.set('PEDIGREE', 'INFLIGHT 03/08/2009 21/07/2012')
    prim_hdu.header.set('USEAFTER', 'Aug 17 2009 00:00:00', )
    prim_hdu.header.set('DESCRIP', 'Updated dispersion values for bluemodes (LP2)')
    prim_hdu.header.set('GIT_TAG', 'disp_spwcs_2020Aug6_bm')
    prim_hdu.header.set('OBSTYPE', 'SPECTROSCOPIC')
    prim_hdu.header.set('INSTRUME', 'COS')
    prim_hdu.header.set('DETECTOR', 'FUV')
    prim_hdu.header.set('COSCOORD', 'USER')
    prim_hdu.header.set('LIFE_ADJ', 2)
    fits.BinTableHDU.add_checksum(prim_hdu)
    fits.BinTableHDU.add_datasum(prim_hdu)
    prim_hdu.header.set('FILENAME', file)
    prim_hdu.header.set('ROOTNAME', file[:-5].upper())
    prim_hdu.header.set('COMMENT', 'Reference file was created by Michael Asfaw.')

    # data pool.
    seg_val = ['FUVA', 'FUVB']
    opt_val = ['G140L', 'G130M', 'G160M']
    aper_val = ['BOA', 'PSA', 'WSA']
    cen_val = [1096, 1105, 1291, 1280, 1300, 1318, 1309, 1327, 1577, 1589, 1600, 1611, 1623]
    nelem_val = [2, 3]
    coef_val = [[1.04502424e+03, 9.95559148e-03, 0.00000000e+00, 0.00000000e+00],
                [8.92490503e+02, 9.92335752e-03, 0.00000000e+00, 0.00000000e+00],
                [9.32680208e+02, 9.92585053e-03, 0.00000000e+00, 0.00000000e+00],
                [-3.59416040e+02, 7.94367813e-02, 3.13384838e-08, -0.00000000e+00],
                [1.03015488e+03, 7.94367813e-02, 3.13384838e-08, -0.00000000e+00],
                [-4.28841955e+01, 7.83294675e-02, 3.59806123e-08, -0.00000000e+00],
                [1.57406737e+03, 1.22419996e-02, 0.00000000e+00, -0.00000000e+00]]
    d_tv_val = [0, -0, -44.41, -42.21, -43.88, -41.68]
    d_val = [0.0000000, -2.1475983, -36.35, 42.3271, 20.2402, 66.0997, -231.6058, -545.549,
             -539.081, -497.598, -493.395, -460.714, -449.035, 4.49852, 27.4868, 53.2402]

    seg_arr = ['FUVA', 'FUVB', 'FUVA', 'FUVB', 'FUVA', 'FUVB']
    opt_elem_arr = ['G130M', 'G130M', 'G130M', 'G130M', 'G130M', 'G130M']
    aper_arr = ['BOA', 'BOA', 'PSA', 'PSA', 'WCA', 'WCA']
    cenwave_arr = [1055, 1055, 1055, 1055, 1055, 1055]
    nelem_arr = [2, 2, 2, 2, 2, 2]
    coef_arr = [[1.04502424e+03, 9.95559148e-03, 0.00000000e+00, 0.00000000e+00],
                [8.92490503e+02, 9.92335752e-03, 0.00000000e+00, 0.00000000e+00],
                [1.04502424e+03, 9.95559148e-03, 0.00000000e+00, 0.00000000e+00],
                [8.92490503e+02, 9.92335752e-03, 0.00000000e+00, 0.00000000e+00],
                [1.04240540e+03, 1.00387000e-02, 0.00000000e+00, 0.00000000e+00],
                [8.89643680e+02, 9.93920000e-03, 0.00000000e+00, 0.00000000e+00]]
    d_tv_arr = [0., 0., 0., 0., 0., 0.]
    d_arr = [0., 0., 0., 0., 0., 0.]
    for i in range(90):
        if i % 2 == 0:
            seg_arr.append(seg_val[0])  # FUVA
            opt_elem_arr.append(opt_val[0])  # G140L
            aper_arr.append(aper_val[0])  # BOA
            cenwave_arr.append(cen_val[3])  # 1280
            nelem_arr.append(nelem_val[0])  # 2
            coef_arr.append(coef_val[0])  # [1.04502424e+03, 9.95559148e-03, 0.00000000e+00, 0.00000000e+00]
            d_tv_arr.append(d_tv_val[0])  # 0
            d_arr.append(d_val[0])  # 0.00000000
        else:
            seg_arr.append(seg_val[1])  # FUVB
            opt_elem_arr.append(opt_val[1])  # G130M
            aper_arr.append(aper_val[1])  # PSA
            cenwave_arr.append(cen_val[8])  # 1577
            nelem_arr.append(nelem_val[1])  # 3
            coef_arr.append(coef_val[1])  # [8.92490503e+02, 9.92335752e-03, 0.00000000e+00, 0.00000000e+00]
            d_tv_arr.append(d_tv_val[2])  # -44.41
            d_arr.append(d_val[2])  # -36.35
        d_arr.append(random.choice(d_val))

    # bin table
    segment = fits.Column(name='SEGMENT', format='4A', disp='A4', array=np.array(seg_arr))
    opt_elem = fits.Column(name='OPT_ELEM', format='5A', disp='A8', array=np.array(opt_elem_arr))
    aperture = fits.Column(name='APERTURE', format='3A', disp='A4', array=np.array(aper_arr))
    cenwave = fits.Column(name='CENWAVE', format='J', unit='angstrom', disp='I5', array=np.array(cenwave_arr))
    nelem = fits.Column(name='NELEM', format='J', disp='I5', array=np.array(nelem_arr))
    coeff = fits.Column(name='COEFF', format='4D', disp='G25.15', array=np.array(coef_arr))
    d_tv = fits.Column(name='D_TV03', format='E', array=np.array(d_tv_arr))
    d = fits.Column(name='D', format='E', disp='G12.6', array=np.array(d_tv_arr))
    # parse the columns
    col_defs = fits.ColDefs([segment, opt_elem, aperture, cenwave, nelem, coeff, d_tv, d])
    hdu = fits.BinTableHDU.from_columns(col_defs)
    fits.BinTableHDU.add_checksum(hdu)
    fits.BinTableHDU.add_datasum(hdu)
    hdu_list = fits.HDUList([prim_hdu, hdu])

    # write the hdu to a fits file
    if os.path.exists(file):
        os.remove(file)
    hdu_list.writeto(file)
    return file


def create_count_file(file=None):
    """
    creates a temp count file for testing avg_image.

    Parameters
    ----------
    file: str
        the filename string

    Returns
    -------
    filename string
    """
    if file is None:
        file = 'test_count.fits'
    rootname = file[:file.index('_')]
    prim_hdu = fits.PrimaryHDU()
    prim_hdu.header.set('NEXTEND', '3', 'Number of standard extensions')
    prim_hdu.header.set('DATE', '2021-07-26', 'date this file was written (yyyy-mm-dd)')
    prim_hdu.header.set('FILENAME', file, 'name of file')
    prim_hdu.header.set('FILETYPE', 'SCI', 'type of data found in data file')
    prim_hdu.header.set('TELESCOP', 'HST', 'telescope used to acquire data')
    prim_hdu.header.set('EQUINOX', 2000.0, 'equinox of celestial coord. system')
    prim_hdu.header.set('ROOTNAME', rootname, 'rootname of the observation set')
    prim_hdu.header.set('IMAGETYP', 'ACCUM', 'type of exposure identifier')
    prim_hdu.header.set('PRIMESI', 'COS', 'instrument designated as prime')
    prim_hdu.header.set('TARGNAME', '1235867', 'proposer\'s')
    prim_hdu.header.set('RA_TARG', 1.500923600000E+02, 'right ascension of the target (deg) (J2000)')
    prim_hdu.header.set('DEC_TARG', 2.361461111111E+00, 'declination of the target (deg) (J2000)')
    prim_hdu.header.set('PROPOSID', 13313, 'PEP proposal identifier')
    prim_hdu.header.set('LINENUM', '03.001', 'proposal logsheet line number')
    prim_hdu.header.set('PR_INV_L', 'Boquien', 'last name of principal investigator')
    prim_hdu.header.set('PR_INV_F', 'Mederic', 'first name of principal investigator')
    prim_hdu.header.set('PR_INV_M', ' ', 'middle name / initial of principal investigat')
    prim_hdu.header.set('OPT_ELEM', 'MIRRORA', 'optical element in use')
    prim_hdu.header.set('DETECTOR', 'NUV', 'FUV OR NUV')
    prim_hdu.header.set('OBSMODE', 'ACCUM', 'operating mode')
    prim_hdu.header.set('OBSTYPE', 'IMAGING', 'imaging or spectroscopic')
    prim_hdu.header.set('APERTURE', 'PSA', 'aperture name')

    # imageHDU
    imgHDU = fits.ImageHDU()
    imgHDU.header.set('EXTNAME', 'SCI', 'extension version number')
    imgHDU.header.set('EXTVER', 1, 'extension version number')
    imgHDU.header.set('ROOTNAME', rootname, 'rootname of the observation set')
    imgHDU.header.set('EXPNAME', rootname, 'exposure identifier')
    imgHDU.header.set('ASN_MTYP', 'EXP-FP', 'Role of the Member in the Association')
    imgHDU.header.set('WCSAXES', 2, 'number of World Coordinate System axes')
    imgHDU.header.set('CD1_1', 6.12377E-06, 'partial of first axis coordinate w.r.t. x')
    imgHDU.header.set('CD1_2', -3.72298E-06, 'partial of first axis coordinate w.r.t. y')
    imgHDU.header.set('CD2_1', -3.72298E-06, 'partial of second axis coordinate w.r.t. x')
    imgHDU.header.set('CD2_2', 6.12377E-06, 'partial of first axis coordinate w.r.t. y')
    imgHDU.header.set('LTV1', 0.0, 'offset in X to subsection start')
    imgHDU.header.set('LTV2', 0.0, 'offset in Y to subsection start')
    imgHDU.header.set('LTM1_1', 1.0, 'reciprocal of sampling rate in X')
    imgHDU.header.set('LTM2_2', 1.0, 'reciprocal of sampling rate in Y')
    imgHDU.header.set('RA_APER', 1.500923600000E+02, 'RA of reference aperture center')
    imgHDU.header.set('DEC_APER', 2.361461111111E+00, 'Declination of reference aperture center')
    imgHDU.header.set('PA_APER', -3.129775238037E+01, 'Position Angle of reference aperture center (de')
    imgHDU.header.set('DISPAXIS', 0, 'dispersion axis; 1 = axis 1, 2 = axis 2, none')
    imgHDU.header.set('SHIFT1A', 0.0, 'wavecal shift determined spectral strip A(pixel')
    imgHDU.header.set('SHIFT1B', 0.0, 'wavecal shift determined spectral strip B(pixel')
    imgHDU.header.set('SHIFT1C', 0.0, 'wavecal shift determined spectral strip C(pixel')
    imgHDU.header.set('SHIFT2A', 0.0, 'Offset in cross-dispersion direction, A (pixels')
    imgHDU.header.set('SHIFT2B', 0.0, 'Offset in cross-dispersion direction, B (pixels')
    imgHDU.header.set('SHIFT2C', 0.0, 'Offset in cross-dispersion direction, C (pixels')
    imgHDU.header.set('DPIXEL1A', 0.0, 'Average fraction part of pixel coordinate(pixel')
    imgHDU.header.set('DPIXEL1B', 0.0, 'Average fraction part of pixel coordinate(pixel')
    imgHDU.header.set('DPIXEL1C', 0.0, 'Average fraction part of pixel coordinate(pixel')
    imgHDU.header.set('SP_LOC_A', -999.0, 'location of spectral extraction region A')
    imgHDU.header.set('SP_LOC_B', -999.0, 'location of spectral extraction region B')
    imgHDU.header.set('SP_LOC_C', -999.0, 'location of spectral extraction region C')
    imgHDU.header.set('SP_OFF_A', -999.0, 'XD spectrum offset from expected loc (stripe A)')
    imgHDU.header.set('SP_OFF_B', -999.0, 'XD spectrum offset from expected loc (stripe B)')
    imgHDU.header.set('SP_OFF_C', -999.0, 'XD spectrum offset from expected loc (stripe C)')
    imgHDU.header.set('SP_NOM_A', -999.0, 'Expected location of spectrum in XD (stripe A)')
    imgHDU.header.set('SP_NOM_B', -999.0, 'Expected location of spectrum in XD (stripe B)')
    imgHDU.header.set('SP_NOM_C', -999.0, 'Expected location of spectrum in XD (stripe C)')
    imgHDU.header.set('SP_SLP_A', -999.0, 'slope of stripe A spectrum')
    imgHDU.header.set('SP_SLP_B', -999.0, 'slope of stripe B spectrum')
    imgHDU.header.set('SP_SLP_C', -999.0, 'slope of stripe C spectrum')
    imgHDU.header.set('SP_HGT_A', -999.0, 'height (pixels) of stripe A extraction region')
    imgHDU.header.set('SP_HGT_B', -999.0, 'height (pixels) of stripe B extraction region')
    imgHDU.header.set('SP_HGT_C', -999.0, 'height (pixels) of stripe C extraction region')
    imgHDU.header.set('X_OFFSET', 0, 'offset of detector in a calibrated image')
    imgHDU.header.set('B_HGT1_A', -999.0, 'height of spectral background 1 stripe A')
    imgHDU.header.set('B_HGT1_B', -999.0, 'height of spectral background 1 stripe B')
    imgHDU.header.set('B_HGT1_C', -999.0, 'height of spectral background 1 stripe C')
    imgHDU.header.set('B_HGT2_A', -999.0, 'height of spectral background 2 stripe A')
    imgHDU.header.set('B_HGT2_B', -999.0, 'height of spectral background 2 stripe B')
    imgHDU.header.set('B_HGT2_C', -999.0, 'height of spectral background 2 stripe C')
    imgHDU.header.set('B_BKG1_A', -999.0, 'location of spectral background 1 stripe A')
    imgHDU.header.set('B_BKG1_B', -999.0, 'location of spectral background 1 stripe B')
    imgHDU.header.set('B_BKG1_C', -999.0, 'location of spectral background 1 stripe C')
    imgHDU.header.set('B_BKG2_A', -999.0, 'location of spectral background 2 stripe A')
    imgHDU.header.set('B_BKG2_B', -999.0, 'location of spectral background 2 stripe B')
    imgHDU.header.set('B_BKG2_C', -999.0, 'location of spectral background 2 stripe C')
    imgHDU.header.set('ORIENTAT', -31.2978, 'position angle of image y axis (deg. e of n)')
    imgHDU.header.set('SUNANGLE', 125.602852, 'angle between sun and V1 axis')
    imgHDU.header.set('MOONANGL', 55.101593, 'angle between moon and V1 axis')
    imgHDU.header.set('SUN_ALT', 69.923927, 'altitude of the sun above Earth\'s limb')
    imgHDU.header.set('FGSLOCK', 'FINE              ', 'commanded FGS lock (FINE,COARSE,GYROS,UNKNOWN)')
    imgHDU.header.set('GYROMODE', 'T', 'number of gyros scheduled, T=3+OBAD')
    imgHDU.header.set('REFFRAME', 'ICRS    ', 'guide star catalog version')
    imgHDU.header.set('DATE-OBS', '2014-04-15', 'UT date of start of observation (yyyy-mm-dd)')
    imgHDU.header.set('TIME-OBS', '09:20:03', 'UT time of start of observation (hh:mm:ss)')
    imgHDU.header.set('EXPSTART', 5.676238893322E+04, 'exposure start time (Modified Julian Date)')
    imgHDU.header.set('EXPEND', 5.676239032211E+04, 'exposure end time (Modified Julian Date)')
    imgHDU.header.set('EXPTIME', 120.000000, 'exposure duration (seconds)--calculated')
    imgHDU.header.set('EXPFLAG', 'NORMAL       ', 'Exposure interruption indicator')
    imgHDU.header.set('EXPSTRTJ', 2.456762888933E+06, 'start time (JD) of exposure')
    imgHDU.header.set('EXPENDJ', 2.456762890322E+06, 'end time (JD) of exposure')
    imgHDU.header.set('PLANTIME', 120.0, 'Planned exposure time (seconds)')
    imgHDU.header.set('NINTERPT', 0, 'Number of Exposure Interrupts')
    imgHDU.header.set('V_HELIO', -999.0, 'Geocentric to heliocentric velocity')
    imgHDU.header.set('V_LSRSTD', 0.0, 'Heliocentric to standard solar LSR')
    imgHDU.header.set('ORBITPER', 5719.436344, 'Orbital Period used on board for Doppler corr.')
    imgHDU.header.set('DOPPER', 0.0, 'Doppler shift period (seconds)')
    imgHDU.header.set('DOPPMAG', -1.000000, 'Doppler shift magnitude (low-res pixels)')
    imgHDU.header.set('DOPPMAGV', 7.014199, 'Doppler shift magnitude (Km/sec)')
    imgHDU.header.set('DOPPON', 'F', 'Doppler correction flag')
    imgHDU.header.set('DOPPZERO', 56762.336061, 'Commanded time of zero Doppler shift (MJD)')
    imgHDU.header.set('ORBTPERT', -1.0, 'Orbital Period used on board for Doppler corr.')
    imgHDU.header.set('DOPMAGT', -1.0, 'Doppler shift magnitude (low-res pixels)')
    imgHDU.header.set('DOPPONT', 'F', 'Doppler correction flag')
    imgHDU.header.set('DOPZEROT', -1.0, 'Commanded time of zero Doppler shift (MJD)')
    imgHDU.header.set('GLOBRATE', 809.025, 'global count rate')
    imgHDU.header.set('NSUBARRY', 1, 'Number of subarrays (1-8)')
    imgHDU.header.set('CORNER0X', 0, 'subarray axis1 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER1X', 0, 'subarray axis1 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER2X', 0, 'subarray axis1 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER3X', 0, 'subarray axis1 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER4X', 0, 'subarray axis1 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER5X', 0, 'subarray axis1 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER6X', 0, 'subarray axis1 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER7X', 0, 'subarray axis1 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER0Y', 0, 'subarray axis2 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER1Y', 0, 'subarray axis2 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER2Y', 0, 'subarray axis2 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER3Y', 0, 'subarray axis2 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER4Y', 0, 'subarray axis2 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER5Y', 0, 'subarray axis2 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER6Y', 0, 'subarray axis2 corner pt in unbinned dect. pix')
    imgHDU.header.set('CORNER7Y', 0, 'subarray axis2 corner pt in unbinned dect. pix')
    imgHDU.header.set('SIZE0X', 1024, 'subarray 0 axis1 size in unbinned detector pixe')
    imgHDU.header.set('SIZE1X', 0, 'subarray 1 axis1 size in unbinned detector pixe')
    imgHDU.header.set('SIZE2X', 0, 'subarray 2 axis1 size in unbinned detector pixe')
    imgHDU.header.set('SIZE3X', 0, 'subarray 3 axis1 size in unbinned detector pixe')
    imgHDU.header.set('SIZE4X', 0, 'subarray 4 axis1 size in unbinned detector pixe')
    imgHDU.header.set('SIZE5X', 0, 'subarray 5 axis1 size in unbinned detector pixe')
    imgHDU.header.set('SIZE6X', 0, 'subarray 6 axis1 size in unbinned detector pixe')
    imgHDU.header.set('SIZE7X', 0, 'subarray 7 axis1 size in unbinned detector pixe')
    imgHDU.header.set('SIZE0Y', 1024, 'subarray 0 axis2 size in unbinned detector pixe')
    imgHDU.header.set('SIZE1Y', 0, 'subarray 1 axis2 size in unbinned detector pixe')
    imgHDU.header.set('SIZE2Y', 0, 'subarray 2 axis2 size in unbinned detector pixe')
    imgHDU.header.set('SIZE3Y', 0, 'subarray 3 axis2 size in unbinned detector pixe')
    imgHDU.header.set('SIZE4Y', 0, 'subarray 4 axis2 size in unbinned detector pixe')
    imgHDU.header.set('SIZE5Y', 0, 'subarray 5 axis2 size in unbinned detector pixe')
    imgHDU.header.set('SIZE6Y', 0, 'subarray 6 axis2 size in unbinned detector pixe')
    imgHDU.header.set('SIZE7Y', 0, 'subarray 7 axis2 size in unbinned detector pixe')

    imgHDU.header.set('PHOTFLAM', 4.816554456084E-18, 'inverse sensitivity, ergs/s/cm2/Ang per count/s')
    imgHDU.header.set('PHOTFNU ', 8.64540709538E-30, 'inverse sensitivity, ergs/s/cm2/Hz per count/s')
    imgHDU.header.set('PHOTBW  ', 382.88, 'RMS bandwidth of filter plus detector (Ang)')
    imgHDU.header.set('PHOTPLAM', 2319.7, 'Pivot wavelength')
    imgHDU.header.set('PHOTZPT ', -21.10, 'ST magnitude zero point')

    img_errHDU = fits.ImageHDU()
    img_errHDU.header.set('EXTNAME', 'ERR', 'extension name')
    img_errHDU.header.set('EXTVER', 1, 'extension version name')
    img_errHDU.header.set('ROOTNAME', rootname, 'rootname of the observation set')
    img_errHDU.header.set('EXPNAME', rootname, 'exposure identifier')
    img_errHDU.header.set('DATAMIN ', 0.0, 'the minimum value of the data')
    img_errHDU.header.set('DATAMAX ', 0.0, 'the maximum value of the data')
    img_errHDU.header.set('BUNIT   ', 'count /s', 'brightness units')

    dqHDU = fits.ImageHDU()
    dqHDU.header.set('EXTNAME', 'DQ', 'extension name')
    dqHDU.header.set('EXTVER', 1, 'extension version name')
    dqHDU.header.set('ROOTNAME', rootname, 'rootname of the observation set')
    dqHDU.header.set('EXPNAME', rootname, 'exposure identifier')
    dqHDU.header.set('DATAMIN ', 0.0, 'the minimum value of the data')
    dqHDU.header.set('DATAMAX ', 0.0, 'the maximum value of the data')
    dqHDU.header.set('BUNIT   ', 'UNITLESS', 'brightness units')
    sci_data = np.zeros((1024, 1024))
    err_data = np.zeros((1024, 1024))
    dq_data = np.zeros((1024, 1024), dtype='int16')
    for i in range(1024):
        for j in range(1024):
            sci_data[i][j] = 0.00000234
            err_data[i][j] = 0.01534185
            dq_data[i][j] = 8
    imgHDU.data = np.array(sci_data)
    img_errHDU.data = np.array(err_data)
    dqHDU.data = np.array(dq_data)
    hdu_list = fits.HDUList([prim_hdu, imgHDU, img_errHDU, dqHDU])

    if os.path.exists(file):
        os.remove(file)
    hdu_list.writeto(file)
    return file