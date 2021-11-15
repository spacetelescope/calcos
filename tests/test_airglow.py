import os
import random

import numpy as np
from astropy.io import fits

from calcos import airglow


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


def test_find_airglow_limits():
    """
    unit test for find_airglow_limits()
    test ran
    - By providing certain values as dict to be used as filter for finding the dispersion
    - testing for both FUV segments
    - creating a temporary disptab ref file.
    - testing for 5 airglow lines
    - calculating the expected pixel numbers by following the math involved in the actual file
        and referring to the values in the ref file we can get the values upto a descent decimal points.

    Returns
    -------
    pass if expected == actual or fail if not.

    """
    # Setup
    inf = {"obstype": "SPECTROSCOPIC", "cenwave": 1055, "aperture": "PSA", "detector": "FUV",
           "opt_elem": "G130M", "segment": "FUVA"}
    seg = ["FUVA", "FUVB"]
    disptab = create_disptab_file('49g17153l_disp.fits')
    airglow_lines = ["Lyman_alpha", "N_I_1200", "O_I_1304", "O_I_1356", "N_I_1134"]
    actual_pxl = [
        [(15421.504705213156, 15738.02214190493), (8853.838672375898, 9135.702216258482)], []]
    # Test
    test_pxl = [[], []]
    # only works for FUV
    for segment in seg:
        for line in airglow_lines:
            limits = airglow.findAirglowLimits(inf, segment, disptab, line)
            if limits is not None:
                x, y = limits
                test_pxl.append((x, y))
    # Verify
    for i in range(len(actual_pxl)):
        assert actual_pxl[i] == test_pxl[i]
