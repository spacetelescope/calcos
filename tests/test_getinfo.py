import generate_tempfiles
from calcos import getinfo


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
    # Setup
    temp_file = "ref_file_names_temp.fits"
    hdu = generate_tempfiles.generate_fits_file(temp_file)
    reffiles = {'flatfile_hdr': 'lref$xab1551cl_flat.fits',
                'flatfile': '/grp/hst/cdbs/lref/xab1551cl_flat.fits',
                'hvtab_hdr': 'N/A',
                'hvtab': 'N/A',
                'xwlkfile_hdr': 'lref$14o2013ql_xwalk.fits',
                'xwlkfile': '/grp/hst/cdbs/lref/14o2013ql_xwalk.fits',
                'ywlkfile_hdr': 'lref$14o2013rl_ywalk.fits',
                'ywlkfile': '/grp/hst/cdbs/lref/14o2013rl_ywalk.fits',
                'bpixtab_hdr': 'lref$36d1836ml_bpix.fits',
                'bpixtab': '/grp/hst/cdbs/lref/36d1836ml_bpix.fits',
                'gsagtab_hdr': 'lref$41g2040ol_gsag.fits',
                'gsagtab': '/grp/hst/cdbs/lref/41g2040ol_gsag.fits',
                'spottab_hdr': 'lref$zas1615jl_spot.fits',
                'spottab': '/grp/hst/cdbs/lref/zas1615jl_spot.fits',
                'brftab_hdr': 'lref$x1u1459il_brf.fits',
                'brftab': '/grp/hst/cdbs/lref/x1u1459il_brf.fits',
                'geofile_hdr': 'lref$x1u1459gl_geo.fits',
                'geofile': '/grp/hst/cdbs/lref/x1u1459gl_geo.fits',
                'dgeofile_hdr': 'N/A',
                'dgeofile': 'N/A',
                'twozxtab_hdr': 'N/A',
                'twozxtab': 'N/A',
                'deadtab_hdr': 'lref$s7g1700gl_dead.fits',
                'deadtab': '/grp/hst/cdbs/lref/s7g1700gl_dead.fits',
                'phafile_hdr': 'N/A',
                'phafile': 'N/A',
                'phatab_hdr': 'lref$wc318317l_pha.fits',
                'phatab': '/grp/hst/cdbs/lref/wc318317l_pha.fits',
                'brsttab_hdr': 'N/A',
                'brsttab': 'N/A',
                'badttab_hdr': 'N/A',
                'badttab': 'N/A',
                'tracetab_hdr': 'N/A',
                'tracetab': 'N/A',
                'xtractab_hdr': 'N/A',
                'xtractab': 'N/A',
                'lamptab_hdr': 'lref$23n1744jl_lamp.fits',
                'lamptab': '/grp/hst/cdbs/lref/23n1744jl_lamp.fits',
                'disptab_hdr': 'lref$05i1639ml_disp.fits',
                'disptab': '/grp/hst/cdbs/lref/05i1639ml_disp.fits',
                'fluxtab_hdr': 'lref$23n1744pl_phot.fits',
                'fluxtab': '/grp/hst/cdbs/lref/23n1744pl_phot.fits',
                'imphttab_hdr': 'N/A',
                'imphttab': 'N/A',
                'phottab_hdr': 'None',
                'phottab': 'None',
                'spwcstab_hdr': 'lref$49g17154l_spwcs.fits',
                'spwcstab': '/grp/hst/cdbs/lref/49g17154l_spwcs.fits',
                'wcptab_hdr': 'lref$u1t1616ql_wcp.fits',
                'wcptab': '/grp/hst/cdbs/lref/u1t1616ql_wcp.fits',
                'tdstab_hdr': 'lref$46t1623fl_tds.fits',
                'tdstab': '/grp/hst/cdbs/lref/46t1623fl_tds.fits',
                'proftab_hdr': 'N/A',
                'proftab': 'N/A'}
    test_reffiles = getinfo.getRefFileNames(hdu[0].header)
    for key in test_reffiles.keys():
        assert reffiles[key] == test_reffiles[key]


def test_reset_switches():
    # Setup
    temp_file = "reset_switches_temp.fits"
    hdu = generate_tempfiles.generate_fits_file(temp_file)
    switches = {"badtcorr": 'OMIT',
                "xwlkcorr": 'COMPLETE',
                "ywlkcorr": 'COMPLETE',
                "trcecorr": 'OMIT',
                "algncorr": 'OMIT',
                "deadcorr": 'COMPLETE',
                "flatcorr": 'COMPLETE',
                "doppcorr": 'COMPLETE',
                "tdscorr": 'PERFORM'}
    copy_switches = switches
    reffiles = {'badttab': 'N/A',
                'xwlkfile': '/grp/hst/cdbs/lref/14o2013ql_xwalk.fits',
                'ywlkfile': '/grp/hst/cdbs/lref/14o2013rl_ywalk.fits',
                'gsagtab_hdr': 'lref$41g2040ol_gsag.fits',
                'gsagtab': '/grp/hst/cdbs/lref/41g2040ol_gsag.fits',
                'spottab_hdr': 'lref$zas1615jl_spot.fits',
                'spottab': '/grp/hst/cdbs/lref/zas1615jl_spot.fits',
                'brftab_hdr': 'lref$x1u1459il_brf.fits',
                'tdstab': '/grp/hst/cdbs/lref/46t1623fl_tds.fits'}
    copy_reffiles = reffiles
    # Test
    getinfo.resetSwitches(copy_switches, copy_reffiles)
    # Verify
    for key in copy_switches.keys():
        print(key,":",copy_switches[key])
    print("#"*100)
    for key in copy_reffiles.keys():
        print(key,":",copy_reffiles[key])
    assert True


# test_reset_switches()