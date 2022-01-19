from calcos import airglow
from generate_tempfiles import create_disptab_file


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
        [], [], (15421.504705213156, 15738.02214190493), (8853.838672375898, 9135.702216258482)]
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
