from calcos import airglow


def test_find_airglow_limits():
    # Setup
    # disp_file = "r7p1113rl_disp.fits"
    inf = {"obstype": "SPECTROSCOPIC", "cenwave": 1291, "aperture": "PSA", "detector": "FUV",
           "opt_elem": "G130M", "segment": "FUVA"}
    seg = ["FUVA", "FUVB"]
    disptab = "49g17153l_disp.fits"
    airglow_lines = ["Lyman_alpha", "N_I_1200", "O_I_1304", "O_I_1356", "N_I_1134"]
    actual_pxl = [
        [(None, None), (None, None), (2317.738059581357, 2905.1194325194842), (7678.470190139145, 8170.837276578324),
         (None, None)], [(8872.529023065608, 9372.529023065608), (7404.66295916028, 7721.085082865634),
                         (None, None), (None, None), (842.3693742688937, 1124.1659525618788)]]
    # Test
    test_pxl = [[], []]
    # only works for FUV
    for i in range(2):
        for j in range(len(airglow_lines)):
            limits = airglow.findAirglowLimits(inf, seg[i], disptab, airglow_lines[j])
            if limits is not None:
                x, y = limits
                test_pxl[i].append((x, y))
    # Verify
    for i in range(len(actual_pxl)):
        assert actual_pxl[i] == test_pxl[i]
