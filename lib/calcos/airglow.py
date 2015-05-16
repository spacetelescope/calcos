from __future__ import absolute_import, print_function
from . import cosutil
from . import dispersion
from .calcosparam import *       # parameter definitions

# Half width (pixels) of airglow region to be excluded.
AIRGLOW_LyA = 250.      # Lyman alpha
AIRGLOW_FUV = 100.      # anything but Lyman alpha, but still FUV
AIRGLOW_NUV = 30.       # any NUV airglow line

# Wavelengths in Angstroms of airglow lines.
# The values in the tuple are the wavelengths of the lines in the multiplet.
AIRGLOW_WAVELENGTHS = {"Lyman_alpha": (1215.67,),
                       "N_I_1200": (1199.550, 1200.223, 1200.710),
                       "O_I_1304": (1302.168, 1304.858, 1306.029),
                       "O_I_1356": (1355.598, 1358.512),
                       "N_I_1134": (1134.165, 1134.415, 1134.980)}

# ?                    "O_I_2973": (2973.154,)}

def findAirglowLimits(info, segment, disptab, airglow_line):
    """Find the pixel region corresponding to a set of airglow lines.

    Parameters
    ----------
    info: dictionary
        Keywords and values.

    segment: str
        Segment or stripe name:  "FUVA", "FUVB", "NUVA", "NUVB", "NUVC".

    disptab: str
        Name of reference table for dispersion solution.

    airglow_line: str
        The key for extracting an element from AIRGLOW_WAVELENGTHS.

    Returns
    -------
    tuple (x0, x1) of floats, or None
        x0 and x1 are the left and right pixel numbers of the region
        that should be omitted to avoid contamination by an airglow line.
        These are inclusive limits (pixels), not the elements of a slice.
        None will be returned if the specified line (or multiplet) is off
        the detector, the mode was not found in a reference table, or
        the obstype is not spectroscopic.
    """
    if info["obstype"] != "SPECTROSCOPIC":
        print("Data is not spectroscopic")
        return None

    wl_airglow = AIRGLOW_WAVELENGTHS[airglow_line]

    if info["detector"] == "FUV":
        axis_length = FUV_X
        if airglow_line == "Lyman_alpha":
            exclude = AIRGLOW_LyA
        else:
            exclude = AIRGLOW_FUV
    else:
        axis_length = NUV_X
        exclude = AIRGLOW_NUV

    # This filter is used for both xtractab and disptab.
    filter = {"opt_elem": info["opt_elem"],
              "cenwave": info["cenwave"],
              "segment": segment,
              "aperture": info["aperture"]}
    
    # currently not necessary:  filter["fpoffset"] = info["fpoffset"]
    disp_rel = dispersion.Dispersion(disptab, filter)
    if not disp_rel.isValid():
        cosutil.printWarning("Dispersion relation is not valid; filter is:")
        cosutil.printContinuation(str(filter))
        disp_rel.close()
        return None
    
    min_wl = min(wl_airglow)
    max_wl = max(wl_airglow)
    # First check whether the airglow line is off the detector.
    # NOTE that we assume that wavelength increases with x.
    wl_left_edge = disp_rel.evalDisp(-exclude)
    if max_wl < wl_left_edge:
        disp_rel.close()
        return None
    wl_right_edge = disp_rel.evalDisp(axis_length - 1. + exclude)
    if min_wl > wl_right_edge:
        disp_rel.close()
        return None

    # x_left and x_right are the pixel coordinates for the minimum
    # and maximum airglow wavelengths in the multiplet.
    x_left = float(disp_rel.evalInvDisp(min_wl, tiny=1.e-8))
    x_right = float(disp_rel.evalInvDisp(max_wl, tiny=1.e-8))
    x0 = x_left - exclude
    x1 = x_right + exclude
    disp_rel.close()

    return (x0, x1)
