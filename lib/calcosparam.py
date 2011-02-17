from __future__ import division         # confidence high

# This file defines parameters used by calcos.

CALCOS_VERSION_NUMBER = "2.14.2"
CALCOS_VERSION_DATE = "2011-02-17"
CALCOS_VERSION = "%s (%s)" % (CALCOS_VERSION_NUMBER, CALCOS_VERSION_DATE)

# These are the values to indicate the detector (original) and user
# (flipped or rotated) COS coordinates.
DETECTOR_COORDINATES = "DETECTOR"
USER_COORDINATES     = "USER"

SPEED_OF_LIGHT = 299792.458     # km/s

DAYS_PER_YEAR = 365.25
SEC_PER_DAY = 86400.

MJD_TO_JD = 2400000.5           # add to MJD to get Julian Day Number

# Live time estimates should not differ by more than this fraction of
# the live time.
LIVETIME_CRITERION = 0.1

# This is the wavelength below which no significant flux could be detected.
MIN_WAVELENGTH = 900.           # Angstroms

# These give the axis lengths of the FUV and NUV detectors, in pixels.
FUV_X = 16384                   # more rapidly varying axis
FUV_Y = 1024
NUV_X = 1024                    # more rapidly varying axis
NUV_Y = 1024

# X_OFFSET is the offset of the detector in a calibrated image.
# Pixel X in a calibrated image = XFULL + X_OFFSET
FUV_X_OFFSET = 0
FUV_EXTENDED_X = FUV_X
NUV_X_OFFSET = 100
NUV_EXTENDED_X = NUV_X + 250

# These are the default binning factors for FUV and NUV "calcos sum" images.
FUV_BIN_X = 1
FUV_BIN_Y = 1
NUV_BIN_X = 1
NUV_BIN_Y = 1

# These give the number of spectra per detector (used in extract.py).
FUV_SPECTRA = 1                 # one spectrum on one FUV segment
NUV_SPECTRA = 3                 # three stripes on NUV detector

# These are the possible values for verbosity.
QUIET = 0
VERBOSE = 1
VERY_VERBOSE = 2

# These are the possible values for the TAGFLASH keyword, and corresponding
# integer codes.
TAGFLASH_NONE = "NONE"
TAGFLASH_AUTO = "AUTO"
TAGFLASH_UNIFORMLY_SPACED = "UNIFORMLY SPACED"
TAGFLASH_TYPE_NONE = 0
TAGFLASH_TYPE_AUTO = 1
TAGFLASH_TYPE_UNIFORMLY_SPACED = 2

# This is the list of segment-specific (or in some cases stripe-specific)
# keywords, with "X" (case sensitive) replaced by "a", "b" or "c".
segment_specific_keywords = \
    ["stimX_lx", "stimX_ly", "stimX_rx", "stimX_ry",
     "stimX0lx", "stimX0ly", "stimX0rx", "stimX0ry",
     "stimXslx", "stimXsly", "stimXsrx", "stimXsry",
     "npha_X", "phalowrX", "phaupprX",
     "tbrst_X", "nbrst_X", "tbadt_X", "nbadt_X",
     "nout_X", "nbadevtX",
     "exptimeX", "neventsX",
     "globrt_X",
     "deadrt_X", "deadmt_X", "livetm_X",
     "sp_loc_X", "sp_off_X", "sp_nom_X", "sp_slp_X", "sp_hgt_X",
     "b_bkg1_X", "b_bkg2_X",
     "b_hgt1_X", "b_hgt2_X",
     "shift1X", "shift2X", "dpixel1X",
     "chi_sq_X", "ndf_X"]

# The pulse height values range from 0 to 127.  The values in the PHA
# column of an EVENTS table, however, come from a five-bit value, i.e.
# the last two bits have been truncated, resulting in their being a
# factor of four smaller.
TWO_BITS = 4

# The following three parameters are used by getTable.
# NOT_APPLICABLE will be assigned as the value of a keyword that is
# missing from the header; this is done because some keywords may
# actually not be present, while others that are not relevant will be
# present but have the value "N/A".
STRING_WILDCARD = "ANY"
NOT_APPLICABLE = "N/A"
INT_WILDCARD = -1

# These are the data quality flags.
DQ_OK = 0                       # no anomalous condition noted
DQ_SOFTERR = 1                  # Reed-Solomon error
DQ_DETECTOR_FLAW = 2            # detector flaw
DQ_DETECTOR_SHADOW = 4          # FUV grid shadow mark or NUV vignetting
DQ_NEAR_EDGE = 8                # spectrum near an edge of the detector
DQ_DEAD = 16                    # dead spot
DQ_HOT = 32                     # hot spot
DQ_BURST = 64                   # count rate implies a burst (FUV)
DQ_PIXEL_OUT_OF_BOUNDS = 128    # pixel out of bounds
DQ_DATA_FILL = 256              # data fill
DQ_PHA_OUT_OF_BOUNDS = 512      # pulse height is either too low or too high
DQ_UNUSED = 1024                # [currently unused]
DQ_BAD_TIME = 2048              # time is within a bad time interval
DQ_BACKGROUND_FEATURE = 4096    # background feature
DQ_LOW_GAIN = 8192              # low gain area
DQ_UNUSED = 16384               # [currently unused]

# Use this when binning TIME-TAG data to images, or extracting spectra from
# TIME-TAG data.
SERIOUS_DQ_FLAGS = (DQ_BURST | DQ_BAD_TIME | DQ_PHA_OUT_OF_BOUNDS)
