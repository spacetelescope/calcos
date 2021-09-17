from __future__ import division, print_function         # confidence high

from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    __version__ = 'UNKNOWN'

# Hack fix for RTD
try:
    from .calcos import *
except ImportError:
    pass

__usage__ = """

1. To run this task from within Python::

    >>> import calcos
    >>> calcos.calcos("rootname_asn.fits")
    >>> calcos.calcos("rootname_rawtag_a.fits")

2. To run this task from the operating system command line::

    # Calibrate an entire association.
    % calcos rootname_asn.fits

    # Calibrate xyz_rawtag_a.fits (and xyz_rawtag_b.fits, if present)
    % calcos xyz_rawtag_a.fits
"""

if __doc__:
    __doc__ += __usage__
else:
    __doc__ = __usage__

