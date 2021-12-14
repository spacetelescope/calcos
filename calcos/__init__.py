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

from stsci.tools import teal

__taskname__ = "calcos"

__usage__ = """

1. To run this task from within Python::

    >>> import calcos
    >>> calcos.calcos("rootname_asn.fits")
    >>> calcos.calcos("rootname_rawtag_a.fits")

    >>> from stsci.tools import teal
    >>> teal.teal("calcos")

2. To run this task using the TEAL GUI to set the parameters under PyRAF::

    >>> import calcos
    >>> teal calcos                     # or 'epar calcos'

3. To run this task from the operating system command line::

    # Calibrate an entire association.
    % calcos rootname_asn.fits

    # Calibrate xyz_rawtag_a.fits (and xyz_rawtag_b.fits, if present)
    % calcos xyz_rawtag_a.fits
"""

if __doc__:
    __doc__ += __usage__
else:
    __doc__ = __usage__

def localcalcos(input,
                verbosity=1, savetmp=False,
                outdir="",
                find=False, cutoff=None,
                shift_file=None,
                csum=False, raw_csum=False,
                compress=False,
                comp_param="gzip,-0.01",
                binx=None, biny=None,
                stimfile=None, livetimefile=None, burstfile=None,
                print_version=False, print_revision=False):

    if print_version:
        print("%s" % CALCOS_VERSION_NUMBER)
        return
    if print_revision:
        print("%s" % CALCOS_VERSION)
        return

    # Split the input string into words, expand environment variables and
    # wildcards, delete duplicates.
    words = splitInputString(input)
    infiles = uniqueInput(words)

    if not outdir:
        outdir = None

    if not shift_file:
        shift_file = None
    if not stimfile:
        stimfile = None
    if not livetimefile:
        livetimefile = None
    if not burstfile:
        burstfile = None

    only_csum = False

    status = 0
    for input in infiles:
        stat = calcos(input, outdir=outdir, verbosity=verbosity,
                      find_target={"flag": find, "cutoff": cutoff},
                      create_csum_image=csum,
                      raw_csum_coords=raw_csum,
                      only_csum=only_csum,
                      binx=binx, biny=biny,
                      compress_csum=compress,
                      compression_parameters=comp_param,
                      shift_file=shift_file,
                      save_temp_files=savetmp,
                      stimfile=stimfile,
                      livetimefile=livetimefile,
                      burstfile=burstfile)
        status |= stat

    return status

def splitInputString(input):
    """Split on comma and/or space.

    Parameters
    ----------
    input: str
        One or more values (e.g. file names), separated by a comma and/or
        a space.

    Returns
    -------
    words: list of strings
    """

    if isinstance(input, str):
        if input.strip() == "":
            words = [""]
        else:
            # First split on comma, then check for blanks.
            temp_words = input.split(",")
            words = []
            for word in temp_words:
                word = word.strip()
                if word == "":
                    words.append("")
                else:
                    words.extend(word.split())
    else:
        words = input

    return words

#
#### Interfaces used by TEAL
#
def run(configobj=None):
    """TEAL interface for running this code."""
    ### version 2013 November 25

    localcalcos(input=configobj["input"],
                verbosity=configobj["verbosity"],
                savetmp=configobj["savetmp"],
                outdir=configobj["outdir"],
                find=configobj["find_target"],
                cutoff=configobj["cutoff"],
                shift_file=configobj["shift_file"],
                csum=configobj["csum"],
                raw_csum=configobj["raw_csum"],
                compress=configobj["compress"],
                comp_param=configobj["comp_param"],
                binx=int(configobj["binx"]),
                biny=int(configobj["biny"]),
                stimfile=configobj["stimfile"],
                livetimefile=configobj["livefile"],
                burstfile=configobj["burstfile"],
                print_version=configobj["print_version"],
                print_revision=configobj["print_revision"])

def getHelpAsString(fulldoc=True):
    """Return help info from <module>.help in the script directory"""

    if fulldoc:
        basedoc = __doc__
    else:
        basedoc = ""
    helpString = basedoc + "\n"
    helpString += "Version " + __version__ + "\n"

    helpString += teal.getHelpFileAsString(__taskname__, __file__)

    return helpString

# Set up doc string without the module level docstring included for
# use with Sphinx, since Sphinx will already include module level docstring
# xxx calcos.__doc__ = getHelpAsString(fulldoc=False)

def help():
    print(getHelpAsString())
