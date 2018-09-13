"""CALCOS regression test helpers."""

import os
import sys

import pytest
from ci_watson.artifactory_helpers import get_bigdata
from ci_watson.hst_helpers import raw_from_asn, ref_from_image, download_crds

from astropy.io import fits
from astropy.io.fits import FITSDiff

__all__ = ['calref_from_image', 'BaseCOS']


def calref_from_image(input_image):
    """
    Return a list of reference filenames, as defined in the primary
    header of the given input image, necessary for calibration; i.e.,
    only those associated with ``*CORR`` set to ``PERFORM`` will be
    considered.
    """
    # NOTE: Add additional mapping as needed.
    # Map mandatory CRDS reference file for instrument/detector combo.
    # This is for file not tied to any particular *CORR or used throughout.
    det_lookup = {
        ('COS', 'FUV'): ['PROFTAB', 'SPWCSTAB'],
        ('COS', 'NUV'): []}

    # NOTE: Add additional mapping as needed.
    # Map *CORR to associated CRDS reference file.
    corr_lookup = {
        'BADTCORR': ['BADTTAB'],
        'TEMPCORR': ['BRFTAB'],
        'GEOCORR': ['GEOFILE'],
        'DGEOCORR': ['DGEOFILE'],
        'YWLKCORR': ['YWLKFILE'],
        'XWLKCORR': ['XWLKFILE'],
        'DEADCORR': ['DEADTAB'],
        'PHACORR': ['PHATAB', 'PHAFILE'],
        'FLATCORR': ['FLATFILE'],
        'WAVECORR': ['LAMPTAB', 'DISPTAB', 'TWOZXTAB', 'XTRACTAB'],
        'BRSTCORR': ['BRSTTAB'],
        'TRCECORR': ['TRACETAB'],
        'ALGNCORR': ['TWOZXTAB'],
        'DQICORR': ['SPOTTAB', 'TRACETAB', 'BPIXTAB', 'GSAGTAB'],
        'X1DCORR': ['WCPTAB', 'TWOZXTAB', 'XTRACTAB'],
        'BACKCORR': ['TWOZXTAB', 'XTRACTAB'],
        'FLUXCORR': ['FLUXTAB', 'TDSTAB', 'PHOTTAB'],
        'WALKCORR': ['WALKTAB']}

    hdr = fits.getheader(input_image, ext=0)
    ref_files = ref_from_image(
        input_image, det_lookup[(hdr['INSTRUME'], hdr['DETECTOR'])])

    for step in corr_lookup:
        # Not all images have the CORR step and it is not always on.
        if (step not in hdr) or (hdr[step].strip().upper() != 'PERFORM'):
            continue

        ref_files += ref_from_image(input_image, corr_lookup[step])

    return list(set(ref_files))  # Remove duplicates


# Base class for actual tests.
# NOTE: Named in a way so pytest will not pick them up here.
# NOTE: bigdata marker requires TEST_BIGDATA environment variable to
#       point to a valid big data directory, whether locally or on Artifactory.
# NOTE: envopt would point tests to "dev" or "stable".
# NOTE: _jail fixture ensures each test runs in a clean tmpdir.
@pytest.mark.bigdata
@pytest.mark.usefixtures('_jail', 'envopt')
class BaseCOS:
    # Timeout in seconds for file downloads.
    timeout = 30

    instrument = 'cos'
    ignore_keywords = ['DATE', 'CAL_VER']

    # To be defined by test class in actual test modules.
    detector = ''

    @pytest.fixture(autouse=True)
    def setup_class(self, envopt):
        """
        Class-level setup that is done at the beginning of the test.

        Parameters
        ----------
        envopt : {'dev', 'stable'}
            This is a ``pytest`` fixture that defines the test
            environment in which input and truth files reside.

        """
        # Since CALCOS still runs in PY2, need to check here because
        # tests can only run in PY3.
        if sys.version_info < (3, ):
            raise SystemError('tests can only run in Python 3')

        self.env = envopt

    def get_input_file(self, filename):
        """
        Copy input file (ASN, RAW, etc) into the working directory.
        If ASN is given, RAW files in the ASN table is also copied.
        The associated CRDS reference files are also copied or
        downloaded, if necessary.

        Data directory layout for CALCOS::

            detector/
                input/
                truth/

        Parameters
        ----------
        filename : str
            Filename of the ASN/RAW/etc to copy over, along with its
            associated files.

        """
        # Copy over main input file.
        dest = get_bigdata('scsb-calcos', self.env, self.detector, 'input',
                           filename)

        # TODO: This logic was based on ONE FUV test. Please revise as needed.
        if filename.endswith('_asn.fits'):
            all_raws = raw_from_asn(filename, '_rawtag_a.fits')
            all_raws += raw_from_asn(filename, '_rawtag_b.fits')
            for raw in all_raws:  # Download RAWs in ASN.
                get_bigdata('scsb-calcos', self.env, self.detector, 'input',
                            raw)
            for raw in raw_from_asn(filename, '_spt.fits'):  # Input SPTs
                get_bigdata('scsb-calcos', self.env, self.detector, 'input',
                            raw)
        else:
            all_raws = [filename]

        first_pass = ('JENKINS_URL' in os.environ and
                      'ssbjenkins' in os.environ['JENKINS_URL'])

        for raw in all_raws:
            ref_files = calref_from_image(raw)

            for ref_file in ref_files:
                # Special reference files that live with inputs.
                if ('$' not in ref_file and
                        os.path.basename(ref_file) == ref_file):
                    get_bigdata('scsb-calcos', self.env, self.detector,
                                'input', ref_file)
                    continue

                # Jenkins cannot see Central Storage on push event,
                # and somehow setting, say, jref to "." does not work anymore.
                # So, we need this hack.
                if '$' in ref_file and first_pass:
                    first_pass = False
                    if not os.path.isdir('/grp/hst/cdbs'):
                        ref_path = os.path.dirname(dest) + os.sep
                        var = ref_file.split('$')[0]
                        os.environ[var] = ref_path  # hacky hack hack

                # Download reference files, if needed only.
                download_crds(ref_file, timeout=self.timeout)

    def compare_outputs(self, outputs, atol=0, rtol=1e-7, raise_error=True,
                        ignore_keywords_overwrite=None):
        """
        Compare CALXXX output with "truth" using ``fitsdiff``.

        Parameters
        ----------
        outputs : list of tuple
            A list of tuples, each containing filename (without path)
            of CALXXX output and truth, in that order. Example::

                [('output1.fits', 'truth1.fits'),
                 ('output2.fits', 'truth2.fits'),
                 ...]

        atol, rtol : float
            Absolute and relative tolerance for data comparison.

        raise_error : bool
            Raise ``AssertionError`` if difference is found.

        ignore_keywords_overwrite : list of str or `None`
            If not `None`, these will overwrite
            ``self.ignore_keywords`` for the calling test.

        Returns
        -------
        report : str
            Report from ``fitsdiff``.
            This is part of error message if ``raise_error=True``.

        """
        all_okay = True
        creature_report = ''

        if ignore_keywords_overwrite is None:
            ignore_keywords = self.ignore_keywords
        else:
            ignore_keywords = ignore_keywords_overwrite

        for actual, desired in outputs:
            desired = get_bigdata('scsb-calcos', self.env, self.detector,
                                  'truth', desired)
            fdiff = FITSDiff(actual, desired, rtol=rtol, atol=atol,
                             ignore_keywords=ignore_keywords)
            creature_report += fdiff.report()

            if not fdiff.identical and all_okay:
                all_okay = False

        if not all_okay and raise_error:
            raise AssertionError(os.linesep + creature_report)

        return creature_report
