"""Tests for COS/NUV G230L wavecal."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestWavecalNUVG230L(BaseCOS):
    detector = 'nuv'

    def test_wavecal_nuv_g230l(self):
        """
        COS regression test
        """
        files_to_download = ['la7v01e4q_rawtag.fits',
                             'la7v01e4q_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la7v01e4q_rawtag.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['la7v01e4q']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag', 'counts', 
                        'flt', 'x1d'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=1e-7)
