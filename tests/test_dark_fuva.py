"""Tests for COS/FUV dark."""

#import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
#@pytest.mark.slow
class TestFUVADark(BaseCOS):
    detector = 'fuv'

    def test_fuva_dark(self):
        """
        FUV COS regression test #2
        """
        files_to_download = ['la7803fiq_rawtag_a.fits', 'la7803fiq_rawtag_b.fits',
                             'la7803fiq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la7803fiq_rawtag_a.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['la7803fiq']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag_a', 'corrtag_b', 'counts_a', 'counts_b',
                        'flt_a', 'flt_b'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, fname))
        self.compare_outputs(outputs, rtol=3e-7)
