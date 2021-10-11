"""Tests for COS/FUVA wavecal, G140L."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestFUVAWavecalG140L(BaseCOS):
    detector = 'fuv'

    def test_fuva_wavecal_g140l(self):
        """
        FUV COS regression test
        """
        files_to_download = ['la8n01qqq_rawtag_a.fits',
                             'la8n01qqq_rawtag_b.fits',
                             'la8n01qqq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la8n01qqq_rawtag_a.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['la8n01qqq']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag_a', 'corrtag_b',
                        'counts_a', 'counts_b',
                        'flt_a', 'flt_b', 'x1d'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=3e-7)
