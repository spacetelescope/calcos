"""Tests for COS/FUV flat."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestFUVAFlat(BaseCOS):
    detector = 'fuv'

    def test_fuva_flat(self):
        """
        COS regression test
        """
        files_to_download = ['la8n01qkq_rawtag_a.fits',
                             'la8n01qkq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la8n01qkq_rawtag_a.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['la8n01qkq']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag_a', 'counts_a', 
                        'flt_a'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=3e-7)
