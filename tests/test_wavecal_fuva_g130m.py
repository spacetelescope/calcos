"""Tests for COS/FUVA wavecal, G130M."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestFUVAWavecalG130M(BaseCOS):
    detector = 'fuv'

    def test_fuva_wavecal_g130m(self):
        """
        FUV COS regression test
        """
        files_to_download = ['lce823m7q_rawtag_a.fits',
                             'lce823m7q_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'lce823m7q_rawtag_a.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['lce823m7q']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag_a', 'counts_a', 
                        'flt_a', 'x1d'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=3e-7)
