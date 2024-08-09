"""Tests for COS/FUVB wavecal, G160M."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestFUVBWavecalG160M(BaseCOS):
    detector = 'fuv'

    def test_fuvb_wavecal_g160m(self):
        """
        FUV COS regression test
        """
        files_to_download = ['la7803fkq_rawtag_a.fits',
                             'la7803fkq_rawtag_b.fits',
                             'la7803fkq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la7803fkq_rawtag_a.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['la7803fkq']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag_a', 'counts_a',
                        'corrtag_b', 'counts_b', 
                        'flt_a', 'flt_b', 'x1d'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=1e-7)
