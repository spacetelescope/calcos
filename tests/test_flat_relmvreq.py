"""Tests for COS/FUV RelMvRec Flat"""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestRelMvRecFlat(BaseCOS):
    detector = 'fuv'

    def test_relmvrec_flat(self):
        """
        FUV COS regression test
        """
        files_to_download = ['ldc1b1waq_rawtag_a.fits', 'ldc1b1waq_rawtag_b.fits',
                             'ldc1b1waq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'ldc1b1waq_rawtag_a.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['ldc1b1waq']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag_a', 'corrtag_b', 'counts_a', 'counts_b',
                        'flt_a', 'flt_b'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=3e-7)
