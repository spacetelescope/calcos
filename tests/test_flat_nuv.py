"""Tests for COS/NUV flat."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestNUVFlat(BaseCOS):
    detector = 'nuv'

    def test_nuv_flat(self):
        """
        COS regression test
        """
        files_to_download = ['la7u05xoq_rawtag.fits',
                             'la7u05xoq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la7u05xoq_rawtag.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['la7u05xoq']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag', 'counts', 
                        'flt'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=3e-7)
