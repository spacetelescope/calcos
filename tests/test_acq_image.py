"""Tests for COS/NUV ACQ/IMAGE."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestNUVCQIMAGE(BaseCOS):
    detector = 'nuv'

    def test_fuv_acq_image(self):
        """
        FUV COS regression test
        """
        files_to_download = ['ldji01ggq_rawacq.fits',
                             'ldji01ggq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'ldji01ggq_rawacq.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['ldji01ggq']
        outputs = []
        for outroot in outroots:
            for sfx in ('counts', 'flt'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=1e-7)

