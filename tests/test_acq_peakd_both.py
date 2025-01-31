"""Tests for COS/BOTH ACQ/PEAKD."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestBOTHACQPEAKD(BaseCOS):
    detector = 'fuv'

    def test_both_acq_peakd(self):
        """
        FUV COS regression test
        """
        files_to_download = ['ld7y02rrq_rawacq.fits',
                             'ld7y02rrq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'ld7y02rrq_rawacq.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # No need to compare results as this test doesn't
        # produce any products.  We are just testing that the
        # code runs to completion
