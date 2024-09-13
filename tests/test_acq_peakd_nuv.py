"""Tests for COS/NUV ACQ/PEAKD."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestNUVACQPEAKD(BaseCOS):
    detector = 'nuv'

    def test_nuv_acq_peakd(self):
        """
        FUV COS regression test
        """
        files_to_download = ['la8q99l7q_rawacq.fits',
                             'la8q99l7q_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la8q99l7q_rawacq.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # No need to compare results as this test doesn't
        # product any products.  We are just testing that the
        # code runs to completion
