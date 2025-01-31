"""Tests for COS/FUV ACQ/PEAKXD."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestFUVACQPEAKXD(BaseCOS):
    detector = 'fuv'

    def test_fuv_acq_peakxd(self):
        """
        FUV COS regression test
        """
        files_to_download = ['la9t01naq_rawacq.fits',
                             'la9t01naq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la9t01naq_rawacq.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # No need to compare results as this test doesn't
        # product any products.  We are just testing that the
        # code runs to completion
