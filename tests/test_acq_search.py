"""Tests for COS/FUV ACQ/SEARCH."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestFUVACQSEARCH(BaseCOS):
    detector = 'fuv'

    def test_fuv_acq_search(self):
        """
        FUV COS regression test
        """
        files_to_download = ['la9t01n9q_rawacq.fits',
                             'la9t01n9q_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la9t01n9q_rawacq.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # No need to compare results as this test doesn't
        # product any products.  We are just testing that the
        # code runs to completion
