"""Tests for COS/FUV timetag."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestFUVTimetag(BaseCOS):
    detector = 'fuv'

    def test_fuv_timetag_1(self):
        """
        FUV COS regression test #1
        """
        files_to_download = ['lckg01070_asn.fits', 'lckg01czq_spt.fits',
                             'lckg01d4q_spt.fits', 'lckg01d9q_spt.fits',
                             'lckg01dcq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        # Run CALCOS
        input_file = 'lckg01070_asn.fits'
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['lckg01070', 'lckg01czq', 'lckg01d4q', 'lckg01d9q',
                    'lckg01dcq']
        outputs = []
        for sfx in ('x1dsum', 'x1dsum1', 'x1dsum2', 'x1dsum3', 'x1dsum4'):
            fname = '{}_{}.fits'.format(outroots[0], sfx)
            comparison_name = 'ref_' + fname
            outputs.append((fname, comparison_name))
        for outroot in outroots[1:]:
            for sfx in ('corrtag_a', 'corrtag_b', 'counts_a', 'counts_b',
                        'flt_a', 'flt_b', 'lampflash', 'x1d'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                comparison_name = 'ref_' + fname
                outputs.append((fname, comparison_name))
        self.compare_outputs(outputs, rtol=3e-7)
