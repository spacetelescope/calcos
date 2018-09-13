"""Tests for COS/FUV timetag."""

#import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
#@pytest.mark.slow
class TestFUVTimetag(BaseCOS):
    detector = 'fuv'

    def test_fuv_timetag_1(self):
        """
        FUV COS regression test #1
        """
        asn_file = 'lckg01070_asn.fits'

        # Prepare input files.
        self.get_input_file(asn_file)

        # Run CALCOS
        calcos.calcos(asn_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['lckg01070', 'lckg01czq', 'lckg01d4q', 'lckg01d9q',
                    'lckg01dcq']
        outputs = []
        for sfx in ('x1dsum', 'x1dsum1', 'x1dsum2', 'x1dsum3', 'x1dsum4'):
            fname = '{}_{}.fits'.format(outroots[0], sfx)
            outputs.append((fname, fname))
        for outroot in outroots[1:]:
            for sfx in ('corrtag_a', 'corrtag_b', 'counts_a', 'counts_b',
                        'flt_a', 'flt_b', 'lampflash', 'x1d'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, fname))
        self.compare_outputs(outputs, rtol=3e-7)
