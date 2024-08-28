"""Tests for COS/NUV G230L sci data."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestNUVSciG230L(BaseCOS):
    detector = 'nuv'

    def test_nuv_sci_g230l(self):
        """
        COS regression test
        """
        files_to_download = ['la8p93030_asn.fits',
                             'la8p93a7q_rawtag.fits',
                             'la8p93a7q_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la8p93030_asn.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['la8p93030', 'la8p93a7q']
        outputs = []
        for sfx in ['x1dsum', 'x1dsum3']:
            fname = f'{outroots[0]}_{sfx}.fits'
            comparison_name = 'ref_' + fname
            outputs.append((fname, comparison_name))
        for outroot in outroots[1:]:
            for sfx in ('corrtag', 'counts', 
                        'flt', 'lampflash', 'x1d'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=1e-7)
