"""Tests for COS/NUV G185M sci data."""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestNUVSciG185M(BaseCOS):
    detector = 'nuv'

    def test_nuv_sci_g185m(self):
        """
        COS regression test
        """
        files_to_download = ['la8q99050_asn.fits',
                             'la8q99jbq_rawtag.fits',
                             'la8q99jbq_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'la8q99050_asn.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['la8q99050', 'la8q99jbq']
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
