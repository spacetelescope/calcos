"""Tests for COS/FUVB wavecal, G160M, relmvreq"""

import pytest

import calcos
from helpers import BaseCOS


# TODO: Mark this as slow when there are faster tests added for CI tests
#       so that this only runs in nightly tests.
@pytest.mark.slow
class TestFUVBWavecalG160MRelMvReq(BaseCOS):
    detector = 'fuv'

    def test_fuvb_wavecal_g160m_relmvreq(self):
        """
        FUV COS regression test
        """
        files_to_download = ['ldd9a3h6q_rawtag_a.fits',
                             'ldd9a3h6q_rawtag_b.fits',
                             'ldd9a3h6q_spt.fits']

        # Prepare input files.
        self.get_input_files(files_to_download)

        input_file = 'ldd9a3h6q_rawtag_a.fits'
        # Run CALCOS
        calcos.calcos(input_file)

        # Compare results.
        # The first outroot is the output from whole ASN,
        # the rest are individual members.
        outroots = ['ldd9a3h6q']
        outputs = []
        for outroot in outroots:
            for sfx in ('corrtag_a', 'counts_a',
                        'corrtag_b', 'counts_b', 
                        'flt_a', 'flt_b', 'x1d'):
                fname = '{}_{}.fits'.format(outroot, sfx)
                outputs.append((fname, 'ref_' + fname))
        self.compare_outputs(outputs, rtol=1e-7)
