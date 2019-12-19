import pytest
import os
import numpy as np

from astropy.io import fits

from calcos.cosutil import getTable


def compare_rec_to_record(rec, record):
    for col in rec.names:
        item = rec[col][0]

        if isinstance(item, np.ndarray):
            assert np.array_equal(item, record[col])

        else:
            assert item == record[col]


@pytest.fixture(scope='module')
def tdstab_file(test_data):
    return os.path.join(test_data, 'test_tds.fits')


@pytest.fixture(scope='module')
def tdstab_table(tdstab_file):
    return fits.getdata(tdstab_file)


@pytest.fixture(params=[(1055, 0), (1096, 1)])
def variable_cenwave_tds_filter(tdstab_table, request):
    cenwave, idx = request.param

    return {'opt_elem': 'G130M', 'aperture': 'PSA', "cenwave": cenwave, "segment": 'FUVA'}, tdstab_table[idx]


def test_cenwave_selection_tds(variable_cenwave_tds_filter, tdstab_file):
    """Test TDSTAB row selection for variable cenwave value. getTable arguments match usage from  doFluxCorr where the
    TDSTAB is used.
    """
    test_filter, expected_result = variable_cenwave_tds_filter
    result = getTable(tdstab_file, test_filter, exactly_one=True)

    compare_rec_to_record(result, expected_result)


def test_cenwave_wildcard_selection(tdstab_table, tdstab_file):
    test_filter = {'opt_elem': 'G185M', 'aperture': 'PSA', 'cenwave': 10, 'segment': 'NUVA'}
    expected = tdstab_table[-1]
    result = getTable(tdstab_file, test_filter, exactly_one=True)

    compare_rec_to_record(result, expected)
