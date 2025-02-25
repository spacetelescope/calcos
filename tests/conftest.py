"""Custom ``pytest`` configurations."""
import pytest
import os

#from astropy.tests.helper import enable_deprecations_as_exceptions

# Turn deprecation warnings into exceptions.
#enable_deprecations_as_exceptions()

# Require these pytest plugins to run.
pytest_plugins = ["pytest_ciwatson"]


# For easy inspection on what dependencies were used in test.
def pytest_report_header(config):
    import sys
    import warnings
    import importlib

    s = "\nFull Python Version: \n{0}\n\n".format(sys.version)

    for module_name in ('numpy', 'astropy', 'scipy', 'matplotlib',
                        'stsci.tools'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                module = importlib.import_module(module_name)
        except ImportError:
            s += "{0}: not available\n".format(module_name)
        else:
            try:
                version = module.__version__
            except AttributeError:
                version = 'unknown (no __version__ attribute)'
            s += "{0}: {1}\n".format(module_name, version)

    return s


@pytest.fixture(scope='session')
def test_data():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
