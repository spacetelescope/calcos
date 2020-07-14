#!/usr/bin/env python

import os
from fnmatch import fnmatch
from setuptools import setup, find_packages, Extension
from numpy import get_include as numpy_includes


def c_sources(parent):
    sources = []
    for root, _, files in os.walk(parent):
        for f in files:
            fn = os.path.join(root, f)
            if fnmatch(fn, '*.c'):
                sources.append(fn)
    return sources


def c_includes(parent, depth=1):
    includes = [parent]
    for root, dirs, _ in os.walk(parent):
        for d in dirs:
            dn = os.path.join(root, d)
            if len(dn.split(os.sep)) - 1 > depth:
                continue
            includes.append(dn)
    return includes


PACKAGENAME = 'calcos'
SOURCES = c_sources('src')
INCLUDES = c_includes('src') + [numpy_includes()]


setup(
    name=PACKAGENAME,
    use_scm_version={'write_to': 'calcos/version.py'},
    setup_requires=['setuptools_scm'],
    install_requires=[
        'astropy>=1.1.1',
        'numpy>=1.10.1',
        'scipy>=0.14',
        'stsci.tools>=3.5',
    ],
    extras_require={
        'docs': [
            'sphinx',
        ],
        'test': [
            'ci_watson',
            'pytest',
            'pytest-cov',
            'codecov',
        ],
    },
    packages=find_packages(),
    package_data={
        PACKAGENAME: [
            'pars/*',
            '*.help',
        ],
    },
    ext_modules=[
        Extension(
            PACKAGENAME + '.ccos',
            sources=SOURCES,
            include_dirs=INCLUDES,
        ),
    ],
    entry_points={
        'console_scripts': {
            'calcos = calcos:main',
        },
    },
    author='Phil Hodge, Robert Jedrzejewski',
    author_email='help@stsci.edu',
    description='Calibration software for COS (Cosmic Origins Spectrograph)',
    url='https://github.com/spacetelescope/calcos',
    license='BSD',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
