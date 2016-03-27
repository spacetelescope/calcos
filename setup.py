#!/usr/bin/env python
import sys
sys.path.insert(1, 'recon')

import recon.release
from glob import glob
from numpy import get_include as np_include
from setuptools import setup, find_packages, Extension


version = recon.release.get_info()
recon.release.write_template(version, 'lib/calcos')

setup(
    name = 'calcos',
    version = version.pep386,
    author = 'Phil Hodge and Rober Jedrzejewski',
    author_email = 'help@stsci.edu',
    description = 'Calibration software for COS (Cosmic Origins Spectrograph)',
    url = 'https://github.com/spacetelescope/calcos',
    classifiers = [
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires = [
        'astropy',
        'nose',
        'numpy',
        'scipy',
        'sphinx',
    ],

    package_dir = {
        '':'lib'
    },
    packages = find_packages('lib'),
    package_data = {
        'calcos': [
            'pars/*',
            '*.help',
        ]
    },
    entry_points = {
        'console_scripts': [
            'calcos=calcos.calcos:main',
        ],
    },
    ext_modules=[
        Extension('calcos.ccos',
            glob('src/*.c'),
            include_dirs=[np_include()],
            define_macros=[('NUMPY','1')]),
    ],
)
