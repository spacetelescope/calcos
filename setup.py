#!/usr/bin/env python
from glob import glob
from numpy import get_include as np_include
from setuptools import setup, find_packages, Extension
from version import get_git_version

git_version = get_git_version()
with open('lib/calcos/version.py', 'w') as version_data:
    version_data.write("__version__ = '{0}'".format(git_version))


setup(
    name = 'calcos',
    version = git_version,
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
