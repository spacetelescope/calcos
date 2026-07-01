#!/usr/bin/env python

from setuptools import setup, Extension
from numpy import get_include as numpy_includes
from pathlib import Path


def c_sources(parent: str) -> list[str]:
    return [str(filename) for filename in Path(parent).glob("*.c")]


def c_includes(parent: str, depth: int = 1):
    return [
        parent,
        *(
            str(filename)
            for filename in Path(parent).iterdir()
            if filename.is_dir() and len(filename.parts) - 1 <= depth
        ),
    ]


PACKAGENAME = "calcos"
SOURCES = c_sources("src")
INCLUDES = c_includes("src") + [numpy_includes()]
MACROS = [
    ("Py_LIMITED_API", 0x030B0000),  # PY_VERSION_HEX for 3.11
]


setup(
    ext_modules=[
        Extension(
            PACKAGENAME + ".ccos",
            sources=SOURCES,
            include_dirs=INCLUDES,
            define_macros=MACROS,
            py_limited_api=True,
        ),
    ],
    options={'bdist_wheel': {'py_limited_api': 'cp311'}},
)
