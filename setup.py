#!/usr/bin/env python

from setuptools import setup, Extension
from numpy import get_include as numpy_includes
from pathlib import Path
import sysconfig


FREE_THREADED_PYTHON = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


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
MACROS = []
if not FREE_THREADED_PYTHON:
    MACROS.append(("Py_LIMITED_API", 0x030B0000))  # PY_VERSION_HEX for 3.11

SETUPTOOLS_OPTIONS = {}
if not FREE_THREADED_PYTHON:
    SETUPTOOLS_OPTIONS["bdist_wheel"] = {"py_limited_api": "cp311"}

setup(
    ext_modules=[
        Extension(
            PACKAGENAME + ".ccos",
            sources=SOURCES,
            include_dirs=INCLUDES,
            define_macros=MACROS,
            py_limited_api=not FREE_THREADED_PYTHON,
        ),
    ],
    options=SETUPTOOLS_OPTIONS,
)
