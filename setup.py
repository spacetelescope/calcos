from distutils.core import setup, Extension
from distutils import sysconfig

import sys, os.path

try:
    import numpy
    import numpy.numarray as nnu
except ImportError:
    print "NUMPY was not found. It may not be installed or it may not be on your PYTHONPATH"

pythoninc = sysconfig.get_python_inc()

numpyinc = numpy.get_include()
numpynumarrayinc = nnu.get_numarray_include_dirs()[0]

args = sys.argv[:]
for a in args:
    if a.startswith ('--local='):
        dir = os.path.abspath (a.split ("=")[1])
        sys.argv.extend ([
                "--install-lib="+dir,
                "--install-scripts=%s" % os.path.join(dir,"calcos"),
                ])
        #remove --local from both sys.argv and args
        args.remove (a)
        sys.argv.remove (a)


def getExtensions_numpy (args):
    ext = [Extension ("calcos.ccos", ["src/ccos.c"],
           define_macros = [('NUMPY', '1')],
           include_dirs = [pythoninc, numpyinc, numpynumarrayinc])]
    return ext


def dosetup (ext):
    r = setup (name = "ccos",
               version = "1.0",
               description = "C extension module for calcos",
               author = "Phil Hodge",
               author_email = "help@stsci.edu",
               platforms = ["Linux", "Solaris", "Mac OS X", "Windows"],
               packages = ["calcos"],
               package_dir = {"calcos":"lib"},
               scripts = ['lib/calcos.py'],
               ext_modules = ext)
    return r


if __name__ == "__main__":
    ext = getExtensions_numpy (args)
    dosetup (ext)
