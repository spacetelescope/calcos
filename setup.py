from distutils.core import setup, Extension
from distutils import sysconfig
#from distutils.command.install_data import install_data
import sys, os.path
import numpy
# import numpy.numarray.util as nnu
import numpy.numarray as nnu

#if not hasattr(sys, 'version_info') or sys.version_info < (2,3,0,'alpha',0):
#    raise SystemExit, "Python 2.3 or later required to build imagestats."

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
