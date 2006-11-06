from distutils.core import setup, Extension
from distutils import sysconfig
from distutils.command.install_data import install_data
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
                ])
        #remove --local from both sys.argv and args
        args.remove (a)
        sys.argv.remove (a)

class smart_install_data (install_data):
    def run (self):
        #need to change self.install_dir to the library dir
        install_cmd = self.get_finalized_command ('install')
        self.install_dir = getattr (install_cmd, 'install_lib')
        return install_data.run (self)

def getExtensions_numpy (args):
    ext = [Extension ("ccos", ["ccos.c"],
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
               packages = [""],
               package_dir = {"":""},
               cmdclass = {'install_data': smart_install_data},
               ext_modules = ext)
    return r


if __name__ == "__main__":
    ext = getExtensions_numpy (args)
    dosetup (ext)
