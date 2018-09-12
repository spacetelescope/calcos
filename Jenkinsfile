// Obtain files from source control system.
if (utils.scm_checkout()) return

// Define each build configuration, copying and overriding values as necessary.
bc0 = new BuildConfig()
bc0.nodetype = "linux-stable"
bc0.name = "egg"
bc0.build_cmds = ["python setup.py egg_info"]

bc1 = utils.copy(bc0)
bc1.name = "release"
// Would be nice if Jenkins can access /grp/hst/cdbs/xxxx directly.
bc1.env_vars = 'TEST_BIGDATA=https://bytesalad.stsci.edu/artifactory/scsb-calcos']
bc1.build_cmds = ["conda config --add channels http://ssb.stsci.edu/astroconda",
                  "conda install -q -y ci-watson",
                  "python setup.py install"]
// TODO: Enable this when test is added
// bc1.test_cmds = ["pytest tests --basetemp=tests_output --junitxml results.xml --bigdata -v"]
bc1.failedUnstableThresh = 1
bc1.failedFailureThresh = 6

// Iterate over configurations that define the (distibuted) build matrix.
// Spawn a host of the given nodetype for each combination and run in parallel.
utils.run([bc0, bc1])
