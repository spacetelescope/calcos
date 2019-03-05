# CALCOS

[![Jenkins CI](https://ssbjenkins.stsci.edu/job/STScI/job/calcos/job/master/badge/icon)](https://ssbjenkins.stsci.edu/job/STScI/job/calcos/job/master/)

Calibration software for HST/COS.

Nightly regression test results are available only from within the STScI
network at this time.
https://boyle.stsci.edu:8081/job/RT/job/calcos/test_results_analyzer/

## JupyterHub Access

*Note:* This is currently still in research-and-development stage and is subject to change.

To run a pre-installed pipeline in JupyterHub:

* Click on https://dev.science.stsci.edu/hub/spawn?image=793754315137.dkr.ecr.us-east-1.amazonaws.com/datb-tc-pipeline-nb:hstdp-snapshot and sign in.
* Click "Terminal" to:
    * Run `pip freeze` to see what is installed.
    * You can download the necessary data files using HTTP/HTTPS protocol.
    * Set up your `lref`, as desired.
    * Grab your notebooks (e.g., using `git clone`) and install any optional software (e.g., using `pip install`).
* Launch your notebook to run the CALCOS pipeline.

Latest release of any packages is not guaranteed in this environment. Amazon Web Services charges may apply.