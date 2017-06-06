import logging
from oggm.tests import *
import os
import osgeo.gdal
from distutils.version import LooseVersion
import matplotlib
from oggm import cfg as oggmcfg
import sys
import socket
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# test dirs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDIR_BASE = os.path.join(CURRENT_DIR, 'tmp')

#### SOME SETUP FOR OGGM
# Some logic to see which environment we are running on

# GDAL version changes the way interpolation is made (sigh...)
HAS_NEW_GDAL = False
if osgeo.gdal.__version__ >= '1.11':
    HAS_NEW_GDAL = True

# Matplotlib version changes plots, too
HAS_MPL_FOR_TESTS = False
if LooseVersion(matplotlib.__version__) >= LooseVersion('2'):
    HAS_MPL_FOR_TESTS = True
    BASELINE_DIR = os.path.join(oggmcfg.CACHE_DIR, 'oggm-sample-data-master',
                                'baseline_images', '2.0.x')


# Some control on which tests to run (useful to avoid too long tests)
# defaults everywhere else than travis
ON_AWS = False
ON_TRAVIS = False
RUN_SLOW_TESTS = False
RUN_DOWNLOAD_TESTS = False
RUN_PREPRO_TESTS = True
RUN_MODEL_TESTS = True
RUN_WORKFLOW_TESTS = True
RUN_GRAPHIC_TESTS = True
RUN_PERFORMANCE_TESTS = False
if os.environ.get('TRAVIS') is not None:
    # specific to travis to reduce global test time
    ON_TRAVIS = True
    RUN_DOWNLOAD_TESTS = False
    matplotlib.use('Agg')

    if sys.version_info < (3, 5):
        # Minimal tests
        RUN_SLOW_TESTS = False
        RUN_PREPRO_TESTS = True
        RUN_MODEL_TESTS = True
        RUN_WORKFLOW_TESTS = True
        RUN_GRAPHIC_TESTS = True
    else:
        # distribute the tests
        RUN_SLOW_TESTS = True
        env = os.environ.get('OGGM_ENV')
        if env == 'prepro':
            RUN_PREPRO_TESTS = True
            RUN_MODEL_TESTS = False
            RUN_WORKFLOW_TESTS = False
            RUN_GRAPHIC_TESTS = False
        if env == 'models':
            RUN_PREPRO_TESTS = False
            RUN_MODEL_TESTS = True
            RUN_WORKFLOW_TESTS = False
            RUN_GRAPHIC_TESTS = False
        if env == 'workflow':
            RUN_PREPRO_TESTS = False
            RUN_MODEL_TESTS = False
            RUN_WORKFLOW_TESTS = True
            RUN_GRAPHIC_TESTS = False
        if env == 'graphics':
            RUN_PREPRO_TESTS = False
            RUN_MODEL_TESTS = False
            RUN_WORKFLOW_TESTS = False
            RUN_GRAPHIC_TESTS = True
elif 'ip-' in socket.gethostname():
    # we are on AWS (hacky way)
    ON_AWS = True
    RUN_SLOW_TESTS = True
    matplotlib.use('Agg')

# give user some control
if os.environ.get('OGGM_SLOW_TESTS') is not None:
    RUN_SLOW_TESTS = True
if os.environ.get('OGGM_DOWNLOAD_TESTS') is not None:
    RUN_DOWNLOAD_TESTS = True

# quick n dirty method to see if internet is on
try:
    _ = urlopen('http://www.google.com', timeout=1)
    HAS_INTERNET = True
except URLError:
    HAS_INTERNET = False
#########################
