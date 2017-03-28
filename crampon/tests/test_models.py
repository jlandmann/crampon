from __future__ import division

import warnings
import logging
import unittest
import os

# Local imports
from crampon.tests import RUN_MODEL_TESTS

# General settings
warnings.filterwarnings("once", category=DeprecationWarning)
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)

# Do we event want to run the tests?
if not RUN_MODEL_TESTS:
    raise unittest.SkipTest('Skipping all model tests.')

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))

# test directory
testdir = os.path.join(current_dir, 'tmp')

do_plot = False

DOM_BORDER = 80
