from __future__ import division

import warnings
import logging
import unittest
import os
import numpy as np
from crampon.core.models import massbalance
from crampon.core.models import flowline

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

class TestMiscModels(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_get_rho_fresh_snow_anderson(self):

        # It's actually stupid that with increasing rho_min also the
        # rho values for higher temperatures change...but that's the equation.

        # test for plausible temp range
        temp_range = np.arange(253., 279, 5.)

        # for standard setting
        desired = np.array([50., 50., 68.15772897, 102.55369631, 147.28336893,
                            200.34524185])
        result = massbalance.get_rho_fresh_snow_anderson(temp_range)
        np.testing.assert_almost_equal(desired, result)

        # for higher min_rho
        desired = np.array([100., 100., 118.15772897, 152.55369631,
                            197.28336893, 250.34524185])
        result = massbalance.get_rho_fresh_snow_anderson(temp_range,
                                                         rho_min=100.)
        np.testing.assert_almost_equal(desired, result)