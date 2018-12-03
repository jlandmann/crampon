from __future__ import division

import warnings
import logging
import unittest
import os
import pytest
from oggm.tests.funcs import get_test_dir, init_hef
from crampon import utils
import shutil
import numpy as np
from crampon.core.models import massbalance
from crampon.core.models import flowline
from crampon import cfg

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


@pytest.mark.internet
class TestMassBalance(unittest.TestCase):

    def setUp(self):
        gdir = init_hef(border=DOM_BORDER)
        self.testdir = os.path.join(get_test_dir(), type(self).__name__)
        utils.mkdir(self.testdir, reset=True)
        self.gdir = gdir.copy_to_basedir(self.testdir, setup='all')

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        if os.path.exists(self.testdir):
            shutil.rmtree(self.testdir)


class TestMiscModels(unittest.TestCase):
    def setUp(self):
        self.test_rho = np.array([100., 200., 300., 400., 500., 600., 700.,
                                  800., 900.])
        self.test_temp = np.arange(253., 279, 5.)

        cfg.initialize()

    def tearDown(self):
        pass

    def test_get_rho_fresh_snow_anderson(self):

        # for standard setting
        desired = np.array([50., 50., 68.15772897, 102.55369631, 147.28336893,
                            200.34524185])
        result = massbalance.get_rho_fresh_snow_anderson(self.test_temp)
        np.testing.assert_almost_equal(desired, result)

        # for higher min_rho
        desired = np.array([100., 100., 118.15772897, 152.55369631,
                            197.28336893, 250.34524185])
        result = massbalance.get_rho_fresh_snow_anderson(self.test_temp,
                                                         rho_min=100.)
        np.testing.assert_almost_equal(desired, result)

    def test_get_thermal_conductivity_yen(self):
        # Cuffey and Paterson 2010 (p. 401) require 2.1 W m-1 K-1 for ice at
        # 0 deg C
        desired = np.array([0.0292653, 0.10771827, 0.23085588, 0.39648408,
                            0.60313785, 0.8497229, 1.13536967, 1.45935903,
                            1.82107944])
        result = massbalance.get_thermal_conductivity_yen(self.test_rho)
        np.testing.assert_almost_equal(desired, result)

    def test_get_snow_thermal_diffusivity(self):
        # Cuffey and Paterson 2010 (p. 401) require 1.09e-6 m2 s-1 for ice at
        # 0 deg C
        desired = np.array([1.39499776e-07, 2.56731943e-07, 3.66809197e-07,
                            4.72483127e-07, 5.74999035e-07, 6.75066586e-07,
                            7.73142927e-07, 8.69546278e-07, 9.81465575e-07])
        result = massbalance.get_snow_thermal_diffusivity(self.test_rho,
                                                          273.15)
        np.testing.assert_almost_equal(desired, result)
