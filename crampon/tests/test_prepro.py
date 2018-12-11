from __future__ import absolute_import, division

import warnings
import crampon.utils

import unittest
import os
import glob
import shutil

import shapely.geometry as shpg
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4
import salem
import xarray as xr

# Local imports
import crampon.cfg as cfg
from crampon.core.preprocessing import gis, climate
from oggm.utils import get_demo_file as get_oggm_demo_file
from crampon.tests import HAS_NEW_GDAL, RUN_PREPRO_TESTS


# General settings
warnings.filterwarnings("once", category=DeprecationWarning)

# Do we event want to run the tests?
if not RUN_PREPRO_TESTS:
    raise unittest.SkipTest('Skipping all prepro tests.')

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))


class TestClimate(unittest.TestCase):

    def setUp(self):
        cfg.initialize()

        # test directory
        self.testdir = os.path.join(cfg.PATHS['test_dir'], 'tmp_prepro')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    def test_get_temperature_at_heights(self):
        # test float input
        t_hgt = climate.get_temperature_at_heights(5, 0.0065, 2500, 2600)
        np.testing.assert_equal(t_hgt, 5.65)

        # test array input
        t_hgt = climate.get_temperature_at_heights(np.array([5, 5]),
                                                   np.array([0.0065, 0.005]),
                                                   2500,
                                                   np.array(
                                                       [2400, 2600, 2700]))
        np.testing.assert_equal(t_hgt,
                                np.array([[4.35, 5.65, 6.3], [4.5, 5.5, 6.]]))

        # test xarray input
        t_hgt = climate.get_temperature_at_heights(xr.DataArray([5, 5]),
                                                   xr.DataArray(
                                                       [0.0065, 0.005]), 2500,
                                                   np.array(
                                                       [2400, 2600, 2700]))
        np.testing.assert_equal(t_hgt, np.array([[4.35, 5.65, 6.3],
                                                        [4.5, 5.5, 6.]]))

    def test_get_precipitation_at_heights(self):
        # test float input
        p_hgt = climate.get_precipitation_at_heights(5, 0.0003, 2500, 2600)
        np.testing.assert_equal(p_hgt, 5.15)

        # test array input
        p_hgt = climate.get_precipitation_at_heights(np.array([5, 10]),
                                                     np.array(
                                                         [0.0003, 0.0005]),
                                                     2500,
                                                     np.array([2600, 2700]))
        np.testing.assert_equal(p_hgt, np.array([[5.15, 5.3], [10.5, 11.]]))

        # test xr.DataArray input
        p_hgt = climate.get_precipitation_at_heights(xr.DataArray([5, 10]),
                                                     xr.DataArray(
                                                         [0.0003, 0.0005]),
                                                     2500,
                                                     np.array([2600, 2700]))
        np.testing.assert_equal(p_hgt, np.array([[5.15, 5.3], [10.5, 11.]]))
