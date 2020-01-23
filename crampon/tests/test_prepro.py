from __future__ import absolute_import, division

import warnings
import crampon.utils

import unittest
import pytest
import os
import glob
import shutil

import shapely.geometry as shpg
from fiona.errors import DriverError
import numpy as np
import pandas as pd
import geopandas as gpd
import netCDF4
import salem
import xarray as xr

# Local imports
import crampon.cfg as cfg
from crampon.core.preprocessing import gis, climate, radiation
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

    def test_prcp_fac_annual_cycle(self):
        cycle = climate.prcp_fac_annual_cycle(np.arange(1, 367))
        np.testing.assert_equal(np.argmin(cycle), 180)
        np.testing.assert_equal(np.argmax(cycle), 363)
        np.testing.assert_almost_equal(np.max(cycle), 1.08, 6)
        np.testing.assert_almost_equal(np.min(cycle), 0.92, 6)


class TestGlacierMeteo(unittest.TestCase):

    def setUp(self):
        cfg.initialize()

        # test directory
        self.testdir = os.path.join(cfg.PATHS['test_dir'], 'tmp_prepro')
        if not os.path.exists(self.testdir):
            os.makedirs(self.testdir)
        self.clean_dir()
        # Todo: this can only be done once we have a lightweight test dataset
        try:
            gdir = crampon.utils.GlacierDirectory('RGI50-11.A10G05')
            self.gmeteo = climate.GlacierMeteo(self.gdir)
        except DriverError:
            self.gdir = None

    def tearDown(self):
        self.rm_dir()

    def rm_dir(self):
        shutil.rmtree(self.testdir)

    def clean_dir(self):
        shutil.rmtree(self.testdir)
        os.makedirs(self.testdir)

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_days_since_solid_precipitation(self):
        pass

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_get_loc(self):
        pass

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_get_tmean_at_heights(self):
        pass

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_get_tmean_for_melt_at_heights(self):
        pass

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_get_positive_tmax_sum_between(self):
        pass

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_get_mean_annual_temperature_at_heights(self):
        pass

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_get_mean_winter_temperature(self):
        pass

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_get_mean_month_temperature(self):
        pass

    @pytest.mark.skip(reason='No lightweight test dataset yet')
    def test_get_precipitation_liquid_solid(self):
        pass


class TestRadiation(unittest.TestCase):

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

    def test_irradiation_top_of_atmosphere(self):
        # test equator, north pole, south pole, 45 deg north
        # for testing, we set the solar constant to a fixed value
        result = radiation.irradiation_top_of_atmosphere(np.arange(1, 367),
                                                         np.array(
                                                             [0., 90., -90.,
                                                              45.]),
                                                         solar_constant=1367.)

        # equator
        np.testing.assert_equal(np.count_nonzero(result[:, 0]), 366)
        np.testing.assert_almost_equal(np.min(result[:, 0]), 386.1, decimal=1)
        np.testing.assert_almost_equal(np.max(result[:, 0]), 438.9, decimal=1)
        np.testing.assert_almost_equal(np.mean(result[:, 0]), 417.0, decimal=1)
        np.testing.assert_almost_equal(np.argmin(result[:, 0]), 173)
        np.testing.assert_almost_equal(np.argmax(result[:, 0]), 69)

        # north pole
        np.testing.assert_equal(np.count_nonzero(result[:, 1]), 182)
        np.testing.assert_almost_equal(np.min(result[:, 1]), 0., 1)
        np.testing.assert_almost_equal(np.max(result[:, 1]), 526.3, 1)
        np.testing.assert_almost_equal(np.mean(result[:, 1]), 169.8, 1)
        np.testing.assert_almost_equal(np.argmin(result[:, 1]), 0)
        np.testing.assert_almost_equal(np.argmax(result[:, 1]), 171)

        # south pole - 184 because of rounding difference
        np.testing.assert_equal(np.count_nonzero(result[:, 2]), 184)
        np.testing.assert_almost_equal(np.min(result[:, 2]), 0., 1)
        np.testing.assert_almost_equal(np.max(result[:, 2]), 561.7, 1)
        np.testing.assert_almost_equal(np.mean(result[:, 2]), 180.2, 1)
        np.testing.assert_almost_equal(np.argmin(result[:, 2]), 81)
        np.testing.assert_almost_equal(np.argmax(result[:, 2]), 354)

        # 45 deg north
        np.testing.assert_equal(np.count_nonzero(result[:, 3]), 366)
        np.testing.assert_almost_equal(np.max(result[:, 3]), 485.3, 1)
        np.testing.assert_almost_equal(np.min(result[:, 3]), 120.7, 1)
        np.testing.assert_almost_equal(np.mean(result[:, 3]), 304.6, 1)
        np.testing.assert_almost_equal(np.argmax(result[:, 3]), 170)
        np.testing.assert_almost_equal(np.argmin(result[:, 3]), 354)

