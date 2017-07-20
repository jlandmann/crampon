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
#import crampon.cfg as cfg
from oggm.utils import get_demo_file as get_oggm_demo_file
from crampon.tests import is_slow, HAS_NEW_GDAL, requires_py3, RUN_PREPRO_TESTS


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

    def test_distribute_climate_parallel_monthly(self):
        # Init
        cfg.PATHS['working_dir'] = self.testdir
        cfg.PATHS['dem_file'] = get_oggm_demo_file('hef_srtm.tif')
        cfg.PATHS['climate_file'] = get_oggm_demo_file('histalp_merged_hef.nc')
        cfg.PARAMS['border'] = 10

        hef_file = get_oggm_demo_file('Hintereisferner.shp')
        entity = gpd.GeoDataFrame.from_file(hef_file).iloc[0]

        gdirs = []

        gdir = crampon.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)

        climate.process_custom_climate_data(gdir)

        ci = gdir.read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1802)
        self.assertEqual(ci['hydro_yr_1'], 2003)

        with netCDF4.Dataset(get_oggm_demo_file('histalp_merged_hef.nc')) as nc_r:
            ref_h = nc_r.variables['hgt'][1, 1]
            ref_p = nc_r.variables['prcp'][:, 1, 1]
            ref_t = nc_r.variables['temp'][:, 1, 1]

        with netCDF4.Dataset(os.path.join(gdir.dir, 'climate_monthly.nc')) as nc_r:
            self.assertTrue(ref_h == nc_r.ref_hgt)
            np.testing.assert_allclose(ref_t, nc_r.variables['temp'][:])
            np.testing.assert_allclose(ref_p, nc_r.variables['prcp'][:])

    def test_distribute_climate_parallel_daily(self):

        cfg.PATHS['climate_file'] = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                                 'data\\bigdata\\TPH_M_merged_new_settunits+missval.nc')
        gries_file = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                                  'data\\test\\shp\\Gries\\'
                                  'RGI50-11.01876.shp')
        entity = gpd.GeoDataFrame.from_file(gries_file).iloc[0]

        gdirs = []

        gdir = crampon.GlacierDirectory(entity, base_dir=self.testdir)
        gis.define_glacier_region(gdir, entity=entity)
        gdirs.append(gdir)

        climate.process_custom_climate_data(gdir)

        ci = gdir.read_pickle('climate_info')
        self.assertEqual(ci['hydro_yr_0'], 1962)
        self.assertEqual(ci['hydro_yr_1'], 2016)

        # at the moment just fake values
        ref_h = np.array([1000])
        ref_p = np.array([1500])
        ref_t = np.array([500])

        with netCDF4.Dataset(
                os.path.join(gdir.dir, 'climate_daily.nc')) as nc_r:
            self.assertTrue(ref_h == nc_r.ref_hgt)
            np.testing.assert_allclose(ref_t, nc_r.variables['temp'][:])
            np.testing.assert_allclose(ref_p, nc_r.variables['prcp'][:])