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