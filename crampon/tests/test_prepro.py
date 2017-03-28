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
from crampon.core.preprocessing import gis, geometry
import crampon.cfg as cfg
from crampon import utils
from crampon.utils import get_oggm_demo_file, tuple2int
from crampon.tests import is_slow, HAS_NEW_GDAL, requires_py3, RUN_PREPRO_TESTS

# General settings
warnings.filterwarnings("once", category=DeprecationWarning)

# Do we event want to run the tests?
if not RUN_PREPRO_TESTS:
    raise unittest.SkipTest('Skipping all prepro tests.')

# Globals
current_dir = os.path.dirname(os.path.abspath(__file__))

