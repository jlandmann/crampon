from __future__ import division
import unittest
import warnings
import pytest
import os
import geopandas as gpd
import matplotlib.pyplot as plt

# Local imports
import crampon.utils
from crampon.tests import requires_internet, RUN_GRAPHIC_TESTS
from crampon import graphics
from crampon.core.preprocessing import gis
import crampon.cfg as cfg
from crampon.utils import get_oggm_demo_file
from oggm.tests.test_graphics import test_googlemap

# General settings
warnings.filterwarnings("once", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning,
                        message=r'.*guessing baseline image.*')

# do we event want to run the tests?
if not RUN_GRAPHIC_TESTS:
    raise unittest.SkipTest('Skipping all graphic tests.')

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTDIR_BASE = os.path.join(CURRENT_DIR, 'tmp')

