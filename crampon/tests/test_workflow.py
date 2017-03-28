from __future__ import division

import warnings
import os
import shutil
import unittest
import pickle
from functools import partial

import pytest

import pandas as pd
import geopandas as gpd
import numpy as np
from numpy.testing import assert_allclose

# Locals
import crampon.cfg as cfg
from crampon import workflow
from crampon.utils import get_demo_file, rmsd, write_centerlines_to_shape
from crampon.tests import is_slow, ON_TRAVIS, RUN_WORKFLOW_TESTS, \
    requires_mpltest, RUN_GRAPHIC_TESTS, BASELINE_DIR
from crampon import utils

# General settings
warnings.filterwarnings("once", category=DeprecationWarning)

# do we event want to run the tests?
if not RUN_WORKFLOW_TESTS:
    raise unittest.SkipTest('Skipping all workflow tests.')

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_workflow')
