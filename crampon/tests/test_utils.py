from __future__ import division

import warnings
import unittest
import os
import shutil

import salem
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from crampon.tests import is_download
from crampon import utils
from crampon import cfg
from oggm.tests.test_utils import TestFuncs, TestInitialize

# General settings
warnings.filterwarnings("once", category=DeprecationWarning)

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_download')
if not os.path.exists(TEST_DIR):
    os.makedirs(TEST_DIR)

# I have to change/set up
#class TestInitialize(unittest.TestCase):
#TestDataFiles