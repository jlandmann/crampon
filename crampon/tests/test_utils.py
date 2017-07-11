from __future__ import division

import warnings
import unittest
import os
import shutil

import salem
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal

from crampon.tests import requires_credentials
from crampon import utils
from crampon import cfg

# General settings
warnings.filterwarnings("once", category=DeprecationWarning)

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_download')
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)



@requires_credentials
class TestCirrusClient(unittest.TestCase):

    def setUp(self):
        self.client = utils.CirrusClient()

    def tearDown(self):
        self.client.close()

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)

    def test_create_connect(self):
        assert isinstance(self.client, utils.CirrusClient)

    def test_list_content(self):
        content =  self.client.list_content('/data/*.pdf')

        assert content == [b'/data/CIRRUS USER GUIDE.pdf']

    def test_get_files(self):
        self.client.get_files('/data', ['./CIRRUS USER GUIDE.pdf'], TEST_DIR)

        assert os.path.exists(os.path.join(TEST_DIR, 'CIRRUS USER GUIDE.pdf'))

    def test_sync_files(self):

        miss, delete = self.client.sync_files('/data', TEST_DIR,
                                              globpattern='./griddata/Product_Description/*ENG.pdf')

        assert len(miss) == 1
        assert len(delete) == 0
