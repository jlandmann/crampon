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
        self.assertIsInstance(self.client, utils.CirrusClient)

    def test_list_content(self):
        content =  self.client.list_content('/data/*.pdf')

        self.assertEqual(content, [b'/data/CIRRUS USER GUIDE.pdf'])

    def test_get_files(self):
        self.client.get_files('/data', ['./CIRRUS USER GUIDE.pdf'], TEST_DIR)

        assert os.path.exists(os.path.join(TEST_DIR,
                                           'data/CIRRUS USER GUIDE.pdf'))

    def test_sync_files(self):

        miss, delete = self.client.sync_files('/data/griddata', TEST_DIR,
                                              globpattern='*Product_Descriptio'
                                                          'n/*ENG.pdf')

        self.assertEqual(len(miss), 1)
        self.assertEqual(len(delete), 0)


class TestMiscFuncs(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)

    def test_leap_years(self):

        a = utils.leap_year(1600, calendar='julian')
        self.assertTrue(a)

        a = utils.leap_year(1600, calendar='standard')
        self.assertTrue(a)

        a = utils.leap_year(1300, calendar='gregorian')
        self.assertFalse(a)


class TestMeteoTSAccessor(unittest.TestCase):

    def setUp(self):
        self.mtsa = utils.read_multiple_netcdfs()
        self.mtsa.crampon

    def tearDown(self):

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)

    def test_ensure_time_continuity(self):
        self.mtsa.crampon.ensure_time_continuity()

    def test_cut_by_glacio_years(self):
        mtsa_cut = self.mtsa.crampon.cut_by_glacio_years()

        begin = mtsa_cut.time[0]
        end = mtsa_cut.time[1]

        self.assertEqual(begin.month, 10)
        self.assertEqual(begin.day, 1)
        self.assertEqual(end.month, 9)
        self.assertEqual(end.day, 30)









