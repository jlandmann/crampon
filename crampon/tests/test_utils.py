from __future__ import division

import warnings
import unittest
import os
import shutil
import time

import salem
import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_almost_equal

from crampon.tests import requires_credentials, requires_vpn
from crampon import utils
from crampon import cfg
from oggm.tests.test_utils import TestDataFiles as OGGMTestDataFiles
from oggm.tests.funcs import get_test_dir, patch_url_retrieve_github
_url_retrieve = None

# General settings
warnings.filterwarnings("once", category=DeprecationWarning)

# Globals
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(CURRENT_DIR, 'tmp_download')
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)



@requires_credentials
@requires_vpn
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


class CramponTestDataFiles(unittest.TestCase):

    def setUp(self):
        self.dldir = os.path.join(get_test_dir(), 'tmp_download')
        utils.mkdir(self.dldir)
        cfg.initialize()
        cfg.PATHS['dl_cache_dir'] = os.path.join(self.dldir, 'dl_cache')
        cfg.PATHS['working_dir'] = os.path.join(self.dldir, 'wd')
        cfg.PATHS['tmp_dir'] = os.path.join(self.dldir, 'extract')
        self.reset_dir()
        utils._urlretrieve = _url_retrieve

    def tearDown(self):
        if os.path.exists(self.dldir):
            shutil.rmtree(self.dldir)
        utils._urlretrieve = patch_url_retrieve_github

    def reset_dir(self):
        if os.path.exists(self.dldir):
            shutil.rmtree(self.dldir)
        utils.mkdir(cfg.PATHS['dl_cache_dir'])
        utils.mkdir(cfg.PATHS['working_dir'])
        utils.mkdir(cfg.PATHS['tmp_dir'])

    def test_get_oggm_demo_files(self):
        return OGGMTestDataFiles.test_download_demo_files

    def test_get_crampon_demo_file(self):
        # At the moment not implemented
        pass


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









