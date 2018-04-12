from __future__ import division

import warnings
import unittest
import os
import shutil
import logging

import pandas as pd
import datetime
from numpy.testing import assert_array_equal, assert_almost_equal

from crampon.tests import requires_credentials, requires_vpn, is_download
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
@is_download
class TestCirrusClient(unittest.TestCase):

    def setUp(self):
        self.client = utils.CirrusClient()

    def tearDown(self):
        self.client.close()

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)

    def test_create_connect(self):
        self.client.create_connect(self.client.cr['cirrus']['host'],
                                   self.client.cr['cirrus']['user'],
                                   self.client.cr['cirrus']['password'])
        self.assertIsInstance(self.client, utils.CirrusClient)
        self.assertTrue(self.client.ssh_open)

    def test__open_sftp(self):
        self.client._open_sftp()
        self.assertTrue(self.client.sftp_open)

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

    def test_closest_date(self):

        date_range = pd.date_range('2018-04-01', '2018-04-05')

        self.assertEqual(
            utils.closest_date(pd.Timestamp('2018-04-01 12:00:01'),
                               date_range), pd.Timestamp('2018-04-02'))
        self.assertEqual(
            utils.closest_date(pd.Timestamp('2018-04-01 12:00:00'),
                               date_range), pd.Timestamp('2018-04-01'))
        self.assertEqual(
            utils.closest_date(pd.Timestamp('2018-04-06 12:00:00'),
                               date_range), pd.Timestamp('2018-04-05'))
        self.assertEqual(
            utils.closest_date(pd.Timestamp('2018-03-06 12:00:00'),
                               date_range), pd.Timestamp('2018-04-01'))

    def test_get_begin_last_flexyear(self):

        self.assertEqual(
            utils.get_begin_last_flexyear(datetime.datetime(2018,4,1)),
            datetime.datetime(2017,10,1))
        self.assertEqual(
            utils.get_begin_last_flexyear(datetime.datetime(2018,4,1), start_month=11, start_day=15),
            datetime.datetime(2017,11,15))
        self.assertEqual(
            utils.get_begin_last_flexyear(datetime.datetime(2098, 4, 1),
                                          start_month=11, start_day=15),
            datetime.datetime(2097, 11, 15))
        self.assertEqual(
            utils.get_begin_last_flexyear(datetime.datetime(2017, 11, 15),
                                          start_month=11, start_day=15),
            datetime.datetime(2017, 11, 15))
        self.assertEqual(
            utils.get_begin_last_flexyear(datetime.datetime(2017, 11, 14),
                                          start_month=11, start_day=15),
            datetime.datetime(2016, 11, 15))

    @requires_credentials
    @requires_vpn
    def test_mount_network_drive(self):

        drive = r'\\speedy10.wsl.ch\data_15\_PROJEKTE\Swiss_Glacier'
        msg = utils.mount_network_drive(drive, r'wsl\landmann')
        self.assertEqual(msg, 0)


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

    @is_download
    def test_get_oggm_demo_files(self):
        return OGGMTestDataFiles.test_download_demo_files

    @is_download
    def test_get_crampon_demo_file(self):
        # At the moment not implemented
        pass


class TestMeteoTSAccessor(unittest.TestCase):

    def setUp(self):
        #self.mtsa = utils.read_multiple_netcdfs()
        #self.mtsa.crampon
        pass

    def tearDown(self):

        if os.path.exists(TEST_DIR):
            shutil.rmtree(TEST_DIR)

    def test_update_with_verified(self):
        pass

    def test_update_with_operational(self):
        pass

    def test_ensure_time_continuity(self):
        #self.mtsa.crampon.ensure_time_continuity()
        pass

    def test_cut_by_glacio_years(self):
        #mtsa_cut = self.mtsa.crampon.cut_by_glacio_years()

        #begin = mtsa_cut.time[0]
        #end = mtsa_cut.time[1]

        #self.assertEqual(begin.month, 10)
        #self.assertEqual(begin.day, 1)
        #self.assertEqual(end.month, 9)
        #self.assertEqual(end.day, 30)
        pass

    def test_postprocess_cirrus(self):
        pass


# the retry tests come from:
# https://www.saltycrane.com/blog/2009/11/trying-out-retry-decorator-python/
# The code is BSD licensed.
class RetryableError(Exception):
    pass


class AnotherRetryableError(Exception):
    pass


class UnexpectedError(Exception):
    pass


class RetryTestCase(unittest.TestCase):

    def test_no_retry_required(self):
        self.counter = 0

        @utils.retry(RetryableError, tries=4, delay=0.1)
        def succeeds():
            self.counter += 1
            return 'success'

        r = succeeds()

        self.assertEqual(r, 'success')
        self.assertEqual(self.counter, 1)

    def test_retries_once(self):
        self.counter = 0

        @utils.retry(RetryableError, tries=4, delay=0.1)
        def fails_once():
            self.counter += 1
            if self.counter < 2:
                raise RetryableError('failed')
            else:
                return 'success'

        r = fails_once()
        self.assertEqual(r, 'success')
        self.assertEqual(self.counter, 2)

    def test_limit_is_reached(self):
        self.counter = 0

        @utils.retry(RetryableError, tries=4, delay=0.1)
        def always_fails():
            self.counter += 1
            raise RetryableError('failed')

        with self.assertRaises(RetryableError):
            always_fails()
        self.assertEqual(self.counter, 4)

    def test_multiple_exception_types(self):
        self.counter = 0

        @utils.retry((RetryableError, AnotherRetryableError), tries=4,
                     delay=0.1)
        def raise_multiple_exceptions():
            self.counter += 1
            if self.counter == 1:
                raise RetryableError('a retryable error')
            elif self.counter == 2:
                raise AnotherRetryableError('another retryable error')
            else:
                return 'success'

        r = raise_multiple_exceptions()
        self.assertEqual(r, 'success')
        self.assertEqual(self.counter, 3)

    def test_unexpected_exception_does_not_retry(self):

        @utils.retry(RetryableError, tries=4, delay=0.1)
        def raise_unexpected_error():
            raise UnexpectedError('unexpected error')

        with self.assertRaises(UnexpectedError):
            raise_unexpected_error()

    def test_using_a_logger(self):
        self.counter = 0

        sh = logging.StreamHandler()
        logger = logging.getLogger(__name__)
        logger.addHandler(sh)

        @utils.retry(RetryableError, tries=4, delay=0.1, log_to=logger)
        def fails_once():
            self.counter += 1
            if self.counter < 2:
                raise RetryableError('failed')
            else:
                return 'success'

        fails_once()






