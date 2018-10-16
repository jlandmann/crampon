"""
A collection of some useful miscellaneous functions.
"""

from __future__ import absolute_import, division

from joblib import Memory
import posixpath
import salem
import os
import pandas as pd
import numpy as np
import logging
import paramiko as pm
import xarray as xr
import rasterio
import subprocess
from rasterio.merge import merge as merge_tool
from rasterio.warp import transform as transform_tool
from rasterio.mask import mask as riomask
import geopandas as gpd
import shapely
import datetime as dt
from configobj import ConfigObj, ConfigObjError
from itertools import product
import dask
import sys
import glob
import fnmatch
import netCDF4
from scipy import stats
from salem import lazy_property, read_shapefile
from functools import partial, wraps
from oggm.utils import *
# Locals
import crampon.cfg as cfg
from pathlib import Path


# I should introduce/alter:
"""get_crampon_demo_file, in general: get_files"""

# Module logger
log = logging.getLogger(__name__)
# Stop paramiko from logging successes to the console


# Joblib
MEMORY = Memory(cachedir=cfg.CACHE_DIR, verbose=0)
SAMPLE_DATA_GH_REPO = 'crampon-sample-data'


def get_oggm_demo_file(fname):
    """ Wraps the oggm.utils.get_demo_file function"""
    get_demo_file(fname)  # Calls the func imported from oggm.utils


def get_crampon_demo_file():
    """This should be done once some test data are allowed to be moved to an 
    external repo"""
    raise NotImplementedError


def retry(exceptions, tries=100, delay=60, backoff=1, log_to=None):
    """
    Retry decorator calling the decorated function with an exponential backoff.

    Amended from Python wiki [1]_ and calazan.com [2]_.

    Parameters
    ----------
    exceptions: str or tuple
        The exception to check. May be a tuple of exceptions to check. If just
        `Exception` is provided, it will retry after any Exception.
    tries: int
        Number of times to try (not retry) before giving up. Default: 100.
    delay: int or float
        Initial delay between retries in seconds. Default: 60.
    backoff: int or float
        Backoff multiplier (e.g. value of 2 will double the delay
        each retry). Default: 1 (no increase).
    log_to: logging.logger
        Logger to use. If None, print.

    References
    -------
    .. [1] https://wiki.python.org/moin/PythonDecoratorLibrary#CA-901f7a51642f4dbe152097ab6cc66fef32bc555f_5
    .. [2] https://www.calazan.com/retry-decorator-for-python-3/
    """
    def deco_retry(f):

        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exceptions as e:
                    msg = '{}, Retrying in {} seconds...'.format(e, mdelay)
                    if log_to:
                        log_to.warning(msg)
                    else:
                        print(msg)
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)

        return f_retry  # true decorator

    return deco_retry


def weighted_quantiles(values, quantiles, sample_weight=None,
                          values_sorted=False, old_style=False):
        """
        A function to approximate quantiles of data with corresponding weights.

        Very close to numpy.percentile, but supports weights. Quantiles should
        be in the range [0, 1].
        Slightly modified and documentation extended from [1]_.

        Parameters
        ----------
        values: numpy.array
            Data array with the values to be weighted.
        quantiles: array-like
            The quantiles to be calculated. Have to be in range [0, 1].
        sample_weight: array-like, same shape as `values`
            Weights for the individual data. If not given, they will be the
            same (one) for each value.
        values_sorted: bool
            If True, will avoid sorting of initial array. Default: False.
        old_style: bool
            If True, will correct output to be consistent with
            numpy.percentile. Default: False

        Returns
        -------
        numpy.array with computed quantiles.

        References
        ----------
        .. [1] https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
        """

        values = np.array(values)
        quantiles = np.array(quantiles)
        if sample_weight is None:
            sample_weight = np.ones(len(values))
        sample_weight = np.array(sample_weight)
        assert np.all(quantiles >= 0) and np.all(
            quantiles <= 1), 'quantiles should be in [0, 1]'

        if not values_sorted:
            sorter = np.argsort(values)
            values = values[sorter]
            sample_weight = sample_weight[sorter]

        weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
        if old_style:
            # To be convenient with numpy.percentile
            weighted_quantiles -= weighted_quantiles[0]
            weighted_quantiles /= weighted_quantiles[-1]
        else:
            weighted_quantiles /= np.sum(sample_weight)
        return np.interp(quantiles, weighted_quantiles, values)


def leap_year(year, calendar='standard'):
    """
    Determine if year is a leap year.
    Amended from http://xarray.pydata.org/en/stable/examples/monthly-means.html

    Parameters
    ----------
    year: int
       The leap year candidate
    calendar: str
       The calendar format to be used. Possible: 'standard', 'gregorian',
        'proleptic_gregorian', 'julian'. Default: 'standard'

    Returns
    -------
    True if year is leap year, else False
    """

    leap = False
    calendar_opts = ['standard', 'gregorian', 'proleptic_gregorian', 'julian']

    if (calendar in calendar_opts) and (year % 4 == 0):
        leap = True
        if ((calendar == 'proleptic_gregorian') and
           (year % 100 == 0) and (year % 400 != 0)):
            leap = False
        elif ((calendar in ['standard', 'gregorian']) and
              (year % 100 == 0) and (year % 400 != 0) and
              (year < 1583)):
            leap = False
    return leap


def closest_date(date, candidates):
    """
    Get closest date to the given one from a candidate list.

    This function is the one suggested on stackoverflow [1]_.

    Parameters
    ----------
    date: type allowing comparison, subtraction and abs, e.g. dt.datetime
        The date to which the closest partner shall be found.
    candidates: list of same input types as for date
        A list of candidates.

    Returns
    -------
    closest: Same type as input date
        The found closest date.

    Examples
    --------
    Find the closest date to 2018-04-01 in a list.

    >>> import datetime as dt
    >>> closest_date(dt.datetime(2018,4,1), [dt.datetime(2018,4,3),
    >>> dt.datetime(2018,3,21)])
    dt.datetime(2018, 4, 3, 0, 0)

    References
    ----------
    .. [1] https://stackoverflow.com/questions/32237862/find-the-closest-date-to-a-given-date
    """
    return min(candidates, key=lambda x: abs(x - date))


def get_begin_last_flexyear(date, start_month=10, start_day=1):
    """
    Get the begin date of the most recent/current ("last") flexible
    year since the given date.

    Parameters
    ----------
    date: datetime.datetime
        Date from which on the most recent begin of the hydrological
        year shall be determined, e.g. today's date.
    start_month: int
        Begin month of the flexible year. Default: 10 (for hydrological
        year)
    start_day: int
        Begin day of month for the flexible year. Default: 1

    Returns
    -------
    last_begin: datetime.datetime
        Begin date of the most recent/current flexible year.

    Examples
    --------
    Find the beginning of the current mass budget year since
    2018-01-24.

    >>> import datetime as dt
    >>> get_begin_last_flexyear(dt.datetime(2018,1,24))
    dt.datetime(2017, 10, 1, 0, 0)
    >>> get_begin_last_flexyear(dt.datetime(2017,11,30),
    >>> start_month=9, start_day=15)
    dt.datetime(2017, 9, 15, 0, 0)
    """

    start_year = date.year if dt.datetime(
        date.year, start_month, start_day) <= date else date.year - 1
    last_begin = dt.datetime(start_year, start_month, start_day)

    return last_begin


def get_nash_sutcliffe_efficiency(simulated, observed):
    """
    Get the Nash-Sutcliffe efficiency coefficient.

    Parameters
    ----------
    simulated: np.array
        An array of model output or similar.
    observed: np.array
        An array of observed values.

    Returns
    -------
    nse: float
        The Nash-Sutcliffe efficiency coefficient between the two series.
    """

    nse = 1 - np.nansum((simulated - observed) ** 2) / np.nansum(
        (observed - np.nanmean(observed)) ** 2)
    return nse


class CirrusClient(pm.SSHClient):
    """
    Class for SSH interaction with Cirrus Server at WSL.
    """
    def __init__(self, credfile=None):
        """
        Initialize.

        Parameters
        ----------
        credfile: str
            Path to the credentials file (must be parsable as
            configobj.ConfigObj).
        """

        pm.SSHClient.__init__(self)

        self.sftp = None
        self.sftp_open = False
        self.ssh_open = False

        if credfile is None:
            credfile = os.path.join(os.path.abspath(os.path.dirname(
                os.path.dirname(__file__))), '.credentials')

        try:
            cr = ConfigObj(credfile, file_error=True)
        except (ConfigObjError, IOError) as e:
            log.critical('Credentials file could not be parsed (%s): %s',
                         credfile, e)
            sys.exit()

        self.cr = cr

        try:
            self.client = self.create_connect(cr['cirrus']['host'],
                                              cr['cirrus']['user'],
                                              cr['cirrus']['password'])
        except:
            raise OSError('Are you in WSL VPN network?')

    def create_connect(self, host, user, password, port=22):
        """"Establish SSH connection."""
        client = pm.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(pm.AutoAddPolicy())
        client.connect(host, port, user, password)
        self.ssh_open = True

        return client

    def _open_sftp(self):
        """Open SFTP connection if not yet done."""
        if not self.sftp_open:
            self.sftp = self.client.open_sftp()
            self.sftp_open = True

    def list_content(self, idir='.', options=''):
        """
        
        Parameters
        ----------
        idir: str
            Directory whose output shall be listed
        options: str
            Options for listing. Any one letter-option from the UNIX 'ls' 
            command is allowed.

        Returns
        -------
        The stdout of the host machine separated into lines as a list.
        """

        # Get the minus for ls options
        if options:
            options = '-' + options

        _, stdout, stderr = self.client.exec_command('ls {} {}'
                                                     .format(options, idir))
        stdout = stdout.read().splitlines()

        return stdout

    def get_files(self, remotedir, remotelist, targetdir):
        """
        Get a file from the host to a local machine.

        Parameters
        ----------
        remotedir: str
            Base remote directory as POSIX style relative path from the $HOME.
        remotelist: list
            Relative paths in POSIX style (from remote directory) to remote
            files which shall be retrieved.
        targetdir: str
            Directory where to put the files (the relative path will be
            maintained).

        Returns
        -------

        """
        self._open_sftp()

        top_sink = posixpath.split(remotedir)[-1]

        # Use posixpath for host (os.path.join joins in style of local OS)
        remotepaths = [posixpath.join(remotedir, r) for r in remotelist]

        # If you don't delete the '/' the windows part will be cut off
        remote_forjoin = [f[1:] if f[0] == '/' else f for f in remotelist]
        localpaths = [os.path.join(targetdir, top_sink, f) for f in
                      remote_forjoin]

        for remotef, localf in zip(remotepaths, localpaths):
            if not os.path.exists(os.path.dirname(localf)):
                os.makedirs(os.path.dirname(localf), exist_ok=True)
            self.sftp.get(remotef, localf)

    def sync_files(self, sourcedir, targetdir, globpattern='*',
                   rm_local=False):
        """
        Synchronize a host machine with local content.

        If there are new files, download them. If there are less files on the
        host machine now than locally, you can choose to delete them with the
        `rm_local` keyword.

        This function has some severe defects, if you don't follow the Cirrus
        "rules" (e.g. the top dir of globpattern may not be the file itself )

        This is a supercheap version of rsync which works via SFTP.
        It doesn't even consider checksums, hashs or so.
        Probably it makes sense to replace this by:
        https://stackoverflow.com/questions/16497166/
        rsync-over-ssh-using-channel-created-by-paramiko-in-python, THE PROBLEM
        IS ONLY THE WINDOWS SYSTEMS
        or this one is better:https://blog.liw.fi/posts/rsync-in-python/

        Parameters
        ----------
        sourcedir: str
            Relative path (from home directory) to a remote destination which
            shall be synced recursively.
        targetdir: str
            Absolute path to a local directory to be synced with remote
            directory.
        globpattern: str
            Pattern used for searching by glob. Default: '*' (list all files).
        rm_local: bool
            DO NOT YET USE!!!!!!!!!!!!!!!!!!!!!
            Remove also local files if they are no longer on the host machine.
            Default: False.

        Returns
        -------
        tuple of lists
            (paths to retrieved files, paths to deleted files)
        """

        # Windows also accepts "/" as separator, so everything in POSIX style
        sourcedir = sourcedir.replace("\\", '/')
        targetdir = targetdir.replace("\\", '/')
        globpattern = globpattern.replace("\\", '/')

        # Determine the "top level sink" (top directory of files to be synced)
        top_sink = posixpath.basename(sourcedir)

        # Do tricky and fake glob on host with stupid permission error catching
        _, stdout, _ = self.client.exec_command('find {} -path "{}" '
                                                '-print 2>/dev/null'
                                                .format(sourcedir,
                                                        globpattern))
        remotelist = stdout.read().splitlines()
        remotelist = [r.decode("utf-8") for r in remotelist]

        locallist = glob.glob(posixpath.join(targetdir, top_sink, globpattern))
        locallist = [l.replace("\\", '/') for l in locallist]  # IMPORTANT

        # copy everything needed from remote to local
        # LOCAL
        avail_loc = fnmatch.filter(locallist, globpattern)
        start_loc = posixpath.join(targetdir, top_sink)
        avail_loc_rel = [posixpath.relpath(p, start=start_loc) for p in
                         avail_loc]

        # REMOTE
        start_host = sourcedir
        avail_host_rel = [posixpath.relpath(p, start=start_host) for p in
                          remotelist]

        missing = list(set(avail_host_rel).difference(avail_loc_rel))

        # there might be empty files from failed syncs
        size_zero = [p for p in avail_loc if os.path.isfile(p) and
                     os.stat(p).st_size == 0]
        if size_zero:
            size_zero_rel = [posixpath.relpath(p, start=start_loc) for p in
                             size_zero]
            # Extend, but only files which are also on the host:
            missing.extend([p for p in size_zero_rel if p in avail_host_rel])

        log.info('Synchronising starts for {} files...'.format(len(missing))
                 .format(len(missing)))
        self.get_files(sourcedir, missing, targetdir)
        log.info('{} remote files were retrieved during file sync.'
                 .format(len(missing)))

        # DO NOT YET USE
        # delete unnecessary stuff if desired
        surplus = []
        if rm_local:
            available = glob.glob(os.path.join(targetdir,
                                               os.path.normpath(globpattern).
                                               split('\\')[0] + '\\**'),
                                  recursive=True)
            available = [p for p in available if os.path.isfile(p)]
            # THIS IS DANGEROUS!!!!!!!! If 'available' is too big, EVERYTHING
            # in that list is deleted
            # surplus = [p for p in available if all(x not in p for x in
            #                                 remote_npaths)]
            surplus = [p for p in available if all(x not in p for x in
                                                   remotelist)]
            print(surplus)
            log.error('Keyword rm_local must not yet be used')
            raise NotImplementedError('Keyword rm_local must not yet be used')

            '''
            if surplus:
                for s in surplus:
                    try:
                        os.remove(s)
                        log.info('{} local surplus files removed during file '
                                 'sync.'.format(len(surplus)))
                    except PermissionError:
                        log.warning('File {} could not be deleted (Permission '
                                    'denied).'.format(s))
            '''
        return missing, surplus

    def close(self):
        if self.sftp_open:
            self.sftp.close()
            self.sftp_open = False
        if self.ssh_open:
            self.client.close()
            self.ssh_open = False


@xr.register_dataset_accessor('crampon')
class MeteoTSAccessor(object):
    def __init__(self, xarray_obj):
        """
        Class for handling Meteo time series, building upon xarray.
        """
        self._obj = xarray_obj

    def update_with_verified(self, ver_path):
        """
        Updates the time series with verified MeteoSwiss data.

        Parameters
        ----------
        ver_path: str
            Path to the file with verified data (in netCDF format).
        
        Returns
        -------
        Updated xarray.Dataset.
        """

        # includes postprocessing
        ver = read_netcdf(ver_path, chunks={'time': 50},
                          tfunc=_cut_with_CH_glac)

        # this does outer joins, too
        comb = ver.combine_first(self._obj)

        # attach attribute 'last date verified' to the netcdf
        # the date itself is not allowed, so convert to str
        comb.assign_attrs({'last_verified': str(ver.time.values[-1])},
                          inplace=True)

        return comb

    def update_with_operational(self, op_path):
        """
        Updates the time series with operational MeteoSwiss data.

        The difference to self.update_with_verified is that no attribute is
        attached and the order in xr.combine_first: values of self._obj are
        prioritized.

        Parameters
        ----------
        op_path:
            Path to file with operational MeteoSwiss data.

        Returns
        -------
        Updated xarray.Dataset.
        """

        # includes postprocessing
        op = read_netcdf(op_path, chunks={'time': 50}, tfunc=_cut_with_CH_glac)

        # this does outer joins, too
        comb = self._obj.combine_first(op)

        return comb

    def ensure_time_continuity(self, freq='D', **kwargs):
        """
        Ensure the time continuity of the time series.
        
        If there are missing time steps, fill them with NaN via the 
        xr.Dataset.resample method.

        Parameters
        ----------
        freq:
            A pandas offset alias (http://pandas.pydata.org/pandas-docs/stable/
            timeseries.html#offset-aliases). Default: 'D' (daily). If None, the
            frequency will be inferred from the data itself (experimental!)
        **kwargs:
            Keywords accepted by xarray.Dataset.resample()

        
        Returns
        -------
        Resampled xarray.Dataset.
        """

        if not freq:
            freq = pd.infer_freq(self._obj.time.values)

        try:
            resampled = self._obj.resample(freq, dim='time', keep_attrs=True,
                                           **kwargs)
        # a TO DO in xarray: if not monotonic, the code throws and error
        except ValueError:
            self._obj = self._obj.sortby('time')
            resampled = self._obj.resample(freq, dim='time', keep_attrs=True,
                                           **kwargs)
        diff_a = len(set(resampled.time.values) - set(self._obj.time.values))
        diff_r = len(set(self._obj.time.values) - set(resampled.time.values))

        log.info('{} time steps were added, {} removed during resampling.'
                 .format(diff_a, diff_r))

        return resampled

    def cut_by_glacio_years(self, method='fixed'):
        """
        Evaluate the contained full glaciological years.

        Parameters
        ----------
        method: str
            'fixed' or 'file' or 'file_peryear: If fixed, the glacio years
            lasts from October, 1st to September 30th. If 'file', a CSV
            declared in 'params.cfg' (to be implemented) gives the
            climatological beginning and end of the glaciological year from
            empirical data

        Returns
        -------
        The MeteoTimeSeries, subsetted to contain only full glaciological
        years.
        """

        if method == 'fixed':
            starts = self._obj.sel(time=(self._obj['time.month'] == 10) &
                                        (self._obj['time.day'] == 1))
            ends = self._obj.sel(time=(self._obj['time.month'] == 9) &
                                      (self._obj['time.day'] == 30))

            if len(starts.time.values) == 0 or len(ends.time.values) == 0:
                raise IndexError("Time series too short to cover even one "
                                 "glaciological year.")

            glacio_start = starts.isel(time=[0]).time.values[0]
            glacio_end = ends.isel(time=[-1]).time.values[0]

            return self._obj.sel(time=slice(pd.to_datetime(glacio_start),
                                            pd.to_datetime(glacio_end)))
        else:
            raise NotImplementedError('At the moment only the fixed method'
                                      'is implemented.')

    def postprocess_cirrus(self):
        """
        Do some postprocessing for erroneous/inconsistent Cirrus data.

        Returns
        -------

        """
        # Pseudo/to do:
        # Adjust variable name/units

        # they changed the name of time coordinate....uffa!
        if "REFERENCE_TS" in self._obj.coords:
            self._obj.rename({"REFERENCE_TS": "time"}, inplace=True)

        # whatever coordinate that is
        if "crs" in self._obj.data_vars:
            self._obj = self._obj.drop(['crs'])

        # whatever coordinate that is
        if 'dummy' in self._obj.coords:
            self._obj = self._obj.drop(['dummy'])

        # whatever coordinate that is
        if 'latitude_longitude' in self._obj.coords:
            self._obj = self._obj.drop(['latitude_longitude'])
        if 'latitude_longitude' in self._obj.variables:
            self._obj = self._obj.drop(['latitude_longitude'])

        if 'longitude_latitude' in self._obj.coords:
            self._obj = self._obj.drop(['longitude_latitude'])
        if 'longitude_latitude' in self._obj.variables:
            self._obj = self._obj.drop(['longitude_latitude'])

        # this is the case for the operational files
        if 'x' in self._obj.coords:
            self._obj.rename({'x': 'lon'}, inplace=True)

        # this is the case for the operational files
        if 'y' in self._obj.coords:
            self._obj.rename({'y': 'lat'}, inplace=True)

        # Latitude can be switched after 2014
        self._obj = self._obj.sortby('lat')

        # make R variable names the same so that we don't get in troubles
        if 'RprelimD' in self._obj.variables:
            self._obj.rename({'RprelimD': 'RD'}, inplace=True)
        if 'RhiresD' in self._obj.variables:
            self._obj.rename({'RhiresD': 'RD'}, inplace=True)

        # radiation: "SIS" & "msg.SIS.D" in one file
        if ('msg.SIS.D' in self._obj.variables) and \
                ('SIS' in self._obj.variables):
            self._obj = self._obj['msg.SIS.D'].combine_first(
                self._obj.SIS).to_dataset(
                name='SIS')
        elif ('msg.SIS.D' in self._obj.variables) and not \
                ('SIS' in self._obj.variables):
            self._obj.rename({'msg.SIS.D': 'SIS'}, inplace=True)

        # THIS IS ABSOLUTELY TEMPORARY AND SHOULD BE REPLACED
        # THE REASON IS A SLIGHT PRECISION PROBLEM IN THE INPUT DATA, CHANGING
        # AT THE 2014/2015 TRANSITION => WE STANDARDIZE THE COORDINATES BY HAND
        lats = np.array([45.75, 45.77083333, 45.79166667, 45.8125,
                         45.83333333, 45.85416667, 45.875, 45.89583333,
                         45.91666667, 45.9375, 45.95833333, 45.97916667,
                         46., 46.02083333, 46.04166667, 46.0625,
                         46.08333333, 46.10416667, 46.125, 46.14583333,
                         46.16666667, 46.1875, 46.20833333, 46.22916667,
                         46.25, 46.27083333, 46.29166667, 46.3125,
                         46.33333333, 46.35416667, 46.375, 46.39583333,
                         46.41666667, 46.4375, 46.45833333, 46.47916667,
                         46.5, 46.52083333, 46.54166667, 46.5625,
                         46.58333333, 46.60416667, 46.625, 46.64583333,
                         46.66666667, 46.6875, 46.70833333, 46.72916667,
                         46.75, 46.77083333, 46.79166667, 46.8125,
                         46.83333333, 46.85416667, 46.875, 46.89583333,
                         46.91666667, 46.9375, 46.95833333, 46.97916667,
                         47., 47.02083333, 47.04166667, 47.0625,
                         47.08333333, 47.10416667, 47.125, 47.14583333,
                         47.16666667, 47.1875, 47.20833333, 47.22916667,
                         47.25, 47.27083333, 47.29166667, 47.3125,
                         47.33333333, 47.35416667, 47.375, 47.39583333,
                         47.41666667, 47.4375, 47.45833333, 47.47916667,
                         47.5, 47.52083333, 47.54166667, 47.5625,
                         47.58333333, 47.60416667, 47.625, 47.64583333,
                         47.66666667, 47.6875, 47.70833333, 47.72916667,
                         47.75, 47.77083333, 47.79166667, 47.8125,
                         47.83333333, 47.85416667, 47.875])

        lons = np.array([5.75, 5.77083333, 5.79166667, 5.8125,
                         5.83333333, 5.85416667, 5.875, 5.89583333,
                         5.91666667, 5.9375, 5.95833333, 5.97916667,
                         6., 6.02083333, 6.04166667, 6.0625,
                         6.08333333, 6.10416667, 6.125, 6.14583333,
                         6.16666667, 6.1875, 6.20833333, 6.22916667,
                         6.25, 6.27083333, 6.29166667, 6.3125,
                         6.33333333, 6.35416667, 6.375, 6.39583333,
                         6.41666667, 6.4375, 6.45833333, 6.47916667,
                         6.5, 6.52083333, 6.54166667, 6.5625,
                         6.58333333, 6.60416667, 6.625, 6.64583333,
                         6.66666667, 6.6875, 6.70833333, 6.72916667,
                         6.75, 6.77083333, 6.79166667, 6.8125,
                         6.83333333, 6.85416667, 6.875, 6.89583333,
                         6.91666667, 6.9375, 6.95833333, 6.97916667,
                         7., 7.02083333, 7.04166667, 7.0625,
                         7.08333333, 7.10416667, 7.125, 7.14583333,
                         7.16666667, 7.1875, 7.20833333, 7.22916667,
                         7.25, 7.27083333, 7.29166667, 7.3125,
                         7.33333333, 7.35416667, 7.375, 7.39583333,
                         7.41666667, 7.4375, 7.45833333, 7.47916667,
                         7.5, 7.52083333, 7.54166667, 7.5625,
                         7.58333333, 7.60416667, 7.625, 7.64583333,
                         7.66666667, 7.6875, 7.70833333, 7.72916667,
                         7.75, 7.77083333, 7.79166667, 7.8125,
                         7.83333333, 7.85416667, 7.875, 7.89583333,
                         7.91666667, 7.9375, 7.95833333, 7.97916667,
                         8., 8.02083333, 8.04166667, 8.0625,
                         8.08333333, 8.10416667, 8.125, 8.14583333,
                         8.16666667, 8.1875, 8.20833333, 8.22916667,
                         8.25, 8.27083333, 8.29166667, 8.3125,
                         8.33333333, 8.35416667, 8.375, 8.39583333,
                         8.41666667, 8.4375, 8.45833333, 8.47916667,
                         8.5, 8.52083333, 8.54166667, 8.5625,
                         8.58333333, 8.60416667, 8.625, 8.64583333,
                         8.66666667, 8.6875, 8.70833333, 8.72916667,
                         8.75, 8.77083333, 8.79166667, 8.8125,
                         8.83333333, 8.85416667, 8.875, 8.89583333,
                         8.91666667, 8.9375, 8.95833333, 8.97916667,
                         9., 9.02083333, 9.04166667, 9.0625,
                         9.08333333, 9.10416667, 9.125, 9.14583333,
                         9.16666667, 9.1875, 9.20833333, 9.22916667,
                         9.25, 9.27083333, 9.29166667, 9.3125,
                         9.33333333, 9.35416667, 9.375, 9.39583333,
                         9.41666667, 9.4375, 9.45833333, 9.47916667,
                         9.5, 9.52083333, 9.54166667, 9.5625,
                         9.58333333, 9.60416667, 9.625, 9.64583333,
                         9.66666667, 9.6875, 9.70833333, 9.72916667,
                         9.75, 9.77083333, 9.79166667, 9.8125,
                         9.83333333, 9.85416667, 9.875, 9.89583333,
                         9.91666667, 9.9375, 9.95833333, 9.97916667,
                         10., 10.02083333, 10.04166667, 10.0625,
                         10.08333333, 10.10416667, 10.125, 10.14583333,
                         10.16666667, 10.1875, 10.20833333, 10.22916667,
                         10.25, 10.27083333, 10.29166667, 10.3125,
                         10.33333333, 10.35416667, 10.375, 10.39583333,
                         10.41666667, 10.4375, 10.45833333, 10.47916667,
                         10.5, 10.52083333, 10.54166667, 10.5625,
                         10.58333333, 10.60416667, 10.625, 10.64583333,
                         10.66666667, 10.6875, 10.70833333, 10.72916667,
                         10.75])

        try:
            self._obj.coords['lat'] = lats
        except ValueError:
            pass
        try:
            self._obj.coords['lon'] = lons
        except ValueError:
            pass

        return self._obj


def daily_climate_from_netcdf(tfile, pfile, rfile, hfile, outfile):
    """
    Create a netCDF file with daily temperature, precipitation and
    elevation reference from given files.

    The file format will be as OGGM likes it.
    The temporal extent of the file will be the inner or outer join of the time
    series extent of the given input files .

    Parameters
    ----------
    tfile: str
        Path to temperature netCDF file.
    pfile: str
        Path to precipitation netCDF file.
    rfile: str
        Path to radiation netCDF file.
    hfile: str
        Path to the elevation netCDF file.
    outfile: str
        Path to and name of the written file.

    Returns
    -------

    """

    temp = read_netcdf(tfile, chunks={'time': 50})
    prec = read_netcdf(pfile, chunks={'time': 50})
    sis = read_netcdf(rfile, chunks={'time': 50})  # shortwave incoming solar
    hgt = read_netcdf(hfile)
    _, hgt = xr.align(temp, hgt, join='left')

    # Rename variables as OGGM likes it
    if 'TabsD' in temp.variables:
        temp.rename({'TabsD': 'temp'}, inplace=True)
    if 'RD' in prec.variables:
        prec.rename({'RD': 'prcp'}, inplace=True)
    if 'SIS' in sis.variables:
        sis.rename({'SIS': 'sis'}, inplace=True)

    # make it one
    nc_ts = xr.merge([temp, prec, sis, hgt])

    # Units cannot be checked anymore at this place (lost in xarray...)

    # ensure it's compressed when exporting
    nc_ts.encoding['zlib'] = True
    nc_ts.to_netcdf(outfile)


def read_netcdf(path, chunks=None, tfunc=None):
    # use a context manager, to ensure the file gets closed after use
    with xr.open_dataset(path, cache=False) as ds:
        # some extra stuff - this is actually stupid and should go away!
        ds = ds.crampon.postprocess_cirrus()

        ds = ds.chunk(chunks=chunks)
        # transform_func should do some sort of selection or
        # aggregation
        if tfunc is not None:
            ds = tfunc(ds)

        # load all data from the transformed dataset, to ensure we can
        # use it after closing each original file
        ds.load()
        return ds


def read_multiple_netcdfs(files, dim='time', chunks=None, tfunc=None):
    """
    Read several netCDF files at once. Requires dask module.

    Changed from:  http://xarray.pydata.org/en/stable/io.html#id7

    Parameters
    ----------
    files: list
        List with paths to the files to be read.
    dim: str
        Dimension along which to concatenate the files.
    tfunc: function
        Transformation function for the data, e.g. 'lambda ds: ds.mean()'
    chunks: dict
        Chunk sizes as can be specified to xarray.open_dataset.

    Returns
    -------
    A concatenation of the input files as xarray.Dataset.
    """

    paths = sorted(files)
    datasets = [read_netcdf(p, chunks, tfunc) for p in paths]

    combined = xr.auto_combine(datasets, concat_dim=dim)
    return combined


# can't write it to cache until we have e.g. a date in the climate_all filename
@MEMORY.cache
def joblib_read_climate_crampon(ncpath, ilon, ilat, default_grad, minmax_grad,
                                use_grad):
    """
    This is a cracked version of the OGGM function with some extras.

    Parameters
    ----------
    ncpath: str
        Path to the netCDF file in OGGM suitable format.
    ilon: int
        Index of a longitude in the netCDF.
    ilat: int
        Index of a latitude in the netCDF.
    default_grad: float
        Default temperature gradient (K/m).
    minmax_grad: tuple
        Min/Max bounds of the local gradient, in case the grid kernel search
        delivers strange values.
    use_grad: int
        Window edge width of surrounding cells used to determine the local
        temperature gradient. Must be an odd number. If 0,
        the ``default_grad`` is used.

    Returns
    -------
    iprcp, itemp, igrad, ihgt:
    Precipitation, temperature, temperature gradient and elevation
    at given latitude/longitude indices.
    """

    # check for oddness or zero
    if not (divmod(use_grad, 2)[1] == 1 or use_grad == 0):
        raise ValueError('Window edge width must be odd number or zero.')

    # read the file and data
    with netCDF4.Dataset(ncpath, mode='r') as nc:
        temp = nc.variables['temp']
        prcp = nc.variables['prcp']
        hgt = nc.variables['hgt']
        igrad = np.zeros(len(nc.dimensions['time'])) + default_grad
        iprcp = prcp[:, ilat, ilon]
        itemp = temp[:, ilat, ilon]
        ihgt = hgt[ilat, ilon]

        # check for missing meteo input and fill (temp mean and precip zero)
        t_miss = np.where(np.isnan(itemp))
        p_miss = np.where(np.isnan(iprcp))

        if t_miss[0].any():
            log.warning("Temp missing for time {}".format(
                [i.strftime('%Y-%m-%d') for i in
                 netCDF4.num2date(nc['time'][t_miss], nc['time'].units,
                                  nc['time'].calendar)]))
            for m in t_miss:
                itemp[m] = (itemp[m - 1] + itemp[m + 1]) / 2.
                t_load = temp[:]
                # either mean of before and after, or value of before
                try:
                    t_load[m[0], :, :][:] = ((temp[m - 1] + temp[m + 1]) / 2.)[
                                            :]
                except IndexError:
                    t_load[m][0] = temp[m - 1][:]
                # important until MaskedArrayFutureWarning is implemented!
                t_load[m[0], :, :][:].mask = temp[m - 1][:].mask
                temp = t_load.copy()
                t_load = None

        if p_miss[0].any():
            log.warning("Precip missing for time {}".format(
                [i.strftime('%Y-%m-%d') for i in
                 netCDF4.num2date(nc['time'][p_miss], nc['time'].units,
                                  nc['time'].calendar)]))

            for n in p_miss:
                p_load = prcp[:]
                iprcp[n] = 0.
                p_load[n[0], :, :][:] = np.zeros_like(p_load[n[0]])[:]
                p_load[n[0], :, :][:].mask = prcp[n - 1][:].mask  # important!
                prcp = p_load.copy()
                p_load = None

        # Temperature gradient
        if use_grad != 0:
            # some min/max constants for the window
            minw = divmod(use_grad, 2)[0]
            maxw = divmod(use_grad, 2)[0] + 1

            ttemp = temp[:, ilat-minw:ilat+maxw, ilon-minw:ilon+maxw]
            thgt = hgt[ilat-minw:ilat+maxw, ilon-minw:ilon+maxw]
            thgt = thgt.flatten()

            for t, loct in enumerate(ttemp):
                # this happens a the grid edges:
                if isinstance(loct, np.ma.masked_array):
                    slope, _, _, p_val, _ = stats.linregress(
                        np.ma.masked_array(thgt, loct.mask).compressed(),
                        loct.flatten().compressed())
                else:
                    slope, _, _, p_val, _ = stats.linregress(thgt,
                                                             loct.flatten())
                # if the result is
                igrad[t] = slope if (p_val < 0.01) else default_grad

            # apply the boundaries, in case the gradient goes wild
            igrad = np.clip(igrad, minmax_grad[0], minmax_grad[1])

    return iprcp, itemp, igrad, ihgt


# IMPORTANT: overwrite OGGM functions with same name
joblib_read_climate = joblib_read_climate_crampon


def _cut_with_CH_glac(xr_ds):
    """
    Preliminary version that cuts an xarray.Dataset to Swiss glacier shapes.

    At the moment this is just a rectangle clip, but more work is on the way:
    https://github.com/pydata/xarray/issues/501

    Parameters
    ----------
    xr_ds: xarray.Dataset
        The Dataset to be clipped.

    Returns
    -------
    The clipped xarray.Dataset.
    """

    xr_ds = xr_ds.where(xr_ds.lat >= 45.7321, drop=True)
    xr_ds = xr_ds.where(xr_ds.lat <= 47.2603, drop=True)
    xr_ds = xr_ds.where(xr_ds.lon >= 6.79963, drop=True)
    xr_ds = xr_ds.where(xr_ds.lon <= 10.4279, drop=True)

    return xr_ds


def make_swisstopo_worksheet(st_dir, in_epsg='21781', out_epsg='4326'):
    """
    Creates a tile worksheet for the swisstopo tiles (DHM25/Swissalti3d).

    Writes the output shapefile in WGS84 lat/lon (EPSG:4326)!

    Parameters
    ----------
    st_dir: str
        Path to the directory where the swisstopo files are placed.
    in_epsg: str
        EPSG number of input ASCII grids as string.
    out_epsg: str
        EPSG number of output shapefile as string.
    """

    st_dir = Path(st_dir)

    # tile to year mapper
    mapper = pd.read_csv(str(st_dir.parent.joinpath(*['worksheets',
                                                      '{}_tile_to_year.csv'
                                                    .format(st_dir.stem)])))

    crs = {'init': 'epsg:{}'.format(in_epsg)}
    ws_gdf = gpd.GeoDataFrame({**{'geometry': []}, **dict()}, crs=crs)

    # '.' as trick to find only files (any extension) in glob
    for i, f in enumerate(st_dir.glob('*.*')):
        # TODO: Replace by general raster bounds to extent polygon function?
        raster = rasterio.open(str(f))
        extent = raster.bounds
        poly = shapely.geometry.polygon.Polygon(
            [[extent[0], extent[3]],
             [extent[2], extent[3]],
             [extent[2], extent[1]],
             [extent[0], extent[1]]])
        ws_gdf.at[i, 'geometry'] = poly
        zone = f.name[2:6]
        ws_gdf.at[i, 'zone'] = zone
        try:
            ws_gdf.at[i, 'date'] = mapper[mapper.tile ==
                                          int(zone)].year.values[0]
        # TODO: This should not happen if the mapper was complete!
        except IndexError:
            pass

    ws_gdf = ws_gdf.to_crs(epsg=out_epsg)

    outpath = str(st_dir.parent.joinpath(*['worksheets', 'worksheet_{}.shp'
                                         .format(st_dir.stem)]))
    ws_gdf.to_file(outpath)


def get_zones_from_worksheet(worksheet, id_col, gdir=None, shape=None,
                            gpd_obj=None):
    """
    Return the zone numbers from a DEM tiles worksheet.

    Parameters
    ----------
    worksheet: str
        Path to the worksheet with the zone numbers.
    id_col: str
        Identifier (ID) column name in the worksheet
    gdir: GlacierDirectory
        GlacierDirectory object containing outlines.shp or outlines_ts.shp.
    shape: str
        Path to a shapefile.
    gpd_obj: geopandas.GeoDataFrame or geopandas.Series
        A GeoDataFrame oder GeoSeries instance.

    Returns
    -------
    A list of zones giving the intersecting zone numbers.
    """

    # Process either input to be GeoDataFrame in the end
    if gdir is not None:
        try:
            gdf = gpd.read_file(gdir.get_filepath('outlines_ts'))
        except OSError:  # no such file or directory
            gdf = gpd.read_file(gdir.get_filepath('outlines'))
    if shape is not None:
        gdf = gpd.read_file(shape)
    if gpd_obj is not None:
        if isinstance(gpd_obj, gpd.GeoSeries):
            gdf = gpd.GeoDataFrame({**{'geometry': gpd_obj.geometry},
                                    **dict()}, index=[0])
        else:
            gdf = gpd_obj.copy()

    # overlay only works when both input have the same CRS
    w_gdf = gpd.read_file(worksheet)
    w_gdf = w_gdf.to_crs(gdf.crs)

    res_is = gpd.overlay(gdf, w_gdf, how='intersection')

    # zones might be duplicates if the glacier shape is 'winding'
    return np.unique(res_is[id_col].tolist())


def mount_network_drive(path, user, log=None):
    """
    Mount a network drive.

    If the network drive is already mounted, Windows will cause a
    'Systemfehler 1219' which does not have any consequences. However, when
    further errors occur (path/user wrong etc.) this will cause a silent fail.
    An option to catch this silent error should be implemented.

    Parameters
    ----------
    path: str
        Path to the network drive.
    user: str
        User to establish connection.

    Returns
    -------
    out: int
        0 if mounting succeeded, 1 if failed.
    """
    out = subprocess.call(r'net use {} {}'.format(path, user),
                          stdout=subprocess.PIPE, shell=True)

    if out == 0:
        msg = 'Network drive {} successfully connected'.format(path)
        log.info(msg) if log else print(msg)
    else:
        raise ConnectionError('Network drive {} connection failed'
                              .format(path))
    return out


def _local_dem_to_xr_dataset(to_merge, acq_date, calendar_startyear=0,
                             miss_val=np.nan):
    """
    Hard-coded function that reads DEMs into xarray.Datasets and adds metadata.

    Opens files, names the variable correctly for merging, gets rid of the
    'band' dimension, assign time coordinate and expands along this coordinate,
    adds encoding and missing value specifications.

    Parameters
    ----------
    to_merge: list
        List of DEM paths to merge.
    acq_date: datetime.datetime
        Datetime object delivering the acquisition date of the DEMs.
    calendar_startyear: int
        Year when the netCDF calendar should begin (e.g. 'Years since 1900').
        Default: 1900
    miss_val: float
        Missing value. Default: -9999.0

    Returns
    -------
    ds_final: xr.Dataset
        The merged dataset of all input DEMs.
    """
    # open and assign name
    xr_das = [xr.open_rasterio(tm, chunks=(50, 50)) for tm in to_merge]
    xr_das = [t.rename('height') for t in xr_das]

    # load and merge; overwrite is necessary, as some rasters cover same area
    xr_das = [d.load() for d in xr_das]
    try:
        merged = xr.merge(xr_das)
    # this happens for DHM25 on Rhone, for example (overlapping tiles)
    except xr.core.merge.MergeError:
        merged = None
        if len(xr_das) > 1:
            while len(xr_das) >= 1:
                if merged is not None:
                    merged = merged.combine_first(xr_das[-1])
                    del xr_das[-1]
                else:
                    merged = xr_das[0].combine_first(xr_das[1])
                    del xr_das[0:2]
        else:
            merged = xr_das[0]

    if isinstance(merged, xr.core.dataarray.DataArray):
        merged = merged.to_dataset(name='height')

    # coord/dim changes
    merged = merged.squeeze(dim='band')
    merged = merged.drop('band')
    # Replaces -9999.0 with NaN: See here:
    # https://github.com/pydata/xarray/issues/1749
    merged = merged.where(merged != -9999.0)
    merged = merged.assign_coords(time=acq_date)
    try:
        final_ds = merged.expand_dims('time')
    except ValueError:
        final_ds = merged.copy()
    # set EPSG:21781
    final_ds.attrs['pyproj_srs'] = '+proj=somerc +lat_0=46.95240555555556 ' \
                             '+lon_0=7.439583333333333 +k_0=1 +x_0=600000 ' \
                             '+y_0=200000 +ellps=bessel ' \
                             '+towgs84=674.374,15.056,405.346,0,0,0,0 ' \
                             '+units=m +no_defs'
    encoding = {'height': {'_FillValue': miss_val, 'units': 'meters', 'date':
        acq_date.strftime("%Y%m%d"), 'standard_name':
        'height_above_reference_ellipsoid'}, 'time': {'calendar': 'standard'},
                'zlib': True}
    final_ds.encoding = encoding

    return final_ds


def get_local_dems(gdir):
    """
    Gather the locally saved DEMs.

    At the moment, solutions for subfolders of cfg.PATHS['dem_dir'] (DHM25 and
    SwissALTI3D) as well as the National Forest Inventory DEMs on the network
    drive are implemented.
    The transformation cod from Swiss coordinates to lat/lon comes from the
    xarray webpage [1]_.

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        A GlacierDirectory instance.

    Returns
    -------
    None

    References
    ----------
    .. [1] http://xarray.pydata.org/en/stable/auto_gallery/plot_rasterio.html#recipes-rasterio
    """

    # get forest inventory DEMs (quite hard-coded for the LFI file names)
    _ = mount_network_drive(cfg.PATHS['lfi_dir'], r'wsl\landmann', log=log)

    # neither os nor pathlib works, so quick'n'dirty:
    lfi_dem_list = glob.glob(cfg.PATHS['lfi_dir']+'\\TIFF\\*.tif')

    if not lfi_dem_list:
        raise FileNotFoundError('National Forest Inventory DEMs not found!')

    # exactly one file name is broken:
    broke_months = ['ADS_103221_672000_165000_2011_0_0_86_50.tif']
    lfi_dem_list = [x for x in lfi_dem_list if not broke_months[0] in x]

    zones = get_zones_from_worksheet(cfg.PATHS['lfi_worksheet'], 'CLNR',
                                     gdir=gdir)

    # 'common' DEMs for this glacier (all years!)
    cdems = []
    for z in zones:
        plist = [x for x in lfi_dem_list if 'ADS_{}'.format(z) in x]
        cdems.extend(plist)

    # check the common dates for all zones
    cdates = [dt.datetime(int(os.path.basename(x).split('_')[4]),
                          int(os.path.basename(x).split('_')[5]),
                          int(os.path.basename(x).split('_')[6])) for
              x in cdems]
    cdates = np.unique(cdates)
    cyears = np.unique([d.year for d in cdates])

    lfi_all = []
    for cd in cyears:
        log.info('Assembling LFI DEMs for {} in {}'.format(gdir.rgi_id, cd))

        lfi_to_merge = [c for c in cdems if '_{}_'.format(str(cd)) in c]

        # stupid check for overlapping resolutions: the bigger number wins
        present_50 = np.sum(['50.tif' in c for c in lfi_to_merge])
        present_25 = np.sum(['25.tif' in c for c in lfi_to_merge])
        if present_50 > 0 and present_25 > 0:
            if present_50 >= present_25:
                lfi_to_merge = [i for i in lfi_to_merge if not '25.tif' in i]
            else:
                lfi_to_merge = [i for i in lfi_to_merge if '25.tif' in i]

        # acquisition dates
        lfi_dates = [dt.datetime(int(os.path.basename(x).split('_')[4]),
                                 int(os.path.basename(x).split('_')[5]),
                                 int(os.path.basename(x).split('_')[6]))
                     for x in lfi_to_merge]
        lfi_dates = np.unique(lfi_dates)

        if len(lfi_dates) > 1:
            log.info('Skipping year {} (has {} acquisition dates).'.format(
                cd, len(lfi_dates)))
            continue

        # merge and append
        lfi_dem = _local_dem_to_xr_dataset(lfi_to_merge, lfi_dates[0])

        # check holes: small => fill, large=> no fill. Check percentage of NaN
        #checked = gis.dem_quality_check(lfi_dem)

        #if checked:
        lfi_all.append(lfi_dem)
        #else:
        #    log.info('Skipping DEM from {} (did not pass quality check)'
        #             .format(lfi_dem.time))
        #    continue

    aligned = xr.align(*lfi_all, join='outer', exclude='time')
    concat = xr.concat(aligned, dim='time')

    concat.to_netcdf(path=gdir.get_filepath('dem_ts'), mode='w',
                     group=cfg.NAMES['LFI'])

    # get DHM25 DEMs
    log.info('Assembling DHM25 DEM for {}'.format(gdir.rgi_id))
    d_list = glob.glob(cfg.PATHS['dem_dir']+'\\*'+cfg.NAMES['DHM25'] +
                       '*\\*.agr')
    d_ws_path = glob.glob(os.path.join(cfg.PATHS['dem_dir'], 'worksheets',
                                       '*' + cfg.NAMES['DHM25'] + '*.shp'))
    d_zones = get_zones_from_worksheet(d_ws_path[0], 'zone', gdir=gdir)
    d_to_merge = []
    for d_z in d_zones:
        d_to_merge.extend([d for d in d_list if str(d_z) in d])
    # TODO: Replace with real acquisition dates!
    d_acq_dates = dt.datetime(1970, 1, 1)
    d_dem = _local_dem_to_xr_dataset(d_to_merge, d_acq_dates)
    d_dem.to_netcdf(path=gdir.get_filepath('dem_ts'), mode='a',
                    group=cfg.NAMES['DHM25'])

    # get SwissALTI3D DEMs
    log.info('Assembling SwissALTI3D DEM for {}'.format(gdir.rgi_id))
    a_list = glob.glob(cfg.PATHS['dem_dir']+'\\*'+
                       cfg.NAMES['SWISSALTI2010']+'*\\*.agr')
    a_ws_path = glob.glob(os.path.join(cfg.PATHS['dem_dir'], 'worksheets',
                                       '*'+cfg.NAMES['SWISSALTI2010']+'*.shp'))
    a_zones = get_zones_from_worksheet(a_ws_path[0], 'zone', gdir=gdir)
    a_to_merge = []
    for a_z in a_zones:
        a_to_merge.extend([d for d in a_list if str(a_z) in d])
    # TODO: Replace with real acquisition dates!
    a_acq_dates = dt.datetime(2010, 1, 1)
    a_dem = _local_dem_to_xr_dataset(a_to_merge, a_acq_dates)
    a_dem.to_netcdf(path=gdir.get_filepath('dem_ts'), mode='a',
                    group=cfg.NAMES['SWISSALTI2010'])


def get_cirrus_yesterday():
    """Check if data has already been delivered."""
    try:
        climate = xr.open_dataset(cfg.PATHS['climate_file'])
        climfile_yesterday = pd.Timestamp(climate.time.values[-1])
    except (KeyError, FileNotFoundError):
        log.warning('Can\'t determine the \'s yesterday from climate file.')
    now = dt.datetime.now()
    if now.hour >= 12 and now.minute >= 30:
        yesterday = (now - dt.timedelta(1))
    else:
        yesterday = (now - dt.timedelta(2))

    return climfile_yesterday


OGGMGlacierDirectory = GlacierDirectory


class GlacierDirectory(object):
    """
    Organizes read and write access to the glacier's files.

    This is an extension of the oggm.GlacierDirectory [1]_ for CRAMPON needs.

    Some functions are the same, but the focus on RGI is loosened while trying
    to implement solutions for handling multitemporal data.
    It handles a glacier directory created in a base directory (default
    is the "per_glacier" folder in the working directory). The role of a
    GlacierDirectory is to give access to file paths and to I/O operations.
    The user should not care about *where* the files are
    located, but should know their name (specified in cfg.BASENAMES).
    If the directory does not exist, it will be created.

    Attributes
    ----------
    dir : str
        path to the directory
    rgi_id : str
        The glacier's RGI identifier (when available)
    glims_id : str
        The glacier's GLIMS identifier (when available)
    rgi_area_km2 : float
        The glacier's RGI area (km2)
    cenlon, cenlat : float
        The glacier centerpoint's lon/lat
    rgi_date : datetime
        The RGI's BGNDATE attribute if available. Otherwise, defaults to
        2003-01-01
    rgi_region : str
        The RGI region name
    name : str
        The RGI glacier name (if Available)
    glacier_type : str
        The RGI glacier type ('Glacier', 'Ice cap', 'Perennial snowfield',
        'Seasonal snowfield')
    terminus_type : str
        The RGI terminus type ('Land-terminating', 'Marine-terminating',
        'Lake-terminating', 'Dry calving', 'Regenerated', 'Shelf-terminating')
    is_tidewater : bool
        Is the glacier a caving glacier?
    inversion_calving_rate : float
        Calving rate used for the inversion

    References
    ----------
    .. [1] http://oggm.readthedocs.io/en/latest/generated/oggm.GlacierDirectory.html#oggm.GlacierDirectory
    """

    def __init__(self, entity, base_dir=None, reset=False):
        """Creates a new directory or opens an existing one.
        Parameters
        ----------
        entity : a `GeoSeries <http://geopandas.org/data_structures.html#geoseries>`_ or str
            glacier entity read from the shapefile (or a valid RGI ID if the
            directory exists)
        base_dir : str
            path to the directory where to open the directory.
            Defaults to `cfg.PATHS['working_dir'] + /per_glacier/`
        reset : bool, default=False
            empties the directory at construction (careful!)
        """

        # Making oggm.GlacierDirectory available for composition
        self.OGGMGD = OGGMGlacierDirectory(entity, base_dir=base_dir, reset=reset)

        if base_dir is None:
            if not cfg.PATHS.get('working_dir', None):
                raise ValueError("Need a valid PATHS['working_dir']!")
            base_dir = os.path.join(cfg.PATHS['working_dir'],
                                    'per_glacier')

        # IDs are also valid entries
        if isinstance(entity, str):
            # TODO: Think if :8 and :11 makes sense for other invs than RGI
            _shp = os.path.join(base_dir, entity[:8], entity[:11],
                                entity, 'outlines.shp')
            entity = read_shapefile(_shp)
            crs = salem.check_crs(entity.crs)
            entity = entity.iloc[0]
            xx, yy = salem.transform_proj(crs, salem.wgs84,
                                          [entity['min_x'],
                                           entity['max_x']],
                                          [entity['min_y'],
                                           entity['max_y']])
        else:
            g = entity['geometry']
            xx, yy = ([g.bounds[0], g.bounds[2]],
                      [g.bounds[1], g.bounds[3]])

        # Extent of the glacier in lon/lat
        self.extent_ll = [xx, yy]

        # Try which inventory it is
        # RGI_v4 no longer supported
        if 'RGIID' in entity:
            raise ValueError('RGI Version 4 is not supported anymore')
        elif 'RGIId' in entity:
            self.inventory = 'RGI'
        elif 'sgi_r2018' in entity:
            self.inventory = 'SGI_2018'
        else:
            raise NotImplementedError('This inventory is not yet understood')

        # TODO: Make other inventories than RGI possible
        if self.inventory == 'RGI':
            # Should be V5
            self.id = entity.RGIId
            self.rgi_id = entity.RGIId
            self.glims_id = entity.GLIMSId
            self.area_km2 = float(entity.Area)
            self.rgi_area_km2 = float(entity.Area)
            self.cenlon = float(entity.CenLon)
            self.cenlat = float(entity.CenLat)
            self.rgi_region = '{:02d}'.format(int(entity.O1Region))
            self.rgi_subregion = (self.rgi_region + '-' +
                                  '{:02d}'.format(
                                      int(entity.O2Region)))
            name = entity.Name
            rgi_datestr = entity.BgnDate

            try:
                gtype = entity.GlacType
            except AttributeError:
                # RGI V6
                gtype = [str(entity.Form), str(entity.TermType)]

            # rgi version can be useful
            self.rgi_version = self.rgi_id.split('-')[0][-2]
            if self.rgi_version not in ['5', '6']:
                raise RuntimeError('RGI Version not understood: '
                                   '{}'.format(self.rgi_version))

            # remove spurious characters and trailing blanks
            self.name = filter_rgi_name(name)

            # region
            reg_names, subreg_names = parse_rgi_meta(version=self.rgi_version)
            n = reg_names.loc[int(self.rgi_region)].values[0]
            self.rgi_region_name = self.rgi_region + ': ' + n
            n = subreg_names.loc[self.rgi_subregion].values[0]
            self.rgi_subregion_name = self.rgi_subregion + ': ' + n

            # Read glacier attrs
            gtkeys = {'0': 'Glacier',
                      '1': 'Ice cap',
                      '2': 'Perennial snowfield',
                      '3': 'Seasonal snowfield',
                      '9': 'Not assigned',
                      }
            ttkeys = {'0': 'Land-terminating',
                      '1': 'Marine-terminating',
                      '2': 'Lake-terminating',
                      '3': 'Dry calving',
                      '4': 'Regenerated',
                      '5': 'Shelf-terminating',
                      '9': 'Not assigned',
                      }
            self.glacier_type = gtkeys[gtype[0]]
            self.terminus_type = ttkeys[gtype[1]]
            self.is_tidewater = self.terminus_type in ['Marine-terminating',
                                                       'Lake-terminating']
            self.inversion_calving_rate = 0.
            self.is_icecap = self.glacier_type == 'Ice cap'

            # Hemisphere
            self.hemisphere = 'sh' if self.cenlat < 0 else 'nh'

            # convert the date
            try:
                rgi_date = pd.to_datetime(rgi_datestr[0:4],
                                          errors='raise', format='%Y')
            except:
                rgi_date = None
            self.rgi_date = rgi_date
        elif self.inventory == 'SGI':
            pass

        # The divides dirs are created by gis.define_glacier_region, but we
        # make the root dir
        self.dir = os.path.join(base_dir, self.id[:8],
                                self.id[:11],
                                self.id)
        if reset and os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        mkdir(self.dir)

        # logging file
        self.logfile = os.path.join(self.dir, 'log.txt')

        # Optimization
        self._mbdf = None

    def __repr__(self):

        summary = ['<crampon.GlacierDirectory>']
        summary += ['Inventory Type: ' + self.inventory]
        if self.name:
            summary += ['  Name: ' + self.name]
        if os.path.isfile(self.get_filepath('glacier_grid')):
            summary += ['  Grid (nx, ny): (' + str(self.grid.nx) + ', ' +
                        str(self.grid.ny) + ')']
            summary += ['  Grid (dx, dy): (' + str(self.grid.dx) + ', ' +
                        str(self.grid.dy) + ')']
        if self.inventory == 'RGI':
            summary += ['  ID: ' + self.rgi_id]
            summary += ['  Region: ' + self.rgi_region_name]
            summary += ['  Subregion: ' + self.rgi_subregion_name]
            summary += ['  Glacier type: ' + str(self.glacier_type)]
            summary += ['  Terminus type: ' + str(self.terminus_type)]
            summary += ['  Area: ' + str(self.rgi_area_km2) + ' km2']
            summary += ['  Lon, Lat: (' + str(self.cenlon) + ', ' +
                        str(self.cenlat) + ')']
        elif self.inventory == 'SGI':
            summary += ['  ID: ' + self.rgi_id]
            summary += ['  Area: ' + str(self.area_km2) + ' km2']
            summary += ['  Lon, Lat: (' + str(self.cenlon) + ', ' +
                        str(self.cenlat) + ')']

        return '\n'.join(summary) + '\n'

    @lazy_property
    def grid(self):
        """A ``salem.Grid`` handling the georeferencing of the local grid"""
        return salem.Grid.from_json(self.get_filepath('glacier_grid'))

    @lazy_property
    def area_m2(self):
        """The glacier's RGI area (m2)."""
        return self.area_km2 * 10 ** 6

    # something we still need, because we use OGGM functions. Might leave soon.
    @lazy_property
    def rgi_area_m2(self):
        return self.area_m2

    # take over some very useful stuff from OGGM
    def copy_to_basedir(self, base_dir, setup='run'):
        return self.OGGMGD.copy_to_basedir(base_dir=base_dir, setup=setup)

    def get_filepath(self, filename, delete=False, filesuffix=''):
        return self.OGGMGD.get_filepath(filename=filename, delete=delete,
                                        filesuffix=filesuffix)

    def has_file(self, filename):
        return self.OGGMGD.has_file(filename=filename)

    def add_to_diagnostics(self, key, value):
        return self.OGGMGD.add_to_diagnostics(key, value)

    def get_diagnostics(self):
        return self.OGGMGD.get_diagnostics()

    def read_pickle(self, filename, use_compression=None, filesuffix=''):
        return self.OGGMGD.read_pickle(filename=filename,
                                       use_compression=use_compression,
                                       filesuffix=filesuffix)

    def write_pickle(self, var, filename, use_compression=None, filesuffix=''):
        return self.OGGMGD.write_pickle(var=var, filename=filename,
                                        use_compression=use_compression,
                                        filesuffix=filesuffix)

    def get_inversion_flowline_hw(self):
        return self.OGGMGD.get_inversion_flowline_hw()

    def log(self, task_name, err=None):
        return self.OGGMGD.log(task_name=task_name, err=err)

    def get_task_status(self, task_name):
        return self.OGGMGD.get_task_status(task_name=task_name)

    def write_monthly_climate_file(self, time, prcp, temp, grad, ref_pix_hgt,
                                   ref_pix_lon, ref_pix_lat,
                                   time_unit='days since 1801-01-01 00:00:00',
                                   file_name='climate_monthly',
                                   filesuffix=''):
        return self.OGGMGD.write_monthly_climate_file(time=time, prcp=prcp,
                                                      temp=temp, grad=grad,
                                                      ref_pix_hgt=ref_pix_hgt,
                                                      ref_pix_lon=ref_pix_lon,
                                                      ref_pix_lat=ref_pix_lat,
                                                      time_unit=time_unit,
                                                      file_name=file_name,
                                                      filesuffix=filesuffix)

    def create_gridded_ncdf_file(self, filename):
        """
        Makes a gridded netcdf file template with time axis.

        The difference to the method in OGGM is that we introduce a time axis
        in order to be able to supply time series of the gridded parameters.
        The other variables have to be created and filled by the calling
        routine.

        Parameters
        ----------
        filename : str
            file name (must be listed in cfg.BASENAMES)
        Returns
        -------
        a ``netCDF4.Dataset`` object.
        """

        # overwrite as default
        fpath = self.get_filepath(filename)
        if os.path.exists(fpath):
            os.remove(fpath)

        nc = netCDF4.Dataset(fpath, 'w', format='NETCDF4')

        xd = nc.createDimension('x', self.grid.nx)
        yd = nc.createDimension('y', self.grid.ny)
        time = nc.createDimension('time', None)

        nc.author = 'OGGM and CRAMPON'
        nc.author_info = 'Open Global Glacier Model and Cryospheric ' \
                         'Monitoring and Prediction Online'
        nc.proj_srs = self.grid.proj.srs

        lon, lat = self.grid.ll_coordinates
        x = self.grid.x0 + np.arange(self.grid.nx) * self.grid.dx
        y = self.grid.y0 + np.arange(self.grid.ny) * self.grid.dy

        v = nc.createVariable('x', 'f4', ('x',), zlib=True)
        v.units = 'm'
        v.long_name = 'x coordinate of projection'
        v.standard_name = 'projection_x_coordinate'
        v[:] = x

        v = nc.createVariable('y', 'f4', ('y',), zlib=True)
        v.units = 'm'
        v.long_name = 'y coordinate of projection'
        v.standard_name = 'projection_y_coordinate'
        v[:] = y

        v = nc.createVariable('longitude', 'f4', ('y', 'x'), zlib=True)
        v.units = 'degrees_east'
        v.long_name = 'longitude coordinate'
        v.standard_name = 'longitude'
        v[:] = lon

        v = nc.createVariable('latitude', 'f4', ('y', 'x'), zlib=True)
        v.units = 'degrees_north'
        v.long_name = 'latitude coordinate'
        v.standard_name = 'latitude'
        v[:] = lat

        v = nc.createVariable('time', 'f4', ('time',), zlib=True)
        v.units = 'days since 1961-01-01'
        v.long_name = 'time'
        v.standard_name = 'time'

        return nc



if __name__ == '__main__':
    #rgigdf = gpd.read_file('C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp')
    #rgigdf = rgigdf.ix[0:1]
    #abc = dem_differencing_results(rgigdf,
    #                               r'\\speedy10.wsl.ch\data_15\_PROJEKTE\Swiss_Glacier\TIFF')
    #make_swisstopo_worksheet(
    #    'C:\\Users\\Johannes\\Documents\\crampon\\data\\DEM\\DHM25L1\\',
    #    in_epsg='21781', out_epsg='4326')
    from crampon import workflow
    glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
    rgidf = gpd.read_file(glaciers)
    rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])]  # Gries
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B3601-6'])] # Oberaletsch
    #oa_ch = gpd.read_file('c:\\users\\johannes\\desktop\\oa_conyexhull.shp')
    #rgidf.geometry.values[0] = oa_ch.geometry.values[0]
    #rgidf.plot()
    #rgidf.crs = {'init': 'epsg:21781'}
    #rgidf = rgidf.to_crs(epsg=4326)
    cfg.initialize(file='C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\'
                        'CH_params.cfg')
    g = workflow.init_glacier_regions(rgidf, reset=False, force=False)
    abc = get_local_dems(g[0])
