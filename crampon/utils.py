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
import datetime
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
    date: type allowing comparison, subtraction and abs, e.g. datetime.datetime
        The date to which the closest partner shall be found.
    candidates: list of same input types as for date
        A list of candidates.

    Returns
    -------
    closest: Same type as input date
        The found closest date.

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
    begin: str
        Begin date of the most recent/current flexible year.

    Examples
    --------
    Find the beginning of the current mass budget year since
    2018-01-24.

    >>> get_begin_last_flexyear(datetime.datetime(2018,1,24))
    datetime.datetime(2017, 10, 1, 0, 0)
    >>> get_begin_last_flexyear(datetime.datetime(2017,11,30),
    >>> start_month=9, start_day=15)
    datetime.datetime(2017, 9, 15, 0, 0)
    """

    start_year = date.year if datetime.datetime(
        date.year, start_month, start_day) <= date else date.year - 1
    last_begin = datetime.datetime(start_year, start_month, start_day)

    return last_begin


def merge_rasters_rasterio(to_merge, outpath=None, outformat="Gtiff"):
    """
    Merges rasters to a single one using rasterio.

    Parameters
    ----------
    to_merge: list or str
        List of paths to the rasters to be merged.
    outpath: str, optional
        Path where to write the merged raster.
    outformat: str, optional
        Any format rasterio/GDAL has a driver for. Default: GeoTiff ('Gtiff').

    Returns
    -------
    merged, profile: tuple of (numpy.ndarray, rasterio.Profile)
        The merged raster and numpy array and its rasterio profile.
    """
    to_merge = [rasterio.open(s) for s in to_merge]
    merged, output_transform = merge_tool(to_merge)

    profile = to_merge[0].profile
    if 'affine' in profile:
        profile.pop('affine')
    profile['transform'] = output_transform
    profile['height'] = merged.shape[1]
    profile['width'] = merged.shape[2]
    profile['driver'] = outformat
    if outpath:
        with rasterio.open(outpath, 'w', **profile) as dst:
            dst.write(merged)
        for rf in to_merge:
            rf.close()

    return merged, profile


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
            #surplus = [p for p in available if all(x not in p for x in
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
        op = read_netcdf(op_path, chunks={'time': 50},
                          tfunc=_cut_with_CH_glac)

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


def daily_climate_from_netcdf(tfile, pfile, hfile, outfile):
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
    pfile: list
        Path to precipitation netCDF file.
    hfile: str
        Path to the elevation netCDF file.
    outfile: str
        Path to and name of the written file.

    Returns
    -------

    """

    temp = read_netcdf(tfile, chunks={'time': 50})
    prec = read_netcdf(pfile, chunks={'time': 50})
    hgt = read_netcdf(hfile)
    _, hgt = xr.align(temp, hgt, join='left')

    # Rename variables as OGGM likes it
    if 'TabsD' in temp.variables:
        temp.rename({'TabsD': 'temp'}, inplace=True)
    if 'RD' in prec.variables:
        prec.rename({'RD': 'prcp'}, inplace=True)

    # make it one
    nc_ts = xr.merge([temp, prec, hgt])

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

    combined = xr.concat(datasets, dim)
    return combined

# we cannot write it to cache until we have e.g. a date in the climate_all filename
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
        deliver strange values.
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


def attach_ginzler_zone(gdf, worksheet):
    """
    Returns the zone numbers of C. Ginzler's zone system for DEM tiles.

    Parameters
    ----------
    gdf: geopandas.GeoDataFrame
        Dataframe containing the polygons that request a zone intersection.

    worksheet: str
        Path to Christian Ginzler's worksheet with the zone numbers.

    Returns
    -------
    The GeoDataFrame with column 'zone' listing the intersecting zone numbers.
    """

    w_gdf = gpd.read_file(worksheet)

    for k, row in gdf.iterrows():
        res_is = gpd.overlay(row, w_gdf, how='intersection')
        gdf['zone'] = res_is.CLNR.values

    return gdf


def dem_differencing_results(shapedf, dem_dir):

    # get the network drive shit correctly
    os.system(r'net use \\speedy10.wsl.ch\Data_14\_PROJEKTE\Swiss_Glacier /user:wsl\ ')
    dem_list = glob.glob(os.path.join(dem_dir, '*.tif'))
    broke_month = ['ADS_103221_672000_165000_2011_0_0_86_50.tif']
    dem_list = [x for x in dem_list if not broke_month in x]

    # make an empty pd.DataFrame for each glacier ID with columns for every
    # possible subtraction time span
    dates = [datetime.datetime(int(os.path.basename(x).split('_')[4]), int(os.path.basename(x).split('_')[5]),
                               int(os.path.basename(x).split('_')[6])) for x in dem_list]
    dates_unique = np.unique(dates)
    years_unique = sorted(np.unique([x.year for x in dates_unique]), reverse=True)
    combos = []
    for i, y in enumerate(years_unique):
        try:
            for j in range(3, len(years_unique)):
                combos.append((y, years_unique[i + j]))
        except IndexError:
            continue
    deltacols = ['deltah_{}_{}'.format(i,j) for i,j in combos]

    res_df = pd.DataFrame(index=shapedf.RGIId.values,
                          columns=years_unique+deltacols)
    res_df.fillna()

    ws = r'\\speedy10.wsl.ch\Data_14\_PROJEKTE\Swiss_Glacier\GIS\Worksheet.shp'
    shapedf = attach_ginzler_zone(shapedf, ws)
    # ineffective, but better than storing everything in the cache
    for _, row in shapedf:
        cdems = []
        for zone in row.zone:
            cdems.extend([x for x in dem_list if 'ADS_{}'.format(zone) in x])

        # check the common dates (all!) for all zones
        cdates = [datetime.datetime(int(os.path.basename(x).split('_')[4]),
                                    int(os.path.basename(x).split('_')[5]),
                                    int(os.path.basename(x).split('_')[6])) for
                  x in cdems]
        for y in [d.year for d in cdates]:
            same_year = [d for d in cdates if y == d.year]
            if len(np.unique(same_year)) > 1:
                cdates = [date for date in cdates if date.year != y]
                cdems = [dem for dem in cdems if not '_{}_'.format(y) in dem]

        riodems = dict()
        for cd in cdates:
            to_merge = [rasterio.open(s) for s in [c for c in cdems if '_{}_'
                                     .format(cd.year) in c]]
            merged, _ = merge_tool(to_merge)

            # apply a glacier mask
            out_image, out_transform = rasterio.mask.mask(merged, row.geometry,
                                                            crop=True)
            out_meta = merged.meta.copy()

            isfinite = np.isfinite(out_image)
            # check the number of NaNs on the glacier area
            if np.sum(~isfinite) > (0.2 * out_image.shape[0] *
                                        out_image.shape[1]):
                log.info('DEM at {} was skipped due to missing values'.format(cd))
                cdates.remove(cd)  # to be consistent
            else:
                riodems[cd] = merged

        assert len([d.year for d in cdates]) == \
               len(np.unique([d.year for d in cdates]))

        # now for the subtraction and insertion
        for i, d in enumerate(sorted(cdates, reverse=True)):
            try:
                for j in range(3, len(cdates)):
                    dem_diff = np.subtract(riodems[d], riodems[sorted(cdates, reverse=True)[i + j]])
                    mean_diff = np.nanmean(dem_diff)
                    res_df.loc[res_df.RGIId==row.RGIId, d.year] = d
                    res_df.loc[res_df.RGIId == row.RGIId, sorted(cdates, reverse=True)[i + j].year] = sorted(cdates, reverse=True)[i + j]
                    res_df.loc[res_df.RGIId == row.RGIId, 'deltah_{}_{}'.format(d.year,sorted(cdates, reverse=True)[i + j].year)] = mean_diff

            except IndexError:
                continue


    #       for each combination(?) in the list:
    #            subtract the DEMs (new-old) and take the mean

    return res_df
