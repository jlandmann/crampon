"""
A collection of some useful miscellaneous functions.
"""
from __future__ import absolute_import, division

import fnmatch
import ftplib
import logging
import posixpath
import shutil
import subprocess
import sys
import tarfile
import netCDF4
from oggm.utils import GlacierDirectory, global_task, SuperclassMeta, mkdir, \
    filter_rgi_name, parse_rgi_meta, tolist, pipe_log, get_demo_file, \
    nicenumber, haversine, entity_task, get_topo_file
import time
import zipfile
from functools import wraps
from itertools import product
# Locals
from pathlib import Path
from typing import Union, Iterable

import geopandas as gpd
import pandas as pd
import paramiko as pm
import salem
import shapely
import shapely.geometry as shpg
import xarray as xr
from configobj import ConfigObj, ConfigObjError
from joblib import Memory
from oggm.utils._workflow import _timeout_handler
from salem import lazy_property, read_shapefile
from scipy import stats, special

from crampon.core.holfuytools import *


# I should introduce/alter:
"""get_crampon_demo_file, in general: get_files"""

# Module logger
log = logging.getLogger(__name__)
# Stop paramiko from logging successes to the console


# Joblib
MEMORY = Memory(cfg.CACHE_DIR, verbose=0)
SAMPLE_DATA_GH_REPO = 'crampon-sample-data'


dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30,
                               31],
       'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}


def get_oggm_demo_file(fname: str):
    """ Wraps the oggm.utils.get_demo_file function"""
    get_demo_file(fname)  # Calls the func imported from oggm.utils


def get_crampon_demo_file(fname: str):
    """This should be done once some test data are allowed to be moved to an 
    external repo"""
    raise NotImplementedError


def retry(exceptions: Union[str, Iterable, tuple, Exception], tries=100,
          delay=60, backoff=1, log_to=None):
    """
    Retry decorator calling the decorated function with an exponential backoff.

    Amended from Python wiki [1]_ and calazan.com [2]_.

    Parameters
    ----------
    exceptions: str or tuple or Exception
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
    .. [1] https://shorturl.at/mqsHX
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
    .. [1] https://shorturl.at/rtwA2
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


def leap_year(year: int, calendar: str = 'standard') -> bool:
    """
    Determine if year is a leap year.
    Amended from https://docs.xarray.dev/en/stable/examples/monthly-means.html

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


def get_dpm(time: pd.DatetimeIndex or xr.CFTimeIndex,
            calendar: str = 'standard') -> np.array:
    """
    Return array of days per month corresponding to months provided in `time`.

    From an old version of the xarray documentation (link was [1]_).

    time: pd.DatetimeIndex, xr.CFTimeIndex
        A time index with attributes 'year' and 'month'.
    calendar: str
        Calendar for which the days per month shall be returned: Allowed:
        'standard', 'gregorian', 'proleptic_gregorian', 'julian'. Default:
        'standard'.

    Returns
    -------
    month_length: np.array
        Array with month lengths in days.

    References
    ----------
    [1]_ : https://bit.ly/3sFrsFT
    """
    month_length = np.zeros(len(time), dtype=np.int)

    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(time.month, time.year)):
        month_length[i] = cal_days[month]
        if leap_year(year, calendar=calendar) and (month == 2):
            month_length[i] += 1
    return month_length


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
    .. [1] https://bit.ly/2M2Mr4K
    """
    return min(candidates, key=lambda x: abs(x - date))


def justify(arr, invalid_val=0, axis=1, side='left'):
    """
    Justify a 2D array.

    This actually means that the invalid_val is trimmed at the desired side and
    padded at the other side so that the shape of the array is kept.
    Adapted and slightly modified in terms of variable names and PEP8 from
    stackoverflow [1]_.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array to be justified
    invalid_val: float, None
        Invalid value to be trimmed.
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for
        axis=0.

    Examples
    --------
    >>> test = np.array([[np.nan, np.nan, np.nan, 3., 0., np.nan],
    >>>                  [np.nan, np.nan, 0., 6., np.nan, np.nan]])
    >>> justify(test, invalid_val=np.nan, side='right')
    array([[nan, nan, nan, nan,  3.,  0.],
       [nan, nan, nan, nan,  0.,  6.]])

    References
    ----------
    .. [1] https://bit.ly/2LMzUT8
    """

    if invalid_val is np.nan:
        mask = ~np.isnan(arr)
    elif (invalid_val is not np.nan) and (invalid_val is not None):
        mask = arr != invalid_val
    else:
        mask = ~pd.isnull(arr)
    justified_mask = np.sort(mask, axis=axis)
    if (side == 'up') | (side == 'left'):
        justified_mask = np.flip(justified_mask, axis=axis)
    out = np.full(arr.shape, invalid_val, dtype=arr.dtype)
    if axis == 1:
        out[justified_mask] = arr[mask]
    else:
        out.T[justified_mask.T] = arr.T[mask.T]
    return out


def get_begin_last_flexyear(date, start_month=None, start_day=None):
    """
    Get the begin date of the most recent/current ("last") flexible
    year since the given date.

    Parameters
    ----------
    date: datetime.datetime or pd.Timestamp
        Date from which on the most recent begin of the hydrological
        year shall be determined, e.g. today's date.
    start_month: int or None, optional
        Begin month of the flexible year. Default: 10 (for hydrological
        year). Default: None (parse from params.cfg)
    start_day: int or None, optional
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
    >>> get_begin_last_flexyear(dt.datetime(2018, 1, 24))
    dt.datetime(2017, 10, 1, 0, 0)
    >>> get_begin_last_flexyear(dt.datetime(2017, 11, 30),
    >>> start_month=9, start_day=15)
    dt.datetime(2017, 9, 15, 0, 0)
    """

    if start_month is None:
        start_month = cfg.PARAMS['begin_mbyear_month']
    if start_day is None:
        start_day = cfg.PARAMS['begin_mbyear_day']

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


def reclassify_heights_widths(gdir, elevation_bin=10., horizontal_dx=None):
    """
    Reclassify the flowline heights and widths of a glacier.

    Parameters
    ----------
    gdir : `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to reclassify the height and widths for.
    elevation_bin : float, optional
        Size of the elevation bins in meters to use for reclassification.
        Mutually exclusive with `horizontal_dx`. Default: 10.
    horizontal_dx : float, optional
        Horizontal distance in meters to use for reclassification.
        Mutually exclusive with `elevation_bin`. Default: None.

    Returns
    -------
    new_heights, new_widths: tuple of np.array
        Reclassified heights and widths.
    """

    heights, widths = gdir.get_inversion_flowline_hw()

    height_classes = np.arange(nicenumber(min(heights), 10, lower=True),
                               nicenumber(max(heights), 10, lower=True),
                               elevation_bin)
    if (elevation_bin is not None) and (horizontal_dx is None):
        height_bins = np.digitize(heights, height_classes)
        new_widths = [sum(widths[np.where(height_bins == i)]) for i in
                      range(1, max(height_bins + 1))]
        return height_classes, new_widths
    elif (elevation_bin is None) and (horizontal_dx is not None):
        fls = g.read_pickle('inversion_flowlines')
        new_heights = []
        new_widths = []
        for fl in fls:
            real_distances = fl.dis_on_line * g.grid.dx
            dx_classes = np.arange(0., nicenumber(np.max(real_distances),
                                                  horizontal_dx),
                                   horizontal_dx)
            dx_bins = np.digitize(real_distances, dx_classes)
            new_heights.extend(
                [np.mean(heights[np.where(dx_bins == i)]) for i in
                 range(1, max(dx_bins + 1))])
            new_widths.extend([np.sum(widths[np.where(dx_bins == i)]) for i in
                               range(1, max(dx_bins + 1))])

        return new_heights, new_widths
    else:
        raise NotImplementedError('A common usage of the elevation_bin and '
                                  'map_dx keywords together is not yet '
                                  'implemented.')


def parse_credentials_file(credfile=None):
    if credfile is None:
        credfile = os.path.join(os.path.abspath(os.path.dirname(
            os.path.dirname(__file__))), '.credentials')

    try:
        cr = ConfigObj(credfile, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Credentials file could not be parsed (%s): %s',
                     credfile, e)
        sys.exit()

    return cr


def unzip_file(zfile):
    with open(zfile, 'rb') as f:
        zipped = zipfile.ZipFile(f)
        zipped.extractall(os.path.dirname(zfile))
        zipped.close()


def untargz_file(targzfile):
    tar = tarfile.open(targzfile, "r:gz")
    tar.extractall(os.path.dirname(targzfile))
    tar.close()


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

        self.cr = parse_credentials_file(credfile)

        try:
            self.client = self.create_connect(self.cr['cirrus']['host'],
                                              self.cr['cirrus']['user'],
                                              self.cr['cirrus']['password'])
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


class WSLSFTPClient:
    """
    Class for interacting with the FTP Server at WSL.

    The WSL FTP server does not support SSH, so we need to use ftplib (paramiko
    tries to open an SSH connection).
    """

    def __init__(self, credfile=None, user='map'):
        """
        Initialize.

        Parameters
        ----------
        credfile: str
            Path to the credentials file (must be parsable as
            configobj.ConfigObj).
        user: str
            User to log in with. 'map' is for daily analysis grids and COSMO-7,
            'hyv-data' is for COSMO-1 and COSMO-E. Default: 'map'.
        """
        self.ftp_open = False

        self.cr = parse_credentials_file(credfile)

        if user == 'map':
            self.client = self.create_connect(self.cr['wsl-map']['host'],
                                              self.cr['wsl-map']['user'],
                                              self.cr['wsl-map']['password'])
        elif user == 'hyv-data':
            self.client = self.create_connect(self.cr['wsl-hyv']['host'],
                                              self.cr['wsl-hyv']['user'],
                                              self.cr['wsl-hyv']['password'])
        else:
            raise ValueError(
                'No credentials known for WSL FTP user {}'.format(user))

    def create_connect(self, host, username, password):
        client = ftplib.FTP(host=host, user=username, passwd=password)
        client.login(user=username, passwd=password)
        self.ftp_open = True
        return client

    def list_content(self, idir='.'):
        """

        Parameters
        ----------
        idir: str
            Directory whose output shall be listed

        Returns
        -------

        """

        return self.client.nlst(idir)

    def cwd(self, idir):
        """
        Change the working directory on the FTP server.

        Parameters
        ----------
        idir: str
            Path so set as workding directory.

        Returns
        -------
        None
        """
        self.client.cwd(idir)

    def get_file(self, remote_file, local_path):
        """
        Retrieve a file from the FTP server.

        Parameters
        ----------
        remote_file: str
            Remote file name.
        local_path: str
            Path to the local file (where file should be stored).

        Returns
        -------
        None
        """
        with open(local_path, 'wb') as f:
            self.client.retrbinary('RETR ' + remote_file, f.write)

    def close(self):
        """
        Close the connection.

        Returns
        -------
        None.
        """
        self.client.quit()


def copy_to_webpage_dir(src_dir: str, glob_pattern='*') -> None:
    """
    Copy folder content to the CRAMPON webpage directory on the web server.

    Uses SFTP as protocol.

    Parameters
    ----------
    src_dir : str
        Source directory that shall be copied.
    glob_pattern: str
        Addition restriction on what shall be selected. Default: '*' (no
        restriction).

    Returns
    -------
    None.
    """

    to_put = glob.glob(os.path.join(src_dir,  glob_pattern))
    to_put = [i for i in to_put if os.path.isfile(i)]
    client = pm.SSHClient()
    client.load_system_host_keys()
    client.set_missing_host_key_policy(pm.AutoAddPolicy())
    cred = parse_credentials_file()
    # first connect to login.ee.ethz.ch, then ssh to web server
    client.connect(cred['webpage_server']['login_host'],
                   cred['webpage_server']['port'],
                   cred['webpage_server']['user'],
                   cred['webpage_server']['password'])
    _ = client.exec_command(cred['webpage_server']['remote_cmd'])

    # now we're on webpage server
    sftp = client.open_sftp()
    for tp in to_put:
        common = os.path.commonpath([src_dir, tp])
        srv_path = './public_html/' + tp[len(common):].replace('\\', '/')
        sftp.put(tp, srv_path)
    sftp.close()
    client.close()


@xr.register_dataset_accessor('crampon')
class MeteoTSAccessor(object):
    def __init__(self, xarray_obj):
        """
        Class for handling Meteo time series, building upon xarray.
        """
        self._obj = xarray_obj
        self.grid = salem.Grid.from_json(os.path.join(cfg.PATHS['climate_dir'],
                                                      'meteoch_grid_ch.json'))
        # take random file as template
        self.roi = xr.open_dataset(glob.glob(
            os.path.join(cfg.PATHS['climate_dir'], '**', 'verified', '**',
                         'TabsD*20180131000000.nc'), recursive=True)[0])

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

        # index must be monotonic and should not have gaps
        dti = pd.DatetimeIndex(self._obj.time.values)
        if not dti.is_monotonic:
            self._obj = self._obj.sortby('time')
        if dti.inferred_freq is None:
            # .mean() = fill with NaN
            self._obj = self._obj.resample(
                time=freq, keep_attrs=True, **kwargs).mean()

        if "R" in self._obj.variables:  # fill gaps with zero
            self._obj.fillna(0.)
        else:  # interpolate linearly # todo also for sunshine?
            self._obj = self._obj.interpolate_na('time')
        diff_a = len(set(self._obj.time.values) - set(dti.values))
        diff_r = len(set(dti.values) - set(self._obj.time.values))

        log.info('{} time steps were added, {} removed during resampling.'
                 .format(diff_a, diff_r))

        return self._obj

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
            self._obj = self._obj.rename({"REFERENCE_TS": "time"})

        # whatever coordinate that is
        if "crs" in self._obj.data_vars:
            self._obj = self._obj.drop_vars(['crs'])

        # whatever coordinate that is
        if 'dummy' in self._obj.coords:
            self._obj = self._obj.drop(['dummy'])

        # whatever coordinate that is
        if 'latitude_longitude' in self._obj.coords:
            self._obj = self._obj.drop_sel(['latitude_longitude'])
        if 'latitude_longitude' in self._obj.variables:
            self._obj = self._obj.drop_vars(['latitude_longitude'])

        if 'longitude_latitude' in self._obj.coords:
            self._obj = self._obj.drop_sel(['longitude_latitude'])
        if 'longitude_latitude' in self._obj.variables:
            self._obj = self._obj.drop_vars(['longitude_latitude'])

        if 'geographic_projection' in self._obj.coords:
            self._obj = self._obj.drop_sel(['geographic_projection'])
        if 'geographic_projection' in self._obj.variables:
            self._obj = self._obj.drop_vars(['geographic_projection'])

        # this is the case for the operational files
        if 'x' in self._obj.coords:
            self._obj = self._obj.rename({'x': 'lon'})

        # this is the case for the operational files
        if 'y' in self._obj.coords:
            self._obj = self._obj.rename({'y': 'lat'})

        # Latitude can be switched after 2014
        self._obj = self._obj.sortby('lat')

        # make R variable names the same so that we don't get in troubles
        if 'RprelimD' in self._obj.variables:
            self._obj = self._obj.rename({'RprelimD': 'RD'})
        if 'RhiresD' in self._obj.variables:
            self._obj = self._obj.rename({'RhiresD': 'RD'})

        # radiation: "SIS" & "msg.SIS.D" in one file
        if ('msg.SIS.D' in self._obj.variables) and \
                ('SIS' in self._obj.variables):
            self._obj = self._obj['msg.SIS.D'].combine_first(
                self._obj.SIS).to_dataset(
                name='SIS')
        elif ('msg.SIS.D' in self._obj.variables) and not \
                ('SIS' in self._obj.variables):
            self._obj = self._obj.rename({'msg.SIS.D': 'SIS'})

        # R contains really low values (partly smaller than 0.0001mm)
        # Christoph Frei says everything smaller than 0.1 can be cut off
        if 'RD' in self._obj.variables:
            self._obj = self._obj.where(self._obj.RD >= 0.1, 0.)

        if (self._obj.time >= pd.Timestamp('2021-12-01')).any() and \
                ('TabsD' in self._obj.data_vars or 'RD' in self._obj.data_vars):
            self._obj = self._obj.salem.subset(ds=self.roi)

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
            pass  # already cut to CH shape
        try:
            self._obj.coords['lon'] = lons
        except ValueError:
            pass  # already cut to CH shape

        return self._obj


def climate_files_from_netcdf(tfile, pfile, hfile, outfile, tminfile=None,
                              tmaxfile=None, rfile=None):
    """
    Create a netCDF file with daily temperature, precipitation and
    elevation reference from given files.

    There will be at least one file created: the OGGM 'climate_monthly' file,
    containing the variables 'temp' and 'prcp'. If the given name for outfile
    is not the path to the monthly climate file, then additionally outfile is
    generated, containg also all other variables. The monthly climate is in any
    case calculated from a monthly resampling of the daily values.
    The temporal extent of the file will be the inner or outer join of the time
    series extent of the given input files .

    Parameters
    ----------
    tfile: str
        Path to mean temperature netCDF file.
    pfile: str
        Path to precipitation netCDF file.
    hfile: str
        Path to the elevation netCDF file.
    outfile: str
        Path to and name of the written file.
    tminfile: str, optional
        Path to minimum temperature netCDF file.
    tmaxfile: str, optional
        Path to maximum temperature netCDF file.
    rfile: str, optional
        Path to radiation netCDF file.

    Returns
    -------

    """

    temp = read_netcdf(tfile, chunks={'time': 50})
    tmin = read_netcdf(tminfile, chunks={'time': 50})
    tmax = read_netcdf(tmaxfile, chunks={'time': 50})
    prec = read_netcdf(pfile, chunks={'time': 50})
    sis = read_netcdf(rfile, chunks={'time': 50})  # shortwave incoming solar
    hgt = read_netcdf(hfile)
    _, hgt = xr.align(temp, hgt, join='left')

    # Rename variables as OGGM likes it
    if 'TabsD' in temp.variables:
        temp = temp.rename({'TabsD': 'temp'})
    if 'TminD' in tmin.variables:
        tmin = tmin.rename({'TminD': 'tmin'})
    if 'TmaxD' in tmax.variables:
        tmax = tmax.rename({'TmaxD': 'tmax'})
    if 'RD' in prec.variables:
        prec = prec.rename({'RD': 'prcp'})
    if 'SIS' in sis.variables:
        sis = sis.rename({'SIS': 'sis'})

    # make it one
    nc_ts = xr.merge([temp, tmin, tmax, prec, sis, hgt])

    # Units cannot be checked anymore at this place (lost in xarray...)

    # it can be that one retrieval was not successful (file not on FTP yet)
    # todo: include 'sis' once it is delivered operationally
    for var in ['temp', 'tmin', 'tmax', 'prcp']:
        if np.isnan(nc_ts[var].isel(time=-1)).all():
            raise FileNotFoundError("Variable '{}' not (yet) on server for "
                                    "{}.".format(var, nc_ts.time[-1]))

    # ensure it's compressed when exporting
    nc_ts.encoding['zlib'] = True
    nc_ts.to_netcdf(outfile)

    # to be sure, produce the OGGM climate file anyway
    if 'climate_monthly' not in outfile:
        outfile_oggm = os.path.join(os.path.dirname(outfile),
                                    'climate_monthly.nc')
        temp_monthly = temp.resample(time="MS").mean()
        prec_monthly = prec.resample(time="MS").sum()
        nc_ts_monthly = xr.merge([temp_monthly, prec_monthly])
        nc_ts_monthly.encoding['zlib'] = True
        nc_ts_monthly.to_netcdf(outfile_oggm)


def read_netcdf(path, chunks=None, tfunc=None, **kwargs):
    # use a context manager, to ensure the file gets closed after use
    with xr.open_dataset(path, cache=False, **kwargs) as ds:
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

    Changed from:  https://xarray.pydata.org/en/stable/io.html#id7

    Parameters
    ----------
    files: list
        List with paths to the files to be read.
    dim: str
        Dimension along which to concatenate the files.
    tfunc: callable
        Transformation function for the data, e.g. 'lambda ds: ds.mean()'
    chunks: dict
        Chunk sizes as can be specified to xarray.open_dataset.

    Returns
    -------
    A concatenation of the input files as xarray.Dataset.
    """

    paths = sorted(files)
    datasets = [read_netcdf(p, chunks, tfunc) for p in paths]

    # concat is much faster, so combine only where necessary
    try:
        combined = xr.concat(datasets, dim='time')
    except ValueError:
        # vars can vary over time (e.g. 'geographic_projection' in msgSISD)
        combined = xr.combine_by_coords(datasets, data_vars='minimal')
    return combined


# can't write it to cache until we have e.g. a date in the climate_all filename
@MEMORY.cache
def joblib_read_climate_crampon(nc_file, ilon, ilat, default_tgrad,
                                minmax_tgrad, use_tgrad, default_pgrad,
                                minmax_pgrad, use_pgrad):
    """
    This is a cracked version of the OGGM function with some extras.

    Parameters
    ----------
    nc_file: str or xr.Dataset
        Path to the netCDF file in OGGM suitable format, or Dataset itself.
    ilon: int
        Index of a longitude in the netCDF.
    ilat: int
        Index of a latitude in the netCDF.
    default_tgrad: float
        Default temperature gradient (K m-1).
    minmax_tgrad: tuple
        Min/Max bounds of the local temperature gradient, in case the grid
        kernel search delivers strange values.
    use_tgrad: int
        Window edge width of surrounding cells used to determine the local
        temperature gradient. Must be an odd number. If 0,
        the ``default_tgrad`` is used.
    default_pgrad: float
        Default precipitation gradient (m-1).
    minmax_pgrad: tuple
        Min/Max bounds of the local precipitation gradient, in case the grid
        kernel search delivers strange values.
    use_pgrad: int
        Window edge width of surrounding cells used to determine the local
        precipitation gradient. Must be an odd number. If 0,
        the ``default_pgrad`` is used.


    Returns
    -------
    iprcp, itemp, isis, itgrad, ipgrad, ihgt:
    Precipitation, temperature, temperature gradient, shortwave incoming solar
    radiation and elevation at given latitude/longitude indices.
    """

    # check for oddness or zero
    if not ((divmod(use_tgrad, 2)[1] == 1 or use_tgrad == 0) or
            (divmod(use_pgrad, 2)[1] == 1 or use_pgrad == 0)):
        raise ValueError('Window edge width must be odd number or zero.')

    # get climate at reference cell
    if isinstance(nc_file, str):
        climate = xr.open_dataset(nc_file)
    else:
        climate = nc_file.copy(deep=True)
    local_climate = climate.isel(dict(lat=ilat, lon=ilon))
    iprcp = local_climate.prcp
    itemp = local_climate.temp
    itmin = local_climate.tmin
    itmax = local_climate.tmax
    isis = local_climate.sis
    ihgt = local_climate.hgt

    # fill temp and radiation with mean, precip with 0.
    itemp.resample(time='D', keep_attrs=True).mean('time')
    itmin.resample(time='D', keep_attrs=True).mean('time')
    itmax.resample(time='D', keep_attrs=True).mean('time')
    isis.resample(time='D', keep_attrs=True).mean('time')
    iprcp.fillna(0.)

    # temperature gradient
    if use_tgrad != 0:
        itgrad, itgrad_unc, itgrad_loc, itgrad_scale = get_tgrad_from_window(climate, ilat, ilon,
                                                   use_tgrad, default_tgrad,
                                                   minmax_tgrad)
    else:
        # todo: come up with estimate for uncertainty in this case
        itgrad = np.zeros(len(climate.time)) + default_tgrad
        itgrad_unc = np.zeros(len(climate.time))
        itgrad_loc = np.full(len(climate.time), np.nan)
        itgrad_scale = np.full(len(climate.time), np.nan)

    # precipitation gradient
    if use_pgrad != 0:
        ipgrad, ipgrad_unc, ipgrad_loc, ipgrad_scale = (
            get_pgrad_from_window(climate, ilat, ilon, use_pgrad,
                                  default_pgrad, minmax_pgrad))
    else:
        ipgrad = np.zeros(len(climate.time)) + default_pgrad
        # todo: come up with estimate for uncertainty in this case
        ipgrad_unc = np.zeros(len(climate.time))
        ipgrad_loc = np.full(len(climate.time), np.nan)
        ipgrad_scale = np.zeros(len(climate.time), np.nan)

    return (iprcp, itemp, itmin, itmax, isis, itgrad, itgrad_unc, ipgrad,
            ipgrad_unc, ihgt, itgrad_loc, itgrad_scale, ipgrad_loc,
            ipgrad_scale)


# IMPORTANT: overwrite OGGM functions with same name
joblib_read_climate = joblib_read_climate_crampon


def _calc_slope(x, y):
    """wrapper that returns the slope from a linear regression fit of x & y"""
    slope = stats.linregress(x.flatten(), y.flatten())[0]  # extract slope only
    return slope


def _calc_pval(x, y):
    """wrapper returning the p-value from a linear regression fit of x & y"""
    pval = stats.linregress(x.flatten(), y.flatten())[3]  # extract slope only
    return pval


def _calc_stderr(x, y):
    """wrapper that returns the stderr from a linear regression fit of x & y"""
    stderr = stats.linregress(x.flatten(), y.flatten())[
        4]  # extract slope only
    return stderr


def bayes_slope_post(h, T, sigma_squared, b_0, g=1.):
    beta_hat = np.sum(h * T) / np.sum(h**2)
    mean = (g * beta_hat) / (1 + g) + b_0 / (1 + g)
    var = (g / (1 + g)) * (sigma_squared / np.sum(h**2))
    return mean, np.sqrt(var)


def bayes_slope_explicit(h, T, sigma_squared, b_0, g=1.):
    alpha_hat = np.mean(T)
    beta_hat = np.sum(h * T) / np.sum(h**2)
    s_t_squared = np.sum((T - alpha_hat - beta_hat * h)**2)

    return np.sqrt(sigma_squared)**(-T.size-2) * np.exp(-( ((1+g)/g) * np.sum(h ** 2 * (beta_t - (g/(1+g))* beta_hat - (1/(1+g)*b_0)) ** 2) + (1 / (1+g)) * np.sum(h ** 2 * (beta_hat + b_0) ** 2) + s_t_squared ) / (2 * sigma_squared))


def bayes_slope_from_t_dist(x, y, df, b_0, g=1.):
    """
    Get Bayesian slope mean and standard deviation using a g-prior of Zellner.

    Source: E-Mail Hansruedi Künsch 14.07.2021 (PDF page 2, approach 1).

    Parameters
    ----------
    x : np.array
        Independent variable, e.g. height.
    y : np.array
        Dependent variable, e.g. temperature or precipitation.
    df : int
        Degrees of freedom for the t-distribution form which is sampled. For
        25 grid cells this is 24.
    b_0 : float
        Prior mean (slope), e.g. -0.0065 K m-1 for temperature.
    g : float
        "g" of the g-prior of Zellner _[1], here we set: Default = 1.

    Returns
    -------
    slope_mean, slope_std, loc, scale
        Bayesian slope mean and standard deviation from a t-distribution
        sample, and the underyling `loc` and `scale` parameters from the t
        distribution.
    """

    alpha_hat = np.mean(y)
    beta_hat = np.sum(x * y) / np.sum(x ** 2)
    loc = (g * beta_hat + b_0) / (1 + g)
    s_t_squared = np.sum((y - alpha_hat - beta_hat * x) ** 2)
    c_squared = (g / (df * (1 + g) * np.sum(x ** 2))) * \
                (s_t_squared + 1 / (1 + g) *
                 np.sum(x ** 2 * (beta_hat - b_0) ** 2))
    scale = np.sqrt(c_squared)
    sample = stats.t.rvs(df=df, loc=loc, scale=scale, size=5000)

    # we sampled from a t-dist, but I am cheeky to make the big sample Normal
    slope_mean = np.mean(sample)
    slope_std = np.std(sample)
    return slope_mean, slope_std, loc, scale


def get_tgrad_from_window(climate, ilat, ilon, win_size, default_tgrad,
                          minmax_tgrad, use_kalmanfilter=False,
                          use_bayes_post=True):
    """
    Get temperature gradient from linear regression of surrounding grid cells.

    Parameters
    ----------
    climate: xr.Dataset
        Climate dataset, needs to contain variables/coordinates "time", "temp"
        and "hgt".
    win_size:
        Search window size in number of grid cells.
    ilat: int
        Index of latitude of the central grid cell.
    ilon: int
        Index of longitude of the central grid cell.
    default_tgrad: float
        Default temperature gradient to be used when regression fails (i.e. not
        significant).
    minmax_tgrad: tuple
        Min/Max bounds of the local temperature gradient, in case the window
        regression kernel search delivers strange values (should not happen for
        the MeteoSwiss grids, but who knows...).
    use_kalmanfilter: bool, optional
        Whether to use a Kalman Filter to calculate the gradient amd its
        uncertainty. This method uses a `gradient climatology` with its mean
        and standard deviation as background and the observed gradient and
        standard error of the slope as `observation`. Median and standard
        deviation of the posterior are used as gradient ("itgrad)" and its
        uncertainty ("itgrad_unc"), respectively. Mutually exclusive with
        `use_bayes_post`. Default: False.
    use_bayes_post: bool, optional
        Use a Bayesian posterior for the slope (see e-mail Hansruedi Künsch
         13.06.2021). Mutually exclusive with `use_kalmanfilter`. Default: True

    Returns
    -------
    itgrad: np.array
        Temperature gradient over time for the given ilat/ilon.
    """

    if use_bayes_post is True and use_kalmanfilter is True:
        raise ValueError('`use_kalmanfilter` is mutually exclusive with '
                         '`use_bayes_post`.')

    itgrad = np.full(len(climate.time), np.nan)
    itgrad_unc = np.full(len(climate.time), np.nan)
    itgrad_loc = np.full(len(climate.time), np.nan)
    itgrad_scale = np.full(len(climate.time), np.nan)

    # some min/max constants for the window
    tminw = divmod(win_size, 2)[0]
    tmaxw = divmod(win_size, 2)[0] + 1

    tlatslice = slice(ilat - tminw, ilat + tmaxw)
    tlonslice = slice(ilon - tminw, ilon + tmaxw)

    # time should be first (we iterate over ttemp.values)
    climate = climate.transpose('time', ...)

    ttemp = climate.temp.isel(dict(lat=tlatslice, lon=tlonslice))
    thgt = climate.hgt.isel(dict(lat=tlatslice, lon=tlonslice))

    # for predictions
    if 'member' in ttemp.dims:
        ttemp_stack = ttemp.stack(ens=['member', 'time'])
        slope = xr.apply_ufunc(_calc_slope, thgt, ttemp_stack, vectorize=True,
                              input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
                              output_core_dims=[[]], output_dtypes=[np.float],
                              dask='parallelized')
        slope = slope.unstack()
        pval = xr.apply_ufunc(_calc_pval, thgt, ttemp_stack, vectorize=True,
                               input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
                               output_core_dims=[[]], output_dtypes=[np.float],
                               dask='parallelized')
        pval = pval.unstack()
        stderr = xr.apply_ufunc(_calc_stderr, thgt, ttemp_stack, vectorize=True,
                               input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
                               output_core_dims=[[]], output_dtypes=[np.float],
                               dask='parallelized')
        stderr = stderr.unstack()

        if use_kalmanfilter is True or use_bayes_post is True:
            # todo implement Kalman filtering for meteorological predictions
            log.warning('Use of Kalman-filter/Bayes gradient not yet '
                        'implemented for meteorological predictions. '
                        'Continuing with standard method....')
        # replace correlations and stderrs with high pvals
        # todo: better use rolling window filling also here
        itgrad = slope.where(pval < 0.01,
                             cfg.PARAMS['temp_default_gradient'])
        itgrad_unc = stderr.where(pval < 0.01, np.nanmean(stderr))

    else:  # past meteorology

        thgt = thgt.values.flatten()

        # make a gradient climatology
        for t, loct in enumerate(ttemp.values):
            # NaNs happen a the grid edges:
            mask = ~np.isnan(loct)
            slope, intcpt, _, p_val, stderr = stats.linregress(
                np.ma.masked_array(thgt, ~mask).compressed(),
                loct[mask].flatten())
            # otherwise we drop the "bad" correlations already
            itgrad[t] = slope if (p_val < 0.01) else np.nan
            itgrad_unc[t] = stderr if (p_val < 0.01) else np.nan

        # fill not significant gradients with DOY rolling window mean
        # todo: this could also be not on DOYs
        grad_ds = xr.DataArray(dims={'time': climate.time.data},
                               data=itgrad.data,
                               coords={'time': ('time', climate.time.data)})
        grad_unc_ds = xr.DataArray(dims={'time': climate.time.data},
                                   data=itgrad_unc.data,
                                   coords={'time': ('time', climate.time.data)})

        grad_rmean = grad_ds.rolling(time=30, center=True,
                                     min_periods=5).mean(
            skipna=True).groupby('time.dayofyear').mean(skipna=True)
        grad_unc_rmean = grad_unc_ds.rolling(time=30, center=True,
            min_periods=5).mean(skipna=True).groupby(
            'time.dayofyear').mean(skipna=True)
        # fill NaNs - to be sure
        grad_rmean = grad_rmean.fillna(default_tgrad)
        # todo: define a mean tgrad uncertainty?
        grad_unc_rmean = grad_unc_rmean.fillna(np.nanmean(itgrad_unc))

        # start the actual calculation
        itgrad = np.full(len(climate.time), np.nan)
        itgrad_unc = np.full(len(climate.time), np.nan)
        itgrad_loc = np.full(len(climate.time), np.nan)
        itgrad_scale = np.full(len(climate.time), np.nan)
        for t, loct in enumerate(ttemp.values):
            # NaNs happen at the grid edges:
            mask = ~np.isnan(loct)
            slope, intcpt, _, p_val, stderr = stats.linregress(
                np.ma.masked_array(thgt, ~mask).compressed(),
                loct[mask].flatten())

            # for Kalman filtering we need them all
            if use_kalmanfilter is True:
                # we insert all slope and correct them later
                itgrad[t] = slope
                itgrad_unc[t] = stderr
            elif use_bayes_post is True:
                # we calculate is directly
                h = np.ma.masked_array(thgt, ~mask).compressed()
                h_diff = h - np.nanmean(h)

                doy = pd.Timestamp(ttemp.isel(time=t).time.values).dayofyear
                climate_tgrad_at_doy = grad_rmean.sel(dayofyear=doy).item()
                bayes_mean, bayes_std, loc, scale = bayes_slope_from_t_dist(
                    h_diff, loct[mask].flatten(), df=win_size**2-1,
                    b_0=climate_tgrad_at_doy)

                # we draw from a t-distribution, but assume Gaussianity
                itgrad[t] = bayes_mean
                itgrad_unc[t] = bayes_std
                itgrad_loc[t] = loc
                itgrad_scale[t] = scale
            else:
                # otherwise we drop the "bad" correlations already
                itgrad[t] = slope if (p_val < 0.01) else np.nan
                itgrad_unc[t] = stderr if (p_val < 0.01) else np.nan

            if itgrad_unc[t] <= 0.:
                print('stop')

        # fill not significant gradients with DOY rolling window mean
        # todo: this could also be not on DOYs
        grad_ds = xr.DataArray(dims={'time': climate.time.data},
                               data=itgrad.data,
                               coords={'time': ('time', climate.time.data)})
        grad_unc_ds = xr.DataArray(dims={'time': climate.time.data},
                                   data=itgrad_unc.data,
                                   coords={'time': ('time',
                                                    climate.time.data)})

        grad_rmean = grad_ds.rolling(time=30, center=True, min_periods=5).mean(
            skipna=True).groupby('time.dayofyear').mean(skipna=True)
        grad_unc_rmean = grad_unc_ds.rolling(
            time=30, center=True, min_periods=5).mean(skipna=True).groupby(
            'time.dayofyear').mean(skipna=True)

        # days with nan should be the same in both itgrad and itgrad_unc
        nan_doys = [pd.Timestamp(x).dayofyear for x in
                    climate.time[np.isnan(itgrad)].values]

        if use_kalmanfilter is False and use_bayes_post is False:
            # fill NaNs - to be sure
            grad_rmean = grad_rmean.fillna(default_tgrad)
            # todo: define a mean tgrad uncertainty?
            grad_unc_rmean = grad_unc_rmean.fillna(np.nanmean(itgrad_unc))
            # fill finally
            itgrad[np.isnan(itgrad)] = (
                grad_rmean.sel(dayofyear=nan_doys).values)
            itgrad_unc[np.isnan(itgrad_unc)] = \
                grad_unc_rmean.sel(dayofyear=nan_doys).values
        elif use_bayes_post is True:
            pass
        elif use_kalmanfilter is True:
            # here, we want the std of the climatology gradient
            grad_rstd = grad_ds.rolling(time=30, center=True,
                min_periods=5).mean(skipna=True).groupby('time.dayofyear').std(
                skipna=True)

            def gaussian_multiply(g1_mean, g1_var, g2_mean, g2_var):
                mean = (g1_var * g2_mean + g2_var * g1_mean) / (
                            g1_var + g2_var)
                variance = (g1_var * g2_var) / (g1_var + g2_var)
                return mean, variance

            def update(prior_mean, prior_var, likelihood_mean, likelihood_var):
                posterior = gaussian_multiply(likelihood_mean, likelihood_var,
                                              prior_mean, prior_var)
                return posterior

            itgrad, itgrad_var = update(
                grad_rmean[climate.time.dt.dayofyear.values-1],
                grad_rstd[climate.time.dt.dayofyear.values-1]**2,
                itgrad, itgrad_unc**2)
            itgrad_unc = np.sqrt(itgrad_var)

        # todo: should this really happen??????
        # apply the boundaries, in case the gradient goes wild
        itgrad = np.clip(itgrad, minmax_tgrad[0], minmax_tgrad[1])

        itgrad = xr.DataArray(itgrad, coords={'time': climate.coords['time']},
                     dims=('time',), name='tgrad')
        itgrad_unc = xr.DataArray(itgrad_unc,
                                  coords={'time': climate.coords['time']},
                                  dims=('time',), name='tgrad_unc')

        if use_bayes_post is True:
            itgrad_loc = xr.DataArray(itgrad_loc,
                                  coords={'time': climate.coords['time']},
                                  dims=('time',), name='tgrad_loc')
            itgrad_scale = xr.DataArray(itgrad_scale,
                                      coords={'time': climate.coords['time']},
                                      dims=('time',), name='tgrad_scale')
        else:
            itgrad_loc = None
            itgrad_scale = None

    return itgrad, itgrad_unc, itgrad_loc, itgrad_scale


def get_pgrad_from_window(climate, ilat, ilon, win_size, default_pgrad,
                          minmax_pgrad, use_kalmanfilter=False,
                          use_bayes_post=True):
    """
    Get precipitation gradient from linear regression of surrounding grid cells.

    # todo: this method should get an overhaul & analyse the orographic facets
            as suggested by Christoph Frei (pers. comm.)

    Parameters
    ----------
    climate: xr.Dataset
        Climate dataset, needs to contain variables/coordinates "time", "prcp"
        and "hgt".
    win_size:
        Search window size in number of grid cells.
    ilat: int
        Index of latitude of the central grid cell.
    ilon: int
        Index of longitude of the central grid cell.
    default_pgrad: float
        Default precipitation gradient to be used when regression fails (i.e.
        not significant).
    minmax_tgrad: tuple
        Min/Max bounds of the local temperature gradient, in case the window
        regression kernel search delivers strange values (should not happen for
        the MeteoSwiss grids, but who knows...).
    use_kalmanfilter: bool, optional
        Whether to use a Kalman Filter to calculate the gradient and its
        uncertainty. This method uses a `gradient climatology` with its mean
        and standard deviation as background and the observed gradient and
        standard error of the slope as `observation`. Median and standard
        deviation of the posterior are used as gradient ("ipgrad)" and its
        uncertainty ("itprad_unc"), respectively. Mutually exclusive with
        `use_bayes_post`. Default: False.
    use_bayes_post: bool, optional
        Use a Bayesian posterior for the slope (see e-mail Hansruedi Künsch
         13.06.2021). Mutually exclusive with `use_kalmanfilter`. Default: True


    Returns
    -------
    igrad: np.array
        Precipitation gradient over time for the given ilat/ilon.
    """

    if use_bayes_post is True and use_kalmanfilter is True:
        raise ValueError('`use_kalmanfilter` is mutually exclusive with '
                         '`use_bayes_post`.')

    ipgrad = np.full(len(climate.time), np.nan)
    ipgrad_unc = np.full(len(climate.time), np.nan)

    # some min/max constants for the window
    pminw = divmod(win_size, 2)[0]
    pmaxw = divmod(win_size, 2)[0] + 1

    platslice = slice(ilat - pminw, ilat + pmaxw)
    plonslice = slice(ilon - pminw, ilon + pmaxw)

    # time should be first (we iterate over pprcp.values)
    climate = climate.transpose('time', ...)

    pprcp = climate.prcp.isel(dict(lat=platslice, lon=plonslice))
    phgt = climate.hgt.isel(dict(lat=platslice, lon=plonslice))

    # todo: no kalman/bayes option for prediction implemented yet
    if 'member' in pprcp.dims:
        ipgrad_loc = np.full(len(climate.time), np.nan)
        ipgrad_scale = np.full(len(climate.time), np.nan)
        pprcp_stack = pprcp.stack(ens=['member', 'time'])
        slope = xr.apply_ufunc(_calc_slope, phgt, pprcp_stack, vectorize=True,
                              input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
                              output_core_dims=[[]], output_dtypes=[np.float],
                              dask='parallelized')
        slope = slope.unstack()
        pval = xr.apply_ufunc(_calc_pval, phgt, pprcp_stack, vectorize=True,
                               input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
                               output_core_dims=[[]], output_dtypes=[np.float],
                               dask='parallelized')
        pval = pval.unstack()
        stderr = xr.apply_ufunc(_calc_stderr, phgt, pprcp_stack, vectorize=True,
                               input_core_dims=[['lat', 'lon'], ['lat', 'lon']],
                               output_core_dims=[[]], output_dtypes=[np.float],
                               dask='parallelized')
        stderr = stderr.unstack()

        # replace correlations and stderrs with high pvals
        # todo: better filling with rolling means also here
        ipgrad = slope.where(pval < 0.01, cfg.PARAMS['prcp_default_gradient'])
        ipgrad_unc = stderr.where(pval < 0.01, np.nanmean(stderr))
    else:
        phgt = phgt.values.flatten()

        # make climatology
        for t, locp in enumerate(pprcp.values):
            # NaNs happen at grid edges, 0 should be excluded for slope
            mask = ~np.isnan(locp) & (locp != 0.)
            flattened_mask = locp[mask].flatten()
            if (~mask).all():
                continue

            slope, icpt, _, p_val, stderr = stats.linregress(
                np.ma.masked_array(phgt, ~mask).compressed(), flattened_mask)

            ipgrad[t] = np.nanmean(
                ((flattened_mask + slope) / flattened_mask) - 1) if (
                    (p_val < 0.01) and (p_val != 0.)) else default_pgrad
            # todo: I think this is bullshit: Gradient - Gradient derived with steeper slope
            ipgrad_unc[t] = stderr if ((p_val < 0.01) and
                                       (p_val != 0.)) else np.nan
                # (ipgrad[t] - np.nanmean(
                # ((flattened_mask + slope + stderr) / flattened_mask) - 1)) if (
                #        (p_val < 0.01) and (p_val != 0.)) else np.nan

        # make gradient climatology - here we select only positive
        # todo: fill prcp gradient with rolling window mean?
        ref_cell = pprcp.values[:, int(win_size / 2),
                   int(win_size / 2)]
        # prec_true = ref_cell > 0.

        # mean is influenced a lot by outliers!
        # ipgrad[np.isnan(ipgrad)] = np.nanmedian(ipgrad)  # & prec_true)] =
        # np.nanmedian(ipgrad)
        # we take std around the median
        # ipgrad_unc[np.isnan(ipgrad_unc)] = np.sqrt(  # & prec_true)] =
        # np.sqrt(np.nanmean((ipgrad_unc - np.nanmedian(ipgrad_unc)) ** 2))
        # apply the boundaries, in case the gradient goes wild
        # ipgrad = np.clip(ipgrad, minmax_pgrad[0], minmax_pgrad[1])

        # fill not significant gradients with DOY rolling window mean
        # todo: this could also be not on DOYs
        grad_ds = xr.DataArray(dims={'time': climate.time.data},
                               data=ipgrad.data,
                               coords={'time': ('time', climate.time.data)})
        grad_unc_ds = xr.DataArray(dims={'time': climate.time.data},
                                   data=ipgrad_unc.data,
                                   coords={'time': ('time',
                                                    climate.time.data)})

        grad_rmean = grad_ds.rolling(time=30, center=True,
                                     min_periods=5).mean(
            skipna=True).groupby('time.dayofyear').mean(skipna=True)
        grad_unc_rmean = grad_unc_ds.rolling(time=30, center=True,
                                             min_periods=5).mean(
            skipna=True).groupby('time.dayofyear').mean(skipna=True)
        # fill NaNs - to be sure
        grad_rmean = grad_rmean.fillna(default_pgrad)
        # todo: define a mean tgrad uncertainty?
        grad_unc_rmean = grad_unc_rmean.fillna(np.nanmean(ipgrad_unc))

        # make actual values
        ipgrad = np.full(len(climate.time), np.nan)
        ipgrad_unc = np.full(len(climate.time), np.nan)
        ipgrad_loc = np.full(len(climate.time), np.nan)
        ipgrad_scale = np.full(len(climate.time), np.nan)
        for t, locp in enumerate(pprcp.values):
            # NaNs happen at grid edges, 0 should be excluded for slope
            mask = ~np.isnan(locp) & (locp != 0.)
            flattened_mask = locp[mask].flatten()
            if (~mask).all():
                continue

            # Todo: Is that a good method?
            # gradient in % m-1: mean(all prcp values + slope for 1 m)
            # p=0. happens if there are only two grids cells
            if use_kalmanfilter is True:
                raise NotImplementedError(
                    'Use of Kalman-filtered precipitation gradients is not yet'
                    ' implemented.')
            elif use_bayes_post is True:
                # we calculate is directly

                mask = ~np.isnan(locp)
                flattened_mask = locp[mask].flatten()

                h = np.ma.masked_array(phgt, ~mask).compressed()
                h_diff = h - np.nanmean(h)
                center_pix = int(np.floor(win_size / 2.))

                doy = pd.Timestamp(pprcp.isel(time=t).time.values).dayofyear
                climate_pgrad_at_doy = grad_rmean.sel(dayofyear=doy).item()

                # if precipitation at reference cell
                if locp[center_pix, center_pix] > 0.:
                    flattened_mask_sqrt = np.sqrt(flattened_mask)

                    bayes_mean, bayes_std, bayes_loc, bayes_scale = (
                        bayes_slope_from_t_dist(h_diff, flattened_mask_sqrt,
                                                df=win_size**2-1,
                        b_0=climate_pgrad_at_doy))#b_0=default_pgrad)
                else:  # no precip at reference cell
                    # see e-mail Hansruedi 17.07.2021
                    alpha = 0.9  # just a guess factor
                    k = np.sum((locp == 0.)) - 1  # other cells w/o precip
                    # do not mask zeros this time
                    mask = ~np.isnan(locp)
                    # todo: sqrt here ok as well?
                    flattened_mask = np.sqrt(locp[mask].flatten())
                    # all non-zero values
                    nonzero = flattened_mask[flattened_mask != 0.]
                    nonzero_h_diff = h_diff.copy()

                    # update selection of h
                    h = np.ma.masked_array(phgt, ~mask).compressed()
                    h_diff = h - np.nanmean(h)

                    # prob. of zero precip. at reference
                    prob_zero_precip = alpha + (1-alpha) * k / \
                                       (flattened_mask.size - 1)
                    # prob of non-zero precip at ref. cell
                    prob_nonzero_precip = (1-alpha) * \
                                          ((flattened_mask.size - 1) - k) / \
                                          (flattened_mask.size - 1)


                    bayes_mean_zero, bayes_std_zero, bayes_loc_zero, bayes_scale_zero = bayes_slope_from_t_dist(
                        h_diff, flattened_mask, df=win_size**2-1,
                        b_0=climate_pgrad_at_doy) # b_0=default_pgrad)

                    # 10000 random samples
                    cases = np.random.choice([0, 1], size=10000,
                                             p=[prob_zero_precip,
                                                prob_nonzero_precip])

                    bayes_mean_list = []
                    bayes_std_list = []
                    bayes_loc_list = []
                    bayes_scale_list = []

                    # for all non-zero cases we randomly assign the values of
                    # another non-zero cell value to the reference cell
                    # worst case no 1 drawn
                    for i in range(max(np.sum(cases), 1)):

                        # todo: sqrt okay here as well?
                        replaced = np.sqrt(locp.copy())
                        replaced[center_pix, center_pix] = \
                            np.random.choice(nonzero, size=1)[0]
                        replaced = replaced.flatten()

                        temp_mean, temp_std, temp_loc, temp_scale = (
                            bayes_slope_from_t_dist(
                        h_diff, replaced, df=win_size**2-1,
                        b_0=climate_pgrad_at_doy))  # b_0=default_pgrad)
                        bayes_mean_list.append(temp_mean)
                        bayes_std_list.append(temp_std)
                        bayes_loc_list.append(temp_loc)
                        bayes_scale_list.append(temp_scale)

                    bayes_mean = np.average(
                        [np.mean(np.array(bayes_mean_list)), bayes_mean_zero],
                        weights=[prob_nonzero_precip, prob_zero_precip])
                    bayes_std = np.average(
                        [np.mean(np.array(bayes_std_list)), bayes_std_zero],
                        weights=[prob_nonzero_precip, prob_zero_precip])
                    bayes_loc = np.average(
                        [np.mean(np.array(bayes_loc_list)), bayes_loc_zero],
                        weights=[prob_nonzero_precip, prob_zero_precip])
                    bayes_scale = np.average(
                        [np.mean(np.array(bayes_scale_list)), bayes_scale_zero],
                        weights=[prob_nonzero_precip, prob_zero_precip])



                ipgrad[t] = bayes_mean
                ipgrad_unc[t] = bayes_std
                ipgrad_loc[t] = bayes_loc
                ipgrad_scale[t] = bayes_scale
                if np.isnan(bayes_mean) or np.isnan(bayes_std):
                    print(pprcp.time.isel(time=t), pprcp.values[t])
            else:
                slope, icpt, _, p_val, stderr = stats.linregress(
                    np.ma.masked_array(phgt, ~mask).compressed(),
                    flattened_mask)

                ipgrad[t] = np.nanmean(((flattened_mask + slope) /
                                        flattened_mask) - 1) if (
                        (p_val < 0.01) and (p_val != 0.)) else default_pgrad
                # todo: I think this is bullshit: Gradient - Gradient derived with steeper slope
                ipgrad_unc[t] = \
                    (ipgrad[t] - np.nanmean(((flattened_mask + slope + stderr)
                                             / flattened_mask) - 1)) if \
                        ((p_val < 0.01) and (p_val != 0.)) else np.nan

        # todo: fill prcp gradient with rolling window mean?
        ref_cell = pprcp.values[:, int(win_size / 2), int(win_size / 2)]
        # prec_true = ref_cell > 0.

        # mean is influenced a lot by outliers!
        # ipgrad[np.isnan(ipgrad)] = np.nanmedian(
        #    ipgrad)  # & prec_true)] = np.nanmedian(ipgrad)
        # we take std around the median
        # ipgrad_unc[np.isnan(ipgrad_unc)] = np.sqrt(  # & prec_true)] = np.sqrt(
        #    np.nanmean((ipgrad_unc - np.nanmedian(ipgrad_unc)) ** 2))
        # apply the boundaries, in case the gradient goes wild
        # ipgrad = np.clip(ipgrad, minmax_pgrad[0], minmax_pgrad[1])

        ipgrad = xr.DataArray(ipgrad, coords={'time': climate.coords['time']},
                              dims=('time',), name='pgrad')
        ipgrad_unc = xr.DataArray(ipgrad_unc,
                                  coords={'time': climate.coords['time']},
                                  dims=('time',), name='pgrad_unc')

        if use_bayes_post is True:
            ipgrad_loc = xr.DataArray(ipgrad_loc,
                                  coords={'time': climate.coords['time']},
                                  dims=('time',), name='pgrad_loc')
            ipgrad_scale = xr.DataArray(ipgrad_scale,
                                  coords={'time': climate.coords['time']},
                                  dims=('time',), name='pgrad_scale')

    return ipgrad, ipgrad_unc, ipgrad_loc, ipgrad_scale


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
            gdf = gdir.read_shapefile('outlines_ts')
        except:  # no such file or directory: bad practice, but no
            gdf = gdir.read_shapefile('outlines')
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
    if os.name == 'nt':
        command = r'net use {} {}'.format(path, user)
    elif os.name == 'posix':
        command = r'dbus-launch bash && gvfs-mount {}@{}'.format(user, path)
    else:
        raise ValueError('Don\'t know how to mount a network drive on your '
                         'operating system "{}"'.format(os.name))

    out = subprocess.call(command, stdout=subprocess.PIPE, shell=True)

    if out == 0:
        msg = 'Network drive {} successfully connected'.format(path)
        log.info(msg) if log else print(msg)
    else:
        msg = 'Network drive {} connection failed'.format(path)
        log.warn(msg) if log else print(msg)

    return out


def _local_dem_to_xr_dataset(to_merge, acq_date, dx, calendar_startyear=0,
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
    # todo: interpolate NaNs here first, otherwise we lose data!!!
    xr_das = [da.interp(x=np.arange(da.x.min(), da.x.max() + dx, dx),
              y=np.arange(da.y.max(), da.y.min() - dx, -dx)) for da in xr_das]
    for da in xr_das:
        da.attrs.update({'transform': (
        dx, da.transform[1], da.x.min().item(), da.transform[3], -dx,
        da.y.min().item()), 'res': (dx, dx)})
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
    # merged = merged.drop_sel('band')
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
    .. [1] https://shorturl.at/eQV47
    """
    # get already the dx to which the DEMs should be interpolated later on
    dx = dx_from_area(gdir.area_km2)
    '''
    # get forest inventory DEMs (quite hard-coded for the LFI file names)
    out = mount_network_drive(cfg.PATHS['lfi_dir'], r'wsl\landmann', log=log)

    if out != 0:
        cfg.PATHS['lfi_dir'] = os.path.join(cfg.PATHS['data_dir'], 'DEM',
                                            'LFI')

    # neither os nor pathlib works, so quick'n'dirty:
    lfi_dem_list = glob.glob(os.path.join(cfg.PATHS['lfi_dir'], 'TIFF', '*.tif'))

    if not lfi_dem_list:
        raise FileNotFoundError('National Forest Inventory DEMs not found!')

    # exactly one file name is broken:
    broke_months = ['ADS_103221_672000_165000_2011_0_0_86_50.tif']
    lfi_dem_list = [x for x in lfi_dem_list if not broke_months[0] in x]

    zones = get_zones_from_worksheet(cfg.PATHS['lfi_worksheet'], 'CLNR',
                                     gdir=gdir)

    # 'common' DEMs for this glacier (all years!)
    cdems = []
    for q in zones:
        match = 'ADS_{}'.format(str(q))
        # somehow, this doesn't work with list comprehension
        for i in lfi_dem_list:
            if match in i:
                cdems.append(i)

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
        lfi_dem = _local_dem_to_xr_dataset(lfi_to_merge, lfi_dates[0], dx)

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
    '''
    # get DHM25 DEMs
    log.info('Assembling DHM25 DEM for {}'.format(gdir.rgi_id))
    d_list = glob.glob(os.path.join(cfg.PATHS['dem_dir'], '*' +
                                    cfg.NAMES['DHM25'].upper() + '*', '*.agr'))
    d_ws_path = glob.glob(os.path.join(cfg.PATHS['dem_dir'], 'worksheets',
                                       '*' + cfg.NAMES['DHM25'].upper() +
                                       '*.shp'))
    d_zones = get_zones_from_worksheet(d_ws_path[0], 'zone', gdir=gdir)
    d_to_merge = []
    for d_z in d_zones:
        d_to_merge.extend([d for d in d_list if str(d_z) in d])
    # TODO: Replace with real acquisition dates!
    d_acq_dates = dt.datetime(1970, 1, 1)
    d_dem = _local_dem_to_xr_dataset(d_to_merge, d_acq_dates, dx)
    d_dem.to_netcdf(path=gdir.get_filepath('dem_ts'), mode='w',
                    group=cfg.NAMES['DHM25'])

    # get SwissALTI3D DEMs
    log.info('Assembling SwissALTI3D DEM for {}'.format(gdir.rgi_id))
    a_list = glob.glob(os.path.join(cfg.PATHS['dem_dir'], '*' +
                       cfg.NAMES['SWISSALTI2010'].upper() + '*', '*.agr'))
    a_ws_path = glob.glob(os.path.join(cfg.PATHS['dem_dir'], 'worksheets',
                                       '*'+cfg.NAMES['SWISSALTI2010'].upper()
                                       + '*.shp'))
    a_zones = get_zones_from_worksheet(a_ws_path[0], 'zone', gdir=gdir)
    a_to_merge = []
    for a_z in a_zones:
        a_to_merge.extend([d for d in a_list if str(a_z) in d])
    # TODO: Replace with real acquisition dates!
    a_acq_dates = dt.datetime(2010, 1, 1)
    a_dem = _local_dem_to_xr_dataset(a_to_merge, a_acq_dates, dx)
    a_dem.to_netcdf(path=gdir.get_filepath('dem_ts'), mode='a',
                    group=cfg.NAMES['SWISSALTI2010'])


def get_cirrus_yesterday():
    """Check if data has already been delivered."""
    try:
        with xr.open_dataset(cfg.PATHS['climate_file']) as climate:
            yesterday = pd.Timestamp(climate.time.values[-1])
    except Exception:
        log.warning('Can\'t determine the Cirrus yesterday from climate file.'
                    ' Trying it with logic...')
        now = dt.datetime.now()
        if now.hour >= 12 and now.minute >= 30:
            yesterday = (now - dt.timedelta(1))
        else:
            yesterday = (now - dt.timedelta(2))

    return yesterday


def dx_from_area(area_km2):
    """
    Get spatial resolution as a function of glacier area and chosen parameters.

    Parameters
    ----------
    area_km2: float
        Glacier area in square kilometers (km-2).

    Returns
    -------
    dx: int
        Spatial resolution rounded to nearest integer.
    """
    dxmethod = cfg.PARAMS['grid_dx_method']
    if dxmethod == 'linear':
        dx = np.rint(cfg.PARAMS['d1'] * area_km2 + cfg.PARAMS['d2'])
    elif dxmethod == 'square':
        dx = np.rint(cfg.PARAMS['d1'] * np.sqrt(area_km2) + cfg.PARAMS['d2'])
    elif dxmethod == 'fixed':
        dx = np.rint(cfg.PARAMS['fixed_dx'])
    else:
        raise ValueError('grid_dx_method not supported: {}'.format(dxmethod))
    # Additional trick for varying dx
    if dxmethod in ['linear', 'square']:
        dx = np.clip(dx, cfg.PARAMS['d2'], cfg.PARAMS['dmax'])

    return dx


def get_possible_parameters_from_past(gdir, mb_model, as_list=False,
                                      latest_climate=False, only_pairs=True,
                                      constrain_with_bw_prcp_fac=True,
                                      suffix=''):
    """
    Get all possible calibration parameters from past calibration.

    This is useful for single-model mass balance predictions as it generates
    uncertainty. It uses itertools.product to mix the calibrated parameters.

    Parameters
    ----------
    gdir: `py:class:crampon:GlacierDirectory`
        The glacier directory to get the parameter combination for.
    mb_model: `py:class:crampon.core.models.massbalance.MassBalanceModel`
    as_list: bool
        True if the result returned should be a list.
    latest_climate: bool
        If only the last 30 years of the claibration period shall be considered.
    only_pairs:
        Determines whether there are only parameter combinations used which
        have already appeared, no random mixing. If False, the all available
        parameters are randomly mixed. Validation has shown that it makes sense
        to use pairs as the parameters are often correlated. #todo: reference
    constrain_with_bw_prcp_fac: bool
        If True and in the last row of the calibration dataframe only a prcp
        factor is given, all possible combinations will be restricted to this
        factor and the possible melt factors. Default: True.
    suffix: str
        Suffix for the calibration file to generate the parameters from.

    Returns
    -------
    p_comb: itertools.product or list
        An itertools.product instance with the parameter combinations, or, if
        set to True, a list.
    """

    cali_df = gdir.get_calibration(mb_model, filesuffix=suffix)

    cali_filtered = cali_df.filter(regex=mb_model.__name__)
    # important: be sure columns match the "order" of the params
    cali_filtered = cali_filtered[[mb_model.__name__ + '_' + i for i in
                         mb_model.cali_params_list]]
    cali_sel = cali_filtered.drop_duplicates().dropna().values

    # TODO: check from theory if this is a clever idea
    if latest_climate:
        try:
            cali_sel = cali_sel[-30:]
        except IndexError:
            pass

    if only_pairs:
        param_prod = cali_sel.copy()
    else:
        param_prod = product(*cali_sel.T)

    # todo: the output format is a mess now: list, array or product

    if constrain_with_bw_prcp_fac:
        log.info('PRCP_FAC is constrained with BW prcp_fac.')
        param_prod = np.array(list(param_prod))
        if ~pd.isnull(cali_df.tail(1)[mb_model.__name__ + '_' +
                                      'prcp_fac']).item() and \
                pd.isnull(cali_df.tail(1)[mb_model.__name__ + '_' +
                                          mb_model.cali_params_list[0]]).item():
            param_prod[:, mb_model.cali_params_list.index('prcp_fac')] = \
                cali_df.tail(1)[mb_model.__name__ + '_prcp_fac']

    if as_list is True:
        return list(param_prod)
    else:
        return param_prod


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
    .. [1] https://shorturl.at/pACMN
    """

    def __init__(self, entity, base_dir=None, reset=False):
        """Creates a new directory or opens an existing one.
        Parameters
        ----------
        entity : a `GeoSeries <https://shorturl.at/bfL39>`_ or str
            glacier entity read from the shapefile (or a valid RGI ID if the
            directory exists)
        base_dir : str
            path to the directory where to open the directory.
            Defaults to `cfg.PATHS['working_dir'] + /per_glacier/`
        reset : bool, default=False
            empties the directory at construction (careful!)
        """

        # Making oggm.GlacierDirectory available for composition
        self.OGGMGD = OGGMGlacierDirectory(entity, base_dir=base_dir,
                                           reset=reset)

        if base_dir is None:
            if not cfg.PATHS.get('working_dir', None):
                raise ValueError("Need a valid PATHS['working_dir']!")
            base_dir = os.path.join(cfg.PATHS['working_dir'],
                                    'per_glacier')

        # IDs are also valid entries
        if isinstance(entity, str):
            id_string = entity
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
            # todo: find a better solution!?
            # self.id = entity.RGIId
            # self.rgi_id = entity.RGIId
            try:
                self.id = id_string
                self.rgi_id = id_string
            except (ValueError, UnboundLocalError):
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
            self.plot_name = name.split('*')[0].strip() if '*' in name else \
                name

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
            # make OGGM compatible
            self.status = self.OGGMGD.status
            self.is_nominal = self.status == 'Nominal glacier'
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

    def read_json(self, filename, filesuffix=''):
        return self.OGGMGD.read_json(filename=filename,
                                        filesuffix=filesuffix)

    def write_json(self, var, filename, filesuffix=''):
        return self.OGGMGD.write_json(var=var, filename=filename,
                                      filesuffix=filesuffix)

    def get_calibration(self, mb_model=None, filesuffix=''):
        """
        Read the glacier calibration as a pandas.DataFrame.

        Parameters
        ----------
        mb_model: `py:class:crampon.core.massbalance.MassBalanceModel` or str
            Model to filter the calibration for.
        filesuffix: str
            Suffix for calibration file name (used in experiments).

        Returns
        -------
        df: pandas.Dataframe
            Parameter calibration for the glacier directory.
        """

        df = pd.read_csv(self.get_filepath('calibration',
                                           filesuffix=filesuffix),
                         index_col=0, parse_dates=[0])

        if mb_model is None:
            return df
        else:
            if isinstance(mb_model, str):
                return df.filter(regex=mb_model)
            else:
                return df.filter(regex=mb_model.__name__)

    def read_shapefile(self, filename, filesuffix=''):
        return self.OGGMGD.read_shapefile(filename, filesuffix=filesuffix)

    def write_shapefile(self, var, filename, filesuffix=''):
        return self.OGGMGD.write_shapefile(var, filename,
                                           filesuffix=filesuffix)

    def get_inversion_flowline_hw(self):
        return self.OGGMGD.get_inversion_flowline_hw()

    def log(self, task_name, *, err=None, task_time=None):
        return self.OGGMGD.log(task_name=task_name, err=err,
                               task_time=task_time)

    def get_task_status(self, task_name):
        return self.OGGMGD.get_task_status(task_name=task_name)

    def write_monthly_climate_file(self, time, prcp, temp, tgrad, pgrad,
                                   ref_pix_hgt, ref_pix_lon, ref_pix_lat, *,
                                   tmin=None, tmax=None, tgrad_sigma=None,
                                   sis=None, prcp_sigma=None, temp_sigma=None,
                                   tmin_sigma=None, tmax_sigma=None,
                                   sis_sigma=None, pgrad_sigma=None,
                                   tgrad_loc=None, tgrad_scale=None,
                                   pgrad_loc=None, pgrad_scale=None,
                                   time_unit='days since 1801-01-01 00:00:00',
                                   file_name='climate_monthly', filesuffix=''):
        """Creates a netCDF4 file with climate data.

        The biggest part of this function is the same as in OGGM, however we
        have to add the radiation as additional in-/output.
        """

        # overwrite as default
        fpath = self.get_filepath(file_name, filesuffix=filesuffix)
        if os.path.exists(fpath):
            os.remove(fpath)

        with netCDF4.Dataset(fpath, 'w', format='NETCDF4') as nc:
            nc.ref_hgt = ref_pix_hgt
            nc.ref_pix_lon = ref_pix_lon
            nc.ref_pix_lat = ref_pix_lat
            nc.ref_pix_dis = haversine(self.cenlon, self.cenlat,
                                       ref_pix_lon, ref_pix_lat)

            dtime = nc.createDimension('time', None)
            # quite hard-coded, but it should anyway not be allowed to mix:
            if 'member' in prcp.dims:
                memv = nc.createDimension('member', None)

            nc.author = 'OGGM and CRAMPON'
            nc.author_info = 'Open Global Glacier Model and Cryospheric ' \
                             'Monitoring and Prediction Online'

            timev = nc.createVariable('time', 'i4', ('time',))
            timev.setncatts({'units': time_unit})
            timev[:] = netCDF4.date2num([t for t in time], time_unit)

            # quite hard-coded, but it shoudl anyway not be allowed to mix:
            if 'member' in prcp.dims:
                memv = nc.createVariable('member', 'i4', ('member',))
                memv[:] = prcp.member.values

            v = nc.createVariable('prcp', 'f4', prcp.dims, zlib=True)
            v.units = 'kg m-2'
            v.long_name = 'total daily precipitation amount'
            v[:] = prcp

            v = nc.createVariable('temp', 'f4', temp.dims, zlib=True)
            v.units = 'degC'
            v.long_name = 'Mean 2m temperature at height ref_hgt'
            v[:] = temp

            if tmin is not None:
                v = nc.createVariable('tmin', 'f4', tmin.dims, zlib=True)
                v.units = 'degC'
                v.long_name = 'Minimum 2m temperature at height ref_hgt'
                v[:] = tmin

            if tmax is not None:
                v = nc.createVariable('tmax', 'f4', tmax.dims, zlib=True)
                v.units = 'degC'
                v.long_name = 'Maximum 2m temperature at height ref_hgt'
                v[:] = tmax

            if sis is not None:
                v = nc.createVariable('sis', 'f4', sis.dims, zlib=True)
                v.units = 'W m-2'
                v.long_name = 'daily mean surface incoming shortwave radiation'
                v[:] = sis

            if prcp_sigma is not None:
                v = nc.createVariable('prcp_sigma', 'f4', prcp_sigma.dims, zlib=True)
                v.units = 'kg m-2'
                v.long_name = 'Precipitation uncertainty as standard deviation'
                v[:] = prcp_sigma

            if temp_sigma is not None:
                v = nc.createVariable('temp_sigma', 'f4', temp_sigma.dims, zlib=True)
                v.units = 'deg C'
                v.long_name = 'Temperature uncertainty as standard deviation'
                v[:] = temp_sigma

            if tmin_sigma is not None:
                v = nc.createVariable('tmin_sigma', 'f4', tmin_sigma.dims, zlib=True)
                v.units = 'deg C'
                v.long_name = 'minimum temperature uncertainty as standard ' \
                              'deviation'
                v[:] = tmin_sigma

            if tmax_sigma is not None:
                v = nc.createVariable('tmax_sigma', 'f4', tmax_sigma.dims, zlib=True)
                v.units = 'deg C'
                v.long_name = 'maximum temperature uncertainty as standard ' \
                              'deviation'
                v[:] = tmax_sigma

            if sis_sigma is not None:
                v = nc.createVariable('sis_sigma', 'f4', sis_sigma.dims, zlib=True)
                v.units = 'W m-2'
                v.long_name = 'daily mean surface incoming shortwave ' \
                              'radiation as standard deviation'
                v[:] = sis_sigma

            v = nc.createVariable('tgrad', 'f4', tgrad.dims, zlib=True)
            v.units = 'K m-1'
            v.long_name = 'temperature gradient'
            v[:] = tgrad

            if tgrad_sigma is not None:
                v = nc.createVariable('tgrad_sigma', 'f4', tgrad_sigma.dims,
                                      zlib=True)
                v.units = 'K m-1'
                v.long_name = 'temperature gradient uncertainty'
                v[:] = tgrad_sigma

            if tgrad_loc is not None:
                v = nc.createVariable('tgrad_loc', 'f4', tgrad_loc.dims,
                                      zlib=True)
                v.units = 'K m-1'
                v.long_name = 'temperature gradient t-distribution loc parameter'
                v[:] = tgrad_loc

            if tgrad_scale is not None:
                v = nc.createVariable('tgrad_scale', 'f4', tgrad_scale.dims,
                                      zlib=True)
                v.units = 'K m-1'
                v.long_name = 'temperature gradient t-distribution scale parameter'
                v[:] = tgrad_scale

            v = nc.createVariable('pgrad', 'f4', pgrad.dims, zlib=True)
            v.units = 'm-1'
            v.long_name = 'precipitation gradient'
            v[:] = pgrad

            if pgrad_sigma is not None:
                v = nc.createVariable('pgrad_sigma', 'f4', pgrad_sigma.dims,
                                      zlib=True)
                v.units = 'm-1'
                v.long_name = 'precipitation gradient uncertainty'
                v[:] = pgrad_sigma

            if pgrad_loc is not None:
                v = nc.createVariable('pgrad_loc', 'f4', pgrad_loc.dims,
                                      zlib=True)
                v.units = 'm-1'
                v.long_name = 'precipitation gradient t-distribution loc parameter'
                v[:] = pgrad_loc

            if pgrad_scale is not None:
                v = nc.createVariable('pgrad_scale', 'f4', pgrad_scale.dims,
                                      zlib=True)
                v.units = ''
                v.long_name = 'precipitation gradient t-distribution scale parameter'
                v[:] = pgrad_scale

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


def idealized_gdir(surface_h, widths_m, map_dx, flowline_dx=1, name=None,
                   identifier=None, coords=None, base_dir=None, reset=False):
    """
    Creates a glacier directory with flowline input data only.

    This is basically a copy of OGGM's idealized_gdir with some changes, e.g.
    a "name" keyword. It is useful for testing, or for idealized experiments.

    Parameters
    ----------
    surface_h : ndarray
        the surface elevation of the flowline's grid points (in m).
    widths_m : ndarray
        the widths of the flowline's grid points (in m).
    map_dx : float
        the grid spacing (in m)
    flowline_dx : int, optional
        the flowline grid spacing (in units of map_dx) Default: 1.
    name: str, optional
        Name of the idealized glacier.
    identifier: str, optional
        ID of the idealized glacier. If no identifier is given the directory
        will get the ID "RGI50-00.00000".
    # TODO: Coords are stupid: What if we don't want a point only?=
    coords: tuple of (lat, lon), optional
        Latitude and longitude of the idealized glacier. If not given, a range
        from zero to the length of `surface_h` is taken as coords.
    base_dir : str
        Path to the directory where to open the directory.
        Defaults to `cfg.PATHS['working_dir'] + /per_glacier/`
    reset : bool, default=False
        Empty the directory at construction.

    Returns
    -------
    gdir: crampon.GlacierDirectory
        A GlacierDirectory instance.
    """

    from oggm.core.centerlines import Centerline

    # Area from geometry
    area_km2 = np.sum(widths_m * map_dx * flowline_dx) * 1e-6

    # Dummy entity - should probably also change the geometry
    # TODO: change the geometry
    entity_gdf = salem.read_shapefile(get_demo_file('Hintereisferner_RGI5.shp'))
    _ = salem.check_crs(entity_gdf.crs)
    entity = entity_gdf.iloc[0]
    entity.Area = area_km2
    if coords:
        entity.CenLat = coords[0]
        entity.CenLon = coords[1]
    else:
        entity.CenLat = 0
        entity.CenLon = 0
    if name:
        entity.Name = name
    else:
        entity.Name = ''
    if identifier:
        entity.RGIId = identifier
    else:
        entity.RGIId = 'RGI50-00.00000'
    entity.O1Region = '00'
    entity.O2Region = '0'
    gdir = GlacierDirectory(entity, base_dir=base_dir, reset=reset)
    gdf = gpd.GeoDataFrame([entity], crs=entity_gdf.crs)
    gdir.write_shapefile(gdf, 'outlines')

    # Idealized flowline
    if coords:
        coords = np.array(coords)
    else:
        coords = np.arange(0, len(surface_h)-0.5, 1)
    line = shpg.LineString(np.vstack([coords, coords * 0.]).T)
    fl = Centerline(line, dx=flowline_dx, surface_h=surface_h)
    fl.widths = widths_m / map_dx
    fl.is_rectangular = np.ones(fl.nx).astype(np.bool)
    gdir.write_pickle([fl], 'inversion_flowlines')

    # Idealized map
    grid = salem.Grid(nxny=(1, 1), dxdy=(map_dx, map_dx), x0y0=(0, 0))
    grid.to_json(gdir.get_filepath('glacier_grid'))

    return gdir


# this function is copied from OGGM
def initialize_merged_gdir_crampon(main, tribs=[], glcdf=None,
                           filename='climate_daily', input_filesuffix=''):
    """Creates a new GlacierDirectory if tributaries are merged to a glacier
    This function should be called after centerlines.intersect_downstream_lines
    and before flowline.merge_tributary_flowlines.
    It will create a new GlacierDirectory, with a suitable DEM and reproject
    the flowlines of the main glacier.
    Parameters
    ----------
    main : oggm.GlacierDirectory
        the main glacier
    tribs : list or dictionary containing oggm.GlacierDirectories
        true tributary glaciers to the main glacier
    glcdf: geopandas.GeoDataFrame
        which contains the main glacier, will be downloaded if None
    filename: str
        Baseline climate file
    input_filesuffix: str
        Filesuffix to the climate file
    Returns
    -------
    merged : oggm.GlacierDirectory
        the new GDir
    """
    from oggm.core.gis import define_glacier_region, merged_glacier_masks

    # If its a dict, select the relevant ones
    if isinstance(tribs, dict):
        tribs = tribs[main.rgi_id]
    # make sure tributaries are iteratable
    tribs = tolist(tribs)

    # read flowlines of the Main glacier
    mfls = main.read_pickle('model_flowlines') # OGGM

    # ------------------------------
    # 0. create the new GlacierDirectory from main glaciers GeoDataFrame
    # Should be passed along, if not download it
    if glcdf is None:
        glcdf = get_rgi_glacier_entities([main.rgi_id])
    # Get index location of the specific glacier
    idx = glcdf.loc[glcdf.RGIId == main.rgi_id].index
    maindf = glcdf.loc[idx].copy()

    # add tributary geometries to maindf
    merged_geometry = maindf.loc[idx, 'geometry'].iloc[0]
    for trib in tribs:
        geom = trib.read_pickle('geometries')['polygon_hr']
        geom = salem.transform_geometry(geom, crs=trib.grid)
        merged_geometry = merged_geometry.union(geom)

    # to get the center point, maximal extensions for DEM and single Polygon:
    maindf.loc[idx, 'geometry'] = merged_geometry.convex_hull

    # make some adjustments to the rgi dataframe
    # 1. update central point of new glacier
    # Could also use the mean of the Flowline centroids, to shift it more
    # downstream. But this is more straight forward.
    maindf.loc[idx, 'CenLon'] = maindf.loc[idx, 'geometry'].centroid.x
    maindf.loc[idx, 'CenLat'] = maindf.loc[idx, 'geometry'].centroid.y
    # 2. update names
    maindf.loc[idx, 'RGIId'] += '_merged'
    if maindf.loc[idx, 'Name'].iloc[0] is None:
        maindf.loc[idx, 'Name'] = main.name + ' (merged)'
    else:
        maindf.loc[idx, 'Name'] += ' (merged)'

    # finally create new Glacier Directory
    # 1. set dx spacing to the one used for the flowlines
    dx_method = cfg.PARAMS['grid_dx_method']
    dx_spacing = cfg.PARAMS['fixed_dx']
    cfg.PARAMS['grid_dx_method'] = 'fixed'
    # changed from OGGM - can be uncommented when dealing with flowlines
    # cfg.PARAMS['fixed_dx'] = mfls[-1].map_dx
    merged = GlacierDirectory(maindf.loc[idx].iloc[0])

    # run define_glacier_region to get a fitting DEM and proper grid
    define_glacier_region(merged, entity=maindf.loc[idx].iloc[0])

    # write gridded data and geometries for visualization
    merged_glacier_masks(merged, merged_geometry)

    # reset dx method
    cfg.PARAMS['grid_dx_method'] = dx_method
    cfg.PARAMS['fixed_dx'] = dx_spacing

    # copy main climate file, climate info and local_mustar to new gdir
    climfilename = filename + '_' + main.rgi_id + input_filesuffix + '.nc'
    climfile = os.path.join(merged.dir, climfilename)
    shutil.copyfile(main.get_filepath(filename, filesuffix=input_filesuffix),
                    climfile)

    # care for the second climate file if needed
    if 'climate_monthly' not in filename:
        climfilename_m = 'climate_monthly_' + main.rgi_id + input_filesuffix \
                         + '.nc'
        climfile_m = os.path.join(merged.dir, climfilename_m)
        try:
            shutil.copyfile(main.get_filepath('climate_monthly',
                                              filesuffix=input_filesuffix),
                            climfile_m)
        except:
            log.warning('No climate_monthly file found for {} during '
                        'GlacierDirectory merging.'.format(main.rgi_id))

    # try/except form OGGM
    try:
        _mufile = os.path.basename(merged.get_filepath('local_mustar')).split(
            '.')
        mufile = _mufile[0] + '_' + main.rgi_id + '.' + _mufile[1]
        shutil.copyfile(main.get_filepath('local_mustar'),
                        os.path.join(merged.dir, mufile))
    except FileNotFoundError:
        log.warning('No mu_star file found for {} during GlacierDirectory '
                    'merging.'.format(main.rgi_id))

    # Crampon extension
    try:
        _califile = os.path.basename(merged.get_filepath('calibration')).split(
            '.')
        califile = _califile[0] + '_' + main.rgi_id + '.' + _califile[1]
        shutil.copyfile(main.get_filepath('calibration'),
                        os.path.join(merged.dir, califile))
    except FileNotFoundError:
        log.warning(
            'No calibration file found for {} during GlacierDirectory '
            'merging.'.format(main.rgi_id))
    # I think I need the climate_info only for the main glacier
    shutil.copyfile(main.get_filepath('climate_info'),
                    merged.get_filepath('climate_info'))

    try:
        _ipotfile = os.path.basename(merged.get_filepath('ipot')).split('.')
        ipotfile = _ipotfile[0] + '_' + main.rgi_id + '.' + _ipotfile[1]
        shutil.copyfile(main.get_filepath('ipot'),
                        os.path.join(merged.dir, ipotfile))
    except FileNotFoundError:
        log.warning('No Ipot file found for {} during GlacierDirectory '
                    'merging.'.format(main.rgi_id))

    try:
        _ippffile = os.path.basename(
            merged.get_filepath('ipot_per_flowline')).split('.')
        ippffile = _ippffile[0] + '_' + main.rgi_id + '.' + _ippffile[1]
        shutil.copyfile(main.get_filepath('ipot_per_flowline'),
                        os.path.join(merged.dir, ippffile))
    except FileNotFoundError:
        log.warning('No Ipot per flowline file found for {} during '
                    'GlacierDirectory merging.'.format(main.rgi_id))

    # reproject the flowlines to the new grid
    for nr, fl in reversed(list(enumerate(mfls))):

        # 1. Step: Change projection to the main glaciers grid
        _line = salem.transform_geometry(fl.line,
                                         crs=main.grid, to_crs=merged.grid)
        # 2. set new line
        fl.set_line(_line)

        # 3. set flow to attributes
        if fl.flows_to is not None:
            fl.set_flows_to(fl.flows_to)
        # remove inflow points, will be set by other flowlines if need be
        fl.inflow_points = []

        # 5. set grid size attributes
        dx = [shpg.Point(fl.line.coords[i]).distance(
            shpg.Point(fl.line.coords[i+1]))
            for i, pt in enumerate(fl.line.coords[:-1])]  # get distance
        # and check if equally spaced
        if not np.allclose(dx, np.mean(dx), atol=1e-2):
            raise RuntimeError('Flowline is not evenly spaced.')
        # dx might very slightly change, but should not
        fl.dx = np.mean(dx).round(2)
        # map_dx should stay exactly the same
        # fl.map_dx = mfls[-1].map_dx
        # fl.dx_meter = fl.map_dx * fl.dx

        # replace flowline within the list
        mfls[nr] = fl

    # Write the reprojecflowlines
    merged.write_pickle(mfls, 'model_flowlines')

    return merged


# overwrite OGGM
initialize_merged_gdir = initialize_merged_gdir_crampon


class LogitNormal(stats.rv_continuous):
    """
    Logit normal distribution class in the style of scipy.

    This is copied from [1]_

    References:
    .. [1] https://bit.ly/3sdRqiZ
    """
    def __init__(self, scale=1, loc=0):
        super().__init__(self)
        self.scale = scale
        self.loc = loc

    def _pdf(self, x):
        return stats.norm.pdf(special.logit(x), loc=self.loc,
                              scale=self.scale) / (x*(1-x))


def physical_to_log(x: float or np.array, lower_bound: float,
                    fudge_fac: float = 0.) -> float or np.array:
    """
    Transformation from physical to log space.

    Fudge factor inspired by https://bit.ly/3skFf4t.

    Parameters
    ----------
    x : float or np.array
        Variable to be transformed.
    lower_bound : float
        Lower bound of the variable in physical space. Set e.g. to zero for a
        [0, Inf] interval of the precipitation correction factor.
    fudge_fac: float, optional
        From https://bit.ly/3skFf4t: Fudge factor (0<fudge_fac<<1) to avoid
        discontinuities at the boundaries in the scaled logit transform.
        Recommended value is 1e-2. Default: 0.

    Returns
    -------
    xts: same as input
        Transformed variable in log space.
    """

    # correct bound to avoid discontinuities
    lower_corr = lower_bound - fudge_fac
    x[x < lower_bound] = lower_bound
    xs = x - lower_corr

    # actual transform
    xts = np.log(xs)

    return xts


def log_to_physical(x: float or np.array, lower_bound: float,
                    fudge_fac: float = 0.) -> float or np.array:
    """
    Transformation from log to physical space.

    Fudge factor inspired by https://bit.ly/3skFf4t.

    Parameters
    ----------
    x : float or np.array
        Variable to be transformed.
    lower_bound : float
        Lower bound of the variable in physical space. Set e.g. to zero for a
        [0, Inf] interval of the precipitation correction factor.
    fudge_fac: float, optional
        From https://bit.ly/3skFf4t: Fudge factor (0<fudge_fac<<1) to avoid
        discontinuities at the boundaries in the scaled logit transform.
        Recommended value is 1e-2. Default: 0.

    Returns
    -------
    xts: same as input
        Transformed variable in physical space.
    """

    # transform back
    xt = np.exp(x)

    # correct boundaries
    lower_corr = lower_bound - fudge_fac
    xts = xt + lower_corr
    xts[xts < lower_bound] = lower_bound
    return xts


def physical_to_logit(x: float or np.array, lower_bound: float,
                      upper_bound: float, fudge_fac: float = 0.) \
        -> float or np.array:
    """
    Transformation from physical to logit space.

    Fudge factor inspired by https://bit.ly/3skFf4t.

    Parameters
    ----------
    x : float or np.array
        Variable to be transformed.
    lower_bound : float
        Lower bound of the variable in physical space. Set e.g. to zero for
        albedo.
    upper_bound : float
        Upper bound of the variable in physical space. Set e.g. to one for
        albedo.
    fudge_fac: float, optional
        From https://bit.ly/3skFf4t: Fudge factor (0<fudge_fac<<1) to avoid
        discontinuities at the boundaries in the scaled logit transform.
        Recommended value is 1e-2. Default: 0.

    Returns
    -------
    xts: same as input
        Transformed variable in logit space.
    """

    # correct the bounds numerically
    l = lower_bound - fudge_fac
    u = upper_bound + fudge_fac

    # do actual transform
    xs = (x - l) / (u - l)
    xts = np.log(xs) - np.log(1 - xs)

    return xts


def logit_to_physical(x: float or np.array, lower_bound: float,
                      upper_bound: float, fudge_fac: float = 0.) \
        -> float or np.array:
    """
    Transformation from physical to logit space.

    Fudge factor inspired by https://bit.ly/3skFf4t.

    Parameters
    ----------
    x : float or np.array
        Variable to be transformed.
    lower_bound : float
        Lower bound of the variable in physical space. Set e.g. to zero for
        albedo.
    upper_bound : float
        Upper bound of the variable in physical space. Set e.g. to one for
        albedo.
    fudge_fac: float, optional
        From https://bit.ly/3skFf4t: Fudge factor (0<fudge_fac<<1) to avoid
        discontinuities at the boundaries in the scaled logit transform.
        Recommended value is 1e-2. Default: 0.

    Returns
    -------
    xts: same as input
        Transformed variable in logit space.
    """
    # correct the bounds numerically
    lower_corr = lower_bound - fudge_fac
    upper_corr = upper_bound + fudge_fac

    # do actual backtransform
    xts = (upper_corr - lower_corr) / (1 + np.exp(-x)) + lower_corr
    xts[xts < lower_bound] = lower_bound
    xts[xts > upper_bound] = upper_bound

    return xts


def popup_html_string(point):
    """
    Make an HTML string used in a popup over a glacier.

    Parameters
    ----------
    point: pd.Series
       The Series must contain the name "RGIId".

    Returns
    -------
    html: str
        HTML string.
    """

    # a little less hardcoding
    pure_sgi_id = point.RGIId.split('.')[1]
    fs = 18
    clickable_mb_dist = '<a href="static/plots/mb_dist/{}_mb_dist_' \
                        'ensemble.png" target="_blank"><p ' \
                        'style="font-size:{}px;"> Mass ' \
                        'Balance Distribution (click to enlarge)</p>' \
                        '<img src="static/plots/mb_dist/{}_mb_dist_' \
                        'ensemble_prev.png" style="max-width:150%; ' \
                        'position:relative; ' \
                        'display:inline; overflow:hidden; margin:0;" ' \
                        '/></a>'.format(pure_sgi_id, fs, pure_sgi_id)
    other_clickables = '<a href="static/plots/mb_spaghetti/{}_intera' \
                       'ctive_mb_spaghetti.html" target="_blank"><p ' \
                       'style="font-size:{}px;"> Mass Balance ' \
                       'Spaghetti</p></a>'.format(pure_sgi_id, fs)

    html = '<p style="font-size:{}px;"> '.format(
        fs) + point.Name + ' (' + pure_sgi_id + ')</p><br>'
    html += '<div style="width:420px; height:250px; text-align: ' \
            'center">{}</div>'.format(clickable_mb_dist)
    html += '<p style="font-size:{}px;"> Further information: ' \
            '</p>'.format(fs)
    html += other_clickables
    return html


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
