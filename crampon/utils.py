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
from configobj import ConfigObj, ConfigObjError
import dask
import sys
import glob
import fnmatch
import netCDF4
from scipy import stats
from salem import lazy_property, read_shapefile
from functools import partial, wraps
from oggm.utils import *  # easiest way to make utils accessible
# Locals
import crampon.cfg as cfg


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

    def update_with_verified(self):
        """
        Updates the time series with verified MeteoSwiss data.
        
        Returns
        -------

        """
        raise NotImplementedError()

    def update_with_operational(self):
        """
        Updates the time series with operational MeteoSwiss data.
        
        Returns
        -------

        """
        raise NotImplementedError()

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

        resampled = self._obj.resample(freq, dim='time', **kwargs)
        diff = len(resampled.time.values) - len(self._obj.time.values)
        log.info('{} time steps were added during resampling.'.format(diff))

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

        if "REFERENCE_TS" in self._obj.coords:
            self._obj.rename({"REFERENCE_TS": "time"}, inplace=True)

        if "crs" in self._obj.data_vars:
            self._obj = self._obj.drop(['crs'])

        return self._obj


def daily_climate_from_netcdf(tfiles, pfiles, hfile, outfile):
    """
    Create a netCDF file with daily temperature, precipitation and
    elevation reference from given files.

    The temporal extent of the file will be the inner or outer join of the time
    series extent of the given input files .

    Parameters
    ----------
    tfiles: list
        Paths to temperature netCDF files.
    pfiles: list
        Paths to precipitation netCDF files.
    hfile: str
        Path to the elevation netCDF file.
    outfile: str
        Path to and name of the written file.

    Returns
    -------

    """

    temp = read_multiple_netcdfs(tfiles)
    prec = read_multiple_netcdfs(pfiles)
    hgt = read_netcdf(hfile)

    # make it one
    nc_ts = xr.merge(temp, prec, hgt)

    # check that all needed variables and coordinates are there:

    # Units
    assert nc_ts._nc.variables['hgt'].units.lower() in ['m', 'meters', 'meter',
                                                        'metres', 'metre']
    assert nc_ts._nc.variables['temp'].units.lower() in ['degc', 'degrees',
                                                         'degree', 'c']
    assert nc_ts._nc.variables['prcp'].units.lower() in ['kg m-2', 'l m-2',
                                                         'mm', 'millimeters',
                                                         'millimeter']
    # Fill NAs
    nc_ts.resample('D', 'time')

    nc_ts.to_netcdf(outfile)


def read_netcdf(path, tfunc=None):
    # use a context manager, to ensure the file gets closed after use
    with xr.open_dataset(path) as ds:

        # some extra stuff - this is actually stupid and should go away!
        ds = ds.crampon.postprocess_cirrus()

        # transform_func should do some sort of selection or
        # aggregation
        if tfunc is not None:
            ds = tfunc(ds)

        # load all data from the transformed dataset, to ensure we can
        # use it after closing each original file
        ds.load()
        return ds


def read_multiple_netcdfs(files, dim='time', tfunc=None):
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

    Returns
    -------

    """

    paths = sorted(files)
    datasets = [read_netcdf(p, tfunc) for p in paths]

    combined = xr.concat(datasets, dim)
    return combined


@MEMORY.cache
def joblib_read_climate(ncpath, ilon, ilat, default_grad, minmax_grad,
                        use_grad=100):
    """
    This is a cracked version of the OGGM function with some extras.

    Parameters
    ----------
    ncpath: str
        Path to the netCDF file in OGGM suitable format.
    ilon: int
        Index of the minimum longitude of the netCDF
    ilat: int
        Index of the minimum latitude of the netCDF
    default_grad: float
        Default temperature gradient (K/m).
    minmax_grad: tuple
        Min/Max bounds of the local gradient, in case the grid kernel search
        deliver strange values.
    use_grad: float
        Window edge width of surrounding cells used to determine the local
        temperature gradient. If ``None``, the ``default_grad`` is used.

    Returns
    -------
    iprcp, itemp, igrad, ihgt
    """

    # read the file and data
    with netCDF4.Dataset(ncpath, mode='r') as nc:
        temp = nc.variables['temp']
        prcp = nc.variables['prcp']
        hgt = nc.variables['hgt']
        igrad = np.zeros(len(nc.dimensions['time'])) + default_grad ####
        ttemp = temp[:, ilat-1:ilat+2, ilon-1:ilon+2]    ######
        itemp = ttemp[:, 1, 1]                         ########
        thgt = hgt[ilat-1:ilat+2, ilon-1:ilon+2]        #######
        ihgt = thgt[1, 1]                               #######
        thgt = thgt.flatten()                           #######
        iprcp = prcp[:, ilat, ilon]                     #######

    # Now the gradient
    if use_grad:
        for t, loct in enumerate(ttemp):
            slope, _, _, p_val, _ = stats.linregress(thgt,
                                                     loct.flatten())
            igrad[t] = slope if (p_val < 0.01) else default_grad

        # apply the boundaries, in case the gradient goes wild
        igrad = np.clip(igrad, minmax_grad[0], minmax_grad[1])

    return iprcp, itemp, igrad, ihgt

'''
@entity_task(log, writes=['meteo'])
def process_meteosuisse_data(gdir):

    # read the file
    fpath = cfg.PATHS['climate_file']
    nc_ts = salem.GeoNetcdf(fpath)

    # geoloc
    lon = nc_ts._nc.variables['lon'][:]
    lat = nc_ts._nc.variables['lat'][:]

    # Gradient defaults
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    def_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    ilon = np.argmin(np.abs(lon - gdir.cenlon))
    ilat = np.argmin(np.abs(lat - gdir.cenlat))
    ref_pix_lon = lon[ilon]
    ref_pix_lat = lat[ilat]
    iprcp, itemp, igrad, ihgt = utils.joblib_read_climate(fpath, ilon,
                                                          ilat, def_grad,
                                                          g_minmax,
                                                          use_grad)
    gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                    ref_pix_lon, ref_pix_lat)
    # metadata
    out = {'climate_source': fpath, 'hydro_yr_0': y0+1, 'hydro_yr_1': y1}
    gdir.write_pickle(out, 'climate_info')
'''

if __name__ == '__main__':
    #cirrus = CirrusClient()
    #a, b = cirrus.sync_files('/data/griddata', 'c:\\users\\johannes\\desktop',
    #                  globpattern='*/daily/TabsD*/netcdf/*')

    flist = glob.glob('C:\\Users\\Johannes\\Desktop\\griddata\\verified\\daily\\TabsD_daily_mean_temperature\\netcdf\\*2015*.nc')
    flist.extend(glob.glob('C:\\Users\\Johannes\\Desktop\\griddata\\verified\\daily\\TabsD_daily_mean_temperature\\netcdf\\*2016*.nc'))
    abc = read_multiple_netcdfs(flist)
    abc.crampon
    abc.crampon.ensure_time_continuity()
    defg = abc.crampon.cut_by_glacio_years()
