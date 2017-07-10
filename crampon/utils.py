"""
A collection of some useful miscellaneous functions.
"""

from __future__ import absolute_import, division

from joblib import Memory
import salem
import os
import pandas as pd
import numpy as np
import logging
import paramiko as pm
import xarray as xr
from configobj import ConfigObj, ConfigObjError
import sys
from glob import glob
from salem import lazy_property, read_shapefile
from functools import partial, wraps
from oggm.utils import *  # easiest way to make utils accessible
""" include e.g.:_urlretrieve, progress_urlretrieve, empty_cache, \
expand_path, SuperClassMeta, mkdir, query_yes_no, haversine, interp_nans,
md, mad, rmsd, rel_err, corrcoef, nicenumber, signchange, year_to_date, \
date_to_year, monthly_timseries, joblib_read_climate, pipe_log, \
glacier_characteristics, DisableLogger, entity_task, global_task, \
GlacierDirectory, log"""
# Locals
import crampon.cfg as cfg


# I should introduce/alter:
"""utils.joblib_read_climate, get_crampon_demo_file, in general: get_files"""

# Module logger
log = logging.getLogger(__name__)
# Stop paramiko from logging successes to the console
#logging.getLogger ('paramiko').addHandler(log.handlers)
#logging.getLogger("paramiko").setLevel(logging.WARNING)

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
            (year % 100 == 0) and
            (year % 400 != 0)):
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
        self.client = self.create_connect(cr['cirrus']['host'],
                                          cr['cirrus']['user'],
                                          cr['cirrus']['password'])

    def create_connect(self, host, user, password, port=22):
        client = pm.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(pm.AutoAddPolicy())
        client.connect(host, port, user, password)
        self.ssh_open = True

        return client

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
            Base remote directory as relative path from the $HOME.
        remotelist: list
            Relative paths (from remote directory) to remote files which shall
            be retrieved.
        local: list
            Directory where to put the files (the relative path will be
            maintained).

        Returns
        -------

        """
        if not self.sftp_open:
            self.sftp = self.client.open_sftp()
            self.sftp_open = True

        remotepaths = [os.path.join(remotedir, r) for r in remotelist]
        localpaths = [os.path.join(targetdir, f) for f in remotelist]

        print(remotepaths)
        print(localpaths)

        for remotef, localf in zip(remotepaths, localpaths):
            if not os.path.exists(os.path.dirname(localf)):
                os.makedirs(os.path.dirname(localf), exist_ok=True)
            print(remotef)
            self.sftp.pwd
            self.sftp.get(remotef, localf)


    def sync_files(self, remotedir, localdir, globpattern='*', rm_local=False):
        """
        Synchronize a host machine with local content.

        If there are new files, download them. If there are less files on the
        host machine now than locally, you can choose to delete them with the
        `rm_local` keyword.

        This is a supercheap version of rsync which works via SFTP.
        Probably it makes sense to replace this by:
        https://stackoverflow.com/questions/16497166/
        rsync-over-ssh-using-channel-created-by-paramiko-in-python, THE PROBLEM
        IS ONLY THE WINDOWS SYSTEMS

        Parameters
        ----------
        remotedir: str
            Relative path (from home directory) to a remote destination which
            shall be synced recursively.
        localdir: str
            Absolute path to a local directory to be synced with remote
            directory.
        globpattern: str
            Pattern used for searching by glob. Default: '*' (list all files).

        Returns
        -------
        """

        remotelist = glob(remotedir + os.sep + globpattern)
        locallist = glob(localdir + os.sep + globpattern)

        # copy everything needed from remote to local
        needed = list(set(remotelist) - set(locallist))
        self.get_files(remotedir, needed, [localdir+os.sep+i for i in needed])

        # delete unnecessary stuff if desired
        if rm_local:
            oldlist = list(set(locallist) - set(remotelist))
            if oldlist:
                for old in oldlist:
                    try:
                        os.rmdir(old)
                    except PermissionError:
                        log.warning('File {} could not be deleted (Permission '
                                    'denied).'.format(old))

    def close(self):
        if self.sftp_open:
            self.sftp.close()
            self.sftp_open = False
        if self.ssh_open:
            self.client.close()
            self.ssh_open = False


class MeteoTimeSeries(xr.Dataset):

    def __init__(self, *args, **kwargs):
        xr.Dataset.__init__(self, *args, **kwargs)

        # Pseudo: If not in cache, download whole series


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

    def check_time_continuity(self):
        """
        Checks the time continuity of the time series.
        
        If there are missing time steps, fill them with NaN via the 
        xr.Dataset.resample method.
        
        Returns
        -------

        """
        # Pseudo:
        # If self.time misses a time step => resample with netcdf time unit

    def digest_input(self, file):
        """
        Take a file and append it to the current series (copy of update_with_*?)
        Parameters
        ----------
        file

        Returns
        -------

        """

'''
@entity_task(log, writes=['meteo'])
def process_meteosuisse_data(gdir):
    """Processes and writes the climate data from a user-defined climate file.

    The input file must have a specific format (see
    oggm-sample-data/test-files/histalp_merged_hef.nc for an example).

    Uses caching for faster retrieval.

    This is the way OGGM does it for the Alps (HISTALP).
    """

    if not (('climate_file' in cfg.PATHS) and
                os.path.exists(cfg.PATHS['climate_file'])):
        raise IOError('Custom climate file not found')

    # read the file
    fpath = cfg.PATHS['climate_file']
    nc_ts = salem.GeoNetcdf(fpath)

    # set temporal subset for the ts data (hydro years)
    yrs = nc_ts.time.year
    y0, y1 = yrs[0], yrs[-1]
    if pd.infer_freq(nc_ts.time) == 'MS':  # month start frequency
        nc_ts.set_period(t0='{}-10-01'.format(y0), t1='{}-09-01'.format(y1))
        time = nc_ts.time
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
    elif pd.infer_freq(nc_ts.time) == 'M':  # month end frequency
        nc_ts.set_period(t0='{}-10-31'.format(y0), t1='{}-09-30'.format(y1))
        time = nc_ts.time
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
    elif pd.infer_freq(nc_ts.time) == 'D':  # day start frequency
        pass  # doesn't matter if it's entire years or not
    else:
        raise NotImplementedError('Climate data frequency not yet understood')

    # Units
    assert nc_ts._nc.variables['hgt'].units.lower() in ['m', 'meters', 'meter']
    assert nc_ts._nc.variables['temp'].units.lower() in ['degC', 'degrees',
                                                         'degree']
    assert nc_ts._nc.variables['prcp'].units.lower() in ['kg m-2', 'l m-2',
                                                         'mm']

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
    cirrus = CirrusClient()