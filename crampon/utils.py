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
