"""
A collection of some useful miscellaneous functions.
"""

from __future__ import absolute_import, division

from joblib import Memory
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
"""utils.joblib_read_climate, get_demo_file, in general: get_files"""

# Joblib
MEMORY = Memory(cachedir=cfg.CACHE_DIR, verbose=0)


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

