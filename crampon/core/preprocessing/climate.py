""" Prepare the meteodata from netCDF4 """

from __future__ import division

import os
from glob import glob
import crampon.cfg as cfg
from crampon.core.models.massbalance import DailyMassBalanceModel
from crampon import utils
from crampon.utils import GlacierDirectory
import itertools
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from crampon import entity_task
import logging
from itertools import product
import salem

# temporary
from crampon.workflow import execute_entity_task

# Module logger
log = logging.getLogger(__name__)


class MeteoSuisseGrid(object):
    """ Interface for MeteoSuisse Input as netCDF4.

    The class interacts with xarray via salem and allows subsetting for each
    glacier, getting reference cells for each glacier, and gradient
    """

    def __init__(self, ncpath):
        self.ddir = ncpath
        self._data = salem.open_xr_dataset(self.ddir)
        self._vartype = None

    @property
    def vartype(self):
        """
        Check which data are contained. Currently implemented:
        "TabsD": mean daily temperature
        "RprelimD": daily precipitation sum

        Returns
        -------
        The "vartype" class attribute
        """
        var_dict = self._data.data_vars

        if "TabsD" in var_dict:
            self._vartype = 'TabsD'
        elif 'RprelimD' in var_dict:
            self._vartype = 'RprelimD'
        else:
            miss_vartype = list(var_dict.keys())[0]
            raise NotImplementedError('MeteoGrid variable {} is not yet '
                                      'implemented'.format(miss_vartype))

        return self._vartype

    @vartype.setter
    def vartype(self, value):
        self._vartype = value

    def subset_by_shape(self, shpdir=None, buffer=0):
        """
        Subset the array based on the given shapefile, including the buffer.

        Parameters
        ----------
        shpdir: str
             Path to the shape used for clipping
        buffer: int
             Cells to be used as buffer around the shape

        Returns
        -------

        """

        shape = salem.read_shapefile(shpdir, cached=True)

        subset = self._data.salem.subset(shape=shape, margin=buffer)

        return subset

    def clip_by_shape(self, shpdir=None):
        """
        Clip the array based on the given shapefile, including the buffer.

        Parameters
        ----------
        shpdir: str
             Path to the shape used for clipping

        Returns
        -------

        """
        shape = salem.read_shapefile(shpdir, cached=True)

        clipped = self._data.salem.subset(shape=shape)

        return clipped

    def get_reference_value(self, shpdir=None):

        shape = salem.read_shapefile(shpdir, cached=True)
        centroid = shape.centroid

        return shape

    def get_gradient(self):
        raise NotImplementedError()

    def downsample(self):
        # This should remain in crampon maybe, as OGGM doesn't need it
        raise NotImplementedError()

    def write_oggm_format(self):
        # Write out an OGGM suitable format
        raise NotImplementedError()

    def merge(self, other):
        """
        Merge with another MeteoSuisseGrid.
        
        Parameters
        ----------
        other: MeteoSuisseGrid to merge with.

        Returns
        -------
        Merged MeteoSuisseGrid.
        """


# This writes 'climate_monthly' in the original version (doesn't fit anymore)
@entity_task(log)
def process_custom_climate_data_crampon(gdir):
    """Processes and writes the climate data from a user-defined climate file.

    This function is strongly related to the OGGM function. The input file must
     have a specific format
     (see oggm-sample-data/test-files/histalp_merged_hef.nc for an example).

    The modifications to the original function allow a more flexible handling
    of the climate file, e.g. with a daily frequency.

    uses caching for faster retrieval.
    """

    if not (('climate_file' in cfg.PATHS) and
            os.path.exists(cfg.PATHS['climate_file'])):
        raise IOError('Custom climate file {} not found'
                      .format(cfg.PATHS['climate_file']))

    # read the file
    fpath = cfg.PATHS['climate_file']
    nc_ts = salem.GeoNetcdf(fpath)

    # geoloc
    lon = nc_ts._nc.variables['lon'][:]
    lat = nc_ts._nc.variables['lat'][:]

    # Gradient defaults
    use_tgrad = cfg.PARAMS['temp_use_local_gradient']
    def_tgrad = cfg.PARAMS['temp_default_gradient']
    tg_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    use_pgrad = cfg.PARAMS['prcp_use_local_gradient']
    def_pgrad = cfg.PARAMS['prcp_default_gradient']
    pg_minmax = cfg.PARAMS['prcp_local_gradient_bounds']

    # get closest grid cell and index
    ilon = np.argmin(np.abs(lon - gdir.cenlon))
    ilat = np.argmin(np.abs(lat - gdir.cenlat))
    ref_pix_lon = lon[ilon]
    ref_pix_lat = lat[ilat]

    # Some special things added in the crampon function
    iprcp, itemp, isis, itgrad, ipgrad, ihgt = \
        utils.joblib_read_climate(fpath, ilon, ilat, def_tgrad, tg_minmax,
                                  use_tgrad, def_pgrad, pg_minmax, use_pgrad)

    # Set temporal subset for the ts data depending on frequency:
    # hydro years if monthly data, else no restriction
    yrs = nc_ts.time.year
    y0, y1 = yrs[0], yrs[-1]
    time = nc_ts.time
    if pd.infer_freq(nc_ts.time) == 'MS':  # month start frequency
        nc_ts.set_period(t0='{}-10-01'.format(y0), t1='{}-09-01'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, isis, itgrad,
                                        ipgrad, ihgt, ref_pix_lon, ref_pix_lat)
    elif pd.infer_freq(nc_ts.time) == 'M':  # month end frequency
        nc_ts.set_period(t0='{}-10-31'.format(y0), t1='{}-09-30'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, isis, itgrad,
                                        ipgrad, ihgt, ref_pix_lon, ref_pix_lat)
    elif pd.infer_freq(nc_ts.time) == 'D':  # day start frequency
        # Doesn't matter if entire years or not, BUT a correction for y1 to be
        # the last hydro/glacio year is needed
        if not '{}-09-30'.format(y1) in nc_ts.time:
            y1 = yrs[-2]
        # Ok, this is NO ERROR: we can use the function
        # ``write_monthly_climate_file`` also to produce a daily climate file:
        # there is no reference to the time in the function! We should just
        # change the ``file_name`` keyword!
        gdir.write_monthly_climate_file(time, iprcp, itemp, isis, itgrad,
                                        ipgrad, ihgt, ref_pix_lon, ref_pix_lat,
                                        file_name='climate_daily',
                                        time_unit=nc_ts._nc.variables['time']
                                        .units)
    else:
        raise NotImplementedError('Climate data frequency not yet understood')

    # for logging
    end_date = time[-1]

    # metadata
    out = {'climate_source': fpath, 'hydro_yr_0': y0 + 1,
           'hydro_yr_1': y1, 'end_date': end_date}
    gdir.write_pickle(out, 'climate_info')


@entity_task(log)
def process_spinup_climate_data(gdir):
    """Processes the homogenized station data before 1961.

    Temperature should be extrapolated from the surrounding stations with
    monthly gradients.
    Precip correction should be calibrated with winter MBs
    """

    if not (('climate_file' in cfg.PATHS) and
            os.path.exists(cfg.PATHS['climate_file'])):
        raise IOError('Custom climate file {} not found'
                      .format(cfg.PATHS['climate_file']))

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

    # get closest grid cell and index
    ilon = np.argmin(np.abs(lon - gdir.cenlon))
    ilat = np.argmin(np.abs(lat - gdir.cenlat))
    ref_pix_lon = lon[ilon]
    ref_pix_lat = lat[ilat]

    # Some special things added in the crampon function
    iprcp, itemp, igrad, ihgt = utils.joblib_read_climate(fpath, ilon, ilat,
                                                          def_grad, g_minmax,
                                                          use_grad)

    # Set temporal subset for the ts data depending on frequency:
    # hydro years if monthly data, else no restriction
    yrs = nc_ts.time.year
    y0, y1 = yrs[0], yrs[-1]
    time = nc_ts.time
    if pd.infer_freq(nc_ts.time) == 'MS':  # month start frequency
        nc_ts.set_period(t0='{}-10-01'.format(y0), t1='{}-09-01'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                        ref_pix_lon, ref_pix_lat)
    elif pd.infer_freq(nc_ts.time) == 'M':  # month end frequency
        nc_ts.set_period(t0='{}-10-31'.format(y0), t1='{}-09-30'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                        ref_pix_lon, ref_pix_lat)
    elif pd.infer_freq(nc_ts.time) == 'D':  # day start frequency
        # Doesn't matter if entire years or not, BUT a correction for y1 to be
        # the last hydro/glacio year is needed
        if not '{}-09-30'.format(y1) in nc_ts.time:
            y1 = yrs[-2]
        # Ok, this is NO ERROR: we can use the function
        # ``write_monthly_climate_file`` also to produce a daily climate file:
        # there is no reference to the time in the function! We should just
        # change the ``file_name`` keyword!
        gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                        ref_pix_lon, ref_pix_lat,
                                        file_name='climate_daily',
                                        time_unit=nc_ts._nc.variables['time']
                                        .units)
    else:
        raise NotImplementedError('Climate data frequency not yet understood')

    # for logging
    end_date = time[-1]

    # metadata
    out = {'climate_source': fpath, 'hydro_yr_0': y0 + 1,
           'hydro_yr_1': y1, 'end_date': end_date}
    gdir.write_pickle(out, 'climate_info')


def climate_file_from_scratch(write_to=None, hfile=None):
    """
    Compile the climate file needed for any CRAMPON calculations.

    The file will contain an up to date meteorological time series from all
    currently available files on Cirrus. In the default setting, the
    configuration must be initialized to provide paths!

    Parameters
    ----------
    write_to: str
        Directory where the Cirrus files should be synchronized to and where
        the processed/concatenated files should be written to. Default: the
        'climate_dir' in the crampon configuration PATHS dictionary.
    hfile: str
        Path to a netCDF file containing a DEM of the area (used for assembling
        the file that OGGM likes. Needs to cover the same area in the same
        extent ans resolution as the meteo files. Default: the 'hfile' in the
        crampon configuration PATHS dictionary.

    Returns
    -------
    outfile: str
        The path to the compiled file.
    """

    if not write_to:
        try:
            write_to = cfg.PATHS['climate_dir']
        except KeyError:
            raise KeyError('Must supply write_to or initialize the crampon'
                           'configuration.')
    if not hfile:
        try:
            hfile = cfg.PATHS['hfile']
        except KeyError:
            raise KeyError('Must supply hfile or initialize the crampon'
                           'configuration.')

    for var, mode in product(['TabsD', 'R', 'msgSISD_'],
                             ['verified', 'operational']):
        all_file = os.path.join(write_to, '{}_{}_all.nc'.format(var, mode))
        cirrus = utils.CirrusClient()
        r, _ = cirrus.sync_files('/data/griddata', write_to
                                 , globpattern='*{}/daily/{}*/netcdf/*'
                                 .format(mode, var))

        # if at least one file was retrieved, assemble everything new
        flist = glob(os.path.join(write_to,
                                  'griddata\\{}\\daily\\{}*\\netcdf\\*.nc'
                                  .format(mode, var)))
        if (len(r) > 0) or ((not os.path.exists(all_file)) and len(flist) > 0):
            # Instead of using open_mfdataset (we need a lot of preprocessing)
            log.info('Concatenating {} {} {} files...'
                     .format(len(flist), var, mode))
            sda = utils.read_multiple_netcdfs(flist, chunks={'time': 50},
                                              tfunc=utils._cut_with_CH_glac)
            log.info('Ensuring time continuity...')
            sda = sda.crampon.ensure_time_continuity()
            sda.encoding['zlib'] = True
            sda.to_netcdf(all_file)

    # update operational with verified
    for var in ['TabsD', 'R', 'msgSISD_']:
        v_op = os.path.join(write_to, '{}_operational_all.nc'.format(var))
        v_ve = os.path.join(write_to, '{}_verified_all.nc'.format(var))

        # TODO: Remove when msgSISD is finally delivered operationally ?
        # can be that operational files are missing (msgSISD)
        try:
            data = utils.read_netcdf(v_op, chunks={'time': 50})
            data = data.crampon.update_with_verified(v_ve)
        except FileNotFoundError:
            data = utils.read_netcdf(v_ve, chunks={'time': 50})
        data.to_netcdf(os.path.join(write_to, '{}_op_ver.nc'.format(var)))

    # combine both
    tfile = glob(os.path.join(write_to, '*{}*_op_ver.nc'.format('TabsD')))[0]
    pfile = glob(os.path.join(write_to, '*{}*_op_ver.nc'.format('R')))[0]
    rfile = glob(os.path.join(write_to, '*{}*_op_ver.nc'.format('msgSISD_')))[
        0]
    outfile = os.path.join(write_to, 'climate_all.nc')

    log.info('Combining TEMP, PRCP, SIS and HGT...')
    utils.daily_climate_from_netcdf(tfile, pfile, rfile, hfile, outfile)
    log.info('Done combining TEMP, PRCP, SIS and HGT.')

    return outfile


# IMPORTANT: overwrite OGGM functions with same name:
process_custom_climate_data = process_custom_climate_data_crampon
