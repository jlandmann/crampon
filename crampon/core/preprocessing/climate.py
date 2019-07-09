""" Prepare the meteodata from netCDF4 """

from __future__ import division

import os
from glob import glob
import crampon.cfg as cfg
from crampon import utils
from crampon.utils import lazy_property
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
import xarray as xr
import shutil
import sys

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
    iprcp, itemp, itmin, itmax, isis, itgrad, ipgrad, ihgt = \
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
        gdir.write_monthly_climate_file(time, iprcp, itemp, itgrad, ipgrad,
                                        ihgt, ref_pix_lon, ref_pix_lat,
                                        tmin=itmin, tmax=itmax, sis=isis)
    elif pd.infer_freq(nc_ts.time) == 'M':  # month end frequency
        nc_ts.set_period(t0='{}-10-31'.format(y0), t1='{}-09-30'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, itgrad, ipgrad,
                                        ihgt, ref_pix_lon, ref_pix_lat,
                                        tmin=itmin, tmax=itmax, sis=isis)
    elif pd.infer_freq(nc_ts.time) == 'D':  # day start frequency
        # Doesn't matter if entire years or not, BUT a correction for y1 to be
        # the last hydro/glacio year is needed
        if not '{}-09-30'.format(y1) in nc_ts.time:
            y1 = yrs[-2]
        # Ok, this is NO ERROR: we can use the function
        # ``write_monthly_climate_file`` also to produce a daily climate file:
        # there is no reference to the time in the function! We should just
        # change the ``file_name`` keyword!
        gdir.write_monthly_climate_file(time, iprcp, itemp, itgrad, ipgrad,
                                        ihgt, ref_pix_lon, ref_pix_lat,
                                        tmin=itmin, tmax=itmax, sis=isis,
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


def make_climate_file(write_to=None, hfile=None, how='from_scratch'):
    """
    Compile the climate file needed for any CRAMPON calculations.

    The file will contain an up to date meteorological time series from all
    currently available files on Cirrus. In the default setting, the
    configuration must be initialized to provide paths!

    Parameters
    ----------
    write_to: str or None
        Directory where the Cirrus files should be synchronized to and where
        the processed/concatenated files should be written to. Default: the
        'climate_dir' in the crampon configuration PATHS dictionary.
    hfile: str or None
        Path to a netCDF file containing a DEM of the area (used for assembling
        the file that OGGM likes. Needs to cover the same area in the same
        extent ans resolution as the meteo files. Default: the 'hfile' in the
        crampon configuration PATHS dictionary.
    how: str
        For 'from_scratch' the climate file is generated from scratch
        (time-consuming), for 'update' only the necessary files are appended to
        an existing climate file (saves some time). Default: 'from_scratch'.

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

    # again other variable names on FTP delivery; SIS D still not operational
    ftp_key_dict = {'BZ13': 'RprelimD', 'BZ14': 'RhiresD', 'BZ51': 'TabsD',
                    'BZ54': 'TminD', 'BZ55': 'TmaxD', 'BZ69': 'msgSISD_',
                    'CZ91': 'TabsD', 'CZ92': 'TminD', 'CZ93': 'TmaxD'}

    # cheap way for platoform-dependent path
    globdir = 'griddata/{}/daily/{}*/netcdf/'
    if sys.platform.startswith('win'):
        globdir = globdir.replace('/', '\\')

    for var, mode in product(['TabsD', 'TmaxD', 'TminD', 'R', 'msgSISD_'],
                             ['verified', 'operational']):

        # radiation not operational
        if (var == 'msgSISD_') and (mode == 'operational'):
            continue

        all_file = os.path.join(write_to, '{}_{}_all.nc'.format(var, mode))

        try:
            cirrus = utils.CirrusClient()
            r, _ = cirrus.sync_files(
                '/data/griddata', write_to,
                globpattern='*{}/daily/{}*[!swissgrid]/netcdf/*'.format(mode,
                                                                        var))
        except OSError:  # no VPN: try to get them from the FTP server...sigh!
            # list of retrieved
            r = []
            # extend write_to
            ftp_write_to = os.path.join(write_to, 'from_ftp')

            # delete all existing
            try:
                shutil.rmtree(ftp_write_to)
            except FileNotFoundError:
                pass
            os.mkdir(ftp_write_to)

            # retrieve
            ftp = utils.WSLSFTPClient()
            ftp_dir = '/data/ftp/map/raingrid/'
            # change directory and THEN(!) list (we only want file names)
            ftp.cwd(ftp_dir)
            files = ftp.list_content()
            klist = [k for k, v in ftp_key_dict.items() if v.startswith(var)]
            keys_verified = ['BZ14', 'BZ69', 'CZ91', 'CZ92', 'CZ93']
            if mode == 'operational':
                klist = [k for k in klist if k not in keys_verified]
            if mode == 'verified':
                klist = [k for k in klist if k in keys_verified]
            files_sel = [f for f in files if any(k in f for k in klist)]
            for fname in files_sel:
                try:
                    dl_file = os.path.join(ftp_write_to, fname)
                    ftp.get_file(fname, dl_file)
                    # extract and remove zipped
                    utils.unzip_file(dl_file)
                    os.remove(dl_file)

                    # sort directly
                    ftype = [ftp_key_dict[k] for k in klist if k in fname][0]

                    move_to_dir = glob(os.path.join(write_to,
                                                    globdir.format(mode,
                                                                   ftype)))[0]

                    # of course, exception...
                    if ftype == 'msgSISD_':
                        ftype = 'msg.SIS.D_'
                    extracted = glob(os.path.join(ftp_write_to, ftype + '*'))[0]
                    move_to_file = os.path.join(move_to_dir,
                                                os.path.basename(extracted))
                    try:
                        shutil.move(extracted, move_to_file)
                        r.append(move_to_file)
                    except PermissionError:  # file already there
                        pass
                except (FileNotFoundError, KeyError):  # ver not always there
                    pass

        # if at least one file was retrieved, assemble everything new
        flist = glob(os.path.join(write_to, (globdir + '*.nc').format(mode, var)))

        # do we want/need to create everything from scratch?
        if how == 'from_scratch' or ((not os.path.exists(all_file)) and len(flist) > 0):
        #if (len(r) > 0) or ((not os.path.exists(all_file)) and len(flist) > 0):
            # Instead of using open_mfdataset (we need a lot of preprocessing)
            log.info('Concatenating {} {} {} files...'
                     .format(len(flist), var, mode))
            sda = utils.read_multiple_netcdfs(flist, chunks={'time': 50},
                                              tfunc=utils._cut_with_CH_glac)
            # todo: throw an error if data are missing (e.g. files retrieved from Cirrus, pause, then only six from FTP)
            log.info('Ensuring time continuity...')
            sda = sda.crampon.ensure_time_continuity()
            sda.encoding['zlib'] = True
            sda.to_netcdf(all_file)
        elif how == 'update':
            log.info('Updating with {} {} {} files...'
                     .format(len(r), var, mode))
            old = utils.read_netcdf(all_file, chunks={'time': 50})
            new = utils.read_multiple_netcdfs(r, tfunc=utils._cut_with_CH_glac)
            sda = old.combine_first(new)
            # todo: do we need to ensure time continuity? What happens if file is missing?
            sda.encoding['zlib'] = True
            sda.to_netcdf(all_file)
        else:
            raise ValueError('Climate file creation creation mode {} is not '
                             'allowed.'.format(mode))

    # update operational with verified
    for var in ['TabsD', 'TmaxD', 'TminD', 'R', 'msgSISD_']:
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
    fstr = '*{}*_op_ver.nc'
    tfile = glob(os.path.join(write_to, fstr.format('TabsD')))[0]
    tminfile = glob(os.path.join(write_to, fstr.format('TminD')))[0]
    tmaxfile = glob(os.path.join(write_to, fstr.format('TmaxD')))[0]
    pfile = glob(os.path.join(write_to, fstr.format('R')))[0]
    rfile = glob(os.path.join(write_to, fstr.format('msgSISD_')))[0]
    outfile = cfg.PATHS['climate_file']

    log.info('Combining TEMP, TMIN, TMAX, PRCP, SIS and HGT...')
    utils.daily_climate_from_netcdf(tfile, tminfile, tmaxfile, pfile, rfile,
                                    hfile, outfile)
    log.info('Done combining TEMP, TMIN, TMAX, PRCP, SIS and HGT.')

    return outfile


# IMPORTANT: overwrite OGGM functions with same name:
process_custom_climate_data = process_custom_climate_data_crampon


class GlacierMeteo(object):
    """
    Interface to the meteorological data belonging to a glacier geometry.
    """

    def __init__(self, gdir):
        """
        Instantiate the GlacierMeteo object.

        It is ~15x faster to get all variables as arrays and then retrieve the
        index location of the desired date every time than getting is directly
        from the xarray object.

        Parameters
        ----------
        gdir: :py:class:`crampon.GlacierDirectory`
            The GlacierDirectory object for which the GlacierMeteo object
            shall be set up.
        """
        self.gdir = gdir
        self._heights = self.gdir.get_inversion_flowline_hw()[0]
        self.meteo = xr.open_dataset(self.gdir.get_filepath('climate_daily'))
        self.index = self.meteo.time.to_index()
        self.tmean = self.meteo.temp.values
        self.tmin = self.meteo.tmin.values
        self.tmax = self.meteo.tmax.values
        self.prcp = self.meteo.prcp.values
        self.sis = self.meteo.sis.values
        self.pgrad = self.meteo.pgrad.values
        self.tgrad = self.meteo.tgrad.values
        self.ref_hgt = self.meteo.ref_hgt
        self._days_since_solid_precipitation = None

    @property
    def heights(self):
        return self._heights

    @lazy_property
    def days_since_solid_precipitation(self, heights, min_amount=0.002,
                                      method=None, **kwargs):
        """
        Lazy-evaluated number of days since there was solid precipitation.


        Parameters
        ----------
        date: datetime.datetime
            Date for which to retrieve the days since last snowfall, including
            the date itself.
        heights: np.array
            Heights where the days since last snowfall shall be determined.
        min_amount: float
            Minimum amount of solid precipitation (m w.e.) used as threshold
            for a solid precipitation event. Default: 0.002 m.w.e. (corresponds
            to roughly 2 cm fresh snow height, a threshold used in Oerlemans
            1998 to characterize a snowfall event. Probably this value should
            be adjusted according to the error of the precipitation input).
        method: str
            Method to use for determination of solid precipitation. Allowed:
            "linear" and "magnusson". Default: None (from configuration).
        **kwargs: dict
            Keywords accepted be the chosen method to determine solid
            precipitation.

        Returns
        -------
        last_solid_day: float
            Days since last day with solid precipitation.
        """
        raise NotImplementedError
        dssp = np.ones((len(heights), len(self.tmean)))
        for i in range(len(self.tmean)):
            precip_s, _ = self.get_precipitation_liquid_solid(date, method=method,
                                                              **kwargs)
            where_precip_s = np.where(precip_s >= min_amount)
            dssp[i, where_precip_s] = 0

    def get_loc(self, date):
        """
        Get index location of a date.

        Parameters
        ----------
        date: datetime.datetime
            A date in the index

        Returns
        -------
        The location of the date in the index as integer.
        """
        return self.index.get_loc(date)

    def get_tmean_at_heights(self, date, heights=None):

        if heights is None:
            heights = self.heights

        date_index = self.get_loc(date)
        temp = self.tmean[date_index]
        tgrad = self.tgrad[date_index]
        temp_at_hgts = get_temperature_at_heights(temp, tgrad, self.ref_hgt,
                                                  heights)

        return temp_at_hgts

    def get_tmean_for_melt_at_heights(self, date, t_melt=None, heights=None):
        """
        Calculate the temperature for melt.

        By default, the inversion fowline heights from the GlacierDirectory
        are used, but heighst can also be supplied as keyword.

        Parameters
        ----------
        date: datetime.datetime
            Date for which the temperatures for melt shall be calculated.
        t_melt: float
            Temperature threshold when melt occurs (deg C).
        heights: np.array
            Heights at which to evaluate the temperature for melt. Default:
            None (take the heights of the GlacierDirectory inversion flowline).

        Returns
        -------
        np.array
            Array with temperatures above melt threshold, clipped to zero.
        """

        temp_at_hgts = self.get_tmean_at_heights(date, heights)
        above_melt = temp_at_hgts - t_melt
        return np.clip(above_melt, 0, None)

    def get_positive_tmax_sum_between(self, date1, date2, heights):
        """
        Calculate accumulated positive maximum temperatures since a given date.

        Used particularly for albedo ageing after Brock (2000).

        Parameters
        ----------
        date1: datetime.datetime
            Date since when the positive temperature sum shall be calculated.
        date2: datetime.datetime
            Date until when the positive temperature sum shall be calculated.
        heights: np.array
            Heights at which to get the positive maximum temperature sum.

        Returns
        -------
        pos_tsum: float
            Sum of positive maximum temperatures in the interval between date1
            and date2.
        """

        interval = self.meteo.sel(time=slice(date1, date2))
        interval_at_heights = get_temperature_at_heights(
            interval.tmax.values.T,
            interval.tgrad.values.T,
            self.ref_hgt,
            heights)
        pos_t = np.clip(interval_at_heights, 0, None)
        pos_tsum = np.nansum(pos_t, axis=0)

        return pos_tsum

    def get_mean_annual_temperature_at_heights(self, heights):
        """
        Get mean annual temperature at heights.

        The function uses a given temperature at a reference height and a
        linear temperature gradient approach.

        Parameters
        ----------
        heights: float, array-like xr.DataArray or xr.Dataset
            Heights where to get the annual mean temperature.

        Returns
        -------
        t_at_heights: same as input
            The temperature distributed to the given heights.
        """

        atemp = self.meteo.temp.resample(time='AS').mean(dim='time')
        agrad = self.meteo.tgrad.resample(time='AS').mean(dim='time')
        t_at_heights = get_temperature_at_heights(atemp, agrad, self.ref_hgt,
                                                  heights)
        return t_at_heights

    def get_mean_winter_temperature(self):
        """
        Calculate the mean temperature of the "winter" months.

        Winter months are defined as # Todo: define winter months
        This method should be used to model temperature penetration into the
        snow/firn pack after Carslaw (1959).

        Returns
        -------

        """

        raise NotImplementedError

    def get_mean_month_temperature(self):
        """
        Calculate the mean monthly temperature.

        Returns
        -------
        mmt: xr.DataArray
            The resampled monthly temperature
        """
        return self.meteo.temp.resample(time='MS').mean(dim='time')

    def get_precipitation_liquid_solid(self, date, heights=None, method=None,
                                       **kwargs):
        """
        Get solid and liquid fraction of precipitation.

        Parameters
        ----------
        date: datetime.datetime
            Date for which to retrieve liquid/solid precipitation.
        heights: np.array
            Heights for which to retrieve solid and liquid precipitation.
        method: str
            Method with which to calculate the liquid and solid shares of
            precipitation. Allowed methods: 'magnusson' (after
            [Magnusson et al. (2017)]_) and 'linear' (linear gradient between
            temperature thresholds). If None, method is determined from
            configuration file.
        **kwargs: keyword, value pairs, optional
            Keywords accepted by the method to determine the fraction of solid
            precipitation.
        Returns
        -------
        prcpsol, prcpliq: np.array, np.array
            Arrays with solid and liquid precipitation.

        References
        ----------
        .. [Magnusson et al. (2017)] https://doi.org/10.1002/2016WR019092
        """

        if heights is None:
            heights = self.heights
        if method is None:
            method = cfg.PARAMS['precip_ratio_method']

        date_index = self.get_loc(date)
        prcp = self.prcp[date_index]
        pgrad = self.pgrad[date_index]
        prcptot = get_precipitation_at_heights(prcp, pgrad, self.ref_hgt,
                                               heights)

        # important: we don't take compound interest formula (p could be neg!)
        prcptot = np.clip(prcptot, 0, None)
        tm_hgts = self.get_tmean_at_heights(date, heights)
        if method.lower() == 'magnusson':
            # todo: instead of calculatng the temp_at_heights again one should make temp_at_heights a kwarg (otherwis get_loc is called two times => cost-intensive)
            frac_solid = get_fraction_of_snowfall_magnusson(tm_hgts, **kwargs)
        elif method.lower() == 'linear':
            frac_solid = get_fraction_of_snowfall_linear(tm_hgts, **kwargs)
        else:
            raise ValueError('Solid precipitation fraction method not '
                             'recognized. Allowed are "magnusson" and '
                             '"linear".')
        prcpsol = prcptot * frac_solid
        prcpliq = prcptot - prcpsol

        return prcpsol, prcpliq


def get_temperature_at_heights(temp, grad, ref_hgt, heights):
    """
    Interpolate a temperature at a reference height to other heights.

    Parameters
    ----------
    temp: float, array-like or xarray.DataArray
        Temperature at the reference height (deg C or K).
    ref_hgt: float, array-like, xr.DataArray or xr.Dataset
        Reference height (m).
    grad: float, array-like or xarray.DataArray
        The temperature gradient to apply (K m-1).
    heights: np.array
        The heights where to interpolate the temperature to.

    Returns
    -------
    np.array
        The temperature at the input heights.
    """
    if isinstance(temp, (int, float, np.float32)):
        return np.ones_like(heights) * temp + grad * (heights - ref_hgt)
    else:
        try:
            return np.ones_like(heights) * np.array(temp[:, np.newaxis]) + \
                   np.array(grad[:, np.newaxis]) * (heights - ref_hgt)
        except IndexError:  # xarray objects
            return np.ones_like(heights) * temp.values[:, np.newaxis] + \
                   grad.values[:, np.newaxis] * (heights - ref_hgt)


def get_precipitation_at_heights(prcp, pgrad, ref_hgt, heights):
    """
    Interpolate precipitation at a reference height to other heights.

    Parameters
    ----------
    prcp: float, array-like or xarray.DataArray
        Precipitation at the reference height (deg C or K).
    ref_hgt: float, array-like or xarray.DataArray
        Reference height (m).
    grad: float, array-like or xarray.DataArray
        The precipitation gradient to apply (K m-1).
    heights: np.array
        The heights where to interpolate the precipitation to.

    Returns
    -------
    np.array
        The precipitation at the input heights.
    """
    if isinstance(prcp, (int, float, np.float32)):
        return np.ones_like(heights) * prcp + prcp * pgrad * (heights -
                                                              ref_hgt)
    else:
        try:
            return np.ones_like(heights) * np.array(prcp[:, np.newaxis]) + \
                   np.array(prcp[:, np.newaxis]) * \
                   np.array(pgrad[:, np.newaxis]) * (heights -ref_hgt)
        except IndexError:
            return np.ones_like(heights) * prcp.values[:, np.newaxis] + \
                   prcp.values[:, np.newaxis] * pgrad.values[:, np.newaxis] * \
                   (heights - ref_hgt)


def get_fraction_of_snowfall_magnusson(temp, t_base=0.54, m_rho=0.31):
    """
    Get the fraction of snowfall according to [Magnusson et al. (2017)]_.

    Parameters
    ----------
    temp: array-like
        Temperature during the snowfall event (deg C).
    t_base: float
        Threshold temperature below which precipitation mostly falls as snow.
        Default: 0.54 deg C (calibrated by [Magnusson et al. (2017)]_).
    m_rho: float
        Temperature that determines the temperature range for mixed
        precipitation. Default: 0.31 deg C(calibrated by
        [Magnusson et al. (2017)]_).

    Returns
    -------
    fs: float
        Fraction of snowfall for a given temperature.

    References
    ----------
    .. [Magnusson et al. (2017)] https://doi.org/10.1002/2016WR019092
    """

    t_rho = (temp - t_base) / m_rho
    fs = 1. / (1. + np.exp(t_rho))
    return fs


def get_fraction_of_snowfall_linear(temp, t_all_solid=None, t_all_liquid=None):
    """
    Get the fraction of snowfall following a linear gradient.

    Parameters
    ----------
    temp: array-like
        Temperature during the snowfall event (deg C).
    t_all_solid: float
        Threshold temperature below which precipitation falls as snow only.
        Default: None (retrieve from parameter file).
    t_all_liquid: float
        Threshold temperature above which precipitation falls as rain only.
        Default: None (retrieve from parameter file).

    Returns
    -------
    fs: float
        Fraction of snowfall for a given temperature.
    """
    if t_all_solid is None:
        t_all_solid = cfg.PARAMS['temp_all_solid']
    if t_all_liquid is None:
        t_all_liquid = cfg.PARAMS['temp_all_liq']
    fac = 1 - (temp - t_all_solid) / (t_all_liquid - t_all_solid)
    fs = np.clip(fac, 0, 1)
    return fs
