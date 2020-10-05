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
import datetime as dt

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


def interpolate_mean_temperature_uncertainty(month_number):
    """
    Interpolate the annual temperature uncertainty for temperature grids.

    Values are from [1]_ for the region "Alps". Actually, only numbers for DJF
    and JJA are given. Here, we suppose linear changes inbetween. The output is
    the mean absolute error in Kelvin. It's absolutely rough, but better than
    nothing.
    This is a table from an e-mail from Christoph Frei (09.04.2020),
    from which we take the errors. He writes that these values (Q84-Q61)/2.
    are somewhat smaller than the RMSE due to the "longtailedness" of the
    distribution. However, we still take them.

             YYY  DJF  MAM  JJA  SON
    Alps    1.08 1.67 0.94 0.85 1.09 K

    Parameters
    ----------
    month_number: np.array
        Month for which to give an estimate of interpolated temperature
        uncertainty.

    Returns
    -------
    np.array
        Temperature mean absolute error for the respective month(s) (K).

    References
    ----------
    .. [1] Frei, C. (2014), Interpolation of temperature in a mountainous
           region using nonlinear profiles and non‐Euclidean distances. Int. J.
           Climatol., 34: 1585-1605. doi:10.1002/joc.3786
    """

    mae = np.array([1.67, 1.67, 1.67, 0.94, 0.94, 0.94, 0.85, 0.85, 0.85,
                    1.09, 1.09, 1.09])
    return mae[month_number.astype(int) - 1]


def interpolate_mean_precipitation_uncertainty(prcp_quant):
    """
    Interpolate the annual temperature uncertainty for precipitation grids.

    Values have been read manually from fig. 4 in _[1] as advised by Christoph
    Frei (Mail 2020-04-09). We ignore the systematic error, since it is covered
    by the calibration of the precipitation correction factor, the calibration
    of the snow redistribution factor and the systematic variation by [2]_.
    The uncertainty value returned is half the distance between Q25 and Q75 of
    the distributions and is used later on in the
    `py:class:crampon.core.preprocessing.climate.GlacierMeteo` to produce a
    multiplicative factor to correct for the random error of precipitation.

    Parameters
    ----------
    prcp_quant: float or array-like
        Quantile of the precipitation in the total precipitation distribution.

    Returns
    -------
    same as input
        Precipitation standard error as a factor to multiply with the
        precipitation.

    References
    ----------
    .. [1] Isotta, F.A., Frei, C., Weilguni, V., Perčec Tadić, M., Lassègues,
           P., Rudolf, B., Pavan, V., Cacciamani, C., Antolini, G.,
           Ratto, S.M., Munari, M., Micheletti, S., Bonati, V., Lussana, C.,
           Ronchi, C., Panettieri, E., Marigo, G. and Vertačnik, G. (2014),
           The climate of daily precipitation in the Alps: development and
           analysis of a high‐resolution grid dataset from pan‐Alpine
           rain‐gauge data. Int. J. Climatol., 34: 1657-1675.
           doi:10.1002/joc.3794
    .. [2] Sevruk, B. (1985): Systematischer Niederschlagsmessfehler in der
           Schweiz. In: Sevruk, B. (1985): Der Niederschlag in der Schweiz.
           Geographischer Verlag Kuemmerly und Frey, Bern.
    """

    # values read from Isotta et al. (2014)
    q = [0.1, 0.3, 0.5, 0.7, 0.85, 0.925, 0.965, 0.985, 0.995]
    values_djf = [0.4, 0.35, 0.275, 0.24, 0.19, 0.18, 0.19, 0.15, 0.16]
    values_jja = [0.6, 0.37, 0.3, 0.17, 0.19, 0.19, 0.17, 0.16, 0.15]
    # We forget the seasonal dependency for the first
    values_mean = (np.array(values_djf) + np.array(values_jja)) / 2.

    indices = np.array(
        [pd.Index(q).get_loc(n, method='nearest') for n in prcp_quant])
    return values_mean[indices]


def interpolate_shortwave_rad_uncertainty(month_number):
    """
    Interpolate the shortwave radiation uncertainty for radiation grids.

    Values are from [1]_ (p.76). At the moment, this value is constant over the
    entire year. [1]_ says something around 20 W m-2 (mean absolute bias), so
    we choose 10 W m-2.
    # todo: 10 is not a good assumption - how to transform MAB into stdev?

    Parameters
    ----------
    month_number: np.array
        Month for which to give an estimate of interpolated temperature
        uncertainty.

    Returns
    -------
    np.array
        Radiation mean absolute error for the respective month(s) (W m-2).

    References
    ----------
    [1]_.. : Stöckli, R.: The HelioMont Surface Solar Radiation Processing.
             MeteoSwiss, 2013.
    """

    return np.ones_like(month_number) * 10.


def prcp_fac_annual_cycle(doy):
    """
    Interpolate the annual cycle of the precipitation correction factor.

    Values are from [1]_, figure 3.1.3 (above 2000 m.a.s.l.). The returned
    value is an *additional factor* that should be multiplied with the
    precipitation correction factor. E.g., when the mean precipitation
    correction factor is 1.25, then this additional factor would let it vary
    between 1.35 (summer) and 1.15 (winter). These additional correction are
    not (yet) captured in the RhiresD product.

    Parameters
    ----------
    doy: int or np.array
        Day(s) of year for which to give an estimate of the additional annual
        precipitation correction factor variability. Here, we take the real
        DOY, not the mass budget DOY.

    Returns
    -------
    np.array
        Temperature mean absolute error for the respective month(s).

    References
    ----------
    [1]_.. : Sevruk, B. (1985): Systematischer Niederschlagsmessfehler in
             der Schweiz. In: Sevruk, B. (1985): Der Niederschlag in der
             Schweiz. Geographischer Verlag  Kuemmerly und Frey, Bern, p.69
    """

    # -90 because the curve starts beginning of April
    # 0.08 = 1.35/1.25 (max/mean)
    return 0.08 * - np.sin((doy - 90.) * (2 * np.pi / 365.)) + 1.


def bias_correct_and_add_geosatclim(gdir: utils.GlacierDirectory,
                                    gsc: xr.DataArray,
                                    diff: xr.DataArray) -> None:
    """
    Bias-correct the radiation processed with Geosatclim to match Heliomont.

    Note: this is only a temporal solution until there is a new version of
    Geosatclim.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory for which to bias-correct Geosatclim.
    gsc: xr.DataArray
        DataArray containing the shortwave incoming solar radiation over
        Switzerland.
    diff: xr.DataArray
        Mean difference between Heliomont and Geosatclim during the common time
        period.
    """
    gsc_glacier = gsc.sel(lat=gdir.cenlat, lon=gdir.cenlon, method='nearest')

    # fit a polynom of 7th degree (seems to work better than trigonometric)
    p7 = np.poly1d(np.polyfit(np.arange(366), diff, 7))

    # correct the values grouped by DOY
    result_list = []
    for i, g in list(gsc_glacier.groupby('time.dayofyear')):
        result_list.append(g - (p7(i - 1)))
    result_ds = xr.concat(result_list, dim='time').sortby('time')

    # overwrite SIS in climate file
    climate_file = gdir.get_filepath('climate_daily')
    with xr.open_dataset(climate_file) as clim:
        clim_copy = clim.copy(deep=True)
    clim_copy.load()
    # trust Heliomont more
    # todo: this processing takes ages. Any better solution? Rechunking?
    combo_clim = clim_copy.combine_first(result_ds.to_dataset(name='sis'))
    combo_clim.load()
    combo_clim.to_netcdf(climate_file)


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
    out = {'baseline_climate_source': fpath, 'baseline_hydro_yr_0': y0 + 1,
           'baseline_hydro_yr_1': y1}
    gdir.write_json(out, 'climate_info')


@entity_task(log, writes=['spinup_climate_daily'])
def process_spinup_climate_data(gdir):
    """
    Process homogenized gridded data before 1961 into a spinup climate.
    """
    # todo: double code with climate processing: how to do a wrapper function?
    fpath = os.path.join(cfg.PATHS['climate_dir'],
                         cfg.BASENAMES['spinup_climate_daily'])
    if not os.path.exists(fpath):
        raise IOError('Spinup climate file {} not found'.format(fpath))

    # read the file
    nc_ts = xr.open_dataset(fpath)

    # geoloc
    lon = nc_ts.coords['lon'].values
    lat = nc_ts.coords['lat'].values

    # Gradient defaults
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    def_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    use_pgrad = cfg.PARAMS['prcp_use_local_gradient']
    def_pgrad = cfg.PARAMS['prcp_default_gradient']
    pg_minmax = cfg.PARAMS['prcp_local_gradient_bounds']

    # get closest grid cell and index
    ilon = np.argmin(np.abs(lon - gdir.cenlon))
    ilat = np.argmin(np.abs(lat - gdir.cenlat))
    ref_pix_lon = lon[ilon]
    ref_pix_lat = lat[ilat]

    # Some special things added in the crampon function
    #iprcp, itemp, igrad, ihgt = utils.joblib_read_climate(fpath, ilon, ilat,
    #                                                      def_grad, g_minmax,
    #                                                      use_grad, def_pgrad,
    #                                                      pg_minmax, use_pgrad)

    local_climate = nc_ts.isel(dict(lat=ilat, lon=ilon))
    iprcp = local_climate.prcp
    itemp = local_climate.temp
    ihgt = local_climate.hgt

    if use_grad != 0:
        itgrad = utils.get_tgrad_from_window(nc_ts, ilat, ilon, use_grad,
                                       def_grad, g_minmax)
    else:
        itgrad = np.zeros(len(nc_ts.time)) + def_grad
    if use_pgrad != 0:
        ipgrad = utils.get_pgrad_from_window(nc_ts, ilat, ilon, use_pgrad,
                                       def_pgrad, pg_minmax)
    else:
        ipgrad = np.zeros(len(nc_ts.time)) + def_pgrad

    # Set temporal subset for the ts data depending on frequency:
    time = [pd.Timestamp(t) for t in nc_ts.time.values]
    if pd.infer_freq(time) == 'D':  # day start frequency
        # Ok, this is NO ERROR: we can use the function
        # ``write_monthly_climate_file`` also to produce a daily climate file:
        # there is no reference to the time in the function! We should just
        # change the ``file_name`` keyword!
        gdir.write_monthly_climate_file(time, iprcp, itemp, itgrad, ipgrad,
                                        ihgt, ref_pix_lon, ref_pix_lat,
                                        file_name='spinup_climate_daily',
                                        time_unit=nc_ts.time.encoding['units'])
    else:
        raise NotImplementedError('Climate data frequency not yet understood')


def make_spinup_climate_file(write_to=None, hfile=None, which=1901):
    """
    Process homogenized gridded data before 1961 [1]_ into a spinup climate.

    Parameters
    ----------
    write_to: str or None
        Directory where the output file should be written to. Default: the
        'climate_dir' in the crampon configuration PATHS dictionary.
    hfile: str or None
        Path to a netCDF file containing a DEM of the area (used for assembling
        the file that OGGM likes. Needs to cover the same area in the same
        extent ans resolution as the meteo files. Default: the 'hfile' in the
        crampon configuration PATHS dictionary.
    which: int
        The begin year of the spinup data. The are two version: one begins 1864
        and is based on 20 homogenized stations for temperature (17 for
        precipitation), the other begins 1901 and is based on 28 homogenized
        stations for temperature (69 for precipitation). Default: 1901.

    References
    ----------
    .. [1] : Isotta, F. A., Begert, M., & Frei, C. ( 2019). Long‐term
             consistent monthly temperature and precipitation grid data sets
             for Switzerland over the past 150 years. Journal of Geophysical
             Research: Atmospheres, 124, 3783– 3799.
             https://doi.org/10.1029/2018JD029910
    .. [2] : https://www.meteoschweiz.admin.ch/home/klima/schweizer-klima-im-detail/doc/Prod_rec.pdf
    """

    # todo: this is double code with make_climate_file: how to remove?
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

    outfile = os.path.join(write_to, cfg.BASENAMES['spinup_climate_daily'])

    globdir = 'griddata/reconstruction/*{}/'
    if sys.platform.startswith('win'):
        globdir = globdir.replace('/', '\\')

    # for this, a VPN to Cirrus must exist, the files are not on the FTP
    try:
        cirrus = utils.CirrusClient()
        for var in ['RrecabsM', 'TrecabsM']:
            _, _ = cirrus.sync_files(
                '/data/griddata', write_to,
                globpattern='*{}*.nc'.format(var))
    except OSError:  # this is almost always sufficient
        log.info('Reconstructed grids were not freshly retrieved from '
                 'Cirrus. Continuing with old files...')
        pass

    to_merge = glob(
        os.path.join(write_to, (globdir + '*.nc').format(str(which))))
    temp = xr.open_dataset([t for t in to_merge if 'TrecabsM' in t][0],
                             decode_times=False)
    prec = xr.open_dataset([r for r in to_merge if 'RrecabsM' in r][0],
                             decode_times=False)

    # drop some MeteoSwiss stuff
    prec = prec.drop(['dummy', 'longitude_latitude'])
    temp = temp.drop(['dummy', 'longitude_latitude'])

    # work on calendar - cftime can't read "months since" for "standard" cal.
    def _fix_time(ds):
        old_units = ds.time.units
        calendar_bgstr = ds.time.units.split(' ')[2]
        calendar_bgyear = int(calendar_bgstr.split('-')[0])
        calendar_endstr = '{}-01-31'.format(
            str(int(calendar_bgyear + np.ceil(max(ds.time) / 12))))
        # take "2SM" to get the 15th of every month. Not perfect, but there is
        # no 'previous' interpolation method yet in xarray (but in scipy!)
        # todo: Switch '2SM' back to 'MS' when interpolate 'previous' available
        ds['time'] = pd.date_range(calendar_bgstr, calendar_endstr,
                                   freq='2SM')[ds.time.values.astype(int)]
        ds.time.encoding['units'] = old_units.replace('months', 'days')
        return ds

    # we assume their structure is the same
    prec = _fix_time(prec)
    temp = _fix_time(temp)

    hgt = utils.read_netcdf(hfile)
    _, hgt = xr.align(temp, hgt, join='left')

    if 'TrecabsM{}'.format(str(which)) in temp.variables:
        temp = temp.rename({'TrecabsM{}'.format(str(which)): 'temp'})
    if 'RrecabsM{}'.format(str(which)) in prec.variables:
        prec = prec.rename({'RrecabsM{}'.format(str(which)): 'prcp'})

    month_length = xr.DataArray(utils.get_dpm(prec.time.to_index(),
                                              calendar='standard'),
                                coords=[prec.time], name='month_length')
    # generate "daily precipitation"
    prec = prec / month_length

    # resample artificially to daily - no 'previous' available yet
    old_t_units = prec.time.units
    # todo: replace 'nearest' with 'previous' when available
    prec = prec.interp(time=pd.date_range(pd.Timestamp(min(prec.time).item()) -
                                          pd.tseries.offsets.MonthBegin(1),
                                          pd.Timestamp(max(prec.time).item()) +
                                          pd.tseries.offsets.MonthEnd(1)),
                       method='nearest', kwargs={'fill_value':'extrapolate'})
    temp = temp.interp(time=pd.date_range(pd.Timestamp(min(temp.time).item()) -
                                          pd.tseries.offsets.MonthBegin(1),
                                          pd.Timestamp(max(temp.time).item()) +
                                          pd.tseries.offsets.MonthEnd(1)),
                       method='nearest', kwargs={'fill_value':'extrapolate'})
    nc_ts = xr.merge([temp, prec, hgt])
    nc_ts = nc_ts.sel(time=slice(None, '1960-12-31'))
    nc_ts.time.encoding['units'] = old_t_units  # no way to keep when interp

    # ensure it's compressed when exporting
    nc_ts.encoding['zlib'] = True
    nc_ts.to_netcdf(outfile)


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
    utils.climate_files_from_netcdf(tfile, pfile, hfile, outfile, tminfile,
                                    tmaxfile, rfile)
    log.info('Done combining TEMP, TMIN, TMAX, PRCP, SIS and HGT.')

    return outfile


def make_nwp_file(write_to=None):
    """
    Compile the numerical weather prediction file for mass balance predictions.

    # todo: make the prediction variable names exactly the same as in the climate file

    Parameters
    ----------

    Returns
    -------

    """
    # todo: manage to get an environment with cfgrib running
    # todo: actually, compared to the other functions above, this function should distribute an existing nwp.nc to all gdirs - the processing below should happen before

    if not write_to:
        try:
            write_to = cfg.PATHS['climate_dir']
        except KeyError:
            raise KeyError('Must supply write_to or initialize the crampon'
                           'configuration.')

    out_file = os.path.join(write_to, 'cosmo_predictions.nc')
    cosmo_dir = os.path.join(write_to, 'cosmo')

    # current midnight modelrun
    midnight = dt.datetime.now().replace(hour=0, minute=0, second=0,
                                         microsecond=0)
    run_hour = '00'
    run_day = midnight.strftime('%y%m%d')  # year w/o century for c7

    # COSMO-7
    ftp = utils.WSLSFTPClient()
    ftp_dir = '/data/ftp/map/ezdods/almo7'
    # we are not allowed to change directory
    files = ftp.list_content(ftp_dir)
    cfiles_c7 = [f for f in files if ((run_day + run_hour in f) and
                                      f.endswith('.tgz'))]
    for cfile in cfiles_c7:
        ftp.get_file(cfile, os.path.join(cosmo_dir, os.path.basename(cfile)))
    ftp.close()

    # COSMO-1 and COSMO-E
    ftp = utils.WSLSFTPClient(user='hyv-data')
    ftp_dir = '/data/ftp/hyv_data/cosmo/cosmo1'
    files = ftp.list_content(ftp_dir)
    cfiles_c1 = [f for f in files if (('SZ90' in f) and
                                      ('{}{}'.format(run_day, run_hour) in f)
                                      and f.endswith('.zip'))]
    for cfile in cfiles_c1:
        ftp.get_file(cfile, os.path.join(cosmo_dir, os.path.basename(cfile)))
    ftp_dir = '/data/ftp/hyv_data/cosmo/cosmoe'
    files = ftp.list_content(ftp_dir)
    cfiles_ce = [f for f in files if (('SZ91' in f) and
                                      ('{}{}'.format(run_day, run_hour) in f)
                                      and f.endswith('.zip'))]
    for cfile in cfiles_ce:
        ftp.get_file(cfile, os.path.join(cosmo_dir, os.path.basename(cfile)))
    ftp.close()
    cfiles_c_1_e = cfiles_c1 + cfiles_ce

    to_untargz = [os.path.join(cosmo_dir, os.path.basename(r)) for r in cfiles_c7]
    to_unzip = [os.path.join(cosmo_dir, os.path.basename(r)) for r in
                  cfiles_c_1_e]

    for dl_zipfile in to_unzip:
        utils.unzip_file(dl_zipfile)
        os.remove(dl_zipfile)
    for dl_targzfile in to_untargz:
        utils.untargz_file(dl_targzfile)
        os.remove(dl_targzfile)

    # COSMO-1
    # this gets 10m wind speed (ws), 10m wind direction (p3031), t2m, d2m
    cosmo1_instant = xr.open_dataset(os.path.join(cosmo_dir, 'wsl_cosmo1_hydro-ch'),
                           engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'heightAboveGround',
                               'stepType': 'instant'}})
    cosmo1_instant_resampled = cosmo1_instant.resample(step='1D').mean()
    # this gets total precipitation (tp)
    cosmo1_accum = xr.open_dataset(os.path.join(cosmo_dir, 'wsl_cosmo1_hydro-ch'),
                           engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'accum'}})
    cosmo1_accum_resampled = cosmo1_accum.resample(step='1D').sum()
    cosmo1 = xr.merge([cosmo1_instant_resampled, cosmo1_instant_resampled])

    # COSMO-7
    # this gets albedo, total cloud cover (0-1) and net shortwave rad. flux (not all times)
    cosmo7_sinstant = xr.open_dataset(os.path.join(cosmo_dir, '{}{}_955'.format(run_day, run_hour), 'grib', 'iaceth7_00000000'),
                           engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'surface',
                               'stepType': 'instant'}})
    # this gets 2m temperature
    cosmo7_aginstant = xr.open_dataset(os.path.join(cosmo_dir, 'iaceth7_03000000'),
                           engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'heightAboveGround',
                               'stepType': 'instant'}})
    # this gets total precipitation (tp)
    cosmo7_saccum = xr.open_dataset(os.path.join(cosmo_dir, 'iaceth7_00000000'),
                           engine='cfgrib', backend_kwargs={
            'filter_by_keys': {'typeOfLevel': 'surface', 'stepType': 'accum'}})

    # COSMO-E
    # 20 runs + control
    # open_mfdataset doesn't work yet
    run_list_instant = []
    run_list_accum = []

    ce_globpath = '**/*wsl_cosmo-e_hydro-ch_*'
    if sys.platform.startswith('win'):
        ce_globpath = ce_globpath.replace('/', '\\')
    ce_paths = glob(cosmo_dir + ce_globpath,
                    recursive=True)
    for ce_p in ce_paths:
        run_list_instant.append(
            # gets 10m wind speed (ws), 10m wind direction (p3031) , t2m, d2m
            xr.open_dataset(ce_p, engine='cfgrib', backend_kwargs={
                'filter_by_keys': {'typeOfLevel': 'heightAboveGround',
                                   'stepType': 'instant'}})
        )

        run_list_accum.append(
            # gets total precipitation (tp)
            xr.open_dataset(ce_p, engine='cfgrib', backend_kwargs={
                'filter_by_keys': {'typeOfLevel': 'surface',
                                   'stepType': 'accum'}})
        )

    # make one file each
    ce_i_all = xr.concat(run_list_instant,
                         pd.Index(np.arange(len(run_list_instant)),
                                  name='run'))
    ce_a_all = xr.concat(run_list_accum,
                         pd.Index(np.arange(len(run_list_accum)), name='run'))

    # variables are from 00 - 00; let's make 0-23 a day and waste the last bit
    ce_i_all_r = ce_i_all.resample(time="1D").mean()
    ce_a_all_r = ce_a_all.resample(time="1D").sum()

    # rename variables to make the crampon suitable
    ce_i_all_r.rename(name_dict={'t2m': 'tmean'}, inplace=True)
    ce_a_all_r.rename(name_dict={'tp': 'prcp'}, inplace=True)

    # merge the variables
    cosmo_ds = xr.merge([ce_i_all_r, ce_a_all_r])

    # make daily means/sums/max/mins for everything
    # merge everything together in one xr.Dataset
    # write out as cosmo_predictions.nc that the mass balance prediction can access it
    cosmo_ds.to_netcdf(cfg.PATHS['nwp_file'])

    # remove the previous files that were downloaded
    second_last_midnight_run = midnight - pd.Timedelta(days=1)
    remove_date_key = second_last_midnight_run.strftime('%y%m%d')
    to_remove = glob(os.path.join(cfg.PATHS['climate_dir'],
                                  'cosmo*{}*'.format(remove_date_key)),
                     recursive=True)
    for tr in to_remove:
        try:
            os.remove(tr)
        except FileNotFoundError:
            pass


# IMPORTANT: overwrite OGGM functions with same name:
process_custom_climate_data = process_custom_climate_data_crampon


@entity_task(log, fallback=process_custom_climate_data)
def update_climate(gdir, clim_all=None):
    """
    Update the climate for a GlacierDirectory.

    This is for the case when there is already a climate file for every
    GlacierDirectory presents and it should just be updated.

    todo: Sooner or later we should implement a better handling of missing
          values etc. or implement a nightly task that does interpolation etc.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to update the climate file for.
    clim_all: xr.Dataset
        The climate file used to

    Returns
    -------
    None
    """
    if clim_all is None:
        need_close = True
        clim_all = xr.open_dataset(cfg.PATHS['climate_file'])
    else:
        need_close = False
    last_day_clim = clim_all.time[-1]
    # todo: radiation still not operational -> take last 62 days to be sure
    last_day = last_day_clim - pd.Timedelta(days=62)

    use_tgrad = cfg.PARAMS['temp_use_local_gradient']
    def_tgrad = cfg.PARAMS['temp_default_gradient']
    tg_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    use_pgrad = cfg.PARAMS['prcp_use_local_gradient']
    def_pgrad = cfg.PARAMS['prcp_default_gradient']
    pg_minmax = cfg.PARAMS['prcp_local_gradient_bounds']

    gclim = xr.open_dataset(gdir.get_filepath('climate_daily'))
    #last_day = gclim.time[-1]

    if last_day < last_day_clim:
        clim_all_sel = clim_all.sel(dict(lat=gclim.ref_pix_lat,
                                         lon=gclim.ref_pix_lon,
                                         time=slice(last_day,
                                                    last_day_clim)))
        clim_all_tsel = clim_all.sel(time=slice(last_day, last_day_clim))
        # todo: save ilat/ilon as gdir attributes!?
        # todo: save tgrad/pgrad in climate_all.nc? (is vectorizing it faster?)
        # we need to add tgrad and pgrad (not in climate file)
        ilon = np.argmin(np.abs(clim_all.lon - gdir.cenlon)).item()
        ilat = np.argmin(np.abs(clim_all.lat - gdir.cenlat)).item()
        clim_all_sel['tgrad'] = (['time'], utils.get_tgrad_from_window(
            clim_all_tsel, ilat=ilat, ilon=ilon, win_size=use_tgrad,
            default_tgrad=def_tgrad, minmax_tgrad=tg_minmax))
        clim_all_sel['pgrad'] = (['time'], utils.get_pgrad_from_window(
            clim_all_tsel, ilat=ilat, ilon=ilon, win_size=use_pgrad,
            default_pgrad=def_pgrad, minmax_pgrad=pg_minmax))
        updated = gclim.combine_first(clim_all_sel)
        gclim.close()
        updated.to_netcdf(gdir.get_filepath('climate_daily'))

    if need_close is True:
        clim_all.close()


class GlacierMeteo(object):
    """
    Interface to the meteorological data belonging to a glacier geometry.
    """

    def __init__(self, gdir, filename='climate_daily'):
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
        self.meteo = xr.open_dataset(self.gdir.get_filepath(filename))
        self.index = self.meteo.time.to_index()
        self.tmean = self.meteo.temp.values
        self.prcp = self.meteo.prcp.values
        self.pgrad = self.meteo.pgrad.values
        self.tgrad = self.meteo.tgrad.values
        self.ref_hgt = self.meteo.ref_hgt
        self._days_since_solid_precipitation = None
        if filename == 'climate_daily':
            self.tmin = self.meteo.tmin.values
            self.tmax = self.meteo.tmax.values
            self.sis = self.meteo.sis.values
    @property
    def heights(self):
        return self._heights

    @lazy_property
    def data_quantiles(self, var_name):
        """ Precipitation distribution quantiles to estimate the error."""
        return self.meteo[var_name].quantiles(np.arange(0, 1.01, 0.01),
                                              dim='time')

    @lazy_property
    def prcp_corr_annual_cycle(self):
        """
        Precipitation corrected by the annual cycle of the correction factor.
        """
        return prcp_fac_annual_cycle(self.meteo['time.dayofyear']) * self.prcp

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

        if type(date) in [list, pd.core.indexes.datetimes.DatetimeIndex]:
            return np.where(self.index.isin(date))
        else:
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

        By default, the inversion flowline heights from the GlacierDirectory
        are used, but heights can also be supplied as keyword.

        Parameters
        ----------
        date: datetime.datetime
            Date for which the temperatures for melt shall be calculated.
        t_melt: float
            Temperature threshold when melt occurs (deg C). Default: None
            (parse from configuration).
        heights: np.array
            Heights at which to evaluate the temperature for melt. Default:
            None (take the heights of the GlacierDirectory inversion flowline).

        Returns
        -------
        np.array
            Array with temperatures above melt threshold, clipped to zero.
        """

        if t_melt is None:
            t_melt = cfg.PARAMS['temp_melt']

        if heights is None:
            heights = self.heights.copy()

        temp_at_hgts = self.get_tmean_at_heights(date, heights)
        # todo: simplify this by clipping (t_melt, None) or using t[t lt t_melt] = 0.
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
