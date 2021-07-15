""" Prepare the meteodata from netCDF4 """

from __future__ import division
from typing import Optional
import os
from glob import glob
import crampon.cfg as cfg
from crampon import utils
from crampon.utils import lazy_property
from scipy.stats import percentileofscore
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
import numba
import cfgrib

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

    def get_reference_value(self, shppath=None):
        """
        Get a reference value for a polygon.

        Parameters
        ----------
        shppath: str
            Path to shapefile for whose polygon the reference value shall be
            retrieved.
        """
        shape = salem.read_shapefile(shppath, cached=True)
        centroid = shape.centroid

        raise NotImplementedError()

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

    Returns
    -------
    None
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
    # todo: all assignments necessary to avoid PermissionError?
    with xr.open_dataset(climate_file) as clim:
        clim_copy = clim.copy(deep=True)
    clim_copy.load()
    clim.close()
    clim = None
    # trust Heliomont more
    # todo: this processing takes ages. Any better solution? Rechunking?
    combo_clim = clim_copy.combine_first(result_ds.to_dataset(name='sis'))
    combo_clim.load()
    combo_clim.to_netcdf(climate_file)


# This writes 'climate_monthly' in the original version (doesn't fit anymore)
@entity_task(log, writes=['climate_info'])
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
    # todo: replace ilon&ilat as floats with ilon/ilat as an intersection of
    #    the grid an the glacier shape
    # nc_ts.set_roi(gdir.get_filepath('outlines'))

    ilon = np.argmin(np.abs(lon - gdir.cenlon))
    ilat = np.argmin(np.abs(lat - gdir.cenlat))
    ref_pix_lon = lon[ilon]
    ref_pix_lat = lat[ilat]

    # Some special things added in the crampon function
    iprcp, itemp, itmin, itmax, isis, itgrad, itgrad_unc, ipgrad, ipgrad_unc, ihgt = \
        utils.joblib_read_climate(fpath, ilon, ilat, def_tgrad, tg_minmax,
                                  use_tgrad, def_pgrad, pg_minmax, use_pgrad)

    # Set temporal subset for the ts data depending on frequency:
    # hydro years if monthly data, else no restriction
    yrs = nc_ts.time.year
    y0, y1 = yrs[0], yrs[-1]
    time = nc_ts.time

    # get the uncertainties according to the functions describing them
    month_numbers = np.array([i.month for i in time])
    prcp_qtls = np.array([percentileofscore(iprcp, p)/100. for p in
                          iprcp.values])
    # todo: the uncertainty values should not be used for the COSMO predictions
    temp_sigma = xr.DataArray(
        interpolate_mean_temperature_uncertainty(month_numbers),
        coords=itemp.coords, dims=itemp.dims, name='temp_sigma')
    prcp_sigma = xr.DataArray(
        interpolate_mean_precipitation_uncertainty(prcp_qtls),
        coords=iprcp.coords, dims=iprcp.dims, name='prcp_sigma')
    sis_sigma = xr.DataArray(
        interpolate_shortwave_rad_uncertainty(month_numbers),
        coords=isis.coords, dims=isis.dims, name='prcp_sigma')
    # todo: what to do with tmin, tmax and the gradients?

    if pd.infer_freq(nc_ts.time) == 'MS':  # month start frequency
        nc_ts.set_period(t0='{}-10-01'.format(y0), t1='{}-09-01'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, itgrad, ipgrad,
                                        ihgt, ref_pix_lon, ref_pix_lat,
                                        tmin=itmin, tmax=itmax,
                                        tgrad_sigma=itgrad_unc, sis=isis,
                                        temp_sigma=temp_sigma,
                                        prcp_sigma=prcp_sigma,
                                        sis_sigma=sis_sigma,
                                        pgrad_sigma=ipgrad_unc
                                        )
    elif pd.infer_freq(nc_ts.time) == 'M':  # month end frequency
        nc_ts.set_period(t0='{}-10-31'.format(y0), t1='{}-09-30'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, itgrad, ipgrad,
                                        ihgt, ref_pix_lon, ref_pix_lat,
                                        tmin=itmin, tmax=itmax,
                                        tgrad_sigma=itgrad_unc, sis=isis,
                                        temp_sigma=temp_sigma,
                                        prcp_sigma=prcp_sigma,
                                        sis_sigma=sis_sigma,
                                        pgrad_sigma=ipgrad_unc)
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
                                        tmin=itmin, tmax=itmax,
                                        tgrad_sigma=itgrad_unc, sis=isis,
                                        temp_sigma=temp_sigma,
                                        prcp_sigma=prcp_sigma,
                                        sis_sigma=sis_sigma,
                                        pgrad_sigma=ipgrad_unc,
                                        file_name='climate_daily',
                                        time_unit=nc_ts._nc.variables['time']
                                        .units)
    else:
        raise NotImplementedError('Climate data frequency not yet understood')

    hs = xr.open_mfdataset(
        os.path.join(cfg.PATHS['climate_dir'],
                     'griddata', 'verified', 'daily',
                     'msgSISD_daily_global_radiation', 'netcdf',
                     'msg.SIS.D_ch02.lonlat_20*.nc'),
        combine='by_coords')
    hs = hs.SIS
    gsc = xr.open_mfdataset(
        os.path.join(cfg.PATHS['climate_dir'],
                     'griddata', 'verified', 'daily',
                     'msgSISD_daily_global_radiation', 'netcdf',
                     'Geosatclim', '*.nc'),
        combine='by_coords')
    gsc = gsc.SIS

    # todo: shall we bias-correct with the mean bias over all Switzerland or
    #  with the pixel-specific bias? Pixel-specific delivers the best result...
    diff = (gsc.mean(['lat', 'lon']) - hs.mean(
        ['lat', 'lon'])).load().groupby('time.dayofyear').mean().values
    # hs_glacier = hs.sel(lat=gdir.cenlat, lon=gdir.cenlon, method='nearest')
    # diff = (gsc_glacier.SIS - hs_glacier.SIS).groupby('time.dayofyear')
    # .mean(skipna=True).values

    # todo: this should be removed when we have the final Geosatclim version
    bias_correct_and_add_geosatclim(gdir, gsc, diff)

    # metadata
    out = {'baseline_climate_source': fpath, 'baseline_hydro_yr_0': y0 + 1,
           'baseline_hydro_yr_1': y1}
    gdir.write_json(out, 'climate_info')


@entity_task(log, writes=['climate_spinup'])
def process_spinup_climate_data(gdir):
    """
    Process homogenized gridded data before 1961 into a spinup climate.
    """
    # todo: double code with climate processing: how to do a wrapper function?
    fpath = os.path.join(cfg.PATHS['climate_dir'],
                         cfg.BASENAMES['climate_spinup'])
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
    # iprcp, itemp, igrad, ihgt = utils.joblib_read_climate(
    #     fpath, ilon, ilat, def_grad, g_minmax, use_grad, def_pgrad,
    #     pg_minmax, use_pgrad)

    local_climate = nc_ts.isel(dict(lat=ilat, lon=ilon))
    iprcp = local_climate.prcp
    itemp = local_climate.temp
    ihgt = local_climate.hgt

    if use_grad != 0:
        itgrad = utils.get_tgrad_from_window(
            nc_ts, ilat, ilon, use_grad, def_grad, g_minmax)
    else:
        itgrad = np.zeros(len(nc_ts.time)) + def_grad
    if use_pgrad != 0:
        ipgrad = utils.get_pgrad_from_window(
            nc_ts, ilat, ilon, use_pgrad, def_pgrad, pg_minmax)
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


@entity_task(log, writes=['nwp_daily_cosmo', 'nwp_daily_ecmwf'])
def process_nwp_data(gdir: utils.GlacierDirectory):
    """Distributes the data from the numerical weather prediction to glaciers.

    Parameters
    ----------
    gdir : utils.GlacierDirectory
        GlacierDirectory to process the prediction for.

    Returns
    -------
    None
    """

    for pred_suffix in ['_cosmo', '_ecmwf']:
        # read the file
        fpath = cfg.PATHS['nwp_file' + pred_suffix]
        nc_ts = xr.open_dataset(fpath)

        # fake the names a bit - that make it compatible with the old routines
        if pred_suffix == '_cosmo':
             nc_ts = nc_ts.rename({'x': 'lon', 'y': 'lat'})
        elif pred_suffix == '_ecmwf':

            nc_ts['lon'] = nc_ts.longitude
            nc_ts['lat'] = nc_ts.latitude
            nc_ts.set_coords(['lat', 'lon'])
            nc_ts = nc_ts.swap_dims({'latitude': 'lat', 'longitude': 'lon'})

        # geoloc
        lon = nc_ts.longitude.values
        lat = nc_ts.latitude.values

        # Gradient defaults
        use_tgrad = cfg.PARAMS['temp_use_local_gradient']
        def_tgrad = cfg.PARAMS['temp_default_gradient']
        tg_minmax = cfg.PARAMS['temp_local_gradient_bounds']

        use_pgrad = cfg.PARAMS['prcp_use_local_gradient']
        def_pgrad = cfg.PARAMS['prcp_default_gradient']
        pg_minmax = cfg.PARAMS['prcp_local_gradient_bounds']

        if pred_suffix == '_cosmo':
            # rotated grid: not entirely correct, but reasonable approximation
            # see SLF: https://models.slf.ch/docserver/meteoio/html/gribio.html
            hav = utils.haversine(lon, lat, gdir.cenlon, gdir.cenlat)
            ilat, ilon = np.where(hav == np.min(hav))
            ilat = ilat.item()
            ilon = ilon.item()
            ref_pix_lon = lon[ilat, ilon]
            ref_pix_lat = lat[ilat, ilon]
        elif pred_suffix == '_ecmwf':
            ilon = np.argmin(np.abs(lon - gdir.cenlon))
            ilat = np.argmin(np.abs(lat - gdir.cenlat))
            ref_pix_lon = lon[ilon]
            ref_pix_lat = lat[ilat]

        # Some special things added in the crampon function
        # we hand over the file directly, because of the fake lat/lon dimensions
        iprcp, itemp, itmin, itmax, isis, itgrad, itgrad_unc, ipgrad, ipgrad_unc, ihgt = \
            utils.joblib_read_climate(
                nc_ts, ilon, ilat, def_tgrad, tg_minmax, use_tgrad, def_pgrad,
                pg_minmax, use_pgrad)

        # get the uncertainties according to the functions describing them
        time = [pd.Timestamp(t) for t in nc_ts.time.values]
        gdir.write_monthly_climate_file(time, iprcp, itemp, itgrad, ipgrad,
                                        ihgt, ref_pix_lon, ref_pix_lat,
                                        tmin=itmin, tmax=itmax,
                                        tgrad_sigma=itgrad_unc, sis=isis,
                                        temp_sigma=None,
                                        prcp_sigma=None,
                                        sis_sigma=None,
                                        pgrad_sigma = ipgrad_unc,
                                        file_name='nwp_daily' + pred_suffix)

        nc_ts.close()


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
    .. [2] : https://bit.ly/3nhbTQV
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

    outfile = os.path.join(write_to, cfg.BASENAMES['climate_spinup'])

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
    prec = prec.drop_sel(['dummy', 'longitude_latitude'])
    temp = temp.drop_sel(['dummy', 'longitude_latitude'])

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

    month_length = xr.DataArray(utils.get_dpm(prec.time.to_index()),
                                coords=[prec.time], name='month_length')
    # generate "daily precipitation"
    prec /= month_length

    # resample artificially to daily - no 'previous' available yet
    old_t_units = prec.time.units
    # todo: replace 'nearest' with 'previous' when available
    prec = prec.interp(time=pd.date_range(pd.Timestamp(min(prec.time).item()) -
                                          pd.tseries.offsets.MonthBegin(1),
                                          pd.Timestamp(max(prec.time).item()) +
                                          pd.tseries.offsets.MonthEnd(1)),
                       method='nearest', kwargs={'fill_value': 'extrapolate'})
    temp = temp.interp(time=pd.date_range(pd.Timestamp(min(temp.time).item()) -
                                          pd.tseries.offsets.MonthBegin(1),
                                          pd.Timestamp(max(temp.time).item()) +
                                          pd.tseries.offsets.MonthEnd(1)),
                       method='nearest', kwargs={'fill_value': 'extrapolate'})
    nc_ts = xr.merge([temp, prec, hgt])
    nc_ts = nc_ts.sel(time=slice(None, '1960-12-31'))
    nc_ts.time.encoding['units'] = old_t_units  # no way to keep when interp

    # ensure it's compressed when exporting
    nc_ts.encoding['zlib'] = True
    for v in nc_ts.data_vars:
        nc_ts[v].encoding.update({'dtype': np.int16, 'scale_factor': 0.01,
                                  '_FillValue': -9999})
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

    if write_to is None:
        try:
            write_to = cfg.PATHS['climate_dir']
        except KeyError:
            raise KeyError('Must supply write_to or initialize the crampon'
                           'configuration.')
    if hfile is None:
        try:
            hfile = cfg.PATHS['hfile']
        except KeyError:
            raise KeyError('Must supply hfile or initialize the crampon'
                           'configuration.')

    # again other variable names on FTP delivery; SIS D still not operational
    ftp_key_dict = {'BZ13': 'RprelimD', 'BZ14': 'RhiresD', 'BZ51': 'TabsD',
                    'BZ54': 'TminD', 'BZ55': 'TmaxD', 'BZ69': 'msgSISD_',
                    'CZ91': 'TabsD', 'CZ92': 'TminD', 'CZ93': 'TmaxD'}

    # encoding used when writing to disk (sves enormous amounts of data
    io_enc = {'dtype': 'int16', 'scale_factor': 0.01, '_FillValue': -9999}

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
            # "?[!y]" is to exclude RhydchprobD
            r, _ = cirrus.sync_files(
                '/data/griddata', write_to,
                globpattern='*{}/daily/{}?[!y]*[!swissgrid]/netcdf/*'
                    .format(mode, var))
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
                    extracted = glob(os.path.join(ftp_write_to,
                                                  ftype + '*'))[0]
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
        flist = glob(os.path.join(
            write_to, (globdir + '*.nc').format(mode, var)))

        # do we want/need to create everything from scratch?
        if how == 'from_scratch' or ((not os.path.exists(all_file)) and
                                     len(flist) > 0):
            # Instead of using open_mfdataset (we need a lot of preprocessing)
            log.info('Concatenating {} {} {} files...'
                     .format(len(flist), var, mode))
            sda = utils.read_multiple_netcdfs(flist, chunks={'time': 50},
                                              tfunc=utils._cut_with_CH_glac)
            # todo: throw an error if data are missing (e.g. files retrieved
            #  from Cirrus, pause, then only six from FTP)
            log.info('Ensuring time continuity...')
            sda = sda.crampon.ensure_time_continuity()
            #sda.encoding['zlib'] = True
            #sda.to_netcdf(all_file)
        elif how == 'update':
            # a file might arrive 2x on FTP; order should be ok (take latest)
            r = np.unique(r)
            if len(r) == 0:
                log.info('{} {} files up to date. Continuing...')
                continue
            log.info('Updating with {} {} {} files...'
                     .format(len(r), var, mode))
            old = utils.read_netcdf(all_file, chunks={'time': 50})
            new = utils.read_multiple_netcdfs(r, tfunc=utils._cut_with_CH_glac)
            sda = old.combine_first(new)
            # todo: ensure time continuity? What happens if file is missing?
        else:
            raise ValueError('Climate file creation creation mode {} is not '
                             'allowed.'.format(mode))

        sda.encoding['zlib'] = True
        # `var` is not always the variable name in the file, so iterate:
        for v in sda.data_vars:
            sda[v].encoding.update(io_enc)
            # needs the smaller scaling - not sure why though (scaled numbers
            # shouldn't be larger than 2**16)
            if v == 'SIS':
                sda[v].encoding.update({'scale_factor': 0.1})
        sda.to_netcdf(all_file)

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
        for v in data.data_vars:
            data[v].encoding.update(io_enc)
            if v == 'SIS':
                data[v].encoding.update({'scale_factor': 0.1})
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


def make_nwp_file_cosmo(write_to: Optional[str] = None) -> None:
    """
    Compile the COSMO numerical weather prediction file.

    Parameters
    ----------
    write_to: str
        Path where to write the NWP file. Default: None (take `nwp_file_cosmo`
        from configuration).

    Returns
    -------

    """
    # todo: actually, compared to the other functions above, this function
    #  should distribute an existing nwp.nc to all gdirs - the processing
    #  below should happen before

    if not write_to:
        try:
            write_to = cfg.PATHS['climate_dir']
        except KeyError:
            raise KeyError('Must supply target directory or initialize the '
                           'crampon configuration.')

    cosmo_dir = os.path.join(write_to, 'cosmo')
    ecmwf_dir = os.path.join(write_to, 'ecmwf')

    # current midnight modelrun
    midnight = dt.datetime.now().replace(hour=0, minute=0, second=0,
                                         microsecond=0)
    run_hour = '00'
    run_day = midnight.strftime('%y%m%d')

    # COSMO-E
    ftp_cosmo = utils.WSLSFTPClient(user='hyv-data')
    ftp_dir_cosmo = '/data/ftp/hyv_data/cosmo/cosmoe'
    files_cosmo = ftp_cosmo.list_content(ftp_dir_cosmo)
    cfiles_ce = [f for f in files_cosmo if (('SZ91' in f) and
                                      ('{}{}'.format(run_day, run_hour) in f)
                                      and f.endswith('.zip'))]

    # download latest COSMO-E file(s)
    for cfile in cfiles_ce:
        local_file = os.path.join(cosmo_dir, os.path.basename(cfile))
        ftp_cosmo.get_file(cfile, local_file)
    ftp_cosmo.close()

    to_unzip = [os.path.join(cosmo_dir, os.path.basename(r)) for r in
                cfiles_ce]

    for dl_zipfile in to_unzip:
        utils.unzip_file(dl_zipfile)
        os.remove(dl_zipfile)

    # we don't want leftover *.idx files
    ce_globpath = '**/*wsl_cosmo-2e_hydro-ch_*[!.idx]'
    if sys.platform.startswith('win'):
        ce_globpath = ce_globpath.replace('/', '\\')
    ce_paths = glob(cosmo_dir + ce_globpath, recursive=True)

    all_dsi = []
    hgt = None
    # 'unknown' is radiation
    vars_of_interest = ['t2m', 'tp', 'unknown', 'p3008']
    for ce_p in ce_paths:
        dss = cfgrib.open_datasets(ce_p)
        run_dsi = []
        for ids, ds in enumerate(dss):
            for v in vars_of_interest:
                # todo: hard code, bcz GRIB1 COSMO spec.def. not valid anymore!
                if (ids == 2) and (v == 'unknown'):  # position should be same
                    grad = ds[v]
                    grad = grad.rename('grad')  # ECMWF name for global rad.
                    grad.attrs['GRIB_missingValue'] = -0.99
                    run_dsi.append(grad)
                elif (v != 'unknown') and ('step' in ds.coords):
                    if ds.step.values.size > 1:  # other shenanigans
                        try:
                            dsi = ds[v]
                            run_dsi.append(dsi)
                        except KeyError:
                            continue
                    elif (ds.step.values.size == 1) and (v == 'p3008') and \
                            (v in ds.data_vars):
                        hgt = ds[v]
        all_dsi.append(run_dsi)

    # make one file
    cosmo_ds = xr.concat([xr.merge(x) for x in all_dsi], dim='member')

    # rename variables to make them CRAMPON suitable
    cosmo_ds = cosmo_ds.rename({'t2m': 'temp', 'tp': 'prcp', 'grad': 'sis'})
    hgt = hgt.rename('hgt')

    # some unit change
    cosmo_ds['temp'] = cosmo_ds['temp'] - cfg.ZERO_DEG_KELVIN  # K to degC

    # make daily means/sums/max/mins for everything
    # variables are from 00 - 00; let's make 0-23 a day and waste the last bit
    xr.set_options(keep_attrs=True)
    cosmo_ds = xr.merge([
        cosmo_ds.temp.resample(step='1D').min().rename('tmin'),
        cosmo_ds.temp.resample(step='1D').max().rename('tmax'),
        cosmo_ds.temp.resample(step='1D').mean(),
        cosmo_ds.sis.resample(step='1D').mean(),
        cosmo_ds.prcp.resample(step='1D').sum(),
        hgt
    ])

    # the last step (day 5) is only midnight for midnight run - cut it off
    if run_hour == '00':
        cosmo_ds = cosmo_ds.isel(step=slice(None, -1))
    else:
        raise ValueError('Forming daily aggregations not yet supported for '
                         'COSMO runs other than 00 (midnight).')

    # create time dimension in day steps
    cosmo_ds['time'] = cosmo_ds.time + cosmo_ds.step
    cosmo_ds = cosmo_ds.swap_dims({'step': 'time'})

    # write out as cosmo_predictions.nc for mass balance prediction to access
    cosmo_ds.to_netcdf(cfg.PATHS['nwp_file_cosmo'])


def make_nwp_file_ecmwf(write_to: Optional[str] = None) -> None:
    """
    Compile the ECMWF numerical weather prediction file.

    Parameters
    ----------
    write_to: str
        Path where to write the NWP file. Default: None (take `nwp_file_ecmwf`
        from configuration).

    Returns
    -------

    """
    # todo: actually, compared to the other functions above, this function
    #  should distribute an existing nwp.nc to all gdirs - the processing
    #  below should happen before

    if not write_to:
        try:
            write_to = cfg.PATHS['climate_dir']
        except KeyError:
            raise KeyError('Must supply target directory or initialize the '
                           'crampon configuration.')

    ecmwf_dir = os.path.join(write_to, 'ecmwf')

    # runs always come in on Monday and Thursdays
    now = pd.Timestamp.now()
    yesterday = now - pd.Timedelta(days=1)
    # todo: 7.50 is CEST: understand the context when
    if now.hour <= 7 and now.minute > 50:
        search_date = now
    else:
        search_date = yesterday

    last_monday = search_date - \
                  pd.Timedelta(days=(search_date.weekday()) % 7, weeks=0)
    last_thursday = search_date - \
                    pd.Timedelta(days=(search_date.weekday() - 3) % 7, weeks=0)
    run_day_str = max(last_monday, last_thursday).strftime('%Y%m%d')

    # ECMWF monthly predictions
    ftp_ecmwf = utils.WSLSFTPClient()
    ftp_dir_ecmwf = '/data/ftp/map/monthlyENS'
    files_ecmwf = ftp_ecmwf.list_content(ftp_dir_ecmwf)

    # find the latest file
    server_file = [f for f in files_ecmwf if
                    ('vareps_grib_{}'.format(run_day_str) in f) and
                    (f.endswith('.tar.gz'))][0]

    # download latest file
    local_file = os.path.join(ecmwf_dir, os.path.basename(server_file))
    ftp_ecmwf.get_file(server_file, local_file)
    ftp_ecmwf.close()

    # tidy up old grib and index files
    old_files = glob(os.path.join(ecmwf_dir, 'latest_*'))
    for of in old_files:
        os.remove(of)

    utils.untargz_file(local_file)
    os.remove(local_file)

    # todo: include control run?
    ens_globpath = '**/latest*.grb'
    if sys.platform.startswith('win'):
        ens_globpath = ens_globpath.replace('/', '\\')
    ens_paths = glob(os.path.join(ecmwf_dir, ens_globpath), recursive=True)

    vars_of_interest = ['t2m', 'ssr', 'tp']
    runs = [cfgrib.open_dataset(e) for e in ens_paths if '_rm' in e]
    control = [cfgrib.open_dataset(e) for e in ens_paths if '_ctr' in e][0]
    hgt = control.z.isel(step=0)   # z is step-dependent in the file
    # from MetPy: convert geopotential to altitude above sea level
    hgt = (hgt * cfg.RE) / (cfg.G * cfg.RE - hgt)
    hgt = hgt.drop('number')

    # make one file
    member_ix = pd.Index(np.arange(len(runs)), name='member')
    run_ds = xr.concat(runs, dim=member_ix)

    # select variables
    run_ds = run_ds[vars_of_interest]
    control = control[vars_of_interest]

    # revert the cumsums of tp and ssr - last step needs to be cut off
    run_ds['tp'] = run_ds.tp.diff('step') * 1000. # convert to mm
    run_ds['ssr'] = run_ds.ssr.diff('step')
    control['tp'] = control.tp.diff('step') * 1000. # convert to mm
    control['ssr'] = control.ssr.diff('step')

    # drop very last step - it would be a lost step when aggregating to days
    run_ds = run_ds.isel(step=slice(None, -1))
    control = control.isel(step=slice(None, -1))

    # rename variables to make them CRAMPON suitable
    rename_dict = {'t2m': 'temp', 'tp': 'prcp', 'ssr': 'sis'}
    run_ds = run_ds.rename(rename_dict)
    control = control.rename(rename_dict)
    hgt = hgt.rename('hgt')

    # some unit changes
    run_ds['temp'] = run_ds['temp'] - cfg.ZERO_DEG_KELVIN  # K to degC
    control['temp'] = control['temp'] - cfg.ZERO_DEG_KELVIN  # K to degC

    # make daily means/sums/max/mins for everything
    # variables are from 00 - 00; let's make 0-23 a day and waste the last bit
    xr.set_options(keep_attrs=True)
    run_ds = xr.merge(
        [run_ds.temp.resample(step='1D').min().rename('tmin'),
            run_ds.temp.resample(step='1D').max().rename('tmax'),
            run_ds.temp.resample(step='1D').mean(),
            run_ds.sis.resample(step='1D').sum(),  # sum because it's J m**-2
            run_ds.prcp.resample(step='1D').sum(), hgt])
    # todo: use control run
    control = xr.merge([run_ds.temp.resample(step='1D').min().rename('tmin'),
                       run_ds.temp.resample(step='1D').max().rename('tmax'),
                       run_ds.temp.resample(step='1D').mean(),
                       run_ds.sis.resample(step='1D').sum(),  # sum: J m**-2
                       run_ds.prcp.resample(step='1D').sum(), hgt])

    # change unit of SIS
    run_ds['sis'] = run_ds['sis'] / cfg.SEC_IN_DAY  # J**m-2 to W**m-2
    control['sis'] = control['sis'] / cfg.SEC_IN_DAY  # J**m-2 to W**m-2

    # create time dimension in day steps
    run_ds['time'] = run_ds.time + run_ds.step
    run_ds = run_ds.swap_dims({'step': 'time'})

    # write out as cosmo_predictions.nc for mass balance prediction to access
    run_ds.to_netcdf(cfg.PATHS['nwp_file_ecmwf'])


def make_nwp_files(write_to_cosmo: Optional[str] = None,
                   write_to_ecmwf: Optional[str] = None) -> None:
    """
    Wrapper function to compile COSMO & ECMWF numerical weather predictions.

    Parameters
    ----------
    write_to_cosmo: str
        Path where to write the COSMO NWP file to. Default: None (take
        `nwp_file_cosmo` from configuration).
    write_to_ecmwf: str
        Path where to write the NWP files to. Default: None (take
        `nwp_file_ecmwf` from configuration).

    Returns
    -------

    """
    make_nwp_file_cosmo(write_to_cosmo)
    make_nwp_file_ecmwf(write_to_ecmwf)


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
    #  can be removed when radiation is operational
    last_day = last_day_clim - pd.Timedelta(days=62)

    use_tgrad = cfg.PARAMS['temp_use_local_gradient']
    def_tgrad = cfg.PARAMS['temp_default_gradient']
    tg_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    use_pgrad = cfg.PARAMS['prcp_use_local_gradient']
    def_pgrad = cfg.PARAMS['prcp_default_gradient']
    pg_minmax = cfg.PARAMS['prcp_local_gradient_bounds']

    gclim = xr.open_dataset(gdir.get_filepath('climate_daily'))
    last_day_gclim = gclim.time[-1]

    # it can be that updating the glacier climate is longer ago than 62 days
    last_day = min(last_day, last_day_gclim)
    # todo: take also care that existing climate_daily does not have gaps!!!

    if last_day < last_day_clim:
        clim_all_sel = clim_all.sel(dict(lat=gclim.ref_pix_lat,
                                         lon=gclim.ref_pix_lon,
                                         time=slice(last_day,
                                                    last_day_clim)))
        clim_all_tsel = clim_all.sel(time=slice(last_day, last_day_clim))
        # todo: save ilat/ilon as gdir attributes!?
        # todo: save tgrad/pgrad in climate_all.nc? (is vectorizing it faster?)
        # we need to add tgrad and pgrad (not in climate file)
        ilon = np.argmin(np.abs(clim_all.lon.values - gdir.cenlon)).item()
        ilat = np.argmin(np.abs(clim_all.lat.values - gdir.cenlat)).item()
        # todo: handing over clim_all_tsel is bullshit, since we take a 30 day
        #  rolling mean to fill gradient gaps
        # selection [0] for tgrad ([1] is uncertainty)
        tg, tgu = utils.get_tgrad_from_window(
            clim_all_tsel, ilat=ilat, ilon=ilon, win_size=use_tgrad,
            default_tgrad=def_tgrad, minmax_tgrad=tg_minmax)
        clim_all_sel['tgrad'] = (['time'], tg)
        clim_all_sel['tgrad_sigma'] = (['time'], tgu)

        # no double output for pgrad
        pg, pgu = utils.get_pgrad_from_window(
            clim_all_tsel, ilat=ilat, ilon=ilon, win_size=use_pgrad,
            default_pgrad=def_pgrad, minmax_pgrad=pg_minmax)
        clim_all_sel['pgrad'] = (['time'], pg)
        clim_all_sel['pgrad_sigma'] = (['time'], pgu)
        updated = gclim.combine_first(clim_all_sel)
        gclim.close()
        updated.to_netcdf(gdir.get_filepath('climate_daily'))

    if need_close is True:
        clim_all.close()

    # todo: update NWP here as well?


class GlacierMeteo(object):
    """
    Interface to the meteorological data belonging to a glacier geometry.
    """

    def __init__(self, gdir, filename='climate_daily', filesuffix='',
                 heights=None, randomize=False, n_random_samples=1000,
                 use_tgrad_uncertainty=False, use_pgrad_uncertainty=False):
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
        filename: str
            File name for the climate file to be read. Default:
            'climate_daily'.
        heights: np.array or None
            Heights of the glacier discretization. Not necessary for
            everything. If None, the values are taken from the flowlines
            heights. Default: None.
        randomize: bool
            If True, randomize the values of meteorological parameters
            with a Gaussian with the `sigma` values as standard deviation.
        n_random_samples : int
            If uncertain estimates for meteorological parameters are
            desired (randomize=True), draw this number of random samples.
            Default: 1000. Attention, choosing this as a high number can
            blow things up easily, especially, when the glacier has many flow
            line heights.
        use_tgrad_uncertainty: bool
            Whether to use the uncertainty in the temperature gradient in
            randomize mode as well. This first accounts for the random
            temperature error in the meteorological grids, and ten applied on
            top a random gradient derived from the standard error of the slope
            of the calculated temperature gradient. Default: False (do not
            consider).
        use_pgrad_uncertainty: bool
            Whether to use the uncertainty in the precipitation gradient in
            randomize mode as well. This first accounts for the random
            temperature error in the meteorological grids, and ten applied on
            top a random gradient derived from the standard error of the slope
            of the calculated temperature gradient. Default: False (do not
            consider).
        """
        self.gdir = gdir
        if heights is None:
            self._heights = self.gdir.get_inversion_flowline_hw()[0]
        else:
            self._heights = heights
        self.meteo = xr.open_dataset(self.gdir.get_filepath(
            filename, filesuffix=filesuffix))
        self.index = self.meteo.time.to_index()
        self.tmean = self.meteo.temp.values
        self.prcp = self.meteo.prcp.values
        self.pgrad = self.meteo.pgrad.values
        self.tgrad = self.meteo.tgrad.values
        self.ref_hgt = self.meteo.ref_hgt
        self._days_since_solid_precipitation = None
        # not all might have these ones (e.g. spinup)
        for v in ['tmin', 'tmax', 'sis']:
            try:
                setattr(self, v, self.meteo[v].values)
            except KeyError:
                setattr(self, v, None)

        # get the uncertainties - passively at the moment
        try:
            # todo: check if it is a good assumption to assign the
            #  uncertainty of tmean of to tmax/tmin
            self.temp_sigma = self.meteo.temp_sigma.values
            self.tmin_sigma = self.meteo.temp_sigma.values
            self.tmax_sigma = self.meteo.temp_sigma.values
        except AttributeError:
            self.temp_sigma = None
        try:
            self.prcp_sigma = self.meteo.prcp_sigma.values
        except AttributeError:
            self.prcp_sigma = None
        try:
            self.sis_sigma = self.meteo.sis_sigma.values
        except AttributeError:
            self.sis_sigma = None
        try:
            self.tgrad_sigma = self.meteo.tgrad_sigma.values
        except AttributeError:
            self.tgrad_sigma = None
        try:
            self.pgrad_sigma = self.meteo.pgrad_sigma.values
        except AttributeError:
            self.pgrad_sigma = None

        self.randomize = randomize
        self.n_random_samples = n_random_samples
        self.use_tgrad_uncertainty = use_tgrad_uncertainty
        self.use_pgrad_uncertainty = use_pgrad_uncertainty

    @property
    def heights(self):
        """Heights for which to get meteorological variables."""
        return self._heights

    @lazy_property
    def tmean_stdev(self):
        """Standard deviation (over time) of mean temperature."""
        return np.nanstd(self.tmean)

    @lazy_property
    def tmin_stdev(self):
        """Standard deviation (over time) of minimum temperature."""
        if self.tmin is not None:
            return np.nanstd(self.tmin)
        else:
            raise ValueError('GlacierMeteo object does not have a tmin '
                             'attribute (not generated from daily '
                             'meteorological data)')

    @lazy_property
    def tmax_stdev(self):
        """Standard deviation (over time) of maximum temperature."""
        if self.tmax is not None:
            return np.nanstd(self.tmax)
        else:
            raise ValueError('GlacierMeteo object does not have a tmax '
                             'attribute (not generated from daily '
                             'meteorological data)')

    @lazy_property
    def prcp_stdev(self):
        """Standard deviation (over time) of precipitation."""
        return np.nanstd(self.prcp)

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
            precip_s, _ = self.get_precipitation_solid_liquid(
                date, method=method, **kwargs)
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

    def randomize_variable(self, date, var, random_seed=None):
        """
        Randomize a variable (create random samples from a Gaussian).

        Parameters
        ----------
        date: pd.Timestamp
            Date on which to randomize the variable.
        var : str
            Variable to randomize.
        random_seed: int or None, optional
            Whether to put a random seed, i.e. make the results reproducible.

        Returns
        -------
        randomized: np.array
             Desired variable randomized.
        """
        ix = self.get_loc(date)
        # todo: distinguish additive noise (temperatures) and multiplicative
        #  noise (prcp, sis) ok?
        var_value = getattr(self, var)[ix]
        var_sigma = getattr(self, var+'_sigma')[ix]

        # if var in ['tmean', 'tmin', 'tmax', 'sis']:
        #    gauss_noise_add = np.random.normal(0, var_sigma,
        #                                       self.n_random_samples)
        #    randomized = gauss_noise_add + var_value
        # elif var in ['prcp']:
        #    gauss_noise_mul = np.random.normal(0, var_sigma,
        #                                       self.n_random_samples)
        #    randomized = gauss_noise_mul * var_value
        if random_seed is not None:
            np.random.seed(random_seed)
        gauss_noise_add = np.random.normal(0, var_sigma,
                                           self.n_random_samples)
        randomized = gauss_noise_add + var_value
        return randomized

    def get_tmean_at_heights(self, date, heights=None, random_seed=None):
        """
        Get the mean temperature at the given heights and date.

        Parameters
        ----------
        date : pd.Timestamp or pd.DatetimeIndex
            Date for which to calculate the mean temperature.
        heights : np.array
            Heights on which to calculate the temperature.
        random_seed: int or None, optional
            Whether to put a random seed, i.e. make the results reproducible.

        Returns
        -------
        temp_at_hgts: np.array
            Mean temperature at the given heights and date.
        """

        if heights is None:
            heights = self.heights

        date_index = self.get_loc(date)
        temp = self.tmean[date_index]
        tgrad = self.tgrad[date_index]
        temp_at_hgts = get_temperature_at_heights(temp, tgrad, self.ref_hgt,
                                                  heights)

        if self.randomize is True:
            temp_sigma = self.temp_sigma[date_index]
            if random_seed is not None:
                np.random.seed(random_seed)
            temp_at_hgts = np.random.normal(
                0, temp_sigma, size=self.n_random_samples) + \
                np.atleast_2d(temp_at_hgts).T

            if self.use_tgrad_uncertainty is True:
                tgrad_sigma = self.tgrad_sigma[date_index]
                if random_seed is not None:
                    np.random.seed(random_seed)
                random_grads = np.random.normal(tgrad, tgrad_sigma,
                                                size=self.n_random_samples)
                temp_at_hgts += np.atleast_2d(random_grads - tgrad) * \
                    np.atleast_2d(heights - self.ref_hgt).T

        return temp_at_hgts

    # todo: remove this again, it's an emergency solution
    def get_tmax_at_heights(self, date, heights=None, random_seed=None):
        """
        Get the maximum temperature at the given heights and date.

        Parameters
        ----------
        date : pd.Timestamp
            Date for which to calculate the mean temperature.
        heights : np.array
            Heights on which to calculate the temperature.
        random_seed: int or None, optional
            Whether to put a random seed, i.e. make the results reproducible.

        Returns
        -------
        temp_at_hgts: np.array
            Maximum temperature at the given heights and date.
        """
        if heights is None:
            heights = self.heights

        date_index = self.get_loc(date)
        temp = self.tmax[date_index]
        tgrad = self.tgrad[date_index]
        temp_at_hgts = get_temperature_at_heights(temp, tgrad, self.ref_hgt,
                                                  heights)

        if self.randomize is True:
            temp_sigma = self.temp_sigma[date_index]
            if random_seed is not None:
                np.random.seed(random_seed)
            temp_at_hgts = np.random.normal(
                0, temp_sigma, size=self.n_random_samples) + \
                np.atleast_2d(temp_at_hgts).T

            if self.use_tgrad_uncertainty is True:
                tgrad_sigma = self.tgrad_sigma[date_index]
                if random_seed is not None:
                    np.random.seed(random_seed)
                random_grads = np.random.normal(tgrad, tgrad_sigma,
                                                size=self.n_random_samples)
                temp_at_hgts += np.atleast_2d(
                    random_grads - tgrad) * np.atleast_2d(
                    heights - self.ref_hgt).T

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
        above_melt = temp_at_hgts - t_melt
        above_melt[above_melt < 0.] = 0.
        return above_melt

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

    def get_precipitation_solid_liquid(self, date, heights=None, method=None,
                                       tmean=None, prcp_fac=1.,
                                       random_seed=None, **kwargs):
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
        tmean: np.array
            An option to give (random) values for tmean, which will determine
            the fraction of solid precipitation and the total precipitation.
            Default: None (generate (random) temperature).
        prcp_fac: float, optional
            Precipitation correction factor. It is applied before applying the
            gradient. Default: 1. (no correction).
        random_seed: int or None, optional
            Whether to put a random seed, i.e. make the results reproducible.
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
        prcp = self.prcp[date_index] * prcp_fac
        pgrad = self.pgrad[date_index]
        # if not rain at reference and no gradient, we can skip the procedure
        if (prcp == 0.) and (np.isnan(pgrad)):
            if self.randomize is True:
                out = np.zeros((len(heights), self.n_random_samples))
                return out, out
            else:
                out = np.zeros(len(heights))
                return out, out
        prcptot = get_precipitation_at_heights(prcp, pgrad, self.ref_hgt,
                                               heights)

        # important: we don't take compound interest formula (p could be neg!)
        prcptot = np.clip(prcptot, 0, None)

        # tmean determines psol
        if tmean is None:
            tm_hgts = self.get_tmean_at_heights(date, heights)
        else:
            tm_hgts = tmean.copy()
        if method.lower() == 'magnusson':
            # todo: instead of calculatng the temp_at_heights again one should
            #  make temp_at_heights a kwarg (otherwise get_loc is called two
            #  times => cost-intensive)
            frac_solid = get_fraction_of_snowfall_magnusson(tm_hgts, **kwargs)
        elif method.lower() == 'linear':
            frac_solid = get_fraction_of_snowfall_linear(tm_hgts, **kwargs)
        else:
            raise ValueError('Solid precipitation fraction method not '
                             'recognized. Allowed are "magnusson" and '
                             '"linear".')

        if self.randomize is True:
            # we have read the relative error: sigma * prcp + prcp
            prcp_sigma = self.prcp_sigma[date_index]
            if random_seed is not None:
                np.random.seed(random_seed)
            prcptot = np.random.normal(
                1., prcp_sigma, size=self.n_random_samples) * \
                np.atleast_2d(prcptot).T
            # brute force clip at zero, since we take a Gaussian error distr.
            prcptot = np.clip(prcptot, 0., None)

            if self.use_pgrad_uncertainty is True:
                pgrad_sigma = self.pgrad_sigma[date_index]
                if random_seed is not None:
                    np.random.seed(random_seed)
                random_grads = np.random.normal(pgrad, pgrad_sigma,
                                                size=self.n_random_samples)
                prcptot += prcptot * (random_grads - pgrad) * \
                           np.atleast_2d((heights + self.ref_hgt)).T

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
    pgrad: float, array-like or xarray.DataArray
        The precipitation gradient to apply (K m-1).
    ref_hgt: float, array-like or xarray.DataArray
        Reference height (m).
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
                   np.array(pgrad[:, np.newaxis]) * (heights - ref_hgt)
        except IndexError:
            return np.ones_like(heights) * prcp.values[:, np.newaxis] + \
                   prcp.values[:, np.newaxis] * pgrad.values[:, np.newaxis] * \
                   (heights - ref_hgt)


@numba.jit(nopython=True)
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
