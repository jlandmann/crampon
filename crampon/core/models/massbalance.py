from __future__ import division

from oggm.core.massbalance import *
from crampon import cfg
from crampon.utils import SuperclassMeta, lazy_property, closest_date
from crampon import utils
import xarray as xr
import datetime as dt
from enum import IntEnum, unique
import cython
from crampon.core.preprocessing import climate
import numba
from collections import OrderedDict
from itertools import product
import pandas as pd


class DailyMassBalanceModel(MassBalanceModel):
    """
    Extension of OGGM's MassBalanceModel, able to calculate daily mass balance.
    """

    cali_params_list = ['mu_star', 'prcp_fac']
    cali_params_guess = OrderedDict(zip(cali_params_list, [10., 1.5]))
    prefix = 'DailyMassBalanceModel_'
    mb_name = prefix + 'MB'

    def __init__(self, gdir, mu_star=None, bias=None, prcp_fac=None,
                 filename='climate_daily', filesuffix='',
                 heights_widths=(None, None), param_ep_func=np.nanmean,
                 cali_suffix=''):

        """
        Model to calculate the daily mass balance of a glacier.

        Parameters
        ----------
        gdir:
        mu_star:
        bias:
        prcp_fac:
        filename:
        filesuffix:
        cali_suffix: str
            Suffix for getting the calibration file.
        # todo: wouldn't it rather make sense to make a gdir from beginning on that has only fewer heights? Conflict with stability of flow approach?!!
        heights_widths: tuple
            Heights and widths if not those from the gdir shall be taken
        param_ep_func: numpy arithmetic function
            Method to use for extrapolation when there are no calibrated
            parameters available for the time step where the mass balance
            should be calcul
        """

        super().__init__()

        # Needed for the calibration parameter names in the csv file
        self.__name__ = type(self).__name__
        self.prefix = self.__name__ + '_'

        # just a temporal dummy
        self.snowcover = None

        # should probably be replaced by a direct access to a file that
        # contains uncertainties (don't know how to handle that yet)
        # todo: think again if this is a good solution to save code
        if any([p is None for p in [mu_star, prcp_fac, bias]]):
            # do not filter for mb_model here, otherwise mu_star is not found
            # ('self' can also be a downstream inheriting class!)
            cali_df = gdir.get_calibration(filesuffix=cali_suffix)

        # DailyMassbalanceModel should also be able to grab OGGM calibrations
        if mu_star is None:
            try:
                mu_star = cali_df[self.__name__ + '_' + 'mu_star']
            except KeyError:
                try:
                    mu_star = cali_df['OGGM_mu_star']
                except KeyError:
                    mu_star = cali_df['mu_star']
        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                bias = cali_df['bias']
            else:
                bias = pd.DataFrame(index=cali_df.index,
                                    data=np.zeros_like(cali_df.index,
                                                       dtype=float))
        if prcp_fac is None:
            try:
                prcp_fac = cali_df[self.__name__ + '_' + 'prcp_fac']
            except KeyError:
                prcp_fac = cali_df['prcp_fac']

        self.gdir = gdir
        self.filesuffix = filesuffix
        self.param_ep_func = param_ep_func
        self.mu_star = mu_star
        self.bias = bias
        self.prcp_fac = prcp_fac
        # temporary solution: Later this should be more flexible: Either the
        # update should happen in the mass balance method directly or the
        # heights/widths should be multitemporal
        if (heights_widths[0] is not None) and (heights_widths[1] is not None):
            self.heights, self.widths = heights_widths
        elif (heights_widths[0] is None) and (heights_widths[1] is None):
            self.heights, self.widths = gdir.get_inversion_flowline_hw()
        else:
            raise ValueError('You must provide either both heights and '
                             'widths or nothing (take heights/widths from '
                             'gdir).')

        # overwrite/add some OGGM stuff
        self.m = None
        self.years = None

        # Parameters
        self.t_solid = cfg.PARAMS['temp_all_solid']
        self.t_liq = cfg.PARAMS['temp_all_liq']
        self.t_melt = cfg.PARAMS['temp_melt']

        # Read file
        fpath = gdir.get_filepath(filename, filesuffix=filesuffix)
        with netCDF4.Dataset(fpath, mode='r') as nc:
            # time as nc time variable
            self.nc_time = nc.variables['time']
            self.nc_time_units = self.nc_time.units
            # time as array of datetime instances
            time = netCDF4.num2date(self.nc_time[:], self.nc_time.units)

            # identify the time span
            self.tspan_meteo = time
            self.tspan_meteo_dtindex = pd.DatetimeIndex(time)

            # Read timeseries
            self.temp = nc.variables['temp'][:]
            self.prcp_unc = nc.variables['prcp'][:]
            if isinstance(self.prcp_fac, pd.Series):
                self.prcp = nc.variables['prcp'][:] * \
                            self.prcp_fac.reindex(index=time,
                                                  method='nearest')\
                                .fillna(value=self.param_ep_func(self.prcp_fac))
            else:
                self.prcp = nc.variables['prcp'][:] * self.prcp_fac
            self.tgrad = nc.variables['tgrad'][:]
            self.pgrad = nc.variables['pgrad'][:]
            self.ref_hgt = nc.ref_hgt

        self.meteo = climate.GlacierMeteo(self.gdir)

        # Public attrs
        self.temp_bias = 0.

        self._time_elapsed = None

        # determine the annual cyclicity of the precipitation correction factor
        # this assumes that every year has the length of a leap year (shouldn't
        # matter) and that the winter calibration phase is always OCT-APR
        prcp_fac_annual_cycle = climate.prcp_fac_annual_cycle(
            np.arange(1, sum(cfg.DAYS_IN_MONTH_LEAP) + 1))
        prcp_fac_cyc_winter_mean = np.mean(np.hstack([
            prcp_fac_annual_cycle[-sum(cfg.DAYS_IN_MONTH_LEAP[-3:]):],
            prcp_fac_annual_cycle[:sum(cfg.DAYS_IN_MONTH_LEAP[:4])]]))
        self.prcp_fac_cycle_multiplier = prcp_fac_annual_cycle / \
                                         prcp_fac_cyc_winter_mean

    @property
    def time_elapsed(self):
        return self._time_elapsed

    @time_elapsed.setter
    def time_elapsed(self, date):
        if self._time_elapsed is not None:
            self._time_elapsed = self._time_elapsed.insert(
                len(self._time_elapsed), date)
        else:
            if isinstance(date, pd.DatetimeIndex):
                self._time_elapsed = date
            else:
                try:
                    self._time_elapsed = pd.DatetimeIndex(start=date, end=date,
                                                          freq='D')
                except TypeError:
                    raise TypeError('Input date type ({}) for elapsed time not'
                                    ' understood'.format(type(date)))

    def get_prcp_sol_liq(self, iprcp, ipgrad, heights, temp):
        # Compute solid precipitation from total precipitation
        # the prec correction with the gradient does not exist in OGGM
        npix = len(heights)
        prcptot = np.ones(npix) * iprcp + iprcp * ipgrad * (
                    heights - self.ref_hgt)
        # important: we don't take compound interest formula (p could be neg!)
        prcptot = np.clip(prcptot, 0, None)
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcptot * np.clip(fac, 0, 1)
        prcpliq = prcptot - prcpsol

        return prcpsol, prcpliq

    def get_tempformelt(self, temp):
        # Compute temperature available for melt
        tempformelt = temp - self.t_melt
        tempformelt[:] = np.clip(tempformelt, 0, tempformelt.max())
        return tempformelt

    def get_daily_mb(self, heights, date=None, **kwargs):
        """
        Calculates the daily mass balance for given heights.

        At the moment the mass balance equation is the simplest formulation:

        MB(z) = PRCP_FAC * PRCP_SOL(z) - mustar * max(T(z) - Tmelt, 0)

        where MB(z) is the mass balance at height z in mm w.e., PRCP_FAC is
        the precipitation correction factor, PRCP_SOL(z) is the solid
        precipitation at height z in mm w.e., mustar is the
        temperature sensitivity of the glacier (mm w.e. K-1 d-1), T(z) is the
        temperature and height z in (deg C) and Tmelt is the temperature
        threshold where melt occurs (deg C).

        The result of the model equation is thus a mass balance in mm w.e. d.1,
        however, for context reasons (dynamical part of the model), the mass
        balance is returned in m ice s-1, using the ice density as given in the
        configuration file. ATTENTION, the mass balance given is not yet
        weighted with the glacier geometry! For this, see method
        `get_daily_specific_mb`.

        Parameters
        ----------
        heights: ndarray
            Heights at which mass balance should be calculated.
        date: datetime.datetime
            Date at which mass balance should be calculated.
        **kwargs: dict-like, optional
            Arguments passed to the pd.DatetimeIndex.get_loc() method that
            selects the fitting melt parameters from the melt parameter
            attributes.

        Returns
        -------
        ndarray:
            Glacier mass balance at the respective heights in m ice s-1.
        """

        # index of the date of MB calculation
        ix = self.tspan_meteo_dtindex.get_loc(date, **kwargs)

        # Read timeseries
        itemp = self.temp[ix] + self.temp_bias
        if isinstance(self.prcp_fac, pd.Series):
            try:
                iprcp_fac = self.prcp_fac[self.prcp_fac.index.get_loc(date,
                                                                      **kwargs)]
            except KeyError:
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            if pd.isnull(iprcp_fac):
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            iprcp = self.prcp_unc[ix] * iprcp_fac
        else:
            iprcp = self.prcp_unc[ix] * self.prcp_fac
        itgrad = self.tgrad[ix]
        ipgrad = self.pgrad[ix]

        # For each height pixel:
        # Compute temp tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + itgrad * (heights - self.ref_hgt)

        tempformelt = self.get_tempformelt(temp)
        prcpsol, _ = self.get_prcp_sol_liq(iprcp, ipgrad, heights, temp)

        # TODO: What happens when `date` is out of range of the cali df index?
        # THEN should it take the mean or raise an error?
        if isinstance(self.mu_star, pd.Series):
            try:
                mu_star = self.mu_star.iloc[self.mu_star.index.get_loc(
                    date, **kwargs)]
            except KeyError:
                mu_star = self.param_ep_func(self.mu_star)
            if pd.isnull(mu_star):
                mu_star = self.param_ep_func(self.mu_star)
        else:
            mu_star = self.mu_star

        if isinstance(self.bias, pd.Series):
            bias = self.bias.iloc[self.bias.index.get_loc(date, **kwargs)]
            # TODO: Think of clever solution when no bias (=0)?
        else:
            bias = self.bias

        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - mu_star * tempformelt - bias

        self.time_elapsed = date

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / cfg.SEC_IN_DAY / cfg.RHO
        return icerate

    def get_daily_specific_mb(self, heights, widths, date=None, **kwargs):
        """Specific mass balance for a given date and geometry (m w.e. d-1).

        Parameters
        ----------
        heights: ndarray
            The altitudes at which the mass-balance will be computed.
        widths: ndarray
            The widths of the flowline (necessary for the weighted average).
        date: datetime.datetime or array of datetime.datetime
            The date(s) when to calculate the specific mass balance.

        Returns
        -------
        The specific mass-balance of (units: mm w.e. d-1)
        """
        if len(np.atleast_1d(date)) > 1:
            out = [self.get_daily_specific_mb(heights, widths, date=d, **kwargs)
                   for d in date]
            return np.asarray(out)

        # m w.e. d-1
        mbs = self.get_daily_mb(heights, date=date, **kwargs) * \
              cfg.SEC_IN_DAY * cfg.RHO / cfg.RHO_W
        mbs_wavg = np.average(mbs, weights=widths)
        return mbs_wavg

    def get_monthly_mb(self, heights, year=None, fl_id=None):
        """Monthly mass-balance at given altitude(s) for a moment in time.
        Units: [m s-1], or meters of ice per second
        Note: `year` is optional because some simpler models have no time
        component.
        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "hydrological floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """

        month_mb = None

    def get_annual_mb(self, heights, year=None, fl_id=None):
        """Like `self.get_monthly_mb()`, but for annual MB.
        For some simpler mass-balance models ``get_monthly_mb()` and
        `get_annual_mb()`` can be equivalent.
        Units: [m s-1], or meters of ice per second
        Note: `year` is optional because some simpler models have no time
        component.
        Parameters
        ----------
        heights: ndarray
            the atitudes at which the mass-balance will be computed
        year: float, optional
            the time (in the "floating year" convention)
        fl_id: float, optional
            the index of the flowline in the fls array (might be ignored
            by some MB models)
        Returns
        -------
        the mass-balance (same dim as `heights`) (units: [m s-1])
        """

        raise NotImplementedError()

    def generate_climatology(self, write_out=True, n_exp=1):
        """
        EXPERIMENTAL!

        For this to be a method and still be able to produce current
        conditions, a snow conditions file must be written
        Otherwise, if the script to write the current conditions is not run
        with the same instance and with snow_init==day of initiation of last
        budget year, the current conditions are wrong!

        Parameters
        ----------
        write_out
        n_exp

        Returns
        -------

        """
        # number of experiments (list!)
        n_exp = list(range(n_exp))

        for exp in n_exp:
            mb = []
            for date in self.span_meteo:
                # Get the mass balance and convert to m per day
                tmp = self.get_daily_specific_mb(self.heights, self.widths,
                                                 date=date)
                mb.append(tmp)

            mb_ds = xr.Dataset({self.prefix + 'MB': (['time', 'n'],
                                       np.atleast_2d(mb).T)},
                               coords={'n': (['n'], exp),
                                       'time': pd.to_datetime(self.span_meteo)},
                               attrs={'prcp_fac': self.prcp_fac,
                                      'mu_star': self.mu_star,
                                      'id': self.gdir.rgi_id,
                                      'name': self.gdir.name})

            # save results
            if write_out:
                if self.filesuffix:
                    self.gdir.write_pickle(mb_ds, 'mb_daily_{}'
                                           .format(self.filesuffix))
                else:
                    self.gdir.write_pickle(mb_ds, 'mb_daily')

        return mb_ds


class DailyMassBalanceModelWithSnow(DailyMassBalanceModel):
    """
    Include SnowCover, so that all classes can inherit from it and there is not conflicts
    """

    prefix = 'DailyMassBalanceModelWithSnow_'
    mb_name = prefix + 'MB'

    def __init__(self, gdir, mu_star=None, bias=None,
                 prcp_fac=None, snow_init=None, snowcover=None,
                 heights_widths=(None, None),
                 filename='climate_daily',
                 filesuffix='', cali_suffix='', snow_redist=True):

        super().__init__(gdir, mu_star=mu_star, bias=bias, prcp_fac=prcp_fac,
                         heights_widths=heights_widths, filename=filename,
                         filesuffix=filesuffix, param_ep_func=np.nanmean,
                         cali_suffix=cali_suffix)

        if snow_init is None:
           self.snow_init = np.atleast_2d(np.zeros_like(self.heights))
        else:
           self.snow_init = np.atleast_2d(snow_init)

        # todo: REPLACE THIS! It's just for testing
        rho_init = np.zeros_like(self.snow_init)
        rho_init.fill(100.)
        origin_date = dt.datetime(1961, 1, 1)
        if snowcover is None:
            self.snowcover = SnowFirnCover(self.heights, self.snow_init,
                                           rho_init, origin_date)
        else:
            self.snowcover = snowcover

        self._snow = np.nansum(self.snowcover.swe, axis=1)

    # todo: this shortcut is stupid...you can't set the attribute, it's just good for getting, but setting is needed in the calibration
    @property
    def snow(self):
        return np.nansum(self.snowcover.swe, axis=1)

    @property
    def snowcover(self):
        return self._snowcover

    @snowcover.setter
    def snowcover(self, value):
        self._snowcover = value

    def get_daily_mb(self, heights, date=None, **kwargs):
        # todo: implement
        raise NotImplementedError

    def get_monthly_mb(self, heights, year=None, fl_id=None):
        # todo: implement
        raise NotImplementedError

    def get_annual_mb(self, heights, year=None, fl_id=None):
        # todo: implement
        raise NotImplementedError


class BraithwaiteModel(DailyMassBalanceModelWithSnow):
    """
    Interface to a mass balance equation containing the Braithwaite melt term.

    Attributes
    ----------

    """

    cali_params_list = ['mu_ice', 'prcp_fac']
    cali_params_guess = OrderedDict(zip(cali_params_list, [6.5, 1.5]))
    prefix = 'BraithwaiteModel_'
    mb_name = prefix + 'MB'

    def __init__(self, gdir, mu_ice=None, mu_snow=None, bias=None,
                 prcp_fac=None, snow_init=None, snowcover=None,
                 heights_widths=(None, None),
                 filename='climate_daily',
                 filesuffix='', cali_suffix=''):
        """
        Implementing the temperature index melt model by Braithwaite.

        This is often said, but there is no good reference for this model
        from Braithwaite actually.
        # todo: find a good Braithwaite paper


        Parameters
        ----------
        gdir: `py:class:crampon.GlacierDirectory`
            The glacier directory to calculate the mass balance for.
        mu_ice: float or None
            The melt factor for ice (mm K-1 d-1). Default: None (retrieve from
            calibrated values).
        mu_snow: float or None
            The melt factor for snow (mm K-1 d-1). Default: None (retrieve from
            calibrated values).
        bias: float or None
            Bias of the mass balance model (mm K-1 d-1). Default: 0.
            # todo: check what the bis does and if I have applied it correctly.
        prcp_fac: float or None
            The correction factor to account for over-/undercatch biasses of
            the input precipitation values. It is defined the way that total
            precipitation determining the accumulation is the product of
            prcp_fac and the input precipitation. This means, for the case that
            there is no correction necessary prcp_fac equals 1. Default: None
            (retrieve from calibrated values).
        snow_init: np.array or None
        snowcover: `py:class:crampon.core.models.massbalance.SnowFirnCover`.
        heights_widths: tuple
            The flowline glacier heights and widths
        filename: str
            The meteorological input file name.
            # todo: call it different, e.g. 'meteo_file'.
        filesuffix: str
            Suffix for experiments
            # todo: this doesn't really make sense either, it is used for reading the climate file...

        References
        ----------

        """

        super().__init__(gdir, mu_star=mu_ice, bias=bias, prcp_fac=prcp_fac,
                         filename=filename, heights_widths=heights_widths,
                         filesuffix=filesuffix, snowcover=snowcover,
                         snow_init=snow_init, cali_suffix=cali_suffix)

        self.ratio_s_i = cfg.PARAMS['ratio_mu_snow_ice']

        if mu_ice is None:
            cali_df = gdir.get_calibration(filesuffix=cali_suffix)
            mu_ice = cali_df[self.__name__ + '_' + 'mu_ice']
        if (mu_ice is not None) and (mu_snow is None):
            mu_snow = mu_ice * self.ratio_s_i
        if (mu_ice is None) and (mu_snow is None):
            cali_df = gdir.get_calibration(filesuffix=cali_suffix)
            mu_snow = cali_df[self.__name__ + '_' + 'mu_ice'] * self.ratio_s_i
        if bias is None:
            cali_df = gdir.get_calibration(filesuffix=cali_suffix)
            if cfg.PARAMS['use_bias_for_run']:
                bias = cali_df[self.__name__ + '_' + 'bias']
            else:
                bias = pd.DataFrame(index=cali_df.index,
                                    data=np.zeros_like(cali_df.index,
                                                       dtype=float))
        if prcp_fac is None:
            cali_df = gdir.get_calibration(filesuffix=cali_suffix)
            prcp_fac = cali_df[self.__name__ + '_' + 'prcp_fac']

        self.mu_ice = mu_ice
        self.mu_snow = mu_snow
        self.bias = bias
        self.prcp_fac = prcp_fac

        #if snow_init is None:
        #    self.snow_init = np.atleast_2d(np.zeros_like(self.heights))
        #else:
        #    self.snow_init = np.atleast_2d(snow_init)

        ## todo: REPLACE THIS! It's just for testing
        #rho_init = np.zeros_like(self.snow_init)
        #rho_init.fill(100.)
        #origin_date = dt.datetime(1961, 1, 1)
        #if snowcover is None:
        #    self.snowcover = SnowFirnCover(self.heights, self.snow_init,
        #                                   rho_init, origin_date)
        #else:
        #    self.snowcover = snowcover

        #self._snow = np.nansum(self.snowcover.swe, axis=1)

        # Read file
        fpath = gdir.get_filepath(filename, filesuffix=filesuffix)
        with netCDF4.Dataset(fpath, mode='r') as nc:
            # time as nc time variable
            self.nc_time = nc.variables['time']
            self.nc_time_units = self.nc_time.units
            # time as array of datetime instances
            time = netCDF4.num2date(self.nc_time[:], self.nc_time.units)

            # identify the time span
            self.tspan_meteo = time
            self.tspan_meteo_dtindex = pd.DatetimeIndex(time)
            # self.years = np.unique([y.year for y in self.nc_time])

            # Read timeseries
            self.temp = nc.variables['temp'][:]
            if isinstance(self.prcp_fac, pd.Series):
                self.prcp = nc.variables['prcp'][:] * \
                            self.prcp_fac.reindex(index=time,
                                                  method='nearest').fillna(
                    value=self.param_ep_func(self.prcp_fac))
            else:
                self.prcp = nc.variables['prcp'][:] * self.prcp_fac
            self.tgrad = nc.variables['tgrad'][:]
            self.pgrad = nc.variables['pgrad'][:]
            self.ref_hgt = nc.ref_hgt

        # Public attrs
        self.temp_bias = 0.

    # todo: this shortcut is stupid...you can't set the attribute, it's just good for getting, but setting is needed in the calibration
    @property
    def snow(self):
        return np.nansum(self.snowcover.swe, axis=1)

    @property
    def snowcover(self):
        return self._snowcover

    @snowcover.setter
    def snowcover(self, value):
        self._snowcover = value

    def get_daily_mb(self, heights, date=None, **kwargs):
        """
        Calculates the daily mass balance for given heights.

        At the moment the mass balance equation is the simplest formulation:

        MB(z) = PRCP_FAC * PRCP_SOL(z) - mustar * max(T(z) - Tmelt, 0)

        where MB(z) is the mass balance at height z in mm w.e., PRCP_FAC is
        the precipitation correction factor, PRCP_SOL(z) is the solid
        precipitation at height z in mm w.e., mustar is the
        temperature sensitivity of the glacier (mm w.e. K-1 d-1), T(z) is the
        temperature and height z in (deg C) and Tmelt is the temperature
        threshold where melt occurs (deg C).

        The result of the model equation is thus a mass balance in mm w.e. d.1,
        however, for context reasons (dynamical part of the model), the mass
        balance is returned in m ice s-1, using the ice density as given in the
        configuration file. ATTENTION, the mass balance given is not yet
        weighted with the glacier geometry! For this, see method
        `get_daily_specific_mb`.

        Parameters
        ----------
        heights: ndarray
            Heights at which mass balance should be calculated.
        date: datetime.datetime
            Date at which mass balance should be calculated.
        **kwargs: dict-like, optional
            Arguments passed to the pd.DatetimeIndex.get_loc() method that
            selects the fitting melt parameters from the melt parameter
            attributes.

        Returns
        -------
        ndarray:
            Glacier mass balance at the respective heights in m ice s-1.
        """

        # index of the date of MB calculation
        ix = self.tspan_meteo_dtindex.get_loc(date, **kwargs)

        # Read timeseries
        itemp = self.temp[ix] + self.temp_bias
        itgrad = self.tgrad[ix]
        ipgrad = self.pgrad[ix]
        if isinstance(self.prcp_fac, pd.Series):
            try:
                iprcp_fac = self.prcp_fac[self.prcp_fac.index.get_loc(date,
                                                                      **kwargs)]
            except KeyError:
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            if pd.isnull(iprcp_fac):
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            iprcp = self.prcp_unc[ix] * iprcp_fac
        else:
            iprcp = self.prcp_unc[ix] * self.prcp_fac

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + itgrad * (heights - self.ref_hgt)

        tempformelt = self.get_tempformelt(temp)
        prcpsol, _ = self.get_prcp_sol_liq(iprcp, ipgrad, heights, temp)

        prcpsol *= self.prcp_fac_cycle_multiplier[date.dayofyear - 1]

        # TODO: What happens when `date` is out of range of the cali df index?
        # THEN should it take the mean or raise an error?
        if isinstance(self.mu_ice, pd.Series):
            try:
                mu_ice = self.mu_ice.iloc[
                    self.mu_ice.index.get_loc(date, **kwargs)]
            except KeyError:
                mu_ice = self.param_ep_func(self.mu_ice)
            if pd.isnull(mu_ice):
                mu_ice = self.param_ep_func(self.mu_ice)
        else:
            mu_ice = self.mu_ice

        if isinstance(self.mu_snow, pd.Series):
            try:
                mu_snow = self.mu_snow.iloc[
                    self.mu_snow.index.get_loc(date, **kwargs)]
            except KeyError:
                mu_snow = self.param_ep_func(self.mu_snow)
            if pd.isnull(mu_snow):
                mu_snow = self.param_ep_func(self.mu_snow)
        else:
            mu_snow = self.mu_snow

        if isinstance(self.bias, pd.Series):
            bias = self.bias.iloc[
                self.bias.index.get_loc(date, **kwargs)]
            # TODO: Think of clever solution when no bias (=0)?
        else:
            bias = self.bias


        # Get snow distribution from yesterday and determine snow/ice from it;
        # this makes more sense as temp is from 0am-0am and precip from 6am-6am
        snowdist = np.where(self.snowcover.age_days[range(
            self.snowcover.n_heights), self.snowcover.top_layer] <= 365)
        mu_comb = np.zeros_like(self.snowcover.height_nodes)
        mu_comb[:] = mu_ice
        np.put(mu_comb, snowdist, mu_snow)

        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - mu_comb * tempformelt - bias

        self.time_elapsed = date

        # todo: take care of temperature!?
        rho = np.ones_like(mb_day) * get_rho_fresh_snow_anderson(
            temp + cfg.ZERO_DEG_KELVIN)
        self.snowcover.ingest_balance(mb_day / 1000., rho, date)  # swe in m

        # todo: is this a good idea? it's totally confusing if half of the snowpack is missing...
        # remove old snow to make model faster
        if date.day == 1:  # this is a good compromise in terms of time
            oldsnowdist = np.where(self.snowcover.age_days > 365)
            self.snowcover.remove_layer(ix=oldsnowdist)

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / cfg.SEC_IN_DAY / cfg.RHO
        return icerate

    def update_snow(self, date, mb):
        """
        Updates the snow cover on the glacier after a mass balance calculation.

        Parameters
        ----------
        date: datetime.datetime
            The date for which to update the snow cover.
        mb: array-like
            Mass balance at given heights.

        Returns
        -------
        self.snow:
            The updated snow cover with the reference height, the date when the
            last fresh snow has fallen on the surface and the amount of snow
            present.
        """
        ix = self.time_elapsed.get_loc(date)

        if ix == 0:
            snow_today = np.clip(self.snow_init + mb, 0., None)
            self.snow = snow_today
        else:
            snow_today = np.clip((self.snow[-1] + mb), 0., None)
            self.snow = np.vstack((self.snow, snow_today))

        return snow_today


class HockModel(DailyMassBalanceModelWithSnow):
    """ This class implements the melt model by Hock (1999).

    The calibration parameter guesses are only used to initiate the calibration
    are rough mean values from [1]_.

     # todo:
    The albedo calculation via the Oerlemans method needs full access to the
    SnowFirnCover incl. its density. This is why we cannot delete snow older
    than 365 days in the `get_daily_mb` method and we even need to densify the
    snow every day!

    References
    ----------
    .. [1] : Gabbi, J., Carenzo, M., Pellicciotti, F., Bauder, A., & Funk, M.
            (2014). A comparison of empirical and physically based glacier
            surface melt models for long-term simulations of glacier response.
            Journal of Glaciology, 60(224), 1140-1154.
            doi:10.3189/2014JoG14J011
    """

    cali_params_list = ['mu_hock', 'a_ice', 'prcp_fac']
    cali_params_guess = OrderedDict(
        zip(cali_params_list, [1.8, 0.013, 1.5]))
    prefix = 'HockModel_'
    mb_name = prefix + 'MB'
    # todo: is there also a factor of 0.5 between a_snow and a_ice?

    def __init__(self, gdir, mu_hock=None, a_ice=None,
                 prcp_fac=None, bias=None, snow_init=None, snowcover=None,
                 filename='climate_daily', heights_widths=(None, None),
                 albedo_method=None, filesuffix='', cali_suffix=''):
        # todo: here we hand over tf to DailyMassBalanceModelWithSnow as mu_star...this is just because otherwise it tries to look for parameters in the calibration that might not be there
        super().__init__(gdir, mu_star=mu_hock, bias=bias, prcp_fac=prcp_fac,
                         heights_widths=heights_widths, filename=filename,
                         filesuffix=filesuffix, snow_init=snow_init,
                         snowcover=snowcover, cali_suffix=cali_suffix)

        if mu_hock is None:
            cali_df = gdir.get_calibration(self, filesuffix=cali_suffix)
            self.mu_hock = cali_df[self.__name__ + '_' + 'mu_hock']
        else:
            self.mu_hock = mu_hock
        if a_ice is None:
            cali_df = gdir.get_calibration(self, filesuffix=cali_suffix)
            self.a_ice = cali_df[self.__name__ + '_' + 'a_ice']
        else:
            self.a_ice = a_ice

        # todo: this is the big question:
        self.a_snow = cfg.PARAMS['ratio_mu_snow_ice'] * self.a_ice

        ipf = gdir.read_pickle('ipot_per_flowline')
        # flatten as we also concatenate flowline heights
        self.ipot = np.array([i for sub in ipf for i in sub])

    def get_daily_mb(self, heights, date=None, **kwargs):
        """
        Calculates the daily mass balance for given heights.

        At the moment the mass balance equation is the simplest formulation:

        MB(z) = PRCP_FAC * PRCP_SOL(z)
                - (mu_hock + mu_{snow/ice} * I_pot) * max(T(z) - Tmelt, 0)

        where MB(z) is the mass balance at height z in mm w.e., PRCP_FAC is
        the precipitation correction factor, PRCP_SOL(z) is the solid
        precipitation at height z in mm w.e., mu_hock is a melt factor
        ( mm d-1 K-1), a_{snow/ice} is the radiation coefficient for snow and
        ice, respectively, I_pot is the potential clear-sky direct solar
        radiation. T(z) is the temperature and height z in (deg C) and Tmelt
        is the temperature threshold where melt occurs (deg C).

        The result of the model equation is thus a mass balance in mm w.e. d.1,
        however, for context reasons (dynamical part of the model), the mass
        balance is returned in m ice s-1, using the ice density as given in the
        configuration file. ATTENTION, the mass balance given is not yet
        weighted with the glacier geometry! For this, see method
        `get_daily_specific_mb`.

        Parameters
        ----------
        heights: ndarray
            Heights at which mass balance should be calculated.
        date: datetime.datetime
            Date at which mass balance should be calculated.
        **kwargs: dict-like, optional
            Arguments passed to the pd.DatetimeIndex.get_loc() method that
            selects the fitting melt parameters from the melt parameter
            attributes.

        Returns
        -------
        ndarray:
            Glacier mass balance at the respective heights in m ice s-1.
        """

        # index of the date of MB calculation
        ix = self.tspan_meteo_dtindex.get_loc(date, **kwargs)

        # Read timeseries
        itemp = self.temp[ix] + self.temp_bias
        itgrad = self.tgrad[ix]
        ipgrad = self.pgrad[ix]
        if isinstance(self.prcp_fac, pd.Series):
            try:
                iprcp_fac = self.prcp_fac[self.prcp_fac.index.get_loc(date,
                                                                      **kwargs)]
            except KeyError:
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            if pd.isnull(iprcp_fac):
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            iprcp = self.prcp_unc[ix] * iprcp_fac
        else:
            iprcp = self.prcp_unc[ix] * self.prcp_fac

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + itgrad * (heights - self.ref_hgt)

        tempformelt = self.get_tempformelt(temp)
        prcpsol, _ = self.get_prcp_sol_liq(iprcp, ipgrad, heights, temp)

        prcpsol *= self.prcp_fac_cycle_multiplier[date.dayofyear - 1]

        # TODO: What happens when `date` is out of range of the cali df index?
        # THEN should it take the mean or raise an error?
        if isinstance(self.mu_hock, pd.Series):
            try:
                mu_hock = self.mu_hock.iloc[
                    self.mu_hock.index.get_loc(date, **kwargs)]
            except KeyError:
                mu_hock = self.param_ep_func(self.mu_hock)
            if pd.isnull(mu_hock):
                mu_hock = self.param_ep_func(self.mu_hock)
        else:
            mu_hock = self.mu_hock

        if isinstance(self.a_ice, pd.Series):
            try:
                a_ice = self.a_ice.iloc[
                    self.a_ice.index.get_loc(date, **kwargs)]
            except KeyError:
                a_ice = self.param_ep_func(self.a_ice)
            if pd.isnull(a_ice):
                a_ice = self.param_ep_func(self.a_ice)
        else:
            a_ice = self.a_ice

        if isinstance(self.a_snow, pd.Series):
            try:
                a_snow = self.a_snow.iloc[
                    self.a_snow.index.get_loc(date, **kwargs)]
            except KeyError:
                a_snow = self.param_ep_func(self.a_snow)
            if pd.isnull(a_snow):
                a_snow = self.param_ep_func(self.a_snow)
        else:
            a_snow = self.a_snow

        if isinstance(self.bias, pd.Series):
            bias = self.bias.iloc[
                self.bias.index.get_loc(date, **kwargs)]
            # TODO: Think of clever solution when no bias (=0)?
        else:
            bias = self.bias

        # Get snow distribution from yesterday and determine snow/ice from it;
        # this makes more sense as temp is from 0am-0am and precip from 6am-6am
        snowdist = np.where(self.snowcover.age_days[range(
            self.snowcover.n_heights), self.snowcover.top_layer] <= 365)
        a_comb = np.zeros_like(self.snowcover.height_nodes)
        a_comb[:] = a_ice
        np.put(a_comb, snowdist, a_snow)

        # todo: find a much better solution for leap years:
        try:
            i_pot = self.ipot[:, date.timetuple().tm_yday-1]
        except IndexError: # end of leap year
            i_pot = self.ipot[:, date.timetuple().tm_yday-2]
        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - (mu_hock + a_comb * i_pot) * tempformelt - bias

        self.time_elapsed = date

        # todo: take care of temperature!?
        rho = np.ones_like(mb_day) * get_rho_fresh_snow_anderson(
            temp + cfg.ZERO_DEG_KELVIN)
        self.snowcover.ingest_balance(mb_day / 1000., rho, date)  # swe in m

        # remove old snow to make model faster
        if date.day == 1:  # this is a good compromise in terms of time
            oldsnowdist = np.where(self.snowcover.age_days > 365)
            self.snowcover.remove_layer(ix=oldsnowdist)

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / cfg.SEC_IN_DAY / cfg.RHO
        return icerate


class PellicciottiModel(DailyMassBalanceModelWithSnow):
    """ This class implements the melt model by Pellicciotti et al. (2005).

    The calibration parameter guesses are only used to initiate the calibration
    are rough mean values from [1]_.

    IMPORTANT: At the moment gridded radiation values are only available back
    to 2004, so the model can't be used before this time until MeteoSwiss
    delivers the beta radiation version back until 1991 in mid of 2019
    (promised). We reflected the limited usage in the "calibration_timespan"
    attribute which is checked for in the calibration.

    # todo:
    The albedo calculation via the Oerlemans method needs full access to the
    SnowFirnCover incl. its density. This is why we cannot delete snow older
    than 365 days in the `get_daily_mb` method and we even need to densify the
    snow every day!

    Attributes
    ----------
    albedo: py:class:`crampon.core.models.massbalance.Albedo`
        An albedo object managing the albedo ageing.

    References
    ----------
    .. [1] : Gabbi, J., Carenzo, M., Pellicciotti, F., Bauder, A., & Funk, M.
            (2014). A comparison of empirical and physically based glacier
            surface melt models for long-term simulations of glacier response.
            Journal of Glaciology, 60(224), 1140-1154.
            doi:10.3189/2014JoG14J011

    """

    cali_params_list = ['tf', 'srf', 'prcp_fac']
    cali_params_guess = OrderedDict(zip(cali_params_list, [0.15, 0.28, 1.5]))
    calibration_timespan = (2005, None)
    prefix = 'PellicciottiModel_'
    mb_name = prefix + 'MB'

    def __init__(self, gdir, tf=None, srf=None, bias=None,
                 prcp_fac=None, snow_init=None, snowcover=None,
                 filename='climate_daily', heights_widths=(None, None),
                 albedo_method=None, filesuffix='', cali_suffix=''):
        # todo: Documentation
        """

        There are some restrictions for in the Brock et al. (2000) albedo
        update mode. For this mode the maximum daily temperature is needed,
        which is only available from 1971 on [1]_. Therefore the albedo update
        mode before 1972 automatically switches to the Oerlemans mode (see
        `py:class:crampon.core.models.massbalance.GlacierAlbedo` for details).
        if the ensemble update mode is chosen, the ensemble won't contain the
        Brock update method in this time range.

        In the albedo ensemble update mode, we are currently only able to work
        with the ensemble mean as long as there is not an EnsembleMassBalance
        class.

        Parameters
        ----------
        gdir
        tf
        srf
        bias
        prcp_fac
        snow_init
        snowcover
        filename
        heights_widths
        filesuffix


        References
        ----------
        [1]_.. : MeteoSwiss TabsD product description:
                 https://www.meteoswiss.admin.ch/content/dam/meteoswiss/fr/climat/le-climat-suisse-en-detail/doc/ProdDoc_TabsD.pdf
        """

        # todo: here we hand over tf to DailyMassBalanceModelWithSnow as mu_star...this is just because otherwise it tries to look for parameters in the calibration that might not be there
        super().__init__(gdir, mu_star=tf, bias=bias, prcp_fac=prcp_fac,
                         heights_widths=heights_widths, filename=filename,
                         filesuffix=filesuffix, snow_init=snow_init,
                         snowcover=snowcover, cali_suffix=cali_suffix)

        self.albedo = GlacierAlbedo(self.heights) #TODO: finish Albedo object

        # todo: improve this IF: default is "brock" and ELIFs would be appropriate...but what is the ELSE then?
        if albedo_method is None:
            self.albedo_method = cfg.PARAMS['albedo_method'].lower()
        else:
            self.albedo_method = albedo_method.lower()

        if self.albedo_method == 'brock':
            self.update_albedo = self.albedo.update_brock
        if self.albedo_method == 'oerlemans':
            self.update_albedo = self.albedo.update_oerlemans
        if self.albedo_method == 'ensemble':
            self.update_albedo = self.albedo.update_ensemble

        if tf is None:
            cali_df = gdir.get_calibration(self, filesuffix=cali_suffix)
            self.tf = cali_df[self.__name__ + '_' + 'tf']
        else:
            self.tf = tf
        if srf is None:
            cali_df = gdir.get_calibration(self, filesuffix=cali_suffix)
            self.srf = cali_df[self.__name__ + '_' + 'srf']
        else:
            self.srf = srf

        # get positive temperature sum since snowfall
        self.tpos_since_snowfall = np.zeros_like(self.heights)

    def get_daily_mb(self, heights, date=None, **kwargs):
        """
        Calculates the daily mass balance for given heights.

        The mass balance equation stems from Pellicciotti et al. (2005):

        If T is bigger than Tmelt:
        MB(z) = PRCP_FAC * PRCP_SOL(z) - TF * T(z) + SRF(1- alpha) * G(z)

        If T is smaller or equal Tmelt:
        MB(z) = PRCP_FAC * PRCP_SOL(z)

        where MB(z) is the mass balance at height z in mm w.e., PRCP_FAC is
        the precipitation correction factor, PRCP_SOL(z) is the solid
        precipitation at height z in mm w.e., TF is the so called
        temperature factor of the glacier (mm w.e. K-1 d-1), T(z) is the
        positive temperature at height z in (deg C), SRF is the so called
        shortwave radiation factor (m2 mm W-1 d-1), G is the incoming shortwave
        radiation (W m-2) and Tmelt is the temperature threshold where melt #
        occurs (deg C).

        The result of the model equation is thus a mass balance in mm w.e. d.1,
        however, for context reasons (dynamical part of the model), the mass
        balance is returned in m ice s-1, using the ice density as given in the
        configuration file. ATTENTION, the mass balance given is not yet
        weighted with the glacier geometry! For this, see method
        `get_daily_specific_mb`.

        Parameters
        ----------
        heights: ndarray
            Heights at which mass balance should be calculated.
        date: datetime.datetime
            Date at which mass balance should be calculated.
        **kwargs: dict-like, optional
            Arguments passed to the pd.DatetimeIndex.get_loc() method that
            selects the fitting melt parameters from the melt parameter
            attributes.

        Returns
        -------
        ndarray:
            Glacier mass balance at the respective heights in m ice s-1.
        """

        # index of the date of MB calculation
        # Todo: Timeit and compare with solution where the dtindex becomes a parameter
        ix = self.meteo.get_loc(date)

        # Read timeseries
        # Todo: what to do with the bias? We don't calculate temp this way anymore
        isis = self.meteo.sis[ix]

        tmean = self.meteo.get_tmean_at_heights(date, heights)
        prcpsol_unc, _ = self.meteo.get_precipitation_liquid_solid(date, heights)

        prcpsol_unc *= self.prcp_fac_cycle_multiplier[date.dayofyear - 1]

        if isinstance(self.prcp_fac, pd.Series):
            try:
                iprcp_fac = self.prcp_fac[self.prcp_fac.index.get_loc(date,
                                                                      **kwargs)]
            except KeyError:
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            if pd.isnull(iprcp_fac):
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            iprcp_corr = prcpsol_unc * iprcp_fac
        else:
            iprcp_corr = prcpsol_unc * self.prcp_fac

        if isinstance(self.tf, pd.Series):
            try:
                tf_now = self.tf.iloc[
                    self.tf.index.get_loc(date, **kwargs)]
            except KeyError:
                tf_now = self.param_ep_func(self.tf)
            if pd.isnull(tf_now):
                tf_now = self.param_ep_func(self.tf)
        else:
            tf_now = self.tf

        if isinstance(self.srf, pd.Series):
            try:
                srf_now = self.srf.iloc[
                    self.srf.index.get_loc(date, **kwargs)]
            except KeyError:
                srf_now = self.param_ep_func(self.srf)
            if pd.isnull(srf_now):
               srf_now = self.param_ep_func(self.srf)
        else:
            srf_now = self.srf

        if isinstance(self.bias, pd.Series):
            bias = self.bias.iloc[
                self.bias.index.get_loc(date, **kwargs)]
            # TODO: Think of clever solution when no bias (=0)?
        else:
            bias = self.bias

        # todo: improve this formulation: would be cool if Albedo was aware of the date so that we could handle this upstream
        if (date.year < cfg.PARAMS['tminmax_available']) and \
                (self.albedo_method == 'brock'):
            age_days = self.snowcover.age_days[np.arange(
                self.snowcover.n_heights), self.snowcover.top_layer]
            albedo = self.albedo.update_oerlemans(self.snowcover)
        elif (date.year < cfg.PARAMS['tminmax_available']) and \
                (self.albedo_method == 'ensemble'):
            raise NotImplementedError('We are not yet able te reduce the '
                                      'ensemble members.')
        else:
            # todo: this complicated as the different methods take input parameters! They should be able to get them themselves
            # todo: getting efficiently pos t sum between doesn't work with arrays???
            if self.albedo_method == 'brock':
                self.tpos_since_snowfall[iprcp_corr > 0.] = 0
                self.tpos_since_snowfall[iprcp_corr <= 0.] += \
                    climate.get_temperature_at_heights(self.meteo.tmax[ix],
                                                       self.meteo.tgrad[ix],
                                                       self.meteo.ref_hgt,
                                                       heights)[iprcp_corr <= 0.]
                albedo = self.update_albedo(t_acc=self.tpos_since_snowfall)
            elif self.albedo_method == 'oerlemans':
                age_days = self.snowcover.age_days[np.arange(
                    self.snowcover.n_heights), self.snowcover.top_layer]
                albedo = self.albedo.update_oerlemans(self.snowcover)
            elif self.albedo_method == 'ensemble':
                albedo = self.albedo.update_ensemble() # NotImplementedError
            else:
                raise ValueError('Albedo method is still not allowed.')


        # todo: where to put the bias here?
        # Pellicciotti(2005): melt really only occurs when temp is above Tt
        Tt = 1.  # Todo: Is this true? I think it should be -1! Ask Francesca if it's a typo in her paper!
        melt_day = tf_now * tmean + srf_now * (1 - albedo) * isis
        melt_day[tmean <= Tt] = 0.

        mb_day = iprcp_corr - melt_day

        # todo: take care of temperature!?
        rho = np.ones_like(mb_day) * get_rho_fresh_snow_anderson(
            tmean + cfg.ZERO_DEG_KELVIN)
        self.snowcover.ingest_balance(mb_day / 1000., rho, date)  # swe in m
        self.time_elapsed = date

        # todo: is this a good idea? it's totally confusing if half of the snowpack is missing...
        # remove old snow to make model faster
        if date.day == 1:  # this is a good compromise in terms of time
            oldsnowdist = np.where(self.snowcover.age_days > 365)
            self.snowcover.remove_layer(ix=oldsnowdist)

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / cfg.SEC_IN_DAY / cfg.RHO
        return icerate


class OerlemansModel(DailyMassBalanceModelWithSnow):
    """ This class implements the melt model by [Oerlemans (2001)]_.

    The calibration parameter guesses are only used to initiate the calibration
    are rough mean values from [Gabbi et al. (2014)]_.

    Attributes
    ----------

    References
    ----------
    [Oerlemans (2001)]_.. : Oerlemans, J. (2001). Glaciers and climate change.
        AA Balkema, Lisse
    [Gabbi et al. (2014)]_.. : Gabbi, J., Carenzo, M., Pellicciotti, F.,
        Bauder, A., & Funk, M. (2014). A comparison of empirical and physically
        based glacier surface melt models for long-term simulations of glacier
        response. Journal of Glaciology, 60(224), 1140-1154.
        doi:10.3189/2014JoG14J011
    """

    cali_params_list = ['c0', 'c1', 'prcp_fac']
    cali_params_guess = OrderedDict(zip(cali_params_list, [-110., 16., 1.5]))
    calibration_timespan = (2005, None)
    prefix = 'OerlemansModel_'
    mb_name = prefix + 'MB'

    def __init__(self, gdir, c0=None, c1=None, prcp_fac=None, bias=None,
                 snow_init=None, snowcover=None, filename='climate_daily',
                 heights_widths=(None, None), albedo_method=None,
                 filesuffix='', cali_suffix=''):
        # todo: here we hand over c0 to DailyMassBalanceModelWithSnow as mu_star...this is just because otherwise it tries to look for parameters in the calibration that might not be there
        super().__init__(gdir, mu_star=c0, bias=bias, prcp_fac=prcp_fac,
                         heights_widths=heights_widths, filename=filename,
                         filesuffix=filesuffix, snow_init=snow_init,
                         snowcover=snowcover, cali_suffix=cali_suffix)

        self.albedo = GlacierAlbedo(self.heights)  # TODO: finish Albedo object

        # todo: this is repeated code: make this function separate and lett PelliModel and this model call it!
        # todo: improve this IF: default is "brock" and ELIFs would be appropriate...but what is the ELSE then?
        if albedo_method is None:
            self.albedo_method = cfg.PARAMS['albedo_method'].lower()
        else:
            self.albedo_method = albedo_method.lower()

        if self.albedo_method == 'brock':
            self.update_albedo = self.albedo.update_brock
        if self.albedo_method == 'oerlemans':
            self.update_albedo = self.albedo.update_oerlemans
        if self.albedo_method == 'ensemble':
            self.update_albedo = self.albedo.update_ensemble

        if c0 is None:
            cali_df = gdir.get_calibration(self, filesuffix=cali_suffix)
            self.c0 = cali_df[self.__name__ + '_' + 'c0']
        else:
            self.c0 = c0
        if c1 is None:
            cali_df = gdir.get_calibration(self, filesuffix=cali_suffix)
            self.c1 = cali_df[self.__name__ + '_' + 'c1']
        else:
            self.c1 = c1

        # get positive temperature sum since snowfall
        self.tpos_since_snowfall = np.zeros_like(self.heights)

    def get_daily_mb(self, heights, date=None, **kwargs):
        """
        Calculates the daily mass balance for given heights.

        The mass balance equation stems from Oerlemans (2001):

        Qm = (1 - alpha(z) * SIS(z) + c0 + c1 * T(z)
        M = PRCP_FAC * PRCP_SOL(z) - (Qm * deltat) / (Lf * RhoW)


        where Qm is the energy available for melt (W m-2), alpha(z) is the
        albedo of the glacier surface, SIS(z) is the shortwave incoming solar
        radiation (W m-2), c0 an empirical factor (W m-2), c1 an empirical
        factor (W m-2 K-1), T(z) is the air temperature at height z in (deg C),
        PRCP_FAC is the precipitation correction factor, PRCP_SOL(z) is the
        solid precipitation at height z in mm w.e.,
        #Todo: Is this true? If we take daily temperatures we could also set deltat to one and then get a melt rate in m w.e. d-1!?
        deltat the integration time
        step (s), Lf the latent heat of fusion (J kg-1) and RhoW the density of
        water (kg m-3).

        The result of the model equation is thus a mass balance in mm w.e. d-1,
        however, for context reasons (dynamical part of the model), the mass
        balance is returned in m ice s-1, using the ice density as given in the
        configuration file. ATTENTION, the mass balance given is not yet
        weighted with the glacier geometry! For this, see method
        `get_daily_specific_mb`.

        Parameters
        ----------
        heights: ndarray
            Heights at which mass balance should be calculated.
        date: datetime.datetime
            Date at which mass balance should be calculated.
        **kwargs: dict-like, optional
            Arguments passed to the pd.DatetimeIndex.get_loc() method that
            selects the fitting melt parameters from the melt parameter
            attributes.

        Returns
        -------
        ndarray:
            Glacier mass balance at the respective heights in m ice s-1.
        """

        # index of the date of MB calculation
        ix = self.tspan_meteo_dtindex.get_loc(date, **kwargs)

        # Read timeseries
        isis = self.meteo.sis[ix]

        tmean = self.meteo.get_tmean_at_heights(date, heights)

        # Todo: make precip calculation method a member of cfg.PARAMS
        prcpsol_unc, _ = self.meteo.get_precipitation_liquid_solid(date,
                                                                   heights)

        prcpsol_unc *= self.prcp_fac_cycle_multiplier[date.dayofyear - 1]

        if isinstance(self.prcp_fac, pd.Series):
            try:
                iprcp_fac = self.prcp_fac[self.prcp_fac.index.get_loc(date,
                                                                      **kwargs)]
            except KeyError:
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            if pd.isnull(iprcp_fac):
                iprcp_fac = self.param_ep_func(self.prcp_fac)
            iprcp_corr = prcpsol_unc * iprcp_fac
        else:
            iprcp_corr = prcpsol_unc * self.prcp_fac

        if isinstance(self.c0, pd.Series):
            try:
                c0 = self.c0.iloc[
                    self.c0.index.get_loc(date, **kwargs)]
            except KeyError:
                c0 = self.param_ep_func(self.c0)
            if pd.isnull(c0):
                c0 = self.param_ep_func(self.c0)
        else:
            c0 = self.c0

        if isinstance(self.c1, pd.Series):
            try:
                c1 = self.c1.iloc[
                    self.c1.index.get_loc(date, **kwargs)]
            except KeyError:
                c1 = self.param_ep_func(self.c1)
            if pd.isnull(c1):
                c1 = self.param_ep_func(self.c1)
        else:
            c1 = self.c1

        if isinstance(self.bias, pd.Series):
            bias = self.bias.iloc[
                self.bias.index.get_loc(date, **kwargs)]
            # TODO: Think of clever solution when no bias (=0)?
        else:
            bias = self.bias

        # todo: this is repeated code with PellicciottiModel
        # todo: improve this formulation: would be cool if Albedo was aware of the date so that we could handle this upstream
        if (date.year < cfg.PARAMS['tminmax_available']) and \
                (self.albedo_method == 'brock'):
            age_days = self.snowcover.age_days[np.arange(
                self.snowcover.n_heights), self.snowcover.top_layer]
            albedo = self.albedo.update_oerlemans(self.snowcover)
        elif (date.year < cfg.PARAMS['tminmax_available']) and \
                (self.albedo_method == 'ensemble'):
            raise NotImplementedError('We are not yet able te reduce the '
                                      'ensemble members.')
        else:
            # todo: this complicated as the different methods take input parameters! They should be able to get them themselves
            # todo: getting efficiently pos t sum between doesn't work with arrays???
            if self.albedo_method == 'brock':
                self.tpos_since_snowfall[iprcp_corr > 0.] = 0
                self.tpos_since_snowfall[iprcp_corr <= 0.] += \
                    climate.get_temperature_at_heights(self.meteo.tmax[ix],
                                                       self.meteo.tgrad[
                                                           ix],
                                                       self.meteo.ref_hgt,
                                                       heights)[
                        iprcp_corr <= 0.]
                albedo = self.update_albedo(t_acc=self.tpos_since_snowfall)
            elif self.albedo_method == 'oerlemans':
                age_days = self.snowcover.age_days[np.arange(
                    self.snowcover.n_heights), self.snowcover.top_layer]
                albedo = self.albedo.update_oerlemans(self.snowcover)
            elif self.albedo_method == 'ensemble':
                albedo = self.albedo.update_ensemble()  # NotImplementedError
            else:
                raise ValueError('Albedo method is still not allowed.')

        # temperature here is in deg C ( Oerlemans 2001, p.48)
        # W m-2 = W m-2 + W m-2 + W m-2 K-1 * deg C
        qmelt = (1 - albedo) * isis + c0 + c1 * tmean
        # melt only happens where qmelt > 0.:
        qmelt = np.clip(qmelt, 0., None)

        # kg m-2 d-1 = W m-2 * s * J-1 kg
        melt = (qmelt * cfg.SEC_IN_DAY) / cfg.LATENT_HEAT_FUSION_WATER

        # kg m-2 = kg m-2 - kg m-2
        mb_day = iprcp_corr - melt

        # todo: take care of temperature!?
        rho = np.ones_like(mb_day) * get_rho_fresh_snow_anderson(
            self.meteo.get_tmean_at_heights(date,
                                            heights) + cfg.ZERO_DEG_KELVIN)
        self.snowcover.ingest_balance(mb_day / 1000., rho, date)  # swe in m
        self.time_elapsed = date

        # todo: is this a good idea? it's totally confusing if half of the snowpack is missing...
        # TODO: THIS IS ACTUALLY NEITHER HERE NOR IN PELLIMODEL ALLOWED; BECAUSE WE USE THE SNOW DENSITY TO DETERMINE THE TYPE, WHICH IN TURN DETERMINES THE ALBEDO; THUS WE EVEN NEED STO DENSIFY!
        # remove old snow to make model faster
        if date.day == 1:  # this is a good compromise in terms of time
            oldsnowdist = np.where(self.snowcover.age_days > 365)
            self.snowcover.remove_layer(ix=oldsnowdist)

        # return kg m-2 s-1 kg-1 m3 = m ice s-1
        icerate = mb_day / cfg.SEC_IN_DAY / cfg.RHO
        return icerate


@unique
class CoverTypes(IntEnum):
    """Available types of a cover."""
    snow = 0
    firn = 1
    poresclosed = 2
    ice = 3


class SnowFirnCover(object):
    """ Implements a an interface to a snow and/or firn cover.

    The idea is that the basic characteristics of a snow and firn cover, e.g.
    snow water equivalent and density are object properties. Different methods
    to densify the layers or to update the layer temperatures are implemented
    and operations are vectorized whenever possible.

    Attributes
    ----------
    heights: numpy.ndarray
        The heights at which the snow/firn cover shall be implemented (m).
    swe: numpy.ndarray
        The snow water equivalent of the snow/firn layers (m w.e.).
    sh: numpy.ndarray
        The snow height of the snow/firn layers (m).
    rho: numpy.ndarray
        The density of the snow/firn layers (kg m-3).
    origin: numpy.ndarray of datetime.datetime
        The origin date of the snow/firn layers.
    temp_profile: numpy.ndarray
        Temperature profile of the layers. If not given, it is initiated with
        zero degrees everywhere (K).
    liq_content: numpy.ndarray
        Liquid content of the snowpack (m w.e.). NOT YET IMPLEMENTED!
        # TODO: to be implemented
    pore_close: float
        Pore close-off density threshold (kg m-3).
    firn_ice_transition: float
        Density threshold for the transition from firn to ice.
    refreezing: bool
        If refreezing should be considered. Just switch off the refreezing for
        testing purposes or if you know what you are doing! Default: True.
    """

    def __init__(self, height_nodes, swe, rho, origin, temperatures=None,
                 liq_content=None, last_update=None, pore_close=None,
                 firn_ice_transition=None, refreezing=True):
        """
        Instantiate a snow and/or firn cover and its necessary methods.

        If no temperatures are given, the temperature profile is assumed to be
        homogeneous.

        Parameters
        ----------
        height_nodes: array-like
            The heights at which the snow/firn cover should be implemented.
        swe: np.array, shape of first dimension same as height_nodes
            Snow water equivalent of an initial layer (m w.e.).
        rho: np.array, same shape swe or None
            Density of an initial layer (kg m-3). If None, the inital density
            is set to NaN. # Todo: make condition for rho=None: if swe >0, then determine initial density from temperature
        origin: datetime.datetime or np.array with same shape as swe
            Origin date of the initial layer.
        temperatures: np.array, same shape as swe, optional
            Temperatures of initial layer (K). If not given, they will be set
            to NaN.
        liq_content: np.array, same shape as swe, optional
            Liquid content of given initial layers (m w.e.). If not given, it
            will be set to zero.
        last_update: np.array, same shape as swe, optional
            Last update of the initial layers.
        pore_close: float, optional
            Pore close-off density (kg m-3). If not given, CRAMPON tries to get
            in from the parameter file.
        firn_ice_transition: float, optional
            Density of the firn-ice transition (kg m-3). If not given, CRAMPON
            tries to get it from the parameter file (it take the ice density).
        refreezing: bool, optional
            Whether to calculate refreezing or not. Default: True (calculate
            refreezing).
        """

        # TODO: SHOULD THERE BE AN "INIT" in front of every parameter? Later we
        # don't use them anymore
        self.height_nodes = height_nodes
        self.init_swe = swe
        self.init_rho = np.full_like(swe, np.nan) if rho is None else rho
        self.init_sh = self.init_swe * (cfg.RHO_W / self.init_rho)
        if isinstance(origin, dt.datetime):
            self.init_origin = [origin] * self.n_heights
        else:
            self.init_origin = origin
        if liq_content is None:
            init_liq_content = np.zeros_like(swe)
            init_liq_content.fill(np.nan)
            self.init_liq_content = init_liq_content
        else:
            self.init_liq_content = liq_content

        if last_update is None:
            self.init_last_update = self.init_origin
        else:
            self.init_last_update = last_update

        # todo: we assume the snow is fresh - should we introduce a kwarg to be able to start at any state?
        self.init_age_days = np.zeros_like(self.init_swe)

        # parameters
        self.refreezing = refreezing
        # if not given, try and retrieve from cfg
        if pore_close is None:
                self.pore_close = cfg.PARAMS['pore_closeoff']
        else:
            self.pore_close = pore_close
        if firn_ice_transition is None:
            self.firn_ice_transition = cfg.RHO
        else:
            self.firn_ice_transition = firn_ice_transition

        # Init homog. grid for temperature modeling: zero at top, + downwards
        #self._tgrid_nodes = [np.array([0., self.init_sh[i]]) for i in
        #                     range(len(self.init_sh))]
        if temperatures is not None:
            self.init_temperature = temperatures
        else:
            # initiate with NaN
            self.init_temperature = \
                (np.ones_like(self.init_swe) * np.nan)

        init_array = np.zeros(
            (self.n_heights, np.atleast_2d(self.init_swe).shape[1] + 1))
        init_array.fill(np.nan)

        # we start putting the initial layer at index 0 (top of array!)
        try:
            self._swe = np.hstack((np.atleast_2d(self.init_swe).T, init_array))
            self._rho = np.hstack((np.atleast_2d(self.init_rho).T, init_array))
            if isinstance(origin, dt.datetime):
                self._origin = np.hstack((np.atleast_2d(
                    np.array([origin for i in range(self.n_heights)])).T,
                                          init_array))
            else:
                self._origin = np.hstack(
                    (np.atleast_2d(self.init_origin).T, init_array))
            self._last_update = np.hstack((
                np.atleast_2d(self.init_last_update).T, init_array))
            self._temperature = np.hstack((
                np.atleast_2d(self.init_temperature).T, init_array))
            self._liq_content = np.hstack(
                (np.ones_like(np.atleast_2d(self.init_liq_content).T),
                 init_array))
            self._age_days = np.hstack((np.atleast_2d(self.init_age_days).T,
                                        init_array))
            init_status = np.atleast_2d(swe).T
            self._status = np.hstack((np.atleast_2d(init_status), init_array))
        except ValueError:  # wrong dimensions: try without transpose
            self._swe = np.hstack((np.atleast_2d(self.init_swe), init_array))
            self._rho = np.hstack((np.atleast_2d(self.init_rho), init_array))
            if isinstance(origin, dt.datetime):
                self._origin = np.hstack((np.atleast_2d(
                    np.array([origin for i in range(self.n_heights)])),
                                          init_array))
            else:
                self._origin = np.hstack(
                    (np.atleast_2d(self.init_origin), init_array))
            self._last_update = np.hstack(
                (np.atleast_2d(self.init_last_update),
                 init_array))
            self._temperature = np.hstack((
                np.atleast_2d(self.init_temperature), init_array))
            self._liq_content = np.hstack(
                (np.atleast_2d(self.init_liq_content),
                 init_array))
            self._age_days = np.hstack((np.atleast_2d(self.init_age_days),
                                        init_array))
            #strrow = np.atleast_2d(swe).astype(str)
            init_status = np.atleast_2d(swe)
            #strrow.fill('')
            self._status = np.hstack((np.atleast_2d(init_status), init_array))
        # for temperature models that don't update layer by layer
        # must be initialized by hand with the desired form
        self._regridded_temperature = None
        self._ice_melt = np.zeros_like(swe)

    @property
    def swe(self):
        return self._swe

    @swe.setter
    def swe(self, value):
        self._swe = value

    @property
    def rho(self):
        return self._rho

    @rho.setter
    def rho(self, value):
        self._rho = value

    @property
    def sh(self):
        return self.swe * np.divide(cfg.RHO_W, self.rho)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        self._temperature = value

    @property
    def regridded_temperature(self):
        return self._regridded_temperature

    @regridded_temperature.setter
    def regridded_temperature(self, value):
        self._regridded_temperature = value

    @property
    def liq_content(self):
        return self._liq_content

    @liq_content.setter
    def liq_content(self, value):
        if value is not None:
            if (value > self.swe).any():
                raise ValueError('Liquid water content of a snow layer cannot '
                                 'be bigger than the snow water equivalent.')
            self._liq_content = value
        else:
            self._liq_content = 0.

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, value):
        self._origin = value

    @property
    def last_update(self):
        return self._last_update

    @last_update.setter
    def last_update(self, value):
        self._last_update = value

    @property
    def age_days(self):
        return self._age_days

    @age_days.setter
    def age_days(self, value):
        self._age_days = value

    @property
    def cold_content(self):
        """
        The cold content of the layer, expressed in J m-2.

        The sign of the cold content is negative!

        Returns
        -------
        ccont: float
            The cold content.
        """

        ccont = cfg.HEAT_CAP_ICE * (self.rho / cfg.RHO_W) * self.sh * \
                (self.temperature - cfg.ZERO_DEG_KELVIN)
        return ccont

    @property
    def refreezing_potential(self):
        """
        The refreezing potential expressed in kg m-2 or m w.e., respectively.

        the sign of the refreezing potential is negative!

        Returns
        -------
        rpot: float
            The refreezing potential.
        """
        rpot = self.cold_content / cfg.LATENT_HEAT_FUSION_WATER
        return rpot

    @property
    def status(self):
        """
        Status of the "porous medium".

        Can be either of "snow", "firn", "pore_closeoff" or "ice", depending on
        density thresholds.

        Returns
        -------

        """

        self._status = np.empty_like(self.swe)  # ensure size
        self._status[self.rho < cfg.PARAMS['snow_firn_threshold']] = CoverTypes['snow'].value
        self._status[(cfg.PARAMS['snow_firn_threshold'] <= self.rho) &
                     (self.rho < cfg.PARAMS['pore_closeoff'])] = \
            CoverTypes['firn'].value
        self._status[
            (cfg.PARAMS['pore_closeoff'] <= self.rho) &
            (self.rho < cfg.RHO)] = CoverTypes['poresclosed'].value
        self._status[self.rho >= cfg.RHO] = CoverTypes['ice'].value

        return self._status

    @property
    def n_heights(self):
        return len(self.height_nodes)

    @property
    def top_layer(self):
        """
        A pointer to the top layer in the snow and firn pack.

        Initially, when there are no layers in the pack, the top index will be
        -1!

        Returns
        -------
        top: np.ndarray
            Array giving the indices of the current top layer.
        """
        layers_bool = np.logical_or(np.isin(self.swe, [0.]),
                                    np.isnan(self.swe))
        top = np.argmax(layers_bool, axis=1) - 1
        return top

    @property
    def ice_melt(self):
        """

        Returns
        -------

        """
        return self._ice_melt

    @ice_melt.setter
    def ice_melt(self, value):
        self._ice_melt = value

    def to_dataset(self):
        """
        Convert the SnowFirnCover object into an xarray dataset.

        Returns
        -------
        ds: xr.Dataset
        """

        time = pd.to_datetime((np.nanmax(self.origin[~pd.isnull(self.origin)])
                               + dt.timedelta(days=np.nanmin(self.age_days))))
        print(time)

        ds = xr.Dataset({'swe': (['fl_id', 'layer', 'time'],
                                 np.atleast_3d(self.swe)),
                         'sh': (['fl_id', 'layer', 'time'],
                                np.atleast_3d(self.sh)),
                         'rho': (['fl_id', 'layer', 'time'],
                                 np.atleast_3d(self.rho)),
                         'origin': (['fl_id', 'layer', 'time'],
                                    np.atleast_3d(self.origin)),
                         'temperature': (['fl_id', 'layer', 'time'],
                                         np.atleast_3d(self.temperature)),
                         'liq_content': (['fl_id', 'layer', 'time'],
                                         np.atleast_3d(self.liq_content)),
                         'age_days': (['fl_id', 'layer', 'time'],
                                      np.atleast_3d(self.age_days)),
                         'last_update': (['fl_id', 'layer', 'time'],
                                      np.atleast_3d(self.last_update)),
                         'ice_melt': (['fl_id', 'time'],
                                      np.atleast_2d(self.ice_melt).T),
                         'heights': (['fl_id', 'time'],
                                     np.atleast_2d(self.height_nodes).T)},
                        coords={'fl_id': (['fl_id', ],
                                          np.arange(self.n_heights)),
                                'layer': (['layer', ],
                                          np.arange(self.swe.shape[1])),
                                'time': (['time', ], [time])
                                }
                        )

        return ds

    @classmethod
    def from_dataset(self, ds=None, path=None, **kwargs):
        """
        Create a SnowFirnCover object from an xarray Dataset.

        Both an xarray.Dataset or a path to an xarray.Dataset can be given
        (mutually exclusive). In both cases the Dataset must contain all
        variables to create a SnowFirnCover, i.e. 'SWE' (snow water equivalent)
         and either of 'SH' (snow height) or 'RHO' (density). The properties
         are then inferred from the input.

        Parameters
        ----------
        ds: xarray.Dataset
            An xarray.Dataset instance containing at least variables
        path:
            Path to an xarray.Dataset
        **kwargs: dict
            Further arguments to be passed to the SnowFirnCover object.

        Returns
        -------
        A SnowFirnCover instance.

        See Also
        --------
        to_dataset: create an xarray.Dataset from a SnowFirnCover instance.
        """

        if (ds is None) and (path is None):
            raise ValueError('Either of "path" or "ds" must be given.')
        if (ds is not None) and (path is not None):
            raise ValueError('Only one of "path" and "ds" can be given.')

        # length of time axis may only be one
        if len(ds.time) > 1:
            raise ValueError('Time dimension may only have length one.')

        ds = ds.isel(time=0)

        if path is not None:
            ds = xr.open_dataset(path)

        swe = ds.swe.values
        # we give preference to RHO (sh is a property based on rho and swe)
        try:
            rho = ds.rho.values
        except KeyError:
            try:
                rho = swe / ds.sh.values
            except KeyError:
                raise ValueError('Input dataset must contain either density ("'
                                 'RHO") or snow height ("SH").')

        # mixing datetime and np.nan was a stupid idea
        origin = ds.origin.values
        origino = origin.astype(object)
        for i, j in product(range(origin.shape[0]), range(origin.shape[1])):
            if origino[i][j] is None:
                origino[i][j] = np.nan
            else:
                origino[i][j] = pd.to_datetime(
                    np.datetime64(origino[i][j], 'ns'))

        try:
            temp = ds.temperature.values
        except:
            temp = None

        try:
            lc = ds.liq_content.values
        except KeyError:
            lc = None

        try:
            lu = ds.last_update.values
        except KeyError:
            lu = None
        sfc_obj = SnowFirnCover(ds.heights.values, swe, rho, origino,
                                temperatures=temp, liq_content=lc,
                                last_update=lu, **kwargs)

        return sfc_obj

    def get_type_indices(self, layertype):
        """
        Get the grid indices as tuples for a given type.

        Parameters
        ----------
        type: str
            Type to retrieve the indices for. Allowed are: 'snow', 'firn',
            'poresclosed' and 'ice'.

        Returns
        -------
        indices: list of tuples
            Indices where the given type occurs.
        """

        # TODO: this is doubled with self.status...but calling self.status is 387µs instead of 55µs
        if layertype == 'snow':
            return np.where(self.rho < cfg.PARAMS['snow_firn_threshold'])
        elif layertype == 'firn':
            return np.where((cfg.PARAMS['snow_firn_threshold'] <= self.rho) &
                            (self.rho < self.pore_close))
        elif layertype == 'poresclosed':
            return np.where((self.pore_close <= self.rho) &
                            (self.rho < self.firn_ice_transition))
        elif layertype == 'ice':
            return np.where(self.firn_ice_transition <= self.rho)
        else:
            raise ValueError('Type {} not accepted. Must be either of "snow", '
                             '"firn", "poresclosed" or "ice".'.format(layertype))

    def add_layer(self, swe, rho, origin, temperature=None, liq_content=None,
                  ix=None, merge_threshold=0.05):
        """
        Add a layer to the snow/firn pack.

        If no density is given, the fresh snow density will be calculated
        after Anderson (1973). For this we assume that the air temperature by
        the time of layer deposition equals the layer temperature

        Parameters
        ----------
        swe: np.array, same shape as self.height_nodes
            Snow water equivalent (m w.e.) of the layer to be added.
        rho: np.array, same shape as self.height_nodes or None
            # Todo: make rho a keyword!?
            Density (kg m-3) of the layer to be added.
        origin: datetime.datetime or pd.Timestamp
            Origin date of the layer to be added.
        temperature: np.array, same shape as self.height_nodes, optional
            Temperature of the layer to be added (K). If no temperature is
            given, we assume 273.15 K (0 deg C).
        liq_content: np.array, same shape as self.height_nodes, optional
            Liquid content of the layer to be added (m water). If no liquid
            content is given, we assume a dry layer (zero liquid content.
        ix: int
            Indices where to add the layer. Default: None (top).
        merge_threshold: float
            Layer height threshold (m) up to which layer height layers should
            be merged. Default: 0.05 m.
        """

        if len(swe) != self.n_heights:
            raise ValueError('Dimensions of SnowFirnCover and mass to be '
                             'added must match.')

        if rho is None:
            rho = np.ones_like(swe)
            rho[swe == 0.] = np.nan
            rho = get_rho_fresh_snow_anderson(temperature)

        if ix is None:
            insert_pos = self.top_layer + 1
        else:
            insert_pos = ix

        if (insert_pos >= self.swe.shape[1] - 1).any():
            shape_0 = self.swe.shape[0]
            shape_1 = self.swe.shape[1]
            bigger_array = np.empty((shape_0, shape_1 + 10))
            bigger_array.fill(np.nan)

            new_swe = bigger_array.copy()
            new_rho = bigger_array.copy()
            new_origin = bigger_array.copy().astype(object)
            new_temperature = bigger_array.copy()
            new_liq_content = bigger_array.copy()
            new_last_update = bigger_array.copy().astype(object)
            new_age_days = bigger_array.copy()
            new_status = bigger_array.copy()

            # status first, because it depends on the others
            new_status[:shape_0, :shape_1] = self.status
            self._status = new_status
            new_swe[:shape_0, :shape_1] = self.swe
            self._swe = new_swe
            new_rho[:shape_0, :shape_1] = self.rho
            self._rho = new_rho
            new_origin[:shape_0, :shape_1] = self.origin
            self._origin = new_origin
            new_temperature[:shape_0, :shape_1] = self.temperature
            self._temperature = new_temperature
            new_liq_content[:shape_0, :shape_1] = self.liq_content
            self._liq_content = new_liq_content
            new_last_update[:shape_0, :shape_1] = self.last_update
            self._last_update = new_last_update
            new_age_days[:shape_0, :shape_1] = self.age_days
            self._age_days = new_age_days

        merge = np.where((swe * np.divide(cfg.RHO_W, rho)) <= merge_threshold)[0]
        add = np.where(swe * np.divide(cfg.RHO_W, rho) > merge_threshold)[0]
        merge_ixtup = (np.arange(self.swe.shape[0])[merge],
                       np.clip(self.top_layer[merge], 0, None))  # clip avoids
        add_ixtup = (np.arange(self.swe.shape[0])[add], insert_pos[add])

        # TODO: What to do with the date? The date determines the rate of densification? /not in anderson()
        self.swe[add_ixtup] = swe[add]
        self.rho[add_ixtup] = rho[add]
        self.origin[add_ixtup] = origin
        self.age_days[add_ixtup] = 0.

        self.swe[merge_ixtup] = np.nansum((self.swe[merge_ixtup], swe[merge]), axis=0)
        self.rho[merge_ixtup] = np.nanmean((self.rho[merge_ixtup], rho[merge]), axis=0)
        self.origin[merge_ixtup] = origin  # TODO: or mx the dates? This might be important for ALBEDO CALCULATION
        self.age_days[merge_ixtup] = 0.  # Todo: here also a mean of ages?

        if temperature is not None:
            self.temperature[add_ixtup] = temperature[add]
            # TODO: MAKE TEMPERATURE CALCULATION CORRECT
            self.temperature[merge_ixtup] = \
                np.nanmean((self.temperature[merge_ixtup],
                            temperature[merge]), axis=0)
        else:
            self.temperature[add_ixtup] = np.nan
            self.temperature[merge_ixtup] = \
                np.nanmean((self.temperature[merge],
                            cfg.ZERO_DEG_KELVIN), axis=0)
        if liq_content is not None:
            self.liq_content[add_ixtup] = liq_content[add]
            self.liq_content[merge_ixtup] = \
                np.nansum((self.liq_content[merge_ixtup],
                           liq_content[merge]), axis=0)
        else:
            self.liq_content[add_ixtup] = 0.
            # merge liq_content stays as it is if we assume zero

    def remove_unnecessary_array_space(self):
        """
        Remove all-NaN

        Returns
        -------
        None
        """
        # IMPORTANT: Keep buffer of one NaN row, otherwise adding a layer at
        # toplayer+1 one next time adds it at the bottom then!)
        keep_indices = ~np.isnan(self.rho).all(axis=0)
        try:
            keep_indices[-1] = True
        except IndexError:  # anyway nothing to remove
            pass

        self.rho = self.rho[:, keep_indices]
        self.swe = self.swe[:, keep_indices]
        self.temperature = self.temperature[:, keep_indices]
        self._origin = self.origin[:, keep_indices]
        self._status = self._status[:, keep_indices]
        self._liq_content = self.liq_content[:, keep_indices]
        self._last_update = self.last_update[:, keep_indices]
        self._age_days = self.age_days[:, keep_indices]

    def remove_layer(self, ix=None):
        """
        Remove a layer from the snow/firn pack.

        Parameters
        ----------
        ix: int
            Where to remove the layer. Default: None (top).
        """

        if ix is not None:
            remove_ix = ix
        else:
            remove_ix = (np.arange(self.n_heights), self.top_layer)

        self.swe[remove_ix] = np.nan
        self.rho[remove_ix] = np.nan
        self.origin[remove_ix] = np.nan
        self.temperature[remove_ix] = np.nan
        self.liq_content[remove_ix] = np.nan
        self.status[remove_ix] = np.nan
        self.last_update[remove_ix] = np.nan
        self.age_days[remove_ix] = np.nan

        self.remove_unnecessary_array_space()

        # if layer to be removed are at the bottom, justify the arrays
        if (len(remove_ix) == 2) and (0 in remove_ix[1]):
            # This adjusts the position of the layers within the array
            self.swe = utils.justify(self.swe, invalid_val=np.nan, side='left')
            self.rho = utils.justify(self.rho, invalid_val=np.nan, side='left')
            self.temperature = utils.justify(self.temperature,
                                             invalid_val=np.nan,
                                             side='left')
            self.age_days = utils.justify(self.age_days, invalid_val=np.nan,
                                          side='left')
            # invalid_val=None ensures a good handlung of dt objects etc.
            self._liq_content = utils.justify(self.liq_content,
                                              invalid_val=None,
                                              side='left')
            # some more hassle with the dates
            intermed_orig = utils.justify(self.origin, invalid_val=None,
                                        side='left')
            intermed_orig[pd.isnull(intermed_orig)] = np.nan
            self._origin = intermed_orig
            intermed_last_update = utils.justify(self.last_update,
                                                 invalid_val=None, side='left')
            intermed_last_update[pd.isnull(intermed_last_update)] = np.nan
            self._last_update = intermed_last_update

    def melt(self, swe):
        """
        Removes a layer from the snow/firn pack.

        Parameters
        ----------
        swe: np.ndarray
            The snow water equivalent to be removed. Should be negative
            numbers.
        """

        if swe.shape[0] != self.swe.shape[0]:
            raise ValueError('Dimensions of SnowFirnCover and mass to be '
                             'removed must match.')

        # change sign to positive to make it comparable with positive SWE
        swe = np.negative(swe)

        swe = swe[:, None]
        cum_swe = np.fliplr(np.nancumsum(np.fliplr(self.swe), axis=1))
        remove_bool = (cum_swe <= swe) & (cum_swe != 0) & ~np.isnan(cum_swe)
        remove = np.where(remove_bool)

        old_swe = self.swe.copy()
        # For the first, we assume no percolation or similar
        self.swe[remove] = np.nan
        self.rho[remove] = np.nan
        self.temperature[remove] = np.nan
        self.liq_content[remove] = np.nan
        self.status[remove] = np.nan
        self.origin[remove] = np.nan
        self.last_update[remove] = np.nan
        self.age_days[remove] = np.nan

        mask = np.ma.array(old_swe, mask=np.invert(remove_bool))
        swe_to_remove = np.nansum(mask, axis=1)
        swe[:, 0] -= swe_to_remove

        top = self.top_layer
        current_swe_top = self.swe[np.arange(self.n_heights), top]
        to_reduce = swe[:, 0]
        new_swe_at_top = current_swe_top - to_reduce
        current_swe = self.swe.copy()
        current_swe[np.arange(self.n_heights), top] = new_swe_at_top
        self.swe = current_swe

        # just to be sure
        assert not (self.swe < 0.).all()

        # TODO: replace pseudo-code??? Does it have to refreeze every time if there is potential? Do we need impermable layers for this? Look at SFM2
        # if self.refreeze_pot > 0.:
        # let it refreeze
        # let the latent heat warm the snowpack up

    def ingest_balance(self, swe_balance, rho, date, temperature=None):
        """
        Ingest a mass balance as snow water equivalent.

        Parameters
        ----------
        swe_balance: np.array
            A balance for all height nodes (must have the shape of
            self.height_nodes) (m w.e. d-1).
        rho: float or np.array
            Density of the positive SWE. If array: must have the same shape as
            swe_balance, but will be clipped to use only positive swe_balance
            (kg m-3).
        temperature:


        Returns
        -------
        None
        """
        # Todo: This is a stupid thing: where should this go? should we make "add_layer" and "melt" private methods so that one *has* to use ingest_swe_balance (ensure that age_days is increased by one each time!)
        self.age_days += 1.

        if (swe_balance > 0.).any():
            swe = np.clip(swe_balance, 0, None)
            swe[swe == 0.] = np.nan

            rho[pd.isnull(swe)] = np.nan

            if temperature is None:
                temperature = swe.copy()
                temperature.fill(cfg.ZERO_DEG_KELVIN)
            self.add_layer(swe=swe_balance, rho=rho, origin=date,
                            temperature=temperature)

        if (swe_balance < 0.).any():
            self.melt(np.clip(swe_balance, None, 0))

        # todo: why dont we use self.remove_layers in self.melt, where unnecessary array space is removed?
        if date.day == 1:
            self.remove_unnecessary_array_space()

    def add_height_nodes(self, nodes):
        """
        If the glaciers advances, this is the way to tell the snow/firn cover.

        THIS FUNCTION IS NOT YET TESTED AT ALL!

        At the moment, and for simplicity reasons, the snow/firn cover at the
        lowest height index is copied to the added node(s).

        Parameters
        ----------
        nodes: array-like, 1D
            The heights that should be added to the existing height nodes.
        """

        # amend also the heights
        self.height_nodes = np.hstack((nodes, self.height_nodes))

        add_n = len(nodes)
        pos = 0  # we only add at the bottom
        self._swe = np.vstack(
            (np.repeat(np.atleast_2d(self.swe[pos]), add_n, axis=0), self.swe))
        self._rho = np.vstack(
            (np.repeat(np.atleast_2d(self.rho[pos]), add_n, axis=0), self.rho))
        self._origin = np.vstack((np.repeat(np.atleast_2d(self.origin[pos]),
                                            add_n, axis=0), self.origin))
        self._temperature = np.vstack((np.repeat(
            np.atleast_2d(self.temperature[pos]), add_n, axis=0),
                                       self.temperature))
        self._regridded_temperature = np.vstack((np.repeat(
            np.atleast_2d(self.regridded_temperature[pos]), add_n, axis=0),
                                       self.regridded_temperature))
        self._liq_content = np.vstack((np.repeat(
            np.atleast_2d(self.liq_content[pos]), add_n, axis=0),
                                       self.liq_content))
        self._last_update = np.vstack((np.repeat(
            np.atleast_2d(self.last_update[pos]), add_n, axis=0),
                                       self.last_update))
        self._status = np.vstack((np.repeat(np.atleast_2d(self.status[pos]),
                                            add_n, axis=0), self.status))

    def remove_height_nodes(self, nodes):
        """
        If the glacier retreats, this is the way to tell the snow/firn cover.

        THIS FUNCTION IS NOT YET TESTED AT ALL!

        The question is here is this should take indices or heights (precision
        problem)

        Parameters
        ----------
        nodes: array-like, 1D
            The heights that should be removed from the existing height nodes.
        """
        self.height_nodes = self.height_nodes

        remove_n = len(nodes)
        pos = 0
        self.height_nodes = self.height_nodes[remove_n:]
        self._swe = self._swe[remove_n:]
        self._rho = self._rho[remove_n:]
        self._origin = self._origin[remove_n:]
        self._temperature = self._temperature[remove_n:]
        self._regridded_temperature = self._temperature[remove_n:]
        self._liq_content = self._liq_content[remove_n:]
        self._last_update = self._last_update[remove_n:]
        self._status = self._status[remove_n:]

    def get_total_height(self):
        """
        Get total height (m) of the firn and snow cover together.

        Returns
        -------
        array of size self.n_heights:
            The total height of the snow and firn cover in meters.
        """
        return np.nansum(self.sh, axis=1)

    def get_total_volume(self, widths=None, map_dx=None):
        """
        Get total volume of the firn and snow cover together.

        If no flowline widths are given, a unit square meter is assumed.
        If flowline widths are given, also map_dx has to be given

        Parameters
        ----------
        widths: np.array, optional
            Flowline widths
        map_dx : float
            Map resolution (m).

        Returns
        -------

        """

        if (widths is not None) and (map_dx is None):
            raise ValueError('If widths are supplied, then also map_dx needs '
                             'to be supplied')

        if widths is None:
            return self.get_total_height()
        else:
            return self.get_total_height() * widths * \
                   cfg.PARAMS['flowline_dx'] * map_dx

    def get_lost_ice_volume(self, widths, map_dx):
        """
        Get the amount of accumulated ice melt that happened when there was no
        firn/snow present.

        Together with a second value from a later point in time
        Returns
        -------

        """
        return self.ice_melt * widths * cfg.PARAMS['flowline_dx'] * map_dx

    def get_height_by_type(self, layertype):
        """
        Get height of the snow and firn cover by its type.

        Parameters
        ----------
        layertype: str
            Either of "snow", "firn", "poresclosed" or ice.

        Returns
        -------
        h: np.array
            Array with summed up heights of the cover type.
        """
        h = np.zeros_like(range(self.n_heights), dtype=np.float32)
        where_type = self.get_type_indices(layertype)
        h_actual = np.nansum(np.atleast_2d(self.sh[where_type]), axis=0)
        h[where_type[0]] = h_actual[where_type[0]]
        return h

    def get_mean_density(self):
        """
        Get density of the overall snow and firn cover.

        The total density is calculated as the density of each layer weighted
        with its snow height.

        Returns
        -------
        total_rho: float
            The total density of the layer column at each height node.
        """

        # we need to mask, np.average is not able to ignore NaNs
        masked_rho = np.ma.masked_array(self.rho, np.isnan(self.rho))
        masked_swe = np.ma.masked_array(self.swe, np.isnan(self.swe))
        return np.average(masked_rho, axis=1, weights=masked_swe)

    def get_accumulated_layer_depths(self, which='center', ix=None):
        """
        Get the depths of every layer.

        Parameters
        ----------
        which: str
            At which position the layer depth shall be calculated. Allowed:
            "center", "top" and "bottom".
        ix: np.array
            Index array telling for which indices the layer depth is desired
        """

        # gives the bottom layer depth
        ovb_dpth = np.fliplr(np.nancumsum(np.fliplr(self.sh), axis=1))

        # alter for special wishes
        if which == 'center':
            # subtract the half of each layer height
            ovb_dpth -= self.sh / 2.
        elif which == 'top':
            ovb_dpth -= np.repeat(np.atleast_2d(
                self.swe[range(self.sh.shape[0]), self.top_layer]).T,
                                  ovb_dpth.shape[1], axis=1)
        elif which == 'bottom':
            pass
        else:
            raise ValueError(
                '"Where" method for calculating layer depth not understood. '
                'Allowed are "center", "bottom" and "top".')

        # NaN part becomes negative as with the NaN from NaN subtraction zero
        # is returned since NumPy 1.9.0
        ovb_dpth = np.clip(ovb_dpth, 0., None)

        if ix is None:
            return ovb_dpth
        else:
            return ovb_dpth[ix]

    def get_overburden_swe(self, ix=None):
        """Get overburden snow water equivalent for a specific layer (m w.e.).

        Parameters
        ----------
        ix: np.array
            Index array telling for which indices the overburden swe is desired

        Returns
        -------
        ovb_swe: np.array
            The snow water equivalent on top of the given layer (m w.e.).
        """
        ovb_swe = np.fliplr(np.nancumsum(np.fliplr(self.swe), axis=1))
        ovb_swe -= np.repeat(np.atleast_2d(
            self.swe[range(self.swe.shape[0]), self.top_layer]).T,
                             ovb_swe.shape[1], axis=1)
        # NaN part becomes negative as with the NaN from NaN subtraction zero
        # is returned since NumPy 1.9.0
        ovb_swe = np.clip(ovb_swe, 0., None)

        if ix is None:
            return ovb_swe
        else:
            return ovb_swe[ix]

    def get_overburden_mass(self, ix=None):
        """
        Get overburden mass for a specific layer (kg).

        Parameters
        ----------
        ix: np.array
            Index array telling for which indices the overburden mass is
            desired.

        Returns
        -------
        np.array
            The mass on top of the given layer (kg).
        """

        return self.get_overburden_swe(ix) * cfg.RHO_W

    def property_weighted_by_height(self, property):
        """
        Produce an array where a property is repeated as often as its layer
        height prescribes.

        Default is to resolve 1 mm thick layers. Layers thinner than 1 mm are
        ignored.

        Parameters
        ----------
        property: one of the self.properties
            The property that should be weighted by the layer height, e.g.
            self.swe or self.rho.

        Returns
        -------
        ret_val: np.array
            An array with repeated values of the property
        """
        sh = self.sh * 1000  # 1000 to resolve 1 mm thick layers
        sh[np.isnan(sh)] = 0.  # np.repeat fails with NaNs
        lens = sh.astype(int).sum(axis=1)
        m = int(1.05 * np.max(lens))  # extend 5 % for plotting beauty
        ret_val = np.zeros((property.shape[0], m))
        mask = (lens[:, None] > np.arange(m))
        ret_val[mask] = np.repeat(property.ravel(), sh.astype(int).ravel())
        ret_val[ret_val == 0] = np.nan

        return ret_val

    def remove_ice_layers(self):
        """
        Removes layers a the bottom (only there!) that have exceeded the
        threshold density for ice.

        Returns
        -------
        None
        """

        # this sets values to NaN
        ice_ix = self.get_type_indices('ice')
        # todo: a check that the layers to be removed are at the bottom!
        self.remove_layer(ice_ix)

    def update_temperature(self, date, airtemp, max_depth=15., deltat=86400,
                           lower_bound=cfg.ZERO_DEG_KELVIN):
        """
        Update the temperature profile of the snow/firn pack.

        The temperature is calculated on an equidistant grid until the maximum
        depth given to insure numeric stability. Temperatures are then mapped
        in a weighted fashion to the given layers.

        Parameters
        ----------
        date: dt.datetime
            The date when temperature should be updated.
        airtemp: float
            The air temperature at `date` (deg C). This is assumed to be the
            uppermost layer temperature
        max_depth: float
            Maximum depth until which the temperature profile is
            calculated (m). Default: 15.
        deltat: float
            Time step (s). Default: 86400 (one day).
        lower_bound: float
            Lower boundary condition temperature (K). Default: 273.15
            (temperate ice).

        Returns
        -------
        None
        """


    def update_temperature_huss(self, max_depth=5., max_depth_temp=0.,
                                surface_temp=-5.):
        """
        Choose the approach after Huss (2013).

        In Huss (2013), the temperature of the firn layers (firn only!) is not
        updated directly, only the refreezing potential is calculated. It is
        assumed that in each year the temperature linearly increases from
        -5 deg C at the surface to 0 deg C in a depth of 5 m. The calculated
        refreezing potential is completely used in order to produce temperate
        firn at the end of the mass budget year.
        The approach is a major simplification for our model as we (a) cannot
        quantify the refreezing potential according to the actual temperatures
        and (b) cannot resolve the refreezing potential for single layers.

        To make sense, the method should be applied around the "end of winter",
        i.e. roughly end of April in the European Alps.

        Parameters
        ----------
        max_depth: float
            Maximum positive depth (m) until which the temperature profile is
            calculated. Default: 5 m.
        max_depth_temp: float
            Temperature (deg C) assumed at the maximum depth until which the
            temperature profile is calculated ("max_depth"). Default: 0 deg C.
        surface_temp: float
            Temperature (deg C) assumed at the surface. Default: -5 deg C.

        Returns
        -------
        None
        """

        # get center depths
        depth = self.get_accumulated_layer_depths()
        temp = np.ones_like(depth)
        temp[np.isnan(depth)] = np.nan
        slope = (surface_temp - max_depth_temp) / (-max_depth)
        temp = slope * depth + surface_temp + cfg.ZERO_DEG_KELVIN
        temp = np.clip(temp, None, cfg.ZERO_DEG_KELVIN)
        # temperature is now the layer temperature at the center depth
        self.temperature = temp

    def update_temperature_glogem(self, tair_mean_month, melt_sum, dt=259200,
                                  simplify_density=False):
        """
        Update the temperature as GloGEM does it.

        The snow/firn cover is discretized in ten vertical steps of one meter
        thickness each. The density is assumed to be increasing from 300 kg m-3
        at the top to 650 kg m-3 at the bottom. The heat penetration equation
        is solved such that the temperature of the uppermost layer is the mean
        monthly air temperature and the equation is only applied during
        "winter", i.e. according to Huss and Hock (2005) the months with less
        than 2 mm w.e. of melt. Instead of taking the whole amount of
        refreezing we try to redistribute it among the existing layers. If the
        amount of refreezing is exhausted, the layer temperature is set to
        zero deg C.

        Huss and Hock (2015) write that the temperature is updated ten times
        per month to ensure numerical stability.

        Parameters
        ----------
        tair_mean_month: float
            Mean 2m air temperature of the month we are integrating within.
        melt_sum:
            Sum of melt (m w.e.) happening in the current month
        dt : float
            Integration time (s). # Todo: This is dangerous, other solution?
            Default: 3*86400=259200 s (3 days)
        simplify_density: bool
            Whether or not to simplify density. If True, we do what GloGEM
            assumes: The layer density increases for each meter of depth
            linearly between 300 kg m-3 and 650 kg m-3. Otherwise the actual
            modeled density is brought onto the GloGEM temperature grid.

        Returns
        -------

        """

        # calculate snow hieght after GloGEM
        #sh_oned = np.atleast_2d(np.ones(int(np.ceil(np.nanmax(self.get_total_height())))))
        sh_oned = np.atleast_2d(np.ones(10))
        sh_gg = np.repeat(sh_oned, self.n_heights, axis=0)

        if self.regridded_temperature is not None:
            told = self.regridded_temperature
        else:
            told = sh_gg * cfg.ZERO_DEG_KELVIN

        #if told.shape[1] > sh_gg.shape[1]:
        #    told = told[:, :sh_gg.shape[1]]
        #elif told.shape[1] < sh_gg.shape[1]:
        #    told = np.vstack((np.repeat(sh_oned * cfg.ZERO_DEG_KELVIN, sh_gg.shape[1]-told.shape[1]), told))
        #else:
        #    pass

        # GloGEM updates temperature only in Months with >2mm w.e. melt
        if melt_sum > 0.02:
            return told

        if simplify_density:
            rho_gg = np.fliplr(np.repeat(
                np.atleast_2d(np.linspace(300, 650, 10))[:len(sh_gg)],
                self.n_heights, axis=0))
        else:
            rho_gg = np.zeros_like(sh_gg)
            hgt_cumsum = self.get_accumulated_layer_depths(which='bottom')
            # iterate over heights
            for h in range(self.n_heights):
                # for each height, make bins
                merge_ix = np.digitize(hgt_cumsum[h], np.cumsum(sh_gg))
                for i in np.unique(merge_ix):
                    rho_gg[h, i] = np.average(
                        self.rho[h][np.where(merge_ix == i)],
                        weights=self.sh[h][np.where(merge_ix == i)])

        # Set top layer to the mean monthly temperature
        told[:, -1] = tair_mean_month + cfg.ZERO_DEG_KELVIN

        #rho_for_alpha = rho_gg
        #rho_for_alpha[:] = 600.
        #alpha = get_snow_thermal_diffusivity(rho_for_alpha, told)
        alpha = get_snow_thermal_diffusivity(rho_gg, told)
        told[:, 1:-1] = told[:, 1:-1] + alpha[:, 1:-1] / sh_gg[:, 1:-1] ** 2 * (
                told[:, 2:] - 2 * told[:, 1:-1] + told[:, 0:-2]) * dt

        self.regridded_temperature = told
        # Todo: Map the gridded temperature back to the original one => a task for np.digitize!?

    def apply_refreezing(self, exhaust=False):
        """
        Apply refreezing where there is potential.

        There is two options: Either the refreezing is applied according to the
        available liquid water content or the whole refreezing potential is
        exhausted.

        Parameters
        ----------
        exhaust: bool, optional
            Whether to exhaust the whole refreezing potential or not. Default:
            False.

        Returns
        -------
        None
        """

        # freeze the snow height status (needed later)
        sh = self.sh

        if exhaust:
            # refr. pot. is negative => minus
            self.swe = self.swe - self.refreezing_potential
            self.rho = (cfg.RHO_W * self.swe) / sh
            # TODO: LET THE TEMPERATURE RISE DUE TO LATENT HEAT OF FUSION!
            energy_phase_change = cfg.LATENT_HEAT_FUSION_WATER * cfg.RHO_W * \
                                  - self.refreezing_potential
            self.temperature = self.temperature + (energy_phase_change / (
                    cfg.HEAT_CAP_ICE * self.rho * self.sh))
        else:
            # see what is there in terms of liquid water content & melt at top
            # let refreeze only this
            raise NotImplementedError

    def densify_firn_huss(self, date, f_firn=2.4, poresclosed_rate=10.,
                          rho_f0_const=False):
        """
        Apply the firn densification after Huss (2013), Reeh (2008) and Herron
        and Langway (1980).

        Parameters
        ----------
        f_firn: float
           A factor empirically determined by Herron and Langway (1980), used
           by Huss 2013 to tune simulated to observed firn densification
        poresclosed_rate: float
            The rate with which the firn pack densifies once the pores are
            closed. Default: 10 kg m-3 a-1 (see Huss 2013).
        rho_f0_const: bool
            Whether or not the initial firn density should be constant or
            calculated from the layer's density (if available). Default: False
            (density calculated, if available).

        Returns
        -------
        None
        """

        # Todo: remove comment out
        #self.merge_firn_layers(date)

        # Mean annual surface temperature, 'presumably' firn temperature at 10m
        # depth (Reeh (2008)); set to 273 according to Huss 2013
        # TODO: this should be actually calculated
        T_ms = cfg.ZERO_DEG_KELVIN  # K

        # some model constant
        k1 = f_firn * 575. * np.exp(
            - cfg.E_FIRN / (cfg.R * T_ms))  # m(-0.5)a(-0.5)

        # what happens between 800 and pore close-off? We just replace it.
        firn_ix = self.get_type_indices('firn')
        poresclosed_ix = self.get_type_indices('poresclosed')

        # init array
        date_arr = np.zeros_like(self.last_update)
        # shape of current layers with today's date
        date_arr[~pd.isnull(self.last_update)] = date
        # make diff in days
        # Todo: if we use last_update here, we have a conflict: It's cool for the snow-firn model transition, but doesn't work any more later when we want to use the non-derived Reeh densifiation formula (we need the time span since the firn became firn
        tdarray = np.asarray(date_arr - self.last_update)
        # take care of NaN
        mask = tdarray == tdarray  # This gives array([True, False])
        tdarray[mask] = [x.days for x in tdarray[mask]]
        # get year fraction
        t = (tdarray / 365.25).astype(float)
        # t may not be zero, otherwise equation fails;so we make it min one day
        t[t == 0.] = 1/365.25

        if rho_f0_const:
            # initial firn density from Huss (2013) is 490, anyway...
            rho_f0 = cfg.PARAMS['snow_firn_threshold']  # kg m-3
        else:
            # todo:change to real init density again!!!!!
            #rho_f0 = self.init_density[firn_ix]
            rho_f0 = 550.

        # just to not divide by zero when a layer forms

            # "b" is in m w.e. a-1 as in Huss (2013). In Reeh(2008): m ice a-1
            b = self.get_overburden_swe() / t  # annual accumulation rate

            c_reeh_ice = k1 * np.sqrt(
                b * cfg.RHO / cfg.RHO_W)  # 550 kg m-3 < rho_f < 800 kg m-3

            # Huss 2013, equation 5
            rho_f = cfg.RHO - (cfg.RHO - rho_f0) * np.exp(
                -c_reeh_ice * t)

            self.rho[firn_ix] = rho_f[firn_ix]
            # do not update last_update here!
            # Todo: change that last_update cannot be updated here!
            # we need to update the last_update for those layers that have crossed the firn-poresclosed threshold = > otherwise we take the wrong basis for the pores_closeoff densification procedure
            self.last_update[np.where(self.status == CoverTypes['poresclosed'].value)] = date

        # TODO: HERE WE ASSUME ONE YEAR => t NEEDS TO BE ADJUSTED
        self.rho[poresclosed_ix] = self.rho[poresclosed_ix] + \
                                   poresclosed_rate * t[poresclosed_ix]
        self.last_update[poresclosed_ix] = date
        self.last_update[self.status == CoverTypes['poresclosed'].value] = date
        self.last_update[self.status == CoverTypes['ice'].value] = date

        # last but not least
        self.remove_ice_layers()
        self.remove_unnecessary_array_space()

    def densify_huss_derivative(self, date, f_firn=2.4, rho_f0_const=False,
                                poresclosed_rate=10.):
        """ Try and implement Reeh"""


        #raise NotImplementedError

        # Mean annual surface temperature, 'presumably' firn temperature at 10m
        # depth (Reeh (2008)); set to 273 according to Huss 2013
        # TODO: this should be actually calculated
        T_ms = cfg.ZERO_DEG_KELVIN  # K

        # some model constant
        k1 = f_firn * 575. * np.exp(
            - cfg.E_FIRN / (cfg.R * T_ms))  # m(-0.5)a(-0.5)

        # what happens between 800 and pore close-off? We just replace it.
        firn_ix = self.get_type_indices('firn')
        poresclosed_ix = self.get_type_indices('poresclosed')

        # careful with days!!!! (the difference might be 364 days or so)

        # TODO: ACTUALLY WE SHOULD DERIVE THE EQUATION AND THEN GO FROM TIME STEP TO TIME STEP
        # init array
        date_arr = np.zeros_like(self.last_update)
        # shape of current layers with today's date
        date_arr[~pd.isnull(self.last_update)] = date
        # make diff in days
        # Todo: if we use last_update here, we have a conflict: It's cool for the snow-firn model transition, but doesn't work any more later when we want to use the non-derived Reeh densifiation formula (we need the time span since the firn became firn
        tdarray = np.asarray(date_arr - self.last_update)
        # take care of NaN
        mask = tdarray == tdarray  # This gives array([True, False])
        tdarray[mask] = [x.days for x in tdarray[mask]]
        # get year fraction
        t = (tdarray / 365.25).astype(float)
        # t may not be zero, otherwise equation fails;so we make it min one day
        t[t == 0.] = 1 / 365.25

        # todo check if this is correct
        t = np.clip(t, None, 1)

        if rho_f0_const:
            # initial firn density from Huss (2013) is 490, anyway...
            rho_f0 = cfg.PARAMS['snow_firn_threshold']  # kg m-3
        else:
            # todo:attention, this is now different from the equation above
            rho_f0 = self.rho.copy()

            # just to not divide by zero when a layer forms
            # "b" is in m w.e. a-1 as in Huss (2013). In Reeh(2008): m ice a-1
            # todo: here we would need to get the accumulation of the last year if positive!)
            dummy = np.zeros_like(t)
            dummy[range(t.shape[0]), self.top_layer] = 1
            dummy = dummy != 0.
            #ovb_swe_mask = np.logical_or((~(t <= 1)), ~(dummy))  # last years accumulation
            ovb_swe_mask = ~np.logical_or(((t <= 1)), (dummy))
            masked = np.ma.array(self.get_overburden_swe(), mask=ovb_swe_mask,
                                 fill_value=np.nan)
            b = np.nanmax(masked, axis=1)
            #b = b.filled(masked.fill_value)  # to eliminate 1e20 from the mask
            #b = np.repeat(np.atleast_2d(b).T, self.rho.shape[1], axis=1)
            #b = np.clip(b, 0, None)
            b = np.apply_along_axis(lambda x: np.clip(x, 0, b), 0,
                                self.get_overburden_swe())
            # todo: there is a problem when there is only one layer: it does not densify, because it has no overburden weight and then it densifies to 900 directly
            b[b == 0.] = self.rho[b == 0.] / 2.

            c_reeh_ice = k1 * np.sqrt(
                b * cfg.RHO / cfg.RHO_W)  # 550 kg m-3 < rho_f < 800 kg m-3

            # Huss 2013, equation 5
            rho_f = cfg.RHO - (cfg.RHO - rho_f0) * np.exp(
                -c_reeh_ice * t)

            self.rho[firn_ix] = rho_f[firn_ix]
            # do not update last_update here!
            # Todo: change that last_update cannot be updated here!
            # we need to update the last_update for those layers that have crossed the firn-poresclosed threshold = > otherwise we take the wrong basis for the pores_closeoff densification procedure
            self.last_update[np.where(self.status == CoverTypes['poresclosed'].value)] = date

        # TODO: HERE WE ASSUME ONE YEAR => t NEEDS TO BE ADJUSTED
        self.rho[poresclosed_ix] = self.rho[poresclosed_ix] + \
                                   poresclosed_rate * t[poresclosed_ix]
        self.last_update[poresclosed_ix] = date
        self.last_update[self.status == CoverTypes['poresclosed'].value] = date
        self.last_update[self.status == CoverTypes['ice'].value] = date

        # last but not least
        self.remove_ice_layers()
        self.remove_unnecessary_array_space()

    def densify_firn_barnola(self, date, beta=-29.166, gamma=84.422,
                             delta=-87.425, epsilon=30.673):
        """
        Apply the firn densification after Barnola et al. (1990)

        Parameters
        ----------
        beta: float
           Empirical factor (see Barnola et al. (1990)).
        gamma: float
           Empirical factor (see Barnola et al. (1990)).
        delta: float
           Empirical factor (see Barnola et al. (1990)).
        epsilon: float
           Empirical factor (see Barnola et al. (1990)).

        Returns
        -------
        None

        """

        firn_ix = self.get_type_indices('firn')
        poresclosed_ix = self.get_type_indices('poresclosed')
        fp_ix = (np.append(firn_ix[0], poresclosed_ix[0]),
                 np.append(firn_ix[1], poresclosed_ix[1]))

        a0 = 25400
        k1 = a0 * np.exp(-60000 / (cfg.R * self.temperature))

        # TODO: dt is in seconds! which unit is rho here=?
        date_arr = np.zeros_like(self.last_update)
        date_arr[~pd.isnull(self.last_update)] = date
        tdarray = np.asarray(date_arr - self.last_update)
        mask = tdarray == tdarray
        tdarray[mask] = [x.days for x in tdarray[mask]]
        days = tdarray.astype(float)

        days2reduce = days.copy()

        for dt in (np.ones(int(np.nanmax(days))) * 24 * 3600):

            current_rho = self.rho
            si_ratio = current_rho / cfg.RHO

            f_b800 = 10 ** ((beta * si_ratio ** 3) +
                            (gamma * si_ratio ** 2) +
                            (delta * si_ratio) +
                            epsilon)
            f_a800 = ((3. / 16.) * (1 - si_ratio)) / \
                     (1. - (1. - si_ratio) ** 0.3) ** 3.

            f = np.zeros_like(current_rho)
            f[current_rho < 800.] = f_b800[current_rho < 800.]
            f[current_rho >= 800.] = f_a800[current_rho >= 800.]

            p_over = self.get_overburden_mass() * cfg.G

            try:
                p_bubble = self.bubble_pressure  # if pores closed
            except AttributeError:
                p_bubble = 0.

            # time factor tells whether layer still needs 2 be densified or not
            # TODO: with this we just densify the layers n times, but we don't account for WHEN they were densified (temperature, current density!)
            time_factor = (days2reduce > 0.).astype(int)
            drho = time_factor * k1 * current_rho * f * \
                   ((p_over - p_bubble)/10**6) ** 3 * dt

            self.rho[fp_ix] += drho[fp_ix]

            # do not forget
            self.last_update[fp_ix] = date

            days2reduce -= 1

        # last but not least
        self.remove_ice_layers()
        self.remove_unnecessary_array_space()

    # Todo: what to do with rhof?
    def densify_snow_anderson(self, date, eta0=3.7e7, etaa=0.081, etab=0.018,
                              snda=2.8e-6, sndb=0.042, sndc=0.046, rhoc=150.,
                              rhof=100., target_dt=24*3600):
        """
        Snow densification according to [Anderson (1976)]_.

        The values for the parameters are taken from the SFM2 model (Essery,
        2015) or Anderson (1976), respectively. As said in the ISBA-ES
        model description (Boone (2009)), snda, sndb, sndc and rhoc are
        actually site-specific parameters, but adopted for ISBA-ES (and SFM2
        as well)  The biggest problem might be that they are not tested on
        perennial snow.

        Parameters
        ----------
        date: dt.datetime
            Date for which to update the snow density.
        eta0: float
            Reference snow viscosity (Pa s). Default: 3.7e7 (`Essery 2018`_).
        etaa: float
            Snow viscosity parameter (1/K). Default: 0.081 (`Essery 2018`_).
        etab: float
            Snow viscosity parameter (m3 kg-1). Default: 0.018
            (`Essery 2018`_).
        snda: float
            Snow densification parameter (s-1). Default 2.8e-6
            (`Essery 2018`_).
        sndb: float
            Snow densification parameter (K-1). Default 0.042 (`Essery 2018`_).
        sndc: float
            Snow densification parameter (m3 kg-1). Default 0.046
            (`Essery 2018`_).
        rhoc: float
            Critical snow density (kg m-3). Default: 150. (`Essery 2018`_).
        rhof: float
            Fresh snow density (kg m-3). Default 100. (`Essery 2018`_).
        target_dt: int
            Target integration time in seconds (the time step over which the
            equation should be integrated). Default: 24 * 3600 (one day).

        Returns
        -------
        None

        .. _Essery 2018:
        https://github.com/RichardEssery/FSM2/blob/master/src/SNOW.F90
        """

        Tm_k = cfg.PARAMS['temp_melt'] + cfg.ZERO_DEG_KELVIN

        # Todo: CHECK HERE IF DATE IS MORE THAN ONE DAY AWAY FROM LAST UPDATE?
        deltat = target_dt

        rho_old = self.rho
        temperature = self.temperature

        # create cumsum, then subtract top layer swe, clip and convert to mass
        ovb_mass = self.get_overburden_mass()

        rho_new = rho_old.copy()
        insert_ix = (self.swe > 0.) & \
                    (self.status == CoverTypes['snow'].value) & \
                    (~np.isnan(self.swe))
        #rho_new[insert_ix] = (rho_old + \
        #              (
        #                       rho_old * cfg.G * ovb_mass * deltat / eta0) * \
        #               np.exp(etaa * (temperature - Tm_k) - etab *
        #                      rho_old) + deltat * \
        #               rho_old * snda * np.exp(
        #    sndb * (temperature - Tm_k) - sndc * np.clip(
        #        rho_old - rhoc, 0., None)))[insert_ix]
        temperature_sub = self.temperature - Tm_k
        clippie = np.clip(rho_old - rhoc, 0., None)
        #rho_topnew=rho_old.copy()
        rho_new[insert_ix] = \
        self._anderson_equation(rho_old, ovb_mass, deltat, eta0, etaa,
                 temperature_sub, etab, snda, sndb, sndc, clippie,
                 G=cfg.G)[insert_ix]
        #rho_new[insert_ix] = rho_topnew[insert_ix]
        self.rho = rho_new
        self.last_update[insert_ix] = date

    @staticmethod
    @numba.jit(nopython=True)
    def _anderson_equation(rho_old, ovb_mass, deltat, eta0, etaa, temperature_sub, etab, snda, sndb, sndc, clippie, G=cfg.G):
        """ This is just the actual equation, made faster with numba"""
        rho_new = (rho_old + (
                    rho_old * G * ovb_mass * deltat / eta0) * np.exp(etaa * (
                    temperature_sub) - etab * rho_old) + deltat * rho_old * snda * np.exp(
            sndb * (temperature_sub) - sndc * clippie))
        return rho_new

    def merge_firn_layers(self, date):
        """
        E.g. application:
        If a layer bunch is older than one year, collapse them into one firn
        layer
        (2) Merge neighbor layers below a layer thickness of 2cm? WHAT TO DO
        WITH THE ORIGIN DATE?

        Parameters
        ----------
        date: datetime.datetime
            The date determines the origin date of the merged layer.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def merge_layers(self, min_sh=0.02):
        """
        Merge similar neighbor layers inplace by a minimum height criterion.

        Parameters
        ----------
        min_sh: float
            Minimum allowed height for a snow layer im meters. Default: 0.02 m.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def return_state(self, param='swe', dataset=False):
        """This should be a function that can be called to get the current
        status of the snow/firn cover as numpy arrays or xr.Dataset. If put in
        a loop, the history can be retrieved."""

        if dataset:
            raise NotImplementedError
        else:
            raise NotImplementedError


def get_rho_fresh_snow_anderson(tair, rho_min=None, df=1.7, ef=15.):
    """
    Get fresh snow density after Anderson (1976).

    Parameters
    ----------
    tair: np.array
        Air temperature during snowfall (K).
    rho_min: float
        Minimum density allowed (kg m-3). Default: None (from configuration).
        With integration steps of approximately one day, probably 100 kg m-3 is
        a reasonable value, as in contrast to the 50 kg m-3 suggested for
        really fresh snow right after deposition.
    df: float
        Parameter according to Anderson (1976) (K). Default: 1.7
    ef: float
        Parameter according to Anderson (1976) (K). Default: 15.

    Returns
    -------
    rho_fresh: np.array
        Density of fresh snow.
    """

    if rho_min is None:
        rho_min = cfg.PARAMS['rho_min_snow']

    # we do not use eq.17 from Essery(2013), bcz we integrate o. 1 day already
    curve = np.array(np.clip(df * (tair - (
                cfg.PARAMS['temp_melt'] + cfg.ZERO_DEG_KELVIN) + ef) ** 1.5,
                    0., None))
    # NaN happens at low temperatures due to square root
    curve[np.isnan(curve)] = 0.
    rho_fresh = rho_min + curve

    return rho_fresh


def get_thermal_conductivity_yen(rhos, clambda=2.22, nlambda=1.88):
    """
    Compute the thermal conductivity after Yen (1981).

    Parameters
    ----------
    rhos: np.array
        The snow density
    clambda: float
        Parameter (W m-1 k-1).
        Default: 2.22 (Douville et al. (1992)).
    nlambda: float
        Parameter (W m-1 k-1).
        Default: 1.88 (Douville et al. (1992)).

    Returns
    -------
    tc: float
        Thermal conductivity.
    """

    return clambda * (rhos / cfg.RHO_W) ** nlambda


def get_snow_thermal_diffusivity(rho, temperature):
    """
    Get the thermal diffusivity for snow/firn.

    Parameters
    ----------
    rho: np.array
        Density of the snow/firn.
    temperature: np.array
        Temperature of the snow/firn.

    Returns
    -------
    alpha: np.array
        Thermal diffusivity (m2 s-1).
    """
    kt = get_thermal_conductivity_yen(rho)  # (W m-1 K-1)
    # Spec. heat cap. Cuffey & Paterson 2010 p. 400 (J kg-1 K-1)
    c = 152.5 + 7.122 * temperature
    alpha = kt / (rho * c)  # thermal diffusivity (m2 s-1)
    return alpha


def get_rho_dv(v_f1, v_f2, rho_f1, rho_f2, delta_v):
    """
    Calculate the density of volume change à la Farinotti.

    Parameters
    ----------
    v_f1: float
        Firn/snow volume at time 1 (m3).
    v_f2: float
        Firn/snow volume at time 2 (m3).
    rho_f1: float
        Firn/snow density of the volume at time 1 (kg m-3).
    rho_f2: float
        Firn/snow density of the volume at time 2 (kg m-3).
    delta_v: float
        Volume change of snow, firn and ice (!) between time 1 and time 2 (m3).

    Returns
    -------
    rho_dv: float
        Density of the volume change between time 1 and time 2 (kg m-3).
    """
    # get the volume for each point on the flowline.
    rho_dv = (v_f2 - v_f1 + delta_v) * (cfg.RHO - rho_f2) / delta_v + v_f1 * (
                rho_f1 - rho_f2) / delta_v + rho_f2
    return rho_dv


class GlacierAlbedo(object, metaclass=SuperclassMeta):
    """Interface to the glacier albedo, implementing various update methods.

    This class automatically handles the surface type classes snow, firn and
    ice. The albedo can either be updated using single methods or using all
    implemented approaches in an ensemble fashion.

    Attributes
    ----------
    heights: array-like
        The
    surface: array-like

    """

    def __init__(self, x, alpha=None, snow_ix=None, firn_ix=None,
                 ice_ix=None, standard_method_snow='Brock',
                 standard_method_firn=None, standard_method_ice=None,
                 a_snow_init=0.9, a_firn_init=0.5):
        """
        Instantiate a snow and/or firn cover and its necessary methods.

        Parameters
        ----------
        x: array-like
            Points at which the albedo should be updated.
        alpha: array-like, same shape as x
            Initial alpha.
        snow_ix: array
            Indices where there is snow.
        firn_ix: array
            Indices where there is firn.
        ice_ix: array
            Indices where there is ice.
        standard_method_snow: str
            Standard method of albedo ageing for surface type "snow". Default:
            Brock (2000).
        standard_method_firn: str
            Standard method of albedo ageing for surface type "firn".
        standard_method_ice: str
            Standard method of albedo ageing for surface type "ice".
        a_snow_init: float
            Initial (maximum) albedo for snow. Default: 0.9
        a_firn_init: float
            Initial (maximum albedo for firn. Default: 0.5
        """
        self.x = x.copy()
        self.a_snow_init = a_snow_init
        self.a_firn_init = a_firn_init

        if alpha is None:
            self.x[snow_ix] = self.a_snow_init

        self.tmax_avail = cfg.PARAMS['tminmax_available']

    def update_brock(self, p1=0.86, p2=0.155, t_acc=None):
        """
        Update the snow albedo using the method in [Brock et al. (2000)]_.

        Parameters
        ----------
        p1: float, optional
           Empirical coefficient, described by the albedo of fresh snow (for
           Ta=1 deg C). Default: (see Pellicciotti et al., 2005 and Gabbi et
           al, 2014).
        p2: float, optional
           Empirical coefficient. Default: (see Pellicciotti et al., 2005 and
           Gabbi et al., 2014)
        t_acc: array-like
            Accumulated daily maximum temperature > 0 deg C since snowfall.
            Default: None (calculate).

        References
        ----------
        [Brock et al. (2000)]_.. : Brock, B., Willis, I., & Sharp, M. (2000).
            Measurement and parameterization of albedo variations at Haut
            Glacier d’Arolla, Switzerland. Journal of Glaciology, 46(155),
            675-688. doi:10.3189/172756500781832675
        """
        # TODO: Maybe change this and make t_acc a non-keyword parameter
        # TODO: Maybe make t_acc a cached property in the GlacierMeteo class!
        if t_acc is None:
            t_acc = self.get_accumulated_temperature()

        # todo: check clipping with francesca: The function can become greater than one!
        # clip at 1. so that alpha doesn't become bigger than p1
        alpha = p1 - p2 * np.log10(np.clip(t_acc, 1., None))
        return alpha

    def update_oerlemans(self, snowcover, a_frsnow_oerlemans=0.75,
                         a_firn_oerlemans=0.53, a_ice_oerlemans=0.34,
                         t_star=21.9, d_star=0.032, event_thresh=0.02):
        """
        Update the snow albedo using the [Oerlemans & Knap (1998)]_ equations.

        This method accounts for the albedo transition for small snow depths on
        a glacier as well. In the original paper, there is always a
        characteristic ice albedo returned as background. This is probably
        because the station  they used was on the glacier tongue. Here we
        extend the model to use the ice albedo as background albedo if there is
        ice below, otherwise the characteristic firn albedo. If a snowfall
        event depth is smaller than a defined threshold then the background
        albedo is returned without mixing.

        Parameters
        ----------
        snowcover: `py:class:crampon.core.models.massbalance.SnowFirnCover`
            A SnowFirnCover object.
        a_frsnow_oerlemans: float, optional
            Characteristic albedo of fresh snow. Default: 0.75 (see Oerlemans &
            Knap (1998)).
        a_firn_oerlemans: float, optional
            Characteristic albedo of firn. Default: 0.53 (see Oerlemans & Knap
            (1998)).
        a_ice_oerlemans: float, optional
            Characteristic albedo of ice. Default: 0.34  (see Oerlemans & Knap
            (1998)).
        t_star: float, optional
            Typical time scale determining how fast a fresh snowfall reaches
            the firn albedo (d). Default: 21.9 (see Oerlemans & Knap (1998)).
        d_star: float, optional
            Characteristic scale for snow depth (m). Default: 0.032  (see
            Oerlemans & Knap (1998)).
        event_thresh: float, optional
            The snowfall event threshold in meters that defines from which
            depth on a fresh snowfall has an influence on the albedo. Default:
            0.02m.

        References
        ----------
        [Oerlemans & Knap (1998)]_.. : Oerlemans, J., & Knap, W. (1998). A 1
            year record of global radiation and albedo in the ablation zone of
            Morteratschgletscher, Switzerland. Journal of Glaciology, 44(147),
            231-238. doi:10.3189/S0022143000002574
        """

        age_days = snowcover.age_days[np.arange(snowcover.n_heights),
                                      snowcover.top_layer]
        snow_depth = snowcover.get_height_by_type('snow')
        firn_depth = snowcover.get_height_by_type('firn')

        # albedo of snow after i days (eq. 4 in paper)
        alpha_i_s = a_firn_oerlemans + \
                    (a_frsnow_oerlemans - a_firn_oerlemans) * \
                    np.exp(-age_days / t_star)

        # background albedo
        a_background = np.ones_like(snow_depth) * a_ice_oerlemans
        a_background[firn_depth > 0.] = a_firn_oerlemans

        # total albedo - smooth transition to ice albedo (eq. 5 in paper)
        # note 1: missing brackets around "a_ice_oerlemans - alpha_i_s" added.
        # note 2: a_ice_oerlemans replaced by a_background to account for firn.
        alpha_i_total = alpha_i_s + (a_background - alpha_i_s) * \
            np.exp(-snow_depth / d_star)

        # here comes the snowfall event threshold (eq. 6 in paper)
        sh_too_low = np.where(snow_depth < event_thresh)
        alpha_i_total[sh_too_low] = a_background[sh_too_low]

        return alpha_i_total

    def update_ensemble(self):
        """ Update the albedo using an ensemble of methods."""

        raise NotImplementedError


class Glacier(object, metaclass=SuperclassMeta):
    """
    Implements a glacier and what it needs:

    - geometry
    - mass balance
    - snow/firn cover
    -
    """


@xr.register_dataset_accessor('mb')
class MassBalance(object, metaclass=SuperclassMeta):
    """
    Basic interface for mass balance objects.

    The object has very limited support to units. If the dataset contains an
    attribute 'units' set to either 'm ice s-1', it is able to convert between
    ice flux (m ice s-1) and water equivalent (m w.e. d-1).
    """

    def __init__(self, xarray_obj):

        self._obj = xarray_obj

    @staticmethod
    def time_cumsum(x, skipna=True, keep_attrs=True):
        """Cumulative sum along time, skipping NaNs."""
        return x.cumsum(dim='time', skipna=skipna, keep_attrs=keep_attrs)

    @staticmethod
    def custom_quantiles(x, qs=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
        """Calculate quantiles with user input."""
        return x.quantile(qs, keep_attrs=True)

    def convert_to_meter_we(self):
        """
        If unit is ice flux (m ice s-1), convert it to meter water equivalent
        per day.

        Returns
        -------
        None
        """
        # todo: This is not flexible, but flexibility requires Pint dependency

        if self._obj.attrs['units'] == 'm ice s-1':
            for mbname, mbvar in self._obj.data_vars.items():
                if 'MB' in mbname:
                    self._obj[mbname] = mbvar * cfg.SEC_IN_DAY * cfg.RHO / \
                                        cfg.RHO_W
            self._obj.attrs['units'] = 'm w.e.'
        else:
            raise ValueError('Check the unit attribute, it should be "m ice '
                             's-1" to convert it to meters water equivalent '
                             'per day.')

    def convert_to_ice_flux(self):
        """
        If unit is meter water equivalent per day, convert it to ice flux (m
        ice s-1).

        Returns
        -------
        None
        """
        # todo: This is not flexible, but flexibility requires Pint dependency

        if self._obj.attrs['units'] == 'm w.e.':
            for mbname, mbvar in self._obj.data_vars.items():
                if 'MB' in mbname:
                    self._obj[mbname] = mbvar / cfg.SEC_IN_DAY * cfg.RHO_W / \
                                        cfg.RHO
            self._obj.attrs['units'] = 'm ice s-1'
        else:
            raise ValueError('Check the unit attribute, it should be "m w.e." '
                             'to ice flux.')

    def make_hydro_years(self, bg_month=10, bg_day=1):
        hydro_years = xr.DataArray(
            [t.year if ((t.month < bg_month) or
                        ((t.month == bg_month) &
                         (t.day < bg_day)))
             else (t.year + 1) for t in
             self._obj.indexes['time']],
            dims='time', name='hydro_years',
            coords={'time': self._obj.time})
        return hydro_years

    def make_hydro_doys(self, hydro_years=None, bg_month=10, bg_day=1):
        if hydro_years is None:
            hydro_years = self.make_hydro_years(bg_month=bg_month,
                                                bg_day=bg_day)
        _, cts = np.unique(hydro_years, return_counts=True)
        doys = [list(range(1, c + 1)) for c in cts]
        doys = [i for sub in doys for i in sub]
        hydro_doys = xr.DataArray(doys, dims='time', name='hydro_doys',
                                  coords={'time': hydro_years.time})
        return hydro_doys

    def append_to_gdir(self, gdir, basename, reset=False):
        """
        Write mass balance to a glacier directory.

        Choose to be aware of possibly existing files or to reset.

        Parameters
        ----------
        reset: bool
            If True, then an existing mass balance file is replaced. If False,
            only an existing variable (if present) is overwritten.

        Returns
        -------
        None
        """

        if reset is False:
            try:
                mb_ds_old = gdir.read_pickle(basename)
                to_write = self._obj.combine_first(mb_ds_old)
            except FileNotFoundError:
                to_write = self._obj.copy(deep=True)
        else:
            to_write = self._obj.copy(deep=True)

        gdir.write_pickle(to_write, basename)

    def create_specific(self, mb_model, from_date=None, to_date=None,
                        write=True):
        """
        Create the time series from a given mass balance model.

        Parameters
        ----------
        MassBalanceModel: crampon.core.models.massbalance.MassBalanceModel
            The model used to create the mass balance time series.
        from_date: datetime.datetime
            The first day of the time series.
        to_date: datetime.datetime
            The last day of the time series.
        write: bool
            Whether to write the result to the glacier directory.

        Returns
        -------
        ts: xr.Dataset
            The mass balance time series.
        """

        self.mb_model = MassBalanceModel

    def select_time(self):
        """
        Select a time slice from the mass balance.

        Returns
        -------

        """

    def extend_until(self, date, write=True):
        """
        Append one/more time steps at the end of the time series.

        Returns
        -------

        """

        last_day = pd.Timestamp(self._obj.time[-1])

        append = self.create_specific(self.mb_model, from_date=last_day,
                                      to_date=date, write=False)

        self._obj = xr.concat(self._obj, append)

        self.append_to_gdir(gdir, basename, reset=False)

    def make_cumsum_quantiles(self, bg_month=10, bg_day=1, quantiles=None):
        """
        Apply cumulative sum and then quantiles to the mass balance data.

        Returns
        -------
        None
        """
        save_attrs = self._obj.attrs  # needed later

        hyears = self.make_hydro_years(bg_month, bg_day)
        hdoys = self.make_hydro_doys(hyears, bg_month, bg_day)

        mb_cumsum = self._obj.groupby(hyears).apply(
            lambda x: self.time_cumsum(x))
        # todo: EMERGENCY SOLUTION as long as we are not able to calculate cumsum with NaNs correctly
        mb_cumsum = mb_cumsum.where(mb_cumsum.MB != 0.)
        if quantiles:
            quant = mb_cumsum.groupby(hdoys) \
                .apply(lambda x: self.custom_quantiles(x, qs=quantiles))
        else:
            quant = mb_cumsum.groupby(hdoys) \
                .apply(lambda x: MassBalance.custom_quantiles(x))

        # insert attributes again...they get lost when grouping!?
        quant.attrs.update(save_attrs)

        return quant

    def to_array(self):
        """
        Output the data as an array.

        If the data is a time series, the
        The first dimension of the array is defined

        Returns
        -------
        arr: np.array
            An array of
        """
        raise NotImplementedError

    def get_balance(self, date1, date2, which='total'):
        """
        Retrieves the balance in a time interval.

        Input must be the mass balance itself, not already a cumulative sum.

        Parameters
        ----------
        date1: datetime.datetime
            Begin of the time interval.
        date2: datetime.datetime
            End of the time interval.
        which: str
            Which balance to retrieve: allowed are 'total' (get the total mass
            balance), 'accumulation' (get accumulation only), 'ablation'
            (ablation only). Default: 'total'.

        Returns
        -------
        mb_sel: type of `self`
            The mass balance object, reduced to the variables that contain mass
            balance information and the type of balance requested.
        """

        mb_sel = self.obj.sel(time=slice(date1, date2))
        mb_sel = mb_sel[[i for i in mb_sel.data_vars.keys() if 'MB' in i]]

        if which == 'total':
            return mb_sel.sum(dim='time', skipna=True)
        elif which == 'accumulation':
            return mb_sel.where(mb_sel > 0.).sum(dim='time', skipna=True)
        elif which == 'ablation':
            return mb_sel.where(mb_sel < 0.).sum(dim='time', skipna=True)
        else:
            raise ValueError('Value {} for balance is not recognized.'.
                             format(which))


class PastMassBalance(MassBalance):
    """
    A class to handle mass balances
    """

    def __init__(self, gdir, mb_model, dataset=None):
        """

        Parameters
        ----------
        gdir
        mb_model
        dataset
        """
        super().__init__(gdir, mb_model, dataset=dataset)

        if dataset is None:
            try:
                self._obj = gdir.read_pickle('mb_daily')
            except:
                raise ValueError('Dataset kwarg must be supplied when gdir '
                                 'has no daily_mb.pkl.')


class CurrentYearMassBalance(MassBalance):

    def __init__(self, gdir, mb_model, dataset=None):

        super().__init__(gdir, mb_model, dataset=dataset)


def run_snowfirnmodel_with_options(gdir, run_start, run_end,
                                   mb=None,
                                   reclassify_heights=None,
                                   snow_densify='anderson',
                                   snow_densify_kwargs=None,
                                   firn_densify='huss',
                                   firn_densify_kwargs= None,
                                   temp_update='huss',
                                   temp_update_kwargs=None,
                                   merge_layers=0.05):
    """
    Run the SnowFirnCover using the given options.

    Parameters
    ----------
    gdir. crampon.GlacierDirectory
        The GlacierDirectory to caculate the densification for.
    run_start: datetime.datetime
        Start date for model run.
    run_end: datetime.datetime
        End date for model run.
    mb: xarray.Dataset
        # Todo: make this more felxible: one should also be able to let the MB calculate online
        The mass balance time series. Default: None (calculate the MB
        durign the model run).
    reclassify_heights: float or None
        Whether to reclassify the glacier flowline heights. This can
        significantly decrease the time to run the model. If a number is
        given, this number determines the elevation bin height used for
        reclassification. Default: None (no reclassification, i.e. the
        concatenated flowline surface heights are used for modeling).
    snow_densify: str
        Option which snow densification model to calibrate. Possible:
        'anderson' (uses the Anderson 1976 densification equation).
    snow_densify_kwargs: dict
        Keyword accepted by the chosen snow densification method.
    firn_densify: str
        Option which firn densification model to calibrate. Possible:
        'huss' (uses the Huss 2013/Reeh 2008 densification equation) and
        'barnola' (uses Barnola 1990).
    firn_densify_kwargs: dict
        Keywords accepted by the chosen firn densification method.
    temp_update: str
        Option which temperature update model to use. Possible: 'exact'
        (models the exact daily temperature penetration thourgh the
        snowpack - super slow!), 'huss' (uses the Huss 2013 temperature
        update), 'glogem' (Huss and Hock 2015) and 'carslaw' (uses Carslaw
        1959 - very slow!).
    temp_update_kwargs: dict
        Keywords accepted by the chosen temperature update method.
    merge_layers: float
        Minimum layer height (m) to which incoming layers shall be merged.

    Returns
    -------

    """


    run_time = pd.date_range(run_start, run_end)
    meteo = climate.GlacierMeteo(gdir)
    tmonmean = meteo.get_mean_month_temperature()

    if reclassify_heights:
        heights, widths = utils.reclassify_heights_widths(gdir,
                                                          elevation_bin=reclassify_heights)
        # just to make the plots look nicer
        heights = heights[::-1]
        widths = widths[::-1]
    else:
        heights, widths = gdir.get_inversion_flowline_hw()

    #if mb is None:
    day_model = BraithwaiteModel(gdir, bias=0.,
                                     snow_init=np.zeros_like(heights))

    init_swe = np.zeros_like(heights)
    init_swe.fill(np.nan)
    init_temp = init_swe
    cover = SnowFirnCover(heights, swe=init_swe, rho=None,
                          origin=run_time[0], temperatures=init_temp,
                          refreezing=False)

    for date in run_time:

        # Get the mass balance and convert to m w.e. per day
        #if mb is None:
        tmp = day_model.get_daily_mb(heights, date=date) * 3600 * 24 * \
                  cfg.RHO / cfg.RHO_W
        #else:
        #    tmp = mb.sel(time=date).MB.values

        swe = tmp.copy()
        rho = np.ones_like(tmp) * get_rho_fresh_snow_anderson(
            meteo.meteo.sel(time=date).temp.values + cfg.ZERO_DEG_KELVIN)
        temperature = swe.copy()
        if temp_update.lower() == 'exact':
            # Todo: place real temperature here
            temperature[~pd.isnull(swe)] = \
                meteo.get_tmean_at_heights(date, heights)[~pd.isnull(swe)]
        temperature[~pd.isnull(swe)] = cfg.ZERO_DEG_KELVIN

        cover.ingest_balance(swe, rho, date, temperature)

        # SNOW DENSIFICATION
        if snow_densify.lower() == 'anderson':
            cover.densify_snow_anderson(date, **snow_densify_kwargs)

        # TEMPERATURE UPDATE
        if temp_update.lower() == 'exact':
            # REGULAR TEMPERATURE UPDATE (SUPER SLOW)
            cover.update_temperature(date, **temp_update_kwargs)
            cover.apply_refreezing()
        elif temp_update.lower() == 'huss':
            if date.day == 30 and date.month == 4:
                cover.update_temperature_huss(**temp_update_kwargs)
                cover.apply_refreezing(exhaust=True)
        elif temp_update.lower() == 'glogem':
            raise NotImplementedError
            if (date.day % 3 == 0.) and date.month in [11, 12, 1, 2, 3]:
                meantemp = tmonmean.sel(time='{}-{}'.format(date.year,
                                                              date.month),
                                          method='nearest').temp.values
                cover.update_temperature_glogem(meantemp, 0.01, **temp_update_kwargs)
                cover.apply_refreezing()
        elif temp_update.lower() == 'carslaw':
            raise NotImplementedError
            if date.day == 30 and date.month == 4:
                duration = 120
                mean_winter_temp = meteo.sel(time=slice(date-dt.timedelta(days=120), date)).temp.mean()
                cover.update_temperature_carslaw(mean_winter_temp, duration, **temp_update_kwargs)
                # Todo: this should also be handed over
                cover.apply_refreezing(exhaust=True)
        else:
            raise ValueError('The chosen temperature update method {} '
                             'does not exist.'.format(temp_update))

        # FIRN DENSIFICATION
        if firn_densify.lower() == 'huss':
            if date.month == 10 and date.day == 1:
                #print('Densifying Firn')
                cover.densify_firn_huss(date, **firn_densify_kwargs)
                if date.year % 10 == 0:
                    print(date)
        elif firn_densify.lower() == 'barnola':
            if date.month == 10 and date.day == 1:
                cover.densify_firn_barnola(date, **firn_densify_kwargs)
        else:
            raise ValueError('The chosen firn densification method {} '
                             'does not exist.'.format(temp_update))

    return cover


if __name__ == '__main__':
    import geopandas as gpd
    from crampon import workflow
    import os
    from crampon import utils

    cfg.initialize(file='C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\'
                    'CH_params.cfg')

    PLOTS_DIR = os.path.join(cfg.PATHS['working_dir'], 'plots')

    # Currently OGGM wants some directories to exist
    # (maybe I'll change this but it can also catch errors in the user config)
    utils.mkdir(cfg.PATHS['working_dir'])

    glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
    rgidf = gpd.read_file(glaciers)
    rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])]  # Gries

    gdirs = workflow.init_glacier_regions(rgidf, reset=False, force=False)

    run_time = pd.date_range('1961-10-10', '2017-10-10')
    day_model = BraithwaiteModel(gdirs[0], bias=0.)
    heights, widths = gdirs[0].get_inversion_flowline_hw()
    init_swe = np.zeros_like(heights)
    init_swe.fill(np.nan)
    init_temperatures = init_swe
    cover = SnowFirnCover(heights, swe=init_swe,
                          rho=None,
                          origin=run_time[0],
                          temperatures=init_temperatures,#np.ones_like(heights)*273.16,
                          refreezing=False)

    # stop the time
    get_mb_time = []
    add_layer_time = []
    melt_time = []
    d_s_a_time = []
    merge_layers_time = []
    densify_firn_huss_time = []

    # number of experiments (list!)
    exp = [1]
    mb = []
    rho_snow_end = []

    temp_temperature = None
    for date in run_time:

        # TODO: this is just a test! remove!
        # densify snow from yesterday
        cover.densify_firn_barnola(date)

        # Get the mass balance and convert to m w.e. per day
        before = dt.datetime.now()
        tmp = day_model.get_daily_mb(heights, date=date) * 3600 * 24 * \
              cfg.RHO / 1000.
        mb.append(tmp)
        after = dt.datetime.now()
        get_mb_time.append(after - before)

        before = dt.datetime.now()
        if (tmp > 0.).any():
            swe = np.clip(tmp, 0, None)
            rho = np.ones_like(tmp) * 100.
            rho[swe == 0.] = np.nan
            cover.add_layer(swe=swe, rho=rho, origin=date)
        after = dt.datetime.now()
        add_layer_time.append(after - before)

        before = dt.datetime.now()
        if (tmp < 0.).any():
            cover.melt(np.clip(tmp, None, 0))
        after = dt.datetime.now()
        melt_time.append(after - before)

        before = dt.datetime.now()
        cover.densify_snow_anderson(date)
        after = dt.datetime.now()
        d_s_a_time.append(after - before)

        print(date)

        if date.day == 30 and date.month==9:
            print('hi')

        if date.day == 1:
            print('Merging', np.nansum(cover.swe > 0.))
            before = dt.datetime.now()
            #cover.merge_layers(min_sh=0.1)
            after = dt.datetime.now()
            merge_layers_time.append(after - before)
            print(np.nansum(cover.swe > 0.))

        if date.month == 10 and date.day == 1:
            #rho_snow_end.append([max([i.rho for i in j if dt.timedelta(date - i.origin).days < 365] for j in len(cover.grid))])
            print('Densifying Firn')
            before = dt.datetime.now()
            cover.densify_firn_huss(date)
            after = dt.datetime.now()
            densify_firn_huss_time.append(after - before)

    print('hi')
