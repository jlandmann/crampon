from __future__ import division

from oggm.core.massbalance import *
from crampon import cfg
from crampon.utils import SuperclassMeta, lazy_property, closest_date
from crampon import utils
import xarray as xr
import datetime as dt
import cython


class DailyMassBalanceModel(MassBalanceModel):
    """
    Extension of OGGM's MassBalanceModel, able to calculate daily mass balance.
    """

    def __init__(self, gdir, mu_star=None, bias=None, prcp_fac=None,
                 filename='climate_daily', filesuffix='',
                 param_ep_func=np.nanmean):

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
        param_ep_func: numpy arithmetic function
            Method to use for extrapolation when there are no calibrated
            parameters available for the time step where the mass balance
            should be calcul
        """
        # should probably be replaced by a direct access to a file that
        # contains uncertainties (don't know how to handle that yet)
        if mu_star is None:
            cali_df = pd.read_csv(gdir.get_filepath('calibration'),
                                  index_col=0,
                                  parse_dates=[0])
            try:
                mu_star = cali_df['OGGM_mu_star']
            except KeyError:
                mu_star = cali_df['mu_star']
        if bias is None:
            cali_df = pd.read_csv(gdir.get_filepath('calibration'),
                                  index_col=0,
                                  parse_dates=[0])
            if cfg.PARAMS['use_bias_for_run']:
                bias = cali_df['bias']
            else:
                bias = pd.DataFrame(index=cali_df.index,
                                    data=np.zeros_like(cali_df.index,
                                                       dtype=float))
        if prcp_fac is None:
            cali_df = pd.read_csv(gdir.get_filepath('calibration'),
                                  index_col=0,
                                  parse_dates=[0])
            try:
                prcp_fac = cali_df['OGGM_prcp_fac']
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
        self.heights, self.widths = gdir.get_inversion_flowline_hw()

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
            self.span_meteo = time

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

        # Public attrs
        self.temp_bias = 0.

        self._time_elapsed = None

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
        ix = pd.DatetimeIndex(self.span_meteo).get_loc(date, **kwargs)

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
        mbs = self.get_daily_mb(heights, date=date, **kwargs) * cfg.SEC_IN_DAY * cfg.RHO /\
              1000.
        mbs_wavg = np.average(mbs, weights=widths)
        return mbs_wavg

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

            mb_ds = xr.Dataset({'MB': (['time', 'n'],
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


class BraithwaiteModel(DailyMassBalanceModel):

    def __init__(self, gdir, mu_ice=None, mu_snow=None, bias=None,
                 prcp_fac=None, snow_init=None, filename='climate_daily',
                 filesuffix=''):

        super().__init__(gdir, mu_star=mu_ice, bias=bias, prcp_fac=prcp_fac,
                         filename=filename, filesuffix=filesuffix)

        if mu_ice is None:
            cali_df = pd.read_csv(gdir.get_filepath('calibration'),
                                  index_col=0,
                                  parse_dates=[0])
            mu_ice = cali_df['mu_ice']
        if mu_snow is None:
            cali_df = pd.read_csv(gdir.get_filepath('calibration'),
                                  index_col=0,
                                  parse_dates=[0])
            mu_snow = cali_df['mu_snow']
        if bias is None:
            cali_df = pd.read_csv(gdir.get_filepath('calibration'),
                                  index_col=0,
                                  parse_dates=[0])
            if cfg.PARAMS['use_bias_for_run']:
                bias = cali_df['bias']
            else:
                bias = pd.DataFrame(index=cali_df.index,
                                    data=np.zeros_like(cali_df.index,
                                                       dtype=float))
        if prcp_fac is None:
            cali_df = pd.read_csv(gdir.get_filepath('calibration'),
                                  index_col=0,
                                  parse_dates=[0])
            try:
                prcp_fac = cali_df['OGGM_prcp_fac']
            except KeyError:
                prcp_fac = cali_df['prcp_fac']

        self.mu_ice = mu_ice
        self.mu_snow = mu_snow
        self.bias = bias
        self.prcp_fac = prcp_fac

        if snow_init is None:
            self.snow_init = np.atleast_2d(np.zeros_like(self.heights))
            self.snow = np.atleast_2d(np.zeros_like(self.heights))
        else:
            self.snow_init = np.atleast_2d(snow_init)
            self.snow = np.atleast_2d(snow_init)

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
            # self.years = np.unique([y.year for y in self.nc_time])

            # Read timeseries
            self.temp = nc.variables['temp'][:]
            # Todo: Think of nicer extrapolation method than fill with mean
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
        ix = pd.DatetimeIndex(self.tspan_meteo).get_loc(date, **kwargs)

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
        snowdist = np.where(self.snow[-1] > 0.)
        mu_comb = np.zeros_like(self.snow[-1])
        mu_comb[:] = mu_ice
        np.put(mu_comb, snowdist, mu_snow)

        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - mu_comb * tempformelt - bias

        self.time_elapsed = date
        snow_cond = self.update_snow(date, mb_day)

        # cheap removal of snow before density model kicks in
        try:
            # check MB pos
            inds = np.where((self.snow[-1] - self.snow[-366]) >= 0.)
            self.snow[-1][inds] = np.clip(self.snow[-1][inds] -
                                          np.clip(self.snow[-365][inds] -
                                                  self.snow[-366][inds], 0.,
                                                  None), 0., None)
        except IndexError: # when date not yet exists
            pass

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


class HockModel(DailyMassBalanceModel):
    """ This class implements the melt model by Hock (1999)."""

    def __init__(self):
        raise NotImplementedError


class PellicciottiModel(DailyMassBalanceModel):
    """ This class implements the melt model by Pellicciotti et al. (2005).

    Attributes
    ----------
    albedo: py:class:`crampon.core.models.massbalance.Albedo`
        An albedo object managing the albedo ageing.
    """

    def __init__(self):

        super.__init__()
        self.albedo = GlacierAlbedo() #TODO: implement Albedo object

    def get_alpha(self, snow, date):
        """
        Calculates the albedo for the snow on the glacier.

        The approach could be after Brock et al. 2000 with values of
        Pellicciotti et al. 2005 (see Gabbi et al. 2014). However, the sum of
        maximum temperatures since the last snowfall would be needed (we only
        have the daily mean so far).


        Parameters
        ----------
        snow: pd.DataFrame()
            The glacier's snow characteristics as defined in the ``snow``
            attribute.
        date: datetime.datetime
            Date for which the albedo should be calculated.

        Returns
        -------
        ndarray
            An array with the albedo.
        """
        return NotImplementedError

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
        ix = pd.DatetimeIndex(self.tspan_meteo).get_loc(date, **kwargs)

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
        prcpsol, _ = self.get_prcp_sol_liq(iprcp, heights, temp)

        # TODO: What happens when `date` is out of range of the cali df index?
        # THEN should it take the mean or raise an error?
        if isinstance(self.tf, pd.Series):
            try:
                mu_ice = self.tf.iloc[
                    self.tf.index.get_loc(date, **kwargs)]
            except KeyError:
                mu_ice = self.param_ep_func(self.tf)
            if pd.isnull(mu_ice):
                mu_ice = self.param_ep_func(self.tf)
        else:
            mu_ice = self.tf

        if isinstance(self.srf, pd.Series):
            try:
                mu_snow = self.srf.iloc[
                    self.srf.index.get_loc(date, **kwargs)]
            except KeyError:
                mu_snow = self.param_ep_func(self.srf)
            if pd.isnull(mu_snow):
                mu_snow = self.param_ep_func(self.srf)
        else:
            mu_snow = self.srf

        if isinstance(self.bias, pd.Series):
            bias = self.bias.iloc[
                self.bias.index.get_loc(date, **kwargs)]
            # TODO: Think of clever solution when no bias (=0)?
        else:
            bias = self.bias

        # Get snow distribution from yesterday and determine snow/ice from it;
        # this makes more sense as temp is from 0am-0am and precip from 6am-6am
        snowdist = np.where(self.snow[-1] > 0.)
        mu_comb = np.zeros_like(self.snow[-1])
        mu_comb[:] = mu_ice
        np.put(mu_comb, snowdist, mu_snow)

        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - mu_comb * tempformelt - bias

        mb_day = self.t * T + self.srf * (1 - self.albedo.get_alpha()) * G

        self.time_elapsed = date
        snow_cond = self.update_snow(date, mb_day)

        # cheap removal of snow before density model kicks in
        try:
            # check MB pos
            inds = np.where((self.snow[-1] - self.snow[-366]) >= 0.)
            self.snow[-1][inds] = np.clip(self.snow[-1][inds] -
                                          np.clip(self.snow[-365][inds] -
                                                  self.snow[-366][inds], 0.,
                                                  None), 0., None)
        except IndexError:  # when date not yet exists
            pass

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / cfg.SEC_IN_DAY / cfg.RHO
        return icerate


class SnowFirnCoverArrays(object):
    """ Implements a an interface to a snow and/or firn cover with property arrays.

    # TODO: Take care of removing ice layers instantly, so that e.g. the method "get:total_rho" doesn't deliver wrong results

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
                 pore_close=None, firn_ice_transition=None, refreezing=True):
        """
        Instantiate a snow and/or firn cover and its necessary methods.

        If no temperatures are given, the temperature profile is assumed to be
        homogeneous.

        Parameters
        ----------
        height_nodes: array-like
            The heights at which the snow/firn cover should be implemented.
        swe: np.array, same shape as height_nodes
            Snow water equivalent of an initial layer (m w.e.).
        rho: np.array, same shape of height_nodes or None
            Density of an initial layer (kg m-3). If None, the inital density
            is set to NaN. # Todo: make condition for rho=None: if swe >0, then determine initial density from temperature
        origin: datetime.datetime or pd.Timestamp
            Origin date of the initial layer.
        temperatures: np.array, same shape of height_nodes, optional
            Temperatures of initial layer (K). If not given, they will be set
            to NaN.
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
        self.init_origin = [origin] * self.n_heights
        init_liq_content = np.zeros_like(height_nodes)
        init_liq_content.fill(np.nan)
        self.init_liq_content = init_liq_content

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
        self._tgrid_nodes = [np.array([0., self.init_sh[i]]) for i in
                             range(len(self.init_sh))]
        if temperatures is not None:
            self.init_temperature = temperatures
            self._tgrid_temperature = temperatures
        else:
            # initiate with zero deg C, if we don't know better
            self.init_temperature = np.ones(height_nodes) * 273.16
            self._tgrid_temperature = np.ones(height_nodes) * 273.16

        init_array = np.zeros((self.n_heights, np.atleast_2d(self.init_swe).shape[1] + 1))
        init_array.fill(np.nan)

        # we start putting the initial layer at index 0 (top of array!)
        self._swe = np.hstack((np.atleast_2d(self.init_swe).T, init_array))
        self._rho = np.hstack((np.atleast_2d(self.init_rho).T, init_array))
        self._origin = np.hstack((np.atleast_2d(
            np.array([origin for i in range(self.n_heights)])).T, init_array))
        self._last_update = self._origin.copy()
        self._temperature = np.hstack((
            np.atleast_2d(self.init_temperature).T, init_array))
        self._liq_content = np.hstack(
            (np.ones_like(np.atleast_2d(self.init_liq_content).T), init_array))
        strrow = np.atleast_2d(swe).T.astype(str)
        strrow.fill('')
        self._status = np.hstack((strrow, init_array))

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
        # if no or positive temperature is given, assume zero degrees Celsius
        self._temperature = value
        #self._temperature[~pd.isnull(value) & (value <= 273.16)] = value[~pd.isnull(value) & (value <= 273.16)]

    @property
    def liq_content(self):
        return self._liq_content

    @liq_content.setter
    def liq_content(self, value):
        if value is not None:
            if value > self.swe:
                raise ValueError('Liquid water content of a snow layer cannot '
                                 'be bigger than the snow water equivalent.')
            self._liq_content = value
        else:
            self._liq_content = 0.

    @property
    def origin(self):
        return self._origin

    @property
    def last_update(self):
        return self._last_update

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
                (self.temperature - 273.15)
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
        self._status[self.rho < cfg.PARAMS['snow_firn_threshold']] = 'snow'
        self._status[(cfg.PARAMS['snow_firn_threshold'] <= self.rho) & (
                self.rho < cfg.PARAMS['pore_closeoff'])] = 'firn'
        self._status[(cfg.PARAMS[
                          'pore_closeoff'] <= self.rho) & (
                                 self.rho < cfg.RHO)] = 'pore_closeoff'
        self._status[self.rho >= cfg.RHO] = 'ice'

        return self._status

    @property
    def n_heights(self):
        return len(self.height_nodes)

    @property
    def top_layer(self):
        """
        A pointer to the top layer in the SnowFirnPack.

        Initially, when there are no layers in the pack, the top index will be
        -1!

        Returns
        -------
        top: np.ndarray
            Array giving the indices of the current top layer.
        """
        # TODO: Maybe check if the layers are consistent for all properties!?
        layers_bool = np.logical_or(np.isin(self.swe, [0.]),
                                    np.isnan(self.swe))
        top = np.argmax(layers_bool, axis=1) - 1
        return top

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

        # TODO. the rule to determine this is doubled with self.status...but calling self.status here might be expensive!?
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
                  ix=None):
        """
        Add a layer to the snow/firn pack.

        If no density is given, the fresh snow density will be calculated
        after Anderson (1973).

        Parameters
        ----------
        ix: int
            Where to add the layer. Default: None (top).
        """

        if len(swe) != self.n_heights:
            raise ValueError('Dimensions of SnowFirnCover and mass to be '
                             'added must match.')

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
            new_status = bigger_array.copy().astype(object)

            #status first, because it depends on the others
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


        if temperature is not None:
            self.temperature[
                range(self.swe.shape[0]), insert_pos] = temperature
        else:
            self.temperature[
                range(self.swe.shape[0]), insert_pos] = 273.16
        if liq_content is not None:
            self.liq_content[
                range(self.swe.shape[0]), insert_pos] = liq_content
        else:
            self.liq_content[range(self.swe.shape[0]), insert_pos] = 0.

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
            remove_ix = self.top_layer

        self.swe[remove_ix] = np.nan
        self.rho[remove_ix] = np.nan
        self.origin[remove_ix] = np.nan
        self.temperature[remove_ix] = np.nan
        self.liq_content[remove_ix] = np.nan
        self.status[remove_ix] = 'nan'
        self.last_update[remove_ix] = np.nan

        self.remove_unnecessary_array_space()

        # if layer to be removed are at the bottom, justify the arrays
        if 0 in remove_ix[1]:
            # This adjusts the position of the layers within the array
            self.swe = utils.justify(self.swe, invalid_val=np.nan, side='left')
            self.rho = utils.justify(self.rho, invalid_val=np.nan, side='left')
            self.temperature = utils.justify(self.temperature,
                                             invalid_val=np.nan,
                                             side='left')
            # invalid_val=None ensures a good handlung of dt objects etc.
            self._liq_content = utils.justify(self.liq_content,
                                              invalid_val=None,
                                              side='left')
            # soem more hassle with the dates
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
        self.status[remove] = ''
        self.origin[remove] = np.nan
        self.last_update[remove] = np.nan

        mask = np.ma.array(old_swe, mask=np.invert(
            (cum_swe <= swe) & (cum_swe != 0) & ~np.isnan(cum_swe)))
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


    def get_lost_ice_volume(self):
        """
        Get the amount of accumulated ice melt that happened when there was no
        firn/snow present.

        Together with a second value from a later point in time
        Returns
        -------

        """
        # TODO: we need a variable where accumulated ice loss is stored . By then subtracting the value between two dates we can get the dV for ice

    def get_snow_height(self):
        """Get height of snow cover only."""
        raise NotImplementedError

    def get_firn_height(self):
        """Get height of firn cover only."""
        raise NotImplementedError

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
        return np.average(self.rho, axis=1, weights=self.swe)

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

        ice_ix = self.get_type_indices('ice')
        self.remove_layer(ice_ix)

    def update_temperature(self, date, max_depth=15., deltat=86400, lower_bound=273.16):
        """
        Update the temperature profile of the snow/firn pack.

        The temperature is calculated on an equidistant grid until the maximum
        depth given to insure numeric stability. Temperatures are then mapped
        in a weighted fashion to the given layers.

        Parameters
        ----------
        max_depth: float
            Maximum depth in meters until which the temperature profile is
            calculated. Default: 15.
        dx: float
            Grid spacing for the homogeneous temperature grid. Default: 0.1m
        lower_bound: float
            Lower boundary condition temperature. Default: 273.16 K (temperate
            ice).

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
        temp = slope * depth + surface_temp + 273.15
        temp = np.clip(temp, None, 273.15)
        # temperature is now the layer temperature at the center depth
        self.temperature = temp

        else:
            self._tgrid_nodes = np.arange(0., max_depth, dx)
        # temperature boundary condition is zero: temperate ice)
        if max(self._tgrid_nodes) == max_depth:
            self._tgrid_temperature[-1] = lower_bound
        else:
            self._tgrid_temperature[-1] = self._tgrid_temperature[-2]
        # Calculate forward the existing temperatures on this grid
        # map the calculated temperatures into the layer structure

        # Update the temperatures of the layers inplace.

    def update_refreezing_potential(self, max_depth=15.):
        """
        Update refreezing potential of each layer inplace.

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
        #for sfp in self.grid:
        #    sfp.densify_firn_huss(date, f_firn=f_firn,
        #                          poresclosed_rate=poresclosed_rate,
        #                          rho_f0_const=rho_f0_const)

        # Todo: remove comment out
        #self.merge_firn_layers(date)

        # Mean annual surface temperature, 'presumably' firn temperature at 10m
        # depth (Reeh (2008)); set to 273 according to Huss 2013
        # TODO: this should be actually calculated
        T_ms = 273.  # K

        # some model constant
        k1 = f_firn * 575. * np.exp(
            - cfg.E_FIRN / (cfg.R * T_ms))  # m(-0.5)a(-0.5)

        # what happens between 800 and pore close-off? We just replace it.
        firn_ix = self.get_type_indices('firn')
        poresclosed_ix = self.get_type_indices('poresclosed')


        # careful with days!!!! (the difference might be 364 days or so)

        # TODO: ACTUALLY WE SHOULD DERIVE THE EQUATION AND THEN GO FROM TIME STEP TO TIME STEP
        #t = (date - self.origin[firn_ix]).days / 365.25
        date_arr = np.zeros_like(self.origin)
        date_arr[~pd.isnull(self.origin)] = date
        t = np.fromiter((d.days for d in (date - self.origin[firn_ix])), dtype=float, count=len(date - self.origin[firn_ix])).reshape(self.origin[firn_ix].shape) / 365.25

        if rho_f0_const:
            # initial firn density from Huss (2013) is 490, anyway...
            rho_f0 = cfg.PARAMS['snow_firn_threshold']  # kg m-3
        else:
            # todo:change to real init density again!!!!!
            #rho_f0 = self.init_density[firn_ix]
            rho_f0 = 550.

        # just to not divide by zero when a layer forms

            # "b" is in m w.e. a-1 as in Huss (2013). In Reeh(2008): m ice a-1
            b = np.divide(self.get_overburden_swe(), t, out=np.zeros_like(self.get_overburden_swe()), where=t!= 0)[firn_ix]
            #b = self.get_overburden_swe(firn_ix) / t  # annual accumulation rate


            c_reeh_ice = k1 * np.sqrt(
                b * cfg.RHO / cfg.RHO_W)  # 550 kg m-3 < rho_f < 800 kg m-3

            # Huss 2013, equation 5
            rho_f = cfg.RHO - (cfg.RHO - rho_f0) * np.exp(
                -c_reeh_ice * t)

            self.rho[firn_ix] = rho_f

        # TODO: apply refreezing here?
        if self.refreezing:
            self.update_refreezing_potential()

            # rho_f += RF_t

        # TODO: HERE WE ASSUME ONE YEAR => t NEEDS TO BE ADJUSTED OR AN ATTRIBUTE "LAST_UPDATED" NEEDS TO BE MADE
        #for h, l in poresclosed_ix:
        pc_age = np.fromiter((d.days for d in (date - self.origin[poresclosed_ix])),
                    dtype=float,
                    count=len(date - self.origin[poresclosed_ix])) / 365.25
        self.rho[poresclosed_ix] = self.rho[poresclosed_ix] + poresclosed_rate * pc_age

        # last but not least
        self.remove_ice_layers()

    def _densify_poresclosed_huss(self, rate):
        """ """
        # TODO: Implement this function to be called within firn densification functions
        pass

    def densify_huss_derivative(self, date, f_firn=2.4):
        """ Try and implement Reeh"""

        self.merge_firn_layers(date)

        firn_ix = self.get_type_indices('firn')
        poresclosed_ix = self.get_type_indices('poresclosed')
        snow_ix = self.get_type_indices('snow')

        for h, l in firn_ix:
            # TODO: Is the Spec_bal really only Snowlayers?
            snow_ix_h = [(hs, ls) for (hs, ls) in snow_ix if hs == h]


            # SPEC_BAL is in M ICE!!!!
            # is the specific balance meant as the annual balance or all overburden pressure?
            #spec_bal = np.sum([self.grid[hs][ls].swe for (hs, ls) in snow_ix_h]) * cfg.RHO_W / cfg.RHO

            # This assumes that the spec bal is all overburden pressure
            spec_bal = self.get_overburden_swe(h, l) * cfg.RHO_W / cfg.RHO

            # TODO: this should be actually calculated
            T_ms = 273.  # K

            # some model constant
            k1 = f_firn * 575. * np.exp(
                - cfg.E_FIRN / (cfg.R * T_ms))  # m(-0.5)a(-0.5)

            # TODO: ACTUALLY WE SHOULD DERIVE THE EQUATION AND THEN GO FROM TIME STEP TO TIME STEP
            t = (date - self.grid[h][l].origin).days / 365.25

            c_reeh_ice = k1 * np.sqrt(
                spec_bal * cfg.RHO / cfg.RHO_W)  # 550 < rho_f < 800 (kg m-3)

            if spec_bal > 0.:

                current_rho = self.grid[h][l].rho

                drho = c_reeh_ice * (cfg.RHO - self.grid[h][l].rho)

                self.grid[h][l].rho += drho

            # do not forget
            self.grid[h][l].last_update = date

        # last but not least
        self.remove_ice_layers()

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

        self.merge_firn_layers(date)

        firn_ix = self.get_type_indices('firn')
        poresclosed_ix = self.get_type_indices('poresclosed')

        a0 = 25400
        for h, l in firn_ix:
            k1 = a0 * np.exp(
                -60000 / (cfg.R * self.grid[h][l].temperature))

            print('before', self.grid[h][l].rho)

            # TODO: dt is in seconds! which unit is rho here=?
            days = (date - self.grid[h][l].last_update).days

            for dt in (np.ones(days) * 24 * 3600):

                current_rho = self.grid[h][l].rho
                si_ratio = current_rho / cfg.RHO

                if current_rho < 800.:
                    f = 10 ** ((beta * si_ratio ** 3) +
                               (gamma * si_ratio ** 2) +
                               (delta * si_ratio) +
                               epsilon)
                else:
                    f = ((3. / 16.) * (1 - si_ratio)) / \
                        (1. - (1. - si_ratio) ** 0.3) ** 3.

                p_over = self.get_overburden_swe(h, l) * 1000. * cfg.G

                try:
                    p_bubble = self.grid[h][l].bubble_pressure  # if pores closed
                except AttributeError:
                    p_bubble = 0.
                drho = k1 * current_rho * f * ((p_over - p_bubble)/10**6) ** 3 * dt

                self.grid[h][l].rho += drho

            print('after', self.grid[h][l].rho)
            if h == 400:
                print('hi')

            # do not forget
            self.grid[h][l].last_update = date

        # last but not least
        self.remove_ice_layers()

    def densify_snow_anderson(self, date, eta0=3.7e7, etaa=0.081, etab=0.018,
                              snda=2.8e-6, sndb=0.042, sndc=0.046, rhoc=150.,
                              rhof=100., target_dt=24*3600):
        """
        Snow densification according to Anderson (1976).

        The values for the parameters are taken from the SFM2 model (Essery,
        2015). The biggest problem might be that they are not tested on
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

        Tm_k = cfg.PARAMS['temp_melt'] + 273.16

        # Todo: CHECK HERE IF DATE IS MORE THAN ONE DAY AWAY FROM LAST UPDATE?
        deltat = target_dt

        rho_old = self.rho
        temperature = self.temperature

        # create cumsum, then subtract top layer swe, clip and convert to mass
        ovb_mass = self.get_overburden_mass()

        rho_new = rho_old.copy()
        insert_ix = (self.swe > 0.) & (self.status == 'snow') & \
                    (~np.isnan(self.swe))
        rho_new[insert_ix] = (rho_old + \
                       (
                               rho_old * cfg.G * ovb_mass * deltat / eta0) * \
                       np.exp(etaa * (temperature - Tm_k) - etab *
                              rho_old) + deltat * \
                       rho_old * snda * np.exp(
            sndb * (temperature - Tm_k) - sndc * np.clip(
                rho_old - rhoc, 0., None)))[insert_ix]
        self.rho = rho_new
        self.last_update[insert_ix] = date

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

        for h, pack in enumerate(self.grid):
            temp_inds = []
            temp_layers = []
            for l, layer in enumerate(pack):
                # merge only by density and not by date/age
                # don't use isinstance, this also returns true for SnowLayer:
                if (type(self.grid[h][l]) == SnowLayer) and \
                        self.grid[h][l].rho >= \
                        cfg.PARAMS['snow_firn_threshold']:
                    temp_inds.append(l)
                    temp_layers.append(self.grid[h][l])

            if h == 0:
                print(h)
                print(h)
            if temp_layers:
                temp_inds.sort(reverse=True)
                for i in temp_inds:
                    self.grid[h].pop(i)

                new_swe = sum([t.swe for t in temp_layers])
                new_rho = sum([t.rho * t.sh for t in temp_layers]) / np.sum(
                    [t.sh for t in temp_layers])
                # give them current date (daily model should have run until
                # then already)
                new_origin = date
                # new_origin = np.max([t.last_update for t in temp_layers])
                # TODO: the temperature should actually go via the energy
                new_temp = sum(
                    [t.temperature * t.swe for t in temp_layers]) / np.sum(
                    [t.swe for t in temp_layers])
                new_liq = sum([t.liq_content for t in temp_layers])
                # TODO: Here needs to a snowlayer, otherwise we get too many firn layers.....one should take care that here the factory can be used, but only the firnlayers < 1 year are merged
                insert_l = FirnLayer(swe=new_swe, rho=new_rho, origin=new_origin,
                                   temperature=new_temp, liq_content=new_liq)
                insert_l.init_density = new_rho

                self.grid[h].insert(min(temp_inds), insert_l)

        # HOW TO ENSURE THAT THE LAYERS ARE NOT MERGED WITH OTHER FIRN LAYERS?
        # INTRODUCE STATUS ATTRIBUTE?

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

        to_merge = np.where(self.sh < min_sh)


        for h in range(len(self.grid)):
            for l in range(len(self.grid[h]) - 1):

                # if we've become too short already
                try:
                    self.grid[h][l+1]
                except IndexError:
                    continue

                if self.grid[h][l].sh < min_sh:
                    # update l
                    layer_one = self.grid[h][l]
                    layer_two = self.grid[h][l + 1]
                    new_swe = np.sum([layer_one.swe + layer_two.swe])
                    new_rho = np.sum([t.rho * t.sh for t in
                                      [layer_one, layer_two]]) / np.sum(
                        [t.sh for t in [layer_one, layer_two]])
                    new_origin = np.max(
                        [t.last_update for t in [layer_one, layer_two]])
                    # TODO: the temperature should actually go via the energy
                    new_temperature = np.sum(
                        [t.temperature * t.swe for t in
                         [layer_one, layer_two]]) / np.sum(
                        [t.swe for t in [layer_one, layer_two]])
                    new_liq_content = np.sum(
                        [t.liq_content for t in [layer_one, layer_two]])
                    insert = SnowLayer(swe=new_swe, rho=new_rho,
                                       origin=new_origin,
                                       temperature=new_temperature,
                                       liq_content=new_liq_content)

                    self.grid[h][l] = insert

                    # remove l+1
                    self.remove_layer(h, l + 1)

    def return_state(self, param='swe', dataset=False):
        """This should be a function that can be called to get the current
        status of the snow/firn cover as numpy arrays or xr.Dataset. If put in
        a loop, the history can be retrieved."""

        if dataset:
            raise NotImplementedError
        else:
            # recipe to convert list of lists to array
            length = len(sorted(self.grid, key=len, reverse=True)[0])
            grid_array = np.array(
                [[np.nan] * (length - len(xi)) + [getattr(i, param) for i in
                                                  xi] for xi in self.grid])

            return grid_array


def get_rho_fresh_snow_anderson(tair, rho_min=50., df=1.7, ef=15.):
    """
    Get fresh snow density after Anderson (1976).

    Parameters
    ----------
    tair: np.array
        Air temperature during snowfall (K).
    rho_min: float
        Minimum density allowed (kg m-3). Default: 50
        (Oleson et al. (2004)).
    df: float
        Parameter according to Anderson (1976) (K). Default: 1.7
    ef: float
        Parameter according to Anderson (1976) (K). Default: 15.

    Returns
    -------
    rho_fresh: np.array
        Density of fresh snow.
    """

    # TODO: Use equation 17 from Essery (2013)? Probably no, because we already integrate over one day
    rho_fresh = rho_min + np.clip(
        df * (tair - cfg.PARAMS['temp_melt'] + ef) ** 1.5, None, 0.)

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


def get_rho_dv():
    """
    An out wrapper around mass balance and firn model that can calculate the
    density of volume change.

    Returns
    -------

    """
    # get the volume for each point on the flowline.

    raise NotImplementedError


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

    def __init__(self, surface, alpha=None, snow_ix=None, firn_ix=None,
                 ice_ix=None, standard_method_snow='Brock',
                 standard_method_firn=None, standard_method_ice=None,
                 a_snow_start=0.9, a_firn_start=0.5):
        """
        Instantiate a snow and/or firn cover and its necessary methods.

        Parameters
        ----------
        x: array-like
            Points at which the albedo should be updated.
        """
        self.x = x
        self.surface = surface
        self.a_snow_start = a_snow_start
        self.a_firn_start = a_firn_start

        if alpha is None:
            self.x[snow_ix] = self.a_snow_start

    def update_brock(self, p1=0.86, p2=0.155, Ta=None):
        """Update the snow albedo using the Brock et al. (2000) equation.

        Parameters
        ----------
        p1: float, optional
           Empirical coefficient, described by the albedo of fresh snow (for
           Ta=1 deg C). Default: (see Pellicciotti et al., 2005 and Gabbi et
           al, 2014).
        p2: float, optional
           Empirical coefficient. Default: (see Pellicciotti et al., 2005 and
           Gabbi et al., 2014)
        Ta: array-like
            Accumulated daily maximum temperature > 0 deg C since snowfall.
            Default: None (calculate).
        """

        Tacc = self.get_accumulated_temperature()

        a = p1 - p2 * np.log10(Tacc)
        raise NotImplementedError

    def get_accumulated_temperature(self):
        """
        Calculate accumulated daily max temperature since last snowfall.
        """

    def update_oerlemans(self, date, a_snow_oerlemans=0.75,
                         a_firn_oerlemans=0.53, t_star=21.9):
        """Update the snow albedo using the Oerlemans & Knap (1998) equation.

        Parameters
        ----------
        date: datetime.datetime
            Date for which to calculate the albedo.
        t_star: float, optional
            Typical time scale determining how fast a fresh snowfall reaches
            the firn albedo. Default: 21.9 (See Oerlemans & Knap (1998)).
        """

        age_days = (self.snow.age - date).days
        a = a_firn_oerlemans + (a_snow_oerlemans - a_firn_oerlemans) + \
            np.exp((age_days) / t_star)
        raise NotImplementedError

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


def _custom_cumsum(x):
    """Cumulative sum along time, skipping NaNs."""
    return x.cumsum(dim='time', skipna=True)


def _custom_quantiles(x, qs=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
    """Calculate quantiles with user input."""
    return x.quantile(qs)


@xr.register_dataset_accessor('mb')
class MassBalance(object, metaclass=SuperclassMeta):
    """
    Basic interface for mass balance objects.
    """
    def __init__(self, xarray_obj, gdir, mb_model):
        """
        Instantiate the MassBalance base class.

        Parameters
        ----------
        gdir: `py:class:crampon.GlacierDirectory`
        mb_model: `py:class:crampon.core.models.massbalance.MassBalanceModel`
            The mass balance model used to calculate the time series.
        dataset: xr.Dataset

        """

        self._obj = xarray_obj
        self.gdir = gdir
        self.mb_model = mb_model
        self.cali_pool = None # TODO: Calibration(); for this I need to write
        # a calibration object and implement cli parameters attributes in the
        # MB_model classes

    @staticmethod
    def _custom_cumsum(x):
        """Cumulative sum along time, skipping NaNs."""
        return x.cumsum(dim='time', skipna=True)

    @staticmethod
    def _custom_quantiles(x, qs=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
        """Calculate quantiles with user input."""
        return x.quantile(qs)

    def apply_cumsum(self):
        """
        Apply cumsum to mass balance data.

        Returns
        -------
        cumsum: xr.Dataset
            The mass balance dataset as cumulative sum.
        """

    def create_specific(self, MassBalanceModel, from_date=None, to_date=None,
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

        if write:
            self.gdir.write_pickle(self._obj, 'mb_daily')
            # TODO: write a write_snow and write_mb method to be called here

    def make_quantiles(self):
        """
        Apply quantiles to the mass balance data.

        Returns
        -------

        """

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

    def get_balance(self, date1, date2, which='total'):
        """
        Retrieves the balance in a time interval.

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

        """

        if which == 'total':
            raise NotImplementedError
        elif which == 'accumulation':
            raise NotImplementedError
        elif which == 'ablation':
            raise NotImplementedError
        else:
            raise ValueError('Value {} for balance is not recognized.'.
                             format(which))


class PastMassBalance(MassBalance):
    """
    A class to handle mass baöan
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
    cover = SnowFirnCoverArrays(heights, swe=init_swe,
                          rho=np.ones_like(heights)*100.,
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
