from __future__ import division

from oggm.core.massbalance import *
from crampon import cfg
from crampon.utils import SuperclassMeta, lazy_property, closest_date
import xarray as xr
import datetime as dt


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
            paramaters available for the time step where the mass balance
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
            self.tgrad = nc.variables['grad'][:]
            self.pgrad = cfg.PARAMS['prcp_grad']
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

    def get_prcp_sol_liq(self, iprcp, heights, temp):
        # Compute solid precipitation from total precipitation
        # the prec correction with the gradient does not exist in OGGM
        npix = len(heights)
        prcptot = np.ones(npix) * iprcp + self.pgrad * iprcp * (heights -
                                                                self.ref_hgt)
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

        # For each height pixel:
        # Compute temp tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + itgrad * (heights - self.ref_hgt)

        tempformelt = self.get_tempformelt(temp)
        prcpsol, _ = self.get_prcp_sol_liq(iprcp, heights, temp)

        # TODO: What happens when `date` is out of range of the cali df index?
        # THEN should it take the mean or raise an error?
        if isinstance(self.mu_star, pd.Series):
            try:
                mu_star = self.mu_star.iloc[self.mu_star.index.get_loc(
                    date, **kwargs)]
            except KeyError:
                mu_star = self.param_ep_func(self.mu_star)
            if pd.isnull(mu_star):
                mu_ice = self.param_ep_func(self.mu_star)
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
            self.tgrad = nc.variables['grad'][:]
            self.pgrad = cfg.PARAMS['prcp_grad']
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
        self.albedo = None # GlacierAlbedo() TODO: implement Albedo object

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
        temperature at height z in (deg C), SRF is the so called shortwave
        radiation factor (m2 mm W-1 d-1), G is the incoming shortwave
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

        mb_day = self.t * T + self.srf * (1- alpha) * G

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


class Layer(object):
    """
    The base class that all snow/firn/preosclosed/ice layers should inherit from.
    """

class SnowLayer(object):
    """
    An object representing a layer of snow.

    Attributes
    ----------
    swe: float
        The snow water equivalent (m w.e.) of the layer.
    rho: float
        The density (kg m-3) of the layer.
    origin: datetime.datetime
        The origin date of the layer (= when the layer was deposited).
    sh: float
        The height of the layer (m), derived from the descript "SH" for snow
        height.
    cold_content: float
        The cold content of the layer (J m-2).
    refreezing_potential: float
        The refreezing potential of the layer (J kg-1)
    liq_content: float
        Liquid content within the layer (m w.e.)
    status: str
        Status of the layer. Can bei either of "snow", "firn", "poresclosed"
        or "ice".
    """

    def __init__(self, swe, rho, origin, temperature=None, liq_content=None):
        """

        Parameters
        ----------
        swe: float
            The snow water equivalent of the snow/firn layer (m w.e.).
        rho: float
            The density of the snow/firn layer (kg m-3).
        origin: datetime.datetime
            The origin date of the snow/firn layer.
        last_update: datetime.datetime
            When the layer was last updated.
        temperature: float
            Temperature of the layer (K).
        liq_content: float
            Liquid content of the snowpack (m w.e.).

        """

        self.swe = swe
        self.rho = rho
        self.temperature = temperature
        self.origin = origin
        self.last_update = origin
        self.liq_content = liq_content  # TODO: IMPLEMENT CALCULATION


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
        if self.swe > 0.:
            self._rho = value
        else:
            self._rho = 0.

    @property
    def sh(self):
        if self.rho > 0.:
            return self.swe * (cfg.RHO_W / self.rho)
        else:
            return 0.

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        # if no or positive temperature is given, assume zero degrees Celsius
        if (value is not None) and (value <= 273.16):
            self._temperature = value
        else:
            self._temperature = 273.16

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
    def cold_content(self):
        """
        The cold content of the layer, expressed in J m-2.

        The sign of the cold content is negative!

        Returns
        -------
        ccont: float
            The cold content.
        """
        ccont = np.negative(cfg.HEAT_CAP_ICE * self.rho * self.sh *
                            self.temperature)
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

    # TODO: Find a more intelligent solution than implementing "not allowed" types in a SnowLayer
    @property
    def status(self):
        if self.rho < cfg.PARAMS['snow_firn_threshold']:
           return 'snow'
        elif cfg.PARAMS['snow_firn_threshold'] <= self.rho < cfg.PARAMS[
            'pore_closeoff']:
            return 'firn'
        elif cfg.PARAMS['pore_closeoff'] <= self.rho < cfg.RHO:
            return 'poresclosed'
        else:
            return 'ice'

    def __repr__(self):
        summary = ['<crampon.{}>'.format(self.__class__.__name__)]
        summary += ['  Snow Water Equivalent (m): ' + str(self.swe)]
        summary += ['  Density (kg m-3): ' + str(self.rho)]
        summary += ['  Snow Height (m): ' + str(self.sh)]
        summary += ['  Liquid Water Content (m): ' + str(self.liq_content)]
        summary += ['  Temperature: ' + str(self.temperature)]
        summary += ['  Cold Content: ' + str(self.cold_content)]
        summary += ['  Refreezing Potential: ' + str(self.refreezing_potential)]
        summary += ['  Origin date: ' + str(self.origin)]

        return '\n'.join(summary) + '\n'


class FirnLayer(SnowLayer):
    """
    An object representing a layer of firn.
    """
    def __init__(self, swe, rho, origin, temperature=None, liq_content=None):
        """
        Instantiate.

        Parameters
        ----------
        swe
        rho
        origin
        temperature
        liq_content
        init_density: float
            The initial density of the firn layer (important for the densification!).

        """
        super().__init__(swe, rho, origin, temperature=temperature,
                         liq_content=liq_content)

        self.init_density = None


class PoresClosedLayer(SnowLayer):
    """
       An object representing a layer of firn.
       """

    def __init__(self, swe, rho, origin, temperature=None, densification_rate=10.):
        """
        Instantiate a layer with close pores.

        Parameters
        ----------
        swe: float
            Snow water equivalent (m w.e.).
        rho: float
            Density (kg m-3).
        origin: datetime.datetime
            Origin date of the layer.
        temperature: float
            Temperature (K).
        densification_rate: float
            Rate per year with which the layer densifies (kg m-3 a-1).
        """
        super().__init__(swe, rho, origin, temperature=temperature,
                         liq_content=0.)

        self.densification_rate = densification_rate

    # TODO: Is this true?
    @property
    def liq_content(self):
        return 0.


class IceLayer(SnowLayer):

    def __init__(self, swe, rho, origin, temperature=None):
        super().__init__(swe, rho, origin, temperature=temperature,
                         liq_content=0.)

    # TODO: Is this true?
    @property
    def liq_content(self):
        return 0.

    # TODO: This is strange, because in the factor we allow values up to 1000.
    @property
    def rho(self):
        return cfg.RHO


class LayerFactory(object):
    """
    This class returns always the correct layer type with a certain input.
    """

    @staticmethod
    def make_layer(swe, rho, origin, temperature=None, liq_content=None):
        """
        Creates Layer objects.

        At the moment SnowLayer, FirnLayer , PoresClosedLayer and IceLayer can
        be created. The decision which object is returned depends solely on
        density thresholds.

        Parameters
        ----------
        swe: float
            Snow water equivalent (m w.e.).
        rho: float
            Density (kg m-3).
        origin: datetime.datetime
            The origin date of the layer to be created.
        temperature: float, optional
            Temperature (K).
        liq_content: float, optional
            Liquid water content (m water).

        Returns
        -------
        A layer object or a NotImplementedError if something goes wrong.
        """
        if 0. < rho < cfg.PARAMS['snow_firn_threshold']:
            return SnowLayer(swe, rho, origin, temperature=temperature,
                             liq_content=liq_content)
        elif cfg.PARAMS['snow_firn_threshold'] <= rho < cfg.PARAMS[
            'pore_closeoff']:
            return FirnLayer(swe, rho, origin, temperature=temperature,
                             liq_content=liq_content)
        elif cfg.PARAMS['pore_closeoff'] <= rho < cfg.RHO:
            return PoresClosedLayer(swe, rho, origin, temperature=temperature)
        elif 1000. > rho >= cfg.RHO:
            return IceLayer(swe, rho, origin, temperature=temperature)
        else:
            raise NotImplementedError('The values given do not seem to fit '
                                      'any implemented layer class.')


class SnowFirnPack(object, metaclass=SuperclassMeta):
    """
    Implements a point snow/firn pack as a vertical collection of layers.
    """

    def __init__(self, swe, rho, origin, temperature=None,
                 liq_content=None, pore_close=None, firn_ice_transition=None,
                 refreezing=True):

        self.init_swe = swe
        self.init_rho = rho
        self.origin = origin
        self.init_temperature = temperature
        self.liq_content = liq_content

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
        self._tgrid_nodes = [np.array([0., self.total_height])]
        if temperature is not None:
            self.temperature = temperature
            self._tgrid_temperature = temperature
        else:
            # initiate with zero deg C, if we don't know better
            self.temperature = np.ones(n_layers) * 273.16
            self._tgrid_temperature = np.ones(n_layers) * 273.16

        self.layers = self.setup_layers()

    def setup_layers(self):
        """
        Set up a grid for the given height nodes.

        The grid is an array containing at each height node a list of
        SnowFirnLayer objects. Index zero of each list is always the layer
        expose to the surface ("top layer"), while the last index is always
        the layer in contact with the underlying surface (ice).

        If there are no layers at a height node, a list will exist, but be
        empty.

        Returns
        -------
        self.grid: np.ndarray of lists
            A grid object
        """

        pack = [0] * self.n_layers

        for n in range(self.n_layers):
            pack[n] = LayerFactory.make_layer(self.init_swe[n],
                                              self.init_rho[n], self.origin[n],
                                              temperature=self.temperature[n],
                                              liq_content=self.liq_content[n])
        return pack

    @property
    def n_layers(self):
        return len(self.layers)

    @property
    def snow_height(self):
        return np.sum([l.sh for l in self.layers if type(l) == SnowLayer])

    @property
    def firn_height(self):
        return np.sum([l.sh for l in self.layers if type(l) == FirnLayer])

    @property
    def poresclosed_height(self):
        return np.sum([l.sh for l in self.layers if type(l) == PoresClosedLayer])

    @property
    def ice_height(self):
        return np.sum([l.sh for l in self.layers if type(l) == IceLayer])

    @property
    def total_height(self):
        return np.sum([l.sh for l in self.layers])

    @property
    def mean_density(self):
        return [np.average([l.rho for l in self.layers],
                           weights=[l.sh for l in self.layers])]

    def add_layer(self, swe, rho, origin, temperature=None, liq_content=None,
                  ix=0):
        """
        Add a layer to the SnowFirnPack.

        Parameters
        ----------
        ix: int
            Where to add the layer. Default: 0 (top).
        """
        l_obj = LayerFactory.make_layer(swe, rho, origin,
                                        temperature=temperature,
                                        liq_content=liq_content)
        self.layers.insert(ix, l_obj)

    def remove_layer(self, ix=0):
        """
        Remove a layer from the SnowFirnPack.

        Parameters
        ----------
        ix: int
            Index of the layer to remove. Default: 0 (top).
        """

        if self.layers[ix]:
            self.layers.pop(ix)
        else:
            raise IndexError('No layer to be removed')

    def melt(self, swe):
        """
        Removes a layer from the snow/firn pack.

        Parameters
        ----------
        swe: float
            The snow water equivalent to be removed. Should be negative
            numbers.
        """

        for mass in enumerate(swe):

            layer = 0  # start at the top
            while mass > 0:
                if self.layers[layer].swe > np.abs(mass):
                    # TODO: Here we should think of a runoff/refreeze routine
                    self.layers[layer].swe += mass  # mass is negative!
                    continue
                else:  # subtract layer completely
                        mass += self.layers[layer].swe
                        self.remove_layer(layer)

                layer += 1


            # TODO: replace pseudo-code??? Does it have to refreeze every time if there is potential? Do we need impermable layers for this? Look at SFM2
            # if self.refreeze_pot > 0.:
            # let it refreeze
            # let the latent heat warm the snowpack up

    def ingest_water_equivalent_top(self, swe, rho, origin, temperature=None,
                                liq_content=None):
        """
        Add a mass balance (positive or negative) to the SnowFirnPack

        Parameters
        ----------
        swe: float
            A mass balance in meters water equivalent (m w.e.) to be ingested
            inplace into the SnowFirnPack.

        Returns
        -------
        None
        """

        if swe > 0.:
            self.add_layer(swe, rho, origin, temperature=temperature,
                           liq_content=liq_content)
        elif swe < 0.:
            self.melt(swe)
        else:
            pass

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

        for l in range(len(self.layers) - 1):
            # if we've become too short already
            try:
                self.layers[l + 1]
            except IndexError:
                continue

            if self.layers[l].sh < min_sh:
                # update l
                layer_one = self.layers[l]
                layer_two = self.layers[l + 1]
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

                insert = LayerFactory.make_layer(swe=new_swe, rho=new_rho,
                                   origin=new_origin,
                                   temperature=new_temperature,
                                   liq_content=new_liq_content)

                self.layers[l] = insert

                # remove l+1
                self.remove_layer(l + 1)

    def update_layers(self):
        """
        Take care of transferring snow to firn to poresclosed to ice layers.

        TODO: This is a bit more complicated, because here we decide when a snow layer is taken into a firn layer aggregation (not only by density!)

        Returns
        -------

        """

        # employ the factory with the properties of each existing layer to have a centralized place where we determine the properties of the layers


class SnowFirnCover(object, metaclass=SuperclassMeta):
    """ Implements a an interface to a snow and/or firn cover.

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
        """

        # TODO: SHOULD THERE BE AN "INIT" in front of every parameter? Later we
        # don't use them anymore
        self.height_nodes = height_nodes
        self.swe = swe
        self.rho = rho
        self.sh = self.swe * (cfg.RHO_W / self.rho)
        self.origin = [origin] * self.n_heights
        self.liq_content = np.zeros_like(height_nodes)

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
        self._tgrid_nodes = [np.array([0., self.sh[i]]) for i in
                             range(len(self.sh))]
        if temperatures is not None:
            self.temperature = temperatures
            self._tgrid_temperature = temperatures
        else:
            # initiate with zero deg C, if we don't know better
            self.temperature = np.ones(height_nodes) * 273.16
            self._tgrid_temperature = np.ones(height_nodes) * 273.16

        self.grid = self.grid_setup()

    @property
    def n_heights(self):
        return len(self.height_nodes)

    def grid_setup(self):
        """
        Set up a grid for the given height nodes.

        The grid is an array containing at each height node a list of
        SnowFirnLayer objects. Index zero of each list is always the layer
        expose to the surface ("top layer"), while the last index is always
        the layer in contact with the underlying surface (ice).

        If there are no layers at a height node, a list will exist, but be
        empty.

        Returns
        -------
        self.grid: np.ndarray of lists
            A grid object
        """

        grid = [0] * self.n_heights

        for n in range(self.n_heights):
            if self.rho[n] < cfg.PARAMS['snow_firn_threshold']:
                grid[n] = [SnowLayer(self.swe[n], self.rho[n], self.origin[n],
                                     self.temperature[n], self.liq_content[n])]
            else:
                grid[n] = [FirnLayer(self.swe[n], self.rho[n], self.origin[n],
                                     self.temperature[n], self.liq_content[n])]
        return grid

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

        if layertype.lower() not in ['snow', 'firn', 'poresclosed', 'ice']:
            raise ValueError('Type {} not accepted. Must be either of "snow", '
                             '"firn", "poresclosed" or "ice".'.format(layertype))

        indices = []
        for h, layers in enumerate(self.grid):
            for l, layer in enumerate(layers):
                if layertype == 'snow':
                    if type(layer) == SnowLayer:
                        indices.append((h, l))
                elif layertype == 'firn':
                    if type(layer) == FirnLayer:
                        indices.append((h, l))
                elif layertype == 'poresclosed':
                    if type(layer) == PoresClosedLayer:
                        indices.append((h, l))
                elif layertype == 'ice':
                    if type(layer) == IceLayer:
                        indices.append((h, l))

        return indices

    def add_layer(self, swe, rho, origin, temperature=None, liq_content=None,
                  ix=0):
        """
        Add a layer to the snow/firn pack.

        Parameters
        ----------
        ix: int
            Where to add the layer. Default: 0 (top).
        """

        if len(swe) != self.n_heights:
            raise ValueError('Dimensions of SnowFirnCover and mass to be '
                             'added must match.')

        for h in np.where(swe > 0)[0]:
            if rho[h] < cfg.PARAMS['snow_firn_threshold']:
                self.grid[h].insert(ix, SnowLayer(swe[h], rho[h], origin))
            else:
                self.grid[h].insert(ix, FirnLayer(swe[h], rho[h], origin))

    def remove_layer(self, height_ix, layer_ix=0):
        """
        Remove a layer from the snow/firn pack.

        Parameters
        ----------
        ix: int
            Where to add the layer. Default: 0 (top).
        """

        if self.grid[height_ix][layer_ix]:
            self.grid[height_ix].pop(layer_ix)
        else:
            raise IndexError('No layer to be removed')

    def melt(self, swe):
        """
        Removes a layer from the snow/firn pack.

        Parameters
        ----------
        swe: np.ndarray
            The snow water equivalent to be removed. Should be negative
            numbers.
        """

        if len(swe) != self.n_heights:
            raise ValueError('Dimensions of SnowFirnCover and mass to be '
                             'removed must match.')

        for height_ix, mass in enumerate(swe):

                layer = 0  # start at the top
                while mass > 0:
                    if self.grid[height_ix][layer].swe > np.abs(mass):
                        # TODO: Here we should think of a runoff/refreeze routine
                        self.grid[height_ix][layer].swe += mass  # mass is negative!
                        continue
                    else:  # subtract layer completely
                            mass += self.grid[height_ix][layer].swe
                            self.remove_layer(height_ix, layer)

                    layer += 1


            # TODO: replace pseudo-code??? Does it have to refreeze every time if there is potential? Do we need impermable layers for this? Look at SFM2
            # if self.refreeze_pot > 0.:
            # let it refreeze
            # let the latent heat warm the snowpack up

    def add_height_nodes(self, nodes):
        """
        If the glaciers advances, this is the way to tell the snow/firn cover.

        For simplicity reasons, the snow/firn cover at the lowest height index
        is copied to the added node(s).

        Parameters
        ----------
        nodes: array-like
            The heights that should be added to the existing height nodes.
        """
        raise NotImplementedError

    def remove_height_nodes(self, nodes):
        """
        If the glacier retreats, this is the way to tell the nosw/firn cover.

        The question is here is this should take indices or heights (precision problem)
        """

    def get_total_height(self):
        """
        Get total height of the firn and snow cover together.
        """

        total_height = [np.sum([l.sh for l in h]) for h in self.grid]
        return total_height

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

        total_rho = [np.average([l.rho for l in h], weights=[l.sh for l in h])
                     for h in self.grid]
        return total_rho

    def get_layer_depths(self, where='center'):
        """
        Get the center depths of every layer.

        The where kwarg should tell if the layer "top"s, "center"s or
        "bottom"s should be retrieved.
        """
        raise NotImplementedError

    def remove_ice_layers(self):
        """
        Removes layers a the bottom (only there!) that have exceeded the
        threshold density for ice.

        Returns
        -------
        None
        """

        ice_ix = self.get_type_indices('ice')

        for h, l in ice_ix:

            # Ensure we are somewhere at the bottom:
            ice_layers = [j for i, j in ice_ix if i == h]
            if all([i in ice_layers for i in range(l, len(self.grid[h]))]):
                self.remove_layer(h, l)

    def update_temperature_profile(self, max_depth=15, dx=0.1,
                                   lower_bound=273.16):
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

        Returns
        -------
        None
        """

        # Make a grid until 15 m depth as attribute if not yet exists;
        total_height = self.get_total_height()
        if self.get_total_height() < max_depth:
            self._tgrid_nodes = np.arange(0., total_height, dx)
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

        Parameters
        ----------
        max_depth: float
            Maximum depth in meters until which the refreezing potential is
            calculated. Default: 15.

        Returns
        -------
        None
        """

        # Update potential inplace
        for h_node in self.grid:
            depth = 0.
            while depth < max_depth:
                for l in h_node:
                    l.get_refreezing_potential()

                    depth += l.get_height()

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

        self.merge_firn_layers(date)

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

        for h, l in firn_ix:

            # careful with days!!!! (the difference might be 364 days or so)

            # TODO: ACTUALLY WE SHOULD DERIVE THE EQUATION AND THEN GO FROM TIME STEP TO TIME STEP
            t = (date - self.grid[h][l].origin).days / 365.25

            if rho_f0_const:
                # initial firn density from Huss (2013) is 490, anyway...
                rho_f0 = cfg.PARAMS['snow_firn_threshold']  # kg m-3
            else:
                rho_f0 = self.grid[h][l].init_density

            # just to not divide by zero when a layer forms
            if t != 0.:
                b = np.sum([i.swe for i in self.grid[h][
                                           :l]]) / t  # annual accumulation rate

                c_reeh_ice = k1 * np.sqrt(
                    b * cfg.RHO / cfg.RHO_W)  # 550 kg m-3 < rho_f < 800 kg m-3

                # Huss 2013, equation 5
                rho_f = cfg.RHO - (cfg.RHO - rho_f0) * np.exp(
                    -c_reeh_ice * t)

                self.grid[h][l].rho = rho_f
            else:
                continue

            # TODO: apply refreezing here?
            if self.refreezing:
                self.update_refreezing_potential()

                #rho_f += RF_t


        # TODO: HERE WE ASSUME ONE YEAR => t NEEDS TO BE ADJUSTED OR AN ATTRIBUTE "LAST_UPDATED" NEEDS TO BE MADE
        for h, l in poresclosed_ix:
            self.grid[h][l].rho = self.grid[h][l].rho + poresclosed_rate * dt.timedelta(date - self.grid[h][l].origin).years

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

        # CHECK HERE IF THE DATE IS MORE THAN ONE DAY AWAY FROM THE LAST UPDATE?
        dt = target_dt

        # UPDATE TEMPERATURES HERE

        height_ix = np.where(self.grid)[0]  # where there is something at all

        # DON'T USE THE RETRIEVED INDICES HERE: otherwise overlying mass could be wrong if layers above are by chance firn ( densification(ice lenses by refreezing!)
        for h in height_ix:
            mass = 0.
            for l in range(len(self.grid[h])):
                rho_old = self.grid[h][l].rho
                mass += rho_old * self.grid[h][l].sh

                if type(self.grid[h][l]) == SnowLayer:
                    self.grid[h][l].rho = rho_old + \
                                          (rho_old * cfg.G * mass * dt / eta0) * \
                                          np.exp(etaa * (self.grid[h][
                                                             l].temperature - Tm_k) - etab *
                                                 rho_old) + dt * \
                                          rho_old * snda * np.exp(
                        sndb * (self.grid[h][l].temperature - Tm_k) - sndc * max(
                            rho_old - rhoc, 0.))

        # to be sure
        self.remove_ice_layers()

    def merge_firn_layers(self, date):
        """
        E.g. application:
        If a layer bunch is older than one year, collapse them into one firn layer
        (2) Merge neighbor layers below a layer thickness of 2cm? WHAT TO DO WITH THE ORIGIN DATE?

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
                        self.grid[h][l].rho >= cfg.PARAMS['snow_firn_threshold']:
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
                # TODO: is it clever to give them the current date? Actually yes, because the daily model should have run until then already
                new_origin = date
                # new_origin = np.max([t.last_update for t in temp_layers])
                # TODO: the temperature should actually go via the energy
                new_temp = sum(
                    [t.temperature * t.swe for t in temp_layers]) / np.sum(
                    [t.swe for t in temp_layers])
                new_liq = sum([t.liq_content for t in temp_layers])
                insert = FirnLayer(swe=new_swe, rho=new_rho, origin=new_origin,
                                   temperature=new_temp, liq_content=new_liq)
                insert.init_density = new_rho

                self.grid[h].insert(min(temp_inds), insert)

        # HOW TO ENSURE THAT THE LAYERS ARE NOT MERGED WITH OTHER FIRN LAYERS? INTRODUCE STATUS ATTRIBUTE?

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
                    # TODO: This is not correct and should be changed after the LayerFactory is built
                    insert = SnowLayer(swe=new_swe, rho=new_rho,
                                       origin=new_origin,
                                       temperature=new_temperature,
                                       liq_content=new_liq_content)

                    self.grid[h][l] = insert

                    # remove l+1
                    self.remove_layer(h, l + 1)


    # MAYBE THIS METHOD SHOULD BE OUTSIDE THE CLASS => IT'S A CONTROLLER
    def run_densification(gdir, from_date=None, to_date=None,
                          recalculate_mb=False):
        """
        Run the densification process and subprocesses.

        Parameters
        ----------
        from_date
        to_date
        recalculate_mb: bool
            Whether to recalculate the mass balance (computationally
            expensive!) or - if available - fall back to the saved mass
            balance in the glacier directory (mb_daily.pkl).

        Returns
        -------

        """

        # partition the date range into full MB years and the time where we
        # need the daily model

        # IMPORTANT: This should by lazy: it should only recalculate the MB

        for d in pd.date_range(from_date, to_date):
            print(d)
            # if d is the end of the MB year:
            # self.densify_firn_huss
            # if d is a day and we are i the part where we use daily models:
            # self.densify_snow

    def return_state(self, dataset=False):
        """This should be a function that can be called to get the current
        status of the snow/firn cover as numpy arrays or xr.Dataset. If put in
        a loop, the history can be retrieved."""

        # recipe to convert list of lists to array
        #x = [[1, 2], [1, 2, 3], [1]]
        #length = len(sorted(x, key=len, reverse=True)[0])
        #y = np.array([xi + [None] * (length - len(xi)) for xi in x])

        #if dataset:
        #ds = xr.Dataset()


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


class MassBalance(object, metaclass=SuperclassMeta):
    """
    Basic interface for mass balance objects.
    """
    def __init__(self, gdir, mb_model, dataset=None):
        """
        Instantiate the MassBalance base class.

        Parameters
        ----------
        gdir: `py:class:crampon.GlacierDirectory`
        mb_model: `py:class:crampon.core.models.massbalance.MassBalanceModel`
            The mass balance model used to calculate the time series.
        dataset: xr.Dataset

        """

        self.gdir = gdir
        self.mb_model = mb_model
        if dataset:
            # TODO: Implement checks here if the given object has all requirements to be a mass balance dataset
            self._obj = dataset
        self.cali_pool = None # TODO: Calibration(); for this I need to write a calibration object and implement cli parameters attributes in the MB_model classes

        @lazy_property
        def apply_cumsum():
            """
            Apply cumsum to mass balance data.

            Returns
            -------
            cumsum: xr.Dataset
                The mass balance dataset as cumulative sum.
            """

    def create_specific(self, MassBalanceModel, from_date=None, to_date=None, write=True):
        """
        Create the time series from a given mass balance model.

        Parameters
        ----------
        MassBalanceModel: crampon.core.models.massbalance.MassBalanceModel
            The model used to create the mass baöance time series.
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
            # TODO: write a write_snow and write_mb method that can be called here

    def make_quantiles(self):
        """
        Apply quantiles to the mass balance data.

        Returns
        -------

        """

    def apply_cumsum(self):
        """
        Apply the cumulative sum.

        Returns
        -------

        """


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
    cover = SnowFirnCover(heights, swe=np.zeros_like(heights),
                          rho=np.ones_like(heights)*100.,
                          origin=run_time[0],
                          temperatures=np.ones_like(heights)*273.16,
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
    for date in run_time:
        # Get the mass balance and convert to m w.e. per day
        before = dt.datetime.now()
        tmp = day_model.get_daily_mb(heights, date=date) * 3600 * 24 * cfg.RHO / 1000.
        mb.append(tmp)
        after = dt.datetime.now()
        get_mb_time.append(after - before)

        before = dt.datetime.now()
        if (tmp > 0.).any():
            cover.add_layer(swe=np.clip(tmp, 0, None), rho=np.ones_like(tmp) * 100.,
                            origin=date)
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

        if date.day == 1:
            print('Merging', np.sum([len(g) for g in cover.grid]))
            before = dt.datetime.now()
            cover.merge_layers(min_sh=0.1)
            after = dt.datetime.now()
            merge_layers_time.append(after - before)
            print(np.sum([len(g) for g in cover.grid]))

        if date.month == 10 and date.day == 1:
            #rho_snow_end.append([max([i.rho for i in j if dt.timedelta(date - i.origin).days < 365] for j in len(cover.grid))])
            print('Densifying a la Huss')
            before = dt.datetime.now()
            cover.densify_firn_huss(date)
            after = dt.datetime.now()
            densify_firn_huss_time.append(after - before)



