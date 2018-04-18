from __future__ import division

from oggm.core.massbalance import *
from crampon.cfg import SEC_IN_DAY
from crampon.utils import SuperclassMeta, lazy_property, closest_date
import xarray as xr


class DailyMassBalanceModel(MassBalanceModel):
    """
    Child of OGGM's PastMassBalanceModel, able to calculate daily mass balance.
    """

    def __init__(self, gdir, mu_star=None, bias=None, prcp_fac=None,
                 filename='climate_daily', filesuffix=''):

        """
        PastMassBalanceModel.__init__(self, gdir=gdir, mu_star=mu_star, bias=bias,
                                      prcp_fac=prcp_fac, filename=filename,
                                      filesuffix=filesuffix)
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
            #if isinstance(self.prcp_fac, pd.Series):
            #    self.prcp = nc.variables['prcp'][:] * \
            #                self.prcp_fac.reindex(index=time,
            #                                      method='nearest')\
            #                    .fillna(value=np.nanmean(self.prcp_fac))
            #else:
            #    self.prcp = nc.variables['prcp'][:] * self.prcp_fac
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
            iprcp = self.prcp_unc[ix] * self.prcp_fac[self.prcp_fac.index.
                get_loc(date, **kwargs)]
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
            mu_star = self.mu_star.iloc[self.mu_star.index.get_loc(
                date, **kwargs)]
        else:
            mu_star = self.mu_star

        if isinstance(self.bias, pd.Series):
            bias = self.bias.iloc[self.bias.index.get_loc(date, **kwargs)]
        else:
            bias = self.bias

        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - mu_star * tempformelt - bias

        self.time_elapsed = date

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / SEC_IN_DAY / cfg.RHO
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
        mbs = self.get_daily_mb(heights, date=date, **kwargs) * SEC_IN_DAY * cfg.RHO /\
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
            #if isinstance(self.prcp_fac, pd.Series):
            #    self.prcp = nc.variables['prcp'][:] * \
            #                self.prcp_fac.reindex(index=time,
            #                                      method='nearest').fillna(
            #        value=np.nanmean(self.prcp_fac))
            #else:
            #    self.prcp = nc.variables['prcp'][:] * self.prcp_fac
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
            iprcp = self.prcp_unc[ix] * self.prcp_fac[self.prcp_fac.index.
                get_loc(date, **kwargs)]
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
            mu_ice = self.mu_ice.iloc[
                self.mu_ice.index.get_loc(date, **kwargs)]
        else:
            mu_ice = self.mu_ice

        if isinstance(self.mu_snow, pd.Series):
            mu_snow = self.mu_snow.iloc[
                self.mu_snow.index.get_loc(date, **kwargs)]
        else:
            mu_snow = self.mu_snow

        if isinstance(self.bias, pd.Series):
            bias = self.bias.iloc[
                self.bias.index.get_loc(date, **kwargs)]
        else:
            bias = self.bias


        # Get snow distribution from yesterday and determine snow/ice from it;
        # Could also get the accumulation first & update snow before melting
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
            if ((self.snow[-1] - self.snow[-366]) >= 0.).any():  # check MB pos
                inds = np.where((self.snow[-1] - self.snow[-366]) >= 0.)
                self.snow[-1][inds] = np.clip(self.snow[-1][inds] - np.clip(self.snow[-365][inds] - self.snow[-366][inds], 0.,
                                        None), 0., None)  # cheap removal
        except IndexError: # when date not yet exists
            pass

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / SEC_IN_DAY / cfg.RHO
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
    """ This class implements the melt model by Pellicciotti et al. (2005)."""

    def __init__(self):
        raise NotImplementedError

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

'''
class SnowFirnModel(object):
    """
    A class that manages firn and snow densification and removal.
    """

    def __init__(self, gdir, f_firn=2.4, f_snow=1.):
        """

        Parameters
        ----------
        gdir: :py:class:`crampon.GlacierDirectory`
            The GlacierDirectory for which to calculate the firn densification.
        f_firn: float
           A factor empirically determined by Herron and Langway (1980), used
           by Huss 2013 to tune simulated to observed firn densification
        f_snow: float
           A factor empirically determined by Herron and Langway (1980), not
           used in Huss 2013. Needs to be calibrated somehow! Default: 1 (i.e.
           use the emergency exit and take what Reeh 2008 found.

        """

        # Mean annual surface temperature, 'presumably' firn temperature at 10m
        # depth (Reeh (2008)); set to 273 according to Huss 2013
        self.T_ms = 273.  # K
        # initial firn density, Huss (2013) (OVERLAPS 550 threshold from Herron
        # & Langway (1980)!!!)
        self.rho_f0 = 490. # kg m-3
        # pore close-off density, Huss 2013 contradicts: paper says 830, script says 845
        self.pore_close = 837.5  # kg m-3
        # constants for snow (k0) and firn (k1) densification
        self.k0 = f_snow * 11. * np.exp(- cfg.E_SNOW / (cfg.R * self.T_ms))  # m-1
        self.k1 = f_firn * 575. * np.exp(- cfg.E_FIRN / (cfg.R * self.T_ms))  # m(-0.5)a(-0.5)
        self.mb_ds = gdir.read_pickle('mb_daily')
        # firn and snow
        #
        self.firnsnow = xr.DataSet({'SWE': (['time', 'n'],
                                    'RHO':['time', 'n'],
                                    'HS': ['time', 'n']},
                           coords={'n': (['n'], ),
                                   'time': pd.to_datetime(day_model.span_meteo)},
                           attrs={'id': gdir.rgi_id,
                                  'name': gdir.name})

        self.firnsnow_pd = pd.DataFrame(columns=['SWE', 'RHO', 'HS'])

    def update_firn_density(self):
        """

        Parameters
        ----------
        b: float
            specific net balance (m ice a-1)

        Returns
        -------

        """


        # Might it be best to let the class inherit an xr.Dataset=?


        # the factors for snow and ice
        c_reeh_snow = self.k0 * b * cfg.RHO / cfg.RHO_W  # rho_f < 550 kg m-3
        c_reeh_ice = self.k1 * np.sqrt(b * cfg.RHO / cfg.RHO_W)  # 550 kg m-3 < rho_f < 800 kg m-3


        firn_dataset[origin_date== today] = np.sum(mb_ds.MB.over_last_budget_year)


        # pseudo code
        for origin_date, layer in firn_dataset.iterrows():
            # problem here: what happens between 800 and pore close-off? Replace 800 with pore close-off?
            if 550 < layer.density < 800:
            if 550 < layer.density < self.pore_close:

                # careful with days!!!! (the difference might be 364 days or so)
                t = datetime.timedelta(datetime.today - layer.origin_date)
                t = convert_timedelta_to_year_decimal_number()
                # Huss 2013, equation 5
                rho_f = cfg.RHO - (cfg.RHO - self.rho_f0) * np.exp(-c * t) + RF_t

            elif layer.density > 800:
            elif layer.density > self.pore_close:



            layer.density = rho_f
            if density < self.pore_close:
                layer.status = 'firn'
            else:
                layer.status = 'porecloseoff'
'''