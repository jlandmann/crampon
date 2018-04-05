from __future__ import division

from oggm.core.massbalance import *
from crampon.cfg import SEC_IN_DAY
from crampon.utils import SuperclassMeta, lazy_property
import xarray as xr
import cython


class DailyMassBalanceModel(MassBalanceModel):

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
            df = pd.read_csv(gdir.get_filepath('local_mustar'))
            mu_star = df['mu_star'][0]
        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                df = pd.read_csv(gdir.get_filepath('local_mustar'))
                bias = df['bias'][0]
            else:
                bias = 0.
        if prcp_fac is None:
            df = pd.read_csv(gdir.get_filepath('local_mustar'))
            prcp_fac = df['prcp_fac'][0]
        self.mu_star = mu_star
        self.bias = bias
        self.prcp_fac = prcp_fac

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
            self.tspan_in = time
            #self.years = np.unique([y.year for y in self.nc_time])

            # Read timeseries
            self.temp = nc.variables['temp'][:]
            self.prcp = nc.variables['prcp'][:] * self.prcp_fac
            self.tgrad = nc.variables['grad'][:]
            self.pgrad = cfg.PARAMS['prcp_grad']
            self.ref_hgt = nc.ref_hgt

        # Public attrs
        self.temp_bias = 0.

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

    def get_daily_mb(self, heights, date=None):
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

        Returns
        -------
        ndarray:
            Glacier mass balance at the respective heights in m ice s-1.
        """

        # index of the date of MB calculation
        ix = pd.DatetimeIndex(self.tspan_in).get_loc(date)
        #ix = netCDF4.date2index(date, self.nc_time, select='exact')

        # Read timeseries
        itemp = self.temp[ix] + self.temp_bias
        iprcp = self.prcp[ix]
        itgrad = self.tgrad[ix]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + itgrad * (heights - self.ref_hgt)

        tempformelt = self.get_tempformelt(temp)
        prcpsol, _ = self.get_prcp_sol_liq(iprcp, heights, temp)

        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - self.mu_star * tempformelt - \
                 self.bias

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / SEC_IN_DAY / cfg.RHO
        return icerate

    def get_daily_specific_mb(self, heights, widths, date=None):
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
            out = [self.get_daily_specific_mb(heights, widths, date=d)
                   for d in date]
            return np.asarray(out)

        # m w.e. d-1
        mbs = self.get_daily_mb(heights, date=date) * SEC_IN_DAY * cfg.RHO /\
              1000.
        mbs_wavg = np.average(mbs, weights=widths)
        return mbs_wavg


class BraithwaiteModel(DailyMassBalanceModel):

    def __init__(self, gdir, mu_ice=None, mu_snow=None, bias=None,
                 prcp_fac=None, snow_init=None, filename='climate_daily',
                 filesuffix=''):

        self.mu_ice = mu_ice
        self.mu_snow = mu_snow
        self.prcp_fac = prcp_fac
        self.bias = bias

        DailyMassBalanceModel.__init__(self, gdir, mu_star=self.mu_ice,
                                       bias=self.bias, prcp_fac=self.prcp_fac,
                                       filename='climate_daily', filesuffix='')

        self.heights, self.widths = gdir.get_inversion_flowline_hw()
        if snow_init:
            self.snow_init = snow_init
        else:
            self.snow_init = np.zeros(
                [len(self.tspan_in), len(self.heights)])
        self.snow = np.atleast_2d(np.zeros_like(self.heights))

        # self.snow = xr.Dataset({'swe': (['time', 'h'], self.snow_init)},
        #                      coords={
        #                          'h': (['h'], range(len(self.heights))),
        #                          'time': pd.to_datetime(self.tspan_in)})

    def get_daily_mb(self, heights, date=None):
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

        Returns
        -------
        ndarray:
            Glacier mass balance at the respective heights in m ice s-1.
        """

        # index of the date of MB calculation
        ix = pd.DatetimeIndex(self.tspan_in).get_loc(date)
        #ix = netCDF4.date2index(date, self.nc_time, select='exact')

        # Read timeseries
        itemp = self.temp[ix] + self.temp_bias
        iprcp = self.prcp[ix]
        itgrad = self.tgrad[ix]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + itgrad * (heights - self.ref_hgt)

        tempformelt = self.get_tempformelt(temp)
        prcpsol, _ = self.get_prcp_sol_liq(iprcp, heights, temp)

        # Get snow distribution from yesterday and determine snow/ice from it
        # One could also get the accumulation first & update snow before melting
        snowdist = np.where(self.snow[-1] > 0.)
        mu_comb = np.zeros_like(self.snow[-1])
        mu_comb[:] = self.mu_ice
        np.put(mu_comb, snowdist, self.mu_snow)

        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - mu_comb * tempformelt - self.bias

        self.update_snow(date, mb_day)

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
            Mass balance for elevation bands.

        Returns
        -------
        self.snow:
            The updated snow cover with the reference height, the date when the
            last fresh snow has fallen on the surface and the amount of snow
            present.
        """

        ix = pd.DatetimeIndex(self.tspan_in).get_loc(date)

        snow_today = None
        #print(mb, self.snow.isel(time=ix-1).swe)
        # for first day, offset with snow_init
        if ix > 1:
            #snow_today = self.snow.isel(time=ix-1).swe + mb
            #print(date, snow_today)
            snow_today = np.clip((self.snow[-1,:] + mb), 0, None)
            self.snow = np.vstack((self.snow, snow_today))
        elif ix == 0:
            snow_today = np.clip(mb, 0, None)
            self.snow = self.snow + snow_today
        else:
            snow_today = np.clip(mb, 0, None)
            self.snow = np.vstack((self.snow, snow_today))
            #snow_today = self.snow.isel(time=ix).swe + mb
        #snow_today = np.clip(snow_today, 0, None)
        #print(self.snow)
        #self.snow.update(snow_today.to_dataset().expand_dims('time'),
        #                 inplace=True)

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
