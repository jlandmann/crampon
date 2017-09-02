from __future__ import division

from oggm.core.models.massbalance import *
from crampon.cfg import SEC_IN_DAY
from crampon.utils import SuperclassMeta, lazy_property


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
            df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
            mu_star = df['mu_star'][0]
        if bias is None:
            if cfg.PARAMS['use_bias_for_run']:
                df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
                bias = df['bias'][0]
            else:
                bias = 0.
        if prcp_fac is None:
            df = pd.read_csv(gdir.get_filepath('local_mustar', div_id=0))
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

        # Public attrs
        self.temp_bias = 0.

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

    def get_daily_mb(self, heights, date=None):

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
        tempformelt = temp - self.t_melt
        tempformelt[:] = np.clip(tempformelt, 0, tempformelt.max())

        # Compute solid precipitation from total precipitation
        # the prec correction with the gradient does not exist OGGM
        prcpsol = np.ones(npix) * iprcp + self.pgrad * (heights - self.ref_hgt)
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol *= np.clip(fac, 0, 1)

        # (mm w.e. d-1) = (mm w.e. d-1) - (mm w.e. d-1 K-1) * K - bias
        mb_day = prcpsol - self.mu_star * tempformelt - \
                 self.bias

        # return ((10e-3 kg m-2) w.e. d-1) * (d s-1) * (kg-1 m3) = m ice s-1
        icerate = mb_day / SEC_IN_DAY / cfg.RHO
        return icerate

    def get_daily_specific_mb(self, heights, widths, date=None):
        """Specific mb for a given date and geometry (m w.e. d-1)

        Parameters
        ----------
        heights: ndarray
            the altitudes at which the mass-balance will be computed
        widths: ndarray
            the widths of the flowline (necessary for the weighted average)
        date: float, optional
            the time (in the "floating year" convention)
        Returns
        -------
        the specific mass-balance (units: mm w.e. d-1)
        """
        if len(np.atleast_1d(date)) > 1:
            out = [self.get_daily_specific_mb(heights, widths, date=d)
                   for d in date]
            return np.asarray(out)

        # m w.e. d-1
        mbs = self.get_daily_mb(heights, date=date) * SEC_IN_DAY * cfg.RHO / 1000.
        mbs_wavg = np.average(mbs, weights=widths)
        return mbs_wavg

