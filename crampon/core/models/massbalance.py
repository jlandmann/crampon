from __future__ import division

from oggm.core.models.massbalance import *
from crampon.cfg import SEC_IN_DAY
from crampon.utils import SuperclassMeta, lazy_property


class DailyMassBalanceModel(PastMassBalanceModel):

    def __init__(self, gdir, mu_star=None, bias=None, prcp_fac=None,
                 filename='climate_daily', filesuffix=''):
        PastMassBalanceModel.__init__(gdir, mu_star=mu_star, bias=bias,
                                      prcp_fac=prcp_fac, filename=filename,
                                      filesuffix=filesuffix)

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

        # overwrite/add some OGGM stuff
        self.m = None
        self.years = None

        fpath = gdir.get_filepath(filename, filesuffix=filesuffix)
        with netCDF4.Dataset(fpath, mode='r') as nc:
            # time
            time = nc.variables['time']
            time = netCDF4.num2date(time[:], time.units)

            # identify the time span
            self.tspan_in = time
            self.years = self.tspan_in.years
            self.days_per_year = [len([date for date in self.tspan_in if self.tspan_in.year == y]) for y in self.years]

            # Read timeseries
            self.temp = nc.variables['temp'][:]
            self.prcp = nc.variables['prcp'][:] * prcp_fac
            self.grad = nc.variables['grad'][:]

            self.ref_hgt = nc.ref_hgt

    def get_daily_mb(self, heights, date=None):

        # index of the date of MB calculation
        ix = netCDF4.date2index(date, self.tspan_in, select='exact')

        # Read timeseries
        itemp = self.temp[ix] + self.temp_bias
        iprcp = self.prcp[ix]
        igrad = self.grad[ix]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        tempformelt = temp - self.t_melt
        tempformelt[:] = np.clip(tempformelt, 0, tempformelt.max())

        # Compute solid precipitation from total precipitation
        prcpsol = np.ones(npix) * iprcp
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol *= np.clip(fac, 0, 1)

        mb_day = prcpsol - self.mu_star * tempformelt - \
                 self.bias
        return mb_day / SEC_IN_DAY / cfg.RHO
