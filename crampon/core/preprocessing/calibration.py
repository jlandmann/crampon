import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import itertools
import scipy.optimize as optimize
import crampon.cfg as cfg
from crampon import workflow
from crampon.core.models.massbalance import BraithwaiteModel

import logging

# Module logger
log = logging.getLogger(__name__)


def to_minimize_spinup_braithwaite(x, gdir):

    mu_snow, mu_ice, prcp_fac = x

    bgmon_hydro = cfg.PARAMS['bgmon_hydro']
    bgday_hydro = cfg.PARAMS['bgday_hydro']

    day_model = BraithwaiteModel(gdir, mu_snow=mu_snow, mu_ice=mu_ice,
                                 prcp_fac=prcp_fac, bias=0.)

    heights, widths = gdir.get_inversion_flowline_hw()

    mb = []
    spinup_trange = day_model.tspan_in

    for date in spinup_trange:
        # Get the mass balance and convert to m per day
        tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
        mb.append(tmp)

    mb_ds = xr.Dataset({'MB': (['time'], mb)},
                       coords={'time': pd.to_datetime(spinup_trange)})

    hydro_years = xr.DataArray([t.year if ((t.month < bgmon_hydro) or
                                           ((t.month == bgmon_hydro) &
                                            (t.day < bgday_hydro)))
                                else (t.year + 1) for t in
                                mb_ds.indexes['time']],
                               dims='time', name='hydro_years',
                               coords={'time': mb_ds.time})

    mb_sum = mb_ds.groupby(hydro_years).apply(
        lambda mb: mb.sum(dim='time', skipna=True))

    print(mb_sum.MB.values, np.mean(mb_sum.MB.values))
    return np.mean(mb_sum.MB.values)


def to_minimize_braithwaite(x, gdir, measured, winteronly=False, unc=None):

    # measured = pd.read_csv(gdir.get_filepath['glacio_method_mb'])
    mu_snow, mu_ice, prcp_fac = x

    print(mu_snow, mu_ice, prcp_fac)

    # make entire MB time series
    # min_date = min(measured.date0.min(), measured.date_s.min(),
    # measured.date1.min())
    min_date = pd.Timestamp('1961-01-01 00:00:00')
    max_date = max(measured.date0.max(), measured.date_s.max(),
                   measured.date1.max())
    maxspan = pd.date_range(min_date, max_date, freq='D')

    day_model = BraithwaiteModel(gdir, mu_snow=mu_snow, mu_ice=mu_ice,
                                 prcp_fac=prcp_fac, bias=0.)

    heights, widths = gdir.get_inversion_flowline_hw()

    mb = []
    for date in maxspan:
        # Get the mass balance and convert to m per day
        tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
        mb.append(tmp)

    mb_ds = xr.Dataset({'MB': (['time'], mb)},
                       coords={'time': pd.to_datetime(maxspan)})

    err = []

    for ind, row in measured.iterrows():

        # annual sum
        if not winteronly:
            # span = pd.date_range(row.date0, row.date1, freq='D')
            # sum = mb_ds.sel(time=span).apply(np.sum)
            span = pd.date_range(measured.date0.min(), row.date1, freq='D')
            asum = mb_ds.sel(time=span).apply(np.sum)

            if unc:
                # err.append((row.Annual - asum.MB.values) / unc)
                err.append((row.mbcumsum - asum.MB.values) / unc)
            else:
                # err.append((row.Annual - asum.MB.values))
                err.append((row.mbcumsum - asum.MB.values))

        wspan = pd.date_range(row.date0, row.date_s, freq='D')
        wsum = mb_ds.sel(time=wspan).apply(np.sum)
        # if ind == measured.index.min():
        #    wspan = pd.date_range(row.date0, row.date_s, freq='D')
        #    wsum = mb_ds.sel(time=wspan).apply(np.sum)
        # else:
        #    wsum_temp = []
        #    for ix in range(measured.index.min(), ind+1):
        #        wspan = pd.date_range(measured.ix[ix].date0,
        #                              measured.ix[ix].date_s, freq='D')
        #        wsum_temp.append(mb_ds.sel(time=wspan))
        #    wsum = xr.concat(*wsum_temp, dim='time')
        #    wsum = wsum.apply(np.sum)

        if unc:
            err.append((row.Winter - wsum.MB.values) / unc)
            # err.append((row.wmbcumsum - wsum.MB.values) / unc)
        else:
            # print("row.Winter", row.Winter, "wsum", wsum, "diff",
            #     (row.Winter - wsum.MB.values))
            err.append((row.Winter - wsum.MB.values))
            # err.append((row.wmbcumsum - wsum.MB.values))

    print(err, len(err))
    return err, mb_ds


if __name__ == '__main__':

    cfg.initialize(file='C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\'
                        'CH_params.cfg')

    glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
    rgidf = gpd.read_file(glaciers)
    rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])]
    g = workflow.init_glacier_regions(rgidf, reset=False, force=False)

    # try to get spinup parameters by requiring the that annual MB equals zero
    spinupres = optimize.least_squares(to_minimize_spinup_braithwaite,
                                       x0=np.array([5., 10., 1.4]),
                                       bounds=((1., 1., 0.1),
                                               (20, 20., 3.)),
                                       verbose=2, args=[g[0]])
    # method='dogbox')  # , loss='soft_l1',
    # f_scale=0.1)

    print(spinupres)
    mb_file = 'c:\\users\\johannes\\documents\\crampon\\data\\MB\\RGI50-11.B4504_gries_obs.dat'
    dp = lambda d: pd.datetime.strptime(str(d), '%Y%m%d')
    # No idea why, but header=0 doesn't work
    colnames = ['id', 'date0', 'date_f', 'date_s', 'date1', 'Winter', 'Annual',
                'ELA', 'AAR', 'Area', 'MinimumElevation', 'MaximumElevation']
    measured = pd.read_csv(mb_file,
                           skiprows=5, sep=' ', skipinitialspace=True,
                           usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                           header=None,
                           names=colnames, parse_dates=[1, 4], date_parser=dp,
                           dtype={'date_s': str})
    for k, row in measured.iterrows():
        measured.loc[k, 'date_s'] = dp(
            '{}{}{}'.format(row.date1.year, str(row.date_s)[:2],
                            str(row.date_s)[2:4]))
    # convert mm w.e. to m w.e.
    measured['Annual'] = measured['Annual'] / 1000.
    measured['Winter'] = measured['Winter'] / 1000.

    # FOR QUICKER TESTING
    measured = measured[measured.id == 4]
    # measured = measured.ix[0]
    # measured = measured.ix[50:55]

    # important: AFTER subselection!:
    measured['mbcumsum'] = np.cumsum(measured['Annual'].values)
    measured['wmbcumsum'] = np.cumsum(measured['Winter'].values)


    def visualize(mb_xrds, msrd, err, x0, ax=None):
        if not ax:
            fig, ax = plt.subplots()
            ax.scatter(msrd.date1.values, msrd.mbcumsum.values)
            ax.hline()
        ax.plot(mb_xrds.sel(
            time=slice(min(msrd.date0), max(msrd.date1))).time,
                np.cumsum(mb_xrds.sel(
                    time=slice(min(msrd.date0), max(msrd.date1))).MB,
                          axis='time'), label=" ".join([str(i) for i in x0]))
        ax.scatter(measured.date1.values, err[0::2],
                   label=" ".join([str(i) for i in x0]))
        ax.scatter(measured.date1.values, err[1::2],
                   label=" ".join([str(i) for i in x0]))


    # by hand
    fig, ax = plt.subplots()
    ax.scatter(measured.date1.values, measured.mbcumsum.values)
    ax.axhline()
    for x0 in itertools.product([6., 7.], [12., 13.], [1.3, 1.4]):
        err, mb_ds = to_minimize_braithwaite(x0, g[0], measured,
                                             winteronly=False)
        visualize(mb_ds, measured, err, x0, ax=ax)
        print(err, mb_ds)

    # res = optimize.least_squares(to_minimize_braithwaite,
    #                             x0=np.array([4., 8., 1.4]),
    #                             bounds=((1., 1., 0.1),
    #                                     (20, 20., 3.)),
    #                             verbose=2, args=(g[0], measured),
    #                             kwargs={'winteronly': False}, method='dogbox')#, loss='soft_l1',
    #                             #f_scale=0.1)
    # (1) winter only w/ melt factor "literature values" to calibrate prcp_fac
    # resw = optimize.least_squares(to_minimize_braithwaite,
    #                             x0=np.array([5., 12., 1.4]),
    #                             bounds=((5., 12., .1), (5.000001, 12.000001, 3.)),
    #                             verbose=2, args=(g[0], measured),
    #                             kwargs={'winteronly': True},loss='soft_l1', f_scale=0.1, method='dogbox')
    ## Now annual and winter to calibrate the melt, take prcp_fac from (1)
    # res = optimize.least_squares(to_minimize_braithwaite,
    #                             x0=np.array([5., 12., resw.x[2]]),
    #                             bounds=((1., 1., resw.x[2]-0.0000001),
    #                                     (20, 20., resw.x[2])),
    #                             verbose=2, args=(g[0], measured),
    #                             kwargs={'winteronly': False},loss='soft_l1', f_scale=0.1, method='dogbox')
    # rranges = (slice(1, 10, 0.5), slice(0.1, 3, 0.5))
    # res = optimize.brute(to_minimize_braithwaite, ranges=rranges, disp=True,
    #                     args=(g))
    #print(res)
