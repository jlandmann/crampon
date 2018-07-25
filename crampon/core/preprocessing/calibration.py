import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import datetime as dt
import glob
import os
import matplotlib.pyplot as plt
import itertools
import scipy.optimize as optimize
import crampon.cfg as cfg
from crampon import workflow
from crampon import tasks
from crampon.core.models.massbalance import BraithwaiteModel

import logging

# Module logger
log = logging.getLogger(__name__)


def get_measured_mb_glamos(gdir, mb_dir=None):
    """
    Gets measured mass balances from GLAMOS as a pd.DataFrame.

    Corrupt and missing data are eliminated, i.e. id numbers:
    0 : not defined / unknown source
    7 : reconstruction from volume change analysis (dV)
    8 : reconstruction from volume change with help of stake data(dV & b_a/b_w)

    Columns "id" (indicator on data base), "date0" (annual MB campaign date at
    begin_mbyear), "date_s" (date of spring campaign), "date1" (annual MB
    campaign date at end), "Winter" (winter MB) and "Annual" (annual MB) are
    kept. File names of the mass balance data must contain the glaciers ID
    (stored in the crampon.GlacierDirectory.id attribute)

    Parameters
    ----------
    gdir: py:class:`crampon.GlacierDirectory`
        A crampon.GlacierDirectory object.
    mb_dir: str, optional
        Path to the directory where the mass balance files are stored. Default:
        None (The path if taken from cfg.PATHS).

    Returns
    -------
    measured: pandas.DataFrame
        The pandas.DataFrame with preprocessed values.
    """

    if mb_dir:
        mb_file = glob.glob(
            os.path.join(mb_dir, '{}*'.format(gdir.rgi_id)))[0]
    mb_file = glob.glob(
        os.path.join(cfg.PATHS['mb_dir'], '{}_*'.format(gdir.rgi_id)))[0]

    # we have varying date formats (e.g. '19440000' for Silvretta)
    def date_parser(d):
        try:
            d = pd.datetime.strptime(str(d), '%Y%m%d')
        except ValueError:
            raise
            #pass
        return d

    # No idea why, but header=0 doesn't work
    # date_parser doesn't work, because of corrupt dates....sigh...
    colnames = ['id', 'date0', 'date_s', 'date1', 'Winter', 'Annual']
    measured = pd.read_csv(mb_file,
                           skiprows=4, sep=' ', skipinitialspace=True,
                           usecols=[0, 1, 3, 4, 5, 6], header=None,
                           names=colnames, dtype={'date_s': str})

    # Skip wrongly constructed MB (and so also some corrupt dates)
    measured = measured[~measured.id.isin([0, 7, 8])]

    # parse dates row by row
    for k, row in measured.iterrows():
        measured.loc[k, 'date0'] = date_parser(measured.loc[k, 'date0'])
        measured.loc[k, 'date1'] = date_parser(measured.loc[k, 'date1'])
        try:
            measured.loc[k, 'date_s'] = date_parser(
                '{}{}{}'.format(measured.loc[k, 'date1'].year,
                                str(row.date_s)[:2], str(row.date_s)[2:4]))
        except ValueError:  # date parsing fails
            measured = measured[measured.index != k]

    # convert mm w.e. to m w.e.
    measured['Annual'] = measured['Annual'] / 1000.
    measured['Winter'] = measured['Winter'] / 1000.

    return measured


def to_minimize_braithwaite_fixedratio(x, gdir, measured, prcp_fac,
                                       winteronly=False, unc=None, y0=None,
                                       y1=None, snow_hist=None, run_hist=None,
                                       ratio_s_i=None):
    """
    Cost function input to scipy.optimize.least_squares to optimize melt
    equation parameters.

    This function is only applicable to glaciers that have a measured
    glaciological mass balance.

    Parameters
    ----------
    x: array-like
        Independent variables. In case a fixed ratio between the melt factor
        for ice (mu_ice) and the melt factor for snow (mu_snow) is given they
        should be mu_ice and the precipitation correction factor (prcp_fac),
        othwerise mu_ice, mu_snow and prcp_fac.
    gdir: py:class:`crampon.GlacierDirectory`
        GlacierDirectory for which the precipitation correction should be
        optimized.
    measured: pandas.DataFrame
    A DataFrame with measured glaciological mass balances.
    y0: float, optional
        Start year of the calibration period. The exact begin_mbyear date is
        taken from the day of the annual balance campaign in the `measured`
        DataFrame. If not given, the date is taken from the minimum date in the
        DataFrame of the measured values.
    y1: float, optional
        Start year of the calibration period. The exact end date is taken
        from the day of the annual balance campaign in the `measured`
        DataFrame. If not given, the date is taken from the maximum date in the
        DataFrame of the measured values.
    snow_init: array-like with shape (heights,)
        Initial snow distribution on the glacier's flowline heights at the
        beginning of the calibration phase.

    Returns
    -------
    err: list
        The residuals (squaring is done by scipy.optimize.least_squares).
    """

    if ratio_s_i:
        mu_ice = x[0]
        mu_snow = mu_ice * ratio_s_i

    # cut measured
    measured_cut = measured[measured.date0.dt.year >= y0]
    measured_cut = measured_cut[measured_cut.date1.dt.year <= y1]

    assert len(measured_cut == 1)

    # make entire MB time series
    min_date = measured[measured.date0.dt.year == y0].date0.values[0]
    if y1:
        max_date = measured[measured.date1.dt.year == y1].date1.values[0] - \
                   pd.Timedelta(days=1)
    else:
        max_date = max(measured.date0.max(), measured.date_s.max(),
                       measured.date1.max() - dt.timedelta(days=1))
    calispan = pd.date_range(min_date, max_date, freq='D')

    heights, widths = gdir.get_inversion_flowline_hw()
    day_model = BraithwaiteModel(gdir, mu_ice=mu_ice, mu_snow=mu_snow,
                                 prcp_fac=prcp_fac, bias=0.)
    # IMPORTANT
    if snow_hist is not None:
        day_model.snow = snow_hist
    if run_hist is not None:
        day_model.time_elapsed = run_hist

    mb = []
    for date in calispan:
        tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
        mb.append(tmp)

    mb_ds = xr.Dataset({'MB': (['time'], mb)},
                       coords={'time': pd.to_datetime(calispan)})

    err = []
    for ind, row in measured_cut.iterrows():
        curr_err = None

        # annual sum
        if not winteronly:
            span = pd.date_range(row.date0, row.date1 - pd.Timedelta(days=1),
                                 freq='D')
            asum = mb_ds.sel(time=span).apply(np.sum)

            if unc:
                curr_err = (row.Annual - asum.MB.values) / unc
            else:
                curr_err = (row.Annual - asum.MB.values)

        err.append(curr_err)
    return err


def to_minimize_braithwaite_fixedratio_wonly(x, gdir, measured, mu_ice,
                                             mu_snow, y0=None, y1=None,
                                             snow_hist=None, run_hist=None):
    """
    Cost function as input to scipy.optimize.least_squares to optimize the
    precipitation correction factor in winter.

    This function is only applicable to glaciers that have a measured
    glaciological mass balance.

    Parameters
    ----------
    x: array-like
        Independent variables. Here: only the precipitation correction factor
        (prcp_fac).
    gdir: py:class:`crapon.GlacierDirectory`
        GlacierDirectory for which the precipitation correction should be
        optimized.
    measured: pandas.DataFrame
    A DataFrame with measured glaciological mass balances.
    mu_ice: float
        The ice melt factor (mm d-1 K-1).
    mu_snow: float
        The snow melt factor (mm d-1 K-1).
    y0: float, optional
        Where to start the mass balance calculation. If not given, the date is
        taken from the minimum date in the DataFrame of the measured values.
    y1: float, optional
        Where to end the mass balance calculation. If not given, the date is
        taken from the maximum date in the DataFrame of the measured values.
    snow_hist: array-like with shape (heights,), optional
        Initial snow distribution on the glacier's flowline heights at the
        beginning of the calibration phase.
    run_hist: pd.DatetimeIndex, optional


    Returns
    -------
    err: list
        The residuals (squaring is done by scipy.optimize.least_squares).
    """
    prcp_fac = x

    # cut measured
    measured_cut = measured[measured.date0.dt.year >= y0]
    measured_cut = measured_cut[measured_cut.date1.dt.year <= y1]

    assert len(measured_cut) == 1

    # make entire MB time series
    min_date = measured[measured.date0.dt.year == y0].date0.values[0]
    if y1:
        max_date = measured[measured.date1.dt.year == y1].date_s.values[0]
    else:
        max_date = max(measured.date0.max(), measured.date_s.max(),
                       measured.date1.max() - dt.timedelta(days=1))
    calispan = pd.date_range(min_date, max_date, freq='D')

    heights, widths = gdir.get_inversion_flowline_hw()
    day_model = BraithwaiteModel(gdir, mu_snow=mu_snow, mu_ice=mu_ice,
                                     prcp_fac=prcp_fac, bias=0.)
    # IMPORTANT
    if snow_hist is not None:
        day_model.snow = snow_hist
    if run_hist is not None:
        day_model.time_elapsed = run_hist

    mb = []
    for date_w in calispan:
        # Get the mass balance and convert to m per day
        tmp = day_model.get_daily_specific_mb(heights, widths, date=date_w)
        mb.append(tmp)

    mb_ds = xr.Dataset({'MB': (['time'], mb)},
                       coords={'time': pd.to_datetime(calispan)})

    err = []
    for ind, row in measured_cut.iterrows():
        curr_err = None

        wspan = pd.date_range(row.date0, row.date_s, freq='D')
        wsum = mb_ds.sel(time=wspan).apply(np.sum)

        curr_err = row.Winter - wsum.MB.values

        err.append(curr_err)

    return err


def calibrate_braithwaite_on_measured_glamos(gdir, ratio_s_i=0.5,
                                             conv_thresh=0.005, it_thresh=50):
    """
    A function to calibrate those glaciers that have a glaciological mass
    balance in GLAMOS.

    Parameters
    ----------
    gdir: py:class:`crampon.GlacierDirectory`
        The GlacierDirectory of the glacier to be calibrated.
    ratio_s_i: float
        The ratio between snow melt factor and ice melt factor. Default: 0.5.
    conv_thresh: float
        Abort criterion for the iterative calibration, defined as the absolute
        gradient of the calibration parameters between two iterations. Default:
         Abort when the absolute gradient is smaller than 0.005.
    it_thresh: float
        Abort criterion for the iterative calibration by the number of
        iterations. This criterion is used when after the given number of
        iteration the convergence threshold hasn't been reached. Default: Abort
        after 50 iterations.

    Returns
    -------
    None
    """

    # Get measured MB and we can't calibrate longer than our meteo history
    measured = get_measured_mb_glamos(gdir)
    if measured.empty:
        log.error('No calibration values left for {}'.format(gdir.rgi_id))
        return

    cmeta = xr.open_dataset(gdir.get_filepath('climate_daily'),
                            drop_variables=['temp', 'prcp', 'hgt', 'grad'])
    measured = measured[(measured.date0 > pd.Timestamp(np.min(cmeta.time).values)) &
                        (measured.date1 < pd.Timestamp(np.max(cmeta.time).values))]

    try:
        cali_df = pd.read_csv(gdir.get_filepath('calibration'), index_col=0,
                              parse_dates=[0])
        # think about an outer join of the date indices here
    except FileNotFoundError:
        # ind1 = ['Braithwaite', 'Braithwaite', 'Braithwaite', 'Hock', 'Hock', 'Hock', 'Hock', 'Pellicciotti', 'Pellicciotti', 'Pellicciotti', 'Oerlemans', 'Oerlemans', 'Oerlemans', 'OGGM', 'OGGM']
        # ind2 = ['mu_snow', 'mu_ice', 'prcp_fac', 'MF', 'r_ice', 'r_snow', 'prcp_fac', 'TF', 'SRF', 'prcp_fac', 'c_0', 'c_1', 'prcp_fac', 'mu_star', 'prcp_fac']
        cali_df = pd.DataFrame(columns=['mu_ice', 'mu_snow', 'prcp_fac',
                                        'mu_star'],
                               index=pd.date_range(measured.date0.min(),
                                                   measured.date1.max()))

    # we don't know initial snow and time of run history
    snow_hist = None
    run_hist = None

    for i, row in measured.iterrows():
        grad = 1
        r_ind = 0

        # Check if we are complete
        if pd.isnull(row.Winter) or pd.isnull(row.Annual):
            log.warning('Mass balance {}/{} not complete. Skipping calibration'
                .format(row.date0.year, row.date1.year))
            cali_df.loc[row.date0:row.date1, 'mu_ice'] = np.nan
            cali_df.loc[row.date0:row.date1, 'mu_snow'] = np.nan
            cali_df.loc[row.date0:row.date1, 'prcp_fac'] = np.nan
            cali_df.to_csv(gdir.get_filepath('calibration'))
            continue

        # say what we are doing
        log.info('Calibrating budget year {}/{}'.format(row.date0.year,
                                                        row.date1.year))

        while grad > conv_thresh:

            # log status
            log.info('{}TH ROUND, grad={}'.format(r_ind, grad))

            # initial guess or params from previous iteration
            if r_ind == 0:
                prcp_fac_guess = 1.5
                mu_ice_guess = 10.
            else:
                mu_ice_guess = spinupres.x[0]

            # log status
            log.info('PARAMETERS:{}, {}'.format(mu_ice_guess, prcp_fac_guess))

            # start with cali on winter MB and optimize only prcp_fac
            spinupres_w = optimize.least_squares(
                to_minimize_braithwaite_fixedratio_wonly,
                x0=np.array([prcp_fac_guess]),
                xtol=0.001,
                bounds=(0.1, 5.),
                verbose=0, args=(gdir, measured, mu_ice_guess,
                                 mu_ice_guess * ratio_s_i),
                kwargs={'y0': row.date0.year, 'y1': row.date1.year,
                        'snow_hist': snow_hist,
                        'run_hist': run_hist})

            # log status
            log.info('After winter cali, prcp_fac:{}'.format(spinupres_w.x[0]))
            prcp_fac_guess = spinupres_w.x[0]

            # take optimized prcp_fac and optimize melt param(s) with annual MB
            spinupres = optimize.least_squares(
                to_minimize_braithwaite_fixedratio,
                x0=np.array([mu_ice_guess]),
                xtol=0.001,
                bounds=(1., 50.),
                verbose=0, args=(gdir, measured, prcp_fac_guess),
                kwargs={'winteronly': False, 'y0': row.date0.year,
                        'y1': row.date1.year, 'snow_hist': snow_hist,
                        'run_hist': run_hist, 'ratio_s_i': ratio_s_i})

            # Check whether abort or go on
            r_ind += 1
            grad = np.abs(mu_ice_guess - spinupres.x[0])
            if r_ind > it_thresh:
                warn_it = 'Iterative calibration reached abort criterion of' \
                          ' {} iterations and was stopped at a parameter ' \
                          'gradient of {}.'.format(r_ind, grad)
                log.warning(warn_it)
                break

        # Report result
        log.info('After whole cali:{}, {}, grad={}'.format(spinupres.x[0],
                                                        prcp_fac_guess, grad))

        # Write in cali df
        cali_df.loc[row.date0:row.date1, 'mu_ice'] = spinupres.x[0]
        cali_df.loc[row.date0:row.date1, 'mu_snow'] = spinupres.x[0] * ratio_s_i
        cali_df.loc[row.date0:row.date1, 'prcp_fac'] = prcp_fac_guess
        cali_df.to_csv(gdir.get_filepath('calibration'))

        # get history for next run
        heights, widths = gdir.get_inversion_flowline_hw()
        curr_model = BraithwaiteModel(gdir, bias=0.)
        mb = []
        for date in pd.date_range(row.date0, row.date1):
            tmp = curr_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)

        error = measured.loc[i].Annual - np.cumsum(mb)[-1]
        log.info('ERROR to measured MB:{}'.format(error))
        snow_hist = curr_model.snow[:-1]  # bcz row.date1 == nextrow.date0
        run_hist = curr_model.time_elapsed[:-1]  # same here


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
    ax.scatter(msrd.date1.values, err[0::2],
               label=" ".join([str(i) for i in x0]))
    ax.scatter(msrd.date1.values, err[1::2],
               label=" ".join([str(i) for i in x0]))


def artificial_snow_init(gdir, aar=0.66, max_swe=1.5):
    """
    Creates an artificial initial snow distribution.

    This initial snow distribution is needed for runs and calibration
    without information from spinup runs or to allow a faster spinup. The
    method to create this initial state is quite simple: the Accumulation

    Parameters
    ----------
    gdir: py:class:`crampon.GlacierDirectory`
        The GlacierDirectory to calculate initial snow conditions for.
    aar: float
        Accumulation Area Ratio. Default: 0.66
    max_swe: float
        Maximum snow water equivalent (m) at the top of the glacier.
        Default: 1.5

    Returns
    -------
    snow_init: array
        An array prescribing a snow distribution with height.
    """
    h, w = gdir.get_inversion_flowline_hw()
    fls = gdir.read_pickle('inversion_flowlines')
    min_hgt = min([min(f.surface_h) for f in fls])
    max_hgt = max([max(f.surface_h) for f in fls])

    ##w=[]
    ##h=[]
    ##for fl in fls:
    ##    widths_all = np.append(w, fl.widths)
    ##    heights_all = np.append(h, fl.surface_h)

    #thrsh_hgt = utils.weighted_quantiles(h, [1 - aar], sample_weight=w)
    #slope = max_swe / (max_hgt - thrsh_hgt)
    #snow_distr = np.clip((h - thrsh_hgt) * slope, 0., None)

    #return snow_distr


def fake_dynamics(gdir, dh_max=-5., da_chg=0.01):
    """
    Apply simple dh (height) and da (area change) to the flowlines.

    Parameters
    ----------
    gdir
    dh_max: max height change at the tongue
    da_chg: area change per year

    Returns
    -------

    """

    fls = gdir.read_pickle('inversion_flowlines')
    min_hgt = min([min(f.surface_h) for f in fls])
    max_hgt = max([max(f.surface_h) for f in fls])

    dh_func = lambda x: dh_max - ((x - min_hgt) / (max_hgt - min_hgt)) * dh_max

    # take everything from the principle fowline


if __name__ == '__main__':

    cfg.initialize(file='C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\'
                        'CH_params.cfg')

    glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
    rgidf = gpd.read_file(glaciers)

    # "the six"
    rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504', 'RGI50-11.A10G05', 'RGI50-11.B5616n-1', 'RGI50-11.A55F03', 'RGI50-11.B4312n-1', 'RGI50-11.C1410'])]
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])]  # Gries OK
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A10G05'])]  # Silvretta OK
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B5616n-1'])]  # Findel OK
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A55F03'])]  # Plaine Morte OK
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4312n-1'])]  # Rhone
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.C1410'])]  # Basòdino "NaN in smoothed DEM"

    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B5616n-0'])]  # Adler     here the error is -1.14 m we afterwards
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A50I19-4'])]  # Tsanfleuron
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A50D01'])]  # Pizol
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A51E12'])]  # St. Anna
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B8315n'])]  # Corbassière takes ages...
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.E2320n'])]  # Corvatsch-S
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A51E08'])]  # Schwarzbach
    #### rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B8214'])]  # Gietro has no values left!!!! (no spring balance)
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B5232'])]  # Hohlaub
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B5229'])]  # Allalin
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B1601'])]  # Sex Rouge
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B5263n'])]  # Schwarzberg
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.E2404'])]  # Murtèl
    # rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A50I07-1'])]  # Plattalva
    # rgidf = rgidf[rgidf.RGIId.isin([''])]  #
    # rgidf = rgidf[rgidf.RGIId.isin([''])]  #
    # rgidf = rgidf[rgidf.RGIId.isin([''])]  #
    # rgidf = rgidf[rgidf.RGIId.isin([''])]  #
    # rgidf = rgidf[rgidf.RGIId.isin([''])]  #

    gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)

    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.process_custom_climate_data,

    ]
    for task in task_list:
        workflow.execute_entity_task(task, gdirs)

    for g in gdirs:
        calibrate_braithwaite_on_measured_glamos(g)

    print('hallo')
