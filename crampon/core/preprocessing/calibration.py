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
from crampon.core.models.massbalance import BraithwaiteModel, \
    run_snowfirnmodel_with_options
from collections import OrderedDict
from crampon import utils
import copy

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
    colnames = ['id', 'date0', 'date_f', 'date_s', 'date1', 'Winter', 'Annual']
    measured = pd.read_csv(mb_file,
                           skiprows=4, sep=' ', skipinitialspace=True,
                           usecols=[0, 1, 2, 3, 4, 5, 6], header=None,
                           names=colnames, dtype={'date_s': str,
                                                  'date_f': str})

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
        except (ValueError, KeyError):  # date parsing fails or has "0000"
            measured = measured[measured.index != k]
        try:
            measured.loc[k, 'date_f'] = date_parser(
                '{}{}{}'.format(measured.loc[k, 'date0'].year,
                                str(row.date_f)[:2], str(row.date_f)[2:4]))
        except (ValueError, KeyError):  # date parsing fails or has "0000"
            measured = measured[measured.index != k]

    # convert mm w.e. to m w.e.
    measured['Annual'] = measured['Annual'] / 1000.
    measured['Winter'] = measured['Winter'] / 1000.

    return measured


def to_minimize_braithwaite_fixedratio(x, gdir, measured, prcp_fac,
                                       winteronly=False, unc=None, y0=None,
                                       y1=None, run_hist=None,
                                       ratio_s_i=None, scov=None):
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
    else:
        mu_ice = x[0]
        mu_snow = x[1]

    # cut measured
    measured_cut = measured[measured.date0.dt.year >= y0]
    measured_cut = measured_cut[measured_cut.date1.dt.year <= y1]

    assert len(measured_cut == 1)

    # make entire MB time series
    # we take the minimum of date0 and the found minimum date from the winter calibration
    # todo: does this agree with the dates in the GLAMOS files? otherwise we are no longer allowed to calculate the error
    min_date = measured[measured.date_f.dt.year == y0].date0.values[0]
    if y1:
        max_date = measured[measured.date1.dt.year == y1].date1.values[0] - \
                   pd.Timedelta(days=1)
    else:
        max_date = max(measured.date_f.max(), measured.date_s.max(),
                       measured.date1.max() - dt.timedelta(days=1))
    calispan = pd.date_range(min_date, max_date, freq='D')

    heights, widths = gdir.get_inversion_flowline_hw()
    day_model = BraithwaiteModel(gdir, mu_ice=mu_ice, mu_snow=mu_snow,
                                 prcp_fac=prcp_fac, bias=0.)
    # IMPORTANT
    if run_hist is not None:
        day_model.time_elapsed = run_hist
    if scov is not None:
        day_model.snowcover = copy.deepcopy(scov)

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
                                             run_hist=None, scov=None):
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
    # todo: this could be outside the function
    measured_cut = measured[measured.date0.dt.year >= y0]
    measured_cut = measured_cut[measured_cut.date1.dt.year <= y1]

    assert len(measured_cut) == 1

    # make entire MB time series
    min_date = measured[measured.date_f.dt.year == y0].date_f.values[0]
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
    if run_hist is not None:
        day_model.time_elapsed = run_hist
    if scov is not None:
        day_model.snowcover = copy.deepcopy(scov)

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
                                             conv_thresh=0.005, it_thresh=50,
                                             filesuffix=''):
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
    filesuffix: str
        Filesuffix for calibration file. Use mainly for experiments. Default:
        no suffx (empty string).

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
        cali_df = pd.read_csv(gdir.get_filepath('calibration',
                                                filesuffix=filesuffix),
                              index_col=0,
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
    run_hist_field = None # at the new field date (date0)
    run_hist_minday = None # at the minimum date as calculated by Matthias
    scov_minday = None
    scov_field = None

    for i, row in measured.iterrows():

        heights, widths = gdir.get_inversion_flowline_hw()

        grad = 1
        r_ind = 0

        # Check if we are complete
        if pd.isnull(row.Winter) or pd.isnull(row.Annual):
            log.warning('Mass balance {}/{} not complete. Skipping calibration'
                .format(row.date0.year, row.date1.year))
            cali_df.loc[row.date0:row.date1, 'mu_ice'] = np.nan
            cali_df.loc[row.date0:row.date1, 'mu_snow'] = np.nan
            cali_df.loc[row.date0:row.date1, 'prcp_fac'] = np.nan
            cali_df.to_csv(gdir.get_filepath('calibration',
                                             filesuffix=filesuffix))
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
                mu_ice_guess = 7.5
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
                verbose=2, args=(gdir, measured, mu_ice_guess,
                                 mu_ice_guess * ratio_s_i),
                kwargs={'y0': row.date0.year, 'y1': row.date1.year,
                        'run_hist': run_hist_minday, 'scov': scov_minday})

            # log status
            log.info('After winter cali, prcp_fac:{}'.format(spinupres_w.x[0]))
            prcp_fac_guess = spinupres_w.x[0]

            # take optimized prcp_fac and optimize melt param(s) with annual MB
            spinupres = optimize.least_squares(
                to_minimize_braithwaite_fixedratio,
                x0=np.array([mu_ice_guess]),
                xtol=0.001,
                bounds=(1., 50.),
                verbose=2, args=(gdir, measured, prcp_fac_guess),
                kwargs={'winteronly': False, 'y0': row.date0.year,
                        'y1': row.date1.year,'ratio_s_i': ratio_s_i,
                        'run_hist': run_hist_field, 'scov': scov_field})

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
        cali_df.to_csv(gdir.get_filepath('calibration', filesuffix=filesuffix))

        curr_model = BraithwaiteModel(gdir, bias=0.)

        if scov_field is not None:
            curr_model.time_elapsed = run_hist_minday
            curr_model.snowcover = copy.deepcopy(scov_field)

        mb = []
        if i < len(measured) - 1:
            end_date = max(row.date1, measured.iloc[i+1].date_f)  # maximum of field and fall date
        else:  # last row
            end_date = row.date1

        forward_time = pd.date_range(row.date0, end_date)
        for date in forward_time:
            tmp = curr_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)

            if date == row.date1:
                run_hist_field = curr_model.time_elapsed[:-1]  # same here
                scov_field = copy.deepcopy(curr_model.snowcover)
            try:
                if date == measured.iloc[i+1].date_f:
                    run_hist_minday = curr_model.time_elapsed[:-1]  # same here
                    scov_minday = copy.deepcopy(curr_model.snowcover)
            except IndexError:
                if date == row.date1:
                    run_hist_minday = curr_model.time_elapsed[:-1]  # same here
                    scov_minday = copy.deepcopy(curr_model.snowcover)

        # todo: this is not correct: we would have to choose the dates between date_f and date1
        error = measured.loc[i].Annual - np.cumsum(mb[:len(forward_time)])[-1]
        log.info('ERROR to measured MB:{}'.format(error))


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


def compile_firn_core_metainfo():
    """
    Compile a pandas Dataframe with metainformation about firn cores available.

    The information contains and index name (a shortened version of the core
    name), an ID, core top height, latitude and longitude of the drilling site,
    drilling date (as exact as possible; for missing months and days JAN-01 is
    assumed), mean accumulation rate and mean accumulation rate uncertainty (if
    available).

    Returns
    -------
    data: pandas.Dataframe
        The info compiled in a dataframe.
    """
    data = pd.read_csv(cfg.PATHS['firncore_dir'] + '\\firncore_meta.csv')

    return data


def make_massbalance_at_firn_core_sites(core_meta, reset=False):
    """
    Makes the mass balance at the firn core drilling sites.

    At the moment, this function uses the BraithwaiteModel, but soon it should
    probably changed to the PellicciottiModel.

    Parameters
    ----------
    core_meta: pd.Dataframe
        A dataframe containing metainformation about the firn cores, explicitly
        an 'id", "height", "lat", "lon"
    reset: bool
        Whether to reset the GlacierDirectory

    Returns
    -------

    """
    if reset:
        for i, row in core_meta.iterrows():
            print(row.id)
            gdir = utils.idealized_gdir(np.array([row.height]),
                                        np.ndarray([int(1.)]), map_dx=1.,
                                        identifier=row.id, name=i,
                                        coords=(row.lat, row.lon), reset=reset)

            tasks.process_custom_climate_data(gdir)

            # Start with any params and then trim on accum rate later
            mb_model = BraithwaiteModel(gdir, mu_ice=10., mu_snow=5.,
                                        prcp_fac=1.0, bias=0.)

            mb = []
            for date in mb_model.tspan_meteo:
                # Get the mass balance and convert to m per day
                tmp = mb_model.get_daily_mb(np.array([row.height]),
                                            date=date) * cfg.SEC_IN_DAY * \
                      cfg.RHO / cfg.RHO_W
                mb.append(tmp)

            mb_ds = xr.Dataset({'MB': (['time', 'n'], np.array(mb))},
                               coords={
                                   'time': pd.to_datetime(
                                       mb_model.time_elapsed),
                                   'n': (['n'], [1])},
                               attrs={'id': gdir.rgi_id,
                                      'name': gdir.name})
            gdir.write_pickle(mb_ds, 'mb_daily')

            new_mb = fit_core_mb_to_mean_acc(gdir, core_meta)

            gdir.write_pickle(new_mb, 'mb_daily_rescaled')


def fit_core_mb_to_mean_acc(gdir, c_df):
    """
    Fit a calculated mass balance at an ice core site to a mean accumulation.

    Parameters
    ----------
    gdir: crampon.GlacierDirectory
        The (idealized) glacier directory for which the mass balance shall be
        adjusted.
    c_df: pandas.Dataframe
        The dataframe with accumulation information about the ice core. Needs
        to contain a "mean_acc" and an "id" column.

    Returns
    -------
    new_mb: xarray.Dataset
        The new, fitted mass balance.
    """
    old_mb = gdir.read_pickle('mb_daily')
    mean_mb = old_mb.apply(np.nanmean)

    mean_acc = c_df.loc[c_df.id == gdir.rgi_id].mean_acc.values[0] / 365.25

    factor = mean_mb.apply(lambda x: x / mean_acc).MB.values
    new_mb = old_mb.apply(lambda x: x / factor)

    return new_mb


def to_minimize_snowfirnmodel(x, gdirs, csites, core_meta, cali_dict,
                              snow_densify='anderson', snow_densify_kwargs={},
                              firn_densify='huss', firn_densify_kwargs={},
                              temp_update='huss', temp_update_kwargs={},
                              merge_layers=0.05):
    """
    Objective function for the firn model calibration.

    Parameters
    ----------
    x: tuple
        Parameters to calibrate. All parameters that should not be calibrated,
        but are required to be different from the standard values have to be
        passed to via the *_kwargs dictionaries.
    gdir: crampon.GlacierDirectory
        The GlacierDirectory containing the mass balance of the firn core.
    snow_densify: str
        Option which snow densification model to calibrate. Possible:
        'anderson' (uses the Anderson 1976 densification equation).
    firn_densify: str
        Option which firn densification model to calibrate. Possible:
        'huss' (uses the Huss 2013/Reeh 2008 densification equation) and
        'barnola' (uses Barnola 1990).
    temp_update: str
        Option which temperature update model to use. Possible: 'exact' (models
        the exact daily temperature penetration thourgh the snowpack - super
        slow!), 'huss' (uses the Huss 2013 temperature update), 'glogem' (Huss
        and Hock 2015) and 'carslaw' (uses Carslaw 1959 - very slow!).
    merge_layers: float
        Minimum layer height (m) to which incoming layers shall be merged.

    Returns
    -------
    errorlist: list
        List with errors for all csites used as input.
    """

    print('Start round', dt.datetime.now())

    errorlist = []
    n_datapoints = []
    core_length = []

    # Todo: now we check if there all all parameters given for each model: give the opportunity to tune only some parameters and maybe log a warning if not all are given
    # Todo: this code is too sensitive and hard-coded. Change this somehow
    # FIRN
    # Huss
    if (firn_densify == 'huss') and ('f_firn' in cali_dict):
        firn_densify_kwargs['f_firn'] = x[list(cali_dict.keys()).index('f_firn')]
    elif (firn_densify == 'barnola') and \
            (all([k in cali_dict for k in
                  ['beta', 'gamma', 'delta', 'epsilon']])):
        list(cali_dict.keys()).index('beta')
        firn_densify_kwargs.update([('beta', x[list(cali_dict.keys()).index('beta')]),
                                    ('gamma', x[list(cali_dict.keys()).index('gamma')]),
                                    ('delta', x[list(cali_dict.keys()).index('delta')]),
                                    ('epsilon', x[list(cali_dict.keys()).index('epsilon')])])
    else:
        raise ValueError('Either firn model parameter values are not complete '
                         'or they are invalid.')

    # SNOW
    if (snow_densify == 'anderson') and (all([k in cali_dict for k in
                                                     [#'eta0', 'etaa', 'etab',
                                                      'snda', 'sndb', 'sndc',
                                                      'rhoc']])):
        snow_densify_kwargs.update([#('eta0', x[list(cali_dict.keys()).index('eta0')]),
                                    #('etaa', x[list(cali_dict.keys()).index('etaa')]),
                                    #('etab', x[list(cali_dict.keys()).index('etab')]),
                                    ('snda', x[list(cali_dict.keys()).index('snda')]),
                                    ('sndb', x[list(cali_dict.keys()).index('sndb')]),
                                    ('sndc', x[list(cali_dict.keys()).index('sndc')]),
                                    ('rhoc', x[list(cali_dict.keys()).index('rhoc')])])
    elif (snow_densify == 'anderson') and ~(all([k in cali_dict for k in
                                                     [#'eta0', 'etaa', 'etab',
                                                      'snda', 'sndb', 'sndc',
                                                      'rhoc']])):
        pass
    else:
        raise ValueError('Either snow model parameter values are not complete '
                         'or they are invalid.')

    # TEMPERATURE
    # todo: we leave calibration of temperature penetration parameters out...change that?

    for i, csite in enumerate(csites):
        # Todo: get rid of "rescaled" to make it more general
        print(gdirs[i].id)
        mb_daily = gdirs[i].read_pickle('mb_daily_rescaled')
        begindate = dt.datetime(1961, 10, 1)

        # try to read dates
        this_core = core_meta[core_meta.id == gdirs[i].id]
        y, m, d = this_core.date.iloc[0].split('-')
        # TODO: What do we actually assume if we have no info?
        try:
            end_date = dt.datetime(int(y), int(m), int(d))
        except ValueError:
            try:
                end_date = dt.datetime(int(y), int(m), 1)
            except ValueError:
                end_date = dt.datetime(int(y), 1, 1)

        print(gdirs[i].id, x)
        cover = run_snowfirnmodel_with_options(gdirs[i], begindate,
                                       end_date,
                                       mb=mb_daily,
                                       snow_densify=snow_densify,
                                       snow_densify_kwargs=snow_densify_kwargs,
                                       firn_densify=firn_densify,
                                       #firn_densify_kwargs={'f_firn': x[0]},
                                       firn_densify_kwargs=firn_densify_kwargs,
                                       temp_update=temp_update,
                                       temp_update_kwargs=temp_update_kwargs,
                                       merge_layers=merge_layers)

        # compare mid of layers in model to mid of layer in firn core
        model_layer_depths = cover.get_accumulated_layer_depths()[0, :]

        # after correction, always lower edges of density intervals are given
        csite_centers = (([0] + csite.depth.tolist()[:-1]) +
                         (csite.depth - ([0] + csite.depth.tolist()[:-1])) /
                         2.).values

        assert np.all(np.diff(model_layer_depths[~np.isnan(model_layer_depths)][::-1]) > 0)
        csite_cut = csite.loc[csite['depth'] <= np.nanmax(model_layer_depths)]

        # todo: remove again
        bins = csite_cut.depth
        digitized = np.digitize(
            model_layer_depths[~np.isnan(model_layer_depths)], bins)
        sh_for_weights = cover.sh[~np.isnan(cover.sh)]
        rho_for_avg = cover.rho[~np.isnan(cover.rho)]

        # todo: the averaging doesn't correct for the edges i fear (it doesn't split up SH into the actual bins)
        # average where sampling rate of model is higher than of core
        coverrho_interp = []
        for b in range(1, len(bins)):
            if np.sum(sh_for_weights[digitized == b]) > 0:
                coverrho_interp.append(np.average(rho_for_avg[digitized == b],
                           weights=sh_for_weights[digitized == b]))
            else:
                coverrho_interp.append(np.nan)

        # where sampling rate of model is lower than of core: interpolate
        csite_cut_centers = (([0] + csite_cut.depth.tolist()[:-1]) +
                             (csite_cut.depth - (
                                         [0] + csite_cut.depth.tolist()[
                                               :-1])) /
                             2.).values
        cri_naninterp = np.interp(
            csite_cut_centers[1:][np.isnan(coverrho_interp)],
            csite_cut.depth[1:][~np.isnan(np.array(coverrho_interp))],
            np.array(coverrho_interp)[~np.isnan(np.array(coverrho_interp))])
        coverrho_interp = np.array(coverrho_interp)
        coverrho_interp[np.isnan(np.array(coverrho_interp))] = cri_naninterp

        error_all = csite_cut[1:].density - coverrho_interp
        # append MSE to errorlist - only one error per core allowed
        error = np.sqrt(np.nanmean(error_all ** 2))
        errorlist.append(error)
        n_datapoints.append(len(csite_cut[1:]))
        core_length.append(np.nanmax(csite_cut.depth) - csite_cut.depth[0])

        plt.figure()
        plt.scatter(csite_cut.depth[1:], coverrho_interp, label='modeled')
        plt.scatter(csite_cut.depth[1:], csite_cut.density[1:], label='measured')
        if 'density_error' in csite.columns:
            plt.errorbar(csite_cut.depth[1:], csite_cut.density[1:],
                         yerr=csite_cut.density_error[1:])
        plt.scatter(csite_cut.depth[1:], error_all, label='error')
        plt.legend()
        plt.title('F_firn: {0:.2f}, RMSE: {1:.2f}, max_rho={2}'
                  .format(x[0], error, int(np.nanmax(coverrho_interp))))
        plt.savefig(
            'c:\\users\\johannes\\desktop\\cali_results\\{}.png'
                .format((gdirs[i].name + '__' + str(x[0])).replace('.', '-')))
        plt.close()

    print('End round', dt.datetime.now())
    print(snow_densify_kwargs, firn_densify_kwargs)

    # weight errors with number of datapoints
    #error_weighted = np.average(errorlist, weights=n_datapoints)
    error_weighted = np.average(errorlist, weights=core_length)
    print('errors for comparison: ', np.nanmean(errorlist), error_weighted)
    return error_weighted


def calibrate_snowfirn_model(cali_dict, core_meta, snow_densify_method,
                             snow_densify_kwargs, firn_densify_method,
                             firn_densify_kwargs, temp_update_method,
                             temp_update_kwargs):
    """
    Calibrate the snow and firn model with its various options.

    Parameters to calibrate are many:
    1) snow models: Anderson 1976: eta0, etaa, etab, snda, sndb, sndc, rhoc
    2) firn models: Huss 2013: f_firn
                    Barnola 1990: beta, gamma, delta, epsilon
    3) temperature models: not considered at the moment

    Parameters
    ----------
    cali_dict: OrderedDict
        Ordered dictionary with parameter names to calibrate as key and initial
        guesses as values.


    Returns
    -------
    None
    """

    # make output xr.Dataset with all possible combos

    # get idealized gdirs for all available core sites (outside functions)

    gdirs = []
    csites = []
    for name, rowdata in core_meta.iterrows():
        path = glob.glob(cfg.PATHS['firncore_dir'] + '\\processed\\{}_density.csv'.format(name))
        data = pd.read_csv(path[0])
        csites.append(data)

        base_dir = cfg.PATHS['working_dir'] + '\\per_glacier'
        print(rowdata.id)
        gdirs.append(utils.GlacierDirectory(rowdata.id, base_dir=base_dir,
                                            reset=False))

    # Parse the settings from the given OrderedDict
    initial_guesses = np.array(list(cali_dict.values()))

    # different calibration values requires different xtols
    if all([i in cali_dict for i in ['eta0', 'etaa', 'etab', 'snda', 'sndb', 'sndc', 'rhoc']]):
        xtol = 1.0e-8  # for values in snow densification after Anderson
    elif all([i in cali_dict for i in ['beta', 'gamma', 'delta', 'epsilon']]):
        xtol = 0.00099
    else:
        xtol = 0.0099  # default for calibrating f_firn

    res = optimize.least_squares(to_minimize_snowfirnmodel, x0=initial_guesses,
                                 xtol=xtol, verbose=2, bounds=(0., np.inf),
                                 args=(gdirs, csites, core_meta,
                                       cali_dict),
                                 kwargs={'snow_densify': snow_densify_method,
                                         'snow_densify_kwargs': snow_densify_kwargs,
                                         'firn_densify': firn_densify_method,
                                         'firn_densify_kwargs': firn_densify_kwargs,
                                         'temp_update': temp_update_method,
                                         'temp_update_kwargs': temp_update_kwargs,
                                         'merge_layers': 0.05})

    return res


def crossvalidate_snowfirnmodel_calibration(to_calibrate, core_meta, snow_densify_method, firn_densify_method, temp_update_method):

    from sklearn.model_selection import LeavePOut
    X = np.arange(len(core_meta))
    lpo = LeavePOut(1)
    lpo.get_n_splits(X)

    print(lpo)

    # todo: this is double code!!!
    gdirs = []
    csites = []
    for name, rowdata in csite_df.iterrows():
        path = glob.glob(cfg.PATHS['firncore_dir'] + '\\processed\\{}_density.csv'.format(name))
        data = pd.read_csv(path[0])
        csites.append(data)

        base_dir = cfg.PATHS['working_dir'] + '\\per_glacier'
        print(rowdata.id)
        gdirs.append(utils.GlacierDirectory(rowdata.id, base_dir=base_dir,
                                            reset=False))

    for train_index, test_index in lpo.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = core_meta.iloc[train_index], core_meta.iloc[test_index]

        print(X_train, type(X_train))

        result = calibrate_snowfirn_model(to_calibrate, X_train,
                                          snow_densify_method,
                                          snow_densify_kwargs,
                                          firn_densify_method,
                                          firn_densify_kwargs,
                                          temp_update_method, temp_kwargs)

        cv_result = to_minimize_snowfirnmodel(result.x,
                                              np.array(gdirs)[test_index],
                                              [csites[i] for i in test_index],
                                              core_meta, to_calibrate,
                                              snow_densify=snow_densify_method,
                              firn_densify=firn_densify_method,
                              temp_update=temp_update_method,
                              merge_layers=0.05)

        print(result.cost, np.nansum(cv_result)**2)


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
