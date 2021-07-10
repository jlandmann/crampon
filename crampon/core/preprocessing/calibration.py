"""
Various calibration functions for the glaciers.
"""

import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
import datetime as dt
import glob
import os
import pickle
import matplotlib.pyplot as plt
import itertools
import scipy.optimize as optimize
import crampon.cfg as cfg
from crampon import workflow
from crampon import tasks, entity_task
from crampon.core.models.massbalance import BraithwaiteModel, \
    run_snowfirnmodel_with_options, PellicciottiModel, HockModel, \
    OerlemansModel, DailyMassBalanceModel, ReveilletModel, GiesenModel, \
    SnowFirnCover
from crampon.core.models import massbalance
from collections import OrderedDict
from crampon import utils
import matplotlib
import copy
import multiprocessing as mp
from crampon.core.preprocessing import radiation
from crampon.core.models import assimilation
from crampon.core.preprocessing import climate
from crampon.core import holfuytools
import logging
import warnings


warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)


class GlamosMassBalanceFile(object):
    """ Interface to GLAMOS mass balance in the files.

    This should sooner or later be the interface both to the mass balances
    retrieved from the database as well as to the file mass balances. Should
    have the function get_measured_mb_glamos as a method.
    """

    def __init__(self, gdir):
        """
        Instantiate.

        Parameters
        ----------
        gdir: `py:class:crampon.GlacierDirectory`
            Glacier directory to get the Glamos MB file for.
        """

        self.quality_flag_dict = {
            0: "Not defined or unknown ",
            1: "Analysis of seasonal stake observations (b_w & b_a)",
            2: "Analysis of annual stake observations but not close to the end"
               " of the period (b_w)",
            3: "Analysis of annual stake observations (b_a)",
            4: "Combined analysis of seasonal stake observations with volume "
               "change (b_w & b_a & dV)",
            5: "Combined analysis of annual stake observations within the "
               "period with volume change (b_w & dV)",
            6: "Combined analysis of annual stake observations with volume "
               "change (b_a & dV)",
            7: "Reconstruction from volume change analysis (dV)",
            8: "Reconstruction from volume change with help of stake data "
               "(dV & b_a/b_w)",
            9: "No measurement, only model results",
        }

        self.bad_flags_default = [0, 7, 8, 9]
        self.spring_max_date_name = 'date_s'
        self.fall_min_date_name = 'date_f'
        self.annual_field_date_name = 'date0'
        self.winter_field_date_name = 'date1'
        self.annual_mb_name = 'Annual'
        self.winter_mb_name = 'Winter'

        self.date_names = [self.spring_max_date_name, self.fall_min_date_name,
                           self.annual_field_date_name,
                           self.winter_field_date_name]
        self.mb_names = [self.annual_mb_name, self.winter_mb_name]


def get_measured_mb_glamos(gdir, mb_dir=None, bw_elev_bands=False):
    """
    Gets measured mass balances from GLAMOS as a pd.DataFrame.

    Corrupt and missing data are eliminated, i.e. id numbers:
    0 : not defined / unknown source
    3 :
    6 : no b_w!!!
    7 : reconstruction from volume change analysis (dV)
    8 : reconstruction from volume change with help of stake data(dV & b_a/b_w)
    9 : No measurement, only model results

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
    bw_elev_bands: bool
        If True, then only the winter mass balance on the elevation bands is
        read. This is a complete rubbish way to do it. Default: false

    Returns
    -------
    measured: pandas.DataFrame
        The pandas.DataFrame with preprocessed values.
    """

    if mb_dir is not None:
        mb_file = glob.glob(
            os.path.join(mb_dir, '{}*'.format(gdir.rgi_id)))[0]
    else:
        mb_file = glob.glob(
            os.path.join(cfg.PATHS['mb_dir'], '{}_*'.format(gdir.rgi_id)))[0]

    def date_parser(d):
        """
        Try to parse the dates.

        We have varying date formats (e.g. '19440000' for Silvretta).

        Parameters
        ----------
        d : str
            Date as string.

        Returns
        -------
        d: pd.Timestamp
            Parsed date (if successful).
        """
        try:
            d = dt.datetime.strptime(str(d), '%Y%m%d')
        except ValueError:
            raise
        return d

    date_colnames = ['id', 'date0', 'date_f', 'date_s', 'date1']
    if bw_elev_bands is False:
        mb_colnames = ['Winter', 'Annual']
        colnames = date_colnames + mb_colnames
        usecols = [0, 1, 2, 3, 4, 5, 6]
    else:
        # get from the header how many columns with elev bands there are, ARGH!
        head = pd.read_csv(
            mb_file, sep=';', skiprows=range(1, 5), skipinitialspace=True,
            header=0, nrows=0, encoding='latin1').columns
        elev_bands = np.linspace(int(head[5]), int(head[6]), int(float(head[4])) + 1)
        elev_bands_mean = (elev_bands[1:] + elev_bands[:-1]) * 0.5
        usecols = np.concatenate([np.arange(5), np.arange(12,
                                                          12 + int(float(head[4])))])
        colnames = date_colnames + list(elev_bands_mean)
        mb_colnames = elev_bands_mean

    # No idea why, but header=0 doesn't work
    # date_parser doesn't work, because of corrupt dates....sigh...
    measured = pd.read_csv(mb_file,
                           skiprows=4, sep=' ', skipinitialspace=True,
                           usecols=usecols, header=None,
                           names=colnames, dtype={'date_s': str,
                                                  'date_f': str,
                                                  'date0': str,
                                                  'date1': str})

    # 'all' because we want to keep WB
    measured = measured.dropna(how='all', subset=mb_colnames)

    # Skip wrongly constructed MB (and so also some corrupt dates)
    measured = measured[~measured.id.isin([0, 7, 8, 9])]

    # parse dates row by row
    for k, row in measured.iterrows():
        try:
            measured.loc[k, 'date0'] = date_parser(measured.loc[k, 'date0'])
            measured.loc[k, 'date1'] = date_parser(measured.loc[k, 'date1'])
        except (ValueError, KeyError):  # date parsing fails or has "0000"
            measured = measured[measured.index != k]
        try:
            measured.loc[k, 'date_s'] = date_parser(
                '{}{}{}'.format(measured.loc[k, 'date1'].year,
                                str(row.date_s)[:2], str(row.date_s)[2:4]))
        except (ValueError, KeyError):  # date parsing fails or has "0000"
            measured = measured[measured.index != k]
        # todo: we actually don't need date_f where only WB is available
        try:
            measured.loc[k, 'date_f'] = date_parser(
                '{}{}{}'.format(measured.loc[k, 'date0'].year,
                                str(row.date_f)[:2], str(row.date_f)[2:4]))
        except (ValueError, KeyError):  # date parsing fails or has "0000"
            measured = measured[measured.index != k]

    # finally, after all the date shenanigans
    measured['date0'] = pd.DatetimeIndex(measured['date0'])
    measured['date_f'] = pd.DatetimeIndex(measured['date_f'])
    measured['date_s'] = pd.DatetimeIndex(measured['date_s'])
    measured['date1'] = pd.DatetimeIndex(measured['date1'])

    # convert mm w.e. to m w.e.
    for c in mb_colnames:
        measured[c] /= 1000.

    return measured


def to_minimize_mass_balance_calibration(x, gdir, mb_model, measured, y0, y1,
                                         *args, winteronly=False, scov=None,
                                         run_hist=None, unc=None,
                                         snow_redist=True,  **kwargs):
    """
    A try to generalize an objective function valid for all MassBalanceModels.

    Parameters
    ----------
    x: tuple
        Parameters to optimize. The parameters must be given exactly in the
        order of the mb_model.cali_params attribute.
    gdir: `py:class:crampon:GlacierDirectory`
        The glacier directory to calibrate.
    mb_model: `crampon.core.models.MassBalanceModel`
        The model to use for calibration
    measured: pandas.DataFrame
        DataFrame with measured glaciological mass balances.
    y0: int
        Start year of the calibration period. The exact begin_mbyear date is
        taken from the day of the annual balance campaign in the `measured`
        DataFrame. If not given, the date is taken from the minimum date in the
        DataFrame of the measured values.
    y1: int
        Start year of the calibration period. The exact end date is taken
        from the day of the annual balance campaign in the `measured`
        DataFrame. If not given, the date is taken from the maximum date in the
        DataFrame of the measured values.
    *args: tuple
    winteronly: bool, optional
        Optimize only the winter mass balance. Default: False.
    scov: crampon.core.models.massbalance.SnowFirnCover or None,
        Snow cover to initiate. Default: None (use initalization from mass
        balance model)
    run_hist: pd.DatetimeIndex or None
        Run history as Time index. Default: None (no history).
    unc: float, optional
        Uncertainty in observed mass balance (m w.e.). Default: None (do not
        account for).
    snow_redist: bool, optional
        Whether to use snow redistribution. Default: True.
    **kwargs: dict
        Keyword arguments accepted by the function.

    Returns
    -------
    err: list
        The residuals (squaring is done by scipy.optimize.least_squares).
    """

    # Here the awkward part comes:
    # Pack x again to an OrderedDict that can be passed to the mb_model
    if winteronly:
        calip_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k in ['prcp_fac']], x))
        other_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k not in ['prcp_fac']], [v for k, v in kwargs.items() if (
                        (k in mb_model.cali_params_guess.keys()) and (
                            k not in ['prcp_fac']))]))
    else:
        calip_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k not in ['prcp_fac']], x))
        other_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k in ['prcp_fac']], [v for k, v in kwargs.items() if k in [
                'prcp_fac']]))  # prcp_fac is the only excluded for annual cali

    params = {**calip_dict, **other_dict}

    measured_cut = measured[measured.date0.dt.year >= y0]
    measured_cut = measured_cut[measured_cut.date1.dt.year <= y1]
    assert len(measured_cut == 1)

    # make entire MB time series
    if winteronly:
        min_date = np.min(
            [measured[measured.date0.dt.year == y0].date0.values[0],
             measured[measured.date_f.dt.year == y0].date_f.values[0]])
        max_date = measured[measured.date1.dt.year == y1].date_s.values[0]
    else:
        # Annual balance will always be between date0 and date1
        min_date = measured[measured.date0.dt.year == y0].date0.values[0]
        max_date = measured[measured.date1.dt.year == y1].date1.values[
                           0] - pd.Timedelta(days=1)

    calispan = pd.date_range(min_date, max_date, freq='D')

    heights, widths = gdir.get_inversion_flowline_hw()
    day_model = mb_model(gdir, **params, bias=0., snow_redist=snow_redist)

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

    # if we have melt in winter between the field dates this pushes prcp_fac
    # up! as we can't measure melt in the winter campaign anyway, we just
    # subtract it away when tuning the winter balance!
    minimum = np.nanmin(np.nancumsum([0.] + mb))
    argmin = np.argmin(np.nancumsum([0.] + mb))

    err = []
    for ind, row in measured_cut.iterrows():

        curr_err = None
        if winteronly:
            wspan = pd.date_range(min_date, row.date_s, freq='D')
            wsum = mb_ds.sel(time=wspan).map(np.sum)

            # correct for melt at the beginning of the winter season
            cs_for_max = np.nancumsum([0.] + mb)
            cs_for_max[:argmin] = minimum
            maximum = np.nanmax(cs_for_max)
            argmax = np.argmax(cs_for_max)
            curr_err = row.Winter - (wsum.MB.values + np.abs(minimum))# + refrozen_since_max)
        else:
            # annual sum
            span = pd.date_range(row.date0, row.date1 - pd.Timedelta(days=1),
                                 freq='D')
            asum = mb_ds.sel(time=span).map(np.sum)

            correction = 0

            if unc:
                curr_err = (row.Annual - (
                            asum.MB.values + np.abs(correction))) / unc
            else:
                curr_err = (row.Annual - (asum.MB.values + np.abs(correction)))

        err.append(curr_err)

    return err


def to_minimize_point_mass_balance(x, gdir, mb_model, measured,
                                   measurement_elevation, date_0, date_1,
                                   scov_date, *args, winteronly=False,
                                   scov=None, unc=None, **kwargs):
    """
    Calibrate on a point mass balance using a pre-calibrated precipitation
    correction factor.

    Parameters
    ----------
    x: tuple
        Parameters to optimize. The parameters must be given exactly in the
        order of the mb_model.cali_params attribute.
    gdir: `py:class:crampon:GlacierDirectory`
        The glacier directory to calibrate.
    mb_model: `crampon.core.models.MassBalanceModel`
        The model to use for calibration
    measured: list
        Measured mass balance in m w.e..
    measurement_elevation: list
        Elevation of the measurement (m).
    date_0: list of pd.Timestamp
        Beginning of the measurement period.
    date_1: list of pd.Timestamp
        End of the measurement period.
    winteronly: bool, optional
        Optimize only the winter mass balance. Default: False.
    scov: crampon.core.models.massbalance.SnowFirnCover or None,
        Snow cover to initiate. Default: None (use initalization from mass
        balance model)
    unc: float, optional
        Uncertainty in observed mass balance (m w.e.). Default: None (do not
        account for).
    scov_date: pd.Timestamp
        Date of snow cover at the beginning. Can differ from the minimum
        observation date (we don't save all the snow covers).
    *args: tuple
    **kwargs: dict
        Keyword arguments accepted by the function.

    Returns
    -------
    err: list
        The residuals (squaring is done by scipy.optimize.least_squares).
    """
    # number of point measurements to calibrate on

    n_points = len(measured)
    min_date = np.min(np.hstack((date_0, scov_date)))
    max_date = np.max(date_1)

    if winteronly:
        calip_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k in ['prcp_fac']], x))
        other_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k not in ['prcp_fac']], [v for k, v in kwargs.items() if (
                    (k in mb_model.cali_params_guess.keys()) and (
                     k not in ['prcp_fac']))]))
    else:
        calip_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k not in ['prcp_fac']], x))
        other_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k in ['prcp_fac']], [v for k, v in kwargs.items() if k in [
                'prcp_fac']]))  # prcp_fac is the only excluded for annual cali

    params = {**calip_dict, **other_dict}
    print(params)

    # calispan = pd.date_range(date_0, date_1, freq='D')
    calispan = pd.date_range(min_date, max_date, freq='D')

    heights_all, widths_all = gdir.get_inversion_flowline_hw()
    # todo: here the cam could end up on the wrong flowline
    obs_ix = np.argmin(np.abs(heights_all -
                              np.atleast_2d(measurement_elevation).T), axis=1)

    day_model = mb_model(gdir, **params, bias=0.)
    # heights_widths=([measurement_elevation], measurement_width))

    # IMPORTANT
    if scov is not None:
        day_model.snowcover = copy.deepcopy(scov)

    mb = []
    conv_fac = ((86400 * cfg.RHO) / cfg.RHO_W)  # m ice s-1 to m w.e. d-1
    for date in calispan:
        # todo: something doesn't work when we pass the measurement
        #  elevation only
        tmp = day_model.get_daily_mb(heights_all, date=date) * conv_fac
        tmp = tmp[obs_ix]
        mb.append(tmp)

    mb_ds = xr.Dataset({'MB': (['time', 'height'], np.array(mb))},
                       # coords={'time': pd.to_datetime(calispan)})
                       coords={'time': (['time', ], calispan),
                               'height': (['height', ],
                                          measurement_elevation)})

    err = []
    for p in range(n_points):
        print(date_0[p], date_1[p])
        mb_sel = mb_ds.sel(time=slice(date_0[p], date_1[p]),
                           height=measurement_elevation[p])
        # if melt in winter between the field dates this pushes prcp_fac
        # up! as we can't measure melt in the winter campaign anyway, we just
        # subtract it away when tuning the winter balance!
        # minimum = np.nanmin(np.nancumsum([0.] + mb))

        # annual sum
        # span = pd.date_range(date_0, date_1 - pd.Timedelta(days=1),
        #                     freq='D')
        span = pd.date_range(date_0[p], date_1[p] - pd.Timedelta(days=1),
                             freq='D')
        mb_sum = mb_sel.sel(time=span).map(np.sum)

        # if measured > 0.:  # there is still snow
        if measured[p] > 0.:  # there is still snow
            # todo: 92 is brute force to catch the rel. min. at the beginning
            minimum = np.nanmin(np.nancumsum([0.] + mb_sel.MB.values[:92]))
            correction = np.abs(minimum)
        else:  # ice
            correction = 0.
        print('ASUM, MEAS, CORR: ', mb_sum.MB.values, measured, correction)

        # err = np.abs((measured - (mb_sum.MB.values + correction)))
        err.append(np.abs((measured[p] - (mb_sum.MB.values + correction))))

    if unc is not None:
        err /= unc

    print("ERRORS: ", err)

    return [e for e in err if ~np.isnan(e)]


def to_minimize_mb_calibration_on_fischer(
        x, gdir, mb_model, mb_annual, min_date, max_date, a_heights, a_widths,
        *args, prcp_corr_only=False, scov=None, run_hist=None, unc=None,
        **kwargs):
    """
    A try to generalize an objective function valid for all MassBalanceModels.

    Parameters
    ----------
    x: tuple
        Parameters to optimize. The parameters must be given exactly in the
        order of the mb_model.cali_params attribute.
    gdir: `py:class:crampon:GlacierDirectory`
        The glacier directory to calibrate.
    mb_model: `crampon.core.models.MassBalanceModel`
        The model to use for calibration
    mb_annual: float
        The mass balance value to calibrate on (m w.e.). This should be an
        annual MB as obtained from the disaggregated geodetic MBs.
    min_date: pd.Datetime, str
        Start date of the calibration phase.
    max_date: pd.Datetime, str
        End date of the calibration phase.
    a_heights: np.array
        Annual heights of the glacier surface.
    a_widths: np.array
        Annual geomtreicla widths fo the glacier.
    *args: tuple
    prcp_corr_only: bool, optional
        Optimize only the precipitation correction factor. Default: False.
    scov: crampon.core.models.massbalance.SnowFirnCover or None,
        Snow cover to initiate. Default: None (use initalization from mass
        balance model)
    run_hist: pd.DatetimeIndex or None
        Run history as Time index. Default: None (no history).
    unc: float, optional
        Uncertainty in observed mass balance (m w.e.). Default: None (do not
        account for).
    **kwargs: dict
        Keyword arguments accepted by the function.

    Returns
    -------
    err: list
        The residuals (squaring is done by scipy.optimize.least_squares).
    """

    # Here the awkward part comes:
    # Pack x again to an OrderedDict that can be passed to the mb_model
    if prcp_corr_only:
        calip_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k in ['prcp_fac']], x))
        other_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k not in ['prcp_fac']], [v for k, v in kwargs.items() if (
                        (k in mb_model.cali_params_guess.keys()) and (
                            k not in ['prcp_fac']))]))
    else:
        calip_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k not in ['prcp_fac']], x))
        other_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k in ['prcp_fac']], [v for k, v in kwargs.items() if k in [
                'prcp_fac']]))  # prcp_fac is the only excluded for annual cali

    params = {**calip_dict, **other_dict}
    calispan = pd.date_range(min_date, max_date, freq='D')

    day_model = mb_model(gdir, **params, bias=0.,
                         heights_widths=(a_heights, a_widths))

    # IMPORTANT
    if run_hist is not None:
        day_model.time_elapsed = run_hist
    if scov is not None:
        day_model.snowcover = copy.deepcopy(scov)

    mb = []
    for date in calispan:
        tmp = day_model.get_daily_specific_mb(a_heights, a_widths, date=date)
        mb.append(tmp)
    mb = np.array(mb)

    # if prcp_fac is varied, acc_sum changes, otherwise the other way around
    acc_sum = np.sum(mb[mb > 0.])
    abl_sum = np.sum(mb[mb < 0.])

    if unc:
        err = (acc_sum + abl_sum - mb_annual) / unc
    else:
        err = (acc_sum + abl_sum - mb_annual)

    return err


def abort_criterion_by_error_tolerance(error, value, thresh=0.01):
    """
    If an error is smaller than a percentage of a value, make it zero.

    This is really just a pragmatic helper function.

    Parameters
    ----------
    error: float
        Error from some optimization.
    value: float
        Value that the error refers to.
    thresh: float
        Ratio of value below which error should become zero.

    Returns
    -------
    float
        Zero if error is smaller than percentage of the given value.
    """
    if np.abs(error) <= np.abs(thresh * value):
        return 0.
    else:
        return error


def _make_hybrid_mb_or_not(gdir, actual_mb_model, to_calibrate_span, a_heights,
                           a_widths, hw_years, early_mb_model=HockModel,
                           run_hist=None, scov=None, **params):
    """
    Create a 'hybrid' mass balance time series, using two different models.

    This is a special function used in calibration on geodetic volume changes,
    when the calibration time span of a radiation-based model does not reach
    back to the acquisition time of the first DEM.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        GlacierDirectory to get the hybrid mass balance for.
    actual_mb_model:
        The 'actual' mass balance that should be used for calibration, e.g.
        `py:class:crampon.core.models.massbalance.PellicciottiModel`.
    to_calibrate_span: pandas.date_range
        The date range from the beginning of the calibration time span till the
        last day.
    a_heights:
        Annual flowline heights in that period.
    a_widths:
        Annual flowlines widths in that period.
    hw_years:
        Years for which annual heights and widths are valid.
    early_mb_model: `py:class:crampon.core.models.massbalance.MassBalanceModel`
        The mass balance model substitute to be used in the first period, when
        the actual mass balance model is not yet available.
    run_hist :
    scov :
    params :

    Returns
    -------

    """
    # check if we need "hybrid calibration" (non radiation dep. model in the
    # time when there is no radiation yet)
    needs_early_model = False
    if hasattr(actual_mb_model, 'calibration_timespan'):
        model_cali_begin = actual_mb_model.calibration_timespan[0]
    else:
        model_cali_begin = pd.Timestamp(to_calibrate_span[0]).year
    radiation_models = ['PellicciottiModel', 'OerlemansModel', 'GiesenModel']
    early_day_model = None
    if (pd.Timestamp(to_calibrate_span[0]).year < model_cali_begin) and (
            actual_mb_model.__name__ in radiation_models):
        needs_early_model = True
        hindex_init = \
            hw_years.index(to_calibrate_span[0].year if
                           to_calibrate_span[0].month < cfg.PARAMS[
                'bgmon_hydro'] else to_calibrate_span[0].year + 1)

        # try with calibrated geod. parameters
        try:
            early_params = gdir.get_calibration(
                early_mb_model.__name__, filesuffix='_fischer_unique')
            early_params.columns = [c.split(early_mb_model.__name___+'_')[1]
                                    for c in early_params.columns]
            early_params = OrderedDict(early_params.mean().to_dict())
        except:
            early_params = early_mb_model.cali_params_guess
        early_day_model = early_mb_model(
            gdir, bias=0., heights_widths=(a_heights[hindex_init],
                                           a_widths[hindex_init]),
            **early_params)
        if run_hist is not None:
            early_day_model.time_elapsed = run_hist
        if scov is not None:
            early_day_model.snowcover = copy.deepcopy(scov)

    # otherwise just start with the actual model and set snowcov and hist
    day_model = actual_mb_model(
        gdir, **params, bias=0., heights_widths=(a_heights[0], a_widths[0]))
    if early_day_model is None:
        # IMPORTANT
        if run_hist is not None:
            day_model.time_elapsed = run_hist
        if scov is not None:
            day_model.snowcover = copy.deepcopy(scov)

    mb = []
    hw_index_old = 0
    first_time_actual_model = True
    for date in to_calibrate_span:
        hw_index = hw_years.index(date.year if date.month < cfg.PARAMS[
            'bgmon_hydro'] else date.year + 1)

        # use early model or not, depending on where we are in time
        if (needs_early_model is True) and date.year < model_cali_begin:
            # just check if we have the correct snow cover
            if hw_index_old != hw_index:
                early_day_model.snowcover.remove_height_nodes(
                    np.arange(len(a_widths[hw_index]),
                              early_day_model.snowcover.swe.shape[0]))
                hw_index_old = hw_index
            tmp = early_day_model.get_daily_specific_mb(a_heights[hw_index],
                                                        a_widths[hw_index],
                                                        date=date)
        else:
            if (first_time_actual_model is True) and \
                    (needs_early_model is True):
                first_time_actual_model = False
                # IMPORTANT
                day_model.time_elapsed = early_day_model.time_elapsed
                day_model.snowcover = copy.deepcopy(early_day_model.snowcover)
            # just check if we have the correct snow cover
            if hw_index_old != hw_index:
                day_model.snowcover.remove_height_nodes(
                    np.arange(len(a_widths[hw_index]),
                              day_model.snowcover.swe.shape[0]))
                hw_index_old = hw_index
            tmp = day_model.get_daily_specific_mb(a_heights[hw_index],
                                                  a_widths[hw_index],
                                                  date=date)

        mb.append(tmp)
    mb = np.array(mb)
    return mb


def to_minimize_mb_calibration_on_fischer_one_set(
        x, gdir, mb_model, mb_total, min_date, max_date, a_heights,
        a_widths, hw_years, *args, prcp_corr_only=False, scov=None,
        run_hist=None, unc=None, early_model=HockModel, **kwargs):
    """
    Takes arrays of heights and widths for the single calibration years.

    # todo: should be unnecessary as soon as dynamics are handled internally.

    Parameters
    ----------
    x: tuple
        Parameters to optimize. The parameters must be given exactly in the
        order of the mb_model.cali_params attribute.
    gdir: `py:class:crampon:GlacierDirectory`
        The glacier directory to calibrate.
    mb_model: `crampon.core.models.MassBalanceModel`
        The model to use for calibration
    mb_total: float
        The mass balance value to calibrate on (m w.e.). This should be the
        total MB as obtained from the geodetic mass balances in the dataset.
    min_date: pd.Datetime, str
        Start date of the calibration phase.
    max_date: pd.Datetime, str
        End date of the calibration phase.
    a_heights: array
        Annual heights of the flowlines.
    a_widths: array
        Annual widths of the flowlines.
    hw_years:
    *args:
    prcp_corr_only: bool
        Whether or not only the precipitation correction factor shall be
        calibrated. Default: False (calibrate all parameters).
    scov: SnowFirnCover or None
        Snow cover to start the calibration phase with or None (let the model
        try to find it). Default: None.
    run_hist:
    unc:
    early_model: DailyMassBalanceModel
        The mass balance model to use when the calibration period reaches
        further back in time than the model can be calibrated. This is relevant
        for the radiation-dependent models that can only be calibrated back
        till 1984, but for geodetic period may begin earlier. This model is
        used to bridge this early period so that a 'hybrid calibration' of the
        radiation-dependent model is still possible. Default: HockModel.
    **kwargs: dict
        Keyword arguments accepted by the function.

    Returns
    -------
    err: list
        The residuals (squaring is done by scipy.optimize.least_squares).
    """

    # Here the awkward part comes:
    # Pack x again to an OrderedDict that can be passed to the mb_model
    if prcp_corr_only:
        calip_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k in ['prcp_fac']], x))
        other_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k not in ['prcp_fac']], [v for k, v in kwargs.items() if (
                        (k in mb_model.cali_params_guess.keys()) and (
                            k not in ['prcp_fac']))]))
    else:
        calip_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k not in ['prcp_fac']], x))
        other_dict = dict(zip(
            [k for k, v in mb_model.cali_params_guess.items() if
             k in ['prcp_fac']], [v for k, v in kwargs.items() if k in [
                'prcp_fac']]))  # prcp_fac is the only excluded for annual cali

    params = {**calip_dict, **other_dict}
    print(params)
    calispan = pd.date_range(min_date, max_date, freq='D')

    # check if we need "hybrid calibration" (non radiation dep. model in the
    # time when there is no radiation yet)
    needs_early_model = False
    if hasattr(mb_model, 'calibration_timespan'):
        model_cali_begin = mb_model.calibration_timespan[0]
    else:
        model_cali_begin = pd.Timestamp(min_date).year
    radiation_models = ['PellicciottiModel', 'OerlemansModel', 'GiesenModel']
    early_day_model = None
    if (pd.Timestamp(min_date).year < model_cali_begin) and (
            mb_model.__name__ in radiation_models):
        # try it with HockModel
        needs_early_model = True
        hindex_init = hw_years.index(
            calispan[0].year if calispan[0].month < cfg.PARAMS[
                'bgmon_hydro'] else calispan[0].year + 1)
        # try HockModel with guess parameters
        try:
            hock_params = gdir.get_calibration('HockModel',
                                               filesuffix='_fischer_unique')
            hock_params.columns = [c.split('HockModel_')[1] for c in
                                   hock_params.columns]
            hock_params = OrderedDict(hock_params.mean().to_dict())
        except:
            hock_params = HockModel.cali_params_guess
        early_day_model = HockModel(
            gdir, bias=0., heights_widths=(a_heights[hindex_init],
                                           a_widths[hindex_init]),
            **HockModel.cali_params_guess)
        if run_hist is not None:
            early_day_model.time_elapsed = run_hist
        if scov is not None:
            early_day_model.snowcover = copy.deepcopy(scov)

    # otherwise just start with the actual model and set snowcov and hist
    day_model = mb_model(gdir, **params, bias=0.,
                         heights_widths=(a_heights[0], a_widths[0]))
    if early_day_model is None:
        # IMPORTANT
        if run_hist is not None:
            day_model.time_elapsed = run_hist
        if scov is not None:
            day_model.snowcover = copy.deepcopy(scov)

    mb = []
    hw_index_old = 0
    first_time_actual_model = True
    for date in calispan:
        hw_index = hw_years.index(date.year if date.month < cfg.PARAMS[
            'bgmon_hydro'] else date.year + 1)

        # use early model or not, depending on where we are in time
        if (needs_early_model is True) and date.year < model_cali_begin:
            # just check if we have the correct snow cover
            if hw_index_old != hw_index:
                early_day_model.snowcover.remove_height_nodes(
                    np.arange(len(a_widths[hw_index]),
                              early_day_model.snowcover.swe.shape[0]))
                hw_index_old = hw_index
            tmp = early_day_model.get_daily_specific_mb(a_heights[hw_index],
                                                        a_widths[hw_index],
                                                        date=date)
        else:
            if (first_time_actual_model is True) and \
                    (needs_early_model is True):
                first_time_actual_model = False
                # IMPORTANT
                day_model.time_elapsed = early_day_model.time_elapsed
                day_model.snowcover = copy.deepcopy(early_day_model.snowcover)
            # just check if we have the correct snow cover
            if hw_index_old != hw_index:
                day_model.snowcover.remove_height_nodes(
                    np.arange(len(a_widths[hw_index]),
                              day_model.snowcover.swe.shape[0]))
                hw_index_old = hw_index
            tmp = day_model.get_daily_specific_mb(a_heights[hw_index],
                                                  a_widths[hw_index],
                                                  date=date)

        mb.append(tmp)
    mb = np.array(mb)

    # if prcp_fac is varied, acc_sum changes, otherwise the other way around
    total_sum = np.sum(mb)

    if unc:
        err = (total_sum - mb_total) / unc
    else:
        err = (total_sum - mb_total)

    thresholded_error = abort_criterion_by_error_tolerance(err, mb_total)

    return thresholded_error


@entity_task(log, writes=['calibration'])
def calibrate_mb_model_on_measured_glamos(gdir, mb_model, conv_thresh=0.005,
                                          it_thresh=50, cali_suffix='',
                                          years=None, snow_redist=True,
                                          reset_snow_redist=True,
                                          **kwargs):
    """
    A function to calibrate those glaciers that have a glaciological mass
    balance in GLAMOS.

    Parameters
    ----------
    gdir: py:class:`crampon.GlacierDirectory`
        The GlacierDirectory of the glacier to be calibrated.
    mb_model: py:class:`crampon.core.massbalance.MassBalanceModel`
        The mass balance model to calibrate parameters for.
    conv_thresh: float
        Abort criterion for the iterative calibration, defined as the absolute
        gradient of the calibration parameters between two iterations. Default:
         Abort when the absolute gradient is smaller than 0.005.
    it_thresh: float
        Abort criterion for the iterative calibration by the number of
        iterations. This criterion is used when after the given number of
        iteration the convergence threshold hasn't been reached. Default: Abort
        after 50 iterations.
    cali_suffix: str
        Suffix to apply to the calibration file name (read & write).
    years: np.array or None
        Which years to calibrate only (in budget years, i.e. "2021" would mean
        the season 2020/2021. Default: None (calibrate everything which is
        there).
    snow_redist: bool, optional
        whether to use snow redistribution. Default: True.
    reset_snow_redist: bool
        Reset the snow redistribution factor. This means that before the
        calibration starts the snow redistrubition factor for a given model
        will be deleted from the glacier directory. Then the glacier will be
        calibrated, snow_redist will be written and the glacier will be
        calibrated again, this time without resetting. Default: True.
    **kwargs: dict
        Keyword arguments accepted by the mass balance model.

    Returns
    -------
    None
    """
    # todo: find a good solution for ratio_s_i
    # todo: do not always hard-code 'mb_model.__name__ + '_' +'
    # todo go through generalized objective function and see if winter/annual mb works ok


    # Get measured MB and we can't calibrate longer than our meteo history
    measured = get_measured_mb_glamos(gdir)
    if measured.empty:
        log.error('No calibration values left for {}'.format(gdir.rgi_id))
        return

    # we might want to calibrate a selection only
    if years is not None:
        # "or", because it could be the last year with winter MB only
        measured = measured.iloc[
            np.where(measured.date0.dt.year.isin(years-1)) or
            np.where(measured.date1.dt.year.values == (years))]

    # we also have to check if we have the climate data to calibrate at all
    cmeta = xr.open_dataset(gdir.get_filepath('climate_daily'),
                            drop_variables=['temp', 'prcp', 'hgt', 'grad'])
    # date_f for when WB is available
    measured = measured[
        (measured.date0 > pd.Timestamp(np.min(cmeta.time).values)) &
        (measured.date_f < pd.Timestamp(np.max(cmeta.time).values))]


    # mainly PellicciottiModel due to limited radiation data availability
    if hasattr(mb_model, 'calibration_timespan'):
        if mb_model.calibration_timespan[0]:
            measured = measured[
                measured.date0.dt.year >= mb_model.calibration_timespan[0]]
        if mb_model.calibration_timespan[1]:
            # date_f for when WB is available
            measured = measured[
                measured.date_f.dt.year < mb_model.calibration_timespan[1]]
        # in case there is nothing left (e.g. Plattalva)
        if measured.empty:
            log.error('No calibration values left for {}'.format(gdir.rgi_id))
            return

    # very important: reset index to exclude index gaps
    measured.reset_index(drop=True, inplace=True)

    # Find out what we will calibrate
    to_calibrate_csv = [mb_model.prefix + i for i in
                        mb_model.cali_params_guess.keys()]

    # Is there already a calibration where we just can append, or new file
    try:
        cali_df = gdir.get_calibration(filesuffix=cali_suffix)
        # we need to extend it potentially, otherwise we can't write new years
        meas_maxdate = max(measured.date1.max(), measured.date_s.max())
        if cali_df.index[-1] < meas_maxdate:
            new_ix = pd.DatetimeIndex(pd.date_range(cali_df.index[0],
                                      end=meas_maxdate), freq='D')
            cali_df = cali_df.reindex(new_ix)
    # think about an outer join of the date indices here
    except FileNotFoundError:
        try:
            cali_df = pd.DataFrame(
                columns=to_calibrate_csv + ['mu_star', 'prcp_fac'],  # 4 OGGM
                index=pd.date_range(measured.date0.min(),
                                    measured.date1.max()))
            # write for first time
            cali_df.to_csv(
                gdir.get_filepath('calibration', filesuffix=cali_suffix))
        except ValueError:  # valid time for MB model is outside measured data
            return

    # we don't know initial snow and time of run history
    run_hist_field = None  # at the new field date (date0)
    run_hist_minday = None  # at the minimum date as calculated by Matthias
    scov_minday = None
    scov_field = None

    for i, row in measured.iterrows():

        heights, widths = gdir.get_inversion_flowline_hw()

        grad = 1
        r_ind = 0

        # Check if we are complete
        if pd.isnull(row.Winter) and pd.isnull(row.Annual):
            log.warning(
                'Mass balance {}/{} not complete. Skipping calibration'.format(
                    row.date0.year, row.date1.year))
            for name in to_calibrate_csv:
                cali_df.loc[row.date0:row.date1, name] = np.nan
            cali_df.to_csv(gdir.get_filepath('calibration',
                                             filesuffix=cali_suffix))
            continue

        # say what we are doing
        log.info('Calibrating budget year {}/{}'.format(row.date0.year,
                                                        row.date1.year))

        # initial_guess: if here is no annual MB, we guess mean params for the
        # winter melt happening
        if pd.isnull(row.Annual):
            # todo: switch this to ParameterGenerator
            param_prod = utils.get_possible_parameters_from_past(
                gdir, mb_model, as_list=True, latest_climate=True,
                only_pairs=True, constrain_with_bw_prcp_fac=False)
            param_dict = dict(zip(mb_model.cali_params_list,
                                  np.nanmedian(param_prod, axis=0)))
        else:
            param_dict = mb_model.cali_params_guess.copy()

        while grad > conv_thresh:

            # log status
            log.info('{}TH ROUND, grad={}, PARAMETERS: {}'
                     .format(r_ind, grad, param_dict.__repr__()))

            # get an odict with all but prcp_fac
            all_but_prcp_fac = [(k, v) for k, v in param_dict.items()
                                if k not in ['prcp_fac']]

            # start with cali on winter MB and optimize only prcp_fac
            spinupres_w = optimize.least_squares(
                to_minimize_mass_balance_calibration,
                x0=np.array([param_dict['prcp_fac']]),
                xtol=0.0001,
                bounds=[i[-1] for i in param_bounds[mb_model.__name__]],
                verbose=2, args=(gdir, mb_model, measured, row.date0.year,
                                 row.date1.year),
                kwargs={'run_hist': run_hist_minday, 'scov': scov_minday,
                        'winteronly': True, 'snow_redist': snow_redist,
                        **OrderedDict(all_but_prcp_fac)})

            # log status
            log.info('After winter cali, prcp_fac:{}'.format(spinupres_w.x[0]))
            param_dict['prcp_fac'] = spinupres_w.x[0]

            # if there is an annual balance:
            # take optimized prcp_fac and optimize melt param(s) with annual MB
            # otherwise: skip and do next round of prcp_fac optimization
            # todo: find a good solution to give bounds to the values?
            if not pd.isnull(row.Annual):
                spinupres = optimize.least_squares(
                    to_minimize_mass_balance_calibration,
                    x0=np.array([j for i, j in all_but_prcp_fac]),
                    xtol=0.0001,
                    bounds=[i[:-1] for i in param_bounds[mb_model.__name__]],
                    verbose=2, args=(gdir, mb_model, measured, row.date0.year,
                                     row.date1.year),
                    kwargs={'winteronly': False, 'run_hist': run_hist_field,
                            'scov': scov_field,
                            'prcp_fac': param_dict['prcp_fac'],
                            'snow_redist': snow_redist})

                # update all but prcp_fac
                for j, d_i in enumerate(all_but_prcp_fac):
                    param_dict[d_i[0]] = spinupres.x[j]

                # Check whether abort or go on
                grad_test_param = all_but_prcp_fac[0][
                    1]  # take 1st, for example
                grad = np.abs(grad_test_param - spinupres.x[0])
            else:
                # only WB is given and prcp_fac is optimized, so we can set all
                # others to NaN (so that we don't enter standard params into
                # the CSV) and break
                for j, d_i in enumerate(all_but_prcp_fac):
                    param_dict[d_i[0]] = np.nan
                break

            r_ind += 1
            if r_ind > it_thresh:
                warn_it = 'Iterative calibration reached abort criterion of' \
                          ' {} iterations and was stopped at a parameter ' \
                          'gradient of {} for {}.'.format(r_ind, grad,
                                                          grad_test_param)
                log.warning(warn_it)
                break

        # Report result
        log.info('After whole cali:{}, grad={}'.format(param_dict.__repr__(),
                                                       grad))

        # todo: do not hard-code this (pass start/end also to minimization routine)
        # determine start_date
        start_date = row.date0  # min(row.date0, row.date_f)

        # determine end_date
        if i < max(measured.index):
            # max(field & fall date)
            end_date = row.date1 - pd.Timedelta(days=1)
        else:  # last row
            if not np.isnan(row.Annual):
                end_date = row.date1 - pd.Timedelta(days=1)
            else:  # onl WB in last row
                end_date = row.date_s - pd.Timedelta(days=1)

        # this is usually the case, but not when matthias' files are wrong (see below)
        forward_end_date = end_date

        # Write in cali df
        for k, v in list(param_dict.items()):
            cali_df.loc[start_date:end_date, mb_model.prefix + k] = v
        if isinstance(mb_model, massbalance.BraithwaiteModel):
            cali_df.loc[start_date:end_date, mb_model.prefix + 'mu_snow'] = \
                cali_df.loc[start_date:end_date, mb_model.prefix + 'mu_ice'] \
                * cfg.PARAMS['ratio_mu_snow_ice']
        if isinstance(mb_model, massbalance.HockModel):
            cali_df.loc[start_date:end_date, mb_model.prefix + 'a_snow'] = \
                cali_df.loc[start_date:end_date, mb_model.prefix + 'a_ice'] \
                * cfg.PARAMS['ratio_a_snow_ice']
        cali_df.to_csv(gdir.get_filepath('calibration',
                                         filesuffix=cali_suffix))

        # prepare history for next round
        curr_model = mb_model(gdir, bias=0., snow_redist=snow_redist)

        mb = []
        # history depends on which start date we choose
        if scov_field is not None:
            if start_date == row.date0:
                curr_model.time_elapsed = run_hist_field
                curr_model.snowcover = copy.deepcopy(scov_field)
            elif start_date == row.date_f:
                curr_model.time_elapsed = run_hist_minday
                curr_model.snowcover = copy.deepcopy(scov_minday)
            else:
                raise ValueError('Start date is wrong')

        # todo: if we start on the same days, this is obsolete!
        # prep. for next annual cali (start at row.date1 == nextrow.date0)
        try:
            next_start_date = measured.loc[i + 1].date0
            next_start_date_winter = min(measured.loc[i + 1].date0,
                                         measured.loc[i + 1].date_f)
        except KeyError:  # last row
            # todo: does tht make sense when we just want to calirate on b_w?
            next_start_date = row.date1
            next_start_date_winter = row.date1

        # todo: Matthias' files can be wrong: when will he correct them?
        if row.date1 < next_start_date:
            log.warning('date1 < date_0 (+1): there is a gap of {} days in the'
                        ' snow cover handed over')
            forward_end_date = max([end_date, next_start_date])

        forward_time = pd.date_range(start_date, forward_end_date)
        mb_comp = 0.
        for date in forward_time:
            tmp = curr_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)
            if start_date <= date <= end_date:
                mb_comp += tmp

            if date == next_start_date - pd.Timedelta(days=1):
                run_hist_field = curr_model.time_elapsed  # same here
                scov_field = copy.deepcopy(curr_model.snowcover)
            # prepare for next winter calibration (starting at nextrow.date_f)
            try:
                if date == next_start_date_winter - pd.Timedelta(days=1):
                    run_hist_minday = curr_model.time_elapsed  # same here
                    scov_minday = copy.deepcopy(curr_model.snowcover)
            except (IndexError, KeyError):  # last row
                if date == row.date1 - pd.Timedelta(days=1):
                    run_hist_minday = curr_model.time_elapsed  # same here
                    scov_minday = copy.deepcopy(curr_model.snowcover)

        error = row.Annual - mb_comp
        log.info('ERROR to measured MB:{}'.format(error))



def calculate_snow_dist_factor(gdir, mb_models=None, reset=True):
    """
    Calculate 1D snow redistribution factor from elevation band mass balances.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to calculate the redistribution factor for.
    mb_models: list of MassBalanceModel or None
        The mass balance models for which to calculate the redistribution
        factor. Default: None.
    reset: bool
        Whether the 'snow_redist' file shall be deleted and recalculated from
        scratch. Default: False (do not delete).

    Returns
    -------
    None
    """

    if reset is True:
        d_path = gdir.get_filepath('snow_redist')
        if os.path.exists(d_path):
            os.remove(d_path)

    measured_bw_distr = get_measured_mb_glamos(gdir, bw_elev_bands=True)
    if mb_models is None:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]
    heights, widths = gdir.get_inversion_flowline_hw()
    conv_fac = ((86400 * cfg.RHO) / cfg.RHO_W)  # m ice s-1 to m w.e. d-1

    # this is copied code from cali on glamos
    cmeta = xr.open_dataset(gdir.get_filepath('climate_daily'),
                            drop_variables=['temp', 'prcp', 'hgt', 'grad'])
    measured_bw_distr = measured_bw_distr[
        (measured_bw_distr.date0 >= pd.Timestamp(np.min(cmeta.time).values)) &
        (measured_bw_distr.date_f <= pd.Timestamp(np.max(cmeta.time).values))]

    ds_list = []
    for mbm, (i, row) in itertools.product(mb_models,
                                           measured_bw_distr.iterrows()):
        # get accumulation in period; snow init doesnt matter
        drange = pd.date_range(row.date_f, row.date_s)
        try:
            day_model = mbm(gdir, bias=0., snow_redist=False)
        except KeyError:  # no calibration for this MB models in cali file
            print('Glacier not calibrated for {}. Skipping...'
                  .format(mbm.__name__))
            continue

        if hasattr(mbm, 'calibration_timespan'):
            if (mbm.calibration_timespan[0] is not None) and (
                    row.date0.year < mbm.calibration_timespan[0]):
                continue
            if (mbm.calibration_timespan[1] is not None) and (
                    row.date_f.year > mbm.calibration_timespan[1]):
                continue

        accum = []
        ablat = []
        total = []
        gmet = climate.GlacierMeteo(gdir)
        for date in drange:
            # get MB an convert to m w.e. per day
            tmp = day_model.get_daily_mb(heights, date=date)
            tmp *= conv_fac
            # todo: get accum from GlacierMeteo, not from modeling...
            #  otherwise "D" gets smaller the more often it is calculated
            #  (precip is already multiplied with D in the MB classes)
            psol, _ = gmet.get_precipitation_solid_liquid(date, heights)
            try:
                iprcp_fac = day_model.prcp_fac[
                    day_model.prcp_fac.index.get_loc(date,)]
            except KeyError:
                iprcp_fac = day_model.param_ep_func(day_model.prcp_fac)
            acc = psol * iprcp_fac
            # accum.append(np.clip(tmp, 0., None))
            accum.append(acc/1000.)  # convert to m
            ablat.append(np.clip(tmp, None, 0.))
            total.append(tmp)

        # get minimum index - after then we assume melt to be frozen and thus
        # accounted for in the GLAMOS measurements
        min_ix = np.argmin(np.cumsum(np.array(total), axis=0), axis=0)
        ablat = np.array(ablat)
        for n in range(min_ix.size):
            ablat[(min_ix[n] + 1):, n] = 0.

        curr_dist = measured_bw_distr[measured_bw_distr.date_f == row.date_f]

        # drop unwanted columns and NaNs (glacier geometry changes)
        curr_dist = curr_dist.drop(
            columns=['id', 'date0', 'date_s', 'date_f', 'date1'])\
            .dropna(axis=1)
        elev_bands = np.array([float(c) for c in curr_dist.columns])
        accum_sum = np.nansum(np.array(accum), axis=0)
        ablat_sum = np.nansum(np.array(ablat), axis=0)

        redist_fac = get_1d_snow_redist_factor(
            elev_bands, curr_dist.values[0], heights, accum_sum,
            mod_abl=ablat_sum)

        # I'm convinced that this is not true anymore, at least after late
        # spring....
        if pd.isnull(row.date1):
            end_insert = pd.Timestamp(row.date_f.year,
                                      cfg.PARAMS['begin_mbyear_month'],
                                      cfg.PARAMS['begin_mbyear_day']) - \
                         pd.Timedelta(days=1)
        else:
            end_insert = row.date1 - dt.timedelta(days=1)
        insert_range = pd.date_range(row.date_f, end_insert)
        ds = xr.Dataset({'D': (['time', 'fl_id', 'model'],
                               np.repeat(np.atleast_3d(redist_fac),
                                         len(insert_range), axis=0))},
                        coords={'time': (['time', ], insert_range),
                                'model': (['model', ], [mbm.__name__]),
                                'fl_id': (
                                    ['fl_id', ], np.arange(heights.size))})
        ds_list.append(ds)

    # override necessary if dates overlap!?
    ds_merge = xr.merge(ds_list, compat='override')
    # fill all NaNs with 1.
    # todo: is this a good idea or does this cause trouble?
    ds_merge = ds_merge.fillna(1.)
    ds_merge.attrs.update({'id': gdir.rgi_id, 'name': gdir.name})

    redist_path = gdir.get_filepath('snow_redist')
    if (reset is True) and (len(mb_models) == len(cfg.MASSBALANCE_MODELS)):
        os.remove(redist_path)
        ds_merge.to_netcdf(redist_path)
    else:
        if gdir.has_file('snow_redist'):
            # load to avoid PermissionError
            with xr.open_dataset(redist_path, autoclose=True).load() as old:
                old_redist = old.copy(deep=True)
            old_redist.load()
            new_redist = ds_merge.combine_first(old_redist)
            new_redist.load()
            old_redist.close()
            old.close()
            old_redist = None
            old = None
            new_redist.to_netcdf(redist_path)
        else:
            ds_merge.to_netcdf(redist_path)
    return ds_merge



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

            mb_ds = xr.Dataset({mb_model.mb_name: (['time', 'n'],
                                                         np.array(mb))},
                               coords={
                                   'time': pd.to_datetime(
                                       mb_model.time_elapsed),
                                   'n': (['n'], [1])},
                               attrs={'id': gdir.rgi_id,
                                      'name': gdir.name})
            gdir.write_pickle(mb_ds, 'mb_daily')

            new_mb = fit_core_mb_to_mean_acc(gdir, core_meta, mb_model)

            gdir.write_pickle(new_mb, 'mb_daily_rescaled')


def fit_core_mb_to_mean_acc(gdir, c_df, mb_model):
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

    factor = mean_mb.apply(lambda x: x / mean_acc)[
        mb_model.mb_name].values
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
