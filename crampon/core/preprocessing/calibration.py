"""
Various calibration functions for the glaciers.
"""

import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import glob
import os
import pickle
import matplotlib.pyplot as plt
import itertools
import scipy.optimize as optimize
from crampon import tasks, entity_task
from crampon.core.models.massbalance import run_snowfirnmodel_with_options, \
    HockModel, DailyMassBalanceModel, SnowFirnCover
from crampon.core.models import massbalance
from collections import OrderedDict
from crampon import utils
import copy
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
    # mb_at_hgts = []
    # conv_fac = ((86400 * cfg.RHO) / cfg.RHO_W)  # m ice s-1 to m w.e. d-1
    for date in calispan:
        tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
        # tmp = day_model.get_daily_mb(heights, date) * conv_fac
        # mb_at_hgts.append(tmp)
        # mb.append(np.average(tmp, weights=widths))
        mb.append(tmp)

    mb_ds = xr.Dataset({'MB': (['time'], mb)},
                       coords={'time': pd.to_datetime(calispan)})
    # mb_at_hgts_ds = xr.Dataset({'MB': (['time', 'fl_id'], mb_at_hgts)},
    #                   coords={'time': pd.to_datetime(calispan),
    #                           'fl_id': (['fl_id',],
    #                           np.arange(len(heights)))})

    # if we have melt in winter between the field dates this pushes prcp_fac
    # up! as we can't measure melt in the winter campaign anyway, we just
    # subtract it away when tuning the winter balance!
    minimum = np.nanmin(np.nancumsum([0.] + mb))
    argmin = np.argmin(np.nancumsum([0.] + mb))

    daily_melt = True
    # if (gdir.rgi_id in assimilation.id_to_stations.keys()) and
    # (daily_melt is True):
    #    assim_data = assimilation.prepare_holfuy_camera_readings(gdir)

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

            # when field visit was after maximum accumulation: let refreeze
            # what can refreeze (otherwise c_prec goes crazy)
            #melt_since_max = (maximum - wsum).MB.item()
            #mean_winter_temp = np.mean([day_model.meteo.get_tmean_at_heights(date, heights) for date in wspan], axis=0)
            #day_model.snowcover.update_temperature_huss(surface_temp=mean_winter_temp)
            #refreeze_potential = np.average(np.nansum(day_model.snowcover.refreezing_potential, axis=1), weights=widths)  # sign is negative
            ## refrozen is part of the observation:
            #if np.abs(refreeze_potential) >= melt_since_max:
            #    refrozen_since_max = melt_since_max
            #else:
            #    refrozen_since_max = np.abs(refreeze_potential)

            if argmax != len(wspan):
                if (maximum - wsum).MB.item() > 0.01:
                    log.info('Difference between maximum {:.2f} on {} and last '
                          'value {:.2f} on {} of {:.2f}.'.format(maximum,
                                                             wspan[argmax].strftime("%Y-%m-%d"), wsum.MB.item(), wspan[
                                                                      -1].strftime("%Y-%m-%d"), (maximum - wsum).MB.item()))
            curr_err = row.Winter - (wsum.MB.values + np.abs(minimum))# + refrozen_since_max)
        else:
            # annual sum
            span = pd.date_range(row.date0, row.date1 - pd.Timedelta(days=1),
                                 freq='D')
            try:
                asum = mb_ds.sel(time=span).map(np.sum)
            except KeyError:
                log.error(f"Given time span {span} is not available in mass "
                          f"balance dataset for {gdir.rgi_id}.")

            correction = 0

            if unc:
                curr_err = (row.Annual - (
                            asum.MB.values + np.abs(correction))) / unc
            else:
                curr_err = (row.Annual - (asum.MB.values + np.abs(correction)))

            # take care of assimilation data
            # if (assim_data.isel(date=0) >= span[0]) and
            # (assim_data.isel(date=-1) <= span[1]):
            #    assim_data = assim_data.sel(date=slice(span[0], span[1]))
            #    stations = assimilation.id_to_station[gdir.rgi_id]
            #    station_hgts = [assimilation.station_to_height[s] for s in
            #    stations]
            #    s_index = np.argmin(
            #        np.abs((heights - np.atleast_2d(station_hgts).T)), axis=1)
            #    for d in assim_data.date.values:
            #        assim_vals = assim_data.sel(date=d,
            #        height=station_hgts).swe
            #        mb_vals = mb_at_hgts_ds.sel(fl_id=s_index, date=d).MB
            #        day_error = np.array(assim_vals) - np.array(mb_vals)
            #        day_error = day_error[~np.isnan(day_error)]
            #        curr_err.append(day_error)

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
        log.info('ASUM, MEAS, CORR: ', mb_sum.MB.values, measured, correction)

        # err = np.abs((measured - (mb_sum.MB.values + correction)))
        err.append(np.abs((measured[p] - (mb_sum.MB.values + correction))))

    if unc is not None:
        err /= unc

    log.info("ERRORS: ", err)

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
    log.info(f"PARAMS: {params}")
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


def update_calibration_file(gdir, cali_df):
    """
    Harmlessly update the calibration file - even when multiprocessing is on.

    Parameters
    ----------
    gdir
    cali_df

    Returns
    -------

    """

    existing = gdir.get_calibration()
    joined = existing.join(cali_df, how='outer')
    joined.to_csv(gdir.get_filepath('calibration'))


def init(lo):
    global lock
    lock = lo


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
    # todo go through generalized objective function and see if winter/annual
    #  mb works ok

    # if (reset_snow_redist is True) and gdir.has_file('snow_redist'):
    #    print('Deleting old snow redist....')
    #    sr_out = None
    #    with xr.open_dataset(gdir.get_filepath('snow_redist')) as sr:
    #        print(sr)
    #        try:
    #            sr_out = sr.where(sr.model != mb_model.__name__, drop=True)
    #            print('worked out')
    #        except AttributeError:
    #            pass
    #    if sr_out is not None:
    #        print(sr_out)
    #        sr_out.to_netcdf(gdir.get_filepath('snow_redist'))

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
            # lock.acquire()
            cali_df.to_csv(gdir.get_filepath('calibration',
                                             filesuffix=cali_suffix))
            # update_calibration_file(gdir, cali_df)
            # lock.release()
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

    # after calculating a snowredist, we should calibr. again but without reset
    # if reset_snow_redist is True:
    #    log.info('Second calibration round as snow redistribution was reset.')
    #    calculate_snow_dist_factor(gdir, reset=True)
    #    calibrate_mb_model_on_measured_glamos(gdir=gdir, mb_model=mb_model,
    #                                          conv_thresh=conv_thresh,
    #                                          it_thresh=it_thresh,
    #                                          cali_suffix=cali_suffix,
    #                                          reset_snow_redist=False,
    #                                          **kwargs)

#@entity_task(log, writes=['calibration_on_holfuy'])
def calibrate_mb_model_on_measured_holfuy(
        gdir, mb_model, conv_thresh=0.005, it_thresh=50,
        cali_suffix='_on_holfuy', snow_redist=True, stations=None,
        reset_snow_redist=True, **kwargs):
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
    stations: list
        List of station IDs to use. Default: None (use all available).
    snow_redist: bool, optional
        Whether to use snow redistribution. Default: True.
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
    # todo go through general obj. function: see if winter/annual mb works ok
    if stations is None:
        stations = holfuytools.id_to_station[gdir.rgi_id]

    # Get measured MB and we can't calibrate longer than our meteo history
    measured, _ = assimilation.prepare_observations(gdir, stations)

    measured_mindate = pd.Timestamp(measured.date.min().values)
    measured_maxdate = pd.Timestamp(measured.date.max().values)

    try:
        snow_cond = gdir.read_pickle('snow_daily').sel(time=measured_mindate,
                                                       model=mb_model.__name__)
    except KeyError:
        # take mean parameters for simplicity (ony one snowcover to handle)
        pg = massbalance.ParameterGenerator(
            gdir, mb_model, latest_climate=True, only_pairs=True,
            constrain_with_bw_prcp_fac=True,
            bw_constrain_year=measured_mindate.year,
            narrow_distribution=0., output_type='array')
        param_prod = pg.from_single_glacier()
        param_dict = dict(
            zip([mb_model.__name__+'_'+p for p in mb_model.cali_params_list],
                np.nanmean(param_prod, axis=0)))
        _, _, _, _, snow_cond = \
            assimilation.make_mb_current_mbyear_heights(
                gdir, mb_models=[mb_model], param_dict=param_dict,
                begin_mbyear=utils.get_begin_last_flexyear(measured_mindate),
                last_day=measured_mindate-pd.Timedelta(days=1), write=False)

    # Find out what we will calibrate
    to_calibrate_csv = [mb_model.prefix + i for i in
                        mb_model.cali_params_guess.keys()]

    # Is there already a calibration where we just can append, or new file
    try:
        cali_df = gdir.get_calibration(filesuffix=cali_suffix)
        # we need to extend it potentially, otherwise we can't write new years
        meas_maxdate = measured.date.max()
        if cali_df.index[-1] < meas_maxdate:
            new_ix = pd.DatetimeIndex(start=cali_df.index[0],
                                      end=meas_maxdate, freq='D')
            cali_df = cali_df.reindex(new_ix)
    # think about an outer join of the date indices here
    except FileNotFoundError:
        try:
            cali_df = pd.DataFrame(columns=to_calibrate_csv +
                                   ['mu_star', 'prcp_fac'],  # 4 OGGM
                                   index=pd.date_range(measured_mindate,
                                                       measured_maxdate))
        except ValueError:  # valid time for MB model is outside measured data
            return

    # we don't know initial snow and time of run history
    # at the new field date (date0)
    run_hist = pd.date_range(utils.get_begin_last_flexyear(measured_mindate),
                             measured_mindate-pd.Timedelta(days=1))
    scov = copy.deepcopy(snow_cond[0])

    for d in measured.date.values:
        meas_sel = measured.sel(date=d)

        heights, widths = gdir.get_inversion_flowline_hw()
        r_ind = 0

        # Check if we are complete
        if pd.isnull(meas_sel.swe).all():
            for name in to_calibrate_csv:
                log.info(f"{meas_sel.date.item()}, {name})
                cali_df.loc[pd.Timestamp(meas_sel.date.item()), name] = np.nan
            cali_df.to_csv(gdir.get_filepath('calibration',
                                             filesuffix=cali_suffix))
            # prepare history for next round
            curr_model = mb_model(gdir, bias=0.)

            mb = []
            # history depends on which start date we choose
            if scov is not None:
                curr_model._time_elapsed = run_hist
                curr_model.snowcover = copy.deepcopy(scov)

                tmp = curr_model.get_daily_specific_mb(heights, widths,
                                                       date=pd.Timestamp(d))
                mb.append(tmp)

            run_hist = curr_model.time_elapsed[:-1]  # same here
            scov = copy.deepcopy(curr_model.snowcover)
            continue

        # todo: adjust snow conditions in scov according to what camera sees

        param_dict = mb_model.cali_params_guess.copy()

        # get an odict with all but prcp_fac
        all_but_prcp_fac = [(k, v) for k, v in param_dict.items()
                            if k not in ['prcp_fac']]

        # todo: no sense to calibrate prcp_fac & melt params on days with both
        #  melt and solid precip. (1) prcp_fac will be tuned to match sum of
        #  both, then melt f. won't change anymore except prcp_fac hits bounds)
        # start with cali on winter MB and optimize only prcp_fac
        if 'FSNOW' in meas_sel.key_remarks:
            try:
                spinupres_w = optimize.least_squares(
                    to_minimize_point_mass_balance,
                    x0=np.array([param_dict['prcp_fac']]),
                    xtol=0.0001,
                    # in this case, bounds should be relaxed
                    bounds=(0.0, None),
                    verbose=2, args=(gdir, mb_model, meas_sel.swe, meas_sel.height,
                                     [d] * len(meas_sel.swe),
                                     [d+pd.Timedelta(days=1)] * len(meas_sel.swe),
                                     [d] * len(meas_sel.swe)),
                    kwargs={'winteronly': True, 'scov': scov, 'prcp_fac':
                            param_dict['prcp_fac']})

                # log status
                log.info('Tuning of prcp_fac:{}'.format(spinupres_w.x[0]))
                param_dict['prcp_fac'] = spinupres_w.x[0]
            except ValueError:  # x0 infeasible
                pass
        else:
            pass

        # if there is an annual balance:
        # take optimized prcp_fac and optimize melt param(s) with annual MB
        # otherwise: skip and do next round of prcp_fac optimization
        # In general, params should be lognormal-distributed (zero-bounded)
        # exception: Oerlamns "c0", which is negative, so the other way around
        if mb_model.__name__ == 'OerlemansModel':
            bounds = [(-np.inf, 0), (0, np.inf)]
        else:
            bounds = [tuple([0] * (len(mb_model.cali_params_list)-1)), tuple([np.inf] * (len(mb_model.cali_params_list)-1))]
        # todo: find a good solution to give bounds to the values?
        if (not pd.isnull(meas_sel.swe).all()) and ((meas_sel.swe <= 0.).any()):
            spinupres = optimize.least_squares(
                to_minimize_point_mass_balance,
                x0=np.array([j for i, j in all_but_prcp_fac]),
                xtol=0.0001,
                #bounds=[i[:-1] for i in param_bounds[mb_model.__name__]],
                bounds=bounds,
                verbose=2, args=(gdir, mb_model, meas_sel.swe, meas_sel.height,
                                 [d] * len(meas_sel.swe),
                                 [d+pd.Timedelta(days=1)] * len(meas_sel.swe),
                                 [d] * len(meas_sel.swe)),
                kwargs={'winteronly': False, 'run_hist': run_hist,
                        'scov': scov,
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

        # Report result
        log.info('After whole cali:{}, grad={}'.format(param_dict.__repr__(),
                                                       grad))

        # Write in cali df
        for k, v in list(param_dict.items()):
            cali_df.loc[d, mb_model.prefix + k] = v
        if isinstance(mb_model, massbalance.BraithwaiteModel):
            cali_df.loc[d, mb_model.prefix + 'mu_snow'] = \
                cali_df.loc[d, mb_model.prefix + 'mu_ice'] \
                * cfg.PARAMS['ratio_mu_snow_ice']
        if isinstance(mb_model, massbalance.HockModel):
            cali_df.loc[d, mb_model.prefix + 'a_snow'] = \
                cali_df.loc[d, mb_model.prefix + 'a_ice'] \
                * cfg.PARAMS['ratio_a_snow_ice']
        cali_df.to_csv(gdir.get_filepath('calibration',
                                         filesuffix=cali_suffix))

        # prepare history for next round
        curr_model = mb_model(gdir, bias=0.)

        mb = []
        # determine start_date
        start_date = d
        end_date = d

        # history depends on which start date we choose
        if scov is not None:
            curr_model.time_elapsed = run_hist
            curr_model.snowcover = copy.deepcopy(scov)
        else:
            raise ValueError('Start date is wrong')

        forward_time = pd.date_range(start_date, end_date)
        for date in forward_time:
            tmp = curr_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)

        run_hist = curr_model.time_elapsed  # same here
        scov = copy.deepcopy(curr_model.snowcover)


def calibrate_on_summer_point_mass_balance(gdir: utils.GlacierDirectory,
                                           obs_d0: pd.Timestamp,
                                           obs_d1: pd.Timestamp,
                                           obs_elev: float,
                                           obs: float,
                                           obs_type: str,
                                           # wb_date: pd.Timestamp,
                                           # wb: float,
                                           mb_model, conv_thresh=0.005,
                                           it_thresh=50, cali_suffix='',
                                           **kwargs):
    """
    This function calibrates a glacier on one given summer point mass balance.

    It is used in the context of experiments with the camera data
    assimilation: we want to test if tuning the parameter on one
    intermediate reading is as good as the particle filter ensemble.
    Therefore we take the precipitation correction factor as it was tuned
    for the glacier-wide winter mass balance (there is not other way) and
    then tune the melt parameters to match the given point mass balance in
    a time span between autumn before and a point of time in summer.

    Parameters
    ----------
    gdir :
    obs_d0 : pd.Timestamp
        Beginning of observation time span.
    obs_d1: pd.Timestamp
        End of observation time span.
    obs_elev: float
        Observation elevation (m).
    obs: float
        Observation in either m snow or m w.e.. The sign should make clear
        whther there has been accumulation(+) in the time span or ablation (-).
    obs_type: str
        Either "snow" or "ice" (decides on the density used)
        # todo: this should probably reviswed and the function should only
        accept m w.e.
    mb_model :
    conv_thresh :
    it_thresh :
    cali_suffix :
    kwargs :

    Returns
    -------
    cali_params: dict
        Dictionary with calibration parameters.
    """
    assert np.array([o in ['snow', 'ice', 'mwe'] for o in obs_type]).all()

    obs = np.asarray(obs)
    obs_type = np.asarray(obs_type)
    unc = np.full_like(obs, np.nan)

    dates_min = np.min(obs_d0)
    dates_max = np.max(obs_d1)

    # get prcp correction factor (should exist already)
    cali_df = gdir.get_calibration(mb_model, cali_suffix)
    prcp_fac = np.nanmean(cali_df.loc[dates_min:dates_max,
                          mb_model.prefix + 'prcp_fac'])
    # prcp_fac = np.nanmean(cali_df.loc[obs_d0:obs_d1, mb_model.prefix +
    #                                                       'prcp_fac'])
    param_guesses = mb_model.cali_params_guess.copy()
    param_dict = mb_model.cali_params_guess.copy()

    # todo: this is for testing of getting it from cali
    param_dict['prcp_fac'] = prcp_fac

    try:
        scov = gdir.read_pickle('snow_daily').sel(model=mb_model.__name__,
                                                  time=dates_min)
        # scov = gdir.read_pickle('snow_daily').sel(model=mb_model.__name__,
        #                                          time=obs_d0)
    except KeyError:
        # the tolerance is a guess
        # scov = gdir.read_pickle('snow_daily').sel(
        #    time=obs_d0, method='nearest', tolerance=pd.Timedelta(
        #        days=31)).sel(model=mb_model.__name__)
        # get one snow cover before
        scov = gdir.read_pickle('snow_daily').sel(
            time=dates_min, method='ffill', tolerance=pd.Timedelta(
                days=62)).sel(model=mb_model.__name__)
        log.warning(
            'The snow cover extracted does not exactly match the observation '
            'time span beginning. Time wanted: {}, time chosen: {}'
            .format(obs_d0, scov.time.values))
    scov_date = scov.time.values
    scov = SnowFirnCover.from_dataset(scov)

    if np.isnan(prcp_fac):
        raise ValueError('Could not read precipitation correction factor '
                         'from calibration file.')

    # if obs_type == 'snow':
    #    obs *= (cfg.PARAMS['autumn_snow_density_guess'] / 1000.)
    # elif obs_type == 'ice':
    #    obs *= (cfg.RHO / 1000.)
    # elif obs_type == 'mwe':
    #    pass
    # else:
    #    raise ValueError('Observation type {} not allowed.'.format(
    #    'obs_type'))
    obs[obs_type == 'snow'] = obs[obs_type == 'snow'] * \
        (cfg.PARAMS['autumn_snow_density_guess'] / 1000.)
    obs[obs_type == 'ice'] = obs[obs_type == 'ice'] * (cfg.RHO / 1000.)

    # generate uncertainty
    # todo: uncertainty values okay?
    unc[obs_type == 'snow'] = obs[obs_type == 'snow'] * 0.2
    unc[obs_type == 'ice'] = 0.1
    unc[obs_type == 'mwe'] = 0.1
    unc = None
    log.info(f"Uncertainty: {unc}")

    """
    # first WB optimization
    wb_result = optimize.least_squares(
        to_minimize_point_mass_balance,
        x0=[v for v in param_guesses.values()],
        xtol=0.0001,
        verbose=2, args=(gdir, mb_model, obs, obs_elev, obs_d0,
                         obs_d1, wb_date, wb),
        kwargs={'scov': scov})  # ,
    # 'prcp_fac': prcp_fac})

    point_result = optimize.least_squares(
                    to_minimize_point_mass_balance,
                    x0=[v for v in param_guesses.values()],
                    xtol=0.0001,
                    verbose=2, args=(gdir, mb_model, obs, obs_elev, obs_d0,
                                     obs_d1, wb_date, wb),
                    kwargs={'scov': scov})#,
                            'prcp_fac': prcp_fac})
    """
    """
    r_ind = 0
    grad = 1
    print(wb, wb_date)
    while grad > conv_thresh:

        # log status
        log.info('{}TH ROUND, grad={}, PARAMETERS: {}'
                 .format(r_ind, grad, param_dict.__repr__()))

        # get an odict with all but prcp_fac
        all_but_prcp_fac = [(k, v) for k, v in param_dict.items()
                            if k not in ['prcp_fac']]

        # start with cali on winter MB and optimize only prcp_fac
        spinupres_w = optimize.least_squares(
            to_minimize_point_mass_balance,
            x0=np.array([param_dict['prcp_fac']]),
            xtol=0.0001,
            bounds=(0.1, 5.),
            verbose=2, args=(gdir, mb_model, wb, obs_elev, obs_d0,
                             wb_date),
            kwargs={'scov': scov,
                    'winteronly': True, **OrderedDict(all_but_prcp_fac)})

        # log status
        log.info('After winter cali, prcp_fac:{}'.format(spinupres_w.x[0]))
        param_dict['prcp_fac'] = spinupres_w.x[0]
        print(param_dict)

        # if there is an annual balance:
        # take optimized prcp_fac and optimize melt param(s) with annual MB
        # otherwise: skip and do next round of prcp_fac optimization
        # todo: find a good solution to give bounds to the values?
        spinupres = optimize.least_squares(
            to_minimize_point_mass_balance,
            x0=np.array([j for i, j in all_but_prcp_fac]),
            xtol=0.0001,
            verbose=2, args=(gdir, mb_model, obs, obs_elev, obs_d0,
                             obs_d1),
            kwargs={'winteronly': False,
                    'scov': scov,
                    'prcp_fac': param_dict['prcp_fac']})

        # update all but prcp_fac
        for j, d_i in enumerate(all_but_prcp_fac):
            param_dict[d_i[0]] = spinupres.x[j]

        # Check whether abort or go on
        grad_test_param = all_but_prcp_fac[0][
            1]  # take 1st, for example
        grad = np.abs(grad_test_param - spinupres.x[0])

        r_ind += 1
        if r_ind > it_thresh:
            warn_it = 'Iterative calibration reached abort criterion of' \
                      ' {} iterations and was stopped at a parameter ' \
                      'gradient of {} for {}.'.format(r_ind, grad,
                                                      grad_test_param)
            log.warning(warn_it)
            break
    """
    all_but_prcp_fac = [(k, v) for k, v in param_dict.items()
                        if k not in ['prcp_fac']]
    spinupres = optimize.least_squares(
        to_minimize_point_mass_balance,
        x0=np.array([j for i, j in all_but_prcp_fac]),
        xtol=0.0001,
        bounds=[i[:-1] for i in param_bounds[mb_model.__name__]],
        verbose=2, args=(gdir, mb_model, obs, obs_elev, obs_d0,
                         obs_d1, scov_date),
        kwargs={'winteronly': False,
                'scov': scov,
                'prcp_fac': param_dict['prcp_fac'],
                'unc': unc})

    for j, d_i in enumerate(all_but_prcp_fac):
        param_dict[d_i[0]] = spinupres.x[j]

    J = spinupres.jac
    cov = np.linalg.inv(J.T.dot(J))
    stdev = np.sqrt(np.diagonal(cov))

    return param_dict


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
            log.warning('Glacier not calibrated for {}. Skipping...'
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


def calibrate_mb_model_on_geod_mb_huss(
        gdir, mb_model, conv_thresh=0.005, it_thresh=50,
        cali_suffix='_fischer', **kwargs):
    """
    Calibrate glaciers on the geod. MB by Fischer et al. (2015) in the way
    Huss et al.(2016) suggests.

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
        Suffix to apply to the calibration file name (read & write). Default:
        "fischer".
    **kwargs: dict
        Keyword arguments accepted by the mass balance model.

    Returns
    -------
    None
    """
    # todo: find a good solution for ratio_s_i
    # todo: do not always hard-code 'mb_model.__name__ + '_' +'
    # todo go through generalized objective function and see if winter/annual
    #  mb works ok

    # read the RandForestPredictor that will give the initial parameter guess
    pred_model_name = os.path.join(cfg.PATHS['working_dir'],
                                   'randomforest_{}.pkl'.format(
                                       mb_model.__name__))
    with open(pred_model_name, 'rb') as pickle_file:
        (rf_features, rf_predictor) = pickle.load(pickle_file)

    # to do define them better upstream
    prcp_fac_bound_low = 0.5
    prcp_fac_bound_high = 2.5

    # Get geodetic MB values
    fischer_df = pd.read_csv(os.path.join(cfg.PATHS['data_dir'],
                                          'fischeretal_2015_geod_mb.csv'),
                             encoding="iso-8859-1")
    fischer_vals = fischer_df[
        fischer_df['SGI2010'].str.contains(gdir.rgi_id.split('.')[1])]

    t1_year = fischer_vals.t1_year.item()
    t2_year = fischer_vals.t2_year.item()
    area1 = fischer_vals.area_t1_km2.item() * 10**6  # km2 -> m2
    area2 = fischer_vals.area_t2_km2.item() * 10**6  # km2 -> m2
    area_change_avg = (area2 - area1) / (t2_year - t1_year)
    gmb_spec_avg = fischer_vals.gmb_spec_t1_t2.item()
    gmb_spec_avg_unc = fischer_vals.uncertainty_gmb_spec_t1_t2.item()

    # use T as a proxy to distribute the MB
    # todo: use T and SIS in years where available
    tsums = []
    psolsums = []
    imeans_summer = []  # JUN-SEP
    gmeteo = climate.GlacierMeteo(gdir)  # needed later
    tmean_glacier = climate.GlacierMeteo(gdir).meteo.temp
    prcp_glacier = climate.GlacierMeteo(gdir).meteo.prcp  # uncorrected!
    imean_glacier = climate.GlacierMeteo(gdir).meteo.sis
    y1y2_span = np.arange(t1_year, t2_year)
    for y in y1y2_span:
        temps = tmean_glacier.sel(
            time=slice('{}-10-01'.format(str(y)), '{}-09-30'
                       .format(str(y + 1)))).values
        prcps = prcp_glacier.sel(
            time=slice('{}-10-01'.format(str(y)),
                       '{}-09-30'.format(str(y + 1)))).values

        temps_pos = np.clip(temps, 0, None)
        tsum = np.sum(temps_pos)
        tsums.append(tsum)
        psolsums.append(np.sum(prcps[temps < 0.]))

        imean_summer = imean_glacier.sel(
            time=slice('{}-06-01'.format(str(y + 1)), '{}-09-30'.format(
                str(y + 1)))).values
        imeans_summer.append(np.mean(imean_summer))

    corr_fac_temp = (1 + (tsums - np.mean(tsums)) / np.mean(tsums))
    corr_fac_prcp = (1 + (psolsums - np.mean(psolsums)) / np.mean(psolsums))
    corr_fac_sis = (1 + (imeans_summer - np.nanmean(imeans_summer)) /
                    np.nanmean(imeans_summer))
    corr_fac = np.mean(corr_fac_prcp, np.nanmean(corr_fac_temp, corr_fac_sis))
    gmb_spec_disagg = gmb_spec_avg * corr_fac
    gmb_spec_unc_disagg = gmb_spec_avg_unc * corr_fac
    area_change_disagg = area_change_avg * corr_fac

    # take name of hydro year as index ("+1")
    disagg_df = pd.DataFrame(index=y1y2_span + 1,
                             columns=['gmb', 'gmb_unc', 'area'],
                             data=np.array([gmb_spec_disagg,
                                            gmb_spec_unc_disagg,
                                            area_change_disagg]).T)

    # todo: we need to process the meteosat radiation data to make this
    #  possible - otherwise only Hock and Braithwaite possible
    # todo: we need to limit the cali period to min 1984, when
    #  radiation model is used
    if mb_model.__name__ in ['OerlemansModel', 'PellicciottiModel']:
        cali_bg_year = 1984
    else:
        cali_bg_year = t1_year

    # Find out what we will calibrate
    to_calibrate_csv = [mb_model.prefix + i for i in
                        mb_model.cali_params_guess.keys()]

    # Is there already a calibration where we just can append, or new file
    try:
        cali_df = gdir.get_calibration(filesuffix=cali_suffix)
    # think about an outer join of the date indices here
    except FileNotFoundError:
        cali_df = pd.DataFrame(
            columns=to_calibrate_csv + ['mu_star', 'prcp_fac'],  # 4 OGGM
            index=pd.date_range('{}-10-01'.format(cali_bg_year),
                                '{}-09-30'.format(t2_year)))

    # we don't know initial snow and time of run history
    run_hist = None  # at the minimum date as calculated by Matthias
    scov = None

    # get initial heights and widths -
    # todo: it should be clear from which year they come - not hard-coded!
    year_init_hw = gdir.rgi_date.year
    heights, widths = gdir.get_inversion_flowline_hw()
    fl_dx = gdir.read_pickle('inversion_flowlines')[-1].dx

    # todo: extend the loss in area backwards in time until 1961 (based on
    #  T sum as proxy) and then run "spinup"
    for year in range(cali_bg_year, t2_year):

        # arbitrary range for which the params are valid - choose the mb year
        valid_range = pd.date_range('{}-10-01'.format(year),
                                    '{}-09-30'.format(year + 1))

        # todo: adapt heights/widths here based on annual MB
        # "+1" for the hydro years
        mb_annual = disagg_df[disagg_df.index == year + 1].gmb.item()
        mb_annual_unc = disagg_df[disagg_df.index == year + 1].gmb_unc.item()

        # area change until date of the outlines
        area_chg = np.sum(disagg_df.loc[year + 1: year_init_hw].area)
        # continue with last width
        # todo: this is to cut off the round tongue: remove the hard-coded!
        if widths.size > 55:
            last_width = np.mean(widths[-55:-25])
        else:  # super short glacier
            last_width = np.mean(widths)
        # continue with slope of lowest
        last_slope = (heights[-1] - heights[-6]) / (5 * fl_dx * gdir.grid.dx)

        # make area chg positive
        n_new_nodes = - area_chg / (fl_dx * gdir.grid.dx) / last_width
        new_heights = last_slope * np.arange(1, np.ceil(n_new_nodes)+1) + \
            heights[-1]
        heights_annual = np.hstack((heights, new_heights))
        widths_annual = np.hstack((widths,  # old width nodes
                                   # new full width nodes
                                   np.repeat([last_width],
                                             np.floor(n_new_nodes)),
                                   # new rest width nodes
                                   np.array([(n_new_nodes %
                                              np.floor(n_new_nodes)) *
                                             last_width])))

        # take care of shape of scov:
        # todo: at the moment this is supercheap: it can only REMOVE,
        #  and ONLY REMOVE AT THE VERY TONGUE
        if scov is not None:
            scov.remove_height_nodes(np.arange(len(widths_annual),
                                               scov.swe.shape[0]))
        grad = 1
        r_ind = 0

        # initial_guess
        # todo: ask if Matthias is ok with them - also with boundaries
        # param_dict = mb_model.cali_params_guess.copy()
        # inital guess with random forest
        tsum_for_rf = np.sum(np.clip(np.average(
            gmeteo.get_tmean_at_heights(valid_range, heights_annual),
            weights=widths_annual, axis=1), 0., None))
        psol_for_rf, _ = gmeteo.get_precipitation_solid_liquid(
            valid_range, heights_annual)
        psum_for_rf = np.sum(np.average(
            psol_for_rf, weights=widths_annual, axis=1))
        zmin = min(heights_annual)
        zmax = max(heights_annual)
        zmed = np.median(heights_annual)
        area = np.sum(widths_annual * fl_dx * gdir.grid.dx)
        hypso = pd.read_csv(gdir.get_filepath('hypsometry'))
        slope = hypso['Slope']
        aspect = hypso['Aspect']

        # todo: store the feature_list elsewhere - in the RandomForestRegressor?
        if mb_model.__name__ == 'BraithwaiteModel':
            feature_list = ['tsum', 'psum', 'Zmin', 'Zmax', 'Zmed', 'Area',
                            'Slope', 'Aspect']
            param_prediction = rf_predictor.predict(
                np.array([tsum_for_rf, psum_for_rf, zmin, zmax, zmed, area,
                          slope, aspect]).reshape(1, -1))
        elif mb_model.__name__ == 'HockModel':
            feature_list = ['tsum', 'psum', 'ipot', 'Zmin', 'Zmax', 'Zmed',
                            'Area', 'Slope', 'Aspect']
            # todo: let iport vary with glacier shape
            ipot = gdir.read_pickle('ipot_per_flowline')
            ipot = np.average(np.mean(np.vstack(ipot), axis=1),
                weights=widths)
            param_prediction = rf_predictor.predict(
                np.array([tsum_for_rf, psum_for_rf, ipot, zmin, zmax, zmed,
                          area, slope, aspect]).reshape(1, -1))
        else:
            raise ValueError('What are the random forest features for {}'
                             .format(mb_model.__name__))

        param_dict = dict(zip(mb_model.cali_params_list.copy(),
                              param_prediction[0]))
        # param_dict = mb_model.cali_params_guess.copy()
        log.info('Initial params predicted: {}'.format(" ".join(
            "{} {}".format(k, v) for k, v in param_dict.items())))

        # say what we are doing
        log.info('Calibrating budget year {}/{}'.format(year, year+1))

        def prcp_fac_cali(pdict, all_but_pfac):
            # start with cali on winter MB and optimize only prcp_fac
            # todo: pass heights_annual and widths_annual here
            spinupres_w = optimize.least_squares(
                to_minimize_mb_calibration_on_fischer,
                x0=np.array([pdict['prcp_fac']]),
                xtol=0.0001,
                #bounds=(prcp_fac_bound_low, prcp_fac_bound_high),
                method='trf',
                verbose=2, args=(gdir, mb_model, mb_annual, valid_range[0],
                                 valid_range[-1], heights_annual,
                                 widths_annual),
                kwargs={'run_hist': run_hist, 'scov': scov,
                        'prcp_corr_only': True,
                        **OrderedDict(all_but_pfac)})
            # log status
            log.info('After winter cali, prcp_fac:{}'.format(spinupres_w.x[0]))

            return spinupres_w

        while grad > conv_thresh:

            # log status
            log.info('{}TH ROUND, grad={}, PARAMETERS: {}'
                     .format(r_ind, grad, param_dict.__repr__()))

            # get an odict with all but prcp_fac
            all_but_prcp_fac = [(k, v) for k, v in param_dict.items()
                                if k not in ['prcp_fac']]

            spinupres_w = prcp_fac_cali(param_dict, all_but_prcp_fac)

            # check if we ran into boundaries: if no, we're done already!
            if ~np.isclose(spinupres_w.x[0], prcp_fac_bound_low, 0.01) and \
                    ~np.isclose(spinupres_w.x[0], prcp_fac_bound_high, 0.01):
                # set the value of the GLOBAL param_dict
                param_dict['prcp_fac'] = spinupres_w.x[0]
                break
            # otherwise: try to not run into boundaries
            else:
                while np.isclose(spinupres_w.x[0], prcp_fac_bound_low, 0.01):
                    # start over again, but with TF 20% higher
                    # todo: 1.2/0.8 influence the final result => TF will stay at these values
                    all_but_prcp_fac = [(x, y * 1.2) for x, y in all_but_prcp_fac]
                    spinupres_w = prcp_fac_cali(param_dict, all_but_prcp_fac)
                    log.info(f"{spinupres_w.x[0]}")
                while np.isclose(spinupres_w.x[0], prcp_fac_bound_high, 0.01):
                    # start over again, but with TF 20% lower
                    all_but_prcp_fac = [(x, y * 0.8) for x, y in all_but_prcp_fac]
                    spinupres_w = prcp_fac_cali(param_dict, all_but_prcp_fac)
                    log.info(f"{spinupres_w.x[0]}")
                # set the value of the GLOBAL param_dict *now*
                param_dict['prcp_fac'] = spinupres_w.x[0]

            # if needed:
            # todo: find a good solution to give bounds to the values?
            # todo: this is now actually not necessary anymore, since the
            #  while loops take care of everything.
            spinupres = optimize.least_squares(
                to_minimize_mb_calibration_on_fischer,
                x0=np.array([j for i, j in all_but_prcp_fac]),
                xtol=0.0001,
                method='trf',
                verbose=2,
                args=(gdir, mb_model, mb_annual, valid_range[0],
                      valid_range[-1], heights_annual, widths_annual),
                kwargs={'prcp_corr_only': False, 'run_hist': run_hist,
                        'scov': scov, 'prcp_fac': param_dict['prcp_fac']})

            # update all but prcp_fac
            for j, d_i in enumerate(all_but_prcp_fac):
                param_dict[d_i[0]] = spinupres.x[j]

            # Check whether abort or go on
            grad_test_param = all_but_prcp_fac[0][
                1]  # take 1st, for example
            grad = np.abs(grad_test_param - spinupres.x[0])

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

        # Write in cali df
        for k, v in list(param_dict.items()):
            cali_df.loc[valid_range[0]:valid_range[-1],
            mb_model.prefix + k] = v
        if isinstance(mb_model, massbalance.BraithwaiteModel):
            cali_df.loc[valid_range[0]:valid_range[-1],
            mb_model.prefix + 'mu_snow'] = \
                cali_df.loc[valid_range[0]:valid_range[-1],
                mb_model.prefix + 'mu_ice'] * cfg.PARAMS['ratio_mu_snow_ice']
        if isinstance(mb_model, massbalance.HockModel):
            cali_df.loc[valid_range[0]:valid_range[-1],
            mb_model.prefix + 'a_snow'] = \
                cali_df.loc[valid_range[0]:valid_range[-1],
                mb_model.prefix + 'a_ice']  * cfg.PARAMS['ratio_a_snow_ice']
        cali_df.to_csv(gdir.get_filepath('calibration',
                                         filesuffix=cali_suffix))

        # prepare history for next round, while handing over cali params now !!
        curr_model = mb_model(gdir, bias=0., **param_dict,
                              heights_widths=(heights_annual, widths_annual))

        mb = []

        # history depends on which start date we choose
        if scov is not None:
            curr_model.time_elapsed = run_hist
            curr_model.snowcover = copy.deepcopy(scov)

        for date in valid_range:
            tmp = curr_model.get_daily_specific_mb(heights_annual,
                                                   widths_annual, date=date)
            mb.append(tmp)

            # prepare for next annual calibration (row.date1 == nextrow.date0)
            if date == valid_range[-1]:
                run_hist = curr_model.time_elapsed[:-1]  # same here
                scov = copy.deepcopy(curr_model.snowcover)

        error = mb_annual - np.sum(mb)
        log.info('ERROR to measured MB:{}'.format(error))


def use_geodetic_mb_for_multipolygon_or_not(id, poly_df, threshold=0.9):
    """
    If a glacier is a multipolygon, decide whether the geodetic mass balance
    for the whole multipolygon can be used for the entity.

    The way we test this is if the entity polyon makes up more than or equal
    the threshold times the total area of the multipolygon.

    Parameters
    ----------
    id: str
        ID of the single entity polygon.
    poly_df: pd.Dataframe or gpd.GeoDataframe
        Dataframe including an ID and area column.
    threshold: float
        Threshold of area ratio to total multipolygon area the that the polygon
        must be equal to at least. Default: 0.9
        .

    Returns
    -------
    bool:
        True if geodetic mass balance can be used, False if not.
    """
    common_id = id.split('.')[1].split('-')[0]
    mpoly_areas = poly_df[poly_df['RGIId'].str.contains(common_id)].Area.values
    mpoly_total_area = np.nansum(mpoly_areas)
    id_area = poly_df.loc[poly_df.RGIId == id, 'Area'].values
    if (id_area / mpoly_total_area) >= threshold:
        return True
    else:
        return False


def calibrate_multipolygons_on_geodetic(gdir, cali_func):
    """
    If a glacier is a multipolygon, decide whether the geodetic mass balance
    for the whole multipolygon can be used for the entity.

    The way we test this is if the entity polyon makes up more than or equal
    the threshold times the total area of the multipolygon.

    Parameters
    ----------
    id: str
        ID of the single entity polygon.
    poly_df: pd.Dataframe or gpd.GeoDataframe
        Dataframe including an ID and area column.
    threshold: float
        Threshold of area ratio to total multipolygon area the that the polygon
        must be equal to at least. Default: 0.9
        .

    Returns
    -------
    bool:
        True if geodetic mass balance can be used, False if not.
    """
    poly_df = os.path.join(cfg.PATHS['data_dir'], 'outlines',
                           'mauro_sgi_merge.shp')

    #merged_gdir = workflow.merge_glacier_tasks(gdirs, main_rgi_id=None,
    #                                           glcdf=poly_df, return_all=True,
    #                                           filename='climate_daily',
    #                                           buffer=100)

    for mbm in [BraithwaiteModel, HockModel]:
        cali_func(merged_gdir[0], mbm)

@entity_task(log, writes=['calibration_fischer_unique'])
def calibrate_mb_model_on_geod_mb_huss_one_paramset(gdir, mb_model,
                                                    conv_thresh=0.005,
                                                    it_thresh=5,
                                                    cali_suffix='_fischer_unique',
                                                    geometry_change='linear',
                                                    **kwargs):
    """
    Calibrate glaciers on geodetic mass balances by [Fischer et al. (2015)]_.

    We do similar to [Huss et al. (2016)]_, delivering only one
    parameter set for the whole time range. However, in order to change the
    problem of always getting stuck in guessed initial parameters for the
    calibration (equifinality!), we use a random forest regressor trained on
    the GLAMOS glaciers to predict the initial parameters and vary the
    precipitation correction factor then. This is because the precipitation
    correction factor has by far the lowet prediction accuracy in the
    RandomForestRegressor validation.

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
        after 5 iterations (otherwise claibration can go crazy).
    cali_suffix: str
        Suffix to apply to the calibration file name (read & write). Default:
        "fischer".
    geometry_change: str
        # todo: take the real for 'half-half' geometries here!!!
        How to determine the geometry change. Possible: (1) 'linear': try and
        imitate a linear area change, (2): 'half-half': take the estimated
        geometry at the beginning of the time period for the first half of the
        calibration time span, the estimated geometry at the end of the period
        for the second half or (3) 'nochange': no geometry change (take the one
        given). The latter can actually be especially useful as long as there
        is no real dynamics involved: We use the mass budget year as a
        predictor for the RandomForestRegressor, which is also trained on not
        changing geometries. Default: 'linear'.
    **kwargs: dict
        Keyword arguments accepted by the mass balance model.

    Returns
    -------
    None

    References
    ----------
    .. [Fischer et al. (2015)]: Fischer, M., Huss, M., & Hölzle, M. (2015).
        Surface elevation and mass changes of all Swiss glaciers 1980–2010.
        The Cryosphere, 9(2), 525-540.
    .. [Huss et al. (2016)]: Huss, M., & Fischer, M. (2016). Sensitivity of
        very small glaciers in the Swiss Alps to future climate change.
        Frontiers in Earth Science, 4, 34.
    """
    # todo: find a good solution for ratio_s_i
    # todo: do not always hard-code 'mb_model.__name__ + '_' +'
    # todo go through generalized objective function and see if winter/annual mb works ok

    # Read the RandForestPredictor that gives the initial parameter guess
    # if glacier in GLAMOS: make new tree in order not to train the forest with itself.
    glamos_ids = cfg.PARAMS['glamos_ids'].copy()
    if gdir.rgi_id in glamos_ids:
        from core import validation
        # make a random forest that is trained without the glacier itself
        glamos_ids.remove(gdir.rgi_id)
        # remove those that "don't work", bcz there are processing problems
        glamos_ids_clean = [e for e in glamos_ids if e not in
                            ['RGI50-11.A51E12', 'RGI50-11.A50I19-4',
                           'RGI50-11.A50I07-1']]
        rf_features, rf_predictor = validation.make_param_random_forest(
            mb_model, os.path.join(cfg.PATHS['working_dir'], 'per_glacier'),
            glamos_ids_clean, write=False)
    else:
        # open the one that is trained with all glamos IDs
        pred_model_name = os.path.join(cfg.PATHS['working_dir'],
                                       'randomforest_{}.pkl'.format(
                                           mb_model.__name__))
        with open(pred_model_name, 'rb') as pickle_file:
            (rf_features, rf_predictor) = pickle.load(pickle_file)

    # to do define them better upstream
    prcp_fac_bound_low = 0.5
    prcp_fac_bound_high = 5.0

    # Get geodetic MB values
    fischer_df = pd.read_csv(os.path.join(cfg.PATHS['data_dir'], 'fischeretal_2015_geod_mb.csv'), encoding="iso-8859-1")
    #rgidf = gpd.read_file(os.path.join(cfg.PATHS['data_dir'], 'outlines',
    # 'mauro_sgi_merge.shp'))

    # check if it's a separated multipolygon
    #if len(gdir.rgi_id.split('.')[1].split('-')) > 1:
    #    result_bool = use_geodetic_mb_for_multipolygon_or_not(gdir.rgi_id, rgidf)
    #    if result_bool is False:
    #        raise ValueError('This multipolygon part cannot be calibrated.')
    if "merged" in gdir.rgi_id:
        common_id = gdir.rgi_id.split('.')[1].split('_')[0]
    else:
        common_id = gdir.rgi_id.split('.')[1].split('-')[0]
    fischer_vals = fischer_df[fischer_df['SGI2010'].str.contains(common_id)]
    t1_year = fischer_vals.t1_year.item()
    t2_year = fischer_vals.t2_year.item()
    area1 = fischer_vals.area_t1_km2.item() * 10**6  # km2 -> m2
    area2 = fischer_vals.area_t2_km2.item() * 10**6  # km2 -> m2
    gmb = fischer_vals.gmb_spec_t1_t2.item() * (t2_year - t1_year)
    area_change_avg = (area2 - area1) / (t2_year - t1_year)
    gmb_spec_avg = fischer_vals.gmb_spec_t1_t2.item()
    gmb_spec_avg_unc = fischer_vals.uncertainty_gmb_spec_t1_t2.item()

    # use weather as proxies to distribute the MB
    tsums = []
    psolsums = []
    imeans_summer = []  # JUN-SEP
    tmean_glacier = climate.GlacierMeteo(gdir).meteo.temp
    prcp_glacier = climate.GlacierMeteo(gdir).meteo.prcp  # uncorrected!
    imean_glacier = climate.GlacierMeteo(gdir).meteo.sis
    y1y2_span = np.arange(t1_year, t2_year)
    for y in y1y2_span:
        temps = tmean_glacier.sel(time=slice('{}-10-01'.format(str(y)),
                                             '{}-09-30'.format(
                                                 str(y + 1)))).values
        prcps = prcp_glacier.sel(time=slice('{}-10-01'.format(str(y)),
                                            '{}-09-30'.format(
                                                str(y + 1)))).values

        temps_pos = np.clip(temps, 0, None)
        tsum = np.sum(temps_pos)
        tsums.append(tsum)
        psolsums.append(np.nansum(prcps[temps < 0.]))

        imean_summer = imean_glacier.sel(
            time=slice('{}-06-01'.format(str(y + 1)),
                       '{}-09-30'.format(
                           str(y + 1)))).values
        imeans_summer.append(np.mean(imean_summer))

    # take the disaggregation out: it's bullshit
    #corr_fac_temp = (1 + (tsums - np.nanmean(tsums)) / np.nanmean(tsums))
    #corr_fac_prcp = (1 + (psolsums - np.nanmean(psolsums)) / np.nanmean(psolsums))
    #corr_fac_sis = (
    #            1 + (imeans_summer - np.nanmean(imeans_summer)) / np.nanmean(
    #        imeans_summer))
    #if np.isnan(corr_fac_sis).any():
    #    corr_fac = np.nanmean([corr_fac_prcp, corr_fac_temp], axis=0)
    #else:
    #    corr_fac = np.nanmean([corr_fac_prcp, np.nanmean([corr_fac_temp, corr_fac_sis], axis=0)], axis=0)
    #gmb_spec_disagg = gmb_spec_avg * corr_fac
    #gmb_spec_unc_disagg = gmb_spec_avg_unc * corr_fac
    #area_change_disagg = area_change_avg * corr_fac

    # take name of hydro year as index ("+1")
    #disagg_df = pd.DataFrame(index=y1y2_span + 1,
    #                         columns=['gmb', 'gmb_unc', 'area'],
    #                         data=np.array([gmb_spec_disagg,
    #                                        gmb_spec_unc_disagg,
    #                                        area_change_disagg]).T)
    df_len = len(y1y2_span)
    disagg_df = pd.DataFrame(index=y1y2_span + 1,
                             columns=['gmb', 'gmb_unc', 'area'],
                             data=np.array([np.full(df_len, gmb_spec_avg),
                                            np.full(df_len, gmb_spec_avg_unc),
                                            np.full(df_len, area_change_avg)]).T)

    # use T as a proxy to distribute the MB
    # todo: use T and SIS in years where available
    tsums = []
    gmeteo = climate.GlacierMeteo(gdir)  # needed later
    tmean_glacier = climate.GlacierMeteo(gdir).meteo.temp

    # todo: we need to take another model until (if geod. cali period starts before) 1984 and then start calibrating
    if (mb_model.__name__ in ['OerlemansModel', 'PellicciottiModel']) and (
            t1_year < 1984):
        cali_bg_year = 1984
    else:
        cali_bg_year = t1_year

    # Find out what we will calibrate
    to_calibrate_csv = [mb_model.prefix + i for i in
                        mb_model.cali_params_guess.keys()]

    # Is there already a calibration where we just can append, or new file
    try:
        cali_df = gdir.get_calibration(filesuffix=cali_suffix)
    # think about an outer join of the date indices here
    except FileNotFoundError:
        cali_df = pd.DataFrame(columns=to_calibrate_csv+['mu_star', 'prcp_fac'],  # 4 OGGM
                               index=pd.date_range('{}-10-01'.format(cali_bg_year),
                                                   '{}-09-30'.format(t2_year)))

    # we don't know initial snow and time of run history
    run_hist = None  # at the minimum date as calculated by Matthias
    scov = None

    # get initial heights and widths
    year_init_hw = gdir.rgi_date.year
    heights, widths = gdir.get_inversion_flowline_hw()
    fl_dx = gdir.read_pickle('inversion_flowlines')[-1].dx

    # arbitrary range for which the params are valid - choose the mb year
    valid_range = pd.date_range('{}-10-01'.format(t1_year),
                                '{}-09-30'.format(t2_year))

    all_predicts = []
    all_heights = []
    all_widths = []
    hwyears = []
    #for year in range(cali_bg_year, t2_year):
    for year in range(t1_year, t2_year):
        grad = 1
        r_ind = 0

        year_span = pd.date_range('{}-10-01'.format(year),
                                    '{}-09-30'.format(year + 1))

        # continue with last width
        # todo: this is to cut off the round tongue: remove the hard-coded numbers!
        if widths.size > 55:
            last_width = np.mean(widths[-55:-25])
        else:  # super short glacier
            last_width = np.mean(widths)
        # continue with slope of lowest; first try last 5 & then become smaller
        for n in np.arange(2, 7)[::-1]:
            try:
                last_slope = (heights[-1] - heights[-n]) / \
                             ((n - 1) * fl_dx * gdir.grid.dx)
                break
            except IndexError:
                continue

        # geometry depends on the method:
        if ((geometry_change == 'half-half') and (year > (t2_year - t1_year) / 2.)) or (geometry_change == 'nochange'):
                # take the heights and widths as given
                heights_annual = heights
                widths_annual = widths
        else:
            if geometry_change == 'half-half':
                area_chg = np.sum(disagg_df.area)
            elif geometry_change == 'linear':
                # area change until date of the outlines
                area_chg = np.sum(disagg_df.loc[year + 1: year_init_hw].area)
            else:
                raise ValueError('Method "{}" for geometry change is not '
                                 'recognized.')
            # make area chg positive
            n_new_nodes = - area_chg / (fl_dx * gdir.grid.dx) / last_width
            new_heights = last_slope * np.arange(1, np.ceil(n_new_nodes) + 1) + \
                          heights[-1]
            heights_annual = np.hstack((heights, new_heights))
            widths_annual = np.hstack((widths,  # old width nodes
                                       np.repeat([last_width],
                                                 np.floor(n_new_nodes)),
                                       # new full width nodes
                                       np.array([(n_new_nodes % max(
                                           np.floor(n_new_nodes),
                                           1)) * last_width])))  # new rest width nodes

        all_heights.append(heights_annual)
        all_widths.append(widths_annual)
        hwyears.append(year+1)
        ## initial_guess
        ## todo: ask if Matthias is ok with them - also with boundaries
        #param_dict = mb_model.cali_params_guess.copy()
        # inital guess with random forest

        tsum_for_rf = np.nansum(np.clip(np.average(gmeteo.get_tmean_at_heights(year_span, heights_annual), weights=widths_annual, axis=1), 0., None))
        psol_for_rf, _ = gmeteo.get_precipitation_solid_liquid(year_span, heights_annual)
        psum_for_rf = np.nansum(np.average(psol_for_rf, weights=widths_annual, axis=1))
        sissum_for_rf = None
        if mb_model.__name__ in ['PellicciottiModel', 'OerlemansModel']:
            sissum_for_rf = np.nansum(
                gmeteo.meteo.sis.sel(time=slice(year_span[0], year_span[-1])))
        zmin = min(heights_annual)
        zmax = max(heights_annual)
        zmed = np.median(heights_annual)
        area = np.sum(widths_annual * fl_dx * gdir.grid.dx)
        hypso = pd.read_csv(gdir.get_filepath('hypsometry'))
        slope = hypso['Slope']
        aspect = hypso['Aspect']
        mbyear = year_span[-1].year

        if psum_for_rf == 0.:
            raise ValueError('Precipitation sum is zero')
        if tsum_for_rf == 0.:
            raise ValueError('Temperature sum is zero')
        #print(year, ~np.isnan(heights_annual).all(),
        #      ~np.isnan(widths_annual).all(), tsum_for_rf, psum_for_rf, year,
        #      tsum_for_rf, psum_for_rf, zmin, zmax, zmed, area, slope, aspect)

        # todo: store the feature_list elsewhere - in the RandomForestRegressor?
        if mb_model.__name__ == 'BraithwaiteModel':
            param_prediction = rf_predictor.predict(np.array([tsum_for_rf, psum_for_rf, mbyear, zmin, zmax, zmed, area, slope, aspect]).reshape(1, -1))
            log.info('Param prediction ({}): {}'.format(mbyear, param_prediction))
        elif mb_model.__name__ == 'HockModel':
            # todo: let ipot vary with glacier shape
            ipot = gdir.read_pickle('ipot_per_flowline')
            ipot = np.average(np.mean(np.vstack(ipot), axis=1),
                                      weights=widths)
            param_prediction = rf_predictor.predict(np.array(
                [area, aspect, slope, zmax, zmed, zmin, ipot, mbyear,
                 psum_for_rf, tsum_for_rf]).reshape(1, -1))
            log.info(f"Param prediction: {param_prediction}")
        elif mb_model.__name__ in ['PellicciottiModel', 'OerlemansModel']:
            feature_list = ['tsum', 'psum', 'sissum', 'Zmin', 'Zmax', 'Zmed',
                            'Area', 'Slope', 'Aspect']
            param_prediction = rf_predictor.predict(np.array(
                [area, aspect, slope, zmax, zmed, zmin, mbyear, psum_for_rf,
                 sissum_for_rf, tsum_for_rf]).reshape(1, -1))
            log.info(f"Param prediction: {param_prediction}")
        else:
            raise ValueError(
                'What are the random forest features for {}'.format(
                    mb_model.__name__))
        all_predicts.append(param_prediction[0])

    all_predicts_mean = np.mean(np.array(all_predicts), axis=0)

    #param_dict = dict(zip(mb_model.cali_params_list.copy(), param_prediction[0]))
    param_dict = dict(zip(mb_model.cali_params_list.copy(), all_predicts_mean))

    #param_dict = mb_model.cali_params_guess.copy()
    log.info('Initial params predicted: {}'.format(" ".join("{} {}".format(k, v) for k, v in param_dict.items())))

    # say what we are doing
    log.info('Calibrating all years together')

    def prcp_fac_cali(pdict, all_but_pfac):
        # start with cali on winter MB and optimize only prcp_fac
        # todo: pass heights_annual and widths_annual here
        spinupres_w = optimize.least_squares(
            to_minimize_mb_calibration_on_fischer_one_set,
            x0=np.array([pdict['prcp_fac']]),
            xtol=0.0001,
            bounds=(prcp_fac_bound_low, prcp_fac_bound_high),
            method='trf',
            verbose=2, args=(gdir, mb_model, gmb, valid_range[0],
                             valid_range[-1], all_heights,
                             all_widths, hwyears),
            kwargs={'run_hist': run_hist, 'scov': scov,
                    'prcp_corr_only': True,
                    **OrderedDict(all_but_pfac)})
        # log status
        log.info('After winter cali, prcp_fac:{}'.format(spinupres_w.x[0]))

        return spinupres_w

    r_ind = 0
    grad = 1
    bounds_error = False
    while grad > conv_thresh:

        # log status
        log.info('{}TH ROUND, grad={}, PARAMETERS: {}'.format(r_ind, grad,
                                                              param_dict.__repr__()))

        # get an odict with all but prcp_fac
        all_but_prcp_fac = [(k, v) for k, v in param_dict.items()
                            if k not in ['prcp_fac']]

        spinupres_w = prcp_fac_cali(param_dict, all_but_prcp_fac)

        # check if we ran into boundaries: if no, we're done already!
        if ~np.isclose(spinupres_w.x[0], prcp_fac_bound_low, 0.01) and ~np.isclose(spinupres_w.x[0], prcp_fac_bound_high, 0.01):
            # set the value of the GLOBAL param_dict
            param_dict['prcp_fac'] = spinupres_w.x[0]
            break
        # otherwise: try to not run into boundaries
        else:
            while np.isclose(spinupres_w.x[0], prcp_fac_bound_low, 0.01):
                # start over again, but with TF 20% higher
                # todo: 1.2/0.8 influence the final result => TF will stay at these values
                all_but_prcp_fac = [(x, y * 1.2) for x, y in all_but_prcp_fac]
                spinupres_w = prcp_fac_cali(param_dict, all_but_prcp_fac)
                r_ind += 1
                if r_ind > it_thresh:
                    bounds_error = True
                    break

            while np.isclose(spinupres_w.x[0], prcp_fac_bound_high, 0.01):
                # start over again, but with TF 20% lower
                all_but_prcp_fac = [(x, y * 0.8) for x, y in all_but_prcp_fac]
                spinupres_w = prcp_fac_cali(param_dict, all_but_prcp_fac)
                r_ind += 1
                if r_ind > it_thresh:
                    bounds_error = True
                    break

            if bounds_error is True:
                log.warning(
                    'Could not calibrate on geodetic mass balances (params: {}'.format(
                        all_but_prcp_fac, spinupres_w.x[0]))
                break
            # set the value of the GLOBAL param_dict *now*
            param_dict['prcp_fac'] = spinupres_w.x[0]

        # if needed:
        # todo: find a good solution to give bounds to the values?
        # todo: this is now actually not necessary anymore, since the wo while loops take care of everything.
        spinupres = optimize.least_squares(
            to_minimize_mb_calibration_on_fischer_one_set,
            x0=np.array([j for i, j in all_but_prcp_fac]),
            xtol=0.0001,
            method='trf',
            verbose=2, args=(gdir, mb_model, gmb, valid_range[0],
                             valid_range[-1], all_heights, all_widths, hwyears),
            kwargs={'prcp_corr_only': False, 'run_hist': run_hist,
                    'scov': scov, 'prcp_fac': param_dict['prcp_fac']})

        # update all but prcp_fac
        for j, d_i in enumerate(all_but_prcp_fac):
            param_dict[d_i[0]] = spinupres.x[j]

        # Check whether abort or go on
        grad_test_param = all_but_prcp_fac[0][
            1]  # take 1st, for example
        grad = np.abs(grad_test_param - spinupres.x[0])

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

    if bounds_error is False:
        # Write in cali df
        for k, v in list(param_dict.items()):
            cali_df.loc[valid_range[0]:valid_range[-1], mb_model.prefix + k] = v
        if isinstance(mb_model, massbalance.BraithwaiteModel):
            cali_df.loc[valid_range[0]:valid_range[-1], mb_model.prefix + 'mu_snow'] = \
                cali_df.loc[valid_range[0]:valid_range[-1], mb_model.prefix + 'mu_ice'] \
                * cfg.PARAMS['ratio_mu_snow_ice']
        if isinstance(mb_model, massbalance.HockModel):
            cali_df.loc[valid_range[0]:valid_range[-1], mb_model.prefix + 'a_snow'] = \
                cali_df.loc[valid_range[0]:valid_range[-1], mb_model.prefix + 'a_ice'] \
                * cfg.PARAMS['ratio_a_snow_ice']
        cali_df.to_csv(gdir.get_filepath('calibration',
                                         filesuffix=cali_suffix))

        # error can be logged from last result
        try:
            log.info('ERROR to measured MB:{}'.format(spinupres.fun[0]))
        except UnboundLocalError:  # we never arrived there
            log.info('ERROR to measured MB:{}'.format(spinupres_w.fun[0]))

        # make a calibration file where there is imposed variability from GLAMOS calibration
        try:
            _impose_variability_on_geodetic_params(gdir, suffix=cali_suffix)
        except FileNotFoundError:
            pass


def _impose_variability_on_geodetic_params(gdir, suffix):
    """
    Take the calibration on GLAMOS mass balances and impose it on the geodetic
    calibration.

    Parameters
    ----------
    gdir: pd.DataFrame
        Dataframe with the calibration on geodetic mass balances.
    suffix: str
        Suffix under which the new Dataframe with imposed variability will be
        stored. This function automatically attaches another "variab" to this
        suffix.

    Returns
    -------
    None
    """

    geod_cali = gdir.get_calibration(filesuffix=suffix)
    geod_models = [p.split('_')[0] for p in geod_cali.columns if 'Model' in p]
    minmax_dates_geod = geod_cali.index.min(), geod_cali.index.max()

    for gm in np.unique(geod_models):
        glamos_cali = massbalance.ParameterGenerator(mb_model=gm).pooled_params
        #glamos_cali = gdir.get_calibration(mb_model=gm)
        minmax_dates_glam = glamos_cali.index.min(), glamos_cali.index.max()
        mask_glam = (glamos_cali.index >= minmax_dates_geod[0]) & (
                glamos_cali.index <= minmax_dates_geod[1])
        mask_geod = (geod_cali.index >= minmax_dates_glam[0]) & (
                geod_cali.index <= minmax_dates_glam[1])
        glamos_cali_masked = glamos_cali.loc[mask_glam]
        glamos_cali_masked = glamos_cali_masked.resample('AS').mean()

        glamos_variability = glamos_cali_masked.copy()
        glamos_variability[:] = 1.
        glamos_variability = glamos_cali_masked / np.nanmean(
            glamos_cali_masked, axis=0)

        geod_cali.loc[
            mask_geod, glamos_variability.columns] *= glamos_variability

    dummy_date = pd.Timestamp('{}-{}-{}'.format(2018,
                                                cfg.PARAMS['bgmon_hydro'],
                                                cfg.PARAMS['bgday_hydro']))
    n_bfill = (dummy_date + pd.tseries.offsets.YearEnd() - dummy_date +
               pd.Timedelta(days=1)).days
    n_ffill = dummy_date.dayofyear - 1
    data = geod_cali.dropna(axis=0, how='all').resample('D').bfill(
        limit=n_bfill).ffill(limit=n_ffill)
    data.to_csv(gdir.get_filepath('calibration', filesuffix=suffix +
                                                            '_variability'))


def get_1d_snow_redist_factor(obs_heights, obs, mod_heights, mod, mod_abl=None):
    """
    Calculate the snow redistribution factor in 1D.

    To get the values for the model heights, values from the observation
    heights are interpolated linearly.

    Parameters
    ----------
    obs_heights: array
        Heights of the observations, e.g. elevation band mean heights (m).
    obs: array
        Observation snow water equivalents measured at the observation heights
        (m w.e.). According to Matthias, this can be both accumulation and
        ablation (not yte separated in GLAMOS!).
    mod_heights: array
        Heights of the model, e.g. flowline node heights (m).
    mod: array
        Modeled accumulation(!) values with the precipitation gradients from
        the meteorological grids applied.
    mod_abl:
        Modeled ablation (!) values in the mean time. This is just an emergency
        solution: Since GLAMOS can be both accumulation and ablation and
        Matthias doesn't want (yet) to extract accumulation only, we calculate
        it with the melt model...even though it's horrible.

    Returns
    -------
    redist_fac: array
        The snow redistribution factor for the model elevations.
    """

    half_intvl = (obs_heights[1] - obs_heights[0]) / 2.
    bin_edges = np.hstack((obs_heights - half_intvl, obs_heights[-1] +
                           half_intvl))
    if mod_abl is not None:
        digitized = np.digitize(mod_heights, bin_edges, right=True)
        mod_abl_binned = np.array([mod_abl[digitized == i].mean() for i in
                                   range(1, len(bin_edges))])
        # there can be NaNs
        if np.isnan(mod_abl_binned).any():
            mod_abl_binned = np.interp(obs_heights,
                                       obs_heights[~np.isnan(mod_abl_binned)],
                                       mod_abl_binned[~np.isnan(mod_abl_binned)])

        obs = obs - mod_abl_binned
    else:
        # emergency break
        return np.ones_like(mod_heights, dtype=float)

    # left and right margins are not totally correct, but ok....
    meas_interp = np.interp(mod_heights, obs_heights, obs)

    redist_fac = np.full_like(mod, np.nan)

    for bin_no in range(1, len(bin_edges)):
        redist_fac[np.where(digitized == bin_no)[0]] = \
            np.mean(meas_interp[np.where(digitized == bin_no)[0]]) / \
            np.mean(mod[np.where(digitized == bin_no)[0]])

    return redist_fac


def visualize(mb_xrds, msrd, err, x0, mbname, ax=None):
    if not ax:
        fig, ax = plt.subplots()
        ax.scatter(msrd.date1.values, msrd.mbcumsum.values)
        ax.hline()
    ax.plot(mb_xrds.sel(
        time=slice(min(msrd.date0), max(msrd.date1))).time,
            np.cumsum(mb_xrds.sel(
                time=slice(min(msrd.date0), max(msrd.date1)))[mbname],
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


def make_massbalance_at_firn_core_sites(core_meta, reset=False, mb_model=None):
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
            log.info(f"Row ID: {row.id}")
            gdir = utils.idealized_gdir(np.array([row.height]),
                                        np.array([int(1.)]), map_dx=1.,
                                        identifier=row.id, name=i,
                                        coords=(row.lat, row.lon), reset=reset)

            tasks.process_custom_climate_data(gdir)

            # Start with any params and then trim on accum rate later
            if mb_model is None:
                mb_model = BraithwaiteModel(gdir,  bias=0.,
                                            **BraithwaiteModel.cali_params_guess)
            else:
                if isinstance(mb_model, utils.SuperclassMeta):
                    try:
                        mb_model = mb_model(gdir,  bias=0.,
                                            **mb_model.cali_params_guess)
                    except FileNotFoundError:  # radiation missing
                        radiation.get_potential_irradiation_with_toposhade(gdir)
                        radiation.distribute_ipot_on_flowlines(gdir)
                else:
                    mb_model = copy.copy(mb_model)

            mb = []
            for date in mb_model.tspan_meteo:
                # Get the mass balance and convert to m per day
                tmp = mb_model.get_daily_mb(np.array([row.height]),
                                            date=date) * cfg.SEC_IN_DAY * \
                      cfg.RHO / cfg.RHO_W
                mb.append(tmp)

            mb_ds = xr.Dataset(
                {'MB': (['time', 'n', 'model'], np.atleast_3d(np.array(mb)))},
                coords={
                    'time': pd.to_datetime(
                        mb_model.tspan_meteo),
                    'n': (['n'], [1]),
                    'model': (['model'], [mb_model.__name__])},
                attrs={'id': gdir.rgi_id,
                       'name': gdir.name})
            mb_ds.mb.append_to_gdir(gdir, 'mb_daily')

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
    mean_mb = old_mb.map(np.nanmean)

    mean_acc = c_df.loc[c_df.id == gdir.rgi_id].mean_acc.values[0] / 365.25

    factor = mean_mb.map(lambda x: x / mean_acc)[
       'MB'].values
    new_mb = old_mb.map(lambda x: x / factor)

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
        mb_daily = gdirs[i].read_pickle('mb_daily_rescaled')
        begindate = dt.datetime(1961, 10, 1)

        # try to read dates
        this_core = core_meta[core_meta.id == gdirs[i].id]
        # todo: is iloc a bug here and we should use "loc" instead?
        y, m, d = this_core.date.iloc[0].split('-')
        # TODO: What do we actually assume if we have no info?
        try:
            end_date = dt.datetime(int(y), int(m), int(d))
        except ValueError:
            try:
                end_date = dt.datetime(int(y), int(m), 1)
            except ValueError:
                end_date = dt.datetime(int(y), 1, 1)

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
        try:
            cri_naninterp = np.interp(
                csite_cut_centers[1:][np.isnan(coverrho_interp)],
                csite_cut.depth[1:][~np.isnan(np.array(coverrho_interp))],
                np.array(coverrho_interp)[~np.isnan(np.array(coverrho_interp))])
        except ValueError:
            raise
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
        plt.savefig('~\\cali_results\\{}.png'
                .format((gdirs[i].name + '__' + str(x[0])).replace('.', '-')))
        plt.close()

    log.info(f"End round: {dt.datetime.now()}\n{snow_densify_kwargs} "
             f"{firn_densify_kwargs}")

    # weight errors with number of datapoints
    #error_weighted = np.average(errorlist, weights=n_datapoints)
    error_weighted = np.average(errorlist, weights=core_length)
    log.info(f"Errors for comparison: {np.nanmean(errorlist)} "
             f"{error_weighted}")
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


def crossvalidate_snowfirnmodel_calibration(
        to_calibrate, core_meta, snow_densify_method, firn_densify_method,
        temp_update_method, suffix=''):

    from sklearn.model_selection import LeavePOut
    X = np.arange(len(core_meta))
    leave_out = 3
    lpo = LeavePOut(leave_out)
    lpo.get_n_splits(X)

    # todo: this is double code!!!
    gdirs = []
    csites = []
    for name, rowdata in csite_df.iterrows():
        path = glob.glob(cfg.PATHS['firncore_dir'] + '\\processed\\{}_density.csv'.format(name))
        data = pd.read_csv(path[0])
        csites.append(data)

        base_dir = cfg.PATHS['working_dir'] + '\\per_glacier'
        gdirs.append(utils.GlacierDirectory(rowdata.id, base_dir=base_dir,
                                            reset=False))

    for train_index, test_index in lpo.split(X):
        log.info(f"TRAIN:{train_index}, TEST: {test_index}")
        X_train, X_test = core_meta.iloc[train_index], core_meta.iloc[test_index]

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

        log.info(f"{result.cost} {np.nansum(cv_result)**2}\n{result.x}")

        with (open(
                os.path.join(cfg.PATHS['firncore_dir'],
                f"firnmodel_cv_results_{suffix}_LO{str(leave_out)}.txt"), 'a')
        as f):
            f.write(f"Train: {X_train.__repr__()}\nTest: {X_test.__repr__()}\n"
                    f"{result.__repr__()}{cv_result.__repr__()}"
                    f"\n\n\n\n\n\n\n\n\n\n\n")


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


# THIS IS PRELIM OGGM STUFF
from crampon.utils import entity_task
from crampon.core.preprocessing import centerlines
def _fallback_mu_star_calibration(gdir):
    """A Fallback function if climate.mu_star_calibration raises an Error.
￼
￼	    This function will still read, expand and write a `local_mustar.json`,
    filled with NANs, if climate.mu_star_calibration fails
    and if cfg.PARAMS['continue_on_error'] = True.
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """
    # read json
    df = gdir.read_json('local_mustar')
    # add these keys which mu_star_calibration would add
    df['mu_star_per_flowline'] = [np.nan]
    df['mu_star_flowline_avg'] = np.nan
    df['mu_star_allsame'] = np.nan
    # write
    gdir.write_json(df, 'local_mustar')

def _recursive_mu_star_calibration(gdir, fls, t_star, first_call=True,
                                   force_mu=None):

    # Do we have a calving glacier? This is only for the first call!
    # The calving mass-balance is distributed over the valid tributaries of the
    # main line, i.e. bad tributaries are not considered for calving
    cmb = calving_mb(gdir) if first_call else 0.

    # Climate period
    mu_hp = int(cfg.PARAMS['mu_star_halfperiod'])
    yr_range = [t_star - mu_hp, t_star + mu_hp]

    # Get the corresponding mu
    heights = np.array([])
    widths = np.array([])
    for fl in fls:
        heights = np.append(heights, fl.surface_h)
        widths = np.append(widths, fl.widths)

    _, temp, prcp = mb_yearly_climate_on_height(gdir, heights,
                                                year_range=yr_range,
                                                flatten=False)

    if force_mu is None:
        try:
            mu_star = optimization.brentq(_mu_star_per_minimization,
                                          cfg.PARAMS['min_mu_star'],
                                          cfg.PARAMS['max_mu_star'],
                                          args=(fls, cmb, temp, prcp, widths),
                                          xtol=1e-5)
        except ValueError:
            # This happens in very rare cases
            _mu_lim = _mu_star_per_minimization(cfg.PARAMS['min_mu_star'],
                                                fls, cmb, temp, prcp, widths)
            if _mu_lim < 0 and np.allclose(_mu_lim, 0):
                mu_star = 0.
            else:
                raise MassBalanceCalibrationError('{} mu* out of specified '
                                                  'bounds.'.format(gdir.rgi_id)
                                                  )

        if not np.isfinite(mu_star):
            raise MassBalanceCalibrationError('{} '.format(gdir.rgi_id) +
                                              'has a non finite mu.')
    else:
        mu_star = force_mu

    # Reset flux
    for fl in fls:
        fl.flux = np.zeros(len(fl.surface_h))

    # Flowlines in order to be sure - start with first guess mu*
    for fl in fls:
        y, t, p = mb_yearly_climate_on_height(gdir, fl.surface_h,
                                              year_range=yr_range,
                                              flatten=False)
        mu = fl.mu_star if fl.mu_star_is_valid else mu_star
        fl.set_apparent_mb(np.mean(p, axis=1) - mu*np.mean(t, axis=1),
                           mu_star=mu)

    # Sometimes, low lying tributaries have a non-physically consistent
    # Mass-balance. These tributaries wouldn't exist with a single
    # glacier-wide mu*, and therefore need a specific calibration.
    # All other mus may be affected
    if cfg.PARAMS['correct_for_neg_flux']:
        if np.any([fl.flux_needs_correction for fl in fls]):

            # We start with the highest Strahler number that needs correction
            not_ok = np.array([fl.flux_needs_correction for fl in fls])
            fl = np.array(fls)[not_ok][-1]

            # And we take all its tributaries
            inflows = centerlines.line_inflows(fl)

            # We find a new mu for these in a recursive call
            # TODO: this is where a flux kwarg can passed to tributaries
            _recursive_mu_star_calibration(gdir, inflows, t_star,
                                           first_call=False)

            # At this stage we should be ok
            assert np.all([~ fl.flux_needs_correction for fl in inflows])
            for fl in inflows:
                fl.mu_star_is_valid = True

            # After the above are OK we have to recalibrate all below
            _recursive_mu_star_calibration(gdir, fls, t_star,
                                           first_call=first_call)

    # At this stage we are good
    for fl in fls:
        fl.mu_star_is_valid = True

@entity_task(log, writes=['inversion_flowlines'],
             fallback=_fallback_mu_star_calibration)
def mu_star_calibration(gdir):
    """Compute the flowlines' mu* and the associated apparent mass-balance.
    If low lying tributaries have a non-physically consistent Mass-balance
    this function will either filter them out or calibrate each flowline with a
    specific mu*. The latter is default and recommended.
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    """

    # Interpolated data
    df = gdir.read_json('local_mustar')
    t_star = df['t_star']
    bias = df['bias']

    # For each flowline compute the apparent MB
    fls = gdir.read_pickle('inversion_flowlines')
    # If someone calls the task a second time we need to reset this
    for fl in fls:
        fl.mu_star_is_valid = False

    force_mu = 0 if df['mu_star_glacierwide'] == 0 else None

    # Let's go
    _recursive_mu_star_calibration(gdir, fls, t_star, force_mu=force_mu)

    # If the user wants to filter the bad ones we remove them and start all
    # over again until all tributaries are physically consistent with one mu
    # This should only work if cfg.PARAMS['correct_for_neg_flux'] == False
    do_filter = [fl.flux_needs_correction for fl in fls]
    if cfg.PARAMS['filter_for_neg_flux'] and np.any(do_filter):
        assert not do_filter[-1]  # This should not happen
        # Keep only the good lines
        # TODO: this should use centerline.line_inflows for more efficiency!
        heads = [fl.orig_head for fl in fls if
                 not fl.flux_needs_correction]
        centerlines.compute_centerlines(gdir, heads=heads, reset=True)
        centerlines.initialize_flowlines(gdir, reset=True)
        if gdir.has_file('downstream_line'):
            centerlines.compute_downstream_line(gdir, reset=True)
            centerlines.compute_downstream_bedshape(gdir, reset=True)
        centerlines.catchment_area(gdir, reset=True)
        centerlines.catchment_intersections(gdir, reset=True)
        centerlines.catchment_width_geom(gdir, reset=True)
        centerlines.catchment_width_correction(gdir, reset=True)
        local_t_star(gdir, tstar=t_star, bias=bias, reset=True)
        # Ok, re-call ourselves
        return mu_star_calibration(gdir, reset=True)

    # Check and write
    rho = cfg.PARAMS['ice_density']
    aflux = fls[-1].flux[-1] * 1e-9 / rho * gdir.grid.dx ** 2
    # If not marine and a bit far from zero, warning
    cmb = calving_mb(gdir)
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=0.01):
        log.warning('(%s) flux should be zero, but is: '
                    '%.4f km3 ice yr-1', gdir.rgi_id, aflux)
    # If not marine and quite far from zero, error
    if cmb == 0 and not np.allclose(fls[-1].flux[-1], 0., atol=1):
        msg = ('({}) flux should be zero, but is: {:.4f} km3 ice yr-1'
               .format(gdir.rgi_id, aflux))
        #raise MassBalanceCalibrationError(msg)
        raise ValueError(msg)
    gdir.write_pickle(fls, 'inversion_flowlines')

    # Store diagnostics
    mus = []
    weights = []
    for fl in fls:
        mus.append(fl.mu_star)
        weights.append(np.sum(fl.widths))
    df['mu_star_per_flowline'] = mus
    df['mu_star_flowline_avg'] = np.average(mus, weights=weights)
    all_same = np.allclose(mus, mus[0], atol=1e-3)
    df['mu_star_allsame'] = all_same
    if all_same:
        if not np.allclose(df['mu_star_flowline_avg'],
                           df['mu_star_glacierwide'],
                           atol=1e-3):
            raise ValueError(
                'Unexpected difference between '
                'glacier wide mu* and the '
                'flowlines mu*.')
    # Write
    gdir.write_json(df, 'local_mustar')


def optimize_ensemble_size(gdir):
    """
    Optimize the ensemble size for predicting mass balance.

    Parameters
    ----------
    gdir

    Returns
    -------

    """
    from sklearn.model_selection import LeavePOut, ShuffleSplit
    from properscoring import crps_ensemble
    from collections import OrderedDict

    import pickle
    def save_obj(obj, fname):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_obj(fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    meas = get_measured_mb_glamos(gdir)
    mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]
    plist = [massbalance.ParameterGenerator(gdir, m).single_glacier_params for
             m in mb_models]

    cutoff = 30

    # (date0, date1, model, params): value
    cache_dict_path = 'C:\\users\\johannes\\desktop\\cache_dict.pkl'
    try:
        cache_dict = load_obj(cache_dict_path)
    except:
        cache_dict = {}

    score_per_ens_size_path = 'C:\\users\\johannes\\desktop\\score_dict.pkl'

    if cutoff is not None:
        meas = meas.tail(cutoff)
        plist = [pdf.tail(cutoff).values for pdf in plist]

    heights, widths = gdir.get_inversion_flowline_hw()
    sc = gdir.read_pickle('snow_daily')

    #error_per_n = []
    score_per_ens_size = {key: [] for key in np.arange(1, len(meas) - 1)}
    for n in range(len(meas)):
        # drop nth year from the series
        meas_drop = meas.drop(meas.index[n])
        test_value = meas.iloc[n].Annual
        test_d0 = meas.iloc[n].date0
        test_d1 = meas.iloc[n].date1
        log.info(f"Dropped {str(n)}th row from measurement dataframe.")

        # error_per_k = []
        # perform LKO for all possible ks on the remaining years
        for k in range(1, len(meas_drop)):
            error_per_k = []
            log.info(f"Performing ShuffleSplit with train size {k} on the "
                     f"remaining measurement frame.")

            # get shuffled splits and limit them in size
            lpo = LeavePOut(len(meas_drop)-k)
            n_splits_lpo = lpo.get_n_splits(meas_drop.values)
            #max_splits = np.min([n_splits_lpo, 5000])
            max_splits = 5000

            shufsplit = ShuffleSplit(n_splits=max_splits, train_size=k, random_state=0)
            log.info(f"N SPLITS: {shufsplit.get_n_splits(meas_drop.values)}")
            #for train_index, test_index in lpo.split(meas_drop.values):
            for z, (train_index, _) in enumerate(shufsplit.split(meas_drop.values)):
                #test_meas = meas_drop.iloc[test_index]
                #train_meas = meas_drop.iloc[train_index]
                if z % 500 == 0:
                    log.info(str(z))

                date_range = pd.date_range(test_d0, test_d1)

                prediction_score = []
                #for tc, tm in enumerate(train_meas.index):
                #    print('Calculating {}th test case'.format(tc))
                    #date_range = pd.date_range(test_meas.ix[tm].date0,
                    #                           test_meas.ix[tm].date1)

                mb_predicted_ensemble = []
                for i, m in enumerate(mb_models):

                    modelparams = plist[i]

                    for ti in train_index:
                        try:
                            pdict = dict(zip(m.cali_params_list, modelparams[ti]))
                        except IndexError:  # one of the short mass balance models radiation-based
                            continue

                        # date0, date1, model, param_dict
                        p_ordered_vals = list(OrderedDict(sorted(pdict.items())).values())
                        if (date_range[0], date_range[-1], m, *p_ordered_vals) in cache_dict.keys():
                            mb_predicted_ensemble.append(cache_dict[(date_range[0], date_range[-1], m, *p_ordered_vals)])
                            continue

                        sc_in = SnowFirnCover.from_dataset(sc.sel(time=date_range[0], model=m.__name__))
                        dm = m(gdir, **pdict, bias=0., snowcover=sc_in)

                        mb = []
                        for date in date_range:
                            # Get the mass balance and convert to m per day
                            tmp = dm.get_daily_specific_mb(heights, widths, date=date)
                            mb.append(tmp)
                        mb_predicted_ensemble.append((np.nansum(mb)))
                        if (date_range[0], date_range[-1], m, *p_ordered_vals) not in cache_dict.keys():
                            cache_dict[(date_range[0], date_range[-1], m, *p_ordered_vals)] = np.nansum(mb)
                        #print('mb_predicted_ensemble: ', mb_predicted_ensemble)
                        #print('measured: ', test_value)
                    #prediction_score.append(np.nanmean(np.abs(np.array(mb_predicted_ensemble) - test_meas.ix[tm].Annual)))
                    #prediction_score.extend(np.abs(
                    #    np.array(mb_predicted_ensemble) - test_value))
                mpe_array = np.array(mb_predicted_ensemble)
                #prediction_score.append(mean_absolute_error(np.full_like(mpe_array, test_value), mpe_array))
                #prediction_score.append(np.abs(np.nanmedian(mpe_array) - test_value))
                prediction_score.append(crps_ensemble(test_value, mpe_array))
                #print('prediction_score: ', np.nanmean(prediction_score))
            #error_per_k.append(prediction_score)
            log.info(f"Prediction score for k={k}: {prediction_score}")
            extended = score_per_ens_size[k]
            extended.extend(prediction_score)
            score_per_ens_size[k] = extended
        #error_per_n.append(error_per_k)
        log.info(f"Prediction per ensemble size: {score_per_ens_size}")
        save_obj(cache_dict, cache_dict_path)
        save_obj(score_per_ens_size, score_per_ens_size_path)
    return score_per_ens_size


def calibrate_cam_glaciers_on_point_mass_balances(point_data_path, out_path):
    """
    For experiments, we pretend that each camera glacier has one
    intermediate measurement that we calibrate on.

    Parameters
    ----------
    point_data_path : str
        Path to the file with the point measurements on the three glaciers.
    out_path : str
        Path to which the dataframe with the calibrated parameters shall be
        written.

    Returns
    -------

    """
    # point_data_path = '...\\crampon\\data\\MB\\point\\cam_point_MBs_Matthias.csv'
    pdata = pd.read_csv(point_data_path, index_col=0, parse_dates=[4, 5])
    results = pd.DataFrame(columns=['RGIId', 'stations'])
    stations_per_glacier = {
        'RGI50-11.A55F03': [1003],
        'RGI50-11.B4312n-1': [1002, 1006, 1007, 1009],
        'RGI50-11.B5616n-1': [1001, 1008]}
    mb_models = [massbalance.BraithwaiteModel, massbalance.HockModel,
                 massbalance.PellicciottiModel,
                 massbalance.OerlemansModel]
    it = 0
    for k, v in stations_per_glacier.items():
        gdir = utils.GlacierDirectory(k)
        for comb_length in range(1, len(v) + 1):
            for combo in itertools.combinations(v, comb_length):
                log.info(f"{combo}")
                pdata_sel = pdata.loc[[c for c in combo]]
                log.info(f"{pdata_sel}")
                for model in mb_models:
                    log.info(model.__name__)
                    cali_result = calibrate_on_summer_point_mass_balance(
                        gdir,
                        pdata_sel.date0.values,
                        pdata_sel.date_p.values,
                        pdata_sel.z.values,
                        pdata_sel.bp.values,
                        list(pdata_sel.otype.values),
                        model)
                    results.loc[it, 'RGIId'] = k
                    results.loc[it, 'stations'] = str(combo)
                    for rk, rv in dict(cali_result).items():
                        results.loc[it, model.__name__ + '_' + rk] = rv
                it += 1
                log.info(f"ITERATION NUMBER: {it}")
    results.to_csv(out_path)
