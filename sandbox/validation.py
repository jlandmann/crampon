from crampon import workflow
from crampon import utils
from crampon import cfg
from crampon.core.preprocessing import calibration
from crampon.core.models.massbalance import MassBalance, BraithwaiteModel, \
    PellicciottiModel, OerlemansModel
import geopandas as gpd
import pandas as pd
import datetime as dt
import numpy as np
import xarray as xr
import copy
from itertools import product


def read_mh_daily_mb(path):
    data = pd.read_csv(path, delim_whitespace=True)
    data[['B(mm_w.e.)', 'Acc(mm_w.e.)', 'Melt(mm_w.e.)']] /= 1000.
    return data


def make_hindcast(g, mb_model, begin, end, snowcover=None, time_elapsed=None,
                  max_pcombs=200, latest_climate=False, from_param_pairs=True,
                  apply_func=None):
    timespan = pd.date_range(begin, end, freq='D')
    if time_elapsed is None:
        time_elapsed = timespan[0] - dt.timedelta(days=1)

    # get parameters and maybe modify them a bit
    param_prod = np.array(list(
        utils.get_possible_parameters_from_past(g, mb_model,
                                                latest_climate=latest_climate,
                                                only_pairs=from_param_pairs)))

    # we sometimes want to test against a prediction with the mean as well
    if apply_func:
        param_prod = np.apply_along_axis(apply_func, axis=0, arr=param_prod)
        param_prod = np.atleast_2d(param_prod) # make it iterable again
    else:
        if len(param_prod) >= max_pcombs:
            param_prod = param_prod[np.random.randint(0, high=len(param_prod),
                                                      size=max_pcombs)]
        else:
            print('Parameter combinations are less than maximum set. Taking '
                  'all available.')

    stacked = None

    heights, widths = g.get_inversion_flowline_hw()
    for it, params in enumerate(param_prod):

        pdict = dict(zip(mb_model.cali_params_list, params))
        print('parameter combination no. {}'.format(it), pdict)

        day_model_curr = mb_model(g, **pdict, bias=0., snowcover=snowcover)
        day_model_curr.time_elapsed = time_elapsed

        mb_now = []
        for date in timespan:
            # Get the mass balance and convert to m per day
            tmp = day_model_curr.get_daily_specific_mb(heights, widths,
                                                       date=date)
            mb_now.append(tmp)

        if stacked is not None:
            stacked = np.vstack((stacked, mb_now))
        else:
            stacked = np.atleast_2d(mb_now)

    # might happen that quantiles are messed up
    stacked = np.sort(stacked, axis=0)

    mb_for_ds = np.atleast_2d(stacked).T
    var_dict = {**{mb_model.mb_name: (['time', 'n'], mb_for_ds)},
                **dict(zip(g.get_calibration(mb_model).columns.values,
                           [(['n'], np.array(p)) for p in
                            zip(*list(param_prod))]))}
    mb_cs = xr.Dataset(var_dict,
                       coords={'n': (['n'], np.arange(mb_for_ds.shape[1])),
                               'time': pd.to_datetime(timespan)},
                       attrs={'id': g.rgi_id, 'name': g.name})
    return mb_cs


def hindcast_winter_and_annual_massbalance(g, mb_model, max_pcombs=200,
                                           repeat=0, latest_climate=False,
                                           only_param_pairs=True,
                                           apply_func=None):
    glamos_mb = calibration.get_measured_mb_glamos(g)

    if hasattr(mb_model, 'calibration_timespan'):
        if mb_model.calibration_timespan[0]:
            glamos_mb = glamos_mb[
                glamos_mb.date0.dt.year >= mb_model.calibration_timespan[0]]
        if mb_model.calibration_timespan[1]:
            glamos_mb = glamos_mb[
                glamos_mb.date1.dt.year < mb_model.calibration_timespan[1]]
    # very important: reset index to exclude index gaps
    glamos_mb.reset_index(drop=True, inplace=True)

    # if we say so, take only recent 30 years
    if latest_climate:
        glamos_mb = glamos_mb[-30:]

    pred_winter = []
    pred_annual = []

    # relevant dates from the mb file
    relevant_dates = glamos_mb[
        ['date0', 'date1', 'date_f', 'date_s']].values.flatten()

    # prepare the snowcover and elapsed time input
    dm = mb_model(g, bias=0.)
    heights, widths = g.get_inversion_flowline_hw()

    if (hasattr(dm, 'calibration_timespan')) and (dm.calibration_timespan[0] is
                                                  not None):
            model_begin = dt.datetime(dm.calibration_timespan[0], 1, 1)
    else:
        model_begin = np.max([np.min(relevant_dates), dt.datetime(1961, 1, 1)])

    # generate "true" snow cover and elapsed time at field/minimum dates
    entire_span = pd.date_range(model_begin, glamos_mb.iloc[-1].date1)

    sc_dict = {}
    te_dict = {}

    print('Preparing snowcover and elapsed time....')
    for date in entire_span:
        dm.get_daily_specific_mb(heights, widths, date=date)
        if date in relevant_dates:
            print(date)
            sc_dict[date] = copy.deepcopy(dm.snowcover)
            te_dict[date] = copy.deepcopy(dm.time_elapsed)

    out_df = pd.DataFrame()
    model_str = mb_model.prefix.split('_')[0]
    out_times = np.unique([d.year for d in relevant_dates])[1:]
    out_members = np.arange(repeat + 1)
    var_dummy = np.empty((len(out_times), len(out_members), 1))
    var_dummy.fill(np.nan)
    time_dummy = np.empty((len(out_times)))
    time_dummy.fill(np.nan)

    out_ds = xr.Dataset({'BW_pred': (['time', 'member', 'model'],
                                     var_dummy.copy()),
                         'BA_pred': (['time', 'member', 'model'],
                                     var_dummy.copy()),
                         'BW': (['time', ], time_dummy.copy()),
                         'BA': (['time', ], time_dummy.copy())},
                        coords={'time': out_times,
                                'member': (['member', ], out_members),
                                'model': (['model', ], [model_str])})

    # start looping over the measured mass balance
    for i, row in glamos_mb.iterrows():

        # to make a nice pandas df
        pred_winter_per_year = []
        pred_annual_per_year = []

        # determine start and end date
        start_date = min(row.date0, row.date_f)
        if i < max(glamos_mb.index):
            # max(field & fall date)
            end_date = max(row.date1, glamos_mb.loc[i + 1].date_f)
        else:  # last row
            end_date = row.date1

        for r in range(-1, repeat):
            print(r, 'TH ITERATION')
            mb_predict = make_hindcast(g, mb_model, start_date,
                                       end_date,
                                       time_elapsed=te_dict[start_date],
                                       snowcover=sc_dict[start_date],
                                       max_pcombs=max_pcombs,
                                       latest_climate=latest_climate,
                                       from_param_pairs=only_param_pairs,
                                       apply_func=apply_func)

            # to be sure, we subtract the minimum
            mb_w = mb_predict.sel(dict(time=slice(row.date_f, row.date_s)))
            mb_a = mb_predict.sel(dict(time=slice(row.date0, row.date1)))

            mb_w_cs = MassBalance.time_cumsum(mb_w)
            mb_a_cs = MassBalance.time_cumsum(mb_a)

            minimum = mb_w_cs[mb_model.mb_name].min(dim='time', skipna=True)
            minimum = minimum.where(minimum < 0., 0.)  # could be weird minimum
            print('corrected for minimum {}'.format(str(minimum.values)))
            # select winter and annual periods as used in the calibration
            # winter date_f - date_s, annual date0 - date1
            wb_predicted = (mb_w_cs[mb_model.mb_name].isel(
                time=-1) - minimum).median(skipna=True)
            ab_predicted = mb_a_cs[mb_model.mb_name].isel(time=-1).median(
                skipna=True)

            print(wb_predicted.item(), row.Winter, '    ',
                  ab_predicted.item(), row.Annual)

            w_err = np.abs(wb_predicted.item() - row.Winter)
            a_err = np.abs(ab_predicted.item() - row.Annual)
            print('err_w', w_err, '    ', 'err_a', a_err)

            pred_winter.append(wb_predicted)
            pred_annual.append(ab_predicted)
            pred_winter_per_year.append(wb_predicted.item())
            pred_annual_per_year.append(ab_predicted.item())

        print(len(pred_winter), len(pred_annual))

        out_ds['BW_pred'].loc[
            dict(time=end_date.year, model=model_str)] = pred_winter_per_year
        out_ds['BA_pred'].loc[
            dict(time=end_date.year, model=model_str)] = pred_annual_per_year
        out_ds['BW'].loc[dict(time=end_date.year)] = row.Winter
        out_ds['BA'].loc[dict(time=end_date.year)] = row.Annual

    print(dt.datetime.now())
    print(pred_winter, pred_annual)
    return out_ds, (pred_winter, pred_annual)
