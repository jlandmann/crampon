from crampon import workflow
from crampon import utils
from crampon import cfg
from crampon.core.preprocessing import calibration, gis, climate
from crampon.core.models.massbalance import MassBalance, BraithwaiteModel, \
    PellicciottiModel, OerlemansModel, HockModel
from crampon.core import holfuytools
from crampon.core.models import massbalance
import matplotlib.pyplot as plt
from crampon.core.holfuytools import prepare_holfuy_camera_readings
from crampon.core.models.massbalance import get_melt_percentiles, \
    extrapolate_melt_percentiles, infer_current_mb_from_melt_percentiles
from operational import mb_production
import geopandas as gpd
import pandas as pd
import datetime as dt
import numpy as np
import xarray as xr
import copy
import os
import pickle
from itertools import product
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut


def read_mh_daily_mb(path):
    """
    Read file with daily mass balances from Matthias Huss.

    Parameters
    ----------
    path : str
        Path to file.

    Returns
    -------
    data: pd.Dataframe
        dataframe with matthias' mass balances.
    """
    data = pd.read_csv(path, delim_whitespace=True)
    data[['B(mm_w.e.)', 'Acc(mm_w.e.)', 'Melt(mm_w.e.)']] /= 1000.
    return data


def make_hindcast(g, mb_model, begin, end, snowcover=None, time_elapsed=None,
                  max_pcombs=200, latest_climate=False, from_param_pairs=True,
                  apply_func=None):
    """
    Make a mass balance hindcast for a specific glacier.

    Parameters
    ----------
    g : py:class:`crampon.GlacierDirectory`
        The GlacierDirectors to be processed.
    mb_model : crampon.core.models.massbalance.DailyMassBalanceModel
        The mass balance mdoel to use.
    begin : pd.Timestamp
        Date when to begin the hindcast.
    end : pd.Timestamp
        Date when to end the hindcast.
    snowcover : crampon.core.model.massbalance.SnowFirnCover or None
        Glacier snow cover to use at model initialization. Default: None (use
        standard initalization in MB model).
    time_elapsed : pd.DatetimeIndex or None
        Time elapsed at the model initalization (important for setting up the
        model instance). Default: None (no time elapsed).
    max_pcombs : int, optional
        Maximum number of parameter combinations to use. Default: 200.
    latest_climate : bool, optional
        Use only the latest glacier climate (30 years) as a basis for parameter
        choice. Default: False (use all available).
    from_param_pairs : bool, optional
        Take only parameter pairs (as calibrated) for the hindcast. Default:
        True.
    apply_func : aggregation function or None, optional
        Apply a an aggregation function to the parameters first, e.g. `np.mean`
        or `np.median`. Default: None (do not apply function).

    Returns
    -------
    mb_cs: xr.Dataset
        Dataset containing the mass balance hindcast.
    """

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
        param_prod = np.atleast_2d(param_prod)  # make it iterable again
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
    """
    Hindcast both winter and annual mass balance.

    Parameters
    ----------
    g : py:class:`crampon.GlacierDirectory`
        The GlacierDirectors to be processed.
    mb_model : crampon.core.models.massbalance.DailyMassBalanceModel
        The mass balance mdoel to use.
    max_pcombs : int, optional
        Maximum number of parameter combinations to use. Default: 200.
    repeat: int, optional
        todo: insert documentation here
    latest_climate : bool, optional
        Use only the latest glacier climate (30 years) as a basis for parameter
        choice. Default: False (use all available).
    only_param_pairs : bool, optional
        Take only parameter pairs (as calibrated) for the hindcast. Default:
        True.
    apply_func : aggregation function or None, optional
        Apply a an aggregation function to the parameters first, e.g. `np.mean`
        or `np.median`. Default: None (do not apply function).

    Returns
    -------

    """
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


class ParameterPredictor(object):
    """
    Interface to a prediction for mass balance model parameters.
    """
    def __init__(self, mb_model):
        """
        Instantiate.

        Parameters
        ----------
        mb_model: crampon.core.models.massbalance.DailyMassBalanceModel
            The mass balance model to predict the parameters for.
        """

        self.mb_model = mb_model
        self.predictands = mb_model.cali_params_list

    def train(self):
        """
        Train the prediction model.

        Returns
        -------

        """

    def predict(self, gdir):
        """

        Parameters
        ----------
        gdir: `py:class:crampon.GlacierDirectory`
            GlacierDirectory to predict parameters for.

        Returns
        -------
        None
        """

    def save_model(self):
        """

        Returns
        -------

        """

    @staticmethod
    def from_file(path):
        """

        Returns
        -------
        An instance of the ParameterPredictor class.
        """


def make_param_random_forest(model, base_dir, mb_glaciers=None, write=True,
                             return_important=False):
    """
    Create a random forest model from calibrated parameters.

    The predictor are the temperature sum (above the melt threshold),
    solid precipitation sum, mean albedo in the time span, mean slope, mean
    exposition, minimum height, maximum height and height range.

    # todo: shall we do a PCA to reduce the number of predictors?

    Parameters
    ----------
    model: a crampon massbalance model
        The model for which parameters should be predicted.
    base_dir: str
        Path to the base_dir of a model run from which to generate the random
        forest (should be ending on 'per_glacier').
    mb_glaciers: list of str or None
        List with thode IDs that have a direct mass balance observation which
        can be used for training the regressor. If None is given, all GLAMOS
        IDs (except those that don't work) will be used. Default: None.
    write: bool
        Whether or not output model shall be written. Default: True.
    return_important: bool
        Whether to return the important features names and the model trained
        with the important features only. Default: False (return all feature
        names and the model trained on all features).

    Returns
    -------
    feature_list, sklearn.ensemble.RandomForestRegressor: list, obj
        List of features names used for training and the fitted
        RandomForestRegressor model.
    """
    # get all ever predicted parameters
    if mb_glaciers is None:
        mb_glaciers = ['RGI50-11.B4312n-1', 'RGI50-11.A55F03',
                       'RGI50-11.B5616n-1', 'RGI50-11.A10G05',
                       'RGI50-11.B4504', 'RGI50-11.C1410',
                       'RGI50-11.B2201', 'RGI50-11.B1601', 'RGI50-11.A50D01',
                       'RGI50-11.E2320n', 'RGI50-11.E2316',
                       'RGI50-11.B5614n', 'RGI50-11.A51E08',
                       'RGI50-11.B3626-1', 'RGI50-11.B5232', 'RGI50-11.A50I06',
                       # 'RGI50-11.A51E12',  # St. Anna
                       #'RGI50-11.B5263n', 'RGI50-11.B5229',
                        'RGI50-11.A50I19-4',  # Clariden
                       # 'RGI50-11.A50I07-1',  # Plattalva
                       ]
    mb_gdirs = [utils.GlacierDirectory(g, base_dir=base_dir) for g in
                mb_glaciers]

    # get some statistics
    workflow.execute_entity_task(gis.simple_glacier_masks, mb_gdirs)

    # add slope and exposition
    df_all = pd.DataFrame()
    for gdir in mb_gdirs:
        hypso = pd.read_csv(gdir.get_filepath('hypsometry'))
        hypso['Zrange'] = hypso['Zmax'] - hypso['Zmin']

        # todo: with the hypsometry we also get the altitude-area distribution:
        #  make a linear fit of it and take the parameters as predictors
        hypso = hypso[['RGIId', 'Zmin', 'Zmax', 'Zmed', 'Area', 'Slope',
                       'Aspect']]
        print(gdir.rgi_id, model.__name__)
        try:
            cali = gdir.get_calibration(mb_model=model)
        except FileNotFoundError:
            continue

        cali['year'] = [pd.Timestamp(t).year if pd.Timestamp(t).month <= 9
                        else pd.Timestamp(t).year+1 for t in cali.index.values]
        # todo: get temperature above melt thresh and precip sum
        gmeteo = climate.GlacierMeteo(gdir)
        h, w = gdir.get_inversion_flowline_hw()
        for y in pd.unique(cali.year.values)[1:-1]:
            sub = cali[cali.year == y]
            drange = pd.date_range(min(sub.index), max(sub.index))
            # todo: add winter precip sum (Sevruk 1985)
            tsum = 0.
            psum = 0.
            # todo: take only summer sissum here (or when T gt 0)!!!!
            sissum = 0.
            for d in drange:
                tsum += np.average(
                    np.clip(gmeteo.get_tmean_at_heights(d, heights=h), 0,
                            None), weights=w)

                # we take the uncorrected precip sum (we don't know the
                # correction factor in advance for the prediction case!)
                psol, _ = gmeteo.get_precipitation_solid_liquid(d, heights=h)
                psum += np.average(psol, weights=w)
                if model.__name__ in ['PellicciottiModel', 'OerlemansModel']:
                    sissum += gmeteo.meteo.sis.sel(time=d)

            cali.loc[cali.year == y, 'tsum'] = tsum
            cali.loc[cali.year == y, 'psum'] = psum
            if model.__name__ in ['PellicciottiModel', 'OerlemansModel']:
                cali.loc[cali.year == y, 'sissum'] = sissum

        cali.drop_duplicates(keep='last', subset='tsum').dropna()
        # take also the MB year as a predictor
        cali['mbyear'] = cali['year']
        cali.set_index('year', inplace=True)
        # todo: shorten the dataframe here
        # cali = cali[cali.index >= 1987]

        # if it's Hock, take also mean glacier ipot as predictor
        # todo: split up by time (ipot during summer, in the ablation area)?
        if model.__name__ == 'HockModel':
            ipot = gdir.read_pickle('ipot_per_flowline')
            cali['ipot'] = np.average(np.mean(np.vstack(ipot), axis=1),
                                      weights=w)

        # todo: get mean albedo (albedo distribution!? albedo "sum"?) per year!

        # duplicate rows of hypsometry (prepare for merge)
        # todo: once glacier flow, this needs to be updated
        hypso = hypso.append([hypso] * (len(cali) - 1), ignore_index=True)
        hypso['year'] = cali.index.values.copy()
        hypso.set_index('year', inplace=True)

        # todo: shorten the dataframe here
        # hypso = hypso[hypso.index >= 1987]

        # merge cali and hypso
        merged = cali.join(hypso)

        if df_all.empty:
            df_all = merged.copy()
        else:
            df_all = pd.concat([df_all, merged])

    df_all = df_all.drop_duplicates().dropna()
    print(df_all.columns.values)

    # extract Labels and
    # Labels are the values we want to predict
    labels = np.array(
        df_all[[model.prefix + p for p in model.cali_params_list]])
    # Remove the labels from the features
    features = df_all.drop(
        [model.prefix + p for p in model.cali_params_list] + ['RGIId'], axis=1)
    # Saving feature names for later use
    feature_list = list(features.columns)
    print(feature_list)
    print(features.shape, labels.shape)

    # Convert to numpy array
    features = np.array(features)
    print(features[0, :])

    # apply the model
    train_features, test_features, train_labels, test_labels = \
        train_test_split(features, labels, test_size=1, random_state=42)
    print('features: ', train_features.shape, test_features.shape)
    print('label', train_labels.shape, test_labels.shape)
    accuracy_list = []
    """
    loo = LeaveOneOut()
    for train_ix, test_ix in loo.split(features):
        print("%s %s" % (train_ix.shape, test_ix.shape))
        train_features, test_features, train_labels, test_labels = 
        features[train_ix], features[test_ix], labels[train_ix], 
        labels[test_ix]

        rf = RandomForestRegressor(n_estimators=1000, random_state=42)
        rf.fit(train_features, train_labels)
    
        # The baseline predictions are the 'standard parameters'
        baseline_preds = np.array(list(model.cali_params_guess.values())[1])
        # Baseline errors, and display average baseline error
        #print(baseline_preds, test_labels)
        baseline_errors = abs(baseline_preds - test_labels)
        print(baseline_errors.shape)

        print('Average baseline error: ', baseline_errors)

        # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)

        # Calculate the absolute errors
        errors = abs(predictions - test_labels)
        print(predictions, test_labels)

        # Print out the mean absolute error (mae)
        print('Mean Absolute Error:', errors)

        # Calculate mean absolute percentage error (MAPE)
        mape = 100 * abs((errors / test_labels))
        print('mape', mape)

        # Calculate and display accuracy
        accuracy = 100 - mape
        print('acc.', accuracy)
        print('Accuracy:', round(np.mean(accuracy), 2), '%.')
        accuracy_list.append(round(np.mean(accuracy), 2))
    """

    # make one training with all and save the model:
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf.fit(features, labels)
    if write is True:
        rf_fname = os.path.join(cfg.PATHS['working_dir'],
                                'randomforest_{}.pkl'.format(
                                    model.__name__))
        pickle.dump((feature_list, rf), open(rf_fname, 'wb'))

    print(accuracy_list, np.mean(accuracy_list))
    # Get numerical feature importances
    importances = list(rf.feature_importances_)

    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for
                           feature, importance in
                           zip(feature_list, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1],
                                 reverse=True)

    importance_threshold = 0.9
    sorted_importance = np.array([i for v, i in feature_importances])
    sorted_features = np.array([v for v, i in feature_importances])
    important_until_ix = np.argmax(np.cumsum(np.array(sorted_importance)) > importance_threshold)

    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in
     feature_importances]

    # New random forest with only the two most important variables
    rf_most_important = RandomForestRegressor(n_estimators=1000,
                                              random_state=42)

    # Extract the two most important features
    important_features = sorted_features[:important_until_ix+1]
    print('IMPORTANT FEATURES: ', important_features)
    important_indices = [feature_list.index(f) for f in important_features]
    train_important = train_features[:, important_indices]
    test_important = test_features[:, important_indices]

    # Train the random forest
    rf_most_important.fit(train_important, train_labels)

    if write is True:
        rf_important_fname = \
            os.path.join(cfg.PATHS['working_dir'],
                         'randomforest_important_{}.pkl'.format(
                             model.__name__))
        pickle.dump((important_features, rf_most_important),
                    open(rf_important_fname, 'wb'))

    # todo: something is wrong here: we should do the same LOO analysis as we
    #  did above, but with most important
    # Make predictions and determine the error
    predictions = rf_most_important.predict(test_important)

    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    print(predictions, test_labels)

    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', errors)

    # Calculate mean absolute percentage error (MAPE)
    mape = 100 * abs((errors / test_labels))
    print('mape', mape)

    # Calculate and display accuracy
    accuracy = 100 - mape
    print('acc.', accuracy)
    print('Accuracy:', np.round(np.mean(accuracy), 2), '%.')

    if return_important is True:
        return important_features, rf_most_important
    else:
        return feature_list, rf


def predict_params_with_random_forest(gdir):
    """
    Take a trained random forest model and predict parameters for a glacier

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to predict the parameters for.

    Returns
    -------

    """
    # glacier meteo
    gmeteo = climate.GlacierMeteo(gdir)
    # todo: change this to changing geometries
    h, w = gdir.get_inversion_flowline_hw()

    # just to be sure
    gis.simple_glacier_masks(gdir)
    # todo: change this to changing geoemtries
    hypso = pd.read_csv(gdir.get_filepath('hypsometry'))
    hypso['Zrange'] = hypso['Zmax'] - hypso['Zmin']

    # todo: with the hypsometry we also get the altitude-area distribution:
    #  make a linear fit of it and take the parameters as predictors
    hypso = hypso[['RGIId', 'Zmin', 'Zmax', 'Zmed', 'Area', 'Slope',
                   'Aspect']]
    cali_df = pd.DataFrame(index=pd.date_range('1961-10-01', '2018-09-30'))

    for mbm in [eval(m) for m in cfg.MASSBALANCE_MODELS]:
        print(mbm.__name__)
        if hasattr(mbm, 'calibration_timespan'):
            if mbm.calibration_timespan[0]:
                cali_df = cali_df[cali_df.index.year >=
                                  mbm.calibration_timespan[0]]
            if mbm.calibration_timespan[1]:
                cali_df = cali_df[cali_df.index.year <=
                                  mbm.calibration_timespan[1]]
        print(cali_df)
        hyears = pd.DataFrame(
            data=[t.year if t.month < cfg.PARAMS[
                'begin_mbyear_month'] else t.year + 1 for t in cali_df.index],
            columns=['hyear'])
        pred_model_name = os.path.join(cfg.PATHS['working_dir'],
                                       'randomforest_{}.pkl'.format(
                                           mbm.__name__))
        with open(pred_model_name, 'rb') as pickle_file:
            (rf_features, rf_model) = pickle.load(pickle_file)
        print(rf_model)
        for i, group in list(cali_df.groupby(hyears.hyear.values)):
            data = group.copy()
            try:
                tsum = 0.
                psum = 0.
                drange = pd.date_range(min(data.index), max(data.index))
                for d in drange:
                    tsum += np.average(
                        gmeteo.get_tmean_at_heights(d, heights=h),
                        weights=w)
                    # we take the uncorrected precip sum (we don't know the
                    # correction factor in advance for the prediction case!)
                    psol, _ = \
                        gmeteo.get_precipitation_solid_liquid(d, heights=h)
                    psum += np.average(psol, weights=w)
                hypso.loc[i, 'tsum'] = tsum
                hypso.loc[i, 'psum'] = psum
                params = rf_model.predict(hypso[['Zmin', 'Zmax', 'Zmed',
                                                 'Area', 'Slope', 'Aspect',
                                                 'tsum', 'psum']])
                print(hypso[['Zmin', 'Zmax', 'Zmed', 'Area', 'Slope', 'Aspect',
                             'tsum', 'psum']])
                for i, p in enumerate(mbm.cali_params_list):
                    cali_df.loc[drange[0]:drange[1], mbm.prefix + p] = \
                        params[i]
            except:
                raise
    print(cali_df.drop_duplicates().dropna(axis=1))
    print('stop')


def cross_validate_percentile_extrapolation(mb_suffix=''):
    """
    Cross-validate the percentile extrapolation among the camera glaciers.

    Parameters
    ----------
    mb_suffix: str
        Suffix for the mass balance file, e.g. '_fischer_unique' for the mass
        balance from calibration on the Fischer geodetic mass balances.
        Default: ''.

    Returns
    -------

    """

    # todo: remove hard-coded stuff
    evaluate_begin = '2018-10-01'
    evaluate_end = '2019-09-30'
    clim_ref_period = (None, None)
    pctlrange = np.arange(101)
    n_samples = 1000
    base_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')
    rgd = utils.GlacierDirectory('RGI50-11.B4312n-1', base_dir=base_dir)
    fgd = utils.GlacierDirectory('RGI50-11.B5616n-test', base_dir=base_dir)
    pgd = utils.GlacierDirectory('RGI50-11.A55F03', base_dir=base_dir)
    glaciers = np.array([rgd, pgd, fgd])
    x = np.arange(glaciers.size)
    leave_out = 1
    lpo = LeavePOut(leave_out)
    lpo.get_n_splits(x)

    for train_index, test_index in lpo.split(x):
        train, test = glaciers[train_index], glaciers[test_index][0]

        obs = [holfuytools.prepare_holfuy_camera_readings(g) for g in train]
        conc = xr.merge(obs)
        first_date_assim = pd.Timestamp(conc.date.values[0])
        last_date_assim = min(pd.Timestamp(evaluate_end),
                              pd.Timestamp(conc.date.values[-1]))

        # infer melt from pctl extrapolation in assimilation time span
        try:
            train_pctls = massbalance.get_melt_percentiles(
                train, last_date_assim, mbclim_suffix=mb_suffix)
        except FileNotFoundError:
            train_pctls = massbalance.get_melt_percentiles(
                train, last_date_assim)
        extrapolated_pctls = massbalance.extrapolate_melt_percentiles(
            train_pctls, np.arange(500000, 800000, 500),
            np.arange(60000, 190000, 500))
        try:
            current_melt = \
                massbalance.infer_current_mb_from_melt_percentiles(
                    [test], extrapolated_pctls, last_date_assim,
                    date_range_obs=pd.date_range(first_date_assim,
                                                 last_date_assim),
                    mbclim_suffix=mb_suffix)
        # temporary exception when not all Fischer calibrations are ready yet
        except FileNotFoundError:
            current_melt = \
                massbalance.infer_current_mb_from_melt_percentiles(
                    [test], extrapolated_pctls, last_date_assim,
                    date_range_obs=pd.date_range(first_date_assim,
                                                 last_date_assim))
        extrap_pctls_melt = np.random.choice(
            np.percentile(current_melt, pctlrange), n_samples)

        # get model melt in assimilation time span => it must be mb_current,
        # because usually we don't have assimilated mass balances
        try:
            mod_all = test.read_pickle('mb_current' + mb_suffix)
        # temporary exception when not all Fischer calibrations are ready yet
        except:
            mod_all = test.read_pickle('mb_current')
        model_mb = mod_all.sel(time=slice(first_date_assim, last_date_assim))
        model_melt = model_mb.where(model_mb.MB < 0.)
        model_melt = model_melt.cumsum(dim='time').isel(time=-1).MB.values
        print('model melt:', np.median(model_melt))

        # get model accum. in assimilation time span (for reassembling later)
        model_accum = model_mb.where(model_mb.MB > 0.)
        model_accum = model_accum.cumsum(dim='time').isel(time=-1).MB.values
        mod_accum_pctls_during_assim = np.random.choice(
            np.percentile(model_accum, pctlrange), n_samples)
        print('model accum:', np.median(model_accum))
        diff_mod_infer = np.median(current_melt) - np.nanmedian(model_melt)
        print('difference : ', diff_mod_infer)

        # get climatological melt sum in the assimilation time span
        testclim = test.read_pickle('mb_daily')
        # todo: is it okay to shorten the reference period? Probably we should
        #  rather take the geodetic climatology for all!
        testclim = testclim.mb.get_climate_reference_period(
            ref_period=clim_ref_period, mbyear_beginmonth=10,
            mbyear_beginday=1)
        model_cmb = testclim.mb.select_doy_span(first_date_assim.dayofyear,
                                                last_date_assim.dayofyear)
        model_cmelt = model_cmb.where(model_cmb.MB < 0.)
        climhyears = model_cmelt.mb.make_hydro_years(
            bg_month=first_date_assim.month,
            bg_day=first_date_assim.day)
        climdoys = model_cmelt.mb.make_hydro_doys(
            climhyears, bg_month=first_date_assim.month,
            bg_day=first_date_assim.day)
        mbcsclim = model_cmelt.groupby(climhyears).map(
            MassBalance.nan_or_cumsum)
        climquant = mbcsclim.groupby(climdoys).map(
            lambda x: MassBalance.custom_quantiles(
                x, qs=np.arange(0., 1.01, 0.01))).isel(hydro_doys=-1).MB.values

        # just plot the three estimates for comparison
        plt.figure()
        plt.plot(sorted(np.array(current_melt).flatten()),
                 label='perc_extrap, span: {}'.format(np.ptp(current_melt)))
        plt.plot(sorted(model_melt.flatten()),
                 label='model, span: {}'.format(np.ptp(model_melt)))
        plt.plot(sorted(climquant.flatten()),
                 label='clim, span: {}'.format(np.ptp(climquant)))
        plt.legend()

        # whole MB year estimates from model only
        mod_est = mod_all.sel(time=slice(evaluate_begin, evaluate_end)).cumsum(
            dim='time').isel(time=-1).MB.values.flatten()

        # model total until one day before assimilation => the initial state
        mod_pctls_until_assim = np.random.choice(np.percentile(mod_all.sel(
            time=slice(evaluate_begin,
                       first_date_assim - pd.Timedelta(days=1))).cumsum(
            dim='time').isel(time=-1).MB.values.flatten(), pctlrange),
                                                 n_samples)
        print('UNTIL ASSIM: ', np.mean(mod_pctls_until_assim))

        # see if we still need to add some model part at the end
        print(last_date_assim, evaluate_end,
              last_date_assim != pd.Timestamp(evaluate_end))
        if last_date_assim != pd.Timestamp(evaluate_end):
            mod_pctls_after_assim = np.random.choice(np.percentile(
                mod_all.sel(time=slice(last_date_assim, evaluate_end)).cumsum(
                    dim='time').isel(time=-1).MB.values.flatten(),
                pctlrange), n_samples)
        else:
            mod_pctls_after_assim = np.zeros(n_samples)

        print(mod_pctls_after_assim)

        extrap_of_pctl_est = \
            mod_pctls_until_assim + mod_accum_pctls_during_assim + \
            extrap_pctls_melt + mod_pctls_after_assim
        print(
            '{} MODEL WHOLE YEAR: {}, {},\n PCTL_EXTRAP WHOLE YEAR: {}, {}'
            .format(test.name, np.median(mod_est), np.std(mod_est),
                    np.median(extrap_of_pctl_est),
                    np.std(extrap_of_pctl_est)))


def validate_percentile_extrapolation_at_glamos_glaciers(
        mb_suffix='', use_snow_redist=False):
    """
    Extrapolate melt percentiles with all three camera glaciers and see if we
    can improve the predicted MB for other GLAMOS glaciers.

    Parameters
    ----------
    mb_suffix: str
        Suffix from the mass balance file to be evaluated, for experiments.
        Default: '' (not suffix).
    use_snow_redist: bool, optional
        Whether to use snow redistribution or not. Default: False (do not use).


    Returns
    -------

    """

    clim_ref_period = (None, None)
    pctlrange = np.arange(101)
    n_samples = 1000
    base_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')
    rgd = utils.GlacierDirectory('RGI50-11.B4312n-1', base_dir=base_dir)
    fgd = utils.GlacierDirectory('RGI50-11.B5616n-1', base_dir=base_dir)
    pgd = utils.GlacierDirectory('RGI50-11.A55F03', base_dir=base_dir)
    obs_glaciers = np.array([rgd, pgd, fgd])

    glac_shp = os.path.join(cfg.PATHS['data_dir'], 'outlines',
                            'mauro_sgi_merge.shp')
    rgidf = gpd.read_file(glac_shp)
    rgidf = rgidf[rgidf.RGIId.isin([
        'RGI50-11.A10G05', 'RGI50-11.B5616n-1', 'RGI50-11.A55F03',
        'RGI50-11.B4312n-1', 'RGI50-11.B4504', 'RGI50-11.C1410',
        'RGI50-11.B5614n', 'RGI50-11.E2320n', 'RGI50-11.E2316',
        'RGI50-11.A51E08', 'RGI50-11.A50D01', 'RGI50-11.B1601',
        'RGI50-11.B2201', 'RGI50-11.A50I19-4', 'RGI50-11.B5232',
        'RGI50-11.B5229', 'RGI50-11.B5263n'
        # 'RGI50-11.A51E12',  # St. Anna  no inversion flowline !?
        # 'RGI50-11.B3626-1',  # Gr. Aletsch -- fails KeyError: 'prcp_fac'
    ])]
    rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B3626-1'])]
    glamos_glaciers = workflow.init_glacier_regions(rgidf)

    # get first and last date of assimilation period (to make it comparable)
    obs = [prepare_holfuy_camera_readings(g) for g in [rgd, pgd, fgd]]
    conc = xr.merge(obs)
    first_date_assim = pd.Timestamp(conc.date.values[0])
    # todo: check how evaluate_end and last_date_assim belong together
    # last_date_assim = min(pd.Timestamp(evaluate_end),
    #                       pd.Timestamp(conc.date.values[-1]))
    last_date_assim = pd.Timestamp(conc.date.values[-1])

    # infer melt from pctl extrapolation in assimilation time span
    try:
        train_pctls = get_melt_percentiles(obs_glaciers, last_date_assim,
                                           mbclim_suffix=mb_suffix)
    except FileNotFoundError:
        train_pctls = get_melt_percentiles(obs_glaciers, last_date_assim)
    extrapolated_pctls = extrapolate_melt_percentiles(train_pctls,
                                                      np.arange(500000,
                                                                800000,
                                                                500),
                                                      np.arange(60000,
                                                                190000,
                                                                500))

    for gg in glamos_glaciers:
        print(gg.rgi_id)
        glamos_mb = calibration.get_measured_mb_glamos(gg)
        glamos_mb = glamos_mb.loc[glamos_mb.date_s.dt.year == 2019]
        glamos_summer = glamos_mb.Annual - glamos_mb.Winter
        evaluate_begin = glamos_mb.date_s.item()  # '2019-04-30'
        evaluate_end = glamos_mb.date1.item()  # '2019-09-30'
        print(evaluate_begin, evaluate_end)
        try:
            current_melt = infer_current_mb_from_melt_percentiles(
                [gg], extrapolated_pctls, last_date_assim,
                date_range_obs=pd.date_range(first_date_assim,
                                             last_date_assim),
                mbclim_suffix=mb_suffix)
        except FileNotFoundError:
            current_melt = infer_current_mb_from_melt_percentiles(
                [gg], extrapolated_pctls, last_date_assim,
                date_range_obs=pd.date_range(first_date_assim,
                                             last_date_assim))
        extrap_pctls_melt = np.random.choice(
            np.percentile(current_melt, pctlrange), n_samples)

        # get model melt in assimilation time span => it must be mb_current,
        # because usually we don't have assimilated mass balances
        begin_mbyear = utils.get_begin_last_flexyear(
            pd.Timestamp(evaluate_begin))
        if use_snow_redist is True:
            try:
                mod_all = gg.read_pickle('mb_current' + mb_suffix)
            except:
                mod_all = gg.read_pickle('mb_current')
            if (mod_all.attrs['snow_redist'] == 'no') and \
                    (gg.rgi_id in cfg.PARAMS['glamos_ids']):
                mod_all = mb_production.make_mb_current_mbyear(
                    gg, begin_mbyear, write=False, use_snow_redist=True,
                    suffix=mb_suffix)
        else:
            mod_all = mb_production.make_mb_current_mbyear(gg,
                                                           begin_mbyear, write=False, use_snow_redist=False,
                                                           suffix=mb_suffix)

        model_mb = mod_all.sel(time=slice(first_date_assim, last_date_assim))
        model_melt = model_mb.where(model_mb.MB < 0.)
        time_index = model_melt.time
        model_melt = model_melt.cumsum(dim='time').assign_coords(
            time=time_index).isel(time=-1).MB.values
        print('model melt:', np.median(model_melt))

        # get model accum in assimilation time span (for reassembling later)
        model_accum = model_mb.where(model_mb.MB > 0.)
        model_accum = model_accum.cumsum(dim='time').isel(time=-1).MB.values
        mod_accum_pctls_during_assim = np.random.choice(
            np.percentile(model_accum, pctlrange), n_samples)
        print('model accum:', np.median(model_accum))
        diff_mod_infer = np.median(current_melt) - np.nanmedian(model_melt)
        print('difference : ', diff_mod_infer)

        # get climatological melt sum in the assimilation time span
        testclim = gg.read_pickle('mb_daily' + mb_suffix)
        if (use_snow_redist is False) and \
                (testclim.attrs['snow_redist'] == 'yes'):
            mb_production.make_mb_clim(
                gg, write=False, use_snow_redist=False, suffix=mb_suffix)
        if (use_snow_redist is True) and (
                testclim.attrs['snow_redist'] == 'no'):
            mb_production.make_mb_clim(
                gg, write=False, use_snow_redist=True, suffix=mb_suffix)

        # todo: is it okay to shorten the reference period? Probably we should
        #  rather take the geodetic climatology for all!
        testclim = testclim.mb.get_climate_reference_period(
            ref_period=clim_ref_period, mbyear_beginmonth=10,
            mbyear_beginday=1)
        model_cmb = testclim.mb.select_doy_span(first_date_assim.dayofyear,
                                                last_date_assim.dayofyear)
        model_cmelt = model_cmb.where(model_cmb.MB < 0.)
        climhyears = model_cmelt.mb.make_hydro_years(
            bg_month=first_date_assim.month,
            bg_day=first_date_assim.day)
        climdoys = model_cmelt.mb.make_hydro_doys(
            climhyears, bg_month=first_date_assim.month,
            bg_day=first_date_assim.day)
        mbcsclim = model_cmelt.groupby(climhyears).map(
            MassBalance.nan_or_cumsum)
        climquant = mbcsclim.groupby(climdoys).apply(
            lambda x: MassBalance.custom_quantiles(
                x, qs=np.arange(0., 1.01, 0.01))).isel(hydro_doys=-1).MB.values

        # just plot the three estimates for comparison
        plt.figure()
        plt.plot(sorted(np.array(current_melt).flatten()),
                 label='perc_extrap, span: {}'.format(np.ptp(current_melt)))
        plt.plot(sorted(model_melt.flatten()),
                 label='model, span: {}'.format(np.ptp(model_melt)))
        plt.plot(sorted(climquant.flatten()),
                 label='clim, span: {}'.format(np.ptp(climquant)))
        plt.legend()

        # whole MB year estimates from model only
        mod_est = mod_all.sel(time=slice(evaluate_begin, evaluate_end)).cumsum(
            dim='time').isel(time=-1).MB.values.flatten()

        # model total until one day before assimilation => the initial state
        mod_pctls_until_assim = np.random.choice(np.percentile(mod_all.sel(
            time=slice(evaluate_begin,
                       first_date_assim - pd.Timedelta(days=1))).cumsum(
            dim='time').isel(time=-1).MB.values.flatten(), pctlrange),
                                                 n_samples)
        print('UNTIL ASSIM: ', np.mean(mod_pctls_until_assim))

        # see if we still need to add some model part at the end
        if last_date_assim <= pd.Timestamp(evaluate_end):
            print(last_date_assim, evaluate_end)
            mod_pctls_after_assim = np.random.choice(np.percentile(
                mod_all.sel(time=slice(last_date_assim, evaluate_end)).cumsum(
                    dim='time').isel(time=-1).MB.values.flatten(),
                pctlrange), n_samples)
        else:
            mod_pctls_after_assim = np.zeros(n_samples)

        extrap_of_pctl_est = \
            mod_pctls_until_assim + mod_accum_pctls_during_assim + \
            extrap_pctls_melt + mod_pctls_after_assim
        print(
            '{} MODEL: {}, {},\n PCTL_EXTRAP: {}, {}\n'.format(
                gg.name, np.median(mod_est), np.std(mod_est),
                np.median(extrap_of_pctl_est), np.std(extrap_of_pctl_est),
                gg.name,
                ))
        # from properscoring import crps_ensemble
        # print(mod_est.shape, extrap_of_pctl_est.shape)
        # print(glamos_summer.shape, extrap_of_pctl_est.shape)
        # print('CRPS MODEL/EXTRAP: ', crps_ensemble(np.median(mod_est),
        # np.atleast_2d(extrap_of_pctl_est).T))
        # print('CRPS GLAMOS SUMMER/EXTRAP: ', crps_ensemble(glamos_summer,
        # np.atleast_2d(extrap_of_pctl_est).T))
        plt.figure()
        plt.scatter(np.zeros_like(extrap_pctls_melt), extrap_pctls_melt)
        plt.scatter(0., np.mean(extrap_pctls_melt))
        plt.scatter(np.ones_like(climquant), climquant)
        plt.scatter(1., np.mean(climquant))
        plt.title(gg.name)
