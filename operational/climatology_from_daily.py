"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Log message format
import logging

# Python imports
import os
# Libs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime as dt
# Locals
import crampon.cfg as cfg
from crampon import workflow
from crampon import tasks
from crampon.workflow import execute_entity_task
from crampon import graphics, utils
from crampon.core.models.massbalance import MassBalance, BraithwaiteModel, \
    PellicciottiModel, OerlemansModel, SnowFirnCover
import numpy as np
import copy

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def make_mb_clim(gdir, mb_model=None, bgyear=1961, endyear=None, write=True,
                 reset=False, suffix=''):
    """
    Make a mass balance climatology for the available calibration period.

    Parameters
    ----------
    gdir::py:class:`crampon.GlacierDirectory`
        The GlacierDirectory to calculate the current mass balance for.
    mb_model:
        Which mass balance model to use. Default: None (use all available).
    bgyear: int
        Begin year for the climatology. This is automatically cut to the
        allowed usage span fo the mass balance model (e.g. due to radiation
        data availability). Default: 1961.
    endyear: int or None
        End year for the climatology. Default: None (take the last full mass
        budget year, i.e. if today's date is within the year is after October
        1st, then include automatically the recently completed year.
    write: bool
        Whether or not to write the result into the GlacierDirectory.
        Default: True.
    suffix: str
        Suffix to use for the calibration file and the output mass balance
        files, e.g. '_fischer_unique'.

    Returns
    -------
    mb_ds, snow_cond, time_elap: xr.Dataset, np.ndarray, pd.DatetimeIndex
        The mass balance as an xarray dataset, the snow conditions during the
        run and the elapsed time.
    """

    if mb_model:
        if type(mb_model) == list:
            mb_models = mb_model
        else:
            mb_models = [mb_model]
    else:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

    try:
        cali_frame = calibration.get_measured_mb_glamos(gdir)
        cali_dates = cali_frame[['date0', 'date1', 'date_s', 'date_f']].values
    except (FileNotFoundError, IndexError):
        cali_dates = None

    today = dt.datetime.now()
    if endyear is None:
        t_month, t_day = today.month, today.day
        if (t_month >= cfg.PARAMS['begin_mbyear_month']) and \
                (t_day >= cfg.PARAMS['begin_mbyear_days']):
            endyear = today.year
        else:
            endyear = today.year - 1

    heights, widths = gdir.get_inversion_flowline_hw()

    ds_list = []
    sc_list = []
    mb_models_used = []
    for exp, mbm in enumerate(mb_models):
        if hasattr(mbm, 'calibration_timespan'):
            if mbm.calibration_timespan[0]:
                bgyear = mbm.calibration_timespan[0]
            if mbm.calibration_timespan[1]:
                # overwrite endyear
                endyear = mbm.calibration_timespan[1]

        begin_clim = utils.get_begin_last_flexyear(dt.datetime(bgyear, 12, 31))
        end_clim = utils.get_begin_last_flexyear(dt.datetime(endyear, 12, 31))

        # todo: pick up the snow cover from the spinup phase here

        if isinstance(mbm, utils.SuperclassMeta):
            print(mbm.__name__)
            try:
                day_model = mbm(gdir, bias=0., cali_suffix=suffix)
                mb_models_used.append(mbm)
            except KeyError:  # model not in cali file (for fischer geod. cali)
                continue
        else:
            day_model = copy.copy(mbm)
        mb = []
        sc_list_one_model = []
        sc_date_list = []
        run_span = pd.date_range(begin_clim, end_clim)
        for date in run_span:
            # Get the mass balance and convert to m per day
            tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)

            if (cali_dates is not None) and (date in cali_dates):
                sc_list_one_model.append(day_model.snowcover.to_dataset(date=date))
                sc_date_list.append(date)

            # todo: when are clever moments to store the snowcover?
            if date == end_clim:
                sc_list_one_model.append(day_model.snowcover.to_dataset(date=date))
                sc_date_list.append(date)

        mb_for_ds = np.moveaxis(np.atleast_3d(np.array(mb)), 1, 0)
        mb_ds = xr.Dataset({'MB': (['time', 'member', 'model'], mb_for_ds)},
                           coords={'member': (['member', ],
                                              np.arange(mb_for_ds.shape[1])),
                                   'model': (['model', ],
                                             [day_model.__name__]),
                                   'time': (['time', ], run_span),
                                   })
        ds_list.append(mb_ds)

        sc_list.append(xr.concat(sc_list_one_model, dim=pd.Index(sc_date_list,
                                            name='time')))

    # merge all models together
    merged = xr.merge(ds_list)
    # todo: units are hard coded and depend on method used above
    merged.attrs.update({'id': gdir.rgi_id, 'name': gdir.name,
                         'units': 'm w.e.'})
    if write:
        merged.mb.append_to_gdir(gdir, 'mb_daily' + suffix, reset=reset)

    # take care of snow
    sc_ds = xr.concat(sc_list, dim=pd.Index([m.__name__ for m in mb_models_used],
                                            name='model'))
    if write:
        gdir.write_pickle(sc_ds, 'snow_daily' + suffix)

    return merged, sc_ds


def make_mb_clim_new(gdir, mb_model=None, write=True, reset=False, suffix=''):
    """
    Make a mass balance climatology for the available calibration period.

    Parameters
    ----------
    gdir::py:class:`crampon.GlacierDirectory`
        The GlacierDirectory to calculate the current mass balance for.
    mb_model:
        Which mass balance model to use. Default: None (use all available).
    bgyear: int
        Begin year for the climatology. This is automatically cut to the
        allowed usage span fo the mass balance model (e.g. due to radiation
        data availability). Default: 1961.
    endyear: int or None
        End year for the climatology. Default: None (take the last full mass
        budget year, i.e. if today's date is within the year is after October
        1st, then include automatically the recently completed year.
    write: bool
        Whether or not to write the result into the GlacierDirectory.
        Default: True.

    Returns
    -------
    mb_ds, snow_cond, time_elap: xr.Dataset, np.ndarray, pd.DatetimeIndex
        The mass balance as an xarray dataset, the snow conditions during the
        run and the elapsed time.
    """

    if mb_model:
        if type(mb_model) == list:
            mb_models = mb_model
        else:
            mb_models = [mb_model]
    else:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

    today = dt.datetime.now()

    t_month, t_day = today.month, today.day
    if (t_month >= 10) and (t_day >= 1):
        endyear = today.year
    else:
        endyear = today.year - 1

    fischer_df = pd.read_csv(
        os.path.join(cfg.PATHS['data_dir'], 'fischeretal_2015_geod_mb.csv'),
        encoding="iso-8859-1")
    fischer_vals = fischer_df[
        fischer_df['SGI2010'].str.contains(gdir.rgi_id.split('.')[1])]
    print(fischer_vals)
    t1_year = fischer_vals.t1_year.item()
    t2_year = fischer_vals.t2_year.item()
    area1 = fischer_vals.area_t1_km2.item() * 10 ** 6  # km2 -> m2
    area2 = fischer_vals.area_t2_km2.item() * 10 ** 6  # km2 -> m2
    area_change_avg = (area2 - area1) / (t2_year - t1_year)
    gmb_spec_avg = fischer_vals.gmb_spec_t1_t2.item()
    gmb_spec_avg_unc = fischer_vals.uncertainty_gmb_spec_t1_t2.item()

    # use T as a proxy to distribute the MB
    # todo: use T and SIS in years where available
    tsums = []
    gmeteo = climate.GlacierMeteo(gdir)  # needed later
    tmean_glacier = climate.GlacierMeteo(gdir).meteo.temp
    y1y2_span = np.arange(t1_year, t2_year)
    for y in y1y2_span:
        temps = tmean_glacier.sel(time=slice('{}-10-01'.format(str(y)),
                                             '{}-09-30'.format(
                                                 str(y + 1)))).values
        temps = np.clip(temps, 0, None)
        tsum = np.sum(temps)
        tsums.append(tsum)
    corr_fac = (1 + (tsums - np.mean(tsums)) / np.mean(tsums))
    gmb_spec_disagg = gmb_spec_avg * corr_fac
    gmb_spec_unc_disagg = gmb_spec_avg_unc * corr_fac
    area_change_disagg = area_change_avg * corr_fac

    # take name of hydro year as index ("+1")
    disagg_df = pd.DataFrame(index=y1y2_span + 1,
                             columns=['gmb', 'gmb_unc', 'area'],
                             data=np.array([gmb_spec_disagg,
                                            gmb_spec_unc_disagg,
                                            area_change_disagg]).T)

    cali_bg_year = t1_year
    year_init_hw = gdir.rgi_date.year
    heights, widths = gdir.get_inversion_flowline_hw()
    fl_dx = gdir.read_pickle('inversion_flowlines')[-1].dx

    ds_list = []
    sc_list = []
    for exp, mbm in enumerate(mb_models):
        if hasattr(mbm, 'calibration_timespan'):
            if mbm.calibration_timespan[0]:
                bgyear = mbm.calibration_timespan[0]
            if mbm.calibration_timespan[1]:
                # overwrite endyear
                endyear = mbm.calibration_timespan[1]

        # todo: constrain bgyear with the beginning with the minimum date of the cli CSV file

        begin_clim = utils.get_begin_last_flexyear(dt.datetime(t1_year, 12, 31))
        end_clim = utils.get_begin_last_flexyear(dt.datetime(t2_year, 12, 31))

        # todo: pick up the snow cover from the spinup phase here

        if isinstance(mbm, utils.SuperclassMeta):
            try:
                day_model = mbm(gdir, bias=0., cali_suffix=suffix)
            except KeyError:  # model not in cali file (for fischer geod. cali)
                mb_models.remove(mbm)
                continue
        else:
            day_model = copy.copy(mbm)
        mb = []
        run_span = pd.date_range(begin_clim, end_clim)

        scov = None
        for year in range(cali_bg_year, t2_year):

            # arbitrary range for which the params are valid - choose the mb year
            valid_range = pd.date_range('{}-10-01'.format(year),
                                        '{}-09-30'.format(year + 1))

            # area change until date of the outlines
            area_chg = np.sum(disagg_df.loc[year + 1: year_init_hw].area)
            # continue with last width
            # todo: this is to cut off the round tongue: remove the hard-coded numbers!
            if widths.size > 55:
                last_width = np.mean(widths[-55:-25])
            else:  # super short glacier
                last_width = np.mean(widths)
            # continue with slope of lowest
            last_slope = (heights[-1] - heights[-6]) / (5 * fl_dx * gdir.grid.dx)

            # make area chg positive
            n_new_nodes = - area_chg / (fl_dx * gdir.grid.dx) / last_width
            new_heights = last_slope * np.arange(1, np.ceil(n_new_nodes) + 1) + \
                          heights[-1]
            heights_annual = np.hstack((heights, new_heights))
            widths_annual = np.hstack((widths,  # old width nodes
                                       np.repeat([last_width], np.floor(n_new_nodes)),
                                       # new full width nodes
                                       np.array([(n_new_nodes % np.floor(
                                           n_new_nodes)) * last_width])))  # new rest width nodes
            if scov is not None:
                scov.remove_height_nodes(
                    np.arange(len(widths_annual), day_model.snowcover.swe.shape[0]))
            # todo: at the moment this is supercheap: it can only REMOVE, and ONLY REMOVE AT THE VERY TONGUE
            if isinstance(mbm, utils.SuperclassMeta):
                day_model = mbm(gdir, bias=0., cali_suffix=suffix, snowcover=scov, heights_widths=(heights_annual, widths_annual))
            else:
                day_model = copy.copy(mbm)


            for date in valid_range:
                # Get the mass balance and convert to m per day
                tmp = day_model.get_daily_specific_mb(heights_annual, widths_annual, date=date)
                mb.append(tmp)

                # todo: when are clever moments to store the snowcover?
                if date == valid_range[-1]:
                    #sc_list.append(day_model.snowcover.to_dataset(date=date))
                    scov = copy.deepcopy(day_model.snowcover)

        mb_for_ds = np.moveaxis(np.atleast_3d(np.array(mb)), 1, 0)
        mb_ds = xr.Dataset({'MB': (['time', 'member', 'model'], mb_for_ds)},
                           coords={'member': (['member', ],
                                              np.arange(mb_for_ds.shape[1])),
                                   'model': (['model', ],
                                             [day_model.__name__]),
                                   'time': (['time', ], run_span[:-1]),
                                   })
        ds_list.append(mb_ds)

    # merge all models together
    merged = xr.merge(ds_list)
    # todo: units are hard coded and depend on method used above
    merged.attrs.update({'id': gdir.rgi_id, 'name': gdir.name,
                         'units': 'm w.e.'})
    if write:
        merged.mb.append_to_gdir(gdir, 'mb_daily'+suffix, reset=reset)

    # take care of snow
    sc_ds = None
    #sc_ds = xr.concat(sc_list, dim=pd.Index([m.__name__ for m in mb_models],
    #                                        name='model'))
    #if reset:
    #    gdir.write_pickle(sc_ds, 'snow_daily'+suffix)

    return merged, sc_ds


def make_mb_current_mbyear(gdir, begin_mbyear, mb_model=None, snowcover=None,
                           write=True, reset=False, suffix=''):
    """
    Make the mass balance of the current mass budget year for a given glacier.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to calculate the current mass balance for.
    begin_mbyear: datetime.datetime
        The beginning of the current mass budget year.
    mb_model: `py:class:crampon.core.models.massbalance.DailyMassBalanceModel`,
              optional
        A mass balance model to use. Default: None (use all available).
    write: bool
        Whether or not to write the result to GlacierDirectory. Default: True
        (write out).
    reset: bool
        Whether to completely overwrite the mass balance file in the
        GlacierDirectory or to append (=update with) the result. Default: False
        (append).
    suffix: str
        Suffix of the calibration file and the mass balance files, e.g.
        '_fischer_unique'. Default: ''.

    Returns
    -------
    mb_now_cs: xr.Dataset
        Mass balance of current mass budget year as cumulative sum.
    """

    if mb_model:
        mb_models = [mb_model]
    else:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

    yesterday = utils.get_cirrus_yesterday()
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    begin_str = begin_mbyear.strftime('%Y-%m-%d')

    curr_year_span = pd.date_range(start=begin_str, end=last_day_str,
                                   freq='D')

    ds_list = []
    for mbm in mb_models:

        stacked = None
        heights, widths = gdir.get_inversion_flowline_hw()

        param_prod = np.array(
            utils.get_possible_parameters_from_past(gdir, mbm, only_pairs=True,
                                                    latest_climate=True,
                                                    as_list=True))

        # todo: select by model or start with ensemble median snowcover?
        sc = SnowFirnCover.from_dataset(snowcover.sel(model=mbm.__name__))

        it = 0
        for params in param_prod:
            print(it, dt.datetime.now())
            it += 1

            pdict = dict(zip(mbm.cali_params_list, params))
            if isinstance(mbm, utils.SuperclassMeta):
                try:
                    day_model_curr = mbm(gdir, **pdict, snowcover=sc, bias=0.,
                                         cali_suffix=suffix)
                except:  # model not in cali file (for fischer geod. cali)
                    mb_models.remove(mbm)
                    continue
            else:
                day_model_curr = copy.copy(mbm)

            mb_now = []
            for date in curr_year_span:
                # Get the mass balance and convert to m per day
                tmp = day_model_curr.get_daily_specific_mb(heights, widths,
                                                           date=date)
                mb_now.append(tmp)

            if stacked is not None:
                stacked = np.vstack((stacked, mb_now))
            else:
                stacked = mb_now

        stacked = np.sort(stacked, axis=0)

        mb_for_ds = np.moveaxis(np.atleast_3d(np.array(stacked)), 1, 0)
        # todo: units are hard coded and depend on method used above
        mb_ds = xr.Dataset({'MB': (['time', 'member', 'model'], mb_for_ds)},
                           coords={'member': (['member', ],
                                              np.arange(mb_for_ds.shape[1])),
                                   'model': (['model', ],
                                             [day_model_curr.__name__]),
                                   'time': (['time', ],
                                            pd.to_datetime(curr_year_span)),
                                   })

        ds_list.append(mb_ds)

    ens_ds = xr.merge(ds_list)
    ens_ds.attrs.update({'id': gdir.rgi_id, 'name': gdir.name,
                         'units': 'm w.e.'})

    if write:
        ens_ds.mb.append_to_gdir(g, 'mb_current', reset=reset)

    return ens_ds





        else:















                                                     loc=3)

    # Write out glacier statistics
