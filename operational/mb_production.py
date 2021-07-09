"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Log message format
import logging

# Python imports
import os
# Libs
import pandas as pd
import xarray as xr
import datetime as dt
import numpy as np
import copy
# Locals
from crampon import cfg
from crampon import utils, entity_task
# only temporary, until process_spinup_climate is a task
from crampon.core.preprocessing import climate, calibration
from crampon.utils import lazy_property
from crampon.core.models.massbalance import MassBalance, BraithwaiteModel, \
    PellicciottiModel, OerlemansModel, HockModel, SnowFirnCover, \
    get_rho_fresh_snow_anderson, ParameterGenerator, MassBalanceModel

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


@entity_task(log, writes=['mb_daily', 'snow_daily'])
def make_mb_clim(gdir: utils.GlacierDirectory,
                 mb_model: MassBalanceModel = None, bgyear: int = 1961,
                 endyear: int or None = None, write: bool = True,
                 reset_file: bool = False, use_snow_redist: bool = True,
                 suffix: str = '') -> None:
    """
    Make a mass balance climatology for the available calibration period.

    Parameters
    ----------
    gdir:`py:class:crampon.GlacierDirectory`
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
        Whether or not to write the result into the GlacierDirectory (leave
        other existing values untouched, except `reset_file` is set to True).
        Default: True.
    reset_file: bool
        Whether or not to delete an existing mass balance file and start from
        scratch. The reason why this is not called "reset"  only is that it
        interferes with the argument `reset` from entity_task. Default: False.
    use_snow_redist: bool
        Whether to apply snow redistribution, if applicable. Default: True.
    suffix: str
        Suffix to use for the calibration file and the output mass balance
        files, e.g. '_fischer_unique'.

    Returns
    -------
    None
    #mb_ds, snow_cond, time_elap: xr.Dataset, np.ndarray, pd.DatetimeIndex
    #    The mass balance as an xarray dataset, the snow conditions during the
    #    run and the elapsed time.
    """

    hyear_bgmonth = cfg.PARAMS['begin_mbyear_month']
    hyear_bgday = cfg.PARAMS['begin_mbyear_day']

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
        # just for some test
        cali_dates = list(cali_dates.flatten())
        cali_dates = np.asarray(cali_dates)
    except (FileNotFoundError, IndexError):
        cali_dates = None

    today = dt.datetime.now()
    if endyear is None:
        t_month, t_day = today.month, today.day
        if (t_month >= hyear_bgmonth) and (t_day >= hyear_bgday):
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

        # todo: constrain bgyear with the min date of the cali CSV file
        begin_clim = utils.get_begin_last_flexyear(dt.datetime(bgyear, 12, 31))
        end_clim = utils.get_begin_last_flexyear(dt.datetime(endyear, 12, 31))

        # todo: pick up the snow cover from the spinup phase here
        if isinstance(mbm, utils.SuperclassMeta):
            print(mbm.__name__)
            try:
                day_model = mbm(gdir, bias=0., cali_suffix=suffix,
                                snow_redist=use_snow_redist)
                mb_models_used.append(mbm)
            except KeyError:  # model not in cali file (for fischer geod. cali)
                continue
        else:
            day_model = copy.copy(mbm)

        # if 'unique' in suffix:
        #    geod_cali = gdir.get_calibration(day_model, suffix=suffix)
        #    clip_years = geod_cali.time.min().year, geod_cali.time.max().year
        #    try:
        #        glamos_params = ParameterGenerator(gdir, mb_model,
        #        latest_climate=False,
        #         only_pairs=True, constrain_with_bw_prcp_fac=True,
        #         bw_constrain_year=None, narrow_distribution=False,
        #         output_type=None, suffix='').from_single_glacier(
        #         clip_years=clip_years)
        #        p_variability = glamos_params - np.nanmean(glamos_params)
        #        params = geod_cali + p_variability
        #    except:
        #        params = None
        # else:
        #    params = None

        mb = []
        sc_list_one_model = []
        sc_date_list = []
        alpha = []
        run_span = pd.date_range(begin_clim, end_clim)
        for date in run_span:
            # Get the mass balance and convert to m per day
            tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)

            if hasattr(day_model, 'albedo'):
                alpha.append(day_model.albedo.alpha[0])
            else:
                alpha.append(None)

            # todo: add spring date
            if ((cali_dates is not None) and (date in cali_dates)) or (
                    (date.month == hyear_bgmonth) and (
                    date.day == hyear_bgday)):
                sc_list_one_model.append(
                    day_model.snowcover.to_dataset(date=date))
                sc_date_list.append(date)

            if (date == end_clim) and not ((date.month == hyear_bgmonth) and (
                    date.day == hyear_bgday)):
                sc_list_one_model.append(
                    day_model.snowcover.to_dataset(date=date))
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

        sc_list.append(xr.concat(sc_list_one_model,
                                 dim=pd.Index(sc_date_list, name='time')))

    # merge all models together
    merged = xr.merge(ds_list)
    merged.attrs.update({'id': gdir.rgi_id, 'name': gdir.name,
                         'units': 'm w.e.', 'snow_redist':
                             'yes' if use_snow_redist is True else 'no',
                         'suffix': suffix})
    if write:
        merged.mb.append_to_gdir(gdir, 'mb_daily' + suffix, reset=reset_file)

    # take care of snow
    sc_ds = xr.concat(
        sc_list, dim=pd.Index([m.__name__ for m in mb_models_used],
                              name='model'))
    if write:
        gdir.write_pickle(sc_ds, 'snow_daily' + suffix)


@entity_task(log, writes=['mb_daily', 'snow_daily'])
def make_mb_clim_new(gdir, mb_model=None, write=True, reset_file=False,
                     suffix=''):
    """
    Make a mass balance climatology for the available calibration period.

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        The GlacierDirectory to calculate the current mass balance for.
    mb_model:
        Which mass balance model to use. Default: None (use all available).
    write: bool
        Whether or not to write the result into the GlacierDirectory.
        Default: True.
    reset_file: bool, optional
        Whether to reset existing files. The reason why this is not called
        "reset"  only is that it interferes with the argument `reset` from
        entity_task. Default: False.
    suffix: str, optional
        Suffix to add to filename. Default: ''.

    Returns
    -------
    None
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

        # todo: constrain bgyear with the min date of the cli CSV file

        begin_clim = utils.get_begin_last_flexyear(
            dt.datetime(t1_year, 12, 31))
        end_clim = utils.get_begin_last_flexyear(
            dt.datetime(t2_year, 12, 31))

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

            # arbitrary range for which params are valid - choose the mb year
            valid_range = pd.date_range('{}-10-01'.format(year),
                                        '{}-09-30'.format(year + 1))

            # area change until date of the outlines
            area_chg = np.sum(disagg_df.loc[year + 1: year_init_hw].area)
            # continue with last width
            # todo: cut off the round tongue: remove the hard-coded numbers!
            if widths.size > 55:
                last_width = np.mean(widths[-55:-25])
            else:  # super short glacier
                last_width = np.mean(widths)
            # continue with slope of lowest
            last_slope = (heights[-1] - heights[-6]) / \
                         (5 * fl_dx * gdir.grid.dx)

            # make area chg positive
            n_new_nodes = - area_chg / (fl_dx * gdir.grid.dx) / last_width
            new_heights = last_slope * np.arange(1, np.ceil(n_new_nodes) + 1)\
                + heights[-1]
            heights_annual = np.hstack((heights, new_heights))
            widths_annual = np.hstack(
                (widths,  # old width nodes
                 # new full width nodes
                 np.repeat([last_width], np.floor(n_new_nodes)),
                 np.array([(n_new_nodes % np.floor(n_new_nodes)) *
                           last_width])))  # new rest width nodes
            if scov is not None:
                scov.remove_height_nodes(
                    np.arange(len(widths_annual),
                              day_model.snowcover.swe.shape[0]))
            # todo: supercheap: can only REMOVE, and ONLY REMOVE AT VERY TONGUE
            if isinstance(mbm, utils.SuperclassMeta):
                day_model = mbm(gdir, bias=0., cali_suffix=suffix,
                                snowcover=scov,
                                heights_widths=(heights_annual, widths_annual))
            else:
                day_model = copy.copy(mbm)

            for date in valid_range:
                # Get the mass balance and convert to m per day
                tmp = day_model.get_daily_specific_mb(
                    heights_annual, widths_annual, date=date)
                mb.append(tmp)

                # todo: when are clever moments to store the snowcover?
                if date == valid_range[-1]:
                    # sc_list.append(day_model.snowcover.to_dataset(date=date))
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
        merged.mb.append_to_gdir(gdir, 'mb_daily'+suffix, reset=reset_file)

    # take care of snow
    sc_ds = None
    # sc_ds = xr.concat(sc_list, dim=pd.Index([m.__name__ for m in mb_models],
    #                                        name='model'))
    # if reset_file:
    #    gdir.write_pickle(sc_ds, 'snow_daily'+suffix)

    # return merged, sc_ds


@entity_task(log, writes=['mb_current', 'snow_current'])
def make_mb_current_mbyear(
        gdir: utils.GlacierDirectory,
        first_day: dt.datetime or pd.Timestamp or None = None,
        last_day: dt.datetime or pd.Timestamp or None = None,
        mb_model: MassBalanceModel or None = None,
        snowcover: SnowFirnCover or xr.Dataset or None = None,
        write: bool = True, reset_file: bool = False,
        use_snow_redist: bool = True,
        suffix: str = '') -> None:
    """
    Make the mass balance of the current mass budget year for a given glacier.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to calculate the current mass balance for.
    first_day: datetime.datetime or pd.Timestamp or None, optional
        Custom begin date of the current mass budget year. This can be used to
        just extend the existing mass balance. If given, reset_file will be
        overwritten to `False`, and a warning will be issued if the two values
        are in conflict.Default: None (take the beginning of the last mass
        budget year as defined in params.cfg).
    last_day: dt.datetime or pd.Timestamp or None, optional
        When the calculation shall be stopped (e.g. for assimilation).
        Default: None (take either the last day with meteo data available or
        the last day of the mass budget year since `begin_mbyear`).
    mb_model: `py:class:crampon.core.models.massbalance.DailyMassBalanceModel`,
              optional
        A mass balance model to use. Default: None (use all available).
    snowcover: SnowFirnCover or xr.Dataset or None, optional
        Snowcover object to initiate the snow cover.
    write: bool, optional
        Whether or not to write the result to GlacierDirectory. Default: True
        (write out).
    reset_file: bool, optional
        Whether to completely overwrite the mass balance file in the
        GlacierDirectory or to append (=update with) the result. The reason why
        this is not called "reset"  only is that it interferes with the
        argument `reset` from entity_task. Default: False (append).
    use_snow_redist: bool, optional
        Whether to apply snow redistribution, if applicable. Default: True.
    suffix: str, optional
        Suffix of the calibration file and the mass balance files, e.g.
        '_fischer_unique'. Default: '' (no suffix).

    Returns
    -------
    mb_now_cs: xr.Dataset
        Mass balance of current mass budget year as cumulative sum.
    """

    if mb_model:
        mb_models = [mb_model]
    else:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

    if last_day is None:
        last_day = utils.get_cirrus_yesterday()

    # we need it often later
    mbyear_begin = utils.get_begin_last_flexyear(pd.Timestamp.now())

    # if first day is not given, take begin of current MB year
    if first_day is None:
        first_day = mbyear_begin
    else:
        if reset_file is True:
            first_day = mbyear_begin
        else:
            # we can be anywhere in the MB year...
            try:
                with gdir.read_pickle('mb_current') as mbc:
                    # first day for us is the last day of the existing file + 1
                    last_day_mbc = mbc.time.values[-1]
            except FileNotFoundError:  # e.g. because it has no calibration
                # this is ok for the moment: either
                # 1) file doesn't exist, bcz of a first time call -> create
                # 2) file doesn't exist, bcz no cali -> fail when fetching cali
                last_day_mbc = mbyear_begin - pd.Timedelta(days=1)
            # todo: check also for end date of snow_current?

            # file might be already up to date
            if (last_day_mbc == last_day) and (reset_file is False):
                return

            # there might be a gap between last_day and what is there already
            if (last_day_mbc + pd.Timedelta(days=1)) == first_day:
                # we have it until the day before `first_day`, that's fine
                pass
            elif (last_day_mbc + pd.Timedelta(days=1)) < first_day:
                # we start one day after the present time series ends
                first_day = last_day_mbc + pd.Timedelta(days=1)
            else:
                # tprobably no snow cover from that day: start from scratch
                log.info("We don't have a starting snowcover for the first da"
                         " chosen. We make the time series again...")
                first_day = mbyear_begin

    # still respect argument over retrieval
    if snowcover is None:
        if first_day == mbyear_begin:
            # take "snow_daily"
            try:
                snowcover = gdir.read_pickle('snow_daily' + suffix)
            except FileNotFoundError:
                pass
        else:
            try:
                snowcover = gdir.read_pickle('snow_current' + suffix)
            except FileNotFoundError:
                pass
        # none of the two was successful:
        if snowcover is None:
            log.warning(
                'No initial snow cover given and no snow cover file found. '
                'Initializing snow cover with assumptions defined in '
                'massbalance.DailyMassBalanceModelWithSnow.')

    # if begin more than one year ago, clip to make current MB max 1 year long
    max_end = pd.Timestamp('{}-{}-{}'.format(
        first_day.year + 1, first_day.month, first_day.day)) - \
        pd.Timedelta(days=1)

    if last_day > max_end:
        # todo: it might also make sense to keep "last_day"
        last_day = max_end
    last_day_str = last_day.strftime('%Y-%m-%d')
    begin_str = first_day.strftime('%Y-%m-%d')

    curr_year_span = pd.date_range(start=begin_str, end=last_day_str,
                                   freq='D')

    ds_list = []
    sc_list = []
    day_model_curr = None
    for mbm in mb_models:

        stacked = None
        heights, widths = gdir.get_inversion_flowline_hw()

        if 'fischer' in suffix:
            constrain_with_bw_prcp_fac = True
            latest_climate = False

        else:
            constrain_with_bw_prcp_fac = True
            latest_climate = True

        pg = ParameterGenerator(
            gdir, mbm, latest_climate=latest_climate, only_pairs=True,
            constrain_with_bw_prcp_fac=constrain_with_bw_prcp_fac,
            bw_constrain_year=first_day.year + 1,
            narrow_distribution=0., output_type='array', suffix=suffix)
        try:
            param_prod = pg.from_single_glacier()
        except pd.core.indexing.IndexingError:
            continue

        # no parameters found, e.g. when using latest_climate = True for A50I06
        if param_prod.size == 0:
            log.error('With current settings (`latest_climate=True`), no '
                      'parameters were found to produce current MB.')
            return

        # todo: select by model or start with ensemble median snowcover?
        it = 0
        sc_list_one_model = []
        for params in param_prod:
            it += 1
            if snowcover is not None:
                # todo: maybe fatal? here we assume the parameters are
                #  iterated in the same order. actually mb_current also comes
                #  from this routine: Solution: include params in the dataset?
                try:
                    # it's snow_current with a time and members
                    sc = SnowFirnCover.from_dataset(
                        snowcover.sel(model=mbm.__name__,
                                      time=curr_year_span[0] -
                                           pd.Timedelta(days=1))
                            .isel(member=it-1))
                except (KeyError, ValueError):
                    # it's snow_daily
                    try:
                        # day before is not written (change upstream!)
                        sc = SnowFirnCover.from_dataset(
                            snowcover.sel(model=mbm.__name__,
                                          time=curr_year_span[0] -
                                               pd.Timedelta(days=1)))
                    except KeyError:
                        # be tolerant
                        sc = SnowFirnCover.from_dataset(
                            snowcover.sel(model=mbm.__name__,
                                          time=curr_year_span[0]))
            else:
                sc = None

            pdict = dict(zip(mbm.cali_params_list, params))
            if isinstance(mbm, utils.SuperclassMeta):
                try:
                    day_model_curr = mbm(gdir, **pdict, snowcover=sc, bias=0.,
                                         cali_suffix=suffix,
                                         snow_redist=use_snow_redist)
                # model not in cali file (for fischer geod. cali)
                except Exception:
                    log.info(
                        'Removing {} from calibration with suffix {}'.format(
                            mbm.__name__, suffix))
                    mb_models.remove(mbm)
                    continue
            else:
                day_model_curr = copy.copy(mbm)

            mb_now = []
            for date in curr_year_span:
                # Get the mass balance and convert to m per day
                tmp = day_model_curr.get_daily_specific_mb(heights, widths,
                                                           date=date)
                day_model_curr.snowcover.densify_snow_anderson(date)
                mb_now.append(tmp)

                # only write snow on last day
                if date == curr_year_span[-1]:
                    sc_list_one_model.append(
                        day_model_curr.snowcover.to_dataset(date=date))

            if stacked is not None:
                stacked = np.vstack((stacked, mb_now))
            else:
                stacked = mb_now

        sc_list.append(
            xr.concat(sc_list_one_model,
                      dim=pd.Index(np.arange(len(sc_list_one_model)),
                                   name='member')))

        if isinstance(stacked, np.ndarray):
            stacked = np.sort(stacked, axis=0)
        else:
            # the Fischer case
            stacked = np.array(stacked)

        if day_model_curr is not None:
            mb_for_ds = np.moveaxis(np.atleast_3d(np.array(stacked)), 1, 0)
            mb_ds = xr.Dataset(
                {'MB': (['time', 'member', 'model'], mb_for_ds)}, coords={
                    'member': (['member', ], np.arange(mb_for_ds.shape[1])),
                    'model': (['model', ], [day_model_curr.__name__]),
                    'time': (['time', ], pd.to_datetime(curr_year_span))})

            ds_list.append(mb_ds)

    ens_ds = xr.merge(ds_list)
    ens_ds.attrs.update(
        {'id': gdir.rgi_id, 'name': gdir.name, 'units': 'm w.e.',
         'snow_redist': 'yes' if use_snow_redist is True else 'no',
         'suffix': suffix,
         'last_updated': pd.Timestamp.now().strftime(
             '%Y-%m-%d %H:%M:%S')})

    if write:
        ens_ds.mb.append_to_gdir(gdir, 'mb_current' + suffix, reset=reset_file)

    # check at the point where we cross the MB budget year
    if dt.date.today == (first_day + dt.timedelta(days=1)):
        mb_curr = gdir.read_pickle('mb_current' + suffix)
        mb_curr = mb_curr.sel(time=slice(first_day, None))
        gdir.write_pickle(mb_curr, 'mb_current' + suffix)

    # assemble and write snow status
    snow_ens = xr.concat(
        sc_list, dim=pd.Index([m.__name__ for m in mb_models], name='model'))
    snow_ens.attrs.update(
        {'id': gdir.rgi_id, 'name': gdir.name, 'suffix': suffix,
         'snow_redist': 'yes' if use_snow_redist is True else 'no'})
    # important for reading as SnowFirnCover later on
    snow_ens['last_update'] = snow_ens['last_update'].astype(object)
    snow_ens['origin'] = snow_ens['origin'].astype(object)

    if write:
        gdir.write_pickle(snow_ens, 'snow_current' + suffix)


@entity_task(log, writes=['mb_spinup', 'snow_spinup'])
def make_spinup_mb(gdir, param_method='guess', length=30):
    """
    Create mass balance in the spinup phase (1930-1960).

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to produce the spinup mass balance for.
    param_method: str
        How parameters should be determined. At the moment, there are two
        options: the first one, 'guess', just uses the best guess parameters as
        defined in the MassbalanceModel class. The second option, 'predict'
        uses a random forest model to predict parameters based on geometrical
        and meteorological features. Default: 'guess'.
    length: int
        Length of the spinup period in years. Default: 30.

    Returns
    -------
    None
    """
    models = []
    if 'HockModel' in cfg.MASSBALANCE_MODELS:
        models.append(HockModel)
    if 'BraithwaiteModel' in cfg.MASSBALANCE_MODELS:
        models.append(BraithwaiteModel)

    if not os.path.exists(gdir.get_filepath('spinup_climate_daily')):
        climate.process_spinup_climate_data(gdir)

    ds_list = []
    sc_list = []
    meteo_begin = 1961
    for mbm in models:

        try:
            cali = gdir.get_calibration(mb_model=mbm)
            cali = cali.dropna()
        except FileNotFoundError:
            cali = None

        # spinup phase time depends on model - try to keep years with guessed
        # parameters as short as possible; rest is done by Braithwaite & Hock
        if hasattr(mbm, 'calibration_timespan') and \
                (mbm.calibration_timespan[0] is not None):
            t0 = '{}-01-01'.format(str(mbm.calibration_timespan[0] - length
                                       - 1))
            t1 = utils.get_begin_last_flexyear(
                dt.datetime(meteo_begin, 12, 31)) - dt.timedelta(days=1)
            time_span = pd.date_range(t0, t1)
        else:
            # defined by the beginning of the meteo data
            time_span = pd.date_range('{}-10-01'.format(
                meteo_begin - length - 1),
                utils.get_begin_last_flexyear(dt.datetime(meteo_begin, 12, 31))
                - dt.timedelta(days=1))

        stacked = None
        heights, widths = gdir.get_inversion_flowline_hw()

        if param_method == 'guess':
            params = list(mbm.cali_params_guess.values())
        elif param_method == 'predict':
            raise NotImplementedError
            # get the random forest model here
            # predict parameters for each of the years in time_span
        else:
            raise ValueError('Parameter determination method "{}" not '
                             'understood.'.format(param_method))

        init_swe = np.zeros_like(heights)
        init_swe.fill(np.nan)
        init_temp = init_swe + cfg.ZERO_DEG_KELVIN
        cover = SnowFirnCover(heights, swe=init_swe, rho=None,
                              origin=time_span[0], temperatures=init_temp,
                              refreezing=False)

        # todo: rubbish if parameters are predicted year-wise \w random forest
        pdict = dict(zip(mbm.cali_params_list, params))
        if isinstance(mbm, utils.SuperclassMeta):
            day_model = mbm(gdir, **pdict, snowcover=None, bias=0.,
                            filename='spinup_climate_daily')
        else:
            day_model = copy.copy(mbm)

        meteo = climate.GlacierMeteo(gdir, filename='spinup_climate_daily')

        mb_now = []
        switch_model = False
        switch_date = pd.Timestamp(meteo_begin, 1, 1)
        for date in time_span:
            if date >= switch_date:
                switch_model = True
                break

            # Get the mass balance and convert to m per day
            tmp = day_model.get_daily_mb(heights, date=date) * \
                cfg.SEC_IN_DAY * cfg.RHO / cfg.RHO_W

            swe = tmp.copy()
            rho = np.ones_like(tmp) * get_rho_fresh_snow_anderson(
                meteo.meteo.sel(time=date).temp.values + cfg.ZERO_DEG_KELVIN)
            temperature = swe.copy()

            cover.ingest_balance(swe, rho, date, temperature)
            cover.densify_snow_anderson(date)

            if date.day == 30 and date.month == 4:
                cover.update_temperature_huss()
                cover.apply_refreezing(exhaust=True)

            if (date.month == 10 and date.day == 1) or date == time_span[-1]:
                cover.densify_firn_huss(date)

            mb_now.append(tmp)

        # phase 2 (meteo data present)
        if switch_model is True:
            # switch back to normal climate file
            meteo = climate.GlacierMeteo(gdir)
            if isinstance(mbm, utils.SuperclassMeta):
                if cali is None:
                    day_model_p2 = mbm(gdir, **pdict, snowcover=None, bias=0.)
                else:
                    day_model_p2 = mbm(gdir, snowcover=None, bias=0.)
            else:
                day_model_p2 = copy.copy(mbm)
            for date in pd.date_range(switch_date, time_span[-1]):
                # try:
                #    for p in mbm.cali_params_list:
                #        setattr(day_model_p2, p,
                #                cali.ix[date, day_model_p2.prefix + p])
                # except KeyError:
                #    # go back to guessed/predicted parameters
                #    for p in mbm.cali_params_list:
                #        setattr(day_model_p2, p,
                #        day_model_p2.cali_params_guess[p])

                # Get the mass balance and convert to m per day
                tmp = day_model_p2.get_daily_mb(heights, date=date) * \
                      cfg.SEC_IN_DAY * cfg.RHO / cfg.RHO_W

                swe = tmp.copy()
                rho = np.ones_like(tmp) * get_rho_fresh_snow_anderson(
                    meteo.meteo.sel(
                        time=date).temp.values + cfg.ZERO_DEG_KELVIN)
                temperature = swe.copy()

                cover.ingest_balance(swe, rho, date, temperature)
                cover.densify_snow_anderson(date)

                if date.day == 30 and date.month == 4:
                    cover.update_temperature_huss()
                    cover.apply_refreezing(exhaust=True)

                if (date.month == 10 and date.day == 1) or \
                        date == time_span[-1]:
                    cover.densify_firn_huss(date)

                mb_now.append(tmp)
            sc_list.append(cover.to_dataset())
        else:
            sc_list.append(cover.to_dataset())

        if stacked is not None:
            stacked = np.vstack((stacked, mb_now))
        else:
            stacked = mb_now

        stacked = np.sort(stacked, axis=0)

        mb_for_ds = np.moveaxis(np.atleast_3d(np.array(stacked)), 1, 0)
        mb_for_ds = mb_for_ds[..., np.newaxis]
        # todo: units are hard coded and depend on method used above
        mb_ds = xr.Dataset({'MB': (['fl_id', 'time', 'member', 'model'],
                                   mb_for_ds)},
                           coords={'member': (['member', ],
                                              [mb_for_ds.shape[-2]]),
                                   'model': (['model', ],
                                             [day_model.__name__]),
                                   'time': (['time', ],
                                            pd.to_datetime(time_span)),
                                   'fl_id': (['fl_id', ],
                                             np.arange(len(heights))),
                                   })

        ds_list.append(mb_ds)

    ens_ds = xr.merge(ds_list)
    ens_ds.attrs.update({'id': gdir.rgi_id, 'name': gdir.name,
                         'units': 'm w.e.', 'param_method': param_method})

    gdir.write_pickle(ens_ds, 'mb_spinup')

    # take care of snow
    sc_ds = xr.concat(sc_list, dim=pd.Index([m.__name__ for m in models],
                                            name='model'))

    gdir.write_pickle(sc_ds, 'snow_spinup')
