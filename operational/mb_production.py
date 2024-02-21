"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Log message format
import logging

# Python imports
import os
from glob import glob
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
    if first_day is None:
        mbyear_begin = utils.get_begin_last_flexyear(pd.Timestamp.now())
    else:
        mbyear_begin = first_day

    # if first day is not given, take begin of current MB year
    if first_day is None:
        first_day = mbyear_begin
    else:
        if reset_file is True:
            pass
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
                # probably no snow cover from that day: start from scratch
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
    else:
        return ens_ds, snow_ens


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


@entity_task(log, writes=['mb_prediction_cosmo', 'mb_prediction_ecmwf'])
def make_mb_prediction(gdir: utils.GlacierDirectory,
                       begin_date: pd.Timestamp or None = None,
                       mb_model: MassBalanceModel or None = None,
                       snowcover: SnowFirnCover or None = None,
                       latest_climate: bool = True,
                       constrain_with_bw_prcp_fac: bool = True,
                       climate_suffix: str = '',
                       cali_suffix: str = '', reset_file: bool = True,
                       write: bool = True) -> None:
    """
    Create a mass balance prediction from numerical weather prediction.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        GlacierDirectory to calculate the forecast for.
    begin_date: pd.Timestamp or None, optional
        Date when forecast should begin. This is only needed for experiments.
        Default: None (take date of today as begin).
    mb_model: DailyMassBalanceModel or None
        Mass balance model to generate the forecast with. If None,
        cfg.MASSBALANCE_MODELS are used. Default: None.
    snowcover: SnowFirnCover or None
        The snow cover at the initialization date of the forecast. Default:
        None.
    latest_climate: bool, optional
        Whether to use parameters from the last 30 years only or not.
        # todo: In operational assimilation mode it should use the
           parameters retrieved from the assimilation.
    constrain_with_bw_prcp_fac: bool, optional
        Whether to constrain the used parameters with the latest precipitation
        correction factor as calibrated in the winter mass balance. Default:
        True.
        # todo: In operational assimilation mode it should use
           the parameters retrieved from the assimilation.
    climate_suffix: str, optional
        Suffix used to retrieve the climate (NWP) file, called `climate_suffix`
        for compatibility. Default: '' (no suffix).
    cali_suffix: str, optional
        Suffix used to retrieve the calibration parameters and current mass
        balance files. Default: '' (no suffix).
    reset_file: bool, optional
        Whether to reset the file completely. Default: True (we want a new
        prediction every day).
    write: bool
        Whether or not to write the forecast as netCDF file. Default: True
        (write).

    Returns
    -------
    None
    """

    now_timestamp = pd.Timestamp.today()
    today_date = now_timestamp.date()
    yesterday = today_date - pd.Timedelta(days=1)
    day_before_yesterday = today_date - pd.Timedelta(days=2)
    begin_mbyear = utils.get_begin_last_flexyear(now_timestamp)

    if mb_model:
        mb_models = [mb_model]
    else:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

    # meteo predictions
    nwp = xr.open_dataset(gdir.get_filepath('nwp_daily' + climate_suffix))

    for i, r in enumerate(nwp.member.values):
        nwp.sel(member=r).to_netcdf(gdir.get_filepath(
            'nwp_daily' + climate_suffix, filesuffix='_{}'.format(i)))

    # make sure the NWP file is updated
    nwp_beginday = pd.Timestamp(nwp.time.values[0])
    nwp_time_timestamps = [pd.Timestamp(t) for t in nwp.time.values]
    if ((climate_suffix == '_cosmo') and
        (nwp_beginday not in [today_date, yesterday])) or \
            ((climate_suffix == '_ecmwf') and
             ((now_timestamp - nwp_beginday) > pd.Timedelta(days=5, hours=8))):
        # try and make a new one
        nwp.close()
        climate.make_nwp_files()
        climate.process_nwp_data(gdir)
        nwp = xr.open_dataset(gdir.get_filepath('nwp_daily' + climate_suffix))
        nwp_beginday = pd.Timestamp(nwp.time.values[0])
        nwp_time_timestamps = [pd.Timestamp(t) for t in nwp.time.values]

    # if curr_mb doesn't reach to at least yesterday, first make current MB
    curr = gdir.read_pickle('mb_current', filesuffix=cali_suffix)
    curr_last_day = pd.Timestamp(curr.time[-1].values)
    if (((now_timestamp.hour < 12) or ((now_timestamp.hour == 12) and
                                       (now_timestamp.minute <= 21))) and
        (curr_last_day not in [yesterday, day_before_yesterday])) or (
        ((now_timestamp.hour > 12) or ((now_timestamp.hour == 12) and
                                       (now_timestamp.minute > 21))) and
            (curr_last_day not in [today_date, yesterday])):
        make_mb_current_mbyear(gdir, suffix=cali_suffix)

    # snow cover realizations from the members of make_mb_current_mbyear
    if snowcover is None:
        curr_snow = gdir.read_pickle('snow_current', filesuffix=cali_suffix)
        # COSMO, ECMWF and MeteoSwiss deliveries
        if ((now_timestamp.hour < 12) or ((now_timestamp.hour == 12) and
                                          (now_timestamp.minute < 30))) and \
           ((now_timestamp.hour > 7) or ((now_timestamp.hour == 7) and
                                         (now_timestamp.minute >= 40))):
            # NWP has to be update by then
            try:
                snowcover = curr_snow.sel(time=nwp_beginday-pd.Timedelta(days=2))
            except KeyError:
                # NWP not updated yet
                nwp.close()
                climate.make_nwp_files()
                climate.process_nwp_data(gdir)
                nwp = xr.open_dataset(
                    gdir.get_filepath('nwp_daily' + climate_suffix))
                nwp_beginday = pd.Timestamp(nwp.time.values[0])
                nwp_time_timestamps = [pd.Timestamp(t) for t in
                                       nwp.time.values]
        else:
            if climate_suffix == '_cosmo':
                if (now_timestamp.hour < 12) or ((now_timestamp.hour == 12) and
                                                 (now_timestamp.minute < 30)):
                    snowcover = curr_snow.sel(
                        time=nwp_beginday - pd.Timedelta(days=2))
                else:
                    snowcover = curr_snow.sel(
                        time=nwp_beginday - pd.Timedelta(days=1))
            elif climate_suffix == '_ecmwf':
                snowcover = curr_snow.isel(time=-1)

    # clip ECMWF forecast (it might be some days old)
    if climate_suffix == '_ecmwf':
        # read again (it might be updated)
        curr = gdir.read_pickle('mb_current', filesuffix=cali_suffix)
        nwp = nwp.sel(time=slice(pd.Timestamp(curr.time[-1].values) +
                                 pd.Timedelta(days=1), None))
        nwp_beginday = pd.Timestamp(nwp.time.values[0])
        nwp_time_timestamps = [pd.Timestamp(t) for t in nwp.time.values]

    # make the actual prediction
    stacked = None
    heights, widths = gdir.get_inversion_flowline_hw()
    n_param_sets = 0

    for mbm in mb_models:
        print(mbm.__name__)

        # get parameters from past
        # todo: retrieve current parameters from assimilation!
        pg = ParameterGenerator(
            gdir, mbm, latest_climate=latest_climate, only_pairs=True,
            constrain_with_bw_prcp_fac=constrain_with_bw_prcp_fac,
            bw_constrain_year=begin_mbyear.year + 1, narrow_distribution=0.,
            output_type='array', suffix=cali_suffix)
        param_prod = pg.from_single_glacier()

        # no parameters found, e.g. when using latest_climate = True for A50I06
        if param_prod.size == 0:
            log.error('With current settings (`latest_climate=True`, no '
                      'parameters were found to produce current MB.')
            return
        print('found params with shape: ', param_prod.shape)
        n_param_sets += param_prod.shape[0]

        for ip, params in enumerate(param_prod):
            print(ip)

            # todo: let GlacierMeteo/DailyMassBalanceModelWithSnow handle ensemble input
            # init conditions of snow and params
            pdict = dict(zip(mbm.cali_params_list, params))
            sc = SnowFirnCover.from_dataset(snowcover.sel(model=mbm.__name__)
                                            .isel(member=ip))

            for ir in range(nwp.member.values.size):
                day_model = mbm(
                    gdir, snowcover=sc, bias=0.,
                    filename='nwp_daily' + climate_suffix,
                    filesuffix='_{}'.format(ir), **pdict)

                mb_pred = []
                for date in nwp_time_timestamps:
                    # Get the mass balance and convert to m per day
                    tmp = day_model.get_daily_specific_mb(heights, widths,
                                                          date=date)

                    mb_pred.append(tmp)

                if stacked is not None:
                    stacked = np.vstack((stacked, mb_pred))
                else:
                    stacked = mb_pred

    mb_for_ds = np.atleast_2d(stacked).T
    var_dict = {**{'MB': (['time', 'member'], mb_for_ds)}}
    mb_ds = xr.Dataset(var_dict, coords={'member': (['member'], np.arange(
        len(mb_models) * param_prod.shape[0] * nwp.member.values.size)),
                                         'time': (
                                         ['time'], nwp_time_timestamps)},
                       attrs={'id': gdir.rgi_id, 'name': gdir.name,
                              'units': 'm w.e.'})

    if write:
        mb_ds.mb.append_to_gdir(
            gdir, 'mb_prediction' + climate_suffix + cali_suffix,
            reset=reset_file)


@entity_task(log, writes=['mb_prediction_cosmo', 'mb_prediction_ecmwf'])
def make_mb_prediction_fast(gdir: utils.GlacierDirectory,
                       begin_date: pd.Timestamp or None = None,
                       mb_model: MassBalanceModel or None = None,
                       snowcover: SnowFirnCover or None = None,
                       latest_climate: bool = True,
                       constrain_with_bw_prcp_fac: bool = True,
                       climate_suffix: str = '',
                       cali_suffix: str = '', reset_file: bool = True,
                       write: bool = True) -> None:
    """
    Create a mass balance prediction from numerical weather prediction.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        GlacierDirectory to calculate the forecast for.
    begin_date: pd.Timestamp or None, optional
        Date when forecast should begin. This is only needed for experiments.
        Default: None (take date of today as begin).
    mb_model: DailyMassBalanceModel or None
        Mass balance model to generate the forecast with. If None,
        cfg.MASSBALANCE_MODELS are used. Default: None.
    snowcover: SnowFirnCover or None
        The snow cover at the initialization date of the forecast. Default:
        None.
    latest_climate: bool, optional
        Whether to use parameters from the last 30 years only or not.
        # todo: In operational assimilation mode it should use the
           parameters retrieved from the assimilation.
    constrain_with_bw_prcp_fac: bool, optional
        Whether to constrain the used parameters with the latest precipitation
        correction factor as calibrated in the winter mass balance. Default:
        True.
        # todo: In operational assimilation mode it should use
           the parameters retrieved from the assimilation.
    climate_suffix: str, optional
        Suffix used to retrieve the climate (NWP) file, called `climate_suffix`
        for compatibility. Default: '' (no suffix).
    cali_suffix: str, optional
        Suffix used to retrieve the calibration parameters and current mass
        balance files. Default: '' (no suffix).
    reset_file: bool, optional
        Whether to reset the file completely. Default: True (we want a new
        prediction every day).
    write: bool
        Whether or not to write the forecast as netCDF file. Default: True
        (write).

    Returns
    -------
    None
    """

    now_timestamp = pd.Timestamp.today()
    today_date = now_timestamp.date()
    yesterday = today_date - pd.Timedelta(days=1)
    day_before_yesterday = today_date - pd.Timedelta(days=2)
    begin_mbyear = utils.get_begin_last_flexyear(now_timestamp)

    if mb_model:
        mb_models = [mb_model]
    else:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

    mb_models = [m(gdir, bias=0.) for m in mb_models]

    # meteo predictions
    nwp = xr.open_dataset(gdir.get_filepath('nwp_daily' + climate_suffix))

    for i, r in enumerate(nwp.member.values):
        nwp.sel(member=r).to_netcdf(gdir.get_filepath(
            'nwp_daily' + climate_suffix, filesuffix='_{}'.format(i)))

    # make sure the NWP file is updated
    nwp_beginday = pd.Timestamp(nwp.time.values[0])
    nwp_time_timestamps = [pd.Timestamp(t) for t in nwp.time.values]
    if ((climate_suffix == '_cosmo') and
        (nwp_beginday not in [today_date, yesterday])) or \
            ((climate_suffix == '_ecmwf') and
             ((now_timestamp - nwp_beginday) > pd.Timedelta(days=5, hours=8))):
        # try and make a new one
        nwp.close()
        climate.make_nwp_files()
        climate.process_nwp_data(gdir)
        nwp = xr.open_dataset(gdir.get_filepath('nwp_daily' + climate_suffix))
        nwp_beginday = pd.Timestamp(nwp.time.values[0])
        nwp_time_timestamps = [pd.Timestamp(t) for t in nwp.time.values]

    # if curr_mb doesn't reach to at least yesterday, first make current MB
    curr = gdir.read_pickle('mb_current', filesuffix=cali_suffix)
    curr_last_day = pd.Timestamp(curr.time[-1].values)
    if (((now_timestamp.hour < 12) or ((now_timestamp.hour == 12) and
                                       (now_timestamp.minute <= 21))) and
        (curr_last_day not in [yesterday, day_before_yesterday])) or (
        ((now_timestamp.hour > 12) or ((now_timestamp.hour == 12) and
                                       (now_timestamp.minute > 21))) and
            (curr_last_day not in [today_date, yesterday])):
        make_mb_current_mbyear(gdir, suffix=cali_suffix)

    # snow cover realizations from the members of make_mb_current_mbyear
    if snowcover is None:
        curr_snow = gdir.read_pickle('snow_current', filesuffix=cali_suffix)
        # COSMO, ECMWF and MeteoSwiss deliveries
        if ((now_timestamp.hour < 12) or ((now_timestamp.hour == 12) and
                                          (now_timestamp.minute < 30))) and \
           ((now_timestamp.hour > 7) or ((now_timestamp.hour == 7) and
                                         (now_timestamp.minute >= 40))):
            # NWP has to be update by then
            try:
                snowcover = curr_snow.sel(time=nwp_beginday-pd.Timedelta(days=2))
            except KeyError:
                # NWP not updated yet
                nwp.close()
                climate.make_nwp_files()
                climate.process_nwp_data(gdir)
                nwp = xr.open_dataset(
                    gdir.get_filepath('nwp_daily' + climate_suffix))
                nwp_beginday = pd.Timestamp(nwp.time.values[0])
                nwp_time_timestamps = [pd.Timestamp(t) for t in
                                       nwp.time.values]
        else:
            if climate_suffix == '_cosmo':
                if (now_timestamp.hour < 12) or ((now_timestamp.hour == 12) and
                                                 (now_timestamp.minute < 30)):
                    snowcover = curr_snow.sel(
                        time=nwp_beginday - pd.Timedelta(days=2))
                else:
                    snowcover = curr_snow.sel(
                        time=nwp_beginday - pd.Timedelta(days=1))
            elif climate_suffix == '_ecmwf':
                snowcover = curr_snow.isel(time=-1)

    # clip ECMWF forecast (it might be some days old)
    if climate_suffix == '_ecmwf':
        # read again (it might be updated)
        curr = gdir.read_pickle('mb_current', filesuffix=cali_suffix)
        nwp = nwp.sel(time=slice(pd.Timestamp(curr.time[-1].values) +
                                 pd.Timedelta(days=1), None))
        nwp_beginday = pd.Timestamp(nwp.time.values[0])
        nwp_time_timestamps = [pd.Timestamp(t) for t in nwp.time.values]

    # make the actual prediction
    stacked = None
    heights, widths = gdir.get_inversion_flowline_hw()
    n_param_sets = 0
    mb_list = []
    for mbm in mb_models:
        print(mbm.__name__)

        def tacc_from_alpha_brock(alpha, p1=0.86, p2=0.155):
            # here we can only take the deep snow equation, otherwise it's not unique
            tacc = 10. ** ((alpha - p1) / (-p2))
            # todo: bullshit
            tacc[tacc < 1.] = 1.
            return tacc

        def point_albedo_brock(swe, t_acc, icedist, p1=0.713, p2=0.112,
                               p3=0.442, p4=0.058, a_u=None, d_star=0.024,
                               alpha_max=0.85, ice_alpha_std=0.075):

            if a_u is None:
                a_u = cfg.PARAMS['ice_albedo_default']
            alpha_ds = np.clip((p1 - p2 * np.log10(t_acc)), None, 1.)
            # shallow snow equation
            alpha_ss = np.clip((a_u + p3 * np.exp(-p4 * t_acc)), None,
                               alpha_max)
            # combining deep and shallow
            alpha = (1. - np.exp(-swe / d_star)) * alpha_ds + np.exp(
                -swe / d_star) * alpha_ss
            return alpha



        def melt_braithwaite(psol=None, mu_ice=None, tmean=None, swe=None,
                             prcp_fac=None, tmelt=0., tmax=None, sis=None):
            tempformelt = tmean - tmelt
            tempformelt[tmean <= tmelt] = 0.
            tempformelt = tempformelt[None, :]
            mu = np.ones_like(swe) * mu_ice
            mu_repeat = np.repeat(mu_ice * cfg.PARAMS['ratio_mu_snow_ice'],
                                  swe.shape[-1], axis=-1)
            mu_repeat = np.repeat(mu_repeat, swe.shape[-2], axis=-2)
            mu[np.where(swe > 0.)] = mu_repeat[np.where(swe > 0.)]
            return mu * tempformelt / 1000.

        def melt_hock(psol=None, mu_hock=None, a_ice=None, tmean=None,
                      ipot=None, prcp_fac=None, swe=None, tmelt=0., tmax=None,
                      sis=None):
            tempformelt = tmean - tmelt
            tempformelt[tmean <= tmelt] = 0.
            a = np.ones_like(swe) * a_ice
            a_repeat = np.repeat(a_ice * cfg.PARAMS['ratio_a_snow_ice'],
                                  swe.shape[-1], axis=-1)
            a_repeat = np.repeat(a_repeat, swe.shape[-2], axis=-2)
            a[np.where(swe > 0.)] = a_repeat[np.where(swe > 0.)]
            melt_day = (mu_hock + a * ipot) * tempformelt
            return melt_day / 1000.

        def melt_pellicciotti(psol=None, tf=None, srf=None, tmean=None,
                              sis=None, alpha=None, tmelt=1., prcp_fac=None,
                              tmax=None):
            melt_day = tf * tmean + srf * (1 - alpha) * sis
            melt_day[:, tmean <= tmelt] = 0.

            return melt_day / 1000.

        def melt_oerlemans(psol=None, c0=None, c1=None, tmean=None, sis=None,
                           alpha=None, prcp_fac=None, tmax=None):
            # todo: IMPORTANT: sign of c0 is changed to make c0 positive (log!)
            qmelt = (1 - alpha) * sis - c0 + c1 * tmean
            # melt only happens where qmelt > 0.:
            qmelt = np.clip(qmelt, 0., None)

            # kg m-2 d-1 = W m-2 * s * J-1 kg
            # we want ice flux, so we drop RHO_W for the first...!?
            melt = (qmelt * cfg.SEC_IN_DAY) / cfg.LATENT_HEAT_FUSION_WATER

            return melt / 1000.

        snowcover_model = snowcover.sel(model=mbm.__name__)
        h, w = gdir.get_inversion_flowline_hw()
        pg = ParameterGenerator(gdir, mbm, latest_climate=True,
            only_pairs=True, constrain_with_bw_prcp_fac=False,
            bw_constrain_year=pd.Timestamp.now().year - 1,
            narrow_distribution=0., output_type='array', suffix='')

        param_prod = pg.from_single_glacier()
        param_prod = param_prod[~np.isnan(param_prod).any(axis=1)]

        if mbm.__name__ in ['PellicciottiModel', 'OerlemansModel']:
            alpha = mbm.albedo.alpha
            tacc = tacc_from_alpha_brock(alpha)

            alpha = np.repeat(alpha[None, None, :], nwp.member.size, axis=1)
            alpha = np.repeat(alpha, param_prod.shape[0], axis=0)
            tacc = np.repeat(tacc[None, None, :], nwp.member.size, axis=1)
            tacc = np.repeat(tacc, param_prod.shape[0], axis=0)

            sis_scale_fac = xr.open_dataarray(
                gdir.get_filepath('sis_scale_factor')).values
            # make leap year compatible
            sis_scale_fac = np.hstack(
                [sis_scale_fac, np.atleast_2d(sis_scale_fac[:, -1]).T])

        # no parameters found, e.g. when using latest_climate = True for A50I06
        if param_prod.size == 0:
            log.error('With current settings (`latest_climate=True`, no '
                      'parameters were found to produce current MB.')
            return
        print('found params with shape: ', param_prod.shape)
        n_param_sets += param_prod.shape[0]

        swe_repeat = (
            np.repeat(np.nansum(snowcover_model.swe.values[:len(param_prod)],
                                axis=-1)[:, None, :],
                      nwp.member.size, axis=1))

        mb_model_list = []
        for date in nwp.time.values:
            date = pd.Timestamp(date)


            temp_at_hgts = climate.get_temperature_at_heights(
                nwp.sel(time=date).temp.values,
                nwp.sel(time=date).tgrad.values, nwp.ref_hgt, h)
            if mbm.__name__ in ['PellicciottiModel', 'OerlemansModel']:
                tmax_at_hgts = climate.get_temperature_at_heights(
                    nwp.sel(time=date).tmax.values,
                    nwp.sel(time=date).tgrad.values, nwp.ref_hgt, h)
            prcp_at_hgts = climate.get_precipitation_at_heights(
                nwp.sel(time=date).prcp.values,
                nwp.sel(time=date).pgrad.values, nwp.ref_hgt, h)

            if mbm.__name__ in ['PellicciottiModel', 'OerlemansModel']:
                alpha = point_albedo_brock(swe_repeat, tacc, swe_repeat==0.)
                tacc += tmax_at_hgts
                ssf = sis_scale_fac[None, :, date.dayofyear]

            if mbm.__name__ == 'BraithwaiteModel':
                melt = melt_braithwaite(mu_ice=param_prod[:, 0][:, None, None],
                                        tmean=temp_at_hgts, swe=swe_repeat,
                                        tmelt=0.)
            elif mbm.__name__ == 'HockModel':
                ipot = mbm.ipot[:, np.clip(date.dayofyear - 1, None, 364)]
                ipot = np.repeat(ipot[None, None, :], nwp.member.size, axis=1)
                ipot = np.repeat(ipot, param_prod.shape[0], axis=0)
                melt = melt_hock(mu_hock=param_prod[:, 0][:, None, None],
                                 a_ice=param_prod[:, 1][:, None, None],
                                 tmean=temp_at_hgts, ipot=ipot, tmelt=0.,
                                 swe=swe_repeat)
            elif mbm.__name__ == 'PellicciottiModel':
                sis = nwp.sel(time=date).sis.values[:, None] * ssf
                #sis = np.repeat(sis[None, None, :], nwp.member.size, axis=1)
                #sis = np.repeat(sis, param_prod.shape[0], axis=0)
                sis = np.repeat(sis[None, :], param_prod.shape[0], axis=0)
                melt = melt_pellicciotti(tf=param_prod[:, 0][:, None, None],
                                         srf=param_prod[:, 1][:, None, None],
                                         tmean=temp_at_hgts,
                              sis=sis, tmelt=1., alpha=alpha)
            elif mbm.__name__ == 'OerlemansModel':
                sis = nwp.sel(time=date).sis.values[:, None] * ssf
                # sis = np.repeat(sis[None, None, :], nwp.member.size, axis=1)
                # sis = np.repeat(sis, param_prod.shape[0], axis=0)
                sis = np.repeat(sis[None, :], param_prod.shape[0], axis=0)
                melt = melt_oerlemans(c0=-param_prod[:, 0][:, None, None],
                                      c1=param_prod[:, 1][:, None, None],
                                      tmean=tmax_at_hgts, sis=sis,
                           alpha=alpha, tmax=None)

            else:
                raise NotImplementedError(
                    'Chosen MassBalanceModel not yet available for fast mass '
                    'balance NWP production.')

            frac_solid = climate.get_fraction_of_snowfall_linear(temp_at_hgts)
            accum = prcp_at_hgts / 1000. \
                    * frac_solid \
                    * mbm.prcp_fac_cycle_multiplier[date.dayofyear - 1] \
                    * mbm.snowdistfac.sel(time=date, method='nearest').D.values\
                    * mbm.prcp_fac[mbm.prcp_fac[~pd.isnull(mbm.prcp_fac)].index.get_loc(date,method='nearest')]
            mb = accum - melt
            swe_repeat += mb
            swe_repeat = np.clip(swe_repeat, 0., None)
            mb_model_list.append(np.average(mb, weights=w, axis=-1))
        mb_list.append(np.array(mb_model_list).T)

    #mb_for_ds = np.atleast_2d(mb_list).T
    var_dict = {**{'MB': (['time', 'member'], np.array(mb_list).reshape((-1, np.array(mb_list).shape[-1])).T)}}
    mb_ds = xr.Dataset(var_dict, coords={'member': (['member'], np.arange(
        n_param_sets * nwp.member.values.size)),
                                         'time': (
                                             ['time'], nwp_time_timestamps)},
                       attrs={'id': gdir.rgi_id, 'name': gdir.name,
                              'units': 'm w.e.'})

    if write:
        mb_ds.mb.append_to_gdir(gdir,
            'mb_prediction' + climate_suffix + cali_suffix, reset=reset_file)


class MBYearHandler(object):
    """
    A class controlling important dates around the mass budget year.
    """

    def __init__(self, mb_clim, mb_current, model, mb_year='current',
                 clim_bgday_method='fixed'):
        """

        Parameters
        ----------
        mb_clim: xr.Dataset
            Dataset containing a mass balance time series
        mb_current: xr.Dataset
            Dataset containing the mass balance since an estimated begin of the
             last mass budget year.
        model: `py:class:crampon.core.models.massbalance.MassBalanceModel` or
               str
            Model MB to retrieve the date characteristics for.
        mb_year: str, int, float
            Determines the mass budget year for which date metrics shall be
            evaluated. Can be either of "current" or a number defining a mass
            budget year. According to the convention, e.g. the mass budget year
            2016/2017 would be described by the mb_year 2017. Default:
            "current" take current mass budget year as defined by October 1st.
        clim_bgday_method: str
            Method how to determine the beginning of the "climatological"
            budget year. Allowed are "fixed" and "modeled". Default: "fixed".
        """
        self.model = model
        if isinstance(model, utils.SuperclassMeta):
            model_pre = model.prefix
        elif isinstance(model, str):
            model_pre = model + '_'
        else:
            raise ValueError('Model parameter must be string or instance of a '
                             'crampon.core.models.massbalance.DailyMassbalance'
                             'Model.')

        self.mb_str = model_pre + 'MB'

        self._mb_clim = mb_clim
        self._mb_current = mb_current

        self.begin_fixdate = utils.get_begin_last_flexyear(dt.datetime.today(),
                                                           10, 1)
        self.clim_bgday_method = clim_bgday_method

        self._begin_clim = None
        self._end_clim = None
        self._begin_current = None
        self._end_current = None
        self._mbyear = None

    @lazy_property
    def mb_year(self, mb_year):
        """Mass budget year."""
        if mb_year == 'current':
            self._mbyear = dt.datetime.now().year
        elif isinstance(mb_year, (int, float)):
            self._mbyear = mb_year
        else:
            raise ValueError('Value for mass budget year specification not '
                             'accepted.')

    @lazy_property
    def begin_clim(self):
        """Begin of the MB climatology."""
        # todo: does it make sense to drop n?
        # clim_cumsum = self._mb_clim.drop_sel('n').apply(
        #    lambda x: MassBalance.time_cumsum(x))
        clim_cumsum = self._mb_clim.drop_sel('member').map(
            lambda x: MassBalance.time_cumsum(x))
        return pd.to_datetime(min(clim_cumsum[self.mb_str]).time.values)

    @lazy_property
    def begin_current(self):
        """begin of the current mass budget year."""
        # begin_current, _ = find_begin_mbyear(self._mb_clim, self._mb_current,
        # self.model)

        # find real begin of MB year: find min around guessed begin_mbyear
        # todo: does it make sense to drop n?
        clim_cumsum = self._mb_clim.drop_sel('n').map(
            lambda x: MassBalance.time_cumsum(x)).isel(time=slice(-366, -1))
        clim_cumsum = self._mb_clim.drop_sel('member').map(
            lambda x: MassBalance.time_cumsum(x)).isel(time=slice(-366, -1))
        # now_cumsum = self._mb_current.sel(quantile=0.5).drop(
        #    ['prcp_fac', 'mu_ice', 'mu_snow', 'quantile'])
        now_cumsum = self._mb_current.sel(quantile=0.5)
        concat = xr.auto_combine([clim_cumsum, now_cumsum.map(
            lambda x: x + clim_cumsum.isel(time=-1)[self.mb_str].values[0])],
                                 concat_dim='time')
        guess_month = 10
        guess_day = 1
        guess_beg = utils.get_begin_last_flexyear(dt.datetime.today(),
                                                  guess_month, guess_day)
        # we look for min from June 1st to December 31st
        search_win \
        = (guess_beg - dt.timedelta(days=90),
                      guess_beg + dt.timedelta(days=90))
        search_mb = concat.sel(time=slice(search_win[0], search_win[1]))
        bg_date_hydro = pd.to_datetime(min(search_mb[self.mb_str]).time.values)

        # add piece from clim or redo current
        # min_current_dt = pd.to_datetime(min(self._mb_current.time.values))
        # if bg_date_hydro > min_current_dt:
        #    make_mb_current_mbyear(g, bg_date_hydro, time_elap, snow_cond)
        #    mb_now_cs, curr_snow =
        # elif bg_date_hydro < min_current_dt:
        #    clim_piece = mb_ds.sel(time=slice(bg_date_hydro, min_current_dt
        #                                      - dt.timedelta(days=1)))
        #    clim_piece = xr.concat(len(QUANTILES) * [clim_piece],
        #                           dim='quantile')
        #    clim_piece['quantile'] = QUANTILES
        #    mb_now_cs_prelim = self._mb_current.apply(
        #        lambda x: x + clim_piece.isel(time=-1)[mb_str].values[0])
        #    mb_now_cs = xr.concat([clim_piece, mb_now_cs_prelim.drop_sel(
        #        ['prcp_fac', 'mu_ice', 'mu_snow'])], dim='time')
        # else:
        #    mb_now_cs = self._mb_current.copy()

        return bg_date_hydro

    @lazy_property
    def end_current(self):
        """End of the current mass bduget year."""
        return utils.get_cirrus_yesterday()

    @lazy_property
    def begin_current_plot(self):
        """Begin of the current plot."""
        return self.begin_current

    @lazy_property
    def begin_clim_plot(self):
        """Begin of the climate plot."""
        if self.clim_bgday_method == 'fixed':
            return self.begin_fixdate
        elif self.clim_bgday_method == 'modeled':
            current_year = dt.datetime.now().year
            guess_max = dt.datetime(current_year, 6, 1)
            hyears = self._mb_clim.mb.make_hydro_years(self._mb_clim,
                                                       guess_max.month,
                                                       guess_max.day)
            hdoys = self._mb_clim.mb.make_hydro_doys(hyears)
            mb_cs = self._mb_clim.groupby(hyears).apply(
                lambda x: MassBalance.time_cumsum(x))
            stack_mean = mb_cs.groupby(hdoys) \
                .apply(lambda x: x.mean())
            return guess_max + dt.timedelta(
                days=min(stack_mean[self.mb_str]).hydro_doys.item())

    def to_csv(self):
        """Write the important dates to as CSV."""
        raise NotImplementedError


# Necessary BEFORE main to make multiprocessing work on Windows
# Initialize CRAMPON (and OGGM, hidden in cfg.py)
# cfg.initialize(file='~\\crampon\\sandbox\\'
#                    'CH_params.cfg')
# cfg.PATHS['working_dir'] = \
#    '~\\modelruns\\Matthias_new'
