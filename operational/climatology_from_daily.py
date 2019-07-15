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
                 reset=False):
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
        mb_models = [mb_model]
    else:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

    today = dt.datetime.now()
    if endyear is None:
        t_month, t_day = today.month, today.day
        if (t_month >= 10) and (t_day >= 1):
            endyear = today.year
        else:
            endyear = today.year - 1

    heights, widths = gdir.get_inversion_flowline_hw()

    ds_list = []
    sc_list = []
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
            day_model = mbm(gdir, bias=0.)
        else:
            day_model = copy.copy(mbm)
        mb = []
        run_span = pd.date_range(begin_clim, end_clim)
        for date in run_span:
            # Get the mass balance and convert to m per day
            tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)

            # todo: when are clever moments to store the snowcover?
            if date == end_clim:
                sc_list.append(day_model.snowcover.to_dataset())

        mb_for_ds = np.moveaxis(np.atleast_3d(np.array(mb)), 1, 0)
        mb_ds = xr.Dataset({'MB': (['time', 'member', 'model'], mb_for_ds)},
                           coords={'member': (['member', ],
                                              np.arange(mb_for_ds.shape[1])),
                                   'model': (['model', ],
                                             [day_model.__name__]),
                                   'time': (['time', ], run_span),
                                   })
        ds_list.append(mb_ds)

    # merge all models together
    merged = xr.merge(ds_list)
    # todo: units are hard coded and depend on method used above
    merged.attrs.update({'id': gdir.rgi_id, 'name': gdir.name,
                         'units': 'm w.e.'})
    if write:
        merged.mb.append_to_gdir(gdir, 'mb_daily', reset=reset)

    # take care of snow
    sc_ds = xr.concat(sc_list, dim=pd.Index([m.__name__ for m in mb_models],
                                            name='model'))

    return merged, sc_ds


def make_mb_current_mbyear(gdir, begin_mbyear, mb_model=None, snowcover=None,
                           write=True, reset=False):
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

    curr_year_span = pd.date_range(start=begin_str, end=yesterday_str,
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
                day_model_curr = mbm(gdir, **pdict, snowcover=sc, bias=0.)
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
