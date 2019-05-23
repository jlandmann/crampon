"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Log message format
import logging

# Python imports
import os
import itertools
# Libs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import datetime as dt
# Locals
import crampon.cfg as cfg
from crampon import workflow
from crampon import tasks
from crampon.workflow import execute_entity_task
from crampon import graphics, utils
from crampon.core.models.massbalance import BraithwaiteModel, \
    PellicciottiModel, OerlemansModel, MassBalance, SnowFirnCover
from operational import climatology_from_daily as cfd
from itertools import product
import copy

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

if __name__ == '__main__':

    reset_climatology = False

    cfg.initialize(
        'c:\\users\\johannes\\documents\\crampon\\sandbox\\CH_params.cfg')

    # Local paths (where to write output and where to download input)
    PLOTS_DIR = os.path.join(cfg.PATHS['working_dir'], 'plots')

    # to suppress all NaN warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Currently OGGM wants some directories to exist
    # (maybe I'll change this but it can also catch errors in the user config)
    utils.mkdir(cfg.PATHS['working_dir'])

    # Mauro's DB ad disguised in RGI
    glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
    rgidf = gpd.read_file(glaciers)

    cfg.PARAMS['continue_on_error'] = False
    cfg.PARAMS['use_multiprocessing'] = False
    cfg.PARAMS['mp_processes'] = 2

    # where problems occur
    problem_glaciers_sgi = ['RGI50-11.00761-0', 'RGI50-11.00794-1',
                            'RGI50-11.01509-0', 'RGI50-11.01538-0',
                            'RGI50-11.01956-0', 'RGI50-11.B6302-1',
                            'RGI50-11.02552-0', 'RGI50-11.B3643n',
                            'RGI50-11.02576-0', 'RGI50-11.02663-0',
                            'RGI50-11.A12E09-0', 'RGI50-11.B3217n',
                            'RGI50-11.A14I03-3', 'RGI50-11.A14G14-0',
                            'RGI50-11.A54I17n-0', 'RGI50-11.A14F13-4',
                            'RGI50-11.B4616-0',  # 'bottleneck' polygons
                            'RGI50-11.02848',
                            # ValueError: no min-cost path found
                            'RGI50-11.01382',
                            # init fls: assert len(hgts) >= 5
                            'RGI50-11.01621',
                            # ValueErr:start pts within cost arr
                            'RGI50-11.01805',
                            # gets stuck at initialize_flowlines
                            'RGI50-11.B3511n', 'RGI50-11.B3603',
                            'RGI50-11.C3509n',
                            'RGI50-11.A10G14-0', 'RGI50-11.B9521n-0',
                            'RGI50-11.02431-1', 'RGI50-11.A51G14-1',
                            'RGI50-11.A51H05-0', 'RGI50-11.02815-1',
                            'RGI50-11.01853-0', 'RGI50-11.01853-0',
                            'RGI50-11.B1610n-1', 'RGI50-11.A54L20-15',
                            'RGI50-11.01051-0', 'RGI50-11.A13J03-0',
                            'RGI50-11.01787-1', 'RGI50-11.B7412-1',
                            'RGI50-11.A55C21n', 'RGI50-11.B5613n',
                            'RGI50-11.B4515n', 'RGI50-11.A55B37n',
                            'RGI50-11.B8603n', 'RGI50-11.C0304-1',
                            'RGI50-11.A14E06n', 'RGI50-11.B4637n',
                            'RGI50-11.B8529n-1', 'RGI50-11.B7404',
                            'RGI50-11.A54G45n', 'RGI50-11.B4716n',
                            'RGI50-11.A54M51n', 'RGI50-11.A13L04-0',
                            'RGI50-11.A51H16-0', 'RGI50-11.A12L03-1',
                            'RGI50-11.B4636n', 'RGI50-11.C5104',
                            'RGI50-11.C9205', 'RGI50-11.B9513',
                            'RGI50-11.B9521n-1', 'RGI50-11.B9523n']  # border

    rgidf = rgidf[~rgidf.RGIId.isin(problem_glaciers_sgi)]
    rgidf = rgidf.sort_values(by='Area', ascending=True)
    rgidf = rgidf[rgidf.Area >= 0.0105]

    log.info('Number of glaciers: {}'.format(len(rgidf)))

    # Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf, reset=False, force=False)
    utils.joblib_read_climate_crampon.clear()
    # Preprocessing tasks
    task_list = [
        #tasks.glacier_masks,
        #tasks.compute_centerlines,
        #tasks.initialize_flowlines,
        #tasks.compute_downstream_line,
        #tasks.catchment_area,
        #tasks.catchment_intersections,
        #tasks.catchment_width_geom,
        #tasks.catchment_width_correction,
        tasks.process_custom_climate_data,

    ]
    for task in task_list:
        execute_entity_task(task, gdirs)

    failure_list = []
    mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

    for g in gdirs:
        print(g.rgi_id)
        for mi, mbm in enumerate(mb_models):

            try:
                cali_df = g.get_calibration(mbm)
                day_model = mbm(g, bias=0.)
            except FileNotFoundError:
                try:
                    day_model = mbm(g, **mbm.cali_params_guess, bias=0.)
                # no inversion flowlines or similar
                except (FileNotFoundError, ModuleNotFoundError) as e:
                    try:
                        # try to produce the data
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
                            execute_entity_task(task, [g])
                        day_model = mbm(g, **mbm.cali_params_guess, bias=0.)
                    except:  # we have to give up
                        failure_list.append((g.rgi_id, e))
                        log.warn('Could not make MB for {}: '
                                 '{}'.format(g.rgi_id, e))
                        print(len(failure_list))
                        continue

            print('make clim', dt.datetime.now())
            # get all we can within calibration period
            mb_clim, snowcov_clim = cfd.make_mb_clim(g, mb_model=copy.copy(
                day_model), reset=False)

            # MB of this budget year
            begin_mbyear = pd.to_datetime(mb_clim.time[-1].values) + \
                           dt.timedelta(days=1)

            # set begin of mb year to Oct 1st & adjust snow conditions
            bg_date_hydro = dt.datetime(begin_mbyear.year, 10, 1)

            yesterday = utils.get_cirrus_yesterday()
            yesterday_str = yesterday.strftime('%Y-%m-%d')
            begin_str = begin_mbyear.strftime('%Y-%m-%d')

            curr_year_span = pd.date_range(start=begin_str, end=yesterday_str,
                                           freq='D')

            ds_list = []

            stacked = None
            heights, widths = g.get_inversion_flowline_hw()

            param_prod = [np.linspace(0.5 * x , 1.5 * x, 20) for x in
                          day_model.cali_params_guess.values()]

            # todo: select by model or start with ensemble median snowcover?
            sc = SnowFirnCover.from_dataset(
                snowcov_clim.sel(model=mbm.__name__))

            it = 0
            for params in np.array(param_prod).T:
                print(it, dt.datetime.now())
                it += 1

                pdict = dict(zip(mbm.cali_params_list, params))
                day_model_curr = mbm(g, **pdict, snowcover=sc, bias=0.)

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
            mbcurr_ds = xr.Dataset(
                {'MB': (['time', 'member', 'model'], mb_for_ds)},
                coords={'member': (['member', ],
                                   np.arange(mb_for_ds.shape[1])),
                        'model': (['model', ],
                                  [day_model_curr.__name__]),
                        'time': (['time', ],
                                 pd.to_datetime(curr_year_span)),
                        })

            mbcurr_ds.attrs.update({'id': g.rgi_id, 'name': g.name,
                                 'units': 'm w.e.'})
            if mi == 0:
                mbcurr_ds.mb.append_to_gdir(g, 'mb_current', reset=True)
            else:
                mbcurr_ds.mb.append_to_gdir(g, 'mb_current', reset=False)


        # read the freshly made MB
        mb_clim = g.read_pickle('mb_daily')
        mb_current = g.read_pickle('mb_current')
        first_occ = mb_clim.sel(time=(
                    (mb_clim.time.dt.month == bg_date_hydro.month) & (
                        mb_clim.time.dt.day == bg_date_hydro.day))).time[
            0].values

        mb_clim = mb_clim.sel(time=slice(pd.to_datetime(first_occ), None))

        # insert attributes again...they get lost when grouping!?
        clim_quant = mb_clim.mb.make_cumsum_quantiles()
        now_quant = mb_current.mb.make_cumsum_quantiles()

        # OGGM standard plots
        if PLOTS_DIR == '':
            exit()
        utils.mkdir(PLOTS_DIR)
        bname = os.path.join(PLOTS_DIR, g.rgi_id.split('.')[1] + '_')

        graphics.plot_cumsum_climatology_and_current(clim=mb_clim,
                                                     current=mb_current, loc=3)
        plt.savefig(bname + 'mb_dist_{}.png'.format('ensemble'), dpi=1000)
        graphics.plot_cumsum_climatology_and_current(clim=mb_clim,
                                                     current=mb_current,
                                                     loc=3)
        plt.savefig(bname + 'mb_dist_{}_prev.png'.format('ensemble'), dpi=40)
        graphics.plot_interactive_mb_spaghetti_html(g, PLOTS_DIR)


