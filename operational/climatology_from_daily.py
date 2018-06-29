"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Log message format
import logging

# Python imports
import os
import glob
# Libs
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime as dt
from itertools import product
# Locals
import crampon.cfg as cfg
from crampon import workflow
from crampon import tasks
from crampon.workflow import execute_entity_task
from crampon import graphics, utils
from crampon.core.models.massbalance import DailyMassBalanceModel, BraithwaiteModel
import numpy as np

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)
log.setLevel('DEBUG')


# Initialize CRAMPON (and OGGM, hidden in cfg.py)
cfg.initialize(file='C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\'
                    'CH_params.cfg')

# Local paths (where to write output and where to download input)
PLOTS_DIR = os.path.join(cfg.PATHS['working_dir'], 'plots')

# Currently OGGM wants some directories to exist
# (maybe I'll change this but it can also catch errors in the user config)
utils.mkdir(cfg.PATHS['working_dir'])

# Read one test glacier
#testglacier = 'C:\\Users\\Johannes\\Documents\\data\\outlines\\RGI\\' \
#              'subset_CH\\subset_CH.shp'
# Mauro's DB ad disguised in RGI
#glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_in_RGI_disguise_entities_old_and_new.shp'
glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
rgidf = gpd.read_file(glaciers)
#rgidf = rgidf[rgidf.RGIId == 'RGI50-11.00536']

# where problems occur
problem_glaciers_sgi = ['RGI50-11.00761-0', 'RGI50-11.00794-1',
                         'RGI50-11.01509-0', 'RGI50-11.01538-0',
                        'RGI50-11.01956-0', 'RGI50-11.B6302-1',
                         'RGI50-11.02552-0', 'RGI50-11.B3643n',
                        'RGI50-11.02576-0', 'RGI50-11.02663-0',
                         'RGI50-11.A12E09-0','RGI50-11.B3217n',
                         'RGI50-11.A14I03-3', 'RGI50-11.A14G14-0',
                        'RGI50-11.A54I17n-0', 'RGI50-11.A14F13-4',
                        'RGI50-11.B4616-0',   # 'bottleneck' polygons
                        'RGI50-11.02848',  # ValueError: no minimum-cost path was found to the specified end point (compute_centerlines)
                        'RGI50-11.01382',  # AssertionError in initialize flowlines : assert len(hgts) >= 5
                        'RGI50-11.01621',  # ValueError: start points must all be within the costs array
                        'RGI50-11.01805',  # gets stuck at initialize_flowlines
                        'RGI50-11.B3511n', 'RGI50-11.B3603', 'RGI50-11.C3509n',
                        'RGI50-11.A10G14-0', 'RGI50-11.B9521n-0',
                        'RGI50-11.02431-1', 'RGI50-11.A51G14-1',
                        'RGI50-11.A51H05-0', 'RGI50-11.02815-1',
                        'RGI50-11.01853-0', 'RGI50-11.01853-0',
                        'RGI50-11.B1610n-1', 'RGI50-11.A54L20-15',
                        'RGI50-11.01051-0', 'RGI50-11.A13J03-0',
                        'RGI50-11.01787-1', 'RGI50-11.B7412-1',
                        #
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
                        'RGI50-11.B9521n-1', 'RGI50-11.B9523n']  # too close to border

rgidf = rgidf[~rgidf.RGIId.isin(problem_glaciers_sgi)]
rgidf = rgidf.sort_values(by='Area', ascending=False)
rgidf = rgidf[rgidf.Area >= 0.0105]
#rgidf = rgidf.tail(50)
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A10G05','RGI50-11.B4504',
#'RGI50-11.A55F03', 'RGI50-11.B4312n-1', 'RGI50-11.B5616n-1', 'RGI50-11.C1410'])] # just to have one REFMB glacier
rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])] # Gries
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4312n-1'])] # Rhone


log.info('Number of glaciers: {}'.format(len(rgidf)))

# necessary to make multiprocessing work on Windows
if __name__ == '__main__':

    # Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf, reset=False, force=False)

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

    for g in gdirs:

        #utils.get_local_dems(g)

        # remove as soon as mustar is daily/calibration is correct!
        day_model = BraithwaiteModel(g, bias=0.)

        ######################################################################
        heights, widths = g.get_inversion_flowline_hw()

        # number of experiments (list!)
        exp = [1]

        mb = []
        print(dt.datetime.now())
        bgmon_hydro = cfg.PARAMS['bgmon_hydro']
        bgday_hydro = cfg.PARAMS['bgday_hydro']

        cali_df = pd.read_csv(g.get_filepath('calibration'), index_col=0,
                              parse_dates=[0])
        begin_clim = cali_df.index[0]
        end_clim = cali_df.index[-1]

        for date in pd.date_range(begin_clim, end_clim):

            # Get the mass balance and convert to m per day
            tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)

        mb_ds = xr.Dataset({'MB': (['time', 'n'],
                                   np.atleast_2d(mb).T)},
                           coords={'n': (['n'], exp),
                                   'time': pd.to_datetime(day_model.time_elapsed)},
                           attrs={'id': g.rgi_id,
                                  'name': g.name})
        print(dt.datetime.now())
        # save intermediate results
        g.write_pickle(mb_ds, 'mb_daily')

        snow_cond = day_model.snow
        time_elap = day_model.time_elapsed
        #######################################################################

        # Remove here beginning of file until first begin of hydro year
        first_occ = mb_ds.sel(time=((mb_ds.time.dt.month == bgmon_hydro) &
                                    (mb_ds.time.dt.day == bgday_hydro))).time[
            0].values

        mb_ds = mb_ds.sel(time=slice(pd.to_datetime(first_occ), None))
        hydro_years = xr.DataArray([t.year if ((t.month < bgmon_hydro) or
                                               ((t.month == bgmon_hydro) &
                                                (t.day < bgday_hydro)))
                                    else (t.year + 1) for t in
                                    mb_ds.indexes['time']],
                                   dims='time', name='hydro_years',
                                   coords={'time': mb_ds.time})
        _, cts = np.unique(hydro_years, return_counts=True)
        doys = [list(range(1, c + 1)) for c in cts]
        doys = [i for sub in doys for i in sub]
        hydro_doys = xr.DataArray(doys, dims='time', name='hydro_doys',
                                   coords={'time': mb_ds.time})


        def custom_cumsum(x):
            return x.cumsum(dim='time', skipna=True)


        def custom_quantiles(x, qs=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
            return x.quantile(qs)

        mb_cumsum = mb_ds.groupby(hydro_years).apply(
            lambda x: custom_cumsum(x))
        quant = mb_cumsum.groupby(hydro_doys) \
                .apply(lambda x: custom_quantiles(x))

        # insert attributes again...they get lost when grouping!?
        quant.attrs.update(mb_ds.attrs)

        # OGGM standard plots
        if PLOTS_DIR == '':
            exit()
        utils.mkdir(PLOTS_DIR)
        bname = os.path.join(PLOTS_DIR, g.rgi_id + '_')

        ################################################
        # MAKE MB UP TO NOW

        # QUICK'N'DIRTY
        now = dt.datetime.now()
        if now.hour >= 12 and now.minute >= 30:
            yesterday = (now - dt.timedelta(1))
        else:
            yesterday = (now - dt.timedelta(2))

        yesterday_str = yesterday.strftime('%Y-%m-%d')

        begin = utils.get_begin_last_flexyear(yesterday,
                                              start_month=bgmon_hydro,
                                              start_day=bgday_hydro)
        begin_str = begin.strftime('%Y-%m-%d')

        curr_year_span = pd.date_range(start=begin_str, end=yesterday_str,
                                     freq='D')

        #############################
        # temporary
        # period to extrapolate the parameters
        extrap_period = (dt.datetime(1961, bgmon_hydro, bgday_hydro),
                         dt.datetime(2017, bgmon_hydro, bgday_hydro))
        param_df = cali_df[(cali_df.index >= extrap_period[0]) &
                           (cali_df.index <= extrap_period[1])]

        stacked = None
        # same quantiles for all parameters (mu_ice & prcp_fac correlated (0.8)
        mu_ice_exp = custom_quantiles(cali_df.mu_ice)
        mu_snow_exp = custom_quantiles(cali_df.mu_snow)
        # TODO: Constrain here prcp_fac for measured glacier if it's "known" after WB campaign
        prcp_fac_exp = custom_quantiles(cali_df.prcp_fac)

        for mu_ice_curr, mu_snow_curr, prcp_fac_curr in zip(mu_ice_exp.values,
                                                            mu_snow_exp.values,
                                                            prcp_fac_exp.values):

            day_model_curr = BraithwaiteModel(g, mu_ice=mu_ice_curr,
                                              mu_snow=mu_snow_curr,
                                              prcp_fac=prcp_fac_curr, bias=0.)

            # Take care of overlap between cali and current MB year
            day_model_curr.time_elapsed = time_elap.difference(curr_year_span)
            day_model_curr.snow = np.compress(
                ~(time_elap.isin(curr_year_span)), snow_cond, axis=0)

            mb_now = []
            for date in curr_year_span:
                date = date.to_pydatetime()
                # Get the mass balance and convert to m per day
                tmp = day_model_curr.get_daily_specific_mb(heights, widths,
                                                      date=date)
                mb_now.append(tmp)

            mb_now_cs_arr = np.cumsum(mb_now)

            if stacked is not None:
                stacked = np.vstack((stacked, mb_now_cs_arr))
            else:
                stacked = mb_now_cs_arr

        mb_now_cs = xr.Dataset({'MB': (['time', 'n'],
                                       np.atleast_2d(stacked).T),
                                'mu_ice': (['n'], mu_ice_exp.values),
                                'mu_snow': (['n'], mu_snow_exp.values),
                                'prcp_fac': (['n'], prcp_fac_exp.values)},
                               coords={'n': (['n'], mu_ice_exp.index.values),
                                       'time': pd.to_datetime(curr_year_span)},
                               attrs={'id': g.rgi_id, 'name': g.name})
        g.write_pickle(mb_now_cs, 'mb_current')

        ##################################################

        graphics.plot_cumsum_climatology_and_current(quant, current=mb_now_cs,
                                                     loc=3)
        plt.savefig(bname + 'test_new.png', dpi=1000)
        graphics.plot_googlemap(g)
        plt.savefig(bname + 'googlemap.png')
        plt.close()
        graphics.plot_domain(g)
        plt.savefig(bname + 'domain.png')
        plt.close()
        graphics.plot_centerlines(g)
        plt.savefig(bname + 'centerlines.png')
        plt.close()
        graphics.plot_catchment_width(g, corrected=True)
        plt.savefig(bname + 'catchment_width.png')
        plt.close()
        graphics.plot_catchment_areas(g)
        plt.savefig(bname + 'catchment_areas.png')
        plt.close()

    # Write out glacier statistics
    df = utils.glacier_characteristics(gdirs)
    fpath = os.path.join(cfg.PATHS['working_dir'], 'glacier_char.csv')
    df.to_csv(fpath)
