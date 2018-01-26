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
import datetime
from itertools import product
# Locals
import crampon.cfg as cfg
from crampon import workflow
from crampon import tasks
from crampon.workflow import execute_entity_task
from crampon import graphics, utils
from crampon.core.models.massbalance import DailyMassBalanceModel
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
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.00638'])] # just to have one REFMB glacier
rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])]


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

        # remove as soon as mustar is daily/calibration is correct!
        prcp_fac = 1.4    # tuned manually to 1284 mm (mean winter balance)
        mu_star = 8.0    # tuned manually to  -1001 mm (mean annual balance)
        print(mu_star, prcp_fac)

        day_model = DailyMassBalanceModel(g, mu_star=mu_star,
                                          prcp_fac=prcp_fac, bias=0.)

        heights, widths = g.get_inversion_flowline_hw()

        # number of experiments (list!)
        exp = [1]

        mb = []
        for date in day_model.tspan_in:

            # Get the mass balance and convert to m per day
            tmp = day_model.get_daily_specific_mb(heights, widths, date=date)
            mb.append(tmp)

        mb_ds = xr.Dataset({'MB': (['time', 'n'],
                                   np.atleast_2d(mb).T)},
                           coords={'n': (['n'], exp),
                                   'time': pd.to_datetime(day_model.tspan_in)},
                           attrs={'prcp_fac': prcp_fac,
                                  'mu_star': mu_star,
                                  'id': g.rgi_id,
                                  'name': g.name})

        # save intermediate results
        g.write_pickle(mb_ds, 'mb_daily')

        # Multi-year daily data
        #mb_yday = []
        #for i in np.arange(365):
        #    mb_yday.append(mb_avg[i::365])
        # Begin day of the glaciological year and the days needed to be rolled
        bgday_glacio = 274
        bgday_tspan = day_model.tspan_in[0].timetuple().tm_yday
        rolld = bgday_tspan - bgday_glacio

        # GroupBy object (by glacio years)
        gyear_groups = mb_ds.roll(time=rolld).groupby(mb_ds.time.dt.year)
        # CumSums starting on bgday_glacio

        ###### BUG?????
        #Now (after applying cumsum) the real 01-10 is 01-01!!!!!

        def cumsum(x):
            return x.cumsum(dim='time', skipna=True)
        mb_yday_cs = gyear_groups.apply((lambda x: x.cumsum(dim='time',
                                                           skipna=True)))

        # Quantiles by day of glaciological year (starting with "JAN-01"!)
        def custom_quantiles(x, qs=np.array([0.1, 0.25, 0.5, 0.75, 0.9])):
            return x.quantile(qs)

        quant = mb_yday_cs.groupby(mb_yday_cs.time.dt.dayofyear) \
            .apply(lambda x: x.quantile([0.1, 0.25, 0.5, 0.75, 0.9]))
        # no need roll them again
        quantr = quant.copy()

        # insert attributes again...they get lost when grouping!?
        quantr.attrs.update(mb_ds.attrs)

        # OGGM standard plots
        if PLOTS_DIR == '':
            exit()
        utils.mkdir(PLOTS_DIR)
        bname = os.path.join(PLOTS_DIR, g.rgi_id + '_')

        ################################################
        # test
        # MAKE MB UP TO NOW
        mb_now = []
        yesterday = (datetime.datetime.now() - datetime.timedelta(2))
        yesterday_str = yesterday.strftime('%Y-%m-%d')

        begin = utils.get_begin_last_flexyear(yesterday)
        begin_str = begin.strftime('%Y-%m-%d')

        for date in pd.date_range(start=begin_str, end=yesterday_str,
                                  freq='D'):
            date = date.to_pydatetime()
            # Get the mass balance and convert to m per day
            tmp = day_model.get_daily_specific_mb(heights, widths,
                                                  date=date)
            mb_now.append(tmp)
        mb_now_cs = np.cumsum([i.mean() for i in mb_now])
        g.write_pickle(mb_now_cs, 'mb_current')

        ##################################################

        graphics.plot_cumsum_climatology_and_current(quantr, current=mb_now_cs, leap=True)
        plt.savefig(bname + 'test_new.png')
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

'''
# Inversion
execute_entity_task(tasks.prepare_for_inversion, gdirs)
tasks.optimize_inversion_params(gdirs)
execute_entity_task(tasks.volume_inversion, gdirs)


'''
