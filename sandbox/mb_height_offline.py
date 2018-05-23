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
PLOTS_DIR = os.path.join(cfg.PATHS['working_dir'], 'plots')
utils.mkdir(cfg.PATHS['working_dir'])

# Read one test glacier
glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
rgidf = gpd.read_file(glaciers)

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

rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])]  # Gries OK
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A10G05'])]  # Silvretta OK
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B5616n-1'])]  # Findel OK
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A55F03'])]  # Plaine Morte OK
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4312n-1'])]  # Rhone
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.C1410'])]  # Basòdino

if __name__ == '__main__':

    # Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf, reset=False, force=False)

    # Preprocessing tasks
    #task_list = [
        #tasks.glacier_masks,
        #tasks.compute_centerlines,
        #tasks.initialize_flowlines,
        #tasks.compute_downstream_line,
        #tasks.catchment_area,
        #tasks.catchment_intersections,
        #tasks.catchment_width_geom,
        #tasks.catchment_width_correction,
        #tasks.process_custom_climate_data,

    #]
    #for task in task_list:
    #        execute_entity_task(task, gdirs)

    for g in gdirs:

        day_model = BraithwaiteModel(g, bias=0.)
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
            tmp = day_model.get_daily_mb(heights, date=date)
            mb.append(tmp)

        mb_ds = xr.Dataset({'MB': (['n', 'height', 'time'],
                                   np.atleast_3d(mb).T)},
                           coords={'n': (['n'], exp),
                                   'time': pd.to_datetime(
                                       day_model.time_elapsed),
                                   'height': (['height'], heights)},
                           attrs={'id': g.rgi_id,
                                  'name': g.name})
        print(dt.datetime.now())
        # save intermediate results
        g.write_pickle(mb_ds, 'mb_daily')