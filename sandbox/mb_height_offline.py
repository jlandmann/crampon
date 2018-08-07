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

#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])]  # Gries OK
rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A10G05'])]  # Silvretta OK
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
        begin_clim = dt.datetime(1961, 9, 1)
        end_clim = dt.datetime(2017, 12, 31)

        for date in pd.date_range(begin_clim, end_clim):

            # Get the mass balance and convert to m w.e. per day
            tmp = day_model.get_daily_mb(heights, date=date) * 3600 * 24 * cfg.RHO /\
              1000.
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