"""Run with a subset of benchmark glaciers"""
from __future__ import division

# Log message format
import logging

# Python imports
import os
# Libs
import geopandas as gpd
import matplotlib.pyplot as plt
# Locals
import crampon.cfg as cfg
from crampon import workflow
from crampon import tasks
from crampon.workflow import execute_entity_task
from crampon import graphics, utils
from oggm.core.models.massbalance import PastMassBalanceModel
import numpy as np

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)


# Initialize CRAMPON (and OGGM, hidden in cfg.py)
cfg.initialize(file='C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\'
                    'testglaciers_params.cfg')

# Local paths (where to write output and where to download input)
print(cfg.PATHS)
PLOTS_DIR = os.path.join(cfg.PATHS['working_dir'], 'plots')

# Currently OGGM wants some directories to exist
# (maybe I'll change this but it can also catch errors in the user config)
utils.mkdir(cfg.PATHS['working_dir'])
utils.mkdir(cfg.PATHS['topo_dir'])
utils.mkdir(cfg.PATHS['cru_dir'])
utils.mkdir(cfg.PATHS['rgi_dir'])

# Read one test glacier
testglacier = 'C:\\users\\johannes\\documents\\crampon\\data\\test\\shp\\' \
              'testglaciers_rgi.shp'
rgidf = gpd.read_file(testglacier)

log.info('Number of glaciers: {}'.format(len(rgidf)))

# necessary to make multiprocessing work on Windows
if __name__ == '__main__':

    # Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)


    # Prepro tasks
    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.compute_downstream_lines,
        tasks.catchment_area,
        tasks.initialize_flowlines,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.process_custom_climate_data,
        tasks.mu_candidates

    ]
    for task in task_list:
        execute_entity_task(task, gdirs)

    # Climate related task
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)

    for g in gdirs:
        # Necessary for inversion input
        tasks.prepare_for_inversion(g, add_debug_var=True)
        past_model = PastMassBalanceModel(g)
        majid = g.read_pickle('major_divide', div_id=0)
        cl = g.read_pickle('inversion_input', div_id=majid)[-1]

        mb = []
        for i, yr in enumerate(np.arange(1962, 2017)):
            for m in np.arange(12):
                yrm = utils.date_to_year(yr, m + 1)
                # Get the mass balance and convert to m per year
                tmp = past_model.get_monthly_mb(cl['hgt'], yrm) * \
                      cfg.SEC_IN_MONTHS[m] * cfg.RHO / 1000.
                mb.append(tmp)

        # mb_tstar = tstar_model.get_annual_mb(z) * cfg.SEC_IN_YEAR *
        # cfg.RHO / 1000.
        # mb_2003 = hist_model.get_annual_mb(z, 2003) * cfg.SEC_IN_YEAR *
        #  cfg.RHO / 1000.

        # Average MB over all glacier heights
        mb_avg = [np.nanmean(i) for i in mb]

        # Plot
        if PLOTS_DIR == '':
            exit()
        utils.mkdir(PLOTS_DIR)
        bname = os.path.join(PLOTS_DIR, g.rgi_id + '_')
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

        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(1962, 2017, 1/12.), mb_avg)
        plt.xlabel('Years')
        plt.ylabel('Monthly MB (m we)')
        plt.grid(which='both')
        plt.title(g.rgi_id + ': ' + g.name)
        plt.savefig(bname + 'past_MB.png')
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