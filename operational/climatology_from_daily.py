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


# Initialize CRAMPON (and OGGM, hidden in cfg.py)
cfg.initialize(file='C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\'
                    'CH_params.cfg')

# Local paths (where to write output and where to download input)
PLOTS_DIR = os.path.join(cfg.PATHS['working_dir'], 'plots')
print(PLOTS_DIR)

# Currently OGGM wants some directories to exist
# (maybe I'll change this but it can also catch errors in the user config)
utils.mkdir(cfg.PATHS['working_dir'])

# Read one test glacier
#testglacier = 'C:\\Users\\Johannes\\Documents\\data\\outlines\\RGI\\' \
#              'subset_CH\\subset_CH.shp'
# Mauro's DB ad disguised in RGI
testglacier = 'C:\\Users\\Johannes\\Desktop\\mauro_in_RGI_disguise_entities_old_and_new.shp'
rgidf = gpd.read_file(testglacier)
#rgidf = rgidf[rgidf.RGIId == 'RGI50-11.00536']

califile = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'data/results_prelim/'
                        'mustar_from_mauro_first_attempt_brute.csv')
cali = pd.read_csv(califile)

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
                        'RGI50-11.02848']  # ValueError: no minimum-cost path was found to the specified end point (compute_centerlines)
rgidf = rgidf[~rgidf.RGIId.isin(problem_glaciers_sgi)]
rgidf = rgidf.head(50)
#rgidf = rgidf[rgidf.Area >= 0.01]
#rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.00638'])] # just to have one REFMB glacier

# Run parameters
cfg.PARAMS['d1'] = 4
cfg.PARAMS['dmax'] = 100
cfg.PARAMS['border'] = 120
cfg.PARAMS['invert_with_sliding'] = False
cfg.PARAMS['min_slope'] = 2
cfg.PARAMS['max_shape_param'] = 0.006
cfg.PARAMS['max_thick_to_width_ratio'] = 0.5
cfg.PARAMS['temp_use_local_gradient'] = True
cfg.PARAMS['optimize_thick'] = True
cfg.PARAMS['force_one_flowline'] = ['RGI50-11.01270']

log.info('Number of glaciers: {}'.format(len(rgidf)))

# necessary to make multiprocessing work on Windows
if __name__ == '__main__':

    # get the needed climatology
    for var, mode in product(['TabsD', 'R'], ['verified', 'operational']):
        cirrus = utils.CirrusClient()
        a, b = cirrus.sync_files('/data/griddata',
                                 'c:\\users\\johannes\\documents\\crampon\\'
                                 'data\\bigdata', globpattern='*{}/daily/{}*/'
                                                              'netcdf/*'
                                 .format(mode, var))

        flist = glob.glob(
            'c:\\users\\johannes\\documents\\crampon\\data\\bigdata\\griddata\\{}\\daily\\{}*\\netcdf\\*.nc'.format(mode, var))

        # We do this instead of using open_mfdataset, as we need a lot of
        # preprocessing
        log.info('Concatenating {} {} {} files...'.format(len(flist), var, mode))
        sda = utils.read_multiple_netcdfs(flist, chunks={'time': 50},
                                          tfunc=utils._cut_with_CH_glac)
        log.info('Ensuring time continuity...')
        sda.crampon.ensure_time_continuity()
        sda.encoding['zlib'] = True
        print(sda.dims, 'end')
        sda.to_netcdf('c:\\users\\johannes\\desktop\\{}_{}_all.nc'
                      .format(var, mode))

    # update operational with verified
    for var in ['TabsD', 'R']:
        v_op = 'c:\\users\\johannes\\desktop\\{}_operational_all.nc'.format(var)
        v_ve = 'c:\\users\\johannes\\desktop\\{}_verified_all.nc'.format(var)

        data = utils.read_netcdf(v_op, chunks={'time': 50})
        data = data.crampon.update_with_verified(v_ve)

        data.to_netcdf('c:\\users\\johannes\\desktop\\{}_op_ver.nc'.format(var))

    # combine both
    tfiles = glob.glob('c:\\users\\johannes\\desktop\\*.nc'.format('TabsD'))
    pfiles = glob.glob('c:\\users\\johannes\\documents\\crampon\\data\\bigdata\\griddata\\verified\\daily\\{}*\\netcdf\\*.nc'.format('R'))
    hfile = glob.glob('c:\\users\\johannes\\documents\\crampon\\data\\test\\hgt.nc')
    outfile = 'c:\\users\\johannes\\documents\\crampon\\data\\bigdata\\climate_all.nc'

    cfg.PATHS['climate_file'] = outfile

    utils.daily_climate_from_netcdf(tfiles, pfiles, hfile, outfile)

    # Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf, reset=True, force=True)

    # Preprocessing tasks
    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.compute_downstream_lines,
        tasks.catchment_area,
        tasks.initialize_flowlines,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.process_custom_climate_data,
        #tasks.mu_candidates

    ]
    for task in task_list:
            execute_entity_task(task, gdirs)

    # Climate related task
    tasks.compute_ref_t_stars(gdirs)
    tasks.distribute_t_stars(gdirs)

    for g in gdirs:
        # Necessary for inversion input
        tasks.prepare_for_inversion(g, add_debug_var=True)

        mu_star = cali[cali.rgi_id == g.rgi_id].mu_star.values[0]
        prcp_fac = cali[cali.rgi_id == g.rgi_id].prcp_fac.values[0]

        try:
            past_model = DailyMassBalanceModel(g, mu_star=mu_star,
                                               prcp_fac=prcp_fac)
        except OSError:
            log.error('local mustar for {} does not exist'.format(g.rgi_id))
            continue
        majid = g.read_pickle('major_divide', div_id=0)
        cl = g.read_pickle('inversion_input', div_id=majid)[-1]

        mb = []
        for date in past_model.tspan_in:

            # for the first: Skip the 29th of Feb
            if date.month == 2 and date.day == 29:
                continue

            # Get the mass balance and convert to m per year
            tmp = past_model.get_daily_mb(cl['hgt'], date=date) * \
                  cfg.SEC_IN_DAY * cfg.RHO / 1000.
            mb.append(tmp)

        # mb_tstar = tstar_model.get_annual_mb(z) * cfg.SEC_IN_YEAR *
        # cfg.RHO / 1000.
        # mb_2003 = hist_model.get_annual_mb(z, 2003) * cfg.SEC_IN_YEAR *
        #  cfg.RHO / 1000.

        # Average MB over all glacier heights
        mb_avg = [np.nanmean(i) for i in mb]

        # Multi-year daily data
        mb_yday = []
        for i in np.arange(365):
            mb_yday.append(mb_avg[i::365])

        # No need to roll the data: ALREADY IN GLACIO YEARS!!
        mb_ymon_cs = np.nancumsum(mb_yday, axis=0)

        # OGGM standard plots
        if PLOTS_DIR == '':
            exit()
        utils.mkdir(PLOTS_DIR)
        bname = os.path.join(PLOTS_DIR, g.rgi_id + '_')

        """
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
        """

        # MB time series (line plot)
        plt.figure(figsize=(8, 5))
        plt.plot(past_model.tspan_in, mb_avg)
        plt.xlabel('Years')
        plt.ylabel('Daily MB (m we)')
        plt.grid(which='both')
        plt.title(g.rgi_id + ': ' + g.name)
        plt.savefig(bname + 'past_MB.png')
        plt.close()

        # Climatology
        fig, ax = plt.subplots(figsize=(10, 5))
        #  plot median
        ax.plot(np.arange(1, 366), [np.nanmedian(m) for m in mb_yday], c='b',
                label='Median')
        #  plot IQR
        ax.fill_between(np.arange(1, 366),
                        [np.nanpercentile(m, 25) for m in mb_yday],
                        [np.nanpercentile(m, 75) for m in mb_yday],
                        facecolor='cornflowerblue', alpha=0.5)
        #  plot 10th to 90th pctl
        ax.fill_between(np.arange(1, 366),
                        [np.nanpercentile(m, 10) for m in mb_yday],
                        [np.nanpercentile(m, 90) for m in mb_yday],
                        facecolor='cornflowerblue', alpha=0.3)
        ax.set_xlabel('Months')
        ax.set_ylabel('Mass Balance (m we)')
        ax.set_xlim(1, 365)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.roll(cfg.DAYS_IN_MONTH, 3))
        # start with November, bcz we set ticks at end:
        ax.set_xticklabels([m for m in 'NDJFMAMJJASO'])
        ax.grid(True, which='both', alpha=0.5)
        plt.title('Multi-year Daily MB Distribution of ' +
                  g.rgi_id + '(' + g.name + ')')
        legend = plt.legend()
        plt.setp(legend.get_title())
        fig.savefig(bname + 'Yday_MB.png')
        plt.close()

        # Climatology CumSum
        fig, ax = plt.subplots(figsize=(10, 5))
        #  plot median
        ax.plot(np.arange(1, 366), [np.nanmedian(m) for m in mb_ymon_cs],
                c='b', label='Median')
        #  plot IQR
        ax.fill_between(np.arange(1, 366),
                        [np.nanpercentile(m, 25) for m in mb_ymon_cs],
                        [np.nanpercentile(m, 75) for m in mb_ymon_cs],
                        facecolor='cornflowerblue', alpha=0.5)
        #  plot 10th to 90th pctl
        ax.fill_between(np.arange(1, 366),
                        [np.nanpercentile(m, 10) for m in mb_ymon_cs],
                        [np.nanpercentile(m, 90) for m in mb_ymon_cs],
                        facecolor='cornflowerblue', alpha=0.3)
        ax.set_xlabel('Days')
        ax.set_ylabel('Cumulative Mass Balance (m we)')
        ax.set_xlim(1, 12)
        start, end = ax.get_xlim()
        ax.xaxis.set_ticks(np.roll(cfg.DAYS_IN_MONTH, 3))
        ax.set_xticklabels([m for m in 'NDJFMAMJJASO'])
        ax.grid(True, which='both', alpha=0.5)
        plt.title('Multi-year Daily Cumulative MB Distribution of ' +
                  g.rgi_id + '(' + g.name + ')')
        legend = plt.legend()
        plt.setp(legend.get_title())
        fig.savefig(bname + 'Yday_Cumul_MB.png')
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