"""
Let's try schedule as the lightweight version of python-crontab....
"""

import schedule
import time
import xarray as xr
import numpy as np
import geopandas as gpd
import datetime
import logging
from joblib import Memory
import crampon.cfg as cfg
from crampon.core.preprocessing.climate import climate_file_from_scratch
from crampon import tasks
from crampon import utils
from crampon.utils import GlacierDirectory, retry
from crampon import tasks
from crampon.workflow import execute_entity_task, init_glacier_regions

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)


def startup_tasks(rgidf):

    gdirs = init_glacier_regions(rgidf, reset=True, force=True)

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

    ]
    for task in task_list:
            execute_entity_task(task, gdirs)


def hourly_tasks(gdirs):
    raise NotImplementedError()


# retry for almost one day every half an hour, if fails
@retry(Exception, tries=45, delay=1800, backoff=1, log_to=log)
def daily_tasks(gdirs):
    """
    A collection of tasks to perform every day.

    Parameters
    ----------
    gdirs: list
        List of crampon.GlacierDirectories to perform the tasks on.

    Returns
    -------
    None
    """

    log.info('Starting daily tasks...')

    # This must be FIRST
    cfile = climate_file_from_scratch()

    # if no news, try later again
    cmeta = xr.open_dataset(cfile, drop_variables=['temp', 'prcp', 'hgt'])
    yesterday_np64 = np.datetime64(datetime.datetime.today().date() -
                                   datetime.timedelta(1))

    # if no new meteo data on server, retry later
    if not cmeta.time.values[-1] == yesterday_np64:
        # otherwise PermissionError in the next try:
        cmeta.close()
        # file from yesterday not yet on WSL server -> retry
        log.info('No new meteo files from yesterday ({}) on WSL server...'
                 .format(yesterday_np64))
        raise FileNotFoundError

    # before we recalculate climate, delete it from cache
    utils.joblib_read_climate_crampon.clear()
    daily_entity_tasks = [
        tasks.process_custom_climate_data
    ]

    for task in daily_entity_tasks:
        execute_entity_task(task, gdirs)


def weekly_tasks(gdirs):
    """
    A collection of tasks to perform every week.

    Parameters
    ----------
    gdirs: list
        List of crampon.GlacierDirectories to perform the tasks on.

    Returns
    -------
    None
    """
    raise NotImplementedError()


def monthly_tasks(gdirs):
    """
    A collection of tasks to perform every month.

    Parameters
    ----------
    gdirs: list
        List of crampon.GlacierDirectories to perform the tasks on.

    Returns
    -------
    None
    """

    # update climate with verified data
    # recalculate the climatology
    # recalculate the current MB
    raise NotImplementedError()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        description='Run all CRAMPON cronjobs and set up/recover the complete '
                    'model if necessary.')
    parser.add_argument('-p', '--params', metavar='params.cfg',
                        help='Path to the cfg file with all needed settings to'
                             ' run CRAMPON.')
    parser.add_argument('-s', '--shapes', metavar='shapes.shp',
                        help='Path to the file with all shapes to be '
                             'calculated. At the moment, this file needs to '
                             'contain RGI-like columns.')
    args = parser.parse_args()

    log.info('Starting initial setup tasks...')
    # Tasks that should be run at every start of the operational workflow
    # 1) initialize
    inifile = args.params
    inifile = 'C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\CH_params.cfg'
    if not inifile:
        dec = input("Sure you want to run with the standard params?[y/n]")
        if 'y' in dec:
            cfg.initialize(file=inifile)
        else:
            raise ValueError('You want to provide a params.cfg file.')
    else:
        cfg.initialize(file=inifile)

    # Now the collections of operational tasks
    args.shapes = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
    rgidf = gpd.read_file(args.shapes)
    # HERE SHOULD GO A CORRECTION FOR THE FAILING POLYGONS (MOVE THEM TO CFG?)
    gdirs = init_glacier_regions(rgidf, reset=False, force=False)

    print('Making initial climate file from scratch')
    # 2) make/update climate file
    _ = climate_file_from_scratch(write_to=cfg.PATHS['climate_dir'],
                                   hfile=cfg.PATHS['hfile'])
    print('Finished making initial climate file from scratch')
    # 3) make MB climatology
    # 4) make MB since beginning of the mass balance year
    # 5) make future MB

    schedule.every().day.at("12:20").do(daily_tasks(gdirs)).tag('daily-tasks')

    while True:
        print('Finished setup tasks, switching to operational...')
        schedule.run_pending()
        time.sleep(1)