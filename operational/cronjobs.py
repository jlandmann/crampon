"""
Let's try schedule as the lightweight version of python-crontab....
"""
from typing import List

import os
import schedule
import time
import xarray as xr
import numpy as np
import geopandas as gpd
import datetime
import logging
import crampon.cfg as cfg
from crampon.core.preprocessing.climate import make_climate_file, make_nwp_file
from crampon import utils
from crampon.utils import retry
from crampon import tasks
from crampon.workflow import execute_entity_task, init_glacier_regions
from operational import mb_production

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)


def startup_tasks(rgidf: gpd.GeoDataFrame) -> None:
    """
    Tasks that should be run at the overall startup of CRAMPON.

    Parameters
    ----------
    rgidf : gpd.GeoDataFrame
        GeoDataFrame with the input geometries and columns so that
        `utils.GlacierDirectory` has everything it needs.

    Returns
    -------
    None
    """

    gdirs = init_glacier_regions(rgidf, reset=True, force=True)

    # Preprocessing tasks
    task_list = [
        tasks.glacier_masks,
        tasks.compute_centerlines,
        tasks.initialize_flowlines,
        tasks.compute_downstream_line,
        tasks.catchment_area,
        tasks.catchment_intersections,
        tasks.catchment_width_geom,
        tasks.catchment_width_correction,
        tasks.compute_downstream_bedshape,
        tasks.process_custom_climate_data,
        tasks.simple_glacier_masks,
        tasks.calculate_and_distribute_ipot

    ]
    for task in task_list:
        execute_entity_task(task, gdirs)


def hourly_tasks(gdirs: List[utils.GlacierDirectory]) -> None:
    """
    Tasks that should be run every hour during operational runs.

    Parameters
    ----------
    gdirs : list
        List of py:class:`crampon.GlacierDirectory` for which the hourly tasks
        should be run.

    Returns
    -------
    None
    """
    raise NotImplementedError()


def daily_tasks(gdirs: List[utils.GlacierDirectory]) -> None:
    """
    Tasks that should be run every day during operational runs.

    Parameters
    ----------
    gdirs : list
        List of py:class:`crampon.GlacierDirectory` for which the daily tasks
        should be run.

    Returns
    -------
    None
    """

    plot_dir = cfg.PATHS['plots_dir']

    log.info('Starting daily tasks...')

    log.info('Downloading weather analyses data...')
    try_download_new_cirrus_files()

    # before we recalculate climate, delete it from cache
    utils.joblib_read_climate_crampon.clear()
    daily_entity_tasks = [
        (tasks.update_climate, {}, 'Updating glacier climate files...'),
        (tasks.make_mb_current_mbyear,
         {'first_day': utils.get_cirrus_yesterday()},
         'Making current mass balance...'),
        (tasks.plot_cumsum_climatology_and_current, {'plot_dir': plot_dir},
         'Plotting the climatology and current...'),
        (tasks.plot_interactive_mb_spaghetti_html, {'plot_dir': plot_dir},
         'Plotting the interactive spaghetti...')
    ]

    for task, kwargs, msg in daily_entity_tasks:
        log.info(msg)
        execute_entity_task(task, gdirs, **kwargs)

    log.info('Making the clickable popup map...')
    tasks.make_mb_popup_map(plot_dir=plot_dir)

    log.info('Copying plots to webpage...')
    # search only sub-folders
    utils.copy_to_webpage_dir(plot_dir, glob_pattern=os.path.join('**', '**'))

    log.info('Trying to make a backup...')
    try_backup(gdirs)


# retry for almost one day every half an hour, if fails
@retry(Exception, tries=45, delay=1800, backoff=1, log_to=log)
def try_download_new_cirrus_files() -> None:
    """
    Try and download the latest files from MeteoSwiss on the WSL Cirrus server.

    Returns
    -------
    None
    """
    # This must be FIRST
    cfile = make_climate_file(how='update')

    # if no news, try later again
    cmeta = xr.open_dataset(cfile, drop_variables=['temp', 'tmin', 'tmax',
                                                   'prcp', 'sis', 'hgt'])
    yesterday_np64 = np.datetime64(datetime.datetime.today().date() -
                                   datetime.timedelta(1))

    # if no new meteo data on server, retry later
    if not cmeta.time.values[-1] == yesterday_np64:
        # otherwise PermissionError in the next try:
        cmeta.close()

        now = datetime.datetime.now()
        if (now.hour >= 12) and (now.minute >= 21):
            # file from yesterday not yet on WSL server -> retry
            log.info('No new meteo files from yesterday ({}) on WSL server...'
                     .format(yesterday_np64))
            raise FileNotFoundError
        else:
            # it's normal that they are not there yet in the morning
            pass


def try_backup(gdirs: List[utils.GlacierDirectory]) -> None:
    """
    Try to make a backup of the selected GlacierDirectories.

    Parameters
    ----------
    gdirs :

    Returns
    -------

    """

    log.info('Trying to backup data....')

    backup_dir_1 = cfg.PATHS['modelrun_backup_dir_1']
    backup_dir_2 = cfg.PATHS['modelrun_backup_dir_2']

    try:
        execute_entity_task(tasks.copy_to_basedir, gdirs,
                            base_dir=backup_dir_1, setup='all')
        log.info('Backup of model run directory completed to'
                 .format(backup_dir_1))
    except (WindowsError, RuntimeError):
        if len(backup_dir_2) > 0:
            execute_entity_task(tasks.copy_to_basedir, gdirs,
                                base_dir=backup_dir_2, setup='all')
            log.info('Backup of model run directory completed to'
                     .format(backup_dir_2))
        else:
            log.warning('Backup of model run directory failed with {} and no '
                        'alternative supplied'.format(backup_dir_1))
    except (WindowsError, RuntimeError):
        log.warning(
            'Backup of model run directory failed using {} and {}.'.format(
                backup_dir_1, backup_dir_2))


def daily_cosmo_tasks(gdirs: List[utils.GlacierDirectory]) -> None:
    """
    Daily tasks related to the COSMO prediction files.

    Returns
    -------

    """
    log.info('Starting COSMO tasks...')

    log.info('Making NWP file from COSMO data...')
    make_nwp_file()

    log.info('Making mass balance prediction from COSMO predictions...')
    execute_entity_task(mb_production.make_mb_prediction, gdirs)


def weekly_tasks(gdirs: List[utils.GlacierDirectory]):
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


def monthly_tasks(gdirs: List[utils.GlacierDirectory]) -> None:
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

    if not inifile:
        dec = input("Sure you want to run with the standard params?[y/n]")
        if 'y' in dec:
            cfg.initialize(file=inifile)
        else:
            raise ValueError('You want to provide a params.cfg file.')
    else:
        cfg.initialize(file=inifile)

    # for later
    if not os.path.exists(cfg.PATHS['plots_dir']):
        utils.mkdir(cfg.PATHS['plots_dir'])

    # Now the collections of operational tasks
    args.shapes = os.path.join(cfg.PATHS['data_dir'], 'outlines',
                               'mauro_sgi_merge.shp')
    rgidf = gpd.read_file(args.shapes)

    # select only GLAMOS IDs for the first
    rgidf = rgidf[rgidf.RGIId.isin(cfg.PARAMS['glamos_ids'])]

    gdirs = init_glacier_regions(rgidf, reset=False, force=False)
    gdirs = [gdirs[17]]

    log.info('Making initial climate file from scratch...')
    # 2) make/update climate file
    _ = make_climate_file(write_to=cfg.PATHS['climate_dir'],
                          hfile=cfg.PATHS['hfile'])

    # 2a) update individual glacier climate files
    log.info('Finished making initial climate file from scratch...')
    execute_entity_task(tasks.update_climate, gdirs)

    # 3) make MB climatology
    log.info('Making mass balance climatology...')
    execute_entity_task(mb_production.make_mb_clim, gdirs)

    # 4) make MB since beginning of the mass balance year
    log.info('Making mass balance of the current mass budget year...')
    execute_entity_task(mb_production.make_mb_current_mbyear, gdirs,
                        reset_file=True)

    # 5) make future MB
    log.info('Making mass balance prediction...')
    execute_entity_task(mb_production.make_mb_prediction, gdirs)

    # Necessary in order not to have spikes anymore
    daily_tasks(gdirs)
    # daily_cosmo_tasks(gdirs)

    # schedule.every().day.at("04:00").do(daily_cosmo_tasks, gdirs)
    # .tag('daily-cosmo-tasks')
    schedule.every().day.at("12:21").do(daily_tasks, gdirs).tag('daily-tasks')

    print('Finished setup tasks, switching to operational...')
    while True:
        schedule.run_pending()
        time.sleep(1)
