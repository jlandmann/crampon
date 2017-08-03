"""
Let's try schedule as the lightweight version of python-crontab....
"""

import schedule
import time
import xarray as xr
import datetime
import logging
import crampon.cfg as cfg
from crampon.core.preprocessing.climate import climate_file_from_scratch
from crampon import tasks
from crampon.utils import GlacierDirectory
from crampon import workflow
from crampon.workflow import execute_entity_task

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)


def startup_tasks():
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

def hourly_tasks(gdirs):
    raise NotImplementedError()


def daily_tasks(gdirs):

    # This must be FIRST
    cfile = climate_file_from_scratch()

    # if no news, try later again
    cmeta = xr.open_dataset(cfile, drop_variables=['temp', 'prcp', 'hgt'])
    if not cmeta.time.values[-1] == datetime.datetime.today():
        time.sleep(1800)
        daily_tasks(gdirs)


    daily_entity_tasks = [
        tasks.process_custom_climate_data
    ]

    for task in daily_entity_tasks:
        execute_entity_task(task, gdirs)


def weekly_tasks(gdirs):
    raise NotImplementedError()


def monthly_tasks(gdirs):
    raise NotImplementedError()



if __name__ == '__main__':

    # Tasks that should be run at every start of the operational workflow
    # 1) initialize
    inifile = None
    if not inifile:
        dec = input("Sure you want to run with the standard params?[y/n]")
        if 'y' in dec:
            cfg.initialize(file=inifile)
        else:
            raise ValueError('You want to provide a params.cfg file.')
    else:
        cfg.initialize(file=inifile)

    # 2) make/update climate file
    _ = climate_file_from_scratch(write_to=cfg.PATHS['climate_dir'],
                                   hfile=cfg.PATHS['hfile'])

    # 3) make MB climatology
    # 4) make MB since beginning of the mass balance year
    # 5) make future MB

    # Now the collections of operational tasks
    gdirs = GlacierDirectory()

    schedule.every().day.at("12:30").do(daily_tasks)

    while True:
        schedule.run_pending()
        time.sleep(1)