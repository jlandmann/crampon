
import salem
import logging
from oggm.workflow import _init_pool_globals, _init_pool, _merge_dicts,\
    _pickle_copier, execute_entity_task, init_glacier_regions


# MPI similar to OGGM - not yet implemented
try:
    import oggm.mpi as ogmpi
    _have_ogmpi = True
except ImportError:
    _have_ogmpi = False

# Module logger
log = logging.getLogger(__name__)


# I have to (re)write
# gis_prepro_tasks, climate_tasks,
# inversion_tasks, DAILY_TASKS
# (check if new data are available and start workflow), CALIBRATION TASKS
