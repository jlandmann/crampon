from __future__ import absolute_import, division
import salem
import os
import logging
import crampon.cfg as cfg
from crampon import utils
from crampon.core.preprocessing import gis
import crampon
from shutil import rmtree
from oggm.workflow import _init_pool_globals, init_mp_pool, _merge_dicts,\
    _pickle_copier, execute_entity_task, init_glacier_regions

# MPI similar to OGGM - not yet implemented
try:
    import oggm.mpi as ogmpi
    _have_ogmpi = True
except ImportError:
    _have_ogmpi = False

# Module logger
log = logging.getLogger(__name__)


def init_glacier_regions_crampon(shapedf=None, reset=False, force=False):
    """
    Set up or take over GlacierDirectories. The first task (always!).

    The function is copied from OGGM, just some names have been changed.
    Sooner or later maybe also 'dem.tif' and 'dem' should be replaced by the
    multitemporal equivalents.

    Set reset=True in order to delete the content of the directories.

    Parameters
    ----------
    shapedf: :obj:`geopandas.GeoDataFrame`, optional
        A GeoDataFrame with geometries from whcih to set up the
        GlacierDirectories.
    reset: bool, optional
        Whether or not the exitsing GlacierDirectories and log shall be
        deleted. Default: False
    force: bool, optional
        Whether or not to ask before deleting GlacierDirectories and log.
        Default: False

    Returns
    -------
    gdirs: list
        A list of the GlacierDirectories.
    """

    if reset and not force:
        reset = utils.query_yes_no('Delete all glacier directories?')

    # if reset delete also the log directory
    if reset:
        fpath = os.path.join(cfg.PATHS['working_dir'], 'log')
        if os.path.exists(fpath):
            rmtree(fpath)

    gdirs = []
    new_gdirs = []
    if shapedf is None:
        if reset:
            raise ValueError('Cannot use reset without a rgi file')
        # The dirs should be there already
        gl_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier')
        for root, _, files in os.walk(gl_dir):
            if files and ('dem.tif' in files):
                gdirs.append(crampon.GlacierDirectory(os.path.basename(root)))
    else:
        for _, entity in shapedf.iterrows():
            gdir = crampon.GlacierDirectory(entity, reset=reset)
            if not os.path.exists(gdir.get_filepath('dem')):
                new_gdirs.append((gdir, dict(entity=entity)))
            gdirs.append(gdir)

    # If not initialized, run the task in parallel
    execute_entity_task(gis.define_glacier_region, new_gdirs)

    return gdirs

init_glacier_regions = init_glacier_regions_crampon
