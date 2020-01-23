from __future__ import absolute_import, division
import salem
import os
import logging
import crampon.cfg as cfg
from crampon import utils
from crampon.core.preprocessing import gis, centerlines
from crampon.core.models import flowline
import crampon
from shutil import rmtree
from oggm.workflow import _init_pool_globals, _merge_dicts,\
    _pickle_copier, execute_entity_task, init_glacier_regions,\
    merge_glacier_tasks

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
        A geopandas.GeoDataFrame with geometries to use for setting up the
        GlacierDirectories.
    reset: bool, optional
        Whether or not the existing GlacierDirectories and log shall be
        deleted. Default: False.
    force: bool, optional
        Whether or not to ask before deleting GlacierDirectories and log.
        Default: False.

    Returns
    -------
    gdirs: list
        A list of the GlacierDirectory objects.
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


def merge_glacier_tasks(gdirs, main_rgi_id=None, return_all=False, buffer=None,
                        **kwargs):
    """
    Shortcut function: run all tasks to merge tributaries to a main glacier.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory`
        all glaciers, main and tributary. Preprocessed and initialised
    main_rgi_id: str
        RGI ID of the main glacier of interest. If None is provided merging
        will start based upon the largest glacier
    return_all : bool
        if main_rgi_id is given and return_all = False: only the main glaicer
        is returned
        if main_rgi_is given and return_all = True, the main glacier and every
        remaining glacier from the initial gdirs list is returned, possible
        merged as well.
    buffer : float
        buffer around a flowline to first better find an overlap with another
        flowline. And second assure some distance between the lines at a
        junction. Will default to `cfg.PARAMS['kbuffer']`.
    kwargs: keyword argument for the recursive merging

    Returns
    -------
    merged_gdirs: list of all merged :py:class:`oggm.GlacierDirectory`
    """

    if len(gdirs) > 100:
        raise RuntimeError('this could take time! I should include an optinal '
                           'parameter to ignore this.')

    # sort all glaciers descending by area
    gdirs.sort(key=lambda x: x.rgi_area_m2, reverse=True)

    # if main glacier is asked, put it in first position
    if main_rgi_id is not None:
        gdir_main = [gd for gd in gdirs if gd.rgi_id == main_rgi_id][0]
        gdirs.remove(gdir_main)
        gdirs = [gdir_main] + gdirs

    merged_gdirs = []
    while len(gdirs) > 1:
        # main glacier is always the first: either given or the largest one
        gdir_main = gdirs.pop(0)
        gdir_merged, gdirs = _recursive_merging(gdirs, gdir_main, **kwargs)
        merged_gdirs.append(gdir_merged)

    # now we have gdirs which contain all the necessary flowlines,
    # time to clean them up
    for gdir in merged_gdirs:
        flowline.clean_merged_flowlines(gdir, buffer=buffer)

    if main_rgi_id is not None and return_all is False:
        return [gd for gd in merged_gdirs if main_rgi_id in gd.rgi_id][0]

    # add the remaining glacier to the final list
    merged_gdirs = merged_gdirs + gdirs

    return merged_gdirs


def _recursive_merging(gdirs, gdir_main, glcdf=None,
                       filename='climate_daily', input_filesuffix=''):
    """ Recursive function to merge all tributary glaciers.
    This function should start with the largest glacier and then be called
    upon all smaller glaciers.
    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory`
        all glaciers, main and tributary. Preprocessed and initialised
    gdir_main: :py:class:`oggm.GlacierDirectory`
        the current main glacier where the others are merge to
    glcdf: geopandas.GeoDataFrame
        which contains the main glaciers, will be downloaded if None
    filename: str
        Baseline climate file
    input_filesuffix: str
        Filesuffix to the climate file
    Returns
    -------
    merged_gdir: :py:class:`oggm.GlacierDirectory`
        the mergeed current main glacier
    gdirs : list of :py:class:`oggm.GlacierDirectory`
        updated list of glaciers, removed the already merged ones
    """
    # find glaciers which intersect with the main
    tributaries = centerlines.intersect_downstream_lines(gdir_main,
                                                         candidates=gdirs)
    if len(tributaries) == 0:
        # if no tributaries: nothing to do
        return gdir_main, gdirs

    # seperate those glaciers which are not already found to be a tributary
    gdirs = [gd for gd in gdirs if gd not in tributaries]

    gdirs_to_merge = []

    for trib in tributaries:
        # for each tributary: check if we can merge additional glaciers to it
        merged, gdirs = _recursive_merging(gdirs, trib, glcdf=glcdf,
                                           filename=filename,
                                           input_filesuffix=input_filesuffix)
        gdirs_to_merge.append(merged)

    # create merged glacier directory
    gdir_merged = utils.initialize_merged_gdir(
        gdir_main, tribs=gdirs_to_merge, glcdf=glcdf, filename=filename,
        input_filesuffix=input_filesuffix)
    flowline.merge_to_one_glacier(gdir_merged, gdirs_to_merge,
                                  filename=filename,
                                  input_filesuffix=input_filesuffix)

    return gdir_merged, gdirs
