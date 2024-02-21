from __future__ import absolute_import, division
from typing import Optional, List

import numpy as np
import pandas as pd
import geopandas as gpd
import salem
import os
import logging
from crampon import cfg
from crampon import utils
from crampon.core.preprocessing import gis, centerlines
from crampon.core.models import flowline
from crampon.graphics import popup_html_string
from crampon.core.models.massbalance import MassBalance, DailyMassBalanceModel
import crampon
from shutil import rmtree
from oggm.workflow import _merge_dicts, _pickle_copier, init_glacier_regions, \
    merge_glacier_tasks
import multiprocessing
from collections.abc import Sequence
from scipy.stats import percentileofscore

# MPI
try:
    import oggm.mpi as ogmpi

    _have_ogmpi = True
except ImportError:
    _have_ogmpi = False

# Module logger
log = logging.getLogger(__name__)

# Multiprocessing Pool
_mp_manager = None
_mp_pool = None


def _init_pool_globals(_cfg_contents, global_lock):
    cfg.unpack_config(_cfg_contents)
    utils.lock = global_lock


def init_mp_pool(reset=False):
    """Necessary because at import time, cfg might be uninitialized"""
    global _mp_manager, _mp_pool
    if _mp_pool and _mp_manager and not reset:
        return _mp_pool

    cfg.CONFIG_MODIFIED = False
    if _mp_pool:
        _mp_pool.terminate()
        _mp_pool = None
    if _mp_manager:
        cfg.set_manager(None)
        _mp_manager.shutdown()
        _mp_manager = None

    if cfg.PARAMS['use_mp_spawn']:
        mp = multiprocessing.get_context('spawn')
    else:
        mp = multiprocessing

    _mp_manager = mp.Manager()

    cfg.set_manager(_mp_manager)
    cfg_contents = cfg.pack_config()

    global_lock = _mp_manager.Lock()

    mpp = cfg.PARAMS['mp_processes']
    _mp_pool = mp.Pool(mpp, initializer=_init_pool_globals,
                       initargs=(cfg_contents, global_lock))
    return _mp_pool


def _merge_dicts(*dicts):
    r = {}
    for d in dicts:
        r.update(d)
    return r


class _pickle_copier(object):
    """Pickleable alternative to functools.partial,
    Which is not pickleable in python2 and thus doesn't work
    with Multiprocessing."""

    def __init__(self, func, kwargs):
        self.call_func = func
        self.out_kwargs = kwargs

    def __call__(self, arg):
        if self.call_func:
            gdir = arg
            call_func = self.call_func
        else:
            call_func, gdir = arg
        if isinstance(gdir, Sequence) and not isinstance(gdir, str):
            gdir, gdir_kwargs = gdir
            gdir_kwargs = _merge_dicts(self.out_kwargs, gdir_kwargs)
            return call_func(gdir, **gdir_kwargs)
        else:
            return call_func(gdir, **self.out_kwargs)


def reset_multiprocessing():
    """Reset multiprocessing state
    Call this if you changed configuration parameters mid-run and need them to
    be re-propagated to child processes.
    """
    global _mp_pool
    if _mp_pool:
        _mp_pool.terminate()
        _mp_pool = None
    cfg.CONFIG_MODIFIED = False


def execute_entity_task(task, gdirs, **kwargs):
    """Execute a task on gdirs.
    If you asked for multiprocessing, it will do it.
    If ``task`` has more arguments than `gdir` they have to be keyword
    arguments.
    Parameters
    ----------
    task : function
         the entity task to apply
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    """

    # Should be iterable
    gdirs = utils.tolist(gdirs)

    if len(gdirs) == 0:
        return

    log.workflow('Execute entity task %s on %d glaciers', task.__name__,
                 len(gdirs))

    if task.__dict__.get('global_task', False):
        return task(gdirs, **kwargs)

    pc = _pickle_copier(task, kwargs)

    if _have_ogmpi:
        if ogmpi.OGGM_MPI_COMM is not None:
            return ogmpi.mpi_master_spin_tasks(pc, gdirs)

    if cfg.PARAMS['use_multiprocessing']:
        mppool = init_mp_pool(cfg.CONFIG_MODIFIED)
        out = mppool.map(pc, gdirs, chunksize=1)
    else:
        out = [pc(gdir) for gdir in gdirs]

    return out


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


def _recursive_merging(gdirs, gdir_main, glcdf=None, filename='climate_daily',
                       input_filesuffix=''):
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
    gdir_merged = utils.initialize_merged_gdir(gdir_main, tribs=gdirs_to_merge,
        glcdf=glcdf, filename=filename, input_filesuffix=input_filesuffix)
    print(gdir_merged)
    flowline.merge_to_one_glacier(gdir_merged, gdirs_to_merge,
                                  filename=filename,
                                  input_filesuffix=input_filesuffix)

    return gdir_merged, gdirs


def fetch_glacier_status(gdirs: Optional[List[utils.GlacierDirectory]] = None,
        mb_model: Optional[DailyMassBalanceModel] = None,
        shp: Optional[str] = None, prefer_mb_suffix: Optional[str] = '',
        allow_unpreferred_mb_suffices: Optional[bool] = True,
        exclude_old_status: Optional[pd.Timedelta] = pd.Timedelta(days=7),
        output_all_attrs: Optional[bool] = True) -> None:
    """
    Fetch the glacier status from the current and climatological mass balance.

    todo: compare to individual climate reference periods im clim

    Parameters
    ----------
    gdirs : gdirs: list of `py:class:utils.GlacierDirectory` or None, optional
        If given, only for these GlacierDirectories a status is fetched.
        Default: None (take all from `shp`).
    mb_model :
    shp: str or None, optional
        Path to the glacier shapefile. If None, search in the data directory.
        Default: None.
    prefer_mb_suffix: str, optional
        Which mass balance suffix shall be preferred for the status, e.g.
        '_fischer_unique_variability'. Default: '' (take the default order
        from 'good' to 'bad'.)
    allow_unpreferred_mb_suffices: bool, optional
        Whether to allow other suffices than the preferred one at all.
        Default: True (for final operational runs).
    exclude_old_status: pd.Timedelta, None
        If a glacier has not been updated recently, it might still have an
        outdated `mb_current´ file. If set, all glaciers with a status older
        than today-exclude_old_status are omitted. Default:
        pd.Timedelta(days=7), i.e. all status older than a week from today are
        omitted.
    output_all_attrs : bool, optional
        Whether to output all attributed from `shp`, or just the one fetched
        here. Default: True (output everything).

    Returns
    -------
    None
    """

    # try "from good to bad":
    suffix_priority_list = ['', '_fischer', '_fischer_unique_variability',
                            '_fischer_unique']
    if (len(prefer_mb_suffix) > 0) and (allow_unpreferred_mb_suffices is True):
        suffix_priority_list = [prefer_mb_suffix] + suffix_priority_list
    elif (len(prefer_mb_suffix) == 0) and \
            (allow_unpreferred_mb_suffices is True):
        pass
    else:
        suffix_priority_list = ['']

    last_accepted_status = pd.Timestamp.now() - exclude_old_status

    if shp is None:
        shp = os.path.join(cfg.PATHS['data_dir'], 'outlines',
                               'mauro_sgi_merge.shp')

    glc_gdf = gpd.GeoDataFrame.from_file(shp)
    glc_gdf = glc_gdf.sort_values(by='Area', ascending=True)

    if gdirs is not None:
        gdirs_ids = np.array([g.rgi_id for g in gdirs])
        glc_gdf = glc_gdf[glc_gdf.RGIId.isin(gdirs_ids)]
    else:
        gdirs = init_glacier_regions(glc_gdf, reset=False, force=False)

    if output_all_attrs is True:
        out_df = glc_gdf.copy(deep=True)
    else:
        out_df = pd.DataFrame(data={'RGIId': glc_gdf.RGIId.values})
    out_df['pctl'] = np.nan
    out_df['cali_source'] = None
    out_df['status_date'] = ''

    log.info('Fetching glacier status...')
    has_current_and_clim = []
    error = 0
    cnt = 0
    for gdir in gdirs:

        cnt += 1
        if cnt % 100 == 0:
            log.info('Checking status of {}th glacier...'.format(cnt))

        clim = None
        current = None
        src_suffix = None

        for mb_src_suffix in suffix_priority_list:
            try:
                clim = gdir.read_pickle('mb_daily' + mb_src_suffix)
                current = gdir.read_pickle('mb_current' + mb_src_suffix)
                src_suffix = mb_src_suffix
                break
            except (FileNotFoundError, AttributeError):
                continue
        if (clim is None) or (current is None):
            continue

        if exclude_old_status is not None:
            if current.time.values[-1] < last_accepted_status:
                log.info('Status for {} older than {}. Omitting...'.format(
                    gdir.rgi_id, last_accepted_status.strftime('%Y-%m-%d')))
                continue

        if mb_model is not None:
            current = current.sel(model=mb_model)

        # stack model and potential members
        clim = clim.stack(ens=['model', 'member'])
        # stack model and potential members
        current = current.stack(ens=['model', 'member'])
        current_csq = current.mb.make_cumsum_quantiles()
        # take median as best estimate of the ensemble
        mbc_values = current_csq.sel(quantile=0.5)
        mbc_value = mbc_values.MB.isel(hydro_doys=-1)

        hydro_years = clim.mb.make_hydro_years()
        mb_cumsum = clim.groupby(hydro_years).apply(
            lambda x: MassBalance.time_cumsum(x))

        # todo: as long as we are not able to calculate cumsum with NaNs
        mb_cumsum = mb_cumsum.where(mb_cumsum.MB != 0.)

        mbd_values = [
            j.MB.sel(time=mbc_values.hydro_doys[-1].item() - 1).median(
                dim='ens', skipna=True).item() for i, j in
            list(mb_cumsum.groupby(hydro_years))[:-1]]
        mbd_values = sorted(mbd_values)
        pctl = percentileofscore(mbd_values, mbc_value.item())
        out_df.loc[out_df.RGIId == gdir.rgi_id, 'pctl'] = pctl
        out_df.loc[out_df.RGIId == gdir.rgi_id, 'cali_source'] = src_suffix
        out_df.loc[out_df.RGIId == gdir.rgi_id, 'status_date'] = \
            pd.to_datetime(str(current.time.values[-1])) .strftime('%Y-%m-%d')
        has_current_and_clim.append(gdir.rgi_id)

    # select the valid subset
    out_df = out_df[out_df.RGIId.isin(has_current_and_clim)]

    log.info('Successfully fetched mass balance status for {} out of {} '
             'glaciers.'.format(len(out_df), len(gdirs)))

    if len(out_df) >= 1:
        # todo : delete existing glacier status, when out_df is empty?
        # attach HTML text for popups
        out_df['popup_html'] = out_df.apply(utils.popup_html_string, axis=1)

        # exclude some strange values
        out_df = out_df[out_df.avg_specif != '']
        out_df = out_df[~pd.isnull(out_df.avg_specif.values)]
        out_df = out_df[out_df.Aspect_mea.values != None]
        out_df['avg_specif'] = out_df.avg_specif.astype('float')
        out_df['pctl'] = out_df.pctl.astype('float')
        out_df['Area'] = out_df.Area.astype('float')
        out_df['year1'] = out_df.year1.astype('float')
        out_df['year2'] = out_df.year2.astype('float')

        # write status to working dir
        out_base = cfg.PATHS['working_dir']
        out_df.to_file(os.path.join(out_base, 'glacier_status.geojson'),
                       driver="GeoJSON")


