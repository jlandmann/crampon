"""  Configuration file and options

A number of globals are defined here to be available everywhere.

COMMENT: the CRAMPON-specific BASENAMES  and PATHS get the prefix "C" in order
to make sure they come from CRAMPON. Later on, they are merged with the
respective OGGM dictionaries.
"""
from __future__ import absolute_import, division

from oggm.cfg import PathOrderedDict, DocumentedDict, set_intersects_db, \
    pack_config, unpack_config, oggm_static_paths, get_lru_handler
from oggm.cfg import initialize as oggminitialize


import logging
import os
import shutil
import sys
import glob
from collections import OrderedDict
from distutils.util import strtobool

import numpy as np
import pandas as pd
import geopandas as gpd
from configobj import ConfigObj, ConfigObjError

# Defaults
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

# Local logger
log = logging.getLogger(__name__)

# Path to the cache directory
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.oggm')
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
# Path to the config file
CONFIG_FILE = os.path.join(os.path.expanduser('~'), '.oggm_config')

# Globals
IS_INITIALIZED = False
CONTINUE_ON_ERROR = False
CPARAMS = OrderedDict()
PARAMS = OrderedDict()
NAMES = OrderedDict()
CPATHS = PathOrderedDict()
PATHS = PathOrderedDict()
CBASENAMES = DocumentedDict()
BASENAMES = DocumentedDict()
RGI_REG_NAMES = False
RGI_SUBREG_NAMES = False
LRUHANDLERS = OrderedDict()

# Constants
SEC_IN_YEAR = 365*24*3600
SEC_IN_DAY = 24*3600
SEC_IN_HOUR = 3600
DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAYS_IN_MONTH_LEAP = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
SEC_IN_MONTHS = [d * SEC_IN_DAY for d in DAYS_IN_MONTH]
CUMSEC_IN_MONTHS = np.cumsum(SEC_IN_MONTHS)
BEGINSEC_IN_MONTHS = np.insert(CUMSEC_IN_MONTHS[:-1], [0], 0)

RHO = 900.  # ice density
RHO_W = 1000.  # water density
LATENT_HEAT_FUSION_WATER = 334000  #(J kg-1)
HEAT_CAP_ICE = 2050  # (J kg-1 K-1)
R = 8.314  # gas constant (J K-1 mol-1)
E_FIRN = 21400.  # activation energy for firn (see Reeh et al., 2008) (J mol-1)
ZERO_DEG_KELVIN = 273.15

G = 9.81  # gravity
N = 3.  # Glen's law's exponent
A = 2.4e-24  # Glen's default creep's parameter
FS = 5.7e-20  # Default sliding parameter from Oerlemans - OUTDATED
TWO_THIRDS = 2./3.
FOUR_THIRDS = 4./3.
ONE_FIFTH = 1./5.

# the mass balance models ready to use
MASSBALANCE_MODELS = ['BraithwaiteModel', 'PellicciottiModel',
                      'OerlemansModel']

# Added from CRAMPON:
_doc = 'CSV output from the calibration for the different models, including' \
       'uncertainties.'
CBASENAMES['calibration'] = ('calibration.csv', _doc)
_doc = 'The daily climate timeseries for this glacier, stored in a netCDF ' \
       'file.'
CBASENAMES['climate_daily'] = ('climate_daily.nc', _doc)
_doc = 'The daily mass balance timeseries for this glacier, stored in a ' \
       'pickle file.'
CBASENAMES['mb_daily'] = ('mb_daily.pkl', _doc)
_doc = 'The daily mass balance timeseries for this glacier in the current ' \
       'budget year, stored in a pickle file.'
CBASENAMES['mb_current'] = ('mb_current.pkl', _doc)
_doc = 'A time series of all available DEMs for the glacier. Contains groups' \
       ' for different resolutions.'
CBASENAMES['homo_dem_ts'] = ('homo_dem_ts.nc', _doc)
_doc = 'A time series of all available DEMs for the glacier, brought to the ' \
       'minimum common resolution.'
CBASENAMES['dem_ts'] = ('dem_ts.nc', _doc)
_doc = 'Uncorrected geodetic mass balance calculations from the DEMs in ' \
       'homo_dem_ts.nc. Contains groups for different resolutions.'
CBASENAMES['gmb_uncorr'] = ('gmb_uncorr.nc', _doc)
_doc = 'Corrected geodetic mass balance calculations from the DEMs in ' \
       'dem_ts.nc. Corrected geodetic mass balances account for mass ' \
       'conservative firn and snow densification processes. Contains groups ' \
       'for different resolutions.'
CBASENAMES['gmb_corr'] = ('gmb_corr.nc', _doc)
_doc = 'The multitemporal glacier outlines in the local projection.'
CBASENAMES['outlines_ts'] = ('outlines_ts.shp', _doc)
_doc = 'A CSV with measured mass balances from the glaciological method.'
CBASENAMES['glacio_method_mb'] = ('glacio_method_mb', _doc)
CBASENAMES['mb_daily_rescaled'] =('mb_daily_rescaled.pkl', 'abc')
_doc = 'A CSV with geodetic volume changes.'
CBASENAMES['geodetic_dv'] = ('geodetic_dv.csv', _doc)
_doc = 'Daily mean potential clear-sky solar irradiation on the glacier grid.'
CBASENAMES['ipot'] = ('ipot.nc', _doc)
_doc = 'Daily mean potential clear-sky solar irradiation on the flowline ' \
       'heights.'
CBASENAMES['ipot_per_flowline'] = ('ipot_per_flowline.pkl', _doc)

# some more standard names, for less hardcoding
NAMES['DHM25'] = 'dhm25'
NAMES['SWISSALTI2010'] = 'alti'
NAMES['LFI'] = 'lfi'


def initialize(file=None):
    """Read the configuration file containing the run's parameters."""

    global IS_INITIALIZED
    global BASENAMES
    global PARAMS
    global PATHS
    global NAMES
    global CONTINUE_ON_ERROR
    global GRAVITY_CONST
    global E_FIRN
    global ZERO_DEG_KELVIN
    global R
    global LATENT_HEAT_FUSION_WATER
    global HEAT_CAP_ICE
    global N
    global A
    global RHO
    global RHO_W
    global RGI_REG_NAMES
    global RGI_SUBREG_NAMES

    # This is necessary as OGGM still refers to its own initialisation
    #oggminitialize(file=file)
    oggminitialize()
    import oggm.cfg as oggmcfg

    # Add the CRAMPON-specific keys to the dicts
    oggmcfg.BASENAMES.update(CBASENAMES)

    if file is None:
        file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'params.cfg')

    log.info('Parameter file: %s', file)

    try:
        cp = ConfigObj(file, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Config file could not be parsed (%s): %s', file, e)
        sys.exit()

    # Default
    oggmcfg.PATHS['working_dir'] = cp['working_dir']
    if not oggmcfg.PATHS['working_dir']:
        oggmcfg.PATHS['working_dir'] = os.path.join(os.path.expanduser('~'),
                                                    'OGGM_WORKING_DIRECTORY')

    # Paths
    oggm_static_paths()

    oggmcfg.PATHS['dem_file'] = cp['dem_file']
    oggmcfg.PATHS['data_dir'] = cp['data_dir']
    oggmcfg.PATHS['hfile'] = cp['hfile']
    oggmcfg.PATHS['climate_file'] = cp['climate_file']
    oggmcfg.PATHS['climate_dir'] = cp['climate_dir']
    oggmcfg.PATHS['lfi_worksheet'] = cp['lfi_worksheet']
    oggmcfg.PATHS['firncore_dir'] = cp['firncore_dir']
    oggmcfg.PATHS['lfi_dir'] = cp['lfi_dir']
    oggmcfg.PATHS['dem_dir'] = cp['dem_dir']
    oggmcfg.PATHS['wgms_rgi_links'] = cp['wgms_rgi_links']
    oggmcfg.PATHS['glathida_rgi_links'] = cp['glathida_rgi_links']
    oggmcfg.PATHS['leclercq_rgi_links'] = cp['leclercq_rgi_links']
    oggmcfg.PATHS['mb_dir'] = cp['mb_dir']
    oggmcfg.PATHS['modelrun_backup_dir_1'] = cp['modelrun_backup_dir_1']
    oggmcfg.PATHS['modelrun_backup_dir_2'] = cp['modelrun_backup_dir_2']

    # run params
    oggmcfg.PARAMS['run_period'] = [int(vk) for vk in cp.as_list('run_period')]

    # Multiprocessing pool
    oggmcfg.PARAMS['use_multiprocessing'] = cp.as_bool('use_multiprocessing')
    oggmcfg.PARAMS['mp_processes'] = cp.as_int('mp_processes')

    # Some non-trivial params
    oggmcfg.PARAMS['grid_dx_method'] = cp['grid_dx_method']
    oggmcfg.PARAMS['topo_interp'] = cp['topo_interp']
    oggmcfg.PARAMS['use_divides'] = cp.as_bool('use_divides')
    oggmcfg.PARAMS['use_intersects'] = cp.as_bool('use_intersects')
    oggmcfg.PARAMS['use_compression'] = cp.as_bool('use_compression')
    oggmcfg.PARAMS['use_tar_shapefiles'] = cp.as_bool('use_tar_shapefiles')
    oggmcfg.PARAMS['mpi_recv_buf_size'] = cp.as_int('mpi_recv_buf_size')
    oggmcfg.PARAMS['use_multiple_flowlines'] = cp.as_bool('use_multiple_flowlines')
    oggmcfg.PARAMS['optimize_thick'] = cp.as_bool('optimize_thick')
    oggmcfg.PARAMS['filter_min_slope'] = cp.as_bool('filter_min_slope')
    oggmcfg.PARAMS['auto_skip_task'] = cp.as_bool('auto_skip_task')
    oggmcfg.PARAMS['run_mb_calibration'] = cp.as_bool('run_mb_calibration')

    # Mass balance
    oggmcfg.PARAMS['ratio_mu_snow_ice'] = cp['ratio_mu_snow_ice']

    # Climate
    oggmcfg.PARAMS['temp_use_local_gradient'] = cp.as_int(
        'temp_use_local_gradient')
    k = 'temp_local_gradient_bounds'
    oggmcfg.PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    oggmcfg.PARAMS['prcp_use_local_gradient'] = cp.as_int(
        'prcp_use_local_gradient')
    k = 'prcp_local_gradient_bounds'
    oggmcfg.PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    oggmcfg.PARAMS['precip_ratio_method'] = cp['precip_ratio_method']
    k = 'tstar_search_window'
    oggmcfg.PARAMS[k] = [int(vk) for vk in cp.as_list(k)]
    oggmcfg.PARAMS['use_bias_for_run'] = cp.as_bool('use_bias_for_run')
    _factor = cp['prcp_scaling_factor']
    if _factor not in ['stddev', 'stddev_perglacier']:
        _factor = cp.as_float('prcp_scaling_factor')
    oggmcfg.PARAMS['prcp_scaling_factor'] = _factor
    oggmcfg.PARAMS['tminmax_available'] = cp.as_int('tminmax_available')
    oggmcfg.PARAMS['albedo_method'] = cp['albedo_method']

    # Inversion
    oggmcfg.PARAMS['invert_with_sliding'] = cp.as_bool('invert_with_sliding')
    _k = 'optimize_inversion_params'
    oggmcfg.PARAMS[_k] = cp.as_bool(_k)

    # Flowline model
    _k = 'use_optimized_inversion_params'
    oggmcfg.PARAMS[_k] = cp.as_bool(_k)

    # Make sure we have a proper cache dir
    from oggm.utils import download_oggm_files
    download_oggm_files()

    CPARAMS['bgday_hydro'] = cp.as_int('bgday_hydro')
    CPARAMS['bgmon_hydro'] = cp.as_int('bgmon_hydro')

    # Delete non-floats
    ltr = ['working_dir', 'dem_file', 'climate_file', 'climate_dir',
           'wgms_rgi_links', 'glathida_rgi_links', 'firncore_dir', 'lfi_dir',
           'lfi_worksheet', 'dem_dir', 'hfile', 'grid_dx_method', 'data_dir',
           'mp_processes', 'use_multiprocessing', 'use_divides',
           'temp_use_local_gradient', 'prcp_use_local_gradient',
           'temp_local_gradient_bounds', 'mb_dir', 'modelrun_backup_dir_1',
           'modelrun_backup_dir_2', 'prcp_local_gradient_bounds',
           'precip_ratio_method', 'topo_interp', 'use_compression',
           'use_tar_shapefiles', 'bed_shape', 'continue_on_error',
           'use_optimized_inversion_params', 'invert_with_sliding',
           'optimize_inversion_params', 'use_multiple_flowlines',
           'leclercq_rgi_links', 'optimize_thick', 'mpi_recv_buf_size',
           'tstar_search_window', 'use_bias_for_run', 'run_period',
           'prcp_scaling_factor', 'tminmax_available', 'use_intersects',
           'filter_min_slope', 'auto_skip_task', 'correct_for_neg_flux',
           'problem_glaciers', 'bgmon_hydro', 'bgday_hydro',
           'run_mb_calibration', 'albedo_method']
    for k in ltr:
        cp.pop(k, None)

    # Other params are floats
    for k in cp:
        oggmcfg.PARAMS[k] = cp.as_float(k)

    # Empty defaults
    from oggm.utils import get_demo_file
    set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))
    IS_INITIALIZED = True

    # Update the dicts in case there are changes
    oggmcfg.PATHS.update(CPATHS)
    oggmcfg.PARAMS.update(CPARAMS)

    BASENAMES = oggmcfg.BASENAMES
    PATHS = oggmcfg.PATHS
    PARAMS = oggmcfg.PARAMS

    # Always call this one! Creates tmp_dir etc.
    oggm_static_paths()
