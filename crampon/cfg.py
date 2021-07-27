"""  Configuration file and options

A number of globals are defined here to be available everywhere.

COMMENT: the CRAMPON-specific BASENAMES  and PATHS get the prefix "C" in order
to make sure they come from CRAMPON. Later on, they are merged with the
respective OGGM dictionaries.
"""

from oggm.cfg import PathOrderedDict, DocumentedDict, set_intersects_db, \
    pack_config, unpack_config, oggm_static_paths, get_lru_handler, \
    set_logging_config, ResettingOrderedDict, ParamsLoggingDict, set_manager, \
    initialize_minimal
from oggm.cfg import initialize as oggminitialize
import oggm.cfg as oggmcfg


import logging
import os
import shutil
import sys
import glob
import json
from collections import OrderedDict
from distutils.util import strtobool
import warnings

import numpy as np
import pandas as pd
try:
    from scipy.signal.windows import gaussian
except AttributeError:
    # Old scipy
    from scipy.signal import gaussian
from configobj import ConfigObj, ConfigObjError
try:
    import geopandas as gpd
except ImportError:
    pass
try:
    import salem
except ImportError:
    pass

from oggm.exceptions import InvalidParamsError

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

# config was changed, indicates that multiprocessing needs a reset
CONFIG_MODIFIED = False

# Share state accross processes
DL_VERIFIED = dict()
DEM_SOURCE_TABLE = dict()

# Machine epsilon
FLOAT_EPS = np.finfo(float).eps

# Globals
IS_INITIALIZED = False
CPARAMS = ParamsLoggingDict()
PARAMS = ParamsLoggingDict()
NAMES = OrderedDict()
CPATHS = PathOrderedDict()
PATHS = PathOrderedDict()
CBASENAMES = DocumentedDict()
BASENAMES = DocumentedDict()
LRUHANDLERS = ResettingOrderedDict()
DATA = ResettingOrderedDict

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
SOLAR_CONSTANT = 1367.  # W m-2
SEALEVEL_PRESSURE = 101325  # Pa
MOLAR_MASS_DRY_AIR = 0.02896968  # kg/mol
FLUX_TO_DAILY_FACTOR = (86400 * RHO) / RHO_W  # m ice s-1 to m w.e. d-1

G = oggmcfg.G  # gravity
RE = 6378000.  # average earth radius
N = 3.  # Glen's law's exponent
A = 2.4e-24  # Glen's default creep's parameter
FS = 5.7e-20  # Default sliding parameter from Oerlemans - OUTDATED
TWO_THIRDS = 2./3.
FOUR_THIRDS = 4./3.
ONE_FIFTH = 1./5.

# the mass balance models ready to use - order might matter!
MASSBALANCE_MODELS = ['BraithwaiteModel', 'HockModel',
                      'PellicciottiModel', 'OerlemansModel']

GAUSSIAN_KERNEL = dict()
for ks in [5, 7, 9]:
    kernel = gaussian(ks, 1)
    GAUSSIAN_KERNEL[ks] = kernel / kernel.sum()

# Added from CRAMPON:
_doc = 'CSV output from the calibration for the different models, including' \
       'uncertainties.'
CBASENAMES['calibration'] = ('calibration.csv', _doc)
_doc = 'CSV output from the calibration on the geodetic mass changes from ' \
       'Fischer et al, 2015.'
CBASENAMES['calibration_fischer_unique'] = \
    ('calibration_fischer_unique.csv', _doc)
_doc = 'CSV output from the calibration on the geodetic mass changes from ' \
       'Fischer et al, 2015.'
CBASENAMES['calibration_fischer_unique_variability'] = \
    ('calibration_fischer_unique_variability.csv', _doc)
_doc = 'CSV output from the calibration on the geodetic mass changes from ' \
       'Fischer et al, 2015.'
CBASENAMES['calibration_fischer'] = ('calibration_fischer.csv', _doc)
_doc = 'The daily climate time series for this glacier, stored in a netCDF ' \
       'file.'
CBASENAMES['climate_daily'] = ('climate_daily.nc', _doc)
_doc = 'The daily ECMWF NWP prediction for this glacier, stored in a netCDF ' \
       'file.'
CBASENAMES['nwp_daily_ecmwf'] = ('nwp_daily_ecmwf.nc', _doc)
_doc = 'The daily COSMO NWP prediction for this glacier, stored in a netCDF ' \
       'file.'
CBASENAMES['nwp_daily_cosmo'] = ('nwp_daily_cosmo.nc', _doc)
_doc = 'The spinup climate time series for this glacier, stored in a netCDF ' \
       'file.'
CBASENAMES['climate_spinup'] = ('climate_spinup.nc', _doc)
_doc = 'The spinup mass balance for this glacier, stored in a ' \
       'netCDF file.'
CBASENAMES['mb_spinup'] = ('mb_spinup.pkl', _doc)
_doc = 'The daily mass balance timeseries for this glacier, stored in a ' \
       'pickle file.'
CBASENAMES['mb_daily'] = ('mb_daily.pkl', _doc)
_doc = 'The daily mass balance timeseries for this glacier from calibration ' \
       'on the Fischer et al., 2015 geodetic mass changes, stored in a ' \
       'pickle file.'
CBASENAMES['mb_daily_fischer'] = ('mb_daily_fischer.pkl', _doc)
_doc = 'The daily mass balance timeseries for this glacier from calibration ' \
       'on the Fischer et al., 2015 geodetic mass changes with one unique ' \
       'parameter set for the whole time period, stored in a pickle file.'
CBASENAMES['mb_daily_fischer_unique'] = ('mb_daily_fischer_unique.pkl', _doc)
_doc = 'The daily mass balance timeseries for this glacier from calibration ' \
       'on the Fischer et al., 2015 geodetic mass changes with one unique ' \
       'parameter set for the whole time period, but imposed parameter ' \
       'variability, stored in a pickle file.'
CBASENAMES['mb_daily_fischer_unique_variability'] = \
    ('mb_daily_fischer_unique_variability.pkl', _doc)
_doc = 'The snow condition at the end of the spinup phase, stored in a ' \
       'pickle file.'
CBASENAMES['snow_spinup'] = ('snow_spinup.pkl', _doc)
_doc = 'The current snow conditions for this glacier on the latest analysis ' \
       'day (yesterday), stored in a pickle file.'
CBASENAMES['snow_current'] = ('snow_current.pkl', _doc)
_doc = 'The current snow conditions for this glacier on the latest analysis ' \
       'day (yesterday) from calibration ' \
       'on the Fischer et al., 2015 geodetic mass changes with one unique ' \
       'parameter set for the whole time period, stored in a pickle file.'
CBASENAMES['snow_current_fischer_unique'] = \
    ('snow_current_fischer_unique.pkl', _doc)
_doc = 'The current snow conditions for this glacier on the latest analysis ' \
       'day (yesterday) from calibration ' \
       'on the Fischer et al., 2015 geodetic mass changes with one unique ' \
       'parameter set for the whole time period, but imposed parameter ' \
       'variability, stored in a pickle file.'
CBASENAMES['snow_current_fischer_unique_variability'] = \
    ('mb_current_fischer_unique_variability.pkl', _doc)
_doc = 'The daily snow condition time series for this glacier in the past, ' \
       'stored in a pickle file.'
CBASENAMES['snow_daily'] = ('snow_daily.pkl', _doc)
_doc = 'The daily snow condition time series for this glacier in the past, ' \
       'stored in a pickle file.'
CBASENAMES['snow_daily_fischer_unique'] = \
    ('snow_daily_fischer_unique.pkl', _doc)
_doc = 'The daily snow condition time series for this glacier in the past, ' \
       'stored in a pickle file.'
CBASENAMES['snow_daily_fischer_unique_variability'] = \
    ('snow_daily_fischer_unique_variability.pkl', _doc)
_doc = 'The daily snow condition time series for this glacier in the past, ' \
       'stored in a pickle file.'
CBASENAMES['snow_daily_fischer'] = ('snow_daily_fischer.pkl', _doc)
_doc = 'The snow redistribution factor over time.'
CBASENAMES['snow_redist'] = ('snow_redist.nc', _doc)
_doc = 'The assimilated mass balance of the current budget year for this ' \
       'glacier, stored in a pickle file.'
CBASENAMES['mb_assim'] = ('mb_assim.pkl', _doc)
_doc = 'The mass balance of the current budget year for this glacier, from ' \
       'the Fischer calibration stored in a pickle file.'
CBASENAMES['mb_current_fischer_unique'] = \
    ('mb_current_fischer_unique.pkl', _doc)
_doc = 'The mass balance of the current budget year for this glacier, from ' \
       'the Fischer calibration stored in a pickle file.'
CBASENAMES['mb_current_fischer_unique_variability'] = \
    ('mb_current_fischer_unique_variability.pkl', _doc)
_doc = 'The mass balance of the current budget year for this glacier, ' \
       'stored in a pickle file.'
CBASENAMES['mb_current'] = ('mb_current.pkl', _doc)
_doc = 'The current mass balance time series on all heights for this glacier' \
       ' stored in a pickle file.'
CBASENAMES['mb_current_heights'] = ('mb_current_heights.pkl', _doc)
_doc = 'The mass balance prediction for this glacier with COSMO predictions,' \
       ' stored in a pickle file.'
CBASENAMES['mb_prediction_cosmo'] = ('mb_prediction_cosmo.pkl', _doc)
_doc = 'The mass balance prediction for this glacier with ECMWF extended ' \
       'range forecasts, stored in a pickle file.'
CBASENAMES['mb_prediction_ecmwf'] = ('mb_prediction_ecmwf.pkl', _doc)
_doc = 'A time series of all available DEMs for the glacier, brought to the ' \
       'minimum common resolution.'
CBASENAMES['homo_dem_ts'] = ('homo_dem_ts.nc', _doc)
_doc = 'A time series of all available DEMs for the glacier. Contains groups' \
       ' for different resolutions.'
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
_doc = 'All assimilation data of the glacier, assembled into a netCDF file.'
CBASENAMES['assim_data'] = ('assim_data.nc', _doc)
_doc = 'Shortwave incoming solar radiation scaling factor (calculated from ' \
       'Ipot).'
CBASENAMES['sis_scale_factor'] = ('sis_scale_factor.nc', _doc)
_doc = 'Satellite images and derived variables like binary snow maps or snow' \
       ' line altitude.'
CBASENAMES['sat_images'] = ('sat_images.nc', _doc)

for bn in oggmcfg.BASENAMES:
    BASENAMES[bn] = (bn, oggmcfg.BASENAMES.doc_str(bn))
for cbn in CBASENAMES:
    BASENAMES[cbn] = (cbn, CBASENAMES.doc_str(cbn))
    oggmcfg.BASENAMES[cbn] = (cbn, CBASENAMES.doc_str(cbn))

# some more standard names, for less hardcoding
NAMES['DHM25'] = 'dhm25'
NAMES['SWISSALTI2010'] = 'alti'
NAMES['LFI'] = 'lfi'


def initialize(file=None, logging_level='INFO', params=None):
    """
    Read the configuration file containing the run's parameters.

    This should be the first call, before using any of the other CRAMPON
    modules for most (all?) CRAMPON simulations. Strong imitation of OGGM
    function.

    Parameters
    ----------
    file : str
        Path to the configuration file (default: CRAMPON params.cfg)
    logging_level : str
        Set a logging level. See :func:`oggm.cfg.set_logging_config` for
        options.
    params : dict
        Overrides for specific parameters from the config file
    """

    global IS_INITIALIZED
    global BASENAMES
    global CPARAMS
    global PARAMS
    global PATHS
    global NAMES
    global DATA
    global E_FIRN
    global ZERO_DEG_KELVIN
    global R
    global RE
    global LATENT_HEAT_FUSION_WATER
    global HEAT_CAP_ICE
    global N
    global A
    global RHO
    global RHO_W
    global RGI_REG_NAMES
    global RGI_SUBREG_NAMES

    # Do not spam
    PARAMS.do_log = False
    oggmcfg.PARAMS.do_log = False

    # This is necessary as OGGM still refers to its own initialisation
    oggminitialize()

    set_logging_config(logging_level=logging_level)

    is_default = False
    if file is None:
        file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                            'params.cfg')
        is_default = True
    try:
        cp = ConfigObj(file, file_error=True)
    except (ConfigObjError, IOError) as e:
        log.critical('Config file could not be parsed (%s): %s', file, e)
        sys.exit()

    if is_default:
        log.workflow('Reading default parameters from the CRAMPON `params.cfg` '
                     'configuration file.')
    else:
        log.workflow('Reading parameters from the user provided '
                     'configuration file: %s', file)

    # Static Paths
    oggm_static_paths()

    # Apply code-side manual params overrides
    if params:
        for k, v in params.items():
            cp[k] = v

    # Paths
    PATHS['working_dir'] = cp['working_dir']
    PATHS['dem_file'] = cp['dem_file']
    PATHS['climate_file'] = cp['climate_file']
    PATHS['nwp_file_cosmo'] = cp['nwp_file_cosmo']

    # Ephemeral paths overrides
    env_wd = os.environ.get('OGGM_WORKDIR')
    if env_wd and not PATHS['working_dir']:
        PATHS['working_dir'] = env_wd
        log.workflow(
            "PATHS['working_dir'] set to env variable $OGGM_WORKDIR: " + env_wd)

    # Multiprocessing pool
    try:
        use_mp = bool(int(os.environ['OGGM_USE_MULTIPROCESSING']))
        msg = 'ON' if use_mp else 'OFF'
        log.workflow('Multiprocessing switched {} '.format(
            msg) + 'according to the ENV variable OGGM_USE_MULTIPROCESSING')
    except KeyError:
        use_mp = cp.as_bool('use_multiprocessing')
        msg = 'ON' if use_mp else 'OFF'
        log.workflow('Multiprocessing switched {} '.format(
            msg) + 'according to the parameter file.')
    PARAMS['use_multiprocessing'] = use_mp

    # Spawn
    try:
        use_mp_spawn = bool(int(os.environ['OGGM_USE_MP_SPAWN']))
        msg = 'ON' if use_mp_spawn else 'OFF'
        log.workflow('MP spawn context switched {} '.format(
            msg) + 'according to the ENV variable OGGM_USE_MP_SPAWN')
    except KeyError:
        use_mp_spawn = cp.as_bool('use_mp_spawn')
    PARAMS['use_mp_spawn'] = use_mp_spawn

    # Number of processes
    mpp = cp.as_int('mp_processes')
    if mpp == -1:
        try:
            mpp = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            log.workflow('Multiprocessing: using slurm allocated '
                         'processors (N={})'.format(mpp))
        except (KeyError, ValueError):
            import multiprocessing
            mpp = multiprocessing.cpu_count()
            log.workflow('Multiprocessing: using all available '
                         'processors (N={})'.format(mpp))
    else:
        log.workflow('Multiprocessing: using the requested number of '
                     'processors (N={})'.format(mpp))
    PARAMS['mp_processes'] = mpp

    # Size of LRU cache
    try:
        lru_maxsize = int(os.environ['LRU_MAXSIZE'])
        log.workflow('Size of LRU cache set to {} '.format(
            lru_maxsize) + 'according to the ENV variable LRU_MAXSIZE')
    except KeyError:
        lru_maxsize = cp.as_int('lru_maxsize')
    PARAMS['lru_maxsize'] = lru_maxsize

    # Some non-trivial params
    PARAMS['continue_on_error'] = cp.as_bool('continue_on_error')
    PARAMS['grid_dx_method'] = cp['grid_dx_method']
    PARAMS['topo_interp'] = cp['topo_interp']
    PARAMS['use_intersects'] = cp.as_bool('use_intersects')
    PARAMS['use_compression'] = cp.as_bool('use_compression')
    PARAMS['border'] = cp.as_int('border')
    PARAMS['mpi_recv_buf_size'] = cp.as_int('mpi_recv_buf_size')
    PARAMS['use_multiple_flowlines'] = cp.as_bool('use_multiple_flowlines')
    PARAMS['filter_min_slope'] = cp.as_bool('filter_min_slope')
    PARAMS['auto_skip_task'] = cp.as_bool('auto_skip_task')
    PARAMS['correct_for_neg_flux'] = cp.as_bool('correct_for_neg_flux')
    PARAMS['filter_for_neg_flux'] = cp.as_bool('filter_for_neg_flux')
    PARAMS['run_mb_calibration'] = cp.as_bool('run_mb_calibration')
    PARAMS['rgi_version'] = cp['rgi_version']
    PARAMS['use_rgi_area'] = cp.as_bool('use_rgi_area')
    PARAMS['compress_climate_netcdf'] = cp.as_bool(
        'compress_climate_netcdf')
    PARAMS['use_tar_shapefiles'] = cp.as_bool('use_tar_shapefiles')
    PARAMS['clip_mu_star'] = cp.as_bool('clip_mu_star')
    PARAMS['clip_tidewater_border'] = cp.as_bool('clip_tidewater_border')
    PARAMS['dl_verify'] = cp.as_bool('dl_verify')
    PARAMS['calving_line_extension'] = cp.as_int('calving_line_extension')
    PARAMS['use_kcalving_for_inversion'] = cp.as_bool(
        'use_kcalving_for_inversion')
    PARAMS['use_kcalving_for_run'] = cp.as_bool('use_kcalving_for_run')
    PARAMS['calving_use_limiter'] = cp.as_bool('calving_use_limiter')
    PARAMS['use_inversion_params_for_run'] = cp.as_bool(
        'use_inversion_params_for_run')
    k = 'error_when_glacier_reaches_boundaries'
    PARAMS[k] = cp.as_bool(k)

    # Climate
    PARAMS['sis_delivery_delay'] = cp.as_bool('sis_delivery_delay')
    PARAMS['baseline_climate'] = cp['baseline_climate'].strip().upper()
    PARAMS['hydro_month_nh'] = cp.as_int('hydro_month_nh')
    PARAMS['hydro_month_sh'] = cp.as_int('hydro_month_sh')
    PARAMS['climate_qc_months'] = cp.as_int('climate_qc_months')
    PARAMS['temp_use_local_gradient'] = cp.as_int(
        'temp_use_local_gradient_cells')  # overwrite

    # Delete non-floats
    ltr = [
        'working_dir', 'dem_file', 'climate_file', 'climate_dir',
        'wgms_rgi_links', 'glathida_rgi_links', 'firncore_dir', 'lfi_dir',
        'lfi_worksheet', 'dem_dir', 'hfile', 'grid_dx_method', 'data_dir',
        'mp_processes', 'use_multiprocessing', 'use_divides',
        'sis_delivery_delay', 'temp_use_local_gradient',
        'temp_use_local_gradient_cells', 'prcp_use_local_gradient',
        'temp_local_gradient_bounds', 'mb_dir', 'modelrun_backup_dir_1',
        'modelrun_backup_dir_2', 'prcp_local_gradient_bounds',
        'precip_ratio_method', 'topo_interp', 'use_compression',
        'use_tar_shapefiles', 'bed_shape', 'continue_on_error',
        'use_optimized_inversion_params', 'invert_with_sliding',
        'optimize_inversion_params', 'use_multiple_flowlines',
        'leclercq_rgi_links', 'optimize_thick', 'nwp_file_ecmwf',
        'mpi_recv_buf_size', 'tstar_search_window', 'use_bias_for_run',
        'run_period', 'prcp_scaling_factor', 'tminmax_available',
        'use_intersects', 'filter_min_slope', 'auto_skip_task',
        'correct_for_neg_flux', 'problem_glaciers', 'bgmon_hydro',
        'bgday_hydro', 'run_mb_calibration', 'albedo_method', 'glamos_ids',
        'begin_mbyear_month', 'begin_mbyear_day', 'swe_bounds', 'alpha_bounds',
        'tacc_bounds', 'nwp_file_cosmo', 'use_mp_spawn', 'working_dir',
        'dem_file', 'climate_file', 'use_tar_shapefiles', 'grid_dx_method',
        'run_mb_calibration', 'compress_climate_netcdf', 'mp_processes',
        'use_multiprocessing', 'climate_qc_months', 'temp_use_local_gradient',
        'temp_local_gradient_bounds', 'topo_interp', 'use_compression',
        'bed_shape', 'continue_on_error', 'use_multiple_flowlines',
        'tstar_search_glacierwide', 'border', 'mpi_recv_buf_size',
        'hydro_month_nh', 'clip_mu_star', 'tstar_search_window',
        'use_bias_for_run', 'hydro_month_sh', 'use_intersects',
        'filter_min_slope', 'clip_tidewater_border', 'auto_skip_task',
        'correct_for_neg_flux', 'filter_for_neg_flux', 'rgi_version',
        'dl_verify', 'use_mp_spawn', 'calving_use_limiter',
        'use_shape_factor_for_inversion', 'use_rgi_area',
        'use_shape_factor_for_fluxbasedmodel', 'baseline_climate',
        'calving_line_extension', 'use_kcalving_for_run', 'lru_maxsize',
        'free_board_marine_terminating', 'use_kcalving_for_inversion',
        'error_when_glacier_reaches_boundaries', 'glacier_length_method',
        'use_inversion_params_for_run', 'ref_mb_valid_window',
        'tidewater_type']
    for k in ltr:
        cp.pop(k, None)

    # Other params are floats
    for k in cp:
        PARAMS[k] = cp.as_float(k)

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

    # Apply code-side manual params overrides (OGGM idea).
    if params:
        for k, v in params.items():
            cp[k] = v

    oggmcfg.PATHS['dem_file'] = cp['dem_file']
    oggmcfg.PATHS['data_dir'] = cp['data_dir']
    oggmcfg.PATHS['hfile'] = cp['hfile']
    oggmcfg.PATHS['climate_file'] = cp['climate_file']
    oggmcfg.PATHS['nwp_file_cosmo'] = cp['nwp_file_cosmo']
    oggmcfg.PATHS['nwp_file_ecmwf'] = cp['nwp_file_ecmwf']
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

    # create place where to store all plots, if not indicated otherwise
    oggmcfg.PATHS['plots_dir'] = os.path.join(cp['working_dir'], 'plots')
    oggmcfg.PATHS['glacier_status'] = os.path.join(cp['working_dir'],
                                                   'glacier_status.geojson')

    # run params
    oggmcfg.PARAMS.do_log = False
    oggmcfg.PARAMS['run_period'] = [int(vk) for vk in cp.as_list('run_period')]
    k = 'glamos_ids'
    oggmcfg.PARAMS[k] = [str(vk) for vk in cp.as_list(k)]

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
    oggmcfg.PARAMS['continue_on_error'] = cp.as_bool('continue_on_error')

    # Climate
    oggmcfg.PARAMS['sis_delivery_delay'] = cp.as_bool('sis_delivery_delay')
    oggmcfg.PARAMS['temp_use_local_gradient'] = cp.as_int(
        'temp_use_local_gradient_cells')  # overwrite
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
    oggmcfg.PARAMS['begin_mbyear_month'] = cp.as_int('begin_mbyear_month')
    oggmcfg.PARAMS['begin_mbyear_day'] = cp.as_int('begin_mbyear_day')
    oggmcfg.PARAMS['albedo_method'] = cp['albedo_method']

    # Inversion
    oggmcfg.PARAMS['invert_with_sliding'] = cp.as_bool('invert_with_sliding')
    _k = 'optimize_inversion_params'
    oggmcfg.PARAMS[_k] = cp.as_bool(_k)

    # Flowline model
    _k = 'use_optimized_inversion_params'
    oggmcfg.PARAMS[_k] = cp.as_bool(_k)

    # bounds
    k = 'swe_bounds'
    oggmcfg.PARAMS[k] = [float(vk) if type(vk) == float else vk for vk in
                         cp.as_list(k)]
    k = 'alpha_bounds'
    oggmcfg.PARAMS[k] = [float(vk) if type(vk) == float else vk for vk in
                         cp.as_list(k)]
    k = 'tacc_bounds'
    oggmcfg.PARAMS[k] = [float(vk) if type(vk) == float else vk for vk in
                         cp.as_list(k)]


    # run params
    PARAMS['run_period'] = [int(vk) for vk in cp.as_list('run_period')]
    k = 'glamos_ids'
    PARAMS[k] = [str(vk) for vk in cp.as_list(k)]

    # Multiprocessing pool
    PARAMS['use_multiprocessing'] = cp.as_bool('use_multiprocessing')
    PARAMS['mp_processes'] = cp.as_int('mp_processes')

    # Some non-trivial params
    PARAMS['grid_dx_method'] = cp['grid_dx_method']
    PARAMS['topo_interp'] = cp['topo_interp']
    PARAMS['use_divides'] = cp.as_bool('use_divides')
    PARAMS['use_intersects'] = cp.as_bool('use_intersects')
    PARAMS['use_compression'] = cp.as_bool('use_compression')
    PARAMS['use_tar_shapefiles'] = cp.as_bool('use_tar_shapefiles')
    PARAMS['mpi_recv_buf_size'] = cp.as_int('mpi_recv_buf_size')
    PARAMS['use_multiple_flowlines'] = cp.as_bool(
        'use_multiple_flowlines')
    PARAMS['optimize_thick'] = cp.as_bool('optimize_thick')
    PARAMS['filter_min_slope'] = cp.as_bool('filter_min_slope')
    PARAMS['auto_skip_task'] = cp.as_bool('auto_skip_task')
    PARAMS['run_mb_calibration'] = cp.as_bool('run_mb_calibration')
    PARAMS['continue_on_error'] = cp.as_bool('continue_on_error')

    # Climate
    PARAMS['temp_use_local_gradient'] = cp.as_int(
        'temp_use_local_gradient_cells')  # overwrite
    k = 'temp_local_gradient_bounds'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    PARAMS['prcp_use_local_gradient'] = cp.as_int(
        'prcp_use_local_gradient')
    k = 'prcp_local_gradient_bounds'
    PARAMS[k] = [float(vk) for vk in cp.as_list(k)]
    PARAMS['precip_ratio_method'] = cp['precip_ratio_method']
    k = 'tstar_search_window'
    PARAMS[k] = [int(vk) for vk in cp.as_list(k)]
    PARAMS['use_bias_for_run'] = cp.as_bool('use_bias_for_run')
    _factor = cp['prcp_scaling_factor']
    if _factor not in ['stddev', 'stddev_perglacier']:
        _factor = cp.as_float('prcp_scaling_factor')
    PARAMS['prcp_scaling_factor'] = _factor
    PARAMS['tminmax_available'] = cp.as_int('tminmax_available')
    PARAMS['begin_mbyear_month'] = cp.as_int('begin_mbyear_month')
    PARAMS['begin_mbyear_day'] = cp.as_int('begin_mbyear_day')
    PARAMS['albedo_method'] = cp['albedo_method']

    # Inversion
    PARAMS['invert_with_sliding'] = cp.as_bool('invert_with_sliding')
    _k = 'optimize_inversion_params'
    PARAMS[_k] = cp.as_bool(_k)

    # Flowline model
    _k = 'use_optimized_inversion_params'
    PARAMS[_k] = cp.as_bool(_k)

    # bounds
    k = 'swe_bounds'
    PARAMS[k] = [float(vk) if type(vk) == float else vk for vk in
                         cp.as_list(k)]
    k = 'alpha_bounds'
    PARAMS[k] = [float(vk) if type(vk) == float else vk for vk in
                         cp.as_list(k)]
    k = 'tacc_bounds'
    PARAMS[k] = [float(vk) if type(vk) == float else vk for vk in
                         cp.as_list(k)]

    # Make sure we have a proper cache dir
    from oggm.utils import download_oggm_files
    download_oggm_files()

    CPARAMS['bgday_hydro'] = cp.as_int('bgday_hydro')
    CPARAMS['bgmon_hydro'] = cp.as_int('bgmon_hydro')
    PARAMS['bgday_hydro'] = cp.as_int('bgday_hydro')
    PARAMS['bgmon_hydro'] = cp.as_int('bgmon_hydro')

    try:
        use_mp_spawn = bool(int(os.environ['OGGM_USE_MP_SPAWN']))
        msg = 'ON' if use_mp_spawn else 'OFF'
        log.workflow('MP spawn context switched {} '.format(
            msg) + 'according to the ENV variable OGGM_USE_MP_SPAWN')
    except KeyError:
        use_mp_spawn = cp.as_bool('use_mp_spawn')
    oggmcfg.PARAMS['use_mp_spawn'] = use_mp_spawn
    PARAMS['use_mp_spawn'] = use_mp_spawn

    # Delete non-floats
    ltr = ['working_dir', 'dem_file', 'climate_file', 'climate_dir',
           'wgms_rgi_links', 'glathida_rgi_links', 'firncore_dir', 'lfi_dir',
           'lfi_worksheet', 'dem_dir', 'hfile', 'grid_dx_method', 'data_dir',
           'mp_processes', 'use_multiprocessing', 'use_divides',
           'temp_use_local_gradient', 'prcp_use_local_gradient',
           'sis_delivery_delay', 'temp_local_gradient_bounds', 'mb_dir',
           'modelrun_backup_dir_1', 'modelrun_backup_dir_2',
           'prcp_local_gradient_bounds',
           'precip_ratio_method', 'topo_interp', 'use_compression',
           'use_tar_shapefiles', 'bed_shape', 'continue_on_error',
           'use_optimized_inversion_params', 'invert_with_sliding',
           'optimize_inversion_params', 'use_multiple_flowlines',
           'leclercq_rgi_links', 'optimize_thick', 'mpi_recv_buf_size',
           'tstar_search_window', 'use_bias_for_run', 'run_period',
           'prcp_scaling_factor', 'tminmax_available', 'use_intersects',
           'filter_min_slope', 'auto_skip_task', 'correct_for_neg_flux',
           'problem_glaciers', 'bgmon_hydro', 'bgday_hydro',
           'run_mb_calibration', 'albedo_method', 'glamos_ids',
           'begin_mbyear_month', 'begin_mbyear_day', 'swe_bounds',
           'alpha_bounds', 'tacc_bounds', 'nwp_file_cosmo', 'use_mp_spawn']
    for k in ltr:
        cp.pop(k, None)

    # Update the dicts in case there are changes
    oggmcfg.PATHS.update(CPATHS)
    oggmcfg.PARAMS.update(PARAMS)

    BASENAMES = oggmcfg.BASENAMES
    PATHS = oggmcfg.PATHS
    PARAMS = oggmcfg.PARAMS

    # Empty defaults
    # MUST COME AFTER UPDATING THE OGGM PARAMS WITH CPARAMS
    set_intersects_db()
    # Empty defaults
    from oggm.utils import get_demo_file
    set_intersects_db(get_demo_file('rgi_intersect_oetztal.shp'))

    IS_INITIALIZED = True

    # Do not spam
    PARAMS.do_log = True
