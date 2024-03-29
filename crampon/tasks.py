from __future__ import absolute_import, division
import crampon.cfg as cfg

""" Shortcuts to the functions of OGGM and crampon
"""

# Entity tasks
from crampon.core.preprocessing.gis import define_glacier_region
from crampon.core.preprocessing.gis import glacier_masks
from crampon.core.preprocessing.gis import simple_glacier_masks
from crampon.core.preprocessing.centerlines import compute_centerlines
from crampon.core.preprocessing.centerlines import compute_downstream_line
from crampon.core.preprocessing.centerlines import compute_downstream_bedshape
from crampon.core.preprocessing.centerlines import catchment_area
from crampon.core.preprocessing.centerlines import catchment_intersections
from crampon.core.preprocessing.centerlines import initialize_flowlines
from crampon.core.preprocessing.centerlines import catchment_width_geom
from crampon.core.preprocessing.centerlines import catchment_width_correction
from crampon.core.preprocessing.climate import process_custom_climate_data
from crampon.core.preprocessing.climate import update_climate, process_nwp_data
from crampon.core.preprocessing.radiation import \
    get_potential_irradiation_with_toposhade, distribute_ipot_on_flowlines, \
    get_potential_irradiation_corripio, calculate_and_distribute_ipot
#from operational.mb_production import make_mb_clim, make_mb_current_mbyear, \
#    make_mb_prediction
from crampon.graphics import plot_cumsum_climatology_and_current, \
    plot_interactive_mb_spaghetti_html
from crampon.workflow import fetch_glacier_status

from oggm.utils import copy_to_basedir
from oggm.core.climate import apparent_mb_from_linear_mb
from oggm.core.inversion import prepare_for_inversion
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.inversion import filter_inversion_output
from oggm.core.inversion import distribute_thickness_interp
from oggm.core.flowline import init_present_time_glacier
from oggm.core.flowline import run_random_climate

# Global tasks
from oggm.core.climate import compute_ref_t_stars
from oggm.core.climate import compute_ref_t_stars
from crampon.graphics import make_mb_popup_map