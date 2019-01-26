from __future__ import absolute_import, division
import crampon.cfg as cfg

""" Shortcuts to the functions of OGGM and crampon
"""

# Entity tasks
from crampon.core.preprocessing.gis import define_glacier_region
from crampon.core.preprocessing.gis import glacier_masks
from crampon.core.preprocessing.centerlines import compute_centerlines
from crampon.core.preprocessing.centerlines import compute_downstream_line
from crampon.core.preprocessing.centerlines import catchment_area
from crampon.core.preprocessing.centerlines import catchment_intersections
from crampon.core.preprocessing.centerlines import initialize_flowlines
from crampon.core.preprocessing.centerlines import catchment_width_geom
from crampon.core.preprocessing.centerlines import catchment_width_correction
from oggm.core.climate import glacier_mu_candidates
from oggm.core.climate import process_cru_data
from crampon.core.preprocessing.climate import process_custom_climate_data

from oggm.utils import copy_to_basedir
from oggm.core.climate import apparent_mb_from_linear_mb
from oggm.core.inversion import prepare_for_inversion
from oggm.core.inversion import mass_conservation_inversion
from oggm.core.inversion import filter_inversion_output
from oggm.core.inversion import distribute_thickness_interp
from oggm.core.flowline import init_present_time_glacier
from oggm.core.flowline import run_random_climate

# Global tasks
from oggm.core.climate import process_histalp_data
from oggm.core.climate import compute_ref_t_stars
from oggm.core.climate import compute_ref_t_stars