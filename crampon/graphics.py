from crampon.utils import entity_task, global_task
import logging
from matplotlib.ticker import NullFormatter
from oggm.graphics import truncate_colormap, _plot_map, plot_googlemap, \
    plot_domain

# Local imports
import crampon.cfg as cfg

# Module logger
log = logging.getLogger(__name__)

nullfmt = NullFormatter()  # no labels
