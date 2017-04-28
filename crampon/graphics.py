from __future__ import division

from crampon.utils import entity_task, global_task
import logging
from matplotlib.ticker import NullFormatter
from oggm.graphics import *

# Local imports
import crampon.cfg as cfg

# Module logger
log = logging.getLogger(__name__)

nullfmt = NullFormatter()  # no labels
