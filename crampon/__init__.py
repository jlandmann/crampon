from __future__ import absolute_import, division
import logging

# Spammers
logging.getLogger("Fiona").setLevel(logging.WARNING)
logging.getLogger("shapely").setLevel(logging.WARNING)
logging.getLogger("rasterio").setLevel(logging.WARNING)
logging.getLogger("paramiko").setLevel(logging.WARNING)

# Basic config from OGGM
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)


try:
    from oggm.mpi import _init_oggm_mpi
    _init_oggm_mpi()
except ImportError:
    pass

# API
from crampon.utils import GlacierDirectory, entity_task, global_task