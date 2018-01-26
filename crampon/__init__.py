from __future__ import absolute_import, division
import logging

# Check if OGGM is installed
try:
    from oggm.version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('oggm is not properly installed. If you are running '
                      'from the source directory, please instead create a '
                      'new virtual environment (using conda or virtualenv) '
                      'and  then install it in-place by running: '
                      'pip install -e .')

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