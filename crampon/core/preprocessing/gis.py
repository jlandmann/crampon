from __future__ import absolute_import, division

from distutils.version import LooseVersion
from salem import Grid, wgs84
import os
import numpy as np
import pyproj
import logging
import xarray as xr
from crampon import entity_task
from crampon import utils
import crampon.cfg as cfg
from functools import partial
import geopandas as gpd
import shapely
import salem
from oggm.core.gis import gaussian_blur, _check_geometry,\
    _interp_polygon, _polygon_to_pix, define_glacier_region, glacier_masks

import rasterio
from rasterio.warp import reproject, Resampling
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool

# Module logger
log = logging.getLogger(__name__)


def merge_rasters_rasterio(to_merge, outpath=None, outformat="Gtiff"):
    """
    Merges rasters to a single one using rasterio.

    Parameters
    ----------
    to_merge: list or str
        List of paths to the rasters to be merged.
    outpath: str, optional
        Path where to write the merged raster.
    outformat: str, optional
        Any format rasterio/GDAL has a driver for. Default: GeoTiff ('Gtiff').

    Returns
    -------
    merged, profile: tuple of (numpy.ndarray, rasterio.Profile)
        The merged raster and numpy array and its rasterio profile.
    """
    to_merge = [rasterio.open(s) for s in to_merge]
    merged, output_transform = merge_tool(to_merge)

    profile = to_merge[0].profile
    if 'affine' in profile:
        profile.pop('affine')
    profile['transform'] = output_transform
    profile['height'] = merged.shape[1]
    profile['width'] = merged.shape[2]
    profile['driver'] = outformat
    if outpath:
        with rasterio.open(outpath, 'w', **profile) as dst:
            dst.write(merged)
        for rf in to_merge:
            rf.close()

    return merged, profile



# This could go to salem via a fork
def utm_grid(center_ll=None, extent=None, ny=600, nx=None,
             origin='lower-left'):
    """Local UTM centered on a specified point.

    Parameters
    ----------
    center_ll : (float, float)
        tuple of lon, lat coordinates where the map will be centered.
    extent : (float, float)
        tuple of eastings, northings giving the extent (in m) of the map
    ny : int
        number of y grid points wanted to cover the map (default: 600)
    nx : int
        number of x grid points wanted to cover the map (mutually exclusive
        with y)
    origin : str
        'lower-left' or 'upper-left'

    Returns
    -------
    A salem.Grid instance
    """

    # Make a local proj
    lon, lat = center_ll
    proj_params = dict(proj='tmerc', lat_0=0., lon_0=lon,
                       k=0.9996, x_0=0, y_0=0, datum='WGS84')
    projloc = pyproj.Proj(proj_params)

    # Define a spatial resolution
    xx = extent[0]
    yy = extent[1]
    if nx is None:
        nx = ny * xx / yy
    else:
        ny = nx * yy / xx

    nx = np.rint(nx)
    ny = np.rint(ny)

    e, n = pyproj.transform(wgs84, projloc, lon, lat)

    if origin == 'upper-left':
        corner = (-xx / 2. + e, yy / 2. + n)
        dxdy = (xx / nx, - yy / ny)
    else:
        corner = (-xx / 2. + e, -yy / 2. + n)
        dxdy = (xx / nx, yy / ny)

    return Grid(proj=projloc, x0y0=corner, nxny=(nx, ny), dxdy=dxdy,
                pixel_ref='corner')
