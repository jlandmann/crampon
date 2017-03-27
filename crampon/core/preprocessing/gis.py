from salem import Grid, wgs84
import numpy as np
import pyproj
from oggm.core.preprocessing.gis import gaussian_blur, _check_geometry,\
    _interp_polygon, _polygon_to_pix, define_glacier_region, glacier_masks


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
