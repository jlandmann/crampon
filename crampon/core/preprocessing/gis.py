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
from oggm.core.gis import gaussian_blur, multi_to_poly,\
    _interp_polygon, _polygon_to_pix, define_glacier_region, glacier_masks
from oggm.utils import get_topo_file

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


@entity_task(log, writes=['glacier_grid', 'dem', 'outlines'])
def define_glacier_region_crampon(gdir, entity=None, reset_dems=False):
    """
    Define the local grid for a glacier entity.

    Very first task: define the glacier's local grid.
    Defines the local projection (Transverse Mercator), centered on the
    glacier. There is some options to set the resolution of the local grid.
    It can be adapted depending on the size of the glacier with:

        dx (m) = d1 * AREA (km) + d2 ; clipped to dmax

    or be set to a fixed value. See the params file for setting these options.

    After defining the grid, the topography and the outlines of the glacier
    are transformed into the local projection. The default interpolation for
    the topography is `cubic`.
    This function is mainly taken over from OGGM, some modification to handle
    multitemporal data have been made.

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        Where to write the data
    entity: :py:class:`geopandas.GeoSeries`
        The glacier geometry to process
    reset_dems: bool
        Whether to reassemble DEMs from sources or not (time-consuming!). If
        DEMs are not yet present, they will be assembled anyway. Default:
        False.

    Returns
    -------
    None
    """

    # choose a spatial resolution with respect to the glacier area
    dxmethod = cfg.PARAMS['grid_dx_method']
    area = gdir.area_km2
    if dxmethod == 'linear':
        dx = np.rint(cfg.PARAMS['d1'] * area + cfg.PARAMS['d2'])
    elif dxmethod == 'square':
        dx = np.rint(cfg.PARAMS['d1'] * np.sqrt(area) + cfg.PARAMS['d2'])
    elif dxmethod == 'fixed':
        dx = np.rint(cfg.PARAMS['fixed_dx'])
    else:
        raise ValueError('grid_dx_method not supported: {}'.format(dxmethod))
    # Additional trick for varying dx
    if dxmethod in ['linear', 'square']:
        dx = np.clip(dx, cfg.PARAMS['d2'], cfg.PARAMS['dmax'])

    log.debug('(%s) area %.2f km, dx=%.1f', gdir.id, area, dx)

    # Make a local glacier map
    proj_params = dict(name='tmerc', lat_0=0., lon_0=gdir.cenlon,
                       k=0.9996, x_0=0, y_0=0, datum='WGS84')
    proj4_str = "+proj={name} +lat_0={lat_0} +lon_0={lon_0} +k={k} " \
                "+x_0={x_0} +y_0={y_0} +datum={datum}".format(**proj_params)
    proj_in = pyproj.Proj("+init=EPSG:4326", preserve_units=True)
    proj_out = pyproj.Proj(proj4_str, preserve_units=True)
    project = partial(pyproj.transform, proj_in, proj_out)

    # TODO: Here single outlines should be transformed and their union used
    # for defining the grid
    # transform geometry to map
    geometry = shapely.ops.transform(project, entity['geometry'])
    geometry = multi_to_poly(geometry, gdir=gdir)
    xx, yy = geometry.exterior.xy

    # Corners, incl. a buffer of N pix
    ulx = np.min(xx) - cfg.PARAMS['border']
    lrx = np.max(xx) + cfg.PARAMS['border']
    uly = np.max(yy) + cfg.PARAMS['border']
    lry = np.min(yy) - cfg.PARAMS['border']
    # n pixels
    nx = np.int((lrx - ulx) / dx)
    ny = np.int((uly - lry) / dx)

    # Back to lon, lat for DEM download/preparation
    tmp_grid = salem.Grid(proj=proj_out, nxny=(nx, ny), x0y0=(ulx, uly),
                          dxdy=(dx, -dx), pixel_ref='corner')
    minlon, maxlon, minlat, maxlat = tmp_grid.extent_in_crs(crs=salem.wgs84)

    # save transformed geometry to disk
    entity = entity.copy()
    entity['geometry'] = geometry
    # Avoid fiona bug: https://github.com/Toblerity/Fiona/issues/365
    for k, s in entity.iteritems():
        if type(s) in [np.int32, np.int64]:
            entity[k] = int(s)
    towrite = gpd.GeoDataFrame(entity).T
    towrite.crs = proj4_str
    # Delete the source before writing
    if 'DEM_SOURCE' in towrite:
        del towrite['DEM_SOURCE']
    towrite.to_file(gdir.get_filepath('outlines'))

    # TODO: This needs rework if it should work also for SGI
    # Also transform the intersects if necessary
    gdf = cfg.PARAMS['intersects_gdf']
    if len(gdf) > 0:
        gdf = gdf.loc[((gdf.RGIId_1 == gdir.id) |
                       (gdf.RGIId_2 == gdir.id))]
        if len(gdf) > 0:
            gdf = salem.transform_geopandas(gdf, to_crs=proj_out)
            if hasattr(gdf.crs, 'srs'):
                # salem uses pyproj
                gdf.crs = gdf.crs.srs
            gdf.to_file(gdir.get_filepath('intersects'))
    else:
        # Sanity check
        if cfg.PARAMS['use_intersects']:
            raise RuntimeError('You seem to have forgotten to set the '
                               'intersects file for this run. OGGM works '
                               'better with such a file. If you know what '
                               'your are doing, set '
                               "cfg.PARAMS['use_intersects'] = False to "
                               "suppress this error.")

    # Open DEM
    source = entity.DEM_SOURCE if hasattr(entity, 'DEM_SOURCE') else None

    if (not os.path.exists(gdir.get_filepath('dem_ts'))) or reset_dems:
        log.info('Assembling local DEMs for {}...'.format(gdir.id))
        utils.get_local_dems(gdir)

    # Use Grid properties to create a transform (see rasterio cookbook)
    dst_transform = rasterio.transform.from_origin(
        ulx, uly, dx, dx  # sign change (2nd dx) is done by rasterio.transform
    )

    # Could be extended so that the cfg file takes all Resampling.* methods
    if cfg.PARAMS['topo_interp'] == 'bilinear':
        resampling = Resampling.bilinear
    elif cfg.PARAMS['topo_interp'] == 'cubic':
        resampling = Resampling.cubic
    else:
        raise ValueError('{} interpolation not understood'
                         .format(cfg.PARAMS['topo_interp']))

    dem_source_list = [cfg.NAMES['DHM25'], cfg.NAMES['SWISSALTI2010'],
                       cfg.NAMES['LFI']]
    for demtype in dem_source_list:
        print(demtype)
        try:
            data = xr.open_dataset(gdir.get_filepath('dem_ts'), group=demtype)
        except OSError:  # group not found
            print('group {} not found'.format(demtype))
            continue

        # check latitude order (latitude needs to be decreasing as we have
        # to create own transform and dy is automatically negative in rasterio)
        if data.coords['y'][0].item() < data.coords['y'][-1].item():
            data = data.sel(y=slice(None, None, -1))

        for t in data.time:
            dem = data.sel(time=t)

            dem_arr = dem.height.values

            src_transform = rasterio.transform. \
                from_origin(min(dem.coords['x'].values),  # left
                            max(dem.coords['y'].values),  # upper
                            np.abs(
                                dem.coords['x'][1].item() - dem.coords['x'][
                                    0].item()),
                            np.abs(
                                dem.coords['y'][1].item() - dem.coords['y'][
                                    0].item()))

            # Set up profile for writing output
            profile = {'crs': proj4_str,
                       'nodata': dem.height.encoding['_FillValue'],
                       'dtype': str(dem.height.encoding['dtype']),
                       'count': 1,
                       'transform': dst_transform,
                       'interleave': 'band',
                       'driver': 'GTiff',
                       'width': nx,
                       'height': ny,
                       'tiled': False}

            base, ext = os.path.splitext(gdir.get_filepath('dem'))
            dem_reproj = base + str(t.item()) + ext
            with rasterio.open(dem_reproj, 'w', **profile) as dest:
                dst_array = np.empty((ny, nx),
                                     dtype=str(dem.height.encoding['dtype']))
                dst_array[:] = np.nan

                reproject(
                    # Source parameters
                    source=dem_arr,
                    src_crs=dem.pyproj_srs,
                    src_transform=src_transform,
                    # Destination parameters
                    destination=dst_array,
                    dst_transform=dst_transform,
                    dst_crs=proj4_str,
                    dst_nodata=np.nan,
                    # Configuration
                    resampling=resampling)

                dest.write(dst_array, 1)

    # Glacier grid
    x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
    glacier_grid = salem.Grid(proj=proj_out, nxny=(nx, ny),  dxdy=(dx, -dx),
                              x0y0=x0y0)
    glacier_grid.to_json(gdir.get_filepath('glacier_grid'))
    gdir.write_pickle(dem_source_list, 'dem_source')

# Important, overwrite OGGM function
define_glacier_region = define_glacier_region_crampon
