from __future__ import absolute_import, division

from distutils.version import LooseVersion
import os
import numpy as np
import pandas as pd
import pyproj
import logging
import xarray as xr
import itertools
from crampon import entity_task
from crampon import utils
import crampon.cfg as cfg
from functools import partial
import geopandas as gpd
import shapely
import salem
from salem import Grid, wgs84
from oggm.core.gis import gaussian_blur, _interp_polygon, _polygon_to_pix, \
    define_glacier_region, glacier_masks, simple_glacier_masks, \
    _parse_source_text
from oggm.utils import get_topo_file, is_dem_source_available
from scipy import stats

from oggm.exceptions import InvalidParamsError

import rasterio
from rasterio.warp import reproject, Resampling, SUPPORTED_RESAMPLING
try:
    # rasterio V > 1.0
    from rasterio.merge import merge as merge_tool
except ImportError:
    from rasterio.tools.merge import merge as merge_tool

# Module logger
log = logging.getLogger(__name__)


DEM_SOURCE_INFO = _parse_source_text()

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


@entity_task(log, writes=['glacier_grid', 'homo_dem_ts', 'outlines'])
def define_glacier_region_crampon(gdir, entity=None, oggm_dem_source='SRTM',
                                  reset_dems=False):
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
        Like in OGGM: the glacier geometry to process - DEPRECATED. It is now
        ignored
    oggm_dem_source: str, optional
        Preferred source for basic OGGM DEM. At the moment, this is set to
        "SRTM", because the SGI IDs are not in the DEFAULT_DEM_SOURCE table.
    reset_dems: bool
        Whether to reassemble DEMs from sources or not (time-consuming!). If
        DEMs are not yet present, they will be assembled anyway. Default:
        False.

    Returns
    -------
    None
    """

    # OGGM
    source = oggm_dem_source

    # Get the local map proj params and glacier extent
    gdf = gdir.read_shapefile('outlines')

    # Get the map proj
    utm_proj = salem.check_crs(gdf.crs)

    # Get glacier extent
    xx, yy = gdf.iloc[0]['geometry'].exterior.xy

    # Define glacier area to use
    area = gdir.area_km2

    # Choose a spatial resolution with respect to the glacier area
    dx = utils.dx_from_area(area)

    log.debug('(%s) area %.2f km, dx=%.1f', gdir.id, area, dx)

    # Safety check
    border = cfg.PARAMS['border']
    if border > 1000:
        raise InvalidParamsError("You have set a cfg.PARAMS['border'] value "
                                 "of {}. ".format(
            cfg.PARAMS['border']) + 'This a very large value, which is '
                                    'currently not supported in OGGM.')

    # For tidewater glaciers we force border to 10
    if gdir.is_tidewater and cfg.PARAMS['clip_tidewater_border']:
        border = 10

    # Corners, incl. a buffer of N pix
    ulx = np.min(xx) - cfg.PARAMS['border']
    lrx = np.max(xx) + cfg.PARAMS['border']
    uly = np.max(yy) + cfg.PARAMS['border']
    lry = np.min(yy) - cfg.PARAMS['border']
    # n pixels
    nx = np.int((lrx - ulx) / dx)
    ny = np.int((uly - lry) / dx)

    # Back to lon, lat for DEM download/preparation
    tmp_grid = salem.Grid(proj=utm_proj, nxny=(nx, ny), x0y0=(ulx, uly),
                          dxdy=(dx, -dx), pixel_ref='corner')
    minlon, maxlon, minlat, maxlat = tmp_grid.extent_in_crs(crs=salem.wgs84)

    # TODO: This needs rework if it should work also for SGI
    # Also transform the intersects if necessary
    gdf = cfg.PARAMS['intersects_gdf']
    if len(gdf) > 0:
        gdf = gdf.loc[((gdf.RGIId_1 == gdir.id) |
                       (gdf.RGIId_2 == gdir.id))]
        if len(gdf) > 0:
            gdf = salem.transform_geopandas(gdf, to_crs=utm_proj)
            if hasattr(gdf.crs, 'srs'):
                # salem uses pyproj
                gdf.crs = gdf.crs.srs
            gdir.write_shapefile(gdf, 'intersects')
    else:
        # Sanity check
        if cfg.PARAMS['use_intersects']:
            raise RuntimeError('You seem to have forgotten to set the '
                               'intersects file for this run. OGGM works '
                               'better with such a file. If you know what '
                               'your are doing, set '
                               "cfg.PARAMS['use_intersects'] = False to "
                               "suppress this error.")

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
    oggm_dem = False
    homo_dems = []
    homo_dates = []
    for demtype in dem_source_list:
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

            # if demtype == cfg.NAMES['LFI']:
            #    # make the check
            #    check_res = dem_quality_check(gdir, dem)
            #    if check_res is None:
            #        continue

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
            profile = {'crs': utm_proj.srs,
                       'nodata': np.nan,
                       'dtype': np.float32,
                       'count': 1,
                       'transform': dst_transform,
                       'interleave': 'band',
                       'driver': 'GTiff',
                       'width': nx,
                       'height': ny,
                       'tiled': False}

            base, ext = os.path.splitext(gdir.get_filepath('dem'))
            dem_reproj = base + str(t.time.dt.year.item()) + ext
            with rasterio.open(dem_reproj, 'w', **profile) as dest:
                dst_array = np.empty((ny, nx),
                                     dtype=np.float32)
                dst_array[:] = np.nan

                reproject(
                    # Source parameters
                    source=dem_arr,
                    src_crs=dem.pyproj_srs,
                    src_transform=src_transform,
                    # Destination parameters
                    destination=dst_array,
                    dst_transform=dst_transform,
                    dst_crs=utm_proj.srs,
                    dst_nodata=np.nan,
                    # Configuration
                    resampling=resampling)

                dest.write(dst_array, 1)

                homo_dems.append(dst_array)
                homo_dates.append(t.time.values)

        # Open DEM
        # We test DEM availability for glacier only (maps can grow big)
        if not is_dem_source_available(source, *gdir.extent_ll):
            log.warning('Source: {} may not be available for glacier {} with '
                        'border {}'.format(source, gdir.rgi_id, border))
        dem_list, dem_source = get_topo_file((minlon, maxlon),
                                             (minlat, maxlat),
                                             rgi_id=gdir.rgi_id, dx_meter=dx,
                                             source=source)
        log.debug('(%s) DEM source: %s', gdir.rgi_id, dem_source)
        log.debug('(%s) N DEM Files: %s', gdir.rgi_id, len(dem_list))

        # Decide how to tag nodata
        def _get_nodata(rio_ds):
            nodata = rio_ds[0].meta.get('nodata', None)
            if nodata is None:
                # badly tagged geotiffs, let's do it ourselves
                nodata = -32767 if source == 'TANDEM' else -9999
            return nodata

        # A glacier area can cover more than one tile:
        if len(dem_list) == 1:
            dem_dss = [rasterio.open(dem_list[0])]  # if one tile, just open it
            dem_data = rasterio.band(dem_dss[0], 1)
            if LooseVersion(rasterio.__version__) >= LooseVersion('1.0'):
                src_transform = dem_dss[0].transform
            else:
                src_transform = dem_dss[0].affine
            nodata = _get_nodata(dem_dss)
        else:
            dem_dss = [rasterio.open(s) for s in dem_list]  # list of rasters
            nodata = _get_nodata(dem_dss)
            dem_data, src_transform = merge_tool(dem_dss,
                                                 nodata=nodata)  # merge

        # Use Grid properties to create a transform (see rasterio cookbook)
        dst_transform = rasterio.transform.from_origin(ulx, uly, dx, dx
            # sign change (2nd dx) is done by rasterio.transform
        )

        # Set up profile for writing output
        profile = dem_dss[0].profile
        profile.update(
            {'crs': utm_proj.srs, 'transform': dst_transform, 'nodata': nodata,
                'width': nx, 'height': ny, 'driver': 'GTiff'})

        # Could be extended so that the cfg file takes all Resampling.* methods
        if cfg.PARAMS['topo_interp'] == 'bilinear':
            resampling = Resampling.bilinear
        elif cfg.PARAMS['topo_interp'] == 'cubic':
            resampling = Resampling.cubic
        else:
            raise InvalidParamsError('{} interpolation not understood'.format(
                cfg.PARAMS['topo_interp']))

        dem_reproj = gdir.get_filepath('dem')
        profile.pop('blockxsize', None)
        profile.pop('blockysize', None)
        profile.pop('compress', None)
        with rasterio.open(dem_reproj, 'w', **profile) as dest:
            dst_array = np.empty((ny, nx), dtype=dem_dss[0].dtypes[0])
            reproject(# Source parameters
                source=dem_data, src_crs=dem_dss[0].crs,
                src_transform=src_transform, src_nodata=nodata,
                # Destination parameters
                destination=dst_array, dst_transform=dst_transform,
                dst_crs=utm_proj.srs, dst_nodata=nodata, # Configuration
                resampling=resampling)
            dest.write(dst_array, 1)

        for dem_ds in dem_dss:
            dem_ds.close()

    oggm_dem = True

    # Glacier grid
    x0y0 = (ulx+dx/2, uly-dx/2)  # To pixel center coordinates
    glacier_grid = salem.Grid(proj=utm_proj, nxny=(nx, ny),  dxdy=(dx, -dx),
                              x0y0=x0y0)
    glacier_grid.to_json(gdir.get_filepath('glacier_grid'))
    gdir.write_pickle(dem_source_list, 'dem_source')

    # Write DEM source info
    gdir.add_to_diagnostics('dem_source', dem_source)
    source_txt = DEM_SOURCE_INFO.get(dem_source, dem_source)
    with open(gdir.get_filepath('dem_source'), 'w') as fw:
        fw.write(source_txt)
        fw.write('\n\n')
        fw.write('# Data files\n\n')
        for fname in dem_list:
            fw.write('{}\n'.format(os.path.basename(fname)))

    # write homo dem time series
    homo_dem_ts = xr.Dataset(
        {'height': (['time', 'y', 'x'], np.array(homo_dems))},
        coords={'x': np.linspace(dst_transform[2], dst_transform[2] + nx *
                                 dst_transform[0], nx),
                'y': np.linspace(dst_transform[5], dst_transform[5] + ny *
                                 dst_transform[4], ny),
                'time': homo_dates},
        attrs={'id': gdir.rgi_id, 'name': gdir.name, 'res': dx,
               'pyproj_srs': utm_proj.srs, 'transform': dst_transform})
    homo_dem_ts = homo_dem_ts.sortby('time')
    homo_dem_ts.to_netcdf(gdir.get_filepath('homo_dem_ts'))

    _ = calculate_geodetic_deltav(gdir)


# Important, overwrite OGGM function
define_glacier_region = define_glacier_region_crampon


def calculate_geodetic_deltav(gdir, fill_threshold=0.1):
    """
    Calculate all possible difference grids and geodetic volume changes.

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        Glacier directory to get the geodetic volume changes for.
    fill_threshold: float
        Maximum allowed ratio of good to missing pixel values. Default: 0.1
        (10%). If this value is exceeded, no geodetic volume change will be
        calculated.

    Returns
    -------
    gvc_df: pandas.Dataframe
        Dataframe containing the geodetic volume changes and the dates 1 and 2
        of the underlying DEMs.
    """

    dems = xr.open_dataset(gdir.get_filepath('homo_dem_ts'))

    # select ROI
    # todo: this should become multitemporal later
    dems_sel = dems.salem.roi(shape=gdir.get_filepath('outlines'))
    gvc_df = pd.DataFrame()

    ix = 0
    for d1, d2 in itertools.combinations(dems_sel.height, 2):
        # only go forward in time
        if d1.time.item() > d2.time.item():
            continue

        diff = d2 - d1

        # fill gaps with with mean
        mean_hgt = (d1 + d2) / 2.
        glacier_mask = (~np.isnan(d1) & ~np.isnan(d2))
        missing_mask = np.ma.masked_array(glacier_mask,
                                          mask=gdir.grid.region_of_interest(
                                              shape=gdir.get_filepath(
                                                  'outlines')).astype(bool))

        # if NaNs are more than the fill threshold, continue
        fill_ratio = round(
            len(missing_mask.nonzero()[0]) / missing_mask.count(), 3)
        if fill_ratio > fill_threshold:
            continue

        slope, itcpt, _, p_val, _ = stats.linregress(
            np.ma.masked_array(mean_hgt.values,
                               ~missing_mask.mask).compressed(),
            diff.values[missing_mask.mask].flatten())
        diff.values[missing_mask.mask] = (
                    itcpt + slope * mean_hgt.values[missing_mask.mask]) if \
            (p_val < 0.01) else np.nan

        # calculate dV assuming that missing cells have the average dV
        dv = np.nansum(diff) * dems_sel.res ** 2

        gvc_df.loc[ix, 'dv'] = round(dv.item())
        gvc_df.loc[ix, 'date0'] = pd.Timestamp(d1.time.item())
        gvc_df.loc[ix, 'date1'] = pd.Timestamp(d2.time.item())
        gvc_df.loc[ix, 'fill_ratio'] = fill_ratio

        ix += 1

    gvc_df.to_csv(gdir.get_filepath('geodetic_dv'))


def scipy_idw(x, y, z, xi, yi):
    """
    Inverse distance weight unsing scipy's radial basis function.

    From _[1].

    Parameters
    ----------
    x: array
        x coordinates of the data points.
    y: array
        y coordinates of the data points.
    z: array
        Value of the data points
    xi: array
        x values of the points where to interpolate.
    yi: array
        y values of the points where to interpolate.

    Returns
    -------
    array:
        Interpolated values at xi and yi.

    References
    ----------
    _[1]: https://stackoverflow.com/questions/3104781/inverse-distance-weighted-idw-interpolation-with-python
    """
    from scipy.interpolate import Rbf
    interp = Rbf(x, y, z, function='linear')
    return interp(xi, yi)


def distance_matrix(x0, y0, x1, y1):
    """
    Make a distance matrix between pairwise observations.

    Modified from _[1].

    Parameters
    ----------
    x0: array
        X coordinates of the first observations.
    y0: array
        Y coordinates of the first observations.
    x1: array
        X coordinates of the second observations.
    y1: array
        Y coordinates of the second observations.

    Returns
    -------
    array:
        Array with distances between each of the observation points.

    References
    ----------
    _[1]: http://stackoverflow.com/questions/1871536
    """
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:, 0], interp[:, 0])
    d1 = np.subtract.outer(obs[:, 1], interp[:, 1])

    return np.hypot(d0, d1)


def simple_idw(x, y, z, xi, yi):
    """
    Simple inverse distance weighting.

    From _[1].


    Parameters
    ----------
    x: array
        X coordinates of the observations.
    y: array
        Y coordinates of the observations.
    z: array
        Observation values.
    xi: array
        X coordinates where to interpolate the observations to.
    yi: array
        Y coordinates where to interpolate the observations to.

    Returns
    -------
    zi: array
        Interpolated (extrapolated) observations at points given in xi and yi.
    """
    dist = distance_matrix(x, y, xi, yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi
