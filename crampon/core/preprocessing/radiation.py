import numpy as np
import math
import xarray as xr
from pysolar.solar import *
from pysolar import radiation
import datetime
import pandas as pd
import rasterio
import netCDF4
import matplotlib.pyplot as plt
from numba import jit
import salem
from crampon import utils
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge as merge_tool
from scipy.interpolate import interp1d
from crampon import cfg


@jit(nopython=True)
def bresenham(x0, y0, x1, y1):
    """
    Yield integer coordinates on the line from (x0, y0) to (x1, y1) [1]_.

    Input coordinates should be integers and the result will contain both the
    start and the end point. Altered from [2]_.

    Parameters
    ----------
    x0: int
        Start x coordinate.
    y0: int
        Start y coordinate.
    x1: int
        End x coordinate.
    y1: int
        End y coordinate,

    Returns
    -------

    References
    ----------
    .. [1] : J. E. Bresenham, "Algorithm for computer control of a digital
             plotter", IBM Systems Journal, Vol. 4, No. 1, pp. 25-30, 1965.
             doi: 10.1147/sj.41.0025
    .. [2] : https://github.com/encukou/bresenham/blob/master/bresenham.py
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if D >= 0:
            y += 1
            D -= 2*dx
        D += 2 * dy


@jit(nopython=True)
def make_elevation_angle_grid(x, y, xx, yy, dx, dem_array):
    """
    Calculate elevation angles from a pixel to other pixels on a DEM array.

    Returns the angles both in radian and degrees.

    Parameters
    ----------
    x: int
        X index of the point to consider.
    y: int
        Y index of the point to consider.
    xx: np.array
        X indices meshgrid of the DEM.
    yy: np.array
        Y indices meshgrid of the DEM.
    dx: float
        DEM resolution (m).
    dem_array: np.array
        Array of the DEM elevation values.

    Returns
    -------
    angle_grid, angle_grid_deg: np.array, np.array
        The angles to all other pixels in radian and degrees.
    """

    # calculate horizontal distance grid from point with Pythagoras
    delta_xy_abs = (np.sqrt(((x - xx) * dx) ** 2 +
                            ((y - yy) * dx) ** 2))

    # calculate vertical height distance grid with respect to this point
    delta_z_abs = dem_array - dem_array[y, x]

    angle_grid = np.arctan(delta_z_abs / delta_xy_abs)
    angle_grid_deg = np.rad2deg(angle_grid)

    return angle_grid, angle_grid_deg


def get_potential_irradiation_without_toposhade(lat_deg, lon_deg, tz='UTC',
                                                freq='10min'):
    """
    Calculate potential irradiation without considering topography.

    We just take 2018 as a dummy year, the frequency at which the potential
    irradiation is calculated can be determined by the `freq` keyword.

    Parameters
    ----------
    lat_deg: float
        Latitude for which to retrieve the potential irradiation (deg).
    lon_deg: float
        Longitude for which to retrieve the potential irradiation (deg).
    tz: str
        Time zone.
    freq: str
        Frequency at which potential irradiation shall be calculated

    Returns
    -------
    alt_azi: pd.Dataframe
        Dataframe with solar azimuth and altitude (in radians!) as well as the
        potential irradiation at given time steps.
    """

    # time zone doesn't matter as we only take daily means
    # take a dummie year
    timespan = pd.date_range('2018-01-01 12:00:00', '2018-12-31 23:55:00',
                             tz=tz, freq=freq)
    alt_azi = pd.DataFrame(index=timespan,
                           columns=['altitude_deg', 'azimuth_deg'])

    for m in timespan:
        # skip if sun is set anyway
        solar_altitude = get_altitude(lat_deg, lon_deg, m)
        if (solar_altitude <= 0.) or (np.isnan(solar_altitude)):
            continue

        alt_azi.loc[m, 'altitude_deg'] = solar_altitude
        alt_azi.loc[m, 'azimuth_deg'] = get_azimuth(lat_deg, lon_deg, m)
        alt_azi.loc[m, 'altitude_rad'] = np.deg2rad(
            alt_azi.loc[m, 'altitude_deg'])
        alt_azi.loc[m, 'azimuth_rad'] = np.deg2rad(
            alt_azi.loc[m, 'azimuth_deg'])
        alt_azi.loc[m, 'ipot'] = radiation.get_radiation_direct(m, alt_azi.loc[
            m, 'altitude_deg'])

    return alt_azi


def ipot_loop(mask, resolution, dem_array, alt_azi):
    """
    Loop over times and all grid pixels to get the potential irradiation.
    
    First, the potential irradiation is calculated for a horizontal surface and 
    then corrected for the pixel slope and aspect.
    Todo: The function has weird parameters and should be generalized.
    
    Parameters
    ----------
    mask: np.array
        Mask of the area of interest.
    resolution: float
        Grid resolution (m).
    dem_array: np.array
        Array of the DEM elevation values.
    alt_azi: pd.Dataframe
        Dataframe with solar altitudes and azimuths in both radians and
        degrees. Needs to have columns 'azimuth_deg', 'azimuth_rad',
        'altitude_deg', altitude_rad' and the index 'time'.

    Returns
    -------
    time_array_corrected: np.array
        A grid with potential solar radiation at the given times, corrected
        for slope and elevation of the pixels.
    """

    # delete all-nan rows
    alt_azi.dropna(inplace=True)
    azis = alt_azi.azimuth_deg.values.astype(float)
    azis_rad = alt_azi.azimuth_rad.values.astype(float)
    altis = alt_azi.altitude_deg.values.astype(float)
    altis_rad = alt_azi.altitude_rad.values.astype(float)
    times = [pd.Timestamp(t, tz='UTC') for t in alt_azi.index.values]

    x_coords = range(dem_array.shape[1])
    y_coords = range(dem_array.shape[0])
    max_x = max(x_coords)
    max_y = max(y_coords)
    radius = max(dem_array.shape)
    xx, yy = np.meshgrid(x_coords, y_coords)

    # prepare the output array
    time_array = np.empty((len(x_coords), len(y_coords), len(times)))
    time_array.fill(np.nan)

    # save time: only go over valid glacier cells
    valid = np.where(mask)

    npixel = 0
    # loop over space, then over time: due to angle grids, this is faster
    for y, x in zip(*valid):
        # todo: remove before making entity task
        if npixel % 100 == 0:
            print('{0} pixels ({1:.2f}%) done.'.format(
                npixel, npixel / np.count_nonzero(mask) * 100))
        npixel += 1

        angle_grid, angle_grid_deg = make_elevation_angle_grid(x, y, xx, yy,
                                                               resolution,
                                                               dem_array)

        for tindex, (azi, azi_rad, alti, t) in enumerate(zip(azis, azis_rad,
                                                             altis, times)):

            # select cells that intersect with azimuth angle (Bresenham line)
            # one index of the edge where the line hits is always a coord max
            dx = radius * math.sin(azi_rad)
            dy = radius * math.cos(azi_rad)
            end_x = int(x + dx)
            end_y = int(y - dy)

            bham_line = np.array(list(bresenham(x, y, end_x, end_y)))
            bham_line_cut = bham_line[(0 <= bham_line[:, 0]) &
                                      (bham_line[:, 0] <= max_x) &
                                      (0 <= bham_line[:, 1]) &
                                      (bham_line[:, 1] <= max_y)]

            # make the decision
            if (angle_grid_deg[bham_line_cut[:, 1],
                               bham_line_cut[:, 0]] > alti).any():
                # shadow
                time_array[x, y, tindex] = 0.
            else:
                # sun - uncorrected for terrain
                time_array[x, y, tindex] = alt_azi.loc[t, 'ipot']

    # correct potential radiation for terrain
    sy, sx = np.gradient(dem_array, resolution)
    slope = np.arctan(np.sqrt(sx ** 2 + sy ** 2))
    aspect = np.arctan2(-sx, sy)

    time_array_corrected = np.empty((len(x_coords), len(y_coords), len(times)))
    time_array_corrected.fill(np.nan)
    for i in np.arange(time_array.shape[2]):
        to_correct = time_array[:, :, i]
        corrected = correct_radiation_for_terrain(to_correct.T, 0., slope,
                                                  aspect, np.pi / 2. -
                                                  altis_rad[i], azis_rad[i])
        # clip zero: diffuse rad is set to zero (direct could be negative!)
        corrected = np.clip(corrected, 0., None)
        time_array_corrected[:, :, i] = corrected.T

    return time_array_corrected


def correct_radiation_for_terrain(r_beam, r_diff, slope, terrain_a, sun_z,
                                  sun_a):
    """
    Correct radiation on a horizontal surface for terrain slope and azimuth.

    The equation is implemented from [1]_ (eq. 69), but sun and terrain azimuth
    angles are given from true north.

    Parameters
    ----------
    r_beam: float or array-like
        Direct beam radiation on a horizontal surface (W m-2).
    r_diff: float or array-like
        Diffuse radiation on a horizontal surface (W m-2).
    slope: float or array-like
        Terrain slope from horizontal (radians).
    sun_z: float or array-like
        Sun zenith angle (radians).
    sun_a: float or array-like
        Sun azimuth angle (radians).
    terrain_a: float or array-like
        Terrain azimuth in radians from true north.

    Returns
    -------
    r_s: float or array-like
        Radiation on a slope with given properties.

    References
    ----------
    .. [1] : Ham, J. M. 2005. Useful Equations and Tables in Micrometeorology.
             In: J.L. Hatfield, J.M. Baker, editors, Micrometeorology in
             Agricultural Systems, Agron. Monogr. 47. ASA, CSSA, and SSSA,
             Madison, WI. p. 533-560. doi:10.2134/agronmonogr47.c23
    """

    r_s = r_beam * ((np.cos(sun_z) * np.cos(slope) + np.sin(sun_z) * np.sin(
        slope) * np.cos(sun_a - terrain_a)) / (np.cos(sun_z))) + r_diff

    return r_s


def scale_irradiation_with_potential_irradiation(sis, ipot):
    """
    Scale the actual measured incoming solar radiation with potential irradiation.

    This is used to apply terrain effects to the MeteoSwiss HelioSat global
    irradiation product.

    Parameters
    ----------
    sis: float
        Mean irradiation over a time span, e.g. one day.
    ipot: np.array
        Grid with the mean potential irradiation over the same time span as
        isis.

    Returns
    -------
    sis_distr: np.array
        The solar incoming radiation distributed on terrain.
    """

    # normalize Ipot
    ipot_norm = (ipot - np.nanmin(ipot)) / (np.nanmax(ipot) - np.nanmin(ipot))

    # Apply normalization to incoming solar radiation
    isis_distr = sis * ipot_norm

    return isis_distr


def test_correct_radiation_for_terrain():
    # test floats
    # sun in zenith, terrain flat
    float_result = correct_radiation_for_terrain(1000., 0., 0., 0., np.pi/2.,
                                                 np.pi)
    # sun in zenith, terrain at 90 deg
    float_result = correct_radiation_for_terrain(1000., 0., np.pi / 4., 0.,
                                                 np.pi / 2., np.pi)


    # array tests


def distribute_ipot_on_flowlines(gdir):
    """
    Distribute potential irradiation on a raster grid to the glacier flowlines.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The glacier to process potential irradiation on the flowlines for.

    Returns
    -------
    None
    """

    fls = gdir.read_pickle('inversion_flowlines')
    catchments = salem.read_shapefile(gdir.get_filepath('flowline_catchments'))
    ipot_reproj = xr.open_dataset(gdir.get_filepath('ipot'))

    dem = xr.open_rasterio(gdir.get_filepath('dem'))
    ds = dem.to_dataset(name='data')
    ds.attrs['pyproj_srs'] = dem.crs

    # Per flowline (important so that later, the indices can be moved)
    ipot_per_flowline = []

    for fl, c in zip(fls,
                     [catchments.iloc[i] for i in range(len(catchments))]):
        # mask all heights in a catchment
        c_heights = ds.salem.roi(geometry=c.geometry,
                                 crs=gdir.grid.proj.srs)
        # use ipot here, not ipot_reproj to not artificially produce values?
        ipot_subset = ipot_reproj.salem.roi(geometry=c.geometry,
                                            crs=gdir.grid.proj.srs)
        fhgt = fl.surface_h

        fl_ipot_arr = np.full((len(fhgt), 365), np.nan, float)
        for q in range(len(fhgt)):
            # todo: check if surface surface_h values are lower edge, uppder edge or mid of the interval
            try:
                cond = (c_heights.sel(band=1).data.values <= fhgt[q]) & (
                        c_heights.sel(band=1).data.values > fhgt[q + 1])
            except IndexError:
                # trick at the end of the flowline
                cond = (c_heights.sel(band=1).data.values <= fhgt[q])
            if not cond.any():
                continue
            ts_per_hgt = ipot_subset.where(cond).mean(dim=['x', 'y'],
                            skipna=True).to_array().values.flatten()
            fl_ipot_arr[q, :] = ts_per_hgt

        nnan_ix = np.unique(np.where(~np.isnan(fl_ipot_arr))[0])
        fl_ipot_interp = interp1d(np.arange(len(fhgt))[nnan_ix],
                                  fl_ipot_arr[nnan_ix, :], axis=0,
                                  fill_value='extrapolate')(range(len(fhgt)))

        ipot_per_flowline.append(fl_ipot_interp)

    gdir.write_pickle(ipot_per_flowline, 'ipot_per_flowline')


def get_potential_irradiation_with_toposhade(gdir):
    """
    Get the potential solar irradiation including topographic shading.

    Parameters
    ----------
    gdir: `py_class:crampon.GlacierDirectory`
        The GlacierDirectory to process the potential solar irradiation for.

    Returns
    -------
    None
    """

    dem = xr.open_rasterio(gdir.get_filepath('dem'))
    with netCDF4.Dataset(gdir.get_filepath('gridded_data'), 'r') as ncd:
        mask = ncd.variables['glacier_mask'][:]
    ds = dem.to_dataset(name='data')
    ds.attrs['pyproj_srs'] = dem.crs
    grid = salem.grid_from_dataset(ds)

    # extended grid to search for sun-blocking terrain
    extend_border = cfg.PARAMS['shading_border']  # meters
    ext_grid = salem.Grid(proj=grid.proj, dxdy=(grid.dx, grid.dy),
                          x0y0=(grid.x0 - extend_border, grid.y0 +
                                extend_border),
                          nxny=(int(grid.nx + 2 * extend_border / grid.dx),
                                int(grid.ny + 2 * extend_border / abs(grid.dy))
                                ))
    # regrid to coarser resolution (save time)
    ext_regrid = ext_grid.regrid(factor=cfg.PARAMS['reduce_rgrid_resolution'])
    print('new_resolution: {}'.format(ext_regrid.dx))

    # todo: get rid of double code from init_glacier_regions
    lon_ex = (ext_grid.extent_in_crs()[0], ext_grid.extent_in_crs()[1])
    lat_ex = (ext_grid.extent_in_crs()[2], ext_grid.extent_in_crs()[3])
    dems_ext_list, _ = utils.get_topo_file(lon_ex=lon_ex, lat_ex=lat_ex)

    if len(dems_ext_list) == 1:
        dem_dss = [rasterio.open(dems_ext_list[0])]  # if one tile, just open
        dem_data = rasterio.band(dem_dss[0], 1)
        src_transform = dem_dss[0].transform
    else:
        dem_dss = [rasterio.open(s) for s in dems_ext_list]  # list of rasters
        dem_data, src_transform = merge_tool(dem_dss)  # merged rasters

    dst_transform = rasterio.transform.from_origin(
        ext_regrid.x0, ext_regrid.y0, ext_regrid.dx, ext_regrid.dx
    )

    # Set up profile for writing output
    profile = dem_dss[0].profile
    profile.update({
        'crs': ext_grid.proj.srs,
        'transform': dst_transform,
        'width': ext_regrid.nx,
        'height': ext_regrid.ny
    })

    resampling = Resampling[cfg.PARAMS['topo_interp'].lower()]
    dst_array = np.empty((ext_regrid.ny, ext_regrid.nx),
                         dtype=dem_dss[0].dtypes[0])
    reproject(
        source=dem_data,
        src_crs=dem_dss[0].crs,
        src_transform=src_transform,
        destination=dst_array,
        dst_transform=dst_transform,
        dst_crs=ext_grid.proj.srs,
        resampling=resampling)
    for dem_ds in dem_dss:
        dem_ds.close()

    regrid_ds = ext_regrid.to_dataset()
    regrid_ds['height'] = (['y', 'x'], dst_array)
    new_mask = regrid_ds.salem.roi(shape=gdir.get_filepath('outlines'))

    # assume: grid is small enough that difference in sun angles doesn't matter
    latitude_deg = np.mean(grid.ll_coordinates[1])  # pos. in the northern h.
    longitude_deg = np.mean(grid.ll_coordinates[0])  # neg. west from 0 deg
    freqstr = '10min'
    alt_azi = get_potential_irradiation_without_toposhade(latitude_deg,
                                                          longitude_deg,
                                                          freq=freqstr)

    ipot_array = ipot_loop(mask=~np.isnan(new_mask).height.values,
                           resolution=ext_regrid.dx, dem_array=dst_array,
                           alt_azi=alt_azi)

    # make daily mean
    group_indices = alt_azi.groupby(by=alt_azi.index.date).indices
    ipot_array_daymean = np.empty((ipot_array.shape[0],
                                  ipot_array.shape[1], len(group_indices)))

    for nth_day, day in enumerate(np.unique(alt_azi.index.date)):

        one_day = ipot_array[:, :, group_indices[pd.Timestamp(day)]]
        # we have to account for time steps during the night
        one_day_mean = np.sum(one_day, axis=2) / (
                    pd.Timedelta(days=1) / pd.to_timedelta(
                pd.tseries.frequencies.to_offset(freqstr)))
        ipot_array_daymean[:, :, nth_day] = one_day_mean

    ipot_ds = ext_regrid.to_dataset()
    ipot_ds = ipot_ds.assign_coords(time=np.unique([np.datetime64(d) for d in
                                                    alt_azi.index.date]))
    ipot_ds['ipot'] = (['x', 'y', 'time'], ipot_array_daymean)
    ipot_ds.ipot.attrs['pyproj_srs'] = ext_regrid.to_dataset().attrs[
        'pyproj_srs']
    ipot_ds = ipot_ds.transpose()
    # distribute back to original grid
    ipot_reproj = ds.salem.transform(ipot_ds)
    ipot_reproj.to_netcdf(gdir.get_filepath('ipot'))
