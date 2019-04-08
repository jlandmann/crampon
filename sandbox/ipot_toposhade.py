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

# https://github.com/encukou/bresenham/blob/master/bresenham.py
@jit(nopython=True)
def bresenham(x0, y0, x1, y1):
    """
    Yield integer coordinates on the line from (x0, y0) to (x1, y1).
    Input coordinates should be integers.
    The result will contain both the start and the end point.
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
def make_angle_grid_numba(x, y, xx, yy, resolution, onedem_array):
    # subtract x, y indices of point to consider from xx and yy
    # calculate horizontal distance grid from point with help of Pythagoras
    delta_xy_abs = (np.sqrt(((x - xx) * resolution) ** 2 + ((y - yy) * resolution) ** 2))  # 432x faster when only bresenham line is calculated
    # calculate a vertical height distance grid with respect to this point
    delta_z_abs = onedem_array - onedem_array[y, x]  # double as slow when bresenham is taken
    # calc angle = arctan(height distance/horizontal distance) for each cell
    angle_grid = np.arctan(delta_z_abs / delta_xy_abs)  # 1000x faster ?
    angle_grid_deg = np.rad2deg(angle_grid)  # 1000x faster
    return angle_grid, angle_grid_deg

#@jit(nopython=True)
def ipot_loop(mask, resolution, onedem_array, azis, azis_rad, altis, altis_rad, times):

    x_coords = range(onedem_array.shape[1])
    y_coords = range(onedem_array.shape[0])
    xx, yy = np.meshgrid(x_coords, y_coords)

    time_array = np.empty((len(x_coords), len(y_coords), len(times)))
    time_array.fill(np.nan)

    # only go over valid glacier cells
    valid = np.where(mask)

    max_x = max(x_coords)
    max_y = max(y_coords)

    npixel = 0
    for y, x in zip(*valid):
        if npixel % 100 == 0:
            print('{0:.2f}% done.'.format(npixel/np.count_nonzero(mask)*100))
        npixel += 1
        print(npixel)
        ## subtract x, y indices of point to consider from xx and yy
        ## calculate horizontal distance grid from point with help of Pythagoras
        #delta_xy_abs = (np.sqrt(((x - xx) * resolution)**2 +
        #                        ((y - yy) * resolution)**2))  # 432x faster when only bresenham line is calculated
        ## calculate a vertical height distance grid with respect to this point
        #delta_z_abs = onedem_array - onedem_array[y, x]  # double as slow when bresenham is taken
        ## calc angle = arctan(height distance/horizontal distance) for each cell
        #angle_grid = np.arctan(delta_z_abs/delta_xy_abs)   # 1000x faster ?
        #angle_grid_deg = np.rad2deg(angle_grid)  # 1000x faster
        angle_grid, angle_grid_deg = make_angle_grid_numba(x, y, xx, yy, resolution, onedem_array)

        tindex = 0
        t0 = datetime.datetime.now()
        for azi, azi_rad, alti, t in zip(azis, azis_rad, altis, times):

            # select cells that intersect with azimuth angle (Bresenham line)
            # one index of the edge where the line hits is always a coordinate max
            # okay, it depends on the quadrant as well!!!
            """
            if azi < 90. or azi > 270.:
                x_intersect = int(np.abs(np.tan(azi_rad) * 0.5 * max_y))
                y_intersect = max_y
            elif 180. < azi < 270.:
                x_intersect = int(np.abs(np.tan(azi_rad)) * 0.5 * - max_y)
                y_intersect = - max_y
            elif azi == 90.:
                x_intersect = int(max_x / 2.)
                y_intersect = 0.
            else:
                x_intersect = -int(max_x / 2.)
                y_intersect = 0.
    
            if x > max_x / 2.:
                x_add = x - max_x / 2.
            else:
                x_add = -(x - max_x / 2.)
            if y > max_y / 2.:
                y_add = y - max_y / 2.
            else:
                y_add = -(y - max_y / 2.)
    
            # move the found intersections so that tey are relative to the actual position on the array
            x_intersect += x_add
            y_intersect += y_add
            """

            dx = radius * math.sin(azi_rad)
            dy = radius * math.cos(azi_rad)
            end_x = int(x + dx)
            end_y = int(y - dy)  # 5.52µs

            # todo: this shoudl be possible outside the loop?
            angle_grid_select = np.array(list(bresenham(x, y, end_x, end_y)))  # 1.03 ms
            # delete indices outside the DEM
            #angle_grid_select2 = [p for p in angle_grid_select if
            #                      ((p[0] in x_coords) and (p[1] in y_coords))]
            angle_grid_select2 = angle_grid_select[
                np.where(angle_grid_select[:, 0] <= max_x) and np.where(
                    angle_grid_select[:, 1] <= max_y)]  # 14.2 µs
            # todo: delete first Bresenham point (the point itself)?
            #angle_grid_select2 = angle_grid_select2[1:]

            #indextuple = tuple(np.array(angle_grid_select2).T)
            #if (angle_grid_deg[(indextuple[1], indextuple[0])] > alti).any():
            if (angle_grid_deg[angle_grid_select2[:, 1],
                               angle_grid_select2[:, 0]] > alti).any():  # 13.2 µs
                # shadow
                time_array[x, y, tindex] = 0.  # 233ns
            else:
                # sun - uncorrected for terrain
                time_array[x, y, tindex] = alt_azi.loc[t, 'ipot']  # 9.52 µs
            tindex += 1

    # correct Ipot for terrain
    sy, sx = np.gradient(onedem_array, resolution)
    aspect = np.arctan2(-sx, sy)
    slope = np.arctan(np.sqrt(sx ** 2 + sy ** 2))

    time_array_corrected = np.empty((len(x_coords), len(y_coords), len(times)))
    time_array_corrected.fill(np.nan)
    for i in np.arange(time_array.shape[2]):
        to_correct = time_array[:, :, i]
        corrected = correct_for_terrain(to_correct.T, 0., slope, aspect,
                                        np.pi / 4. - altis_rad[i], azis_rad[i])
        time_array_corrected[:, :, i] = corrected.T

    return time_array_corrected


def correct_for_terrain(r_beam, r_diff, slope, terrain_a, sun_z, sun_a):
    """
    Correct radiation on a horizontal surface for terrain slope and azimuth.

    The equation is implemented from [1]_, but but sun and terrain azimuth
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
    terrain_azi: float or array-like
        Terrain azimuth in degrees from north.
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


def test_correct_for_terrain():
    # test floats
    float_result = correct_for_terrain(1000., 0., 0., 0., np.pi/2., np.pi)



if __name__ == '__main__':

    from crampon import cfg
    from matplotlib.animation import ArtistAnimation
    cfg.initialize('c:\\users\\johannes\\documents\\crampon\\sandbox\\CH_params.cfg')
    # dems = xr.open_dataset('c:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\RGI50-11\\RGI50-11.B4\\RGI50-11.B4504\\homo_dem_ts.nc')
    # onedem = dems.sel(time='2015-08-30')

    t1 = datetime.datetime.now()

    gdir = utils.GlacierDirectory('RGI50-11.B4312n-1', base_dir='c:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier')

    onedem = xr.open_rasterio(gdir.get_filepath('dem'))
    with netCDF4.Dataset(
            gdir.get_filepath('gridded_data'),
            'r') as ncd:
        mask = ncd.variables['glacier_mask'][:]
    test_ds = onedem.to_dataset(name='data')
    test_ds.attrs['pyproj_srs'] = onedem.crs
    grid = salem.grid_from_dataset(test_ds)
    #onedem_array = onedem.values

    latitude_deg = np.mean(grid.ll_coordinates[1])  # positive in the northern hemisphere
    longitude_deg = np.mean(grid.ll_coordinates[0])  # negative reckoning west from Greenwich

    # make an extended grid to search for sun-blocking terrain
    extend_border = 10000  # meters
    ext_grid = salem.Grid(proj=grid.proj, dxdy=(grid.dx, grid.dy),
                          x0y0=(grid.x0 - extend_border, grid.y0 +
                                extend_border),
                          nxny=(int(grid.nx + 2 * extend_border / grid.dx),
                                int(grid.ny + 2 * extend_border / abs(grid.dy))))
    # regrid to 50m
    ext_regrid = ext_grid.regrid(factor=ext_grid.dx/50.)

    lon_ex = (ext_grid.extent_in_crs()[0], ext_grid.extent_in_crs()[1])
    lat_ex = (ext_grid.extent_in_crs()[2], ext_grid.extent_in_crs()[3])
    dems_ext_list, _ = utils.get_topo_file(lon_ex=lon_ex, lat_ex=lat_ex)

    if len(dems_ext_list) == 1:
        dem_dss = [rasterio.open(dems_ext_list[0])]  # if one tile, just open it
        dem_data = rasterio.band(dem_dss[0], 1)
        src_transform = dem_dss[0].transform
    else:
        dem_dss = [rasterio.open(s) for s in dems_ext_list]  # list of rasters
        dem_data, src_transform = merge_tool(dem_dss)  # merged rasters

    # Use Grid properties to create a transform (see rasterio cookbook)
    dst_transform = rasterio.transform.from_origin(
        ext_regrid.x0, ext_regrid.y0, ext_regrid.dx, ext_regrid.dx  # sign change (2nd dx) is done by rasterio.transform
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
    dst_array = np.empty((ext_regrid.ny, ext_regrid.nx), dtype=dem_dss[0].dtypes[0])
    reproject(
        # Source parameters
        source=dem_data,
        src_crs=dem_dss[0].crs,
        src_transform=src_transform,
        # Destination parameters
        destination=dst_array,
        dst_transform=dst_transform,
        dst_crs=ext_grid.proj.srs,
        # Configuration
        resampling=resampling)
    for dem_ds in dem_dss:
        dem_ds.close()

    ds = ext_regrid.to_dataset()
    ds['height'] = (['y', 'x'], dst_array)
    new_mask = ds.salem.roi(shape=gdir.get_filepath('outlines'))
    yearspan = pd.date_range('2018-07-01 00:00:00', '2018-07-01 24:00:00', tz='UTC',
                             freq='1H')  # freq='10min')
    alt_azi = pd.DataFrame(index=yearspan,
                           columns=['altitude_deg', 'azimuth_deg'])

    for m in yearspan:
        # todo: deal with time zone info :tzinfo=datetime.timezone.utc)

        # skip if sun is set anyway
        solar_altitude = get_altitude(latitude_deg, longitude_deg, m)
        if solar_altitude <= 0.:
            continue

        alt_azi.loc[m, 'altitude_deg'] = solar_altitude
        alt_azi.loc[m, 'azimuth_deg'] = get_azimuth(latitude_deg,
                                                    longitude_deg, m)
        alt_azi.loc[m, 'altitude_rad'] = np.deg2rad(
            alt_azi.loc[m, 'altitude_deg'])
        alt_azi.loc[m, 'azimuth_rad'] = np.deg2rad(
            alt_azi.loc[m, 'azimuth_deg'])
        alt_azi.loc[m, 'ipot'] = radiation.get_radiation_direct(m,
                                                                alt_azi.loc[m,
                                                                            'altitude_deg'])

    azis = alt_azi.azimuth_deg.values.astype(float)
    azis_rad = alt_azi.azimuth_rad.values.astype(float)
    altis = alt_azi.altitude_deg.values.astype(float)
    altis_rad = alt_azi.altitude_rad.values.astype(float)
    times = [pd.Timestamp(t, tz='UTC') for t in alt_azi.index.values]

    radius = max(dst_array.shape)

    ipot_array = ipot_loop(mask=~np.isnan(new_mask).height.values,
                           resolution=ext_regrid.dx,#resolution=np.abs(onedem.transform[0]),
              onedem_array=dst_array, azis=azis, azis_rad=azis_rad,
              altis=altis, altis_rad=altis_rad, times=times)

    # make daily mean
    group_indices = alt_azi.groupby(by=alt_azi.index.dt.date).indices

    ipot_array_daymean = np.array((ipot_array.shape[0],
                                  ipot_array.shape[1], 366))
    nth_day = 0
    for gi in group_indices:
        one_day = ipot_array[:, :, gi]
        one_day_mean = np.mean(one_day, axis=2)
        ipot_array_daymean[:, :, nth_day] = one_day_mean
        nth_day += 1

    grid_ds = ext_regrid.to_dataset()
    grid_ds = grid_ds.expand_dims('time')
    grid_ds.time = np.arange(366)
    ds['ipot'] = (['y', 'x', 't'], ipot_array_daymean)

    # distribute back to original grid

    # assign value to flowline



    t2 = datetime.datetime.now()


    # make an animated plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ims = []
    max_val = np.nanmax(ipot_array)
    min_val = np.nanmin(ipot_array)
    for i in range(ipot_array.shape[2]):
        im = ax.imshow(ipot_array[:, :, i], aspect='auto',
                       cmap='gist_ncar', vmin=min_val, vmax=max_val,
                       animated=True)
        t = ax.annotate(alt_azi.iloc[i].name.strftime('%m-%d'), (2, 50), color='k')
        if i == 0:
            fig.colorbar(im)
        ims.append([im, t])
    ani = ArtistAnimation(fig, ims, interval=50, blit=True,
                          repeat_delay=1000)
    print('it took', t2-t1)
    print('hi')