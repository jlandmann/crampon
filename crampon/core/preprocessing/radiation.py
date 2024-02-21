import numpy as np
import math
import xarray as xr
from pysolar.solar import *
from pysolar import radiation
import pandas as pd
import rasterio
import netCDF4
from numba import jit
import salem
from crampon import utils
from rasterio.warp import reproject, Resampling
from rasterio.merge import merge as merge_tool
from scipy.interpolate import interp1d
from crampon import cfg
from crampon.utils import entity_task
import logging
import pyproj

# Module logger
log = logging.getLogger(__name__)


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

    d = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        if d >= 0:
            y += 1
            d -= 2*dx
        d += 2 * dy


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


def barometric_pressure(z):
    """
     Get the barometric pressure as a function of height.

    Parameters
    ----------
    z : float or array_like
        Height(s) for which to get the barometric pressure.

    Returns
    -------
    p_z: same as input
        Pressure at height(s) z, given in Pascal!
    """
    # todo: check: if we take the humid temperature lapse rate, should we
    #  also take the molar mass of humid air?

    t_lapse_rate = cfg.PARAMS['temp_default_gradient']
    t_standard = cfg.ZERO_DEG_KELVIN + 15.
    p_z = cfg.SEALEVEL_PRESSURE * (1 - ((t_lapse_rate * z) / t_standard)) ** (
                (cfg.G * cfg.MOLAR_MASS_DRY_AIR) / (cfg.R * t_lapse_rate))
    return p_z


@jit(nopython=True)
def ratio_mean_to_current_sun_earth_distance(doy):
    """
    Give an approximated ratio of the mean to the current sun-earth distance.

    Approximated from ([1]_ in [2]_.

    Parameters
    ----------
    doy : int or array_like
        Days of year for which to get

    Returns
    -------
    d0_d: same as input
         Ratio(s) of the mean to the current sun-earth distance.

    References
    ----------
    .. [1] : Iqbal, M.: An introduction to solar radiation, Academic Press, New
             York, 1983.
    .. [2] : https://bit.ly/2sxT3hE
    """
    # ratio of mean & current sun-earth distance
    d0_d = (1 + 0.033 * np.cos(((2. * np.pi * doy) / 365.)))
    return d0_d


def get_terrain_slope_from_array(z, resolution):
    """
    Get the grid slope (radian) from an array of grid cell elevations.

    Parameters
    ----------
    z: array_like
        Array with grid elevations (m).
    resolution: float
        Grid resolution (m)

    Returns
    -------
    slope: same as z
        Array with slope entries (radian).
    """
    sx, sy = np.gradient(z, resolution)
    slope = np.arctan(np.sqrt(sx ** 2 + sy ** 2))
    return slope


def get_terrain_azimuth_from_array(z, resolution):
    """
    Get the grid slope (radian) from an array of grid cell elevations.

    Parameters
    ----------
    z: array_like
        Array with grid elevations (m).
    resolution: float
        Grid resolution (m)

    Returns
    -------
    slope: same as z
        Array with slope entries (radian).
    """
    sx, sy = np.gradient(z, resolution)
    aspect = np.arctan2(-sy, sx)

    # make it full circle, going counter-clockwise from north
    aspect[aspect < 0.] += 2 * np.pi

    return aspect


def get_incidence_angle_garnier(terrain_slope, terrain_azi, sun_zen, sun_azi):
    """
    Get angle of incidence between the normal to the grid slope & solar beam.

    This implements the equation from [1]_ and used in [2]_. However, we do not
    get the cosine of the incidence angel, but the angle directly.

    Parameters
    ----------
    terrain_slope: float or array_like
        Terrain slope angle (radian).
    terrain_azi: float or array_like
        Terrain azimuth angle (radian).
    sun_zen: float or array_like
        Sun zenith angle (radian).
    sun_azi: float or array_like
        Sun azimuth angle (radian).

    Returns
    -------
    theta: same as input
        Angle(s) of incidence for the given parameters.

    References
    ----------
    [1].. Garnier, B. J., & Ohmura, A. (1968). A method of calculating the
        direct shortwave radiation income of slopes. Journal of Applied
        Meteorology, 7(5), 796-800.
    [2].. Hock, R. (1999). A distributed temperature-index ice-and snowmelt
        model including potential direct solar radiation. Journal of
        Glaciology, 45(149), 101-111.
    """
    theta = np.arccos(np.cos(terrain_slope) * np.cos(sun_zen) +
                      np.sin(terrain_slope) * np.sin(sun_zen) *
                      np.cos(sun_azi - terrain_azi))
    return theta


def get_ipot_hock(doy, z, terrain_slope, terrain_azi, sun_zen, sun_azi,
                  clearsky_transmiss=0.75):
    """
    Model the potential irradiation as in  [1]_.

    Parameters
    ----------
    doy: int
        Day of year for which Ipot shall be calculated.
    z: float or array_like
        Grid height for which to get potential irradiation (m).
    terrain_slope: float or array_like
        Slope angle of terrain for which to get potential irradiation (radian).
    terrain_azi: float or array_like
        Azimuth of terrain for which to get potential irradiation (radian).
    sun_zen: float or array_like
        Sun zenith angle (radians). Must have the same shape as
        `terrain_slope`/`terrain_azi`.
    sun_azi : float or array_like
        Sun azimuth angle (radians). Must have the same shape as
        `terrain_slope`/`terrain_azi`.
    clearsky_transmiss: float or array_like
        Clear-sky transmissivity. Just as in [1]_, we assume 0.75 as a default
        ([2]_), which lies within the range 0.6 0.9 reported in other studies
        [3]_. Clear-sky transmissivity can actually vary in space an time.
        Default: 0.75 (spatially and temporally constant).

    Returns
    -------
    ipot: array_like
        Array with potential irradiation.

    References
    ----------
    [1].. Hock, R. (1999). A distributed temperature-index ice-and snowmelt
        model including potential direct solar radiation. Journal of
        Glaciology, 45(149), 101-111.
    [2].. Hock, R. (1998). Modelling of glacier melt and discharge (Doctoral
        dissertation, ETH Zurich).
    [3].. Oke, T. R. (1987). Boundary layer climates. Routledge.
    """

    p = barometric_pressure(z)
    d0_d = ratio_mean_to_current_sun_earth_distance(doy)
    theta = get_incidence_angle_garnier(terrain_slope, terrain_azi, sun_zen,
                                        sun_azi)
    ipot = cfg.SOLAR_CONSTANT * (d0_d ** 2.) * clearsky_transmiss ** (
                p / (cfg.SEALEVEL_PRESSURE * np.cos(sun_zen))) * np.cos(theta)

    # set negative/inf values to zero
    # todo: rather check if sun is below horizon
    ipot[ipot < 0.] = 0.
    ipot[np.isinf(ipot)] = 0.

    return ipot


def get_potential_irradiation_without_toposhade(lat_deg, lon_deg, tz='UTC',
                                                freq='10min', t1=None,
                                                t2=None):
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
    t1: pd.Timestamp or None
        Begin date/time, if Ipot shall not be calculated for a whole year.
        Default: None (whole year).
    t2: pd.Timestamp or None
        End date/time, if Ipot shall not be calculated for a whole year.
        Default: None (whole year).

    Returns
    -------
    alt_azi: pd.Dataframe
        Dataframe with solar azimuth and altitude (in radians!) as well as the
        potential irradiation at given time steps.
    """

    # time zone doesn't matter as we only take daily means
    # take a dummie year
    if t1 is None:
        t1 = pd.Timestamp('2018-01-01 00:00:00')
    if t2 is None:
        t2 = pd.Timestamp('2019-01-01 00:00:00') - \
             pd.tseries.frequencies.to_offset(freq)

    timespan = pd.date_range(t1, t2, tz=tz, freq=freq)
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


def ipot_loop(mask, resolution, dem_array, alt_azi,
              freqstr):
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
    freqstr: str
        Interval between the calculations of Ipot (e.g. 10min).

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

    # todo 1): make a grid of the glacier with buffer,
    #  2) map indices of this grid to the bigger grid

    # end of day indices
    eod_ix = \
        np.cumsum([len(x) for x in
                   alt_azi.index.groupby(alt_azi.index.date).values()]) - 1

    # prepare the output array
    time_array = np.empty((len(x_coords), len(y_coords), len(times)))
    time_array.fill(np.nan)

    n_times_per_day = int(pd.Timedelta(days=1) / pd.to_timedelta(
                    pd.tseries.frequencies.to_offset(freqstr)))
    small_array = np.full(n_times_per_day, np.nan)

    # save time: only go over valid glacier cells
    valid = np.where(mask)

    sy, sx = np.gradient(dem_array, resolution)
    slope = np.arctan(np.sqrt(sx ** 2 + sy ** 2))
    aspect = np.arctan2(-sx, sy)

    npixel = 0
    insert_at = 0
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
        it = 0
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
                # small_array[it] = 0.
            else:
                # sun - uncorrected for terrain
                time_array[x, y, tindex] = alt_azi.loc[t, 'ipot']
                # rad_corr = correct_radiation_for_terrain(
                # alt_azi.loc[t, 'ipot'], 0., slope[y, x], aspect[y, x],
                # np.pi / 2. - altis_rad[tindex], azis_rad[tindex])
                # clip zero: diff. rad is set to 0 (direct could be negative!)
                # rad_corr = np.clip(rad_corr, 0., None)
                # small_array[it] = alt_azi.loc[t, 'ipot']
                # small_array[it] = rad_corr

            # if tindex in eod_ix:  # switch of day
            #    insert_at, = np.where(eod_ix == tindex)
            #    time_array[x, y, insert_at] = np.nansum(small_array) /
            #    n_times_per_day
            #    small_array = np.full(144, np.nan)
            #    it = 0
            # else:
            #    it += 1

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
        corrected[corrected < 0.] = 0.  # prob. faster than np.clip(arr. large)
        time_array_corrected[:, :, i] = corrected.T

    return time_array_corrected


@jit(nopython=True)
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


@entity_task(writes=['sis_scale_factor'])
def scale_irradiation_with_potential_irradiation(gdir, sis=None,
                                                 diff_rad_ratio=0.2):
    """
    Scale actual measured incoming solar radiation with potential irradiation.

    This is used to apply terrain effects to the MeteoSwiss HelioSat global
    irradiation product.

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        GlacierDirectory to scale Ipot for.
    sis: float, optional
        Mean irradiation over a time span, e.g. one day.
    diff_rad_ratio: float
        Ratio of diffuse to total radiation. We can only guess here,
        until we include also SISDIR. Default: 0.2 (20% of SIS is diffuse
        radiation).

    Returns
    -------
    sis_distr: np.array
        The solar incoming radiation distributed on terrain.
    """
    cd = xr.open_dataset(gdir.get_filepath('climate_daily'))
    ipot = xr.open_dataset(gdir.get_filepath('ipot'))
    ippf = gdir.read_pickle('ipot_per_flowline')
    ippf = np.array([i for sub in ippf for i in sub])

    p = pyproj.Proj(init='epsg:4326')
    # resulting projection, WGS84, long, lat

    lon, lat = [cd.ref_pix_lon, cd.ref_pix_lat]
    cenx, ceny = pyproj.transform(p, gdir.grid.proj, lon, lat)
    # cell size is approx. 1.6km x 2.3 km
    # todo: this can be made more precisely - cell size varies throughout CH
    ipot_sel = ipot.where((ipot.x >= cenx - 800.) & (ipot.x >= cenx + 800) & (
                ipot.y >= ceny - 1150.) & (ipot.y <= ceny + 1150.))
    ipot_cell_mean = ipot_sel.mean(dim=['x', 'y'])

    # todo: get the diffuse ratio from SIS/SISDIR
    sis_corr_fac = ippf / np.atleast_2d(ipot_cell_mean.ipot.values)
    # if ippf is zero, we have a problem. Everywhere should be diffuse rad.
    sis_corr_fac = np.clip(sis_corr_fac, diff_rad_ratio, None)

    da = xr.DataArray(sis_corr_fac, dims={
        'fl_id': (['fl_id'], np.arange(sis_corr_fac.shape[0])),
        'doy': (['doy'], np.arange(sis_corr_fac.shape[1]))},
                      coords={'fl_id': (['fl_id'],
                                        np.arange(sis_corr_fac.shape[0])),
                              'doy': (['doy'],
                                      np.arange(sis_corr_fac.shape[1]))},
                      name='sis_scale_fac')

    da.to_netcdf(gdir.get_filepath('sis_scale_factor'))

    """
    # todo: this assumes that ISIS is the mean! To make this happen,
    #  we actually need to know which Ipot cells the SIS cell covers. Then
    #  we don't take the mean of Ipot in general, but the mean over this area.
    # Ipot as a factor of the mean
    ipot_fac = ipot.ipot / ipot.ipot.mean(dim=['x', 'y'])
    ipot_fac = np.clip(ipot_fac, diff_rad_ratio, None)

    # todo: this does not include diffuse radiation: if at a time step
    #  during the day Ipot was 0, then this will also be counted as zero for
    #  ISIS. This is not true, however: It should be counted as 20% of the
    #  total (diffuse part) or so.
    # Apply normalization to incoming solar radiation
    isis_distr = sis * ipot_fac

    return isis_distr
    """


# For whatever reason this 'HAS TO'
# @entity_task(log, writes=['ipot_per_flowline'])
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
            # todo: check if surface surface_h values are lower edge, uppder
            #  edge or mid of the interval
            try:
                cond = (c_heights.sel(band=1).data.values <= fhgt[q]) & (
                        c_heights.sel(band=1).data.values > fhgt[q + 1])
            except IndexError:
                # trick at the end of the flowline
                cond = (c_heights.sel(band=1).data.values <= fhgt[q])
            if not cond.any():
                continue
            ts_per_hgt = ipot_subset.where(cond)\
                .mean(dim=['x', 'y'], skipna=True).to_array().values.flatten()
            fl_ipot_arr[q, :] = ts_per_hgt

        nnan_ix = np.unique(np.where(~np.isnan(fl_ipot_arr))[0])
        fl_ipot_interp = interp1d(np.arange(len(fhgt))[nnan_ix],
                                  fl_ipot_arr[nnan_ix, :], axis=0,
                                  fill_value='extrapolate')(range(len(fhgt)))

        ipot_per_flowline.append(fl_ipot_interp)

    gdir.write_pickle(ipot_per_flowline, 'ipot_per_flowline')


# For whatever reason (OGM complains) this has to stay commented out
# @entity_task(log, writes=['ipot'])
def get_potential_irradiation_with_toposhade(gdir, dem_source='SRTM',
                                             grid_reduce_fac=None):
    """
    Get the potential solar irradiation including topographic shading.

    Parameters
    ----------
    gdir: `py_class:crampon.GlacierDirectory`
        The GlacierDirectory to process the potential solar irradiation for.
    dem_source: str or None
        The source of the DEM that is use for raytracing. Can be any source
        that is accepted by util.get_topo_file. If None and a "dem_file" is
        provided in the params file, then this DEM will be taken. This can,
        however, cause problem  e.g. at Swiss borders as the domain is extended
        by 10km.
        # todo: merge SRTM and own source to get the best estimate or make a
        query if whole domain os covered by dem_file -> else, take SRTM
    grid_reduce_fac: float
        Factor that determines how coarsely the grid will be resampled (avoid
        MemoryError). Default: None (values form cfg.PARAMS is taken).

    Returns
    -------
    None
    """

    dem = xr.open_rasterio(gdir.get_filepath('dem'))
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
    if grid_reduce_fac is None:
        grid_reduce_fac = cfg.PARAMS['reduce_rgrid_resolution']
    ext_regrid = ext_grid.regrid(factor=grid_reduce_fac)
    print('new_resolution: {}'.format(ext_regrid.dx))

    # todo: get rid of double code from init_glacier_regions
    lon_ex = (ext_grid.extent_in_crs()[0], ext_grid.extent_in_crs()[1])
    lat_ex = (ext_grid.extent_in_crs()[2], ext_grid.extent_in_crs()[3])
    dems_ext_list, _ = utils.get_topo_file(lon_ex=lon_ex, lat_ex=lat_ex,
                                           source=dem_source)

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
    # todo: make ipot calculation at finer grids possible
    if np.isnan(new_mask.height.values).all():  # tiny glac. (< new grid size)
        new_mask = regrid_ds.salem.roi(shape=gdir.get_filepath('outlines'),
                                       all_touched=True)

    # assume: grid is small enough that difference in sun angles doesn't matter
    latitude_deg = np.mean(grid.ll_coordinates[1])  # pos. in the northern h.
    longitude_deg = np.mean(grid.ll_coordinates[0])  # neg. west from 0 deg
    freqstr = '10min'
    alt_azi = get_potential_irradiation_without_toposhade(latitude_deg,
                                                          longitude_deg,
                                                          freq=freqstr)

    ipot_array = ipot_loop(mask=~np.isnan(new_mask).height.values,
                           resolution=ext_regrid.dx, dem_array=dst_array,
                           alt_azi=alt_azi, src_grid=ext_regrid, dst_grid=grid,
                           freqstr=freqstr)

    # make daily mean
    group_indices = alt_azi.groupby(by=alt_azi.index.date).indices
    ipot_array_daymean = np.empty((ipot_array.shape[0],
                                  ipot_array.shape[1], len(group_indices)))

    for nth_day, day in enumerate(np.unique(alt_azi.index.date)):

        one_day = ipot_array[:, :, group_indices[pd.Timestamp(day)]]
        # we have to account for time steps during the night
        one_day_mean = np.sum(one_day, axis=2) / \
            (pd.Timedelta(days=1) /
             pd.to_timedelta(pd.tseries.frequencies.to_offset(freqstr)))
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
    ipot_reproj.ipot.encoding.update({'dtype': 'int16', 'scale_factor': 0.01,
                                      '_FillValue': -9999})
    ipot_reproj.to_netcdf(gdir.get_filepath('ipot'))


def irradiation_top_of_atmosphere(doy, latitude_deg, solar_constant=None):
    """
    Calculate irradiation at the top of the atmosphere.

    The calculation depends on the position (latitude) of the earth, day of
    year and the solar constant. The output value is a daily mean. Equations
    from [1]_ as presented in [2]_ and [3]_.

    Parameters
    ----------
    doy: array
        Day(s) of year for which to get the TOA irradiation.
    latitude_deg: array
        Latitude(s) for which to get the TOA irradiation.
    solar_constant: float or None
        The solar constant. If None, then value from cfg is taken. Default:
        None.

    Returns
    -------
    s_toa: array
        Solar irradiation at the top of atmosphere for the given parameters.

    References
    ----------
    .. [1] : Iqbal, M.: An introduction to solar radiation, Academic Press, New
             York, 1983.
    .. [2] : https://bit.ly/2sxT3hE
    .. [3] : https://bit.ly/2Si0KDD
    """

    if solar_constant is None:
        solar_constant = cfg.SOLAR_CONSTANT

    # extend shapes and make radians
    doy = np.atleast_2d(doy).T
    latitude_rad = np.atleast_2d((np.deg2rad(latitude_deg)))

    # declination  of the sun (approximated as in [2]_.)
    delta = 0.4093 * np.sin(((2. * np.pi) / 365. * doy) - 1.3944)

    # Approx. ratio of mean & current sun-earth distance
    d0d = ratio_mean_to_current_sun_earth_distance(doy)

    # cosine of sun hour angle
    cos_omega = - np.tan(latitude_rad) * np.tan(delta)
    # limit cosine of omega to +- 1, so that there can be polar day and night
    cos_omega = np.clip(cos_omega, -1., 1.)
    omega = np.arccos(cos_omega)

    # putting all together
    s_toa = solar_constant * d0d * 1 / np.pi * (
        omega * np.sin(latitude_rad) * np.sin(delta) + np.cos(latitude_rad)
        * np.cos(delta) * np.sin(omega))

    return s_toa


def get_declination(doy):
    """
    Get the solar declination.

    Parameters
    ----------
    doy : int
        Day of year.

    Returns
    -------
    decl: int
        Declination for the given day of year.
    """
    tt = 2 * math.pi * doy / 366
    decl = (0.322003 -
            22.971 * math.cos(tt) -
            0.357898 * math.cos(2 * tt) -
            0.14398 * math.cos(3 * tt) +
            3.94638 * math.sin(tt) +
            0.019334 * math.sin(2 * tt) +
            0.05928 * math.sin(3 * tt))  # solar declination in degrees
    return decl


def get_declination_corripio(jd):
    """
    Get the solar declination according to Corripio (2003).

    Parameters
    ----------
    jd : int or np.array
        Julian day.

    Returns
    -------
    same as jd:
        Declination angle in degrees.
    """
    jdc = (jd - 2451545.0) / 36525.0
    sec = 21.448 - jdc * (46.8150 + jdc * (0.00059 - jdc * 0.001813))
    e0 = 23.0 + (26.0 + (sec / 60.0)) / 60.0
    oblcorr = e0 + 0.00256 * np.cos(np.deg2rad(125.04 - 1934.136 * jdc))
    l0 = 280.46646 + jdc * (36000.76983 + jdc * 0.0003032)
    l0 = (l0 - 360 * (int(l0 / 360))) % 360
    gmas = 357.52911 + jdc * (35999.05029 - 0.0001537 * jdc)
    gmas = np.deg2rad(gmas)
    seqcent = np.sin(gmas) * (1.914602 - jdc * (0.004817 + 0.000014 * jdc)) + \
        np.sin(2 * gmas) * (0.019993 - 0.000101 * jdc) + np.sin(3 * gmas) \
        * 0.000289

    suntl = l0 + seqcent
    sal = suntl - 0.00569 - 0.00478 * np.sin(np.deg2rad(125.04 - 1934.136 *
                                                        jdc))
    delta = np.arcsin(np.sin(np.deg2rad(oblcorr)) * np.sin(np.deg2rad(sal)))
    return np.rad2deg(delta)


def sunvector_corripio(latitude, longitude, time):
    """
    Get the unit vector (x, y, z) in direction of the sun.

    This is modified after [1]_.

    Parameters
    ----------
    latitude: float
        Latitude in degrees.
    longitude: float
        Longitude in degrees.
    time: pd.Timestamp, time zone aware
        A time zone aware time stamp.

    Returns
    -------
    (3,) np.array
        Unit vector pointing towards the sun.

    References
    ----------
    [1].. Corripio, J. G.: Vectorial algebra algorithms for calculating terrain
          parameters from DEMs and solar radiation modelling in mountainous
          terrain. International Journal of Geographical Information Science,
          Taylor & Francis, 2003, 17, 1-23.
    """
    omegar = np.deg2rad(get_hour_angle(time, longitude))
    deltar = np.deg2rad(get_declination(time.dayofyear))
    lambdar = np.deg2rad(latitude)

    svx = -np.sin(omegar) * np.cos(deltar)
    svy = np.sin(lambdar) * np.cos(omegar) * np.cos(deltar) - np.cos(
        lambdar) * np.sin(deltar)
    svz = np.cos(lambdar) * np.cos(omegar) * np.cos(deltar) + np.sin(
        lambdar) * np.sin(deltar)
    return np.array([svx, svy, svz])


@jit(nopython=True)
def get_shade_corripio(dem, sunvector, cols, rows, dl):
    """
    Get the terrain affected by show after Corripio (2003).

    This is a reimplementation of the Fortran script used in Javier Corripio's
    R package 'insol' [1]_ and [2]_.
    # todo: check License
    # todo: eliminate some fortran leftovers (double code etc.)
    # todo: check stripes in images: artefact of plotting or bug?

    Parameters
    ----------
    dem: (N, M) np.ndarray
        DEM with corresponding grid cell elevations as a numpy array.
    sunvector: (3,) np.array
        Unit vector (x, y, z) in direction of the sun.
    cols: int
        Number of columns (element 0 of "shape") of DEM.
    rows: int
        Number of rows (element 1 of "shape") of DEM.
    dl: float
        DEM resolution (m).

    Returns
    -------
    sombra: (N, M) np.ndarray
        Array with 1s (direct beam sunlight) and 0s (cast shadow) for the given
        DEM and sun vector.

    References
    ----------
    [1].. https://cran.r-project.org/web/packages/insol/index.html
    [2].. Corripio, J. G.: Vectorial algebra algorithms for calculating terrain
          parameters from DEMs and solar radiation modelling in mountainous
          terrain. International Journal of Geographical Information Science,
          Taylor & Francis, 2003, 17, 1-23.
    """

    inversesunvector = - sunvector / np.max(np.abs(sunvector[:2]))

    normalsunvector = np.full_like(sunvector, np.nan)
    vectortoorigin = np.full_like(sunvector, np.nan)  # needed later
    normalsunvector[2] = np.sqrt(sunvector[0] ** 2 + sunvector[1] ** 2)
    normalsunvector[0] = - sunvector[0] * sunvector[2] / normalsunvector[2]
    normalsunvector[1] = - sunvector[1] * sunvector[2] / normalsunvector[2]

    newshape = (cols, rows)
    z = np.reshape(dem, newshape)

    # casx is int, this makes the value large enough to compare effectively
    casx = np.rint(1e6 * sunvector[0])
    casy = np.rint(1e6 * sunvector[1])

    # "x" entry sunvector negative -> sun is on the West: begin with grid cols
    if casx < 0.:
        f_i = 0  # fixed i
    else:
        f_i = cols - 1

    if casy < 0.:
        f_j = 0
    else:
        f_j = rows - 1

    # set shading to 1 as default (no shade)
    sombra = np.copy(dem)
    sombra[:] = 1.

    j = f_j
    for i in range(cols):
        n = 0

        # todo: this is probably stupid
        zcompare = - np.inf  # init value lower than any possible zprojection

        while True:
            dx = inversesunvector[0] * n
            dy = inversesunvector[1] * n

            idx = int(np.rint(i + dx))
            jdy = int(np.rint(j + dy))
            if (idx < 0) or (idx > cols-1) or (jdy < 0) or (jdy > rows-1):
                break
            vectortoorigin[0] = dx * dl
            vectortoorigin[1] = dy * dl
            vectortoorigin[2] = z[idx, jdy]

            zprojection = np.dot(vectortoorigin, normalsunvector)
            if zprojection < zcompare:
                sombra[idx, jdy] = 0
            else:
                zcompare = zprojection

            n += 1

    i = f_i
    for j in range(rows):
        n = 0

        # todo: this is probably stupid
        zcompare = - np.inf  # init value lower than any possible zprojection

        while True:
            dx = inversesunvector[0] * n
            dy = inversesunvector[1] * n
            idx = int(np.rint(i + dx))
            jdy = int(np.rint(j + dy))
            if (idx < 0) or (idx > cols-1) or (jdy < 0) or (jdy > rows-1):
                break
            vectortoorigin[0] = dx * dl
            vectortoorigin[1] = dy * dl
            vectortoorigin[2] = z[idx, jdy]
            zprojection = np.dot(vectortoorigin, normalsunvector)

            if zprojection < zcompare:
                sombra[idx, jdy] = 0
            else:
                zcompare = zprojection
            n += 1

    return sombra


def _fallback_ipot(gdir):
    print('Ipot could not be calculated for {} ({}).'.format(gdir.rgi_id,
                                                             gdir.name))


@entity_task(log, writes=['ipot'], fallback=_fallback_ipot)
def get_potential_irradiation_corripio(gdir):
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

    # todo: get rid of double code from init_glacier_regions
    lon_ex = (ext_grid.extent_in_crs()[0], ext_grid.extent_in_crs()[1])
    lat_ex = (ext_grid.extent_in_crs()[2], ext_grid.extent_in_crs()[3])

    # try and get and combination of SwissALTI3D and SRTM (for borders)
    # take the SWISSAlti3D
    src_list = ['USER', 'SRTM']
    dems_ext_list_sa, _ = utils.get_topo_file(lon_ex=lon_ex, lat_ex=lat_ex,
                                              source=src_list[0])
    dems_ext_list_sr, _ = utils.get_topo_file(lon_ex=lon_ex, lat_ex=lat_ex,
                                              source=src_list[1])

    potential_dem_dict = {}
    for src_name, potential_dem in zip(src_list,
                                       [dems_ext_list_sa, dems_ext_list_sr]):
        if len(potential_dem) == 1:
            dem_dss = [rasterio.open(potential_dem[0])]  # one tile, just open
            dem_data = rasterio.band(dem_dss[0], 1)
            src_transform = dem_dss[0].transform
        else:
            dem_dss = [rasterio.open(s) for s in
                       potential_dem]  # list of rasters
            dem_data, src_transform = merge_tool(dem_dss)  # merged rasters

        dst_transform = rasterio.transform.from_origin(
            ext_grid.x0, ext_grid.y0, ext_grid.dx, ext_grid.dx
        )

        # Set up profile for writing output
        profile = dem_dss[0].profile
        profile.update({
            'crs': ext_grid.proj.srs,
            'transform': dst_transform,
            'dtype': rasterio.float32,
            'width': ext_grid.nx,
            'height': ext_grid.ny
        })

        resampling = Resampling[cfg.PARAMS['topo_interp'].lower()]
        dst_array = np.empty((ext_grid.ny, ext_grid.nx),
                             dtype=rasterio.float32)
        reproject(
            source=dem_data,
            src_crs=dem_dss[0].crs,
            src_transform=src_transform,
            destination=dst_array,
            dst_transform=dst_transform,
            dst_nodata=np.nan,
            dst_crs=ext_grid.proj.srs,
            resampling=resampling)
        for dem_ds in dem_dss:
            dem_ds.close()

        # make a dataset and save in dict
        candidate = ext_grid.to_dataset()
        candidate['height'] = (['y', 'x'], dst_array)
        potential_dem_dict[src_name] = candidate

    # debias cheaply and combine
    srtm = potential_dem_dict['USER']
    alti = potential_dem_dict['SRTM']
    bias = np.nanmean(srtm.height.values - alti.height.values)
    print('BIAS (SRTM-SwissAlti3D): {} m'.format(bias))
    srtm += bias
    # combine with preference for SwissALTI3D
    regrid_ds = alti.combine_first(srtm)

    #  mask for assembling Ipot
    buf_pix = 5
    sub_mask = regrid_ds.salem.subset(shape=gdir.get_filepath('outlines'),
                                      margin=buf_pix)

    # get (i,j) coords of subgrid => slice arrays instead of datasets (faster)
    coords = ext_grid.transform(sub_mask.x.values, sub_mask.y.values,
                                crs=sub_mask.pyproj_srs, nearest=True)
    ax0_min_ix = min(coords[0])
    ax0_max_ix = max(coords[0])
    ax1_min_ix = min(coords[1])
    ax1_max_ix = max(coords[1])

    # ".T" important: let the sun come "from the right side" (defined by axis)
    dem_arr = dst_array.T

    tsteps = 52560
    tstep_minutes = 10
    freq = '{}min'.format(tstep_minutes)
    steps_per_day = ((24 * 60) / tstep_minutes)
    testsom_arr = np.empty((int(np.ceil(tsteps / steps_per_day)),
                            sub_mask.height.shape[1],
                            sub_mask.height.shape[0]))

    timezone = 'Europe/Berlin'
    trange = pd.date_range(pd.Timestamp('2019-01-01 00:00:00', tz=timezone),
                           periods=tsteps, freq=freq)

    dem_res = (sub_mask.x[1] - sub_mask.x[0]).item()
    sub_mask_heights = sub_mask.height.values
    slope = get_terrain_slope_from_array(sub_mask_heights, dem_res)
    aspect = get_terrain_azimuth_from_array(sub_mask_heights, dem_res)

    temp_init = np.zeros((int(steps_per_day), *sub_mask.height.values.T.shape))
    temp = temp_init.copy()
    day = trange[0].day
    time_ix = 0
    day_ix = 0
    trange_used = []
    trange_used.append(trange[0])
    for tnum, t in enumerate(trange):

        sun_z_deg = get_altitude(gdir.cenlat, gdir.cenlon, t)
        if sun_z_deg < 0.:  # sun is set
            continue

        sun_z = np.pi / 2. - np.deg2rad(
            np.clip(sun_z_deg, 0., 90.))
        sun_a = np.deg2rad(get_azimuth(gdir.cenlat, gdir.cenlon, t))

        sunvec = sunvector_corripio(gdir.cenlat, gdir.cenlon, t)
        rows = dem_arr.shape[1]
        cols = dem_arr.shape[0]
        shade = get_shade_corripio(dem_arr, sunvec, cols, rows, dem_res)

        # clip to the actual desired shape
        shade = shade[ax0_min_ix: ax0_max_ix + 1, ax1_min_ix: ax1_max_ix + 1]

        ipot_hock = get_ipot_hock(t.dayofyear, sub_mask_heights, slope,
                                  aspect, sun_z, sun_a)

        ipot_corr = shade * ipot_hock.T

        if (t.day == day) or (freq == '1440min'):
            temp[time_ix, :, :] = ipot_corr
            time_ix += 1

        if (t.day != day) or (freq == '1440min'):
            print('day_ix: ', day_ix)
            testsom_arr[day_ix, :, :] = (
                        np.nansum(temp, axis=0) / steps_per_day)
            temp = temp_init.copy()
            day_ix += 1
            time_ix = 0
            day = t.day
            trange_used.append(t)

    # important: attach the last ones
    if freq != '1440min':
        testsom_arr[day_ix, :, :] = (np.nansum(temp, axis=0) / steps_per_day)
    trange_used.append(t)

    ipot_dates = np.unique([d.date() for d in trange_used])
    ipot_dates_np64 = [np.datetime64(d) for d in ipot_dates]
    ipot_ds = sub_mask.salem.grid.to_dataset().assign_coords(
        time=ipot_dates_np64)
    # order of x, y, time must be exactly like this for salem transform to work
    ipot_ds['ipot'] = (['x', 'y', 'time'], np.moveaxis(testsom_arr, 0, 2))
    ipot_ds.ipot.attrs['pyproj_srs'] = sub_mask.pyproj_srs
    ipot_ds = ipot_ds.transpose()
    # distribute back to original grid and write
    ipot_reproj = ds.salem.transform(ipot_ds)
    # apply tricks to not consume endless disk space when writing
    ipot_reproj.to_netcdf(gdir.get_filepath('ipot'),
                          encoding={'ipot': {'dtype': np.uint16,
                                             'scale_factor': 0.01,
                                             '_FillValue': 9999}})


@entity_task(log, fallback=_fallback_ipot)
def calculate_and_distribute_ipot(gdir: utils.GlacierDirectory) -> None:
    """
    Shortcut function to get hae both creation and distribution in one.

    Parameters
    ----------
    gdir: `py_class:crampon.GlacierDirectory`
        The GlacierDirectory to process the potential solar irradiation for.

    Returns
    -------
    None
    """

    get_potential_irradiation_corripio(gdir)
    distribute_ipot_on_flowlines(gdir)
