from __future__ import division

from crampon.utils import entity_task, global_task
from crampon import utils
import logging
from matplotlib.ticker import NullFormatter
from oggm.graphics import *
import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.animation as animation
import matplotlib
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map
import numpy as np
from glob import glob
import folium
from folium import plugins
from folium.features import DivIcon
import branca
import mpld3
import shapely

# Local imports
import crampon.cfg as cfg

# Module logger
log = logging.getLogger(__name__)

nullfmt = NullFormatter()  # no labels


def plot_fog_mb_glaciers(fog_dir=None, y=None, x=None):
    """
    Plot available FoG MB glaciers in a certain area.
    
    Parameters
    ----------
    fog_dir: str
        Directory where the FoG subfiles are stored. Version 2016-08 should 
        work.
    y: list
        Latitude extent on which to be plotted.
    x: list
        Longitude extent on which to be plotted.

    Returns
    -------
    
    """
    fog_a = pd.read_csv(os.path.join(fog_dir,
                                     'WGMS-FoG-2016-08-A-GLACIER.csv'),
                        encoding='iso8859_15')
    fog_ee = pd.read_csv(os.path.join(fog_dir,
                                      'WGMS-FoG-2016-08-EE-MASS-BALANCE.csv'),
                         encoding='iso8859_15')

    ext_ids = fog_a[(fog_a.LONGITUDE >= np.nanmin(x)) &
                    (fog_a.LONGITUDE <= np.nanmax(x)) &
                    (fog_a.LATITUDE >= np.nanmin(y)) &
                    (fog_a.LATITUDE <= np.nanmax(y))].WGMS_ID.values

    # Select and create set of MB glaciers in extent
    fog_ee = fog_ee[fog_ee.WGMS_ID.isin(ext_ids)]
    # Where we have yearly measures and no altitude range
    fog_ee = fog_ee[fog_ee.LOWER_BOUND.isin([9999])]

    # Select back in FOG A what shall be plotted (inefficient!?)
    fog_a = fog_a[fog_a.WGMS_ID.isin(np.unique(fog_ee.WGMS_ID))]

    # prepare the figure
    f, ax = plt.subplots()

    g = GoogleVisibleMap(x=[np.nanmin(x), np.nanmax(x)],
                         y=[np.nanmin(y), np.nanmax(y)],
                         maptype='satellite')

    # the google static image is a standard rgb image
    ggl_img = g.get_vardata()
    sm = Map(g.grid, factor=1, countries=True)
    sm.set_rgb(ggl_img)  # add background rgb image
    sm.visualize(ax=ax)
    x, y = sm.grid.transform(fog_a.LONGITUDE.values, fog_a.LATITUDE.values)
    ax.scatter(x, y, color='orange', s=50, edgecolors='k', linewidths=1)
    for i, txt in enumerate(fog_a.NAME.values):
        ax.annotate(txt, (x[i], y[i]))
    ax.set_title('Available MB glacier in FoG')

    # make it nice
    plt.show()


def plot_entity_param_dist(gdirs):
    """
    Plot the distribution of entity parameters mustar, tstar.
    
    Parameters
    ----------
    gdirs

    Returns
    -------

    """

    df_comb = pd.DataFrame([])

    for gdir in gdirs:
        try:
            tdf = pd.read_csv(os.path.join(gdir, 'local_mustar.csv'))
        except OSError:
            continue

        if not df_comb.empty:
            df_comb = df_comb.append(tdf, ignore_index=True)
        else:
            df_comb = tdf.copy()

    fig = plt.figure()
    ax = plt.gca()
    ax.plot(df_comb['mu_star'], df_comb['prcp_fac'], 'o')
    ax.set_xscale('log')
    plt.show()

    return df_comb


def make_mb_popup_map(shp_loc='C:\\Users\\Johannes\\Desktop\\mauro_in_RGI_disguise_entities_old_and_new.shp',
                      ch_loc='C:\\Users\\Johannes\\Documents\\data\\borders\\swissBOUNDARIES3D\\BOUNDARIES_2017\\DATEN\V200\\SHAPEFILE_LV03\\VEC200_LANDESGEBIET_LV03.shp'):
    """
    Create a clickable map of all Swiss glacier outlines.

    Parameters
    ----------
    shp_loc: str
        Path to the glacier shapefile.
    ch_loc: str
        Path to the shapefile with Swiss borders.

    Returns
    -------

    """

    # A map zoomed on Switzerland
    m = folium.Map(location=[46.81, 8.21], tiles='cartoDB Positron',
                   zoom_start=9, control_scale=True)

    # Make a full screen option
    plugins.Fullscreen(position='topright', title='Full screen',
                       titleCancel='Exit full screen',
                       forceSeparateButton=True).add_to(m)

    # Add anywhere LatLon PopUp
    m.add_child(folium.LatLngPopup())

    # Add tile layers to choose from as background
    # ATTENTION: Some fail silently and the LayerControl just disappears
    # The last one added here is switched on by default
    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('http://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
                     attr='Map data: &copy; <a href="http://www.openstreetmap.'
                          'org/copyright">OpenStreetMap</a>, <a href="http://'
                          'viewfinderpanoramas.org">SRTM</a> | Map style: &'
                          'copy; <a href="https://opentopomap.org">OpenTopoMap'
                          '</a> (<a href="https://creativecommons.org/licenses'
                          '/by-sa/3.0/">CC-BY-SA</a>)',
                     name='OpenTopoMap').add_to(m)
    folium.TileLayer('stamenterrain', name='Stamen Terrain').add_to(m)
    # Copyright/usage restriction unclear:
    #folium.TileLayer('http://server.arcgisonline.com/ArcGIS/rest/services/'
    #                 'World_Imagery/MapServer/tile/{z}/{y}/{x}',
    #                 attr='Tiles &copy; Esri &mdash; Source: Esri, i-cubed, '
    #                      'USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN,'
    #                      ' IGP, UPR-EGP, and the GIS User Community')

    # Marker Clusters
    # N = 100

    #data = np.array(
    #    [
    #        np.random.uniform(low=45.8, high=47.8, size=N),
    #        # Random latitudes in Europe.
    #        np.random.uniform(low=6., high=11., size=N),
    #        # Random longitudes in Europe.
    #        range(N),  # Popups texts are simple numbers.
    #    ]
    #).T
    #
    #plugins.MarkerCluster(data).add_to(m)

    # Add Swiss boundaries for a slight highlighting
    ch_gdf = gpd.GeoDataFrame.from_file(ch_loc)
    ch_gjs = ch_gdf[ch_gdf.ICC != 'CH'].to_crs(epsg='4326').to_json()
    style_function = lambda ch_gjs: dict(stroke="#FFFFFF", color="#555555",
                                         weight=2, strokeopacity=1.,
                                         fillopacity=0.05)

    ch_poly = folium.features.GeoJson(ch_gjs,
                                      name='Swiss National Border',
                                      style_function=style_function)
    m.add_child(ch_poly)

    # Add glacier shapes
    glc_gdf = gpd.GeoDataFrame.from_file(shp_loc)
    glc_gdf = glc_gdf.sort_values(by='Area', ascending=False)
    glc_gdf = glc_gdf.head(1000)
    glc_gdf = glc_gdf[['Name', 'RGIId', 'CenLat', 'CenLon', 'Area',
                       'geometry']]
    cmap = plt.cm.get_cmap('autumn', len(glc_gdf))

    # THE RANDOM COLORS IS JUST TO PRODUCE FAKE DATA!!!!!!
    nums = np.random.choice(['yellow', 'orange', 'red'], size=len(glc_gdf))
    glc_gdf['randcolor'] = nums
    glc_gjs = glc_gdf.to_json()

    style_func2 = lambda feature: {
        'fillColor': feature['properties']['randcolor'],
        'color': feature['properties']['randcolor'],
        'weight': 1,
        'fillOpacity': 1,
    }
    polys = folium.features.GeoJson(glc_gjs,
                                    style_function=style_func2,
                                    name='Swiss Glacier Inventory 2010')
    m.add_child(polys)
    """
    # Test labels
    marker_cluster = folium.MarkerCluster().add_to(m)

    sorted = glc_gdf.sort_values('Area', ascending=False)
    sorted = sorted.reset_index()
    locations = sorted[['CenLat', 'CenLon']]
    locationlist = locations.values.tolist()
    # At the moment, the browser doesn't load the map anymore with more than
    # 10 markers with divicons
    #for i, row in enumerate(sorted.values[:10]):
    #    folium.Marker(locationlist[i], icon=folium.DivIcon(icon_size=(5, 5),
    #                                                       icon_anchor=(10, 10),
    #    html='<div style="font-size: 12pt">{}</div>'.format(glc_gdf['Name'][i]),
    #    )).add_to(marker_cluster)
    for i, row in enumerate(sorted.ix[:10]):
        folium.Marker(locationlist[i], icon=folium.DivIcon(icon_size=(5, 5),
                                                           icon_anchor=(10, 10),
        html='<div style="font-size: 12pt">{}</div>'.format(sorted.iloc[i]['Name']),
        )).add_to(marker_cluster)
    """
    ## Add PopUp on GeoJSON - makes the map superslow
    #for i, row in glc_gdf.iterrows():
    #    gj = folium.GeoJson(
    #        data={
    #            "type": "Polygon",
    #           "coordinates": [list(zip(row.geometry.exterior.xy[0],
    #                                    row.geometry.exterior.xy[1]))]
    #        }
    #    )
    #
    #    gj.add_child(folium.Popup("outline Popup on GeoJSON"))
    #    gj.add_to(m)

    # Add the layer control icon (do not forget!!!)
    folium.LayerControl().add_to(m)

    # Save
    m.save('c:\\users\\johannes\\desktop\\index.html')


def plot_compare_cali(wdir=None, dev=5.):
    """
    Compare OGGM and CRAMPON calibration with scatterplot.

    Parameters
    ----------
    wdir: str
        An OGGM/CRAMPON working directory with local_mustar.csv and
        mustar_from_mauro.csv
    dev: float
        The deviation from the calibration value (in percent), used to fit the
        colormaps

    Returns
    -------

    """

    if not wdir:
        raise('Please give a valid working directory.')

    comp = [i for i in glob(wdir + '\\per_glacier\\*\\*\\*\\') if
            (os.path.exists(i+'local_mustar.csv') and
             os.path.exists(i+'mustar_from_mauro.csv'))]

    oggm_m = []
    oggm_p = []
    crm_m = []
    crm_p = []
    # use error as color for scatter dots
    err_perc = []
    for p in comp:
        oggm_df = pd.read_csv(p+'local_mustar.csv')
        crm_df = pd.read_csv(p+'mustar_from_mauro.csv')

        oggm_m.append(oggm_df.mu_star.iloc[0])
        oggm_p.append(oggm_df.prcp_fac.iloc[0])
        crm_m.append(crm_df.mu_star.iloc[0])
        crm_p.append(crm_df.prcp_fac.iloc[0])
        err_perc.append(crm_df.err_perc.iloc[0])

    ax = plt.gca()
    ax.scatter(oggm_m, oggm_p)
    pcm = ax.scatter(crm_m, crm_p, c=err_perc, cmap='hsv', vmin=-dev, vmax=dev)
    ax.set_xscale('log')
    plt.grid(True, which="both")
    pcm.cmap.set_under('k')
    pcm.cmap.set_over('k')
    plt.colorbar(pcm, ax=ax, label='Error(%)', extend='both')
    plt.xlabel(r'Temperature Sensitivity (${\mu}^{*}$)')
    plt.ylabel(r'Precipitation Correction Factor')
    plt.show()


# this is for the custom legend entry to make the quantile colors visible
class AnyObject(object):
    pass


class AnyObjectHandler(object):

    def __init__(self, color='b', facecolor='cornflowerblue'):
        self.facecolor= facecolor
        self.color = color

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width],
                           [0.5 * height, 0.5 * height],
                           linestyle='-', color=self.color)
        patch1 = mpatches.Rectangle([x0, y0 + 0.25 * height], width,
                                    0.5 * height,
                                    facecolor=self.facecolor,
                                    alpha=0.5,
                                    transform=handlebox.get_transform())
        patch2 = mpatches.Rectangle([x0, y0], width, height,
                                    facecolor=self.facecolor,
                                    alpha=0.3,
                                    transform=handlebox.get_transform())
        handlebox.add_artist(l1)
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch2)
        return [l1, patch1, patch2]

#@entity_task(log)
def plot_cumsum_climatology_and_current(clim, current, fs=17, loc=0):
    """
    Make the standard plot containing the MB climatology in the background and
    the current MB year on top.

    In leap years, the plot has length 366, otherwise length 365 and DOY 366
    is removed.

    An option should be added that lets you choose the time span of the
    climatology displayed in the background (added to filename and legend!)

    Parameters
    ----------
    clim: xarray.Dataset
        The mass balance climatology. The mass balance variable should be named
         'MB' and mass balance should have two coordinates (e.g. 'time' and 'n'
         (some experiment)).
    current: xarray.Dataset
        The current year's mass balance. The mass balance variable should be
        named 'MB' and mass balance should have two coordinates (e.g. 'time'
        and 'n' (some experiment)).
    fs: int
        Font size for title, axis labels and legend.
    loc: int
        Legend position as passed to plt.legend(). Sometimes the placing of the
        legend fails when done automatically, so this helps to keep control.
        Default: 0 ('best' position).

    Returns
    -------
    None
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # currently no better solution to add one year
    xtime = pd.date_range(pd.to_datetime(current.isel(time=0).time.values),
                             periods=366)
    if ((xtime.month == 2) & (xtime.day == 29)).any():
        xvals = np.arange(366)
    else:
        xtime = xtime[:-1]
        xvals = np.arange(365)
        clim = clim.sel(hydro_doys=slice(None, 365))

    # plot median
    p1, = ax.plot(xvals, clim.MB.values[:, 2], c='b', label='Median')
    # plot IQR
    ax.fill_between(xvals, clim.MB.values[:, 1], clim.MB.values[:, 3],
                    facecolor='cornflowerblue', alpha=0.5)
    # plot 10th to 90th pctl
    ax.fill_between(xvals, clim.MB.values[:, 0], clim.MB.values[:, 4],
                    facecolor='cornflowerblue', alpha=0.3)
    # plot MB of this glacio year up to now
    mb_now_cs_pad = np.lib.pad(current.MB.values,
                               ((0, len(xvals) - current.MB.shape[0]), (0, 0)),
                               'constant',
                               constant_values=(np.nan, np.nan))
    p4, = ax.plot(xvals, mb_now_cs_pad[:, 2], c='darkorange', label='Median')
    ax.fill_between(xvals, mb_now_cs_pad[:, 1], mb_now_cs_pad[:, 3],
                    facecolor='orange', alpha=0.5)
    # plot 10th to 90th pctl
    ax.fill_between(xvals, mb_now_cs_pad[:, 0], mb_now_cs_pad[:, 4],
                    facecolor='orange', alpha=0.3)
    ax.set_xlabel('Months', fontsize=16)
    ax.set_ylabel('Cumulative Mass Balance (m w.e.)', fontsize=fs)
    ax.set_xlim(xvals.min(), xvals.max())
    plt.tick_params(axis='both', which='major', labelsize=fs)
    mbeg = xtime[np.where(xtime.day == 1)]
    if xtime[0].day == 1:
        xtpos_init = [0]
        month_int = mbeg.month
    else:
        xtpos_init = [(xtime[0].replace(month=xtime[0].month+1, day=1) -
                        xtime[0]).days]
        month_int = mbeg.month
    xtpos = np.cumsum(xtpos_init +
                      [(mbeg[i]-mbeg[i-1]).days for i in range(1, len(mbeg))])
    ax.xaxis.set_ticks(xtpos)
    ax.xaxis.set_ticks(xvals, minor=True)
    mstr = 'JFMAMJJAS0ND'
    ax.set_xticklabels([mstr[i-1] for i in month_int], fontsize=fs)
    ax.grid(True, which='both', alpha=0.5)
    plt.title('Daily Cumulative MB Distribution of {} ({})'
              .format(clim.attrs['id'].split('.')[1],
                      clim.attrs['name']), fontsize=fs)

    entry_one = AnyObject()
    entry_two = AnyObject()
    ax.legend([entry_one, entry_two],
              ['Climatology Median, IQR, 10th/90th PCTL',
               'Current Year Median, IQR, 10th/90th PCTL'], fontsize=fs,
              loc=loc,
              handler_map={entry_one: AnyObjectHandler(color='b',
                                                       facecolor='cornflowerblue'),
                           entry_two: AnyObjectHandler(color='darkorange',
                                                       facecolor='orange')})
    plt.tight_layout()


def plot_cumsum_allyears(gdir):
    # input must be  mb_daily.pkl
    mb_ds = gdir.read_pickle('mb_daily')
    mb_ds_cs = mb_ds.apply(lambda x: x.cumsum(dim='time', skipna=True))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(mb_ds_cs.MB.values)
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Cumulative Mass Balance (m we)', fontsize=16)
    plt.title('Cumulative MB Distribution of ' +
              mb_ds.attrs['id'].split('.')[1] + ' (' + mb_ds.attrs[
                  'name'] + ')', fontsize=16)
    xtpos = np.append([0], np.cumsum([np.count_nonzero(
        [i.year for i in pd.DatetimeIndex(mb_ds.time.values)] == y) for y in
                                      np.unique([t.year for t in
                                                 pd.DatetimeIndex(
                                                     mb_ds.time.values)])]))
    plt.tick_params(axis='both', which='major', labelsize=16)
    ax.xaxis.set_ticks(xtpos[::5])
    ax.set_xticklabels([y for y in np.unique(
        [t.year for t in pd.DatetimeIndex(mb_ds.time.values)])
                        ][::5], fontsize=16)


def plot_animated_swe(mb_model):
    """
    Plots the snow water equivalent of a glacier animated over time.

    Parameters
    ----------
    mb_model: MassBalanceModel Object with `snow` attribute
        The model whose snow to plot.

    Returns
    -------
    ani: animation.FuncAnimation
        The produced animation.
    """
    fig, ax = plt.subplots()
    line, = ax.plot(np.arange(len(mb_model.heights)), mb_model.heights)
    ax.set_ylim(
        (np.min(mb_model.snow) / 1000., np.max(mb_model.snow) / 1000.))
    plt.xlabel('Concatenated flowline heights')
    plt.ylabel('Snow Water Equivalent (m w.e.)')
    time_text = ax.text(.5, .5, '', fontsize=15)

    def animate(i, data, line, time_text):
        line.set_ydata(data.snow[i] / 1000.)  # update the data
        time_text.set_text(data.time_elapsed[i].strftime("%Y-%m-%d"))
        return line,

    ani = animation.FuncAnimation(fig, animate,
                                  frames=mb_model.snow.shape[0],
                                  fargs=(mb_model, line, time_text),
                                  interval=10, repeat_delay=1.)
    return ani
