from __future__ import division

from crampon.utils import entity_task, global_task
from crampon import utils
from crampon import workflow
from crampon.core.models.massbalance import MassBalance
import xarray as xr
from matplotlib.ticker import NullFormatter
from oggm.graphics import *
from oggm.utils import nicenumber
import os
import datetime as dt
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
import re
from folium import plugins
from folium.features import DivIcon
import branca
import mpld3
import shapely
from bokeh.plotting import figure as bkfigure
from bokeh.plotting import output_file, save, show
from bokeh.models.tools import HoverTool, UndoTool, RedoTool
from bokeh.models import DataRange1d, Label
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.io import show, export_png
from bokeh.palettes import grey
from scipy.stats import percentileofscore

# Local imports
import crampon.cfg as cfg

# Module logger
log = logging.getLogger(__name__)

nullfmt = NullFormatter()  # no labels

# colors to display climatology
CLIM_COLORS = [('b', 'cornflowerblue'), ('darkmagenta', 'magenta'),
               ('deeppink', 'lightpink')]
# colors to display current analysis
CURR_COLORS = [('darkorange', 'orange'), ('darkgreen', 'forestgreen'),
               ('darkcyan', 'cyan'), ('darkgoldenrod', 'gold'),
               ('lightseagreen', 'turquoise'),
               ('yellowgreen', 'darkolivegreen')]
# colors to display mass balance forecasts
FCST_COLORS = [('darkred', 'red')]


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
def plot_cumsum_climatology_and_current(gdir=None, clim=None, current=None,
                                        mb_model=None, clim_begin=(10, 1),
                                        current_begin=(10, 1),
                                        abs_deviations=False, fs=14, loc=0):
    """
    Make the standard plot containing the MB climatology in the background and
    the current MB year on top.

    In leap years, the plot has length 366, otherwise length 365 and DOY 366
    is removed.

    An option should be added that lets you choose the time span of the
    climatology displayed in the background (added to filename and legend!)

    Parameters
    ----------
    clim: xarray.Dataset or None
        The mass balance climatology. The mass balance variable should be named
        'MB', where MassBalanceModel can be any
        MassBalanceModel and mass balance should have two coordinates (e.g.
        'time' and 'n' (some experiment)).
    current: xarray.Dataset
        The current year's mass balance. The mass balance variable should be
        named 'MassBalanceModel_MB', where MassBalanceModel can be any mass
        balance model class and mass balance should have two coordinates (e.g.
        'time' and 'n' (some experiment)).
    mb_model: str or None
        Mass balance model to make the plot for. must be contained in both the
        mass balance climatology and current mass balance file. Default: None
        (make plot for ensmeble mass balance estimate).
    clim_begin: tuple
        Tuple of (month, day) when the mass budget year normally begins.
    current_begin: tuple
        Tuple of (month, day) when the mass budget year begins in the current
        mass budget year.
    abs_deviations: bool
        # TODO: Redo this! There is a bug somewhere
        Whether or not to plot also the current absolute deviations of the
        prediction from the refrence median. Size of the figure will be
        adjusted accordingly to make both plots fit. Default: False.
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

    #clim = gdir.read_pickle('mb_daily')
    #current = gdir.read_pickle('mb_current')

    if mb_model is not None:
        try:
            clim = clim.sel(model=mb_model)
            current = current.sel(model=mb_model)
        except KeyError:
            log.error('Could not make mass balance distribution plot for {}. '
                      '"{}" needs to be contained in both mass balance '
                      'climatology and current mass balance.'.format(
                clim.rgi_id, mb_model))

    # make cumsum and quantiles
    cq = clim.mb.make_cumsum_quantiles(bg_month=clim_begin[0],
                                       bg_day=clim_begin[0])
    nq = current.mb.make_cumsum_quantiles(bg_month=current_begin[0],
                                          bg_day=current_begin[1])



    # todo: kick out all variables that are not MB (e.g. saved along cali params)

    if abs_deviations:
        fig, (ax, ax2) = plt.subplots(2, figsize=(10, 5),
                                      gridspec_kw={'height_ratios': [7, 1]})
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

    # Say if time on x axis is 365 or 366 long
    xtime = pd.date_range(
        dt.datetime(current.isel(time=0).time.dt.year, 10, 1),
        periods=366)
    if ((xtime.month == 2) & (xtime.day == 29)).any():
        xvals = np.arange(366)
    else:
        xtime = xtime[:-1]
        xvals = np.arange(365)
        cq = cq.sel(hydro_doys=slice(None, 365))

    # Say how much we would have to pad in front/back to match this time frame
    pad_f = (pd.Timestamp(current.isel(time=0).time.item()) -
             dt.datetime(current.isel(time=0).time.dt.year.item(),
                         clim_begin[0], clim_begin[1])).days

    #for ivar, (mbname, mbvar) in enumerate(current.data_vars.items()):
    if pad_f >= 0:
        pad_b = len(xvals) - current.MB.shape[0] - pad_f
        mb_now_cs_pad = np.lib.pad(nq.MB.values,
                                   ((pad_f, pad_b), (0, 0)),
                                   'constant',
                                   constant_values=(np.nan, np.nan))
    else:
        # todo: clip at the begin necessary if curr_beg different from clim_beg
        #current_clip = nq.isel(time=slice(
        #    dt.datetime(current.isel(time=0).time.dt.year.item(),
        #                clim_begin[0], clim_begin[1]), None))
        #mb_now_cs_pad = np.lib.pad(current_clip.MB.values, (
        #(0, len(xvals) - current_clip.MB.shape[0]), (0, 0)),
        #                           'constant',
        #                           constant_values=(np.nan, np.nan))
        mb_now_cs_pad = np.lib.pad(nq.MB.values, (
        (0, len(xvals) - nq.MB.shape[0]), (0, 0)),
                                   'constant',
                                   constant_values=(np.nan, np.nan))

    # plot median
    climvals = cq.MB.values
    model_ix = 0  # this can be used later if we plot models individually
    p1, = ax.plot(xvals, climvals[:, 2], c=CLIM_COLORS[model_ix][0],
                  label='Median')
    # plot IQR
    ax.fill_between(xvals, climvals[:, 1], climvals[:, 3],
                    facecolor=CLIM_COLORS[model_ix][1], alpha=0.5)
    # plot 10th to 90th pctl
    ax.fill_between(xvals, climvals[:, 0], climvals[:, 4],
                    facecolor=CLIM_COLORS[model_ix][1], alpha=0.3)
    # plot MB of this glacio year up to now
    p4, = ax.plot(xvals, mb_now_cs_pad[:, 2], c=CURR_COLORS[model_ix][0],
                  label='Median')
    ax.fill_between(xvals, mb_now_cs_pad[:, 1], mb_now_cs_pad[:, 3],
                    facecolor=CURR_COLORS[model_ix][1], alpha=0.5)
    # plot 10th to 90th pctl
    ax.fill_between(xvals, mb_now_cs_pad[:, 0], mb_now_cs_pad[:, 4],
                    facecolor=CURR_COLORS[model_ix][1], alpha=0.3)
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
    plt.suptitle('Daily Cumulative MB Distribution of {} ({})'
              .format(clim.attrs['id'].split('.')[1],
                      clim.attrs['name']), fontsize=fs)

    # todo: get labels correct for plotting more mb models
    entry_one = AnyObject()
    entry_two = AnyObject()
    ax.legend([entry_one, entry_two],
              ['Climatology Median, IQR, IDR',
               'Current Year Median, IQR, IDR'], fontsize=fs,
              loc=loc,
              handler_map={entry_one: AnyObjectHandler(color='b',
                                                       facecolor='cornflowerblue'),
                           entry_two: AnyObjectHandler(color='darkorange',
                                                       facecolor='orange')})
    # say when we have updated
    date_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    bprops = dict(facecolor='w', alpha=0.5)
    plt.text(0.05, 0.05, 'Last updated: {}'.format(date_str),
             transform=fig.transFigure, bbox=bprops)


    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # include suptitle


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


def plot_interactive_mb_spaghetti_html(gdir, plot_dir):
    """
    Makes an interactive spaghetti plot of all avaiable mass balance years.

    When hovering over the plot, the cumulative mass balance curves are
    highlighted and the value is displayed as tooltip.

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        The glacier directory for which to produce the plot.
    plot_dir:
        Directory where to store the plot (HTML file). Usually, this is
        something like `os.path.join(cfg.PATHS['working_dir'], 'plots')`.

    Returns
    -------
    None
    """

    mb_hist = gdir.read_pickle('mb_daily')
    mb_curr = gdir.read_pickle('mb_current')

    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'xticks': [],
                                                         'yticks': []})

    arr = None
    mbcarr = None
    years = []
    mbcyears = []
    models = []

    hydro_years = mb_hist.mb.make_hydro_years()
    for y, group in list(mb_hist.groupby(hydro_years)):
        # make use of static method
        to_plot = group.apply(MassBalance.time_cumsum)

        # the last hydro years contains one values only
        if y == max(hydro_years.values):
            continue

        for ivar, (mbname, mbvar) in enumerate(to_plot.data_vars.items()):
            if not 'MB' in mbname:
                continue
            # extend every year to 366 days and stack
            if arr is None:
                arr = np.lib.pad(mbvar.values.flatten(),
                                 (0, 366 - len(mbvar.values)),
                                 mode='constant', constant_values=(
                        np.nan, np.nan))
            else:
                arr = np.vstack((arr, np.lib.pad(mbvar.values.flatten(),
                                                 (0, 366 - len(mbvar.values)),
                                                 mode='constant',
                                                 constant_values=(np.nan,
                                                                  np.nan))))
            years.append(y)
            models.append(mbname.split('_')[0])

    # add current median
    for ivar, (mbname, mbvar) in enumerate(mb_curr.data_vars.items()):
        if not 'MB' in mbname:
            continue
        print(mbname)
        if mbcarr is None:
            mbcarr = np.lib.pad(mb_curr[mbname].values[:, 2],
                                (0,
                                 366 - mb_curr[mbname].values[:, 2].shape[0]),
                                mode='constant',
                                constant_values=(np.nan, np.nan))
        else:
            mbcarr = np.vstack(
                (mbcarr, np.lib.pad(mb_curr[mbname].values[:, 2],
                                    (0,
                                     366 -
                                     mb_curr[mbname].values[:,
                                     2].shape[0]),
                                    mode='constant',
                                    constant_values=(
                                        np.nan, np.nan))))
        mbcyears.append(mb_curr.time[-1].dt.year.item())
        models.append(mbname.split('_')[0])

    arr = np.vstack((arr, mbcarr))
    years.extend(mbcyears)

    custompalette = grey(len(years) - len(mbcyears)) + \
                    [c for c, _ in CURR_COLORS[:len(mbcyears)]]

    xs = [np.arange(arr.shape[1])] * arr.shape[0]
    ys = arr.tolist()
    desc = years
    color = custompalette
    alpha = (len(years) - len(mbcyears)) * [0.3] + len(mbcyears) * [1.0]
    source = ColumnDataSource(dict(
        xs=xs,
        ys=ys,
        desc=desc,
        model=models,
        color=color,
        alpha=alpha,
    ))

    xdr = DataRange1d()
    ydr = DataRange1d()

    plot = bkfigure(plot_width=1200, plot_height=800)
    plot.title.text = 'Cumulative Mass Balance of {} ({})'.\
        format(mb_hist.attrs['id'].split('.')[1], mb_hist.attrs['name'])
    plot.xaxis.axis_label = 'Days of Hydrological Year'
    plot.yaxis.axis_label = 'Cumulative Mass Balance'
    xticks = np.cumsum(np.append([0], np.roll(cfg.DAYS_IN_MONTH, -3)[:-1]))
    xlabel_dict = {}
    for i, s in zip(xticks, [i for i in 'ONDJFMAMJJAS']):
        xlabel_dict[str(i)] = s
    plot.xaxis.ticker = xticks
    plot.xaxis.major_label_overrides = xlabel_dict
    plot.grid.ticker = FixedTicker(ticks=xticks)

    plot.multi_line('xs', 'ys', source=source,
                    color='color', alpha='alpha', line_width=4,
                    hover_line_alpha=1.0, hover_line_color='color')

    TOOLTIPS = [
        ("year", "@desc"),
        ("(HYD-DOY,CUM-MB)", "($x{0.}, $y)"),
        ("MODEL", "@model"),
    ]
    plot.add_tools(HoverTool(
        tooltips=TOOLTIPS,
        mode='mouse',
    ))

    # Add a note to say when it was last updated
    date_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    update_txt = Label(x=3., y=3., x_units='screen', y_units='screen',
                       border_line_color='black', border_line_alpha=1.0,
                       background_fill_color='white',
                       background_fill_alpha=0.5,
                       text='Last updated: {}'.format(date_str))
    plot.add_layout(update_txt)

    plot.add_tools(UndoTool())
    plot.add_tools(RedoTool())

    output_file(os.path.join(plot_dir, gdir.rgi_id +
                             '_interactive_mb_spaghetti.html'))
    export_png(plot, os.path.join(plot_dir, gdir.rgi_id +
                                  '_interactive_mb_spaghetti.png'))
    save(plot)
    show(plot)


def plot_topo_per_glacier(start_year=2005, end_year=None):
    """
    Make plot of Switzerland telling how many DEMs there are per glacier.

    Parameters
    ----------
    start_year: int
        Year when to start counting DEMs. Default: 2005 (start of National
        Forest Inventory Flights).
    end_year: int or None
        Year when to stop counting DEMs. Default: None (take all since
        start_year until present).

    Returns
    -------

    """

    gdirs = workflow.init_glacier_regions()
    gdf = None

    # project everything in WGS84
    proj_epsg_number = 4326

    for g in gdirs:
        try:
            shp = gpd.read_file(g.get_filepath('outlines'))
            shp_wgs84 = shp.to_crs(epsg=proj_epsg_number)
            dems = xr.open_dataset(g.get_filepath('homo_dem_ts'))
            years = np.array([t.dt.year for t in dems.time])
            if end_year:
                years = years[np.where(start_year <= years <= end_year)]
            else:
                years = years[np.where(start_year <= years)]

            if gdf is None:
                gdf = shp_wgs84
            else:
                gdf = gdf.append(shp_wgs84)
            gdf.loc[gdf.RGIId == g.rgi_id, 'n_dems'] = len(years)
        except FileNotFoundError:
            continue

    fig, ax = plt.subplots()
    ds = salem.open_xr_dataset(cfg.PATHS['hfile'])
    ds_sub = ds.salem.subset(corners=((45.7321, 47.2603), (6.79963, 10.4279)),
                             crs=salem.wgs84)
    smap = ds_sub.salem.get_map(countries=False)
    _ = smap.set_topography(ds_sub.hgt)
    smap.set_shapefile(os.path.join(cfg.PATHS['data_dir'],'outlines',
                                    'VEC200_LANDESGEBIET_LV03.shp'),
                       edgecolor='k', facecolor='None')
    smap.visualize(ax=ax)
    points = gdf.to_crs(epsg=4326).copy()
    points['geometry'] = points['geometry'].centroid
    points['grid_x'], points['grid_y'] = smap.grid.transform(
        points.geometry.x.values, points.geometry.y.values)
    colors = matplotlib.colors.Normalize(vmin=0.0, vmax=max(points.n_dems))
    for n_dems in range(int(max(points.n_dems))):
        points_sel = points.loc[points.n_dems == n_dems]
        ax.scatter(points_sel.grid_x, points_sel.grid_y,
                   color=plt.cm.get_cmap('Reds')(colors(int(n_dems))),
                   label=str(n_dems), s=50)
    if end_year:
        leg_title = 'DEMs per glacier from {} to {}'.format(start_year,
                                                            end_year)
    else:
        leg_title = 'DEMs per glacier since {}'.format(start_year)
    plt.legend(title=leg_title, loc=4, ncol=int(max(points.n_dems)),
               handletextpad=0.3, handlelength=0.7, framealpha=0.9)

