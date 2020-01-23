from __future__ import division

from oggm.graphics import *
from crampon.utils import entity_task, global_task
from crampon import utils
from crampon import workflow
from crampon.core.models.massbalance import MassBalance
import xarray as xr
from matplotlib.ticker import NullFormatter
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
from bokeh.models import Legend, LegendItem
from scipy.stats import percentileofscore
from scipy.stats import pearsonr

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


def make_mb_popup_map(mb_model=None, shp_loc=None, ch_loc=None):
    """
    Create a clickable map of all Swiss glacier outlines.

    the plot will be saved automatically to the 'plots' directory of the
    working directory.

    Parameters
    ----------
    mb_model: `py:class:crampon.core.models.massbalance.MassBalanceModel`, str
        The mass balance model output to plot. Default: None (use ensemble
        median estimate).
    shp_loc: str
        Path to the glacier shapefile. If None, search in the data directory.
        Default: None.
    ch_loc: str
        Path to the shapefile with Swiss borders. If None, search in the data
        directory. Default: None.

    Returns
    -------
    None
    """

    if isinstance(mb_model, utils.SuperclassMeta):
        mb_model = mb_model.__name__

    if shp_loc is None:
        shp_loc = os.path.join(cfg.PATHS['data_dir'], 'outlines',
                               'mauro_sgi_merge.shp')
    if ch_loc is None:
        ch_loc = os.path.join(cfg.PATHS['data_dir'], 'outlines',
                              'VEC200_LANDESGEBIET_LV03.shp')

    # A map zoomed on Switzerland
    m = folium.Map(location=[46.44, 8.37], tiles='cartoDB Positron',
                   zoom_start=9, control_scale=True)

    # Make a full screen option
    plugins.Fullscreen(position='topright',
                       force_separate_button=True).add_to(m)

    # for fun ;-)
    folium.plugins.Terminator().add_to(m)

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

    # Add Swiss boundaries for a slight highlighting
    ch_gdf = gpd.GeoDataFrame.from_file(ch_loc)
    ch_gjs = ch_gdf[ch_gdf.ICC != 'CH'].to_crs(epsg='4326').to_json()
    style_function = lambda ch_gjs: dict(stroke="#FFFFFF", color="#555555",
                                         weight=2, strokeopacity=1.,
                                         fillopacity=0.)

    ch_poly = folium.features.GeoJson(ch_gjs,
                                      name='Swiss National Border',
                                      style_function=style_function)
    m.add_child(ch_poly)

    # Add glacier shapes
    glc_gdf = gpd.GeoDataFrame.from_file(shp_loc)
    glc_gdf = glc_gdf.sort_values(by='Area', ascending=False)
    glc_gdf = glc_gdf[['Name', 'RGIId', 'CenLat', 'CenLon', 'Area',
                       'geometry']]

    # todo: externalize this function and make it an entity task
    has_current_and_clim = []
    error = 0
    for id in glc_gdf.RGIId.values:
        try:
            gd = utils.GlacierDirectory(id, base_dir=os.path.join(
                cfg.PATHS['working_dir'], 'per_glacier'))
            clim = gd.read_pickle('mb_daily')
            current = gd.read_pickle('mb_current')
        except:
            error += 1
            continue
        if mb_model is not None:
            current = current.sel(model=mb_model)
        current_csq = current.mb.make_cumsum_quantiles()
        mbc_values = current_csq.sel(quantile=0.5)
        mbc_value = mbc_values.MB.isel(hydro_doys=-1)

        hydro_years = clim.mb.make_hydro_years()
        mb_cumsum = clim.groupby(hydro_years).apply(
            lambda x: MassBalance.time_cumsum(x))

        # todo: EMERGENCY SOLUTION as long as we are not able to calculate
        #  cumsum with NaNs correctly
        mb_cumsum = mb_cumsum.where(mb_cumsum.MB != 0.)

        mbd_values = [j.MB.sel(
            time=pd.to_datetime(j.time.values[0]) + dt.timedelta(
                days=mbc_values.hydro_doys[-1].item() - 1)).median(
            dim='model', skipna=True).item() for i, j in
                      list(mb_cumsum.groupby(hydro_years))[:-1]]
        mbd_values = sorted(mbd_values)
        pctl = percentileofscore(mbd_values, mbc_value.item())
        glc_gdf.loc[glc_gdf.RGIId == id, 'pctl'] = pctl
        has_current_and_clim.append(id)
        log.info('Successfully calculated mb distribution for {}'.format(id))
    glc_gdf = glc_gdf[glc_gdf.RGIId.isin(has_current_and_clim)]

    cmap_str = 'rainbow_r'
    colors = matplotlib.colors.Normalize(vmin=0.0, vmax=100)
    glc_gdf['polycolor'] = [matplotlib.colors.rgb2hex(i) for i in
                            [plt.cm.get_cmap(cmap_str)(colors(j)) for j in
                             glc_gdf.pctl.values]]
    # branca colormap as folium input, but it doesn't have all mpl ones
    colormap_full = [plt.cm.get_cmap(cmap_str)(colors(j)) for j in range(101)]
    branca_colormap_full = branca.colormap.LinearColormap(
        colormap_full, vmin=0, vmax=100).scale(0, 100).to_step(100)
    branca_colormap_full.caption = 'Current MB median as percentile of ' \
                                   'climatology'
    m.add_child(branca_colormap_full)

    style_func2 = lambda feature: {
        'fillColor': feature['properties']['polycolor'],
        'color': feature['properties']['polycolor'],
        'weight': 1,
        'fillOpacity': 0.8,
    }

    def highlight_function(feature):
        return {
            'fillColor': feature['properties']['polycolor'],
            'color': feature['properties']['polycolor'],
            'weight': 3,
            'fillOpacity': 1
        }

    def popup_html_string(point):
        """
        Make an HTML string used in a popup over a glacier.

        Parameters
        ----------
        point: pd.Series
           The Series must contain

        Returns
        -------
        html: str
            HTML string.
        """
        pure_sgi_id = point.RGIId.split('.')[1]
        clickable_mb_dist = \
            '<a href="https://crampon.glamos.ch/plots/mb_dist/{}_mb_dist_' \
            'ensemble.png" target="_blank"><p>Mass Balance Distribution</p>' \
            '<img src="https://crampon.glamos.ch/plots/mb_dist/{}_mb_dist_' \
            'ensemble_prev.png" style="max-width:100%; position:relative; ' \
            'display:inline; overflow:hidden; margin:0;" /></a>'.format(
                pure_sgi_id, pure_sgi_id)
        other_clickables = \
            '<a href="https://crampon.glamos.ch/plots/mb_spaghetti/{}_intera' \
            'ctive_mb_spaghetti.html" target="_blank"><p>MB Spaghetti</p></a' \
            '>'.format(pure_sgi_id)

        html = '<b> ' + point.Name + ' (' + pure_sgi_id + ')</b><br><br>'
        html += '<div style="width:5000; height:1000; text-align: center">{}' \
                '</div>'.format(clickable_mb_dist)
        html += 'Further information:<br>'
        html += other_clickables
        return html

    glc_gdf['popup_html'] = glc_gdf.apply(popup_html_string, axis=1)

    layer_geom = folium.FeatureGroup(name='layer', control=False)

    glc_gjs = glc_gdf.__geo_interface__
    for i in range(len(glc_gjs["features"])):
        temp_geojson = {"features": [glc_gjs["features"][i]],
                        "type": "FeatureCollection"}
        temp_geojson_layer = folium.GeoJson(temp_geojson,
                                            highlight_function=
                                            highlight_function,
                                            control=False,
                                            style_function=style_func2,
                                            smooth_factor=0.5)
        folium.Popup(
            temp_geojson["features"][0]["properties"]['popup_html']).add_to(
            temp_geojson_layer)
        temp_geojson_layer.add_to(layer_geom)

    layer_geom.add_to(m)

    # Add the layer control icon (do not forget!!!)
    folium.LayerControl().add_to(m)

    # tell when it was updated
    date_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    update_html = '<div style="position: fixed; bottom: 39px; left: 5px; ' \
                  'width: 210px; height: 21px; border:2px solid grey; ' \
                  'z-index:9999; font-size:11px;background-color:white;' \
                  'opacity:0.6"' \
                  '> Last updated: {} </div>'.format(date_str)
    m.get_root().html.add_child(folium.Element(update_html))

    # Save
    plots_dir = os.path.join(cfg.PATHS['working_dir'], 'plots')
    m.save(os.path.join(plots_dir, 'status_map.html'))


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


def plot_cumsum_allyears(gdir, mb_model=None):
    """
    Plot the cumulative sum of the mass balance of all years available.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to show the data for.
    mb_model: str or None
        If only one mass balance model shall be shown. Default: None (show all
        available).

    Returns
    -------
    None
    """

    mb_ds = gdir.read_pickle('mb_daily')
    mb_ds_cs = mb_ds.apply(lambda x: x.cumsum(dim='time', skipna=True))
    if mb_model is not None:
        mb_ds_cs = mb_ds_cs.sel(model=mb_model)

    fig, ax = plt.subplots(figsize=(10, 5))
    mb_ds_cs.MB.plot(x='time', hue='model', ax=ax)
    ax.set_xlabel('Time', fontsize=14)
    ax.set_ylabel('Cumulative Mass Balance (m w.e.)', fontsize=14)
    plt.title('Cumulative MB Distribution of ' +
              mb_ds.attrs['id'].split('.')[1] + ' (' + mb_ds.attrs[
                  'name'] + ')', fontsize=14)


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


def plot_interactive_mb_spaghetti_html(gdir, plot_dir, mb_models=None):
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

    mb_clim = gdir.read_pickle('mb_daily')
    mb_curr = gdir.read_pickle('mb_current')

    if mb_models is not None:
        mb_clim = mb_clim.sel(model=mb_models)
        mb_curr = mb_curr.sel(model=mb_models)

    fig, ax = plt.subplots(figsize=(14, 10), subplot_kw={'xticks': [],
                                                         'yticks': []})

    arr = None
    mbcarr = None
    years = []
    mbcyears = []
    models = []

    hydro_years = mb_clim.mb.make_hydro_years()
    for y, group in list(mb_clim.groupby(hydro_years)):
        # make use of static method
        to_plot = MassBalance.time_cumsum(group)
        # todo: EMERGENCY SOLUTION as long as we are not able to calculate cumsum with NaNs correctly
        to_plot = to_plot.where(to_plot.MB != 0.)
        to_plot = to_plot.median(dim='member')

        # the last hydro years contains one values only
        if y == max(hydro_years.values):
            continue

        for model in to_plot.model.values:
            # extend every year to 366 days and stack
            gsel_mbvals = to_plot.sel(model=model).MB.values
            if arr is None:
                arr = np.lib.pad(gsel_mbvals.flatten(),
                                 (0, 366 - len(gsel_mbvals)),
                                 mode='constant', constant_values=(
                        np.nan, np.nan))
            else:
                arr = np.vstack((arr, np.lib.pad(gsel_mbvals.flatten(),
                                                 (0, 366 - len(gsel_mbvals)),
                                                 mode='constant',
                                                 constant_values=(np.nan,
                                                                  np.nan))))
            years.append(y)
            models.append(model)

    mb_curr_cs = MassBalance.time_cumsum(mb_curr)
    # todo: EMERGENCY SOLUTION as long as we are not able to calculate cumsum
    #  with NaNs correctly
    mb_curr_cs = mb_curr_cs.where(mb_curr_cs.MB != 0.)
    # add current median
    for model in mb_curr_cs.model.values:
        mb_curr_sel = mb_curr_cs.sel(model=model)
        mb_curr_med = mb_curr_sel.median(dim='member').MB.values

        if mbcarr is None:
            mbcarr = np.lib.pad(mb_curr_med,
                                (0,
                                 366 - mb_curr_med.shape[0]),
                                mode='constant',
                                constant_values=(np.nan, np.nan))
        else:
            mbcarr = np.vstack(
                (mbcarr, np.lib.pad(mb_curr_med,
                                    (0,
                                     366 -
                                     mb_curr_med.shape[0]),
                                    mode='constant',
                                    constant_values=(
                                        np.nan, np.nan))))
        mbcyears.append(mb_curr.time[-1].dt.year.item())
        models.append(model)

    arr = np.vstack((arr, mbcarr))
    years.extend(mbcyears)

    greyshades = len(years) - len(mbcyears)
    if greyshades > 256:  # the length of the palette
        greyshades = np.lib.pad(np.arange(256), pad_width=greyshades-256,
                                mode='symmetric')
    current_colors = [c for c, _ in CURR_COLORS[:len(mbcyears)]]
    custompalette = grey(greyshades) + current_colors

    xs = [np.arange(arr.shape[1])] * arr.shape[0]
    ys = arr.tolist()
    desc = years
    color = custompalette
    alpha = (len(years) - len(mbcyears)) * [0.3] + len(mbcyears) * [0.9]
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
        format(mb_clim.attrs['id'].split('.')[1], mb_clim.attrs['name'])
    plot.xaxis.axis_label = 'Days of Hydrological Year'
    plot.yaxis.axis_label = 'Cumulative Mass Balance'
    if utils.leap_year(mb_curr.time[0].dt.year.item()+1):
        dim = cfg.DAYS_IN_MONTH_LEAP
    else:
        dim = cfg.DAYS_IN_MONTH
    xticks = np.cumsum(np.append([0], np.roll(dim, 3)[:-1]))
    xlabel_dict = {}
    for i, s in zip(xticks, [i for i in 'ONDJFMAMJJAS']):
        xlabel_dict[str(i)] = s
    plot.xaxis.ticker = xticks
    plot.xaxis.major_label_overrides = xlabel_dict
    plot.grid.ticker = FixedTicker(ticks=xticks)

    r = plot.multi_line('xs', 'ys', source=source,
                        color='color', alpha='alpha', line_width=4,
                        hover_line_alpha=1.0, hover_line_color='color')

    legendentries = ['Past mass balances'] + list(mb_curr_cs.model.values)

    def leg_idx(x):
        if x != 0:
            return int((len(years) - len(mbcyears)) + x - 1)
        else:
            return int(len(years) / 2.)

    legend = Legend(
        items=[LegendItem(label=le, renderers=[r], index=leg_idx(i)) for
               i, le in enumerate(legendentries)])
    plot.add_layout(legend)
    plot.legend.location = "top_left"

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


def plot_animated_potential_irradiation(gdir):
    """
    Make an animation of the daily potential irradiation of a glacier.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The glacier directory to make the plot for.

    Returns
    -------
    ani: matplotlib.animation.ArtistAnimation
        The produced animation object.
    """
    # todo: try this with a GoogleVisibleMap (terrain) in the background!
    # https://github.com/fmaussion/salem/blob/master/docs/examples/plot_googlestatic.py

    ipot_ds = xr.open_dataset((gdir.get_filepath('ipot')))
    ipot_array = ipot_ds.ipot.values
    time = ipot_ds.time.values

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ims = []
    max_val = np.nanmax(ipot_array)
    min_val = np.nanmin(ipot_array)
    valid = np.where(~np.isnan(ipot_array))
    for i in range(ipot_array.shape[2]):
        im = ax.imshow(ipot_array[min(valid[0]):max(valid[0]),
                       min(valid[1]):max(valid[1]),
                       i].T, aspect='auto',
                       cmap='gist_ncar', vmin=min_val, vmax=max_val,
                       animated=True)
        t = ax.annotate(time[i].strftime('%m-%d'), (0.01, 0.98),
                        xycoords='axes fraction', verticalalignment='top',
                        color='k')
        if i == 0:
            fig.colorbar(im)
        ims.append([im, t])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)

    return ani


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    From https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"], threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    From https://matplotlib.org/3.1.0/gallery/images_contours_and_fields/image_annotated_heatmap.html

    Parameters
    ----------
    im:
        The AxesImage to be labeled.
    data:
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt:
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors:
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold:
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    data = data.filled(np.nan)

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def make_annotated_heatmap(df, pval_thresh=0.01):
    """
    Create a correlation heatmap with variable name annotations.

    Parameters
    ----------
    df: pd.DataFrame
        A pandas DataFrame containing columns that should be correlated.

    Returns
    -------
    None
    """

    #df = df[~pd.isnull(df)]
    corr = df.corr()

    # get the p-values (pd doesn't take care of them), make use of method kw
    def pearsonr_pval(x, y):
        return pearsonr(x, y)[1]
    pvals = df.corr(method=pearsonr_pval)

    corr[pvals > pval_thresh] = np.nan

    im, _ = heatmap(corr.values, corr.columns.values, corr.columns.values,
                    cmap="RdBu_r", vmin=-1, vmax=1,
                    cbarlabel="correlation coeff.")

    def func(x, pos):
        return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

    annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

    plt.tight_layout()
    plt.show()

def plot_holfuy_station_availability():
    from crampon.core.models import assimilation
    from crampon import utils
    import matplotlib.dates as mdates

    fontsize = 30

    def add_index(ixlist, g, s):
        if g == 'RGI50-11.A55F03':
            ix = 0
        if g == 'RGI50-11.B4312n-1':
            if s == 2235.:
                ix = 1
            if s == 2241.:
                ix = 2
            if s == 2392.:
                ix = 3
            if s == 2589.:
                ix = 4
        if g == 'RGI50-11.B5616n-1':
            if s == 2551.:
                ix = 5
            if s == 3015.:
                ix = 6
        ixlist.append(ix)
        return ixlist

    label_list = ['PLM 2689m', 'RHO 2235m', 'RHO 2241m', 'RHO 2392m',
                  'RHO 2589m', 'FIN 2551m', 'FIN 3015m']

    ice_beg = []
    ice_end = []
    snow_beg = []
    snow_end = []
    other_beg = []
    other_end = []
    ice_ixs = []
    snow_ixs = []
    other_ixs = []
    for id in ['RGI50-11.A55F03', 'RGI50-11.B4312n-1', 'RGI50-11.B5616n-1']:
        gdir = utils.GlacierDirectory(id, base_dir=cfg.PATHS[
                                                       'working_dir'] + '\\per_glacier\\')
        obs_merge = assimilation.prepare_holfuy_camera_readings(gdir,
                                                                ice_only=False, exclude_keywords=False)
        for h in obs_merge.height.values:
            station_phase = obs_merge.sel(height=h).phase
            station_swe = obs_merge.sel(height=h).swe
            station_phase_i = station_phase.where((station_phase == 'i') & (~pd.isnull(station_swe)),
                                                  drop=True)
            station_phase_s = station_phase.where((station_phase == 's') & (~pd.isnull(station_swe)),
                                                  drop=True)
            station_phase_nan = station_phase[pd.isnull(station_swe)]
            print(h)
            ice_beg.append(station_phase_i.date.values[0])
            ice_ixs = add_index(ice_ixs, id, h)
            print('ICE DAYS: ', str(station_phase_i.date.values.size))
            for i in range(1, len(station_phase_i.date.values)):
                if station_phase_i.date.values[i - 1] + pd.Timedelta(days=1) == \
                        station_phase_i.date.values[i]:
                    pass
                else:
                    ice_end.append(station_phase_i.date.values[i - 1])
                    ice_beg.append(station_phase_i.date.values[i])
                    ice_ixs = add_index(ice_ixs, id, h)
            ice_end.append(station_phase_i.date.values[-1])
            # ice_ixs = add_index(ice_ixs, id, h)

            snow_beg.append(station_phase_s.date.values[0])
            snow_ixs = add_index(snow_ixs, id, h)
            print('SNOW DAYS: ', str(station_phase_s.date.values.size))
            for i in range(1, len(station_phase_s.date.values)):
                if station_phase_s.date.values[i - 1] + pd.Timedelta(days=1) == \
                        station_phase_s.date.values[i]:
                    pass
                else:
                    snow_end.append(station_phase_s.date.values[i - 1])
                    snow_beg.append(station_phase_s.date.values[i])
                    snow_ixs = add_index(snow_ixs, id, h)
            snow_end.append(station_phase_s.date.values[-1])
            # snow_ixs = add_index(snow_ixs, id, h)

            if len(station_phase_nan) > 0:
                # If Nan is at beginning, the TS has been expanded to go in one xr.Dataset
                beg_ix = 0
                if station_phase_nan.date.values[beg_ix] == obs_merge.date.values[0]:
                    while station_phase_nan.date.values[beg_ix] + pd.Timedelta(days=1) == station_phase_nan.date.values[beg_ix+1]:
                        beg_ix += 1
                other_beg.append(station_phase_nan.date.values[beg_ix])
                other_ixs = add_index(other_ixs, id, h)
                for i in range(beg_ix+1, len(station_phase_nan.date.values)):
                    if station_phase_nan.date.values[i - 1] + pd.Timedelta(
                            days=1) == \
                            station_phase_nan.date.values[i]:
                        pass
                    else:
                        other_end.append(station_phase_nan.date.values[i - 1])
                        other_beg.append(station_phase_nan.date.values[i])
                        other_ixs = add_index(other_ixs, id, h)
                other_end.append(station_phase_nan.date.values[-1])
                # other_ixs = add_index(other_ixs, id, h)

    ice_beg = np.array(ice_beg)
    ice_end = np.array(ice_end) + pd.Timedelta(days=1)
    snow_beg = np.array(snow_beg)
    snow_end = np.array(snow_end) + pd.Timedelta(days=1)
    other_beg = np.array(other_beg)
    other_end = np.array(other_end) + pd.Timedelta(days=1)
    ice_ixs = np.array(ice_ixs)
    snow_ixs = np.array(snow_ixs)
    other_ixs = np.array(other_ixs)

    fig, ax = plt.subplots(figsize=(12,8))
    # Plot the data
    ax.barh(snow_ixs + 1, snow_end - snow_beg, left=snow_beg, height=0.7,
            align='center', facecolor='w', edgecolor='k', label='snow')
    ax.barh(ice_ixs + 1, ice_end - ice_beg, left=ice_beg, height=0.7,
            align='center', facecolor='b', edgecolor='k', label='ice')
    ax.barh(other_ixs + 1, other_end - other_beg, left=other_beg, height=0.7,
            align='center', alpha=0.2, facecolor='r', edgecolor='k',
            label='fail/f-snow')

    # make separating lines between the glaciers
    ax.axhline(1.5, c='k')
    ax.axhline(5.5, c='k')

    ax.xaxis_date()
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%d'))
    ax.yaxis.set_ticklabels([''] + label_list, fontsize=fontsize)
    ax.set_xlim(mdates.date2num(pd.Timestamp('2019-06-15')),
                mdates.date2num(pd.Timestamp('2019-10-15')))
    ax.set_title('Station availability 2019', fontsize=fontsize)

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.show()
