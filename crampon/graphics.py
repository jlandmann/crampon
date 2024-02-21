from __future__ import division
from typing import Optional, List

import pickle

from oggm.graphics import *
from crampon.utils import entity_task, popup_html_string
from crampon import utils, workflow
from crampon.core.preprocessing.climate import GlacierMeteo
from crampon.core.models.massbalance import MassBalance, DailyMassBalanceModel, \
    CurrentYearMassBalance, BraithwaiteModel, HockModel, PellicciottiModel, \
    OerlemansModel
from crampon.core import holfuytools
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
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, \
        VPacker
import matplotlib
from salem import get_demo_file, DataLevels, GoogleVisibleMap, Map
import numpy as np
from glob import glob
import folium
from folium import plugins
import branca
import mpld3
from shapely.geometry import Point
from bokeh.plotting import figure as bkfigure
from bokeh.plotting import output_file, save
from bokeh.models.tools import HoverTool, UndoTool, RedoTool
from bokeh.models import DataRange1d, Label
from bokeh.models.sources import ColumnDataSource
from bokeh.models.tickers import FixedTicker
from bokeh.io import export_png
from bokeh.io import show as bkshow
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
               ('deeppink', 'lightpink'), ('yellowgreen', 'darkolivegreen')]
# colors to display current analysis
CURR_COLORS = [('darkorange', 'orange'), ('lightseagreen', 'turquoise'),
               ('darkcyan', 'cyan'), ('darkgoldenrod', 'gold')]
# colors to display mass balance forecasts
FCST_COLORS = [('darkred', 'red'), ('darkgreen', 'forestgreen')]


def rand_jitter(arr: np.ndarray, scale_fac: float = 0.01) -> np.ndarray:
    """
    Add random jitter to an array of data.

    Edited from [1]_.

    Parameters
    ----------
    arr: np.ndarray
        2D array of data.
    scale_fac: float
        Factor that scales the peak-to-peak range of the array values to give
        the standard deviation of the Gaussian from which values for jitter
        are drawn. Default: 0.01.

    Returns
    -------
    np.ndarray
        Same array, but jitter added.

    References
    ----------
    .. [1] https://bit.ly/2ygPzmt
    """
    stdev = scale_fac * (np.nanmax(arr) - np.nanmin(arr))
    return arr + np.random.randn(len(arr)) * stdev


def jitter(x, y, s=20, c='b', marker='o', cmap=None, norm=None, vmin=None,
           vmax=None, alpha=None, linewidths=None, verts=None,  # hold=None,
           **kwargs):
    """
    Make a jitter plot instead of a scatter plot.

    Parameters
    ----------
    x :
    y :
    s :
    c :
    marker :
    cmap :
    norm :
    vmin :
    vmax :
    alpha :
    linewidths :
    verts :
    # hold :
    kwargs :

    Returns
    -------

    """
    return plt.scatter(rand_jitter(x), rand_jitter(y), s=s, c=c, marker=marker,
                       cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                       alpha=alpha, linewidths=linewidths, verts=verts,
                       # hold=hold,
                       **kwargs)


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
                         y=[np.nanmin(y), np.nanmax(y)])

    # the Google static image is a standard rgb image
    ggl_img = g.get_vardata()
    sm = Map(g.grid, factor=1)
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

    # fig = plt.figure()
    ax = plt.gca()
    ax.plot(df_comb['mu_star'], df_comb['prcp_fac'], 'o')
    ax.set_xscale('log')
    plt.show()

    return df_comb


def make_mb_popup_map(
        gdirs: Optional[List[utils.GlacierDirectory]] = None,
        mb_model: Optional[DailyMassBalanceModel] = None,
        shp_loc: Optional[str] = None, ch_loc: Optional[str] = None,
        save_suffix: Optional[str] = '', prefer_mb_suffix: Optional[str] = '',
        allow_unpreferred_mb_suffices: Optional[bool] = True,
        exclude_old_status: Optional[pd.Timedelta] = pd.Timedelta(days=7),
        plot_dir: Optional[str] = None) -> None:
    """
    Create a clickable map of all Swiss glacier outlines.

    the plot will be saved automatically to the 'plots' directory of the
    working directory.

    Parameters
    ----------
    gdirs: list of `py:class:utils.GlacierDirectory` or None, optional
        If given, only these GlacierDirectories are put on the map. Mutually
        exclusive with `shp_loc`. Default: None (take from shape_loc)
    mb_model: `py:class:crampon.core.models.massbalance.MassBalanceModel`, str
        The mass balance model output to plot. Default: None (use ensemble
        median estimate).
    shp_loc: str or None, optional
        Path to the glacier shapefile. If None, search in the data directory.
        Default: None.
    ch_loc: str or None, optional
        Path to the shapefile with Swiss borders. If None, search in the data
        directory. Default: None.
    save_suffix: str, optional
        Suffix to use when saving the plot. Default: '' (no suffix).
    prefer_mb_suffix: str, optional
        Which mass balance suffix shall be preferred when making the map, e.g.
        '_fischer_unique_variability'. Default: '' (take the default order
        from 'good' to 'bad'.)
    allow_unpreferred_mb_suffices: bool, optional
        Whether to allow other suffices than the preferred one at all.
        Default: True (for final operational runs).
    exclude_old_status: pd.Timedelta, None
        If a glacier has not been updated recently, it might still have an
        outdated `mb_current´ file. If set, all glaciers with a status older
        than today-exclude_old_status are omitted. Default:
        pd.Timedelta(days=7), i.e. all status older than a week from today are
        omitted.
    plot_dir: str or None, optional
        Directory where to save the plot. Default: None (do not save).

    Returns
    -------
    None
    """

    if (gdirs is not None) and (shp_loc is not None):
        raise ValueError('Arguments `gdirs` and `shp_loc` are mutually '
                         'exclusive.')

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

    # Add glacier shapes: read shp_loc anyway, if gdirs are not given
    if gdirs is None:
        glc_gdf = gpd.GeoDataFrame.from_file(shp_loc)
        glc_gdf = glc_gdf.sort_values(by='Area', ascending=True)
        gdirs = workflow.init_glacier_regions(glc_gdf)
    else:
        crs_int = 4326
        glc_gdf = gpd.GeoDataFrame(crs='epsg:{}'.format(crs_int),
                                   columns=['geometry'])
        for gd in gdirs:
            ol_path = gd.get_filepath('outlines')
            gd_gdf = gpd.GeoDataFrame.from_file(ol_path)
            gd_gdf = gd_gdf.to_crs(epsg=crs_int)
            glc_gdf = glc_gdf.append(gd_gdf)

    try:
        glc_gdf = gpd.read_file(cfg.PATHS['glacier_status'])
    except FileNotFoundError:
        workflow.fetch_glacier_status(
            gdirs, mb_model=mb_model, shp=glc_gdf,
            prefer_mb_suffix=prefer_mb_suffix,
            allow_unpreferred_mb_suffices=allow_unpreferred_mb_suffices,
            exclude_old_status=exclude_old_status)
    glc_gdf = gpd.read_file(cfg.PATHS['glacier_status'])

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
        'fillOpacity': 0.9,
    }

    def highlight_function(feature):
        """Highlights a feature when hovering over it."""
        return {
            'fillColor': feature['properties']['polycolor'],
            'color': feature['properties']['polycolor'],
            'weight': 3,
            'fillOpacity': 1
        }

    # glc_gdf['popup_html'] = glc_gdf.apply(popup_html_string, axis=1)

    layer_geom = folium.FeatureGroup(name='layer', control=False)

    glc_gjs = glc_gdf.__geo_interface__
    for i in range(len(glc_gjs["features"])):
        temp_geojson = {"features": [glc_gjs["features"][i]],
                        "type": "FeatureCollection"}
        temp_geojson_layer = folium.GeoJson(
            temp_geojson, highlight_function=highlight_function,
            control=False, style_function=style_func2, smooth_factor=0.5)
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
    if plot_dir is not None:
        final_dir = os.path.join(plot_dir, 'status_map')
        if not os.path.isdir(final_dir):
            os.mkdir(final_dir)
        m.save(os.path.join(final_dir,
                            'status_map{}.html'.format(save_suffix)))


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
        raise ValueError('Please give a valid working directory.')

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
    """Just some helper object."""
    pass


class AnyObjectHandler(object):
    """Manages items in a custom legend."""

    def __init__(self, color='b', facecolor='cornflowerblue', linestyle='-',
                 edgecolor=None, edge_linestyle=None, linewidth=None):
        self.facecolor = facecolor
        self.color = color
        self.linestyle = linestyle
        self.edgecolor = edgecolor
        self.edge_linestyle = edge_linestyle
        self.linewidth = linewidth

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        l1 = mlines.Line2D([x0, y0 + width],
                           [0.5 * height, 0.5 * height],
                           linestyle='-', color=self.color)
        patch1 = mpatches.Rectangle((x0, y0 + 0.25 * height), width,
                                    0.5 * height, facecolor=self.facecolor,
                                    alpha=0.5, edgecolor=self.edgecolor,
                                    linestyle=self.edge_linestyle,
                                    transform=handlebox.get_transform(),
                                    linewidth=self.linewidth)
        patch2 = mpatches.Rectangle((x0, y0), width, height,
                                    facecolor=self.facecolor, alpha=0.3,
                                    edgecolor=self.edgecolor,
                                    linestyle=self.edge_linestyle,
                                    transform=handlebox.get_transform(),
                                    linewidth=self.linewidth)
        handlebox.add_artist(l1)
        handlebox.add_artist(patch1)
        handlebox.add_artist(patch2)
        return [l1, patch1, patch2]


@entity_task(log)
def plot_cumsum_climatology_and_current(
        gdir=None, clim=None, current=None, pred_cosmo=None, pred_ecmwf=None,
        suffix='', mb_model=None, clim_begin=None, current_begin=None,
        abs_deviations=False, fs=14, loc=0, plot_dir=None):
    """
    Make the standard plot containing the MB climatology in the background and
    the current MB year on top.

    In leap years, the plot has length 366, otherwise length 365 and DOY 366
    is removed.

    An option should be added that lets you choose the time span of the
    climatology displayed in the background (added to filename and legend!)

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory` or None, optional
        The GlacierDirectory to plot the data for. Mutually exclusive with
        [`clim` and `current`]. Default: None.
    clim: xarray.Dataset or None, optional
        The mass balance climatology. The mass balance variable should be named
        'MB', where MassBalanceModel can be any
        MassBalanceModel and mass balance should have two coordinates (e.g.
        'time' and 'n' (some experiment)). This parameter is mutually exclusive
        with `gdir`. Default: None.
    current: xarray.Dataset or None, optional
        The current year's mass balance. The mass balance variable should be
        named 'MassBalanceModel_MB', where MassBalanceModel can be any mass
        balance model class and mass balance should have two coordinates (e.g.
        'time' and 'n' (some experiment)). This parameter is mutually exclusive
        with `gdir`. Default: None.
    pred_cosmo: xarray.Dataset or None, optional
        Mass balance prediction from COSMO numerical weather predictions.
    pred_ecmwf: xarray.Dataset or None, optional
        Mass balance prediction from ECMWF extended range predictions.
    suffix : str, optional
        Which climate and current files with which suffix exactly to look for.
        Only valid, when `gdir` is passed and thus mutually exclusive with
        [`clim` and `current`]!. Accepted are all suffices that are also
        possible in terms of calibration. If at least one of the desired files
        is not there, an error is logged. Default: '' (no suffix).
    mb_model: str or None, optional
        Mass balance model to make the plot for. must be contained in both the
        mass balance climatology and current mass balance file. Default: None
        (make plot for ensemble mass balance estimate).
    clim_begin: tuple or None, optional
        Tuple of (month, day) when the mass budget year begins. Default: (None,
        None) (take from params).
    current_begin: tuple or None, optional
        Tuple of (month, day) when the mass budget year begins in the current
        mass budget year. Default: (None, None) (take from params).
    abs_deviations: bool, optional
        # TODO: Redo this! There is a bug somewhere
        Whether or not to plot also the current absolute deviations of the
        prediction from the reference median. Size of the figure will be
        adjusted accordingly to make both plots fit. Default: False.
    fs: int, optional
        Font size for title, axis labels and legend. Default: 14.
    loc: int, optional
        Legend position as passed to plt.legend(). Sometimes the placing of the
        legend fails when done automatically, so this helps to keep control.
        Default: 0 ('best' position).
    plot_dir: str or None
        Where to save the plot to. Default: None (do not save).

    Returns
    -------
    None
    """

    # some initial check
    if (gdir is not None) and ((clim is not None) or (current is not None) or
                               (pred_cosmo is not None) or
                               (pred_ecmwf is not None)):
        raise ValueError('Parameters `gdir` and [`clim`, `current`, '
                         '`pred_cosmo`, `pred_ecmwf`] are mutually exclusive.')

    if gdir is not None:
        try:
            clim = gdir.read_pickle('mb_daily', filesuffix=suffix)
            current = gdir.read_pickle('mb_current', filesuffix=suffix)
        except FileNotFoundError:
            # either of the files is not there
            log.info('No MB current plot generated for {} (not both mb_daily{}'
                     ' and mb_current{} files found).'.format(gdir.rgi_id,
                                                             suffix, suffix))
            return
        # be gentle here:
        no_pred_msg = 'No mass balance prediction from {} data found for {}.'
        try:
            pred_cosmo = gdir.read_pickle('mb_prediction_cosmo',
                                          filesuffix=suffix)
        except FileNotFoundError:
            log.warning(no_pred_msg.format('COSMO', gdir.id))
        try:
            pred_ecmwf = gdir.read_pickle('mb_prediction_ecmwf',
                                          filesuffix=suffix)
        except FileNotFoundError:
            log.warning(no_pred_msg.format('ECMWF', gdir.id))
    else:
        pass

    if clim_begin is None:
        clim_begin = (cfg.PARAMS['bgmon_hydro'], cfg.PARAMS['bgday_hydro'])
    if current_begin is None:
        current_begin = (cfg.PARAMS['bgmon_hydro'], cfg.PARAMS['bgday_hydro'])

    if mb_model is not None:
        try:
            clim = clim.sel(model=mb_model)
            current = current.sel(model=mb_model)
        except KeyError:
            log.error(
                'Could not make mass balance distribution plot for {}. "{}" '
                'needs to be contained in both mass balance climatology and '
                'current mass balance.'.format(clim.rgi_id, mb_model))

    # make cumsum and quantiles
    cq = clim.mb.make_cumsum_quantiles(bg_month=clim_begin[0],
                                       bg_day=clim_begin[1])
    nq = current.mb.make_cumsum_quantiles(bg_month=current_begin[0],
                                          bg_day=current_begin[1])
    pred_outdated_msg = \
        '{} prediction is outdated (last day of current MB is later than ' \
        'prediction start). Continuing without prediction...'
    if pred_cosmo is not None:
        pq_cosmo = pred_cosmo.mb.make_cumsum_quantiles(
            bg_month=current_begin[0], bg_day=current_begin[1])

        # make sure current and prediction do not overlap
        pq_cosmo = pq_cosmo.sel(hydro_doys=slice(nq.hydro_doys.max(), None))
        if pq_cosmo.hydro_doys.size == 0.:
            log.warning(pred_outdated_msg.format('COSMO'))
            pred_cosmo = None
            pq_cosmo = None
    if pred_ecmwf is not None:
        pq_ecmwf = pred_ecmwf.mb.make_cumsum_quantiles(
            bg_month=current_begin[0], bg_day=current_begin[1])

        # make sure current and prediction do not overlap
        pq_ecmwf = pq_ecmwf.sel(hydro_doys=slice(nq.hydro_doys.max(), None))
        if pq_ecmwf.hydro_doys.size == 0.:
            log.warning(pred_outdated_msg.format('ECMWF'))
            pred_ecmwf = None
            pq_ecmwf = None

    # todo: kick out all variables except MB (e.g. saved along cali params)
    if abs_deviations:
        fig, (ax, ax2) = plt.subplots(2, figsize=(10, 5),
                                      gridspec_kw={'height_ratios': [7, 1]})
    else:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax2 = None

    # Say if time on x-axis is 365 or 366 long
    xtime = pd.date_range(
        dt.datetime(current.isel(time=0).time.dt.year.item(), 10, 1),
        periods=366)
    if ((xtime.month == 2) & (xtime.day == 29)).any():
        xvals = np.arange(366)
    else:
        xtime = xtime[:-1]
        xvals = np.arange(365)
        cq = cq.sel(hydro_doys=slice(None, 365))
        if pred_cosmo is not None:
            pq_cosmo = pq_cosmo.sel(hydro_doys=slice(None, 365))
        # todo: extend plot over the boundaries of the end of the MB year
        if pred_ecmwf is not None:
            pq_ecmwf = pq_ecmwf.sel(hydro_doys=slice(None, 365))

    # Say how much we would have to pad in front/back to match this time frame
    pad_f = (pd.Timestamp(current.isel(time=0).time.item()) -
             dt.datetime(current.isel(time=0).time.dt.year.item(),
                         clim_begin[0], clim_begin[1])).days

    # for ivar, (mbname, mbvar) in enumerate(current.data_vars.items()):
    if pad_f >= 0:
        pad_b = len(xvals) - current.MB.shape[0] - pad_f
        mb_now_cs_pad = np.lib.pad(nq.MB.values,
                                   ((pad_f, pad_b), (0, 0)),
                                   'constant',
                                   constant_values=(np.nan, np.nan))
    else:
        # todo: clip at the begin necessary if curr_beg different from clim_beg
        # current_clip = nq.isel(time=slice(
        #    dt.datetime(current.isel(time=0).time.dt.year.item(),
        #                clim_begin[0], clim_begin[1]), None))
        # mb_now_cs_pad = np.lib.pad(current_clip.MB.values, (
        # (0, len(xvals) - current_clip.MB.shape[0]), (0, 0)),
        #                           'constant',
        #                           constant_values=(np.nan, np.nan))
        mb_now_cs_pad = np.lib.pad(
            nq.MB.values, ((0, len(xvals) - nq.MB.shape[0]), (0, 0)),
            'constant', constant_values=(np.nan, np.nan))

    # pad prediction easier - might work for others as well?
    if pred_cosmo is not None:
        pq_cosmo_pad = np.full((len(xvals), pq_cosmo.MB.shape[1]), np.nan)
        pq_cosmo_pad[pq_cosmo.hydro_doys.values - 1, :] = pq_cosmo.MB.values
        # just in case there is an overlap
        pq_cosmo_pad[:nq.hydro_doys.values[-1], :] = np.nan

        # shift prediction to the last quantiles of current
        last_day_mb_current = mb_now_cs_pad[nq.MB.shape[0] - 1, :]
        pq_cosmo_pad += np.atleast_2d(last_day_mb_current)
        # insert last day of mb_current as well (make visual transition nice)
        pq_cosmo_pad[nq.hydro_doys.values[-1] - 1, :] = last_day_mb_current
    if pred_ecmwf is not None:
        pq_ecmwf_pad = np.full((len(xvals), pq_ecmwf.MB.shape[1]), np.nan)
        pq_ecmwf_pad[pq_ecmwf.hydro_doys.values - 1, :] = pq_ecmwf.MB.values
        # just in case there is an overlap
        pq_ecmwf_pad[:nq.hydro_doys.values[-1], :] = np.nan

        # shift prediction to the last quantiles of current
        last_day_mb_current = mb_now_cs_pad[nq.MB.shape[0] - 1, :]
        pq_ecmwf_pad += np.atleast_2d(last_day_mb_current)
        # insert last day of mb_current as well (make visual transition nice)
        pq_ecmwf_pad[nq.hydro_doys.values[-1] - 1, :] = \
            last_day_mb_current

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
    # plot MB prediction - ECMWF first, so that COSMO is on top
    ecmwf_color_ix = 1
    if pred_ecmwf is not None:
        p5, = ax.plot(xvals, pq_ecmwf_pad[:, 2],
                      c=FCST_COLORS[ecmwf_color_ix][0], label='Median')
        ax.fill_between(xvals, pq_ecmwf_pad[:, 1], pq_ecmwf_pad[:, 3],
                        facecolor=FCST_COLORS[ecmwf_color_ix][1], alpha=0.5)
        # plot 10th to 90th pctl
        ax.fill_between(xvals, pq_ecmwf_pad[:, 0], pq_ecmwf_pad[:, 4],
                        facecolor=FCST_COLORS[ecmwf_color_ix][1], alpha=0.3)
    cosmo_color_ix = 0
    if pred_cosmo is not None:
        p6, = ax.plot(xvals, pq_cosmo_pad[:, 2],
                      c=FCST_COLORS[cosmo_color_ix][0], label='Median')
        ax.fill_between(xvals, pq_cosmo_pad[:, 1], pq_cosmo_pad[:, 3],
                        facecolor=FCST_COLORS[cosmo_color_ix][1], alpha=0.5)
        # plot 10th to 90th pctl
        ax.fill_between(xvals, pq_cosmo_pad[:, 0], pq_cosmo_pad[:, 4],
                        facecolor=FCST_COLORS[cosmo_color_ix][1], alpha=0.3)
    ax.set_xlabel('Months', fontsize=16)
    ax.set_ylabel('Cumulative Mass Balance (m w.e.)', fontsize=fs)
    ax.set_xlim(xvals.min(), xvals.max())
    plt.tick_params(which='major', labelsize=fs)
    mbeg = xtime[np.where(xtime.day == 1)]
    if xtime[0].day == 1:
        xtpos_init = [0]
        month_int = mbeg.month
    else:
        xtpos_init = [(xtime[0].replace(month=xtime[0].month + 1, day=1) -
                       xtime[0]).days]
        month_int = mbeg.month
    xtpos = np.cumsum(xtpos_init +
                      [(mbeg[i]-mbeg[i-1]).days for i in range(1, len(mbeg))])
    ax.xaxis.set_ticks(xtpos)
    ax.xaxis.set_ticks(xvals, minor=True)
    mstr = 'JFMAMJJAS0ND'
    ax.set_xticklabels([mstr[i-1] for i in month_int], fontsize=fs)
    ax.grid(True, which='both', alpha=0.5)
    plt.suptitle(
        'Daily Cumulative MB Distribution of {} ({})'
        .format(clim.attrs['id'].split('.')[1], clim.attrs['name']),
        fontsize=fs)

    # todo: get labels correct for plotting more mb models
    entry_one = AnyObject()
    entry_two = AnyObject()

    handles = [entry_one, entry_two]
    labels = ['Climatology Median, IQR, IDR', 'Current Year Median, IQR, IDR']
    handler_map = {
        entry_one: AnyObjectHandler(
            color=CLIM_COLORS[model_ix][0],
            facecolor=CLIM_COLORS[model_ix][1]),
        entry_two: AnyObjectHandler(
            color=CURR_COLORS[model_ix][0],
            facecolor=CURR_COLORS[model_ix][1])
    }

    if pred_cosmo is not None:
        entry_three = AnyObject()
        handles.append(entry_three)
        labels.append('COSMO Prediction Median, IQR, IDR')
        handler_map.update({entry_three: AnyObjectHandler(
            color=FCST_COLORS[cosmo_color_ix][0],
            facecolor=FCST_COLORS[cosmo_color_ix][1])})
    if pred_ecmwf is not None:
        entry_three = AnyObject()
        handles.append(entry_three)
        labels.append('ECMWF Prediction Median, IQR, IDR')
        handler_map.update({entry_three: AnyObjectHandler(
            color=FCST_COLORS[ecmwf_color_ix][0],
            facecolor=FCST_COLORS[ecmwf_color_ix][1])})

    ax.legend(handles, labels, fontsize=fs, loc=loc, handler_map=handler_map)
    # say when we have updated
    date_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    bprops = dict(facecolor='w', alpha=0.5)
    plt.text(0.05, 0.05, 'Last updated: {}'.format(date_str),
             transform=fig.transFigure, bbox=bprops)

    # just temporary/provisionally (hopefully): date when SIS delivery ended
    sis_end = np.nan
    if gdir is not None:
        sis = GlacierMeteo(gdir).meteo.sis.sel(time=slice(
            current.time.values[0], None))
        sis_end = pd.Timestamp(sis.isel(time=sis.notnull().argmin(
            dim='time').item()).time.values)
    else:  # try the hard way
        potential_sis_end = []
        try_sis_models = ['PellicciottiModel', 'OerlemansModel']
        for sism in try_sis_models:
            try:
                se = pd.Timestamp(current.isel(time=current.sel(
                    model=sism).MB.notnull().argmin(
                    dim='time').max().item()).time.values)
                potential_sis_end.append(se)
            except KeyError:
                pass  # model not in file
        if len(potential_sis_end) > 0:
            sis_end = np.max(potential_sis_end)
    if ~pd.isnull(sis_end):
        ax.axvline((sis_end - current.time.values[0]).days, c='y')
        ax.text((sis_end - current.time.values[0]).days + 1, plt.ylim()[0] + 0.05,
                'End radiation delivery', transform=ax.transData, va='bottom',
                fontsize=fs, c='y')

    if abs_deviations:
        clim_med = clim.MB.values[:, 2][:len(current.MB.values)]
        # from cumsum back to actual MBs (stupid!)
        clim_mbs = np.append(
            np.array([clim_med[0]]), np.array(clim_med[1:] - clim_med[:-1]),
            axis=0)
        curr_mbs = np.append(
            np.atleast_2d(current.MB.values[0]),
            current.MB.values[1:] - current.MB.values[:-1], axis=0)

        # we take only the median
        clim_acc = np.sum(clim_mbs[clim_mbs > 0.])
        clim_abl = np.sum(clim_mbs[clim_mbs < 0.])
        # here we take all PCTLs
        curr_acc = np.sum(np.clip(curr_mbs, 0., None), axis=0)
        curr_abl = np.sum(np.clip(curr_mbs, None, 0.), axis=0)
        barlist = ['Accumulation', 'Ablation', 'Budget']
        bar_ypos = np.arange(len(barlist))
        # absolute devs
        devs = np.concatenate((np.atleast_2d(curr_acc - clim_acc),
                               np.atleast_2d(curr_abl - clim_abl),
                               np.atleast_2d(current.MB.values[-1, :] -
                                             clim_med[-1])), axis=0)
        if ax2 is not None:
            ax2.barh(bar_ypos, devs[:, 4], left=devs[:, 0], color='g',
                     edgecolor='g', alpha=0.3)
            ax2.barh(bar_ypos, devs[:, 3], left=devs[:, 1], color='g',
                     edgecolor='g', alpha=0.5)
            ax2.barh(bar_ypos, np.repeat(.1, 3), left=devs[:, 2], color='g',
                     edgecolor='g')
            maxabsdev = np.nanmax(np.abs(devs))
            nextnumber = 10 ** np.ceil(np.log10(maxabsdev))
            ax2.set_xlim([-nicenumber(maxabsdev, nextnumber),
                          nicenumber(maxabsdev, nextnumber)])

            ax2.set_yticks(bar_ypos)
            ax2.set_yticklabels(barlist, fontsize=12)
            ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.set_xlabel('Deviation of PCTLs from climatology (mm w.e.)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # include suptitle

    if gdir is not None:
        if plot_dir is None:
            return
        if not os.path.exists(os.path.join(plot_dir, 'mb_dist')):
            os.makedirs(os.path.join(plot_dir, 'mb_dist'))
        bname = os.path.join(plot_dir, 'mb_dist',
                             gdir.rgi_id.split('.')[1] + '_')
        plt.savefig(bname + 'mb_dist_{}.png'.format('ensemble' + suffix),
                    dpi=1000)
        plt.savefig(bname + 'mb_dist_{}_prev.png'.format('ensemble' + suffix),
                    dpi=40)


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
    ani: matplotlib.animation.FuncAnimation
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
        """Animate function as input for `FuncAnimation`."""
        line.set_ydata(data.snow[i] / 1000.)  # update the data
        time_text.set_text(data.time_elapsed[i].strftime("%Y-%m-%d"))
        return line,

    ani = animation.FuncAnimation(fig, animate,
                                  frames=mb_model.snow.shape[0],
                                  fargs=(mb_model, line, time_text),
                                  interval=10, repeat_delay=1.)
    return ani


@entity_task(log)
def plot_interactive_mb_spaghetti_html(
        gdir: utils.GlacierDirectory, plot_dir: Optional[str] = None,
        mb_models: Optional[list] = None, fontsize: Optional[int] = 18,
        show: Optional[bool] = False):
    """
    Makes an interactive spaghetti plot of all available mass balance years.

    When hovering over the plot, the cumulative mass balance curves are
    highlighted and the value is displayed as tooltip.

    Parameters
    ----------
    gdir: :py:class:`crampon.GlacierDirectory`
        The glacier directory for which to produce the plot.
    plot_dir: str or None, optional
        Directory where to store the plot (HTML file). Default: None (do not
        save).
    mb_models: list of crampon.core.models.massbalance.DailyMassbalanceModel or
        None, optional
        Mass balance models to plot. If None, all model in `mb_clim` and
        `mb_current`, respectively, are used.
    fontsize: int, optional
        Basic font size to use. Default: 18.
    show: bool
        whether to show the plot once done (i.e. open the HTML file in a
        browser). Default: False (not handy during the operational runs).

    Returns
    -------
    None
    """

    try:
        mb_clim = gdir.read_pickle('mb_daily')
        mb_curr = gdir.read_pickle('mb_current')
    except FileNotFoundError:
        return

    if mb_models is not None:
        mb_clim = mb_clim.sel(model=mb_models)
        mb_curr = mb_curr.sel(model=mb_models)

    arr = None
    mbcarr = None
    years = []
    mbcyears = []
    models = []

    hydro_years = mb_clim.mb.make_hydro_years()
    for y, group in list(mb_clim.groupby(hydro_years)):
        # make use of static method
        to_plot = MassBalance.time_cumsum(group)
        # todo: EMERGENCY SOLUTION since we can't calc. CS with NaNs correctly
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

    # CurrentYearMassBalance cut off radiation models when SIS is not delivered
    mb_curr_cs = CurrentYearMassBalance.time_cumsum(mb_curr)
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
    current_colors = [c for c, _ in CURR_COLORS[:len(mbcyears)]]
    if greyshades > 256:  # the length of the palette
        custompalette = []
        for n in range(int(np.floor(greyshades / 256.))):
            custompalette.extend(grey(256))

        custompalette.extend(grey(int(greyshades % 256.)))
        custompalette.extend(current_colors)
    else:
        custompalette = grey(greyshades) + tuple(current_colors)

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
        date=np.repeat(
            np.atleast_2d(
                np.array(pd.date_range(mb_curr.time[0].item(), freq='D',
                                    periods=mbcarr.shape[1]),
                      dtype=np.datetime64
                         )), len(xs), axis=0)
    ))

    plot = bkfigure(plot_width=1200, plot_height=800)
    plot.title.text = 'Cumulative Mass Balance of {} ({})'.\
        format(mb_clim.attrs['id'].split('.')[1], mb_clim.attrs['name'])
    plot.title.text_font_size = '{}pt'.format(fontsize)

    plot.xaxis.axis_label = 'Days of Hydrological Year'
    plot.xaxis.axis_label_text_font_size = '{}pt'.format(fontsize)
    plot.yaxis.axis_label = 'Cumulative Mass Balance'
    plot.yaxis.axis_label_text_font_size = '{}pt'.format(fontsize)

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
    plot.xaxis.major_label_text_font_size = '{}pt'.format(fontsize)
    plot.yaxis.major_label_text_font_size = '{}pt'.format(fontsize)
    plot.grid.ticker = FixedTicker(ticks=xticks)

    r = plot.multi_line('xs', 'ys', source=source,
                        color='color', alpha='alpha', line_width=4,
                        hover_line_alpha=1.0, hover_line_color='color')

    legendentries = ['Past mass balances'] + list(mb_curr_cs.model.values)

    def leg_idx(x):
        """Legend index."""
        if x != 0:
            return int((len(years) - len(mbcyears)) + x - 1)
        else:
            return int(len(years) / 2.)

    legend = Legend(
        items=[LegendItem(label=le, renderers=[r], index=leg_idx(i)) for
               i, le in enumerate(legendentries)],
        title_text_font_size='{}pt'.format(fontsize),
        label_text_font_size='{}pt'.format(fontsize))
    plot.add_layout(legend)
    plot.legend.location = "top_left"

    TOOLTIPS = [
        ("date", "@date{%F}"),
        ("year", "@desc"),
        ("(HYD-DOY,CUM-MB)", "($x{0.}, $y)"),
        ("MODEL", "@model"),
    ]
    plot.add_tools(HoverTool(
        tooltips=TOOLTIPS,
        mode='mouse',
        formatters={'date': 'datetime'}
    ))

    # Add a note to say when it was last updated
    date_str = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    update_txt = Label(x=3., y=3., x_units='screen', y_units='screen',
                       border_line_color='black', border_line_alpha=1.0,
                       background_fill_color='white',
                       background_fill_alpha=0.5,
                       text='Last updated: {}'.format(date_str)
                       )
    plot.add_layout(update_txt)

    plot.add_tools(UndoTool())
    plot.add_tools(RedoTool())

    pure_id = gdir.rgi_id.split('.')[1]
    if plot_dir is not None:
        if not os.path.exists(os.path.join(plot_dir, 'mb_spaghetti')):
            os.makedirs(os.path.join(plot_dir, 'mb_spaghetti'))
        output_file(os.path.join(plot_dir, 'mb_spaghetti', pure_id +
                                 '_interactive_mb_spaghetti.html'))
        export_png(plot, filename=os.path.join(
            plot_dir, 'mb_spaghetti',
            pure_id + '_interactive_mb_spaghetti.png'))
        save(plot)
    if show is True:
        bkshow(plot)


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
    smap.set_shapefile(os.path.join(cfg.PATHS['data_dir'], 'outlines',
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
        try:
            t_str = pd.Timestamp(time[i]).strftime('%M-%D %m-%d')
        except Exception:
            t_str = pd.Timestamp(time[i]).strftime('%m-%d')

        t = ax.annotate(t_str, (0.01, 0.98), xycoords='axes fraction',
                        verticalalignment='top', color='k')
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

    From https://bit.ly/3olKhve.

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

    From https://bit.ly/2Xu2Cup.

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
    **textkw: dict
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

    # exclude those numbers where correlation is not significant
    """
    mat = df.values.T
    K = len(df.columns)
    correl = np.empty((K, K), dtype=float)
    p_vals = np.empty((K, K), dtype=float)

    for i, ac in enumerate(mat):
        for j, bc in enumerate(mat):
            if i > j:
                continue
            else:
                corr = stats.pearsonr(ac, bc)
                # corr = stats.kendalltau(ac, bc)

            correl[i, j] = corr[0]
            correl[j, i] = corr[0]
            p_vals[i, j] = corr[1]
            p_vals[j, i] = corr[1]
    """

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
    pval_thresh: float, optional
        Threshold value for the p-value of the correlation. Default: 0.01.

    Returns
    -------
    None
    """

    corr = df.corr()

    # get the p-values (pd doesn't take care of them), make use of method kw
    def pearsonr_pval(x, y):
        """Get p-value of a correlation."""
        return pearsonr(x, y)[1]
    pvals = df.corr(method=pearsonr_pval)

    corr[pvals > pval_thresh] = np.nan

    im, _ = heatmap(corr.values, corr.columns.values, corr.columns.values,
                    cmap="RdBu_r", vmin=-1, vmax=1,
                    cbarlabel="correlation coeff.")

    def func(x, pos):
        """Format displayed values."""
        return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")

    annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)

    plt.tight_layout()
    plt.show()


def plot_holfuy_station_availability(fontsize=14, barheight=0.8):
    """
    Plot horizontal bars that indicate when the Holfuy stations were available.

    Colors tell which surface type (snow/ice) prevailed when and when there
    were failures.

    Parameters
    ----------
    fontsize: int
        Font size to be used for annotations. Default: 14.
    barheight : float
        Height of the bars. Default: 0.8 (as in matplotlib).

    Returns
    -------
    None
    """

    def add_index(ixlist, g, s):
        """Tell where the bars shall be placed on the y-axis."""
        ix = None
        if g == 'RGI50-11.A55F03':
            ix = 0
        if g == 'RGI50-11.B4312n-1':
            if s == 2233.:
                ix = 0.5
            if s == 2235.:
                ix = 1
            if s == 2392.:
                ix = 1.5
            if s == 2589.:
                ix = 2.
        if g == 'RGI50-11.B5616n-1':
            if s == 2564.:
                ix = 2.5
            if s == 3021.:
                ix = 3.
        ixlist.append(ix)
        return ixlist

    label_list = ['PLM 1', 'RHO 1', 'RHO 2', 'RHO 3',
                  'RHO 4', 'FIN 1', 'FIN 2']

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
        gdir = utils.GlacierDirectory(
            id, base_dir=os.path.join(cfg.PATHS['working_dir'], 'per_glacier'))
        obs_merge = holfuytools.prepare_holfuy_camera_readings(
            gdir, ice_only=False, exclude_keywords=False,
            exclude_initial_snow=False)
        for h in obs_merge.height.values:
            station_phase = obs_merge.sel(height=h).phase
            station_swe = obs_merge.sel(height=h).swe
            station_phase_i = station_phase.where((station_phase == 'i') &
                                                  (~pd.isnull(station_swe)),
                                                  drop=True)
            station_phase_s = station_phase.where((station_phase == 's') &
                                                  (~pd.isnull(station_swe)),
                                                  drop=True)
            station_phase_nan = station_phase[pd.isnull(station_swe)]
            ice_beg.append(station_phase_i.date.values[0])
            ice_ixs = add_index(ice_ixs, id, h)
            for i in range(1, len(station_phase_i.date.values)):
                if station_phase_i.date.values[i - 1] + \
                        pd.Timedelta(days=1) == station_phase_i.date.values[i]:
                    pass
                else:
                    ice_end.append(station_phase_i.date.values[i - 1])
                    ice_beg.append(station_phase_i.date.values[i])
                    ice_ixs = add_index(ice_ixs, id, h)
            ice_end.append(station_phase_i.date.values[-1])

            snow_beg.append(station_phase_s.date.values[0])
            snow_ixs = add_index(snow_ixs, id, h)
            print('SNOW DAYS: ', str(station_phase_s.date.values.size))

            for i in range(1, len(station_phase_s.date.values)):
                if station_phase_s.date.values[i - 1] + pd.Timedelta(days=1) \
                        == station_phase_s.date.values[i]:
                    pass
                else:
                    snow_end.append(station_phase_s.date.values[i - 1])
                    snow_beg.append(station_phase_s.date.values[i])
                    snow_ixs = add_index(snow_ixs, id, h)
            snow_end.append(station_phase_s.date.values[-1])

            if len(station_phase_nan) > 0:
                # If Nan at begin, TS has been expanded to go in one xr.Dataset
                beg_ix = 0
                if station_phase_nan.date.values[beg_ix] == \
                        obs_merge.date.values[0]:
                    while station_phase_nan.date.values[beg_ix] + \
                            pd.Timedelta(days=1) == \
                            station_phase_nan.date.values[beg_ix+1]:
                        beg_ix += 1
                other_beg.append(station_phase_nan.date.values[beg_ix])
                other_ixs = add_index(other_ixs, id, h)
                for i in range(beg_ix+1, len(station_phase_nan.date.values)):
                    if station_phase_nan.date.values[i - 1] + pd.Timedelta(
                            days=1) == station_phase_nan.date.values[i]:
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

    fig, ax = plt.subplots(figsize=(12, 4))
    # Plot the data
    ax.barh(snow_ixs + 1, snow_end - snow_beg, left=snow_beg, height=barheight,
            align='center', facecolor='w', edgecolor='k', label='snow')
    ax.barh(ice_ixs + 1, ice_end - ice_beg, left=ice_beg, height=barheight,
            align='center', facecolor='b', edgecolor='k', label='ice')
    ax.barh(other_ixs + 1, other_end - other_beg, left=other_beg,
            height=barheight, align='center', alpha=0.2, facecolor='r',
            edgecolor='k', label='fail')

    # make separating lines between the glaciers
    ax.axhline(1.25, c='k')
    ax.axhline(3.25, c='k')

    ax.xaxis_date()
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d.%m.'))
    ax.set_xlabel('Year 2019', fontsize=fontsize)
    ax.yaxis.set_ticklabels([''] + label_list, fontsize=fontsize)
    ax.set_xlim(mdates.date2num(pd.Timestamp('2019-06-15')),
                mdates.date2num(pd.Timestamp('2019-10-15')))
    ax.set_title('Station availability 2019', fontsize=fontsize)

    # make ticks thicker
    ax.xaxis.set_tick_params(width=5)
    ax.yaxis.set_tick_params(width=5)

    plt.legend(fontsize=fontsize, loc='upper right')
    plt.show()


class GoogleVisibleMapCrampon(salem.GoogleCenterMap):
    """Google static map automatically sized and zoomed to a selected region.

    It's usually more practical to use than GoogleCenterMap.
    """

    def __init__(self, x, y, crs=salem.wgs84, size_x=640, size_y=640, scale=1,
                 maptype='satellite', use_cache=True, zoom=20, **kwargs):
        """Initialize

        Parameters
        ----------
        x : array
          x coordinates of the points to include on the map
        y : array
          y coordinates of the points to include on the map
        crs : proj or Grid
          coordinate reference system of x, y
        size_x : int
          image size
        size_y : int
          image size
        scale : int
          image scaling factor. 1, 2. 2 is higher resolution but takes
          longer to download
        maptype : str, default: 'satellite'
          'roadmap', 'satellite', 'hybrid', 'terrain'
        use_cache : bool, default: True
          store the downloaded image in the cache to avoid future downloads
        kwargs : **
          any keyword accepted by motionless.CenterMap (e.g. `key` for the API)

        Notes
        -----
        To obtain the exact domain specified in `x` and `y` you may have to
        play with the `size_x` and `size_y` kwargs.
        """

        global API_KEY
        import pyproj

        if 'center_ll' in kwargs:
            raise ValueError('incompatible kwargs.')

        # Transform to lonlat
        crs = salem.gis.check_crs(crs)
        if isinstance(crs, pyproj.Proj):
            lon, lat = gis.transform_proj(crs, salem.wgs84, x, y)
        elif isinstance(crs, salem.Grid):
            lon, lat = crs.ij_to_crs(x, y, crs=salem.wgs84)
        else:
            raise NotImplementedError()

        # surely not the smartest way to do but should be enough for now
        mc = (np.mean(lon), np.mean(lat))
        while zoom >= 0:
            grid = salem.gis.googlestatic_mercator_grid(
                center_ll=mc, nx=size_x, ny=size_y, zoom=zoom, scale=scale)
            dx, dy = grid.transform(lon, lat, maskout=True)
            if np.any(dx.mask):
                zoom -= 1
            else:
                break

        if 'key' not in kwargs:
            if API_KEY is None:
                with open(utils.get_demo_file('.api_key'), 'r') as f:
                    API_KEY = f.read().replace('\n', '')
            kwargs['key'] = API_KEY

        salem.GoogleCenterMap.__init__(
            self, center_ll=mc, size_x=size_x, size_y=size_y, zoom=zoom,
            scale=scale, maptype=maptype, use_cache=use_cache, **kwargs)


def camera_station_map(holfuy_data_path):
    """
    Make a map of the camera stations.

    Parameters
    ----------
    holfuy_data_path: str
        Path to the file called `holfuy_data.csv`, e.g.
        '..\\documents\\holfuyretriever\\holfuy_data.csv'.

    Returns
    -------

    """
    holfuy_df = pd.read_csv(holfuy_data_path)
    holfuy_df['geometry'] = list(zip(holfuy_df.Easting, holfuy_df.Northing))
    holfuy_df['geometry'] = holfuy_df['geometry'].apply(Point)
    gdf = gpd.GeoDataFrame(holfuy_df, geometry='geometry',
                           crs=21781)
    # for whatever reason, the crs needs extra invitation
    gdf.crs = {'init': 'epsg:21781'}
    points = gdf.to_crs(epsg=4326).copy()

    g = GoogleVisibleMap(x=np.array([6.3, 9.8]), y=np.array([46.2, 46.8]),
                         scale=2, size_x=640, size_y=480)
    ggl_img = g.get_vardata()

    fig, ax = plt.subplots()
    smap = Map(g.grid, factor=1, countries=False)
    smap.set_rgb(ggl_img)
    smap.set_scale_bar(
        location=(0.88, 0.94), add_bbox=True, bbox_kwargs={'facecolor': 'w'},
        text_kwargs={'fontsize': 14})  # add scale
    smap.set_shapefile(
        os.path.join(cfg.PATHS['data_dir'], 'outlines',
                     'VEC200_LANDESGEBIET_LV03.shp'),
        edgecolor='yellow', facecolor='None')
    smap.visualize(ax=ax)
    points['grid_x'], points['grid_y'] = smap.grid.transform(
        points.geometry.x.values, points.geometry.y.values)
    ax.scatter(points.grid_x, points.grid_y, color='r', s=80)
    plt.setp(ax.get_xticklabels(), fontsize=20)
    plt.setp(ax.get_yticklabels(), fontsize=20)
    plt.show()


def multicolor_axislabel(ax, text_list, color_list, axis='x', anchorpad=0,
                         **kwargs):
    """
    Create axes labels with multiple colors.

    Edited from [1]_.

    Parameters
    ----------
    ax: matplotlib.Axis
        Axes object where the labels should be drawn
    text_list: list of str
        List of all the text items.
    color_list: list of str
        Corresponding list of colors for the text.
    axis: str
        Defines the labels to be drawn. Possible: 'x', 'y', or 'both'.
        Default: 'x'.
    anchorpad: float, optional
        Some pad value for the label box (?). Default: 0.
    **kwargs: dict
        Text properties as dictionary.

    References
    ----------
    .. [1]  https://bit.ly/3hr1Abc
    """

    # x-axis label
    if axis == 'x' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',
                                               va='bottom', **kwargs))
                 for text, color in zip(text_list, color_list)]
        xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad,
                                          frameon=False,
                                          bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes,
                                          borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis == 'y' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left',
                                               va='bottom', rotation=90,
                                               **kwargs))
                 for text, color in zip(text_list[::-1], color_list[::-1])]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(
            loc=3, child=ybox, pad=anchorpad, frameon=False,
            bbox_to_anchor=(-0.05, 0.2), bbox_transform=ax.transAxes,
            borderpad=0.)
        ax.add_artist(anchored_ybox)


def overview_camera_mb_cumsum(fontsize=32):
    """
    Plot an overview of the cumulative mass balance at the Holfuy cameras.

    # todo: at the moment this is super unflexible: only 2019, no snow etc

    Parameters
    ----------
    fontsize: int, optional
        Font size to use for the axis labels. Default: 20.

    Returns
    -------
    None
    """
    from crampon.core import holfuytools
    fig, (ax1, ax3, ax5) = plt.subplots(
        3, gridspec_kw={'height_ratios': [6, 3, 1.9]}, sharex=True,
        figsize=(132, 99))

    fig, ax2 = plt.subplots()
    colors = ['b', 'orange', 'g']
    types = ['-', '--', '-.', ':']
    paper_names = {
        ('Rhonegletscher', 2233): 'RHO 1',
        ('Rhonegletscher', 2235): 'RHO 2',
        ('Rhonegletscher', 2392): 'RHO 3',
        ('Rhonegletscher', 2589): 'RHO 4',
        ('Findelengletscher', 2564): 'FIN 1',
        ('Findelengletscher', 3021): 'FIN 2',
        ('Glacier De La Plaine Morte', 2681): 'PLM 1',
    }

    ax4 = None

    for i, sid in enumerate(holfuytools.id_to_station.keys()):
        gdir = utils.GlacierDirectory(
            sid,
            base_dir=os.path.join(cfg.PATHS['working_dir'], 'per_glacier'))
        obs_merge = holfuytools.prepare_holfuy_camera_readings(
            gdir, ice_only=False, holfuy_path='.\\holfuyretriever\\')

        for j, h in enumerate(sorted(obs_merge.height.values)):
            to_plot_x = obs_merge.sel(height=h).date.values
            to_plot_y = obs_merge.sel(height=h).swe.cumsum().values
            to_plot_nocumsum = obs_merge.sel(height=h).swe.values

            # mask the time when cumulative sum is not increasing
            mask = np.logical_or(np.isnan(obs_merge.sel(height=h).swe.values),
                                 (obs_merge.sel(height=h).swe.cumsum(
                                 ).values == 0.))

            # to_plot_x[mask] = np.nan
            to_plot_y[mask] = np.nan

            gname = gdir.name
            plot_name = gname.split('*')[0].strip() if '*' in gname else gname
            ax1.plot(to_plot_x[to_plot_y != 0.], to_plot_y[to_plot_y != 0.],
                     label='{} ({}m.a.s.l.)'.format(paper_names[(plot_name, h
                                                                 )], h),
                     c=colors[i],
                     ls=types[j])
            ax2.plot(to_plot_x[to_plot_y != 0.],
                     # range(len(to_plot_y[to_plot_y != 0.])),
                     to_plot_y[to_plot_y != 0.],
                     label='{} ({}m.a.s.l.)'.format(paper_names[(plot_name, h
                                                                 )], h),
                     c=colors[i],
                     ls=types[j])

            if h == 2564.:  # station 1008

                # todo: delete
                #climate.process_custom_climate_data_crampon(gdir)

                gmet = xr.open_dataset(gdir.get_filepath(
                    'climate_daily')).sel(time=to_plot_x)
                # ax3.plot(to_plot_x, t_air.temp, linewidth=2, color='r')
                # ax3.fill_between(to_plot_x, t_air.tmin, t_air.tmax,
                #                 color='r', alpha=0.3)
                gmeteo = GlacierMeteo(gdir)
                station_tmean_deg_c = [gmeteo.get_tmean_for_melt_at_heights(
                    d, heights=np.array([h])) for d in to_plot_x]
                station_tmean_k = np.array(station_tmean_deg_c) + \
                    cfg.ZERO_DEG_KELVIN
                sis_scale_fac = xr.open_dataarray(gdir.get_filepath(
                    'sis_scale_factor')).values
                heights, _ = gdir.get_inversion_flowline_hw()
                fl_id = np.argmin(np.abs((h - heights)))
                doys = [pd.Timestamp(d).dayofyear for d in to_plot_x]
                ssf = sis_scale_fac[fl_id, doys]
                station_sis = gmet.sis.sel(time=to_plot_x).values
                station_sis_scaled = station_sis * ssf
                ax3.plot(to_plot_x,
                         (station_tmean_k - np.min(station_tmean_k)) /
                         np.ptp(station_tmean_k), linewidth=2, color='r',
                         label='$T_{mean}$')
                heat_wave_mask = ((station_tmean_k - np.min(
                    station_tmean_k))/np.ptp(station_tmean_k)) >= 0.8
                print(((station_tmean_k - np.min(
                    station_tmean_k))/np.ptp(station_tmean_k)))
                print(heat_wave_mask)
                print(to_plot_x)
                #print('Heat wave dates:', to_plot_x[heat_wave_mask])
                ax3.plot(to_plot_x, (station_sis_scaled - np.min(
                    station_sis_scaled)) / np.ptp(
                    station_sis_scaled), linewidth=2, color='g',
                         label='G')
                handles, labels = ax3.get_legend_handles_labels()
                handles.append(mpatches.Patch(color='none',
                                              label='at {}'.format(
                                                  paper_names[(plot_name,
                                                               h)])))
                ax3.legend(handles=handles, fontsize=fontsize-5)
                ax4 = ax3.twinx()
                ax4.bar(to_plot_x, gmet.prcp.values, color='b')
                ax4.set_ylabel('Precipitation (mm)',
                               color='b', fontsize=fontsize)
                ax3.set_ylabel('Normalized scale', fontsize=fontsize)

                # tricky, but the value is cumulative after a day with redrill
                to_plot_nocumsum[32] /= 2
                ax5.bar(to_plot_x, to_plot_nocumsum)

    plt.setp(ax1.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax1.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax2.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax2.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax3.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax3.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax4.get_yticklabels(), fontsize=fontsize)
    plt.setp(ax5.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax5.get_yticklabels(), fontsize=fontsize)
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax5.grid()
    ax1.set_ylabel('Cumulative MB (m w.e.)', fontsize=fontsize)
    ax2.set_ylabel('Cumulative MB (m w.e.)', fontsize=fontsize)
    ax5.set_ylabel('MB (m w.e.)', fontsize=fontsize)
    ax2.set_xlabel('Time', fontsize=fontsize)
    ax5.set_xlabel('Time', fontsize=fontsize)
    ax2.set_xlabel(None)
    ax3.set_xlabel(None)
    # make ticks thicker
    ax1.xaxis.set_tick_params(width=5)
    ax1.yaxis.set_tick_params(width=5)
    ax2.xaxis.set_tick_params(width=5)
    ax2.yaxis.set_tick_params(width=5)
    ax3.xaxis.set_tick_params(width=5)
    ax3.yaxis.set_tick_params(width=5)
    ax5.xaxis.set_tick_params(width=5)
    ax5.yaxis.set_tick_params(width=5)

    ax1.legend(fontsize=fontsize-5)
    ax2.legend(fontsize=fontsize-5)

    ax5.set_xlim(
        obs_merge.date.values[0] - np.timedelta64(24 * 3600 * 10 ** 9),
        np.datetime64('2019-10-03'))


def plot_cali_param_space(gdir=None, cali_suffix='', how='countour'):
    """

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
    cali_suffix: str
    how: str

    # todo: this is not finished yet

    Returns
    -------

    """
    # get cali

    # there is also a cool example for a 2d scatter plus histograms here:
    # https://bit.ly/35gyVBa
    
    # this is a template for a 3d scatter plot with projection on planes from
    # https://bit.ly/35iP9cX
    x = np.random.random(100)
    y = np.random.random(100)
    z = np.sin(3 * x ** 2 + y ** 2)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z)

    ax.plot(x, z, 'r+', zdir='y', zs=1.5)
    ax.plot(y, z, 'g+', zdir='x', zs=-0.5)
    ax.plot(x, y, 'k+', zdir='z', zs=-1.5)

    ax.set_xlim([-0.5, 1.5])
    ax.set_ylim([-0.5, 1.5])
    ax.set_zlim([-1.5, 1.5])

    plt.show()


def plot_ensemble_size_optimization(error_dict):
    """

    Parameters
    ----------
    error_dict

    Returns
    -------

    """

    fig, ax = plt.subplots()
    sorted_dict = OrderedDict(error_dict)
    std_error = []
    for k, v in error_dict.items():
        std_error.append(np.std(v) / np.sqrt(len(v)))
    xvals = np.array([i[0] for i in list(sorted_dict.items())])
    yvals = np.array([i[1] for i in list(sorted_dict.items())])
    yvals_med = np.mean(yvals, axis=1)
    ax.plot(xvals, yvals_med)
    ax.fill_between(
        xvals, yvals_med + std_error, yvals_med - std_error, alpha=0.2)
    ax.set_title('Ensemble size optimization')
    ax.set_xlabel('Ensemble size')
    ax.set_ylabel('CRPS +/- std error (m w.e.)')


def plot_comparison_glamos_model_perc_extrap_summer_balance():
    """
    Plot MB predicted with percentile extrapolation and GLAMOS summer balance.

    Returns
    -------
    None
    """
    plt.style.use('seaborn-talk')
    area = [2.665, 0.0862, 0.0594, 0.2694, 2.595, 4.785, 1.8887, 0.3013,
            2.168, 2.04]
    names = ['Silvretta', 'Pizol', 'Schwarzbach', 'Sex Rouge',
             'Tsanfleuron', 'Gries', 'Basòdino', 'Murtel', 'Clariden',
             'Adler']
    labels = [x for _, x in sorted(zip(area, names), reverse=True)]
    glamos = [-2.88, -2.58, -2.49, -3.35, -3.7, -3.37, -2.817, -2.395, -2.97,
              -1.93]
    glamos = [x for _, x in sorted(zip(area, glamos), reverse=True)]
    glamos_std = [0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175,
                  0.175, 0.175]
    glamos_std = [x for _, x in sorted(zip(area, glamos_std), reverse=True)]
    model = [-2.47, -3.52, -3.78, -3.56, -3.73, -3.71, -2.81, -2.36,
             -2.77, -1.98]
    model = [x for _, x in sorted(zip(area, model), reverse=True)]
    model_std = [1.38, 0.68, 1.58, 1.38, 1.37, 1.04, 1.48, 1.04, 1.54,
                 1.26]
    model_std = [x for _, x in sorted(zip(area, model_std), reverse=True)]
    perc_extrap = [-2.79, -3.45, -3.53, -3.35, -3.67, -3.4, -2.8, -2.19,
                   -2.78, -2.01]
    perc_extrap = [x for _, x in sorted(zip(area, perc_extrap), reverse=True)]
    perc_extrap_std = [0.36, 0.23, 0.32, 0.23, 0.29, 0.26, 0.24, 0.36,
                       0.28, 0.29]
    perc_extrap_std = [x for _, x in sorted(zip(area, perc_extrap_std),
                                            reverse=True)]
    plt.figure()
    plt.errorbar(np.arange(0., 10., 1), glamos, yerr=glamos_std,
                 label='GLAMOS analysis', fmt='o')
    plt.errorbar(np.arange(-0.2, 9.8, 1), model, yerr=model_std,
                 label='MODEL only', fmt='o')
    plt.errorbar(np.arange(0.2, 10.2, 1), perc_extrap, yerr=perc_extrap_std,
                 label='Percentile extrapolation', fmt='o')
    plt.ylabel('Summer Mass Balance (m w.e.)', fontsize=18)
    plt.title(
        'Comparison between GLAMOS analysis, model prediction and percentile '
        'extrapolation based on GLAMOS climatologies',
        fontsize=16)
    plt.grid(which='y')
    plt.legend(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(np.arange(10), labels, fontsize=18, rotation=45)
    [plt.axvline(p, c='grey', alpha=0.5, lw=1.5) for p in np.arange(9) + 0.5]
    plt.tight_layout()


def plot_comparison_glamos_model_perc_extrap_summer_balance_clim_from_gmb():
    """
    Plot MB predicted with percentile extrapolation with calibration on GMB and
    GLAMOS summer balance.
    """
    plt.style.use('seaborn-talk')
    area = [2.665, 0.0862, 0.0594, 0.2694, 2.595, 4.785, 1.8887, 0.3013]
    labels = ['Silvretta', 'Pizol', 'Schwarzbach', 'Sex Rouge',
              'Tsanfleuron', 'Gries', 'Basòdino',
              'Murtel']  # , 'Clariden','Adler']
    labels = [x for _, x in sorted(zip(area, labels), reverse=True)]
    glamos = [-2.88, -2.58, -2.49, -3.35, -3.7, -3.37, -2.817, -2.395]  # ,
    # -2.97, -1.93]
    glamos = [x for _, x in sorted(zip(area, glamos), reverse=True)]
    glamos_std = [0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175, 0.175]  # ,
    # 0.175, 0.175]
    glamos_std = [x for _, x in sorted(zip(area, glamos_std), reverse=True)]
    model = [-3.94, -3.88, -3.95, -3.5, -3.72, -2.87, -3.7, -3.56]
    model = [x for _, x in sorted(zip(area, model), reverse=True)]
    model_std = [0.95, 2.33, 0.79, 1.88, 1.66, 1.4, 0.85, 0.75]
    model_std = [x for _, x in sorted(zip(area, model_std), reverse=True)]
    perc_extrap = [-3.85, -3.27, -5.53, -3.16, -3.35, -2.52, -3.19, -2.82]
    perc_extrap = [x for _, x in sorted(zip(area, perc_extrap), reverse=True)]
    perc_extrap_std = [0.29, 0.45, 0.16, 0.37, 0.43, 0.33, 0.2, 0.21]
    perc_extrap_std = [x for _, x in sorted(zip(area, perc_extrap_std),
                                            reverse=True)]
    plt.figure()
    plt.errorbar(np.arange(0., len(glamos), 1), glamos, yerr=glamos_std,
                 label='GLAMOS analysis', fmt='o')
    plt.errorbar(np.arange(-0.2, len(glamos) - 0.2, 1), model, yerr=model_std,
                 label='MODEL only', fmt='o')
    plt.errorbar(np.arange(0.2, len(glamos) + 0.2, 1), perc_extrap,
                 yerr=perc_extrap_std, label='Percentile extrapolation',
                 fmt='o')
    plt.ylabel('Summer Mass Balance (m w.e.)', fontsize=18)
    plt.grid()
    plt.legend(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(np.arange(10), labels, fontsize=18, rotation=45)
    plt.tight_layout()


def plot_2d_reconstruction_mean_spread(gdir, ens_var, n_mean_levels=20):
    """
    Plot a 2D reconstruction of ensemble mean and spread for a given glacier.

    Parameters
    ----------
    gdir : `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to create the plot for.
    ens_var : np.array
        The variable to be plotted. First dimension should be the length of the
        concatenated flow lines, second dimension the ensemble members.
    n_mean_levels: int
        Number of intervals for plotting mean contours.

    Returns
    -------
    None
    """
    # todo: implement differentiation by catchments!!
    fl_h, _ = gdir.get_inversion_flowline_hw()
    varmean = np.nanmean(ens_var, axis=1)
    varstd = np.nanstd(ens_var, axis=1)
    dem = xr.open_rasterio(gdir.get_filepath('dem'))
    dem_ds = dem.to_dataset(name='data')
    dem_ds.attrs['pyproj_srs'] = dem.crs
    dem_ds['varmean'] = np.nan
    dem_ds['varstd'] = np.nan

    for i in range(ens_var.shape[0] - 1):

        mask = ((dem_ds["data"] <= fl_h[i]) & (dem_ds["data"] > fl_h[i + 1]))
        dem_ds['varmean'] = xr.where(mask, varmean[i], dem_ds["varmean"])
        dem_ds['varstd'] = xr.where(mask, varstd[i], dem_ds["varstd"])

        ol = gpd.read_file(gdir.get_filepath('outlines'))
        dem_ds = dem_ds.salem.roi(shape=gdir.get_filepath('outlines'),
                                  crs=gdir.grid.proj.srs)
        fig, ax = plt.subplots()
        xr.plot.contour(dem_ds.varmean[0], levels=n_mean_levels,
                        ax=ax, add_colorbar=True, cmap='seismic_r', center=0)
        xr.plot.contourf(dem_ds.varstd[0], ax=ax, cmap='Greys')
        ol.plot(ax=ax, edgecolor='g', facecolor='None')
        plt.show()
