from __future__ import division

# Log message format
import logging

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import datetime as dt
# Locals
import crampon.cfg as cfg
from crampon import workflow
from crampon import utils
from crampon.core.models.massbalance import BraithwaiteModel, PellicciottiModel, \
    OerlemansModel, SnowFirnCover
import numpy as np
from scipy import optimize
import multiprocessing as mp
S

# Logging options
logging.basicConfig(format='%(asctime)s: %(name)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.DEBUG)
# Module logger
log = logging.getLogger(__name__)
log.setLevel('DEBUG')

# to suppress all NaN warnings
import warnings
warnings.filterwarnings("ignore")


# todo: this has to be here, otherwise cfg sets to basic...why?
cfg.initialize('C:\\users\\johannes\\documents\\crampon\\sandbox\\CH_params.cfg')


def to_minimize_deltav(x, g, mb_model, f_firn, i):

    t1 = dt.datetime.now()
    param_dict = dict(zip(
        [k for k, v in mb_model.cali_params_guess.items()], x))

    print('Parameters: {}'.format(param_dict.__repr__()))

    # read geodetic dV, measured MB, and calibration
    geodetic_dv = pd.read_csv(g.get_filepath('geodetic_dv'),
                              parse_dates=[2, 3], index_col=0)
    # todo: remove, just for testing!!!
    geodetic_dv = geodetic_dv[0:i]
    print('Optimizing for one time span only: {}'.format(geodetic_dv))

    grid_dx = g.grid.dx

    all_d0 = list(geodetic_dv.date0.values)
    all_d1 = list(geodetic_dv.date1.values)

    result_df = pd.DataFrame(columns=['d1', 'd2'],
                             data=np.array([all_d0, all_d1]).T)

    d1_timestamps = [pd.Timestamp(i) for i in result_df.d1.values]
    d2_timestamps = [pd.Timestamp(i) for i in result_df.d2.values]

    day_model = mb_model(g, bias=0., **param_dict)

    begin_clim = utils.get_begin_last_flexyear(dt.datetime(1961, 12, 31),
                                               10, 1)
    end_clim = utils.get_begin_last_flexyear(dt.datetime(2018, 12, 31), 10,
                                             1)
    heights, widths = g.get_inversion_flowline_hw()
    init_swe = np.zeros_like(heights)
    init_swe.fill(np.nan)
    init_temp = init_swe
    cover = SnowFirnCover(heights, swe=init_swe, rho=None,
                          origin=begin_clim, temperatures=init_temp)

    # number of experiments (list!)
    exp = [1]
    mb = []

    last_clim_year_begin = end_clim

    factor = cfg.SEC_IN_DAY * cfg.RHO / 1000.
    rho_input = np.ones_like(heights) * 100

    img_list_for_ani = []
    #ice_melt_list = []

    # todo: can we stop here or do we need to go to end_clim?
    for date in pd.date_range(begin_clim, max(d1_timestamps+d2_timestamps)):  #end_clim):

        # Get the mass balance and convert to m per day
        before = dt.datetime.now()
        if date.day == 1 and date.month == 1:
            print(date)
        tmp = day_model.get_daily_mb(heights, date=date)  # 7.02 ms
        cover.ingest_balance(tmp * factor, rho=np.ones_like(heights) * 100,
                             date=date)  # 48.5ms

        ice_melt = np.clip(tmp * factor, None, 0.)
        #ice_melt[~np.logical_or((cover.sh == 0), np.isnan(cover.sh))[np.arange(cover.n_heights), cover.top_layer]] = 0.
        ice_melt[np.where(cover.get_total_height() != 0)] = 0.  # 9.98 ms
        if not np.all(ice_melt[150:] <= ice_melt[149:-1]):
            print('stop')
        #print(date, np.sum(cover.get_lost_ice_volume(widths=widths, map_dx=grid_dx)), np.sum(cover.ice_melt), np.sum(ice_melt))
        cover.ice_melt += ice_melt
        #ice_melt_list.append(np.sum(cover.ice_melt))

        # densify the snow - daily
        cover.densify_snow_anderson(date)  # 88.5 ms

        # cheap refreezing
        if date.day == 30 and date.month == 4:
            cover.update_temperature_huss()
            cover.apply_refreezing(exhaust=True)

        if date.day == 1 and date.month == 10:
            cover.densify_firn_huss(date, f_firn=f_firn)

        # todo:  densify the firn - we can do it once as this model anyway only depends on the initial density?
        if date in d1_timestamps:
            print('date in d1 ({})'.format(date))
            # todo: if it doesn't work, here's a possibility to run it with a different f_firn
            cover.densify_firn_huss(date, f_firn=f_firn)

            # todo: apply more sophisticated refreezing here
            # let refreeze
            cover.update_temperature_huss()
            cover.apply_refreezing(exhaust=True)

            mask_f1 = cover.get_mean_density().mask
            rho_f1 = np.average(cover.get_mean_density(),
                                weights=np.ma.masked_array(widths,
                                                           mask=mask_f1))
            if isinstance(rho_f1, np.ma.core.MaskedConstant):
                result_df.loc[result_df.d1 == date, 'rho_f1'] = np.nan
            else:
                result_df.loc[result_df.d1 == date, 'rho_f1'] = rho_f1
            result_df.loc[result_df.d1 == date, 'v_f1'] = np.sum(
                cover.get_total_volume(widths=widths, map_dx=grid_dx))
            result_df.loc[result_df.d1 == date, 'v_i1'] = np.sum(
                cover.get_lost_ice_volume(widths=widths, map_dx=grid_dx))

        #if date.day == 1:
        #    img_list_for_ani.append(np.flipud(cover.property_weighted_by_height(cover.rho).T))

        if np.isnan(cover.rho).all() and (date.year != 1961):
            print('Fucking date: {}'.format(date))
            print('stop')

        if date in d2_timestamps:
            print('date in d2 ({})'.format(date))
            # todo: if it doesn't work, here's a possibility to run it with a different f_firn
            cover.densify_firn_huss(date)

            # todo: apply more sophisticated refreezing here
            # let refreeze
            cover.update_temperature_huss()
            cover.apply_refreezing(exhaust=True)

            mask_f2 = cover.get_mean_density().mask
            rho_f2 = np.average(cover.get_mean_density(), weights=np.ma.masked_array(widths, mask=mask_f2))
            if isinstance(rho_f2, np.ma.core.MaskedConstant):
                result_df.loc[result_df.d2 == date, 'rho_f2'] = np.nan
            else:
                result_df.loc[result_df.d2 == date, 'rho_f2'] = rho_f2
            result_df.loc[result_df.d2 == date, 'v_f2'] = np.sum(cover.get_total_volume(widths=widths, map_dx=grid_dx))
            result_df.loc[result_df.d2 == date, 'v_i2'] = np.sum(cover.get_lost_ice_volume(widths=widths, map_dx=grid_dx))

            # todo: can we break here or do we need to search for next  GLAMOS MB date?


    # check if delta_v from from model and from DEM subtraction agree (model useful)
    # if not, model not useful
    result_df['delta_v_model'] = (result_df.v_f2 - result_df.v_f1) + \
                                 (result_df.v_i2 - result_df.v_i1)  # ice volumes are negative

    # kick out DEMs with uncertain dates
    #geodetic_dv_valid = geodetic_dv[
    #    (geodetic_dv.date0 != np.datetime64('2010-01-01')) & (
    #            geodetic_dv.date1 != np.datetime64('2010-01-01')) & (
    #            geodetic_dv.date0 != np.datetime64(
    #        '1970-01-01'))]
    #result_df_valid = result_df[
    #    (result_df.d1 != np.datetime64('2010-01-01')) & (
    #            result_df.d2 != np.datetime64('2010-01-01')) & (
    #            result_df.d1 != np.datetime64('1970-01-01'))]
    geodetic_dv_valid = geodetic_dv.copy()
    result_df_valid = result_df.copy()

    geodetic_dv_valid = geodetic_dv_valid.rename(index=str,
                                                 columns={'date0': 'd1',
                                                          'date1': 'd2'})

    merged_dfs = geodetic_dv_valid.merge(result_df_valid, on=['d1', 'd2'])

    print('modeled: ', merged_dfs.delta_v_model)
    print('measured: ', merged_dfs.dv)
    print('error: ', merged_dfs.delta_v_model - merged_dfs.dv)
    print('error ratio: ', (merged_dfs.delta_v_model - merged_dfs.dv) / merged_dfs.dv)
    print('Time to run one optimization step: {}'.format(dt.datetime.now() - t1))

    return merged_dfs.delta_v_model - merged_dfs.dv


def run_iterations(gdir, mb_model, cfg_dict, f_firn, params):

    cfg.unpack_config(cfg_dict)

    geodetic_dv = pd.read_csv(gdir.get_filepath('geodetic_dv'),
                              parse_dates=[2, 3], index_col=0)

    for i in range(1, len(geodetic_dv)):

        result = optimize.least_squares(to_minimize_deltav, xtol=10e-3,
                                        x0=np.array(list(params)),
                                        verbose=2, bounds=(0., np.inf),
                                        args=(gdir, mb_model, f_firn, i))
        result_list.append(result)
        print(gdir, mu_ice, prcp_fac, result_list)

    print(gdir, mu_ice, prcp_fac, result_list)

    lock.acquire()
    with open('result_file_iterative_method.txt', 'a') as f:
        f.write(str(params.__repr__()) + '\n')
        f.write(str(result_list) + '\n\n\n\n\n\n\n\n')
    lock.release()
    return result_list


def init(l):
    global lock
    lock = l


if __name__ == '__main__':

    cfg.initialize('c:\\users\\johannes\\documents\\crampon\\sandbox\\CH_params.cfg')

    # Currently OGGM wants some directories to exist
    # (maybe I'll change this but it can also catch errors in the user config)
    utils.mkdir(cfg.PATHS['working_dir'])

    # Mauro's DB ad disguised in RGI
    glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
    rgidf = gpd.read_file(glaciers)
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A10G05', 'RGI50-11.B4504',
    #                                'RGI50-11.A55F03', 'RGI50-11.B4312n-1',
    #                                'RGI50-11.B5616n-1', 'RGI50-11.C1410'])]
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4504'])]
    rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A55F03'])]

    log.info('Number of glaciers: {}'.format(len(rgidf)))

    # Go - initialize working directories
    gdirs = workflow.init_glacier_regions(rgidf, reset=False, force=False)

    f_firn = 2.4
    mu_ice = 10.
    mu_snow = mu_ice/2.
    prcp_fac = 1.5

    result_list = []

    for gdir in gdirs:

        geodetic_dv = pd.read_csv(gdir.get_filepath('geodetic_dv'),
                                  parse_dates=[2, 3], index_col=0)
        for i in range(19, len(geodetic_dv)):

            result = optimize.least_squares(to_minimize_deltav, xtol=10e-3,
                                            x0=np.array([mu_ice, prcp_fac]),
                                            verbose=2, bounds=(0., np.inf),
                                            args=(gdir, f_firn, i))
            result_list.append(result)
            print(result_list)
        print('hi')
