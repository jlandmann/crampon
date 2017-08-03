""" Prepare the meteodata from netCDF4 """

from __future__ import division

import os
from glob import glob
import crampon.cfg as cfg
from crampon.core.models.massbalance import DailyMassBalanceModel
from crampon import utils
from crampon.utils import date_to_year, GlacierDirectory
#from oggm.core.preprocessing.climate import *
import itertools
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from crampon import entity_task
import logging
from itertools import product

# temporary
from crampon.workflow import execute_entity_task

# Module logger
log = logging.getLogger(__name__)

# To (re)write:
# mb_climate_on_height, _get_ref_glaciers, _get_optimal_scaling_factor


class MeteoSuisseGrid(object):
    """ Interface for MeteoSuisse Input as netCDF4.

    The class interacts with xarray via salem and allows subsetting for each
    glacier, getting reference cells for each glacier, and gradient
    """

    def __init__(self, ncpath):
        self.ddir = ncpath
        self._data = salem.open_xr_dataset(self.ddir)
        self._vartype = None

    @property
    def vartype(self):
        """
        Check which data are contained. Currently implemented:
        "TabsD": mean daily temperature
        "RprelimD": daily precipitation sum

        Returns
        -------
        The "vartype" class attribute
        """
        var_dict = self._data.data_vars

        if "TabsD" in var_dict:
            self._vartype = 'TabsD'
        elif 'RprelimD' in var_dict:
            self._vartype = 'RprelimD'
        else:
            miss_vartype = list(var_dict.keys())[0]
            raise NotImplementedError('MeteoGrid variable {} is not yet '
                                      'implemented'.format(miss_vartype))

        return self._vartype

    @vartype.setter
    def vartype(self, value):
        self._vartype = value

    def subset_by_shape(self, shpdir=None, buffer=0):
        """
        Subset the array based on the given shapefile, including the buffer.

        Parameters
        ----------
        shpdir: str
             Path to the shape used for clipping
        buffer: int
             Cells to be used as buffer around the shape

        Returns
        -------

        """

        shape = salem.read_shapefile(shpdir, cached=True)

        subset = self._data.salem.subset(shape=shape, margin=buffer)

        return subset

    def clip_by_shape(self, shpdir=None):
        """
        Clip the array based on the given shapefile, including the buffer.

        Parameters
        ----------
        shpdir: str
             Path to the shape used for clipping

        Returns
        -------

        """
        shape = salem.read_shapefile(shpdir, cached=True)

        clipped = self._data.salem.subset(shape=shape)

        return clipped

    def get_reference_value(self, shpdir=None):

        shape = salem.read_shapefile(shpdir, cached=True)
        centroid = shape.centroid

        return shape

    def get_gradient(self):
        raise NotImplementedError()

    def downsample(self):
        # This should remain in crampon maybe, as OGGM doesn't need it
        raise NotImplementedError()

    def write_oggm_format(self):
        # Write out an OGGM suitable format
        raise NotImplementedError()

    def merge(self, other):
        """
        Merge with another MeteoSuisseGrid.
        
        Parameters
        ----------
        other: MeteoSuisseGrid to merge with.

        Returns
        -------
        Merged MeteoSuisseGrid.
        """


# This writes 'climate_monthly' in the original version (doesn't fit anymore)
@entity_task(log)
def process_custom_climate_data_crampon(gdir):
    """Processes and writes the climate data from a user-defined climate file.

    This function is strongly related to the OGGM function. The input file must
     have a specific format
     (see oggm-sample-data/test-files/histalp_merged_hef.nc for an example).

    The modifications to the original function allow a more flexible handling
    of the climate file, e.g. with a daily frequency.

    uses caching for faster retrieval.
    """

    if not (('climate_file' in cfg.PATHS) and
            os.path.exists(cfg.PATHS['climate_file'])):
        raise IOError('Custom climate file not found')

    # read the file
    fpath = cfg.PATHS['climate_file']
    nc_ts = salem.GeoNetcdf(fpath)

    # geoloc
    lon = nc_ts._nc.variables['lon'][:]
    lat = nc_ts._nc.variables['lat'][:]

    # Gradient defaults
    use_grad = cfg.PARAMS['temp_use_local_gradient']
    def_grad = cfg.PARAMS['temp_default_gradient']
    g_minmax = cfg.PARAMS['temp_local_gradient_bounds']

    # get closest grid cell and index
    ilon = np.argmin(np.abs(lon - gdir.cenlon))
    ilat = np.argmin(np.abs(lat - gdir.cenlat))
    ref_pix_lon = lon[ilon]
    ref_pix_lat = lat[ilat]

    # Some special things added in the crampon function
    iprcp, itemp, igrad, ihgt = utils.joblib_read_climate(fpath, ilon,
                                                          ilat, def_grad,
                                                          g_minmax,
                                                          use_grad)

    # Set temporal subset for the ts data depending on frequency:
    # hydro years if monthly data, else no restriction
    yrs = nc_ts.time.year
    y0, y1 = yrs[0], yrs[-1]
    time = nc_ts.time
    if pd.infer_freq(nc_ts.time) == 'MS':  # month start frequency
        nc_ts.set_period(t0='{}-10-01'.format(y0), t1='{}-09-01'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                        ref_pix_lon, ref_pix_lat)
    elif pd.infer_freq(nc_ts.time) == 'M':  # month end frequency
        nc_ts.set_period(t0='{}-10-31'.format(y0), t1='{}-09-30'.format(y1))
        ny, r = divmod(len(time), 12)
        if r != 0:
            raise ValueError('Climate data should be N full years exclusively')
        gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                        ref_pix_lon, ref_pix_lat)
    elif pd.infer_freq(nc_ts.time) == 'D':  # day start frequency
        # Doesn't matter if entire years or not, BUT a correction for y1 to be
        # the last hydro/glacio year is needed
        if not '{}-09-30'.format(y1) in nc_ts.time:
            y1 = yrs[-2]
        # Ok, this is NO ERROR: we can use the function
        # ``write_monthly_climate_file`` also to produce a daily climate file:
        # there is no reference to the time in the function! We should just
        # change the ``file_name`` keyword!
        gdir.write_monthly_climate_file(time, iprcp, itemp, igrad, ihgt,
                                        ref_pix_lon, ref_pix_lat,
                                        file_name='climate_daily',
                                        time_unit=nc_ts._nc.variables['time']
                                        .units)
    else:
        raise NotImplementedError('Climate data frequency not yet understood')

    # for logging
    end_date = time[-1]

    # metadata
    out = {'climate_source': fpath, 'hydro_yr_0': y0 + 1,
           'hydro_yr_1': y1, 'end_date': end_date}
    gdir.write_pickle(out, 'climate_info')


def climate_file_from_scratch(write_to=os.path.expanduser('documents\\crampon\\data\\bigdata'),
                              hfile='c:\\users\\johannes\\documents\\crampon\\data\\test\\hgt.nc'):
    """
    Compile the climate file needed for any CRAMPON calculations.

    The file will contain an up to date meteorological time series from all
    currently available files on Cirrus.

    Parameters
    ----------
    write_to: str
        Directory where the Cirrus files should be synchronized to and where
        the processed/concatenated files should be written to.
    hfile: str
        Path to a netCDF file containing a DEM of the area (used for assembling
        the file that OGGM likes. Needs to cover the same area in the same
        extent ans resolution as the meteo files.

    Returns
    -------
    outfile: str
        The path to the compiled file.
    """

    for var, mode in product(['TabsD', 'R'], ['verified', 'operational']):
        cirrus = utils.CirrusClient()
        _, _ = cirrus.sync_files('/data/griddata', write_to
                                 , globpattern='*{}/daily/{}*/netcdf/*'
                                 .format(mode, var))

        flist = glob.glob(os.path.join(write_to,
                                       'griddata\\{}\\daily\\{}*\\netcdf\\*.nc'
                                       .format(mode, var)))

        # Instead of using open_mfdataset, as we need a lot of preprocessing
        log.info('Concatenating {} {} {} files...'.format(len(flist), var, mode))
        sda = utils.read_multiple_netcdfs(flist, chunks={'time': 50},
                                          tfunc=utils._cut_with_CH_glac)
        log.info('Ensuring time continuity...')
        sda = sda.crampon.ensure_time_continuity()
        sda.encoding['zlib'] = True
        sda.to_netcdf(os.path.join(write_to, '{}_{}_all.nc'.format(var,
                                                                   mode)))

    # update operational with verified
    for var in ['TabsD', 'R']:
        v_op = os.path.join(write_to, '{}_operational_all.nc'.format(var))
        v_ve = os.path.join(write_to, '{}_verified_all.nc'.format(var))

        data = utils.read_netcdf(v_op, chunks={'time': 50})
        data = data.crampon.update_with_verified(v_ve)

        data.to_netcdf(os.path.join(write_to, '{}_op_ver.nc'.format(var)))

    # combine both
    tfile = glob.glob(os.path.join(write_to, '*{}*_op_ver.nc'
                                   .format('TabsD')))[0]
    pfile = glob.glob(os.path.join(write_to, '*{}*_op_ver.nc'
                                   .format('R')))[0]
    outfile = os.path.join(write_to, 'climate_all.nc')

    utils.daily_climate_from_netcdf(tfile, pfile, hfile, outfile)

    return outfile

'''
def past_mb_anytime(gdir, cali_params, ym_prod, cl, insitu):
    """
    
    Parameters
    ----------
    gdir
    ym_prod
    cl
    insitu

    Returns
    -------

    """
    mustar, prcp_fac = cali_params
    past_model = PastMassBalanceModel(gdir, mu_star=mustar, prcp_fac=prcp_fac)

    mb_monthly = []
    for y, m in ym_prod:
        ym = date_to_year(y, m)
        # Get the mass balance and convert to m per year
        tmp = past_model.get_monthly_mb(cl['hgt'], ym) * \
              cfg.SEC_IN_MONTHS[m-1] * cfg.RHO / 1000.
        mb_monthly.append(sum(tmp))
    mb = (sum(mb_monthly))
    return np.abs(np.subtract(np.ones_like(mb)*insitu, mb))
'''


#@entity_task(log, writes=['mustar_from_mauro'])
def mustar_from_deltah(gdir, deltah_df, mustar_rg=None,
                       prcp_fac_rg=None, recurs=0, err_thresh=5.):
    """
    Get mustar from an elevation change.
    
    Parameters
    ----------
    gdir: oggm.utils.GlacierDirectory
        An OGGM GlacierDirectory.
    deltah_df: pandas.DataFrame or geopandas.GeoDataFrame
        A DataFrame with thickness changes im millimeters of ice and begin date
        and end date of the measurement period. A the moment, the column names
        are still odd and result from putting FoG column names in the 13 spaces
        limited shapefile attributes
    mustar_rg: list-like
        A range for the temperature sensitivity to be tested
    prcp_fac_rg: list-like
        A range for the precipitation correction factor to be tested
    recurs: int
        Number of recursions (needed if parameters bump into corners)
    err_thresh: float
        Allowed error threshold in reproduction of given ice thickness change.
    
    Returns
    -------
    mb: array
        The mass balance calculated with all parameter combinations (mm w.e.)
    err: array
        The squared error with respect to the calibration value (mm w.e. ** 2).
    """

    p_comb = list(itertools.product(enumerate(mustar_rg),
                                    enumerate(prcp_fac_rg)))

    # the column names are odd, but due to the 13 char limitation in SHPs
    ybeg = int(str(deltah_df[deltah_df.RGIId == gdir.rgi_id].REFERENCE_.iloc[0])[:4])
    yend = int(str(deltah_df[deltah_df.RGIId == gdir.rgi_id].BgnDate.iloc[0])[:4])

    # thickness change in meters of ice
    insitu_ally = deltah_df[deltah_df.RGIId == gdir.rgi_id]['THICKNESS_'].iloc[0] \
                  / 1000.
    ym_prod = list(itertools.product([ybeg], [9, 10, 11, 12])) +\
              list(itertools.product(np.arange(ybeg + 1, yend),
                                     np.arange(1, 13))) + \
              list(itertools.product([yend], [1, 2, 3, 4, 5, 6, 7, 8]))
    ##### HERE SHOULD GO A PROCEDURE TO ACCOUNT FOR UNCERTAINTIES IN DEM DATES

    majid = gdir.read_pickle('major_divide', div_id=0)
    maj_fl = gdir.read_pickle('inversion_flowlines', div_id=majid)[-1]
    maj_hgt = maj_fl.surface_h

    mb = np.ones((len(mustar_rg), len(prcp_fac_rg)))
    i = 0
    #result_minimize = optimize.minimize(past_mb_anytime,
    #                                    [(np.mean(mustar_rg),
    #                                     np.mean(prcp_fac_rg))],
    #                                    args=(gdir, ym_prod, cl, insitu_ally),
    #                                    bounds=((None, None), (None, None)))
    #result_de = optimize.differential_evolution(past_mb_anytime,
    #                                            [(np.min(mustar_rg),
    #                                              np.max(mustar_rg)),
    #                                             (np.min(prcp_fac_rg),
    #                                              np.max(prcp_fac_rg))],
    #                                            args=(gdir, ym_prod, cl, insitu_ally))

    for (mi, mustar), (pi, prcp_fac) in p_comb:
        day_model = DailyMassBalanceModel(gdir, mu_star=mustar,
                                           prcp_fac=prcp_fac, bias=0.)

        mb_monthly = []
        for date in day_model.tspan_in:

            try:
            # Get the mass balance and convert to m per day
                tmp = day_model.get_daily_mb(maj_hgt, date=date) * \
                      cfg.SEC_IN_DAY * cfg.RHO / 1000.
            # It happens that e.g. RGI50-11.00638 has the 1st DEM date before
            #  the Meteo time series begins (OGGM problem!?)
            except IndexError:
                if i == 0:
                    print('Skipped date {} for glacier {}'.format(date,
                                                                  gdir.rgi_id))
                continue
            mb_monthly.append(sum(tmp))
        mb[mi][pi] = sum(mb_monthly)

        i += 1
        if i % 1000 == 0:
            print(i)

    # compute absolute error and make it squared
    # see: http://www.benkuhn.net/squared for example reasons
    err = np.power(np.abs(np.subtract(np.ones_like(mb) * insitu_ally,
                                      mb)), 2)

    # check if the minimum error got caught at the edges of the definition
    # interval of mustar and prcp_fac
    min_ind = np.where(err == err.min())

    """
    # Ineffective, as whole range is calculated again
    if (min_ind[0] == 0 or min_ind[1] == 0) and recurs == 0:
        #new_mrg = np.append(mustar_rg - np.abs(np.max(mustar_rg)), mustar_rg) # the range becomes negative, this makes no sense!!!
        #new_prg = np.append(prcp_fac_rg - np.abs(np.max(prcp_fac_rg)), prcp_fac_rg) # the range becomes negative, this makes no sense!!!
        # let's try to decrease the step width instead
        new_mrg = np.arange(np.min(mustar_rg), np.max(mustar_rg),
                            ((np.max(mustar_rg) - np.min(mustar_rg)) /
                             (len(mustar_rg) - 1)) / 2.)
        new_prg = np.arange(np.min(prcp_fac_rg), np.max(prcp_fac_rg),
                            ((np.max(prcp_fac_rg) - np.min(prcp_fac_rg)) /
                             (len(prcp_fac_rg) - 1)) / 2.)
        print('Cali param values bumped into lower bounds')
        print(new_mrg, new_prg)
        mb, err = mustar_from_deltah(gdir=gdir, deltah_df=deltah_df,
                                     mustar_rg=new_mrg,
                                     prcp_fac_rg=new_prg, recurs=1)
        min_ind = np.where(err == err.min())
        print(mustar_rg[min_ind[0]], prcp_fac_rg[min_ind[1]])
        return mb, err
    elif (min_ind[0] == err.shape[0] or min_ind[1] == err.shape[1]) and recurs == 0:
        new_mrg = np.append(mustar_rg, mustar_rg + np.abs(np.max(mustar_rg)))
        new_prg = np.append(prcp_fac_rg, prcp_fac_rg +
                            np.abs(np.max(prcp_fac_rg)))
        print('Cali param values bumped into upper bounds')
        print(new_mrg, new_prg)
        mb, err = mustar_from_deltah(gdir=gdir, deltah_df=deltah_df,
                                     mustar_rg=new_mrg,
                                     prcp_fac_rg=new_prg, recurs=1)
        min_ind = np.where(err == err.min())
        print(mustar_rg[min_ind[0]], prcp_fac_rg[min_ind[1]])
        return mb, err
    """

    # Error greater than thresholt
    """
    if (np.sqrt([err[min_ind]][0]) / insitu_ally)[0] * 100 > err_thresh and recurs == 0:
        print('Error too big ({}\%) for {}'.format((np.sqrt([err[min_ind]][0])
                                                    / insitu_ally)[0] * 100,
                                                   gdir.rgi_id))
        # let's try to decrease the step width instead
        new_mrg = np.arange(np.min(mustar_rg), np.max(mustar_rg),
                            ((np.max(mustar_rg) - np.min(mustar_rg)) / (len(mustar_rg) - 1)) / 2.)
        new_prg = np.arange(np.min(prcp_fac_rg), np.max(prcp_fac_rg),
                            ((np.max(prcp_fac_rg) - np.min(prcp_fac_rg)) / (len(prcp_fac_rg) - 1)) / 2.)
        mb, err = mustar_from_deltah(gdir=gdir, deltah_df=deltah_df,
                                     mustar_rg=new_mrg,
                                     prcp_fac_rg=new_prg, recurs=1)
        min_ind = np.where(err == err.min())
        print(mustar_rg[min_ind[0]], prcp_fac_rg[min_ind[1]])
        return mb, err

    else:
    """
    # Take up to 50 values close to minimum as long as they are smaller
    # than error threshold
    N = 50
    n_lowest_val = err[np.unravel_index(err.argsort(axis=None),
                                        err.shape)][:N]

    # Indices for lowest mustasr and precipitation
    n_lowest_m = np.unravel_index(err.argsort(axis=None), err.shape)[0][:N]
    n_lowest_p = np.unravel_index(err.argsort(axis=None), err.shape)[1][:N]

    n_lowest_val = np.where(np.abs(np.sqrt(n_lowest_val) / insitu_ally * 100) <= err_thresh)
    print(err[n_lowest_m, n_lowest_p])
    print(mustar_rg[n_lowest_m], prcp_fac_rg[n_lowest_p])

    print(mustar_rg[min_ind[0]], prcp_fac_rg[min_ind[1]])
    print('Error in Percent:{}'.format(
        (np.sqrt([err[min_ind]][0]) / insitu_ally)[0] * 100))

    for div_id in [0] + list(gdir.divide_ids):
        # Scalars in a small dataframe for later
        df = pd.DataFrame()
        df['rgi_id'] = [gdir.rgi_id] * len(n_lowest_m)
        df['t_star'] = [None] * len(n_lowest_m)
        df['mu_star'] = mustar_rg[n_lowest_m]#[mustar_rg[min_ind[0]]][0]
        df['prcp_fac'] = prcp_fac_rg[n_lowest_p]#[prcp_fac_rg[min_ind[1]]][0]
        df['bias'] = np.sqrt(err[n_lowest_m, n_lowest_p]) #[err[min_ind]][0]
        df['err_perc'] = np.abs((np.sqrt([err[n_lowest_m, n_lowest_p]]) / insitu_ally)[0] * 100) #(np.sqrt([err[min_ind]][0]) / insitu_ally)[0] * 100
        #  drop lines where error too big
        df = df[df.err_perc <= err_thresh]
        df.to_csv(gdir.dir+'\\mustar_from_mauro.csv',
                  index=False)

    return mb, err


# IMPORTANT: overwrite OGGM functions with same name:
process_custom_climate_data = process_custom_climate_data_crampon

if __name__ == '__main__':
    # Initialize CRAMPON (and OGGM, hidden in cfg.py), IMPORTANT!
    cfg.initialize(file='C:\\Users\\Johannes\\Documents\\crampon\\sandbox\\'
                        'CH_params.cfg')

    # for testglaciers
    #gdirs = [GlacierDirectory(i) for i in
    #         glob('C:\\Users\\Johannes\\Desktop\\testglaciers_safe\\'
    #              'per_glacier\\RGI50-11\\*\\R*\\')]

    # Swiss 'problem glaciers'
    problem_glaciers_sgi = ['RGI50-11.00761-0', 'RGI50-11.00794-1',
                            'RGI50-11.01509-0', 'RGI50-11.01538-0',
                            'RGI50-11.01956-0', 'RGI50-11.B6302-1',
                            'RGI50-11.02552-0', 'RGI50-11.B3643n',
                            'RGI50-11.02576-0', 'RGI50-11.02663-0',
                            'RGI50-11.A12E09-0', 'RGI50-11.B3217n',
                            'RGI50-11.A14I03-3', 'RGI50-11.A14G14-0',
                            'RGI50-11.A54I17n-0', 'RGI50-11.A14F13-4',
                            'RGI50-11.B4616-0',  # 'bottleneck' polygons
                            'RGI50-11.02848']  # ValueError: no minimum-cost path was found to the specified end point (compute_centerlines)

    # for all calculated glaciers
    gdirs = []
    for i in glob('C:\\Users\\Johannes\\Documents\\modelruns\\CH\\per_glacier\\RGI50-11\\*\\R*\\'):
        if any([j in i for j in problem_glaciers_sgi]):
            continue

        try:
            gdirs.append(GlacierDirectory(i.split('\\')[-2]))
        except:
            print(i)
            continue

    # remove
    gdirs = gdirs[:2]
    ################

    deltav_file = 'C:\\Users\\Johannes\\Desktop\\mauro_in_RGI_disguise_entities_old_and_new.shp'
    deltah = gpd.read_file(deltav_file)
    deltah = deltah[['RGIId', 'THICKNESS_', 'BgnDate', 'REFERENCE_']]

    # set ranges and steps for calibration parameters
    mrg_step = 1.
    prg_step = 0.1
    mrg = np.arange(0., 80. + mrg_step, mrg_step)
    prg = np.arange(0.05, 2.5 + prg_step, prg_step)
    prg = [2.5]


    for gdir in gdirs:
        # read from OGGM
        #tdf = pd.read_csv(gdir.get_filepath('local_mustar'))
        #oggm_mustar = tdf.mu_star.iloc[0]
        #oggm_prcp_fac = tdf.prcp_fac.iloc[0]

        mb, err = mustar_from_deltah(gdir=gdir, mustar_rg=mrg,
                                     deltah_df=deltah, prcp_fac_rg=prg)
    '''
        # if already calibrated, then skip
        if os.path.exists(gdir.dir + '\\mustar_from_mauro.csv'):
            gdirs.remove(gdir)


    #try:
    #    mb, err = mustar_from_deltah(gdir=gdir, deltah=deltah,
    #                                 mustar_rg=mrg, prcp_fac_rg=prg)
    #except (FileNotFoundError, ValueError) as ex:
    #    print('{} for {}'.format(ex, gdir.dir))
    #    errorlist.append([ex, gdir.dir])
    #    continue
    '''

    execute_entity_task(mustar_from_deltah, gdirs, deltah_df=deltah,
                        mustar_rg=mrg, prcp_fac_rg=prg)


    '''
    # show mass balances produced by the model
    plt.imshow(mb, cmap='nipy_spectral')
    plt.show()

    # show the error between model and value to be calibrated on
    fig, ax = plt.subplots()
    img = ax.imshow(err, interpolation='none', cmap='nipy_spectral')

    # scatter the absolute minimum value of the error
    min_ind = np.where(err == err.min())
    plt.scatter(min_ind[1], min_ind[0], c='b')

    # scatter the N lowest error values
    N = 10
    n_lowest_val = err[np.unravel_index(err.argsort(axis=None), err.shape)][:N]
    n_lowest_x = np.unravel_index(err.argsort(axis=None), err.shape)[1][:N]
    n_lowest_y = np.unravel_index(err.argsort(axis=None), err.shape)[0][:N]

    plt.scatter(n_lowest_x, n_lowest_y, c='y')
    for i, txt in enumerate(n_lowest_val):
        plt.annotate(txt, (n_lowest_x[i], n_lowest_y[i]), color='white')

    # the distance between lowest values and OGGM values
    dist = np.hypot(n_lowest_x - [oggm_mustar],
                    n_lowest_y - [oggm_prcp_fac])

    # make normalized distances and errors and add them in order to get a
    # best tradeoff by weighting
    # see: http://stackoverflow.com/questions/5284646/rank-items-in-an-array-using-python-numpy
    dist_norm = (dist - np.min(dist)) / np.ptp(dist)
    n_lowest_val_norm = (n_lowest_val - np.min(n_lowest_val)) / \
                        np.ptp(n_lowest_val)
    norms_added = dist_norm + n_lowest_val_norm
    new_error = n_lowest_val[np.argmin(norms_added)]

    if new_error == n_lowest_val[0]:
        plt.title('{}: Parameter combinations stays the same.'.format(
            gdir.rgi_id)
        )
    else:
        plt.title('{}: Took {} instead of {}'.format(gdir.rgi_id,
                                                     new_error,
                                                     n_lowest_val[0]))

    # scatter the differential evolution result
    # plt.scatter((rd.x[0] - np.min(mrg)) * mrg_step,
    #            (rd.x[1] - np.min(prg)) * prg_step, c='g')

    # scatter the minimize result
    # plt.scatter((rm.x[0] - np.min(mrg)) * mrg_step,
    #            (rm.x[1] - np.min(prg)) * prg_step, c='r')

  
    # scatter what oggm finds - and error that this paramcombi would cause
    plt.scatter((oggm_mustar - np.min(mrg)) / mrg_step,
                (oggm_prcp_fac - np.min(prg)) / prg_step, c='y')

    oggm_mb, oggm_err = mustar_from_deltah(gdir=gdir,
                                           deltav_file=deltav_file,
                                           mustar_rg=np.arange(oggm_mustar,
                                                               oggm_mustar
                                                               + 0.01, 0.01),
                                           prcp_fac_rg=np.arange(
                                               oggm_prcp_fac, oggm_prcp_fac
                                               + 0.01, 0.01))
    plt.annotate(oggm_err[0][0],
                 ((oggm_mustar - np.min(mrg)) / mrg_step,
                 (oggm_prcp_fac - np.min(prg)) / prg_step))
    '''

    # make the colorbar and show
    '''
    plt.colorbar(img, ax=ax)
    plt.show()
    '''