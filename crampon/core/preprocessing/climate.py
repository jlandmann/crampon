""" Prepare the meteodata from netCDF4 """

from __future__ import division

import netCDF4 as nc
from glob import glob
import os
import salem
import crampon.cfg as cfg
import numpy as np
from crampon.core.models.massbalance import PastMassBalanceModel
from crampon.utils import date_to_year, GlacierDirectory
import crampon.utils
from oggm.core.preprocessing.climate import *
import itertools
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from crampon import entity_task
import logging

# temporary
from crampon.workflow import execute_entity_task

# Module logger
log = logging.getLogger(__name__)

# To (re)write:
# mb_climate_on_height, _get_ref_glaciers, _get_optimal_scaling_factor

"""
# Parameters to be kicked out later on
testprec_dir = os.path.join(os.getcwd(), 'data\\test\\prec')
testtemp_dir = os.path.join(os.getcwd(), 'data\\test\\temp')
shape_dir = os.path.join(os.getcwd(), 'data\\test\\shp')
aletsch_shp = shape_dir + '\\Aletsch\\G008032E46504N.shp'

testtemp = glob(testtemp_dir+'\\*.nc')[0]
testprec = glob(testtemp_dir+'\\*.nc')[0]
"""


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

'''
def past_mb_anytime(gdir, cali_params, ym_prod, cl):
    """
    
    Parameters
    ----------
    gdir
    ym_prod
    cl

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
    return np.abs(np.subtract(np.ones_like(mb)*-0.71586703*27, mb))
'''


@entity_task(log, writes=['mustar_from_mauro'])
def mustar_from_deltav(gdir, deltav=None, mustar_rg=None,
                       prcp_fac_rg=None, recurs=0):
    """
    Get mustar from a volume change.
    
    Parameters
    ----------
    gdir: oggm.utils.GlacierDirectory
        An OGGM GlacierDirectory.
    mustar_rg: list-like
        A range for the temperature sensitivity to be tested
    prcp_fac_rg: list-like
        A range for the precipitation correction factor to be tested
    recurs: int
        Number of recursions (needed if parameters bump into corners)
    
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
    ybeg = int(str(deltav[deltav.RGIId == gdir.rgi_id].REFERENCE_.iloc[0])[:4])
    yend = int(str(deltav[deltav.RGIId == gdir.rgi_id].BgnDate.iloc[0])[:4])

    # thickness change in meters of ice
    insitu_ally = deltav[deltav.RGIId == gdir.rgi_id]['THICKNESS_'].iloc[0] \
                  / 1000.
    ym_prod = list(itertools.product([ybeg], [9, 10, 11, 12])) +\
              list(itertools.product(np.arange(ybeg + 1, yend),
                                     np.arange(1, 13))) + \
              list(itertools.product([yend], [1, 2, 3, 4, 5, 6, 7, 8]))
    ##### HERE SHOULD GO A PROCEDURE TO ACCOUNT FOR UNCERTAINTIES IN DEM DATES

    #ym_prod = list(itertools.product(np.arange(years[gdir.rgi_id][0], years[gdir.rgi_id][1]),
    #                                 np.arange(1, 13)))    ##### STIMMT DAS? VON WANN GENAU SIND MAUROS DHMS=?
    majid = gdir.read_pickle('major_divide', div_id=0)
    cl = gdir.read_pickle('inversion_input', div_id=majid)[-1]

    mb = np.ones((len(mustar_rg), len(prcp_fac_rg)))
    i = 0
    #result_minimize = optimize.minimize(past_mb_anytime,
    #                                    [(np.mean(mustar_rg),
    #                                     np.mean(prcp_fac_rg))],
    #                                    args=(gdir, ym_prod, cl),
    #                                    bounds=((None, None), (None, None)))
    #result_de = optimize.differential_evolution(past_mb_anytime,
    #                                            [(np.min(mustar_rg),
    #                                              np.max(mustar_rg)),
    #                                             (np.min(prcp_fac_rg),
    #                                              np.max(prcp_fac_rg))],
    #                                            args=(gdir, ym_prod, cl))

    for (mi, mustar), (pi, prcp_fac) in p_comb:
        past_model = PastMassBalanceModel(gdir, mu_star=mustar,
                                          prcp_fac=prcp_fac)

        mb_monthly = []
        for y, m in ym_prod:
            ym = date_to_year(y, m)
            # Get the mass balance and convert to m of ice per year
            try:
                tmp = past_model.get_monthly_mb(cl['hgt'], ym) * \
                      cfg.SEC_IN_MONTHS[m - 1] * cfg.RHO / 1000.
            # It happens that e.g. RGI50-11.00638 has the 1st DEM date before
            #  the Meteo time series begins (OGGM problem!?)
            except IndexError:
                if i == 0:
                    print('Skipped year/month {},{} for glacier {}'.format(y,
                                                                           m,
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
    # Ineffective, as whole range is calculated again
    if (min_ind[0] == 0 or min_ind[1] == 0) and recurs == 0:
        #new_mrg = np.append(mustar_rg - np.abs(np.max(mustar_rg)), mustar_rg) # the range becomes negative, this makes no sense!!!
        #new_prg = np.append(prcp_fac_rg - np.abs(np.max(prcp_fac_rg)), prcp_fac_rg) # the range becomes negative, this makes no sense!!!
        # let's try to decrease the step width instead
        new_mrg = np.arange(np.min(mustar_rg), np.max(mustar_rg),
                            ((np.max(mustar_rg) - np.min(mustar_rg)) / (len(mustar_rg) - 1)) / 2.)
        new_prg = np.arange(np.min(prcp_fac_rg), np.max(prcp_fac_rg),
                            ((np.max(prcp_fac_rg) - np.min(prcp_fac_rg)) / (len(prcp_fac_rg) - 1)) / 2.)
        print('Cali param values bumped into lower bounds')
        print(new_mrg, new_prg)
        mb, err = mustar_from_deltav(gdir=gdir, deltav=deltav,
                                     mustar_rg=new_mrg,
                                     prcp_fac_rg=new_prg, recurs=1)
        min_ind = np.where(err == err.min())
        print(mustar_rg[min_ind[0]], prcp_fac_rg[min_ind[1]])
        return mb, err
    elif (min_ind[0] == err.shape[0] or min_ind[1] == err.shape[1]) and recurs == 0:
        new_mrg = np.append(mustar_rg, mustar_rg + np.abs(np.max(mustar_rg)))
        new_prg = np.append(prcp_fac_rg, prcp_fac_rg + np.abs(np.max(prcp_fac_rg)))
        print('Cali param values bumped into upper bounds')
        print(new_mrg, new_prg)
        mb, err = mustar_from_deltav(gdir=gdir, deltav=deltav,
                                     mustar_rg=new_mrg,
                                     prcp_fac_rg=new_prg, recurs=1)
        min_ind = np.where(err == err.min())
        print(mustar_rg[min_ind[0]], prcp_fac_rg[min_ind[1]])
        return mb, err

    # Error greater than 5%
    elif (np.sqrt([err[min_ind]][0]) / insitu_ally)[0] * 100 > 5. and recurs == 0:
        print('Error too big ({}\%) for {}'.format((np.sqrt([err[min_ind]][0]) / insitu_ally)[0] * 100, gdir.rgi_id))
        # let's try to decrease the step width instead
        new_mrg = np.arange(np.min(mustar_rg), np.max(mustar_rg),
                            ((np.max(mustar_rg) - np.min(mustar_rg)) / (len(mustar_rg) - 1)) / 2.)
        new_prg = np.arange(np.min(prcp_fac_rg), np.max(prcp_fac_rg),
                            ((np.max(prcp_fac_rg) - np.min(prcp_fac_rg)) / (len(prcp_fac_rg) - 1)) / 2.)
        mb, err = mustar_from_deltav(gdir=gdir, deltav=deltav,
                                     mustar_rg=new_mrg,
                                     prcp_fac_rg=new_prg, recurs=1)
        min_ind = np.where(err == err.min())
        print(mustar_rg[min_ind[0]], prcp_fac_rg[min_ind[1]])
        return mb, err

    else:
        print(mustar_rg[min_ind[0]], prcp_fac_rg[min_ind[1]])
        print('Error in Percent:{}'.format(
            (np.sqrt([err[min_ind]][0]) / insitu_ally)[0] * 100))

        for div_id in [0] + list(gdir.divide_ids):
            # Scalars in a small dataframe for later
            df = pd.DataFrame()
            df['rgi_id'] = [gdir.rgi_id]
            df['t_star'] = [None]
            df['mu_star'] = [mustar_rg[min_ind[0]]][0]
            df['prcp_fac'] = [prcp_fac_rg[min_ind[1]]][0]
            df['bias'] = [err[min_ind]][0]
            df['err_perc'] = (np.sqrt([err[min_ind]][0]) / insitu_ally)[0] * 100
            df.to_csv(gdir.dir+'\\mustar_from_mauro.csv',
                      index=False)

        return mb, err


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

    deltav_file = 'C:\\Users\\Johannes\\Desktop\\mauro_in_RGI_disguise_entities_old_and_new.shp'
    deltav = gpd.read_file(deltav_file)
    deltav = deltav[['RGIId', 'THICKNESS_', 'BgnDate', 'REFERENCE_']]

    # set ranges and steps for calibration parameters
    mrg_step = 0.25
    prg_step = 0.1
    mrg = np.arange(0., 80. + mrg_step, mrg_step)
    prg = np.arange(0.05, 2.5 + prg_step, prg_step)

    for gdir in gdirs:
        # read from OGGM
        #tdf = pd.read_csv(gdir.get_filepath('local_mustar'))
        #oggm_mustar = tdf.mu_star.iloc[0]
        #oggm_prcp_fac = tdf.prcp_fac.iloc[0]

        #mb, err, rm, rd = mustar_from_deltav(gdir=gdir, mustar_rg=mrg,
        #                                     prcp_fac_rg=prg)

        # if already calibrated, then skip
        if os.path.exists(gdir.dir + '\\mustar_from_mauro.csv'):
            gdirs.remove(gdir)


    #try:
    #    mb, err = mustar_from_deltav(gdir=gdir, deltav=deltav,
    #                                 mustar_rg=mrg, prcp_fac_rg=prg)
    #except (FileNotFoundError, ValueError) as ex:
    #    print('{} for {}'.format(ex, gdir.dir))
    #    errorlist.append([ex, gdir.dir])
    #    continue

    execute_entity_task(mustar_from_deltav, gdirs, deltav=deltav,
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

    oggm_mb, oggm_err = mustar_from_deltav(gdir=gdir,
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