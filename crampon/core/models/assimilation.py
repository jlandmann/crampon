"""Implement assimilation utilities."""
import crampon
from crampon import cfg
from crampon import utils
from typing import Optional, Union
from itertools import permutations, combinations
import pandas as pd
import geopandas as gpd
import numpy as np
from crampon.core.models.massbalance import BraithwaiteModel, HockModel, \
    PellicciottiModel, OerlemansModel, SnowFirnCover, \
    EnsembleMassBalanceModel, ParameterGenerator, DailyMassBalanceModel
from crampon.core.preprocessing import climate
from crampon import tasks, graphics
import datetime as dt
import copy
import xarray as xr
from scipy import stats, optimize
import scipy.linalg as spla
from sklearn.neighbors import KernelDensity
import glob
import os
from crampon.core.holfuytools import *
import logging
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
from properscoring import crps_ensemble
# _normpdf and _normcdf are are faster than scipy.stats.norm.pdf/cdf
from properscoring._crps import _normpdf, _normcdf

# todo: double code wit massbalance
prcp_fac_annual_cycle = climate.prcp_fac_annual_cycle(
            np.arange(1, sum(cfg.DAYS_IN_MONTH_LEAP) + 1))
prcp_fac_cyc_winter_mean = np.mean(np.hstack([
    prcp_fac_annual_cycle[-sum(cfg.DAYS_IN_MONTH_LEAP[-3:]):],
    prcp_fac_annual_cycle[:sum(cfg.DAYS_IN_MONTH_LEAP[:4])]]))
prcp_fac_cycle_multiplier = prcp_fac_annual_cycle / \
                                 prcp_fac_cyc_winter_mean

# snow line/albedo from Sentinel/Landsat, Holfuy mass balance/surface type/
# extrapolated melt pctl
assim_types = ['snowline_sen', 'snowline_ls', 'albedo_sen', 'albedo_ls',
               'holfuy_mb', 'holfuy_sfctype', 'holfuy_mpctl']

log = logging.getLogger(__name__)


class InitialConditions(object):
    """Use pymc3??"""
    def __init__(self):
        self.swe = None
        self.alpha = None
        self.mb = None
        self.tacc = None


class AssimilationData(object):
    """ Interface to all assimilation data of a glacier."""
    def __init__(self, gdir):

        self.gdir = gdir

        try:
            self._obj = xr.open_dataset(gdir.get_filepath('assim_data'))
        except FileNotFoundError:
            self._obj = self.initiate_assim_obj(self.gdir)

    def initiate_assim_obj(self, gdir: utils.GlacierDirectory) -> xr.Dataset:
        """
        # todo: why gdir here?
        Parameters
        ----------
        gdir: `py:class:crampon.GlacierDirectory`
            the GlacierD

        Returns
        -------

        """
        # todo: groups: years (so that the glacier geometry can change)
        # x and y shall come from the glacier grid
        # todo: should uncertainty be one value per date or spatially
        #  distributed? (depends on variable!e.g. for albedo, cloud
        #  probability could increase uncertainty)

        xr_ds = xr.Dataset({'albedo': (['x', 'y', 'date', 'method', 'member',
                                        'source'], ),
                            'MB': (['height', 'fl_id', 'date', 'model',
                                    'member', 'source'],),
                            'swe_prob': (['x', 'y', 'date', 'model', 'member',
                                          'source']),
                            },
                           coords={
                               'source': (['source', ],),
                               'date': (['date', ],),
                               'x': (['x', ], ),
                               'y': (['y', ], ),
                               'height': (['fl_id', ]),
                               'fl_id': (['fl_id', ]),
                               'member': (['member', ],),
                               'model': (['model', ],),
                                })
        return xr_ds

    def ingest_observations(self, obs: xr.DataArray or xr.Dataset) -> None:
        """
        Add observations to the assimilation data.

        Parameters
        ----------
        obs :

        Returns
        -------

        """

    def retrieve_observations(self, date1: pd.Timestamp or None = None,
                              date2: pd.Timestamp or None = None,
                              which: list or None = None) -> xr.Dataset:
        """
        Retrieve observation in a time span.

        Parameters
        ----------
        date1 :
        date2 :
        which :

        Returns
        -------

        """

    def append_to_file(self, path: str or None = None) -> None:
        """
        Carefully append new observations to file.
        # todo: does this make sense?
        Parameters
        ----------
        path :

        Returns
        -------

        """


def resample_weighted_distribution(samples, weights, n_samples=500, axis=-1):
    """
    Resample a weighted distribution.

    Can be useful e.g. for a faster calculation of the CRPS when using the
    particle filter.

    Parameters
    ----------
    samples : np.array
        Samples of a variable.
    weights : np.array
        Weights for the samples.
    n_samples : int
        How many samples should be generated?. Default: 500.
    axis: int
        Axis to sort along, relevant if samples/weights are more than 1D.
        Default: -1.

    Returns
    -------
    resamp: np.array
        Resampled
    """
    # todo: test this function, and test it for N-D arrays

    # get samples sorted
    idx = np.argsort(samples, axis=axis)

    # sort samples/weights
    samples = np.take_along_axis(samples, idx, axis=axis)
    weights = np.take_along_axis(weights, idx, axis=axis)

    # Resample by quantiles
    alphas = (np.arange(n_samples) + 0.5) / n_samples

    weights_cs = np.cumsum(weights, axis=-1)

    resamp = np.full((weights.shape[0], n_samples), np.nan)
    resamp_w = np.full((weights.shape[0], n_samples), np.nan)
    for k, a in enumerate(alphas):
        ix = np.where(((weights_cs[:, :-1] < a) & (a <= weights_cs[:, 1:])))
        resamp[:, k] = samples[ix]
        resamp_w[:, k] = weights[ix]

    return resamp, resamp_w


def get_prior_param_distributions(gdir, model, n_samples, fit='gauss',
                                  seed=None, std_scale_fac=1.0):
    """

    Parameters
    ----------
    gdir: crampon.utils.GlacierDirectory
        The glacier directory to be processed.
    model: crampon.core.models.massbalance.DailyMassBalanceModel
        The model to create the parameter prior for.
    n_samples: int
        Sample size to be drawn from the distribution fitted to the calibrated
        parameters.
    fit: {'gauss'} str
        Which distribution to fit to the calibrated parameters. Default:
        'gauss' (Gaussian distribution).
    seed: int or None
        Random seed for repeatable experiments. Default: None (no seed).
    std_scale_fac: float
        Factor to scale the standard deviation of the input distribution. This
        helps if the number of calibrated parameters is low. Default: 1.0 (
        do not extend).

    Returns
    -------
    params_prior: np.ndarray
        Prior parameters as array (N, M), where N is the sample size an M
        are the parameters in the order of mbm_model.cali_params_list.
    """
    np.random.seed(seed)

    pg = ParameterGenerator(gdir, model, latest_climate=True)
    calibrated = pg.single_glacier_params.values

    sign = None  # necessary for c0 (it's negative...logarithm!)
    if (model.__name__ == 'OerlemansModel') and (fit == 'lognormal'):
        sign = np.sign(calibrated)[:1, :]
        calibrated = np.abs(calibrated)

    if fit == 'gauss':
        params_prior = np.random.multivariate_normal(
            np.mean(calibrated, axis=0),
            np.cov(calibrated.T) * std_scale_fac,
            size=n_samples
        )
    elif fit == 'uniform':
        params_prior = np.array([np.random.uniform(
            np.mean(calibrated[:, i]) -
            std_scale_fac * (np.mean(calibrated[:, i]) -
                             np.min(calibrated[:, i])),
            np.mean(calibrated[:, i]) +
            std_scale_fac * (np.max(calibrated[:, i]) -
                             np.mean(calibrated[:, i])), n_samples)
            for i in range(calibrated.shape[1])]).T  # .T to make compatible
    elif fit == 'lognormal':
        # todo: is this multiplication with the scale factor correct?
        params_prior = np.random.multivariate_normal(
            np.log(np.mean(calibrated, axis=0)),
            np.cov(np.log(calibrated.T) * std_scale_fac),
            size=n_samples)
        params_prior = np.exp(params_prior)
    else:
        raise NotImplementedError('Parameter fit == {} not allowed. We cannot '
                                  'fit anything else than Gauss at the '
                                  'moment.'.format(fit))

    # give back the sign
    if sign is not None:
        params_prior = sign * params_prior
    return params_prior


def get_prior_param_distributions_gabbi(model, n_samples, fit='gauss',
                                        seed=None):
    """
    Get param priors according to Gabbi et al. (2014).

    Parameters
    ----------
    model :
    n_samples :
    fit :
    seed :

    Returns
    -------

    """

    gabbi_param_bounds = {
        'mu_ice': (3.0, 10.0),
        'mu_hock': (0.01, 3.6),
        'a_ice': (0.01, 0.0264),
        'tf': (0.01, 7.44),
        'srf': (0.01, 0.29),
        'c0': (-225., -5.),
        'c1': (1., 33.),
        'prcp_fac': (0.9, 2.1)
    }

    # todo: take over param bounds from model class/cfg
    # theta_prior = np.array([np.random.uniform(gabbi_param_bounds[p][0],
    #                                          gabbi_param_bounds[p][1],
    #                                          n_samples)
    #                        for p in model.cali_params_list]).T
    # theta_prior = np.array([np.abs(np.clip(np.random.normal(np.mean(
    #    gabbi_param_bounds[p]), np.abs(np.ptp(gabbi_param_bounds[p])/4.),
    #    n_samples),
    #    gabbi_param_bounds[p][0], gabbi_param_bounds[p][1]))
    #        for p in model.cali_params_list]).T
    if fit == 'gauss':
        # th 6 is empirical: we want (roughly) to define sigma such that 99% of
        # values are within the bounds
        np.random.seed(seed)
        theta_prior = np.array([np.clip(np.random.normal(np.mean(
            gabbi_param_bounds[p]), np.abs(np.ptp(gabbi_param_bounds[p]) / 6.),
            n_samples), gabbi_param_bounds[p][0], gabbi_param_bounds[p][1])
            for p in model.cali_params_list]).T
    elif fit == 'uniform':
        np.random.seed(0)
        theta_prior = np.array([np.random.uniform(gabbi_param_bounds[p][0],
                                                  gabbi_param_bounds[p][1],
                                                  n_samples)
                                for p in model.cali_params_list]).T
    else:
        raise ValueError('Value {} for fit parameter is not allowed.',
                         format(fit))
    return theta_prior


def calculate_ensemble_ranks(obs: np.ndarray, ens_preds: np.ndarray):
    """
    Calculate the ranks of observations in an ensemble prediction.

    This is used to prepare data for a rank histogram. Adapted from [1]_.

    Parameters
    ----------
    obs : np.array
        Array with observation values.
    ens_preds : np.array
        Array with ensemble predictions.

    Returns
    -------
    ranks: np.array

    References
    ----------
    .. [1] https://github.com/oliverangelil/rankhistogram/blob/master/ranky.py
    """

    combined = np.vstack((obs[np.newaxis], ens_preds))
    ranks = np.apply_along_axis(lambda x: stats.rankdata(x, method='min'), 0,
                                combined)
    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)

    for i in range(1, len(tie)):
        index = ranks[ties == tie[i]]
        ranks[ties == tie[i]] = [np.random.randint(index[j], index[j] + tie[i]
                                                   + 1, tie[i])[0] for j in
                                 range(len(index))]
    return ranks


def update_particle_filter(particles: np.ndarray, wgts: np.ndarray,
                           obs: float, obs_std: float,
                           truncate: float or None = None) -> np.ndarray:
    """
    Update the weights of the prior.

    See _[1] (p.423) for details.

    Parameters
    ----------
    particles: np.array
        Particles of the prior (m w.e. d-1).
    wgts: np.array
        Particles weights of the prior.
    obs: float
        Observation (m w.e. d-1)
    obs_std: float
        Observation uncertainty given as the standard deviation (m w.e. d-1).
    truncate: float or None
         Where the PDF shall be truncated: this is especially useful if the
         sign of the observation shall be kept. E.g. -0.01 m w.e. have been
         measured with an error of 0.02 m w.e., but we know for sure the MB is
         below zero. This means we want to clip off the PDF at 0.

    Returns
    -------
    new_weights: np.array
        New weights for the particles.

    References
    ----------
    .. [1] Labbe, R. (2018): Kalman and Bayesian models in Python.
           https://bit.ly/1Ndubgh
    """
    w = copy.deepcopy(wgts)
    if truncate is not None:
        # raise NotImplementedError
        # myclip_a = -0.17  # left clipping boundary
        # todo: check if signs are always correct
        # myclip_a = particles + obs
        # if truncate + obs < 0.: # clip right
        if obs < 0.:
            myclip_a = -np.inf
            myclip_b = truncate  # right clipping boundary
        else:  # clip left
            myclip_a = truncate
            myclip_b = np.inf  # right clipping boundary
        a, b = (myclip_a - obs) / obs_std, (myclip_b - obs) / obs_std
        # w *= stats.truncnorm.pdf(particles, a, b, obs, obs_std)
        w += np.log(stats.truncnorm.pdf(particles, a, b, obs, obs_std))
    else:
        # w *= stats.norm(particles, obs_std).pdf(obs)
        w += np.log(stats.norm(particles, obs_std).pdf(obs))
    w += 1.e-300  # avoid round-off to zero
    # new_wgts = w / np.sum(w)  # normalize
    new_wgts = np.exp(w) / np.sum(np.exp(w))  # normalize
    return new_wgts


def effective_n(w):
    """
    Calculate effective N.

    N is an approximation for the number of particles contributing meaningful
    information determined by their weight. When this number is small, the
    particle filter should resample the particles to redistribute the weights
    among more particles. See _[1] (p.424) for details. From _[2].

    Parameters
    ----------
    w: np.array
        Weights of all particles.

    Returns
    -------
    array:
        Array with effective Ns.

    References
    ----------
    _[1] : Labbe, R. (2018): Kalman and Bayesian models in Python.
           https://bit.ly/1Ndubgh
    _[2] : https://bit.ly/2M0z1ow
    """

    return 1. / np.sum(np.square(w), axis=0)


def get_effective_n_thresh(n_particles, ratio=0.5):
    """
    Get the threshold of effective particles in the Particle Filter.

    If the number off effective particles drops below this threshold,
    resampling is done. Here, we default it to half the size of the particles
    used in the Particle Filter (see _[1]).

    Parameters
    ----------
    n_particles: int
        Number of particles used for the Particle Filter.
    ratio: float
        Ratio of the total particle number that determines the size of
        effective particles. Default: 0.5 (see _[1]).

    Returns
    -------
    int
        The effective n particles threshold number.

    References
    ----------
    _[1]: Labbe, R. (2018): Kalman and Bayesian Filters in Python, p. 424.
            https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg
    """

    return int(ratio * n_particles)


def estimate_state(particles, weights):
    """
    Give weighted average & variance of an estimate of particles & weights.

    Parameters
    ----------
    particles: np.array
        Particles describing the state.
    weights: np.array
        Weights given to the respective particles.

    Returns
    -------
    tuple:
        Weighted average and variance of the weighted particles.
    """
    mean = np.average(particles, weights=weights, axis=0)
    var = np.average((particles - mean)**2, weights=weights, axis=0)
    return mean, var


def simple_resample(particles: np.ndarray, weights: np.ndarray) -> tuple:
    """
    Perform a `simple resampling` as in [1]_.

    Sample from the current particles as often as the total number particles.
    The probability of selecting any given particle should be proportional to
    its weight.

    Parameters
    ----------
    particles: array
        Array with particles to be resampled.
    weights: array
        Array with particle weights.

    Returns
    -------
    (particles, weights): tuple of (array, array)
        Tuple of the resampled particles and the reset weights.

    Reference
    ---------
     .. [1]: Labbe, R. (2018): Kalman and Bayesian Filters in Python, p. 424.
             https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg
    """

    n = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, np.random.random(n))
    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / n)
    return particles, weights


def stratified_resample(weights: np.array, n_samples: Optional[int] = None,
                        one_random_number: bool = False,
                        seed: Optional[int] = None) -> np.array:
    """
    Copied stratified_resample from filterpy, but extended by own choice of N.

    todo: Try and make a PR at filterpy
    Parameters
    ----------
    weights : np.array
        Array with particle weights.
    n_samples : int, optional
        How many samples to generate during the resampling procedure. If None,
        the amount of present weights will be sampled again. Default: None.
    one_random_number: bool
        Whether to use only one random number during the resampling to keep
        frequency of particle j less than 1 away from expected value N * w_j.
        Default: False
    seed: int, optional
        Which random number generator to use (for repeatable experiments).
        Default: None (non-repeatable).


    Returns
    -------
    indexes: np.array
        Array with the indices to be resampled.
    """

    if n_samples is None:
        n_samples = len(weights)
    else:
        n_samples = int(n_samples)
    # make N subdivisions, chose a random position within each one
    np.random.seed(seed)
    if one_random_number is True:
        rn = np.random.random(1)  # version Hansruedi Künsch suggested to
        # keep frequency of particle j less than 1 away from expected value
        # N * w_j
    else:
        rn = np.random.random(n_samples)  # filterpy version
    positions = (rn + range(n_samples)) / n_samples
    indexes = np.zeros(n_samples, 'i')
    cumulative_sum = np.cumsum(weights)

    # avoid precision error
    cumulative_sum[-1] = 1.

    i, j = 0, 0
    while i < n_samples:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def resample_from_index(particles, weights, indices):
    """
    Perform a resampling from given indices, as in [1]_.

    Parameters
    ----------
    particles: array
        Array with particles to be resampled.
    weights: array
        Array with particle weights.
    indices: array
        Indices used for choosing the resampled particles.

    Returns
    -------
    (particles, weights): tuple of (array, array)
        Tuple of the resampled particles and the reset weights.

    Reference
    ---------
     .. [1]: Labbe, R. (2018): Kalman and Bayesian Filters in Python, p. 424.
             https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg
    """
    particles[:] = particles[indices]
    weights[:] = weights[indices]
    # particles[:] = particles[indices[:], range(indices.shape[1])]
    # weights[:] = weights[indices[:], range(indices.shape[1])]
    weights.fill(1.0 / len(weights))
    # weights.fill(1.0 / weights.shape[0])
    return particles, weights


def resample_from_index_augmented(particles, weights, indices):
    """
    Perform a resampling from given indices, as in [1]_.

    Parameters
    ----------
    particles: array
        Array with particles to be resampled.
    weights: array
        Array with particle weights.
    indices: array
        Indices used for choosing the resampled particles.

    Returns
    -------
    (particles, weights): tuple of (array, array)
        Tuple of the resampled particles and the reset weights.

    Reference
    ---------
     .. [1]: Labbe, R. (2018): Kalman and Bayesian Filters in Python, p. 424.
             https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg
    """
    particles = particles[:, indices, :]
    weights = weights[:, indices]
    weights.fill(1.0 / len(weights))
    return particles, weights


def resample_particles(particles, weights,
                       resamp_method=stratified_resample,
                       n_eff_thresh=None):
    """
    Effectively resample a given pair of particles and weights with a method.

    This uses a method from filterpy.monte_carlo to generate indices from which
    the resampling is done.

    Parameters
    ----------
    particles: array
        2D array where first dimension is the particles, second dimension the
        spatial discretization.
    weights: array
        2D array where first dimension is the weights, second dimension the
        spatial discretization.
    resamp_method: method from filterpy.monte_carlo
        A resampling method from filterpy.monte_carlo. Default:
        stratified_resample.
    n_eff_thresh: int or None
        The actual threshold of effective particles below which resampling is
        done. If None, the value defaults to half the number of particles.
        Default: None.

    Returns
    -------
    (particles, weights): tuple of (array, array)
        Tuple of the resampled particles and the reset weights.
    """
    if n_eff_thresh is None:
        n_eff_thresh = get_effective_n_thresh(particles.shape[0])

    n_eff = effective_n(weights)
    if (n_eff < n_eff_thresh).any():
        need_resamp_ix = np.where(n_eff < n_eff_thresh)
        print(len(need_resamp_ix))
        for nri in need_resamp_ix[0]:
            try:
                indices = resamp_method(weights[:, nri])
            except IndexError:
                worked = False
                indices = None
                while worked is False:
                    try:
                        # avoid some stupid precision error
                        resamp_input = weights[:, nri] * (
                                    1. / np.cumsum(weights[:, nri])[-1])
                        indices = resamp_method(resamp_input)
                        worked = True
                        print('worked')
                    except IndexError:
                        indices = None
                        print('did not work')
                        pass
            particles[:, nri], weights[:, nri] = resample_from_index(
                particles[:, nri], weights[:, nri], indices)

    return particles, weights


class OBSPriorPostFigure(object):
    """
    Some figure which I can't remember anymore.
    """
    pass
    """
    fig, [ax1, ax2] = plt.subplots(2, sharex=True)
    # swe on ax1
    avg = np.average(swe_mod, weights=self.weights[0, :], axis=1)
    ax1.errorbar(np.arange(self.particles.shape[0]), avg, yerr=np.sqrt(
        np.average((swe_mod - np.atleast_2d(avg).T) ** 2,
                   weights=self.weights[0, :], axis=1)), label='MOD PRIOR')

    ax1.legend()
    lc = ['g', 'b', 'k']
    c = ["g", "b", 'k']
    xs = np.arange(self.particles.shape[0])
    ys_o = fsca_obs_mean
    ys_m = np.average(fsca_mod, weights=self.weights[0, :], axis=1)
    std_m = np.sqrt(np.average((fsca_mod - np.atleast_2d(ys_m).T) ** 2,
                               weights=self.weights[0, :], axis=1))
    std_m_top = ys_m + std_m
    std_m_btm = ys_m - std_m
    std_o = fsca_obs_std
    std_o_top = ys_o + std_o
    std_o_btm = ys_o - std_o
    ax2.plot(xs, ys_m, linestyle='-', color=lc[0], lw=2, zorder=100,
             label='MOD fSCA PRIOR')
    ax2.fill_between(xs, std_m_btm, std_m_top, facecolor=c[0], alpha=0.3,
                     zorder=100)
    ax2.plot(xs, ys_o, linestyle='-', color=lc[1], lw=2, label='OBS fSCA')
    ax2.fill_between(xs, std_o_btm, std_o_top, facecolor=c[1], alpha=0.3)
    """


class AEPFOverviewFigure(object):
    """AEPF overview figure."""
    def __init__(self, colors=None, models=None):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20, 15),
                                                 sharex='all')
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4

        if models is None:
            self.models = [eval(m) for m in cfg.MASSBALANCE_MODELS]
        else:
            self.models = models

        self.n_models = len(self.models)

        if colors is None:
            self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        else:
            self.colors = colors

    def add_model_pred_post(self, date):
        """
        Plot prediction and posterios by model.

        Parameters
        ----------
        date :

        Returns
        -------

        """

        # plot prediction by model
        y_jitter = np.array([pd.Timedelta(hours=2 * td) for td in
                             np.linspace(-2.0, 2.0, self.n_models)])
        y_vals = np.array([date] * self.n_models) + y_jitter
        self.ax1.scatter(y_vals, mod_pred_mean, c=self.colors[:self.n_models])
        self.ax1.errorbar(y_vals, mod_pred_mean, yerr=mod_pred_std, fmt='o',
                          zorder=0)

        aepf.plot_state_errorbars(self.ax1, date, colors=['y'],
                                  space_ix=s_index[voi])

        if obs is not None:
            # plot obs
            self.ax1.errorbar(date, obs[0][voi], yerr=obs_std[0][voi], c='k',
                         marker='o', fmt='o')

    def add_param_errorbars(self, date, pmean, pstd, modelname):
        """
        Add parameter errorbars to the plot.

        Parameters
        ----------
        date :
        pmean :
        pstd :
        modelname :

        Returns
        -------

        """
        model_ix = [m.__name__ for m in self.models].index(modelname)
        self.ax2.errorbar(date, pmean, yerr=pstd, fmt='o',
                          c=self.colors[model_ix])

    def add_particles_per_model(self, date, n_model_particles):
        """

        Parameters
        ----------
        date :
        n_model_particles :

        Returns
        -------

        """

        for ppi, pp in enumerate(n_model_particles):
            if ppi == 0:
                self.ax3.bar(date, pp, color=self.colors[ppi])
            else:
                self.ax3.bar(date, pp, bottom=np.sum(n_model_particles[:ppi]),
                             color=self.colors[ppi])

    def add_obs_pred_post(self, date):
        """
        Add observation, prediction and posterior.

        Parameters
        ----------
        date :

        Returns
        -------

        """
        aepf.plot_state_errorbars(ax4, date - pd.Timedelta(hours=3),
                                  var_ix=alpha_ix, colors=['gold'],
                                  space_ix=list(range(int(len(h) / 2))))
        aepf.plot_state_errorbars(ax4, date - pd.Timedelta(hours=3),
                                  var_ix=alpha_ix, colors=['y'], space_ix=list(
                range(int(len(h) / 2), len(h))))

    def show(self):
        """
        Mimic plt.show()

        Returns
        -------

        """
        plt.show()


class State:
    """Basic class to represent the state of a system."""
    def __init__(self):
        self.n_spatial_dims = 0
        self.n_state_vars = 0
        pass

    def from_dataset(self, dataset, time=None):
        """
        Create a state from an xarray dataset.

        Parameters
        ----------
        dataset: xr.Dataset
        time: str or pd.Timestamp or None
            Time in the Dataset from which the state shall be generated. Can
            be None if there is not time axis in the dataset. Default: None.

        Returns
        -------

        """


class AugmentedState(np.ndarray):
    """Basic class to represent the state of a system."""
    pass


class GlacierState(State):
    """ Represent the state of a glacier in CRAMPON."""
    pass


class AugmentedGlacierState(GlacierState):
    """ Augmented glacier state representation."""
    pass


class MultiModelGlacierState(GlacierState):
    """ Represent a multi-model glacier state."""
    pass


class AugmentedMultiModelGlacierState(MultiModelGlacierState):
    """Represent an augmented multi-model glacier state."""
    pass


class AssimilationMethod(object):
    """Ducktyping interface to a general assimilation method."""

    def predict(self):
        """
        A prediction method.

        Returns
        -------

        """
        raise NotImplementedError

    def update(self):
        """
        An update method.
        Returns
        -------

        """
        raise NotImplementedError


class SmoothingMethod(AssimilationMethod):
    """Interface to smoothing assimilation methods."""

    def predict(self):
        """
        A prediction method.

        Returns
        -------

        """
        raise NotImplementedError

    def update(self):
        """
        An update method.
        Returns
        -------

        """
        raise NotImplementedError


class FilteringMethod(AssimilationMethod):
    """Interface to filtering assimilation methods."""

    def predict(self):
        """
        A prediction method.

        Returns
        -------

        """
        raise NotImplementedError

    def update(self):
        """
        An update method.
        Returns
        -------

        """
        raise NotImplementedError


# class ParticleFilter(FilteringMethod):
#    """Interface to the Particle Filter."""


class ParticleFilter(object):
    """
    A particle filter to model the state of the model as it develops in time.
    Adapted from Urban Analytics _[1].

    References
    ----------
    _[1] : https://bit.ly/2M0z1ow
    """

    def __init__(self, n_particles: int, particles: np.ndarray or None = None,
                 weights: np.ndarray or None = None,
                 n_eff_thresh: int or None = None,
                 do_save: bool = True, do_plot: bool = True) -> None:
        """
        Initialise.

        Parameters
        ----------
        n_particles: int
            The number of particles used for filtering.
        particles: (N, M) array-like, optional
            Particles to start off with. First dimension should be n_particles
            long, second dimension should be the spatial discretization.
        weights: (N, M) array-like, optional
            Weights to start off with. First dimension should be n_particles
            long, second dimension should be the spatial discretization. If
            given, must have the same dimensions as particles.
        n_eff_thresh: int, optional
            The actual threshold of effective particles below which resampling
            is done. If None, the value defaults to half the number of
            particles. Default: None. (see _[1]).
        do_save: bool
            True if results should be saved when using the `step` method.
        do_plot: bool
            True if results should be plotted when using the `step` method.

        References
        ----------
        _[1]: Labbe, R. (2018): Kalman and Bayesian Filters in Python, p. 424.
            https://drive.google.com/open?id=0By_SW19c1BfhSVFzNHc0SjduNzg

        """

        self.n_particles = n_particles

        if particles is not None:
            self.particles = particles
        else:
            self.particles = None

        if weights is not None:
            self.weights = weights
            # small check
            assert self.weights.shape == self.particles.shape
        elif (weights is None) and (particles is not None):
            self.weights = np.ones_like(self.particles) / self.n_particles
        else:
            self.weights = None

        if n_eff_thresh is None:
            self.n_eff_thresh = get_effective_n_thresh(self.n_particles)
        else:
            self.n_eff_thresh = n_eff_thresh

        self.do_save = do_save
        self.do_plot = do_plot

        '''
        for key, value in filter_params.items():
            setattr(self, key, value)
        self.time = 0
        self.number_of_iterations = model_params['batch_iterations']
        self.base_model = Model(model_params)
        self.models = list([copy.deepcopy(self.base_model) for _ in
                            range(self.number_of_particles)])
        self.dimensions = len(self.base_model.agents2state())
        self.states = np.zeros((self.number_of_particles, self.dimensions))
        self.weights = np.ones(self.number_of_particles)
        self.indexes = np.zeros(self.number_of_particles, 'i')
        if self.do_save:
            self.active_agents = []
            self.means = []
            self.mean_errors = []
            self.variances = []
            self.unique_particles = []
        '''

    def __repr__(self):
        summary = ['<crampon.core.models.assimilation.ParticleFilter>']
        summary += ['  N Particles: ' + str(self.n_particles)]
        summary += ['  N effective threshold: ' + str(self.n_eff_thresh)]
        summary += ['  Average effective particles: ' + str(
            int(np.mean(effective_n(self.weights))))]
        summary += ['  Filter dimensions: ' + str(self.particles.shape)]
        summary += ['  Do save: ' + str(self.do_save)]
        summary += ['  Do plot: ' + str(self.do_plot)]
        summary += ['  Particle mean: ' + str(np.nanmean(self.particles))]
        summary += ['  Particle median: ' + str(np.nanmedian(self.particles))]
        summary += ['  Particle variance: ' + str(np.nanvar(self.particles))]
        summary += ['  Particle stddev: ' + str(np.nanstd(self.particles))]
        return '\n'.join(summary) + '\n'

    @classmethod
    def from_file(cls, path: str):
        """
        Generate a Particle Filter object from a netCDF file.

        Parameters
        ----------
        path: str
            Path to the netCDF file from which the class shall be instantiated.

        Returns
        -------
        pf: ParticleFilter
            A ParticleFilter object.
        """
        ds = xr.open_dataset(path)
        return cls.from_dataset(ds)

    @staticmethod
    def from_dataset(xr_ds: xr.Dataset):
        """
        Generate a Particle Filter object from an xarray Dataset.

        Parameters
        ----------
        xr_ds: xarray.Dataset
            An xarray Dataset from which the class shall be instantiated.

        Returns
        -------
        pf: ParticleFilter
            A ParticleFilter object.
        """

        if 'n_eff_thresh' in xr_ds.attrs:
            n_eff_thresh = xr_ds.attrs['n_eff_thresh']
        else:
            n_eff_thresh = None
        if 'save' in xr_ds.attrs:
            save = xr_ds.attrs['save']
        else:
            save = None
        if 'plot' in xr_ds.attrs:
            plot = xr_ds.attrs['plot']
        else:
            plot = None
        return ParticleFilter(n_particles=xr_ds.particle_id.size,
                              particles=xr_ds.particles.values,
                              weights=xr_ds.weights.values,
                              n_eff_thresh=n_eff_thresh,
                              do_save=save, do_plot=plot)

    def to_file(self, path: str) -> None:
        """
        Write to netCDF file.

        Parameters
        ----------
        path: str
            Path to write the object to.

        Returns
        -------
        None
        """
        ds = self.to_dataset()
        ds.encoding['zlib'] = True
        ds.to_netcdf(path)

    def to_dataset(self) -> xr.Dataset:
        """
        Export as xarray Dataset.

        Returns
        -------
        ds: xarray.Dataset
            A Dataset containing the class attributes as variables.
        """
        xr_ds = xr.Dataset(
            {'particles': (['particle_id', 'fl_id'], self.particles),
             'weights': (['particle_id', 'fl_id'], self.weights)},
            coords={'particle_id': (['particle_id', ],
                                    range(self.n_particles)),
                    'fl_id': (range(self.particles.shape[1]))},
            attrs={'n_eff_thresh': self.n_eff_thresh, 'plot': self.do_plot,
                   'save': self.do_save})
        return xr_ds

    def predict_gaussian(self, pred: np.ndarray, pred_std: np.ndarray) -> None:
        """
        Predict with model runs, assuming their distribution is Gaussian.

        Parameters
        ----------
        pred: np.array
            Model prediction of one time step.
        pred_std: np.array
            Model error as standard deviation.

        Returns
        -------
        None
        """
        assert self.particles.shape[1] == pred.shape[0]
        self.particles = predict_particle_filter(self.particles, pred,
                                                 pred_std)

    def predict_gamma(self, pred: np.ndarray) -> None:
        """
        Predict with model runs, assuming they are Gamma-distributed.

        Returns
        -------

        """
        raise NotImplementedError

    def predict_random(self, pred: np.ndarray) -> None:
        """
        Predict with model runs, adding them randomly to the particles.

        Parameters
        ----------
        pred: array
            Model prediction.

        Returns
        -------
        None.
        """
        pred_rand = create_random_particles_choice(pred, self.n_particles)
        self.particles += pred_rand

    def update_gaussian(self, obs: float or np.ndarray,
                        obs_std: float or np.ndarray) -> None:
        """
        Update the weights of the prior, assuming observations are Gaussian.

        See _[1] (p.423) for details.

        Parameters
        ----------
        obs: float
            Observation
        obs_std: float
            Observation uncertainty given as the standard deviation.

        Returns
        -------
        new_weights: np.array
            New weights for the particles.

        References
        ----------
        _[1] : Labbe, R. (2015): Kalman and Bayesian models in Python.
               https://bit.ly/1Ndubgh
        """
        w = copy.deepcopy(self.weights)
        # w *= stats.norm(self.particles, obs_std).pdf(obs)
        w += np.log(stats.norm(self.particles, obs_std).pdf(obs))
        w += 1.e-300  # avoid round-off to zero
        # self.weights = w / np.sum(w)  # normalize
        self.weights = np.exp(w) / np.sum(np.exp(w))  # normalize

    def update_truncated_gaussian(self, obs: float, obs_std: float,
                                  truncate: float) -> None:
        """
        Update weights of the prior, assuming observations are a truncated
        Gaussian distribution.

        # todo: this is still in experimental phase!!!

        See _[1] (p.423) for details.

        Parameters
        ----------
        obs: float
            Observation.
        obs_std: float
            Observation uncertainty given as the standard deviation.
        truncate: float
             Where the PDF shall be truncated: this is especially useful if the
             sign of the observation shall be kept. E.g. -0.01 m w.e. have been
             measured with an error of 0.02 m w.e., but we know for sure the MB
             is below zero. This means we want to clip off the PDF at 0.

        Returns
        -------
        new_weights: np.array
            New weights for the particles.

        References
        ----------
        _[1] : Labbe, R. (2018): Kalman and Bayesian models in Python.
               https://bit.ly/1Ndubgh
        """
        w = copy.deepcopy(self.weights)
        w[:] = 1.
        # todo: check if signs are always correct
        if obs < 0.:
            clip_a = -np.inf
            clip_b = truncate  # right clipping boundary
        else:  # clip left
            clip_a = truncate
            clip_b = np.inf  # right clipping boundary
        a, b = (clip_a - obs) / obs_std, (clip_b - obs) / obs_std
        # todo: the truncnorm still needs to be normalized?
        # w *= stats.truncnorm.pdf(self.particles, a, b, obs, obs_std)
        w += np.log(stats.truncnorm.pdf(self.particles, a, b, obs, obs_std))
        w += 1.e-300  # avoid round-off to zero
        self.weights = np.exp(w) / np.sum(np.exp(w))  # normalize

    def update_gamma(self, obs: float, obs_std: float,
                     truncate: float) -> None:
        """
        Update weights of prior, assuming observations are Gamma-distributed.

        See _[1] (p.423) for details.

        Parameters
        ----------
        obs: float
            Observation.
        obs_std: float
            Observation uncertainty given as the standard deviation.
        truncate: float
             Where the PDF shall be truncated: this is especially useful if the
             sign of the observation shall be kept. E.g. -0.01 m w.e. have been
             measured with an error of 0.02 m w.e., but we know for sure the MB
             is below zero. This means we want to clip off the PDF at 0.

        Returns
        -------
        new_weights: np.array
            New weights for the particles.

        References
        ----------
        _[1] : Labbe, R. (2015): Kalman and Bayesian models in Python.
               https://bit.ly/1Ndubgh
        """
        raise NotImplementedError
        w = copy.deepcopy(self.weights)
        a = (mu / sigma) ** 2  # shape factor
        # w *= stats.gamma.pdf(self.particles, a, obs, obs_std)
        w += np.log(stats.gamma.pdf(self.particles, a, obs, obs_std))
        w += 1.e-300  # avoid round-off to zero
        self.weights = np.exp(w) / np.sum(np.exp(w))  # normalize

    def estimate_gaussian_state(self) -> tuple:
        """
        Give weighted average & variance of particles & weights.

        Returns
        -------
        tuple:
            Weighted average and variance of the weighted particles.
        """
        return estimate_state(self.particles, self.weights)

    def resample_if_needed(self, resamp_method=stratified_resample,
                           n_eff_thresh: int or None = None) -> None:
        """
        Resample the particles, if the effective n threshold is exceeded.

        # todo: the handling of the condition (effective n thresh exceeded
            yes/no) is in the function resample_particles...this should
            probably be changed.

        Parameters
        ----------
        resamp_method: method from filterpy.monte_carlo
            The method how to resample the particles. See _[1] for reference.
            Default: stratified_resample
        n_eff_thresh: int
            Threshold for the effective number of particles.

        Returns
        -------
        None

        References
        ----------
        _[1]: Labbe, R. (2015): Kalman and Bayesian models in Python, p. 432.
               https://bit.ly/1Ndubgh
        """

        if n_eff_thresh is None:
            n_eff_thresh = self.n_eff_thresh

        p, w = resample_particles(self.particles, self.weights,
                                  resamp_method=resamp_method,
                                  n_eff_thresh=n_eff_thresh)
        self.particles = p
        self.weights = w

    def step(self, pred: np.ndarray, date: pd.Timestamp,
             obs: np.ndarray or None = None,
             obs_std: np.ndarray or None = None,
             predict_method: str = 'random',
             update_method: str = 'gauss') -> None:
        """
        Make a step forward in time.

        Parameters
        ----------
        pred: (N, M) array_like
            Array with model predictions. N is the particles dimension, M, is
            the discretization dimension.
        date: pd.Timestamp
            Date/time for which the step is valid.
        obs: (N, M), (M,) array_like or None
            # todo: adjust shapes if observations don't have n_particles yet!?
            Array with observations or None. It can be either the raw
            observations in shape (N, M) or one observation value per
            discretization (M,). If None, the update step is skipped. Default:
            None.
        obs_std: array_like or None
            Array with observation standard deviations or None. It can be
            either the raw observations in shape (N, M) or one observation
            value per discretization node (M,). Only makes sense in combination
            with `obs`. Default: None.
        predict_method: str
            Method used to make a prediction on the particles: Must be either
            of ['gauss'|'random'|'gamma']. If 'gauss', then a Gaussian particle
            distribution will the generated from the model predictions. If
            'gamma', a Gamma distribution will be generated for the
            observations. If 'random', all given observations will be added
            randomly on existing particles.
        update_method: str
            Method used for updating the prior with the likelihood. Must be
            either of ['gauss'|'truncgauss'|'random'|'gamma']. Default:
            'gauss'.

        Returns
        -------
        None
        """

        # 1) predict
        if predict_method == 'gauss':
            # todo: find a way to insert std dev here
            self.predict_gaussian(pred, np.nanstd(pred, axis=1))
        elif predict_method == 'random':
            self.predict_random(pred)
        elif predict_method == 'gamma':
            self.predict_gamma(pred)
        else:
            raise ValueError(
                'Prediction method {} does not exist.'.format(predict_method))

        # 2) update, if there is an observation
        if obs is not None:

            if update_method == 'gauss':
                self.update_gaussian(obs, obs_std)
            elif update_method == 'truncgauss':
                self.update_truncated_gaussian(obs, obs_std)
            elif update_method == 'random':
                raise NotImplementedError
            elif update_method == 'gamma':
                self.update_gamma(obs, obs_std)
            else:
                raise ValueError(
                    'Update method {} does not exist.'.format(update_method))
        else:
            log.info('No observation available on {}.'.format(date))

        # 3) resample, if needed
        self.resample_if_needed()

        # 4) save, if wanted
        if self.do_save is True:
            raise NotImplementedError
            # self.to_file()

        # 5) plot, if wanted
        if self.do_plot is True:
            self.plot()

        # if self.do_save:
        #    self.save()
        # if self.do_ani:
        #    self.ani()

        # if self.plot_save:
        #    self.p_save()

    def plot(self):
        """
        Plot the state somehow.

        Returns
        -------

        """
        pass

    """
    def reweight(self):
        '''
        Reweight

        DESCRIPTION
        Add noise to the base model state to get a measured state. Calculate
        the distance between the particle states and the measured base model
        state and then calculate the new particle weights as 1/distance.
        Add a small term to avoid dividing by 0. Normalise the weights.
        '''
        measured_state = (self.base_model.agents2state()
                          + np.random.normal(0, self.model_std ** 2,
                                             size=self.states.shape))
        distance = np.linalg.norm(self.states - measured_state, axis=1)
        self.weights = 1 / (distance + 1e-99) ** 2
        self.weights /= np.sum(self.weights)
        return

    def resample(self):
        '''
        Resample

        DESCRIPTION
        Calculate a random partition of (0,1) and then
        take the cumulative sum of the particle weights.
        Carry out a systematic resample of particles.
        Set the new particle states and weights and then
        update agent locations in particle models using
        multiprocessing methods.
        '''
        offset_partition = ((np.arange(self.number_of_particles)
                             + np.random.uniform()) / self.number_of_particles)
        cumsum = np.cumsum(self.weights)
        i, j = 0, 0
        while i < self.number_of_particles:
            if offset_partition[i] < cumsum[j]:
                self.indexes[i] = j
                i += 1
            else:
                j += 1

        self.states[:] = self.states[self.indexes]
        self.weights[:] = self.weights[self.indexes]

        self.unique_particles.append(len(np.unique(self.states, axis=0)))

        self.models = pool.starmap(assign_agents, list(
            zip(range(self.number_of_particles),
                [self] * self.number_of_particles)))

        return

    def save(self):
        '''
        Save

        DESCRIPTION
        Calculate number of active agents, mean, and variance
        of particles and calculate mean error between the mean
        and the true base model state. Plot active agents,mean
        error and mean variance.
        '''
        self.active_agents.append(
            sum([agent.active == 1 for agent in self.base_model.agents]))

        active_states = [agent.active == 1 for agent in self.base_model.agents
                         for _ in range(2)]

        if any(active_states):
            mean = np.average(self.states[:, active_states],
                              weights=self.weights, axis=0)
            variance = np.average((self.states[:, active_states] - mean) ** 2,
                                  weights=self.weights, axis=0)

            self.means.append(mean)
            self.variances.append(np.average(variance))

            truth_state = self.base_model.agents2state()
            self.mean_errors.append(
                np.linalg.norm(mean - truth_state[active_states], axis=0))

        return

    def p_save(self):
        '''
        Plot Save

        DESCRIPTION
        Plot active agents, mean error and mean variance.
        '''
        plt.figure(2)
        plt.plot(self.active_agents)
        plt.ylabel('Active agents')
        plt.show()

        plt.figure(3)
        plt.plot(self.mean_errors)
        plt.ylabel('Mean Error')
        plt.show()

        plt.figure(4)
        plt.plot(self.variances)
        plt.ylabel('Mean Variance')
        plt.show()

        plt.figure(5)
        plt.plot(self.unique_particles)
        plt.ylabel('Unique Particles')
        plt.show()

        print('Max mean error = ', max(self.mean_errors))
        print('Average mean error = ', np.average(self.mean_errors))
        print('Max mean variance = ', max(self.variances[2:]))
        print('Average mean variance = ', np.average(self.variances[2:]))

    def ani(self):
        '''
        Animate

        DESCRIPTION
        Plot the base model state and some of the
        particles. Only do this if there is at least 1 active
        agent in the base model. We adjust the markersizes of
        each particle to represent the weight of that particle.
        We then plot some of the agent locations in the particles
        and draw lines between the particle agent location and
        the agent location in the base model.
        '''
        if any([agent.active == 1 for agent in self.base_model.agents]):

            plt.figure(1)
            plt.clf()

            markersizes = self.weights
            if np.std(markersizes) != 0:
                markersizes *= 4 / np.std(markersizes)  # revar
            markersizes += 8 - np.mean(markersizes)  # remean

            particle = -1
            for model in self.models:
                particle += 1
                markersize = np.clip(markersizes[particle], .5, 8)
                for agent in model.agents[:self.agents_to_visualise]:
                    if agent.active == 1:
                        unique_id = agent.unique_id
                        if self.base_model.agents[unique_id].active == 1:
                            locs = np.array(
                                [self.base_model.agents[unique_id].location,
                                 agent.location]).T
                            plt.plot(*locs, '-k', alpha=.1, linewidth=.3)
                            plt.plot(*agent.location, 'or', alpha=.3,
                                     markersize=markersize)

            for agent in self.base_model.agents:
                if agent.active == 1:
                    plt.plot(*agent.location, 'sk', markersize=4)

            plt.axis(np.ravel(self.base_model.boundaries, 'F'))
            plt.pause(1 / 4)
    """


# class AugmentedEnsembleParticleFilter(object):
class AugmentedEnsembleParticleFilter(object):  # ParticleFilter):
    """
    The interface to a multiple model particle filter as CRAMPON uses it.

    Attributes
    ----------
    models: list
        A list of mass balance model instances to use.
    n_models: int
        The total numbers of models. Should be low to let the adaptive
        resampling work.
    n_tot: int
        Total number of particles from all single models together.
    """

    def __init__(self, models: list, n_particles: int, spatial_dims: tuple,
                 n_phys_vars: int, n_aug_vars: int,
                 model_prob: Optional[list] = None,
                 model_prob_log: Optional[list] = None,
                 name: Optional[str] = None):
        """
        Instantiate.

        Parameters
        ----------
        models: np.array
            List of models to use.
        n_particles: int
            Total number of particles to be used.
        spatial_dims: tuple
            Tuple of shape of the spatial discretization. E.g. for a glacier
            with 100 flow line nodes, this would be (100,).
        n_phys_vars: int
            Number of variables in the physical state.
        n_aug_vars: int
            Number of variables to augment the state with.
        model_prob: np.array or None
            Array of model probabilities 0 <= p <= 1 (same length as `models`).
            If None, all models are equally likely. Default: None.

        Returns
        -------
        None
        """

        # todo: let the call to super make sense
        # super().__init__(n_particles, do_plot=False, do_save=False)

        # some inital settings
        self._M_t_all = None
        self._S_t_all = None

        self.models = models
        self.model_names = [m.__name__ for m in self.models]
        self.n_models = len(models)
        self.model_range = np.arange(self.n_models)
        self.n_tot = self.n_particles = n_particles

        if len(spatial_dims) > 1:
            raise NotImplementedError('Only 1D discretizations are allowed '
                                      'at the moment.')
        self.spatial_dims = spatial_dims
        self.n_phys_vars = n_phys_vars
        self.n_aug_vars = n_aug_vars
        # tell which indices in the actual state are occupied by what
        self.state_indices = {
            'm': 0,  # model index
            'xi': np.arange(1, self.n_phys_vars),  # phys. state
            'theta': np.arange(self.n_phys_vars, self.n_phys_vars +
                               self.n_aug_vars)  # parameters
                         }

        if model_prob is not None and model_prob_log is not None:
            raise ValueError('Arguments "model_prob" and "model_prob_log" are '
                             'mutually exclusive.')
        if model_prob is None and model_prob_log is None:
            self._model_prob_log = np.repeat(np.log(1. / self.n_models),
                                             self.n_models)
            self._n_model_particles = self.n_tot / self.n_models
        elif model_prob is not None:
            self._model_prob_log = np.array(np.log(model_prob))
            self._n_model_particles = \
                [int(n) for n in self.model_prob * self.n_tot]
        elif model_prob_log is not None:
            self._model_prob_log = np.array(model_prob_log)
            self._n_model_particles = \
                [int(n) for n in np.exp(self._model_prob_log) * self.n_tot]

        # initialize particles and weights
        self.particles = np.zeros((*self.spatial_dims, self.n_tot,
                                   self.n_phys_vars +
                                   self.n_aug_vars))
        self.particles[..., self.state_indices['m']] = \
            np.repeat(self.model_range, self._n_model_particles)

        # axis where the particles are
        self.particle_ax = len(self.spatial_dims)

        self._log_weights = np.log(np.ones((*self.spatial_dims, self.n_tot)) /
                                   self.n_tot)

        # the index where to retrieve the statistics from -> possible since
        # currently weights are equal
        self.stats_ix = 0

        # just a variables to help name plots
        self.fig_counter = 0
        if name is None:
            self.name = ''
        else:
            self.name = name

    @property
    def weights(self):
        """The particle weights."""
        return np.exp(self.log_weights)

    @property
    def log_weights(self):
        """Logarithm of the particle weights."""
        return self._log_weights

    @log_weights.setter
    def log_weights(self, value):
        self._log_weights = value

    @property
    def model_prob(self):
        """Model probability in the linear domain."""
        return np.exp(self.model_prob_log)

    @property
    def model_prob_log(self):
        """Model probability in the logarithmic domain."""
        self._model_prob_log = self.M_t_all + np.log(
            [np.sum(np.exp(self.model_weights_log[i] - self.M_t_all[i])) for
             i in self.model_range])
        if np.isnan(self._model_prob_log).any():
            raise ValueError('Log model probability contains NaN.')
        return self._model_prob_log

    @model_prob_log.setter
    def model_prob_log(self, value):
        assert np.sum(np.exp(value)) == 1.
        self._model_prob_log = value

    @property
    def n_model_particles(self):
        """Number of particles assigned to each model."""
        model_indices = self.particles[self.stats_ix, :, self.state_indices[
            'm']]
        _, counts = np.unique(model_indices, return_counts=True)
        self._n_model_particles = counts
        return self._n_model_particles

    @n_model_particles.setter
    def n_model_particles(self, value):
        self._n_model_particles = value

    @property
    def model_indices_all(self):
        """Get particle indices of all models as a list."""
        return np.array([self.particles[self.stats_ix, :,
                         self.state_indices['m']] == j for j in
                         self.model_range])

    @property
    def model_weights(self):
        """Get particle weights per model."""
        return [self.weights[self.stats_ix, mix] for mix in
                self.model_indices_all]

    @property
    def model_weights_log(self):
        """Logarithmic weights per model as a list."""
        return [self.log_weights[self.stats_ix, mix] for mix in
                self.model_indices_all]

    @property
    def M_t_all(self):
        """Auxiliary quantity giving the maximum particle weight of a model."""
        self._M_t_all = np.array([np.max(i) for i in self.model_weights_log])
        return self._M_t_all

    @M_t_all.setter
    def M_t_all(self, value):
        self._M_t_all = value

    @property
    def M_t(self):
        """Auxiliary quantity giving the maximum weight of all weights."""
        return np.max(self.log_weights[self.stats_ix, :])

    @property
    def S_t_all(self):
        """Auxiliary quantity."""
        self._S_t_all = np.array([np.sum(np.exp(self.model_weights_log[i] -
                                                self.M_t_all[i])) for i in
                                  self.model_range])
        # this should not happen
        if np.isnan(self._S_t_all).any():
            raise ValueError('S_t_all contains NaN.')
        return self._S_t_all

    @S_t_all.setter
    def S_t_all(self, value):
        self._S_t_all = value

    @property
    def params_per_model(self):
        """Get the parameters distribution per model as a list."""
        # particle indices
        p_ix = [0] + list(np.cumsum(self.n_model_particles))
        param_dist = self.particles[self.stats_ix, :, self.n_phys_vars:]
        params_per_model = [param_dist[p_ix[i]:p_ix[i+1]] for i in range(
            len(p_ix)-1)]
        return params_per_model

    @property
    def param_mean(self):
        """Get state mean of parameters."""
        mix = self.model_indices_all
        p_m_nan = [np.average(self.particles[self.stats_ix, mix[i]][:,
                              self.state_indices['theta']],
                              weights=np.exp(self.log_weights[self.stats_ix,
                                                              mix[i]] -
                                             self.model_prob_log[i]),
                              axis=self.particle_ax - 1) for i in
                   self.model_range]
        p_m = [p[~np.isnan(p)] for p in p_m_nan]
        return p_m

    @property
    def param_covariance(self):
        """Get the state covariance of parameters."""
        mix = self.model_indices_all
        tix = self.state_indices['theta']
        big_covlist = []
        for j in range(self.n_models):

            model_ptcls_nan = self.particles[self.stats_ix, mix[j]][:, tix]
            model_ptcls = model_ptcls_nan[:, ~np.isnan(model_ptcls_nan).all(
                                                    axis=0)]

            model_wgts = np.exp(
                self.log_weights[self.stats_ix, mix[j]] -
                self.model_prob_log[j]
            )
            big_covlist.append(np.cov(model_ptcls.T, aweights=model_wgts))

        return np.array(big_covlist)

    @property
    def effective_n_per_model(self):
        """Effective number of particles per model."""
        n_eff = [effective_n(w/np.sum(w)) for w in self.model_weights]
        return n_eff

    @classmethod
    def from_file(cls, path: str):
        """
        Generate an Augmented Ensemble Particle Filter object from netCDF file.

        Parameters
        ----------
        path: str
            Path to the netCDF file from which the class shall be instantiated.

        Returns
        -------
        pf: AugmentedEnsembleParticleFilter
            An AugmentedEnsembleParticleFilter object.
        """
        ds = xr.open_dataset(path)
        return cls.from_dataset(ds)

    @classmethod
    def from_dataset(cls, xr_ds: xr.Dataset,
                     date: Optional[pd.Timestamp] = None):
        """
        Generate a Particle Filter object from an xarray Dataset.

        Parameters
        ----------
        xr_ds: xarray.Dataset
            An xarray Dataset from which the class shall be instantiated.
        date : pd.Timestamp, optional
            Date information that shall be retrieved form the dataset. For
            this, the dataset must contain a time coordinate. If not date is
            given, the latest date from the dataset is chosen. Default: None
            (choose latest).

        Returns
        -------
        pf: AugmentedEnsembleParticleFilter
            An AugmentedEnsembleParticleFilter object.
        """

        if 'model' in xr_ds.coords:
            model_names = xr_ds.model.values
        elif 'models' in xr_ds.attrs:
            model_names = xr_ds.attrs['models']
        else:
            raise ValueError('Models must be given either in datasets coords '
                             'or attributes.')

        if date is not None:
            xr_ds = xr_ds.sel(time=date)
        else:
            xr_ds = xr_ds.isel(time=-1)

        mb_models = [eval(m) for m in model_names]
        n_particles = len(xr_ds.particle_id.values)
        spatial_dims = (len(xr_ds.fl_id.values),)
        n_phys_vars = xr_ds.attrs['n_phys_vars']
        n_aug_vars = xr_ds.attrs['n_aug_vars']
        model_prob_log = xr_ds.attrs['model_prob_log']
        aepf = AugmentedEnsembleParticleFilter(
            models=mb_models, n_particles=n_particles,
            spatial_dims=spatial_dims, n_phys_vars=n_phys_vars,
            n_aug_vars=n_aug_vars, name=gdir.plot_name,
            model_prob_log=model_prob_log)

        # init particles
        aepf.particles = xr_ds.particles.values
        aepf.log_weights = np.repeat(xr_ds.log_weights.values[None, :],
                                     len(xr_ds.fl_id.values), axis=0)
        return aepf

    def to_file(self, path: str) -> None:
        """
        Write to netCDF file.

        Parameters
        ----------
        path: str
            Path to write the object to.

        Returns
        -------
        None
        """
        ds = self.to_dataset()
        # netCDF cannot handle multidim. array attributes (list(dict.items()))
        del ds.attrs['state_indices']

        ds.encoding['zlib'] = True
        # todo: more enc./drop height dependency of weights to save disk space?
        print(ds.encoding)
        ds.to_netcdf(path)

    def to_dataset(self, date: pd.Timestamp) -> xr.Dataset:
        """
        Export as xarray Dataset.

        Parameters
        ----------
        date: pd.Timestamp



        Returns
        -------
        ds: xarray.Dataset
            A Dataset containing the class attributes as variables.
        """

        xr_ds = xr.Dataset(
            {'particles': (['fl_id', 'particle_id', 'data', 'time'],
                           self.particles[..., None]),
             'log_weights': (['particle_id', 'time'],
                             self.log_weights[self.stats_ix][..., None])},
            coords={'particle_id': (['particle_id', ],
                                    range(self.n_particles)),
                    'fl_id': (range(self.particles.shape[0])),
                    'data': (range(self.particles.shape[-1])),
                    'time': [date]},
            attrs={'models': self.model_names, 'n_phys_vars': self.n_phys_vars,
                   'n_aug_vars': self.n_aug_vars,
                   'state_indices': self.state_indices,
                   'model_prob_log': self.model_prob_log})
        xr_ds.data.encoding = {'dtype': 'int16', 'scale_factor': 0.00001,
                               '_FillValue': -9999., 'zlib': True}
        return xr_ds

    def get_initial_conditions(self):
        """
        The function to produce the initial conditions and to set the value
        for x0 should go here.

        If only __init__ is called, only a very basic info should be set to x0.
        If this method is called additionally, the state shall be filled with
        real initial conditions from modeling.

        Returns
        -------
        None
        """

    def get_observation_quantiles_hansruedi(self, obs, obs_std, mb_ix,
                                            mb_init_ix, eval_ix,
                                            obs_first_dim=0,
                                            generate_n_obs=1000,
                                            by_model=False):
        """
        Get the quantile of the observation at the weighted particle
        distribution

        Parameters
        ----------
        obs : np.array
            Array with observations. First dim is variables, second dim is
            space.
        obs_std: np.array
            Array with standard deviation of the observations.
        mb_ix : int
            Index of the mass balance entry in the particles array.
            # todo: this is hardcoded
        mb_init_ix : int
            Index of the initial mass balance (at camera setup) entry in the
            particles array.
            # todo: this is hardcoded
        eval_ix: list of int
            Indices where the evaluation shall be made.
        obs_first_dim : int
            This is a bullshit keyword. It is just temporary and tells that, at
            the moment, we are just able to look at the mass balance.
        generate_n_obs: int, optional
            How many observations to generate.
        by_model: bool


        Returns
        -------
        obs_quantiles: np.array
            Array with quantiles of the observation at the weighted particle
            distribution.
        """

        #  todo: the zeros are hard-coded (the keyword is stupid, but make it
        #  easier afterwards
        obs_quantiles = []  # np.full_like(obs, np.nan)
        quantile_range = np.arange(0.0, 1.01, 0.01)
        for i, eix in enumerate(eval_ix):
            if ~np.isnan(obs[obs_first_dim, i]):
                # subtract the initial MB
                actual_mb = self.particles[eix, :, mb_ix] - \
                            self.particles[eix, :, mb_init_ix]

                if by_model is True:
                    raise NotImplementedError
                    model_indices = self.model_indices_all
                    intermediate_list = []
                    for mi in model_indices:
                        w_quantiles = utils.weighted_quantiles(
                            actual_mb[mi],
                            quantile_range,
                            sample_weight=self.weights[
                                eix,
                                mi])
                        intermediate_list.append(
                            np.argmin(np.abs(w_quantiles -
                                             np.atleast_2d(
                                                 np.random.normal(
                                                     obs[obs_first_dim, i],
                                                     obs_std[
                                                         obs_first_dim, i],
                                                     generate_n_obs)).T),
                                      axis=1))
                    obs_quantiles.append(intermediate_list)
                else:
                    w_quantiles = np.sum(
                        self.weights[eix, :] * (
                                obs[obs_first_dim, i] - actual_mb /
                                obs_std[obs_first_dim, i])
                    )
                    obs_quantiles.append(w_quantiles)

        return obs_quantiles

    def get_observation_quantiles(
            self, obs, obs_std, mb_ix, mb_init_ix, eval_ix, obs_first_dim=0,
            generate_n_obs=1000, by_model=False):
        """
        Get the quantile of the observation at the weighted particle
        distribution.

        Parameters
        ----------
        obs : np.array
            Array with observations. First dim is variables, second dim is
            space.
        obs_std: np.array
            Array with standard deviation of the observations.
        mb_ix : int
            Index of the mass balance entry in the particles array.
            # todo: this is hardcoded
        mb_init_ix : int
            Index of the initial mass balance (at camera setup) entry in the
            particles array.
            # todo: this is hardcoded
        eval_ix: list of int
            Indices where the evaluation shall be made.
        obs_first_dim : int
            This is a bullshit keyword. It is just temporary and tells that, at
            the moment, we are just able to look at the mass balance.
        generate_n_obs: int, optional
            How many observations to generate.
        by_model: bool


        Returns
        -------
        obs_quantiles: np.array
            Array with quantiles of the observation at the weighted particle
            distribution.
        """

        # todo: the zeros are hard-coded (the keyword is stupid, but make it
        #  easier afterwards
        obs_quantiles = []
        quantile_range = np.arange(0.0, 1.01, 0.01)
        for i, eix in enumerate(eval_ix):
            if ~np.isnan(obs[obs_first_dim, i]):
                # subtract the initial MB
                actual_mb = self.particles[eix, :, mb_ix] - \
                            self.particles[eix, :, mb_init_ix]

                if by_model is True:
                    model_indices = self.model_indices_all
                    intermediate_list = []
                    for mi in model_indices:
                        w_quantiles = utils.weighted_quantiles(
                                np.random.normal(
                                    obs[obs_first_dim, i],
                                    obs_std[obs_first_dim, i],
                                    generate_n_obs
                                ), quantile_range
                        )
                        intermediate_list.append(
                            np.argmin(np.abs(np.atleast_2d(w_quantiles).T -
                                             actual_mb[mi]), axis=0))
                    obs_quantiles.append(intermediate_list)
                else:
                    w_quantiles = utils.weighted_quantiles(
                        np.random.normal(obs[obs_first_dim, i],
                                         obs_std[obs_first_dim, i],
                                         generate_n_obs), quantile_range
                    )
                    obs_quantiles.append(
                        np.argmin(np.abs(
                            np.atleast_2d(w_quantiles).T - actual_mb), axis=0))

        return obs_quantiles

    def set_conditions(self, space_index, phys_vars=None, aug_vars=None,
                       log_weights=None):
        """
        (Re)set the conditions at some space.

        Parameters
        ----------
        space_index : np.array of int
            The space indices where conditions shall be reset.
        phys_vars : np.array or None
            Values for the physical state variables. Default: None (do not
            update).
        aug_vars : np.array or None
            Values for the augmented state variables. Default: None (do not
            update).
        log_weights : np.array or None
            Values for the logarithmic weights, if they should be updated as
            well (Dangerous!?). Default: None (do not update weights).

        Returns
        -------
        None.
        """

        # todo: is this "roundtrip" when assigning new particles necessary?
        ptcls = self.particles.copy()
        if phys_vars is not None:
            ptcls[space_index, :, self.state_indices['xi']] = phys_vars
        if aug_vars is not None:
            ptcls[space_index, :, self.state_indices['theta']] = aug_vars
        self.particles = ptcls

        if log_weights is not None:
            lwgts = self.log_weights.copy()
            lwgts[:] = log_weights
            self.log_weights = lwgts

    def model_indices_j(self, j):
        """
        Get particle indices of the models with index `j`.

        Parameters
        ----------
        j: int
            Model number (starting at zero!).

        Returns
        -------
        tuple:
            Indices where particles with the respective model index occur.
        """
        return np.where(self.particles[self.stats_ix, :,
                        self.state_indices['m']] == j)

    def predict(self, mb_models_inst, gmeteo, date, h, ssf,
                ipot, ipot_sigma, alpha_ix, mod_ix, swe_ix, tacc_ix, mb_ix,
                tacc_ice, model_error_mean, model_error_std, obs_merge=None,
                param_random_walk=False, snowredistfac=None,
                use_psol_multiplier=False, alpha_underlying=None, seed=None):
        """
        Calculate a mass balance prediction.

        At the moment, the SWE and albedo are updated with the accumulation of
        a day first, before the ablation is calculated. Since temperature is
        from 00-00 am and precipitation is from 06-06 am though, it might even
        make sense to calculate ablation first and then update SWE and albedo
        for the next day.
        # todo: consider calculating ablation first, then update SWE/alpha

        Parameters
        ----------
        mb_models_inst : list
            List of instantiated `py:class:crampon.core.models.massbalance.
            DailyMassBalanceModelWithSnow`.
        gmeteo : `crampon.core.preprocessing.climate.GlacierMeteo`
            Glacier meteorology class containing the necessary meteo input.
        date : pd.Timestamp
            Date to calculate the mass balance for.
        h : np.array
            Array with heights of the glacier flowline.
        ssf : np.array
            SIS scaling factor for the day at heights.
        ipot : np.array
            Potential irradiation for the day at height (W m-2).
        ipot_sigma : float
            Potential irradiation uncertainty as standard deviation (W m-2).
        alpha_ix : int
            Index at which albedo is stored in the particle array.
        mod_ix : int
            Index at which model index is stored in the particle array.
        swe_ix : int
            Index at which snow water equivalent is stored in the particle
            array.
        tacc_ix : int
            Index at which accumulated temperature since last snowfall is
            stored in the particle array.
        mb_ix : int
            Index at which cumulative mass balance is stored in the particle
            array.
        tacc_ice : float
            Default value for the sum of positive temperatures that an ice
            surface has experienced (determines the albedo of ice).
        model_error_mean : float
            Additive model error mean.
        model_error_std :
            Additive model error standard deviation.
        obs_merge : xr.Dataset, optional
            Dataset with the merged observations (m w.e.). Default: None.
        param_random_walk : bool, optional
            Whether to let the parameters do a random walk. Default: False.
        snowredistfac : bool, optional
            Which snow redistribution factor to use. Default: None.
        use_psol_multiplier : bool, optional
            Whether to use the periodic multiplier for solid precipitation.
            Default: False.
        alpha_underlying: np.array
            Underlying albedo beneath the top.
        seed : int, optional
            Seed to use to make experiments reproducible. Default: None (do not
            use seed).

        Returns
        -------
        None
        """

        doy = date.dayofyear

        # specify some albedo noise already (faster)
        # todo:here we should probably use the median elevation
        #  (weights by widths) instead of the mean
        alpha_grad_noise = list(np.atleast_2d(stats.truncnorm.rvs(
            0, np.inf, size=self.n_tot) * 0.0001).T *
                                np.atleast_2d((h - np.mean(h))))
        alpha_loc_noise = list(np.random.normal(0, 0.15, self.n_tot))
        # todo: make the noise anticorrelated!
        swe_grad_noise = list(np.atleast_2d(stats.truncnorm.rvs(
            0, np.inf, size=self.n_tot) * 0.0001).T * np.atleast_2d(
            (h - np.mean(h))))
        swe_loc_noise = list(np.random.normal(0, 0.05, self.n_tot))

        # get the prediction from the MB models
        params_per_model = self.params_per_model.copy()

        # predict model variables
        mb_per_model = []
        alpha_per_model = []
        swe_per_model = []
        tacc_per_model = []

        # todo: change std to weighted std
        print('N model particles:', self.n_model_particles)
        try:
            print('MU_ICE: ', np.average(params_per_model[0][:, 0],
                                         weights=self.model_weights[0]),
                  np.std(params_per_model[0][:, 0]))
        except ZeroDivisionError:
            pass

        for i, m in enumerate(mb_models_inst):
            # todo: select meteo actually doesn't have to be done per model
            # METEO UNCERTAINTY from the standard devs that we have assigned
            gmeteo.n_random_samples = self.n_model_particles[i]
            # prcp, tmean, tmax, sis
            if seed is None:
                seed = np.random.randint(0, 1000, 1)[0]
            temp_rand = gmeteo.get_tmean_at_heights(date, h, random_seed=seed)
            tmax_rand = gmeteo.get_tmax_at_heights(date, h, random_seed=seed)
            psol_rand, _ = gmeteo.get_precipitation_solid_liquid(
                date, h, tmean=temp_rand, random_seed=seed)

            # the 'RAIN' indicator means for sure no accumulation.
            if obs_merge is not None:
                try:
                    remark = obs_merge.sel(date=date).key_remarks.values
                    psol_rand[(~pd.isnull(remark)) & (
                                ('RAIN' in remark) &
                                ('SNOW' not in remark))] = 0.
                # todo: IndexError when more elevation bands than observations
                except (IndexError, KeyError):  # day excluded by keyword
                    pass

            # correct precipitation for systematic, not calibrated biases
            if use_psol_multiplier is True:
                psol_rand *= prcp_fac_cycle_multiplier[doy - 1]
            if snowredistfac is not None:
                psol_rand *= np.atleast_2d(snowredistfac[:, i]).T

            sis_rand = np.atleast_2d(gmeteo.randomize_variable(
                date, 'sis', random_seed=seed))
            # necessary because of the dimensions
            sis_rand = sis_rand * ssf

            # todo: is perfect correlation a good assumption?
            ipot_reshape = np.repeat(np.atleast_2d(ipot).T,
                                     self.n_model_particles[i], axis=1)
            np.random.seed(seed)
            ipot_rand = ipot_reshape + np.atleast_2d(np.random.normal(
                0., ipot_sigma, self.n_model_particles[i]))

            # PARAM UNCERTAINTY from current filter
            model_params = params_per_model[i]

            # get the alphas from a model
            alpha = np.array([self.particles[k, :, alpha_ix][self.particles[
                                                             k, :,
                                                             mod_ix] == i]
                              for k in range(self.spatial_dims[0])])
            swe = np.array([self.particles[k, :, swe_ix][self.particles[k, :,
                                                         mod_ix] == i] for k in
                            range(self.spatial_dims[0])])
            tacc = np.array([self.particles[k, :, tacc_ix][self.particles[k,
                                                           :, mod_ix] == i] for
                             k in
                             range(self.spatial_dims[0])])

            mb_daily = []
            # todo: vectorize MB calculation per model!!!
            for pi in range(model_params.shape[0]):
                # compile a dictionary with the current values (makes it safer)
                if param_random_walk is True:
                    np.random.seed(seed)
                    random_pi = np.random.choice(range(model_params.shape[0]))
                    param_dict = dict(zip(m.cali_params_list,
                                          model_params[random_pi, :len(
                                              m.cali_params_list)]))
                else:
                    param_dict = dict(
                        zip(m.cali_params_list,
                            model_params[pi, :len(m.cali_params_list)])
                    )

                param_dict.update({'psol': psol_rand[:, pi],
                                   'tmean': temp_rand[:, pi],
                                   'tmax': tmax_rand[:, pi],
                                   'sis': sis_rand[:, pi]})

                # todo: this below here might need a rehaul as a whole

                # todo: first update SWE here?????? we also update alpha at
                #  first and then calculate the MB (ablation)
                before_snow_ix = swe[:, pi] > 0.

                # numerically
                if (swe[:, pi] < 0.).any() or np.isnan(swe[:, pi]).any():
                    print('Swe smaller than zero or NaN in predict')
                    swe[:, pi][swe[:, pi] < 0.] = 0.

                # update tacc/albedo depending on SWE
                tacc[param_dict['psol'] >= 1., pi] = 0.
                old_snow_ix = ((param_dict['psol'] < 1.) & (swe[:, pi] > 0.))
                tacc[old_snow_ix, pi] += np.clip(
                    param_dict['tmax'][old_snow_ix], 0., None)
                no_snow_ix = ((param_dict['psol'] < 1.) & (swe[:, pi] == 0.))
                # setting a chosen fixed ice albedo tacc
                # if no_snow_ix.any():
                #    tacc[no_snow_ix, pi] = tacc_ice
                if (no_snow_ix & before_snow_ix).any():
                    tacc_ice = tacc_from_alpha_brock(
                        alpha_underlying[no_snow_ix & before_snow_ix])
                    tacc[no_snow_ix & before_snow_ix, pi] = \
                        np.sort(np.random.normal(
                            tacc_ice, 1000., tacc[no_snow_ix & before_snow_ix,
                                                  pi].shape))[::-1]
                    # tacc[no_snow_ix & before_snow_ix, pi] =
                    # alpha_underlying[no_snow_ix & before_snow_ix, pi]

                # todo: the underlying albedo should depend on the SWE
                # todo: a_u=None is a test to make alpha varying
                alpha[:, pi] = point_albedo_brock(
                    swe[:, pi], tacc[:, pi], swe[:, pi] == 0.,
                    a_u=alpha_underlying)
                # make Gaussian Noise
                # todo: how to add noise such that albedo is not greater one?
                alpha[:, pi] += alpha_loc_noise.pop()  # add noise by shifting
                alpha[:, pi] += alpha_grad_noise.pop()  # rotating the profile
                alpha[:, pi] = np.clip(alpha[:, pi], 0.01, 0.99)

                swe[:, pi] += swe_loc_noise.pop()  # add noise by shifting
                swe[:, pi] += swe_grad_noise.pop()  # rotating the profile
                swe[:, pi] = np.clip(swe[:, pi], 0., None)

                if (swe[:, pi] < 0.).any() or np.isnan(swe[:, pi]).any():
                    print('Swe smaller than zero or NaN in predict')
                    print(swe[:, pi][swe[:, pi] < 0.])
                if (psol_rand[:, pi] < 0.).any():
                    print('Psol smaller than zero in predict')
                    print(psol_rand[:, pi][psol_rand[:, pi] < 0.])

                # mirror back the changes in alpha to tacc
                # todo: check if this is a good idea - the relation is
                #  one-directional
                tacc[:, pi] = tacc_from_alpha_brock(alpha[:, pi])

                # the model decides which MB to produce
                if m.__name__ == 'BraithwaiteModel':
                    melt = melt_braithwaite(**param_dict, swe=swe[:, pi])
                elif m.__name__ == 'HockModel':
                    melt = melt_hock(**param_dict, ipot=ipot_rand[:, pi],
                                     swe=swe[:, pi])
                elif m.__name__ == 'PellicciottiModel':
                    melt = melt_pellicciotti(**param_dict, alpha=alpha[:, pi])
                elif m.__name__ == 'OerlemansModel':
                    melt = melt_oerlemans(**param_dict, alpha=alpha[:, pi])
                else:
                    raise ValueError('Mass balance model not implemented for'
                                     ' particle filter.')

                # m w.e. = mm / 1000. - m w.e.
                mb = param_dict['psol'] / 1000. - melt

                mb_daily.append(mb)
                swe[:, pi] += mb

            # important: clip SWE (we do it only once per model to save time)
            swe = np.clip(swe, 0., None)

            # add some additional model error (if defined) and append to list
            np.random.seed(seed)
            mb_per_model.append(np.array(mb_daily) +
                                np.atleast_2d(
                                    np.random.normal(model_error_mean,
                                                     model_error_std,
                                                     len(mb_daily))).T)
            alpha_per_model.append(alpha)
            swe_per_model.append(swe)
            tacc_per_model.append(tacc)

        # assign all values to filter (would probably be better to merge all
        # first and then insert)
        particles = self.particles.copy()
        # 'F' to insert in the correct order
        for im in range(len(mb_models_inst)):
            # replace swe
            particles[:, :, swe_ix][particles[:, :, mod_ix] == im]\
                = \
                swe_per_model[im].flatten()
            # replace alpha
            particles[:, :, alpha_ix][particles[:, :, mod_ix] == im] = \
                alpha_per_model[im].flatten()
            # *ADD* MB (+=) # MB needs Fortran flattening here!!!
            particles[:, :, mb_ix][particles[:, :, mod_ix] == im] \
                += \
                mb_per_model[im].flatten('F')
            # T_acc
            particles[:, :, tacc_ix][particles[:, :, mod_ix] == im] \
                = tacc_per_model[im].flatten()

        self.particles = particles.copy()

    def update(self, obs, obs_std, R, obs_ix, obs_init_mb_ix, obs_spatial_ix,
               sla_ix=None, swe_ix=None, date=None):
        """
        Update the prior estimate with observations.

        Parameters
        ----------
        obs :
        obs_std :
        R :
        obs_ix :
        obs_init_mb_ix :
        obs_spatial_ix :

        Returns
        -------

        """

        Rinv = spla.pinv(R)

        obs_indices = np.where(~np.isnan(obs))

        # todo: implement covariance solution with (a) more than one obs
        #  variable and (b) more than one obs locations
        # todo: not logic a this position, but weights over spatial domain
        #  should be all the same -> replace with self.stats_ix?
        w = copy.deepcopy(self.log_weights[obs_indices[-1][0], :])

        # just a copy for the plot that is not updated
        w_predict = copy.deepcopy(self.log_weights[obs_indices[-1][0], :])
        w_predict = np.exp(w_predict)
        for obs_count, obs_loc in enumerate(obs_indices[-1]):
            obs_s_ix = obs_spatial_ix[obs_loc]
            mb_particles = self.particles[obs_s_ix, :, obs_ix]
            mb_particles_init = self.particles[obs_s_ix, :, obs_init_mb_ix]

            h_status = (mb_particles - mb_particles_init) / 0.9
            h_obs = obs[0, obs_loc] / 0.9
            h_obs_std = obs_std[0, obs_loc] / 0.9

            """
            # likelihood figure as promised in the author response
            colors = ["b", "g", "c", "m"]
            violinstats = ['cmedians', 'cmins', 'cmaxes', 'cbars'] 
                # 'cquantiles',
            fig, ax = plt.subplots()
            mptcls = [0] + list(np.cumsum(self.n_model_particles))
            mprobs = self.model_prob
            jitter_std = 0.04
            seg_ext = 0.2  # extend violin segments (MED etc.) to exceed jitter
            scat_alpha = 0.2

            # Braithwaite
            bptcls = h_status[mptcls[0]:mptcls[1]] * 0.9
            ax.scatter(bptcls, np.random.normal(2, jitter_std, len(bptcls)),
                       c=colors[0], s=0.5, zorder=0, alpha=scat_alpha)
            violin_parts = ax.violinplot(bptcls, vert=False, positions=[2],
                                         showmedians=True, showextrema=True)#,
                                         #quantiles=[0.1, 0.9])
            for pc in violin_parts['bodies']:
                pc.set_color(colors[0])
                pc.set_edgecolor(colors[0])
            for v in violinstats:
                violin_parts[v].set_color(colors[0])
                violin_parts[v].set_edgecolor(colors[0])
                if v != 'cbars':
                    violin_parts[v].set(segments=[
                        violin_parts[v].get_segments()[0] + np.array(
                            [[0., -seg_ext], [0., +seg_ext]])])


            # Hock
            hptcls = h_status[mptcls[1]:mptcls[2]] * 0.9
            violin_parts = ax.violinplot(hptcls, vert=False, positions=[3],
                                         showmedians=True, showextrema=True)#,
                                         #quantiles=[0.1, 0.9])
            for pc in violin_parts['bodies']:
                pc.set_color(colors[1])
                pc.set_edgecolor(colors[1])
            for v in violinstats:
                violin_parts[v].set_color(colors[1])
                violin_parts[v].set_edgecolor(colors[1])
                if v != 'cbars':
                    violin_parts[v].set(segments=[
                        violin_parts[v].get_segments()[0] + np.array(
                            [[0., -seg_ext], [0., +seg_ext]])])
            ax.scatter(hptcls, np.random.normal(3, jitter_std, len(hptcls)),
                       c=colors[1], s=0.5, zorder=0, alpha=scat_alpha)

            # Pelliciotti
            pptcls = h_status[mptcls[2]:mptcls[3]] * 0.9
            violin_parts = ax.violinplot(pptcls, vert=False, positions=[4],
                                         showmedians=True, showextrema=True)#,
                                         #quantiles=[0.1, 0.9])
            for pc in violin_parts['bodies']:
                pc.set_color(colors[2])
                pc.set_edgecolor(colors[2])
            for v in violinstats:
                violin_parts[v].set_color(colors[2])
                violin_parts[v].set_edgecolor(colors[2])
                if v != 'cbars':
                    violin_parts[v].set(segments=[
                        violin_parts[v].get_segments()[0] + np.array(
                            [[0., -seg_ext], [0., +seg_ext]])])
            ax.scatter(pptcls, np.random.normal(4, jitter_std, len(pptcls)),
                       c=colors[2], s=0.5, zorder=0, alpha=scat_alpha)


            # Oerlemans
            optcls = h_status[mptcls[3]:mptcls[4]] * 0.9
            violin_parts = ax.violinplot(optcls, vert=False, positions=[5],
                                         showmedians=True, showextrema=True)#,
                                         #quantiles=[0.1, 0.9])
            for pc in violin_parts['bodies']:
                pc.set_color(colors[3])
                pc.set_edgecolor(colors[3])
            for v in violinstats:
                violin_parts[v].set_color(colors[3])
                violin_parts[v].set_edgecolor(colors[3])
                if v != 'cbars':
                    violin_parts[v].set(segments=[
                        violin_parts[v].get_segments()[0] + np.array(
                            [[0., -seg_ext], [0., +seg_ext]])])
            ax.scatter(optcls, np.random.normal(5, jitter_std, len(optcls)),
                       c=colors[3], s=0.5, zorder=0, alpha=scat_alpha)

            # Ensemble
            resamp_indices = stratified_resample(w_predict, self.n_tot,
                                one_random_number=True)
            ens_ptcls, _ = resample_from_index(h_status*0.9, w_predict,
                                               resamp_indices)
            violin_parts = ax.violinplot(ens_ptcls, vert=False, positions=[1],
                                         showmedians=True, showextrema=True)#,
                                         #quantiles=[0.1, 0.9])
            for pc in violin_parts['bodies']:
                pc.set_color('darkgoldenrod')
                pc.set_edgecolor('darkgoldenrod')
            for v in violinstats:
                violin_parts[v].set_color('darkgoldenrod')
                violin_parts[v].set_edgecolor('darkgoldenrod')
                if v != 'cbars':
                    seg = violin_parts[v].get_segments()[0]
                    violin_parts[v].set(segments=[
                        violin_parts[v].get_segments()[0] + np.array(
                            [[0., -seg_ext], [0., +seg_ext]])])
            ax.scatter(ens_ptcls, np.random.normal(
                1, jitter_std, len(ens_ptcls)), c='darkgoldenrod', s=0.5,
                       zorder=0, alpha=scat_alpha)

            # observation
            obsptcls = np.random.normal(h_obs * 0.9, h_obs_std * 0.9, 10000)
            violin_parts = ax.violinplot(obsptcls,
                vert=False, positions=[0], showmedians=True, showextrema=True)
                #quantiles=[0.1, 0.9])
            for pc in violin_parts['bodies']:
                pc.set_color('k')
                pc.set_edgecolor('k')
            for v in violinstats:
                violin_parts[v].set_color('k')
                violin_parts[v].set_edgecolor('k')
                if v != 'cbars':
                    seg = violin_parts[v].get_segments()[0]
                    violin_parts[v].set(segments=[
                        violin_parts[v].get_segments()[0] + np.array(
                            [[0., -seg_ext], [0., +seg_ext]])])
            ax.scatter(obsptcls, np.random.normal(
                0, jitter_std, len(obsptcls)), c='k', s=0.5, zorder=0,
                       alpha=scat_alpha)

            likeli = np.exp(
                -((h_obs - h_status) ** 2.) / (2. * h_obs_std ** 2))
            ax.scatter(h_status * 0.9, likeli - 1.5)

            #for i, (mp, ml) in enumerate(zip([h_status[mptcls[0]:mptcls[1]],
            #                                  h_status[mptcls[1]:mptcls[2]],
            #                                  h_status[mptcls[2]:mptcls[3]],
            #                                  h_status[mptcls[3]:mptcls[4]]],
            #                                 [likeli[mptcls[0]:mptcls[1]],
            #                                  likeli[mptcls[1]:mptcls[2]],
            #                                  likeli[mptcls[2]:mptcls[3]],
            #                                  likeli[mptcls[3]:mptcls[4]]])):
            #    ax.scatter(mp * 0.9, ml + i + 1.5, c=colors[i])

            labels = ['Likelihood', 'Observation',
                      'Ensemble', 'Braithwaite', 'Hock', 'Pellicciotti',
                      'Oerlemans']
            ax.yaxis.set_tick_params(direction='out')
            ax.yaxis.set_ticks_position('left')
            ax.set_yticks([-1.5] + list(np.arange(len(labels) - 1)))
            ax.set_yticklabels(labels)
            ax.set_xlabel('$b_{sfc}(t,z)$')
            ax.set_ylabel('$b_{sfc}(t,z)$')
            ax.axhline(1.5)
            ax.axhline(-0.35)
            ax.set_title('{}'.format(date.strftime('%Y-%m-%d')))
            plt.tight_layout()
            plt.savefig(
            'C:\\Users\\Johannes\\Desktop\\temp\\likelihood\\likelihood_fig
            _{}_{}_{}.png'.format(self.name, date.strftime('%Y-%m-%d'), 
            obs_count), dpi=300)
            plt.legend(loc=0)
            plt.close()
            """

            # procedure to account for the fact that the observations are
            # lognormal-distributed? only relevant on first day probably...
            # plt.figure()
            # np.random.normal(h_obs, h_obs_std, 10000)
            # log_mean = np.nanmean(np.log(- obsptcls))  # okay to clip NaNs?
            # log_std = np.nanstd(np.log(- obsptcls))  # okay to clip the NaNs?
            # log_dist = np.random.normal(log_mean, log_std, 10000)
            # plt.scatter(-np.exp(log_dist),
            #            np.random.normal(0, 0.04, len(log_dist)))
            # plt.scatter(obsptcls, np.random.normal(0, 0.04, len(obsptcls)))
            # plt.figure()
            # plt.scatter(h_status, np.exp(
            #     -((log_mean - np.log(-h_status)) ** 2.) /
            #     (2. * log_std ** 2)))

            w += -((h_obs - h_status) ** 2.) / (2. * h_obs_std ** 2)
            # w += -((log_mean - np.log(-h_status)) ** 2.) /
            # (2. * log_std ** 2)
        # if sla_ix is not None:
        #    set_to_zero = np.where(self.particles[sla_ix:, :, swe_ix] > 0.)

        w -= np.max(w)

        new_wgts = w - np.log(np.sum(np.exp(w)))  # normalize

        self.log_weights = np.repeat(new_wgts[np.newaxis, ...],
                                     self.log_weights.shape[0],
                                     axis=0)

        self.fig_counter += 1

    def update_with_alpha_obs(self, obs, alpha_ix, fudge_fac=0.01, date=''):
        """
        Update step with observations of albedo.

        Parameters
        ----------
        obs: np.ndarray
            Array with dimensions (M, N), where M is the spatial dimension
            (elevations), and N is the observations of alpha in an elevation
            bin. For an elevation to be an ivalid observations (too many
            missing values), the whole slice [m, :] should be NaN.
        alpha_ix: int
            Position index of the albedo in the `particles` array.
        nan_thresh: float
            Maximum ratio of allowed NaNs in an elevation band. Default: 0.2.
        fudge_fac: float
            Factor to avoid numerical issues when transforming to logit space.
            Recommended and default: 0.01.

        Returns
        -------
        None
        """

        fig, [ax1, ax2] = plt.subplots(2, sharex=True)
        avg = np.average(self.particles[:, :, alpha_ix],
                         weights=self.weights[0, :], axis=1)
        ax1.errorbar(np.arange(self.particles.shape[0]), avg, yerr=np.sqrt(
            np.average(
                (self.particles[:, :, alpha_ix] - np.atleast_2d(avg).T) ** 2,
                weights=self.weights[0, :], axis=1)), label='MOD')
        ax1.errorbar(np.arange(self.particles.shape[0]),
                     np.nanmean(obs, axis=1), yerr=np.nanstd(obs, axis=1),
                     label='OBS')
        ax1.legend()
        lc = ['g', 'b', 'k']
        c = ["g", "b", 'k']
        xs = np.arange(self.particles.shape[0])
        ys_o = np.nanmean(obs, axis=1)
        ys_m = avg
        std_m = np.sqrt(np.average(
            (self.particles[:, :, alpha_ix] - np.atleast_2d(avg).T) ** 2,
            weights=self.weights[0, :], axis=1))
        std_m_top = ys_m + std_m
        std_m_btm = ys_m - std_m
        std_o = np.nanstd(obs, axis=1)
        std_o_top = ys_o + std_o
        std_o_btm = ys_o - std_o
        ax2.plot(xs, ys_m, linestyle='-', color=lc[0], lw=2, zorder=100)
        ax2.fill_between(xs, std_m_btm, std_m_top, facecolor=c[0], alpha=0.3,
                         zorder=100)
        ax2.plot(xs, ys_o, linestyle='-', color=lc[1], lw=2)
        ax2.fill_between(xs, std_o_btm, std_o_top, facecolor=c[1], alpha=0.3)

        # the update should be made in logit space
        obs_logit = utils.physical_to_logit(obs, 0., 1., fudge_fac)
        model_logit = utils.physical_to_logit(
            self.particles[:, :, alpha_ix], 0., 1., fudge_fac)

        # do not waste calculation time
        obs_indices = np.where(~np.isnan(obs).all(axis=1))
        w = copy.deepcopy(self.log_weights[0, :])

        for obs_loc in obs_indices[0]:
            model_logit_slice = model_logit[obs_loc, :]
            obs_logit_slice = obs_logit[obs_loc, :]
            obs_logit_slice_mean = np.nanmean(obs_logit_slice)
            obs_logit_slice_std = np.nanstd(obs_logit_slice)

            w += -((obs_logit_slice_mean - model_logit_slice) ** 2.) / \
                 (2. * obs_logit_slice_std ** 2)
        # try to vectorize logit
        # todo: not sure if taking the mean is allowed here, but it avoids
        #  numerical issues -> check if it's allowed
        # w += np.mean(-((np.nanmean(obs_logit[obs_indices[0]], axis=1)[:, np.newaxis] - model_logit[obs_indices[0], :]) ** 2.) / \
        #         (2. * np.nanstd(obs_logit[obs_indices[0]], axis=1)[:, np.newaxis] ** 2), axis=0)
        # try to vectorize physical space (Gaussian)
        # w += np.mean(-((np.nanmean(obs[obs_indices[0]], axis=1)[:,
        #               np.newaxis] - self.particles[:, :, alpha_ix][obs_indices[0],
        #                             :]) ** 2.) / \
        #          (2. * np.nanstd(obs[obs_indices[0]], axis=1)[:, np.newaxis] ** 2), axis=0)
        w -= np.max(w)
        new_wgts = w - np.log(np.sum(np.exp(w)))  # normalize

        """
        y = np.exp(new_wgts)
        x = self.particles[0, :, alpha_ix]
        xymask = ~pd.isnull(x) & ~pd.isnull(y)
        y = y[xymask]
        x = x[xymask]
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        fig, ax = plt.subplots()
        ax.semilogy()
        ax.scatter(x, y, c=z, s=100, cmap='inferno')
        ax.set_ylim(0., np.max(y))
        plt.title(date)

        x = self.particles[-1, :, alpha_ix]
        y = np.exp(new_wgts)
        xymask = ~pd.isnull(x) & ~pd.isnull(y)
        y = y[xymask]
        x = x[xymask]
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        fig, ax = plt.subplots()
        ax.semilogy()
        ax.scatter(x, y, c=z, s=100, cmap='inferno')
        ax.set_ylim(0., np.max(y))
        plt.title(date)

        #plt.figure()
        #p_flat = self.particles[:, :, alpha_ix].flatten()
        #o_flat = obs.flatten()
        #rand_sel_p = np.random.randint(0, len(p_flat), 10000)
        #rand_sel_o = np.random.randint(0, len(o_flat), 10000)
        #plt.hist(obs.flatten()[rand_sel_o], bins=50, alpha=0.7, label='OBS')
        #plt.hist(p_flat[rand_sel_p], bins=50, alpha=0.7, label='MOD')
        #plt.legend()
        """

        self.log_weights = np.repeat(new_wgts[np.newaxis, ...],
                                     self.log_weights.shape[0], axis=0)

        # add posterior to the plot
        try:
            avg = np.average(self.particles[:, :, alpha_ix],
                             weights=self.weights[0, :], axis=1)
            ax1.errorbar(np.arange(self.particles.shape[0]), avg, yerr=np.sqrt(
                np.average((self.particles[:, :, alpha_ix] - np.atleast_2d(
                    avg).T) ** 2, weights=self.weights[0, :], axis=1)),
                         label='POST')
            ax1.legend()
            xs = np.arange(self.particles.shape[0])
            ys_m = avg
            std_m = np.sqrt(np.average(
                (self.particles[:, :, alpha_ix] - np.atleast_2d(avg).T) ** 2,
                weights=self.weights[0, :], axis=1))
            std_m_top = ys_m + std_m
            std_m_btm = ys_m - std_m
            ax2.plot(xs, ys_m, linestyle='-', color=lc[2], lw=2, zorder=100)
            ax2.fill_between(xs, std_m_btm, std_m_top, facecolor=c[2],
                             alpha=0.3, zorder=100)
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color=lc[0]),
                            Line2D([0], [0], color=lc[1]),
                            Line2D([0], [0], color=lc[2])]
            ax2.legend(custom_lines, labels=['PRIOR', 'OBS', 'POST'])
            fig.suptitle(date)
        except (np.linalg.LinAlgError, ValueError):
            pass

    def update_with_snowline_obs(self, obs, swe_ix=None, date=''):
        """

        Returns
        -------

        """
        obs_indices = np.where(~np.isnan(obs))
        try:
            w = copy.deepcopy(self.log_weights[obs_indices[-1][0], :])
        except IndexError:
            return

        swe_mod = self.particles[:, :, swe_ix]
        fsca_mod = swe_to_fsca_zaitchik(swe_mod)
        fsca_obs_mean = np.nanmean(obs, axis=0)
        fsca_obs_std = np.nanstd(obs, axis=0)

        """
        fig, [ax1, ax2] = plt.subplots(2, sharex=True)
        # swe on ax1
        avg = np.average(swe_mod, weights=self.weights[0, :], axis=1)
        ax1.errorbar(np.arange(self.particles.shape[0]), avg, yerr=np.sqrt(
            np.average((swe_mod - np.atleast_2d(avg).T) ** 2,
                weights=self.weights[0, :], axis=1)), label='MOD PRIOR')

        ax1.legend()
        lc = ['g', 'b', 'k']
        c = ["g", "b", 'k']
        xs = np.arange(self.particles.shape[0])
        ys_o = fsca_obs_mean
        ys_m = np.average(fsca_mod, weights=self.weights[0, :], axis=1)
        std_m = np.sqrt(np.average((fsca_mod - np.atleast_2d(ys_m).T) ** 2,
            weights=self.weights[0, :], axis=1))
        std_m_top = ys_m + std_m
        std_m_btm = ys_m - std_m
        std_o = fsca_obs_std
        std_o_top = ys_o + std_o
        std_o_btm = ys_o - std_o
        ax2.plot(xs, ys_m, linestyle='-', color=lc[0], lw=2, zorder=100, 
                 label='MOD fSCA PRIOR')
        ax2.fill_between(xs, std_m_btm, std_m_top, facecolor=c[0], alpha=0.3,
                         zorder=100)
        ax2.plot(xs, ys_o, linestyle='-', color=lc[1], lw=2, label='OBS fSCA')
        ax2.fill_between(xs, std_o_btm, std_o_top, facecolor=c[1], alpha=0.3)
        """
        for obs_loc in np.unique(obs_indices[-1]):
            # obs_s_ix = obs_spatial_ix[obs_loc]
            # swe_mod_slice = swe_mod[obs_s_ix, ...]
            # swe_particles_init = p[obs_s_ix, :, obs_init_mb_ix] necessary???

            # swe_obs_slice = obs[obs_loc, :]
            # obs_logit_slice_mean = np.nanmean(obs_logit_slice)
            # obs_logit_slice_std = np.nanstd(obs_logit_slice)

            # swe_obs = (swe_particles - swe_particles_init) / 0.9
            fsca_mod_slice = fsca_mod[obs_loc, ...]
            fsca_obs_slice_mean = fsca_obs_mean[obs_loc]
            fsca_obs_slice_std = fsca_obs_std[obs_loc]

            # todo: implement covariance solution
            w += -((fsca_obs_slice_mean - fsca_mod_slice) ** 2.) / (
                        2. * fsca_obs_slice_std ** 2)

        # if sla_ix is not None:
        #    set_to_zero = np.where(self.particles[sla_ix:, :, swe_ix] > 0.)

        w -= np.max(w)

        new_wgts = w - np.log(np.sum(np.exp(w)))  # normalize

        self.log_weights = np.repeat(new_wgts[np.newaxis, ...],
                                     self.log_weights.shape[0], axis=0)

        """
        # add posterior to the plot
        try:
            # swe on ax1
            avg_post = np.average(self.particles[:, :, swe_ix],
                             weights=self.weights[0, :], axis=1)
            ax1.errorbar(np.arange(self.particles.shape[0]), avg_post, 
                         yerr=np.sqrt(
                np.average((self.particles[:, :, swe_ix] - np.atleast_2d(
                    avg_post).T) ** 2, weights=self.weights[0, :], axis=1)),
                         label='SWE MOD POST')
            ax1.legend()
            xs = np.arange(self.particles.shape[0])
            avg_fsca = np.average(fsca_mod,
                             weights=self.weights[0, :], axis=1)
            ys_m = avg_fsca
            std_m = np.sqrt(np.average(
                (fsca_mod - np.atleast_2d(avg_fsca).T) ** 2,
                weights=self.weights[0, :], axis=1))
            std_m_top = ys_m + std_m
            std_m_btm = ys_m - std_m
            ax2.plot(xs, ys_m, linestyle='-', color=lc[2], lw=2, zorder=100)
            ax2.fill_between(xs, std_m_btm, std_m_top, facecolor=c[2],
                             alpha=0.3, zorder=100)
            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color=lc[0]),
                            Line2D([0], [0], color=lc[1]),
                            Line2D([0], [0], color=lc[2])]
            ax2.legend(custom_lines, labels=['fSCA PRIOR', 'fSCA OBS', 
            'fSCA POST'])
            fig.suptitle(date)
        except (np.linalg.LinAlgError, ValueError):
            pass
        """

    def update_together(self, obs, mb_ix, obs_init_mb_ix, swe_ix, alpha_ix,
                        fudge_fac=0.0001):
        """
        Update the prior estimate with observations.

        Parameters
        ----------
        obs :
        mb_ix :
        obs_std :
        R :
        obs_ix :
        obs_init_mb_ix :
        obs_spatial_ix :
        fudge_fac:

        Returns
        -------

        """

        """
        Rinv = spla.pinv(R)

        obs_indices = np.where(~np.isnan(obs))
        """
        # todo: implement covariance solution with (a) more than one obs
        #  variable and (b) more than one obs locations
        # todo: not logic a this position, but weights over spatial domain
        #  should be all the same -> replace with self.stats_ix?
        w = copy.deepcopy(self.log_weights[obs_indices[-1][0], :])
        """
        for obs_count, obs_loc in enumerate(obs_indices[-1]):
            obs_s_ix = obs_spatial_ix[obs_loc]
            mb_particles = self.particles[obs_s_ix, :, obs_ix]
            mb_particles_init = self.particles[obs_s_ix, :, obs_init_mb_ix]

            h_status = (mb_particles - mb_particles_init) / 0.9
            h_obs = obs[0, obs_loc] / 0.9
            h_obs_std = obs_std[0, obs_loc] / 0.9

            # todo: R or Rinv?
            w += (-0.5 * ((h_obs - h_status).T @ Rinv) @ (h_obs - h_status))

            # old: w += -((h_obs - h_status) ** 2.) / (2. * h_obs_std ** 2)

        w -= np.max(w)

        new_wgts = w - np.log(np.sum(np.exp(w)))  # normalize
        """

        # transform modeled variables
        ptcls = self.particles[..., [mb_ix, swe_ix, alpha_ix]]
        mb_particles_init = self.particles[..., obs_init_mb_ix]
        # correct for init MB
        ptcls[..., 0] = (ptcls[..., 0] -
                         self.particles[..., obs_init_mb_ix]) / 0.9
        # swe to fSCA
        ptcls[..., 1] = utils.physical_to_logit(swe_to_fsca_zaitchik(ptcls[..., 1]), 0., 1., fudge_fac)
        # alpha to logit space
        ptcls[..., 2] = utils.physical_to_logit(ptcls[..., 2], 0., 1., fudge_fac)

        # same for the observations
        obs[..., 0] /= 0.9
        obs[..., 1] = utils.physical_to_logit(obs[..., 1], 0., 1., fudge_fac)
        obs[..., 2] = utils.physical_to_logit(obs[..., 2], 0., 1., fudge_fac)

        obs_mean = np.nanmean(obs, axis=1)
        obs_std = np.nanstd(obs, axis=1)

        # determine R by the squared STDs of the OBSs
        Rinv = np.linalg.inv(np.eye(3) * obs_std**2)

        dep = obs_mean - ptcls
        w += np.dot(dep.T, np.dot(Rinv, dep))
        # hä?
        # w += np.nansum(((obs_mean - ptcls) ** 2.)/ (2. * R), axis=1)

        w -= np.max(w)

        new_wgts = w - np.log(np.sum(np.exp(w)))  # normalize

        self.log_weights = np.repeat(new_wgts[np.newaxis, ...],
                                     self.log_weights.shape[0],
                                     axis=0)

    def resample(self, phi=0.1, gamma=0.05, diversify=False, seed=None):
        """
        Resample adaptively.

        We choose the number of particles to be resampled per model as the
        minimum contribution $\phi$ plus some excess frequency L_{t,j} which
        depends on the model probability.

        Returns
        -------
        None
        """

        # first step: save posterior mean and covariance of parameters
        pmean = self.param_mean
        pcov = self.param_covariance

        # get number of excess frequencies
        excess_particles = int(self.n_tot * (1 - self.n_models * phi))
        excess_shares = [np.clip(pi_t_j - np.log(phi), 0, None) for pi_t_j in
                         self.model_prob_log]

        excess_shares /= np.sum(excess_shares)

        np.random.seed(seed)
        L_t_j = np.random.choice(self.n_models, excess_particles,
                                 p=excess_shares)

        # determine number of particles to be resampled per model
        N_t_j = phi * self.n_tot + np.array([np.sum(L_t_j == i) for i in
                                             range(self.n_models)])

        # apply resampling procedure per model with arbitrary resampling method
        # divide first by model prob., to get the weights right for resampling
        # must be a list, since particles sizes can differ
        weights_for_resamp = [
            np.exp(self.model_weights_log[i] -
                   self.model_prob_log[i]) for i in self.model_range]
        resample_indices = [stratified_resample(weights_for_resamp[i],
                                                N_t_j[i],
                                                one_random_number=True) for i
                            in range(self.n_models)]
        print('UNIQUE PTCLS to RESAMPLE: ', [len(np.unique(ri)) for ri in
              resample_indices])
        ric = [list(resample_indices[0])]
        maxima = np.cumsum([x for x in self.n_model_particles]) - 1
        ric.extend(
            [resample_indices[i] + maxima[i - 1] for i in
             np.arange(self.n_models)[:-1]+1])

        ric = np.concatenate(ric)
        # ancest_ix.append(ric)

        new_p = []
        for m in range(self.n_models):
            npm, _ = resample_from_index_augmented(
                self.particles[:, self.model_indices_all[m]],
                self.weights[:, self.model_indices_all[m]],
                resample_indices[m])
            new_p.append(npm)

        n_test = np.concatenate(new_p, axis=1)[0, :,
                                               self.state_indices['xi'][0]]
        b_test = self.particles[0, :, self.state_indices['xi'][0]]
        anc = []
        for p in n_test:
            anc.extend(list(np.where(b_test == p)[0]))
        # ancest_all.append(anc)

        # compensate for over-/underrepresentation
        new_weights_per_model = self.model_prob_log - np.log(N_t_j)

        if (new_weights_per_model == 0.).any() or np.isinf(
                new_weights_per_model).any():
            raise ValueError('New weights contain zero on Inf.')

        self.particles = np.concatenate(new_p, axis=1)

        # important: AFTER setting new particles; later comment: why again?
        self.log_weights = np.repeat(
            np.hstack(
                [np.tile(new_weights_per_model[i], int(N_t_j[i])) for i in
                 self.model_range])[np.newaxis, ...],
            self.spatial_dims[0], axis=0)

        if diversify is True:
            self.diversify_parameters(pmean, pcov, gamma=gamma)

    def diversify_parameters(self, post_means, post_covs, gamma=0.05):
        """
        Apply a parameter diversification after [1]_.

        Parameters
        ----------
        post_means:
            Posterior parameter means.
        post_covs:
            Posterior parameter covariances.
        gamma: float, optional
            Reasonable values are 0.05 or 0.1. Default: 0.05.

        Returns
        -------
        None

        References
        ----------
        .. [1] Liu, J. & West, M. Doucet, A.; de Freitas, J. F. G. & Gordon,
               N. J. (Eds.): Combined parameter and state estimation in
               simulation based filtering. Sequential Monte Carlo in
               Practice, Springer, 2001, 197-223.
        """

        param_means = post_means
        param_covs = post_covs
        mix = self.model_indices_all

        model_ptcls_all = []
        for j in range(self.n_models):
            # model_ptcls = self.particles[mix[j]]
            model_ptcls = self.particles[self.stats_ix, mix[j]]
            mu_t_j = param_means[j]
            cov_theta_t_j = param_covs[j]
            theta_tilde_t_k_nan = self.particles[
                                      self.stats_ix, mix[j]][
                                  :, self.state_indices['theta']]
            theta_tilde_t_k = theta_tilde_t_k_nan[
                              :, ~np.isnan(theta_tilde_t_k_nan).all(axis=0)]

            psi_k_t = np.random.multivariate_normal(np.zeros(
                    cov_theta_t_j.shape[0]), cov_theta_t_j,
                    size=len(theta_tilde_t_k))

            theta_tilde_t_k_new = \
                mu_t_j + (1 - gamma) * (theta_tilde_t_k - mu_t_j) + \
                np.sqrt(1 - (1 - gamma)**2) * psi_k_t

            theta_tilde_t_k_new_pad = np.pad(
                theta_tilde_t_k_new,
                ((0, 0), (0, self.n_aug_vars - theta_tilde_t_k_new.shape[1])),
                'constant', constant_values=np.nan
            )

            model_ptcls[:, self.state_indices['theta']] = \
                theta_tilde_t_k_new_pad
            model_ptcls_all.append(model_ptcls)

        self.particles = np.repeat(
            np.concatenate(model_ptcls_all)[np.newaxis, :, :],
            self.spatial_dims[0], axis=0
        )

    def evolve_theta(
            self, mu_0: list, Sigma_0: list, rho: float = 0.9,
            change_mean: bool = True, seed: Optional[int] = None):
        """
        Make the model parameters time-varying.

        Parameters
        ----------
        mu_0: list of arrays
            Initial parameter means.
        Sigma_0 : list of arrays
            Initial parameter covariance
        rho : float
            Memory parameter between 0 and 1. Values should e.g. be 0.8 or
            0.9. Default: 0.9.
        change_mean: bool
            if the mean should be change back to the prior. If no (False),
            the only the variability of the prior is given back to the
            parameter distribution.
        seed: int, optional
            Which random number generator seeds to use (for repeatable
            experiments). Default: None (non-repeatable).

        Returns
        -------
        None.
        """

        mix = self.model_indices_all
        model_ptcls_all = []
        for j in range(self.n_models):

            model_ptcls = self.particles[:, mix[j]]
            # todo: check if cov matrix should be transposed
            theta_j_nan = self.particles[:, mix[j]][..., self.state_indices[
                'theta']]
            theta_j = theta_j_nan[
                ..., ~np.isnan(theta_j_nan[0, ...]).all(axis=0)]
            theta_j = np.log(theta_j)
            np.random.seed(seed)
            zeta_t = np.random.multivariate_normal(np.zeros(len(Sigma_0[j])),
                                                   (1 - rho ** 2) * Sigma_0[j],
                                                   size=theta_j.shape[1])
            if change_mean is True:
                theta_j_new = rho * theta_j + (1 - rho) * mu_0[j] + zeta_t
            else:
                print('NOT CHANGING MEMORY MEAN')
                theta_j_new = theta_j + zeta_t

            theta_j_new = np.exp(theta_j_new)

            theta_j_new_pad = np.pad(theta_j_new, ((0, 0),
                                                   (0, 0), (0,
                                                            self.n_aug_vars -
                                                            theta_j_new.shape[
                                                                -1])),
                                     'constant', constant_values=np.nan)

            model_ptcls[..., self.state_indices['theta']] = \
                theta_j_new_pad
            model_ptcls_all.append(model_ptcls)

        # todo: here we assume that the order er of particles stays
        #  completely the same (and so the weights)- > check this
        self.particles = np.concatenate(model_ptcls_all, axis=1)

    def step(self, pred: np.ndarray, date: pd.Timestamp,
             obs: np.ndarray or None = None,
             obs_std: np.ndarray or None = None,
             predict_method: str = 'random',
             update_method: str = 'gauss') -> None:
        """
        Make a step forward in time.

        Parameters
        ----------
        pred: (N, M) array_like
            Array with model predictions. N is the particles dimension, M, is
            the discretization dimension.
        date: pd.Timestamp
            Date/time for which the step is valid.
        obs: (N, M), (M,) array_like or None
            # todo: adjust shapes if observations don't have n_particles yet!?
            Array with observations or None. It can be either the raw
            observations in shape (N, M) or one observation value per
            discretization (M,). If None, the update step is skipped. Default:
            None.
        obs_std: array_like or None
            Array with observation standard deviations or None. It can be
            either the raw observations in shape (N, M) or one observation
            value per discretization node (M,). Only makes sense in combination
            with `obs`. Default: None.
        predict_method: str
            Method used to make a prediction on the particles: Must be either
            of ['gauss'|'random'|'gamma']. If 'gauss', then a Gaussian particle
            distribution will the generated from the model predictions. If
            'gamma', a Gamma distribution will be generated for the
            observations. If 'random', all given observations will be added
            randomly on existing particles.
        update_method: str
            Method used for updating the prior with the likelihood. Must be
            either of ['gauss'|'truncgauss'|'random'|'gamma']. Default:
            'gauss'.

        Returns
        -------
        None
        """

        # 1) predict
        if predict_method == 'gauss':
            # todo: find a way to insert std dev here
            self.predict_gaussian(pred, np.nanstd(pred, axis=1))
        elif predict_method == 'random':
            self.predict_random(pred)
        elif predict_method == 'gamma':
            self.predict_gamma(pred)
        else:
            raise ValueError(
                'Prediction method {} does not exist.'.format(predict_method))

        # 2) update, if there is an observation
        if obs is not None:

            if update_method == 'gauss':
                self.update_gaussian(obs, obs_std)
            elif update_method == 'truncgauss':
                self.update_truncated_gaussian(obs, obs_std)
            elif update_method == 'random':
                raise NotImplementedError
            elif update_method == 'gamma':
                self.update_gamma(obs, obs_std)
            else:
                raise ValueError(
                    'Update method {} does not exist.'.format(update_method))
        else:
            log.info('No observation available on {}.'.format(date))

        # 3) resample, if needed
        self.resample_if_needed()

        # 4) save, if wanted
        if self.do_save is True:
            raise NotImplementedError
            # self.to_file()

        # 5) plot, if wanted
        if self.do_plot is True:
            self.plot()

        # if self.do_save:
        #    self.save()
        # if self.do_ani:
        #    self.ani()

        # if self.plot_save:
        #    self.p_save()

    def estimate_state(self, space_ix=None, ptcls_ix=None, space_avg=False,
                       return_std=False):
        """
        Calculate mean and variance of the weighted particles.

        Parameters
        ----------
        space_ix:
            For which spatial index the state shall be retrieved. Default: None
            (all).
        space_avg: bool, optional
            Whether to average spatially or not. Default: False (do not average
             over space).
        ptcls_ix:
            Which particles to use for estimating the state (for getting states
             by model). Default: None (get state for all particles).
        return_std: bool, optional
            Whether the standard deviation shall be returned instead of the
            variance. Default: False.

        Returns
        -------
        mean, var: tuple
            Tuple of np.ndarrays with mean and variance (or standard deviation)
            of the desired particles.
        """

        # todo: SHIT!!!!! For the MB state we need to subtract the initial MB
        #  at camera setup! => build this option
        if space_ix is not None:
            ptcls = self.particles[space_ix, ...].copy()
        else:
            ptcls = self.particles.copy()

        wgts = self.weights[self.stats_ix, ...].copy()  # shouldn't matter
        wgts += 1e-300  # to avoid weights round-off to zero in physical space

        if ptcls_ix is not None:
            ptcls = ptcls[:, ptcls_ix, :]
            wgts = wgts[ptcls_ix]

        mean = np.average(ptcls, weights=wgts, axis=-2)
        if isinstance(space_ix, np.integer):
            mean_subtract = mean
        else:
            mean_subtract = np.moveaxis(np.atleast_3d(mean), 2, 1)
        var = np.average((ptcls - mean_subtract) ** 2,
                         weights=wgts, axis=-2)
        # todo: correct?
        if space_avg is True:
            mean = np.mean(mean, axis=0)
            var = np.mean(var, axis=0)

        if return_std is True:
            var = np.sqrt(var)
        return mean, var

    def estimate_state_by_model(self, space_ix=None, space_avg=False,
                                return_std=False):

        """
        Wrapper around `estimate_state` to return all statistics per model.

        Parameters
        ----------
        space_ix :
        ptcls_ix :
        space_avg :
        return_std :

        Returns
        -------

        """
        mean_all = []
        disp_all = []
        for mi in self.model_indices_all:
            mmean, mdisp = self.estimate_state(
                space_ix=space_ix, ptcls_ix=mi, space_avg=space_avg,
                return_std=return_std
            )

            mean_all.append(mmean)
            disp_all.append(mdisp)

        return mean_all, disp_all

    def plot_state_errorbars(self, ax, date, var_ix=1, colors=None,
                             space_ix=None, by_model=False):
        """
        Plot the models estimates with errorbars.

        Parameters
        ----------
        ax :
        date:
        var_ix :
            Default: 1 (plot mass balance).
            # todo: take into account reset MB?
        colors:
        space_ix:
        by_model:

        Returns
        -------

        """
        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        if by_model is True:
            mod_mean, mod_disp = self.estimate_state_by_model(
                space_ix=space_ix, return_std=True)

            # select the desired variable
            mod_mean = [m[var_ix] for m in mod_mean]
            mod_disp = [d[var_ix] for d in mod_disp]

            y_jitter = np.array([pd.Timedelta(hours=2 * td) for td in
                                 np.linspace(-2.0, 2.0, self.n_models)])
            y_vals = np.array([date] * self.n_models) + y_jitter
            # ax.scatter(y_vals, mod_mean, c=colors[:self.n_models])
            ax.errorbar(y_vals, mod_mean, yerr=mod_disp, fmt='o', zorder=0,
                        c=colors[:self.n_models])
        else:
            # plot ensemble estimate
            mod_mean, mod_disp = self.estimate_state(space_ix=space_ix,
                                                     return_std=True)

            # select the desired variable
            mod_mean = mod_mean[..., var_ix]
            mod_disp = mod_disp[..., var_ix]

            if isinstance(space_ix, list):
                # it might happen that for the space subset weights sum to zero
                mod_mean = np.average(
                    mod_mean, weights=self.weights[0, :][space_ix] + 1e-300)
                mod_disp = np.average(
                    mod_disp, weights=self.weights[0, :][space_ix] + 1e-300)

            # ax.scatter(date, mod_mean, c=colors[:self.n_models])
            ax.errorbar(date, mod_mean, yerr=mod_disp, fmt='o', zorder=0,
                        c=colors[0])

    def plot_particles_per_model(self, ax, date, colors=None):
        """
        Plot stacked bars showing how many particles each model currently has.

        Parameters
        ----------
        ax :
        date :
        colors :

        Returns
        -------

        """

        if colors is None:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        ppm = self.n_model_particles
        for ppi, pp in enumerate(ppm):
            if ppi == 0:
                ax.bar(date, pp, color=colors[ppi])
            else:
                ax.bar(date, pp, bottom=np.sum(ppm[:ppi]), color=colors[ppi])


def melt_braithwaite(psol=None, mu_ice=None, tmean=None, swe=None,
                     prcp_fac=None, tmelt=0., tmax=None, sis=None):
    """
    Calculate melt according to Braithwaite.

    Parameters
    ----------
    psol :
    mu_ice :
    tmean :
    swe :
    prcp_fac :
    tmelt :
    tmax :
    sis :

    Returns
    -------

    """
    tempformelt = tmean - tmelt
    tempformelt[tmean <= tmelt] = 0.
    mu = np.ones_like(swe) * mu_ice
    mu[swe > 0.] = mu_ice * cfg.PARAMS['ratio_mu_snow_ice']

    return mu * tempformelt / 1000.


def melt_hock(psol=None, mu_hock=None, a_ice=None, tmean=None,
              ipot=None, prcp_fac=None, swe=None, tmelt=0., tmax=None,
              sis=None):
    """
    Calculate melt according to Hock.

    Parameters
    ----------
    psol :
    mu_hock :
    a_ice :
    tmean :
    ipot :
    prcp_fac :
    swe :
    tmelt :
    tmax :
    sis :

    Returns
    -------

    """
    tempformelt = tmean - tmelt
    tempformelt[tmean <= tmelt] = 0.
    a = np.ones_like(swe) * a_ice
    a[swe > 0.] = a_ice * cfg.PARAMS['ratio_a_snow_ice']
    melt_day = (mu_hock + a * ipot) * tempformelt
    return melt_day / 1000.


def melt_pellicciotti(psol=None, tf=None, srf=None, tmean=None,
                      sis=None, alpha=None, tmelt=1., prcp_fac=None,
                      tmax=None):
    """
    Calculate melt according to Pellicciotti.

    Parameters
    ----------
    psol :
    tf :
    srf :
    tmean :
    sis :
    alpha :
    tmelt :
    prcp_fac :
    tmax :

    Returns
    -------

    """
    melt_day = tf * tmean + srf * (1 - alpha) * sis
    melt_day[tmean <= tmelt] = 0.

    return melt_day / 1000.


def melt_oerlemans(psol=None, c0=None, c1=None, tmean=None, sis=None,
                   alpha=None, prcp_fac=None, tmax=None):
    """
    Calculate melt according to Oerlemans.

    Parameters
    ----------
    psol :
    c0 :
    c1 :
    tmean :
    sis :
    alpha :
    prcp_fac :
    tmax :

    Returns
    -------

    """
    # IMPORTANT: sign of c0 is changed to make c0 positive (log!)
    qmelt = (1 - alpha) * sis - c0 + c1 * tmean
    # melt only happens where qmelt > 0.:
    qmelt = np.clip(qmelt, 0., None)

    # kg m-2 d-1 = W m-2 * s * J-1 kg
    # we want ice flux, so we drop RHO_W for the first...!?
    melt = (qmelt * cfg.SEC_IN_DAY) / cfg.LATENT_HEAT_FUSION_WATER

    return melt / 1000.


def point_albedo_brock(swe, t_acc, icedist, p1=0.713, p2=0.112, p3=0.442,
                       p4=0.058, a_u=None, d_star=0.024, alpha_max=0.85,
                       ice_alpha_std=0.075):
    """
    Calculate the point albedo with the Brock model.

    Parameters
    ----------
    swe : np.ndarray
        Array of shape (M, N), where M is a spatial discretization and N
    t_acc :
    icedist :
    p1 :
    p2 :
    p3 :
    p4 :
    a_u :
    d_star :
    alpha_max :
    ice_alpha_std : float
        Standard deviation for the ice albedo.

    Returns
    -------

    """
    if a_u is None:
        # todo: how to construct a logit normal distr. here?
        # todo: this has been changed after the second manuscript submission
        a_u = stats.truncnorm.rvs(
            -cfg.PARAMS['ice_albedo_default'] / ice_alpha_std,
            (1. - cfg.PARAMS['ice_albedo_default']) / ice_alpha_std,
            loc=cfg.PARAMS['ice_albedo_default'],
            scale=0.1, size=swe.size)
    # todo: clipping compatible with assimilation of albedo? I guess no...
    # alpha_ds = np.clip((p1 - p2 * np.log10(t_acc)), None, alpha_max)
    alpha_ds = np.clip((p1 - p2 * np.log10(t_acc)), None, 1.)

    # shallow snow equation
    alpha_ss = np.clip((a_u + p3 * np.exp(-p4 * t_acc)), None, alpha_max)
    # combining deep and shallow
    alpha = (1. - np.exp(-swe / d_star)) * alpha_ds + np.exp(
        -swe / d_star) * alpha_ss
    # where there is ice, put its default albedo - with noise!
    # todo: comment back in again!?
    # if isinstance(a_u, float):
    #    alpha[icedist] = cfg.PARAMS['ice_albedo_default']
    # else:
    #    alpha[icedist] = a_u[:, 0]
    return alpha


def tacc_from_alpha_brock(alpha, p1=0.86, p2=0.155):
    """
    Infer the sum of positive accumulated temperature from the albedo.

    Parameters
    ----------
    alpha :
    p1 :
    p2 :

    Returns
    -------

    """
    # here we can only take the deep snow equation, otherwise it's not unique
    tacc = 10.**((alpha - p1) / (-p2))
    # todo: bullshit
    tacc[tacc < 1.] = 1.
    return tacc


def swe_to_fsca_zaitchik(swe, full_sca_swe=0.013, tau=4.):
    """
    Turn snow water equivalent into the fraction of snow covered area by the
    formulation of [1]_

    Parameters
    ----------
    swe : array-like
        Array with snow water equivalent.
    full_sca_swe : float
        Value of snow water equivalent needed for an area to be considered as
        100% snow covered (fSCA=1). Default= 0.013 m w.e. (value for bare soil
        in the publication).
    tau : float
        Tunable factor. Default: 4. (as in the publication).

    Returns
    -------
    fSCA as a function of the SWE input.

    References
    ----------
    .. [1]:
    """
    return np.minimum(1 - (np.exp(- tau * swe / full_sca_swe) - (
                swe / full_sca_swe) * np.exp(-tau)), 1.)


def get_initial_conditions(gdir, date, n_samples, begin_mbyear=None,
                           min_std_swe=0.025, min_std_alpha=0.05,
                           min_std_tacc=10., param_dict=None, fl_ids=None,
                           alpha_underlying=None, param_prior_distshape=None,
                           param_prior_std_scalefactor=None, mb_models=None,
                           generate_params=None, detrend_params=None,
                           swe_and_tacc_dist='logit', seed=None):
    """
    Get initial conditions of SWE and albedo.

    The initial conditions are drawn from Gaussians fitted to the model
    background at the start date.

    Parameters
    ----------
    gdir : `py:class:crampon.GlacierDirectory`
        A CRAMPON GlacierDirectory.
    date : pd.Timestamp
        Date for which to get the initial conditions.
    n_samples : int
        Number of initial conditions to generate.
    begin_mbyear : pd.Timestamp or None
        Start of the mass budget year from which the calculation shall
        start. Default: None (compute from `date` and the specifications in
        the parameter configuration file).
    min_std_swe : float
        Minimum standard deviation for the snow water equivalent. Introduces
        some minimum noise to correct for potentially wrong model estimates.
        Default: 0.025 m w.e..
    min_std_alpha :
        Minimum standard deviation for the albedo. Introduces
        some minimum noise to correct for potentially wrong model estimates.
        Default: 0.05.
    min_std_tacc
    param_dict
    fl_ids: array-like or None
        Flow line IDs to select. Default: None (return all flow lines)
    alpha_underlying
    param_prior_distshape
    param_prior_std_scalefactor
    mb_models
    generate_params
    detrend_params: bool, optional
        Whether or not to detrend the calibrated parameters: If True, a line is
        fitted through the parameters, and its slope is used to remove the
        trend from past parameters. This might be useful to compensate for the
        lack in geometry change. Default: False.
    swe_and_tacc_dist
    seed

    Returns
    -------

    """

    # todo: this function should actually account for correlations in swe,
    #  alpha and MB -> not easy though, since not all models have alpha

    if begin_mbyear is None:
        begin_mbyear = utils.get_begin_last_flexyear(date)

    n_samples_per_model = int(n_samples / len(mb_models))
    n_samples_per_model_cs = np.zeros(len(mb_models) + 1, dtype=int)
    n_samples_per_model_cs[1:] = np.cumsum(np.array(len(mb_models) *
                                                    [n_samples_per_model]))
    n_params_per_model = [len(m.cali_params_list) for m in mb_models]
    n_params_per_model_cs = np.zeros(len(n_params_per_model) + 1, dtype=int)
    n_params_per_model_cs[1:] = np.cumsum(np.array([len(m.cali_params_list) for
                                                    m in mb_models]))

    params = []
    params_constr = []
    if param_dict is None:
        for mbm in mb_models:

            pg = ParameterGenerator(
                gdir, mbm, latest_climate=True, only_pairs=True,
                narrow_distribution=0.,
                # bw_constrain_year=begin_mbyear.year+1,
                output_type='array', constrain_with_bw_prcp_fac=False)

            params.append(pg.from_single_glacier(detrend=detrend_params))

            pg_constr = ParameterGenerator(
                gdir, mbm, latest_climate=True, only_pairs=True,
                narrow_distribution=0., bw_constrain_year=begin_mbyear.year+1,
                output_type='array', constrain_with_bw_prcp_fac=True)

            params_constr.append(pg_constr.from_single_glacier(
                detrend=detrend_params))
    # else:
    #    params = [np.atleast_2d(
    #        np.array([v for k, v in param_dict.items() if m in k])) for m
    #        in [m.__name__ for m in mb_models]]

    init_cond = make_mb_current_mbyear_heights(
        gdir, begin_mbyear=begin_mbyear, last_day=date, write=False,
        param_dict=param_dict, mb_models=mb_models)

    stack_order = ['model', 'member']
    # now the problem is that the min date is not necessarily OCT-1
    mb_init_raw_sel = init_cond[0].sel(time=slice(
        utils.get_begin_last_flexyear(date), None)).stack(ens=stack_order)
    # make cumsum and stack models and members
    mb_init_raw = mb_init_raw_sel.cumsum(dim='time').isel(time=-1).MB.values
    mb_mean = np.nanmean(mb_init_raw, axis=1)
    mb_std = np.nanstd(mb_init_raw, axis=1)
    # ... so we only take random draws from the model runs
    np.random.seed(seed)
    mb_init_random = mb_init_raw[
                     :, np.random.randint(0, mb_init_raw.shape[1],
                                          size=n_samples)
                     ]
    swe_mean = np.nanmean(np.array(init_cond[1]), axis=0)
    alpha_mean = np.nanmean(np.array(init_cond[2]), axis=0)
    tacc_mean = np.nanmean(np.array(init_cond[3]), axis=0)
    swe_std = np.clip(np.nanstd(np.array(init_cond[1]), axis=0),
                      min_std_swe, None)
    alpha_std = np.clip(np.nanstd(np.array(init_cond[2]), axis=0),
                        min_std_alpha, None)
    tacc_std = np.clip(np.nanstd(np.array(init_cond[3]), axis=0), min_std_tacc,
                       None)
    np.random.seed(seed)
    rand_num = np.random.randn(n_samples)
    swe_init = np.clip(rand_num * np.atleast_2d(swe_std).T +
                       np.atleast_2d(swe_mean).T, 0., None)
    # todo: by choosing the same random number, we assume correlation=-1
    #  between tacc and swe: the correlation should come from the model though
    # clip at overall modeled t_acc maximum - not ideal, but otherwise the
    # numbers can go crazy
    if len(init_cond[3]) > 0:
        tacc_init = np.clip(rand_num * np.atleast_2d(tacc_std).T +
                            np.atleast_2d(tacc_mean).T, np.min(init_cond[3]),
                            np.max(init_cond[3]))
        alpha_init = point_albedo_brock(swe_init, tacc_init, swe_init == 0.0,
                                        a_u=np.atleast_2d(alpha_underlying).T)
    else:
        tacc_init = np.full_like(swe_init, np.nan)
        alpha_init = np.full_like(swe_init, np.nan)

    if param_dict is not None:
        if fl_ids is not None:
            swe_init = swe_init[fl_ids, :]
            alpha_init = alpha_init[fl_ids, :]
            mb_init_random = mb_init_random[fl_ids, :]
            mb_init_raw = mb_init_raw[fl_ids, :]
            tacc_init = tacc_init[fl_ids, :]
            init_cond_all = init_cond[0].isel(fl_id=fl_ids)
        else:
            init_cond_all = init_cond[0].copy()
        params_prior = [p[n_samples_per_model_cs[i]:
                          n_samples_per_model_cs[i+1],
                        n_params_per_model_cs[i]:
                        n_params_per_model_cs[i+1]]
                        for i, p in enumerate(params)]
        # no need to calculate covariances
        return init_cond_all, mb_init_raw, mb_init_random, swe_init, \
               alpha_init, tacc_init, theta_mean_log, theta_cov_log, \
               params_prior

    if param_prior_distshape is None:
        param_prior_distshape = 'lognormal'
    if param_prior_std_scalefactor is None:
        param_prior_std_scalefactor = [1.0, 1.0, 1.0, 1.0]
    if mb_models is None:
        mb_models = cfg.MASSBALANCE_MODELS
    if generate_params is None:
        generate_params = 'past'

    # see whether models are mixed between their runtimes
    if np.array(init_cond[2]).shape[0] != np.array(init_cond[1]).shape[0]:
        common_model_time_start = np.array(init_cond[2]).shape[0]
    else:
        common_model_time_start = 0

    # we take the overall covariance - calculating by elevation gives
    # too few variability
    # todo: replace the 'common_model_time_start:' as soon as we calculate
    #  covariances by model
    log_const = 0.000001
    tacc_upper_bound = 1.9e6  # from the albedo equation (alpha > 0.01)
    # just a random value so that swe doesn't go crazy
    swe_upper_bound = 3 * np.max(
        np.array(init_cond[1])[common_model_time_start:, :].flatten())

    mb_for_cov = mb_init_raw.T[common_model_time_start:, :].flatten()

    if swe_and_tacc_dist == 'logit':
        swe_for_cov = utils.physical_to_logit(
            np.array(init_cond[1])[common_model_time_start:, :].flatten(),
            0., swe_upper_bound, log_const)  # SWE
    elif swe_and_tacc_dist == 'log':
        swe_for_cov = np.log(
            np.array(init_cond[1])[common_model_time_start:, :].flatten()
            + log_const),  # SWE

    if len(init_cond[3]) > 0:
        alpha_for_cov = utils.physical_to_logit(
            np.array(init_cond[2]).flatten(), 0., 1., 1e-2)  # alpha
        if swe_and_tacc_dist == 'logit':
            tacc_for_cov = utils.physical_to_logit(
                np.array(init_cond[3]).flatten(), 0., tacc_upper_bound,
                log_const)    # tacc
        elif swe_and_tacc_dist == 'log':
            tacc_for_cov = np.log(np.array(init_cond[3]).flatten() + log_const)
        all_for_cov = [mb_for_cov, swe_for_cov, alpha_for_cov, tacc_for_cov]
    else:
        all_for_cov = [mb_for_cov, swe_for_cov]

    # variable covariance
    var_cov = np.cov(all_for_cov)
    # todo: the covariance have to be calclated **by model**
    # todo: what about spatial correlation? (with height) - it's neglected now!
    res_all = np.full((np.array(init_cond[1]).shape[1], n_samples,
                       var_cov.shape[0]), np.nan)
    for h in range(np.array(init_cond[1]).shape[1]):

        mb_mean_for_cov = mb_mean[h]
        if swe_and_tacc_dist == 'logit':
            swe_mean_for_cov = utils.physical_to_logit(
                swe_mean[h], 0., swe_upper_bound, log_const)
        elif swe_and_tacc_dist == 'log':
            swe_mean_for_cov = np.log(swe_mean[h] + log_const)
        if len(init_cond[3]) > 0:
            alpha_mean_for_cov = utils.physical_to_logit(alpha_mean[h], 0., 1,
                                                         1e-2)
            if swe_and_tacc_dist == 'logit':
                tacc_mean_for_cov = utils.physical_to_logit(
                    tacc_mean[h], 0., tacc_upper_bound, log_const)
            elif swe_and_tacc_dist == 'log':
                np.log(tacc_mean[h] + log_const)

            all_mean_for_cov = [mb_mean_for_cov, swe_mean_for_cov,
                                alpha_mean_for_cov, tacc_mean_for_cov]
        else:
            all_mean_for_cov = [mb_mean_for_cov, swe_mean_for_cov]

        res = np.random.default_rng().multivariate_normal(
            mean=all_mean_for_cov, cov=var_cov, size=n_samples)
        res_all[h, :] = res

    # print(mb_init_raw.shape, np.array(init_cond[2]).shape,
    # np.array(init_cond[3]).shape, params[0].shape,
    # params[1].shape, params[2].shape, params[3].shape)

    cov_input_vars = all_for_cov.copy()

    # [mb_init_raw.T[common_model_time_start:, :].flatten(),  # MB
    #              # np.log(np.array(init_cond[1])[common_model_time_start:, :].flatten() + log_const),  # SWE
    #              utils.physical_to_logit(
    #                  np.array(init_cond[1])[common_model_time_start:, :].flatten(), 0.,
    #                  swe_upper_bound,
    #                  log_const),  # SWE
    #              # np.log(np.array(init_cond[2]).flatten),  # alpha
    #              utils.physical_to_logit(np.array(init_cond[2]).flatten(),
    #                                      0., 1., 1e-2),  # alpha
    #              # np.log(np.array(init_cond[3]).flatten() + log_const),
    #              utils.physical_to_logit(np.array(init_cond[3]).flatten(),
    #                                      0., tacc_upper_bound, log_const)]  # tacc

    # remove NaN rows from winter cali
    params = [p[~np.isnan(p).any(axis=1)] for p in params]

    # todo: un-hardcode the "2"
    if len(init_cond[3]) > 0 and len(mb_models) > 1:
        print(len(init_cond[3]))
        cov_input_params_phys = [
            [np.repeat(p[:, j].flatten(), 2 * np.array(init_cond[1]).shape[1])
             for j in range(n_params_per_model[i])]
            for i, p in enumerate(params)]
    else:
        cov_input_params_phys = [
            [np.repeat(p[:, j].flatten(), np.array(init_cond[1]).shape[1]) for
             j in range(n_params_per_model[i])] for i, p in enumerate(params)]

    params_mean_phys = [
        [np.mean(p[:, j]) for j in range(n_params_per_model[i])]
        for i, p in enumerate(params)]

    # change the value of Oerlemans to be positive (for taking the log)
    if 'OerlemansModel' in [m.__name__ for m in mb_models]:
        oerle_index = [m.__name__ for m in mb_models].index('OerlemansModel')
        c0_index = OerlemansModel.cali_params_list.index("c0")
        # 1) correct cov_input_params
        cov_input_params_phys[oerle_index][c0_index] = \
            -cov_input_params_phys[oerle_index][c0_index]
        # 2) correct params_mean_log
        params_mean_phys[oerle_index][c0_index] = \
            -params_mean_phys[oerle_index][c0_index]

    # flatten and TAKE LOG
    cov_input_params_flat = [np.log(val) for sublist in cov_input_params_phys
                             for val in sublist]
    params_mean_log_flat = [np.log(val) for sublist in params_mean_phys
                            for val in sublist]

    #var_cov = np.cov([cov_input_vars] + [
    #                  np.log(np.repeat(params[0][:, 0].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(params[0][:, 1].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(params[1][:, 0].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(params[1][:, 1].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(params[1][:, 2].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(params[2][:, 0].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(params[2][:, 1].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(params[2][:, 2].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(-params[3][:, 0].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),  # neg parameter in Oerlemans
    #                  np.log(np.repeat(params[3][:, 1].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1])),
    #                  np.log(np.repeat(params[3][:, 2].flatten(),
    #                            2 * np.array(init_cond[1]).shape[1]))])  # tacc
    var_cov = np.cov(cov_input_vars + cov_input_params_flat)  # tacc
    res_all = np.full(
        (np.array(init_cond[1]).shape[1], n_samples, var_cov.shape[0]),
        np.nan)
    for h in range(np.array(init_cond[1]).shape[1]):
        #res = np.random.default_rng().multivariate_normal(
        #    mean=[mb_mean[h],
        #          #np.log(swe_mean[h] + log_const),
        #          #np.log(alpha_mean[h], log_const),
        #          utils.physical_to_logit(swe_mean[h], 0., swe_upper_bound, log_const),
        #          utils.physical_to_logit(alpha_mean[h], 0., 1, 1e-2),
        #          #np.log(tacc_mean[h] + log_const),
        #          utils.physical_to_logit(tacc_mean[h], 0.,
        #                                  tacc_upper_bound, log_const),
        #          np.log(np.mean(params[0][:, 0])),
        #          np.log(np.mean(params[0][:, 1])),
        #          np.log(np.mean(params[1][:, 0])),
        #          np.log(np.mean(params[1][:, 1])),
        #          np.log(np.mean(params[1][:, 2])),
        #          np.log(np.mean(params[2][:, 0])),
        #          np.log(np.mean(params[2][:, 1])),
        #          np.log(np.mean(params[2][:, 2])),
        #          np.log(-np.mean(params[3][:, 0])),  # neg parameter in Oerlemans
        #          np.log(np.mean(params[3][:, 1])),
        #          np.log(np.mean(params[3][:, 2])), ], cov=var_cov,
        #    size=n_samples)
        mb_mean_for_cov = mb_mean[h]
        if swe_and_tacc_dist == 'logit':
            swe_mean_for_cov = utils.physical_to_logit(swe_mean[h], 0.,
                                                       swe_upper_bound,
                                                       log_const)
        elif swe_and_tacc_dist == 'log':
            swe_mean_for_cov = np.log(swe_mean[h] + log_const),
        if len(init_cond[3]) > 0:
            alpha_mean_for_cov = utils.physical_to_logit(alpha_mean[h], 0., 1,
                                                         1e-2)
            # np.log(alpha_mean[h] + log_const),
            if swe_and_tacc_dist == 'logit':
                tacc_mean_for_cov = utils.physical_to_logit(tacc_mean[h], 0.,
                                                            tacc_upper_bound,
                                                            log_const)
            elif swe_and_tacc_dist == 'log':
                tacc_mean_for_cov = np.log(tacc_mean[h] + log_const)
            all_mean_for_cov = [mb_mean_for_cov, swe_mean_for_cov,
                                alpha_mean_for_cov, tacc_mean_for_cov] + \
                               params_mean_log_flat
        else:
            all_mean_for_cov = [mb_mean_for_cov, swe_mean_for_cov] + \
                               params_mean_log_flat

        res = np.random.default_rng().multivariate_normal(
            mean=all_mean_for_cov, cov=var_cov,
            #mean=[mb_mean[h],  # np.log(swe_mean[h] + log_const),
            #      # np.log(alpha_mean[h], log_const),
            #      utils.physical_to_logit(swe_mean[h], 0., swe_upper_bound,
            #                              log_const),
            #      utils.physical_to_logit(alpha_mean[h], 0., 1, 1e-2),
            #      # np.log(tacc_mean[h] + log_const),
            #      utils.physical_to_logit(tacc_mean[h], 0., tacc_upper_bound,
            #                              log_const)] +
            #      params_mean_log_flat, cov=var_cov,
            size=n_samples)
        res_all[h, :] = res

    # what was the difference between raw and random again?
    mb_init_random = res_all[..., 0]
    if swe_and_tacc_dist == 'logit':
        swe_init = utils.logit_to_physical(res_all[..., 1], 0.,
                                           swe_upper_bound, log_const)
    elif swe_and_tacc_dist == 'log':
        swe_init = np.exp(res_all[..., 1]) - log_const

    if len(init_cond[3]) > 0:
        alpha_init = utils.logit_to_physical(res_all[..., 2], 0., 1., 1e-2)

        if swe_and_tacc_dist == 'logit':
            tacc_init = utils.logit_to_physical(res_all[..., 3], 0.,
                                                tacc_upper_bound, log_const)
        elif swe_and_tacc_dist == 'log':
            tacc_init = np.exp(res_all[..., 3]) - log_const

        params_init = np.exp(res_all[..., 4:])
    else:
        alpha_init = np.full_like(swe_init, np.nan)
        tacc_init = np.full_like(swe_init, np.nan)
        params_init = np.exp(res_all[..., 2:])
    # params_init[..., 8] = -params_init[..., 8]  # the Oerlemans one
    if 'OerlemansModel' in [m.__name__ for m in mb_models]:
        c0_index_flat = int(n_params_per_model_cs[oerle_index])
        print('c0_index_flat = ', c0_index_flat, ', should be 8')
                # the Oerlemans one
        params_init[..., c0_index_flat] = -params_init[..., c0_index_flat]

    # sort to make elevation dependency consistent
    # todo: taking alpha as a "sort template" might not be the cleverest idea
    sort_ix = np.argsort(alpha_init, axis=1)
    mb_init_random = np.take_along_axis(mb_init_random, sort_ix, axis=1)
    swe_init = np.take_along_axis(swe_init, sort_ix, axis=1)

    # todo: NEW: emergency fix, because height dependency of SWE not existing
    #  => this gives too much noise when calculating T_ACC in predict step
    rand_num = np.random.randn(n_samples)
    swe_init = np.sort(np.clip(rand_num * np.atleast_2d(swe_std).T +
                               np.atleast_2d(swe_mean).T, 0., None), axis=1)

    if len(init_cond[3]) > 0:
        alpha_init = np.take_along_axis(alpha_init, sort_ix, axis=1)
        tacc_init = np.take_along_axis(tacc_init, sort_ix, axis=1)
    params_init = np.take_along_axis(params_init, np.atleast_3d(sort_ix),
                                     axis=1)

    if fl_ids is not None:
        swe_init = swe_init[fl_ids, :]
        alpha_init = alpha_init[fl_ids, :]
        mb_init_random = mb_init_random[fl_ids, :]
        mb_init_raw = mb_init_raw[fl_ids, :]
        tacc_init = tacc_init[fl_ids, :]
        init_cond_all = init_cond[0].isel(fl_id=fl_ids)
    else:
        init_cond_all = init_cond[0].copy()

    # it's not enough to take the correlation of alpha and tacc: later, we
    # take a deterministic relationship
    tacc_init[:] = tacc_from_alpha_brock(alpha_init)

    # theta_cov_log = [np.cov([np.log(params[0][:, 0].flatten()),
    #                         np.log(params[0][:, 1].flatten())]),
    #                 np.cov([np.log(params[1][:, 0].flatten()),
    #                 np.log(params[1][:, 1].flatten()),
    #                 np.log(params[1][:, 2].flatten())]),
    #                 np.cov([np.log(params[2][:, 0].flatten()),
    #                 np.log(params[2][:, 1].flatten()),
    #                 np.log(params[2][:, 2].flatten())]),
    #                 # neg parameter in Oerlemans
    #                 np.cov([np.log(-params[3][:, 0].flatten()),
    #                         np.log(params[3][:, 1].flatten()),
    #                         np.log(params[3][:, 2].flatten())])]
    # theta_mean_log = [np.array([np.log(np.mean(params[0][:, 0])),
    #              np.log(np.mean(params[0][:, 1]))]),
    #              np.array([np.log(np.mean(params[1][:, 0])),
    #              np.log(np.mean(params[1][:, 1])),
    #              np.log(np.mean(params[1][:, 2]))]),
    #              np.array([np.log(np.mean(params[2][:, 0])),
    #              np.log(np.mean(params[2][:, 1])),
    #              np.log(np.mean(params[2][:, 2]))]),
    #              # neg parameter in Oerlemans
    #              np.array([np.log(-np.mean(params[3][:, 0])),
    #              np.log(np.mean(params[3][:, 1])),
    #              np.log(np.mean(params[3][:, 2]))])]

    if 'OerlemansModel' in [m.__name__ for m in mb_models]:
        params_changed = params.copy()
        params_changed[oerle_index][:, c0_index] = \
            -params_changed[oerle_index][:, c0_index]
        theta_cov_log = [np.cov(np.log(p.T)) for p in params_changed]
    else:
        theta_cov_log = [np.cov(np.log(p.T)) for p in params]

    if 'OerlemansModel' in [m.__name__ for m in mb_models]:
        theta_mean_log = [np.log([np.mean(p, axis=0)]) for p in params_changed]
    else:
        theta_mean_log = [np.log([np.mean(p, axis=0)]) for p in params]

    if 'OerlemansModel' in [m.__name__ for m in mb_models]:
        params_init[..., c0_index_flat] = -params_init[..., c0_index_flat]

    params_prior = [params_init[0, n_samples_per_model_cs[i]:
                                   n_samples_per_model_cs[i + 1],
                    n_params_per_model_cs[i]: n_params_per_model_cs[i + 1]]
                    for i in range(len(mb_models))]
    return init_cond_all, mb_init_raw, mb_init_random, swe_init, alpha_init, \
           tacc_init, theta_mean_log, theta_cov_log, params_prior


# noinspection PyUnresolvedReferences,PyUnresolvedReferences
def run_aepf(gid: str, mb_models: Optional[list] = None,
             stations: Optional[list] = None,
             generate_params: Optional[str] = None,
             param_fit: Optional[str] = 'lognormal',
             unmeasured_period_param_dict: Optional[dict] = None,
             prior_param_dict: Optional[dict] = None,
             evolve_params: Optional[bool] = True,
             update: Optional[bool] = True,
             return_params: Optional[bool] = False,
             param_method: Optional[str] = 'memory',
             make_init_cond: Optional[bool] = True,
             change_memory_mean: Optional[bool] = True,
             qhisto_by_model: Optional[bool] = False,
             limit_to_camera_elevations: Optional[bool] = False,
             reset_albedo_and_swe_at_obs_init: Optional[bool] = False,
             pdata: Optional[pd.DataFrame] = None,
             return_probs: Optional[bool] = False,
             crps_ice_only: Optional[bool] = False,
             use_tgrad_uncertainty: Optional[bool] = True,
             use_pgrad_uncertainty: Optional[bool] = True,
             assimilate_albedo: Optional[bool] = False,
             assimilate_fsca: Optional[bool] = False,
             detrend_params: Optional[bool] = False,
             init_cond_to: Optional[pd.Timestamp] = None) -> tuple:
    """
    Run the augmented ensemble particle filter.

    Parameters
    ----------
    gid: str
        Glacier id to be processed.
    mb_models: list of
    `py:class:crampon.core.models.massbalance.DailyMassBalanceModelWithSnow`
        The mass balance models top be used for the filter. Default: None
        (use those listed in cfg.MASSBALANCE_MODELS).
    stations: list or None
        List of int with stations numbers to process, or None (process all).
    generate_params: str or None
        Possible options: 'past' (generate parameters from past
        calibration), 'gabbi' (take Jeannette Gabbi's parameter priors from
        her 2014 paper), 'past_mean' (take the mean from the past
        calibration'), 'past_gabbi' (take the mean from Jeanette Gabbi's
        priors). Default: None ( equals to "past").
    param_fit: str, optional
        Which function shall be fitted to the parameters from past
        calibration.
    unmeasured_period_param_dict: dict, optional
        Explicit parameters for the unmeasured periods (autumn and early
        summer) handed over for testing.
    prior_param_dict:
        Explicit parameters for the prior parameter distributions handed over
        for testing.
    evolve_params: bool
        Whether to evolve the parameters over time, either using the
        diversification by Liu et al. or the memory equation. Default: True
        (only switch off for experiments)
        # todo the method is hard-coded within the function at the moment
    update: bool
        Whether to perform the update step of the particle filter. Default:
        True (only switch off for experiments).
    return_params: bool
        Whether to write out the parameters that the particle filter has
        found. Default: False.
    param_method: str
        How parameters shall be diversified. Either "liu" or "memory".
        Default: 'memory'.
    make_init_cond: bool
        Generate initial conditions with a spinup run during the mass budget
        year. Default: True.
    change_memory_mean: bool
        When using the "memory" diversification method, this keyword
        determines whether the mean should also be changed back towards the
        prior parameter distribution (experiment setting it to "False" when
        you believe the prior mean is not trustworthy). Default: True.
    qhisto_by_model: bool
        Whether the quantile histogram shall be made by model. Default:
        False (make one quantile histogram for all).
    limit_to_camera_elevations: bool, optional
        Whether the calculation domain shall be limited to the camera
        elevations only. This makes the calculation way faster, but only makes
        sense if only camera data are assimilated. Default: False.
    reset_albedo_and_swe_at_obs_init: bool, optional
        Whether albedo and snow water equivalent on the glacier shall be reset
        when the first camera observation is made (the info of the first camera
        observation is: no SWE at the cam location, and the albedo should be
        the one of ice). This option is recommended for testing only, since it
        sets hard thresholds. Default: False.
    pdata: pd.Dataframe, optional
        Point observations of glacier mass balance as 'intermediate readings',
        in particular so select potential initial conditions at the beginning
        of the assimilation period.
    return_probs: bool, optional
        Whether to return the model probabilities of the particle filter.
        Default: False.
    crps_ice_only: bool, optional
        Whether to calculate the CRPS on days without snow cover only. Default:
        False.
    use_tgrad_uncertainty: bool, optional
        Include the temperature gradient uncertainty as well. Default: True.
    use_pgrad_uncertainty: bool, optional
        Include the precipitation gradient uncertainty as well. Default: True.
    assimilate_albedo: bool, optional
        Whether to assimilate albedo observations. Default: False.
    assimilate_fsca: bool, optional
        Whether to assimilate observations of snow cover on the glacier
        (fraction of snow-covered area). Default: False.
    detrend_params: bool, optional
        Whether to detrend model prior parameters (paramaters might have a
        temporal trend in the past, since we do not account for glacier
        dynamics).
    init_cond_to: pd.Timestamp, optional
        Until when the initial conditions shall be calculated without
        assimilation. Default: None (determine from first camera observation
        in the time series).


    Returns
    -------
    tuple
    """

    # a check
    if return_params is True and return_probs is True:
        raise ValueError("Can't return both parameters and model probability.")

    # how many particles
    n_particles = cfg.PARAMS['n_particles']
    n_phys_vars = cfg.PARAMS['n_phys_vars']
    n_aug_vars = cfg.PARAMS['n_aug_vars']
    # indices state: 0= MB, 1=alpha, 2=m, 3=swe, 4:tacc, 5:=params
    mod_ix = cfg.PARAMS['mod_ix']
    mb_ix = cfg.PARAMS['mb_ix']
    alpha_ix = cfg.PARAMS['alpha_ix']
    swe_ix = cfg.PARAMS['swe_ix']
    tacc_ix = cfg.PARAMS['tacc_ix']
    obs_init_mb_ix = cfg.PARAMS['obs_init_mb_ix']
    theta_start_ix = cfg.PARAMS['theta_start_ix']
    param_prior_distshape = cfg.PARAMS['param_prior_distshape']  # param_fit
    param_prior_std_scalefactor = cfg.PARAMS['param_prior_std_scalefactor']
    phi = cfg.PARAMS['phi']
    gamma = cfg.PARAMS['gamma']  # 0.05
    model_error_mean = cfg.PARAMS['model_error_mean']  # 0.
    model_error_std = cfg.PARAMS['model_error_std']  # 0.
    colors = cfg.PARAMS['colors']  # ["b", "g", "c", "m"]
    tacc_ice = cfg.PARAMS['tacc_ice']  # 4100.  # random value over 15 years
    theta_memory = cfg.PARAMS['theta_memory']  # 0.9 model parameter memory
    obs_std_scale_fac = cfg.PARAMS['obs_std_scale_fac']  # 1.0
    sis_sigma = cfg.PARAMS['sis_sigma']  # 15. - try bigger/smaller STD for SIS
    ipot_sigma = cfg.PARAMS['ipot_sigma']

    min_std_alpha = cfg.PARAMS['min_std_alpha']  # 0.0 # 0.05
    min_std_swe = cfg.PARAMS['min_std_swe']  # 0.0  # 0.025  # m w.e.
    min_std_tacc = cfg.PARAMS['min_std_tacc']  # 0.0  # 10.0

    fixed_obs_std = cfg.PARAMS['fixed_obs_std']  # None  # m w.e.
    max_nan_thresh = cfg.PARAMS['max_nan_thresh']  # 0.2

    # to keep results as reproducible as possible
    seed = 0

    # some diagnostics
    all_mprobs = []
    all_mparts = []
    all_dspans = []
    glacier_names = []
    ancest_ix = []
    ancest_all = []
    mb_anc_ptcls = []
    mb_anc_ptcls_after = []
    alpha_anc_ptcls = []
    alpha_anc_ptcls_after = []

    # todo: change this and make it flexible
    # we stop one day earlier than the field date - anyway no obs anymore
    if gid == 'RGI50-11.B4312n-1':
        run_end_date = '2019-09-11'  # '2019-08-11'#
    elif gid == 'RGI50-11.B5616n-1':
        run_end_date = '2019-09-16'  # '2019-09-17'#'2019-09-16'#
    elif gid == 'RGI50-11.A55F03':
        run_end_date = '2019-09-29'  # '2019-09-18'  # '2019-09-30'#''#
    elif gid == 'RGI50-11.B5616n-test':
        run_end_date = '2019-09-16'  # '2019-09-17'#
    elif gid == 'RGI50-11.A10G05':
        run_end_date = '2019-09-20'
        # run_end_date = '2020-09-14'
    elif gid == 'RGI50-11.A51D10':
        run_end_date = '2019-09-30'
    else:
        raise ValueError('In this provisional version of code, you need to '
                         'specify an end date for the run of your glacier.')

    if mb_models is None:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]
    modelnames = [m.__name__ for m in mb_models]
    n_params_per_model = [len(m.cali_params_list) for m in mb_models]
    log.info('MB MODELS: ' + ' '.join([m.__name__ for m in mb_models]))

    init_particles_per_model = int(n_particles / len(mb_models))

    # the gdir do we want to process
    gdir = utils.GlacierDirectory(gid)
    fl_h, fl_w = gdir.get_inversion_flowline_hw()

    # get the observations
    # todo: change back if we want multi-stations
    if stations is None:
        try:
            stations = id_to_station[gdir.rgi_id]
        except KeyError:
            log.warning('Glacier does not have camera observations!')
            update = False
            stations = None

    if stations is not None:
        station_hgts = [station_to_height[s] for s in stations]
        obs_merge, obs_merge_cs = prepare_observations(gdir, stations)
        if fixed_obs_std is not None:
            log.info('Setting OBS STD to fixed value!')
            obs_merge['swe_std'][:] = fixed_obs_std

        if init_cond_to is None:
            first_obs_date = obs_merge_cs.date[
                np.argmin(np.isnan(obs_merge_cs).values, axis=1)].values
            first_obs_date = np.array([
                pd.Timestamp(d) for d in first_obs_date])

        # todo: minus to revert the minus from the function! This is shit!
        obs_std_we_default = - utils.obs_operator_dh_stake_to_mwe(
            cfg.PARAMS['dh_obs_manual_std_default'])
        # todo: reconsider: could camera end up being on the wrong flowline?
        s_index = np.argmin(
            np.abs((fl_h - np.atleast_2d(station_hgts).T)), axis=1)
        print('s_index: ', s_index)
        # limit height to where we observe with camera
    else:
        s_index = []
        obs_merge_cs = None
        obs_merge = None

    # calculate only at camera elevations (for testing)
    if limit_to_camera_elevations is True:
        h = fl_h[s_index]
        w = fl_w[s_index]
        fl_ids = s_index
        spatial_cam_ix = np.arange(len(s_index))
    else:
        h = fl_h
        w = fl_w
        fl_ids = np.arange(len(fl_h))
        spatial_cam_ix = s_index

    if (assimilate_albedo is True) and \
            not os.path.isfile(gdir.get_filepath('alpha_proc')):
        alpha_ds = xr.open_dataset(gdir.get_filepath('alpha_ens'))
        cm = xr.open_dataset(gdir.get_filepath('sat_images')).cmask

        alpha_ds = alpha_ds.where(cm == 0, np.nan)
        ol = gdir.get_filepath('outlines')
        dem_path = gdir.get_filepath('dem')
        dem = xr.open_rasterio(dem_path).isel(band=0)
        dem.attrs['pyproj_srs'] = dem.attrs['crs']
        alpha_ds.attrs['pyproj_srs'] = dem.pyproj_srs
        dem = alpha_ds.salem.transform(dem.to_dataset(name='height'))
        dem_roi_template = dem.salem.roi(shape=ol)
        dem_roi = dem_roi_template.copy(deep=True)
        dem_roi['broadband'] = alpha_ds.broadband.values  # set coord
        dem_roi['time'] = alpha_ds.time.values  # set coord
        dem_roi['alpha'] = (('broadband', 'time', 'y', 'x'),
                            alpha_ds.albedo.data)
        dem_roi = dem_roi.salem.roi(shape=ol)
        bins = np.arange(np.nanmin(h), np.nanmax(h)+10., 10.)
        gb = dem_roi.groupby_bins(dem_roi.height, bins=bins)
        alpha_obs = np.full((len(alpha_ds.time.values), len(h), int(np.max(
            [g.alpha.values.size / len(alpha_ds.time.values) for _, g in
             list(gb)]))), np.nan)
        xs = [np.mean([b.left, b.right]) for b, _ in list(gb)]
        # todo: do this by polygon of the flowline catchments!
        for j, ih in enumerate(h):
            group_index = np.argmin(np.abs(ih - xs))
            group = list(gb)[group_index][1]
            pix_sum = group.count(dim='stacked_y_x').alpha\
                .mean(dim='broadband')
            valid_ratio = group.count(dim='stacked_y_x').alpha\
                              .mean(dim='broadband') / group.count().height
            valid_ix = np.where((valid_ratio >= 1 - max_nan_thresh)
                                & (pix_sum >= 10))[0]
            alpha_obs[valid_ix, j, :np.nanmean(group.alpha.values, axis=0).shape[1]] = np.nanmean(group.alpha.values, axis=0)[valid_ix, :]

        alpha_proc = xr.DataArray(
            alpha_obs, coords={
                'time': dem_roi.time.values, 'fl_id': np.arange(len(h)),
                'samples': np.arange(int(np.max([g.alpha.values.size /
                                                 len(alpha_ds.time.values)
                                                 for _, g in list(gb)])))},
            name='albedo', dims=['time', 'fl_id', 'samples'])

        # make again statistics of the overall cloud cover
        cm_roi = cm.salem.roi(shape=ol)
        cloud_cov_ratio = np.nansum(cm_roi.values, axis=(1, 2)) / np.sum(
            ~np.isnan(cm_roi.values), axis=(1, 2))
        # be even more strict: allow only 10% cloud cover
        alpha_proc = alpha_proc.isel(time=np.where(cloud_cov_ratio <= 0.1)[0])
        alpha_proc.to_netcdf(gdir.get_filepath('alpha_proc'))

        hrange = alpha_proc.fl_id.values
        alb = alpha_proc.mean(dim='samples')\
            .min(dim='time', skipna=True).values
        mask = ~np.isnan(alb)

        # take median of observed summer albedo as background albedo for ice
        # rolling over 6 fl_ids is just visually nice on Findelen
        a_under_obs_at_fl = np.clip(alpha_proc.isel(
            time=np.where([
                m in [7, 8, 9] for m in alpha_proc.time.dt.month])[0])
                                    .median(dim=['samples', 'time'],
                                            skipna=True)
                                    .rolling(fl_id=6, center=True)
                                    .mean(skipna=True), 0.1,
                                    cfg.PARAMS['ice_albedo_default'])
        # extrapolate missing values form elevations with too few pixels
        sort_ix = np.argsort(h[~np.isnan(a_under_obs_at_fl)])
        a_under_obs_at_fl = np.interp(
            h, h[~np.isnan(a_under_obs_at_fl)][sort_ix],
            a_under_obs_at_fl[~np.isnan(a_under_obs_at_fl)][sort_ix])
    else:
        alpha_ds = None
        if assimilate_albedo is True:
            alpha_proc = xr.open_dataset(gdir.get_filepath('alpha_proc'))
        else:
            alpha_proc = None
        a_under_obs_at_fl = np.full_like(h, cfg.PARAMS['ice_albedo_default'])

    if (assimilate_fsca is True) and \
            not os.path.isfile(gdir.get_filepath('swe_proc')):
        swe_ds = xr.open_dataset(gdir.get_filepath('snowprob'))
        cm = xr.open_dataset(gdir.get_filepath('sat_images')).cmask
        swe_ds = swe_ds.where(cm == 0, np.nan)
        ol = gpd.read_file(gdir.get_filepath('outlines'))
        dem_path = gdir.get_filepath('dem')

        # normalize percentage of ice & snow, in case there is also cloud prob.
        swe_ds = xr.DataArray(swe_ds.snow / (swe_ds.snow + swe_ds.ice))\
            .to_dataset(name='snow')

        dem = xr.open_rasterio(dem_path).isel(band=0)
        dem.attrs['pyproj_srs'] = dem.attrs['crs']
        for var in swe_ds.data_vars:
            swe_ds[var].attrs['pyproj_srs'] = ol.crs.to_proj4()
        swe_ds.attrs['pyproj_srs'] = ol.crs.to_proj4()
        dem = swe_ds.salem.transform(dem.to_dataset(name='height'))
        dem_roi_template = dem.salem.roi(shape=ol)
        dem_roi = dem_roi_template.copy(deep=True)
        dem_roi['time'] = swe_ds.time.values  # set coord
        dem_roi['swe'] = (('time', 'y', 'x'), swe_ds.snow.data)
        dem_roi = dem_roi.salem.roi(shape=ol)
        bins = np.arange(np.nanmin(h), np.nanmax(h)+10., 10.)
        gb = dem_roi.groupby_bins(dem_roi.height, bins=bins)
        swe_obs = np.full((len(swe_ds.time.values), len(h), int(np.max(
            [g.swe.values.size / len(swe_ds.time.values) for _, g in
             list(gb)]))), np.nan)
        xs = [np.mean([b.left, b.right]) for b, _ in list(gb)]
        # todo: do this by polygon of the flowline catchments!
        for j, ih in enumerate(h):
            group_index = np.argmin(np.abs(ih - xs))
            group = list(gb)[group_index][1]
            pix_sum = group.count(dim='stacked_y_x').swe
            valid_ratio = group.count(
                dim='stacked_y_x').swe / group.count().height
            valid_ix = np.where(
                (valid_ratio >= 1 - max_nan_thresh) & (pix_sum >= 10))[0]
            swe_obs[valid_ix, j,
            :group.swe.values.shape[1]] = group.swe.values[valid_ix, :]

        swe_proc = xr.DataArray(
            swe_obs, coords={
                'time': dem_roi.time.values, 'fl_id': np.arange(len(h)),
                'samples': np.arange(int(np.max([g.swe.values.size /
                                                 len(swe_ds.time.values)
                                                 for _, g in list(gb)])))},
            name='swe_prob', dims=['time', 'fl_id', 'samples'])

        # make again statistics of the overall cloud cover
        cm_roi = cm.salem.roi(shape=ol)
        cloud_cov_ratio = np.nansum(cm_roi.values, axis=(1, 2)) / np.sum(
            ~np.isnan(cm_roi.values), axis=(1, 2))
        # be even more strict: allow only 10% cloud cover
        swe_proc = swe_proc.isel(time=np.where(cloud_cov_ratio <= 0.1)[0])
        swe_proc.to_netcdf(gdir.get_filepath('swe_proc'))

    else:
        swe_ds = None
        swe_proc = xr.open_dataset(gdir.get_filepath('swe_proc'))

    gmeteo = climate.GlacierMeteo(
        gdir, randomize=True, n_random_samples=n_particles, heights=fl_h,
        use_tgrad_uncertainty=use_tgrad_uncertainty,
        use_pgrad_uncertainty=use_pgrad_uncertainty)
    # attention: This doesn't change sis_sigma in gmeteo.meteo!!!
    if sis_sigma is not None:
        gmeteo.sis_sigma = np.ones_like(gmeteo.sis_sigma) * sis_sigma

    # date when we have to start the calculate last year
    if obs_merge_cs is not None:
        begin_mbyear = utils.get_begin_last_flexyear(
            pd.Timestamp(obs_merge_cs.date.values[0]))
    else:
        begin_mbyear = utils.get_begin_last_flexyear(
            pd.Timestamp(run_end_date))
    if pdata is not None:
        # earliest OBS is the minimum of all date0s
        autumn_obs_mindate = min(pdata.loc[stations].date0.values)
        # begin of mb year is min of the value in params.cfg and earliest OBS
        autumn_obs_begin_mbyear_min = pd.Timestamp(
            min(autumn_obs_mindate, np.datetime64(begin_mbyear)))
    else:
        # otherwise: both is at the values of params.cfg
        autumn_obs_mindate = begin_mbyear
        autumn_obs_begin_mbyear_min = begin_mbyear

    # get start values for alpha and SWE
    if make_init_cond is True:
        start_at_mb_year_begin = True
        if start_at_mb_year_begin is True:
            if init_cond_to is None:
                init_cond_to = pd.Timestamp(
                    min(first_obs_date)) - pd.Timedelta(days=1)
            init_cond_from = min(utils.get_begin_last_flexyear(init_cond_to),
                                 autumn_obs_begin_mbyear_min)

        else:
            if init_cond_to is None:
                init_cond_to = pd.Timestamp(
                    min(first_obs_date)) - pd.Timedelta(days=1)
            init_cond_from = autumn_obs_begin_mbyear_min
        if init_cond_to is None:
            init_cond_to = pd.Timestamp(min(first_obs_date)) - \
                           pd.Timedelta(days=1)
        else:
            pass
        print('Getting initial conditions from ', init_cond_from,
              ' to ', init_cond_to)
        mb_init_field, mb_init_homo, mb_init, swe_init, alpha_init, \
        tacc_init, theta_priors_means, theta_priors_cov, theta_priors  \
            = get_initial_conditions(
                gdir, init_cond_to, n_particles,
                begin_mbyear=init_cond_from,
                param_dict=unmeasured_period_param_dict, fl_ids=fl_ids,
                min_std_alpha=min_std_alpha, min_std_swe=min_std_swe,
                min_std_tacc=min_std_tacc, alpha_underlying=a_under_obs_at_fl,
                param_prior_distshape=param_prior_distshape,
                param_prior_std_scalefactor=param_prior_std_scalefactor,
                mb_models=mb_models, generate_params=generate_params,
                detrend_params=detrend_params, seed=seed)
        log.info('Initial conditions from ' + str(init_cond_to) +
              ' to ' + str(init_cond_to))
        print(np.nanmin(swe_init), np.nanmin(alpha_init),
              np.nanmin(tacc_init),
              np.mean(np.average(mb_init, weights=w, axis=0)))
    else:
        swe_init = np.zeros((len(fl_ids), n_particles))
        alpha_init = np.ones((len(fl_ids), n_particles)) * cfg.PARAMS[
            'ice_albedo_default']
        mb_init = np.zeros((len(fl_ids), n_particles))
        tacc_init = tacc_from_alpha_brock(alpha_init)
        # todo: this is changed to make alpha vary
        # tacc_init[swe_init == 0.] = tacc_ice
        tacc_init[swe_init == 0.] = tacc_from_alpha_brock(alpha_init)[
            swe_init == 0.]
        if init_cond_to is None:
            # still needed
            init_cond_to = pd.Timestamp(min(first_obs_date)) \
                           - pd.Timedelta(days=1)

    if prior_param_dict is not None:
        # overwrite
        theta_priors = [
            np.atleast_2d(np.array([
                v for k, v in prior_param_dict.items() if m in k]))
            for m in [m.__name__ for m in mb_models]]
        theta_priors = [np.repeat(t, init_particles_per_model, axis=0)
                        for t in theta_priors]
        theta_priors_cov = [np.ones_like(t) for t in theta_priors]
        theta_priors_means = theta_priors.copy()

    # try to better constrain the variability of alpha
    alpha_init_mean = np.mean(alpha_init, axis=1)
    if alpha_ds is not None:
        dem_roi = dem_roi_template.copy(deep=True)
        dem_roi['broadband'] = alpha_ds.broadband.values  # set coord
        dem_roi['time'] = alpha_ds.time.values
        dem_roi['alpha'] = (('broadband', 'time', 'y', 'x'),
                            alpha_ds.albedo.data)
        bins = np.arange(np.nanmin(h), np.nanmax(h), 10.)
        gb = dem_roi.groupby_bins(dem_roi.height, bins=bins)
        alpha_obs_std = np.full(len(h), np.nan)
        xs = [np.mean([b.left, b.right]) for b, _ in list(gb)]
        for j, ih in enumerate(h):
            group_index = np.argmin(np.abs(ih - xs))
            group = list(gb)[group_index][1]
            alpha_obs_std[j] = np.nanstd(group.alpha.values.flatten())

    if pdata is not None:
        # select initial conditions according to stake measurements
        # brute force
        if gdir.rgi_id == 'RGI50-11.B4312n-1':
            firstobs_heights = np.array([2235., 2589.])
        elif gdir.rgi_id == 'RGI50-11.B5616n-1':
            firstobs_heights = np.array([2564., 3021.])
        elif gdir.rgi_id == 'RGI50-11.A55F03':
            firstobs_heights = np.array([2681.])
        else:
            raise ValueError('Glacier ID not recognized for pdata.')
        firstobs_fl_ids = np.argmin(
            np.abs((fl_h - np.atleast_2d(firstobs_heights).T)), axis=1)
        print(pdata.z.values, firstobs_heights)
        # todo: we don't want two dates, they should be the same (hopefully)
        pdata_init = pdata.loc[pdata['z'].isin(firstobs_heights)]
        # this is to get the rows in correct order
        pdata_init['fl_id'] = np.argmin(
            np.abs((fl_h - np.atleast_2d(pdata_init['z']).T)), axis=1)
        pdata_init = pdata_init.sort_values('fl_id')
        firstobs_fl_ids = sorted(firstobs_fl_ids)
        print('PDATA init: ', pdata_init)

        pdata_init_swe = pdata_init['swe_bp'].values
        pdata_init_otype = pdata_init['otype'].values
        pdata_init_date = pdata_init['date_p'].values[0]  # 0 bcz it should
        pdata_init_unc = pdata_init['bp_unc'].values
        # be the same
        # select runs close to the stake observations in stake OBS period
        mb_init_all_cs_pdata = mb_init_field.stack(
            ens=['model', 'member']).sel(
            time=slice(autumn_obs_mindate, pdata_init_date)).cumsum(
            dim='time').isel(time=-1).MB.values
        mb_init_all_cs_first_obs_init = mb_init_field.stack(ens=['model',
                                                            'member']).sel(
            time=slice(autumn_obs_mindate, min(first_obs_date))).cumsum(
            dim='time').isel(time=-1).MB.values
        # minimum (before JAN) must be subtracted if there is snow
        mb_init_all_cs_first_obs_init_min = mb_init_field.stack(
            ens=['model', 'member']).sel(
            time=slice(autumn_obs_mindate,
                       pd.Timestamp('{}-12-31'.format(
                           pd.Timestamp(autumn_obs_mindate).year)))).cumsum(
            dim='time').min(dim='time')
        mb_init_all_cs_first_obs_init_argmin = \
            np.argmin(np.average(
                mb_init_field.stack(ens=['model', 'member']).sel(
                    time=slice(autumn_obs_mindate,
                               pd.Timestamp('{}-12-31'.format(
                                   pd.Timestamp(autumn_obs_mindate).year)))
                ).cumsum(dim='time').MB.values, weights=w, axis=0), axis=0)
        run_sel = []
        run_sel_diff = []
        # if np.isnan(pdata_init_unc):
        #    bp_unc = 0.05  # reading uncertainty for stakes
        # else:
        #    bp_unc = pdata_init_unc
        for foix, sfl in enumerate(firstobs_fl_ids):
            if np.isnan(pdata_init_unc[foix]):
                bp_unc = 0.05  # reading uncertainty for stakes
            else:
                bp_unc = pdata_init_unc[foix]
            point_mb = pdata_init_swe[foix]

            # if pdata_init_otype[foix] == 'ice':

            np.random.seed(seed)
            gauss_range = np.random.normal(point_mb, bp_unc, size=n_particles)
            gauss_range = np.clip(gauss_range, point_mb - bp_unc,
                                  point_mb + bp_unc)

            for unc_bp in gauss_range:
                run_sel.append(
                    np.argmin(np.abs(mb_init_all_cs_pdata[sfl, :] - unc_bp)))
                run_sel_diff.append(
                    mb_init_all_cs_pdata[
                        sfl, np.argmin(np.abs(mb_init_all_cs_pdata[sfl, :]
                                              - unc_bp))] - unc_bp)
            if (pdata_init_otype[foix] == 'snow') \
                    and (gdir.rgi_id == 'RGI50-11.A55F03') \
                    and (pd.Timestamp(pdata_init_date).year == 2019):
                # snow
                # find additionally runs closest to zero at first camera
                # observation!!!- > this works at Plaine Morte in this year,
                # because the snow-ice transition is right here
                gauss_range = np.random.normal(0., bp_unc,
                                               size=n_particles)
                gauss_range = np.clip(gauss_range, - bp_unc, bp_unc)
                for unc_bp in gauss_range:
                    model_swe = (mb_init_all_cs_first_obs_init -
                                 mb_init_all_cs_first_obs_init_min).MB.values
                    member_no = np.argmin(np.abs(model_swe[sfl, :] - unc_bp))
                    run_sel.append(member_no)
                    run_sel_diff.append(mb_init_all_cs_first_obs_init[
                                            sfl, member_no])
        if len(run_sel) == 0:
            raise ValueError

        np.random.seed(seed)
        rand_choice_ix = np.random.choice(range(len(run_sel)),
                                          n_particles)
        # mb_init = mb_init_homo[:, np.array(run_sel)[rand_choice_ix]]
        mb_init = mb_init_all_cs_first_obs_init[
                  :, np.array(run_sel)[rand_choice_ix]]
        mb_init -= np.array(run_sel_diff)[rand_choice_ix]

        # mb_init = mb_init_homo[:, np.random.choice(run_sel, n_particles)]
        if first_obs_date is not None:
            print(mb_init, 'AVERAGE: ', np.mean(
                np.average(mb_init, weights=w, axis=0)),
                  pd.Timestamp(min(first_obs_date)))

    # get indices of those runs
    # select those runs in cumulative MB beginning at OCT-1
    ipot_per_fl = gdir.read_pickle('ipot_per_flowline')
    ipot_per_fl = np.array([i for sub in ipot_per_fl for i in sub])
    # ipot_year = ipot_per_fl[s_index, :]
    ipot_year = ipot_per_fl[fl_ids, :]
    # make leap year compatible
    ipot_year = np.hstack([ipot_year, np.atleast_2d(ipot_year[:, -1]).T])

    # time span
    date_span = pd.date_range(init_cond_to + pd.Timedelta(days=1),
                              run_end_date)
    print('DATE SPAN: ', date_span[0], date_span[-1])

    # get prior parameter distributions
    # todo: it's confusing that the params are returned in physical domain,
    #  but means and cov in log domain
    # theta_priors, theta_priors_means, theta_priors_cov = \
    #    prepare_prior_parameters(gdir, mb_models, init_particles_per_model,
    #                             param_prior_distshape,
    #                             param_prior_std_scalefactor,
    #                             generate_params, param_dict=prior_param_dict,
    #                             seed=seed)
    # get snow redist fac
    try:
        snowredist_ds = xr.open_dataset(gdir.get_filepath('snow_redist'))
    except FileNotFoundError:
        snowredist_ds = None
        log.info('No Snow redistribution factor found.')

    aepf = AugmentedEnsembleParticleFilter(
        mb_models, n_particles, spatial_dims=(len(h),),
        n_phys_vars=n_phys_vars, n_aug_vars=n_aug_vars, name=gdir.plot_name)

    # init particles
    x_0 = np.full((len(h), n_particles, n_phys_vars+n_aug_vars), np.nan)
    x_0[:, :, mb_ix] = 0.  # mass balance
    x_0[:, :, alpha_ix] = alpha_init
    x_0[:, :, mod_ix] = np.repeat(np.arange(len(mb_models)),
                                  n_particles/len(mb_models))
    x_0[:, :, tacc_ix] = tacc_init  # accum. max temp.
    x_0[:, :, swe_ix] = swe_init
    x_0[:, :, obs_init_mb_ix] = 0.

    # assign params per model (can we save the model loop?)
    for mix in range(len(mb_models)):
        theta_m = theta_priors[mix]
        n_theta = theta_m.shape[1]
        # make parameters fit
        theta_m_reshape = np.tile(theta_m, (x_0.shape[0], 1))
        x_0[:, :, theta_start_ix:theta_start_ix+n_theta][
            x_0[:, :, mod_ix] == mix] = theta_m_reshape

    aepf.particles = x_0
    aepf._M_t_all = np.array([np.max(aepf.model_weights_log[i]) for i in
                              aepf.model_range])
    aepf._S_t_all = np.array([np.sum(np.exp(aepf.model_weights_log[i] -
                                            aepf.M_t_all[i]))
                              for i in aepf.model_range])

    mb_models_inst = [m(gdir, bias=0., heights_widths=(h, w)) for m in
                      mb_models]

    """          
    # separate alpha filter
        an_particles = 1000
    alpha_models = ['Brock']
    an_phys_vars = 1
    an_aug_vars = 3
    alpha_aepf = AugmentedEnsembleParticleFilter(alpha_models, an_particles,
        spatial_dims=(len(h),), n_phys_vars=an_phys_vars,
        n_aug_vars=an_aug_vars, name=gdir.plot_name)
    # init particles
    aalpha_ix = 1
    ax_0 = np.full((len(h), an_particles, an_phys_vars + an_aug_vars), np.nan)
    ax_0[:, :, alpha_ix] = alpha_init
    ax_0[:, :, mod_ix] = np.repeat(np.arange(len(alpha_models)),
                                   an_particles / len(alpha_models))
    # assign params per model (can we save the model loop?)
    for amix in range(len(alpha_models)):
        atheta_m = atheta_priors[amix]
        an_theta = atheta_m.shape[1]
        # make parameters fit
        atheta_m_reshape = np.tile(atheta_m, (ax_0.shape[0], 1))
        ax_0[:, :, atheta_start_ix:atheta_start_ix + an_theta][
            ax_0[:, :, amod_ix] == amix] = atheta_m_reshape
        alpha_aepf.particles = ax_0
        alpha_aepf._M_t_all = np.array(
            [np.max(alpha_aepf.model_weights_log[i]) for i in
             alpha_aepf.model_range])
        alpha_aepf._S_t_all = np.array([np.sum(
            np.exp(alpha_aepf.model_weights_log[i] - alpha_aepf.M_t_all[i]))
                                        for i in alpha_aepf.model_range])
    """

    sis_scale_fac = xr.open_dataarray(gdir.get_filepath(
        'sis_scale_factor')).values
    # make leap year compatible
    sis_scale_fac = np.hstack([sis_scale_fac,
                               np.atleast_2d(sis_scale_fac[:, -1]).T])

    if plot_dict['plot_mb_and_ensemble_evolution_overview'] is True:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20, 15),
                                                 sharex='all')
    if plot_dict['plot_parameter_evolution'] is True:
        fig3, (ax6, ax7, ax8, ax9) = plt.subplots(4, figsize=(20, 15),
                                                  sharex='all')
        ax6twin = ax6.twinx()
        ax7twin = ax7.twinx()
        ax8twin = ax8.twinx()
        ax9twin = ax9.twinx()
    if plot_dict['plot_parameter_evolution_ratio'] is True:
        fig4, (ax10, ax11, ax12, ax13) = plt.subplots(4, figsize=(20, 15),
                                                      sharex='all')
    if plot_dict['plot_median_absolute_departure'] is True:
        fig2 = plt.figure(figsize=(15, 7))
        ax5 = fig2.add_subplot(111)

    # list of departures (for plotting the mean at the end)
    dep_list = []
    crps_list = []
    crps_1_list = []
    crps_2_list = []
    crps_3_list = []
    mprob_list = []
    mpart_list = []
    crps_by_model_old = []
    crps_by_model_new = []
    crps_by_model_dates = []
    run_example_list = []
    np.random.seed(seed)
    if pdata is None:
        rand_45 = np.random.choice(range(n_particles), 45)
    else:
        rand_45 = np.random.choice(range(n_particles),
                                   mb_init_all_cs_first_obs_init.shape[1])

    # obs_shape = np.atleast_2d(obs_merge_cs.isel(date=0).values).shape
    # todo: this might be saved if we check for the obs first, before the we
    #  make the prediction
    # mb_anal_before = np.full(obs_shape, 0.)
    mb_anal_before_all = np.zeros((len(h), n_particles))
    # mb_anal_std_before = np.full(obs_shape, 0.)

    param_dates_list = []
    params_std_list = []
    params_avg_list = []

    # quantiles of observation at weighted ensemble
    quant_list = []
    quant_list_pm = []

    # analyze_error is to eliminate CRPS values after long offtimes (unfair) -
    # set to True initially, because, we want to capture the first day CRPS
    analyze_error = True

    samp_ptcls_list = []
    samp_mptcls_list = []

    pmean_list = []
    perror_list = []
    mmean_list = []
    mstd_list = []
    obs_list = []
    obsstd_list = []

    voi = 0
    param_estimates_init = None
    for date in date_span:
        log.info(str(date) + ' \nMPROB: ', aepf.model_prob, '\nM_T_ALL: ',
                 aepf.M_t_all)

        mprob_list.append(aepf.model_prob)
        mpart_list.append(aepf.n_model_particles)

        doy = date.dayofyear
        ipot = ipot_year[:, doy]
        ssf = sis_scale_fac[fl_ids, doy, np.newaxis]

        try:
            snowredistfac = snowredist_ds.sel(time=date).D.values
        except (KeyError, AttributeError):
            snowredistfac = None

        aepf.predict(mb_models_inst, gmeteo, date, h, ssf, ipot,
                     ipot_sigma, alpha_ix, mod_ix, swe_ix, tacc_ix, mb_ix,
                     tacc_ice, model_error_mean, model_error_std,
                     obs_merge=obs_merge,
                     param_random_walk=cfg.PARAMS['param_random_walk'],
                     snowredistfac=snowredistfac,
                     alpha_underlying=a_under_obs_at_fl, seed=seed)

        if (aepf.particles[..., alpha_ix] < 0.).any():
            log.warning('Albedo is smaller than zero')
            print(aepf.particles[..., alpha_ix]
                  [aepf.particles[..., alpha_ix] < 0.])

        if plot_dict['plot_mb_and_ensemble_evolution_overview'] is True:
            params_per_model = aepf.params_per_model
            # Braithwaite mu_ice
            ax2.errorbar(
                date, np.mean(params_per_model[0][:, 0]),
                yerr=np.std(params_per_model[0][:, 0]), fmt='o', c=colors[0])
            # Braithwaite prcp_fac
            ax2.errorbar(date, np.mean(params_per_model[0][:, 1]),
                         yerr=np.std(params_per_model[0][:, 1]), fmt='o',
                         c=colors[0])
            # Pellicciotti tf
            try:
                pelli_ix = [m.__name__ for m in mb_models]\
                    .index('PellicciottiModel')
                ax2.errorbar(
                    date, np.mean(params_per_model[pelli_ix][:, 0]),
                    yerr=np.std(params_per_model[pelli_ix][:, 0]), fmt='o',
                    c=colors[pelli_ix])
            except ValueError:
                pass

        if plot_dict['plot_parameter_evolution'] is True:
            # do the param evolution plot
            param_estimates = aepf.estimate_state_by_model(return_std=True,
                                                           space_avg=True)
            if date == date_span[0]:
                param_estimates_init = copy.deepcopy(param_estimates)
            ax_list = [ax6, ax7, ax8, ax9]
            ax_twin_list = [ax6twin, ax7twin, ax8twin, ax9twin]
            marker_list = ['o', '^', '*']
            for mi, m in enumerate(mb_models):
                pmean_list_model = []
                perror_list_model = []
                for pi, p in enumerate(m.cali_params_list):
                    pmean = param_estimates[0][mi][6 + pi]
                    if m.__name__ == 'OerlemansModel' \
                            and pi == m.cali_params_list.index('c0'):
                        pmean -= 100
                    perror = param_estimates[1][mi][6+pi]
                    if pi == len(m.cali_params_list)-1:
                        ax_twin_list[mi].errorbar(
                            date, pmean, yerr=perror, fmt=marker_list[-1],
                            c=colors[mi], capsize=5)
                    else:
                        ax_list[mi].errorbar(
                            date, pmean, yerr=perror, fmt=marker_list[pi],
                            c=colors[mi], capsize=5)
                    pmean_list_model.append(pmean)
                    perror_list_model.append(perror)
                pmean_list.append(pmean_list_model)
                perror_list.append(perror_list_model)

        if plot_dict['plot_parameter_evolution_ratio'] is True:
            ax_list = [ax10, ax11, ax12, ax13]
            for mi, m in enumerate(mb_models):
                for pi, p in enumerate(m.cali_params_list):
                    pratio = param_estimates[0][mi][6+pi] / \
                             param_estimates_init[0][mi][6+pi]
                    if m.__name__ == 'OerlemansModel' and \
                            pi == m.cali_params_list.index('c0'):
                        pratio = (param_estimates[0][mi][6+pi] - 100) / \
                                 (param_estimates_init[0][mi][6+pi] - 100)
                    ax_list[mi].scatter(date, pratio,
                                        marker=marker_list[pi], c=colors[mi])

        # update
        try:
            obs = np.atleast_2d(obs_merge_cs.sel(date=date).values)
            if np.isnan(obs).all():
                obs = None
                obs_std = None
                obs_phase = None
            else:
                # sometimes we manually increased the stdev
                std_manual = np.atleast_2d(obs_merge.sel(
                    date=date).swe_std.values)
                obs_std = std_manual.copy()
                obs_std[~np.isnan(std_manual)] = \
                    std_manual[~np.isnan(std_manual)] * obs_std_scale_fac
                obs_std[np.isnan(std_manual)] = \
                    obs_std_we_default * obs_std_scale_fac
                obs_phase = \
                    np.atleast_2d(obs_merge.sel(date=date).phase.values)

            # keep track of when camera is set up/gaps (need for reset MB)
            if date in first_obs_date:
                print('FIRST OBS', date)
                dix = date == first_obs_date
                if limit_to_camera_elevations is True:
                    # todo: implement indices_below for dix only
                    to_insert = mb_anal_before_all[dix, :].copy()
                    print(to_insert)
                    ptcls = aepf.particles.copy()
                    ptcls[dix, :, obs_init_mb_ix] = to_insert
                    if reset_albedo_and_swe_at_obs_init is True:
                        ptcls[dix, :, alpha_ix] = \
                            cfg.PARAMS['ice_albedo_default']
                        ptcls[dix, :, swe_ix] = 0.
                    aepf.particles = ptcls
                    print(aepf.particles[dix, :, obs_init_mb_ix])
                else:
                    to_insert = mb_anal_before_all[s_index[dix], :].copy()
                    print(to_insert)
                    ptcls = aepf.particles.copy()
                    ptcls[s_index[dix], :, obs_init_mb_ix] = to_insert
                    if reset_albedo_and_swe_at_obs_init is True:
                        indices_below = fl_h <= np.max(fl_h[s_index[dix]])
                        np.random.seed(seed)
                        ptcls[indices_below, :, alpha_ix] = \
                            np.random.normal(
                                cfg.PARAMS['ice_albedo_default'],
                                min_std_alpha,
                                ptcls[indices_below, :, alpha_ix].shape)
                        np.random.seed(seed)
                        ptcls[indices_below, :, swe_ix] = \
                            np.clip(
                                np.random.normal(
                                    0, min_std_swe,
                                    ptcls[indices_below, :, swe_ix].shape),
                                0., None)
                    aepf.particles = ptcls
                    print(aepf.particles[s_index[dix], :, obs_init_mb_ix])
        except:
            obs = None
            obs_std = None

        # todo: this is hardcoded: we take the first which is not NaN
        # 'valid observation index'
        if (obs is not None) and (voi is None):
            # voi shouldn't change anymore
            voi = np.where(~np.isnan(obs))[-1][0]
        if obs is not None:
            samp_mptcls_list.append(
                list(
                    np.random.choice((aepf.particles[s_index[voi], :, mb_ix] -
                                      aepf.particles[s_index[voi], :,
                                       obs_init_mb_ix]).flatten(),
                                     p=aepf.weights[s_index[voi], :].flatten(),
                                     size=1000)))

        box_offset = 0.2
        if obs is not None:
            boxvals = [aepf.particles[s_index[voi], aepf.model_indices_all[
                mi]][:, mb_ix]-obs[0][voi] for mi in aepf.model_range] + [
                aepf.particles[s_index[voi], :, mb_ix] - obs[0][voi]]
            boxpos = [(date - date_span[0]).days - aepf.n_models / 2. *
                      box_offset + box_offset * mi for mi in range(
                aepf.n_models)]

            if plot_dict['plot_median_absolute_departure'] is True:
                for mi in aepf.model_range:
                    ax5.boxplot(
                        boxvals[mi], positions=[boxpos[mi]], patch_artist=True,
                        boxprops=dict(facecolor=colors[mi], color=colors[mi]))
                ax5.axvline(boxpos[mi] + box_offset)

        # todo: make it an average
        if len(s_index) > 0:
            mod_pred_mean = [
                np.mean(aepf.particles[s_index[voi], :, mb_ix][
                            aepf.particles[s_index[voi], :, mod_ix] == i]) for
                i in range(aepf.n_models)]
            mod_pred_std = [
                np.std(aepf.particles[s_index[voi], :, mb_ix][
                           aepf.particles[s_index[voi], :, mod_ix] == i]) for
                i in range(aepf.n_models)]
            print('MOD PRED MEAN: ', mod_pred_mean)
            print('MOD PRED STD: ', mod_pred_std)
        if obs is not None:
            print('OBS: ', obs[0], ', OBS_STD: ', obs_std[0])
        else:
            print('No camera observations today.')
        print('PREDICTED: ', np.average(
            aepf.particles[s_index, :, mb_ix] -
            aepf.particles[s_index, :, obs_init_mb_ix], axis=1,
            weights=aepf.weights[s_index, :]))

        if plot_dict['plot_mb_and_ensemble_evolution_overview'] is True:
            # plot prediction by model
            y_jitter = np.array([pd.Timedelta(hours=2 * td) for td in
                                 np.linspace(-2.0, 2.0, aepf.n_models)])
            y_vals = np.array([date] * len(mb_models)) + y_jitter
            if len(s_index) > 0:
                ax1.scatter(y_vals, mod_pred_mean, c=colors[:len(mb_models)])
                ax1.errorbar(y_vals, mod_pred_mean, yerr=mod_pred_std, fmt='o',
                             zorder=0)

                aepf.plot_state_errorbars(ax1, date, colors=['y'],
                                          space_ix=s_index[voi])

            # plot particles distribution per model
            aepf.plot_particles_per_model(ax3, date, colors=colors)

            aepf.plot_state_errorbars(ax4, date - pd.Timedelta(hours=3),
                                      var_ix=alpha_ix, colors=['gold'],
                                      space_ix=list(range(int(len(h)/2))))
            aepf.plot_state_errorbars(ax4, date - pd.Timedelta(hours=3),
                                      var_ix=alpha_ix, colors=['y'],
                                      space_ix=list(range(int(len(h)/2),
                                                          len(h))))

            if obs is not None:
                # plot obs
                ax1.errorbar(date, obs[0][voi], yerr=obs_std[0][voi], c='k',
                             marker='o', fmt='o')

        if obs is not None:
            if analyze_error is True:
                if limit_to_camera_elevations is True:
                    # dep_list.append(
                    #    (obs -
                    #     np.average(aepf.particles[:, :, mb_ix] -
                    #                aepf.particles[:, :, obs_init_mb_ix],
                    #                weights=aepf.weights[aepf.stats_ix, :],
                    #                axis=1))
                    # )
                    crps_list.append(
                        crps_ensemble(obs * cfg.RHO_W / cfg.RHO,
                                      (aepf.particles[:, :, mb_ix] -
                                       aepf.particles[:, :, obs_init_mb_ix]) *
                                      cfg.RHO_W / cfg.RHO,
                                      aepf.weights[aepf.stats_ix, :])
                    )
                else:
                    # dep_list.append(
                    #    (obs -
                    #     np.average(aepf.particles[s_index, :, mb_ix] -
                    #               aepf.particles[s_index, :, obs_init_mb_ix],
                    #                weights=aepf.weights[s_index, :],
                    #                axis=1))
                    # )
                    # todo: check if crps calc.is correct ('observational
                    #  error is neglected')
                    # properscoring_crps = crps_ensemble(
                    # obs[0].T /cfg.RHO*cfg.RHO_W,
                    # (aepf.particles[s_index, :,mb_ix] -
                    # aepf.particles[s_index,:,obs_init_mb_ix])/
                    # cfg.RHO*cfg.RHO_W, aepf.weights[s_index, :])
                    # crps_1 = crps_by_observation_height(aepf,
                    # obs.T/cfg.RHO*cfg.RHO_W, obs_std.T/cfg.RHO*cfg.RHO_W,
                    # s_index, mb_ix, obs_init_mb_ix)

                    resamp_ix = stratified_resample(
                        aepf.weights[aepf.stats_ix, :], n_samples=1000,
                        one_random_number=True, seed=seed)
                    resamp_p, _ = resample_from_index_augmented(
                        ((aepf.particles[s_index, :, mb_ix] -
                          aepf.particles[s_index, :, obs_init_mb_ix]) *
                         cfg.RHO_W / cfg.RHO)[:, :, np.newaxis],
                        aepf.weights[s_index, :], resamp_ix
                    )

                    properscoring_crps = crps_ensemble(
                        obs[0].T * cfg.RHO_W / cfg.RHO, resamp_p[..., 0])
                    crps_1 = crps_by_observation_height_direct(
                        resamp_p[..., 0], obs.T * cfg.RHO_W / cfg.RHO,
                        obs_std.T / cfg.RHO * cfg.RHO_W)
                    four_ixs = np.random.uniform(
                        low=0, high=resamp_p.shape[1]-1, size=4).astype(int)
                    crps_2 = crps_by_observation_height_direct(
                        resamp_p[:, four_ixs, 0], obs.T * cfg.RHO_W / cfg.RHO,
                        obs_std.T / cfg.RHO * cfg.RHO_W
                    )

                    if crps_ice_only is True:
                        print(obs_phase)
                        properscoring_crps[obs_phase[0] == 's'] = np.nan
                        crps_1[obs_phase[0] == 's'] = np.nan
                        crps_2[obs_phase[0] == 's'] = np.nan

                    crps_list.append(properscoring_crps)
                    crps_1_list.append(crps_1)
                    crps_2_list.append(crps_2)
            else:
                # dep_list.append(np.full_like(obs, np.nan))
                # todo: check if crps calc.is correct ('observational error is
                #  neglected')
                crps_list.append(np.full_like(obs[0].T, np.nan))
                crps_1_list.append(np.full_like(obs[0].T, np.nan))
                crps_2_list.append(np.full_like(obs[0].T, np.nan))

            analyze_error = True
            print('CRPS at VOI: ', crps_list[-1], crps_1_list[-1])
            print('MEAN CRPS at VOI: ', np.nanmean(crps_list[-1]),
                  np.nanmean(crps_1_list[-1]))
            obs_quant = aepf.get_observation_quantiles(
                obs, obs_std, mb_ix=mb_ix, mb_init_ix=obs_init_mb_ix,
                eval_ix=s_index, by_model=False
            )
            quant_list.append(obs_quant)
            obs_quant_pm = aepf.get_observation_quantiles(
                obs, obs_std, mb_ix=mb_ix, mb_init_ix=obs_init_mb_ix,
                eval_ix=s_index, by_model=True
            )
            quant_list_pm.append(obs_quant_pm)

            # obs covariance
            R = np.mat(np.eye(obs.shape[1]))

            if update is True:
                aepf.update(obs, obs_std, R, obs_ix=mb_ix,
                            obs_init_mb_ix=obs_init_mb_ix,
                            obs_spatial_ix=spatial_cam_ix, date=date)

            if (aepf.particles[..., alpha_ix] < 0.).any():
                print('Albedo smaller than zero here')
                print(aepf.particles[..., alpha_ix][
                          aepf.particles[..., alpha_ix] < 0.])

        obs_list.append(obs)
        obsstd_list.append(obs_std)
        mmean, mstd = aepf.estimate_state(return_std=True)
        mmean_list.append(mmean[spatial_cam_ix, mb_ix]
                          - mmean[spatial_cam_ix, obs_init_mb_ix])
        mstd_list.append(mstd[spatial_cam_ix, mb_ix])

        # todo: remove this! it's only for testing!!!
        if (assimilate_albedo is True) and (date in alpha_proc.time):
            a_obs = alpha_proc.sel(time=date).albedo.values

            if a_obs.shape[0] != len(fl_h):
                a_obs = a_obs.T
            # on the first day with alpha observations
            if date == alpha_proc.time[alpha_proc.time >= date_span[0]][0]:
                # todo: is debiasing a good idea?
                bias = np.atleast_2d(
                    np.average(aepf.particles[..., alpha_ix],
                               weights=aepf.weights)
                    - np.nanmean(a_obs, axis=1)).T
                aepf.particles[..., alpha_ix] = \
                    np.clip(aepf.particles[..., alpha_ix]
                            - np.nanmedian(bias), 0.01, 0.99)
                aepf.particles[..., tacc_ix] = \
                    tacc_from_alpha_brock(aepf.particles[..., alpha_ix])

                # make obs more uncertain
                a_logit = utils.physical_to_logit(a_obs, 0., 1., 1e-2)
                a_obs = utils.logit_to_physical(a_logit * np.random.normal(
                    np.ones(a_obs.shape[0])[:, np.newaxis],
                    np.nanstd(a_logit, axis=1)[:, np.newaxis],
                    size=a_obs.shape), 0., 1., 1e-2)

            aepf.update_with_alpha_obs(a_obs, alpha_ix, date=date)

            if plot_dict['plot_mb_and_ensemble_evolution_overview'] is True:
                # alpha from Pellicciotti
                aepf.plot_state_errorbars(
                    ax4, date+pd.Timedelta(hours=3), var_ix=alpha_ix,
                    colors=['chartreuse'], space_ix=list(range(int(len(h)/2))),
                    by_model=False)
                aepf.plot_state_errorbars(
                    ax4, date + pd.Timedelta(hours=3), var_ix=alpha_ix,
                    colors=['g'], space_ix=list(range(int(len(h)/2), len(h))),
                    by_model=False)
                ax4.errorbar(
                    date, np.nanmean(np.nanmean(a_obs, axis=1)
                                     [list(range(int(len(h)/2)))]), fmt='o',
                    yerr=np.nanmean(np.nanstd(a_obs, axis=1)
                                    [list(range(int(len(h)/2)))]),
                    c='lightcoral')
                ax4.errorbar(
                    date, np.nanmean(np.nanmean(a_obs, axis=1)
                                     [range(int(len(h)/2), len(h))]), fmt='o',
                    yerr=np.nanmean(np.nanstd(a_obs, axis=1)
                                    [list(range(int(len(h)/2), len(h)))]),
                    c='firebrick')

        if (assimilate_fsca is True) and (date in swe_proc.time):
            s_obs = swe_proc.sel(time=date).swe_prob.values
            if s_obs.shape[-1] != len(fl_h):
                s_obs = s_obs.T
            # on the first day with alpha observations
            a_obs = alpha_proc.sel(time=date).albedo.values

            if a_obs.shape[0] != len(fl_h):
                a_obs = a_obs.T
            mb_obs = np.full_like(h, np.nan)
            mb_obs[spatial_cam_ix] = obs
            obs_all = np.vstack([mb_obs, a_obs, s_obs])
            aepf.update_together(obs_all, mb_ix, obs_init_mb_ix, swe_ix, alpha_ix)
            aepf.update_with_snowline_obs(s_obs, swe_ix, date=date)

        if len(s_index) > 0:
            print('UPDATED: ', np.average(
                aepf.particles[s_index, :, mb_ix] -
                aepf.particles[s_index, :, obs_init_mb_ix],
                weights=aepf.weights[s_index, :], axis=1)
                  )

            try:
                samp_ptcls_list.append(
                    list(np.random.choice(
                        aepf.particles[s_index[voi], :, mb_ix] -
                        aepf.particles[s_index[voi], :, obs_init_mb_ix],
                        p=aepf.weights[s_index[voi], :], size=1000)))
            except ValueError:
                print(aepf.weights[s_index[voi], :])
                print(aepf.log_weights[s_index[voi], :])
                pass

        print('MB today entire glacier: ', np.average(np.average(
            aepf.particles[..., mb_ix] - aepf.particles[..., obs_init_mb_ix],
            weights=aepf.weights, axis=1), weights=w)
              )

        if len(s_index) > 0:
            mb_anc_ptcls.append(aepf.particles[s_index, :, mb_ix])
            alpha_anc_ptcls.append(aepf.particles[s_index, :, alpha_ix])

        if ((alpha_proc is not None) and (date in alpha_proc.time))\
                or ((swe_proc is not None) and (date in swe_proc.time)) or \
                (obs is not None):
            # resample
            if (evolve_params is True) and (param_method == 'liu'):
                aepf.resample(phi=phi, gamma=gamma, diversify=True, seed=seed)
            else:
                aepf.resample(phi=phi, gamma=gamma, diversify=False, seed=seed)

            if len(s_index) > 0:
                mb_anc_ptcls_after\
                    .append(aepf.particles[s_index, :, mb_ix])
                alpha_anc_ptcls_after\
                    .append(aepf.particles[s_index, :, alpha_ix])
            if obs is not None:
                crps_array_old = np.full((len(mb_models), len(obs.T)), np.nan)
                crps_array_new = np.full((len(mb_models), len(obs.T)), np.nan)
                for mii, mic in enumerate(aepf.model_indices_all):
                    for oi, o in enumerate(obs[0].T):
                        if ~np.isnan(o):
                            mb_ppm = aepf.particles[spatial_cam_ix[oi], mic, mb_ix] - aepf.particles[spatial_cam_ix[oi], mic, obs_init_mb_ix]
                            properscoring_crps = crps_ensemble(o * cfg.RHO_W / cfg.RHO, mb_ppm)
                            crps_1 = crps_by_observation_height_direct(np.atleast_2d(mb_ppm), np.array([o]) * cfg.RHO_W / cfg.RHO, np.atleast_2d(obs_std.T[oi]) / cfg.RHO * cfg.RHO_W)
                            crps_array_old[mii, oi] = properscoring_crps
                            crps_array_new[mii, oi] = crps_1
                crps_by_model_old.append(crps_array_old)
                crps_by_model_new.append(crps_array_new)
                crps_by_model_dates.append(date)

            if plot_dict['plot_mb_and_ensemble_evolution_overview'] is True:
                aepf.plot_state_errorbars(
                    ax1, date + pd.Timedelta(hours=0.75), colors=['r'],
                    space_ix=s_index[voi])
        else:  # no observation this time
            print('NO RESAMPLING TODAY.')
            analyze_error = False
            samp_ptcls_list.append(list(np.full(1000, np.nan)))

        if return_params is True:
            write_param_dates_list.append(date)
            # 1) write weights
            p_avg = [
                np.average(aepf.params_per_model[k],
                           weights=aepf.model_weights[k], axis=0) for k in
                range(len(mb_models))]
            params_avg_list.append(p_avg)
            # 2) write actual params

            p_std = np.sqrt([
                np.average((aepf.params_per_model[k]-p_avg[k])**2,
                           weights=aepf.model_weights[k], axis=0) for k in
                range(len(mb_models))])
            params_std_list.append(p_std)

        if (alpha_ds is not None) and (date in alpha_ds.time) \
                or (obs is not None):
            if (evolve_params is True) and (param_method == 'memory'):
                aepf.evolve_theta(theta_priors_means, theta_priors_cov,
                                  rho=theta_memory,
                                  change_mean=change_memory_mean,
                                  seed=seed)

        #  check if minimum std requirements are fulfilled:
        # increase_std_alpha = np.std(aepf.particles[..., alpha_ix],
        #                            axis=1) < min_std_alpha
        # print('Albedo std increased at indices: ',
        #       fl_ids[increase_std_alpha])
        # aepf.particles[increase_std_alpha, :, alpha_ix] = \
        #    aepf.particles[increase_std_alpha, :, alpha_ix] + \
        #    np.random.normal(np.atleast_2d(np.zeros(
        #    np.sum(increase_std_alpha))).T, np.atleast_2d(
        #    np.ones(np.sum(increase_std_alpha)) * min_std_alpha).T,
        #    aepf.particles[increase_std_alpha, :, alpha_ix].shape)
        # increase_std_swe = np.std(aepf.particles[..., swe_ix],
        #                      axis=1) < min_std_swe
        # print('Albedo std increased at indices: ', fl_ids[increase_std_swe])
        # aepf.particles[increase_std_swe, :, alpha_ix] =
        # aepf.particles[increase_std_swe, :, swe_ix] + np.clip(
        #    np.random.normal(np.atleast_2d(
        #    np.zeros(np.sum(increase_std_swe))).T,
        #    np.atleast_2d(np.ones(np.sum(increase_std_swe)) * min_std_swe).T,
        #    aepf.particles[increase_std_swe, :, swe_ix].shape), 0., None)

        # save analysis
        mb_anal_before = np.atleast_2d(
            np.average(aepf.particles[..., mb_ix], axis=1,
                       weights=aepf.weights))
        mb_anal_before_all = np.atleast_2d(aepf.particles[..., mb_ix])
        mb_anal_std_before = np.sqrt(
            np.average((aepf.particles[..., mb_ix] - mb_anal_before.T) ** 2,
                       weights=aepf.weights, axis=1))

        run_example_list.append(np.average(
            aepf.particles[..., mb_ix], weights=w, axis=0)[rand_45])

    # calculate weighted average over heights at the end of the MB year
    mb_until_assim = mb_init
    resamp_ix = stratified_resample(aepf.weights[aepf.stats_ix, :],
                                    aepf.particles.shape[1])
    mb_during_assim_eq_weights = \
        np.array([resample_from_index(aepf.particles[i, :, mb_ix],
                                      aepf.weights[i, ...], resamp_ix)[0]
                  for i in np.arange(aepf.particles.shape[0])])

    if (aepf.particles[..., alpha_ix] < 0.).any():
        print('Albedo smaller than zero there')
        print(
            aepf.particles[..., alpha_ix][aepf.particles[..., alpha_ix] < 0.])

    # todo: shuffle doesn't matter?
    # todo_subtract OBS_INIT=?
    np.random.seed(seed)
    [np.random.shuffle(mb_during_assim_eq_weights[x, ...]) for x in range(
        mb_during_assim_eq_weights.shape[0])]
    mb_total = mb_until_assim + mb_during_assim_eq_weights

    if plot_dict['plot_glamos_point_comparison'] is True:
        if gid == 'RGI50-11.B4312n-1':
            rho_stake_ids = np.argmin(
                np.abs((fl_h - np.atleast_2d(
                    [3234., 3113, 2924., 2741, 2595., 2458., 2345., 2279.,
                     2228., 2838., 2306, 2222.]).T)), axis=1)
            # to convert numbers in GLAMOS table to m w.e.
            rho_densities = [0.52, 0.47, 0.47, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                             0.9, 0.9, 0.9]
            idxs = np.argmin(np.abs((fl_h - np.atleast_2d(
                [3234., 3113., 2924., 2741., 2595., 2458., 2345., 2279., 2228.,
                 2838., 2306., 2222.]).T)), axis=1)
            ms = mb_total[idxs, :]
            print('MB at GLAMOS stakes: ',
                  np.mean(mb_total[rho_stake_ids, :], axis=1))
            print('Height distances: ', np.min(
                np.abs((fl_h - np.atleast_2d([
                    3234., 3113., 2924., 2741., 2595., 2458., 2345., 2279.,
                    2228., 2838., 2306., 2222.]).T)), axis=1))
            plt.figure()
            rho_glamos = np.array(
                [250, 91, 92, -178, -483, -645, -635, -548, -702, -83, -640,
                 -759]) / 100. * rho_densities
            rho_mean = np.mean(ms, axis=1)
            rho_std = np.std(ms, axis=1)

            plt.scatter([3234., 3113., 2924., 2741., 2595., 2458., 2345.,
                         2279., 2228., 2838., 2306., 2222.],
                        rho_glamos, label='GLAMOS')
            plt.errorbar([3234., 3113., 2924., 2741., 2595., 2458., 2345.,
                          2279., 2228., 2838., 2306., 2222.], rho_mean,
                         yerr=rho_std, fmt='o', label='MOD', c='g')
            mad = np.mean(np.abs(rho_mean - rho_glamos))
            plt.title('RHONE MAD: {:.2f} m w.e.'.format(mad))
            plt.xlabel('Elevation (m)')
            plt.ylabel('Mass Balance in OBS period (m w.e.)')
            plt.legend()
            plt.show()

        if gid == 'RGI50-11.B5616n-1':
            fin_stake_ids = np.argmin(np.abs((fl_h - np.atleast_2d(
                [2619., 2597., 2680., 2788., 2920., 3036., 3122., 3149., 3087.,
                 3258., 3255., 3341., 3477.]).T)), axis=1)
            # to convert numbers in GLAMOS table to m w.e. (already in m w.e.)
            print('MB at GLAMOS stakes: ',
                  np.mean(mb_total[fin_stake_ids, :], axis=1))
            print('Height distances: ', np.min(np.abs((fl_h - np.atleast_2d(
                [2602.8, 2664.3, 2787.6, 2920.7, 3036.4, 3126.5, 3155.0,
                 3090.3, 3260.0, 3261.4, 3344.3, 3485.1]).T)), axis=1))
            plt.figure()
            fin_glamos = np.array(
                [-5720, -5880, -3930, -2000, -1840, -1060, -1990, -1630,   20,
                 -580,  440, 1460]) / 1000.  # already in mm w.e.
            idxs = np.argmin(np.abs((fl_h - np.atleast_2d(
                [2602.8, 2664.3, 2787.6, 2920.7, 3036.4, 3126.5, 3155.0,
                 3090.3, 3260.0, 3261.4, 3344.3, 3485.1]).T)), axis=1)
            ms = mb_total[idxs, :]
            fin_mean = np.mean(ms, axis=1)
            fin_std = np.std(ms, axis=1)

            plt.scatter(
                [2602.8, 2664.3, 2787.6, 2920.7, 3036.4, 3126.5, 3155.0,
                 3090.3, 3260.0, 3261.4, 3344.3, 3485.1], fin_glamos,
                label='GLAMOS')
            plt.errorbar(
                [2602.8, 2664.3, 2787.6, 2920.7, 3036.4, 3126.5, 3155.0,
                 3090.3, 3260.0, 3261.4, 3344.3, 3485.1], fin_mean,
                yerr=fin_std, fmt='o', label='MOD', c='g')
            mad = np.mean(np.abs(fin_mean - fin_glamos))
            plt.title('FINDELEN MAD: {:.2f} m w.e.'.format(mad))
            plt.xlabel('Elevation (m)')
            plt.ylabel('Mass Balance in OBS period (m w.e.)')
            plt.legend()
            plt.show()
        if gid == 'RGI50-11.A55F03':
            plm_stake_ids = np.argmin(np.abs((fl_h - np.atleast_2d(
                [2692.0, 2715.0, 2753.0, 2660.0, 2681.0]).T)), axis=1)
            # to convert numbers in GLAMOS table to m w.e. (already in m w.e.)
            print('MB at GLAMOS stakes: ',
                  np.mean(mb_total[plm_stake_ids, :], axis=1))
            print('Height distances: ', np.min(np.abs((fl_h - np.atleast_2d(
                [2692.0, 2715.0, 2753.0, 2660.0, 2681.0]).T)), axis=1))
            plt.figure()
            # already in mm w.e.
            plm_glamos = np.array(
                [-2016, -1530, -1467, -1692, -1863]) / 1000.
            idxs = np.argmin(np.abs((fl_h - np.atleast_2d(
                [2692.0, 2715.0, 2753.0, 2660.0, 2681.0]).T)), axis=1)
            ms = mb_total[idxs, :]
            plm_mean = np.mean(ms, axis=1)
            plm_std = np.std(ms, axis=1)

            plt.scatter(
                [2692.0, 2715.0, 2753.0, 2660.0, 2681.0], plm_glamos,
                label='GLAMOS')
            plt.errorbar(
                [2692.0, 2715.0, 2753.0, 2660.0, 2681.0], plm_mean,
                yerr=plm_std, fmt='o', label='MOD', c='g')
            mad = np.mean(np.abs(plm_mean - plm_glamos))
            plt.title('PLM MAD: {:.2f} m w.e.'.format(mad))
            plt.xlabel('Elevation (m)')
            plt.ylabel('Mass Balance in OBS period (m w.e.)')
            plt.legend()
            plt.show()

    mb_total_avg = np.average(mb_total, weights=w, axis=0)
    print('MB total on {}: {} $\pm$ {}'.format(date_span[-1], np.mean(
        mb_total_avg), np.std(mb_total_avg)))
    unc_before_pf = np.average(np.std(mb_until_assim, axis=1), weights=w,
                               axis=0)
    unc_during_pf = np.average(np.std(mb_during_assim_eq_weights, axis=1),
                               weights=w, axis=0)
    print('UNCERTAINTY before PF:{}, while PF: {}'.format(
        unc_before_pf, unc_during_pf))
    state = aepf.estimate_state(return_std=True)
    print('ALT: UNCERTAINTY before PF:{}, while PF: {}'.format(
        np.std(np.average(mb_until_assim, weights=w, axis=0)),
        np.average(state[0], weights=w, axis=0)))
    mb_total = np.sort(mb_until_assim) + np.sort(mb_during_assim_eq_weights)
    mb_total_avg = np.average(mb_total, weights=w, axis=0)
    print('MB total sorted on {}: {} $\pm$ {}'.format(date_span[-1], np.mean(
        mb_total_avg), np.std(mb_total_avg)))

    if plot_dict['plot_mb_and_ensemble_evolution_overview']:
        ax1.set_ylabel('Mass Balance (m w.e.)')
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                        Line2D([0], [0], color=colors[1], lw=4),
                        Line2D([0], [0], color=colors[2], lw=4),
                        Line2D([0], [0], color=colors[3], lw=4),
                        Line2D([0], [0], color='k', lw=4),
                        Line2D([0], [0], color='y', lw=4),
                        Line2D([0], [0], color='r', lw=4)]
        ax1.legend(custom_lines,
                   ['Braithwaite', 'Hock', 'Pellicciotti', 'Oerlemans', 'OBS',
                    'PRED', 'POST'])
        ax1.grid()
        ax2.set_ylabel('$\mu^*_{ice}$ (mm ice K-1 d-1)\nTF ()')
        ax2.legend()
        ax2.grid()
        ax3.set_ylabel('Number of model particles')
        ax3.legend()
        ax3.grid()
        ax4.set_ylabel('Albedo distribution')
        ax4.set_ylabel('Albedo distribution')
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='firebrick'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='gold'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='y'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='chartreuse'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='g')]
        ax4.legend(custom_lines,
                   ['OBS TOP', 'OBS BOTTOM', 'PRIOR TOP', 'PRIOR BOTTOM',
                    'POST TOP', 'POST BOTTOM'], scatterpoints=1)
        ax4.grid()

        ax4.grid()
        fig.suptitle('Mass Balance and Ensemble Evolution on {} ({})'.format(
            gdir.name, stations))
        fig.savefig('C:\\users\\johannes\\documents\\publications'
                    '\\Paper_Cameras\\test\\{}_panel.png'.format(stations),
                    dpi=500)
    if plot_dict['plot_median_absolute_departure'] is True:
        ax5.axhline(0.)
        ax5.set_xlabel('Days since Camera Setup')
        ax5.set_ylabel('Departure (PRED-OBS) (m w.e.)')
        ax5.legend(custom_lines[:-3], ['Braithwaite', 'Hock', 'Pellicciotti',
                                       'Oerlemans'])
        fig2.suptitle(
            '{} ({}). Median absolute ensemble departure: {:.3f} m w.e.'
                .format(gdir.name, stations, np.nanmedian(np.abs(
                    [item for sublist in dep_list for item in sublist]))))
        fig2.savefig('C:\\users\\johannes\\documents\\publications'
                     '\\Paper_Cameras\\test\\{}_boxplot.png'.format(stations),
                     dpi=500)
    print('Median absolute departure old method: ',
          np.nanmedian(np.abs([
              item for sublist in dep_list for item in sublist]))
          )
    print('Median absolute departure new method: ',
          np.nanmedian([np.abs(np.nanmean(sublist)) for sublist in dep_list]))

    fontsize = 20
    if plot_dict['plot_parameter_evolution']:
        fig3.suptitle('Parameter Evolution on {} ({})'.format(
            gdir.name, stations), fontsize=fontsize)
        markersize_legend = 12

        ax6_legend_elements = [
            Line2D([0], [0], marker='o', color='none',
                   label='$DDF_{ice}$', markerfacecolor=colors[0],
                   markersize=markersize_legend),
            Line2D([0], [0], marker='^', color='none',
                   label='$prcp_{scale}$', markerfacecolor=colors[0],
                   markersize=markersize_legend)
                ]
        ax7_legend_elements = [
            Line2D([0], [0], marker='o', color='none',
                   label='$MF$', markerfacecolor=colors[1],
                   markersize=markersize_legend),
            Line2D([0], [0], marker='*', color='none',  label='$a_{ice}$',
                   markerfacecolor=colors[1], markersize=15),
            Line2D([0], [0], marker='^', color='none',
                   label='$prcp_{scale}$', markerfacecolor=colors[1],
                   markersize=markersize_legend)]
        ax8_legend_elements = [
            Line2D([0], [0], marker='o', color='none',
                   label='$TF$', markerfacecolor=colors[2],
                   markersize=markersize_legend),
            Line2D([0], [0], marker='*', color='none',
                   label='$SRF$', markerfacecolor=colors[2],
                   markersize=markersize_legend),
            Line2D([0], [0], marker='^', color='none',
                   label='$prcp_{scale}$', markerfacecolor=colors[2],
                   markersize=markersize_legend)]
        ax9_legend_elements = [
            Line2D([0], [0], marker='o', color='none',
                   label='$-c_{0}-100$', markerfacecolor=colors[3],
                   markersize=markersize_legend),
            Line2D([0], [0], marker='*',  color='none', label='$c_{1}$',
                   markerfacecolor=colors[3], markersize=15),
            Line2D([0], [0], marker='^', color='none',
                   label='$prcp_{scale}$', markerfacecolor=colors[3],
                   markersize=markersize_legend)
        ]
        ax6.set_ylabel('BraithwaiteModel', fontsize=fontsize)
        ax6.legend(handles=ax6_legend_elements, loc=4, fontsize=fontsize, framealpha=0.5)
        ax6.grid()
        ax7.set_ylabel('HockModel', fontsize=fontsize)
        ax7.legend(handles=ax7_legend_elements, loc=4, fontsize=fontsize, framealpha=0.5)
        ax7.grid()
        ax8.set_ylabel('PellicciottiModel', fontsize=fontsize)
        ax8.legend(handles=ax8_legend_elements, loc=4, fontsize=fontsize, framealpha=0.5)
        ax8.grid()
        ax9.set_ylabel('OerlemansModel', fontsize=fontsize)
        ax9.legend(handles=ax9_legend_elements, loc=4, fontsize=fontsize, framealpha=0.5)
        ax9.grid()
        plt.setp(ax6.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax7.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax8.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax9.get_yticklabels(), fontsize=fontsize)
        plt.setp(ax9.get_xticklabels(), fontsize=fontsize)
        fig3.savefig('C:\\users\\johannes\\documents\\publications'
                     '\\Paper_Cameras\\test\\{}_param_evol.png'.format(
            stations), dpi=500)

    if plot_dict['plot_parameter_evolution_ratio'] is True:
        fig4.suptitle('Parameter Evolution (ratio) on {} ({})'
                      .format(gdir.name, stations))
        ax10.set_ylabel('BraithwaiteModel', fontsize=fontsize)
        ax10.legend(handles=ax6_legend_elements, loc=4, fontsize=fontsize)
        ax10.grid()
        ax11.set_ylabel('HockModel')
        ax11.legend(handles=ax7_legend_elements, loc=4, fontsize=fontsize)
        ax11.grid()
        ax12.set_ylabel('PellicciottiModel')
        ax12.legend(handles=ax8_legend_elements, loc=4, fontsize=fontsize)
        ax12.grid()
        ax13.set_ylabel('OerlemansModel')
        ax13.legend(handles=ax9_legend_elements, loc=4, fontsize=fontsize)
        ax13.grid()
        fig4.savefig('C:\\users\\johannes\\documents\\publications'
                     '\\Paper_Cameras\\test\\{}_param_evol_ratio.png'.format(
            stations), dpi=500)

    if plot_dict['plot_median_crps'] is True:
        plt.figure()
        # todo: WRONG DATES
        plt.plot(date_span[:len(crps_list)],
                 np.nanmean(np.array(crps_list), axis=1), label='old CRPS')
        plt.plot(date_span[:len(crps_1_list)],
                 np.nanmean(np.array(crps_1_list), axis=1), label='CRPS 1')
        plt.legend()
        plt.title('{}, MEDIAN CRPS: {}, {}, {}'.format(
            gdir.name,
            np.nanmedian([item for sublist in crps_list for item in sublist]),
            np.nanmedian([item for sublist in crps_1_list for item in sublist]),
            np.nanmedian([item for sublist in crps_2_list for item in sublist]),
            np.nanmedian([item for sublist in crps_3_list for item in sublist])))

        print('{}, MEDIAN CRPS: {}, {}, {}'.format(gdir.name, np.nanmedian(
            [item for sublist in crps_list for item in sublist]), np.nanmedian(
            [item for sublist in crps_1_list for item in sublist]), np.nanmedian(
            [item for sublist in crps_2_list for item in sublist]), np.nanmedian(
            [item for sublist in crps_3_list for item in sublist])))

    if plot_dict['plot_talagrand_histogram'] is True:
        # quantile histogram for all models
        plt.figure()
        plt.hist([np.nanmean(i[0]) for i in quant_list])
        plt.title('{}, {} calibration parameters ({} fit); station mean'
            .format(gdir.name, generate_params, param_prior_distshape))
        plt.ylabel('N counts')
        plt.xlabel('Percentiles of OBS at weighted ensemble prediction')

        plt.figure()
        plt.hist(np.array([i[0] for i in quant_list]).flatten())
        plt.title('{}, {} calibration parameters ({} fit); stations '
                  'individual'.format(gdir.name, generate_params,
                                      param_prior_distshape))
        plt.ylabel('N counts')
        plt.xlabel('Percentiles of OBS at weighted ensemble prediction')

        # quantiles histogram per model
        figm, axms = plt.subplots(int(np.floor(np.sqrt(len(mb_models)))),
                                  int(np.ceil(np.sqrt(len(mb_models)))),
                                  sharex='all')
        figm.suptitle('{}, {} calibration parameters ({} fit); stations '
                      'individual'.format(gdir.name, generate_params,
                                          param_prior_distshape))
        try:
            axms_flat = axms.flat
        except AttributeError:  # only one model
            axms_flat = [axms]
        for i, axm in enumerate(axms_flat):
            sub_list = quant_list_pm[i]
            axm.hist(np.array([i[0] for i in sub_list]).flatten())
            axm.set_xlabel(mb_models[i].__name__)

    if plot_dict['plot_crps_by_model'] is True:
        plt.figure()
        plt.plot(crps_by_model_dates,
                 np.nanmean(np.array(crps_by_model_new), axis=1),
                 label=['Braithwaite', 'Hock', 'Pellicciotti', 'Oerlemans'])
        plt.legend()
        plt.title('Proper CRPS by model for {}'.format(gdir.name))

        plt.figure()
        plt.plot(crps_by_model_dates,
                 np.nanmean(np.array(crps_by_model_old), axis=1),
                 label=['Braithwaite', 'Hock', 'Pellicciotti', 'Oerlemans'])
        plt.legend()
        plt.title('Non-proper CRPS by model for {}'.format(gdir.name))

    if plot_dict['plot_ancestors'] is True:
        ancestor_plot(gdir=gdir, stat=voi, to_analyze_mb=mb_anc_ptcls,
                      to_analyze_alpha=alpha_anc_ptcls, forward=True,
                      plot_stat='diff_median')

        ancestor_plot(gdir=gdir, stat=voi, to_analyze_mb=mb_anc_ptcls,
                      to_analyze_alpha=alpha_anc_ptcls, forward=True,
                      plot_stat='percentiles')

        ancestor_plot(gdir=gdir, stat=voi, to_analyze_mb=mb_anc_ptcls,
                      to_analyze_alpha=alpha_anc_ptcls, forward=False,
                      plot_stat='diff_median')

        ancestor_plot(gdir=gdir, stat=voi, to_analyze_mb=mb_anc_ptcls,
                      to_analyze_alpha=alpha_anc_ptcls, forward=False,
                      plot_stat='percentiles')

    """
    stat = voi
    to_analyze_mb = mb_anc_ptcls
    to_analyze_alpha = alpha_anc_ptcls
    forward = True,
    plot_stat = 'percentiles'
    if forward is True:
        ma = to_analyze_mb[0][stat, :]
        al = to_analyze_alpha[0][stat, :]
        ixs = ancest_ix[0]
    else:
        ma = to_analyze_mb[-1][stat, :]
        al = to_analyze_alpha[-1][stat, :]
        ixs = ancest_ix[-1]
    if forward is True:
        iterrange = range(len(to_analyze_mb))
    else:
        iterrange = range(len(to_analyze_mb))[::-1][1:]

    perc_low_ma = []
    perc_med_ma = []
    perc_high_ma = []
    perc_low_al = []
    perc_med_al = []
    perc_high_al = []
    
    # todo: introduce this  plot to plot_dict
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
    for i in iterrange:
        if plot_stat == 'diff_median':
            ax1.scatter(np.full_like(np.unique(ma), i),
                        np.unique(ma) - np.median(ma))
            ax2.scatter(np.full_like(np.unique(al), i),
                        np.unique(al) - np.median(al))

        if plot_stat == 'filter':
            ax1.scatter(np.full_like(np.unique(ma), i), np.unique(ma))
            ax1.scatter(np.full_like(np.unique(al), i), np.unique(al))

        perc_low_ma.append(np.percentile(ma, 5))
        perc_med_ma.append(np.percentile(ma, 50))
        perc_high_ma.append(np.percentile(ma, 95))
        perc_low_al.append(np.percentile(ma, 5))
        perc_med_al.append(np.percentile(ma, 50))
        perc_high_al.append(np.percentile(ma, 95))

        ax2.scatter(np.full_like(np.unique(ixs), i), np.unique(ixs))

        try:
            ixs = ixs[ancest_ix[i]]
            ma = to_analyze_mb[i][stat, :][ixs]
            al = to_analyze_alpha[i][stat, :][ixs]
        except IndexError:
            pass

    if plot_stat == 'percentiles':
        ax1.fill_between(iterrange, perc_low_ma, perc_high_ma)
        ax1.plot(iterrange, perc_med_ma, color='orange')
        ax2.fill_between(iterrange, perc_low_al, perc_high_al)
        ax2.plot(iterrange, perc_med_al, color='orange')
    fig.suptitle('{} {} iteration'.format(
        gdir.name, 'forward' if forward is True else 'backward'))
    ax3.set_ylabel('Surviving indices')
    ax1.set_ylabel('{}'.format('MB'))
    ax2.set_ylabel('{}'.format('Albedo'))
    ax3.set_xlabel('Time steps (with observations)')

    stat = voi
    str_var = 'MB'
    if str_var == 'MB':
        to_analyze = mb_anc_ptcls
    elif str_var == 'albedo':
        to_analyze = alpha_anc_ptcls
    forward = True
    plot_stat = 'percentiles'

    if forward is True:
        var = to_analyze[0][stat, :]
        ixs = ancest_ix[0]
    else:
        var = to_analyze[-1][stat, :]
        ixs = ancest_ix[-1]
    if forward is True:
        iterrange = range(len(to_analyze))
    else:
        iterrange = range(len(to_analyze))[::-1][1:]

    perc_low = []
    perc_high = []

    fig, (ax1, ax2) = plt.subplots(2, sharex='all')
    for i in iterrange:
        if plot_stat == 'diff_median':
            ax1.scatter(np.full_like(np.unique(var), i),
                        np.unique(var) - np.median(var))

        if plot_stat == 'filter':
            ax1.scatter(np.full_like(np.unique(var), i), np.unique(var))

        perc_low.append(np.percentile(var, 5))
        perc_high.append(np.percentile(var, 95))

        ax2.scatter(np.full_like(np.unique(ixs), i), np.unique(ixs))

        try:
            ixs = ixs[ancest_ix[i]]
            var = to_analyze[i][stat, :][ixs]
        except IndexError:
            pass
    if plot_stat == 'percentiles':
        ax1.fill_between(iterrange, perc_low, perc_high)
    fig.suptitle('{} {} iteration'.format(
        gdir.name, 'forward' if forward is True else 'backward'))
    ax2.set_ylabel('Surviving indices')
    ax1.set_ylabel('{}'.format(str_var))
    ax2.set_xlabel('Time steps (with observations)')
    plt.show()
    stat = voi
    str_var = 'MB'
    if str_var == 'MB':
        to_analyze = mb_anc_ptcls
    elif str_var == 'albedo':
        to_analyze = alpha_anc_ptcls
    forward = False
    plot_stat = 'percentiles'

    if forward is True:
        var = to_analyze[0][stat, :]
        ixs = ancest_ix[0]
    else:
        var = to_analyze[-1][stat, :]
        ixs = ancest_ix[-1]
    if forward is True:
        iterrange = range(len(to_analyze))
    else:
        iterrange = range(len(to_analyze))[::-1][1:]

    perc_low = []
    perc_high = []

    fig, (ax1, ax2) = plt.subplots(2, sharex='all')
    for i in iterrange:
        if plot_stat == 'diff_median':
            ax1.scatter(np.full_like(np.unique(var), i),
                        np.unique(var) - np.median(var))

        if plot_stat == 'filter':
            ax1.scatter(np.full_like(np.unique(var), i), np.unique(var))

        perc_low.append(np.percentile(var, 5))
        perc_high.append(np.percentile(var, 95))

        ax2.scatter(np.full_like(np.unique(ixs), i), np.unique(ixs))

        try:
            ixs = ixs[ancest_ix[i]]
        except IndexError:
            return  # need to find a better solution for this
        var = to_analyze[i][stat, :][ixs]
    if plot_stat == 'percentiles':
        ax1.fill_between(iterrange, perc_low, perc_high)
    fig.suptitle('{} {} iteration'.format(
        gdir.name, 'forward' if forward is True else 'backward'))
    ax2.set_ylabel('Surviving indices')
    ax1.set_ylabel('{}'.format(str_var))
    ax2.set_xlabel('Time steps (with observations)')
    plt.show()
    stat = voi
    str_var = 'albedo'
    if str_var == 'MB':
        to_analyze = mb_anc_ptcls
    elif str_var == 'albedo':
        to_analyze = alpha_anc_ptcls
    forward = False
    plot_stat = 'percentiles'

    if forward is True:
        var = to_analyze[0][stat, :]
        ixs = ancest_ix[0]
    else:
        var = to_analyze[-1][stat, :]
        ixs = ancest_ix[-1]
    if forward is True:
        iterrange = range(len(to_analyze))
    else:
        iterrange = range(len(to_analyze))[::-1][1:]

    perc_low = []
    perc_high = []

    fig, (ax1, ax2) = plt.subplots(2, sharex='all')
    for i in iterrange:
        if plot_stat == 'diff_median':
            ax1.scatter(np.full_like(np.unique(var), i),
                        np.unique(var) - np.median(var))

        if plot_stat == 'filter':
            ax1.scatter(np.full_like(np.unique(var), i), np.unique(var))

        perc_low.append(np.percentile(var, 5))
        perc_high.append(np.percentile(var, 95))

        ax2.scatter(np.full_like(np.unique(ixs), i), np.unique(ixs))

        ixs = ixs[ancest_ix[i]]
        var = to_analyze[i][stat, :][ixs]
    if plot_stat == 'percentiles':
        ax1.fill_between(iterrange, perc_low, perc_high)
    fig.suptitle('{} {} iteration'.format(
        gdir.name, 'forward' if forward is True else 'backward'))
    ax2.set_ylabel('Surviving indices')
    ax1.set_ylabel('{}'.format(str_var))
    ax2.set_xlabel('Time steps (with observations)')
    plt.show()
    """

    if return_params is True:
        return param_dates_list, params_std_list, \
               params_avg_list
    elif return_probs is True:
        return date_span, mprob_list, mpart_list, pmean_list, perror_list, \
               [np.mean(mb_total_avg)], [np.std(mb_total_avg)]
        # mmean_list, mstd_list
    else:
        return None


# @entity_task(writes='aepf_status')
def step_aepf_from_to(gdir: crampon.GlacierDirectory, 
                      d1: Optional[pd.Timestamp] = None, 
                      d2: Optional[pd.Timestamp] = pd.Timestamp.now()) -> None:
    """
    Calculate time steps forward with the AugmentedEnsembleParticleFilter.
    
    Parameters
    ----------
    gdir: crampon.GlacierDirectory
        The GlacierDirectory to process.
    d1 : pd.Timestamp, optional
        Date where to start the forward calculation. If None, the date is 
        determined from the read AEPF netCDF state file. Default: None.
    d2 : pd.Timestamp, optional
        Date where to end the forward calculation. If None, the date is 
        set to be closest to "now". Default: pd.Timestamp.now().

    Returns
    -------
    None
    """

    # get filepath to AEPF file
    aepf_path = gdir.get_filepath('aepf_status')

    # read AEPF netCDF file
    aepf = AugmentedEnsembleParticleFilter.from_file(aepf_path)
    # step from min(last day, d1) to d2
    for date in pd.date_range(d1, d2):

        # predict
        aepf.predict(mb_models_inst, gmeteo, date, h, ssf, ipot,
                     ipot_sigma, alpha_ix, mod_ix, swe_ix, tacc_ix, mb_ix,
                     tacc_ice, model_error_mean, model_error_std,
                     obs_merge=obs_merge,
                     param_random_walk=cfg.PARAMS['param_random_walk'],
                     snowredistfac=snowredistfac,
                     alpha_underlying=a_under_obs_at_fl, seed=seed)

        # update
        aepf.update(obs, obs_std, R, obs_ix=mb_ix,
                    obs_init_mb_ix=obs_init_mb_ix,
                    obs_spatial_ix=spatial_cam_ix, date=date)

        # resample
        aepf.resample(phi=phi, gamma=gamma, diversify=False, seed=seed)

        # param evolve
        aepf.evolve_theta(theta_priors_means, theta_priors_cov,
                          rho=theta_memory,
                          change_mean=change_memory_mean, seed=seed)

    # calculate percentiles of distribution and write out to somewhere
    percentiles = calculate_percentiles(aepf.particles['mb'],
                                        weights=aepf.weights)
    percentiles.write()
    
    # write AEPF netCDF file
    aepf.to_file(aepf_path, date=d2)


def ancestor_plot(gdir, stat, to_analyze_mb, to_analyze_alpha, forward,
                  plot_stat):
    """
    Plot the ancestors of an assimilation model run.

    Parameters
    ----------
    gdir :
    stat :
    to_analyze_mb :
    to_analyze_alpha :
    forward :
    plot_stat :

    Returns
    -------

    """

    if forward is True:
        ma = to_analyze_mb[0][stat, :]
        al = to_analyze_alpha[0][stat, :]
        ixs = ancest_ix[0]
    else:
        ma = to_analyze_mb[-1][stat, :]
        al = to_analyze_alpha[-1][stat, :]
        ixs = ancest_ix[-1]
    if forward is True:
        iterrange = range(len(to_analyze_mb))
    else:
        iterrange = range(len(to_analyze_mb))[::-1][1:]

    perc_low_ma = []
    perc_med_ma = []
    perc_high_ma = []
    perc_low_al = []
    perc_med_al = []
    perc_high_al = []

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex='all')
    for i in iterrange:
        if plot_stat == 'diff_median':
            ax1.scatter(np.full_like(np.unique(ma), i),
                        np.unique(ma) - np.median(ma))
            ax2.scatter(np.full_like(np.unique(al), i),
                        np.unique(al) - np.median(al))

        if plot_stat == 'filter':
            ax1.scatter(np.full_like(np.unique(ma), i), np.unique(ma))
            ax1.scatter(np.full_like(np.unique(al), i), np.unique(al))

        perc_low_ma.append(np.percentile(ma, 5))
        perc_med_ma.append(np.percentile(ma, 50))
        perc_high_ma.append(np.percentile(ma, 95))
        perc_low_al.append(np.percentile(ma, 5))
        perc_med_al.append(np.percentile(ma, 50))
        perc_high_al.append(np.percentile(ma, 95))

        ax3.scatter(np.full_like(np.unique(ixs), i), np.unique(ixs))

        print(len(ancest_ix), i)
        try:
            ixs = ixs[ancest_ix[i]]
        except IndexError:  # brutal, but helps for the moment
            return
        ma = to_analyze_mb[i][stat, :][ixs]
        al = to_analyze_alpha[i][stat, :][ixs]
    if plot_stat == 'percentiles':
        ax1.fill_between(iterrange, perc_low_ma, perc_high_ma)
        ax1.plot(iterrange, perc_med_ma, color='orange')
        ax2.fill_between(iterrange, perc_low_al, perc_high_al)
        ax2.plot(iterrange, perc_med_al, color='orange')
    fig.suptitle('{} {} iteration'.format(
        gdir.name, 'forward' if forward is True else 'backward'))
    ax3.set_ylabel('Surviving indices')
    ax1.set_ylabel('{}'.format('MB'))
    ax2.set_ylabel('{}'.format('Albedo'))
    ax3.set_xlabel('Time steps (with observations)')


def A_normal(mu, sigma_sqrd):
    sigma = np.sqrt(sigma_sqrd)
    return 2 * sigma * _normpdf(mu / sigma) + mu * (
                2 * _normcdf(mu / sigma) - 1)


def auxcrpsC(m, s):
    return 2. * s * stats.norm.pdf(m / s, 0.) + \
           m * (2. * stats.norm.cdf(m / s, 1., 0.) - 1.)


def crpsmixnC(w, m, s, y):
    N = m.size
    crps1 = 0.0
    crps2 = 0.0
    W = 0.0

    for i in range(N):
        W += w[i]
        crps1 += w[i] * auxcrpsC(y - m[i], s[i])

        crps3 = 0.5 * w[i] * auxcrpsC(0.0, np.sqrt(2) * s[i])
        si2 = s[i] * s[i]
        for j in range(i):
            crps3 += w[j] * auxcrpsC(m[i] - m[j], np.sqrt(si2 + s[j]**2))
        crps2 += w[i] * crps3
    return (crps1 - crps2 / W) / W


def crps_mixnorm(y, m, s, w=None):
    res = []
    if w is None:
        w = np.ones(m.shape[1])  # mixture components (particles?)
        for n in range(m.shape[0]):
            res.append(crpsmixnC(w, m[n, :], s[n, :], y[n]))
    else:
        for n in range(m.shape[0]):
            res.append(crpsmixnC(w[n, :], m[n, :], s[n, :], y[n]))
    return res


def crps_by_observation_height(aepf, h_obs, h_obs_std, obs_ix, mb_ix,
                               obs_init_mb_ix):

    wgts = aepf.weights[obs_ix, :][0]
    ptcls = aepf.particles[obs_ix, :, mb_ix] - \
            aepf.particles[obs_ix, :, obs_init_mb_ix]
    # obs_origin_len = len(h_obs)
    # obs_notnan = np.where(~np.isnan(h_obs))[0]
    # ptcls = ptcls[obs_notnan]
    # h_obs = h_obs[obs_notnan]
    # h_obs_std = h_obs_std[obs_notnan]

    first_term = np.nansum(
        wgts * A_normal(h_obs - ptcls / cfg.RHO * cfg.RHO_W, h_obs_std**2),
        axis=1)
    len_weights = len(wgts)
    second_term = -0.5 * np.nansum([
        np.nansum(wgts * wgts[j] * A_normal(
            (ptcls.T - ptcls[:, j]).T / cfg.RHO * cfg.RHO_W, 2 * h_obs_std**2),
                  axis=1) for j in range(len_weights)], axis=0)
    second_term[np.where(np.isnan(h_obs))[0]] = np.nan
    # crps = first_term + second_term
    # result = np.full(obs_origin_len, np.nan)
    # result[obs_notnan] = crps
    return first_term + second_term


def corr_term(mu, sigma):
    """
    Potential correction term for truncated Gaussians as described in Gneiting
    2006 eq.5
    """
    return -2 * sigma * _normpdf(mu / sigma) * _normcdf(
        -mu / sigma) + sigma / np.sqrt(np.pi) * _normcdf(
        -np.sqrt(2) * mu / sigma) + mu * (_normcdf(-mu / sigma)) ** 2


def crps_by_observation_height_direct(ptcls, h_obs, h_obs_std, wgts=None):
    """ PTCLS, h_obs and h_obs_std must be given in OBS SPACE!!!"""
    if wgts is None:
        wgts = np.ones(ptcls.shape[1]) / ptcls.shape[1]
    # weights may not be n-dimensional at the moment
    assert wgts.ndim == 1
    first_term = np.nansum(wgts * A_normal(h_obs - ptcls, h_obs_std**2),
                           axis=1)
    len_weights = len(wgts)
    second_term = -0.5 * np.nansum([
        np.nansum(wgts * wgts[j] * A_normal((ptcls.T - ptcls[:, j]).T,
                                            2 * h_obs_std**2), axis=1) for j in
        range(len_weights)], axis=0)
    second_term[np.where(np.isnan(h_obs))[0]] = np.nan
    if ((first_term + second_term) < 0.).any():
        print('CALC ERROR IN CRPS')
    return first_term + second_term


def crps_by_observation_height_direct_vectorized(ptcls, h_obs, h_obs_std,
                                                 wgts=None):
    """ PTCLS, h_obs and h_obs_std must be given in OBS SPACE!!!"""
    if wgts is None:
        wgts = np.atleast_2d(np.ones_like(ptcls)) / ptcls.shape[1]
    # check if wgts really all sum to one
    # assert np.isclose(np.sum(wgts, axis=1), np.ones_like(wgts.shape[1]),
    # atol=0.001)
    first_term = np.sum(wgts * A_normal(h_obs - ptcls, h_obs_std**2), axis=1)
    forecasts_diff = (np.expand_dims(ptcls, -1) - np.expand_dims(ptcls, -2))
    weights_matrix = (np.expand_dims(wgts, -1) * np.expand_dims(wgts, -2))
    second_term = -0.5 * np.nansum(
        weights_matrix * A_normal(forecasts_diff, 2 * h_obs_std**2))
    return first_term + second_term


def crps_by_water_equivalent(aepf, h_obs, h_obs_std, obs_ix, mb_ix,
                             obs_init_mb_ix):
    wgts = aepf.weights[obs_ix, :][0]
    ptcls = aepf.particles[obs_ix, :, mb_ix] - \
            aepf.particles[obs_ix, :, obs_init_mb_ix]
    len_weights = len(wgts)
    first_term = np.sum(
        wgts * A_normal(cfg.RHO / 1000. * h_obs - ptcls,
                        (cfg.RHO / 1000. * h_obs_std**2)), axis=1)
    second_term = - 0.5 * np.sum([
        np.sum((wgts * wgts[j])[:, np.newaxis] * np.abs(ptcls.T - ptcls[:, j]),
               axis=0) for j in range(len_weights)]) \
        - 0.5 * A_normal(0., 2. * (cfg.RHO / 1000 * h_obs_std**2))
    # this stays 2D -> select [0]
    return first_term + second_term[:, 0]


def prepare_prior_parameters(
        gdir, mb_models, init_particles_per_model, param_prior_distshape,
        param_prior_std_scalefactor, generate_params=None, param_dict=None,
        seed=None) -> tuple:
    """

    Parameters
    ----------
    gdir : `py:class:crampon.GlacierDirectory`
         The GlacierDirectory to prepare the parameters for.
    mb_models : list
        list of MassBalanceModels to get the parameters for
    init_particles_per_model : int
        How many parameter set shall be generated.
    param_prior_distshape : str
        Which shape shall the parameter distribution have?
    param_prior_std_scalefactor :
        Do we want to scale the standard deviation of the parameter
        distribution to make it wider/narrower?
    generate_params : str
        The way we want to generate parameters, i.e. whether we want to get
        them from past calibration or from the Gabbi 2014-paper. Possible one
        of: ['past', 'gabbi', 'mean_past', 'mean_gabbi', None]. Default: None
        (equals 'past').
    param_dict :
        If the parameters shall stem from this dictionary only.
    seed: int, optional
        Whether to use a random number generator seed (for repeatable
        experiments). Default: None (non-repeatable).

    Returns
    -------
    theta_priors, theta_priors_means, theta_priors_cov: tuple
        Prior distributions, prior means & prior covariances of the parameters.
    """

    # get prior parameter distributions
    # the "np.abs" is to get c0 in OerlemansModel positive! (we take the
    #  log afterwards)
    if (generate_params == 'past') or (generate_params is None):
        theta_priors = [np.abs(
            get_prior_param_distributions(
                gdir, m, init_particles_per_model, fit=param_prior_distshape,
                std_scale_fac=param_prior_std_scalefactor[i], seed=seed))
            for i, m in enumerate(mb_models)]
        theta_priors_means = [np.median(np.log(tj), axis=0) for tj in
                              theta_priors]
        theta_priors_cov = [np.cov(np.log(tj).T) for tj in theta_priors]

    elif generate_params == 'gabbi':
        # get mean and cov of log(initial prior)
        theta_priors = [
            np.abs(get_prior_param_distributions_gabbi(
                m, init_particles_per_model, fit=param_prior_distshape))
            for m in mb_models]

        theta_priors_means = [np.mean(np.log(tj), axis=0) for tj in
                              theta_priors]
        theta_priors_cov = [np.cov(np.log(tj).T) for tj in theta_priors]
        theta_priors_cov = [np.eye(t.shape[0]) * t for t in theta_priors_cov]
    elif generate_params == 'mean_past':
        theta_priors = [np.abs(
            get_prior_param_distributions(
                gdir, m, init_particles_per_model, fit=param_prior_distshape,
                std_scale_fac=param_prior_std_scalefactor[i], seed=seed))
            for i, m in enumerate(mb_models)]
        theta_priors = [np.atleast_2d(np.mean(tj, axis=0)) for tj in
                        theta_priors]
        theta_priors = [np.repeat(tp, init_particles_per_model, axis=0) for tp
                        in theta_priors]
        theta_priors_means = [np.mean(np.log(tj), axis=0) for tj in
                              theta_priors]
        theta_priors_cov = [np.cov(np.log(tj).T) for tj in theta_priors]
        # set covariance to zero (only mean parameters)
        theta_priors_cov = [np.zeros_like(tjc) for tjc in theta_priors_cov]
    elif generate_params == 'mean_gabbi':
        # get mean and cov of log(initial prior)
        theta_priors = [np.abs(get_prior_param_distributions_gabbi(
            m, init_particles_per_model)) for m in mb_models]
        theta_priors = [np.atleast_2d(np.mean(tj, axis=0)) for tj in
                        theta_priors]
        theta_priors = [np.repeat(tp, init_particles_per_model, axis=0) for tp
                        in theta_priors]
        theta_priors_means = [np.mean(np.log(tj), axis=0) for tj in
                              theta_priors]
        theta_priors_cov = [np.cov(np.log(tj).T) for tj in theta_priors]
    else:
        raise ValueError('Value {} for parameter generation is not '
                         'allowed'.format(generate_params))

    if param_dict is not None:
        theta_priors = [np.atleast_2d(np.abs(np.array(
            [param_dict[m.__name__ + '_' + p] for p in m.cali_params_list])))
                        for m in mb_models]
        theta_priors = [np.repeat(tp, init_particles_per_model, axis=0) for tp
                        in theta_priors]
        theta_priors_means = [np.mean(np.log(tj), axis=0) for tj in
                              theta_priors]
        theta_priors_cov = [np.cov(np.log(tj).T) for tj in theta_priors]
        theta_priors_cov = [np.zeros_like(c) for c in theta_priors_cov]

    return theta_priors, theta_priors_means, theta_priors_cov


def prepare_observations(gdir, stations=None, ice_only=False,
                         exclude_initial_snow=True, cam_mb_path=None):
    """
    Prepares the camera observations for usage.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to prepare the observations for.
    stations: list, optional
        The station to retrieve the observations for. Default: None (all
        available stations on the glacier).
    ice_only: bool, optional
        Whether to consider ice melt observations only. Default: False
        (retrieve also snow accumulation).
    exclude_initial_snow: bool, optional
        Whether the initla phase where the camera is mounted on a snow surface
        shall be excluded. Default: True.
    cam_mb_path: str, optional
        Path to the camera mass balances. Default: None (retrieve from
        configuration).

    Returns
    -------
    obs_merge, obs_merge_cs: xr.Dataset, xr.Dataset
        Daily and cumulative observations from the cameras.

    """
    if stations is None:
        stations = id_to_station[gdir.rgi_id]
    station_hgts = [station_to_height[s] for s in stations]

    # 'c:\\users\\johannes\\documents\\holfuyretriever\\'
    if cam_mb_path is None:
        # todo: maybe put this into cfg
        cam_mb_path = os.path.join(cfg.PATHS['data_dir'], 'MB', 'cams')

    obs_merge = utils.prepare_holfuy_camera_readings(
        gdir, ice_only=ice_only, exclude_initial_snow=exclude_initial_snow,
        stations=stations, holfuy_path=cam_mb_path)
    obs_merge = obs_merge.sel(height=station_hgts)
    obs_merge_cs = obs_merge.swe.cumsum(dim='date')
    obs_merge_cs = obs_merge_cs.where(~np.isnan(obs_merge.swe), np.nan)

    return obs_merge, obs_merge_cs


def validation_run(stations, param_ts, weight_ts, model_weight_ts):
    """
    Run stations with given parameter and weights time series.

    Parameters
    ----------
    stations :
    param_ts :
    weight_ts :
    model_weight_ts :

    Returns
    -------

    """


def reference_run_constant_param(point_cali_result_path):
    """
    Perform a reference run with constant parameters.

    todo: this function is misplaced!

    Parameters
    ----------
    point_cali_result_path: str
        Path to calibrated parameters, when one observation ('intermediate
        reading') is used as a calibration source.

    Returns
    -------
    None
    """

    point_cali_results = pd.read_csv(point_cali_result_path)

    res_list = []
    mad_list = []
    for i in range(len(point_cali_results)):
        rdict = point_cali_results.iloc[i].to_dict()
        gdir = utils.GlacierDirectory(rdict['RGIId'])
        fl_h, fl_w = gdir.get_inversion_flowline_hw()
        stations = id_to_station[gdir.rgi_id]
        station_hgts = [station_to_height[s] for s in stations]
        obs_merge = utils.prepare_holfuy_camera_readings(
            gdir, ice_only=False, exclude_initial_snow=True, stations=stations)
        obs_merge = obs_merge.sel(height=station_hgts)
        quadro_result = make_mb_current_mbyear_heights(
            gdir, begin_mbyear=pd.Timestamp('2018-10-01'), param_dict=rdict,
            write=False, reset=False)
        res_list.append(quadro_result[0])
        s_index = np.argmin(np.abs((fl_h - np.atleast_2d(station_hgts).T)),
                            axis=1)
        mb_at_obs = quadro_result[0].sel(fl_id=s_index).squeeze(['member'])
        obs_merge = obs_merge.rename({'date': 'time'})
        first_obs_times = obs_merge.time[
            np.argmin(np.isnan(obs_merge.swe), axis=1)].values
        mad_per_obs = []
        for f, h, d in zip(mb_at_obs.fl_id.values, obs_merge.height.values,
                           first_obs_times):
            obs_merge_sel = obs_merge.sel(height=h).swe.cumsum(dim='time')
            mb_at_obs_sel = mb_at_obs.sel(fl_id=f,
                                          time=slice(d, None)).MB.cumsum(
                dim='time')
            mad_per_obs.append(
                np.median(np.abs(obs_merge_sel - mb_at_obs_sel)))

        # mad = np.abs(mb_at_obs.MB.mean(dim='model') - obs_merge.swe)
        mad = np.mean(mad_per_obs)
        print('MEAN of MEDIAN ABSOLUTE DEVIATIONS: ', mad)
        mad_list.append(mad)
        fig, ax = plt.subplots()
        obs_merge_sel.plot(ax=ax)
        mb_at_obs_sel.mean(dim='model').plot(ax=ax)


def station_init_conditions_at_first_setup(station, year=2019):
    """
    When the first camera is set up on a glacier, this gives an estimate of
    the init conditions also at other camera locations.

    The alternative is to calculate forward with the function that produces
    mb_current.

    Parameters
    ----------
    station : int
         Station ID.
    year: int
         Year when camera was set up.

    Returns
    -------
    init_cond : dict
         Dictionary with initial conditions.
    """

    # todo: include year

    ic = {
        1001: {'mb': (0., 0.), 'alpha': 'from_model', 'tacc': 'from_model',
               'swe': (0.9, 0.15)},
        1002: {'mb': (0., 0.), 'alpha': 'from_model', 'tacc': 'from_model',
               'swe': 'from_model'},
        1003: {'mb': (0., 0.), 'alpha': 'from_model', 'tacc': 'from_model',
               'swe': 'from_model'},
        1006: {'mb': (0., 0.), 'alpha': 'from_model', 'tacc': 'from_model',
               'swe': 'from_model'},
        1007: {'mb': (0., 0.), 'alpha': 'from_model', 'tacc': 'from_model',
               'swe': 'from_model'},
        1008: {'mb': (0., 0.), 'alpha': 'from_model', 'tacc': 'from_model',
               'swe': 'from_model'},
        1009: {'mb': (0., 0.), 'alpha': 'from_model', 'tacc': 'from_model',
               'swe': 'from_model'},
    }


def get_background_cov_canadian(x_t, diff_period=6):
    """
    Canadian quick method to get background error covariance from model run.

    The method assumes that the "truth" appearing in an error definition
    :math:`\epsilon = \mathbf{x}^b - \mathbf{x}^t` needed to calculate
    a background error covariance matrix can be approximated as follows (see
    [1]_):

    .. math::
        \mathbf{x}^b - \mathbf{x}^t \sim \frac{\mathbf{x}^b(t + T) -
        \mathbf{x}^b(T)}{\sqrt{2}}

    # todo: the SQRT in the equation comes from Ross' slides...why is it there?

    Parameters
    ----------
    x_t: (sample_size, N) np.ndarray
        Array with free model run. First dimension is `sample_size`,
        i.e. number of time steps the model has run, second dimension is the
        model states at these times.
    diff_period: int
        Differencing period `T` which is used to form the differences of
        fields. `T` is in time step units of the model given (here: usually
        one day). Default: 6 (see [1]_).

    Returns
    -------
    B: np.matrix
        Background error covariance matrix.

    References
    ----------
    .. [1] Polavarapu S., Ren S., Rochon Y., Sankey D., Ek N., Koshyk J.,
        Tarasick D., Data assimilation with the Canadian middle atmosphere
        model. Atmos.-Ocean 43: 77-100 (2005).
    """
    total_steps, sam_size = x_t.shape

    ind_sample_0 = range(0, total_steps - diff_period)
    ind_sample_plus = ind_sample_0 + np.repeat(diff_period,
                                               total_steps -
                                               diff_period) - 1
    x_sample = x_t[ind_sample_0, :] - x_t[ind_sample_plus, :]
    B = np.mat(np.cov(x_sample, rowvar=False))
    return B


def get_background_cov_simple(x_t):
    """
    A very simple method to obtain the climatological background error
    covariance.

    Obtained from a long run of a model.

    Parameters
    ----------
    x_t: (sample_size, N) np.ndarray
        Array with free model run. First dimension is `sample_size`,
        i.e. number of time steps the model has run, second dimension is the
        model states at these times.

    Returns
    ----------
    B: np.matrix
        The background error covariance matrix.
    """
    raise NotImplementedError
    run_range = pd.date_range()
    total_steps = len(run_range)
    tmax = run_range[-1]
    x0 = np.array([-10, -10, 25])
    _, xt = model(x0, tmax)
    samfreq = 16
    err2 = 2

    # Precreate the matrix
    ind_sample = range(0, total_steps, samfreq)
    x_sample = xt[ind_sample, :]
    B = np.mat(np.cov(x_sample, rowvar=0))
    alpha = err2 / np.amax(np.diag(B))
    B = alpha * B

    return B


def variogram_function_hock(lag_distance: float or np.ndarray) -> float or \
                                                                  np.ndarray:
    """
    Get covariance from the variogram fitted in [1]_.

    the variogram is calculated using 40-60 mass balance stake measurements on
    Storglaciaeren, fitting the variogram power function:

    .. math::
        \gamma^* = a \cdot h^b + n

    where :math:`\gamma^*` is the semi-variance approximated by the estimator
    of Matheron, a and b are fitting parameters, h is lag distance and n is a
    fitting parameter called the nugget effect.

    Parameters
    ----------
    lag_distance: float or np.ndarray
        Lag distances between to measurements giving the according covariance.

    Returns
    -------
    cov: float or np.ndarray
        The according covariances between mass balance measurements at the
        given lag distance.

    References
    ----------
    .. [1] Hock, R. & Jensen, H.: Application of Kriging Interpolation for
           Glacier Mass Balance Computations. Geografiska Annaler: Series A,
           Physical Geography, 1999, 81, pp. 611-619.
    """
    # fitted values from Regine
    a = None
    b = None
    n = None
    cov = a * lag_distance ** b + n

    return cov


def krige_camera_mb(mb, lag_distances, variogram_func=variogram_function_hock):
    """
    Actually krige an camera mass balance observation.

    Parameters
    ----------
    mb :
    lag_distances :
    variogram_func :

    Returns
    -------
    values, stdev: tuple of np.array
        Values kriged to the lag distances and their respective standard
        deviations.
    """


def predict_particle_filter(particles: np.ndarray, pred: np.ndarray,
                            pred_std: np.ndarray) -> np.ndarray:
    """
    Predict step of the particle filter, using already given inputs.

    Parameters
    ----------
    particles: np.array
        Array with state particles.
    pred: np.array
        Model mean prediction of one time step.
    pred_std: np.array
        Model error as standard deviation.

    Returns
    -------
    particles: np.array
        Randomly perturbed particles, using a normal distribution of the model
        prediction and its errors.
    """
    mb_dist = pred + (np.random.randn(*particles.shape) * pred_std)
    particles += mb_dist
    return particles


def create_gaussian_particles(mean: float, std: float, n: tuple) -> np.ndarray:
    """
    Create n-dim set of random numbers from the standard normal distribution.

    Parameters
    ----------
    mean: np.array
        Mean of the gaussian from which particle shall be generated.
    std: np.array
        Standard deviation of the gaussian from which particles shall be
        generated.
    n: tuple
        Dimensions of the particle array to be drawn from the gaussian.

    Returns
    -------
    particles: np.ndarray with shape n
         Random particles generated from a gaussian distribution.
    """
    particles = mean + (np.random.randn(*n) * std)
    return particles


def create_particles_kde(values: np.ndarray, n: int, kernel: str = 'gaussian',
                         bw: float = 1.) -> np.ndarray:
    """
    Create n-dim set of random numbers by Kernel density estimation of values.

    Parameters
    ----------
    values: (N, M) array
        Array of values to extend to shape (n, M)
    n: int
        Number of samples to draw.
    kernel: str
        Any kernel allowed by sklearn.neighbors.KernelDensity. One in
        ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
        (scipy 0.21.2). Default: 'gaussian'.
    bw: float
        Kernel bandwidth (see _[1]).

    Returns
    -------
    np.ndarray with shape (n, M)
         Random particles generated by kernel density estimation.

    References
    ----------
    _[1]: https://bit.ly/2SdRJuz
    """

    kde = np.apply_along_axis(
        lambda x: KernelDensity(kernel=kernel, bandwidth=bw).fit(
            x[:, np.newaxis]), 0, values)
    particles = np.apply_along_axis(
        lambda x: x[0].sample(n_samples=n, random_state=0), 1,
        np.atleast_2d(kde).T)[:, :, 0].T
    return particles


def create_random_particles_choice(values: np.ndarray, n: int) -> np.ndarray:
    """
    Create n-dim set of random numbers by drawing N samples from values.

    Parameters
    ----------
    values: (N, M) array_like
        Array of values to randomly extend to shape (N, M)
    n: int
        Number of samples to draw.

    Returns
    -------
    np.ndarray with shape (n, M)
         Random particles generated from drawing samples from values.
    """

    row_selector = np.indices((n, values.shape[1]))[1]
    run_selector = np.random.randint(0, high=values.shape[0], size=(
        n, values.shape[1]))
    return values[run_selector, row_selector]


def distribute_point_mb_measurement_on_heights(fl_heights: np.ndarray,
                                               model_mb: np.ndarray,
                                               obs: float, obs_hgt: float,
                                               obs_std: float,
                                               prec_sol_at_hgts: np.ndarray,
                                               unc_method: str = 'linear') \
        -> tuple:

    """
    Distribute a point mass balance measurement on all glacier heights.

    This function uses the mass balance gradient to distribute a given point
    mass balance measurement on all glacier heights. It uses the mass
    balance gradient from the model. In the
    default setting, the measurement uncertainty changes linearly with height,
    but also no change with height or inverse distance weighting is possible.
    To distribute the measurement, it uses the mass balance gradient from the
    model.

    # to do: does it make sense to distribute the measurement on all flowline
    catchments (different aspect!?) or just on the one where the measurement
    is?
    # to do: would it make sense to keep the ratio between model and
    measurement uncertainty (determined at the measurement site) constant on
    all heights?

    Parameters
    ----------
    fl_heights: np.array
        Array with flowline heights, flattened.
    model_mb: np.array
        Mass balance as modeled by the ensemble members (m w.e.).
    obs: float
        Mass balance measurement, converted to m w.e.. (m w.e.).
    obs_hgt: float
        Height at which mass balance measurement was made.
    obs_std: float
        The measurement uncertainty given as the standard deviation (m w.e.).
    prec_sol_at_hgts: np.array
        Solid precipitation at the flowline heights.
    unc_method: str
        Method to distribute the uncertainty with height. Allowed are 'linear'
        for a linear decrease of the observation uncertainty weight with height
        distance, 'idw' for and inverse distance weighting of the observation
        uncertainty with height. If None is passed, the uncertainty does not
        change with height. Default: 'linear'.

    Returns
    -------
    meas_hgt_mean, meas_hgt_std: np.array, np.array
        All extrapolated measurements and their uncertainties at all input
        heights given.
    """

    # m_index = np.argmin(np.abs(fl_heights - obs_hgt))
    m_index = np.argmin(np.abs(fl_heights - np.atleast_2d(obs_hgt).T), axis=1)

    # delta_h = fl_heights - fl_heights[m_index]
    # mb_diff = (model_mb[:, m_index] - model_mb.T).T
    delta_h = fl_heights - np.atleast_2d(fl_heights[m_index]).T
    mb_diff = (np.atleast_3d(model_mb[:, m_index].T) - np.atleast_3d(
        model_mb.T).T).T

    # distribute the obs along the mb gradient
    # obs_distr = obs - np.nanmedian(mb_diff, axis=0)
    obs_distr = np.atleast_2d(obs) - np.nanmedian(mb_diff, axis=1)

    # ensure that e.g. mass balance doesn't become positive without precip!
    # todo: are there more cases to cover?
    if (model_mb >= 0.0).any() and (model_mb < 0.0).any():
        obs_distr = np.clip(obs_distr, None, 0.)

    # same if distributed crosses zero at a certain point
    if (obs_distr >= 0.0).any() and (obs_distr < 0.0).any() and \
            (prec_sol_at_hgts == 0.).any():
        obs_distr[prec_sol_at_hgts == 0., :] = \
            np.clip(obs_distr[prec_sol_at_hgts == 0., :], None, 0.)

    # Distribute the stdev of the measurement (becomes bigger with dh!?)
    # todo: test multiple
    mod_std = np.std(mb_diff, axis=1)
    # mod_std = np.std(mb_diff, axis=0)
    if unc_method == 'linear':
        meas_std_distr = distribute_std_by_linear_distance(
            delta_h, np.atleast_2d(obs_std).T, mod_std.T)
    elif unc_method == 'idw':
        meas_std_distr = distribute_std_by_inverse_distance(
            delta_h, obs_std, mod_std)
    elif unc_method is None:
        meas_std_distr = np.ones_like(mod_std, dtype=np.float32) * obs_std
    else:
        raise ValueError('Method for extrapolating uncertainty is not '
                         'supported.')

    return obs_distr, meas_std_distr


def distribute_std_by_linear_distance(dist: np.ndarray, obs_std: float,
                                      mod_std: float) -> np.ndarray:
    """
    Distribute the standard deviation of observations linearly with distance.

    At the measurement site, the measurement standard deviation gets the weight
    one and the model ensemble standard deviation gets the weight zero. At the
    most distant point of the glacier (in terms of height distance, not
    horizontally!), it is the other way around.

    Parameters
    ----------
    dist: np.array
        Vertical distance of the heights where mass balance is modeled to the
        measurement site height.
    obs_std: float
        Standard deviation of the observation (m w.e.).
    mod_std: np.array
        Standard deviation of the model ensemble at the heights where mass
        balance is modeled.

    Returns
    -------
    std_distr: np.array
        Standard deviation distributed on the heights where mass balance is
        modeled.
    """

    dist = np.abs(dist)
    dist[dist == 0.] = 1.e-300  # avoid division by zero
    wgt = np.nanmax(dist) - dist

    # Make weights sum to one
    wgt /= np.nanmax(dist)

    # Multiply the weights for each interpolated point by all observed Z-values
    # todo: test multiple
    # std_distr = obs_std * weights + mod_std * (1 - weights)
    std_distr = np.atleast_2d(obs_std) * wgt + mod_std * (1 - wgt)

    return std_distr


def distribute_std_by_inverse_distance(dist: np.ndarray, obs_std: float,
                                       mod_std: np.ndarray) -> np.ndarray:
    """
    Distribute the standard deviation of the observation weighted by inverse
    distance the measurement site.

    At the measurement site, the measurement standard deviation gets the weight
    one and the model ensemble standard deviation gets the weight zero. At the
    most distant point of the glacier (in terms of height distance, not
    horizontally!), it is the other way around. In between, there is a

    Parameters
    ----------
    dist: np.array
        Vertical distance of the heights where mass balance is modeled to the
        measurement site height.
    obs_std: float
        Standard deviation of the observation (m w.e.).
    mod_std: np.array
        Standard deviation of the model ensemble at the heights where mass
        balance is modeled.

    Returns
    -------
    std_distr: np.array
        Standard deviation distributed on the heights where mass balance is
        modeled.
    """

    dist = np.abs(dist)

    zero_ix = np.where(dist == 0.)
    dist[zero_ix] = np.nan  # avoid division by zero

    wts = 1.0 / (dist / np.nanmax(dist))
    wts /= np.nanmax(wts)  # weight one is at the closest neighbor
    # (avoid inf)

    # insert 1 back in
    wts[zero_ix] = 1.

    # Multiply the weights for each interpolated point by all observed Z-values
    std_distr = obs_std * wts + mod_std * (1 - wts)

    return std_distr


def get_possible_height_gradients(arr: np.ndarray, heights: np.ndarray,
                                  axis: int or None = None,
                                  shuffle: bool = True) -> np.ndarray:
    """
    A function to retrieve possible height gradients along an axis in an array.

    Parameters
    ----------
    arr: np.array
        The array for which to calculate the gradients.
    heights: np.array
        The heights for the individual nodes.
    axis: int
        Axis along which to calculate the gradients
    shuffle: bool
        If True, members are shuffled such that the amount of possible
        gradients is increased. Default: True.

    Returns
    -------
    grad: np.array
        Array with possible gradients. Shape is like arr if shuffle is False,
        otherwise
    """

    # Determine over which axis to calculate the gradient
    if arr.ndim > 1:
        # problem with square array when axis not given
        if (np.array([arr.shape[i] == arr.shape[i+1] for i, _ in
                      enumerate(arr.shape[:-1])]).all()) and (axis is None):
            raise ValueError('If array is square, axis must be given.')
        else:
            axis = [arr.shape.index(a) for a in arr.shape if heights.shape[0]
                    == a][0]

    # height difference between elements
    delta_h = np.gradient(heights)

    # if we want all possible gradients from the ensemble member, we need to
    # shuffle
    if shuffle is True:
        grad = None
        # shuffle only uses first dimension
        if arr.shape.index(heights.shape[0]) == 0:
            arr = arr.T.copy()
            transpose = True
        else:
            transpose = False

        for n in range(arr.shape[0]):
            if grad is None:
                grad = np.gradient(arr, delta_h, axis=axis)
            else:
                # shuffles in-place
                # np.random.shuffle(arr)
                disarrange(arr, axis=0)  # todo: axis=0 correct for every case?
                grad = np.vstack(grad, np.gradient(arr, delta_h, axis=axis))
        # check if we need to transpose back
        if transpose is True:
            grad = grad.T.copy()
    else:
        grad = np.gradient(np.array(arr), delta_h, axis=axis)

    return grad


def disarrange(a: np.ndarray, axis: int = 0):
    """
    Shuffle `a` in-place along the given axis, but each one-dimensional slice
    independently.
    From https://github.com/numpy/numpy/issues/5173
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


def make_mb_current_mbyear_particle_filter(
        gdir: utils.GlacierDirectory, begin_mbyear: pd.Timestamp,
        mb_model: DailyMassBalanceModel = None,
        snowcover: SnowFirnCover = None, write: bool = True,
        reset: bool = False, filesuffix: str = '') -> xr.Dataset:
    """
    Make the mass balance of the current mass budget year for a given glacier
    using measured data to assimilate into the.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to calculate the current mass balance for.
    begin_mbyear: datetime.datetime
        The beginning of the current mass budget year.
    mb_model: `py:class:crampon.core.models.massbalance.DailyMassBalanceModel`,
              optional
        A mass balance model to use. Default: None (use all available).
    snowcover:
    write: bool
        Whether or not to write the result to GlacierDirectory. Default: True
        (write out).
    reset: bool
        Whether to completely overwrite the mass balance file in the
        GlacierDirectory or to append (=update with) the result. Default: False
        (append).
    filesuffix: str
        File suffix for output MB file, used for experiments. Default:'' (no
        suffix).

    Returns
    -------
    mb_now_cs: xr.Dataset
        Mass balance of current mass budget year as cumulative sum.
    """

    yesterday = min(utils.get_cirrus_yesterday(),
                    begin_mbyear+pd.Timedelta(days=366))
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    begin_str = begin_mbyear.strftime('%Y-%m-%d')

    curr_year_span = pd.date_range(start=begin_str, end=yesterday_str,
                                   freq='D')

    heights, widths = gdir.get_inversion_flowline_hw()
    # make a NaN-filled array
    startarray = np.full((len(heights), len(curr_year_span) + 1), np.nan)

    # mass balance of the models
    mb_mod_all = copy.deepcopy(startarray)
    mb_mod_std_all = copy.deepcopy(startarray)

    # assimilated mass balance
    mb_all_assim = copy.deepcopy(startarray)
    mb_all_assim_std = copy.deepcopy(startarray)

    # observed mass balance as water equivalent
    obs_we_extrap_all = copy.deepcopy(startarray)
    obs_we_std_extrap_all = copy.deepcopy(startarray)

    # start off cumsum with zero
    mb_mod_all[:, 0] = 0.
    mb_all_assim[:, 0] = 0.

    stations = id_to_station[gdir.rgi_id]
    obs_merge = utils.prepare_holfuy_camera_readings(gdir)

    # todo: minus to revert the minus from the function! This is shit!
    obs_std_we_default = - utils.obs_operator_dh_stake_to_mwe(
        cfg.PARAMS['dh_obs_manual_std_default'])
    station_hgts = [station_to_height[s] for s in stations]
    s_index = np.argmin(np.abs((heights -
                                np.atleast_2d(station_hgts).T)), axis=1)

    weights = None
    N_particles = 10000  # makes it a bit faster
    n_eff_thresh = get_effective_n_thresh(N_particles)

    mb_spec_assim = np.full((100, len(curr_year_span) + 1), np.nan)
    mb_spec_assim[:, 0] = 0.

    tasks.update_climate(gdir)
    _ = make_mb_current_mbyear_heights(
        gdir, begin_mbyear=begin_mbyear, reset=True, filesuffix=filesuffix)

    gmeteo = climate.GlacierMeteo(gdir)

    mb_mod = gdir.read_pickle('mb_current_heights' + filesuffix)
    mb_mod_stack = mb_mod.stack(ens=('member', 'model'))
    mb_mod_stack_ens = mb_mod.stack(ens=('member', 'model')).MB.values

    mb_mod_all = mb_mod_stack.median(dim=['ens'], skipna=True).MB.values
    mb_mod_std_all = mb_mod_stack.std(dim=['ens'], skipna=True).MB.values
    # todo: once scipy is updated to 1.3.0, use this:
    # mb_mod_std_all = mb_mod_stack.apply(stats.median_absolute_deviation,
    # axis=?, nan_policy='ignore')

    mbscs = mb_mod_stack.cumsum(dim='time', skipna=True)
    mbscs = mbscs.where(~np.isnan(mb_mod_stack))
    mbscs = mbscs.where(mbscs != 0.)
    mb_mod_cumsum_std_all = mbscs.std(dim='ens', skipna=True).MB.values
    # todo: once scipy is updated to 1.3.0, replace this:
    # arr = mb_mod.sel(fl_id=s_index[0]).cumsum(dim='time', skipna=True).stack(
    #    dim=['member', 'model']).MB.values
    # scale = 1.4826
    # med = np.apply_over_axes(np.nanmedian, arr, 1)
    # mad = np.median(np.abs(arr - med), axis=1)
    # mb_mod_cumsum_std_all  = scale * mad
    mb_mod_cumsum_all = mbscs.cumsum(dim='ens', skipna=True).MB.values

    # mb_mod_cumsum_std_all = mb_mod_stack.cumsum(
    #    dim='time',  # this is to test if we have to use the cumsum std
    #    skipna=True).std(dim='ens', skipna=True).MB.values

    t_index = mb_mod.time.indexes['time']
    first_date_ix = None
    first_date = None

    bias_list = []
    particles_old = np.zeros((N_particles, len(heights)))
    for n_day, date in enumerate(curr_year_span):

        date_ix = t_index.get_loc(date)
        print('\n\n\nDate:', date, 'date_ix: ', date_ix)

        mb_mod_date = mb_mod_all[:, date_ix]
        mb_mod_std_date = mb_mod_std_all[:, date_ix]

        # get the daily bias
        if (date_ix == first_date_ix) or (first_date_ix is None):
            bias = np.zeros_like(s_index, dtype=float)
        else:
            bias = np.nanmean(
                mb_mod_all[s_index, max(date_ix - 10, 0):date_ix + 1], axis=1)\
                   - np.nanmean(
                obs_merge.sel(date=slice(max(first_date,
                                             date - dt.timedelta(days=10)),
                                         date)).swe.values, axis=1)

        # if date has an OBS and that OBS is not NaN
        if (np.datetime64(date) in obs_merge.date) and ~np.isnan(
                obs_merge.sel(date=date).swe.values).all():

            # set variables for the first day with OBS ever
            if first_date_ix is None:
                first_date_ix = t_index.get_loc(date)
            if first_date is None:
                first_date = date

            if weights is None:  # initiate
                weights = np.ones((N_particles, len(heights)),
                                  dtype=np.float32)
                particles = create_gaussian_particles(
                    np.atleast_2d(mb_mod_date),
                    # todo: change back?
                    np.atleast_2d(mb_mod_std_date),
                    # np.atleast_2d(mb_mod_cumsum_std_date),
                    (N_particles, len(heights)))
            # there was already a day with OBS before and weights is not None:
            else:  # predict with model + uncertainty
                # trick: add std deviation randomly to all particles
                # strictly speaking for each particles in a row randn should
                # be ran separately, but particle number should be big enough
                # to make everything random
                print('MODEL:', mb_mod_date[s_index[0]],
                      mb_mod_std_date[s_index[0]])

                # todo: correct mean and std if mb is zero => this should be
                #  base on how close temperature is to zero deg; the further
                #  away, the smaller the MB
                if (mb_mod_date == 0.).any():
                    possible_maxtemp = \
                        gmeteo.get_tmean_at_heights(date) + \
                        climate.interpolate_mean_temperature_uncertainty(
                            np.array([date.month]))
                    positive_part = np.clip(possible_maxtemp, 0, None)
                    possible_melt = \
                        BraithwaiteModel.cali_params_guess['mu_ice'] / 1000. *\
                        positive_part
                    # set all 0s to min value (we need stddev for the method)
                    possible_melt[possible_melt == 0.] = 0.0001
                    mb_mod_std_date[mb_mod_date == 0.] = \
                        possible_melt[mb_mod_date == 0.]

                # make the predict step
                particles_old = copy.deepcopy(particles)
                particles = predict_particle_filter(
                    particles, mb_mod_date, mb_mod_std_date)

            mean_plot = np.nanmean(particles[:, s_index])
            std_plot = np.nanstd(particles[:, s_index])
            model_violin = particles[:, s_index[0]]
            print('Particle mean/std after predict: ', mean_plot, std_plot)

            obs_we = mb_all_assim[s_index, date_ix - 1] + obs_merge.sel(
                date=date).swe.values + bias
            # todo: this is new for truncate
            obs_we_old = mb_all_assim[s_index, date_ix - 1]
            bias_list.append(bias)
            print('bias: ', bias)
            obs_std_we = obs_merge.sel(date=date).swe_std.values

            # check if there is given uncertainty, otherwise switch to default
            if np.isnan(obs_std_we).any():
                obs_std_we[np.where(np.isnan(obs_std_we))] = obs_std_we_default

            if np.isnan(weights).any():
                print('THERE ARE NAN WEIGHTS')

            # iterate over stations, find new weights for each, and take mean
            new_weights = np.zeros((particles.shape[0], len(s_index)))
            for i, s in enumerate(s_index):
                checka = copy.deepcopy(weights[:, s])
                new_weights[:, i] = update_particle_filter(
                    particles[:, s], checka, obs_we[i], obs_std_we[i],
                    truncate=particles_old[:, s] - obs_we[i],
                    obs_old=obs_we_old[i])  # todo: this is new for truncate
            new_weights = np.nanmean(new_weights, axis=1)
            new_weights = np.repeat(np.atleast_2d(new_weights).T,
                                    weights.shape[1], axis=1)

            weights = copy.deepcopy(new_weights)
            if np.isnan(weights).any():
                raise ValueError('Some weights are NaN.')

            # estimate new state
            mb_mean_assim, mb_var_assim = estimate_state(particles, weights)
            mb_std_assim = np.sqrt(mb_var_assim)

            mb_all_assim[:, n_day + 1] = mb_mean_assim
            mb_all_assim_std[:, n_day+1] = mb_std_assim

            # resample the distribution if necessary
            particles, weights = resample_particles(
                particles, weights, n_eff_thresh=n_eff_thresh)
            print('UNIQUE PARTICLES: ',
                  len(np.unique(particles[:, s_index[0]])))
            print('MEAN: ', np.mean(particles[:, s_index[0]]))

            mb_spec_assim[:, n_day + 1] = \
                np.nanpercentile(np.average(np.sort(particles, axis=0),
                                            weights=widths, axis=1),
                                 np.arange(0, 100)) + \
                mb_spec_assim[:, first_date_ix]
        # if date does not have OBS or OBS is NaN
        else:
            if weights is None:
                mb_all_assim[:, n_day + 1] = mb_mod_date
                mb_all_assim_std[:, n_day+1] = mb_mod_std_date
                mb_spec_assim[:, n_day + 1] = np.nanpercentile(
                    np.average(mbscs.sel(time=date).MB.values, weights=widths,
                               axis=0), np.arange(0, 100))

            # all other days
            else:
                # write already the MB for the next day here
                particles_prediction = predict_particle_filter(
                    particles, mb_mod_date, mb_mod_std_date)
                # after prediction, weights should be one again, so no average
                mb_all_assim[:, n_day+1] = np.mean(
                    particles_prediction, axis=0)
                mb_all_assim_std[:, n_day+1] = np.std(particles, axis=0)
                mb_spec_assim[:, n_day + 1] = np.nanpercentile(
                    np.average(mb_mod_stack.sel(time=date).MB.values,
                               weights=widths, axis=0),
                    np.arange(0, 100)) + mb_spec_assim[:, n_day]
    if date_ix + 1 < 365:
        x_len = date_ix + 1
    else:
        x_len = 365

    # fake legend for violin plot
    blue_patch = mpatches.Patch(color='blue')
    green_patch = mpatches.Patch(color='green')
    darkgr_patch = mpatches.Patch(color='red')
    fake_handles = [blue_patch, green_patch, darkgr_patch]
    fake_labels = ['ASSIM', 'MODEL', 'OBS']
    plt.legend(fake_handles, fake_labels, fontsize=20)
    plt.grid()

    for s in np.arange(len(s_index)):
        fig, ax = plt.subplots()
        ax.errorbar(np.arange(x_len),
                    np.cumsum(mb_mod_all[s_index[s]])[:x_len],
                    mb_mod_cumsum_std_all[s_index[s]][:x_len],
                    elinewidth=0.5, label='model')
        both = np.cumsum(mb_mod_all[s_index[s]])[:x_len]
        both_std = mb_mod_cumsum_std_all[s_index[s]][:x_len]
        both[first_date_ix:] = mb_all_assim[s_index[s], first_date_ix:x_len] +\
            both[first_date_ix - 1]
        both_std[first_date_ix:] = mb_all_assim_std[s_index[s],
                                   first_date_ix:x_len] + \
                                   both_std[first_date_ix - 1]
        ax.errorbar(np.arange(x_len) + 0.5, both, both_std, elinewidth=0.5,
                    label='assimilated')
        plt.xlabel('DOY of mass budget year (OCT-SEP)', fontsize=20)
        plt.ylabel('MB (m w.e.)', fontsize=20)
        plt.setp(ax.get_xticklabels(), fontsize=20)
        plt.setp(ax.get_yticklabels(), fontsize=20)

        plt.legend(fontsize=20, loc='upper right')
        plt.grid()

    plt.figure()
    from crampon import graphics
    clim = gdir.read_pickle('mb_daily')
    mb_for_ds = np.atleast_3d(mb_spec_assim)
    mb_for_ds = np.moveaxis(np.atleast_3d(mb_for_ds), 1, 0)[:366, :, :]
    mb_for_ds = mb_for_ds[1:, :, :] - mb_for_ds[:-1, :, :]
    curr = xr.Dataset({'MB': (['time', 'member', 'model'], mb_for_ds)},
                      coords={'member': (['member', ],
                                         np.arange(mb_for_ds.shape[1])),
                              'model': (['model', ],
                                        ['AssimilatedMassBalanceModel']),
                              'time': (['time', ],
                                       pd.to_datetime(curr_year_span[:365])),
                              })
    gdir.write_pickle(curr, 'mb_assim' + filesuffix)
    plt.figure()
    graphics.plot_cumsum_climatology_and_current(gdir, clim=clim, current=curr)
    plt.show()

    return mb_all_assim

    """
    mean_for_ds = np.moveaxis(np.atleast_3d(np.array(mb_all_assim)), 1, 0)
    std_for_ds = np.moveaxis(np.atleast_3d(np.array(mb_all_assim)), 1, 0)
    # todo: units are hard coded and depend on method used above
    mb_assim_ds = xr.Dataset({'mb_mean': (['time', 'fl_id'], mean_for_ds),
                              'mb_std': (['time', 'fl_id'], std_for_ds),
                              'particles': (['fl_id'], particles),
                              'weights': (['fl_id'], particles),},
                       coords={'time': (['time', ],
                                        pd.to_datetime(curr_year_span)),
                               'fl_id': (['fl_id',], np.arange(len(heights)))},
                       attrs={'products': 'Manual dh stake readings'})

    ds_list.append(mb_assim_ds)

    ens_ds = xr.merge(ds_list)
    ens_ds.attrs.update({'id': gdir.rgi_id, 'name': gdir.name,
                         'units': 'm w.e.'})

    if write:
        ens_ds.mb.append_to_gdir(gdir, 'mb_current', reset=reset)

    # check at the point where we cross the MB budget year
    if dt.date.today == (begin_mbyear + dt.timedelta(days=1)):
        mb_curr = gdir.read_pickle('mb_current')
        mb_curr = mb_curr.sel(time=slice(begin_mbyear, None))
        gdir.write_pickle(mb_curr, 'mb_current')

    return ens_ds
    """


def step_particle_filter_cameras(gdir: utils.GlacierDirectory,
                                 date: pd.Timestamp,
                                 pf: ParticleFilter or None = None,
                                 n_particles: int = 10000,
                                 prediction: np.ndarray = None,
                                 models: list = None,
                                 predict_method: str = 'random',
                                 update_method: str = 'gauss',
                                 **kwargs: dict) -> ParticleFilter:
    """
    Make step forward in time using camera mass balance observations, if any.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to run the Particle Filter for.
    date: pd.Timestamp
        Date for which to make the step
    pf: ParticleFilter or None
        The Particle Filter object to start with. Default: None (generate one)
    n_particles: int or None
        Number of particles to use in the Particle Filter, if no particle
        filter object can be read from somewhere. Default: 10000.
    prediction: (N, M) array-like, optional
        Model balance prediction (prior) for the date. If None, it will be
        calculated using the mass balance ensemble. Default: None.
    models: list of crampon.core.massbalance.MassBalanceModel
        If predictions are not given, this determines which mass balance models
        shall be used for the prediction. Default: None (use all in
        cfg.MASSBALANCE_MODELS).
    predict_method: str
        Method used for generating a prior from model values. Default:
        'random'.
    update_method: str
        Method used for updating the prior with a likelihood distribution.
        Default: 'gauss'.
    **kwargs: dict
            Other keywords passed to ParticleFilter.

    Returns
    -------
    None
    """
    # todo: implement cumulative MB

    # if no prediction given, generate one
    if prediction is None:
        h, w = gdir.get_inversion_flowline_hw()
        # todo: make a prediction mode
        ens_mbm = EnsembleMassBalanceModel(gdir)
        prediction = np.array(ens_mbm.get_daily_mb(h, date))
        prediction = create_random_particles_choice(prediction, n_particles)

    # try and read state from gdir. If not there, start from beginning
    try:
        state = gdir.read_pickle('mb_assim_heights').sel(
            time=date - pd.Timedelta(days=1))
        # some things need to be renamed
        state = state.rename(
            {'member': 'particle_id', 'MB': 'particles'}).squeeze('model')
        if 'weights' not in state.data_vars:
            state['weights'] = (('particle_id',), np.ones_like(
                state.particles.values) / state.particle_id.size)
        pf = ParticleFilter.from_dataset(state)
        n_particles = pf.n_particles
    except (FileNotFoundError, KeyError, AttributeError, ValueError):
        pf = ParticleFilter(n_particles=n_particles, particles=prediction)

    print(pf)

    # set the params
    for key, value in kwargs.items():
        setattr(pf, key, value)

    print(pf)

    # look for camera data
    date_obs = utils.prepare_holfuy_camera_readings(gdir).sel(date=date)
    obs = date_obs.swe.values
    obs_std = date_obs.swe_std.values

    # todo: adjust the uncertainty as going further away from the camera

    # step forward
    print('OBS/STD: ', obs, obs_std)
    pf.step(prediction, date, obs, obs_std, predict_method=predict_method,
            update_method=update_method)

    return pf


def plot_pdf_model_obs_post(obs: float, obs_std: float, mod: float,
                            mod_std: float, post: float,
                            post_std: float) -> None:
    """
    Plot fitted Gaussian distributions of means and standard deviations.

    This function is meant to compare the prior, likelihood and posterior
    distribution during the assimilation process by approximating them as
    Gaussians.

    Parameters
    ----------
    obs: float
        Observation mean.
    obs_std: float
        Observation standard deviation.
    mod: float
        Model mean.
    mod_std: float
        Model standard deviation.
    post: float
        Posterior mean.
    post_std: float
        Posterior standard deviation.

    Returns
    -------
    None
    """

    valmin = min(obs, mod, post)
    valmax = max(obs, mod, post)
    ptp = valmax - valmin
    x_range = np.linspace(valmin - 0.05 * ptp, valmax + 0.05 * ptp, 1000)

    plt.plot(x_range, stats.norm(obs, obs_std).pdf(x_range), label='OBS')
    plt.plot(x_range, stats.norm(post, post_std).pdf(x_range), label='POST')
    plt.plot(x_range, stats.norm(mod, mod_std).pdf(x_range), label='MODEL')
    plt.legend()
    plt.show()


def make_mb_current_mbyear_heights(
        gdir: utils.GlacierDirectory, begin_mbyear: pd.Timestamp,
        last_day: Optional[Union[dt.datetime, pd.Timestamp]] = None,
        mb_models: Optional[Union[DailyMassBalanceModel, list]] = None,
        snowcover: SnowFirnCover = None, write: bool = True,
        reset: bool = False, filesuffix: str = '',
        param_dict: Optional[dict] = None,
        constrain_bw_with_prcp_fac: bool = True) -> xr.Dataset:
    """
    Make the mass balance at flowline heights of the current mass budget year.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to calculate the current mass balance for.
    begin_mbyear: datetime.datetime
        The beginning of the current mass budget year.
    last_day: dt.datetime or pd.Timestamp or None
        Last day to calculate the mass balance for. Default: None (take
        "Cirrus yesterday").
    mb_models: `py:class:crampon.core.models.massbalance.DailyMassBalanceModel`,
              or list, optional
        A mass balance model to use. Default: None (use all available).
    snowcover: `py:class:crampon.core.models.massbalance.SnowFirnCover`
        The snow/firn cover to sue at the beginning of the calculation.
        Default: None (read the according one from the GlacierDirectory)
    write: bool
        Whether or not to write the result to GlacierDirectory. Default: True
        (write out).
    reset: bool
        Whether to completely overwrite the mass balance file in the
        GlacierDirectory or to append (=update with) the result. Default: False
        (append).
    filesuffix: str
        Suffix to be used for the mass balance calculation.
    param_dict: dict or None
        Dictionary with parameters to use for the mass balance calculation. If
        None, take all available parameters from past calibration. Default:
        None.
    constrain_bw_with_prcp_fac: bool, optional
        Whether to contrain the used parameters with the precipitation
        correction factor tuned on the winter mass balance. Default: True.

    Returns
    -------
    mb_now_cs: xr.Dataset
        Mass balance of current mass budget year.
    """

    if isinstance(mb_models, DailyMassBalanceModel):
        mb_models = [mb_models]
    elif isinstance(mb_models, list):
        pass
    elif mb_models is None:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]
    else:
        raise ValueError('Value mb_models {} not accepted.'.format(mb_models))

    if snowcover is None:
        snowcover = gdir.read_pickle('snow_daily' + filesuffix)

    if last_day is None:
        yesterday = utils.get_cirrus_yesterday()
    else:
        yesterday = last_day
    if begin_mbyear is not None:
        yesterday = min(yesterday, begin_mbyear+pd.Timedelta(days=366))
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    begin_str = begin_mbyear.strftime('%Y-%m-%d')

    print(begin_str, yesterday_str)
    curr_year_span = pd.date_range(start=begin_str, end=yesterday_str,
                                   freq='D')
    print(curr_year_span)
    ds_list = []
    heights, widths = gdir.get_inversion_flowline_hw()

    conv_fac = cfg.FLUX_TO_DAILY_FACTOR  # ((86400 * cfg.RHO) / cfg.RHO_W)  # m
    # ice s-1 to m w.e.
    # d-1

    sc_list_one_model = []
    sfc_obj_list_one_model = []
    alpha_list = []
    tacc_list = []
    for mbm in mb_models:
        stacked = None

        pg = ParameterGenerator(
            gdir, mbm, latest_climate=True,
            constrain_with_bw_prcp_fac=constrain_bw_with_prcp_fac,
            bw_constrain_year=begin_mbyear.year + 1, narrow_distribution=0.,
            output_type='array', suffix=filesuffix)

        if param_dict is None:
            param_prod = pg.from_single_glacier()
        else:
            param_prod = np.atleast_2d([param_dict[mbm.__name__+'_'+p] for p
                                        in mbm.cali_params_list])

        print(param_prod)

        sc = SnowFirnCover.from_dataset(snowcover.sel(model=mbm.__name__,
                                                      time=curr_year_span[0]))
        for params in param_prod:
            pdict = dict(zip(mbm.cali_params_list, params))
            if isinstance(mbm, utils.SuperclassMeta):
                day_model_curr = mbm(gdir, **pdict, snowcover=sc, bias=0.)
            else:
                day_model_curr = copy.copy(mbm)

            mb_temp = []
            for date in curr_year_span:

                # get MB an convert to m w.e. per day
                tmp = day_model_curr.get_daily_mb(heights, date=date)
                tmp *= conv_fac
                mb_temp.append(tmp)

            sc_list_one_model.append(np.nansum(day_model_curr.snowcover.swe,
                                               axis=1))
            sfc_obj_list_one_model.append(day_model_curr.snowcover)
            if hasattr(day_model_curr, 'albedo'):
                alpha_list.append(day_model_curr.albedo.alpha)
                tacc_list.append(day_model_curr.tpos_since_snowfall)

            if stacked is not None:
                stacked = np.vstack((stacked, np.atleast_3d(mb_temp).T))
            else:
                stacked = np.atleast_3d(mb_temp).T

        mb_for_ds = np.moveaxis(np.atleast_3d(np.array(stacked)), 1, 0)
        mb_for_ds.shape += (1,) * (4 - mb_for_ds.ndim)  # add dim for model
        mb_ds = xr.Dataset(
            {'MB': (['fl_id', 'member', 'time', 'model'], mb_for_ds)},
            coords={'member': (['member', ],
                               np.arange(mb_for_ds.shape[1])),
                    'model': (['model', ],
                              [day_model_curr.__name__]),
                    'time': (['time', ],
                             pd.to_datetime(curr_year_span)),
                    'fl_id': (['fl_id', ],
                              np.arange(len(heights))),
                    })

        ds_list.append(mb_ds)

    ens_ds = xr.merge(ds_list)
    # sort in the order we want it
    ens_ds = xr.concat([ens_ds.MB.sel(model=m)
                        for m in cfg.MASSBALANCE_MODELS if m in
                        ens_ds.coords['model']], dim='model')\
        .to_dataset(name='MB')
    ens_ds.attrs.update({'id': gdir.rgi_id, 'name': gdir.name,
                         'units': 'm w.e.'})

    if write:
        ens_ds.mb.append_to_gdir(gdir, 'mb_current_heights' + filesuffix,
                                 reset=reset)

        # check at the point where we cross the MB budget year
        if dt.date.today == (begin_mbyear + dt.timedelta(days=1)):
            mb_curr = gdir.read_pickle('mb_current')
            mb_curr = mb_curr.sel(time=slice(begin_mbyear, None))
            gdir.write_pickle(mb_curr, 'mb_current')

    # todo: this should actually account for correlation ob mb, swe and alpha
    return ens_ds, sc_list_one_model, alpha_list, tacc_list, \
           sfc_obj_list_one_model


def make_variogram(glob_path: str) -> None:
    """
    Plot a variogram of the Holfuy camera readings.

    Parameters
    ----------
    glob_path: str
        Path to the files with manual reading from the Holfuy cameras.

    Returns
    -------
    None
    """
    # 'C:\\users\\johannes\\documents\\holfuyretriever\\manual*.csv'
    flist = glob.glob(glob_path)
    meas_list = [utils.read_holfuy_camera_reading(f) for f in flist]
    meas_list = [m[['dh']] for m in meas_list]
    test = [m.rename({'dh': 'dh' + '_' + flist[i].split('.')[0][-4:]}, axis=1)
            for i, m in enumerate(meas_list)]
    conc = pd.concat(test, axis=1)
    corr_df = conc.cov()

    plt.figure()
    for s in all_stations:
        c = 'dh_' + str(s)
        distances = np.abs(
            station_to_height[int(c.split('_')[1])] - np.array(
                [station_to_height[int(i.split('_')[1])] for i in
                 corr_df.index]))
        print(distances)
        distances_sort = sorted(distances)
        print(distances_sort)
        dh_sorted_by_dist = [x for _, x in
                             sorted(zip(distances, corr_df[c].values))]
        # plt.figure()
        plt.plot(distances_sort[1:], dh_sorted_by_dist[1:])
        for i, d in enumerate(distances_sort):
            if i == 0:
                continue
            plt.text(d, dh_sorted_by_dist[i], '{}'.format(
                [x for _, x in sorted(zip(distances, corr_df.index.values))][
                    i].split('_')[1]))
        # plt.title('Correlogram for station {}'.format(str(s)))
        # plt.xlabel('Absolute height distance (m)')
        # plt.xlim(0, None)
        # plt.ylabel('$R^2$')
    plt.title('Correlogram for station {}'.format(str(s)))
    plt.xlabel('Absolute height distance (m)')
    plt.xlim(0, None)
    plt.ylabel('$R^2$')


def crossval_multilinear_regression(glob_path: str) -> None:
    """
    Cross-validate a prediction of mass balances at the cameras from
    multilinear regression

    Parameters
    ----------
    glob_path: str
        Path to the manual Holfuy camera readings.

    Returns
    -------
    None.
    """
    from scipy.spatial.distance import squareform, pdist

    use_horizontal_distance = True

    hdata_path = os.path.join(cfg.PATHS['data_dir'], 'holfuy_data.csv')

    # 'C:\\users\\johannes\\documents\\holfuyretriever\\manual*.csv'
    flist = glob.glob(glob_path)
    meas_list = [utils.read_holfuy_camera_reading(f) for f in flist]
    meas_list = [m[['dh']] for m in meas_list]
    test = [m.rename({'dh': 'dh' + '_' + flist[i].split('.')[0][-4:]}, axis=1)
            for i, m in enumerate(meas_list)]
    conc = pd.concat(test, axis=1)
    conc_drop = conc.dropna()

    if use_horizontal_distance is True:
        hvals = pd.read_csv(hdata_path)
        hvals = hvals.sort_values(by='Station')
        hdist = pd.DataFrame(
            squareform(pdist(hvals.loc[:, ['Easting', 'Northing']])),
            columns=hvals.Station.unique(), index=hvals.Station.unique())
    from sklearn.model_selection import LeavePOut
    labels = conc.columns.values
    leave_out = 1
    lpo = LeavePOut(leave_out)
    lpo.get_n_splits(labels)

    for train_index, test_index in lpo.split(labels):
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        X = conc_drop[train_labels]
        X = sm.add_constant(X)  # add an intercept
        y = conc_drop[test_labels]

        test_labels_int = [int(n.split('_')[1]) for n in test_labels]
        wgts = hdist[test_labels_int].values[
                   hdist[test_labels_int].values != 0.] / np.sum(
            hdist[test_labels_int].values[hdist[test_labels_int].values != 0.])
        # bring to correct shape
        wgts = np.repeat(np.atleast_2d(wgts), len(conc_drop), axis=0)

        model = sm.WLS(y, X, weights=wgts).fit()
        predictions = model.predict(X)
        print(model.summary())

        plt.figure()
        plt.plot(y, label='OBS')
        plt.plot(predictions, label='PRED')
        plt.legend()
        plt.title(' '.join(list(test_labels)) +
                  ' RMSE: ' + '{:.3f}'.format(
            np.sqrt(np.mean((predictions.values - y.values) ** 2))))


def variogram_cloud(conc, hdist) -> None:
    """
    Plot the variogram cloud as in [Bivand et al. 2013]_ (p. 220).

    Returns
    -------
    None

    References
    ----------
    ..[Bivand et al. 2013]: Bivand, R. S., Pebesma, E. J., Gomez-Rubio, V., &
        Pebesma, E. J. (2008). Applied spatial data analysis with R . New York:
        Springer.
    """
    fig, ax = plt.subplots()
    for k, row in conc.iterrows():
        for c1, c2 in permutations(conc.columns.values, 2):
            if ~np.isnan(row[c1]) and ~np.isnan(row[c2]):
                ax.scatter(hdist[int(c1.split('_')[1])][int(c2.split('_')[1])],
                           (row[c1] - row[c2]) ** 2, c='b', marker='+')
    ax.set_ylim(0, None)
    ax.set_title('Cloud variogram for all stations')
    ax.set_xlabel('Horizontal distance (m)')
    ax.set_ylabel('Squared difference ($m^2$)')


def simple_variogram(conc, hdist) -> None:
    """
    Plot a simple variogram.

    Returns
    -------
    None
    """

    round_to = 30000.
    variance_list = []
    dist_list = []
    for k, row in conc.iterrows():
        for c1, c2 in permutations(conc.columns.values, 2):
            if ~np.isnan(row[c1]) and ~np.isnan(row[c2]):
                variance_list.append((row[c1] - row[c2]) ** 2)
                dist_list.append(
                    hdist[int(c1.split('_')[1])][int(c2.split('_')[1])])
    h_list = []
    gamma_hat = []
    for h in np.unique(np.array(dist_list)):
        h_list.append(round(h / round_to) * round_to)
        variance_for_h = np.array(variance_list)[np.where(dist_list == h)[0]]
        N_h = len(variance_for_h)
        gamma_hat.append(1 / (2 * N_h) * np.sum(variance_for_h))
    plt.scatter(np.unique(h_list),
                [np.mean(np.array(gamma_hat)[np.where(h_list == huni)[0]]) for
                 huni in np.unique(h_list)], c='r')
    plt.show()


def sample_variogram_100_variations(conc, hdist) -> None:
    """
    Produces figure similar to 8.5 in [Bivand et al. (2013)]_, but only points.

    Returns
    -------
    None.

    References
    ----------
    ..[Bivand et al. 2013]: Bivand, R. S., Pebesma, E. J., Gomez-Rubio, V., &
        Pebesma, E. J. (2008). Applied spatial data analysis with R . New York:
        Springer.
    """
    variance_list = []
    dist_list = []
    for k, row in conc.iterrows():
        for c1, c2 in permutations(conc.columns.values, 2):
            if ~np.isnan(row[c1]) and ~np.isnan(row[c2]):
                variance_list.append((row[c1] - row[c2]) ** 2)
                dist_list.append(
                    hdist[int(c1.split('_')[1])][int(c2.split('_')[1])])
    h_list = []
    gamma_hat = []
    for h in np.unique(np.array(dist_list)):
        h_list.append(h)
        variance_for_h = np.array(variance_list)[np.where(dist_list == h)[0]]
        N_h = len(variance_for_h)
        gamma_hat.append(1 / (2 * N_h) * np.sum(variance_for_h))
    fig, (ax1, ax2) = plt.subplots(2)
    for n in range(100):
        change_indices = np.random.choice(len(gamma_hat), len(gamma_hat),
                                          replace=False)
        assert (np.array(gamma_hat) != np.array(gamma_hat)[
            change_indices]).any()
        ax1.scatter(h_list, np.array(gamma_hat)[change_indices], c='b')
        res = sm.OLS(np.array(gamma_hat)[change_indices],
                     sm.add_constant(h_list)).fit()
        ax2.scatter(res.rsquared, res.pvalues[1], c='b')
    ax1.set_ylim(0., None)
    res = sm.OLS(np.array(gamma_hat)[change_indices],
                 sm.add_constant(h_list)).fit()
    ax1.scatter(h_list, np.array(gamma_hat), c='r', label='True distribution')
    ax2.scatter(res.rsquared, res.pvalues[1], c='r', label='True distribution')
    ax1.set_title(
        'Sample variogram vs. 100 variograms for randomly re-allocated data')
    ax1.set_xlabel('Horizontal distance x (m)')
    ax1.set_ylabel('Semivariance $\hat{\gamma}$')
    ax2.set_title(
        'Linear Slope and p-value for variogram vs. 100 variograms for '
        'randomly re-allocated data')
    ax2.set_xlabel('Slope ($\dfrac{d \hat{\gamma}}{d x}$)')
    ax2.set_ylabel('p-value')
    ax1.legend()
    ax2.legend()
    plt.show()


def sample_variogram_from_residuals(conc, hdist) -> None:
    """
    Accounts for the fact that the mean of MB varies spatially.

    See eq. 8.5 in [Bivand et al. 2013]_.
    We first do a multilinear regression of mean temperature and SIS to predict
    dh, then we take the residuals of these fits to produce a variogram.

    Returns
    -------
    None

    References
    ----------
    ..[Bivand et al. 2013]: Bivand, R. S., Pebesma, E. J., Gomez-Rubio, V., &
        Pebesma, E. J. (2008). Applied spatial data analysis with R . New York:
        Springer.
    """

    # this alternative takes the model residuals as they are (no mean) and
    # calculates the mean only when they are subtracted: the point being: the
    # semivariance values are more reasonable (10e-4 instead of 10e-33)
    eps_list = []
    station_list = []
    for s in conc.columns.values:
        sid = station_to_glacier[int(s.split('_')[1])]
        s_hgt = station_to_height[int(s.split('_')[1])]
        gd = utils.GlacierDirectory(sid)
        gm = climate.GlacierMeteo(gd)
        sis = gm.meteo.sel(time=conc.index).sis
        tmean = gm.meteo.sel(time=conc.index).temp
        tgrad = gm.meteo.sel(time=conc.index).tgrad
        tmean_at_hgt = tmean + tgrad * (s_hgt - gm.ref_hgt)
        notnan_ix = \
            np.where(~np.isnan(sis) & ~np.isnan(tmean_at_hgt) & ~pd.isnull(
                conc[s]))[0]
        y = conc[s][notnan_ix]
        X = sm.add_constant(
            np.vstack((sis[notnan_ix], tmean_at_hgt[notnan_ix])).T)
        model = sm.OLS(y, X).fit()
        eps_list.append(model.resid)
        station_list.append(s)
    variance_list = []
    dist_list = []
    for c1, c2 in permutations(station_list, 2):
        variance_list.append(np.square(
            np.mean(eps_list[station_list.index(c1)] - eps_list[
                station_list.index(c2)])))
        dist_list.append(hdist[int(c1.split('_')[1])][int(c2.split('_')[1])])
    h_list = []
    gamma_hat = []
    for h in np.unique(np.array(dist_list)):
        h_list.append(h)
        variance_for_h = np.array(variance_list)[np.where(dist_list == h)[0]]
        N_h = len(variance_for_h)
        gamma_hat.append(1 / (2 * N_h) * np.sum(variance_for_h))
    plt.figure()
    for i, h_i in enumerate(h_list):
        plt.scatter(h_list[i], gamma_hat[i], c='r')
    plt.ylim(min(gamma_hat) - 0.1 * min(gamma_hat),
             max(gamma_hat) + 0.1 * max(gamma_hat))


def camera_randomforestregressor(X) -> None:
    """
    Use a random forest regressor to predict camera values.

    Returns
    -------
    None
    """
    from sklearn.ensemble import RandomForestRegressor

    # train with LOO strategy
    meas = np.empty_like(X)
    for i_col, drop_col in enumerate(X.columns.values):
        y = X[[drop_col]]
        for n in range(len(X)):
            rf = RandomForestRegressor(n_estimators=1000, random_state=42)
            rf.fit(X.drop(X.index[n]).drop([drop_col], axis=1),
                   y.drop(y.index[n]))
            # print(X.drop(X.index[n]).drop([drop_col], axis = 1),
            #       y.drop(y.index[n]))
            predictions = rf.predict(
                np.atleast_2d(X.drop([drop_col], axis=1).iloc[n]))
            # print(y.iloc[n])
            errors = abs(predictions - y.iloc[n])
            # print(predictions, y.iloc[n])
            print('MAE: ', errors)
            meas[n, i_col] = errors.item() / predictions
    """
    [0.046615][0.05]
    Mean Absolute Error: [0.003385]
    [0.05107][0.04]
    Mean Absolute Error: [0.01107]
    [0.05548][0.04]
    Mean Absolute Error: [0.01548]
    [0.054315][0.05]
    Mean Absolute Error: [0.004315]
    [0.05823][0.04]
    Mean Absolute Error: [0.01823]
    [0.05729][0.04]
    Mean Absolute Error: [0.01729]
    [0.0522][0.06] 
    Mean Absolute Error: [0.0078]
    [0.05742][0.045]
    Mean Absolute Error: [0.01242]
    [0.06191][0.05]
    Mean Absolute Error: [0.01191]
    [0.06191][0.05]
    Mean Absolute Error: [0.01191]
    [0.05952][0.06]
    Mean Absolute Error: [0.00048]
    [0.06456][0.05]
    Mean Absolute Error: [0.01456]
    [0.06][0.07]
    Mean Absolute Error: [0.01]
    """


def try_gaussian_proc_regression(hdist, corr_df):
    """
    Try Gauss process regression to model observational random error in space.

    Returns
    -------

    """
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK
    np.random.seed(1)
    # Mesh the input space for evaluations of the real function, the
    # prediction and its MSE
    x = np.atleast_2d(np.linspace(0, 100000, 100)).T
    # now the noisy case
    X = hdist.values[np.triu_indices(hdist.shape[0], k=1)].flatten()
    X = np.atleast_2d(X).T
    # Observations and noise
    y = corr_df.values[np.triu_indices(hdist.shape[0], k=1)].flatten()
    dy = 0.1
    noise = dy
    y += noise
    # Instantiate a Gaussian Process model
    kernel = CK(1000.0, (1e1, 1e6)) * RBF(1000, (1e1, 1e4))
    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                  n_restarts_optimizer=10)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    # Plot the function, the prediction and the 95% confidence interval based
    # on the MSE
    plt.figure()
    # plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10,
                 label='Observations')
    plt.plot(x, y_pred, 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
             np.concatenate([y_pred - 1.9600 * sigma,
                             (y_pred + 1.9600 * sigma)[::-1]]),
             alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    # plt.ylim(0, 1)
    plt.legend(loc='upper left')
    plt.show()


def linreg_of_station_regressions_perturbed(conc, hdist) -> None:
    """
    Do a linear regression of station values, but perturb.

    # todo: perturb what?

    Returns
    -------
    None
    """

    result_df = pd.DataFrame([], columns=['s1', 's2', 'dist', 'R2_corr',
                                          'pval_corr'])
    iterations = 1000
    mu = 0.
    sigma = 0.005
    from scipy import stats

    for i, (s1, s2) in enumerate(combinations(conc.columns.values, 2)):
        print(i, s1, s2)
        id_1 = station_to_glacier[s1.split('_')[1]]
        id_2 = station_to_glacier[s2.split('_')[1]]

        # gdir_1 = GlacierDirectory(id_1, base_dir='C:\\users\\johannes\\
        # documents\\modelruns\\CH\\per_glacier\\')
        # gdir_2 = GlacierDirectory(id_2,
        # base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\
        # per_glacier\\')

        # gm_1 = GlacierMeteo(gdir_1)
        # gm_2 = GlacierMeteo(gdir_2)

        conccopy = conc[[s1, s2]].copy()
        conccopy.dropna(axis=0, inplace=True)

        for n in range(iterations):
            noise = np.random.normal(mu, sigma, [len(conccopy), 2])
            conccopy += noise

            # sun_1 = gm_1.meteo.sel(time=conccopy.index).sis
            # sun_2 = gm_2.meteo.sel(time=conccopy.index).sis
            # sun_dev_1 = sun_1 - np.nanmean(sun_1) / np.nanmean(sun_1)
            # sun_dev_2 = sun_2 - np.nanmean(sun_2) / np.nanmean(sun_2)
            # sun_notnan = np.where(~np.isnan(sun_dev_1) |
            #                       ~np.isnan(sun_dev_2))[0]
            # print(sun_dev_1, sun_dev_2)
            # weighted with incoming solar
            # y = conccopy[s2][sun_notnan]*sun_dev_2[sun_notnan]
            # X = sm.add_constant(conccopy[s1][sun_notnan] *
            # sun_dev_1[sun_notnan].values)

            y = conccopy[s2]
            X = sm.add_constant(conccopy[s1])

            model = sm.OLS(y, X).fit()

            # this is some bits copied from the influence plot function
            influence = model.get_influence()
            infl = influence
            resids = infl.resid_studentized_external
            alpha = .05
            cutoff = stats.t.ppf(1. - alpha / 2, model.df_resid)
            large_resid = np.abs(resids) > cutoff
            leverage = infl.hat_matrix_diag
            large_leverage = leverage > \
                statsmodels.graphics.regressionplots._high_leverage(model)
            large_points = np.logical_or(large_resid, large_leverage)

            y_corr = y[~large_points]
            X_corr = X[~large_points]

            model_corr = sm.OLS(y_corr, X_corr).fit()

            # plt.figure()

            # plt.scatter(X_corr.values[:, 1], y_corr)
            # plt.plot(np.arange(np.min(X_corr.values[:, 1]),
            #     np.max(X_corr.values[:, 1])+0.01, 0.01), model_corr.params[0]
            #     + np.arange(np.min(X_corr.values[:, 1]),
            #     np.max(X_corr.values[:, 1])+0.01, 0.01) *
            #     model_corr.params[1])
            # plt.xlabel(s1)
            # plt.ylabel(s2)

            # x_low, _ = plt.xlim()
            # y_low, _ = plt.ylim()
            # plt.annotate(
            #    'R2: {:.2f}\np-val: {:.2f}'.format(model_corr.rsquared,
            #    model_corr.pvalues[1]), (x_low, y_low))

            result_df.loc[i * n + n, 's1'] = int(s1.split('_')[1])
            result_df.loc[i * n + n, 's2'] = int(s2.split('_')[1])
            result_df.loc[i * n + n, 'hdist'] = hdist[int(s1.split('_')[1])][
                hdist.index == int(s2.split('_')[1])].item()
            result_df.loc[i * n + n, 'vdist'] = abs(
                station_to_height[int(s1.split('_')[1])] -
                station_to_height[int(s2.split('_')[1])])
            result_df.loc[i * n + n, 'R2_corr'] = model_corr.rsquared
            result_df.loc[i * n + n, 'pval_corr'] = model_corr.pvalues[1]

        # fig, ax = plt.subplots(figsize=(12, 8))
        # fig = sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
    rdf_valid = result_df[result_df.pval_corr <= 0.01]
    plt.scatter(rdf_valid.dist, rdf_valid.R2_corr)

    variomodel = sm.OLS(rdf_valid.R2_corr.values.astype(np.float32),
                        sm.add_constant(
                            rdf_valid[['hdist', 'vdist']].values.astype(
                                np.float32))).fit()
    plt.plot(np.arange(0, np.max(result_df.hdist), 1000),
             variomodel.params[0] + np.arange(0, np.max(result_df.hdist),
                                              1000) * variomodel.params[1],
             label='uncorr, R2: {:.2f}, p: {:.2f}'.format(variomodel.rsquared,
                                                          variomodel.pvalues[
                                                              1]))

    influence = variomodel.get_influence()
    infl = influence
    resids = infl.resid_studentized_external
    alpha = .05
    cutoff = stats.t.ppf(1. - alpha / 2, variomodel.df_resid)
    large_resid = np.abs(resids) > cutoff
    leverage = infl.hat_matrix_diag
    large_leverage = \
        leverage > statsmodels.graphics.regressionplots._high_leverage(
        variomodel)
    large_points = np.logical_or(large_resid, large_leverage)

    y_corr = y[~large_points]
    X_corr = X[~large_points]

    variomodel = sm.OLS(
        np.delete(rdf_valid.R2_corr.values.astype(np.float32), large_points),
        sm.add_constant(
            np.delete(rdf_valid[['hdist', 'vdist']].values.astype(np.float32),
                      large_points))).fit()
    plt.plot(np.arange(0, np.max(result_df.hdist), 1000),
             variomodel.params[0] + np.arange(0, np.max(result_df.hdist),
                                              1000) * variomodel.params[1],
             label='corr, R2: {:.2f}, p: {:.2f}'.format(variomodel.rsquared,
                                                        variomodel.pvalues[1]))
    plt.legend()
    plt.title('Linear regression of significant station correlations')


def linreg_of_stations_regressions_kickout_outliers(conc, hdist) -> None:
    """
    Do a linear regression of camera station values, but leave outliers out.

    Outliers are defined by high leverage.

    Returns
    -------
    None
    """
    result_df = pd.DataFrame([], columns=['s1', 's2', 'dist', 'R2_corr',
                                          'pval_corr'])
    iterations = 1000
    mu = 0.
    sigma = 0.005
    from scipy import stats

    for i, (s1, s2) in enumerate(combinations(conc.columns.values, 2)):
        print(i, s1, s2)
        id_1 = station_to_glacier[int(s1.split('_')[1])]
        id_2 = station_to_glacier[int(s2.split('_')[1])]

        # gdir_1 = GlacierDirectory(id_1,
        #     base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\'
        #     'per_glacier\\')
        # gdir_2 = GlacierDirectory(id_2, base_dir='C:\\users\\johannes\\'
        # 'documents\\modelruns\\CH\\per_glacier\\')

        # gm_1 = GlacierMeteo(gdir_1)
        # gm_2 = GlacierMeteo(gdir_2)

        conccopy = conc[[s1, s2]].copy()
        conccopy.dropna(axis=0, inplace=True)

        for n in range(iterations):
            noise = np.random.normal(mu, sigma, [len(conccopy), 2])
            conccopy += noise

            # sun_1 = gm_1.meteo.sel(time=conccopy.index).sis
            # sun_2 = gm_2.meteo.sel(time=conccopy.index).sis
            # sun_dev_1 = sun_1 - np.nanmean(sun_1) / np.nanmean(sun_1)
            # sun_dev_2 = sun_2 - np.nanmean(sun_2) / np.nanmean(sun_2)
            # sun_notnan = np.where(~np.isnan(sun_dev_1) |
            # ~np.isnan(sun_dev_2))[0]
            # print(sun_dev_1, sun_dev_2)
            # weighted with incoming solar
            # y = conccopy[s2][sun_notnan]*sun_dev_2[sun_notnan]
            # X = sm.add_constant(conccopy[s1][sun_notnan] *
            # sun_dev_1[sun_notnan].values)

            y = conccopy[s2]
            X = sm.add_constant(conccopy[s1])

            model = sm.OLS(y, X).fit()

            # this is some bits copied from the influence plot function
            influence = model.get_influence()
            infl = influence
            resids = infl.resid_studentized_external
            alpha = .05
            cutoff = stats.t.ppf(1. - alpha / 2, model.df_resid)
            large_resid = np.abs(resids) > cutoff
            leverage = infl.hat_matrix_diag
            large_leverage = \
                leverage > statsmodels.graphics.regressionplots._high_leverage(
                    model)
            large_points = np.logical_or(large_resid, large_leverage)

            y_corr = y[~large_points]
            X_corr = X[~large_points]

            model_corr = sm.OLS(y_corr, X_corr).fit()

            # plt.figure()

            # plt.scatter(X_corr.values[:, 1], y_corr)
            # plt.plot(np.arange(np.min(X_corr.values[:, 1]),
            # np.max(X_corr.values[:, 1])+0.01, 0.01), model_corr.params[0] +
            # np.arange(np.min(X_corr.values[:, 1]),
            # np.max(X_corr.values[:, 1])+0.01, 0.01) * model_corr.params[1])
            # plt.xlabel(s1)
            # plt.ylabel(s2)

            # x_low, _ = plt.xlim()
            # y_low, _ = plt.ylim()
            # plt.annotate(
            #    'R2: {:.2f}\np-val: {:.2f}'.format(model_corr.rsquared,
            #    model_corr.pvalues[1]),
            #    (x_low, y_low))

            result_df.loc[i * n + n, 's1'] = int(s1.split('_')[1])
            result_df.loc[i * n + n, 's2'] = int(s2.split('_')[1])
            result_df.loc[i * n + n, 'hdist'] = hdist[int(s1.split('_')[1])][
                hdist.index == int(s2.split('_')[1])].item()
            result_df.loc[i * n + n, 'vdist'] = abs(
                station_to_height[int(s1.split('_')[1])] -
                station_to_height[int(s2.split('_')[1])])
            result_df.loc[i * n + n, 'R2_corr'] = model_corr.rsquared
            result_df.loc[i * n + n, 'pval_corr'] = model_corr.pvalues[1]

        # fig, ax = plt.subplots(figsize=(12, 8))
        # fig = sm.graphics.influence_plot(model, ax=ax, criterion="cooks")
    rdf_valid = result_df[result_df.pval_corr <= 0.01]
    plt.scatter(rdf_valid.dist, rdf_valid.R2_corr)

    variomodel = sm.OLS(rdf_valid.R2_corr.values.astype(np.float32),
                        sm.add_constant(
                            rdf_valid[['hdist', 'vdist']].values.astype(
                                np.float32))).fit()
    plt.plot(np.arange(0, np.max(result_df.hdist), 1000),
             variomodel.params[0] + np.arange(0, np.max(result_df.hdist),
                                              1000) *
             variomodel.params[1],
             label='uncorr, R2: {:.2f}, p: {:.2f}'.format(variomodel.rsquared,
                                                          variomodel.pvalues[
                                                              1]))

    influence = variomodel.get_influence()
    infl = influence
    resids = infl.resid_studentized_external
    alpha = .05
    cutoff = stats.t.ppf(1. - alpha / 2, variomodel.df_resid)
    large_resid = np.abs(resids) > cutoff
    leverage = infl.hat_matrix_diag
    large_leverage = \
        leverage > statsmodels.graphics.regressionplots._high_leverage(
            variomodel)
    large_points = np.logical_or(large_resid, large_leverage)

    variomodel = sm.OLS(
        np.delete(rdf_valid.R2_corr.values.astype(np.float32), large_points),
        sm.add_constant(
            np.delete(rdf_valid[['hdist', 'vdist']].values.astype(np.float32),
                      large_points))).fit()
    plt.plot(np.arange(0, np.max(result_df.hdist), 1000),
             variomodel.params[0] + np.arange(0, np.max(result_df.hdist),
                                              1000) *
             variomodel.params[1],
             label='corr, R2: {:.2f}, p: {:.2f}'.format(variomodel.rsquared,
                                                        variomodel.pvalues[1]))
    plt.legend()
    plt.title('Linear regression of significant station correlations')


def correlation_of_dh_residuals(conc, hdist) -> None:
    """
    Correlate the residuals of Holfuy camera dh readings.

    Returns
    -------

    """
    fontsize = 18
    for s1, s2 in combinations(conc.columns.values, 2):
        conc_sel = conc[[s1, s2]]
        conc_sel = conc_sel.dropna(axis=0)
        resid_1 = conc_sel[s1] - conc_sel[s1].mean()
        resid_2 = conc_sel[s2] - conc_sel[s2].mean()
        model = sm.OLS(resid_1, sm.add_constant(resid_2)).fit()

        fig, ax = plt.subplots()
        ax.scatter(resid_1, resid_2)
        ax.plot(np.arange(min(resid_1) - 0.01, max(resid_1) + 0.01, 0.01),
                model.params[0] + np.arange(min(resid_1) - 0.01,
                                            max(resid_1) + 0.01, 0.01) *
                model.params[1],
                label='$R^2$: {:.2f},\np: {:.2f}'.format(model.rsquared,
                                                         model.pvalues[1]))
        plt.legend(fontsize=fontsize)
        plt.setp(ax.get_xticklabels(), fontsize=fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=fontsize)
        plt.xlabel('Residuals {} (m)'.format(s1), fontsize=fontsize)
        plt.ylabel('Residuals {} (m)'.format(s2), fontsize=fontsize)
        plt.tight_layout()


def spatial_correlation_of_station_correlations(
        conc, hdist, alpha: float = 0.95) -> None:
    """
    Plots the station residual correlations ($R^2$) over horizontal distance.

    The plot includes a confidence interval.

    Parameters
    ----------
    conc:
    hdist:
    alpha: float
        Confidence interval for the fit.

    Returns
    -------
    None.
    """
    fontsize = 18
    fig, ax = plt.subplots()
    r2list = []
    distlist = []
    for s1, s2 in combinations(conc.columns.values, 2):
        conc_sel = conc[[s1, s2]]
        conc_sel = conc_sel.dropna(axis=0)
        resid_1 = conc_sel[s1] - conc_sel[s1].mean()
        resid_2 = conc_sel[s2] - conc_sel[s2].mean()
        model = sm.OLS(resid_1, sm.add_constant(resid_2)).fit()
        r2list.append(model.rsquared)
        dist = hdist[int(s1.split('_')[1])][int(s2.split('_')[1])]
        distlist.append(dist)
        ax.scatter(dist, model.rsquared, c='b')
    distmodel = sm.OLS(r2list, sm.add_constant(distlist)).fit()
    ax.plot(np.arange(min(distlist) - 1.,
                      max(distlist) + 1., 1.),
            distmodel.params[0] + np.arange(min(distlist) - 1.,
                                            max(distlist) + 1., 1.) *
            distmodel.params[1], c='g',
            label='$R^2$: {:.2f},\np: {:.2f}'.format(model.rsquared,
                                                     model.pvalues[1]))
    _, stdata, _ = summary_table(distmodel, alpha=0.05)
    predict_mean_ci_low, predict_mean_ci_upp = stdata[:, 4:6].T
    ax.fill_between(sorted(distlist),
                    sorted(predict_mean_ci_low, reverse=True),
                    sorted(predict_mean_ci_upp, reverse=True),
                    facecolor='r', alpha=0.2, label='Confidence interval 0.05')
    plt.legend(fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.title('Correlation of residuals over distance', fontsize=fontsize)
    plt.xlabel('Distance (m)', fontsize=fontsize)
    plt.ylabel('$R^2$', fontsize=fontsize)
    plt.tight_layout()

    fontsize = 18
    fig, ax = plt.subplots()
    r2list = []
    distlist = []
    for s1, s2 in combinations(conc.columns.values, 2):
        conc_sel = conc[[s1, s2]]
        conc_sel = conc_sel.dropna(axis=0)
        resid_1 = conc_sel[s1] - conc_sel[s1].mean()
        resid_2 = conc_sel[s2] - conc_sel[s2].mean()
        model = sm.OLS(resid_1, sm.add_constant(resid_2)).fit()
        r2list.append(model.rsquared)
        dist = hdist[int(s1.split('_')[1])][int(s2.split('_')[1])] / 1000.
        distlist.append(dist)
        ax.scatter(dist, model.rsquared, c='b')
    distmodel = sm.OLS(r2list, sm.add_constant(distlist)).fit()
    ax.plot(np.arange(min(distlist) - 1., max(distlist) + 1., 1.),
            distmodel.params[0] + np.arange(min(distlist) - 1.,
                                            max(distlist) + 1., 1.) *
            distmodel.params[1], c='g',
            label='$R^2$: {:.2f},\np: {:.2f}'.format(model.rsquared,
                                                     model.pvalues[1]))
    _, stdata, _ = summary_table(distmodel, alpha=0.05)
    predict_mean_ci_low, predict_mean_ci_upp = stdata[:, 4:6].T
    ax.fill_between(sorted(distlist),
                    sorted(predict_mean_ci_low, reverse=True),
                    sorted(predict_mean_ci_upp, reverse=True),
                    facecolor='r', alpha=0.2, label='Confidence interval 95%')
    plt.legend(fontsize=fontsize)
    plt.setp(ax.get_xticklabels(), fontsize=fontsize)
    plt.setp(ax.get_yticklabels(), fontsize=fontsize)
    plt.title('Correlation of residuals over distance', fontsize=fontsize)
    plt.xlabel('Distance (km)', fontsize=fontsize)
    plt.ylabel('$R^2$', fontsize=fontsize)
    plt.tight_layout()


def fit_exponential_decay(result_df):
    """
    Creepy way to fit exponential decay to a R^2 over distance scatter plot.
    # todo: this function is horribly hard-coded.
    Returns
    -------

    """

    def fit_func(dist, a, c):
        """
        Exponential decay function.

        Parameters
        ----------
        dist : np.array
            Independent data.
        a : float
            Shape parameter.
        c : float
            Shape parameter.

        Returns
        -------
        Exponential decay function for independent data and parameters.
        """
        return c * (np.exp(1 - (dist / a)))

    popt, pcov = optimize.curve_fit(fit_func, result_df.hdist, result_df.R2)
    plt.scatter(result_df.hdist, result_df.R2, label='True')
    plt.plot(np.arange(0, 80000, 1000),
             fit_func(np.arange(0, 80000, 1000), *popt), 'g--',
             label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
    plt.legend()


if __name__ == '__main__':
    import geopandas as gpd
    from crampon import workflow
    import warnings
    from crampon import utils
    from itertools import product

    warnings.filterwarnings('ignore')

    cfg.initialize('C:\\users\\johannes\\documents\\crampon\\sandbox\\CH_params.cfg')
    cfg.PATHS['working_dir'] = 'c:\\users\\johannes\\documents\\modelruns\\Matthias_new'

    point_data_path = 'C:\\users\\johannes\\documents\\crampon\\data\\MB\\point\\cam_point_MBs_Matthias.csv'
    pdata = pd.read_csv(point_data_path,
                        index_col=0, parse_dates=[4, 5])
    pdata['swe_bp'] = np.nan
    pdata.loc[pdata.otype == 'snow', 'swe_bp'] = \
        pdata.loc[pdata.otype == 'snow', 'bp'] * (pdata.loc[pdata.otype == 'snow', 'rho'] / 1000.)
    pdata.loc[pdata.otype == 'ice', 'swe_bp'] = \
        pdata.loc[pdata.otype == 'ice', 'bp'] * (cfg.RHO / 1000.)

    all_mprobs = []
    all_mparts = []
    all_dspans = []
    glacier_names = []


    ancest_ix = []
    ancest_all = []
    mb_anc_ptcls = []
    mb_anc_ptcls_after = []
    alpha_anc_ptcls = []
    alpha_anc_ptcls_after = []


    #for m in [eval(m) for m in cfg.MASSBALANCE_MODELS]:
    #rdict = {}
    #gdir = utils.GlacierDirectory('RGI50-11.B4312n-1')
    ## get mean parameters
    #pg = ParameterGenerator(gdir, m, latest_climate=True)
    #calibrated = pg.single_glacier_params.values
    #calibrated = np.mean(calibrated, axis=0)
    #rdict = {**rdict, **dict(zip(
    #    [m.__name__ + '_' + m.cali_params_list[i] for i in
    #     range(len(m.cali_params_list))], calibrated))}
    #rgd = utils.GlacierDirectory('RGI50-11.B4312n-1')
    #ruppd = {}
    #for mbm in [eval(m) for m in cfg.MASSBALANCE_MODELS]:
    #    pg = ParameterGenerator(rgd, mbm, latest_climate=True, only_pairs=True,
    #                            constrain_with_bw_prcp_fac=True,
    #                            bw_constrain_year=2018 + 1,
    #                            narrow_distribution=0., output_type='array',
    #                            suffix='')
    #    param_prod = np.median(pg.from_single_glacier(), axis=0)
    #    ruppd.update(dict(zip([mbm.__name__+'_'+i for i in mbm.cali_params_list], param_prod)))
    #print(ruppd)
    """
    dspan, mprobs, mparts = run_aepf('RGI50-11.B4312n-1', mb_models=None, generate_params='past',
              param_fit='lognormal', param_method='memory',
              change_memory_mean=True, make_init_cond=True,
              qhisto_by_model=False, adjust_pelli=False, pdata=pdata,
              reset_albedo_and_swe_at_obs_init=False, write_probs=True,
                                     adjust_heights=False, prior_param_dict=None,
                                     unmeasured_period_param_dict=None, 
                                     crps_ice_only=False, 
                                     use_tgrad_uncertainty=True,
                                     use_pgrad_uncertainty=True, 
                                     assimilate_albedo=False)
    all_mprobs.append(mprobs)
    all_mparts.append(mparts)
    all_dspans.append(dspan)
    glacier_names.append('Rhonegletscher')
    """




    ancest_ix = []
    ancest_all = []
    mb_anc_ptcls = []
    mb_anc_ptcls_after = []
    alpha_anc_ptcls = []
    alpha_anc_ptcls_after = []
    #rdict = {}
    #gdir = utils.GlacierDirectory('RGI50-11.B5616n-1')
    ## get mean parameters
    #pg = ParameterGenerator(gdir, m, latest_climate=True)
    #calibrated = pg.single_glacier_params.values
    #calibrated = np.mean(calibrated, axis=0)
    #rdict = {**rdict, **dict(zip(
    #    [m.__name__ + '_' + m.cali_params_list[i] for i in
    #     range(len(m.cali_params_list))], calibrated))}

    #fgd = utils.GlacierDirectory('RGI50-11.B5616n-1')
    #fuppd = {}
    #for mbm in [eval(m) for m in cfg.MASSBALANCE_MODELS]:
    #    pg = ParameterGenerator(fgd, mbm, latest_climate=True, only_pairs=True,
    #                            constrain_with_bw_prcp_fac=True,
    #                            bw_constrain_year=2018 + 1,
    #                            narrow_distribution=0., output_type='array',
    #                            suffix='')
    #    param_prod = np.median(pg.from_single_glacier(), axis=0)
    #    fuppd.update(dict(zip([mbm.__name__+'_'+i for i in mbm.cali_params_list], param_prod)))
    #print(fuppd)


    dspan, mprobs, mparts = run_aepf('RGI50-11.B5616n-1', mb_models=None, generate_params='past',
                                     param_fit='lognormal',
                                     param_method='memory',
                                     change_memory_mean=True,
                                     make_init_cond=True,
                                     qhisto_by_model=False, adjust_pelli=False,
                                     adjust_heights=False,
                                     reset_albedo_and_swe_at_obs_init=False,
                                     pdata=pdata, write_probs=True, prior_param_dict=None,
                                     unmeasured_period_param_dict=None, crps_ice_only=False, use_tgrad_uncertainty=False)
    all_mprobs.append(mprobs)
    all_mparts.append(mparts)
    all_dspans.append(dspan)
    glacier_names.append('Findelgletscher')


    #rdict = {}
    #gdir = utils.GlacierDirectory('RGI50-11.A55F03')
    ## get mean parameters
    #pg = ParameterGenerator(gdir, m, latest_climate=True)
    #calibrated = pg.single_glacier_params.values
    #calibrated = np.mean(calibrated, axis=0)
    #rdict = {**rdict, **dict(zip(
    #    [m.__name__ + '_' + m.cali_params_list[i] for i in
    #     range(len(m.cali_params_list))], calibrated))}

    #pgd = utils.GlacierDirectory('RGI50-11.A55F03')
    #puppd = {}
    #for mbm in [eval(m) for m in cfg.MASSBALANCE_MODELS]:
    #    pg = ParameterGenerator(pgd, mbm, latest_climate=True, only_pairs=True,
    #                            constrain_with_bw_prcp_fac=True,
    #                            bw_constrain_year=2018 + 1,
    #                            narrow_distribution=0., output_type='array',
    #                            suffix='')
    #    param_prod = np.median(pg.from_single_glacier(), axis=0)
    #    puppd.update(dict(zip([mbm.__name__+'_'+i for i in mbm.cali_params_list], param_prod)))
    #print(puppd)
    dspan, mprobs, mparts = run_aepf('RGI50-11.A55F03', mb_models=None, generate_params='past',
             param_fit='lognormal', param_method='memory',
             change_memory_mean=True, make_init_cond=True,
             qhisto_by_model=False, adjust_pelli=False,
             adjust_heights=False, reset_albedo_and_swe_at_obs_init=False,
             pdata=pdata, write_probs=True, prior_param_dict=None,
             unmeasured_period_param_dict=None, crps_ice_only=False,
                                     use_tgrad_uncertainty=False)
    all_mprobs.append(mprobs)
    all_mparts.append(mparts)
    all_dspans.append(dspan)
    glacier_names.append('Plaine Morte')


    """
    for gp, fit in itertools.product(['gabbi', 'past'], ['gauss',
                                                              'lognormal',
                                                              'uniform']):
        try:
            run_aepf('RGI50-11.B4312n-1', generate_params=gp,
                     param_fit=fit, param_method='memory',
                     change_memory_mean=True, make_init_cond=False,
                     qhisto_by_model=True)
        except ValueError:
            plt.show()
            pass
    """

    """
    for gp, fit in itertools.product(['gabbi', 'past'], ['gauss',
                                                              'lognormal','uniform']):


        print(gp, fit)
        try:
            run_aepf('RGI50-11.A55F03', generate_params=gp,
                     param_fit=fit, param_method='memory',
                     change_memory_mean=True, make_init_cond=True,
                     qhisto_by_model=False, adjust_pelli=False, pdata=pdata)
        except:
            plt.show()
            pass
        try:
            run_aepf('RGI50-11.B4312n-1', generate_params=gp,
                     param_fit=fit, param_method='memory',
                     change_memory_mean=True, make_init_cond=True,
                     qhisto_by_model=False, adjust_pelli=False, pdata=pdata)
        except:
            plt.show()
            pass
        try:
            run_aepf('RGI50-11.B5616n-1', generate_params=gp,
                     param_fit=fit, param_method='memory',
                     change_memory_mean=True, make_init_cond=True,
                     qhisto_by_model=False, adjust_pelli=False,
                     adjust_heights=-45., pdata=pdata)
        except:
            plt.show()
            pass
    """

    # run_aepf('RGI50-11.B5616n-1', generate_params='past',
    #         param_fit='lognormal')
    #run_aepf('RGI50-11.B4312n-1', generate_params='gabbi',
    #         param_fit='gauss')

    #run_aepf('RGI50-11.A55F03', generate_params='past',
    #         param_fit='lognormal')
    #run_aepf('RGI50-11.B5616n-1', generate_params='past',
    #         param_fit='lognormal')
    #run_aepf('RGI50-11.B4312n-1', generate_params='gabbi',
    #         param_fit='gauss')

    #run_aepf('RGI50-11.A55F03', write_params=True)

    """
    # this is to test the point calibration
   
    """

    #run_aepf('RGI50-11.B5616n-1', generate_params='past')
    #run_aepf('RGI50-11.B4312n-1', generate_params='past')
    #run_aepf('RGI50-11.A55F03', generate_params='past')

    try:
        pass
        #run_aepf('RGI50-11.B5616n-1', generate_params='mean_past',
        #         validate=None)
        #run_aepf('RGI50-11.B4312n-1')
        #plt.show()
        #run_aepf('RGI50-11.A55F03')
        #plt.show()
        #run_aepf('RGI50-11.B5616n-1', stations=[1008])
        #run_aepf('RGI50-11.B5616n-1', stations=[1001])
        #run_aepf('RGI50-11.B4312n-1', stations=[1002])
        #run_aepf('RGI50-11.B4312n-1', stations=[1006])
        #run_aepf('RGI50-11.B4312n-1', stations=[1007])
        #run_aepf('RGI50-11.B4312n-1', stations=[1009])
    except Exception as e:
        print(e)
        raise
    print('Done')
    """
    glaciers = 'C:\\Users\\Johannes\\Desktop\\mauro_sgi_merge.shp'
    rgidf = gpd.read_file(glaciers)

    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B4312n-1'])]
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A55F03', 'RGI50-11.B4312n-1', 'RGI50-11.B5616n-1'])]
    #rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.B5616n-test'])]
    rgidf = rgidf[rgidf.RGIId.isin(['RGI50-11.A55F03'])]

    gdirs = workflow.init_glacier_regions(rgidf, reset=False, force=False)
    gdirs = [utils.GlacierDirectory('RGI50-11.B5616n-test', base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier')]

    for gdir in gdirs:
        tasks.update_climate(gdir)
        startday = dt.datetime(2018, 10, 1)
        startcover = gdir.read_pickle('snow_daily')
        startcover = startcover.sel(time=startday)

        with plt.style.context('seaborn-talk'):
            check = make_mb_current_mbyear_particle_filter(gdir, startday,
                                         snowcover=startcover, write=True, reset=False,
                                         filesuffix='')
    """
    """
    from crampon import graphics
    flist = glob.glob(
        'C:\\users\\johannes\\documents\\holfuyretriever\\manual*.csv')
    meas_list = [read_holfuy_camera_reading(f) for f in flist]
    meas_list = [m[['dh']] for m in meas_list]
    test = [m.rename({'dh': 'dh' + '_' + flist[i].split('.')[0][-4:]}, axis=1)
            for i, m in enumerate(meas_list)]
    conc = pd.concat(test, axis=1)
    graphics.make_annotated_heatmap(conc['dh'])
    conc.corr()
    conc.cov()
    """