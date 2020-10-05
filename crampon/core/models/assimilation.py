
from crampon import cfg
from crampon import utils
import pandas as pd
import numpy as np
from crampon.core.models.massbalance import BraithwaiteModel, HockModel, \
    PellicciottiModel, OerlemansModel, SnowFirnCover, \
    EnsembleMassBalanceModel, ParameterGenerator, MassBalanceModel
from crampon.core.preprocessing import climate
from crampon import tasks
import datetime as dt
import copy
import xarray as xr
import filterpy as fp
from filterpy import monte_carlo
from scipy import stats, optimize, special
import scipy.linalg as spla
from sklearn.neighbors import KernelDensity
import glob
import os
from crampon.core.holfuytools import *
import logging
import numba
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from properscoring import crps_ensemble
# _normpdf and _normcdf are are faster than scipy.stats.norm.pdf/cdf
from properscoring._crps import _crps_ensemble_vectorized, _normpdf, _normcdf

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
        # todo: should uncertainty be one value per date or spatially distributed? (depends on variable!e.g. for albedo, cloud probability could increase uncertainty)

        xr_ds = xr.Dataset({'albedo': (['x', 'y', 'date', 'model', 'member',
                                        'source'], ),
                            'MB': (['height', 'fl_id', 'date', 'model',
                                    'member', 'source'],)
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
        The model to the the paramater prior for
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
        helps if the number of calibrated paramaters is low. Default: 1.0 (
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
            np.mean(calibrated, axis=0), np.cov(
            calibrated.T) * std_scale_fac, size=n_samples)
    elif fit == 'uniform':
        params_prior = np.array([np.random.uniform(
            np.mean(calibrated[:, i]) - std_scale_fac * (
                        np.mean(calibrated[:, i]) - np.min(calibrated[:, i])),
                                           np.mean(
            calibrated[:, i]) + std_scale_fac * (np.max(calibrated[:, i]) - np.mean(
            calibrated[:, i])), n_samples)
        for i in range(calibrated.shape[1])]).T  # .T to make compatible
    elif fit == 'lognormal':
        # todo: is this multiplication with the scale factor correct?
        params_prior = np.random.multivariate_normal(
            np.log(np.mean(calibrated, axis=0)),
            np.cov(np.log(calibrated.T) * std_scale_fac),
            size=n_samples)
        params_prior = np.exp(params_prior)
    else:
        raise NotImplementedError('Paramater fit == {} not allowed. We cannot '
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

    #param_bounds = {
    #    'mu_ice': (3.0, 10.0),
    #    'mu_hock': (0.01, 3.6),
    #    'a_ice': (0.01, 0.0264),
    #    'tf': (0.01, 7.44),
    #    'srf': (0.01, 0.29),
    #    'c0': (-225., -5.),
    #    'c1': (1., 33.),
    #    'prcp_fac': (0.9, 2.1)
    #}

    # todo: take over param bounds from model class/cfg
    #theta_prior = np.array([np.random.uniform(param_bounds[p][0],
    #                                          param_bounds[p][1], n_samples)
    #                        for p in model.cali_params_list]).T
    #theta_prior = np.array([np.abs(np.clip(np.random.normal(np.mean(
    #    param_bounds[p]), np.abs(np.ptp(param_bounds[p])/4.), n_samples),
    #    param_bounds[p][0], param_bounds[p][1]))
    #        for p in model.cali_params_list]).T
    if fit == 'gauss':
        # th 6 is empirical: we want (roughly) to define sigma such that 99% of
        # values are within the bounds
        np.random.seed(seed)
        theta_prior = np.array([np.clip(np.random.normal(np.mean(
            param_bounds[p]), np.abs(np.ptp(param_bounds[p]) / 6.), n_samples),
            param_bounds[p][0], param_bounds[p][1])
            for p in model.cali_params_list]).T
    elif fit == 'uniform':
        np.random.seed(0)
        theta_prior = np.array([np.random.uniform(param_bounds[p][0],
                                               param_bounds[p][1], n_samples)
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
                           truncate: float or None=None) -> np.ndarray:
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
    # todo: w[:] = 1. was there in Labbe.
    # w[:] = 1.
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
        #w *= stats.truncnorm.pdf(particles, a, b, obs, obs_std)
        w += np.log(stats.truncnorm.pdf(particles, a, b, obs, obs_std))
    else:
        #w *= stats.norm(particles, obs_std).pdf(obs)
        w += np.log(stats.norm(particles, obs_std).pdf(obs))
    w += 1.e-300  # avoid round-off to zero
    #new_wgts = w / np.sum(w)  # normalize
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

    # todo: try out using sum instead of nansum (should work!)
    return 1. / np.nansum(np.square(w), axis=0)


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


def stratified_resample(weights, n_samples=None, one_random_number=False,
                        seed=None):
    """
    Copied form filterpy, but extended by an own choice of N.

    todo: Try and make a PR at filterpy
    Parameters
    ----------
    weights :
    n_samples :

    Returns
    -------

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
        # N*w_j
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
                #todo: try and replace this with the actual method keyword!!!???
                indices = resamp_method(weights[:, nri])
            except IndexError:
                worked = False
                while worked is False:
                    try:
                        # avoid some stupid precision error
                        input = weights[:, nri] * (
                                    1. / np.cumsum(weights[:, nri])[-1])
                        indices = resamp_method(input)
                        worked = True
                        print('worked')
                    except IndexError:
                        print('did not work')
                        pass
            particles[:, nri], weights[:, nri] = resample_from_index(
                particles[:, nri], weights[:, nri], indices)

    return particles, weights


class State(np.ndarray):
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
    """Represent an aougmented multi-model glacier state."""
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


class FilteringMethod(AssimilationMethod):
    """Interface to filtering assimilation methods."""


#class ParticleFilter(FilteringMethod):
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
        save: bool
            True if results should be saved when using the `step` method.
        plot: bool
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
                              save=save, plot=plot)

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
        # todo: w[:] = 1. was there in Labbe.
        #w[:] = 1.
        #w *= stats.norm(self.particles, obs_std).pdf(obs)
        w += np.log(stats.norm(self.particles, obs_std).pdf(obs))
        w += 1.e-300  # avoid round-off to zero
        #self.weights = w / np.sum(w)  # normalize
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
        #w *= stats.truncnorm.pdf(self.particles, a, b, obs, obs_std)
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
        # todo: w[:] = 1. was there in Labbe.
        # w[:] = 1.
        a = (mu / sigma) ** 2 # shape factor
        #w *= stats.gamma.pdf(self.particles, a, obs, obs_std)
        w += np.log(stats.gamma.pdf(self.particles, a, obs, obs_std))
        w += 1.e-300  # avoid round-off to zero
        self.weights = np.exp(w) / np.sum(np.exp(w))  # normalize

    def update_random(self):
        """
        Update the prior randomly.

        Returns
        -------

        """
        # todo: (how) is this possible?
        raise NotImplementedError

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

        # todo: the handling of the condition (effective n thresh exceeded yes/no) is in the function resample_particles...this should probably be changed.

        Parameters
        ----------
        method: method from filterpy.monte_carlo
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
                self.update_truncated_gaussian()
            elif update_method == 'random':
                raise NotImplementedError
            elif update_method == 'gamma':
                self.update_gamma()
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
            #self.to_file()

        # 5) plot, if wanted
        if self.do_plot is True:
            self.plot()

        #if self.do_save:
        #    self.save()
        #if self.do_ani:
        #    self.ani()

        #if self.plot_save:
        #    self.p_save()

    def plot(self):
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


#class AugmentedEnsembleParticleFilter(object):
class AugmentedEnsembleParticleFilter(ParticleFilter):
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
                 n_phys_vars: int, n_aug_vars: int, model_prob: list or None
                 = None):
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

        #super().__init__(n_particles, do_plot=False, do_save=False)

        self.models = models
        self.n_models = len(models)
        self.model_range = np.arange(self.n_models)
        self.n_tot = n_particles

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

        # todo: model prob. should be given as log right away
        if model_prob is None:
            self._model_prob_log = np.repeat(np.log(1. / self.n_models),
            self.n_models)
            self._n_model_particles = self.n_tot / self.n_models
        else:
            assert np.isclose(np.sum(model_prob), 1.)
            self._model_prob_log = np.array(np.log(model_prob))
            self._n_model_particles = self.model_prob * self.n_tot

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

        # todo: most stupid thing ever: the index where to retrieve the
        #  statistics from (particles per model etc)
        self.stats_ix = 0

    @property
    def weights(self):
        return np.exp(self.log_weights)

    @property
    def log_weights(self):
        """Logarithm of the weights."""
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
        """Model proability in the logarithmic domain."""
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
        # todo: this does not account for spatial dim
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
        """Logarithmic eights per model as a list."""
        return [self.log_weights[self.stats_ix, mix] for mix in
                         self.model_indices_all]

    @property
    def M_t_all(self):
        # todo: is this correct?
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
                    self.model_prob_log[j])
            big_covlist.append(np.cov(model_ptcls.T, aweights=model_wgts))

        return np.array(big_covlist)

    @property
    def effective_n_per_model(self):
        """Effective number of particles per model."""
        n_eff = [effective_n(w/np.sum(w)) for w in self.model_weights]
        return n_eff

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
        by_model: bool


        Returns
        -------
        obs_quantiles: np.array
            Array with quantiles of the observation at the weighted particle
            distribution.
        """

        # todo: the zeros are hard-coded (the keyword is stupid, but make it
        #  easier afterwards
        obs_quantiles = []#np.full_like(obs, np.nan)
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
                    w_quantiles = np.sum(self.weights[eix, :] *
                                         (obs[obs_first_dim, i] - actual_mb/obs_std[
                                                         obs_first_dim, i]))
                    obs_quantiles.append(w_quantiles)

        return obs_quantiles

    def get_observation_quantiles(self, obs, obs_std, mb_ix, mb_init_ix,
                                  eval_ix,
                                 obs_first_dim=0, generate_n_obs=1000,
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
                                    obs_std[
                                        obs_first_dim, i],
                                    generate_n_obs),
                            quantile_range)
                        intermediate_list.append(
                            np.argmin(np.abs(np.atleast_2d(w_quantiles).T -
                                             actual_mb[mi]), axis=0))
                    obs_quantiles.append(intermediate_list)
                else:
                    w_quantiles = utils.weighted_quantiles(np.random.normal(
                                                                  obs[
                                                                      obs_first_dim, i],
                                                                  obs_std[
                                                                      obs_first_dim, i],
                                                                  generate_n_obs),
                                                           quantile_range)
                    obs_quantiles.append(np.argmin(np.abs(np.atleast_2d(w_quantiles).T -actual_mb),
                                                   axis=0))

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

    def apply_to_models(self, func):
        """
        Decorator to apply something so all models.

        Returns
        -------

        """
        # todo: no idea what I wanted with this function - code doesn't make
        #  sense!?
        return np.array([func(i) for i in self.model_range])

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

    def predict(self, mb_models_inst, gmeteo, date, h, obs_merge, ssf,
                ipot, ipot_sigma, alpha_ix, mod_ix, swe_ix, tacc_ix, mb_ix,
                tacc_ice, model_error_mean, model_error_std,
                param_random_walk=False, snowredistfac=False,
                use_psol_multiplier=False, seed=None):
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
            Date to claculate the mass balance for.
        h : np.array
            Array with heights of the glacier flowline.
        obs_merge : xr.Dataset
            Dataset with the merged observations (m w.e.).
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
        param_random_walk : bool, optional
            Whether to let the parameters do a random walk. Default: False.
        snowredistfac : bool, optional
            Whether to use a snow redsitribution factor. Default: False.
        use_psol_multiplier : bool, optional
            Whether to use the periodic multiplier for solid precipitation.
            Default: False.
        seed : int, optional
            Seed to use to make experiments reproducible. Default: None (do not
            use seed).

        Returns
        -------
        None
        """

        doy = date.dayofyear

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
            try:
                remark = obs_merge.sel(date=date).key_remarks.values
                psol_rand[(~pd.isnull(remark)) & (
                            ('RAIN' in remark) & ('SNOW' not in remark))] = 0.
            # todo: IndexError when more elevation bands than observations
            except (IndexError, KeyError):  # day excluded by keyword
                pass

            # correct precipitation for systematic, not calibrated biases
            if use_psol_multiplier is True:
                psol_rand *= prcp_fac_cycle_multiplier[doy - 1]
            if snowredistfac is not False:
                psol_rand *= np.atleast_2d(snowredistfac[:, i]).T

            sis_rand = np.atleast_2d(gmeteo.randomize_variable(date, 'sis',
                                                               random_seed=seed))
            sis_rand *= ssf

            # todo: is perfect correlation a good assumption?
            ipot_reshape = np.repeat(np.atleast_2d(ipot).T,
                                     self.n_model_particles[i], axis=1)
            np.random.seed(seed)
            ipot_rand = ipot_reshape + \
                        np.atleast_2d(np.random.normal(
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
                    param_dict = dict(zip(m.cali_params_list, model_params[pi,
                                                              :len(m.cali_params_list)]))

                param_dict.update({'psol': psol_rand[:, pi],
                                   'tmean': temp_rand[:, pi],
                                   'tmax': tmax_rand[:, pi],
                                   'sis': sis_rand[:, pi]})

                # todo: first update SWE here?????? we also update alpha at t first and then calculate the MB (ablation)
                swe[:, pi] += param_dict['psol']

                # update tacc/albedo depending on SWE
                tacc[param_dict['psol'] >= 1., pi] = 0.
                old_snow_ix = ((param_dict['psol'] < 1.) & (swe[:, pi] > 0.))
                tacc[old_snow_ix, pi] += np.clip(param_dict['tmax'][old_snow_ix], 0., None)
                no_snow_ix = ((param_dict['psol'] < 1.) & (swe[:, pi] == 0.))
                # setting a chosen fixed ice albedo tacc
                if no_snow_ix.any():
                    tacc[no_snow_ix, pi] = tacc_ice

                # todo: the underlying albedo should depend on the SWE
                alpha[:, pi] = point_albedo_brock(swe[:, pi], tacc[:, pi],
                                                  swe[:, pi] == 0.,
                                                  a_u=cfg.PARAMS['ice_albedo_default'])

                # the model decides which MB to produce
                if m.__name__ == 'BraithwaiteModel':
                    melt = melt_braithwaite(**param_dict, swe=swe[:, pi])
                elif m.__name__ == 'HockModel':
                    melt = melt_hock(**param_dict, ipot=ipot_rand[:, pi],
                                   swe=swe[:, pi])
                elif m.__name__ == 'PellicciottiModel':
                    melt = melt_pellicciotti(**param_dict, alpha=alpha[:,
                                                                   pi])
                elif m.__name__ == 'OerlemansModel':
                    melt = melt_oerlemans(**param_dict, alpha=alpha[:, pi])
                else:
                    raise ValueError('Mass balance model not implemented for'
                                     ' particle filter.')

                # m w.e. = mm / 1000. - m w.e.
                mb = param_dict['psol'] / 1000. - melt

                mb_daily.append(mb)
                swe[:, pi] -= melt

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
            particles[:, :, alpha_ix][particles[:, :, mod_ix] ==
                                           im] \
                = \
                alpha_per_model[im].flatten()
            # *ADD* MB (+=) # MB needs Fortran flattening here!!!
            particles[:, :, mb_ix][particles[:, :, mod_ix] == im] \
                += \
                mb_per_model[im].flatten('F')
            # T_acc
            particles[:, :, tacc_ix][particles[:, :, mod_ix] == im] \
                = tacc_per_model[im].flatten()

        self.particles = particles.copy()

    def update(self, obs, obs_std, R, obs_ix, obs_init_mb_ix, obs_spatial_ix):
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
        #  should be all the same
        w = copy.deepcopy(self.log_weights[obs_indices[-1][0], :])

        for obs_loc in obs_indices[-1]:
            obs_s_ix = obs_spatial_ix[obs_loc]
            mb_particles = self.particles[obs_s_ix, :, obs_ix]
            mb_particles_init = self.particles[obs_s_ix, :, obs_init_mb_ix]

            # todo: this is the version that updates in observation space
            h_status = (mb_particles - mb_particles_init) / 0.9
            h_obs = obs[0, obs_loc] / 0.9
            h_obs_std = obs_std[0, obs_loc] / 0.9
            # todo: implement covariance solution > one obs variable

            # todo: this is the version that updates in observation space
            w += -((h_obs - h_status) ** 2.) / (2. * h_obs_std ** 2)

        w -= np.max(w)

        new_wgts = w - np.log(np.sum(np.exp(w)))  # normalize

        # todo: here is the point where we assume perfect correlation in space
        self.log_weights = np.repeat(new_wgts[np.newaxis, ...],
                                     self.log_weights.shape[0],
                                     axis=0)

    def resample(self, phi=0.1, gamma=0.05, diversify=False, seed=None):
        """
        Resample adaptively.

        We choose the number of particles to be resampled per model as the
        minimum contribution \phi plus some excess frequency L_{t,j} which
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
        weights_for_resamp = [np.exp(self.model_weights_log[i] -
                                     self.model_prob_log[i]) for i in self.model_range]
        resample_indices = [stratified_resample(weights_for_resamp[i],
                                                N_t_j[i],
                                                one_random_number=True) for i
                            in range(self.n_models)]
        print('UNIQUE PTCLS to RESAMPLE: ', [len(np.unique(ri)) for ri in
              resample_indices])

        new_p = []
        for m in range(self.n_models):
            npm, _ = resample_from_index_augmented(self.particles[:,
                                                       self.model_indices_all[m]],
                                                     self.weights[:,
                                                   self.model_indices_all[m]],
                                                     resample_indices[m])
            new_p.append(npm)

        # compensate for over-/underrepresentation
        new_weights_per_model = self.model_prob_log - np.log(N_t_j)

        if (new_weights_per_model == 0.).any() or np.isinf(
                new_weights_per_model).any():
            raise ValueError('New weights contain zero on Inf.')

        self.particles = np.concatenate(new_p, axis=1)

        # important: AFTER setting new particles; later comment: why again?
        self.log_weights = np.repeat(np.hstack([np.tile(new_weights_per_model[i], int(N_t_j[i]))
             for i in self.model_range])[np.newaxis, ...],
                                     self.spatial_dims[0], axis=0)

        if diversify is True:
            self.diversify_parameters(pmean, pcov, gamma=gamma)

    def diversify_parameters(self, post_means, post_covs, gamma=0.05):
        """
        Apply a parameter diversification after [1]_.

        Parameters
        ----------
        gamma: float
            Reasonable values are 0.05 or 0.1. Default:0.05.

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
            theta_tilde_t_k_nan = self.particles[self.stats_ix, mix[j]][:,
                                  self.state_indices['theta']]
            theta_tilde_t_k = theta_tilde_t_k_nan[:,
                              ~np.isnan(theta_tilde_t_k_nan).all(axis=0)]

            psi_k_t = np.random.multivariate_normal(np.zeros(
                    cov_theta_t_j.shape[0]), cov_theta_t_j,
                    size=len(theta_tilde_t_k))

            theta_tilde_t_k_new = mu_t_j + (1 - gamma) * \
                                  (theta_tilde_t_k - mu_t_j) + \
                                  np.sqrt(1 - (1 - gamma)**2) * psi_k_t

            theta_tilde_t_k_new_pad = np.pad(theta_tilde_t_k_new, (
            (0, 0), (0, self.n_aug_vars - theta_tilde_t_k_new.shape[1])),
                   'constant', constant_values=np.nan)

            model_ptcls[:, self.state_indices['theta']] = theta_tilde_t_k_new_pad
            model_ptcls_all.append(model_ptcls)

        self.particles = np.repeat(np.concatenate(model_ptcls_all)[np.newaxis, :, :], self.spatial_dims[0], axis=0)

    def evolve_theta(self, mu_0, Sigma_0, rho=0.9, change_mean=True, seed=None):
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
            todo: JUST FOR TESTING
            if the mean should be change back to the prior. If no (False),
            the only the variability of the prior is given back to the
            parameter distribution.

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
            theta_j = theta_j_nan[..., ~np.isnan(theta_j_nan[0, ...]).all(axis=0)]
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
                self.update_truncated_gaussian()
            elif update_method == 'random':
                raise NotImplementedError
            elif update_method == 'gamma':
                self.update_gamma()
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
            #self.to_file()

        # 5) plot, if wanted
        if self.do_plot is True:
            self.plot()

        #if self.do_save:
        #    self.save()
        #if self.do_ani:
        #    self.ani()

        #if self.plot_save:
        #    self.p_save()

    def plot(self):
        pass


def melt_braithwaite(psol=None, mu_ice=None, tmean=None, swe=None,
                     prcp_fac=None, tmelt=0., tmax=None, sis=None):
    tempformelt = tmean - tmelt
    mu = np.ones_like(swe) * mu_ice
    mu[swe > 0.] = mu_ice * cfg.PARAMS['ratio_mu_snow_ice']

    return mu * tempformelt / 1000.


def melt_hock(psol=None, mu_hock=None, a_ice=None, tmean=None,
              ipot=None, prcp_fac=None, swe=None, tmelt=0., tmax=None,
              sis=None):
    tempformelt = tmean - tmelt
    a = np.ones_like(swe) * a_ice
    a[swe > 0.] = a_ice * cfg.PARAMS['ratio_a_snow_ice']
    melt_day = (mu_hock + a * ipot) * tempformelt
    return melt_day / 1000.


def melt_pellicciotti(psol=None, tf=None, srf=None, tmean=None,
                      sis=None, alpha=None, tmelt=1., prcp_fac=None,
                      tmax=None):
    melt_day = tf * tmean + srf * (1 - alpha) * sis
    melt_day[tmean <= tmelt] = 0.

    return melt_day / 1000.


def melt_oerlemans(psol=None, c0=None, c1=None, tmean=None, sis=None,
                   alpha=None, prcp_fac=None, tmax=None):
    # todo: IMPORTANT: sign of c0 is changed to make c0 positive (log!)
    qmelt = (1 - alpha) * sis - c0 + c1 * tmean
    # melt only happens where qmelt > 0.:
    qmelt = np.clip(qmelt, 0., None)

    # kg m-2 d-1 = W m-2 * s * J-1 kg
    # we want ice flux, so we drop RHO_W for the first...!?
    melt = (qmelt * cfg.SEC_IN_DAY) / cfg.LATENT_HEAT_FUSION_WATER

    return melt / 1000.


def point_albedo_brock(swe, t_acc, icedist, p1=0.713, p2=0.112, p3=0.442,
                     p4=0.058, a_u=None, d_star=0.024, alpha_max=0.85):
    if a_u is None:
        a_u = cfg.PARAMS['ice_albedo_default']
    alpha_ds = np.clip((p1 - p2 * np.log10(t_acc)), None, alpha_max)
    # shallow snow equation
    alpha_ss = np.clip((a_u + p3 * np.exp(-p4 * t_acc)), None, alpha_max)
    # combining deep and shallow
    alpha = (1. - np.exp(-swe / d_star)) * alpha_ds + np.exp(
        -swe / d_star) * alpha_ss
    # where there is ice, put its default albedo
    alpha[icedist] = cfg.PARAMS['ice_albedo_default']
    return alpha


def tacc_from_alpha_brock(alpha, p1=0.86, p2=0.155):
    # here we can only take the deep snow equation, otherwise it's not unique
    tacc = 10.**((alpha - p1) / (-p2))
    tacc[tacc < 1.] = 1.
    return tacc


def get_initial_conditions(gdir, date, n_samples, begin_mbyear=None,
                           min_std_swe=0.025, min_std_alpha=0.05,
                           min_std_tacc=10., param_dict=None, fl_ids=None,
                           seed=None):
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
    fl_ids: array-like or None
        Flow line IDs to select. Default: None (return all flow lines)

    Returns
    -------

    """

    # todo: this function should actually account for correlations in swe,
    #  alpha and MB -> not easy though, since not all models have alpha

    if begin_mbyear is None:
        begin_mbyear = utils.get_begin_last_flexyear(date)

    init_cond = make_mb_current_mbyear_heights(gdir,
                                   begin_mbyear=begin_mbyear,
                                   last_day=date,
                                   write=False, param_dict=param_dict)

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
    mb_init_random = mb_init_raw[:, np.random.randint(0, mb_init_raw.shape[1],
                                               size=n_samples)]
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
    # todo: by choosing the same random number, we assume correaltion=-1 between tacc and swe: the correlation should come from the modes though
    tacc_init = np.clip(rand_num * np.atleast_2d(tacc_std).T +
                        np.atleast_2d(tacc_mean).T, 0., None)
    # todo: this is bullshit we set a minimum std for swe, but tacc can be the same everywhere...to alpha will be reset in the first assimilation day
    swe_zero = np.where(swe_init == 0.)
    # todo: this should even be better constrained
    # todo: this should be <=!?
    alpha_init = point_albedo_brock(swe_init, tacc_init, swe_init==0.0)

    if fl_ids is not None:
        swe_init = swe_init[fl_ids, :]
        alpha_init = alpha_init[fl_ids, :]
        mb_init_random = mb_init_random[fl_ids, :]
        mb_init_raw = mb_init_raw[fl_ids, :]
        tacc_init = tacc_init[fl_ids, :]
        init_cond_all = init_cond[0].isel(fl_id=fl_ids)
    else:
        init_cond_all = init_cond[0].copy(deep=True)
    return init_cond_all, mb_init_raw, mb_init_random, swe_init, alpha_init, tacc_init


def run_aepf(id, mb_models=None, stations=None, generate_params=None,
             param_fit='lognormal', unmeasured_period_param_dict=None,
             prior_param_dict=None,
             evolve_params=True, update=True, write_params=False,
             param_method='memory', make_init_cond=True,
             change_memory_mean=True,
             qhisto_by_model=False, limit_to_camera_elevations=False,
             adjust_heights=False, adjust_pelli=True,
             reset_albedo_and_swe_at_obs_init=False, pdata=None,
             write_probs=False, crps_ice_only=False,
             use_tgrad_uncertainty=False):
    """
    Run the augmented ensemble particle filter.

    Parameters
    ----------
    id: str
        Glacier id to be processed.
    mb_models: list of `py:class:crampon.core.models.massbalance.DailyMassBalanceModelWithSnow`
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
    unmeasured_period_param_dict:
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
    write_params: bool
        whether to write out the parameters that the particle filter has
        found. Default: False.
    param_method: str
        How parameters shall be diversified. Either "liu" or "memory".
        Default: 'memory'.
    make_init_cond: bool
        # todo: this is just a shortcut when thing have to go fast
        Generate initial conditions with a spinup run during the mass budget
        year. Default: True.
    change_memory_mean: bool
        # todo: this is just an experiment
        When using the "memory" diversification method, this keyword
        determines whether the mean should also be changed back towards the
        prior parameter distribution (experiment setting it to "False" when
        you believe the prior mean is not trustworthy). Default: True.
    qhisto_by_model: bool
        Whether the quantile histogram shall be made by model. Default:
        False (make one quantile histogram for all).


    Returns
    -------
    None
    """
    from crampon.core.models import massbalance
    # how many particles
    n_particles = 10000
    n_phys_vars = 6
    n_aug_vars = 4
    # indices state: 0= MB, 1=alpha, 2=m, 3=swe, 4:tacc, 5:=params
    mod_ix = 0
    mb_ix = 1
    alpha_ix = 2
    swe_ix = 3
    tacc_ix = 4
    obs_init_mb_ix = 5
    theta_start_ix = 6
    param_random_walk = False
    # todo: there are still come unallowed cases, e.g. lognormal and gabbi
    param_prior_distshape = param_fit
    param_prior_std_scalefactor = [2.0, 2.0, 2.0, 2.0]
    phi = 0.1
    gamma = 0.05
    model_error_mean = 0.
    model_error_std = 0.0
    colors = ["b", "g", "c", "m"]
    tacc_ice = 4100.  # random value over 15 years
    theta_memory = 0.9  # the model parameter memory parameter
    obs_std_scale_fac = 1.0
    sis_sigma = 15.  # try a bigger/smaller STDEV for SIS

    min_std_alpha = 0.0#0.05
    min_std_swe = 0.0#0.025  # m w.e.
    min_std_tacc = 0.0#10.0

    fixed_obs_std = None  # m w.e.

    print('MIN_STD_ALPHA:, ', min_std_alpha, 'MIN_STD_SWE: ', min_std_swe,
          'MIN_STD_TACC: ', min_std_tacc)

    seed = 0

    # todo: change this and make it flexible
    # we stop one day earlier than the field date - anyway no obs anymore
    if id == 'RGI50-11.B4312n-1':
        run_end_date = '2019-09-11'#'2019-09-12'#'2019-10-01'#
    elif id == 'RGI50-11.B5616n-1':
        run_end_date = '2019-09-16'#'2019-09-17'#'2019-09-16'#
    elif id == 'RGI50-11.A55F03':
        run_end_date = '2019-09-18'#'2019-09-30'#''#
    elif id == 'RGI50-11.B5616n-test':
        run_end_date = '2019-09-16'#'2019-09-17'#
    else:
        raise ValueError('In this provisional version of code, you need to '
                         'specify an end date for the run of your glacier.')

    assert param_method in ['liu', 'memory']

    # todo: SIgma for I_pot?
    ipot_sigma = 15.  # W m-2

    if mb_models is None:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]
    print('MB MODELS: ', [m.__name__ for m in mb_models])

    init_particles_per_model = int(n_particles / len(mb_models))

    # the gdir do we want to process
    gdir = utils.GlacierDirectory(id)
    print(gdir.get_filepath('ipot'))  # just for testing
    fl_h, fl_w = gdir.get_inversion_flowline_hw()

    if adjust_heights is not False:
        h_r = ((np.max(fl_h) - fl_h) / (np.max(fl_h) - np.min(fl_h)))
        # todo: this is dh parametrization for medium-size glacier
        fl_h += ((h_r - 0.05)**4 + 0.19*(h_r-0.05) + 0.01) * adjust_heights

    # get the observations
    # todo: change back if we want multi-stations
    if stations is None:
        stations = id_to_station[gdir.rgi_id]
    else:
        stations = stations
    station_hgts = [station_to_height[s] for s in stations]

    obs_merge, obs_merge_cs = prepare_observations(gdir, stations)
    if fixed_obs_std is not None:
        obs_merge['swe_std'][:] = fixed_obs_std
        print(obs_merge.swe_std)

    first_obs_date = obs_merge_cs.date[np.argmin(np.isnan(obs_merge_cs),
                                                 axis=1)].values
    first_obs_date = np.array([pd.Timestamp(d) for d in first_obs_date])

    # todo: minus to revert the minus from the function! This is shit!
    obs_std_we_default = - utils.obs_operator_dh_stake_to_mwe(
        cfg.PARAMS['dh_obs_manual_std_default'])
    # todo: reconsider: could camera end up being on the wrong flowline?
    s_index = np.argmin(np.abs((fl_h - np.atleast_2d(station_hgts).T)), axis=1)
    print('s_index: ', s_index)
    # todo: it can be that two cameras end up on the same node -> avoid
    # limit height to where we observe with camera
    if limit_to_camera_elevations:
        h = fl_h[s_index]
        w = fl_w[s_index]
        fl_ids = s_index
    else:
        h = fl_h
        w = fl_w
        fl_ids = np.arange(len(fl_h))

    gmeteo = climate.GlacierMeteo(gdir, randomize=True,
                                  n_random_samples=n_particles, heights=fl_h,
                                  use_tgrad_uncertainty=use_tgrad_uncertainty)
    # todo: ATTENTION: This doesn't change sis_sigma in gmeteo.meteo!!!
    if sis_sigma is not None:
        gmeteo.sis_sigma = np.ones_like(gmeteo.sis_sigma) * sis_sigma

    # date when we have to start the calculate last year
    begin_mbyear = utils.get_begin_last_flexyear(
        pd.Timestamp(obs_merge_cs.date.values[0]))
    if pdata is not None:
        # earliest OBS is the minimum of all date0s
        autumn_obs_mindate = min(pdata.loc[stations].date0.values)
        # begin of the mb year is the minimum of the value in params.cfg and the earliest OBS
        autumn_obs_begin_mbyear_min = pd.Timestamp(min(autumn_obs_mindate,
                                                       np.datetime64(begin_mbyear)))
    else:
        # otherwise: both is at the values of params.cfg
        autumn_obs_mindate = begin_mbyear
        autumn_obs_begin_mbyear_min = begin_mbyear

    # get start values for alpha and SWE
    if make_init_cond is True:
        print('Getting initial conditions from ', autumn_obs_begin_mbyear_min,
              ' to ', pd.Timestamp(min(first_obs_date)))
        mb_init_field, mb_init_homo, mb_init, swe_init, alpha_init, tacc_init = get_initial_conditions(gdir,
                                                      pd.Timestamp(
                                                          min(first_obs_date)),
                                                      n_particles,
                                                      begin_mbyear=autumn_obs_begin_mbyear_min,
                                                      param_dict=unmeasured_period_param_dict,
                                                      fl_ids=fl_ids,
                                                      min_std_alpha=min_std_alpha,
                                                      min_std_swe=min_std_swe,
                                                      min_std_tacc=min_std_tacc,
                                                      seed=seed)
        print('Initial conditions from ', autumn_obs_begin_mbyear_min,
              ' to ', pd.Timestamp(min(first_obs_date)))
    else:
        swe_init = np.zeros((len(fl_ids), n_particles))
        alpha_init = np.ones((len(fl_ids), n_particles)) * cfg.PARAMS[
            'ice_albedo_default']
        mb_init = np.zeros((len(fl_ids), n_particles))

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
        firstobs_fl_ids = np.argmin(np.abs((fl_h - np.atleast_2d(firstobs_heights).T)), axis=1)
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
        mb_init_all_cs_pdata = mb_init_field.stack(ens=['model',
        'member']).sel(
            time=slice(autumn_obs_mindate, pdata_init_date)).cumsum(
            dim='time').isel(time=-1).MB.values
        mb_init_all_cs_first_obs_init = mb_init_field.stack(ens=['model',
                                                            'member']).sel(
            time=slice(autumn_obs_mindate, min(first_obs_date))).cumsum(
            dim='time').isel(time=-1).MB.values
        # minimum (before JAN) must be subtracted if there is snow
        mb_init_all_cs_first_obs_init_min = mb_init_field.stack(
            ens=['model', 'member']).sel(time=slice(autumn_obs_mindate,
                                                    pd.Timestamp('{}-12-31'.format(pd.Timestamp(autumn_obs_mindate).year)))).cumsum(dim='time').min(dim='time')
        mb_init_all_cs_first_obs_init_argmin = \
            np.argmin(np.average(mb_init_field.stack(ens=['model', 'member']).
                                 sel(time=slice(autumn_obs_mindate, pd.Timestamp('{}-12-31'.format(pd.Timestamp(autumn_obs_mindate).year)))).cumsum(dim='time').MB.values, weights=w, axis=0), axis=0)

        run_sel = []
        run_sel_diff = []
        #if np.isnan(pdata_init_unc):
        #    bp_unc = 0.05  # reading uncertainty for stakes
        #else:
        #    bp_unc = pdata_init_unc
        for foix, sfl in enumerate(firstobs_fl_ids):
            if np.isnan(pdata_init_unc[foix]):
                bp_unc = 0.05  # reading uncertainty for stakes
            else:
                bp_unc = pdata_init_unc[foix]
            point_mb = pdata_init_swe[foix]

            #if pdata_init_otype[foix] == 'ice':

            np.random.seed(seed)
            gauss_range = np.random.normal(point_mb, bp_unc, size=n_particles)
            gauss_range = np.clip(gauss_range, point_mb - bp_unc, point_mb + bp_unc)

            for unc_bp in gauss_range:
                run_sel.append(np.argmin(np.abs(mb_init_all_cs_pdata[sfl,
                                                      :] - unc_bp)))
                run_sel_diff.append(mb_init_all_cs_pdata[sfl,
                                                         np.argmin(np.abs(mb_init_all_cs_pdata[sfl,:] - unc_bp))] - unc_bp)
            #else:  # snow
            #    # find the runs closest to zero at first camera observation!!!
            #    for unc_bp in np.arange(-bp_unc, bp_unc, 0.01):
            #        model_swe = (mb_init_all_cs_first_obs_init - \
            #                    mb_init_all_cs_first_obs_init_min).MB.values
            #        member_no = np.argmin(np.abs(model_swe[sfl, :] - unc_bp))
            #        run_sel.append(member_no)
            #        run_sel_diff.append(mb_init_all_cs_first_obs_init[sfl,
            #                                                          member_no])
        if len(run_sel) == 0:
            raise ValueError

        np.random.seed(seed)
        rand_choice_ix = np.random.choice(range(len(run_sel)),
                                          n_particles)
        #mb_init = mb_init_homo[:, np.array(run_sel)[rand_choice_ix]]
        mb_init = mb_init_all_cs_first_obs_init[:, np.array(run_sel)[rand_choice_ix]]
        mb_init -= np.array(run_sel_diff)[rand_choice_ix]

        #mb_init = mb_init_homo[:, np.random.choice(run_sel, n_particles)]
    print(mb_init, 'AVERAGE: ', np.mean(np.average(mb_init, weights=w,
                                                   axis=0)), pd.Timestamp(
                                                      min(first_obs_date)))

    # get indices of those runs
    # select those runs in cumulative MB beginning at OCT-1


    ipot_per_fl = gdir.read_pickle('ipot_per_flowline')
    ipot_per_fl = np.array([i for sub in ipot_per_fl for i in sub])
    #ipot_year = ipot_per_fl[s_index, :]
    ipot_year = ipot_per_fl[fl_ids, :]

    # time span
    date_span = pd.date_range(obs_merge_cs.date.values[0], run_end_date)
    print('DATE SPAN: ', date_span[0], date_span[-1])

    # get prior parameter distributions
    # todo: it's confusing that the params are return in linear domain,
    #  but means and cov in log domain
    theta_priors, theta_priors_means, theta_priors_cov = \
        prepare_prior_parameters(gdir, mb_models, init_particles_per_model,
                                 param_prior_distshape,
                                 param_prior_std_scalefactor,
                                 generate_params, param_dict=prior_param_dict,
                                 adjust_pelli=adjust_pelli, seed=seed)
    # get snow redist fac
    snowredist_ds = xr.open_dataset(gdir.get_filepath('snow_redist'))

    aepf = AugmentedEnsembleParticleFilter(mb_models, n_particles,
                                           spatial_dims=(len(h),),
                                           n_phys_vars=n_phys_vars,
                                           n_aug_vars=n_aug_vars)

    # init particles
    #tacc_temp = tacc_from_alpha_brock(alpha_init)
    tacc_temp = tacc_init.copy()
    swe_temp = swe_init
    tacc_temp[swe_temp == 0.] = tacc_ice
    x_0 = np.full((len(h), n_particles, n_phys_vars+n_aug_vars), np.nan)
    x_0[:, :, mb_ix] = 0.  # mass balance
    x_0[:, :, alpha_ix] = alpha_init
    x_0[:, :, mod_ix] = np.repeat(np.arange(len(mb_models)),
                                  n_particles/len(mb_models))
    x_0[:, :, tacc_ix] = tacc_temp  # accum. max temp.
    x_0[:, :, swe_ix] = swe_temp
    x_0[:, :, obs_init_mb_ix] = 0.

    # assign params per model (can we save the model loop?)
    for mix in range(len(mb_models)):
        theta_m = theta_priors[mix]
        n_theta = theta_m.shape[1]
        # make parameters fit
        theta_m_reshape = np.tile(theta_m, (x_0.shape[0], 1))
        x_0[:, :, theta_start_ix:theta_start_ix+n_theta][x_0[:, :, mod_ix] == mix] = \
            theta_m_reshape

    aepf.particles = x_0
    aepf._M_t_all = np.array([np.max(aepf.model_weights_log[i]) for i in
                              aepf.model_range])
    aepf._S_t_all = np.array([np.sum(np.exp(aepf.model_weights_log[i] -
                                            aepf.M_t_all[i]))
                              for i in aepf.model_range])

    mb_models_inst = [m(gdir, bias=0., heights_widths=(h, w)) for m in
                      mb_models]

    sis_scale_fac = xr.open_dataarray(gdir.get_filepath(
        'sis_scale_factor')).values
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(20, 15), sharex=True)
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
    run_example_list = []
    np.random.seed(seed)
    if pdata is None:
        rand_45 = np.random.choice(range(n_particles), 45)
    else:
        rand_45 = np.random.choice(range(n_particles),
                                   mb_init_all_cs_first_obs_init.shape[1])

    obs_shape = np.atleast_2d(obs_merge_cs.isel(date=0).values).shape
    # todo: this might be saved if we check for the obs first, before the we
    #  make the prediction
    mb_anal_before = np.full(obs_shape, 0.)
    mb_anal_before_all = np.zeros((len(h), n_particles))
    mb_anal_std_before = np.full(obs_shape, 0.)

    write_param_dates_list = []
    write_params_std_list = []
    write_params_avg_list = []

    # quantiles of observation at weighted ensemble
    quant_list = []
    quant_list_pm = []

    # analyze_error is to eliminate CRPS values after long offtimes (unfair) - set to True initially, because, we want to capture the first day CRPS
    analyze_error = True

    for date in date_span:
        print(date)
        print('MPROB: ', aepf.model_prob)
        print('M_T_ALL: ', aepf.M_t_all)

        mprob_list.append(aepf.model_prob)
        mpart_list.append(aepf.n_model_particles)

        doy = date.dayofyear
        try:
            ipot = ipot_year[:, doy]
        except IndexError:
            # leap year
            ipot = ipot_year[364]

        # np leap year problem here
        #ssf = sis_scale_fac[s_index, doy, np.newaxis]
        ssf = sis_scale_fac[fl_ids, doy, np.newaxis]
        try:
            snowredistfac = snowredist_ds.sel(time=date).D.values
        except KeyError:
            #print('Not snowredistfac found for {}'.format(date))
            snowredistfac = None
            pass
        aepf.predict(mb_models_inst, gmeteo, date, h, obs_merge, ssf, ipot,
                     ipot_sigma, alpha_ix, mod_ix, swe_ix, tacc_ix, mb_ix,
                     tacc_ice, model_error_mean, model_error_std,
                     param_random_walk=param_random_walk,
                     snowredistfac=snowredistfac, seed=seed)

        print(aepf.particles[-1, :, swe_ix])

        params_per_model = aepf.params_per_model
        # Braithwaite mu_ice
        ax2.errorbar(date, np.mean(params_per_model[0][:, 0]),
                     yerr=np.std(params_per_model[0][:, 0]), fmt='o',
                     c=colors[0])
        # Braithwaite prcp_fac
        ax2.errorbar(date, np.mean(params_per_model[0][:, 1]),
                     yerr=np.std(params_per_model[0][:, 1]), fmt='o',
                     c=colors[0])
        # Pellicciotti tf
        try:
            pelli_ix = [m.__name__ for m in mb_models].index('PellicciottiModel')
            ax2.errorbar(date, np.mean(params_per_model[pelli_ix][:, 0]),
                         yerr=np.std(params_per_model[pelli_ix][:, 0]), fmt='o',
                         c=colors[2])
        except ValueError:
            pass

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
                obs_std[~np.isnan(std_manual)] = std_manual[~np.isnan(std_manual)] * obs_std_scale_fac
                obs_std[np.isnan(std_manual)] = obs_std_we_default * obs_std_scale_fac
                obs_phase = np.atleast_2d(obs_merge.sel(date=date).phase.values)


            # keep track of when camera is set up/gaps (need for reset MB)
            # todo: check std
            if date in first_obs_date:
                print('FIRST OBS', date)
                dix = date == first_obs_date
                # todo: changed for Hansruedi's multiple cam option
                if limit_to_camera_elevations is True:
                    # todo: implement indices_below for dix only
                    to_insert = mb_anal_before_all[dix, :].copy()
                    print(to_insert)
                    ptcls = aepf.particles.copy()
                    ptcls[dix, :, obs_init_mb_ix] = to_insert
                    if reset_albedo_and_swe_at_obs_init is True:
                        ptcls[dix, :, alpha_ix] = cfg.PARAMS['ice_albedo_default']
                        ptcls[dix, :, swe_ix] = 0.
                    aepf.particles = ptcls
                    print(aepf.particles[dix, :, obs_init_mb_ix])
                else:
                    to_insert = mb_anal_before_all[s_index[dix], :].copy()
                    print(to_insert)
                    ptcls = aepf.particles.copy()
                    ptcls[s_index[dix], :, obs_init_mb_ix] = to_insert
                    #plt.figure()
                    #plt.fill_between(fl_ids, np.mean(ptcls[:, :, alpha_ix], axis=1) - np.std(ptcls[:, :, alpha_ix], axis=1), np.mean(ptcls[:, :, alpha_ix], axis=1)+np.std(ptcls[:, :, alpha_ix], axis=1))
                    #plt.axvline(np.min(fl_ids[s_index[dix]]))
                    if reset_albedo_and_swe_at_obs_init is True:
                        indices_below = fl_h <= np.max(fl_h[s_index[dix]])
                        #ptcls[s_index[dix], :, alpha_ix] = cfg.PARAMS['ice_albedo_default']
                        #ptcls[s_index[dix], :, swe_ix] = 0.
                        np.random.seed(seed)
                        ptcls[indices_below, :, alpha_ix] = np.random.normal(cfg.PARAMS[
                            'ice_albedo_default'], min_std_alpha, ptcls[indices_below, :, alpha_ix].shape)
                        np.random.seed(seed)
                        ptcls[indices_below, :, swe_ix] = np.clip(np.random.normal(
                            0, min_std_swe, ptcls[indices_below, :, swe_ix].shape), 0., None)
                    aepf.particles = ptcls
                    print(aepf.particles[s_index[dix], :, obs_init_mb_ix])
        except:
            obs = None
            obs_std = None

        # todo: this is hardcoded: we take the first which is not NaN
        # 'valid observation index'
        if date == date_span[0]:
            # voi shouldn't change anymore
            voi = np.where(~np.isnan(obs))[-1][0]

        box_offset = 0.2
        if obs is not None:
            boxvals = [aepf.particles[s_index[voi], aepf.model_indices_all[
                mi]][:,
                       mb_ix]-obs[0][voi] for mi in aepf.model_range] + [
                aepf.particles[s_index[voi], :, mb_ix] - obs[0][voi]]
            boxpos = [(date - date_span[0]).days - aepf.n_models / 2. *
                      box_offset + box_offset * mi for mi in range(
                aepf.n_models)]
            for mi in aepf.model_range:
                ax5.boxplot(boxvals[mi], positions=[boxpos[mi]],
                            patch_artist=True,
                boxprops=dict(facecolor=colors[mi], color=colors[mi]))
            ax5.axvline(boxpos[mi] + box_offset)

        mod_pred_mean = [np.mean(aepf.particles[s_index[voi], :, mb_ix][
                                     aepf.particles[s_index[voi], :, mod_ix] ==
                                      i]) for
                         i in range(aepf.n_models)]
        mod_pred_std = [np.std(aepf.particles[s_index[voi], :, mb_ix][
        aepf.particles[s_index[voi], :, mod_ix] == i]) for i in range(
            aepf.n_models)]
        print('MOD PRED MEAN: ', mod_pred_mean)
        print('MOD PRED STD: ', mod_pred_std)
        if obs is not None:
            print('OBS: ', obs[0], ', OBS_STD: ', obs_std[0])
        else:
            print('OBS: ', obs, ', OBS_STD: ', obs_std)
        print('PREDICTED: ', np.average(aepf.particles[s_index, :, mb_ix] -
                                        aepf.particles[s_index, :, obs_init_mb_ix], axis=1, weights=aepf.weights[s_index, :]))

        # plot models
        y_jitter = np.array([pd.Timedelta(hours=2 * td) for td in
                             np.linspace(-2.0, 2.0, aepf.n_models)])
        y_vals = np.array([date] * len(mb_models)) + y_jitter
        ax1.scatter(y_vals, mod_pred_mean, c=colors[:len(mb_models)])
        ax1.errorbar(y_vals, mod_pred_mean, yerr=mod_pred_std, fmt='o',
                     zorder=0)

        # plot ensemble prediction
        ens_pred_mean = np.average(aepf.particles[s_index[voi], :, mb_ix],
                                   weights=aepf.weights[s_index[voi], :])
        ens_pred_std = np.sqrt(np.cov(aepf.particles[s_index[voi], :, mb_ix],
                                         aweights=aepf.weights[s_index[voi],
                                                               :]))
        ax1.errorbar(date, ens_pred_mean, yerr=ens_pred_std, c='y',
                     marker='o', fmt='o')

        if obs is not None:
            # plot obs
            ax1.errorbar(date, obs[0][voi], yerr=obs_std[0][voi], c='k',
            marker='o', fmt='o')

            if analyze_error is True:
                if limit_to_camera_elevations is True:
                    dep_list.append((obs - np.average(aepf.particles[:, :,
                                                  mb_ix] - aepf.particles[:, :,
                                                  obs_init_mb_ix],
                                       weights=aepf.weights[aepf.stats_ix, :],
                                                  axis=1)))
                    crps_list.append(crps_ensemble(obs*cfg.RHO_W/cfg.RHO, (aepf.particles[:, :,
                                                  mb_ix] - aepf.particles[:, :,
                                                  obs_init_mb_ix])*cfg.RHO_W/cfg.RHO, aepf.weights[aepf.stats_ix, :]))
                else:
                    dep_list.append((obs - np.average(aepf.particles[s_index, :,
                                                  mb_ix]- aepf.particles[s_index, :,
                                                  obs_init_mb_ix],
                                                  weights=aepf.weights[
                                                          s_index, :],
                                                  axis=1)))
                    # todo: check if crps calc.is correct ('observational error is neglected')
                    #properscoring_crps = crps_ensemble(obs[0].T /cfg.RHO*cfg.RHO_W,
                    #                               (aepf.particles[s_index, :,mb_ix] -
                    #                               aepf.particles[s_index,:,obs_init_mb_ix])/cfg.RHO*cfg.RHO_W,
                    #                               aepf.weights[s_index, :])
                    #crps_1 = crps_by_observation_height(aepf, obs.T/cfg.RHO*cfg.RHO_W, obs_std.T/cfg.RHO*cfg.RHO_W, s_index, mb_ix, obs_init_mb_ix)
                    resamp_ix = stratified_resample(
                        aepf.weights[aepf.stats_ix, :], n_samples=1000,
                        one_random_number=True, seed=seed)
                    resamp_p, _ = \
                        resample_from_index_augmented(((aepf.particles[s_index, :, mb_ix] - aepf.particles[s_index, :, obs_init_mb_ix]) * cfg.RHO_W / cfg.RHO)[:, :, np.newaxis], aepf.weights[s_index, :], resamp_ix)
                    properscoring_crps = crps_ensemble(obs[0].T * cfg.RHO_W / cfg.RHO,
                                             resamp_p[..., 0])
                    crps_1 = crps_by_observation_height_direct(
                        resamp_p[..., 0], obs.T * cfg.RHO_W / cfg.RHO,
                        obs_std.T / cfg.RHO * cfg.RHO_W)
                    four_ixs = np.random.uniform(low=0, high=resamp_p.shape[1]-1, size=4).astype(int)
                    crps_2 = crps_by_observation_height_direct(resamp_p[:, four_ixs, 0], obs.T * cfg.RHO_W / cfg.RHO,
                        obs_std.T / cfg.RHO * cfg.RHO_W)
                    if crps_ice_only is True:
                        print(obs_phase)
                        properscoring_crps[obs_phase[0] == 's'] = np.nan
                        crps_1[obs_phase[0] == 's'] = np.nan
                        crps_2[obs_phase[0] == 's'] = np.nan

                    crps_list.append(properscoring_crps)
                    crps_1_list.append(crps_1)
                    crps_2_list.append(crps_2)
            else:
                dep_list.append(np.full_like(obs, np.nan))
                # todo: check if crps calc.is correct ('observational error is neglected')
                crps_list.append(np.full_like(obs[0].T, np.nan))
                crps_1_list.append(np.full_like(obs[0].T, np.nan))
                crps_2_list.append(np.full_like(obs[0].T, np.nan))

            analyze_error = True
            print('DEPARTURE AT VOI: ', dep_list[-1])
            print('CRPS at VOI: ', crps_list[-1], crps_1_list[-1])
            # plot particles distribution per model
            ppm = aepf.n_model_particles
            for ppi, pp in enumerate(ppm):
                if ppi == 0:
                    ax3.bar(date, pp, color=colors[ppi])
                else:
                    ax3.bar(date, pp, bottom=np.sum(ppm[:ppi]),
                            color=colors[ppi])

            # alpha from Pellicciotti
            ax4.errorbar(date, np.mean(aepf.particles[s_index[voi], :,
                                                      alpha_ix][
                                           aepf.particles[s_index[voi], :,mod_ix] ==
                                           2]),
                         yerr=np.std(aepf.particles[s_index[voi], :, alpha_ix][
                                         aepf.particles[s_index[voi], :,
                                         mod_ix] == 2]),
                         fmt='o', c='cornflowerblue')

            obs_quant = aepf.get_observation_quantiles(obs, obs_std,
                                                       mb_ix=mb_ix,
                                                       mb_init_ix=obs_init_mb_ix,
                                                       eval_ix=s_index,
                                                       by_model=False)
            quant_list.append(obs_quant)
            obs_quant_pm = aepf.get_observation_quantiles(obs, obs_std,
                                                       mb_ix=mb_ix,
                                                       mb_init_ix=obs_init_mb_ix,
                                                       eval_ix=s_index,
                                                       by_model=True)
            quant_list_pm.append(obs_quant_pm)

            # obs covariance
            R = np.mat(np.eye(obs.shape[1]))

            if update is True:
                if limit_to_camera_elevations is True:
                    aepf.update(obs, obs_std, R, obs_ix=mb_ix,
                                obs_init_mb_ix=obs_init_mb_ix,
                                obs_spatial_ix=np.arange(len(s_index)))
                else:
                    aepf.update(obs, obs_std, R, obs_ix=mb_ix,
                                obs_init_mb_ix=obs_init_mb_ix,
                                obs_spatial_ix=s_index)
            print('UPDATED: ', np.average(aepf.particles[s_index, :, mb_ix] -
                                          aepf.particles[s_index, :, obs_init_mb_ix],
                                          weights=aepf.weights[s_index, :], axis=1))
            if date == pd.Timestamp('2019-07-24'):
                print('stop')
            print('MB today entire glacier: ',
                  np.average(np.average(aepf.particles[..., mb_ix]
                                        - aepf.particles[..., obs_init_mb_ix],
                                        weights=aepf.weights, axis=1), weights=w))
            print('MB today entire glacier: ', np.average(np.average(
                aepf.particles[..., mb_ix], weights=aepf.weights, axis=1),
                                                          weights=w))
            # resample
            if (evolve_params is True) and (param_method == 'liu'):
                aepf.resample(phi=phi, gamma=gamma, diversify=True, seed=seed)
            else:
                aepf.resample(phi=phi, gamma=gamma, diversify=False, seed=seed)

            ax1.errorbar(date+pd.Timedelta(hours=0.75), np.average(
                aepf.particles[s_index[voi], :, mb_ix],
                weights=aepf.weights[s_index[voi], :]),
                         yerr=np.sqrt(np.cov(aepf.particles[s_index[voi], :,
mb_ix],
                                             aweights=aepf.weights[s_index[
                                                                       voi],:])),
                         c='r', fmt='o')
        else:  # no observation this time
            analyze_error = False

        if write_params is True:
            write_param_dates_list.append(date)
            # 1) write weights
            p_avg = [
                np.average(aepf.params_per_model[k], weights=aepf.model_weights[k],
                           axis=0) for k in range(len(mb_models))]
            write_params_avg_list.append(p_avg)
            # 2) write actual params

            p_std = np.sqrt([np.average((aepf.params_per_model[k]-p_avg[k])**2, weights=aepf.model_weights[k], axis=0) for k in range(len(mb_models))])
            write_params_std_list.append(p_std)

        if (evolve_params is True) and (param_method == 'memory'):
            aepf.evolve_theta(theta_priors_means, theta_priors_cov,
                              rho=theta_memory, change_mean=change_memory_mean,
                              seed=seed)

        #  check if minimum std requirements are fulfilled:
        increase_std_alpha = np.std(aepf.particles[..., alpha_ix],
                                    axis=1) < min_std_alpha
        #print('Albedo std increased at indices: ', fl_ids[increase_std_alpha])
        #aepf.particles[increase_std_alpha, :, alpha_ix] = \
        #    aepf.particles[increase_std_alpha, :, alpha_ix] + \
        #    np.random.normal(np.atleast_2d(np.zeros(np.sum(increase_std_alpha))).T, np.atleast_2d(np.ones(np.sum(increase_std_alpha)) * min_std_alpha).T, aepf.particles[increase_std_alpha, :, alpha_ix].shape)
        #increase_std_swe = np.std(aepf.particles[..., swe_ix],
        #                      axis=1) < min_std_swe
        #print('Albedo std increased at indices: ', fl_ids[increase_std_swe])
        #aepf.particles[increase_std_swe, :, alpha_ix] = aepf.particles[increase_std_swe, :, swe_ix] + np.clip(
        #    np.random.normal(np.atleast_2d(np.zeros(np.sum(increase_std_swe))).T, np.atleast_2d(np.ones(np.sum(increase_std_swe)) * min_std_swe).T, aepf.particles[increase_std_swe, :, swe_ix].shape), 0., None)

        # save analysis
        mb_anal_before = np.atleast_2d(np.average(aepf.particles[..., mb_ix],
                                              axis=1,
                                    weights=aepf.weights))
        mb_anal_before_all = np.atleast_2d(aepf.particles[..., mb_ix])
        mb_anal_std_before = np.sqrt(np.average((aepf.particles[...,
                                                    mb_ix]-mb_anal_before.T)**2,
                                                    weights=aepf.weights, axis=1))

        run_example_list.append(np.average(aepf.particles[..., mb_ix], weights=w, axis=0)[rand_45])
        #run_avg_total.append()

    # calculate weighted average over heights at the end of the MB year
    mb_until_assim = mb_init
    resamp_ix = stratified_resample(aepf.weights[aepf.stats_ix, :],
                                    aepf.particles.shape[1])
    mb_during_assim_eq_weights = np.array([resample_from_index(aepf.particles[i, :, mb_ix],
                                                               aepf.weights[
                                                                   i, ...],
                                                               resamp_ix)[0]
                                           for i in
                                           np.arange(
                                               aepf.particles.shape[0])])
    # todo: shuffle doesn't matter?
    # todo_subtract OBS_INIT=?
    np.random.seed(seed)
    [np.random.shuffle(mb_during_assim_eq_weights[x, ...]) for x in range(
        mb_during_assim_eq_weights.shape[0])]
    mb_total = mb_until_assim + mb_during_assim_eq_weights
    if id == 'RGI50-11.B4312n-1':
        rho_stake_ids = np.argmin(np.abs((fl_h - np.atleast_2d([3234., 3113,
            2924., 2741, 2595., 2458., 2345., 2279., 2228., 2838., 2306,
            2222.]).T)), axis=1)
        # to convert numbers in GLAMOS table to m w.e.
        rho_densities = [0.52, 0.47, 0.47, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,
                        0.9, 0.9]
        idxs = np.argmin(np.abs((fl_h - np.atleast_2d(
            [3234., 3113., 2924., 2741., 2595., 2458., 2345., 2279., 2228.,
             2838., 2306., 2222.]).T)), axis=1)
        ms = mb_total[idxs, :]
        print('MB at GLAMOS stakes: ',
              np.mean(mb_total[rho_stake_ids, :], axis=1))
        print('Height distances: ', np.min(np.abs((fl_h - np.atleast_2d([3234.,
              3113., 2924., 2741., 2595., 2458., 2345., 2279., 2228., 2838.,
              2306., 2222.]).T)), axis=1))
        plt.figure()
        rho_glamos = np.array(
            [250, 91, 92, -178, -483, -645, -635, -548, -702, -83, -640,
             -759]) / 100. * rho_densities
        rho_mean = np.mean(ms, axis=1)
        rho_std = np.std(ms, axis=1)

        plt.scatter([3234., 3113., 2924., 2741., 2595., 2458., 2345., 2279., 2228., 2838., 2306., 2222.], rho_glamos, label = 'GLAMOS')
        plt.errorbar([3234., 3113., 2924., 2741., 2595., 2458., 2345., 2279., 2228., 2838., 2306., 2222.], rho_mean, yerr = rho_std, fmt = 'o', label = 'MOD', c = 'g')
        mad = np.mean(np.abs(rho_mean - rho_glamos))
        plt.title('RHONE MAD: {:.2f} m w.e.'.format(mad))
        plt.xlabel('Elevation (m)')
        plt.ylabel('Mass Balance in OBS period (m w.e.)')
        plt.legend()
        plt.show()

    if id == 'RGI50-11.B5616n-1':
        fin_stake_ids = np.argmin(np.abs((fl_h - np.atleast_2d(
            [2619., 2597., 2680., 2788., 2920., 3036., 3122., 3149., 3087.,
             3258., 3255., 3341., 3477.]).T)), axis=1)
        # to convert numbers in GLAMOS table to m w.e. (already in m w.e.)
        fin_densities = np.ones(13)
        print('MB at GLAMOS stakes: ',
              np.mean(mb_total[fin_stake_ids, :], axis=1))
        print('Height distances: ', np.min(np.abs((fl_h - np.atleast_2d(
            [2619., 2597., 2680., 2788., 2920., 3036., 3122., 3149., 3087.,
             3258., 3255., 3341., 3477.]).T)), axis=1))
    if id == 'RGI50-11.A55F03':
        plm_stake_ids = np.argmin(np.abs((fl_h - np.atleast_2d(
            [2694., 2715., 2753., 2663., 2682.]).T)), axis=1)
        # to convert numbers in GLAMOS table to m w.e. (already in m w.e.)
        fin_densities = np.ones(5)
        print('MB at GLAMOS stakes: ',
              np.mean(mb_total[plm_stake_ids, :], axis=1))
        print('Height distances: ', np.min(np.abs((fl_h - np.atleast_2d(
            [2694., 2715., 2753., 2663., 2682.]).T)), axis=1))
    mb_total_avg = np.average(mb_total, weights=w, axis=0)
    print('MB total on {}: {} $\pm$ {}'.format(date_span[-1], np.mean(
        mb_total_avg), np.std(mb_total_avg)))
    print('UNCERTAINTY before PF:{}, while PF: {}'.format(np.std(np.average(
        mb_until_assim, weights=w, axis=0)), np.std(np.average(mb_during_assim_eq_weights, weights=w, axis=0))))
    mb_total = np.sort(mb_until_assim) + np.sort(
        mb_during_assim_eq_weights)
    mb_total_avg = np.average(mb_total, weights=w, axis=0)
    print('MB total sorted on {}: {} $\pm$ {}'.format(date_span[-1], np.mean(
        mb_total_avg), np.std(mb_total_avg)))

    ax1.set_ylabel('Mass Balance (m w.e.)')
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[0], lw=4),
                    Line2D([0], [0], color=colors[1], lw=4),
                    Line2D([0], [0], color=colors[2], lw=4),
                    Line2D([0], [0], color=colors[3], lw=4),
                    Line2D([0], [0], color='k', lw=4),
                    Line2D([0], [0], color='y', lw=4),
                    Line2D([0], [0], color='r', lw=4)]
    ax1.legend(custom_lines, ['Braithwaite', 'Hock', 'Pellicciotti',
                              'Oerlemans', 'OBS', 'PRED', 'POST'])
               #scatterpoints=1, ncol=4, fontsize=8)
    ax1.grid()
    ax2.set_ylabel('$\mu^*_{ice}$ (mm ice K-1 d-1)\nTF ()')
    ax2.legend()
    ax2.grid()
    ax3.set_ylabel('Number of model particles')
    ax3.legend()
    ax3.grid()
    ax4.set_ylabel('Albedo distribution')
    ax4.legend()
    ax4.grid()
    fig.suptitle('Mass Balance and Ensemble Evolution on {} ({})'.format(
        gdir.name, stations))
    ax5.axhline(0.)
    ax5.set_xlabel('Days since Camera Setup')
    ax5.set_ylabel('Departure (PRED-OBS) (m w.e.)')
    ax5.legend(custom_lines[:-3], ['Braithwaite', 'Hock', 'Pellicciotti',
                                   'Oerlemans'])
    fig2.suptitle(''
                  '{} ({}). Median absolute ensemble departure: {:.3f} m '
                  'w.e.'.format(gdir.name, stations, np.nanmedian(np.abs(
        [item for sublist in dep_list for item in sublist]))))
    print('Median absolute departure old method: ', np.nanmedian(np.abs(
        [item for sublist in dep_list for item in sublist])))
    print('Median absolute departure new method: ', np.nanmedian(
        [np.abs(np.nanmean(sublist)) for sublist in dep_list]))
    fig2.savefig('C:\\users\\johannes\\documents\\publications'
                 '\\Paper_Cameras\\test\\{}_boxplot.png'.format(stations),
                 dpi=500)
    fig.savefig('C:\\users\\johannes\\documents\\publications'
                 '\\Paper_Cameras\\test\\{}_panel.png'.format(stations),
                dpi=500)

    plt.figure()

    # todo: WRONG DATES!!!!!
    plt.plot(date_span[:len(crps_list)], np.nanmean(np.array(crps_list), axis=1), label='old CRPS')
    plt.plot(date_span[:len(crps_1_list)], np.nanmean(np.array(crps_1_list), axis=1), label='CRPS 1')
    plt.legend()
    plt.title('{}, MEDIAN CRPS: {}, {}, {}'.format(gdir.name,
                                           np.nanmedian([item for sublist in crps_list for item in sublist]),
                                           np.nanmedian([item for sublist in crps_1_list for item in sublist]),
                                           np.nanmedian([item for sublist in crps_2_list for item in sublist]),
                                           np.nanmedian([item for sublist in crps_3_list for item in sublist])))

    print('{}, MEDIAN CRPS: {}, {}, {}'.format(gdir.name, np.nanmedian(
        [item for sublist in crps_list for item in sublist]), np.nanmedian(
        [item for sublist in crps_1_list for item in sublist]), np.nanmedian(
        [item for sublist in crps_2_list for item in sublist]), np.nanmedian(
        [item for sublist in crps_3_list for item in sublist])))
    # quantile histogram for all models
    plt.figure()
    plt.hist([np.nanmean(i[0]) for i in quant_list])
    plt.title('{}, {} calibration parameters ({} fit); station mean'.format(
    gdir.name, generate_params, param_prior_distshape))
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
    # todo: flexible numer of subplots only works until 6 MB models
    figm, axms = plt.subplots(int(np.floor(np.sqrt(len(mb_models)))), int(np.ceil(np.sqrt(len(mb_models)))), sharex=True)
    figm.suptitle('{}, {} calibration parameters ({} fit); stations individual'.format(gdir.name, generate_params,
                                  param_prior_distshape))
    try:
        axms_flat = axms.flat
    except AttributeError:  # only one model
        axms_flat = [axms]
    for i, axm in enumerate(axms_flat):
        sub_list = quant_list_pm[i]
        axm.hist(np.array([i[0] for i in sub_list]).flatten())
        axm.set_xlabel(mb_models[i].__name__)

    # todo: say that write_params and write_probs are mutually exclusive
    if write_params is True:
        return write_param_dates_list, write_params_std_list, write_params_avg_list
    elif write_probs is True:
        return date_span, mprob_list, mpart_list
    else:
        return None


def A_normal(mu, sigma_sqrd):
    #return 2 * sigma * _normpdf(mu / sigma) + \
    #       mu * (2 * stats.norm.cdf(mu / sigma) - 1)
    # todo: once, sigma was sigma_squared as parameter
    sigma = np.sqrt(sigma_sqrd)
    return 2 * sigma * _normpdf(mu / sigma) + mu * (
                2 * _normcdf(mu / sigma) - 1)


def auxcrpsC(m, s):
    #return 2. * s * dnorm_0(m / s, 0.) + m * (2. * pnorm_0(m / s, 1., 0.) - 1.)
    return 2. * s * stats.norm.pdf(m / s, 0.) + m * (2. * stats.norm.cdf(m / s, 1., 0.) - 1.)


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


def crps_by_observation_height(aepf, h_obs, h_obs_std, obs_ix, mb_ix, obs_init_mb_ix):
    wgts = aepf.weights[obs_ix, :][0]
    ptcls = aepf.particles[obs_ix, :, mb_ix] - aepf.particles[obs_ix, :, obs_init_mb_ix]
    #obs_origin_len = len(h_obs)
    #obs_notnan = np.where(~np.isnan(h_obs))[0]
    #ptcls = ptcls[obs_notnan]
    #h_obs = h_obs[obs_notnan]
    #h_obs_std = h_obs_std[obs_notnan]

    first_term = np.nansum(wgts * A_normal(h_obs - ptcls / cfg.RHO * cfg.RHO_W, h_obs_std**2), axis=1)
    len_weights = len(wgts)
    second_term = -0.5 * np.nansum([np.nansum(wgts * wgts[l] * A_normal((ptcls.T - ptcls[:, l]).T /
                   cfg.RHO * cfg.RHO_W, 2 * h_obs_std**2), axis=1) for l in
                                 range(len_weights)], axis=0)
    second_term[np.where(np.isnan(h_obs))[0]] = np.nan
    crps = first_term + second_term
    #result = np.full(obs_origin_len, np.nan)
    #result[obs_notnan] = crps
    return first_term + second_term


def corr_term(mu, sigma):
    """Potential correction term for trunacted gaussians as describe in Gneiting 2006 eq.5 """
    return -2 * sigma * _normpdf(mu / sigma) * _normcdf(
        -mu / sigma) + sigma / np.sqrt(np.pi) * _normcdf(
        -np.sqrt(2) * mu / sigma) + mu * (_normcdf(-mu / sigma)) ** 2


def crps_by_observation_height_direct(ptcls, h_obs, h_obs_std, wgts=None):
    """ PTCLS, h_obs and h_obs_std must be given in OBS SPACE!!!"""
    if wgts is None:
        wgts = np.ones(ptcls.shape[1]) / ptcls.shape[1]
    # weights may not be n-dimensional at the moment
    assert wgts.ndim == 1
    first_term = np.nansum(wgts * A_normal(h_obs - ptcls, h_obs_std**2), axis=1)
    len_weights = len(wgts)
    second_term = -0.5 * np.nansum([np.nansum(wgts * wgts[l] * A_normal((ptcls.T - ptcls[:, l]).T, 2 * h_obs_std**2), axis=1) for l in
                                 range(len_weights)], axis=0)
    second_term[np.where(np.isnan(h_obs))[0]] = np.nan
    if ((first_term + second_term) < 0.).any():
        print('CALC ERROR IN CRPS')
    return first_term + second_term


def crps_by_observation_height_direct_vectorized(ptcls, h_obs, h_obs_std, wgts=None):
    """ PTCLS, h_obs and h_obs_std must be given in OBS SPACE!!!"""
    if wgts is None:
        wgts = np.atleast_2d(np.ones_like(ptcls)) / ptcls.shape[1]
    # check if wgts really all sum to one
    #assert np.isclose(np.sum(wgts, axis=1), np.ones_like(wgts.shape[1]), atol=0.001)
    first_term = np.sum(wgts * A_normal(h_obs - ptcls, h_obs_std**2), axis=1)
    forecasts_diff = (np.expand_dims(ptcls, -1) - np.expand_dims(ptcls, -2))
    weights_matrix = (np.expand_dims(wgts, -1) * np.expand_dims(wgts, -2))
    second_term = -0.5 * np.nansum(weights_matrix * A_normal((forecasts_diff), 2 * h_obs_std**2))
    return first_term + second_term


def crps_by_water_equivalent(aepf, h_obs, h_obs_std, obs_ix, mb_ix, obs_init_mb_ix):
    wgts = aepf.weights[obs_ix, :][0]
    ptcls = aepf.particles[obs_ix, :, mb_ix] - aepf.particles[obs_ix,:,obs_init_mb_ix]
    len_weights = len(wgts)
    first_term = np.sum(wgts * A_normal(cfg.RHO / 1000. * h_obs - ptcls, (
                cfg.RHO / 1000. * h_obs_std**2)), axis=1)
    second_term = - 0.5 * np.sum([np.sum((wgts * wgts[l])[:, np.newaxis] * np.abs(ptcls.T - ptcls[:, l]), axis=0)
         for l in range(len_weights)]) - 0.5 * A_normal(0., 2. * (
                cfg.RHO / 1000 * h_obs_std**2))
    # this stays 2D -> select [0]
    return first_term + second_term[:, 0]


def prepare_prior_parameters(gdir, mb_models, init_particles_per_model,
                             param_prior_distshape,
                             param_prior_std_scalefactor, generate_params=None,
                             param_dict=None, adjust_pelli=True, seed=None):
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
        The way we want to generate parameters.
    param_dict :
        If the parameters shall stem from this dictionary only.

    Returns
    -------

    """

    # get prior parameter distributions
    # todo: the np.abs is to get c0 in OerlemansModel positive!!!! (we take the
    #  log afterwards)
    if (generate_params == 'past') or (generate_params is None):
        theta_priors = [np.abs(get_prior_param_distributions(gdir, m,
                                                             init_particles_per_model,
                                                             fit=param_prior_distshape,
                                                             std_scale_fac=
                                                             param_prior_std_scalefactor[
                                                                 i], seed=seed))
                        for i, m in enumerate(mb_models)]
        theta_priors_means = [np.median(np.log(tj), axis=0) for tj in
                              theta_priors]
        theta_priors_cov = [np.cov(np.log(tj).T) for tj in theta_priors]

        #todo: it is necessary. but how to justify?
        # make tf in Pellicciotti noisy
        if adjust_pelli is True:
            pelli_prior = theta_priors[2]
            pelli_prior[:, 0] = np.clip(
                np.random.normal(6.5, 3.0, len(pelli_prior[:,
                                               0])), 2.0, 11.0)
            theta_priors[2] = pelli_prior

    elif generate_params == 'gabbi':
        # get mean and cov of log(initial prior)
        theta_priors = [np.abs(get_prior_param_distributions_gabbi(m,
                                                                   init_particles_per_model,
                                                                   fit=param_prior_distshape))
                        for m in mb_models]

        theta_priors_means = [np.mean(np.log(tj), axis=0) for tj in
                              theta_priors]
        theta_priors_cov = [np.cov(np.log(tj).T) for tj in theta_priors]
        theta_priors_cov = [np.eye(t.shape[0]) * t for t in theta_priors_cov]
    elif generate_params == 'mean_past':
        theta_priors = [np.abs(get_prior_param_distributions(gdir, m,
                                                             init_particles_per_model,
                                                             fit=param_prior_distshape,
                                                             std_scale_fac=
                                                             param_prior_std_scalefactor[
                                                                 i], seed=seed))
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
        theta_priors = [np.abs(get_prior_param_distributions_gabbi(m,
                                                                   init_particles_per_model))
                        for m in mb_models]
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
                         exclude_initial_snow=True):
    """
    Prepares the camera observations for usage.

    Returns
    -------

    """
    if stations is None:
        stations = id_to_station[gdir.rgi_id]
    station_hgts = [station_to_height[s] for s in stations]

    obs_merge = utils.prepare_holfuy_camera_readings(gdir, ice_only=ice_only,
                                                     exclude_initial_snow=exclude_initial_snow,
                                                     stations=stations)
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

def reference_run_constant_param():
    """
    Perform a reference run with constant parameters.

    todo: this function is misplaced!

    Returns
    -------

    """

    results = pd.read_csv('C:\\Users\Johannes\Documents\crampon\data'
                          '\\point_calibration_results_new.csv')
    res_list = []
    mad_list = []
    for i in range(len(results)):
        rdict = results.iloc[i].to_dict()
        gdir = utils.GlacierDirectory(rdict['RGIId'],
                                      base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier')
        fl_h, fl_w = gdir.get_inversion_flowline_hw()
        stations = id_to_station[gdir.rgi_id]
        station_hgts = [station_to_height[s] for s in stations]
        obs_merge = utils.prepare_holfuy_camera_readings(gdir, ice_only=False,
                                                         exclude_initial_snow=True,
                                                         stations=stations)
        obs_merge = obs_merge.sel(height=station_hgts)
        quadro_result = make_mb_current_mbyear_heights(gdir,
                                                                 begin_mbyear=pd.Timestamp(
                                                                     '2018-10-01'),
                                                                 param_dict=rdict,
                                                                 write=False,
                                                                 reset=False)
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
        1001:{'mb': (0., 0.), 'alpha':('from_model'), 'tacc':('from_model'),
              'swe': (0.9, 0.15)},
        1002:{'mb': (0., 0.), 'alpha':('from_model'), 'tacc':('from_model'),
              'swe':('from_model')},
        1003:{'mb': (0., 0.), 'alpha':('from_model'), 'tacc':('from_model'),
              'swe':('from_model')},
        1006:{'mb': (0., 0.), 'alpha':('from_model'), 'tacc':('from_model'), 'swe':('from_model')},
        1007:{'mb': (0., 0.), 'alpha':('from_model'), 'tacc':('from_model'), 'swe':('from_model')},
        1008:{'mb': (0., 0.), 'alpha':('from_model'), 'tacc':('from_model'), 'swe':('from_model')},
        1009:{'mb': (0., 0.), 'alpha':('from_model'), 'tacc':('from_model'), 'swe':('from_model')},
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

    # todo: the suquare root in the equation comes from Ross' slides...why
    is it thee?

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
        Lag distances between to measurements giving the according coavriance.

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
                                               unc_method: str = 'linear') -> tuple:

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
    obs_std: float
        The measurement uncertainty given as the standard deviation (m w.e.).
    prec_sol_at_heights: np.array
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
    # todo: test multiple assims
    #m_index = np.argmin(np.abs(fl_heights - obs_hgt))
    m_index = np.argmin(np.abs(fl_heights - np.atleast_2d(obs_hgt).T), axis=1)

    #delta_h = fl_heights - fl_heights[m_index]
    #mb_diff = (model_mb[:, m_index] - model_mb.T).T
    delta_h = fl_heights - np.atleast_2d(fl_heights[m_index]).T
    mb_diff = (np.atleast_3d(model_mb[:, m_index].T) - np.atleast_3d(
        model_mb.T).T).T

    # distribute the obs along the mb gradient
    #obs_distr = obs - np.nanmedian(mb_diff, axis=0)
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
    #mod_std = np.std(mb_diff, axis=0)
    if unc_method == 'linear':
        # todo: test multiple
        #meas_std_distr = distribute_std_by_linear_distance(delta_h, obs_std,
        #                                                   mod_std)
        meas_std_distr = distribute_std_by_linear_distance(delta_h,
                                                           np.atleast_2d(obs_std).T,
                                                           mod_std.T)
    elif unc_method == 'idw':
        meas_std_distr = distribute_std_by_inverse_distance(delta_h, obs_std,
                                                           mod_std)
    elif unc_method is None:
        meas_std_distr = np.ones_like(mod_std, dtype=np.float32) * obs_std
    else:
        raise ValueError('Method for extrapolating uncertainty is not '
                         'supported.')

    return obs_distr, meas_std_distr


def distribute_std_by_linear_distance(dist: np.ndarray, obs_std: float,
                                      mod_std: float) -> np.ndarray:
    """
    Distribute the standard deviation of the observation linearly with distance.

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
    #std_distr = obs_std * weights + mod_std * (1 - weights)
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
                #np.random.shuffle(arr)
                disarrange(arr, axis=0)  # todo: is axis=0 correct for every case?
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
    # todo: finish!?
    From https://github.com/numpy/numpy/issues/5173
    """
    b = a.swapaxes(axis, -1)
    # Shuffle `b` in-place along the last axis.  `b` is a view of `a`,
    # so `a` is shuffled in place, too.
    shp = b.shape[:-1]
    for ndx in np.ndindex(shp):
        np.random.shuffle(b[ndx])
    return


def plot_gradient(gradient: np.ndarray) -> None:
    """
    Plot gradient.

    todo: No ide what this function was for

    Parameters
    ----------
    gradient: array

    Returns
    -------
    None
    """
    from mpl_toolkits.mplot3d import Axes3D

    mean = np.nanmean(gradient, axis=0)
    std = np.nanstd(gradient, axis=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for n in np.arange(mean.shape[0]):
        dist = stats.norm(mean[n], std[n]).pdf(
            np.linspace(np.nanpercentile(mean - std, 2),
                        np.nanpercentile(mean + std, 98), 500))
        ax.plot(np.linspace(np.nanpercentile(mean - std, 2),
                            np.nanpercentile(mean + std, 98), 500),
                dist, n)
    plt.show()


def make_mb_current_mbyear_particle_filter(gdir: utils.GlacierDirectory,
                                           begin_mbyear: pd.Timestamp,
                                           mb_model: MassBalanceModel = None,
                                           snowcover: SnowFirnCover = None,
                                           write: bool = True,
                                           reset: bool = False,
                                           filesuffix: str = '') -> xr.Dataset:
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

    yesterday = min(utils.get_cirrus_yesterday(), begin_mbyear+pd.Timedelta(days=366))
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
    #stations = id_to_station['RGI50-11.B5616n-1']
    obs_merge = utils.prepare_holfuy_camera_readings(gdir)

    # todo: new format
    #meas['obs_we'] = obs_operator_dh_stake_to_mwe(meas['dh'])
    #meas['obs_std_we'] = obs_operator_dh_stake_to_mwe(meas['uncertainty'])
    # todo: minus to revert the minus from the function! This is shit!
    obs_std_we_default = - utils.obs_operator_dh_stake_to_mwe(cfg.PARAMS['dh_obs_manual_std_default'])
    #station_hgt = station_to_height[station]
    station_hgts = [station_to_height[s] for s in stations]
    #s_index = np.argmin(np.abs(heights - station_hgt))
    s_index = np.argmin(np.abs((heights - np.atleast_2d(station_hgts).T)), axis=1)

    weights = None
    N_particles = 10000  # makes it a bit faster
    n_eff_thresh = get_effective_n_thresh(N_particles)

    mb_spec_assim = np.full((100, len(curr_year_span) + 1), np.nan)
    mb_spec_assim[:, 0] = 0.

    tasks.update_climate(gdir)
    _ = make_mb_current_mbyear_heights(gdir, begin_mbyear=begin_mbyear, reset=True,
                                   filesuffix=filesuffix)

    gmeteo = climate.GlacierMeteo(gdir)

    mb_mod = gdir.read_pickle('mb_current_heights' + filesuffix)
    mb_mod_stack = mb_mod.stack(ens=('member', 'model'))
    mb_mod_stack_ens = mb_mod.stack(ens=('member', 'model')).MB.values

    # todo: so war es vorher
    mb_mod_all = mb_mod_stack.median(dim=['ens'], skipna=True).MB.values

    # todo: so war es vorher
    mb_mod_std_all = mb_mod_stack.std(dim=['ens'], skipna=True).MB.values
    # todo: once scipy is updated to 1.3.0, use this:
    # mb_mod_std_all = mb_mod_stack.apply(stats.median_absolute_deviation, axis=?, nan_policy='ignore')

    #experiment für std um median
    #med = np.nanmedian(mb_mod_stack.cumsum(dim='time').MB.values, axis=2)
    #mb_mod_std_all = np.nanmedian(
    #    np.abs(mb_mod_stack.cumsum(dim='time').MB.values - np.atleast_3d(med)),
    #    axis=2)

    #mb_mod_std_all = mb_mod_stack.std(dim='ens', skipna=True).cumsum(dim='time',  # this is to test if we have to use the cumsum std
    #                                                  skipna=True).MB.values


    mbscs = mb_mod_stack.cumsum(dim='time', skipna=True)
    mbscs = mbscs.where(~np.isnan(mb_mod_stack))
    mbscs = mbscs.where(mbscs != 0.)
    mb_mod_cumsum_std_all = mbscs.std(dim='ens', skipna=True).MB.values
    # todo: once scipy is updated to 1.3.0, replace this:
    #arr = mb_mod.sel(fl_id=s_index[0]).cumsum(dim='time', skipna=True).stack(
    #    dim=['member', 'model']).MB.values
    #scale = 1.4826
    #med = np.apply_over_axes(np.nanmedian, arr, 1)
    #mad = np.median(np.abs(arr - med), axis=1)
    # mb_mod_cumsum_std_all  = scale * mad
    mb_mod_cumsum_all = mbscs.cumsum(dim='ens', skipna=True).MB.values

    #mb_mod_cumsum_std_all = mb_mod_stack.cumsum(
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
                mb_mod_all[s_index, max(date_ix - 10, 0):date_ix + 1],
                axis=1) - np.nanmean(obs_merge.sel(
                date=slice(max(first_date, date - dt.timedelta(days=10)),
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
                    #np.atleast_2d(mb_mod_cumsum_std_date),
                    (N_particles, len(heights)))
            # there was already a day with OBS before and weights is not None:
            else:  # predict with model + uncertainty
                # trick: add std deviation randomly to all particles
                # strictly speaking for each particles in a row randn should be ran separately, but particle number should be big enough to make everything random
                print('MODEL:', mb_mod_date[s_index[0]], mb_mod_std_date[s_index[0]])

                # todo: correct mean and std if mb is zero => this should be base on how close temperature is to zero deg; the further away, the smaller the MB
                if (mb_mod_date == 0.).any():
                    possible_maxtemp = gmeteo.get_tmean_at_heights(
                        date) + climate.interpolate_mean_temperature_uncertainty(
                        np.array([date.month]))
                    positive_part = np.clip(possible_maxtemp, 0, None)
                    possible_melt = BraithwaiteModel.cali_params_guess['mu_ice']/1000. * positive_part
                    # set all zeros to a min value (we need stddev for the method to work)
                    possible_melt[possible_melt == 0.] = 0.0001
                    mb_mod_std_date[mb_mod_date == 0.] = possible_melt[mb_mod_date == 0.]

                # make the predict step
                particles_old = copy.deepcopy(particles)
                particles = predict_particle_filter(particles, mb_mod_date, mb_mod_std_date)

            mean_plot = np.nanmean(particles[:, s_index])
            std_plot = np.nanstd(particles[:, s_index])
            model_violin = particles[:, s_index[0]]
            print('Particle mean/std after predict: ', mean_plot, std_plot)

            # todo: THIS IS CHANGED FOR THE DEBIASING TEST
            #obs_we = mb_all_assim[s_index, date_ix-1] + obs_merge.sel(date=date).swe.values
            obs_we = mb_all_assim[s_index, date_ix - 1] + obs_merge.sel(
                date=date).swe.values + bias
            # todo: this is new for truncate
            obs_we_old = mb_all_assim[s_index, date_ix - 1]
            bias_list.append(bias)
            print('bias: ', bias)
            #obs_std_we = mb_all_assim_std[s_index, date_ix-1] + obs_merge.sel(date=date).swe_std.values
            obs_std_we = obs_merge.sel(date=date).swe_std.values

            # check if there is given uncertainty, otherwise switch to default
            if np.isnan(obs_std_we).any():
                obs_std_we[np.where(np.isnan(obs_std_we))] = obs_std_we_default

            # delete some NaNs from ensemble
            #ens_for_dist = mb_mod_stack_ens[:, date_ix, :].T[
            #    ~np.isnan(mb_mod_stack_ens[:, date_ix, :].T).any(axis=1)]

            # extrapolate the measurement along the flowlines
            # todo: => unused for the moment!
            #_, psol_at_hgts = gmeteo.get_precipitation_solid_liquid(date, heights)
            #obs_we_extrap, obs_std_we_extrap = distribute_point_mb_measurement_on_heights(
            #    heights, ens_for_dist, obs_we, station_hgts,
            #    obs_std_we, psol_at_hgts, unc_method='linear')

            ## if we have an OBS, and it's not the first one
            #if date_ix > first_date_ix:
            #    # todo: why is one array transposed?
            #    obs_we_extrap_all[:, n_day+1] = obs_we_extrap[:, 0]
            #    obs_we_std_extrap_all[:, n_day+1] = obs_std_we_extrap[0, :]
            #    obs_we_extrap = np.nansum(obs_we_extrap_all, axis=1)
            #    obs_std_we_extrap = np.nanmean(obs_we_std_extrap_all, axis=1)
            #    print('OBS: ', obs_we_extrap[s_index],
            #          obs_std_we_extrap[s_index])

            if np.isnan(weights).any():
                print('THERE ARE NAN WEIGHTS')

            # todo: new method to digest multiple measurements at once without extrapolating
            # iterate over stations, find new weights for each, and take mean
            new_weights = np.zeros((particles.shape[0], len(s_index)))
            for i, s in enumerate(s_index):
                checka = copy.deepcopy(weights[:, s])
                new_weights[:, i] = update_particle_filter(particles[:, s],
                                                     checka, obs_we[i],
                                                     obs_std_we[i], truncate=particles_old[:, s] - obs_we[i], obs_old=obs_we_old[i])  # todo: this is new for truncate
            new_weights = np.nanmean(new_weights, axis=1)
            new_weights = np.repeat(np.atleast_2d(new_weights).T, weights.shape[1], axis=1)

            weights = copy.deepcopy(new_weights)
            if np.isnan(weights).any():
                raise ValueError('Some weights are NaN.')

            # estimate new state
            mb_mean_assim, mb_var_assim = estimate_state(particles, weights)
            mb_std_assim = np.sqrt(mb_var_assim)

            mb_all_assim[:, n_day + 1] = mb_mean_assim
            mb_all_assim_std[:, n_day+1] = mb_std_assim

            # resample the distribution if necessary
            particles, weights = resample_particles(particles, weights,
                                                    n_eff_thresh=n_eff_thresh)
            print('UNIQUE PARTICLES: ', len(np.unique(particles[:, s_index[0]])))
            print('MEAN: ', np.mean(particles[:, s_index[0]]))

            mb_spec_assim[:, n_day + 1] = np.nanpercentile(np.average(np.sort(particles, axis=0), weights=widths, axis=1), np.arange(0, 100)) + mb_spec_assim[:, first_date_ix]

            # plot some stuff for visualization
            #plt.errorbar(n_day + 0.04, mb_all_assim[s_index[0], n_day+1],
            #             yerr=mb_std_assim[s_index[0]], fmt='o', label='POST',
            #             c='b')

            # POST
            ##test, _ = resample_particles(particles, weights, n_eff_thresh=N_particles)
            #vv = plt.violinplot(np.random.normal(mb_all_assim[s_index[0], n_day+1], mb_std_assim[s_index[0]], 5000), [n_day], showmeans=False, showextrema=False, showmedians=False)
            ##vv = plt.violinplot(test[s_index[0]], [n_day])
            #for pc in vv['bodies']:
            #    pc.set_facecolor('blue')
            #    pc.set_edgecolor('blue')
            #    pc.set_alpha(0.5)
            ##plt.scatter(n_day+0.04-weights[:, s_index[0]], particles[:, s_index[0]])

            ## MODEL
            ## todo: change this back???
            ##vv = plt.violinplot(mb_mod_stack.sel(time=date, fl_id=s_index[0]).MB.dropna(dim='ens').values, [n_day])
            ##for n_obs in np.arange(mean_plot.shape[1]):
            #vv = plt.violinplot(np.random.normal(mean_plot, std_plot, 5000), [n_day], showmeans=False, showextrema=False, showmedians=False)
            #for pc in vv['bodies']:
            #    pc.set_facecolor('g')
            #    pc.set_edgecolor('g')
            #    pc.set_alpha(0.5)

            #plt.errorbar([n_day+0.02], mean_plot, label='MODEL', yerr=std_plot, fmt='o', c='g')

            ## OBS
            #vv = plt.violinplot(np.random.normal(obs_we[-1],obs_std_we[-1], 5000), [n_day], showmeans=False, showextrema=False, showmedians=False)
            #for pc in vv['bodies']:
            #    pc.set_facecolor('r')
            #    pc.set_edgecolor('r')
            #    pc.set_alpha(0.5)
            ###print('hello')
        # if date does not have OBS or OBS is NaN
        else:
            #mb_spec_assim[:, n_day + 1] = np.pad(
            #    np.average(mbscs.sel(time=date).MB.values, weights=widths,
            #               axis=0), ((0, N_particles - mbscs.ens.size)),
            #    'constant', constant_values=(np.nan, np.nan))

            #mb_spec_assim[:, n_day + 1] = mb_all_assim[:, first_date_ix:x_len] + mb_spec_assim[:, first_date_ix - 1]
            # until first day - does this make sense?
            if weights is None:
                mb_all_assim[:, n_day + 1] = mb_mod_date
                mb_all_assim_std[:, n_day+1] = mb_mod_std_date
                mb_spec_assim[:, n_day + 1] = np.nanpercentile(
                    np.average(mbscs.sel(time=date).MB.values, weights=widths,
                               axis=0), np.arange(0, 100))

            # all other days
            else:
                # write already the MB for the next day here
                particles_prediction = predict_particle_filter(particles,
                                                               mb_mod_date,
                                                               mb_mod_std_date)
                # after prediction, weights should be one again, so no average
                mb_all_assim[:, n_day+1] = np.mean(particles_prediction, axis=0)
                mb_all_assim_std[:, n_day+1] = np.std(particles, axis=0)
                mb_spec_assim[:, n_day + 1] = np.nanpercentile(
                    np.average(mb_mod_stack.sel(time=date).MB.values, weights=widths,
                               axis=0), np.arange(0, 100)) + mb_spec_assim[:, n_day]
    if date_ix + 1 < 365:
        x_len = date_ix + 1
    else:
        x_len = 365

    # fake legend for violin plot
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    blue_patch = mpatches.Patch(color='blue')
    green_patch = mpatches.Patch(color='green')
    darkgr_patch = mpatches.Patch(color='red')
    fake_handles = [blue_patch, green_patch, darkgr_patch]
    fake_labels = ['ASSIM','MODEL','OBS']
    plt.legend(fake_handles, fake_labels, fontsize=20)
    #plt.setp(ax.get_xticklabels(), fontsize=20)
    #plt.setp(ax.get_yticklabels(), fontsize=20)
    plt.grid()
    #plt.plot(np.arange(first_date_ix, date_ix + 1),
    #         np.cumsum(mb_mod_all[s_index[0], first_date_ix:]))
    #plt.plot(np.arange(first_date_ix, date_ix + 1), mb_all_assim[s_index, first_date_ix:][0][1:])

    for s in np.arange(len(s_index)):
        fig, ax = plt.subplots()
        ax.errorbar(np.arange(x_len),
                     np.cumsum(mb_mod_all[s_index[s]])[:x_len],
                     mb_mod_cumsum_std_all[s_index[s]][:x_len], elinewidth=0.5, label='model')
        both = np.cumsum(mb_mod_all[s_index[s]])[:x_len]
        both_std = mb_mod_cumsum_std_all[s_index[s]][:x_len]
        both[first_date_ix:] = mb_all_assim[s_index[s], first_date_ix:x_len] + \
                               both[first_date_ix - 1]
        both_std[first_date_ix:] = mb_all_assim_std[s_index[s],
                                   first_date_ix:x_len] + both_std[
                                       first_date_ix - 1]
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
    print('hi')


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

    print(pf)

    return pf


def plot_pdf_model_obs_post(obs: float, obs_std: float, mod: float,
                            mod_std: float, post: float,
                            post_std: float)-> None:
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

    plt.plot(x_range,
             stats.norm(obs, obs_std).pdf(
                 x_range), label='OBS')
    plt.plot(x_range,
                 stats.norm(post,
                            post_std).pdf(
                 x_range), label='POST')
    plt.plot(x_range,
             stats.norm(mod, mod_std).pdf(
                 x_range), label='MODEL')
    plt.legend()
    plt.show()


def make_mb_current_mbyear_heights(gdir: utils.GlacierDirectory,
                                   begin_mbyear: pd.Timestamp,
                                   last_day: dt.datetime or pd.Timestamp or
                                             None = None,
                                   mb_model: MassBalanceModel = None,
                                   snowcover: SnowFirnCover = None,
                                   write: bool = True, reset: bool = False,
                                   filesuffix:str = '', param_dict=None,
                                   constrain_bw_with_prcp_fac: bool = True) -> \
xr.Dataset:
    """
    Make the mass balance at flowline heights of the current mass budget year.

    Parameters
    ----------
    gdir: `py:class:crampon.GlacierDirectory`
        The GlacierDirectory to calculate the current mass balance for.
    begin_mbyear: datetime.datetime
        The beginning of the current mass budget year.
    mb_model: `py:class:crampon.core.models.massbalance.DailyMassBalanceModel`,
              optional
        A mass balance model to use. Default: None (use all available).
    snowcover: `py:class:crampon.core.models.massbalance.SnowFirnCover`
        The snow/firn cover to sue at the beginning of the calculation.
        Default: None (read the acccording one from the GlacierDirectory)
    write: bool
        Whether or not to write the result to GlacierDirectory. Default: True
        (write out).
    reset: bool
        Whether to completely overwrite the mass balance file in the
        GlacierDirectory or to append (=update with) the result. Default: False
        (append).
    filesuffix: str
        Suffix to be used for the mass balance calculation.

    Returns
    -------
    mb_now_cs: xr.Dataset
        Mass balance of current mass budget year.
    """

    if mb_model:
        mb_models = [mb_model]
    else:
        mb_models = [eval(m) for m in cfg.MASSBALANCE_MODELS]

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

    curr_year_span = pd.date_range(start=begin_str, end=yesterday_str,
                                   freq='D')
    ds_list = []
    heights, widths = gdir.get_inversion_flowline_hw()

    conv_fac = cfg.FLUX_TO_DAILY_FACTOR#((86400 * cfg.RHO) / cfg.RHO_W)  # m
    # ice s-1 to m w.e.
    # d-1

    sc_list_one_model = []
    sfc_obj_list_one_model = []
    alpha_list = []
    tacc_list = []
    for mbm in mb_models:
        stacked = None

        pg = ParameterGenerator(gdir, mbm, latest_climate=True,
                                      only_pairs=True,
                                      constrain_with_bw_prcp_fac=constrain_bw_with_prcp_fac,
                                      bw_constrain_year=begin_mbyear.year + 1,
                                      narrow_distribution=0.,
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

            # todo: is tink the commented out version was wrong!
            #sc_list_one_model.append(np.nansum(sc.swe, axis=1))
            sc_list_one_model.append(np.nansum(day_model_curr.snowcover.swe, axis=1))
            sfc_obj_list_one_model.append(day_model_curr.snowcover)
            if hasattr(day_model_curr, 'albedo'):
                alpha_list.append(day_model_curr.albedo.alpha)
                tacc_list.append(day_model_curr.tpos_since_snowfall)

            if stacked is not None:
                stacked = np.vstack((stacked, np.atleast_3d(mb_temp).T))
            else:
                stacked = np.atleast_3d(mb_temp).T

        stacked = np.sort(stacked, axis=0)

        mb_for_ds = np.moveaxis(np.atleast_3d(np.array(stacked)), 1, 0)
        mb_for_ds.shape += (1,) * (4 - mb_for_ds.ndim)  # add dim for model
        # todo: units are hard coded and depend on method used above
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
    return ens_ds, sc_list_one_model, alpha_list, tacc_list, sfc_obj_list_one_model


def make_variogram(glob_path: str) -> None:
    """
    Plot a variogram of the Holfuy camera readings.

    # todo: what was the purpose of this function?
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
        #plt.title('Correlogram for station {}'.format(str(s)))
        #plt.xlabel('Absolute height distance (m)')
        #plt.xlim(0, None)
        #plt.ylabel('$R^2$')
    plt.title('Correlogram for station {}'.format(str(s)))
    plt.xlabel('Absolute height distance (m)')
    plt.xlim(0, None)
    plt.ylabel('$R^2$')


def crossval_multilinear_regression(glob_path: str) -> None:
    """
    Cross-validate a prediction of mass balances at the cameras from multilinear regression

    Parameters
    ----------
    glob_path: str
        Path to the manual Holfuy camera readings.

    Returns
    -------
    None.
    """
    import statsmodels.api as sm
    from scipy.spatial.distance import squareform, pdist

    use_horizontal_distance = True

    # todo: remove hard code
    hdata_path = 'c:\\users\\johannes\\documents\\holfuyretriever\\holfuy_data.csv'

    # 'C:\\users\\johannes\\documents\\holfuyretriever\\manual*.csv'
    flist = glob.glob(glob_path)
    meas_list = [utils.read_holfuy_camera_reading(f) for f in flist]
    meas_list = [m[['dh']] for m in meas_list]
    test = [m.rename({'dh': 'dh' + '_' + flist[i].split('.')[0][-4:]}, axis=1)
            for i, m in enumerate(meas_list)]
    conc = pd.concat(test, axis=1)
    conc_drop = conc.dropna(axis=0)

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
        #easy_cols = X.rename(
        #    {'dh_1001': 1001, 'dh_1002': 1002, 'dh_1003': 1003,
        #     'dh_1006': 1006,
        #     'dh_1007': 1007, 'dh_1008': 1008, 'dh_1009': 1009}, axis=1)
        #wgts = pd.concat([hdist[[int(n.split('_')[1]) for n in test_labels]].T,
        #                  easy_cols], axis=0).loc[[int(n.split('_')[1]) for n in test_labels]]
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
                  ' RMSE: ' + ('{:.3f}').format(
            np.sqrt(np.mean((predictions.values - y.values) ** 2))))


def variogram_cloud() -> None:
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


def simple_variogram() -> None:
    """
    Plot a simple variogram.

    # todo: as in Bivand et al???

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


def sample_variogram_100_variations() -> None:
    """
    Produces a figure similar to 8.5 in [Bivand et al. (2013)]_, but only points.

    Returns
    -------
    None

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
    fig, (ax1, ax2) = plt.subplots(2, 1)
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
        'Linear Slope and p-value for variogram vs. 100 variograms for randomly re-allocated data')
    ax2.set_xlabel('Slope ($\dfrac{d \hat{\gamma}}{d x}$)')
    ax2.set_ylabel('p-value')
    ax1.legend()
    ax2.legend()
    plt.show()


def sample_variogram_from_residuals() -> None:
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

    """
    eps_list = []
    station_list = []
    # get residuals - the mean varies in space
    for s in conc.columns.values:
        id = station_to_glacier[int(s.split('_')[1])]
        s_hgt = station_to_height[int(s.split('_')[1])]

        gd = GlacierDirectory(id,
                              base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\')
        gm = GlacierMeteo(gd)
        sis = gm.meteo.sel(time=conc.index).sis
        tmean = gm.meteo.sel(time=conc.index).temp
        tgrad = gm.meteo.sel(time=conc.index).tgrad
        tmean_at_hgt = tmean + tgrad * (s_hgt - gm.ref_hgt)

        notnan_ix = \
        np.where(~np.isnan(sis) & ~np.isnan(tmean_at_hgt) & ~pd.isnull(conc[s]))[0]

        y = conc[s][notnan_ix]
        X = sm.add_constant(np.vstack((sis[notnan_ix], tmean_at_hgt[notnan_ix])).T)

        model = sm.OLS(y, X).fit()
        eps_list.append(np.mean(model.resid))
        station_list.append(s)

    # calculate semivariance gamma hat
    variance_list = []
    dist_list = []
    for c1, c2 in permutations(station_list, 2):
        variance_list.append(np.square(
            eps_list[station_list.index(c1)] - eps_list[
                station_list.index(c2)]))
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
    plt.ylim(min(gamma_hat), max(gamma_hat))
    """

    # this alternative takes the model residuals as they are (no mean) and calculates the mean only when they are subtracted:
    # the point being: the semivariance values are more reasonable (10e-4 instead of 10e-33)
    # todo: check, but I think this is true:
    eps_list = []
    station_list = []
    for s in conc.columns.values:
        id = station_to_glacier[int(s.split('_')[1])]
        s_hgt = station_to_height[int(s.split('_')[1])]
        gd = GlacierDirectory(id,
                              base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\')
        gm = GlacierMeteo(gd)
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


def camera_randomforestregressor() -> None:
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
            # print(X.drop(X.index[n]).drop([drop_col], axis = 1), y.drop(y.index[n]))
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


def try_gaussian_proc_regression():
    """
    Try Gauss process regression to model observational random error in space.

    Returns
    -------

    """
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    np.random.seed(1)
    # Mesh the input space for evaluations of the real function, the prediction and
    # its MSE
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
    kernel = C(1000.0, (1e1, 1e6)) * RBF(1000, (1e1, 1e4))
    # Instantiate a Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, alpha=dy ** 2,
                                  n_restarts_optimizer=10)
    # Fit to data using Maximum Likelihood Estimation of the parameters
    gp.fit(X, y)
    # Make the prediction on the meshed x-axis (ask for MSE as well)
    y_pred, sigma = gp.predict(x, return_std=True)
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
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


def linreg_of_station_regressions_perturbed() -> None:
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

        # gdir_1 = GlacierDirectory(id_1, base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\')
        # gdir_2 = GlacierDirectory(id_2,
        #                          base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\')

        # gm_1 = GlacierMeteo(gdir_1)
        # gm_2 = GlacierMeteo(gdir_2)

        conccopy = conc[[s1, s2]].copy()
        conccopy.dropna(axis=0, inplace=True)

        for n in range(iterations):
            noise = np.random.normal(mu, sigma, [len(conccopy), 2])
            conccopy = conccopy + noise

            # sun_1 = gm_1.meteo.sel(time=conccopy.index).sis
            # sun_2 = gm_2.meteo.sel(time=conccopy.index).sis
            # sun_dev_1 = sun_1 - np.nanmean(sun_1) / np.nanmean(sun_1)
            # sun_dev_2 = sun_2 - np.nanmean(sun_2) / np.nanmean(sun_2)
            # sun_notnan = np.where(~np.isnan(sun_dev_1) | ~np.isnan(sun_dev_2))[0]
            # print(sun_dev_1, sun_dev_2)
            # weighted with incoming solar
            # y = conccopy[s2][sun_notnan]*sun_dev_2[sun_notnan]
            # X = sm.add_constant(conccopy[s1][sun_notnan]*sun_dev_1[sun_notnan].values)

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
            large_leverage = leverage > statsmodels.graphics.regressionplots._high_leverage(
                model)
            large_points = np.logical_or(large_resid, large_leverage)

            y_corr = y[~large_points]
            X_corr = X[~large_points]

            model_corr = sm.OLS(y_corr, X_corr).fit()

            # plt.figure()

            # plt.scatter(X_corr.values[:, 1], y_corr)
            # plt.plot(np.arange(np.min(X_corr.values[:, 1]), np.max(X_corr.values[:, 1])+0.01, 0.01), model_corr.params[0] + np.arange(np.min(X_corr.values[:, 1]), np.max(X_corr.values[:, 1])+0.01, 0.01) * model_corr.params[1])
            # plt.xlabel(s1)
            # plt.ylabel(s2)

            # x_low, _ = plt.xlim()
            # y_low, _ = plt.ylim()
            # plt.annotate(
            #    'R2: {:.2f}\np-val: {:.2f}'.format(model_corr.rsquared, model_corr.pvalues[1]),
            #    (x_low, y_low))

            result_df.loc[i * n + n, 's1'] = int(s1.split('_')[1])
            result_df.loc[i * n + n, 's2'] = int(s2.split('_')[1])
            result_df.loc[i * n + n, 'hdist'] = hdist[int(s1.split('_')[1])][
                hdist.index == int(s2.split('_')[1])].item()
            result_df.loc[i * n + n, 'vdist'] = abs(
                assimilation.station_to_height[int(s1.split('_')[1])] -
                assimilation.station_to_height[int(s2.split('_')[1])])
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
    large_leverage = leverage > statsmodels.graphics.regressionplots._high_leverage(
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


def linreg_of_stations_regressions_kickout_outliers() -> None:
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

        # gdir_1 = GlacierDirectory(id_1, base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\')
        # gdir_2 = GlacierDirectory(id_2,
        #                          base_dir='C:\\users\\johannes\\documents\\modelruns\\CH\\per_glacier\\')

        # gm_1 = GlacierMeteo(gdir_1)
        # gm_2 = GlacierMeteo(gdir_2)

        conccopy = conc[[s1, s2]].copy()
        conccopy.dropna(axis=0, inplace=True)

        for n in range(iterations):
            noise = np.random.normal(mu, sigma, [len(conccopy), 2])
            conccopy = conccopy + noise

            # sun_1 = gm_1.meteo.sel(time=conccopy.index).sis
            # sun_2 = gm_2.meteo.sel(time=conccopy.index).sis
            # sun_dev_1 = sun_1 - np.nanmean(sun_1) / np.nanmean(sun_1)
            # sun_dev_2 = sun_2 - np.nanmean(sun_2) / np.nanmean(sun_2)
            # sun_notnan = np.where(~np.isnan(sun_dev_1) | ~np.isnan(sun_dev_2))[0]
            # print(sun_dev_1, sun_dev_2)
            # weighted with incoming solar
            # y = conccopy[s2][sun_notnan]*sun_dev_2[sun_notnan]
            # X = sm.add_constant(conccopy[s1][sun_notnan]*sun_dev_1[sun_notnan].values)

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
            large_leverage = leverage > statsmodels.graphics.regressionplots._high_leverage(
                model)
            large_points = np.logical_or(large_resid, large_leverage)

            y_corr = y[~large_points]
            X_corr = X[~large_points]

            model_corr = sm.OLS(y_corr, X_corr).fit()

            # plt.figure()

            # plt.scatter(X_corr.values[:, 1], y_corr)
            # plt.plot(np.arange(np.min(X_corr.values[:, 1]), np.max(X_corr.values[:, 1])+0.01, 0.01), model_corr.params[0] + np.arange(np.min(X_corr.values[:, 1]), np.max(X_corr.values[:, 1])+0.01, 0.01) * model_corr.params[1])
            # plt.xlabel(s1)
            # plt.ylabel(s2)

            # x_low, _ = plt.xlim()
            # y_low, _ = plt.ylim()
            # plt.annotate(
            #    'R2: {:.2f}\np-val: {:.2f}'.format(model_corr.rsquared, model_corr.pvalues[1]),
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
    large_leverage = leverage > statsmodels.graphics.regressionplots._high_leverage(
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


def correlation_of_dh_residuals() -> None:
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


def spatial_correlation_of_station_correlations(alpha: float = 0.95) -> None:
    """
    Plots the station residual correlations ($R^2$) over horizontal distance.

    The plot includes a confidence interval.

    Parameters
    ----------
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
    ax.plot(np.arange(min(distlist) - 1., max(distlist) + 1., 1.),
            distmodel.params[0] + np.arange(min(distlist) - 1.,
                                        max(distlist) + 1., 1.) *
            distmodel.params[1], c='g',
            label='$R^2$: {:.2f},\np: {:.2f}'.format(model.rsquared,
                                                     model.pvalues[1]))
    _, stdata, _ = summary_table(distmodel, alpha=0.05)
    predict_mean_ci_low, predict_mean_ci_upp = stdata[:, 4:6].T
    ax.fill_between(sorted(distlist), sorted(predict_mean_ci_low, reverse=True),
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


def fit_exponential_decay():
    """
    Creepy function to fit an exponential decay to a R^2 over distance scatter plot.
    # todo: this function is horribly hard-coded.
    Returns
    -------

    """
    from scipy.optimize import curve_fit
    def fit_func(dist, a, c):
        return c * (np.exp(1 - (dist / a)))

    popt, pcov = curve_fit(fit_func, result_df.hdist, result_df.R2)
    plt.scatter(result_df.hdist, result_df.R2, label='True')
    plt.plot(np.arange(0, 80000, 1000),
             fit_func(np.arange(0, 80000, 1000), *popt), 'g--',
             label='fit: a=%5.3f, c=%5.3f' % tuple(popt))
    plt.legend()


def make_station_CV_boxplot() -> None:
    """
    Make a boxplot of the cross-validation error when predicting station mass
    balance.

    Returns
    -------
    None
    """
    import statsmodels.api as sm

    conc_drop = conc.dropna(axis=0)
    from sklearn.model_selection import LeavePOut
    labels = conc.columns.values
    leave_out = 1
    lpo = LeavePOut(leave_out)
    lpo.get_n_splits(labels)

    boxes = []
    labs = []
    for train_index, test_index in lpo.split(labels):
        train_labels = labels[train_index]
        test_labels = labels[test_index]
        X = conc_drop[train_labels]
        X = sm.add_constant(X)  # add an intercept
        y = conc_drop[test_labels]

        test_labels_int = [int(n.split('_')[1]) for n in test_labels]

        model = sm.OLS(y, X).fit()
        predictions = model.predict(X)
        print(model.summary())
        boxes.append(2 * ((predictions.values - y.values.flatten()) / (
                    np.abs(predictions.values) + np.abs(y.values.flatten()))))
        labs.append(' '.join(list(test_labels)))

    plt.boxplot(boxes, labels=labs)
    plt.axhline(0.)
    plt.legend()
    plt.xlabel('Station')
    plt.ylabel('Relative Percent Difference')
    plt.title(
        'Cross-validation from multilinear regression, relative percent difference')
    plt.savefig(
        'C:\\Users\\Johannes\\Desktop\\crampon_prelim_plots\\CV_boxplot_linear_prediction_relative_percent_diff.png')


if __name__ == '__main__':
    import geopandas as gpd
    from crampon import workflow
    from crampon import cfg
    import warnings
    from crampon import tasks
    import itertools

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

    dspan, mprobs, mparts = run_aepf('RGI50-11.B4312n-1', mb_models=None, generate_params='past',
              param_fit='lognormal', param_method='memory',
              change_memory_mean=True, make_init_cond=True,
              qhisto_by_model=False, adjust_pelli=False, pdata=pdata,
              reset_albedo_and_swe_at_obs_init=False, write_probs=True,
                                     adjust_heights=False, prior_param_dict=None,
                                     unmeasured_period_param_dict=None, crps_ice_only=False, use_tgrad_uncertainty=False)
    all_mprobs.append(mprobs)
    all_mparts.append(mparts)
    all_dspans.append(dspan)
    glacier_names.append('Rhonegletscher')


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