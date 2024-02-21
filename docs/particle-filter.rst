Particle Filtering
==================

In CRAMPON, we use an ensemble of mass balance models into which we assimilate in situ and remote observations.
To obtain the optimal result respecting both inputs and their respective uncertainties, we use the particle filter.
The particles filter is a sequential Monte Carlo data assimilation method that handles probability distributions as the sum of weights possibles states, so called "particles".
This has the advantage that we neither need to make assumptions how a variable is distributed (e.g. Gaussian), nor do we need to assume linearity of the mass balance model equations.
Linearity is not always given in the melt equations, since there are for example hard thresholds :math:`T > T_{melt}`.

However, we have four models about which we assume initially that they are all equally true, the standard particle mgiht run into a problem:
Since it might occure for a model that it perofmrs good an one day, but bad on the next day, we need to ensure that models are not lost from the ensemble - they might recover later and outperform the other models.
This is why we invented the augmented ensemble particle filter (AEPF).
The AEPF assigns to each model a minimum contribution in terms of particles to the overall ensemble, no matter how bad it performs on a particular day.
The compensate that this is wrong actually, it lowers the weights of the particles instead.
This has the advantage that the actual contribution of a model to the ensemble stays the same, however, it is ensured that there are still enough particles to describe variances sufficiently correct.
However, the world is not perfect.
It can happen that the particles weights of a model become very very small after two or three days that a model performed bad.
This requires again a certain set of measures to not let the ensemble collapse.
Here is a short description of the whole workflow.

1. set initial conditions
2. predict with model
3. update using observations
4. resample if necessary
     1. Check :math:`N_eff` per model
     2. Resample per model
     3. ensure that minimum particle size is kept and lower weights
     4. diversify parameters using the equation from [Liu and ]_.