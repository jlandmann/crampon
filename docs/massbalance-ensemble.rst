Ensemble Mass Balance Modeling
==============================

CRAMPON does its best to give the full picture of what mass balance could have been on a particular day on a particular glacier.
This approach does include a `multi-model ensemble` approach.
A multi-model ensemble lives on the idea that one model is not enough to model a process, but due to physicial deficiencies in individual models an added value is generated thorugh employing more models:
each of them is wrong in its own way, but together they make up for their individual flaws.
For temperature index mass balance modeling, this is exactly the case:
each of the melt model that CRAMPON uses is a simplification of the energy balance equation:

..math::


Since we don't have a good idea what especially the latent and sensible heat fluxes, as well as the long wave radiation fluxes look like, we cannot use this equation directly.
What temperature index models aim for instead is the best possible representation of the energy balance equation, but `parametrizing` unknown terms into tunable factors.
An review of the temperature index melt models and their performance can be found in [Hock (2003)]_ and [Gabbi et al.(2014)]_.

Moreover, it has to be clarified that, when we speak about "mass balance modeling", we actually always mean the surface mass balance.



Surface Accumulation
--------------------

To compute surface mass gain at different heights, we employ a simple, but widely used accumulation model in global modeling, as it can be found e.g. in [Huss and Hock (2015)]_ or [Maussion et al. (2019)]_:

.. math::
    C_{sfc}(t,z) = c_{prec}(t) \cdot P_s(t,z_{ref}) \cdot [ 1 + (z - z_{ref}) \cdot \frac{\partial{P}}{\partial{z}}]


where :math:`C_{sfc}(t,z)` is the surface accumulation at time step t and elevation z (calculated in :math:`m\, water\ equivalent\ t^{-1}`), :math:`c_{prec}(t)` is the unitless multiplicative precipitation correction factor, :math:`P_i(t,z_{ref})` is the sum of solid precipitation at height of the precipitation reference cell :math:`z_{ref}` and time step t, :math:`z` is the mean height of an elevation band, and :math:`\frac{\partial{P}}{\partial{z}}` is the precipitation lapse rate.
Following \citet{sevruk1985systematischer}, :math:`c_{prec}` varies :math:`\pm 8\%` periodically around its mean during one year, being highest in winter and lowest in summer.
This is to account for average variations in precipitation gauge undercatch.
The water phase change in the temperature range around 0 :math:`^\circ C` is modeled using a linear gradient between 0 :math:`^\circ C` and 2 :math:`^\circ C`.

If a glacier is in the \ac{GLAMOS} measurement  program and has winter mass balance measurements, it is also possible to account for the effect of snow redistribution:
by adjusting a factor :math:`D(z)` the model mass balance height gradient is tuned such that it better matches the interpolated distribution of measured winter mass balances:

.. math::
   D(z) \cdot B_{sfc}(z) = B_{sfc, interp}(z)


with :math:`B_{sfc}(z)` and :math:`B_{sfc, interp}(z)` being the modeled/interpolated winter mass balance measurements (:math:`m\ w.e.`).
The application of this factor seems irrelevant for a 1.5D-model that can have elevation model pixels subject to several different aspects and slope per elevation band.
However, the aggregation of 2D snow redistribution factors showed that it can still be crucial to apply, especially since precipitation gradients in the gridded data input are a considerable source of error.


Surface Ablation
----------------

To model ablation, we set up ensemble of four temperature index melt models.
The individual ensemble models differ in the degree of complexity they use to describe the surface energy balance \citep{hock2003temperature}.
Each of the models uses meteorological input as variables and has parameters that can be tuned in a calibration procedure.
The ensemble contains (symbols from \citet{cogley2011glossary}) the following models:

The classical "BraithwaiteModel"
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is an implementation of the model presented in [Braithwaite and Olesen (1989)]_ and [Braithwaite (1995)]_:

 .. math::
     A_{sfc}(t,z) = \mu_{snow/ice} \cdot max(T(t,z) - T_{melt}, 0)

where :math:`A_{sfc}(t,z)` is the melt at time step t and elevation z (:math:`m\,w.e.\ d^{-1}`), :math:`\mu_{snow/ice}` is the temperature sensitivity of the surface type (snow/ice) (:math:`m\,w.e.\ K^{-1} t^{-1}`), :math:`max()` is the maximum operator, :math:`T(t,z)` is the temperature at time step t and elevation z (:math:`^\circ C`) and :math:`T_{melt}` is the threshold temperature for melt (:math:`^\circ C`).
For this application, we set :math:`T_{melt}` to 0 :math:`^\circ C`.

The "HockModel"
^^^^^^^^^^^^^^^

extension of this model using potential incoming solar radiation as another predictor for melt \citep{hock1999distributed}:

.. math:: \label{eq:hockmodel}
            A_{sfc}(t,z) = (MF + a_{snow/ice} I_{pot}(t,z)) \cdot max(T_(t,z) - T_{melt}, 0)

where :math:`MF` is the temperature melt factor (:math:`m\ w.e.\ K^{-1} t^{-1}`), :math:`a_{snow/ice}` is the radiation coefficient for snow and ice, respectively (:math:`m\ w.e.\ m^2\ d^{-1} W^{-1}\ K^{-1}`), and :math:`I_{pot}(t,z)` is the potential clear-sky direct solar radiation at time t and elevation z.
:math:`I_{pot}(t,z)` is derived on the glacier grid as daily means from ten minute interval snapshots and then assigned to the respective flowline nodes.
A more detailed description of how potential irradiation is generated can be found under the :ref:`potential-irradiation` section.


The "PellicciottiModel"
^^^^^^^^^^^^^^^^^^^^^^^
version employing explicit surface albedo and actual solar irradiation \citep{pellicciotti2005enhanced}:

.. math::
        A_{sfc}(t,z) = \left\{\begin{array}{lr}
        TF \cdot T(t, z) + SRF \cdot (1 - \alpha(t,z)) \cdot G(t,z),&  \text{for } T(t, z) > T_{melt}\\
        0, & \text{for } T(t,z) \leq T_{melt}\\
        \end{array}\right.

where :math:`TF` is the temperature factor (:math:`m\ w.e.\ K^{-1} t^{-1}`), :math:`SRF` is the shortwave radiation factor (:math:`m^{3} t^{-1} W^{-1}`), :math:`\alpha(t,z)` is the albedo at time t and elevation z and :math:`G(t,z)` is the incoming shortwave radiation (:math:`W m^{-2}`) at time t and elevation z.
Note that in this case, as in the original publication, :math:`T_{melt}` is equal to 1 :math:`^\circ C`.

The "OerlemansModel"
^^^^^^^^^^^^^^^^^^^^
approach calculating melt energy as the residual term of a simplified surface energy balance equation \citep{oerlemans2001glaciers}:

.. math::
    Q_m(t,z) = (1-\alpha(t,z)) \cdot G(t,z) + c_0 + c_1 \cdot T(t,z)

.. math::
           A_{sfc}(t,z) = \frac{Q_m(t,z)\,dt}{L_f\,\rho_{w}}

where $:math:`Q_m(t,z)` is the melt energy (\:math:`W m^{-2}`) at time t and height z, :math:`c_0` and :math:`c_1` are empirical factors (:math:`W m^{-2}` and :math:`W m^{-2} K^{-1}`, respectively), dt is a time step (here 86400 s, :math:`L_f` is the latent heat of fusion (:math:`J\,kg^{-1}`), and :math:`\rho_w` is the density of water (:math:`kg\, m^{-3}`).



Albedo
------

Some of these equations require submodels for albedo.
Here, albedo is approximated according to the logarithmic decay equation of \citet{brock2000measurement} as applied in \citet{pellicciotti2005enhanced}:

\begin{equation} \label{eq:brockmodel}
        \alpha(t, z) = p1 - p2 \cdot log_{10}(T_{acc}(z))
\end{equation}

where p1 is the empirical albedo of fresh snow at 1 :math:`^\circ C` (here: 0.86), p2 is an empirical coefficient (here: 0.155), and :math:`T_{acc}(z)` is the accumulated daily maximum temperature >\,0\ :math:`^\circ C` since a snowfall event at height z.


References
----------
.. [Braithwaite and Olesen (1989)] Braithwaite, R. J., & Olesen, O. B. (1989). Calculation of glacier ablation from air temperature, West Greenland. In Glacier fluctuations and climatic change (pp. 219-233). Springer, Dordrecht.
.. [Braithwaite (1995)] Braithwaite, R. J. (1995). Positive degree-day factors for ablation on the Greenland ice sheet studied by energy-balance modelling. Journal of Glaciology, 41(137), 153-160.
.. [Gabbi et al. (2014)] Gabbi, J., Carenzo, M., Pellicciotti, F., Bauder, A., & Funk, M. (2014). A comparison of empirical and physically based glacier surface melt models for long-term simulations of glacier response. Journal of Glaciology, 60(224), 1140-1154.
.. [Hock (2003)] Hock, R. (2003). Temperature index melt modelling in           mountain areas. Journal of hydrology, 282(1-4), 104-115.
.. [Huss and Hock (2015)] Huss, M., & Hock, R. (2015). A new model for global glacier change and sea-level rise. Frontiers in Earth Science, 3, 54.
.. [Maussion et al. (2019)] Maussion, F., Butenko, A., Champollion, N., Dusch, M., Eis, J., Fourteau, K., ... & Recinos, B. (2019). The Open Global Glacier Model (OGGM) v1. 1. Geoscientific Model Development, 12(3), 909-931.