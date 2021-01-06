Snow Lines from Satellite Observations
======================================

Snow lines can also deliver an indirect hint on the mass balances of the glacier:
in spring, when the glacier tongue becomes snow free, the snow line is an indicator of the zero mass balance location since the glacier mass minimum in the previous autumn.
Even though this is not true anymore when a fresh snow fall "disturbs" the continuous "becoming snow free" of a glacier, the snow line is still an important information.
It delivers a transition between ice with low albedo and snow with high albedo, which is crucial in order to calculate the correct melt estimates spatially.

Trying to derive a snow line is a two-step procedure:
first, a binary snow/ice map of the glacier is derived making use of spectral information.
Then, some geometric considerations are made to obtain an estimate of the mean snow line altitude.
Since for CRAMPON we are interested in both the uncertainty of the binary snow maps and the snow line altitudes, we apply three algorithms to calculate these two stages.

Snow Line Retrieval Algorithms
------------------------------

1. The ASMAG algorithm
""""""""""""""""""""""

The "Automated Snow Mapping on Glaciers" (ASMAG) algorithm by [Rastner_et_al(2019)]_ makes use of the different spectral properties of snow and ice in the near-infrared (NIR)electromagnetic spectrum.
While snow reflects a lot in the NIR spectral range, ice is a bad reflector.
Since there is thus ideally a distinct reflectance transition between these two surface types, which is just a bit different from glacier to glacier, the Otsu threshold ([Otsu_()] is used to divide the NIR band histogram of the glacier into two subregions:


2. The Naegeli algorithm
""""""""""""""""""""""""

3. An alternate version of the Naegeli algorithm
""""""""""""""""""""""""""""""""""""""""""""""""

4. Our custom composite
"""""""""""""""""""""""


Validation
----------

put here:

1. figures that illustrate how the algorithms work
2. calidation figures
3. a GIF of the snow line during one summer season.

References
----------
.. [Rastner_et_al(2019)]
.. [Naegeli_et_al(2019)]
.. [Geibel_et_al(2019)]
.. [Otsu]