Input Uncertainty
=================

All mass balance models in CRAMPON are driven by meteorological inputs.
These inputs are often interpolated from measurements at ground stations, such as in the case of temperature and precipitation.
This comes with a lot of problems:
one of these problems is the systematic error, when using the interpolated grid values at the glacier scale.
The problems though is that it is very hard to quantify this error, since to get an estimate of its magnitude we would need to be able to measure e.g. temperature on a whole grid cell.
We only account of the systematic errors indirectly by tuning the mass balance model parameters.
Among other things, the systematic error is something whe try to "tune away", for example by introducing the precipitation correction factor.


So in CRAMPON we only care about random errors.
MeteoSwiss is generally good at giving random errors estimates, even though some values used in CRAMPON are read by hand from scientific publications.
Here is an overview of the error magnitudes we assume for different parameters, as well as an estimate of how the random errors could be distributed.


========= ============== ============ ======
Parameter Error Estimate Distribution Source
--------- -------------- ------------ ------
Temperature
Precipitation
Shortwave Irradiation
Potential Irradiation
========= ============== ============

The errors for temperature depend on the season when they are measured.
Here we assume a trapezoidal temperature error function over the course of the year.
For precipitation, the errors does not only depend on the seasons, but also on the quantile of the precipitation qith respect to the overall precipitation distribution.
This is because the precipitation interpolation algorithm is likely to underestimate high precipitation amounts, while at the same time it has a strong tendency not to extrapolate precipitation to zero.
Shortwave irradiation

Mention further:

1. we only pull a random number from the errors distribution once per glacier, since we assume high error correlation on the glacier scale (the measurement network is much sparser than the glacier scale).
2. strictly speaking, this is not true for radiation, since it is derived from a satellite
3. the errors for potential irradiation is just a guess, since doing experiments ont his is expensive and out of scope.
4. tell early in the text that radiation is not interpolated from station measurements, but derived from satellite data


References
==========

.. [Frei_(2014]
.. [Isotta_et_al(2019)]
.. [Stoeckli_(2013)]
