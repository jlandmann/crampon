.. _potential-irradiation:

Potential Irradiation
=====================

Potential irradiation is an important input to the melt model presented in
[Hock (1999)]_.
Here, we calculate the daily mean potential irradiation :math:`I_{pot}`
assigned to glacier flowline nodes in a three step procedure:

1. Calculate a mask with cast shadows on the terrain

2. Calculate potential direct-beam irradiation based on terrain parameters (no shadows included)

3. Combine both together to obtain potential irradiation and zero where there are shadows

4. Distribute the potential irradiation on flowlines.

The first three steps are performed on 2D grids to account for spatial variability of incoming solar radiation. Particularly, the cast shadow
calculation is calculated on a 10km extended grid around the glacier
outlines. This is to account for long-range shadows, which are particularly
important at times around dusk and dawn. In order to obtain daily means from single snapshots of irradiation, we calculate :math:`I_{pot}` at 10 minute intervals, sum within 24 hours, and then divide by the number of time steps per day (here:144).

GIS basics
----------
say here:

1. if available, we take the SwissAlti3D
2. if glacier is at the border, we cheaply debias SRTM and mosaic the two DEMs
3. we resample the obtained grid to the resolution of the glacier grid in order to match the grid resolution of :math:`I_{pot}` with that of the glacier.


Potential Direct-Beam Irradiation
---------------------------------
The potential irradiation in a terrain is not only base on cast shadows, but also on topographics parameters like slope and aspect.
The latter modify the radiation flux on a horizontal surface.

[Hock (1999)]_ suggests a way to calculate terrain effects in a single equation for :math:`I_{pot, terrain}` in :math:`(W\,m^{-2})`:

.. math::
    I_{pot, terrain} = I_0 {\left(\frac{R_m}{R}\right)}^2 \cdot
\psi_a^{\left(\frac{P}{P_0\ \cos Z_s}\right)} \cos \theta

where :math:`I_0` is the solar constant (1367 :math:`(W\,m^{-2})`), :math:`{\left(\frac{R_m}{R})\right)}^2` is a correction factor to account for the elliptic shape of the Earth's orbit around the sun with :math:`R` being the actual distance between sun and earth on a particular day of year and :math:`R_m` being the mean sun-earth distance, :math:`\psi_a` being the clear-sky atmospheric transmissivity, :math:`P` being the atmospheric pressure (Pa), :math:`P_0` being the mean sea level pressure (1013.25 hPa), :math:`Z_s` is the sun zenith angle and :math:`\theta` is the so called `incidence angle`.
A lot of the values in this equation are approximations of real values.
Like this, the ratio of :math:`R_m` to :math:`R` is calculated using the approach by [Iqbal (1983)]_:

.. math::
   \left(\frac{R_m}{R}\right) = 1 + 0.033 \cos((\frac{2.\ \pi\ DOY}{365.})

where :math:`DOY` is the day of year.
:math:`\psi_a` is actually varying in space and time, but since we do not have accurate data at the scales we need them, we follow [Hock (1999)]_ and set :math:`\psi_a` to a constant value of 0.75 as a compromise (see [Hock (1998)]_ and [Oke (1987)]_ for references).
Atmospheric pressure is also variable in space an time, but for our application it can be approximated reasonably well with the barometric formula:

.. math::
    P(z) = P_0  (1 - \left(\frac{\frac{\partial T}{\partial z} z}{T_{standard}}\right)^{\left(\frac{g  M}{R  \frac{\partial T}{\partial z }}\right)}

with :math`\frac{\partial T}{\partial z}` being the atmospheric temperature
lapse rate, prescribed to be 0.0065 :math:`K m^{-1}`, :math:`z` being the grid cell evelation (m), :math:`T_{standard}` being the standard temperature (288.15K), g being the gravitational acceleration (9.81 :math:`kg m^{-2}`), M the molar mass of dry air (0.02896968 :math:`kg mol^{-1}`) and R the universal gas constant (8.314 :math:`J K^{-1} mol^{-1}`).
Last but not least, the incidence angle is a construct to approximate the angle between a normal to a grid cell slope and a solar beam directed toward the cell center.
Like in [Hock (1999)]_, we use the formula suggested by [Garnier and Ohmura (1968)]_ to calculate :math:`\theta`:

.. math::
    \theta = \arccos \left(\cos \beta_g \cos Z_s + \sin \beta_g \sin Z_s \cos
(A_s - A_g)\right)

where :math:`\beta_g` is the grid slope, :math:`A_s` is the solar azimuth angle, :math:`A_g` is the terrain azimuth, which is also called `aspect`.
In this implementation, all azimuths are zero at true north and increases clockwise.

Just the implementation of this equation already gives a realistic impression of incident solar radiation.

.. figure:: _static/xyz.jpg
        :width: 80%

        An example of potential irradiation in the Rhonegletscher area on DATE & TIME, calculated after [Hock (1999)]_. The underlying DEM is SwissAlti3D, here.

This is an example for Rhonegletscher, TIME AND DATE.
However, a major flaw is that it does not account for cast shadows.
According to definition of direct-beam potential solar irradiation, the :math:`I_{pot}` should be zero in shadows.


Cast Shadows
------------

A cast shadows mask is generated using the algorithm proposed by
[Corripio_2003]_. This algorithm calculates the so called `sun vector`, a unit vector normal to the surface pointing towards the sun:

.. math::
    x_s = \begin{bmatrix} x \\ y \\ z \end{bmatrix}


The final result of the computation is a shadow mask `m_s`:

This shadow mask has zero entries where there is shadow and one entries where a direct sun beam can reach the surface.


Combine potential irradiation and cast shadows
----------------------------------------------

We obtain the actual potential irradiation by multiplying the shadow array and the potential terrain parameter based irradiation:

.. math::
    I_{pot} = I_{pot,terrain} \cdot m_s


Distribute :math:`I_{pot}` on flowlines
---------------------------------------

In a similar way the glacier grid is transformed into flowlines with weighted nodes, we also distribute the potential irradiation onto the glacier grid nodes.
An iterator loops through all glacier flowlines and maps their respective area on the :math:`I_{pot}` grid.
Then, the mean over all pixels that belong into the elevation of a flow line node is assigned to the respective node.



References
----------
.. [Corripio_2003] Corripio, J. G.: Vectorial algebra algorithms for
    calculating terrain parameters from DEMs and solar radiation modelling in
    mountainous terrain. International Journal of Geographical Information
    Science, Taylor & Francis, 2003, 17, 1-23.
.. [Garnier & Omura (1968)] Garnier, B. J., & Ohmura, A. (1968). A method of
    calculating the direct shortwave radiation income of slopes. Journal of
    Applied Meteorology, 7(5), 796-800.
.. [Hock (1998)] Hock, R. (1998). Modelling of glacier melt and discharge     (Doctoral dissertation, ETH Zurich).
.. [Hock (1999)] Hock, R. (1999). A distributed temperature-index ice-and
    snowmelt model including potential direct solar radiation. Journal of
    Glaciology, 45(149), 101-111.
.. [Iqbal (1983)] Iqbal, M.: An introduction to solar radiation, Academic       Press, New York, 1983.
.. [Oke (1987)] Oke, T. R. (1987). Boundary layer climates. Routledge.