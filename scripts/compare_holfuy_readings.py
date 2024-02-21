import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from crampon import cfg
import os

readings_path = "C:\\Users\\Johannes\\Documents\\holfuyretriever"
readings_comp_path = "C:\\Users\\Johannes\\Documents\\publications\\Paper_" + \
                     "Cameras\\camera_reading_comparison"


matthias = pd.read_csv(
    os.path.join(readings_comp_path, 'Camera_Readings_Matthias',
                 'melt_fi700.csv'),
    sep=';', parse_dates=[0], index_col=0, dayfirst=True)
chris = pd.read_csv(
    os.path.join(readings_comp_path,
                 '1001_MB_reading_christophe_ogier_edited.csv'),
    sep=';', index_col=0, parse_dates=[0], dayfirst=True)
johannes = pd.read_csv(os.path.join(readings_path, 'manual_reading_1001.csv'),
                       index_col=0, parse_dates=[0])
elias = pd.read_csv(os.path.join(readings_comp_path,
                                 'clicks_n_beers_elias_20200515_edited.csv'),
                    index_col=0, parse_dates=[0], dayfirst=True)
jane = pd.read_csv(
    os.path.join(readings_comp_path,
                 'template_clicks_n_beers_jane_edited.csv'),
    index_col=0, parse_dates=[0])
michi = pd.read_csv(
    os.path.join(readings_comp_path, 'clicks_n_beers_michi_edited.csv'),
    index_col=0, parse_dates=[0], dayfirst=True)


# make cumsum the actual values
elias = elias[~np.isnan(elias.dh.values)]
michi = michi[~np.isnan(michi.dh.values)]
elias['dh'] = elias.dh.values - np.hstack((np.array([0.]),
                                           elias.dh.values[:-1]))
michi['dh'] = michi.dh.values - np.hstack((np.array([np.nan]),
                                           michi.dh.values[:-1]))

rho_snow_guess = cfg.PARAMS['autumn_snow_density_guess']

# convert to SWE
chris['swe'] = np.nan
chris.loc[chris.phase == 'i', 'swe'] = chris['dh'][chris.phase == 'i'] * \
                                       cfg.RHO / 1000.
chris.loc[chris.phase == 's', 'swe'] = chris['dh'][chris.phase == 's'] * \
                                       rho_snow_guess / 1000.
johannes['swe'] = np.nan
johannes.loc[johannes.phase == 'i', 'swe'] = \
    johannes['dh'][johannes.phase == 'i'] * cfg.RHO / 1000.
johannes.loc[johannes.phase == 's', 'swe'] = \
    johannes['dh'][johannes.phase == 's'] * rho_snow_guess / 1000.

elias['swe'] = np.nan
elias.loc[elias.phase == 'i', 'swe'] = \
    elias['dh'][elias.phase == 'i'] * cfg.RHO / 1000.
elias.loc[elias.phase == 's', 'swe'] = \
    elias['dh'][elias.phase == 's'] * rho_snow_guess / 1000.

jane['swe'] = np.nan
jane.loc[jane.phase == 'i', 'swe'] = \
    jane['dh'][jane.phase == 'i'] * cfg.RHO / 1000.
jane.loc[jane.phase == 's', 'swe'] = \
    jane['dh'][jane.phase == 's'] * rho_snow_guess / 1000.

michi['swe'] = np.nan
michi.loc[michi.phase == 'i', 'swe'] = \
    michi['dh'][michi.phase == 'i'] * cfg.RHO / 1000.
michi.loc[michi.phase == 's', 'swe'] = \
    michi['dh'][michi.phase == 's'] * rho_snow_guess / 1000.


fig, ax = plt.subplots()
(chris.swe/100)[chris.swe <= 0.].plot(ax=ax, label='Chris')
(-matthias.swe/100.)[matthias.swe >= 0.].plot(ax=ax, label='Matthias')
(-johannes.swe).plot(ax=ax, label='Johannes')
(-elias.swe/100.).plot(ax=ax, label='Elias')
(-jane.swe/100.).plot(ax=ax, label='Jane')
(-michi.swe).plot(ax=ax, label='Michi')
plt.ylabel('MB (m w.e.)')
plt.legend()

meanlist = []
result_array = np.full((6, 6), np.nan)
for x, y in itertools.combinations([('Chris', chris.swe/100., 0),
                                   ('Johannes', -johannes.swe, 1),
                                   ('Matthias', -matthias.swe/100., 2),
                                   ('Jane', -jane.swe/100., 3),
                                   ('Michi', -michi.swe, 4),
                                   ('Elias', -elias.swe/100., 5)], 2):
    mad = np.nanmean(np.abs((-x[1])-(-y[1])))
    meanlist.append([x[0], y[0], mad])
    result_array[x[2], y[2]] = mad
print(meanlist)
print('MEAN of MEAN ABSOLUTE DEVIATIONS: ', np.mean([i[2] for i in meanlist]))

plt.figure()
plt.boxplot([i[2] for i in meanlist])

plt.figure()
plt.imshow(result_array, cmap='Reds')
plt.colorbar()
