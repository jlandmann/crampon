import salem
import glob
import xarray as xr
import numpy as np
import itertools
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import stats
import os
import pandas as pd

base_path = 'C:\\Users\\Johannes\\Documents\\mauro_DEMs_test\\' \
            'test_differencing\\PLM'

all_dems = glob.glob(base_path + '\\*_adjOn_swissALTI3D_full.tif')
# all_dems = glob.glob(base_path + '\\*.tif')

# dem_1 = xr.open_rasterio(
#    base_path + '\\A55F_03_20100809_zShift_adjOn_swissALTI3D_full.tif')
# dem_2 = xr.open_rasterio(
#    base_path + '\\A55F_03_20130903_zShift_adjOn_swissALTI3D_full.tif')

outline = 'C:\\Users\\Johannes\\Documents\\modelruns\\CH\\per_glacier\\' \
          'RGI50-11\\RGI50-11.A5\\RGI50-11.A55F03\\outlines.shp'
# outline = 'C:\\Users\Johannes\Documents\modelruns\CH\per_glacier\RGI50-11' \
#          '\RGI50-11.A5\RGI50-11.A55F03\outlines.shp'
# outline = 'C:\\Users\Johannes\Documents\modelruns\CH\per_glacier\RGI50-11' \
#          '\RGI50-11.A1\RGI50-11.A10G05\outlines.shp'
gname = 'PLM'

ct = 0
for d1_path, d2_path in itertools.combinations(all_dems, 2):
    d1_date = pd.Timestamp(os.path.basename(d1_path).split('_')[2])
    d2_date = pd.Timestamp(os.path.basename(d2_path).split('_')[2])

    if d1_date >= d2_date:
        continue
    else:
        print(d1_date, d2_date)

    dem_1 = xr.open_rasterio(d1_path)
    dem_2 = xr.open_rasterio(d2_path)

    # rename some variables for salem
    dem_1.attrs['pyproj_srs'] = dem_1.crs
    dem_2.attrs['pyproj_srs'] = dem_2.crs

    # replace 9999.0 with np.nan
    dem_1 = dem_1.where(dem_1 != dem_1.attrs['nodatavals'][0])
    dem_1.name = 'height'
    dem_2 = dem_2.where(dem_2 != dem_2.attrs['nodatavals'][0])
    dem_2.name = 'height'

    # clip DEMs with outline
    ol = gpd.read_file(outline)
    dem_1_clipped = dem_1.salem.roi(geometry=ol.geometry.iloc[0], crs=ol.crs,
                                    all_touched=False)
    dem_2_clipped = dem_2.salem.roi(geometry=ol.geometry.iloc[0], crs=ol.crs,
                                    all_touched=False)
    dem_2_clipped = dem_1_clipped.salem.transform(dem_2_clipped)
    dem_1 = None
    dem_2 = None

    # subtract DEMs
    diff = dem_2_clipped - dem_1_clipped

    # check covered percentage
    print(diff.values.shape)

    # add the CRS
    diff.attrs['pyproj_srs'] = dem_2_clipped.crs
    mask = diff.salem.grid.region_of_interest(outline)
    ratio_covered = np.count_nonzero(~np.isnan((diff.values[0, ...]*mask))) / \
        np.count_nonzero(mask)

    if ratio_covered < 0.75:
        print('Ratio covered too small: {:.2f}'.format(ratio_covered))
        continue
    else:
        print('Ratio covered: {:.2f}'.format(ratio_covered))

    # mean dh on existing cells
    print('Mean dh = {} m'.format(np.mean(diff)))

    # fill missing values with values from regression
    mean_height = (dem_2_clipped + dem_1_clipped) / 2.
    regr = stats.linregress(mean_height.values[~np.isnan(mean_height.values)],
                            diff.values[~np.isnan(diff.values)])
    mean_height_interp_x = mean_height.interpolate_na(dim='x')
    diff_filled = diff.isel(band=0).values.copy()
    diff_filled[np.isnan(diff.values[0, ...]) is True] = (
                mean_height_interp_x.values[0, ...][
                    np.isnan(diff.values[0, ...]) is True] * regr[0] + regr[1])
    diff_filled[np.where(mask == 0.)] = np.nan  # excl. interp at concave edges
    diff_clipped = diff.copy()
    diff_clipped.loc[1] = diff_filled
    diff_clipped = diff_clipped.salem.subset(shape=outline, margin=20)
    diff_clipped = diff_clipped.isel(band=0)

    # calculate volume change
    dv = np.nansum(diff_filled) * np.abs(diff.y[1] - diff.y[0]) * \
        np.abs(diff.x[1] - diff.x[0])
    print(d1_path, d2_path)
    print('DV = {} m3'.format(dv))

    fig, ax = plt.subplots()
    try:
        diff_clipped.plot(ax=ax, center=0., cmap='RdBu', robust=True,
                          cbar_kwargs={'label': 'Elevation difference (m)'})
    except:
        print('stop')
    ax.set_title('{} - {}: {} m3'.format(d2_date.isoformat()[:10],
                                         d1_date.isoformat()[:10],
                                         int(dv.item())))
    plt.savefig(os.path.dirname(d1_path) + '\\' + gname + '_' +
                d2_date.strftime('%Y-%m-%d') + '_' +
                d1_date.strftime('%Y-%m-%d') + '.png')
    plt.close()
plt.show()
