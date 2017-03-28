""" Prepare the meteodata from netCDF4 """

from __future__ import division

import netCDF4 as nc
import glob
import os
import salem

# To (re)write:
# mb_climate_on_height, _get_ref_glaciers, _get_optimal_scaling_factor

# Parameters to be kicked out later on
testprec_dir = os.path.join(os.getcwd(), 'data\\test\\prec')
testtemp_dir = os.path.join(os.getcwd(), 'data\\test\\temp')
shape_dir = os.path.join(os.getcwd(), 'data\\test\\shp')
aletsch_shp = shape_dir + '\\Aletsch\\G008032E46504N.shp'

testtemp = glob.glob(testtemp_dir+'\\*.nc')[0]
testprec = glob.glob(testtemp_dir+'\\*.nc')[0]


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


if __name__ == '__main__':
    print(glob.glob(testtemp_dir+'*TabsD*'))
    temp = MeteoSuisseGrid(testtemp)
    print(temp._data)
