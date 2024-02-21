"""
Created on Wdnesday October 20, 2021
@author: Aaron Cremona
Accounting for debris cover on glaciers
"""

# Built ins
import logging
# External libs
import numpy as np
import rasterio
import os
import xarray as xr
import pandas as pd
import netCDF4
from scipy.interpolate import interp1d
from scipy import optimize as optimization
# Locals
import oggm
import oggm.cfg as cfg
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.utils import (SuperclassMeta, lazy_property, floatyear_to_date,
                        date_to_floatyear, monthly_timeseries, ncDataset,
                        tolist, clip_min, clip_max, clip_array)
from oggm import utils                       
from oggm.exceptions import InvalidWorkflowError, InvalidParamsError
from oggm import entity_task
from oggm.core.gis import rasterio_to_gdir
from oggm.utils import ncDataset

# Module logger
log = logging.getLogger(__name__)


# Add debris data to their Glacier directories, same as in PyGEM (David Rounce), 
debris_fp = '/scratch-fourth/acremona/crampon/data/debris/' # this needs to be changed if working on another computer
#debris_fp = cfg.PATHS['debris_fp']

# Add the new name "hd" to the list of things that the GlacierDirectory understands
if not 'debris_hd' in cfg.BASENAMES:
    cfg.BASENAMES['debris_hd'] = ('debris_hd.tif', 'Raster of debris thickness data')
if not 'debris_ed' in cfg.BASENAMES:
    cfg.BASENAMES['debris_ed'] = ('debris_ed.tif', 'Raster of debris correction factor data')

@entity_task(log, writes=['debris_hd', 'debris_ed'])
def debris_to_gdir(gdir, debris_dir=debris_fp, add_to_gridded=True, hd_max=5, hd_min=0, ed_max=10, ed_min=0):
    """Reproject the debris thickness and correction factor files to the given glacier directory
    
    Variables are exported as new files in the glacier directory.
    Reprojecting debris data from one map proj to another is done. 
    We use bilinear interpolation to reproject the velocities to the local glacier map.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
    assert os.path.exists(debris_dir), "Error: debris directory does not exist."

    #hd_dir = debris_dir + 'hd_tifs/' + gdir.rgi_region + '/'
    #ed_dir = debris_dir + 'Ed_tifs/' + gdir.rgi_region + '/'
    
    #glac_str_nolead = str(int(gdir.rgi_region)) + '.' + gdir.rgi_id.split('-')[1].split('.')[1]
    
    # If debris thickness data exists, then write to glacier directory
    #if os.path.exists(hd_dir + glac_str_nolead + '_hdts_m.tif'):
    #    hd_fn = hd_dir + glac_str_nolead + '_hdts_m.tif'
    #elif os.path.exists(hd_dir + glac_str_nolead + '_hdts_m_extrap.tif'):
    #    hd_fn = hd_dir + glac_str_nolead + '_hdts_m_extrap.tif'
    
    if os.path.exists(debris_dir + 'SGI_2016_debriscover_raster_byte.tif'):
        hd_fn = debris_dir + 'SGI_2016_debriscover_raster_byte.tif'
    else: 
        hd_fn = None
        
    if hd_fn is not None:
        rasterio_to_gdir(gdir, hd_fn, 'debris_hd', resampling='bilinear')
    if add_to_gridded and hd_fn is not None:
        output_fn = gdir.get_filepath('debris_hd')
        
        # append the debris data to the gridded dataset
        with rasterio.open(output_fn) as src:
            grids_file = gdir.get_filepath('gridded_data')
            with ncDataset(grids_file, 'a') as nc:
                # Mask values
                glacier_mask = nc['glacier_mask'][:] 
                data = src.read(1) * glacier_mask
                data[data>hd_max] = 0
                data[data<hd_min] = 0
                
                # Write data
                vn = 'debris_hd'
                if vn in nc.variables:
                    v = nc.variables[vn]
                else:
                    v = nc.createVariable(vn, 'f8', ('y', 'x', ), zlib=True)
                v.units = 'm'
                v.long_name = 'Debris thicknness'
                v[:] = data

                
@entity_task(log, writes=['inversion_flowlines'])
def compute_debris_ratio(gdir):
    """Calculate the debris ratio, i.e. for every elevation band (of the flowlines node)
    the fraction of area covered by them.
    
    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """
            
    fls = gdir.read_pickle('inversion_flowlines')
    cis = gdir.read_pickle('geometries')['catchment_indices']
    
    grids_file = gdir.get_filepath('gridded_data')
    with utils.ncDataset(grids_file) as nc:
        topo = nc.variables['topo'][:]
        debris_hd = nc.variables['debris_hd'][:]

    # clip on catchments
    for j, ci in enumerate(cis[::-1]):
        with utils.ncDataset(grids_file) as nc:
            catch_topo = nc.variables['glacier_mask'][:] * np.NaN
            catch_debris = nc.variables['glacier_mask'][:] * np.NaN
        catch_topo[tuple(ci.T)] = topo[tuple(ci.T)]
        catch_debris[tuple(ci.T)] = debris_hd[tuple(ci.T)]
        
        # define bins for elevation bands
        bins = []
        bins.append(np.nanmax(catch_topo))
        for i in range(len(fls[::-1][j]._surface_h) - 1):
            bins.append((fls[::-1][j]._surface_h[i]+fls[::-1][j]._surface_h[i+1])/2)
        bins.append(np.nanmin(catch_topo))
        
        # cut the bands
        topo_digi = np.digitize(catch_topo, bins) - 1  # I prefer the left
        debris_ratio = []
        for bi in range(len(bins) - 1):
            # The coordinates of the current bin
            bin_coords = topo_digi == bi
            #calculate debris ratio for every band
            debris_ratio.append(np.nansum(catch_debris[bin_coords])/len(catch_debris[bin_coords])) 
            
        debris_ratio = np.array(debris_ratio)
        
        # create new attribute debris_ratio to centerline object fls
        fls[::-1][j].debris_ratio = debris_ratio
        
    # write to pickle
    gdir.write_pickle(fls, 'inversion_flowlines')
