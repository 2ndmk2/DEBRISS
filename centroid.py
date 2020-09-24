from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from astropy.timeseries import LombScargle
import astropy.units as u
from astroquery.mast import Tesscut
from astropy.coordinates import SkyCoord
import os

def centroid_move_calc(target_pixel, r_aperture = None):
                                    
        ## Preparation
        t_dim, x_dim, y_dim = np.shape(target_pixel)
        x_arr = np.arange(x_dim)
        y_arr = np.arange(y_dim)
        x_arr = x_arr - np.mean(x_arr)
        y_arr = y_arr - np.mean(y_arr)
        tp_non_zero = np.copy(target_pixel)
        tp_non_zero[tp_non_zero<0] = 0
        Y,X = np.indices(target_pixel[0].shape)
        
        flag_aperture = np.ones( (t_dim, x_dim, y_dim))
        x_cen =  (x_dim-1)/2.0 
        y_cen =  (y_dim-1)/2.0
        r = np.sqrt((Y-y_cen)**2 + (X-x_cen)**2)

        if r_aperture is not None:
            flag_aperture[:,r>r_aperture] = 0
    
        aa, bb = np.meshgrid(x_arr,y_arr)
        
        target_pixel_time = np.einsum("ijk->i", tp_non_zero*flag_aperture)        
        
        ## Photocenter
        x_centroid = np.einsum("ijk, jk->i", tp_non_zero*flag_aperture, aa)/(target_pixel_time)
        y_centroid = np.einsum("ijk, jk->i", tp_non_zero*flag_aperture, bb)/(target_pixel_time)
        x_centroid[target_pixel_time==0] = 0
        y_centroid[target_pixel_time==0] = 0

        return x_centroid, y_centroid

def centroid_tpfs(tpfs, file_name,  r_aperture=None):

    for (i, tpf) in enumerate(tpfs):
        x_cen, y_cen = centroid_move_calc(tpf, r_aperture = r_aperture)
        plt.plot(x_cen)
        plt.plot(y_cen)
        plt.savefig(file_name +"_%d.png" % i)
        plt.close()

