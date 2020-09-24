from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from astropy.timeseries import LombScargle
import astropy.units as u
import tess_lc

file_names = ["SY_cha/tess-s0011-3-4_164.123040_-77.196550_50x50_astrocut.fits","SY_cha/tess-s0012-3-3_164.123040_-77.196550_50x50_astrocut.fits"]
lc_svd_now, lc_ori = tess.return_lcs_fromfiles(file_names)
power, frequency, period = tess.ana_sum(lc_svd_now/np.median(lc_svd_now))
power2, frequency2, period  =tess.ana_sum( lc_ori/np.median( lc_ori), save= False)
plt.plot(1/frequency, power, color ="r")
plt.plot(1/frequency2, power2)
plt.xlim(0,30)




