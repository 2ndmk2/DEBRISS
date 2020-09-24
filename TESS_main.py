import numpy as np
import matplotlib.pyplot as plt
import tess_lc
import importlib
importlib.reload(tess_lc)

name_arr, ra_arr, dec_arr, spt_arr, T_arr, ldisk = tess_lc.catalog_load_and_save("./debris/table.dat")
table = np.load("catalog.npz")
tess_lc.main(name_arr, ra_arr, dec_arr, until_i =10000, size =70 , start_star = "", replace=True, pca_comp = 3, ap_wd=5)



