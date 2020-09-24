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
import centroid
def load_tpf(file_name):
    tpf = []
    hdul = fits.open(file_name)
    header = hdul[1].header
    image = hdul[1].data
    image = image
    for i in range(len(image)):
        tpf.append(image[i][4])
    hdul.close()
    tpf = np.array(tpf)
    return tpf

def flag_bkg_momentum_dumps(bkg, factor = 0.5, factor_lw = 0.5):

    flag_finite = np.isfinite(bkg)
    median_80 = np.percentile(bkg[flag_finite], q=80 )
    median_20 = np.percentile(bkg[flag_finite],  q=20 ) 
    flag_for_med = (bkg < median_80) * (bkg > median_20) 
    print(median_80, median_20)
    bkg_median = np.median(bkg[flag_for_med])
    print(bkg_median)
    return (bkg < bkg_median * (1 + factor) )* (bkg > bkg_median * (1 - factor_lw) )



def tpf_bkg_sub(tpf):
    bkg_sum = 0
    nt, nx ,ny = np.shape(tpf)
    median_50 = np.percentile(tpf, axis=(1,2), q=50 )
    median_16 = np.percentile(tpf, axis=(1,2), q=16 )
    flags = []
    bkg_arr = []
    for i in range(nt):
        light_all = np.ravel(tpf[i])
        flag = light_all < median_50[i] + 3* (median_50[i] - median_16[i] )
        bkg_med = np.median(light_all[flag])
        if len(light_all[flag]) ==0:
            flags.append(False)
        else:
            flags.append(True)
        tpf[i] = tpf[i] - bkg_med
        bkg_sum += bkg_med
        if not np.isfinite(bkg_med):
            bkg_arr.append(0)
        else:
            bkg_arr.append(bkg_med)
    bkg_arr = np.array(bkg_arr)
    flags_bkg = flag_bkg_momentum_dumps(bkg_arr)
    flags = np.array(flags) * flags_bkg
    return tpf, bkg_sum, np.array(flags), bkg_arr 

def index_make(arr):
    nx, ny = np.shape(arr)
    index_arr_i = np.zeros((nx, ny))
    index_arr_j = np.zeros((nx, ny))
    for i in range(nx):
        for j in range(ny):
            index_arr_i[i][j] = i
            index_arr_j[i][j] = j
    return index_arr_i, index_arr_j

def source_pos(pos_peaks_pre, edge_wd = 3, dist_thr =3):
    nx, ny = np.shape(pos_peaks_pre)
    index_i, index_j = index_make(pos_peaks_pre)
    index_i_pos = index_i[pos_peaks_pre]
    index_j_pos = index_j[pos_peaks_pre]
    pos_stars = []
    for i in range(len(index_i_pos)):
        i_pos, j_pos = index_i_pos[i], index_j_pos[i]
        if i_pos > edge_wd and i_pos < nx - edge_wd and j_pos > edge_wd and j_pos <ny- edge_wd :
            pos_stars.append([i_pos, j_pos])
    pos_stars = np.array(pos_stars)
    
    print("Num of stars:", len(pos_stars))
    if len(pos_stars)==0:
        return []
    flag = []
    for i in range(len(pos_stars)):
        flag_now = 1
        for j in range(len(pos_stars)):
            if i!=j:
                dist = np.sqrt((pos_stars[i][0] - pos_stars[j][0])**2 + (pos_stars[i][1] - pos_stars[j][1])**2 )
                if dist < dist_thr:
                    flag_now = 0
                
        flag.append(flag_now==1)
    flag = np.array(flag)
    pos_stars = pos_stars[flag]    
    return pos_stars

def lc_make(movie, pos_stars, ap_wd=3):
    
    lcs = []
    for i in range(len(pos_stars)):
        x_pos, y_pos = int(pos_stars[i][0]), int(pos_stars[i][1])
        tpf_sum = np.sum(movie[:,x_pos-ap_wd:x_pos+ap_wd, y_pos-ap_wd:y_pos+ap_wd], axis = (1,2))
        tpf_sum = sigma_clipping(tpf_sum, split=40, wd = 5)
        lcs.append(tpf_sum)

    return np.array(lcs)

def extract_tpf(movie, x_cen, y_cen, ap_wd=3):
    
    movie = movie[:,x_cen-ap_wd:x_cen+ap_wd, y_cen-ap_wd:y_cen+ap_wd]

    return np.array(movie)

## PCA
def x_mean(x):
    n, num_band = np.shape(x)
    x_mean_arr = np.zeros(np.shape(x))
    for i in range(n):
        x_mean_arr[i] = np.mean(x[i])
    return x_mean_arr

def flux_sub_mean_func(x, axis = None):
    xmean = x.mean(axis=axis, keepdims=True)
    zscore = (x-xmean)
    return zscore

def flux_std(x):
    n, num_band = np.shape(x)
    x_std_arr = np.zeros(n)
    for i in range(n):
        x_std_arr[i] = np.std(x[i])
    return x_std_arr      

def svd_fit_and_subtract(flux_arr, use_svd_flag, ncomp=3):
    
    n1, p1 = np.shape(flux_arr)
    flux_mean_arr = x_mean(flux_arr)
    flux_sub_mean = np.zeros(np.shape(flux_arr))
    
    
    for i in range(n1):
        flux_sub_mean[i] = flux_sub_mean_func(flux_arr[i])
    
    
    flux_arr_for_pca = flux_arr[use_svd_flag]
    n_pca, p_pca = np.shape(flux_arr_for_pca)
    std_flux_pca = flux_std( flux_arr_for_pca)
    flux_arr_for_pca_sub_mean = np.zeros(np.shape(flux_arr_for_pca))
    
    for i in range(n_pca):
        flux_arr_for_pca_sub_mean[i] = flux_sub_mean_func(flux_arr_for_pca[i]) / std_flux_pca[i]
    

    ## SVD of the data
    U, s ,V_svd = np.linalg.svd(flux_arr_for_pca_sub_mean, full_matrices = True)
    max_value = max(n_pca, p_pca)
    min_value = min(n_pca, p_pca)
    S = np.zeros((max_value , max_value )) ###
    S[:min_value, :min_value] = np.diag(s) ### diag of singular values
    sigma_arr = s*s/p_pca ### significance of each component
    
    if 2 * ncomp > n_pca:
        ncomp = int(n_pca/2)

    V_sub_from_lc = V_svd[:ncomp] ## Component for subtraction 
    coeff_V = np.dot(flux_sub_mean,V_sub_from_lc.T) ## Coefficient for subtraction 
    sub_component = np.dot(coeff_V, V_sub_from_lc) ## Lightcurve for subtraction 
    after_subtracted = flux_sub_mean - sub_component + flux_mean_arr ## Output after subtraction of systematics
    return list(after_subtracted), list(V_sub_from_lc), list(sigma_arr), V_svd
    
def cpos_peaenter_pos_arg(ks, nx, ny):
    cen_x, cen_y = ((nx-1)/2.0), ((ny-1)/2.0)
    for i in range(len(pos_peaks)):
        x, y = pos_peaks[i][0], pos_peaks[i][1] 
        if cen_x+1  >= x and cen_x-1 <= x and cen_y+1  >= y and cen_y-1 <= y:
            return i
    return 0

def flag_cen_wd(pos_stars, nx, ny, wd=5):
    flag_nearby_center = []
    for i in range(len(pos_stars)):
        if pos_stars[i][0] > nx/2 -wd and pos_stars[i][0] < nx/2 +wd and pos_stars[i][1] > ny/2 -wd and pos_stars[i][1] < ny/2 +wd:
            flag_nearby_center.append(False)
        else:
            flag_nearby_center.append(True)
    return np.array(flag_nearby_center)
def center_remove(pos_peaks, wd = 10):
    nx, ny = np.shape(pos_peaks)
    x_cen = int(nx/2)
    y_cen = int(ny/2)

    for i in range(nx):
        for j in range(ny):
            if i < x_cen + wd and i >x_cen - wd and j > y_cen - wd and j < y_cen + wd:
                pos_peaks[i][j] =0
    return pos_peaks

def return_lc_from_tpf(tpf_before_bkg, threshold_source=100, pca_comp = 5, ap_wd=3):
    tpf, bkg_sum, flags, bkg_arr= tpf_bkg_sub(tpf_before_bkg)
    tpf = tpf[flags]
    image = np.sum(tpf, axis = 0)/len(tpf)
    nx, ny = np.shape(image)    
    neighborhood = generate_binary_structure(2,2) 
    local_max = maximum_filter(image, footprint=neighborhood)==image
    pos_peaks = local_max * (image>threshold_source)
    if len(pos_peaks[pos_peaks ==1]) ==0:
        return [], [], [], 0, image,  [],[], [], bkg_arr, []

    pos_peaks = center_remove(pos_peaks) ##remove center
    pos_stars= source_pos(pos_peaks, ap_wd)
    pos_stars = np.append(np.array([[int(nx/2),int(ny/2)]]), pos_stars, axis = 0) ##add center


    lcs = lc_make(tpf, pos_stars)
    svd_flag = flag_cen_wd(pos_stars, nx, ny, 5)
    if len(pos_stars) ==1:
        return [], [], [], 0, image,  pos_peaks, pos_stars, [], bkg_arr, []
    svd_flag[0]=False
    lc_after_pca, V_sub_from_lc, sigma_arr, svd_comp=svd_fit_and_subtract(lcs, svd_flag, ncomp=pca_comp)
    tpf_target = extract_tpf(tpf, int(pos_stars[0][0]), int(pos_stars[0][1]), )
    return lc_after_pca, lcs, svd_comp, 0, image, pos_peaks, pos_stars, flags, bkg_arr, tpf_target

def i_min_max(i, wd, L):
    if i<wd:
        return 0, 2*wd
    if i> L-wd-1:
        return L-1-2*wd, L
    return i-wd, i+wd

def sigma_clipping(lcs, split=10, wd = 4):
    dx = 1.0/float(split)
    split_ind = np.arange(dx, 1-dx, dx)
    indices = [int(len(lcs) * n) for n in split_ind]
    lcs_splited = np.split(lcs, indices)
    lcs_std = []
    for i in range(len(lcs_splited)):
        lcs_std.append(np.std(lcs_splited[i]))
    lcs_std = np.median(lcs_std)
    lcs_new = []
    for i in range(len(lcs)):
        i_min, i_max = i_min_max(i, wd, len(lcs))
        med_now = np.median(lcs[i_min:i_max])
        if np.abs(med_now - lcs[i]) >3 * lcs_std:
            lcs_new.append(med_now)
        else:
            lcs_new.append(lcs[i])


    return np.array(lcs_new)
def numpy_array_from_list_to_object(arr):

    i_num = len(arr)
    lcs = np.zeros(i_num, dtype = object)
    for i in range(i_num):
        lcs[i] = arr[i]
    return lcs

def return_lc_from_tpfs(tpfs, times, threshold_source=100, pca_comp = 5, ap_wd= 3):
    lcs_pca = []
    lcs_ori = []
    lcs_all_pca =[]
    lcs_all_ori =[]
    icen_arr = []
    images = []
    pos_peaks =[]
    pos_stars = []
    time_arr = []
    bkg_arrs = []
    tpf_targets = []
    flags_used = []
    svd_comps = []

    for (i, tpf) in enumerate(tpfs):
        lc_after_pca, lc_ori, svd_comp, i_cen, image, pos_peak, pos_star, flags, bkg_arr, tpf_target = return_lc_from_tpf(tpf, threshold_source=threshold_source, pca_comp = pca_comp, ap_wd=ap_wd)
        if len(pos_star) ==0:
            continue
        lcs_pca.append(lc_after_pca[i_cen])
        lcs_ori.append(lc_ori[i_cen])
        lcs_all_pca.append(lc_after_pca)
        lcs_all_ori.append(lc_ori)
        icen_arr.append(i_cen)
        images.append(image)
        pos_peaks.append(pos_peak)
        pos_stars.append(pos_star)
        time_arr.append(times[i])
        bkg_arrs.append(bkg_arr)
        tpf_targets.append(tpf_target)
        flags_used.append(flags)
        svd_comps.append(svd_comp)


    lcs_all_pca = numpy_array_from_list_to_object(lcs_all_pca)
    lcs_all_ori = numpy_array_from_list_to_object(lcs_all_ori)
    svd_comps = numpy_array_from_list_to_object(svd_comps)

    return lcs_pca, lcs_ori, lcs_all_pca, lcs_all_ori, svd_comps, \
    np.array(icen_arr), np.array(images), pos_peaks, pos_stars, time_arr,\
    bkg_arrs, np.array(tpf_targets), np.array(flags_used)



def return_lc(file_name, threshold_source=100000, pca_comp = 5):
    tpf_before_bkg = load_tpf(file_name)
    tpf, bkg_sum= tpf_bkg_sub(tpf_before_bkg)
    image = np.sum(tpf, axis = 0)
    nx, ny = np.shape(image)    
    neighborhood = generate_binary_structure(2,2)
    local_max = maximum_filter(image, footprint=neighborhood)==image
    pos_peaks = local_max * (image>threshold_source)
    pos_stars= source_pos(pos_peaks)
    lcs = lc_make(tpf, pos_stars)
    svd_flag = flag_cen_wd(pos_stars, nx, ny, 5)
    i_cen = center_pos_arg(pos_stars, nx, ny)
    lc_after_pca, V_sub_from_lc, sigma_arr=svd_fit_and_subtract(lcs, svd_flag, ncomp=pca_comp)
    return lc_after_pca, lcs, i_cen
    
def return_lcs_fromfiles(file_names):
    lcs_pca = []
    lcs_ori = []

    for file_name in file_names:
        lc_after_pca, lc_ori, i_cen = return_lc(file_name, threshold_source=100000)
        lcs_pca = np.append(lcs_pca, lc_after_pca[i_cen])
        lcs_ori = np.append(lcs_ori, lc_ori[i_cen])

    return lcs_pca, lcs_ori



def catalog_load_and_save(file_name, file_out="catalog"):
    star_num = 0
    file = open(file_name, "r")
    lines = file.readlines()
    flag = 0
    ra_arr = []
    dec_arr =[]
    name_arr = []
    spt_arr = []
    T_arr = []
    ldisk_frac_arr =[]
    for line in lines:
        if line[0:7] == "HR 9102":
            flag = 1
        if flag >0:
            name = line[0:25].rstrip()
            name2 = line[25]
            name = (name + name2).rstrip().replace(" ", "")
            ra = np.array(line[26:39].split())
            dec = line[39:51].split()
            c = SkyCoord("%sh%sm%s" % (ra[0], ra[1], ra[2]), "%sd%sm%s" % (dec[0], dec[1], dec[2]), frame='icrs')
            ra, dec = c.ra.degree , c.dec.degree 
            ra_arr.append(ra)
            dec_arr.append(dec)
            name_arr.append(name) 
            star_num+=1
            spt_arr.append(line[51:60].rstrip())
            T_arr.append(int(line[60:65].rstrip()))
            ldisk_frac_arr.append(float(line[104:110]))
        if line[0:7] == "del Scl":
            flag =0            
    np.savez(file_out, name = np.array(name_arr), ra = np.array(ra_arr), dec = np.array(dec_arr), spt = np.array(spt_arr), \
             T = np.array(T_arr), ldisk = np.array(ldisk_frac_arr))            
    return np.array(name_arr), np.array(ra_arr), np.array(dec_arr), np.array(spt_arr), np.array(T_arr), np.array(ldisk_frac_arr)

def load_tpf_hdulists(hdulist):
    tpf_arr = []
    time_arr = []
    for hdul in hdulist:
        tpf = []
        header = hdul[1].header
        image = hdul[1].data

        time_arr.append( hdul[1].data["TIME"])
        image = image
        for i in range(len(image)):
            tpf.append(image[i][4])

        hdul.close()
        tpf = np.array(tpf)
        tpf_arr.append(tpf)
    return tpf_arr, time_arr


def load_hdu(hdu):
    tpf = []
    header = hdu[1].header
    image = hdu[1].data
    time = hdu[1].data["TIME"]
    for i in range(len(image)):
        tpf.append(image[i][4])
    hdu.close()
    tpf = np.array(tpf)
    return tpf, header, time 

def data_dl(ra, dec, size):
    cutout_coord = SkyCoord(ra, dec, unit="deg")
    hdulist = Tesscut.get_cutouts(coordinates=cutout_coord, size=size )
    return hdulist

def arrs_to_1darr_med(arr):

    arr_return = []
    for i in range(len(arr)):
        arr_return = np.append(arr_return, arr[i]/np.median(arr[i]))

    return arr_return

def plot_2dimage(file_name, image, pos_star):
    plt.imshow(np.log10(image))
    plt.colorbar()
    for i in range(len(pos_star)):
        plt.scatter(pos_star[i][1], pos_star[i][0], color ="r")
    plt.savefig(file_name)
    plt.close()
    
def plot_2dimages(file_out_folder, images, pos_stars):
    
    for i in range(len(images)):
        file_name = os.path.join(file_out_folder, "image%d.png" % i)
        plot_2dimage(file_name, images[i], pos_stars[i])
def times_flagged(times, flags):
    time = []

    for i in range(len(times)):
        time = np.append(time, times[i][flags[i]])
    return time
    
def plot_lcs(lcs, times, flags, icen_arr, target_lc, target_pca, file_name):

    ymax, ymin = np.max(target_pca), np.min(target_pca)
    wd_y = (ymax - ymin) * 0.5
    time = times_flagged(times, flags)
    plt.plot(time, target_lc, color="r")
    plt.plot(time, target_pca, color="b")

    for i in range(len(lcs)):
        for j in range(len(lcs[i])):
            if j != icen_arr[i]:
                plt.plot(times[i][flags[i]], lcs[i][j]/np.median(lcs[i][j]), alpha = 0.6, lw = .11, color="k")
    plt.ylim(ymin - wd_y, ymax + wd_y)
    plt.savefig(file_name+"all.png", bbox_inches="tight")
    plt.close()

    plt.plot(time, target_lc, color="r")
    plt.plot(time, target_pca, color="b")
    plt.ylim(ymin - wd_y, ymax + wd_y)
    plt.savefig(file_name+"target.png", bbox_inches="tight")
    plt.close()    

def plot_bkg(times, bkgs, flags, file_name):
    for i in range(len(bkgs)):
        plt.plot(times[i][flags[i]], bkgs[i][flags[i]], color ="k")
        plt.plot(times[i][(flags[i]==0)], bkgs[i][(flags[i]==0)], color = "r")


    plt.savefig(file_name+"bkg.png", bbox_inches="tight")
    plt.close()  


def ana_lomb(time, lcs, save = True, name ="sy_cha"):

    frequency, power = LombScargle(time * u.day, lcs).autopower()
    i_max = np.argmax(power[:200])
    plt.plot(1/frequency, power)  
    plt.xlim(0,30)
    if save:
        plt.savefig("%s.png" % name , layout ="tight")
    plt.close()
    return power, frequency, 1/frequency[i_max]


def main(name_arr, ra_arr, dec_arr, size = 50, start_star = "", folder_output = "../tess_output", until_i=-1, pca_comp = 5, ap_wd=3, replace = False ):


    flag_star = 0
    if len(start_star) ==0:
        flag_star = 1


    for i in range(len(ra_arr)):
        folder_out_now = folder_output + "/output/%s" % name_arr[i]
        np_folder_out_now = folder_output + "/output/%s/lcs.npz" % name_arr[i]

        if name_arr[i] == start_star:
            flag_star = 1

        if flag_star == 0:
            continue

        if not os.path.exists(folder_out_now) or replace:

            try:
                if not os.path.exists(folder_out_now):
                    os.makedirs(folder_out_now)    

                print("---Analyze:", name_arr[i], "-----")
                ra_now , dec_now = ra_arr[i], dec_arr[i]
                print(ra_now, dec_now)
                hdulist = data_dl(ra_now, dec_now, size)
                print("Num of Sectors:", len(hdulist))
                if len(hdulist) ==0:
                    continue
                tpfs,time_arr = load_tpf_hdulists(hdulist)

                ## lightcurves are all flagged
                target_pca, target_ori, lcs_pca_all, lcs_ori_all, svd_comps, icen_arr, images, pos_peaks, \
                pos_stars, times, bkg_arrs, tpfs_target, flags_used = return_lc_from_tpfs(tpfs, time_arr, pca_comp = pca_comp, ap_wd=ap_wd) 
                plot_2dimages(folder_out_now, images, pos_stars)
                if len(pos_stars) ==0:
                    continue
                target_time = times_flagged(times, flags_used)
                target_pca_1d = arrs_to_1darr_med(target_pca)
                target_ori_1d = arrs_to_1darr_med(target_ori)
                np.savez(np_folder_out_now, target_pca =target_pca, target_ori =target_ori, target_time = target_time, lcs_pca_all = lcs_pca_all, lcs_ori_all = lcs_ori_all, \
                    svd_comps = svd_comps, icen_arr = icen_arr, pos_stars = pos_stars, time = times, bkg = bkg_arrs, tpf_target = tpfs_target)
                plot_lcs(lcs_ori_all, times, flags_used, icen_arr, target_ori_1d, target_pca_1d, folder_out_now +"/lcs")
                plot_bkg(times, bkg_arrs, flags_used, folder_out_now +"/lcs")
                ana_lomb(target_time, target_ori_1d, save = True, name = folder_out_now +"/lomb_ori")
                ana_lomb(target_time, target_pca_1d, save = True, name =folder_out_now +"/lomb_pca")
                centroid.centroid_tpfs(tpfs_target, folder_out_now +"/cent",  r_aperture=None)
                print("---Analyze End:", name_arr[i], "-----")
                print()
            except:
                print("error:", name_arr[i])

        if i==until_i:
            break

def svd_add(lc, svd_comps, ncomp=3):
    n_sec, n_data = np.shape(lc)
    for i in range(n_sec):
        lc_mean = np.mean(lc)
        flux_sub_mean = lc - lc_mean
        V_sub_from_lc = svd_comps[i][:ncomp] ## Component for subtraction 
        coeff_V = np.dot(flux_sub_mean,V_sub_from_lc.T) ## Coefficient for subtraction 
        sub_component = np.dot(coeff_V, V_sub_from_lc) ## Lightcurve for subtraction 
        after_subtracted = flux_sub_mean - sub_component + lc_mean ## Output after subtraction of systematics

    return after_subtracted


def lc_analysis(name_star, folder_name = "./output"):
    folder_name_all = os.path.join(folder_name, name_star)
    file_lcs = os.path.join(folder_name_all, "lcs.npz")
    data = np.load(file_lcs)
    lc_ori = data["target_ori"]
    lc_pca = data["target_pca"]
    time = data["times"]
    flag_time = data["times"]





