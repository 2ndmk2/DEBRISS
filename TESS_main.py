import numpy as np
import matplotlib.pyplot as plt
import tess_lc
import importlib
import argparse
parser = argparse.ArgumentParser(description='Process TESS DATA')
parser.add_argument('-nstar', type=int, help='number of stars for analysis', default=100000)
parser.add_argument('-npix', type=int, help='width of tpf', default=70)
parser.add_argument('-npca', type=int, help='num of pca', default=3)
parser.add_argument('-r_ap', type=int, help='aperture radius', default=3)
parser.add_argument('-file_table', type=str, help='input file for table', default="./debris/table.dat")
parser.add_argument('-result_folder', type=str, help='input file for table', default="../tess_output")


args = parser.parse_args()
name_arr, ra_arr, dec_arr, spt_arr, T_arr, ldisk = tess_lc.catalog_load_and_save(args.file_table, file_out=args.table_out)
tess_lc.main(name_arr, ra_arr, dec_arr, until_i =args.nstar,folder_output = args.result_folder, size =args.npix , start_star = "", replace=True, pca_comp = args.npca, ap_wd=args.r_ap)



