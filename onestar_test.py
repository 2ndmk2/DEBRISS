import numpy as np
import matplotlib.pyplot as plt
import tess_lc
import importlib
import argparse
parser = argparse.ArgumentParser(description='Process TESS DATA')
parser.add_argument('-npix', type=int, help='width of tpf', default=70)
parser.add_argument('-npca', type=int, help='num of pca', default=3)
parser.add_argument('-r_ap', type=int, help='aperture radius', default=3)
parser.add_argument('-ra', type=int, help='Ra', default=164.1266167)
parser.add_argument('-dec', type=int, help='Ra', default=-77.1942783)
parser.add_argument('-name', type=str, help='Ra', default="SYCha")


args = parser.parse_args()
name_arr = [args.name]
ra_arr = [args.ra]
dec_arr = [args.dec]

tess_lc.main(name_arr, ra_arr, dec_arr, size =args.npix , start_star = "", replace=True, pca_comp = args.npca, ap_wd=args.r_ap)



