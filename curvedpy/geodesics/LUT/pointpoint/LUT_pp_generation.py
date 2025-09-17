import numpy as np
# from scipy.spatial.transform import Rotation as R

# from curvedpy.geodesics.LUT.skydome.skydome_LUT_generation import load_data_file

from time import time
import os
import os
import pickle
from PIL import Image
from random import random
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import curvedpy as cp



def generate_LUT(lp_range=20, lp_number = 5, cam_loc = np.array([10,0,0]), prepend_filename="LUT_pp", save_directory=".", verbose=False):

    gi = cp.BlackholeGeodesicPointPointIntegrator()

    list_l = np.linspace(-lp_range, lp_range, lp_number)#40)
    list_p = np.linspace(0.000001, lp_range+0.000001, lp_number)#40)

    list_trans_l = np.zeros((list_l.shape[0],list_p.shape[0]))
    list_trans_p = np.zeros((list_l.shape[0],list_p.shape[0]))

    # lgr, pgr = np.meshgrid(list_l,list_p)#, indexing='ij')

    # If you want to have the geodesics
    list_geos = []

    if verbose: print("Run pp interpolation..")
    for il, l in enumerate(list_l):
        if verbose: print(il)
        for ip, p in enumerate(list_p):
            if l != cam_loc[0] and p!= cam_loc[1]:
                if np.linalg.norm([l, p]) > 2*gi.gi.M:
                    translation_lpn, _, _, _, geo, _ = gi.geodesic_pp_lpn_no_vec(x_f_lpn=np.array([l, p,0]), x0_lpn = cam_loc, image_nr=1)
                    list_trans_l[il, ip], list_trans_p[il, ip], _ = translation_lpn
                    list_geos.append(geo)

    filename = prepend_filename+f"_lp_range_{lp_range}_lp_number_{lp_number}.pkl"
    save_path = os.path.join(save_directory, filename)

    with open(save_path, 'wb') as f:
        pickle.dump([list_l, list_p, list_trans_l, list_trans_p], f)

    if verbose: print(f".. done .. saved to {save_path}")

    return read_in_LUT(save_directory, filename)


def read_in_LUT(save_directory, filename, return_all=False):
    save_path = os.path.join(save_directory, filename)

    with open(save_path, 'rb') as f:
        list_l, list_p, list_trans_l, list_trans_p = pickle.load(f)

        interp_trans_l = RegularGridInterpolator((list_l, list_p), list_trans_l)
        interp_trans_p = RegularGridInterpolator((list_l, list_p), list_trans_p)

    if return_all:
        return interp_trans_l, interp_trans_p, list_l, list_p, list_trans_l, list_trans_p
    else:
        return interp_trans_l, interp_trans_p

def generate_LUT_sph(image_nr=1, r_range=20, r_ph_number = 5, cam_loc = np.array([10,0,0]), prepend_filename="LUT_pp", save_directory=".", verbose=False):

    gi = cp.BlackholeGeodesicPointPointIntegrator()

    list_ph = np.linspace(0.000001, np.pi-0.000001, r_ph_number)#40)
    list_r = np.linspace(2*gi.gi.M+0.000001, r_range+0.000001, r_ph_number)#40)

    list_trans_l = np.zeros((list_r.shape[0],list_ph.shape[0]))
    list_trans_p = np.zeros((list_r.shape[0],list_ph.shape[0]))

    # lgr, pgr = np.meshgrid(list_l,list_p)#, indexing='ij')

    # If you want to have the geodesics
    list_geos = []

    if verbose: print("Run pp interpolation..")
    for ir, r in enumerate(list_r):
        if verbose: print(ir)
        for iph, ph in enumerate(list_ph):  
            if r != np.linalg.norm(cam_loc) and r*np.cos(ph) != cam_loc[0] and r*np.sin(ph) != cam_loc[1]:
                l = np.cos(ph)*r
                p = np.sin(ph)*r
                # if verbose: print(l ,p, r, ph)
                translation_lpn, _, _, _, geo, _ = gi.geodesic_pp_lpn_no_vec(x_f_lpn=np.array([l, p,0]), x0_lpn = cam_loc, image_nr=image_nr)

                list_trans_l[ir, iph], list_trans_p[ir, iph], _ = translation_lpn
                list_geos.append(geo)

    filename = prepend_filename+f"_{image_nr=}_r_range_{r_range}_r_ph_number_{r_ph_number}.pkl"
    save_path = os.path.join(save_directory, filename)

    with open(save_path, 'wb') as f:
        pickle.dump([list_r, list_ph, list_trans_l, list_trans_p], f)

    if verbose: print(f".. done .. saved to {save_path}")

    return read_in_LUT(save_directory, filename)


def read_in_LUT_sph(save_directory, filename, return_all=False):
    save_path = os.path.join(save_directory, filename)

    with open(save_path, 'rb') as f:
        list_r, list_ph, list_trans_l, list_trans_p = pickle.load(f)

        interp_trans_l = RegularGridInterpolator((list_r, list_ph), list_trans_l)
        interp_trans_p = RegularGridInterpolator((list_r, list_ph), list_trans_p)

    if return_all:
        return interp_trans_l, interp_trans_p, list_r, list_ph, list_trans_l, list_trans_p
    else:
        return interp_trans_l, interp_trans_p
