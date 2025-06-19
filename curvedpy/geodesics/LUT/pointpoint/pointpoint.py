import numpy as np
from scipy.spatial.transform import Rotation as R

# from curvedpy.geodesics.LUT.skydome.skydome_LUT_generation import load_data_file

from time import time
import os
import pickle
from PIL import Image
from random import random
from scipy.interpolate import RegularGridInterpolator

from curvedpy.utils.coordinates_LPN import impact_vector, unit_vectors_lpn, matrix_conversion_lpn_xyz #impact_vector_2 as impact_vector

import curvedpy as cp

def generate_pp_inter_data():
    nr = 40
    step = 1
    angle = [-1/2*np.pi + np.pi*i/nr for i in range(nr)]
    k_cam = np.array([[np.sin(th), np.cos(th), 0] for th in angle])
    x_cam = np.array([[10, 0.00001, 0] for i in range(nr)])
    cores=9
    split_factor=1

    gi = cp.BlackholeGeodesicIntegrator(verbose=False)
    results = gi.geodesic_mp(   k0_xyz = k_cam, \
                            x0_xyz = x_cam, \
                            cores=cores,\
                            max_step=0.1,\
                            split_factor = split_factor,\
                            R_end = 150., curve_end = 200,\
                            )
    from_x, from_y = [], []
    from_r, from_th = [], []
    point_from, point_from_r_th = [], []
    to_x, to_y = [], []
    point_to = []
    to_inter_to_x, to_inter_to_y = [], []
    for i, res in enumerate(results):
        k, v, _ = res
        k = k.T
        v = v.T
        # print(k.shape)
        # print(v.shape)

        length = 0
        for i in range(len(k)):
            if i == 0: 
                k0 = k[0]/np.linalg.norm(k[0])
                x0 = v[0]
                length += np.linalg.norm(x0-x_cam)
            else:
                # print(v[i])
                x, y, z = v[i]
                if x <= 90 and x >= -90 and y <= 90 and y > 0:
                    if i != len(k)-1:
                        #length += np.linalg.norm(v[i]-v[i-1])
                    # else:
                        length += np.linalg.norm(v[i+1]-v[i])

                    r = np.sqrt(x**2+y**2)
                    th = np.sin(y/r)
                    from_r.append(r)
                    from_th.append(th)
                    point_from_r_th.append((r, th))

                    from_x.append(x)
                    from_y.append(y)
                    point_from.append((x,y))

                    #Bereken waar het geprojecteerde punt zit
                    to = x0 + k0*length
                    to_x.append(to[0])
                    to_y.append(to[1])
                    point_to.append((to_x,to_y))

                    to_inter_to_x.append((r, th, to[0]))
                    to_inter_to_y.append((r, th, to[1]))



    # Carve out a square grid in x, y

    # Calculate projection point

    # Make interpolation object

    return results, from_x, from_y, to_x, to_y, point_from, point_from_r_th, point_to, to_inter_to_x, to_inter_to_y