import numpy as np
# from scipy.spatial.transform import Rotation as R

# from curvedpy.geodesics.LUT.skydome.skydome_LUT_generation import load_data_file

from time import time
import os
import pickle
from PIL import Image
from random import random
from scipy.interpolate import RegularGridInterpolator

# from curvedpy.utils.coordinates_LPN import impact_vector, unit_vectors_lpn, matrix_conversion_lpn_xyz #impact_vector_2 as impact_vector
from curvedpy.geodesics.pointpoint import BlackholeGeodesicPointPointIntegrator


def generate_LUT():

    x_st, x_end = -5, 5
    y_st, y_end = -5, 5
    l_x = np.linspace(x_st, x_end, step)
    l_y = np.linspace(y_st, y_end, step)



    for ix, x in enumerate(l_x):
        for iy, y in enumerate(l_y):
            k0, L, l_k, l_x, l_results = gi.geodesic_pp(x_f_xyz, image_nr=1, eps_r=0.1)