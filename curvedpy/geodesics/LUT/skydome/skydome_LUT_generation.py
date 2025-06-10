import random
import os
import pickle
import numpy as np
import curvedpy as cp
from curvedpy.geodesics.blackhole import BlackholeGeodesicIntegrator

from curvedpy.utils.coordinates_LPN import angle, create_quat
from curvedpy.utils.coordinates_LPN import impact_vector, impact_vector_no_vec #impact_vector_2 as impact_vector

from scipy.interpolate import RegularGridInterpolator



def curve_props(k, x, last_index = -1):
    """Calculate impact properties and angle of deflection from curvedpy geodesics

        Keyword arguments:
        k -- np.array containing spatial components of the 4momenta of geodesic points 
        x -- np.array containing spatial components of the 4position of a geodesic
        last_index --   optional index pointing to the last point to use from the k 
                        array to calculate the deflection (default -1)
    """
    x = np.column_stack(x)
    k = np.column_stack(k)
    
    ray_origin, ray_direction, ray_dir_end = x[0], k[0], k[last_index]

    ray_origin, ray_direction = np.array([ray_origin]), np.array([ray_direction])
    ray_dir_end = np.array([ray_dir_end])

    dTh = angle(ray_direction, ray_dir_end)
    # print("WHAAAA", ray_origin.shape, ray_direction.shape)
    lp, l_, p_ = impact_vector(ray_origin, ray_direction)

    l, p = lp.T

    # The angle between two vectors is ambiguous. This means arccos always picks
    # out the smallest angle between two vectors and the angle will never be bigger than 
    # pi. But we want the angle between the vector up to 2pi. Now if the angle between 
    # the start and end vector is larger than pi, we reverse the direction of theta
    # in order to get the proper rotation matrix out.
    if ray_dir_end.dot(p_[0]) >= 0.0:
        dTh= -dTh
    
    return dTh, l, p, l_, p_

def generate_data_file(save = True, save_directory = ".", add_str_to_filename= "", \
                        num_density=10, num_dens_scale_up_factor = 100,\
                        p_range = [0,200], l_range = [-200,200], adapt_grid=False,\
                        m=1.0, max_step=0.1, R_end = 300., curve_end = 5000, verbose=False):

    bi = cp.BlackholeGeodesicIntegrator(mass = m)

    p_start, p_end = p_range
    l_start, l_end = l_range

    print("Starting run!")
    print(f"    {p_range=} {l_range=}")
    print(f"    {num_density=} {num_dens_scale_up_factor=}")

    l_l = np.linspace(l_start, l_end, int(num_density*(l_end-l_start)))
    if adapt_grid:
        R_sch = 2*m
        if R_sch > p_end:
            R_sch = p_end

        p_split = 5*R_sch
        if p_split > p_end:
            p_split = p_end

        l_p_part1 = np.linspace(p_start, R_sch, int(num_density*(R_sch-p_start)))
        # if R_sch < p_end and p_split < p_end:
        if R_sch+0.01 < p_split: l_p_part2 = np.linspace(R_sch+0.01, p_split, int(num_density*(p_split-R_sch)*num_dens_scale_up_factor))
        else: l_p_part2 = []
        if p_split+0.01 < p_end: l_p_part3 = np.linspace(p_split+0.01, p_end, int(num_density*(p_end-p_split)))
        else: l_p_part3 = []

        l_p = np.concat([l_p_part1, l_p_part2, l_p_part3])
    else:
        l_p = np.linspace(p_start, p_end, int(num*(p_end-p_start)))
    

    th = np.zeros((l_l.shape[0],l_p.shape[0]))
    hit = np.zeros((l_l.shape[0],l_p.shape[0]))

    if verbose: print(f"Running {num_density=} {num_dens_scale_up_factor=} {coordinates=} {m=} {max_step=} {R_end=} {curve_end=} {verbose=}")

    l_x = []

    for il, x in enumerate(l_l):
        if il%10 == 0: print("l", il)
        for ip, y in enumerate(l_p):
            if verbose: print(il, ip, x, y)
            #for j in range(100):
            if y == 0.0:
                y += 0.00001
            x0 = [-x, y, 0]
            k0 = [-1, 0, 0]
            R0 = np.linalg.norm(x0)

            if R0 > 2*m+0.1:
                kgeo, xgeo, _ = bi.geodesic(k0_xyz=k0, x0_xyz=x0, max_step=max_step, R_end = R_end, curve_end = curve_end)
                l_x.append(xgeo)
                dTh, l, p, l_, p_ = curve_props(kgeo, xgeo)
                hit[il, ip] = _['hit_blackhole']
                th[il, ip] = dTh
            else:
                print("norun")
                hit[il, ip] = -1 # Start in BH

    filename = add_str_to_filename+\
        f"adaptgrid{adapt_grid}_num_density{num_density}_num_dens_scale_up_factor{num_dens_scale_up_factor}_m{m}_step{max_step}_Rend{R_end}_curve_end{curve_end}_p{p_start}-{p_end}_l{l_start}-{l_end}"+\
        ".pkl"

    save_path = os.path.join(save_directory, filename)

    with open(save_path, 'wb') as f:
        pickle.dump([hit, th, l_l, l_p], f)

    return hit, th, l_l, l_p, l_x

def load_data_file(file_path, give_interpolate = True):
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            hit, th, l_l, l_p = pickle.load(f)
            if give_interpolate:
                interp_hit, interp_th = create_interpolation(hit, th, l_l, l_p)
                return interp_hit, interp_th, [hit, th, l_l, l_p]
            else:
                return hit, th, l_l, l_p

def create_interpolation(hit, th, l_l, l_p):
    interp_hit = RegularGridInterpolator((l_l, l_p), hit)
    interp_th = RegularGridInterpolator((l_l, l_p), th)

    return interp_hit, interp_th












