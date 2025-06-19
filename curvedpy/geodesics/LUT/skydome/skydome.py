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

def render_background(\
            interp_hit, interp_l_end, interp_p_end, \
            width, height, \
            a_texture, height_im, width_im, \
            samples = 1,\
            x_camera = np.array([0.0000001,0,50]),\
            camera_rotation_euler_props = ['x', 0],\
            k0=[], x0=[], \
            flat=False, verbose=False, timing=True, debug = False):
    
    if len(k0) == 0:
        k0, x0 = get_camera(x_camera, width, height, timing)

    if debug: images = []
    for s in range(samples):
        _ = time()
        proj = make_bg( k0, x0, interp_hit, interp_l_end, interp_p_end, \
                        width, height, a_texture, height_im, width_im, \
                        x_camera, camera_rotation_euler_props,\
                        flat, verbose, timing)
        if timing: print(s, time()-_)
        

        if debug: images.append(proj)
        if s == 0:
            im = proj/samples
        else:
            im = im + proj/samples
    
    # im = im/samples
    #im[:,:,3]=255
    
    if debug: images.append(im)
    
    im = Image.fromarray(im.astype('uint8'))
#    im.save("ani_test_sampling/fr_"+str(fr)+"_"+str(sampling)+"_inter.png","PNG")
    
    if debug:
        return im, images
    return im

def make_bg(k0, x0, interp_hit, interp_l_end, interp_p_end, \
            width, height, \
            a_texture, height_im, width_im, \
            x_camera = np.array([0.0000001,0,50]),\
            camera_rotation_euler_props = ['x', 0],\
            flat=False, verbose=False, timing=True):
    '''Returns an array with an image of a black hole, using given camera properties and a texture'''

    # You can make an image from the returned array using for example:
    # im = Image.fromarray(proj.astype('uint8'))
    # im.save("ani1080p_test_img/"+str(fr)+"_inter.png","PNG")

    # TODO:
    # - Now width and height cant be different
    # - Vectorize camera k0 generation: 2x as fast when vectorized with einsum instead of list comprehension
    # - Multisampling
    # - Write tests

    
    # Camera position
    # x_camera = np.array([0.0000001,0,50])#np.array([0,400,0])
    # Camera rotation
    # camera_rotation_euler_props=['x', 0]#['x', -90]
    camera_rotation_euler = R.from_euler(camera_rotation_euler_props[0], camera_rotation_euler_props[1], degrees=True)
    camera_rotation_matrix = camera_rotation_euler.as_matrix()


    if timing: _ = time()
    k0 = np.einsum('ij,bj->bi', camera_rotation_matrix, k0)    # This means summing over j, because it is not present after the ->
                                                               # It means NO summing over b, because it IS present after the -> eventhough it appears twice.
                                                               # In effect this multiplies all k0 vectors with one matrix
    if timing: print("Time, rotating k0", time() - _)

    # Looking up in the LUT
    if timing: _ = time()
    if flat:
        k_end = np.array(k0)
        mask = [True for i in range(len(k_end))]
    else:
        k_end, hit = get_k_end_xyz(k0, x0, interp_hit, interp_l_end, interp_p_end)
        mask = hit == 0
    if timing: print("Time, LUT", time() - _)



    # Get pixel coordinates 
    if timing: _ = time()
    k_end = (k_end.T / np.linalg.norm(k_end, axis = 1)).T
    if timing: print("t norm k_end", time() - _)

    if timing: _ = time()
    k_end = np.column_stack(k_end)
    th_im = np.arccos(k_end[2])/np.pi # [0,1] 
    #arctan2 loopt van [-pi, pi]
    phi_im = 1/2 - (1/2*np.arctan2(k_end[1], k_end[0])/np.pi) # [0,1]
    if timing: print("Time, projecting", time() - _)

    if timing: _ = time()
    y_im, x_im = (th_im*(height_im-1)).astype(int), ((1-phi_im)*(width_im-1)).astype(int)
    if timing: print("Time, creating image indices", time() - _)

    #y_im, x_im = int(th_im*(height_im-1)), int((1-phi_im)*(width_im-1))



    # proj = np.zeros(width*height*4)
    # proj = proj.reshape((width*height, 4))
    channels = a_texture.shape[2]
    if channels == 4:
        b = np.array([0,0,0,255], dtype=a_texture.dtype)
    else:
        b = np.array([0,0,0], dtype=a_texture.dtype)

    if timing: _ = time()
    proj = np.array([a_texture[y_im[i], x_im[i]] if mask[i] else b for i in range(len(mask))])
    # print("WHOO", height,width)
    proj = proj.reshape((height, width, channels))
    if timing: print("Time, image array building", time() - _)


    if verbose: 
        print("k0", k0.shape)
        print("l", l.shape)
        print("l_", l_.shape)
        print("p", p.shape)
        print("p_", p_.shape)

        print("max p", np.max(p))
        print("max l", np.max(l))
        print("min l", np.min(l))

        print("hit", hit.shape)
        print("th", th.shape)
        print("n", n.shape)
        print("w", a_w.shape)
        print("xyz", a_xyz.shape)
        print("k_end", k_end.shape)
        print("th_im", th_im.shape)
        print("phi_im", phi_im.shape)
        print(y_im)
        print(x_im)
        print("channels", channels)

    return proj

def get_camera(x_camera, width, height, timing=False):

    # Generate k0: ray directions out of the camera
    field_of_view_x = 0.9
    field_of_view_y = 0.9
    aspectratio = height/width
    # This can be vectorized!?
    dy = aspectratio/height
    dx = 1./width
    if timing: _ = time()
    k0 = np.array([\
        np.array([  (field_of_view_x * (x-int(width/2))/width)+dx*(random()-0.5), \
                    (field_of_view_y * (y-int(height/2))/height * aspectratio)+dy*(random()-0.5), \
                    -1]) \
        for y in range(height) for x in range(width) \
        ])
    if timing: print("Time, creating k0", time() - _)

    # Ray start is at the camera location
    x0 = np.full(k0.shape, x_camera)

    return k0, x0

def get_k_end_xyz(k0, x0, interp_hit, interp_l_end, interp_p_end):
    '''Returns the end direction of a geodesic by looking it up in a LUT'''

    # Get unit vectors lpn system and get transformation matrices
    l_hat, p_hat, n_hat = unit_vectors_lpn(k0, x0)
    M_xyz_lpn, M_lpn_xyz = matrix_conversion_lpn_xyz(l_hat, p_hat, n_hat)
    
    # Transform x0 to lpn system
    # x0_lpn = M_xyz_lpn[0]@x0[0]
    x0_lpn = np.einsum('bij,bj->bi', M_xyz_lpn, x0) # This means summing over j, because it is not present after the ->
                                                    # It means NO summing over b, because it IS present after the -> eventhough it appears twice.
    l0, p0, n0 = x0_lpn.T
    lp0 = np.column_stack( np.array([l0, p0]) )
    n0 = np.zeros(len(l0))
    
    # Looking up if it hits the BH and the end direction
    hit = interp_hit(lp0)
    k_end_lpn = np.array([interp_l_end(lp0), interp_p_end(lp0), n0]).T

    # Transforming the end direction to xyz coordinates
    k_end_xyz = np.einsum('bij,bj->bi', M_lpn_xyz, k_end_lpn)
    return k_end_xyz, hit


def load_data_file(file_path, give_interpolate = True):
    '''Loads data for creating interpolations of hit BH and the end direction of a geodesic'''
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            hit, l_l_end, l_p_end, l_l, l_p = pickle.load(f)
            if give_interpolate:
                interp_hit, interp_l_end, interp_p_end = create_interpolation(hit, l_l_end, l_p_end, l_l, l_p)
                return interp_hit, interp_l_end, interp_p_end, [hit, l_l_end, l_p_end, l_l, l_p]
            else:
                return hit, l_l_end, l_p_end, l_l, l_p


def create_interpolation(hit, l_l_end, l_p_end, l_l, l_p):
    '''Creates numpy interpolation objects'''
    interp_hit = RegularGridInterpolator((l_l, l_p), hit)
    interp_l_end = RegularGridInterpolator((l_l, l_p), l_l_end)
    interp_p_end = RegularGridInterpolator((l_l, l_p), l_p_end)

    return interp_hit, interp_l_end, interp_p_end


def generate_inter_data(save_directory = ".", add_str_to_filename = ""):
    '''Generates the data for a LUT where you can lookup if a geodesic hits the BH and what its end direciton is'''
    cores = 9
    max_step=0.1
    R_end = 300
    curve_end = 5000
    
    m = 1
    num_density = 2
    num_dens_scale_up_factor = 4

    p_start, p_end = -100, 100
    l_start, l_end = -100, 100
    
    R_sch = 2*m
    
    # p_split > R_sch and p_split<p_end
    p_split = 4



    l_l = np.linspace(l_start, l_end, int(num_density*(l_end-l_start)))
    
    l_p_part1 = np.linspace(p_start, R_sch, int(num_density*(R_sch-p_start)))
    l_p_part2 = np.linspace(R_sch+0.01, p_split, int(num_density*(p_split-R_sch)*num_dens_scale_up_factor))
    l_p_part3 = np.linspace(p_split+0.01, p_end, int(num_density*(p_end-p_split)))
    l_p = np.concat([l_p_part1, l_p_part2, l_p_part3])
    

    l_l_end = np.zeros((l_l.shape[0],l_p.shape[0]))
    l_p_end = np.zeros((l_l.shape[0],l_p.shape[0]))
    hit = np.zeros((l_l.shape[0],l_p.shape[0]))

    ##
    l_k0, l_x0 = [], []
    l_i_lpn = []
    for il, l in enumerate(l_l):
        # if il%10 == 0: print("l", il)
        for ip, p in enumerate(l_p):
            #for j in range(100):
            if p == 0.0:
                p += 0.00001
            x0 = np.array([l, p, 0]) # CHECK THIS -x!!!!
            k0 = np.array([1, 0, 0])
            R0 = np.linalg.norm(x0)

            if R0 > 2*m+0.1:
                l_k0.append(k0)
                l_x0.append(x0)
                l_i_lpn.append([il, ip])
            else:
                print("norun")
                hit[il, ip] = -1 # Start in BH

    bi = cp.BlackholeGeodesicIntegrator(mass = m)
    print(f"Running {len(l_x0)=} models on {cores=}")
    _start = time()
    results = bi.geodesic_mp(cores=cores, k0_xyz=l_k0, x0_xyz=l_x0, max_step=max_step, R_end = R_end, curve_end = curve_end)
    print("Geodesics runtime: ", time()-_start)
    
    k_end_lpn = [results[i][0].T[-1] for i in range(len(results))]

    # You can cehck if n component is zero, should be!

    for i, result in enumerate(results):
        kgeo, xgeo, res = result
        il, ip = l_i_lpn[i]
        # dTh, l, p, l_, p_ = curve_props(kgeo, xgeo)
        hit[il, ip] = res['hit_blackhole']
        # th[il, ip] = dTh
        l_l_end[il, ip] = k_end_lpn[i][0]
        l_p_end[il, ip] = k_end_lpn[i][1]

    interp_hit = RegularGridInterpolator((l_l, l_p), hit)
    interp_l_end = RegularGridInterpolator((l_l, l_p), l_l_end)
    interp_p_end = RegularGridInterpolator((l_l, l_p), l_p_end)


    filename = add_str_to_filename+\
        f"p_split{p_split}_num_density{num_density}_num_dens_scale_up_factor{num_dens_scale_up_factor}"+\
        f"_m{m}_step{max_step}_Rend{R_end}_curve_end{curve_end}_p{p_start}-{p_end}_l{l_start}-{l_end}"+\
        ".pkl"
    
    save_path = os.path.join(save_directory, filename)

    with open(save_path, 'wb') as f:
        pickle.dump([hit, l_l_end, l_p_end, l_l, l_p], f)
    
    return interp_hit, interp_l_end, interp_p_end