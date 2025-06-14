import numpy as np
from scipy.spatial.transform import Rotation as R

from curvedpy.geodesics.LUT.skydome.skydome_LUT_generation import load_data_file

from time import time

# from curvedpy.utils.lpn_coordinates import impact_vector_2 as impact_vector
from curvedpy.utils.coordinates_LPN import impact_vector #impact_vector_2 as impact_vector

# from curvedpy.utils.utils import impact_vector


# NEEDS COMPLETE REWRITE! 15 5 2025

def make_bg_from_file(width, height, a_texture, height_im, width_im, verbose=False, timing=True):
    # LUT = "../LUT_data/test/adaptgridTrue_num20_coordSPH_m1.0_step0.1_Rend300.0_p0-200_l-200-200.pkl"
    LUT = "../LUT_data/_num50_coordSPH_m1.0_step0.1_Rend300.0_p0-200_l-200-200.pkl"
    
    #LUT += "_num50_coordSPH_m1.0_step0.1_Rend300.0_p0-600_l-600-600.pkl"
    #"../LUT_data/_num100_coordSPH_m1.0_step0.1_Rend300.0_p0-100_l-50-50.pkl"
    interp_hit, interp_th, data_pre = load_data_file(LUT, give_interpolate=True)

    return make_bg(interp_hit, interp_th, width, height, a_texture, height_im, width_im, verbose, timing)


def make_bg(interp_hit, interp_th, width, height, \
            a_texture, height_im, width_im, \
            x_camera = np.array([0.0000001,0,50]),\
            camera_rotation_euler_props = ['x', 0],\
            flat=False, verbose=False, timing=True):


    # Camera x resolution
    # width = 3
    # Camera y resolution
    # height = 3
    field_of_view_x = 0.9
    field_of_view_y = 0.9
    aspectratio = 1
    # Camera position
    # x_camera = np.array([0.0000001,0,50])#np.array([0,400,0])
    # Camera rotation
    # camera_rotation_euler_props=['x', 0]#['x', -90]
    camera_rotation_euler = R.from_euler(camera_rotation_euler_props[0], camera_rotation_euler_props[1], degrees=True)
    camera_rotation_matrix = camera_rotation_euler.as_matrix()


    # Generate k0: ray directions out of the camera
    if timing: _ = time()
    k0 = np.array(\
        [ camera_rotation_matrix@np.array(\
        [field_of_view_x * (x-int(width/2))/width, \
        field_of_view_y * (y-int(height/2))/height * aspectratio, \
        -1]) for y in range(height) for x in range(width) ]\
        )
    if timing: print("t k0", time() - _)

    # Ray start is at the camera location
    x0 = np.full(k0.shape, x_camera)

    # if timing: _ = time()
    # l = x0.dot(k0.T)
    # l_ = (l * k0.T).T
    # p_ = x0 - l_ #- l*ray_direction
    # p = np.linalg.norm(p_, axis=1)
    # if timing: print("t pl", time() - _)

    # Calculate impact parameters p and l of k0
    if timing: _ = time()
    lp, l_, p_ = impact_vector(k0, x0)
    l, p = lp.T
    if timing: print("t pl NEW", time() - _)

    print("minmax", np.min(l), np.max(l), np.max(p))

    # (Solve the hit black hole information)

    # Interpolate to get deflection thetas
    if timing: _ = time()
    if np.any(l<-200): print("SMALL L")
    if np.any(l>200): print("LARGE L")
    l[l<-200] = -200
    l[l>200] = 200
    lp = np.column_stack( np.array([l, p]) )
    hit = interp_hit(lp) #hit[ip,il]#
    mask = hit == 0
    th = interp_th(lp)
    if timing: print("t interp", time() - _)

    # for h in hit: print(h)

    # Calculate rotation normals
    if timing: _ = time()
    n = np.cross(l_,p_)
    if any(np.linalg.norm(n, axis = 1) == 0):
        for i, ni in enumerate(n):
            if np.linalg.norm(ni) == 0:
                print("div zero: ", ni, p[i], l[i], p_[i], l_[i])
    n = (n.T / np.linalg.norm(n, axis = 1)).T
    # Rotate k0 to an exit angle
    a_w = np.cos(th / 2.)
    a_xyz = (np.sin(th/2.) * n.T).T
    if timing: print("t cross", time() - _)

    if flat:
        k_end = np.array(k0)
    else:
        if timing: _ = time()
        k_end = np.array([R.from_quat(np.append(a_xyz[i], a_w[i])).as_matrix()@k0[i] if hit[i] == 0 else np.array([0,0,0]) for i in range(len(a_w)) ])

        # k_end = np.array([R.from_quat(np.append(a_xyz[i], a_w[i])).apply(k0[i]) if hit[i] == 0 else np.array([0,0,0]) for i in range(len(a_w)) ])
        if timing: print("t apply rot", time() - _)

    # Get pixel coordinates 
    if timing: _ = time()
    k_end = (k_end.T / np.linalg.norm(k_end, axis = 1)).T
    if timing: print("t norm k_end", time() - _)

    if timing: _ = time()
    k_end = np.column_stack(k_end)
    th_im = np.arccos(k_end[2])/np.pi # [0,1] 
    #arctan2 loopt van [-pi, pi]
    phi_im = 1/2 - (1/2*np.arctan2(k_end[1], k_end[0])/np.pi) # [0,1]

    y_im, x_im = (th_im*(height_im-1)).astype(int), ((1-phi_im)*(width_im-1)).astype(int)
    if timing: print("t projecting", time() - _)

    #y_im, x_im = int(th_im*(height_im-1)), int((1-phi_im)*(width_im-1))



    # proj = np.zeros(width*height*4)
    # proj = proj.reshape((width*height, 4))
    channels = a_texture.shape[2]
    if channels == 4:
        b = np.array([0,0,0,255])
    else:
        b = np.array([0,0,0])

    if timing: _ = time()
    proj = np.array([a_texture[y_im[i], x_im[i]] if mask[i] else b for i in range(len(mask))])
    print("WHOO", height,width)
    proj = proj.reshape((height, width, channels))
    if timing: print("t image building", time() - _)



    # [f(x) if condition else g(x) for x in sequence]
    # for x
    # a_texture[y, x], (512, 512, 4)

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


