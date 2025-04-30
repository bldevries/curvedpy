import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import RegularGridInterpolator

# Gives the deflection of a ray towards the skydome
def get_skydome_direction(k0, x0, interp_hit, interp_th, data_pre, verbose=False):
    if interp_hit != None:
        hit, th, p0_, l0_ = get_hit_theta_interpolated(k0, x0, interp_hit, interp_th, data_pre, verbose=verbose)
    else:
        hit, th, p0_, l0_ = get_hit_theta(k0, x0 , data_pre, verbose=verbose)

    if hit == 0:
        q = create_quat(th, np.cross(l0_,p0_))
        return hit, q.apply(k0)
    else:
        return hit, None

# Angle between two 3-vectors
def angle(v1, v2):
    arg = np.dot(v1, v2)/ (np.linalg.norm(v1)*np.linalg.norm(v2))
    if arg <= 1 and arg >= -1:
        return np.arccos(arg)
    else:
        if round(abs(arg), 4) == 1:
            return 0.0
        else:
            print("Arccos outside domain")


# Calculate the impact vector (p) and distance vector (l) that form a plane 
# containing a geodesic in the Schwarzschild metric
#
# RAY_DIR NEEDS TO BE NORMALIZED!!
#
def impact_vector(ray_origin, ray_direction):
    # Keep in mind the min/plus character of l!
    l = ray_origin.dot(ray_direction)
    l_ = l * ray_direction
    p_ = ray_origin - l_ #- l*ray_direction
    p = np.linalg.norm(p_)
    return p, l, p_, l_
    

# Create a quaternion for rotating a 3-vector in a plane (with normal n) by
# an angle theta
# q = create_quat(dTh, np.cross(p_, l_))
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
def create_quat(theta, n):
    n = n / np.linalg.norm(n)
    w = np.cos(theta / 2)
    xyz = np.sin(theta/2) * n
    return R.from_quat(np.append(xyz, w))


# Get the hit and theta values closest to those belonging to
# a geodesic with initial values k0 and x0.
def get_hit_theta(k0,x0, data_pre, verbose=False):
    def find_nearest(array, value):
        #array = np.asarray(array)
        idx = np.nanargmin(np.abs(array - value))
        return idx#, array[idx]

    hit, theta, l_p, l_l = data_pre
    
    p0, l0, p0_, l0_ = impact_vector(x0, k0)

    ip0 = find_nearest(l_p, p0)
    il0 = find_nearest(l_l, l0)
    #hit[ip,il], th[ip,il]
    
    h = hit[ip0,il0]#interp_hit([p, l])
    if h == 1 or h == -1:
        th = None
    else:
        th = theta[ip0, il0] #interp_th([p,l])[0]
    
    return h, th, p0_, l0_


def get_hit_theta_interpolated(k0,x0, interp_hit, interp_th, data_pre, verbose=False):
    def find_nearest(array, value):
        #array = np.asarray(array)
        idx = np.nanargmin(np.abs(array - value))
        return idx#, array[idx]

    hit, th, l_p, l_l = data_pre
    
    p, l, p0_, l0_ = impact_vector(x0, k0)

    ip = find_nearest(l_p, p)
    il = find_nearest(l_l, l)
    #hit[ip,il], th[ip,il]
    
    h = hit[ip,il]#interp_hit([p, l])
    th = interp_th([p,l])[0]
    
    return h, th, p0_, l0_