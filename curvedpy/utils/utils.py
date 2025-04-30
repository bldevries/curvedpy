import numpy as np

# # Get the impact parameter and vector
# def getImpactParam(self, loc_hit, dir_hit):
#     # We create a line extending the dir_hit vector
#     line = list(zip(*[loc_hit + dir_hit*l for l in range(20)]))
#     # This line is used to construct the impact_vector
#     impact_vector = loc_hit - loc_hit.dot(dir_hit)*dir_hit
#     # We save the length of the impact_vector. This is called the impact parameter in
#     # scattering problems
#     impact_par = np.linalg.norm(impact_vector)
#     # We normalize the impact vector. This way we get, together with dir_hit, an 
#     # othonormal basis
#     if impact_par != 0:
#         impact_vector_normed = impact_vector/impact_par # !!! Check this, gives errors sometimes
#     else:
#         impact_vector_normed = impact_vector

#     return impact_vector_normed, impact_par


# Get the impact parameter and vector
def getImpactParam(ray_origin, ray_direction):
    impact_vector = ray_origin - ray_origin.dot(ray_direction)*ray_direction
    # We save the length of the impact_vector. This is called the impact parameter in
    # scattering problems
    impact_par = np.linalg.norm(impact_vector)
    # We normalize the impact vector. This way we get, together with dir_hit, an 
    # othonormal basis
    if impact_par != 0:
        impact_vector_normed = impact_vector/impact_par # !!! Check this, gives errors sometimes
    else:
        impact_vector_normed = impact_vector

    return impact_vector_normed, impact_par


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


