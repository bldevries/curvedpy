import numpy as np

# ################################################################################################
# ################################################################################################
# class coordinates_LPN:
# ################################################################################################
# ################################################################################################

#################################################
def impact_vector(k, x):
    """ 
    Returns the impact parameter (p) and the distance to it (l) based on 
    location (x) and direction (k). Also returns the impact vector (p_) 
    and distance vector (l_)

    Returns:
        float: l
        float: p
        np.array: l_
        np.array: p_
    """

    nk = 1./np.linalg.norm(k, axis=1)
    k = (k.T*nk).T#, axis=1)

    l = np.einsum('ij,ij->i', x, k) # Row-wise inner product of vectors
    #l_ = l * k
    l_ = (l * k.T).T
    
    p_ = x - l_
    p = np.linalg.norm(p_, axis=1)

    return np.array([l, p]).T, l_, p_

#################################################
def impact_vector_no_vec(k, x):
    """ 
    Returns the impact parameter (p) and the distance to it (l).
    Also returns the impact vector (p_) and distance vector (l_)

    This is a non vectorized version

    Returns:
        float: l
        float: p
        np.array: l_
        np.array: p_
    """

    k = np.array([k])
    x = np.array([x])
    _, l_, p_ = impact_vector(k, x)
    l, p = _.T
    return np.array([l[0], p[0]]).T, l_[0], p_[0]



#################################################
def unit_vectors_lpn(k, x):
    l_ = k
    n_ = np.cross(l_, x)
    # for i, in_ in enumerate(n_):
    #     print(in_, np.linalg.norm(in_), l_[i])
    #     if np.linalg.norm(in_) == 0.0: # Singular!
    #         print("Fixing")
    #         ra = np.random.rand(3)
    #         n_[i] = np.cross(l_[i], ra) # WAAROM WERKT DEZE ASSIGNMENT NIET????
    #         print(n_[i], ra, l_[i], np.cross(l_[i], ra))
    # print(n_)
    p_ = np.cross(n_, l_)

    norm_l = np.linalg.norm(l_, axis=1)
    norm_n = np.linalg.norm(n_, axis=1)
    norm_p = np.linalg.norm(p_, axis=1)

    l_hat = (1./norm_l * l_.T).T
    p_hat = (1./norm_p * p_.T).T
    n_hat = (1./norm_n * n_.T).T

    return l_hat, p_hat, n_hat

#################################################
def unit_vectors_lpn_no_vec(k, x):
    k = np.array([k])
    x = np.array([x])
    l_hat, p_hat, n_hat = unit_vectors_lpn(k, x)
    return l_hat[0], p_hat[0], n_hat[0]

#################################################
def matrix_conversion_lpn_xyz(l_hat, p_hat, n_hat):
    """ Matrices to go from xyz to lpn coordinate system and back"""

    # Can this be faster without list comprehension!!

    # M_xyz_lpn = np.array([l_hat[0], p_hat[0], n_hat[0]]).T
    M_lpn_xyz = [np.array([l_hat[i], p_hat[i], n_hat[i]]).T for i in range(l_hat.shape[0])]
    M_xyz_lpn = [np.linalg.inv(M) for M in M_lpn_xyz]
    return M_xyz_lpn, M_lpn_xyz#np.linalg.inv(M_xyz_lpn)

#################################################
def matrix_conversion_lpn_xyz_no_vec( l_hat, p_hat, n_hat):
    """ Matrices to go from xyz to lpn coordinate system and back. Non vectorized. """

    l_hat = np.array([l_hat])
    p_hat = np.array([p_hat])
    n_hat = np.array([n_hat])

    M_xyz_lpn, M_lpn_xyz = matrix_conversion_lpn_xyz(l_hat, p_hat, n_hat)

    return M_xyz_lpn[0], M_lpn_xyz[0]

#################################################
def angle(v1, v2):
    """Return the angle between two 3-vectors. Vectorized."""
    arg = np.einsum('ij,ij->i', v1, v2) # Row-wise inner product of vectors
    norm = np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)
    #arg = np.dot(v1, v2)/ (np.linalg.norm(v1)*np.linalg.norm(v2))
    arg = (1/norm * arg.T).T
  
    return np.arccos(arg)

#################################################
# Create a quaternion for rotating a 3-vector in a plane (with normal n) by
# an angle theta
# q = create_quat(dTh, np.cross(p_, l_))
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_quat.html
def create_quat(theta, n):
    """ Creates a quaternion rotation matrix to rotate th degrees around the vector n"""
    n = n / np.linalg.norm(n)
    w = np.cos(theta / 2)
    xyz = np.sin(theta/2) * n
    return R.from_quat(np.append(xyz, w))




#################################################
# Remove these?
#################################################
def lp_to_sph2(self, x_lp):
    # k_lp = [1, 0]
    l, p = x_lp.T
    phi = np.atan2(p, l)
    r = np.sqrt(l**2 + p**2)
    x_sph2 = np.array([r, phi]).T

    dr_l = l/r
    dr_p = p/r
    dphi_l = -p/r**2
    dphi_p = -l/r**2

    # Since k_lp = [1, 0]
    k_sph2 = np.array([dr_l, dphi_l]).T

    return k_sph2, x_sph2

#################################################
# Remove these?
#################################################
def sph2_to_lp(self, k_sph2, x_sph2):
    r, phi = x_sph2.T
    l = np.cos(phi) * r
    p = np.sin(phi) * r

    dl_r = l/r
    dl_phi = -p
    dp_r = p/r
    dp_phi = l

    M = np.array([[dl_r, dl_phi], [dp_r, dp_phi]])

    x_lp = np.array([l, p]).T
    k_lp = np.array([M@k for k in k_sph2.T]) # BEter Doen!!

    return k_lp, x_lp


#################################################
# Remove these?
#################################################
def unit_vectors_lpn_slow(self, k, x):
    """Slow but classical way of calculating """
    l_ = k
    n_ = np.cross(l_, x)
    # for i, in_ in enumerate(n_):
    #     print(in_, np.linalg.norm(in_), l_[i])
    #     if np.linalg.norm(in_) == 0.0: # Singular!
    #         print("Fixing")
    #         ra = np.random.rand(3)
    #         n_[i] = np.cross(l_[i], ra) # WAAROM WERKT DEZE ASSIGNMENT NIET????
    #         print(n_[i], ra, l_[i], np.cross(l_[i], ra))
    # print(n_)
    p_ = np.cross(n_, l_)

    norm_l = np.linalg.norm(l_, axis=1)
    norm_n = np.linalg.norm(n_, axis=1)
    norm_p = np.linalg.norm(p_, axis=1)

    l_hat = (1./norm_l * l_.T).T
    p_hat = (1./norm_p * p_.T).T
    n_hat = (1./norm_n * n_.T).T

    return l_hat, p_hat, n_hat

