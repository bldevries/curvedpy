#import sympy as sp

import autograd.numpy as np  # Thinly-wrapped numpy

#from autograd import grad #
#from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
from autograd import jacobian  # for functions that vectorize over inputs

from curvedpy.utils.conversions_4D import Conversions4D
from scipy.optimize import fsolve


# !!
# Schwarzschild Metrics implemented using Numpy and Autograd
# !!
# Author: B.L. de Vries

################################################################################################
################################################################################################
class SchwarzschildMetricXYZ_AUTOGRAD:
################################################################################################
################################################################################################

    conversions4D = Conversions4D()


    ################################################################################################
    # INIT
    ################################################################################################
    def __init__(self, mass=1.0, time_like = False, verbose=False):
        
        self.M = mass
        self.r_s_value = 2*self.M 

        # Type of geodesic
        self.time_like = time_like # No Massive particle geodesics yet
        self.verbose = verbose

        # List of functions giving the derivative of the metric to each of the
        # four coordinates
        # We are skipping the time coordinate since the metric does not depend on it
        def zero(t, x, y, z): return np.zeros((4,4))
        self.g__mu__nu_diff = [zero] + [jacobian(self.g__mu__nu, i) for i in [1,2,3]]

    ################################################################################################
    # Metric
    ################################################################################################
    def g__mu__nu(self, t, x, y, z):
        r_s=self.r_s_value
        r = np.sqrt(x**2 + y**2 + z**2)
        alp = r_s / (r**2 * (-r_s + r))

        g__00 = -1. + r_s/r
        g__01, g__02, g__03 = 0.,0.,0.
        g__10 = 0.
        g__11 = 1. + x**2 * alp
        g__12 = x * y * alp
        g__13 = x * z * alp
        g__20 = 0.
        g__21 = x * y * alp
        g__22 = 1. + y**2 * alp
        g__23 = y * z * alp
        g__30 = 0.
        g__31 = x * z * alp
        g__32 = y * z * alp
        g__33 = 1 + z**2 * alp

        return np.array([ [g__00,g__01,g__02,g__03], \
                            [g__10,g__11,g__12,g__13], \
                            [g__20,g__21,g__22,g__23], \
                            [g__30,g__31,g__32,g__33]])

    ################################################################################################
    # Inverse metric (raising indices)
    ################################################################################################
    def g_mu_nu(self, t, x, y, z):
        r_s=self.r_s_value
        r = np.sqrt(x**2 + y**2 + z**2)

        g_00 = -1./(1-r_s/r)
        g_01, g_02, g_03 = 0.,0.,0.
        g_10 = 0.
        g_11 = 1 - r_s  * x**2/r**3
        g_12 = -1. *r_s * x * y/r**3
        g_13 = -1. *r_s * x * z/r**3
        g_20 = 0.
        g_21 = -1. * r_s * x * y/r**3
        g_22 = 1. - r_s * y**2/r**3
        g_23 = -1. * r_s * y * z/r**3
        g_30 = 0.
        g_31 = -1. * r_s * x * z/r**3
        g_32 = -1. * r_s * y * z/r**3
        g_33 = 1 - r_s * z**2/r**3

        return np.array([   [g_00,g_01,g_02,g_03], \
                            [g_10,g_11,g_12,g_13], 
                            [g_20,g_21,g_22,g_23], 
                            [g_30,g_31,g_32,g_33]])

    ################################################################################################
    #
    ################################################################################################
    def get_dk(self, kt_val, kx_val, ky_val, kz_val, t_val, x_val, y_val, z_val):
        # Calc g, g_inv and g_diff at given coords
        #g = self.g__mu__nu(t_val, x_val, y_val, z_val)
        g_inv = self.g_mu_nu(t_val, x_val, y_val, z_val)
        g_diff = [self.g__mu__nu_diff[i](t_val, x_val, y_val, z_val) for i in [0,1,2,3]]

        # Calc the connection Symbols at given coords
        gam_t = np.array([[self.gamma_func(g_inv, g_diff, 0, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_x = np.array([[self.gamma_func(g_inv, g_diff, 1, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_y = np.array([[self.gamma_func(g_inv, g_diff, 2, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_z = np.array([[self.gamma_func(g_inv, g_diff, 3, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])

        # Building up the geodesic equation: 
        # Derivatives: k_beta = d x^beta / d lambda
        #self.k_t, self.k_x, self.k_y, self.k_z = sp.symbols('k_t k_x k_y k_z', real=True)
        #self.k = [self.k_t, self.k_x, self.k_y, self.k_z]
        k = [kt_val, kx_val, ky_val, kz_val]
    
        # Second derivatives: d k_beta = d^2 x^beta / d lambda^2
        dk_t = np.sum(np.array([- gam_t[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]]))
        dk_x = np.sum(np.array([- gam_x[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]]))
        dk_y = np.sum(np.array([- gam_y[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]]))
        dk_z = np.sum(np.array([- gam_z[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]]))

        return dk_t, dk_x, dk_y, dk_z

    ################################################################################################
    # Connection Symbols
    ################################################################################################
    def gamma_func(self, g_inv, g_diff, sigma, mu, nu):
        # 29 Jan 2025
        # A list comprehension does NOT speed the following code up!
        # Something like this is NOT better:
        # g_sigma_mu_nu = np.sum(np.array( [ 1/2 * g_inv[sigma, rho] * (g_diff[mu][nu, rho] \
        #               + g_diff[nu][rho, mu] - g_diff[rho][mu, nu] ) for rho in [0,1,2,3]] ) )
        # This is because you make a list and then sum, while you do not need to make the list.
        # I found that it is faster to use a normal for loop like this:
        g_sigma_mu_nu = 0
        for rho in [0,1,2,3]:
            g_sigma_mu_nu += 1/2 * g_inv[sigma, rho] * (\
                            g_diff[mu][nu, rho] + \
                            g_diff[nu][rho, mu] - \
                            g_diff[rho][mu, nu] )
        return g_sigma_mu_nu

    ################################################################################################
    # Norm of the four vector k
    ################################################################################################
    def norm_k(self, k4, x4):#k_t, k_x, k_y, k_z, t, x, y, z):
        # Norm of k
        # the norm of k determines if you have a massive particle (-1), a mass-less photon (0) 
        # or a space-like curve (1)
        return (k4.T @ self.g__mu__nu(*x4) @ k4)

    ################################################################################################
    # Solve k_t for the starting 4-momentum 
    ################################################################################################
    def k_t_from_norm(self, k0, x0, t=0):
        # Now we calculate k_t using the norm. This eliminates one of the differential equations.
        # time_like = True: calculates a geodesic for a massive particle
        # time_like = False: calculates a geodesic for a photon

        # Starting coordinates in 4D
        x4_0 = np.array([t, *x0])

        if (self.time_like):
            def wrap(k_t):
                # Starting 4-vector
                k4_0 = np.concatenate((k_t, k0))
                return self.norm_k(k4_0, x4_0)+1

            k_t_from_norm = fsolve(wrap, 1.0)
        else:
            def wrap(k_t):
                # Starting 4-vector
                k4_0 = np.concatenate((k_t, k0))
                return self.norm_k(k4_0, x4_0)
                
            k_t_from_norm = fsolve(wrap, 1.0)

        return k_t_from_norm[0]

