#import sympy as sp

import numpy as np  # Thinly-wrapped numpy
import sympy as sp

#from autograd import grad #
#from autograd import elementwise_grad as egrad  # for functions that vectorize over inputs
#from autograd import jacobian  # for functions that vectorize over inputs

from curvedpy.utils.stringified_equations import KerrSchildStrings
from sympy.parsing.sympy_parser import parse_expr

from curvedpy.utils.conversions_4D import Conversions4D
from curvedpy.utils.conversions import Conversions

from scipy.optimize import fsolve


# !!
# Kerr Schild Metrics implemented using Numpy and Autograd
# !!
# Author: B.L. de Vries

################################################################################################
################################################################################################
class KerrSchildMetricXYZ:
################################################################################################
################################################################################################

    conversions4D = Conversions4D()
    conversions = Conversions()


    ################################################################################################
    # INIT
    ################################################################################################
    def __init__(self, mass=1.0, a = 0.1, time_like = False, verbose=False):
        self.mass = mass
        self.a = a

        # Type of geodesic
        self.time_like = time_like # No Massive particle geodesics yet
        self.verbose = verbose

        t, x, y, z, r, a, m = sp.symbols('t x y z r a m')
        R2 = x**2 + y**2 + z**2
        r_sub = sp.sqrt( (R2 - a**2 + sp.sqrt((R2 - a**2)**2 +4*a**2*z**2 ) )/2 )

        self.r_lamb = sp.lambdify([x, y, z, a], r_sub, 'numpy')
        self.g__mu__nu = sp.lambdify([t, x, y, z, r, a, m], parse_expr(KerrSchildStrings.str_g__mu__nu), 'numpy')
        self.g_mu_nu = sp.lambdify([t, x, y, z, r, a, m], parse_expr(KerrSchildStrings.str_g_mu_nu), 'numpy')
        self.g__mu__nu_diff = [sp.lambdify([t, x, y, z, r, a, m], parse_expr(KerrSchildStrings.str_g__mu__nu_diff[i]), 'numpy') \
                                for i in [0,1,2,3]]


        # Outer horizon in BL coordinate r
        self.r_plus = self.mass + np.sqrt(self.mass**2 - self.a**2) # See Visser 2008, page 24, eq 104

    ################################################################################################
    #
    ################################################################################################
    def check_horizin(self, x, y, z):
        rH = self.mass+ np.sqrt(self.mass**2 - self.a**2)

        # This could be another way to determine if we hit the horizon
        # rIt = self.conversions.coord_conversion_xyz_bl(x, y, z, self.a)[0]
        # return rIt - (rH+0.000001)
        
        # This is the equation given by Visser
        # See Visser 2008 page 28 (eq 126)
        return x**2 + y**2 + (2*self.mass/self.r_plus) * z**2 - 2*self.mass*self.r_plus

    ################################################################################################
    #
    ################################################################################################
    def get_dk(self, kt_val, kx_val, ky_val, kz_val, t_val, x_val, y_val, z_val):
        # Calc g_inv and g_diff at given coords
        r_val = self.r_lamb(x_val, y_val, z_val, self.a)
        g_inv = self.g_mu_nu(t_val, x_val, y_val, z_val, r_val, self.a, self.mass)
        g_diff = [self.g__mu__nu_diff[i](t_val, x_val, y_val, z_val, r_val, self.a, self.mass) for i in [0,1,2,3]]

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
        r_val = self.r_lamb(x4[1], x4[2], x4[3], self.a)

        return (k4.T @ self.g__mu__nu(*x4, r_val, self.a, self.mass) @ k4) #t, x, y, z, r, a, m

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




