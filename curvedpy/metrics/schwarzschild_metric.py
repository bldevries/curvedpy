import sympy as sp
import numpy as np
from curvedpy.utils.conversions_4D import Conversions4D
from scipy.optimize import fsolve


# !!
# Schwarzschild Metrics implemented using Numpy and Sympy
# !!
# Author: B.L. de Vries

################################################################################################
################################################################################################
class SchwarzschildMetricXYZ:
################################################################################################
################################################################################################

    conversions4D = Conversions4D()

    def __init__(self, mass=1.0, time_like = False, verbose=False):
        
        self.M = mass
        self.r_s_value = 2*self.M 

        # Type of geodesic
        self.time_like = time_like # No Massive particle geodesics yet
        self.verbose = verbose

        # Define symbolic variables
        #self.t, self.r, self.th, self.ph, self.r_s = sp.symbols("t r \\theta \\phi r_s", real=True)
        self.t, self.x, self.y, self.z, = sp.symbols('t x y z', real=True)
        self.r, self.r_s  = sp.symbols('r r_s', positive=True, real=True)
        self.alp = sp.symbols('alp')
        

        self.r_sub = (self.x**2 + self.y**2 + self.z**2)**sp.Rational(1,2)
        self.alp_sub = self.r_s/(self.r**2*(-self.r_s+self.r)) # NOTE THE r_s is in here for most entries!

        flip_sign_convention = -1

        g00 = -1*( 1-self.r_s/self.r )
        g01, g02, g03 = 0,0,0
        g10 = 0
        g11 = -1*( -1-self.x**2 * self.alp )
        g12 = flip_sign_convention*( -self.x*self.y*self.alp )
        g13 = flip_sign_convention*( -self.x*self.z*self.alp )
        g20 = 0
        g21 = flip_sign_convention*( -self.x*self.y*self.alp )
        g22 = -1*( -1-self.y**2*self.alp )
        g23 = flip_sign_convention*( -self.y*self.z*self.alp )
        g30 = 0
        g31 = flip_sign_convention*( -self.x*self.z*self.alp )
        g32 = flip_sign_convention*( -self.y*self.z*self.alp )
        g33 = -1*( -1 - self.z**2*self.alp )

        self.g__mu__nu_cart = sp.Matrix([[g00,g01,g02,g03], [g10,g11,g12,g13], [g20,g21,g22,g23], [g30,g31,g32,g33]])
        self.g__mu__nu_cart_pre_sub = self.g__mu__nu_cart

        self.g__mu__nu_cart = self.g__mu__nu_cart.subs(self.alp, self.alp_sub).subs(self.r, self.r_sub)

        g_00 = -1*( 1/(1-self.r_s/self.r) )
        g_01, g_02, g_03 = 0,0,0
        g_10 = 0
        g_11 = -1*(-1+self.r_s  * self.x**2/self.r**3)
        g_12 = flip_sign_convention*self.r_s * self.x * self.y/self.r**3
        g_13 = flip_sign_convention*self.r_s * self.x * self.z/self.r**3
        g_20 = 0
        g_21 = flip_sign_convention* self.r_s * self.x * self.y/self.r**3
        g_22 = -1*( -1 + self.r_s * self.y**2/self.r**3 )
        g_23 = flip_sign_convention*( self.r_s * self.y * self.z/self.r**3 )
        g_30 = 0
        g_31 = flip_sign_convention* ( self.r_s * self.x * self.z/self.r**3 )
        g_32 = flip_sign_convention*( self.r_s * self.y * self.z/self.r**3 )
        g_33 = -1*( -1 + self.r_s * self.z**2/self.r**3 )

        self.g_mu_nu_cart = sp.Matrix([[g_00,g_01,g_02,g_03], [g_10,g_11,g_12,g_13], [g_20,g_21,g_22,g_23], [g_30,g_31,g_32,g_33]]).subs(self.r, self.r_sub)
        self.g_mu_nu_cart = self.g_mu_nu_cart.subs(self.alp, self.alp_sub).subs(self.r, self.r_sub)

        self.g__mu__nu_cart_diff = [self.g__mu__nu_cart.diff(self.t), self.g__mu__nu_cart.diff(self.x), \
                                     self.g__mu__nu_cart.diff(self.y), self.g__mu__nu_cart.diff(self.z)]

        # We lambdify these to get numpy arrays
        self.g__mu__nu_cart_lamb = sp.lambdify([self.t, self.x, self.y, self.z, self.r_s], self.g__mu__nu_cart, "numpy")
        self.g_mu_nu_cart_lamb = sp.lambdify([self.t, self.x, self.y, self.z, self.r_s], self.g_mu_nu_cart, "numpy")
        self.g__mu__nu_cart_diff_lamb = sp.lambdify([self.t, self.x, self.y, self.z, self.r_s], self.g__mu__nu_cart_diff, "numpy")

        # Norm of k
        # the norm of k determines if you have a massive particle (-1), a mass-less photon (0) 
        # or a space-like curve (1)
        self.k_t, self.k_x, self.k_y, self.k_z = sp.symbols('k_t k_x k_y k_z', real=True)
        self.k_mu_cart = sp.Matrix([self.k_t, self.k_x, self.k_y, self.k_z])
        self.norm_k = (self.k_mu_cart.T*self.g__mu__nu_cart*self.k_mu_cart)[0]

        self.norm_k_lamb = sp.lambdify([self.k_t, self.k_x, self.k_y, self.k_z, self.x, self.y, self.z, \
                                               self.r_s], self.norm_k, "numpy")


    ################################################################################################
    #
    ################################################################################################
    def get_dk(self, kt_val, kx_val, ky_val, kz_val, t_val, x_val, y_val, z_val):
        # Calc g, g_inv and g_diff at given coords
        g = self.g__mu__nu_cart_lamb(t_val, x_val, y_val, z_val, self.r_s_value)
        g_inv = self.g_mu_nu_cart_lamb(t_val, x_val, y_val, z_val, self.r_s_value)
        g_diff = self.g__mu__nu_cart_diff_lamb(t_val, x_val, y_val, z_val, self.r_s_value)

        # Calc the connection Symbols at given coords
        gam_t = np.array([[self.gamma_func(g, g_inv, g_diff, 0, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_x = np.array([[self.gamma_func(g, g_inv, g_diff, 1, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_y = np.array([[self.gamma_func(g, g_inv, g_diff, 2, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_z = np.array([[self.gamma_func(g, g_inv, g_diff, 3, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])

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

    # Connection Symbols
    def gamma_func(self, g, g_inv, g_diff, sigma, mu, nu):
        # 29 Jan 2025
        # A list comprehension does NOT speed the following code up!
        # This something like this is NOT better:
        # g_sigma_mu_nu = np.sum(np.array( [ 1/2 * g_inv[sigma, rho] * (g_diff[mu][nu, rho] + g_diff[nu][rho, mu] - g_diff[rho][mu, nu] ) for rho in [0,1,2,3]] ) )
        # This is because you make a list and then sum, while you do not need to make the list. Normal for loop is fasters that I found so far.
        g_sigma_mu_nu = 0
        for rho in [0,1,2,3]:
            g_sigma_mu_nu += 1/2 * g_inv[sigma, rho] * (\
                            g_diff[mu][nu, rho] + \
                            g_diff[nu][rho, mu] - \
                            g_diff[rho][mu, nu] )
        return g_sigma_mu_nu

    def k_t_from_norm(self, k0, x0, t=0):
        # Now we calculate k_t using the norm. This eliminates one of the differential equations.
        # time_like = True: calculates a geodesic for a massive particle
        # time_like = False: calculates a geodesic for a photon
        if (self.time_like):
            def wrap(k_t): return self.norm_k_lamb(k_t, k0[0], k0[1], k0[2], x0[0], x0[1], x0[2], self.r_s_value)+1
            k_t_from_norm = fsolve(wrap, 1.0)
        else:
            k_t_from_norm = root = fsolve(self.norm_k_lamb, 1.0, args = (k0[0], k0[1], k0[2], x0[0], x0[1], x0[2], self.r_s_value) )

        return k_t_from_norm[0]



################################################################################################
################################################################################################
class SchwarzschildMetricSpherical:
################################################################################################
################################################################################################

    conversions4D = Conversions4D()

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, time_like = False, eps_theta = 0.00000001, verbose=False):

        self.pole_metric_XYZ = SchwarzschildMetricXYZ(mass, time_like, verbose)

        # Connection Symbols
        def gamma_func(sigma, mu, nu):
            coord_symbols = [self.t, self.r, self.th, self.ph]
            gamma_sigma_mu_nu = 0
            for rho in [0,1,2,3]:
                gamma_sigma_mu_nu += 1/2 * self.g_mu_nu[sigma, rho] * (\
                                self.g__mu__nu_diff[mu][nu, rho] + \
                                self.g__mu__nu_diff[nu][rho, mu] - \
                                self.g__mu__nu_diff[rho][mu, nu] )
            return gamma_sigma_mu_nu

        self.M = mass
        #self.r_s_value = 2*self.M 
        self.r_s = 2*self.M 

        # Type of geodesic
        self.time_like = time_like # No Massive particle geodesics yet
        self.verbose = verbose
        if verbose:
            print("Geodesic SS Integrator Settings: ")
            print(f"  - {self.M=}")
            print(f"  - {self.r_s=}")
            print(f"  - {self.time_like=}")
            print(f"  - {self.verbose=}")
            print("--")

        # Define symbolic variables
        self.t, self.r, self.th, self.ph, = sp.symbols('t r \\theta \\phi', real=True)
        #self.r_s  = sp.symbols('r_s', positive=True, real=True)

        self.g__mu__nu = sp.Matrix([\
                            [-1*(1-self.r_s/self.r), 0, 0, 0],\
                            [0, 1/(1-self.r_s/self.r), 0, 0],\
                            [0, 0, self.r**2, 0],\
                            [0, 0, 0, self.r**2 * sp.sin(self.th)**2]\
                            ])

        self.g_mu_nu = self.g__mu__nu.inv()
        self.g__mu__nu_diff = [self.g__mu__nu.diff(self.t), self.g__mu__nu.diff(self.r), \
                                     self.g__mu__nu.diff(self.th), self.g__mu__nu.diff(self.ph)]

        # We lambdify these to get numpy arrays
        self.g__mu__nu_lamb = sp.lambdify([self.t, self.r, self.th, self.ph], self.g__mu__nu, "numpy")
        self.g_mu_nu_lamb = sp.lambdify([self.t, self.r, self.th, self.ph], self.g_mu_nu, "numpy")
        self.g__mu__nu_diff_lamb = sp.lambdify([self.t, self.r, self.th, self.ph], self.g__mu__nu_diff, "numpy")

        # Protect against dev by zero
        self.g_mu_nu = self.g_mu_nu.subs(self.th, self.th+eps_theta) 

        # Connection Symbols
        self.gam_t = sp.Matrix([[gamma_func(0,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_r = sp.Matrix([[gamma_func(1,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_th = sp.Matrix([[gamma_func(2,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_ph = sp.Matrix([[gamma_func(3,mu, nu).simplify() for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        if verbose: print("Done connection symbols")

        # Building up the geodesic equation: 
        # Derivatives: k_beta = d x^beta / d lambda
        self.k_t, self.k_r, self.k_th, self.k_ph = sp.symbols('k_t k_r k_th k_ph', real=True)
        self.k = [self.k_t, self.k_r, self.k_th, self.k_ph]
    
        # Second derivatives: d k_beta = d^2 x^beta / d lambda^2
        self.dk_t = sum([- self.gam_t[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_r = sum([- self.gam_r[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_th = sum([- self.gam_th[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_ph = sum([- self.gam_ph[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        if verbose: print("Done diff of k")

       # Norm of k
        # the norm of k determines if you have a massive particle (-1), a mass-less photon (0) 
        # or a space-like curve (1)
        self.k = sp.Matrix([self.k_t, self.k_r, self.k_th, self.k_ph])
        self.norm_k = (self.k.T*self.g__mu__nu*self.k)[0]
        self.norm_k_lamb = sp.lambdify([self.k_t, self.k_r, self.k_th, self.k_ph, self.r, self.th, self.ph], \
                                               self.norm_k, "numpy")

        # Now we calculate k_t using the norm. This eliminates one of the differential equations.
        # time_like = True: calculates a geodesic for a massive particle (not implemented yet)
        # time_like = False: calculates a geodesic for a photon
        if (self.time_like):
            self.k_t_from_norm = sp.solve(self.norm_k+1, self.k_t)[1]
        else:
            self.k_t_from_norm = sp.solve(self.norm_k, self.k_t)[1]
        if verbose: print("Done norm of k")

        # Lambdify versions
        self.dk_t_lamb = sp.lambdify([  self.k_t, self.k_r, self.k_th, self.k_ph, \
                                        self.t, self.r, self.th, self.ph], \
                                        self.dk_t, "numpy")
        self.dk_r_lamb = sp.lambdify([  self.k_t, self.k_r, self.k_th, self.k_ph, \
                                        self.t, self.r, self.th, self.ph], \
                                        self.dk_r, "numpy")
        self.dk_th_lamb = sp.lambdify([ self.k_t, self.k_r, self.k_th, self.k_ph, \
                                        self.t, self.r, self.th, self.ph], \
                                        self.dk_th, "numpy")
        self.dk_ph_lamb = sp.lambdify([ self.k_t, self.k_r, self.k_th, self.k_ph, \
                                        self.t, self.r, self.th, self.ph], \
                                        self.dk_ph, "numpy")
        self.k_t_from_norm_lamb = sp.lambdify([ self.k_r, self.k_th, self.k_ph, \
                                                self.t, self.r, self.th, self.ph], \
                                                self.k_t_from_norm, "numpy")
        if verbose: print("Done lambdifying")

    ################################################################################################
    #
    ################################################################################################
    def get_dk(self, kt_val, kr_val, kth_val, kph_val, t_val, r_val, th_val, ph_val):
        print(round(r_val, 3), round(th_val, 3), round(ph_val,3))
        if r_val*np.sin(th_val) < 2: # If you are within a cylindre around the z-axis
            
            # Use xyz coordinates
            k4_sph = np.array([kt_val, kr_val, kth_val, kph_val])
            x4_sph = np.array([t_val, r_val, th_val, ph_val])

            k4_xyz, x4_xyz = self.conversions4D.convert_sph4_to_xyz4(k4_sph, x4_sph)

            print(f"Start value {x4_sph=} and in cart {x4_xyz=}")
            
            dk_t, dk_x, dk_y, dk_z = self.pole_metric_XYZ.get_dk(*k4_xyz, *x4_xyz)

            print("dk xyz", dk_x, dk_y, dk_z)

            dk_xyz = np.array([dk_t, dk_x, dk_y, dk_z])
            
            dk_sph, x4_sph = self.conversions4D.convert_xyz4_to_sph4(dk_xyz, x4_xyz)

            print("dk sph", dk_sph)

            print(f"And transf back {x4_sph=}")

            return dk_sph

        else:
            # Use spherical coordinates
            return \
                self.dk_t_lamb(kt_val, kr_val, kth_val, kph_val, t_val, r_val, th_val, ph_val), \
                self.dk_r_lamb(kt_val, kr_val, kth_val, kph_val, t_val, r_val, th_val, ph_val), \
                self.dk_th_lamb(kt_val, kr_val, kth_val, kph_val, t_val, r_val, th_val, ph_val), \
                self.dk_ph_lamb(kt_val, kr_val, kth_val, kph_val, t_val, r_val, th_val, ph_val), \
    
    ################################################################################################
    #
    ################################################################################################
    def get_r_s(self):
        return self.r_s

    ################################################################################################
    #
    ################################################################################################
    def oneform(self, k4_mu, x4_mu):

        if k4_mu.shape[0] == 4:
            k4_mu = np.column_stack(k4_mu)
        if x4_mu.shape[0] == 4:
            x4_mu = np.column_stack(x4_mu)

        k4__mu = np.column_stack(np.array([self.g__mu__nu_lamb(*x4_mu[i])@k4_mu[i] for i in range(len(k4_mu))]))

        return k4__mu


