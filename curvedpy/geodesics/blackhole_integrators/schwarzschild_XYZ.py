import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import time
from curvedpy.utils.conversions import Conversions

# -------------------------------
# ----NAMING CONVENTIONS USED----
# -------------------------------
#
# UPPER & LOWER INDICES
# _ indicates an upper/contravariant index
# __ indicates a lower/covariant index
# For example g__mu__nu is the metric g with two lower indices while g_mu_nu is the metric 
# with two upper indices, which is the inverse of g__mu__nu
# Another example is 4-momentum k_mu, which has a 0 (time) component of k_0 or k_t
# The 4-momentum as a oneform is then k__mu and its zeroth component is k__0
# An example tensor could be T_mu__nu__alp. This tensor has one upper/contravariant (mu) and 
# two lower/covariant/oneform (nu, alp) indices.
#
# COORDINATES
# the used coordinate system for a tensor is indicated with by appending for example 
# _xyz, _sph, _bl after the indices.
# Example: x_mu_bl: this 4vector with one upper index is given in Boyer-Lindquist coordinates
# Example: g__mu__nu_sph: this metric tensor with two lower indices is given in spherical coordinates
#
# MISC
# If a vector x_mu has only 3 components, they are the three spatial components


# https://f.yukterez.net/einstein.equations/files/8.html#transformation
# https://physics.stackexchange.com/questions/672252/how-to-compute-and-interpret-schwartzchild-black-hole-metric-line-element-ds-i
class GeodesicIntegratorSchwarzschildXYZ:

    conversions = Conversions()

    ################################################################################################
    #
    ################################################################################################
    def __init__(self, mass=1.0, time_like = False, verbose=False):

        self.M = mass
        self.r_s_value = 2*self.M 

        # Type of geodesic
        self.time_like = time_like # No Massive particle geodesics yet
        self.verbose = verbose
        if verbose:
            print("Geodesic SS Integrator Settings: ")
            print(f"  - {self.M=}")
            print(f"  - {self.r_s_value=}")
            print(f"  - {self.time_like=}")
            print(f"  - {self.verbose=}")
            print("--")

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
    #
    ################################################################################################
    def calc_trajectory(self, k0_xyz, x0_xyz, *args, **kargs):

        if not isinstance(k0_xyz, np.ndarray): k0_xyz = np.array(k0_xyz)
        if not isinstance(x0_xyz, np.ndarray): x0_xyz = np.array(x0_xyz)


        if k0_xyz.shape != x0_xyz.shape:
            print("k and x are not the same shape")
            return

        if k0_xyz.ndim == 1:
            if k0_xyz.shape[0] != 3 or x0_xyz.shape[0] != 3:
                print("k or x do not have 3 components")
                return
            k0_xyz = k0_xyz.reshape(1,3)
            x0_xyz = x0_xyz.reshape(1,3)

        else:
            if k0_xyz.shape[1] != 3 or x0_xyz.shape[1] != 3:
                print("k or x do not have 3 components")
                return

        if len(k0_xyz) == 1:
            return self.calc_trajectory_xyz(k0_xyz[0], x0_xyz[0], *args, **kargs)
        else:
            return [self.calc_trajectory_xyz(k0_xyz[i], x0_xyz[i], *args, **kargs) for i in range(len(x0_xyz))]

    ################################################################################################
    # calc_trajectory
    ################################################################################################
    # This function does the numerical integration of the geodesic equation using scipy's solve_ivp
    def calc_trajectory_xyz(self, \
                        k0_xyz = np.array([1, 0.0, 0.0]), x0_xyz = np.array([-10, 10, 0]), \
                        R_end = -1,\
                        curve_start = 0, \
                        curve_end = 50, \
                        nr_points_curve = 50, \
                        method = "RK45",\
                        max_step = np.inf,\
                        first_step = None,\
                        rtol = 1e-3,\
                        atol = 1e-6,\
                        verbose = False \
                       ):

        # Calculate from norm of starting condition
        k0_t = self.k_t_from_norm(k0=k0_xyz, x0= x0_xyz, t = 0)
        values_0 = [k0_t, *k0_xyz, *x0_xyz]


        r0 = np.linalg.norm(x0_xyz)
        if R_end == -1:
            R_end = np.inf
        elif R_end < r0:
            R_end = np.sqrt(x0**2 + y0**2 + z0**2)*1.01

        if r0 > self.r_s_value:
            # Step function needed for solve_ivp
            def step(lamb, new):
                k_t_new, k_x_new, k_y_new, k_z_new, x_new, y_new, z_new = new

                dk_t, dk_x, dk_y, dk_z = self.get_dk(k_t_new, k_x_new, k_y_new, k_z_new, \
                                                    t_val=lamb, x_val=x_new, y_val=y_new, z_val=z_new)

                dx, dy, dz = k_x_new, k_y_new, k_z_new
                return( dk_t, dk_x, dk_y, dk_z, dx, dy, dz)

            def hit_blackhole(t, y): 
                eps = 0.01
                k_t, k_x, k_y, k_z, x, y, z = y
                return np.sqrt(x**2+y**2+z**2) - (self.r_s_value+eps)
            hit_blackhole.terminal = True

            def reached_end(t, y): 
                k_t, k_x, k_y, k_z, x, y, z = y
                return np.sqrt(x**2+y**2+z**2) - R_end
            reached_end.terminal = True
            
            if nr_points_curve == 0:
                t_pts = None
            else:
                t_pts = np.linspace(curve_start, curve_end, nr_points_curve)

            start = time.time()
            events = [hit_blackhole]
            events.append(reached_end)

            result = solve_ivp(step, (curve_start, curve_end), values_0, t_eval=t_pts, \
                               events=events,\
                               method=method,\
                               max_step = max_step,\
                               first_step = first_step,\
                               atol=atol,\
                               rtol = rtol)
            end = time.time()
            if verbose: print("New: ", result.message, end-start, "sec")

            result.update({"R_end": R_end})
            result.update({"hit_blackhole": len(result.t_events[0])>0})
            result.update({"start_inside_hole": False})
            result.update({"end_check": len(result.t_events[1])>0})

            k_t, k_x, k_y, k_z, x, y, z = result.y
            t = result.t

            k3 = np.array([k_x, k_y, k_z])
            k4 = np.array([k_t, k_x, k_y, k_z])
            x3 = np.array([x, y, z])
            x4 = np.array([t, x, y, z])
            
            result.update({"k4_xyz": k4, "x4_xyz": x4})
            result.update({"k3_xyz": k3, "x3_xyz": x3})

            return k3, x3, result

        else:
            if verbose: print("Starting location inside the blackhole.")
            result = {"start_inside_hole": True}

            return result

