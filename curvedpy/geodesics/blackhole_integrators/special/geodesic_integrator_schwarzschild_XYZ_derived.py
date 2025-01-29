import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
#import multiprocessing as mp
from curvedpy import Conversions

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
        self.t, self.r, self.th, self.ph, self.r_s = sp.symbols("t r \\theta \\phi r_s", real=True)
        self.x, self.y, self.z = sp.symbols('x, y, z', real=True)

        self.g__mu__nu_sph = sp.Matrix([\
            [-1*(1-self.r_s/self.r), 0, 0, 0],\
            [0, 1/(1-self.r_s/self.r), 0, 0],\
            [0, 0, self.r**2, 0],\
            [0, 0, 0, self.r**2 * sp.sin(self.th)**2]\
            ])

        self.g__mu__nu_sph_lamb = sp.lambdify([self.t, self.r, self.th, self.ph, self.r_s], self.g__mu__nu_sph, "numpy")
        
        # self.to_cart_sub_list = [(self.th, sp.acos(self.z/sp.sqrt(self.x**2+self.y**2+self.z**2))), \
        #                          (self.r,  sp.sqrt(self.x**2+self.y**2+self.z**2)),\
        #                          (self.ph, sp.atan2(self.y, self.x))]
        self.to_cart_sub_list = [(self.th, sp.acos(self.z/(self.x**2+self.y**2+self.z**2)**0.5)), \
                                 (self.r,  (self.x**2+self.y**2+self.z**2)**0.5),\
                                 (self.ph, sp.atan2(self.y, self.x))]
        print("g")
        self.g__mu__nu_cart = self.g__mu__nu_sph.subs(self.to_cart_sub_list)
        self.g__mu__nu_cart.simplify()
        print("g_inv")
        self.g_mu_nu_cart = self.g__mu__nu_cart.inv()
        #self.g_inv.subs(self.to_cart_sub_list)
        self.g_mu_nu_cart.simplify()

        #YOU ALSO NEED TO TRANSFORM IT!
        print("transforming")
        self.g__mu__nu_cart = self.trans_to_xyz(self.g__mu__nu_cart)
        self.g_mu_nu_cart = self.trans_inv_to_xyz(self.g_mu_nu_cart)

        # I WOULD FIRST DIFF
        print("g_diff")
        self.g__mu__nu_cart_diff = [self.g__mu__nu_cart.diff(self.t), self.g__mu__nu_cart.diff(self.x), \
                                    self.g__mu__nu_cart.diff(self.y), self.g__mu__nu_cart.diff(self.z)]

        # We lambdify these to get numpy arrays
        print("g_lamb")
        self.g__mu__nu_cart_lamb = sp.lambdify([self.t, self.x, self.y, self.z, self.r_s], self.g__mu__nu_cart)
        self.g_mu_nu_cart_lamb = sp.lambdify([self.t, self.x, self.y, self.z, self.r_s], self.g_mu_nu_cart)
        self.g__mu__nu_cart_diff_lamb = sp.lambdify([self.t, self.x, self.y, self.z, self.r_s], self.g__mu__nu_cart_diff)

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

    def trans_to_xyz(self, g__mu__nu_cart):
        r = sp.sqrt(self.x**2+self.y**2+self.z**2)
        th = sp.acos(self.z/r)
        phi = sp.atan2(self.y,self.x)
        M = sp.Matrix([[1, 0, 0, 0],\
                        [0, r.diff(self.x), r.diff(self.y), r.diff(self.z)],\
                        [0, th.diff(self.x), th.diff(self.y), th.diff(self.z)],\
                        [0, phi.diff(self.x), phi.diff(self.y), phi.diff(self.z)],\
                     ])

        return M.T*g__mu__nu_cart*M # !check


    def trans_inv_to_xyz(self, g_mu_nu):
        x = self.r * sp.sin(self.th) * sp.cos(self.ph)
        y = self.r * sp.sin(self.th) * sp.sin(self.ph)
        z = self.r * sp.cos(self.th)
        M = sp.Matrix([[1, 0, 0, 0],\
                      [0, x.diff(self.r), x.diff(self.th), x.diff(self.ph)],\
                      [0, y.diff(self.r), y.diff(self.th), y.diff(self.ph)],\
                      [0, z.diff(self.r), z.diff(self.th), z.diff(self.ph)],\
                     ]).subs(self.to_cart_sub_list)

        return M*g_mu_nu*M.T 


    ################################################################################################
    #
    ################################################################################################
    def get_dk(self, kt_val, kx_val, ky_val, kz_val, t_val, x_val, y_val, z_val):
        # Calc g, g_inv and g_diff at given coords
        g = self.g__mu__nu_cart_lamb(t_val, x_val, y_val, z_val, self.r_s_value)
        g_inv = self.g_mu_nu_cart_lamb(t_val, x_val, y_val, z_val, self.r_s_value)
        g_diff = self.g__mu__nu_cart_diff_lamb(t_val, x_val, y_val, z_val, self.r_s_value)

        # Calc the connection Symbols at given coords
        gam_t = sp.Matrix([[self.gamma_func(g, g_inv, g_diff, 0, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_x = sp.Matrix([[self.gamma_func(g, g_inv, g_diff, 1, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_y = sp.Matrix([[self.gamma_func(g, g_inv, g_diff, 2, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        gam_z = sp.Matrix([[self.gamma_func(g, g_inv, g_diff, 3, mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])

        # Building up the geodesic equation: 
        # Derivatives: k_beta = d x^beta / d lambda
        k = [kt_val, kx_val, ky_val, kz_val]
    
        # Second derivatives: d k_beta = d^2 x^beta / d lambda^2
        dk_t = sum([- gam_t[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        dk_x = sum([- gam_x[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        dk_y = sum([- gam_y[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        dk_z = sum([- gam_z[nu, mu]*k[mu]*k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])

        return dk_t, dk_x, dk_y, dk_z

    # Connection Symbols
    def gamma_func(self, g, g_inv, g_diff, sigma, mu, nu):

        g_sigma_mu_nu = 0
        for rho in [0,1,2,3]:
            if g[sigma, rho] != 0:
                g_sigma_mu_nu += 1/2 * g_inv[sigma, rho] * (\
                                g_diff[mu][nu, rho] + \
                                g_diff[nu][rho, mu] - \
                                g_diff[rho][mu, nu] )
            else:
                g_sigma_mu_nu += 0
        return g_sigma_mu_nu

    def k_t_from_norm(self, k0, x0, t=0):
        sub_list = [(self.k_x, k0[0]),\
                    (self.k_y, k0[1]),\
                    (self.k_z, k0[2]),\
                    (self.t, t),\
                    (self.x, x0[0]),\
                    (self.y, x0[1]),\
                    (self.z, x0[2]),\
                    (self.r_s, self.r_s_value),\
                    ]
        norm_k_subbed = self.norm_k.subs(sub_list)

        # Now we calculate k_t using the norm. This eliminates one of the differential equations.
        # time_like = True: calculates a geodesic for a massive particle
        # time_like = False: calculates a geodesic for a photon
        if (self.time_like):
            k_t_from_norm = sp.solve(norm_k_subbed+1, self.k_t)#[1]
        else:
            k_t_from_norm = sp.solve(norm_k_subbed, self.k_t)#[1]
        if len(k_t_from_norm) > 1:
            k_t_from_norm = k_t_from_norm[1]
        return float(k_t_from_norm)
        # self.k_t_from_norm_lamb = sp.lambdify([self.k_x, self.k_y, self.k_z, self.x, self.y, self.z, self.t, self.r_s], \
        #                                         self.k_t_from_norm, "numpy")

    def calc_oneform(self, v_mu, g__mu__nu):
        v_nu = g__mu__nu * v_mu
        return v_nu


    ################################################################################################
    #
    ################################################################################################
    def calc_trajectory(self, k0_xyz, x0_xyz, *args, **kargs):
        #mp_on = False

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
        k0_t = self.k_t_from_norm(k0=k0_xyz, x0= x0_xyz, t = 0)#, r_s = self.r_s_value)
        #k0_t = self.k_t_from_norm_lamb(*k0_xyz, *x0_xyz, t = 0, r_s = self.r_s_value)
        values_0 = [k0_t, *k0_xyz, *x0_xyz]#[ k0_x, x0, k0_y, y0, k0_z, z0, k0_t ]


        r0 = np.linalg.norm(x0_xyz)
        if R_end == -1:
            R_end = np.inf
        elif R_end < r0:
            R_end = np.sqrt(x0**2 + y0**2 + z0**2)*1.01

        if r0 > self.r_s_value:
            # Step function needed for solve_ivp
            def step(lamb, new):
                k_t_new, k_x_new, k_y_new, k_z_new, x_new, y_new, z_new = new

                #print(new)
                
                dk_t, dk_x, dk_y, dk_z = self.get_dk(k_t_new, k_x_new, k_y_new, k_z_new, \
                                                    t_val=lamb, x_val=x_new, y_val=y_new, z_val=z_new)

                dx, dy, dz = k_x_new, k_y_new, k_z_new

                #new_k_x, new_x, new_k_y, new_y, new_k_z, new_z, new_k_t = new

                # new_dk_t = self.dk_t_lamb(*new, t = lamb, r_s = self.r_s_value)
                # #dr = new_k_r
                # new_dk_x = self.dk_x_lamb(*new, t = lamb, r_s = self.r_s_value)
                # dx = new_k_x
                # new_dk_y = self.dk_y_lamb(*new, t = lamb, r_s = self.r_s_value)
                # dy = new_k_y
                # new_dk_z = self.dk_z_lamb(*new, t = lamb, r_s = self.r_s_value)
                # dz = new_k_z

                # return( new_dk_x, dx, new_dk_y, dy, new_dk_z, dz, new_dk_t)
                return( dk_t, dk_x, dk_y, dk_z, dx, dy, dz)

            def hit_blackhole(t, y): 
                eps = 0.01
                #k_x, x, k_y, y, k_z, z, k_t = y
                k_t, k_x, k_y, k_z, x, y, z = y
                #if verbose: print("Event - hit_blackhole: ", r-self.r_s_value)
                return np.sqrt(x**2+y**2+z**2) - (self.r_s_value+eps)
            hit_blackhole.terminal = True

            def reached_end(t, y): 
                #k_x, x, k_y, y, k_z, z, k_t = y
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

