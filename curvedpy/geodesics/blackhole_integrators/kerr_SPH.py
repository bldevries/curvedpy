import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
#import multiprocessing as mp
from curvedpy.utils.conversions import Conversions

class GeodesicIntegratorKerr:

    conversions = Conversions()

    def __init__(self, \
            mass=1.0, a = 0.0, time_like = False, \
            pre_sub_metric_params = False, \
            calc_kt_per_ray=True, \
            simplify_inv=True, \
            verbose=False, verbose_init = False ):

        if verbose_init: print("Integrator - started init")

        self.calc_kt_per_ray = calc_kt_per_ray

        # Connection Symbols
        def gamma_func(sigma, mu, nu):
            coord_symbols = [self.t, self.r, self.th, self.ph]
            g_sigma_mu_nu = 0
            for rho in [0,1,2,3]:
                if self.g[sigma, rho] != 0:
                    g_sigma_mu_nu += 1/2 * self.g_inv[sigma, rho] * (\
                                    self.g_diff[mu][nu, rho] + \
                                    self.g_diff[nu][rho, mu] - \
                                    self.g_diff[rho][mu, nu] )
                else:
                    g_sigma_mu_nu += 0
            return g_sigma_mu_nu

        self.M = mass
        self.r_s_value = 2*self.M 
        self.a_value = a # Remember that most of the time we integrate backwards in time and the rotation of the blackhole is reversed!

        # Type of geodesic
        self.time_like = time_like

        self.verbose = verbose 


        if verbose_init:
            print("Geodesic Kerr Integrator Settings: ")
            print(f"  - {self.M=}")
            print(f"  - {self.a_value=}")
            print(f"  - {self.r_s_value=}")
            print(f"  - {self.time_like=}")
            print(f"  - {self.verbose=} does not work from init yet")
            print(f"  - {pre_sub_metric_params=}")
            print(f"  - {self.calc_kt_per_ray=}")
            print(f"  - {simplify_inv=}")
            print("--")

        # Define symbolic variables
        self.t, self.r, self.th, self.ph, self.r_s, self.a = sp.symbols("t r \\theta \\phi r_s a", positive=True, real=True)
        self.Sig, self.Del = sp.symbols("\\Sigma \\Delta")
        self.Sig_sub = self.r**2 + self.a**2 * sp.cos(self.th)**2
        self.Del_sub = self.r**2 - self.r_s*self.r + self.a**2

        g00 = -(1-self.r_s*self.r/self.Sig)
        g11 = self.Sig/self.Del
        g22 = self.Sig
        #g33 = (self.r**2 + self.a**2 + self.r_s*self.r*self.a**2 * sp.sin(self.th)**2 / self.Sig )*sp.sin(self.th)**2 
        g33 = (sp.sin(self.th)**2/self.Sig)*((self.r**2 + self.a**2)**2 - self.a**2 * self.Del *sp.sin(self.th)**2)
        g03 = -(self.r_s * self.r * self.a * sp.sin(self.th)**2/self.Sig) # Carefull, no 2* here since it comes into the metric twice at 03 and 30

        # The metrix in terms of Sigma and Delta
        self.g_simple = sp.Matrix([\
            [g00, 0, 0, g03],\
            [0, g11, 0, 0],\
            [0, 0, g22, 0],\
            [g03, 0, 0, g33]\
            ])

        # For the Christoffel symbols we need the inverse of the metric
        # For the Kerr metric, which is not diagonal, we need to do this
        # properly
        if verbose_init: start = time.time()
        self.g_simple_inv = self.g_simple.inv()
        if simplify_inv: self.g_simple_inv.simplify()
        if verbose_init: print("Integrator - done inverse g in (sec): ", round(time.time()-start, 5))

        # Defining the four momentum per mass to use in the Geodesic Equation
        # Derivatives: k_beta = d x^beta / d lambda
        self.k_t, self.k_r, self.k_th, self.k_ph = sp.symbols('k_t k_r k_th k_ph', real=True)
        self.k = sp.Matrix([self.k_t, self.k_r, self.k_th, self.k_ph])
        # Also calculate the 1 form of the momentum per mass 4 vector
        self.k__ = self.g_simple*self.k
        self.k__ = self.k__.subs([(self.Del, self.Del_sub), (self.Sig, self.Sig_sub)])

        # Norm of k
        # the norm of k determines if you have a massive particle (-1), a mass-less photon (0) 
        # or a space-like curve (1)
        self.norm_k = (self.k.T*self.g_simple*self.k)[0]

        # Now we calculate k_t using the norm. This eliminates one of the differential equations.
        # time_like = True: calculates a geodesic for a massive particle
        # time_like = False: calculates a geodesic for a photon
        if verbose_init: start = time.time()
        if not self.calc_kt_per_ray:
            if (self.time_like):
                self.k_t_from_norm = sp.solve(self.norm_k+1, self.k_t)[1]
            else:
                self.k_t_from_norm = sp.solve(self.norm_k, self.k_t)[1]
        if verbose_init: print("Integrator - Done norm of k in:", round(time.time()-start, 5))
        

        # Getting the metric in full and only dependend on the coordinates
        self.g = self.g_simple.subs([(self.Del, self.Del_sub), (self.Sig, self.Sig_sub)])
        #self.g = self.g.subs(self.Sig, self.Sig_sub)
        self.g_inv = self.g_simple_inv.subs([(self.Del, self.Del_sub), (self.Sig, self.Sig_sub)])
        self.norm_k = self.norm_k.subs([(self.Del, self.Del_sub), (self.Sig, self.Sig_sub)])
        
        if pre_sub_metric_params:
            # Subbing these in can make things a lot faster in some cases.
            self.g = self.g.subs([(self.a, self.a_value), (self.r_s, self.r_s_value)])
            self.g_inv = self.g_inv.subs([(self.a, self.a_value), (self.r_s, self.r_s_value)])

        # We already differentiate the metric to all the coordinates, otherwise we
        # will do this multiple time for the same elements when we calculating the 
        # Christoffel symbols
        if verbose_init: start = time.time()
        self.g_diff = [self.g.diff(self.t), self.g.diff(self.r), self.g.diff(self.th), self.g.diff(self.ph)]
        if verbose_init: print("Integrator - done with g_diff in: ", round(time.time()-start, 5))


        # Time to start calculating the Christoffel Symbols
        if verbose_init: start = time.time()
        self.gam_t = sp.Matrix([[gamma_func(0,mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_r = sp.Matrix([[gamma_func(1,mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_th = sp.Matrix([[gamma_func(2,mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        self.gam_ph = sp.Matrix([[gamma_func(3,mu, nu) for mu in [0,1,2,3]] for nu in [0,1,2,3]])
        if verbose_init: print("Integrator - done connection symbols in: ", round(time.time()-start, 5))

        #self.k = [self.k_t, self.k_r, self.k_th, self.k_ph]
        

        # Calculating the directional derivative of the four momentum per mass, also to use in the Geodesic equation
        # Second derivatives: d k_beta = d^2 x^beta / d lambda^2
        if verbose_init: start = time.time()
        self.dk_t = sum([- self.gam_t[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_r = sum([- self.gam_r[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_th = sum([- self.gam_th[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        self.dk_ph = sum([- self.gam_ph[nu, mu]*self.k[mu]*self.k[nu] for mu in [0,1,2,3] for nu in [0,1,2,3]])
        if verbose_init: print("Integrator - Done diff of k in: ", round(time.time()-start, 5))


        # Lambdify versions of the directional derivatives of the four momentum per mass
        if verbose_init: start = time.time()
        self.dk_t_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                      self.k_t, self.t, self.r_s, self.a], \
                                     self.dk_t, "numpy")
        self.dk_r_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                      self.k_t, self.t, self.r_s, self.a], \
                                     self.dk_r, "numpy")
        self.dk_th_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                      self.k_t, self.t, self.r_s, self.a], \
                                     self.dk_th, "numpy")
        self.dk_ph_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                      self.k_t, self.t, self.r_s, self.a], \
                                     self.dk_ph, "numpy")

        self.norm_k_lamb = sp.lambdify([self.k_t, self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                               self.r_s, self.a], self.norm_k, "numpy")

        if not self.calc_kt_per_ray:
            self.k_t_from_norm = self.k_t_from_norm.subs([(self.Del, self.Del_sub), (self.Sig, self.Sig_sub)])
            self.k_t_from_norm_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                                   self.r_s, self.a], self.k_t_from_norm, "numpy")

        self.lamb_k__ = sp.lambdify([self.k_t, self.k_r, self.k_th, self.k_ph, self.t, self.r, self.th, self.ph, self.r_s, self.a],\
                                 self.k__, "numpy")

        if verbose_init: 
            print("Inegrator - Done lambdifying in:", round(time.time()-start, 5))
            print("Integrator - init done")
            print()




    ################################################################################################
    #
    ################################################################################################
    def get_k_t_from_norm(self, k_r_0, r0, k_th_0, th0, k_ph_0, ph0):
        sub_list = [(self.t, 0), (self.r, r0), (self.th, th0), (self.ph, ph0), \
                    (self.k_r, k_r_0), (self.k_th, k_th_0), (self.k_ph, k_ph_0), \
                    (self.r_s, self.r_s_value), (self.a, self.a_value)]

        if (self.time_like):
            k_t_from_norm = sp.solve(self.norm_k.subs(sub_list)+1, self.k_t)[1]#[1] #sp.solve(self.norm_k+1, self.k_t)[1]
        else:
            k_t_from_norm = sp.solve(self.norm_k.subs(sub_list), self.k_t)[1]#[1] #sp.solve(self.norm_k, self.k_t)[1]

        
        return k_t_from_norm

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
    #
    ################################################################################################
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

        if verbose: print("Calculate Traj Started")
        if verbose: print(f"{max_step=}")
        if not isinstance(x0_xyz,np.ndarray):
            x0_xyz = np.array(x0_xyz)

        if not isinstance(k0_xyz,np.ndarray):
            k0_xyz = np.array(k0_xyz)

        x0_bl, k0_bl = self.conversions.convert_xyz_to_bl(x0_xyz, k0_xyz, a = self.a_value)
        k_r_0, k_th_0, k_ph_0 = k0_bl
        r0, th0, ph0 = x0_bl

        result = self.calc_trajectory_bl(\
                        k_r_0 = k_r_0, r0 = r0, k_th_0=k_th_0, th0=th0, k_ph_0=k_ph_0, ph0=ph0,\
                        R_end = R_end, curve_start = curve_start, curve_end = curve_end, nr_points_curve = nr_points_curve, \
                        method = method, max_step = max_step, first_step = first_step, rtol = rtol, atol = atol,\
                        verbose = verbose )
                       

        k_r, r, k_th, th, k_ph, ph, k_t = result.y
        t = result.t

        k_bl = np.array([k_r, k_th, k_ph])
        x_bl = np.array([r, th, ph])

        if verbose: start = time.time()
        # SHOULD I NOT CHANGE COORDS USING 4 VECTORS????
        x_xyz, k_xyz = self.conversions.convert_bl_to_xyz(x_bl, k_bl, self.a_value)
        if verbose: print("  Converting result to xyz in: ", round(time.time()-start, 5))

        x4_xyz = np.array([t, *x_xyz])
        k4_xyz = np.array([k_t, *k_xyz])

        result.update({"k4_xyz": k4_xyz, "x4_xyz": x4_xyz})

        return k_xyz, x_xyz, result

    ################################################################################################
    # calc_trajectory_bl
    ################################################################################################
    # This function does the numerical integration of the geodesic equation using scipy's solve_ivp
    def calc_trajectory_bl(self, \
                        k_r_0 = 0., r0 = 10.0, k_th_0 = 0.0, th0 = 1/2*np.pi, k_ph_0 = 0.1, ph0 = 0.0,\
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
        if verbose: start = time.time()
        if self.calc_kt_per_ray:
            k_t_0 = self.get_k_t_from_norm(k_r_0, r0, k_th_0, th0, k_ph_0, ph0)
        else:
            k_t_0 = self.k_t_from_norm_lamb(k_r_0, r0, k_th_0, th0, k_ph_0, ph0, self.r_s_value, self.a_value)
        if verbose: print("  Calculated k_t in: ", round(time.time()-start, 5))

        if R_end == -1:
            R_end = np.inf
        elif R_end < r0:
            R_end = r0*1.01

        if r0 > self.r_s_value:
            # Step function needed for solve_ivp
            def step(lamb, new):

                # if verbose:
                #     k_r, r, k_th, th, k_ph, ph, k_t = new
                #     x, y, z = self.conversions.coord_conversion_bl_xyz(r, th, ph, self.a_value)
                #     print(round(x, 4), round(y,4), round(z,4))

                new_k_r, new_r, new_k_th, new_th, new_k_ph, new_ph, new_k_t = new

                new_dk_t = self.dk_t_lamb(*new, t = lamb, r_s = self.r_s_value, a = self.a_value)
                dr = new_k_r
                new_dk_r = self.dk_r_lamb(*new, t = lamb, r_s = self.r_s_value, a = self.a_value)
                dr = new_k_r
                new_dk_th = self.dk_th_lamb(*new, t = lamb, r_s = self.r_s_value, a = self.a_value)
                dth = new_k_th
                new_dk_ph = self.dk_ph_lamb(*new, t = lamb, r_s = self.r_s_value, a = self.a_value)
                dph = new_k_ph

                return( new_dk_r, dr, new_dk_th, dth, new_dk_ph, dph, new_dk_t)

            def hit_blackhole(t, y): 
                eps = 0.01
                k_r, r, k_th, th, k_ph, ph, k_t = y
                #if verbose: print("Event - hit_blackhole: ", r-self.r_s_value)
                return r - (self.r_s_value+eps)
            hit_blackhole.terminal = True

            def reached_end(t, y): 
                k_r, r, k_th, th, k_ph, ph, k_t = y
                return r - R_end
            reached_end.terminal = True
            
            values_0 = [ k_r_0, r0, k_th_0, th0, k_ph_0, ph0, k_t_0 ]
            if nr_points_curve == 0:
                t_pts = None
            else:
                t_pts = np.linspace(curve_start, curve_end, nr_points_curve)

            events = [hit_blackhole]
            events.append(reached_end)

            start = time.time()
            result = solve_ivp(step, (curve_start, curve_end), values_0, t_eval=t_pts, \
                               events=events,\
                               method=method,\
                               max_step = max_step,\
                               first_step = first_step,\
                               atol=atol,\
                               rtol = rtol)
            if verbose: 
                print("  Finished solver: ", result.message)
                print("   Blackhole hit? ", len(result.t_events[0])>0)
                print("   Time: ",round(time.time()-start, 5), "sec")

            result.update({"R_end": R_end})
            result.update({"hit_blackhole": len(result.t_events[0])>0})
            result.update({"start_inside_hole": False})
            result.update({"end_check": len(result.t_events[1])>0})


        else:
            if verbose: print("Starting location inside the blackhole.")
            result = {"start_inside_hole": True}

        return result


    ################################################################################################
    # Helper functions
    ################################################################################################

    def one_form(self, vec4, x4):
        return self.one_form_lamb(*vec4, *x4)
        # #tv, xv, yv, zv = x
        # #rv, thv, phv = self.conversions.coord_conversion_xyz_bl(xv, yv, zv, self.a_value)
        # tv, rv, thv, phv = x_bl
        # sub_list = [(self.t, tv), (self.r, rv), (self.th, thv), (self.ph, phv), (self.r_s, self.r_s_value), (self.a, self.a_value)]
        # #print(sub_list)
        # return self.g.subs(sub_list)

    # Conserved angular momentum per mass
    def ang_mom(self, r, k_ph):
        return r**2*k_ph

    def energy_photon(self, k_r, r, k_ph, M_blackhole):
        return np.sqrt(k_r**2 +(1-2*M_blackhole/r)*self.ang_mom(r, k_ph)**2/r**2)

    def energy_massive(self, k_r, r, k_ph, M_blackhole):
        return np.sqrt(k_r**2 +(1-2*M_blackhole/r)*(1+self.ang_mom(r, k_ph)**2/r**2))
