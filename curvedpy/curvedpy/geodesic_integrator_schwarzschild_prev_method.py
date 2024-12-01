import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
import time
import multiprocessing as mp
from curvedpy import Conversions

class GeodesicIntegratorSchwarzschild_prev_method:

    conversions = Conversions()

    def __init__(self, mass=1.0, time_like = False, verbose=False):

        # Connection Symbols
        def gamma_func(sigma, mu, nu):
            coord_symbols = [self.t, self.r, self.th, self.ph]
            g_sigma_mu_nu = 0
            for rho in [0,1,2,3]:
                if self.g[sigma, rho] != 0:
                    g_sigma_mu_nu += 1/2 * 1/self.g[sigma, rho] * (\
                                    self.g[nu, rho].diff(coord_symbols[mu]) + \
                                    self.g[rho, mu].diff(coord_symbols[nu]) - \
                                    self.g[mu, nu].diff(coord_symbols[rho]) )
                else:
                    g_sigma_mu_nu += 0
            return g_sigma_mu_nu

        self.M = mass
        self.r_s_value = 2*self.M 

        # Type of geodesic
        self.time_like = time_like # No Massive particle geodesics yet

        # Define symbolic variables
        self.t, self.r, self.th, self.ph, self.r_s = sp.symbols("t r \\theta \\phi r_s")

        self.g = sp.Matrix([\
            [-1*(1-self.r_s/self.r), 0, 0, 0],\
            [0, 1/(1-self.r_s/self.r), 0, 0],\
            [0, 0, self.r**2, 0],\
            [0, 0, 0, self.r**2 * sp.sin(self.th)**2]\
            ])

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
        self.norm_k = (self.k.T*self.g*self.k)[0]
        self.norm_k_lamb = sp.lambdify([self.k_t, self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                               self.r_s], self.norm_k, "numpy")

        # Now we calculate k_t using the norm. This eliminates one of the differential equations.
        # time_like = True: calculates a geodesic for a massive particle (not implemented yet)
        # time_like = False: calculates a geodesic for a photon
        if (self.time_like):
            self.k_t_from_norm = sp.solve(self.norm_k+1, self.k_t)[1]
        else:
            self.k_t_from_norm = sp.solve(self.norm_k, self.k_t)[1]
        if verbose: print("Done norm of k")

        # Lambdify versions
        self.dk_r_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                      self.k_t, self.t, self.r_s], \
                                     self.dk_r, "numpy")
        self.dk_th_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                      self.k_t, self.t, self.r_s], \
                                     self.dk_th, "numpy")
        self.dk_ph_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                      self.k_t, self.t, self.r_s], \
                                     self.dk_ph, "numpy")
        self.k_t_from_norm_lamb = sp.lambdify([self.k_r, self.r, self.k_th, self.th, self.k_ph, self.ph, \
                                               self.r_s], self.k_t_from_norm, "numpy")
        if verbose: print("Done lambdifying")


    ################################################################################################
    #
    ################################################################################################
    def calc_trajectory(self, k0_xyz, x0_xyz, *args, **kargs):
        mp_on = False

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

        if mp_on == True:
            print("Doing mp", __name__)

            #if __name__ == '__main__':
            print("Got into main")
            pool = mp.Pool(mp.cpu_count())
            print([(k0_xyz_i,x0_xyz_i) for k0_xyz_i, x0_xyz_i in zip(k0_xyz, x0_xyz)])
            results = [pool.apply(self.calc_trajectory_xyz, args=(k0_xyz_i,x0_xyz_i)) for k0_xyz_i, x0_xyz_i in zip(k0_xyz, x0_xyz)]
            pool.close()    

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

        print(curve_start, curve_end)

        if not isinstance(x0_xyz,np.ndarray):
            x0_xyz = np.array(x0_xyz)

        if not isinstance(k0_xyz,np.ndarray):
            k0_xyz = np.array(k0_xyz)

        x0_sph, k0_sph = self.conversions.convert_xyz_to_sph(x0_xyz, k0_xyz)
        k_r_0, k_th_0, k_ph_0 = k0_sph
        r0, th0, ph0 = x0_sph

        result = self.calc_trajectory_sph(\
                        k_r_0 = k_r_0, r0 = r0, k_th_0=k_th_0, th0=th0, k_ph_0=k_ph_0, ph0=ph0,\
                        R_end = R_end, curve_start = curve_start, curve_end = curve_end, nr_points_curve = nr_points_curve, \
                        method = method, max_step = max_step, first_step = first_step, rtol = rtol, atol = atol,\
                        verbose = verbose )
                       

        k_r, r, k_th, th, k_ph, ph = result.y

        k_sph = np.array([k_r, k_th, k_ph])
        x_sph = np.array([r, th, ph])
        x_xyz, k_xyz = self.conversions.convert_sph_to_xyz(x_sph, k_sph)

        return k_xyz, x_xyz, result

    ################################################################################################
    # calc_trajectory_sph
    ################################################################################################
    # This function does the numerical integration of the geodesic equation using scipy's solve_ivp
    def calc_trajectory_sph(self, \
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

        if R_end == -1:
            R_end = np.inf
        elif R_end < r0:
            R_end = r0*1.01

        if r0 > self.r_s_value:
            # Step function needed for solve_ivp
            def step(lamb, new):
                new_k_r, new_r, new_k_th, new_th, new_k_ph, new_ph = new
                new_k_t = self.k_t_from_norm_lamb(*new, self.r_s_value)

                new_dk_r = self.dk_r_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dr = new_k_r
                new_dk_th = self.dk_th_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dth = new_k_th
                new_dk_ph = self.dk_ph_lamb(*new, new_k_t, t = 0, r_s = self.r_s_value)
                dph = new_k_ph

                return( new_dk_r, dr, new_dk_th, dth, new_dk_ph, dph)

            def hit_blackhole(t, y): 
                eps = 0.5
                k_r, r, k_th, th, k_ph, ph = y
                #if verbose: print("Test Event Hit BH: ", x, y, z, self.r_s_value, x**2 + y**2 + z**2 - self.r_s_value**2)
                return r - self.r_s_value
            hit_blackhole.terminal = True

            def reached_end(t, y): 
                #k_x, x, k_y, y, k_z, z = y
                k_r, r, k_th, th, k_ph, ph = y
                #print("integrator check end", r, r0)
                #if verbose: print("Test Event End: ", np.sqrt(x**2 + y**2 + z**2), R_end, x**2 + y**2 + z**2 - R_end**2)
                return r - R_end
            reached_end.terminal = True
            
            values_0 = [ k_r_0, r0, k_th_0, th0, k_ph_0, ph0 ]
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


        else:
            if verbose: print("Starting location inside the blackhole.")
            result = {"start_inside_hole": True}

        return result
